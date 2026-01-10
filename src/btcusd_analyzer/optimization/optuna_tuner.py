"""
Optuna Tuner - Hyperparameter-Optimierung mit Optuna
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
from torch.utils.data import DataLoader

from ..core.logger import get_logger
from ..models.base import BaseModel, ModelFactory
from ..trainer.trainer import Trainer
from ..trainer.callbacks import EarlyStopping


@dataclass
class SearchSpace:
    """
    Definiert den Suchraum fuer Hyperparameter.

    Attributes:
        name: Name des Parameters
        param_type: 'int', 'float', 'categorical', 'loguniform'
        low: Untere Grenze (fuer int/float)
        high: Obere Grenze (fuer int/float)
        choices: Moegliche Werte (fuer categorical)
        log: Logarithmische Skala (fuer float)
    """
    name: str
    param_type: str
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    log: bool = False


@dataclass
class TunerConfig:
    """Konfiguration fuer den Optuna Tuner."""
    # Allgemein
    n_trials: int = 50
    timeout: Optional[int] = None  # Sekunden
    n_jobs: int = 1  # Parallele Trials

    # Pruning
    use_pruning: bool = True
    pruning_warmup_steps: int = 5
    pruning_interval_steps: int = 1

    # Training pro Trial
    epochs_per_trial: int = 30
    early_stopping_patience: int = 5

    # Optimierungsziel
    direction: str = 'maximize'  # 'maximize' fuer Accuracy, 'minimize' fuer Loss
    metric: str = 'val_accuracy'


class OptunaTuner:
    """
    Hyperparameter-Optimierung mit Optuna.

    Features:
    - Automatische Suche nach optimalen Hyperparametern
    - Pruning von schlechten Trials
    - Unterstuetzung fuer alle Modell-Architekturen
    - Speichern/Laden von Studien
    - Visualisierung der Ergebnisse

    Usage:
        tuner = OptunaTuner(train_loader, val_loader)
        tuner.add_search_space('hidden_size', 'int', low=32, high=256)
        tuner.add_search_space('learning_rate', 'float', low=1e-5, high=1e-2, log=True)
        best_params = tuner.optimize(n_trials=50)
    """

    # Standard-Suchraum
    DEFAULT_SEARCH_SPACE = [
        SearchSpace('hidden_size', 'int', low=32, high=256),
        SearchSpace('num_layers', 'int', low=1, high=4),
        SearchSpace('dropout', 'float', low=0.0, high=0.5),
        SearchSpace('learning_rate', 'float', low=1e-5, high=1e-2, log=True),
        SearchSpace('batch_size', 'categorical', choices=[16, 32, 64, 128]),
    ]

    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_type: str = 'bilstm',
        input_size: int = 6,
        num_classes: int = 3,
        config: Optional[TunerConfig] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialisiert den Optuna Tuner.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            model_type: Modell-Architektur ('lstm', 'bilstm', 'transformer', etc.)
            input_size: Anzahl Input-Features
            num_classes: Anzahl Klassen
            config: Tuner-Konfiguration
            device: PyTorch Device
        """
        self.logger = get_logger()

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_type = model_type.lower()
        self.input_size = input_size
        self.num_classes = num_classes
        self.config = config or TunerConfig()

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Suchraum
        self.search_space: List[SearchSpace] = []

        # Optuna Study
        self.study: Optional[optuna.Study] = None

        # Ergebnisse
        self.best_params: Dict[str, Any] = {}
        self.best_value: float = 0.0
        self.trials_history: List[Dict] = []

        self.logger.info(f'OptunaTuner initialisiert: {model_type.upper()} auf {self.device}')

    def add_search_space(
        self,
        name: str,
        param_type: str,
        low: Optional[float] = None,
        high: Optional[float] = None,
        choices: Optional[List[Any]] = None,
        log: bool = False
    ):
        """
        Fuegt einen Parameter zum Suchraum hinzu.

        Args:
            name: Parameter-Name
            param_type: 'int', 'float', 'categorical'
            low: Untere Grenze
            high: Obere Grenze
            choices: Kategorische Werte
            log: Logarithmische Skala
        """
        space = SearchSpace(
            name=name,
            param_type=param_type,
            low=low,
            high=high,
            choices=choices,
            log=log
        )
        self.search_space.append(space)
        self.logger.debug(f'Suchraum hinzugefuegt: {name} ({param_type})')

    def use_default_search_space(self):
        """Verwendet den Standard-Suchraum."""
        self.search_space = self.DEFAULT_SEARCH_SPACE.copy()
        self.logger.info('Standard-Suchraum geladen')

    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Generiert Hyperparameter-Vorschlaege fuer einen Trial."""
        params = {}

        for space in self.search_space:
            if space.param_type == 'int':
                params[space.name] = trial.suggest_int(
                    space.name, int(space.low), int(space.high)
                )
            elif space.param_type == 'float':
                if space.log:
                    params[space.name] = trial.suggest_float(
                        space.name, space.low, space.high, log=True
                    )
                else:
                    params[space.name] = trial.suggest_float(
                        space.name, space.low, space.high
                    )
            elif space.param_type == 'categorical':
                params[space.name] = trial.suggest_categorical(
                    space.name, space.choices
                )

        return params

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Optuna Objective-Funktion.

        Wird fuer jeden Trial aufgerufen und gibt den zu optimierenden Wert zurueck.
        """
        # Hyperparameter vorschlagen
        params = self._suggest_params(trial)

        self.logger.debug(f'Trial {trial.number}: {params}')

        try:
            # Modell erstellen
            model_params = {
                'input_size': self.input_size,
                'num_classes': self.num_classes,
                'hidden_size': params.get('hidden_size', 100),
                'num_layers': params.get('num_layers', 2),
                'dropout': params.get('dropout', 0.2),
            }

            model = ModelFactory.create(self.model_type, **model_params)
            model.to(self.device)

            # Trainer erstellen
            callbacks = [
                EarlyStopping(
                    patience=self.config.early_stopping_patience,
                    monitor=self.config.metric
                )
            ]

            trainer = Trainer(model, device=self.device, callbacks=callbacks)

            # Training mit Pruning-Callback
            best_metric = 0.0

            for epoch in range(1, self.config.epochs_per_trial + 1):
                # Eine Epoche trainieren
                train_loss, train_acc = trainer._train_epoch(self.train_loader)
                val_loss, val_acc = trainer._validate_epoch(self.val_loader)

                # Metric fuer Optuna
                if self.config.metric == 'val_accuracy':
                    current_metric = val_acc
                else:
                    current_metric = -val_loss  # Negativ weil wir maximieren

                best_metric = max(best_metric, current_metric)

                # Pruning pruefen
                if self.config.use_pruning and epoch >= self.config.pruning_warmup_steps:
                    trial.report(current_metric, epoch)

                    if trial.should_prune():
                        self.logger.debug(f'Trial {trial.number} gepruned bei Epoch {epoch}')
                        raise optuna.TrialPruned()

                # Early Stopping pruefen
                if trainer.stop_training:
                    break

            # Trial-Ergebnis speichern
            self.trials_history.append({
                'trial': trial.number,
                'params': params,
                'value': best_metric,
                'epochs': epoch
            })

            return best_metric

        except optuna.TrialPruned:
            raise
        except Exception as e:
            self.logger.error(f'Trial {trial.number} fehlgeschlagen: {e}')
            return float('-inf') if self.config.direction == 'maximize' else float('inf')

    def optimize(
        self,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Startet die Hyperparameter-Optimierung.

        Args:
            n_trials: Anzahl Trials (ueberschreibt Config)
            timeout: Timeout in Sekunden (ueberschreibt Config)
            show_progress: Progress-Bar anzeigen

        Returns:
            Beste gefundene Hyperparameter
        """
        if not self.search_space:
            self.logger.warning('Kein Suchraum definiert, verwende Standard')
            self.use_default_search_space()

        n_trials = n_trials or self.config.n_trials
        timeout = timeout or self.config.timeout

        self.logger.info(f'Starte Optimierung: {n_trials} Trials, Modell: {self.model_type}')

        # Sampler und Pruner
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(
            n_startup_trials=self.config.pruning_warmup_steps,
            n_warmup_steps=self.config.pruning_warmup_steps,
            interval_steps=self.config.pruning_interval_steps
        ) if self.config.use_pruning else None

        # Study erstellen
        self.study = optuna.create_study(
            direction=self.config.direction,
            sampler=sampler,
            pruner=pruner
        )

        # Optimierung starten
        self.study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=self.config.n_jobs,
            show_progress_bar=show_progress
        )

        # Beste Parameter speichern
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value

        self.logger.success(f'Optimierung abgeschlossen!')
        self.logger.info(f'Beste Params: {self.best_params}')
        self.logger.info(f'Bester Wert: {self.best_value:.4f}')

        return self.best_params

    def get_best_model(self) -> BaseModel:
        """
        Erstellt ein Modell mit den besten Parametern.

        Returns:
            Modell-Instanz mit optimierten Hyperparametern
        """
        if not self.best_params:
            raise ValueError('Keine Optimierung durchgefuehrt')

        model_params = {
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'hidden_size': self.best_params.get('hidden_size', 100),
            'num_layers': self.best_params.get('num_layers', 2),
            'dropout': self.best_params.get('dropout', 0.2),
        }

        model = ModelFactory.create(self.model_type, **model_params)
        return model

    def save_results(self, filepath: Path):
        """Speichert Optimierungsergebnisse."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        results = {
            'timestamp': datetime.now().isoformat(),
            'model_type': self.model_type,
            'n_trials': len(self.trials_history),
            'best_params': self.best_params,
            'best_value': self.best_value,
            'config': {
                'n_trials': self.config.n_trials,
                'epochs_per_trial': self.config.epochs_per_trial,
                'direction': self.config.direction,
                'metric': self.config.metric
            },
            'trials': self.trials_history
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.success(f'Ergebnisse gespeichert: {filepath}')

    def get_importance(self) -> Dict[str, float]:
        """
        Gibt die Wichtigkeit der Hyperparameter zurueck.

        Returns:
            Dictionary {param_name: importance}
        """
        if self.study is None:
            return {}

        try:
            importance = optuna.importance.get_param_importances(self.study)
            return dict(importance)
        except Exception as e:
            self.logger.warning(f'Konnte Importance nicht berechnen: {e}')
            return {}

    def plot_optimization_history(self):
        """Plottet die Optimierungshistorie (falls matplotlib verfuegbar)."""
        if self.study is None:
            return

        try:
            from optuna.visualization import plot_optimization_history
            fig = plot_optimization_history(self.study)
            fig.show()
        except ImportError:
            self.logger.warning('Plotly nicht installiert fuer Visualisierung')

    def plot_param_importances(self):
        """Plottet die Parameter-Wichtigkeit."""
        if self.study is None:
            return

        try:
            from optuna.visualization import plot_param_importances
            fig = plot_param_importances(self.study)
            fig.show()
        except ImportError:
            self.logger.warning('Plotly nicht installiert fuer Visualisierung')
