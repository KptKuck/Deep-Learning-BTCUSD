"""
Auto-Trainer - Automatisches Durchprobieren verschiedener Modelle und Parameter.

Features:
- Automatische Modell- und Hyperparameter-Suche
- Einstellbare Komplexitaetsstufen (1-5)
- Early Stopping mit konfigurierbarer Patience
- Learning Rate Scheduler (ReduceLROnPlateau)
- Gradient Clipping
- Class Weights fuer unbalancierte Daten
- Erweiterte Metriken (F1-Score, Precision, Recall)
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from btcusd_analyzer.core.logger import get_logger
from btcusd_analyzer.models.factory import ModelFactory
from btcusd_analyzer.models.hf_transformer import is_hf_available


@dataclass
class AutoTrainResult:
    """Ergebnis eines Auto-Training Durchlaufs."""
    rank: int = 0
    model_type: str = ''
    config: Dict[str, Any] = field(default_factory=dict)
    train_acc: float = 0.0
    val_acc: float = 0.0
    train_loss: float = 0.0
    val_loss: float = 0.0
    f1_score: float = 0.0
    precision: Dict[str, float] = field(default_factory=dict)
    recall: Dict[str, float] = field(default_factory=dict)
    confusion_mat: Optional[List[List[int]]] = None
    epochs_trained: int = 0
    training_time: float = 0.0
    best_epoch: int = 0
    model_state: Optional[Dict] = None
    num_parameters: int = 0
    # Trainingsverlauf pro Epoche
    training_history: Dict[str, List[float]] = field(default_factory=lambda: {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []
    })


# Auto-Trainer Konfigurationen nach Komplexitaet
AUTO_TRAINER_CONFIGS = {
    1: {  # Schnell (~5 Min)
        'max_epochs': 20,
        'patience': 3,
        'configs': [
            ('BiLSTM', {'hidden_sizes': [128, 128]}),
            ('Transformer', {'d_model': 64, 'nhead': 4, 'num_encoder_layers': 2}),
            ('BiGRU', {'hidden_sizes': [128, 128]}),
        ]
    },
    2: {  # Schnell+ (~10 Min)
        'max_epochs': 30,
        'patience': 4,
        'configs': [
            ('BiLSTM', {'hidden_sizes': [128, 128]}),
            ('BiLSTM', {'hidden_sizes': [256, 128]}),
            ('BiGRU', {'hidden_sizes': [128, 128]}),
            ('Transformer', {'d_model': 64, 'nhead': 4, 'num_encoder_layers': 2}),
            ('CNN', {'num_filters': 64}),
        ]
    },
    3: {  # Standard (~20 Min)
        'max_epochs': 50,
        'patience': 5,
        'configs': [
            ('BiLSTM', {'hidden_sizes': [128, 128]}),
            ('BiLSTM', {'hidden_sizes': [256, 128, 64]}),
            ('BiLSTM', {'hidden_sizes': [64, 128, 256]}),
            ('BiGRU', {'hidden_sizes': [128, 128]}),
            ('BiGRU', {'hidden_sizes': [256, 128, 64]}),
            ('Transformer', {'d_model': 64, 'nhead': 4, 'num_encoder_layers': 2}),
            ('Transformer', {'d_model': 128, 'nhead': 4, 'num_encoder_layers': 4}),
            ('CNN-LSTM', {'hidden_size': 128}),
            ('CNN', {'num_filters': 64}),
            ('LSTM', {'hidden_sizes': [256, 256]}),
        ]
    },
    4: {  # Ausfuehrlich (~40 Min)
        'max_epochs': 75,
        'patience': 7,
        'configs': [
            # BiLSTM Varianten
            ('BiLSTM', {'hidden_sizes': [128, 128]}),
            ('BiLSTM', {'hidden_sizes': [256, 128, 64]}),
            ('BiLSTM', {'hidden_sizes': [64, 128, 256]}),
            ('BiLSTM', {'hidden_sizes': [256, 64, 256]}),
            ('BiLSTM', {'hidden_sizes': [256, 128], 'use_attention': True}),
            # BiGRU Varianten
            ('BiGRU', {'hidden_sizes': [128, 128]}),
            ('BiGRU', {'hidden_sizes': [256, 128, 64]}),
            ('BiGRU', {'hidden_sizes': [64, 128, 256]}),
            # Transformer
            ('Transformer', {'d_model': 64, 'nhead': 4, 'num_encoder_layers': 2}),
            ('Transformer', {'d_model': 128, 'nhead': 4, 'num_encoder_layers': 4}),
            ('Transformer', {'d_model': 256, 'nhead': 8, 'num_encoder_layers': 6}),
            # CNN
            ('CNN', {'num_filters': 64}),
            ('CNN', {'num_filters': 128}),
            # CNN-LSTM
            ('CNN-LSTM', {'hidden_size': 128}),
            ('CNN-LSTM', {'hidden_size': 256, 'bidirectional': True}),
        ]
    },
    5: {  # Gruendlich (~60 Min)
        'max_epochs': 100,
        'patience': 10,
        'configs': [
            # BiLSTM Varianten
            ('BiLSTM', {'hidden_sizes': [128, 128]}),
            ('BiLSTM', {'hidden_sizes': [256, 128, 64]}),
            ('BiLSTM', {'hidden_sizes': [64, 128, 256]}),
            ('BiLSTM', {'hidden_sizes': [256, 64, 256]}),
            ('BiLSTM', {'hidden_sizes': [512, 256, 128, 64]}),
            ('BiLSTM', {'hidden_sizes': [256, 128], 'use_attention': True}),
            ('BiLSTM', {'hidden_sizes': [256, 128], 'use_layer_norm': True}),
            # BiGRU Varianten
            ('BiGRU', {'hidden_sizes': [128, 128]}),
            ('BiGRU', {'hidden_sizes': [256, 128, 64]}),
            ('BiGRU', {'hidden_sizes': [64, 128, 256]}),
            # Transformer Varianten
            ('Transformer', {'d_model': 64, 'nhead': 4, 'num_encoder_layers': 2}),
            ('Transformer', {'d_model': 128, 'nhead': 4, 'num_encoder_layers': 4}),
            ('Transformer', {'d_model': 256, 'nhead': 8, 'num_encoder_layers': 6}),
            # CNN Varianten
            ('CNN', {'num_filters': 64}),
            ('CNN', {'num_filters': 128}),
            # CNN-LSTM Varianten
            ('CNN-LSTM', {'hidden_size': 128}),
            ('CNN-LSTM', {'hidden_size': 256, 'bidirectional': True}),
            # LSTM (unidirektional)
            ('LSTM', {'hidden_sizes': [256, 256]}),
            ('LSTM', {'hidden_sizes': [512, 256]}),
            # Dropout Variationen
            ('BiLSTM', {'hidden_sizes': [256, 128], 'dropout': 0.2}),
            ('BiLSTM', {'hidden_sizes': [256, 128], 'dropout': 0.4}),
            ('BiLSTM', {'hidden_sizes': [256, 128], 'dropout': 0.5}),
            # HuggingFace (falls installiert)
            ('HF-Transformer', {'d_model': 64, 'num_encoder_layers': 2}),
            ('HF-Transformer', {'d_model': 128, 'num_encoder_layers': 4}),
        ]
    }
}


class AutoTrainer:
    """
    Automatisches Training verschiedener Modellkonfigurationen.

    Testet automatisch verschiedene Modelle und Hyperparameter
    und rankt sie nach Validation Accuracy.

    Features:
    - Early Stopping bei Stagnation
    - Learning Rate Scheduler
    - Gradient Clipping
    - Class Weights fuer unbalancierte Daten
    - Erweiterte Metriken

    Usage:
        trainer = AutoTrainer(train_loader, val_loader, input_size=6)
        results = trainer.run(complexity=3, progress_callback=callback)
        best_model, config = trainer.get_best_model()
    """

    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        input_size: int,
        num_classes: int = 3,
        device: Optional[torch.device] = None,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialisiert den AutoTrainer.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            input_size: Anzahl Input-Features
            num_classes: Anzahl Klassen
            device: PyTorch Device (auto-detect wenn None)
            class_names: Namen der Klassen fuer Reports
        """
        self.logger = get_logger()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.input_size = input_size
        self.num_classes = num_classes
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names or ['HOLD', 'BUY', 'SELL']

        self.results: List[AutoTrainResult] = []
        self._stop_requested = False

        # Class Weights aus Training-Daten berechnen
        self.class_weights = self._compute_class_weights()

        self.logger.info(f'AutoTrainer initialisiert auf {self.device}')

    def _compute_class_weights(self) -> torch.Tensor:
        """Berechnet Class Weights aus den Training-Daten."""
        all_labels = []
        for _, labels in self.train_loader:
            all_labels.append(labels)

        labels = torch.cat(all_labels)
        class_counts = torch.bincount(labels, minlength=self.num_classes).float()

        # Inverse Frequency Weighting
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * self.num_classes

        return class_weights.to(self.device)

    def stop(self):
        """Stoppt das Training nach dem aktuellen Modell."""
        self._stop_requested = True
        self.logger.info('Auto-Training Stop angefordert')

    def run(
        self,
        complexity: int = 3,
        progress_callback: Optional[Callable[[int, int, str, Dict[str, float]], None]] = None,
        learning_rate: float = 0.001
    ) -> List[AutoTrainResult]:
        """
        Fuehrt Auto-Training durch.

        Args:
            complexity: Komplexitaetsstufe 1-5
            progress_callback: Callback(current, total, model_name, metrics_dict)
                               metrics_dict: {'progress': 0-1, 'train_loss', 'train_acc', 'val_loss', 'val_acc'}
            learning_rate: Basis-Lernrate

        Returns:
            Liste der AutoTrainResult, sortiert nach val_acc
        """
        self._stop_requested = False
        self.results = []

        complexity = max(1, min(5, complexity))
        config = AUTO_TRAINER_CONFIGS[complexity]
        configs = config['configs']

        # Filtere HF-Transformer wenn nicht verfuegbar
        if not is_hf_available():
            configs = [(m, p) for m, p in configs if not m.lower().startswith('hf')]

        total = len(configs)
        self.logger.info(f'Starte Auto-Training: Komplexitaet {complexity}, {total} Konfigurationen')

        for i, (model_type, params) in enumerate(configs):
            if self._stop_requested:
                self.logger.info('Auto-Training abgebrochen')
                break

            # Progress Update
            if progress_callback:
                progress_callback(i, total, model_type, {'progress': 0.0})

            self.logger.info(f'[{i+1}/{total}] Training: {model_type} - {params}')

            # Erstelle epoch_callback der Metriken weitergibt
            def make_epoch_callback(model_idx, model_name):
                def callback(e, ep, tl, ta, vl, va):
                    if progress_callback:
                        progress_callback(model_idx, total, model_name, {
                            'progress': e / ep,
                            'train_loss': tl,
                            'train_acc': ta,
                            'val_loss': vl,
                            'val_acc': va
                        })
                return callback

            try:
                result = self._train_single(
                    model_type=model_type,
                    params=params,
                    max_epochs=config['max_epochs'],
                    patience=config['patience'],
                    learning_rate=learning_rate,
                    epoch_callback=make_epoch_callback(i, model_type)
                )
                self.results.append(result)

                self.logger.info(
                    f'  -> Val ACC: {result.val_acc:.2%}, F1: {result.f1_score:.3f}, '
                    f'Epochen: {result.epochs_trained}/{config["max_epochs"]}'
                )

            except Exception as e:
                self.logger.error(f'Fehler bei {model_type}: {e}')
                continue

        # Nach Val ACC sortieren und ranken
        self.results.sort(key=lambda x: x.val_acc, reverse=True)
        for i, r in enumerate(self.results):
            r.rank = i + 1

        self.logger.success(f'Auto-Training abgeschlossen: {len(self.results)} Modelle getestet')

        return self.results

    def _train_single(
        self,
        model_type: str,
        params: Dict[str, Any],
        max_epochs: int,
        patience: int,
        learning_rate: float,
        epoch_callback: Optional[Callable[[int, int, float, float, float, float], None]] = None
    ) -> AutoTrainResult:
        """
        Trainiert ein einzelnes Modell.

        Args:
            model_type: Modell-Typ
            params: Modell-Parameter
            max_epochs: Maximale Epochen
            patience: Early Stopping Patience
            learning_rate: Lernrate
            epoch_callback: Callback(epoch, max_epochs, train_loss, train_acc, val_loss, val_acc)

        Returns:
            AutoTrainResult
        """
        start_time = time.time()

        # Modell erstellen
        model = ModelFactory.create(
            model_name=model_type,
            input_size=self.input_size,
            num_classes=self.num_classes,
            **params
        )
        model.to(self.device)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Optimizer mit Weight Decay
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Learning Rate Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            patience=patience // 2,
            factor=0.5,
            min_lr=1e-6
        )

        # Loss mit Class Weights
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        # Training Loop mit Early Stopping
        best_val_acc = 0.0
        best_epoch = 0
        best_state = None
        epochs_without_improvement = 0

        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0

        # Trainingsverlauf speichern
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'lr': []
        }

        for epoch in range(1, max_epochs + 1):
            if self._stop_requested:
                break

            # Training
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in self.train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_loss = running_loss / len(self.train_loader)
            train_acc = correct / total

            # Validation
            model.eval()
            running_loss = 0.0
            correct = 0
            total = 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_loss = running_loss / len(self.val_loader)
            val_acc = correct / total

            # LR Scheduler Step
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_acc)

            # Trainingsverlauf speichern
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(current_lr)

            # Early Stopping Check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                break

            # Epoch Callback mit Metriken
            if epoch_callback:
                epoch_callback(epoch, max_epochs, train_loss, train_acc, val_loss, val_acc)

        # Metriken berechnen
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
        recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
        conf_mat = confusion_matrix(all_labels, all_preds).tolist()

        precision_dict = {self.class_names[i]: float(p) for i, p in enumerate(precision)}
        recall_dict = {self.class_names[i]: float(r) for i, r in enumerate(recall)}

        training_time = time.time() - start_time

        return AutoTrainResult(
            model_type=model_type,
            config=params,
            train_acc=train_acc,
            val_acc=best_val_acc,
            train_loss=train_loss,
            val_loss=val_loss,
            f1_score=f1,
            precision=precision_dict,
            recall=recall_dict,
            confusion_mat=conf_mat,
            epochs_trained=epoch,
            training_time=training_time,
            best_epoch=best_epoch,
            model_state=best_state,
            num_parameters=num_params,
            training_history=history
        )

    def get_best_model(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Gibt das beste Modell zurueck.

        Returns:
            Tuple (model, config)
        """
        if not self.results:
            raise ValueError('Kein Training durchgefuehrt')

        best = self.results[0]
        model = ModelFactory.create(
            model_name=best.model_type,
            input_size=self.input_size,
            num_classes=self.num_classes,
            **best.config
        )
        model.load_state_dict(best.model_state)
        model.to(self.device)

        return model, best.config

    def get_results_summary(self) -> str:
        """Gibt eine Zusammenfassung der Ergebnisse zurueck."""
        if not self.results:
            return "Keine Ergebnisse"

        lines = [
            "=" * 80,
            "AUTO-TRAINER ERGEBNISSE",
            "=" * 80,
            f"{'Rang':<5}{'Modell':<15}{'Config':<25}{'Val ACC':<10}{'F1':<8}{'Epochen':<10}{'Params':<12}",
            "-" * 80
        ]

        for r in self.results[:10]:  # Top 10
            config_str = str(r.config.get('hidden_sizes', r.config))[:22]
            lines.append(
                f"{r.rank:<5}{r.model_type:<15}{config_str:<25}"
                f"{r.val_acc:<10.2%}{r.f1_score:<8.3f}{r.epochs_trained:<10}{r.num_parameters:<12,}"
            )

        lines.append("=" * 80)
        return "\n".join(lines)


def get_complexity_info(complexity: int) -> Dict[str, Any]:
    """
    Gibt Informationen zu einer Komplexitaetsstufe zurueck.

    Args:
        complexity: Stufe 1-5

    Returns:
        Dict mit max_epochs, patience, num_configs
    """
    complexity = max(1, min(5, complexity))
    config = AUTO_TRAINER_CONFIGS[complexity]

    return {
        'max_epochs': config['max_epochs'],
        'patience': config['patience'],
        'num_configs': len(config['configs']),
        'configs': [(m, p) for m, p in config['configs']]
    }
