"""
SessionManager - Verwaltet Session-Daten fuer Reproduzierbarkeit

Speichert alle relevanten Daten einer Session:
- Trainingsdaten (Sequenzen, Labels)
- Backtest-Daten
- Modelle
- Konfiguration
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import numpy as np
import pandas as pd


class SessionManager:
    """
    Verwaltet Session-Daten fuer spaetere Reproduzierbarkeit.

    Session-Ordner Struktur:
        log/session-YYYY-MM-DD_HHhMMmSSs/
            ├── session_config.json      # Alle Parameter
            ├── training_data.npz        # Sequenzen + Labels
            ├── backtest_data.csv        # Reservierte Backtest-Daten
            ├── model_*.pt               # Trainiertes Modell
            └── model_*.json             # Modell-Plakette
    """

    def __init__(self, session_dir: Path):
        """
        Initialisiert den SessionManager.

        Args:
            session_dir: Pfad zum Session-Ordner
        """
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Standard-Dateinamen
        self.config_file = self.session_dir / 'session_config.json'
        self.training_data_file = self.session_dir / 'training_data.npz'
        self.backtest_data_file = self.session_dir / 'backtest_data.csv'

    # =========================================================================
    # Session-Konfiguration
    # =========================================================================

    def save_config(self, config: Dict[str, Any]):
        """
        Speichert die Session-Konfiguration.

        Args:
            config: Dictionary mit allen Parametern
        """
        # Timestamp hinzufuegen
        config['saved_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        config['session_name'] = self.session_dir.name

        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False, default=str)

    def load_config(self) -> Optional[Dict[str, Any]]:
        """Laedt die Session-Konfiguration."""
        if not self.config_file.exists():
            return None

        with open(self.config_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    # =========================================================================
    # Trainingsdaten
    # =========================================================================

    def save_training_data(self,
                           sequences: np.ndarray,
                           labels: np.ndarray,
                           features: List[str],
                           params: Dict[str, Any]):
        """
        Speichert die Trainingsdaten.

        Args:
            sequences: Feature-Sequenzen (N, lookback, features)
            labels: Labels (N,)
            features: Liste der Feature-Namen
            params: Parameter (lookback, lookforward, etc.)
        """
        np.savez_compressed(
            self.training_data_file,
            sequences=sequences,
            labels=labels,
            features=np.array(features, dtype=object),
            params=json.dumps(params)
        )

    def load_training_data(self) -> Optional[Dict[str, Any]]:
        """
        Laedt die Trainingsdaten.

        Returns:
            Dictionary mit sequences, labels, features, params
        """
        if not self.training_data_file.exists():
            return None

        data = np.load(self.training_data_file, allow_pickle=True)
        return {
            'sequences': data['sequences'],
            'labels': data['labels'],
            'features': list(data['features']),
            'params': json.loads(str(data['params']))
        }

    # =========================================================================
    # Backtest-Daten
    # =========================================================================

    def save_backtest_data(self, data: pd.DataFrame):
        """
        Speichert die Backtest-Daten.

        Args:
            data: DataFrame mit OHLCV-Daten
        """
        data.to_csv(self.backtest_data_file, index=True)

    def load_backtest_data(self) -> Optional[pd.DataFrame]:
        """Laedt die Backtest-Daten."""
        if not self.backtest_data_file.exists():
            return None

        return pd.read_csv(self.backtest_data_file, index_col=0, parse_dates=True)

    # =========================================================================
    # Modelle
    # =========================================================================

    def save_model(self, model_path: Path) -> Path:
        """
        Kopiert ein Modell in den Session-Ordner.

        Args:
            model_path: Pfad zum Modell (.pt Datei)

        Returns:
            Pfad zur kopierten Datei im Session-Ordner
        """
        model_path = Path(model_path)
        dest_path = self.session_dir / model_path.name

        # .pt Datei kopieren
        shutil.copy2(model_path, dest_path)

        # .json Datei kopieren (falls vorhanden)
        json_path = model_path.with_suffix('.json')
        if json_path.exists():
            shutil.copy2(json_path, self.session_dir / json_path.name)

        return dest_path

    def get_model_path(self) -> Optional[Path]:
        """Gibt den Pfad zum Modell im Session-Ordner zurueck."""
        models = list(self.session_dir.glob('*.pt'))
        if models:
            # Neuestes Modell zurueckgeben
            return max(models, key=lambda p: p.stat().st_mtime)
        return None

    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Laedt die Modell-Info (JSON-Plakette)."""
        model_path = self.get_model_path()
        if model_path is None:
            return None

        json_path = model_path.with_suffix('.json')
        if not json_path.exists():
            return None

        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # =========================================================================
    # Session-Uebersicht
    # =========================================================================

    def get_summary(self) -> Dict[str, Any]:
        """Gibt eine Uebersicht der Session zurueck."""
        summary = {
            'session_dir': str(self.session_dir),
            'session_name': self.session_dir.name,
            'has_config': self.config_file.exists(),
            'has_training_data': self.training_data_file.exists(),
            'has_backtest_data': self.backtest_data_file.exists(),
            'has_model': self.get_model_path() is not None,
        }

        # Config laden falls vorhanden
        if summary['has_config']:
            config = self.load_config()
            if config:
                summary['source_file'] = config.get('source_file', '-')
                summary['features_count'] = len(config.get('features', []))

        # Model-Info laden falls vorhanden
        model_info = self.get_model_info()
        if model_info:
            summary['model_accuracy'] = model_info.get('best_accuracy', 0)
            summary['model_type'] = model_info.get('model_type', '-')

        return summary

    @staticmethod
    def list_sessions(log_dir: Path) -> List[Dict[str, Any]]:
        """
        Listet alle verfuegbaren Sessions auf.

        Args:
            log_dir: Pfad zum Log-Verzeichnis

        Returns:
            Liste von Session-Summaries
        """
        sessions = []
        log_dir = Path(log_dir)

        if not log_dir.exists():
            return sessions

        # Alle Session-Ordner finden (nicht .txt Dateien)
        for item in sorted(log_dir.iterdir(), reverse=True):
            if item.is_dir() and item.name.startswith('session-'):
                manager = SessionManager(item)
                summary = manager.get_summary()

                # Nur Sessions mit Daten anzeigen
                if summary['has_training_data'] or summary['has_model']:
                    sessions.append(summary)

        return sessions
