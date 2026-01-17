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

from btcusd_analyzer.core.logger import get_logger


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
        self._logger = get_logger()
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Standard-Dateinamen
        self.config_file = self.session_dir / 'session_config.json'
        self.training_data_file = self.session_dir / 'training_data.npz'
        self.backtest_data_file = self.session_dir / 'backtest_data.csv'

        self._logger.debug(f"[SessionManager] Initialisiert: {self.session_dir.name}")

    # =========================================================================
    # Session-Konfiguration
    # =========================================================================

    def save_config(self, config: Dict[str, Any]):
        """
        Speichert die Session-Konfiguration.

        Args:
            config: Dictionary mit allen Parametern
        """
        self._logger.debug(f"[SessionManager] === SAVE CONFIG START ===")
        self._logger.debug(f"[SessionManager] Zieldatei: {self.config_file}")
        self._logger.debug(f"[SessionManager] Config-Keys: {list(config.keys())}")

        # Timestamp hinzufuegen
        config['saved_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        config['session_name'] = self.session_dir.name

        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False, default=str)

        file_size = self.config_file.stat().st_size / 1024
        self._logger.debug(f"[SessionManager] Config gespeichert: {file_size:.1f} KB")
        self._logger.debug(f"[SessionManager] === SAVE CONFIG DONE ===")

    def set_status(self, status: str):
        """
        Setzt den Session-Status.

        Args:
            status: 'prepared' oder 'trained'
        """
        self._logger.debug(f"[SessionManager] === SET STATUS: {status} ===")

        # Bestehende Config laden oder neue erstellen
        config = self.load_config() or {}
        config['status'] = status
        config['status_updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        self.save_config(config)
        self._logger.debug(f"[SessionManager] Status gesetzt: {status}")

        # Session-DB aktualisieren
        self._update_session_db({'status': status})

    def get_status(self) -> Optional[str]:
        """
        Gibt den Session-Status zurueck.

        Returns:
            'prepared', 'trained' oder None
        """
        config = self.load_config()
        if config:
            status = config.get('status')
            self._logger.debug(f"[SessionManager] Status gelesen: {status}")
            return status
        return None

    def load_config(self) -> Optional[Dict[str, Any]]:
        """Laedt die Session-Konfiguration."""
        self._logger.debug(f"[SessionManager] === LOAD CONFIG START ===")
        self._logger.debug(f"[SessionManager] Quelldatei: {self.config_file}")

        if not self.config_file.exists():
            self._logger.debug(f"[SessionManager] Config existiert nicht!")
            return None

        with open(self.config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        self._logger.debug(f"[SessionManager] Config geladen, Keys: {list(config.keys())}")
        if 'features' in config:
            self._logger.debug(f"[SessionManager] Features: {config['features']}")
        if 'training_info' in config:
            self._logger.debug(f"[SessionManager] Training-Info: {config['training_info']}")
        self._logger.debug(f"[SessionManager] === LOAD CONFIG DONE ===")

        return config

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
        self._logger.debug(f"[SessionManager] === SAVE TRAINING DATA START ===")
        self._logger.debug(f"[SessionManager] Zieldatei: {self.training_data_file}")
        self._logger.debug(f"[SessionManager] Sequences Shape: {sequences.shape}")
        self._logger.debug(f"[SessionManager] Sequences dtype: {sequences.dtype}")
        self._logger.debug(f"[SessionManager] Labels Shape: {labels.shape}")
        self._logger.debug(f"[SessionManager] Labels dtype: {labels.dtype}")
        self._logger.debug(f"[SessionManager] Labels unique: {np.unique(labels)}")
        self._logger.debug(f"[SessionManager] Features ({len(features)}): {features}")
        self._logger.debug(f"[SessionManager] Params: {params}")

        np.savez_compressed(
            self.training_data_file,
            sequences=sequences,
            labels=labels,
            features=np.array(features, dtype=object),
            params=json.dumps(params)
        )

        file_size = self.training_data_file.stat().st_size / (1024 * 1024)
        self._logger.debug(f"[SessionManager] Training-Daten gespeichert: {file_size:.2f} MB")
        self._logger.debug(f"[SessionManager] === SAVE TRAINING DATA DONE ===")

    def load_training_data(self) -> Optional[Dict[str, Any]]:
        """
        Laedt die Trainingsdaten.

        Returns:
            Dictionary mit sequences, labels, features, params
        """
        self._logger.debug(f"[SessionManager] === LOAD TRAINING DATA START ===")
        self._logger.debug(f"[SessionManager] Quelldatei: {self.training_data_file}")

        if not self.training_data_file.exists():
            self._logger.debug(f"[SessionManager] Training-Daten existieren nicht!")
            return None

        file_size = self.training_data_file.stat().st_size / (1024 * 1024)
        self._logger.debug(f"[SessionManager] Dateigroesse: {file_size:.2f} MB")

        data = np.load(self.training_data_file, allow_pickle=True)

        sequences = data['sequences']
        labels = data['labels']
        features = list(data['features'])
        params = json.loads(str(data['params']))

        self._logger.debug(f"[SessionManager] Sequences Shape: {sequences.shape}")
        self._logger.debug(f"[SessionManager] Sequences dtype: {sequences.dtype}")
        self._logger.debug(f"[SessionManager] Labels Shape: {labels.shape}")
        self._logger.debug(f"[SessionManager] Labels dtype: {labels.dtype}")
        self._logger.debug(f"[SessionManager] Labels unique: {np.unique(labels)}")
        self._logger.debug(f"[SessionManager] Features ({len(features)}): {features}")
        self._logger.debug(f"[SessionManager] Params: {params}")
        self._logger.debug(f"[SessionManager] === LOAD TRAINING DATA DONE ===")

        return {
            'sequences': sequences,
            'labels': labels,
            'features': features,
            'params': params
        }

    # =========================================================================
    # Backtest-Daten
    # =========================================================================

    def save_backtest_data(self, data: pd.DataFrame):
        """
        Speichert die Backtest-Daten.

        Stellt sicher, dass DateTime korrekt als benannter Index gespeichert wird,
        damit sie beim Laden wiederhergestellt werden kann.

        Args:
            data: DataFrame mit OHLCV-Daten
        """
        self._logger.debug(f"[SessionManager] === SAVE BACKTEST DATA START ===")
        self._logger.debug(f"[SessionManager] Zieldatei: {self.backtest_data_file}")
        self._logger.debug(f"[SessionManager] Input Shape: {data.shape}")
        self._logger.debug(f"[SessionManager] Input Columns: {list(data.columns)}")
        self._logger.debug(f"[SessionManager] Input Index Type: {type(data.index).__name__}")

        data = data.copy()

        # Falls DateTime als Spalte vorhanden, als Index setzen
        if 'DateTime' in data.columns:
            self._logger.debug(f"[SessionManager] DateTime als Spalte gefunden -> setze als Index")
            data = data.set_index('DateTime')

        # Index-Namen sicherstellen (wichtig fuer korrektes Laden)
        if isinstance(data.index, pd.DatetimeIndex):
            data.index.name = 'DateTime'

        self._logger.debug(f"[SessionManager] Final Index: {data.index.name}, Type: {type(data.index).__name__}")
        self._logger.debug(f"[SessionManager] Final Columns: {list(data.columns)}")
        self._logger.debug(f"[SessionManager] Zeitraum: {data.index[0]} bis {data.index[-1]}")

        data.to_csv(self.backtest_data_file, index=True)

        file_size = self.backtest_data_file.stat().st_size / 1024
        self._logger.debug(f"[SessionManager] Backtest-Daten gespeichert: {file_size:.1f} KB")
        self._logger.debug(f"[SessionManager] === SAVE BACKTEST DATA DONE ===")

    def load_backtest_data(self) -> Optional[pd.DataFrame]:
        """
        Laedt die Backtest-Daten.

        Unterstuetzt sowohl das alte Format (numerischer Index + DateTime-Spalte)
        als auch das neue Format (DateTime als Index).

        Stellt DateTime sowohl als Index (DatetimeIndex) als auch als
        Spalte zur Verfuegung, damit nachfolgende Module flexibel darauf
        zugreifen koennen.
        """
        self._logger.debug(f"[SessionManager] === LOAD BACKTEST DATA START ===")
        self._logger.debug(f"[SessionManager] Quelldatei: {self.backtest_data_file}")

        if not self.backtest_data_file.exists():
            self._logger.debug(f"[SessionManager] Backtest-Daten existieren nicht!")
            return None

        file_size = self.backtest_data_file.stat().st_size / 1024
        self._logger.debug(f"[SessionManager] Dateigroesse: {file_size:.1f} KB")

        # Zuerst ohne index_col lesen um Format zu erkennen
        df = pd.read_csv(self.backtest_data_file)

        # Erste Spalte pruefen: "Unnamed: 0" = numerischer Index (altes Format)
        # "DateTime" = neues Format
        first_col = df.columns[0]
        self._logger.debug(f"[SessionManager] Erste Spalte: '{first_col}'")
        self._logger.debug(f"[SessionManager] Alle Spalten: {list(df.columns)}")

        if first_col == 'DateTime':
            # Neues Format: DateTime ist bereits Index
            self._logger.debug(f"[SessionManager] Neues Format erkannt (DateTime als erste Spalte)")
            df = df.set_index('DateTime')
            df.index = pd.to_datetime(df.index)
        elif 'DateTime' in df.columns:
            # Altes Format: DateTime als separate Spalte
            self._logger.debug(f"[SessionManager] Altes Format erkannt (DateTime als separate Spalte)")
            # Numerischen Index verwerfen (Unnamed: 0)
            if first_col.startswith('Unnamed'):
                df = df.drop(columns=[first_col])
            df = df.set_index('DateTime')
            df.index = pd.to_datetime(df.index)
        else:
            # Fallback: Erste Spalte als Index versuchen
            self._logger.debug(f"[SessionManager] Fallback: Erste Spalte '{first_col}' als Index")
            df = df.set_index(df.columns[0])
            df.index = pd.to_datetime(df.index)

        df.index.name = 'DateTime'

        # DateTime auch als Spalte hinzufuegen fuer Module die sie erwarten
        # (z.B. FeatureProcessor fuer hour_sin/hour_cos)
        if 'DateTime' not in df.columns:
            df['DateTime'] = df.index

        self._logger.debug(f"[SessionManager] Geladen: {len(df)} Zeilen")
        self._logger.debug(f"[SessionManager] Spalten: {list(df.columns)}")
        self._logger.debug(f"[SessionManager] Index Type: {type(df.index).__name__}")
        self._logger.debug(f"[SessionManager] Zeitraum: {df.index[0]} bis {df.index[-1]}")
        self._logger.debug(f"[SessionManager] === LOAD BACKTEST DATA DONE ===")

        return df

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
        self._logger.debug(f"[SessionManager] === SAVE MODEL START ===")
        self._logger.debug(f"[SessionManager] Quellpfad: {model_path}")

        model_path = Path(model_path)
        dest_path = self.session_dir / model_path.name

        self._logger.debug(f"[SessionManager] Zielpfad: {dest_path}")

        # .pt Datei kopieren
        shutil.copy2(model_path, dest_path)
        pt_size = dest_path.stat().st_size / (1024 * 1024)
        self._logger.debug(f"[SessionManager] .pt kopiert: {pt_size:.2f} MB")

        # .json Datei kopieren (falls vorhanden)
        json_path = model_path.with_suffix('.json')
        if json_path.exists():
            json_dest = self.session_dir / json_path.name
            shutil.copy2(json_path, json_dest)
            json_size = json_dest.stat().st_size / 1024
            self._logger.debug(f"[SessionManager] .json kopiert: {json_size:.1f} KB")
        else:
            self._logger.debug(f"[SessionManager] Keine .json Datei gefunden: {json_path}")

        self._logger.debug(f"[SessionManager] === SAVE MODEL DONE ===")
        return dest_path

    def get_model_path(self) -> Optional[Path]:
        """Gibt den Pfad zum Modell im Session-Ordner zurueck."""
        self._logger.debug(f"[SessionManager] Suche Modelle in: {self.session_dir}")
        models = list(self.session_dir.glob('*.pt'))
        self._logger.debug(f"[SessionManager] Gefundene .pt Dateien: {[m.name for m in models]}")

        if models:
            # Neuestes Modell zurueckgeben
            newest = max(models, key=lambda p: p.stat().st_mtime)
            self._logger.debug(f"[SessionManager] Neuestes Modell: {newest.name}")
            return newest

        self._logger.debug(f"[SessionManager] Kein Modell gefunden")
        return None

    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Laedt die Modell-Info (JSON-Plakette)."""
        self._logger.debug(f"[SessionManager] === GET MODEL INFO ===")
        model_path = self.get_model_path()
        if model_path is None:
            self._logger.debug(f"[SessionManager] Kein Modell -> keine Info")
            return None

        json_path = model_path.with_suffix('.json')
        self._logger.debug(f"[SessionManager] Suche JSON: {json_path}")

        if not json_path.exists():
            self._logger.debug(f"[SessionManager] JSON existiert nicht!")
            return None

        with open(json_path, 'r', encoding='utf-8') as f:
            info = json.load(f)

        self._logger.debug(f"[SessionManager] Model-Info Keys: {list(info.keys())}")
        self._logger.debug(f"[SessionManager] Model-Type: {info.get('model_type', '-')}")
        self._logger.debug(f"[SessionManager] Accuracy: {info.get('best_accuracy', 0)}")
        self._logger.debug(f"[SessionManager] Hidden-Sizes: {info.get('hidden_sizes', '-')}")
        self._logger.debug(f"[SessionManager] Input-Size: {info.get('input_size', '-')}")
        self._logger.debug(f"[SessionManager] Num-Classes: {info.get('num_classes', '-')}")

        return info

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
            'status': None,
        }

        # Config laden falls vorhanden
        if summary['has_config']:
            config = self.load_config()
            if config:
                summary['source_file'] = config.get('source_file', '-')
                summary['features_count'] = len(config.get('features', []))
                summary['status'] = config.get('status')

        # Model-Info laden falls vorhanden
        model_info = self.get_model_info()
        if model_info:
            summary['model_accuracy'] = model_info.get('best_accuracy', 0)
            summary['model_type'] = model_info.get('model_type', '-')

        return summary

    # =========================================================================
    # Session-Datenbank Integration
    # =========================================================================

    def _get_session_db(self):
        """
        Gibt die SessionDatabase zurueck (lazy loading).

        Ermittelt das data-Verzeichnis aus dem Session-Pfad.
        """
        try:
            from .session_database import SessionDatabase

            # data-Verzeichnis ermitteln: session_dir ist in log/, data ist daneben
            # z.B. log/session-xxx -> data/sessions.json
            project_dir = self.session_dir.parent.parent
            data_dir = project_dir / 'data'

            return SessionDatabase(data_dir)
        except Exception as e:
            self._logger.debug(f"[SessionManager] SessionDatabase nicht verfuegbar: {e}")
            return None

    def _update_session_db(self, updates: dict):
        """
        Aktualisiert die Session in der Datenbank.

        Args:
            updates: Dictionary mit zu aktualisierenden Feldern
        """
        db = self._get_session_db()
        if db:
            session_id = self.session_dir.name
            # Immer path mitgeben, falls Session neu erstellt wird
            updates_with_path = {
                'path': str(self.session_dir),
                **updates
            }
            db.update_session(session_id, updates_with_path)

    def register_in_db(self, session_info: dict = None):
        """
        Registriert diese Session in der Datenbank.

        Args:
            session_info: Optionale zusaetzliche Infos
        """
        db = self._get_session_db()
        if db:
            info = {
                'id': self.session_dir.name,
                'path': str(self.session_dir),
                'has_training_data': self.training_data_file.exists(),
                'has_backtest_data': self.backtest_data_file.exists(),
                'has_model': self.get_model_path() is not None,
            }
            if session_info:
                info.update(session_info)

            db.add_session(info)
            self._logger.debug(f"[SessionManager] In DB registriert: {self.session_dir.name}")

    @staticmethod
    def list_sessions(log_dir: Path) -> List[Dict[str, Any]]:
        """
        Listet alle verfuegbaren Sessions auf.

        Verwendet die SessionDatabase fuer schnelles Laden.
        Fuehrt bei Bedarf automatisch Migration durch.

        Args:
            log_dir: Pfad zum Log-Verzeichnis

        Returns:
            Liste von Session-Summaries
        """
        from .session_database import SessionDatabase

        log_dir = Path(log_dir)

        # data-Verzeichnis ermitteln
        project_dir = log_dir.parent
        data_dir = project_dir / 'data'

        try:
            db = SessionDatabase(data_dir)

            # Pruefen ob DB leer ist -> Migration
            sessions = db.list_sessions()
            if not sessions and log_dir.exists():
                # Migration durchfuehren
                db.migrate_from_folders(log_dir)
                sessions = db.list_sessions()

            # Format anpassen fuer Kompatibilitaet mit bestehendem Code
            result = []
            for s in sessions:
                result.append({
                    'session_dir': s.get('path', ''),
                    'session_name': s.get('id', ''),
                    'has_config': True,  # Wenn in DB, dann hat es Config
                    'has_training_data': s.get('has_training_data', False),
                    'has_backtest_data': s.get('has_backtest_data', False),
                    'has_model': s.get('has_model', False),
                    'status': s.get('status'),
                    'model_accuracy': s.get('model_accuracy', 0),
                    'model_type': s.get('model_type', '-'),
                    'features_count': s.get('num_features', 0),
                })

            return result

        except Exception as e:
            # Fallback: Ordner-Scan (alte Methode)
            get_logger().warning(f"[SessionManager] DB-Fehler, nutze Ordner-Scan: {e}")

            sessions = []
            if not log_dir.exists():
                return sessions

            for item in sorted(log_dir.iterdir(), reverse=True):
                if item.is_dir() and item.name.startswith('session-'):
                    manager = SessionManager(item)
                    summary = manager.get_summary()

                    if summary['has_training_data'] or summary['has_model']:
                        sessions.append(summary)

            return sessions
