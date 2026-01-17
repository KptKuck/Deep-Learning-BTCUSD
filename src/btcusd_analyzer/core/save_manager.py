"""
SaveManager - Zentrale Speicher-Koordination fuer Sessions

Ersetzt verteilte Speicher-Logik durch:
- Explizites Speichern (kein Autosave)
- Nachfrage bei Ueberschreiben
- Atomare Transaktionen mit Rollback
- Ein Session-Ordner als Single Source of Truth
"""

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from btcusd_analyzer.core.exceptions import SaveError
from btcusd_analyzer.core.logger import get_logger


class OverwriteAction(Enum):
    """Moegliche Aktionen bei Ueberschreiben."""
    CANCEL = 'cancel'
    OVERWRITE = 'overwrite'
    NEW_SESSION = 'new'


@dataclass
class SaveCheckResult:
    """Ergebnis der Speicher-Pruefung."""
    can_save: bool = True
    needs_confirmation: bool = False
    existing_data: List[str] = field(default_factory=list)
    changes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class SessionConfig:
    """
    Konfiguration einer Session.

    Single Source of Truth fuer alle Session-Metadaten.
    """
    session_id: str
    status: str = 'created'  # 'created', 'prepared', 'trained'
    created_at: str = ''
    updated_at: str = ''

    # Datenquellen
    source_file: str = ''
    date_range: Dict[str, str] = field(default_factory=dict)

    # Preparation Info
    features: List[str] = field(default_factory=list)
    num_features: int = 0
    num_samples: int = 0
    lookback: int = 0
    lookforward: int = 0
    preparation_params: Dict[str, Any] = field(default_factory=dict)

    # Training Info
    model_info: Optional[Dict[str, Any]] = None
    training_metrics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary fuer JSON-Serialisierung."""
        return {
            'session_id': self.session_id,
            'status': self.status,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'source_file': self.source_file,
            'date_range': self.date_range,
            'features': self.features,
            'num_features': self.num_features,
            'num_samples': self.num_samples,
            'lookback': self.lookback,
            'lookforward': self.lookforward,
            'preparation_params': self.preparation_params,
            'model_info': self.model_info,
            'training_metrics': self.training_metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionConfig':
        """Erstellt SessionConfig aus Dictionary."""
        return cls(
            session_id=data.get('session_id', ''),
            status=data.get('status', 'created'),
            created_at=data.get('created_at', ''),
            updated_at=data.get('updated_at', ''),
            source_file=data.get('source_file', ''),
            date_range=data.get('date_range', {}),
            features=data.get('features', []),
            num_features=data.get('num_features', 0),
            num_samples=data.get('num_samples', 0),
            lookback=data.get('lookback', 0),
            lookforward=data.get('lookforward', 0),
            preparation_params=data.get('preparation_params', {}),
            model_info=data.get('model_info'),
            training_metrics=data.get('training_metrics'),
        )


class SaveManager:
    """
    Zentrale Speicher-Koordination fuer Sessions.

    Regelt:
    - Was existiert bereits?
    - Was wuerde ueberschrieben?
    - Nachfrage an Benutzer (via SaveCheckResult)
    - Atomares Speichern mit Rollback

    Speicherorte (alle im Session-Ordner):
    - session_config.json: Single Source of Truth
    - training_data.npz: Sequenzen + Labels
    - backtest_data.csv: Backtest-Daten
    - model.pt: Trainiertes Modell
    - model.json: Modell-Metadaten
    """

    # Dateinamen-Konstanten
    CONFIG_FILE = 'session_config.json'
    TRAINING_DATA_FILE = 'training_data.npz'
    BACKTEST_DATA_FILE = 'backtest_data.csv'
    MODEL_FILE = 'model.pt'
    MODEL_INFO_FILE = 'model.json'
    BACKUP_SUFFIX = '.backup'

    def __init__(self, session_dir: Path):
        """
        Initialisiert den SaveManager.

        Args:
            session_dir: Pfad zum Session-Ordner
        """
        self._logger = get_logger()
        self.session_dir = Path(session_dir)
        self.session_id = self.session_dir.name

        # Datei-Pfade
        self.config_file = self.session_dir / self.CONFIG_FILE
        self.training_data_file = self.session_dir / self.TRAINING_DATA_FILE
        self.backtest_data_file = self.session_dir / self.BACKTEST_DATA_FILE
        self.model_file = self.session_dir / self.MODEL_FILE
        self.model_info_file = self.session_dir / self.MODEL_INFO_FILE

        # Transaktions-State
        self._transaction_active = False
        self._backup_files: List[Path] = []

        self._logger.debug(f"[SaveManager] Initialisiert: {self.session_id}")

    # =========================================================================
    # Pruefungen (VOR dem Speichern aufrufen)
    # =========================================================================

    def check_save_prepared(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        features: List[str]
    ) -> SaveCheckResult:
        """
        Prueft ob Trainingsdaten gespeichert werden koennen.

        Args:
            sequences: Feature-Sequenzen (N, lookback, features)
            labels: Labels (N,)
            features: Liste der Feature-Namen

        Returns:
            SaveCheckResult mit Details was passieren wuerde
        """
        self._logger.debug("[SaveManager] check_save_prepared()")

        result = SaveCheckResult()

        # Was existiert bereits?
        if self.training_data_file.exists():
            result.existing_data.append('training_data.npz')
            result.needs_confirmation = True

            # Vergleiche mit bestehenden Daten
            old_data = self._load_training_data_quick()
            if old_data is not None:
                old_samples = old_data.get('num_samples', 0)
                new_samples = sequences.shape[0]
                if old_samples != new_samples:
                    result.changes.append(
                        f"Samples: {old_samples} -> {new_samples}"
                    )

                old_features = old_data.get('features', [])
                if set(old_features) != set(features):
                    result.changes.append(
                        f"Features: {len(old_features)} -> {len(features)}"
                    )

        if self.backtest_data_file.exists():
            result.existing_data.append('backtest_data.csv')
            result.needs_confirmation = True

        # Warnungen
        if self._has_model():
            result.warnings.append(
                "Achtung: Bestehendes Modell wird ungueltig wenn Daten geaendert werden!"
            )

        return result

    def check_save_trained(
        self,
        model: torch.nn.Module,
        metrics: Dict[str, Any]
    ) -> SaveCheckResult:
        """
        Prueft ob Modell gespeichert werden kann.

        Args:
            model: Das trainierte Modell
            metrics: Training-Metriken

        Returns:
            SaveCheckResult mit Details
        """
        self._logger.debug("[SaveManager] check_save_trained()")

        result = SaveCheckResult()

        # Status pruefen
        config = self.load_config()
        if config is None:
            result.can_save = False
            result.warnings.append("Keine Session-Config gefunden")
            return result

        if config.status == 'created':
            result.can_save = False
            result.warnings.append("Zuerst Trainingsdaten vorbereiten!")
            return result

        # Bestehendes Modell?
        if self._has_model():
            result.existing_data.append('model.pt')
            result.needs_confirmation = True

            old_info = config.model_info or {}
            new_acc = metrics.get('best_accuracy', 0)
            old_acc = old_info.get('accuracy', 0)

            if new_acc < old_acc:
                result.warnings.append(
                    f"Neues Modell ({new_acc:.1f}%) ist schlechter als "
                    f"bestehendes ({old_acc:.1f}%)!"
                )

            result.changes.append(f"Accuracy: {old_acc:.1f}% -> {new_acc:.1f}%")

        return result

    # =========================================================================
    # Speichern
    # =========================================================================

    def save_prepared(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        features: List[str],
        backtest_data: pd.DataFrame,
        params: Dict[str, Any],
        force: bool = False,
        source_file: str = '',
        date_range: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> SaveCheckResult:
        """
        Speichert vorbereitete Session.

        Args:
            sequences: Feature-Sequenzen (N, lookback, features)
            labels: Labels (N,)
            features: Liste der Feature-Namen
            backtest_data: DataFrame mit Backtest-Daten
            params: Vorbereitungs-Parameter
            force: True = Ueberschreiben ohne Nachfrage
            source_file: Quell-Datei der Daten
            date_range: Zeitraum der Daten
            **kwargs: Zusaetzliche Parameter fuer Config

        Returns:
            SaveCheckResult (bei needs_confirmation=True und force=False
            wurde NICHT gespeichert!)
        """
        self._logger.debug(f"[SaveManager] save_prepared() force={force}")

        # 1. Pruefung
        check = self.check_save_prepared(sequences, labels, features)

        # 2. Nachfrage erforderlich?
        if check.needs_confirmation and not force:
            return check  # GUI muss nachfragen!

        # 3. Validierung
        self._validate_training_data(sequences, labels, features)
        self._validate_backtest_data(backtest_data)

        # 4. Session-Ordner sicherstellen
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # 5. Transaktion
        self._begin_transaction()
        try:
            # Trainingsdaten speichern
            self._save_training_data(sequences, labels, features, params)

            # Backtest-Daten speichern
            self._save_backtest_data(backtest_data)

            # Config aktualisieren
            config = self.load_config() or SessionConfig(session_id=self.session_id)
            config.status = 'prepared'
            config.updated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            config.source_file = source_file
            config.date_range = date_range or {}
            config.features = features
            config.num_features = len(features)
            config.num_samples = sequences.shape[0]
            config.lookback = params.get('lookback', 0)
            config.lookforward = params.get('lookforward', 0)
            config.preparation_params = params

            # Zusaetzliche Parameter aus kwargs
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            self._save_config(config)

            # Session-DB aktualisieren
            self._update_session_db(config)

            self._commit_transaction()

            self._logger.info(f"[SaveManager] Session prepared: {self.session_id}")

            check.can_save = True
            check.needs_confirmation = False
            return check

        except Exception as e:
            self._rollback_transaction()
            raise SaveError(
                f"Speichern fehlgeschlagen: {e}",
                operation='save_prepared'
            )

    def save_trained(
        self,
        model: torch.nn.Module,
        metrics: Dict[str, Any],
        optimizer: Optional[torch.optim.Optimizer] = None,
        force: bool = False
    ) -> SaveCheckResult:
        """
        Speichert trainiertes Modell.

        Args:
            model: Das trainierte Modell
            metrics: Training-Metriken (accuracy, loss, etc.)
            optimizer: Optional der Optimizer (fuer Resume)
            force: True = Ueberschreiben ohne Nachfrage

        Returns:
            SaveCheckResult
        """
        self._logger.debug(f"[SaveManager] save_trained() force={force}")

        check = self.check_save_trained(model, metrics)

        if not check.can_save:
            return check

        if check.needs_confirmation and not force:
            return check

        self._begin_transaction()
        try:
            # Modell speichern
            self._save_model(model, metrics, optimizer)

            # Config aktualisieren
            config = self.load_config()
            if config is None:
                raise SaveError("Keine Config gefunden", operation='save_trained')

            config.status = 'trained'
            config.updated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            config.model_info = {
                'accuracy': metrics.get('best_accuracy', 0),
                'loss': metrics.get('best_loss', 0),
                'epochs': metrics.get('epochs', 0),
                'model_type': getattr(model, 'model_type', 'unknown'),
            }
            config.training_metrics = metrics

            self._save_config(config)

            # Session-DB aktualisieren
            self._update_session_db(config)

            self._commit_transaction()

            self._logger.info(
                f"[SaveManager] Modell gespeichert: {self.session_id} "
                f"(Accuracy: {metrics.get('best_accuracy', 0):.1f}%)"
            )

            check.can_save = True
            check.needs_confirmation = False
            return check

        except Exception as e:
            self._rollback_transaction()
            raise SaveError(
                f"Modell-Speicherung fehlgeschlagen: {e}",
                operation='save_trained'
            )

    # =========================================================================
    # Laden
    # =========================================================================

    def load_config(self) -> Optional[SessionConfig]:
        """Laedt die Session-Konfiguration."""
        if not self.config_file.exists():
            return None

        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return SessionConfig.from_dict(data)
        except Exception as e:
            self._logger.error(f"[SaveManager] Config laden fehlgeschlagen: {e}")
            return None

    def load_training_data(self) -> Optional[Dict[str, Any]]:
        """
        Laedt die Trainingsdaten.

        Returns:
            Dictionary mit sequences, labels, features, params
        """
        if not self.training_data_file.exists():
            return None

        try:
            data = np.load(self.training_data_file, allow_pickle=True)
            return {
                'sequences': data['sequences'],
                'labels': data['labels'],
                'features': list(data['features']),
                'params': json.loads(str(data['params']))
            }
        except Exception as e:
            self._logger.error(f"[SaveManager] Training-Daten laden fehlgeschlagen: {e}")
            return None

    def load_backtest_data(self) -> Optional[pd.DataFrame]:
        """Laedt die Backtest-Daten."""
        if not self.backtest_data_file.exists():
            return None

        try:
            df = pd.read_csv(self.backtest_data_file)

            # DateTime als Index setzen
            first_col = df.columns[0]
            if first_col == 'DateTime':
                df = df.set_index('DateTime')
                df.index = pd.to_datetime(df.index)
            elif 'DateTime' in df.columns:
                if first_col.startswith('Unnamed'):
                    df = df.drop(columns=[first_col])
                df = df.set_index('DateTime')
                df.index = pd.to_datetime(df.index)

            df.index.name = 'DateTime'
            if 'DateTime' not in df.columns:
                df['DateTime'] = df.index

            return df
        except Exception as e:
            self._logger.error(f"[SaveManager] Backtest-Daten laden fehlgeschlagen: {e}")
            return None

    # =========================================================================
    # Private Hilfsmethoden
    # =========================================================================

    def _has_model(self) -> bool:
        """Prueft ob ein Modell existiert."""
        return self.model_file.exists() or any(self.session_dir.glob('*.pt'))

    def _load_training_data_quick(self) -> Optional[Dict[str, Any]]:
        """Laedt Trainingsdaten-Metadaten schnell (ohne volle Daten)."""
        if not self.training_data_file.exists():
            return None

        try:
            with np.load(self.training_data_file, allow_pickle=True) as data:
                return {
                    'num_samples': data['sequences'].shape[0],
                    'features': list(data['features'])
                }
        except Exception:
            return None

    def _validate_training_data(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        features: List[str]
    ):
        """Validiert Trainingsdaten vor dem Speichern."""
        if sequences is None or len(sequences) == 0:
            raise SaveError("Keine Sequenzen vorhanden", operation='validate')

        if labels is None or len(labels) == 0:
            raise SaveError("Keine Labels vorhanden", operation='validate')

        if len(sequences) != len(labels):
            raise SaveError(
                f"Sequenzen ({len(sequences)}) und Labels ({len(labels)}) "
                f"haben unterschiedliche Laenge",
                operation='validate'
            )

        if not features:
            raise SaveError("Keine Features angegeben", operation='validate')

        if len(sequences.shape) != 3:
            raise SaveError(
                f"Sequenzen muessen 3D sein (N, lookback, features), "
                f"aber sind {sequences.shape}",
                operation='validate'
            )

        if sequences.shape[2] != len(features):
            raise SaveError(
                f"Feature-Anzahl stimmt nicht: "
                f"{sequences.shape[2]} vs {len(features)}",
                operation='validate'
            )

    def _validate_backtest_data(self, data: pd.DataFrame):
        """Validiert Backtest-Daten vor dem Speichern."""
        if data is None or len(data) == 0:
            raise SaveError("Keine Backtest-Daten vorhanden", operation='validate')

        required_cols = {'Open', 'High', 'Low', 'Close'}
        missing = required_cols - set(data.columns)
        if missing:
            raise SaveError(
                f"Fehlende Spalten in Backtest-Daten: {missing}",
                operation='validate'
            )

    def _save_training_data(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        features: List[str],
        params: Dict[str, Any]
    ):
        """Speichert Trainingsdaten."""
        self._logger.debug(
            f"[SaveManager] Speichere Training-Daten: "
            f"{sequences.shape[0]} Samples, {len(features)} Features"
        )

        np.savez_compressed(
            self.training_data_file,
            sequences=sequences,
            labels=labels,
            features=np.array(features, dtype=object),
            params=json.dumps(params)
        )

    def _save_backtest_data(self, data: pd.DataFrame):
        """Speichert Backtest-Daten."""
        self._logger.debug(f"[SaveManager] Speichere Backtest-Daten: {len(data)} Zeilen")

        data = data.copy()

        # DateTime als Index sicherstellen
        if 'DateTime' in data.columns:
            data = data.set_index('DateTime')

        if isinstance(data.index, pd.DatetimeIndex):
            data.index.name = 'DateTime'

        data.to_csv(self.backtest_data_file, index=True)

    def _save_model(
        self,
        model: torch.nn.Module,
        metrics: Dict[str, Any],
        optimizer: Optional[torch.optim.Optimizer] = None
    ):
        """Speichert Modell."""
        self._logger.debug(f"[SaveManager] Speichere Modell nach {self.model_file}")

        # Modell-Info sammeln
        model_info = {
            'model_type': getattr(model, 'model_type', 'unknown'),
            'input_size': getattr(model, 'input_size', 0),
            'hidden_sizes': getattr(model, 'hidden_sizes', []),
            'num_layers': getattr(model, 'num_layers', 0),
            'dropout': getattr(model, 'dropout', 0),
            'num_classes': getattr(model, 'num_classes', 3),
            'best_accuracy': metrics.get('best_accuracy', 0),
            'best_loss': metrics.get('best_loss', 0),
            'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        # Checkpoint erstellen
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_info': model_info,
            'metrics': metrics,
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        # Speichern
        torch.save(checkpoint, self.model_file)

        # JSON-Metadaten separat speichern
        with open(self.model_info_file, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)

    def _save_config(self, config: SessionConfig):
        """Speichert Config."""
        self._logger.debug(f"[SaveManager] Speichere Config: status={config.status}")

        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config.to_dict(), f, indent=2, ensure_ascii=False, default=str)

    # =========================================================================
    # Transaktions-Management
    # =========================================================================

    def _begin_transaction(self):
        """Startet eine Transaktion mit Backups."""
        self._logger.debug("[SaveManager] Begin Transaction")
        self._transaction_active = True
        self._backup_files = []

        # Backups erstellen fuer existierende Dateien
        for file_path in [
            self.config_file,
            self.training_data_file,
            self.backtest_data_file,
            self.model_file,
            self.model_info_file
        ]:
            if file_path.exists():
                backup_path = file_path.with_suffix(
                    file_path.suffix + self.BACKUP_SUFFIX
                )
                shutil.copy2(file_path, backup_path)
                self._backup_files.append(backup_path)
                self._logger.debug(f"[SaveManager] Backup: {file_path.name}")

    def _commit_transaction(self):
        """Bestaetigt Transaktion und loescht Backups."""
        self._logger.debug("[SaveManager] Commit Transaction")

        # Backups loeschen
        for backup_path in self._backup_files:
            if backup_path.exists():
                backup_path.unlink()

        self._backup_files = []
        self._transaction_active = False

    def _rollback_transaction(self):
        """Macht Transaktion rueckgaengig."""
        self._logger.warning("[SaveManager] Rollback Transaction")

        # Backups wiederherstellen
        for backup_path in self._backup_files:
            if backup_path.exists():
                original_path = backup_path.with_suffix('')
                # Suffix entfernen (.json.backup -> .json)
                suffix = backup_path.suffix
                if suffix == self.BACKUP_SUFFIX:
                    stem = backup_path.stem  # z.B. "session_config.json"
                    original_path = backup_path.parent / stem

                shutil.move(str(backup_path), str(original_path))
                self._logger.debug(f"[SaveManager] Restored: {original_path.name}")

        self._backup_files = []
        self._transaction_active = False

    # =========================================================================
    # Session-Datenbank Integration
    # =========================================================================

    def _get_session_db(self):
        """Gibt die SessionDatabase zurueck (lazy loading)."""
        try:
            from btcusd_analyzer.core.session_database import SessionDatabase
            from btcusd_analyzer.core.config import Config

            # sessions-Verzeichnis aus Config holen (nicht aus parent ableiten!)
            # Das funktioniert auch wenn session_dir in log/ liegt
            config = Config()
            sessions_dir = config.paths.get_sessions_root()
            return SessionDatabase(sessions_dir)
        except Exception as e:
            self._logger.debug(f"[SaveManager] SessionDatabase nicht verfuegbar: {e}")
            return None

    def _update_session_db(self, config: SessionConfig):
        """Aktualisiert die Session in der Datenbank."""
        db = self._get_session_db()
        if db is None:
            return

        session_info = {
            'id': self.session_id,
            'path': str(self.session_dir),
            'status': config.status,
            'has_training_data': self.training_data_file.exists(),
            'has_backtest_data': self.backtest_data_file.exists(),
            'has_model': self._has_model(),
            'features': config.features,
            'num_features': config.num_features,
            'num_samples': config.num_samples,
        }

        if config.model_info:
            session_info['model_accuracy'] = config.model_info.get('accuracy', 0)
            session_info['model_type'] = config.model_info.get('model_type', '-')

        try:
            existing = db.get_session(self.session_id)
            if existing:
                db.update_session(self.session_id, session_info)
            else:
                db.add_session(session_info)
        except Exception as e:
            self._logger.warning(f"[SaveManager] DB-Update fehlgeschlagen: {e}")

    # =========================================================================
    # Factory-Methoden
    # =========================================================================

    @staticmethod
    def create_session(sessions_root: Path, timestamp: Optional[str] = None) -> 'SaveManager':
        """
        Erstellt eine neue Session.

        Args:
            sessions_root: Root-Ordner fuer Sessions (z.B. 'sessions/')
            timestamp: Optional custom Timestamp

        Returns:
            SaveManager fuer die neue Session
        """
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')

        session_id = f"session-{timestamp}"
        session_dir = Path(sessions_root) / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        manager = SaveManager(session_dir)

        # Initiale Config erstellen
        config = SessionConfig(
            session_id=session_id,
            status='created',
            created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            updated_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        )
        manager._save_config(config)

        return manager

    @staticmethod
    def from_session_dir(session_dir: Path) -> 'SaveManager':
        """
        Oeffnet eine bestehende Session.

        Args:
            session_dir: Pfad zum Session-Ordner

        Returns:
            SaveManager fuer die Session
        """
        session_dir = Path(session_dir)
        if not session_dir.exists():
            raise SaveError(f"Session-Ordner nicht gefunden: {session_dir}")

        return SaveManager(session_dir)
