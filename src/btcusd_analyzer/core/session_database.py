"""
SessionDatabase - Zentrale Datenbank fuer alle Sessions.

Speichert Session-Metadaten in einer JSON-Datei fuer schnellen Zugriff,
ohne dass die einzelnen Session-Ordner gescannt werden muessen.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from .logger import get_logger


class SessionDatabase:
    """
    Zentrale Datenbank fuer alle Sessions.

    Speichert Metadaten wie Status, Accuracy, Features in einer JSON-Datei.
    Ermoeglicht schnelles Auflisten und Filtern von Sessions.
    """

    SCHEMA_VERSION = "1.0"
    DB_FILENAME = "sessions.json"

    def __init__(self, data_dir: Path):
        """
        Initialisiert die SessionDatabase.

        Args:
            data_dir: Pfad zum data-Verzeichnis (dort wird sessions.json gespeichert)
        """
        self._logger = get_logger()
        self.data_dir = Path(data_dir)
        self.db_path = self.data_dir / self.DB_FILENAME

        # Verzeichnis erstellen falls noetig
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._ensure_db_exists()
        self._logger.debug(f"[SessionDB] Initialisiert: {self.db_path}")

    def _ensure_db_exists(self):
        """Erstellt die DB-Datei falls nicht vorhanden."""
        if not self.db_path.exists():
            self._save({
                "schema_version": self.SCHEMA_VERSION,
                "sessions": []
            })
            self._logger.debug(f"[SessionDB] Neue DB erstellt: {self.db_path}")

    def _load(self) -> dict:
        """Laedt die DB von Disk."""
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self._logger.warning(f"[SessionDB] DB-Ladefehler: {e}, erstelle neue DB")
            return {"schema_version": self.SCHEMA_VERSION, "sessions": []}

    def _save(self, data: dict):
        """Speichert die DB auf Disk."""
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def _session_exists(self, session_id: str) -> bool:
        """Prueft ob eine Session bereits in der DB existiert."""
        data = self._load()
        return any(s.get('id') == session_id for s in data['sessions'])

    def add_session(self, session_info: dict) -> str:
        """
        Fuegt eine neue Session zur DB hinzu.

        Args:
            session_info: Dictionary mit Session-Metadaten
                - id: Session-ID (optional, wird aus path generiert)
                - path: Pfad zum Session-Ordner
                - status: 'prepared' oder 'trained'
                - features: Liste der Features
                - num_samples: Anzahl der Samples
                - etc.

        Returns:
            Session-ID
        """
        data = self._load()

        # Session-ID generieren falls nicht vorhanden
        session_id = session_info.get('id')
        if not session_id and 'path' in session_info:
            session_id = Path(session_info['path']).name
        if not session_id:
            session_id = f"session-{datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')}"

        # Pruefe ob bereits vorhanden
        if self._session_exists(session_id):
            self._logger.debug(f"[SessionDB] Session existiert bereits: {session_id}")
            self.update_session(session_id, session_info)
            return session_id

        # Neue Session hinzufuegen
        session_info['id'] = session_id
        session_info['created_at'] = datetime.now().isoformat()
        session_info['updated_at'] = session_info['created_at']

        data['sessions'].append(session_info)
        self._save(data)

        self._logger.debug(f"[SessionDB] Session hinzugefuegt: {session_id}")
        return session_id

    def update_session(self, session_id: str, updates: dict):
        """
        Aktualisiert eine bestehende Session.

        Args:
            session_id: Session-ID
            updates: Dictionary mit zu aktualisierenden Feldern
        """
        data = self._load()

        for session in data['sessions']:
            if session.get('id') == session_id:
                # Alte Werte erhalten, neue ueberschreiben
                session.update(updates)
                session['updated_at'] = datetime.now().isoformat()
                self._save(data)
                self._logger.debug(f"[SessionDB] Session aktualisiert: {session_id}")
                return

        # Session nicht gefunden -> neue hinzufuegen
        self._logger.debug(f"[SessionDB] Session nicht gefunden, fuege hinzu: {session_id}")
        updates['id'] = session_id
        self.add_session(updates)

    def get_session(self, session_id: str) -> Optional[dict]:
        """
        Holt Session-Info aus der DB.

        Args:
            session_id: Session-ID

        Returns:
            Session-Dictionary oder None
        """
        data = self._load()
        for session in data['sessions']:
            if session.get('id') == session_id:
                return session
        return None

    def list_sessions(self, status: str = None) -> List[dict]:
        """
        Listet alle Sessions, optional gefiltert nach Status.

        Args:
            status: Optional - 'prepared', 'trained' oder None fuer alle

        Returns:
            Liste von Session-Dictionaries, neueste zuerst
        """
        data = self._load()
        sessions = data.get('sessions', [])

        if status:
            sessions = [s for s in sessions if s.get('status') == status]

        # Neueste zuerst (nach created_at sortieren)
        sessions = sorted(
            sessions,
            key=lambda x: x.get('created_at', ''),
            reverse=True
        )

        self._logger.debug(f"[SessionDB] list_sessions(status={status}): {len(sessions)} Sessions")
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """
        Loescht eine Session aus der DB.

        Args:
            session_id: Session-ID

        Returns:
            True wenn geloescht, False wenn nicht gefunden
        """
        data = self._load()
        original_count = len(data['sessions'])

        data['sessions'] = [s for s in data['sessions'] if s.get('id') != session_id]

        if len(data['sessions']) < original_count:
            self._save(data)
            self._logger.debug(f"[SessionDB] Session geloescht: {session_id}")
            return True

        return False

    def migrate_from_folders(self, log_dir: Path) -> int:
        """
        Migriert bestehende Session-Ordner in die DB.

        Scannt das Log-Verzeichnis nach Session-Ordnern und fuegt
        diese zur DB hinzu, falls noch nicht vorhanden.

        Args:
            log_dir: Pfad zum Log-Verzeichnis

        Returns:
            Anzahl der migrierten Sessions
        """
        from .session_manager import SessionManager

        log_dir = Path(log_dir)
        if not log_dir.exists():
            self._logger.debug(f"[SessionDB] Log-Dir existiert nicht: {log_dir}")
            return 0

        migrated_count = 0

        self._logger.info(f"[SessionDB] Migriere Sessions aus: {log_dir}")

        for item in sorted(log_dir.iterdir(), reverse=True):
            if item.is_dir() and item.name.startswith('session-'):
                # Pruefe ob bereits in DB
                if self._session_exists(item.name):
                    self._logger.debug(f"[SessionDB] Ueberspringe (bereits in DB): {item.name}")
                    continue

                # Session-Info sammeln
                try:
                    manager = SessionManager(item)
                    summary = manager.get_summary()

                    # Nur Sessions mit Daten migrieren
                    if not (summary.get('has_training_data') or summary.get('has_model')):
                        continue

                    # Config laden fuer zusaetzliche Infos
                    config = manager.load_config() or {}

                    session_info = {
                        'id': item.name,
                        'path': str(item),
                        'status': summary.get('status') or ('trained' if summary.get('has_model') else 'prepared'),
                        'model_version': 'legacy',
                        'model_accuracy': summary.get('model_accuracy', 0),
                        'has_training_data': summary.get('has_training_data', False),
                        'has_backtest_data': summary.get('has_backtest_data', False),
                        'has_model': summary.get('has_model', False),
                        'features': config.get('features', []),
                        'num_features': len(config.get('features', [])),
                    }

                    # Training-Info hinzufuegen falls vorhanden
                    if 'training_info' in config:
                        session_info['num_samples'] = config['training_info'].get('actual_samples', 0)

                    self.add_session(session_info)
                    migrated_count += 1
                    self._logger.debug(f"[SessionDB] Migriert: {item.name}")

                except Exception as e:
                    self._logger.warning(f"[SessionDB] Migration fehlgeschlagen fuer {item.name}: {e}")

        self._logger.info(f"[SessionDB] Migration abgeschlossen: {migrated_count} Sessions")
        return migrated_count

    def get_statistics(self) -> dict:
        """
        Gibt Statistiken ueber alle Sessions zurueck.

        Returns:
            Dictionary mit Statistiken
        """
        data = self._load()
        sessions = data.get('sessions', [])

        trained = [s for s in sessions if s.get('status') == 'trained']
        prepared = [s for s in sessions if s.get('status') == 'prepared']

        accuracies = [s.get('model_accuracy', 0) for s in trained if s.get('model_accuracy', 0) > 0]

        return {
            'total_sessions': len(sessions),
            'trained_sessions': len(trained),
            'prepared_sessions': len(prepared),
            'avg_accuracy': sum(accuracies) / len(accuracies) if accuracies else 0,
            'max_accuracy': max(accuracies) if accuracies else 0,
            'schema_version': data.get('schema_version', 'unknown'),
        }
