"""
Logger Modul - Einheitliches Logging-System
Entspricht dem MATLAB Logger mit farbiger Konsolenausgabe und Datei-Logging
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import colorlog
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False


class Logger:
    """
    Einheitliches Logging-System mit Konsolen- und Datei-Ausgabe.

    Unterstuetzte Log-Level:
    - TRACE: Sehr detaillierte Debug-Informationen
    - DEBUG: Debug-Informationen
    - INFO: Allgemeine Informationen
    - SUCCESS: Erfolgreiche Operationen
    - WARNING: Warnungen
    - ERROR: Fehler

    Attributes:
        name: Name des Loggers
        log_dir: Verzeichnis fuer Log-Dateien
        log_file: Pfad zur aktuellen Log-Datei
    """

    # Custom Log-Level
    TRACE = 5
    SUCCESS = 25

    # Farben fuer Konsolen-Ausgabe (ANSI)
    COLORS = {
        'TRACE': 'cyan',
        'DEBUG': 'white',
        'INFO': 'blue',
        'SUCCESS': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }

    _instances: dict = {}

    def __new__(cls, name: str = 'btcusd_analyzer', log_dir: Optional[str] = None):
        """Singleton-Pattern pro Logger-Name"""
        if name not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[name] = instance
        return cls._instances[name]

    def __init__(self, name: str = 'btcusd_analyzer', log_dir: Optional[str] = None):
        """
        Initialisiert den Logger.

        Args:
            name: Name des Loggers (z.B. 'btcusd_analyzer')
            log_dir: Verzeichnis fuer Log-Dateien (default: <projekt>/log)
        """
        if hasattr(self, '_initialized'):
            return

        self.name = name

        # Log-Verzeichnis: Entweder explizit angegeben oder im Projektverzeichnis
        if log_dir:
            self.log_dir = Path(log_dir)
        else:
            # Projektverzeichnis ermitteln (3 Ebenen hoch von dieser Datei)
            # __file__ -> core/logger.py -> core -> btcusd_analyzer -> src -> btcusd_analyzer_python
            project_dir = Path(__file__).parent.parent.parent.parent
            self.log_dir = project_dir / 'log'

        self.log_file: Optional[Path] = None

        # Custom Log-Level registrieren
        logging.addLevelName(self.TRACE, 'TRACE')
        logging.addLevelName(self.SUCCESS, 'SUCCESS')

        # Logger erstellen
        self._logger = logging.getLogger(name)
        self._logger.setLevel(self.TRACE)
        self._logger.handlers = []  # Vorherige Handler entfernen

        # Konsolen-Handler
        self._setup_console_handler()

        # Datei-Handler
        self._setup_file_handler()

        self._initialized = True

    def _setup_console_handler(self):
        """Richtet den Konsolen-Handler mit Farbausgabe ein."""
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.TRACE)

        if COLORLOG_AVAILABLE:
            formatter = colorlog.ColoredFormatter(
                '%(log_color)s[%(asctime)s] [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                log_colors=self.COLORS
            )
        else:
            formatter = logging.Formatter(
                '[%(asctime)s] [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)

    def _setup_file_handler(self):
        """Richtet den Datei-Handler ein."""
        # Log-Verzeichnis erstellen
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Log-Datei mit Zeitstempel
        # Doppelpunkt (:) ist unter Windows nicht erlaubt in Dateinamen
        timestamp = datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')
        self.log_file = self.log_dir / f'session-{timestamp}.txt'

        # delay=False fuer sofortiges Oeffnen
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8', delay=False)
        file_handler.setLevel(self.TRACE)

        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        self._logger.addHandler(file_handler)

    def trace(self, message: str):
        """Sehr detaillierte Debug-Informationen."""
        self._logger.log(self.TRACE, message)

    def debug(self, message: str):
        """Debug-Informationen."""
        self._logger.debug(message)

    def info(self, message: str):
        """Allgemeine Informationen."""
        self._logger.info(message)

    def success(self, message: str):
        """Erfolgreiche Operationen."""
        self._logger.log(self.SUCCESS, message)

    def warning(self, message: str):
        """Warnungen."""
        self._logger.warning(message)

    def error(self, message: str):
        """Fehler."""
        self._logger.error(message)

    def critical(self, message: str):
        """Kritische Fehler."""
        self._logger.critical(message)

    def timing(self, operation: str, duration_ms: float):
        """Logging von Timing-Informationen."""
        self.trace(f'[TIMING] {operation}: {duration_ms:.1f} ms')

    def set_level(self, level: str):
        """
        Setzt das Log-Level.

        Args:
            level: 'TRACE', 'DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR'
        """
        level_map = {
            'TRACE': self.TRACE,
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'SUCCESS': self.SUCCESS,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
        }
        if level.upper() in level_map:
            self._logger.setLevel(level_map[level.upper()])

    def get_log_file_path(self) -> Optional[str]:
        """Gibt den Pfad zur aktuellen Log-Datei zurueck."""
        return str(self.log_file) if self.log_file else None

    def get_session_dir(self) -> Optional[Path]:
        """
        Gibt den Session-Ordner zurueck (erstellt ihn falls noetig).

        Der Session-Ordner hat den gleichen Namen wie die Log-Datei (ohne .txt).
        Beispiel: log/session-2026-01-14_20h18m11s/
        """
        if self.log_file is None:
            return None

        # Session-Ordner = Log-Datei ohne .txt Endung
        session_dir = self.log_file.with_suffix('')
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def get_session_name(self) -> Optional[str]:
        """Gibt den Session-Namen zurueck (z.B. 'session-2026-01-14_20h18m11s')."""
        if self.log_file is None:
            return None
        return self.log_file.stem


# Globale Logger-Instanz
_default_logger: Optional[Logger] = None


def get_logger(name: str = 'btcusd_analyzer', log_dir: Optional[str] = None) -> Logger:
    """
    Gibt eine Logger-Instanz zurueck.

    Args:
        name: Name des Loggers
        log_dir: Verzeichnis fuer Log-Dateien

    Returns:
        Logger-Instanz
    """
    global _default_logger

    if _default_logger is None or _default_logger.name != name:
        _default_logger = Logger(name, log_dir)

    return _default_logger


# Convenience-Funktionen fuer direkten Zugriff
def trace(message: str):
    """Globale trace() Funktion."""
    get_logger().trace(message)


def debug(message: str):
    """Globale debug() Funktion."""
    get_logger().debug(message)


def info(message: str):
    """Globale info() Funktion."""
    get_logger().info(message)


def success(message: str):
    """Globale success() Funktion."""
    get_logger().success(message)


def warning(message: str):
    """Globale warning() Funktion."""
    get_logger().warning(message)


def error(message: str):
    """Globale error() Funktion."""
    get_logger().error(message)
