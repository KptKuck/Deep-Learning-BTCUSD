"""
Logger Modul - Einheitliches Logging-System
Entspricht dem MATLAB Logger mit farbiger Konsolenausgabe und Datei-Logging

Thread-Safe: Verwendet QueueHandler fuer multi-threaded Anwendungen (QThread + Main)
"""

import logging
import logging.handlers
import queue
import threading
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

    # Farben fuer Konsolen-Ausgabe (colorlog)
    COLORS = {
        'TRACE': 'cyan',
        'DEBUG': 'light_black',  # Grau - sichtbar auf dunklem Hintergrund
        'INFO': 'blue',
        'SUCCESS': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }

    _instances: dict = {}
    _lock = threading.Lock()  # Thread-safe Singleton

    def __new__(cls, name: str = 'btcusd_analyzer', log_dir: Optional[str] = None):
        """Singleton-Pattern pro Logger-Name (thread-safe)"""
        with cls._lock:
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

        # Queue fuer thread-safe Logging
        self._log_queue: queue.Queue = queue.Queue(-1)  # Unbegrenzt
        self._queue_handler = None
        self._console_handler = None
        self._file_handler = None

        # Handler Setup
        self._setup_handlers()

        self._initialized = True

    def _setup_handlers(self):
        """
        Richtet alle Handler ein.

        Architektur (100% Queue-basiert zur Vermeidung von Deadlocks):
        - Alle Log-Aufrufe gehen nur in eine Queue (lock-free im aufrufenden Thread)
        - Ein separater Writer-Thread liest aus der Queue und schreibt zu:
          * Konsole (stdout)
          * Datei
        """
        # Log-Verzeichnis erstellen
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Log-Datei mit Zeitstempel
        timestamp = datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')
        self.log_file = self.log_dir / f'session-{timestamp}.txt'

        # === Nur QueueHandler am Logger (lock-free) ===
        self._queue_handler = logging.handlers.QueueHandler(self._log_queue)
        self._logger.addHandler(self._queue_handler)

        # === Handler fuer den Writer-Thread (werden dort verwendet) ===

        # Konsolen-Handler mit Farben (wird im Writer-Thread verwendet)
        self._console_handler = logging.StreamHandler()
        self._console_handler.setLevel(self.TRACE)

        if COLORLOG_AVAILABLE:
            # Farbiger Formatter fuer Konsole
            color_formatter = colorlog.ColoredFormatter(
                '%(log_color)s[%(asctime)s] [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                log_colors=self.COLORS
            )
            self._console_handler.setFormatter(color_formatter)
        else:
            # Fallback ohne Farben
            console_formatter = logging.Formatter(
                '[%(asctime)s] [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            self._console_handler.setFormatter(console_formatter)

        # Datei-Formatter (immer ohne Farben)
        log_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Datei-Handler (wird im Writer-Thread verwendet)
        self._file_handler = logging.FileHandler(
            self.log_file, encoding='utf-8', delay=False
        )
        self._file_handler.setLevel(self.TRACE)
        self._file_handler.setFormatter(log_formatter)

        # Writer-Thread fuer Queue -> Console + File
        self._start_file_writer_thread()

    def _start_file_writer_thread(self):
        """
        Startet einen Daemon-Thread der die Queue liest und zu Console + Datei schreibt.

        Dieser Ansatz vermeidet Deadlocks da:
        - Log-Aufrufe nur in die Queue schreiben (praktisch lock-free)
        - Nur dieser eine Thread tatsaechlich I/O macht
        - Kein Lock zwischen GUI-Thread und Worker-Thread
        """
        def writer_loop():
            while True:
                try:
                    record = self._log_queue.get(timeout=0.5)
                    if record is None:
                        break
                    # Beide Handler bedienen
                    try:
                        self._console_handler.emit(record)
                    except Exception:
                        pass  # Konsolen-Fehler ignorieren
                    try:
                        self._file_handler.emit(record)
                    except Exception:
                        pass  # Datei-Fehler ignorieren
                except queue.Empty:
                    continue
                except Exception:
                    pass  # Alle anderen Fehler ignorieren

        self._writer_thread = threading.Thread(target=writer_loop, daemon=True, name="LogWriter")
        self._writer_thread.start()

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
