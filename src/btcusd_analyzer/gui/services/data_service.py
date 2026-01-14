"""
DataService - Zentraler Service fuer Daten-Operationen

Extrahiert aus main_window.py fuer bessere Wartbarkeit.
"""

from pathlib import Path
from typing import Optional, Tuple, Callable
from datetime import date

import pandas as pd

from ...core.logger import get_logger
from ...data.reader import CSVReader


class DataService:
    """
    Service fuer alle Daten-bezogenen Operationen.

    Funktionen:
    - CSV laden (einzeln und automatisch letzte Datei)
    - Binance Download
    - Datenvalidierung
    - Datenanalyse

    Attributes:
        data: Aktuell geladener DataFrame
        data_path: Pfad zur geladenen Datei
        data_dir: Verzeichnis fuer Daten
    """

    def __init__(self, data_dir: Path, log_callback: Optional[Callable] = None):
        """
        Initialisiert den DataService.

        Args:
            data_dir: Verzeichnis fuer Daten
            log_callback: Optionale Callback-Funktion fuer Logging
        """
        self.data_dir = Path(data_dir)
        self.logger = get_logger()
        self._log_callback = log_callback

        # State
        self.data: Optional[pd.DataFrame] = None
        self.data_path: Optional[Path] = None

    def _log(self, message: str, level: str = 'INFO'):
        """Internes Logging."""
        if self._log_callback:
            self._log_callback(message, level)
        else:
            log_method = getattr(self.logger, level.lower(), self.logger.info)
            log_method(message)

    def load_csv(self, filepath: Path) -> Optional[pd.DataFrame]:
        """
        Laedt eine CSV-Datei.

        Args:
            filepath: Pfad zur CSV-Datei

        Returns:
            DataFrame oder None bei Fehler
        """
        try:
            reader = CSVReader()
            df = reader.read(filepath)

            if df is not None:
                self.data = df
                self.data_path = filepath
                reader.log_data_info(df, filepath)
                self._log(f'CSV geladen: {filepath.name}', 'SUCCESS')
                return df
            else:
                self._log(f'CSV konnte nicht geladen werden: {filepath}', 'ERROR')
                return None

        except Exception as e:
            self._log(f'CSV-Fehler: {e}', 'ERROR')
            return None

    def load_last_csv(self, pattern: str = 'BTCUSD*.csv') -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
        """
        Laedt die neueste CSV-Datei.

        Args:
            pattern: Glob-Pattern fuer Dateisuche

        Returns:
            Tuple aus (DataFrame, Pfad) oder (None, None)
        """
        try:
            reader = CSVReader(self.data_dir)
            df, filepath = reader.load_last_csv(pattern)

            if df is not None:
                self.data = df
                self.data_path = filepath
                self._log(f'Letzte CSV geladen: {filepath.name}', 'SUCCESS')

            return df, filepath

        except Exception as e:
            self._log(f'Auto-Load fehlgeschlagen: {e}', 'WARNING')
            return None, None

    def download_binance(
        self,
        symbol: str,
        from_date: date,
        to_date: date,
        interval: str = '1h',
        save: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Laedt Daten von Binance herunter.

        Args:
            symbol: Trading-Symbol (z.B. 'BTCUSDT')
            from_date: Startdatum
            to_date: Enddatum
            interval: Kerzendauer ('1h', '4h', '1d', etc.)
            save: Ob Daten gespeichert werden sollen

        Returns:
            DataFrame oder None bei Fehler
        """
        from_date_str = from_date.strftime('%Y-%m-%d')
        to_date_str = to_date.strftime('%Y-%m-%d')

        self._log(f'Download gestartet: {symbol} {from_date_str} bis {to_date_str}', 'INFO')

        try:
            from ...data.downloader import BinanceDownloader

            downloader = BinanceDownloader(
                symbol=symbol,
                data_dir=self.data_dir
            )

            df = downloader.download(
                start_date=from_date_str,
                end_date=to_date_str,
                interval=interval,
                save=save
            )

            if df is not None:
                self.data = df
                # Dateipfad rekonstruieren
                symbol_short = symbol.replace('USDT', 'USD')
                filename = f'{symbol_short}_{from_date_str}_{to_date_str}.csv'
                self.data_path = self.data_dir / filename
                self._log(f'Download erfolgreich: {len(df)} Datensaetze', 'SUCCESS')
                return df
            else:
                self._log('Download fehlgeschlagen - keine Daten erhalten', 'ERROR')
                return None

        except ImportError as e:
            self._log(f'Modul nicht installiert: {e}', 'ERROR')
            self._log('Hinweis: pip install python-binance', 'WARNING')
            raise
        except Exception as e:
            self._log(f'Download-Fehler: {e}', 'ERROR')
            raise

    def get_data_info(self) -> dict:
        """
        Gibt Informationen ueber die geladenen Daten zurueck.

        Returns:
            Dictionary mit Daten-Informationen
        """
        if self.data is None or self.data.empty:
            return {}

        reader = CSVReader()
        return reader.get_data_info(self.data)

    def validate_data(self, strict: bool = False) -> list:
        """
        Validiert die geladenen Daten.

        Args:
            strict: Wenn True, werden auch Warnungen als Fehler gewertet

        Returns:
            Liste von Validierungsfehlern
        """
        if self.data is None:
            return ['Keine Daten geladen']

        reader = CSVReader()
        return reader.validate(self.data, strict=strict)

    def clear(self):
        """Setzt den Service zurueck."""
        self.data = None
        self.data_path = None
