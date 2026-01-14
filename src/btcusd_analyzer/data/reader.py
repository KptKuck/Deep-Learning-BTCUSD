"""
CSV Reader Modul - Laedt BTC-Daten aus CSV-Dateien
Entspricht read_btc_data.m aus dem MATLAB-Projekt
"""

import time
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np

from ..core.logger import get_logger
from ..core.exceptions import (
    DataValidationError,
    DataFormatError,
    MissingDataError,
)


class CSVReader:
    """
    Liest BTC-Kursdaten aus CSV-Dateien.

    Unterstuetzte Formate:
    - MetaTrader Export (DateTime, Date, Time, Open, High, Low, Close, TickVol, Vol, Spread)
    - Standard OHLCV (Date, Open, High, Low, Close, Volume)

    Attributes:
        data_dir: Verzeichnis mit CSV-Dateien
        logger: Logger-Instanz
    """

    # Erwartete Spalten
    REQUIRED_COLUMNS = ['Open', 'High', 'Low', 'Close']
    OPTIONAL_COLUMNS = ['Volume', 'TickVol', 'Vol', 'Spread']

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialisiert den CSV Reader.

        Args:
            data_dir: Verzeichnis mit CSV-Dateien (default: <projekt>/data)
        """
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            # Projektverzeichnis ermitteln (3 Ebenen hoch von dieser Datei)
            # __file__ -> data/reader.py -> data -> btcusd_analyzer -> src -> btcusd_analyzer_python
            project_dir = Path(__file__).parent.parent.parent.parent
            self.data_dir = project_dir / 'data'

        self.logger = get_logger()

    def read(self, filepath: Path) -> Optional[pd.DataFrame]:
        """
        Liest eine CSV-Datei und gibt einen DataFrame zurueck.

        Args:
            filepath: Pfad zur CSV-Datei

        Returns:
            DataFrame mit OHLCV-Daten oder None bei Fehler
        """
        start_time = time.perf_counter()
        filepath = Path(filepath)

        if not filepath.exists():
            self.logger.error(f'Datei nicht gefunden: {filepath}')
            return None

        try:
            # CSV einlesen
            df = pd.read_csv(filepath)

            # DateTime-Spalte erstellen/verarbeiten
            df = self._process_datetime(df)

            # Spalten validieren
            if not self._validate_columns(df):
                return None

            # Datentypen konvertieren
            for col in self.REQUIRED_COLUMNS:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # NaN-Werte entfernen
            df = df.dropna(subset=self.REQUIRED_COLUMNS)

            # Nach DateTime sortieren
            df = df.sort_values('DateTime').reset_index(drop=True)

            duration_ms = (time.perf_counter() - start_time) * 1000
            self.logger.timing('read_btc_data', duration_ms)

            return df

        except Exception as e:
            self.logger.error(f'Fehler beim Lesen der CSV: {e}')
            return None

    def _process_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Verarbeitet DateTime-Spalten."""
        if 'DateTime' in df.columns:
            # MetaTrader Format
            df['DateTime'] = pd.to_datetime(df['DateTime'])
        elif 'Date' in df.columns and 'Time' in df.columns:
            # Separate Date und Time Spalten
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        elif 'Date' in df.columns:
            # Nur Date vorhanden
            df['DateTime'] = pd.to_datetime(df['Date'])
        elif 'Timestamp' in df.columns:
            # Unix Timestamp
            df['DateTime'] = pd.to_datetime(df['Timestamp'], unit='s')
        else:
            self.logger.warning('Keine DateTime-Spalte gefunden, erstelle Index')
            df['DateTime'] = pd.date_range(start='2020-01-01', periods=len(df), freq='h')

        return df

    def _validate_columns(self, df: pd.DataFrame) -> bool:
        """Validiert dass alle erforderlichen Spalten vorhanden sind."""
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            self.logger.error(f'Fehlende Spalten: {missing}')
            return False
        return True

    def validate(self, df: pd.DataFrame, strict: bool = False) -> List[str]:
        """
        Fuehrt umfassende Validierung der Daten durch.

        Args:
            df: DataFrame mit OHLCV-Daten
            strict: Wenn True, werden auch Warnungen als Fehler gewertet

        Returns:
            Liste von Validierungsfehlern (leer wenn alles OK)

        Raises:
            DataValidationError: Bei kritischen Fehlern (wenn strict=True)
        """
        errors: List[str] = []
        warnings: List[str] = []

        if df is None:
            errors.append("DataFrame ist None")
            if strict:
                raise DataValidationError("Keine Daten vorhanden", errors)
            return errors

        if df.empty:
            errors.append("DataFrame ist leer")
            if strict:
                raise DataValidationError("Keine Daten vorhanden", errors)
            return errors

        # 1. Spalten-Validierung
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            errors.append(f"Fehlende Pflichtspalten: {', '.join(missing_cols)}")

        # 2. Datentyp-Validierung
        for col in self.REQUIRED_COLUMNS:
            if col in df.columns:
                if not np.issubdtype(df[col].dtype, np.number):
                    warnings.append(f"Spalte '{col}' ist nicht numerisch")

        # 3. Wertebereich-Validierung (OHLC-Logik)
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            # High sollte >= Low sein
            invalid_hl = (df['High'] < df['Low']).sum()
            if invalid_hl > 0:
                errors.append(f"{invalid_hl} Zeilen mit High < Low")

            # Negative Preise
            for col in ['Open', 'High', 'Low', 'Close']:
                neg_count = (df[col] < 0).sum()
                if neg_count > 0:
                    errors.append(f"{neg_count} negative Werte in '{col}'")

            # Null-Preise
            for col in ['Open', 'High', 'Low', 'Close']:
                zero_count = (df[col] == 0).sum()
                if zero_count > 0:
                    warnings.append(f"{zero_count} Null-Werte in '{col}'")

        # 4. NaN-Validierung
        for col in self.REQUIRED_COLUMNS:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    warnings.append(f"{nan_count} NaN-Werte in '{col}'")

        # 5. Duplikate-Validierung
        if 'DateTime' in df.columns:
            dup_count = df['DateTime'].duplicated().sum()
            if dup_count > 0:
                warnings.append(f"{dup_count} doppelte DateTime-Eintraege")

        # 6. Zeitreihen-Validierung
        if 'DateTime' in df.columns and len(df) > 1:
            # Nicht-chronologische Reihenfolge
            if not df['DateTime'].is_monotonic_increasing:
                warnings.append("DateTime nicht chronologisch sortiert")

            # Grosse Luecken erkennen
            time_diffs = df['DateTime'].diff().dropna()
            if len(time_diffs) > 0:
                median_diff = time_diffs.median()
                large_gaps = (time_diffs > median_diff * 10).sum()
                if large_gaps > 0:
                    warnings.append(f"{large_gaps} grosse Zeitluecken erkannt")

        # Warnungen bei strict-Modus als Fehler werten
        if strict:
            all_issues = errors + warnings
            if all_issues:
                raise DataValidationError(
                    f"Datenvalidierung fehlgeschlagen: {len(all_issues)} Probleme",
                    invalid_fields=all_issues
                )

        # Warnungen loggen
        for warning in warnings:
            self.logger.warning(f"Validierung: {warning}")

        return errors

    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Gibt Informationen ueber die Daten zurueck.

        Args:
            df: DataFrame mit OHLCV-Daten

        Returns:
            Dictionary mit Daten-Informationen
        """
        if df is None or df.empty:
            return {}

        # Zeitintervall berechnen
        if len(df) > 1:
            time_diff = (df['DateTime'].iloc[1] - df['DateTime'].iloc[0]).total_seconds()
            if time_diff < 3600:
                interval = f'{int(time_diff / 60)} Min'
            elif time_diff < 86400:
                interval = f'{int(time_diff / 3600)} Std'
            else:
                interval = f'{int(time_diff / 86400)} Tag(e)'
        else:
            interval = 'N/A'

        return {
            'records': len(df),
            'start_date': df['DateTime'].iloc[0].strftime('%Y-%m-%d'),
            'end_date': df['DateTime'].iloc[-1].strftime('%Y-%m-%d'),
            'interval': interval,
            'price_min': df['Low'].min(),
            'price_max': df['High'].max(),
            'price_avg': df['Close'].mean(),
            'price_range_pct': ((df['High'].max() - df['Low'].min()) / df['Low'].min()) * 100,
            'columns': list(df.columns),
        }

    def log_data_info(self, df: pd.DataFrame, filepath: Optional[Path] = None):
        """Loggt Informationen ueber die geladenen Daten."""
        info = self.get_data_info(df)
        if not info:
            return

        self.logger.success(f'Daten geladen: {info["records"]} Datensätze '
                           f'({info["start_date"]} bis {info["end_date"]})')

        if filepath:
            size_mb = Path(filepath).stat().st_size / (1024 * 1024)
            self.logger.debug(f'Dateipfad: {filepath}')
            self.logger.debug(f'Dateigröße: {size_mb:.2f} MB | '
                             f'Spalten: {len(info["columns"])} | '
                             f'Intervall: ~{info["interval"]}')

        self.logger.debug(f'Zeitraum: {info["records"] / 24:.1f} Tage '
                         f'({info["start_date"]} bis {info["end_date"]})')
        self.logger.debug(f'Preis: Min={info["price_min"]:.2f} | '
                         f'Max={info["price_max"]:.2f} | '
                         f'Avg={info["price_avg"]:.2f} | '
                         f'Range={info["price_range_pct"]:.1f}%')
        self.logger.trace(f'Spalten: {", ".join(info["columns"])}')

        # Erste 10 Datensaetze als Trace ausgeben
        self.logger.trace('=== Erste 10 Datensaetze ===')
        display_cols = ['DateTime', 'Open', 'High', 'Low', 'Close']
        if 'Volume' in df.columns:
            display_cols.append('Volume')
        elif 'TickVol' in df.columns:
            display_cols.append('TickVol')

        for i, row in df.head(10).iterrows():
            dt_str = row['DateTime'].strftime('%Y-%m-%d %H:%M') if hasattr(row['DateTime'], 'strftime') else str(row['DateTime'])[:16]
            vol = row.get('Volume', row.get('TickVol', 0))
            self.logger.trace(
                f'[{i:4d}] {dt_str} | O:{row["Open"]:>10.2f} H:{row["High"]:>10.2f} '
                f'L:{row["Low"]:>10.2f} C:{row["Close"]:>10.2f} V:{vol:>12.2f}'
            )

    def find_latest_csv(self, pattern: str = 'BTCUSD*.csv') -> Optional[Path]:
        """
        Findet die neueste CSV-Datei im Datenverzeichnis.

        Args:
            pattern: Glob-Pattern fuer Dateinamen

        Returns:
            Pfad zur neuesten Datei oder None
        """
        if not self.data_dir.exists():
            self.logger.warning(f'Datenverzeichnis existiert nicht: {self.data_dir}')
            return None

        files = list(self.data_dir.glob(pattern))
        if not files:
            self.logger.warning(f'Keine Dateien gefunden mit Pattern: {pattern}')
            return None

        # Nach Aenderungszeit sortieren
        latest = max(files, key=lambda f: f.stat().st_mtime)
        self.logger.debug(f'Letzte CSV-Datei gefunden: {latest.name}')
        return latest

    def load_last_csv(self, pattern: str = 'BTCUSD*.csv') -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
        """
        Laedt die neueste CSV-Datei.

        Args:
            pattern: Glob-Pattern fuer Dateinamen

        Returns:
            Tuple aus (DataFrame, Dateipfad) oder (None, None)
        """
        start_time = time.perf_counter()

        filepath = self.find_latest_csv(pattern)
        if filepath is None:
            return None, None

        self.logger.debug(f'Lade letzte CSV-Datei: {filepath}')
        df = self.read(filepath)

        if df is not None:
            self.log_data_info(df, filepath)

        duration_ms = (time.perf_counter() - start_time) * 1000
        self.logger.timing('loadLastCSVData', duration_ms)

        return df, filepath
