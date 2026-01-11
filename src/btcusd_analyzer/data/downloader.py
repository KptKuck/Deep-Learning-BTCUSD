"""
Binance Downloader Modul - Laedt historische BTC-Daten von Binance
Entspricht download_btc_data.m aus dem MATLAB-Projekt
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ..core.logger import get_logger


class BinanceDownloader:
    """
    Laedt historische Kursdaten von der Binance API.

    Unterstuetzte Zeitrahmen:
    - 1m, 3m, 5m, 15m, 30m (Minuten)
    - 1h, 2h, 4h, 6h, 8h, 12h (Stunden)
    - 1d, 3d (Tage)
    - 1w, 1M (Woche, Monat)

    Attributes:
        symbol: Trading-Paar (z.B. 'BTCUSDT')
        data_dir: Verzeichnis zum Speichern der Daten
    """

    # Binance API Endpoints
    BASE_URL = 'https://api.binance.com'
    KLINES_ENDPOINT = '/api/v3/klines'

    # Maximale Anzahl Klines pro Anfrage
    MAX_LIMIT = 1000

    # Mapping: Intervall -> Millisekunden
    INTERVAL_MS = {
        '1m': 60000,
        '3m': 180000,
        '5m': 300000,
        '15m': 900000,
        '30m': 1800000,
        '1h': 3600000,
        '2h': 7200000,
        '4h': 14400000,
        '6h': 21600000,
        '8h': 28800000,
        '12h': 43200000,
        '1d': 86400000,
        '3d': 259200000,
        '1w': 604800000,
        '1M': 2592000000,
    }

    def __init__(self, symbol: str = 'BTCUSDT', data_dir: Optional[Path] = None):
        """
        Initialisiert den Downloader.

        Args:
            symbol: Trading-Paar (z.B. 'BTCUSDT')
            data_dir: Verzeichnis zum Speichern (default: ./Daten_csv)
        """
        self.symbol = symbol.upper()
        self.data_dir = Path(data_dir) if data_dir else Path.cwd() / 'Daten_csv'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger()
        self._client = None

    def _get_client(self):
        """Erstellt Binance Client (lazy loading)."""
        if self._client is None:
            try:
                from binance.client import Client
                self._client = Client()  # Ohne API-Key fuer oeffentliche Daten
                self.logger.debug('Binance Client initialisiert')
            except ImportError:
                self.logger.error('python-binance nicht installiert: pip install python-binance')
                raise
        return self._client

    def download(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = '1h',
        save: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Laedt historische Klines von Binance.

        Args:
            start_date: Startdatum (Format: 'YYYY-MM-DD')
            end_date: Enddatum (default: heute)
            interval: Zeitintervall (z.B. '1h', '4h', '1d')
            save: Speichern als CSV

        Returns:
            DataFrame mit OHLCV-Daten oder None bei Fehler
        """
        start_time = time.perf_counter()

        self.logger.debug(f'=== Binance Download gestartet ===')
        self.logger.debug(f'Symbol: {self.symbol}')
        self.logger.debug(f'Intervall: {interval}')
        self.logger.debug(f'Start-Datum (Input): {start_date}')
        self.logger.debug(f'End-Datum (Input): {end_date}')
        self.logger.debug(f'Speichern: {save}')
        self.logger.debug(f'Zielverzeichnis: {self.data_dir}')

        # Datum parsen
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            self.logger.debug(f'Start-Datum (parsed): {start_dt}')
        except ValueError as e:
            self.logger.error(f'Ungültiges Start-Datum Format: {start_date} - {e}')
            return None

        try:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.now()
            self.logger.debug(f'End-Datum (parsed): {end_dt}')
        except ValueError as e:
            self.logger.error(f'Ungültiges End-Datum Format: {end_date} - {e}')
            return None

        # Zeitraum validieren
        days_diff = (end_dt - start_dt).days
        self.logger.debug(f'Zeitraum: {days_diff} Tage')

        if days_diff < 0:
            self.logger.error(f'Start-Datum liegt nach End-Datum!')
            return None

        self.logger.info(f'Lade {self.symbol} Daten: {start_date} bis {end_dt.strftime("%Y-%m-%d")} ({days_diff} Tage)')

        try:
            self.logger.debug('Initialisiere Binance Client...')
            client = self._get_client()
            self.logger.debug('Binance Client bereit')

            # Alle Klines in Chunks laden
            all_klines = []
            current_start = int(start_dt.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)
            chunk_count = 0

            self.logger.debug(f'Start-Timestamp (ms): {current_start}')
            self.logger.debug(f'End-Timestamp (ms): {end_ms}')
            self.logger.debug(f'API URL: {self.BASE_URL}{self.KLINES_ENDPOINT}')

            while current_start < end_ms:
                chunk_count += 1
                chunk_start_time = time.perf_counter()

                self.logger.debug(f'Chunk {chunk_count}: Anfrage ab {datetime.fromtimestamp(current_start/1000)}')

                klines = client.get_klines(
                    symbol=self.symbol,
                    interval=interval,
                    startTime=current_start,
                    limit=self.MAX_LIMIT
                )

                chunk_duration = (time.perf_counter() - chunk_start_time) * 1000

                if not klines:
                    self.logger.debug(f'Chunk {chunk_count}: Keine weiteren Daten')
                    break

                all_klines.extend(klines)
                self.logger.debug(f'Chunk {chunk_count}: {len(klines)} Klines empfangen ({chunk_duration:.0f}ms)')

                # Naechster Chunk
                last_time = klines[-1][0]
                current_start = last_time + self.INTERVAL_MS.get(interval, 3600000)

            self.logger.debug(f'Download abgeschlossen: {chunk_count} Chunks, {len(all_klines)} Klines total')

            if not all_klines:
                self.logger.warning('Keine Daten erhalten')
                return None

            # In DataFrame konvertieren
            self.logger.debug('Konvertiere Klines zu DataFrame...')
            df = self._klines_to_dataframe(all_klines)
            self.logger.debug(f'DataFrame erstellt: {len(df)} Zeilen, {len(df.columns)} Spalten')
            self.logger.debug(f'Spalten: {list(df.columns)}')

            # Auf Zeitraum filtern
            df_before_filter = len(df)
            df = df[(df['DateTime'] >= start_dt) & (df['DateTime'] <= end_dt)]
            self.logger.debug(f'Nach Zeitraum-Filter: {len(df)} Zeilen (vorher: {df_before_filter})')

            if len(df) > 0:
                self.logger.debug(f'Erster Datensatz: {df.iloc[0]["DateTime"]}')
                self.logger.debug(f'Letzter Datensatz: {df.iloc[-1]["DateTime"]}')
                self.logger.debug(f'Preis-Bereich: {df["Low"].min():.2f} - {df["High"].max():.2f}')

            duration_ms = (time.perf_counter() - start_time) * 1000
            self.logger.debug(f'Gesamtdauer: {duration_ms:.0f}ms')
            self.logger.success(f'{len(df)} Datensaetze geladen in {duration_ms/1000:.1f}s')

            # Speichern
            if save:
                self.logger.debug('Speichere CSV...')
                filepath = self._save_csv(df, start_date, end_dt.strftime('%Y-%m-%d'), interval)
                file_size_kb = filepath.stat().st_size / 1024
                self.logger.debug(f'Dateigroesse: {file_size_kb:.1f} KB')
                self.logger.success(f'Gespeichert: {filepath}')

            self.logger.debug('=== Binance Download beendet ===')
            return df

        except ImportError as e:
            self.logger.error(f'Modul-Fehler: {e}')
            self.logger.debug('Hinweis: pip install python-binance')
            return None
        except Exception as e:
            self.logger.error(f'Download-Fehler: {e}')
            self.logger.debug(f'Fehler-Typ: {type(e).__name__}')
            import traceback
            self.logger.debug(f'Traceback: {traceback.format_exc()}')
            return None

    def _klines_to_dataframe(self, klines: List) -> pd.DataFrame:
        """Konvertiert Binance Klines in DataFrame."""
        columns = [
            'OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume',
            'CloseTime', 'QuoteVolume', 'Trades', 'TakerBuyBase',
            'TakerBuyQuote', 'Ignore'
        ]

        df = pd.DataFrame(klines, columns=columns)

        # DateTime konvertieren
        df['DateTime'] = pd.to_datetime(df['OpenTime'], unit='ms')
        df['Date'] = df['DateTime'].dt.strftime('%Y.%m.%d')
        df['Time'] = df['DateTime'].dt.strftime('%H:%M')

        # Numerische Spalten
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Relevante Spalten auswaehlen
        df = df[['DateTime', 'Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # TickVol als Alias fuer Volume (Kompatibilitaet mit MATLAB-Format)
        df['TickVol'] = df['Volume']
        df['Vol'] = 0
        df['Spread'] = 0

        return df.reset_index(drop=True)

    def _save_csv(self, df: pd.DataFrame, start_date: str, end_date: str, interval: str) -> Path:
        """Speichert DataFrame als CSV."""
        # Dateiname: BTCUSD_2025-01-01_2026-01-10.csv
        symbol_short = self.symbol.replace('USDT', 'USD')
        filename = f'{symbol_short}_{start_date}_{end_date}.csv'
        filepath = self.data_dir / filename

        # MetaTrader-kompatibles Format
        df_export = df[['DateTime', 'Date', 'Time', 'Open', 'High', 'Low',
                        'Close', 'TickVol', 'Vol', 'Spread']].copy()
        df_export.to_csv(filepath, index=False)

        return filepath

    def update_data(self, existing_df: pd.DataFrame, interval: str = '1h') -> pd.DataFrame:
        """
        Aktualisiert vorhandene Daten bis heute.

        Args:
            existing_df: Vorhandener DataFrame
            interval: Zeitintervall

        Returns:
            Aktualisierter DataFrame
        """
        last_date = existing_df['DateTime'].max()
        start_date = (last_date + timedelta(hours=1)).strftime('%Y-%m-%d')

        self.logger.info(f'Aktualisiere Daten ab {start_date}')

        new_df = self.download(start_date, interval=interval, save=False)

        if new_df is not None and not new_df.empty:
            # Zusammenfuehren und Duplikate entfernen
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=['DateTime'], keep='last')
            combined = combined.sort_values('DateTime').reset_index(drop=True)

            self.logger.success(f'{len(new_df)} neue Datensätze hinzugefuegt')
            return combined

        return existing_df

    def get_available_intervals(self) -> List[str]:
        """Gibt verfuegbare Zeitintervalle zurueck."""
        return list(self.INTERVAL_MS.keys())
