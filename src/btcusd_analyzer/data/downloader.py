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

        # Datum parsen
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.now()

        self.logger.info(f'Lade {self.symbol} Daten: {start_date} bis {end_dt.strftime("%Y-%m-%d")}')
        self.logger.debug(f'Intervall: {interval}')

        try:
            client = self._get_client()

            # Alle Klines in Chunks laden
            all_klines = []
            current_start = int(start_dt.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)

            while current_start < end_ms:
                klines = client.get_klines(
                    symbol=self.symbol,
                    interval=interval,
                    startTime=current_start,
                    limit=self.MAX_LIMIT
                )

                if not klines:
                    break

                all_klines.extend(klines)

                # Naechster Chunk
                last_time = klines[-1][0]
                current_start = last_time + self.INTERVAL_MS.get(interval, 3600000)

                self.logger.trace(f'Geladen: {len(all_klines)} Klines...')

            if not all_klines:
                self.logger.warning('Keine Daten erhalten')
                return None

            # In DataFrame konvertieren
            df = self._klines_to_dataframe(all_klines)

            # Auf Zeitraum filtern
            df = df[(df['DateTime'] >= start_dt) & (df['DateTime'] <= end_dt)]

            duration_ms = (time.perf_counter() - start_time) * 1000
            self.logger.timing('download_btc_data', duration_ms)
            self.logger.success(f'{len(df)} Datensätze geladen')

            # Speichern
            if save:
                filepath = self._save_csv(df, start_date, end_dt.strftime('%Y-%m-%d'), interval)
                self.logger.success(f'Gespeichert: {filepath}')

            return df

        except Exception as e:
            self.logger.error(f'Download-Fehler: {e}')
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
