"""
Labeler Modul - Erkennung von Tages-Extrema fuer Labeling
Entspricht find_daily_extrema.m aus dem MATLAB-Projekt
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.logger import get_logger


@dataclass
class Extremum:
    """Repraesentiert ein lokales Extremum."""
    index: int
    timestamp: pd.Timestamp
    price: float
    type: str  # 'HIGH' oder 'LOW'


class DailyExtremaLabeler:
    """
    Erkennt taegliche Hochs und Tiefs und generiert Labels.

    Die Labels werden basierend auf zukuenftigen Preisbewegungen generiert:
    - BUY: Preis steigt signifikant innerhalb des lookforward-Fensters
    - SELL: Preis faellt signifikant innerhalb des lookforward-Fensters
    - HOLD: Keine signifikante Bewegung

    Attributes:
        lookforward: Anzahl Perioden fuer Label-Bestimmung
        threshold_pct: Minimale Preisaenderung fuer BUY/SELL (in %)
    """

    # Label-Mapping
    LABELS = {'HOLD': 0, 'BUY': 1, 'SELL': 2}
    LABEL_NAMES = ['HOLD', 'BUY', 'SELL']

    def __init__(self, lookforward: int = 100, threshold_pct: float = 2.0):
        """
        Initialisiert den Labeler.

        Args:
            lookforward: Anzahl Perioden fuer Zukunftsbetrachtung
            threshold_pct: Schwellwert fuer signifikante Bewegung (%)
        """
        self.lookforward = lookforward
        self.threshold_pct = threshold_pct
        self.logger = get_logger()

    def find_daily_extrema(self, df: pd.DataFrame) -> Tuple[List[Extremum], List[Extremum]]:
        """
        Findet taegliche Hochs und Tiefs.

        Args:
            df: DataFrame mit OHLCV-Daten (muss DateTime-Spalte haben)

        Returns:
            Tuple aus (highs, lows) Listen
        """
        if 'DateTime' not in df.columns:
            self.logger.error('DateTime-Spalte fehlt')
            return [], []

        # Nach Datum gruppieren
        df = df.copy()
        df['Date'] = df['DateTime'].dt.date

        highs = []
        lows = []

        for date, group in df.groupby('Date'):
            if len(group) < 2:
                continue

            # Tageshoch
            high_idx = group['High'].idxmax()
            high_row = df.loc[high_idx]
            highs.append(Extremum(
                index=high_idx,
                timestamp=high_row['DateTime'],
                price=high_row['High'],
                type='HIGH'
            ))

            # Tagestief
            low_idx = group['Low'].idxmin()
            low_row = df.loc[low_idx]
            lows.append(Extremum(
                index=low_idx,
                timestamp=low_row['DateTime'],
                price=low_row['Low'],
                type='LOW'
            ))

        self.logger.debug(f'Gefunden: {len(highs)} Tageshochs, {len(lows)} Tagestiefs')
        return highs, lows

    def generate_labels(self, df: pd.DataFrame, method: str = 'future_return') -> np.ndarray:
        """
        Generiert Labels basierend auf der gewaehlten Methode.

        Args:
            df: DataFrame mit OHLCV-Daten
            method: Label-Methode ('future_return', 'extrema', 'binary')

        Returns:
            NumPy Array mit Labels (0=HOLD, 1=BUY, 2=SELL)
        """
        if method == 'future_return':
            return self._label_by_future_return(df)
        elif method == 'extrema':
            return self._label_by_extrema(df)
        elif method == 'binary':
            return self._label_binary(df)
        else:
            self.logger.error(f'Unbekannte Label-Methode: {method}')
            return np.zeros(len(df), dtype=np.int64)

    def _label_by_future_return(self, df: pd.DataFrame) -> np.ndarray:
        """
        Labelt basierend auf zukuenftiger Rendite.

        - BUY: Max-Rendite > threshold
        - SELL: Min-Rendite < -threshold
        - HOLD: Sonst
        """
        n = len(df)
        labels = np.zeros(n, dtype=np.int64)  # Default: HOLD

        close_prices = df['Close'].values

        for i in range(n - self.lookforward):
            future_prices = close_prices[i + 1:i + 1 + self.lookforward]
            current_price = close_prices[i]

            # Max und Min Rendite im Fenster
            max_return = ((future_prices.max() - current_price) / current_price) * 100
            min_return = ((future_prices.min() - current_price) / current_price) * 100

            if max_return > self.threshold_pct and max_return > abs(min_return):
                labels[i] = self.LABELS['BUY']
            elif min_return < -self.threshold_pct and abs(min_return) > max_return:
                labels[i] = self.LABELS['SELL']
            # Sonst bleibt HOLD (0)

        # Logging der Label-Verteilung
        unique, counts = np.unique(labels, return_counts=True)
        for u, c in zip(unique, counts):
            pct = c / n * 100
            self.logger.debug(f'Label {self.LABEL_NAMES[u]}: {c} ({pct:.1f}%)')

        return labels

    def _label_by_extrema(self, df: pd.DataFrame) -> np.ndarray:
        """
        Labelt basierend auf lokalen Extrema.

        - BUY: Nahe einem lokalen Tief
        - SELL: Nahe einem lokalen Hoch
        - HOLD: Sonst
        """
        n = len(df)
        labels = np.zeros(n, dtype=np.int64)

        highs, lows = self.find_daily_extrema(df)

        # Extrema-Indizes markieren
        for high in highs:
            if 0 <= high.index < n:
                labels[high.index] = self.LABELS['SELL']

        for low in lows:
            if 0 <= low.index < n:
                labels[low.index] = self.LABELS['BUY']

        return labels

    def _label_binary(self, df: pd.DataFrame) -> np.ndarray:
        """
        Binaeres Labeling: UP oder DOWN (kein HOLD).

        - BUY (1): Naechster Close > aktueller Close
        - SELL (2): Naechster Close <= aktueller Close
        """
        n = len(df)
        labels = np.zeros(n, dtype=np.int64)

        close_prices = df['Close'].values

        for i in range(n - 1):
            if close_prices[i + 1] > close_prices[i]:
                labels[i] = self.LABELS['BUY']
            else:
                labels[i] = self.LABELS['SELL']

        return labels

    def get_label_weights(self, labels: np.ndarray) -> np.ndarray:
        """
        Berechnet Klassengewichte fuer unbalancierte Daten.

        Args:
            labels: Array mit Labels

        Returns:
            Array mit Gewichten pro Klasse
        """
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)

        weights = np.zeros(len(self.LABELS), dtype=np.float32)
        for u, c in zip(unique, counts):
            # Inverse Haeufigkeit als Gewicht
            weights[u] = total / (len(unique) * c)

        return weights

    def get_class_distribution(self, labels: np.ndarray) -> dict:
        """
        Gibt die Klassenverteilung zurueck.

        Args:
            labels: Array mit Labels

        Returns:
            Dictionary mit Klassenverteilung
        """
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)

        return {
            self.LABEL_NAMES[u]: {
                'count': int(c),
                'percentage': round(c / total * 100, 2)
            }
            for u, c in zip(unique, counts)
        }
