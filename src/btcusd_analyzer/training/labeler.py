"""
Labeler Modul - Erkennung von Tages-Extrema fuer Labeling
Entspricht find_daily_extrema.m aus dem MATLAB-Projekt
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.logger import get_logger

# Optional: scipy fuer Peak-Detection
try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class LabelingMethod(Enum):
    """Verfuegbare Labeling-Methoden."""
    FUTURE_RETURN = "future_return"
    ZIGZAG = "zigzag"
    PEAKS = "peaks"
    FRACTALS = "fractals"
    PIVOTS = "pivots"
    EXTREMA_DAILY = "extrema_daily"
    BINARY = "binary"


@dataclass
class LabelingConfig:
    """Konfiguration fuer Label-Generierung."""
    method: LabelingMethod = LabelingMethod.FUTURE_RETURN
    # Gemeinsame Parameter
    lookforward: int = 100
    threshold_pct: float = 2.0
    # ZigZag Parameter
    zigzag_threshold: float = 5.0

    # Peak Detection Parameter (alle scipy.signal.find_peaks Parameter)
    # Basis-Parameter
    peak_distance: int = 10  # Mindestabstand zwischen Peaks
    peak_prominence: float = 0.5  # Wie stark der Peak herausragt (in %)
    peak_width: Optional[float] = None  # Minimale Breite des Peaks
    # Erweiterte Parameter
    peak_height: Optional[float] = None  # Absoluter Mindest-Peakwert (in % vom Mittelwert)
    peak_threshold: Optional[float] = None  # Mindestdifferenz zu direkten Nachbarn (in %)
    peak_plateau_size: Optional[int] = None  # Minimale Plateau-Groesse
    peak_wlen: Optional[int] = None  # Fenstergroesse fuer Prominence-Berechnung
    peak_rel_height: float = 0.5  # Relative Hoehe fuer Width-Berechnung (0-1)

    # Williams Fractals Parameter
    fractal_order: int = 2
    # Pivot Points Parameter
    pivot_lookback: int = 5


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

    def generate_labels(
        self,
        df: pd.DataFrame,
        method: Optional[str] = None,
        config: Optional[LabelingConfig] = None
    ) -> np.ndarray:
        """
        Generiert Labels basierend auf der gewaehlten Methode.

        Args:
            df: DataFrame mit OHLCV-Daten
            method: Label-Methode als String (deprecated, fuer Rueckwaertskompatibilitaet)
            config: LabelingConfig mit Methode und Parametern (bevorzugt)

        Returns:
            NumPy Array mit Labels (0=HOLD, 1=BUY, 2=SELL)
        """
        # Config erstellen falls nur method angegeben
        if config is None:
            if method is None:
                method = 'future_return'
            config = LabelingConfig(
                method=LabelingMethod(method) if method in [m.value for m in LabelingMethod] else LabelingMethod.FUTURE_RETURN,
                lookforward=self.lookforward,
                threshold_pct=self.threshold_pct
            )

        # Methode ausfuehren
        if config.method == LabelingMethod.FUTURE_RETURN:
            return self._label_by_future_return(df)
        elif config.method == LabelingMethod.ZIGZAG:
            return self._label_by_zigzag(df, config)
        elif config.method == LabelingMethod.PEAKS:
            return self._label_by_peaks(df, config)
        elif config.method == LabelingMethod.FRACTALS:
            return self._label_by_fractals(df, config)
        elif config.method == LabelingMethod.PIVOTS:
            return self._label_by_pivots(df, config)
        elif config.method == LabelingMethod.EXTREMA_DAILY:
            return self._label_by_extrema(df)
        elif config.method == LabelingMethod.BINARY:
            return self._label_binary(df)
        else:
            self.logger.error(f'Unbekannte Label-Methode: {config.method}')
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

    # =========================================================================
    # Neue Extrema-Erkennungsmethoden
    # =========================================================================

    def find_zigzag_extrema(
        self,
        df: pd.DataFrame,
        threshold_pct: float = 5.0
    ) -> Tuple[List[int], List[int]]:
        """
        ZigZag-Algorithmus: Erkennt Extrema bei Richtungswechsel nach X% Bewegung.

        Args:
            df: DataFrame mit OHLCV-Daten
            threshold_pct: Mindest-Prozentbewegung fuer Richtungswechsel

        Returns:
            Tuple aus (high_indices, low_indices)
        """
        close = df['Close'].values
        n = len(close)

        if n < 2:
            return [], []

        high_indices = []
        low_indices = []

        # Initialisierung: Erstes Extremum finden
        trend = 0  # 1 = aufwaerts, -1 = abwaerts
        last_high_idx = 0
        last_low_idx = 0
        last_high = close[0]
        last_low = close[0]

        for i in range(1, n):
            price = close[i]

            if trend == 0:
                # Noch kein Trend - warte auf erste signifikante Bewegung
                if price >= last_low * (1 + threshold_pct / 100):
                    trend = 1
                    last_high = price
                    last_high_idx = i
                    low_indices.append(last_low_idx)
                elif price <= last_high * (1 - threshold_pct / 100):
                    trend = -1
                    last_low = price
                    last_low_idx = i
                    high_indices.append(last_high_idx)
                else:
                    # Aktualisiere potentielle Extrema
                    if price > last_high:
                        last_high = price
                        last_high_idx = i
                    if price < last_low:
                        last_low = price
                        last_low_idx = i

            elif trend == 1:  # Aufwaertstrend
                if price > last_high:
                    last_high = price
                    last_high_idx = i
                elif price <= last_high * (1 - threshold_pct / 100):
                    # Richtungswechsel nach unten
                    high_indices.append(last_high_idx)
                    trend = -1
                    last_low = price
                    last_low_idx = i

            else:  # trend == -1, Abwaertstrend
                if price < last_low:
                    last_low = price
                    last_low_idx = i
                elif price >= last_low * (1 + threshold_pct / 100):
                    # Richtungswechsel nach oben
                    low_indices.append(last_low_idx)
                    trend = 1
                    last_high = price
                    last_high_idx = i

        self.logger.debug(
            f'ZigZag ({threshold_pct}%): {len(high_indices)} Hochs, {len(low_indices)} Tiefs'
        )
        return high_indices, low_indices

    def find_peaks_extrema(
        self,
        df: pd.DataFrame,
        config: Optional[LabelingConfig] = None,
        # Legacy-Parameter fuer Rueckwaertskompatibilitaet
        prominence: float = 0.5,
        distance: int = 10
    ) -> Tuple[List[int], List[int]]:
        """
        Peak-Detection mit scipy.signal.find_peaks.

        Unterstuetzt alle Parameter von scipy.signal.find_peaks:
        - distance: Mindestabstand zwischen Peaks
        - prominence: Wie stark der Peak herausragt
        - width: Minimale Breite des Peaks
        - height: Absoluter Mindest-Peakwert
        - threshold: Mindestdifferenz zu direkten Nachbarn
        - plateau_size: Minimale Plateau-Groesse
        - wlen: Fenstergroesse fuer Prominence-Berechnung
        - rel_height: Relative Hoehe fuer Width-Berechnung

        Args:
            df: DataFrame mit OHLCV-Daten
            config: LabelingConfig mit allen Peak-Parametern (bevorzugt)
            prominence: Legacy-Parameter (% des Preises)
            distance: Legacy-Parameter Mindestabstand

        Returns:
            Tuple aus (high_indices, low_indices)
        """
        # Parameter aus Config oder Legacy-Werte
        if config is not None:
            distance = config.peak_distance
            prominence = config.peak_prominence
            width = config.peak_width
            height = config.peak_height
            threshold = config.peak_threshold
            plateau_size = config.peak_plateau_size
            wlen = config.peak_wlen
            rel_height = config.peak_rel_height
        else:
            width = None
            height = None
            threshold = None
            plateau_size = None
            wlen = None
            rel_height = 0.5

        if not SCIPY_AVAILABLE:
            self.logger.warning('scipy nicht verfuegbar - verwende Fallback')
            return self._find_peaks_fallback(df, distance)

        close = df['Close'].values
        n = len(close)
        price_mean = close.mean()
        price_range = close.max() - close.min()

        # Parameter-Dictionary fuer find_peaks aufbauen
        peak_params = {'distance': distance}

        # Prominenz als absoluter Wert (% der Preisspanne)
        if prominence is not None and prominence > 0:
            peak_params['prominence'] = price_range * (prominence / 100)

        # Width (direkt in Datenpunkten)
        if width is not None and width > 0:
            peak_params['width'] = width

        # Height als absoluter Wert (% vom Mittelwert)
        if height is not None and height > 0:
            abs_height = price_mean * (height / 100)
            peak_params['height'] = abs_height

        # Threshold als absoluter Wert (% der Preisspanne)
        if threshold is not None and threshold > 0:
            peak_params['threshold'] = price_range * (threshold / 100)

        # Plateau Size (direkt in Datenpunkten)
        if plateau_size is not None and plateau_size > 0:
            peak_params['plateau_size'] = plateau_size

        # Wlen - Fenstergroesse fuer Prominence
        if wlen is not None and wlen > 0:
            peak_params['wlen'] = wlen

        # Rel_height fuer Width-Berechnung
        if rel_height is not None and 0 < rel_height < 1:
            peak_params['rel_height'] = rel_height

        self.logger.debug(f'Peak-Detection Parameter: {peak_params}')

        # Hochpunkte finden
        high_indices, high_props = find_peaks(close, **peak_params)

        # Tiefpunkte finden (invertierte Daten)
        # Bei Height muessen wir das Vorzeichen umkehren
        low_params = peak_params.copy()
        if 'height' in low_params:
            # Fuer invertierte Daten: Minimum wird Maximum
            low_params['height'] = -close.min() + price_mean * (height / 100) if height else None
            if low_params['height'] is None or low_params['height'] <= 0:
                del low_params['height']

        low_indices, low_props = find_peaks(-close, **low_params)

        self.logger.debug(
            f'Peaks gefunden: {len(high_indices)} Hochs, {len(low_indices)} Tiefs'
        )
        return list(high_indices), list(low_indices)

    def _find_peaks_fallback(
        self,
        df: pd.DataFrame,
        distance: int = 10
    ) -> Tuple[List[int], List[int]]:
        """Fallback Peak-Detection ohne scipy."""
        close = df['Close'].values
        n = len(close)

        high_indices = []
        low_indices = []

        for i in range(distance, n - distance):
            window = close[i - distance:i + distance + 1]
            center_val = close[i]

            if center_val == window.max():
                high_indices.append(i)
            elif center_val == window.min():
                low_indices.append(i)

        return high_indices, low_indices

    def find_williams_fractals(
        self,
        df: pd.DataFrame,
        order: int = 2
    ) -> Tuple[List[int], List[int]]:
        """
        Williams Fractals: N-Bar Fractal Pattern.

        Ein Fractal-Hoch ist ein Punkt, der hoeher ist als N Bars links und rechts.
        Ein Fractal-Tief ist ein Punkt, der niedriger ist als N Bars links und rechts.

        Args:
            df: DataFrame mit OHLCV-Daten
            order: Anzahl Bars links/rechts (default: 2 = klassisches Williams)

        Returns:
            Tuple aus (high_indices, low_indices)
        """
        high = df['High'].values
        low = df['Low'].values
        n = len(high)

        high_indices = []
        low_indices = []

        for i in range(order, n - order):
            # Fractal-Hoch pruefen
            is_fractal_high = True
            for j in range(1, order + 1):
                if high[i] <= high[i - j] or high[i] <= high[i + j]:
                    is_fractal_high = False
                    break
            if is_fractal_high:
                high_indices.append(i)

            # Fractal-Tief pruefen
            is_fractal_low = True
            for j in range(1, order + 1):
                if low[i] >= low[i - j] or low[i] >= low[i + j]:
                    is_fractal_low = False
                    break
            if is_fractal_low:
                low_indices.append(i)

        self.logger.debug(
            f'Fractals (order={order}): {len(high_indices)} Hochs, {len(low_indices)} Tiefs'
        )
        return high_indices, low_indices

    def find_pivot_points(
        self,
        df: pd.DataFrame,
        lookback: int = 5
    ) -> Tuple[List[int], List[int]]:
        """
        Pivot Points: Lokale Maxima/Minima in einem Fenster.

        Args:
            df: DataFrame mit OHLCV-Daten
            lookback: Fenstergroesse links und rechts

        Returns:
            Tuple aus (high_indices, low_indices)
        """
        high = df['High'].values
        low = df['Low'].values
        n = len(high)

        high_indices = []
        low_indices = []

        for i in range(lookback, n - lookback):
            window_high = high[i - lookback:i + lookback + 1]
            window_low = low[i - lookback:i + lookback + 1]

            # Pivot High: Hoechster Punkt im Fenster
            if high[i] == window_high.max():
                high_indices.append(i)

            # Pivot Low: Tiefster Punkt im Fenster
            if low[i] == window_low.min():
                low_indices.append(i)

        self.logger.debug(
            f'Pivots (lookback={lookback}): {len(high_indices)} Hochs, {len(low_indices)} Tiefs'
        )
        return high_indices, low_indices

    def _label_by_zigzag(self, df: pd.DataFrame, config: LabelingConfig) -> np.ndarray:
        """Labelt basierend auf ZigZag-Extrema."""
        n = len(df)
        labels = np.zeros(n, dtype=np.int64)

        high_indices, low_indices = self.find_zigzag_extrema(
            df, threshold_pct=config.zigzag_threshold
        )

        for idx in high_indices:
            if 0 <= idx < n:
                labels[idx] = self.LABELS['SELL']

        for idx in low_indices:
            if 0 <= idx < n:
                labels[idx] = self.LABELS['BUY']

        self._log_label_distribution(labels)
        return labels

    def _label_by_peaks(self, df: pd.DataFrame, config: LabelingConfig) -> np.ndarray:
        """Labelt basierend auf Peak-Detection mit allen scipy.signal.find_peaks Parametern."""
        n = len(df)
        labels = np.zeros(n, dtype=np.int64)

        # Verwende die volle Config mit allen Parametern
        high_indices, low_indices = self.find_peaks_extrema(df, config=config)

        for idx in high_indices:
            if 0 <= idx < n:
                labels[idx] = self.LABELS['SELL']

        for idx in low_indices:
            if 0 <= idx < n:
                labels[idx] = self.LABELS['BUY']

        self._log_label_distribution(labels)
        return labels

    def _label_by_fractals(self, df: pd.DataFrame, config: LabelingConfig) -> np.ndarray:
        """Labelt basierend auf Williams Fractals."""
        n = len(df)
        labels = np.zeros(n, dtype=np.int64)

        high_indices, low_indices = self.find_williams_fractals(
            df, order=config.fractal_order
        )

        for idx in high_indices:
            if 0 <= idx < n:
                labels[idx] = self.LABELS['SELL']

        for idx in low_indices:
            if 0 <= idx < n:
                labels[idx] = self.LABELS['BUY']

        self._log_label_distribution(labels)
        return labels

    def _label_by_pivots(self, df: pd.DataFrame, config: LabelingConfig) -> np.ndarray:
        """Labelt basierend auf Pivot Points."""
        n = len(df)
        labels = np.zeros(n, dtype=np.int64)

        high_indices, low_indices = self.find_pivot_points(
            df, lookback=config.pivot_lookback
        )

        for idx in high_indices:
            if 0 <= idx < n:
                labels[idx] = self.LABELS['SELL']

        for idx in low_indices:
            if 0 <= idx < n:
                labels[idx] = self.LABELS['BUY']

        self._log_label_distribution(labels)
        return labels

    def _log_label_distribution(self, labels: np.ndarray) -> None:
        """Loggt die Label-Verteilung."""
        n = len(labels)
        unique, counts = np.unique(labels, return_counts=True)
        for u, c in zip(unique, counts):
            pct = c / n * 100
            self.logger.debug(f'Label {self.LABEL_NAMES[u]}: {c} ({pct:.1f}%)')

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
