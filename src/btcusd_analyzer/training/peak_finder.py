"""
PeakFinder Modul - Erkennung von Preisextrema (Hochs/Tiefs)

Dieses Modul ist ausschliesslich fuer die Peak-Erkennung zustaendig.
Die Umwandlung in Labels erfolgt separat im Labeler-Modul.

Peaks sind Rohpositionen (Indizes) im Preischart:
- Hochs (Highs): Lokale Maxima
- Tiefs (Lows): Lokale Minima

Die Terminologie ist klar getrennt:
- Peak/Extremum = Position im Chart (Index)
- Label = Klassifikation fuer Training (BUY/SELL/HOLD)
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

from ..core.logger import get_logger


# Optional: scipy fuer Peak-Detection
try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class PeakMethod(Enum):
    """Verfuegbare Peak-Erkennungsmethoden."""
    ZIGZAG = "zigzag"
    SCIPY_PEAKS = "scipy_peaks"
    WILLIAMS_FRACTALS = "fractals"
    PIVOT_POINTS = "pivots"
    DAILY_EXTREMA = "daily_extrema"


@dataclass
class PeakConfig:
    """Konfiguration fuer Peak-Erkennung."""
    method: PeakMethod = PeakMethod.SCIPY_PEAKS

    # ZigZag Parameter
    zigzag_threshold_pct: float = 5.0

    # Scipy Peak Detection Parameter
    peak_distance: int = 10
    peak_prominence_pct: float = 0.5
    peak_width: Optional[int] = None
    peak_height_pct: Optional[float] = None
    peak_threshold_pct: Optional[float] = None
    peak_plateau_size: Optional[int] = None
    peak_wlen: Optional[int] = None
    peak_rel_height: float = 0.5

    # Williams Fractals Parameter
    fractal_order: int = 2

    # Pivot Points Parameter
    pivot_lookback: int = 5


@dataclass
class PeakResult:
    """Ergebnis der Peak-Erkennung."""
    high_indices: List[int]  # Indizes der Hochpunkte
    low_indices: List[int]   # Indizes der Tiefpunkte
    method: PeakMethod
    config: PeakConfig

    @property
    def num_highs(self) -> int:
        return len(self.high_indices)

    @property
    def num_lows(self) -> int:
        return len(self.low_indices)

    @property
    def total_peaks(self) -> int:
        return self.num_highs + self.num_lows


class PeakFinder:
    """
    Erkennt Preisextrema (Hochs und Tiefs) in Zeitreihendaten.

    Diese Klasse ist ausschliesslich fuer die Peak-Erkennung zustaendig.
    Die Umwandlung der gefundenen Peaks in Labels (BUY/SELL/HOLD)
    erfolgt separat durch den Labeler.

    Unterstuetzte Methoden:
    - ZigZag: Richtungswechsel nach X% Bewegung
    - Scipy Peaks: scipy.signal.find_peaks mit Prominence
    - Williams Fractals: N-Bar Fractal Pattern
    - Pivot Points: Lokale Maxima/Minima in Fenster
    - Daily Extrema: Taegliche Hochs/Tiefs

    Example:
        >>> finder = PeakFinder()
        >>> result = finder.find_peaks(df, PeakConfig(method=PeakMethod.SCIPY_PEAKS))
        >>> print(f"Gefunden: {result.num_highs} Hochs, {result.num_lows} Tiefs")
    """

    def __init__(self):
        self.logger = get_logger()

    def find_peaks(
        self,
        df: pd.DataFrame,
        config: Optional[PeakConfig] = None
    ) -> PeakResult:
        """
        Findet Peaks basierend auf der konfigurierten Methode.

        Args:
            df: DataFrame mit OHLCV-Daten (muss Close-Spalte haben)
            config: PeakConfig mit Methode und Parametern

        Returns:
            PeakResult mit high_indices und low_indices
        """
        if config is None:
            config = PeakConfig()

        self.logger.debug(f"[PeakFinder] Methode: {config.method.value}")

        if config.method == PeakMethod.ZIGZAG:
            highs, lows = self._find_zigzag(df, config)
        elif config.method == PeakMethod.SCIPY_PEAKS:
            highs, lows = self._find_scipy_peaks(df, config)
        elif config.method == PeakMethod.WILLIAMS_FRACTALS:
            highs, lows = self._find_fractals(df, config)
        elif config.method == PeakMethod.PIVOT_POINTS:
            highs, lows = self._find_pivots(df, config)
        elif config.method == PeakMethod.DAILY_EXTREMA:
            highs, lows = self._find_daily_extrema(df)
        else:
            self.logger.error(f"Unbekannte Peak-Methode: {config.method}")
            highs, lows = [], []

        self.logger.debug(f"[PeakFinder] Gefunden: {len(highs)} Hochs, {len(lows)} Tiefs")

        return PeakResult(
            high_indices=highs,
            low_indices=lows,
            method=config.method,
            config=config
        )

    def _find_zigzag(
        self,
        df: pd.DataFrame,
        config: PeakConfig
    ) -> Tuple[List[int], List[int]]:
        """
        ZigZag-Algorithmus: Erkennt Extrema bei Richtungswechsel nach X% Bewegung.

        Args:
            df: DataFrame mit OHLCV-Daten
            config: PeakConfig mit zigzag_threshold_pct

        Returns:
            Tuple aus (high_indices, low_indices)
        """
        close = df['Close'].values
        n = len(close)
        threshold_pct = config.zigzag_threshold_pct

        if n < 2:
            return [], []

        high_indices = []
        low_indices = []

        # Initialisierung
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
                    high_indices.append(last_high_idx)
                    trend = -1
                    last_low = price
                    last_low_idx = i

            else:  # trend == -1, Abwaertstrend
                if price < last_low:
                    last_low = price
                    last_low_idx = i
                elif price >= last_low * (1 + threshold_pct / 100):
                    low_indices.append(last_low_idx)
                    trend = 1
                    last_high = price
                    last_high_idx = i

        self.logger.debug(f"[PeakFinder] ZigZag ({threshold_pct}%): "
                         f"{len(high_indices)} Hochs, {len(low_indices)} Tiefs")
        return high_indices, low_indices

    def _find_scipy_peaks(
        self,
        df: pd.DataFrame,
        config: PeakConfig
    ) -> Tuple[List[int], List[int]]:
        """
        Peak-Detection mit scipy.signal.find_peaks.

        Args:
            df: DataFrame mit OHLCV-Daten
            config: PeakConfig mit allen scipy-Parametern

        Returns:
            Tuple aus (high_indices, low_indices)
        """
        if not SCIPY_AVAILABLE:
            self.logger.warning("scipy nicht verfuegbar - verwende Fallback")
            return self._find_peaks_fallback(df, config.peak_distance)

        close = df['Close'].values
        price_mean = close.mean()
        price_range = close.max() - close.min()

        # Parameter-Dictionary aufbauen
        peak_params: Dict[str, Any] = {'distance': config.peak_distance}

        # Prominenz als absoluter Wert (% der Preisspanne)
        if config.peak_prominence_pct > 0:
            peak_params['prominence'] = price_range * (config.peak_prominence_pct / 100)

        # Width (direkt in Datenpunkten)
        if config.peak_width is not None and config.peak_width > 0:
            peak_params['width'] = config.peak_width

        # Height als absoluter Wert (% vom Mittelwert)
        if config.peak_height_pct is not None and config.peak_height_pct > 0:
            peak_params['height'] = price_mean * (config.peak_height_pct / 100)

        # Threshold als absoluter Wert (% der Preisspanne)
        if config.peak_threshold_pct is not None and config.peak_threshold_pct > 0:
            peak_params['threshold'] = price_range * (config.peak_threshold_pct / 100)

        # Plateau Size (direkt in Datenpunkten)
        if config.peak_plateau_size is not None and config.peak_plateau_size > 0:
            peak_params['plateau_size'] = config.peak_plateau_size

        # Wlen - Fenstergroesse fuer Prominence
        if config.peak_wlen is not None and config.peak_wlen > 0:
            peak_params['wlen'] = config.peak_wlen

        # Rel_height fuer Width-Berechnung
        if 0 < config.peak_rel_height < 1:
            peak_params['rel_height'] = config.peak_rel_height

        self.logger.debug(f"[PeakFinder] scipy.find_peaks Parameter: {peak_params}")

        # Hochpunkte finden
        high_indices, _ = find_peaks(close, **peak_params)

        # Tiefpunkte finden (invertierte Daten)
        low_params = peak_params.copy()
        if 'height' in low_params:
            del low_params['height']  # Height macht bei invertierten Daten keinen Sinn

        low_indices, _ = find_peaks(-close, **low_params)

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

    def _find_fractals(
        self,
        df: pd.DataFrame,
        config: PeakConfig
    ) -> Tuple[List[int], List[int]]:
        """
        Williams Fractals: N-Bar Fractal Pattern.

        Ein Fractal-Hoch ist ein Punkt, der hoeher ist als N Bars links und rechts.

        Args:
            df: DataFrame mit OHLCV-Daten
            config: PeakConfig mit fractal_order

        Returns:
            Tuple aus (high_indices, low_indices)
        """
        high = df['High'].values
        low = df['Low'].values
        n = len(high)
        order = config.fractal_order

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

        self.logger.debug(f"[PeakFinder] Fractals (order={order}): "
                         f"{len(high_indices)} Hochs, {len(low_indices)} Tiefs")
        return high_indices, low_indices

    def _find_pivots(
        self,
        df: pd.DataFrame,
        config: PeakConfig
    ) -> Tuple[List[int], List[int]]:
        """
        Pivot Points: Lokale Maxima/Minima in einem Fenster.

        Args:
            df: DataFrame mit OHLCV-Daten
            config: PeakConfig mit pivot_lookback

        Returns:
            Tuple aus (high_indices, low_indices)
        """
        high = df['High'].values
        low = df['Low'].values
        n = len(high)
        lookback = config.pivot_lookback

        high_indices = []
        low_indices = []

        for i in range(lookback, n - lookback):
            window_high = high[i - lookback:i + lookback + 1]
            window_low = low[i - lookback:i + lookback + 1]

            # Pivot High
            if high[i] == window_high.max():
                high_indices.append(i)

            # Pivot Low
            if low[i] == window_low.min():
                low_indices.append(i)

        self.logger.debug(f"[PeakFinder] Pivots (lookback={lookback}): "
                         f"{len(high_indices)} Hochs, {len(low_indices)} Tiefs")
        return high_indices, low_indices

    def _find_daily_extrema(
        self,
        df: pd.DataFrame
    ) -> Tuple[List[int], List[int]]:
        """
        Findet taegliche Hochs und Tiefs.

        Args:
            df: DataFrame mit OHLCV-Daten (muss DateTime-Spalte haben)

        Returns:
            Tuple aus (high_indices, low_indices)
        """
        if 'DateTime' not in df.columns:
            self.logger.error("DateTime-Spalte fehlt fuer taegliche Extrema")
            return [], []

        df_copy = df.copy()
        df_copy['Date'] = df_copy['DateTime'].dt.date

        high_indices = []
        low_indices = []

        for date, group in df_copy.groupby('Date'):
            if len(group) < 2:
                continue

            # Tageshoch
            high_idx = group['High'].idxmax()
            high_indices.append(high_idx)

            # Tagestief
            low_idx = group['Low'].idxmin()
            low_indices.append(low_idx)

        self.logger.debug(f"[PeakFinder] Tages-Extrema: "
                         f"{len(high_indices)} Hochs, {len(low_indices)} Tiefs")
        return high_indices, low_indices
