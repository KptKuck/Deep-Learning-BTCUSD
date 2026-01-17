"""
Labeler Modul - Generierung von Trainings-Labels aus Peaks

Dieses Modul ist fuer die Label-Generierung zustaendig.
Die Peak-Erkennung erfolgt separat im PeakFinder-Modul.

Labels sind Klassifikationen fuer das Training:
- BUY: Kaufsignal (bei Tiefpunkten)
- SELL: Verkaufssignal (bei Hochpunkten)
- HOLD: Halten/Warten (dazwischen)

Die Terminologie ist klar getrennt:
- Peak/Extremum = Position im Chart (Index) -> PeakFinder
- Label = Klassifikation fuer Training -> Labeler
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.logger import get_logger
from .peak_finder import PeakFinder, PeakConfig, PeakMethod, PeakResult


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
    num_classes: int = 3  # 2 oder 3

    # ZigZag Parameter
    zigzag_threshold: float = 5.0

    # Peak Detection Parameter (fuer PeakFinder)
    peak_distance: int = 10
    peak_prominence: float = 0.5
    peak_width: Optional[float] = None
    peak_height: Optional[float] = None
    peak_threshold: Optional[float] = None
    peak_plateau_size: Optional[int] = None
    peak_wlen: Optional[int] = None
    peak_rel_height: float = 0.5

    # Williams Fractals Parameter
    fractal_order: int = 2

    # Pivot Points Parameter
    pivot_lookback: int = 5


@dataclass
class LabelResult:
    """Ergebnis der Label-Generierung."""
    labels: np.ndarray  # Label-Array
    peak_result: Optional[PeakResult]  # Peaks (falls Peak-basierte Methode)
    num_classes: int
    method: LabelingMethod

    # Convenience-Properties fuer Chart-Anzeige
    @property
    def buy_indices(self) -> List[int]:
        """Indizes der BUY-Signale (Tiefpunkte)."""
        if self.peak_result:
            return self.peak_result.low_indices
        return []

    @property
    def sell_indices(self) -> List[int]:
        """Indizes der SELL-Signale (Hochpunkte)."""
        if self.peak_result:
            return self.peak_result.high_indices
        return []


class Labeler:
    """
    Generiert Trainings-Labels aus Preisdaten.

    Diese Klasse ist fuer die Label-Generierung zustaendig.
    Fuer Peak-basierte Methoden wird der PeakFinder verwendet.

    Die Labels werden basierend auf der gewaehlten Methode generiert:
    - Peak-basiert: Peaks werden zu BUY/SELL, Rest zu HOLD
    - Future-Return: Basierend auf zukuenftiger Rendite
    - Binary: Einfaches UP/DOWN

    Unterstuetzt 2 oder 3 Klassen:
    - 2 Klassen: BUY (0), SELL (1) - binaere Klassifikation
    - 3 Klassen: HOLD (0), BUY (1), SELL (2) - mit neutraler Klasse

    Example:
        >>> labeler = Labeler()
        >>> result = labeler.generate_labels(df, LabelingConfig(method=LabelingMethod.PEAKS))
        >>> print(f"Labels: {result.labels.shape}, Peaks: {len(result.buy_indices)} BUY")
    """

    # Label-Mapping fuer 3 Klassen
    LABELS_3 = {'HOLD': 0, 'BUY': 1, 'SELL': 2}
    LABEL_NAMES_3 = ['HOLD', 'BUY', 'SELL']

    # Label-Mapping fuer 2 Klassen
    LABELS_2 = {'BUY': 0, 'SELL': 1}
    LABEL_NAMES_2 = ['BUY', 'SELL']

    def __init__(self, num_classes: int = 3):
        """
        Initialisiert den Labeler.

        Args:
            num_classes: Anzahl Klassen (2=BUY/SELL, 3=BUY/HOLD/SELL)
        """
        self.num_classes = num_classes
        self.logger = get_logger()
        self.peak_finder = PeakFinder()
        self._set_label_mapping(num_classes)

    def _set_label_mapping(self, num_classes: int):
        """Setzt das Label-Mapping basierend auf der Klassenanzahl."""
        if num_classes == 2:
            self.LABELS = self.LABELS_2
            self.LABEL_NAMES = self.LABEL_NAMES_2
        else:
            self.LABELS = self.LABELS_3
            self.LABEL_NAMES = self.LABEL_NAMES_3

    def generate_labels(
        self,
        df: pd.DataFrame,
        config: Optional[LabelingConfig] = None
    ) -> LabelResult:
        """
        Generiert Labels basierend auf der konfigurierten Methode.

        Args:
            df: DataFrame mit OHLCV-Daten
            config: LabelingConfig mit Methode und Parametern

        Returns:
            LabelResult mit labels, peak_result und Metadaten
        """
        if config is None:
            config = LabelingConfig()

        self._set_label_mapping(config.num_classes)
        self.num_classes = config.num_classes

        self.logger.debug(f"[Labeler] Methode: {config.method.value}, "
                         f"Klassen: {config.num_classes}")

        # Methode ausfuehren
        if config.method == LabelingMethod.FUTURE_RETURN:
            labels, peak_result = self._label_by_future_return(df, config)
        elif config.method == LabelingMethod.BINARY:
            labels, peak_result = self._label_binary(df, config)
        else:
            # Alle anderen sind Peak-basiert
            labels, peak_result = self._label_by_peaks(df, config)

        self._log_label_distribution(labels)

        return LabelResult(
            labels=labels,
            peak_result=peak_result,
            num_classes=config.num_classes,
            method=config.method
        )

    def _label_by_peaks(
        self,
        df: pd.DataFrame,
        config: LabelingConfig
    ) -> Tuple[np.ndarray, PeakResult]:
        """
        Labelt basierend auf Peak-Erkennung.

        Args:
            df: DataFrame mit OHLCV-Daten
            config: LabelingConfig

        Returns:
            Tuple aus (labels, peak_result)
        """
        # Methode auf PeakMethod mappen
        method_map = {
            LabelingMethod.ZIGZAG: PeakMethod.ZIGZAG,
            LabelingMethod.PEAKS: PeakMethod.SCIPY_PEAKS,
            LabelingMethod.FRACTALS: PeakMethod.WILLIAMS_FRACTALS,
            LabelingMethod.PIVOTS: PeakMethod.PIVOT_POINTS,
            LabelingMethod.EXTREMA_DAILY: PeakMethod.DAILY_EXTREMA,
        }

        peak_method = method_map.get(config.method, PeakMethod.SCIPY_PEAKS)

        # PeakConfig erstellen
        peak_config = PeakConfig(
            method=peak_method,
            zigzag_threshold_pct=config.zigzag_threshold,
            peak_distance=config.peak_distance,
            peak_prominence_pct=config.peak_prominence,
            peak_width=int(config.peak_width) if config.peak_width else None,
            peak_height_pct=config.peak_height,
            peak_threshold_pct=config.peak_threshold,
            peak_plateau_size=config.peak_plateau_size,
            peak_wlen=config.peak_wlen,
            peak_rel_height=config.peak_rel_height,
            fractal_order=config.fractal_order,
            pivot_lookback=config.pivot_lookback,
        )

        # Peaks finden
        peak_result = self.peak_finder.find_peaks(df, peak_config)

        # Labels aus Peaks generieren
        n = len(df)
        labels_3class = np.zeros(n, dtype=np.int64)

        # Hochpunkte -> SELL (2 bei 3 Klassen)
        for idx in peak_result.high_indices:
            if 0 <= idx < n:
                labels_3class[idx] = self.LABELS_3['SELL']

        # Tiefpunkte -> BUY (1 bei 3 Klassen)
        for idx in peak_result.low_indices:
            if 0 <= idx < n:
                labels_3class[idx] = self.LABELS_3['BUY']

        # Bei 2 Klassen: Konvertieren
        if config.num_classes == 2:
            labels = self._convert_to_binary(labels_3class, df)
        else:
            labels = labels_3class

        return labels, peak_result

    def _label_by_future_return(
        self,
        df: pd.DataFrame,
        config: LabelingConfig
    ) -> Tuple[np.ndarray, Optional[PeakResult]]:
        """
        Labelt basierend auf zukuenftiger Rendite.

        3 Klassen:
        - BUY: Max-Rendite > threshold
        - SELL: Min-Rendite < -threshold
        - HOLD: Sonst

        2 Klassen (kein HOLD):
        - BUY: Max-Rendite > abs(Min-Rendite)
        - SELL: abs(Min-Rendite) >= Max-Rendite
        """
        n = len(df)
        labels = np.zeros(n, dtype=np.int64)
        close_prices = df['Close'].values

        lookforward = config.lookforward
        threshold = config.threshold_pct

        for i in range(n - lookforward):
            future_prices = close_prices[i + 1:i + 1 + lookforward]
            current_price = close_prices[i]

            max_return = ((future_prices.max() - current_price) / current_price) * 100
            min_return = ((future_prices.min() - current_price) / current_price) * 100

            if config.num_classes == 2:
                if max_return > abs(min_return):
                    labels[i] = self.LABELS['BUY']
                else:
                    labels[i] = self.LABELS['SELL']
            else:
                if max_return > threshold and max_return > abs(min_return):
                    labels[i] = self.LABELS['BUY']
                elif min_return < -threshold and abs(min_return) > max_return:
                    labels[i] = self.LABELS['SELL']
                # Sonst bleibt HOLD (0)

        return labels, None

    def _label_binary(
        self,
        df: pd.DataFrame,
        config: LabelingConfig
    ) -> Tuple[np.ndarray, Optional[PeakResult]]:
        """
        Binaeres Labeling: UP oder DOWN.

        - BUY: Naechster Close > aktueller Close
        - SELL: Naechster Close <= aktueller Close
        """
        n = len(df)
        self._set_label_mapping(2)
        labels = np.zeros(n, dtype=np.int64)

        close_prices = df['Close'].values

        for i in range(n - 1):
            if close_prices[i + 1] > close_prices[i]:
                labels[i] = self.LABELS['BUY']
            else:
                labels[i] = self.LABELS['SELL']

        return labels, None

    def _convert_to_binary(
        self,
        labels_3class: np.ndarray,
        df: pd.DataFrame
    ) -> np.ndarray:
        """
        Konvertiert 3-Klassen Labels zu 2-Klassen.

        Bei Extrema-basierten Methoden bleiben viele Punkte ohne Label (HOLD).
        Diese werden basierend auf dem Preistrend gelabelt.
        """
        n = len(labels_3class)
        close = df['Close'].values
        result = np.zeros(n, dtype=np.int64)

        # BUY (1 bei 3-Klassen) -> BUY (0 bei 2-Klassen)
        buy_indices = np.where(labels_3class == 1)[0]
        result[buy_indices] = 0

        # SELL (2 bei 3-Klassen) -> SELL (1 bei 2-Klassen)
        sell_indices = np.where(labels_3class == 2)[0]
        result[sell_indices] = 1

        # HOLD-Positionen basierend auf Preistrend fuellen
        for i in range(n):
            if labels_3class[i] == 0:  # War HOLD
                if i < n - 1:
                    if close[i + 1] > close[i]:
                        result[i] = 0  # BUY
                    else:
                        result[i] = 1  # SELL
                else:
                    result[i] = 0  # Letzter Punkt: Default BUY

        return result

    def _log_label_distribution(self, labels: np.ndarray) -> None:
        """Loggt die Label-Verteilung."""
        n = len(labels)
        unique, counts = np.unique(labels, return_counts=True)
        for u, c in zip(unique, counts):
            pct = c / n * 100
            label_name = self.LABEL_NAMES[u] if u < len(self.LABEL_NAMES) else f'Unknown({u})'
            self.logger.debug(f"[Labeler] {label_name}: {c} ({pct:.1f}%)")

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


# =============================================================================
# Legacy-Kompatibilitaet (fuer bestehenden Code)
# =============================================================================

class DailyExtremaLabeler(Labeler):
    """
    Legacy-Wrapper fuer Abwaertskompatibilitaet.

    DEPRECATED: Verwende stattdessen Labeler direkt.
    """

    def __init__(
        self,
        lookforward: int = 100,
        threshold_pct: float = 2.0,
        num_classes: int = 3
    ):
        super().__init__(num_classes=num_classes)
        self.lookforward = lookforward
        self.threshold_pct = threshold_pct

        # Legacy-Attribute fuer bestehenden Code
        self.buy_signal_indices: List[int] = []
        self.sell_signal_indices: List[int] = []

    def generate_labels(
        self,
        df: pd.DataFrame,
        method: Optional[str] = None,
        config: Optional[LabelingConfig] = None
    ) -> np.ndarray:
        """
        Legacy-Methode fuer Abwaertskompatibilitaet.

        Returns:
            np.ndarray (nur Labels, nicht LabelResult)
        """
        self.logger.debug(f"[Labeler] generate_labels aufgerufen: "
                         f"{len(df)} Zeilen, method={method}")

        # Config erstellen falls noetig
        if config is None:
            if method is None:
                method = 'peaks'

            method_map = {
                'future_return': LabelingMethod.FUTURE_RETURN,
                'zigzag': LabelingMethod.ZIGZAG,
                'peaks': LabelingMethod.PEAKS,
                'fractals': LabelingMethod.FRACTALS,
                'pivots': LabelingMethod.PIVOTS,
                'extrema_daily': LabelingMethod.EXTREMA_DAILY,
                'binary': LabelingMethod.BINARY,
            }

            config = LabelingConfig(
                method=method_map.get(method, LabelingMethod.PEAKS),
                lookforward=self.lookforward,
                threshold_pct=self.threshold_pct,
                num_classes=self.num_classes,
            )

        # Neuen Labeler aufrufen
        result = super().generate_labels(df, config)

        # Legacy-Attribute setzen
        self.buy_signal_indices = result.buy_indices
        self.sell_signal_indices = result.sell_indices

        return result.labels

    def find_daily_extrema(
        self,
        df: pd.DataFrame
    ) -> Tuple[List[any], List[any]]:
        """Legacy-Methode - verwendet jetzt PeakFinder."""
        from .peak_finder import PeakConfig, PeakMethod

        config = PeakConfig(method=PeakMethod.DAILY_EXTREMA)
        result = self.peak_finder.find_peaks(df, config)

        # Alte Extremum-Objekte simulieren (fuer Kompatibilitaet)
        @dataclass
        class Extremum:
            index: int
            timestamp: any
            price: float
            type: str

        highs = []
        for idx in result.high_indices:
            if 'DateTime' in df.columns:
                ts = df['DateTime'].iloc[idx]
            else:
                ts = df.index[idx]
            highs.append(Extremum(
                index=idx,
                timestamp=ts,
                price=df['High'].iloc[idx],
                type='HIGH'
            ))

        lows = []
        for idx in result.low_indices:
            if 'DateTime' in df.columns:
                ts = df['DateTime'].iloc[idx]
            else:
                ts = df.index[idx]
            lows.append(Extremum(
                index=idx,
                timestamp=ts,
                price=df['Low'].iloc[idx],
                type='LOW'
            ))

        return highs, lows

    # Legacy-Methoden delegieren an PeakFinder
    def find_zigzag_extrema(self, df, threshold_pct=5.0):
        config = PeakConfig(method=PeakMethod.ZIGZAG, zigzag_threshold_pct=threshold_pct)
        result = self.peak_finder.find_peaks(df, config)
        return result.high_indices, result.low_indices

    def find_peaks_extrema(self, df, config=None, prominence=0.5, distance=10):
        if config:
            peak_config = PeakConfig(
                method=PeakMethod.SCIPY_PEAKS,
                peak_distance=config.peak_distance,
                peak_prominence_pct=config.peak_prominence,
            )
        else:
            peak_config = PeakConfig(
                method=PeakMethod.SCIPY_PEAKS,
                peak_distance=distance,
                peak_prominence_pct=prominence,
            )
        result = self.peak_finder.find_peaks(df, peak_config)
        return result.high_indices, result.low_indices

    def find_williams_fractals(self, df, order=2):
        config = PeakConfig(method=PeakMethod.WILLIAMS_FRACTALS, fractal_order=order)
        result = self.peak_finder.find_peaks(df, config)
        return result.high_indices, result.low_indices

    def find_pivot_points(self, df, lookback=5):
        config = PeakConfig(method=PeakMethod.PIVOT_POINTS, pivot_lookback=lookback)
        result = self.peak_finder.find_peaks(df, config)
        return result.high_indices, result.low_indices


# Fuer Import-Kompatibilitaet
from dataclasses import dataclass as _dataclass

@_dataclass
class Extremum:
    """Legacy-Klasse fuer Abwaertskompatibilitaet."""
    index: int
    timestamp: any
    price: float
    type: str
