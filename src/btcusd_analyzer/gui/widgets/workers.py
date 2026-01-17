"""
Worker-Threads fuer Hintergrund-Berechnungen in der GUI

Terminologie:
- PeakFinderWorker: Findet Peaks (Hochs/Tiefs) in den Daten
- LabelGeneratorWorker: Generiert Labels (BUY/SELL/HOLD) aus Peaks
"""

import numpy as np
import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal


class PeakFinderWorker(QThread):
    """
    Worker-Thread fuer Peak-Erkennung.

    Findet Hochs (Hochpunkte) und Tiefs (Tiefpunkte) in Preisdaten.
    Die Peaks sind reine Positionen (Indizes), keine Labels.

    Signals:
        result_ready: Emittiert wenn Peaks gefunden wurden
            - high_indices: Liste der Hochpunkt-Indizes
            - low_indices: Liste der Tiefpunkt-Indizes
            - peak_result: PeakResult Objekt mit Metadaten
        progress: Status-Meldung
        error: Fehlermeldung
    """

    result_ready = pyqtSignal(dict)  # {high_indices, low_indices, peak_result}
    progress = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, data: pd.DataFrame, config, parent=None):
        super().__init__(parent)
        self.data = data
        self.config = config

    def run(self):
        """Fuehrt die Peak-Erkennung im Hintergrund aus."""
        try:
            from ...training.peak_finder import PeakFinder, PeakConfig, PeakMethod
            from ...training.labeler import LabelingConfig, LabelingMethod
            from ...core.logger import get_logger

            logger = get_logger()

            self.progress.emit('Initialisiere PeakFinder...')

            # Config verarbeiten
            if isinstance(self.config, dict):
                config = LabelingConfig(**self.config)
            elif isinstance(self.config, LabelingConfig):
                config = self.config
            else:
                config = self.config

            # Methode auf PeakMethod mappen
            method_map = {
                LabelingMethod.FUTURE_RETURN: PeakMethod.SCIPY_PEAKS,  # Fallback
                LabelingMethod.ZIGZAG: PeakMethod.ZIGZAG,
                LabelingMethod.PEAKS: PeakMethod.SCIPY_PEAKS,
                LabelingMethod.FRACTALS: PeakMethod.WILLIAMS_FRACTALS,
                LabelingMethod.PIVOTS: PeakMethod.PIVOT_POINTS,
                LabelingMethod.EXTREMA_DAILY: PeakMethod.DAILY_EXTREMA,
                LabelingMethod.BINARY: PeakMethod.SCIPY_PEAKS,  # Fallback
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

            self.progress.emit('Suche Peaks...')

            # PeakFinder ausfuehren
            peak_finder = PeakFinder()
            peak_result = peak_finder.find_peaks(self.data, peak_config)

            self.progress.emit('Peaks gefunden!')

            # Ergebnis zurueckgeben - korrekte Terminologie
            high_indices = list(peak_result.high_indices)
            low_indices = list(peak_result.low_indices)

            logger.debug(f"[PeakFinderWorker] Gefunden: "
                        f"{len(high_indices)} Hochs, {len(low_indices)} Tiefs")

            result = {
                'high_indices': high_indices,  # Hochpunkte (SELL-Kandidaten)
                'low_indices': low_indices,    # Tiefpunkte (BUY-Kandidaten)
                'peak_result': peak_result,
                # Legacy-Attribute fuer Abwaertskompatibilitaet
                'buy_indices': low_indices,    # Tiefs -> BUY
                'sell_indices': high_indices,  # Hochs -> SELL
            }

            logger.debug('[PeakFinderWorker] Sende result_ready Signal...')
            self.result_ready.emit(result)
            logger.debug('[PeakFinderWorker] Signal gesendet')

        except Exception as e:
            import traceback
            error_msg = f'{str(e)}\n\n{traceback.format_exc()}'
            self.error.emit(error_msg)


class LabelGeneratorWorker(QThread):
    """
    Worker-Thread fuer Label-Generierung.

    Generiert Labels (BUY/SELL/HOLD) aus den gefundenen Peaks.
    Die Labels sind Klassifikationen fuer das Training.

    Signals:
        result_ready: Emittiert wenn Labels generiert wurden
            - labels: NumPy Array mit Labels
            - stats: Dictionary mit Label-Statistiken
            - label_result: LabelResult Objekt mit Metadaten
        progress: Status-Meldung
        error: Fehlermeldung
    """

    result_ready = pyqtSignal(dict)  # {labels, stats, label_result}
    progress = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(
        self,
        data: pd.DataFrame,
        high_indices: list,
        low_indices: list,
        num_classes: int = 3,
        parent=None
    ):
        super().__init__(parent)
        self.data = data
        self.high_indices = high_indices
        self.low_indices = low_indices
        self.num_classes = num_classes

    def run(self):
        """Generiert Labels im Hintergrund."""
        try:
            from ...core.logger import get_logger

            logger = get_logger()

            self.progress.emit('Generiere Labels...')

            n = len(self.data)

            if self.num_classes == 3:
                # HOLD=0, BUY=1, SELL=2
                labels = np.zeros(n, dtype=np.int32)

                # Tiefpunkte -> BUY
                for idx in self.low_indices:
                    if 0 <= idx < n:
                        labels[idx] = 1  # BUY

                # Hochpunkte -> SELL
                for idx in self.high_indices:
                    if 0 <= idx < n:
                        labels[idx] = 2  # SELL
            else:
                # BUY=0, SELL=1 (nur an Peak-Positionen, Rest=-1)
                labels = np.full(n, -1, dtype=np.int32)

                # Tiefpunkte -> BUY
                for idx in self.low_indices:
                    if 0 <= idx < n:
                        labels[idx] = 0  # BUY

                # Hochpunkte -> SELL
                for idx in self.high_indices:
                    if 0 <= idx < n:
                        labels[idx] = 1  # SELL

            # Statistiken berechnen
            if self.num_classes == 3:
                num_buy = int(np.sum(labels == 1))
                num_sell = int(np.sum(labels == 2))
                num_hold = int(np.sum(labels == 0))
            else:
                num_buy = int(np.sum(labels == 0))
                num_sell = int(np.sum(labels == 1))
                num_hold = 0

            stats = {
                'num_buy': num_buy,
                'num_sell': num_sell,
                'num_hold': num_hold,
                'total': num_buy + num_sell + num_hold,
                'num_classes': self.num_classes,
            }

            self.progress.emit('Labels generiert!')

            logger.debug(f"[LabelGeneratorWorker] Labels: "
                        f"{num_buy} BUY, {num_sell} SELL, {num_hold} HOLD")

            result = {
                'labels': labels,
                'stats': stats,
            }
            self.result_ready.emit(result)

        except Exception as e:
            import traceback
            error_msg = f'{str(e)}\n\n{traceback.format_exc()}'
            self.error.emit(error_msg)
