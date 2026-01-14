"""
Worker-Threads fuer Hintergrund-Berechnungen in der GUI
"""

import numpy as np
import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal


class PeakFinderWorker(QThread):
    """Worker-Thread fuer Peak-Erkennung."""

    # Umbenennt um Konflikt mit QThread.finished zu vermeiden
    result_ready = pyqtSignal(dict)  # {buy_indices, sell_indices, labeler}
    progress = pyqtSignal(str)   # Status-Meldung
    error = pyqtSignal(str)      # Fehlermeldung

    def __init__(self, data: pd.DataFrame, config: dict, parent=None):
        super().__init__(parent)
        self.data = data
        self.config = config

    def run(self):
        """Fuehrt die Peak-Erkennung im Hintergrund aus."""
        try:
            from ...training.labeler import DailyExtremaLabeler, LabelingConfig
            from ...core.logger import get_logger

            logger = get_logger()

            self.progress.emit('Initialisiere Labeler...')

            # Config erstellen (bereits als LabelingConfig oder dict)
            if isinstance(self.config, dict):
                config = LabelingConfig(**self.config)
            else:
                config = self.config

            # Labeler erstellen
            labeler = DailyExtremaLabeler(
                lookforward=config.lookforward,
                threshold_pct=config.threshold_pct,
                num_classes=3
            )

            self.progress.emit('Suche Peaks...')

            # Labels generieren (findet auch Peaks)
            _ = labeler.generate_labels(self.data, config=config)

            self.progress.emit('Peaks gefunden!')

            # Ergebnis zurueckgeben
            buy_indices = list(labeler.buy_signal_indices)
            sell_indices = list(labeler.sell_signal_indices)

            logger.debug(f'[PeakFinderWorker] Erstelle Ergebnis: {len(buy_indices)} BUY, {len(sell_indices)} SELL')

            result = {
                'buy_indices': buy_indices,
                'sell_indices': sell_indices,
                'labeler': labeler
            }

            logger.debug('[PeakFinderWorker] Sende result_ready Signal...')
            self.result_ready.emit(result)
            logger.debug('[PeakFinderWorker] Signal gesendet')

        except Exception as e:
            import traceback
            error_msg = f'{str(e)}\n\n{traceback.format_exc()}'
            self.error.emit(error_msg)


class LabelGeneratorWorker(QThread):
    """Worker-Thread fuer Label-Generierung."""

    # Umbenennt um Konflikt mit QThread.finished zu vermeiden
    result_ready = pyqtSignal(dict)  # {labels, stats}
    progress = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, labeler, data: pd.DataFrame, num_classes: int, parent=None):
        super().__init__(parent)
        self.labeler = labeler
        self.data = data
        self.num_classes = num_classes

    def run(self):
        """Generiert Labels im Hintergrund."""
        try:
            self.progress.emit('Generiere Labels...')

            # Labeler mit neuer Klassenanzahl konfigurieren
            self.labeler.num_classes = self.num_classes

            # Labels generieren
            labels = self.labeler.generate_labels(self.data)

            # Statistiken berechnen
            unique, counts = np.unique(labels[labels >= 0], return_counts=True)
            stats = dict(zip(unique.astype(int), counts.astype(int)))

            self.progress.emit('Labels generiert!')

            result = {
                'labels': labels,
                'stats': stats,
                'labeler': self.labeler
            }
            self.result_ready.emit(result)

        except Exception as e:
            self.error.emit(str(e))
