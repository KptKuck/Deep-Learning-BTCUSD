"""
Worker-Threads fuer Hintergrund-Berechnungen in der GUI
"""

import numpy as np
import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal


class PeakFinderWorker(QThread):
    """Worker-Thread fuer Peak-Erkennung."""

    finished = pyqtSignal(dict)  # {buy_indices, sell_indices, labeler}
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
            result = {
                'buy_indices': labeler.buy_signal_indices.copy(),
                'sell_indices': labeler.sell_signal_indices.copy(),
                'labeler': labeler
            }
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))


class LabelGeneratorWorker(QThread):
    """Worker-Thread fuer Label-Generierung."""

    finished = pyqtSignal(dict)  # {labels, stats}
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
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))
