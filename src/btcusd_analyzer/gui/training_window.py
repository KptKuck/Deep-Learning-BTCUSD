"""
Training Window - Re-Export aus dem modularen training/ Paket.

Die eigentliche Implementierung befindet sich in:
- gui/training/training_window.py  (Hauptfenster)
- gui/training/config_panel.py     (Konfiguration)
- gui/training/visualization_panel.py (Plots, Log)
- gui/training/auto_trainer_widget.py (Auto-Trainer)
- gui/training/training_worker.py  (QThread)
"""

# Re-Export fuer Rueckwaertskompatibilitaet
from .training.training_window import TrainingWindow

__all__ = ['TrainingWindow']
