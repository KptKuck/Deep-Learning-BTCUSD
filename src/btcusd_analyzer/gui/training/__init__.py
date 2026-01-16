"""
Training GUI Module - Modulare Komponenten fuer das Training-Fenster.

Struktur:
- TrainingWindow: Hauptfenster (orchestriert alle Komponenten)
- ConfigPanel: Modell- und Training-Konfiguration
- VisualizationPanel: Plots, Metriken, Log
- AutoTrainerWidget: Auto-Training Ergebnisse
- TrainingWorker: Asynchrones Training in QThread
"""

from .training_window import TrainingWindow
from .config_panel import ConfigPanel
from .visualization_panel import VisualizationPanel
from .auto_trainer_widget import AutoTrainerWidget
from .training_worker import TrainingWorker

__all__ = [
    'TrainingWindow',
    'ConfigPanel',
    'VisualizationPanel',
    'AutoTrainerWidget',
    'TrainingWorker',
]
