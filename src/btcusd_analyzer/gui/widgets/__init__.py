"""
GUI Widgets - Wiederverwendbare UI-Komponenten

Dieses Modul enthaelt ausgelagerte Widgets fuer die GUI:
- FeatureCategoryWidget: Feature-Auswahl nach Kategorien
- PeakFinderWorker: Worker fuer Hintergrund-Peak-Erkennung
- LabelGeneratorWorker: Worker fuer Hintergrund-Label-Generierung
- FeatureCache: Cache fuer berechnete Features
- PipelineState: Status-Management fuer Datenvorbereitungs-Pipeline
"""

from .webserver_control import WebServerControl
from .feature_widgets import (
    FeatureDefinition,
    FeatureCategoryWidget,
    FEATURE_REGISTRY,
    CATEGORY_CONFIG,
)
from .pipeline_state import PipelineState, PipelineStage
from .workers import PeakFinderWorker, LabelGeneratorWorker
from .feature_cache import FeatureCache

__all__ = [
    'WebServerControl',
    'FeatureDefinition',
    'FeatureCategoryWidget',
    'FEATURE_REGISTRY',
    'CATEGORY_CONFIG',
    'PipelineState',
    'PipelineStage',
    'PeakFinderWorker',
    'LabelGeneratorWorker',
    'FeatureCache',
]
