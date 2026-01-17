"""Training Module - Datenaufbereitung fuer Training

Terminologie:
- Peak = Position im Chart (Index eines Hochs/Tiefs) -> PeakFinder
- Label = Klassifikation fuer Training (BUY/SELL/HOLD) -> Labeler
"""

# Peak-Erkennung (Rohpositionen)
from .peak_finder import (
    PeakFinder,
    PeakConfig,
    PeakMethod,
    PeakResult,
)

# Label-Generierung (Klassifikationen)
from .labeler import (
    Labeler,
    LabelingConfig,
    LabelingMethod,
    LabelResult,
    DailyExtremaLabeler,  # Legacy
)

from .sequence import (
    SequenceGenerator,
    expand_labels_lookahead,
    compute_class_weights,
)
from .normalizer import ZScoreNormalizer
from .auto_trainer import (
    AutoTrainer,
    AutoTrainResult,
    AUTO_TRAINER_CONFIGS,
    get_complexity_info
)

__all__ = [
    # Peak-Erkennung
    'PeakFinder',
    'PeakConfig',
    'PeakMethod',
    'PeakResult',
    # Label-Generierung
    'Labeler',
    'LabelingConfig',
    'LabelingMethod',
    'LabelResult',
    'DailyExtremaLabeler',  # Legacy
    # Sequenzen
    'SequenceGenerator',
    'expand_labels_lookahead',
    'compute_class_weights',
    # Normalisierung
    'ZScoreNormalizer',
    # Auto-Training
    'AutoTrainer',
    'AutoTrainResult',
    'AUTO_TRAINER_CONFIGS',
    'get_complexity_info',
]
