"""
Prepare Data Window - Trainingsdaten Vorbereitung
Refactored: 4 Tabs mit eigenem Chart pro Tab
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Dict, Any, List, Tuple

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox, QScrollArea, QFrame,
    QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox, QSplitter,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView, QSlider
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QFont

import pandas as pd
import numpy as np

from .chart_widget import ChartWidget
from .styles import StyleFactory, COLORS


# =============================================================================
# Feature-Definitionen
# =============================================================================

@dataclass
class FeatureDefinition:
    """Definition eines einzelnen Features mit optionalen Parametern."""
    name: str                    # Feature-Name (z.B. "SMA", "RSI")
    display_name: str            # Anzeigename (z.B. "Simple MA")
    category: str                # Kategorie (price, technical, etc.)
    default_enabled: bool = False
    params: Dict[str, Any] = field(default_factory=dict)
    param_ranges: Dict[str, tuple] = field(default_factory=dict)  # (min, max, step)


# Feature-Registry mit allen verfuegbaren Features
FEATURE_REGISTRY: Dict[str, List[FeatureDefinition]] = {
    'price': [
        FeatureDefinition('Open', 'Open', 'price', default_enabled=True),
        FeatureDefinition('High', 'High', 'price', default_enabled=True),
        FeatureDefinition('Low', 'Low', 'price', default_enabled=True),
        FeatureDefinition('Close', 'Close', 'price', default_enabled=True),
        FeatureDefinition('PriceChange', 'Price Change', 'price', default_enabled=True),
        FeatureDefinition('PriceChangePct', 'Price Change %', 'price', default_enabled=True),
        FeatureDefinition('Range', 'Range (H-L)', 'price'),
        FeatureDefinition('RangePct', 'Range %', 'price'),
        FeatureDefinition('TypicalPrice', 'Typical Price', 'price'),
        FeatureDefinition('OHLC4', 'OHLC/4', 'price'),
    ],
    'technical': [
        FeatureDefinition('SMA', 'SMA', 'technical',
                         params={'period': 20},
                         param_ranges={'period': (5, 200, 1)}),
        FeatureDefinition('EMA', 'EMA', 'technical',
                         params={'period': 20},
                         param_ranges={'period': (5, 200, 1)}),
        FeatureDefinition('RSI', 'RSI', 'technical',
                         params={'period': 14},
                         param_ranges={'period': (5, 50, 1)}),
        FeatureDefinition('MACD', 'MACD', 'technical'),
        FeatureDefinition('MACD_Signal', 'MACD Signal', 'technical'),
        FeatureDefinition('MACD_Hist', 'MACD Hist', 'technical'),
        FeatureDefinition('BB_Upper', 'BB Upper', 'technical',
                         params={'period': 20},
                         param_ranges={'period': (10, 50, 1)}),
        FeatureDefinition('BB_Lower', 'BB Lower', 'technical',
                         params={'period': 20},
                         param_ranges={'period': (10, 50, 1)}),
        FeatureDefinition('BB_Width', 'BB Width', 'technical',
                         params={'period': 20},
                         param_ranges={'period': (10, 50, 1)}),
    ],
    'volatility': [
        FeatureDefinition('ATR', 'ATR', 'volatility',
                         params={'period': 14},
                         param_ranges={'period': (5, 50, 1)}),
        FeatureDefinition('ATR_Pct', 'ATR %', 'volatility',
                         params={'period': 14},
                         param_ranges={'period': (5, 50, 1)}),
        FeatureDefinition('RollingStd', 'Rolling Std', 'volatility',
                         params={'period': 20},
                         param_ranges={'period': (5, 100, 1)}),
        FeatureDefinition('RollingStd_Pct', 'Rolling Std %', 'volatility',
                         params={'period': 20},
                         param_ranges={'period': (5, 100, 1)}),
        FeatureDefinition('HighLowRange', 'H-L Range %', 'volatility'),
        FeatureDefinition('ReturnVol', 'Return Vol', 'volatility',
                         params={'period': 20},
                         param_ranges={'period': (5, 100, 1)}),
        FeatureDefinition('ParkinsonVol', 'Parkinson Vol', 'volatility',
                         params={'period': 20},
                         param_ranges={'period': (5, 100, 1)}),
    ],
    'volume': [
        FeatureDefinition('Volume', 'Volume', 'volume'),
        FeatureDefinition('RelativeVolume', 'Relative Vol', 'volume'),
    ],
    'time': [
        FeatureDefinition('hour_sin', 'Hour (sin)', 'time'),
        FeatureDefinition('hour_cos', 'Hour (cos)', 'time'),
    ],
}

# Kategorie-Anzeigenamen und Farben
CATEGORY_CONFIG = {
    'price': {'name': 'Preis-Features', 'color': '#4da8da'},
    'technical': {'name': 'Technische Indikatoren', 'color': '#b19cd9'},
    'volatility': {'name': 'Volatilitaet', 'color': '#ff9966'},
    'volume': {'name': 'Volumen', 'color': '#7fe6b3'},
    'time': {'name': 'Zeit-Features', 'color': '#e6b333'},
}


# =============================================================================
# Worker-Threads fuer Hintergrund-Berechnungen
# =============================================================================

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
            from ..training.labeler import DailyExtremaLabeler, LabelingConfig

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


# =============================================================================
# Feature-Cache fuer optimierte Berechnung
# =============================================================================

class FeatureCache:
    """
    Intelligenter Cache fuer Feature-Berechnungen.
    Cached berechnete Features und berechnet nur bei Aenderungen neu.
    """

    def __init__(self):
        self._cache: Dict[str, np.ndarray] = {}
        self._data_id: Optional[int] = None  # ID der Daten (id(df))

    def invalidate(self):
        """Leert den gesamten Cache."""
        self._cache.clear()
        self._data_id = None

    def set_data(self, data: pd.DataFrame):
        """Setzt neue Basisdaten und invalidiert Cache wenn noetig."""
        new_id = id(data)
        if new_id != self._data_id:
            self._cache.clear()
            self._data_id = new_id

    def get(self, key: str) -> Optional[np.ndarray]:
        """Holt gecachtes Feature oder None."""
        return self._cache.get(key)

    def set(self, key: str, values: np.ndarray):
        """Speichert berechnetes Feature."""
        self._cache[key] = values

    def has(self, key: str) -> bool:
        """Prueft ob Feature gecached ist."""
        return key in self._cache

    def compute_or_get(self, key: str, compute_fn) -> np.ndarray:
        """Berechnet Feature falls nicht gecached."""
        if key not in self._cache:
            self._cache[key] = compute_fn()
        return self._cache[key]

    @property
    def cached_features(self) -> Dict[str, np.ndarray]:
        """Gibt alle gecachten Features zurueck."""
        return self._cache.copy()

    def compute_feature(self, data: pd.DataFrame, feat_name: str,
                        feat_params: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Berechnet ein einzelnes Feature mit Caching.

        Args:
            data: DataFrame mit OHLCV Daten
            feat_name: Name des Features
            feat_params: Parameter (z.B. {'period': 20})

        Returns:
            numpy array mit Feature-Werten oder None
        """
        # Cache-Key mit Parametern
        param_str = '_'.join(f'{k}{v}' for k, v in sorted(feat_params.items()))
        cache_key = f'{feat_name}_{param_str}' if param_str else feat_name

        # Falls gecached, zurueckgeben
        if self.has(cache_key):
            return self.get(cache_key)

        # Spalten-Namen mapping (case-insensitive)
        col_map = {col.lower(): col for col in data.columns}
        close_col = col_map.get('close', 'Close')
        high_col = col_map.get('high', 'High')
        low_col = col_map.get('low', 'Low')
        open_col = col_map.get('open', 'Open')

        result = None
        feat_lower = feat_name.lower()

        # Basis-Features aus Daten
        if feat_lower in col_map:
            result = data[col_map[feat_lower]].values

        # Preis-Features
        elif feat_name == 'PriceChange':
            if close_col in data.columns:
                result = data[close_col].diff().values
        elif feat_name == 'PriceChangePct':
            if close_col in data.columns:
                result = data[close_col].pct_change().values * 100
        elif feat_name == 'Range':
            if high_col in data.columns and low_col in data.columns:
                result = (data[high_col] - data[low_col]).values
        elif feat_name == 'RangePct':
            if all(c in data.columns for c in [high_col, low_col, close_col]):
                result = ((data[high_col] - data[low_col]) / data[close_col] * 100).values
        elif feat_name == 'TypicalPrice':
            if all(c in data.columns for c in [high_col, low_col, close_col]):
                result = ((data[high_col] + data[low_col] + data[close_col]) / 3).values
        elif feat_name == 'OHLC4':
            if all(c in data.columns for c in [open_col, high_col, low_col, close_col]):
                result = ((data[open_col] + data[high_col] +
                          data[low_col] + data[close_col]) / 4).values

        # Technische Indikatoren
        elif feat_name == 'SMA':
            period = feat_params.get('period', 20)
            if close_col in data.columns:
                result = data[close_col].rolling(period).mean().values
        elif feat_name == 'EMA':
            period = feat_params.get('period', 20)
            if close_col in data.columns:
                result = data[close_col].ewm(span=period).mean().values
        elif feat_name == 'RSI':
            period = feat_params.get('period', 14)
            if close_col in data.columns:
                delta = data[close_col].diff()
                gain = delta.where(delta > 0, 0).rolling(period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
                rs = gain / loss.replace(0, np.nan)
                result = (100 - 100 / (1 + rs)).values

        # Volatilitaet
        elif feat_name == 'ATR':
            period = feat_params.get('period', 14)
            if all(c in data.columns for c in [high_col, low_col, close_col]):
                tr = np.maximum(
                    data[high_col] - data[low_col],
                    np.maximum(
                        abs(data[high_col] - data[close_col].shift(1)),
                        abs(data[low_col] - data[close_col].shift(1))
                    )
                )
                result = tr.rolling(period).mean().values
        elif feat_name == 'ATR_Pct':
            period = feat_params.get('period', 14)
            if all(c in data.columns for c in [high_col, low_col, close_col]):
                tr = np.maximum(
                    data[high_col] - data[low_col],
                    np.maximum(
                        abs(data[high_col] - data[close_col].shift(1)),
                        abs(data[low_col] - data[close_col].shift(1))
                    )
                )
                atr = tr.rolling(period).mean()
                result = (atr / data[close_col] * 100).values
        elif feat_name == 'RollingStd':
            period = feat_params.get('period', 20)
            if close_col in data.columns:
                result = data[close_col].rolling(period).std().values
        elif feat_name == 'RollingStd_Pct':
            period = feat_params.get('period', 20)
            if close_col in data.columns:
                std = data[close_col].rolling(period).std()
                result = (std / data[close_col] * 100).values

        # Cachen und zurueckgeben
        if result is not None:
            self.set(cache_key, result)
        return result


# =============================================================================
# PipelineState - Zustandsverwaltung fuer die 4-Tab Pipeline
# =============================================================================

class PipelineStage(IntEnum):
    """Stufen der Datenvorbereitungs-Pipeline."""
    NONE = 0
    PEAKS = 1       # Tab 1: Peaks gefunden
    LABELS = 2      # Tab 2: Labels generiert
    FEATURES = 3    # Tab 3: Features ausgewaehlt
    SAMPLES = 4     # Tab 4: Samples berechnet


class PipelineState:
    """
    Verwaltet den Zustand der 4-Tab Pipeline.
    Ersetzt die 4 einzelnen Boolean-Flags durch einen konsistenten Zustand.
    """

    def __init__(self):
        self._current_stage = PipelineStage.NONE

    @property
    def current_stage(self) -> PipelineStage:
        """Gibt die aktuelle Pipeline-Stufe zurueck."""
        return self._current_stage

    def advance_to(self, stage: PipelineStage):
        """Setzt die Pipeline auf eine bestimmte Stufe (wenn hoeher als aktuell)."""
        if stage > self._current_stage:
            self._current_stage = stage

    def set_stage(self, stage: PipelineStage):
        """Setzt die Pipeline auf eine bestimmte Stufe (direkt)."""
        self._current_stage = stage

    def invalidate_from(self, stage: PipelineStage):
        """
        Invalidiert alle Stufen ab der angegebenen Stufe.
        Wenn z.B. Peaks neu gesucht werden, sind Labels/Features/Samples ungueltig.
        """
        if self._current_stage >= stage:
            # Setze auf die Stufe davor
            self._current_stage = PipelineStage(max(0, stage - 1))

    def is_valid(self, stage: PipelineStage) -> bool:
        """Prueft ob eine bestimmte Stufe erreicht wurde."""
        return self._current_stage >= stage

    def reset(self):
        """Setzt die Pipeline zurueck."""
        self._current_stage = PipelineStage.NONE

    # Convenience Properties fuer Rueckwaertskompatibilitaet
    @property
    def peaks_valid(self) -> bool:
        return self.is_valid(PipelineStage.PEAKS)

    @property
    def labels_valid(self) -> bool:
        return self.is_valid(PipelineStage.LABELS)

    @property
    def features_valid(self) -> bool:
        return self.is_valid(PipelineStage.FEATURES)

    @property
    def samples_valid(self) -> bool:
        return self.is_valid(PipelineStage.SAMPLES)


# =============================================================================
# FeatureCategoryWidget
# =============================================================================

class FeatureCategoryWidget(QGroupBox):
    """Widget fuer eine Feature-Kategorie mit Checkboxen und Parametern."""

    feature_changed = pyqtSignal()

    def __init__(self, category_key: str, features: List[FeatureDefinition],
                 color: str, parent=None):
        category_name = CATEGORY_CONFIG.get(category_key, {}).get('name', category_key)
        super().__init__(f"{category_name} ({len(features)})", parent)

        self.category_key = category_key
        self.features = features
        self.color = color
        self.feature_widgets: Dict[str, Dict[str, Any]] = {}

        self._init_ui()

    def _init_ui(self):
        """Initialisiert die UI-Komponenten."""
        self.setFont(QFont('Segoe UI', 10, QFont.Weight.Bold))
        self.setStyleSheet(self._get_group_style())

        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(8, 12, 8, 8)

        # "Alle" Checkbox
        self.all_checkbox = QCheckBox('Alle')
        self.all_checkbox.setStyleSheet('color: white; font-weight: bold;')
        self.all_checkbox.stateChanged.connect(self._on_all_toggled)
        layout.addWidget(self.all_checkbox)

        # Trennlinie
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet('background-color: #444;')
        layout.addWidget(line)

        # Features
        for feat_def in self.features:
            feat_widget = self._create_feature_row(feat_def)
            layout.addLayout(feat_widget)

        # Initial "Alle" Status aktualisieren
        self._update_all_checkbox_state()

    def _create_feature_row(self, feat_def: FeatureDefinition) -> QHBoxLayout:
        """Erstellt eine Zeile fuer ein Feature mit Checkbox und optionalen Parametern."""
        row = QHBoxLayout()
        row.setSpacing(5)

        # Checkbox
        cb = QCheckBox(feat_def.display_name)
        cb.setChecked(feat_def.default_enabled)
        cb.setStyleSheet('color: white;')
        cb.setMinimumWidth(100)
        cb.stateChanged.connect(self._on_feature_toggled)
        row.addWidget(cb)

        # Parameter-Widgets speichern
        widget_data: Dict[str, Any] = {'checkbox': cb, 'params': {}}

        # Parameter-SpinBoxen (falls vorhanden)
        if feat_def.params:
            for param_name, default_val in feat_def.params.items():
                # Label
                label = QLabel(f'{param_name}:')
                label.setStyleSheet('color: #aaa; font-size: 10px;')
                row.addWidget(label)

                # SpinBox
                spin = QSpinBox()
                spin.setFixedWidth(60)
                spin.setStyleSheet('''
                    QSpinBox {
                        background-color: #3a3a3a;
                        border: 1px solid #555;
                        border-radius: 3px;
                        color: white;
                        padding: 2px;
                    }
                ''')

                # Range setzen
                if param_name in feat_def.param_ranges:
                    min_val, max_val, step = feat_def.param_ranges[param_name]
                    spin.setRange(int(min_val), int(max_val))
                    spin.setSingleStep(int(step))
                else:
                    spin.setRange(1, 500)

                spin.setValue(int(default_val))
                spin.valueChanged.connect(self._on_param_changed)
                row.addWidget(spin)

                widget_data['params'][param_name] = spin

        row.addStretch()
        self.feature_widgets[feat_def.name] = widget_data

        return row

    def _on_all_toggled(self, state: int):
        """Wird aufgerufen wenn 'Alle' Checkbox geaendert wird."""
        checked = state == Qt.CheckState.Checked.value
        for feat_name, widgets in self.feature_widgets.items():
            widgets['checkbox'].blockSignals(True)
            widgets['checkbox'].setChecked(checked)
            widgets['checkbox'].blockSignals(False)
        self.feature_changed.emit()

    def _on_feature_toggled(self):
        """Wird aufgerufen wenn ein Feature-Checkbox geaendert wird."""
        self._update_all_checkbox_state()
        self.feature_changed.emit()

    def _on_param_changed(self):
        """Wird aufgerufen wenn ein Parameter geaendert wird."""
        self.feature_changed.emit()

    def _update_all_checkbox_state(self):
        """Aktualisiert den Status der 'Alle' Checkbox."""
        all_checked = all(w['checkbox'].isChecked() for w in self.feature_widgets.values())
        none_checked = not any(w['checkbox'].isChecked() for w in self.feature_widgets.values())

        self.all_checkbox.blockSignals(True)
        if all_checked:
            self.all_checkbox.setCheckState(Qt.CheckState.Checked)
        elif none_checked:
            self.all_checkbox.setCheckState(Qt.CheckState.Unchecked)
        else:
            self.all_checkbox.setCheckState(Qt.CheckState.PartiallyChecked)
        self.all_checkbox.blockSignals(False)

    def get_selected_features(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Gibt alle ausgewaehlten Features mit ihren Parametern zurueck."""
        selected = []
        for feat_name, widgets in self.feature_widgets.items():
            if widgets['checkbox'].isChecked():
                params = {}
                for param_name, spin in widgets['params'].items():
                    params[param_name] = spin.value()
                selected.append((feat_name, params))
        return selected

    def get_selected_count(self) -> int:
        """Gibt die Anzahl der ausgewaehlten Features zurueck."""
        return sum(1 for w in self.feature_widgets.values() if w['checkbox'].isChecked())

    def _get_group_style(self) -> str:
        """Gibt das GroupBox-Stylesheet zurueck."""
        return f'''
            QGroupBox {{
                color: {self.color};
                border: 1px solid #333;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        '''


# =============================================================================
# PrepareDataWindow
# =============================================================================

class PrepareDataWindow(QMainWindow):
    """
    Fenster zur Vorbereitung der Trainingsdaten.

    4 Tabs mit eigenem Chart:
    - Tab 1: Find Peaks - Peak-Erkennung
    - Tab 2: Set Labels - Label-Generierung
    - Tab 3: Features - Feature-Auswahl
    - Tab 4: Samples - Sample-Generierung
    """

    # Signal wenn Daten vorbereitet wurden
    data_prepared = pyqtSignal(object, object)  # (training_data, training_info)
    # Signal fuer Log-Meldungen an MainWindow
    log_message = pyqtSignal(str, str)  # message, level

    def __init__(self, data: pd.DataFrame, parent=None):
        super().__init__(parent)
        self._parent = parent

        self.data = data
        self.total_points = len(data)

        # Parameter
        self.params = {
            'lookback': 100,
            'lookforward': 10,
            'include_hold': True,
            'hold_ratio': 1.0,
            'min_distance_factor': 0.3,
            'auto_gen': True,
            'random_seed': 42,
            'normalize_method': 'zscore',
        }

        # Ergebnis-Variablen
        self.detected_peaks: Dict[str, List[int]] = {'buy_indices': [], 'sell_indices': []}
        self.generated_labels: Optional[np.ndarray] = None
        self.selected_features: List[str] = []
        self.result_info: Dict[str, Any] = {}
        self.current_num_classes = 3

        # Tab-Validierungsstatus (Pipeline)
        self.peaks_valid = False       # Tab 1: Peaks gefunden?
        self.labels_valid = False      # Tab 2: Labels generiert?
        self.features_valid = False    # Tab 3: Features gewaehlt?
        self.samples_valid = False     # Tab 4: Samples berechnet?

        # Feature-Cache initialisieren
        self._feature_cache = FeatureCache()
        self._feature_cache.set_data(data)

        # Worker-Referenzen
        self._peak_worker: Optional[PeakFinderWorker] = None
        self._label_worker: Optional[LabelGeneratorWorker] = None

        # UI initialisieren
        self._init_ui()
        self._update_dynamic_limits()

    def _log(self, message: str, level: str = 'INFO'):
        """Loggt eine Nachricht an MainWindow."""
        if self._parent and hasattr(self._parent, '_log'):
            self._parent._log(f'[Prepare] {message}', level)
        self.log_message.emit(message, level)

    def _init_ui(self):
        """Initialisiert die UI-Komponenten."""
        self.setWindowTitle('Trainingsdaten Vorbereitung')
        self.setMinimumSize(1600, 950)
        self.setStyleSheet(self._get_stylesheet())

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main Layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Titel
        title = QLabel('Trainingsdaten Vorbereitung')
        title.setFont(QFont('Segoe UI', 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet('color: white; padding: 10px;')
        main_layout.addWidget(title)

        # Tab Widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(self._get_tab_stylesheet())
        main_layout.addWidget(self.tab_widget)

        # Tab 1: Find Peaks
        tab1 = self._create_find_peaks_tab()
        self.tab_widget.addTab(tab1, "Find Peaks")

        # Tab 2: Set Labels
        tab2 = self._create_set_labels_tab()
        self.tab_widget.addTab(tab2, "Set Labels")

        # Tab 3: Features
        tab3 = self._create_features_tab()
        self.tab_widget.addTab(tab3, "Features")

        # Tab 4: Samples
        tab4 = self._create_samples_tab()
        self.tab_widget.addTab(tab4, "Samples")

        # Initial: Tabs 2, 3, 4 deaktiviert
        self.tab_widget.setTabEnabled(1, False)
        self.tab_widget.setTabEnabled(2, False)
        self.tab_widget.setTabEnabled(3, False)

        # Status Label
        self.status_label = QLabel('1. Peak-Methode waehlen und Peaks finden')
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet('color: #aaaaaa; padding: 10px; font-size: 12px;')
        main_layout.addWidget(self.status_label)

    # =========================================================================
    # Tab 1: Find Peaks
    # =========================================================================

    def _create_find_peaks_tab(self) -> QWidget:
        """Erstellt Tab 1: Peak-Erkennung."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)

        # Splitter: Links Parameter | Rechts Chart
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Linke Seite: Parameter
        params_widget = self._create_find_peaks_params()
        splitter.addWidget(params_widget)

        # Rechte Seite: Chart
        self.peaks_chart = ChartWidget('Peaks')
        splitter.addWidget(self.peaks_chart)

        # Splitter-Proportionen
        splitter.setSizes([400, 1000])

        return tab

    def _create_find_peaks_params(self) -> QWidget:
        """Erstellt die Parameter-Seite fuer Tab 1."""
        widget = QWidget()
        widget.setMinimumWidth(380)
        widget.setMaximumWidth(450)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setSpacing(10)

        # Peak-Methode Gruppe
        method_group = QGroupBox('Peak-Methode')
        method_group.setFont(QFont('Segoe UI', 11, QFont.Weight.Bold))
        method_group.setStyleSheet(self._group_style('#4da8da'))
        method_layout = QVBoxLayout(method_group)

        method_row = QHBoxLayout()
        method_row.addWidget(QLabel('Methode:'))
        self.peak_method_combo = QComboBox()
        self.peak_method_combo.addItems([
            'Future Return',
            'ZigZag',
            'Peak Detection',
            'Williams Fractals',
            'Pivot Points',
            'Tages-Extrema',
            'Binary'
        ])
        self.peak_method_combo.currentIndexChanged.connect(self._on_peak_method_changed)
        method_row.addWidget(self.peak_method_combo)
        method_layout.addLayout(method_row)

        layout.addWidget(method_group)

        # Gemeinsame Parameter
        common_group = QGroupBox('Gemeinsame Parameter')
        common_group.setFont(QFont('Segoe UI', 11, QFont.Weight.Bold))
        common_group.setStyleSheet(self._group_style('#7fe6b3'))
        common_layout = QGridLayout(common_group)

        # Lookforward
        common_layout.addWidget(QLabel('Lookforward:'), 0, 0)
        self.peak_lookforward_spin = QSpinBox()
        self.peak_lookforward_spin.setRange(1, 500)
        self.peak_lookforward_spin.setValue(100)
        common_layout.addWidget(self.peak_lookforward_spin, 0, 1)

        # Threshold
        common_layout.addWidget(QLabel('Threshold %:'), 1, 0)
        self.peak_threshold_spin = QDoubleSpinBox()
        self.peak_threshold_spin.setRange(0.1, 20.0)
        self.peak_threshold_spin.setValue(2.0)
        self.peak_threshold_spin.setSingleStep(0.1)
        common_layout.addWidget(self.peak_threshold_spin, 1, 1)

        layout.addWidget(common_group)

        # Methoden-spezifische Parameter
        self.method_params_group = QGroupBox('Methoden-Parameter')
        self.method_params_group.setFont(QFont('Segoe UI', 11, QFont.Weight.Bold))
        self.method_params_group.setStyleSheet(self._group_style('#ffb366'))
        method_params_layout = QGridLayout(self.method_params_group)

        # ZigZag Threshold
        self.zigzag_label = QLabel('ZigZag Threshold %:')
        method_params_layout.addWidget(self.zigzag_label, 0, 0)
        self.zigzag_threshold_spin = QDoubleSpinBox()
        self.zigzag_threshold_spin.setRange(1.0, 20.0)
        self.zigzag_threshold_spin.setValue(5.0)
        self.zigzag_threshold_spin.setSingleStep(0.5)
        method_params_layout.addWidget(self.zigzag_threshold_spin, 0, 1)

        # Peak Detection Parameter
        self.peak_distance_label = QLabel('Distance:')
        method_params_layout.addWidget(self.peak_distance_label, 1, 0)
        self.peak_distance_spin = QSpinBox()
        self.peak_distance_spin.setRange(1, 500)
        self.peak_distance_spin.setValue(10)
        method_params_layout.addWidget(self.peak_distance_spin, 1, 1)

        self.prominence_label = QLabel('Prominence %:')
        method_params_layout.addWidget(self.prominence_label, 2, 0)
        self.prominence_spin = QDoubleSpinBox()
        self.prominence_spin.setRange(0.0, 10.0)
        self.prominence_spin.setValue(0.5)
        self.prominence_spin.setSingleStep(0.1)
        method_params_layout.addWidget(self.prominence_spin, 2, 1)

        self.peak_width_label = QLabel('Width:')
        method_params_layout.addWidget(self.peak_width_label, 3, 0)
        self.peak_width_spin = QSpinBox()
        self.peak_width_spin.setRange(0, 100)
        self.peak_width_spin.setValue(0)
        method_params_layout.addWidget(self.peak_width_spin, 3, 1)

        # Fractal Order
        self.fractal_label = QLabel('Fractal Order:')
        method_params_layout.addWidget(self.fractal_label, 4, 0)
        self.fractal_order_spin = QSpinBox()
        self.fractal_order_spin.setRange(1, 5)
        self.fractal_order_spin.setValue(2)
        method_params_layout.addWidget(self.fractal_order_spin, 4, 1)

        # Pivot Lookback
        self.pivot_label = QLabel('Pivot Lookback:')
        method_params_layout.addWidget(self.pivot_label, 5, 0)
        self.pivot_lookback_spin = QSpinBox()
        self.pivot_lookback_spin.setRange(1, 20)
        self.pivot_lookback_spin.setValue(5)
        method_params_layout.addWidget(self.pivot_lookback_spin, 5, 1)

        layout.addWidget(self.method_params_group)

        # Initial: Parameter-Sichtbarkeit
        self._update_method_params_visibility()

        # Peaks finden Button
        self.find_peaks_btn = QPushButton('Peaks finden')
        self.find_peaks_btn.setFont(QFont('Segoe UI', 12, QFont.Weight.Bold))
        self.find_peaks_btn.setFixedHeight(45)
        self.find_peaks_btn.setStyleSheet(self._button_style((0.2, 0.7, 0.3)))
        self.find_peaks_btn.clicked.connect(self._find_peaks)
        layout.addWidget(self.find_peaks_btn)

        # Status
        self.peaks_status = QLabel('Keine Peaks gefunden')
        self.peaks_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.peaks_status.setStyleSheet('color: #aaa; padding: 5px;')
        layout.addWidget(self.peaks_status)

        # Daten-Info
        info_group = self._create_data_info_group()
        layout.addWidget(info_group)

        layout.addStretch()

        scroll.setWidget(scroll_content)

        main_layout = QVBoxLayout(widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

        return widget

    def _on_peak_method_changed(self, index: int):
        """Wird aufgerufen wenn die Peak-Methode geaendert wird."""
        self._update_method_params_visibility()
        self._invalidate_peaks()

    def _update_method_params_visibility(self):
        """Zeigt/versteckt Methoden-spezifische Parameter."""
        method_idx = self.peak_method_combo.currentIndex()

        # Alle verstecken
        for widget in [self.zigzag_label, self.zigzag_threshold_spin,
                       self.peak_distance_label, self.peak_distance_spin,
                       self.prominence_label, self.prominence_spin,
                       self.peak_width_label, self.peak_width_spin,
                       self.fractal_label, self.fractal_order_spin,
                       self.pivot_label, self.pivot_lookback_spin]:
            widget.hide()

        # Methoden-spezifisch anzeigen
        if method_idx == 1:  # ZigZag
            self.zigzag_label.show()
            self.zigzag_threshold_spin.show()
            self.method_params_group.show()
        elif method_idx == 2:  # Peak Detection
            self.peak_distance_label.show()
            self.peak_distance_spin.show()
            self.prominence_label.show()
            self.prominence_spin.show()
            self.peak_width_label.show()
            self.peak_width_spin.show()
            self.method_params_group.show()
        elif method_idx == 3:  # Fractals
            self.fractal_label.show()
            self.fractal_order_spin.show()
            self.method_params_group.show()
        elif method_idx == 4:  # Pivots
            self.pivot_label.show()
            self.pivot_lookback_spin.show()
            self.method_params_group.show()
        else:
            self.method_params_group.hide()

    def _find_peaks(self):
        """Findet Peaks basierend auf der gewaehlten Methode (im Worker-Thread)."""
        from ..training.labeler import LabelingConfig, LabelingMethod

        # Button deaktivieren waehrend Suche
        self.find_peaks_btn.setEnabled(False)
        self.peaks_status.setText('Suche Peaks...')
        self.peaks_status.setStyleSheet('color: #4da8da;')

        # Methoden-Mapping
        method_map = {
            0: LabelingMethod.FUTURE_RETURN,
            1: LabelingMethod.ZIGZAG,
            2: LabelingMethod.PEAKS,
            3: LabelingMethod.FRACTALS,
            4: LabelingMethod.PIVOTS,
            5: LabelingMethod.EXTREMA_DAILY,
            6: LabelingMethod.BINARY,
        }

        # Config erstellen
        config = LabelingConfig(
            method=method_map[self.peak_method_combo.currentIndex()],
            lookforward=self.peak_lookforward_spin.value(),
            threshold_pct=self.peak_threshold_spin.value(),
            num_classes=3,  # Immer 3 fuer Peak-Erkennung
            zigzag_threshold=self.zigzag_threshold_spin.value(),
            peak_distance=self.peak_distance_spin.value(),
            peak_prominence=self.prominence_spin.value(),
            peak_width=self.peak_width_spin.value() if self.peak_width_spin.value() > 0 else None,
            fractal_order=self.fractal_order_spin.value(),
            pivot_lookback=self.pivot_lookback_spin.value(),
        )

        # Worker starten
        self._peak_worker = PeakFinderWorker(self.data, config, self)
        self._peak_worker.progress.connect(self._on_peak_progress)
        self._peak_worker.finished.connect(self._on_peaks_found)
        self._peak_worker.error.connect(self._on_peak_error)
        self._peak_worker.start()

    def _on_peak_progress(self, message: str):
        """Callback fuer Peak-Finding Fortschritt."""
        self.peaks_status.setText(message)

    def _on_peaks_found(self, result: dict):
        """Callback wenn Peaks gefunden wurden."""
        # Worker-Referenz freigeben
        self._peak_worker = None
        self.find_peaks_btn.setEnabled(True)

        # Ergebnisse speichern
        self.labeler = result['labeler']
        self.detected_peaks = {
            'buy_indices': result['buy_indices'],
            'sell_indices': result['sell_indices']
        }

        num_buy = len(self.detected_peaks['buy_indices'])
        num_sell = len(self.detected_peaks['sell_indices'])

        # Status aktualisieren
        self.peaks_valid = True
        self.labels_valid = False
        self.features_valid = False
        self.samples_valid = False

        self.peaks_status.setText(f'Gefunden: {num_buy} BUY-Peaks, {num_sell} SELL-Peaks')
        self.peaks_status.setStyleSheet('color: #33b34d;')

        # Chart aktualisieren
        close_col = 'Close' if 'Close' in self.data.columns else 'close'
        prices = self.data[close_col].values
        self.peaks_chart.update_price_chart(
            prices,
            self.detected_peaks['buy_indices'],
            self.detected_peaks['sell_indices'],
            'Erkannte Peaks'
        )

        # Tabs aktualisieren
        self._update_tab_status()

        self._log(f'Peaks gefunden: {num_buy} BUY, {num_sell} SELL', 'SUCCESS')

    def _on_peak_error(self, error_msg: str):
        """Callback bei Peak-Finding Fehler."""
        self._peak_worker = None
        self.find_peaks_btn.setEnabled(True)
        self.peaks_status.setText(f'Fehler: {error_msg}')
        self.peaks_status.setStyleSheet('color: #cc4d33;')
        self._log(f'Peak-Erkennung fehlgeschlagen: {error_msg}', 'ERROR')

    def _invalidate_peaks(self):
        """Invalidiert Peaks wenn Parameter geaendert werden."""
        if self.peaks_valid:
            self.peaks_valid = False
            self.labels_valid = False
            self.features_valid = False
            self.samples_valid = False
            self.peaks_status.setText('Peaks ungueltig - neu suchen!')
            self.peaks_status.setStyleSheet('color: #cc4d33;')
            self._update_tab_status()

    # =========================================================================
    # Tab 2: Set Labels
    # =========================================================================

    def _create_set_labels_tab(self) -> QWidget:
        """Erstellt Tab 2: Label-Generierung."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Linke Seite: Parameter
        params_widget = self._create_set_labels_params()
        splitter.addWidget(params_widget)

        # Rechte Seite: Chart
        self.labels_chart = ChartWidget('Labels')
        splitter.addWidget(self.labels_chart)

        splitter.setSizes([400, 1000])

        return tab

    def _create_set_labels_params(self) -> QWidget:
        """Erstellt die Parameter-Seite fuer Tab 2."""
        widget = QWidget()
        widget.setMinimumWidth(380)
        widget.setMaximumWidth(450)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setSpacing(10)

        # Klassen-Auswahl
        class_group = QGroupBox('Label-Klassen')
        class_group.setFont(QFont('Segoe UI', 11, QFont.Weight.Bold))
        class_group.setStyleSheet(self._group_style('#e6b333'))
        class_layout = QVBoxLayout(class_group)

        self.num_classes_combo = QComboBox()
        self.num_classes_combo.addItems([
            '3 Klassen (BUY / HOLD / SELL)',
            '2 Klassen (BUY / SELL)'
        ])
        self.num_classes_combo.setToolTip(
            '3 Klassen: Peaks werden BUY/SELL, Rest wird HOLD\n'
            '2 Klassen: Nur BUY/SELL, kein HOLD'
        )
        class_layout.addWidget(self.num_classes_combo)

        # Info-Text
        info_label = QLabel(
            'Die erkannten Peaks aus Tab 1 werden\n'
            'in Labels umgewandelt:\n'
            '• BUY-Peaks → Label BUY\n'
            '• SELL-Peaks → Label SELL\n'
            '• Rest → Label HOLD (bei 3 Klassen)'
        )
        info_label.setStyleSheet('color: #aaaaaa; padding: 10px;')
        class_layout.addWidget(info_label)

        layout.addWidget(class_group)

        # Labels generieren Button
        self.generate_labels_btn = QPushButton('Labels generieren')
        self.generate_labels_btn.setFont(QFont('Segoe UI', 12, QFont.Weight.Bold))
        self.generate_labels_btn.setFixedHeight(45)
        self.generate_labels_btn.setStyleSheet(self._button_style((0.2, 0.7, 0.3)))
        self.generate_labels_btn.clicked.connect(self._generate_labels)
        layout.addWidget(self.generate_labels_btn)

        # Status
        self.labels_status = QLabel('Keine Labels generiert')
        self.labels_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.labels_status.setStyleSheet('color: #aaa; padding: 5px;')
        layout.addWidget(self.labels_status)

        # Statistik
        stats_group = QGroupBox('Label-Statistik')
        stats_group.setFont(QFont('Segoe UI', 11, QFont.Weight.Bold))
        stats_group.setStyleSheet(self._group_style('#808080'))
        stats_layout = QGridLayout(stats_group)

        stats_layout.addWidget(QLabel('BUY Labels:'), 0, 0)
        self.buy_count_label = QLabel('-')
        self.buy_count_label.setStyleSheet('color: #33cc33; font-weight: bold;')
        stats_layout.addWidget(self.buy_count_label, 0, 1)

        stats_layout.addWidget(QLabel('SELL Labels:'), 1, 0)
        self.sell_count_label = QLabel('-')
        self.sell_count_label.setStyleSheet('color: #cc3333; font-weight: bold;')
        stats_layout.addWidget(self.sell_count_label, 1, 1)

        stats_layout.addWidget(QLabel('HOLD Labels:'), 2, 0)
        self.hold_count_label = QLabel('-')
        self.hold_count_label.setStyleSheet('color: #808080; font-weight: bold;')
        stats_layout.addWidget(self.hold_count_label, 2, 1)

        stats_layout.addWidget(QLabel('Gesamt:'), 3, 0)
        self.total_count_label = QLabel('-')
        self.total_count_label.setStyleSheet('color: white; font-weight: bold;')
        stats_layout.addWidget(self.total_count_label, 3, 1)

        layout.addWidget(stats_group)

        layout.addStretch()

        scroll.setWidget(scroll_content)

        main_layout = QVBoxLayout(widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

        return widget

    def _generate_labels(self):
        """Generiert Labels aus den erkannten Peaks."""
        if not self.peaks_valid:
            self.labels_status.setText('Erst Peaks finden!')
            self.labels_status.setStyleSheet('color: #cc4d33;')
            return

        self.labels_status.setText('Generiere Labels...')
        self.labels_status.setStyleSheet('color: #4da8da;')

        try:
            # Klassenanzahl
            num_classes = 3 if self.num_classes_combo.currentIndex() == 0 else 2
            self.current_num_classes = num_classes

            # Labels-Array erstellen
            n = len(self.data)
            if num_classes == 3:
                # HOLD=0, BUY=1, SELL=2
                labels = np.zeros(n, dtype=np.int32)
                for idx in self.detected_peaks['buy_indices']:
                    if 0 <= idx < n:
                        labels[idx] = 1  # BUY
                for idx in self.detected_peaks['sell_indices']:
                    if 0 <= idx < n:
                        labels[idx] = 2  # SELL
            else:
                # BUY=0, SELL=1 (nur an Peak-Positionen)
                labels = np.full(n, -1, dtype=np.int32)  # -1 = ignorieren
                for idx in self.detected_peaks['buy_indices']:
                    if 0 <= idx < n:
                        labels[idx] = 0  # BUY
                for idx in self.detected_peaks['sell_indices']:
                    if 0 <= idx < n:
                        labels[idx] = 1  # SELL

            self.generated_labels = labels

            # Statistik berechnen
            if num_classes == 3:
                num_buy = np.sum(labels == 1)
                num_sell = np.sum(labels == 2)
                num_hold = np.sum(labels == 0)
            else:
                num_buy = np.sum(labels == 0)
                num_sell = np.sum(labels == 1)
                num_hold = 0

            # UI aktualisieren
            self.buy_count_label.setText(str(num_buy))
            self.sell_count_label.setText(str(num_sell))
            self.hold_count_label.setText(str(num_hold) if num_classes == 3 else '-')
            self.total_count_label.setText(str(num_buy + num_sell + num_hold))

            # Status
            self.labels_valid = True
            self.features_valid = False
            self.samples_valid = False

            self.labels_status.setText(f'Labels generiert: {num_buy} BUY, {num_sell} SELL')
            self.labels_status.setStyleSheet('color: #33b34d;')

            # Chart aktualisieren
            close_col = 'Close' if 'Close' in self.data.columns else 'close'
            prices = self.data[close_col].values
            self.labels_chart.update_labels_chart(prices, labels, num_classes, 'Labels')

            # Tabs aktualisieren
            self._update_tab_status()

            self._log(f'Labels generiert: {num_buy} BUY, {num_sell} SELL', 'SUCCESS')

        except Exception as e:
            self.labels_status.setText(f'Fehler: {e}')
            self.labels_status.setStyleSheet('color: #cc4d33;')
            self._log(f'Label-Generierung fehlgeschlagen: {e}', 'ERROR')

    # =========================================================================
    # Tab 3: Features
    # =========================================================================

    def _create_features_tab(self) -> QWidget:
        """Erstellt Tab 3: Feature-Auswahl mit Uebersicht und Detail-Chart."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Charts ZUERST erstellen (werden von _create_features_params referenziert)
        self.features_chart = ChartWidget('Features (Uebersicht)')
        self.features_detail_chart = ChartWidget('Features (Detail)')

        # Linke Seite: Parameter
        params_widget = self._create_features_params()
        splitter.addWidget(params_widget)

        # Rechte Seite: Zwei Charts uebereinander
        charts_widget = QWidget()
        charts_layout = QVBoxLayout(charts_widget)
        charts_layout.setContentsMargins(0, 0, 0, 0)
        charts_layout.setSpacing(5)

        # Uebersichts-Chart (oben)
        charts_layout.addWidget(self.features_chart, 1)

        # Detail-Bereich (unten)
        detail_widget = QWidget()
        detail_layout = QVBoxLayout(detail_widget)
        detail_layout.setContentsMargins(5, 5, 5, 5)
        detail_layout.setSpacing(5)

        # Detail-Kontrollen
        detail_controls = QHBoxLayout()

        detail_controls.addWidget(QLabel('Bereich:'))

        self.detail_start_spin = QSpinBox()
        self.detail_start_spin.setRange(0, max(0, self.total_points - 100))
        self.detail_start_spin.setValue(0)
        self.detail_start_spin.setFixedWidth(80)
        self.detail_start_spin.setStyleSheet('''
            QSpinBox {
                background-color: #3a3a3a;
                border: 1px solid #555;
                border-radius: 3px;
                color: white;
                padding: 3px;
            }
        ''')
        self.detail_start_spin.valueChanged.connect(self._update_detail_chart)
        detail_controls.addWidget(self.detail_start_spin)

        detail_controls.addWidget(QLabel('Fenster:'))

        self.detail_window_spin = QSpinBox()
        self.detail_window_spin.setRange(50, 500)
        self.detail_window_spin.setValue(100)
        self.detail_window_spin.setSingleStep(10)
        self.detail_window_spin.setFixedWidth(70)
        self.detail_window_spin.setStyleSheet('''
            QSpinBox {
                background-color: #3a3a3a;
                border: 1px solid #555;
                border-radius: 3px;
                color: white;
                padding: 3px;
            }
        ''')
        self.detail_window_spin.valueChanged.connect(self._on_detail_window_changed)
        detail_controls.addWidget(self.detail_window_spin)

        # Slider fuer schnelle Navigation
        self.detail_slider = QSlider(Qt.Orientation.Horizontal)
        self.detail_slider.setRange(0, max(0, self.total_points - 100))
        self.detail_slider.setValue(0)
        self.detail_slider.setStyleSheet('''
            QSlider::groove:horizontal {
                background: #333;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4da8da;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: #4da8da;
                border-radius: 4px;
            }
        ''')
        self.detail_slider.valueChanged.connect(self._on_detail_slider_changed)
        detail_controls.addWidget(self.detail_slider, 1)

        detail_layout.addLayout(detail_controls)
        detail_layout.addWidget(self.features_detail_chart, 1)

        charts_layout.addWidget(detail_widget, 1)

        # Datentabelle (unten)
        table_widget = self._create_features_table()
        charts_layout.addWidget(table_widget)

        splitter.addWidget(charts_widget)
        splitter.setSizes([420, 1000])

        # Initiale Chart-Aktualisierung mit Standard-Features
        self._update_features_preview()

        return tab

    def _create_features_params(self) -> QWidget:
        """Erstellt die Parameter-Seite fuer Tab 3 mit allen Feature-Kategorien."""
        widget = QWidget()
        widget.setMinimumWidth(420)
        widget.setMaximumWidth(500)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setSpacing(8)

        # Feature-Kategorie Widgets erstellen
        self.category_widgets: Dict[str, FeatureCategoryWidget] = {}

        for cat_key in ['price', 'technical', 'volatility', 'volume', 'time']:
            features = FEATURE_REGISTRY.get(cat_key, [])
            if features:
                color = CATEGORY_CONFIG.get(cat_key, {}).get('color', '#808080')
                cat_widget = FeatureCategoryWidget(cat_key, features, color, self)
                cat_widget.feature_changed.connect(self._on_feature_changed)
                self.category_widgets[cat_key] = cat_widget
                layout.addWidget(cat_widget)

        # Normalisierung
        norm_group = QGroupBox('Normalisierung')
        norm_group.setFont(QFont('Segoe UI', 10, QFont.Weight.Bold))
        norm_group.setStyleSheet(self._group_style('#808080'))
        norm_layout = QHBoxLayout(norm_group)

        norm_layout.addWidget(QLabel('Methode:'))
        self.norm_combo = QComboBox()
        self.norm_combo.addItems(['zscore', 'minmax', 'none'])
        norm_layout.addWidget(self.norm_combo)

        layout.addWidget(norm_group)

        # Feature-Info
        self.feature_info = QLabel('Aktive Features: 0')
        self.feature_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feature_info.setStyleSheet('color: #4da8da; font-weight: bold; padding: 10px;')
        layout.addWidget(self.feature_info)

        # Feature-Details
        self.features_status = QLabel('')
        self.features_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.features_status.setStyleSheet('color: #aaa; font-size: 10px; padding: 5px;')
        self.features_status.setWordWrap(True)
        layout.addWidget(self.features_status)

        # Features bestaetigen Button
        self.confirm_features_btn = QPushButton('Features bestaetigen')
        self.confirm_features_btn.setFont(QFont('Segoe UI', 12, QFont.Weight.Bold))
        self.confirm_features_btn.setFixedHeight(45)
        self.confirm_features_btn.setStyleSheet(self._button_style((0.2, 0.7, 0.3)))
        self.confirm_features_btn.clicked.connect(self._confirm_features)
        layout.addWidget(self.confirm_features_btn)

        layout.addStretch()

        scroll.setWidget(scroll_content)

        main_layout = QVBoxLayout(widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

        # Initial Feature-Count aktualisieren
        self._update_feature_count()

        return widget

    def _create_features_table(self) -> QWidget:
        """Erstellt die Datentabelle fuer Feature-Werte mit Navigation."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Navigationsleiste
        nav_layout = QHBoxLayout()

        # Titel
        table_title = QLabel('Feature-Daten')
        table_title.setStyleSheet('color: #4da8da; font-weight: bold;')
        nav_layout.addWidget(table_title)

        nav_layout.addStretch()

        # Navigation Buttons
        self.table_first_btn = QPushButton('<<')
        self.table_first_btn.setFixedWidth(40)
        self.table_first_btn.setStyleSheet(self._nav_button_style())
        self.table_first_btn.clicked.connect(lambda: self._navigate_table('first'))
        nav_layout.addWidget(self.table_first_btn)

        self.table_prev_btn = QPushButton('<')
        self.table_prev_btn.setFixedWidth(40)
        self.table_prev_btn.setStyleSheet(self._nav_button_style())
        self.table_prev_btn.clicked.connect(lambda: self._navigate_table('prev'))
        nav_layout.addWidget(self.table_prev_btn)

        # Index-Anzeige
        self.table_index_label = QLabel('0 - 9')
        self.table_index_label.setStyleSheet('color: white; padding: 0 10px;')
        self.table_index_label.setMinimumWidth(80)
        self.table_index_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_layout.addWidget(self.table_index_label)

        self.table_next_btn = QPushButton('>')
        self.table_next_btn.setFixedWidth(40)
        self.table_next_btn.setStyleSheet(self._nav_button_style())
        self.table_next_btn.clicked.connect(lambda: self._navigate_table('next'))
        nav_layout.addWidget(self.table_next_btn)

        self.table_last_btn = QPushButton('>>')
        self.table_last_btn.setFixedWidth(40)
        self.table_last_btn.setStyleSheet(self._nav_button_style())
        self.table_last_btn.clicked.connect(lambda: self._navigate_table('last'))
        nav_layout.addWidget(self.table_last_btn)

        # Direkte Index-Eingabe
        nav_layout.addWidget(QLabel('Idx:'))
        self.table_index_spin = QSpinBox()
        self.table_index_spin.setRange(0, max(0, self.total_points - 10))
        self.table_index_spin.setValue(0)
        self.table_index_spin.setFixedWidth(80)
        self.table_index_spin.setStyleSheet('''
            QSpinBox {
                background-color: #3a3a3a;
                border: 1px solid #555;
                border-radius: 3px;
                color: white;
                padding: 3px;
            }
        ''')
        self.table_index_spin.valueChanged.connect(self._on_table_index_changed)
        nav_layout.addWidget(self.table_index_spin)

        layout.addLayout(nav_layout)

        # Tabelle
        self.features_table = QTableWidget()
        self.features_table.setStyleSheet('''
            QTableWidget {
                background-color: #1a1a1a;
                border: 1px solid #333;
                gridline-color: #333;
                color: white;
            }
            QTableWidget::item {
                padding: 3px;
            }
            QTableWidget::item:selected {
                background-color: #4da8da;
            }
            QHeaderView::section {
                background-color: #333;
                color: white;
                padding: 5px;
                border: 1px solid #444;
                font-weight: bold;
            }
        ''')
        self.features_table.setAlternatingRowColors(True)
        self.features_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.features_table.verticalHeader().setVisible(False)
        self.features_table.setFixedHeight(280)

        layout.addWidget(self.features_table)

        # Aktueller Tabellenindex
        self._table_start_index = 0

        return widget

    def _nav_button_style(self) -> str:
        """Gibt das Stylesheet fuer Navigations-Buttons zurueck."""
        return '''
            QPushButton {
                background-color: #444;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #555;
            }
            QPushButton:pressed {
                background-color: #333;
            }
        '''

    def _navigate_table(self, direction: str):
        """Navigiert in der Datentabelle."""
        rows_per_page = 10
        max_start = max(0, self.total_points - rows_per_page)

        if direction == 'first':
            self._table_start_index = 0
        elif direction == 'prev':
            self._table_start_index = max(0, self._table_start_index - rows_per_page)
        elif direction == 'next':
            self._table_start_index = min(max_start, self._table_start_index + rows_per_page)
        elif direction == 'last':
            self._table_start_index = max_start

        # SpinBox aktualisieren ohne Signal
        self.table_index_spin.blockSignals(True)
        self.table_index_spin.setValue(self._table_start_index)
        self.table_index_spin.blockSignals(False)

        self._update_features_table()

    def _on_table_index_changed(self, value: int):
        """Wird aufgerufen wenn der Tabellenindex geaendert wird."""
        self._table_start_index = value
        self._update_features_table()

    def _update_features_table(self):
        """Aktualisiert die Datentabelle mit den aktuellen Feature-Werten."""
        if not hasattr(self, 'features_table') or self.features_table is None:
            return
        if not hasattr(self, '_cached_features_dict') or not self._cached_features_dict:
            self.features_table.clear()
            self.features_table.setRowCount(0)
            self.features_table.setColumnCount(0)
            return

        rows_per_page = 10
        start = self._table_start_index
        end = min(start + rows_per_page, self.total_points)

        # Index-Label aktualisieren
        self.table_index_label.setText(f'{start} - {end - 1}')

        # Spalten: Index + alle Features
        feature_names = list(self._cached_features_dict.keys())
        columns = ['Index'] + feature_names

        self.features_table.setColumnCount(len(columns))
        self.features_table.setHorizontalHeaderLabels(columns)
        self.features_table.setRowCount(end - start)

        # Daten einfuegen
        for row, idx in enumerate(range(start, end)):
            # Index-Spalte
            idx_item = QTableWidgetItem(str(idx))
            idx_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.features_table.setItem(row, 0, idx_item)

            # Feature-Werte
            for col, feat_name in enumerate(feature_names, start=1):
                values = self._cached_features_dict[feat_name]
                if idx < len(values):
                    val = values[idx]
                    if np.isnan(val):
                        text = 'NaN'
                    else:
                        # Formatierung je nach Groesse
                        if abs(val) >= 1000:
                            text = f'{val:,.0f}'
                        elif abs(val) >= 1:
                            text = f'{val:.2f}'
                        else:
                            text = f'{val:.4f}'
                else:
                    text = '-'

                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.features_table.setItem(row, col, item)

        # Spaltenbreiten anpassen
        self.features_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Fixed)
        self.features_table.setColumnWidth(0, 60)
        for col in range(1, len(columns)):
            self.features_table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeMode.Stretch)

    def _on_feature_changed(self):
        """Wird aufgerufen wenn sich die Feature-Auswahl aendert."""
        self._update_feature_count()

        if self.features_valid:
            self.features_valid = False
            self.samples_valid = False
            self._update_tab_status()

        # Chart-Vorschau aktualisieren
        self._update_features_preview()

    def _update_feature_count(self):
        """Aktualisiert die Anzeige der aktiven Feature-Anzahl."""
        total = 0
        details = []

        for cat_key, cat_widget in self.category_widgets.items():
            count = cat_widget.get_selected_count()
            total += count
            if count > 0:
                cat_name = CATEGORY_CONFIG.get(cat_key, {}).get('name', cat_key)
                details.append(f"{cat_name[:8]}: {count}")

        self.feature_info.setText(f'Aktive Features: {total}')
        if details:
            self.features_status.setText(' | '.join(details))
        else:
            self.features_status.setText('Keine Features ausgewaehlt')

    def _get_all_selected_features(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Sammelt alle aktiven Features aus allen Kategorien."""
        all_features = []
        for cat_widget in self.category_widgets.values():
            all_features.extend(cat_widget.get_selected_features())
        return all_features

    def _update_features_preview(self):
        """Aktualisiert den Feature-Chart mit allen aktiven Features (normalisiert)."""
        # Sicherheitspruefung: Chart und Daten muessen existieren
        if not hasattr(self, 'features_chart') or self.features_chart is None:
            return
        if not hasattr(self, 'data') or self.data is None or self.data.empty:
            return

        selected_features = self._get_all_selected_features()

        if not selected_features:
            self.features_chart.clear()
            return

        try:
            # Feature-Dict fuer Chart erstellen (mit Caching)
            features_dict: Dict[str, np.ndarray] = {}

            for feat_name, feat_params in selected_features:
                # Feature ueber Cache berechnen (oder gecachtes holen)
                values = self._feature_cache.compute_feature(
                    self.data, feat_name, feat_params
                )

                if values is not None:
                    # Display-Name mit Parametern
                    if feat_params:
                        param_str = ','.join(f'{v}' for v in feat_params.values())
                        display_name = f'{feat_name}({param_str})'
                    else:
                        display_name = feat_name
                    features_dict[display_name] = values

            # Charts und Tabelle aktualisieren
            if features_dict:
                self.features_chart.update_features_chart(features_dict)
                # Feature-Dict speichern fuer Detail-Chart und Tabelle
                self._cached_features_dict = features_dict
                self._update_detail_chart()
                self._update_features_table()
            else:
                self.features_chart.clear()
                if hasattr(self, 'features_detail_chart'):
                    self.features_detail_chart.clear()
                self._cached_features_dict = {}
                self._update_features_table()

        except Exception as e:
            self._log(f'Feature-Vorschau Fehler: {e}', 'WARNING')
            self.features_chart.clear()

    def _on_detail_slider_changed(self, value: int):
        """Wird aufgerufen wenn der Detail-Slider bewegt wird."""
        self.detail_start_spin.blockSignals(True)
        self.detail_start_spin.setValue(value)
        self.detail_start_spin.blockSignals(False)
        self._update_detail_chart()

    def _on_detail_window_changed(self, value: int):
        """Wird aufgerufen wenn die Fenstergroesse geaendert wird."""
        # Slider-Maximum anpassen
        max_start = max(0, self.total_points - value)
        self.detail_slider.setMaximum(max_start)
        self.detail_start_spin.setMaximum(max_start)
        self._update_detail_chart()

    def _update_detail_chart(self):
        """Aktualisiert den Detail-Chart mit dem ausgewaehlten Bereich."""
        if not hasattr(self, 'features_detail_chart') or self.features_detail_chart is None:
            return

        if not hasattr(self, '_cached_features_dict') or not self._cached_features_dict:
            self.features_detail_chart.clear()
            return

        try:
            start = self.detail_start_spin.value()
            window = self.detail_window_spin.value()
            end = min(start + window, self.total_points)

            # Ausschnitt der Features erstellen
            detail_features = {}
            for name, values in self._cached_features_dict.items():
                detail_features[name] = values[start:end]

            if detail_features:
                self.features_detail_chart.update_features_chart(
                    detail_features,
                    title=f'Detail: Index {start} - {end} ({end - start} Punkte)'
                )
            else:
                self.features_detail_chart.clear()

        except Exception as e:
            self._log(f'Detail-Chart Fehler: {e}', 'WARNING')

    def _confirm_features(self):
        """Bestaetigt die Feature-Auswahl."""
        selected = self._get_all_selected_features()
        count = len(selected)

        if count == 0:
            self.feature_info.setText('Mindestens ein Feature auswaehlen!')
            self.feature_info.setStyleSheet('color: #cc4d33; font-weight: bold;')
            return

        # Features mit Parametern speichern
        self.selected_features_with_params = selected  # List[Tuple[name, params_dict]]

        # Fuer Rueckwaertskompatibilitaet: Nur Feature-Namen als Liste
        self.selected_features = [f[0] for f in selected]

        # Feature-Parameter speichern (fuer FeatureProcessor)
        self.feature_params = {f[0]: f[1] for f in selected if f[1]}

        self.features_valid = True
        self.samples_valid = False
        self._update_tab_status()

        self.feature_info.setText(f'Features bestaetigt: {count}')
        self.feature_info.setStyleSheet('color: #33b34d; font-weight: bold;')

        # Zu Tab 4 wechseln
        self.tab_widget.setCurrentIndex(3)

        self._log(f'Features bestaetigt: {count}', 'SUCCESS')

    # =========================================================================
    # Tab 4: Samples
    # =========================================================================

    def _create_samples_tab(self) -> QWidget:
        """Erstellt Tab 4: Sample-Generierung."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Linke Seite: Parameter
        params_widget = self._create_samples_params()
        splitter.addWidget(params_widget)

        # Rechte Seite: Chart
        self.samples_chart = ChartWidget('Samples')
        splitter.addWidget(self.samples_chart)

        splitter.setSizes([400, 1000])

        return tab

    def _create_samples_params(self) -> QWidget:
        """Erstellt die Parameter-Seite fuer Tab 4."""
        widget = QWidget()
        widget.setMinimumWidth(380)
        widget.setMaximumWidth(450)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setSpacing(10)

        # Sequenz-Parameter
        seq_group = QGroupBox('Sequenz-Parameter')
        seq_group.setFont(QFont('Segoe UI', 11, QFont.Weight.Bold))
        seq_group.setStyleSheet(self._group_style('#4da8da'))
        seq_layout = QGridLayout(seq_group)

        # Lookback
        seq_layout.addWidget(QLabel('Lookback:'), 0, 0)
        self.lookback_spin = QSpinBox()
        self.lookback_spin.setRange(10, 500)
        self.lookback_spin.setValue(self.params['lookback'])
        seq_layout.addWidget(self.lookback_spin, 0, 1)

        # Lookforward
        seq_layout.addWidget(QLabel('Lookforward:'), 1, 0)
        self.lookforward_spin = QSpinBox()
        self.lookforward_spin.setRange(1, 200)
        self.lookforward_spin.setValue(self.params['lookforward'])
        seq_layout.addWidget(self.lookforward_spin, 1, 1)

        layout.addWidget(seq_group)

        # HOLD-Samples (nur bei 3 Klassen)
        self.hold_group = QGroupBox('HOLD-Samples')
        self.hold_group.setFont(QFont('Segoe UI', 11, QFont.Weight.Bold))
        self.hold_group.setStyleSheet(self._group_style('#b19cd9'))
        hold_layout = QGridLayout(self.hold_group)

        self.include_hold_cb = QCheckBox('HOLD-Samples erstellen')
        self.include_hold_cb.setChecked(True)
        self.include_hold_cb.setStyleSheet('color: white;')
        hold_layout.addWidget(self.include_hold_cb, 0, 0, 1, 2)

        hold_layout.addWidget(QLabel('Verhaeltnis:'), 1, 0)
        self.hold_ratio_spin = QDoubleSpinBox()
        self.hold_ratio_spin.setRange(0.1, 5.0)
        self.hold_ratio_spin.setValue(1.0)
        self.hold_ratio_spin.setSingleStep(0.1)
        hold_layout.addWidget(self.hold_ratio_spin, 1, 1)

        layout.addWidget(self.hold_group)

        # Vorschau Button
        self.preview_btn = QPushButton('Vorschau berechnen')
        self.preview_btn.setFont(QFont('Segoe UI', 11, QFont.Weight.Bold))
        self.preview_btn.setFixedHeight(40)
        self.preview_btn.setStyleSheet(self._button_style((0.3, 0.6, 0.9)))
        self.preview_btn.clicked.connect(self._calculate_preview)
        layout.addWidget(self.preview_btn)

        # Generieren Button
        self.generate_btn = QPushButton('Daten generieren && Schliessen')
        self.generate_btn.setFont(QFont('Segoe UI', 12, QFont.Weight.Bold))
        self.generate_btn.setFixedHeight(45)
        self.generate_btn.setStyleSheet(self._button_style((0.2, 0.7, 0.3)))
        self.generate_btn.setEnabled(False)
        self.generate_btn.clicked.connect(self._generate_and_close)
        layout.addWidget(self.generate_btn)

        # Status
        self.samples_status = QLabel('Vorschau berechnen um fortzufahren')
        self.samples_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.samples_status.setStyleSheet('color: #aaa; padding: 5px;')
        layout.addWidget(self.samples_status)

        layout.addStretch()

        scroll.setWidget(scroll_content)

        main_layout = QVBoxLayout(widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

        return widget

    def _calculate_preview(self):
        """Berechnet die Vorschau."""
        if not self.features_valid:
            self.samples_status.setText('Erst Features bestaetigen!')
            self.samples_status.setStyleSheet('color: #cc4d33;')
            return

        self.samples_status.setText('Berechne Vorschau...')
        self.samples_status.setStyleSheet('color: #4da8da;')

        try:
            lookback = self.lookback_spin.value()
            lookforward = self.lookforward_spin.value()

            # Pruefe Datenlaenge
            if lookback + lookforward >= len(self.data):
                self.samples_status.setText('Lookback + Lookforward zu gross!')
                self.samples_status.setStyleSheet('color: #cc4d33;')
                return

            # Sample-Statistik berechnen
            num_classes = self.current_num_classes

            if num_classes == 3:
                num_buy = len(self.detected_peaks['buy_indices'])
                num_sell = len(self.detected_peaks['sell_indices'])
                if self.include_hold_cb.isChecked():
                    num_hold = int((num_buy + num_sell) * self.hold_ratio_spin.value())
                else:
                    num_hold = 0
            else:
                num_buy = len(self.detected_peaks['buy_indices'])
                num_sell = len(self.detected_peaks['sell_indices'])
                num_hold = 0

            total = num_buy + num_sell + num_hold

            # Info speichern
            self.result_info = {
                'num_buy': num_buy,
                'num_sell': num_sell,
                'num_hold': num_hold,
                'total': total,
                'lookback': lookback,
                'lookforward': lookforward,
                'num_features': len(self.selected_features),
                'num_classes': num_classes,
            }

            # Chart aktualisieren
            self.samples_chart.update_samples_chart(num_buy, num_sell, num_hold)

            self.samples_valid = True
            self.generate_btn.setEnabled(True)

            self.samples_status.setText(f'Vorschau: {total} Samples')
            self.samples_status.setStyleSheet('color: #33b34d;')

        except Exception as e:
            self.samples_status.setText(f'Fehler: {e}')
            self.samples_status.setStyleSheet('color: #cc4d33;')

    def _generate_and_close(self):
        """Generiert die Trainingsdaten und schliesst das Fenster."""
        if not self.samples_valid:
            return

        self.samples_status.setText('Generiere Trainingsdaten...')

        try:
            # Training-Daten erstellen (Platzhalter - echte Implementierung spaeter)
            num_classes = self.result_info.get('num_classes', 3)

            training_data = {
                'X': np.random.randn(
                    self.result_info['total'],
                    self.result_info['lookback'],
                    self.result_info['num_features']
                ),
                'Y': np.random.randint(0, num_classes, self.result_info['total']),
                'params': self.params.copy(),
            }

            training_info = self.result_info.copy()
            training_info['params'] = self.params.copy()
            training_info['features'] = self.selected_features.copy()

            # Signal senden
            self.data_prepared.emit(training_data, training_info)

            self._log(f"Trainingsdaten generiert: {self.result_info['total']} Samples", 'SUCCESS')

            self.close()

        except Exception as e:
            self.samples_status.setText(f'Fehler: {e}')
            self.samples_status.setStyleSheet('color: #cc4d33;')
            self._log(f'Fehler bei Datengenerierung: {e}', 'ERROR')

    # =========================================================================
    # Hilfsmethoden
    # =========================================================================

    def _update_tab_status(self):
        """Aktualisiert Tab-Status basierend auf Validitaet."""
        self.tab_widget.setTabEnabled(1, self.peaks_valid)
        self.tab_widget.setTabEnabled(2, self.labels_valid)
        self.tab_widget.setTabEnabled(3, self.features_valid)

        # Status-Text
        if not self.peaks_valid:
            self.status_label.setText('1. Peak-Methode waehlen und Peaks finden')
            self.status_label.setStyleSheet('color: #aaa;')
        elif not self.labels_valid:
            self.status_label.setText('2. Labels generieren')
            self.status_label.setStyleSheet('color: #e6b333;')
        elif not self.features_valid:
            self.status_label.setText('3. Features auswaehlen und bestaetigen')
            self.status_label.setStyleSheet('color: #e6b333;')
        elif not self.samples_valid:
            self.status_label.setText('4. Samples konfigurieren und generieren')
            self.status_label.setStyleSheet('color: #e6b333;')
        else:
            self.status_label.setText('Bereit zum Generieren!')
            self.status_label.setStyleSheet('color: #33b34d;')

    def _update_dynamic_limits(self):
        """Aktualisiert dynamische Limits basierend auf Datengroesse."""
        n = len(self.data)
        max_lookback = min(500, n // 4)
        max_lookforward = min(200, n // 8)

        if hasattr(self, 'lookback_spin'):
            self.lookback_spin.setMaximum(max_lookback)
        if hasattr(self, 'lookforward_spin'):
            self.lookforward_spin.setMaximum(max_lookforward)
        if hasattr(self, 'peak_lookforward_spin'):
            self.peak_lookforward_spin.setMaximum(max_lookforward)

    def _create_data_info_group(self) -> QGroupBox:
        """Erstellt die Daten-Info Gruppe."""
        group = QGroupBox('Geladene Daten')
        group.setFont(QFont('Segoe UI', 11, QFont.Weight.Bold))
        group.setStyleSheet(self._group_style('#808080'))
        layout = QGridLayout(group)

        layout.addWidget(QLabel('Datenpunkte:'), 0, 0)
        layout.addWidget(QLabel(f'{self.total_points:,}'), 0, 1)

        # Zeitraum
        if hasattr(self.data.index, 'strftime'):
            start = self.data.index[0].strftime('%d.%m.%y')
            end = self.data.index[-1].strftime('%d.%m.%y')
        else:
            start = str(self.data.index[0])[:10]
            end = str(self.data.index[-1])[:10]

        layout.addWidget(QLabel('Zeitraum:'), 1, 0)
        layout.addWidget(QLabel(f'{start} - {end}'), 1, 1)

        # Preisbereich
        close_col = 'Close' if 'Close' in self.data.columns else 'close'
        if close_col in self.data.columns:
            min_price = self.data[close_col].min()
            max_price = self.data[close_col].max()
            layout.addWidget(QLabel('Preis:'), 2, 0)
            layout.addWidget(QLabel(f'{min_price:.0f} - {max_price:.0f}'), 2, 1)

        return group

    def _group_style(self, color: str) -> str:
        """Generiert GroupBox-Stylesheet."""
        return StyleFactory.group_style(hex_color=color)

    def _button_style(self, color: tuple) -> str:
        """Generiert Button-Stylesheet."""
        return StyleFactory.button_style(color)

    def _get_tab_stylesheet(self) -> str:
        """Gibt das Tab-Widget Stylesheet zurueck."""
        return StyleFactory.tab_style()

    def _get_stylesheet(self) -> str:
        """Gibt das Fenster-Stylesheet zurueck."""
        return (StyleFactory.window_style() +
                StyleFactory.combobox_style() +
                StyleFactory.spinbox_style())
