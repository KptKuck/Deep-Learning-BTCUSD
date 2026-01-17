"""
Feature-Widgets - UI-Komponenten fuer Feature-Auswahl
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple

from PyQt6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QCheckBox, QSpinBox,
    QLabel, QFrame, QWidget, QPushButton, QSizePolicy, QGridLayout
)
from PyQt6.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve, QSize
from PyQt6.QtGui import QFont


# Kompakte Styles fuer Feature-Widgets
COMPACT_CHECKBOX_STYLE = """
    QCheckBox {
        font-size: 10px;
        spacing: 2px;
        padding: 1px 3px;
    }
    QCheckBox::indicator {
        width: 12px;
        height: 12px;
    }
"""

COMPACT_SPINBOX_STYLE = """
    QSpinBox {
        background-color: #3a3a3a;
        border: 1px solid #555;
        border-radius: 2px;
        color: white;
        padding: 1px;
        font-size: 10px;
        max-width: 45px;
    }
"""


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

# Kategorie-Anzeigenamen und Farben (kompaktere Namen)
CATEGORY_CONFIG = {
    'price': {'name': 'Preis', 'color': '#4da8da'},
    'technical': {'name': 'Technisch', 'color': '#b19cd9'},
    'volatility': {'name': 'Volatilitaet', 'color': '#ff9966'},
    'volume': {'name': 'Volumen', 'color': '#7fe6b3'},
    'time': {'name': 'Zeit', 'color': '#e6b333'},
}


# =============================================================================
# FeatureCategoryWidget (Collapsible)
# =============================================================================

class FeatureCategoryWidget(QWidget):
    """Aufklappbares Widget fuer eine Feature-Kategorie mit Checkboxen und Parametern."""

    feature_changed = pyqtSignal()

    def __init__(self, category_key: str, features: List[FeatureDefinition],
                 color: str, parent=None):
        super().__init__(parent)

        self.category_key = category_key
        self.category_name = CATEGORY_CONFIG.get(category_key, {}).get('name', category_key)
        self.features = features
        self.color = color
        self.feature_widgets: Dict[str, Dict[str, Any]] = {}
        self._is_collapsed = True  # Startet eingeklappt

        self._init_ui()

    def _init_ui(self):
        """Initialisiert die UI-Komponenten (kompaktes Layout)."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header-Button (immer sichtbar) - kompakter
        self.header_btn = QPushButton()
        self.header_btn.setCheckable(True)
        self.header_btn.setChecked(False)
        self.header_btn.clicked.connect(self._toggle_collapse)
        self._update_header_text()
        self.header_btn.setStyleSheet(self._get_header_style())
        self.header_btn.setFixedHeight(28)  # Kompakter: 28 statt 36
        main_layout.addWidget(self.header_btn)

        # Content-Container (aufklappbar)
        self.content_widget = QWidget()
        self.content_widget.setStyleSheet(f'''
            QWidget {{
                background-color: #2a2a2a;
                border: 1px solid #333;
                border-top: none;
                border-bottom-left-radius: 4px;
                border-bottom-right-radius: 4px;
            }}
        ''')

        content_layout = QVBoxLayout(self.content_widget)
        content_layout.setSpacing(2)  # Kompakter: 2 statt 4
        content_layout.setContentsMargins(4, 4, 4, 4)  # Kompakter: 4 statt 8

        # Obere Zeile: "Alle" Checkbox (kompakt, inline)
        top_row = QHBoxLayout()
        top_row.setSpacing(4)

        self.all_checkbox = QCheckBox('Alle')
        self.all_checkbox.setStyleSheet('color: white; font-weight: bold; font-size: 10px;')
        self.all_checkbox.stateChanged.connect(self._on_all_toggled)
        top_row.addWidget(self.all_checkbox)

        # Trennstrich
        separator = QLabel('|')
        separator.setStyleSheet('color: #555; font-size: 10px;')
        top_row.addWidget(separator)

        # Feature-Checkboxen in Grid-Layout (4 pro Zeile)
        self.features_grid = QGridLayout()
        self.features_grid.setSpacing(3)
        self.features_grid.setContentsMargins(0, 0, 0, 0)

        # Features mit Parametern separat
        features_with_params = []
        features_without_params = []

        for feat_def in self.features:
            if feat_def.params:
                features_with_params.append(feat_def)
            else:
                features_without_params.append(feat_def)

        # Einfache Features in Grid (4 pro Zeile)
        col = 0
        row = 0
        max_cols = 4

        for feat_def in features_without_params:
            cb = self._create_compact_checkbox(feat_def)
            self.features_grid.addWidget(cb, row, col)
            col += 1
            if col >= max_cols:
                col = 0
                row += 1

        # Layout zusammenbauen
        top_row.addLayout(self.features_grid)
        top_row.addStretch()
        content_layout.addLayout(top_row)

        # Features mit Parametern in separaten Zeilen (kompakter)
        if features_with_params:
            params_layout = QHBoxLayout()
            params_layout.setSpacing(8)
            params_layout.setContentsMargins(0, 2, 0, 0)

            for feat_def in features_with_params:
                feat_widget = self._create_compact_param_widget(feat_def)
                params_layout.addWidget(feat_widget)

            params_layout.addStretch()
            content_layout.addLayout(params_layout)

        main_layout.addWidget(self.content_widget)

        # Initial: eingeklappt
        self.content_widget.setVisible(False)

        # Initial "Alle" Status und Header aktualisieren
        self._update_all_checkbox_state()
        self._update_header_text()

    def _toggle_collapse(self):
        """Klappt den Content-Bereich ein/aus."""
        self._is_collapsed = not self._is_collapsed
        self.content_widget.setVisible(not self._is_collapsed)
        self.header_btn.setChecked(not self._is_collapsed)
        self._update_header_text()

    def _update_header_text(self):
        """Aktualisiert den Header-Text mit Pfeil und Anzahl."""
        arrow = "v" if not self._is_collapsed else ">"
        selected = self.get_selected_count()
        total = len(self.features)
        self.header_btn.setText(f"  {arrow}  {self.category_name}  [{selected}/{total}]")

    def _get_header_style(self) -> str:
        """Gibt das Header-Button Stylesheet zurueck (kompakter)."""
        return f'''
            QPushButton {{
                background-color: #333;
                color: {self.color};
                border: 1px solid #444;
                border-radius: 4px;
                text-align: left;
                padding-left: 8px;
                font-weight: bold;
                font-size: 10px;
            }}
            QPushButton:hover {{
                background-color: #3a3a3a;
                border-color: {self.color};
            }}
            QPushButton:checked {{
                background-color: #2a2a2a;
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
            }}
        '''

    def _create_compact_checkbox(self, feat_def: FeatureDefinition) -> QCheckBox:
        """Erstellt eine kompakte Checkbox ohne Parameter."""
        cb = QCheckBox(feat_def.name)  # Kurzer Name statt display_name
        cb.setChecked(feat_def.default_enabled)
        cb.setStyleSheet(COMPACT_CHECKBOX_STYLE + ' QCheckBox { color: white; }')
        cb.stateChanged.connect(self._on_feature_toggled)

        # Widget-Daten speichern
        self.feature_widgets[feat_def.name] = {'checkbox': cb, 'params': {}}

        return cb

    def _create_compact_param_widget(self, feat_def: FeatureDefinition) -> QWidget:
        """Erstellt ein kompaktes Widget fuer Feature mit Parametern."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Checkbox
        cb = QCheckBox(feat_def.name)
        cb.setChecked(feat_def.default_enabled)
        cb.setStyleSheet(COMPACT_CHECKBOX_STYLE + ' QCheckBox { color: white; }')
        cb.stateChanged.connect(self._on_feature_toggled)
        layout.addWidget(cb)

        widget_data: Dict[str, Any] = {'checkbox': cb, 'params': {}}

        # Parameter inline: [SMA] [20]
        for param_name, default_val in feat_def.params.items():
            spin = QSpinBox()
            spin.setStyleSheet(COMPACT_SPINBOX_STYLE)
            spin.setFixedWidth(45)

            if param_name in feat_def.param_ranges:
                min_val, max_val, step = feat_def.param_ranges[param_name]
                spin.setRange(int(min_val), int(max_val))
                spin.setSingleStep(int(step))
            else:
                spin.setRange(1, 500)

            spin.setValue(int(default_val))
            spin.valueChanged.connect(self._on_param_changed)
            layout.addWidget(spin)

            widget_data['params'][param_name] = spin

        self.feature_widgets[feat_def.name] = widget_data

        return widget

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
        self._update_header_text()
        self.feature_changed.emit()

    def _on_feature_toggled(self):
        """Wird aufgerufen wenn ein Feature-Checkbox geaendert wird."""
        self._update_all_checkbox_state()
        self._update_header_text()
        self.feature_changed.emit()

    def _on_param_changed(self):
        """Wird aufgerufen wenn ein Parameter geaendert wird."""
        self.feature_changed.emit()

    def _update_all_checkbox_state(self):
        """Aktualisiert den Status der 'Alle' Checkbox."""
        if not self.feature_widgets:
            return

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

    def expand(self):
        """Klappt die Kategorie auf."""
        if self._is_collapsed:
            self._toggle_collapse()

    def collapse(self):
        """Klappt die Kategorie zu."""
        if not self._is_collapsed:
            self._toggle_collapse()

    def is_collapsed(self) -> bool:
        """Gibt zurueck ob die Kategorie eingeklappt ist."""
        return self._is_collapsed
