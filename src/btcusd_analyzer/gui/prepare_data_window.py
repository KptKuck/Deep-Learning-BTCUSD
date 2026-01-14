"""
Prepare Data Window - Trainingsdaten Vorbereitung
Refactored: 4 Tabs mit eigenem Chart pro Tab
"""

from typing import Optional, Dict, Any, List, Tuple

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox, QScrollArea, QFrame,
    QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox, QSplitter,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView, QSlider
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

import pandas as pd
import numpy as np

from .chart_widget import ChartWidget
from .styles import StyleFactory, COLORS
from .widgets import (
    FeatureDefinition,
    FeatureCategoryWidget,
    FEATURE_REGISTRY,
    CATEGORY_CONFIG,
    PipelineState,
    PipelineStage,
    PeakFinderWorker,
    LabelGeneratorWorker,
    FeatureCache,
)


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

        # Pipeline-Status (ersetzt die 4 Boolean-Flags)
        self._pipeline = PipelineState()

        # Feature-Cache initialisieren
        self._feature_cache = FeatureCache()
        self._feature_cache.set_data(data)

        # Worker-Referenzen
        self._peak_worker: Optional[PeakFinderWorker] = None
        self._label_worker: Optional[LabelGeneratorWorker] = None

        # UI initialisieren
        self._init_ui()
        self._update_dynamic_limits()

    # Properties fuer Abwaertskompatibilitaet mit Boolean-Flags
    @property
    def peaks_valid(self) -> bool:
        return self._pipeline.peaks_valid

    @peaks_valid.setter
    def peaks_valid(self, value: bool):
        if value:
            self._pipeline.advance_to(PipelineStage.PEAKS)
        else:
            self._pipeline.invalidate_from(PipelineStage.PEAKS)

    @property
    def labels_valid(self) -> bool:
        return self._pipeline.labels_valid

    @labels_valid.setter
    def labels_valid(self, value: bool):
        if value:
            self._pipeline.advance_to(PipelineStage.LABELS)
        else:
            self._pipeline.invalidate_from(PipelineStage.LABELS)

    @property
    def features_valid(self) -> bool:
        return self._pipeline.features_valid

    @features_valid.setter
    def features_valid(self, value: bool):
        if value:
            self._pipeline.advance_to(PipelineStage.FEATURES)
        else:
            self._pipeline.invalidate_from(PipelineStage.FEATURES)

    @property
    def samples_valid(self) -> bool:
        return self._pipeline.samples_valid

    @samples_valid.setter
    def samples_valid(self, value: bool):
        if value:
            self._pipeline.advance_to(PipelineStage.SAMPLES)
        else:
            self._pipeline.invalidate_from(PipelineStage.SAMPLES)

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
        self.peak_method_combo.setCurrentIndex(2)  # Peak Detection als Standard
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
        """Erstellt Tab 3: Feature-Auswahl mit Sub-Tabs fuer verschiedene Ansichten."""
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

        # Rechte Seite: Sub-Tab Widget fuer verschiedene Ansichten
        self.features_view_tabs = QTabWidget()
        self.features_view_tabs.setStyleSheet(self._get_subtab_stylesheet())

        # Sub-Tab 1: Uebersicht-Chart
        overview_tab = self._create_features_overview_subtab()
        self.features_view_tabs.addTab(overview_tab, 'Uebersicht')

        # Sub-Tab 2: Detail-Chart mit Slider
        detail_tab = self._create_features_detail_subtab()
        self.features_view_tabs.addTab(detail_tab, 'Detail')

        # Sub-Tab 3: Datentabelle
        table_tab = self._create_features_table_subtab()
        self.features_view_tabs.addTab(table_tab, 'Daten')

        splitter.addWidget(self.features_view_tabs)
        splitter.setSizes([420, 1000])

        # Initiale Chart-Aktualisierung mit Standard-Features
        self._update_features_preview()

        return tab

    def _get_subtab_stylesheet(self) -> str:
        """Stylesheet fuer Sub-Tabs (kompakter als Haupt-Tabs)."""
        return '''
            QTabWidget::pane {
                border: 1px solid #444;
                background-color: #2a2a2a;
                border-radius: 4px;
            }
            QTabBar::tab {
                background: #333;
                color: #999;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-size: 11px;
            }
            QTabBar::tab:selected {
                background: #4da8da;
                color: white;
            }
            QTabBar::tab:hover:!selected {
                background: #444;
                color: #ccc;
            }
        '''

    def _create_features_overview_subtab(self) -> QWidget:
        """Erstellt den Uebersicht Sub-Tab mit dem Haupt-Feature-Chart."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.addWidget(self.features_chart)
        return widget

    def _create_features_detail_subtab(self) -> QWidget:
        """Erstellt den Detail Sub-Tab mit Slider-Navigation."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Detail-Kontrollen
        controls = QHBoxLayout()

        controls.addWidget(QLabel('Bereich:'))

        self.detail_start_spin = QSpinBox()
        self.detail_start_spin.setRange(0, max(0, self.total_points - 100))
        self.detail_start_spin.setValue(0)
        self.detail_start_spin.setFixedWidth(80)
        self.detail_start_spin.setStyleSheet(StyleFactory.spinbox_style())
        self.detail_start_spin.valueChanged.connect(self._update_detail_chart)
        controls.addWidget(self.detail_start_spin)

        controls.addWidget(QLabel('Fenster:'))

        self.detail_window_spin = QSpinBox()
        self.detail_window_spin.setRange(50, 500)
        self.detail_window_spin.setValue(100)
        self.detail_window_spin.setSingleStep(10)
        self.detail_window_spin.setFixedWidth(70)
        self.detail_window_spin.setStyleSheet(StyleFactory.spinbox_style())
        self.detail_window_spin.valueChanged.connect(self._on_detail_window_changed)
        controls.addWidget(self.detail_window_spin)

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
        controls.addWidget(self.detail_slider, 1)

        layout.addLayout(controls)
        layout.addWidget(self.features_detail_chart, 1)

        return widget

    def _create_features_table_subtab(self) -> QWidget:
        """Erstellt den Daten Sub-Tab mit der Feature-Tabelle."""
        return self._create_features_table()

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
        # Keine feste Hoehe mehr - fuellt den gesamten Sub-Tab aus

        layout.addWidget(self.features_table, 1)  # stretch=1 fuer volle Hoehe

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
