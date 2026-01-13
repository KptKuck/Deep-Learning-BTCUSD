"""
Prepare Data Window - Trainingsdaten Vorbereitung
Refactored: 4 Tabs mit eigenem Chart pro Tab
"""

from typing import Optional, Dict, Any, List

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox, QScrollArea, QFrame,
    QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox, QSplitter,
    QTabWidget
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

import pandas as pd
import numpy as np

from .chart_widget import ChartWidget


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
        """Findet Peaks basierend auf der gewaehlten Methode."""
        from ..training.labeler import DailyExtremaLabeler, LabelingConfig, LabelingMethod

        self.peaks_status.setText('Suche Peaks...')
        self.peaks_status.setStyleSheet('color: #4da8da;')

        try:
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

            # Labeler erstellen und Peaks finden
            self.labeler = DailyExtremaLabeler(
                lookforward=config.lookforward,
                threshold_pct=config.threshold_pct,
                num_classes=3
            )
            _ = self.labeler.generate_labels(self.data, config=config)

            # Peaks speichern
            self.detected_peaks = {
                'buy_indices': self.labeler.buy_signal_indices.copy(),
                'sell_indices': self.labeler.sell_signal_indices.copy()
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

        except Exception as e:
            self.peaks_status.setText(f'Fehler: {e}')
            self.peaks_status.setStyleSheet('color: #cc4d33;')
            self._log(f'Peak-Erkennung fehlgeschlagen: {e}', 'ERROR')

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
        """Erstellt Tab 3: Feature-Auswahl."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Linke Seite: Parameter
        params_widget = self._create_features_params()
        splitter.addWidget(params_widget)

        # Rechte Seite: Chart
        self.features_chart = ChartWidget('Features')
        splitter.addWidget(self.features_chart)

        splitter.setSizes([400, 1000])

        return tab

    def _create_features_params(self) -> QWidget:
        """Erstellt die Parameter-Seite fuer Tab 3."""
        widget = QWidget()
        widget.setMinimumWidth(380)
        widget.setMaximumWidth(450)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setSpacing(10)

        # Preis-Features
        price_group = QGroupBox('Preis-Features')
        price_group.setFont(QFont('Segoe UI', 11, QFont.Weight.Bold))
        price_group.setStyleSheet(self._group_style('#4da8da'))
        price_layout = QVBoxLayout(price_group)

        self.feature_checks = {}
        price_features = [
            ('Close', True), ('High', True), ('Low', True), ('Open', True),
            ('PriceChange', True), ('PriceChangePct', True)
        ]
        for feat, default in price_features:
            cb = QCheckBox(feat)
            cb.setChecked(default)
            cb.setStyleSheet('color: white;')
            cb.stateChanged.connect(self._on_feature_changed)
            self.feature_checks[feat] = cb
            price_layout.addWidget(cb)

        layout.addWidget(price_group)

        # Volumen-Features
        vol_group = QGroupBox('Volumen-Features')
        vol_group.setFont(QFont('Segoe UI', 11, QFont.Weight.Bold))
        vol_group.setStyleSheet(self._group_style('#7fe6b3'))
        vol_layout = QVBoxLayout(vol_group)

        for feat in ['Volume', 'RelativeVolume']:
            cb = QCheckBox(feat)
            cb.setChecked(False)
            cb.setStyleSheet('color: white;')
            cb.stateChanged.connect(self._on_feature_changed)
            self.feature_checks[feat] = cb
            vol_layout.addWidget(cb)

        layout.addWidget(vol_group)

        # Volatilitaets-Features
        volatility_group = QGroupBox('Volatilitaets-Features')
        volatility_group.setFont(QFont('Segoe UI', 11, QFont.Weight.Bold))
        volatility_group.setStyleSheet(self._group_style('#ff9966'))
        volatility_layout = QVBoxLayout(volatility_group)

        volatility_features = ['ATR', 'ATR_Pct', 'RollingStd', 'HighLowRange']
        for feat in volatility_features:
            cb = QCheckBox(feat)
            cb.setChecked(False)
            cb.setStyleSheet('color: white;')
            cb.stateChanged.connect(self._on_feature_changed)
            self.feature_checks[feat] = cb
            volatility_layout.addWidget(cb)

        layout.addWidget(volatility_group)

        # Normalisierung
        norm_group = QGroupBox('Normalisierung')
        norm_group.setFont(QFont('Segoe UI', 11, QFont.Weight.Bold))
        norm_group.setStyleSheet(self._group_style('#b19cd9'))
        norm_layout = QHBoxLayout(norm_group)

        norm_layout.addWidget(QLabel('Methode:'))
        self.norm_combo = QComboBox()
        self.norm_combo.addItems(['zscore', 'minmax', 'none'])
        norm_layout.addWidget(self.norm_combo)

        layout.addWidget(norm_group)

        # Feature-Info
        self.feature_info = QLabel('Aktive Features: 6')
        self.feature_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feature_info.setStyleSheet('color: #4da8da; font-weight: bold; padding: 10px;')
        layout.addWidget(self.feature_info)

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

        return widget

    def _on_feature_changed(self):
        """Wird aufgerufen wenn sich die Feature-Auswahl aendert."""
        count = self._count_selected_features()
        self.feature_info.setText(f'Aktive Features: {count}')

        if self.features_valid:
            self.features_valid = False
            self.samples_valid = False
            self._update_tab_status()

        # Chart-Vorschau aktualisieren
        self._update_features_chart()

    def _count_selected_features(self) -> int:
        """Zaehlt die ausgewaehlten Features."""
        count = 0
        for name, cb in self.feature_checks.items():
            if cb.isChecked():
                count += 1
        return count

    def _update_features_chart(self):
        """Aktualisiert den Feature-Chart."""
        close_col = 'Close' if 'Close' in self.data.columns else 'close'
        high_col = 'High' if 'High' in self.data.columns else 'high'
        low_col = 'Low' if 'Low' in self.data.columns else 'low'

        features = {}

        for name, cb in self.feature_checks.items():
            if cb.isChecked():
                if name == 'Close' and close_col in self.data.columns:
                    features['Close'] = self.data[close_col].values
                elif name == 'High' and high_col in self.data.columns:
                    features['High'] = self.data[high_col].values
                elif name == 'Low' and low_col in self.data.columns:
                    features['Low'] = self.data[low_col].values

        if features:
            self.features_chart.update_features_chart(features)
        else:
            self.features_chart.clear()

    def _confirm_features(self):
        """Bestaetigt die Feature-Auswahl."""
        count = self._count_selected_features()

        if count == 0:
            self.feature_info.setText('Mindestens ein Feature auswaehlen!')
            self.feature_info.setStyleSheet('color: #cc4d33; font-weight: bold;')
            return

        # Features speichern
        self.selected_features = [name for name, cb in self.feature_checks.items() if cb.isChecked()]

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
        return f'''
            QGroupBox {{
                color: {color};
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

    def _button_style(self, color: tuple) -> str:
        """Generiert Button-Stylesheet."""
        r, g, b = [int(c * 255) for c in color]
        r_h, g_h, b_h = [min(255, int(c * 255 * 1.2)) for c in color]
        r_p, g_p, b_p = [int(c * 255 * 0.8) for c in color]

        return f'''
            QPushButton {{
                background-color: rgb({r}, {g}, {b});
                color: white;
                border: none;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: rgb({r_h}, {g_h}, {b_h});
            }}
            QPushButton:pressed {{
                background-color: rgb({r_p}, {g_p}, {b_p});
            }}
            QPushButton:disabled {{
                background-color: rgb(80, 80, 80);
                color: rgb(120, 120, 120);
            }}
        '''

    def _get_tab_stylesheet(self) -> str:
        """Gibt das Tab-Widget Stylesheet zurueck."""
        return '''
            QTabWidget::pane {
                border: 1px solid #333;
                background-color: #2a2a2a;
            }
            QTabBar::tab {
                background: #333;
                color: #aaa;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: #4da8da;
                color: white;
            }
            QTabBar::tab:hover:!selected {
                background: #444;
            }
            QTabBar::tab:disabled {
                background: #222;
                color: #555;
            }
        '''

    def _get_stylesheet(self) -> str:
        """Gibt das Fenster-Stylesheet zurueck."""
        return '''
            QMainWindow {
                background-color: #262626;
            }
            QWidget {
                color: white;
            }
            QLabel {
                color: white;
            }
            QScrollArea {
                background-color: #2e2e2e;
            }
            QComboBox {
                background-color: #3a3a3a;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px;
                color: white;
            }
            QComboBox:drop-down {
                border: none;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #3a3a3a;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 3px;
                color: white;
            }
            QCheckBox {
                color: white;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
        '''
