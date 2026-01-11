"""
Prepare Data Window - Trainingsdaten Vorbereitung
Portiert von MATLAB prepare_training_data_gui.m
"""

from typing import Optional, Dict, Any
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox, QScrollArea, QFrame,
    QCheckBox, QComboBox, QSlider, QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

import pandas as pd
import numpy as np

# Matplotlib fuer Charts
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PrepareDataWindow(QMainWindow):
    """
    Fenster zur Vorbereitung der Trainingsdaten.

    Layout: 3 Spalten (380px | flexible | 320px)
    - Links: Parameter (Lookback/Lookforward, Features, HOLD-Samples, Normalisierung)
    - Mitte: Chart-Preview mit Signal-Markern
    - Rechts: Statistik-Tabelle + Legende
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
            # Features
            'use_close': True,
            'use_high': True,
            'use_low': True,
            'use_open': True,
            'use_price_change': True,
            'use_price_change_pct': True,
        }

        # Ergebnis-Variablen
        self.result_X = None
        self.result_Y = None
        self.result_info = {}
        self.preview_computed = False

        # UI initialisieren
        self._init_ui()
        self._update_seq_info()
        self._calculate_extrema()

    def _log(self, message: str, level: str = 'INFO'):
        """Loggt eine Nachricht an MainWindow."""
        if self._parent and hasattr(self._parent, '_log'):
            self._parent._log(f'[Prepare] {message}', level)
        self.log_message.emit(message, level)

    def _init_ui(self):
        """Initialisiert die UI-Komponenten."""
        self.setWindowTitle('Trainingsdaten Vorbereitung')
        self.setMinimumSize(1500, 950)
        self.setStyleSheet(self._get_stylesheet())

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main Layout: 3 Spalten
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Linke Spalte: Parameter
        left_panel = self._create_param_panel()
        splitter.addWidget(left_panel)

        # Mitte: Chart
        center_panel = self._create_chart_panel()
        splitter.addWidget(center_panel)

        # Rechte Spalte: Statistik
        right_panel = self._create_stats_panel()
        splitter.addWidget(right_panel)

        # Splitter Proportionen
        splitter.setSizes([380, 800, 320])

    def _create_param_panel(self) -> QWidget:
        """Erstellt das linke Parameter-Panel."""
        panel = QWidget()
        panel.setMinimumWidth(380)
        panel.setMaximumWidth(400)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        # Scroll Area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(10)

        # Titel
        title = QLabel('Parameter Einstellungen')
        title.setFont(QFont('Segoe UI', 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet('color: white;')
        scroll_layout.addWidget(title)

        # Gruppe 1: Sequenz-Parameter
        seq_group = self._create_seq_group()
        scroll_layout.addWidget(seq_group)

        # Gruppe 2: Feature-Auswahl
        feature_group = self._create_feature_group()
        scroll_layout.addWidget(feature_group)

        # Gruppe 3: HOLD-Samples
        hold_group = self._create_hold_group()
        scroll_layout.addWidget(hold_group)

        # Gruppe 4: Normalisierung
        norm_group = self._create_norm_group()
        scroll_layout.addWidget(norm_group)

        # Gruppe 5: Daten-Info
        data_group = self._create_data_info_group()
        scroll_layout.addWidget(data_group)

        # Status Label
        self.status_label = QLabel('Bitte Parameter einstellen und Vorschau berechnen.')
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet('color: #aaaaaa;')
        scroll_layout.addWidget(self.status_label)

        # Buttons
        self.preview_btn = QPushButton('Vorschau berechnen')
        self.preview_btn.setFont(QFont('Segoe UI', 11, QFont.Weight.Bold))
        self.preview_btn.setFixedHeight(45)
        self.preview_btn.setStyleSheet(self._button_style((0.3, 0.6, 0.9)))
        self.preview_btn.clicked.connect(self._calculate_preview)
        scroll_layout.addWidget(self.preview_btn)

        self.generate_btn = QPushButton('Daten generieren && Schliessen')
        self.generate_btn.setFont(QFont('Segoe UI', 11, QFont.Weight.Bold))
        self.generate_btn.setFixedHeight(45)
        self.generate_btn.setStyleSheet(self._button_style((0.2, 0.7, 0.3)))
        self.generate_btn.setEnabled(False)
        self.generate_btn.clicked.connect(self._generate_and_close)
        scroll_layout.addWidget(self.generate_btn)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        return panel

    def _create_seq_group(self) -> QGroupBox:
        """Erstellt die Sequenz-Parameter Gruppe."""
        group = QGroupBox('Sequenz-Parameter')
        group.setFont(QFont('Segoe UI', 13, QFont.Weight.Bold))
        group.setStyleSheet('''
            QGroupBox {
                color: #4da8da;
                border: 1px solid #333333;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        ''')
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        # Lookback
        lookback_layout = QHBoxLayout()
        lookback_label = QLabel('Lookback:')
        lookback_label.setFont(QFont('Segoe UI', 12, QFont.Weight.Bold))
        lookback_label.setFixedWidth(100)
        lookback_layout.addWidget(lookback_label)

        self.lookback_value = QLabel(str(self.params['lookback']))
        self.lookback_value.setFont(QFont('Segoe UI', 14, QFont.Weight.Bold))
        self.lookback_value.setStyleSheet('color: #7fef7f;')
        self.lookback_value.setFixedWidth(60)
        self.lookback_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lookback_layout.addWidget(self.lookback_value)

        lookback_btns = self._create_adjust_buttons('lookback')
        lookback_layout.addWidget(lookback_btns)

        layout.addLayout(lookback_layout)

        # Lookforward
        lookforward_layout = QHBoxLayout()
        lookforward_label = QLabel('Lookforward:')
        lookforward_label.setFont(QFont('Segoe UI', 12, QFont.Weight.Bold))
        lookforward_label.setFixedWidth(100)
        lookforward_layout.addWidget(lookforward_label)

        self.lookforward_value = QLabel(str(self.params['lookforward']))
        self.lookforward_value.setFont(QFont('Segoe UI', 14, QFont.Weight.Bold))
        self.lookforward_value.setStyleSheet('color: #7fef7f;')
        self.lookforward_value.setFixedWidth(60)
        self.lookforward_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lookforward_layout.addWidget(self.lookforward_value)

        lookforward_btns = self._create_adjust_buttons('lookforward')
        lookforward_layout.addWidget(lookforward_btns)

        layout.addLayout(lookforward_layout)

        # Sequenzlaenge Info
        self.seq_info_label = QLabel('')
        self.seq_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.seq_info_label.setStyleSheet('color: #aaaaaa;')
        layout.addWidget(self.seq_info_label)

        return group

    def _create_adjust_buttons(self, param_name: str) -> QWidget:
        """Erstellt +/- Buttons fuer Parameter-Anpassung."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Button-Definitionen: --, -, +, ++
        buttons = [
            ('--', -10, (0.8, 0.2, 0.2)),   # Dunkelrot
            ('-', -1, (0.6, 0.3, 0.3)),      # Hellrot
            ('+', 1, (0.2, 0.6, 0.2)),       # Hellgruen
            ('++', 10, (0.2, 0.8, 0.2)),     # Dunkelgruen
        ]

        for text, amount, color in buttons:
            btn = QPushButton(text)
            btn.setFixedSize(50, 30)
            btn.setFont(QFont('Segoe UI', 12, QFont.Weight.Bold))
            btn.setStyleSheet(self._button_style(color))
            btn.clicked.connect(lambda checked, p=param_name, a=amount:
                               self._adjust_param(p, a))
            layout.addWidget(btn)

        return widget

    def _adjust_param(self, param_name: str, amount: int):
        """Passt einen Parameter an."""
        new_value = max(1, self.params[param_name] + amount)
        self.params[param_name] = new_value

        if param_name == 'lookback':
            self.lookback_value.setText(str(new_value))
        elif param_name == 'lookforward':
            self.lookforward_value.setText(str(new_value))

        self._update_seq_info()
        self.preview_computed = False
        self.generate_btn.setEnabled(False)

    def _update_seq_info(self):
        """Aktualisiert die Sequenzlaenge-Anzeige."""
        seq_len = self.params['lookback'] + self.params['lookforward']
        self.seq_info_label.setText(f'Sequenzlaenge: {seq_len} Datenpunkte')

    def _create_feature_group(self) -> QGroupBox:
        """Erstellt die Feature-Auswahl Gruppe."""
        group = QGroupBox('Feature-Auswahl')
        group.setFont(QFont('Segoe UI', 13, QFont.Weight.Bold))
        group.setStyleSheet('''
            QGroupBox {
                color: #ffb366;
                border: 1px solid #333333;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        ''')
        layout = QGridLayout(group)
        layout.setSpacing(5)

        features = [
            ('use_close', 'Close'),
            ('use_high', 'High'),
            ('use_low', 'Low'),
            ('use_open', 'Open'),
            ('use_price_change', 'Preisaenderung'),
            ('use_price_change_pct', 'Aenderung (%)'),
        ]

        for i, (key, label) in enumerate(features):
            cb = QCheckBox(label)
            cb.setChecked(self.params[key])
            cb.setFont(QFont('Segoe UI', 12))
            cb.setStyleSheet('color: white;')
            cb.stateChanged.connect(lambda state, k=key: self._update_feature(k, state))
            layout.addWidget(cb, i // 2, i % 2)

        return group

    def _update_feature(self, key: str, state: int):
        """Aktualisiert eine Feature-Einstellung."""
        self.params[key] = state == Qt.CheckState.Checked.value
        self.preview_computed = False
        self.generate_btn.setEnabled(False)

    def _create_hold_group(self) -> QGroupBox:
        """Erstellt die HOLD-Samples Gruppe."""
        group = QGroupBox('HOLD-Samples (Negativ-Beispiele)')
        group.setFont(QFont('Segoe UI', 13, QFont.Weight.Bold))
        group.setStyleSheet('''
            QGroupBox {
                color: #e680e6;
                border: 1px solid #333333;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        ''')
        layout = QGridLayout(group)
        layout.setSpacing(5)

        # Include HOLD Checkbox
        self.include_hold_cb = QCheckBox('HOLD-Samples erstellen')
        self.include_hold_cb.setChecked(self.params['include_hold'])
        self.include_hold_cb.setFont(QFont('Segoe UI', 12))
        self.include_hold_cb.setStyleSheet('color: white;')
        self.include_hold_cb.stateChanged.connect(self._update_hold_enabled)
        layout.addWidget(self.include_hold_cb, 0, 0, 1, 2)

        # Auto Checkbox
        self.auto_cb = QCheckBox('Auto')
        self.auto_cb.setChecked(self.params['auto_gen'])
        self.auto_cb.setFont(QFont('Segoe UI', 12, QFont.Weight.Bold))
        self.auto_cb.setStyleSheet('color: #7fff7f;')
        self.auto_cb.setToolTip('Automatische Generierung (garantiert gewuenschtes Verhaeltnis)')
        self.auto_cb.stateChanged.connect(self._update_auto_gen)
        layout.addWidget(self.auto_cb, 0, 2)

        # Hold Ratio
        layout.addWidget(QLabel('Verhaeltnis zu Signalen:'), 1, 0)
        self.hold_ratio_slider = QSlider(Qt.Orientation.Horizontal)
        self.hold_ratio_slider.setRange(10, 300)
        self.hold_ratio_slider.setValue(int(self.params['hold_ratio'] * 100))
        self.hold_ratio_slider.valueChanged.connect(self._update_hold_ratio)
        layout.addWidget(self.hold_ratio_slider, 1, 1)

        self.hold_ratio_label = QLabel(f"{self.params['hold_ratio']:.1f}x")
        self.hold_ratio_label.setStyleSheet('color: white;')
        layout.addWidget(self.hold_ratio_label, 1, 2)

        # Min Distance Factor
        self.distance_title = QLabel('Min. Abstand Faktor:')
        self.distance_title.setStyleSheet('color: #999999;')
        layout.addWidget(self.distance_title, 2, 0)

        self.distance_slider = QSlider(Qt.Orientation.Horizontal)
        self.distance_slider.setRange(10, 50)
        self.distance_slider.setValue(int(self.params['min_distance_factor'] * 100))
        self.distance_slider.setEnabled(not self.params['auto_gen'])
        self.distance_slider.valueChanged.connect(self._update_distance)
        layout.addWidget(self.distance_slider, 2, 1)

        self.distance_label = QLabel(f"{self.params['min_distance_factor']:.1f}")
        self.distance_label.setStyleSheet('color: #999999;')
        layout.addWidget(self.distance_label, 2, 2)

        return group

    def _update_hold_enabled(self, state: int):
        """Aktualisiert ob HOLD-Samples erstellt werden."""
        self.params['include_hold'] = state == Qt.CheckState.Checked.value
        self.preview_computed = False
        self.generate_btn.setEnabled(False)

    def _update_auto_gen(self, state: int):
        """Aktualisiert Auto-Generierung."""
        self.params['auto_gen'] = state == Qt.CheckState.Checked.value
        self.distance_slider.setEnabled(not self.params['auto_gen'])
        self.preview_computed = False
        self.generate_btn.setEnabled(False)

    def _update_hold_ratio(self, value: int):
        """Aktualisiert das HOLD-Verhaeltnis."""
        self.params['hold_ratio'] = value / 100.0
        self.hold_ratio_label.setText(f"{self.params['hold_ratio']:.1f}x")
        self.preview_computed = False
        self.generate_btn.setEnabled(False)

    def _update_distance(self, value: int):
        """Aktualisiert den Mindestabstand-Faktor."""
        self.params['min_distance_factor'] = value / 100.0
        self.distance_label.setText(f"{self.params['min_distance_factor']:.1f}")
        self.preview_computed = False
        self.generate_btn.setEnabled(False)

    def _create_norm_group(self) -> QGroupBox:
        """Erstellt die Normalisierung Gruppe."""
        group = QGroupBox('Normalisierung')
        group.setFont(QFont('Segoe UI', 13, QFont.Weight.Bold))
        group.setStyleSheet('''
            QGroupBox {
                color: #7fe6b3;
                border: 1px solid #333333;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        ''')
        layout = QGridLayout(group)
        layout.setSpacing(5)

        # Methode
        layout.addWidget(QLabel('Methode:'), 0, 0)
        self.norm_combo = QComboBox()
        self.norm_combo.addItems(['Z-Score (Standard)', 'Min-Max [0,1]', 'Keine'])
        self.norm_combo.setCurrentIndex(0)
        self.norm_combo.currentIndexChanged.connect(self._update_norm_method)
        layout.addWidget(self.norm_combo, 0, 1)

        # Random Seed
        layout.addWidget(QLabel('Random Seed:'), 1, 0)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 99999)
        self.seed_spin.setValue(self.params['random_seed'])
        self.seed_spin.valueChanged.connect(self._update_seed)
        layout.addWidget(self.seed_spin, 1, 1)

        return group

    def _update_norm_method(self, index: int):
        """Aktualisiert die Normalisierungsmethode."""
        methods = ['zscore', 'minmax', 'none']
        self.params['normalize_method'] = methods[index]
        self.preview_computed = False
        self.generate_btn.setEnabled(False)

    def _update_seed(self, value: int):
        """Aktualisiert den Random Seed."""
        self.params['random_seed'] = value

    def _create_data_info_group(self) -> QGroupBox:
        """Erstellt die Daten-Info Gruppe."""
        group = QGroupBox('Geladene Daten')
        group.setFont(QFont('Segoe UI', 13, QFont.Weight.Bold))
        group.setStyleSheet('''
            QGroupBox {
                color: #aaaaaa;
                border: 1px solid #333333;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        ''')
        layout = QGridLayout(group)
        layout.setSpacing(3)

        # Datenpunkte
        layout.addWidget(QLabel('Datenpunkte:'), 0, 0)
        layout.addWidget(QLabel(f'{self.total_points:,}'), 0, 1)

        # Zeitraum
        layout.addWidget(QLabel('Zeitraum:'), 1, 0)
        if hasattr(self.data.index, 'strftime'):
            start = self.data.index[0].strftime('%d.%m.%y')
            end = self.data.index[-1].strftime('%d.%m.%y')
        else:
            start = str(self.data.index[0])[:10]
            end = str(self.data.index[-1])[:10]
        layout.addWidget(QLabel(f'{start} - {end}'), 1, 1)

        # Preisbereich
        layout.addWidget(QLabel('Preisbereich:'), 2, 0)
        close_col = 'Close' if 'Close' in self.data.columns else 'close'
        if close_col in self.data.columns:
            min_price = self.data[close_col].min()
            max_price = self.data[close_col].max()
            layout.addWidget(QLabel(f'{min_price:.0f} - {max_price:.0f} USD'), 2, 1)
        else:
            layout.addWidget(QLabel('-'), 2, 1)

        # Tages-Extrema
        layout.addWidget(QLabel('Tages-Extrema:'), 3, 0)
        self.extrema_label = QLabel('Berechne...')
        self.extrema_label.setStyleSheet('color: #7fe67f;')
        layout.addWidget(self.extrema_label, 3, 1)

        return group

    def _calculate_extrema(self):
        """Berechnet die Anzahl der Tages-Extrema."""
        try:
            # Vereinfachte Berechnung: Lokale Hochs und Tiefs
            close_col = 'Close' if 'Close' in self.data.columns else 'close'
            if close_col not in self.data.columns:
                self.extrema_label.setText('Keine Close-Daten')
                return

            prices = self.data[close_col].values

            # Finde lokale Maxima und Minima
            highs = 0
            lows = 0

            for i in range(1, len(prices) - 1):
                if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                    highs += 1
                if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                    lows += 1

            self.extrema_label.setText(f'{highs} Hochs, {lows} Tiefs')
        except Exception as e:
            self.extrema_label.setText(f'Fehler: {e}')

    def _create_chart_panel(self) -> QWidget:
        """Erstellt das mittlere Chart-Panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Matplotlib Figure
        self.figure = Figure(figsize=(8, 6), facecolor='#262626')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self._style_axis(self.ax)

        layout.addWidget(self.canvas)

        return panel

    def _style_axis(self, ax):
        """Stylt eine Matplotlib-Achse im Dark Theme."""
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#444444')

    def _create_stats_panel(self) -> QWidget:
        """Erstellt das rechte Statistik-Panel."""
        panel = QWidget()
        panel.setMinimumWidth(320)
        panel.setMaximumWidth(350)
        layout = QVBoxLayout(panel)

        # Statistik-Tabelle
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(['Parameter', 'Wert'])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.stats_table.setAlternatingRowColors(True)
        self.stats_table.setStyleSheet('''
            QTableWidget {
                background-color: #1a1a1a;
                color: white;
                gridline-color: #333333;
            }
            QTableWidget::item:alternate {
                background-color: #262626;
            }
            QHeaderView::section {
                background-color: #333333;
                color: white;
                padding: 5px;
                border: none;
            }
        ''')

        # Initiale Zeilen
        initial_stats = [
            ('BUY Signale', '-'),
            ('SELL Signale', '-'),
            ('HOLD Samples', '-'),
            ('Gesamt', '-'),
            ('---', '---'),
            ('Sequenzlaenge', str(self.params['lookback'] + self.params['lookforward'])),
            ('Features', '-'),
            ('Input Shape', '-'),
            ('Balance', '-'),
            ('---', '---'),
            ('Lookback', str(self.params['lookback'])),
            ('Lookforward', str(self.params['lookforward'])),
            ('Normalisierung', 'Z-Score'),
        ]

        self.stats_table.setRowCount(len(initial_stats))
        for i, (param, value) in enumerate(initial_stats):
            self.stats_table.setItem(i, 0, QTableWidgetItem(param))
            self.stats_table.setItem(i, 1, QTableWidgetItem(value))

        layout.addWidget(self.stats_table)

        # Legende
        legend_group = QGroupBox('Legende')
        legend_group.setFont(QFont('Segoe UI', 11, QFont.Weight.Bold))
        legend_group.setStyleSheet('color: white;')
        legend_layout = QGridLayout(legend_group)

        # BUY
        buy_marker = QLabel('^')
        buy_marker.setFont(QFont('Segoe UI', 16, QFont.Weight.Bold))
        buy_marker.setStyleSheet('color: #33cc33;')
        legend_layout.addWidget(buy_marker, 0, 0)
        legend_layout.addWidget(QLabel('BUY (Hoch)'), 0, 1)

        # SELL
        sell_marker = QLabel('v')
        sell_marker.setFont(QFont('Segoe UI', 16, QFont.Weight.Bold))
        sell_marker.setStyleSheet('color: #cc3333;')
        legend_layout.addWidget(sell_marker, 1, 0)
        legend_layout.addWidget(QLabel('SELL (Tief)'), 1, 1)

        # Ungueltig
        invalid_marker = QLabel('o')
        invalid_marker.setFont(QFont('Segoe UI', 16))
        invalid_marker.setStyleSheet('color: #808080;')
        legend_layout.addWidget(invalid_marker, 2, 0)
        legend_layout.addWidget(QLabel('Ungueltig (Randbereich)'), 2, 1)

        layout.addWidget(legend_group)

        return panel

    def _calculate_preview(self):
        """Berechnet die Vorschau der Trainingsdaten."""
        self.status_label.setText('Berechne Vorschau...')
        self.status_label.setStyleSheet('color: #4da8da;')

        try:
            from ..training.labeler import DailyExtremaLabeler

            close_col = 'Close' if 'Close' in self.data.columns else 'close'
            prices = self.data[close_col].values

            lookback = self.params['lookback']
            lookforward = self.params['lookforward']

            # Verwende echten Labeler
            labeler = DailyExtremaLabeler(
                lookforward=lookforward,
                threshold_pct=2.0  # Standard Schwellwert
            )

            # Generiere Labels basierend auf Auto-Modus
            if self.params['auto_gen']:
                # Auto-Modus: Future Return Methode
                labels = labeler.generate_labels(self.data, method='future_return')
            else:
                # Manuell: Extrema-basiert
                labels = labeler.generate_labels(self.data, method='extrema')

            # Finde Signal-Indizes (ohne Randbereich)
            buy_indices = []
            sell_indices = []

            for i in range(lookback, len(prices) - lookforward):
                if labels[i] == 1:  # BUY
                    buy_indices.append(i)
                elif labels[i] == 2:  # SELL
                    sell_indices.append(i)

            num_buy = len(buy_indices)
            num_sell = len(sell_indices)
            num_hold = int((num_buy + num_sell) * self.params['hold_ratio']) if self.params['include_hold'] else 0
            total = num_buy + num_sell + num_hold

            # Feature-Anzahl
            num_features = sum([
                self.params['use_close'],
                self.params['use_high'],
                self.params['use_low'],
                self.params['use_open'],
                self.params['use_price_change'],
                self.params['use_price_change_pct'],
            ])

            seq_len = lookback + lookforward

            # Speichere Info
            self.result_info = {
                'num_buy': num_buy,
                'num_sell': num_sell,
                'num_hold': num_hold,
                'total': total,
                'seq_len': seq_len,
                'num_features': num_features,
                'buy_indices': buy_indices,
                'sell_indices': sell_indices,
            }

            # Update Statistik-Tabelle
            stats = [
                ('BUY Signale', str(num_buy)),
                ('SELL Signale', str(num_sell)),
                ('HOLD Samples', str(num_hold)),
                ('Gesamt', str(total)),
                ('---', '---'),
                ('Sequenzlaenge', str(seq_len)),
                ('Features', str(num_features)),
                ('Input Shape', f'{total} x {seq_len} x {num_features}'),
                ('Balance', f'{num_buy}:{num_sell}:{num_hold}'),
                ('---', '---'),
                ('Lookback', str(lookback)),
                ('Lookforward', str(lookforward)),
                ('Normalisierung', self.params['normalize_method']),
            ]

            for i, (param, value) in enumerate(stats):
                self.stats_table.item(i, 1).setText(value)

            # Update Chart
            self._update_chart(prices, buy_indices, sell_indices)

            self.preview_computed = True
            self.generate_btn.setEnabled(True)
            self.status_label.setText(f'Vorschau berechnet: {total} Sequenzen')
            self.status_label.setStyleSheet('color: #68d391;')

        except Exception as e:
            self.status_label.setText(f'Fehler: {e}')
            self.status_label.setStyleSheet('color: #fc8181;')

    def _update_chart(self, prices, buy_indices, sell_indices):
        """Aktualisiert den Chart mit Signalen."""
        self.ax.clear()
        self._style_axis(self.ax)

        # Preis-Linie
        self.ax.plot(prices, color='white', linewidth=0.5, alpha=0.8)

        # BUY Marker
        if buy_indices:
            self.ax.scatter(buy_indices, prices[buy_indices],
                           marker='^', color='#33cc33', s=50, zorder=5,
                           label=f'BUY ({len(buy_indices)})')

        # SELL Marker
        if sell_indices:
            self.ax.scatter(sell_indices, prices[sell_indices],
                           marker='v', color='#cc3333', s=50, zorder=5,
                           label=f'SELL ({len(sell_indices)})')

        self.ax.set_title(f'Signal-Vorschau: {len(buy_indices)} BUY, {len(sell_indices)} SELL',
                         color='white', fontsize=12)
        self.ax.legend(loc='upper left', facecolor='#333333', edgecolor='#555555',
                      labelcolor='white')

        self.figure.tight_layout()
        self.canvas.draw()

    def _generate_and_close(self):
        """Generiert die Trainingsdaten und schliesst das Fenster."""
        if not self.preview_computed:
            return

        self.status_label.setText('Generiere Trainingsdaten...')

        try:
            # Hier wuerde die eigentliche Daten-Generierung stattfinden
            # Demo: Erstelle Dummy-Daten
            training_data = {
                'X': np.random.randn(self.result_info['total'],
                                    self.result_info['seq_len'],
                                    self.result_info['num_features']),
                'Y': np.random.randint(0, 3, self.result_info['total']),
                'params': self.params,
            }

            training_info = self.result_info.copy()
            training_info['params'] = self.params

            # Signal senden
            self.data_prepared.emit(training_data, training_info)

            self.close()

        except Exception as e:
            self.status_label.setText(f'Fehler: {e}')
            self.status_label.setStyleSheet('color: #fc8181;')

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
            QGroupBox {
                background-color: #333333;
            }
            QScrollArea {
                background-color: #2e2e2e;
            }
        '''
