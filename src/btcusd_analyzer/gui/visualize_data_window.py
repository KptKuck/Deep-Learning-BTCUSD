"""
Visualize Data Window - Trainingsdaten Visualisierung
Portiert von MATLAB visualize_training_data_gui.m
"""

from typing import Optional, Dict, Any, List

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox, QScrollArea, QFrame, QSplitter, QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

import pandas as pd
import numpy as np

# Matplotlib fuer Charts
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .styles import StyleFactory


class VisualizeDataWindow(QMainWindow):
    """
    Fenster zur Visualisierung der Trainingsdaten.

    Layout: 2 Spalten (flexible | 280px)
    - Links: 2 Charts (BTCUSD + Aktuelle Sequenz)
    - Rechts: Navigation (Signal-Filter, Zoom, Achsen-Skalierung)
    """

    # Signal fuer Log-Meldungen an MainWindow
    log_message = pyqtSignal(str, str)  # message, level

    def __init__(self, data: pd.DataFrame, training_data: Dict,
                 training_info: Dict, parent=None):
        super().__init__(parent)
        self._parent = parent

        self.data = data
        self.training_data = training_data
        self.training_info = training_info

        # Navigation State
        self.current_index = 0
        self.filter_mode = 'ALL'  # 'ALL', 'BUY', 'SELL'
        self.zoom_level = 1.0
        self.x_scale = 1.0
        self.y_scale = 1.0

        # Signal-Listen
        self.buy_indices = training_info.get('buy_indices', [])
        self.sell_indices = training_info.get('sell_indices', [])
        self.all_signals = self._merge_signals()

        # UI initialisieren
        self._init_ui()
        self._update_charts()

    def _log(self, message: str, level: str = 'INFO'):
        """Loggt eine Nachricht an MainWindow."""
        if self._parent and hasattr(self._parent, '_log'):
            self._parent._log(f'[Visualize] {message}', level)
        self.log_message.emit(message, level)

    def _merge_signals(self) -> List[tuple]:
        """Erstellt eine sortierte Liste aller Signale."""
        signals = []
        for idx in self.buy_indices:
            signals.append((idx, 'BUY'))
        for idx in self.sell_indices:
            signals.append((idx, 'SELL'))
        signals.sort(key=lambda x: x[0])
        return signals

    def _get_filtered_signals(self) -> List[tuple]:
        """Gibt die gefilterten Signale zurueck."""
        if self.filter_mode == 'ALL':
            return self.all_signals
        elif self.filter_mode == 'BUY':
            return [(idx, 'BUY') for idx, t in self.all_signals if t == 'BUY']
        elif self.filter_mode == 'SELL':
            return [(idx, 'SELL') for idx, t in self.all_signals if t == 'SELL']
        return self.all_signals

    def _init_ui(self):
        """Initialisiert die UI-Komponenten."""
        self.setWindowTitle('5 - Visualize')

        # Relative Fenstergroesse (95% Hoehe, 90% Breite)
        screen = QApplication.primaryScreen()
        if screen:
            screen_rect = screen.availableGeometry()
            window_width = int(screen_rect.width() * 0.90)
            window_height = int(screen_rect.height() * 0.95)
        else:
            window_width, window_height = 1400, 900

        self.setMinimumSize(1200, 800)
        self.resize(window_width, window_height)
        self.setStyleSheet(self._get_stylesheet())

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main Layout: 2 Spalten
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Linke Spalte: Charts
        left_panel = self._create_chart_panel()
        splitter.addWidget(left_panel)

        # Rechte Spalte: Navigation
        right_panel = self._create_nav_panel()
        splitter.addWidget(right_panel)

        # Splitter Proportionen
        splitter.setSizes([1100, 280])

    def _create_chart_panel(self) -> QWidget:
        """Erstellt das linke Chart-Panel mit 2 Charts."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        # Chart 1: BTCUSD mit Signalen (2x Hoehe)
        self.figure1 = Figure(figsize=(10, 4), facecolor='#262626')
        self.canvas1 = FigureCanvas(self.figure1)
        self.ax1 = self.figure1.add_subplot(111)
        self._style_axis(self.ax1)
        layout.addWidget(self.canvas1, stretch=2)

        # Chart 2: Aktuelle Sequenz (1x Hoehe)
        self.figure2 = Figure(figsize=(10, 3), facecolor='#262626')
        self.canvas2 = FigureCanvas(self.figure2)
        self.ax2 = self.figure2.add_subplot(111)
        self._style_axis(self.ax2)
        layout.addWidget(self.canvas2, stretch=1)

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

    def _create_nav_panel(self) -> QWidget:
        """Erstellt das rechte Navigations-Panel."""
        panel = QWidget()
        panel.setMinimumWidth(280)
        panel.setMaximumWidth(300)
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        # Scroll Area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(15)

        # === Signal-Info ===
        info_group = self._create_info_group()
        scroll_layout.addWidget(info_group)

        # === Schrittmodus ===
        step_group = self._create_step_group()
        scroll_layout.addWidget(step_group)

        # === Chart Zoom ===
        zoom_group = self._create_zoom_group()
        scroll_layout.addWidget(zoom_group)

        # === Achsen-Skalierung ===
        scale_group = self._create_scale_group()
        scroll_layout.addWidget(scale_group)

        # === Sequenz-Info ===
        seq_info_group = self._create_seq_info_group()
        scroll_layout.addWidget(seq_info_group)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        return panel

    def _create_info_group(self) -> QGroupBox:
        """Erstellt die Signal-Info Gruppe."""
        group = QGroupBox('Signal-Info')
        group.setFont(QFont('Segoe UI', 12, QFont.Weight.Bold))
        group.setStyleSheet(self._group_style('#4da8da'))
        layout = QVBoxLayout(group)

        # Statistik
        num_buy = len(self.buy_indices)
        num_sell = len(self.sell_indices)
        self.stats_label = QLabel(f'BUY: {num_buy} | SELL: {num_sell}')
        self.stats_label.setStyleSheet('color: white;')
        layout.addWidget(self.stats_label)

        # Aktuelles Signal
        self.signal_label = QLabel('Signal: -')
        self.signal_label.setStyleSheet('color: white;')
        layout.addWidget(self.signal_label)

        # Input Range
        self.range_label = QLabel('Input: [-]')
        self.range_label.setStyleSheet('color: #aaaaaa;')
        layout.addWidget(self.range_label)

        return group

    def _create_step_group(self) -> QGroupBox:
        """Erstellt die Schrittmodus Gruppe."""
        group = QGroupBox('Schrittmodus')
        group.setFont(QFont('Segoe UI', 12, QFont.Weight.Bold))
        group.setStyleSheet(self._group_style('#33b34d'))
        layout = QVBoxLayout(group)

        # Filter-Buttons mit Navigation
        filters = [
            ('ALL', 'Alle', '#808080'),
            ('BUY', 'BUY', '#33cc33'),
            ('SELL', 'SELL', '#cc3333'),
        ]

        for filter_mode, label, color in filters:
            row = QHBoxLayout()

            # Zurueck
            prev_btn = QPushButton('<')
            prev_btn.setFixedSize(40, 30)
            prev_btn.setStyleSheet(self._button_style_hex('#555555'))
            prev_btn.clicked.connect(lambda checked, m=filter_mode: self._prev_signal(m))
            row.addWidget(prev_btn)

            # Label
            filter_btn = QPushButton(label)
            filter_btn.setStyleSheet(self._button_style_hex(color))
            filter_btn.clicked.connect(lambda checked, m=filter_mode: self._set_filter(m))
            row.addWidget(filter_btn, stretch=1)

            # Vorwaerts
            next_btn = QPushButton('>')
            next_btn.setFixedSize(40, 30)
            next_btn.setStyleSheet(self._button_style_hex('#555555'))
            next_btn.clicked.connect(lambda checked, m=filter_mode: self._next_signal(m))
            row.addWidget(next_btn)

            layout.addLayout(row)

        # Position Anzeige
        self.position_label = QLabel('0 / 0')
        self.position_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.position_label.setFont(QFont('Segoe UI', 12, QFont.Weight.Bold))
        self.position_label.setStyleSheet('color: white;')
        layout.addWidget(self.position_label)

        return group

    def _create_zoom_group(self) -> QGroupBox:
        """Erstellt die Chart Zoom Gruppe."""
        group = QGroupBox('Chart Zoom')
        group.setFont(QFont('Segoe UI', 12, QFont.Weight.Bold))
        group.setStyleSheet(self._group_style('#6666cc'))
        layout = QGridLayout(group)

        # Zoom Buttons
        zoom_in = QPushButton('Zoom In')
        zoom_in.setStyleSheet(self._button_style_hex('#4da8da'))
        zoom_in.clicked.connect(self._zoom_in)
        layout.addWidget(zoom_in, 0, 0)

        zoom_out = QPushButton('Zoom Out')
        zoom_out.setStyleSheet(self._button_style_hex('#4da8da'))
        zoom_out.clicked.connect(self._zoom_out)
        layout.addWidget(zoom_out, 0, 1)

        zoom_reset = QPushButton('Reset')
        zoom_reset.setStyleSheet(self._button_style_hex('#808080'))
        zoom_reset.clicked.connect(self._zoom_reset)
        layout.addWidget(zoom_reset, 0, 2)

        # Zoom Level Anzeige
        self.zoom_label = QLabel('Zoom: 100%')
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_label.setStyleSheet('color: white;')
        layout.addWidget(self.zoom_label, 1, 0, 1, 3)

        return group

    def _create_scale_group(self) -> QGroupBox:
        """Erstellt die Achsen-Skalierung Gruppe."""
        group = QGroupBox('Achsen-Skalierung')
        group.setFont(QFont('Segoe UI', 12, QFont.Weight.Bold))
        group.setStyleSheet(self._group_style('#e6b333'))
        layout = QGridLayout(group)

        # X-Achse
        x_plus = QPushButton('X+')
        x_plus.setStyleSheet(self._button_style_hex('#4da8da'))
        x_plus.clicked.connect(lambda: self._scale_axis('x', 0.7))
        layout.addWidget(x_plus, 0, 0)

        x_minus = QPushButton('X-')
        x_minus.setStyleSheet(self._button_style_hex('#4da8da'))
        x_minus.clicked.connect(lambda: self._scale_axis('x', 1.4))
        layout.addWidget(x_minus, 0, 1)

        # Y-Achse
        y_plus = QPushButton('Y+')
        y_plus.setStyleSheet(self._button_style_hex('#33b34d'))
        y_plus.clicked.connect(lambda: self._scale_axis('y', 0.7))
        layout.addWidget(y_plus, 1, 0)

        y_minus = QPushButton('Y-')
        y_minus.setStyleSheet(self._button_style_hex('#33b34d'))
        y_minus.clicked.connect(lambda: self._scale_axis('y', 1.4))
        layout.addWidget(y_minus, 1, 1)

        # Reset
        scale_reset = QPushButton('Reset')
        scale_reset.setStyleSheet(self._button_style_hex('#808080'))
        scale_reset.clicked.connect(self._scale_reset)
        layout.addWidget(scale_reset, 2, 0, 1, 2)

        return group

    def _create_seq_info_group(self) -> QGroupBox:
        """Erstellt die Sequenz-Info Gruppe."""
        group = QGroupBox('Sequenz-Info')
        group.setFont(QFont('Segoe UI', 12, QFont.Weight.Bold))
        group.setStyleSheet(self._group_style('#aaaaaa'))
        layout = QVBoxLayout(group)

        lookback = self.training_info.get('params', {}).get('lookback', 50)
        lookforward = self.training_info.get('params', {}).get('lookforward', 100)

        self.seq_label = QLabel(f'Seq: {lookback}+{lookforward}')
        self.seq_label.setStyleSheet('color: white;')
        layout.addWidget(self.seq_label)

        self.idx_label = QLabel('Idx: - | Seq: -')
        self.idx_label.setStyleSheet('color: #aaaaaa;')
        layout.addWidget(self.idx_label)

        return group

    def _group_style(self, color: str) -> str:
        """Generiert GroupBox-Style."""
        return StyleFactory.group_style(hex_color=color)

    def _button_style_hex(self, color: str) -> str:
        """Generiert Button-Style aus Hex-Farbe."""
        return StyleFactory.button_style_hex(color, padding='5px 10px')

    # === Navigation ===

    def _set_filter(self, mode: str):
        """Setzt den Filter-Modus."""
        self.filter_mode = mode
        self.current_index = 0
        self._update_charts()
        self._update_position_label()

    def _prev_signal(self, mode: str):
        """Geht zum vorherigen Signal."""
        self.filter_mode = mode
        signals = self._get_filtered_signals()
        if signals and self.current_index > 0:
            self.current_index -= 1
            self._update_charts()
        self._update_position_label()

    def _next_signal(self, mode: str):
        """Geht zum naechsten Signal."""
        self.filter_mode = mode
        signals = self._get_filtered_signals()
        if signals and self.current_index < len(signals) - 1:
            self.current_index += 1
            self._update_charts()
        self._update_position_label()

    def _update_position_label(self):
        """Aktualisiert die Positions-Anzeige."""
        signals = self._get_filtered_signals()
        total = len(signals)
        current = self.current_index + 1 if total > 0 else 0
        self.position_label.setText(f'{current} / {total}')

    # === Zoom ===

    def _zoom_in(self):
        """Zoomt hinein."""
        self.zoom_level *= 0.5
        self.zoom_level = max(0.1, self.zoom_level)
        self.zoom_label.setText(f'Zoom: {int(100/self.zoom_level)}%')
        self._update_charts()

    def _zoom_out(self):
        """Zoomt heraus."""
        self.zoom_level *= 2.0
        self.zoom_level = min(10.0, self.zoom_level)
        self.zoom_label.setText(f'Zoom: {int(100/self.zoom_level)}%')
        self._update_charts()

    def _zoom_reset(self):
        """Setzt den Zoom zurueck."""
        self.zoom_level = 1.0
        self.zoom_label.setText('Zoom: 100%')
        self._update_charts()

    # === Achsen-Skalierung ===

    def _scale_axis(self, axis: str, factor: float):
        """Skaliert eine Achse."""
        if axis == 'x':
            self.x_scale *= factor
            self.x_scale = max(0.2, min(5.0, self.x_scale))
        else:
            self.y_scale *= factor
            self.y_scale = max(0.2, min(5.0, self.y_scale))
        self._update_charts()

    def _scale_reset(self):
        """Setzt die Skalierung zurueck."""
        self.x_scale = 1.0
        self.y_scale = 1.0
        self._update_charts()

    # === Charts ===

    def _update_charts(self):
        """Aktualisiert beide Charts."""
        signals = self._get_filtered_signals()

        if not signals:
            return

        if self.current_index >= len(signals):
            self.current_index = len(signals) - 1

        current_signal = signals[self.current_index]
        signal_idx, signal_type = current_signal

        # Chart 1: BTCUSD mit allen Signalen
        self._update_main_chart(signal_idx)

        # Chart 2: Aktuelle Sequenz
        self._update_sequence_chart(signal_idx, signal_type)

        # Update Labels
        self.signal_label.setText(f'Signal: {signal_type} @ Index {signal_idx}')
        self._update_position_label()

    def _update_main_chart(self, highlight_idx: int):
        """Aktualisiert den Haupt-Chart."""
        self.ax1.clear()
        self._style_axis(self.ax1)

        close_col = 'Close' if 'Close' in self.data.columns else 'close'
        prices = self.data[close_col].values

        # Bestimme Anzeigebereich basierend auf Zoom
        total_len = len(prices)
        window_size = int(total_len * self.zoom_level)
        center = highlight_idx

        start = max(0, center - window_size // 2)
        end = min(total_len, start + window_size)
        start = max(0, end - window_size)

        # Preis-Linie
        x_range = range(start, end)
        self.ax1.plot(x_range, prices[start:end], color='#4da8da', linewidth=1)

        # BUY Marker
        buy_in_range = [i for i in self.buy_indices if start <= i < end]
        if buy_in_range:
            self.ax1.scatter(buy_in_range, prices[buy_in_range],
                           marker='^', color='#33cc33', s=80, zorder=5,
                           label='BUY')

        # SELL Marker
        sell_in_range = [i for i in self.sell_indices if start <= i < end]
        if sell_in_range:
            self.ax1.scatter(sell_in_range, prices[sell_in_range],
                           marker='v', color='#cc3333', s=80, zorder=5,
                           label='SELL')

        # Highlight aktuelles Signal
        if start <= highlight_idx < end:
            self.ax1.scatter([highlight_idx], [prices[highlight_idx]],
                           marker='o', facecolors='none', edgecolors='yellow',
                           s=200, linewidths=3, zorder=10, label='Aktuell')

        self.ax1.set_title('BTCUSD Chart mit Training-Signalen', color='white')
        self.ax1.legend(loc='upper left', facecolor='#333333', edgecolor='#555555',
                       labelcolor='white')

        self.figure1.tight_layout()
        self.canvas1.draw()

    def _update_sequence_chart(self, signal_idx: int, signal_type: str):
        """Aktualisiert den Sequenz-Chart."""
        self.ax2.clear()
        self._style_axis(self.ax2)

        close_col = 'Close' if 'Close' in self.data.columns else 'close'
        prices = self.data[close_col].values

        lookback = self.training_info.get('params', {}).get('lookback', 50)
        lookforward = self.training_info.get('params', {}).get('lookforward', 100)

        start = max(0, signal_idx - lookback)
        end = min(len(prices), signal_idx + lookforward)

        seq_prices = prices[start:end]

        # Normalisiere fuer Anzeige
        if len(seq_prices) > 0:
            seq_normalized = (seq_prices - seq_prices.min()) / (seq_prices.max() - seq_prices.min() + 1e-8)
        else:
            seq_normalized = seq_prices

        x = range(len(seq_normalized))

        # Preis-Linie
        self.ax2.plot(x, seq_normalized, color='#4da8da', linewidth=2)

        # Markiere Signal-Position
        signal_pos = signal_idx - start
        if 0 <= signal_pos < len(seq_normalized):
            color = '#33cc33' if signal_type == 'BUY' else '#cc3333'
            marker = '^' if signal_type == 'BUY' else 'v'
            self.ax2.scatter([signal_pos], [seq_normalized[signal_pos]],
                           marker=marker, color=color, s=150, zorder=5)
            self.ax2.axvline(x=signal_pos, color=color, linestyle='--', alpha=0.5)

        # Bereiche markieren
        self.ax2.axvspan(0, lookback, alpha=0.1, color='green', label='Lookback')
        self.ax2.axvspan(lookback, len(seq_normalized), alpha=0.1, color='orange', label='Lookforward')

        self.ax2.set_title(f'Sequenz: {signal_type} Signal', color='white')
        self.ax2.set_xlabel('Sequenz-Index', color='white')
        self.ax2.set_ylabel('Normalisierter Preis', color='white')

        self.figure2.tight_layout()
        self.canvas2.draw()

        # Update Idx Label
        self.idx_label.setText(f'Idx: {signal_idx} | Seq: {start}-{end}')

    def _get_stylesheet(self) -> str:
        """Gibt das Fenster-Stylesheet zurueck."""
        return StyleFactory.window_style()
