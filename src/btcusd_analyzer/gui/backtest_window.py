"""
Backtest Window - GUI fuer Backtesting mit Live-Simulation (MATLAB-Style)

Layout: 3-Spalten (280px | flexible | 280px)
- Links: Steuerung (Start/Stop/Einzelschritt/Reset, Geschwindigkeit, Position-Info, Trade-Log)
- Mitte: Charts (Preis+Signale, Equity-Kurve)
- Rechts: Performance-Statistiken (P/L, Trade-Stats, Signal-Verteilung)
"""

import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QSlider, QProgressBar, QTextEdit,
    QSplitter, QCheckBox, QScrollArea, QTabWidget, QDialog, QTableWidget,
    QTableWidgetItem, QHeaderView, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QColor

import pandas as pd
import numpy as np
import pyqtgraph as pg

from .styles import get_stylesheet, COLORS, StyleFactory


class BacktestWindow(QMainWindow):
    """
    Backtest-Fenster mit Live-Simulation (MATLAB-Style).

    Features:
    - Schritt-fuer-Schritt Durchlauf der Daten
    - Start/Stop/Einzelschritt Steuerung
    - Gewinn/Verlust Berechnung
    - Visualisierung von Trades und Equity-Kurve
    - Performance-Statistiken
    """

    # Signal fuer Log-Meldungen an MainWindow
    log_message = pyqtSignal(str, str)  # message, level

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("4 - Backtest")
        self.setMinimumSize(1200, 800)
        self._parent = parent

        # Daten
        self.data: Optional[pd.DataFrame] = None
        self.signals: Optional[pd.Series] = None
        self.model = None
        self.model_info: Optional[Dict] = None

        # Backtester-Status
        self.is_running = False
        self.is_paused = False
        self.current_index = 0
        self.sequence_length = 0

        # Trading-Status
        self.position = 'NONE'  # 'NONE', 'LONG', 'SHORT'
        self.entry_price = 0.0
        self.entry_index = 0
        self.total_pnl = 0.0
        self.trades: List[Dict] = []
        self.signal_history: List[Dict] = []

        # Kapital und Equity
        self.initial_capital = 10000.0
        self.current_equity = self.initial_capital
        self.equity_curve: List[float] = [self.initial_capital]
        self.equity_indices: List[int] = [0]

        # Signal-Zaehler
        self.buy_count = 0
        self.sell_count = 0
        self.hold_count = 0

        # Geschwindigkeit
        self.steps_per_second = 10
        self.turbo_mode = False

        # Geschwindigkeitsmessung
        self._step_count = 0
        self._last_speed_update = 0.0
        self._speed_update_timer: Optional[QTimer] = None

        # Vorbereitete Sequenzen fuer Modell-Vorhersage
        self.prepared_sequences = None
        self.sequence_offset = 0  # Index-Offset zwischen Daten und Sequenzen

        # Timer
        self.backtest_timer: Optional[QTimer] = None

        # DEBUG-Modus
        self.debug_mode = False

        # Signal-Invertierung
        self.invert_signals = False

        self._setup_ui()
        self.setStyleSheet(get_stylesheet())

    def _log(self, message: str, level: str = 'INFO'):
        """Loggt eine Nachricht an MainWindow und lokales Log."""
        # An MainWindow senden (falls parent _log hat)
        if self._parent and hasattr(self._parent, '_log'):
            self._parent._log(f'[Backtest] {message}', level)
        # Signal emittieren
        self.log_message.emit(message, level)
        # Auch lokal loggen falls vorhanden
        if hasattr(self, 'trade_log'):
            self.trade_log.append(f'[{level}] {message}')

    def _debug(self, message: str):
        """Loggt DEBUG-Nachricht nur wenn debug_mode aktiv ist."""
        if self.debug_mode:
            self._log(f'[DEBUG] {message}', 'DEBUG')

    def _setup_ui(self):
        """Erstellt die 3-Spalten Benutzeroberflaeche."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 3-Spalten Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Linke Spalte: Steuerung (280px)
        left_panel = self._create_control_panel()
        left_panel.setFixedWidth(280)

        # Mitte: Charts (flexibel)
        center_panel = self._create_chart_panel()

        # Rechte Spalte: Statistiken (280px)
        right_panel = self._create_stats_panel()
        right_panel.setFixedWidth(280)

        splitter.addWidget(left_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([280, 640, 280])

        main_layout.addWidget(splitter)

    def _create_control_panel(self) -> QWidget:
        """Erstellt das Steuerungs-Panel (linke Spalte)."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        panel = QWidget()
        panel.setStyleSheet(f"background-color: rgb(46, 46, 46);")
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # Titel
        title = QLabel("Backtester Steuerung")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: white;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # === Steuerungs-Buttons ===
        control_group = QGroupBox("Steuerung")
        control_group.setStyleSheet(self._group_style((0.3, 0.7, 1)))
        control_layout = QGridLayout(control_group)
        control_layout.setSpacing(8)

        # Start Button
        self.start_btn = QPushButton("Start")
        self.start_btn.setStyleSheet(self._button_style((0.2, 0.7, 0.3)))
        self.start_btn.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.start_btn.clicked.connect(self._start_backtest)
        control_layout.addWidget(self.start_btn, 0, 0)

        # Stop Button
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet(self._button_style((0.8, 0.3, 0.2)))
        self.stop_btn.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_backtest)
        control_layout.addWidget(self.stop_btn, 0, 1)

        # Einzelschritt Button
        self.step_btn = QPushButton("Einzelschritt")
        self.step_btn.setStyleSheet(self._button_style((0.5, 0.5, 0.7)))
        self.step_btn.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.step_btn.clicked.connect(self._single_step)
        control_layout.addWidget(self.step_btn, 1, 0)

        # Reset Button
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setStyleSheet(self._button_style((0.5, 0.5, 0.5)))
        self.reset_btn.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.reset_btn.clicked.connect(self._reset_backtest)
        control_layout.addWidget(self.reset_btn, 1, 1)

        # Trade-Statistik Button
        self.stats_btn = QPushButton("Trade-Statistik")
        self.stats_btn.setStyleSheet(self._button_style((0.3, 0.6, 0.9)))
        self.stats_btn.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.stats_btn.clicked.connect(self._show_trade_statistics)
        control_layout.addWidget(self.stats_btn, 2, 0, 1, 2)  # Volle Breite

        layout.addWidget(control_group)

        # === Geschwindigkeit ===
        speed_group = QGroupBox("Geschwindigkeit")
        speed_group.setStyleSheet(self._group_style((0.9, 0.7, 0.3)))
        speed_layout = QVBoxLayout(speed_group)

        # Slider mit Label
        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Schritte/Sek:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 500)
        self.speed_slider.setValue(10)
        self.speed_slider.valueChanged.connect(self._update_speed)
        slider_row.addWidget(self.speed_slider)
        self.speed_label = QLabel("10")
        self.speed_label.setStyleSheet("color: white; min-width: 30px;")
        slider_row.addWidget(self.speed_label)
        speed_layout.addLayout(slider_row)

        # Turbo-Modus
        self.turbo_check = QCheckBox("Turbo-Modus (keine Chart-Updates)")
        self.turbo_check.setStyleSheet("color: rgb(128, 255, 128);")
        self.turbo_check.toggled.connect(self._toggle_turbo)
        speed_layout.addWidget(self.turbo_check)

        # Debug-Modus
        self.debug_check = QCheckBox("DEBUG-Modus (ausfuehrliches Log)")
        self.debug_check.setStyleSheet("color: rgb(255, 200, 100);")
        self.debug_check.toggled.connect(self._toggle_debug)
        speed_layout.addWidget(self.debug_check)

        # Signal-Invertierung
        self.invert_check = QCheckBox("Signale invertieren (BUY<->SELL)")
        self.invert_check.setStyleSheet("color: rgb(255, 128, 255);")
        self.invert_check.setToolTip("Tauscht BUY und SELL Signale")
        self.invert_check.toggled.connect(self._toggle_invert)
        speed_layout.addWidget(self.invert_check)

        # Aktuelle Geschwindigkeit
        speed_info = QHBoxLayout()
        speed_info.addWidget(QLabel("Aktuell:"))
        self.actual_speed_label = QLabel("- Schritte/Sek")
        self.actual_speed_label.setStyleSheet("color: rgb(77, 230, 255); font-weight: bold;")
        speed_info.addWidget(self.actual_speed_label)
        speed_layout.addLayout(speed_info)

        layout.addWidget(speed_group)

        # === Aktuelle Position ===
        position_group = QGroupBox("Aktuelle Position")
        position_group.setStyleSheet(self._group_style((0.5, 0.9, 0.5)))
        pos_layout = QGridLayout(position_group)
        pos_layout.setColumnStretch(1, 1)

        labels = [
            ("Position:", "position_label", "NONE"),
            ("Einstiegspreis:", "entry_price_label", "-"),
            ("Aktueller Preis:", "current_price_label", "-"),
            ("Unrealisiert:", "unrealized_pnl_label", "-"),
        ]

        for row, (text, attr, default) in enumerate(labels):
            pos_layout.addWidget(QLabel(text), row, 0)
            label = QLabel(default)
            label.setStyleSheet("color: white;")
            setattr(self, attr, label)
            pos_layout.addWidget(label, row, 1)

        layout.addWidget(position_group)

        # === Fortschritt ===
        progress_group = QGroupBox("Fortschritt")
        progress_group.setStyleSheet(self._group_style((0.7, 0.7, 0.7)))
        prog_layout = QGridLayout(progress_group)
        prog_layout.setColumnStretch(1, 1)

        prog_labels = [
            ("Datenpunkt:", "datapoint_label", "0 / 0"),
            ("Datum:", "date_label", "-"),
            ("Letztes Signal:", "signal_label", "-"),
        ]

        for row, (text, attr, default) in enumerate(prog_labels):
            prog_layout.addWidget(QLabel(text), row, 0)
            label = QLabel(default)
            label.setStyleSheet("color: white;" if row < 2 else "color: gray; font-weight: bold;")
            setattr(self, attr, label)
            prog_layout.addWidget(label, row, 1)

        layout.addWidget(progress_group)

        # === Trade-Log ===
        tradelog_group = QGroupBox("Trade-Log")
        tradelog_group.setStyleSheet(self._group_style((0.9, 0.5, 0.9)))
        tradelog_layout = QVBoxLayout(tradelog_group)

        self.tradelog_text = QTextEdit()
        self.tradelog_text.setReadOnly(True)
        self.tradelog_text.setStyleSheet("""
            QTextEdit {
                background-color: rgb(38, 38, 38);
                color: rgb(204, 204, 204);
                font-family: 'Consolas', monospace;
                font-size: 10px;
            }
        """)
        self.tradelog_text.setPlainText("Kein Trade")
        self.tradelog_text.setMaximumHeight(150)
        tradelog_layout.addWidget(self.tradelog_text)

        layout.addWidget(tradelog_group)

        layout.addStretch()

        # Schliessen Button
        close_btn = QPushButton("Schliessen")
        close_btn.setStyleSheet(self._button_style((0.4, 0.4, 0.4)))
        close_btn.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        scroll.setWidget(panel)
        return scroll

    def _create_chart_panel(self) -> QWidget:
        """Erstellt das Chart-Panel (mittlere Spalte) mit Tabs."""
        panel = QWidget()
        panel.setStyleSheet(f"background-color: rgb(46, 46, 46);")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
            from matplotlib.figure import Figure

            # Toolbar und Zoom-Kontrollen oben (immer sichtbar)
            controls_widget = QWidget()
            controls_layout = QVBoxLayout(controls_widget)
            controls_layout.setContentsMargins(0, 0, 0, 0)
            controls_layout.setSpacing(2)

            # Zoom-Kontrollen
            zoom_controls = self._create_zoom_controls()
            controls_layout.addWidget(zoom_controls)

            layout.addWidget(controls_widget)

            # Tab-Widget fuer Charts
            self.chart_tabs = QTabWidget()
            self.chart_tabs.setStyleSheet(self._tab_style())

            # === Tab 1: Preis-Chart ===
            price_widget = QWidget()
            price_layout = QVBoxLayout(price_widget)
            price_layout.setContentsMargins(0, 0, 0, 0)

            self.price_figure = Figure(figsize=(8, 6), facecolor='#262626')
            self.price_canvas = FigureCanvas(self.price_figure)
            self.ax_price = self.price_figure.add_subplot(111)
            self._setup_price_chart()

            # Toolbar fuer Preis-Chart
            self.price_toolbar = NavigationToolbar(self.price_canvas, self)
            self.price_toolbar.setStyleSheet(self._toolbar_style())
            price_layout.addWidget(self.price_toolbar)
            price_layout.addWidget(self.price_canvas)

            self.chart_tabs.addTab(price_widget, "Preis + Signale")

            # === Tab 2: Trade-Chart (pyqtgraph - schneller) ===
            trade_widget = QWidget()
            trade_layout = QVBoxLayout(trade_widget)
            trade_layout.setContentsMargins(0, 0, 0, 0)
            trade_layout.setSpacing(2)

            # pyqtgraph konfigurieren
            pg.setConfigOptions(antialias=True)

            # PlotWidget erstellen
            self.trade_plot = pg.PlotWidget()
            self.trade_plot.setBackground('#262626')
            self.trade_plot.setLabel('left', 'Preis ($)')
            self.trade_plot.setLabel('bottom', 'Zeit')
            self.trade_plot.showGrid(x=True, y=True, alpha=0.3)
            self.trade_plot.getAxis('left').setPen(pg.mkPen(color='#808080'))
            self.trade_plot.getAxis('bottom').setPen(pg.mkPen(color='#808080'))
            self.trade_plot.getAxis('left').setTextPen(pg.mkPen(color='#aaaaaa'))
            self.trade_plot.getAxis('bottom').setTextPen(pg.mkPen(color='#aaaaaa'))

            # Plot-Items erstellen (werden wiederverwendet)
            self.trade_price_line = self.trade_plot.plot([], [], pen=pg.mkPen(color='#8899bb', width=1))
            self.trade_entry_scatter = pg.ScatterPlotItem(size=12, pen=pg.mkPen(color='white', width=1))
            self.trade_exit_scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(color='white', width=1))
            self.trade_plot.addItem(self.trade_entry_scatter)
            self.trade_plot.addItem(self.trade_exit_scatter)

            # Click-Event fuer Scatter
            self.trade_entry_scatter.sigClicked.connect(self._on_scatter_clicked)
            self.trade_exit_scatter.sigClicked.connect(self._on_scatter_clicked)

            # Titel-Label oben
            self.trade_chart_title = QLabel("Trades | 0 Trades | 0W / 0L | Win-Rate: 0.0%")
            self.trade_chart_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.trade_chart_title.setStyleSheet("color: white; font-size: 11px; padding: 3px;")

            trade_layout.addWidget(self.trade_chart_title)
            trade_layout.addWidget(self.trade_plot, stretch=1)

            # Trade-Navigation
            nav_widget = QWidget()
            nav_layout = QHBoxLayout(nav_widget)
            nav_layout.setContentsMargins(5, 2, 5, 2)
            nav_layout.setSpacing(10)

            self.btn_prev_trade = QPushButton("<< Vorheriger")
            self.btn_prev_trade.setStyleSheet(self._nav_button_style())
            self.btn_prev_trade.clicked.connect(self._prev_trade)

            self.trade_nav_label = QLabel("Trade 0/0")
            self.trade_nav_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.trade_nav_label.setStyleSheet("color: #aaa; font-size: 11px; min-width: 80px;")

            self.btn_next_trade = QPushButton("Naechster >>")
            self.btn_next_trade.setStyleSheet(self._nav_button_style())
            self.btn_next_trade.clicked.connect(self._next_trade)

            nav_layout.addStretch()
            nav_layout.addWidget(self.btn_prev_trade)
            nav_layout.addWidget(self.trade_nav_label)
            nav_layout.addWidget(self.btn_next_trade)
            nav_layout.addStretch()

            trade_layout.addWidget(nav_widget)

            # Trade-Detail-Panel unter dem Chart
            self.trade_detail_label = QLabel("Klicke auf einen Trade-Marker im Chart fuer Details")
            self.trade_detail_label.setStyleSheet("""
                QLabel {
                    background-color: #1a1a1a;
                    color: #808080;
                    padding: 8px;
                    border: 1px solid #333;
                    font-family: Consolas, monospace;
                    font-size: 11px;
                }
            """)
            self.trade_detail_label.setMinimumHeight(35)
            trade_layout.addWidget(self.trade_detail_label)

            # Trade-Linien Liste (fuer Verbindungslinien Entry->Exit)
            self.trade_lines = []
            self.trade_labels = []
            self.current_trade_view = -1  # Aktuell angezeigter Trade-Index

            self.chart_tabs.addTab(trade_widget, "Trades")

            # === Tab 3: Equity-Chart ===
            equity_widget = QWidget()
            equity_layout = QVBoxLayout(equity_widget)
            equity_layout.setContentsMargins(0, 0, 0, 0)

            self.equity_figure = Figure(figsize=(8, 6), facecolor='#262626')
            self.equity_canvas = FigureCanvas(self.equity_figure)
            self.ax_equity = self.equity_figure.add_subplot(111)
            self._setup_equity_chart()

            # Toolbar fuer Equity-Chart
            self.equity_toolbar = NavigationToolbar(self.equity_canvas, self)
            self.equity_toolbar.setStyleSheet(self._toolbar_style())
            equity_layout.addWidget(self.equity_toolbar)
            equity_layout.addWidget(self.equity_canvas)

            self.chart_tabs.addTab(equity_widget, "Equity")

            layout.addWidget(self.chart_tabs, stretch=1)

        except ImportError:
            layout.addWidget(QLabel("matplotlib nicht installiert"))

        return panel

    def _tab_style(self) -> str:
        """Gibt das Stylesheet fuer die Tab-Widgets zurueck."""
        return '''
            QTabWidget::pane {
                border: 1px solid #4d4d4d;
                background-color: #262626;
            }
            QTabBar::tab {
                background-color: #333333;
                color: #b3b3b3;
                padding: 8px 20px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #4da8da;
                color: white;
            }
            QTabBar::tab:hover:!selected {
                background-color: #444444;
            }
        '''

    def _toolbar_style(self) -> str:
        """Gibt das Stylesheet fuer die Matplotlib-Toolbar zurueck."""
        return '''
            QToolBar {
                background-color: #333333;
                border: none;
                spacing: 5px;
            }
            QToolButton {
                background-color: #444444;
                border: none;
                border-radius: 3px;
                padding: 5px;
                color: white;
            }
            QToolButton:hover {
                background-color: #555555;
            }
            QToolButton:checked {
                background-color: #4da8da;
            }
        '''

    def _nav_button_style(self) -> str:
        """Gibt das Stylesheet fuer die Trade-Navigations-Buttons zurueck."""
        return '''
            QPushButton {
                background-color: #3a3a3a;
                color: #aaa;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px 15px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                color: white;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
            }
        '''

    def _create_zoom_controls(self) -> QWidget:
        """Erstellt die Zoom-Kontroll-Buttons fuer den Preis-Chart."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # X-Zoom
        layout.addWidget(QLabel('X:'))

        zoom_x_in = QPushButton('+')
        zoom_x_in.setFixedWidth(30)
        zoom_x_in.setStyleSheet(StyleFactory.button_style_hex('#555555', padding='3px 8px'))
        zoom_x_in.clicked.connect(lambda: self._zoom_price_axis('x', 0.8))
        layout.addWidget(zoom_x_in)

        zoom_x_out = QPushButton('-')
        zoom_x_out.setFixedWidth(30)
        zoom_x_out.setStyleSheet(StyleFactory.button_style_hex('#555555', padding='3px 8px'))
        zoom_x_out.clicked.connect(lambda: self._zoom_price_axis('x', 1.25))
        layout.addWidget(zoom_x_out)

        # Y-Zoom
        layout.addWidget(QLabel('Y:'))

        zoom_y_in = QPushButton('+')
        zoom_y_in.setFixedWidth(30)
        zoom_y_in.setStyleSheet(StyleFactory.button_style_hex('#555555', padding='3px 8px'))
        zoom_y_in.clicked.connect(lambda: self._zoom_price_axis('y', 0.8))
        layout.addWidget(zoom_y_in)

        zoom_y_out = QPushButton('-')
        zoom_y_out.setFixedWidth(30)
        zoom_y_out.setStyleSheet(StyleFactory.button_style_hex('#555555', padding='3px 8px'))
        zoom_y_out.clicked.connect(lambda: self._zoom_price_axis('y', 1.25))
        layout.addWidget(zoom_y_out)

        # Reset
        reset_btn = QPushButton('Reset')
        reset_btn.setStyleSheet(StyleFactory.button_style_hex('#666666', padding='3px 10px'))
        reset_btn.clicked.connect(self._reset_price_zoom)
        layout.addWidget(reset_btn)

        # Aktuellen Bereich anzeigen
        follow_btn = QPushButton('Folgen')
        follow_btn.setStyleSheet(StyleFactory.button_style_hex('#4da8da', padding='3px 10px'))
        follow_btn.setToolTip('Zum aktuellen Datenpunkt springen')
        follow_btn.clicked.connect(self._follow_current)
        layout.addWidget(follow_btn)

        layout.addStretch()

        return widget

    def _get_current_chart(self):
        """Gibt den aktuell ausgewaehlten Chart (ax, canvas) zurueck.
        Fuer den Trade-Chart (Tab 1, pyqtgraph) wird (None, None) zurueckgegeben."""
        if not hasattr(self, 'chart_tabs'):
            return self.ax_price, self.price_canvas

        current_tab = self.chart_tabs.currentIndex()
        if current_tab == 0:
            return self.ax_price, self.price_canvas
        elif current_tab == 1:
            # Trade-Chart ist pyqtgraph, hat eigenes Zoom/Pan
            return None, None
        else:
            return self.ax_equity, self.equity_canvas

    def _zoom_price_axis(self, axis: str, factor: float):
        """Zoomt eine einzelne Achse des aktuell sichtbaren Charts."""
        ax, canvas = self._get_current_chart()

        if ax is None:
            # pyqtgraph Chart - nutze Maus-Zoom stattdessen
            return

        if axis == 'x':
            xlim = ax.get_xlim()
            center = (xlim[0] + xlim[1]) / 2
            width = (xlim[1] - xlim[0]) * factor
            ax.set_xlim(center - width/2, center + width/2)
        else:
            ylim = ax.get_ylim()
            center = (ylim[0] + ylim[1]) / 2
            height = (ylim[1] - ylim[0]) * factor
            ax.set_ylim(center - height/2, center + height/2)
        canvas.draw()

    def _reset_price_zoom(self):
        """Setzt den Zoom des aktuell sichtbaren Charts zurueck."""
        ax, canvas = self._get_current_chart()

        if ax is None:
            # pyqtgraph Chart - Auto-Range aktivieren
            if hasattr(self, 'trade_plot'):
                self.trade_plot.autoRange()
            return

        ax.autoscale()
        canvas.draw()

    def _follow_current(self):
        """Springt zum aktuellen Datenpunkt im Chart."""
        if self.data is None or self.current_index == 0:
            return

        ax, canvas = self._get_current_chart()

        # Fenster um aktuellen Index (200 Punkte sichtbar)
        window_size = 200
        start_idx = max(0, self.current_index - window_size // 2)
        end_idx = min(len(self.data), self.current_index + window_size // 2)

        if ax is None:
            # pyqtgraph Chart
            if hasattr(self, 'trade_plot'):
                self.trade_plot.setXRange(start_idx, end_idx, padding=0.02)
            return

        ax.set_xlim(start_idx, end_idx)

        # Y-Limits an sichtbaren Bereich anpassen
        visible_data = self.data.iloc[start_idx:end_idx]
        if len(visible_data) > 0:
            y_min = visible_data['Close'].min()
            y_max = visible_data['Close'].max()
            y_margin = (y_max - y_min) * 0.05
            ax.set_ylim(y_min - y_margin, y_max + y_margin)

        canvas.draw()

        self.price_canvas.draw()

    def _setup_price_chart(self):
        """Konfiguriert den Preis-Chart."""
        ax = self.ax_price
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#4d4d4d')
        ax.set_title('Preis und Signale', color='white', fontsize=12)
        ax.set_ylabel('Preis (USD)', color='white')
        ax.grid(True, alpha=0.3, color='#4d4d4d')

    def _setup_trade_chart(self):
        """Konfiguriert den Trade-Chart."""
        ax = self.ax_trade
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#4d4d4d')
        ax.set_title('Trades', color='white', fontsize=12)
        ax.set_ylabel('Preis (USD)', color='white')
        ax.grid(True, alpha=0.3, color='#4d4d4d')

    def _setup_equity_chart(self):
        """Konfiguriert den Equity-Chart."""
        ax = self.ax_equity
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#4d4d4d')
        ax.set_title('Equity-Kurve', color='white', fontsize=12)
        ax.set_ylabel('Equity (USD)', color='white')
        ax.grid(True, alpha=0.3, color='#4d4d4d')

    def _create_stats_panel(self) -> QWidget:
        """Erstellt das Statistik-Panel (rechte Spalte)."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        panel = QWidget()
        panel.setStyleSheet(f"background-color: rgb(46, 46, 46);")
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # Titel
        title = QLabel("Performance")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: white;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # === Gewinn/Verlust ===
        pnl_group = QGroupBox("Gewinn / Verlust")
        pnl_group.setStyleSheet(self._group_style((0.3, 0.9, 0.3)))
        pnl_layout = QGridLayout(pnl_group)
        pnl_layout.setColumnStretch(1, 1)

        pnl_labels = [
            ("Startkapital:", "start_capital_label", f"${self.initial_capital:,.2f}"),
            ("Aktuell:", "equity_label", f"${self.current_equity:,.2f}"),
            ("Gesamt P/L:", "total_pnl_label", "$0.00"),
            ("P/L %:", "pnl_percent_label", "0.00%"),
            ("Max Drawdown:", "drawdown_label", "0.00%"),
        ]

        for row, (text, attr, default) in enumerate(pnl_labels):
            lbl = QLabel(text)
            lbl.setStyleSheet("color: rgb(179, 179, 179);")
            pnl_layout.addWidget(lbl, row, 0)
            value_lbl = QLabel(default)
            if row == 1:  # Aktuell
                value_lbl.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
            elif row in [2, 3]:  # P/L
                value_lbl.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
                value_lbl.setStyleSheet("color: gray;")
            else:
                value_lbl.setStyleSheet("color: white;")
            setattr(self, attr, value_lbl)
            pnl_layout.addWidget(value_lbl, row, 1)

        layout.addWidget(pnl_group)

        # === Trade-Statistik ===
        stats_group = QGroupBox("Trade-Statistik")
        stats_group.setStyleSheet(self._group_style((0.9, 0.7, 0.3)))
        stats_layout = QGridLayout(stats_group)
        stats_layout.setColumnStretch(1, 1)

        trade_labels = [
            ("Anzahl Trades:", "num_trades_label", "0", "white"),
            ("Gewinner:", "winners_label", "0", "rgb(77, 230, 77)"),
            ("Verlierer:", "losers_label", "0", "rgb(230, 77, 77)"),
            ("Win-Rate:", "winrate_label", "0.00%", "white"),
            ("Avg. Gewinn:", "avg_win_label", "$0.00", "rgb(77, 230, 77)"),
            ("Avg. Verlust:", "avg_loss_label", "$0.00", "rgb(230, 77, 77)"),
        ]

        for row, (text, attr, default, color) in enumerate(trade_labels):
            lbl = QLabel(text)
            lbl.setStyleSheet("color: rgb(179, 179, 179);")
            stats_layout.addWidget(lbl, row, 0)
            value_lbl = QLabel(default)
            value_lbl.setStyleSheet(f"color: {color};")
            setattr(self, attr, value_lbl)
            stats_layout.addWidget(value_lbl, row, 1)

        layout.addWidget(stats_group)

        # === Signal-Verteilung ===
        signal_group = QGroupBox("Signal-Verteilung")
        signal_group.setStyleSheet(self._group_style((0.7, 0.7, 0.9)))
        signal_layout = QGridLayout(signal_group)
        signal_layout.setColumnStretch(1, 1)

        signal_labels = [
            ("BUY:", "buy_count_label", "0", "rgb(77, 230, 77)"),
            ("SELL:", "sell_count_label", "0", "rgb(230, 77, 77)"),
            ("HOLD:", "hold_count_label", "0", "gray"),
        ]

        for row, (text, attr, default, color) in enumerate(signal_labels):
            lbl = QLabel(text)
            lbl.setStyleSheet("color: rgb(179, 179, 179);")
            signal_layout.addWidget(lbl, row, 0)
            value_lbl = QLabel(default)
            value_lbl.setStyleSheet(f"color: {color};")
            setattr(self, attr, value_lbl)
            signal_layout.addWidget(value_lbl, row, 1)

        layout.addWidget(signal_group)

        # === Modell-Info (aufklappbar) ===
        self.model_info_group = QGroupBox("Modell-Info")
        self.model_info_group.setStyleSheet(self._group_style((0.6, 0.8, 1.0)))
        self.model_info_group.setCheckable(True)
        self.model_info_group.setChecked(False)  # Standardmaessig eingeklappt
        model_info_layout = QVBoxLayout(self.model_info_group)

        # TextEdit fuer formatierte Modell-Informationen
        self.model_info_text = QTextEdit()
        self.model_info_text.setReadOnly(True)
        self.model_info_text.setMaximumHeight(250)
        self.model_info_text.setStyleSheet('''
            QTextEdit {
                background-color: #1a1a2e;
                border: 1px solid #333;
                border-radius: 3px;
                color: #ccc;
                font-family: Consolas, monospace;
                font-size: 9px;
            }
        ''')
        self.model_info_text.setPlainText("Kein Modell geladen")
        model_info_layout.addWidget(self.model_info_text)

        layout.addWidget(self.model_info_group)

        layout.addStretch()

        scroll.setWidget(panel)
        return scroll

    def _group_style(self, color: tuple) -> str:
        """Generiert GroupBox-Style mit farbigem Titel."""
        return StyleFactory.group_style(title_color=color)

    def _button_style(self, color: tuple) -> str:
        """Generiert Button-Style aus RGB-Tuple (0-1 Range)."""
        return StyleFactory.button_style(color)

    def set_data(self, data: pd.DataFrame, signals: pd.Series = None,
                 model=None, model_info: Dict = None):
        """Setzt die Backtest-Daten."""
        self._debug(f"set_data() aufgerufen")
        self._debug(f"  data: {type(data).__name__}, {len(data) if data is not None else 'None'} Zeilen")
        self._debug(f"  signals: {type(signals).__name__ if signals is not None else 'None'}")
        self._debug(f"  model: {type(model).__name__ if model is not None else 'None'}")
        self._debug(f"  model_info: {model_info}")

        self.data = data
        self.signals = signals
        self.model = model
        self.model_info = model_info

        if model_info:
            lookback = model_info.get('lookback_size', 60)
            lookforward = model_info.get('lookforward_size', 5)
            self.sequence_length = lookback + lookforward
            self.current_index = self.sequence_length + 1
            self._debug(f"  lookback={lookback}, lookforward={lookforward}, sequence_length={self.sequence_length}")

            # Sequenzen fuer Modell-Vorhersage vorbereiten
            if model is not None and data is not None:
                self._debug(f"  Bereite Sequenzen vor...")
                self._prepare_sequences(lookback)
            else:
                self._debug(f"  WARNUNG: Keine Sequenz-Vorbereitung (model={model is not None}, data={data is not None})")

            # Model-Info Anzeige aktualisieren
            self._update_model_info_display()
        else:
            self._debug(f"  WARNUNG: Keine model_info vorhanden!")

        self._update_datapoint_label()
        self._initialize_charts()
        self._debug(f"set_data() abgeschlossen")

    def _prepare_sequences(self, lookback: int):
        """Bereitet Sequenzen fuer Modell-Vorhersagen vor."""
        self._debug(f"_prepare_sequences(lookback={lookback}) gestartet")
        try:
            import torch
            from ..data.processor import FeatureProcessor
            from ..training.normalizer import ZScoreNormalizer

            # Features aus model_info holen (gleiche wie beim Training)
            features = self.model_info.get('features', ['Open', 'High', 'Low', 'Close', 'PriceChange', 'PriceChangePct'])
            self._log(f"Features aus Training: {features}")
            self._debug(f"  Features: {features}")

            # Features berechnen
            processor = FeatureProcessor(features=features)
            processed = processor.process(self.data)
            self._debug(f"  Processed Shape: {processed.shape}")
            self._debug(f"  Processed Columns: {list(processed.columns)}")

            # Feature-Matrix (nur trainierte Features)
            feature_cols = [f for f in features if f in processed.columns]
            if not feature_cols:
                # Fallback auf OHLC
                feature_cols = ['Open', 'High', 'Low', 'Close']
                self._log(f"Fallback auf OHLC-Features", 'WARNING')
            self._debug(f"  Verwendete Features: {feature_cols}")

            feature_data = processed[feature_cols].values.astype(np.float32)
            self._debug(f"  Feature-Data Shape: {feature_data.shape}")

            # NaN behandeln
            nan_count = np.isnan(feature_data).sum()
            self._debug(f"  NaN-Werte vor Behandlung: {nan_count}")
            feature_data = np.nan_to_num(feature_data, nan=0.0)

            # Normalisierung (wie beim Training)
            normalizer = ZScoreNormalizer()
            feature_data = normalizer.fit_transform(feature_data)
            self._debug(f"  Normalisierte Daten - Min: {feature_data.min():.4f}, Max: {feature_data.max():.4f}")

            # Sequenzen erstellen
            sequences = []
            for i in range(lookback, len(feature_data)):
                seq = feature_data[i - lookback:i]
                sequences.append(seq)

            if sequences:
                self.prepared_sequences = np.array(sequences)
                self.sequence_offset = lookback
                self._log(f"Sequenzen vorbereitet: {len(sequences)} ({len(feature_cols)} Features)")
                self._debug(f"  Sequenzen Shape: {self.prepared_sequences.shape}")
                self._debug(f"  Sequence Offset: {self.sequence_offset}")
            else:
                self._log("Keine Sequenzen generiert", 'WARNING')
                self._debug(f"  FEHLER: Leere Sequenzliste!")

        except Exception as e:
            import traceback
            self._log(f"Sequenz-Vorbereitung fehlgeschlagen: {e}", 'ERROR')
            self._log(traceback.format_exc(), 'ERROR')
            self._debug(f"  EXCEPTION: {e}")

    def _update_model_info_display(self):
        """Aktualisiert die Modell-Info Anzeige im Stats-Panel."""
        if not self.model_info:
            self.model_info_text.setPlainText("Kein Modell geladen")
            return

        info = self.model_info
        lines = []

        # Modell-Identifikation
        lines.append("=== MODELL ===")
        lines.append(f"Typ:        {info.get('model_type', '-')}")
        lines.append(f"Trainiert:  {info.get('trained_at', '-')}")
        if info.get('training_duration_sec'):
            mins = info['training_duration_sec'] / 60
            lines.append(f"Dauer:      {mins:.1f} min")

        # Architektur
        lines.append("")
        lines.append("=== ARCHITEKTUR ===")
        lines.append(f"Hidden:     {info.get('hidden_size', '-')}")
        lines.append(f"Layers:     {info.get('num_layers', '-')}")
        lines.append(f"Dropout:    {info.get('dropout', '-')}")
        lines.append(f"Klassen:    {info.get('num_classes', '-')}")

        # Daten-Parameter
        lines.append("")
        lines.append("=== DATEN ===")
        lines.append(f"Lookback:   {info.get('lookback_size', '-')}")
        lines.append(f"Lookfwd:    {info.get('lookforward_size', '-')}")
        lines.append(f"Lookahead:  {info.get('lookahead_bars', '-')}")
        lines.append(f"Split:      {info.get('train_test_split', '-')}%")

        # Samples
        if info.get('total_samples'):
            lines.append(f"Samples:    {info.get('total_samples', 0):,}")
            lines.append(f"  Train:    {info.get('train_samples', 0):,}")
            lines.append(f"  Val:      {info.get('val_samples', 0):,}")

        # Features
        lines.append("")
        lines.append("=== FEATURES ===")
        features = info.get('features', [])
        lines.append(f"Anzahl:     {len(features)}")
        if features:
            # Features in Kurzform
            feat_str = ', '.join(features[:5])
            if len(features) > 5:
                feat_str += f", +{len(features)-5}"
            lines.append(f"Liste:      {feat_str}")

        # Training
        lines.append("")
        lines.append("=== TRAINING ===")
        lines.append(f"Epochen:    {info.get('epochs_trained', '-')}/{info.get('epochs_configured', '-')}")
        lines.append(f"Batch:      {info.get('batch_size', '-')}")
        lines.append(f"LR:         {info.get('learning_rate', '-')}")
        lines.append(f"Patience:   {info.get('patience', '-')}")
        lines.append(f"Early Stop: {'Ja' if info.get('stopped_early') else 'Nein'}")

        # Ergebnisse
        lines.append("")
        lines.append("=== ERGEBNISSE ===")
        lines.append(f"Accuracy:   {info.get('best_accuracy', 0):.1f}%")
        lines.append(f"Val Loss:   {info.get('final_val_loss', 0):.4f}")
        lines.append(f"BUY Peaks:  {info.get('num_buy_peaks', '-')}")
        lines.append(f"SELL Peaks: {info.get('num_sell_peaks', '-')}")

        # Class Weights
        if info.get('class_weights'):
            cw = info['class_weights']
            lines.append("")
            lines.append("=== CLASS WEIGHTS ===")
            labels = ['HOLD', 'BUY', 'SELL']
            for i, w in enumerate(cw[:3]):
                lines.append(f"{labels[i]:6}:     {w:.2f}")

        self.model_info_text.setPlainText('\n'.join(lines))

    def _start_backtest(self):
        """Startet den Backtest."""
        self._debug(f"_start_backtest() aufgerufen")

        if self.data is None:
            self._debug(f"  ABBRUCH: data ist None")
            return

        if self.is_running:
            self._debug(f"  ABBRUCH: bereits running")
            return

        self._debug(f"  Start-Index: {self.current_index}")
        self._debug(f"  Daten: {len(self.data)} Zeilen")
        self._debug(f"  Modell: {type(self.model).__name__ if self.model else 'None'}")
        self._debug(f"  Prepared Sequences: {self.prepared_sequences.shape if self.prepared_sequences is not None else 'None'}")

        self.is_running = True
        self.is_paused = False

        # Buttons aktualisieren
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.step_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)

        # Timer starten
        interval = int(1000 / self.steps_per_second)
        self.backtest_timer = QTimer()
        self.backtest_timer.timeout.connect(self._timer_callback)
        self.backtest_timer.start(interval)
        self._debug(f"  Timer gestartet: {interval}ms Intervall ({self.steps_per_second} steps/s)")

        # Geschwindigkeitsmessung starten
        self._step_count = 0
        self._last_speed_update = time.perf_counter()
        self._speed_update_timer = QTimer()
        self._speed_update_timer.timeout.connect(self._update_actual_speed)
        self._speed_update_timer.start(500)  # Alle 500ms aktualisieren

        # Geschwindigkeit anzeigen (Ziel)
        self.actual_speed_label.setText(f"0 / {self.steps_per_second} Schritte/Sek")

        self._add_tradelog("Backtest gestartet")

    def _stop_backtest(self):
        """Stoppt den Backtest."""
        self.is_running = False
        self.is_paused = True

        if self.backtest_timer:
            self.backtest_timer.stop()
            self.backtest_timer = None

        # Geschwindigkeitsmess-Timer stoppen
        if self._speed_update_timer:
            self._speed_update_timer.stop()
            self._speed_update_timer = None

        # Buttons aktualisieren
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.step_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)

        self.actual_speed_label.setText("Gestoppt")
        self._add_tradelog("Backtest gestoppt")

        # Charts final aktualisieren
        self._update_charts()

    def _reset_backtest(self):
        """Setzt den Backtest zurueck."""
        # Timer stoppen
        if self.backtest_timer:
            self.backtest_timer.stop()
            self.backtest_timer = None

        # Geschwindigkeitsmess-Timer stoppen
        if self._speed_update_timer:
            self._speed_update_timer.stop()
            self._speed_update_timer = None

        # Status zuruecksetzen
        self.is_running = False
        self.is_paused = False
        self.current_index = self.sequence_length + 1 if self.sequence_length > 0 else 0

        self.position = 'NONE'
        self.entry_price = 0.0
        self.entry_index = 0
        self.total_pnl = 0.0
        self.trades = []
        self.signal_history = []

        self.current_equity = self.initial_capital
        self.equity_curve = [self.initial_capital]
        self.equity_indices = [self.current_index]

        self.buy_count = 0
        self.sell_count = 0
        self.hold_count = 0

        # UI zuruecksetzen
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.step_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)

        self._reset_labels()
        self._initialize_charts()
        self.tradelog_text.setPlainText("Kein Trade")

    def _single_step(self):
        """Fuehrt einen einzelnen Schritt aus."""
        if self.data is None:
            return

        if self.current_index <= len(self.data):
            self._process_step()
            self._update_ui()
            self._update_charts()

    def _timer_callback(self):
        """Timer-Callback fuer automatischen Durchlauf."""
        if not self.is_running:
            return

        if self.data is None or self.current_index > len(self.data):
            self._finalize_backtest()
            return

        self._process_step()
        self._step_count += 1  # Schritt zaehlen fuer Geschwindigkeitsmessung
        self._update_ui()

        # Charts nur im Nicht-Turbo-Modus aktualisieren
        if not self.turbo_mode and self.current_index % 10 == 0:
            self._update_charts()

    def _process_step(self):
        """Verarbeitet einen einzelnen Backtest-Schritt."""
        if self.data is None or self.current_index > len(self.data):
            return

        idx = self.current_index - 1  # 0-basiert

        # Aktueller Preis
        current_price = self.data['Close'].iloc[idx]

        # Signal ermitteln (aus Signalen oder Modell)
        signal = self._get_signal(idx)

        # Signal speichern
        self.signal_history.append({
            'index': self.current_index,
            'signal': signal,
            'price': current_price
        })

        # Signal-Zaehler
        if signal == 1:
            self.buy_count += 1
        elif signal == 2:
            self.sell_count += 1
        else:
            self.hold_count += 1

        # Trading-Logik
        self._process_signal(signal, current_price, self.current_index)

        # Equity aktualisieren
        self._update_equity(current_price)

        # Naechster Schritt
        self.current_index += 1

    def _invert_signal(self, signal: int) -> int:
        """Invertiert ein Signal (BUY<->SELL), HOLD bleibt unveraendert."""
        if signal == 1:  # BUY -> SELL
            return 2
        elif signal == 2:  # SELL -> BUY
            return 1
        return signal  # HOLD bleibt 0

    def _get_signal(self, idx: int) -> int:
        """Ermittelt das Signal fuer den aktuellen Index.

        Rueckgabe-Mapping (intern):
            0 = HOLD (kein Trade)
            1 = BUY (Long oeffnen)
            2 = SELL (Short oeffnen)

        Bei 2-Klassen-Modellen: Modell gibt 0=BUY, 1=SELL zurueck
        Bei 3-Klassen-Modellen: Modell gibt 0=HOLD, 1=BUY, 2=SELL zurueck
        """
        signal = 0  # Default: HOLD

        # Wenn Signale vorhanden, diese verwenden
        if self.signals is not None and idx < len(self.signals):
            sig = self.signals.iloc[idx]
            if sig == 'BUY' or sig == 1:
                signal = 1
            elif sig == 'SELL' or sig == 2:
                signal = 2
            else:
                signal = 0

        # Modell-Vorhersage wenn verfuegbar
        elif self.model is not None and self.prepared_sequences is not None:
            try:
                import torch
                # Pruefe ob Index im Bereich der vorbereiteten Sequenzen liegt
                seq_idx = idx - self.sequence_offset
                if 0 <= seq_idx < len(self.prepared_sequences):
                    sequence = self.prepared_sequences[seq_idx]
                    if not isinstance(sequence, torch.Tensor):
                        sequence = torch.FloatTensor(sequence)
                    sequence = sequence.unsqueeze(0)  # Batch-Dimension hinzufuegen

                    self.model.eval()
                    with torch.no_grad():
                        prediction = self.model.predict(sequence)
                        raw_signal = int(prediction[0])

                        # Signal-Mapping basierend auf num_classes
                        num_classes = 3
                        if self.model_info:
                            num_classes = self.model_info.get('num_classes', 3)

                        if num_classes == 2:
                            # 2-Klassen: 0=BUY, 1=SELL -> intern 1=BUY, 2=SELL
                            signal = raw_signal + 1  # 0->1 (BUY), 1->2 (SELL)
                        else:
                            # 3-Klassen: 0=HOLD, 1=BUY, 2=SELL (passt bereits)
                            signal = raw_signal
                else:
                    if self.debug_mode and idx % 100 == 0:
                        self._debug(f"Idx {idx}: seq_idx={seq_idx} ausserhalb Bereich [0, {len(self.prepared_sequences)})")
            except Exception as e:
                if self.debug_mode:
                    self._debug(f"Idx {idx}: Modell-Fehler: {e}")

        # Signal-Invertierung wenn aktiviert
        if self.invert_signals and signal != 0:
            signal = self._invert_signal(signal)

        # Debug-Logging
        if self.debug_mode and idx % 100 == 0:
            signal_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            invert_str = " (invertiert)" if self.invert_signals else ""
            self._debug(f"Idx {idx}: signal={signal_names.get(signal, signal)}{invert_str}")

        return signal

    def _process_signal(self, signal: int, price: float, idx: int):
        """Verarbeitet ein Trading-Signal."""
        # signal: 0=HOLD, 1=BUY, 2=SELL

        if signal == 1:  # BUY
            if self.position == 'SHORT':
                self._close_trade(price, idx, 'BUY Signal')
                self._open_trade('LONG', price, idx)
            elif self.position == 'NONE':
                self._open_trade('LONG', price, idx)

        elif signal == 2:  # SELL
            if self.position == 'LONG':
                self._close_trade(price, idx, 'SELL Signal')
                self._open_trade('SHORT', price, idx)
            elif self.position == 'NONE':
                self._open_trade('SHORT', price, idx)

    def _open_trade(self, new_position: str, price: float, idx: int):
        """Oeffnet eine neue Position."""
        self.position = new_position
        self.entry_price = price
        self.entry_index = idx

        if not self.turbo_mode:
            self._add_tradelog(f"{new_position} @ ${price:,.2f}")

    def _close_trade(self, price: float, idx: int, reason: str):
        """Schliesst die aktuelle Position."""
        if self.position == 'NONE':
            return

        # P/L berechnen
        if self.position == 'LONG':
            pnl = price - self.entry_price
        else:  # SHORT
            pnl = self.entry_price - price

        # Trade speichern
        trade = {
            'entry_index': self.entry_index,
            'exit_index': idx,
            'position': self.position,
            'entry_price': self.entry_price,
            'exit_price': price,
            'pnl': pnl,
            'reason': reason
        }
        self.trades.append(trade)

        # P/L aktualisieren
        self.total_pnl += pnl
        self.current_equity = self.initial_capital + self.total_pnl

        # Trade-Log
        pnl_str = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"
        self._add_tradelog(f"#{len(self.trades)} {self.position}: {self.entry_price:,.2f} -> {price:,.2f} ({pnl_str})")

        self.position = 'NONE'
        self.entry_price = 0.0
        self.entry_index = 0

    def _update_equity(self, current_price: float):
        """Aktualisiert die Equity-Kurve."""
        unrealized = 0.0
        if self.position == 'LONG':
            unrealized = current_price - self.entry_price
        elif self.position == 'SHORT':
            unrealized = self.entry_price - current_price

        self.equity_curve.append(self.current_equity + unrealized)
        self.equity_indices.append(self.current_index)

    def _update_ui(self):
        """Aktualisiert die UI-Elemente."""
        self._update_datapoint_label()
        self._update_position_labels()
        self._update_pnl_labels()
        self._update_trade_stats()
        self._update_signal_counts()

    def _update_datapoint_label(self):
        """Aktualisiert die Fortschritts-Anzeige."""
        total = len(self.data) if self.data is not None else 0
        self.datapoint_label.setText(f"{self.current_index} / {total}")

        if self.data is not None and self.current_index <= len(self.data):
            idx = self.current_index - 1
            if 'DateTime' in self.data.columns or self.data.index.name == 'DateTime':
                dt = self.data.index[idx] if self.data.index.name else self.data['DateTime'].iloc[idx]
                if hasattr(dt, 'strftime'):
                    self.date_label.setText(dt.strftime('%d.%m.%Y %H:%M'))

            self.current_price_label.setText(f"${self.data['Close'].iloc[idx]:,.2f}")

        # Letztes Signal
        if self.signal_history:
            last_sig = self.signal_history[-1]['signal']
            if last_sig == 1:
                self.signal_label.setText("BUY")
                self.signal_label.setStyleSheet("color: rgb(77, 230, 77); font-weight: bold;")
            elif last_sig == 2:
                self.signal_label.setText("SELL")
                self.signal_label.setStyleSheet("color: rgb(230, 77, 77); font-weight: bold;")
            else:
                self.signal_label.setText("HOLD")
                self.signal_label.setStyleSheet("color: gray; font-weight: bold;")

    def _update_position_labels(self):
        """Aktualisiert die Positions-Anzeige."""
        self.position_label.setText(self.position)
        if self.position == 'LONG':
            self.position_label.setStyleSheet("color: rgb(77, 230, 77); font-weight: bold;")
        elif self.position == 'SHORT':
            self.position_label.setStyleSheet("color: rgb(230, 77, 77); font-weight: bold;")
        else:
            self.position_label.setStyleSheet("color: gray;")

        if self.entry_price > 0:
            self.entry_price_label.setText(f"${self.entry_price:,.2f}")
        else:
            self.entry_price_label.setText("-")

        # Unrealisierter P/L
        if self.position != 'NONE' and self.data is not None and self.current_index <= len(self.data):
            current_price = self.data['Close'].iloc[self.current_index - 1]
            if self.position == 'LONG':
                unrealized = current_price - self.entry_price
            else:
                unrealized = self.entry_price - current_price

            self.unrealized_pnl_label.setText(f"${unrealized:,.2f}")
            if unrealized >= 0:
                self.unrealized_pnl_label.setStyleSheet("color: rgb(77, 230, 77);")
            else:
                self.unrealized_pnl_label.setStyleSheet("color: rgb(230, 77, 77);")
        else:
            self.unrealized_pnl_label.setText("-")
            self.unrealized_pnl_label.setStyleSheet("color: white;")

    def _update_pnl_labels(self):
        """Aktualisiert die P/L-Anzeige."""
        self.equity_label.setText(f"${self.current_equity:,.2f}")
        self.total_pnl_label.setText(f"${self.total_pnl:,.2f}")

        pnl_pct = ((self.current_equity - self.initial_capital) / self.initial_capital) * 100
        self.pnl_percent_label.setText(f"{pnl_pct:.2f}%")

        # Farbe
        color = "rgb(77, 230, 77)" if self.total_pnl >= 0 else "rgb(230, 77, 77)"
        self.total_pnl_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.pnl_percent_label.setStyleSheet(f"color: {color}; font-weight: bold;")

        # Drawdown
        if len(self.equity_curve) > 1:
            peak = max(self.equity_curve)
            current = self.equity_curve[-1]
            drawdown = ((peak - current) / peak) * 100 if peak > 0 else 0
            self.drawdown_label.setText(f"{drawdown:.2f}%")

    def _update_trade_stats(self):
        """Aktualisiert die Trade-Statistiken."""
        num_trades = len(self.trades)
        self.num_trades_label.setText(str(num_trades))

        if num_trades > 0:
            pnls = [t['pnl'] for t in self.trades]
            winners = sum(1 for p in pnls if p > 0)
            losers = sum(1 for p in pnls if p <= 0)

            self.winners_label.setText(str(winners))
            self.losers_label.setText(str(losers))
            self.winrate_label.setText(f"{(winners / num_trades) * 100:.2f}%")

            winning_pnls = [p for p in pnls if p > 0]
            losing_pnls = [p for p in pnls if p <= 0]

            if winning_pnls:
                self.avg_win_label.setText(f"${np.mean(winning_pnls):,.2f}")
            if losing_pnls:
                self.avg_loss_label.setText(f"${np.mean(losing_pnls):,.2f}")

    def _update_signal_counts(self):
        """Aktualisiert die Signal-Zaehler."""
        self.buy_count_label.setText(str(self.buy_count))
        self.sell_count_label.setText(str(self.sell_count))
        self.hold_count_label.setText(str(self.hold_count))

    def _reset_labels(self):
        """Setzt alle Labels zurueck."""
        self.position_label.setText("NONE")
        self.position_label.setStyleSheet("color: gray;")
        self.entry_price_label.setText("-")
        self.current_price_label.setText("-")
        self.unrealized_pnl_label.setText("-")
        self.signal_label.setText("-")
        self.signal_label.setStyleSheet("color: gray;")

        self.equity_label.setText(f"${self.initial_capital:,.2f}")
        self.total_pnl_label.setText("$0.00")
        self.total_pnl_label.setStyleSheet("color: gray;")
        self.pnl_percent_label.setText("0.00%")
        self.pnl_percent_label.setStyleSheet("color: gray;")
        self.drawdown_label.setText("0.00%")

        self.num_trades_label.setText("0")
        self.winners_label.setText("0")
        self.losers_label.setText("0")
        self.winrate_label.setText("0.00%")
        self.avg_win_label.setText("$0.00")
        self.avg_loss_label.setText("$0.00")

        self.buy_count_label.setText("0")
        self.sell_count_label.setText("0")
        self.hold_count_label.setText("0")

        self.actual_speed_label.setText("- Schritte/Sek")

    def _initialize_charts(self):
        """Initialisiert die Charts."""
        if not hasattr(self, 'ax_price'):
            return

        # Preis-Chart
        self.ax_price.clear()
        self._setup_price_chart()

        if self.data is not None:
            self.ax_price.plot(self.data.index, self.data['Close'],
                              color='#6699e6', linewidth=1)

        self.price_figure.tight_layout()
        self.price_canvas.draw()

        # Equity-Chart
        self.ax_equity.clear()
        self._setup_equity_chart()

        self.ax_equity.axhline(y=self.initial_capital, color='gray',
                               linestyle='--', linewidth=1)

        self.equity_figure.tight_layout()
        self.equity_canvas.draw()

    def _update_charts(self):
        """Aktualisiert die Charts unter Beibehaltung der Zoom-Einstellungen."""
        if not hasattr(self, 'ax_price'):
            return

        # Aktuelle Zoom-Limits speichern
        price_xlim = self.ax_price.get_xlim()
        price_ylim = self.ax_price.get_ylim()
        equity_xlim = self.ax_equity.get_xlim() if hasattr(self, 'ax_equity') else None
        equity_ylim = self.ax_equity.get_ylim() if hasattr(self, 'ax_equity') else None

        # Preis-Chart
        self.ax_price.clear()
        self._setup_price_chart()

        if self.data is not None:
            self.ax_price.plot(self.data.index, self.data['Close'],
                              color='#6699e6', linewidth=1)

            # BUY Signale
            buy_indices = [s['index'] - 1 for s in self.signal_history if s['signal'] == 1]
            if buy_indices:
                buy_prices = [self.data['Close'].iloc[i] for i in buy_indices if i < len(self.data)]
                buy_dates = [self.data.index[i] for i in buy_indices if i < len(self.data)]
                self.ax_price.scatter(buy_dates, buy_prices, marker='^', c='lime', s=50, zorder=5)

            # SELL Signale
            sell_indices = [s['index'] - 1 for s in self.signal_history if s['signal'] == 2]
            if sell_indices:
                sell_prices = [self.data['Close'].iloc[i] for i in sell_indices if i < len(self.data)]
                sell_dates = [self.data.index[i] for i in sell_indices if i < len(self.data)]
                self.ax_price.scatter(sell_dates, sell_prices, marker='v', c='red', s=50, zorder=5)

            # Aktuelle Position markieren
            if self.current_index <= len(self.data):
                idx = self.current_index - 1
                self.ax_price.axvline(x=self.data.index[idx], color='yellow',
                                      linestyle='--', linewidth=1, alpha=0.7)

        self.ax_price.set_title(f'Preis und Signale | BUY: {self.buy_count}, SELL: {self.sell_count}, HOLD: {self.hold_count}',
                                color='white', fontsize=11)

        # Zoom-Limits wiederherstellen (wenn gueltig)
        if price_xlim[0] != 0.0 or price_xlim[1] != 1.0:
            self.ax_price.set_xlim(price_xlim)
            self.ax_price.set_ylim(price_ylim)

        self.price_figure.tight_layout()
        self.price_canvas.draw()

        # Trade-Chart
        self._update_trade_chart()

        # Equity-Chart
        self.ax_equity.clear()
        self._setup_equity_chart()

        if len(self.equity_indices) > 1 and self.data is not None:
            valid_indices = [i - 1 for i in self.equity_indices if i - 1 < len(self.data) and i > 0]
            if valid_indices:
                dates = [self.data.index[i] for i in valid_indices]
                equity_vals = self.equity_curve[:len(valid_indices)]
                self.ax_equity.plot(dates, equity_vals, color='lime', linewidth=1.5)

        self.ax_equity.axhline(y=self.initial_capital, color='gray',
                               linestyle='--', linewidth=1)

        pnl_pct = ((self.current_equity - self.initial_capital) / self.initial_capital) * 100
        self.ax_equity.set_title(f'Equity-Kurve | P/L: ${self.total_pnl:,.2f} ({pnl_pct:.2f}%)',
                                 color='white', fontsize=11)

        # Zoom-Limits wiederherstellen (wenn gueltig)
        if equity_xlim is not None and (equity_xlim[0] != 0.0 or equity_xlim[1] != 1.0):
            self.ax_equity.set_xlim(equity_xlim)
            self.ax_equity.set_ylim(equity_ylim)

        self.equity_figure.tight_layout()
        self.equity_canvas.draw()

    def _update_trade_chart(self):
        """Aktualisiert den Trade-Chart mit Entry/Exit und P/L (pyqtgraph)."""
        if not hasattr(self, 'trade_plot'):
            return

        if self.data is None:
            return

        # Alte Linien und Labels entfernen
        for line in self.trade_lines:
            self.trade_plot.removeItem(line)
        self.trade_lines.clear()
        for label in self.trade_labels:
            self.trade_plot.removeItem(label)
        self.trade_labels.clear()

        # X-Achse: Indizes verwenden (schneller als Timestamps)
        x_data = np.arange(len(self.data))
        y_data = self.data['Close'].values

        # Preis-Linie aktualisieren
        self.trade_price_line.setData(x_data, y_data)

        # Trade-Marker sammeln
        entry_spots = []
        exit_spots = []
        wins = 0
        losses = 0

        for trade_index, trade in enumerate(self.trades):
            entry_idx = trade.get('entry_index', 0) - 1
            exit_idx = trade.get('exit_index', 0) - 1
            pnl = trade.get('pnl', 0)

            if entry_idx < 0 or exit_idx < 0:
                continue
            if entry_idx >= len(self.data) or exit_idx >= len(self.data):
                continue

            entry_price = trade.get('entry_price', 0)
            exit_price = trade.get('exit_price', 0)
            trade_type = trade.get('position', 'LONG')

            # Farbe basierend auf P/L
            if pnl > 0:
                line_color = '#33cc33'
                wins += 1
            else:
                line_color = '#cc3333'
                losses += 1

            # Verbindungslinie zwischen Entry und Exit
            line = self.trade_plot.plot(
                [entry_idx, exit_idx], [entry_price, exit_price],
                pen=pg.mkPen(color=line_color, width=2)
            )
            self.trade_lines.append(line)

            # Entry-Marker
            entry_color = '#33cc33' if trade_type == 'LONG' else '#cc3333'
            entry_symbol = 't' if trade_type == 'LONG' else 't1'  # Dreieck hoch/runter
            entry_spots.append({
                'pos': (entry_idx, entry_price),
                'brush': pg.mkBrush(entry_color),
                'symbol': entry_symbol,
                'size': 12,
                'data': trade_index
            })

            # Exit-Marker
            exit_spots.append({
                'pos': (exit_idx, exit_price),
                'brush': pg.mkBrush(line_color),
                'symbol': 'x',
                'size': 10,
                'data': trade_index
            })

            # P/L Label
            pnl_text = f'+${pnl:.0f}' if pnl > 0 else f'-${abs(pnl):.0f}'
            label = pg.TextItem(text=pnl_text, color=line_color, anchor=(0, 0.5))
            label.setPos(exit_idx + 1, exit_price)
            self.trade_plot.addItem(label)
            self.trade_labels.append(label)

        # Aktuelle offene Position
        if self.position != 'NONE' and self.entry_index > 0:
            entry_idx = self.entry_index - 1
            if entry_idx < len(self.data):
                current_idx = min(self.current_index - 1, len(self.data) - 1)
                current_price = self.data['Close'].iloc[current_idx]

                # Gestrichelte Linie fuer offene Position
                line = self.trade_plot.plot(
                    [entry_idx, current_idx], [self.entry_price, current_price],
                    pen=pg.mkPen(color='#e6b333', width=2, style=Qt.PenStyle.DashLine)
                )
                self.trade_lines.append(line)

                # Entry-Marker fuer offene Position
                open_color = '#33cc33' if self.position == 'LONG' else '#cc3333'
                open_symbol = 't' if self.position == 'LONG' else 't1'
                entry_spots.append({
                    'pos': (entry_idx, self.entry_price),
                    'brush': pg.mkBrush(open_color),
                    'symbol': open_symbol,
                    'size': 12,
                    'data': -1  # Offene Position, kein Trade-Index
                })

        # Scatter-Daten setzen
        self.trade_entry_scatter.setData(entry_spots)
        self.trade_exit_scatter.setData(exit_spots)

        # Titel aktualisieren
        total_trades = len(self.trades)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        self.trade_chart_title.setText(
            f"Trades | {total_trades} Trades | {wins}W / {losses}L | Win-Rate: {win_rate:.1f}%"
        )

        # Navigation Label aktualisieren
        if total_trades > 0:
            current = self.current_trade_view + 1 if self.current_trade_view >= 0 else 0
            self.trade_nav_label.setText(f"Trade {current}/{total_trades}")
        else:
            self.trade_nav_label.setText("Trade 0/0")

        # Standard-Zoom: Auf letzten Trade oder letzte 24h
        if self.trades and self.current_trade_view < 0:
            # Zeige letzten Trade mit 24h Kontext
            self._goto_trade(len(self.trades) - 1)
        elif not self.trades and len(self.data) > 0:
            # Keine Trades: zeige letzte 24h
            end_idx = len(self.data) - 1
            start_idx = max(0, end_idx - 24)
            self.trade_plot.setXRange(start_idx, end_idx, padding=0.02)

    def _on_scatter_clicked(self, scatter, points):
        """pyqtgraph Scatter-Click Handler."""
        if points:
            point = points[0]
            trade_idx = point.data()
            if trade_idx is not None and 0 <= trade_idx < len(self.trades):
                self.current_trade_view = trade_idx
                self._update_trade_detail(self.trades[trade_idx], trade_idx)
                self._update_nav_label()

    def _goto_trade(self, trade_idx: int):
        """Zoomt auf einen bestimmten Trade."""
        if not self.trades or trade_idx < 0 or trade_idx >= len(self.trades):
            return

        self.current_trade_view = trade_idx
        trade = self.trades[trade_idx]

        # Trade-Bereich ermitteln
        entry_idx = trade.get('entry_index', 0) - 1
        exit_idx = trade.get('exit_index', 0) - 1

        if entry_idx < 0 or exit_idx < 0:
            return

        # +/- 12 Stunden Kontext (24h gesamt)
        start = max(0, entry_idx - 12)
        end = min(len(self.data) - 1, exit_idx + 12)

        self.trade_plot.setXRange(start, end, padding=0.02)

        # Y-Range automatisch anpassen
        if start < len(self.data) and end < len(self.data):
            visible_data = self.data.iloc[start:end + 1]
            y_min = visible_data['Close'].min() * 0.998
            y_max = visible_data['Close'].max() * 1.002
            self.trade_plot.setYRange(y_min, y_max, padding=0)

        self._update_trade_detail(trade, trade_idx)
        self._update_nav_label()

    def _prev_trade(self):
        """Zum vorherigen Trade navigieren."""
        if self.trades and self.current_trade_view > 0:
            self._goto_trade(self.current_trade_view - 1)

    def _next_trade(self):
        """Zum naechsten Trade navigieren."""
        if self.trades and self.current_trade_view < len(self.trades) - 1:
            self._goto_trade(self.current_trade_view + 1)

    def _update_nav_label(self):
        """Aktualisiert das Trade-Navigations-Label."""
        if self.trades:
            self.trade_nav_label.setText(f"Trade {self.current_trade_view + 1}/{len(self.trades)}")

    def _update_trade_detail(self, trade: dict, idx: int):
        """Aktualisiert das Trade-Detail-Panel mit den Trade-Daten."""
        entry_idx = trade.get('entry_index', 0) - 1
        exit_idx = trade.get('exit_index', 0) - 1
        pnl = trade.get('pnl', 0)
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        trade_type = trade.get('position', 'LONG')
        reason = trade.get('reason', '-')

        # Zeit-Informationen
        entry_time = "-"
        exit_time = "-"
        duration = "-"
        if self.data is not None and entry_idx >= 0 and exit_idx >= 0:
            if entry_idx < len(self.data):
                entry_dt = self._get_datetime(entry_idx)
                entry_time = entry_dt.strftime('%Y-%m-%d %H:%M') if entry_dt else "-"
            if exit_idx < len(self.data):
                exit_dt = self._get_datetime(exit_idx)
                exit_time = exit_dt.strftime('%Y-%m-%d %H:%M') if exit_dt else "-"
            # Echte Zeitdauer berechnen
            if entry_dt and exit_dt:
                time_diff = exit_dt - entry_dt
                duration = self._format_duration(time_diff)

        # P/L Prozent
        pnl_pct = (pnl / entry_price * 100) if entry_price > 0 else 0

        # Farbe basierend auf P/L
        pnl_color = '#33cc33' if pnl >= 0 else '#cc3333'
        pnl_text = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"

        # Detail-Text formatieren
        text = (f"#{idx + 1} | {trade_type} | "
                f"Entry: {entry_time} @ ${entry_price:,.2f} | "
                f"Exit: {exit_time} @ ${exit_price:,.2f} | "
                f"P/L: {pnl_text} ({pnl_pct:+.2f}%) | "
                f"Dauer: {duration} | Grund: {reason}")

        self.trade_detail_label.setText(text)
        self.trade_detail_label.setStyleSheet(f"""
            QLabel {{
                background-color: #1a1a1a;
                color: {pnl_color};
                padding: 8px;
                border: 1px solid {pnl_color};
                font-family: Consolas, monospace;
                font-size: 11px;
            }}
        """)

    def _finalize_backtest(self):
        """Finalisiert den Backtest."""
        self._debug(f"_finalize_backtest() aufgerufen")
        self._stop_backtest()

        # Offene Position schliessen
        if self.position != 'NONE' and self.data is not None:
            final_price = self.data['Close'].iloc[-1]
            self._close_trade(final_price, len(self.data), 'Backtest Ende')

        self._update_charts()
        self._add_tradelog("=== Backtest abgeschlossen ===")
        self._add_tradelog(f"Gesamt P/L: ${self.total_pnl:,.2f}")
        self._add_tradelog(f"Trades: {len(self.trades)}")

        # DEBUG-Zusammenfassung
        self._debug(f"=== Backtest Zusammenfassung ===")
        self._debug(f"  Datenpunkte verarbeitet: {self.current_index}")
        self._debug(f"  Signale: BUY={self.buy_count}, SELL={self.sell_count}, HOLD={self.hold_count}")
        self._debug(f"  Trades: {len(self.trades)}")
        self._debug(f"  Gesamt P/L: ${self.total_pnl:,.2f}")
        self._debug(f"  End-Equity: ${self.current_equity:,.2f}")
        if self.trades:
            wins = sum(1 for t in self.trades if t.get('pnl', 0) > 0)
            losses = len(self.trades) - wins
            self._debug(f"  Gewinn-Trades: {wins}, Verlust-Trades: {losses}")
        self._debug(f"=== Ende Zusammenfassung ===")

    def _update_speed(self, value: int):
        """Aktualisiert die eingestellte Geschwindigkeit."""
        self.steps_per_second = value
        self.speed_label.setText(str(value))

        # Timer neu starten falls aktiv
        if self.is_running and self.backtest_timer:
            self.backtest_timer.setInterval(int(1000 / value))

    def _update_actual_speed(self):
        """Berechnet und zeigt die tatsaechliche Geschwindigkeit an."""
        now = time.perf_counter()
        elapsed = now - self._last_speed_update

        if elapsed > 0:
            actual_speed = self._step_count / elapsed
            self.actual_speed_label.setText(
                f"{actual_speed:.1f} / {self.steps_per_second} Schritte/Sek"
            )

        # Zaehler zuruecksetzen
        self._step_count = 0
        self._last_speed_update = now

    def _toggle_turbo(self, checked: bool):
        """Schaltet den Turbo-Modus um."""
        self.turbo_mode = checked
        if not checked:
            self._update_charts()

    def _toggle_debug(self, checked: bool):
        """Schaltet den DEBUG-Modus um."""
        self.debug_mode = checked
        if checked:
            self._log("DEBUG-Modus aktiviert", 'INFO')
            self._debug_dump_state()
        else:
            self._log("DEBUG-Modus deaktiviert", 'INFO')

    def _toggle_invert(self, checked: bool):
        """Schaltet die Signal-Invertierung um."""
        self.invert_signals = checked
        if checked:
            self._log("Signal-Invertierung aktiviert (BUY<->SELL)", 'INFO')
        else:
            self._log("Signal-Invertierung deaktiviert", 'INFO')

    def _debug_dump_state(self):
        """Gibt den aktuellen Zustand im DEBUG-Modus aus."""
        self._debug(f"=== Backtester State Dump ===")
        self._debug(f"Daten: {len(self.data) if self.data is not None else 'None'} Zeilen")
        self._debug(f"Modell: {type(self.model).__name__ if self.model else 'None'}")
        self._debug(f"Model-Info: {self.model_info}")
        self._debug(f"Sequenz-Laenge: {self.sequence_length}")
        self._debug(f"Prepared Sequences: {self.prepared_sequences.shape if self.prepared_sequences is not None else 'None'}")
        self._debug(f"Aktueller Index: {self.current_index}")
        self._debug(f"Position: {self.position}")
        self._debug(f"Equity: ${self.current_equity:,.2f}")
        self._debug(f"Total P/L: ${self.total_pnl:,.2f}")
        self._debug(f"Trades: {len(self.trades)}")
        self._debug(f"=== Ende State Dump ===")

    def _add_tradelog(self, message: str):
        """Fuegt eine Nachricht zum Trade-Log hinzu."""
        current = self.tradelog_text.toPlainText()
        if current == "Kein Trade":
            self.tradelog_text.setPlainText(message)
        else:
            lines = current.split('\n')
            lines.append(message)
            # Nur letzte 20 Zeilen behalten
            if len(lines) > 20:
                lines = lines[-20:]
            self.tradelog_text.setPlainText('\n'.join(lines))

        # Scroll nach unten
        scrollbar = self.tradelog_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def closeEvent(self, event):
        """Behandelt das Schliessen des Fensters."""
        if self.backtest_timer:
            self.backtest_timer.stop()
        if self._speed_update_timer:
            self._speed_update_timer.stop()
        event.accept()

    def _get_datetime(self, idx: int):
        """Holt DateTime aus dem DataFrame (Index oder Spalte)."""
        if self.data is None or idx < 0 or idx >= len(self.data):
            return None
        try:
            # Versuche Index (wenn DateTime als Index gesetzt)
            dt = self.data.index[idx]
            if hasattr(dt, 'strftime'):
                return dt
            # Versuche DateTime-Spalte
            if 'DateTime' in self.data.columns:
                dt_val = self.data['DateTime'].iloc[idx]
                if isinstance(dt_val, str):
                    return pd.to_datetime(dt_val)
                if hasattr(dt_val, 'strftime'):
                    return dt_val
        except:
            pass
        return None

    def _format_duration(self, td) -> str:
        """Formatiert eine Zeitdauer lesbar."""
        total_seconds = int(td.total_seconds())
        if total_seconds < 0:
            return "0m"

        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        minutes = (total_seconds % 3600) // 60

        if days > 0:
            return f"{days}d {hours}h"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"

    def _show_trade_statistics(self):
        """Zeigt den Trade-Statistik Dialog."""
        dialog = TradeStatisticsDialog(self, self.trades, self.data, self.equity_curve,
                                       self.initial_capital, self.current_equity)
        dialog.exec()


class TradeStatisticsDialog(QDialog):
    """Dialog fuer detaillierte Trade-Statistiken."""

    def __init__(self, parent, trades: list, data, equity_curve: list,
                 initial_capital: float, current_equity: float):
        super().__init__(parent)
        self.trades = trades
        self.data = data
        self.equity_curve = equity_curve
        self.initial_capital = initial_capital
        self.current_equity = current_equity

        self.setWindowTitle("Trade-Statistik")
        self.setMinimumSize(900, 700)
        self.setStyleSheet(get_stylesheet())

        self._setup_ui()

    def _setup_ui(self):
        """Erstellt die UI-Komponenten."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Tab-Widget fuer verschiedene Statistik-Bereiche
        tabs = QTabWidget()
        tabs.setStyleSheet(self._tab_style())

        # Tab 1: Uebersicht
        tabs.addTab(self._create_overview_tab(), "Uebersicht")

        # Tab 2: Trade-Liste
        tabs.addTab(self._create_trades_tab(), "Trade-Liste")

        # Tab 3: Long/Short Analyse
        tabs.addTab(self._create_long_short_tab(), "Long/Short")

        # Tab 4: Zeit-Analyse
        tabs.addTab(self._create_time_analysis_tab(), "Zeit-Analyse")

        layout.addWidget(tabs)

        # Export Button
        export_btn = QPushButton("Export als CSV")
        export_btn.setStyleSheet(StyleFactory.button_style_hex('#4da8da', padding='8px 20px'))
        export_btn.clicked.connect(self._export_csv)
        layout.addWidget(export_btn)

    def _create_overview_tab(self) -> QWidget:
        """Erstellt den Uebersicht-Tab."""
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # Linke Spalte: Konto
        left_group = QGroupBox("Konto-Uebersicht")
        left_group.setStyleSheet(self._group_style('#33b34d'))
        left_layout = QGridLayout(left_group)

        stats = self._calculate_account_stats()
        account_rows = [
            ("Startkapital:", f"${stats['start_capital']:,.2f}"),
            ("Aktuelles Kapital:", f"${stats['current_capital']:,.2f}"),
            ("Gesamt P/L:", f"${stats['total_pnl']:,.2f}"),
            ("Gesamt P/L %:", f"{stats['total_pnl_pct']:+.2f}%"),
            ("Max. Equity:", f"${stats['max_equity']:,.2f}"),
            ("Min. Equity:", f"${stats['min_equity']:,.2f}"),
            ("Max. Drawdown:", f"${stats['max_drawdown']:,.2f} ({stats['max_drawdown_pct']:.2f}%)"),
        ]

        for row, (label, value) in enumerate(account_rows):
            left_layout.addWidget(QLabel(label), row, 0)
            val_label = QLabel(value)
            val_label.setStyleSheet("color: white; font-weight: bold;")
            if 'P/L' in label:
                color = '#33cc33' if stats['total_pnl'] >= 0 else '#cc3333'
                val_label.setStyleSheet(f"color: {color}; font-weight: bold;")
            left_layout.addWidget(val_label, row, 1)

        layout.addWidget(left_group)

        # Rechte Spalte: Trade-Statistik
        right_group = QGroupBox("Trade-Statistik")
        right_group.setStyleSheet(self._group_style('#e6b333'))
        right_layout = QGridLayout(right_group)

        trade_stats = self._calculate_trade_stats()
        trade_rows = [
            ("Anzahl Trades:", str(trade_stats['total_trades'])),
            ("Gewinner:", f"{trade_stats['winners']} ({trade_stats['win_rate']:.1f}%)"),
            ("Verlierer:", f"{trade_stats['losers']} ({100-trade_stats['win_rate']:.1f}%)"),
            ("Profit Factor:", f"{trade_stats['profit_factor']:.2f}"),
            ("Avg. Trade:", f"${trade_stats['avg_trade']:,.2f}"),
            ("Avg. Gewinn:", f"${trade_stats['avg_win']:,.2f}"),
            ("Avg. Verlust:", f"${trade_stats['avg_loss']:,.2f}"),
            ("Groesster Gewinn:", f"${trade_stats['max_win']:,.2f}"),
            ("Groesster Verlust:", f"${trade_stats['max_loss']:,.2f}"),
            ("Win Streak:", str(trade_stats['max_win_streak'])),
            ("Loss Streak:", str(trade_stats['max_loss_streak'])),
        ]

        for row, (label, value) in enumerate(trade_rows):
            right_layout.addWidget(QLabel(label), row, 0)
            val_label = QLabel(value)
            val_label.setStyleSheet("color: white; font-weight: bold;")
            right_layout.addWidget(val_label, row, 1)

        layout.addWidget(right_group)

        return widget

    def _create_trades_tab(self) -> QWidget:
        """Erstellt den Trade-Liste Tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Tabelle
        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(9)
        self.trades_table.setHorizontalHeaderLabels([
            '#', 'Typ', 'Entry Zeit', 'Entry Preis', 'Exit Zeit', 'Exit Preis',
            'P/L', 'P/L %', 'Dauer'
        ])

        # Header-Styling
        header = self.trades_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setStyleSheet("QHeaderView::section { background-color: #333; color: white; padding: 5px; }")

        self.trades_table.setStyleSheet("""
            QTableWidget {
                background-color: #1a1a1a;
                color: white;
                gridline-color: #333;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QTableWidget::item:selected {
                background-color: #4da8da;
            }
        """)

        # Daten einfuegen
        self.trades_table.setRowCount(len(self.trades))
        for row, trade in enumerate(self.trades):
            self._add_trade_row(row, trade)

        layout.addWidget(self.trades_table)

        return widget

    def _add_trade_row(self, row: int, trade: dict):
        """Fuegt eine Trade-Zeile in die Tabelle ein."""
        entry_idx = trade.get('entry_index', 0) - 1
        exit_idx = trade.get('exit_index', 0) - 1
        pnl = trade.get('pnl', 0)
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        trade_type = trade.get('position', 'LONG')

        # Zeit-Informationen
        entry_time = ""
        exit_time = ""
        duration = ""
        if self.data is not None and entry_idx >= 0 and exit_idx >= 0:
            if entry_idx < len(self.data):
                entry_dt = self._get_datetime(entry_idx)
                entry_time = entry_dt.strftime('%Y-%m-%d %H:%M') if entry_dt else ""
            if exit_idx < len(self.data):
                exit_dt = self._get_datetime(exit_idx)
                exit_time = exit_dt.strftime('%Y-%m-%d %H:%M') if exit_dt else ""
            # Echte Zeitdauer berechnen
            if entry_dt and exit_dt:
                time_diff = exit_dt - entry_dt
                duration = self._format_duration(time_diff)
            else:
                duration_bars = exit_idx - entry_idx
                duration = f"{duration_bars} Bars"

        # P/L Prozent
        pnl_pct = (pnl / entry_price * 100) if entry_price > 0 else 0

        # Farbe basierend auf P/L
        color = QColor('#33cc33') if pnl >= 0 else QColor('#cc3333')

        items = [
            str(row + 1),
            trade_type,
            entry_time,
            f"${entry_price:,.2f}",
            exit_time,
            f"${exit_price:,.2f}",
            f"${pnl:,.2f}",
            f"{pnl_pct:+.2f}%",
            duration
        ]

        for col, text in enumerate(items):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if col in [6, 7]:  # P/L Spalten
                item.setForeground(color)
            self.trades_table.setItem(row, col, item)

    def _create_long_short_tab(self) -> QWidget:
        """Erstellt den Long/Short Analyse Tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Tabelle fuer Long/Short Vergleich
        table = QTableWidget()
        table.setColumnCount(4)
        table.setRowCount(7)
        table.setHorizontalHeaderLabels(['Metrik', 'Long', 'Short', 'Gesamt'])
        table.setVerticalHeaderLabels([''] * 7)

        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setStyleSheet("QHeaderView::section { background-color: #333; color: white; padding: 5px; }")

        table.setStyleSheet("""
            QTableWidget {
                background-color: #1a1a1a;
                color: white;
                gridline-color: #333;
            }
        """)

        # Statistiken berechnen
        long_stats = self._calculate_type_stats('LONG')
        short_stats = self._calculate_type_stats('SHORT')
        total_stats = self._calculate_trade_stats()

        rows = [
            ('Trades', long_stats['count'], short_stats['count'], total_stats['total_trades']),
            ('Gewinner', long_stats['winners'], short_stats['winners'], total_stats['winners']),
            ('Win-Rate', f"{long_stats['win_rate']:.1f}%", f"{short_stats['win_rate']:.1f}%", f"{total_stats['win_rate']:.1f}%"),
            ('P/L', f"${long_stats['pnl']:,.2f}", f"${short_stats['pnl']:,.2f}", f"${total_stats['total_pnl']:,.2f}"),
            ('Avg. Trade', f"${long_stats['avg_trade']:,.2f}", f"${short_stats['avg_trade']:,.2f}", f"${total_stats['avg_trade']:,.2f}"),
            ('Avg. Dauer', f"{long_stats['avg_duration']:.1f} Bars", f"{short_stats['avg_duration']:.1f} Bars", f"{total_stats['avg_duration']:.1f} Bars"),
            ('Profit Factor', f"{long_stats['profit_factor']:.2f}", f"{short_stats['profit_factor']:.2f}", f"{total_stats['profit_factor']:.2f}"),
        ]

        for row, (metric, long_val, short_val, total_val) in enumerate(rows):
            table.setItem(row, 0, QTableWidgetItem(metric))
            table.setItem(row, 1, QTableWidgetItem(str(long_val)))
            table.setItem(row, 2, QTableWidgetItem(str(short_val)))
            table.setItem(row, 3, QTableWidgetItem(str(total_val)))

        layout.addWidget(table)

        return widget

    def _create_time_analysis_tab(self) -> QWidget:
        """Erstellt den Zeit-Analyse Tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        time_stats = self._calculate_time_stats()

        # Zeit-Statistiken Gruppe
        time_group = QGroupBox("Zeit-Statistiken")
        time_group.setStyleSheet(self._group_style('#4da8da'))
        time_layout = QGridLayout(time_group)

        time_rows = [
            ("Backtest-Zeitraum:", time_stats['period']),
            ("Erster Trade:", time_stats['first_trade']),
            ("Letzter Trade:", time_stats['last_trade']),
            ("Trading-Tage:", str(time_stats['trading_days'])),
            ("Trades pro Tag:", f"{time_stats['trades_per_day']:.2f}"),
            ("Avg. Trade-Dauer:", time_stats['avg_duration']),
            ("Min. Trade-Dauer:", time_stats['min_duration']),
            ("Max. Trade-Dauer:", time_stats['max_duration']),
            ("Laengster Gewinn-Trade:", time_stats['longest_win']),
            ("Laengster Verlust-Trade:", time_stats['longest_loss']),
            ("Zeit im Markt:", time_stats['total_time_in_market']),
            ("Zeit im Markt (%):", time_stats['time_in_market_pct']),
        ]

        for row, (label, value) in enumerate(time_rows):
            time_layout.addWidget(QLabel(label), row, 0)
            val_label = QLabel(value)
            val_label.setStyleSheet("color: white; font-weight: bold;")
            time_layout.addWidget(val_label, row, 1)

        layout.addWidget(time_group)

        # Performance nach Tageszeit (wenn Daten vorhanden)
        if self.data is not None and len(self.trades) > 0:
            hourly_group = QGroupBox("Performance nach Stunde (Entry)")
            hourly_group.setStyleSheet(self._group_style('#b19cd9'))
            hourly_layout = QVBoxLayout(hourly_group)

            hourly_table = QTableWidget()
            hourly_stats = self._calculate_hourly_stats()

            hourly_table.setColumnCount(4)
            hourly_table.setRowCount(len(hourly_stats))
            hourly_table.setHorizontalHeaderLabels(['Stunde', 'Trades', 'Win-Rate', 'P/L'])

            header = hourly_table.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            header.setStyleSheet("QHeaderView::section { background-color: #333; color: white; }")

            hourly_table.setStyleSheet("""
                QTableWidget { background-color: #1a1a1a; color: white; gridline-color: #333; }
            """)

            for row, (hour, stats) in enumerate(sorted(hourly_stats.items())):
                hourly_table.setItem(row, 0, QTableWidgetItem(f"{hour:02d}:00"))
                hourly_table.setItem(row, 1, QTableWidgetItem(str(stats['count'])))
                hourly_table.setItem(row, 2, QTableWidgetItem(f"{stats['win_rate']:.1f}%"))
                pnl_item = QTableWidgetItem(f"${stats['pnl']:,.2f}")
                pnl_item.setForeground(QColor('#33cc33' if stats['pnl'] >= 0 else '#cc3333'))
                hourly_table.setItem(row, 3, pnl_item)

            hourly_layout.addWidget(hourly_table)
            layout.addWidget(hourly_group)

        layout.addStretch()
        return widget

    def _get_datetime(self, idx: int):
        """Holt DateTime aus dem DataFrame (Index oder Spalte)."""
        if self.data is None or idx < 0 or idx >= len(self.data):
            return None
        try:
            # Versuche Index (wenn DateTime als Index gesetzt)
            dt = self.data.index[idx]
            if hasattr(dt, 'strftime'):
                return dt
            # Versuche DateTime-Spalte
            if 'DateTime' in self.data.columns:
                dt_val = self.data['DateTime'].iloc[idx]
                # Falls String, in Timestamp konvertieren
                if isinstance(dt_val, str):
                    return pd.to_datetime(dt_val)
                if hasattr(dt_val, 'strftime'):
                    return dt_val
        except:
            pass
        return None

    def _format_duration(self, td) -> str:
        """Formatiert eine Zeitdauer lesbar."""
        total_seconds = int(td.total_seconds())
        if total_seconds < 0:
            return "0m"

        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        minutes = (total_seconds % 3600) // 60

        if days > 0:
            return f"{days}d {hours}h"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"

    def _calculate_account_stats(self) -> dict:
        """Berechnet Konto-Statistiken."""
        max_equity = max(self.equity_curve) if self.equity_curve else self.initial_capital
        min_equity = min(self.equity_curve) if self.equity_curve else self.initial_capital

        # Max Drawdown berechnen
        max_drawdown = 0
        peak = self.initial_capital
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            drawdown = peak - eq
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        total_pnl = self.current_equity - self.initial_capital
        total_pnl_pct = (total_pnl / self.initial_capital * 100) if self.initial_capital > 0 else 0
        max_drawdown_pct = (max_drawdown / peak * 100) if peak > 0 else 0

        return {
            'start_capital': self.initial_capital,
            'current_capital': self.current_equity,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'max_equity': max_equity,
            'min_equity': min_equity,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
        }

    def _calculate_trade_stats(self) -> dict:
        """Berechnet Trade-Statistiken."""
        if not self.trades:
            return {
                'total_trades': 0, 'winners': 0, 'losers': 0, 'win_rate': 0,
                'profit_factor': 0, 'avg_trade': 0, 'avg_win': 0, 'avg_loss': 0,
                'max_win': 0, 'max_loss': 0, 'max_win_streak': 0, 'max_loss_streak': 0,
                'total_pnl': 0, 'avg_duration': 0
            }

        pnls = [t.get('pnl', 0) for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        # Streaks berechnen
        win_streak = 0
        loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        for pnl in pnls:
            if pnl > 0:
                win_streak += 1
                loss_streak = 0
                max_win_streak = max(max_win_streak, win_streak)
            else:
                loss_streak += 1
                win_streak = 0
                max_loss_streak = max(max_loss_streak, loss_streak)

        # Durchschnittliche Dauer
        durations = []
        for t in self.trades:
            entry_idx = t.get('entry_index', 0)
            exit_idx = t.get('exit_index', 0)
            if entry_idx > 0 and exit_idx > 0:
                durations.append(exit_idx - entry_idx)

        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = (total_wins / total_losses) if total_losses > 0 else float('inf') if total_wins > 0 else 0

        return {
            'total_trades': len(self.trades),
            'winners': len(wins),
            'losers': len(losses),
            'win_rate': (len(wins) / len(self.trades) * 100) if self.trades else 0,
            'profit_factor': profit_factor if profit_factor != float('inf') else 999.99,
            'avg_trade': sum(pnls) / len(pnls) if pnls else 0,
            'avg_win': sum(wins) / len(wins) if wins else 0,
            'avg_loss': sum(losses) / len(losses) if losses else 0,
            'max_win': max(pnls) if pnls else 0,
            'max_loss': min(pnls) if pnls else 0,
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
            'total_pnl': sum(pnls),
            'avg_duration': sum(durations) / len(durations) if durations else 0,
        }

    def _calculate_type_stats(self, trade_type: str) -> dict:
        """Berechnet Statistiken fuer einen Trade-Typ (LONG/SHORT)."""
        type_trades = [t for t in self.trades if t.get('position') == trade_type]

        if not type_trades:
            return {
                'count': 0, 'winners': 0, 'win_rate': 0, 'pnl': 0,
                'avg_trade': 0, 'avg_duration': 0, 'profit_factor': 0
            }

        pnls = [t.get('pnl', 0) for t in type_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        durations = []
        for t in type_trades:
            entry_idx = t.get('entry_index', 0)
            exit_idx = t.get('exit_index', 0)
            if entry_idx > 0 and exit_idx > 0:
                durations.append(exit_idx - entry_idx)

        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = (total_wins / total_losses) if total_losses > 0 else float('inf') if total_wins > 0 else 0

        return {
            'count': len(type_trades),
            'winners': len(wins),
            'win_rate': (len(wins) / len(type_trades) * 100) if type_trades else 0,
            'pnl': sum(pnls),
            'avg_trade': sum(pnls) / len(pnls) if pnls else 0,
            'avg_duration': sum(durations) / len(durations) if durations else 0,
            'profit_factor': profit_factor if profit_factor != float('inf') else 999.99,
        }

    def _calculate_time_stats(self) -> dict:
        """Berechnet Zeit-bezogene Statistiken mit echten Zeitdauern."""
        if not self.trades or self.data is None:
            return {
                'period': '-', 'first_trade': '-', 'last_trade': '-',
                'trading_days': 0, 'trades_per_day': 0,
                'avg_duration': '-', 'min_duration': '-', 'max_duration': '-',
                'longest_win': '-', 'longest_loss': '-',
                'total_time_in_market': '-', 'time_in_market_pct': '-'
            }

        # Zeitraum
        start_dt = self._get_datetime(0)
        end_dt = self._get_datetime(len(self.data) - 1)
        period_start = start_dt.strftime('%Y-%m-%d') if start_dt else '-'
        period_end = end_dt.strftime('%Y-%m-%d') if end_dt else '-'
        period = f"{period_start} bis {period_end}"

        # Erster und letzter Trade
        first_trade = '-'
        last_trade = '-'
        if self.trades:
            first_idx = self.trades[0].get('entry_index', 0) - 1
            last_idx = self.trades[-1].get('exit_index', 0) - 1
            first_dt = self._get_datetime(first_idx)
            last_dt = self._get_datetime(last_idx)
            if first_dt:
                first_trade = first_dt.strftime('%Y-%m-%d %H:%M')
            if last_dt:
                last_trade = last_dt.strftime('%Y-%m-%d %H:%M')

        # Echte Zeitdauern berechnen
        durations = []  # in Sekunden
        win_durations = []
        loss_durations = []
        total_time_in_market = 0

        for t in self.trades:
            entry_idx = t.get('entry_index', 0) - 1
            exit_idx = t.get('exit_index', 0) - 1
            pnl = t.get('pnl', 0)

            entry_dt = self._get_datetime(entry_idx)
            exit_dt = self._get_datetime(exit_idx)

            if entry_dt and exit_dt:
                dur_seconds = (exit_dt - entry_dt).total_seconds()
                durations.append(dur_seconds)
                total_time_in_market += dur_seconds
                if pnl > 0:
                    win_durations.append(dur_seconds)
                else:
                    loss_durations.append(dur_seconds)

        # Trading-Tage (einzigartige Tage)
        trading_days = set()
        for t in self.trades:
            entry_idx = t.get('entry_index', 0) - 1
            entry_dt = self._get_datetime(entry_idx)
            if entry_dt:
                trading_days.add(entry_dt.strftime('%Y-%m-%d'))

        # Gesamtzeit des Backtests
        total_backtest_time = 0
        if start_dt and end_dt:
            total_backtest_time = (end_dt - start_dt).total_seconds()

        # Zeit im Markt als Prozent
        time_in_market_pct = (total_time_in_market / total_backtest_time * 100) if total_backtest_time > 0 else 0

        # Hilfsfunktion fuer Zeitformatierung
        def format_seconds(secs):
            if secs == 0:
                return "0m"
            td = pd.Timedelta(seconds=secs)
            return self._format_duration(td)

        return {
            'period': period,
            'first_trade': first_trade,
            'last_trade': last_trade,
            'trading_days': len(trading_days),
            'trades_per_day': len(self.trades) / len(trading_days) if trading_days else 0,
            'avg_duration': format_seconds(sum(durations) / len(durations)) if durations else '-',
            'min_duration': format_seconds(min(durations)) if durations else '-',
            'max_duration': format_seconds(max(durations)) if durations else '-',
            'longest_win': format_seconds(max(win_durations)) if win_durations else '-',
            'longest_loss': format_seconds(max(loss_durations)) if loss_durations else '-',
            'total_time_in_market': format_seconds(total_time_in_market),
            'time_in_market_pct': f"{time_in_market_pct:.1f}%",
        }

    def _calculate_hourly_stats(self) -> dict:
        """Berechnet Statistiken nach Stunde."""
        hourly = {}

        for t in self.trades:
            entry_idx = t.get('entry_index', 0) - 1
            if entry_idx >= 0 and entry_idx < len(self.data):
                try:
                    hour = self.data.index[entry_idx].hour
                except:
                    continue

                if hour not in hourly:
                    hourly[hour] = {'count': 0, 'wins': 0, 'pnl': 0}

                hourly[hour]['count'] += 1
                hourly[hour]['pnl'] += t.get('pnl', 0)
                if t.get('pnl', 0) > 0:
                    hourly[hour]['wins'] += 1

        # Win-Rate berechnen
        for hour in hourly:
            count = hourly[hour]['count']
            hourly[hour]['win_rate'] = (hourly[hour]['wins'] / count * 100) if count > 0 else 0

        return hourly

    def _export_csv(self):
        """Exportiert die Trades als CSV."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Trades exportieren", "trades_export.csv", "CSV Files (*.csv)"
        )

        if filename:
            try:
                import csv
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['#', 'Typ', 'Entry Zeit', 'Entry Preis', 'Exit Zeit',
                                    'Exit Preis', 'P/L', 'P/L %', 'Dauer (Bars)'])

                    for i, trade in enumerate(self.trades):
                        entry_idx = trade.get('entry_index', 0) - 1
                        exit_idx = trade.get('exit_index', 0) - 1
                        entry_time = str(self.data.index[entry_idx]) if entry_idx >= 0 and entry_idx < len(self.data) else ''
                        exit_time = str(self.data.index[exit_idx]) if exit_idx >= 0 and exit_idx < len(self.data) else ''
                        entry_price = trade.get('entry_price', 0)
                        pnl = trade.get('pnl', 0)
                        pnl_pct = (pnl / entry_price * 100) if entry_price > 0 else 0
                        duration = exit_idx - entry_idx if entry_idx >= 0 and exit_idx >= 0 else 0

                        writer.writerow([
                            i + 1, trade.get('position', ''), entry_time,
                            f"{trade.get('entry_price', 0):.2f}", exit_time,
                            f"{trade.get('exit_price', 0):.2f}",
                            f"{pnl:.2f}", f"{pnl_pct:.2f}", duration
                        ])

            except Exception as e:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Export Fehler", f"Fehler beim Export: {e}")

    def _tab_style(self) -> str:
        """Gibt das Tab-Stylesheet zurueck."""
        return '''
            QTabWidget::pane { border: 1px solid #4d4d4d; background-color: #262626; }
            QTabBar::tab { background-color: #333; color: #b3b3b3; padding: 8px 20px;
                          margin-right: 2px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background-color: #4da8da; color: white; }
            QTabBar::tab:hover:!selected { background-color: #444; }
        '''

    def _group_style(self, color: str) -> str:
        """Gibt das GroupBox-Stylesheet zurueck."""
        return f'''
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {color};
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                color: {color};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
            QLabel {{ color: #b3b3b3; }}
        '''
