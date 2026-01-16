"""
Chart Panel - Charts fuer den Backtest (mittlere Spalte).
"""

from typing import List, Dict, Optional, Callable

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTabWidget,
    QSlider, QSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal

import pandas as pd
import numpy as np
import pyqtgraph as pg
from pyqtgraph import DateAxisItem

from ..styles import StyleFactory, COLORS


class CandlestickItem(pg.GraphicsObject):
    """Kerzenchart-Item fuer pyqtgraph."""

    def __init__(self, data: np.ndarray = None):
        """
        data: ndarray mit Spalten [timestamp, open, high, low, close]
        timestamp ist Unix-Timestamp (Sekunden)
        """
        pg.GraphicsObject.__init__(self)
        self.data = data
        self.picture = None
        self.generatePicture()

    def setData(self, data: np.ndarray):
        """Setzt neue Daten und zeichnet neu."""
        self.data = data
        self.picture = None
        self.generatePicture()
        self.update()

    def generatePicture(self):
        """Generiert das QPicture fuer die Kerzen."""
        from PyQt6.QtGui import QPicture, QPainter, QPen, QBrush, QColor
        from PyQt6.QtCore import QRectF

        self.picture = QPicture()
        p = QPainter(self.picture)

        if self.data is None or len(self.data) == 0:
            p.end()
            return

        # Kerzenbreite berechnen (basierend auf Zeitabstand)
        if len(self.data) > 1:
            time_diff = self.data[1, 0] - self.data[0, 0]
            w = time_diff * 0.6  # 60% des Abstands
        else:
            w = 3600 * 0.6  # Default: 1 Stunde

        for row in self.data:
            t, o, h, l, c = row[0], row[1], row[2], row[3], row[4]

            if c >= o:
                # Bullish Kerze - helles Grau
                color = QColor('#a0a0a0')
            else:
                # Bearish Kerze - dunkles Grau
                color = QColor('#505050')

            # Docht (Linie)
            p.setPen(pg.mkPen(color, width=1))
            p.drawLine(pg.QtCore.QPointF(t, l), pg.QtCore.QPointF(t, h))

            # Kerzenkoerper
            p.setBrush(QBrush(color))
            body_top = max(o, c)
            body_bottom = min(o, c)
            body_height = body_top - body_bottom
            if body_height < 0.01:  # Mindesthoehe fuer Doji
                body_height = (h - l) * 0.01 if (h - l) > 0 else 1

            p.drawRect(QRectF(t - w/2, body_bottom, w, body_height))

        p.end()

    def paint(self, p, *args):
        if self.picture:
            self.picture.play(p)

    def boundingRect(self):
        if self.picture is None:
            return pg.QtCore.QRectF(0, 0, 1, 1)
        return pg.QtCore.QRectF(self.picture.boundingRect())


class ChartPanel(QWidget):
    """
    Chart-Panel fuer den Backtester (mittlere Spalte).

    Enthaelt:
    - Preis-Chart mit Signalen (matplotlib)
    - Trade-Chart mit Entry/Exit (pyqtgraph - schneller)
    - Equity-Chart (matplotlib)
    """

    # Signals
    trade_clicked = pyqtSignal(int)  # trade_index

    def __init__(self, parent=None):
        super().__init__(parent)
        self._has_matplotlib = False
        self.data: Optional[pd.DataFrame] = None
        self.trades: List[Dict] = []
        self.signal_history: List[Dict] = []
        self.equity_curve: List[float] = []
        self.equity_indices: List[int] = []
        self.initial_capital = 10000.0

        # Trade-Navigation
        self.current_trade_view = -1
        self.trade_lines = []
        self.trade_labels = []

        # Chart-Modus: 'candle' oder 'line'
        self.chart_mode = 'candle'

        # Signal-Offsets (vertikal, in Dollar)
        self.signal_entry_offset = 0.0  # Offset fuer Entry-Marker
        self.signal_exit_offset = 0.0   # Offset fuer Exit-Marker

        self._setup_ui()

    def _setup_ui(self):
        """Erstellt die UI-Komponenten."""
        self.setStyleSheet("background-color: rgb(46, 46, 46);")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
            from matplotlib.figure import Figure

            self._has_matplotlib = True

            # Zoom-Kontrollen oben
            zoom_controls = self._create_zoom_controls()
            layout.addWidget(zoom_controls)

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

            self.price_toolbar = NavigationToolbar(self.price_canvas, self)
            self.price_toolbar.setStyleSheet(self._toolbar_style())
            price_layout.addWidget(self.price_toolbar)
            price_layout.addWidget(self.price_canvas)

            self.chart_tabs.addTab(price_widget, "Preis + Signale")

            # === Tab 2: Trade-Chart (pyqtgraph) ===
            trade_widget = self._create_trade_tab()
            self.chart_tabs.addTab(trade_widget, "Trades")

            # === Tab 3: Equity-Chart ===
            equity_widget = QWidget()
            equity_layout = QVBoxLayout(equity_widget)
            equity_layout.setContentsMargins(0, 0, 0, 0)

            self.equity_figure = Figure(figsize=(8, 6), facecolor='#262626')
            self.equity_canvas = FigureCanvas(self.equity_figure)
            self.ax_equity = self.equity_figure.add_subplot(111)
            self._setup_equity_chart()

            self.equity_toolbar = NavigationToolbar(self.equity_canvas, self)
            self.equity_toolbar.setStyleSheet(self._toolbar_style())
            equity_layout.addWidget(self.equity_toolbar)
            equity_layout.addWidget(self.equity_canvas)

            self.chart_tabs.addTab(equity_widget, "Equity")

            layout.addWidget(self.chart_tabs, stretch=1)

        except ImportError:
            layout.addWidget(QLabel("matplotlib nicht installiert"))

    def _create_trade_tab(self) -> QWidget:
        """Erstellt den Trade-Chart Tab mit pyqtgraph und Kerzenchart."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # pyqtgraph konfigurieren
        pg.setConfigOptions(antialias=True)

        # DateAxisItem fuer echte Zeitachse
        date_axis = DateAxisItem(orientation='bottom')

        # PlotWidget mit DateAxisItem erstellen
        self.trade_plot = pg.PlotWidget(axisItems={'bottom': date_axis})
        self.trade_plot.setBackground('#262626')
        self.trade_plot.setLabel('left', 'Preis ($)')
        self.trade_plot.setLabel('bottom', 'Zeit')
        self.trade_plot.showGrid(x=True, y=True, alpha=0.3)
        self.trade_plot.getAxis('left').setPen(pg.mkPen(color='#808080'))
        self.trade_plot.getAxis('bottom').setPen(pg.mkPen(color='#808080'))
        self.trade_plot.getAxis('left').setTextPen(pg.mkPen(color='#aaaaaa'))
        self.trade_plot.getAxis('bottom').setTextPen(pg.mkPen(color='#aaaaaa'))

        # Kerzenchart-Item
        self.candlestick_item = CandlestickItem()
        self.trade_plot.addItem(self.candlestick_item)

        # Linienchart-Item (alternativ zu Kerzen)
        self.trade_price_line = self.trade_plot.plot([], [], pen=pg.mkPen(color='#6699cc', width=1.5))
        self.trade_price_line.setVisible(False)  # Initial unsichtbar

        # Timestamp-Array fuer Koordinaten-Umrechnung
        self.timestamps: np.ndarray = None

        # Scatter-Items fuer Trade-Marker
        self.trade_entry_scatter = pg.ScatterPlotItem(size=14, pen=pg.mkPen(color='white', width=1.5))
        self.trade_exit_scatter = pg.ScatterPlotItem(size=12, pen=pg.mkPen(color='white', width=1.5))
        self.trade_plot.addItem(self.trade_entry_scatter)
        self.trade_plot.addItem(self.trade_exit_scatter)

        # Click-Event fuer Scatter
        self.trade_entry_scatter.sigClicked.connect(self._on_scatter_clicked)
        self.trade_exit_scatter.sigClicked.connect(self._on_scatter_clicked)

        # Titel-Label oben
        self.trade_chart_title = QLabel("Trades | 0 Trades | 0W / 0L | Win-Rate: 0.0%")
        self.trade_chart_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.trade_chart_title.setStyleSheet("color: white; font-size: 11px; padding: 3px;")

        layout.addWidget(self.trade_chart_title)
        layout.addWidget(self.trade_plot, stretch=1)

        # Trade-Navigation mit Chart-Modus-Toggle
        nav_widget = QWidget()
        nav_layout = QHBoxLayout(nav_widget)
        nav_layout.setContentsMargins(5, 2, 5, 2)
        nav_layout.setSpacing(10)

        # Chart-Modus Toggle
        self.btn_chart_mode = QPushButton("Linie")
        self.btn_chart_mode.setStyleSheet(self._toggle_button_style(False))
        self.btn_chart_mode.setFixedWidth(70)
        self.btn_chart_mode.setToolTip("Zwischen Kerzen- und Linienchart wechseln")
        self.btn_chart_mode.clicked.connect(self._toggle_chart_mode)

        self.btn_prev_trade = QPushButton("<< Vorheriger")
        self.btn_prev_trade.setStyleSheet(self._nav_button_style())
        self.btn_prev_trade.clicked.connect(self._prev_trade)

        self.trade_nav_label = QLabel("Trade 0/0")
        self.trade_nav_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.trade_nav_label.setStyleSheet("color: #aaa; font-size: 11px; min-width: 80px;")

        self.btn_next_trade = QPushButton("Naechster >>")
        self.btn_next_trade.setStyleSheet(self._nav_button_style())
        self.btn_next_trade.clicked.connect(self._next_trade)

        nav_layout.addWidget(self.btn_chart_mode)
        nav_layout.addStretch()
        nav_layout.addWidget(self.btn_prev_trade)
        nav_layout.addWidget(self.trade_nav_label)
        nav_layout.addWidget(self.btn_next_trade)
        nav_layout.addStretch()

        layout.addWidget(nav_widget)

        # Signal-Offset Kontrollen
        offset_widget = self._create_signal_offset_controls()
        layout.addWidget(offset_widget)

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
        layout.addWidget(self.trade_detail_label)

        return widget

    def _create_signal_offset_controls(self) -> QWidget:
        """Erstellt Kontrollen fuer vertikale Signal-Offsets."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(8)

        # Style fuer Labels
        label_style = "color: #aaa; font-size: 10px;"
        spinbox_style = """
            QSpinBox {
                background-color: #333;
                color: white;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 2px 5px;
                min-width: 60px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #444;
                border: none;
                width: 15px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #555;
            }
        """

        # Entry-Offset
        entry_label = QLabel("Entry Y-Offset ($):")
        entry_label.setStyleSheet(label_style)
        layout.addWidget(entry_label)

        self.entry_offset_spin = QSpinBox()
        self.entry_offset_spin.setRange(-5000, 5000)
        self.entry_offset_spin.setValue(0)
        self.entry_offset_spin.setSingleStep(10)
        self.entry_offset_spin.setStyleSheet(spinbox_style)
        self.entry_offset_spin.setToolTip("Vertikaler Offset fuer Entry-Marker (positiv = hoeher)")
        self.entry_offset_spin.valueChanged.connect(self._on_entry_offset_changed)
        layout.addWidget(self.entry_offset_spin)

        layout.addSpacing(20)

        # Exit-Offset
        exit_label = QLabel("Exit Y-Offset ($):")
        exit_label.setStyleSheet(label_style)
        layout.addWidget(exit_label)

        self.exit_offset_spin = QSpinBox()
        self.exit_offset_spin.setRange(-5000, 5000)
        self.exit_offset_spin.setValue(0)
        self.exit_offset_spin.setSingleStep(10)
        self.exit_offset_spin.setStyleSheet(spinbox_style)
        self.exit_offset_spin.setToolTip("Vertikaler Offset fuer Exit-Marker (positiv = hoeher)")
        self.exit_offset_spin.valueChanged.connect(self._on_exit_offset_changed)
        layout.addWidget(self.exit_offset_spin)

        layout.addSpacing(20)

        # Reset Button
        reset_btn = QPushButton("Reset")
        reset_btn.setFixedWidth(50)
        reset_btn.setStyleSheet("""
            QPushButton { background-color: #444; color: #aaa; border: 1px solid #555;
                         border-radius: 3px; padding: 3px 8px; font-size: 10px; }
            QPushButton:hover { background-color: #555; color: white; }
        """)
        reset_btn.setToolTip("Beide Offsets auf 0 zuruecksetzen")
        reset_btn.clicked.connect(self._reset_signal_offsets)
        layout.addWidget(reset_btn)

        layout.addStretch()

        return widget

    def _on_entry_offset_changed(self, value: int):
        """Handler fuer Entry-Offset Aenderung."""
        self.signal_entry_offset = float(value)
        self._refresh_trade_markers()

    def _on_exit_offset_changed(self, value: int):
        """Handler fuer Exit-Offset Aenderung."""
        self.signal_exit_offset = float(value)
        self._refresh_trade_markers()

    def _reset_signal_offsets(self):
        """Setzt beide Offsets auf 0 zurueck."""
        self.entry_offset_spin.setValue(0)
        self.exit_offset_spin.setValue(0)
        self.signal_entry_offset = 0.0
        self.signal_exit_offset = 0.0
        self._refresh_trade_markers()

    def _refresh_trade_markers(self):
        """Aktualisiert nur die Trade-Marker mit den aktuellen Offsets."""
        if not hasattr(self, 'trade_plot') or self.data is None:
            return
        if self.timestamps is None or len(self.timestamps) == 0:
            return

        # Aktuelle Trade-Daten holen
        if not hasattr(self, '_last_trade_params'):
            return

        params = self._last_trade_params
        self._update_trade_markers_only(
            params['trades'],
            params['position'],
            params['entry_price'],
            params['entry_index'],
            params['current_index']
        )

    def _update_trade_markers_only(self, trades: List[Dict], position: str,
                                    entry_price: float, entry_index: int, current_index: int):
        """Aktualisiert nur die Trade-Marker (ohne Chart-Daten neu zu laden)."""
        # Alte Linien und Labels entfernen
        for line in self.trade_lines:
            self.trade_plot.removeItem(line)
        self.trade_lines.clear()
        for label in self.trade_labels:
            self.trade_plot.removeItem(label)
        self.trade_labels.clear()

        # Trade-Marker sammeln
        entry_spots = []
        exit_spots = []
        wins = 0
        losses = 0

        for trade_index, trade in enumerate(trades):
            entry_idx = trade.get('entry_index', 0) - 1
            exit_idx = trade.get('exit_index', 0) - 1
            pnl = trade.get('pnl', 0)

            if entry_idx < 0 or exit_idx < 0:
                continue
            if entry_idx >= len(self.data) or exit_idx >= len(self.data):
                continue

            trade_entry_price = trade.get('entry_price', 0)
            exit_price = trade.get('exit_price', 0)
            trade_type = trade.get('position', 'LONG')

            # Timestamps fuer Entry/Exit
            entry_ts = self.timestamps[entry_idx]
            exit_ts = self.timestamps[exit_idx]

            # Preise mit Offset
            entry_price_offset = trade_entry_price + self.signal_entry_offset
            exit_price_offset = exit_price + self.signal_exit_offset

            # Farbe basierend auf P/L
            if pnl > 0:
                line_color = '#33cc33'
                wins += 1
            else:
                line_color = '#cc3333'
                losses += 1

            # Verbindungslinie (mit Offsets)
            line = self.trade_plot.plot(
                [entry_ts, exit_ts], [entry_price_offset, exit_price_offset],
                pen=pg.mkPen(color=line_color, width=2)
            )
            self.trade_lines.append(line)

            # Entry-Marker (mit Offset)
            entry_color = '#33cc33' if trade_type == 'LONG' else '#cc3333'
            entry_symbol = 't' if trade_type == 'LONG' else 't1'
            entry_spots.append({
                'pos': (entry_ts, entry_price_offset),
                'brush': pg.mkBrush(entry_color),
                'symbol': entry_symbol,
                'size': 14,
                'data': trade_index
            })

            # Exit-Marker (mit Offset)
            exit_spots.append({
                'pos': (exit_ts, exit_price_offset),
                'brush': pg.mkBrush(line_color),
                'symbol': 'x',
                'size': 12,
                'data': trade_index
            })

            # P/L Label
            time_offset = (self.timestamps[1] - self.timestamps[0]) if len(self.timestamps) > 1 else 3600
            pnl_text = f'+${pnl:.0f}' if pnl > 0 else f'-${abs(pnl):.0f}'
            label = pg.TextItem(text=pnl_text, color=line_color, anchor=(0, 0.5))
            label.setPos(exit_ts + time_offset, exit_price_offset)
            self.trade_plot.addItem(label)
            self.trade_labels.append(label)

        # Aktuelle offene Position
        if position != 'NONE' and entry_index > 0:
            entry_idx = entry_index - 1
            if entry_idx < len(self.data):
                curr_idx = min(current_index - 1, len(self.data) - 1)
                curr_price = self.data['Close'].iloc[curr_idx]

                entry_ts = self.timestamps[entry_idx]
                curr_ts = self.timestamps[curr_idx]

                entry_price_offset = entry_price + self.signal_entry_offset
                curr_price_offset = curr_price + self.signal_exit_offset

                line = self.trade_plot.plot(
                    [entry_ts, curr_ts], [entry_price_offset, curr_price_offset],
                    pen=pg.mkPen(color='#e6b333', width=2, style=Qt.PenStyle.DashLine)
                )
                self.trade_lines.append(line)

                open_color = '#33cc33' if position == 'LONG' else '#cc3333'
                open_symbol = 't' if position == 'LONG' else 't1'
                entry_spots.append({
                    'pos': (entry_ts, entry_price_offset),
                    'brush': pg.mkBrush(open_color),
                    'symbol': open_symbol,
                    'size': 14,
                    'data': -1
                })

        # Scatter-Daten setzen
        self.trade_entry_scatter.setData(entry_spots)
        self.trade_exit_scatter.setData(exit_spots)

    def _create_zoom_controls(self) -> QWidget:
        """Erstellt die Zoom-Kontrollen."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # X-Zoom
        layout.addWidget(QLabel('X:'))

        zoom_x_in = QPushButton('+')
        zoom_x_in.setFixedWidth(30)
        zoom_x_in.setStyleSheet(StyleFactory.button_style_hex('#555555', padding='3px 8px'))
        zoom_x_in.clicked.connect(lambda: self._zoom_axis('x', 0.8))
        layout.addWidget(zoom_x_in)

        zoom_x_out = QPushButton('-')
        zoom_x_out.setFixedWidth(30)
        zoom_x_out.setStyleSheet(StyleFactory.button_style_hex('#555555', padding='3px 8px'))
        zoom_x_out.clicked.connect(lambda: self._zoom_axis('x', 1.25))
        layout.addWidget(zoom_x_out)

        # Y-Zoom
        layout.addWidget(QLabel('Y:'))

        zoom_y_in = QPushButton('+')
        zoom_y_in.setFixedWidth(30)
        zoom_y_in.setStyleSheet(StyleFactory.button_style_hex('#555555', padding='3px 8px'))
        zoom_y_in.clicked.connect(lambda: self._zoom_axis('y', 0.8))
        layout.addWidget(zoom_y_in)

        zoom_y_out = QPushButton('-')
        zoom_y_out.setFixedWidth(30)
        zoom_y_out.setStyleSheet(StyleFactory.button_style_hex('#555555', padding='3px 8px'))
        zoom_y_out.clicked.connect(lambda: self._zoom_axis('y', 1.25))
        layout.addWidget(zoom_y_out)

        # Reset
        reset_btn = QPushButton('Reset')
        reset_btn.setStyleSheet(StyleFactory.button_style_hex('#666666', padding='3px 10px'))
        reset_btn.clicked.connect(self._reset_zoom)
        layout.addWidget(reset_btn)

        # Folgen
        follow_btn = QPushButton('Folgen')
        follow_btn.setStyleSheet(StyleFactory.button_style_hex('#4da8da', padding='3px 10px'))
        follow_btn.setToolTip('Zum aktuellen Datenpunkt springen')
        follow_btn.clicked.connect(self._follow_current)
        layout.addWidget(follow_btn)

        layout.addStretch()
        return widget

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

    def _get_current_chart(self):
        """Gibt den aktuell ausgewaehlten Chart (ax, canvas) zurueck."""
        if not hasattr(self, 'chart_tabs'):
            return self.ax_price, self.price_canvas

        current_tab = self.chart_tabs.currentIndex()
        if current_tab == 0:
            return self.ax_price, self.price_canvas
        elif current_tab == 1:
            # Trade-Chart ist pyqtgraph
            return None, None
        else:
            return self.ax_equity, self.equity_canvas

    def _zoom_axis(self, axis: str, factor: float):
        """Zoomt eine einzelne Achse des aktuell sichtbaren Charts."""
        ax, canvas = self._get_current_chart()

        if ax is None:
            return  # pyqtgraph hat eigenes Zoom

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

    def _reset_zoom(self):
        """Setzt den Zoom zurueck."""
        ax, canvas = self._get_current_chart()

        if ax is None:
            if hasattr(self, 'trade_plot'):
                self.trade_plot.autoRange()
            return

        ax.autoscale()
        canvas.draw()

    def _follow_current(self):
        """Springt zum aktuellen Datenpunkt."""
        if self.data is None or self.current_index == 0:
            return

        ax, canvas = self._get_current_chart()

        window_size = 200
        start_idx = max(0, self.current_index - window_size // 2)
        end_idx = min(len(self.data), self.current_index + window_size // 2)

        if ax is None:
            # Trade-Chart mit Timestamps
            if hasattr(self, 'trade_plot') and self.timestamps is not None and len(self.timestamps) > 0:
                start_ts = self.timestamps[start_idx]
                end_ts = self.timestamps[min(end_idx, len(self.timestamps) - 1)]
                self.trade_plot.setXRange(start_ts, end_ts, padding=0.02)

                # Y-Range anpassen
                visible_data = self.data.iloc[start_idx:end_idx]
                if len(visible_data) > 0:
                    y_min = visible_data['Low'].min() * 0.998
                    y_max = visible_data['High'].max() * 1.002
                    self.trade_plot.setYRange(y_min, y_max, padding=0)
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

    # === Oeffentliche Methoden ===

    def set_data(self, data: pd.DataFrame, initial_capital: float = 10000.0):
        """Setzt die Backtest-Daten."""
        self.data = data
        self.initial_capital = initial_capital
        self.current_index = 0

    def initialize_charts(self):
        """Initialisiert die Charts."""
        if not self._has_matplotlib:
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

    def update_charts(self, current_index: int, signal_history: List[Dict],
                     trades: List[Dict], equity_curve: List[float],
                     equity_indices: List[int], buy_count: int, sell_count: int,
                     hold_count: int, total_pnl: float, current_equity: float,
                     position: str = 'NONE', entry_price: float = 0.0,
                     entry_index: int = 0):
        """Aktualisiert alle Charts."""
        self.current_index = current_index
        self.signal_history = signal_history
        self.trades = trades
        self.equity_curve = equity_curve
        self.equity_indices = equity_indices

        if not self._has_matplotlib:
            return

        # Aktuelle Zoom-Limits speichern
        price_xlim = self.ax_price.get_xlim()
        price_ylim = self.ax_price.get_ylim()
        equity_xlim = self.ax_equity.get_xlim() if hasattr(self, 'ax_equity') else None
        equity_ylim = self.ax_equity.get_ylim() if hasattr(self, 'ax_equity') else None

        # Preis-Chart aktualisieren
        self.ax_price.clear()
        self._setup_price_chart()

        if self.data is not None:
            self.ax_price.plot(self.data.index, self.data['Close'],
                              color='#6699e6', linewidth=1)

            # BUY Signale
            buy_indices = [s['index'] - 1 for s in signal_history if s['signal'] == 1]
            if buy_indices:
                buy_prices = [self.data['Close'].iloc[i] for i in buy_indices if i < len(self.data)]
                buy_dates = [self.data.index[i] for i in buy_indices if i < len(self.data)]
                self.ax_price.scatter(buy_dates, buy_prices, marker='^', c='lime', s=50, zorder=5)

            # SELL Signale
            sell_indices = [s['index'] - 1 for s in signal_history if s['signal'] == 2]
            if sell_indices:
                sell_prices = [self.data['Close'].iloc[i] for i in sell_indices if i < len(self.data)]
                sell_dates = [self.data.index[i] for i in sell_indices if i < len(self.data)]
                self.ax_price.scatter(sell_dates, sell_prices, marker='v', c='red', s=50, zorder=5)

            # Aktuelle Position markieren
            if current_index <= len(self.data):
                idx = current_index - 1
                self.ax_price.axvline(x=self.data.index[idx], color='yellow',
                                      linestyle='--', linewidth=1, alpha=0.7)

        self.ax_price.set_title(f'Preis und Signale | BUY: {buy_count}, SELL: {sell_count}, HOLD: {hold_count}',
                                color='white', fontsize=11)

        # Zoom wiederherstellen
        if price_xlim[0] != 0.0 or price_xlim[1] != 1.0:
            self.ax_price.set_xlim(price_xlim)
            self.ax_price.set_ylim(price_ylim)

        self.price_figure.tight_layout()
        self.price_canvas.draw()

        # Trade-Chart aktualisieren
        self._update_trade_chart(trades, position, entry_price, entry_index, current_index)

        # Equity-Chart aktualisieren
        self.ax_equity.clear()
        self._setup_equity_chart()

        if len(equity_indices) > 1 and self.data is not None:
            valid_indices = [i - 1 for i in equity_indices if i - 1 < len(self.data) and i > 0]
            if valid_indices:
                dates = [self.data.index[i] for i in valid_indices]
                equity_vals = equity_curve[:len(valid_indices)]
                self.ax_equity.plot(dates, equity_vals, color='lime', linewidth=1.5)

        self.ax_equity.axhline(y=self.initial_capital, color='gray',
                               linestyle='--', linewidth=1)

        pnl_pct = ((current_equity - self.initial_capital) / self.initial_capital) * 100
        self.ax_equity.set_title(f'Equity-Kurve | P/L: ${total_pnl:,.2f} ({pnl_pct:.2f}%)',
                                 color='white', fontsize=11)

        if equity_xlim is not None and (equity_xlim[0] != 0.0 or equity_xlim[1] != 1.0):
            self.ax_equity.set_xlim(equity_xlim)
            self.ax_equity.set_ylim(equity_ylim)

        self.equity_figure.tight_layout()
        self.equity_canvas.draw()

    def _update_trade_chart(self, trades: List[Dict], position: str,
                           entry_price: float, entry_index: int, current_index: int):
        """Aktualisiert den Trade-Chart mit Kerzen und echter Zeitachse."""
        if not hasattr(self, 'trade_plot') or self.data is None:
            return

        # Parameter speichern fuer Offset-Updates
        self._last_trade_params = {
            'trades': trades,
            'position': position,
            'entry_price': entry_price,
            'entry_index': entry_index,
            'current_index': current_index
        }

        # Alte Linien und Labels entfernen
        for line in self.trade_lines:
            self.trade_plot.removeItem(line)
        self.trade_lines.clear()
        for label in self.trade_labels:
            self.trade_plot.removeItem(label)
        self.trade_labels.clear()

        # Timestamps aus DataFrame extrahieren
        self.timestamps = self._extract_timestamps()

        # Chart-Daten aktualisieren je nach Modus
        if self.chart_mode == 'candle':
            # Kerzendaten erstellen: [timestamp, open, high, low, close]
            candle_data = np.column_stack([
                self.timestamps,
                self.data['Open'].values,
                self.data['High'].values,
                self.data['Low'].values,
                self.data['Close'].values
            ])
            self.candlestick_item.setData(candle_data)
        else:
            # Liniendaten aktualisieren (Close-Preis)
            self.trade_price_line.setData(self.timestamps, self.data['Close'].values)

        # Trade-Marker sammeln
        entry_spots = []
        exit_spots = []
        wins = 0
        losses = 0

        for trade_index, trade in enumerate(trades):
            entry_idx = trade.get('entry_index', 0) - 1
            exit_idx = trade.get('exit_index', 0) - 1
            pnl = trade.get('pnl', 0)

            if entry_idx < 0 or exit_idx < 0:
                continue
            if entry_idx >= len(self.data) or exit_idx >= len(self.data):
                continue

            trade_entry_price = trade.get('entry_price', 0)
            exit_price = trade.get('exit_price', 0)
            trade_type = trade.get('position', 'LONG')

            # Timestamps fuer Entry/Exit
            entry_ts = self.timestamps[entry_idx]
            exit_ts = self.timestamps[exit_idx]

            # Preise mit Offset
            entry_price_offset = trade_entry_price + self.signal_entry_offset
            exit_price_offset = exit_price + self.signal_exit_offset

            # Farbe basierend auf P/L
            if pnl > 0:
                line_color = '#33cc33'
                wins += 1
            else:
                line_color = '#cc3333'
                losses += 1

            # Verbindungslinie (mit Offsets)
            line = self.trade_plot.plot(
                [entry_ts, exit_ts], [entry_price_offset, exit_price_offset],
                pen=pg.mkPen(color=line_color, width=2)
            )
            self.trade_lines.append(line)

            # Entry-Marker (mit Offset)
            entry_color = '#33cc33' if trade_type == 'LONG' else '#cc3333'
            entry_symbol = 't' if trade_type == 'LONG' else 't1'
            entry_spots.append({
                'pos': (entry_ts, entry_price_offset),
                'brush': pg.mkBrush(entry_color),
                'symbol': entry_symbol,
                'size': 14,
                'data': trade_index
            })

            # Exit-Marker (mit Offset)
            exit_spots.append({
                'pos': (exit_ts, exit_price_offset),
                'brush': pg.mkBrush(line_color),
                'symbol': 'x',
                'size': 12,
                'data': trade_index
            })

            # P/L Label (mit Offset)
            time_offset = (self.timestamps[1] - self.timestamps[0]) if len(self.timestamps) > 1 else 3600
            pnl_text = f'+${pnl:.0f}' if pnl > 0 else f'-${abs(pnl):.0f}'
            label = pg.TextItem(text=pnl_text, color=line_color, anchor=(0, 0.5))
            label.setPos(exit_ts + time_offset, exit_price_offset)
            self.trade_plot.addItem(label)
            self.trade_labels.append(label)

        # Aktuelle offene Position
        if position != 'NONE' and entry_index > 0:
            entry_idx = entry_index - 1
            if entry_idx < len(self.data):
                curr_idx = min(current_index - 1, len(self.data) - 1)
                curr_price = self.data['Close'].iloc[curr_idx]

                entry_ts = self.timestamps[entry_idx]
                curr_ts = self.timestamps[curr_idx]

                entry_price_offset = entry_price + self.signal_entry_offset
                curr_price_offset = curr_price + self.signal_exit_offset

                line = self.trade_plot.plot(
                    [entry_ts, curr_ts], [entry_price_offset, curr_price_offset],
                    pen=pg.mkPen(color='#e6b333', width=2, style=Qt.PenStyle.DashLine)
                )
                self.trade_lines.append(line)

                open_color = '#33cc33' if position == 'LONG' else '#cc3333'
                open_symbol = 't' if position == 'LONG' else 't1'
                entry_spots.append({
                    'pos': (entry_ts, entry_price_offset),
                    'brush': pg.mkBrush(open_color),
                    'symbol': open_symbol,
                    'size': 14,
                    'data': -1
                })

        # Scatter-Daten setzen
        self.trade_entry_scatter.setData(entry_spots)
        self.trade_exit_scatter.setData(exit_spots)

        # Titel aktualisieren
        total_trades = len(trades)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        self.trade_chart_title.setText(
            f"Trades | {total_trades} Trades | {wins}W / {losses}L | Win-Rate: {win_rate:.1f}%"
        )

        # Navigation Label
        if total_trades > 0:
            current = self.current_trade_view + 1 if self.current_trade_view >= 0 else 0
            self.trade_nav_label.setText(f"Trade {current}/{total_trades}")
        else:
            self.trade_nav_label.setText("Trade 0/0")

    def _extract_timestamps(self) -> np.ndarray:
        """Extrahiert Unix-Timestamps aus dem DataFrame."""
        if self.data is None:
            return np.array([])

        # Versuche DateTime aus Index
        try:
            if hasattr(self.data.index, 'to_pydatetime'):
                # DatetimeIndex
                return np.array([dt.timestamp() for dt in self.data.index.to_pydatetime()])
        except:
            pass

        # Versuche DateTime-Spalte
        if 'DateTime' in self.data.columns:
            try:
                dt_col = pd.to_datetime(self.data['DateTime'])
                return np.array([dt.timestamp() for dt in dt_col])
            except:
                pass

        # Fallback: Kuenstliche Timestamps (1 Stunde Abstand)
        return np.arange(len(self.data)) * 3600

    def _on_scatter_clicked(self, scatter, points):
        """pyqtgraph Scatter-Click Handler."""
        if points:
            point = points[0]
            trade_idx = point.data()
            if trade_idx is not None and 0 <= trade_idx < len(self.trades):
                self.current_trade_view = trade_idx
                self._update_trade_detail(self.trades[trade_idx], trade_idx)
                self._update_nav_label()
                self.trade_clicked.emit(trade_idx)

    def _goto_trade(self, trade_idx: int):
        """Zoomt auf einen bestimmten Trade (mit Timestamps)."""
        if not self.trades or trade_idx < 0 or trade_idx >= len(self.trades):
            return
        if self.timestamps is None or len(self.timestamps) == 0:
            return

        self.current_trade_view = trade_idx
        trade = self.trades[trade_idx]

        entry_idx = trade.get('entry_index', 0) - 1
        exit_idx = trade.get('exit_index', 0) - 1

        if entry_idx < 0 or exit_idx < 0:
            return

        # Mit Timestamps arbeiten
        start = max(0, entry_idx - 12)
        end = min(len(self.data) - 1, exit_idx + 12)

        start_ts = self.timestamps[start]
        end_ts = self.timestamps[end]
        self.trade_plot.setXRange(start_ts, end_ts, padding=0.02)

        if start < len(self.data) and end < len(self.data):
            visible_data = self.data.iloc[start:end + 1]
            y_min = visible_data['Low'].min() * 0.998
            y_max = visible_data['High'].max() * 1.002
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
        """Aktualisiert das Trade-Detail-Panel."""
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
            if entry_dt and exit_dt:
                time_diff = exit_dt - entry_dt
                duration = self._format_duration(time_diff)

        pnl_pct = (pnl / entry_price * 100) if entry_price > 0 else 0
        pnl_color = '#33cc33' if pnl >= 0 else '#cc3333'
        pnl_text = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"

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

    def _get_datetime(self, idx: int):
        """Holt DateTime aus dem DataFrame."""
        if self.data is None or idx < 0 or idx >= len(self.data):
            return None
        try:
            dt = self.data.index[idx]
            if hasattr(dt, 'strftime'):
                return dt
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

    def _tab_style(self) -> str:
        """Gibt das Tab-Stylesheet zurueck."""
        return '''
            QTabWidget::pane { border: 1px solid #4d4d4d; background-color: #262626; }
            QTabBar::tab { background-color: #333333; color: #b3b3b3; padding: 8px 20px;
                          margin-right: 2px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background-color: #4da8da; color: white; }
            QTabBar::tab:hover:!selected { background-color: #444444; }
        '''

    def _toolbar_style(self) -> str:
        """Gibt das Toolbar-Stylesheet zurueck."""
        return '''
            QToolBar { background-color: #333333; border: none; spacing: 5px; }
            QToolButton { background-color: #444444; border: none; border-radius: 3px; padding: 5px; color: white; }
            QToolButton:hover { background-color: #555555; }
            QToolButton:checked { background-color: #4da8da; }
        '''

    def _nav_button_style(self) -> str:
        """Gibt das Navigations-Button-Stylesheet zurueck."""
        return '''
            QPushButton { background-color: #3a3a3a; color: #aaa; border: 1px solid #555;
                         border-radius: 3px; padding: 5px 15px; font-size: 11px; }
            QPushButton:hover { background-color: #4a4a4a; color: white; }
            QPushButton:pressed { background-color: #2a2a2a; }
        '''

    def _toggle_button_style(self, active: bool) -> str:
        """Gibt das Toggle-Button-Stylesheet zurueck."""
        if active:
            return '''
                QPushButton { background-color: #4da8da; color: white; border: 1px solid #5bc0eb;
                             border-radius: 3px; padding: 5px 10px; font-size: 11px; font-weight: bold; }
                QPushButton:hover { background-color: #5bc0eb; }
            '''
        else:
            return '''
                QPushButton { background-color: #3a3a3a; color: #aaa; border: 1px solid #555;
                             border-radius: 3px; padding: 5px 10px; font-size: 11px; }
                QPushButton:hover { background-color: #4a4a4a; color: white; }
            '''

    def _toggle_chart_mode(self):
        """Wechselt zwischen Kerzen- und Linienchart."""
        if self.chart_mode == 'candle':
            self.chart_mode = 'line'
            self.btn_chart_mode.setText("Kerzen")
            self.btn_chart_mode.setStyleSheet(self._toggle_button_style(True))
            # Kerzen ausblenden, Linie einblenden
            self.candlestick_item.setVisible(False)
            self.trade_price_line.setVisible(True)
        else:
            self.chart_mode = 'candle'
            self.btn_chart_mode.setText("Linie")
            self.btn_chart_mode.setStyleSheet(self._toggle_button_style(False))
            # Linie ausblenden, Kerzen einblenden
            self.trade_price_line.setVisible(False)
            self.candlestick_item.setVisible(True)

        # Chart neu zeichnen wenn Daten vorhanden
        if self.data is not None and self.timestamps is not None:
            self._refresh_chart_data()

    def _refresh_chart_data(self):
        """Aktualisiert die Chart-Daten basierend auf dem aktuellen Modus."""
        if self.data is None or self.timestamps is None:
            return

        if self.chart_mode == 'candle':
            # Kerzendaten aktualisieren
            candle_data = np.column_stack([
                self.timestamps,
                self.data['Open'].values,
                self.data['High'].values,
                self.data['Low'].values,
                self.data['Close'].values
            ])
            self.candlestick_item.setData(candle_data)
        else:
            # Liniendaten aktualisieren (Close-Preis)
            self.trade_price_line.setData(self.timestamps, self.data['Close'].values)
