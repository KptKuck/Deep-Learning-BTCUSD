"""
ChartWidget - Wiederverwendbares Chart-Widget mit Zoom-Kontrollen
Fuer PrepareDataWindow Tabs
"""

from typing import Optional, List, Dict, Any
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
)
from PyQt6.QtGui import QFont

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class ChartWidget(QWidget):
    """
    Wiederverwendbares Chart-Widget mit Matplotlib und Zoom-Kontrollen.

    Features:
    - Preis-Linie mit optionalen Markern
    - Zoom-Kontrollen (X+/-, Y+/-, Reset, Autozoom)
    - NavigationToolbar fuer Pan/Zoom
    - Dark Theme Styling
    """

    def __init__(self, title: str = 'Chart', parent=None):
        super().__init__(parent)
        self.title = title

        # Daten speichern fuer Zoom-Funktionen
        self._prices: Optional[np.ndarray] = None
        self._signal_indices: List[int] = []

        self._init_ui()

    def _init_ui(self):
        """Initialisiert die UI-Komponenten."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Matplotlib Figure
        self.figure = Figure(figsize=(8, 5), facecolor='#262626')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self._style_axis()

        # Navigation Toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet('''
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
        ''')

        # Zoom-Kontrollen
        zoom_controls = self._create_zoom_controls()

        layout.addWidget(self.toolbar)
        layout.addWidget(zoom_controls)
        layout.addWidget(self.canvas, 1)  # Stretch factor 1

    def _create_zoom_controls(self) -> QWidget:
        """Erstellt die Zoom-Kontroll-Buttons."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setSpacing(10)

        # Autozoom Button
        self.autozoom_btn = QPushButton('Erste 20 Signale')
        self.autozoom_btn.setStyleSheet(self._button_style('#4da8da'))
        self.autozoom_btn.clicked.connect(self._autozoom_signals)
        layout.addWidget(self.autozoom_btn)

        # X-Zoom
        layout.addWidget(QLabel('X:'))

        zoom_x_in = QPushButton('+')
        zoom_x_in.setFixedWidth(30)
        zoom_x_in.setStyleSheet(self._button_style('#555555'))
        zoom_x_in.clicked.connect(lambda: self._zoom_axis('x', 0.8))
        layout.addWidget(zoom_x_in)

        zoom_x_out = QPushButton('-')
        zoom_x_out.setFixedWidth(30)
        zoom_x_out.setStyleSheet(self._button_style('#555555'))
        zoom_x_out.clicked.connect(lambda: self._zoom_axis('x', 1.25))
        layout.addWidget(zoom_x_out)

        # Y-Zoom
        layout.addWidget(QLabel('Y:'))

        zoom_y_in = QPushButton('+')
        zoom_y_in.setFixedWidth(30)
        zoom_y_in.setStyleSheet(self._button_style('#555555'))
        zoom_y_in.clicked.connect(lambda: self._zoom_axis('y', 0.8))
        layout.addWidget(zoom_y_in)

        zoom_y_out = QPushButton('-')
        zoom_y_out.setFixedWidth(30)
        zoom_y_out.setStyleSheet(self._button_style('#555555'))
        zoom_y_out.clicked.connect(lambda: self._zoom_axis('y', 1.25))
        layout.addWidget(zoom_y_out)

        # Reset
        reset_btn = QPushButton('Reset')
        reset_btn.setStyleSheet(self._button_style('#666666'))
        reset_btn.clicked.connect(self._reset_zoom)
        layout.addWidget(reset_btn)

        layout.addStretch()

        return widget

    def _style_axis(self):
        """Stylt die Matplotlib-Achse im Dark Theme."""
        self.ax.set_facecolor('#1a1a1a')
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')
        for spine in self.ax.spines.values():
            spine.set_color('#444444')

    def _button_style(self, hex_color: str) -> str:
        """Generiert Button-Stylesheet."""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

        r_h, g_h, b_h = [min(255, int(c * 1.2)) for c in (r, g, b)]
        r_p, g_p, b_p = [int(c * 0.8) for c in (r, g, b)]

        return f'''
            QPushButton {{
                background-color: rgb({r}, {g}, {b});
                color: white;
                border: none;
                border-radius: 4px;
                padding: 5px 10px;
            }}
            QPushButton:hover {{
                background-color: rgb({r_h}, {g_h}, {b_h});
            }}
            QPushButton:pressed {{
                background-color: rgb({r_p}, {g_p}, {b_p});
            }}
        '''

    def update_price_chart(self, prices: np.ndarray,
                           buy_indices: List[int] = None,
                           sell_indices: List[int] = None,
                           title: str = None):
        """
        Aktualisiert den Chart mit Preis-Daten und optionalen Markern.

        Args:
            prices: Preis-Array
            buy_indices: Indizes fuer BUY-Marker (gruene ^)
            sell_indices: Indizes fuer SELL-Marker (rote v)
            title: Optionaler Titel
        """
        self._prices = prices
        self._signal_indices = []

        if buy_indices:
            self._signal_indices.extend(buy_indices)
        if sell_indices:
            self._signal_indices.extend(sell_indices)
        self._signal_indices = sorted(self._signal_indices)

        # Chart leeren und neu zeichnen
        self.ax.clear()
        self._style_axis()

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

        # Titel
        chart_title = title or self.title
        if buy_indices or sell_indices:
            num_buy = len(buy_indices) if buy_indices else 0
            num_sell = len(sell_indices) if sell_indices else 0
            chart_title = f'{chart_title}: {num_buy} BUY, {num_sell} SELL'

        self.ax.set_title(chart_title, color='white', fontsize=12)

        if buy_indices or sell_indices:
            self.ax.legend(loc='upper left', facecolor='#333333',
                          edgecolor='#555555', labelcolor='white')

        # Initial-Zoom auf erste 20 Signale
        if self._signal_indices:
            self._autozoom_signals()
        else:
            self.figure.tight_layout()
            self.canvas.draw()

    def update_labels_chart(self, prices: np.ndarray, labels: np.ndarray,
                            num_classes: int = 3, title: str = None):
        """
        Aktualisiert den Chart mit Label-Visualisierung.

        Args:
            prices: Preis-Array
            labels: Label-Array (0=HOLD/BUY, 1=BUY/SELL, 2=SELL bei 3 Klassen)
            num_classes: Anzahl Klassen (2 oder 3)
            title: Optionaler Titel
        """
        self._prices = prices

        self.ax.clear()
        self._style_axis()

        # Preis-Linie
        self.ax.plot(prices, color='white', linewidth=0.5, alpha=0.8)

        if num_classes == 2:
            # 2 Klassen: BUY=0, SELL=1
            buy_idx = np.where(labels == 0)[0]
            sell_idx = np.where(labels == 1)[0]
        else:
            # 3 Klassen: HOLD=0, BUY=1, SELL=2
            buy_idx = np.where(labels == 1)[0]
            sell_idx = np.where(labels == 2)[0]
            hold_idx = np.where(labels == 0)[0]

            # HOLD Marker (klein, grau)
            if len(hold_idx) > 0:
                # Nur jeden 10. HOLD-Punkt anzeigen (Performance)
                hold_sample = hold_idx[::10]
                self.ax.scatter(hold_sample, prices[hold_sample],
                               marker='.', color='#666666', s=10, alpha=0.3,
                               label=f'HOLD ({len(hold_idx)})')

        # BUY Marker
        if len(buy_idx) > 0:
            self.ax.scatter(buy_idx, prices[buy_idx],
                           marker='^', color='#33cc33', s=50, zorder=5,
                           label=f'BUY ({len(buy_idx)})')

        # SELL Marker
        if len(sell_idx) > 0:
            self.ax.scatter(sell_idx, prices[sell_idx],
                           marker='v', color='#cc3333', s=50, zorder=5,
                           label=f'SELL ({len(sell_idx)})')

        # Signal-Indizes fuer Zoom
        self._signal_indices = sorted(list(buy_idx) + list(sell_idx))

        # Titel
        chart_title = title or 'Labels'
        self.ax.set_title(chart_title, color='white', fontsize=12)
        self.ax.legend(loc='upper left', facecolor='#333333',
                      edgecolor='#555555', labelcolor='white')

        if self._signal_indices:
            self._autozoom_signals()
        else:
            self.figure.tight_layout()
            self.canvas.draw()

    def update_features_chart(self, features: Dict[str, np.ndarray], title: str = None):
        """
        Aktualisiert den Chart mit Feature-Vorschau (normalisiert).

        Args:
            features: Dict mit Feature-Name -> Feature-Array
            title: Optionaler Titel
        """
        self.ax.clear()
        self._style_axis()

        # Farben fuer verschiedene Features
        colors = ['#4da8da', '#33cc33', '#cc3333', '#e6b333',
                  '#b19cd9', '#80cbc4', '#ff9966', '#66ccff']

        for i, (name, values) in enumerate(features.items()):
            # Z-Score Normalisierung fuer Vergleichbarkeit
            if np.std(values) > 0:
                normalized = (values - np.mean(values)) / np.std(values)
            else:
                normalized = values

            color = colors[i % len(colors)]
            self.ax.plot(normalized, color=color, linewidth=0.8,
                        alpha=0.7, label=name)

        chart_title = title or f'Features ({len(features)} ausgewaehlt)'
        self.ax.set_title(chart_title, color='white', fontsize=12)
        self.ax.legend(loc='upper left', facecolor='#333333',
                      edgecolor='#555555', labelcolor='white', fontsize=8)

        self.figure.tight_layout()
        self.canvas.draw()

    def update_samples_chart(self, num_buy: int, num_sell: int,
                             num_hold: int = 0, title: str = None):
        """
        Aktualisiert den Chart mit Sample-Verteilung (Balkendiagramm).

        Args:
            num_buy: Anzahl BUY-Samples
            num_sell: Anzahl SELL-Samples
            num_hold: Anzahl HOLD-Samples
            title: Optionaler Titel
        """
        self.ax.clear()
        self._style_axis()

        if num_hold > 0:
            labels = ['BUY', 'SELL', 'HOLD']
            values = [num_buy, num_sell, num_hold]
            colors = ['#33cc33', '#cc3333', '#666666']
        else:
            labels = ['BUY', 'SELL']
            values = [num_buy, num_sell]
            colors = ['#33cc33', '#cc3333']

        bars = self.ax.bar(labels, values, color=colors, edgecolor='white', linewidth=1)

        # Werte auf Balken anzeigen
        for bar, val in zip(bars, values):
            self.ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        str(val), ha='center', va='bottom', color='white',
                        fontsize=12, fontweight='bold')

        total = sum(values)
        chart_title = title or f'Sample-Verteilung (Gesamt: {total})'
        self.ax.set_title(chart_title, color='white', fontsize=12)
        self.ax.set_ylabel('Anzahl', color='white')

        self.figure.tight_layout()
        self.canvas.draw()

    def _zoom_axis(self, axis: str, factor: float):
        """Zoomt eine einzelne Achse."""
        if axis == 'x':
            xlim = self.ax.get_xlim()
            center = (xlim[0] + xlim[1]) / 2
            width = (xlim[1] - xlim[0]) * factor
            self.ax.set_xlim(center - width/2, center + width/2)
        else:
            ylim = self.ax.get_ylim()
            center = (ylim[0] + ylim[1]) / 2
            height = (ylim[1] - ylim[0]) * factor
            self.ax.set_ylim(center - height/2, center + height/2)
        self.canvas.draw()

    def _reset_zoom(self):
        """Setzt den Zoom zurueck."""
        self.ax.autoscale()
        self.canvas.draw()

    def _autozoom_signals(self):
        """Zoomt auf die ersten 20 Signale."""
        if not self._signal_indices or self._prices is None:
            self.figure.tight_layout()
            self.canvas.draw()
            return

        # Zeige die ersten 20 Signale
        num_signals = min(20, len(self._signal_indices))
        end_idx = self._signal_indices[num_signals - 1]

        # Puffer hinzufuegen
        start_idx = max(0, self._signal_indices[0] - 50)
        end_idx = min(len(self._prices), end_idx + 50)

        self.ax.set_xlim(start_idx, end_idx)

        # Y-Limits an sichtbaren Bereich anpassen
        visible_prices = self._prices[start_idx:end_idx]
        if len(visible_prices) > 0:
            y_min, y_max = visible_prices.min(), visible_prices.max()
            y_margin = (y_max - y_min) * 0.05
            self.ax.set_ylim(y_min - y_margin, y_max + y_margin)

        self.figure.tight_layout()
        self.canvas.draw()

    def clear(self):
        """Leert den Chart."""
        self.ax.clear()
        self._style_axis()
        self.ax.set_title(self.title, color='white', fontsize=12)
        self.canvas.draw()
