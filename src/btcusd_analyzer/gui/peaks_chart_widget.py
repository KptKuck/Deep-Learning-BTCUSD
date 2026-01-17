"""
PeaksChartWidget - PyQtChart-basiertes Chart-Widget fuer Peak-Visualisierung
Ersetzt Matplotlib fuer den Find Peaks Tab
"""

from typing import Optional, List
import numpy as np

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import Qt, QPointF, QMargins
from PyQt6.QtGui import QColor, QPen, QBrush, QPainter, QWheelEvent, QMouseEvent
from PyQt6.QtCharts import (
    QChart, QChartView, QLineSeries, QScatterSeries, QValueAxis
)

from .styles import StyleFactory, COLORS


class InteractiveChartView(QChartView):
    """
    Erweiterter QChartView mit Mausrad-Zoom und Panning.

    Features:
    - Mausrad: Zoom auf beiden Achsen (Shift = nur X, Ctrl = nur Y)
    - Linke Maustaste + Drag: Panning (Verschieben)
    - Rechte Maustaste + Drag: Rechteck-Zoom
    """

    def __init__(self, chart: QChart, parent=None):
        super().__init__(chart, parent)
        self._last_mouse_pos: Optional[QPointF] = None
        self._panning = False

    def wheelEvent(self, event: QWheelEvent):
        """Mausrad-Zoom mit Modifiern."""
        # Zoom-Faktor berechnen
        delta = event.angleDelta().y()
        factor = 0.9 if delta > 0 else 1.1  # Zoom in/out

        # Modifier pruefen
        modifiers = event.modifiers()
        zoom_x = True
        zoom_y = True

        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            zoom_y = False  # Nur X-Achse
        elif modifiers & Qt.KeyboardModifier.ControlModifier:
            zoom_x = False  # Nur Y-Achse

        # Zoom um Mausposition
        chart = self.chart()
        if chart:
            # Mausposition in Chart-Koordinaten
            mouse_pos = event.position()
            chart_pos = chart.mapToValue(mouse_pos)

            for axis in chart.axes():
                if isinstance(axis, QValueAxis):
                    is_x_axis = axis.alignment() == Qt.AlignmentFlag.AlignBottom
                    if (is_x_axis and zoom_x) or (not is_x_axis and zoom_y):
                        # Zoom um den Punkt unter der Maus
                        axis_min = axis.min()
                        axis_max = axis.max()
                        center = chart_pos.x() if is_x_axis else chart_pos.y()

                        # Neuen Bereich berechnen
                        left = center - (center - axis_min) * factor
                        right = center + (axis_max - center) * factor
                        axis.setRange(left, right)

        event.accept()

    def mousePressEvent(self, event: QMouseEvent):
        """Startet Panning bei linker Maustaste."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._panning = True
            self._last_mouse_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        elif event.button() == Qt.MouseButton.RightButton:
            # Rechteck-Zoom mit rechter Maustaste
            super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Panning bei gedruckter linker Maustaste."""
        if self._panning and self._last_mouse_pos is not None:
            chart = self.chart()
            if chart:
                # Delta berechnen
                current_pos = event.position()
                delta = current_pos - self._last_mouse_pos

                # In Chart-Koordinaten umrechnen
                for axis in chart.axes():
                    if isinstance(axis, QValueAxis):
                        is_x_axis = axis.alignment() == Qt.AlignmentFlag.AlignBottom
                        axis_range = axis.max() - axis.min()

                        # Pixel zu Wert-Verhaeltnis
                        if is_x_axis:
                            plot_area = chart.plotArea()
                            pixels = plot_area.width()
                            value_delta = -delta.x() * axis_range / pixels
                        else:
                            plot_area = chart.plotArea()
                            pixels = plot_area.height()
                            value_delta = delta.y() * axis_range / pixels

                        axis.setRange(axis.min() + value_delta, axis.max() + value_delta)

                self._last_mouse_pos = current_pos
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Beendet Panning."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._panning = False
            self._last_mouse_pos = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class PeaksChartWidget(QWidget):
    """
    PyQtChart-basiertes Chart-Widget fuer Peak-Visualisierung.

    Features:
    - Preis-Linie (weiss)
    - High-Peaks (rot, Raute)
    - Low-Peaks (gruen, Dreieck)
    - Zoom-Kontrollen (Erste 20 Peaks, X+/-, Y+/-, Reset)
    - Dark Theme
    - Rubber Band Zoom (Maus-Selektion)
    """

    def __init__(self, title: str = 'Peaks', parent=None):
        super().__init__(parent)
        self.title = title

        # Daten speichern fuer Zoom-Funktionen
        self._prices: Optional[np.ndarray] = None
        self._signal_indices: List[int] = []

        # Original-Achsenbereiche fuer Reset
        self._original_x_range: tuple = (0, 100)
        self._original_y_range: tuple = (0, 100)

        self._init_ui()

    def _init_ui(self):
        """Initialisiert die UI-Komponenten."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Chart erstellen
        self.chart = QChart()
        self.chart.setBackgroundBrush(QBrush(QColor(COLORS['bg_primary'])))
        self.chart.setTitleBrush(QBrush(QColor('white')))
        self.chart.setTitle(self.title)
        self.chart.legend().setLabelColor(QColor('white'))
        self.chart.legend().setBrush(QBrush(QColor('#333333')))
        self.chart.setMargins(QMargins(10, 10, 10, 10))

        # Serien erstellen
        self._create_series()

        # Achsen erstellen
        self._create_axes()

        # ChartView mit interaktivem Zoom und Panning
        self.chart_view = InteractiveChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.chart_view.setBackgroundBrush(QBrush(QColor(COLORS['bg_primary'])))

        # Zoom-Kontrollen
        zoom_controls = self._create_zoom_controls()

        layout.addWidget(zoom_controls)
        layout.addWidget(self.chart_view, 1)

    def _create_series(self):
        """Erstellt die Daten-Serien."""
        # Preis-Linie
        self.price_series = QLineSeries()
        self.price_series.setName('Preis')
        pen = QPen(QColor('white'))
        pen.setWidthF(0.8)
        self.price_series.setPen(pen)
        self.chart.addSeries(self.price_series)

        # Low-Peaks (gruen, Dreieck nach oben)
        self.low_peaks_series = QScatterSeries()
        self.low_peaks_series.setName('Low-Peaks')
        self.low_peaks_series.setMarkerShape(QScatterSeries.MarkerShape.MarkerShapeTriangle)
        self.low_peaks_series.setMarkerSize(12)
        self.low_peaks_series.setColor(QColor(COLORS['success']))
        self.low_peaks_series.setBorderColor(QColor(COLORS['success']))
        self.chart.addSeries(self.low_peaks_series)

        # High-Peaks (rot, Raute - da gedrehtes Dreieck nicht moeglich)
        self.high_peaks_series = QScatterSeries()
        self.high_peaks_series.setName('High-Peaks')
        self.high_peaks_series.setMarkerShape(QScatterSeries.MarkerShape.MarkerShapeRotatedRectangle)
        self.high_peaks_series.setMarkerSize(12)
        self.high_peaks_series.setColor(QColor(COLORS['error']))
        self.high_peaks_series.setBorderColor(QColor(COLORS['error']))
        self.chart.addSeries(self.high_peaks_series)

    def _create_axes(self):
        """Erstellt und stylt die Achsen."""
        # X-Achse (Index)
        self.x_axis = QValueAxis()
        self.x_axis.setTitleText('Index')
        self.x_axis.setTitleBrush(QBrush(QColor('white')))
        self.x_axis.setLabelsBrush(QBrush(QColor('white')))
        self.x_axis.setGridLineColor(QColor('#444444'))
        self.x_axis.setLinePen(QPen(QColor('#444444')))
        self.chart.addAxis(self.x_axis, Qt.AlignmentFlag.AlignBottom)

        # Y-Achse (Preis)
        self.y_axis = QValueAxis()
        self.y_axis.setTitleText('Preis')
        self.y_axis.setTitleBrush(QBrush(QColor('white')))
        self.y_axis.setLabelsBrush(QBrush(QColor('white')))
        self.y_axis.setGridLineColor(QColor('#444444'))
        self.y_axis.setLinePen(QPen(QColor('#444444')))
        self.chart.addAxis(self.y_axis, Qt.AlignmentFlag.AlignLeft)

        # Serien an Achsen binden
        self.price_series.attachAxis(self.x_axis)
        self.price_series.attachAxis(self.y_axis)
        self.low_peaks_series.attachAxis(self.x_axis)
        self.low_peaks_series.attachAxis(self.y_axis)
        self.high_peaks_series.attachAxis(self.x_axis)
        self.high_peaks_series.attachAxis(self.y_axis)

    def _create_zoom_controls(self) -> QWidget:
        """Erstellt die Zoom-Kontroll-Buttons."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setSpacing(10)

        # Autozoom Button
        self.autozoom_btn = QPushButton('Erste 20 Peaks')
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

        # Pan-Buttons
        layout.addWidget(QLabel('Pan:'))

        pan_left = QPushButton('<')
        pan_left.setFixedWidth(30)
        pan_left.setStyleSheet(self._button_style('#555555'))
        pan_left.clicked.connect(lambda: self._pan_axis('x', -0.2))
        layout.addWidget(pan_left)

        pan_right = QPushButton('>')
        pan_right.setFixedWidth(30)
        pan_right.setStyleSheet(self._button_style('#555555'))
        pan_right.clicked.connect(lambda: self._pan_axis('x', 0.2))
        layout.addWidget(pan_right)

        # Reset
        reset_btn = QPushButton('Reset')
        reset_btn.setStyleSheet(self._button_style('#666666'))
        reset_btn.clicked.connect(self._reset_zoom)
        layout.addWidget(reset_btn)

        # Hinweis
        hint_label = QLabel('Mausrad=Zoom, Drag=Pan')
        hint_label.setStyleSheet('color: #888888; font-size: 10px;')
        layout.addWidget(hint_label)

        layout.addStretch()

        return widget

    def _button_style(self, hex_color: str) -> str:
        """Generiert Button-Stylesheet."""
        return StyleFactory.button_style_hex(hex_color, padding='5px 10px')

    def update_peaks_chart(self, prices: np.ndarray,
                           low_indices: List[int] = None,
                           high_indices: List[int] = None,
                           title: str = None):
        """
        Aktualisiert den Chart mit Preis-Daten und Peaks.

        Args:
            prices: Preis-Array
            low_indices: Indizes fuer Low-Peaks (gruene Dreiecke)
            high_indices: Indizes fuer High-Peaks (rote Rauten)
            title: Optionaler Titel
        """
        self._prices = prices
        self._signal_indices = []

        if low_indices:
            self._signal_indices.extend(low_indices)
        if high_indices:
            self._signal_indices.extend(high_indices)
        self._signal_indices = sorted(self._signal_indices)

        # Preis-Linie mit replace() - viel schneller als einzelne append()-Aufrufe
        price_points = [QPointF(float(i), float(price)) for i, price in enumerate(prices)]
        self.price_series.replace(price_points)

        # Low-Peaks mit replace()
        low_points = []
        if low_indices:
            low_points = [
                QPointF(float(idx), float(prices[idx]))
                for idx in low_indices if 0 <= idx < len(prices)
            ]
        self.low_peaks_series.replace(low_points)
        num_low = len(low_points)

        # High-Peaks mit replace()
        high_points = []
        if high_indices:
            high_points = [
                QPointF(float(idx), float(prices[idx]))
                for idx in high_indices if 0 <= idx < len(prices)
            ]
        self.high_peaks_series.replace(high_points)
        num_high = len(high_points)

        # Legende aktualisieren
        self.low_peaks_series.setName(f'Low-Peaks ({num_low})')
        self.high_peaks_series.setName(f'High-Peaks ({num_high})')

        # Titel setzen
        chart_title = title or self.title
        if low_indices or high_indices:
            chart_title = f'{chart_title}: {num_low} Low, {num_high} High'
        self.chart.setTitle(chart_title)

        # Achsen-Bereiche setzen
        x_min, x_max = 0, len(prices) - 1
        y_min, y_max = float(np.min(prices)), float(np.max(prices))
        y_margin = (y_max - y_min) * 0.05

        self._original_x_range = (x_min, x_max)
        self._original_y_range = (y_min - y_margin, y_max + y_margin)

        self.x_axis.setRange(x_min, x_max)
        self.y_axis.setRange(y_min - y_margin, y_max + y_margin)

        # Initial-Zoom auf erste 20 Peaks
        if self._signal_indices:
            self._autozoom_signals()

    def _zoom_axis(self, axis: str, factor: float):
        """Zoomt eine einzelne Achse."""
        if axis == 'x':
            x_min = self.x_axis.min()
            x_max = self.x_axis.max()
            center = (x_min + x_max) / 2
            width = (x_max - x_min) * factor
            self.x_axis.setRange(center - width / 2, center + width / 2)
        else:
            y_min = self.y_axis.min()
            y_max = self.y_axis.max()
            center = (y_min + y_max) / 2
            height = (y_max - y_min) * factor
            self.y_axis.setRange(center - height / 2, center + height / 2)

    def _pan_axis(self, axis: str, fraction: float):
        """Verschiebt eine Achse um einen Bruchteil des sichtbaren Bereichs."""
        if axis == 'x':
            x_min = self.x_axis.min()
            x_max = self.x_axis.max()
            delta = (x_max - x_min) * fraction
            self.x_axis.setRange(x_min + delta, x_max + delta)
        else:
            y_min = self.y_axis.min()
            y_max = self.y_axis.max()
            delta = (y_max - y_min) * fraction
            self.y_axis.setRange(y_min + delta, y_max + delta)

    def _reset_zoom(self):
        """Setzt den Zoom zurueck auf Original-Bereiche."""
        self.x_axis.setRange(*self._original_x_range)
        self.y_axis.setRange(*self._original_y_range)

    def _autozoom_signals(self):
        """Zoomt auf die ersten 20 Peaks."""
        if not self._signal_indices or self._prices is None:
            return

        # Zeige die ersten 20 Peaks
        num_signals = min(20, len(self._signal_indices))
        end_idx = self._signal_indices[num_signals - 1]

        # Puffer hinzufuegen
        start_idx = max(0, self._signal_indices[0] - 50)
        end_idx = min(len(self._prices) - 1, end_idx + 50)

        self.x_axis.setRange(start_idx, end_idx)

        # Y-Limits an sichtbaren Bereich anpassen
        visible_prices = self._prices[int(start_idx):int(end_idx) + 1]
        if len(visible_prices) > 0:
            y_min = float(np.min(visible_prices))
            y_max = float(np.max(visible_prices))
            y_margin = (y_max - y_min) * 0.05
            self.y_axis.setRange(y_min - y_margin, y_max + y_margin)

    def clear(self):
        """Leert den Chart."""
        self.price_series.clear()
        self.low_peaks_series.clear()
        self.high_peaks_series.clear()
        self.chart.setTitle(self.title)
        self._prices = None
        self._signal_indices = []
