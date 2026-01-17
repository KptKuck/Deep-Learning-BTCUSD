"""
FeaturesChartWidget - PyQtChart-basiertes Chart-Widget fuer Feature-Visualisierung
Ersetzt Matplotlib fuer den Features Tab mit interaktivem Zoom, Pan und Crosshair
"""

from typing import Optional, Dict, List
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QCheckBox, QScrollArea, QFrame
)
from PyQt6.QtCore import Qt, QPointF, QMargins, pyqtSignal
from PyQt6.QtGui import QColor, QPen, QBrush, QPainter, QWheelEvent, QMouseEvent

from PyQt6.QtCharts import (
    QChart, QChartView, QLineSeries, QValueAxis
)

from .styles import StyleFactory, COLORS
from ..core.logger import get_logger


class InteractiveChartView(QChartView):
    """
    Erweiterter QChartView mit Mausrad-Zoom, Panning und Crosshair.

    Features:
    - Mausrad: Zoom auf beiden Achsen (Shift = nur X, Ctrl = nur Y)
    - Linke Maustaste + Drag: Panning (Verschieben)
    - Crosshair bei Mausbewegung
    """

    # Signal mit X-Index bei Mausbewegung
    crosshairMoved = pyqtSignal(int)

    def __init__(self, chart: QChart, parent=None):
        super().__init__(chart, parent)
        self._last_mouse_pos: Optional[QPointF] = None
        self._panning = False
        self._crosshair_x: Optional[float] = None
        self._crosshair_enabled = True
        self.setMouseTracking(True)

    def wheelEvent(self, event: QWheelEvent):
        """Mausrad-Zoom mit Modifiern."""
        delta = event.angleDelta().y()
        factor = 0.9 if delta > 0 else 1.1

        modifiers = event.modifiers()
        zoom_x = True
        zoom_y = True

        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            zoom_y = False
        elif modifiers & Qt.KeyboardModifier.ControlModifier:
            zoom_x = False

        chart = self.chart()
        if chart:
            mouse_pos = event.position()
            chart_pos = chart.mapToValue(mouse_pos)

            for axis in chart.axes():
                if isinstance(axis, QValueAxis):
                    is_x_axis = axis.alignment() == Qt.AlignmentFlag.AlignBottom
                    if (is_x_axis and zoom_x) or (not is_x_axis and zoom_y):
                        axis_min = axis.min()
                        axis_max = axis.max()
                        center = chart_pos.x() if is_x_axis else chart_pos.y()

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
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Panning bei gedruckter linker Maustaste, sonst Crosshair."""
        chart = self.chart()
        if not chart:
            super().mouseMoveEvent(event)
            return

        current_pos = event.position()

        if self._panning and self._last_mouse_pos is not None:
            delta = current_pos - self._last_mouse_pos

            for axis in chart.axes():
                if isinstance(axis, QValueAxis):
                    is_x_axis = axis.alignment() == Qt.AlignmentFlag.AlignBottom
                    axis_range = axis.max() - axis.min()

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
            # Crosshair-Position aktualisieren
            if self._crosshair_enabled:
                chart_pos = chart.mapToValue(current_pos)
                x_index = int(round(chart_pos.x()))
                self._crosshair_x = chart_pos.x()
                self.crosshairMoved.emit(x_index)
                self.viewport().update()  # Neuzeichnen fuer Crosshair

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

    def set_crosshair_position(self, x_value: float):
        """Setzt Crosshair-Position extern (fuer Synchronisation)."""
        self._crosshair_x = x_value
        self.viewport().update()

    def set_crosshair_enabled(self, enabled: bool):
        """Aktiviert/Deaktiviert Crosshair."""
        self._crosshair_enabled = enabled
        if not enabled:
            self._crosshair_x = None
        self.viewport().update()

    def drawForeground(self, painter: QPainter, rect):
        """Zeichnet Crosshair im Vordergrund."""
        super().drawForeground(painter, rect)

        if self._crosshair_x is None or not self._crosshair_enabled:
            return

        chart = self.chart()
        if not chart:
            return

        plot_area = chart.plotArea()

        # X-Position in Pixel umrechnen
        x_pixel = chart.mapToPosition(QPointF(self._crosshair_x, 0)).x()

        # Nur zeichnen wenn im sichtbaren Bereich
        if plot_area.left() <= x_pixel <= plot_area.right():
            pen = QPen(QColor('#ffff00'))  # Gelb
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setWidthF(1.0)
            painter.setPen(pen)
            painter.drawLine(
                int(x_pixel), int(plot_area.top()),
                int(x_pixel), int(plot_area.bottom())
            )


class FeaturesChartWidget(QWidget):
    """
    PyQtChart-basiertes Chart-Widget fuer Feature-Visualisierung.

    Features:
    - Dynamische Serien fuer beliebige Features
    - Z-Score Normalisierung
    - Zoom/Pan via InteractiveChartView
    - Checkbox-Steuerung fuer Feature-Visibility
    - Synchronisiertes Crosshair
    - Dark Theme
    """

    # Signal mit X-Index bei Mausbewegung (weitergeleitet von InteractiveChartView)
    crosshairMoved = pyqtSignal(int)

    # Farbpalette fuer Features (erweiterbar)
    FEATURE_COLORS = [
        '#4da8da',  # Hellblau
        '#ff6b6b',  # Rot
        '#98d8aa',  # Gruen
        '#ffd93d',  # Gelb
        '#c9b1ff',  # Violett
        '#ff9f43',  # Orange
        '#74b9ff',  # Blau
        '#fd79a8',  # Pink
        '#00b894',  # Tuerkis
        '#fdcb6e',  # Gold
        '#e17055',  # Koralle
        '#0984e3',  # Dunkelblau
        '#00cec9',  # Cyan
        '#6c5ce7',  # Lila
        '#fab1a0',  # Pfirsich
        '#81ecec',  # Hellcyan
        '#dfe6e9',  # Grau
        '#ffeaa7',  # Hellgelb
        '#55efc4',  # Mint
        '#a29bfe',  # Lavendel
    ]

    def __init__(self, title: str = 'Features', parent=None):
        super().__init__(parent)
        self.title = title
        self._logger = get_logger()

        # Dynamische Verwaltung
        self._series_map: Dict[str, QLineSeries] = {}
        self._visibility_map: Dict[str, bool] = {}
        self._color_map: Dict[str, QColor] = {}
        self._checkbox_map: Dict[str, QCheckBox] = {}

        # Daten speichern
        self._data_length: int = 0
        self._feature_names: List[str] = []

        # Original-Achsenbereiche fuer Reset
        self._original_x_range: tuple = (0, 100)
        self._original_y_range: tuple = (-3, 3)  # Z-Score Bereich

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
        self.chart.legend().hide()  # Legende ausblenden - Checkboxen uebernehmen
        self.chart.setMargins(QMargins(10, 10, 10, 10))

        # Achsen erstellen
        self._create_axes()

        # ChartView mit interaktivem Zoom, Panning und Crosshair
        self.chart_view = InteractiveChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.chart_view.setBackgroundBrush(QBrush(QColor(COLORS['bg_primary'])))

        # Crosshair-Signal weiterleiten
        self.chart_view.crosshairMoved.connect(self.crosshairMoved.emit)

        # Zoom-Kontrollen
        zoom_controls = self._create_zoom_controls()

        # Checkbox-Container fuer Feature-Visibility
        self.checkbox_container = self._create_checkbox_container()

        layout.addWidget(zoom_controls)
        layout.addWidget(self.chart_view, 1)
        layout.addWidget(self.checkbox_container)

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

        # Y-Achse (Z-Score)
        self.y_axis = QValueAxis()
        self.y_axis.setTitleText('Z-Score')
        self.y_axis.setTitleBrush(QBrush(QColor('white')))
        self.y_axis.setLabelsBrush(QBrush(QColor('white')))
        self.y_axis.setGridLineColor(QColor('#444444'))
        self.y_axis.setLinePen(QPen(QColor('#444444')))
        self.chart.addAxis(self.y_axis, Qt.AlignmentFlag.AlignLeft)

    def _create_zoom_controls(self) -> QWidget:
        """Erstellt die Zoom-Kontroll-Buttons."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setSpacing(10)

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

    def _create_checkbox_container(self) -> QWidget:
        """Erstellt den Container fuer Feature-Checkboxen."""
        container = QFrame()
        container.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_secondary']};
                border-radius: 4px;
                padding: 4px;
            }}
        """)

        # ScrollArea fuer viele Checkboxen
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setMaximumHeight(40)
        scroll.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
        """)

        # Inneres Widget fuer Checkboxen
        self.checkbox_widget = QWidget()
        self.checkbox_layout = QHBoxLayout(self.checkbox_widget)
        self.checkbox_layout.setContentsMargins(4, 2, 4, 2)
        self.checkbox_layout.setSpacing(8)
        self.checkbox_layout.addStretch()

        scroll.setWidget(self.checkbox_widget)

        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

        return container

    def _button_style(self, hex_color: str) -> str:
        """Generiert Button-Stylesheet."""
        return StyleFactory.button_style_hex(hex_color, padding='5px 10px')

    def _get_color_for_feature(self, feature_name: str, index: int) -> QColor:
        """
        Gibt konsistente Farbe fuer ein Feature zurueck.
        Gleiche Feature-Namen bekommen immer die gleiche Farbe.
        """
        if feature_name in self._color_map:
            return self._color_map[feature_name]

        # Neue Farbe zuweisen
        color_idx = index % len(self.FEATURE_COLORS)
        color = QColor(self.FEATURE_COLORS[color_idx])
        self._color_map[feature_name] = color
        return color

    def _normalize_zscore(self, data: np.ndarray) -> np.ndarray:
        """Z-Score Normalisierung."""
        mean = np.nanmean(data)
        std = np.nanstd(data)
        if std == 0 or np.isnan(std):
            return np.zeros_like(data)
        return (data - mean) / std

    def update_features_chart(self, features_dict: Dict[str, np.ndarray],
                               title: str = None):
        """
        Aktualisiert den Chart mit Feature-Daten.

        Dynamisch:
        1. Entferne Serien fuer Features die nicht mehr in features_dict sind
        2. Erstelle neue Serien fuer neue Features
        3. Update Daten fuer bestehende Features
        4. Regeneriere Checkboxen

        Args:
            features_dict: Dict mit Feature-Name -> Daten-Array
            title: Optionaler Titel
        """
        if not features_dict:
            self.clear()
            return

        current_features = set(features_dict.keys())
        existing_features = set(self._series_map.keys())

        # 1. Entferne alte Serien
        features_to_remove = existing_features - current_features
        for feature_name in features_to_remove:
            series = self._series_map.pop(feature_name)
            self.chart.removeSeries(series)
            self._visibility_map.pop(feature_name, None)

        # 2. Erstelle neue Serien und update bestehende
        self._feature_names = list(features_dict.keys())
        self._data_length = 0

        y_min, y_max = float('inf'), float('-inf')

        for idx, (feature_name, data) in enumerate(features_dict.items()):
            # Z-Score normalisieren
            normalized = self._normalize_zscore(data)
            self._data_length = max(self._data_length, len(normalized))

            # Y-Bereich aktualisieren
            valid_data = normalized[~np.isnan(normalized)]
            if len(valid_data) > 0:
                y_min = min(y_min, float(np.min(valid_data)))
                y_max = max(y_max, float(np.max(valid_data)))

            # Serie erstellen oder holen
            if feature_name not in self._series_map:
                series = QLineSeries()
                series.setName(feature_name)
                color = self._get_color_for_feature(feature_name, idx)
                pen = QPen(color)
                pen.setWidthF(1.5)  # Dickere Linien fuer bessere Sichtbarkeit
                series.setPen(pen)

                self.chart.addSeries(series)
                series.attachAxis(self.x_axis)
                series.attachAxis(self.y_axis)

                self._series_map[feature_name] = series
                self._visibility_map[feature_name] = True

                self._logger.debug(f"[FeaturesChart] Serie '{feature_name}' erstellt, Farbe: {color.name()}")
            else:
                series = self._series_map[feature_name]

            # Daten setzen (mit replace() fuer Performance)
            points = [
                QPointF(float(i), float(val) if not np.isnan(val) else 0.0)
                for i, val in enumerate(normalized)
            ]
            series.replace(points)

            # Sichtbarkeit anwenden
            series.setVisible(self._visibility_map.get(feature_name, True))

        # Checkboxen neu erstellen
        self._rebuild_checkboxes()

        # Titel setzen
        chart_title = title or self.title
        self.chart.setTitle(f'{chart_title} ({len(features_dict)} Features)')

        # Achsen-Bereiche setzen
        x_min, x_max = 0, max(1, self._data_length - 1)

        # Y-Bereich mit Margin
        if y_min == float('inf'):
            y_min, y_max = -3, 3
        y_margin = (y_max - y_min) * 0.1
        y_min -= y_margin
        y_max += y_margin

        self._original_x_range = (x_min, x_max)
        self._original_y_range = (y_min, y_max)

        self.x_axis.setRange(x_min, x_max)
        self.y_axis.setRange(y_min, y_max)

        self._logger.debug(f"[FeaturesChart] Update: {len(features_dict)} Features, "
                          f"X:{x_min}-{x_max}, Y:{y_min:.2f}-{y_max:.2f}")

    def _rebuild_checkboxes(self):
        """Erstellt Checkboxen dynamisch basierend auf aktuellen Features."""
        # Alte Checkboxen entfernen
        for cb in self._checkbox_map.values():
            self.checkbox_layout.removeWidget(cb)
            cb.deleteLater()
        self._checkbox_map.clear()

        # Stretch am Ende entfernen
        while self.checkbox_layout.count() > 0:
            item = self.checkbox_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Neue Checkboxen erstellen
        for feature_name in self._feature_names:
            color = self._color_map.get(feature_name, QColor('#ffffff'))
            cb = QCheckBox(feature_name)
            cb.setChecked(self._visibility_map.get(feature_name, True))

            # Farbcodierte Checkbox
            cb.setStyleSheet(f"""
                QCheckBox {{
                    color: {color.name()};
                    font-size: 11px;
                    spacing: 4px;
                }}
                QCheckBox::indicator {{
                    width: 14px;
                    height: 14px;
                    border: 1px solid {color.name()};
                    border-radius: 2px;
                }}
                QCheckBox::indicator:checked {{
                    background-color: {color.name()};
                }}
            """)

            cb.stateChanged.connect(
                lambda state, name=feature_name:
                    self.set_feature_visible(name, state == Qt.CheckState.Checked.value)
            )

            self._checkbox_map[feature_name] = cb
            self.checkbox_layout.addWidget(cb)

        self.checkbox_layout.addStretch()

    def set_feature_visible(self, feature_name: str, visible: bool):
        """Setzt die Sichtbarkeit eines Features."""
        self._visibility_map[feature_name] = visible

        if feature_name in self._series_map:
            self._series_map[feature_name].setVisible(visible)

        # Checkbox synchronisieren
        if feature_name in self._checkbox_map:
            cb = self._checkbox_map[feature_name]
            if cb.isChecked() != visible:
                cb.blockSignals(True)
                cb.setChecked(visible)
                cb.blockSignals(False)

    def get_visible_features(self) -> List[str]:
        """Gibt Liste der sichtbaren Features zurueck."""
        return [name for name, visible in self._visibility_map.items() if visible]

    def set_crosshair_position(self, x_index: int):
        """Setzt die Crosshair-Position (fuer Synchronisation)."""
        self.chart_view.set_crosshair_position(float(x_index))

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

    def clear(self):
        """Leert den Chart."""
        for series in self._series_map.values():
            self.chart.removeSeries(series)

        self._series_map.clear()
        self._visibility_map.clear()
        self._feature_names.clear()
        self._data_length = 0

        # Checkboxen leeren
        for cb in self._checkbox_map.values():
            self.checkbox_layout.removeWidget(cb)
            cb.deleteLater()
        self._checkbox_map.clear()

        self.chart.setTitle(self.title)
