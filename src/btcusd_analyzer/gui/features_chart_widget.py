"""
FeaturesChartWidget - pyqtgraph-basiertes Chart-Widget fuer Feature-Visualisierung
Gestapelte Subplots fuer jedes Feature mit synchronisiertem Zoom/Pan
"""

from typing import Optional, Dict, List
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QCheckBox, QScrollArea, QFrame, QSplitter
)
from PyQt6.QtCore import Qt, pyqtSignal

import pyqtgraph as pg

from .styles import COLORS
from ..core.logger import get_logger


class FeaturesChartWidget(QWidget):
    """
    pyqtgraph-basiertes Chart-Widget fuer Feature-Visualisierung.

    Features:
    - Gestapelte Subplots fuer jedes Feature (untereinander)
    - Synchronisierter X-Zoom/Pan ueber alle Plots
    - Eingebautes Mausrad-Zoom und Drag-Panning
    - Z-Score Normalisierung
    - Checkbox-Steuerung fuer Feature-Visibility
    - Dark Theme
    """

    # Signal mit X-Index bei Mausbewegung
    crosshairMoved = pyqtSignal(int)

    # Farbpalette fuer Features
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

        # pyqtgraph konfigurieren
        pg.setConfigOptions(antialias=True)

        # Dynamische Verwaltung
        self._plot_items: Dict[str, pg.PlotItem] = {}  # Feature -> PlotItem
        self._plot_data: Dict[str, pg.PlotDataItem] = {}  # Feature -> Daten-Linie
        self._visibility_map: Dict[str, bool] = {}
        self._color_map: Dict[str, str] = {}
        self._checkbox_map: Dict[str, QCheckBox] = {}

        # Daten speichern
        self._data_length: int = 0
        self._feature_names: List[str] = []
        self._features_dict: Dict[str, np.ndarray] = {}

        # Crosshair-Linien
        self._crosshair_lines: List[pg.InfiniteLine] = []

        self._init_ui()

    def _init_ui(self):
        """Initialisiert die UI-Komponenten."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Zoom-Kontrollen
        zoom_controls = self._create_zoom_controls()
        layout.addWidget(zoom_controls)

        # GraphicsLayoutWidget fuer gestapelte Plots
        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.graphics_widget.setBackground(COLORS['bg_primary'])
        layout.addWidget(self.graphics_widget, 1)

        # Checkbox-Container fuer Feature-Visibility
        self.checkbox_container = self._create_checkbox_container()
        layout.addWidget(self.checkbox_container)

    def _create_zoom_controls(self) -> QWidget:
        """Erstellt die Zoom-Kontroll-Buttons."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # Reset
        reset_btn = QPushButton('Reset Zoom')
        reset_btn.setStyleSheet(self._button_style('#666666'))
        reset_btn.clicked.connect(self._reset_zoom)
        layout.addWidget(reset_btn)

        # Auto-Range Y
        auto_y_btn = QPushButton('Auto Y')
        auto_y_btn.setStyleSheet(self._button_style('#555555'))
        auto_y_btn.clicked.connect(self._auto_range_y)
        layout.addWidget(auto_y_btn)

        # Hinweis
        hint_label = QLabel('Mausrad=Zoom, Drag=Pan, Rechtsklick=Optionen')
        hint_label.setStyleSheet('color: #888888; font-size: 10px;')
        layout.addWidget(hint_label)

        layout.addStretch()

        # Titel
        self.title_label = QLabel(self.title)
        self.title_label.setStyleSheet('color: white; font-weight: bold; font-size: 12px;')
        layout.addWidget(self.title_label)

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
        return f"""
            QPushButton {{
                background-color: {hex_color};
                color: white;
                border: none;
                border-radius: 3px;
                padding: 5px 10px;
            }}
            QPushButton:hover {{
                background-color: {hex_color}cc;
            }}
            QPushButton:pressed {{
                background-color: {hex_color}99;
            }}
        """

    def _get_color_for_feature(self, feature_name: str, index: int) -> str:
        """Gibt konsistente Farbe fuer ein Feature zurueck."""
        if feature_name in self._color_map:
            return self._color_map[feature_name]

        color_idx = index % len(self.FEATURE_COLORS)
        color = self.FEATURE_COLORS[color_idx]
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

        Erstellt gestapelte Subplots - einen pro Feature.

        Args:
            features_dict: Dict mit Feature-Name -> Daten-Array
            title: Optionaler Titel
        """
        if not features_dict:
            self.clear()
            return

        self._features_dict = features_dict
        self._feature_names = list(features_dict.keys())
        self._data_length = max(len(data) for data in features_dict.values())

        # Alte Plots entfernen
        self.graphics_widget.clear()
        self._plot_items.clear()
        self._plot_data.clear()
        self._crosshair_lines.clear()

        # Erster Plot fuer X-Achsen-Synchronisation
        first_plot: Optional[pg.PlotItem] = None

        # Gestapelte Plots erstellen (untereinander)
        for idx, (feature_name, data) in enumerate(features_dict.items()):
            # Z-Score normalisieren
            normalized = self._normalize_zscore(data)
            x_data = np.arange(len(normalized))

            # Farbe holen
            color = self._get_color_for_feature(feature_name, idx)

            # Neuen Plot erstellen
            plot = self.graphics_widget.addPlot(row=idx, col=0)
            plot.setLabel('left', feature_name, color=color, size='9pt')
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.getAxis('left').setPen(pg.mkPen(color=color, width=1))
            plot.getAxis('left').setTextPen(pg.mkPen(color=color))
            plot.getAxis('bottom').setPen(pg.mkPen(color='#808080'))
            plot.getAxis('bottom').setTextPen(pg.mkPen(color='#aaaaaa'))

            # Nur beim letzten Plot X-Achsen-Label zeigen
            if idx < len(features_dict) - 1:
                plot.hideAxis('bottom')

            # Daten-Linie hinzufuegen
            pen = pg.mkPen(color=color, width=1.5)
            line = plot.plot(x_data, normalized, pen=pen)
            self._plot_data[feature_name] = line
            self._plot_items[feature_name] = plot

            # Crosshair-Linie hinzufuegen
            crosshair = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen(color='#ffff00', width=1, style=Qt.PenStyle.DashLine))
            crosshair.setVisible(False)
            plot.addItem(crosshair)
            self._crosshair_lines.append(crosshair)

            # X-Achsen synchronisieren
            if first_plot is None:
                first_plot = plot
            else:
                plot.setXLink(first_plot)

            # Sichtbarkeit setzen
            visible = self._visibility_map.get(feature_name, True)
            self._visibility_map[feature_name] = visible
            if not visible:
                plot.hide()

            # Mouse-Move Handler fuer Crosshair
            plot.scene().sigMouseMoved.connect(
                lambda pos, p=plot: self._on_mouse_moved(pos, p)
            )

        # Checkboxen aktualisieren
        self._rebuild_checkboxes()

        # Titel aktualisieren
        chart_title = title or self.title
        self.title_label.setText(f'{chart_title} ({len(features_dict)} Features)')

        self._logger.debug(f"[FeaturesChart] Update: {len(features_dict)} Features gestapelt")

    def _on_mouse_moved(self, pos, plot: pg.PlotItem):
        """Handler fuer Mausbewegung - aktualisiert Crosshair."""
        if plot.sceneBoundingRect().contains(pos):
            mouse_point = plot.vb.mapSceneToView(pos)
            x_index = int(round(mouse_point.x()))

            # Alle Crosshair-Linien aktualisieren
            for crosshair in self._crosshair_lines:
                crosshair.setPos(mouse_point.x())
                crosshair.setVisible(True)

            # Signal emittieren
            if 0 <= x_index < self._data_length:
                self.crosshairMoved.emit(x_index)

    def _rebuild_checkboxes(self):
        """Erstellt Checkboxen dynamisch basierend auf aktuellen Features."""
        # Alte Checkboxen entfernen
        for cb in self._checkbox_map.values():
            self.checkbox_layout.removeWidget(cb)
            cb.deleteLater()
        self._checkbox_map.clear()

        # Stretch entfernen
        while self.checkbox_layout.count() > 0:
            item = self.checkbox_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Neue Checkboxen erstellen
        for feature_name in self._feature_names:
            color = self._color_map.get(feature_name, '#ffffff')
            cb = QCheckBox(feature_name)
            cb.setChecked(self._visibility_map.get(feature_name, True))

            # Farbcodierte Checkbox
            cb.setStyleSheet(f"""
                QCheckBox {{
                    color: {color};
                    font-size: 11px;
                    spacing: 4px;
                }}
                QCheckBox::indicator {{
                    width: 14px;
                    height: 14px;
                    border: 1px solid {color};
                    border-radius: 2px;
                }}
                QCheckBox::indicator:checked {{
                    background-color: {color};
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
        """Setzt die Sichtbarkeit eines Features (zeigt/versteckt den Plot)."""
        self._visibility_map[feature_name] = visible

        if feature_name in self._plot_items:
            plot = self._plot_items[feature_name]
            if visible:
                plot.show()
            else:
                plot.hide()

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
        """Setzt die Crosshair-Position (fuer externe Synchronisation)."""
        for crosshair in self._crosshair_lines:
            crosshair.setPos(float(x_index))
            crosshair.setVisible(True)

    def _reset_zoom(self):
        """Setzt den Zoom zurueck auf Originalbereiche."""
        for plot in self._plot_items.values():
            plot.autoRange()

    def _auto_range_y(self):
        """Setzt nur Y-Achsen auf Auto-Range."""
        for plot in self._plot_items.values():
            plot.enableAutoRange(axis='y')

    def clear(self):
        """Leert den Chart."""
        self.graphics_widget.clear()
        self._plot_items.clear()
        self._plot_data.clear()
        self._visibility_map.clear()
        self._feature_names.clear()
        self._features_dict.clear()
        self._data_length = 0
        self._crosshair_lines.clear()

        # Checkboxen leeren
        for cb in self._checkbox_map.values():
            self.checkbox_layout.removeWidget(cb)
            cb.deleteLater()
        self._checkbox_map.clear()

        self.title_label.setText(self.title)
