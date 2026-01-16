"""
Control Panel - Steuerungs-UI fuer den Backtest (linke Spalte).
"""

from typing import Callable, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QSlider, QTextEdit,
    QScrollArea, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from ..styles import StyleFactory


class ControlPanel(QWidget):
    """
    Steuerungs-Panel fuer den Backtester (linke Spalte).

    Enthaelt:
    - Start/Stop/Einzelschritt/Reset Buttons
    - Geschwindigkeits-Slider
    - Turbo/Debug/Invert Optionen
    - Positions-Anzeige
    - Fortschritts-Anzeige
    - Trade-Log
    """

    # Signals
    start_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    step_clicked = pyqtSignal()
    reset_clicked = pyqtSignal()
    stats_clicked = pyqtSignal()
    speed_changed = pyqtSignal(int)
    turbo_toggled = pyqtSignal(bool)
    debug_toggled = pyqtSignal(bool)
    invert_toggled = pyqtSignal(bool)
    close_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Erstellt die UI-Komponenten."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        panel = QWidget()
        panel.setStyleSheet("background-color: rgb(46, 46, 46);")
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # Titel
        title = QLabel("Backtester Steuerung")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: white;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Steuerungs-Buttons
        layout.addWidget(self._create_control_group())

        # Geschwindigkeit
        layout.addWidget(self._create_speed_group())

        # Aktuelle Position
        layout.addWidget(self._create_position_group())

        # Fortschritt
        layout.addWidget(self._create_progress_group())

        # Trade-Log
        layout.addWidget(self._create_tradelog_group())

        layout.addStretch()

        # Schliessen Button
        close_btn = QPushButton("Schliessen")
        close_btn.setStyleSheet(self._button_style((0.4, 0.4, 0.4)))
        close_btn.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        close_btn.clicked.connect(self.close_clicked.emit)
        layout.addWidget(close_btn)

        scroll.setWidget(panel)

        # Main Layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

    def _create_control_group(self) -> QGroupBox:
        """Erstellt die Steuerungs-Buttons."""
        group = QGroupBox("Steuerung")
        group.setStyleSheet(self._group_style((0.3, 0.7, 1)))
        layout = QGridLayout(group)
        layout.setSpacing(8)

        # Start Button
        self.start_btn = QPushButton("Start")
        self.start_btn.setStyleSheet(self._button_style((0.2, 0.7, 0.3)))
        self.start_btn.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.start_btn.clicked.connect(self.start_clicked.emit)
        layout.addWidget(self.start_btn, 0, 0)

        # Stop Button
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet(self._button_style((0.8, 0.3, 0.2)))
        self.stop_btn.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_clicked.emit)
        layout.addWidget(self.stop_btn, 0, 1)

        # Einzelschritt Button
        self.step_btn = QPushButton("Einzelschritt")
        self.step_btn.setStyleSheet(self._button_style((0.5, 0.5, 0.7)))
        self.step_btn.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.step_btn.clicked.connect(self.step_clicked.emit)
        layout.addWidget(self.step_btn, 1, 0)

        # Reset Button
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setStyleSheet(self._button_style((0.5, 0.5, 0.5)))
        self.reset_btn.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.reset_btn.clicked.connect(self.reset_clicked.emit)
        layout.addWidget(self.reset_btn, 1, 1)

        # Trade-Statistik Button
        self.stats_btn = QPushButton("Trade-Statistik")
        self.stats_btn.setStyleSheet(self._button_style((0.3, 0.6, 0.9)))
        self.stats_btn.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.stats_btn.clicked.connect(self.stats_clicked.emit)
        layout.addWidget(self.stats_btn, 2, 0, 1, 2)

        return group

    def _create_speed_group(self) -> QGroupBox:
        """Erstellt die Geschwindigkeits-Kontrollen."""
        group = QGroupBox("Geschwindigkeit")
        group.setStyleSheet(self._group_style((0.9, 0.7, 0.3)))
        layout = QVBoxLayout(group)

        # Slider mit Label
        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Schritte/Sek:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 500)
        self.speed_slider.setValue(10)
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        slider_row.addWidget(self.speed_slider)
        self.speed_label = QLabel("10")
        self.speed_label.setStyleSheet("color: white; min-width: 30px;")
        slider_row.addWidget(self.speed_label)
        layout.addLayout(slider_row)

        # Turbo-Modus
        self.turbo_check = QCheckBox("Turbo-Modus (keine Chart-Updates)")
        self.turbo_check.setStyleSheet("color: rgb(128, 255, 128);")
        self.turbo_check.toggled.connect(self.turbo_toggled.emit)
        layout.addWidget(self.turbo_check)

        # Debug-Modus
        self.debug_check = QCheckBox("DEBUG-Modus (ausfuehrliches Log)")
        self.debug_check.setStyleSheet("color: rgb(255, 200, 100);")
        self.debug_check.toggled.connect(self.debug_toggled.emit)
        layout.addWidget(self.debug_check)

        # Signal-Invertierung
        self.invert_check = QCheckBox("Signale invertieren (BUY<->SELL)")
        self.invert_check.setStyleSheet("color: rgb(255, 128, 255);")
        self.invert_check.setToolTip("Tauscht BUY und SELL Signale")
        self.invert_check.toggled.connect(self.invert_toggled.emit)
        layout.addWidget(self.invert_check)

        # Aktuelle Geschwindigkeit
        speed_info = QHBoxLayout()
        speed_info.addWidget(QLabel("Aktuell:"))
        self.actual_speed_label = QLabel("- Schritte/Sek")
        self.actual_speed_label.setStyleSheet("color: rgb(77, 230, 255); font-weight: bold;")
        speed_info.addWidget(self.actual_speed_label)
        layout.addLayout(speed_info)

        return group

    def _create_position_group(self) -> QGroupBox:
        """Erstellt die Positions-Anzeige."""
        group = QGroupBox("Aktuelle Position")
        group.setStyleSheet(self._group_style((0.5, 0.9, 0.5)))
        layout = QGridLayout(group)
        layout.setColumnStretch(1, 1)

        labels = [
            ("Position:", "position_label", "NONE"),
            ("Einstiegspreis:", "entry_price_label", "-"),
            ("Aktueller Preis:", "current_price_label", "-"),
            ("Unrealisiert:", "unrealized_pnl_label", "-"),
        ]

        for row, (text, attr, default) in enumerate(labels):
            layout.addWidget(QLabel(text), row, 0)
            label = QLabel(default)
            label.setStyleSheet("color: white;")
            setattr(self, attr, label)
            layout.addWidget(label, row, 1)

        return group

    def _create_progress_group(self) -> QGroupBox:
        """Erstellt die Fortschritts-Anzeige."""
        group = QGroupBox("Fortschritt")
        group.setStyleSheet(self._group_style((0.7, 0.7, 0.7)))
        layout = QGridLayout(group)
        layout.setColumnStretch(1, 1)

        prog_labels = [
            ("Datenpunkt:", "datapoint_label", "0 / 0"),
            ("Datum:", "date_label", "-"),
            ("Letztes Signal:", "signal_label", "-"),
        ]

        for row, (text, attr, default) in enumerate(prog_labels):
            layout.addWidget(QLabel(text), row, 0)
            label = QLabel(default)
            label.setStyleSheet("color: white;" if row < 2 else "color: gray; font-weight: bold;")
            setattr(self, attr, label)
            layout.addWidget(label, row, 1)

        return group

    def _create_tradelog_group(self) -> QGroupBox:
        """Erstellt das Trade-Log."""
        group = QGroupBox("Trade-Log")
        group.setStyleSheet(self._group_style((0.9, 0.5, 0.9)))
        layout = QVBoxLayout(group)

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
        layout.addWidget(self.tradelog_text)

        return group

    def _on_speed_changed(self, value: int):
        """Handler fuer Speed-Slider Aenderungen."""
        self.speed_label.setText(str(value))
        self.speed_changed.emit(value)

    # === Oeffentliche Methoden fuer Updates ===

    def set_running_state(self, is_running: bool):
        """Setzt den Button-Zustand basierend auf Running-Status."""
        self.start_btn.setEnabled(not is_running)
        self.stop_btn.setEnabled(is_running)
        self.step_btn.setEnabled(not is_running)
        self.reset_btn.setEnabled(not is_running)

    def set_actual_speed(self, text: str):
        """Setzt die aktuelle Geschwindigkeits-Anzeige."""
        self.actual_speed_label.setText(text)

    def update_position(self, position: str, entry_price: float, current_price: float, unrealized: float):
        """Aktualisiert die Positions-Anzeige."""
        self.position_label.setText(position)
        if position == 'LONG':
            self.position_label.setStyleSheet("color: rgb(77, 230, 77); font-weight: bold;")
        elif position == 'SHORT':
            self.position_label.setStyleSheet("color: rgb(230, 77, 77); font-weight: bold;")
        else:
            self.position_label.setStyleSheet("color: gray;")

        if entry_price > 0:
            self.entry_price_label.setText(f"${entry_price:,.2f}")
        else:
            self.entry_price_label.setText("-")

        if current_price > 0:
            self.current_price_label.setText(f"${current_price:,.2f}")
        else:
            self.current_price_label.setText("-")

        if position != 'NONE':
            self.unrealized_pnl_label.setText(f"${unrealized:,.2f}")
            if unrealized >= 0:
                self.unrealized_pnl_label.setStyleSheet("color: rgb(77, 230, 77);")
            else:
                self.unrealized_pnl_label.setStyleSheet("color: rgb(230, 77, 77);")
        else:
            self.unrealized_pnl_label.setText("-")
            self.unrealized_pnl_label.setStyleSheet("color: white;")

    def update_progress(self, current: int, total: int, date_str: str, signal: int):
        """Aktualisiert die Fortschritts-Anzeige."""
        self.datapoint_label.setText(f"{current} / {total}")
        self.date_label.setText(date_str)

        if signal == 1:
            self.signal_label.setText("BUY")
            self.signal_label.setStyleSheet("color: rgb(77, 230, 77); font-weight: bold;")
        elif signal == 2:
            self.signal_label.setText("SELL")
            self.signal_label.setStyleSheet("color: rgb(230, 77, 77); font-weight: bold;")
        else:
            self.signal_label.setText("HOLD")
            self.signal_label.setStyleSheet("color: gray; font-weight: bold;")

    def add_tradelog(self, message: str):
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

    def reset_labels(self):
        """Setzt alle Labels zurueck."""
        self.position_label.setText("NONE")
        self.position_label.setStyleSheet("color: gray;")
        self.entry_price_label.setText("-")
        self.current_price_label.setText("-")
        self.unrealized_pnl_label.setText("-")
        self.signal_label.setText("-")
        self.signal_label.setStyleSheet("color: gray;")
        self.actual_speed_label.setText("- Schritte/Sek")
        self.tradelog_text.setPlainText("Kein Trade")

    def _group_style(self, color: tuple) -> str:
        """Generiert GroupBox-Style mit farbigem Titel."""
        return StyleFactory.group_style(title_color=color)

    def _button_style(self, color: tuple) -> str:
        """Generiert Button-Style aus RGB-Tuple (0-1 Range)."""
        return StyleFactory.button_style(color)
