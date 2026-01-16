"""
Main Window - Haupt-GUI des BTCUSD Analyzers
Portiert von MATLAB btc_analyzer_gui.m
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox, QStatusBar, QScrollArea,
    QFileDialog, QMessageBox, QComboBox, QFrame, QDateEdit,
    QTextEdit, QSlider, QCheckBox, QSplitter, QTableWidget,
    QTableWidgetItem, QHeaderView, QApplication
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QDate
from PyQt6.QtGui import QFont, QAction, QColor

import logging

import json
import numpy as np
import pandas as pd

from ..core.config import Config
from ..core.logger import get_logger, Logger
from ..data.reader import CSVReader
from .styles import get_stylesheet, COLORS, StyleFactory


class GUILogHandler(logging.Handler):
    """Handler der Logger-Meldungen an die GUI weiterleitet."""

    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self.setLevel(Logger.TRACE)

    def emit(self, record):
        try:
            # Level-Name ermitteln
            level = record.levelname
            if record.levelno == Logger.TRACE:
                level = 'TRACE'
            elif record.levelno == Logger.SUCCESS:
                level = 'SUCCESS'

            # TIMING aus Message erkennen
            msg = self.format(record)
            if msg.startswith('[TIMING]'):
                level = 'TIMING'
                msg = msg[8:].strip()  # [TIMING] prefix entfernen

            # _from_handler=True verhindert doppeltes Logging in die Datei
            self.callback(msg, level, _from_handler=True)
        except Exception:
            self.handleError(record)


class MainWindow(QMainWindow):
    """
    Haupt-Fenster des BTCUSD Analyzers.

    Layout: 2 Spalten (340-420px | flexible)
    - Links: Bedienelemente (scrollbar)
    - Rechts: Logger mit HTML-Ausgabe

    Gruppen:
    1. Daten Laden (CSV + Binance Download)
    2. Datenanalyse
    3. BILSTM Training
    4. Modell & Vorhersage
    5. Parameter Management
    """

    # Signals
    data_loaded = pyqtSignal(object)  # DataFrame
    model_loaded = pyqtSignal(object)  # Model
    training_data_ready = pyqtSignal(object)  # Training Data

    def __init__(self, base_dir: Optional[Path] = None):
        super().__init__()

        # Konfiguration
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.config = Config(self.base_dir)
        self.logger = get_logger('btcusd_analyzer', self.config.paths.log_dir)

        # State
        self.data: Optional[pd.DataFrame] = None
        self.data_path: Optional[Path] = None
        self.training_data = None
        self.training_info = None
        self.backtest_info = None  # Separates Backtest-Daten (nicht im Training verwendet)
        self.model = None
        self.model_path: Optional[Path] = None
        self.model_info = None

        # Logger Einstellungen
        self.logger_mode = 'both'  # 'window', 'both', 'file'
        self.log_level = 5  # 1-5 (ERROR bis TRACE)
        self.enable_timing = False
        self._gui_handler = None

        # UI initialisieren
        self._init_ui()
        self._connect_signals()

        # GUI Log-Handler registrieren (nach UI-Init, damit log_text existiert)
        self._setup_gui_log_handler()

        # Stylesheet anwenden
        self.setStyleSheet(get_stylesheet())

        # Logging
        self.logger.info('BTCUSD Analyzer GUI gestartet')
        self.logger.info(f'Log-Datei: {self.logger.get_log_file_path()}')

        # Auto-Load letzte Daten und Session
        QTimer.singleShot(100, self._auto_load_last_data)
        QTimer.singleShot(200, self._auto_load_latest_session)

    def _init_ui(self):
        """Initialisiert die UI-Komponenten."""
        self.setWindowTitle('1 - Main')
        self.setMinimumSize(1400, 950)

        # Fenstergroesse als Prozent der Bildschirmgroesse (85%)
        screen_percent = 0.85
        screen = QApplication.primaryScreen()
        if screen:
            available_geometry = screen.availableGeometry()
            width = int(available_geometry.width() * screen_percent)
            height = int(available_geometry.height() * screen_percent)
            self.resize(width, height)
            # Fenster zentrieren
            x = (available_geometry.width() - width) // 2 + available_geometry.x()
            y = (available_geometry.height() - height) // 2 + available_geometry.y()
            self.move(x, y)
        else:
            # Fallback falls kein Screen verfuegbar
            self.resize(1820, 1235)

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main Layout: 2 Spalten (340-420px | flexible)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Splitter fuer resizable Spalten
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Linke Spalte: Bedienelemente
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)

        # Rechte Spalte: Logger
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)

        # Splitter Proportionen (linke Spalte breiter)
        splitter.setSizes([380, 1020])
        splitter.setStretchFactor(0, 0)  # Linke Spalte nicht stretchen
        splitter.setStretchFactor(1, 1)  # Rechte Spalte stretchen

        # Menu Bar
        self._create_menu_bar()

        # Status Bar
        self._create_status_bar()

    def _create_left_panel(self) -> QWidget:
        """Erstellt das linke Control-Panel mit Scroll-Funktion."""
        panel = QWidget()
        panel.setMinimumWidth(340)
        panel.setMaximumWidth(420)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        # Scroll Area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(8)
        scroll_layout.setContentsMargins(5, 5, 5, 5)

        # Titel
        title_label = QLabel('BTCUSD Analyzer')
        title_label.setFont(QFont('Segoe UI', 12, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setFixedHeight(28)
        scroll_layout.addWidget(title_label)

        # === GRUPPE 1: Daten Laden ===
        data_group = self._create_data_load_group()
        scroll_layout.addWidget(data_group)

        # === GRUPPE 2: Datenanalyse ===
        analyze_group = self._create_analyze_group()
        scroll_layout.addWidget(analyze_group)

        # === GRUPPE 3: BILSTM Training ===
        train_group = self._create_training_group()
        scroll_layout.addWidget(train_group)

        # === GRUPPE 4: Modell & Vorhersage ===
        model_group = self._create_model_group()
        scroll_layout.addWidget(model_group)

        # === GRUPPE 5: Parameter Management ===
        param_group = self._create_param_group()
        scroll_layout.addWidget(param_group)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        return panel

    def _create_group_title(self, text: str, color: tuple) -> QLabel:
        """Erstellt einen Gruppen-Titel mit Hintergrundfarbe."""
        label = QLabel(text)
        label.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setFixedHeight(20)
        r, g, b = [int(c * 255) for c in color]
        label.setStyleSheet(f'''
            QLabel {{
                background-color: rgb({r}, {g}, {b});
                color: white;
                border-radius: 3px;
                padding: 1px;
            }}
        ''')
        return label

    def _create_data_load_group(self) -> QGroupBox:
        """Erstellt Gruppe 1: Daten Laden - Uebersichtliche Struktur."""
        group = QGroupBox()
        group.setStyleSheet('QGroupBox { border: none; }')
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        # Titel
        title = self._create_group_title('Daten Laden', (0.2, 0.4, 0.6))
        layout.addWidget(title)

        # === Status-Anzeige fuer geladene Daten ===
        status_frame = QFrame()
        status_frame.setStyleSheet('''
            QFrame {
                background-color: #1a1a2e;
                border: 1px solid #333355;
                border-radius: 3px;
                padding: 3px;
            }
        ''')
        status_layout = QVBoxLayout(status_frame)
        status_layout.setContentsMargins(5, 3, 5, 3)
        status_layout.setSpacing(1)

        # Datei-Status
        self.data_file_label = QLabel('Keine Daten geladen')
        self.data_file_label.setFont(QFont('Segoe UI', 8))
        self.data_file_label.setStyleSheet('color: #888888;')
        status_layout.addWidget(self.data_file_label)

        # Daten-Info (Anzahl, Zeitraum)
        self.data_info_label = QLabel('-')
        self.data_info_label.setFont(QFont('Segoe UI', 8))
        self.data_info_label.setStyleSheet('color: #666666;')
        status_layout.addWidget(self.data_info_label)

        layout.addWidget(status_frame)

        # === Untergruppe 1.1: Lokale Datei ===
        local_group = QGroupBox('Lokal')
        local_group.setFont(QFont('Segoe UI', 8, QFont.Weight.Bold))
        local_layout = QVBoxLayout(local_group)
        local_layout.setSpacing(3)
        local_layout.setContentsMargins(4, 8, 4, 4)

        # Button-Reihe: Oeffnen | Letzte laden
        btn_row = QHBoxLayout()

        self.load_file_btn = QPushButton('CSV...')
        self.load_file_btn.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        self.load_file_btn.setFixedHeight(28)
        self.load_file_btn.setStyleSheet(self._button_style((0.4, 0.4, 0.4)))
        self.load_file_btn.setToolTip('CSV-Datei aus Ordner waehlen')
        self.load_file_btn.clicked.connect(self._load_csv)
        btn_row.addWidget(self.load_file_btn)

        self.load_last_data_btn = QPushButton('Letzte')
        self.load_last_data_btn.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        self.load_last_data_btn.setFixedHeight(28)
        self.load_last_data_btn.setStyleSheet(self._button_style((0.3, 0.5, 0.4)))
        self.load_last_data_btn.setToolTip('Zuletzt verwendete Datei laden')
        self.load_last_data_btn.clicked.connect(self._load_last_data)
        btn_row.addWidget(self.load_last_data_btn)

        local_layout.addLayout(btn_row)

        # Daten-Ordner oeffnen
        self.open_folder_btn = QPushButton('Ordner')
        self.open_folder_btn.setFont(QFont('Segoe UI', 8))
        self.open_folder_btn.setFixedHeight(22)
        self.open_folder_btn.setStyleSheet(self._button_style((0.35, 0.35, 0.4)))
        self.open_folder_btn.setToolTip('Daten-Verzeichnis im Explorer oeffnen')
        self.open_folder_btn.clicked.connect(self._open_data_folder)
        local_layout.addWidget(self.open_folder_btn)

        layout.addWidget(local_group)

        # === Untergruppe 1.2: Binance API Download ===
        binance_group = QGroupBox('Binance Download')
        binance_group.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        binance_group.setStyleSheet('''
            QGroupBox {
                border: 1px solid #444466;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 4px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                color: #90cdf4;
            }
        ''')
        binance_layout = QVBoxLayout(binance_group)
        binance_layout.setSpacing(6)
        binance_layout.setContentsMargins(8, 12, 8, 8)

        # === Zeile 1: Symbol und Intervall ===
        row1 = QHBoxLayout()
        row1.setSpacing(8)

        symbol_label = QLabel('Symbol:')
        symbol_label.setFont(QFont('Segoe UI', 9))
        symbol_label.setStyleSheet('color: #aaaaaa;')
        row1.addWidget(symbol_label)

        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'])
        self.symbol_combo.setCurrentText('BTCUSDT')
        self.symbol_combo.setFixedHeight(26)
        self.symbol_combo.setFont(QFont('Segoe UI', 9))
        row1.addWidget(self.symbol_combo, 1)

        interval_label = QLabel('Intervall:')
        interval_label.setFont(QFont('Segoe UI', 9))
        interval_label.setStyleSheet('color: #aaaaaa;')
        row1.addWidget(interval_label)

        self.interval_combo = QComboBox()
        self.interval_combo.addItems(['1m', '5m', '15m', '1h', '4h', '1d', '1w'])
        self.interval_combo.setCurrentText('1h')
        self.interval_combo.setFixedHeight(26)
        self.interval_combo.setFixedWidth(60)
        self.interval_combo.setFont(QFont('Segoe UI', 9))
        row1.addWidget(self.interval_combo)

        binance_layout.addLayout(row1)

        # === Zeile 2: Zeitraum (Von - Bis) ===
        date_frame = QFrame()
        date_frame.setStyleSheet('''
            QFrame {
                background-color: #1a1a2e;
                border: 1px solid #333355;
                border-radius: 4px;
                padding: 4px;
            }
        ''')
        date_layout = QHBoxLayout(date_frame)
        date_layout.setContentsMargins(8, 6, 8, 6)
        date_layout.setSpacing(8)

        from_label = QLabel('Von:')
        from_label.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        from_label.setStyleSheet('color: #68d391;')
        date_layout.addWidget(from_label)

        self.from_date = QDateEdit()
        self.from_date.setCalendarPopup(True)
        self.from_date.setDate(QDate(2025, 1, 1))
        self.from_date.setDisplayFormat('dd.MM.yyyy')
        self.from_date.setFixedHeight(26)
        self.from_date.setFont(QFont('Segoe UI', 9))
        date_layout.addWidget(self.from_date)

        date_layout.addSpacing(8)

        to_label = QLabel('Bis:')
        to_label.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        to_label.setStyleSheet('color: #fc8181;')
        date_layout.addWidget(to_label)

        self.to_date = QDateEdit()
        self.to_date.setCalendarPopup(True)
        self.to_date.setDate(QDate(2025, 12, 31))
        self.to_date.setDisplayFormat('dd.MM.yyyy')
        self.to_date.setFixedHeight(26)
        self.to_date.setFont(QFont('Segoe UI', 9))
        date_layout.addWidget(self.to_date)

        binance_layout.addWidget(date_frame)

        # === Zeile 3: Schnellauswahl ===
        quick_frame = QFrame()
        quick_layout = QHBoxLayout(quick_frame)
        quick_layout.setContentsMargins(0, 0, 0, 0)
        quick_layout.setSpacing(4)

        quick_label = QLabel('Schnell:')
        quick_label.setFont(QFont('Segoe UI', 9))
        quick_label.setStyleSheet('color: #888888;')
        quick_layout.addWidget(quick_label)

        quick_periods = [('1M', 30), ('3M', 90), ('6M', 180), ('1J', 365), ('2J', 730)]
        for text, days in quick_periods:
            btn = QPushButton(text)
            btn.setFixedSize(42, 24)
            btn.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
            btn.setStyleSheet(self._button_style((0.3, 0.4, 0.5)))
            btn.setToolTip(f'Letzte {days} Tage')
            btn.clicked.connect(lambda checked, d=days: self._set_quick_period(d))
            quick_layout.addWidget(btn)

        quick_layout.addStretch()
        binance_layout.addWidget(quick_frame)

        # === Zeile 4: Download Button ===
        self.download_btn = QPushButton('Download starten')
        self.download_btn.setFont(QFont('Segoe UI', 10, QFont.Weight.Bold))
        self.download_btn.setFixedHeight(32)
        self.download_btn.setStyleSheet(self._button_style((0.2, 0.55, 0.9)))
        self.download_btn.setToolTip('Daten von Binance API herunterladen')
        self.download_btn.clicked.connect(self._download_data)
        binance_layout.addWidget(self.download_btn)

        layout.addWidget(binance_group)

        return group

    def _set_quick_period(self, days: int):
        """Setzt den Zeitraum auf die letzten X Tage."""
        from datetime import date, timedelta
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        self.from_date.setDate(QDate(start_date.year, start_date.month, start_date.day))
        self.to_date.setDate(QDate(end_date.year, end_date.month, end_date.day))
        self._log(f'Zeitraum gesetzt: Letzte {days} Tage', 'DEBUG')

    def _load_last_data(self):
        """Laedt die zuletzt verwendete CSV-Datei."""
        try:
            reader = CSVReader(self.config.paths.data_dir)
            df, filepath = reader.load_last_csv()

            if df is not None:
                self.data = df
                self.data_path = filepath
                self.data_loaded.emit(df)
                self._log(f'Letzte Datei geladen: {filepath.name}', 'SUCCESS')
                self._update_data_status()
            else:
                self._log('Keine vorherige Datei gefunden', 'WARNING')
        except Exception as e:
            self._log(f'Fehler beim Laden: {e}', 'ERROR')

    def _open_data_folder(self):
        """Oeffnet den Daten-Ordner im Explorer."""
        import subprocess
        import platform

        folder = self.config.paths.data_dir
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)

        if platform.system() == 'Windows':
            subprocess.Popen(['explorer', str(folder)])
        elif platform.system() == 'Darwin':
            subprocess.Popen(['open', str(folder)])
        else:
            subprocess.Popen(['xdg-open', str(folder)])

        self._log(f'Ordner geoeffnet: {folder}', 'DEBUG')

    def _update_data_status(self):
        """Aktualisiert die Daten-Status-Anzeige."""
        if self.data is not None and self.data_path is not None:
            # Dateiname
            self.data_file_label.setText(self.data_path.name)
            self.data_file_label.setStyleSheet('color: #68d391;')

            # Zeitraum und Anzahl
            count = len(self.data)
            if hasattr(self.data.index, 'min') and hasattr(self.data.index, 'max'):
                try:
                    start = self.data.index.min()
                    end = self.data.index.max()
                    if hasattr(start, 'strftime'):
                        start_str = start.strftime('%d.%m.%y')
                        end_str = end.strftime('%d.%m.%y')
                    else:
                        start_str = str(start)[:10]
                        end_str = str(end)[:10]
                    self.data_info_label.setText(
                        f'{count:,} | {start_str}-{end_str}'
                    )
                except Exception:
                    self.data_info_label.setText(f'{count:,} Datensaetze')
            else:
                self.data_info_label.setText(f'{count:,} Datensaetze')

            self.data_info_label.setStyleSheet('color: #90cdf4;')
        else:
            self.data_file_label.setText('Keine Daten geladen')
            self.data_file_label.setStyleSheet('color: #888888;')
            self.data_info_label.setText('-')
            self.data_info_label.setStyleSheet('color: #666666;')

    def _create_analyze_group(self) -> QGroupBox:
        """Erstellt Gruppe 2: Datenanalyse."""
        group = QGroupBox()
        group.setStyleSheet('QGroupBox { border: none; }')
        layout = QVBoxLayout(group)
        layout.setSpacing(4)

        # Titel
        title = self._create_group_title('Datenanalyse', (0.3, 0.5, 0.3))
        layout.addWidget(title)

        # Buttons nebeneinander
        btn_row = QHBoxLayout()

        self.analyze_btn = QPushButton('Analyse')
        self.analyze_btn.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        self.analyze_btn.setFixedHeight(26)
        self.analyze_btn.setStyleSheet(self._button_style((0.2, 0.8, 0.4)))
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self._analyze_data)
        btn_row.addWidget(self.analyze_btn)

        self.prepare_btn = QPushButton('Vorbereiten')
        self.prepare_btn.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        self.prepare_btn.setFixedHeight(26)
        self.prepare_btn.setStyleSheet(self._button_style((0.8, 0.4, 0.2)))
        self.prepare_btn.setEnabled(False)
        self.prepare_btn.setToolTip('Training vorbereiten')
        self.prepare_btn.clicked.connect(self._prepare_training_data)
        btn_row.addWidget(self.prepare_btn)

        layout.addLayout(btn_row)

        # Signale visualisieren
        self.visualize_btn = QPushButton('Signale visualisieren')
        self.visualize_btn.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        self.visualize_btn.setFixedHeight(26)
        self.visualize_btn.setStyleSheet(self._button_style((0.2, 0.8, 0.8)))
        self.visualize_btn.setEnabled(False)
        self.visualize_btn.clicked.connect(self._visualize_signals)
        layout.addWidget(self.visualize_btn)

        # Trainingsdaten aus Workspace laden
        self.load_training_btn = QPushButton('Trainingsdaten laden')
        self.load_training_btn.setFont(QFont('Segoe UI', 8))
        self.load_training_btn.setFixedHeight(22)
        self.load_training_btn.setStyleSheet(self._button_style((0.5, 0.3, 0.6)))
        self.load_training_btn.setToolTip('Trainingsdaten aus Workspace laden')
        self.load_training_btn.clicked.connect(self._load_training_data)
        layout.addWidget(self.load_training_btn)

        return group

    def _create_training_group(self) -> QGroupBox:
        """Erstellt Gruppe 3: BILSTM Training."""
        group = QGroupBox()
        group.setStyleSheet('QGroupBox { border: none; }')
        layout = QVBoxLayout(group)
        layout.setSpacing(4)

        # Titel
        title = self._create_group_title('BILSTM Training', (0.6, 0.2, 0.8))
        layout.addWidget(title)

        # Training GUI Button
        self.train_gui_btn = QPushButton('Training GUI...')
        self.train_gui_btn.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        self.train_gui_btn.setFixedHeight(28)
        self.train_gui_btn.setStyleSheet(self._button_style((0.6, 0.2, 0.8)))
        self.train_gui_btn.setEnabled(False)
        self.train_gui_btn.clicked.connect(self._open_training_gui)
        layout.addWidget(self.train_gui_btn)

        return group

    def _create_model_group(self) -> QGroupBox:
        """Erstellt Gruppe 4: Modell & Vorhersage."""
        group = QGroupBox()
        group.setStyleSheet('QGroupBox { border: none; }')
        layout = QVBoxLayout(group)
        layout.setSpacing(4)

        # Titel
        title = self._create_group_title('Modell & Vorhersage', (1.0, 0.6, 0.2))
        layout.addWidget(title)

        # Button-Reihe: Laden | Letztes | Vorhersage
        btn_row = QHBoxLayout()

        self.load_model_btn = QPushButton('Laden')
        self.load_model_btn.setFont(QFont('Segoe UI', 8, QFont.Weight.Bold))
        self.load_model_btn.setFixedHeight(24)
        self.load_model_btn.setStyleSheet(self._button_style((0.5, 0.5, 0.5)))
        self.load_model_btn.setToolTip('Modell aus Datei laden')
        self.load_model_btn.clicked.connect(self._load_model)
        btn_row.addWidget(self.load_model_btn)

        self.load_last_btn = QPushButton('Letztes')
        self.load_last_btn.setFont(QFont('Segoe UI', 8, QFont.Weight.Bold))
        self.load_last_btn.setFixedHeight(24)
        self.load_last_btn.setStyleSheet(self._button_style((0.4, 0.6, 0.7)))
        self.load_last_btn.setToolTip('Zuletzt verwendetes Modell laden')
        self.load_last_btn.clicked.connect(self._load_last_model)
        btn_row.addWidget(self.load_last_btn)

        self.predict_btn = QPushButton('Predict')
        self.predict_btn.setFont(QFont('Segoe UI', 8, QFont.Weight.Bold))
        self.predict_btn.setFixedHeight(24)
        self.predict_btn.setStyleSheet(self._button_style((1.0, 0.6, 0.2)))
        self.predict_btn.setToolTip('Vorhersage mit geladenem Modell')
        self.predict_btn.setEnabled(False)
        self.predict_btn.clicked.connect(self._make_prediction)
        btn_row.addWidget(self.predict_btn)

        layout.addLayout(btn_row)

        # Button-Reihe: Backtester | Session laden
        btn_row2 = QHBoxLayout()

        self.backtest_btn = QPushButton('Backtester')
        self.backtest_btn.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        self.backtest_btn.setFixedHeight(26)
        self.backtest_btn.setStyleSheet(self._button_style((0.7, 0.4, 0.8)))
        self.backtest_btn.setEnabled(False)
        self.backtest_btn.clicked.connect(self._open_backtester)
        btn_row2.addWidget(self.backtest_btn)

        self.load_session_btn = QPushButton('Session')
        self.load_session_btn.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        self.load_session_btn.setFixedHeight(26)
        self.load_session_btn.setStyleSheet(self._button_style((0.3, 0.6, 0.8)))
        self.load_session_btn.setToolTip('Session-Ordner laden (Daten + Modell)')
        self.load_session_btn.clicked.connect(self._load_session)
        btn_row2.addWidget(self.load_session_btn)

        layout.addLayout(btn_row2)

        # Modell Info Labels
        self.model_name_label = QLabel('Modell: -')
        self.model_name_label.setFont(QFont('Segoe UI', 8))
        self.model_name_label.setStyleSheet('color: #999999;')
        layout.addWidget(self.model_name_label)

        self.model_folder_label = QLabel('Ordner: -')
        self.model_folder_label.setFont(QFont('Segoe UI', 8))
        self.model_folder_label.setStyleSheet('color: #808080;')
        layout.addWidget(self.model_folder_label)

        return group

    def _create_param_group(self) -> QGroupBox:
        """Erstellt Gruppe 5: Parameter Management."""
        group = QGroupBox()
        group.setStyleSheet('QGroupBox { border: none; }')
        layout = QVBoxLayout(group)
        layout.setSpacing(4)

        # Titel
        title = self._create_group_title('Parameter', (0.4, 0.7, 0.5))
        layout.addWidget(title)

        # Buttons nebeneinander
        btn_row = QHBoxLayout()

        self.save_params_btn = QPushButton('Speichern')
        self.save_params_btn.setFont(QFont('Segoe UI', 8, QFont.Weight.Bold))
        self.save_params_btn.setFixedHeight(24)
        self.save_params_btn.setStyleSheet(self._button_style((0.3, 0.6, 0.4)))
        self.save_params_btn.setToolTip('Parameter speichern')
        self.save_params_btn.clicked.connect(self._save_parameters)
        btn_row.addWidget(self.save_params_btn)

        self.load_params_btn = QPushButton('Laden')
        self.load_params_btn.setFont(QFont('Segoe UI', 8, QFont.Weight.Bold))
        self.load_params_btn.setFixedHeight(24)
        self.load_params_btn.setStyleSheet(self._button_style((0.5, 0.8, 0.6)))
        self.load_params_btn.setToolTip('Parameter laden')
        self.load_params_btn.clicked.connect(self._load_parameters)
        btn_row.addWidget(self.load_params_btn)

        layout.addLayout(btn_row)

        return group

    def _create_status_panel(self) -> QWidget:
        """Erstellt das detaillierte Status-Panel fuer Trainingsdaten."""
        panel = QWidget()
        panel.setMinimumWidth(280)
        panel.setMaximumWidth(350)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)

        # === Titel ===
        title = QLabel('Trainingsdaten')
        title.setFont(QFont('Segoe UI', 12, QFont.Weight.Bold))
        title.setStyleSheet('color: #90cdf4;')
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # === Scroll Area fuer alle Inhalte ===
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(8)
        scroll_layout.setContentsMargins(0, 0, 5, 0)

        # === 1. Status-Gruppe ===
        status_group = self._create_info_group('Status', [
            ('Pipeline', 'status_pipeline', '⬜ Ausstehend'),
            ('Rohdaten', 'status_raw', '-'),
            ('Labels', 'status_labels', '-'),
            ('Sequenzen', 'status_sequences', '-'),
        ])
        scroll_layout.addWidget(status_group)

        # === 2. DataFrame Info ===
        df_group = self._create_info_group('DataFrame', [
            ('Datensaetze', 'df_rows', '-'),
            ('Zeitraum', 'df_period', '-'),
            ('Intervall', 'df_interval', '-'),
            ('Spalten', 'df_columns', '-'),
            ('Speicher', 'df_memory', '-'),
        ])
        scroll_layout.addWidget(df_group)

        # === 3. Sequenz-Parameter ===
        seq_group = self._create_info_group('Sequenzen', [
            ('Lookback', 'seq_lookback', '-'),
            ('Lookforward', 'seq_lookforward', '-'),
            ('Features', 'seq_features', '-'),
            ('Gesamt', 'seq_total', '-'),
        ])
        scroll_layout.addWidget(seq_group)

        # === 4. Label-Verteilung ===
        label_group = self._create_info_group('Labels', [
            ('BUY', 'label_buy', '-'),
            ('SELL', 'label_sell', '-'),
            ('HOLD', 'label_hold', '-'),
            ('Balance', 'label_balance', '-'),
        ])
        scroll_layout.addWidget(label_group)

        # === 5. Training Split ===
        split_group = self._create_info_group('Train/Val Split', [
            ('Training', 'split_train', '-'),
            ('Validation', 'split_val', '-'),
            ('Ratio', 'split_ratio', '-'),
        ])
        scroll_layout.addWidget(split_group)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        return panel

    def _create_info_group(self, title: str, fields: list) -> QGroupBox:
        """
        Erstellt eine Info-Gruppe mit Label-Wert-Paaren.

        Args:
            title: Gruppentitel
            fields: Liste von (label, attr_name, default_value) Tupeln
        """
        group = QGroupBox(title)
        group.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        group.setStyleSheet('''
            QGroupBox {
                background-color: #1a1a2e;
                border: 1px solid #333355;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 4px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                color: #68d391;
            }
        ''')

        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(2)

        for label_text, attr_name, default_value in fields:
            row = QHBoxLayout()
            row.setSpacing(4)

            label = QLabel(f'{label_text}:')
            label.setFont(QFont('Segoe UI', 9))
            label.setStyleSheet('color: #888888;')
            label.setFixedWidth(80)
            row.addWidget(label)

            value_label = QLabel(default_value)
            value_label.setFont(QFont('Segoe UI', 9))
            value_label.setStyleSheet('color: #cccccc;')
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            row.addWidget(value_label, 1)

            # Speichere Referenz zum Value-Label
            setattr(self, f'info_{attr_name}', value_label)

            layout.addLayout(row)

        return group

    def _create_right_panel(self) -> QWidget:
        """Erstellt das rechte Panel mit Logger und Status-Panel nebeneinander."""
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # === Linke Seite: Logger ===
        logger_widget = QWidget()
        logger_layout = QVBoxLayout(logger_widget)
        logger_layout.setContentsMargins(0, 0, 0, 0)
        logger_layout.setSpacing(5)

        # Logger Header
        header_layout = QHBoxLayout()

        # Titel
        logger_title = QLabel('Logger')
        logger_title.setFont(QFont('Segoe UI', 14, QFont.Weight.Bold))
        header_layout.addWidget(logger_title)

        # Modus Dropdown
        self.logger_mode_combo = QComboBox()
        self.logger_mode_combo.addItems(['Fenster', 'Fenster+Datei', 'Nur Datei'])
        self.logger_mode_combo.setCurrentIndex(1)  # Default: Fenster+Datei
        self.logger_mode_combo.currentIndexChanged.connect(self._update_logger_mode)
        header_layout.addWidget(self.logger_mode_combo)

        # Level Dropdown
        self.logger_level_combo = QComboBox()
        self.logger_level_combo.addItems(['ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE'])
        self.logger_level_combo.setCurrentIndex(4)  # Default: TRACE
        self.logger_level_combo.setToolTip('Log-Level: Zeigt alle Meldungen bis zu diesem Level')
        self.logger_level_combo.currentIndexChanged.connect(self._update_logger_level)
        header_layout.addWidget(self.logger_level_combo)

        # Clear Button
        clear_btn = QPushButton('Clear')
        clear_btn.setFixedWidth(60)
        clear_btn.clicked.connect(self._clear_log)
        header_layout.addWidget(clear_btn)

        # Timing Checkbox
        self.timing_checkbox = QCheckBox('Timing')
        self.timing_checkbox.setToolTip('Zeitmessung aktivieren (in TRACE immer aktiv)')
        self.timing_checkbox.stateChanged.connect(self._update_timing)
        header_layout.addWidget(self.timing_checkbox)

        header_layout.addStretch()

        # Schrift Label
        font_label = QLabel('Schrift:')
        header_layout.addWidget(font_label)

        # Font Size Slider
        self.font_slider = QSlider(Qt.Orientation.Horizontal)
        self.font_slider.setRange(8, 14)
        self.font_slider.setValue(10)
        self.font_slider.setFixedWidth(100)
        self.font_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.font_slider.setTickInterval(2)
        self.font_slider.valueChanged.connect(self._update_log_font_size)
        header_layout.addWidget(self.font_slider)

        logger_layout.addLayout(header_layout)

        # Logger Textbereich (HTML-faehig)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self._log_font_size = 10  # Initiale Schriftgroesse
        self._update_log_stylesheet()
        logger_layout.addWidget(self.log_text)

        layout.addWidget(logger_widget, 1)  # Stretch factor 1

        # === Rechte Seite: Status-Panel ===
        status_panel = self._create_status_panel()
        layout.addWidget(status_panel, 0)  # Kein Stretch

        return panel

    def _button_style(self, color: tuple) -> str:
        """Generiert Button-Stylesheet aus RGB-Tuple (0-1 Range)."""
        return StyleFactory.button_style(color, padding='5px 10px')

    def _create_status_bar(self):
        """Erstellt die Status-Leiste."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.status_label = QLabel('Bereit')
        self.status_bar.addWidget(self.status_label, 1)

        self.gpu_indicator = QLabel()
        self.status_bar.addPermanentWidget(self.gpu_indicator)

        # GPU Status aktualisieren
        QTimer.singleShot(500, self._update_gpu_status)

    def _create_menu_bar(self):
        """Erstellt die Menu-Leiste."""
        menu_bar = self.menuBar()

        # Datei Menu
        file_menu = menu_bar.addMenu('Datei')

        load_action = QAction('CSV laden...', self)
        load_action.triggered.connect(self._load_csv)
        file_menu.addAction(load_action)

        file_menu.addSeparator()

        exit_action = QAction('Beenden', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Fenster Menu
        window_menu = menu_bar.addMenu('Fenster')

        training_action = QAction('Training GUI...', self)
        training_action.triggered.connect(self._open_training_gui)
        window_menu.addAction(training_action)

        backtest_action = QAction('Backtester...', self)
        backtest_action.triggered.connect(self._open_backtester)
        window_menu.addAction(backtest_action)

        backtrader_action = QAction('Backtrader Pro...', self)
        backtrader_action.triggered.connect(self._open_backtrader)
        window_menu.addAction(backtrader_action)

        walk_forward_action = QAction('Walk-Forward Analyse...', self)
        walk_forward_action.triggered.connect(self._open_walk_forward)
        window_menu.addAction(walk_forward_action)

        window_menu.addSeparator()

        trading_action = QAction('Live Trading...', self)
        trading_action.triggered.connect(self._open_trading)
        window_menu.addAction(trading_action)

        webserver_action = QAction('Web Dashboard...', self)
        webserver_action.triggered.connect(self._open_webserver)
        window_menu.addAction(webserver_action)

        # Hilfe Menu
        help_menu = menu_bar.addMenu('Hilfe')

        about_action = QAction('Ueber...', self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _connect_signals(self):
        """Verbindet Signals mit Slots."""
        self.data_loaded.connect(self._on_data_loaded)
        self.model_loaded.connect(self._on_model_loaded)
        self.training_data_ready.connect(self._on_training_data_ready)

    # === Log-Methoden ===

    def _log(self, message: str, level: str = 'INFO', _from_handler: bool = False):
        """
        Schreibt eine Nachricht in den Logger.

        Args:
            message: Die Log-Nachricht
            level: Log-Level (TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR)
            _from_handler: True wenn der Aufruf vom GUILogHandler kommt (intern)
        """
        timestamp = datetime.now().strftime('%H:%M:%S')

        # Farben nach Level
        colors = {
            'TRACE': '#888888',
            'DEBUG': '#b19cd9',
            'INFO': '#90cdf4',
            'SUCCESS': '#68d391',
            'WARNING': '#fbd38d',
            'ERROR': '#fc8181',
            'CRITICAL': '#ff6b6b',
            'TIMING': '#80cbc4',
        }

        bg_colors = {
            'TRACE': '#2d2d2d',
            'DEBUG': '#3d3d5c',
            'INFO': '#2d3748',
            'SUCCESS': '#22543d',
            'WARNING': '#744210',
            'ERROR': '#742a2a',
            'CRITICAL': '#5c1a1a',
            'TIMING': '#1a3d3d',
        }

        color = colors.get(level, '#cccccc')
        bg = bg_colors.get(level, '#2d2d2d')

        html = f'''<div style="background-color: {bg}; padding: 2px 5px; margin: 1px 0; border-radius: 3px;">
            <span style="color: #a0aec0;">[{timestamp}]</span>
            <span style="color: {color}; font-weight: bold;">[{level}]</span>
            <span style="color: {color};">{message}</span>
        </div>'''

        self.log_text.append(html)

        # Auto-scroll
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

        # NUR an den echten Logger weiterleiten, wenn der Aufruf NICHT vom GUILogHandler kommt
        # (sonst wurde die Meldung bereits geloggt und wuerde doppelt erscheinen)
        if not _from_handler:
            level_map = {
                'TRACE': Logger.TRACE,
                'DEBUG': logging.DEBUG,
                'INFO': logging.INFO,
                'SUCCESS': Logger.SUCCESS,
                'WARNING': logging.WARNING,
                'ERROR': logging.ERROR,
                'CRITICAL': logging.CRITICAL,
            }

            # Temporaer den GUI-Handler deaktivieren um Endlosschleife zu vermeiden
            if self._gui_handler:
                self._gui_handler.setLevel(logging.CRITICAL + 1)

            # An echten Logger senden
            log_level = level_map.get(level, logging.INFO)
            self.logger._logger.log(log_level, message)

            # GUI-Handler wieder aktivieren
            if self._gui_handler:
                self._gui_handler.setLevel(Logger.TRACE)

    def _setup_gui_log_handler(self):
        """Registriert den GUI-Handler beim Logger."""
        self._gui_handler = GUILogHandler(self._log)
        # Formatter ohne Timestamp (wird in _log hinzugefuegt)
        formatter = logging.Formatter('%(message)s')
        self._gui_handler.setFormatter(formatter)
        # Handler zum internen Logger hinzufuegen
        self.logger._logger.addHandler(self._gui_handler)

    def _update_log_level(self, level: int):
        """Aktualisiert das Log-Level (1=ERROR bis 5=TRACE)."""
        level_map = {
            1: logging.ERROR,
            2: logging.WARNING,
            3: Logger.SUCCESS,
            4: logging.DEBUG,
            5: Logger.TRACE,
        }
        self.log_level = level
        if self._gui_handler:
            self._gui_handler.setLevel(level_map.get(level, Logger.TRACE))

    # === Event Handler ===

    def _auto_load_last_data(self):
        """Laedt automatisch die letzte CSV-Datei."""
        try:
            reader = CSVReader(self.config.paths.data_dir)
            df, filepath = reader.load_last_csv()

            if df is not None:
                self.data = df
                self.data_path = filepath
                self.data_loaded.emit(df)
                self._log(f'Auto-Load: {filepath.name}', 'SUCCESS')
        except Exception as e:
            self._log(f'Auto-Load fehlgeschlagen: {e}', 'WARNING')

    def _load_csv(self):
        """Oeffnet Dialog zum Laden einer CSV-Datei."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, 'CSV-Datei laden',
            str(self.config.paths.data_dir),
            'CSV Dateien (*.csv)'
        )

        if filepath:
            try:
                reader = CSVReader()
                df = reader.read(Path(filepath))

                if df is not None:
                    self.data = df
                    self.data_path = Path(filepath)
                    reader.log_data_info(df, self.data_path)
                    self.data_loaded.emit(df)
                    self._log(f'CSV geladen: {Path(filepath).name}', 'SUCCESS')
                else:
                    QMessageBox.warning(self, 'Fehler', 'Konnte CSV nicht laden')
            except Exception as e:
                self._log(f'CSV-Fehler: {e}', 'ERROR')
                QMessageBox.warning(self, 'Fehler', f'Konnte CSV nicht laden: {e}')

    def _download_data(self):
        """Startet Binance Download."""
        symbol = self.symbol_combo.currentText()
        from_date = self.from_date.date().toPyDate()
        to_date = self.to_date.date().toPyDate()
        interval = self.interval_combo.currentText()

        # Datum als String formatieren (YYYY-MM-DD)
        from_date_str = from_date.strftime('%Y-%m-%d')
        to_date_str = to_date.strftime('%Y-%m-%d')

        self._log(f'Download gestartet: {symbol} {from_date_str} bis {to_date_str}, Intervall: {interval}', 'INFO')
        self._log(f'Zielverzeichnis: {self.config.paths.data_dir}', 'DEBUG')

        try:
            from ..data.downloader import BinanceDownloader

            # Symbol wird im Konstruktor uebergeben
            downloader = BinanceDownloader(
                symbol=symbol,
                data_dir=self.config.paths.data_dir
            )

            # Download mit String-Datums
            df = downloader.download(
                start_date=from_date_str,
                end_date=to_date_str,
                interval=interval,
                save=True
            )

            if df is not None:
                self.data = df
                # Dateipfad rekonstruieren
                symbol_short = symbol.replace('USDT', 'USD')
                filename = f'{symbol_short}_{from_date_str}_{to_date_str}.csv'
                self.data_path = self.config.paths.data_dir / filename
                self.data_loaded.emit(df)
                self._log(f'Download erfolgreich: {len(df)} Datensaetze', 'SUCCESS')
                self._update_data_status()
            else:
                self._log('Download fehlgeschlagen - keine Daten erhalten', 'ERROR')
        except ImportError as e:
            self._log(f'Modul nicht installiert: {e}', 'ERROR')
            self._log('Hinweis: pip install python-binance', 'WARNING')
            QMessageBox.warning(self, 'Fehler', f'python-binance nicht installiert:\npip install python-binance')
        except Exception as e:
            self._log(f'Download-Fehler: {e}', 'ERROR')
            import traceback
            self._log(f'Details: {traceback.format_exc()}', 'DEBUG')
            QMessageBox.warning(self, 'Fehler', f'Download fehlgeschlagen: {e}')

    def _analyze_data(self):
        """Analysiert die geladenen Daten."""
        if self.data is None:
            return

        self._log('Datenanalyse gestartet...', 'INFO')

        # Grundlegende Statistiken
        df = self.data
        self._log(f'Zeitraum: {df.index[0]} bis {df.index[-1]}', 'INFO')
        self._log(f'Datenpunkte: {len(df):,}', 'INFO')

        # Preis-Statistiken
        if 'Close' in df.columns:
            close = df['Close']
            self._log(f'Preis Min: ${close.min():,.2f}', 'INFO')
            self._log(f'Preis Max: ${close.max():,.2f}', 'INFO')
            self._log(f'Preis Mean: ${close.mean():,.2f}', 'INFO')

            # Volatilitaet
            returns = close.pct_change().dropna()
            volatility = returns.std() * np.sqrt(365 * 24)  # Annualisiert
            self._log(f'Volatilitaet (ann.): {volatility:.1%}', 'INFO')

        # Fehlende Werte
        missing = df.isnull().sum().sum()
        if missing > 0:
            self._log(f'Fehlende Werte: {missing}', 'WARNING')
        else:
            self._log('Keine fehlenden Werte', 'SUCCESS')

        self._log('Analyse abgeschlossen', 'SUCCESS')

    def _prepare_training_data(self):
        """Oeffnet das Trainingsdaten-Vorbereitungsfenster."""
        if self.data is None:
            QMessageBox.warning(self, 'Fehler', 'Bitte zuerst Daten laden')
            return

        self._log('Oeffne Trainingsdaten-Vorbereitung...', 'INFO')

        try:
            from .prepare_data_window import PrepareDataWindow
            self.prepare_window = PrepareDataWindow(self.data, self)
            self.prepare_window.data_prepared.connect(self._on_training_data_prepared)
            self.prepare_window.show()
        except ImportError:
            self._log('PrepareDataWindow noch nicht implementiert', 'WARNING')
            QMessageBox.information(self, 'Info', 'Trainingsdaten-Vorbereitung noch nicht implementiert')

    def _on_training_data_prepared(self, training_data, training_info, backtest_info):
        """Callback wenn Trainingsdaten vorbereitet wurden."""
        self.training_data = training_data
        self.training_info = training_info
        self.backtest_info = backtest_info  # Separater Backtest-Datensatz
        self.training_data_ready.emit(training_data)
        self._log('Trainingsdaten bereit', 'SUCCESS')

        # Backtest-Info loggen
        if backtest_info:
            backtest_points = backtest_info.get('num_points', 0)
            self._log(f'Backtest-Daten reserviert: {backtest_points} Datenpunkte', 'INFO')

        # Status-Panel aktualisieren
        self._update_status_panel_from_training(training_data, training_info)

    def _visualize_signals(self):
        """Oeffnet das Signal-Visualisierungsfenster."""
        if self.training_data is None:
            QMessageBox.warning(self, 'Fehler', 'Bitte zuerst Trainingsdaten vorbereiten')
            return

        self._log('Oeffne Signal-Visualisierung...', 'INFO')

        try:
            from .visualize_data_window import VisualizeDataWindow
            self.visualize_window = VisualizeDataWindow(
                self.data, self.training_data, self.training_info, self
            )
            self.visualize_window.show()
        except ImportError:
            self._log('VisualizeDataWindow noch nicht implementiert', 'WARNING')
            QMessageBox.information(self, 'Info', 'Signal-Visualisierung noch nicht implementiert')

    def _load_training_data(self):
        """Laedt Trainingsdaten aus Datei."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, 'Trainingsdaten laden',
            str(self.config.paths.results_dir),
            'NPZ Dateien (*.npz);;MAT Dateien (*.mat)'
        )

        if filepath:
            self._log(f'Lade Trainingsdaten: {Path(filepath).name}', 'INFO')
            try:
                if filepath.endswith('.npz'):
                    data = np.load(filepath, allow_pickle=True)
                    self.training_data = {
                        'X_train': data.get('X_train'),
                        'y_train': data.get('y_train'),
                        'X_val': data.get('X_val'),
                        'y_val': data.get('y_val'),
                    }
                    # Info extrahieren falls vorhanden
                    if 'info' in data:
                        self.training_info = data['info'].item()
                    else:
                        self.training_info = {'source': filepath}

                    self._log(f'Training Samples: {len(self.training_data["X_train"]):,}', 'INFO')
                    self._log('Trainingsdaten geladen', 'SUCCESS')
                    self.training_data_ready.emit(self.training_data)
                else:
                    self._log(f'Unterstuetztes Format: .npz', 'WARNING')
            except Exception as e:
                self._log(f'Fehler beim Laden: {e}', 'ERROR')

    def _open_training_gui(self):
        """Oeffnet das Training-GUI Fenster."""
        if self.training_data is None:
            QMessageBox.warning(self, 'Fehler', 'Bitte zuerst Trainingsdaten vorbereiten')
            return

        self._log('Oeffne Training GUI...', 'INFO')

        try:
            from .training_window import TrainingWindow
            self.training_window = TrainingWindow(self)
            # Trainingsdaten und Info an das Fenster uebergeben
            self.training_window.training_data = self.training_data
            self.training_window.training_info = self.training_info
            # DataLoader aus training_data erstellen
            self.training_window.prepare_data_loaders(self.training_data)
            self.training_window.training_completed.connect(self._on_training_completed)
            self.training_window.show()
        except Exception as e:
            self._log(f'Training GUI Fehler: {e}', 'ERROR')

    def _on_training_completed(self, model, results):
        """Callback wenn Training abgeschlossen."""
        self.model = model
        self.model_loaded.emit(model)

        # model_info aus results und training_info erstellen
        training_info = self.training_info or {}
        self.model_info = {
            'model_type': results.get('model_type', 'bilstm'),
            'input_size': training_info.get('num_features', len(training_info.get('features', []))),
            'hidden_size': results.get('hidden_size', 100),
            'num_layers': results.get('num_layers', 2),
            'num_classes': training_info.get('num_classes', 2),
            'sequence_length': training_info.get('params', {}).get('lookback', 100),
            'features': training_info.get('features', []),
            'accuracy': results.get('best_accuracy', 0),
        }

        # backtest_info sollte bereits durch _on_training_data_prepared gesetzt sein
        if self.backtest_info and 'data' in self.backtest_info:
            num_points = len(self.backtest_info['data'])
            self._log(f"Backtest-Daten verfuegbar: {num_points} Punkte", 'INFO')

        self._log('Training abgeschlossen', 'SUCCESS')

    def _load_model(self):
        """Laedt ein gespeichertes Modell."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, 'Modell laden',
            str(self.config.paths.results_dir),
            'PyTorch Modelle (*.pt *.pth)'
        )

        if filepath:
            self._log(f'Lade Modell: {Path(filepath).name}', 'INFO')
            try:
                import torch
                from ..models.factory import ModelFactory

                checkpoint = torch.load(filepath, map_location='cpu')
                self.model_path = Path(filepath)

                # Model-Info aus Checkpoint extrahieren
                if 'model_info' in checkpoint:
                    self.model_info = checkpoint['model_info']
                    model_type = self.model_info.get('model_type', 'bilstm')
                    input_size = self.model_info.get('input_size', 6)
                    hidden_size = self.model_info.get('hidden_size', 100)
                    num_layers = self.model_info.get('num_layers', 2)
                    num_classes = self.model_info.get('num_classes', 3)

                    # Modell erstellen und Gewichte laden
                    self.model = ModelFactory.create(
                        model_type,
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        num_classes=num_classes
                    )
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.model.eval()

                    self._log(f'Modell: {model_type.upper()}', 'INFO')
                    self._log(f'Parameter: {self.model.count_parameters():,}', 'INFO')
                else:
                    self._log('Checkpoint ohne model_info - manuelle Konfiguration erforderlich', 'WARNING')

                self.model_loaded.emit(self.model)
                self._log('Modell geladen', 'SUCCESS')
            except Exception as e:
                self._log(f'Modell-Ladefehler: {e}', 'ERROR')

    def _load_last_model(self):
        """Laedt das zuletzt verwendete Modell."""
        self._log('Suche letztes Modell...', 'INFO')

        # Suche in results und models Verzeichnissen
        search_dirs = [self.config.paths.results_dir, self.config.paths.models_dir]
        model_files = []

        for search_dir in search_dirs:
            if search_dir.exists():
                model_files.extend(search_dir.rglob('*.pt'))
                model_files.extend(search_dir.rglob('*.pth'))

        if model_files:
            # Nach Aenderungszeit sortieren (neueste zuerst)
            model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            latest = model_files[0]
            self._log(f'Gefunden: {latest.name}', 'INFO')

            # Modell laden
            import torch
            try:
                checkpoint = torch.load(latest, map_location='cpu')
                self.model_path = latest
                self._log('Letztes Modell geladen', 'SUCCESS')
            except Exception as e:
                self._log(f'Fehler: {e}', 'ERROR')
        else:
            self._log('Kein vorheriges Modell gefunden', 'WARNING')

    def _load_session(self):
        """Laedt eine komplette Session (Daten + Modell)."""
        from PyQt6.QtWidgets import QFileDialog
        from .session_loader_dialog import SessionLoaderDialog
        from ..core.session_manager import SessionManager

        # Debug: Sessions auflisten
        log_dir = self.config.paths.log_dir
        self._log(f"Suche Sessions in: {log_dir}", 'DEBUG')
        sessions = SessionManager.list_sessions(log_dir)
        self._log(f"Gefundene Sessions: {len(sessions)}", 'DEBUG')

        dialog = SessionLoaderDialog(log_dir, parent=self)
        if dialog.exec():
            session_dir = dialog.get_selected_session()
            if session_dir:
                self._load_session_from_dir(session_dir)

    def _load_session_from_dir(self, session_dir):
        """Laedt Session-Daten aus einem Ordner."""
        from pathlib import Path
        from ..core.session_manager import SessionManager

        try:
            session_dir = Path(session_dir)
            manager = SessionManager(session_dir)

            self._log(f"Lade Session: {session_dir.name}", 'INFO')

            # 1. Backtest-Daten laden
            backtest_data = manager.load_backtest_data()
            if backtest_data is not None:
                self.backtest_info = {'data': backtest_data}
                self._log(f"Backtest-Daten geladen: {len(backtest_data)} Punkte", 'SUCCESS')

            # 2. Modell laden
            model_path = manager.get_model_path()
            if model_path:
                import torch
                from ..models import ModelFactory

                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

                # Model-Info extrahieren
                model_info = checkpoint.get('model_info', {})
                self.model_info = model_info

                # Modell rekonstruieren
                model_name = model_info.get('model_type', 'bilstm')
                self.model = ModelFactory.create(
                    model_name,
                    input_size=model_info.get('input_size', 6),
                    hidden_size=model_info.get('hidden_size', 128),
                    num_layers=model_info.get('num_layers', 2),
                    num_classes=model_info.get('num_classes', 3),
                    dropout=model_info.get('dropout', 0.2)
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model_path = model_path

                self._log(f"Modell geladen: {model_path.name}", 'SUCCESS')
                self._log(f"  Accuracy: {model_info.get('best_accuracy', 0):.1f}%", 'INFO')

                # UI aktualisieren
                self.model_name_label.setText(f'Modell: {model_path.name}')
                self.model_folder_label.setText(f'Session: {session_dir.name}')
                self.backtest_btn.setEnabled(True)
                self.predict_btn.setEnabled(True)

            # 3. Config laden
            config = manager.load_config()
            if config:
                self._log(f"Config: {len(config.get('features', []))} Features", 'DEBUG')

            self._log(f"Session geladen: {session_dir.name}", 'SUCCESS')

        except Exception as e:
            import traceback
            self._log(f"Session-Ladefehler: {e}", 'ERROR')
            self._log(traceback.format_exc(), 'ERROR')

    def _auto_load_latest_session(self):
        """Laedt automatisch die neueste Session mit Modell."""
        from ..core.session_manager import SessionManager

        try:
            sessions = SessionManager.list_sessions(self.config.paths.log_dir)
            self._log(f"Suche Sessions in: {self.config.paths.log_dir}", 'DEBUG')
            self._log(f"Gefundene Sessions: {len(sessions)}", 'DEBUG')

            # Finde neueste Session mit Modell
            for session in sessions:
                if session.get('has_model'):
                    session_dir = session['session_dir']
                    self._log(f"Auto-Load Session: {session['session_name']}", 'INFO')
                    self._load_session_from_dir(session_dir)
                    return

            self._log("Keine Session mit Modell gefunden", 'INFO')

        except Exception as e:
            self._log(f"Auto-Load Session fehlgeschlagen: {e}", 'WARNING')

    def _make_prediction(self):
        """Fuehrt eine Vorhersage durch."""
        if self.model is None:
            QMessageBox.warning(self, 'Fehler', 'Bitte zuerst Modell laden')
            return

        if self.training_data is None:
            QMessageBox.warning(self, 'Fehler', 'Bitte zuerst Trainingsdaten vorbereiten')
            return

        self._log('Vorhersage gestartet...', 'INFO')

        try:
            import torch

            self.model.eval()
            X_val = self.training_data.get('X_val')

            if X_val is None:
                self._log('Keine Validierungsdaten vorhanden', 'WARNING')
                return

            # Tensor konvertieren falls noetig
            if not isinstance(X_val, torch.Tensor):
                X_val = torch.FloatTensor(X_val)

            with torch.no_grad():
                predictions = self.model.predict(X_val)
                probabilities = self.model.predict_proba(X_val)

            # Statistiken
            unique, counts = np.unique(predictions, return_counts=True)
            class_names = self.config.training.class_names

            self._log('Vorhersage-Verteilung:', 'INFO')
            for cls, count in zip(unique, counts):
                pct = count / len(predictions) * 100
                name = class_names[cls] if cls < len(class_names) else f'Klasse {cls}'
                self._log(f'  {name}: {count:,} ({pct:.1f}%)', 'INFO')

            self._log('Vorhersage abgeschlossen', 'SUCCESS')
        except Exception as e:
            self._log(f'Vorhersage-Fehler: {e}', 'ERROR')

    def _open_backtester(self):
        """Oeffnet das Backtester-Fenster."""
        # Auto-Load: Neueste Session laden wenn kein Modell/Backtest-Daten vorhanden
        if self.model is None or self.backtest_info is None:
            self._auto_load_latest_session()

        # Pruefen ob Daten vorhanden
        if self.data is None:
            QMessageBox.warning(self, 'Fehler', 'Bitte zuerst Daten laden')
            return

        # Warnung wenn kein Modell, aber trotzdem fortfahren (fuer Chart-Ansicht)
        if self.model is None:
            self._log('Kein Modell geladen - Backtester ohne Vorhersagen', 'WARNING')

        self._log('Oeffne Backtester...', 'INFO')

        try:
            from .backtest_window import BacktestWindow
            self.backtest_window = BacktestWindow(parent=self)

            # Backtest-Daten verwenden falls verfuegbar (nicht im Training verwendet)
            if self.backtest_info and 'data' in self.backtest_info:
                backtest_data = self.backtest_info['data']
                self._log(f'Verwende separate Backtest-Daten: {len(backtest_data)} Punkte', 'INFO')
            else:
                backtest_data = self.data
                self._log('Keine separaten Backtest-Daten - verwende alle Daten', 'WARNING')

            self.backtest_window.set_data(
                data=backtest_data,
                model=self.model,
                model_info=self.model_info
            )
            self.backtest_window.show()
        except Exception as e:
            self._log(f'Backtester Fehler: {e}', 'ERROR')

    def _open_backtrader(self):
        """Oeffnet das Backtrader Pro Fenster."""
        # Auto-Load: Neueste Session laden wenn kein Modell vorhanden
        if self.model is None:
            self._auto_load_latest_session()

        # Pruefen ob Daten vorhanden
        if self.data is None:
            QMessageBox.warning(self, 'Fehler', 'Bitte zuerst Daten laden')
            return

        self._log('Oeffne Backtrader Pro...', 'INFO')

        try:
            from .backtrader_window import BacktraderWindow
            self.backtrader_window = BacktraderWindow(parent=self)

            # Backtest-Daten verwenden falls verfuegbar
            if self.backtest_info and 'data' in self.backtest_info:
                backtest_data = self.backtest_info['data']
                self._log(f'Verwende separate Backtest-Daten: {len(backtest_data)} Punkte', 'INFO')
            else:
                backtest_data = self.data

            self.backtrader_window.set_data(
                data=backtest_data,
                model=self.model,
                model_info=self.model_info
            )
            self.backtrader_window.show()
        except Exception as e:
            self._log(f'Backtrader Fehler: {e}', 'ERROR')
            import traceback
            traceback.print_exc()

    def _open_walk_forward(self):
        """Oeffnet das Walk-Forward Analyse Fenster."""
        # Auto-Load: Neueste Session laden wenn kein Modell vorhanden
        if self.model is None:
            self._auto_load_latest_session()

        # Pruefen ob Daten vorhanden
        if self.data is None:
            QMessageBox.warning(self, 'Fehler', 'Bitte zuerst Daten laden')
            return

        self._log('Oeffne Walk-Forward Analyse...', 'INFO')

        try:
            from .walk_forward_window import WalkForwardWindow
            self.walk_forward_window = WalkForwardWindow(parent=self)

            # Backtest-Daten verwenden falls verfuegbar
            if self.backtest_info and 'data' in self.backtest_info:
                backtest_data = self.backtest_info['data']
                self._log(f'Verwende separate Backtest-Daten: {len(backtest_data)} Punkte', 'INFO')
            else:
                backtest_data = self.data

            # Training-Config aus training_info erstellen
            training_config = {}
            if self.training_info:
                training_config = {
                    'lookback': self.training_info.get('params', {}).get('lookback', 50),
                    'lookforward': self.training_info.get('params', {}).get('lookforward', 100),
                    'features': self.training_info.get('features', []),
                    'hidden_size': self.model_info.get('hidden_size', 128) if self.model_info else 128,
                    'num_layers': self.model_info.get('num_layers', 2) if self.model_info else 2,
                }

            self.walk_forward_window.set_data(
                data=backtest_data,
                model=self.model,
                model_info=self.model_info,
                training_config=training_config
            )
            self.walk_forward_window.show()
        except Exception as e:
            self._log(f'Walk-Forward Fehler: {e}', 'ERROR')
            import traceback
            traceback.print_exc()

    def _open_trading(self):
        """Oeffnet das Trading-Fenster."""
        self._log('Oeffne Live Trading...', 'INFO')

        try:
            from .trading_window import TradingWindow
            self.trading_window = TradingWindow(self)
            self.trading_window.show()
        except Exception as e:
            self._log(f'Trading Fenster Fehler: {e}', 'ERROR')

    def _open_webserver(self):
        """Oeffnet das Web Dashboard."""
        self._log('Web Dashboard noch nicht integriert', 'INFO')
        QMessageBox.information(self, 'Info', 'Web Dashboard kommt in einer spaeteren Version')

    def _save_parameters(self):
        """Speichert die aktuellen Parameter."""
        self._log('Parameter speichern...', 'INFO')

        filepath, _ = QFileDialog.getSaveFileName(
            self, 'Parameter speichern',
            str(self.config.paths.base_dir / 'parameters.json'),
            'JSON Dateien (*.json)'
        )

        if filepath:
            try:
                params = {
                    'training': {
                        'lookback': self.config.training.lookback,
                        'lookforward': self.config.training.lookforward,
                        'epochs': self.config.training.epochs,
                        'batch_size': self.config.training.batch_size,
                        'learning_rate': self.config.training.learning_rate,
                        'hidden_size': self.config.training.hidden_size,
                        'num_layers': self.config.training.num_layers,
                        'dropout': self.config.training.dropout,
                        'features': self.config.training.features,
                    },
                    'backtest': {
                        'initial_capital': self.config.backtest.initial_capital,
                        'commission': self.config.backtest.commission,
                        'slippage': self.config.backtest.slippage,
                    },
                    'saved_at': datetime.now().isoformat()
                }

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(params, f, indent=2, ensure_ascii=False)

                self._log(f'Parameter gespeichert: {Path(filepath).name}', 'SUCCESS')
            except Exception as e:
                self._log(f'Speicherfehler: {e}', 'ERROR')

    def _load_parameters(self):
        """Laedt gespeicherte Parameter."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, 'Parameter laden',
            str(self.config.paths.base_dir),
            'JSON Dateien (*.json)'
        )

        if filepath:
            self._log(f'Lade Parameter: {Path(filepath).name}', 'INFO')
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    params = json.load(f)

                # Training-Parameter uebernehmen
                if 'training' in params:
                    t = params['training']
                    self.config.training.lookback = t.get('lookback', self.config.training.lookback)
                    self.config.training.lookforward = t.get('lookforward', self.config.training.lookforward)
                    self.config.training.epochs = t.get('epochs', self.config.training.epochs)
                    self.config.training.batch_size = t.get('batch_size', self.config.training.batch_size)
                    self.config.training.learning_rate = t.get('learning_rate', self.config.training.learning_rate)
                    self.config.training.hidden_size = t.get('hidden_size', self.config.training.hidden_size)
                    self.config.training.num_layers = t.get('num_layers', self.config.training.num_layers)
                    self.config.training.dropout = t.get('dropout', self.config.training.dropout)
                    if 'features' in t:
                        self.config.training.features = t['features']

                # Backtest-Parameter uebernehmen
                if 'backtest' in params:
                    b = params['backtest']
                    self.config.backtest.initial_capital = b.get('initial_capital', self.config.backtest.initial_capital)
                    self.config.backtest.commission = b.get('commission', self.config.backtest.commission)
                    self.config.backtest.slippage = b.get('slippage', self.config.backtest.slippage)

                self._log('Parameter geladen', 'SUCCESS')
            except Exception as e:
                self._log(f'Ladefehler: {e}', 'ERROR')

    def _update_logger_mode(self, index):
        """Aktualisiert den Logger-Modus."""
        modes = ['window', 'both', 'file']
        self.logger_mode = modes[index]
        self._log(f'Logger-Modus: {self.logger_mode_combo.currentText()}', 'DEBUG')

    def _update_logger_level(self, index):
        """Aktualisiert das Log-Level."""
        self.log_level = index + 1
        self._log(f'Log-Level: {self.logger_level_combo.currentText()}', 'DEBUG')

    def _clear_log(self):
        """Leert den Log-Text."""
        self.log_text.clear()

    def _update_timing(self, state):
        """Aktualisiert die Timing-Einstellung."""
        self.enable_timing = state == Qt.CheckState.Checked.value

    def _update_log_font_size(self, size: int):
        """Aktualisiert die Log-Schriftgroesse."""
        self._log_font_size = size
        self._update_log_stylesheet()

    def _update_log_stylesheet(self):
        """Aktualisiert das Stylesheet des Log-Widgets mit aktueller Schriftgroesse."""
        self.log_text.setStyleSheet(f'''
            QTextEdit {{
                background-color: #1a1a1a;
                color: #cccccc;
                border: 1px solid #333333;
                border-radius: 4px;
                font-family: Consolas, monospace;
                font-size: {self._log_font_size}pt;
            }}
        ''')

    def _update_gpu_status(self):
        """Aktualisiert den GPU-Status."""
        try:
            from ..utils.helpers import get_gpu_info
            info = get_gpu_info()

            if info['cuda_available']:
                device = info['devices'][0]
                self.gpu_indicator.setText(f"GPU: {device['name']}")
                self.gpu_indicator.setStyleSheet('color: #68d391; font-weight: bold;')
                self._log(f"GPU verfuegbar: {device['name']} ({device['total_memory_gb']:.1f} GB)", 'SUCCESS')
            else:
                self.gpu_indicator.setText('CPU')
                self.gpu_indicator.setStyleSheet('color: #fbd38d; font-weight: bold;')
                self._log('Keine GPU verfuegbar, verwende CPU', 'WARNING')
        except Exception as e:
            self.gpu_indicator.setText('GPU: ?')
            self._log(f'GPU-Status Fehler: {e}', 'ERROR')

    def _show_about(self):
        """Zeigt About-Dialog."""
        QMessageBox.about(
            self,
            'Ueber BTCUSD Analyzer',
            'BTCUSD Analyzer v0.1.0\n\n'
            'BILSTM Neural Network fuer BTC Trendwechsel-Erkennung\n\n'
            'PyTorch + PyQt6\n\n'
            'Portiert von MATLAB'
        )

    # === Status-Panel Update ===

    def _update_info_label(self, attr_name: str, value: str, color: str = None):
        """
        Aktualisiert ein Info-Label im Status-Panel.

        Args:
            attr_name: Name des Attributs (z.B. 'df_rows')
            value: Anzuzeigender Wert
            color: Optionale Farbe (hex)
        """
        label = getattr(self, f'info_{attr_name}', None)
        if label:
            label.setText(value)
            if color:
                label.setStyleSheet(f'color: {color};')
            else:
                label.setStyleSheet('color: #cccccc;')

    def _update_status_panel_from_data(self, df: pd.DataFrame):
        """Aktualisiert das Status-Panel mit DataFrame-Infos."""
        if df is None:
            return

        # Status
        self._update_info_label('status_raw', '✅ Geladen', COLORS['success'])
        self._update_info_label('status_pipeline', '⏳ Daten geladen', COLORS['warning'])

        # DataFrame Info
        count = len(df)
        self._update_info_label('df_rows', f'{count:,}')

        # Zeitraum
        try:
            if hasattr(df.index, 'min') and hasattr(df.index, 'max'):
                start = df.index.min()
                end = df.index.max()
                if hasattr(start, 'strftime'):
                    self._update_info_label('df_period', f'{start.strftime("%d.%m.%y")} - {end.strftime("%d.%m.%y")}')
                    # Intervall berechnen
                    if len(df) > 1:
                        delta = (df.index[1] - df.index[0]).total_seconds()
                        if delta < 60:
                            interval = f'{int(delta)}s'
                        elif delta < 3600:
                            interval = f'{int(delta/60)}m'
                        elif delta < 86400:
                            interval = f'{int(delta/3600)}h'
                        else:
                            interval = f'{int(delta/86400)}d'
                        self._update_info_label('df_interval', interval)
        except Exception:
            pass

        # Spalten
        self._update_info_label('df_columns', str(len(df.columns)))

        # Speicher
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        self._update_info_label('df_memory', f'{memory_mb:.2f} MB')

    def _update_status_panel_from_training(self, training_data: dict, training_info: dict):
        """Aktualisiert das Status-Panel mit Trainingsdaten-Infos."""
        if training_info is None:
            return

        # Status aktualisieren
        self._update_info_label('status_labels', '✅ Generiert', COLORS['success'])
        self._update_info_label('status_sequences', '✅ Erstellt', COLORS['success'])
        self._update_info_label('status_pipeline', '✅ Bereit', COLORS['success'])

        # Sequenz-Parameter
        if 'params' in training_info:
            params = training_info['params']
            self._update_info_label('seq_lookback', str(params.get('lookback', '-')))
            self._update_info_label('seq_lookforward', str(params.get('lookforward', '-')))

        # Features und Gesamt
        self._update_info_label('seq_features', str(training_info.get('num_features', '-')))
        self._update_info_label('seq_total', f"{training_info.get('total', 0):,}")

        # Label-Verteilung
        num_buy = training_info.get('num_buy', 0)
        num_sell = training_info.get('num_sell', 0)
        num_hold = training_info.get('num_hold', 0)
        total = num_buy + num_sell + num_hold

        self._update_info_label('label_buy', f'{num_buy:,} ({100*num_buy/total:.1f}%)' if total > 0 else '-', COLORS['success'])
        self._update_info_label('label_sell', f'{num_sell:,} ({100*num_sell/total:.1f}%)' if total > 0 else '-', COLORS['error'])
        self._update_info_label('label_hold', f'{num_hold:,} ({100*num_hold/total:.1f}%)' if total > 0 else '-')

        # Balance
        if num_buy > 0 and num_sell > 0:
            ratio = max(num_buy, num_sell) / min(num_buy, num_sell)
            if ratio < 1.5:
                balance_text = f'✅ Gut ({ratio:.2f}:1)'
                balance_color = COLORS['success']
            elif ratio < 3.0:
                balance_text = f'⚠ Maessig ({ratio:.2f}:1)'
                balance_color = COLORS['warning']
            else:
                balance_text = f'❌ Unbalanciert ({ratio:.2f}:1)'
                balance_color = COLORS['error']
            self._update_info_label('label_balance', balance_text, balance_color)

        # Train/Val Split (80/20 default)
        if total > 0:
            train_count = int(total * 0.8)
            val_count = total - train_count
            self._update_info_label('split_train', f'{train_count:,}')
            self._update_info_label('split_val', f'{val_count:,}')
            self._update_info_label('split_ratio', '80 / 20 %')

    def _reset_status_panel(self):
        """Setzt alle Status-Panel Felder zurueck."""
        default_fields = [
            'status_pipeline', 'status_raw', 'status_labels', 'status_sequences',
            'df_rows', 'df_period', 'df_interval', 'df_columns', 'df_memory',
            'seq_lookback', 'seq_lookforward', 'seq_features', 'seq_total',
            'label_buy', 'label_sell', 'label_hold', 'label_balance',
            'split_train', 'split_val', 'split_ratio'
        ]
        for field in default_fields:
            self._update_info_label(field, '-')
        self._update_info_label('status_pipeline', '⬜ Ausstehend')

    # === Signal Handler ===

    def _on_data_loaded(self, df: pd.DataFrame):
        """Wird aufgerufen wenn Daten geladen wurden."""
        count = len(df)

        self.status_label.setText(f'Daten geladen: {count:,} Datensaetze')

        # Status-Anzeige aktualisieren
        self._update_data_status()

        # Status-Panel aktualisieren
        self._update_status_panel_from_data(df)

        # Buttons aktivieren
        self.analyze_btn.setEnabled(True)
        self.prepare_btn.setEnabled(True)

    def _on_model_loaded(self, model):
        """Wird aufgerufen wenn ein Modell geladen wurde."""
        if hasattr(model, 'name'):
            self.model_name_label.setText(f'Modell: {model.name}')

        if self.model_path:
            self.model_folder_label.setText(f'Ordner: {self.model_path.parent.name}')

        self.predict_btn.setEnabled(True)
        self.backtest_btn.setEnabled(True)
        self.status_label.setText('Modell bereit')

    def _on_training_data_ready(self, training_data):
        """Wird aufgerufen wenn Trainingsdaten bereit sind."""
        self.train_gui_btn.setEnabled(True)
        self.visualize_btn.setEnabled(True)
        self.status_label.setText('Trainingsdaten bereit')

    def closeEvent(self, event):
        """Wird beim Schliessen des Fensters aufgerufen."""
        # GUI-Handler entfernen bevor Logger-Aufruf
        if self._gui_handler:
            self.logger._logger.removeHandler(self._gui_handler)
            self._gui_handler = None
        self.logger.info('BTCUSD Analyzer beendet')
        event.accept()
