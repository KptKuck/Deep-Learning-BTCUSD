"""
Main Window - Haupt-GUI des BTCUSD Analyzers
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox, QTabWidget, QStatusBar,
    QFileDialog, QMessageBox, QProgressBar, QComboBox, QSpinBox,
    QDoubleSpinBox, QSplitter, QFrame, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QAction

import pandas as pd

from ..core.config import Config
from ..core.logger import get_logger
from ..data.reader import CSVReader
from ..data.processor import FeatureProcessor
from ..models import ModelFactory
from .styles import get_stylesheet, COLORS


class MainWindow(QMainWindow):
    """
    Haupt-Fenster des BTCUSD Analyzers.

    Enthaelt:
    - Daten-Panel (CSV laden, Binance Download)
    - Modell-Panel (Modell laden/erstellen)
    - Training-Tab
    - Backtest-Tab
    - Status-Leiste
    """

    # Signals
    data_loaded = pyqtSignal(object)  # DataFrame
    model_loaded = pyqtSignal(object)  # Model

    def __init__(self, base_dir: Optional[Path] = None):
        super().__init__()

        # Konfiguration
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.config = Config(self.base_dir)
        self.logger = get_logger('btcusd_analyzer', self.config.paths.log_dir)

        # State
        self.data: Optional[pd.DataFrame] = None
        self.data_path: Optional[Path] = None
        self.model = None
        self.model_path: Optional[Path] = None

        # UI initialisieren
        self._init_ui()
        self._connect_signals()

        # Stylesheet anwenden
        self.setStyleSheet(get_stylesheet())

        # Logging
        self.logger.info('BTCUSD Analyzer GUI gestartet')
        self.logger.info(f'Log-Datei: {self.logger.get_log_file_path()}')

        # Auto-Load letzte Daten
        QTimer.singleShot(100, self._auto_load_last_data)

    def _init_ui(self):
        """Initialisiert die UI-Komponenten."""
        self.setWindowTitle('BTCUSD Analyzer')
        self.setMinimumSize(1200, 800)

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main Layout mit Splitter
        main_layout = QHBoxLayout(central_widget)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Linke Seite: Control Panel
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)

        # Rechte Seite: Tab Widget
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)

        # Splitter Proportionen
        splitter.setSizes([350, 850])

        # Status Bar
        self._create_status_bar()

        # Menu Bar
        self._create_menu_bar()

    def _create_left_panel(self) -> QWidget:
        """Erstellt das linke Control-Panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        # Scroll Area fuer langes Panel
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(15)

        # === Daten Panel ===
        data_group = self._create_data_panel()
        scroll_layout.addWidget(data_group)

        # === Modell Panel ===
        model_group = self._create_model_panel()
        scroll_layout.addWidget(model_group)

        # === GPU Status ===
        gpu_group = self._create_gpu_panel()
        scroll_layout.addWidget(gpu_group)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        return panel

    def _create_data_panel(self) -> QGroupBox:
        """Erstellt das Daten-Panel."""
        group = QGroupBox('Daten')
        layout = QVBoxLayout(group)

        # Status Label
        self.data_status_label = QLabel('Keine Daten geladen')
        self.data_status_label.setWordWrap(True)
        layout.addWidget(self.data_status_label)

        # Info Labels
        info_layout = QGridLayout()

        info_layout.addWidget(QLabel('Datensaetze:'), 0, 0)
        self.data_count_label = QLabel('-')
        info_layout.addWidget(self.data_count_label, 0, 1)

        info_layout.addWidget(QLabel('Zeitraum:'), 1, 0)
        self.data_range_label = QLabel('-')
        info_layout.addWidget(self.data_range_label, 1, 1)

        info_layout.addWidget(QLabel('Intervall:'), 2, 0)
        self.data_interval_label = QLabel('-')
        info_layout.addWidget(self.data_interval_label, 2, 1)

        layout.addLayout(info_layout)

        # Buttons
        btn_layout = QHBoxLayout()

        self.load_csv_btn = QPushButton('CSV laden')
        self.load_csv_btn.clicked.connect(self._load_csv)
        btn_layout.addWidget(self.load_csv_btn)

        self.download_btn = QPushButton('Download')
        self.download_btn.clicked.connect(self._download_data)
        btn_layout.addWidget(self.download_btn)

        layout.addLayout(btn_layout)

        return group

    def _create_model_panel(self) -> QGroupBox:
        """Erstellt das Modell-Panel."""
        group = QGroupBox('Modell')
        layout = QVBoxLayout(group)

        # Status Label
        self.model_status_label = QLabel('Kein Modell geladen')
        self.model_status_label.setWordWrap(True)
        layout.addWidget(self.model_status_label)

        # Modell-Auswahl
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel('Architektur:'))

        self.model_combo = QComboBox()
        available_models = ModelFactory.available()
        self.model_combo.addItems([m.upper() for m in available_models])
        model_layout.addWidget(self.model_combo)

        layout.addLayout(model_layout)

        # Parameter
        param_layout = QGridLayout()

        param_layout.addWidget(QLabel('Hidden Size:'), 0, 0)
        self.hidden_size_spin = QSpinBox()
        self.hidden_size_spin.setRange(32, 512)
        self.hidden_size_spin.setValue(self.config.training.hidden_size)
        param_layout.addWidget(self.hidden_size_spin, 0, 1)

        param_layout.addWidget(QLabel('Num Layers:'), 1, 0)
        self.num_layers_spin = QSpinBox()
        self.num_layers_spin.setRange(1, 8)
        self.num_layers_spin.setValue(self.config.training.num_layers)
        param_layout.addWidget(self.num_layers_spin, 1, 1)

        param_layout.addWidget(QLabel('Dropout:'), 2, 0)
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.5)
        self.dropout_spin.setSingleStep(0.05)
        self.dropout_spin.setValue(self.config.training.dropout)
        param_layout.addWidget(self.dropout_spin, 2, 1)

        layout.addLayout(param_layout)

        # Buttons
        btn_layout = QHBoxLayout()

        self.load_model_btn = QPushButton('Modell laden')
        self.load_model_btn.clicked.connect(self._load_model)
        btn_layout.addWidget(self.load_model_btn)

        self.create_model_btn = QPushButton('Neu erstellen')
        self.create_model_btn.clicked.connect(self._create_model)
        btn_layout.addWidget(self.create_model_btn)

        layout.addLayout(btn_layout)

        # Modell Info
        self.model_info_label = QLabel('')
        self.model_info_label.setWordWrap(True)
        self.model_info_label.setStyleSheet(f'color: {COLORS["text_secondary"]}')
        layout.addWidget(self.model_info_label)

        return group

    def _create_gpu_panel(self) -> QGroupBox:
        """Erstellt das GPU-Status Panel."""
        group = QGroupBox('System')
        layout = QVBoxLayout(group)

        # GPU Status
        gpu_layout = QHBoxLayout()
        gpu_layout.addWidget(QLabel('GPU:'))

        self.gpu_status_label = QLabel('Pruefe...')
        gpu_layout.addWidget(self.gpu_status_label)
        gpu_layout.addStretch()

        layout.addLayout(gpu_layout)

        # GPU Info aktualisieren
        QTimer.singleShot(500, self._update_gpu_status)

        return group

    def _create_right_panel(self) -> QWidget:
        """Erstellt das rechte Tab-Panel."""
        self.tab_widget = QTabWidget()

        # Training Tab
        training_tab = self._create_training_tab()
        self.tab_widget.addTab(training_tab, 'Training')

        # Backtest Tab
        backtest_tab = self._create_backtest_tab()
        self.tab_widget.addTab(backtest_tab, 'Backtest')

        # Log Tab
        log_tab = self._create_log_tab()
        self.tab_widget.addTab(log_tab, 'Log')

        return self.tab_widget

    def _create_training_tab(self) -> QWidget:
        """Erstellt den Training-Tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Training Parameter
        param_group = QGroupBox('Training Parameter')
        param_layout = QGridLayout(param_group)

        param_layout.addWidget(QLabel('Epochen:'), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(self.config.training.epochs)
        param_layout.addWidget(self.epochs_spin, 0, 1)

        param_layout.addWidget(QLabel('Batch Size:'), 0, 2)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(8, 256)
        self.batch_size_spin.setValue(self.config.training.batch_size)
        param_layout.addWidget(self.batch_size_spin, 0, 3)

        param_layout.addWidget(QLabel('Learning Rate:'), 1, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setValue(self.config.training.learning_rate)
        param_layout.addWidget(self.lr_spin, 1, 1)

        param_layout.addWidget(QLabel('Lookback:'), 1, 2)
        self.lookback_spin = QSpinBox()
        self.lookback_spin.setRange(10, 200)
        self.lookback_spin.setValue(self.config.training.lookback)
        param_layout.addWidget(self.lookback_spin, 1, 3)

        layout.addWidget(param_group)

        # Progress
        progress_group = QGroupBox('Fortschritt')
        progress_layout = QVBoxLayout(progress_group)

        self.training_progress = QProgressBar()
        self.training_progress.setTextVisible(True)
        progress_layout.addWidget(self.training_progress)

        self.training_status_label = QLabel('Bereit')
        progress_layout.addWidget(self.training_status_label)

        layout.addWidget(progress_group)

        # Buttons
        btn_layout = QHBoxLayout()

        self.start_training_btn = QPushButton('Training starten')
        self.start_training_btn.setProperty('class', 'success')
        self.start_training_btn.clicked.connect(self._start_training)
        btn_layout.addWidget(self.start_training_btn)

        self.stop_training_btn = QPushButton('Stoppen')
        self.stop_training_btn.setProperty('class', 'danger')
        self.stop_training_btn.setEnabled(False)
        self.stop_training_btn.clicked.connect(self._stop_training)
        btn_layout.addWidget(self.stop_training_btn)

        layout.addLayout(btn_layout)

        # Platzhalter fuer Plot
        plot_group = QGroupBox('Training Plot')
        plot_layout = QVBoxLayout(plot_group)
        self.plot_placeholder = QLabel('Plot wird hier angezeigt...')
        self.plot_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.plot_placeholder.setMinimumHeight(300)
        self.plot_placeholder.setStyleSheet(f'background-color: {COLORS["bg_secondary"]}; border-radius: 4px;')
        plot_layout.addWidget(self.plot_placeholder)

        layout.addWidget(plot_group)
        layout.addStretch()

        return tab

    def _create_backtest_tab(self) -> QWidget:
        """Erstellt den Backtest-Tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Backtest Parameter
        param_group = QGroupBox('Backtest Parameter')
        param_layout = QGridLayout(param_group)

        param_layout.addWidget(QLabel('Startkapital:'), 0, 0)
        self.capital_spin = QDoubleSpinBox()
        self.capital_spin.setRange(100, 1000000)
        self.capital_spin.setValue(self.config.backtest.initial_capital)
        self.capital_spin.setPrefix('$')
        param_layout.addWidget(self.capital_spin, 0, 1)

        param_layout.addWidget(QLabel('Kommission:'), 0, 2)
        self.commission_spin = QDoubleSpinBox()
        self.commission_spin.setRange(0, 0.01)
        self.commission_spin.setSingleStep(0.0001)
        self.commission_spin.setDecimals(4)
        self.commission_spin.setValue(self.config.backtest.commission)
        self.commission_spin.setSuffix('%')
        param_layout.addWidget(self.commission_spin, 0, 3)

        layout.addWidget(param_group)

        # Progress
        self.backtest_progress = QProgressBar()
        layout.addWidget(self.backtest_progress)

        # Ergebnisse
        results_group = QGroupBox('Ergebnisse')
        results_layout = QGridLayout(results_group)

        self.result_labels = {}
        metrics = ['Total P/L', 'Return %', 'Win Rate', 'Profit Factor', 'Max Drawdown', 'Sharpe']

        for i, metric in enumerate(metrics):
            row, col = i // 2, (i % 2) * 2
            results_layout.addWidget(QLabel(f'{metric}:'), row, col)
            self.result_labels[metric] = QLabel('-')
            self.result_labels[metric].setStyleSheet(f'font-weight: bold; color: {COLORS["accent"]}')
            results_layout.addWidget(self.result_labels[metric], row, col + 1)

        layout.addWidget(results_group)

        # Buttons
        btn_layout = QHBoxLayout()

        self.start_backtest_btn = QPushButton('Backtest starten')
        self.start_backtest_btn.setProperty('class', 'success')
        self.start_backtest_btn.clicked.connect(self._start_backtest)
        btn_layout.addWidget(self.start_backtest_btn)

        self.stop_backtest_btn = QPushButton('Stoppen')
        self.stop_backtest_btn.setProperty('class', 'danger')
        self.stop_backtest_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_backtest_btn)

        layout.addLayout(btn_layout)
        layout.addStretch()

        return tab

    def _create_log_tab(self) -> QWidget:
        """Erstellt den Log-Tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        from PyQt6.QtWidgets import QTextEdit

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont('Consolas', 10))
        layout.addWidget(self.log_text)

        # Log-Datei lesen Button
        refresh_btn = QPushButton('Log aktualisieren')
        refresh_btn.clicked.connect(self._refresh_log)
        layout.addWidget(refresh_btn)

        return tab

    def _create_status_bar(self):
        """Erstellt die Status-Leiste."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.status_label = QLabel('Bereit')
        self.status_bar.addWidget(self.status_label, 1)

        self.gpu_indicator = QLabel()
        self.status_bar.addPermanentWidget(self.gpu_indicator)

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

        # Hilfe Menu
        help_menu = menu_bar.addMenu('Hilfe')

        about_action = QAction('Ueber...', self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _connect_signals(self):
        """Verbindet Signals mit Slots."""
        self.data_loaded.connect(self._on_data_loaded)
        self.model_loaded.connect(self._on_model_loaded)

    # === Event Handler ===

    def _auto_load_last_data(self):
        """Laedt automatisch die letzte CSV-Datei."""
        reader = CSVReader(self.config.paths.data_dir)
        df, filepath = reader.load_last_csv()

        if df is not None:
            self.data = df
            self.data_path = filepath
            self.data_loaded.emit(df)

    def _load_csv(self):
        """Oeffnet Dialog zum Laden einer CSV-Datei."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, 'CSV-Datei laden',
            str(self.config.paths.data_dir),
            'CSV Dateien (*.csv)'
        )

        if filepath:
            reader = CSVReader()
            df = reader.read(Path(filepath))

            if df is not None:
                self.data = df
                self.data_path = Path(filepath)
                reader.log_data_info(df, self.data_path)
                self.data_loaded.emit(df)
            else:
                QMessageBox.warning(self, 'Fehler', 'Konnte CSV nicht laden')

    def _download_data(self):
        """Oeffnet Download-Dialog."""
        QMessageBox.information(self, 'Info', 'Download-Dialog noch nicht implementiert')

    def _load_model(self):
        """Laedt ein gespeichertes Modell."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, 'Modell laden',
            str(self.config.paths.results_dir),
            'PyTorch Modelle (*.pt *.pth)'
        )

        if filepath:
            try:
                model_name = self.model_combo.currentText().lower()
                model_class = ModelFactory._registry.get(model_name)

                if model_class:
                    self.model, checkpoint = model_class.load(Path(filepath))
                    self.model_path = Path(filepath)
                    self.model_loaded.emit(self.model)
                    self.logger.success(f'Modell geladen: {filepath}')
            except Exception as e:
                QMessageBox.warning(self, 'Fehler', f'Konnte Modell nicht laden: {e}')
                self.logger.error(f'Modell-Ladefehler: {e}')

    def _create_model(self):
        """Erstellt ein neues Modell."""
        model_name = self.model_combo.currentText().lower()

        try:
            self.model = ModelFactory.create(
                model_name,
                input_size=len(self.config.training.features),
                hidden_size=self.hidden_size_spin.value(),
                num_layers=self.num_layers_spin.value(),
                num_classes=self.config.training.num_classes,
                dropout=self.dropout_spin.value()
            )

            self.model_loaded.emit(self.model)
            self.logger.success(f'{model_name.upper()} Modell erstellt')

        except Exception as e:
            QMessageBox.warning(self, 'Fehler', f'Konnte Modell nicht erstellen: {e}')
            self.logger.error(f'Modell-Erstellungsfehler: {e}')

    def _start_training(self):
        """Startet das Training."""
        if self.data is None:
            QMessageBox.warning(self, 'Fehler', 'Bitte zuerst Daten laden')
            return

        if self.model is None:
            QMessageBox.warning(self, 'Fehler', 'Bitte zuerst Modell erstellen oder laden')
            return

        self.logger.info('Training gestartet...')
        self.start_training_btn.setEnabled(False)
        self.stop_training_btn.setEnabled(True)
        self.training_status_label.setText('Training laeuft...')

        # TODO: Training in separatem Thread starten

    def _stop_training(self):
        """Stoppt das Training."""
        self.logger.warning('Training gestoppt')
        self.start_training_btn.setEnabled(True)
        self.stop_training_btn.setEnabled(False)
        self.training_status_label.setText('Training gestoppt')

    def _start_backtest(self):
        """Startet den Backtest."""
        if self.data is None:
            QMessageBox.warning(self, 'Fehler', 'Bitte zuerst Daten laden')
            return

        if self.model is None:
            QMessageBox.warning(self, 'Fehler', 'Bitte zuerst Modell laden')
            return

        self.logger.info('Backtest gestartet...')
        # TODO: Backtest implementieren

    def _refresh_log(self):
        """Aktualisiert den Log-Text."""
        log_path = self.logger.get_log_file_path()
        if log_path and Path(log_path).exists():
            with open(log_path, 'r', encoding='utf-8') as f:
                self.log_text.setPlainText(f.read())
            # Scroll nach unten
            self.log_text.verticalScrollBar().setValue(
                self.log_text.verticalScrollBar().maximum()
            )

    def _update_gpu_status(self):
        """Aktualisiert den GPU-Status."""
        from ..utils.helpers import get_gpu_info

        info = get_gpu_info()

        if info['cuda_available']:
            device = info['devices'][0]
            self.gpu_status_label.setText(f"{device['name']} ({device['total_memory_gb']:.1f} GB)")
            self.gpu_status_label.setStyleSheet(f'color: {COLORS["success"]}')
            self.gpu_indicator.setText('GPU')
            self.gpu_indicator.setStyleSheet(f'color: {COLORS["success"]}; font-weight: bold;')
        else:
            self.gpu_status_label.setText('Nicht verfuegbar (CPU)')
            self.gpu_status_label.setStyleSheet(f'color: {COLORS["warning"]}')
            self.gpu_indicator.setText('CPU')
            self.gpu_indicator.setStyleSheet(f'color: {COLORS["warning"]}; font-weight: bold;')

    def _show_about(self):
        """Zeigt About-Dialog."""
        QMessageBox.about(
            self,
            'Ueber BTCUSD Analyzer',
            'BTCUSD Analyzer v0.1.0\n\n'
            'Neural Network basierte BTC Trendwechsel-Erkennung\n\n'
            'PyTorch + PyQt6'
        )

    # === Signal Handler ===

    def _on_data_loaded(self, df: pd.DataFrame):
        """Wird aufgerufen wenn Daten geladen wurden."""
        reader = CSVReader()
        info = reader.get_data_info(df)

        self.data_status_label.setText(f'Geladen: {self.data_path.name if self.data_path else "OK"}')
        self.data_status_label.setStyleSheet(f'color: {COLORS["success"]}')

        self.data_count_label.setText(f'{info["records"]:,}')
        self.data_range_label.setText(f'{info["start_date"]} - {info["end_date"]}')
        self.data_interval_label.setText(info['interval'])

        self.status_label.setText(f'Daten geladen: {info["records"]:,} Datensaetze')

    def _on_model_loaded(self, model):
        """Wird aufgerufen wenn ein Modell geladen wurde."""
        self.model_status_label.setText(f'{model.name} geladen')
        self.model_status_label.setStyleSheet(f'color: {COLORS["success"]}')

        self.model_info_label.setText(
            f'Parameter: {model.count_parameters():,}\n'
            f'Hidden: {model.hidden_size}, Layers: {model.num_layers}'
        )

        self.status_label.setText(f'{model.name} Modell bereit')

    def closeEvent(self, event):
        """Wird beim Schliessen des Fensters aufgerufen."""
        self.logger.info('BTCUSD Analyzer beendet')
        event.accept()
