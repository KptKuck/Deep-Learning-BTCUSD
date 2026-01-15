"""
Walk-Forward Window - GUI fuer Walk-Forward Analyse

Bietet Konfiguration, Ausfuehrung und Visualisierung von Walk-Forward Backtests
mit vier Modi: Inference Only, Inference+Live, Retrain per Split, Live Simulation.
"""

from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox, QDoubleSpinBox, QSpinBox,
    QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QTextEdit, QProgressBar, QFileDialog, QMessageBox,
    QTabWidget, QComboBox, QRadioButton, QButtonGroup, QScrollArea,
    QFrame, QDateEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QDate
from PyQt6.QtGui import QFont

from .styles import get_stylesheet, COLORS, StyleFactory
from .chart_widget import ChartWidget
from ..backtester import (
    BacktestMode, WalkForwardType, BacktestEngine,
    WalkForwardConfig, WalkForwardEngine, WalkForwardResult,
    ExportConfig, ResultManager
)


# =============================================================================
# Worker Thread fuer Walk-Forward Analyse
# =============================================================================

class WalkForwardWorker(QThread):
    """Worker-Thread fuer asynchrone Walk-Forward Analyse."""
    finished = pyqtSignal(object)       # WalkForwardResult
    progress = pyqtSignal(int, str)     # Progress %, Status
    split_completed = pyqtSignal(int, dict)  # Split-ID, Metriken
    error = pyqtSignal(str)             # Fehlermeldung

    def __init__(self, engine: WalkForwardEngine):
        super().__init__()
        self.engine = engine
        self._cancelled = False

    def run(self):
        """Fuehrt Walk-Forward Analyse aus."""
        try:
            # Progress-Callback registrieren
            self.engine.progress_callback = self._on_progress

            self.progress.emit(0, "Starte Walk-Forward Analyse...")
            result = self.engine.run()

            if not self._cancelled:
                self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))

    def _on_progress(self, progress: int, message: str):
        """Progress-Callback vom Engine."""
        if not self._cancelled:
            self.progress.emit(progress, message)

    def cancel(self):
        """Bricht die Analyse ab."""
        self._cancelled = True
        if hasattr(self.engine, 'cancel'):
            self.engine.cancel()


# =============================================================================
# Walk-Forward Window
# =============================================================================

class WalkForwardWindow(QMainWindow):
    """
    Hauptfenster fuer Walk-Forward Analyse.

    Features:
    - Vier Backtest-Modi mit unterschiedlichen Trade-Offs
    - Rolling und Anchored Walk-Forward
    - Purged/Embargo Gap Konfiguration
    - Backtrader oder Simple Engine
    - Excel-Export der Ergebnisse
    - Visualisierung der Equity-Kurven
    """

    def __init__(self, data: pd.DataFrame = None, model=None,
                 model_info: Dict = None, training_config: Dict = None,
                 parent=None):
        super().__init__(parent)

        self.data = data
        self.model = model
        self.model_info = model_info or {}
        self.training_config = training_config or {}

        self.engine: Optional[WalkForwardEngine] = None
        self.worker: Optional[WalkForwardWorker] = None
        self.result: Optional[WalkForwardResult] = None

        self._init_ui()
        self.setStyleSheet(get_stylesheet())

        if data is not None:
            self._update_data_info()

    def _init_ui(self):
        """Initialisiert die UI-Komponenten."""
        self.setWindowTitle('Walk-Forward Analyse')
        self.setMinimumSize(1400, 1260)  # 40% mehr Hoehe (900 * 1.4)

        # Central Widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Splitter fuer flexible Aufteilung
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # === Linke Seite: Konfiguration (scrollbar) ===
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_panel = self._create_config_panel()
        left_scroll.setWidget(left_panel)
        splitter.addWidget(left_scroll)

        # === Rechte Seite: Ergebnisse ===
        right_panel = self._create_results_panel()
        splitter.addWidget(right_panel)

        # Splitter-Verhaeltnis
        splitter.setSizes([400, 1000])

    def _create_config_panel(self) -> QWidget:
        """Erstellt das Konfigurations-Panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 10, 0)

        # === Daten-Info ===
        data_group = QGroupBox("Daten & Modell")
        data_group.setStyleSheet(StyleFactory.group_style(hex_color=COLORS['accent']))
        data_layout = QGridLayout(data_group)

        self.data_rows_label = QLabel("Zeilen: -")
        self.data_range_label = QLabel("Zeitraum: -")
        self.data_model_label = QLabel("Modell: -")
        self.data_features_label = QLabel("Features: -")

        data_layout.addWidget(self.data_rows_label, 0, 0)
        data_layout.addWidget(self.data_range_label, 1, 0)
        data_layout.addWidget(self.data_model_label, 2, 0)
        data_layout.addWidget(self.data_features_label, 3, 0)

        layout.addWidget(data_group)

        # === Zeitraum-Auswahl ===
        time_group = QGroupBox("Zeitraum")
        time_group.setStyleSheet(StyleFactory.group_style(hex_color='#27ae60'))
        time_layout = QGridLayout(time_group)

        # Checkbox fuer benutzerdefinierten Zeitraum (standardmaessig aktiviert)
        self.custom_range_check = QCheckBox("Benutzerdefinierten Zeitraum verwenden")
        self.custom_range_check.setChecked(True)  # Standardmaessig aktiviert
        self.custom_range_check.stateChanged.connect(self._toggle_date_range)
        time_layout.addWidget(self.custom_range_check, 0, 0, 1, 2)

        # Von-Datum (standardmaessig aktiviert)
        time_layout.addWidget(QLabel("Von:"), 1, 0)
        self.from_date = QDateEdit()
        self.from_date.setCalendarPopup(True)
        self.from_date.setDisplayFormat("dd.MM.yyyy")
        self.from_date.setEnabled(True)  # Standardmaessig aktiviert
        time_layout.addWidget(self.from_date, 1, 1)

        # Bis-Datum (standardmaessig aktiviert)
        time_layout.addWidget(QLabel("Bis:"), 2, 0)
        self.to_date = QDateEdit()
        self.to_date.setCalendarPopup(True)
        self.to_date.setDisplayFormat("dd.MM.yyyy")
        self.to_date.setEnabled(True)  # Standardmaessig aktiviert
        time_layout.addWidget(self.to_date, 2, 1)

        # Schnellauswahl-Buttons (standardmaessig aktiviert)
        quick_layout = QHBoxLayout()
        self._quick_btns = []  # Initialisiere Liste vor der Schleife
        quick_periods = [('1M', 30), ('3M', 90), ('6M', 180), ('1J', 365)]
        for text, days in quick_periods:
            btn = QPushButton(text)
            btn.setFixedWidth(40)
            btn.setEnabled(True)  # Standardmaessig aktiviert
            btn.clicked.connect(lambda checked, d=days: self._set_quick_period(d))
            quick_layout.addWidget(btn)
            self._quick_btns.append(btn)
        quick_layout.addStretch()
        time_layout.addLayout(quick_layout, 3, 0, 1, 2)

        layout.addWidget(time_group)

        # === Backtest-Modus ===
        mode_group = QGroupBox("Backtest-Modus")
        mode_group.setStyleSheet(StyleFactory.group_style(hex_color='#9b59b6'))
        mode_layout = QVBoxLayout(mode_group)

        self.mode_buttons = QButtonGroup()

        mode_info = [
            ("inference_only", "Inference Only", "Session-Modell, Batch-Verarbeitung (schnellster)"),
            ("inference_live", "Inference + Live", "Session-Modell, Bar-by-Bar (kein Look-Ahead)"),
            ("retrain", "Retrain per Split", "Neues Training pro Split, Batch-Inferenz"),
            ("live_sim", "Live Simulation", "Neues Training + Bar-by-Bar (realistischster)"),
        ]

        for i, (mode_id, title, desc) in enumerate(mode_info):
            radio = QRadioButton(title)
            radio.setToolTip(desc)
            self.mode_buttons.addButton(radio, i)
            mode_layout.addWidget(radio)

            desc_label = QLabel(f"  {desc}")
            desc_label.setStyleSheet("color: #888; font-size: 11px;")
            mode_layout.addWidget(desc_label)

        self.mode_buttons.button(0).setChecked(True)  # Default: Inference Only

        layout.addWidget(mode_group)

        # === Walk-Forward Typ ===
        wf_group = QGroupBox("Walk-Forward Typ")
        wf_group.setStyleSheet(StyleFactory.group_style(hex_color='#e67e22'))
        wf_layout = QVBoxLayout(wf_group)

        self.wf_type_buttons = QButtonGroup()

        self.rolling_radio = QRadioButton("Rolling Window")
        self.rolling_radio.setToolTip("Fixe Trainings-Fenstergroesse, wandert durch Zeit")
        self.wf_type_buttons.addButton(self.rolling_radio, 0)
        wf_layout.addWidget(self.rolling_radio)

        self.anchored_radio = QRadioButton("Anchored (Expanding)")
        self.anchored_radio.setToolTip("Training immer vom Anfang, wachsender Datensatz")
        self.wf_type_buttons.addButton(self.anchored_radio, 1)
        wf_layout.addWidget(self.anchored_radio)

        self.rolling_radio.setChecked(True)

        layout.addWidget(wf_group)

        # === Split-Konfiguration ===
        split_group = QGroupBox("Split-Konfiguration")
        split_group.setStyleSheet(StyleFactory.group_style(hex_color='#3498db'))
        split_layout = QGridLayout(split_group)

        # Anzahl Splits
        split_layout.addWidget(QLabel("Anzahl Splits:"), 0, 0)
        self.n_splits_spin = QSpinBox()
        self.n_splits_spin.setRange(2, 50)
        self.n_splits_spin.setValue(10)
        split_layout.addWidget(self.n_splits_spin, 0, 1)

        # Train-Anteil
        split_layout.addWidget(QLabel("Train-Anteil:"), 1, 0)
        self.train_ratio_spin = QDoubleSpinBox()
        self.train_ratio_spin.setRange(0.5, 0.95)
        self.train_ratio_spin.setValue(0.8)
        self.train_ratio_spin.setSingleStep(0.05)
        split_layout.addWidget(self.train_ratio_spin, 1, 1)

        # Min Train-Samples
        split_layout.addWidget(QLabel("Min Train-Samples:"), 2, 0)
        self.min_train_spin = QSpinBox()
        self.min_train_spin.setRange(100, 50000)
        self.min_train_spin.setValue(5000)
        self.min_train_spin.setSingleStep(500)
        split_layout.addWidget(self.min_train_spin, 2, 1)

        # Min Test-Samples
        split_layout.addWidget(QLabel("Min Test-Samples:"), 3, 0)
        self.min_test_spin = QSpinBox()
        self.min_test_spin.setRange(50, 10000)
        self.min_test_spin.setValue(500)
        self.min_test_spin.setSingleStep(100)
        split_layout.addWidget(self.min_test_spin, 3, 1)

        # Purged Gap
        split_layout.addWidget(QLabel("Purged Gap (Bars):"), 4, 0)
        self.purged_gap_spin = QSpinBox()
        self.purged_gap_spin.setRange(0, 500)
        self.purged_gap_spin.setValue(50)
        self.purged_gap_spin.setToolTip("Gap zwischen Train und Test zur Vermeidung von Leakage")
        split_layout.addWidget(self.purged_gap_spin, 4, 1)

        # Embargo Gap
        split_layout.addWidget(QLabel("Embargo Gap (Bars):"), 5, 0)
        self.embargo_gap_spin = QSpinBox()
        self.embargo_gap_spin.setRange(0, 200)
        self.embargo_gap_spin.setValue(0)
        self.embargo_gap_spin.setToolTip("Zusaetzlicher Gap nach Test-Ende")
        split_layout.addWidget(self.embargo_gap_spin, 5, 1)

        layout.addWidget(split_group)

        # === Trading-Konfiguration ===
        trading_group = QGroupBox("Trading-Konfiguration")
        trading_group.setStyleSheet(StyleFactory.group_style(hex_color=COLORS['success']))
        trading_layout = QGridLayout(trading_group)

        # Engine
        trading_layout.addWidget(QLabel("Engine:"), 0, 0)
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["Simple", "Backtrader"])
        trading_layout.addWidget(self.engine_combo, 0, 1)

        # Initial Capital
        trading_layout.addWidget(QLabel("Startkapital:"), 1, 0)
        self.capital_spin = QDoubleSpinBox()
        self.capital_spin.setRange(100, 10000000)
        self.capital_spin.setValue(10000)
        self.capital_spin.setPrefix("$ ")
        self.capital_spin.setDecimals(0)
        trading_layout.addWidget(self.capital_spin, 1, 1)

        # Commission
        trading_layout.addWidget(QLabel("Kommission:"), 2, 0)
        self.commission_spin = QDoubleSpinBox()
        self.commission_spin.setRange(0, 1)
        self.commission_spin.setValue(0.001)
        self.commission_spin.setSuffix(" %")
        self.commission_spin.setDecimals(4)
        trading_layout.addWidget(self.commission_spin, 2, 1)

        # Slippage
        trading_layout.addWidget(QLabel("Slippage:"), 3, 0)
        self.slippage_spin = QDoubleSpinBox()
        self.slippage_spin.setRange(0, 1)
        self.slippage_spin.setValue(0.0)
        self.slippage_spin.setSuffix(" %")
        self.slippage_spin.setDecimals(4)
        trading_layout.addWidget(self.slippage_spin, 3, 1)

        # Position Size
        trading_layout.addWidget(QLabel("Position Size:"), 4, 0)
        self.stake_spin = QDoubleSpinBox()
        self.stake_spin.setRange(0.001, 1000)
        self.stake_spin.setValue(1.0)
        self.stake_spin.setDecimals(3)
        trading_layout.addWidget(self.stake_spin, 4, 1)

        # Short erlauben
        self.allow_short_check = QCheckBox("Short-Positionen erlauben")
        self.allow_short_check.setChecked(True)
        trading_layout.addWidget(self.allow_short_check, 5, 0, 1, 2)

        layout.addWidget(trading_group)

        # === Optionen ===
        options_group = QGroupBox("Optionen")
        options_group.setStyleSheet(StyleFactory.group_style(hex_color='#808080'))
        options_layout = QVBoxLayout(options_group)

        self.gpu_parallel_check = QCheckBox("GPU-Parallelisierung nutzen")
        self.gpu_parallel_check.setChecked(True)
        options_layout.addWidget(self.gpu_parallel_check)

        self.save_models_check = QCheckBox("Modelle speichern")
        self.save_models_check.setChecked(False)
        self.save_models_check.setToolTip("Speichert trainierte Modelle pro Split")
        options_layout.addWidget(self.save_models_check)

        self.verbose_check = QCheckBox("Detailliertes Logging")
        self.verbose_check.setChecked(True)
        options_layout.addWidget(self.verbose_check)

        layout.addWidget(options_group)

        # === Buttons ===
        button_layout = QVBoxLayout()

        self.run_btn = QPushButton("Analyse starten")
        self.run_btn.setStyleSheet(StyleFactory.button_style((0.2, 0.7, 0.3)))
        self.run_btn.setMinimumHeight(45)
        self.run_btn.clicked.connect(self._run_analysis)
        button_layout.addWidget(self.run_btn)

        self.cancel_btn = QPushButton("Abbrechen")
        self.cancel_btn.setStyleSheet(StyleFactory.button_style((0.8, 0.3, 0.2)))
        self.cancel_btn.clicked.connect(self._cancel_analysis)
        self.cancel_btn.setEnabled(False)
        button_layout.addWidget(self.cancel_btn)

        self.export_btn = QPushButton("Excel Export")
        self.export_btn.setStyleSheet(StyleFactory.button_style((0.3, 0.5, 0.7)))
        self.export_btn.clicked.connect(self._export_results)
        self.export_btn.setEnabled(False)
        button_layout.addWidget(self.export_btn)

        layout.addLayout(button_layout)

        # === Progress ===
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #aaa;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch()

        return panel

    def _create_results_panel(self) -> QWidget:
        """Erstellt das Ergebnis-Panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        # Tab Widget
        self.results_tabs = QTabWidget()
        self.results_tabs.setStyleSheet(StyleFactory.tab_style())

        # === Tab 1: Uebersicht ===
        overview_tab = self._create_overview_tab()
        self.results_tabs.addTab(overview_tab, "Uebersicht")

        # === Tab 2: Splits ===
        splits_tab = self._create_splits_tab()
        self.results_tabs.addTab(splits_tab, "Splits")

        # === Tab 3: Trades ===
        trades_tab = self._create_trades_tab()
        self.results_tabs.addTab(trades_tab, "Trades")

        # === Tab 4: Equity ===
        equity_tab = self._create_equity_tab()
        self.results_tabs.addTab(equity_tab, "Equity")

        # === Tab 5: Log ===
        log_tab = self._create_log_tab()
        self.results_tabs.addTab(log_tab, "Log")

        layout.addWidget(self.results_tabs)

        return panel

    def _create_overview_tab(self) -> QWidget:
        """Erstellt den Uebersicht-Tab."""
        tab = QWidget()
        layout = QHBoxLayout(tab)

        # === Gesamt-Metriken ===
        total_group = QGroupBox("Gesamt-Performance")
        total_group.setStyleSheet(StyleFactory.group_style(hex_color=COLORS['success']))
        total_layout = QGridLayout(total_group)

        self.metric_labels = {}
        metrics = [
            ('total_return', 'Total Return:', '0 %'),
            ('sharpe_ratio', 'Sharpe Ratio:', '0.00'),
            ('max_drawdown', 'Max Drawdown:', '0 %'),
            ('profit_factor', 'Profit Factor:', '0.00'),
            ('win_rate', 'Win Rate:', '0 %'),
            ('total_trades', 'Anzahl Trades:', '0'),
            ('winning_trades', 'Gewinner:', '0'),
            ('losing_trades', 'Verlierer:', '0'),
            ('avg_trade', 'Durchschn. Trade:', '0'),
        ]

        for i, (key, label, default) in enumerate(metrics):
            lbl = QLabel(label)
            val = QLabel(default)
            val.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.metric_labels[key] = val
            total_layout.addWidget(lbl, i, 0)
            total_layout.addWidget(val, i, 1)

        layout.addWidget(total_group)

        # === Split-Statistiken ===
        split_stats_group = QGroupBox("Split-Statistiken")
        split_stats_group.setStyleSheet(StyleFactory.group_style(hex_color='#3498db'))
        split_stats_layout = QGridLayout(split_stats_group)

        split_metrics = [
            ('total_splits', 'Anzahl Splits:', '0'),
            ('profitable_splits', 'Profitable Splits:', '0'),
            ('best_split_return', 'Bester Split:', '0 %'),
            ('worst_split_return', 'Schlechtester Split:', '0 %'),
            ('avg_split_return', 'Durchschn. Return:', '0 %'),
            ('split_return_std', 'Return Std:', '0 %'),
        ]

        for i, (key, label, default) in enumerate(split_metrics):
            lbl = QLabel(label)
            val = QLabel(default)
            val.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.metric_labels[key] = val
            split_stats_layout.addWidget(lbl, i, 0)
            split_stats_layout.addWidget(val, i, 1)

        layout.addWidget(split_stats_group)

        # === Ausfuehrung ===
        exec_group = QGroupBox("Ausfuehrung")
        exec_group.setStyleSheet(StyleFactory.group_style(hex_color='#808080'))
        exec_layout = QGridLayout(exec_group)

        exec_metrics = [
            ('mode', 'Modus:', '-'),
            ('wf_type', 'Walk-Forward Typ:', '-'),
            ('engine_type', 'Engine:', '-'),
            ('execution_time', 'Laufzeit:', '-'),
            ('start_date', 'Start:', '-'),
            ('end_date', 'Ende:', '-'),
        ]

        for i, (key, label, default) in enumerate(exec_metrics):
            lbl = QLabel(label)
            val = QLabel(default)
            val.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.metric_labels[key] = val
            exec_layout.addWidget(lbl, i, 0)
            exec_layout.addWidget(val, i, 1)

        layout.addWidget(exec_group)

        return tab

    def _create_splits_tab(self) -> QWidget:
        """Erstellt den Splits-Tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.splits_table = QTableWidget()
        self.splits_table.setColumnCount(9)
        self.splits_table.setHorizontalHeaderLabels([
            'Split', 'Train Start', 'Train Ende', 'Test Start', 'Test Ende',
            'Return', 'Sharpe', 'Max DD', 'Trades'
        ])
        self.splits_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.splits_table.setAlternatingRowColors(True)

        layout.addWidget(self.splits_table)

        return tab

    def _create_trades_tab(self) -> QWidget:
        """Erstellt den Trades-Tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Filter
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Split:"))
        self.trade_split_filter = QComboBox()
        self.trade_split_filter.addItem("Alle")
        self.trade_split_filter.currentTextChanged.connect(self._filter_trades)
        filter_layout.addWidget(self.trade_split_filter)
        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(10)
        self.trades_table.setHorizontalHeaderLabels([
            'Split', 'Trade #', 'Einstieg', 'Ausstieg', 'Richtung',
            'Entry', 'Exit', 'PnL', 'PnL %', 'Bars'
        ])
        self.trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.trades_table.setAlternatingRowColors(True)

        layout.addWidget(self.trades_table)

        return tab

    def _create_equity_tab(self) -> QWidget:
        """Erstellt den Equity-Tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.equity_chart = ChartWidget("Equity Curve")
        layout.addWidget(self.equity_chart)

        return tab

    def _create_log_tab(self) -> QWidget:
        """Erstellt den Log-Tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #1a1a1a; font-family: Consolas, monospace;")

        layout.addWidget(self.log_text)

        # Clear-Button
        clear_btn = QPushButton("Log loeschen")
        clear_btn.setStyleSheet(StyleFactory.button_style((0.4, 0.4, 0.4)))
        clear_btn.clicked.connect(self.log_text.clear)
        layout.addWidget(clear_btn)

        return tab

    def _update_data_info(self):
        """Aktualisiert die Daten-Info Anzeige."""
        if self.data is None:
            return

        self.data_rows_label.setText(f"Zeilen: {len(self.data):,}")

        if hasattr(self.data.index, 'min'):
            try:
                start = pd.to_datetime(self.data.index.min())
                end = pd.to_datetime(self.data.index.max())
                self.data_range_label.setText(
                    f"Zeitraum: {start.strftime('%Y-%m-%d')} - {end.strftime('%Y-%m-%d')}"
                )
                # Datum-Felder mit Daten-Zeitraum initialisieren
                self.from_date.setDate(QDate(start.year, start.month, start.day))
                self.to_date.setDate(QDate(end.year, end.month, end.day))
                # Grenzen setzen
                self.from_date.setMinimumDate(QDate(start.year, start.month, start.day))
                self.from_date.setMaximumDate(QDate(end.year, end.month, end.day))
                self.to_date.setMinimumDate(QDate(start.year, start.month, start.day))
                self.to_date.setMaximumDate(QDate(end.year, end.month, end.day))
            except:
                pass

        if self.model is not None:
            model_name = self.model_info.get('model_name', 'BILSTM')
            self.data_model_label.setText(f"Modell: {model_name}")
        else:
            self.data_model_label.setText("Modell: Nicht geladen")

        n_features = self.data.shape[1] if len(self.data.shape) > 1 else 1
        self.data_features_label.setText(f"Features: {n_features}")

    def _toggle_date_range(self, state):
        """Aktiviert/Deaktiviert die Datums-Auswahl."""
        enabled = state == Qt.CheckState.Checked.value
        self.from_date.setEnabled(enabled)
        self.to_date.setEnabled(enabled)
        for btn in getattr(self, '_quick_btns', []):
            btn.setEnabled(enabled)

    def _set_quick_period(self, days: int):
        """Setzt den Zeitraum auf die letzten X Tage der verfuegbaren Daten."""
        if self.data is None:
            return
        try:
            end = pd.to_datetime(self.data.index.max())
            start = end - pd.Timedelta(days=days)
            # Mindestens ab Daten-Anfang
            data_start = pd.to_datetime(self.data.index.min())
            if start < data_start:
                start = data_start
            self.from_date.setDate(QDate(start.year, start.month, start.day))
            self.to_date.setDate(QDate(end.year, end.month, end.day))
            self._log(f"Zeitraum gesetzt: Letzte {days} Tage")
        except Exception as e:
            self._log(f"Fehler beim Setzen des Zeitraums: {e}")

    def _get_filtered_data(self) -> pd.DataFrame:
        """Gibt die nach Zeitraum gefilterten Daten zurueck."""
        if self.data is None:
            return None

        if not self.custom_range_check.isChecked():
            return self.data

        # Zeitraum aus GUI
        from_date = self.from_date.date().toPyDate()
        to_date = self.to_date.date().toPyDate()

        # In Pandas Datetime konvertieren
        from_dt = pd.Timestamp(from_date)
        to_dt = pd.Timestamp(to_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        # Filtern
        mask = (self.data.index >= from_dt) & (self.data.index <= to_dt)
        filtered = self.data.loc[mask]

        self._log(f"Daten gefiltert: {len(filtered):,} von {len(self.data):,} Zeilen")
        self._log(f"Zeitraum: {from_date} bis {to_date}")

        return filtered

    def set_data(self, data: pd.DataFrame, model=None, model_info: Dict = None,
                 training_config: Dict = None):
        """Setzt Daten und Modell fuer Analyse."""
        self.data = data
        self.model = model
        self.model_info = model_info or {}
        self.training_config = training_config or {}
        self._update_data_info()

    def _get_config(self) -> WalkForwardConfig:
        """Erstellt WalkForwardConfig aus GUI-Einstellungen."""
        # Modus bestimmen
        mode_id = self.mode_buttons.checkedId()
        mode_map = {
            0: BacktestMode.INFERENCE_ONLY,
            1: BacktestMode.INFERENCE_LIVE,
            2: BacktestMode.RETRAIN_PER_SPLIT,
            3: BacktestMode.LIVE_SIMULATION,
        }
        mode = mode_map.get(mode_id, BacktestMode.INFERENCE_ONLY)

        # Walk-Forward Typ
        wf_type = WalkForwardType.ROLLING if self.rolling_radio.isChecked() else WalkForwardType.ANCHORED

        # Engine
        engine = BacktestEngine.BACKTRADER if self.engine_combo.currentIndex() == 1 else BacktestEngine.SIMPLE

        return WalkForwardConfig(
            mode=mode,
            walk_forward_type=wf_type,
            engine=engine,
            n_splits=self.n_splits_spin.value(),
            train_ratio=self.train_ratio_spin.value(),
            min_train_samples=self.min_train_spin.value(),
            min_test_samples=self.min_test_spin.value(),
            purged_gap=self.purged_gap_spin.value(),
            embargo_gap=self.embargo_gap_spin.value(),
            initial_capital=self.capital_spin.value(),
            commission=self.commission_spin.value() / 100,
            slippage=self.slippage_spin.value() / 100,
            stake=self.stake_spin.value(),
            allow_short=self.allow_short_check.isChecked(),
            use_gpu=self.gpu_parallel_check.isChecked(),
            save_models=self.save_models_check.isChecked(),
            verbose=self.verbose_check.isChecked(),
        )

    def _run_analysis(self):
        """Startet die Walk-Forward Analyse."""
        if self.data is None:
            QMessageBox.warning(self, "Fehler", "Keine Daten geladen!")
            return

        if self.model is None:
            mode_id = self.mode_buttons.checkedId()
            if mode_id in [0, 1]:  # Inference Modi brauchen Modell
                QMessageBox.warning(self, "Fehler",
                                  "Fuer Inference-Modi wird ein Modell benoetigt!")
                return

        try:
            config = self._get_config()
        except Exception as e:
            QMessageBox.critical(self, "Konfigurationsfehler", str(e))
            return

        # Daten filtern falls benutzerdefinierter Zeitraum
        analysis_data = self._get_filtered_data()
        if analysis_data is None or len(analysis_data) == 0:
            QMessageBox.warning(self, "Fehler", "Keine Daten im ausgewaehlten Zeitraum!")
            return

        # Log starten
        self._log(f"=== Walk-Forward Analyse gestartet ===")
        self._log(f"Modus: {config.mode.value}")
        self._log(f"Typ: {config.walk_forward_type.value}")
        self._log(f"Splits: {config.n_splits}")
        self._log(f"Datenpunkte: {len(analysis_data):,}")
        self._log("")

        # Engine erstellen mit gefilterten Daten
        self.engine = WalkForwardEngine(
            config=config,
            data=analysis_data,
            model=self.model,
            model_info=self.model_info,
            training_config=self.training_config
        )

        # UI aktualisieren
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.export_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Worker starten
        self.worker = WalkForwardWorker(self.engine)
        self.worker.finished.connect(self._on_analysis_finished)
        self.worker.progress.connect(self._on_progress)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _cancel_analysis(self):
        """Bricht die Analyse ab."""
        if self.worker:
            self._log("Analyse wird abgebrochen...")
            self.worker.cancel()
            self.worker.wait()
            self._log("Analyse abgebrochen.")

        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Abgebrochen")

    def _on_progress(self, progress: int, message: str):
        """Update Progress-Anzeige."""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)
        self._log(message)

    def _on_analysis_finished(self, result: WalkForwardResult):
        """Wird aufgerufen wenn Analyse fertig ist."""
        self.result = result
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.export_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Analyse abgeschlossen!")

        self._log("")
        self._log(f"=== Analyse abgeschlossen ===")
        self._log(f"Laufzeit: {result.execution_time:.1f} Sekunden")
        self._log(f"Total Return: {result.total_return:.2%}")
        self._log(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")

        self._update_results(result)

    def _on_error(self, error_msg: str):
        """Wird bei Fehlern aufgerufen."""
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Fehler: {error_msg}")
        self._log(f"FEHLER: {error_msg}")
        QMessageBox.critical(self, "Analyse Fehler", error_msg)

    def _update_results(self, result: WalkForwardResult):
        """Aktualisiert die Ergebnis-Anzeige."""
        # Gesamt-Metriken
        color = COLORS['success'] if result.total_return >= 0 else COLORS['error']

        self.metric_labels['total_return'].setText(f"{result.total_return:.2%}")
        self.metric_labels['total_return'].setStyleSheet(f"color: {color};")
        self.metric_labels['sharpe_ratio'].setText(f"{result.sharpe_ratio:.3f}")
        self.metric_labels['max_drawdown'].setText(f"{result.max_drawdown:.2%}")
        self.metric_labels['profit_factor'].setText(f"{result.profit_factor:.2f}")
        self.metric_labels['win_rate'].setText(f"{result.win_rate:.1%}")
        self.metric_labels['total_trades'].setText(str(result.total_trades))
        self.metric_labels['winning_trades'].setText(str(result.winning_trades))
        self.metric_labels['losing_trades'].setText(str(result.losing_trades))
        self.metric_labels['avg_trade'].setText(f"{result.avg_trade:.4f}")

        # Split-Statistiken
        self.metric_labels['total_splits'].setText(str(result.total_splits))

        if result.split_results:
            returns = [s.total_return for s in result.split_results]
            profitable = sum(1 for r in returns if r > 0)

            self.metric_labels['profitable_splits'].setText(f"{profitable}/{len(returns)}")
            self.metric_labels['best_split_return'].setText(f"{max(returns):.2%}")
            self.metric_labels['worst_split_return'].setText(f"{min(returns):.2%}")
            self.metric_labels['avg_split_return'].setText(f"{np.mean(returns):.2%}")
            self.metric_labels['split_return_std'].setText(f"{np.std(returns):.2%}")

        # Ausfuehrungs-Info
        self.metric_labels['execution_time'].setText(f"{result.execution_time:.1f} s")
        if result.start_date:
            self.metric_labels['start_date'].setText(result.start_date.strftime("%Y-%m-%d"))
        if result.end_date:
            self.metric_labels['end_date'].setText(result.end_date.strftime("%Y-%m-%d"))

        # Splits-Tabelle
        self._update_splits_table(result)

        # Trades-Tabelle
        self._update_trades_table(result)

    def _update_splits_table(self, result: WalkForwardResult):
        """Aktualisiert die Splits-Tabelle."""
        self.splits_table.setRowCount(len(result.split_results))

        for i, split in enumerate(result.split_results):
            self.splits_table.setItem(i, 0, QTableWidgetItem(str(split.split_id)))

            # Daten formatieren
            train_start = split.train_start.strftime("%Y-%m-%d") if split.train_start else "-"
            train_end = split.train_end.strftime("%Y-%m-%d") if split.train_end else "-"
            test_start = split.test_start.strftime("%Y-%m-%d") if split.test_start else "-"
            test_end = split.test_end.strftime("%Y-%m-%d") if split.test_end else "-"

            self.splits_table.setItem(i, 1, QTableWidgetItem(train_start))
            self.splits_table.setItem(i, 2, QTableWidgetItem(train_end))
            self.splits_table.setItem(i, 3, QTableWidgetItem(test_start))
            self.splits_table.setItem(i, 4, QTableWidgetItem(test_end))

            # Return mit Farbcodierung
            return_item = QTableWidgetItem(f"{split.total_return:.2%}")
            if split.total_return >= 0:
                return_item.setForeground(Qt.GlobalColor.green)
            else:
                return_item.setForeground(Qt.GlobalColor.red)
            self.splits_table.setItem(i, 5, return_item)

            self.splits_table.setItem(i, 6, QTableWidgetItem(f"{split.sharpe_ratio:.3f}"))
            self.splits_table.setItem(i, 7, QTableWidgetItem(f"{split.max_drawdown:.2%}"))
            self.splits_table.setItem(i, 8, QTableWidgetItem(str(split.n_trades)))

    def _update_trades_table(self, result: WalkForwardResult):
        """Aktualisiert die Trades-Tabelle."""
        # Filter aktualisieren
        self.trade_split_filter.clear()
        self.trade_split_filter.addItem("Alle")
        for split in result.split_results:
            self.trade_split_filter.addItem(f"Split {split.split_id}")

        # Alle Trades sammeln
        all_trades = []
        for split in result.split_results:
            for trade in split.trades:
                all_trades.append((split.split_id, trade))

        self.trades_table.setRowCount(len(all_trades))

        for i, (split_id, trade) in enumerate(all_trades):
            self.trades_table.setItem(i, 0, QTableWidgetItem(str(split_id)))
            self.trades_table.setItem(i, 1, QTableWidgetItem(str(trade.trade_id)))

            entry_time = trade.entry_time.strftime("%Y-%m-%d %H:%M") if trade.entry_time else "-"
            exit_time = trade.exit_time.strftime("%Y-%m-%d %H:%M") if trade.exit_time else "-"

            self.trades_table.setItem(i, 2, QTableWidgetItem(entry_time))
            self.trades_table.setItem(i, 3, QTableWidgetItem(exit_time))
            self.trades_table.setItem(i, 4, QTableWidgetItem(trade.direction))
            self.trades_table.setItem(i, 5, QTableWidgetItem(f"${trade.entry_price:.2f}"))
            self.trades_table.setItem(i, 6, QTableWidgetItem(f"${trade.exit_price:.2f}"))

            # PnL mit Farbe
            pnl_item = QTableWidgetItem(f"${trade.pnl:.2f}")
            pnl_pct_item = QTableWidgetItem(f"{trade.pnl_percent:.2%}")

            if trade.pnl >= 0:
                pnl_item.setForeground(Qt.GlobalColor.green)
                pnl_pct_item.setForeground(Qt.GlobalColor.green)
            else:
                pnl_item.setForeground(Qt.GlobalColor.red)
                pnl_pct_item.setForeground(Qt.GlobalColor.red)

            self.trades_table.setItem(i, 7, pnl_item)
            self.trades_table.setItem(i, 8, pnl_pct_item)
            self.trades_table.setItem(i, 9, QTableWidgetItem(str(trade.bars_held)))

    def _filter_trades(self, filter_text: str):
        """Filtert Trades nach Split."""
        if not self.result:
            return

        if filter_text == "Alle" or not filter_text:
            # Alle anzeigen
            for row in range(self.trades_table.rowCount()):
                self.trades_table.setRowHidden(row, False)
        else:
            # Nach Split filtern
            split_id = filter_text.replace("Split ", "")
            for row in range(self.trades_table.rowCount()):
                item = self.trades_table.item(row, 0)
                if item and item.text() == split_id:
                    self.trades_table.setRowHidden(row, False)
                else:
                    self.trades_table.setRowHidden(row, True)

    def _export_results(self):
        """Exportiert Ergebnisse nach Excel."""
        if self.result is None:
            return

        # Pfad waehlen
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Ergebnisse exportieren", "", "Excel (*.xlsx)"
        )

        if not filepath:
            return

        if not filepath.endswith('.xlsx'):
            filepath += '.xlsx'

        # Export durchfuehren
        try:
            config = ExportConfig(
                include_summary=True,
                include_splits=True,
                include_trades=True,
                include_equity=True,
                include_config=True,
                chart_equity=True
            )

            manager = ResultManager(config)
            result_path = manager.export_results(
                self.result,
                self._get_config(),
                Path(filepath)
            )

            if result_path:
                self._log(f"Export erfolgreich: {result_path}")
                QMessageBox.information(
                    self, "Export",
                    f"Ergebnisse exportiert nach:\n{result_path}"
                )
            else:
                QMessageBox.warning(self, "Export", "Export fehlgeschlagen!")

        except Exception as e:
            self._log(f"Export-Fehler: {e}")
            QMessageBox.critical(self, "Export Fehler", str(e))

    def _log(self, message: str):
        """Schreibt Nachricht ins Log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
