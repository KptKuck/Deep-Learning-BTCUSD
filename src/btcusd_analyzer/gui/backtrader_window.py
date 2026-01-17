"""
Backtrader Window - GUI fuer professionelles Backtesting mit Backtrader

Bietet Konfiguration, Ausfuehrung und Visualisierung von Backtests
basierend auf dem BILSTM-Modell.
"""

from typing import Optional, Dict
from pathlib import Path

import pandas as pd
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox, QDoubleSpinBox, QSpinBox,
    QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QTextEdit, QProgressBar, QFileDialog, QMessageBox,
    QTabWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

from .styles import get_stylesheet, COLORS, StyleFactory
from .chart_widget import ChartWidget
from ..backtester import BacktraderEngine, BacktestResult


# =============================================================================
# Worker Thread fuer Backtest
# =============================================================================

class BacktestWorker(QThread):
    """Worker-Thread fuer asynchrone Backtest-Ausfuehrung."""
    finished = pyqtSignal(object)  # BacktestResult
    progress = pyqtSignal(str)     # Status-Nachricht
    error = pyqtSignal(str)        # Fehlermeldung

    def __init__(self, engine: BacktraderEngine, config: Dict):
        super().__init__()
        self.engine = engine
        self.config = config

    def run(self):
        """Fuehrt Backtest aus."""
        try:
            self.progress.emit("Bereite Daten vor...")
            self.engine.prepare_data()

            self.progress.emit("Fuehre Backtest aus...")
            result = self.engine.run_backtest(**self.config)

            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# =============================================================================
# Backtrader Window
# =============================================================================

class BacktraderWindow(QMainWindow):
    """
    Hauptfenster fuer Backtrader-Integration.

    Features:
    - Konfiguration (Kapital, Kommission, Slippage, etc.)
    - Backtest-Ausfuehrung mit Fortschrittsanzeige
    - Ergebnis-Anzeige (Metriken, Trades, Charts)
    - Export-Funktionen
    """

    def __init__(self, data: pd.DataFrame = None, model=None,
                 model_info: Dict = None, signals: pd.Series = None,
                 parent=None):
        super().__init__(parent)

        self.data = data
        self.model = model
        self.model_info = model_info or {}
        self.signals = signals

        self.engine: Optional[BacktraderEngine] = None
        self.worker: Optional[BacktestWorker] = None
        self.result: Optional[BacktestResult] = None

        self._init_ui()
        self.setStyleSheet(get_stylesheet())

        if data is not None:
            self._update_data_info()

    def _init_ui(self):
        """Initialisiert die UI-Komponenten."""
        self.setWindowTitle('4.1 - Backtrader')
        self.setMinimumSize(1200, 800)

        # Central Widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Splitter fuer flexible Aufteilung
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # === Linke Seite: Konfiguration ===
        left_panel = self._create_config_panel()
        splitter.addWidget(left_panel)

        # === Rechte Seite: Ergebnisse ===
        right_panel = self._create_results_panel()
        splitter.addWidget(right_panel)

        # Splitter-Verhaeltnis
        splitter.setSizes([350, 850])

    def _create_config_panel(self) -> QWidget:
        """Erstellt das Konfigurations-Panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        # === Daten-Info ===
        data_group = QGroupBox("Daten")
        data_group.setStyleSheet(StyleFactory.group_style(hex_color=COLORS['accent']))
        data_layout = QGridLayout(data_group)

        self.data_rows_label = QLabel("Zeilen: -")
        self.data_range_label = QLabel("Zeitraum: -")
        self.data_model_label = QLabel("Modell: -")

        data_layout.addWidget(self.data_rows_label, 0, 0)
        data_layout.addWidget(self.data_range_label, 1, 0)
        data_layout.addWidget(self.data_model_label, 2, 0)

        layout.addWidget(data_group)

        # === Kapital & Kosten ===
        capital_group = QGroupBox("Kapital & Kosten")
        capital_group.setStyleSheet(StyleFactory.group_style(hex_color='#4da8da'))
        capital_layout = QGridLayout(capital_group)

        # Initial Capital
        capital_layout.addWidget(QLabel("Startkapital:"), 0, 0)
        self.capital_spin = QDoubleSpinBox()
        self.capital_spin.setRange(100, 10000000)
        self.capital_spin.setValue(10000)
        self.capital_spin.setPrefix("$ ")
        self.capital_spin.setDecimals(0)
        capital_layout.addWidget(self.capital_spin, 0, 1)

        # Commission
        capital_layout.addWidget(QLabel("Kommission:"), 1, 0)
        self.commission_spin = QDoubleSpinBox()
        self.commission_spin.setRange(0, 1)
        self.commission_spin.setValue(0.001)
        self.commission_spin.setSuffix(" %")
        self.commission_spin.setDecimals(4)
        self.commission_spin.setSingleStep(0.0001)
        capital_layout.addWidget(self.commission_spin, 1, 1)

        # Slippage
        capital_layout.addWidget(QLabel("Slippage:"), 2, 0)
        self.slippage_spin = QDoubleSpinBox()
        self.slippage_spin.setRange(0, 1)
        self.slippage_spin.setValue(0.0)
        self.slippage_spin.setSuffix(" %")
        self.slippage_spin.setDecimals(4)
        self.slippage_spin.setSingleStep(0.0001)
        capital_layout.addWidget(self.slippage_spin, 2, 1)

        layout.addWidget(capital_group)

        # === Position Sizing ===
        position_group = QGroupBox("Position Sizing")
        position_group.setStyleSheet(StyleFactory.group_style(hex_color='#9b59b6'))
        position_layout = QGridLayout(position_group)

        # Stake
        position_layout.addWidget(QLabel("Position Size:"), 0, 0)
        self.stake_spin = QDoubleSpinBox()
        self.stake_spin.setRange(0.001, 1000)
        self.stake_spin.setValue(1.0)
        self.stake_spin.setDecimals(3)
        position_layout.addWidget(self.stake_spin, 0, 1)

        # Percent of Capital Checkbox
        self.stake_pct_check = QCheckBox("% des Kapitals")
        self.stake_pct_check.setChecked(False)
        position_layout.addWidget(self.stake_pct_check, 1, 0, 1, 2)

        layout.addWidget(position_group)

        # === Optionen ===
        options_group = QGroupBox("Optionen")
        options_group.setStyleSheet(StyleFactory.group_style(hex_color='#e67e22'))
        options_layout = QVBoxLayout(options_group)

        self.allow_short_check = QCheckBox("Short-Positionen erlauben")
        self.allow_short_check.setChecked(True)
        options_layout.addWidget(self.allow_short_check)

        self.invert_signals_check = QCheckBox("Signale invertieren")
        self.invert_signals_check.setChecked(False)
        options_layout.addWidget(self.invert_signals_check)

        layout.addWidget(options_group)

        # === Buttons ===
        button_layout = QVBoxLayout()

        self.run_btn = QPushButton("Backtest starten")
        self.run_btn.setStyleSheet(StyleFactory.button_style((0.2, 0.7, 0.3)))
        self.run_btn.setMinimumHeight(40)
        self.run_btn.clicked.connect(self._run_backtest)
        button_layout.addWidget(self.run_btn)

        self.export_btn = QPushButton("Ergebnisse exportieren")
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
        layout.addWidget(self.status_label)

        layout.addStretch()

        return panel

    def _create_results_panel(self) -> QWidget:
        """Erstellt das Ergebnis-Panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        # Tab Widget fuer verschiedene Ansichten
        self.results_tabs = QTabWidget()
        self.results_tabs.setStyleSheet(StyleFactory.tab_style())

        # === Tab 1: Uebersicht ===
        overview_tab = self._create_overview_tab()
        self.results_tabs.addTab(overview_tab, "Uebersicht")

        # === Tab 2: Trades ===
        trades_tab = self._create_trades_tab()
        self.results_tabs.addTab(trades_tab, "Trades")

        # === Tab 3: Equity Chart ===
        chart_tab = self._create_chart_tab()
        self.results_tabs.addTab(chart_tab, "Chart")

        layout.addWidget(self.results_tabs)

        return panel

    def _create_overview_tab(self) -> QWidget:
        """Erstellt den Uebersicht-Tab."""
        tab = QWidget()
        layout = QHBoxLayout(tab)

        # === Konto-Metriken ===
        account_group = QGroupBox("Konto")
        account_group.setStyleSheet(StyleFactory.group_style(hex_color=COLORS['success']))
        account_layout = QGridLayout(account_group)

        self.metric_labels = {}
        account_metrics = [
            ('initial_capital', 'Startkapital:', '$ 0'),
            ('final_value', 'Endwert:', '$ 0'),
            ('total_return', 'Gewinn/Verlust:', '$ 0'),
            ('total_return_pct', 'Return:', '0 %'),
            ('max_drawdown', 'Max Drawdown:', '$ 0'),
            ('max_drawdown_pct', 'Max DD %:', '0 %'),
        ]

        for i, (key, label, default) in enumerate(account_metrics):
            lbl = QLabel(label)
            val = QLabel(default)
            val.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.metric_labels[key] = val
            account_layout.addWidget(lbl, i, 0)
            account_layout.addWidget(val, i, 1)

        layout.addWidget(account_group)

        # === Risiko-Metriken ===
        risk_group = QGroupBox("Risiko")
        risk_group.setStyleSheet(StyleFactory.group_style(hex_color='#e67e22'))
        risk_layout = QGridLayout(risk_group)

        risk_metrics = [
            ('sharpe_ratio', 'Sharpe Ratio:', '0.00'),
            ('sortino_ratio', 'Sortino Ratio:', '0.00'),
            ('calmar_ratio', 'Calmar Ratio:', '0.00'),
            ('profit_factor', 'Profit Factor:', '0.00'),
        ]

        for i, (key, label, default) in enumerate(risk_metrics):
            lbl = QLabel(label)
            val = QLabel(default)
            val.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.metric_labels[key] = val
            risk_layout.addWidget(lbl, i, 0)
            risk_layout.addWidget(val, i, 1)

        layout.addWidget(risk_group)

        # === Trade-Metriken ===
        trade_group = QGroupBox("Trades")
        trade_group.setStyleSheet(StyleFactory.group_style(hex_color='#9b59b6'))
        trade_layout = QGridLayout(trade_group)

        trade_metrics = [
            ('total_trades', 'Anzahl Trades:', '0'),
            ('winning_trades', 'Gewinner:', '0'),
            ('losing_trades', 'Verlierer:', '0'),
            ('win_rate', 'Win Rate:', '0 %'),
            ('avg_trade_pnl', 'Avg Trade:', '$ 0'),
            ('max_win', 'Bester Trade:', '$ 0'),
            ('max_loss', 'Schlechtester:', '$ 0'),
        ]

        for i, (key, label, default) in enumerate(trade_metrics):
            lbl = QLabel(label)
            val = QLabel(default)
            val.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.metric_labels[key] = val
            trade_layout.addWidget(lbl, i, 0)
            trade_layout.addWidget(val, i, 1)

        layout.addWidget(trade_group)

        # === Signal-Metriken ===
        signal_group = QGroupBox("Signale")
        signal_group.setStyleSheet(StyleFactory.group_style(hex_color='#3498db'))
        signal_layout = QGridLayout(signal_group)

        signal_metrics = [
            ('buy_signals', 'BUY Signale:', '0'),
            ('sell_signals', 'SELL Signale:', '0'),
            ('hold_signals', 'HOLD Signale:', '0'),
        ]

        for i, (key, label, default) in enumerate(signal_metrics):
            lbl = QLabel(label)
            val = QLabel(default)
            val.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.metric_labels[key] = val
            signal_layout.addWidget(lbl, i, 0)
            signal_layout.addWidget(val, i, 1)

        layout.addWidget(signal_group)

        return tab

    def _create_trades_tab(self) -> QWidget:
        """Erstellt den Trades-Tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(7)
        self.trades_table.setHorizontalHeaderLabels([
            'Entry Datum', 'Exit Datum', 'Richtung', 'Entry Preis',
            'Exit Preis', 'P/L', 'P/L %'
        ])
        self.trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.trades_table.setAlternatingRowColors(True)

        layout.addWidget(self.trades_table)

        return tab

    def _create_chart_tab(self) -> QWidget:
        """Erstellt den Chart-Tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.equity_chart = ChartWidget("Equity Curve")
        layout.addWidget(self.equity_chart)

        return tab

    def _update_data_info(self):
        """Aktualisiert die Daten-Info Anzeige."""
        if self.data is None:
            return

        self.data_rows_label.setText(f"Zeilen: {len(self.data):,}")

        # Zeitraum
        if hasattr(self.data.index, 'min'):
            start = self.data.index.min()
            end = self.data.index.max()
            self.data_range_label.setText(f"Zeitraum: {start} - {end}")

        # Modell
        if self.model is not None:
            model_name = self.model_info.get('model_name', 'BILSTM')
            self.data_model_label.setText(f"Modell: {model_name}")
        elif self.signals is not None:
            self.data_model_label.setText("Modell: Vordefinierte Signale")
        else:
            self.data_model_label.setText("Modell: Nicht geladen")

    def set_data(self, data: pd.DataFrame, model=None, model_info: Dict = None,
                 signals: pd.Series = None):
        """Setzt Daten und Modell fuer Backtest."""
        self.data = data
        self.model = model
        self.model_info = model_info or {}
        self.signals = signals
        self._update_data_info()

    def _run_backtest(self):
        """Startet den Backtest."""
        if self.data is None:
            QMessageBox.warning(self, "Fehler", "Keine Daten geladen!")
            return

        if self.model is None and self.signals is None:
            QMessageBox.warning(self, "Fehler",
                              "Weder Modell noch vordefinierte Signale vorhanden!")
            return

        # Engine erstellen
        self.engine = BacktraderEngine(self.data, self.model, self.model_info)

        # Konfiguration sammeln
        config = {
            'initial_capital': self.capital_spin.value(),
            'commission': self.commission_spin.value() / 100,  # Prozent -> Dezimal
            'slippage': self.slippage_spin.value() / 100,
            'stake': self.stake_spin.value(),
            'stake_pct': self.stake_spin.value() if self.stake_pct_check.isChecked() else None,
            'allow_short': self.allow_short_check.isChecked(),
            'invert_signals': self.invert_signals_check.isChecked(),
            'signals': self.signals,
        }

        # UI aktualisieren
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate

        # Worker starten
        self.worker = BacktestWorker(self.engine, config)
        self.worker.finished.connect(self._on_backtest_finished)
        self.worker.progress.connect(self._on_progress)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, message: str):
        """Update Fortschritts-Anzeige."""
        self.status_label.setText(message)

    def _on_backtest_finished(self, result: BacktestResult):
        """Wird aufgerufen wenn Backtest fertig ist."""
        self.result = result
        self.run_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Backtest abgeschlossen!")

        self._update_results(result)

    def _on_error(self, error_msg: str):
        """Wird bei Fehlern aufgerufen."""
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Fehler: {error_msg}")
        QMessageBox.critical(self, "Backtest Fehler", error_msg)

    def _update_results(self, result: BacktestResult):
        """Aktualisiert die Ergebnis-Anzeige."""
        # Konto-Metriken
        self.metric_labels['initial_capital'].setText(f"$ {result.initial_capital:,.2f}")
        self.metric_labels['final_value'].setText(f"$ {result.final_value:,.2f}")

        # Return mit Farbe
        color = COLORS['success'] if result.total_return >= 0 else COLORS['error']
        self.metric_labels['total_return'].setText(f"$ {result.total_return:,.2f}")
        self.metric_labels['total_return'].setStyleSheet(f"color: {color};")
        self.metric_labels['total_return_pct'].setText(f"{result.total_return_pct:.2f} %")
        self.metric_labels['total_return_pct'].setStyleSheet(f"color: {color};")

        self.metric_labels['max_drawdown'].setText(f"$ {result.max_drawdown:,.2f}")
        self.metric_labels['max_drawdown_pct'].setText(f"{result.max_drawdown_pct:.2f} %")

        # Risiko-Metriken
        self.metric_labels['sharpe_ratio'].setText(f"{result.sharpe_ratio:.2f}")
        self.metric_labels['sortino_ratio'].setText(f"{result.sortino_ratio:.2f}")
        self.metric_labels['calmar_ratio'].setText(f"{result.calmar_ratio:.2f}")
        self.metric_labels['profit_factor'].setText(f"{result.profit_factor:.2f}")

        # Trade-Metriken
        self.metric_labels['total_trades'].setText(str(result.total_trades))
        self.metric_labels['winning_trades'].setText(str(result.winning_trades))
        self.metric_labels['losing_trades'].setText(str(result.losing_trades))
        self.metric_labels['win_rate'].setText(f"{result.win_rate:.1f} %")
        self.metric_labels['avg_trade_pnl'].setText(f"$ {result.avg_trade_pnl:.2f}")
        self.metric_labels['max_win'].setText(f"$ {result.max_win:.2f}")
        self.metric_labels['max_loss'].setText(f"$ {result.max_loss:.2f}")

        # Signal-Metriken
        self.metric_labels['buy_signals'].setText(str(result.buy_signals))
        self.metric_labels['sell_signals'].setText(str(result.sell_signals))
        self.metric_labels['hold_signals'].setText(str(result.hold_signals))

        # Trades-Tabelle aktualisieren
        self._update_trades_table(result.trades)

    def _update_trades_table(self, trades: list):
        """Aktualisiert die Trades-Tabelle."""
        self.trades_table.setRowCount(len(trades))

        for i, trade in enumerate(trades):
            # Entry Datum
            entry_date = trade.get('entry_date', '')
            self.trades_table.setItem(i, 0, QTableWidgetItem(str(entry_date)))

            # Exit Datum
            exit_date = trade.get('exit_date', '')
            self.trades_table.setItem(i, 1, QTableWidgetItem(str(exit_date)))

            # Richtung
            size = trade.get('size', 0)
            direction = 'LONG' if size > 0 else 'SHORT'
            self.trades_table.setItem(i, 2, QTableWidgetItem(direction))

            # Entry Preis
            entry_price = trade.get('entry_price', 0)
            self.trades_table.setItem(i, 3, QTableWidgetItem(f"$ {entry_price:,.2f}"))

            # Exit Preis
            exit_price = trade.get('exit_price', 0)
            self.trades_table.setItem(i, 4, QTableWidgetItem(f"$ {exit_price:,.2f}"))

            # P/L
            pnl = trade.get('pnl', 0)
            pnl_item = QTableWidgetItem(f"$ {pnl:,.2f}")
            if pnl >= 0:
                pnl_item.setForeground(Qt.GlobalColor.green)
            else:
                pnl_item.setForeground(Qt.GlobalColor.red)
            self.trades_table.setItem(i, 5, pnl_item)

            # P/L %
            pnl_pct = trade.get('pnl_pct', 0)
            pnl_pct_item = QTableWidgetItem(f"{pnl_pct:.2f} %")
            if pnl_pct >= 0:
                pnl_pct_item.setForeground(Qt.GlobalColor.green)
            else:
                pnl_pct_item.setForeground(Qt.GlobalColor.red)
            self.trades_table.setItem(i, 6, pnl_pct_item)

    def _export_results(self):
        """Exportiert Ergebnisse als CSV."""
        if self.result is None:
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Ergebnisse exportieren", "", "CSV (*.csv)"
        )

        if not filepath:
            return

        # Metriken als DataFrame
        metrics = {
            'Metrik': [
                'Startkapital', 'Endwert', 'Return', 'Return %',
                'Max Drawdown', 'Max Drawdown %', 'Sharpe Ratio',
                'Profit Factor', 'Win Rate', 'Total Trades',
                'Winning Trades', 'Losing Trades'
            ],
            'Wert': [
                self.result.initial_capital, self.result.final_value,
                self.result.total_return, self.result.total_return_pct,
                self.result.max_drawdown, self.result.max_drawdown_pct,
                self.result.sharpe_ratio, self.result.profit_factor,
                self.result.win_rate, self.result.total_trades,
                self.result.winning_trades, self.result.losing_trades
            ]
        }

        df = pd.DataFrame(metrics)
        df.to_csv(filepath, index=False)

        # Trades separat speichern
        if self.result.trades:
            trades_path = filepath.replace('.csv', '_trades.csv')
            trades_df = pd.DataFrame(self.result.trades)
            trades_df.to_csv(trades_path, index=False)

        QMessageBox.information(self, "Export", f"Ergebnisse exportiert nach:\n{filepath}")
