"""
Backtest Window - GUI fuer Backtesting mit Ergebnis-Visualisierung
"""

from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QProgressBar, QTextEdit, QSplitter, QFileDialog, QMessageBox,
    QCheckBox, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QScrollArea, QDateEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QDate
from PyQt6.QtGui import QFont, QColor

import pandas as pd
import numpy as np

from .styles import get_stylesheet, COLORS


class BacktestWorker(QThread):
    """Worker-Thread fuer Backtesting ohne GUI-Blockierung."""

    progress_updated = pyqtSignal(int)  # progress percent
    backtest_finished = pyqtSignal(object)  # BacktestResult
    backtest_error = pyqtSignal(str)  # error message
    log_message = pyqtSignal(str)  # log output

    def __init__(
        self,
        backtester,
        data: pd.DataFrame,
        signals: pd.Series,
        config: Dict[str, Any]
    ):
        super().__init__()
        self.backtester = backtester
        self.data = data
        self.signals = signals
        self.config = config

    def run(self):
        """Fuehrt den Backtest durch."""
        try:
            self.log_message.emit(f"Starte Backtest mit {self.backtester.name}")
            self.log_message.emit(f"Datenpunkte: {len(self.data):,}")
            self.log_message.emit(f"Startkapital: ${self.config.get('initial_capital', 10000):,.2f}")

            self.progress_updated.emit(10)

            # Backtester konfigurieren
            self.backtester.set_params(
                commission=self.config.get('commission', 0.001),
                slippage=self.config.get('slippage', 0.0),
            )

            self.progress_updated.emit(30)

            # Backtest ausfuehren
            result = self.backtester.run(
                self.data,
                self.signals,
                initial_capital=self.config.get('initial_capital', 10000)
            )

            self.progress_updated.emit(100)
            self.backtest_finished.emit(result)

        except Exception as e:
            self.backtest_error.emit(str(e))


class BacktestWindow(QMainWindow):
    """
    Backtest-Fenster mit Ergebnis-Visualisierung.

    Features:
    - Backtester-Auswahl (Internal, VectorBT, Backtrader, etc.)
    - Parameter-Konfiguration (Gebuehren, Slippage)
    - Equity-Kurve Visualisierung
    - Trade-Liste
    - Performance-Metriken
    - Export-Funktionen
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("BTCUSD Analyzer - Backtest")
        self.setMinimumSize(1200, 800)

        # State
        self.data = None
        self.signals = None
        self.result = None
        self.worker = None

        self._setup_ui()
        self.setStyleSheet(get_stylesheet())

    def _setup_ui(self):
        """Erstellt die Benutzeroberflaeche."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Linke Seite: Konfiguration
        left_panel = self._create_config_panel()
        left_panel.setFixedWidth(320)

        # Rechte Seite: Ergebnisse
        right_panel = self._create_results_panel()

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([320, 880])

        main_layout.addWidget(splitter)

    def _create_config_panel(self) -> QWidget:
        """Erstellt das Konfigurations-Panel."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        # Backtester-Auswahl
        bt_group = QGroupBox("Backtester")
        bt_layout = QVBoxLayout(bt_group)

        self.backtester_combo = QComboBox()
        self._update_backtester_list()
        bt_layout.addWidget(self.backtester_combo)

        # Info-Label
        self.bt_info_label = QLabel("Interner Backtester")
        self.bt_info_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        bt_layout.addWidget(self.bt_info_label)

        self.backtester_combo.currentTextChanged.connect(self._on_backtester_changed)

        layout.addWidget(bt_group)

        # Kapital & Kosten
        capital_group = QGroupBox("Kapital & Kosten")
        capital_layout = QGridLayout(capital_group)

        capital_layout.addWidget(QLabel("Startkapital ($):"), 0, 0)
        self.capital_spin = QDoubleSpinBox()
        self.capital_spin.setRange(100, 10000000)
        self.capital_spin.setValue(10000)
        self.capital_spin.setDecimals(2)
        self.capital_spin.setSingleStep(1000)
        capital_layout.addWidget(self.capital_spin, 0, 1)

        capital_layout.addWidget(QLabel("Gebuehren (%):"), 1, 0)
        self.commission_spin = QDoubleSpinBox()
        self.commission_spin.setRange(0, 5)
        self.commission_spin.setValue(0.1)
        self.commission_spin.setDecimals(3)
        self.commission_spin.setSingleStep(0.01)
        capital_layout.addWidget(self.commission_spin, 1, 1)

        capital_layout.addWidget(QLabel("Slippage (%):"), 2, 0)
        self.slippage_spin = QDoubleSpinBox()
        self.slippage_spin.setRange(0, 2)
        self.slippage_spin.setValue(0)
        self.slippage_spin.setDecimals(3)
        self.slippage_spin.setSingleStep(0.01)
        capital_layout.addWidget(self.slippage_spin, 2, 1)

        layout.addWidget(capital_group)

        # Zeitraum
        period_group = QGroupBox("Zeitraum")
        period_layout = QGridLayout(period_group)

        self.use_full_period = QCheckBox("Gesamten Zeitraum verwenden")
        self.use_full_period.setChecked(True)
        self.use_full_period.toggled.connect(self._toggle_date_range)
        period_layout.addWidget(self.use_full_period, 0, 0, 1, 2)

        period_layout.addWidget(QLabel("Von:"), 1, 0)
        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setEnabled(False)
        period_layout.addWidget(self.start_date, 1, 1)

        period_layout.addWidget(QLabel("Bis:"), 2, 0)
        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDate(QDate.currentDate())
        self.end_date.setEnabled(False)
        period_layout.addWidget(self.end_date, 2, 1)

        layout.addWidget(period_group)

        # Daten-Info
        data_group = QGroupBox("Daten")
        data_layout = QVBoxLayout(data_group)

        self.data_status_label = QLabel("Keine Daten geladen")
        self.data_status_label.setStyleSheet(f"color: {COLORS['warning']};")
        data_layout.addWidget(self.data_status_label)

        self.signals_status_label = QLabel("Keine Signale")
        self.signals_status_label.setStyleSheet(f"color: {COLORS['warning']};")
        data_layout.addWidget(self.signals_status_label)

        layout.addWidget(data_group)

        # Buttons
        btn_layout = QVBoxLayout()

        self.run_btn = QPushButton("Backtest starten")
        self.run_btn.setStyleSheet(f"background-color: {COLORS['success']}; font-weight: bold;")
        self.run_btn.clicked.connect(self._run_backtest)
        btn_layout.addWidget(self.run_btn)

        self.export_btn = QPushButton("Ergebnisse exportieren")
        self.export_btn.clicked.connect(self._export_results)
        self.export_btn.setEnabled(False)
        btn_layout.addWidget(self.export_btn)

        layout.addLayout(btn_layout)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        layout.addStretch()

        scroll.setWidget(panel)
        return scroll

    def _create_results_panel(self) -> QWidget:
        """Erstellt das Ergebnis-Panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Tabs
        tabs = QTabWidget()

        # Tab 1: Uebersicht
        overview_tab = self._create_overview_tab()
        tabs.addTab(overview_tab, "Uebersicht")

        # Tab 2: Equity-Kurve
        equity_tab = self._create_equity_tab()
        tabs.addTab(equity_tab, "Equity-Kurve")

        # Tab 3: Trade-Liste
        trades_tab = self._create_trades_tab()
        tabs.addTab(trades_tab, "Trades")

        # Tab 4: Log
        log_tab = self._create_log_tab()
        tabs.addTab(log_tab, "Log")

        layout.addWidget(tabs)
        return panel

    def _create_overview_tab(self) -> QWidget:
        """Erstellt den Uebersicht-Tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Haupt-Metriken (gross)
        main_metrics = QGroupBox("Haupt-Metriken")
        main_layout = QGridLayout(main_metrics)

        self.metric_labels = {}
        metrics = [
            ('total_return', 'Gesamt-Return', '%'),
            ('total_pnl', 'Gesamt P/L', '$'),
            ('win_rate', 'Win Rate', '%'),
            ('profit_factor', 'Profit Factor', ''),
            ('sharpe_ratio', 'Sharpe Ratio', ''),
            ('max_drawdown', 'Max Drawdown', '%'),
            ('num_trades', 'Anzahl Trades', ''),
            ('avg_trade', 'Avg Trade', '$'),
        ]

        for i, (key, label, suffix) in enumerate(metrics):
            row, col = divmod(i, 4)

            name_label = QLabel(label)
            name_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
            main_layout.addWidget(name_label, row * 2, col)

            value_label = QLabel("-")
            value_label.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
            value_label.setStyleSheet(f"color: {COLORS['text_primary']};")
            main_layout.addWidget(value_label, row * 2 + 1, col)

            self.metric_labels[key] = (value_label, suffix)

        layout.addWidget(main_metrics)

        # Zusaetzliche Metriken
        extra_metrics = QGroupBox("Zusaetzliche Metriken")
        extra_layout = QGridLayout(extra_metrics)

        extra_items = [
            ('avg_winner', 'Avg Winner', '$'),
            ('avg_loser', 'Avg Loser', '$'),
            ('largest_winner', 'Groesster Winner', '$'),
            ('largest_loser', 'Groesster Loser', '$'),
            ('sortino_ratio', 'Sortino Ratio', ''),
            ('avg_duration', 'Avg Trade Dauer', ''),
        ]

        for i, (key, label, suffix) in enumerate(extra_items):
            row, col = divmod(i, 3)

            name_label = QLabel(label)
            name_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
            extra_layout.addWidget(name_label, row * 2, col)

            value_label = QLabel("-")
            value_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
            extra_layout.addWidget(value_label, row * 2 + 1, col)

            self.metric_labels[key] = (value_label, suffix)

        layout.addWidget(extra_metrics)
        layout.addStretch()

        return widget

    def _create_equity_tab(self) -> QWidget:
        """Erstellt den Equity-Kurve Tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure

            self.equity_figure = Figure(figsize=(10, 6), facecolor=COLORS['bg_primary'])
            self.equity_canvas = FigureCanvas(self.equity_figure)

            self.equity_ax = self.equity_figure.add_subplot(111)
            self._setup_equity_plot_style()

            layout.addWidget(self.equity_canvas)

        except ImportError:
            layout.addWidget(QLabel("matplotlib nicht installiert"))

        return widget

    def _setup_equity_plot_style(self):
        """Konfiguriert den Equity-Plot Style."""
        ax = self.equity_ax
        ax.set_facecolor(COLORS['bg_secondary'])
        ax.tick_params(colors=COLORS['text_secondary'])
        for spine in ax.spines.values():
            spine.set_color(COLORS['border'])
        ax.set_title('Equity-Kurve', color=COLORS['text_primary'], fontsize=14)
        ax.set_xlabel('Zeit', color=COLORS['text_secondary'])
        ax.set_ylabel('Portfolio-Wert ($)', color=COLORS['text_secondary'])
        ax.grid(True, alpha=0.3, color=COLORS['border'])

    def _update_equity_plot(self):
        """Aktualisiert den Equity-Plot."""
        if not hasattr(self, 'equity_ax') or self.result is None:
            return

        self.equity_ax.clear()
        self._setup_equity_plot_style()

        equity = self.result.equity_curve
        if len(equity) > 0:
            self.equity_ax.plot(equity.index, equity.values, color=COLORS['accent'], linewidth=2)
            self.equity_ax.fill_between(equity.index, equity.values, alpha=0.3, color=COLORS['accent'])

            # Anfangskapital-Linie
            initial = self.capital_spin.value()
            self.equity_ax.axhline(y=initial, color=COLORS['text_secondary'],
                                   linestyle='--', alpha=0.5, label=f'Start: ${initial:,.0f}')
            self.equity_ax.legend(facecolor=COLORS['bg_tertiary'], labelcolor=COLORS['text_primary'])

        self.equity_figure.tight_layout()
        self.equity_canvas.draw()

    def _create_trades_tab(self) -> QWidget:
        """Erstellt den Trades-Tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Filter
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter:"))

        self.trade_filter_combo = QComboBox()
        self.trade_filter_combo.addItems(['Alle', 'Nur Gewinner', 'Nur Verlierer'])
        self.trade_filter_combo.currentTextChanged.connect(self._filter_trades)
        filter_layout.addWidget(self.trade_filter_combo)
        filter_layout.addStretch()

        layout.addLayout(filter_layout)

        # Tabelle
        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(8)
        self.trades_table.setHorizontalHeaderLabels([
            'Entry', 'Exit', 'Position', 'Entry $', 'Exit $', 'P/L $', 'P/L %', 'Dauer'
        ])
        self.trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        layout.addWidget(self.trades_table)
        return widget

    def _create_log_tab(self) -> QWidget:
        """Erstellt den Log-Tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['bg_secondary']};
                color: {COLORS['text_primary']};
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
            }}
        """)
        layout.addWidget(self.log_text)

        return widget

    def _update_backtester_list(self):
        """Aktualisiert die Liste verfuegbarer Backtester."""
        try:
            from ..backtesting.adapters import BacktesterFactory
            available = BacktesterFactory.available()
        except Exception:
            available = ['internal']

        self.backtester_combo.clear()
        self.backtester_combo.addItems(available)

    def _on_backtester_changed(self, name: str):
        """Callback wenn Backtester geaendert wird."""
        info_texts = {
            'internal': 'Interner Backtester (Referenz-Implementierung)',
            'vectorbt': 'VectorBT - Schnellstes Framework, vektorisiert',
            'backtrader': 'Backtrader - Feature-reich, grosse Community',
            'backtestingpy': 'Backtesting.py - Einfach, gute Visualisierung'
        }
        self.bt_info_label.setText(info_texts.get(name, ''))

    def _toggle_date_range(self, use_full: bool):
        """Aktiviert/Deaktiviert die Datumauswahl."""
        self.start_date.setEnabled(not use_full)
        self.end_date.setEnabled(not use_full)

    def set_data(self, data: pd.DataFrame, signals: pd.Series = None):
        """Setzt die Backtest-Daten."""
        self.data = data
        self.signals = signals

        # Status aktualisieren
        if data is not None:
            n = len(data)
            start = data.index[0].strftime('%Y-%m-%d') if hasattr(data.index[0], 'strftime') else str(data.index[0])
            end = data.index[-1].strftime('%Y-%m-%d') if hasattr(data.index[-1], 'strftime') else str(data.index[-1])
            self.data_status_label.setText(f"{n:,} Datenpunkte ({start} - {end})")
            self.data_status_label.setStyleSheet(f"color: {COLORS['success']};")

            # Datumsbereich setzen
            if hasattr(data.index[0], 'to_pydatetime'):
                self.start_date.setDate(QDate(data.index[0].year, data.index[0].month, data.index[0].day))
                self.end_date.setDate(QDate(data.index[-1].year, data.index[-1].month, data.index[-1].day))

        if signals is not None:
            n_buys = (signals == 'BUY').sum()
            n_sells = (signals == 'SELL').sum()
            self.signals_status_label.setText(f"{n_buys} BUY, {n_sells} SELL Signale")
            self.signals_status_label.setStyleSheet(f"color: {COLORS['success']};")

        self._log("Daten geladen")

    def _run_backtest(self):
        """Startet den Backtest."""
        if self.data is None:
            QMessageBox.warning(self, "Fehler", "Keine Daten geladen!")
            return

        if self.signals is None:
            QMessageBox.warning(self, "Fehler", "Keine Signale vorhanden!")
            return

        # Backtester erstellen
        try:
            from ..backtesting.adapters import BacktesterFactory
            backtester = BacktesterFactory.create(self.backtester_combo.currentText())
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Backtester-Erstellung fehlgeschlagen:\n{e}")
            return

        # Config
        config = {
            'initial_capital': self.capital_spin.value(),
            'commission': self.commission_spin.value() / 100,
            'slippage': self.slippage_spin.value() / 100,
        }

        # Daten filtern nach Zeitraum
        data = self.data
        signals = self.signals
        if not self.use_full_period.isChecked():
            start = pd.Timestamp(self.start_date.date().toPyDate())
            end = pd.Timestamp(self.end_date.date().toPyDate())
            mask = (data.index >= start) & (data.index <= end)
            data = data[mask]
            signals = signals[mask]

        # Worker starten
        self.worker = BacktestWorker(backtester, data, signals, config)
        self.worker.progress_updated.connect(self._on_progress)
        self.worker.backtest_finished.connect(self._on_backtest_finished)
        self.worker.backtest_error.connect(self._on_backtest_error)
        self.worker.log_message.connect(self._log)
        self.worker.start()

        # UI
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

    def _on_progress(self, value: int):
        """Callback fuer Fortschritt."""
        self.progress_bar.setValue(value)

    def _on_backtest_finished(self, result):
        """Callback wenn Backtest abgeschlossen."""
        self.result = result
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.export_btn.setEnabled(True)

        # Metriken aktualisieren
        self._update_metrics()

        # Equity-Plot
        self._update_equity_plot()

        # Trades-Tabelle
        self._update_trades_table()

        # Log
        self._log("\n" + result.summary())

        QMessageBox.information(
            self, "Backtest abgeschlossen",
            f"Backtest erfolgreich!\n\n"
            f"Trades: {result.num_trades}\n"
            f"Return: {result.total_return_pct:.2f}%\n"
            f"Win Rate: {result.win_rate:.1f}%"
        )

    def _on_backtest_error(self, error: str):
        """Callback bei Fehler."""
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        self._log(f"FEHLER: {error}")
        QMessageBox.critical(self, "Backtest-Fehler", error)

    def _update_metrics(self):
        """Aktualisiert die Metriken-Anzeige."""
        if self.result is None:
            return

        r = self.result

        # Werte setzen
        updates = {
            'total_return': r.total_return_pct,
            'total_pnl': r.total_pnl,
            'win_rate': r.win_rate,
            'profit_factor': r.profit_factor,
            'sharpe_ratio': r.sharpe_ratio,
            'max_drawdown': r.max_drawdown_pct,
            'num_trades': r.num_trades,
            'avg_trade': r.avg_trade_pnl,
            'avg_winner': r.avg_winner,
            'avg_loser': r.avg_loser,
            'largest_winner': r.largest_winner,
            'largest_loser': r.largest_loser,
            'sortino_ratio': r.sortino_ratio,
            'avg_duration': str(r.avg_trade_duration).split('.')[0] if r.avg_trade_duration else '-',
        }

        for key, value in updates.items():
            if key in self.metric_labels:
                label, suffix = self.metric_labels[key]

                if isinstance(value, float):
                    if suffix == '$':
                        text = f"${value:,.2f}"
                    elif suffix == '%':
                        text = f"{value:.2f}%"
                    else:
                        text = f"{value:.2f}"
                else:
                    text = str(value)

                label.setText(text)

                # Farbe basierend auf Wert
                if key in ['total_return', 'total_pnl', 'avg_trade']:
                    if isinstance(value, (int, float)) and value > 0:
                        label.setStyleSheet(f"color: {COLORS['success']};")
                    elif isinstance(value, (int, float)) and value < 0:
                        label.setStyleSheet(f"color: {COLORS['error']};")

    def _update_trades_table(self):
        """Aktualisiert die Trades-Tabelle."""
        self.trades_table.setRowCount(0)

        if self.result is None or not self.result.trades:
            return

        for trade in self.result.trades:
            row = self.trades_table.rowCount()
            self.trades_table.insertRow(row)

            # Entry/Exit Time
            entry_str = trade.entry_time.strftime('%Y-%m-%d %H:%M') if hasattr(trade.entry_time, 'strftime') else str(trade.entry_time)
            exit_str = trade.exit_time.strftime('%Y-%m-%d %H:%M') if hasattr(trade.exit_time, 'strftime') else str(trade.exit_time)

            self.trades_table.setItem(row, 0, QTableWidgetItem(entry_str))
            self.trades_table.setItem(row, 1, QTableWidgetItem(exit_str))
            self.trades_table.setItem(row, 2, QTableWidgetItem(trade.position))
            self.trades_table.setItem(row, 3, QTableWidgetItem(f"${trade.entry_price:,.2f}"))
            self.trades_table.setItem(row, 4, QTableWidgetItem(f"${trade.exit_price:,.2f}"))

            # P/L mit Farbe
            pnl_item = QTableWidgetItem(f"${trade.pnl:,.2f}")
            pnl_pct_item = QTableWidgetItem(f"{trade.pnl_pct:.2f}%")

            if trade.pnl > 0:
                pnl_item.setForeground(QColor(COLORS['success']))
                pnl_pct_item.setForeground(QColor(COLORS['success']))
            else:
                pnl_item.setForeground(QColor(COLORS['error']))
                pnl_pct_item.setForeground(QColor(COLORS['error']))

            self.trades_table.setItem(row, 5, pnl_item)
            self.trades_table.setItem(row, 6, pnl_pct_item)

            # Dauer
            duration_str = str(trade.duration).split('.')[0] if trade.duration else '-'
            self.trades_table.setItem(row, 7, QTableWidgetItem(duration_str))

    def _filter_trades(self, filter_type: str):
        """Filtert die Trades-Tabelle."""
        if self.result is None:
            return

        for row in range(self.trades_table.rowCount()):
            pnl_text = self.trades_table.item(row, 5).text()
            pnl = float(pnl_text.replace('$', '').replace(',', ''))

            show = True
            if filter_type == 'Nur Gewinner' and pnl <= 0:
                show = False
            elif filter_type == 'Nur Verlierer' and pnl >= 0:
                show = False

            self.trades_table.setRowHidden(row, not show)

    def _export_results(self):
        """Exportiert die Ergebnisse."""
        if self.result is None:
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Ergebnisse speichern", "", "JSON (*.json);;CSV (*.csv)"
        )

        if not filepath:
            return

        try:
            if filepath.endswith('.json'):
                import json
                with open(filepath, 'w') as f:
                    json.dump(self.result.to_dict(), f, indent=2)
            else:
                # Trades als CSV
                trades_data = [{
                    'entry_time': t.entry_time,
                    'exit_time': t.exit_time,
                    'position': t.position,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'pnl': t.pnl,
                    'pnl_pct': t.pnl_pct
                } for t in self.result.trades]
                pd.DataFrame(trades_data).to_csv(filepath, index=False)

            self._log(f"Ergebnisse exportiert: {filepath}")
            QMessageBox.information(self, "Export", f"Ergebnisse gespeichert:\n{filepath}")

        except Exception as e:
            QMessageBox.critical(self, "Export-Fehler", str(e))

    def _log(self, message: str):
        """Fuegt eine Nachricht zum Log hinzu."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def get_result(self):
        """Gibt das Backtest-Ergebnis zurueck."""
        return self.result
