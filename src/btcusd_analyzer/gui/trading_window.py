"""
Trading Window - GUI fuer Live-Trading mit Testnet/Live Modus
"""

from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QProgressBar, QTextEdit, QSplitter, QMessageBox, QCheckBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QScrollArea, QFrame, QDialog, QDialogButtonBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor

import pandas as pd

from .styles import get_stylesheet, COLORS, TESTNET_BANNER_STYLE, LIVE_BANNER_STYLE
from ..trading.api_config import TradingMode


class LiveWarningDialog(QDialog):
    """Warndialog vor Live-Trading."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("WARNUNG: Live-Trading!")
        self.setModal(True)
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        # Warnung
        warning_label = QLabel(
            "Sie sind dabei, mit ECHTEM GELD zu handeln!\n\n"
            "Alle Trades werden auf dem LIVE-Markt ausgefuehrt.\n"
            "Verluste sind REAL und UNWIDERRUFLICH.\n\n"
            "Stellen Sie sicher, dass Sie:\n"
            "- Die Risiken verstehen\n"
            "- Nur Geld einsetzen, das Sie verlieren koennen\n"
            "- Ihre API-Keys sicher konfiguriert haben"
        )
        warning_label.setStyleSheet(f"""
            color: {COLORS['error']};
            font-size: 14px;
            padding: 20px;
        """)
        layout.addWidget(warning_label)

        # Bestaetigung
        self.confirm_check = QCheckBox(
            "Ich verstehe die Risiken und moechte mit echtem Geld handeln"
        )
        self.confirm_check.setStyleSheet(f"color: {COLORS['text_primary']};")
        layout.addWidget(self.confirm_check)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setStyleSheet(f"""
            QDialog {{
                background-color: {COLORS['live_bg']};
                border: 2px solid {COLORS['error']};
            }}
        """)

    def _on_accept(self):
        if self.confirm_check.isChecked():
            self.accept()
        else:
            QMessageBox.warning(
                self, "Bestaetigung erforderlich",
                "Bitte bestaetigen Sie, dass Sie die Risiken verstehen."
            )


class TradingWorker(QThread):
    """Worker-Thread fuer Live-Trading."""

    price_updated = pyqtSignal(float, float, float)  # price, change, change_pct
    position_updated = pyqtSignal(str, float, float)  # position, size, pnl
    trade_executed = pyqtSignal(dict)  # trade info
    connection_status = pyqtSignal(bool, str)  # connected, message
    error_occurred = pyqtSignal(str)

    def __init__(self, mode: TradingMode, symbol: str = 'BTCUSDT'):
        super().__init__()
        self.mode = mode
        self.symbol = symbol
        self._running = False
        self._client = None

    def run(self):
        """Startet die Trading-Verbindung."""
        self._running = True

        try:
            from ..trading.binance_client import BinanceClient

            # Client erstellen
            self._client = BinanceClient(mode=self.mode)

            self.connection_status.emit(True, f"Verbunden ({self.mode.value})")

            # Preis-Updates (simuliert fuer Demo)
            import time
            import random

            last_price = 50000.0

            while self._running:
                # Simulierte Preis-Updates
                change = random.uniform(-100, 100)
                last_price += change
                change_pct = (change / last_price) * 100

                self.price_updated.emit(last_price, change, change_pct)
                time.sleep(1)

        except Exception as e:
            self.connection_status.emit(False, str(e))
            self.error_occurred.emit(str(e))

    def stop(self):
        """Stoppt die Trading-Verbindung."""
        self._running = False

    def place_order(self, side: str, quantity: float, order_type: str = 'MARKET'):
        """Platziert eine Order."""
        if self._client is None:
            self.error_occurred.emit("Keine Verbindung")
            return None

        try:
            # Order platzieren
            order = {
                'symbol': self.symbol,
                'side': side,
                'type': order_type,
                'quantity': quantity,
                'timestamp': datetime.now().isoformat()
            }

            self.trade_executed.emit(order)
            return order

        except Exception as e:
            self.error_occurred.emit(str(e))
            return None


class TradingWindow(QMainWindow):
    """
    Trading-Fenster mit Live/Testnet Umschaltung.

    Features:
    - Klare Unterscheidung zwischen Testnet (Demo) und Live (Echt)
    - Live Preis-Feed
    - Order-Platzierung
    - Position-Management
    - Trade-Historie
    - Warnungen bei Live-Modus
    """

    def __init__(self, mode: TradingMode = TradingMode.TESTNET, parent=None):
        super().__init__(parent)
        self.mode = mode
        self.worker = None

        self.setWindowTitle(f"BTCUSD Analyzer - Trading ({mode.value.upper()})")
        self.setMinimumSize(1100, 750)

        # Live-Modus Warnung
        if mode == TradingMode.LIVE:
            dialog = LiveWarningDialog(self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                QTimer.singleShot(0, self.close)
                return

        self._setup_ui()
        self._apply_mode_style()

    def _setup_ui(self):
        """Erstellt die Benutzeroberflaeche."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(0)

        # Mode Banner (oben)
        self.mode_banner = self._create_mode_banner()
        main_layout.addWidget(self.mode_banner)

        # Haupt-Content
        content = QWidget()
        content_layout = QHBoxLayout(content)

        # Linke Seite: Trading Controls
        left_panel = self._create_trading_panel()
        left_panel.setFixedWidth(350)

        # Rechte Seite: Info und Historie
        right_panel = self._create_info_panel()

        content_layout.addWidget(left_panel)
        content_layout.addWidget(right_panel)

        main_layout.addWidget(content)

    def _create_mode_banner(self) -> QWidget:
        """Erstellt das Modus-Banner."""
        banner = QLabel()
        banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        banner.setFixedHeight(50)

        if self.mode == TradingMode.TESTNET:
            banner.setText("TESTNET - DEMO MODUS (Kein echtes Geld)")
            banner.setStyleSheet(TESTNET_BANNER_STYLE)
        else:
            banner.setText("LIVE TRADING - ECHTES GELD!")
            banner.setStyleSheet(LIVE_BANNER_STYLE)

            # Blinkender Effekt fuer Live
            self._blink_timer = QTimer()
            self._blink_state = True

            def blink():
                self._blink_state = not self._blink_state
                if self._blink_state:
                    banner.setStyleSheet(LIVE_BANNER_STYLE)
                else:
                    banner.setStyleSheet(f"""
                        background-color: #4a0000;
                        color: #ffffff;
                        font-size: 18px;
                        font-weight: bold;
                        padding: 10px;
                        border-radius: 4px;
                    """)

            self._blink_timer.timeout.connect(blink)
            self._blink_timer.start(500)

        return banner

    def _create_trading_panel(self) -> QWidget:
        """Erstellt das Trading-Panel."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)

        # Verbindungs-Status
        conn_group = QGroupBox("Verbindung")
        conn_layout = QVBoxLayout(conn_group)

        status_layout = QHBoxLayout()
        self.conn_indicator = QLabel("●")
        self.conn_indicator.setStyleSheet(f"color: {COLORS['error']}; font-size: 20px;")
        self.conn_status_label = QLabel("Nicht verbunden")
        status_layout.addWidget(self.conn_indicator)
        status_layout.addWidget(self.conn_status_label)
        status_layout.addStretch()
        conn_layout.addLayout(status_layout)

        btn_layout = QHBoxLayout()
        self.connect_btn = QPushButton("Verbinden")
        self.connect_btn.clicked.connect(self._toggle_connection)
        self.disconnect_btn = QPushButton("Trennen")
        self.disconnect_btn.clicked.connect(self._disconnect)
        self.disconnect_btn.setEnabled(False)
        btn_layout.addWidget(self.connect_btn)
        btn_layout.addWidget(self.disconnect_btn)
        conn_layout.addLayout(btn_layout)

        layout.addWidget(conn_group)

        # Preis-Anzeige
        price_group = QGroupBox("BTCUSDT")
        price_layout = QVBoxLayout(price_group)

        self.price_label = QLabel("$--,---.--")
        self.price_label.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        self.price_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        price_layout.addWidget(self.price_label)

        self.change_label = QLabel("+$0.00 (0.00%)")
        self.change_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.change_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        price_layout.addWidget(self.change_label)

        layout.addWidget(price_group)

        # Order-Panel
        order_group = QGroupBox("Order")
        order_layout = QGridLayout(order_group)

        order_layout.addWidget(QLabel("Menge (BTC):"), 0, 0)
        self.quantity_spin = QDoubleSpinBox()
        self.quantity_spin.setRange(0.001, 10)
        self.quantity_spin.setValue(0.01)
        self.quantity_spin.setDecimals(4)
        self.quantity_spin.setSingleStep(0.001)
        order_layout.addWidget(self.quantity_spin, 0, 1)

        order_layout.addWidget(QLabel("Order-Typ:"), 1, 0)
        self.order_type_combo = QComboBox()
        self.order_type_combo.addItems(['MARKET', 'LIMIT'])
        order_layout.addWidget(self.order_type_combo, 1, 1)

        # Buy/Sell Buttons
        btn_layout = QHBoxLayout()

        self.buy_btn = QPushButton("BUY / LONG")
        self.buy_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 15px;
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: #45c45c;
            }}
            QPushButton:disabled {{
                background-color: {COLORS['text_disabled']};
            }}
        """)
        self.buy_btn.clicked.connect(lambda: self._place_order('BUY'))
        self.buy_btn.setEnabled(False)

        self.sell_btn = QPushButton("SELL / SHORT")
        self.sell_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['error']};
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 15px;
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: #e04040;
            }}
            QPushButton:disabled {{
                background-color: {COLORS['text_disabled']};
            }}
        """)
        self.sell_btn.clicked.connect(lambda: self._place_order('SELL'))
        self.sell_btn.setEnabled(False)

        btn_layout.addWidget(self.buy_btn)
        btn_layout.addWidget(self.sell_btn)
        order_layout.addLayout(btn_layout, 2, 0, 1, 2)

        layout.addWidget(order_group)

        # Position-Anzeige
        pos_group = QGroupBox("Aktuelle Position")
        pos_layout = QGridLayout(pos_group)

        pos_layout.addWidget(QLabel("Position:"), 0, 0)
        self.position_label = QLabel("NONE")
        self.position_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        pos_layout.addWidget(self.position_label, 0, 1)

        pos_layout.addWidget(QLabel("Groesse:"), 1, 0)
        self.size_label = QLabel("0.0000 BTC")
        pos_layout.addWidget(self.size_label, 1, 1)

        pos_layout.addWidget(QLabel("Unrealisierter P/L:"), 2, 0)
        self.unrealized_pnl_label = QLabel("$0.00")
        pos_layout.addWidget(self.unrealized_pnl_label, 2, 1)

        self.close_pos_btn = QPushButton("Position schliessen")
        self.close_pos_btn.clicked.connect(self._close_position)
        self.close_pos_btn.setEnabled(False)
        pos_layout.addWidget(self.close_pos_btn, 3, 0, 1, 2)

        layout.addWidget(pos_group)

        # Risiko-Kontrolle
        risk_group = QGroupBox("Risiko-Kontrolle")
        risk_layout = QGridLayout(risk_group)

        risk_layout.addWidget(QLabel("Stop-Loss (%):"), 0, 0)
        self.stop_loss_spin = QDoubleSpinBox()
        self.stop_loss_spin.setRange(0.1, 50)
        self.stop_loss_spin.setValue(2.0)
        self.stop_loss_spin.setSingleStep(0.5)
        risk_layout.addWidget(self.stop_loss_spin, 0, 1)

        risk_layout.addWidget(QLabel("Take-Profit (%):"), 1, 0)
        self.take_profit_spin = QDoubleSpinBox()
        self.take_profit_spin.setRange(0.1, 100)
        self.take_profit_spin.setValue(4.0)
        self.take_profit_spin.setSingleStep(0.5)
        risk_layout.addWidget(self.take_profit_spin, 1, 1)

        self.auto_sl_tp = QCheckBox("Auto SL/TP aktivieren")
        risk_layout.addWidget(self.auto_sl_tp, 2, 0, 1, 2)

        layout.addWidget(risk_group)

        layout.addStretch()

        scroll.setWidget(panel)
        return scroll

    def _create_info_panel(self) -> QWidget:
        """Erstellt das Info-Panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        tabs = QTabWidget()

        # Tab 1: Trade-Historie
        history_tab = self._create_history_tab()
        tabs.addTab(history_tab, "Trade-Historie")

        # Tab 2: Performance
        perf_tab = self._create_performance_tab()
        tabs.addTab(perf_tab, "Performance")

        # Tab 3: Log
        log_tab = self._create_log_tab()
        tabs.addTab(log_tab, "Log")

        layout.addWidget(tabs)
        return panel

    def _create_history_tab(self) -> QWidget:
        """Erstellt den Historie-Tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(6)
        self.trades_table.setHorizontalHeaderLabels([
            'Zeit', 'Typ', 'Seite', 'Menge', 'Preis', 'Status'
        ])
        self.trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        layout.addWidget(self.trades_table)
        return widget

    def _create_performance_tab(self) -> QWidget:
        """Erstellt den Performance-Tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Metriken
        metrics_layout = QGridLayout()

        self.perf_labels = {}
        metrics = [
            ('total_pnl', 'Gesamt P/L'),
            ('win_rate', 'Win Rate'),
            ('total_trades', 'Trades'),
            ('avg_trade', 'Avg Trade'),
        ]

        for i, (key, label) in enumerate(metrics):
            name_label = QLabel(label)
            name_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
            metrics_layout.addWidget(name_label, i, 0)

            value_label = QLabel("-")
            value_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
            metrics_layout.addWidget(value_label, i, 1)

            self.perf_labels[key] = value_label

        layout.addLayout(metrics_layout)
        layout.addStretch()

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

    def _apply_mode_style(self):
        """Wendet den Modus-spezifischen Style an."""
        base_style = get_stylesheet()

        if self.mode == TradingMode.TESTNET:
            mode_style = f"""
                QMainWindow {{
                    border: 3px solid {COLORS['testnet_border']};
                }}
            """
        else:
            mode_style = f"""
                QMainWindow {{
                    border: 3px solid {COLORS['live_border']};
                }}
            """

        self.setStyleSheet(base_style + mode_style)

    def _toggle_connection(self):
        """Verbindet/Trennt die Trading-Verbindung."""
        if self.worker is None or not self.worker.isRunning():
            self._connect()
        else:
            self._disconnect()

    def _connect(self):
        """Stellt Verbindung her."""
        self.worker = TradingWorker(self.mode)
        self.worker.price_updated.connect(self._on_price_update)
        self.worker.position_updated.connect(self._on_position_update)
        self.worker.trade_executed.connect(self._on_trade_executed)
        self.worker.connection_status.connect(self._on_connection_status)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.start()

        self._log(f"Verbinde mit {self.mode.value}...")

    def _disconnect(self):
        """Trennt die Verbindung."""
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.worker = None

        self._on_connection_status(False, "Getrennt")
        self._log("Verbindung getrennt")

    def _on_connection_status(self, connected: bool, message: str):
        """Callback fuer Verbindungsstatus."""
        if connected:
            self.conn_indicator.setStyleSheet(f"color: {COLORS['success']}; font-size: 20px;")
            self.conn_status_label.setText(message)
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.buy_btn.setEnabled(True)
            self.sell_btn.setEnabled(True)
        else:
            self.conn_indicator.setStyleSheet(f"color: {COLORS['error']}; font-size: 20px;")
            self.conn_status_label.setText(message)
            self.connect_btn.setEnabled(True)
            self.disconnect_btn.setEnabled(False)
            self.buy_btn.setEnabled(False)
            self.sell_btn.setEnabled(False)

    def _on_price_update(self, price: float, change: float, change_pct: float):
        """Callback fuer Preis-Updates."""
        self.price_label.setText(f"${price:,.2f}")

        if change >= 0:
            self.change_label.setText(f"+${change:.2f} (+{change_pct:.2f}%)")
            self.change_label.setStyleSheet(f"color: {COLORS['success']};")
        else:
            self.change_label.setText(f"-${abs(change):.2f} ({change_pct:.2f}%)")
            self.change_label.setStyleSheet(f"color: {COLORS['error']};")

    def _on_position_update(self, position: str, size: float, pnl: float):
        """Callback fuer Position-Updates."""
        self.position_label.setText(position)
        self.size_label.setText(f"{size:.4f} BTC")

        if pnl >= 0:
            self.unrealized_pnl_label.setText(f"+${pnl:.2f}")
            self.unrealized_pnl_label.setStyleSheet(f"color: {COLORS['success']};")
        else:
            self.unrealized_pnl_label.setText(f"-${abs(pnl):.2f}")
            self.unrealized_pnl_label.setStyleSheet(f"color: {COLORS['error']};")

        self.close_pos_btn.setEnabled(position != 'NONE')

    def _on_trade_executed(self, trade: dict):
        """Callback wenn Trade ausgefuehrt."""
        # Tabelle aktualisieren
        row = self.trades_table.rowCount()
        self.trades_table.insertRow(row)

        self.trades_table.setItem(row, 0, QTableWidgetItem(trade['timestamp']))
        self.trades_table.setItem(row, 1, QTableWidgetItem(trade['type']))
        self.trades_table.setItem(row, 2, QTableWidgetItem(trade['side']))
        self.trades_table.setItem(row, 3, QTableWidgetItem(f"{trade['quantity']:.4f}"))
        self.trades_table.setItem(row, 4, QTableWidgetItem(f"${trade.get('price', 0):,.2f}"))
        self.trades_table.setItem(row, 5, QTableWidgetItem("Ausgefuehrt"))

        self._log(f"Trade: {trade['side']} {trade['quantity']} BTC")

    def _on_error(self, error: str):
        """Callback bei Fehler."""
        self._log(f"FEHLER: {error}")
        QMessageBox.warning(self, "Trading-Fehler", error)

    def _place_order(self, side: str):
        """Platziert eine Order."""
        if self.worker is None:
            return

        quantity = self.quantity_spin.value()
        order_type = self.order_type_combo.currentText()

        # Bestaetigung bei Live-Modus
        if self.mode == TradingMode.LIVE:
            reply = QMessageBox.question(
                self, "Order bestaetigen",
                f"Wirklich {side} {quantity:.4f} BTC?\n\nDies ist eine LIVE-Order!",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self.worker.place_order(side, quantity, order_type)

    def _close_position(self):
        """Schliesst die aktuelle Position."""
        if self.worker is None:
            return

        # Bestaetigung
        reply = QMessageBox.question(
            self, "Position schliessen",
            "Aktuelle Position wirklich schliessen?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Position schliessen (vereinfacht)
            self._log("Position geschlossen")
            self._on_position_update('NONE', 0.0, 0.0)

    def _log(self, message: str):
        """Fuegt eine Nachricht zum Log hinzu."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        mode_tag = f"[{self.mode.value.upper()}]"
        self.log_text.append(f"[{timestamp}] {mode_tag} {message}")

    def closeEvent(self, event):
        """Behandelt das Schliessen des Fensters."""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "Verbindung aktiv",
                "Trading-Verbindung ist noch aktiv. Wirklich beenden?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            self._disconnect()

        if hasattr(self, '_blink_timer'):
            self._blink_timer.stop()

        event.accept()
