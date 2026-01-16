"""
Stats Panel - Performance-Statistiken fuer den Backtest (rechte Spalte).
"""

from typing import Dict, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout,
    QGroupBox, QLabel, QScrollArea, QTextEdit
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from ..styles import StyleFactory


class StatsPanel(QWidget):
    """
    Statistik-Panel fuer den Backtester (rechte Spalte).

    Enthaelt:
    - Gewinn/Verlust Uebersicht
    - Trade-Statistik
    - Signal-Verteilung
    - Modell-Info (aufklappbar)
    """

    def __init__(self, initial_capital: float = 10000.0, parent=None):
        super().__init__(parent)
        self.initial_capital = initial_capital
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
        title = QLabel("Performance")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: white;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Gewinn/Verlust
        layout.addWidget(self._create_pnl_group())

        # Trade-Statistik
        layout.addWidget(self._create_trade_stats_group())

        # Signal-Verteilung
        layout.addWidget(self._create_signal_group())

        # Modell-Info (aufklappbar)
        layout.addWidget(self._create_model_info_group())

        layout.addStretch()

        scroll.setWidget(panel)

        # Main Layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

    def _create_pnl_group(self) -> QGroupBox:
        """Erstellt die P/L-Anzeige."""
        group = QGroupBox("Gewinn / Verlust")
        group.setStyleSheet(self._group_style((0.3, 0.9, 0.3)))
        layout = QGridLayout(group)
        layout.setColumnStretch(1, 1)

        pnl_labels = [
            ("Startkapital:", "start_capital_label", f"${self.initial_capital:,.2f}"),
            ("Aktuell:", "equity_label", f"${self.initial_capital:,.2f}"),
            ("Gesamt P/L:", "total_pnl_label", "$0.00"),
            ("P/L %:", "pnl_percent_label", "0.00%"),
            ("Max Drawdown:", "drawdown_label", "0.00%"),
        ]

        for row, (text, attr, default) in enumerate(pnl_labels):
            lbl = QLabel(text)
            lbl.setStyleSheet("color: rgb(179, 179, 179);")
            layout.addWidget(lbl, row, 0)
            value_lbl = QLabel(default)
            if row == 1:  # Aktuell
                value_lbl.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
            elif row in [2, 3]:  # P/L
                value_lbl.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
                value_lbl.setStyleSheet("color: gray;")
            else:
                value_lbl.setStyleSheet("color: white;")
            setattr(self, attr, value_lbl)
            layout.addWidget(value_lbl, row, 1)

        return group

    def _create_trade_stats_group(self) -> QGroupBox:
        """Erstellt die Trade-Statistik."""
        group = QGroupBox("Trade-Statistik")
        group.setStyleSheet(self._group_style((0.9, 0.7, 0.3)))
        layout = QGridLayout(group)
        layout.setColumnStretch(1, 1)

        trade_labels = [
            ("Anzahl Trades:", "num_trades_label", "0", "white"),
            ("Gewinner:", "winners_label", "0", "rgb(77, 230, 77)"),
            ("Verlierer:", "losers_label", "0", "rgb(230, 77, 77)"),
            ("Win-Rate:", "winrate_label", "0.00%", "white"),
            ("Avg. Gewinn:", "avg_win_label", "$0.00", "rgb(77, 230, 77)"),
            ("Avg. Verlust:", "avg_loss_label", "$0.00", "rgb(230, 77, 77)"),
        ]

        for row, (text, attr, default, color) in enumerate(trade_labels):
            lbl = QLabel(text)
            lbl.setStyleSheet("color: rgb(179, 179, 179);")
            layout.addWidget(lbl, row, 0)
            value_lbl = QLabel(default)
            value_lbl.setStyleSheet(f"color: {color};")
            setattr(self, attr, value_lbl)
            layout.addWidget(value_lbl, row, 1)

        return group

    def _create_signal_group(self) -> QGroupBox:
        """Erstellt die Signal-Verteilung."""
        group = QGroupBox("Signal-Verteilung")
        group.setStyleSheet(self._group_style((0.7, 0.7, 0.9)))
        layout = QGridLayout(group)
        layout.setColumnStretch(1, 1)

        signal_labels = [
            ("BUY:", "buy_count_label", "0", "rgb(77, 230, 77)"),
            ("SELL:", "sell_count_label", "0", "rgb(230, 77, 77)"),
            ("HOLD:", "hold_count_label", "0", "gray"),
        ]

        for row, (text, attr, default, color) in enumerate(signal_labels):
            lbl = QLabel(text)
            lbl.setStyleSheet("color: rgb(179, 179, 179);")
            layout.addWidget(lbl, row, 0)
            value_lbl = QLabel(default)
            value_lbl.setStyleSheet(f"color: {color};")
            setattr(self, attr, value_lbl)
            layout.addWidget(value_lbl, row, 1)

        return group

    def _create_model_info_group(self) -> QGroupBox:
        """Erstellt die Modell-Info Gruppe (aufklappbar)."""
        group = QGroupBox("Modell-Info")
        group.setStyleSheet(self._group_style((0.6, 0.8, 1.0)))
        group.setCheckable(True)
        group.setChecked(False)  # Standardmaessig eingeklappt
        layout = QVBoxLayout(group)

        self.model_info_text = QTextEdit()
        self.model_info_text.setReadOnly(True)
        self.model_info_text.setMaximumHeight(250)
        self.model_info_text.setStyleSheet('''
            QTextEdit {
                background-color: #1a1a2e;
                border: 1px solid #333;
                border-radius: 3px;
                color: #ccc;
                font-family: Consolas, monospace;
                font-size: 9px;
            }
        ''')
        self.model_info_text.setPlainText("Kein Modell geladen")
        layout.addWidget(self.model_info_text)

        self.model_info_group = group
        return group

    # === Oeffentliche Methoden fuer Updates ===

    def update_pnl(self, equity: float, total_pnl: float, drawdown_pct: float):
        """Aktualisiert die P/L-Anzeige."""
        self.equity_label.setText(f"${equity:,.2f}")
        self.total_pnl_label.setText(f"${total_pnl:,.2f}")

        pnl_pct = ((equity - self.initial_capital) / self.initial_capital) * 100
        self.pnl_percent_label.setText(f"{pnl_pct:.2f}%")

        # Farbe
        color = "rgb(77, 230, 77)" if total_pnl >= 0 else "rgb(230, 77, 77)"
        self.total_pnl_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.pnl_percent_label.setStyleSheet(f"color: {color}; font-weight: bold;")

        self.drawdown_label.setText(f"{drawdown_pct:.2f}%")

    def update_trade_stats(self, num_trades: int, winners: int, losers: int,
                          winrate: float, avg_win: float, avg_loss: float):
        """Aktualisiert die Trade-Statistik."""
        self.num_trades_label.setText(str(num_trades))
        self.winners_label.setText(str(winners))
        self.losers_label.setText(str(losers))
        self.winrate_label.setText(f"{winrate:.2f}%")
        self.avg_win_label.setText(f"${avg_win:,.2f}")
        self.avg_loss_label.setText(f"${avg_loss:,.2f}")

    def update_signal_counts(self, buy_count: int, sell_count: int, hold_count: int):
        """Aktualisiert die Signal-Zaehler."""
        self.buy_count_label.setText(str(buy_count))
        self.sell_count_label.setText(str(sell_count))
        self.hold_count_label.setText(str(hold_count))

    def update_model_info(self, model_info: Optional[Dict]):
        """Aktualisiert die Modell-Info Anzeige."""
        if not model_info:
            self.model_info_text.setPlainText("Kein Modell geladen")
            return

        info = model_info
        lines = []

        # Modell-Identifikation
        lines.append("=== MODELL ===")
        lines.append(f"Typ:        {info.get('model_type', '-')}")
        lines.append(f"Trainiert:  {info.get('trained_at', '-')}")
        if info.get('training_duration_sec'):
            mins = info['training_duration_sec'] / 60
            lines.append(f"Dauer:      {mins:.1f} min")

        # Architektur
        lines.append("")
        lines.append("=== ARCHITEKTUR ===")
        lines.append(f"Hidden:     {info.get('hidden_size', '-')}")
        lines.append(f"Layers:     {info.get('num_layers', '-')}")
        lines.append(f"Dropout:    {info.get('dropout', '-')}")
        lines.append(f"Klassen:    {info.get('num_classes', '-')}")

        # Daten-Parameter
        lines.append("")
        lines.append("=== DATEN ===")
        lines.append(f"Lookback:   {info.get('lookback_size', '-')}")
        lines.append(f"Lookfwd:    {info.get('lookforward_size', '-')}")
        lines.append(f"Lookahead:  {info.get('lookahead_bars', '-')}")
        lines.append(f"Split:      {info.get('train_test_split', '-')}%")

        # Samples
        if info.get('total_samples'):
            lines.append(f"Samples:    {info.get('total_samples', 0):,}")
            lines.append(f"  Train:    {info.get('train_samples', 0):,}")
            lines.append(f"  Val:      {info.get('val_samples', 0):,}")

        # Features
        lines.append("")
        lines.append("=== FEATURES ===")
        features = info.get('features', [])
        lines.append(f"Anzahl:     {len(features)}")
        if features:
            # Features in Kurzform
            feat_str = ', '.join(features[:5])
            if len(features) > 5:
                feat_str += f", +{len(features)-5}"
            lines.append(f"Liste:      {feat_str}")

        # Training
        lines.append("")
        lines.append("=== TRAINING ===")
        lines.append(f"Epochen:    {info.get('epochs_trained', '-')}/{info.get('epochs_configured', '-')}")
        lines.append(f"Batch:      {info.get('batch_size', '-')}")
        lines.append(f"LR:         {info.get('learning_rate', '-')}")
        lines.append(f"Patience:   {info.get('patience', '-')}")
        lines.append(f"Early Stop: {'Ja' if info.get('stopped_early') else 'Nein'}")

        # Ergebnisse
        lines.append("")
        lines.append("=== ERGEBNISSE ===")
        lines.append(f"Accuracy:   {info.get('best_accuracy', 0):.1f}%")
        lines.append(f"Val Loss:   {info.get('final_val_loss', 0):.4f}")
        lines.append(f"BUY Peaks:  {info.get('num_buy_peaks', '-')}")
        lines.append(f"SELL Peaks: {info.get('num_sell_peaks', '-')}")

        # Class Weights
        if info.get('class_weights'):
            cw = info['class_weights']
            lines.append("")
            lines.append("=== CLASS WEIGHTS ===")
            labels = ['HOLD', 'BUY', 'SELL']
            for i, w in enumerate(cw[:3]):
                lines.append(f"{labels[i]:6}:     {w:.2f}")

        self.model_info_text.setPlainText('\n'.join(lines))

    def reset_labels(self):
        """Setzt alle Labels zurueck."""
        self.equity_label.setText(f"${self.initial_capital:,.2f}")
        self.total_pnl_label.setText("$0.00")
        self.total_pnl_label.setStyleSheet("color: gray;")
        self.pnl_percent_label.setText("0.00%")
        self.pnl_percent_label.setStyleSheet("color: gray;")
        self.drawdown_label.setText("0.00%")

        self.num_trades_label.setText("0")
        self.winners_label.setText("0")
        self.losers_label.setText("0")
        self.winrate_label.setText("0.00%")
        self.avg_win_label.setText("$0.00")
        self.avg_loss_label.setText("$0.00")

        self.buy_count_label.setText("0")
        self.sell_count_label.setText("0")
        self.hold_count_label.setText("0")

    def _group_style(self, color: tuple) -> str:
        """Generiert GroupBox-Style mit farbigem Titel."""
        return StyleFactory.group_style(title_color=color)
