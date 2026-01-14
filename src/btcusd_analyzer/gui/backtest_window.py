"""
Backtest Window - GUI fuer Backtesting mit Live-Simulation (MATLAB-Style)

Layout: 3-Spalten (280px | flexible | 280px)
- Links: Steuerung (Start/Stop/Einzelschritt/Reset, Geschwindigkeit, Position-Info, Trade-Log)
- Mitte: Charts (Preis+Signale, Equity-Kurve)
- Rechts: Performance-Statistiken (P/L, Trade-Stats, Signal-Verteilung)
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QSlider, QProgressBar, QTextEdit,
    QSplitter, QCheckBox, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont

import pandas as pd
import numpy as np

from .styles import get_stylesheet, COLORS, StyleFactory


class BacktestWindow(QMainWindow):
    """
    Backtest-Fenster mit Live-Simulation (MATLAB-Style).

    Features:
    - Schritt-fuer-Schritt Durchlauf der Daten
    - Start/Stop/Einzelschritt Steuerung
    - Gewinn/Verlust Berechnung
    - Visualisierung von Trades und Equity-Kurve
    - Performance-Statistiken
    """

    # Signal fuer Log-Meldungen an MainWindow
    log_message = pyqtSignal(str, str)  # message, level

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("4 - Backtest")
        self.setMinimumSize(1200, 800)
        self._parent = parent

        # Daten
        self.data: Optional[pd.DataFrame] = None
        self.signals: Optional[pd.Series] = None
        self.model = None
        self.model_info: Optional[Dict] = None

        # Backtester-Status
        self.is_running = False
        self.is_paused = False
        self.current_index = 0
        self.sequence_length = 0

        # Trading-Status
        self.position = 'NONE'  # 'NONE', 'LONG', 'SHORT'
        self.entry_price = 0.0
        self.entry_index = 0
        self.total_pnl = 0.0
        self.trades: List[Dict] = []
        self.signal_history: List[Dict] = []

        # Kapital und Equity
        self.initial_capital = 10000.0
        self.current_equity = self.initial_capital
        self.equity_curve: List[float] = [self.initial_capital]
        self.equity_indices: List[int] = [0]

        # Signal-Zaehler
        self.buy_count = 0
        self.sell_count = 0
        self.hold_count = 0

        # Geschwindigkeit
        self.steps_per_second = 10
        self.turbo_mode = False

        # Vorbereitete Sequenzen fuer Modell-Vorhersage
        self.prepared_sequences = None
        self.sequence_offset = 0  # Index-Offset zwischen Daten und Sequenzen

        # Timer
        self.backtest_timer: Optional[QTimer] = None

        self._setup_ui()
        self.setStyleSheet(get_stylesheet())

    def _log(self, message: str, level: str = 'INFO'):
        """Loggt eine Nachricht an MainWindow und lokales Log."""
        # An MainWindow senden (falls parent _log hat)
        if self._parent and hasattr(self._parent, '_log'):
            self._parent._log(f'[Backtest] {message}', level)
        # Signal emittieren
        self.log_message.emit(message, level)
        # Auch lokal loggen falls vorhanden
        if hasattr(self, 'trade_log'):
            self.trade_log.append(f'[{level}] {message}')

    def _setup_ui(self):
        """Erstellt die 3-Spalten Benutzeroberflaeche."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 3-Spalten Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Linke Spalte: Steuerung (280px)
        left_panel = self._create_control_panel()
        left_panel.setFixedWidth(280)

        # Mitte: Charts (flexibel)
        center_panel = self._create_chart_panel()

        # Rechte Spalte: Statistiken (280px)
        right_panel = self._create_stats_panel()
        right_panel.setFixedWidth(280)

        splitter.addWidget(left_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([280, 640, 280])

        main_layout.addWidget(splitter)

    def _create_control_panel(self) -> QWidget:
        """Erstellt das Steuerungs-Panel (linke Spalte)."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        panel = QWidget()
        panel.setStyleSheet(f"background-color: rgb(46, 46, 46);")
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # Titel
        title = QLabel("Backtester Steuerung")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: white;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # === Steuerungs-Buttons ===
        control_group = QGroupBox("Steuerung")
        control_group.setStyleSheet(self._group_style((0.3, 0.7, 1)))
        control_layout = QGridLayout(control_group)
        control_layout.setSpacing(8)

        # Start Button
        self.start_btn = QPushButton("Start")
        self.start_btn.setStyleSheet(self._button_style((0.2, 0.7, 0.3)))
        self.start_btn.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.start_btn.clicked.connect(self._start_backtest)
        control_layout.addWidget(self.start_btn, 0, 0)

        # Stop Button
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet(self._button_style((0.8, 0.3, 0.2)))
        self.stop_btn.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_backtest)
        control_layout.addWidget(self.stop_btn, 0, 1)

        # Einzelschritt Button
        self.step_btn = QPushButton("Einzelschritt")
        self.step_btn.setStyleSheet(self._button_style((0.5, 0.5, 0.7)))
        self.step_btn.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.step_btn.clicked.connect(self._single_step)
        control_layout.addWidget(self.step_btn, 1, 0)

        # Reset Button
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setStyleSheet(self._button_style((0.5, 0.5, 0.5)))
        self.reset_btn.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.reset_btn.clicked.connect(self._reset_backtest)
        control_layout.addWidget(self.reset_btn, 1, 1)

        layout.addWidget(control_group)

        # === Geschwindigkeit ===
        speed_group = QGroupBox("Geschwindigkeit")
        speed_group.setStyleSheet(self._group_style((0.9, 0.7, 0.3)))
        speed_layout = QVBoxLayout(speed_group)

        # Slider mit Label
        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Schritte/Sek:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 500)
        self.speed_slider.setValue(10)
        self.speed_slider.valueChanged.connect(self._update_speed)
        slider_row.addWidget(self.speed_slider)
        self.speed_label = QLabel("10")
        self.speed_label.setStyleSheet("color: white; min-width: 30px;")
        slider_row.addWidget(self.speed_label)
        speed_layout.addLayout(slider_row)

        # Turbo-Modus
        self.turbo_check = QCheckBox("Turbo-Modus (keine Chart-Updates)")
        self.turbo_check.setStyleSheet("color: rgb(128, 255, 128);")
        self.turbo_check.toggled.connect(self._toggle_turbo)
        speed_layout.addWidget(self.turbo_check)

        # Aktuelle Geschwindigkeit
        speed_info = QHBoxLayout()
        speed_info.addWidget(QLabel("Aktuell:"))
        self.actual_speed_label = QLabel("- Schritte/Sek")
        self.actual_speed_label.setStyleSheet("color: rgb(77, 230, 255); font-weight: bold;")
        speed_info.addWidget(self.actual_speed_label)
        speed_layout.addLayout(speed_info)

        layout.addWidget(speed_group)

        # === Aktuelle Position ===
        position_group = QGroupBox("Aktuelle Position")
        position_group.setStyleSheet(self._group_style((0.5, 0.9, 0.5)))
        pos_layout = QGridLayout(position_group)
        pos_layout.setColumnStretch(1, 1)

        labels = [
            ("Position:", "position_label", "NONE"),
            ("Einstiegspreis:", "entry_price_label", "-"),
            ("Aktueller Preis:", "current_price_label", "-"),
            ("Unrealisiert:", "unrealized_pnl_label", "-"),
        ]

        for row, (text, attr, default) in enumerate(labels):
            pos_layout.addWidget(QLabel(text), row, 0)
            label = QLabel(default)
            label.setStyleSheet("color: white;")
            setattr(self, attr, label)
            pos_layout.addWidget(label, row, 1)

        layout.addWidget(position_group)

        # === Fortschritt ===
        progress_group = QGroupBox("Fortschritt")
        progress_group.setStyleSheet(self._group_style((0.7, 0.7, 0.7)))
        prog_layout = QGridLayout(progress_group)
        prog_layout.setColumnStretch(1, 1)

        prog_labels = [
            ("Datenpunkt:", "datapoint_label", "0 / 0"),
            ("Datum:", "date_label", "-"),
            ("Letztes Signal:", "signal_label", "-"),
        ]

        for row, (text, attr, default) in enumerate(prog_labels):
            prog_layout.addWidget(QLabel(text), row, 0)
            label = QLabel(default)
            label.setStyleSheet("color: white;" if row < 2 else "color: gray; font-weight: bold;")
            setattr(self, attr, label)
            prog_layout.addWidget(label, row, 1)

        layout.addWidget(progress_group)

        # === Trade-Log ===
        tradelog_group = QGroupBox("Trade-Log")
        tradelog_group.setStyleSheet(self._group_style((0.9, 0.5, 0.9)))
        tradelog_layout = QVBoxLayout(tradelog_group)

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
        tradelog_layout.addWidget(self.tradelog_text)

        layout.addWidget(tradelog_group)

        layout.addStretch()

        # Schliessen Button
        close_btn = QPushButton("Schliessen")
        close_btn.setStyleSheet(self._button_style((0.4, 0.4, 0.4)))
        close_btn.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        scroll.setWidget(panel)
        return scroll

    def _create_chart_panel(self) -> QWidget:
        """Erstellt das Chart-Panel (mittlere Spalte)."""
        panel = QWidget()
        panel.setStyleSheet(f"background-color: rgb(46, 46, 46);")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure

            # Preis-Chart (2/3 der Hoehe)
            self.price_figure = Figure(figsize=(8, 4), facecolor='#262626')
            self.price_canvas = FigureCanvas(self.price_figure)
            self.ax_price = self.price_figure.add_subplot(111)
            self._setup_price_chart()
            layout.addWidget(self.price_canvas, stretch=2)

            # Equity-Chart (1/3 der Hoehe)
            self.equity_figure = Figure(figsize=(8, 2), facecolor='#262626')
            self.equity_canvas = FigureCanvas(self.equity_figure)
            self.ax_equity = self.equity_figure.add_subplot(111)
            self._setup_equity_chart()
            layout.addWidget(self.equity_canvas, stretch=1)

        except ImportError:
            layout.addWidget(QLabel("matplotlib nicht installiert"))

        return panel

    def _setup_price_chart(self):
        """Konfiguriert den Preis-Chart."""
        ax = self.ax_price
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#4d4d4d')
        ax.set_title('Preis und Signale', color='white', fontsize=12)
        ax.set_ylabel('Preis (USD)', color='white')
        ax.grid(True, alpha=0.3, color='#4d4d4d')

    def _setup_equity_chart(self):
        """Konfiguriert den Equity-Chart."""
        ax = self.ax_equity
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#4d4d4d')
        ax.set_title('Equity-Kurve', color='white', fontsize=12)
        ax.set_ylabel('Equity (USD)', color='white')
        ax.grid(True, alpha=0.3, color='#4d4d4d')

    def _create_stats_panel(self) -> QWidget:
        """Erstellt das Statistik-Panel (rechte Spalte)."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        panel = QWidget()
        panel.setStyleSheet(f"background-color: rgb(46, 46, 46);")
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # Titel
        title = QLabel("Performance")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: white;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # === Gewinn/Verlust ===
        pnl_group = QGroupBox("Gewinn / Verlust")
        pnl_group.setStyleSheet(self._group_style((0.3, 0.9, 0.3)))
        pnl_layout = QGridLayout(pnl_group)
        pnl_layout.setColumnStretch(1, 1)

        pnl_labels = [
            ("Startkapital:", "start_capital_label", f"${self.initial_capital:,.2f}"),
            ("Aktuell:", "equity_label", f"${self.current_equity:,.2f}"),
            ("Gesamt P/L:", "total_pnl_label", "$0.00"),
            ("P/L %:", "pnl_percent_label", "0.00%"),
            ("Max Drawdown:", "drawdown_label", "0.00%"),
        ]

        for row, (text, attr, default) in enumerate(pnl_labels):
            lbl = QLabel(text)
            lbl.setStyleSheet("color: rgb(179, 179, 179);")
            pnl_layout.addWidget(lbl, row, 0)
            value_lbl = QLabel(default)
            if row == 1:  # Aktuell
                value_lbl.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
            elif row in [2, 3]:  # P/L
                value_lbl.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
                value_lbl.setStyleSheet("color: gray;")
            else:
                value_lbl.setStyleSheet("color: white;")
            setattr(self, attr, value_lbl)
            pnl_layout.addWidget(value_lbl, row, 1)

        layout.addWidget(pnl_group)

        # === Trade-Statistik ===
        stats_group = QGroupBox("Trade-Statistik")
        stats_group.setStyleSheet(self._group_style((0.9, 0.7, 0.3)))
        stats_layout = QGridLayout(stats_group)
        stats_layout.setColumnStretch(1, 1)

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
            stats_layout.addWidget(lbl, row, 0)
            value_lbl = QLabel(default)
            value_lbl.setStyleSheet(f"color: {color};")
            setattr(self, attr, value_lbl)
            stats_layout.addWidget(value_lbl, row, 1)

        layout.addWidget(stats_group)

        # === Signal-Verteilung ===
        signal_group = QGroupBox("Signal-Verteilung")
        signal_group.setStyleSheet(self._group_style((0.7, 0.7, 0.9)))
        signal_layout = QGridLayout(signal_group)
        signal_layout.setColumnStretch(1, 1)

        signal_labels = [
            ("BUY:", "buy_count_label", "0", "rgb(77, 230, 77)"),
            ("SELL:", "sell_count_label", "0", "rgb(230, 77, 77)"),
            ("HOLD:", "hold_count_label", "0", "gray"),
        ]

        for row, (text, attr, default, color) in enumerate(signal_labels):
            lbl = QLabel(text)
            lbl.setStyleSheet("color: rgb(179, 179, 179);")
            signal_layout.addWidget(lbl, row, 0)
            value_lbl = QLabel(default)
            value_lbl.setStyleSheet(f"color: {color};")
            setattr(self, attr, value_lbl)
            signal_layout.addWidget(value_lbl, row, 1)

        layout.addWidget(signal_group)

        # === Modell-Info (aufklappbar) ===
        self.model_info_group = QGroupBox("Modell-Info")
        self.model_info_group.setStyleSheet(self._group_style((0.6, 0.8, 1.0)))
        self.model_info_group.setCheckable(True)
        self.model_info_group.setChecked(False)  # Standardmaessig eingeklappt
        model_info_layout = QVBoxLayout(self.model_info_group)

        # TextEdit fuer formatierte Modell-Informationen
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
        model_info_layout.addWidget(self.model_info_text)

        layout.addWidget(self.model_info_group)

        layout.addStretch()

        scroll.setWidget(panel)
        return scroll

    def _group_style(self, color: tuple) -> str:
        """Generiert GroupBox-Style mit farbigem Titel."""
        return StyleFactory.group_style(title_color=color)

    def _button_style(self, color: tuple) -> str:
        """Generiert Button-Style aus RGB-Tuple (0-1 Range)."""
        return StyleFactory.button_style(color)

    def set_data(self, data: pd.DataFrame, signals: pd.Series = None,
                 model=None, model_info: Dict = None):
        """Setzt die Backtest-Daten."""
        self.data = data
        self.signals = signals
        self.model = model
        self.model_info = model_info

        if model_info:
            lookback = model_info.get('lookback_size', 60)
            lookforward = model_info.get('lookforward_size', 5)
            self.sequence_length = lookback + lookforward
            self.current_index = self.sequence_length + 1

            # Sequenzen fuer Modell-Vorhersage vorbereiten
            if model is not None and data is not None:
                self._prepare_sequences(lookback)

            # Model-Info Anzeige aktualisieren
            self._update_model_info_display()

        self._update_datapoint_label()
        self._initialize_charts()

    def _prepare_sequences(self, lookback: int):
        """Bereitet Sequenzen fuer Modell-Vorhersagen vor."""
        try:
            import torch
            from ..data.processor import FeatureProcessor
            from ..training.normalizer import ZScoreNormalizer

            # Features aus model_info holen (gleiche wie beim Training)
            features = self.model_info.get('features', ['Open', 'High', 'Low', 'Close', 'PriceChange', 'PriceChangePct'])
            self._log(f"Features aus Training: {features}")

            # Features berechnen
            processor = FeatureProcessor(features=features)
            processed = processor.process(self.data)

            # Feature-Matrix (nur trainierte Features)
            feature_cols = [f for f in features if f in processed.columns]
            if not feature_cols:
                # Fallback auf OHLC
                feature_cols = ['Open', 'High', 'Low', 'Close']
                self._log(f"Fallback auf OHLC-Features", 'WARNING')

            feature_data = processed[feature_cols].values.astype(np.float32)

            # NaN behandeln
            feature_data = np.nan_to_num(feature_data, nan=0.0)

            # Normalisierung (wie beim Training)
            normalizer = ZScoreNormalizer()
            feature_data = normalizer.fit_transform(feature_data)

            # Sequenzen erstellen
            sequences = []
            for i in range(lookback, len(feature_data)):
                seq = feature_data[i - lookback:i]
                sequences.append(seq)

            if sequences:
                self.prepared_sequences = np.array(sequences)
                self.sequence_offset = lookback
                self._log(f"Sequenzen vorbereitet: {len(sequences)} ({len(feature_cols)} Features)")
            else:
                self._log("Keine Sequenzen generiert", 'WARNING')

        except Exception as e:
            import traceback
            self._log(f"Sequenz-Vorbereitung fehlgeschlagen: {e}", 'ERROR')
            self._log(traceback.format_exc(), 'ERROR')

    def _update_model_info_display(self):
        """Aktualisiert die Modell-Info Anzeige im Stats-Panel."""
        if not self.model_info:
            self.model_info_text.setPlainText("Kein Modell geladen")
            return

        info = self.model_info
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

    def _start_backtest(self):
        """Startet den Backtest."""
        if self.data is None:
            return

        if self.is_running:
            return

        self.is_running = True
        self.is_paused = False

        # Buttons aktualisieren
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.step_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)

        # Timer starten
        interval = int(1000 / self.steps_per_second)
        self.backtest_timer = QTimer()
        self.backtest_timer.timeout.connect(self._timer_callback)
        self.backtest_timer.start(interval)

        # Geschwindigkeit anzeigen
        self.actual_speed_label.setText(f"{self.steps_per_second} Schritte/Sek")

        self._add_tradelog("Backtest gestartet")

    def _stop_backtest(self):
        """Stoppt den Backtest."""
        self.is_running = False
        self.is_paused = True

        if self.backtest_timer:
            self.backtest_timer.stop()
            self.backtest_timer = None

        # Buttons aktualisieren
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.step_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)

        self.actual_speed_label.setText("Gestoppt")
        self._add_tradelog("Backtest gestoppt")

        # Charts final aktualisieren
        self._update_charts()

    def _reset_backtest(self):
        """Setzt den Backtest zurueck."""
        # Timer stoppen
        if self.backtest_timer:
            self.backtest_timer.stop()
            self.backtest_timer = None

        # Status zuruecksetzen
        self.is_running = False
        self.is_paused = False
        self.current_index = self.sequence_length + 1 if self.sequence_length > 0 else 0

        self.position = 'NONE'
        self.entry_price = 0.0
        self.entry_index = 0
        self.total_pnl = 0.0
        self.trades = []
        self.signal_history = []

        self.current_equity = self.initial_capital
        self.equity_curve = [self.initial_capital]
        self.equity_indices = [self.current_index]

        self.buy_count = 0
        self.sell_count = 0
        self.hold_count = 0

        # UI zuruecksetzen
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.step_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)

        self._reset_labels()
        self._initialize_charts()
        self.tradelog_text.setPlainText("Kein Trade")

    def _single_step(self):
        """Fuehrt einen einzelnen Schritt aus."""
        if self.data is None:
            return

        if self.current_index <= len(self.data):
            self._process_step()
            self._update_ui()
            self._update_charts()

    def _timer_callback(self):
        """Timer-Callback fuer automatischen Durchlauf."""
        if not self.is_running:
            return

        if self.data is None or self.current_index > len(self.data):
            self._finalize_backtest()
            return

        self._process_step()
        self._update_ui()

        # Charts nur im Nicht-Turbo-Modus aktualisieren
        if not self.turbo_mode and self.current_index % 10 == 0:
            self._update_charts()

    def _process_step(self):
        """Verarbeitet einen einzelnen Backtest-Schritt."""
        if self.data is None or self.current_index > len(self.data):
            return

        idx = self.current_index - 1  # 0-basiert

        # Aktueller Preis
        current_price = self.data['Close'].iloc[idx]

        # Signal ermitteln (aus Signalen oder Modell)
        signal = self._get_signal(idx)

        # Signal speichern
        self.signal_history.append({
            'index': self.current_index,
            'signal': signal,
            'price': current_price
        })

        # Signal-Zaehler
        if signal == 1:
            self.buy_count += 1
        elif signal == 2:
            self.sell_count += 1
        else:
            self.hold_count += 1

        # Trading-Logik
        self._process_signal(signal, current_price, self.current_index)

        # Equity aktualisieren
        self._update_equity(current_price)

        # Naechster Schritt
        self.current_index += 1

    def _get_signal(self, idx: int) -> int:
        """Ermittelt das Signal fuer den aktuellen Index."""
        # Wenn Signale vorhanden, diese verwenden
        if self.signals is not None and idx < len(self.signals):
            sig = self.signals.iloc[idx]
            if sig == 'BUY' or sig == 1:
                return 1
            elif sig == 'SELL' or sig == 2:
                return 2
            else:
                return 0

        # Modell-Vorhersage wenn verfuegbar
        if self.model is not None and self.prepared_sequences is not None:
            try:
                import torch
                # Pruefe ob Index im Bereich der vorbereiteten Sequenzen liegt
                seq_idx = idx - self.sequence_offset
                if 0 <= seq_idx < len(self.prepared_sequences):
                    sequence = self.prepared_sequences[seq_idx]
                    if not isinstance(sequence, torch.Tensor):
                        sequence = torch.FloatTensor(sequence)
                    sequence = sequence.unsqueeze(0)  # Batch-Dimension hinzufuegen

                    self.model.eval()
                    with torch.no_grad():
                        prediction = self.model.predict(sequence)
                        return int(prediction[0])
            except Exception:
                pass

        return 0  # HOLD als Default

    def _process_signal(self, signal: int, price: float, idx: int):
        """Verarbeitet ein Trading-Signal."""
        # signal: 0=HOLD, 1=BUY, 2=SELL

        if signal == 1:  # BUY
            if self.position == 'SHORT':
                self._close_trade(price, idx, 'BUY Signal')
                self._open_trade('LONG', price, idx)
            elif self.position == 'NONE':
                self._open_trade('LONG', price, idx)

        elif signal == 2:  # SELL
            if self.position == 'LONG':
                self._close_trade(price, idx, 'SELL Signal')
                self._open_trade('SHORT', price, idx)
            elif self.position == 'NONE':
                self._open_trade('SHORT', price, idx)

    def _open_trade(self, new_position: str, price: float, idx: int):
        """Oeffnet eine neue Position."""
        self.position = new_position
        self.entry_price = price
        self.entry_index = idx

        if not self.turbo_mode:
            self._add_tradelog(f"{new_position} @ ${price:,.2f}")

    def _close_trade(self, price: float, idx: int, reason: str):
        """Schliesst die aktuelle Position."""
        if self.position == 'NONE':
            return

        # P/L berechnen
        if self.position == 'LONG':
            pnl = price - self.entry_price
        else:  # SHORT
            pnl = self.entry_price - price

        # Trade speichern
        trade = {
            'entry_index': self.entry_index,
            'exit_index': idx,
            'position': self.position,
            'entry_price': self.entry_price,
            'exit_price': price,
            'pnl': pnl,
            'reason': reason
        }
        self.trades.append(trade)

        # P/L aktualisieren
        self.total_pnl += pnl
        self.current_equity = self.initial_capital + self.total_pnl

        # Trade-Log
        pnl_str = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"
        self._add_tradelog(f"#{len(self.trades)} {self.position}: {self.entry_price:,.2f} -> {price:,.2f} ({pnl_str})")

        self.position = 'NONE'
        self.entry_price = 0.0
        self.entry_index = 0

    def _update_equity(self, current_price: float):
        """Aktualisiert die Equity-Kurve."""
        unrealized = 0.0
        if self.position == 'LONG':
            unrealized = current_price - self.entry_price
        elif self.position == 'SHORT':
            unrealized = self.entry_price - current_price

        self.equity_curve.append(self.current_equity + unrealized)
        self.equity_indices.append(self.current_index)

    def _update_ui(self):
        """Aktualisiert die UI-Elemente."""
        self._update_datapoint_label()
        self._update_position_labels()
        self._update_pnl_labels()
        self._update_trade_stats()
        self._update_signal_counts()

    def _update_datapoint_label(self):
        """Aktualisiert die Fortschritts-Anzeige."""
        total = len(self.data) if self.data is not None else 0
        self.datapoint_label.setText(f"{self.current_index} / {total}")

        if self.data is not None and self.current_index <= len(self.data):
            idx = self.current_index - 1
            if 'DateTime' in self.data.columns or self.data.index.name == 'DateTime':
                dt = self.data.index[idx] if self.data.index.name else self.data['DateTime'].iloc[idx]
                if hasattr(dt, 'strftime'):
                    self.date_label.setText(dt.strftime('%d.%m.%Y %H:%M'))

            self.current_price_label.setText(f"${self.data['Close'].iloc[idx]:,.2f}")

        # Letztes Signal
        if self.signal_history:
            last_sig = self.signal_history[-1]['signal']
            if last_sig == 1:
                self.signal_label.setText("BUY")
                self.signal_label.setStyleSheet("color: rgb(77, 230, 77); font-weight: bold;")
            elif last_sig == 2:
                self.signal_label.setText("SELL")
                self.signal_label.setStyleSheet("color: rgb(230, 77, 77); font-weight: bold;")
            else:
                self.signal_label.setText("HOLD")
                self.signal_label.setStyleSheet("color: gray; font-weight: bold;")

    def _update_position_labels(self):
        """Aktualisiert die Positions-Anzeige."""
        self.position_label.setText(self.position)
        if self.position == 'LONG':
            self.position_label.setStyleSheet("color: rgb(77, 230, 77); font-weight: bold;")
        elif self.position == 'SHORT':
            self.position_label.setStyleSheet("color: rgb(230, 77, 77); font-weight: bold;")
        else:
            self.position_label.setStyleSheet("color: gray;")

        if self.entry_price > 0:
            self.entry_price_label.setText(f"${self.entry_price:,.2f}")
        else:
            self.entry_price_label.setText("-")

        # Unrealisierter P/L
        if self.position != 'NONE' and self.data is not None and self.current_index <= len(self.data):
            current_price = self.data['Close'].iloc[self.current_index - 1]
            if self.position == 'LONG':
                unrealized = current_price - self.entry_price
            else:
                unrealized = self.entry_price - current_price

            self.unrealized_pnl_label.setText(f"${unrealized:,.2f}")
            if unrealized >= 0:
                self.unrealized_pnl_label.setStyleSheet("color: rgb(77, 230, 77);")
            else:
                self.unrealized_pnl_label.setStyleSheet("color: rgb(230, 77, 77);")
        else:
            self.unrealized_pnl_label.setText("-")
            self.unrealized_pnl_label.setStyleSheet("color: white;")

    def _update_pnl_labels(self):
        """Aktualisiert die P/L-Anzeige."""
        self.equity_label.setText(f"${self.current_equity:,.2f}")
        self.total_pnl_label.setText(f"${self.total_pnl:,.2f}")

        pnl_pct = ((self.current_equity - self.initial_capital) / self.initial_capital) * 100
        self.pnl_percent_label.setText(f"{pnl_pct:.2f}%")

        # Farbe
        color = "rgb(77, 230, 77)" if self.total_pnl >= 0 else "rgb(230, 77, 77)"
        self.total_pnl_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.pnl_percent_label.setStyleSheet(f"color: {color}; font-weight: bold;")

        # Drawdown
        if len(self.equity_curve) > 1:
            peak = max(self.equity_curve)
            current = self.equity_curve[-1]
            drawdown = ((peak - current) / peak) * 100 if peak > 0 else 0
            self.drawdown_label.setText(f"{drawdown:.2f}%")

    def _update_trade_stats(self):
        """Aktualisiert die Trade-Statistiken."""
        num_trades = len(self.trades)
        self.num_trades_label.setText(str(num_trades))

        if num_trades > 0:
            pnls = [t['pnl'] for t in self.trades]
            winners = sum(1 for p in pnls if p > 0)
            losers = sum(1 for p in pnls if p <= 0)

            self.winners_label.setText(str(winners))
            self.losers_label.setText(str(losers))
            self.winrate_label.setText(f"{(winners / num_trades) * 100:.2f}%")

            winning_pnls = [p for p in pnls if p > 0]
            losing_pnls = [p for p in pnls if p <= 0]

            if winning_pnls:
                self.avg_win_label.setText(f"${np.mean(winning_pnls):,.2f}")
            if losing_pnls:
                self.avg_loss_label.setText(f"${np.mean(losing_pnls):,.2f}")

    def _update_signal_counts(self):
        """Aktualisiert die Signal-Zaehler."""
        self.buy_count_label.setText(str(self.buy_count))
        self.sell_count_label.setText(str(self.sell_count))
        self.hold_count_label.setText(str(self.hold_count))

    def _reset_labels(self):
        """Setzt alle Labels zurueck."""
        self.position_label.setText("NONE")
        self.position_label.setStyleSheet("color: gray;")
        self.entry_price_label.setText("-")
        self.current_price_label.setText("-")
        self.unrealized_pnl_label.setText("-")
        self.signal_label.setText("-")
        self.signal_label.setStyleSheet("color: gray;")

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

        self.actual_speed_label.setText("- Schritte/Sek")

    def _initialize_charts(self):
        """Initialisiert die Charts."""
        if not hasattr(self, 'ax_price'):
            return

        # Preis-Chart
        self.ax_price.clear()
        self._setup_price_chart()

        if self.data is not None:
            self.ax_price.plot(self.data.index, self.data['Close'],
                              color='#6699e6', linewidth=1)

        self.price_figure.tight_layout()
        self.price_canvas.draw()

        # Equity-Chart
        self.ax_equity.clear()
        self._setup_equity_chart()

        self.ax_equity.axhline(y=self.initial_capital, color='gray',
                               linestyle='--', linewidth=1)

        self.equity_figure.tight_layout()
        self.equity_canvas.draw()

    def _update_charts(self):
        """Aktualisiert die Charts."""
        if not hasattr(self, 'ax_price'):
            return

        # Preis-Chart
        self.ax_price.clear()
        self._setup_price_chart()

        if self.data is not None:
            self.ax_price.plot(self.data.index, self.data['Close'],
                              color='#6699e6', linewidth=1)

            # BUY Signale
            buy_indices = [s['index'] - 1 for s in self.signal_history if s['signal'] == 1]
            if buy_indices:
                buy_prices = [self.data['Close'].iloc[i] for i in buy_indices if i < len(self.data)]
                buy_dates = [self.data.index[i] for i in buy_indices if i < len(self.data)]
                self.ax_price.scatter(buy_dates, buy_prices, marker='^', c='lime', s=50, zorder=5)

            # SELL Signale
            sell_indices = [s['index'] - 1 for s in self.signal_history if s['signal'] == 2]
            if sell_indices:
                sell_prices = [self.data['Close'].iloc[i] for i in sell_indices if i < len(self.data)]
                sell_dates = [self.data.index[i] for i in sell_indices if i < len(self.data)]
                self.ax_price.scatter(sell_dates, sell_prices, marker='v', c='red', s=50, zorder=5)

            # Aktuelle Position markieren
            if self.current_index <= len(self.data):
                idx = self.current_index - 1
                self.ax_price.axvline(x=self.data.index[idx], color='yellow',
                                      linestyle='--', linewidth=1, alpha=0.7)

        self.ax_price.set_title(f'Preis und Signale | BUY: {self.buy_count}, SELL: {self.sell_count}, HOLD: {self.hold_count}',
                                color='white', fontsize=11)
        self.price_figure.tight_layout()
        self.price_canvas.draw()

        # Equity-Chart
        self.ax_equity.clear()
        self._setup_equity_chart()

        if len(self.equity_indices) > 1 and self.data is not None:
            valid_indices = [i - 1 for i in self.equity_indices if i - 1 < len(self.data) and i > 0]
            if valid_indices:
                dates = [self.data.index[i] for i in valid_indices]
                equity_vals = self.equity_curve[:len(valid_indices)]
                self.ax_equity.plot(dates, equity_vals, color='lime', linewidth=1.5)

        self.ax_equity.axhline(y=self.initial_capital, color='gray',
                               linestyle='--', linewidth=1)

        pnl_pct = ((self.current_equity - self.initial_capital) / self.initial_capital) * 100
        self.ax_equity.set_title(f'Equity-Kurve | P/L: ${self.total_pnl:,.2f} ({pnl_pct:.2f}%)',
                                 color='white', fontsize=11)
        self.equity_figure.tight_layout()
        self.equity_canvas.draw()

    def _finalize_backtest(self):
        """Finalisiert den Backtest."""
        self._stop_backtest()

        # Offene Position schliessen
        if self.position != 'NONE' and self.data is not None:
            final_price = self.data['Close'].iloc[-1]
            self._close_trade(final_price, len(self.data), 'Backtest Ende')

        self._update_charts()
        self._add_tradelog("=== Backtest abgeschlossen ===")
        self._add_tradelog(f"Gesamt P/L: ${self.total_pnl:,.2f}")
        self._add_tradelog(f"Trades: {len(self.trades)}")

    def _update_speed(self, value: int):
        """Aktualisiert die Geschwindigkeit."""
        self.steps_per_second = value
        self.speed_label.setText(str(value))

        # Timer neu starten falls aktiv
        if self.is_running and self.backtest_timer:
            self.backtest_timer.setInterval(int(1000 / value))
            self.actual_speed_label.setText(f"{value} Schritte/Sek")

    def _toggle_turbo(self, checked: bool):
        """Schaltet den Turbo-Modus um."""
        self.turbo_mode = checked
        if not checked:
            self._update_charts()

    def _add_tradelog(self, message: str):
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

    def closeEvent(self, event):
        """Behandelt das Schliessen des Fensters."""
        if self.backtest_timer:
            self.backtest_timer.stop()
        event.accept()
