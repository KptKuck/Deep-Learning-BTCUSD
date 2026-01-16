"""
Backtest Window - Hauptfenster fuer Backtesting mit Live-Simulation.

Orchestriert die modularen Komponenten:
- ControlPanel: Steuerung (links)
- ChartPanel: Charts (mitte)
- StatsPanel: Statistiken (rechts)
"""

import time
from typing import Optional, Dict, List

from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QSplitter
from PyQt6.QtCore import Qt, QTimer, pyqtSignal

import pandas as pd
import numpy as np

from .control_panel import ControlPanel
from .stats_panel import StatsPanel
from .chart_panel import ChartPanel
from .trade_statistics_dialog import TradeStatisticsDialog
from ..styles import get_stylesheet


class BacktestWindow(QMainWindow):
    """
    Backtest-Fenster mit Live-Simulation.

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
        self.position = 'NONE'
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

        # Geschwindigkeitsmessung
        self._step_count = 0
        self._last_speed_update = 0.0
        self._speed_update_timer: Optional[QTimer] = None

        # Vorbereitete Sequenzen fuer Modell-Vorhersage
        self.prepared_sequences = None
        self.sequence_offset = 0

        # Timer
        self.backtest_timer: Optional[QTimer] = None

        # Optionen
        self.debug_mode = False
        self.invert_signals = False

        # Benchmark-Tracking
        self._benchmark_enabled = True
        self._benchmark_data: Dict[str, List[float]] = {}
        self._benchmark_call_count: Dict[str, int] = {}

        self._setup_ui()
        self.setStyleSheet(get_stylesheet())

    def _log(self, message: str, level: str = 'INFO'):
        """Loggt eine Nachricht."""
        if self._parent and hasattr(self._parent, '_log'):
            self._parent._log(f'[Backtest] {message}', level)
        self.log_message.emit(message, level)
        self.control_panel.add_tradelog(f'[{level}] {message}')

    def _debug(self, message: str):
        """Loggt DEBUG-Nachricht nur wenn debug_mode aktiv ist."""
        if self.debug_mode:
            self._log(f'[DEBUG] {message}', 'DEBUG')

    def _benchmark_start(self) -> float:
        """Startet Zeitmessung fuer Benchmark."""
        if self._benchmark_enabled:
            return time.perf_counter()
        return 0.0

    def _benchmark_record(self, func_name: str, start_time: float):
        """Zeichnet Benchmark-Zeit auf."""
        if not self._benchmark_enabled or start_time == 0.0:
            return
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if func_name not in self._benchmark_data:
            self._benchmark_data[func_name] = []
            self._benchmark_call_count[func_name] = 0
        self._benchmark_data[func_name].append(elapsed_ms)
        self._benchmark_call_count[func_name] += 1

    def _benchmark_report(self):
        """Gibt Benchmark-Report aus."""
        if not self._benchmark_data:
            self._log("[BENCHMARK] Keine Daten gesammelt", 'INFO')
            return

        self._log("=" * 60, 'INFO')
        self._log("[BENCHMARK] Funktions-Performance Report", 'INFO')
        self._log("=" * 60, 'INFO')

        # Sortiere nach Gesamtzeit (absteigend)
        sorted_funcs = sorted(
            self._benchmark_data.keys(),
            key=lambda f: sum(self._benchmark_data[f]),
            reverse=True
        )

        for func_name in sorted_funcs:
            times = self._benchmark_data[func_name]
            count = self._benchmark_call_count[func_name]
            total_ms = sum(times)
            avg_ms = total_ms / len(times) if times else 0
            min_ms = min(times) if times else 0
            max_ms = max(times) if times else 0

            self._log(
                f"  {func_name:30s} | "
                f"Calls: {count:6d} | "
                f"Total: {total_ms:8.1f}ms | "
                f"Avg: {avg_ms:6.3f}ms | "
                f"Min: {min_ms:6.3f}ms | "
                f"Max: {max_ms:6.3f}ms",
                'INFO'
            )

        # Gesamtzeit
        total_all = sum(sum(times) for times in self._benchmark_data.values())
        self._log("-" * 60, 'INFO')
        self._log(f"  {'GESAMT':30s} | Total: {total_all:8.1f}ms", 'INFO')
        self._log("=" * 60, 'INFO')

    def _benchmark_reset(self):
        """Setzt Benchmark-Daten zurueck."""
        self._benchmark_data.clear()
        self._benchmark_call_count.clear()

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
        self.control_panel = ControlPanel()
        self.control_panel.setFixedWidth(280)
        self._connect_control_signals()

        # Mitte: Charts (flexibel)
        self.chart_panel = ChartPanel()

        # Rechte Spalte: Statistiken (280px)
        self.stats_panel = StatsPanel(self.initial_capital)
        self.stats_panel.setFixedWidth(280)

        splitter.addWidget(self.control_panel)
        splitter.addWidget(self.chart_panel)
        splitter.addWidget(self.stats_panel)
        splitter.setSizes([280, 640, 280])

        main_layout.addWidget(splitter)

    def _connect_control_signals(self):
        """Verbindet die Signale des Control-Panels."""
        self.control_panel.start_clicked.connect(self._start_backtest)
        self.control_panel.stop_clicked.connect(self._stop_backtest)
        self.control_panel.step_clicked.connect(self._single_step)
        self.control_panel.reset_clicked.connect(self._reset_backtest)
        self.control_panel.stats_clicked.connect(self._show_trade_statistics)
        self.control_panel.speed_changed.connect(self._update_speed)
        self.control_panel.turbo_toggled.connect(self._toggle_turbo)
        self.control_panel.debug_toggled.connect(self._toggle_debug)
        self.control_panel.invert_toggled.connect(self._toggle_invert)
        self.control_panel.close_clicked.connect(self.close)

    def set_data(self, data: pd.DataFrame, signals: pd.Series = None,
                 model=None, model_info: Dict = None):
        """Setzt die Backtest-Daten."""
        self._debug(f"set_data() aufgerufen")
        self._debug(f"  data: {type(data).__name__}, {len(data) if data is not None else 'None'} Zeilen")
        self._debug(f"  signals: {type(signals).__name__ if signals is not None else 'None'}")
        self._debug(f"  model: {type(model).__name__ if model is not None else 'None'}")

        self.data = data
        self.signals = signals
        self.model = model
        self.model_info = model_info

        if model_info:
            lookback = model_info.get('lookback_size', 60)
            lookforward = model_info.get('lookforward_size', 5)
            self.sequence_length = lookback + lookforward
            self.current_index = self.sequence_length + 1

            if model is not None and data is not None:
                self._prepare_sequences(lookback)

            self.stats_panel.update_model_info(model_info)

        self._update_datapoint_label()
        self.chart_panel.set_data(data, self.initial_capital)
        self.chart_panel.initialize_charts()
        self._debug(f"set_data() abgeschlossen")

    def _prepare_sequences(self, lookback: int):
        """Bereitet Sequenzen fuer Modell-Vorhersagen vor."""
        self._debug(f"_prepare_sequences(lookback={lookback}) gestartet")
        try:
            import torch
            from ...data.processor import FeatureProcessor
            from ...training.normalizer import ZScoreNormalizer

            features = self.model_info.get('features', ['Open', 'High', 'Low', 'Close', 'PriceChange', 'PriceChangePct'])
            self._log(f"Features aus Training: {features}")

            processor = FeatureProcessor(features=features)
            processed = processor.process(self.data)

            feature_cols = [f for f in features if f in processed.columns]
            if not feature_cols:
                feature_cols = ['Open', 'High', 'Low', 'Close']
                self._log(f"Fallback auf OHLC-Features", 'WARNING')

            feature_data = processed[feature_cols].values.astype(np.float32)
            feature_data = np.nan_to_num(feature_data, nan=0.0)

            normalizer = ZScoreNormalizer()
            feature_data = normalizer.fit_transform(feature_data)

            sequences = []
            for i in range(lookback, len(feature_data)):
                seq = feature_data[i - lookback:i]
                sequences.append(seq)

            if sequences:
                self.prepared_sequences = np.array(sequences)
                self.sequence_offset = lookback
                self._log(f"Sequenzen vorbereitet: {len(sequences)} ({len(feature_cols)} Features)")

        except Exception as e:
            import traceback
            self._log(f"Sequenz-Vorbereitung fehlgeschlagen: {e}", 'ERROR')
            self._log(traceback.format_exc(), 'ERROR')

    def _start_backtest(self):
        """Startet den Backtest."""
        self._debug(f"_start_backtest() aufgerufen")

        if self.data is None or self.is_running:
            return

        self.is_running = True
        self.is_paused = False
        self.control_panel.set_running_state(True)

        # Timer starten
        interval = int(1000 / self.steps_per_second)
        self.backtest_timer = QTimer()
        self.backtest_timer.timeout.connect(self._timer_callback)
        self.backtest_timer.start(interval)

        # Geschwindigkeitsmessung
        self._step_count = 0
        self._last_speed_update = time.perf_counter()
        self._speed_update_timer = QTimer()
        self._speed_update_timer.timeout.connect(self._update_actual_speed)
        self._speed_update_timer.start(500)

        self.control_panel.set_actual_speed(f"0 / {self.steps_per_second} Schritte/Sek")
        self.control_panel.add_tradelog("Backtest gestartet")

    def _stop_backtest(self):
        """Stoppt den Backtest."""
        self.is_running = False
        self.is_paused = True

        if self.backtest_timer:
            self.backtest_timer.stop()
            self.backtest_timer = None

        if self._speed_update_timer:
            self._speed_update_timer.stop()
            self._speed_update_timer = None

        self.control_panel.set_running_state(False)
        self.control_panel.set_actual_speed("Gestoppt")
        self.control_panel.add_tradelog("Backtest gestoppt")
        self._update_charts()

    def _reset_backtest(self):
        """Setzt den Backtest zurueck."""
        if self.backtest_timer:
            self.backtest_timer.stop()
            self.backtest_timer = None
        if self._speed_update_timer:
            self._speed_update_timer.stop()
            self._speed_update_timer = None

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

        self.control_panel.set_running_state(False)
        self.control_panel.reset_labels()
        self.stats_panel.reset_labels()
        self.chart_panel.initialize_charts()

        # Benchmark-Daten zuruecksetzen
        self._benchmark_reset()

    def _single_step(self):
        """Fuehrt einen einzelnen Schritt aus."""
        if self.data is None or self.current_index > len(self.data):
            return

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
        self._step_count += 1
        self._update_ui()

        if not self.turbo_mode and self.current_index % 10 == 0:
            self._update_charts()

    def _process_step(self):
        """Verarbeitet einen einzelnen Backtest-Schritt."""
        t_total = self._benchmark_start()

        if self.data is None or self.current_index > len(self.data):
            return

        idx = self.current_index - 1
        current_price = self.data['Close'].iloc[idx]

        t_signal = self._benchmark_start()
        signal = self._get_signal(idx)
        self._benchmark_record('_get_signal', t_signal)

        self.signal_history.append({
            'index': self.current_index,
            'signal': signal,
            'price': current_price
        })

        if signal == 1:
            self.buy_count += 1
        elif signal == 2:
            self.sell_count += 1
        else:
            self.hold_count += 1

        t_proc = self._benchmark_start()
        self._process_signal(signal, current_price, self.current_index)
        self._benchmark_record('_process_signal', t_proc)

        t_equity = self._benchmark_start()
        self._update_equity(current_price)
        self._benchmark_record('_update_equity', t_equity)

        self.current_index += 1
        self._benchmark_record('_process_step', t_total)

    def _get_signal(self, idx: int) -> int:
        """Ermittelt das Signal fuer den aktuellen Index."""
        signal = 0

        if self.signals is not None and idx < len(self.signals):
            sig = self.signals.iloc[idx]
            if sig == 'BUY' or sig == 1:
                signal = 1
            elif sig == 'SELL' or sig == 2:
                signal = 2

        elif self.model is not None and self.prepared_sequences is not None:
            try:
                import torch
                seq_idx = idx - self.sequence_offset
                if 0 <= seq_idx < len(self.prepared_sequences):
                    sequence = self.prepared_sequences[seq_idx]
                    if not isinstance(sequence, torch.Tensor):
                        sequence = torch.FloatTensor(sequence)
                    sequence = sequence.unsqueeze(0)

                    self.model.eval()
                    with torch.no_grad():
                        prediction = self.model.predict(sequence)
                        raw_signal = int(prediction[0])

                        num_classes = self.model_info.get('num_classes', 3) if self.model_info else 3
                        if num_classes == 2:
                            signal = raw_signal + 1
                        else:
                            signal = raw_signal
            except Exception as e:
                if self.debug_mode:
                    self._debug(f"Idx {idx}: Modell-Fehler: {e}")

        if self.invert_signals and signal != 0:
            signal = 2 if signal == 1 else 1

        return signal

    def _process_signal(self, signal: int, price: float, idx: int):
        """Verarbeitet ein Trading-Signal."""
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
            self.control_panel.add_tradelog(f"{new_position} @ ${price:,.2f}")

    def _close_trade(self, price: float, idx: int, reason: str):
        """Schliesst die aktuelle Position."""
        if self.position == 'NONE':
            return

        if self.position == 'LONG':
            pnl = price - self.entry_price
        else:
            pnl = self.entry_price - price

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

        self.total_pnl += pnl
        self.current_equity = self.initial_capital + self.total_pnl

        pnl_str = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"
        self.control_panel.add_tradelog(f"#{len(self.trades)} {self.position}: {self.entry_price:,.2f} -> {price:,.2f} ({pnl_str})")

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
        t_total = self._benchmark_start()

        t = self._benchmark_start()
        self._update_datapoint_label()
        self._benchmark_record('_update_datapoint_label', t)

        t = self._benchmark_start()
        self._update_position_labels()
        self._benchmark_record('_update_position_labels', t)

        t = self._benchmark_start()
        self._update_pnl_labels()
        self._benchmark_record('_update_pnl_labels', t)

        t = self._benchmark_start()
        self._update_trade_stats()
        self._benchmark_record('_update_trade_stats', t)

        t = self._benchmark_start()
        self._update_signal_counts()
        self._benchmark_record('_update_signal_counts', t)

        self._benchmark_record('_update_ui', t_total)

    def _update_datapoint_label(self):
        """Aktualisiert die Fortschritts-Anzeige."""
        total = len(self.data) if self.data is not None else 0
        date_str = "-"
        signal = 0

        if self.data is not None and self.current_index <= len(self.data):
            idx = self.current_index - 1
            if 'DateTime' in self.data.columns or self.data.index.name == 'DateTime':
                dt = self.data.index[idx] if self.data.index.name else self.data['DateTime'].iloc[idx]
                if hasattr(dt, 'strftime'):
                    date_str = dt.strftime('%d.%m.%Y %H:%M')

            current_price = self.data['Close'].iloc[idx]
            self.control_panel.current_price_label.setText(f"${current_price:,.2f}")

        if self.signal_history:
            signal = self.signal_history[-1]['signal']

        self.control_panel.update_progress(self.current_index, total, date_str, signal)

    def _update_position_labels(self):
        """Aktualisiert die Positions-Anzeige."""
        current_price = 0.0
        unrealized = 0.0

        if self.data is not None and self.current_index <= len(self.data):
            current_price = self.data['Close'].iloc[self.current_index - 1]
            if self.position == 'LONG':
                unrealized = current_price - self.entry_price
            elif self.position == 'SHORT':
                unrealized = self.entry_price - current_price

        self.control_panel.update_position(self.position, self.entry_price, current_price, unrealized)

    def _update_pnl_labels(self):
        """Aktualisiert die P/L-Anzeige."""
        drawdown_pct = 0.0
        if len(self.equity_curve) > 1:
            peak = max(self.equity_curve)
            current = self.equity_curve[-1]
            drawdown_pct = ((peak - current) / peak) * 100 if peak > 0 else 0

        self.stats_panel.update_pnl(self.current_equity, self.total_pnl, drawdown_pct)

    def _update_trade_stats(self):
        """Aktualisiert die Trade-Statistiken."""
        num_trades = len(self.trades)
        winners = 0
        losers = 0
        avg_win = 0.0
        avg_loss = 0.0
        winrate = 0.0

        if num_trades > 0:
            pnls = [t['pnl'] for t in self.trades]
            winners = sum(1 for p in pnls if p > 0)
            losers = sum(1 for p in pnls if p <= 0)
            winrate = (winners / num_trades) * 100

            winning_pnls = [p for p in pnls if p > 0]
            losing_pnls = [p for p in pnls if p <= 0]

            if winning_pnls:
                avg_win = np.mean(winning_pnls)
            if losing_pnls:
                avg_loss = np.mean(losing_pnls)

        self.stats_panel.update_trade_stats(num_trades, winners, losers, winrate, avg_win, avg_loss)

    def _update_signal_counts(self):
        """Aktualisiert die Signal-Zaehler."""
        self.stats_panel.update_signal_counts(self.buy_count, self.sell_count, self.hold_count)

    def _update_charts(self):
        """Aktualisiert die Charts."""
        t = self._benchmark_start()
        self.chart_panel.update_charts(
            current_index=self.current_index,
            signal_history=self.signal_history,
            trades=self.trades,
            equity_curve=self.equity_curve,
            equity_indices=self.equity_indices,
            buy_count=self.buy_count,
            sell_count=self.sell_count,
            hold_count=self.hold_count,
            total_pnl=self.total_pnl,
            current_equity=self.current_equity,
            position=self.position,
            entry_price=self.entry_price,
            entry_index=self.entry_index
        )
        self._benchmark_record('_update_charts', t)

    def _finalize_backtest(self):
        """Finalisiert den Backtest."""
        self._debug(f"_finalize_backtest() aufgerufen")
        self._stop_backtest()

        if self.position != 'NONE' and self.data is not None:
            final_price = self.data['Close'].iloc[-1]
            self._close_trade(final_price, len(self.data), 'Backtest Ende')

        self._update_charts()
        self.control_panel.add_tradelog("=== Backtest abgeschlossen ===")

        # Benchmark-Report ausgeben
        self._benchmark_report()
        self.control_panel.add_tradelog(f"Gesamt P/L: ${self.total_pnl:,.2f}")
        self.control_panel.add_tradelog(f"Trades: {len(self.trades)}")

    def _update_speed(self, value: int):
        """Aktualisiert die eingestellte Geschwindigkeit."""
        self.steps_per_second = value
        if self.is_running and self.backtest_timer:
            self.backtest_timer.setInterval(int(1000 / value))

    def _update_actual_speed(self):
        """Berechnet und zeigt die tatsaechliche Geschwindigkeit an."""
        now = time.perf_counter()
        elapsed = now - self._last_speed_update

        if elapsed > 0:
            actual_speed = self._step_count / elapsed
            self.control_panel.set_actual_speed(
                f"{actual_speed:.1f} / {self.steps_per_second} Schritte/Sek"
            )

        self._step_count = 0
        self._last_speed_update = now

    def _toggle_turbo(self, checked: bool):
        """Schaltet den Turbo-Modus um."""
        self.turbo_mode = checked
        if not checked:
            self._update_charts()

    def _toggle_debug(self, checked: bool):
        """Schaltet den DEBUG-Modus um."""
        self.debug_mode = checked
        if checked:
            self._log("DEBUG-Modus aktiviert", 'INFO')
        else:
            self._log("DEBUG-Modus deaktiviert", 'INFO')

    def _toggle_invert(self, checked: bool):
        """Schaltet die Signal-Invertierung um."""
        self.invert_signals = checked
        if checked:
            self._log("Signal-Invertierung aktiviert (BUY<->SELL)", 'INFO')
        else:
            self._log("Signal-Invertierung deaktiviert", 'INFO')

    def _show_trade_statistics(self):
        """Zeigt den Trade-Statistik Dialog."""
        dialog = TradeStatisticsDialog(
            self, self.trades, self.data, self.equity_curve,
            self.initial_capital, self.current_equity
        )
        dialog.exec()

    def closeEvent(self, event):
        """Behandelt das Schliessen des Fensters."""
        if self.backtest_timer:
            self.backtest_timer.stop()
        if self._speed_update_timer:
            self._speed_update_timer.stop()
        event.accept()
