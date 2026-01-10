"""
Internal Backtester - Eigene Backtest-Engine
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from ..core.logger import get_logger
from .base import BacktesterInterface, BacktestResult, Trade


class InternalBacktester(BacktesterInterface):
    """
    Interne Backtest-Engine.

    Eine einfache aber effektive Backtest-Implementierung die als
    Referenz dient und keine externen Dependencies benoetigt.

    Features:
    - Long/Short Positionen
    - Kommission und Slippage
    - Equity-Kurve Tracking
    - Stop-Loss und Take-Profit (optional)
    """

    def __init__(self):
        self.logger = get_logger()

        # Standard-Parameter
        self.commission = 0.001  # 0.1%
        self.slippage = 0.0005  # 0.05%
        self.position_size = 1.0  # 100% des Kapitals
        self.stop_loss: Optional[float] = None
        self.take_profit: Optional[float] = None

    @property
    def name(self) -> str:
        return "Internal"

    def set_params(self, **kwargs) -> None:
        """Setzt Backtest-Parameter."""
        if 'commission' in kwargs:
            self.commission = kwargs['commission']
        if 'slippage' in kwargs:
            self.slippage = kwargs['slippage']
        if 'position_size' in kwargs:
            self.position_size = kwargs['position_size']
        if 'stop_loss' in kwargs:
            self.stop_loss = kwargs['stop_loss']
        if 'take_profit' in kwargs:
            self.take_profit = kwargs['take_profit']

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        initial_capital: float = 10000.0
    ) -> BacktestResult:
        """
        Fuehrt Backtest durch.

        Args:
            data: DataFrame mit OHLCV-Daten (muss 'Close', 'DateTime' enthalten)
            signals: Series mit Signalen ('BUY', 'SELL', 'HOLD' oder 0, 1, 2)
            initial_capital: Startkapital

        Returns:
            BacktestResult mit allen Trades und Metriken
        """
        self.logger.info('Starte Backtest...')

        # Signale konvertieren
        if signals.dtype in ['int64', 'int32', 'float64']:
            signal_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            signals = signals.map(signal_map)

        # State
        trades: List[Trade] = []
        equity = [initial_capital]
        cash = initial_capital
        position = None  # {'type': 'LONG'/'SHORT', 'entry_price': float, 'entry_time': timestamp, 'size': float}

        close_prices = data['Close'].values
        timestamps = data['DateTime'].values if 'DateTime' in data.columns else data.index

        for i in range(len(data)):
            signal = signals.iloc[i] if i < len(signals) else 'HOLD'
            price = close_prices[i]
            timestamp = timestamps[i]

            # Aktuelle Position pruefen auf Stop-Loss/Take-Profit
            if position is not None:
                should_close, close_reason = self._check_exit_conditions(position, price)
                if should_close:
                    trade = self._close_position(position, price, timestamp, close_reason)
                    trades.append(trade)
                    cash += trade.pnl + (position['size'] * position['entry_price'])
                    position = None

            # Signal verarbeiten
            if signal == 'BUY' and position is None:
                # Long Position eroeffnen
                entry_price = price * (1 + self.slippage)
                size = (cash * self.position_size) / entry_price
                position = {
                    'type': 'LONG',
                    'entry_price': entry_price,
                    'entry_time': timestamp,
                    'size': size
                }
                cash -= size * entry_price * (1 + self.commission)

            elif signal == 'SELL' and position is not None and position['type'] == 'LONG':
                # Long Position schliessen
                trade = self._close_position(position, price, timestamp, 'SIGNAL')
                trades.append(trade)
                cash += trade.pnl + (position['size'] * position['entry_price'])
                position = None

            # Equity aktualisieren
            current_equity = cash
            if position is not None:
                if position['type'] == 'LONG':
                    current_equity += position['size'] * price
            equity.append(current_equity)

        # Offene Position am Ende schliessen
        if position is not None:
            trade = self._close_position(position, close_prices[-1], timestamps[-1], 'END')
            trades.append(trade)

        # Ergebnis erstellen
        equity_series = pd.Series(equity[1:], index=timestamps)

        result = BacktestResult(
            trades=trades,
            equity_curve=equity_series,
            initial_capital=initial_capital
        )

        self.logger.success(f'Backtest abgeschlossen: {len(trades)} Trades, '
                           f'Return: {result.total_return_pct:.2f}%')

        return result

    def _close_position(
        self,
        position: dict,
        price: float,
        timestamp,
        reason: str
    ) -> Trade:
        """Schliesst eine Position und erstellt Trade-Objekt."""
        exit_price = price * (1 - self.slippage)
        commission = position['size'] * exit_price * self.commission

        return Trade(
            entry_time=pd.Timestamp(position['entry_time']),
            exit_time=pd.Timestamp(timestamp),
            entry_price=position['entry_price'],
            exit_price=exit_price,
            position=position['type'],
            size=position['size'],
            signal=reason,
            commission=commission
        )

    def _check_exit_conditions(self, position: dict, current_price: float) -> tuple:
        """
        Prueft Stop-Loss und Take-Profit Bedingungen.

        Returns:
            Tuple aus (should_close, reason)
        """
        if position['type'] == 'LONG':
            entry = position['entry_price']

            # Stop-Loss
            if self.stop_loss is not None:
                stop_price = entry * (1 - self.stop_loss)
                if current_price <= stop_price:
                    return True, 'STOP_LOSS'

            # Take-Profit
            if self.take_profit is not None:
                tp_price = entry * (1 + self.take_profit)
                if current_price >= tp_price:
                    return True, 'TAKE_PROFIT'

        return False, ''

    def run_walk_forward(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        initial_capital: float = 10000.0,
        train_size: float = 0.7,
        test_size: float = 0.3,
        n_splits: int = 5
    ) -> List[BacktestResult]:
        """
        Walk-Forward Analyse mit mehreren Zeitperioden.

        Args:
            data: Kompletter DataFrame
            signals: Komplette Signal-Serie
            initial_capital: Startkapital
            train_size: Anteil Trainingsdaten (nicht verwendet im Backtest)
            test_size: Anteil Testdaten
            n_splits: Anzahl der Splits

        Returns:
            Liste von BacktestResult fuer jeden Split
        """
        results = []
        n = len(data)
        split_size = n // n_splits

        for i in range(n_splits):
            start_idx = i * split_size
            end_idx = min((i + 1) * split_size, n)

            split_data = data.iloc[start_idx:end_idx].reset_index(drop=True)
            split_signals = signals.iloc[start_idx:end_idx].reset_index(drop=True)

            result = self.run(split_data, split_signals, initial_capital)
            results.append(result)

            self.logger.debug(f'Split {i + 1}/{n_splits}: '
                             f'{result.num_trades} Trades, '
                             f'Return: {result.total_return_pct:.2f}%')

        return results
