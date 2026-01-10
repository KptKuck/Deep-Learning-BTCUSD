"""
Backtesting Base Modul - Abstrakte Schnittstelle und Datenstrukturen
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import pandas as pd


@dataclass
class Trade:
    """
    Repraesentiert einen einzelnen Trade.

    Attributes:
        entry_time: Zeitpunkt des Einstiegs
        exit_time: Zeitpunkt des Ausstiegs
        entry_price: Einstiegspreis
        exit_price: Ausstiegspreis
        position: 'LONG' oder 'SHORT'
        size: Positionsgroesse
        pnl: Profit/Loss in absoluten Werten
        pnl_pct: Profit/Loss in Prozent
        signal: Urspruengliches Signal ('BUY', 'SELL')
    """
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    position: str  # 'LONG' oder 'SHORT'
    size: float = 1.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    signal: str = ''
    commission: float = 0.0

    def __post_init__(self):
        """Berechnet P/L wenn nicht gesetzt."""
        if self.pnl == 0.0 and self.entry_price > 0:
            if self.position == 'LONG':
                self.pnl = (self.exit_price - self.entry_price) * self.size - self.commission
                self.pnl_pct = ((self.exit_price / self.entry_price) - 1) * 100
            else:  # SHORT
                self.pnl = (self.entry_price - self.exit_price) * self.size - self.commission
                self.pnl_pct = ((self.entry_price / self.exit_price) - 1) * 100

    @property
    def is_winner(self) -> bool:
        """True wenn Trade profitabel war."""
        return self.pnl > 0

    @property
    def duration(self) -> pd.Timedelta:
        """Dauer des Trades."""
        return self.exit_time - self.entry_time


@dataclass
class BacktestResult:
    """
    Ergebnis eines Backtests.

    Enthaelt alle Trades, Equity-Kurve und Performance-Metriken.
    """
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    initial_capital: float = 10000.0

    # Berechnete Metriken
    total_pnl: float = 0.0
    total_return_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    num_trades: int = 0
    avg_trade_pnl: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    largest_winner: float = 0.0
    largest_loser: float = 0.0
    avg_trade_duration: Optional[pd.Timedelta] = None

    def __post_init__(self):
        """Berechnet Metriken aus Trades."""
        if self.trades:
            self.calculate_metrics()

    def calculate_metrics(self):
        """Berechnet alle Performance-Metriken."""
        if not self.trades:
            return

        self.num_trades = len(self.trades)
        pnls = [t.pnl for t in self.trades]

        # Basis-Metriken
        self.total_pnl = sum(pnls)
        self.total_return_pct = (self.total_pnl / self.initial_capital) * 100

        # Win Rate
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]
        self.win_rate = (len(winners) / self.num_trades) * 100 if self.num_trades > 0 else 0

        # Profit Factor
        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 1
        self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Durchschnittliche Trades
        self.avg_trade_pnl = self.total_pnl / self.num_trades if self.num_trades > 0 else 0
        self.avg_winner = sum(winners) / len(winners) if winners else 0
        self.avg_loser = sum(losers) / len(losers) if losers else 0
        self.largest_winner = max(winners) if winners else 0
        self.largest_loser = min(losers) if losers else 0

        # Trade-Dauer
        durations = [t.duration for t in self.trades]
        self.avg_trade_duration = sum(durations, pd.Timedelta(0)) / len(durations)

        # Drawdown
        if len(self.equity_curve) > 0:
            self._calculate_drawdown()
            self._calculate_sharpe_sortino()

    def _calculate_drawdown(self):
        """Berechnet Maximum Drawdown."""
        equity = self.equity_curve
        peak = equity.expanding().max()
        drawdown = equity - peak
        self.max_drawdown = abs(drawdown.min())
        self.max_drawdown_pct = (self.max_drawdown / peak.max()) * 100 if peak.max() > 0 else 0

    def _calculate_sharpe_sortino(self):
        """Berechnet Sharpe und Sortino Ratio."""
        if len(self.equity_curve) < 2:
            return

        # Taegliche Returns
        returns = self.equity_curve.pct_change().dropna()

        if len(returns) == 0 or returns.std() == 0:
            return

        # Sharpe Ratio (annualisiert, angenommen taegliche Daten)
        risk_free_rate = 0.0  # Vereinfachung
        excess_returns = returns - risk_free_rate / 252
        self.sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * (252 ** 0.5)

        # Sortino Ratio (nur negative Returns)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0 and negative_returns.std() > 0:
            self.sortino_ratio = (returns.mean() / negative_returns.std()) * (252 ** 0.5)

    def to_dict(self) -> dict:
        """Konvertiert zu Dictionary fuer JSON-Export."""
        return {
            'num_trades': self.num_trades,
            'total_pnl': round(self.total_pnl, 2),
            'total_return_pct': round(self.total_return_pct, 2),
            'win_rate': round(self.win_rate, 2),
            'profit_factor': round(self.profit_factor, 2),
            'max_drawdown': round(self.max_drawdown, 2),
            'max_drawdown_pct': round(self.max_drawdown_pct, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 2),
            'sortino_ratio': round(self.sortino_ratio, 2),
            'avg_trade_pnl': round(self.avg_trade_pnl, 2),
            'avg_winner': round(self.avg_winner, 2),
            'avg_loser': round(self.avg_loser, 2),
            'largest_winner': round(self.largest_winner, 2),
            'largest_loser': round(self.largest_loser, 2),
        }

    def summary(self) -> str:
        """Gibt eine Zusammenfassung als String zurueck."""
        lines = [
            "=== Backtest Ergebnis ===",
            f"Trades: {self.num_trades}",
            f"Total P/L: ${self.total_pnl:,.2f} ({self.total_return_pct:+.2f}%)",
            f"Win Rate: {self.win_rate:.1f}%",
            f"Profit Factor: {self.profit_factor:.2f}",
            f"Max Drawdown: ${self.max_drawdown:,.2f} ({self.max_drawdown_pct:.1f}%)",
            f"Sharpe Ratio: {self.sharpe_ratio:.2f}",
            f"Avg Trade: ${self.avg_trade_pnl:,.2f}",
            f"Avg Winner: ${self.avg_winner:,.2f}",
            f"Avg Loser: ${self.avg_loser:,.2f}",
        ]
        return '\n'.join(lines)


class BacktesterInterface(ABC):
    """
    Abstrakte Schnittstelle fuer alle Backtester.

    Diese Klasse definiert die gemeinsame API fuer alle Backtester-Implementierungen,
    sei es die interne Engine oder Adapter fuer externe Frameworks.
    """

    @abstractmethod
    def run(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        initial_capital: float = 10000.0
    ) -> BacktestResult:
        """
        Fuehrt Backtest durch.

        Args:
            data: DataFrame mit OHLCV-Daten
            signals: Series mit Signalen ('BUY', 'SELL', 'HOLD')
            initial_capital: Startkapital

        Returns:
            BacktestResult mit allen Trades und Metriken
        """
        pass

    @abstractmethod
    def set_params(self, **kwargs) -> None:
        """
        Setzt Backtester-spezifische Parameter.

        Args:
            **kwargs: Parameter wie commission, slippage, position_size, etc.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name des Backtesters."""
        pass
