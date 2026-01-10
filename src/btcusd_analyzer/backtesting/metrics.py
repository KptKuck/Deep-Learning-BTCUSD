"""
Performance Metrics Modul - Berechnung von Trading-Metriken
"""

from typing import List, Optional

import numpy as np
import pandas as pd


class PerformanceMetrics:
    """
    Berechnet erweiterte Performance-Metriken fuer Backtests.

    Enthalt Methoden zur Berechnung von:
    - Risiko-adjustierten Returns (Sharpe, Sortino, Calmar)
    - Drawdown-Analysen
    - Trade-Statistiken
    - Benchmark-Vergleiche
    """

    @staticmethod
    def sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Berechnet den Sharpe Ratio.

        Args:
            returns: Serie von Returns
            risk_free_rate: Risikofreier Zinssatz (jaehrlich)
            periods_per_year: Anzahl Perioden pro Jahr

        Returns:
            Annualisierter Sharpe Ratio
        """
        if len(returns) < 2 or returns.std() == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / periods_per_year
        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)

    @staticmethod
    def sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Berechnet den Sortino Ratio (nur Downside-Risiko).

        Args:
            returns: Serie von Returns
            risk_free_rate: Risikofreier Zinssatz
            periods_per_year: Anzahl Perioden pro Jahr

        Returns:
            Annualisierter Sortino Ratio
        """
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - risk_free_rate / periods_per_year
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0

        return (excess_returns.mean() / downside_returns.std()) * np.sqrt(periods_per_year)

    @staticmethod
    def calmar_ratio(
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """
        Berechnet den Calmar Ratio (Return / Max Drawdown).

        Args:
            returns: Serie von Returns
            periods_per_year: Anzahl Perioden pro Jahr

        Returns:
            Calmar Ratio
        """
        if len(returns) < 2:
            return 0.0

        # Cumulative Returns
        cum_returns = (1 + returns).cumprod()
        max_dd = PerformanceMetrics.max_drawdown(cum_returns)

        if max_dd == 0:
            return float('inf') if returns.mean() > 0 else 0.0

        annual_return = returns.mean() * periods_per_year
        return annual_return / max_dd

    @staticmethod
    def max_drawdown(equity_curve: pd.Series) -> float:
        """
        Berechnet den Maximum Drawdown.

        Args:
            equity_curve: Equity-Kurve (kumuliert)

        Returns:
            Maximum Drawdown als Dezimalzahl
        """
        if len(equity_curve) < 2:
            return 0.0

        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        return abs(drawdown.min())

    @staticmethod
    def drawdown_series(equity_curve: pd.Series) -> pd.Series:
        """
        Berechnet die komplette Drawdown-Serie.

        Args:
            equity_curve: Equity-Kurve

        Returns:
            Serie mit Drawdown-Werten
        """
        peak = equity_curve.expanding().max()
        return (equity_curve - peak) / peak

    @staticmethod
    def max_drawdown_duration(equity_curve: pd.Series) -> int:
        """
        Berechnet die laengste Drawdown-Periode.

        Args:
            equity_curve: Equity-Kurve

        Returns:
            Anzahl Perioden im laengsten Drawdown
        """
        peak = equity_curve.expanding().max()
        in_drawdown = equity_curve < peak

        # Laengste Serie von True-Werten
        max_duration = 0
        current_duration = 0

        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_duration

    @staticmethod
    def win_rate(pnls: List[float]) -> float:
        """
        Berechnet die Win Rate.

        Args:
            pnls: Liste von Trade P/Ls

        Returns:
            Win Rate in Prozent
        """
        if not pnls:
            return 0.0

        winners = sum(1 for p in pnls if p > 0)
        return (winners / len(pnls)) * 100

    @staticmethod
    def profit_factor(pnls: List[float]) -> float:
        """
        Berechnet den Profit Factor (Gross Profit / Gross Loss).

        Args:
            pnls: Liste von Trade P/Ls

        Returns:
            Profit Factor
        """
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    @staticmethod
    def expectancy(pnls: List[float]) -> float:
        """
        Berechnet die Erwartungswert pro Trade.

        Args:
            pnls: Liste von Trade P/Ls

        Returns:
            Erwarteter P/L pro Trade
        """
        if not pnls:
            return 0.0

        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]

        win_rate = len(winners) / len(pnls)
        avg_win = np.mean(winners) if winners else 0
        avg_loss = abs(np.mean(losers)) if losers else 0

        return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    @staticmethod
    def risk_reward_ratio(pnls: List[float]) -> float:
        """
        Berechnet das Risk/Reward Verhaeltnis.

        Args:
            pnls: Liste von Trade P/Ls

        Returns:
            Risk/Reward Ratio
        """
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]

        if not winners or not losers:
            return 0.0

        avg_win = np.mean(winners)
        avg_loss = abs(np.mean(losers))

        return avg_win / avg_loss if avg_loss > 0 else float('inf')

    @staticmethod
    def recovery_factor(total_return: float, max_drawdown: float) -> float:
        """
        Berechnet den Recovery Factor.

        Args:
            total_return: Gesamtrendite
            max_drawdown: Maximum Drawdown

        Returns:
            Recovery Factor
        """
        if max_drawdown == 0:
            return float('inf') if total_return > 0 else 0.0
        return total_return / max_drawdown

    @staticmethod
    def consecutive_wins_losses(pnls: List[float]) -> tuple:
        """
        Berechnet maximale aufeinanderfolgende Gewinne/Verluste.

        Args:
            pnls: Liste von Trade P/Ls

        Returns:
            Tuple aus (max_consecutive_wins, max_consecutive_losses)
        """
        if not pnls:
            return 0, 0

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for pnl in pnls:
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif pnl < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:
                current_wins = 0
                current_losses = 0

        return max_wins, max_losses

    @staticmethod
    def benchmark_comparison(
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> dict:
        """
        Vergleicht Strategie mit Benchmark.

        Args:
            strategy_returns: Returns der Strategie
            benchmark_returns: Returns des Benchmarks

        Returns:
            Dictionary mit Vergleichsmetriken
        """
        # Alpha und Beta berechnen
        if len(strategy_returns) != len(benchmark_returns):
            min_len = min(len(strategy_returns), len(benchmark_returns))
            strategy_returns = strategy_returns[:min_len]
            benchmark_returns = benchmark_returns[:min_len]

        cov = np.cov(strategy_returns, benchmark_returns)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 0

        strategy_mean = strategy_returns.mean() * 252
        benchmark_mean = benchmark_returns.mean() * 252
        alpha = strategy_mean - beta * benchmark_mean

        # Tracking Error
        tracking_diff = strategy_returns - benchmark_returns
        tracking_error = tracking_diff.std() * np.sqrt(252)

        # Information Ratio
        info_ratio = (strategy_mean - benchmark_mean) / tracking_error if tracking_error > 0 else 0

        return {
            'alpha': alpha,
            'beta': beta,
            'tracking_error': tracking_error,
            'information_ratio': info_ratio,
            'strategy_return': strategy_mean,
            'benchmark_return': benchmark_mean,
            'outperformance': strategy_mean - benchmark_mean
        }
