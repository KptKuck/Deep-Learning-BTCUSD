"""
Backtest GUI Module - Modulare Komponenten fuer das Backtest-Fenster.

Struktur:
- BacktestWindow: Hauptfenster (orchestriert alle Komponenten)
- ControlPanel: Steuerung (Start/Stop, Geschwindigkeit, Position)
- ChartPanel: Charts (Preis, Trades, Equity)
- StatsPanel: Performance-Statistiken
- TradeStatisticsDialog: Detaillierte Trade-Analyse
"""

from .backtest_window import BacktestWindow
from .control_panel import ControlPanel
from .chart_panel import ChartPanel
from .stats_panel import StatsPanel
from .trade_statistics_dialog import TradeStatisticsDialog

__all__ = [
    'BacktestWindow',
    'ControlPanel',
    'ChartPanel',
    'StatsPanel',
    'TradeStatisticsDialog',
]
