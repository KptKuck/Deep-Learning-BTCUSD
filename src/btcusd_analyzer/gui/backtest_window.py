"""
Backtest Window - Re-Export aus dem modularen backtest/ Paket.

Die eigentliche Implementierung befindet sich in:
- gui/backtest/backtest_window.py   (Hauptfenster)
- gui/backtest/control_panel.py     (Steuerung)
- gui/backtest/chart_panel.py       (Charts)
- gui/backtest/stats_panel.py       (Statistiken)
- gui/backtest/trade_statistics_dialog.py (Analyse-Dialog)
"""

# Re-Export fuer Rueckwaertskompatibilitaet
from .backtest.backtest_window import BacktestWindow
from .backtest.trade_statistics_dialog import TradeStatisticsDialog

__all__ = ['BacktestWindow', 'TradeStatisticsDialog']
