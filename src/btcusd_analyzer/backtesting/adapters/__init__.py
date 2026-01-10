"""Backtesting Adapters - Externe Framework-Anbindungen"""

from .factory import BacktesterFactory

# Optionale Adapter (nur importieren wenn verfuegbar)
try:
    from .vectorbt import VectorBTAdapter
except ImportError:
    VectorBTAdapter = None

try:
    from .backtrader import BacktraderAdapter
except ImportError:
    BacktraderAdapter = None

try:
    from .backtestingpy import BacktestingPyAdapter
except ImportError:
    BacktestingPyAdapter = None

__all__ = [
    'BacktesterFactory',
    'VectorBTAdapter',
    'BacktraderAdapter',
    'BacktestingPyAdapter',
]
