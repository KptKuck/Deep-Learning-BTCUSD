"""
VectorBT Adapter - Schnelles vektorisiertes Backtesting

VectorBT ist ein performantes Framework fuer vektorisiertes Backtesting.
Es nutzt NumPy und Numba fuer maximale Geschwindigkeit.

Installation: pip install vectorbt
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..base import BacktesterInterface, BacktestResult, Trade


class VectorBTAdapter(BacktesterInterface):
    """
    Adapter fuer VectorBT Framework.

    VectorBT ist besonders geeignet fuer:
    - Schnelles Backtesting grosser Datensaetze
    - Parameter-Optimierung (viele Durchlaeufe)
    - Portfolio-Analysen

    Features:
    - Vektorisierte Berechnungen
    - Automatische Metriken
    - Integrierte Visualisierung

    Usage:
        adapter = VectorBTAdapter()
        adapter.set_params(fees=0.001, slippage=0.0005)
        result = adapter.run(data, signals)
    """

    def __init__(self):
        """Initialisiert den VectorBT Adapter."""
        self._vbt = None
        self._params: Dict[str, Any] = {
            'fees': 0.001,          # 0.1% Gebuehren
            'slippage': 0.0,        # Slippage
            'freq': '1h',           # Daten-Frequenz
            'init_cash': 10000.0,   # Startkapital
            'size_type': 'amount',  # 'amount' oder 'percent'
            'size': 1.0,            # Positionsgroesse
        }
        self._load_vectorbt()

    def _load_vectorbt(self):
        """Laedt VectorBT mit Fehlerbehandlung."""
        try:
            import vectorbt as vbt
            self._vbt = vbt
        except ImportError:
            raise ImportError(
                "VectorBT nicht installiert. "
                "Installation: pip install vectorbt"
            )

    @property
    def name(self) -> str:
        return "VectorBT"

    def set_params(self, **kwargs) -> None:
        """
        Setzt VectorBT-spezifische Parameter.

        Args:
            fees: Transaktionsgebuehren (Dezimal, z.B. 0.001 = 0.1%)
            slippage: Slippage (Dezimal)
            freq: Daten-Frequenz ('1h', '1d', etc.)
            init_cash: Startkapital
            size_type: 'amount' oder 'percent'
            size: Positionsgroesse
        """
        self._params.update(kwargs)

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        initial_capital: float = 10000.0
    ) -> BacktestResult:
        """
        Fuehrt Backtest mit VectorBT durch.

        Args:
            data: DataFrame mit OHLCV-Daten (muss 'Close' enthalten)
            signals: Series mit Signalen ('BUY', 'SELL', 'HOLD')
            initial_capital: Startkapital

        Returns:
            BacktestResult mit Trades und Metriken
        """
        vbt = self._vbt

        # Daten vorbereiten
        close = data['Close'] if 'Close' in data.columns else data['close']

        # Signale zu Entry/Exit konvertieren
        entries = (signals == 'BUY').values
        exits = (signals == 'SELL').values

        # Portfolio erstellen
        portfolio = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            init_cash=initial_capital,
            fees=self._params['fees'],
            slippage=self._params['slippage'],
            freq=self._params.get('freq', '1h'),
            size=self._params.get('size', 1.0),
            size_type=self._params.get('size_type', 'amount'),
        )

        # Ergebnis konvertieren
        return self._convert_result(portfolio, data, initial_capital)

    def _convert_result(
        self,
        portfolio,
        data: pd.DataFrame,
        initial_capital: float
    ) -> BacktestResult:
        """Konvertiert VectorBT-Ergebnis zu BacktestResult."""
        vbt = self._vbt

        # Trades extrahieren
        trades_df = portfolio.trades.records_readable
        trades = []

        if len(trades_df) > 0:
            for _, row in trades_df.iterrows():
                # VectorBT Trade-Spalten
                entry_idx = int(row.get('Entry Index', row.get('entry_idx', 0)))
                exit_idx = int(row.get('Exit Index', row.get('exit_idx', 0)))

                # Zeitstempel aus Index
                entry_time = data.index[entry_idx] if entry_idx < len(data) else data.index[-1]
                exit_time = data.index[exit_idx] if exit_idx < len(data) else data.index[-1]

                # Preise
                entry_price = row.get('Entry Price', row.get('entry_price', 0))
                exit_price = row.get('Exit Price', row.get('exit_price', 0))

                # PnL
                pnl = row.get('PnL', row.get('pnl', 0))
                pnl_pct = row.get('Return', row.get('return', 0)) * 100

                # Position (VectorBT: direction)
                direction = row.get('Direction', row.get('direction', 'Long'))
                position = 'LONG' if 'long' in str(direction).lower() else 'SHORT'

                trade = Trade(
                    entry_time=pd.Timestamp(entry_time),
                    exit_time=pd.Timestamp(exit_time),
                    entry_price=float(entry_price),
                    exit_price=float(exit_price),
                    position=position,
                    size=float(row.get('Size', row.get('size', 1.0))),
                    pnl=float(pnl),
                    pnl_pct=float(pnl_pct),
                    signal='BUY' if position == 'LONG' else 'SELL'
                )
                trades.append(trade)

        # Equity Curve
        equity_curve = portfolio.value()
        if isinstance(equity_curve, pd.DataFrame):
            equity_curve = equity_curve.iloc[:, 0]

        # BacktestResult erstellen
        result = BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            initial_capital=initial_capital
        )

        # Zusaetzliche VectorBT-Metriken uebernehmen
        try:
            stats = portfolio.stats()
            result.sharpe_ratio = float(stats.get('Sharpe Ratio', 0) or 0)
            result.sortino_ratio = float(stats.get('Sortino Ratio', 0) or 0)
            result.max_drawdown_pct = float(stats.get('Max Drawdown [%]', 0) or 0)
        except Exception:
            pass

        return result

    def get_portfolio(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        initial_capital: float = 10000.0
    ):
        """
        Gibt das VectorBT Portfolio-Objekt zurueck.

        Nuetzlich fuer erweiterte Analysen und Visualisierungen.

        Args:
            data: DataFrame mit OHLCV-Daten
            signals: Series mit Signalen
            initial_capital: Startkapital

        Returns:
            VectorBT Portfolio-Objekt
        """
        vbt = self._vbt
        close = data['Close'] if 'Close' in data.columns else data['close']

        entries = (signals == 'BUY').values
        exits = (signals == 'SELL').values

        portfolio = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            init_cash=initial_capital,
            fees=self._params['fees'],
            slippage=self._params['slippage'],
        )

        return portfolio

    def optimize_params(
        self,
        data: pd.DataFrame,
        signals_func,
        param_grid: Dict[str, list],
        metric: str = 'sharpe_ratio'
    ) -> Dict[str, Any]:
        """
        Optimiert Parameter mit VectorBT's eingebauter Optimierung.

        Args:
            data: DataFrame mit OHLCV-Daten
            signals_func: Funktion die Signale generiert: signals_func(data, **params) -> Series
            param_grid: Dictionary mit Parameter-Listen
            metric: Metrik zur Optimierung ('sharpe_ratio', 'total_return', etc.)

        Returns:
            Beste Parameter
        """
        vbt = self._vbt

        # Diese Methode erfordert fortgeschrittene VectorBT-Nutzung
        # Hier nur als Platzhalter/Dokumentation

        # Hinweis: VectorBT's eingebaute Optimierung ist fuer Signal-Parameter
        # gedacht (z.B. MA-Perioden), nicht fuer Modell-Hyperparameter.
        # Fuer Modell-Optimierung wird OptunaTuner empfohlen.
        raise NotImplementedError(
            "VectorBT-Parameter-Optimierung ist nicht implementiert. "
            "Fuer Hyperparameter-Suche verwenden Sie OptunaTuner aus "
            "btcusd_analyzer.optimization.optuna_tuner"
        )
