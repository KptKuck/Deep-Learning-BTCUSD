"""
Backtrader Adapter - Feature-reiches Backtesting-Framework

Backtrader ist ein ausgereiftes Framework mit vielen Features:
- Komplexe Strategien
- Multiple Data Feeds
- Broker-Simulation
- Live-Trading Anbindung

Installation: pip install backtrader
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..base import BacktesterInterface, BacktestResult, Trade


class BacktraderAdapter(BacktesterInterface):
    """
    Adapter fuer Backtrader Framework.

    Backtrader ist besonders geeignet fuer:
    - Komplexe Strategien mit mehreren Indikatoren
    - Event-basiertes Backtesting
    - Realistische Broker-Simulation

    Features:
    - Order-Typen (Market, Limit, Stop, etc.)
    - Position Sizing
    - Commission-Modelle
    - Sizer-Klassen

    Usage:
        adapter = BacktraderAdapter()
        adapter.set_params(commission=0.001, stake=1)
        result = adapter.run(data, signals)
    """

    def __init__(self):
        """Initialisiert den Backtrader Adapter."""
        self._bt = None
        self._params: Dict[str, Any] = {
            'commission': 0.001,    # 0.1% Gebuehren
            'stake': 1,             # Einheiten pro Trade
            'cash': 10000.0,        # Startkapital
            'slippage_perc': 0.0,   # Slippage in Prozent
            'slippage_fixed': 0.0,  # Fixer Slippage
        }
        self._trades: List[Dict] = []
        self._load_backtrader()

    def _load_backtrader(self):
        """Laedt Backtrader mit Fehlerbehandlung."""
        try:
            import backtrader as bt
            self._bt = bt
        except ImportError:
            raise ImportError(
                "Backtrader nicht installiert. "
                "Installation: pip install backtrader"
            )

    @property
    def name(self) -> str:
        return "Backtrader"

    def set_params(self, **kwargs) -> None:
        """
        Setzt Backtrader-spezifische Parameter.

        Args:
            commission: Transaktionsgebuehren (Dezimal)
            stake: Einheiten pro Trade
            cash: Startkapital
            slippage_perc: Slippage in Prozent
            slippage_fixed: Fixer Slippage
        """
        self._params.update(kwargs)

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        initial_capital: float = 10000.0
    ) -> BacktestResult:
        """
        Fuehrt Backtest mit Backtrader durch.

        Args:
            data: DataFrame mit OHLCV-Daten
            signals: Series mit Signalen ('BUY', 'SELL', 'HOLD')
            initial_capital: Startkapital

        Returns:
            BacktestResult mit Trades und Metriken
        """
        bt = self._bt

        # Strategie-Klasse dynamisch erstellen
        SignalStrategy = self._create_strategy_class(signals)

        # Cerebro Engine
        cerebro = bt.Cerebro()

        # Startkapital
        cerebro.broker.setcash(initial_capital)

        # Commission
        cerebro.broker.setcommission(commission=self._params['commission'])

        # Slippage
        if self._params['slippage_perc'] > 0:
            cerebro.broker.set_slippage_perc(self._params['slippage_perc'])
        if self._params['slippage_fixed'] > 0:
            cerebro.broker.set_slippage_fixed(self._params['slippage_fixed'])

        # Daten hinzufuegen
        bt_data = self._convert_data(data)
        cerebro.adddata(bt_data)

        # Strategie hinzufuegen
        cerebro.addstrategy(SignalStrategy, stake=self._params['stake'])

        # Analyzer hinzufuegen
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

        # Backtest starten
        self._trades = []
        results = cerebro.run()
        strategy = results[0]

        # Ergebnis extrahieren
        return self._extract_result(
            cerebro, strategy, data, initial_capital
        )

    def _create_strategy_class(self, signals: pd.Series):
        """Erstellt eine Backtrader-Strategie basierend auf Signalen."""
        bt = self._bt
        adapter = self

        class SignalStrategy(bt.Strategy):
            params = (('stake', 1),)

            def __init__(self):
                self.signals = signals.values
                self.signal_index = 0
                self.order = None

            def notify_order(self, order):
                if order.status in [order.Completed]:
                    self.order = None

            def notify_trade(self, trade):
                if trade.isclosed:
                    adapter._trades.append({
                        'entry_time': bt.num2date(trade.dtopen),
                        'exit_time': bt.num2date(trade.dtclose),
                        'entry_price': trade.price,
                        'exit_price': trade.price + trade.pnl / trade.size,
                        'size': abs(trade.size),
                        'pnl': trade.pnl,
                        'pnl_pct': trade.pnlcomm / (trade.price * abs(trade.size)) * 100,
                        'commission': trade.commission,
                    })

            def next(self):
                if self.order:
                    return

                idx = len(self) - 1
                if idx >= len(self.signals):
                    return

                signal = self.signals[idx]

                if signal == 'BUY' and not self.position:
                    self.order = self.buy(size=self.params.stake)
                elif signal == 'SELL' and self.position:
                    self.order = self.close()

        return SignalStrategy

    def _convert_data(self, data: pd.DataFrame):
        """Konvertiert DataFrame zu Backtrader-Data."""
        bt = self._bt

        # Spaltennamen normalisieren
        df = data.copy()
        df.columns = [c.lower() for c in df.columns]

        # Backtrader erwartet spezifische Spalten
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                if col == 'volume':
                    df[col] = 0
                else:
                    raise ValueError(f"Spalte '{col}' fehlt in Daten")

        # Index zu datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Backtrader PandasData
        bt_data = bt.feeds.PandasData(
            dataname=df,
            datetime=None,  # Index verwenden
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=-1,
        )

        return bt_data

    def _extract_result(
        self,
        cerebro,
        strategy,
        data: pd.DataFrame,
        initial_capital: float
    ) -> BacktestResult:
        """Extrahiert Ergebnis aus Backtrader."""
        bt = self._bt

        # Trades konvertieren
        trades = []
        for t in self._trades:
            trade = Trade(
                entry_time=pd.Timestamp(t['entry_time']),
                exit_time=pd.Timestamp(t['exit_time']),
                entry_price=float(t['entry_price']),
                exit_price=float(t['exit_price']),
                position='LONG',  # Backtrader default
                size=float(t['size']),
                pnl=float(t['pnl']),
                pnl_pct=float(t['pnl_pct']),
                commission=float(t['commission']),
                signal='BUY'
            )
            trades.append(trade)

        # Equity Curve erstellen (vereinfacht)
        final_value = cerebro.broker.getvalue()
        equity_curve = pd.Series(
            [initial_capital, final_value],
            index=[data.index[0], data.index[-1]]
        )

        # BacktestResult
        result = BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            initial_capital=initial_capital
        )

        # Analyzer-Daten
        try:
            trade_analysis = strategy.analyzers.trades.get_analysis()
            sharpe = strategy.analyzers.sharpe.get_analysis()
            drawdown = strategy.analyzers.drawdown.get_analysis()

            if 'sharperatio' in sharpe:
                result.sharpe_ratio = float(sharpe['sharperatio'] or 0)

            if 'max' in drawdown:
                result.max_drawdown_pct = float(drawdown['max'].get('drawdown', 0) or 0)

        except Exception:
            pass

        return result

    def get_cerebro(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        initial_capital: float = 10000.0
    ):
        """
        Gibt das konfigurierte Backtrader Cerebro-Objekt zurueck.

        Nuetzlich fuer erweiterte Konfiguration oder Visualisierung.

        Args:
            data: DataFrame mit OHLCV-Daten
            signals: Series mit Signalen
            initial_capital: Startkapital

        Returns:
            Backtrader Cerebro-Objekt (nicht ausgefuehrt)
        """
        bt = self._bt

        SignalStrategy = self._create_strategy_class(signals)

        cerebro = bt.Cerebro()
        cerebro.broker.setcash(initial_capital)
        cerebro.broker.setcommission(commission=self._params['commission'])

        bt_data = self._convert_data(data)
        cerebro.adddata(bt_data)
        cerebro.addstrategy(SignalStrategy, stake=self._params['stake'])

        return cerebro

    def plot(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        initial_capital: float = 10000.0,
        **kwargs
    ):
        """
        Fuehrt Backtest aus und zeigt Backtrader-Plot.

        Args:
            data: DataFrame mit OHLCV-Daten
            signals: Series mit Signalen
            initial_capital: Startkapital
            **kwargs: Zusaetzliche Plot-Parameter
        """
        cerebro = self.get_cerebro(data, signals, initial_capital)
        cerebro.run()
        cerebro.plot(**kwargs)
