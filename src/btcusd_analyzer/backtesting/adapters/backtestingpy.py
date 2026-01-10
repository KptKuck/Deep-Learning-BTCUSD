"""
Backtesting.py Adapter - Einfaches und leichtgewichtiges Backtesting

Backtesting.py ist ein minimalistisches Framework:
- Einfache API
- Schnelle Ausfuehrung
- Gute Visualisierung

Installation: pip install backtesting
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..base import BacktesterInterface, BacktestResult, Trade


class BacktestingPyAdapter(BacktesterInterface):
    """
    Adapter fuer Backtesting.py Framework.

    Backtesting.py ist besonders geeignet fuer:
    - Einfache Strategien
    - Schnelles Prototyping
    - Klare Visualisierungen

    Features:
    - Einfache API
    - Integrierte Visualisierung
    - Parameter-Optimierung

    Usage:
        adapter = BacktestingPyAdapter()
        adapter.set_params(commission=0.001, trade_on_close=True)
        result = adapter.run(data, signals)
    """

    def __init__(self):
        """Initialisiert den Backtesting.py Adapter."""
        self._bt = None
        self._Strategy = None
        self._params: Dict[str, Any] = {
            'commission': 0.001,       # 0.1% Gebuehren
            'margin': 1.0,             # Margin (1.0 = kein Hebel)
            'trade_on_close': True,    # Trade bei Close
            'hedging': False,          # Hedging erlauben
            'exclusive_orders': True,  # Nur eine Order gleichzeitig
            'cash': 10000.0,           # Startkapital
        }
        self._load_backtesting()

    def _load_backtesting(self):
        """Laedt Backtesting.py mit Fehlerbehandlung."""
        try:
            from backtesting import Backtest, Strategy
            self._bt = Backtest
            self._Strategy = Strategy
        except ImportError:
            raise ImportError(
                "Backtesting.py nicht installiert. "
                "Installation: pip install backtesting"
            )

    @property
    def name(self) -> str:
        return "Backtesting.py"

    def set_params(self, **kwargs) -> None:
        """
        Setzt Backtesting.py-spezifische Parameter.

        Args:
            commission: Transaktionsgebuehren (Dezimal)
            margin: Margin-Anforderung (1.0 = kein Hebel)
            trade_on_close: Trade bei Close-Preis
            hedging: Hedging erlauben
            exclusive_orders: Nur eine Order gleichzeitig
            cash: Startkapital
        """
        self._params.update(kwargs)

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        initial_capital: float = 10000.0
    ) -> BacktestResult:
        """
        Fuehrt Backtest mit Backtesting.py durch.

        Args:
            data: DataFrame mit OHLCV-Daten
            signals: Series mit Signalen ('BUY', 'SELL', 'HOLD')
            initial_capital: Startkapital

        Returns:
            BacktestResult mit Trades und Metriken
        """
        # Daten vorbereiten
        df = self._prepare_data(data, signals)

        # Strategie erstellen
        SignalStrategy = self._create_strategy(df)

        # Backtest konfigurieren
        bt = self._bt(
            df,
            SignalStrategy,
            cash=initial_capital,
            commission=self._params['commission'],
            margin=self._params['margin'],
            trade_on_close=self._params['trade_on_close'],
            hedging=self._params['hedging'],
            exclusive_orders=self._params['exclusive_orders'],
        )

        # Ausfuehren
        stats = bt.run()

        # Ergebnis konvertieren
        return self._convert_result(stats, bt, data, initial_capital)

    def _prepare_data(self, data: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
        """Bereitet Daten fuer Backtesting.py vor."""
        df = data.copy()

        # Spaltennamen kapitalisieren (Backtesting.py Anforderung)
        df.columns = [c.capitalize() for c in df.columns]

        # Sicherstellen dass alle benoetigten Spalten existieren
        required = ['Open', 'High', 'Low', 'Close']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Spalte '{col}' fehlt in Daten")

        if 'Volume' not in df.columns:
            df['Volume'] = 0

        # Signale hinzufuegen
        df['Signal'] = signals.values

        return df

    def _create_strategy(self, data: pd.DataFrame):
        """Erstellt eine Backtesting.py-Strategie basierend auf Signalen."""
        Strategy = self._Strategy

        class SignalStrategy(Strategy):
            def init(self):
                # Signale aus Daten holen
                self.signals = self.data.Signal

            def next(self):
                signal = self.signals[-1]

                if signal == 'BUY' and not self.position:
                    self.buy()
                elif signal == 'SELL' and self.position:
                    self.position.close()

        return SignalStrategy

    def _convert_result(
        self,
        stats,
        bt,
        data: pd.DataFrame,
        initial_capital: float
    ) -> BacktestResult:
        """Konvertiert Backtesting.py-Ergebnis zu BacktestResult."""
        # Trades extrahieren
        trades = []
        trades_df = stats._trades if hasattr(stats, '_trades') else pd.DataFrame()

        if len(trades_df) > 0:
            for _, row in trades_df.iterrows():
                # Backtesting.py Trade-Spalten
                entry_time = row.get('EntryTime', data.index[0])
                exit_time = row.get('ExitTime', data.index[-1])
                entry_price = row.get('EntryPrice', 0)
                exit_price = row.get('ExitPrice', 0)
                pnl = row.get('PnL', 0)
                size = row.get('Size', 1)

                # Position aus Size bestimmen
                position = 'LONG' if size > 0 else 'SHORT'

                trade = Trade(
                    entry_time=pd.Timestamp(entry_time),
                    exit_time=pd.Timestamp(exit_time),
                    entry_price=float(entry_price),
                    exit_price=float(exit_price),
                    position=position,
                    size=abs(float(size)),
                    pnl=float(pnl),
                    pnl_pct=float(row.get('ReturnPct', 0)),
                    signal='BUY' if position == 'LONG' else 'SELL'
                )
                trades.append(trade)

        # Equity Curve
        equity_curve = stats._equity_curve['Equity'] if hasattr(stats, '_equity_curve') else pd.Series([initial_capital])

        # BacktestResult
        result = BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            initial_capital=initial_capital
        )

        # Metriken aus Stats uebernehmen
        try:
            result.total_pnl = float(stats.get('Equity Final [$]', initial_capital) - initial_capital)
            result.total_return_pct = float(stats.get('Return [%]', 0))
            result.sharpe_ratio = float(stats.get('Sharpe Ratio', 0) or 0)
            result.max_drawdown_pct = float(stats.get('Max. Drawdown [%]', 0))
            result.win_rate = float(stats.get('Win Rate [%]', 0))
            result.num_trades = int(stats.get('# Trades', 0))
            result.profit_factor = float(stats.get('Profit Factor', 0) or 0)
        except Exception:
            pass

        return result

    def get_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        initial_capital: float = 10000.0
    ):
        """
        Gibt das Backtesting.py Backtest-Objekt zurueck.

        Nuetzlich fuer erweiterte Konfiguration oder Visualisierung.

        Args:
            data: DataFrame mit OHLCV-Daten
            signals: Series mit Signalen
            initial_capital: Startkapital

        Returns:
            Backtesting.py Backtest-Objekt (nicht ausgefuehrt)
        """
        df = self._prepare_data(data, signals)
        SignalStrategy = self._create_strategy(df)

        return self._bt(
            df,
            SignalStrategy,
            cash=initial_capital,
            commission=self._params['commission'],
            margin=self._params['margin'],
            trade_on_close=self._params['trade_on_close'],
        )

    def plot(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        initial_capital: float = 10000.0,
        filename: Optional[str] = None,
        **kwargs
    ):
        """
        Fuehrt Backtest aus und zeigt interaktiven Plot.

        Args:
            data: DataFrame mit OHLCV-Daten
            signals: Series mit Signalen
            initial_capital: Startkapital
            filename: Optional: HTML-Datei speichern
            **kwargs: Zusaetzliche Plot-Parameter
        """
        bt = self.get_backtest(data, signals, initial_capital)
        stats = bt.run()
        bt.plot(filename=filename, **kwargs)
        return stats

    def optimize(
        self,
        data: pd.DataFrame,
        strategy_class,
        param_ranges: Dict[str, range],
        maximize: str = 'Sharpe Ratio',
        initial_capital: float = 10000.0,
        **kwargs
    ):
        """
        Optimiert Strategie-Parameter mit Backtesting.py.

        Args:
            data: DataFrame mit OHLCV-Daten
            strategy_class: Backtesting.py Strategy-Klasse
            param_ranges: Dictionary mit Parameter-Ranges
            maximize: Metrik zur Optimierung
            initial_capital: Startkapital
            **kwargs: Zusaetzliche Backtest-Parameter

        Returns:
            Beste Stats und Parameter
        """
        bt = self._bt(
            data,
            strategy_class,
            cash=initial_capital,
            commission=self._params['commission'],
            **kwargs
        )

        stats, heatmap = bt.optimize(
            maximize=maximize,
            **param_ranges,
            return_heatmap=True
        )

        return stats, heatmap
