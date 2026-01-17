"""
Backtrader Engine - Integration von Backtrader fuer professionelles Backtesting

Nutzt das trainierte BILSTM-Modell fuer Signal-Generierung und
fuehrt Backtests mit Backtrader durch.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd
import backtrader as bt

from ..core.logger import get_logger
from ..data.processor import FeatureProcessor
from ..training.normalizer import ZScoreNormalizer


logger = get_logger()


# =============================================================================
# Datenstrukturen
# =============================================================================

@dataclass
class BacktestResult:
    """Ergebnis eines Backtrader-Backtests."""
    # Konto-Metriken
    initial_capital: float = 10000.0
    final_value: float = 0.0
    total_return: float = 0.0
    total_return_pct: float = 0.0

    # Risiko-Metriken
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Trade-Metriken
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_winning_trade: float = 0.0
    avg_losing_trade: float = 0.0
    max_win: float = 0.0
    max_loss: float = 0.0

    # Equity-Kurve
    equity_curve: List[float] = field(default_factory=list)
    dates: List = field(default_factory=list)

    # Trade-Liste
    trades: List[Dict] = field(default_factory=list)

    # Signale
    buy_signals: int = 0
    sell_signals: int = 0
    hold_signals: int = 0


# =============================================================================
# Custom DataFeed mit Signal-Spalte
# =============================================================================

class BilstmDataFeed(bt.feeds.PandasData):
    """
    Erweiterter PandasData Feed mit Signal-Spalte fuer BILSTM-Vorhersagen.

    Signal-Werte:
        0 = HOLD
        1 = BUY
        2 = SELL
    """
    # Zusaetzliche Daten-Linie fuer Signal
    lines = ('signal',)

    # Parameter-Mapping
    params = (
        ('datetime', None),  # Index ist datetime
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', None),
        ('signal', 'signal'),
    )


# =============================================================================
# Trading Strategy
# =============================================================================

class BilstmStrategy(bt.Strategy):
    """
    Trading-Strategie basierend auf BILSTM-Modell-Signalen.

    Logik:
        - Signal 1 (BUY): Long-Position oeffnen oder Short schliessen
        - Signal 2 (SELL): Short-Position oeffnen oder Long schliessen
        - Signal 0 (HOLD): Keine Aktion
    """
    params = (
        ('stake', 1.0),              # Position Size (Anzahl Einheiten)
        ('stake_pct', None),         # Position Size als % des Kapitals
        ('allow_short', True),       # Short-Positionen erlauben
        ('printlog', False),         # Trade-Logging
        ('invert_signals', False),   # Signale invertieren (BUY<->SELL)
    )

    def __init__(self):
        """Initialisiert die Strategie."""
        self.signal = self.data.signal
        self.order = None
        self.entry_price = None
        self.entry_bar = None

        # Trade-Tracking
        self.trade_list = []
        self.signal_counts = {'buy': 0, 'sell': 0, 'hold': 0}

    def log(self, txt: str, dt=None):
        """Logging-Hilfsfunktion."""
        if self.params.printlog:
            dt = dt or self.data.datetime.date(0)
            logger.info(f'{dt}: {txt}')

    def notify_order(self, order):
        """Wird bei Order-Status-Aenderungen aufgerufen."""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            if order.isbuy():
                self.log(f'BUY EXECUTED @ {order.executed.price:.2f}')
                self.entry_price = order.executed.price
                self.entry_bar = len(self)
            else:
                self.log(f'SELL EXECUTED @ {order.executed.price:.2f}')
                if self.entry_price:
                    pnl = order.executed.price - self.entry_price
                    self.log(f'TRADE P/L: {pnl:.2f}')
                self.entry_price = order.executed.price
                self.entry_bar = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        """Wird bei Trade-Abschluss aufgerufen."""
        if trade.isclosed:
            self.trade_list.append({
                'entry_date': bt.num2date(trade.dtopen),
                'exit_date': bt.num2date(trade.dtclose),
                'size': trade.size,
                'entry_price': trade.price,
                'exit_price': trade.price + (trade.pnl / abs(trade.size) if trade.size != 0 else 0),
                'pnl': trade.pnl,
                'pnl_pct': trade.pnlcomm / self.broker.get_value() * 100 if self.broker.get_value() > 0 else 0,
                'commission': trade.commission,
            })
            self.log(f'TRADE CLOSED - P/L: {trade.pnl:.2f} (Comm: {trade.commission:.2f})')

    def next(self):
        """Haupt-Trading-Logik - wird fuer jeden Bar aufgerufen."""
        # Warten falls Order pending
        if self.order:
            return

        # Signal auslesen
        current_signal = int(self.signal[0])

        # Optional: Signale invertieren
        if self.params.invert_signals:
            if current_signal == 1:
                current_signal = 2
            elif current_signal == 2:
                current_signal = 1

        # Signal-Zaehlung
        if current_signal == 0:
            self.signal_counts['hold'] += 1
        elif current_signal == 1:
            self.signal_counts['buy'] += 1
        elif current_signal == 2:
            self.signal_counts['sell'] += 1

        # Position Size berechnen
        if self.params.stake_pct:
            stake = (self.broker.get_value() * self.params.stake_pct / 100) / self.data.close[0]
        else:
            stake = self.params.stake

        # Trading-Logik
        if not self.position:
            # Keine Position - neue oeffnen
            if current_signal == 1:  # BUY
                self.order = self.buy(size=stake)
                self.log(f'BUY ORDER @ {self.data.close[0]:.2f}')
            elif current_signal == 2 and self.params.allow_short:  # SELL (Short)
                self.order = self.sell(size=stake)
                self.log(f'SELL ORDER @ {self.data.close[0]:.2f}')
        else:
            # Position vorhanden - Exit pruefen
            if self.position.size > 0:  # Long Position
                if current_signal == 2:  # SELL Signal -> Close Long
                    self.order = self.close()
                    self.log(f'CLOSE LONG @ {self.data.close[0]:.2f}')
            elif self.position.size < 0:  # Short Position
                if current_signal == 1:  # BUY Signal -> Close Short
                    self.order = self.close()
                    self.log(f'CLOSE SHORT @ {self.data.close[0]:.2f}')


# =============================================================================
# Backtrader Engine
# =============================================================================

class BacktraderEngine:
    """
    Wrapper-Klasse fuer einfache Backtrader-Integration.

    Verwendung:
        engine = BacktraderEngine(data, model, model_info)
        result = engine.run_backtest(initial_capital=10000, commission=0.001)
        print(result.sharpe_ratio)
    """

    def __init__(self, data: pd.DataFrame, model=None, model_info: Dict = None):
        """
        Initialisiert die Engine.

        Args:
            data: DataFrame mit OHLCV-Daten (muss datetime-Index haben)
            model: Trainiertes BILSTM-Modell (optional)
            model_info: Modell-Metadaten mit 'features', 'lookback_size', etc.
        """
        self.data = data.copy()
        self.model = model
        self.model_info = model_info or {}

        self.cerebro = None
        self.results = None
        self.prepared_data = None

    def prepare_data(self, signals: pd.Series = None) -> pd.DataFrame:
        """
        Bereitet Daten vor und fuegt Signal-Spalte hinzu.

        Args:
            signals: Optional vordefinierte Signale (pd.Series mit BUY/SELL/HOLD)

        Returns:
            DataFrame mit 'signal' Spalte
        """
        df = self.data.copy()

        # Stelle sicher dass datetime Index vorhanden ist
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'DateTime' in df.columns:
                df.set_index('DateTime', inplace=True)
            elif 'Date' in df.columns:
                df.set_index('Date', inplace=True)

        # Volume hinzufuegen falls nicht vorhanden
        if 'Volume' not in df.columns:
            df['Volume'] = 0

        # Signal-Spalte initialisieren
        df['signal'] = 0

        if signals is not None:
            # Vordefinierte Signale verwenden
            signal_map = {'BUY': 1, 'SELL': 2, 'HOLD': 0}
            df['signal'] = signals.map(lambda x: signal_map.get(x, 0) if isinstance(x, str) else x)

        elif self.model is not None:
            # Signale aus Modell berechnen
            df = self._compute_model_signals(df)

        self.prepared_data = df
        return df

    def _compute_model_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Berechnet Signale aus dem BILSTM-Modell."""
        import torch

        # Feature-Extraktion
        features = self.model_info.get('features', ['Open', 'High', 'Low', 'Close', 'PriceChange', 'PriceChangePct'])
        processor = FeatureProcessor(features=features)
        feature_df = processor.process(df)
        feature_matrix = processor.get_feature_matrix(feature_df)

        # Normalisierung
        normalizer = ZScoreNormalizer()
        normalized = normalizer.fit_transform(feature_matrix)

        # Sequenzen erstellen
        lookback = self.model_info.get('lookback_size', 60)
        sequences = []
        for i in range(lookback, len(normalized)):
            seq = normalized[i-lookback:i]
            sequences.append(seq)

        if not sequences:
            logger.warning("[BacktraderEngine] Keine Sequenzen erstellt - zu wenig Daten")
            return df

        sequences = np.array(sequences)

        # Modell-Vorhersagen
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(sequences)
            if hasattr(self.model, 'device'):
                X = X.to(self.model.device)
            predictions = self.model.predict(X)
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.cpu().numpy()

        # Signal-Mapping (abhaengig von num_classes)
        num_classes = self.model_info.get('num_classes', 3)
        if num_classes == 2:
            # Binaer: 0->BUY, 1->SELL
            predictions = predictions + 1
        # Bei 3 Klassen: 0=HOLD, 1=BUY, 2=SELL (kein Mapping noetig)

        # Signale in DataFrame einfuegen
        signal_start = lookback
        df.iloc[signal_start:signal_start + len(predictions), df.columns.get_loc('signal')] = predictions

        logger.info(f"[BacktraderEngine] Modell-Signale berechnet: {len(predictions)} Vorhersagen")
        return df

    def run_backtest(self,
                     initial_capital: float = 10000.0,
                     commission: float = 0.001,
                     slippage: float = 0.0,
                     stake: float = 1.0,
                     stake_pct: float = None,
                     allow_short: bool = True,
                     invert_signals: bool = False,
                     signals: pd.Series = None) -> BacktestResult:
        """
        Fuehrt den Backtest aus.

        Args:
            initial_capital: Startkapital
            commission: Kommission pro Trade (0.001 = 0.1%)
            slippage: Slippage pro Trade
            stake: Feste Position Size (Anzahl Einheiten)
            stake_pct: Position Size als % des Kapitals
            allow_short: Short-Positionen erlauben
            invert_signals: Signale invertieren
            signals: Optional vordefinierte Signale

        Returns:
            BacktestResult mit allen Metriken
        """
        # Daten vorbereiten
        if self.prepared_data is None:
            self.prepare_data(signals)

        # Cerebro initialisieren
        self.cerebro = bt.Cerebro()

        # Broker konfigurieren
        self.cerebro.broker.setcash(initial_capital)
        self.cerebro.broker.setcommission(commission=commission)
        if slippage > 0:
            self.cerebro.broker.set_slippage_perc(slippage)

        # Daten hinzufuegen
        data_feed = BilstmDataFeed(dataname=self.prepared_data)
        self.cerebro.adddata(data_feed)

        # Strategie hinzufuegen
        self.cerebro.addstrategy(
            BilstmStrategy,
            stake=stake,
            stake_pct=stake_pct,
            allow_short=allow_short,
            invert_signals=invert_signals,
            printlog=False,
        )

        # Analyzer hinzufuegen
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0)
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')

        # Backtest ausfuehren
        logger.info(f"[BacktraderEngine] Starte Backtest mit {initial_capital:.2f} Startkapital")
        self.results = self.cerebro.run()

        # Ergebnisse extrahieren
        return self._extract_results(initial_capital)

    def _extract_results(self, initial_capital: float) -> BacktestResult:
        """Extrahiert Ergebnisse aus Cerebro."""
        result = BacktestResult(initial_capital=initial_capital)

        strat = self.results[0]

        # Konto-Metriken
        result.final_value = self.cerebro.broker.getvalue()
        result.total_return = result.final_value - initial_capital
        result.total_return_pct = (result.total_return / initial_capital) * 100

        # Drawdown
        try:
            dd = strat.analyzers.drawdown.get_analysis()
            result.max_drawdown = dd.get('max', {}).get('moneydown', 0.0)
            result.max_drawdown_pct = dd.get('max', {}).get('drawdown', 0.0)
        except Exception:
            pass

        # Sharpe Ratio
        try:
            sharpe = strat.analyzers.sharpe.get_analysis()
            result.sharpe_ratio = sharpe.get('sharperatio', 0.0) or 0.0
        except Exception:
            pass

        # Trade-Analyse
        try:
            ta = strat.analyzers.trades.get_analysis()
            result.total_trades = ta.get('total', {}).get('closed', 0)
            result.winning_trades = ta.get('won', {}).get('total', 0)
            result.losing_trades = ta.get('lost', {}).get('total', 0)

            if result.total_trades > 0:
                result.win_rate = (result.winning_trades / result.total_trades) * 100

            result.avg_winning_trade = ta.get('won', {}).get('pnl', {}).get('average', 0.0)
            result.avg_losing_trade = ta.get('lost', {}).get('pnl', {}).get('average', 0.0)
            result.max_win = ta.get('won', {}).get('pnl', {}).get('max', 0.0)
            result.max_loss = ta.get('lost', {}).get('pnl', {}).get('max', 0.0)

            total_won = ta.get('won', {}).get('pnl', {}).get('total', 0.0)
            total_lost = abs(ta.get('lost', {}).get('pnl', {}).get('total', 0.0))
            if total_lost > 0:
                result.profit_factor = total_won / total_lost

            if result.total_trades > 0:
                result.avg_trade_pnl = result.total_return / result.total_trades
        except Exception:
            pass

        # Signal-Counts
        result.buy_signals = strat.signal_counts.get('buy', 0)
        result.sell_signals = strat.signal_counts.get('sell', 0)
        result.hold_signals = strat.signal_counts.get('hold', 0)

        # Trade-Liste
        result.trades = strat.trade_list

        # Calmar Ratio berechnen
        if result.max_drawdown_pct > 0:
            # Vereinfachte Berechnung: Return / MaxDD
            result.calmar_ratio = result.total_return_pct / result.max_drawdown_pct

        logger.info(f"[BacktraderEngine] Backtest abgeschlossen: Return {result.total_return_pct:.2f}%, "
                   f"Sharpe {result.sharpe_ratio:.2f}, Trades {result.total_trades}")

        return result

    def get_equity_curve(self) -> pd.DataFrame:
        """
        Gibt die Equity-Kurve zurueck.

        Hinweis: Muss nach run_backtest() aufgerufen werden.
        """
        # TODO: Observer fuer Equity-Curve hinzufuegen
        return pd.DataFrame()

    def plot(self, filename: str = None):
        """
        Erstellt Backtrader-Plot.

        Args:
            filename: Optional Dateiname zum Speichern
        """
        if self.cerebro is None:
            raise ValueError("Backtest muss zuerst ausgefuehrt werden")

        self.cerebro.plot(style='candlestick')
