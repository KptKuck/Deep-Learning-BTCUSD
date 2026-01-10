"""
Live Trader - Live-Trading Engine
"""

import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from ..core.logger import get_logger
from ..models.base import BaseModel
from .binance_client import BinanceClient, TradingMode


class PositionSide(Enum):
    """Position-Richtung."""
    NONE = "NONE"
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Position:
    """Repraesentiert eine offene Position."""
    side: PositionSide
    entry_price: float
    entry_time: datetime
    size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    @property
    def is_open(self) -> bool:
        return self.side != PositionSide.NONE


@dataclass
class TradeResult:
    """Ergebnis eines Trades."""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    side: str
    size: float
    pnl: float
    pnl_pct: float
    exit_reason: str


class LiveTrader:
    """
    Live-Trading Engine.

    Fuehrt Trades basierend auf Modell-Vorhersagen aus.

    WICHTIG: Startet IMMER im TESTNET-Modus!

    Features:
    - Modell-basierte Signalgenerierung
    - Position-Management
    - Stop-Loss / Take-Profit
    - Trade-Logging
    - Callbacks fuer UI-Updates

    Attributes:
        client: BinanceClient-Instanz
        model: Trainiertes Modell
        position: Aktuelle Position
    """

    def __init__(
        self,
        model: Optional[BaseModel] = None,
        mode: TradingMode = TradingMode.TESTNET,
        symbol: str = 'BTCUSDT'
    ):
        """
        Initialisiert den Live Trader.

        Args:
            model: Trainiertes Modell (kann spaeter gesetzt werden)
            mode: Trading-Modus (default: TESTNET!)
            symbol: Trading-Paar
        """
        self.logger = get_logger()
        self.client = BinanceClient(mode=mode, symbol=symbol)
        self.model = model
        self.symbol = symbol

        # State
        self.position = Position(
            side=PositionSide.NONE,
            entry_price=0.0,
            entry_time=datetime.now(),
            size=0.0
        )
        self.trades: List[TradeResult] = []
        self.is_running = False

        # Risiko-Parameter
        self.position_size_pct = 0.1  # 10% des Kapitals
        self.max_position_size = 1.0  # Max 1 BTC
        self.stop_loss_pct = 0.02  # 2%
        self.take_profit_pct = 0.05  # 5%

        # Callbacks
        self._on_signal: Optional[Callable] = None
        self._on_trade: Optional[Callable] = None
        self._on_update: Optional[Callable] = None

        self.logger.info(f'LiveTrader initialisiert: {mode.value.upper()}')

    @property
    def mode(self) -> TradingMode:
        """Aktueller Trading-Modus."""
        return self.client.mode

    @property
    def is_testnet(self) -> bool:
        """True wenn Testnet-Modus."""
        return self.client.is_testnet

    @property
    def is_live(self) -> bool:
        """True wenn Live-Modus."""
        return self.client.is_live

    def set_model(self, model: BaseModel):
        """Setzt das Modell."""
        self.model = model
        self.logger.info(f'Modell gesetzt: {model.name}')

    def set_callbacks(
        self,
        on_signal: Optional[Callable] = None,
        on_trade: Optional[Callable] = None,
        on_update: Optional[Callable] = None
    ):
        """Setzt Callback-Funktionen."""
        self._on_signal = on_signal
        self._on_trade = on_trade
        self._on_update = on_update

    def set_risk_params(
        self,
        position_size_pct: float = 0.1,
        max_position_size: float = 1.0,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.05
    ):
        """Setzt Risiko-Parameter."""
        self.position_size_pct = position_size_pct
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def switch_mode(self, mode: TradingMode) -> bool:
        """
        Wechselt den Trading-Modus.

        ACHTUNG: Nur moeglich wenn keine Position offen!

        Args:
            mode: Neuer Modus

        Returns:
            True bei Erfolg
        """
        if self.position.is_open:
            self.logger.error('Moduswechsel nicht moeglich: Position offen!')
            return False

        if self.is_running:
            self.logger.error('Moduswechsel nicht moeglich: Trading aktiv!')
            return False

        return self.client.switch_mode(mode)

    def predict_signal(self, sequence: np.ndarray) -> tuple:
        """
        Generiert Signal aus Sequenz.

        Args:
            sequence: Input-Sequenz [seq_length, features]

        Returns:
            Tuple aus (signal, probabilities)
            signal: 'HOLD', 'BUY', oder 'SELL'
        """
        if self.model is None:
            return 'HOLD', np.array([1.0, 0.0, 0.0])

        self.model.eval()
        with torch.no_grad():
            # Batch-Dimension hinzufuegen
            x = torch.FloatTensor(sequence).unsqueeze(0)

            if hasattr(self.model, 'device'):
                x = x.to(self.model.device)

            probs = self.model.predict_proba(x).cpu().numpy()[0]
            pred = np.argmax(probs)

        signal_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        signal = signal_map[pred]

        return signal, probs

    def execute_signal(self, signal: str) -> bool:
        """
        Fuehrt ein Signal aus.

        Args:
            signal: 'BUY', 'SELL', oder 'HOLD'

        Returns:
            True wenn Trade ausgefuehrt
        """
        current_price = self.client.get_ticker_price()

        if signal == 'BUY' and not self.position.is_open:
            return self._open_long(current_price)

        elif signal == 'SELL' and self.position.side == PositionSide.LONG:
            return self._close_position(current_price, 'SIGNAL')

        return False

    def _open_long(self, price: float) -> bool:
        """Oeffnet eine Long-Position."""
        # Positionsgroesse berechnen
        balance = self.client.get_balance('USDT')
        size = min(
            (balance * self.position_size_pct) / price,
            self.max_position_size
        )

        if size <= 0:
            self.logger.warning('Nicht genug Balance fuer Trade')
            return False

        # Order erstellen
        order = self.client.create_market_order('BUY', size)

        if order:
            self.position = Position(
                side=PositionSide.LONG,
                entry_price=price,
                entry_time=datetime.now(),
                size=size,
                stop_loss=price * (1 - self.stop_loss_pct),
                take_profit=price * (1 + self.take_profit_pct)
            )

            self.logger.success(f'LONG geoeffnet: {size:.6f} BTC @ ${price:,.2f}')

            if self._on_trade:
                self._on_trade('OPEN_LONG', price, size)

            return True

        return False

    def _close_position(self, price: float, reason: str) -> bool:
        """Schliesst die aktuelle Position."""
        if not self.position.is_open:
            return False

        # Order erstellen
        order = self.client.create_market_order('SELL', self.position.size)

        if order:
            # Trade-Ergebnis berechnen
            pnl = (price - self.position.entry_price) * self.position.size
            pnl_pct = ((price / self.position.entry_price) - 1) * 100

            trade = TradeResult(
                entry_time=self.position.entry_time,
                exit_time=datetime.now(),
                entry_price=self.position.entry_price,
                exit_price=price,
                side='LONG',
                size=self.position.size,
                pnl=pnl,
                pnl_pct=pnl_pct,
                exit_reason=reason
            )
            self.trades.append(trade)

            self.logger.success(f'Position geschlossen: ${pnl:+,.2f} ({pnl_pct:+.2f}%) - {reason}')

            # Position zuruecksetzen
            self.position = Position(
                side=PositionSide.NONE,
                entry_price=0.0,
                entry_time=datetime.now(),
                size=0.0
            )

            if self._on_trade:
                self._on_trade('CLOSE', price, trade)

            return True

        return False

    def check_stop_loss_take_profit(self) -> bool:
        """
        Prueft Stop-Loss und Take-Profit.

        Returns:
            True wenn Position geschlossen wurde
        """
        if not self.position.is_open:
            return False

        price = self.client.get_ticker_price()

        # Stop-Loss
        if self.position.stop_loss and price <= self.position.stop_loss:
            self.logger.warning(f'Stop-Loss ausgeloest @ ${price:,.2f}')
            return self._close_position(price, 'STOP_LOSS')

        # Take-Profit
        if self.position.take_profit and price >= self.position.take_profit:
            self.logger.success(f'Take-Profit erreicht @ ${price:,.2f}')
            return self._close_position(price, 'TAKE_PROFIT')

        return False

    def get_stats(self) -> Dict:
        """
        Gibt Trading-Statistiken zurueck.

        Returns:
            Dictionary mit Stats
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0
            }

        pnls = [t.pnl for t in self.trades]
        winners = [p for p in pnls if p > 0]

        return {
            'total_trades': len(self.trades),
            'win_rate': (len(winners) / len(self.trades)) * 100,
            'total_pnl': sum(pnls),
            'avg_pnl': sum(pnls) / len(pnls),
            'best_trade': max(pnls),
            'worst_trade': min(pnls)
        }

    def start(self, interval_seconds: float = 60.0):
        """
        Startet die Trading-Loop.

        Args:
            interval_seconds: Intervall zwischen Checks
        """
        if self.model is None:
            self.logger.error('Kein Modell gesetzt!')
            return

        self.is_running = True
        self.logger.info(f'Trading gestartet ({self.mode.value})')

        try:
            while self.is_running:
                # Stop-Loss/Take-Profit pruefen
                self.check_stop_loss_take_profit()

                # Update-Callback
                if self._on_update:
                    self._on_update(self.position, self.client.get_ticker_price())

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            self.logger.warning('Trading durch Benutzer gestoppt')
        finally:
            self.is_running = False

    def stop(self):
        """Stoppt die Trading-Loop."""
        self.is_running = False
        self.logger.info('Trading gestoppt')
