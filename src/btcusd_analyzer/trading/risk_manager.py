"""
Risk Manager - Risiko-Kontrolle fuer Trading
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Callable
from enum import Enum
from threading import Lock

from ..core.logger import get_logger
from .order_manager import Order, OrderSide, Position, PositionSide


class RiskLevel(Enum):
    """Risiko-Level."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class RiskLimits:
    """Risiko-Limits Konfiguration."""
    # Position Limits
    max_position_size: float = 1.0           # Max BTC
    max_position_value: float = 50000.0      # Max USD
    max_leverage: float = 1.0                # Kein Hebel

    # Verlust Limits
    max_daily_loss: float = 500.0            # Max Tagesverlust USD
    max_daily_loss_pct: float = 5.0          # Max Tagesverlust %
    max_trade_loss: float = 100.0            # Max Verlust pro Trade USD
    max_trade_loss_pct: float = 2.0          # Max Verlust pro Trade %

    # Drawdown Limits
    max_drawdown: float = 1000.0             # Max Drawdown USD
    max_drawdown_pct: float = 10.0           # Max Drawdown %

    # Order Limits
    max_orders_per_hour: int = 10            # Rate Limiting
    max_orders_per_day: int = 50
    min_order_interval: int = 5              # Sekunden

    # Stop-Loss
    require_stop_loss: bool = True           # SL zwingend
    default_stop_loss_pct: float = 2.0       # Default SL %
    max_stop_loss_distance_pct: float = 10.0 # Max SL Abstand

    # Take-Profit
    default_take_profit_pct: float = 4.0     # Default TP %
    min_risk_reward_ratio: float = 1.5       # Min R:R


@dataclass
class RiskMetrics:
    """Aktuelle Risiko-Metriken."""
    current_exposure: float = 0.0            # Aktuelle Position USD
    daily_pnl: float = 0.0                   # Tages P/L
    peak_equity: float = 0.0                 # Hoechster Kontostand
    current_drawdown: float = 0.0            # Aktueller Drawdown
    orders_today: int = 0                    # Orders heute
    orders_this_hour: int = 0                # Orders diese Stunde
    last_order_time: Optional[datetime] = None
    risk_level: RiskLevel = RiskLevel.LOW
    warnings: List[str] = field(default_factory=list)


class RiskManager:
    """
    Risiko-Management fuer Live-Trading.

    Features:
    - Position-Limits
    - Verlust-Limits (Daily, Trade, Drawdown)
    - Order Rate-Limiting
    - Stop-Loss/Take-Profit Enforcement
    - Risk-Level Monitoring
    - Automatische Warnungen

    Usage:
        risk = RiskManager(limits=RiskLimits())

        # Vor Order pruefen
        if risk.check_order(order, position, account_balance):
            submit_order(order)
        else:
            print(risk.get_rejection_reason())

        # Nach Trade aktualisieren
        risk.update_after_trade(trade_pnl)

        # Risk-Level pruefen
        if risk.get_risk_level() == RiskLevel.CRITICAL:
            close_all_positions()
    """

    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        initial_balance: float = 10000.0
    ):
        """
        Initialisiert den Risk-Manager.

        Args:
            limits: Risiko-Limits
            initial_balance: Startkapital
        """
        self.logger = get_logger()
        self.limits = limits or RiskLimits()
        self.initial_balance = initial_balance

        self.metrics = RiskMetrics(peak_equity=initial_balance)
        self._lock = Lock()

        # Tracking
        self._daily_trades: List[Dict] = []
        self._hourly_orders: List[datetime] = []
        self._rejection_reason: str = ""

        # Callbacks
        self._on_risk_warning: List[Callable[[str, RiskLevel], None]] = []
        self._on_limit_breached: List[Callable[[str], None]] = []

        self.logger.info('RiskManager initialisiert')

    def check_order(
        self,
        order: Order,
        position: Position,
        account_balance: float,
        current_price: float
    ) -> bool:
        """
        Prueft ob eine Order den Risiko-Limits entspricht.

        Args:
            order: Zu pruefende Order
            position: Aktuelle Position
            account_balance: Kontoguthaben
            current_price: Aktueller Preis

        Returns:
            True wenn Order erlaubt
        """
        self._rejection_reason = ""
        warnings = []

        # 1. Rate Limiting
        if not self._check_rate_limits():
            return False

        # 2. Position Size Check
        new_position_size = self._calculate_new_position_size(order, position)
        position_value = new_position_size * current_price

        if new_position_size > self.limits.max_position_size:
            self._rejection_reason = f"Max Position Size ueberschritten: {new_position_size:.4f} > {self.limits.max_position_size}"
            return False

        if position_value > self.limits.max_position_value:
            self._rejection_reason = f"Max Position Value ueberschritten: ${position_value:,.2f} > ${self.limits.max_position_value:,.2f}"
            return False

        # 3. Daily Loss Check
        if self.metrics.daily_pnl < 0:
            if abs(self.metrics.daily_pnl) >= self.limits.max_daily_loss:
                self._rejection_reason = f"Daily Loss Limit erreicht: ${abs(self.metrics.daily_pnl):,.2f}"
                self._trigger_limit_breached("daily_loss")
                return False

            daily_loss_pct = (abs(self.metrics.daily_pnl) / self.initial_balance) * 100
            if daily_loss_pct >= self.limits.max_daily_loss_pct:
                self._rejection_reason = f"Daily Loss % erreicht: {daily_loss_pct:.1f}%"
                self._trigger_limit_breached("daily_loss_pct")
                return False

        # 4. Drawdown Check
        if self.metrics.current_drawdown >= self.limits.max_drawdown:
            self._rejection_reason = f"Max Drawdown erreicht: ${self.metrics.current_drawdown:,.2f}"
            self._trigger_limit_breached("max_drawdown")
            return False

        drawdown_pct = (self.metrics.current_drawdown / self.metrics.peak_equity) * 100 if self.metrics.peak_equity > 0 else 0
        if drawdown_pct >= self.limits.max_drawdown_pct:
            self._rejection_reason = f"Max Drawdown % erreicht: {drawdown_pct:.1f}%"
            self._trigger_limit_breached("max_drawdown_pct")
            return False

        # 5. Stop-Loss Requirement
        if self.limits.require_stop_loss and position.stop_loss is None:
            warnings.append("Kein Stop-Loss gesetzt")
            self._trigger_risk_warning("Stop-Loss fehlt", RiskLevel.MEDIUM)

        # 6. Max Trade Loss Check
        potential_loss = self._calculate_potential_loss(order, position, current_price)
        if potential_loss > self.limits.max_trade_loss:
            warnings.append(f"Potenzieller Verlust hoch: ${potential_loss:,.2f}")
            self._trigger_risk_warning(f"Hoher potenzieller Verlust: ${potential_loss:,.2f}", RiskLevel.HIGH)

        # Update metrics warnings
        with self._lock:
            self.metrics.warnings = warnings

        # Order erlaubt
        self._record_order()
        return True

    def _check_rate_limits(self) -> bool:
        """Prueft Rate-Limits."""
        now = datetime.now()

        with self._lock:
            # Orders diese Stunde zaehlen
            hour_ago = now - timedelta(hours=1)
            self._hourly_orders = [t for t in self._hourly_orders if t > hour_ago]

            if len(self._hourly_orders) >= self.limits.max_orders_per_hour:
                self._rejection_reason = f"Rate Limit: Max {self.limits.max_orders_per_hour} Orders/Stunde"
                return False

            # Orders heute zaehlen
            if self.metrics.orders_today >= self.limits.max_orders_per_day:
                self._rejection_reason = f"Rate Limit: Max {self.limits.max_orders_per_day} Orders/Tag"
                return False

            # Min Intervall
            if self.metrics.last_order_time:
                elapsed = (now - self.metrics.last_order_time).total_seconds()
                if elapsed < self.limits.min_order_interval:
                    self._rejection_reason = f"Min Intervall: {self.limits.min_order_interval}s"
                    return False

        return True

    def _calculate_new_position_size(self, order: Order, position: Position) -> float:
        """Berechnet neue Positionsgroesse nach Order."""
        if order.side == OrderSide.BUY:
            if position.side == PositionSide.LONG or position.side == PositionSide.NONE:
                return position.size + order.quantity
            else:  # SHORT
                return abs(position.size - order.quantity)
        else:  # SELL
            if position.side == PositionSide.SHORT or position.side == PositionSide.NONE:
                return position.size + order.quantity
            else:  # LONG
                return abs(position.size - order.quantity)

    def _calculate_potential_loss(
        self,
        order: Order,
        position: Position,
        current_price: float
    ) -> float:
        """Berechnet potenziellen Verlust."""
        if position.stop_loss:
            # Mit Stop-Loss
            if position.side == PositionSide.LONG:
                loss_per_unit = position.entry_price - position.stop_loss
            else:
                loss_per_unit = position.stop_loss - position.entry_price
            return loss_per_unit * (position.size + order.quantity)
        else:
            # Ohne Stop-Loss: Default % verwenden
            return current_price * order.quantity * (self.limits.default_stop_loss_pct / 100)

    def _record_order(self) -> None:
        """Zeichnet Order fuer Rate-Limiting auf."""
        now = datetime.now()
        with self._lock:
            self._hourly_orders.append(now)
            self.metrics.orders_today += 1
            self.metrics.orders_this_hour = len(self._hourly_orders)
            self.metrics.last_order_time = now

    def update_after_trade(self, pnl: float, is_closed: bool = False) -> None:
        """
        Aktualisiert Metriken nach Trade.

        Args:
            pnl: Profit/Loss des Trades
            is_closed: True wenn Trade geschlossen
        """
        with self._lock:
            self.metrics.daily_pnl += pnl

            # Peak Equity tracken
            current_equity = self.initial_balance + self.metrics.daily_pnl
            if current_equity > self.metrics.peak_equity:
                self.metrics.peak_equity = current_equity

            # Drawdown berechnen
            self.metrics.current_drawdown = self.metrics.peak_equity - current_equity

            # Trade aufzeichnen
            if is_closed:
                self._daily_trades.append({
                    'time': datetime.now(),
                    'pnl': pnl
                })

            # Risk Level aktualisieren
            self._update_risk_level()

    def update_position(self, position: Position, current_price: float) -> None:
        """
        Aktualisiert Exposure basierend auf Position.

        Args:
            position: Aktuelle Position
            current_price: Aktueller Preis
        """
        with self._lock:
            self.metrics.current_exposure = position.size * current_price

            # SL/TP Checks
            if position.is_open:
                if position.stop_loss:
                    if position.side == PositionSide.LONG and current_price <= position.stop_loss:
                        self._trigger_risk_warning("Stop-Loss erreicht!", RiskLevel.HIGH)
                    elif position.side == PositionSide.SHORT and current_price >= position.stop_loss:
                        self._trigger_risk_warning("Stop-Loss erreicht!", RiskLevel.HIGH)

                if position.take_profit:
                    if position.side == PositionSide.LONG and current_price >= position.take_profit:
                        self._trigger_risk_warning("Take-Profit erreicht!", RiskLevel.LOW)
                    elif position.side == PositionSide.SHORT and current_price <= position.take_profit:
                        self._trigger_risk_warning("Take-Profit erreicht!", RiskLevel.LOW)

    def _update_risk_level(self) -> None:
        """Aktualisiert das Risk-Level."""
        level = RiskLevel.LOW

        # Daily Loss
        daily_loss_pct = (abs(self.metrics.daily_pnl) / self.initial_balance) * 100 if self.metrics.daily_pnl < 0 else 0

        if daily_loss_pct > self.limits.max_daily_loss_pct * 0.8:
            level = RiskLevel.CRITICAL
        elif daily_loss_pct > self.limits.max_daily_loss_pct * 0.5:
            level = RiskLevel.HIGH
        elif daily_loss_pct > self.limits.max_daily_loss_pct * 0.25:
            level = RiskLevel.MEDIUM

        # Drawdown
        drawdown_pct = (self.metrics.current_drawdown / self.metrics.peak_equity) * 100 if self.metrics.peak_equity > 0 else 0

        if drawdown_pct > self.limits.max_drawdown_pct * 0.8:
            level = max(level, RiskLevel.CRITICAL, key=lambda x: list(RiskLevel).index(x))
        elif drawdown_pct > self.limits.max_drawdown_pct * 0.5:
            level = max(level, RiskLevel.HIGH, key=lambda x: list(RiskLevel).index(x))

        self.metrics.risk_level = level

        if level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            self._trigger_risk_warning(f"Risk Level: {level.value}", level)

    def get_rejection_reason(self) -> str:
        """Gibt den Grund der letzten Ablehnung zurueck."""
        return self._rejection_reason

    def get_risk_level(self) -> RiskLevel:
        """Gibt aktuelles Risk-Level zurueck."""
        return self.metrics.risk_level

    def get_metrics(self) -> RiskMetrics:
        """Gibt aktuelle Metriken zurueck."""
        with self._lock:
            return self.metrics

    def calculate_position_size(
        self,
        account_balance: float,
        risk_per_trade_pct: float,
        entry_price: float,
        stop_loss_price: float
    ) -> float:
        """
        Berechnet optimale Positionsgroesse basierend auf Risiko.

        Args:
            account_balance: Kontoguthaben
            risk_per_trade_pct: Risiko pro Trade in %
            entry_price: Einstiegspreis
            stop_loss_price: Stop-Loss Preis

        Returns:
            Empfohlene Positionsgroesse
        """
        risk_amount = account_balance * (risk_per_trade_pct / 100)
        price_risk = abs(entry_price - stop_loss_price)

        if price_risk <= 0:
            return 0.0

        position_size = risk_amount / price_risk

        # Limits anwenden
        position_size = min(position_size, self.limits.max_position_size)
        position_value = position_size * entry_price
        if position_value > self.limits.max_position_value:
            position_size = self.limits.max_position_value / entry_price

        return position_size

    def calculate_stop_loss(
        self,
        entry_price: float,
        side: PositionSide,
        method: str = 'percentage'
    ) -> float:
        """
        Berechnet Stop-Loss Preis.

        Args:
            entry_price: Einstiegspreis
            side: Position-Seite
            method: 'percentage' oder 'atr'

        Returns:
            Stop-Loss Preis
        """
        if method == 'percentage':
            pct = self.limits.default_stop_loss_pct / 100
            if side == PositionSide.LONG:
                return entry_price * (1 - pct)
            else:
                return entry_price * (1 + pct)
        else:
            # ATR-basiert wuerde ATR-Daten erfordern
            return self.calculate_stop_loss(entry_price, side, 'percentage')

    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss_price: float,
        side: PositionSide
    ) -> float:
        """
        Berechnet Take-Profit basierend auf Risk-Reward.

        Args:
            entry_price: Einstiegspreis
            stop_loss_price: Stop-Loss Preis
            side: Position-Seite

        Returns:
            Take-Profit Preis
        """
        risk = abs(entry_price - stop_loss_price)
        reward = risk * self.limits.min_risk_reward_ratio

        if side == PositionSide.LONG:
            return entry_price + reward
        else:
            return entry_price - reward

    def reset_daily(self) -> None:
        """Setzt taegliche Metriken zurueck."""
        with self._lock:
            self.metrics.daily_pnl = 0.0
            self.metrics.orders_today = 0
            self._daily_trades = []
        self.logger.info('Taegliche Metriken zurueckgesetzt')

    # === Callbacks ===

    def on_risk_warning(self, callback: Callable[[str, RiskLevel], None]) -> None:
        """Registriert Callback fuer Risiko-Warnungen."""
        self._on_risk_warning.append(callback)

    def on_limit_breached(self, callback: Callable[[str], None]) -> None:
        """Registriert Callback wenn Limit ueberschritten."""
        self._on_limit_breached.append(callback)

    def _trigger_risk_warning(self, message: str, level: RiskLevel) -> None:
        self.logger.warning(f'[RISK {level.value}] {message}')
        for callback in self._on_risk_warning:
            try:
                callback(message, level)
            except Exception as e:
                self.logger.error(f'Callback-Fehler: {e}')

    def _trigger_limit_breached(self, limit_name: str) -> None:
        self.logger.error(f'[LIMIT BREACHED] {limit_name}')
        for callback in self._on_limit_breached:
            try:
                callback(limit_name)
            except Exception as e:
                self.logger.error(f'Callback-Fehler: {e}')
