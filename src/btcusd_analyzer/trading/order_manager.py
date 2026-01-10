"""
Order Manager - Verwaltung von Orders und Positionen
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Callable
from threading import Lock
import uuid

from ..core.logger import get_logger


class OrderSide(Enum):
    """Order-Seite."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order-Typ."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"


class OrderStatus(Enum):
    """Order-Status."""
    PENDING = "PENDING"           # Noch nicht gesendet
    NEW = "NEW"                   # Gesendet, warten auf Ausfuehrung
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"             # Vollstaendig ausgefuehrt
    CANCELED = "CANCELED"         # Storniert
    REJECTED = "REJECTED"         # Abgelehnt
    EXPIRED = "EXPIRED"           # Abgelaufen


class PositionSide(Enum):
    """Position-Seite."""
    NONE = "NONE"
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Order:
    """Repraesentiert eine einzelne Order."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    exchange_id: Optional[str] = None  # ID von der Exchange
    symbol: str = "BTCUSDT"
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    price: Optional[float] = None      # Fuer Limit-Orders
    stop_price: Optional[float] = None # Fuer Stop-Orders
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    error_message: str = ""

    @property
    def is_active(self) -> bool:
        """True wenn Order noch aktiv (nicht abgeschlossen)."""
        return self.status in [OrderStatus.PENDING, OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]

    @property
    def is_filled(self) -> bool:
        """True wenn Order vollstaendig ausgefuehrt."""
        return self.status == OrderStatus.FILLED

    @property
    def fill_percentage(self) -> float:
        """Prozent der ausgefuehrten Menge."""
        if self.quantity <= 0:
            return 0.0
        return (self.filled_quantity / self.quantity) * 100

    def to_dict(self) -> Dict:
        """Konvertiert zu Dictionary."""
        return {
            'id': self.id,
            'exchange_id': self.exchange_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'avg_fill_price': self.avg_fill_price,
            'commission': self.commission,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


@dataclass
class Position:
    """Repraesentiert eine offene Position."""
    symbol: str = "BTCUSDT"
    side: PositionSide = PositionSide.NONE
    size: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    opened_at: Optional[datetime] = None

    @property
    def is_open(self) -> bool:
        """True wenn Position offen."""
        return self.side != PositionSide.NONE and self.size > 0

    @property
    def pnl_percentage(self) -> float:
        """Unrealisierter P/L in Prozent."""
        if self.entry_price <= 0:
            return 0.0
        return ((self.current_price - self.entry_price) / self.entry_price) * 100

    def update_pnl(self, current_price: float) -> None:
        """Aktualisiert unrealisierten P/L."""
        self.current_price = current_price

        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (current_price - self.entry_price) * self.size
        elif self.side == PositionSide.SHORT:
            self.unrealized_pnl = (self.entry_price - current_price) * self.size

    def to_dict(self) -> Dict:
        """Konvertiert zu Dictionary."""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'size': self.size,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'opened_at': self.opened_at.isoformat() if self.opened_at else None
        }


class OrderManager:
    """
    Verwaltet Orders und Positionen.

    Features:
    - Order-Erstellung und Tracking
    - Position-Management
    - Order-Historie
    - Event-Callbacks

    Usage:
        manager = OrderManager(client)

        # Order erstellen
        order = manager.create_order(
            side=OrderSide.BUY,
            quantity=0.01,
            order_type=OrderType.MARKET
        )

        # Order senden
        manager.submit_order(order)

        # Position pruefen
        position = manager.get_position()

        # Callbacks registrieren
        manager.on_order_filled(callback_function)
    """

    def __init__(self, client=None, symbol: str = "BTCUSDT"):
        """
        Initialisiert den Order-Manager.

        Args:
            client: BinanceClient Instanz
            symbol: Trading-Paar
        """
        self.logger = get_logger()
        self.client = client
        self.symbol = symbol

        # Orders und Positionen
        self._orders: Dict[str, Order] = {}
        self._position = Position(symbol=symbol)
        self._order_history: List[Order] = []

        # Thread-Safety
        self._lock = Lock()

        # Callbacks
        self._on_order_filled: List[Callable[[Order], None]] = []
        self._on_order_canceled: List[Callable[[Order], None]] = []
        self._on_position_changed: List[Callable[[Position], None]] = []

    def create_order(
        self,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Order:
        """
        Erstellt eine neue Order (sendet noch nicht).

        Args:
            side: BUY oder SELL
            quantity: Menge
            order_type: Order-Typ
            price: Limit-Preis (optional)
            stop_price: Stop-Preis (optional)

        Returns:
            Order-Objekt
        """
        order = Order(
            symbol=self.symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price
        )

        with self._lock:
            self._orders[order.id] = order

        self.logger.debug(f'Order erstellt: {order.id} {side.value} {quantity}')
        return order

    def submit_order(self, order: Order) -> bool:
        """
        Sendet eine Order an die Exchange.

        Args:
            order: Order-Objekt

        Returns:
            True wenn erfolgreich
        """
        if self.client is None:
            self.logger.error('Kein Client konfiguriert')
            order.status = OrderStatus.REJECTED
            order.error_message = 'Kein Client'
            return False

        try:
            # Order an Exchange senden
            if order.order_type == OrderType.MARKET:
                result = self.client.create_market_order(
                    side=order.side.value,
                    quantity=order.quantity,
                    symbol=order.symbol
                )
            elif order.order_type == OrderType.LIMIT:
                result = self.client.create_limit_order(
                    side=order.side.value,
                    quantity=order.quantity,
                    price=order.price,
                    symbol=order.symbol
                )
            else:
                self.logger.error(f'Nicht unterstuetzter Order-Typ: {order.order_type}')
                order.status = OrderStatus.REJECTED
                return False

            if result:
                order.exchange_id = str(result.get('orderId', ''))
                order.status = OrderStatus.NEW
                order.updated_at = datetime.now()

                # Bei Market-Order sofortige Ausfuehrung pruefen
                if result.get('status') == 'FILLED':
                    self._handle_order_filled(order, result)

                self.logger.success(f'Order gesendet: {order.id}')
                return True
            else:
                order.status = OrderStatus.REJECTED
                return False

        except Exception as e:
            self.logger.error(f'Order-Fehler: {e}')
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            return False

    def cancel_order(self, order_id: str) -> bool:
        """
        Storniert eine Order.

        Args:
            order_id: Interne Order-ID

        Returns:
            True wenn erfolgreich
        """
        with self._lock:
            order = self._orders.get(order_id)

        if not order:
            self.logger.warning(f'Order nicht gefunden: {order_id}')
            return False

        if not order.is_active:
            self.logger.warning(f'Order nicht mehr aktiv: {order_id}')
            return False

        if self.client and order.exchange_id:
            success = self.client.cancel_order(
                order_id=int(order.exchange_id),
                symbol=order.symbol
            )
            if success:
                order.status = OrderStatus.CANCELED
                order.updated_at = datetime.now()
                self._trigger_order_canceled(order)
                return True

        return False

    def cancel_all_orders(self) -> int:
        """
        Storniert alle aktiven Orders.

        Returns:
            Anzahl stornierter Orders
        """
        canceled = 0
        with self._lock:
            active_orders = [o for o in self._orders.values() if o.is_active]

        for order in active_orders:
            if self.cancel_order(order.id):
                canceled += 1

        self.logger.info(f'{canceled} Orders storniert')
        return canceled

    def _handle_order_filled(self, order: Order, result: Dict) -> None:
        """Behandelt ausgefuehrte Order."""
        order.status = OrderStatus.FILLED
        order.filled_quantity = float(result.get('executedQty', order.quantity))
        order.avg_fill_price = float(result.get('price', 0) or result.get('cummulativeQuoteQty', 0) / order.filled_quantity if order.filled_quantity > 0 else 0)
        order.updated_at = datetime.now()

        # Position aktualisieren
        self._update_position_from_order(order)

        # Callbacks
        self._trigger_order_filled(order)

        # Zur Historie hinzufuegen
        with self._lock:
            self._order_history.append(order)

    def _update_position_from_order(self, order: Order) -> None:
        """Aktualisiert Position basierend auf ausgefuehrter Order."""
        with self._lock:
            pos = self._position

            if order.side == OrderSide.BUY:
                if pos.side == PositionSide.NONE or pos.side == PositionSide.LONG:
                    # Position aufbauen/erhoehen
                    new_size = pos.size + order.filled_quantity
                    new_entry = ((pos.entry_price * pos.size) + (order.avg_fill_price * order.filled_quantity)) / new_size if new_size > 0 else order.avg_fill_price
                    pos.side = PositionSide.LONG
                    pos.size = new_size
                    pos.entry_price = new_entry
                    if not pos.opened_at:
                        pos.opened_at = datetime.now()
                elif pos.side == PositionSide.SHORT:
                    # Short-Position reduzieren/schliessen
                    if order.filled_quantity >= pos.size:
                        # Position geschlossen
                        pos.realized_pnl += (pos.entry_price - order.avg_fill_price) * pos.size
                        remaining = order.filled_quantity - pos.size
                        if remaining > 0:
                            # Wechsel zu Long
                            pos.side = PositionSide.LONG
                            pos.size = remaining
                            pos.entry_price = order.avg_fill_price
                            pos.opened_at = datetime.now()
                        else:
                            pos.side = PositionSide.NONE
                            pos.size = 0
                            pos.entry_price = 0
                            pos.opened_at = None
                    else:
                        # Position reduziert
                        pos.realized_pnl += (pos.entry_price - order.avg_fill_price) * order.filled_quantity
                        pos.size -= order.filled_quantity

            elif order.side == OrderSide.SELL:
                if pos.side == PositionSide.NONE or pos.side == PositionSide.SHORT:
                    # Short-Position aufbauen/erhoehen
                    new_size = pos.size + order.filled_quantity
                    new_entry = ((pos.entry_price * pos.size) + (order.avg_fill_price * order.filled_quantity)) / new_size if new_size > 0 else order.avg_fill_price
                    pos.side = PositionSide.SHORT
                    pos.size = new_size
                    pos.entry_price = new_entry
                    if not pos.opened_at:
                        pos.opened_at = datetime.now()
                elif pos.side == PositionSide.LONG:
                    # Long-Position reduzieren/schliessen
                    if order.filled_quantity >= pos.size:
                        pos.realized_pnl += (order.avg_fill_price - pos.entry_price) * pos.size
                        remaining = order.filled_quantity - pos.size
                        if remaining > 0:
                            pos.side = PositionSide.SHORT
                            pos.size = remaining
                            pos.entry_price = order.avg_fill_price
                            pos.opened_at = datetime.now()
                        else:
                            pos.side = PositionSide.NONE
                            pos.size = 0
                            pos.entry_price = 0
                            pos.opened_at = None
                    else:
                        pos.realized_pnl += (order.avg_fill_price - pos.entry_price) * order.filled_quantity
                        pos.size -= order.filled_quantity

        self._trigger_position_changed(pos)

    # === Getter ===

    def get_order(self, order_id: str) -> Optional[Order]:
        """Holt Order nach ID."""
        with self._lock:
            return self._orders.get(order_id)

    def get_active_orders(self) -> List[Order]:
        """Holt alle aktiven Orders."""
        with self._lock:
            return [o for o in self._orders.values() if o.is_active]

    def get_position(self) -> Position:
        """Holt aktuelle Position."""
        with self._lock:
            return self._position

    def get_order_history(self, limit: int = 100) -> List[Order]:
        """Holt Order-Historie."""
        with self._lock:
            return self._order_history[-limit:]

    # === Callbacks ===

    def on_order_filled(self, callback: Callable[[Order], None]) -> None:
        """Registriert Callback fuer ausgefuehrte Orders."""
        self._on_order_filled.append(callback)

    def on_order_canceled(self, callback: Callable[[Order], None]) -> None:
        """Registriert Callback fuer stornierte Orders."""
        self._on_order_canceled.append(callback)

    def on_position_changed(self, callback: Callable[[Position], None]) -> None:
        """Registriert Callback fuer Position-Aenderungen."""
        self._on_position_changed.append(callback)

    def _trigger_order_filled(self, order: Order) -> None:
        for callback in self._on_order_filled:
            try:
                callback(order)
            except Exception as e:
                self.logger.error(f'Callback-Fehler: {e}')

    def _trigger_order_canceled(self, order: Order) -> None:
        for callback in self._on_order_canceled:
            try:
                callback(order)
            except Exception as e:
                self.logger.error(f'Callback-Fehler: {e}')

    def _trigger_position_changed(self, position: Position) -> None:
        for callback in self._on_position_changed:
            try:
                callback(position)
            except Exception as e:
                self.logger.error(f'Callback-Fehler: {e}')

    # === Position-Management ===

    def close_position(self) -> Optional[Order]:
        """
        Schliesst die aktuelle Position.

        Returns:
            Order-Objekt oder None
        """
        pos = self.get_position()

        if not pos.is_open:
            self.logger.warning('Keine offene Position')
            return None

        # Gegenorder erstellen
        close_side = OrderSide.SELL if pos.side == PositionSide.LONG else OrderSide.BUY

        order = self.create_order(
            side=close_side,
            quantity=pos.size,
            order_type=OrderType.MARKET
        )

        if self.submit_order(order):
            self.logger.info(f'Position geschlossen: {pos.side.value} {pos.size}')
            return order

        return None

    def set_stop_loss(self, price: float) -> None:
        """Setzt Stop-Loss fuer aktuelle Position."""
        with self._lock:
            self._position.stop_loss = price
        self.logger.info(f'Stop-Loss gesetzt: {price}')

    def set_take_profit(self, price: float) -> None:
        """Setzt Take-Profit fuer aktuelle Position."""
        with self._lock:
            self._position.take_profit = price
        self.logger.info(f'Take-Profit gesetzt: {price}')

    def update_position_price(self, current_price: float) -> None:
        """Aktualisiert den aktuellen Preis der Position."""
        with self._lock:
            self._position.update_pnl(current_price)
