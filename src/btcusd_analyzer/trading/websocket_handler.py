"""
WebSocket Handler - Echtzeit-Daten von Binance
"""

import json
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Callable, Any
from enum import Enum
from threading import Thread, Event
import queue

from ..core.logger import get_logger
from .api_config import TradingMode


class StreamType(Enum):
    """WebSocket Stream-Typen."""
    TRADE = "trade"           # Einzelne Trades
    KLINE = "kline"           # Candlestick-Daten
    TICKER = "ticker"         # 24h Ticker
    MINI_TICKER = "miniTicker" # Mini Ticker
    DEPTH = "depth"           # Order Book
    AGG_TRADE = "aggTrade"    # Aggregierte Trades


@dataclass
class TickerData:
    """Ticker-Daten."""
    symbol: str
    price: float
    price_change: float
    price_change_pct: float
    high_24h: float
    low_24h: float
    volume_24h: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TradeData:
    """Trade-Daten."""
    symbol: str
    trade_id: int
    price: float
    quantity: float
    buyer_maker: bool
    timestamp: datetime


@dataclass
class KlineData:
    """Kline/Candlestick-Daten."""
    symbol: str
    interval: str
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: datetime
    is_closed: bool


class WebSocketHandler:
    """
    WebSocket-Handler fuer Binance Echtzeit-Daten.

    Features:
    - Automatische Reconnection
    - Multiple Streams
    - Thread-safe Callbacks
    - Testnet/Live Support

    Usage:
        ws = WebSocketHandler(mode=TradingMode.TESTNET)

        # Callbacks registrieren
        ws.on_ticker(lambda data: print(f"Preis: {data.price}"))
        ws.on_trade(lambda data: print(f"Trade: {data}"))

        # Streams starten
        ws.subscribe_ticker("BTCUSDT")
        ws.start()

        # Spaeter stoppen
        ws.stop()
    """

    WS_ENDPOINTS = {
        TradingMode.LIVE: "wss://stream.binance.com:9443/ws",
        TradingMode.TESTNET: "wss://testnet.binance.vision/ws"
    }

    def __init__(self, mode: TradingMode = TradingMode.TESTNET):
        """
        Initialisiert den WebSocket-Handler.

        Args:
            mode: Trading-Modus (TESTNET oder LIVE)
        """
        self.logger = get_logger()
        self.mode = mode
        self._endpoint = self.WS_ENDPOINTS[mode]

        # State
        self._running = False
        self._connected = False
        self._reconnect_count = 0
        self._max_reconnects = 10
        self._reconnect_delay = 5  # Sekunden

        # Streams
        self._subscriptions: List[str] = []
        self._stream_params: Dict[str, Any] = {}

        # Threading
        self._thread: Optional[Thread] = None
        self._stop_event = Event()
        self._message_queue: queue.Queue = queue.Queue()

        # Callbacks
        self._on_ticker: List[Callable[[TickerData], None]] = []
        self._on_trade: List[Callable[[TradeData], None]] = []
        self._on_kline: List[Callable[[KlineData], None]] = []
        self._on_connected: List[Callable[[], None]] = []
        self._on_disconnected: List[Callable[[str], None]] = []
        self._on_error: List[Callable[[str], None]] = []

        # WebSocket
        self._ws = None

        self.logger.info(f'WebSocketHandler initialisiert ({mode.value})')

    def subscribe_ticker(self, symbol: str = "BTCUSDT") -> None:
        """
        Abonniert Ticker-Stream.

        Args:
            symbol: Trading-Paar
        """
        stream = f"{symbol.lower()}@ticker"
        if stream not in self._subscriptions:
            self._subscriptions.append(stream)
            self._stream_params[stream] = {'type': StreamType.TICKER, 'symbol': symbol}
            self.logger.debug(f'Ticker-Stream abonniert: {symbol}')

    def subscribe_trade(self, symbol: str = "BTCUSDT") -> None:
        """
        Abonniert Trade-Stream.

        Args:
            symbol: Trading-Paar
        """
        stream = f"{symbol.lower()}@trade"
        if stream not in self._subscriptions:
            self._subscriptions.append(stream)
            self._stream_params[stream] = {'type': StreamType.TRADE, 'symbol': symbol}
            self.logger.debug(f'Trade-Stream abonniert: {symbol}')

    def subscribe_kline(self, symbol: str = "BTCUSDT", interval: str = "1m") -> None:
        """
        Abonniert Kline/Candlestick-Stream.

        Args:
            symbol: Trading-Paar
            interval: Zeitintervall (1m, 5m, 15m, 1h, etc.)
        """
        stream = f"{symbol.lower()}@kline_{interval}"
        if stream not in self._subscriptions:
            self._subscriptions.append(stream)
            self._stream_params[stream] = {
                'type': StreamType.KLINE,
                'symbol': symbol,
                'interval': interval
            }
            self.logger.debug(f'Kline-Stream abonniert: {symbol} @ {interval}')

    def subscribe_depth(self, symbol: str = "BTCUSDT", levels: int = 10) -> None:
        """
        Abonniert Order Book Stream.

        Args:
            symbol: Trading-Paar
            levels: Anzahl Level (5, 10, 20)
        """
        stream = f"{symbol.lower()}@depth{levels}@100ms"
        if stream not in self._subscriptions:
            self._subscriptions.append(stream)
            self._stream_params[stream] = {
                'type': StreamType.DEPTH,
                'symbol': symbol,
                'levels': levels
            }
            self.logger.debug(f'Depth-Stream abonniert: {symbol} ({levels} levels)')

    def unsubscribe(self, stream: str) -> None:
        """
        Entfernt Stream-Abonnement.

        Args:
            stream: Stream-Name
        """
        if stream in self._subscriptions:
            self._subscriptions.remove(stream)
            self._stream_params.pop(stream, None)

    def start(self) -> bool:
        """
        Startet WebSocket-Verbindung.

        Returns:
            True wenn erfolgreich gestartet
        """
        if self._running:
            self.logger.warning('WebSocket laeuft bereits')
            return False

        if not self._subscriptions:
            self.logger.error('Keine Streams abonniert')
            return False

        self._stop_event.clear()
        self._running = True
        self._reconnect_count = 0

        self._thread = Thread(target=self._run_async_loop, daemon=True)
        self._thread.start()

        self.logger.info('WebSocket-Handler gestartet')
        return True

    def stop(self) -> None:
        """Stoppt WebSocket-Verbindung."""
        if not self._running:
            return

        self._stop_event.set()
        self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

        self.logger.info('WebSocket-Handler gestoppt')

    def _run_async_loop(self) -> None:
        """Fuehrt asyncio Event-Loop aus."""
        try:
            asyncio.run(self._connect_and_listen())
        except Exception as e:
            self.logger.error(f'Event-Loop Fehler: {e}')
            self._trigger_error(str(e))

    async def _connect_and_listen(self) -> None:
        """Verbindet und empfaengt Nachrichten."""
        try:
            import websockets
        except ImportError:
            self.logger.error('websockets nicht installiert: pip install websockets')
            self._trigger_error('websockets Paket fehlt')
            return

        while self._running and not self._stop_event.is_set():
            try:
                # Multi-Stream URL
                streams = '/'.join(self._subscriptions)
                url = f"{self._endpoint}/{streams}"

                self.logger.debug(f'Verbinde zu: {url}')

                async with websockets.connect(url) as ws:
                    self._ws = ws
                    self._connected = True
                    self._reconnect_count = 0

                    self.logger.success('WebSocket verbunden')
                    self._trigger_connected()

                    # Nachrichten empfangen
                    async for message in ws:
                        if self._stop_event.is_set():
                            break
                        self._handle_message(message)

            except Exception as e:
                self._connected = False
                self.logger.warning(f'WebSocket-Fehler: {e}')
                self._trigger_disconnected(str(e))

                # Reconnect
                if self._running and not self._stop_event.is_set():
                    self._reconnect_count += 1
                    if self._reconnect_count > self._max_reconnects:
                        self.logger.error('Max Reconnects erreicht')
                        self._trigger_error('Max Reconnects erreicht')
                        break

                    self.logger.info(f'Reconnect in {self._reconnect_delay}s ({self._reconnect_count}/{self._max_reconnects})')
                    await asyncio.sleep(self._reconnect_delay)

        self._connected = False

    def _handle_message(self, message: str) -> None:
        """
        Verarbeitet empfangene Nachricht.

        Args:
            message: JSON-Nachricht
        """
        try:
            data = json.loads(message)

            # Event-Typ ermitteln
            event_type = data.get('e')

            if event_type == '24hrTicker':
                self._handle_ticker(data)
            elif event_type == 'trade':
                self._handle_trade(data)
            elif event_type == 'kline':
                self._handle_kline(data)
            elif 'lastUpdateId' in data:
                # Order Book (depth) hat kein 'e' Feld
                self._handle_depth(data)
            else:
                self.logger.debug(f'Unbekannter Event-Typ: {event_type}')

        except json.JSONDecodeError as e:
            self.logger.error(f'JSON Parse-Fehler: {e}')
        except Exception as e:
            self.logger.error(f'Message-Handler Fehler: {e}')

    def _handle_ticker(self, data: dict) -> None:
        """Verarbeitet Ticker-Daten."""
        ticker = TickerData(
            symbol=data.get('s', ''),
            price=float(data.get('c', 0)),
            price_change=float(data.get('p', 0)),
            price_change_pct=float(data.get('P', 0)),
            high_24h=float(data.get('h', 0)),
            low_24h=float(data.get('l', 0)),
            volume_24h=float(data.get('v', 0)),
            timestamp=datetime.fromtimestamp(data.get('E', 0) / 1000)
        )

        for callback in self._on_ticker:
            try:
                callback(ticker)
            except Exception as e:
                self.logger.error(f'Ticker-Callback Fehler: {e}')

    def _handle_trade(self, data: dict) -> None:
        """Verarbeitet Trade-Daten."""
        trade = TradeData(
            symbol=data.get('s', ''),
            trade_id=data.get('t', 0),
            price=float(data.get('p', 0)),
            quantity=float(data.get('q', 0)),
            buyer_maker=data.get('m', False),
            timestamp=datetime.fromtimestamp(data.get('T', 0) / 1000)
        )

        for callback in self._on_trade:
            try:
                callback(trade)
            except Exception as e:
                self.logger.error(f'Trade-Callback Fehler: {e}')

    def _handle_kline(self, data: dict) -> None:
        """Verarbeitet Kline-Daten."""
        k = data.get('k', {})

        kline = KlineData(
            symbol=data.get('s', ''),
            interval=k.get('i', ''),
            open_time=datetime.fromtimestamp(k.get('t', 0) / 1000),
            open=float(k.get('o', 0)),
            high=float(k.get('h', 0)),
            low=float(k.get('l', 0)),
            close=float(k.get('c', 0)),
            volume=float(k.get('v', 0)),
            close_time=datetime.fromtimestamp(k.get('T', 0) / 1000),
            is_closed=k.get('x', False)
        )

        for callback in self._on_kline:
            try:
                callback(kline)
            except Exception as e:
                self.logger.error(f'Kline-Callback Fehler: {e}')

    def _handle_depth(self, data: dict) -> None:
        """Verarbeitet Order Book Daten (placeholder)."""
        # Order Book Verarbeitung kann spaeter erweitert werden
        pass

    # === Callback Registration ===

    def on_ticker(self, callback: Callable[[TickerData], None]) -> None:
        """Registriert Ticker-Callback."""
        self._on_ticker.append(callback)

    def on_trade(self, callback: Callable[[TradeData], None]) -> None:
        """Registriert Trade-Callback."""
        self._on_trade.append(callback)

    def on_kline(self, callback: Callable[[KlineData], None]) -> None:
        """Registriert Kline-Callback."""
        self._on_kline.append(callback)

    def on_connected(self, callback: Callable[[], None]) -> None:
        """Registriert Verbindungs-Callback."""
        self._on_connected.append(callback)

    def on_disconnected(self, callback: Callable[[str], None]) -> None:
        """Registriert Disconnect-Callback."""
        self._on_disconnected.append(callback)

    def on_error(self, callback: Callable[[str], None]) -> None:
        """Registriert Error-Callback."""
        self._on_error.append(callback)

    def _trigger_connected(self) -> None:
        for callback in self._on_connected:
            try:
                callback()
            except Exception as e:
                self.logger.error(f'Connected-Callback Fehler: {e}')

    def _trigger_disconnected(self, reason: str) -> None:
        for callback in self._on_disconnected:
            try:
                callback(reason)
            except Exception as e:
                self.logger.error(f'Disconnected-Callback Fehler: {e}')

    def _trigger_error(self, message: str) -> None:
        for callback in self._on_error:
            try:
                callback(message)
            except Exception as e:
                self.logger.error(f'Error-Callback Fehler: {e}')

    # === Properties ===

    @property
    def is_connected(self) -> bool:
        """Prueft ob verbunden."""
        return self._connected

    @property
    def is_running(self) -> bool:
        """Prueft ob Handler laeuft."""
        return self._running

    @property
    def subscriptions(self) -> List[str]:
        """Gibt aktive Subscriptions zurueck."""
        return self._subscriptions.copy()

    def get_status(self) -> Dict[str, Any]:
        """Gibt Status-Informationen zurueck."""
        return {
            'mode': self.mode.value,
            'connected': self._connected,
            'running': self._running,
            'endpoint': self._endpoint,
            'subscriptions': self._subscriptions,
            'reconnect_count': self._reconnect_count
        }


class WebSocketManager:
    """
    Manager fuer mehrere WebSocket-Verbindungen.

    Verwaltet separate Handler fuer verschiedene Symbole oder Zwecke.
    """

    def __init__(self, mode: TradingMode = TradingMode.TESTNET):
        """
        Initialisiert den Manager.

        Args:
            mode: Trading-Modus
        """
        self.logger = get_logger()
        self.mode = mode
        self._handlers: Dict[str, WebSocketHandler] = {}

    def create_handler(self, name: str) -> WebSocketHandler:
        """
        Erstellt neuen WebSocket-Handler.

        Args:
            name: Handler-Name

        Returns:
            Neuer WebSocketHandler
        """
        if name in self._handlers:
            raise ValueError(f'Handler "{name}" existiert bereits')

        handler = WebSocketHandler(self.mode)
        self._handlers[name] = handler
        return handler

    def get_handler(self, name: str) -> Optional[WebSocketHandler]:
        """Gibt Handler nach Name zurueck."""
        return self._handlers.get(name)

    def start_all(self) -> None:
        """Startet alle Handler."""
        for name, handler in self._handlers.items():
            if not handler.is_running:
                handler.start()
                self.logger.debug(f'Handler gestartet: {name}')

    def stop_all(self) -> None:
        """Stoppt alle Handler."""
        for name, handler in self._handlers.items():
            if handler.is_running:
                handler.stop()
                self.logger.debug(f'Handler gestoppt: {name}')

    def remove_handler(self, name: str) -> None:
        """Entfernt Handler."""
        if name in self._handlers:
            handler = self._handlers[name]
            if handler.is_running:
                handler.stop()
            del self._handlers[name]

    def get_all_status(self) -> Dict[str, Dict]:
        """Gibt Status aller Handler zurueck."""
        return {
            name: handler.get_status()
            for name, handler in self._handlers.items()
        }
