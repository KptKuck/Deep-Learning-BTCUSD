"""
Binance Client - API Wrapper mit Live/Testnet Umschaltung
"""

import os
from enum import Enum
from typing import Dict, List, Optional

from dotenv import load_dotenv

from ..core.logger import get_logger


class TradingMode(Enum):
    """Trading-Modus: Testnet (Demo) oder Live (Echtes Geld)."""
    TESTNET = "testnet"
    LIVE = "live"


class BinanceClient:
    """
    Binance API Client mit Live/Testnet Umschaltung.

    WICHTIG: TESTNET ist der Standard-Modus!
    Live-Modus erfordert explizite Aktivierung.

    Endpoints:
    - TESTNET: https://testnet.binance.vision
    - LIVE: https://api.binance.com

    Attributes:
        mode: Aktueller Trading-Modus
        symbol: Trading-Paar (z.B. 'BTCUSDT')
    """

    ENDPOINTS = {
        TradingMode.LIVE: {
            'rest': 'https://api.binance.com',
            'ws': 'wss://stream.binance.com:9443/ws'
        },
        TradingMode.TESTNET: {
            'rest': 'https://testnet.binance.vision',
            'ws': 'wss://testnet.binance.vision/ws'
        }
    }

    def __init__(
        self,
        mode: TradingMode = TradingMode.TESTNET,
        symbol: str = 'BTCUSDT'
    ):
        """
        Initialisiert den Binance Client.

        Args:
            mode: Trading-Modus (default: TESTNET!)
            symbol: Trading-Paar
        """
        self.mode = mode
        self.symbol = symbol.upper()
        self.logger = get_logger()

        self.api_key: str = ''
        self.api_secret: str = ''

        self._client = None
        self._ws_client = None

        # API-Keys laden
        self._load_api_keys()

        self.logger.info(f'BinanceClient initialisiert: {self.mode.value.upper()}')

    def _load_api_keys(self):
        """Laedt API-Keys basierend auf Modus."""
        load_dotenv()

        if self.mode == TradingMode.LIVE:
            self.api_key = os.getenv('BINANCE_LIVE_API_KEY', '')
            self.api_secret = os.getenv('BINANCE_LIVE_SECRET', '')
            if self.api_key:
                self.logger.warning('LIVE API-Keys geladen - ECHTES GELD!')
        else:
            self.api_key = os.getenv('BINANCE_TESTNET_API_KEY', '')
            self.api_secret = os.getenv('BINANCE_TESTNET_SECRET', '')
            if self.api_key:
                self.logger.info('Testnet API-Keys geladen')

    def _get_client(self):
        """Erstellt oder gibt Binance Client zurueck (lazy loading)."""
        if self._client is None:
            try:
                from binance.client import Client

                if self.mode == TradingMode.TESTNET:
                    self._client = Client(
                        self.api_key,
                        self.api_secret,
                        testnet=True
                    )
                else:
                    self._client = Client(
                        self.api_key,
                        self.api_secret
                    )

                self.logger.debug('Binance Client erstellt')

            except ImportError:
                self.logger.error('python-binance nicht installiert')
                raise

        return self._client

    @property
    def endpoint(self) -> str:
        """Aktueller REST-Endpoint."""
        return self.ENDPOINTS[self.mode]['rest']

    @property
    def ws_endpoint(self) -> str:
        """Aktueller WebSocket-Endpoint."""
        return self.ENDPOINTS[self.mode]['ws']

    @property
    def is_testnet(self) -> bool:
        """True wenn Testnet-Modus."""
        return self.mode == TradingMode.TESTNET

    @property
    def is_live(self) -> bool:
        """True wenn Live-Modus."""
        return self.mode == TradingMode.LIVE

    def switch_mode(self, mode: TradingMode) -> bool:
        """
        Wechselt den Trading-Modus.

        Args:
            mode: Neuer Modus

        Returns:
            True bei Erfolg
        """
        if mode == self.mode:
            return True

        old_mode = self.mode
        self.mode = mode

        # Client zuruecksetzen
        self._client = None
        self._ws_client = None

        # Neue Keys laden
        self._load_api_keys()

        self.logger.warning(f'Modus gewechselt: {old_mode.value} -> {mode.value}')

        return True

    # === Konto-Informationen ===

    def get_account(self) -> Optional[Dict]:
        """
        Holt Kontoinformationen.

        Returns:
            Account-Info oder None bei Fehler
        """
        try:
            client = self._get_client()
            return client.get_account()
        except Exception as e:
            self.logger.error(f'Fehler beim Abrufen der Kontodaten: {e}')
            return None

    def get_balance(self, asset: str = 'USDT') -> float:
        """
        Holt Balance fuer ein Asset.

        Args:
            asset: Asset-Symbol (z.B. 'USDT', 'BTC')

        Returns:
            Verfuegbare Balance
        """
        account = self.get_account()
        if account:
            for balance in account.get('balances', []):
                if balance['asset'] == asset:
                    return float(balance['free'])
        return 0.0

    def get_all_balances(self) -> Dict[str, float]:
        """
        Holt alle Balances (> 0).

        Returns:
            Dictionary {asset: balance}
        """
        account = self.get_account()
        balances = {}

        if account:
            for balance in account.get('balances', []):
                free = float(balance['free'])
                locked = float(balance['locked'])
                if free > 0 or locked > 0:
                    balances[balance['asset']] = free + locked

        return balances

    # === Marktdaten ===

    def get_ticker_price(self, symbol: Optional[str] = None) -> float:
        """
        Holt aktuellen Preis.

        Args:
            symbol: Trading-Paar (default: self.symbol)

        Returns:
            Aktueller Preis
        """
        try:
            client = self._get_client()
            ticker = client.get_symbol_ticker(symbol=symbol or self.symbol)
            return float(ticker['price'])
        except Exception as e:
            self.logger.error(f'Fehler beim Abrufen des Preises: {e}')
            return 0.0

    def get_orderbook(self, symbol: Optional[str] = None, limit: int = 10) -> Dict:
        """
        Holt Orderbuch.

        Args:
            symbol: Trading-Paar
            limit: Anzahl Eintraege

        Returns:
            Orderbuch mit bids und asks
        """
        try:
            client = self._get_client()
            return client.get_order_book(symbol=symbol or self.symbol, limit=limit)
        except Exception as e:
            self.logger.error(f'Fehler beim Abrufen des Orderbuchs: {e}')
            return {'bids': [], 'asks': []}

    # === Order-Management ===

    def create_market_order(
        self,
        side: str,
        quantity: float,
        symbol: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Erstellt eine Market-Order.

        Args:
            side: 'BUY' oder 'SELL'
            quantity: Menge
            symbol: Trading-Paar

        Returns:
            Order-Info oder None bei Fehler
        """
        try:
            client = self._get_client()

            order = client.create_order(
                symbol=symbol or self.symbol,
                side=side.upper(),
                type='MARKET',
                quantity=quantity
            )

            self.logger.success(f'Market Order: {side} {quantity} {symbol or self.symbol}')
            return order

        except Exception as e:
            self.logger.error(f'Order-Fehler: {e}')
            return None

    def create_limit_order(
        self,
        side: str,
        quantity: float,
        price: float,
        symbol: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Erstellt eine Limit-Order.

        Args:
            side: 'BUY' oder 'SELL'
            quantity: Menge
            price: Limit-Preis
            symbol: Trading-Paar

        Returns:
            Order-Info oder None bei Fehler
        """
        try:
            client = self._get_client()

            order = client.create_order(
                symbol=symbol or self.symbol,
                side=side.upper(),
                type='LIMIT',
                timeInForce='GTC',
                quantity=quantity,
                price=str(price)
            )

            self.logger.success(f'Limit Order: {side} {quantity} @ {price}')
            return order

        except Exception as e:
            self.logger.error(f'Order-Fehler: {e}')
            return None

    def cancel_order(self, order_id: int, symbol: Optional[str] = None) -> bool:
        """
        Storniert eine Order.

        Args:
            order_id: Order-ID
            symbol: Trading-Paar

        Returns:
            True bei Erfolg
        """
        try:
            client = self._get_client()
            client.cancel_order(symbol=symbol or self.symbol, orderId=order_id)
            self.logger.info(f'Order storniert: {order_id}')
            return True
        except Exception as e:
            self.logger.error(f'Stornierung fehlgeschlagen: {e}')
            return False

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Holt offene Orders.

        Args:
            symbol: Trading-Paar (None fuer alle)

        Returns:
            Liste offener Orders
        """
        try:
            client = self._get_client()
            if symbol:
                return client.get_open_orders(symbol=symbol)
            return client.get_open_orders()
        except Exception as e:
            self.logger.error(f'Fehler beim Abrufen offener Orders: {e}')
            return []

    def get_order_status(self, order_id: int, symbol: Optional[str] = None) -> Optional[Dict]:
        """
        Holt Order-Status.

        Args:
            order_id: Order-ID
            symbol: Trading-Paar

        Returns:
            Order-Info oder None
        """
        try:
            client = self._get_client()
            return client.get_order(symbol=symbol or self.symbol, orderId=order_id)
        except Exception as e:
            self.logger.error(f'Fehler beim Abrufen des Order-Status: {e}')
            return None

    # === Utility ===

    def test_connection(self) -> bool:
        """
        Testet die API-Verbindung.

        Returns:
            True wenn erfolgreich
        """
        try:
            client = self._get_client()
            client.ping()
            self.logger.success(f'Verbindung OK ({self.mode.value})')
            return True
        except Exception as e:
            self.logger.error(f'Verbindungstest fehlgeschlagen: {e}')
            return False

    def get_server_time(self) -> Optional[int]:
        """
        Holt Server-Zeit.

        Returns:
            Server-Zeit in Millisekunden
        """
        try:
            client = self._get_client()
            return client.get_server_time()['serverTime']
        except Exception as e:
            self.logger.error(f'Fehler beim Abrufen der Serverzeit: {e}')
            return None
