"""
Web Server - Status-Dashboard fuer LAN-Zugriff
"""

import socket
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Thread, Lock
from typing import Any, Dict, Optional, List, Callable

from flask import Flask, jsonify, render_template_string, request

from ..core.logger import get_logger


@dataclass
class AppState:
    """
    Repraesentiert den aktuellen Zustand der Anwendung.

    Wird vom StatusServer verwendet um den Status anzuzeigen.
    """
    # Allgemein
    start_time: datetime = field(default_factory=datetime.now)

    # Daten
    data_loaded: bool = False
    data_count: int = 0
    date_range: str = '-'

    # Modell
    model_loaded: bool = False
    model_name: str = '-'
    model_accuracy: float = 0.0

    # Training
    training_active: bool = False
    current_epoch: int = 0
    total_epochs: int = 0
    current_loss: float = 0.0

    # Backtest
    backtest_active: bool = False
    backtest_progress: float = 0.0
    backtest_pnl: float = 0.0

    # Trading
    trading_mode: str = 'OFF'  # 'OFF', 'TESTNET', 'LIVE'
    trading_active: bool = False
    current_position: str = 'NONE'
    trading_pnl: float = 0.0

    def get_uptime(self) -> str:
        """Gibt die Laufzeit zurueck."""
        delta = datetime.now() - self.start_time
        hours, remainder = divmod(int(delta.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


# HTML Template fuer das Dashboard
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>BTCUSD Analyzer - Status</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="{{ refresh_seconds }}">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 20px;
            min-height: 100vh;
        }
        .header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #16213e, #1a1a2e);
            border-radius: 10px;
            margin-bottom: 20px;
            border: 1px solid #0f3460;
        }
        .header h1 {
            color: #4da8da;
            margin-bottom: 10px;
        }
        .header p {
            color: #888;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .card {
            background: #16213e;
            border-radius: 10px;
            padding: 20px;
            border-left: 4px solid #0f3460;
            transition: transform 0.2s;
        }
        .card:hover {
            transform: translateY(-2px);
        }
        .card.active { border-left-color: #00ff88; }
        .card.warning { border-left-color: #ffaa00; }
        .card.danger { border-left-color: #ff4444; }
        .card h3 {
            margin-top: 0;
            margin-bottom: 15px;
            color: #4da8da;
            font-size: 1.1em;
        }
        .status-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #0f3460;
        }
        .status-row:last-child {
            border-bottom: none;
        }
        .status-label {
            color: #888;
        }
        .status-value {
            font-weight: bold;
            color: #fff;
        }
        .badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }
        .badge-success { background: #00ff88; color: #000; }
        .badge-warning { background: #ffaa00; color: #000; }
        .badge-danger { background: #ff4444; color: #fff; }
        .badge-off { background: #555; color: #fff; }

        .progress-bar {
            background: #0f3460;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
            margin-top: 10px;
        }
        .progress-bar-fill {
            background: linear-gradient(90deg, #4da8da, #00ff88);
            height: 100%;
            transition: width 0.5s;
        }

        .footer {
            text-align: center;
            margin-top: 20px;
            color: #555;
            font-size: 12px;
        }

        @media (max-width: 600px) {
            body { padding: 10px; }
            .header { padding: 15px; }
            .header h1 { font-size: 1.5em; }
            .card { padding: 15px; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>BTCUSD Analyzer</h1>
        <p>Status-Dashboard | {{ status.server.ip }}:{{ status.server.port }} | Uptime: {{ status.server.uptime }}</p>
    </div>

    <div class="grid">
        <!-- Daten Status -->
        <div class="card {% if status.data.loaded %}active{% endif %}">
            <h3>Daten</h3>
            <div class="status-row">
                <span class="status-label">Status</span>
                <span class="badge {% if status.data.loaded %}badge-success{% else %}badge-off{% endif %}">
                    {{ 'Geladen' if status.data.loaded else 'Nicht geladen' }}
                </span>
            </div>
            <div class="status-row">
                <span class="status-label">Datensaetze</span>
                <span class="status-value">{{ '{:,}'.format(status.data.records) }}</span>
            </div>
            <div class="status-row">
                <span class="status-label">Zeitraum</span>
                <span class="status-value">{{ status.data.date_range }}</span>
            </div>
        </div>

        <!-- Modell Status -->
        <div class="card {% if status.model.loaded %}active{% endif %}">
            <h3>Modell</h3>
            <div class="status-row">
                <span class="status-label">Status</span>
                <span class="badge {% if status.model.loaded %}badge-success{% else %}badge-off{% endif %}">
                    {{ 'Geladen' if status.model.loaded else 'Nicht geladen' }}
                </span>
            </div>
            <div class="status-row">
                <span class="status-label">Name</span>
                <span class="status-value">{{ status.model.name }}</span>
            </div>
            <div class="status-row">
                <span class="status-label">Accuracy</span>
                <span class="status-value">{{ '%.1f' % status.model.accuracy }}%</span>
            </div>
        </div>

        <!-- Training Status -->
        <div class="card {% if status.training.active %}warning{% endif %}">
            <h3>Training</h3>
            <div class="status-row">
                <span class="status-label">Status</span>
                <span class="badge {% if status.training.active %}badge-warning{% else %}badge-off{% endif %}">
                    {{ 'Laeuft' if status.training.active else 'Inaktiv' }}
                </span>
            </div>
            <div class="status-row">
                <span class="status-label">Epoch</span>
                <span class="status-value">{{ status.training.epoch }} / {{ status.training.total_epochs }}</span>
            </div>
            <div class="status-row">
                <span class="status-label">Loss</span>
                <span class="status-value">{{ '%.4f' % status.training.loss if status.training.loss else '-' }}</span>
            </div>
            {% if status.training.active and status.training.total_epochs > 0 %}
            <div class="progress-bar">
                <div class="progress-bar-fill" style="width: {{ (status.training.epoch / status.training.total_epochs * 100)|int }}%"></div>
            </div>
            {% endif %}
        </div>

        <!-- Backtest Status -->
        <div class="card {% if status.backtest.active %}warning{% endif %}">
            <h3>Backtest</h3>
            <div class="status-row">
                <span class="status-label">Status</span>
                <span class="badge {% if status.backtest.active %}badge-warning{% else %}badge-off{% endif %}">
                    {{ 'Laeuft' if status.backtest.active else 'Inaktiv' }}
                </span>
            </div>
            <div class="status-row">
                <span class="status-label">Fortschritt</span>
                <span class="status-value">{{ '%.0f' % status.backtest.progress }}%</span>
            </div>
            <div class="status-row">
                <span class="status-label">P/L</span>
                <span class="status-value" style="color: {{ '#00ff88' if status.backtest.pnl >= 0 else '#ff4444' }}">
                    ${{ '%.2f' % status.backtest.pnl }}
                </span>
            </div>
            {% if status.backtest.active %}
            <div class="progress-bar">
                <div class="progress-bar-fill" style="width: {{ status.backtest.progress|int }}%"></div>
            </div>
            {% endif %}
        </div>

        <!-- Trading Status -->
        <div class="card {% if status.trading.active %}{% if status.trading.mode == 'LIVE' %}danger{% else %}active{% endif %}{% endif %}">
            <h3>Live-Trading</h3>
            <div class="status-row">
                <span class="status-label">Modus</span>
                <span class="badge {% if status.trading.mode == 'LIVE' %}badge-danger{% elif status.trading.mode == 'TESTNET' %}badge-success{% else %}badge-off{% endif %}">
                    {{ status.trading.mode }}
                </span>
            </div>
            <div class="status-row">
                <span class="status-label">Status</span>
                <span class="badge {% if status.trading.active %}badge-warning{% else %}badge-off{% endif %}">
                    {{ 'Aktiv' if status.trading.active else 'Inaktiv' }}
                </span>
            </div>
            <div class="status-row">
                <span class="status-label">Position</span>
                <span class="status-value">{{ status.trading.position }}</span>
            </div>
            <div class="status-row">
                <span class="status-label">P/L</span>
                <span class="status-value" style="color: {{ '#00ff88' if status.trading.pnl >= 0 else '#ff4444' }}">
                    ${{ '%.2f' % status.trading.pnl }}
                </span>
            </div>
        </div>
    </div>

    <div class="footer">
        <p>BTCUSD Analyzer v0.1.0 | Auto-Refresh: {{ refresh_seconds }}s | {{ now }}</p>
    </div>
</body>
</html>
"""


class StatusServer:
    """
    Web-Server fuer LAN-weites Status-Dashboard.

    Features:
    - Start/Stop Kontrolle
    - Status-API (JSON)
    - HTML-Dashboard mit Auto-Refresh
    - Responsive Design (Desktop + Mobile)

    Attributes:
        app_state: Referenz auf Anwendungszustand
        host: Server-Host (0.0.0.0 fuer alle Interfaces)
        port: Server-Port
    """

    def __init__(
        self,
        app_state: Optional[AppState] = None,
        host: str = '0.0.0.0',
        port: int = 5000
    ):
        """
        Initialisiert den Status-Server.

        Args:
            app_state: AppState-Instanz (oder None fuer eigene)
            host: Server-Host
            port: Server-Port
        """
        self.app_state = app_state or AppState()
        self.host = host
        self.port = port

        self._app = Flask(__name__)
        self._server = None
        self._thread = None
        self._running = False

        self._setup_routes()

    def _setup_routes(self):
        """Richtet die Server-Routen ein."""

        @self._app.route('/')
        def index():
            """Haupt-Dashboard."""
            return render_template_string(
                DASHBOARD_TEMPLATE,
                status=self._get_status(),
                refresh_seconds=5,
                now=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )

        @self._app.route('/api/status')
        def api_status():
            """JSON API fuer Status."""
            return jsonify(self._get_status_dict())

        @self._app.route('/api/health')
        def api_health():
            """Health-Check Endpoint."""
            return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})

    def _get_status(self) -> Dict[str, Any]:
        """Sammelt Status als verschachteltes Objekt."""
        return {
            'server': {
                'ip': self._get_local_ip(),
                'port': self.port,
                'uptime': self.app_state.get_uptime()
            },
            'data': {
                'loaded': self.app_state.data_loaded,
                'records': self.app_state.data_count,
                'date_range': self.app_state.date_range
            },
            'model': {
                'loaded': self.app_state.model_loaded,
                'name': self.app_state.model_name,
                'accuracy': self.app_state.model_accuracy
            },
            'training': {
                'active': self.app_state.training_active,
                'epoch': self.app_state.current_epoch,
                'total_epochs': self.app_state.total_epochs,
                'loss': self.app_state.current_loss
            },
            'backtest': {
                'active': self.app_state.backtest_active,
                'progress': self.app_state.backtest_progress,
                'pnl': self.app_state.backtest_pnl
            },
            'trading': {
                'mode': self.app_state.trading_mode,
                'active': self.app_state.trading_active,
                'position': self.app_state.current_position,
                'pnl': self.app_state.trading_pnl
            }
        }

    def _get_status_dict(self) -> dict:
        """Flache Version des Status fuer API."""
        s = self.app_state
        return {
            'server_ip': self._get_local_ip(),
            'server_port': self.port,
            'uptime': s.get_uptime(),
            'data_loaded': s.data_loaded,
            'data_count': s.data_count,
            'data_range': s.date_range,
            'model_loaded': s.model_loaded,
            'model_name': s.model_name,
            'model_accuracy': s.model_accuracy,
            'training_active': s.training_active,
            'training_epoch': s.current_epoch,
            'training_total_epochs': s.total_epochs,
            'training_loss': s.current_loss,
            'backtest_active': s.backtest_active,
            'backtest_progress': s.backtest_progress,
            'backtest_pnl': s.backtest_pnl,
            'trading_mode': s.trading_mode,
            'trading_active': s.trading_active,
            'trading_position': s.current_position,
            'trading_pnl': s.trading_pnl
        }

    def _get_local_ip(self) -> str:
        """Ermittelt lokale IP-Adresse fuer LAN-Zugriff."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return '127.0.0.1'

    @property
    def is_running(self) -> bool:
        """True wenn Server laeuft."""
        return self._running

    def start(self) -> bool:
        """
        Startet den Server.

        Returns:
            True bei Erfolg, False wenn bereits laeuft
        """
        if self._running:
            return False

        try:
            from werkzeug.serving import make_server
            self._server = make_server(self.host, self.port, self._app, threaded=True)

            self._thread = Thread(target=self._server.serve_forever)
            self._thread.daemon = True
            self._thread.start()
            self._running = True

            print(f'Web-Dashboard gestartet: http://{self._get_local_ip()}:{self.port}')
            return True

        except Exception as e:
            print(f'Server-Start fehlgeschlagen: {e}')
            return False

    def stop(self) -> bool:
        """
        Stoppt den Server.

        Returns:
            True bei Erfolg, False wenn nicht laeuft
        """
        if not self._running:
            return False

        try:
            self._server.shutdown()
            self._running = False
            print('Web-Dashboard gestoppt')
            return True
        except Exception as e:
            print(f'Server-Stop fehlgeschlagen: {e}')
            return False

    def get_url(self) -> str:
        """Gibt die LAN-URL zurueck."""
        return f'http://{self._get_local_ip()}:{self.port}'

    def get_local_ip(self) -> str:
        """Oeffentliche Methode fuer IP-Abfrage."""
        return self._get_local_ip()


class ExtendedStatusServer(StatusServer):
    """
    Erweiterter Status-Server mit zusaetzlichen Features.

    Zusaetzliche Features:
    - WebSocket-Verbindungsstatus
    - Trading-Historie
    - Performance-Metriken
    - Konfigurierbare Refresh-Rate
    - Event-Callbacks
    - API fuer Steuerung (optional)

    Usage:
        server = ExtendedStatusServer(app_state)
        server.on_client_connected(lambda ip: print(f'Client: {ip}'))
        server.start()
    """

    def __init__(
        self,
        app_state: Optional[AppState] = None,
        host: str = '0.0.0.0',
        port: int = 5000,
        refresh_seconds: int = 5,
        enable_control_api: bool = False
    ):
        """
        Initialisiert den erweiterten Server.

        Args:
            app_state: AppState-Instanz
            host: Server-Host
            port: Server-Port
            refresh_seconds: Auto-Refresh Intervall
            enable_control_api: Steuerungs-Endpunkte aktivieren
        """
        super().__init__(app_state, host, port)
        self.logger = get_logger()
        self.refresh_seconds = refresh_seconds
        self.enable_control_api = enable_control_api

        self._lock = Lock()
        self._connected_clients: List[str] = []
        self._request_count = 0

        # Callbacks
        self._on_client_connected: List[Callable[[str], None]] = []
        self._on_command_received: List[Callable[[str, Dict], None]] = []

        # Erweiterte Routen
        self._setup_extended_routes()

    def _setup_extended_routes(self):
        """Richtet erweiterte Routen ein."""

        @self._app.route('/api/metrics')
        def api_metrics():
            """Performance-Metriken."""
            with self._lock:
                self._request_count += 1

            return jsonify({
                'request_count': self._request_count,
                'connected_clients': len(self._connected_clients),
                'uptime': self.app_state.get_uptime(),
                'refresh_interval': self.refresh_seconds
            })

        @self._app.route('/api/trading/history')
        def api_trading_history():
            """Trading-Historie (Platzhalter)."""
            return jsonify({
                'trades': [],
                'total_pnl': self.app_state.trading_pnl,
                'message': 'Historie noch nicht implementiert'
            })

        if self.enable_control_api:
            @self._app.route('/api/control/stop_trading', methods=['POST'])
            def control_stop_trading():
                """Stoppt Trading (erfordert enable_control_api)."""
                self._trigger_command('stop_trading', request.json or {})
                return jsonify({'status': 'command_sent', 'command': 'stop_trading'})

            @self._app.route('/api/control/emergency_stop', methods=['POST'])
            def control_emergency_stop():
                """Notfall-Stop."""
                self._trigger_command('emergency_stop', request.json or {})
                return jsonify({'status': 'command_sent', 'command': 'emergency_stop'})

        @self._app.before_request
        def track_client():
            """Trackt verbundene Clients."""
            client_ip = request.remote_addr
            with self._lock:
                if client_ip not in self._connected_clients:
                    self._connected_clients.append(client_ip)
                    self._trigger_client_connected(client_ip)

        # Ueberschreibe index fuer konfigurierbaren Refresh
        @self._app.route('/')
        def index_extended():
            """Haupt-Dashboard mit konfigurierbarlem Refresh."""
            with self._lock:
                self._request_count += 1

            return render_template_string(
                DASHBOARD_TEMPLATE,
                status=self._get_status(),
                refresh_seconds=self.refresh_seconds,
                now=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )

    # === Callbacks ===

    def on_client_connected(self, callback: Callable[[str], None]) -> None:
        """Registriert Callback fuer neue Clients."""
        self._on_client_connected.append(callback)

    def on_command_received(self, callback: Callable[[str, Dict], None]) -> None:
        """Registriert Callback fuer Steuerkommandos."""
        self._on_command_received.append(callback)

    def _trigger_client_connected(self, client_ip: str) -> None:
        self.logger.debug(f'Neuer Client verbunden: {client_ip}')
        for callback in self._on_client_connected:
            try:
                callback(client_ip)
            except Exception as e:
                self.logger.error(f'Client-Callback Fehler: {e}')

    def _trigger_command(self, command: str, params: Dict) -> None:
        self.logger.info(f'Kommando empfangen: {command}')
        for callback in self._on_command_received:
            try:
                callback(command, params)
            except Exception as e:
                self.logger.error(f'Command-Callback Fehler: {e}')

    def set_refresh_seconds(self, seconds: int) -> None:
        """Setzt das Refresh-Intervall."""
        self.refresh_seconds = max(1, min(60, seconds))

    def get_connected_clients(self) -> List[str]:
        """Gibt Liste verbundener Client-IPs zurueck."""
        with self._lock:
            return self._connected_clients.copy()

    def get_request_count(self) -> int:
        """Gibt Anzahl der Anfragen zurueck."""
        return self._request_count
