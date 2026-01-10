"""
WebServer Control Widget - GUI-Komponente fuer Server-Steuerung
"""

from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QGroupBox, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSpinBox, QCheckBox,
    QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QDesktopServices, QFont
from PyQt6.QtCore import QUrl

from ...web.server import StatusServer, ExtendedStatusServer, AppState
from ...core.logger import get_logger


class WebServerControl(QGroupBox):
    """
    Widget zur Steuerung des Web-Dashboard Servers.

    Features:
    - Start/Stop Server
    - Port-Konfiguration
    - URL-Anzeige (klickbar)
    - Status-Indikator
    - Verbundene Clients anzeigen

    Signals:
        server_started: Emittiert wenn Server startet
        server_stopped: Emittiert wenn Server stoppt
    """

    server_started = pyqtSignal()
    server_stopped = pyqtSignal()

    # Styling
    STYLE_INACTIVE = """
        QGroupBox {
            border: 2px solid #555;
            border-radius: 8px;
            margin-top: 10px;
            padding-top: 10px;
            background-color: #1e1e1e;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            color: #888;
        }
    """

    STYLE_ACTIVE = """
        QGroupBox {
            border: 2px solid #00aa44;
            border-radius: 8px;
            margin-top: 10px;
            padding-top: 10px;
            background-color: #1a2e1a;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            color: #00ff88;
        }
    """

    def __init__(
        self,
        app_state: Optional[AppState] = None,
        parent: Optional[QWidget] = None
    ):
        """
        Initialisiert das Widget.

        Args:
            app_state: AppState-Instanz fuer Server
            parent: Parent-Widget
        """
        super().__init__('Web-Dashboard', parent)
        self.logger = get_logger()
        self.app_state = app_state or AppState()
        self._server: Optional[StatusServer] = None

        self._setup_ui()
        self._connect_signals()
        self._update_style()

        # Status-Timer
        self._status_timer = QTimer()
        self._status_timer.timeout.connect(self._update_status)

    def _setup_ui(self) -> None:
        """Erstellt die UI-Komponenten."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Status-Zeile
        status_layout = QHBoxLayout()

        self.status_indicator = QLabel()
        self.status_indicator.setFixedSize(12, 12)
        self.status_indicator.setStyleSheet("""
            background-color: #555;
            border-radius: 6px;
        """)
        status_layout.addWidget(self.status_indicator)

        self.status_label = QLabel('Inaktiv')
        self.status_label.setStyleSheet('color: #888;')
        status_layout.addWidget(self.status_label)

        status_layout.addStretch()

        self.clients_label = QLabel('')
        self.clients_label.setStyleSheet('color: #4da8da; font-size: 11px;')
        status_layout.addWidget(self.clients_label)

        layout.addLayout(status_layout)

        # Trennlinie
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet('background-color: #333;')
        layout.addWidget(line)

        # URL-Anzeige
        self.url_label = QLabel('')
        self.url_label.setOpenExternalLinks(False)  # Manuell handhaben
        self.url_label.setStyleSheet("""
            color: #4da8da;
            font-size: 13px;
            padding: 5px;
        """)
        self.url_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.url_label.mousePressEvent = self._open_url
        layout.addWidget(self.url_label)

        # Port-Konfiguration
        port_layout = QHBoxLayout()

        port_label = QLabel('Port:')
        port_label.setStyleSheet('color: #aaa;')
        port_layout.addWidget(port_label)

        self.port_spinbox = QSpinBox()
        self.port_spinbox.setRange(1024, 65535)
        self.port_spinbox.setValue(5000)
        self.port_spinbox.setStyleSheet("""
            QSpinBox {
                background-color: #2a2a2a;
                color: #fff;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 4px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #3a3a3a;
            }
        """)
        port_layout.addWidget(self.port_spinbox)

        port_layout.addStretch()

        # Extended Server Option
        self.extended_checkbox = QCheckBox('Erweiterte API')
        self.extended_checkbox.setToolTip('Aktiviert zusaetzliche API-Endpunkte')
        self.extended_checkbox.setStyleSheet('color: #888;')
        port_layout.addWidget(self.extended_checkbox)

        layout.addLayout(port_layout)

        # Start/Stop Button
        self.toggle_button = QPushButton('Server starten')
        self.toggle_button.setCheckable(True)
        self.toggle_button.setMinimumHeight(40)
        self.toggle_button.setFont(QFont('Segoe UI', 10, QFont.Weight.Bold))
        self._update_button_style(False)
        layout.addWidget(self.toggle_button)

        # Info-Text
        self.info_label = QLabel('Zugriff von anderen Geraeten im Netzwerk')
        self.info_label.setStyleSheet('color: #666; font-size: 11px;')
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info_label)

    def _connect_signals(self) -> None:
        """Verbindet Signale."""
        self.toggle_button.clicked.connect(self._toggle_server)

    def _toggle_server(self, checked: bool) -> None:
        """Startet oder stoppt den Server."""
        if checked:
            self._start_server()
        else:
            self._stop_server()

    def _start_server(self) -> None:
        """Startet den Server."""
        port = self.port_spinbox.value()
        use_extended = self.extended_checkbox.isChecked()

        try:
            if use_extended:
                self._server = ExtendedStatusServer(
                    app_state=self.app_state,
                    port=port
                )
            else:
                self._server = StatusServer(
                    app_state=self.app_state,
                    port=port
                )

            if self._server.start():
                self.status_indicator.setStyleSheet("""
                    background-color: #00ff88;
                    border-radius: 6px;
                """)
                self.status_label.setText('Aktiv')
                self.status_label.setStyleSheet('color: #00ff88;')

                url = self._server.get_url()
                self.url_label.setText(f'<a href="{url}" style="color: #4da8da;">{url}</a>')
                self._current_url = url

                self.port_spinbox.setEnabled(False)
                self.extended_checkbox.setEnabled(False)
                self._update_button_style(True)
                self._update_style()

                # Status-Updates starten
                self._status_timer.start(5000)

                self.logger.success(f'Web-Dashboard gestartet: {url}')
                self.server_started.emit()
            else:
                self.toggle_button.setChecked(False)
                self.logger.error('Server-Start fehlgeschlagen')

        except Exception as e:
            self.toggle_button.setChecked(False)
            self.logger.error(f'Server-Fehler: {e}')

    def _stop_server(self) -> None:
        """Stoppt den Server."""
        if self._server:
            self._server.stop()
            self._server = None

        self._status_timer.stop()

        self.status_indicator.setStyleSheet("""
            background-color: #555;
            border-radius: 6px;
        """)
        self.status_label.setText('Inaktiv')
        self.status_label.setStyleSheet('color: #888;')
        self.url_label.setText('')
        self.clients_label.setText('')

        self.port_spinbox.setEnabled(True)
        self.extended_checkbox.setEnabled(True)
        self._update_button_style(False)
        self._update_style()

        self.logger.info('Web-Dashboard gestoppt')
        self.server_stopped.emit()

    def _update_status(self) -> None:
        """Aktualisiert Status-Anzeige."""
        if isinstance(self._server, ExtendedStatusServer):
            clients = self._server.get_connected_clients()
            if clients:
                self.clients_label.setText(f'{len(clients)} Client(s)')
            else:
                self.clients_label.setText('')

    def _update_button_style(self, active: bool) -> None:
        """Aktualisiert Button-Styling."""
        if active:
            self.toggle_button.setText('Server stoppen')
            self.toggle_button.setStyleSheet("""
                QPushButton {
                    background-color: #8b0000;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 10px;
                }
                QPushButton:hover {
                    background-color: #a00000;
                }
                QPushButton:pressed {
                    background-color: #700000;
                }
            """)
        else:
            self.toggle_button.setText('Server starten')
            self.toggle_button.setStyleSheet("""
                QPushButton {
                    background-color: #2d5a2d;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 10px;
                }
                QPushButton:hover {
                    background-color: #3d6a3d;
                }
                QPushButton:pressed {
                    background-color: #1d4a1d;
                }
            """)

    def _update_style(self) -> None:
        """Aktualisiert GroupBox-Styling."""
        if self._server and self._server.is_running:
            self.setStyleSheet(self.STYLE_ACTIVE)
        else:
            self.setStyleSheet(self.STYLE_INACTIVE)

    def _open_url(self, event) -> None:
        """Oeffnet URL im Browser."""
        if hasattr(self, '_current_url'):
            QDesktopServices.openUrl(QUrl(self._current_url))

    def get_server(self) -> Optional[StatusServer]:
        """Gibt aktuelle Server-Instanz zurueck."""
        return self._server

    def is_running(self) -> bool:
        """True wenn Server laeuft."""
        return self._server is not None and self._server.is_running

    def set_app_state(self, app_state: AppState) -> None:
        """Setzt neuen AppState."""
        self.app_state = app_state
        if self._server:
            self._server.app_state = app_state

    def shutdown(self) -> None:
        """Faehrt Server herunter (Cleanup)."""
        if self._server and self._server.is_running:
            self._stop_server()
