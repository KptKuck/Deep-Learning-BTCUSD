"""
API Config - Sichere Verwaltung von API-Schlüsseln
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from dotenv import load_dotenv, set_key

from ..core.logger import get_logger


class TradingMode(Enum):
    """Trading-Modi."""
    TESTNET = "testnet"
    LIVE = "live"


@dataclass
class APICredentials:
    """API-Schluessel fuer einen Modus."""
    api_key: str = ""
    api_secret: str = ""
    is_configured: bool = False

    def __post_init__(self):
        self.is_configured = bool(self.api_key and self.api_secret)

    def validate(self) -> bool:
        """Prueft ob Credentials gueltig sind."""
        if not self.api_key or not self.api_secret:
            return False
        # Basis-Validierung der Laenge
        if len(self.api_key) < 20 or len(self.api_secret) < 20:
            return False
        return True

    def mask(self) -> str:
        """Gibt maskierte Version des API-Keys zurueck."""
        if not self.api_key:
            return "Nicht konfiguriert"
        return f"{self.api_key[:8]}...{self.api_key[-4:]}"


class APIConfig:
    """
    Zentrale Verwaltung von API-Schlüsseln.

    Features:
    - Laden aus .env Datei
    - Sichere Speicherung
    - Validierung
    - Maskierung fuer Anzeige

    Umgebungsvariablen:
    - BINANCE_TESTNET_API_KEY
    - BINANCE_TESTNET_SECRET
    - BINANCE_LIVE_API_KEY
    - BINANCE_LIVE_SECRET

    Usage:
        config = APIConfig()
        config.load()

        # Credentials holen
        testnet = config.get_credentials(TradingMode.TESTNET)
        live = config.get_credentials(TradingMode.LIVE)

        # Neue Keys setzen
        config.set_credentials(TradingMode.TESTNET, api_key, api_secret)
        config.save()
    """

    ENV_KEYS = {
        TradingMode.TESTNET: {
            'api_key': 'BINANCE_TESTNET_API_KEY',
            'api_secret': 'BINANCE_TESTNET_SECRET'
        },
        TradingMode.LIVE: {
            'api_key': 'BINANCE_LIVE_API_KEY',
            'api_secret': 'BINANCE_LIVE_SECRET'
        }
    }

    def __init__(self, env_path: Optional[Path] = None):
        """
        Initialisiert die API-Konfiguration.

        Args:
            env_path: Pfad zur .env Datei (default: Projektroot)
        """
        self.logger = get_logger()
        self.env_path = env_path or self._find_env_file()

        self._credentials: Dict[TradingMode, APICredentials] = {
            TradingMode.TESTNET: APICredentials(),
            TradingMode.LIVE: APICredentials()
        }

        self._loaded = False

    def _find_env_file(self) -> Path:
        """Sucht die .env Datei."""
        # Suche in verschiedenen Verzeichnissen
        search_paths = [
            Path.cwd() / '.env',
            Path.cwd().parent / '.env',
            Path(__file__).parent.parent.parent.parent.parent / '.env',
            Path.home() / '.btcusd_analyzer' / '.env'
        ]

        for path in search_paths:
            if path.exists():
                return path

        # Default: Erstelle im Projektverzeichnis
        return Path.cwd() / '.env'

    def load(self) -> bool:
        """
        Laedt API-Schluessel aus .env Datei.

        Returns:
            True wenn erfolgreich
        """
        try:
            if self.env_path.exists():
                load_dotenv(self.env_path)

            for mode in TradingMode:
                keys = self.ENV_KEYS[mode]
                api_key = os.getenv(keys['api_key'], '')
                api_secret = os.getenv(keys['api_secret'], '')

                self._credentials[mode] = APICredentials(
                    api_key=api_key,
                    api_secret=api_secret
                )

            self._loaded = True
            self.logger.debug(f'API-Konfiguration geladen von {self.env_path}')
            return True

        except Exception as e:
            self.logger.error(f'Fehler beim Laden der API-Konfiguration: {e}')
            return False

    def save(self) -> bool:
        """
        Speichert API-Schluessel in .env Datei.

        Returns:
            True wenn erfolgreich
        """
        try:
            # Sicherstellen dass Verzeichnis existiert
            self.env_path.parent.mkdir(parents=True, exist_ok=True)

            # .env Datei erstellen falls nicht vorhanden
            if not self.env_path.exists():
                self.env_path.touch()

            for mode in TradingMode:
                keys = self.ENV_KEYS[mode]
                creds = self._credentials[mode]

                set_key(str(self.env_path), keys['api_key'], creds.api_key)
                set_key(str(self.env_path), keys['api_secret'], creds.api_secret)

            self.logger.success(f'API-Konfiguration gespeichert: {self.env_path}')
            return True

        except Exception as e:
            self.logger.error(f'Fehler beim Speichern der API-Konfiguration: {e}')
            return False

    def get_credentials(self, mode: TradingMode) -> APICredentials:
        """
        Holt Credentials fuer einen Modus.

        Args:
            mode: Trading-Modus

        Returns:
            APICredentials Objekt
        """
        if not self._loaded:
            self.load()

        return self._credentials.get(mode, APICredentials())

    def set_credentials(
        self,
        mode: TradingMode,
        api_key: str,
        api_secret: str
    ) -> bool:
        """
        Setzt Credentials fuer einen Modus.

        Args:
            mode: Trading-Modus
            api_key: API-Schluessel
            api_secret: API-Secret

        Returns:
            True wenn gueltig
        """
        creds = APICredentials(api_key=api_key, api_secret=api_secret)

        if not creds.validate():
            self.logger.warning(f'Ungueltige Credentials fuer {mode.value}')
            return False

        self._credentials[mode] = creds

        if mode == TradingMode.LIVE:
            self.logger.warning('LIVE API-Keys konfiguriert - Vorsicht!')
        else:
            self.logger.info('Testnet API-Keys konfiguriert')

        return True

    def is_configured(self, mode: TradingMode) -> bool:
        """
        Prueft ob Modus konfiguriert ist.

        Args:
            mode: Trading-Modus

        Returns:
            True wenn API-Keys vorhanden
        """
        return self._credentials.get(mode, APICredentials()).is_configured

    def get_status(self) -> Dict[str, Dict]:
        """
        Gibt Status aller Modi zurueck.

        Returns:
            Dictionary mit Status-Informationen
        """
        return {
            mode.value: {
                'configured': creds.is_configured,
                'api_key_masked': creds.mask(),
                'valid': creds.validate() if creds.is_configured else False
            }
            for mode, creds in self._credentials.items()
        }

    def clear_credentials(self, mode: TradingMode) -> None:
        """
        Loescht Credentials fuer einen Modus.

        Args:
            mode: Trading-Modus
        """
        self._credentials[mode] = APICredentials()
        self.logger.info(f'Credentials geloescht: {mode.value}')

    def export_status(self, filepath: Path) -> bool:
        """
        Exportiert Status (ohne Secrets) zu JSON.

        Args:
            filepath: Ziel-Pfad

        Returns:
            True wenn erfolgreich
        """
        try:
            status = self.get_status()
            with open(filepath, 'w') as f:
                json.dump(status, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f'Export fehlgeschlagen: {e}')
            return False


# Singleton-Instanz
_config_instance: Optional[APIConfig] = None


def get_api_config() -> APIConfig:
    """Gibt Singleton-Instanz der API-Konfiguration zurueck."""
    global _config_instance
    if _config_instance is None:
        _config_instance = APIConfig()
        _config_instance.load()
    return _config_instance
