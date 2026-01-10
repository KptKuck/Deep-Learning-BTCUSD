"""
Config Modul - Zentrale Konfiguration und Pfadverwaltung
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv


@dataclass
class PathConfig:
    """Pfad-Konfiguration"""
    base_dir: Path = field(default_factory=lambda: Path.cwd())
    data_dir: Path = field(default_factory=lambda: Path.cwd() / 'data')
    results_dir: Path = field(default_factory=lambda: Path.cwd() / 'results')
    log_dir: Path = field(default_factory=lambda: Path.cwd() / 'logs')
    models_dir: Path = field(default_factory=lambda: Path.cwd() / 'models')

    def __post_init__(self):
        """Erstellt Verzeichnisse falls nicht vorhanden."""
        for path in [self.data_dir, self.results_dir, self.log_dir, self.models_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def get_session_dir(self, prefix: str = '') -> Path:
        """Erstellt Session-Verzeichnis mit Zeitstempel."""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        session_name = f'{prefix}_{timestamp}' if prefix else timestamp
        session_dir = self.results_dir / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir


@dataclass
class TrainingConfig:
    """Training-Parameter"""
    # Sequenz-Parameter
    lookback: int = 50
    lookforward: int = 100

    # Training-Parameter
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2

    # Early Stopping
    patience: int = 10
    min_delta: float = 0.001

    # Modell-Parameter
    hidden_size: int = 100
    num_layers: int = 2
    dropout: float = 0.2
    num_classes: int = 3  # HOLD, BUY, SELL

    # Features
    features: List[str] = field(default_factory=lambda: [
        'Open', 'High', 'Low', 'Close', 'PriceChange', 'PriceChangePct'
    ])

    # Klassen
    class_names: List[str] = field(default_factory=lambda: ['HOLD', 'BUY', 'SELL'])

    @property
    def sequence_length(self) -> int:
        """Gesamte Sequenzlaenge (lookback + lookforward)."""
        return self.lookback + self.lookforward

    @property
    def input_size(self) -> int:
        """Anzahl der Input-Features."""
        return len(self.features)


@dataclass
class BacktestConfig:
    """Backtest-Parameter"""
    initial_capital: float = 10000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    position_size: float = 1.0  # 100% pro Trade


@dataclass
class TradingConfig:
    """Live-Trading Parameter"""
    # API-Konfiguration (aus .env geladen)
    testnet_api_key: str = ''
    testnet_api_secret: str = ''
    live_api_key: str = ''
    live_api_secret: str = ''

    # Trading-Parameter
    symbol: str = 'BTCUSDT'
    position_size_pct: float = 0.1  # 10% des Kapitals
    max_position_size: float = 1.0  # Max 1 BTC
    stop_loss_pct: float = 0.02  # 2% Stop-Loss
    take_profit_pct: float = 0.05  # 5% Take-Profit

    def load_api_keys(self):
        """Laedt API-Keys aus Umgebungsvariablen."""
        load_dotenv()
        self.testnet_api_key = os.getenv('BINANCE_TESTNET_API_KEY', '')
        self.testnet_api_secret = os.getenv('BINANCE_TESTNET_SECRET', '')
        self.live_api_key = os.getenv('BINANCE_LIVE_API_KEY', '')
        self.live_api_secret = os.getenv('BINANCE_LIVE_SECRET', '')


@dataclass
class WebConfig:
    """Web-Dashboard Konfiguration"""
    host: str = '0.0.0.0'  # Alle Netzwerk-Interfaces
    port: int = 5000
    auto_refresh_seconds: int = 5


@dataclass
class GUIConfig:
    """GUI-Konfiguration (Dark Theme)"""
    # Hintergrundfarben
    bg_color: tuple = (0.15, 0.15, 0.15)
    bg_color_hex: str = '#262626'

    # Statusfarben
    color_success: tuple = (0.2, 0.7, 0.3)
    color_success_hex: str = '#33b34d'
    color_warning: tuple = (0.9, 0.7, 0.2)
    color_warning_hex: str = '#e6b333'
    color_error: tuple = (0.8, 0.3, 0.2)
    color_error_hex: str = '#cc4d33'
    color_neutral: tuple = (0.5, 0.5, 0.5)
    color_neutral_hex: str = '#808080'

    # Text
    text_color: str = '#ffffff'
    text_color_secondary: str = '#aaaaaa'

    # Trading-Modi Farben
    testnet_bg: str = '#1a2e1a'
    testnet_border: str = '#33cc33'
    live_bg: str = '#3d1a1a'
    live_border: str = '#ff3333'


class Config:
    """
    Zentrale Konfigurationsklasse.

    Beinhaltet alle Konfigurationen fuer:
    - Pfade
    - Training
    - Backtest
    - Live-Trading
    - Web-Dashboard
    - GUI

    Usage:
        config = Config()
        config.training.epochs = 200
        config.paths.get_session_dir('training')
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialisiert die Konfiguration.

        Args:
            base_dir: Basis-Verzeichnis (default: aktuelles Arbeitsverzeichnis)
        """
        base = Path(base_dir) if base_dir else Path.cwd()

        self.paths = PathConfig(
            base_dir=base,
            data_dir=base / 'data',
            results_dir=base / 'results',
            log_dir=base / 'logs',
            models_dir=base / 'models'
        )
        self.training = TrainingConfig()
        self.backtest = BacktestConfig()
        self.trading = TradingConfig()
        self.web = WebConfig()
        self.gui = GUIConfig()

        # API-Keys laden
        self.trading.load_api_keys()

    def save(self, filepath: Path):
        """Speichert Konfiguration als Text-Datei."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('# BTCUSD Analyzer - Konfiguration\n')
            f.write(f'# Erstellt: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')

            f.write('## Training\n')
            f.write(f'lookback: {self.training.lookback}\n')
            f.write(f'lookforward: {self.training.lookforward}\n')
            f.write(f'epochs: {self.training.epochs}\n')
            f.write(f'batch_size: {self.training.batch_size}\n')
            f.write(f'learning_rate: {self.training.learning_rate}\n')
            f.write(f'hidden_size: {self.training.hidden_size}\n')
            f.write(f'num_layers: {self.training.num_layers}\n')
            f.write(f'dropout: {self.training.dropout}\n')
            f.write(f'features: {", ".join(self.training.features)}\n\n')

            f.write('## Backtest\n')
            f.write(f'initial_capital: {self.backtest.initial_capital}\n')
            f.write(f'commission: {self.backtest.commission}\n')
            f.write(f'slippage: {self.backtest.slippage}\n')

    @classmethod
    def load(cls, filepath: Path) -> 'Config':
        """Laedt Konfiguration aus Datei (vereinfacht)."""
        config = cls()
        # TODO: Implementiere vollstaendiges Laden
        return config
