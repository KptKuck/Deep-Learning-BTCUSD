"""
Feature Processor Modul - Generiert Features aus Rohdaten
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from ..core.logger import get_logger


class FeatureProcessor:
    """
    Generiert Features aus OHLCV-Rohdaten.

    Verfuegbare Features:
    - Basis: Open, High, Low, Close, Volume
    - Abgeleitet: PriceChange, PriceChangePct, Range, RangePct
    - Technisch: SMA, EMA, RSI, MACD, Bollinger Bands, ATR

    Attributes:
        features: Liste der zu generierenden Features
    """

    # Standard-Features (wie im MATLAB-Projekt)
    DEFAULT_FEATURES = ['Open', 'High', 'Low', 'Close', 'PriceChange', 'PriceChangePct']

    def __init__(self, features: Optional[List[str]] = None):
        """
        Initialisiert den Feature Processor.

        Args:
            features: Liste der gewuenschten Features (default: DEFAULT_FEATURES)
        """
        self.features = features or self.DEFAULT_FEATURES.copy()
        self.logger = get_logger()

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generiert alle konfigurierten Features.

        Args:
            df: DataFrame mit OHLCV-Rohdaten

        Returns:
            DataFrame mit generierten Features
        """
        result = df.copy()

        for feature in self.features:
            if feature in result.columns:
                continue  # Bereits vorhanden

            if hasattr(self, f'_calc_{feature.lower()}'):
                method = getattr(self, f'_calc_{feature.lower()}')
                result[feature] = method(result)
                self.logger.debug(f'Feature "{feature}" generiert')
            else:
                self.logger.warning(f'Unbekanntes Feature: {feature}')

        return result

    def get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extrahiert Feature-Matrix aus DataFrame.

        Args:
            df: DataFrame mit Features

        Returns:
            NumPy Array mit Shape (n_samples, n_features)
        """
        # Fehlende Features generieren
        df = self.process(df)

        # Nur konfigurierte Features extrahieren
        available = [f for f in self.features if f in df.columns]
        if len(available) != len(self.features):
            missing = set(self.features) - set(available)
            self.logger.warning(f'Fehlende Features: {missing}')

        return df[available].values.astype(np.float32)

    # === Abgeleitete Features ===

    def _calc_pricechange(self, df: pd.DataFrame) -> pd.Series:
        """Absolute Preisaenderung."""
        return df['Close'].diff().fillna(0)

    def _calc_pricechangepct(self, df: pd.DataFrame) -> pd.Series:
        """Prozentuale Preisaenderung."""
        return df['Close'].pct_change().fillna(0) * 100

    def _calc_range(self, df: pd.DataFrame) -> pd.Series:
        """Absolute Preisspanne (High - Low)."""
        return df['High'] - df['Low']

    def _calc_rangepct(self, df: pd.DataFrame) -> pd.Series:
        """Prozentuale Preisspanne."""
        return ((df['High'] - df['Low']) / df['Low']) * 100

    def _calc_typical_price(self, df: pd.DataFrame) -> pd.Series:
        """Typischer Preis (HLC/3)."""
        return (df['High'] + df['Low'] + df['Close']) / 3

    def _calc_hlc3(self, df: pd.DataFrame) -> pd.Series:
        """Alias fuer Typical Price."""
        return self._calc_typical_price(df)

    def _calc_ohlc4(self, df: pd.DataFrame) -> pd.Series:
        """OHLC/4 Durchschnitt."""
        return (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4

    # === Technische Indikatoren ===

    def _calc_sma(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Simple Moving Average."""
        return df['Close'].rolling(window=period).mean()

    def _calc_ema(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Exponential Moving Average."""
        return df['Close'].ewm(span=period, adjust=False).mean()

    def _calc_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def _calc_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range."""
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def _calc_macd(self, df: pd.DataFrame) -> pd.Series:
        """MACD Line (12-26 EMA Differenz)."""
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        return ema12 - ema26

    def _calc_macd_signal(self, df: pd.DataFrame) -> pd.Series:
        """MACD Signal Line (9 EMA des MACD)."""
        macd = self._calc_macd(df)
        return macd.ewm(span=9, adjust=False).mean()

    def _calc_macd_hist(self, df: pd.DataFrame) -> pd.Series:
        """MACD Histogram."""
        macd = self._calc_macd(df)
        signal = self._calc_macd_signal(df)
        return macd - signal

    def _calc_bb_upper(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.Series:
        """Bollinger Band Upper."""
        sma = df['Close'].rolling(window=period).mean()
        std = df['Close'].rolling(window=period).std()
        return sma + (std_dev * std)

    def _calc_bb_lower(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.Series:
        """Bollinger Band Lower."""
        sma = df['Close'].rolling(window=period).mean()
        std = df['Close'].rolling(window=period).std()
        return sma - (std_dev * std)

    def _calc_bb_width(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.Series:
        """Bollinger Band Width."""
        upper = self._calc_bb_upper(df, period, std_dev)
        lower = self._calc_bb_lower(df, period, std_dev)
        sma = df['Close'].rolling(window=period).mean()
        return (upper - lower) / sma * 100

    def add_feature(self, name: str):
        """Fuegt ein Feature zur Liste hinzu."""
        if name not in self.features:
            self.features.append(name)

    def remove_feature(self, name: str):
        """Entfernt ein Feature aus der Liste."""
        if name in self.features:
            self.features.remove(name)

    def get_feature_names(self) -> List[str]:
        """Gibt die Liste der konfigurierten Features zurueck."""
        return self.features.copy()
