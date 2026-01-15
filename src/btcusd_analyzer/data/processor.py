"""
Feature Processor Modul - Generiert Features aus Rohdaten
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from ..core.logger import get_logger
from ..core.exceptions import DataValidationError, MissingDataError


class FeatureProcessor:
    """
    Generiert Features aus OHLCV-Rohdaten.

    Verfuegbare Features:
    - Basis: Open, High, Low, Close
    - Abgeleitet: PriceChange, PriceChangePct, Range, RangePct
    - Volumen: Volume, RelativeVolume
    - Zeit (zyklisch): hour_sin, hour_cos
    - Technisch: SMA, EMA, RSI, MACD, Bollinger Bands, ATR

    Attributes:
        features: Liste der zu generierenden Features
    """

    # Standard-Features (wie im MATLAB-Projekt)
    DEFAULT_FEATURES = ['Open', 'High', 'Low', 'Close', 'PriceChange', 'PriceChangePct']

    # Alle verfuegbaren Features fuer GUI-Auswahl
    AVAILABLE_FEATURES = {
        'price': ['Open', 'High', 'Low', 'Close', 'PriceChange', 'PriceChangePct'],
        'volume': ['Volume', 'RelativeVolume'],
        'time': ['hour_sin', 'hour_cos'],
        'volatility': ['ATR', 'ATR_Pct', 'RollingStd', 'RollingStd_Pct', 'HighLowRange', 'ReturnVol', 'ParkinsonVol'],
        'technical': ['SMA', 'EMA', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'BB_Width'],
    }

    # Erforderliche Basis-Spalten fuer Feature-Berechnung
    REQUIRED_COLUMNS = ['Open', 'High', 'Low', 'Close']

    def __init__(self, features: Optional[List[str]] = None):
        """
        Initialisiert den Feature Processor.

        Args:
            features: Liste der gewuenschten Features (default: DEFAULT_FEATURES)
        """
        self.features = features or self.DEFAULT_FEATURES.copy()
        self.logger = get_logger()

    def validate_input(self, df: pd.DataFrame, strict: bool = False) -> List[str]:
        """
        Validiert die Eingabedaten vor der Feature-Berechnung.

        Args:
            df: DataFrame mit OHLCV-Rohdaten
            strict: Wenn True, wird bei Fehlern eine Exception geworfen

        Returns:
            Liste von Validierungsfehlern (leer wenn alles OK)

        Raises:
            DataValidationError: Bei kritischen Fehlern (wenn strict=True)
            MissingDataError: Wenn DataFrame None oder leer ist
        """
        errors: List[str] = []

        # Null-Check
        if df is None:
            if strict:
                raise MissingDataError("DataFrame ist None")
            return ["DataFrame ist None"]

        if df.empty:
            if strict:
                raise MissingDataError("DataFrame ist leer")
            return ["DataFrame ist leer"]

        # Spalten-Validierung
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            msg = f"Fehlende Pflichtspalten: {', '.join(missing_cols)}"
            errors.append(msg)
            if strict:
                raise DataValidationError(msg, invalid_fields=missing_cols)

        # Datentyp-Validierung
        for col in self.REQUIRED_COLUMNS:
            if col in df.columns and not np.issubdtype(df[col].dtype, np.number):
                errors.append(f"Spalte '{col}' ist nicht numerisch")

        # NaN-Check in Pflichtspalten
        for col in self.REQUIRED_COLUMNS:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    errors.append(f"{nan_count} NaN-Werte in '{col}'")

        # Mindestanzahl Datenpunkte
        min_required = 20  # Fuer technische Indikatoren wie SMA20
        if len(df) < min_required:
            errors.append(f"Zu wenig Datenpunkte: {len(df)} < {min_required}")

        if strict and errors:
            raise DataValidationError(
                f"Input-Validierung fehlgeschlagen: {len(errors)} Fehler",
                invalid_fields=errors
            )

        return errors

    def process(self, df: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
        """
        Generiert alle konfigurierten Features.

        Args:
            df: DataFrame mit OHLCV-Rohdaten
            validate: Wenn True, werden Eingabedaten validiert

        Returns:
            DataFrame mit generierten Features

        Raises:
            DataValidationError: Bei ungueltigen Eingabedaten (wenn validate=True)
        """
        # Optionale Validierung
        if validate:
            errors = self.validate_input(df)
            if errors:
                self.logger.warning(f"Input-Validierung: {len(errors)} Probleme gefunden")
                for err in errors:
                    self.logger.warning(f"  - {err}")

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

    # === Volatilitaets-Features ===

    def _calc_atr_pct(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """ATR als Prozentsatz des Preises (normalisierte Volatilitaet)."""
        atr = self._calc_atr(df, period)
        return (atr / df['Close']) * 100

    def _calc_rollingstd(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Rolling Standard Deviation der Close-Preise."""
        return df['Close'].rolling(window=period).std()

    def _calc_rollingstd_pct(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Rolling Std als Prozentsatz des Preises."""
        std = df['Close'].rolling(window=period).std()
        return (std / df['Close']) * 100

    def _calc_highlowrange(self, df: pd.DataFrame) -> pd.Series:
        """High-Low Range als Prozentsatz des Close (Tagesvolatilitaet)."""
        return ((df['High'] - df['Low']) / df['Close']) * 100

    def _calc_returnvol(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Volatilitaet der Returns (Standardabweichung der prozentualen Aenderungen)."""
        returns = df['Close'].pct_change() * 100
        return returns.rolling(window=period).std()

    def _calc_parkinsonvol(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Parkinson Volatilitaet - effizienterer Schaetzer als Close-basierte Volatilitaet.
        Nutzt High-Low Range fuer bessere Schaetzung.
        Formel: sqrt(1/(4*ln(2)) * sum(ln(H/L)^2) / n)
        """
        log_hl = np.log(df['High'] / df['Low'])
        log_hl_sq = log_hl ** 2
        # Faktor: 1/(4*ln(2)) ≈ 0.361
        parkinson = np.sqrt(log_hl_sq.rolling(window=period).mean() / (4 * np.log(2)))
        # In Prozent umrechnen
        return parkinson * 100

    # === Volumen-Features ===

    def _calc_volume(self, df: pd.DataFrame) -> pd.Series:
        """Rohes Handelsvolumen."""
        if 'Vol' in df.columns:
            return df['Vol']
        elif 'TickVol' in df.columns:
            return df['TickVol']
        return pd.Series(0, index=df.index)

    def _calc_relativevolume(self, df: pd.DataFrame) -> pd.Series:
        """Relatives Volumen (aktuell / 20-Perioden-Durchschnitt)."""
        vol = self._calc_volume(df)
        sma20 = vol.rolling(window=20).mean()
        return (vol / sma20).fillna(1.0)

    # === Zeit-Features (zyklisch kodiert) ===

    def _calc_hour_sin(self, df: pd.DataFrame) -> pd.Series:
        """Stunde als Sinus (zyklisch kodiert)."""
        if 'DateTime' not in df.columns:
            self.logger.warning('DateTime-Spalte fehlt fuer hour_sin')
            return pd.Series(0, index=df.index)
        hour = df['DateTime'].dt.hour
        return np.sin(2 * np.pi * hour / 24)

    def _calc_hour_cos(self, df: pd.DataFrame) -> pd.Series:
        """Stunde als Cosinus (zyklisch kodiert)."""
        if 'DateTime' not in df.columns:
            self.logger.warning('DateTime-Spalte fehlt fuer hour_cos')
            return pd.Series(0, index=df.index)
        hour = df['DateTime'].dt.hour
        return np.cos(2 * np.pi * hour / 24)

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
