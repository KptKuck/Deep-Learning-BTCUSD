"""
Pytest Konfiguration und gemeinsame Fixtures
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


@pytest.fixture
def sample_ohlcv_data():
    """Erstellt Beispiel-OHLCV-Daten fuer Tests."""
    np.random.seed(42)
    n_samples = 1000

    # Startpreis
    base_price = 50000.0

    # Random Walk fuer Close-Preise
    returns = np.random.normal(0.0001, 0.02, n_samples)
    close = base_price * np.cumprod(1 + returns)

    # OHLCV generieren
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n_samples)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n_samples)))
    open_ = np.roll(close, 1)
    open_[0] = base_price
    volume = np.random.uniform(100, 1000, n_samples)

    # DataFrame erstellen
    dates = pd.date_range(
        start='2024-01-01',
        periods=n_samples,
        freq='h'
    )

    df = pd.DataFrame({
        'Open': open_,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)

    return df


@pytest.fixture
def sample_features(sample_ohlcv_data):
    """Erstellt Beispiel-Features aus OHLCV-Daten."""
    df = sample_ohlcv_data.copy()

    # Basis-Features
    df['PriceChange'] = df['Close'].diff()
    df['PriceChangePct'] = df['Close'].pct_change() * 100
    df['Range'] = df['High'] - df['Low']
    df['RangePct'] = df['Range'] / df['Close'] * 100

    # SMA
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()

    # RSI (vereinfacht)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df.dropna()


@pytest.fixture
def sample_labels(sample_features):
    """Erstellt Beispiel-Labels (0=HOLD, 1=BUY, 2=SELL)."""
    n = len(sample_features)
    # Zufaellige Labels mit typischer Verteilung (mehr HOLD)
    np.random.seed(42)
    labels = np.random.choice(
        [0, 1, 2],
        size=n,
        p=[0.6, 0.2, 0.2]  # 60% HOLD, 20% BUY, 20% SELL
    )
    return labels


@pytest.fixture
def sample_sequences(sample_features, sample_labels):
    """Erstellt Beispiel-Sequenzen fuer Model-Training."""
    lookback = 50
    features = ['Open', 'High', 'Low', 'Close', 'PriceChange', 'PriceChangePct']

    X = sample_features[features].values
    y = sample_labels

    # Sequenzen erstellen
    sequences = []
    targets = []

    for i in range(lookback, len(X)):
        sequences.append(X[i-lookback:i])
        targets.append(y[i])

    return np.array(sequences), np.array(targets)


@pytest.fixture
def temp_csv_file(tmp_path, sample_ohlcv_data):
    """Erstellt temporaere CSV-Datei mit Testdaten."""
    filepath = tmp_path / "test_data.csv"
    sample_ohlcv_data.to_csv(filepath)
    return filepath


@pytest.fixture
def project_root():
    """Gibt den Projekt-Root-Pfad zurueck."""
    return Path(__file__).parent.parent
