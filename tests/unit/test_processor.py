"""
Unit Tests fuer FeatureProcessor
"""

import pytest
import pandas as pd
import numpy as np


class TestFeatureProcessor:
    """Tests fuer die FeatureProcessor-Klasse."""

    def test_processor_init(self):
        """Test: Processor erfolgreich initialisieren."""
        from btcusd_analyzer.data.processor import FeatureProcessor

        processor = FeatureProcessor()
        assert processor is not None

    def test_process_basic_features(self, sample_ohlcv_data):
        """Test: Basis-Features werden berechnet."""
        from btcusd_analyzer.data.processor import FeatureProcessor

        processor = FeatureProcessor()
        features = ['PriceChange', 'PriceChangePct']

        result = processor.process(sample_ohlcv_data, features=features)

        assert result is not None
        for feat in features:
            assert feat in result.columns, f"Feature '{feat}' fehlt"

    def test_process_sma(self, sample_ohlcv_data):
        """Test: SMA-Berechnung korrekt."""
        from btcusd_analyzer.data.processor import FeatureProcessor

        processor = FeatureProcessor()
        features = ['SMA_10', 'SMA_20']

        result = processor.process(sample_ohlcv_data, features=features)

        # Nach dropna sollten SMA-Werte vorhanden sein
        result = result.dropna()
        assert len(result) > 0

        # SMA_10 sollte Close-Durchschnitt der letzten 10 Perioden sein
        # Pruefe exemplarisch den letzten Wert
        expected_sma10 = sample_ohlcv_data['Close'].iloc[-10:].mean()
        actual_sma10 = result['SMA_10'].iloc[-1]
        assert abs(expected_sma10 - actual_sma10) < 0.01, "SMA_10 Berechnung fehlerhaft"

    def test_process_rsi(self, sample_ohlcv_data):
        """Test: RSI im gueltigen Bereich (0-100)."""
        from btcusd_analyzer.data.processor import FeatureProcessor

        processor = FeatureProcessor()
        features = ['RSI']

        result = processor.process(sample_ohlcv_data, features=features)
        result = result.dropna()

        assert (result['RSI'] >= 0).all(), "RSI < 0 gefunden"
        assert (result['RSI'] <= 100).all(), "RSI > 100 gefunden"

    def test_process_empty_features_list(self, sample_ohlcv_data):
        """Test: Leere Feature-Liste gibt Originaldaten zurueck."""
        from btcusd_analyzer.data.processor import FeatureProcessor

        processor = FeatureProcessor()
        result = processor.process(sample_ohlcv_data, features=[])

        assert result is not None
        assert len(result) == len(sample_ohlcv_data)

    def test_available_features(self):
        """Test: AVAILABLE_FEATURES ist definiert und nicht leer."""
        from btcusd_analyzer.data.processor import FeatureProcessor

        processor = FeatureProcessor()
        available = processor.AVAILABLE_FEATURES

        assert isinstance(available, dict)
        assert len(available) > 0

    def test_process_preserves_index(self, sample_ohlcv_data):
        """Test: DateTime-Index bleibt erhalten."""
        from btcusd_analyzer.data.processor import FeatureProcessor

        processor = FeatureProcessor()
        features = ['PriceChange']

        result = processor.process(sample_ohlcv_data, features=features)

        assert isinstance(result.index, pd.DatetimeIndex)
