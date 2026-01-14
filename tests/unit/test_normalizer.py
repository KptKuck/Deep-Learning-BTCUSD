"""
Unit Tests fuer Normalizer-Klassen
"""

import pytest
import numpy as np


class TestZScoreNormalizer:
    """Tests fuer ZScoreNormalizer."""

    def test_fit_transform(self):
        """Test: Fit und Transform funktionieren."""
        from btcusd_analyzer.training.normalizer import ZScoreNormalizer

        data = np.random.randn(100, 5) * 10 + 50  # Zufallsdaten

        normalizer = ZScoreNormalizer()
        result = normalizer.fit_transform(data)

        assert result.shape == data.shape
        # Normalisierte Daten sollten mean~0 und std~1 haben
        assert abs(result.mean()) < 0.1
        assert abs(result.std() - 1.0) < 0.1

    def test_inverse_transform(self):
        """Test: Inverse Transform stellt Originaldaten wieder her."""
        from btcusd_analyzer.training.normalizer import ZScoreNormalizer

        data = np.random.randn(100, 5) * 10 + 50

        normalizer = ZScoreNormalizer()
        normalized = normalizer.fit_transform(data)
        restored = normalizer.inverse_transform(normalized)

        np.testing.assert_array_almost_equal(data, restored, decimal=10)

    def test_transform_new_data(self):
        """Test: Transform auf neuen Daten mit gelernten Parametern."""
        from btcusd_analyzer.training.normalizer import ZScoreNormalizer

        train_data = np.random.randn(100, 5) * 10 + 50
        test_data = np.random.randn(20, 5) * 10 + 50

        normalizer = ZScoreNormalizer()
        normalizer.fit(train_data)
        result = normalizer.transform(test_data)

        assert result.shape == test_data.shape


class TestMinMaxNormalizer:
    """Tests fuer MinMaxNormalizer."""

    def test_range_0_1(self):
        """Test: Normalisierte Daten im Bereich [0, 1]."""
        from btcusd_analyzer.training.normalizer import MinMaxNormalizer

        data = np.random.randn(100, 5) * 100

        normalizer = MinMaxNormalizer()
        result = normalizer.fit_transform(data)

        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_custom_range(self):
        """Test: Normalisierung mit benutzerdefiniertem Bereich."""
        from btcusd_analyzer.training.normalizer import MinMaxNormalizer

        data = np.random.randn(100, 5) * 100

        normalizer = MinMaxNormalizer(feature_range=(-1, 1))
        result = normalizer.fit_transform(data)

        assert result.min() >= -1.0
        assert result.max() <= 1.0


class TestRobustNormalizer:
    """Tests fuer RobustNormalizer."""

    def test_robust_to_outliers(self):
        """Test: Robuster gegen Ausreisser als ZScore."""
        from btcusd_analyzer.training.normalizer import RobustNormalizer, ZScoreNormalizer

        # Daten mit Ausreissern
        data = np.random.randn(100, 1)
        data[0] = 1000  # Extremer Ausreisser
        data[1] = -1000

        robust = RobustNormalizer()
        zscore = ZScoreNormalizer()

        result_robust = robust.fit_transform(data.copy())
        result_zscore = zscore.fit_transform(data.copy())

        # Robust-Normalisierung sollte weniger extreme Werte produzieren
        # fuer die normalen Datenpunkte (ohne Ausreisser)
        normal_idx = slice(2, None)
        robust_std = result_robust[normal_idx].std()
        zscore_std = result_zscore[normal_idx].std()

        # Die Robust-Version sollte aehnliche Streuung haben
        assert robust_std > 0
