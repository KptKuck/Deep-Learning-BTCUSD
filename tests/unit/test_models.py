"""
Unit Tests fuer Model-Klassen
"""

import pytest
import torch
import numpy as np


class TestBiLSTMModel:
    """Tests fuer BiLSTM-Modell."""

    def test_model_creation(self):
        """Test: Modell erfolgreich erstellen."""
        from btcusd_analyzer.models.bilstm import BiLSTMModel

        model = BiLSTMModel(
            input_size=6,
            hidden_size=64,
            num_layers=2,
            num_classes=3,
            dropout=0.2
        )

        assert model is not None

    def test_forward_pass(self):
        """Test: Forward-Pass mit korrekter Ausgabegroesse."""
        from btcusd_analyzer.models.bilstm import BiLSTMModel

        batch_size = 16
        seq_length = 50
        input_size = 6
        num_classes = 3

        model = BiLSTMModel(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            num_classes=num_classes
        )

        x = torch.randn(batch_size, seq_length, input_size)
        output = model(x)

        assert output.shape == (batch_size, num_classes)

    def test_predict(self):
        """Test: Predict gibt Klassen-Indizes zurueck."""
        from btcusd_analyzer.models.bilstm import BiLSTMModel

        batch_size = 16
        seq_length = 50
        input_size = 6

        model = BiLSTMModel(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            num_classes=3
        )
        model.eval()

        x = torch.randn(batch_size, seq_length, input_size)
        predictions = model.predict(x)

        assert len(predictions) == batch_size
        assert all(0 <= p <= 2 for p in predictions)

    def test_predict_proba(self):
        """Test: Predict_proba gibt Wahrscheinlichkeiten zurueck."""
        from btcusd_analyzer.models.bilstm import BiLSTMModel

        batch_size = 16
        seq_length = 50
        input_size = 6
        num_classes = 3

        model = BiLSTMModel(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            num_classes=num_classes
        )
        model.eval()

        x = torch.randn(batch_size, seq_length, input_size)
        proba = model.predict_proba(x)

        assert proba.shape == (batch_size, num_classes)
        # Wahrscheinlichkeiten sollten sich zu 1 summieren
        sums = proba.sum(axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(batch_size), decimal=5)

    def test_count_parameters(self):
        """Test: Parameter-Zaehlung funktioniert."""
        from btcusd_analyzer.models.bilstm import BiLSTMModel

        model = BiLSTMModel(
            input_size=6,
            hidden_size=64,
            num_layers=2,
            num_classes=3
        )

        params = model.count_parameters()

        assert params > 0
        assert isinstance(params, int)


class TestModelFactory:
    """Tests fuer ModelFactory."""

    def test_create_bilstm(self):
        """Test: BiLSTM-Modell erstellen."""
        from btcusd_analyzer.models.factory import ModelFactory

        model = ModelFactory.create(
            'bilstm',
            input_size=6,
            hidden_size=64,
            num_layers=2,
            num_classes=3
        )

        assert model is not None
        assert 'BiLSTM' in model.__class__.__name__

    def test_create_gru(self):
        """Test: GRU-Modell erstellen."""
        from btcusd_analyzer.models.factory import ModelFactory

        model = ModelFactory.create(
            'gru',
            input_size=6,
            hidden_size=64,
            num_layers=2,
            num_classes=3
        )

        assert model is not None

    def test_create_cnn(self):
        """Test: CNN-Modell erstellen."""
        from btcusd_analyzer.models.factory import ModelFactory

        model = ModelFactory.create(
            'cnn',
            input_size=6,
            hidden_size=64,
            num_layers=2,
            num_classes=3
        )

        assert model is not None

    def test_create_unknown_raises(self):
        """Test: Unbekannter Modelltyp wirft Fehler."""
        from btcusd_analyzer.models.factory import ModelFactory

        with pytest.raises((ValueError, KeyError)):
            ModelFactory.create('unknown_model', input_size=6)

    def test_available_models(self):
        """Test: Verfuegbare Modelle abfragen."""
        from btcusd_analyzer.models.factory import ModelFactory

        available = ModelFactory.available_models()

        assert isinstance(available, (list, tuple, dict))
        assert 'bilstm' in [m.lower() for m in available]
