"""
Integration Tests fuer die Daten-Pipeline
"""

import pytest
import numpy as np
import torch


class TestDataPipeline:
    """Tests fuer die vollstaendige Daten-Pipeline."""

    def test_csv_to_features(self, temp_csv_file):
        """Test: CSV laden und Features berechnen."""
        from btcusd_analyzer.data.reader import CSVReader
        from btcusd_analyzer.data.processor import FeatureProcessor

        # CSV laden
        reader = CSVReader()
        data = reader.read(temp_csv_file)
        assert data is not None

        # Features berechnen
        processor = FeatureProcessor()
        features = ['PriceChange', 'PriceChangePct', 'SMA_10', 'RSI']
        processed = processor.process(data, features=features)

        assert processed is not None
        assert len(processed) > 0

    def test_features_to_sequences(self, sample_features):
        """Test: Features zu Sequenzen konvertieren."""
        from btcusd_analyzer.training.sequence import SequenceGenerator

        feature_cols = ['Open', 'High', 'Low', 'Close', 'PriceChange', 'PriceChangePct']
        lookback = 50

        generator = SequenceGenerator(
            lookback=lookback,
            features=feature_cols
        )

        # Labels erstellen (fuer Test)
        labels = np.random.choice([0, 1, 2], size=len(sample_features))

        X, y = generator.create_sequences(sample_features, labels)

        assert X is not None
        assert y is not None
        assert X.shape[1] == lookback
        assert X.shape[2] == len(feature_cols)
        assert len(y) == len(X)

    def test_sequences_to_model(self, sample_sequences):
        """Test: Sequenzen durch Modell laufen lassen."""
        from btcusd_analyzer.models.bilstm import BiLSTMModel

        X, y = sample_sequences
        input_size = X.shape[2]

        model = BiLSTMModel(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            num_classes=3
        )
        model.eval()

        # Batch verarbeiten
        batch = torch.FloatTensor(X[:32])

        with torch.no_grad():
            output = model(batch)
            predictions = model.predict(batch)
            probabilities = model.predict_proba(batch)

        assert output.shape == (32, 3)
        assert len(predictions) == 32
        assert probabilities.shape == (32, 3)

    def test_full_pipeline(self, temp_csv_file):
        """Test: Vollstaendige Pipeline von CSV zu Vorhersage."""
        from btcusd_analyzer.data.reader import CSVReader
        from btcusd_analyzer.data.processor import FeatureProcessor
        from btcusd_analyzer.training.normalizer import ZScoreNormalizer
        from btcusd_analyzer.models.bilstm import BiLSTMModel
        import torch

        # 1. Daten laden
        reader = CSVReader()
        data = reader.read(temp_csv_file)

        # 2. Features berechnen
        processor = FeatureProcessor()
        feature_names = ['Open', 'High', 'Low', 'Close', 'PriceChange', 'PriceChangePct']
        processed = processor.process(data, features=['PriceChange', 'PriceChangePct'])
        processed = processed.dropna()

        # 3. Sequenzen erstellen
        lookback = 50
        X_data = processed[['Open', 'High', 'Low', 'Close', 'PriceChange', 'PriceChangePct']].values

        sequences = []
        for i in range(lookback, len(X_data)):
            sequences.append(X_data[i-lookback:i])
        X = np.array(sequences)

        # 4. Normalisieren
        normalizer = ZScoreNormalizer()
        X_flat = X.reshape(-1, X.shape[-1])
        normalizer.fit(X_flat)
        X_normalized = np.array([normalizer.transform(seq) for seq in X])

        # 5. Modell erstellen und Vorhersage
        model = BiLSTMModel(
            input_size=6,
            hidden_size=32,
            num_layers=1,
            num_classes=3
        )
        model.eval()

        batch = torch.FloatTensor(X_normalized[:16])
        with torch.no_grad():
            predictions = model.predict(batch)

        assert len(predictions) == 16
        assert all(0 <= p <= 2 for p in predictions)
