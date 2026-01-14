"""
Model Factory - Erstellt Modelle basierend auf Namen.
"""

from typing import Optional
from btcusd_analyzer.models.bilstm import BiLSTMClassifier
from btcusd_analyzer.models.gru import GRUClassifier
from btcusd_analyzer.models.cnn import CNNClassifier
from btcusd_analyzer.models.cnn_lstm import CNNLSTMClassifier


class ModelFactory:
    """Factory-Klasse zum Erstellen von Modellen."""

    @staticmethod
    def create(
        model_name: str,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        **kwargs
    ):
        """
        Erstellt ein Modell basierend auf dem Namen.

        Args:
            model_name: Name des Modells ('bilstm', 'bigru', 'cnn', 'cnn_lstm')
            input_size: Anzahl der Input-Features
            hidden_size: Groesse des Hidden States (LSTM/GRU) oder num_filters (CNN)
            num_layers: Anzahl der Layer
            num_classes: Anzahl der Ausgabeklassen (default: 3 für BUY/SELL/HOLD)
            dropout: Dropout-Rate
            **kwargs: Weitere modellspezifische Parameter

        Returns:
            Initialisiertes Modell

        Raises:
            ValueError: Wenn model_name unbekannt ist
        """
        model_name = model_name.lower().strip()

        # LSTM Varianten
        if model_name in ['bilstm', 'bi_lstm', 'bilstmclassifier']:
            return BiLSTMClassifier(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=dropout,
                bidirectional=True
            )
        elif model_name in ['lstm']:
            return BiLSTMClassifier(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=dropout,
                bidirectional=False
            )

        # GRU Varianten
        elif model_name in ['bigru', 'bi_gru']:
            return GRUClassifier(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=dropout,
                bidirectional=True
            )
        elif model_name in ['gru']:
            return GRUClassifier(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=dropout,
                bidirectional=False
            )

        # CNN
        elif model_name in ['cnn', '1dcnn']:
            kernel_size = kwargs.get('kernel_size', 3)
            return CNNClassifier(
                input_size=input_size,
                num_filters=hidden_size,  # hidden_size wird als num_filters verwendet
                kernel_size=kernel_size,
                num_conv_layers=num_layers,
                num_classes=num_classes,
                dropout=dropout
            )

        # CNN-LSTM Hybrid
        elif model_name in ['cnn_lstm', 'cnn-lstm', 'cnnlstm']:
            num_conv_layers = kwargs.get('num_conv_layers', 2)
            num_lstm_layers = kwargs.get('num_lstm_layers', num_layers)
            bidirectional = kwargs.get('bidirectional', True)

            return CNNLSTMClassifier(
                input_size=input_size,
                num_filters=hidden_size // 2,  # Weniger Filter, da CNN+LSTM kombiniert
                kernel_size=kwargs.get('kernel_size', 3),
                num_conv_layers=num_conv_layers,
                hidden_size=hidden_size,
                num_lstm_layers=num_lstm_layers,
                num_classes=num_classes,
                dropout=dropout,
                bidirectional=bidirectional
            )

        else:
            raise ValueError(
                f"Unbekanntes Modell: '{model_name}'. "
                f"Verfügbare Modelle: {', '.join(ModelFactory.get_available_models())}"
            )

    @staticmethod
    def get_available_models():
        """Gibt Liste der verfügbaren Modellnamen zurück."""
        return [
            'BiLSTM', 'LSTM',
            'BiGRU', 'GRU',
            'CNN',
            'CNN-LSTM'
        ]
