"""
CNN-LSTM Hybrid Classifier fuer Trendwechsel-Erkennung.
"""

import torch
import torch.nn as nn

from btcusd_analyzer.models.base import BaseModel


class CNNLSTMClassifier(BaseModel):
    """
    Hybrid CNN-LSTM Classifier.

    Kombiniert die Staerken beider Architekturen:
    - CNN extrahiert lokale Muster und Features
    - LSTM lernt temporale Abhaengigkeiten

    Architektur:
    - 1D CNN Layers fuer Feature-Extraktion
    - Bidirektionales LSTM fuer Sequenz-Modellierung
    - Fully Connected Layer fuer Klassifikation

    Eingabe: (batch_size, sequence_length, input_size)
    Ausgabe: (batch_size, num_classes)
    """

    def __init__(
        self,
        input_size: int,
        num_filters: int = 64,
        kernel_size: int = 3,
        num_conv_layers: int = 2,
        hidden_size: int = 128,
        num_lstm_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Initialisiert den CNN-LSTM Classifier.

        Args:
            input_size: Anzahl der Input-Features pro Zeitschritt
            num_filters: Anzahl Filter im ersten Conv-Layer
            kernel_size: Kernel-Groesse fuer Convolutions
            num_conv_layers: Anzahl der Conv-Layer
            hidden_size: Groesse des LSTM Hidden States
            num_lstm_layers: Anzahl der LSTM-Layer
            num_classes: Anzahl der Ausgabeklassen (HOLD=0, BUY=1, SELL=2)
            dropout: Dropout-Rate
            bidirectional: Bidirektionales LSTM verwenden
        """
        name = 'CNN-BiLSTM' if bidirectional else 'CNN-LSTM'
        super().__init__(name=f'{name}Classifier')

        self.input_size = input_size
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_conv_layers = num_conv_layers
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # CNN Feature Extractor
        cnn_layers = []
        in_channels = input_size

        for i in range(num_conv_layers):
            out_channels = num_filters * (2 ** i)

            cnn_layers.extend([
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)
        self.cnn_output_size = in_channels

        # LSTM Sequence Modeler
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Dropout + FC
        self.dropout = nn.Dropout(dropout)
        fc_input_size = hidden_size * self.num_directions
        self.fc = nn.Linear(fc_input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Vorwaertsdurchlauf.

        Args:
            x: Eingabetensor (batch_size, sequence_length, input_size)

        Returns:
            Ausgabetensor (batch_size, num_classes) - Logits fuer jede Klasse
        """
        # CNN Feature Extraction
        # Conv1d erwartet (batch, channels, length)
        x = x.transpose(1, 2)  # (batch, input_size, seq_len)
        x = self.cnn(x)  # (batch, num_filters, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, num_filters)

        # LSTM Sequence Modeling
        output, (h_n, c_n) = self.lstm(x)

        # Letzten Hidden State verwenden
        if self.bidirectional:
            h_forward = h_n[-2, :, :]
            h_backward = h_n[-1, :, :]
            h = torch.cat([h_forward, h_backward], dim=1)
        else:
            h = h_n[-1, :, :]

        # Classification
        h = self.dropout(h)
        logits = self.fc(h)

        return logits
