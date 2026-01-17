"""
1D CNN Classifier fuer Trendwechsel-Erkennung.
"""

import torch
import torch.nn as nn

from btcusd_analyzer.models.base import BaseModel


class CNNClassifier(BaseModel):
    """
    1D Convolutional Neural Network Classifier.

    CNN ist gut fuer lokale Muster-Erkennung in Zeitreihen.

    Architektur:
    - Mehrere 1D Convolutional Layers mit ReLU
    - MaxPooling zwischen Conv-Layern
    - Global Average Pooling am Ende
    - Fully Connected Layer fuer Klassifikation

    Eingabe: (batch_size, sequence_length, input_size)
    Ausgabe: (batch_size, num_classes)
    """

    def __init__(
        self,
        input_size: int,
        num_filters: int = 64,
        kernel_size: int = 3,
        num_conv_layers: int = 3,
        num_classes: int = 3,
        dropout: float = 0.3
    ):
        """
        Initialisiert den CNN Classifier.

        Args:
            input_size: Anzahl der Input-Features pro Zeitschritt
            num_filters: Anzahl Filter pro Conv-Layer (verdoppelt sich pro Layer)
            kernel_size: Kernel-Groesse fuer Convolutions
            num_conv_layers: Anzahl der Conv-Layer
            num_classes: Anzahl der Ausgabeklassen (HOLD=0, BUY=1, SELL=2)
            dropout: Dropout-Rate
        """
        super().__init__(name='CNNClassifier')

        self._log_debug(f"__init__() - input_size={input_size}, num_filters={num_filters}, "
                       f"kernel_size={kernel_size}, num_layers={num_conv_layers}, num_classes={num_classes}")

        self.input_size = input_size
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_conv_layers = num_conv_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout

        # Conv-Layers erstellen
        layers = []
        in_channels = input_size

        for i in range(num_conv_layers):
            out_channels = num_filters * (2 ** i)  # Verdopple Filter pro Layer

            # Conv1d erwartet (batch, channels, length)
            layers.extend([
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2  # Same padding
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(dropout)
            ])

            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Fully Connected
        self.fc = nn.Linear(in_channels, num_classes)

        # Log Parameter-Anzahl
        num_params = sum(p.numel() for p in self.parameters())
        self._log_debug(f"__init__() - Modell erstellt: {num_params:,} Parameter, "
                       f"final_channels={in_channels}, output={num_classes}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Vorwaertsdurchlauf.

        Args:
            x: Eingabetensor (batch_size, sequence_length, input_size)

        Returns:
            Ausgabetensor (batch_size, num_classes) - Logits fuer jede Klasse
        """
        # Conv1d erwartet (batch, channels, length)
        # Input ist (batch, length, channels)
        x = x.transpose(1, 2)  # (batch, input_size, seq_len)

        # Convolutional Layers
        x = self.conv_layers(x)  # (batch, channels, seq_len')

        # Global Average Pooling
        x = self.global_avg_pool(x)  # (batch, channels, 1)
        x = x.squeeze(-1)  # (batch, channels)

        # Classification
        logits = self.fc(x)  # (batch, num_classes)

        return logits
