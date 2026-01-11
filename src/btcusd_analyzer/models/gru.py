"""
GRU Classifier fuer Trendwechsel-Erkennung.
"""

import torch
import torch.nn as nn

from btcusd_analyzer.models.base import BaseModel


class GRUClassifier(BaseModel):
    """
    GRU (Gated Recurrent Unit) Classifier fuer Klassifikation von Zeitreihen.

    GRU ist schneller als LSTM und hat weniger Parameter, aber oft aehnliche Performance.

    Architektur:
    - GRU (optional bidirektional)
    - Dropout fuer Regularisierung
    - Fully Connected Layer fuer Klassifikation

    Eingabe: (batch_size, sequence_length, input_size)
    Ausgabe: (batch_size, num_classes)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Initialisiert den GRU Classifier.

        Args:
            input_size: Anzahl der Input-Features pro Zeitschritt
            hidden_size: Groesse des Hidden States
            num_layers: Anzahl der GRU-Layer
            num_classes: Anzahl der Ausgabeklassen (HOLD=0, BUY=1, SELL=2)
            dropout: Dropout-Rate (nur zwischen Layern, nicht am Output)
            bidirectional: Bidirektionales GRU verwenden
        """
        name = 'BiGRU' if bidirectional else 'GRU'
        super().__init__(name=f'{name}Classifier')

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # GRU Layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Dropout vor FC
        self.dropout = nn.Dropout(dropout)

        # Fully Connected Layer
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
        # GRU
        # output: (batch, seq_len, hidden_size * num_directions)
        # h_n: (num_layers * num_directions, batch, hidden_size)
        output, h_n = self.gru(x)

        # Letzten Hidden State verwenden
        if self.bidirectional:
            # Konkateniere forward und backward vom letzten Layer
            h_forward = h_n[-2, :, :]  # Vorletzter Layer (forward)
            h_backward = h_n[-1, :, :]  # Letzter Layer (backward)
            h = torch.cat([h_forward, h_backward], dim=1)
        else:
            h = h_n[-1, :, :]  # Letzter Layer

        # Dropout + FC
        h = self.dropout(h)
        logits = self.fc(h)

        return logits
