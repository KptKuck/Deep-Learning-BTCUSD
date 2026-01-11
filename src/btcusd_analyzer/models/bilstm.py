"""
Bidirectional LSTM Classifier fuer Trendwechsel-Erkennung.
"""

import torch
import torch.nn as nn

from btcusd_analyzer.models.base import BaseModel


class BiLSTMClassifier(BaseModel):
    """
    Bidirektionaler LSTM Classifier fuer Klassifikation von Zeitreihen.

    Architektur:
    - Bidirektionales LSTM (liest Sequenz vor- und rueckwaerts)
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
        Initialisiert den BiLSTM Classifier.

        Args:
            input_size: Anzahl der Input-Features pro Zeitschritt
            hidden_size: Groesse des Hidden States
            num_layers: Anzahl der LSTM-Layer
            num_classes: Anzahl der Ausgabeklassen (HOLD=0, BUY=1, SELL=2)
            dropout: Dropout-Rate (nur zwischen Layern, nicht am Output)
            bidirectional: Bidirektionales LSTM verwenden
        """
        name = 'BiLSTM' if bidirectional else 'LSTM'
        super().__init__(name=f'{name}Classifier')

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM Layer
        self.lstm = nn.LSTM(
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
        # Bei bidirectional: hidden_size * 2 (forward + backward)
        fc_input_size = hidden_size * self.num_directions
        self.fc = nn.Linear(fc_input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Vorwaertsdurchlauf.

        Args:
            x: Eingabetensor (batch_size, sequence_length, input_size)

        Returns:
            Logits (batch_size, num_classes)
        """
        # LSTM durchlauf
        # lstm_out: (batch_size, sequence_length, hidden_size * num_directions)
        # hidden: (num_layers * num_directions, batch_size, hidden_size)
        lstm_out, (hidden, cell) = self.lstm(x)

        # Letzten Hidden State verwenden
        if self.bidirectional:
            # Konkateniere forward und backward Hidden States der letzten Layer
            # hidden[-2]: letzte forward Layer
            # hidden[-1]: letzte backward Layer
            hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            # Nur letzte Layer
            hidden_cat = hidden[-1]

        # Dropout
        hidden_cat = self.dropout(hidden_cat)

        # Klassifikation
        out = self.fc(hidden_cat)

        return out

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gibt Klassenvorhersagen zurueck.

        Args:
            x: Eingabetensor (batch_size, sequence_length, input_size)

        Returns:
            Klassenindizes (batch_size,)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gibt Klassenwahrscheinlichkeiten zurueck.

        Args:
            x: Eingabetensor (batch_size, sequence_length, input_size)

        Returns:
            Wahrscheinlichkeiten (batch_size, num_classes)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            proba = torch.softmax(logits, dim=1)
        return proba

    def get_config(self) -> dict:
        """Gibt die Modellkonfiguration zurueck."""
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_classes': self.num_classes,
            'dropout': self.dropout_rate,
            'bidirectional': self.bidirectional
        }
