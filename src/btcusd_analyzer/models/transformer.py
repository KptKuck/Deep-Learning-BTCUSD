"""
Transformer Classifier fuer Trendwechsel-Erkennung.

PyTorch native Implementation mit:
- Positional Encoding
- Transformer Encoder
- Classification Head
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from btcusd_analyzer.models.base import BaseModel


class PositionalEncoding(nn.Module):
    """
    Positional Encoding fuer Transformer.

    Fuegt Positionsinformationen zur Eingabe hinzu, da Transformer
    keine inhärente Sequenzordnung haben.
    """

    def __init__(self, d_model: int, max_seq_length: int = 500, dropout: float = 0.1):
        """
        Args:
            d_model: Dimension des Modells
            max_seq_length: Maximale Sequenzlaenge
            dropout: Dropout-Rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Positional Encoding berechnen
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_seq_length, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        # Als Buffer registrieren (nicht trainierbar)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor (seq_len, batch_size, d_model) oder (batch_size, seq_len, d_model)

        Returns:
            Tensor mit Positional Encoding
        """
        # Anpassen an batch_first=True Format
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:x.size(1), 0, :].unsqueeze(0)
        return self.dropout(x)


class TransformerClassifier(BaseModel):
    """
    Transformer-basierter Classifier fuer Zeitreihen-Klassifikation.

    Architektur:
    - Input Embedding (Linear Projection)
    - Positional Encoding
    - Transformer Encoder (Multiple Layers)
    - Global Average Pooling
    - Classification Head

    Eingabe: (batch_size, sequence_length, input_size)
    Ausgabe: (batch_size, num_classes)
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 256,
        num_classes: int = 3,
        dropout: float = 0.1,
        max_seq_length: int = 500,
        activation: str = 'gelu'
    ):
        """
        Initialisiert den Transformer Classifier.

        Args:
            input_size: Anzahl der Input-Features pro Zeitschritt
            d_model: Dimension des Transformer-Modells
            nhead: Anzahl der Attention Heads
            num_encoder_layers: Anzahl der Encoder Layers
            dim_feedforward: Dimension des Feedforward Networks
            num_classes: Anzahl der Ausgabeklassen
            dropout: Dropout-Rate
            max_seq_length: Maximale Sequenzlaenge
            activation: Aktivierungsfunktion ('relu' oder 'gelu')
        """
        super().__init__(name='TransformerClassifier')

        self._log_debug(f"__init__() - input_size={input_size}, d_model={d_model}, "
                       f"nhead={nhead}, layers={num_encoder_layers}, num_classes={num_classes}")

        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.max_seq_length = max_seq_length
        self.activation = activation

        self._log_debug(f"__init__() - dim_feedforward={dim_feedforward}, dropout={dropout}, "
                       f"max_seq_length={max_seq_length}, activation={activation}")

        # Input Embedding: Projiziere Features auf d_model Dimension
        self.input_embedding = nn.Linear(input_size, d_model)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True  # Pre-LN fuer stabileres Training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model)
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

        # Log Parameter-Anzahl
        num_params = sum(p.numel() for p in self.parameters())
        self._log_debug(f"__init__() - Modell erstellt: {num_params:,} Parameter")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Vorwaertsdurchlauf.

        Args:
            x: Eingabetensor (batch_size, sequence_length, input_size)

        Returns:
            Logits (batch_size, num_classes)
        """
        # Input Embedding
        x = self.input_embedding(x)  # (batch, seq, d_model)

        # Positional Encoding
        x = self.pos_encoder(x)

        # Transformer Encoder
        x = self.transformer(x)  # (batch, seq, d_model)

        # Global Average Pooling ueber die Sequenz
        x = x.mean(dim=1)  # (batch, d_model)

        # Classification
        logits = self.classifier(x)  # (batch, num_classes)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Gibt Klassenvorhersagen zurueck."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Gibt Klassenwahrscheinlichkeiten zurueck."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            proba = torch.softmax(logits, dim=1)
        return proba

    def get_config(self) -> dict:
        """Gibt die Modellkonfiguration zurueck."""
        return {
            'input_size': self.input_size,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_encoder_layers': self.num_encoder_layers,
            'dim_feedforward': self.dim_feedforward,
            'num_classes': self.num_classes,
            'dropout': self.dropout_rate,
            'max_seq_length': self.max_seq_length,
            'activation': self.activation
        }

    def get_num_parameters(self) -> int:
        """Gibt die Anzahl der trainierbaren Parameter zurueck."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
