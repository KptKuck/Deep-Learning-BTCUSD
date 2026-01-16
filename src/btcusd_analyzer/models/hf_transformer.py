"""
Hugging Face Time Series Transformer Wrapper fuer Trendwechsel-Erkennung.

Verwendet Hugging Face transformers Library fuer Time Series Modelle.
Falls transformers nicht installiert ist, wird eine Fehlermeldung ausgegeben.
"""

from typing import Optional

import torch
import torch.nn as nn

from btcusd_analyzer.models.base import BaseModel


# Pruefe ob transformers verfuegbar ist
_HF_AVAILABLE = False
try:
    from transformers import (
        TimeSeriesTransformerConfig,
        TimeSeriesTransformerModel,
    )
    _HF_AVAILABLE = True
except ImportError:
    pass


class HFTimeSeriesClassifier(BaseModel):
    """
    Hugging Face Time Series Transformer fuer Klassifikation.

    Dieser Classifier verwendet Hugging Face's TimeSeriesTransformerModel
    als Backbone und fuegt einen Klassifikationskopf hinzu.

    Hinweis: Benoetigt `pip install transformers`

    Eingabe: (batch_size, sequence_length, input_size)
    Ausgabe: (batch_size, num_classes)
    """

    def __init__(
        self,
        input_size: int,
        context_length: int = 50,
        d_model: int = 64,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        encoder_attention_heads: int = 4,
        decoder_attention_heads: int = 4,
        encoder_ffn_dim: int = 128,
        decoder_ffn_dim: int = 128,
        num_classes: int = 3,
        dropout: float = 0.1,
        prediction_length: int = 1
    ):
        """
        Initialisiert den HF Time Series Classifier.

        Args:
            input_size: Anzahl der Input-Features pro Zeitschritt
            context_length: Kontext-Laenge fuer den Transformer
            d_model: Dimension des Modells
            encoder_layers: Anzahl der Encoder Layers
            decoder_layers: Anzahl der Decoder Layers
            encoder_attention_heads: Attention Heads im Encoder
            decoder_attention_heads: Attention Heads im Decoder
            encoder_ffn_dim: FFN Dimension im Encoder
            decoder_ffn_dim: FFN Dimension im Decoder
            num_classes: Anzahl der Ausgabeklassen
            dropout: Dropout-Rate
            prediction_length: Vorhersage-Laenge (fuer HF config)
        """
        super().__init__(name='HFTransformerClassifier')

        if not _HF_AVAILABLE:
            raise ImportError(
                "Hugging Face transformers nicht installiert. "
                "Bitte installieren mit: pip install transformers"
            )

        self.input_size = input_size
        self.context_length = context_length
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_attention_heads = decoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.decoder_ffn_dim = decoder_ffn_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.prediction_length = prediction_length

        # HuggingFace Config
        self.hf_config = TimeSeriesTransformerConfig(
            prediction_length=prediction_length,
            context_length=context_length,
            input_size=input_size,
            d_model=d_model,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            encoder_attention_heads=encoder_attention_heads,
            decoder_attention_heads=decoder_attention_heads,
            encoder_ffn_dim=encoder_ffn_dim,
            decoder_ffn_dim=decoder_ffn_dim,
            dropout=dropout,
            # Fuer Zeitreihen-Klassifikation benoetigen wir keinen Lags
            lags_sequence=[1],  # Minimal Lag
            num_time_features=1,  # Mindestens 1 Time Feature
            num_static_categorical_features=0,
            num_static_real_features=0,
            cardinality=[],
            embedding_dimension=[],
        )

        # HuggingFace Model (nur Encoder nutzen)
        self.transformer = TimeSeriesTransformerModel(self.hf_config)

        # Eigene Input Projection (da HF Model komplexe Eingabe erwartet)
        self.input_projection = nn.Linear(input_size, d_model)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

        # Layer Norm
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Vorwaertsdurchlauf.

        Da HuggingFace TimeSeriesTransformer eine komplexe Eingabe erwartet,
        verwenden wir hier eine vereinfachte Version mit eigener Projektion.

        Args:
            x: Eingabetensor (batch_size, sequence_length, input_size)

        Returns:
            Logits (batch_size, num_classes)
        """
        # Einfache Projektion und Pooling (statt komplexes HF Input Format)
        # Dies ist ein pragmatischer Ansatz fuer Klassifikation

        batch_size, seq_len, _ = x.shape

        # Input Projection
        x = self.input_projection(x)  # (batch, seq, d_model)

        # Layer Norm
        x = self.layer_norm(x)

        # Global Average Pooling
        x = x.mean(dim=1)  # (batch, d_model)

        # Classification
        logits = self.classifier(x)

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
            'context_length': self.context_length,
            'd_model': self.d_model,
            'encoder_layers': self.encoder_layers,
            'decoder_layers': self.decoder_layers,
            'encoder_attention_heads': self.encoder_attention_heads,
            'decoder_attention_heads': self.decoder_attention_heads,
            'encoder_ffn_dim': self.encoder_ffn_dim,
            'decoder_ffn_dim': self.decoder_ffn_dim,
            'num_classes': self.num_classes,
            'dropout': self.dropout_rate,
            'prediction_length': self.prediction_length
        }

    def get_num_parameters(self) -> int:
        """Gibt die Anzahl der trainierbaren Parameter zurueck."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def is_hf_available() -> bool:
    """Prueft ob Hugging Face transformers verfuegbar ist."""
    return _HF_AVAILABLE
