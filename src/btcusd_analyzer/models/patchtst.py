"""
PatchTST (Patch Time Series Transformer) fuer Trendwechsel-Erkennung.

PatchTST teilt die Zeitreihe in Patches auf und verwendet einen Transformer
zur Klassifikation. Dies ist effektiver als Standard-Transformer fuer lange
Sequenzen, da die Patch-Aufteilung lokale Muster besser erfasst.

Paper: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
https://arxiv.org/abs/2211.14730

Falls transformers nicht installiert ist, wird eine Fehlermeldung ausgegeben.
"""

from typing import Optional

import torch
import torch.nn as nn

from btcusd_analyzer.models.base import BaseModel


# Pruefe ob transformers mit PatchTST verfuegbar ist
_PATCHTST_AVAILABLE = False
try:
    from transformers import (
        PatchTSTConfig,
        PatchTSTForClassification,
    )
    _PATCHTST_AVAILABLE = True
except ImportError:
    pass


class PatchTSTClassifier(BaseModel):
    """
    PatchTST Classifier fuer Zeitreihen-Klassifikation.

    PatchTST verwendet Patching (Aufteilung der Zeitreihe in ueberlappende
    Segmente) und Channel-Independence fuer effizientes Lernen.

    Vorteile gegenueber Standard-Transformer:
    - Bessere Erfassung lokaler Muster durch Patches
    - Reduzierte Komplexitaet O(N/P)^2 statt O(N)^2
    - Channel-Independence: Jedes Feature wird separat verarbeitet

    Hinweis: Benoetigt `pip install transformers>=4.36.0`

    Eingabe: (batch_size, sequence_length, input_size)
    Ausgabe: (batch_size, num_classes)
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int = 3,
        context_length: int = 100,
        patch_length: int = 16,
        stride: int = 8,
        d_model: int = 64,
        num_attention_heads: int = 4,
        num_hidden_layers: int = 2,
        ffn_dim: int = 128,
        dropout: float = 0.1,
        head_dropout: float = 0.1,
        pooling_type: str = "mean",
        channel_attention: bool = False,
        use_cls_token: bool = False
    ):
        """
        Initialisiert den PatchTST Classifier.

        Args:
            input_size: Anzahl der Input-Features (Channels)
            num_classes: Anzahl der Ausgabeklassen
            context_length: Laenge der Eingabesequenz
            patch_length: Groesse eines Patches
            stride: Schrittweite zwischen Patches (< patch_length fuer Ueberlappung)
            d_model: Dimension des Transformer-Modells
            num_attention_heads: Anzahl der Attention Heads
            num_hidden_layers: Anzahl der Transformer Encoder Layers
            ffn_dim: Dimension des Feed-Forward Networks
            dropout: Dropout-Rate im Transformer
            head_dropout: Dropout-Rate im Classification Head
            pooling_type: Pooling-Methode ("mean" oder "flatten")
            channel_attention: Channel Attention aktivieren
            use_cls_token: CLS Token verwenden statt Pooling
        """
        super().__init__(name='PatchTSTClassifier')

        if not _PATCHTST_AVAILABLE:
            raise ImportError(
                "PatchTST nicht verfuegbar. "
                "Bitte installieren mit: pip install transformers>=4.36.0"
            )

        self.input_size = input_size
        self.num_classes = num_classes
        self.context_length = context_length
        self.patch_length = patch_length
        self.stride = stride
        self.d_model = d_model
        self.num_hidden_layers = num_hidden_layers
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout
        self.head_dropout = head_dropout
        self.pooling_type = pooling_type
        self.channel_attention = channel_attention
        self.use_cls_token = use_cls_token

        # Attention Heads anpassen: d_model muss durch num_heads teilbar sein
        actual_heads = num_attention_heads
        while d_model % actual_heads != 0 and actual_heads > 1:
            actual_heads -= 1
        self.num_attention_heads = actual_heads

        # PatchTST Konfiguration
        self.config = PatchTSTConfig(
            num_input_channels=input_size,
            context_length=context_length,
            patch_length=patch_length,
            patch_stride=stride,
            d_model=d_model,
            num_attention_heads=actual_heads,
            num_hidden_layers=num_hidden_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            head_dropout=head_dropout,
            pooling_type=pooling_type,
            channel_attention=channel_attention,
            use_cls_token=use_cls_token,
            # Klassifikations-spezifisch
            num_targets=num_classes,
        )

        # PatchTST Modell fuer Klassifikation
        self.model = PatchTSTForClassification(self.config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Vorwaertsdurchlauf.

        Args:
            x: Eingabetensor (batch_size, sequence_length, input_size)

        Returns:
            Logits (batch_size, num_classes)
        """
        # PatchTST erwartet: (batch_size, num_input_channels, context_length)
        # Unsere Eingabe: (batch_size, sequence_length, input_size)
        # Transponieren: input_size -> num_channels, sequence_length -> context_length
        x = x.transpose(1, 2)  # (batch, input_size, seq_len)

        # Sequenz auf context_length anpassen falls noetig
        seq_len = x.shape[2]
        if seq_len > self.context_length:
            # Nur die letzten context_length Werte nehmen
            x = x[:, :, -self.context_length:]
        elif seq_len < self.context_length:
            # Mit Nullen auffuellen (links padding)
            padding = torch.zeros(
                x.shape[0], x.shape[1], self.context_length - seq_len,
                device=x.device, dtype=x.dtype
            )
            x = torch.cat([padding, x], dim=2)

        # PatchTST Vorwaertsdurchlauf
        outputs = self.model(past_values=x)

        # Logits extrahieren
        logits = outputs.prediction_logits

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
            'num_classes': self.num_classes,
            'context_length': self.context_length,
            'patch_length': self.patch_length,
            'stride': self.stride,
            'd_model': self.d_model,
            'num_attention_heads': self.num_attention_heads,
            'num_hidden_layers': self.num_hidden_layers,
            'ffn_dim': self.ffn_dim,
            'dropout': self.dropout_rate,
            'head_dropout': self.head_dropout,
            'pooling_type': self.pooling_type,
            'channel_attention': self.channel_attention,
            'use_cls_token': self.use_cls_token
        }

    def get_num_parameters(self) -> int:
        """Gibt die Anzahl der trainierbaren Parameter zurueck."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def is_patchtst_available() -> bool:
    """Prueft ob PatchTST verfuegbar ist."""
    return _PATCHTST_AVAILABLE
