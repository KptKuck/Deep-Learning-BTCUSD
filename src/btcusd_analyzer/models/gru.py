"""
GRU Classifier fuer Trendwechsel-Erkennung.

Erweiterte Version mit:
- Variable hidden_sizes pro Layer
- Layer Normalization
- Multi-Head Attention
- Residual/Skip Connections
"""

from typing import List, Optional, Union

import torch
import torch.nn as nn

from btcusd_analyzer.models.base import BaseModel


class GRUClassifier(BaseModel):
    """
    GRU (Gated Recurrent Unit) Classifier fuer Klassifikation von Zeitreihen.

    GRU ist schneller als LSTM und hat weniger Parameter, aber oft aehnliche Performance.

    Architektur:
    - Multiple GRU-Layer mit variablen hidden_sizes
    - Projection-Layer zwischen unterschiedlichen Groessen
    - Optionale Layer Normalization nach jedem GRU
    - Optionaler Multi-Head Attention Mechanismus
    - Optionale Residual/Skip Connections
    - Dropout fuer Regularisierung
    - Fully Connected Layer fuer Klassifikation

    Eingabe: (batch_size, sequence_length, input_size)
    Ausgabe: (batch_size, num_classes)
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Union[int, List[int]] = 128,
        num_layers: Optional[int] = None,
        num_classes: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_layer_norm: bool = False,
        use_attention: bool = False,
        use_residual: bool = False,
        attention_heads: int = 4
    ):
        """
        Initialisiert den GRU Classifier.

        Args:
            input_size: Anzahl der Input-Features pro Zeitschritt
            hidden_sizes: Hidden Size pro Layer als Liste [128, 64] oder int (alte API)
            num_layers: Anzahl der Layer (nur fuer Rueckwaertskompatibilitaet)
            num_classes: Anzahl der Ausgabeklassen (HOLD=0, BUY=1, SELL=2)
            dropout: Dropout-Rate
            bidirectional: Bidirektionales GRU verwenden
            use_layer_norm: Layer Normalization nach jedem GRU Layer
            use_attention: Multi-Head Attention nach letztem GRU
            use_residual: Residual/Skip Connections (Layer i zu i+2)
            attention_heads: Anzahl der Attention Heads (nur wenn use_attention=True)
        """
        name = 'BiGRU' if bidirectional else 'GRU'
        super().__init__(name=f'{name}Classifier')

        # hidden_sizes normalisieren
        if isinstance(hidden_sizes, int):
            n_layers = num_layers if num_layers is not None else 2
            self.hidden_sizes = [hidden_sizes] * n_layers
        else:
            self.hidden_sizes = list(hidden_sizes)

        self.input_size = input_size
        self.num_layers = len(self.hidden_sizes)
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.use_layer_norm = use_layer_norm
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.attention_heads = attention_heads

        # GRU-Layer (ModuleList fuer variable Groessen)
        self.gru_layers = nn.ModuleList()

        # Layer Normalization (optional)
        if use_layer_norm:
            self.layer_norms = nn.ModuleList()
        else:
            self.layer_norms = None

        # Residual/Skip Projections (optional)
        if use_residual and self.num_layers > 2:
            self.skip_projections = nn.ModuleList()
        else:
            self.skip_projections = None

        # GRU-Layer erstellen
        for i, hidden_size in enumerate(self.hidden_sizes):
            # Input-Groesse fuer diesen Layer
            if i == 0:
                layer_input_size = input_size
            else:
                layer_input_size = self.hidden_sizes[i - 1] * self.num_directions

            # GRU Layer
            self.gru_layers.append(
                nn.GRU(
                    input_size=layer_input_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                    dropout=0,
                    bidirectional=bidirectional
                )
            )

            # Layer Normalization
            if use_layer_norm:
                self.layer_norms.append(
                    nn.LayerNorm(hidden_size * self.num_directions)
                )

        # Residual/Skip Projections
        if self.skip_projections is not None:
            for i in range(self.num_layers - 2):
                source_size = self.hidden_sizes[i] * self.num_directions
                target_size = self.hidden_sizes[i + 2] * self.num_directions

                if source_size != target_size:
                    self.skip_projections.append(nn.Linear(source_size, target_size))
                else:
                    self.skip_projections.append(nn.Identity())

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Attention (optional)
        final_hidden_size = self.hidden_sizes[-1] * self.num_directions

        if use_attention:
            actual_heads = attention_heads
            while final_hidden_size % actual_heads != 0 and actual_heads > 1:
                actual_heads -= 1
            self.actual_attention_heads = actual_heads

            self.attention = nn.MultiheadAttention(
                embed_dim=final_hidden_size,
                num_heads=actual_heads,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(final_hidden_size)
        else:
            self.attention = None
            self.attention_norm = None

        # Fully Connected Layer
        self.fc = nn.Linear(final_hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Vorwaertsdurchlauf.

        Args:
            x: Eingabetensor (batch_size, sequence_length, input_size)

        Returns:
            Logits (batch_size, num_classes)
        """
        layer_outputs = []

        for i, gru in enumerate(self.gru_layers):
            # GRU durchlauf
            gru_out, hidden = gru(x)

            # Layer Normalization
            if self.layer_norms is not None:
                gru_out = self.layer_norms[i](gru_out)

            # Residual/Skip Connection
            if self.skip_projections is not None and i >= 2:
                skip_idx = i - 2
                residual = self.skip_projections[skip_idx](layer_outputs[skip_idx])
                gru_out = gru_out + residual

            layer_outputs.append(gru_out)

            # Dropout zwischen Layern
            if i < self.num_layers - 1:
                gru_out = self.dropout(gru_out)

            x = gru_out

        # Attention
        if self.attention is not None:
            attn_out, _ = self.attention(gru_out, gru_out, gru_out)
            gru_out = self.attention_norm(gru_out + attn_out)

        # Pooling
        if self.attention is not None:
            pooled = gru_out.mean(dim=1)
        else:
            pooled = gru_out[:, -1, :]

        # Dropout + FC
        pooled = self.dropout(pooled)
        out = self.fc(pooled)

        return out

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
            'hidden_sizes': self.hidden_sizes,
            'num_layers': self.num_layers,
            'num_classes': self.num_classes,
            'dropout': self.dropout_rate,
            'bidirectional': self.bidirectional,
            'use_layer_norm': self.use_layer_norm,
            'use_attention': self.use_attention,
            'use_residual': self.use_residual,
            'attention_heads': self.attention_heads
        }

    def get_num_parameters(self) -> int:
        """Gibt die Anzahl der trainierbaren Parameter zurueck."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
