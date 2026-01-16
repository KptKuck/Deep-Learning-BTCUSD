"""
Bidirectional LSTM Classifier fuer Trendwechsel-Erkennung.

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


class BiLSTMClassifier(BaseModel):
    """
    Bidirektionaler LSTM Classifier fuer Klassifikation von Zeitreihen.

    Architektur:
    - Multiple LSTM-Layer mit variablen hidden_sizes
    - Projection-Layer zwischen unterschiedlichen Groessen
    - Optionale Layer Normalization nach jedem LSTM
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
        Initialisiert den BiLSTM Classifier.

        Args:
            input_size: Anzahl der Input-Features pro Zeitschritt
            hidden_sizes: Hidden Size pro Layer als Liste [128, 64] oder int (alte API)
            num_layers: Anzahl der Layer (nur fuer Rueckwaertskompatibilitaet, wird ignoriert wenn hidden_sizes Liste ist)
            num_classes: Anzahl der Ausgabeklassen (HOLD=0, BUY=1, SELL=2)
            dropout: Dropout-Rate
            bidirectional: Bidirektionales LSTM verwenden
            use_layer_norm: Layer Normalization nach jedem LSTM Layer
            use_attention: Multi-Head Attention nach letztem LSTM
            use_residual: Residual/Skip Connections (Layer i zu i+2)
            attention_heads: Anzahl der Attention Heads (nur wenn use_attention=True)
        """
        name = 'BiLSTM' if bidirectional else 'LSTM'
        super().__init__(name=f'{name}Classifier')

        # hidden_sizes normalisieren
        if isinstance(hidden_sizes, int):
            # Alte API: hidden_size + num_layers
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

        # LSTM-Layer (ModuleList fuer variable Groessen)
        self.lstm_layers = nn.ModuleList()

        # Projection-Layer (zwischen unterschiedlichen hidden_sizes)
        self.projections = nn.ModuleList()

        # Layer Normalization (optional)
        if use_layer_norm:
            self.layer_norms = nn.ModuleList()
        else:
            self.layer_norms = None

        # Residual/Skip Projections (optional, fuer Layer i zu i+2)
        if use_residual and self.num_layers > 2:
            self.skip_projections = nn.ModuleList()
        else:
            self.skip_projections = None

        # LSTM-Layer erstellen
        for i, hidden_size in enumerate(self.hidden_sizes):
            # Input-Groesse fuer diesen Layer
            if i == 0:
                layer_input_size = input_size
            else:
                # Output des vorherigen Layers
                layer_input_size = self.hidden_sizes[i - 1] * self.num_directions

            # LSTM Layer (einzeln, um unterschiedliche hidden_sizes zu ermoeglichen)
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=layer_input_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                    dropout=0,  # Dropout manuell zwischen Layern
                    bidirectional=bidirectional
                )
            )

            # Projection Layer (falls naechster Layer andere Groesse hat)
            if i < self.num_layers - 1:
                current_out = hidden_size * self.num_directions
                next_in = self.hidden_sizes[i + 1] * self.num_directions

                if current_out != next_in:
                    self.projections.append(nn.Linear(current_out, next_in))
                else:
                    self.projections.append(nn.Identity())
            else:
                # Kein Projection nach letztem Layer
                self.projections.append(None)

            # Layer Normalization
            if use_layer_norm:
                self.layer_norms.append(
                    nn.LayerNorm(hidden_size * self.num_directions)
                )

        # Residual/Skip Projections erstellen (Layer i -> Layer i+2)
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

        # Attention nach letztem LSTM (optional)
        final_hidden_size = self.hidden_sizes[-1] * self.num_directions

        if use_attention:
            # Stelle sicher dass embed_dim durch num_heads teilbar ist
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

        # Fully Connected Layer fuer Klassifikation
        self.fc = nn.Linear(final_hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Vorwaertsdurchlauf.

        Args:
            x: Eingabetensor (batch_size, sequence_length, input_size)

        Returns:
            Logits (batch_size, num_classes)
        """
        # Speichere Outputs fuer Residual Connections
        layer_outputs = []

        # Durch alle LSTM-Layer
        for i, lstm in enumerate(self.lstm_layers):
            # LSTM durchlauf
            lstm_out, (hidden, cell) = lstm(x)

            # Layer Normalization (optional)
            if self.layer_norms is not None:
                lstm_out = self.layer_norms[i](lstm_out)

            # Residual/Skip Connection (von Layer i-2)
            if self.skip_projections is not None and i >= 2:
                skip_idx = i - 2
                residual = self.skip_projections[skip_idx](layer_outputs[skip_idx])
                lstm_out = lstm_out + residual

            # Output speichern fuer spaetere Skip Connections
            layer_outputs.append(lstm_out)

            # Dropout zwischen Layern (nicht nach letztem Layer)
            if i < self.num_layers - 1:
                lstm_out = self.dropout(lstm_out)

                # Projection falls naechster Layer andere Groesse hat
                if self.projections[i] is not None:
                    lstm_out = self.projections[i](lstm_out)

            # Input fuer naechsten Layer
            x = lstm_out

        # Attention (optional, ueber die gesamte Sequenz)
        if self.attention is not None:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            lstm_out = self.attention_norm(lstm_out + attn_out)

        # Letzten Zeitschritt oder Hidden State verwenden
        # Hier: Mean Pooling ueber die Sequenz fuer bessere Repraesentation
        if self.attention is not None:
            # Bei Attention: Mean Pooling
            pooled = lstm_out.mean(dim=1)
        else:
            # Ohne Attention: Letzten Hidden State verwenden
            # (forward + backward konkateniert)
            if self.bidirectional:
                # Letzter Zeitschritt enthaelt bereits forward + backward
                pooled = lstm_out[:, -1, :]
            else:
                pooled = lstm_out[:, -1, :]

        # Dropout vor FC
        pooled = self.dropout(pooled)

        # Klassifikation
        out = self.fc(pooled)

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
