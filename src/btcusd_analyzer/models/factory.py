"""
Model Factory - Erstellt Modelle basierend auf Namen.

Erweiterte Version mit:
- Unterstuetzung fuer variable hidden_sizes
- Neue Modelle: Transformer, HF-Transformer
- Preset-Definitionen fuer schnelle Konfiguration
"""

from typing import Any, Dict, List, Optional, Union

from btcusd_analyzer.models.bilstm import BiLSTMClassifier
from btcusd_analyzer.models.gru import GRUClassifier
from btcusd_analyzer.models.cnn import CNNClassifier
from btcusd_analyzer.models.cnn_lstm import CNNLSTMClassifier
from btcusd_analyzer.models.transformer import TransformerClassifier
from btcusd_analyzer.models.hf_transformer import HFTimeSeriesClassifier, is_hf_available
from btcusd_analyzer.models.patchtst import PatchTSTClassifier, is_patchtst_available


# Preset-Definitionen fuer LSTM/GRU Architekturen
LSTM_PRESETS = {
    'Standard [128, 128]': {'hidden_sizes': [128, 128]},
    'Pyramid [256, 128, 64]': {'hidden_sizes': [256, 128, 64]},
    'Inverted [64, 128, 256]': {'hidden_sizes': [64, 128, 256]},
    'Bottleneck [256, 64, 256]': {'hidden_sizes': [256, 64, 256]},
    'Deep [512, 256, 128, 64]': {'hidden_sizes': [512, 256, 128, 64]},
    'Small [64, 64]': {'hidden_sizes': [64, 64]},
    'Large [256, 256]': {'hidden_sizes': [256, 256]},
    'Custom': None  # Fuer manuelle Eingabe
}

# Preset-Definitionen fuer Transformer
TRANSFORMER_PRESETS = {
    'Small (d=64, h=4, L=2)': {'d_model': 64, 'nhead': 4, 'num_encoder_layers': 2, 'dim_feedforward': 128},
    'Medium (d=128, h=4, L=4)': {'d_model': 128, 'nhead': 4, 'num_encoder_layers': 4, 'dim_feedforward': 256},
    'Large (d=256, h=8, L=6)': {'d_model': 256, 'nhead': 8, 'num_encoder_layers': 6, 'dim_feedforward': 512},
    'XL (d=512, h=8, L=8)': {'d_model': 512, 'nhead': 8, 'num_encoder_layers': 8, 'dim_feedforward': 1024},
    'Custom': None
}

# Preset-Definitionen fuer CNN
CNN_PRESETS = {
    'Small (32 Filter)': {'num_filters': 32, 'num_conv_layers': 2},
    'Medium (64 Filter)': {'num_filters': 64, 'num_conv_layers': 3},
    'Large (128 Filter)': {'num_filters': 128, 'num_conv_layers': 4},
    'Custom': None
}

# Preset-Definitionen fuer PatchTST
PATCHTST_PRESETS = {
    'Small (d=32, L=2)': {'d_model': 32, 'num_hidden_layers': 2, 'num_attention_heads': 2, 'ffn_dim': 64},
    'Medium (d=64, L=3)': {'d_model': 64, 'num_hidden_layers': 3, 'num_attention_heads': 4, 'ffn_dim': 128},
    'Large (d=128, L=4)': {'d_model': 128, 'num_hidden_layers': 4, 'num_attention_heads': 4, 'ffn_dim': 256},
    'XL (d=256, L=6)': {'d_model': 256, 'num_hidden_layers': 6, 'num_attention_heads': 8, 'ffn_dim': 512},
    'Custom': None
}


class ModelFactory:
    """Factory-Klasse zum Erstellen von Modellen."""

    @staticmethod
    def create(
        model_name: str,
        input_size: int,
        hidden_sizes: Optional[Union[int, List[int]]] = None,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        # Neue Parameter fuer erweiterte Features
        use_layer_norm: bool = False,
        use_attention: bool = False,
        use_residual: bool = False,
        attention_heads: int = 4,
        # Transformer-spezifische Parameter
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 256,
        # CNN-spezifische Parameter
        num_filters: int = 64,
        kernel_size: int = 3,
        **kwargs
    ):
        """
        Erstellt ein Modell basierend auf dem Namen.

        Args:
            model_name: Name des Modells ('bilstm', 'bigru', 'cnn', 'transformer', etc.)
            input_size: Anzahl der Input-Features
            hidden_sizes: Liste der hidden_sizes pro Layer [256, 128, 64] (neu)
            hidden_size: Einzelne hidden_size (fuer Rueckwaertskompatibilitaet)
            num_layers: Anzahl der Layer (fuer Rueckwaertskompatibilitaet)
            num_classes: Anzahl der Ausgabeklassen (default: 3 fuer BUY/SELL/HOLD)
            dropout: Dropout-Rate
            use_layer_norm: Layer Normalization aktivieren
            use_attention: Attention-Mechanismus aktivieren
            use_residual: Residual/Skip Connections aktivieren
            attention_heads: Anzahl der Attention Heads
            d_model: Transformer Model Dimension
            nhead: Transformer Attention Heads
            num_encoder_layers: Anzahl Transformer Encoder Layers
            dim_feedforward: Transformer FFN Dimension
            num_filters: CNN Filter Anzahl
            kernel_size: CNN Kernel Groesse
            **kwargs: Weitere modellspezifische Parameter

        Returns:
            Initialisiertes Modell

        Raises:
            ValueError: Wenn model_name unbekannt ist
        """
        model_name = model_name.lower().strip()

        # hidden_sizes normalisieren
        if hidden_sizes is None:
            # Fallback auf alte API
            hidden_sizes = hidden_size

        # LSTM Varianten
        if model_name in ['bilstm', 'bi_lstm', 'bilstmclassifier']:
            return BiLSTMClassifier(
                input_size=input_size,
                hidden_sizes=hidden_sizes,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=dropout,
                bidirectional=True,
                use_layer_norm=use_layer_norm,
                use_attention=use_attention,
                use_residual=use_residual,
                attention_heads=attention_heads
            )
        elif model_name in ['lstm']:
            return BiLSTMClassifier(
                input_size=input_size,
                hidden_sizes=hidden_sizes,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=dropout,
                bidirectional=False,
                use_layer_norm=use_layer_norm,
                use_attention=use_attention,
                use_residual=use_residual,
                attention_heads=attention_heads
            )

        # GRU Varianten
        elif model_name in ['bigru', 'bi_gru']:
            return GRUClassifier(
                input_size=input_size,
                hidden_sizes=hidden_sizes,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=dropout,
                bidirectional=True,
                use_layer_norm=use_layer_norm,
                use_attention=use_attention,
                use_residual=use_residual,
                attention_heads=attention_heads
            )
        elif model_name in ['gru']:
            return GRUClassifier(
                input_size=input_size,
                hidden_sizes=hidden_sizes,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=dropout,
                bidirectional=False,
                use_layer_norm=use_layer_norm,
                use_attention=use_attention,
                use_residual=use_residual,
                attention_heads=attention_heads
            )

        # Transformer (PyTorch nativ)
        elif model_name in ['transformer', 'transformerclassifier']:
            return TransformerClassifier(
                input_size=input_size,
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                dim_feedforward=dim_feedforward,
                num_classes=num_classes,
                dropout=dropout,
                max_seq_length=kwargs.get('max_seq_length', 500)
            )

        # HuggingFace Transformer
        elif model_name in ['hf-transformer', 'hf_transformer', 'huggingface']:
            if not is_hf_available():
                raise ImportError(
                    "Hugging Face transformers nicht installiert. "
                    "Bitte installieren mit: pip install transformers"
                )
            return HFTimeSeriesClassifier(
                input_size=input_size,
                context_length=kwargs.get('context_length', 50),
                d_model=d_model,
                encoder_layers=num_encoder_layers,
                num_classes=num_classes,
                dropout=dropout
            )

        # CNN
        elif model_name in ['cnn', '1dcnn']:
            return CNNClassifier(
                input_size=input_size,
                num_filters=num_filters if num_filters != 64 else hidden_size,
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
                num_filters=num_filters // 2 if num_filters != 64 else hidden_size // 2,
                kernel_size=kernel_size,
                num_conv_layers=num_conv_layers,
                hidden_size=hidden_size,
                num_lstm_layers=num_lstm_layers,
                num_classes=num_classes,
                dropout=dropout,
                bidirectional=bidirectional
            )

        # PatchTST
        elif model_name in ['patchtst', 'patch_tst', 'patch-tst']:
            if not is_patchtst_available():
                raise ImportError(
                    "PatchTST nicht verfuegbar. "
                    "Bitte installieren mit: pip install transformers>=4.36.0"
                )
            return PatchTSTClassifier(
                input_size=input_size,
                num_classes=num_classes,
                context_length=kwargs.get('context_length', 100),
                patch_length=kwargs.get('patch_length', 16),
                stride=kwargs.get('stride', 8),
                d_model=d_model if d_model != 128 else kwargs.get('d_model', 64),
                num_attention_heads=kwargs.get('num_attention_heads', nhead),
                num_hidden_layers=kwargs.get('num_hidden_layers', num_encoder_layers),
                ffn_dim=kwargs.get('ffn_dim', dim_feedforward // 2),
                dropout=dropout,
                head_dropout=kwargs.get('head_dropout', dropout),
                pooling_type=kwargs.get('pooling_type', 'mean'),
                channel_attention=kwargs.get('channel_attention', False),
                use_cls_token=kwargs.get('use_cls_token', False)
            )

        else:
            raise ValueError(
                f"Unbekanntes Modell: '{model_name}'. "
                f"Verfuegbare Modelle: {', '.join(ModelFactory.get_available_models())}"
            )

    @staticmethod
    def get_available_models() -> List[str]:
        """Gibt Liste der verfuegbaren Modellnamen zurueck."""
        models = [
            'BiLSTM', 'LSTM',
            'BiGRU', 'GRU',
            'CNN',
            'CNN-LSTM',
            'Transformer'
        ]

        # HF-Transformer nur wenn verfuegbar
        if is_hf_available():
            models.append('HF-Transformer')

        # PatchTST nur wenn verfuegbar
        if is_patchtst_available():
            models.append('PatchTST')

        return models

    @staticmethod
    def get_presets(model_name: str) -> Dict[str, Optional[Dict]]:
        """
        Gibt Presets fuer ein bestimmtes Modell zurueck.

        Args:
            model_name: Name des Modells

        Returns:
            Dictionary mit Preset-Namen und Parametern
        """
        model_name = model_name.lower().strip()

        if model_name in ['bilstm', 'lstm', 'bigru', 'gru', 'bi_lstm', 'bi_gru']:
            return LSTM_PRESETS
        elif model_name in ['transformer', 'hf-transformer', 'hf_transformer']:
            return TRANSFORMER_PRESETS
        elif model_name in ['cnn', '1dcnn']:
            return CNN_PRESETS
        elif model_name in ['patchtst', 'patch_tst', 'patch-tst']:
            return PATCHTST_PRESETS
        else:
            return {'Custom': None}

    @staticmethod
    def get_default_params(model_name: str) -> Dict[str, Any]:
        """
        Gibt Standard-Parameter fuer ein Modell zurueck.

        Args:
            model_name: Name des Modells

        Returns:
            Dictionary mit Standard-Parametern
        """
        model_name = model_name.lower().strip()

        base_params = {
            'num_classes': 3,
            'dropout': 0.3
        }

        if model_name in ['bilstm', 'lstm', 'bigru', 'gru', 'bi_lstm', 'bi_gru']:
            return {
                **base_params,
                'hidden_sizes': [128, 128],
                'use_layer_norm': False,
                'use_attention': False,
                'use_residual': False,
                'attention_heads': 4
            }
        elif model_name in ['transformer']:
            return {
                **base_params,
                'd_model': 128,
                'nhead': 4,
                'num_encoder_layers': 2,
                'dim_feedforward': 256,
                'dropout': 0.1
            }
        elif model_name in ['hf-transformer', 'hf_transformer']:
            return {
                **base_params,
                'd_model': 64,
                'encoder_layers': 2,
                'context_length': 50,
                'dropout': 0.1
            }
        elif model_name in ['cnn', '1dcnn']:
            return {
                **base_params,
                'num_filters': 64,
                'kernel_size': 3,
                'num_layers': 3
            }
        elif model_name in ['cnn-lstm', 'cnn_lstm']:
            return {
                **base_params,
                'hidden_size': 128,
                'num_filters': 64,
                'num_conv_layers': 2,
                'num_lstm_layers': 2,
                'bidirectional': True
            }
        elif model_name in ['patchtst', 'patch_tst', 'patch-tst']:
            return {
                **base_params,
                'context_length': 100,
                'patch_length': 16,
                'stride': 8,
                'd_model': 64,
                'num_attention_heads': 4,
                'num_hidden_layers': 2,
                'ffn_dim': 128,
                'dropout': 0.1,
                'pooling_type': 'mean'
            }
        else:
            return base_params

    @staticmethod
    def model_requires_hidden_sizes(model_name: str) -> bool:
        """Prueft ob ein Modell hidden_sizes Parameter unterstuetzt."""
        model_name = model_name.lower().strip()
        return model_name in ['bilstm', 'lstm', 'bigru', 'gru', 'bi_lstm', 'bi_gru']

    @staticmethod
    def model_is_transformer(model_name: str) -> bool:
        """Prueft ob ein Modell ein Transformer ist."""
        model_name = model_name.lower().strip()
        return model_name in ['transformer', 'hf-transformer', 'hf_transformer', 'huggingface', 'patchtst', 'patch_tst', 'patch-tst']
