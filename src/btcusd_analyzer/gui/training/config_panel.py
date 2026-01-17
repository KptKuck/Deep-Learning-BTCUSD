"""
Config Panel - Modell- und Training-Konfiguration mit kontextabhaengiger UI.

Zeigt nur relevante Parameter basierend auf dem ausgewaehlten Modell.
"""

from typing import Dict, Any, Optional, List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QHBoxLayout,
    QGroupBox, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QLineEdit, QScrollArea, QSlider, QProgressBar, QFileDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer

import torch

# GPU Monitoring (optional) - nvidia-ml-py bevorzugt, pynvml als Fallback
import warnings
PYNVML_AVAILABLE = False
pynvml = None
try:
    # nvidia-ml-py direkt importieren (neuer Name)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        import pynvml as _pynvml
        pynvml = _pynvml
        PYNVML_AVAILABLE = True
except ImportError:
    pass

from ..styles import COLORS
from ...models.factory import (
    ModelFactory, LSTM_PRESETS, TRANSFORMER_PRESETS, CNN_PRESETS, PATCHTST_PRESETS
)


class ConfigPanel(QWidget):
    """
    Konfigurations-Panel fuer Modell und Training.

    Zeigt kontextabhaengig nur die Parameter, die fuer das
    ausgewaehlte Modell relevant sind.
    """

    # Signals
    model_changed = pyqtSignal(str)  # Modell-Typ geaendert
    start_training = pyqtSignal()
    stop_training = pyqtSignal()
    start_auto_training = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()

        # Initial UI-Status setzen
        self._on_model_changed(self.model_combo.currentText())

    def _setup_ui(self):
        """Erstellt die UI-Komponenten."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(5)

        # Modell-Auswahl GroupBox
        self._create_model_group(layout)

        # Modell-spezifische Parameter (dynamisch)
        self._create_lstm_params_group(layout)
        self._create_transformer_params_group(layout)
        self._create_cnn_params_group(layout)
        self._create_patchtst_params_group(layout)

        # Erweiterte LSTM/GRU Optionen
        self._create_advanced_group(layout)

        # Training-Parameter
        self._create_training_group(layout)

        # Optionen (Early Stopping, Speichern)
        self._create_options_group(layout)

        # Device Info
        self._create_device_group(layout)

        # Auto-Trainer
        self._create_auto_trainer_group(layout)

        # Buttons
        self._create_buttons(layout)

        layout.addStretch()
        scroll.setWidget(panel)

        # Scroll-Widget als Hauptlayout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

    def _create_model_group(self, layout: QVBoxLayout):
        """Erstellt die Modell-Auswahl GroupBox."""
        group = QGroupBox("Modell")
        grid = QGridLayout(group)

        # Architektur
        grid.addWidget(QLabel("Architektur:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(ModelFactory.get_available_models())
        grid.addWidget(self.model_combo, 0, 1)

        # Preset
        grid.addWidget(QLabel("Preset:"), 1, 0)
        self.preset_combo = QComboBox()
        grid.addWidget(self.preset_combo, 1, 1)

        # Dropout (gemeinsam fuer alle Modelle)
        grid.addWidget(QLabel("Dropout:"), 2, 0)
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.8)
        self.dropout_spin.setValue(0.2)
        self.dropout_spin.setSingleStep(0.05)
        grid.addWidget(self.dropout_spin, 2, 1)

        layout.addWidget(group)

    def _create_lstm_params_group(self, layout: QVBoxLayout):
        """Erstellt LSTM/GRU Parameter GroupBox."""
        self.lstm_group = QGroupBox("LSTM/GRU Parameter")
        grid = QGridLayout(self.lstm_group)

        # Hidden Sizes
        grid.addWidget(QLabel("Hidden Sizes:"), 0, 0)
        self.hidden_sizes_edit = QLineEdit("128, 128")
        self.hidden_sizes_edit.setToolTip("Komma-getrennte Liste: z.B. 256, 128, 64")
        grid.addWidget(self.hidden_sizes_edit, 0, 1)

        layout.addWidget(self.lstm_group)

    def _create_transformer_params_group(self, layout: QVBoxLayout):
        """Erstellt Transformer Parameter GroupBox."""
        self.transformer_group = QGroupBox("Transformer Parameter")
        grid = QGridLayout(self.transformer_group)

        # d_model
        grid.addWidget(QLabel("d_model:"), 0, 0)
        self.d_model_spin = QSpinBox()
        self.d_model_spin.setRange(32, 1024)
        self.d_model_spin.setValue(128)
        self.d_model_spin.setSingleStep(32)
        grid.addWidget(self.d_model_spin, 0, 1)

        # Attention Heads
        grid.addWidget(QLabel("Attention Heads:"), 1, 0)
        self.nhead_spin = QSpinBox()
        self.nhead_spin.setRange(1, 16)
        self.nhead_spin.setValue(4)
        grid.addWidget(self.nhead_spin, 1, 1)

        # Encoder Layers
        grid.addWidget(QLabel("Encoder Layers:"), 2, 0)
        self.encoder_layers_spin = QSpinBox()
        self.encoder_layers_spin.setRange(1, 12)
        self.encoder_layers_spin.setValue(2)
        grid.addWidget(self.encoder_layers_spin, 2, 1)

        layout.addWidget(self.transformer_group)

    def _create_cnn_params_group(self, layout: QVBoxLayout):
        """Erstellt CNN Parameter GroupBox."""
        self.cnn_group = QGroupBox("CNN Parameter")
        grid = QGridLayout(self.cnn_group)

        # Num Filters
        grid.addWidget(QLabel("Filter:"), 0, 0)
        self.num_filters_spin = QSpinBox()
        self.num_filters_spin.setRange(16, 512)
        self.num_filters_spin.setValue(64)
        self.num_filters_spin.setSingleStep(16)
        grid.addWidget(self.num_filters_spin, 0, 1)

        # Conv Layers
        grid.addWidget(QLabel("Conv Layers:"), 1, 0)
        self.num_conv_layers_spin = QSpinBox()
        self.num_conv_layers_spin.setRange(1, 6)
        self.num_conv_layers_spin.setValue(2)
        grid.addWidget(self.num_conv_layers_spin, 1, 1)

        # Hidden Size (fuer CNN-LSTM)
        grid.addWidget(QLabel("Hidden Size:"), 2, 0)
        self.cnn_hidden_spin = QSpinBox()
        self.cnn_hidden_spin.setRange(16, 512)
        self.cnn_hidden_spin.setValue(128)
        self.cnn_hidden_spin.setSingleStep(16)
        grid.addWidget(self.cnn_hidden_spin, 2, 1)

        layout.addWidget(self.cnn_group)

    def _create_patchtst_params_group(self, layout: QVBoxLayout):
        """Erstellt PatchTST Parameter GroupBox."""
        self.patchtst_group = QGroupBox("PatchTST Parameter")
        grid = QGridLayout(self.patchtst_group)

        # Context Length
        grid.addWidget(QLabel("Context Length:"), 0, 0)
        self.context_length_spin = QSpinBox()
        self.context_length_spin.setRange(16, 512)
        self.context_length_spin.setValue(100)
        self.context_length_spin.setSingleStep(10)
        self.context_length_spin.setToolTip("Laenge der Eingabesequenz")
        grid.addWidget(self.context_length_spin, 0, 1)

        # Patch Length
        grid.addWidget(QLabel("Patch Length:"), 1, 0)
        self.patch_length_spin = QSpinBox()
        self.patch_length_spin.setRange(4, 64)
        self.patch_length_spin.setValue(16)
        self.patch_length_spin.setToolTip("Groesse eines Patches")
        grid.addWidget(self.patch_length_spin, 1, 1)

        # Stride
        grid.addWidget(QLabel("Stride:"), 2, 0)
        self.stride_spin = QSpinBox()
        self.stride_spin.setRange(1, 32)
        self.stride_spin.setValue(8)
        self.stride_spin.setToolTip("Schrittweite zwischen Patches")
        grid.addWidget(self.stride_spin, 2, 1)

        # FFN Dim
        grid.addWidget(QLabel("FFN Dim:"), 3, 0)
        self.ffn_dim_spin = QSpinBox()
        self.ffn_dim_spin.setRange(32, 1024)
        self.ffn_dim_spin.setValue(128)
        self.ffn_dim_spin.setSingleStep(32)
        grid.addWidget(self.ffn_dim_spin, 3, 1)

        layout.addWidget(self.patchtst_group)

    def _create_advanced_group(self, layout: QVBoxLayout):
        """Erstellt erweiterte LSTM/GRU Optionen."""
        self.advanced_group = QGroupBox("Erweiterte Architektur")
        grid = QGridLayout(self.advanced_group)

        self.use_layer_norm_check = QCheckBox("Layer Normalization")
        self.use_layer_norm_check.setToolTip("Normalisierung nach jedem LSTM/GRU Layer")
        grid.addWidget(self.use_layer_norm_check, 0, 0, 1, 2)

        self.use_attention_check = QCheckBox("Attention")
        self.use_attention_check.setToolTip("Self-Attention nach letztem LSTM/GRU")
        grid.addWidget(self.use_attention_check, 1, 0)

        self.attention_heads_spin = QSpinBox()
        self.attention_heads_spin.setRange(1, 16)
        self.attention_heads_spin.setValue(4)
        grid.addWidget(self.attention_heads_spin, 1, 1)

        self.use_residual_check = QCheckBox("Residual Connections")
        self.use_residual_check.setToolTip("Skip-Connections zwischen Layern")
        grid.addWidget(self.use_residual_check, 2, 0, 1, 2)

        layout.addWidget(self.advanced_group)

    def _create_training_group(self, layout: QVBoxLayout):
        """Erstellt Training-Parameter GroupBox."""
        group = QGroupBox("Training")
        grid = QGridLayout(group)

        # Epochen
        grid.addWidget(QLabel("Epochen:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(100)
        grid.addWidget(self.epochs_spin, 0, 1)

        # Learning Rate
        grid.addWidget(QLabel("Learning Rate:"), 1, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.1)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.0001)
        grid.addWidget(self.lr_spin, 1, 1)

        # Batch Size
        grid.addWidget(QLabel("Batch Size:"), 2, 0)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(8, 512)
        self.batch_size_spin.setValue(32)
        self.batch_size_spin.setSingleStep(8)
        grid.addWidget(self.batch_size_spin, 2, 1)

        layout.addWidget(group)

    def _create_options_group(self, layout: QVBoxLayout):
        """Erstellt Optionen GroupBox."""
        group = QGroupBox("Optionen")
        grid = QGridLayout(group)
        grid.setVerticalSpacing(3)

        # Early Stopping
        self.early_stopping_check = QCheckBox("Early Stopping")
        self.early_stopping_check.setChecked(True)
        grid.addWidget(self.early_stopping_check, 0, 0)

        grid.addWidget(QLabel("Patience:"), 0, 1)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 50)
        self.patience_spin.setValue(10)
        grid.addWidget(self.patience_spin, 0, 2)

        # Speicher-Optionen
        self.save_best_check = QCheckBox("Modell speichern")
        self.save_best_check.setChecked(True)
        grid.addWidget(self.save_best_check, 1, 0, 1, 3)

        self.save_history_check = QCheckBox("History speichern")
        self.save_history_check.setChecked(True)
        grid.addWidget(self.save_history_check, 2, 0, 1, 3)

        # Sync Training
        self.sync_training_check = QCheckBox("Sync Training (GPU-stabil)")
        self.sync_training_check.setChecked(True)
        self.sync_training_check.setToolTip("Training im Main-Thread - stabiler fuer GPU")
        grid.addWidget(self.sync_training_check, 3, 0, 1, 3)

        # Speicherpfad
        grid.addWidget(QLabel("Pfad:"), 4, 0)
        self.save_path_label = QLabel("models/")
        self.save_path_label.setStyleSheet("font-size: 9px; color: #aaaaaa;")
        grid.addWidget(self.save_path_label, 4, 1)
        self.save_path_btn = QPushButton("...")
        self.save_path_btn.setFixedWidth(30)
        self.save_path_btn.clicked.connect(self._select_save_path)
        grid.addWidget(self.save_path_btn, 4, 2)

        layout.addWidget(group)

    def _create_device_group(self, layout: QVBoxLayout):
        """Erstellt Device Info GroupBox."""
        group = QGroupBox("Device")
        grid = QGridLayout(group)
        grid.setVerticalSpacing(3)

        # GPU/CPU Switch
        grid.addWidget(QLabel("GPU:"), 0, 0)
        self.use_gpu_check = QCheckBox("Aktiv")
        self.use_gpu_check.setChecked(torch.cuda.is_available())
        self.use_gpu_check.setEnabled(torch.cuda.is_available())
        grid.addWidget(self.use_gpu_check, 0, 1)

        # GPU Name
        device_text = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        color = COLORS['success'] if torch.cuda.is_available() else COLORS['warning']
        self.device_label = QLabel(device_text)
        self.device_label.setStyleSheet(f"color: {color}; font-size: 10px;")
        grid.addWidget(self.device_label, 1, 0, 1, 2)

        # GPU Memory + Utilization
        if torch.cuda.is_available():
            # NVML initialisieren fuer GPU-Auslastung
            self._nvml_initialized = False
            self._nvml_handle = None
            if PYNVML_AVAILABLE and pynvml is not None:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=FutureWarning)
                        pynvml.nvmlInit()
                        self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        self._nvml_initialized = True
                except Exception:
                    pass

            # GPU Memory Bar
            self.gpu_memory_bar = QProgressBar()
            self.gpu_memory_bar.setMinimum(0)
            self.gpu_memory_bar.setMaximum(100)
            self.gpu_memory_bar.setValue(0)
            self.gpu_memory_bar.setTextVisible(True)
            self.gpu_memory_bar.setFormat("%p%")
            self.gpu_memory_bar.setMaximumHeight(18)
            grid.addWidget(self.gpu_memory_bar, 2, 0, 1, 2)

            mem_layout = QHBoxLayout()
            self.gpu_used_label = QLabel("Belegt: - GB")
            self.gpu_used_label.setStyleSheet("color: #aaaaaa; font-size: 9px;")
            mem_layout.addWidget(self.gpu_used_label)

            self.gpu_free_label = QLabel("Frei: - GB")
            self.gpu_free_label.setStyleSheet(f"color: {COLORS['success']}; font-size: 9px;")
            mem_layout.addWidget(self.gpu_free_label)
            grid.addLayout(mem_layout, 3, 0, 1, 2)

            # GPU Utilization (Auslastung)
            grid.addWidget(QLabel("Auslastung:"), 4, 0)
            self.gpu_util_bar = QProgressBar()
            self.gpu_util_bar.setMinimum(0)
            self.gpu_util_bar.setMaximum(100)
            self.gpu_util_bar.setValue(0)
            self.gpu_util_bar.setTextVisible(True)
            self.gpu_util_bar.setFormat("%p%")
            self.gpu_util_bar.setMaximumHeight(18)
            self.gpu_util_bar.setToolTip("GPU-Rechenauslastung (benoetigt pynvml)")
            grid.addWidget(self.gpu_util_bar, 4, 1)

            # GPU Memory Timer
            self.gpu_timer = QTimer()
            self.gpu_timer.timeout.connect(self._update_gpu_stats)
            self.gpu_timer.start(500)  # 500ms fuer fluessigerere Anzeige
            self._update_gpu_stats()

            # GPU Test Button
            self.gpu_test_btn = QPushButton("GPU Test")
            self.gpu_test_btn.setToolTip("Testet GPU-Training mit synthetischen Daten")
            grid.addWidget(self.gpu_test_btn, 5, 0, 1, 2)
        else:
            self.gpu_memory_bar = None
            self.gpu_used_label = None
            self.gpu_free_label = None
            self.gpu_util_bar = None
            self.gpu_test_btn = None
            self._nvml_initialized = False
            self._nvml_handle = None

        layout.addWidget(group)

    def _create_auto_trainer_group(self, layout: QVBoxLayout):
        """Erstellt Auto-Trainer GroupBox."""
        group = QGroupBox("Auto-Trainer")
        grid = QGridLayout(group)

        self.auto_trainer_check = QCheckBox("Auto-Modus aktivieren")
        self.auto_trainer_check.setToolTip("Automatisch verschiedene Modelle und Parameter testen")
        grid.addWidget(self.auto_trainer_check, 0, 0, 1, 2)

        grid.addWidget(QLabel("Komplexitaet:"), 1, 0)
        self.complexity_slider = QSlider(Qt.Orientation.Horizontal)
        self.complexity_slider.setRange(1, 5)
        self.complexity_slider.setValue(3)
        self.complexity_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.complexity_slider.setTickInterval(1)
        grid.addWidget(self.complexity_slider, 1, 1)

        self.complexity_label = QLabel("Standard (12 Configs)")
        self.complexity_label.setStyleSheet("color: #aaaaaa; font-size: 9px;")
        grid.addWidget(self.complexity_label, 2, 0, 1, 2)

        # Batch-Size fuer Auto-Trainer
        grid.addWidget(QLabel("Batch-Size:"), 3, 0)
        self.auto_batch_spin = QSpinBox()
        self.auto_batch_spin.setRange(32, 512)
        self.auto_batch_spin.setValue(128)
        self.auto_batch_spin.setSingleStep(32)
        self.auto_batch_spin.setToolTip("Groessere Batches = bessere GPU-Auslastung")
        grid.addWidget(self.auto_batch_spin, 3, 1)

        # Mixed Precision
        self.amp_check = QCheckBox("Mixed Precision (FP16)")
        self.amp_check.setChecked(True)
        self.amp_check.setToolTip("Halbiert Speicherverbrauch auf modernen GPUs")
        grid.addWidget(self.amp_check, 4, 0, 1, 2)

        layout.addWidget(group)

    def _create_buttons(self, layout: QVBoxLayout):
        """Erstellt die Buttons."""
        btn_layout = QVBoxLayout()

        self.start_btn = QPushButton("Training starten")
        self.start_btn.setStyleSheet(f"background-color: {COLORS['success']}; font-weight: bold;")
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Training stoppen")
        self.stop_btn.setStyleSheet(f"background-color: {COLORS['error']}; font-weight: bold;")
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)

        layout.addLayout(btn_layout)

    def _connect_signals(self):
        """Verbindet interne Signals."""
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        self.auto_trainer_check.stateChanged.connect(self._on_auto_trainer_toggled)
        self.complexity_slider.valueChanged.connect(self._on_complexity_changed)
        self.start_btn.clicked.connect(self._on_start_clicked)
        self.stop_btn.clicked.connect(self.stop_training.emit)

    def _on_model_changed(self, model_name: str):
        """Aktualisiert UI basierend auf Modell-Auswahl."""
        is_lstm_gru = ModelFactory.model_requires_hidden_sizes(model_name)
        is_transformer = model_name.lower() in ['transformer', 'hf-transformer']
        is_patchtst = model_name.lower() == 'patchtst'
        is_cnn = model_name.lower() in ['cnn', 'cnn-lstm']

        # GroupBoxen ein-/ausblenden
        self.lstm_group.setVisible(is_lstm_gru)
        self.transformer_group.setVisible(is_transformer or is_patchtst)
        self.cnn_group.setVisible(is_cnn)
        self.patchtst_group.setVisible(is_patchtst)
        self.advanced_group.setVisible(is_lstm_gru)

        # Presets aktualisieren
        presets = ModelFactory.get_presets(model_name)
        self.preset_combo.clear()
        self.preset_combo.addItems(list(presets.keys()))

        # Signal emittieren
        self.model_changed.emit(model_name)

    def _on_preset_changed(self, preset_name: str):
        """Aktualisiert Parameter basierend auf Preset."""
        model_name = self.model_combo.currentText()
        presets = ModelFactory.get_presets(model_name)

        if preset_name == 'Custom' or preset_name not in presets or presets[preset_name] is None:
            return

        params = presets[preset_name]

        # LSTM/GRU Presets
        if 'hidden_sizes' in params:
            self.hidden_sizes_edit.setText(', '.join(map(str, params['hidden_sizes'])))

        # Transformer Presets
        if 'd_model' in params:
            self.d_model_spin.setValue(params['d_model'])
        if 'nhead' in params:
            self.nhead_spin.setValue(params['nhead'])
        if 'num_encoder_layers' in params:
            self.encoder_layers_spin.setValue(params['num_encoder_layers'])

        # PatchTST Presets
        if 'num_hidden_layers' in params:
            self.encoder_layers_spin.setValue(params['num_hidden_layers'])
        if 'num_attention_heads' in params:
            self.nhead_spin.setValue(params['num_attention_heads'])
        if 'ffn_dim' in params:
            self.ffn_dim_spin.setValue(params['ffn_dim'])

        # CNN Presets
        if 'num_filters' in params:
            self.num_filters_spin.setValue(params['num_filters'])
        if 'num_conv_layers' in params:
            self.num_conv_layers_spin.setValue(params['num_conv_layers'])

    def _on_auto_trainer_toggled(self, state: int):
        """Aktiviert/deaktiviert Auto-Trainer Modus."""
        is_auto = state == Qt.CheckState.Checked.value

        # Manuelle Parameter deaktivieren
        self.model_combo.setEnabled(not is_auto)
        self.preset_combo.setEnabled(not is_auto)
        self.lstm_group.setEnabled(not is_auto)
        self.transformer_group.setEnabled(not is_auto)
        self.cnn_group.setEnabled(not is_auto)
        self.patchtst_group.setEnabled(not is_auto)
        self.advanced_group.setEnabled(not is_auto)

        # Auto-Trainer Controls aktivieren
        self.complexity_slider.setEnabled(is_auto)
        self.auto_batch_spin.setEnabled(is_auto)
        self.amp_check.setEnabled(is_auto)

        # Button-Text aendern
        self.start_btn.setText("Auto-Training starten" if is_auto else "Training starten")

    def _on_complexity_changed(self, value: int):
        """Aktualisiert Komplexitaets-Label."""
        complexity_info = {
            1: "Schnell (3 Configs)",
            2: "Einfach (6 Configs)",
            3: "Standard (12 Configs)",
            4: "Erweitert (18 Configs)",
            5: "Gruendlich (25 Configs)"
        }
        self.complexity_label.setText(complexity_info.get(value, ""))

    def _on_start_clicked(self):
        """Behandelt Start-Button Klick."""
        if self.auto_trainer_check.isChecked():
            self.start_auto_training.emit()
        else:
            self.start_training.emit()

    def _select_save_path(self):
        """Oeffnet Dialog zur Pfadauswahl."""
        path = QFileDialog.getExistingDirectory(self, "Speicherort waehlen")
        if path:
            self.save_path_label.setText(path)

    def _update_gpu_stats(self):
        """Aktualisiert GPU-Speicher und Auslastung."""
        if not torch.cuda.is_available() or self.gpu_memory_bar is None:
            return

        try:
            # Speicher via PyTorch
            allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            free = total - allocated
            percent = int((allocated / total) * 100)

            self.gpu_memory_bar.setValue(percent)
            self.gpu_used_label.setText(f"Belegt: {allocated:.2f} GB")
            self.gpu_free_label.setText(f"Frei: {free:.2f} GB")

            # Auslastung via NVML
            if self._nvml_initialized and self._nvml_handle and self.gpu_util_bar:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=FutureWarning)
                        util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                        self.gpu_util_bar.setValue(util.gpu)
                except Exception:
                    self.gpu_util_bar.setValue(0)
        except Exception:
            pass

    def __del__(self):
        """Destruktor - NVML sauber beenden."""
        if hasattr(self, '_nvml_initialized') and self._nvml_initialized and pynvml is not None:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    pynvml.nvmlShutdown()
            except Exception:
                pass

    def get_device(self) -> torch.device:
        """Gibt das ausgewaehlte Device zurueck."""
        if self.use_gpu_check.isChecked() and torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    def get_model_config(self) -> Dict[str, Any]:
        """Gibt die aktuelle Modell-Konfiguration zurueck."""
        model_name = self.model_combo.currentText()
        config = {
            'model_name': model_name,
            'dropout': self.dropout_spin.value(),
            'num_classes': 3  # BUY, HOLD, SELL
        }

        # LSTM/GRU Parameter
        if ModelFactory.model_requires_hidden_sizes(model_name):
            hidden_sizes_str = self.hidden_sizes_edit.text()
            config['hidden_sizes'] = [int(x.strip()) for x in hidden_sizes_str.split(',')]
            config['use_layer_norm'] = self.use_layer_norm_check.isChecked()
            config['use_attention'] = self.use_attention_check.isChecked()
            config['use_residual'] = self.use_residual_check.isChecked()
            config['attention_heads'] = self.attention_heads_spin.value()

        # Transformer Parameter
        if model_name.lower() in ['transformer', 'hf-transformer', 'patchtst']:
            config['d_model'] = self.d_model_spin.value()
            config['nhead'] = self.nhead_spin.value()
            config['num_encoder_layers'] = self.encoder_layers_spin.value()

        # PatchTST Parameter
        if model_name.lower() == 'patchtst':
            config['context_length'] = self.context_length_spin.value()
            config['patch_length'] = self.patch_length_spin.value()
            config['stride'] = self.stride_spin.value()
            config['ffn_dim'] = self.ffn_dim_spin.value()
            config['num_hidden_layers'] = self.encoder_layers_spin.value()
            config['num_attention_heads'] = self.nhead_spin.value()

        # CNN Parameter
        if model_name.lower() in ['cnn', 'cnn-lstm']:
            config['num_filters'] = self.num_filters_spin.value()
            config['num_conv_layers'] = self.num_conv_layers_spin.value()
            config['hidden_size'] = self.cnn_hidden_spin.value()

        return config

    def get_training_config(self) -> Dict[str, Any]:
        """Gibt die aktuelle Training-Konfiguration zurueck."""
        return {
            'epochs': self.epochs_spin.value(),
            'learning_rate': self.lr_spin.value(),
            'batch_size': self.batch_size_spin.value(),
            'early_stopping': self.early_stopping_check.isChecked(),
            'patience': self.patience_spin.value(),
            'save_best': self.save_best_check.isChecked(),
            'save_history': self.save_history_check.isChecked(),
            'save_path': self.save_path_label.text(),
            'sync_training': self.sync_training_check.isChecked()
        }

    def get_auto_trainer_config(self) -> Dict[str, Any]:
        """Gibt Auto-Trainer Konfiguration zurueck."""
        return {
            'complexity': self.complexity_slider.value(),
            'batch_size': self.auto_batch_spin.value(),
            'use_amp': self.amp_check.isChecked()
        }

    def set_training_active(self, active: bool):
        """Setzt UI-Status waehrend Training."""
        self.start_btn.setEnabled(not active)
        self.stop_btn.setEnabled(active)
        self.model_combo.setEnabled(not active)
        self.preset_combo.setEnabled(not active)
