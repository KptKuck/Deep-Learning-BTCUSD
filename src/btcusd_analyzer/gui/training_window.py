"""
Training Window - GUI fuer Modell-Training mit Live-Visualisierung

Erweitert mit:
- Architektur-Presets (LSTM, Transformer)
- Variable hidden_sizes pro Layer
- Auto-Trainer Integration
- Erweiterte Modell-Optionen (LayerNorm, Attention, Residual)
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import inspect

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QProgressBar, QTextEdit, QSplitter, QFileDialog, QMessageBox,
    QCheckBox, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QScrollArea, QSlider, QLineEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont

import torch
import numpy as np

from .styles import get_stylesheet, COLORS
from ..models.factory import ModelFactory, LSTM_PRESETS, TRANSFORMER_PRESETS, CNN_PRESETS


class TrainingWorker(QThread):
    """Worker-Thread fuer Training ohne GUI-Blockierung."""

    # Signals
    epoch_completed = pyqtSignal(int, float, float, float, float)  # epoch, train_loss, train_acc, val_loss, val_acc
    training_finished = pyqtSignal(dict)  # final results
    training_error = pyqtSignal(str)  # error message
    log_message = pyqtSignal(str)  # log output

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config: Dict[str, Any],
        device: torch.device
    ):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self._stop_requested = False

    def run(self):
        """Fuehrt das Training durch."""
        try:
            from ..trainer.trainer import Trainer
            from ..trainer.callbacks import EarlyStopping, ModelCheckpoint

            # WICHTIG: CUDA-Kontext im Worker-Thread initialisieren
            # Das Modell muss im selben Thread auf GPU verschoben werden,
            # in dem auch das Training stattfindet
            if self.device.type == 'cuda':
                # Neuen CUDA-Kontext fuer diesen Thread erstellen
                torch.cuda.set_device(0)
                torch.cuda.empty_cache()

                # Modell im Worker-Thread auf GPU verschieben
                self.model = self.model.to(self.device)
                self.log_message.emit(f"Modell auf GPU verschoben (im Worker-Thread)")
                self.log_message.emit(f"GPU Memory geleert")

            # Callbacks
            callbacks = []
            if self.config.get('early_stopping', True):
                callbacks.append(EarlyStopping(
                    patience=self.config.get('patience', 10),
                    monitor='val_loss'
                ))

            if self.config.get('save_best', True):
                save_path = Path(self.config.get('save_path', 'models'))
                save_path.mkdir(parents=True, exist_ok=True)
                filepath = str(save_path / 'model_epoch{epoch}_acc{val_accuracy}.pt')
                callbacks.append(ModelCheckpoint(
                    filepath=filepath,
                    monitor='val_accuracy'
                ))

            # Trainer erstellen
            trainer = Trainer(
                model=self.model,
                device=self.device,
                callbacks=callbacks
            )

            # Custom progress callback fuer GUI-Updates
            def progress_callback(metrics):
                if self._stop_requested:
                    trainer.stop_training = True
                    return

                self.epoch_completed.emit(
                    metrics.epoch,
                    metrics.train_loss,
                    metrics.train_accuracy,
                    metrics.val_loss,
                    metrics.val_accuracy
                )

            # Training starten
            self.log_message.emit(f"Training gestartet: {self.model.name}")
            self.log_message.emit(f"Device: {self.device}")
            self.log_message.emit(f"Epochen: {self.config.get('epochs', 100)}")

            history = trainer.train(
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                epochs=self.config.get('epochs', 100),
                learning_rate=self.config.get('learning_rate', 0.001),
                progress_callback=progress_callback
            )

            # Finales Modell automatisch speichern
            save_path = Path(self.config.get('save_path', 'models'))
            save_path.mkdir(parents=True, exist_ok=True)

            best_acc = max(history.val_accuracy) if history.val_accuracy else 0
            model_name = self.model.name.lower().replace(' ', '_')
            final_model_path = save_path / f'{model_name}_final_acc{best_acc:.1f}.pt'

            self.model.save(final_model_path, metrics={
                'best_accuracy': best_acc,
                'final_loss': history.val_loss[-1] if history.val_loss else 0,
                'epochs_trained': len(history.epochs)
            })
            self.log_message.emit(f"Modell gespeichert: {final_model_path}")

            # Ergebnis senden
            self.training_finished.emit({
                'history': history,
                'best_accuracy': best_acc,
                'final_loss': history.val_loss[-1] if history.val_loss else 0,
                'stopped_early': trainer.stop_training,
                'model_path': str(final_model_path)
            })

            # Trainer-Referenzen aufraeumen
            del trainer
            del history

        except torch.cuda.OutOfMemoryError as e:
            # GPU Speicher freigeben
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            error_msg = f"GPU Out of Memory: {e}\n\nVersuche kleinere Batch Size oder weniger Features."
            self.training_error.emit(error_msg)

        except RuntimeError as e:
            # Haeufig CUDA-Fehler
            error_str = str(e)
            if 'CUDA' in error_str or 'cuda' in error_str:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                error_msg = f"CUDA Fehler: {error_str}\n\nMoegliche Loesungen:\n- Batch Size reduzieren\n- GPU-Treiber aktualisieren\n- CPU statt GPU verwenden"
            else:
                error_msg = f"Runtime Fehler: {error_str}"
            self.training_error.emit(error_msg)

        except Exception as e:
            # Allgemeiner Fehler - detailliertes Logging
            import traceback
            import sys

            exc_type, exc_value, exc_tb = sys.exc_info()
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)

            error_msg = f"Training Fehler: {e}\n\nTyp: {exc_type.__name__}\n\nTraceback:\n{''.join(tb_lines)}"
            self.log_message.emit(f"KRITISCHER FEHLER: {error_msg}")
            self.training_error.emit(error_msg)

        finally:
            # Gruendliches Speicher-Cleanup im Worker
            import gc
            try:
                # Modell auf CPU verschieben vor Cleanup
                if self.model is not None:
                    try:
                        self.model.cpu()
                    except Exception:
                        pass

                # GPU aufraeumen
                if torch.cuda.is_available():
                    for _ in range(3):
                        torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                    # Memory Status loggen
                    allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
                    reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
                    self.log_message.emit(f"Worker Cleanup: GPU {allocated:.2f}GB alloc / {reserved:.2f}GB reserved")

                # GC forcieren
                gc.collect()

            except Exception as cleanup_err:
                self.log_message.emit(f"Cleanup Warnung: {cleanup_err}")

    def stop(self):
        """Stoppt das Training."""
        self._stop_requested = True


class TrainingWindow(QMainWindow):
    """
    Training-Fenster mit Live-Visualisierung.

    Features:
    - Modell-Auswahl und Konfiguration
    - Hyperparameter-Einstellung
    - Live Loss/Accuracy Plot
    - Training-Log
    - Fortschrittsanzeige
    - Stop/Resume Funktion
    """

    # Signals
    log_message = pyqtSignal(str, str)  # message, level
    training_completed = pyqtSignal(object, object)  # model, results

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("3 - Training")
        self.setMinimumSize(1200, 800)
        self._parent = parent

        # State
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.worker = None
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self._stop_requested = False  # Fuer synchrones Training

        # Trainingsdaten (werden von MainWindow gesetzt)
        self.training_data = None
        self.training_info = None

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._setup_ui()
        self.setStyleSheet(get_stylesheet())

    def _log(self, message: str, level: str = 'INFO'):
        """Loggt eine Nachricht an MainWindow und lokales Log."""
        # Aufrufenden Funktionsnamen ermitteln
        caller_frame = inspect.currentframe().f_back
        caller_name = caller_frame.f_code.co_name if caller_frame else 'unknown'

        # Nachricht mit Funktionsnamen
        formatted_message = f'{caller_name}() - {message}'

        # An MainWindow senden (falls parent _log hat)
        if self._parent and hasattr(self._parent, '_log'):
            self._parent._log(f'[Training] {formatted_message}', level)
        # Signal emittieren
        self.log_message.emit(formatted_message, level)
        # Auch lokal loggen falls vorhanden
        if hasattr(self, 'log_text'):
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.log_text.append(f'[{timestamp}] [{level}] {formatted_message}')

    def _setup_ui(self):
        """Erstellt die Benutzeroberflaeche."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Linke Seite: Konfiguration
        left_panel = self._create_config_panel()
        left_panel.setFixedWidth(350)

        # Rechte Seite: Visualisierung
        right_panel = self._create_visualization_panel()

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([350, 850])

        main_layout.addWidget(splitter)

    def _create_config_panel(self) -> QWidget:
        """Erstellt das Konfigurations-Panel."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(5)

        # Modell-Auswahl
        model_group = QGroupBox("Modell")
        model_layout = QGridLayout(model_group)

        model_layout.addWidget(QLabel("Architektur:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(ModelFactory.get_available_models())
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        model_layout.addWidget(self.model_combo, 0, 1)

        # Preset-Auswahl (fuer LSTM/GRU)
        model_layout.addWidget(QLabel("Preset:"), 1, 0)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(LSTM_PRESETS.keys()))
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        model_layout.addWidget(self.preset_combo, 1, 1)

        # Hidden Sizes (editierbar)
        model_layout.addWidget(QLabel("Hidden Sizes:"), 2, 0)
        self.hidden_sizes_edit = QLineEdit("128, 128")
        self.hidden_sizes_edit.setToolTip("Komma-getrennte Liste: z.B. 256, 128, 64")
        model_layout.addWidget(self.hidden_sizes_edit, 2, 1)

        # Transformer-Parameter (anfangs versteckt)
        self.d_model_label = QLabel("d_model:")
        self.d_model_spin = QSpinBox()
        self.d_model_spin.setRange(32, 1024)
        self.d_model_spin.setValue(128)
        self.d_model_spin.setSingleStep(32)
        model_layout.addWidget(self.d_model_label, 3, 0)
        model_layout.addWidget(self.d_model_spin, 3, 1)

        self.nhead_label = QLabel("Attention Heads:")
        self.nhead_spin = QSpinBox()
        self.nhead_spin.setRange(1, 16)
        self.nhead_spin.setValue(4)
        model_layout.addWidget(self.nhead_label, 4, 0)
        model_layout.addWidget(self.nhead_spin, 4, 1)

        self.encoder_layers_label = QLabel("Encoder Layers:")
        self.encoder_layers_spin = QSpinBox()
        self.encoder_layers_spin.setRange(1, 12)
        self.encoder_layers_spin.setValue(2)
        model_layout.addWidget(self.encoder_layers_label, 5, 0)
        model_layout.addWidget(self.encoder_layers_spin, 5, 1)

        # Legacy Hidden Size (fuer CNN, CNN-LSTM)
        model_layout.addWidget(QLabel("Hidden Size:"), 6, 0)
        self.hidden_size_spin = QSpinBox()
        self.hidden_size_spin.setRange(16, 512)
        self.hidden_size_spin.setValue(128)
        self.hidden_size_spin.setSingleStep(16)
        model_layout.addWidget(self.hidden_size_spin, 6, 1)

        model_layout.addWidget(QLabel("Layers:"), 7, 0)
        self.num_layers_spin = QSpinBox()
        self.num_layers_spin.setRange(1, 6)
        self.num_layers_spin.setValue(2)
        model_layout.addWidget(self.num_layers_spin, 7, 1)

        model_layout.addWidget(QLabel("Dropout:"), 8, 0)
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.8)
        self.dropout_spin.setValue(0.2)
        self.dropout_spin.setSingleStep(0.05)
        model_layout.addWidget(self.dropout_spin, 8, 1)

        layout.addWidget(model_group)

        # Erweiterte Optionen (LayerNorm, Attention, Residual)
        advanced_group = QGroupBox("Erweiterte Architektur")
        advanced_layout = QGridLayout(advanced_group)

        self.use_layer_norm_check = QCheckBox("Layer Normalization")
        self.use_layer_norm_check.setToolTip("Normalisierung nach jedem LSTM/GRU Layer")
        advanced_layout.addWidget(self.use_layer_norm_check, 0, 0, 1, 2)

        self.use_attention_check = QCheckBox("Attention")
        self.use_attention_check.setToolTip("Self-Attention nach letztem LSTM/GRU")
        advanced_layout.addWidget(self.use_attention_check, 1, 0)

        self.attention_heads_spin = QSpinBox()
        self.attention_heads_spin.setRange(1, 16)
        self.attention_heads_spin.setValue(4)
        advanced_layout.addWidget(self.attention_heads_spin, 1, 1)

        self.use_residual_check = QCheckBox("Residual Connections")
        self.use_residual_check.setToolTip("Skip-Connections zwischen Layern")
        advanced_layout.addWidget(self.use_residual_check, 2, 0, 1, 2)

        layout.addWidget(advanced_group)

        # Initial UI-Status setzen
        self._on_model_changed(self.model_combo.currentText())

        # Training-Parameter
        train_group = QGroupBox("Training")
        train_layout = QGridLayout(train_group)

        train_layout.addWidget(QLabel("Epochen:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(100)
        train_layout.addWidget(self.epochs_spin, 0, 1)

        train_layout.addWidget(QLabel("Learning Rate:"), 1, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.1)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.0001)
        train_layout.addWidget(self.lr_spin, 1, 1)

        train_layout.addWidget(QLabel("Batch Size:"), 2, 0)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(8, 512)
        self.batch_size_spin.setValue(32)
        self.batch_size_spin.setSingleStep(8)
        train_layout.addWidget(self.batch_size_spin, 2, 1)

        layout.addWidget(train_group)

        # Early Stopping & Speichern kombiniert
        options_group = QGroupBox("Optionen")
        options_layout = QGridLayout(options_group)
        options_layout.setVerticalSpacing(3)

        # Early Stopping
        self.early_stopping_check = QCheckBox("Early Stopping")
        self.early_stopping_check.setChecked(True)
        options_layout.addWidget(self.early_stopping_check, 0, 0)

        options_layout.addWidget(QLabel("Patience:"), 0, 1)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 50)
        self.patience_spin.setValue(10)
        options_layout.addWidget(self.patience_spin, 0, 2)

        # Speicher-Optionen
        self.save_best_check = QCheckBox("Modell speichern")
        self.save_best_check.setChecked(True)
        options_layout.addWidget(self.save_best_check, 1, 0, 1, 3)

        self.save_history_check = QCheckBox("History speichern")
        self.save_history_check.setChecked(True)
        options_layout.addWidget(self.save_history_check, 2, 0, 1, 3)

        # Synchrones Training (stabiler fuer GPU)
        self.sync_training_check = QCheckBox("Sync Training (GPU-stabil)")
        self.sync_training_check.setChecked(True)  # Default: an (stabiler)
        self.sync_training_check.setToolTip("Training im Main-Thread ausfuehren.\nStabiler fuer GPU, aber GUI blockiert waehrend Training.")
        options_layout.addWidget(self.sync_training_check, 3, 0, 1, 3)

        # Speicherpfad kompakt
        options_layout.addWidget(QLabel("Pfad:"), 4, 0)
        self.save_path_edit = QLabel("models/")
        self.save_path_edit.setStyleSheet("font-size: 9px; color: #aaaaaa;")
        options_layout.addWidget(self.save_path_edit, 4, 1)
        self.save_path_btn = QPushButton("...")
        self.save_path_btn.setFixedWidth(30)
        self.save_path_btn.clicked.connect(self._select_save_path)
        options_layout.addWidget(self.save_path_btn, 4, 2)

        layout.addWidget(options_group)

        # Device Info (GPU Status wie MATLAB) - kompakter
        device_group = QGroupBox("Device")
        device_layout = QGridLayout(device_group)
        device_layout.setVerticalSpacing(3)

        # GPU/CPU Switch
        device_layout.addWidget(QLabel("GPU:"), 0, 0)
        self.use_gpu_check = QCheckBox("Aktiv")
        self.use_gpu_check.setChecked(torch.cuda.is_available())
        self.use_gpu_check.setEnabled(torch.cuda.is_available())
        self.use_gpu_check.stateChanged.connect(self._update_device)
        device_layout.addWidget(self.use_gpu_check, 0, 1)

        # GPU Name
        device_text = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        if torch.cuda.is_available():
            device_text = f"{torch.cuda.get_device_name(0)}"
        self.device_label = QLabel(device_text)
        self.device_label.setStyleSheet(f"color: {'#33b34d' if torch.cuda.is_available() else '#e6b333'}; font-size: 10px;")
        device_layout.addWidget(self.device_label, 1, 0, 1, 2)

        # GPU Memory Bar (wie MATLAB)
        if torch.cuda.is_available():
            # Progress Bar fuer GPU-Speicher
            self.gpu_memory_bar = QProgressBar()
            self.gpu_memory_bar.setMinimum(0)
            self.gpu_memory_bar.setMaximum(100)
            self.gpu_memory_bar.setValue(0)
            self.gpu_memory_bar.setTextVisible(True)
            self.gpu_memory_bar.setFormat("%p%")
            self.gpu_memory_bar.setMaximumHeight(18)
            device_layout.addWidget(self.gpu_memory_bar, 2, 0, 1, 2)

            # Speicher-Labels kompakt
            mem_label_layout = QHBoxLayout()
            self.gpu_used_label = QLabel("Belegt: - GB")
            self.gpu_used_label.setStyleSheet("color: #aaaaaa; font-size: 9px;")
            mem_label_layout.addWidget(self.gpu_used_label)

            self.gpu_free_label = QLabel("Frei: - GB")
            self.gpu_free_label.setStyleSheet("color: #33b34d; font-size: 9px;")
            mem_label_layout.addWidget(self.gpu_free_label)

            device_layout.addLayout(mem_label_layout, 3, 0, 1, 2)

            # GPU Memory Timer starten
            self.gpu_timer = QTimer()
            self.gpu_timer.timeout.connect(self._update_gpu_memory)
            self.gpu_timer.start(1000)
            self._update_gpu_memory()

            # GPU Test Button
            self.gpu_test_btn = QPushButton("GPU Test")
            self.gpu_test_btn.setToolTip("Testet GPU-Training mit synthetischen Daten")
            self.gpu_test_btn.clicked.connect(self._run_gpu_test)
            device_layout.addWidget(self.gpu_test_btn, 4, 0, 1, 2)

        layout.addWidget(device_group)

        # Auto-Trainer GroupBox
        auto_group = QGroupBox("Auto-Trainer")
        auto_layout = QGridLayout(auto_group)

        self.auto_trainer_check = QCheckBox("Auto-Modus aktivieren")
        self.auto_trainer_check.setToolTip("Automatisch verschiedene Modelle und Parameter testen")
        self.auto_trainer_check.stateChanged.connect(self._on_auto_trainer_toggled)
        auto_layout.addWidget(self.auto_trainer_check, 0, 0, 1, 2)

        auto_layout.addWidget(QLabel("Komplexitaet:"), 1, 0)
        self.complexity_slider = QSlider(Qt.Orientation.Horizontal)
        self.complexity_slider.setRange(1, 5)
        self.complexity_slider.setValue(3)
        self.complexity_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.complexity_slider.setTickInterval(1)
        self.complexity_slider.valueChanged.connect(self._on_complexity_changed)
        auto_layout.addWidget(self.complexity_slider, 1, 1)

        self.complexity_label = QLabel("Standard (12 Configs)")
        self.complexity_label.setStyleSheet("color: #aaaaaa; font-size: 9px;")
        auto_layout.addWidget(self.complexity_label, 2, 0, 1, 2)

        # Batch-Size fuer Auto-Trainer
        auto_layout.addWidget(QLabel("Batch-Size:"), 3, 0)
        self.auto_batch_spin = QSpinBox()
        self.auto_batch_spin.setRange(32, 512)
        self.auto_batch_spin.setValue(128)  # Default: 128 fuer bessere GPU-Auslastung
        self.auto_batch_spin.setSingleStep(32)
        self.auto_batch_spin.setToolTip("Groessere Batches = bessere GPU-Auslastung (128-256 empfohlen)")
        auto_layout.addWidget(self.auto_batch_spin, 3, 1)

        # Mixed Precision (AMP)
        self.amp_check = QCheckBox("Mixed Precision (FP16)")
        self.amp_check.setChecked(True)
        self.amp_check.setToolTip("Halbiert Speicherverbrauch, verdoppelt Durchsatz auf modernen GPUs")
        auto_layout.addWidget(self.amp_check, 4, 0, 1, 2)

        layout.addWidget(auto_group)

        # Buttons
        btn_layout = QVBoxLayout()

        self.start_btn = QPushButton("Training starten")
        self.start_btn.setStyleSheet(f"background-color: {COLORS['success']}; font-weight: bold;")
        self.start_btn.clicked.connect(self._start_training)
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Training stoppen")
        self.stop_btn.setStyleSheet(f"background-color: {COLORS['error']}; font-weight: bold;")
        self.stop_btn.clicked.connect(self._stop_training)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)

        layout.addLayout(btn_layout)
        layout.addStretch()

        scroll.setWidget(panel)
        return scroll

    def _create_visualization_panel(self) -> QWidget:
        """Erstellt das Visualisierungs-Panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Tabs fuer verschiedene Ansichten
        tabs = QTabWidget()

        # Tab 1: Live-Plot
        plot_tab = self._create_plot_tab()
        tabs.addTab(plot_tab, "Training-Verlauf")

        # Tab 2: Metriken-Tabelle
        metrics_tab = self._create_metrics_tab()
        tabs.addTab(metrics_tab, "Metriken")

        # Tab 3: Log
        log_tab = self._create_log_tab()
        tabs.addTab(log_tab, "Log")

        # Tab 4: Auto-Trainer Ergebnisse
        auto_tab = self._create_auto_trainer_tab()
        tabs.addTab(auto_tab, "Auto-Trainer")

        layout.addWidget(tabs)

        # Fortschrittsanzeige
        progress_group = QGroupBox("Fortschritt")
        progress_layout = QVBoxLayout(progress_group)

        # Epoch Progress
        epoch_layout = QHBoxLayout()
        self.epoch_label = QLabel("Epoch: 0 / 0")
        self.epoch_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        epoch_layout.addWidget(self.epoch_label)
        epoch_layout.addStretch()

        self.time_label = QLabel("Zeit: --:--")
        epoch_layout.addWidget(self.time_label)
        progress_layout.addLayout(epoch_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        # Aktuelle Metriken
        metrics_layout = QHBoxLayout()

        self.train_loss_label = QLabel("Train Loss: -")
        self.train_acc_label = QLabel("Train Acc: -")
        self.val_loss_label = QLabel("Val Loss: -")
        self.val_acc_label = QLabel("Val Acc: -")

        for label in [self.train_loss_label, self.train_acc_label,
                      self.val_loss_label, self.val_acc_label]:
            label.setStyleSheet(f"color: {COLORS['text_secondary']};")
            metrics_layout.addWidget(label)

        progress_layout.addLayout(metrics_layout)
        layout.addWidget(progress_group)

        return panel

    def _create_plot_tab(self) -> QWidget:
        """Erstellt den Plot-Tab mit matplotlib."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure

            # Figure erstellen
            self.figure = Figure(figsize=(8, 6), facecolor=COLORS['bg_primary'])
            self.canvas = FigureCanvas(self.figure)

            # Subplots
            self.ax_loss = self.figure.add_subplot(211)
            self.ax_acc = self.figure.add_subplot(212)

            self._setup_plot_style()
            layout.addWidget(self.canvas)

        except ImportError:
            # Fallback wenn matplotlib nicht verfuegbar
            layout.addWidget(QLabel("matplotlib nicht installiert"))

        return widget

    def _setup_plot_style(self):
        """Konfiguriert den Plot-Style."""
        for ax in [self.ax_loss, self.ax_acc]:
            ax.set_facecolor(COLORS['bg_secondary'])
            ax.tick_params(colors=COLORS['text_secondary'])
            ax.spines['bottom'].set_color(COLORS['border'])
            ax.spines['top'].set_color(COLORS['border'])
            ax.spines['left'].set_color(COLORS['border'])
            ax.spines['right'].set_color(COLORS['border'])

        self.ax_loss.set_title('Loss', color=COLORS['text_primary'])
        self.ax_loss.set_xlabel('Epoch', color=COLORS['text_secondary'])
        self.ax_loss.set_ylabel('Loss', color=COLORS['text_secondary'])

        self.ax_acc.set_title('Accuracy', color=COLORS['text_primary'])
        self.ax_acc.set_xlabel('Epoch', color=COLORS['text_secondary'])
        self.ax_acc.set_ylabel('Accuracy (%)', color=COLORS['text_secondary'])

        self.figure.tight_layout()

    def _update_plot(self):
        """Aktualisiert den Plot."""
        if not hasattr(self, 'ax_loss'):
            return

        epochs = list(range(1, len(self.history['train_loss']) + 1))

        # Loss Plot
        self.ax_loss.clear()
        self._setup_plot_style()
        if epochs:
            self.ax_loss.plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
            self.ax_loss.plot(epochs, self.history['val_loss'], 'r-', label='Validation', linewidth=2)
            self.ax_loss.legend(facecolor=COLORS['bg_tertiary'], labelcolor=COLORS['text_primary'])
            self.ax_loss.set_title('Loss', color=COLORS['text_primary'])

        # Accuracy Plot
        self.ax_acc.clear()
        if epochs:
            self.ax_acc.plot(epochs, self.history['train_acc'], 'b-', label='Train', linewidth=2)
            self.ax_acc.plot(epochs, self.history['val_acc'], 'r-', label='Validation', linewidth=2)
            self.ax_acc.legend(facecolor=COLORS['bg_tertiary'], labelcolor=COLORS['text_primary'])
            self.ax_acc.set_title('Accuracy', color=COLORS['text_primary'])

        self.figure.tight_layout()
        self.canvas.draw()

    def _create_metrics_tab(self) -> QWidget:
        """Erstellt den Metriken-Tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(5)
        self.metrics_table.setHorizontalHeaderLabels([
            'Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc'
        ])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        layout.addWidget(self.metrics_table)
        return widget

    def _create_log_tab(self) -> QWidget:
        """Erstellt den Log-Tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['bg_secondary']};
                color: {COLORS['text_primary']};
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
            }}
        """)

        layout.addWidget(self.log_text)

        # Clear Button
        clear_btn = QPushButton("Log leeren")
        clear_btn.clicked.connect(lambda: self.log_text.clear())
        layout.addWidget(clear_btn)

        return widget

    def _select_save_path(self):
        """Oeffnet Dialog zur Pfadauswahl."""
        path = QFileDialog.getExistingDirectory(self, "Speicherort waehlen")
        if path:
            self.save_path_edit.setText(path)

    def set_data(self, train_loader, val_loader):
        """Setzt die Daten-Loader."""
        self.train_loader = train_loader
        self.val_loader = val_loader
        self._log(f"Daten geladen: {len(train_loader.dataset)} Training, {len(val_loader.dataset)} Validation")

    def prepare_data_loaders(self, training_data: Dict[str, Any], batch_size: int = 64, val_split: float = 0.2):
        """
        Erstellt DataLoader aus training_data Dictionary.

        Args:
            training_data: Dict mit 'X' (Sequenzen) und 'Y' (Labels)
            batch_size: Batch-Groesse
            val_split: Anteil fuer Validierung (default 20%)
        """
        from torch.utils.data import TensorDataset, DataLoader

        X = training_data['X']
        Y = training_data['Y']

        # In PyTorch Tensoren konvertieren
        X_tensor = torch.FloatTensor(X)
        Y_tensor = torch.LongTensor(Y)

        # Train/Val Split
        total = len(X)
        val_size = int(total * val_split)
        train_size = total - val_size

        # Sequenzieller Split (nicht random, da Zeitreihen)
        X_train = X_tensor[:train_size]
        Y_train = Y_tensor[:train_size]
        X_val = X_tensor[train_size:]
        Y_val = Y_tensor[train_size:]

        # Datasets
        train_dataset = TensorDataset(X_train, Y_train)
        val_dataset = TensorDataset(X_val, Y_val)

        # DataLoader mit CUDA-Optimierung
        use_cuda = self.device.type == 'cuda'
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,  # Zeitreihen nicht shufflen
            pin_memory=use_cuda,  # Schnellerer CPU->GPU Transfer
            num_workers=0  # Windows: keine Multiprocessing-Worker
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=use_cuda,
            num_workers=0
        )

        self._log(f"DataLoader erstellt: {train_size} Training, {val_size} Validation (Batch: {batch_size})", level='SUCCESS')

    def _cleanup_previous_training(self):
        """Raeumt vorheriges Modell und GPU-Speicher auf vor neuem Training."""
        import gc

        self._log("Cleanup: Raeume vorheriges Training auf...")

        # Worker beenden falls noch aktiv
        if self.worker is not None:
            self._log("Cleanup: Worker stoppen...")
            if self.worker.isRunning():
                self.worker.stop()
                self.worker.wait(5000)  # Max 5 Sekunden warten

            # Worker-Referenzen explizit loeschen
            if hasattr(self.worker, 'model') and self.worker.model is not None:
                try:
                    self.worker.model.cpu()
                except Exception:
                    pass
                self.worker.model = None

            if hasattr(self.worker, 'train_loader'):
                self.worker.train_loader = None
            if hasattr(self.worker, 'val_loader'):
                self.worker.val_loader = None

            self.worker.deleteLater()  # Qt-seitig aufräumen
            self.worker = None

        # Altes Modell vom GPU entfernen
        if self.model is not None:
            self._log("Cleanup: Modell entfernen...")
            try:
                self.model.cpu()  # Erst auf CPU verschieben
            except Exception:
                pass
            del self.model
            self.model = None

        # History zuruecksetzen
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        # GPU Cache komplett leeren - mehrfach fuer gruendliches Cleanup
        if torch.cuda.is_available():
            for _ in range(3):  # Mehrfach leeren
                torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Warten bis alles abgeschlossen

            # GPU Memory Status loggen
            allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
            self._log(f"Cleanup: GPU nach Cleanup - {allocated:.2f}GB alloc / {reserved:.2f}GB reserved")

        # Python Garbage Collection forcieren - mehrere Generationen
        gc.collect(0)  # Generation 0
        gc.collect(1)  # Generation 1
        gc.collect(2)  # Generation 2 (alle)

        self._log("Cleanup: Abgeschlossen")

    def _start_training(self):
        """Startet das Training."""
        if self.train_loader is None or self.val_loader is None:
            QMessageBox.warning(self, "Fehler", "Keine Trainingsdaten geladen!")
            return

        # Auto-Trainer Modus?
        if self.auto_trainer_check.isChecked():
            self._run_auto_training()
            return

        # Altes Modell und GPU-Speicher aufraeumen vor neuem Training
        self._cleanup_previous_training()

        # Modell erstellen
        try:
            from ..models import ModelFactory

            model_name = self.model_combo.currentText()

            # Input-Size aus Daten ermitteln
            sample = next(iter(self.train_loader))
            input_size = sample[0].shape[-1]

            # num_classes aus training_info holen (falls vorhanden), sonst default 3
            num_classes = 3
            if self.training_info and 'num_classes' in self.training_info:
                num_classes = self.training_info['num_classes']

            # Parameter je nach Modell-Typ
            is_lstm_gru = ModelFactory.model_requires_hidden_sizes(model_name)
            is_transformer = ModelFactory.model_is_transformer(model_name)

            if is_lstm_gru:
                # LSTM/GRU mit variablen hidden_sizes
                hidden_sizes = self._parse_hidden_sizes()
                self.model = ModelFactory.create(
                    model_name,
                    input_size=input_size,
                    hidden_sizes=hidden_sizes,
                    num_classes=num_classes,
                    dropout=self.dropout_spin.value(),
                    use_layer_norm=self.use_layer_norm_check.isChecked(),
                    use_attention=self.use_attention_check.isChecked(),
                    use_residual=self.use_residual_check.isChecked(),
                    attention_heads=self.attention_heads_spin.value()
                )
            elif is_transformer:
                # Transformer
                self.model = ModelFactory.create(
                    model_name,
                    input_size=input_size,
                    d_model=self.d_model_spin.value(),
                    nhead=self.nhead_spin.value(),
                    num_encoder_layers=self.encoder_layers_spin.value(),
                    num_classes=num_classes,
                    dropout=self.dropout_spin.value()
                )
            else:
                # CNN, CNN-LSTM (legacy)
                self.model = ModelFactory.create(
                    model_name,
                    input_size=input_size,
                    hidden_size=self.hidden_size_spin.value(),
                    num_layers=self.num_layers_spin.value(),
                    num_classes=num_classes,
                    dropout=self.dropout_spin.value()
                )

            self._log(f"Klassenanzahl: {num_classes}")

            self._log(f"Modell erstellt: {self.model.name}")
            self._log(f"Parameter: {sum(p.numel() for p in self.model.parameters()):,}")

        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Modell-Erstellung fehlgeschlagen:\n{e}")
            return

        # Training-Config
        config = {
            'epochs': self.epochs_spin.value(),
            'batch_size': self.batch_size_spin.value(),
            'learning_rate': self.lr_spin.value(),
            'early_stopping': self.early_stopping_check.isChecked(),
            'patience': self.patience_spin.value(),
            'save_best': self.save_best_check.isChecked(),
            'save_path': self.save_path_edit.text()
        }

        # History zuruecksetzen
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.metrics_table.setRowCount(0)

        # UI aktualisieren
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setMaximum(config['epochs'])
        self._start_time = datetime.now()

        # Timer fuer Zeit-Update (VOR Training initialisieren!)
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_time)
        self.timer.start(1000)

        # Synchrones oder asynchrones Training
        if self.sync_training_check.isChecked():
            # Synchrones Training im Main-Thread (stabiler fuer GPU)
            self._run_sync_training(config)
        else:
            # Asynchrones Training im Worker-Thread
            self.worker = TrainingWorker(
                self.model, self.train_loader, self.val_loader, config, self.device
            )
            self.worker.epoch_completed.connect(self._on_epoch_completed)
            self.worker.training_finished.connect(self._on_training_finished)
            self.worker.training_error.connect(self._on_training_error)
            self.worker.log_message.connect(lambda msg: self._log(msg, 'INFO'))
            self.worker.start()

    def _run_sync_training(self, config: dict):
        """
        Fuehrt Training synchron im Main-Thread aus.
        Stabiler fuer GPU, da kein Thread-Wechsel bei CUDA-Operationen.
        """
        from PyQt6.QtWidgets import QApplication
        from pathlib import Path
        from datetime import datetime
        import gc

        training_start = datetime.now()
        self._log("=== Synchrones Training gestartet ===")
        self._stop_requested = False

        try:
            # Modell auf GPU verschieben
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                self.model = self.model.to(self.device)
                self._log(f"Modell auf GPU verschoben: {self.device}")

            # Loss und Optimizer (mit Class Weights falls vorhanden)
            class_weights = self.training_data.get('class_weights')
            if class_weights is not None:
                class_weights = class_weights.to(self.device)
                self._log(f"Class Weights: {class_weights.cpu().numpy().round(2)}")
                criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
            else:
                criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=1e-5
            )

            # Learning Rate Scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )

            epochs = config['epochs']
            best_val_acc = 0.0
            patience_counter = 0
            patience = config.get('patience', 10)

            for epoch in range(1, epochs + 1):
                if self._stop_requested:
                    self._log("Training durch Benutzer gestoppt")
                    break

                # === Training ===
                self.model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                for batch_idx, (data, target) in enumerate(self.train_loader):
                    data = data.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)

                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = criterion(output, target)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = output.max(1)
                    train_total += target.size(0)
                    train_correct += predicted.eq(target).sum().item()

                    # GUI responsiv halten
                    if batch_idx % 10 == 0:
                        QApplication.processEvents()
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()

                train_acc = 100.0 * train_correct / train_total
                avg_train_loss = train_loss / len(self.train_loader)

                # === Validierung ===
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for data, target in self.val_loader:
                        data = data.to(self.device)
                        target = target.to(self.device)

                        output = self.model(data)
                        loss = criterion(output, target)

                        val_loss += loss.item()
                        _, predicted = output.max(1)
                        val_total += target.size(0)
                        val_correct += predicted.eq(target).sum().item()

                val_acc = 100.0 * val_correct / val_total
                avg_val_loss = val_loss / len(self.val_loader)

                # Scheduler Step
                scheduler.step(avg_val_loss)

                # GUI Update
                self._on_epoch_completed(epoch, avg_train_loss, train_acc, avg_val_loss, val_acc)
                QApplication.processEvents()

                # Beste Accuracy tracken (immer, nicht nur bei Early Stopping)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Early Stopping Check
                if config.get('early_stopping', True):
                    if patience_counter >= patience:
                        self._log(f"Early Stopping nach {epoch} Epochen")
                        break

            # Modell speichern
            if config.get('save_best', True):
                from datetime import datetime
                import json
                save_path = Path(config.get('save_path', 'models'))
                save_path.mkdir(parents=True, exist_ok=True)
                model_name = self.model.name.lower().replace(' ', '_')
                timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                final_path = save_path / f'{model_name}_{timestamp}_acc{best_val_acc:.1f}.pt'

                # Trainingszeit berechnen
                training_end = datetime.now()
                training_duration = (training_end - training_start).total_seconds()

                # Class Weights extrahieren
                class_weights_list = None
                if self.training_data and 'class_weights' in self.training_data:
                    cw = self.training_data['class_weights']
                    class_weights_list = cw.tolist() if hasattr(cw, 'tolist') else list(cw)

                # Vollstaendige model_info (Trainings-Plakette)
                model_info = {
                    # === Modell-Identifikation ===
                    'model_type': model_name,
                    'trained_at': timestamp,
                    'training_duration_sec': round(training_duration, 1),

                    # === Architektur ===
                    'input_size': self.model.input_size if hasattr(self.model, 'input_size') else None,
                    'hidden_size': self.hidden_size_spin.value(),
                    'num_layers': self.num_layers_spin.value(),
                    'dropout': self.dropout_spin.value() if hasattr(self, 'dropout_spin') else 0.2,
                    'bidirectional': True,

                    # === Daten-Parameter ===
                    'num_classes': self.training_info.get('num_classes', 3) if self.training_info else 3,
                    'lookback_size': self.training_info.get('params', {}).get('lookback', 100) if self.training_info else 100,
                    'lookforward_size': self.training_info.get('params', {}).get('lookforward', 10) if self.training_info else 10,
                    'lookahead_bars': self.training_info.get('lookahead_bars', 0) if self.training_info else 0,
                    'train_test_split': self.training_info.get('params', {}).get('train_test_split', 80) if self.training_info else 80,

                    # === Samples ===
                    'total_samples': self.training_info.get('actual_samples', 0) if self.training_info else 0,
                    'train_samples': len(self.train_loader.dataset) if self.train_loader else 0,
                    'val_samples': len(self.val_loader.dataset) if self.val_loader else 0,

                    # === Features ===
                    'features': self.training_info.get('features', []) if self.training_info else [],
                    'num_features': len(self.training_info.get('features', [])) if self.training_info else 0,

                    # === Training-Hyperparameter ===
                    'epochs_trained': epoch,
                    'epochs_configured': config['epochs'],
                    'batch_size': config['batch_size'],
                    'learning_rate': config['learning_rate'],
                    'optimizer': 'Adam',
                    'scheduler': 'ReduceLROnPlateau',
                    'early_stopping': config.get('early_stopping', True),
                    'patience': patience,

                    # === Class Weights ===
                    'class_weights': class_weights_list,

                    # === Ergebnisse ===
                    'best_accuracy': round(best_val_acc, 2),
                    'final_val_loss': round(avg_val_loss, 4),
                    'stopped_early': patience_counter >= patience,

                    # === Peak-Detection (aus training_info) ===
                    'num_buy_peaks': len(self.training_info.get('buy_indices', [])) if self.training_info else 0,
                    'num_sell_peaks': len(self.training_info.get('sell_indices', [])) if self.training_info else 0,
                }

                # Modell speichern
                self.model.save(final_path,
                    metrics={'best_accuracy': best_val_acc, 'epochs_trained': epoch},
                    model_info=model_info
                )
                self._log(f"Modell gespeichert: {final_path}")

                # JSON-Plakette separat speichern (fuer einfaches Lesen)
                json_path = final_path.with_suffix('.json')
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(model_info, f, indent=2, ensure_ascii=False)
                self._log(f"Trainings-Info: {json_path.name}")

                # Modell in Session-Ordner kopieren
                self._save_model_to_session(final_path)

            # Training beendet - model_info an results anhaengen
            self._on_training_finished({
                'best_accuracy': best_val_acc,
                'final_loss': avg_val_loss,
                'stopped_early': patience_counter >= patience,
                'model_path': str(final_path) if config.get('save_best', True) else '',
                'model_type': model_name,
                'hidden_size': self.hidden_size_spin.value(),
                'num_layers': self.num_layers_spin.value(),
            })

        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            self._on_training_error(f"GPU Out of Memory: {e}\n\nReduziere Batch Size.")

        except Exception as e:
            import traceback
            self._on_training_error(f"Training Fehler: {e}\n\n{traceback.format_exc()}")

        finally:
            # Cleanup
            if self.device.type == 'cuda':
                if self.model is not None:
                    try:
                        self.model.cpu()
                    except Exception:
                        pass
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()

    def _stop_training(self):
        """Stoppt das Training."""
        self._stop_requested = True  # Fuer synchrones Training
        if self.worker:
            self.worker.stop()
        self._log("Training wird gestoppt...")

    def _on_epoch_completed(self, epoch: int, train_loss: float, train_acc: float,
                            val_loss: float, val_acc: float):
        """Callback nach jeder Epoche."""
        # History aktualisieren (Accuracy kommt bereits als Prozent vom Trainer)
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)

        # Labels aktualisieren
        self.epoch_label.setText(f"Epoch: {epoch} / {self.epochs_spin.value()}")
        self.progress_bar.setValue(epoch)

        self.train_loss_label.setText(f"Train Loss: {train_loss:.4f}")
        self.train_acc_label.setText(f"Train Acc: {train_acc:.1f}%")
        self.val_loss_label.setText(f"Val Loss: {val_loss:.4f}")
        self.val_acc_label.setText(f"Val Acc: {val_acc:.1f}%")

        # Tabelle aktualisieren
        row = self.metrics_table.rowCount()
        self.metrics_table.insertRow(row)
        self.metrics_table.setItem(row, 0, QTableWidgetItem(str(epoch)))
        self.metrics_table.setItem(row, 1, QTableWidgetItem(f"{train_loss:.4f}"))
        self.metrics_table.setItem(row, 2, QTableWidgetItem(f"{train_acc:.1f}%"))
        self.metrics_table.setItem(row, 3, QTableWidgetItem(f"{val_loss:.4f}"))
        self.metrics_table.setItem(row, 4, QTableWidgetItem(f"{val_acc:.1f}%"))
        self.metrics_table.scrollToBottom()

        # Plot aktualisieren
        self._update_plot()

        # Log
        self._log(f"Epoch {epoch}: Loss={train_loss:.4f}, Acc={train_acc:.1f}%, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.1f}%")

    def _on_training_finished(self, results: dict):
        """Callback wenn Training abgeschlossen."""
        if hasattr(self, 'timer') and self.timer:
            self.timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        # best_accuracy kommt bereits als Prozentwert vom Trainer
        best_acc = results.get('best_accuracy', 0)
        model_path = results.get('model_path', '')

        self._log(f"\nTraining abgeschlossen!", level='SUCCESS')
        self._log(f"Beste Validation Accuracy: {best_acc:.1f}%", level='SUCCESS')

        if results.get('stopped_early'):
            self._log("(Early Stopping aktiviert)")

        if model_path:
            self._log(f"Modell gespeichert: {model_path}", level='SUCCESS')

        # Signal an MainWindow senden
        self.training_completed.emit(self.model, results)

        QMessageBox.information(
            self, "Training abgeschlossen",
            f"Training erfolgreich!\n\nBeste Accuracy: {best_acc:.1f}%\n\nModell: {model_path}"
        )

    def _save_model_to_session(self, model_path: Path):
        """Kopiert das Modell in den Session-Ordner."""
        try:
            from ..core.logger import get_logger
            from ..core.session_manager import SessionManager

            logger = get_logger()
            session_dir = logger.get_session_dir()

            if session_dir is None:
                self._log("Session-Ordner nicht verfuegbar", level='WARNING')
                return

            manager = SessionManager(session_dir)
            dest_path = manager.save_model(model_path)
            self._log(f"Modell in Session kopiert: {dest_path.name}")

        except Exception as e:
            self._log(f"Modell-Kopie in Session fehlgeschlagen: {e}", level='WARNING')

    def _on_training_error(self, error: str):
        """Callback bei Fehler."""
        if hasattr(self, 'timer') and self.timer:
            self.timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        self._log(f"FEHLER: {error}", level='ERROR')
        QMessageBox.critical(self, "Training-Fehler", error)

    def _update_time(self):
        """Aktualisiert die Zeit-Anzeige."""
        if hasattr(self, '_start_time'):
            elapsed = datetime.now() - self._start_time
            minutes, seconds = divmod(int(elapsed.total_seconds()), 60)
            hours, minutes = divmod(minutes, 60)
            self.time_label.setText(f"Zeit: {hours:02d}:{minutes:02d}:{seconds:02d}")


    def get_model(self):
        """Gibt das trainierte Modell zurueck."""
        return self.model

    def _update_device(self, state: int):
        """Aktualisiert das Device basierend auf GPU-Checkbox."""
        if state == Qt.CheckState.Checked.value and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.device_label.setText(torch.cuda.get_device_name(0))
            self.device_label.setStyleSheet("color: #33b34d;")
            self._log("Device gewechselt: GPU (CUDA)")
        else:
            self.device = torch.device('cpu')
            self.device_label.setText("CPU")
            self.device_label.setStyleSheet("color: #e6b333;")
            self._log("Device gewechselt: CPU")

    def _update_gpu_memory(self):
        """Aktualisiert die GPU-Speicheranzeige."""
        if not torch.cuda.is_available() or not hasattr(self, 'gpu_memory_bar'):
            return

        try:
            # Speicher-Informationen abrufen
            allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)  # GB
            reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)  # GB
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
            free = total - reserved

            # Prozent berechnen (reserved statt allocated fuer genauere Anzeige)
            percent = int((reserved / total) * 100)

            # Progress Bar aktualisieren
            self.gpu_memory_bar.setValue(percent)

            # Farbe basierend auf Auslastung
            if percent < 50:
                color = "#33b34d"  # Gruen
            elif percent < 80:
                color = "#e6b333"  # Orange
            else:
                color = "#cc4d33"  # Rot

            self.gpu_memory_bar.setStyleSheet(f"""
                QProgressBar {{
                    border: 1px solid #555555;
                    border-radius: 3px;
                    background-color: #1a1a1a;
                    text-align: center;
                    color: white;
                }}
                QProgressBar::chunk {{
                    background-color: {color};
                }}
            """)

            # Labels aktualisieren
            self.gpu_used_label.setText(f"Belegt: {reserved:.1f} GB")
            self.gpu_free_label.setText(f"Frei: {free:.1f} GB")

        except Exception as e:
            # Bei Fehler stumm ignorieren
            pass

    def _run_gpu_test(self):
        """
        Fuehrt einen GPU-Trainingstest mit synthetischen Daten durch.
        Generiert 1000 Samples mit 4 Features und trainiert 5 Epochen.
        """
        if not torch.cuda.is_available():
            QMessageBox.warning(self, "GPU Test", "Keine GPU verfuegbar!")
            return

        self._log("=== GPU Test gestartet ===")
        self.gpu_test_btn.setEnabled(False)
        self.gpu_test_btn.setText("Test laeuft...")

        try:
            # GPU aufräumen
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Synthetische Daten generieren: 1000 Samples, Lookback=50, 4 Features
            num_samples = 1000
            lookback = 50
            num_features = 4
            num_classes = 3

            self._log(f"Generiere {num_samples} synthetische Samples...")
            self._log(f"Shape: ({num_samples}, {lookback}, {num_features})")

            # Zufaellige Sequenzen
            X = np.random.randn(num_samples, lookback, num_features).astype(np.float32)
            # Zufaellige Labels (0=HOLD, 1=BUY, 2=SELL)
            Y = np.random.randint(0, num_classes, size=num_samples)

            # In Tensoren konvertieren und auf GPU laden
            X_tensor = torch.FloatTensor(X)
            Y_tensor = torch.LongTensor(Y)

            self._log(f"Daten auf GPU laden...")

            # Train/Val Split (80/20)
            split_idx = int(num_samples * 0.8)
            X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
            Y_train, Y_val = Y_tensor[:split_idx], Y_tensor[split_idx:]

            from torch.utils.data import TensorDataset, DataLoader

            train_dataset = TensorDataset(X_train, Y_train)
            val_dataset = TensorDataset(X_val, Y_val)

            batch_size = 32
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

            self._log(f"DataLoader erstellt: {len(train_dataset)} Train, {len(val_dataset)} Val")

            # Kleines Test-Modell erstellen
            from ..models import ModelFactory

            model = ModelFactory.create(
                'bilstm',
                input_size=num_features,
                hidden_size=32,  # Klein fuer Test
                num_layers=1,
                num_classes=num_classes,
                dropout=0.1
            )

            self._log(f"Modell: {model.name}, Parameter: {sum(p.numel() for p in model.parameters()):,}")

            # Auf GPU verschieben
            device = torch.device('cuda')
            model = model.to(device)
            self._log(f"Modell auf GPU geladen")

            # Loss und Optimizer (mit Class Weights falls vorhanden)
            class_weights = self.training_data.get('class_weights') if self.training_data else None
            if class_weights is not None:
                class_weights = class_weights.to(device)
                criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
            else:
                criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # 5 Test-Epochen trainieren
            num_epochs = 5
            self._log(f"Starte Training: {num_epochs} Epochen...")

            for epoch in range(1, num_epochs + 1):
                model.train()
                train_loss = 0.0
                correct = 0
                total = 0

                for batch_idx, (data, target) in enumerate(train_loader):
                    # Auf GPU
                    data = data.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)

                    # Forward
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)

                    # Backward
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()

                    # Periodisch synchronisieren
                    if batch_idx % 10 == 0:
                        torch.cuda.synchronize()

                train_acc = 100.0 * correct / total
                avg_loss = train_loss / len(train_loader)

                # GPU Memory Status
                allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)

                self._log(f"Epoch {epoch}/{num_epochs}: Loss={avg_loss:.4f}, Acc={train_acc:.1f}%, "
                         f"GPU: {allocated:.2f}GB alloc / {reserved:.2f}GB reserved")

            # Aufräumen
            model.cpu()
            del model, optimizer, criterion
            del train_loader, val_loader, train_dataset, val_dataset
            del X_tensor, Y_tensor, X_train, X_val, Y_train, Y_val

            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            import gc
            gc.collect()

            self._log("=== GPU Test ERFOLGREICH ===", level='SUCCESS')
            QMessageBox.information(
                self, "GPU Test",
                f"GPU Test erfolgreich!\n\n"
                f"- {num_samples} Samples trainiert\n"
                f"- {num_epochs} Epochen ohne Fehler\n"
                f"- GPU: {torch.cuda.get_device_name(0)}\n\n"
                f"GPU funktioniert korrekt."
            )

        except torch.cuda.OutOfMemoryError as e:
            self._log(f"GPU Out of Memory: {e}", level='ERROR')
            torch.cuda.empty_cache()
            QMessageBox.critical(
                self, "GPU Test Fehler",
                f"GPU Out of Memory!\n\n{e}\n\nReduziere Batch Size oder Modellgroesse."
            )

        except RuntimeError as e:
            error_str = str(e)
            self._log(f"Runtime Error: {error_str}", level='ERROR')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            QMessageBox.critical(
                self, "GPU Test Fehler",
                f"CUDA/Runtime Fehler:\n\n{error_str}\n\n"
                f"Moegliche Ursachen:\n"
                f"- GPU-Treiber veraltet\n"
                f"- CUDA Version inkompatibel\n"
                f"- GPU wird von anderem Prozess verwendet"
            )

        except Exception as e:
            import traceback
            self._log(f"Fehler: {e}\n{traceback.format_exc()}", level='ERROR')
            QMessageBox.critical(self, "GPU Test Fehler", f"Unerwarteter Fehler:\n\n{e}")

        finally:
            self.gpu_test_btn.setEnabled(True)
            self.gpu_test_btn.setText("GPU Test")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _create_auto_trainer_tab(self) -> QWidget:
        """Erstellt den Auto-Trainer Ergebnisse Tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Splitter fuer Tabelle links und Detail rechts
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Linke Seite: Ergebnis-Tabelle
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        table_layout.setContentsMargins(0, 0, 0, 0)

        self.auto_results_table = QTableWidget()
        self.auto_results_table.setColumnCount(8)
        self.auto_results_table.setHorizontalHeaderLabels([
            'Rang', 'Modell', 'Config', 'Train ACC', 'Val ACC', 'F1', 'Epochen', 'Parameter'
        ])
        self.auto_results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.auto_results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.auto_results_table.itemSelectionChanged.connect(self._on_auto_result_selected)
        table_layout.addWidget(self.auto_results_table)

        splitter.addWidget(table_widget)

        # Rechte Seite: Trainingsverlauf Detail
        detail_widget = QWidget()
        detail_layout = QVBoxLayout(detail_widget)
        detail_layout.setContentsMargins(5, 0, 0, 0)

        # Header
        detail_header = QLabel("Trainingsverlauf")
        detail_header.setStyleSheet("font-weight: bold; font-size: 12px;")
        detail_layout.addWidget(detail_header)

        # Modell-Info
        self.detail_model_label = QLabel("Modell: -")
        detail_layout.addWidget(self.detail_model_label)

        self.detail_config_label = QLabel("Config: -")
        detail_layout.addWidget(self.detail_config_label)

        # Metriken
        metrics_group = QGroupBox("Metriken")
        metrics_layout = QGridLayout(metrics_group)

        metrics_layout.addWidget(QLabel("Best Epoch:"), 0, 0)
        self.detail_best_epoch_label = QLabel("-")
        metrics_layout.addWidget(self.detail_best_epoch_label, 0, 1)

        metrics_layout.addWidget(QLabel("Trainingszeit:"), 1, 0)
        self.detail_time_label = QLabel("-")
        metrics_layout.addWidget(self.detail_time_label, 1, 1)

        metrics_layout.addWidget(QLabel("F1-Score:"), 2, 0)
        self.detail_f1_label = QLabel("-")
        metrics_layout.addWidget(self.detail_f1_label, 2, 1)

        detail_layout.addWidget(metrics_group)

        # Trainingsverlauf Chart (als Text-Tabelle)
        history_group = QGroupBox("Verlauf pro Epoche")
        history_layout = QVBoxLayout(history_group)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels(['Epoche', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc'])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.history_table.setMaximumHeight(200)
        history_layout.addWidget(self.history_table)

        detail_layout.addWidget(history_group)

        # Precision/Recall pro Klasse
        class_group = QGroupBox("Precision/Recall pro Klasse")
        class_layout = QVBoxLayout(class_group)

        self.class_metrics_table = QTableWidget()
        self.class_metrics_table.setColumnCount(3)
        self.class_metrics_table.setHorizontalHeaderLabels(['Klasse', 'Precision', 'Recall'])
        self.class_metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.class_metrics_table.setMaximumHeight(100)
        class_layout.addWidget(self.class_metrics_table)

        detail_layout.addWidget(class_group)

        detail_layout.addStretch()

        splitter.addWidget(detail_widget)

        # Splitter-Verhaeltnis (60% Tabelle, 40% Detail)
        splitter.setSizes([600, 400])

        layout.addWidget(splitter)

        # Buttons
        btn_layout = QHBoxLayout()
        self.adopt_best_btn = QPushButton("Bestes Modell uebernehmen")
        self.adopt_best_btn.clicked.connect(self._adopt_best_model)
        self.adopt_best_btn.setEnabled(False)
        btn_layout.addWidget(self.adopt_best_btn)

        self.export_results_btn = QPushButton("Ergebnisse exportieren")
        self.export_results_btn.clicked.connect(self._export_auto_results)
        self.export_results_btn.setEnabled(False)
        btn_layout.addWidget(self.export_results_btn)

        layout.addLayout(btn_layout)

        return widget

    def _on_auto_result_selected(self):
        """Zeigt Details zum ausgewaehlten Auto-Training Ergebnis."""
        if not hasattr(self, 'auto_trainer') or not self.auto_trainer.results:
            return

        selected = self.auto_results_table.selectedItems()
        if not selected:
            return

        row = selected[0].row()
        if row >= len(self.auto_trainer.results):
            return

        result = self.auto_trainer.results[row]

        # Modell-Info aktualisieren
        self.detail_model_label.setText(f"Modell: {result.model_type}")

        if 'hidden_sizes' in result.config:
            config_str = f"hidden_sizes: {result.config['hidden_sizes']}"
        elif 'd_model' in result.config:
            config_str = f"d_model: {result.config.get('d_model')}, nhead: {result.config.get('nhead', 4)}"
        else:
            config_str = str(result.config)
        self.detail_config_label.setText(f"Config: {config_str}")

        # Metriken
        self.detail_best_epoch_label.setText(f"{result.best_epoch}")
        self.detail_time_label.setText(f"{result.training_time:.1f}s")
        self.detail_f1_label.setText(f"{result.f1_score:.4f}")

        # Trainingsverlauf
        history = result.training_history
        self.history_table.setRowCount(len(history.get('train_loss', [])))

        for i in range(len(history.get('train_loss', []))):
            self.history_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.history_table.setItem(i, 1, QTableWidgetItem(f"{history['train_loss'][i]:.4f}"))
            self.history_table.setItem(i, 2, QTableWidgetItem(f"{history['train_acc'][i]:.2%}"))
            self.history_table.setItem(i, 3, QTableWidgetItem(f"{history['val_loss'][i]:.4f}"))
            self.history_table.setItem(i, 4, QTableWidgetItem(f"{history['val_acc'][i]:.2%}"))

        # Ans Ende scrollen (beste Epochen sind meist am Ende)
        self.history_table.scrollToBottom()

        # Precision/Recall pro Klasse
        self.class_metrics_table.setRowCount(len(result.precision))
        for i, (class_name, prec) in enumerate(result.precision.items()):
            rec = result.recall.get(class_name, 0.0)
            self.class_metrics_table.setItem(i, 0, QTableWidgetItem(class_name))
            self.class_metrics_table.setItem(i, 1, QTableWidgetItem(f"{prec:.4f}"))
            self.class_metrics_table.setItem(i, 2, QTableWidgetItem(f"{rec:.4f}"))

    def _on_model_changed(self, model_name: str):
        """Aktualisiert UI basierend auf Modell-Auswahl."""
        is_lstm_gru = ModelFactory.model_requires_hidden_sizes(model_name)
        is_transformer = ModelFactory.model_is_transformer(model_name)

        # LSTM/GRU spezifische UI
        self.preset_combo.setVisible(is_lstm_gru or is_transformer)
        self.hidden_sizes_edit.setVisible(is_lstm_gru)

        # Transformer spezifische UI
        self.d_model_label.setVisible(is_transformer)
        self.d_model_spin.setVisible(is_transformer)
        self.nhead_label.setVisible(is_transformer)
        self.nhead_spin.setVisible(is_transformer)
        self.encoder_layers_label.setVisible(is_transformer)
        self.encoder_layers_spin.setVisible(is_transformer)

        # Legacy Controls
        self.hidden_size_spin.setVisible(not is_lstm_gru and not is_transformer)
        self.num_layers_spin.setVisible(not is_lstm_gru and not is_transformer)

        # Erweiterte Optionen nur fuer LSTM/GRU
        if hasattr(self, 'use_layer_norm_check'):
            self.use_layer_norm_check.setEnabled(is_lstm_gru)
            self.use_attention_check.setEnabled(is_lstm_gru)
            self.use_residual_check.setEnabled(is_lstm_gru)
            self.attention_heads_spin.setEnabled(is_lstm_gru)

        # Presets aktualisieren
        presets = ModelFactory.get_presets(model_name)
        self.preset_combo.clear()
        self.preset_combo.addItems(list(presets.keys()))

    def _on_preset_changed(self, preset_name: str):
        """Aktualisiert Parameter basierend auf Preset."""
        model_name = self.model_combo.currentText()
        presets = ModelFactory.get_presets(model_name)

        if preset_name == 'Custom' or preset_name not in presets or presets[preset_name] is None:
            return

        params = presets[preset_name]

        # LSTM/GRU Presets
        if 'hidden_sizes' in params:
            hidden_sizes = params['hidden_sizes']
            self.hidden_sizes_edit.setText(', '.join(map(str, hidden_sizes)))

        # Transformer Presets
        if 'd_model' in params:
            self.d_model_spin.setValue(params['d_model'])
        if 'nhead' in params:
            self.nhead_spin.setValue(params['nhead'])
        if 'num_encoder_layers' in params:
            self.encoder_layers_spin.setValue(params['num_encoder_layers'])

    def _on_auto_trainer_toggled(self, state: int):
        """Aktiviert/deaktiviert Auto-Trainer Modus."""
        is_auto = state == Qt.CheckState.Checked.value

        # Manuelle Parameter deaktivieren wenn Auto-Modus aktiv
        self.model_combo.setEnabled(not is_auto)
        self.preset_combo.setEnabled(not is_auto)
        self.hidden_sizes_edit.setEnabled(not is_auto)
        self.hidden_size_spin.setEnabled(not is_auto)
        self.num_layers_spin.setEnabled(not is_auto)
        self.d_model_spin.setEnabled(not is_auto)
        self.nhead_spin.setEnabled(not is_auto)
        self.encoder_layers_spin.setEnabled(not is_auto)

        # Auto-Trainer spezifische Controls aktivieren
        self.complexity_slider.setEnabled(is_auto)
        self.auto_batch_spin.setEnabled(is_auto)
        self.amp_check.setEnabled(is_auto)

        if is_auto:
            self.start_btn.setText("Auto-Training starten")
        else:
            self.start_btn.setText("Training starten")

    def _on_complexity_changed(self, value: int):
        """Aktualisiert Komplexitaets-Label."""
        from ..training.auto_trainer import get_complexity_info

        info = get_complexity_info(value)
        labels = {
            1: "Schnell",
            2: "Schnell+",
            3: "Standard",
            4: "Ausfuehrlich",
            5: "Gruendlich"
        }
        self.complexity_label.setText(
            f"{labels.get(value, 'Standard')} ({info['num_configs']} Configs, max {info['max_epochs']} Epochen)"
        )

    def _parse_hidden_sizes(self) -> List[int]:
        """Parst hidden_sizes aus dem Textfeld."""
        text = self.hidden_sizes_edit.text().strip()
        if not text:
            return [128, 128]

        try:
            sizes = [int(s.strip()) for s in text.split(',') if s.strip()]
            return sizes if sizes else [128, 128]
        except ValueError:
            return [128, 128]

    def _run_auto_training(self):
        """Fuehrt Auto-Training durch."""
        from PyQt6.QtWidgets import QApplication
        from ..training.auto_trainer import AutoTrainer

        if self.training_data is None:
            QMessageBox.warning(self, "Fehler", "Keine Trainingsdaten geladen!")
            return

        self._cleanup_previous_training()

        # Batch-Size und AMP aus GUI
        batch_size = self.auto_batch_spin.value()
        use_amp = self.amp_check.isChecked()

        # DataLoader mit neuer Batch-Size erstellen
        self._log(f"Erstelle DataLoader mit Batch-Size {batch_size}...")
        self.prepare_data_loaders(self.training_data, batch_size=batch_size)

        # Input Size ermitteln
        sample = next(iter(self.train_loader))
        input_size = sample[0].shape[-1]
        num_classes = self.training_info.get('num_classes', 3) if self.training_info else 3

        complexity = self.complexity_slider.value()

        self._log(f"=== Auto-Training gestartet (Komplexitaet {complexity}) ===")
        self._log(f"Batch-Size: {batch_size}, Mixed Precision: {'Ja' if use_amp else 'Nein'}")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._start_time = datetime.now()

        # Timer starten
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_time)
        self.timer.start(1000)

        try:
            # AutoTrainer erstellen
            self.auto_trainer = AutoTrainer(
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                input_size=input_size,
                num_classes=num_classes,
                device=self.device,
                use_amp=use_amp
            )

            # Progress Callback
            def progress_callback(current, total, model_name, metrics):
                self.epoch_label.setText(f"Modell: {current+1}/{total} - {model_name}")
                # Gesamtfortschritt: abgeschlossene Modelle + Fortschritt des aktuellen Modells
                overall_progress = (current + metrics.get('progress', 0)) / total * 100
                self.progress_bar.setMaximum(100)
                self.progress_bar.setValue(int(overall_progress))

                # Metriken anzeigen
                if 'train_loss' in metrics:
                    self.train_loss_label.setText(f"Train Loss: {metrics['train_loss']:.4f}")
                    self.train_acc_label.setText(f"Train Acc: {metrics['train_acc']:.2%}")
                    self.val_loss_label.setText(f"Val Loss: {metrics['val_loss']:.4f}")
                    self.val_acc_label.setText(f"Val Acc: {metrics['val_acc']:.2%}")

                QApplication.processEvents()

                if self._stop_requested:
                    self.auto_trainer.stop()

            # Training ausfuehren
            results = self.auto_trainer.run(
                complexity=complexity,
                progress_callback=progress_callback,
                learning_rate=self.lr_spin.value()
            )

            # Progress auf 100% setzen
            self.progress_bar.setValue(100)
            self.epoch_label.setText("Auto-Training abgeschlossen")

            # Ergebnisse anzeigen
            self._display_auto_results(results)

            self._log(f"Auto-Training abgeschlossen: {len(results)} Modelle getestet")
            if results:
                best = results[0]
                self._log(f"Bestes Modell: {best.model_type} - Val ACC: {best.val_acc:.2%}, F1: {best.f1_score:.3f}")

        except Exception as e:
            import traceback
            self._log(f"Auto-Training Fehler: {e}\n{traceback.format_exc()}", level='ERROR')
            QMessageBox.critical(self, "Fehler", f"Auto-Training fehlgeschlagen:\n{e}")

        finally:
            self.timer.stop()
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self._stop_requested = False

    def _display_auto_results(self, results):
        """Zeigt Auto-Training Ergebnisse in der Tabelle an."""
        self.auto_results_table.setRowCount(0)

        for r in results:
            row = self.auto_results_table.rowCount()
            self.auto_results_table.insertRow(row)

            # Config String erstellen
            if 'hidden_sizes' in r.config:
                config_str = str(r.config['hidden_sizes'])
            elif 'd_model' in r.config:
                config_str = f"d={r.config.get('d_model')}, h={r.config.get('nhead', 4)}"
            else:
                config_str = str(r.config)[:25]

            self.auto_results_table.setItem(row, 0, QTableWidgetItem(str(r.rank)))
            self.auto_results_table.setItem(row, 1, QTableWidgetItem(r.model_type))
            self.auto_results_table.setItem(row, 2, QTableWidgetItem(config_str))
            self.auto_results_table.setItem(row, 3, QTableWidgetItem(f"{r.train_acc:.2%}"))
            self.auto_results_table.setItem(row, 4, QTableWidgetItem(f"{r.val_acc:.2%}"))
            self.auto_results_table.setItem(row, 5, QTableWidgetItem(f"{r.f1_score:.3f}"))
            self.auto_results_table.setItem(row, 6, QTableWidgetItem(str(r.epochs_trained)))
            self.auto_results_table.setItem(row, 7, QTableWidgetItem(f"{r.num_parameters:,}"))

        self.adopt_best_btn.setEnabled(len(results) > 0)
        self.export_results_btn.setEnabled(len(results) > 0)

    def _adopt_best_model(self):
        """Uebernimmt das beste Modell aus dem Auto-Training."""
        if not hasattr(self, 'auto_trainer') or not self.auto_trainer.results:
            return

        try:
            self.model, config = self.auto_trainer.get_best_model()
            best = self.auto_trainer.results[0]

            # Model-Combo aktualisieren
            index = self.model_combo.findText(best.model_type, Qt.MatchFlag.MatchFixedString)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)

            # Parameter setzen
            if 'hidden_sizes' in config:
                self.hidden_sizes_edit.setText(', '.join(map(str, config['hidden_sizes'])))

            self._log(f"Bestes Modell uebernommen: {best.model_type}", level='SUCCESS')
            self.training_completed.emit(self.model, {'best_accuracy': best.val_acc * 100})

            QMessageBox.information(
                self, "Modell uebernommen",
                f"Bestes Modell: {best.model_type}\n"
                f"Val Accuracy: {best.val_acc:.2%}\n"
                f"F1-Score: {best.f1_score:.3f}"
            )

        except Exception as e:
            self._log(f"Fehler beim Uebernehmen: {e}", level='ERROR')
            QMessageBox.critical(self, "Fehler", f"Modell konnte nicht uebernommen werden:\n{e}")

    def _export_auto_results(self):
        """Exportiert Auto-Training Ergebnisse als JSON."""
        if not hasattr(self, 'auto_trainer') or not self.auto_trainer.results:
            return

        import json

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Ergebnisse speichern", "", "JSON Dateien (*.json)"
        )

        if not filepath:
            return

        try:
            results_data = []
            for r in self.auto_trainer.results:
                results_data.append({
                    'rank': r.rank,
                    'model_type': r.model_type,
                    'config': r.config,
                    'val_acc': r.val_acc,
                    'train_acc': r.train_acc,
                    'f1_score': r.f1_score,
                    'precision': r.precision,
                    'recall': r.recall,
                    'epochs_trained': r.epochs_trained,
                    'training_time': r.training_time,
                    'num_parameters': r.num_parameters
                })

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)

            self._log(f"Ergebnisse exportiert: {filepath}", level='SUCCESS')

        except Exception as e:
            self._log(f"Export fehlgeschlagen: {e}", level='ERROR')

    def closeEvent(self, event):
        """Behandelt das Schliessen des Fensters."""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "Training laeuft",
                "Training laeuft noch. Wirklich beenden?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            self.worker.stop()
            self.worker.wait()
        event.accept()
