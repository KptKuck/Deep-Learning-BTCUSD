"""
Training Window - GUI fuer Modell-Training mit Live-Visualisierung
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
    QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont

import torch
import numpy as np

from .styles import get_stylesheet, COLORS


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
        self.setWindowTitle("BTCUSD Analyzer - Training")
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
        # Nur implementierte Modelle anzeigen
        self.model_combo.addItems([
            'BiLSTM', 'LSTM', 'GRU', 'BiGRU', 'CNN', 'CNN-LSTM'
        ])
        model_layout.addWidget(self.model_combo, 0, 1)

        model_layout.addWidget(QLabel("Hidden Size:"), 1, 0)
        self.hidden_size_spin = QSpinBox()
        self.hidden_size_spin.setRange(16, 512)
        self.hidden_size_spin.setValue(100)
        self.hidden_size_spin.setSingleStep(16)
        model_layout.addWidget(self.hidden_size_spin, 1, 1)

        model_layout.addWidget(QLabel("Layers:"), 2, 0)
        self.num_layers_spin = QSpinBox()
        self.num_layers_spin.setRange(1, 6)
        self.num_layers_spin.setValue(2)
        model_layout.addWidget(self.num_layers_spin, 2, 1)

        model_layout.addWidget(QLabel("Dropout:"), 3, 0)
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.8)
        self.dropout_spin.setValue(0.2)
        self.dropout_spin.setSingleStep(0.05)
        model_layout.addWidget(self.dropout_spin, 3, 1)

        layout.addWidget(model_group)

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

        # Altes Modell und GPU-Speicher aufraeumen vor neuem Training
        self._cleanup_previous_training()

        # Modell erstellen
        try:
            from ..models import ModelFactory

            model_name = self.model_combo.currentText().lower().replace('-', '_')
            if model_name == 'bilstm':
                model_name = 'bilstm'
            elif model_name == 'bigru':
                model_name = 'bigru'
            elif model_name == 'cnn_lstm':
                model_name = 'cnn_lstm'
            elif model_name == 'n_beats':
                model_name = 'nbeats'

            # Input-Size aus Daten ermitteln
            sample = next(iter(self.train_loader))
            input_size = sample[0].shape[-1]

            # num_classes aus training_info holen (falls vorhanden), sonst default 3
            num_classes = 3
            if self.training_info and 'num_classes' in self.training_info:
                num_classes = self.training_info['num_classes']

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
        import gc

        self._log("=== Synchrones Training gestartet ===")
        self._stop_requested = False

        try:
            # Modell auf GPU verschieben
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                self.model = self.model.to(self.device)
                self._log(f"Modell auf GPU verschoben: {self.device}")

            # Loss und Optimizer
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

                # Early Stopping Check
                if config.get('early_stopping', True):
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            self._log(f"Early Stopping nach {epoch} Epochen")
                            break

            # Modell speichern
            if config.get('save_best', True):
                save_path = Path(config.get('save_path', 'models'))
                save_path.mkdir(parents=True, exist_ok=True)
                model_name = self.model.name.lower().replace(' ', '_')
                final_path = save_path / f'{model_name}_final_acc{best_val_acc:.1f}.pt'
                self.model.save(final_path, metrics={
                    'best_accuracy': best_val_acc,
                    'epochs_trained': epoch
                })
                self._log(f"Modell gespeichert: {final_path}")

            # Training beendet
            self._on_training_finished({
                'best_accuracy': best_val_acc,
                'final_loss': avg_val_loss,
                'stopped_early': patience_counter >= patience,
                'model_path': str(final_path) if config.get('save_best', True) else ''
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

            # Loss und Optimizer
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
