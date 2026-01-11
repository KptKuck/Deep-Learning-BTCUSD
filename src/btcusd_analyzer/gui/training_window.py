"""
Training Window - GUI fuer Modell-Training mit Live-Visualisierung
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

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

            # Callbacks
            callbacks = []
            if self.config.get('early_stopping', True):
                callbacks.append(EarlyStopping(
                    patience=self.config.get('patience', 10),
                    monitor='val_loss'
                ))

            if self.config.get('save_best', True):
                save_path = Path(self.config.get('save_path', 'models'))
                callbacks.append(ModelCheckpoint(
                    save_path=save_path,
                    monitor='val_accuracy'
                ))

            # Trainer erstellen
            trainer = Trainer(
                model=self.model,
                device=self.device,
                callbacks=callbacks
            )

            # Custom epoch callback fuer GUI-Updates
            def on_epoch_end(epoch, logs):
                if self._stop_requested:
                    trainer.stop_training = True
                    return

                self.epoch_completed.emit(
                    epoch,
                    logs.get('train_loss', 0),
                    logs.get('train_accuracy', 0),
                    logs.get('val_loss', 0),
                    logs.get('val_accuracy', 0)
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
                on_epoch_end=on_epoch_end
            )

            # Ergebnis senden
            self.training_finished.emit({
                'history': history,
                'best_accuracy': max(history.get('val_accuracy', [0])),
                'final_loss': history.get('val_loss', [0])[-1] if history.get('val_loss') else 0,
                'stopped_early': trainer.stop_training
            })

        except Exception as e:
            self.training_error.emit(str(e))

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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("BTCUSD Analyzer - Training")
        self.setMinimumSize(1200, 800)

        # State
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.worker = None
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._setup_ui()
        self.setStyleSheet(get_stylesheet())

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
        layout.setSpacing(10)

        # Modell-Auswahl
        model_group = QGroupBox("Modell")
        model_layout = QGridLayout(model_group)

        model_layout.addWidget(QLabel("Architektur:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            'BiLSTM', 'LSTM', 'GRU', 'BiGRU', 'CNN', 'CNN-LSTM',
            'TCN', 'Transformer', 'Informer', 'TFT', 'N-BEATS'
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

        # Early Stopping
        es_group = QGroupBox("Early Stopping")
        es_layout = QGridLayout(es_group)

        self.early_stopping_check = QCheckBox("Aktivieren")
        self.early_stopping_check.setChecked(True)
        es_layout.addWidget(self.early_stopping_check, 0, 0, 1, 2)

        es_layout.addWidget(QLabel("Patience:"), 1, 0)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 50)
        self.patience_spin.setValue(10)
        es_layout.addWidget(self.patience_spin, 1, 1)

        layout.addWidget(es_group)

        # Speicher-Optionen
        save_group = QGroupBox("Speichern")
        save_layout = QVBoxLayout(save_group)

        self.save_best_check = QCheckBox("Bestes Modell speichern")
        self.save_best_check.setChecked(True)
        save_layout.addWidget(self.save_best_check)

        self.save_history_check = QCheckBox("Training-History speichern")
        self.save_history_check.setChecked(True)
        save_layout.addWidget(self.save_history_check)

        save_path_layout = QHBoxLayout()
        self.save_path_edit = QLabel("models/")
        self.save_path_btn = QPushButton("...")
        self.save_path_btn.setFixedWidth(40)
        self.save_path_btn.clicked.connect(self._select_save_path)
        save_path_layout.addWidget(QLabel("Pfad:"))
        save_path_layout.addWidget(self.save_path_edit, 1)
        save_path_layout.addWidget(self.save_path_btn)
        save_layout.addLayout(save_path_layout)

        layout.addWidget(save_group)

        # Device Info (GPU Status wie MATLAB)
        device_group = QGroupBox("GPU Status & Speicher")
        device_layout = QVBoxLayout(device_group)

        # GPU/CPU Switch
        switch_layout = QHBoxLayout()
        switch_layout.addWidget(QLabel("Device:"))
        self.use_gpu_check = QCheckBox("GPU verwenden")
        self.use_gpu_check.setChecked(torch.cuda.is_available())
        self.use_gpu_check.setEnabled(torch.cuda.is_available())
        self.use_gpu_check.stateChanged.connect(self._update_device)
        switch_layout.addWidget(self.use_gpu_check)
        device_layout.addLayout(switch_layout)

        # GPU Name
        device_text = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        if torch.cuda.is_available():
            device_text = f"{torch.cuda.get_device_name(0)}"
        self.device_label = QLabel(device_text)
        self.device_label.setStyleSheet(f"color: {'#33b34d' if torch.cuda.is_available() else '#e6b333'};")
        device_layout.addWidget(self.device_label)

        # GPU Memory Bar (wie MATLAB)
        if torch.cuda.is_available():
            mem_layout = QGridLayout()

            # Progress Bar fuer GPU-Speicher
            self.gpu_memory_bar = QProgressBar()
            self.gpu_memory_bar.setMinimum(0)
            self.gpu_memory_bar.setMaximum(100)
            self.gpu_memory_bar.setValue(0)
            self.gpu_memory_bar.setTextVisible(True)
            self.gpu_memory_bar.setFormat("%p% belegt")
            mem_layout.addWidget(self.gpu_memory_bar, 0, 0, 1, 2)

            # Speicher-Labels
            self.gpu_used_label = QLabel("Belegt: - GB")
            self.gpu_used_label.setStyleSheet("color: #aaaaaa;")
            mem_layout.addWidget(self.gpu_used_label, 1, 0)

            self.gpu_free_label = QLabel("Frei: - GB")
            self.gpu_free_label.setStyleSheet("color: #33b34d;")
            mem_layout.addWidget(self.gpu_free_label, 1, 1)

            device_layout.addLayout(mem_layout)

            # GPU Memory Timer starten
            self.gpu_timer = QTimer()
            self.gpu_timer.timeout.connect(self._update_gpu_memory)
            self.gpu_timer.start(1000)
            self._update_gpu_memory()

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

    def _start_training(self):
        """Startet das Training."""
        if self.train_loader is None or self.val_loader is None:
            QMessageBox.warning(self, "Fehler", "Keine Trainingsdaten geladen!")
            return

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

            self.model = ModelFactory.create(
                model_name,
                input_size=input_size,
                hidden_size=self.hidden_size_spin.value(),
                num_layers=self.num_layers_spin.value(),
                num_classes=3,
                dropout=self.dropout_spin.value()
            )

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

        # Worker starten
        self.worker = TrainingWorker(
            self.model, self.train_loader, self.val_loader, config, self.device
        )
        self.worker.epoch_completed.connect(self._on_epoch_completed)
        self.worker.training_finished.connect(self._on_training_finished)
        self.worker.training_error.connect(self._on_training_error)
        self.worker.log_message.connect(self._log)
        self.worker.start()

        # UI aktualisieren
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setMaximum(config['epochs'])
        self._start_time = datetime.now()

        # Timer fuer Zeit-Update
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_time)
        self.timer.start(1000)

    def _stop_training(self):
        """Stoppt das Training."""
        if self.worker:
            self.worker.stop()
            self._log("Training wird gestoppt...")

    def _on_epoch_completed(self, epoch: int, train_loss: float, train_acc: float,
                            val_loss: float, val_acc: float):
        """Callback nach jeder Epoche."""
        # History aktualisieren
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc * 100)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc * 100)

        # Labels aktualisieren
        self.epoch_label.setText(f"Epoch: {epoch} / {self.epochs_spin.value()}")
        self.progress_bar.setValue(epoch)

        self.train_loss_label.setText(f"Train Loss: {train_loss:.4f}")
        self.train_acc_label.setText(f"Train Acc: {train_acc*100:.1f}%")
        self.val_loss_label.setText(f"Val Loss: {val_loss:.4f}")
        self.val_acc_label.setText(f"Val Acc: {val_acc*100:.1f}%")

        # Tabelle aktualisieren
        row = self.metrics_table.rowCount()
        self.metrics_table.insertRow(row)
        self.metrics_table.setItem(row, 0, QTableWidgetItem(str(epoch)))
        self.metrics_table.setItem(row, 1, QTableWidgetItem(f"{train_loss:.4f}"))
        self.metrics_table.setItem(row, 2, QTableWidgetItem(f"{train_acc*100:.1f}%"))
        self.metrics_table.setItem(row, 3, QTableWidgetItem(f"{val_loss:.4f}"))
        self.metrics_table.setItem(row, 4, QTableWidgetItem(f"{val_acc*100:.1f}%"))
        self.metrics_table.scrollToBottom()

        # Plot aktualisieren
        self._update_plot()

        # Log
        self._log(f"Epoch {epoch}: Loss={train_loss:.4f}, Acc={train_acc*100:.1f}%, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc*100:.1f}%")

    def _on_training_finished(self, results: dict):
        """Callback wenn Training abgeschlossen."""
        self.timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        best_acc = results.get('best_accuracy', 0) * 100
        self._log(f"\nTraining abgeschlossen!")
        self._log(f"Beste Validation Accuracy: {best_acc:.1f}%")

        if results.get('stopped_early'):
            self._log("(Early Stopping aktiviert)")

        QMessageBox.information(
            self, "Training abgeschlossen",
            f"Training erfolgreich!\n\nBeste Accuracy: {best_acc:.1f}%"
        )

    def _on_training_error(self, error: str):
        """Callback bei Fehler."""
        self.timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        self._log(f"FEHLER: {error}")
        QMessageBox.critical(self, "Training-Fehler", error)

    def _update_time(self):
        """Aktualisiert die Zeit-Anzeige."""
        if hasattr(self, '_start_time'):
            elapsed = datetime.now() - self._start_time
            minutes, seconds = divmod(int(elapsed.total_seconds()), 60)
            hours, minutes = divmod(minutes, 60)
            self.time_label.setText(f"Zeit: {hours:02d}:{minutes:02d}:{seconds:02d}")

    def _log(self, message: str):
        """Fuegt eine Nachricht zum Log hinzu."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

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
