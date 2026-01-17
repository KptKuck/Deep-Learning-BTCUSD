"""
Training Window - Hauptfenster fuer Modell-Training.

Refaktorierte Version mit modularer Struktur:
- ConfigPanel: Modell- und Training-Konfiguration
- VisualizationPanel: Plots, Metriken, Log
- AutoTrainerWidget: Auto-Training Ergebnisse
- TrainingWorker: Asynchrones Training
"""

from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import gc
import json
import inspect

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QSplitter, QMessageBox, QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer

import torch
import numpy as np

from ..styles import get_stylesheet, COLORS
from ..save_confirm_dialog import SaveConfirmDialog, ask_save_confirmation
from .config_panel import ConfigPanel
from .visualization_panel import VisualizationPanel
from .auto_trainer_widget import AutoTrainerWidget
from .training_worker import TrainingWorker
from ...models.factory import ModelFactory
from ...core.save_manager import SaveManager, OverwriteAction, SaveCheckResult


class TrainingWindow(QMainWindow):
    """
    Training-Fenster mit Live-Visualisierung.

    Modulare Struktur:
    - Linke Seite: ConfigPanel (Modell, Training, Device, Auto-Trainer)
    - Rechte Seite: VisualizationPanel (Plots, Metriken, Log)
    """

    # Signals
    log_message = pyqtSignal(str, str)  # message, level
    training_completed = pyqtSignal(object, object)  # model, results

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("3 - Training")

        # Relative Fenstergroesse (95% Hoehe, 85% Breite)
        screen = QApplication.primaryScreen()
        if screen:
            screen_rect = screen.availableGeometry()
            window_width = int(screen_rect.width() * 0.85)
            window_height = int(screen_rect.height() * 0.95)
        else:
            window_width, window_height = 1200, 800

        self.setMinimumSize(1000, 700)
        self.resize(window_width, window_height)
        self._parent = parent

        # State
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.worker = None
        self._stop_requested = False
        self._start_time = None

        # Trainingsdaten (werden von MainWindow gesetzt)
        self.training_data = None
        self.training_info = None

        # Speicher-Status
        self._unsaved_model = False
        self._trained_model_info = None  # Modell-Infos nach Training

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._setup_ui()
        self._connect_signals()
        self.setStyleSheet(get_stylesheet())

    def _setup_ui(self):
        """Erstellt die Benutzeroberflaeche."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Linke Seite: Config Panel
        self.config_panel = ConfigPanel()
        self.config_panel.setFixedWidth(350)

        # Rechte Seite: Visualization Panel
        self.viz_panel = VisualizationPanel()

        # Auto-Trainer Widget hinzufuegen
        self.auto_trainer_widget = AutoTrainerWidget()
        self.viz_panel.add_auto_trainer_tab(self.auto_trainer_widget)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.config_panel)
        splitter.addWidget(self.viz_panel)
        splitter.setSizes([350, 850])

        main_layout.addWidget(splitter)

        # Timer fuer Zeit-Update
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_time)

    def _connect_signals(self):
        """Verbindet Signals zwischen Komponenten."""
        # Config Panel Signals
        self.config_panel.start_training.connect(self._start_training)
        self.config_panel.stop_training.connect(self._stop_training)
        self.config_panel.start_auto_training.connect(self._run_auto_training)

        # Speichern-Button (NEU: manuelles Speichern statt Auto-Save)
        if hasattr(self.config_panel, 'save_btn'):
            self.config_panel.save_btn.clicked.connect(self._on_save_clicked)

        # GPU Test Button (falls vorhanden)
        if self.config_panel.gpu_test_btn:
            self.config_panel.gpu_test_btn.clicked.connect(self._run_gpu_test)

        # Auto-Trainer Widget
        self.auto_trainer_widget.adopt_model.connect(self._adopt_auto_model)
        self.auto_trainer_widget.log_message.connect(self._log)

    def _log(self, message: str, level: str = 'INFO'):
        """Loggt eine Nachricht."""
        current_frame = inspect.currentframe()
        caller_frame = current_frame.f_back if current_frame else None
        caller_name = caller_frame.f_code.co_name if caller_frame else 'unknown'
        formatted_message = f'{caller_name}() - {message}'

        if self._parent and hasattr(self._parent, '_log'):
            self._parent._log(f'[Training] {formatted_message}', level)

        self.log_message.emit(formatted_message, level)
        self.viz_panel.log(formatted_message, level)

    def set_data(self, train_loader, val_loader):
        """Setzt die Daten-Loader."""
        self.train_loader = train_loader
        self.val_loader = val_loader
        self._log(f"Daten geladen: {len(train_loader.dataset)} Training, {len(val_loader.dataset)} Validation")

    def prepare_data_loaders(self, training_data: Dict[str, Any], batch_size: int = 64, val_split: float = 0.2):
        """Erstellt DataLoader aus training_data Dictionary."""
        from torch.utils.data import TensorDataset, DataLoader

        X = training_data['X']
        Y = training_data['Y']

        X_tensor = torch.FloatTensor(X)
        Y_tensor = torch.LongTensor(Y)

        n_samples = len(X_tensor)
        n_val = int(n_samples * val_split)
        n_train = n_samples - n_val

        indices = torch.randperm(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_dataset = TensorDataset(X_tensor[train_indices], Y_tensor[train_indices])
        val_dataset = TensorDataset(X_tensor[val_indices], Y_tensor[val_indices])

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            pin_memory=True if torch.cuda.is_available() else False
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            pin_memory=True if torch.cuda.is_available() else False
        )

        self._log(f"DataLoader erstellt: {n_train} Train, {n_val} Val, Batch={batch_size}")

    def _cleanup_previous_training(self):
        """Rauemt vorheriges Training auf."""
        if self.model is not None:
            try:
                self.model.cpu()
            except Exception:
                pass
            del self.model
            self.model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        gc.collect()

    def _start_training(self):
        """Startet das Training."""
        if self.train_loader is None:
            if self.training_data is not None:
                batch_size = self.config_panel.batch_size_spin.value()
                self.prepare_data_loaders(self.training_data, batch_size=batch_size)
            else:
                QMessageBox.warning(self, "Fehler", "Keine Trainingsdaten geladen!")
                return

        self._cleanup_previous_training()

        # Modell erstellen
        try:
            assert self.train_loader is not None
            model_config = self.config_panel.get_model_config()
            sample = next(iter(self.train_loader))
            input_size = sample[0].shape[-1]

            num_classes = 3
            if self.training_info and 'num_classes' in self.training_info:
                num_classes = self.training_info['num_classes']

            self.model = ModelFactory.create(
                model_config['model_name'],
                input_size=input_size,
                num_classes=num_classes,
                **{k: v for k, v in model_config.items() if k not in ['model_name', 'num_classes']}
            )

            self._log(f"Neues Modell erstellt: {self.model.name} (zufaellige Gewichte)")
            self._log(f"Parameter: {sum(p.numel() for p in self.model.parameters()):,}")

        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Modell-Erstellung fehlgeschlagen:\n{e}")
            return

        # Training-Config
        training_config = self.config_panel.get_training_config()
        config = {
            **training_config,
            'epochs': self.config_panel.epochs_spin.value()
        }

        # UI aktualisieren
        self.viz_panel.reset()
        self.config_panel.set_training_active(True)
        self._start_time = datetime.now()
        self.timer.start(1000)

        # Device aktualisieren
        self.device = self.config_panel.get_device()

        # Synchrones oder asynchrones Training
        if config.get('sync_training', True):
            self._run_sync_training(config)
        else:
            self.worker = TrainingWorker(
                self.model, self.train_loader, self.val_loader, config, self.device
            )
            self.worker.epoch_completed.connect(self._on_epoch_completed)
            self.worker.training_finished.connect(self._on_training_finished)
            self.worker.training_error.connect(self._on_training_error)
            self.worker.log_message.connect(lambda msg: self._log(msg, 'INFO'))
            self.worker.start()

    def _run_sync_training(self, config: dict):
        """Fuehrt Training synchron im Main-Thread aus."""
        from PyQt6.QtWidgets import QApplication

        training_start = datetime.now()
        self._log("=== Synchrones Training gestartet ===")
        self._stop_requested = False

        try:
            assert self.model is not None
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                self.model = self.model.to(self.device)
                self._log(f"Modell auf GPU verschoben: {self.device}")

            # Loss und Optimizer
            class_weights = self.training_data.get('class_weights') if self.training_data else None
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

                # Training
                assert self.train_loader is not None
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

                    if batch_idx % 10 == 0:
                        QApplication.processEvents()

                train_acc = 100.0 * train_correct / train_total
                avg_train_loss = train_loss / len(self.train_loader)

                # Validierung
                assert self.val_loader is not None
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

                scheduler.step(avg_val_loss)

                # UI Update
                self.viz_panel.update_epoch(epoch, avg_train_loss, train_acc/100, avg_val_loss, val_acc/100, epochs)
                self._log(f"Epoch {epoch}: Loss={avg_train_loss:.4f}, Acc={train_acc:.1f}%, Val Acc={val_acc:.1f}%")
                QApplication.processEvents()

                # Early Stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1

                if config.get('early_stopping', True) and patience_counter >= patience:
                    self._log(f"Early Stopping nach {epoch} Epochen")
                    break

            # Modell fuer Speichern vorbereiten (kein Auto-Save mehr!)
            self._prepare_model_for_save(config, best_val_acc, epoch, avg_val_loss, training_start)

            self._on_training_finished({
                'best_accuracy': best_val_acc,
                'final_loss': avg_val_loss,
                'stopped_early': patience_counter >= patience
            })

        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            self._on_training_error(f"GPU Out of Memory: {e}")

        except Exception as e:
            import traceback
            self._on_training_error(f"Training Fehler: {e}\n\n{traceback.format_exc()}")

        finally:
            if self.device.type == 'cuda':
                if self.model is not None:
                    try:
                        self.model.cpu()
                    except Exception:
                        pass
                torch.cuda.empty_cache()
            gc.collect()

    def _prepare_model_for_save(self, config: dict, best_val_acc: float, epoch: int,
                                  avg_val_loss: float, training_start: datetime):
        """Bereitet das Modell fuer das Speichern vor (ohne automatisch zu speichern)."""
        assert self.model is not None
        model_name = self.model.name.lower().replace(' ', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        training_duration = (datetime.now() - training_start).total_seconds()

        # Modell-Parameter direkt vom Modell holen
        model_info = {
            'model_type': model_name,
            'trained_at': timestamp,
            'training_duration_sec': round(training_duration, 1),
            'input_size': getattr(self.model, 'input_size', None),
            'num_classes': getattr(self.model, 'num_classes',
                                   self.training_info.get('num_classes', 3) if self.training_info else 3),
            'epochs_trained': epoch,
            'best_accuracy': round(best_val_acc, 2),
            'final_val_loss': round(avg_val_loss, 4),
            # BiLSTM/LSTM Parameter
            'hidden_sizes': getattr(self.model, 'hidden_sizes', config.get('hidden_sizes')),
            'dropout': getattr(self.model, 'dropout_rate', config.get('dropout')),
            'use_layer_norm': getattr(self.model, 'use_layer_norm', config.get('use_layer_norm', False)),
            'use_attention': getattr(self.model, 'use_attention', config.get('use_attention', False)),
            'use_residual': getattr(self.model, 'use_residual', config.get('use_residual', False)),
            'bidirectional': getattr(self.model, 'bidirectional', True),
            # Transformer-spezifische Parameter
            'd_model': getattr(self.model, 'd_model', config.get('d_model')),
            'nhead': getattr(self.model, 'nhead', config.get('nhead')),
            'num_encoder_layers': getattr(self.model, 'num_encoder_layers', config.get('num_encoder_layers')),
            'dim_feedforward': getattr(self.model, 'dim_feedforward', config.get('dim_feedforward')),
        }

        # Fuer spaeteres Speichern merken
        self._trained_model_info = model_info
        self._unsaved_model = True

        self._log(f"Modell bereit: {model_name} - Acc: {best_val_acc:.1f}%")
        self._log("Klicken Sie 'Session speichern' um das Modell zu sichern!", 'WARNING')

        # ConfigPanel aktualisieren (falls Speichern-Button vorhanden)
        if hasattr(self.config_panel, 'save_btn'):
            self.config_panel.save_btn.setEnabled(True)

    def _on_save_clicked(self):
        """Handler fuer den Speichern-Button - verwendet SaveManager."""
        if not self._unsaved_model or self.model is None:
            QMessageBox.warning(self, "Fehler", "Kein trainiertes Modell vorhanden!")
            return

        try:
            from ...core.config import Config
            from ...core.logger import get_logger

            # Session-Dir ermitteln
            logger = get_logger()
            session_dir = logger.get_session_dir()

            if session_dir is None:
                # Neue Session erstellen
                config = Config()
                session_dir = config.paths.create_new_session()
                self._log(f"Neue Session erstellt: {session_dir.name}", 'INFO')

            # SaveManager initialisieren
            save_manager = SaveManager(session_dir)

            # Metriken zusammenstellen
            metrics = {
                'best_accuracy': self._trained_model_info.get('best_accuracy', 0) if self._trained_model_info else 0,
                'best_loss': self._trained_model_info.get('final_val_loss', 0) if self._trained_model_info else 0,
                'epochs': self._trained_model_info.get('epochs_trained', 0) if self._trained_model_info else 0,
            }

            # Pruefung ob Modell existiert
            check_result = save_manager.check_save_trained(self.model, metrics)

            if not check_result.can_save:
                QMessageBox.warning(
                    self, "Fehler",
                    f"Speichern nicht moeglich:\n{', '.join(check_result.warnings)}"
                )
                return

            # Bei bestehendem Modell: Nachfrage
            if check_result.needs_confirmation:
                action = ask_save_confirmation(check_result, self)

                if action == OverwriteAction.CANCEL:
                    self._log("Speichern abgebrochen", 'INFO')
                    return

                if action == OverwriteAction.NEW_SESSION:
                    # Neue Session erstellen
                    config = Config()
                    session_dir = config.paths.create_new_session()
                    save_manager = SaveManager(session_dir)
                    self._log(f"Neue Session erstellt: {session_dir.name}", 'INFO')

            # Speichern
            self._log("Speichere Modell...")

            result = save_manager.save_trained(
                model=self.model,
                metrics=metrics,
                force=True  # Bestaetigung bereits erfolgt
            )

            if result.can_save:
                self._log(f"Modell gespeichert: {session_dir.name}", 'SUCCESS')
                self._unsaved_model = False

                # Speichern-Button deaktivieren
                if hasattr(self.config_panel, 'save_btn'):
                    self.config_panel.save_btn.setEnabled(False)

                QMessageBox.information(
                    self, "Gespeichert",
                    f"Modell gespeichert in:\n{session_dir.name}"
                )
            else:
                self._log(f"Speichern fehlgeschlagen: {result.warnings}", 'ERROR')
                QMessageBox.warning(self, "Fehler", "Speichern fehlgeschlagen!")

        except Exception as e:
            import traceback
            self._log(f"Speichern fehlgeschlagen: {e}", 'ERROR')
            self._log(traceback.format_exc(), 'DEBUG')
            QMessageBox.critical(self, "Fehler", f"Speichern fehlgeschlagen:\n{e}")

    def _stop_training(self):
        """Stoppt das Training."""
        self._stop_requested = True
        if self.worker:
            self.worker.stop()
        self._log("Training wird gestoppt...")

    def _on_epoch_completed(self, epoch: int, train_loss: float, train_acc: float,
                            val_loss: float, val_acc: float):
        """Callback nach jeder Epoche."""
        total_epochs = self.config_panel.epochs_spin.value()
        self.viz_panel.update_epoch(epoch, train_loss, train_acc/100, val_loss, val_acc/100, total_epochs)
        self._log(f"Epoch {epoch}: Loss={train_loss:.4f}, Acc={train_acc:.1f}%, Val Acc={val_acc:.1f}%")

    def _on_training_finished(self, results: dict):
        """Callback wenn Training beendet."""
        self.timer.stop()
        self.config_panel.set_training_active(False)

        best_acc = results.get('best_accuracy', 0)
        self._log(f"Training abgeschlossen - Beste Accuracy: {best_acc:.1f}%")

        self.training_completed.emit(self.model, results)

    def _on_training_error(self, error_msg: str):
        """Callback bei Training-Fehler."""
        self.timer.stop()
        self.config_panel.set_training_active(False)
        self._log(f"FEHLER: {error_msg}", 'ERROR')
        QMessageBox.critical(self, "Training Fehler", error_msg)

    def _update_time(self):
        """Aktualisiert die Zeitanzeige."""
        if self._start_time:
            elapsed = (datetime.now() - self._start_time).total_seconds()
            self.viz_panel.set_time(int(elapsed))

    def _run_auto_training(self):
        """Fuehrt Auto-Training durch."""
        from PyQt6.QtWidgets import QApplication
        from ...training.auto_trainer import AutoTrainer

        if self.training_data is None:
            QMessageBox.warning(self, "Fehler", "Keine Trainingsdaten geladen!")
            return

        self._cleanup_previous_training()

        auto_config = self.config_panel.get_auto_trainer_config()
        batch_size = auto_config['batch_size']
        use_amp = auto_config['use_amp']
        complexity = auto_config['complexity']

        self._log(f"Erstelle DataLoader mit Batch-Size {batch_size}...")
        self.prepare_data_loaders(self.training_data, batch_size=batch_size)

        assert self.train_loader is not None
        sample = next(iter(self.train_loader))
        input_size = sample[0].shape[-1]
        num_classes = self.training_info.get('num_classes', 3) if self.training_info else 3

        self._log(f"=== Auto-Training gestartet (Komplexitaet {complexity}) ===")
        self.config_panel.set_training_active(True)
        self._start_time = datetime.now()
        self.timer.start(1000)

        try:
            assert self.train_loader is not None and self.val_loader is not None
            auto_trainer = AutoTrainer(
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                input_size=input_size,
                num_classes=num_classes,
                device=self.device,
                use_amp=use_amp
            )

            def progress_callback(current, total, model_name, metrics):
                overall_progress = (current + metrics.get('progress', 0)) / total * 100
                self.viz_panel.progress_bar.setMaximum(100)
                self.viz_panel.progress_bar.setValue(int(overall_progress))
                self.viz_panel.epoch_label.setText(f"Modell: {current+1}/{total} - {model_name}")
                QApplication.processEvents()

                if self._stop_requested:
                    auto_trainer.stop()

            results = auto_trainer.run(
                complexity=complexity,
                progress_callback=progress_callback,
                learning_rate=self.config_panel.lr_spin.value()
            )

            self.auto_trainer_widget.display_results(results)
            self.viz_panel.switch_to_tab(3)  # Auto-Trainer Tab

            self._log(f"Auto-Training abgeschlossen: {len(results)} Modelle getestet")
            if results:
                best = results[0]
                self._log(f"Bestes: {best.model_type} - Val ACC: {best.val_acc:.2%}")

        except Exception as e:
            import traceback
            self._log(f"Auto-Training Fehler: {e}", 'ERROR')
            QMessageBox.critical(self, "Fehler", f"Auto-Training fehlgeschlagen:\n{e}")

        finally:
            self.timer.stop()
            self.config_panel.set_training_active(False)
            self._stop_requested = False

    def _adopt_auto_model(self, result):
        """Uebernimmt ein Modell aus dem Auto-Training."""
        try:
            assert self.train_loader is not None
            sample = next(iter(self.train_loader))
            input_size = sample[0].shape[-1]
            num_classes = self.training_info.get('num_classes', 3) if self.training_info else 3

            self.model = ModelFactory.create(
                result.model_type,
                input_size=input_size,
                num_classes=num_classes,
                **result.config
            )

            if result.model_state:
                self.model.load_state_dict(result.model_state)

            self._log(f"Modell uebernommen: {result.model_type}")
            QMessageBox.information(
                self, "Modell uebernommen",
                f"Modell: {result.model_type}\nVal ACC: {result.val_acc:.2%}"
            )

        except Exception as e:
            self._log(f"Fehler beim Uebernehmen: {e}", 'ERROR')
            QMessageBox.critical(self, "Fehler", str(e))

    def _run_gpu_test(self):
        """Fuehrt einen GPU-Test durch."""
        if not torch.cuda.is_available():
            QMessageBox.warning(self, "GPU Test", "Keine GPU verfuegbar!")
            return

        self._log("=== GPU Test gestartet ===")
        if self.config_panel.gpu_test_btn:
            self.config_panel.gpu_test_btn.setEnabled(False)
            self.config_panel.gpu_test_btn.setText("Test laeuft...")

        try:
            torch.cuda.empty_cache()

            # Synthetische Daten
            X = np.random.randn(1000, 50, 4).astype(np.float32)
            Y = np.random.randint(0, 3, size=1000)

            X_tensor = torch.FloatTensor(X)
            Y_tensor = torch.LongTensor(Y)

            from torch.utils.data import TensorDataset, DataLoader
            dataset = TensorDataset(X_tensor, Y_tensor)
            loader = DataLoader(dataset, batch_size=32, shuffle=False)

            model = ModelFactory.create('bilstm', input_size=4, hidden_sizes=[32], num_classes=3)
            model = model.to(torch.device('cuda'))

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            for epoch in range(5):
                for data, target in loader:
                    data = data.cuda()
                    target = target.cuda()
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                self._log(f"GPU Test Epoch {epoch+1}/5")

            model.cpu()
            del model, optimizer, criterion
            torch.cuda.empty_cache()
            gc.collect()

            self._log("=== GPU Test ERFOLGREICH ===")
            QMessageBox.information(self, "GPU Test", "GPU Test erfolgreich!")

        except Exception as e:
            self._log(f"GPU Test Fehler: {e}", 'ERROR')
            QMessageBox.critical(self, "GPU Test Fehler", str(e))

        finally:
            if self.config_panel.gpu_test_btn:
                self.config_panel.gpu_test_btn.setEnabled(True)
                self.config_panel.gpu_test_btn.setText("GPU Test")
            torch.cuda.empty_cache()

    def closeEvent(self, event):
        """Cleanup beim Schliessen."""
        self._stop_requested = True
        self.timer.stop()
        if self.worker:
            self.worker.stop()
            self.worker.wait(5000)
        self._cleanup_previous_training()
        event.accept()
