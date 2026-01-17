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
from .config_panel import ConfigPanel
from .visualization_panel import VisualizationPanel
from .auto_trainer_widget import AutoTrainerWidget
from .training_worker import TrainingWorker
from ...models.factory import ModelFactory


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

        # GPU Test Button (falls vorhanden)
        if self.config_panel.gpu_test_btn:
            self.config_panel.gpu_test_btn.clicked.connect(self._run_gpu_test)

        # Auto-Trainer Widget
        self.auto_trainer_widget.adopt_model.connect(self._adopt_auto_model)
        self.auto_trainer_widget.log_message.connect(self._log)

    def _log(self, message: str, level: str = 'INFO'):
        """Loggt eine Nachricht."""
        caller_frame = inspect.currentframe().f_back
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

            # Modell speichern
            if config.get('save_best', True):
                self._save_model(config, best_val_acc, epoch, avg_val_loss, training_start)

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

    def _save_model(self, config: dict, best_val_acc: float, epoch: int, avg_val_loss: float, training_start: datetime):
        """Speichert das trainierte Modell."""
        self._log("=== MODEL SAVE START ===", 'DEBUG')

        save_path = Path(config.get('save_path', 'models'))
        save_path.mkdir(parents=True, exist_ok=True)
        self._log(f"Save-Path: {save_path}", 'DEBUG')

        model_name = self.model.name.lower().replace(' ', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        final_path = save_path / f'{model_name}_{timestamp}_acc{best_val_acc:.1f}.pt'
        self._log(f"Model-Name: {model_name}", 'DEBUG')
        self._log(f"Final-Path: {final_path}", 'DEBUG')

        training_duration = (datetime.now() - training_start).total_seconds()

        # Debug: Modell-Attribute auslesen
        self._log("--- Lese Modell-Attribute ---", 'DEBUG')
        self._log(f"model.name: {self.model.name}", 'DEBUG')
        self._log(f"model.input_size: {getattr(self.model, 'input_size', 'NICHT GEFUNDEN')}", 'DEBUG')
        self._log(f"model.hidden_sizes: {getattr(self.model, 'hidden_sizes', 'NICHT GEFUNDEN')}", 'DEBUG')
        self._log(f"model.num_classes: {getattr(self.model, 'num_classes', 'NICHT GEFUNDEN')}", 'DEBUG')
        self._log(f"model.dropout_rate: {getattr(self.model, 'dropout_rate', 'NICHT GEFUNDEN')}", 'DEBUG')
        self._log(f"model.bidirectional: {getattr(self.model, 'bidirectional', 'NICHT GEFUNDEN')}", 'DEBUG')
        self._log(f"model.use_layer_norm: {getattr(self.model, 'use_layer_norm', 'NICHT GEFUNDEN')}", 'DEBUG')
        self._log(f"model.use_attention: {getattr(self.model, 'use_attention', 'NICHT GEFUNDEN')}", 'DEBUG')

        # Modell-Parameter direkt vom Modell holen (zuverlaessiger als config)
        model_info = {
            'model_type': model_name,
            'trained_at': timestamp,
            'training_duration_sec': round(training_duration, 1),
            'input_size': getattr(self.model, 'input_size', None),
            'num_classes': getattr(self.model, 'num_classes', self.training_info.get('num_classes', 3) if self.training_info else 3),
            'epochs_trained': epoch,
            'best_accuracy': round(best_val_acc, 2),
            'final_val_loss': round(avg_val_loss, 4),
            # BiLSTM/LSTM Parameter direkt vom Modell
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

        self._log("--- Model-Info Dict ---", 'DEBUG')
        for key, value in model_info.items():
            self._log(f"  {key}: {value}", 'DEBUG')

        # State-Dict Keys analysieren
        state_dict = self.model.state_dict()
        self._log(f"State-Dict Keys ({len(state_dict)}): {list(state_dict.keys())[:10]}...", 'DEBUG')

        self.model.save(final_path, metrics={'best_accuracy': best_val_acc}, model_info=model_info)
        pt_size = final_path.stat().st_size / (1024 * 1024)
        self._log(f"Modell gespeichert: {final_path} ({pt_size:.2f} MB)")

        # JSON speichern
        json_path = final_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        json_size = json_path.stat().st_size / 1024
        self._log(f"JSON gespeichert: {json_path} ({json_size:.1f} KB)", 'DEBUG')

        # Modell in Session-Ordner kopieren
        self._log("--- Kopiere in Session ---", 'DEBUG')
        try:
            from ...core.logger import get_logger
            from ...core.session_manager import SessionManager

            session_dir = get_logger().get_session_dir()
            self._log(f"Session-Dir: {session_dir}", 'DEBUG')

            if session_dir:
                manager = SessionManager(session_dir)
                dest_path = manager.save_model(final_path)
                self._log(f"In Session kopiert: {dest_path}", 'DEBUG')

                # Status auf 'trained' setzen
                self._log("--- Setze Status auf 'trained' ---", 'DEBUG')
                manager.set_status('trained')

                # Session-DB aktualisieren mit Modell-Infos
                self._log("--- Aktualisiere SessionDB ---", 'DEBUG')
                update_data = {
                    'status': 'trained',
                    'has_model': True,
                    'model_accuracy': model_info.get('best_accuracy', 0),
                    'model_version': 'bilstm_v1',
                    'model_type': model_info.get('model_type', 'bilstm'),
                }
                # Features und Samples aus training_info hinzufuegen (falls vorhanden)
                if self.training_info:
                    features = self.training_info.get('features', [])
                    if features:
                        update_data['features'] = features
                        update_data['num_features'] = len(features)
                    num_samples = self.training_info.get('actual_samples', 0)
                    if num_samples:
                        update_data['num_samples'] = num_samples
                manager._update_session_db(update_data)
            else:
                self._log("Keine Session-Dir verfuegbar!", 'WARNING')
        except Exception as e:
            import traceback
            self._log(f"Modell konnte nicht in Session kopiert werden: {e}", 'WARNING')
            self._log(traceback.format_exc(), 'DEBUG')

        self._log("=== MODEL SAVE DONE ===", 'DEBUG')

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

        sample = next(iter(self.train_loader))
        input_size = sample[0].shape[-1]
        num_classes = self.training_info.get('num_classes', 3) if self.training_info else 3

        self._log(f"=== Auto-Training gestartet (Komplexitaet {complexity}) ===")
        self.config_panel.set_training_active(True)
        self._start_time = datetime.now()
        self.timer.start(1000)

        try:
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
