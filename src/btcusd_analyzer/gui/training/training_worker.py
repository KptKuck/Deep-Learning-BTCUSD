"""
Training Worker - QThread fuer Training ohne GUI-Blockierung.
"""

from pathlib import Path
from typing import Dict, Any
import gc

from PyQt6.QtCore import QThread, pyqtSignal
import torch


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
            from ...trainer.trainer import Trainer
            from ...trainer.callbacks import EarlyStopping, ModelCheckpoint

            # WICHTIG: CUDA-Kontext im Worker-Thread initialisieren
            if self.device.type == 'cuda':
                torch.cuda.set_device(0)
                torch.cuda.empty_cache()
                self.model = self.model.to(self.device)
                self.log_message.emit("Modell auf GPU verschoben (im Worker-Thread)")

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

            # Finales Modell speichern
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

            del trainer
            del history

        except torch.cuda.OutOfMemoryError as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            error_msg = f"GPU Out of Memory: {e}\n\nVersuche kleinere Batch Size oder weniger Features."
            self.training_error.emit(error_msg)

        except RuntimeError as e:
            error_str = str(e)
            if 'CUDA' in error_str or 'cuda' in error_str:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                error_msg = f"CUDA Fehler: {error_str}\n\nMoegliche Loesungen:\n- Batch Size reduzieren\n- GPU-Treiber aktualisieren\n- CPU statt GPU verwenden"
            else:
                error_msg = f"Runtime Fehler: {error_str}"
            self.training_error.emit(error_msg)

        except Exception as e:
            import traceback
            import sys

            exc_type, exc_value, exc_tb = sys.exc_info()
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)

            error_msg = f"Training Fehler: {e}\n\nTyp: {exc_type.__name__}\n\nTraceback:\n{''.join(tb_lines)}"
            self.log_message.emit(f"KRITISCHER FEHLER: {error_msg}")
            self.training_error.emit(error_msg)

        finally:
            self._cleanup()

    def _cleanup(self):
        """Speicher aufraeumen nach Training."""
        try:
            if self.model is not None:
                try:
                    self.model.cpu()
                except Exception:
                    pass

            if torch.cuda.is_available():
                for _ in range(3):
                    torch.cuda.empty_cache()
                torch.cuda.synchronize()

                allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
                self.log_message.emit(f"Worker Cleanup: GPU {allocated:.2f}GB alloc / {reserved:.2f}GB reserved")

            gc.collect()

        except Exception as cleanup_err:
            self.log_message.emit(f"Cleanup Warnung: {cleanup_err}")

    def stop(self):
        """Stoppt das Training."""
        self._stop_requested = True
