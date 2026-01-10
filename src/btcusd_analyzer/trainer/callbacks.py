"""
Callbacks Modul - Training Callbacks fuer Early Stopping, Checkpointing, etc.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class Callback(ABC):
    """
    Abstrakte Basisklasse fuer Training Callbacks.

    Callbacks werden an bestimmten Punkten waehrend des Trainings aufgerufen:
    - on_train_begin/end: Anfang/Ende des gesamten Trainings
    - on_epoch_begin/end: Anfang/Ende jeder Epoche
    - on_batch_begin/end: Anfang/Ende jedes Batches
    """

    def __init__(self):
        self.model = None
        self.trainer = None

    def set_model(self, model):
        """Setzt das Modell."""
        self.model = model

    def set_trainer(self, trainer):
        """Setzt den Trainer."""
        self.trainer = trainer

    def on_train_begin(self):
        """Wird am Anfang des Trainings aufgerufen."""
        pass

    def on_train_end(self):
        """Wird am Ende des Trainings aufgerufen."""
        pass

    def on_epoch_begin(self, epoch: int):
        """Wird am Anfang jeder Epoche aufgerufen."""
        pass

    def on_epoch_end(self, epoch: int, logs: Dict[str, float]):
        """Wird am Ende jeder Epoche aufgerufen."""
        pass

    def on_batch_begin(self, batch: int):
        """Wird am Anfang jedes Batches aufgerufen."""
        pass

    def on_batch_end(self, batch: int, logs: Dict[str, float]):
        """Wird am Ende jedes Batches aufgerufen."""
        pass


class CallbackList:
    """Container fuer mehrere Callbacks."""

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []

    def append(self, callback: Callback):
        """Fuegt einen Callback hinzu."""
        self.callbacks.append(callback)

    def set_model(self, model):
        """Setzt das Modell fuer alle Callbacks."""
        for callback in self.callbacks:
            callback.set_model(model)

    def set_trainer(self, trainer):
        """Setzt den Trainer fuer alle Callbacks."""
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end()

    def on_epoch_begin(self, epoch: int):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch: int, logs: Dict[str, float]):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch: int):
        for callback in self.callbacks:
            callback.on_batch_begin(batch)

    def on_batch_end(self, batch: int, logs: Dict[str, float]):
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)


class EarlyStopping(Callback):
    """
    Stoppt das Training wenn keine Verbesserung mehr stattfindet.

    Ueberwacht eine Metrik und stoppt wenn sich diese nicht verbessert.

    Attributes:
        monitor: Zu ueberwachende Metrik ('val_loss' oder 'val_accuracy')
        patience: Anzahl Epochen ohne Verbesserung bis zum Stopp
        min_delta: Minimale Aenderung die als Verbesserung zaehlt
        mode: 'min' fuer Loss, 'max' fuer Accuracy
    """

    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'auto',
        restore_best_weights: bool = True
    ):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        # Mode bestimmen
        if mode == 'auto':
            self.mode = 'min' if 'loss' in monitor else 'max'
        else:
            self.mode = mode

        self.best_value: Optional[float] = None
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0

    def on_train_begin(self):
        self.wait = 0
        self.stopped_epoch = 0
        self.best_value = None
        self.best_weights = None

    def on_epoch_end(self, epoch: int, logs: Dict[str, float]):
        current = logs.get(self.monitor)
        if current is None:
            return

        if self.best_value is None:
            self.best_value = current
            self._save_best_weights()
        elif self._is_improvement(current):
            self.best_value = current
            self.wait = 0
            self._save_best_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.trainer.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    self.model.load_state_dict(self.best_weights)

    def _is_improvement(self, current: float) -> bool:
        """Prueft ob der aktuelle Wert eine Verbesserung ist."""
        if self.mode == 'min':
            return current < self.best_value - self.min_delta
        else:
            return current > self.best_value + self.min_delta

    def _save_best_weights(self):
        """Speichert die besten Gewichte."""
        if self.restore_best_weights:
            self.best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}


class ModelCheckpoint(Callback):
    """
    Speichert das Modell zu bestimmten Zeitpunkten.

    Kann so konfiguriert werden, dass nur das beste Modell
    oder jede Epoche gespeichert wird.

    Attributes:
        filepath: Pfad-Template fuer Checkpoints (kann {epoch}, {val_accuracy} enthalten)
        monitor: Zu ueberwachende Metrik
        save_best_only: Nur bestes Modell speichern
        mode: 'min' oder 'max'
    """

    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_accuracy',
        save_best_only: bool = True,
        mode: str = 'auto'
    ):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only

        # Mode bestimmen
        if mode == 'auto':
            self.mode = 'min' if 'loss' in monitor else 'max'
        else:
            self.mode = mode

        self.best_value: Optional[float] = None

    def on_train_begin(self):
        self.best_value = None

    def on_epoch_end(self, epoch: int, logs: Dict[str, float]):
        current = logs.get(self.monitor)
        if current is None:
            return

        # Dateipfad formatieren
        filepath = self.filepath.format(
            epoch=epoch,
            **{k: f'{v:.2f}' for k, v in logs.items()}
        )

        if self.save_best_only:
            if self.best_value is None or self._is_improvement(current):
                self.best_value = current
                self.trainer.save_checkpoint(Path(filepath), logs)
        else:
            self.trainer.save_checkpoint(Path(filepath), logs)

    def _is_improvement(self, current: float) -> bool:
        """Prueft ob der aktuelle Wert eine Verbesserung ist."""
        if self.mode == 'min':
            return current < self.best_value
        else:
            return current > self.best_value


class LearningRateLogger(Callback):
    """Loggt die Learning Rate nach jeder Epoche."""

    def __init__(self):
        super().__init__()
        self.learning_rates = []

    def on_epoch_end(self, epoch: int, logs: Dict[str, float]):
        if self.trainer and self.trainer.optimizer:
            lr = self.trainer.optimizer.param_groups[0]['lr']
            self.learning_rates.append(lr)


class ProgressCallback(Callback):
    """
    Callback fuer Fortschritts-Updates (z.B. fuer GUI).

    Ruft eine Callback-Funktion mit Fortschrittsinformationen auf.
    """

    def __init__(self, callback_fn):
        """
        Args:
            callback_fn: Funktion die mit (epoch, total_epochs, metrics) aufgerufen wird
        """
        super().__init__()
        self.callback_fn = callback_fn
        self.total_epochs = 0

    def on_train_begin(self):
        # Total epochs wird vom Trainer gesetzt
        pass

    def on_epoch_end(self, epoch: int, logs: Dict[str, float]):
        if self.callback_fn:
            self.callback_fn(epoch, self.total_epochs, logs)


class GradientMonitor(Callback):
    """
    Ueberwacht Gradienten auf Exploding/Vanishing Gradients.

    Warnt wenn Gradienten zu gross oder zu klein werden.
    """

    def __init__(self, log_frequency: int = 10, threshold_high: float = 100.0, threshold_low: float = 1e-7):
        super().__init__()
        self.log_frequency = log_frequency
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self.gradient_norms = []

    def on_batch_end(self, batch: int, logs: Dict[str, float]):
        if batch % self.log_frequency != 0:
            return

        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        self.gradient_norms.append(total_norm)

        # Warnungen
        if total_norm > self.threshold_high:
            print(f'[WARNING] Exploding Gradients: {total_norm:.4f}')
        elif total_norm < self.threshold_low:
            print(f'[WARNING] Vanishing Gradients: {total_norm:.4f}')
