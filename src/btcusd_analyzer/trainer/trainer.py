"""
Trainer Modul - Training Loop fuer Neural Networks
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..core.logger import get_logger
from ..models.base import BaseModel
from .callbacks import Callback, CallbackList, EarlyStopping, ModelCheckpoint


@dataclass
class TrainingMetrics:
    """Speichert Metriken waehrend des Trainings."""
    epoch: int = 0
    train_loss: float = 0.0
    train_accuracy: float = 0.0
    val_loss: float = 0.0
    val_accuracy: float = 0.0
    learning_rate: float = 0.0
    duration_ms: float = 0.0


@dataclass
class TrainingHistory:
    """Speichert die komplette Trainingshistorie."""
    train_loss: List[float] = field(default_factory=list)
    train_accuracy: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    epochs: List[int] = field(default_factory=list)

    def add(self, metrics: TrainingMetrics):
        """Fuegt Metriken hinzu."""
        self.epochs.append(metrics.epoch)
        self.train_loss.append(metrics.train_loss)
        self.train_accuracy.append(metrics.train_accuracy)
        self.val_loss.append(metrics.val_loss)
        self.val_accuracy.append(metrics.val_accuracy)
        self.learning_rates.append(metrics.learning_rate)

    def to_dict(self) -> Dict:
        """Konvertiert zu Dictionary."""
        return {
            'epochs': self.epochs,
            'train_loss': self.train_loss,
            'train_accuracy': self.train_accuracy,
            'val_loss': self.val_loss,
            'val_accuracy': self.val_accuracy,
            'learning_rates': self.learning_rates
        }

    @property
    def best_val_accuracy(self) -> float:
        """Beste Validierungs-Accuracy."""
        return max(self.val_accuracy) if self.val_accuracy else 0.0

    @property
    def best_epoch(self) -> int:
        """Epoche mit bester Validierungs-Accuracy."""
        if not self.val_accuracy:
            return 0
        return self.epochs[np.argmax(self.val_accuracy)]


class Trainer:
    """
    Training Loop fuer Neural Network Modelle.

    Unterstuetzt:
    - GPU/CPU Training
    - Callbacks (Early Stopping, Model Checkpointing)
    - Learning Rate Scheduling
    - Klassengewichtung fuer unbalancierte Daten
    - Progress Callbacks fuer GUI-Integration

    Attributes:
        model: Das zu trainierende Modell
        device: Training Device (CPU/GPU)
        history: Trainingshistorie
    """

    def __init__(
        self,
        model: BaseModel,
        device: Optional[torch.device] = None,
        callbacks: Optional[List[Callback]] = None
    ):
        """
        Initialisiert den Trainer.

        Args:
            model: Zu trainierendes Modell
            device: Training Device (default: auto)
            callbacks: Liste von Callbacks
        """
        self.logger = get_logger()

        # Device bestimmen
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        # Modell auf Device verschieben
        self.model = model.to(device)
        self.logger.info(f'Training auf {device}')

        # Callbacks
        self.callbacks = CallbackList(callbacks or [])

        # Training State
        self.history = TrainingHistory()
        self.current_epoch = 0
        self.stop_training = False

        # Optimizer und Scheduler (werden in train() gesetzt)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.criterion: Optional[nn.Module] = None

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        learning_rate: float = 0.001,
        class_weights: Optional[np.ndarray] = None,
        progress_callback: Optional[Callable[[TrainingMetrics], None]] = None
    ) -> TrainingHistory:
        """
        Fuehrt das Training durch.

        Args:
            train_loader: DataLoader fuer Trainingsdaten
            val_loader: DataLoader fuer Validierungsdaten
            epochs: Anzahl Epochen
            learning_rate: Lernrate
            class_weights: Optionale Klassengewichte
            progress_callback: Callback fuer Fortschritts-Updates

        Returns:
            TrainingHistory mit allen Metriken
        """
        self.logger.info(f'Starte Training: {epochs} Epochen')

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )

        # Learning Rate Scheduler (verbose wurde in PyTorch 2.x entfernt)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        # Loss Function mit optionalen Klassengewichten
        if class_weights is not None:
            weights = torch.FloatTensor(class_weights).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Callbacks initialisieren
        self.callbacks.set_model(self.model)
        self.callbacks.set_trainer(self)
        self.callbacks.on_train_begin()

        self.stop_training = False

        try:
            for epoch in range(1, epochs + 1):
                if self.stop_training:
                    self.logger.warning('Training gestoppt')
                    break

                self.current_epoch = epoch
                self.callbacks.on_epoch_begin(epoch)

                start_time = time.perf_counter()

                # Training
                train_loss, train_acc = self._train_epoch(train_loader)

                # Validierung
                val_loss, val_acc = self._validate_epoch(val_loader)

                # Scheduler Step
                self.scheduler.step(val_loss)

                duration_ms = (time.perf_counter() - start_time) * 1000

                # Metriken erstellen
                metrics = TrainingMetrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    train_accuracy=train_acc,
                    val_loss=val_loss,
                    val_accuracy=val_acc,
                    learning_rate=self.optimizer.param_groups[0]['lr'],
                    duration_ms=duration_ms
                )

                self.history.add(metrics)

                # Logging
                self.logger.info(
                    f'Epoch {epoch}/{epochs} | '
                    f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
                    f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | '
                    f'{duration_ms:.0f}ms'
                )

                # Progress Callback
                if progress_callback:
                    progress_callback(metrics)

                # Callbacks
                self.callbacks.on_epoch_end(epoch, {
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc
                })

        finally:
            self.callbacks.on_train_end()

        self.logger.success(f'Training beendet nach {self.current_epoch} Epochen')
        self.logger.info(f'Beste Val Accuracy: {self.history.best_val_accuracy:.2f}% '
                        f'(Epoch {self.history.best_epoch})')

        return self.history

    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Trainiert eine Epoche."""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.criterion(output, target)

            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validiert eine Epoche."""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Evaluiert das Modell auf Testdaten.

        Args:
            test_loader: DataLoader fuer Testdaten

        Returns:
            Dictionary mit Evaluations-Metriken
        """
        self.model.eval()

        all_preds = []
        all_targets = []
        total_loss = 0.0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                _, predicted = output.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        accuracy = 100.0 * np.mean(all_preds == all_targets)
        avg_loss = total_loss / len(test_loader)

        # Per-Class Accuracy
        class_acc = {}
        for cls in np.unique(all_targets):
            mask = all_targets == cls
            class_acc[cls] = 100.0 * np.mean(all_preds[mask] == all_targets[mask])

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'class_accuracy': class_acc,
            'predictions': all_preds,
            'targets': all_targets
        }

    def stop(self):
        """Stoppt das Training."""
        self.stop_training = True
        self.logger.warning('Training-Stopp angefordert')

    def save_checkpoint(self, filepath: Path, metrics: Optional[Dict] = None):
        """Speichert einen Checkpoint."""
        self.model.save(
            filepath,
            optimizer=self.optimizer,
            epoch=self.current_epoch,
            metrics=metrics
        )

    def load_checkpoint(self, filepath: Path):
        """Laedt einen Checkpoint."""
        model, checkpoint = self.model.load(filepath, self.device)
        self.model = model

        if 'optimizer_state_dict' in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.current_epoch = checkpoint.get('epoch', 0)
        self.logger.info(f'Checkpoint geladen: Epoch {self.current_epoch}')
