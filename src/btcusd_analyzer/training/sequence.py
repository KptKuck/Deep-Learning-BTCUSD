"""
Sequenz-Generator Modul - Erstellt Sequenzen fuer LSTM/Transformer Training
Entspricht prepare_training_data.m aus dem MATLAB-Projekt
"""

from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from ..core.logger import get_logger
from .normalizer import ZScoreNormalizer


def expand_labels_lookahead(
    labels: np.ndarray,
    lookahead: int = 5,
    hold_label: int = 0,
    buy_label: int = 1,
    sell_label: int = 2
) -> np.ndarray:
    """
    Erweitert Labels mit Lookahead - Bars vor einem Peak erhalten das Peak-Label.

    Das Modell lernt so, einen Peak vorherzusagen BEVOR er eintritt.

    Beispiel mit lookahead=3:
        Original:  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0]
        Erweitert: [0, 0, 1, 1, 1, 1, 0, 0, 2, 2, 2, 0, 0]
                          ^--lookahead--^     ^--lookahead--^

    Args:
        labels: Original-Label-Array
        lookahead: Anzahl Bars vor dem Peak die das Label erhalten
        hold_label: Label-Wert fuer HOLD (wird nicht erweitert)
        buy_label: Label-Wert fuer BUY
        sell_label: Label-Wert fuer SELL

    Returns:
        Erweitertes Label-Array
    """
    expanded = labels.copy()
    n = len(labels)

    # BUY-Peaks erweitern
    buy_indices = np.where(labels == buy_label)[0]
    for idx in buy_indices:
        start = max(0, idx - lookahead)
        expanded[start:idx] = buy_label

    # SELL-Peaks erweitern
    sell_indices = np.where(labels == sell_label)[0]
    for idx in sell_indices:
        start = max(0, idx - lookahead)
        expanded[start:idx] = sell_label

    return expanded


def compute_class_weights(
    labels: np.ndarray,
    num_classes: int = 3,
    ignore_label: int = -1
) -> torch.Tensor:
    """
    Berechnet Class Weights fuer unbalancierte Daten.

    Verwendung: criterion = nn.CrossEntropyLoss(weight=class_weights)

    Args:
        labels: Label-Array
        num_classes: Anzahl Klassen
        ignore_label: Label das ignoriert werden soll

    Returns:
        Tensor mit Gewichten pro Klasse
    """
    # Nur gueltige Labels zaehlen
    valid_labels = labels[labels != ignore_label]

    if len(valid_labels) == 0:
        return torch.ones(num_classes)

    # Haeufigkeit pro Klasse
    counts = np.zeros(num_classes)
    for i in range(num_classes):
        counts[i] = np.sum(valid_labels == i)

    # Inverse Frequency als Gewicht
    # Klassen mit weniger Samples bekommen hoeheres Gewicht
    total = counts.sum()
    weights = total / (num_classes * counts + 1e-6)

    # Normalisieren damit Durchschnitt = 1
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


class SequenceGenerator:
    """
    Generiert Sequenzen aus Zeitreihendaten fuer Neural Network Training.

    Die Sequenzen haben folgende Struktur:
    - Input: [batch, seq_length, features]
    - Output: [batch, num_classes] (One-Hot) oder [batch] (Label-Index)

    Attributes:
        lookback: Anzahl historischer Zeitschritte
        lookforward: Anzahl zukuenftiger Zeitschritte (fuer Labeling)
        normalize: Ob Normalisierung angewendet werden soll
    """

    def __init__(
        self,
        lookback: int = 50,
        lookforward: int = 100,
        normalize: bool = True
    ):
        """
        Initialisiert den Sequenz-Generator.

        Args:
            lookback: Anzahl Zeitschritte fuer Input-Sequenz
            lookforward: Anzahl Zeitschritte fuer Label-Berechnung
            normalize: Z-Score Normalisierung pro Sequenz
        """
        self.lookback = lookback
        self.lookforward = lookforward
        self.normalize = normalize
        self.normalizer = ZScoreNormalizer() if normalize else None
        self.logger = get_logger()

    @property
    def sequence_length(self) -> int:
        """Gesamte Sequenzlaenge."""
        return self.lookback + self.lookforward

    def generate(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        ignore_label: int = -1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generiert Sequenzen aus Feature-Matrix und Labels.

        Args:
            features: Feature-Matrix [n_samples, n_features]
            labels: Label-Array [n_samples]
            ignore_label: Label-Wert der ignoriert werden soll (default: -1)

        Returns:
            Tuple aus (X, y) Arrays
            X: [n_sequences, lookback, n_features]
            y: [n_sequences]
        """
        n_samples, n_features = features.shape
        n_sequences = n_samples - self.sequence_length + 1

        if n_sequences <= 0:
            self.logger.error(f'Nicht genug Daten: {n_samples} Samples, '
                             f'benötigt mindestens {self.sequence_length}')
            return np.array([]), np.array([])

        self.logger.debug(f'Generiere {n_sequences} Sequenzen '
                         f'(Lookback: {self.lookback}, Lookforward: {self.lookforward})')

        # Temporaere Listen fuer Filterung
        X_list = []
        y_list = []

        for i in range(n_sequences):
            # Label-Index berechnen
            label_idx = i + self.lookback - 1
            if label_idx >= len(labels):
                label_idx = len(labels) - 1

            current_label = labels[label_idx]

            # Samples mit ignore_label ueberspringen
            if current_label == ignore_label:
                continue

            # Input-Sequenz (lookback Zeitschritte)
            seq = features[i:i + self.lookback].copy()

            # Normalisierung pro Sequenz (Z-Score)
            if self.normalize and self.normalizer:
                seq = self.normalizer.fit_transform(seq)

            X_list.append(seq)
            y_list.append(current_label)

        if len(X_list) == 0:
            self.logger.warning('Keine gueltigen Sequenzen generiert (alle Labels ignoriert)')
            return np.array([]), np.array([])

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)

        self.logger.success(f'{len(X)} Sequenzen generiert '
                           f'({n_sequences - len(X)} ignoriert)')
        return X, y

    def generate_single(
        self,
        features: np.ndarray,
        start_idx: int
    ) -> Optional[np.ndarray]:
        """
        Generiert eine einzelne Sequenz fuer Inferenz.

        Args:
            features: Feature-Matrix [n_samples, n_features]
            start_idx: Start-Index der Sequenz

        Returns:
            Sequenz-Array [lookback, n_features] oder None
        """
        if start_idx < 0 or start_idx + self.lookback > len(features):
            return None

        seq = features[start_idx:start_idx + self.lookback].copy()

        if self.normalize and self.normalizer:
            seq = self.normalizer.fit_transform(seq)

        return seq.astype(np.float32)

    def create_dataset(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        validation_split: float = 0.2,
        shuffle: bool = True
    ) -> Tuple['SequenceDataset', 'SequenceDataset']:
        """
        Erstellt PyTorch Datasets fuer Training und Validierung.

        Args:
            features: Feature-Matrix
            labels: Label-Array
            validation_split: Anteil der Validierungsdaten
            shuffle: Ob Daten gemischt werden sollen

        Returns:
            Tuple aus (train_dataset, val_dataset)
        """
        X, y = self.generate(features, labels)

        n_samples = len(X)
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val

        if shuffle:
            indices = np.random.permutation(n_samples)
            X = X[indices]
            y = y[indices]

        train_dataset = SequenceDataset(X[:n_train], y[:n_train])
        val_dataset = SequenceDataset(X[n_train:], y[n_train:])

        self.logger.debug(f'Train: {n_train} Samples, Val: {n_val} Samples')
        return train_dataset, val_dataset

    def create_dataloaders(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 32,
        validation_split: float = 0.2,
        num_workers: int = 0
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Erstellt PyTorch DataLoader fuer Training und Validierung.

        Args:
            features: Feature-Matrix
            labels: Label-Array
            batch_size: Batch-Groesse
            validation_split: Anteil der Validierungsdaten
            num_workers: Anzahl Worker-Prozesse

        Returns:
            Tuple aus (train_loader, val_loader)
        """
        train_dataset, val_dataset = self.create_dataset(
            features, labels, validation_split, shuffle=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

        return train_loader, val_loader


class SequenceDataset(Dataset):
    """
    PyTorch Dataset fuer Sequenzdaten.

    Attributes:
        X: Feature-Sequenzen [n_samples, seq_length, n_features]
        y: Labels [n_samples]
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialisiert das Dataset.

        Args:
            X: Feature-Sequenzen
            y: Labels
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        """Anzahl der Samples."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gibt ein Sample zurueck.

        Args:
            idx: Index des Samples

        Returns:
            Tuple aus (features, label)
        """
        return self.X[idx], self.y[idx]

    @property
    def input_shape(self) -> Tuple[int, int]:
        """Shape eines einzelnen Inputs (seq_length, n_features)."""
        return tuple(self.X.shape[1:])

    @property
    def n_features(self) -> int:
        """Anzahl Features."""
        return self.X.shape[2]

    @property
    def seq_length(self) -> int:
        """Sequenzlaenge."""
        return self.X.shape[1]
