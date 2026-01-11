"""
Basis-Klasse fuer alle neuronalen Netzwerk-Modelle.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Abstrakte Basisklasse fuer alle Modelle."""

    def __init__(self, name: str = 'BaseModel'):
        """
        Initialisiert das Basismodell.

        Args:
            name: Name des Modells
        """
        super().__init__()
        self._name = name

    @property
    def name(self) -> str:
        """Gibt den Modellnamen zurueck."""
        return self._name

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Vorwaertsdurchlauf des Modells.

        Args:
            x: Eingabetensor der Form (batch_size, sequence_length, input_size)

        Returns:
            Ausgabetensor der Form (batch_size, num_classes)
        """
        pass

    def count_parameters(self) -> int:
        """Zaehlt die trainierbaren Parameter."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_device(self) -> torch.device:
        """Gibt das Geraet des Modells zurueck."""
        return next(self.parameters()).device

    def save(self, filepath: str, optimizer: Optional[torch.optim.Optimizer] = None,
             epoch: Optional[int] = None, **kwargs) -> None:
        """
        Speichert das Modell.

        Args:
            filepath: Pfad zur Speicherdatei
            optimizer: Optional - Optimizer zum Mitspeichern
            epoch: Optional - Aktuelle Epoche
            **kwargs: Zusaetzliche Daten zum Speichern
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_name': self._name,
            **kwargs
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if epoch is not None:
            checkpoint['epoch'] = epoch

        torch.save(checkpoint, filepath)

    def load(self, filepath: str, optimizer: Optional[torch.optim.Optimizer] = None,
             strict: bool = True) -> dict:
        """
        Laedt das Modell.

        Args:
            filepath: Pfad zur Speicherdatei
            optimizer: Optional - Optimizer zum Laden
            strict: Strikte Uebereinstimmung der Gewichte

        Returns:
            Checkpoint-Dictionary mit zusaetzlichen Daten
        """
        checkpoint = torch.load(filepath, map_location=self.get_device())

        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint

    def __repr__(self) -> str:
        return f'{self._name}(parameters={self.count_parameters():,})'
