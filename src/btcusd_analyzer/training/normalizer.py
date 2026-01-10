"""
Normalizer Modul - Z-Score Normalisierung fuer Sequenzdaten
"""

from typing import Optional, Tuple

import numpy as np


class ZScoreNormalizer:
    """
    Z-Score Normalisierung pro Sequenz oder global.

    Die Normalisierung transformiert Daten zu:
    - Mittelwert = 0
    - Standardabweichung = 1

    Formel: z = (x - mean) / std

    Attributes:
        mean: Gespeicherter Mittelwert (fuer inverse Transformation)
        std: Gespeicherte Standardabweichung
        epsilon: Kleine Konstante zur Vermeidung von Division durch 0
    """

    def __init__(self, epsilon: float = 1e-8):
        """
        Initialisiert den Normalizer.

        Args:
            epsilon: Minimaler Wert fuer Standardabweichung
        """
        self.epsilon = epsilon
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, data: np.ndarray) -> 'ZScoreNormalizer':
        """
        Berechnet Mittelwert und Standardabweichung.

        Args:
            data: Array mit Shape (n_samples, n_features) oder (seq_length, n_features)

        Returns:
            Self fuer Method Chaining
        """
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        # Vermeidung von Division durch 0
        self.std = np.maximum(self.std, self.epsilon)
        self._fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Wendet Z-Score Normalisierung an.

        Args:
            data: Array mit Shape (n_samples, n_features)

        Returns:
            Normalisiertes Array
        """
        if not self._fitted:
            raise ValueError('Normalizer muss zuerst gefittet werden')
        return (data - self.mean) / self.std

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fittet und transformiert in einem Schritt.

        Args:
            data: Array mit Shape (n_samples, n_features)

        Returns:
            Normalisiertes Array
        """
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Kehrt die Normalisierung um.

        Args:
            data: Normalisiertes Array

        Returns:
            Original-skaliertes Array
        """
        if not self._fitted:
            raise ValueError('Normalizer muss zuerst gefittet werden')
        return (data * self.std) + self.mean

    def get_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gibt Normalisierungsparameter zurueck.

        Returns:
            Tuple aus (mean, std)
        """
        if not self._fitted:
            raise ValueError('Normalizer nicht gefittet')
        return self.mean.copy(), self.std.copy()

    def set_params(self, mean: np.ndarray, std: np.ndarray):
        """
        Setzt Normalisierungsparameter manuell.

        Args:
            mean: Mittelwert pro Feature
            std: Standardabweichung pro Feature
        """
        self.mean = mean
        self.std = np.maximum(std, self.epsilon)
        self._fitted = True

    def reset(self):
        """Setzt den Normalizer zurueck."""
        self.mean = None
        self.std = None
        self._fitted = False


class MinMaxNormalizer:
    """
    Min-Max Normalisierung auf Bereich [0, 1] oder [feature_range].

    Formel: x_norm = (x - min) / (max - min)

    Attributes:
        feature_range: Zielbereich (default: (0, 1))
        min_val: Gespeichertes Minimum
        max_val: Gespeichertes Maximum
    """

    def __init__(self, feature_range: Tuple[float, float] = (0, 1), epsilon: float = 1e-8):
        """
        Initialisiert den Min-Max Normalizer.

        Args:
            feature_range: Zielbereich fuer normalisierte Werte
            epsilon: Minimaler Unterschied zwischen min und max
        """
        self.feature_range = feature_range
        self.epsilon = epsilon
        self.min_val: Optional[np.ndarray] = None
        self.max_val: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, data: np.ndarray) -> 'MinMaxNormalizer':
        """
        Berechnet Minimum und Maximum.

        Args:
            data: Array mit Shape (n_samples, n_features)

        Returns:
            Self fuer Method Chaining
        """
        self.min_val = np.min(data, axis=0)
        self.max_val = np.max(data, axis=0)
        # Vermeidung von Division durch 0
        diff = self.max_val - self.min_val
        diff = np.maximum(diff, self.epsilon)
        self.max_val = self.min_val + diff
        self._fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Wendet Min-Max Normalisierung an.

        Args:
            data: Array mit Shape (n_samples, n_features)

        Returns:
            Normalisiertes Array
        """
        if not self._fitted:
            raise ValueError('Normalizer muss zuerst gefittet werden')

        # Normalisieren auf [0, 1]
        data_norm = (data - self.min_val) / (self.max_val - self.min_val)

        # Skalieren auf feature_range
        range_min, range_max = self.feature_range
        return data_norm * (range_max - range_min) + range_min

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fittet und transformiert in einem Schritt.

        Args:
            data: Array mit Shape (n_samples, n_features)

        Returns:
            Normalisiertes Array
        """
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Kehrt die Normalisierung um.

        Args:
            data: Normalisiertes Array

        Returns:
            Original-skaliertes Array
        """
        if not self._fitted:
            raise ValueError('Normalizer muss zuerst gefittet werden')

        # Zurueck auf [0, 1]
        range_min, range_max = self.feature_range
        data_01 = (data - range_min) / (range_max - range_min)

        # Zurueck auf Original-Skala
        return data_01 * (self.max_val - self.min_val) + self.min_val


class RobustNormalizer:
    """
    Robuste Normalisierung basierend auf Median und IQR.

    Weniger empfindlich gegenueber Ausreissern als Z-Score.
    Formel: z = (x - median) / IQR

    Attributes:
        median: Gespeicherter Median
        iqr: Interquartilsabstand (Q3 - Q1)
    """

    def __init__(self, epsilon: float = 1e-8):
        """
        Initialisiert den Robust Normalizer.

        Args:
            epsilon: Minimaler IQR-Wert
        """
        self.epsilon = epsilon
        self.median: Optional[np.ndarray] = None
        self.iqr: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, data: np.ndarray) -> 'RobustNormalizer':
        """
        Berechnet Median und IQR.

        Args:
            data: Array mit Shape (n_samples, n_features)

        Returns:
            Self fuer Method Chaining
        """
        self.median = np.median(data, axis=0)
        q1 = np.percentile(data, 25, axis=0)
        q3 = np.percentile(data, 75, axis=0)
        self.iqr = np.maximum(q3 - q1, self.epsilon)
        self._fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Wendet robuste Normalisierung an.

        Args:
            data: Array mit Shape (n_samples, n_features)

        Returns:
            Normalisiertes Array
        """
        if not self._fitted:
            raise ValueError('Normalizer muss zuerst gefittet werden')
        return (data - self.median) / self.iqr

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fittet und transformiert in einem Schritt."""
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Kehrt die Normalisierung um."""
        if not self._fitted:
            raise ValueError('Normalizer muss zuerst gefittet werden')
        return (data * self.iqr) + self.median
