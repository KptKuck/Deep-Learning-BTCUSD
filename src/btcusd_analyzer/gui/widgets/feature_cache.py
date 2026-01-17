"""
FeatureCache - Intelligenter Cache fuer Feature-Berechnungen
"""

from typing import Dict, Any, Optional, Callable
import numpy as np
import pandas as pd


class FeatureCache:
    """
    Intelligenter Cache fuer Feature-Berechnungen.
    Cached berechnete Features und berechnet nur bei Aenderungen neu.
    """

    def __init__(self):
        self._cache: Dict[str, np.ndarray] = {}
        self._data_id: Optional[int] = None  # ID der Daten (id(df))

    def invalidate(self):
        """Leert den gesamten Cache."""
        self._cache.clear()
        self._data_id = None

    def set_data(self, data: pd.DataFrame):
        """Setzt neue Basisdaten und invalidiert Cache wenn noetig."""
        new_id = id(data)
        if new_id != self._data_id:
            self._cache.clear()
            self._data_id = new_id

    def get(self, key: str) -> Optional[np.ndarray]:
        """Holt gecachtes Feature oder None."""
        return self._cache.get(key)

    def set(self, key: str, values: np.ndarray):
        """Speichert berechnetes Feature."""
        self._cache[key] = values

    def has(self, key: str) -> bool:
        """Prueft ob Feature gecached ist."""
        return key in self._cache

    def compute_or_get(self, key: str, compute_fn: Callable) -> np.ndarray:
        """Berechnet Feature falls nicht gecached."""
        if key not in self._cache:
            self._cache[key] = compute_fn()
        return self._cache[key]

    @property
    def cached_features(self) -> Dict[str, np.ndarray]:
        """Gibt alle gecachten Features zurueck."""
        return self._cache.copy()

    def compute_feature(self, data: pd.DataFrame, feat_name: str,
                        feat_params: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Berechnet ein einzelnes Feature mit Caching.

        Args:
            data: DataFrame mit OHLCV Daten
            feat_name: Name des Features
            feat_params: Parameter (z.B. {'period': 20})

        Returns:
            numpy array mit Feature-Werten oder None
        """
        # Cache-Key mit Parametern
        param_str = '_'.join(f'{k}{v}' for k, v in sorted(feat_params.items()))
        cache_key = f'{feat_name}_{param_str}' if param_str else feat_name

        # Aus Cache holen falls vorhanden
        if self.has(cache_key):
            return self.get(cache_key)

        # Feature berechnen
        try:
            from ...data.processor import FeatureProcessor

            # FeatureProcessor erwartet features im Konstruktor
            processor = FeatureProcessor(features=[feat_name])

            # Feature berechnen
            result = processor.process(data, validate=False)

            if feat_name in result.columns:
                values = result[feat_name].values
                self.set(cache_key, values)
                return values

        except Exception:
            pass

        return None

    def __len__(self) -> int:
        """Gibt die Anzahl gecachter Features zurueck."""
        return len(self._cache)

    def __repr__(self) -> str:
        return f'FeatureCache(cached={len(self._cache)} features)'
