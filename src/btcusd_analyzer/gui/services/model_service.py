"""
ModelService - Zentraler Service fuer Modell-Operationen

Extrahiert aus main_window.py fuer bessere Wartbarkeit.
"""

from pathlib import Path
from typing import Optional, Callable, List, Dict, Any

import numpy as np


class ModelService:
    """
    Service fuer alle Modell-bezogenen Operationen.

    Funktionen:
    - Modell laden (einzeln und letztes Modell)
    - Modell speichern
    - Vorhersagen durchfuehren

    Attributes:
        model: Aktuell geladenes Modell
        model_path: Pfad zum geladenen Modell
        model_info: Metadaten zum Modell
    """

    def __init__(
        self,
        models_dir: Path,
        results_dir: Path,
        log_callback: Optional[Callable] = None
    ):
        """
        Initialisiert den ModelService.

        Args:
            models_dir: Verzeichnis fuer Modelle
            results_dir: Verzeichnis fuer Ergebnisse
            log_callback: Optionale Callback-Funktion fuer Logging
        """
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self._log_callback = log_callback

        # State
        self.model = None
        self.model_path: Optional[Path] = None
        self.model_info: Optional[Dict[str, Any]] = None

    def _log(self, message: str, level: str = 'INFO'):
        """Internes Logging."""
        if self._log_callback:
            self._log_callback(message, level)

    def load_model(self, filepath: Path) -> bool:
        """
        Laedt ein gespeichertes Modell.

        Args:
            filepath: Pfad zur Modell-Datei

        Returns:
            True bei Erfolg, False bei Fehler
        """
        self._log(f'Lade Modell: {filepath.name}', 'INFO')

        try:
            import torch
            from ...models.factory import ModelFactory

            checkpoint = torch.load(filepath, map_location='cpu')
            self.model_path = filepath

            # Model-Info aus Checkpoint extrahieren
            if 'model_info' in checkpoint:
                self.model_info = checkpoint['model_info']
                model_type = self.model_info.get('model_type', 'bilstm')
                input_size = self.model_info.get('input_size', 6)
                hidden_size = self.model_info.get('hidden_size', 100)
                num_layers = self.model_info.get('num_layers', 2)
                num_classes = self.model_info.get('num_classes', 3)

                # Modell erstellen und Gewichte laden
                self.model = ModelFactory.create(
                    model_type,
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    num_classes=num_classes
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()

                self._log(f'Modell: {model_type.upper()}', 'INFO')
                self._log(f'Parameter: {self.model.count_parameters():,}', 'INFO')
                self._log('Modell geladen', 'SUCCESS')
                return True
            else:
                self._log('Checkpoint ohne model_info - manuelle Konfiguration erforderlich', 'WARNING')
                return False

        except Exception as e:
            self._log(f'Modell-Ladefehler: {e}', 'ERROR')
            return False

    def load_last_model(self) -> bool:
        """
        Laedt das zuletzt verwendete Modell.

        Returns:
            True bei Erfolg, False bei Fehler
        """
        self._log('Suche letztes Modell...', 'INFO')

        # Suche in results und models Verzeichnissen
        search_dirs = [self.results_dir, self.models_dir]
        model_files: List[Path] = []

        for search_dir in search_dirs:
            if search_dir.exists():
                model_files.extend(search_dir.rglob('*.pt'))
                model_files.extend(search_dir.rglob('*.pth'))

        if not model_files:
            self._log('Kein vorheriges Modell gefunden', 'WARNING')
            return False

        # Nach Aenderungszeit sortieren (neueste zuerst)
        model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest = model_files[0]
        self._log(f'Gefunden: {latest.name}', 'INFO')

        return self.load_model(latest)

    def predict(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Fuehrt eine Vorhersage durch.

        Args:
            X: Input-Daten als numpy array [batch, seq_len, features]

        Returns:
            Vorhersagen als numpy array oder None bei Fehler
        """
        if self.model is None:
            self._log('Kein Modell geladen', 'ERROR')
            return None

        try:
            import torch

            self.model.eval()

            with torch.no_grad():
                if isinstance(X, np.ndarray):
                    X_tensor = torch.from_numpy(X).float()
                else:
                    X_tensor = X

                # Vorhersage
                outputs = self.model(X_tensor)
                predictions = torch.argmax(outputs, dim=1).numpy()

            return predictions

        except Exception as e:
            self._log(f'Vorhersage-Fehler: {e}', 'ERROR')
            return None

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Fuehrt eine Vorhersage mit Wahrscheinlichkeiten durch.

        Args:
            X: Input-Daten als numpy array [batch, seq_len, features]

        Returns:
            Wahrscheinlichkeiten als numpy array [batch, num_classes] oder None
        """
        if self.model is None:
            self._log('Kein Modell geladen', 'ERROR')
            return None

        try:
            import torch
            import torch.nn.functional as F

            self.model.eval()

            with torch.no_grad():
                if isinstance(X, np.ndarray):
                    X_tensor = torch.from_numpy(X).float()
                else:
                    X_tensor = X

                outputs = self.model(X_tensor)
                proba = F.softmax(outputs, dim=1).numpy()

            return proba

        except Exception as e:
            self._log(f'Vorhersage-Fehler: {e}', 'ERROR')
            return None

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Gibt eine Zusammenfassung des geladenen Modells zurueck.

        Returns:
            Dictionary mit Modell-Informationen
        """
        if self.model is None:
            return {}

        summary = {
            'loaded': True,
            'path': str(self.model_path) if self.model_path else None,
            'parameters': self.model.count_parameters(),
        }

        if self.model_info:
            summary.update(self.model_info)

        return summary

    def clear(self):
        """Setzt den Service zurueck."""
        self.model = None
        self.model_path = None
        self.model_info = None
