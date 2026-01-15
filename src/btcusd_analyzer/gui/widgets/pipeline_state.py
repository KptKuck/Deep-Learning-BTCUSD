"""
PipelineState - Status-Management fuer die Datenvorbereitungs-Pipeline
"""

from enum import IntEnum


class PipelineStage(IntEnum):
    """Stufen der Datenvorbereitungs-Pipeline."""
    NONE = 0       # Keine Stufe abgeschlossen
    PEAKS = 1      # Tab 1: Peaks gefunden
    LABELS = 2     # Tab 2: Labels generiert
    FEATURES = 3   # Tab 3: Features ausgewaehlt
    SAMPLES = 4    # Tab 4: Samples berechnet


class PipelineState:
    """
    Verwaltet den Status der Datenvorbereitungs-Pipeline.
    Ersetzt die 4 separaten Boolean-Flags durch eine zentrale Zustandsverwaltung.
    """

    def __init__(self):
        self._current_stage = PipelineStage.NONE

    @property
    def current_stage(self) -> PipelineStage:
        """Gibt die aktuelle Stufe zurueck."""
        return self._current_stage

    def advance_to(self, stage: PipelineStage):
        """Setzt die Pipeline auf eine bestimmte Stufe."""
        self._current_stage = stage

    def invalidate_from(self, stage: PipelineStage):
        """
        Invalidiert alle Stufen ab der angegebenen Stufe.
        Wird aufgerufen wenn sich Parameter aendern.
        """
        if self._current_stage >= stage:
            self._current_stage = PipelineStage(max(0, stage - 1))

    def is_valid(self, stage: PipelineStage) -> bool:
        """Prueft ob eine Stufe gueltig (abgeschlossen) ist."""
        return self._current_stage >= stage

    @property
    def peaks_valid(self) -> bool:
        """Kompatibilitaet: Prueft ob Peaks gefunden wurden."""
        return self.is_valid(PipelineStage.PEAKS)

    @property
    def labels_valid(self) -> bool:
        """Kompatibilitaet: Prueft ob Labels generiert wurden."""
        return self.is_valid(PipelineStage.LABELS)

    @property
    def features_valid(self) -> bool:
        """Kompatibilitaet: Prueft ob Features ausgewaehlt wurden."""
        return self.is_valid(PipelineStage.FEATURES)

    @property
    def samples_valid(self) -> bool:
        """Kompatibilitaet: Prueft ob Samples berechnet wurden."""
        return self.is_valid(PipelineStage.SAMPLES)

    def reset(self):
        """Setzt die Pipeline zurueck."""
        self._current_stage = PipelineStage.NONE

    def __repr__(self) -> str:
        return f'PipelineState(stage={self._current_stage.name})'
