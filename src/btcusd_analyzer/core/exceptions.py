"""
Custom Exceptions - Benutzerdefinierte Fehlerklassen fuer BTCUSD Analyzer

Hierarchie:
    BTCAnalyzerError (Basis)
    ├── DataError (Datenfehler)
    │   ├── DataValidationError (Validierungsfehler)
    │   ├── DataFormatError (Formatfehler)
    │   └── MissingDataError (Fehlende Daten)
    ├── ModelError (Modellfehler)
    │   ├── ModelNotFoundError (Modell nicht gefunden)
    │   └── ModelLoadError (Ladefehler)
    ├── ConfigError (Konfigurationsfehler)
    ├── TradingError (Trading-Fehler)
    │   ├── APIError (API-Fehler)
    │   └── InsufficientFundsError (Kapitalfehler)
    └── SaveError (Speicherfehler)
"""

from typing import List, Optional


class BTCAnalyzerError(Exception):
    """Basisklasse fuer alle BTCUSD Analyzer Fehler."""

    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message}\nDetails: {self.details}"
        return self.message


# =============================================================================
# Datenfehler
# =============================================================================

class DataError(BTCAnalyzerError):
    """Basisklasse fuer Datenfehler."""
    pass


class DataValidationError(DataError):
    """
    Wird geworfen wenn Daten die Validierung nicht bestehen.

    Beispiele:
    - Fehlende Pflichtspalten
    - Ungueltige Werte
    - Inkonsistente Daten
    """

    def __init__(self, message: str, invalid_fields: Optional[List[str]] = None,
                 details: Optional[str] = None):
        super().__init__(message, details)
        self.invalid_fields = invalid_fields or []

    def __str__(self) -> str:
        base = super().__str__()
        if self.invalid_fields:
            return f"{base}\nUngueltige Felder: {', '.join(self.invalid_fields)}"
        return base


class DataFormatError(DataError):
    """
    Wird geworfen wenn das Datenformat nicht erkannt wird.

    Beispiele:
    - Unbekanntes CSV-Format
    - Falsches Datumsformat
    - Ungueltige Kodierung
    """

    def __init__(self, message: str, expected_format: Optional[str] = None,
                 actual_format: Optional[str] = None, details: Optional[str] = None):
        super().__init__(message, details)
        self.expected_format = expected_format
        self.actual_format = actual_format


class MissingDataError(DataError):
    """
    Wird geworfen wenn benoetigte Daten fehlen.

    Beispiele:
    - Datei nicht gefunden
    - Leerer DataFrame
    - Fehlende Spalten
    """

    def __init__(self, message: str, missing_items: Optional[List[str]] = None,
                 details: Optional[str] = None):
        super().__init__(message, details)
        self.missing_items = missing_items or []

    def __str__(self) -> str:
        base = super().__str__()
        if self.missing_items:
            return f"{base}\nFehlend: {', '.join(self.missing_items)}"
        return base


# =============================================================================
# Modellfehler
# =============================================================================

class ModelError(BTCAnalyzerError):
    """Basisklasse fuer Modellfehler."""
    pass


class ModelNotFoundError(ModelError):
    """Wird geworfen wenn ein Modell nicht gefunden wird."""
    pass


class ModelLoadError(ModelError):
    """
    Wird geworfen wenn ein Modell nicht geladen werden kann.

    Beispiele:
    - Korrupte Checkpoint-Datei
    - Inkompatible Modellversion
    - Fehlende Modell-Metadaten
    """

    def __init__(self, message: str, model_path: Optional[str] = None,
                 details: Optional[str] = None):
        super().__init__(message, details)
        self.model_path = model_path


# =============================================================================
# Konfigurationsfehler
# =============================================================================

class ConfigError(BTCAnalyzerError):
    """
    Wird geworfen bei Konfigurationsfehlern.

    Beispiele:
    - Ungueltige Konfigurationsdatei
    - Fehlende Pflichtparameter
    - Widersprüchliche Einstellungen
    """

    def __init__(self, message: str, config_key: Optional[str] = None,
                 details: Optional[str] = None):
        super().__init__(message, details)
        self.config_key = config_key


# =============================================================================
# Trading-Fehler
# =============================================================================

class TradingError(BTCAnalyzerError):
    """Basisklasse fuer Trading-Fehler."""
    pass


class APIError(TradingError):
    """
    Wird geworfen bei API-Fehlern (Binance, etc.).

    Beispiele:
    - Authentifizierungsfehler
    - Rate-Limit ueberschritten
    - Netzwerkfehler
    """

    def __init__(self, message: str, status_code: Optional[int] = None,
                 details: Optional[str] = None):
        super().__init__(message, details)
        self.status_code = status_code


class InsufficientFundsError(TradingError):
    """Wird geworfen wenn nicht genuegend Kapital vorhanden ist."""
    pass


# =============================================================================
# Speicherfehler
# =============================================================================

class SaveError(BTCAnalyzerError):
    """
    Wird geworfen bei Fehlern waehrend des Speicherns.

    Beispiele:
    - Transaktion fehlgeschlagen
    - Validierung fehlgeschlagen
    - Rollback erforderlich
    """

    def __init__(self, message: str, operation: Optional[str] = None,
                 details: Optional[str] = None):
        super().__init__(message, details)
        self.operation = operation
