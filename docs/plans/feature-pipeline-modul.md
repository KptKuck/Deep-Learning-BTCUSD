# Plan: FeatureProcessingPipeline Modul

## Ziel
Zentralisierung der Feature-Verarbeitung in einem wiederverwendbaren Modul.
Eliminierung von Code-Duplizierung bei NaN-Handling, Normalisierung und Feature-Extraktion.

---

## Aktueller Stand (Probleme)

### Code-Duplizierung an 3+ Stellen:

| Datei | Zeilen | Was wird dupliziert |
|-------|--------|---------------------|
| `prepare_data_window.py` | 1373-1415 | FeatureProcessor + NaN-Handling |
| `backtest_window.py` | 215-244 | FeatureProcessor + NaN-Handling |
| `walk_forward.py` | 489, 754, 976 | FeatureProcessor + Normalisierung |

### Duplizierter Code-Block (Beispiel):
```python
# In BEIDEN prepare_data_window.py UND backtest_window.py:
processor = FeatureProcessor(features=feature_list)
processed = processor.process(data)
feature_data = processed[feature_cols].values.astype(np.float32)

nan_mask = ~np.isnan(feature_data).any(axis=1)
nan_count = np.sum(~nan_mask)
if nan_count > 0:
    feature_data = feature_data[nan_mask]
    # + DataFrame sync
    self._log(f"{nan_count} Zeilen mit NaN entfernt", 'INFO')
```

---

## Loesung: FeatureProcessingPipeline

### Neue Datei
`src/btcusd_analyzer/data/feature_pipeline.py`

### Klassen-Design

```python
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd

from .processor import FeatureProcessor
from ..training.normalizer import ZScoreNormalizer
from ..core.logger import get_logger


@dataclass
class PipelineResult:
    """Ergebnis der Feature-Pipeline."""
    feature_matrix: np.ndarray      # (N x F) Float32 Array
    data_clean: pd.DataFrame        # Synchronisierter DataFrame (ohne NaN-Zeilen)
    nan_count: int                  # Anzahl entfernter Zeilen
    feature_names: List[str]        # Liste der Feature-Namen
    normalizer: Optional[ZScoreNormalizer] = None  # Gefitteter Normalizer


class FeatureProcessingPipeline:
    """
    Zentrale Pipeline fuer Feature-Verarbeitung.

    Kombiniert:
    1. Feature-Berechnung (FeatureProcessor)
    2. NaN-Zeilen entfernen (Indikator-Warmup)
    3. Optionale Z-Score Normalisierung

    Verwendung:
    -----------
    # Training (fit normalizer):
    pipeline = FeatureProcessingPipeline(features=['Open', 'SMA', 'RSI'])
    result = pipeline.process(train_data, normalize=True, fit_normalizer=True)

    # Inference (use fitted normalizer):
    pipeline.set_normalizer(saved_normalizer)
    result = pipeline.process(test_data, normalize=True, fit_normalizer=False)
    """

    def __init__(self, features: List[str], logger=None):
        """
        Args:
            features: Liste der Feature-Namen (z.B. ['Open', 'SMA', 'RSI'])
            logger: Optionaler Logger (sonst wird globaler Logger verwendet)
        """
        self._features = features
        self._logger = logger or get_logger()
        self._normalizer: Optional[ZScoreNormalizer] = None
        self._processor: Optional[FeatureProcessor] = None

    def process(self,
                data: pd.DataFrame,
                normalize: bool = True,
                fit_normalizer: bool = True,
                log_nan_removal: bool = True
    ) -> PipelineResult:
        """
        Verarbeitet Daten durch die komplette Pipeline.

        Args:
            data: DataFrame mit OHLCV + DateTime Spalten
            normalize: Z-Score Normalisierung anwenden?
            fit_normalizer: Normalizer neu fitten? (False bei Inference)
            log_nan_removal: NaN-Entfernung loggen?

        Returns:
            PipelineResult mit allen Ergebnissen
        """
        # 1. Feature-Berechnung
        self._processor = FeatureProcessor(features=self._features)
        processed_df = self._processor.process(data.copy())

        # 2. Feature-Matrix extrahieren
        feature_cols = [f for f in self._features if f in processed_df.columns]
        feature_matrix = processed_df[feature_cols].values.astype(np.float32)

        # 3. NaN-Zeilen entfernen
        nan_mask = ~np.isnan(feature_matrix).any(axis=1)
        nan_count = int(np.sum(~nan_mask))

        if nan_count > 0:
            feature_matrix = feature_matrix[nan_mask]
            data_clean = data.iloc[nan_mask].reset_index(drop=True)

            if log_nan_removal:
                self._logger.info(
                    f"[Pipeline] {nan_count} Zeilen mit NaN entfernt (Indikator-Warmup)"
                )
        else:
            data_clean = data.reset_index(drop=True)

        # 4. Optionale Normalisierung
        normalizer_out = None
        if normalize:
            if fit_normalizer or self._normalizer is None:
                self._normalizer = ZScoreNormalizer()
                feature_matrix = self._normalizer.fit_transform(feature_matrix)
            else:
                feature_matrix = self._normalizer.transform(feature_matrix)
            normalizer_out = self._normalizer

        return PipelineResult(
            feature_matrix=feature_matrix,
            data_clean=data_clean,
            nan_count=nan_count,
            feature_names=feature_cols,
            normalizer=normalizer_out
        )

    def get_normalizer(self) -> Optional[ZScoreNormalizer]:
        """Gibt den gefitteten Normalizer zurueck."""
        return self._normalizer

    def set_normalizer(self, normalizer: ZScoreNormalizer):
        """Setzt einen vortrainierten Normalizer (fuer Inference)."""
        self._normalizer = normalizer

    @property
    def features(self) -> List[str]:
        """Gibt die Feature-Liste zurueck."""
        return self._features.copy()
```

---

## Integration

### 1. prepare_data_window.py (Training)

**Vorher (Zeilen 1373-1415):**
```python
processor = FeatureProcessor(features=selected_features)
processed = processor.process(self.data.copy())
features = processed[feature_cols].values.astype(np.float32)

nan_mask = ~np.isnan(features).any(axis=1)
nan_count = np.sum(~nan_mask)
if nan_count > 0:
    features = features[nan_mask]
    train_labels = train_labels[nan_mask]
    self._log(f"{nan_count} Zeilen mit NaN entfernt (Indikator-Warmup)", 'INFO')
```

**Nachher:**
```python
from ..data.feature_pipeline import FeatureProcessingPipeline

pipeline = FeatureProcessingPipeline(features=selected_features, logger=self._logger)
result = pipeline.process(self.data, normalize=False)

features = result.feature_matrix
train_labels = train_labels[~np.isnan(self.data[...]).any(axis=1)]  # Sync labels
# nan_count wird automatisch geloggt
```

### 2. backtest_window.py (Inference)

**Vorher (Zeilen 215-244):**
```python
processor = FeatureProcessor(features=self.model_info['features'])
processed = processor.process(self.data.copy())
feature_data = processed[feature_cols].values.astype(np.float32)

nan_mask = ~np.isnan(feature_data).any(axis=1)
nan_count = np.sum(~nan_mask)
if nan_count > 0:
    feature_data = feature_data[nan_mask]
    self.data = self.data.iloc[nan_mask].reset_index(drop=True)
    self._log(f"{nan_count} Zeilen mit NaN entfernt", 'INFO')
```

**Nachher:**
```python
from ..data.feature_pipeline import FeatureProcessingPipeline

pipeline = FeatureProcessingPipeline(
    features=self.model_info['features'],
    logger=self._logger
)
pipeline.set_normalizer(self.model_info.get('normalizer'))

result = pipeline.process(self.data, normalize=True, fit_normalizer=False)

feature_data = result.feature_matrix
self.data = result.data_clean
```

### 3. walk_forward.py (Training + Inference pro Fold)

**Nachher:**
```python
# Pro Fold:
pipeline = FeatureProcessingPipeline(features=self.features)

# Training
train_result = pipeline.process(train_data, normalize=True, fit_normalizer=True)
train_features = train_result.feature_matrix

# Validation (gleicher Normalizer)
val_result = pipeline.process(val_data, normalize=True, fit_normalizer=False)
val_features = val_result.feature_matrix
```

---

## Dateien zu aendern

| Datei | Aenderung |
|-------|-----------|
| `data/feature_pipeline.py` | **NEU** - FeatureProcessingPipeline Klasse |
| `data/validator.py` | **NEU** - DataStreamValidator Klasse |
| `data/__init__.py` | Exports hinzufuegen |
| `gui/prepare_data_window.py` | Import + Nutzung Pipeline |
| `gui/backtest/backtest_window.py` | Import + Nutzung Pipeline + Validator |
| `backtester/walk_forward.py` | Import + Nutzung (optional) |
| `backtester/backtrader_engine.py` | Import + Nutzung (optional) |

---

## Implementierungs-Reihenfolge

1. **Pipeline-Modul erstellen** (`feature_pipeline.py`)
2. **Unit-Test** - Pipeline isoliert testen
3. **prepare_data_window.py** - Integration
4. **backtest_window.py** - Integration
5. **walk_forward.py** - Integration (optional, komplexer)
6. **Commit** - "Add centralized FeatureProcessingPipeline"

---

## Verifikation

```bash
# 1. Import-Test
python -c "from btcusd_analyzer.data.feature_pipeline import FeatureProcessingPipeline; print('OK')"

# 2. Funktions-Test
python -c "
from btcusd_analyzer.data.feature_pipeline import FeatureProcessingPipeline
import pandas as pd

# Dummy-Daten
df = pd.DataFrame({
    'Open': [100, 101, 102, 103, 104],
    'High': [101, 102, 103, 104, 105],
    'Low': [99, 100, 101, 102, 103],
    'Close': [100.5, 101.5, 102.5, 103.5, 104.5],
    'Volume': [1000, 1100, 1200, 1300, 1400],
})

pipeline = FeatureProcessingPipeline(features=['Open', 'Close'])
result = pipeline.process(df, normalize=False)

print(f'Matrix Shape: {result.feature_matrix.shape}')
print(f'NaN entfernt: {result.nan_count}')
print(f'Features: {result.feature_names}')
"

# 3. Anwendung starten und Prepare Data + Backtest testen
python -m btcusd_analyzer.main
```

---

## Vorteile nach Implementierung

| Aspekt | Vorher | Nachher |
|--------|--------|---------|
| Code-Zeilen pro Aufruf | ~15 | ~5 |
| NaN-Handling | 3x dupliziert | 1x zentral |
| Normalisierung | Manuell verwaltet | Automatisch |
| Testbarkeit | Schwierig | Einfach (Unit-Tests) |
| Erweiterbarkeit | Aenderungen an 3+ Stellen | Aenderung an 1 Stelle |

---

## Risiken / Bedenken

1. **Label-Synchronisation**: Bei Training muessen Labels mit NaN-Mask synchronisiert werden
   - Loesung: Pipeline gibt `nan_mask` oder Index zurueck fuer manuelle Sync

2. **Rueckwaertskompatibilitaet**: Bestehender Code muss funktionieren
   - Loesung: Schrittweise Migration, alte Methode parallel erhalten

3. **Walk-Forward Komplexitaet**: Hat eigene Normalisierungs-Logik pro Fold
   - Loesung: Pipeline unterstuetzt `fit_normalizer=True/False` pro Aufruf

---

## Modul 2: DataStreamValidator (Tapeten-Muster-Pruefung)

### Ziel
Sicherstellen, dass der Backtester **exakt dieselben Daten** bekommt wie das Modell beim Training.
Wie zwei Tapeten mit Muster - pruefen ob sie nahtlos aneinander passen.

### Problem
Aktuell kann es zu subtilen Diskrepanzen kommen:
- Feature-Reihenfolge unterschiedlich
- Normalisierungs-Parameter abweichend
- NaN-Handling inkonsistent
- Sequence-Laenge/Padding unterschiedlich

### Loesung: DataStreamValidator

```python
@dataclass
class ValidationReport:
    """Ergebnis der Daten-Validierung."""
    is_valid: bool
    errors: List[str]           # Kritische Fehler (Abbruch)
    warnings: List[str]         # Warnungen (weiter moeglich)
    fingerprint_match: bool     # Hash-Vergleich
    details: Dict[str, Any]     # Detaillierte Metriken


class DataStreamValidator:
    """
    Validiert Datenstrom-Konsistenz zwischen Training und Inference.

    Prueft:
    1. Feature-Namen und Reihenfolge
    2. Normalisierungs-Parameter (mean, std)
    3. Daten-Shape und Typ
    4. Fingerprint/Hash der Pipeline-Konfiguration
    5. Sequence-Parameter (Laenge, Stride)
    6. Zeitindex-Konsistenz (Intervall, Format, Luecken)
    """

    def __init__(self, training_config: Dict[str, Any]):
        """
        Args:
            training_config: Gespeicherte Konfiguration vom Training
                - features: List[str]
                - normalizer_params: Dict (mean, std per feature)
                - sequence_length: int
                - nan_handling: str
                - pipeline_hash: str
                - time_config: Dict (interval_seconds, first_timestamp, last_timestamp)
        """
        self._config = training_config
        self._logger = get_logger()

    @staticmethod
    def create_fingerprint(pipeline: FeatureProcessingPipeline) -> str:
        """
        Erstellt einen Hash-Fingerprint der Pipeline-Konfiguration.
        Wird beim Training gespeichert und beim Backtest verglichen.
        """
        config_str = json.dumps({
            'features': sorted(pipeline.features),
            'normalizer_mean': pipeline.get_normalizer().mean_.tolist() if pipeline.get_normalizer() else None,
            'normalizer_std': pipeline.get_normalizer().std_.tolist() if pipeline.get_normalizer() else None,
        }, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def validate_inference_data(self,
                                 inference_pipeline: FeatureProcessingPipeline,
                                 sample_data: np.ndarray) -> ValidationReport:
        """
        Validiert Inference-Daten gegen Training-Konfiguration.

        Args:
            inference_pipeline: Pipeline fuer Backtest/Inference
            sample_data: Erste N Zeilen der verarbeiteten Daten

        Returns:
            ValidationReport mit Ergebnis
        """
        errors = []
        warnings = []
        details = {}

        # 1. Feature-Namen pruefen
        expected_features = self._config['features']
        actual_features = inference_pipeline.features

        if expected_features != actual_features:
            if set(expected_features) != set(actual_features):
                errors.append(
                    f"Feature-Mismatch: Erwartet {expected_features}, "
                    f"erhalten {actual_features}"
                )
            else:
                warnings.append(
                    f"Feature-Reihenfolge unterschiedlich: "
                    f"Erwartet {expected_features}, erhalten {actual_features}"
                )

        details['features_expected'] = expected_features
        details['features_actual'] = actual_features

        # 2. Normalizer-Parameter pruefen
        if 'normalizer_params' in self._config:
            expected_mean = np.array(self._config['normalizer_params']['mean'])
            expected_std = np.array(self._config['normalizer_params']['std'])

            actual_norm = inference_pipeline.get_normalizer()
            if actual_norm:
                # Toleranz fuer Floating-Point-Vergleich
                if not np.allclose(expected_mean, actual_norm.mean_, rtol=1e-5):
                    errors.append("Normalizer-Mean weicht ab!")
                    details['mean_diff'] = np.abs(expected_mean - actual_norm.mean_).max()

                if not np.allclose(expected_std, actual_norm.std_, rtol=1e-5):
                    errors.append("Normalizer-Std weicht ab!")
                    details['std_diff'] = np.abs(expected_std - actual_norm.std_).max()
            else:
                warnings.append("Kein Normalizer in Inference-Pipeline")

        # 3. Daten-Shape pruefen
        if sample_data is not None:
            expected_features_count = len(expected_features)
            actual_features_count = sample_data.shape[1] if len(sample_data.shape) > 1 else 1

            if expected_features_count != actual_features_count:
                errors.append(
                    f"Feature-Anzahl: Erwartet {expected_features_count}, "
                    f"erhalten {actual_features_count}"
                )

            details['data_shape'] = sample_data.shape
            details['data_dtype'] = str(sample_data.dtype)

        # 4. Fingerprint vergleichen
        fingerprint_match = False
        if 'pipeline_hash' in self._config:
            actual_hash = self.create_fingerprint(inference_pipeline)
            fingerprint_match = (actual_hash == self._config['pipeline_hash'])

            if not fingerprint_match:
                warnings.append(
                    f"Pipeline-Fingerprint unterschiedlich: "
                    f"Training={self._config['pipeline_hash']}, "
                    f"Inference={actual_hash}"
                )

            details['hash_expected'] = self._config['pipeline_hash']
            details['hash_actual'] = actual_hash

        # 5. Sequence-Parameter (falls vorhanden)
        if 'sequence_length' in self._config:
            details['sequence_length'] = self._config['sequence_length']

        # 6. Zeitindex-Pruefung wird separat aufgerufen (benoetigt DataFrame)

        is_valid = len(errors) == 0

        return ValidationReport(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            fingerprint_match=fingerprint_match,
            details=details
        )

    def validate_time_index(self,
                             data: pd.DataFrame,
                             datetime_col: str = 'DateTime') -> Tuple[List[str], List[str], Dict]:
        """
        Validiert den Zeitindex der Daten.

        Prueft:
        - Zeitintervall konsistent (z.B. immer 1h)
        - Keine Luecken im Index
        - Format passt zu Training
        - Zeitzone konsistent

        Args:
            data: DataFrame mit DateTime-Spalte
            datetime_col: Name der DateTime-Spalte

        Returns:
            (errors, warnings, details)
        """
        errors = []
        warnings = []
        details = {}

        if datetime_col not in data.columns:
            errors.append(f"DateTime-Spalte '{datetime_col}' nicht gefunden")
            return errors, warnings, details

        # DateTime parsen
        try:
            timestamps = pd.to_datetime(data[datetime_col])
        except Exception as e:
            errors.append(f"DateTime-Parsing fehlgeschlagen: {e}")
            return errors, warnings, details

        # Zeitdifferenzen berechnen
        time_diffs = timestamps.diff().dropna()
        if len(time_diffs) == 0:
            warnings.append("Zu wenig Daten fuer Zeitintervall-Analyse")
            return errors, warnings, details

        # Haeufigste Differenz = erwartetes Intervall
        actual_interval = time_diffs.mode().iloc[0]
        actual_interval_seconds = actual_interval.total_seconds()

        details['actual_interval_seconds'] = actual_interval_seconds
        details['actual_interval_str'] = str(actual_interval)
        details['first_timestamp'] = str(timestamps.iloc[0])
        details['last_timestamp'] = str(timestamps.iloc[-1])
        details['total_rows'] = len(timestamps)

        # Mit Training-Intervall vergleichen
        if 'time_config' in self._config:
            expected_interval = self._config['time_config'].get('interval_seconds')

            if expected_interval and abs(actual_interval_seconds - expected_interval) > 1:
                errors.append(
                    f"Zeitintervall-Mismatch: Training={expected_interval}s, "
                    f"Backtest={actual_interval_seconds}s"
                )

            # Training-Zeitraum info
            details['training_interval_seconds'] = expected_interval
            details['training_first'] = self._config['time_config'].get('first_timestamp')
            details['training_last'] = self._config['time_config'].get('last_timestamp')

        # Luecken erkennen
        expected_count = len(time_diffs)
        actual_matches = (time_diffs == actual_interval).sum()
        gap_count = expected_count - actual_matches

        if gap_count > 0:
            gap_percentage = (gap_count / expected_count) * 100
            details['gap_count'] = gap_count
            details['gap_percentage'] = gap_percentage

            if gap_percentage > 5:
                warnings.append(
                    f"Zeitluecken erkannt: {gap_count} ({gap_percentage:.1f}%) "
                    f"Intervalle weichen vom Standard ab"
                )
            elif gap_percentage > 20:
                errors.append(
                    f"Zu viele Zeitluecken: {gap_count} ({gap_percentage:.1f}%)"
                )

        # Duplikate pruefen
        duplicates = timestamps.duplicated().sum()
        if duplicates > 0:
            errors.append(f"Duplizierte Zeitstempel: {duplicates}")
            details['duplicate_count'] = duplicates

        # Sortierung pruefen
        if not timestamps.is_monotonic_increasing:
            errors.append("Zeitstempel nicht chronologisch sortiert!")

        return errors, warnings, details

    @staticmethod
    def create_time_config(data: pd.DataFrame, datetime_col: str = 'DateTime') -> Dict:
        """
        Erstellt Zeit-Konfiguration fuer Speicherung beim Training.

        Args:
            data: Training-DataFrame
            datetime_col: Name der DateTime-Spalte

        Returns:
            Dict mit Zeitkonfiguration
        """
        timestamps = pd.to_datetime(data[datetime_col])
        time_diffs = timestamps.diff().dropna()
        interval = time_diffs.mode().iloc[0] if len(time_diffs) > 0 else pd.Timedelta(hours=1)

        return {
            'interval_seconds': interval.total_seconds(),
            'interval_str': str(interval),
            'first_timestamp': str(timestamps.iloc[0]),
            'last_timestamp': str(timestamps.iloc[-1]),
            'row_count': len(timestamps),
        }

    def log_report(self, report: ValidationReport):
        """Loggt den Validierungsbericht."""
        if report.is_valid:
            self._logger.success("[Validator] Daten-Validierung erfolgreich")
            if report.fingerprint_match:
                self._logger.debug("[Validator] Pipeline-Fingerprint stimmt ueberein")
        else:
            self._logger.error("[Validator] Daten-Validierung FEHLGESCHLAGEN!")
            for error in report.errors:
                self._logger.error(f"[Validator] {error}")

        for warning in report.warnings:
            self._logger.warning(f"[Validator] {warning}")
```

### Integration

#### Beim Training speichern (prepare_data_window.py / training_window.py):

```python
# Nach erfolgreichem Training:
validation_config = {
    'features': pipeline.features,
    'normalizer_params': {
        'mean': pipeline.get_normalizer().mean_.tolist(),
        'std': pipeline.get_normalizer().std_.tolist(),
    },
    'sequence_length': sequence_length,
    'pipeline_hash': DataStreamValidator.create_fingerprint(pipeline),
    'training_timestamp': datetime.now().isoformat(),
    # NEU: Zeitindex-Konfiguration
    'time_config': DataStreamValidator.create_time_config(training_data),
}

# In model_info speichern
model_info['validation_config'] = validation_config
```

#### Beim Backtest pruefen (backtest_window.py):

```python
# Vor Inference:
if 'validation_config' in self.model_info:
    validator = DataStreamValidator(self.model_info['validation_config'])

    # 1. Pipeline-Validierung (Features, Normalizer, Hash)
    report = validator.validate_inference_data(pipeline, feature_data[:100])

    # 2. Zeitindex-Validierung (Intervall, Luecken, Sortierung)
    time_errors, time_warnings, time_details = validator.validate_time_index(self.data)
    report.errors.extend(time_errors)
    report.warnings.extend(time_warnings)
    report.details.update(time_details)
    report.is_valid = report.is_valid and len(time_errors) == 0

    validator.log_report(report)

    if not report.is_valid:
        # Optional: Abbruch oder Warnung
        self._log("WARNUNG: Daten-Validierung fehlgeschlagen!", 'WARNING')
        # raise ValueError("Daten-Inkonsistenz erkannt")
```

### Flowchart

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TRAINING (prepare_data_window)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
              ┌───────────────────────────────────────────┐
              │         FeatureProcessingPipeline          │
              │                                           │
              │  Features: [Open, SMA, RSI, ...]          │
              │  Normalizer: mean=[...], std=[...]        │
              │  NaN-Handling: remove rows                │
              └───────────────────────────────────────────┘
                                      │
                                      ▼
              ┌───────────────────────────────────────────┐
              │       DataStreamValidator.create_fingerprint()                │
              │                                           │
              │  Hash: "a1b2c3d4e5f6..."                  │
              └───────────────────────────────────────────┘
                                      │
                                      ▼
              ┌───────────────────────────────────────────┐
              │         Speichern in model_info           │
              │                                           │
              │  validation_config = {                    │
              │    features, normalizer_params,           │
              │    sequence_length, pipeline_hash         │
              │  }                                        │
              └───────────────────────────────────────────┘
                                      │
                                      │
════════════════════════════════════════════════════════════════════════════════
                              MODEL-DATEI (.pth)
════════════════════════════════════════════════════════════════════════════════
                                      │
                                      │
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INFERENCE (backtest_window)                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
              ┌───────────────────────────────────────────┐
              │         FeatureProcessingPipeline          │
              │         (mit geladenem Normalizer)         │
              └───────────────────────────────────────────┘
                                      │
                                      ▼
              ┌───────────────────────────────────────────┐
              │       DataStreamValidator.validate()       │
              │                                           │
              │  Prueft:                                  │
              │  [x] Feature-Namen identisch?             │
              │  [x] Feature-Reihenfolge identisch?       │
              │  [x] Normalizer mean/std identisch?       │
              │  [x] Pipeline-Hash identisch?             │
              │  [x] Daten-Shape korrekt?                 │
              │  [x] Zeitintervall identisch? (1h/15m/..) │
              │  [x] Keine Zeitluecken?                   │
              │  [x] Chronologisch sortiert?              │
              │  [x] Keine Duplikate?                     │
              └───────────────────────────────────────────┘
                                      │
                          ┌───────────┴───────────┐
                          │                       │
                          ▼                       ▼
                    ┌──────────┐           ┌──────────┐
                    │  VALID   │           │ INVALID  │
                    │    ✓     │           │    ✗     │
                    └──────────┘           └──────────┘
                          │                       │
                          ▼                       ▼
                    Backtest                Warnung/Abbruch
                    starten                 anzeigen
```

### Vorteile

| Aspekt | Ohne Validator | Mit Validator |
|--------|----------------|---------------|
| Feature-Reihenfolge | Stille Fehler | Sofort erkannt |
| Normalizer-Drift | Unbemerkt | Hash-Vergleich |
| Zeitintervall-Mismatch | Falsche Predictions | Sofort erkannt |
| Daten-Luecken | Unbemerkt | Warnung mit % |
| Duplikate | Verfaelschte Ergebnisse | Blockiert |
| Debug-Zeit | Stunden | Sekunden |
| Vertrauen | Unsicher | Bewiesen |

---

## Status

- [ ] Pipeline-Modul erstellen
- [ ] DataStreamValidator erstellen
- [ ] Unit-Tests
- [ ] prepare_data_window.py integrieren
- [ ] backtest_window.py integrieren
- [ ] walk_forward.py integrieren (optional)
- [ ] Dokumentation
- [ ] Commit
