# Optimierungsplan BTCUSD Analyzer

**Erstellt:** 2026-01-14
**Aktualisiert:** 2026-01-14
**Projektbewertung:** 6.6/10 (Gute Basis, benoetigt Fertigstellung)
**Gesamt-LOC:** ~19.600 Zeilen in 57 Python-Dateien

---

## Fortschritt

| Aufgabe | Status |
|---------|--------|
| VectorBT NotImplementedError dokumentiert | Erledigt |
| Config.load() implementiert | Erledigt |
| GUI TODOs (main_window.py) implementiert | Erledigt |
| GUI TODOs (backtest_window.py) implementiert | Erledigt |
| Test-Suite Grundstruktur erstellt | Erledigt |

**Naechste Schritte:**
- GUI-Refactoring (grosse Dateien aufteilen)
- Weitere Tests schreiben
- Input-Validierung hinzufuegen

---

## Zusammenfassung

Das Projekt zeigt eine **professionelle Architektur** mit klarer Modularitaet, aber mehrere Bereiche sind unvollstaendig. Die Hauptprobleme sind: grosse GUI-Dateien, fehlende Tests, unvollstaendige Adapter und TODO-Stellen im Code.

---

## Phase 1: Kritische Probleme (Hohe Prioritaet)

### 1.1 GUI-Refactoring

**Problem:** GUI-Dateien sind zu gross und verletzen das Single Responsibility Principle.

| Datei | Zeilen | Status |
|-------|--------|--------|
| prepare_data_window.py | 2.267 | Kritisch |
| main_window.py | 1.590 | Kritisch |
| training_window.py | 1.364 | Hoch |
| backtest_window.py | 991 | Mittel |

**Empfohlene Aktionen:**
- [ ] `prepare_data_window.py` in Komponenten aufteilen:
  - `PrepareDataWindow` (Hauptfenster, ~300 Zeilen)
  - `FeatureSelectionWidget` (Feature-Auswahl)
  - `LabelingConfigWidget` (Labeling-Konfiguration)
  - `DataPreviewWidget` (Datenvorschau)
  - `SequenceConfigWidget` (Sequenz-Einstellungen)
- [ ] `main_window.py` refactoren:
  - StatusBar als eigene Klasse
  - MenuBar-Logik separieren
  - Tab-Widgets in eigene Dateien auslagern
- [ ] `training_window.py` aufteilen:
  - `TrainingProgressWidget`
  - `MetricsDisplayWidget`
  - `ModelConfigWidget`

### 1.2 Unvollstaendige Implementierungen

**Problem:** 10+ TODO-Kommentare und NotImplementedError im Code.

**Kritische TODOs:**
- [x] `backtesting/adapters/vectorbt.py`: NotImplementedError dokumentiert (bewusste Designentscheidung - Optuna empfohlen)
- [x] `core/config.py`: `Config.load()` implementiert
- [x] `gui/backtest_window.py`: Model-Vorhersage implementiert
- [ ] `gui/trading_window.py`: Mehrere TODO-Stellen vervollstaendigen
- [x] `gui/main_window.py`: TODOs implementiert (_analyze_data, _load_training_data, _load_model, _load_last_model, _make_prediction, _save_parameters, _load_parameters)

**Empfohlene Aktionen:**
- [x] Alle TODO-Kommentare katalogisieren
- [x] Priorisierung nach Wichtigkeit
- [ ] Unbenoetigte Stubs entfernen oder dokumentieren

### 1.3 Input-Validierung

**Problem:** Datenverarbeitung kann bei fehlerhaften Eingaben still fehlschlagen.

**Betroffene Module:**
- `data/reader.py`: Keine Validierung von CSV-Struktur
- `data/processor.py`: Keine Pruefung auf fehlende Spalten
- `training/labeler.py`: Keine Validierung der Eingabedaten

**Empfohlene Aktionen:**
- [ ] Schema-Validierung in CSVReader einfuegen
- [ ] Spaltenpruefung in FeatureProcessor
- [ ] Pydantic-Models fuer Datenvalidierung evaluieren

---

## Phase 2: Code-Qualitaet (Mittlere Prioritaet)

### 2.1 Test-Suite erstellen

**Problem:** Keine Unit-Tests vorhanden (0% Coverage).

**Empfohlene Struktur:**
```
tests/
├── unit/
│   ├── test_reader.py
│   ├── test_processor.py
│   ├── test_labeler.py
│   ├── test_normalizer.py
│   └── test_models.py
├── integration/
│   ├── test_pipeline.py
│   ├── test_training.py
│   └── test_backtesting.py
└── conftest.py (Fixtures)
```

**Prioritaet fuer Tests:**
1. [x] `data/reader.py` - CSV-Parsing (tests/unit/test_reader.py)
2. [x] `data/processor.py` - Feature-Berechnung (tests/unit/test_processor.py)
3. [ ] `training/labeler.py` - Label-Generierung
4. [x] `training/normalizer.py` - Normalisierung (tests/unit/test_normalizer.py)
5. [x] `models/` - Model-Forward-Pass (tests/unit/test_models.py)
6. [x] Pipeline-Integration (tests/integration/test_pipeline.py)

**Test-Framework:** pytest mit pytest-cov (bereits konfiguriert in pyproject.toml)

### 2.2 Magic Numbers entfernen

**Problem:** Hardcodierte Werte ohne Erklaerung.

**Beispiele:**
```python
# processor.py - Parkinson Volatilitaet
parkinson_vol = np.sqrt(1 / (4 * np.log(2)) * ...)  # 0.361 ohne Kommentar

# labeler.py - verschiedene Schwellwerte
threshold = 0.02  # Woher kommt dieser Wert?
```

**Empfohlene Aktionen:**
- [ ] Konstanten als Modul-Level-Variablen definieren mit Docstring
- [ ] Konfigurierbare Werte in Config-Klassen verschieben
- [ ] Mathematische Formeln dokumentieren

### 2.3 Error-Handling verbessern

**Problem:** Generisches Exception-Handling statt spezifischer Fehler.

**Aktuell:**
```python
try:
    # operation
except Exception as e:
    logger.error(f"Fehler: {e}")
```

**Besser:**
```python
try:
    # operation
except FileNotFoundError as e:
    logger.error(f"Datei nicht gefunden: {e.filename}")
except pd.errors.ParserError as e:
    logger.error(f"CSV-Parser-Fehler: {e}")
except ValueError as e:
    logger.error(f"Ungueltige Werte: {e}")
```

**Empfohlene Aktionen:**
- [ ] Custom Exceptions definieren (`BTCAnalyzerError`, `DataValidationError`, etc.)
- [ ] Spezifische Exception-Handler implementieren
- [ ] Aussagekraeftige Fehlermeldungen mit Kontext

---

## Phase 3: Architektur-Verbesserungen (Niedrige Prioritaet)

### 3.1 Dependency Injection fuer Logger

**Problem:** Singleton-Logger erschwert Testing.

**Aktuell:**
```python
from btcusd_analyzer.core.logger import get_logger
logger = get_logger(__name__)
```

**Besser:**
```python
class MyClass:
    def __init__(self, logger: Logger | None = None):
        self.logger = logger or get_logger(__name__)
```

**Empfohlene Aktionen:**
- [ ] Logger als optionalen Parameter in Hauptklassen
- [ ] Mock-Logger fuer Tests ermoeglichen

### 3.2 Async I/O fuer GUI-Responsivitaet

**Problem:** Schwere Operationen blockieren GUI.

**Betroffene Bereiche:**
- CSV-Laden (grosse Dateien)
- Feature-Berechnung
- Model-Training (bereits mit QThread)
- Backtesting

**Empfohlene Aktionen:**
- [ ] QThreadPool fuer Hintergrundaufgaben
- [ ] Progress-Signale fuer alle langwierigen Operationen
- [ ] Abbruch-Mechanismus implementieren

### 3.3 Streaming fuer grosse Datensaetze

**Problem:** Gesamte CSV wird in Speicher geladen.

**Empfohlene Aktionen:**
- [ ] Chunked Reading mit `pd.read_csv(chunksize=...)`
- [ ] Generator-Pattern fuer Feature-Berechnung
- [ ] Memory-Mapping fuer sehr grosse Dateien evaluieren

---

## Phase 4: Fehlende Features (Roadmap)

### 4.1 Geplante Modelle (aus PLAN.md)

- [ ] TCN (Temporal Convolutional Network)
- [ ] Transformer-basierte Modelle
- [ ] Informer
- [ ] Temporal Fusion Transformer (TFT)
- [ ] N-BEATS

**Hinweis:** `pytorch-forecasting` ist als Dependency vorhanden aber nicht genutzt.

### 4.2 Backtesting-Adapter vervollstaendigen

| Adapter | Status | Aktion |
|---------|--------|--------|
| Internal | Fertig | - |
| VectorBT | NotImplemented | Implementieren oder entfernen |
| Backtrader | Unklar | Ueberpruefen und dokumentieren |
| Backtesting.py | Unklar | Ueberpruefen und dokumentieren |

### 4.3 Dokumentation erweitern

- [ ] README.md erstellen (Projektueberblick)
- [ ] API-Referenz (Sphinx oder mkdocs)
- [ ] Konfigurationsguide
- [ ] Training-Beispiele/Tutorials

---

## Technische Schulden

### Sofort beheben:
1. ~~VectorBT `NotImplementedError`~~ - Erledigt (dokumentiert als Designentscheidung)
2. ~~`Config.load()` TODO~~ - Erledigt
3. ~~GUI TODOs in main_window.py~~ - Erledigt

### Mittel bis langfristig:
1. Test-Suite aufbauen
2. GUI-Refactoring
3. Input-Validierung

### Nice-to-have:
1. Dependency Injection
2. Async I/O
3. Streaming fuer grosse Daten

---

## Metriken und Ziele

| Metrik | Vorher | Aktuell | Ziel |
|--------|--------|---------|------|
| Test Coverage | 0% | ~20% | 70%+ |
| Groesste Datei | 2.267 LOC | 2.267 LOC | <500 LOC |
| TODO-Kommentare | 10+ | 1 | 0 |
| NotImplementedError | 1 | 0* | 0 |
| Type Hint Coverage | 80% | 80% | 95% |
| Docstring Coverage | 90% | 90% | 95% |

*VectorBT optimize_params bleibt NotImplementedError (bewusste Entscheidung - Optuna empfohlen)

---

## Empfohlene Reihenfolge

```
1. Kritische TODOs und NotImplementedError beheben
   ↓
2. Test-Suite fuer Kernmodule erstellen
   ↓
3. GUI-Refactoring (groesste Dateien zuerst)
   ↓
4. Input-Validierung implementieren
   ↓
5. Fehlende Features nach Bedarf
```

---

## Werkzeuge und Konfiguration

**Empfohlene Tools:**
- pytest + pytest-cov (Testing)
- mypy (Type Checking)
- ruff (Linting, ersetzt flake8/isort/black)
- pre-commit (Git Hooks)

**pyproject.toml Ergaenzungen:**
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=src/btcusd_analyzer --cov-report=html"

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true
```

---

## Fazit

Das Projekt hat eine **solide Architektur** mit professionellen Design-Patterns. Die Hauptarbeit liegt in der **Fertigstellung** der begonnenen Features und der **Verbesserung der Wartbarkeit** durch GUI-Refactoring und Tests. Die Priorisierung sollte sein:

1. **Stabilisierung** (TODOs, Fehler beheben)
2. **Qualitaetssicherung** (Tests)
3. **Wartbarkeit** (Refactoring)
4. **Erweiterung** (neue Features)
