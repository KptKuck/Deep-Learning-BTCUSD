# Claude Code Projekteinstellungen

## Projekt
- **Name:** BTCUSD Analyzer (Python)
- **Sprache:** Python 3.11+
- **Framework:** PyQt6, PyTorch
- **Zweck:** BILSTM Neural Network fuer BTC Trendwechsel-Erkennung

## Kommunikation
- **Sprache:** Deutsch
- **Commit-Messages:** Englisch
- **Kommentare im Code:** Deutsch

## Code-Style
- PEP 8 konform
- Type Hints verwenden
- Docstrings fuer alle oeffentlichen Funktionen/Klassen
- Variablennamen: snake_case
- Datenklassen: `@dataclass` oder Pydantic verwenden
  - Keine manuellen `__init__`, `__repr__`, `__eq__` Methoden
  - Ausnahme: Komplexe Initialisierungslogik (z.B. PyQt6 Widgets)

## Projektstruktur
```
btc_analyzer_python/
├── src/btcusd_analyzer/     # Hauptpaket
│   ├── core/                # Logger, Config
│   ├── data/                # CSV Reader, Processor
│   ├── gui/                 # PyQt6 GUI
│   ├── models/              # BILSTM Model
│   ├── trainer/             # Training Logic
│   └── utils/               # Hilfsfunktionen
├── data/                    # CSV-Daten (gitignored)
├── log/                     # Log-Dateien (gitignored)
├── models/                  # Trainierte Modelle (gitignored)
└── venv/                    # Virtual Environment (gitignored)
```

## GUI-Design
- Dunkles Theme (BackgroundColor: #262626)
- Farbschema aus MATLAB portiert:
  - Erfolg/BUY: #33b34d (gruen)
  - Warnung: #e6b333 (orange)
  - Fehler/SELL: #cc4d33 (rot)
  - Neutral/HOLD: #808080 (grau)

### Fenster-Indexierung (WICHTIG)
Alle GUI-Fenster muessen einen numerischen Index im Titel haben:
- Hauptfenster: ganze Zahlen (1, 2, 3, ...)
- Unter-Dialoge: Dezimalstellen (4.1, 4.2, ...)

Format: `setWindowTitle("X.X - Fenstername")`

### Log-Meldungen bei GUI-Oeffnung (WICHTIG)
Log-Meldungen beim Oeffnen von GUI-Fenstern muessen den Index enthalten:
```python
# RICHTIG:
self._log('Oeffne 3 - Training...', 'INFO')
self._log('Oeffne 4.2 - Walk-Forward...', 'INFO')

# FALSCH:
self._log('Oeffne Training GUI...', 'INFO')
self._log('Oeffne Walk-Forward Analyse...', 'INFO')
```
Format: `'Oeffne X.X - Fenstername...'`

### Relative Groessenangaben (WICHTIG)
Alle GUI-Groessenangaben muessen relativ zur Bildschirmgroesse sein (in %):
- GUIs sollen sich unabhaengig von der Bildschirmaufloesung skalieren
- Feste Pixelwerte nur fuer Minimum-Groessen verwenden

**Beispiel-Pattern:**
```python
from PyQt6.QtWidgets import QApplication

screen = QApplication.primaryScreen()
if screen:
    screen_rect = screen.availableGeometry()
    window_width = int(screen_rect.width() * 0.80)   # 80% der Breite
    window_height = int(screen_rect.height() * 0.95) # 95% der Hoehe
else:
    window_width, window_height = 1100, 900  # Fallback

self.setMinimumSize(800, 600)  # Absolute Minimum-Werte
self.resize(window_width, window_height)
```

## Git
- Commit-Messages auf Englisch
- Co-Authored-By Tag bei Commits
- Push nur auf explizite Anfrage

## Entwicklung
- Anwendung starten: `python -m btcusd_analyzer.main`
- Paket installieren: `pip install -e .`
- PyTorch mit CUDA 12.8: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`

## Edit-Befehle (WICHTIG)
Vor jedem `Edit` immer zuerst `Read` ausfuehren (nicht nur `Grep`),
da die Datei durch Linter/IDE zwischenzeitlich modifiziert werden kann.

## Wichtige Pfade
- **Log-Verzeichnis:** `C:\Work\MatLab\btc_analyzer_python\log`
  - Bei Aufforderung "Log pruefen" immer zuerst hier nachschauen

## Praeferenzen
- Keine Emojis im Code oder Kommentaren
- Kompakte Antworten bevorzugt
- Nur relevante Dateien modifizieren

## Logging
- Logger (`core/logger.py`) ist **Queue-basiert** und Thread-safe
- Format: `[Klasse/Funktion] Nachricht`
- Log-Level:
  - **DEBUG**: Detaillierte technische Informationen
    ```python
    self._logger.debug(f"[SessionManager] Session geladen: {session_id}")
    ```
  - **INFO**: Allgemeine Statusmeldungen
    ```python
    self._logger.info(f"[Training] Epoche {epoch} gestartet")
    ```
  - **WARNING**: Potentielle Probleme, Ausfuehrung wird fortgesetzt
    ```python
    self._logger.warning(f"[Processor] NaN-Werte gefunden: {count}")
    ```
  - **ERROR**: Fehler, die behandelt werden muessen
    ```python
    self._logger.error(f"[DataLoader] Datei nicht gefunden: {path}")
    ```
