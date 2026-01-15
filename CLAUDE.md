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

## Git
- Commit-Messages auf Englisch
- Co-Authored-By Tag bei Commits
- Push nur auf explizite Anfrage

## Entwicklung
- Anwendung starten: `python -m btcusd_analyzer.main`
- Paket installieren: `pip install -e .`
- PyTorch mit CUDA 12.8: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`

## Wichtige Pfade
- **Log-Verzeichnis:** `C:\Work\MatLab\btc_analyzer_python\log`
  - Bei Aufforderung "Log pruefen" immer zuerst hier nachschauen

## Praeferenzen
- Keine Emojis im Code oder Kommentaren
- Kompakte Antworten bevorzugt
- Nur relevante Dateien modifizieren

## Threading/Logging - WICHTIG
**NIEMALS direktes Logging in QThread Worker-Threads verwenden!**

Python's `logging` Modul verwendet interne Locks die mit Qt's Event-Loop
zu sporadischen Deadlocks fuehren. Getestet und bestaetigt am 15.01.2026.

### Loesung fuer Worker-Threads:
1. **Callback-basiertes Logging:** Worker bekommt `log_callback` Parameter
2. **Qt Signal:** Worker sendet Log-Nachrichten via `pyqtSignal(str, str)`
3. **Main-Thread loggt:** Slot empfaengt Signal und ruft Logger auf

### Betroffene Module:
- `backtester/walk_forward.py` - verwendet `_CallbackLogger`
- `data/processor.py` - verwendet `_NullLogger` (nur DEBUG-Meldungen)
- `training/labeler.py` - verwendet `_NullLogger` (nur DEBUG-Meldungen)
- `gui/walk_forward_window.py` - `WalkForwardWorker.log_message` Signal

### Pattern fuer neue Worker:
```python
class MyWorker(QThread):
    log_message = pyqtSignal(str, str)  # level, message

    def run(self):
        # NICHT: logger.debug("...")
        # STATTDESSEN:
        self.log_message.emit("debug", "Nachricht")
```
