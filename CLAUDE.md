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

**Aktuelle Zuordnung:**
| Index | Fenster | Datei |
|-------|---------|-------|
| 1 | Main | main_window.py |
| 1.1 | Session Manager | session_manager_window.py |
| 2 | Prepare Data | prepare_data_window.py |
| 3 | Training | training/training_window.py |
| 4 | Backtest | backtest/backtest_window.py |
| 4.1 | Backtrader | backtrader_window.py |
| 4.2 | Walk-Forward | walk_forward_window.py |
| 4.3 | Trade-Statistik | backtest/trade_statistics_dialog.py |
| 4.4 | Zeitraum | backtest/timerange_dialog.py |
| 5 | Visualize | visualize_data_window.py |
| 6 | Trading | trading_window.py |

Format: `setWindowTitle("X.X - Fenstername")`

## Git
- Commit-Messages auf Englisch
- Co-Authored-By Tag bei Commits
- Push nur auf explizite Anfrage

## Entwicklung
- Anwendung starten: `python -m btcusd_analyzer.main`
- Paket installieren: `pip install -e .`
- PyTorch mit CUDA 12.8: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`

## Bash-Befehle (WICHTIG)
Windows-Pfade immer in Anfuehrungszeichen setzen, KEIN `/d` Flag verwenden:
```bash
# RICHTIG:
cd "c:\Work\MatLab\btc_analyzer_python" && git status

# FALSCH (funktioniert nicht):
cd /d c:\Work\MatLab\btc_analyzer_python && git status
```

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

## Threading/Logging - WICHTIG
Der Logger (`core/logger.py`) ist jetzt **100% Queue-basiert** und Thread-safe:
- Log-Aufrufe schreiben nur in eine Queue (lock-free)
- Ein separater Daemon-Thread schreibt zu Console/Datei
- Kein Deadlock-Risiko mehr mit Qt Event-Loop

### Architektur:
```
[Main-Thread] --> logger.debug() --> Queue --> [LogWriter-Thread] --> Console/File
[Worker-Thread] --> logger.debug() --> Queue --> [LogWriter-Thread] --> Console/File
```

### Module die den Logger verwenden:
- `core/logger.py` - Queue-basierter Logger mit colorlog
- `backtester/walk_forward.py` - verwendet `_CallbackLogger` (fuer detaillierte Kontrolle)
- `data/processor.py` - verwendet `get_logger()` direkt (Thread-safe)
- `training/labeler.py` - verwendet `get_logger()` direkt (Thread-safe)
- `gui/walk_forward_window.py` - `WalkForwardWorker.log_message` Signal

### Pattern fuer neue Worker (optional, fuer feine Kontrolle):
```python
class MyWorker(QThread):
    log_message = pyqtSignal(str, str)  # level, message

    def run(self):
        # Option 1: Direkt loggen (Thread-safe dank Queue)
        from ..core.logger import get_logger
        get_logger().debug("Nachricht")

        # Option 2: Via Signal (fuer GUI-Updates)
        self.log_message.emit("debug", "Nachricht")
```
