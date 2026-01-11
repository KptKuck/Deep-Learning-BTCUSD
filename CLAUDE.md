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
