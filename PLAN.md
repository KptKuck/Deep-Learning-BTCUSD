# BTCUSD Analyzer - Python Port

## Uebersicht
Portierung des MATLAB BTCUSD Analyzers nach Python mit PyQt6 (GUI) und PyTorch (ML).

## Technologie-Stack
- **GUI:** PyQt6
- **ML Framework:** PyTorch
- **Hyperparameter-Optimierung:** Optuna
- **Trading API:** python-binance
- **Daten:** pandas, numpy

## Projektstruktur
```
btcusd_analyzer_python/
├── src/
│   └── btcusd_analyzer/
│       ├── __init__.py
│       ├── main.py                 # Einstiegspunkt
│       ├── core/
│       │   ├── config.py           # Konfiguration, Pfade
│       │   └── logger.py           # Logging-System
│       ├── data/
│       │   ├── downloader.py       # Binance API Download
│       │   ├── reader.py           # CSV Laden
│       │   └── processor.py        # Feature-Generierung
│       ├── training/
│       │   ├── labeler.py          # Daily Extrema, Labeling
│       │   ├── sequence.py         # Sequenz-Generierung
│       │   └── normalizer.py       # Z-Score Normalisierung
│       ├── models/
│       │   ├── base.py             # Basis-Klasse fuer alle Modelle
│       │   ├── lstm.py             # LSTM Netzwerk (unidirektional)
│       │   ├── bilstm.py           # BiLSTM Netzwerk
│       │   ├── gru.py              # GRU Netzwerk
│       │   ├── cnn.py              # 1D CNN
│       │   ├── cnn_lstm.py         # CNN-LSTM Hybrid
│       │   ├── tcn.py              # Temporal Convolutional Network
│       │   ├── transformer.py      # Transformer
│       │   ├── informer.py         # Informer (effizient fuer lange Sequenzen)
│       │   ├── tft.py              # Temporal Fusion Transformer
│       │   └── nbeats.py           # N-BEATS (interpretierbar)
│       ├── trainer/
│       │   ├── trainer.py          # Training-Loop
│       │   └── callbacks.py        # Early Stopping, etc.
│       ├── optimization/
│       │   └── optuna_tuner.py     # Hyperparameter-Suche
│       ├── backtesting/
│       │   ├── base.py             # Abstrakte Backtester-Schnittstelle
│       │   ├── backtester.py       # Eigene Backtest-Engine (Referenz-Impl.)
│       │   ├── metrics.py          # Performance-Metriken
│       │   └── adapters/           # Adapter fuer externe Frameworks
│       │       ├── __init__.py
│       │       ├── factory.py      # BacktesterFactory
│       │       ├── vectorbt.py     # VectorBT Adapter
│       │       ├── backtrader.py   # Backtrader Adapter
│       │       └── backtestingpy.py # Backtesting.py Adapter
│       ├── trading/
│       │   ├── live_trader.py      # Live-Trading Engine
│       │   ├── binance_client.py   # Binance API Wrapper (Live + Testnet)
│       │   └── api_config.py       # API-Key Verwaltung
│       ├── gui/
│       │   ├── main_window.py      # Haupt-GUI
│       │   ├── training_window.py  # Training-GUI
│       │   ├── backtest_window.py  # Backtest-GUI
│       │   ├── widgets/            # Wiederverwendbare Widgets
│       │   └── styles.py           # Dark Theme
│       ├── web/
│       │   ├── server.py           # Flask Webserver
│       │   ├── routes.py           # API Endpoints
│       │   └── templates/          # HTML Templates
│       │       ├── index.html      # Status-Dashboard
│       │       └── base.html       # Basis-Template
│       └── utils/
│           └── helpers.py          # Hilfsfunktionen
├── tests/
├── requirements.txt
├── requirements-backtesting.txt
└── pyproject.toml
```

## MATLAB -> Python Mapping

| MATLAB Datei | Python Modul | Beschreibung |
|--------------|--------------|--------------|
| btc_analyzer_gui.m | gui/main_window.py | Haupt-GUI |
| download_btc_data.m | data/downloader.py | Binance API |
| read_btc_data.m | data/reader.py | CSV Laden |
| prepare_training_data.m | training/sequence.py | Sequenzen |
| find_daily_extrema.m | training/labeler.py | Labels |
| train_bilstm_model.m | trainer/trainer.py | Training |
| train_gui.m | gui/training_window.py | Training-GUI |
| backtest_gui.m | gui/backtest_window.py | Backtest-GUI |

## Modell-Architekturen (11 Stueck)

| Modell | Typ | Staerken | Use Case | Factory-Name |
|--------|-----|----------|----------|--------------|
| LSTM | Rekurrent | Einfach, bewaehrt | Baseline | `lstm` |
| BiLSTM | Rekurrent | Vergangenheit + Zukunft | Klassifikation | `bilstm` |
| GRU | Rekurrent | Schneller als LSTM | Schnelles Training | `gru`, `bigru` |
| CNN | Konvolution | Lokale Muster | Feature-Extraktion | `cnn` |
| CNN-LSTM | Hybrid | Kombiniert Staerken | Komplexe Muster | `cnn_lstm` |
| TCN | Konvolution | Paralleles Training | Lange Sequenzen | `tcn` |
| Transformer | Attention | Globale Abhaengigkeiten | Komplexe Beziehungen | `transformer` |
| Informer | Attention | O(L log L) Effizienz | Sehr lange Sequenzen | `informer` |
| TFT | Attention | Interpretierbar, Multi-Horizon | Production-ready | `tft` |
| N-BEATS | Residual | Trend/Seasonality | Interpretierbarkeit | `nbeats`, `nbeats-g` |

### Verwendung
```python
from btcusd_analyzer.models import ModelFactory

# Verfuegbare Modelle
print(ModelFactory.available())

# Modell erstellen
model = ModelFactory.create('bilstm', input_size=6, hidden_size=100, num_classes=3)
```

## Backtesting-Abstraktionsschicht

### Verfuegbare Backtester

| Backtester | Framework | Staerken |
|------------|-----------|----------|
| `internal` | Eigen | Immer verfuegbar, Referenz |
| `vectorbt` | VectorBT | Schnellstes, vektorisiert |
| `backtrader` | Backtrader | Feature-reich, Community |
| `backtestingpy` | Backtesting.py | Einfach, Visualisierung |

### Verwendung
```python
from btcusd_analyzer.backtesting.adapters import BacktesterFactory

# Verfuegbare Backtester
print(BacktesterFactory.available())

# Backtester erstellen
backtester = BacktesterFactory.create('vectorbt', fees=0.001)

# Backtest ausfuehren
result = backtester.run(data, signals, initial_capital=10000)
print(result.summary())
```

## Live-Trading mit Testnet-Support

### Modus-Umschaltung
```python
from btcusd_analyzer.trading import BinanceClient, TradingMode

# Testnet (Standard, sicher)
client = BinanceClient(mode=TradingMode.TESTNET)

# Live (echtes Geld!)
client = BinanceClient(mode=TradingMode.LIVE)
```

### Visuelle Unterscheidung (GUI)

| Element | TESTNET (Demo) | LIVE (Echtes Geld) |
|---------|----------------|---------------------|
| Hintergrund | Dunkelgruen | Dunkelrot |
| Status-Banner | "TESTNET - DEMO MODUS" (gruen) | "LIVE TRADING - ECHTES GELD!" (rot) |
| Rahmenfarbe | Gruen | Rot |

### Sicherheits-Features
1. **Standard: Testnet** - App startet immer im Testnet-Modus
2. **Doppelte Bestaetigung** - Wechsel zu Live erfordert Dialog + Checkbox
3. **Visuelle Warnung** - Permanentes Banner, andere Farben
4. **Session-Lock** - Modus kann waehrend aktiver Position nicht gewechselt werden
5. **Logging** - Alle Trades werden mit Modus geloggt

## Web-Dashboard (LAN-Status-Server)

### Features
- Status-Dashboard fuer alle Hauptfunktionen
- Auto-Refresh alle 5 Sekunden
- Responsive Design (Desktop + Mobile)
- Start/Stop Kontrolle ueber GUI
- JSON API unter `/api/status`

### Verwendung
```python
from btcusd_analyzer.web import StatusServer, AppState

# Status-Server erstellen
app_state = AppState()
server = StatusServer(app_state, port=5000)

# Starten
server.start()
print(f"Dashboard: {server.get_url()}")

# Stoppen
server.stop()
```

## Hyperparameter-Optimierung (Optuna)

### Verwendung
```python
from btcusd_analyzer.optimization import OptunaTuner, TunerConfig

config = TunerConfig(
    n_trials=50,
    epochs_per_trial=30,
    direction='maximize',
    metric='val_accuracy'
)

tuner = OptunaTuner(train_loader, val_loader, model_type='bilstm', config=config)
tuner.use_default_search_space()
best_params = tuner.optimize()

# Bestes Modell erstellen
model = tuner.get_best_model()
```

## Implementierungs-Phasen

### Phase 1: Projektsetup & Daten [DONE]
- Projektstruktur erstellen
- requirements.txt, pyproject.toml
- Logger implementieren
- CSV Reader (pandas)
- Binance Downloader

### Phase 2: Training Pipeline [DONE]
- Labeling (Daily Extrema)
- Sequenz-Generierung
- Z-Score Normalisierung
- LSTM/BiLSTM Modelle
- Trainer mit Callbacks

### Phase 3: Weitere Architekturen (Basis) [DONE]
- GRU Modell
- CNN Modell (1D Convolutions)
- CNN-LSTM Hybrid
- Modell-Factory Pattern

### Phase 4: Fortgeschrittene Architekturen [DONE]
- TCN (Temporal Convolutional Network)
- Transformer (Self-Attention)
- Informer (effizient fuer lange Sequenzen)
- TFT (Temporal Fusion Transformer)
- N-BEATS (interpretierbar)

### Phase 5: Hyperparameter-Optimierung [DONE]
- Optuna Integration
- Suchraeume definieren
- Pruning fuer fruehes Stoppen

### Phase 6: Backtesting [DONE]
- Abstrakte Backtester-Schnittstelle
- Eigene Backtest-Engine
- Performance-Metriken
- VectorBT Adapter
- Backtrader Adapter
- Backtesting.py Adapter

### Phase 7: GUI [DONE]
- PyQt6 Haupt-Fenster
- Dark Theme
- Training-GUI
- Backtest-GUI

### Phase 8: Live-Trading [DONE]
- Binance Client (Live + Testnet)
- Live-Trader Engine
- Visuelle Modus-Unterscheidung

### Phase 9: Web-Dashboard [DONE]
- Flask Webserver
- Status-Dashboard (HTML)
- Auto-Refresh
- Start/Stop Kontrolle

## Dependencies

### Basis-Dependencies (requirements.txt)
```
torch>=2.0.0
PyQt6>=6.5.0
pandas>=2.0.0
numpy>=1.24.0
optuna>=3.0.0
python-binance>=1.0.0
matplotlib>=3.7.0
scikit-learn>=1.2.0
einops>=0.6.0
flask>=3.0.0
python-dotenv>=1.0.0
```

### Optionale Dependencies (requirements-backtesting.txt)
```
vectorbt>=0.26.0
backtrader>=1.9.78
backtesting>=0.3.3
```

### Installation
```bash
# Basis-Installation
pip install -r requirements.txt

# Mit allen Backtesting-Frameworks
pip install -r requirements.txt -r requirements-backtesting.txt

# Nur VectorBT (empfohlen)
pip install vectorbt
```

## Schnellstart

```bash
# 1. Projekt installieren
cd btcusd_analyzer_python
pip install -e .

# 2. GUI starten
python -m btcusd_analyzer.main

# 3. Oder: CLI verwenden
python -c "from btcusd_analyzer.models import ModelFactory; print(ModelFactory.available())"
```

## Verifikation
1. Unit-Tests fuer jedes Modul
2. CSV laden und Features generieren
3. Training starten, Loss-Kurve pruefen
4. Modell speichern/laden
5. Backtest durchfuehren
6. GUI-Interaktion testen
