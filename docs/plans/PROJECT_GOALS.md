# BTCUSD Analyzer - Projektziele

## Vision

Eine Desktop-Anwendung fuer automatisiertes BTC-Trading mit zwei Haupt-Workflows:

1. **Einzeltest-Workflow** - Manuelles Training und Evaluation eines Modells
2. **Automatik-Workflow** - Vollautomatisches Live-Trading mit rollierendem Retraining

---

## Die zwei Hauptziele

### Hauptziel 1: Backtester zur Modell-Evaluation

Ein Backtesting-System das sich in zwei Modi aufteilt:

#### A) Einzeltest (Manueller Workflow)

Schrittweise Durchfuehrung durch den Benutzer:

```
┌─────────────────────────────────────────────────────────────────┐
│                     EINZELTEST WORKFLOW                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. DATEN LADEN                                                 │
│     └── Historische Daten von Binance laden                     │
│                                                                 │
│  2. VERARBEITEN / VORBEREITEN                                   │
│     ├── Features berechnen (Indikatoren)                        │
│     ├── NaN-Zeilen entfernen                                    │
│     └── Normalisieren                                           │
│                                                                 │
│  3. LERNZIELE DEFINIEREN                                        │
│     ├── Peaks erkennen (Hochs/Tiefs)                            │
│     ├── Labels generieren (BUY/SELL/HOLD)                       │
│     └── Sequenzen erstellen                                     │
│                                                                 │
│  4. MODELL TRAINIEREN                                           │
│     ├── BILSTM / GRU / Transformer                              │
│     ├── Train/Validation Split                                  │
│     └── Callbacks (EarlyStopping, Checkpoint)                   │
│                                                                 │
│  5. BACKTEST                                                    │
│     ├── Out-of-Sample Daten                                     │
│     ├── Trades simulieren                                       │
│     └── Performance-Metriken berechnen                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### B) Walk-Forward Run (Kompletter Durchlauf)

Automatisierte Evaluation ueber mehrere Zeitfenster:

```
┌─────────────────────────────────────────────────────────────────┐
│                    WALK-FORWARD WORKFLOW                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Konfiguration:                                                 │
│  ├── Fenster-Typ: ANCHORED (wachsend) / ROLLING (fix)          │
│  ├── Window Size, Step Size                                     │
│  └── Purge/Embargo Gap                                          │
│                                                                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │ Fold 1  │  │ Fold 2  │  │ Fold 3  │  │ Fold N  │            │
│  │Train|OOS│→ │Train|OOS│→ │Train|OOS│→ │Train|OOS│            │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘            │
│                                                                 │
│  Pro Fold:                                                      │
│  ├── Daten vorbereiten                                          │
│  ├── Modell trainieren                                          │
│  ├── OOS Predictions                                            │
│  └── Metriken sammeln                                           │
│                                                                 │
│  Ergebnis:                                                      │
│  ├── Kombinierte OOS Equity Curve                               │
│  ├── Stabilitaets-Metriken                                      │
│  └── Overfitting-Analyse (Train vs OOS)                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### Hauptziel 2: Vollautomatik-Modus (Live Trading)

Langzeit-Betrieb mit rollierendem Retraining und Live-Trading:

#### Dateneingang: Live vs Simuliert

Im Live-Modus liefert Binance pro Zeiteinheit **eine einzelne OHLCV-Candle**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATENEINGANG                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LIVE-MODUS (Binance API):                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Timeframe: 1H oder 1M (waehlbar)                        │   │
│  │                                                          │   │
│  │  Pro Candle-Abschluss:                                   │   │
│  │  → Open, High, Low, Close, Volume                        │   │
│  │  → 1 Zeile alle 1H (oder 1M)                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                    │
│                            ▼                                    │
│  SIMULIERTER MODUS (CSV-Puffer):                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  CSV mit historischen Daten                              │   │
│  │                                                          │   │
│  │  Schrittweise Einspeisung:                               │   │
│  │  → 1 Zeile pro Tick (simuliert Live-Eingang)             │   │
│  │  → Exakt gleiches Format wie Live                        │   │
│  │  → Zeitverzoegerung optional (Echtzeit-Simulation)       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  WICHTIG: Beide Modi liefern identisches Datenformat!           │
│  → DateTime, Open, High, Low, Close, Volume                     │
│  → 1 Zeile pro Timeframe-Einheit                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Vollautomatik-Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                   VOLLAUTOMATIK WORKFLOW                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    CANDLE-EINGANG                         │  │
│  │                                                           │  │
│  │  Live:      Binance WebSocket/REST → 1 OHLCV pro Tick    │  │
│  │  Simuliert: CSV-Puffer → 1 Zeile pro Tick                │  │
│  │                                                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                    │
│                            ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                 VERARBEITUNGS-ZYKLUS                      │  │
│  │                 (Pro neue Candle)                         │  │
│  │                                                           │  │
│  │   ┌─────────┐    ┌─────────┐    ┌─────────┐              │  │
│  │   │FEATURE  │ →  │PREDICT  │ →  │ TRADE   │              │  │
│  │   │berechnen│    │Signal   │    │ausfuehren│             │  │
│  │   └─────────┘    └─────────┘    └─────────┘              │  │
│  │                                                           │  │
│  │   Lookback-Fenster: Letzte N Candles fuer Features        │  │
│  │                                                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                    │
│                            ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                 ROLLIERENDES RETRAINING                   │  │
│  │                 (Nach X Candles)                          │  │
│  │                                                           │  │
│  │   Trigger: Alle X Stunden/Tage oder bei Performance-Drop │  │
│  │   → Neue Labels aus echten Ergebnissen                    │  │
│  │   → Modell mit aktuellen Daten nachtrainieren             │  │
│  │                                                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                    │
│                            ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    ORDER EXECUTION                        │  │
│  │                                                           │  │
│  │   Live:      Binance API → Echte Orders                   │  │
│  │   Simuliert: Paper Trading → Virtuelle Orders             │  │
│  │                                                           │  │
│  │   Risk Management:                                        │  │
│  │   ├── Position Sizing                                     │  │
│  │   ├── Stop-Loss                                           │  │
│  │   └── Max Drawdown Limit                                  │  │
│  │                                                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Simulierter vs Live Modus - Vergleich

| Aspekt | Simuliert | Live |
|--------|-----------|------|
| **Datenquelle** | CSV-Datei | Binance API |
| **Einspeisung** | 1 Zeile pro Tick aus Puffer | 1 Candle pro Timeframe |
| **Format** | OHLCV (identisch) | OHLCV (identisch) |
| **Timeframe** | 1H / 1M (waehlbar) | 1H / 1M (waehlbar) |
| **Geschwindigkeit** | Schnell (kein Warten) | Echtzeit |
| **Orders** | Paper Trading (virtuell) | Echte Binance Orders |
| **Zweck** | Testen ohne Risiko | Produktiv handeln |

---

## Workflow-Vergleich

| Aspekt | Einzeltest | Walk-Forward | Vollautomatik |
|--------|------------|--------------|---------------|
| **Zweck** | Modell entwickeln | Modell evaluieren | Produktiv handeln |
| **Benutzer-Eingriff** | Manuell pro Schritt | Start, dann automatisch | Vollautomatisch |
| **Datenquelle** | Historisch (einmalig) | Historisch (Fenster) | Live + Puffer |
| **Training** | Einmal | Pro Fold | Rollierend |
| **Ergebnis** | Ein Modell | Stabilitaets-Analyse | Trades + Profit |
| **GUI-Fenster** | 2, 3, 4 | 4.2 | 6 |

---

## Status der Implementierung

### Einzeltest-Workflow
| Schritt | Status |
|---------|--------|
| Daten laden (Binance/CSV) | Fertig |
| Features berechnen | Fertig |
| NaN-Handling | Fertig (dupliziert) |
| Labels generieren | Fertig |
| Sequenzen erstellen | Fertig |
| Modell trainieren | Fertig |
| Backtest | Fertig |

### Walk-Forward-Workflow
| Schritt | Status |
|---------|--------|
| Anchored Window | Fertig |
| Rolling Window | Fertig |
| Fold-Metriken | Fertig |
| OOS Equity Curve | Fertig |

### Vollautomatik-Workflow
| Schritt | Status |
|---------|--------|
| Daten-Puffer (CSV schrittweise) | Geplant |
| Rollierendes Retraining | Geplant |
| Live Predictions | In Arbeit |
| Binance Order API | In Arbeit |
| Risk Management | In Arbeit |
| Simulierter Live-Modus | Geplant |

---

## Architektur-Ziele (Infrastruktur)

Diese Module unterstuetzen beide Hauptziele:

| Modul | Zweck | Status |
|-------|-------|--------|
| **FeatureProcessingPipeline** | Zentrale Feature-Verarbeitung fuer alle Workflows | Geplant |
| **DataStreamValidator** | Konsistenz Training/Backtest pruefen | Geplant |
| **SaveManager** | Sessions atomar speichern | Geplant |

---

## Non-Goals

| Non-Goal | Begruendung |
|----------|-------------|
| Multi-Asset Trading | Fokus auf BTCUSD |
| HFT (High Frequency) | Stunden-Candles, kein Millisekunden-Trading |
| Web-App | Desktop-fokussiert (PyQt6) |
| Cloud-Deployment | Lokale Ausfuehrung |
| Portfolio-Management | Einzelnes Asset |

---

## Technische Eckdaten

| Aspekt | Wert |
|--------|------|
| Sprache | Python 3.11+ |
| GUI | PyQt6 |
| ML Framework | PyTorch |
| Haupt-Modell | BILSTM |
| Datenquelle | Binance API / CSV |
| Zeitrahmen | 1h Candles (primaer) |
| Asset | BTCUSD |

---

## GUI-Fenster Zuordnung

```
Hauptziel 1: Backtester
├── 2 - PrepareData      (Einzeltest: Schritte 1-3)
├── 3 - Training         (Einzeltest: Schritt 4)
├── 4 - Backtest         (Einzeltest: Schritt 5)
└── 4.2 - Walk-Forward   (Kompletter Run)

Hauptziel 2: Vollautomatik
└── 6 - Trading          (Live + Simuliert)
```

---

## Erfolgskriterien

### Hauptziel 1: Backtester
- [ ] Einzeltest-Workflow vollstaendig durchfuehrbar
- [ ] Walk-Forward liefert zuverlaessige OOS-Metriken
- [ ] Keine Daten-Inkonsistenzen (Validator)
- [ ] Reproduzierbare Ergebnisse

### Hauptziel 2: Vollautomatik
- [ ] Simulierter Live-Modus mit CSV-Puffer funktioniert
- [ ] Rollierendes Retraining ohne Absturz
- [ ] Binance Orders werden korrekt ausgefuehrt
- [ ] Risk Management greift bei Verlusten
- [ ] Langzeit-Betrieb (Tage/Wochen) stabil

---

## Roadmap

```
Q1 2026: Robustheit (Infrastruktur)
├── FeatureProcessingPipeline
├── DataStreamValidator
└── SaveManager Refactoring

Q2 2026: Vollautomatik-Modus
├── Daten-Puffer (CSV schrittweise)
├── Simulierter Live-Modus
├── Rollierendes Retraining
└── Binance Order API

Q3 2026: Produktiv-Betrieb
├── Risk Management verfeinern
├── Monitoring / Alerting
└── Langzeit-Tests
```

---

## Dokumentation

| Datei | Inhalt |
|-------|--------|
| `docs/PROJECT_GOALS.md` | Dieses Dokument |
| `docs/plans/feature-pipeline-modul.md` | Pipeline + Validator Plan |
| `docs/plans/project-architecture-flowchart.md` | Architektur-Diagramme |
| `CLAUDE.md` | Technische Regeln fuer Entwicklung |
