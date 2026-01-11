# Plan: Labeling-System Ueberarbeitung mit Tab-Struktur

## Ziel
1. PrepareDataWindow mit Tabs strukturieren (Labeling | Sample-Generierung)
2. Neue Extrema-Erkennungsmethoden implementieren
3. Labels werden erst im Labeling-Tab ueber Start-Button erstellt

---

## Neue GUI-Struktur

```
+------------------------------------------------------------------+
|                  Trainingsdaten Vorbereitung                      |
+------------------------------------------------------------------+
|  [Tab: Labeling]  |  [Tab: Sample-Generierung]                   |
+------------------------------------------------------------------+
|                                                                   |
|  +-- Linke Spalte --+  +-- Mitte: Chart --+  +-- Rechts: Stats --+
|  |                  |  |                   |  |                   |
|  | Tab-spezifische  |  | Preis-Chart mit   |  | Label-Statistik   |
|  | Parameter        |  | BUY/SELL Markern  |  | Verteilung        |
|  |                  |  |                   |  |                   |
|  | [Start Button]   |  |                   |  |                   |
|  +------------------+  +-------------------+  +-------------------+
|                                                                   |
+------------------------------------------------------------------+
```

---

## Tab 1: Labeling

### Inhalt (linke Spalte)
- **Methoden-Auswahl** (Dropdown)
  - Future Return (default)
  - ZigZag
  - Peak Detection
  - Williams Fractals
  - Pivot Points
  - Tages-Extrema
  - Binary

- **Gemeinsame Parameter**
  - Lookforward: SpinBox (1-500, default 100)
  - Threshold %: SpinBox (0.1-20.0, default 2.0)

- **Methoden-spezifische Parameter** (dynamisch sichtbar)
  - ZigZag: zigzag_threshold (1-20%)
  - Peaks: prominence (0.1-5.0), distance (1-100)
  - Fractals: order (1-5)
  - Pivots: lookback (1-20)

- **Extrema-Offset** (fuer alle Extrema-Methoden ausser Future Return)
  - Offset Punkte: SpinBox (-50 bis +50, default 0)
  - Verschiebt gefundene Extrema um X Zeitpunkte
  - **Negativ** = frueher handeln (VOR dem Extremum)
  - **Positiv** = spaeter handeln (NACH dem Extremum)
  - Beispiel: Offset -3 bei einem Hoch → SELL-Signal 3 Bars VOR dem Hoch
  - Nutzen: Frueherer Einstieg um besseren Preis zu bekommen

---

### Auto-Label Modus

- **[x] Auto-Modus** Checkbox (default: aus)
- **Ziel-Anzahl Labels:** SpinBox (10-1000, default 100)
  - Gibt an wie viele BUY+SELL Signale generiert werden sollen

- **Funktionsweise:**
  1. System waehlt automatisch beste Methode (ZigZag oder Peaks)
  2. Passt Parameter iterativ an bis Zielanzahl erreicht
  3. Optimiert threshold/prominence bis `|gefunden - ziel| < 10%`

- **Algorithmus:**
  ```
  ziel = 100 Labels

  1. Starte mit ZigZag threshold=5%
  2. Generiere Labels, zaehle BUY+SELL
  3. Wenn zu wenig: threshold verringern
     Wenn zu viel: threshold erhoehen
  4. Binaeere Suche bis Ziel erreicht
  5. Wende Offset an (falls gesetzt)
  ```

- **Vorteile:**
  - Einfache Bedienung: Nur 1 Wert eingeben
  - Konsistente Datenmenge fuer Training
  - Automatische Parameter-Optimierung

---

- **[Labels generieren]** Button (gruen)
  - Erstellt Labels basierend auf Methode
  - Aktualisiert Chart mit Markern
  - Aktualisiert Statistik-Panel

### Workflow
1. Methode waehlen
2. Parameter einstellen
3. "Labels generieren" klicken
4. Vorschau im Chart pruefen
5. Weiter zu Tab 2

---

## Tab 2: Sample-Generierung

### Inhalt (linke Spalte)
- **Sequenz-Parameter**
  - Lookback: SpinBox mit +/- Buttons (default 100)

- **Feature-Auswahl**
  - Checkboxen: Close, High, Low, Open, PriceChange, PriceChangePct

- **HOLD-Samples**
  - Include HOLD: Checkbox
  - Ratio: Slider (0.1-3.0)
  - Auto/Manual Toggle

- **Normalisierung**
  - Dropdown: zscore, minmax, none

---

### Dataset-Split (Train / Validate / Test)

- **Visualisierung:** 3 Slider nebeneinander mit Live-Prozent-Anzeige

```
Train:    [====60%====]  60%
Validate: [===20%===]    20%
Test:     [===20%===]    20%
                        -----
                        100%
```

- **Parameter:**
  - `train_pct`: 10-90% (default 60%)
  - `val_pct`: 5-40% (default 20%)
  - `test_pct`: 5-40% (default 20%)
  - **Summe muss immer 100% ergeben**

- **Kopplung der Slider:**
  - Wenn Train geaendert → Val+Test proportional anpassen
  - Oder: Lock-Button pro Slider um Wert zu fixieren

- **Einfache Variante:** 2 Slider
  - Train %: Slider (50-90%, default 60%)
  - Test %: Slider (5-30%, default 20%)
  - Validate = 100% - Train - Test (automatisch berechnet)

- **Anzeige:**
  ```
  Train:    4200 Samples (60%)
  Validate: 1400 Samples (20%)
  Test:     1400 Samples (20%)
  ```

---

- **[Vorschau berechnen]** Button
- **[Daten generieren & Schliessen]** Button (erst aktiv nach Vorschau)

### Voraussetzung
- Labels muessen in Tab 1 generiert worden sein
- Sonst: Warnung "Bitte zuerst Labels generieren"

---

## Phase 1: Backend - Neue Labeling-Methoden

### 1.1 Enum und Konfiguration
**Datei:** `src/btcusd_analyzer/training/labeler.py`

```python
class LabelingMethod(Enum):
    FUTURE_RETURN = "future_return"
    ZIGZAG = "zigzag"
    PEAKS = "peaks"
    FRACTALS = "fractals"
    PIVOTS = "pivots"
    EXTREMA_DAILY = "extrema_daily"
    BINARY = "binary"

@dataclass
class LabelingConfig:
    method: LabelingMethod = LabelingMethod.FUTURE_RETURN
    lookforward: int = 100
    threshold_pct: float = 2.0
    zigzag_threshold: float = 5.0
    prominence: float = 0.5
    distance: int = 10
    fractal_order: int = 2
    pivot_lookback: int = 5
    extrema_offset: int = 0  # Verschiebung der Extrema in Zeitpunkten
    # Auto-Modus
    auto_mode: bool = False
    target_label_count: int = 100  # Zielanzahl BUY+SELL Labels
```

### 1.4 Auto-Label Funktion implementieren
```python
def auto_generate_labels(self, df: pd.DataFrame, target_count: int,
                         offset: int = 0) -> Tuple[np.ndarray, LabelingConfig]:
    """
    Generiert automatisch Labels basierend auf Zielanzahl.

    Passt ZigZag-Threshold iterativ an bis Zielanzahl erreicht.

    Args:
        df: DataFrame mit OHLCV-Daten
        target_count: Gewuenschte Anzahl BUY+SELL Labels
        offset: Extrema-Offset (optional)

    Returns:
        Tuple aus (labels, verwendete_config)
    """
    prices = df['Close'].values
    tolerance = 0.1  # 10% Toleranz

    # Binaere Suche fuer optimalen Threshold
    low, high = 0.5, 20.0
    best_labels = None
    best_config = None
    best_diff = float('inf')

    for _ in range(20):  # Max 20 Iterationen
        threshold = (low + high) / 2

        # Labels mit aktuellem Threshold generieren
        config = LabelingConfig(
            method=LabelingMethod.ZIGZAG,
            zigzag_threshold=threshold,
            extrema_offset=offset
        )
        labels = self._label_by_zigzag(df, config)

        # Offset anwenden
        if offset != 0:
            labels = self.apply_extrema_offset(labels, offset)

        # Zaehlen
        count = np.sum(labels != 0)  # BUY + SELL
        diff = abs(count - target_count)

        # Bestes Ergebnis speichern
        if diff < best_diff:
            best_diff = diff
            best_labels = labels.copy()
            best_config = config

        # Abbruch wenn Ziel erreicht
        if diff <= target_count * tolerance:
            break

        # Threshold anpassen
        if count < target_count:
            high = threshold  # Weniger Labels → niedrigerer Threshold
        else:
            low = threshold   # Mehr Labels → hoeherer Threshold

    self.logger.info(f"Auto-Label: {np.sum(best_labels != 0)} Labels "
                     f"(Ziel: {target_count}, Threshold: {best_config.zigzag_threshold:.2f}%)")

    return best_labels, best_config
```

### 1.3 Offset-Funktion implementieren
```python
def apply_extrema_offset(self, labels: np.ndarray, offset: int) -> np.ndarray:
    """
    Verschiebt gefundene Extrema-Labels um offset Zeitpunkte.

    Args:
        labels: Original-Labels (0=HOLD, 1=BUY, 2=SELL)
        offset: Verschiebung (-50 bis +50)
                Negativ = frueher (vor Extremum)
                Positiv = spaeter (nach Extremum)

    Returns:
        Verschobene Labels
    """
    if offset == 0:
        return labels

    n = len(labels)
    shifted = np.zeros(n, dtype=np.int64)  # Alle HOLD

    for i in range(n):
        if labels[i] != 0:  # BUY oder SELL
            new_idx = i + offset
            if 0 <= new_idx < n:
                shifted[new_idx] = labels[i]

    return shifted
```

### 1.2 Neue Methoden implementieren
- `find_zigzag_extrema()` - Richtungswechsel bei X% Bewegung
- `find_peaks_extrema()` - scipy.find_peaks mit Prominenz
- `find_williams_fractals()` - N-Bar Fractal Pattern
- `find_pivot_points()` - Lokale Max/Min im Fenster

---

## Phase 2: GUI-Umstrukturierung

### 2.1 Tab-Widget einfuegen
**Datei:** `src/btcusd_analyzer/gui/prepare_data_window.py`

Import hinzufuegen:
```python
from PyQt6.QtWidgets import QTabWidget
```

In `_create_param_panel()`:
```python
def _create_param_panel(self) -> QWidget:
    panel = QWidget()
    layout = QVBoxLayout(panel)

    # Tab Widget
    self.tab_widget = QTabWidget()
    self.tab_widget.setStyleSheet('''
        QTabWidget::pane { border: 1px solid #333; }
        QTabBar::tab {
            background: #333; color: white;
            padding: 10px 20px;
        }
        QTabBar::tab:selected { background: #4da8da; }
    ''')

    # Tab 1: Labeling
    labeling_tab = self._create_labeling_tab()
    self.tab_widget.addTab(labeling_tab, "Labeling")

    # Tab 2: Sample-Generierung
    samples_tab = self._create_samples_tab()
    self.tab_widget.addTab(samples_tab, "Samples")

    layout.addWidget(self.tab_widget)
    return panel
```

### 2.2 Labeling-Tab erstellen
```python
def _create_labeling_tab(self) -> QWidget:
    """Tab fuer Label-Generierung."""
    widget = QWidget()
    layout = QVBoxLayout(widget)

    # Methoden-Gruppe
    method_group = self._create_method_group()
    layout.addWidget(method_group)

    # Parameter-Gruppe (dynamisch)
    self.param_group = self._create_labeling_params_group()
    layout.addWidget(self.param_group)

    # Generate Button
    self.generate_labels_btn = QPushButton('Labels generieren')
    self.generate_labels_btn.setStyleSheet(self._button_style((0.2, 0.7, 0.3)))
    self.generate_labels_btn.clicked.connect(self._generate_labels)
    layout.addWidget(self.generate_labels_btn)

    # Status
    self.labels_status = QLabel('Keine Labels generiert')
    self.labels_status.setStyleSheet('color: #aaa;')
    layout.addWidget(self.labels_status)

    layout.addStretch()
    return widget
```

### 2.3 Samples-Tab erstellen
```python
def _create_samples_tab(self) -> QWidget:
    """Tab fuer Sample-Generierung."""
    widget = QWidget()
    layout = QVBoxLayout(widget)

    # Bestehende Gruppen verschieben:
    layout.addWidget(self._create_seq_group())      # Lookback
    layout.addWidget(self._create_feature_group())  # Features
    layout.addWidget(self._create_hold_group())     # HOLD-Samples
    layout.addWidget(self._create_norm_group())     # Normalisierung
    layout.addWidget(self._create_split_group())    # NEU: Train/Val/Test Split

    # Buttons
    self.preview_btn = QPushButton('Vorschau berechnen')
    self.preview_btn.clicked.connect(self._calculate_preview)
    layout.addWidget(self.preview_btn)

    self.generate_btn = QPushButton('Daten generieren & Schliessen')
    self.generate_btn.setEnabled(False)
    self.generate_btn.clicked.connect(self._generate_and_close)
    layout.addWidget(self.generate_btn)

    layout.addStretch()
    return widget
```

### 2.5 Dataset-Split Gruppe erstellen
```python
def _create_split_group(self) -> QGroupBox:
    """Erstellt die Train/Validate/Test Split Gruppe."""
    group = QGroupBox('Dataset-Split')
    group.setFont(QFont('Segoe UI', 12, QFont.Weight.Bold))
    layout = QGridLayout(group)

    # Train Slider
    layout.addWidget(QLabel('Train:'), 0, 0)
    self.train_slider = QSlider(Qt.Orientation.Horizontal)
    self.train_slider.setRange(50, 90)
    self.train_slider.setValue(60)
    self.train_slider.valueChanged.connect(self._update_split)
    layout.addWidget(self.train_slider, 0, 1)
    self.train_pct_label = QLabel('60%')
    self.train_pct_label.setStyleSheet('color: #33b34d; font-weight: bold;')
    layout.addWidget(self.train_pct_label, 0, 2)

    # Test Slider
    layout.addWidget(QLabel('Test:'), 1, 0)
    self.test_slider = QSlider(Qt.Orientation.Horizontal)
    self.test_slider.setRange(5, 30)
    self.test_slider.setValue(20)
    self.test_slider.valueChanged.connect(self._update_split)
    layout.addWidget(self.test_slider, 1, 1)
    self.test_pct_label = QLabel('20%')
    self.test_pct_label.setStyleSheet('color: #cc4d33; font-weight: bold;')
    layout.addWidget(self.test_pct_label, 1, 2)

    # Validate (automatisch berechnet)
    layout.addWidget(QLabel('Validate:'), 2, 0)
    self.val_pct_label = QLabel('20%')
    self.val_pct_label.setStyleSheet('color: #e6b333; font-weight: bold;')
    layout.addWidget(self.val_pct_label, 2, 1, 1, 2)

    # Sample-Anzahl Anzeige
    self.split_info = QLabel('Train: 0 | Val: 0 | Test: 0')
    self.split_info.setStyleSheet('color: #aaa; font-size: 10px;')
    layout.addWidget(self.split_info, 3, 0, 1, 3)

    return group

def _update_split(self):
    """Aktualisiert die Split-Anzeige."""
    train_pct = self.train_slider.value()
    test_pct = self.test_slider.value()
    val_pct = 100 - train_pct - test_pct

    # Validierung: Val darf nicht negativ werden
    if val_pct < 5:
        # Test reduzieren
        test_pct = 100 - train_pct - 5
        self.test_slider.blockSignals(True)
        self.test_slider.setValue(test_pct)
        self.test_slider.blockSignals(False)
        val_pct = 5

    self.train_pct_label.setText(f'{train_pct}%')
    self.test_pct_label.setText(f'{test_pct}%')
    self.val_pct_label.setText(f'{val_pct}%')

    # Sample-Anzahl berechnen (falls Labels vorhanden)
    if hasattr(self, 'labels') and self.labels is not None:
        total = np.sum(self.labels != 0)  # BUY + SELL
        train_n = int(total * train_pct / 100)
        val_n = int(total * val_pct / 100)
        test_n = total - train_n - val_n
        self.split_info.setText(f'Train: {train_n} | Val: {val_n} | Test: {test_n}')

    self.params['train_pct'] = train_pct
    self.params['val_pct'] = val_pct
    self.params['test_pct'] = test_pct
```

### 2.4 Label-Generierung separieren
```python
def _generate_labels(self):
    """Generiert Labels basierend auf aktueller Methode."""
    from ..training.labeler import DailyExtremaLabeler, LabelingConfig, LabelingMethod

    # Config aus GUI erstellen
    method_map = {
        0: LabelingMethod.FUTURE_RETURN,
        1: LabelingMethod.ZIGZAG,
        2: LabelingMethod.PEAKS,
        3: LabelingMethod.FRACTALS,
        4: LabelingMethod.PIVOTS,
        5: LabelingMethod.EXTREMA_DAILY,
        6: LabelingMethod.BINARY,
    }

    config = LabelingConfig(
        method=method_map[self.method_combo.currentIndex()],
        lookforward=self.lookforward_spin.value(),
        threshold_pct=self.threshold_spin.value(),
        # ... weitere Parameter
    )

    labeler = DailyExtremaLabeler()
    self.labels = labeler.generate_labels(self.data, config=config)
    self.labels_generated = True

    # UI aktualisieren
    self._update_chart_with_labels()
    self._update_stats_panel()
    self.labels_status.setText(f'Labels generiert: {len(self.labels)}')
    self.labels_status.setStyleSheet('color: #7f7;')
```

---

## Datenstruktur: training_data Dictionary

Das `training_data` Dictionary wird erweitert um Test-Daten:

```python
training_data = {
    # Bisherige Felder
    'X': np.ndarray,           # Alle Sequenzen
    'Y': np.ndarray,           # Alle Labels

    # NEU: Aufgeteilte Daten
    'X_train': np.ndarray,     # Training-Sequenzen
    'Y_train': np.ndarray,     # Training-Labels
    'X_val': np.ndarray,       # Validation-Sequenzen
    'Y_val': np.ndarray,       # Validation-Labels
    'X_test': np.ndarray,      # Test-Sequenzen
    'Y_test': np.ndarray,      # Test-Labels

    # Split-Info
    'split': {
        'train_pct': 60,
        'val_pct': 20,
        'test_pct': 20,
        'train_count': 4200,
        'val_count': 1400,
        'test_count': 1400
    },

    # NEU: Zeitliche Indizes fuer Backtester
    'time_indices': {
        'train_start': 0,
        'train_end': 5040,      # Index in Original-Daten
        'val_start': 5040,
        'val_end': 6720,
        'test_start': 6720,
        'test_end': 8400
    },

    # Original-Daten Referenz fuer Backtester
    'source_data': {
        'df_path': str,         # Pfad zur CSV-Datei
        'datetime_col': 'DateTime',
        'price_col': 'Close'
    }
}
```

---

## Walk-Forward Optimierung (Zukunft)

### Konzept

Walk-Forward ist ein adaptives Training, bei dem das Modell sich kontinuierlich an Marktveraenderungen anpasst:

```
Zeitachse: ════════════════════════════════════════════════════════►

Runde 1:  [████ TRAIN 4W ████][██ TEST 2W ██]
Runde 2:       [████ TRAIN 4W ████][██ TEST 2W ██]
Runde 3:            [████ TRAIN 4W ████][██ TEST 2W ██]
                                    ...
```

### Vorteile
- **Out-of-Sample Testing**: Test-Daten sind immer ungesehen
- **Markt-Adaption**: Reagiert auf Regime-Wechsel (Trend → Seitwaerts → Crash)
- **Realistisch**: Simuliert echten Live-Einsatz
- **Keine Overfitting-Gefahr**: Jeder Test ist wirklich "frisch"

### Warum kurze Trainings-Fenster?
- Lange Zeitreihen (1+ Jahr) glaetten Muster zu stark
- Marktverhalten aendert sich (Volatilitaet, Korrelationen)
- 4-6 Wochen Training fangen aktuelle Muster besser ein
- Modell "vergisst" veraltete Muster automatisch

### Parameter (Zukunft - WFO-Modus)

```python
@dataclass
class WalkForwardConfig:
    train_window: int = 672     # 4 Wochen bei H1 (24*7*4)
    test_window: int = 336      # 2 Wochen bei H1
    step_size: int = 168        # 1 Woche Verschiebung
    min_samples: int = 50       # Mindest-BUY/SELL pro Fenster

    # Automatischer Re-Train Trigger
    retrain_on_accuracy_drop: bool = True
    accuracy_threshold: float = 55.0  # Unter 55% → Re-Train
```

---

## FullAutoTest Modul (Zukunft)

### Konzept

Ein vollautomatischer Walk-Forward Test ueber einen langen Zeitraum (z.B. 1 Jahr).
Fuehrt den kompletten Zyklus mehrfach aus und aggregiert die Ergebnisse.

**Design-Prinzipien:**
- **Headless**: Keine GUI, keine Textausgaben waehrend der Ausfuehrung
- **Performance-First**: Optimiert auf maximale Geschwindigkeit
- **Parallelisierung**: Mehrere Runden gleichzeitig auf verschiedenen CPU-Kernen
- **Minimal I/O**: Nur finale Ergebnisse werden geschrieben

```
FullAutoTest: Rollierender 1-Jahres-Test
═══════════════════════════════════════════════════════════════════►

Runde 1:  [Load][Label][Train][Test] → Result 1
               ↓
Runde 2:       [Load][Label][Train][Test] → Result 2
                    ↓
Runde 3:            [Load][Label][Train][Test] → Result 3
                         ↓
...                      ...
                              ↓
Runde N:                      [Load][Label][Train][Test] → Result N
                                   ↓
                            [Aggregierte Ergebnisse]
                            [Report generieren]
```

### Ablauf pro Runde

```
1. Daten laden      → df[start:end] aus CSV
2. Extrema finden   → ZigZag/Peaks auf Train-Fenster
3. Labels erstellen → BUY/SELL/HOLD generieren
4. Samples bauen    → Sequenzen mit Lookback
5. Trainieren       → Modell auf Train-Daten
6. Testen           → Backtest auf Test-Daten
7. Ergebnis speichern → JSON/CSV append
8. Fenster verschieben → start += step_size
9. → Zurueck zu 1 (bis Datenende)
```

### Konfiguration

```python
@dataclass
class FullAutoTestConfig:
    """Konfiguration fuer vollautomatischen Walk-Forward Test."""

    # Datenquelle
    data_path: str                    # Pfad zur CSV-Datei
    symbol: str = 'BTCUSD'

    # Zeitfenster (in Bars/Stunden bei H1)
    train_window: int = 672           # 4 Wochen Training
    val_window: int = 168             # 1 Woche Validation
    test_window: int = 336            # 2 Wochen Test
    step_size: int = 168              # 1 Woche Verschiebung

    # Labeling
    labeling_method: str = 'zigzag'   # zigzag, peaks, future_return
    labeling_threshold: float = 3.0   # ZigZag Threshold %
    auto_label: bool = True           # Automatische Threshold-Anpassung
    target_labels: int = 50           # Ziel-Anzahl Labels pro Fenster

    # Features
    features: List[str] = field(default_factory=lambda: [
        'Open', 'High', 'Low', 'Close', 'PriceChange', 'PriceChangePct'
    ])
    normalization: str = 'zscore'     # zscore, minmax, none

    # Training
    model_type: str = 'bilstm'        # bilstm, lstm, gru, cnn_lstm
    hidden_size: int = 100
    num_layers: int = 2
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 0.001
    early_stopping_patience: int = 10

    # Backtest
    initial_capital: float = 10000.0
    commission: float = 0.001         # 0.1%
    slippage: float = 0.0005          # 0.05%

    # Output
    output_dir: str = 'results/full_auto_test'
    save_models: bool = False         # Modelle speichern? (Default: aus fuer Speed)

    # Parallelisierung
    n_workers: int = -1               # -1 = alle CPU-Kerne, 1 = sequentiell
    use_gpu: bool = True              # GPU fuer Training nutzen
```

### Hauptklasse (Headless + Parallel)

```python
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

class FullAutoTest:
    """
    Vollautomatischer Walk-Forward Test (Headless).

    Fuehrt den kompletten Zyklus (Load → Label → Train → Test)
    rollierend ueber einen Datensatz aus.

    Features:
    - Keine GUI/Textausgaben waehrend Lauf
    - Parallele Ausfuehrung auf mehreren CPU-Kernen
    - Ergebnisse nur am Ende geschrieben
    """

    def __init__(self, config: FullAutoTestConfig):
        self.config = config
        self.results: List[WalkForwardResult] = []

        # Worker-Anzahl bestimmen
        if config.n_workers == -1:
            self.n_workers = cpu_count()
        else:
            self.n_workers = max(1, config.n_workers)

    def run(self) -> FullAutoTestReport:
        """
        Startet den vollautomatischen Test.

        Returns:
            FullAutoTestReport mit aggregierten Ergebnissen
        """
        # Daten einmal laden (shared)
        df = self._load_data()
        total_bars = len(df)
        window_size = (self.config.train_window +
                       self.config.val_window +
                       self.config.test_window)

        # Alle Runden-Indizes berechnen
        rounds = []
        round_idx = 0
        while True:
            start_idx = round_idx * self.config.step_size
            end_idx = start_idx + window_size
            if end_idx > total_bars:
                break
            rounds.append((round_idx, start_idx, end_idx))
            round_idx += 1

        n_rounds = len(rounds)

        # Parallel oder sequentiell ausfuehren
        if self.n_workers > 1:
            results = self._run_parallel(df, rounds)
        else:
            results = self._run_sequential(df, rounds)

        # Nach round_idx sortieren (parallel kann unsortiert sein)
        self.results = sorted(results, key=lambda r: r.round_idx)

        # Report generieren und speichern
        report = self._generate_report()
        self._save_final_report(report)

        return report

    def _run_parallel(self, df: pd.DataFrame, rounds: list) -> List[WalkForwardResult]:
        """Fuehrt Runden parallel auf mehreren CPU-Kernen aus."""
        results = []

        # ProcessPoolExecutor fuer echte Parallelitaet (GIL-frei)
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Jobs erstellen - jeder bekommt Kopie der relevanten Daten
            futures = []
            for round_idx, start_idx, end_idx in rounds:
                window_df = df.iloc[start_idx:end_idx].copy()
                future = executor.submit(
                    _run_single_round_worker,  # Standalone-Funktion (pickle-faehig)
                    window_df,
                    round_idx,
                    self.config
                )
                futures.append(future)

            # Ergebnisse einsammeln
            for future in futures:
                result = future.result()
                results.append(result)

        return results

    def _run_sequential(self, df: pd.DataFrame, rounds: list) -> List[WalkForwardResult]:
        """Fuehrt Runden sequentiell aus (single-threaded)."""
        results = []
        for round_idx, start_idx, end_idx in rounds:
            window_df = df.iloc[start_idx:end_idx].copy()
            result = _run_single_round_worker(window_df, round_idx, self.config)
            results.append(result)
        return results


# Standalone Worker-Funktion (muss auf Top-Level sein fuer pickle)
def _run_single_round_worker(
    df: pd.DataFrame,
    round_idx: int,
    config: FullAutoTestConfig
) -> WalkForwardResult:
    """
    Fuehrt eine einzelne Runde aus (Worker-Funktion).

    Muss standalone sein fuer ProcessPoolExecutor (pickle).
    Keine Logging/Print-Ausgaben!
    """
    # 1. Split definieren
    train_end = config.train_window
    val_end = train_end + config.val_window
    test_end = val_end + config.test_window

    test_df = df.iloc[val_end:test_end]

    # 2. Labels generieren (nur auf Train+Val)
    labeler = DailyExtremaLabeler()
    labels = labeler.generate_labels(
        df.iloc[:val_end],
        method=config.labeling_method
    )

    # 3. Features + Sequenzen erstellen
    processor = FeatureProcessor(features=config.features)
    generator = SequenceGenerator(lookback=100)

    X, Y = generator.generate(
        processor.process(df.iloc[:val_end]),
        labels
    )

    # 4. Chronologischer Split
    split_idx = int(len(X) * (train_end / val_end))
    X_train, Y_train = X[:split_idx], Y[:split_idx]
    X_val, Y_val = X[split_idx:], Y[split_idx:]

    # 5. Modell erstellen + trainieren (silent)
    model = ModelFactory.create(
        config.model_type,
        input_size=len(config.features),
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_classes=3
    )

    trainer = ModelTrainer(model, verbose=False)  # Silent!
    history = trainer.train(
        X_train, Y_train, X_val, Y_val,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate
    )

    # 6. Test-Predictions
    test_labels = labeler.generate_labels(test_df, method=config.labeling_method)
    X_test, Y_test = generator.generate(processor.process(test_df), test_labels)
    predictions = model.predict(X_test)

    # 7. Backtest
    backtester = InternalBacktester()
    backtester.set_params(
        commission=config.commission,
        slippage=config.slippage
    )

    backtest_result = backtester.run(
        data=test_df,
        signals=pd.Series(predictions),
        initial_capital=config.initial_capital
    )

    return WalkForwardResult(
        round_idx=round_idx,
        train_start=df.iloc[0]['DateTime'],
        train_end=df.iloc[train_end]['DateTime'],
        test_start=df.iloc[val_end]['DateTime'],
        test_end=df.iloc[test_end - 1]['DateTime'],
        train_accuracy=max(history.val_accuracy) if history.val_accuracy else 0,
        test_accuracy=_calc_accuracy(predictions, Y_test),
        backtest_result=backtest_result,
        num_trades=backtest_result.num_trades,
        total_return=backtest_result.total_return_pct,
        win_rate=backtest_result.win_rate,
        max_drawdown=backtest_result.max_drawdown_pct
    )


def _calc_accuracy(predictions, labels) -> float:
    """Berechnet Accuracy (standalone fuer Worker)."""
    if len(predictions) == 0:
        return 0.0
    return (predictions == labels).sum() / len(labels) * 100
```

### Ergebnis-Datenstrukturen

```python
@dataclass
class WalkForwardResult:
    """Ergebnis einer einzelnen Walk-Forward Runde."""
    round_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_accuracy: float
    test_accuracy: float
    backtest_result: BacktestResult
    num_trades: int
    total_return: float
    win_rate: float
    max_drawdown: float


@dataclass
class FullAutoTestReport:
    """Aggregierter Report ueber alle Runden."""
    config: FullAutoTestConfig
    results: List[WalkForwardResult]

    # Aggregierte Metriken
    total_rounds: int
    avg_test_accuracy: float
    avg_return: float
    total_return: float           # Kumuliert ueber alle Runden
    avg_win_rate: float
    avg_drawdown: float
    max_drawdown: float           # Schlechteste Runde
    total_trades: int

    def summary(self) -> str:
        """Gibt Zusammenfassung als String."""
        return f"""
=== FullAutoTest Report ===
Runden: {self.total_rounds}
Zeitraum: {self.results[0].train_start} bis {self.results[-1].test_end}

Performance:
  Avg Test Accuracy: {self.avg_test_accuracy:.1f}%
  Total Return: {self.total_return:.1f}%
  Avg Return/Runde: {self.avg_return:.1f}%

Trading:
  Total Trades: {self.total_trades}
  Avg Win Rate: {self.avg_win_rate:.1f}%
  Avg Drawdown: {self.avg_drawdown:.1f}%
  Max Drawdown: {self.max_drawdown:.1f}%
"""

    def to_dataframe(self) -> pd.DataFrame:
        """Konvertiert Ergebnisse zu DataFrame."""
        return pd.DataFrame([
            {
                'round': r.round_idx,
                'train_start': r.train_start,
                'test_end': r.test_end,
                'train_acc': r.train_accuracy,
                'test_acc': r.test_accuracy,
                'return': r.total_return,
                'win_rate': r.win_rate,
                'trades': r.num_trades,
                'drawdown': r.max_drawdown
            }
            for r in self.results
        ])

    def save_csv(self, path: str):
        """Speichert Ergebnisse als CSV."""
        self.to_dataframe().to_csv(path, index=False)

    def save_json(self, path: str):
        """Speichert Report als JSON."""
        import json
        with open(path, 'w') as f:
            json.dump({
                'config': asdict(self.config),
                'summary': {
                    'total_rounds': self.total_rounds,
                    'avg_test_accuracy': self.avg_test_accuracy,
                    'total_return': self.total_return,
                    'avg_win_rate': self.avg_win_rate,
                    'max_drawdown': self.max_drawdown
                },
                'rounds': [asdict(r) for r in self.results]
            }, f, indent=2, default=str)
```

### CLI-Interface (optional)

```python
# Aufruf via Kommandozeile
# python -m btcusd_analyzer.full_auto_test --config config.yaml

def main():
    import argparse
    parser = argparse.ArgumentParser(description='FullAutoTest')
    parser.add_argument('--data', required=True, help='Pfad zur CSV')
    parser.add_argument('--train-weeks', type=int, default=4)
    parser.add_argument('--test-weeks', type=int, default=2)
    parser.add_argument('--step-weeks', type=int, default=1)
    parser.add_argument('--output', default='results/')
    args = parser.parse_args()

    config = FullAutoTestConfig(
        data_path=args.data,
        train_window=args.train_weeks * 168,  # H1 Bars
        test_window=args.test_weeks * 168,
        step_size=args.step_weeks * 168,
        output_dir=args.output
    )

    runner = FullAutoTest(config)
    report = runner.run()
    print(report.summary())
```

### Parallelisierungs-Strategie

```
Parallele Ausfuehrung auf 8-Kern CPU:

Daten:     [═══════════════════════════════════════════════]
                    (einmal geladen, shared)

Worker 1:  [Runde 1] ─────────────► Result 1
Worker 2:  [Runde 2] ─────────────► Result 2
Worker 3:  [Runde 3] ─────────────► Result 3
Worker 4:  [Runde 4] ─────────────► Result 4
Worker 5:  [Runde 5] ─────────────► Result 5
Worker 6:  [Runde 6] ─────────────► Result 6
Worker 7:  [Runde 7] ─────────────► Result 7
Worker 8:  [Runde 8] ─────────────► Result 8
                     │
                     ▼
           [Alle fertig? Naechste 8 Runden...]
                     │
                     ▼
           [Final Report generieren]
```

**Technische Details:**
- `ProcessPoolExecutor` fuer echte Parallelitaet (umgeht Python GIL)
- Jeder Worker bekommt Kopie des relevanten Datenfensters
- Kein Shared State zwischen Workern
- Worker-Funktion muss Top-Level sein (pickle-Requirement)
- GPU-Training: Ein Worker pro GPU (wenn `use_gpu=True`)

**Speicher-Optimierung:**
- Daten werden nur einmal geladen
- Jeder Worker bekommt nur sein Zeitfenster (nicht gesamte CSV)
- Modelle werden nicht gespeichert (default)
- Ergebnisse sind kompakt (nur Metriken, keine Trades-Details)

### Dateien fuer FullAutoTest

| Datei | Beschreibung |
|-------|--------------|
| `src/btcusd_analyzer/automation/full_auto_test.py` | Hauptklasse + Worker |
| `src/btcusd_analyzer/automation/config.py` | FullAutoTestConfig |
| `src/btcusd_analyzer/automation/results.py` | WalkForwardResult, Report |

### Datenfluss: Training → Backtest

```
PrepareDataWindow
       │
       ▼
[Labeling] → Labels generieren
       │
       ▼
[Samples] → Sequenzen + Split erstellen
       │
       ├──────────────────────────────┐
       ▼                              ▼
TrainingWindow                   Backtester
       │                              │
       ▼                              ▼
[Train auf X_train]          [Test auf X_test]
[Validate auf X_val]         [Simuliere Trades]
       │                              │
       ▼                              ▼
   Modell.pt                   Backtest-Report
                               (Equity-Kurve,
                                Win-Rate, etc.)
```

### Wichtig: Chronologischer Split

Der Split muss **chronologisch** erfolgen (keine Randomisierung!):

```
Gesamte Zeitreihe:
[═══════════════════════════════════════════════════]
 Jan                                              Dez

Nach Split:
[██████ TRAIN 60% ██████][█ VAL 20% █][█ TEST 20% █]
 Jan           Jul       Sep         Nov         Dez
```

**Warum chronologisch?**
- Verhindert "Zukunfts-Leak" (Modell sieht keine zukuenftigen Daten)
- Test-Daten repraesentieren echte Out-of-Sample Performance
- Backtester kann auf Test-Daten realistische Simulation durchfuehren

### Implementierung: Chronologischer Split

```python
def create_chronological_split(X: np.ndarray, Y: np.ndarray,
                               train_pct: float = 0.6,
                               val_pct: float = 0.2) -> dict:
    """
    Teilt Daten chronologisch (KEIN Shuffle!).

    Args:
        X: Sequenzen (n_samples, lookback, features)
        Y: Labels (n_samples,)
        train_pct: Anteil Training (0.0-1.0)
        val_pct: Anteil Validation (0.0-1.0)

    Returns:
        Dictionary mit aufgeteilten Daten
    """
    n = len(X)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    return {
        'X_train': X[:train_end],
        'Y_train': Y[:train_end],
        'X_val': X[train_end:val_end],
        'Y_val': Y[train_end:val_end],
        'X_test': X[val_end:],
        'Y_test': Y[val_end:],
        'time_indices': {
            'train_start': 0,
            'train_end': train_end,
            'val_start': train_end,
            'val_end': val_end,
            'test_start': val_end,
            'test_end': n
        }
    }
```

---

## Backtester-Integration (bestehende Implementierung)

### Vorhandene Klassen (aus PLAN.md portiert)

Die Backtesting-Engine ist bereits implementiert in `src/btcusd_analyzer/backtesting/`:

```
backtesting/
├── base.py           # BacktesterInterface, Trade, BacktestResult
├── backtester.py     # InternalBacktester
├── metrics.py        # PerformanceMetrics
└── adapters/
    ├── factory.py       # BacktesterFactory
    ├── vectorbt.py      # VectorBT Adapter
    ├── backtrader.py    # Backtrader Adapter
    └── backtestingpy.py # Backtesting.py Adapter
```

### BacktestResult Datenstruktur

```python
@dataclass
class BacktestResult:
    trades: List[Trade]              # Alle ausgefuehrten Trades
    equity_curve: pd.Series          # Equity ueber Zeit
    initial_capital: float = 10000.0

    # Metriken (automatisch berechnet)
    total_pnl: float                 # Gesamter P/L
    total_return_pct: float          # Return in %
    win_rate: float                  # Gewinnrate %
    profit_factor: float             # Gross Profit / Gross Loss
    max_drawdown: float              # Max Drawdown absolut
    max_drawdown_pct: float          # Max Drawdown %
    sharpe_ratio: float              # Risiko-adjustierte Rendite
    sortino_ratio: float             # Downside-Risk adjustiert
    num_trades: int                  # Anzahl Trades
    avg_trade_pnl: float             # Durchschnittlicher Trade
    avg_winner: float                # Durchschnittlicher Gewinner
    avg_loser: float                 # Durchschnittlicher Verlierer
```

### Trade Datenstruktur

```python
@dataclass
class Trade:
    entry_time: pd.Timestamp    # Einstiegszeitpunkt
    exit_time: pd.Timestamp     # Ausstiegszeitpunkt
    entry_price: float          # Einstiegspreis
    exit_price: float           # Ausstiegspreis
    position: str               # 'LONG' oder 'SHORT'
    size: float = 1.0           # Positionsgroesse
    pnl: float = 0.0            # Profit/Loss absolut
    pnl_pct: float = 0.0        # Profit/Loss %
    signal: str = ''            # Urspruengliches Signal
    commission: float = 0.0     # Gebuehren
```

### Verwendung mit Test-Daten

```python
from btcusd_analyzer.backtesting import InternalBacktester, BacktesterFactory

# Backtester erstellen
backtester = InternalBacktester()
backtester.set_params(
    commission=0.001,    # 0.1% Gebuehren
    slippage=0.0005,     # 0.05% Slippage
    stop_loss=0.05,      # 5% Stop-Loss
    take_profit=0.10     # 10% Take-Profit
)

# Test-Daten aus training_data verwenden
test_start = training_data['time_indices']['test_start']
test_end = training_data['time_indices']['test_end']

# Original-Preisdaten fuer Backtest (NICHT die Sequenzen!)
test_data = df.iloc[test_start:test_end].copy()

# Signale aus Modell-Predictions
# signals: 0=HOLD, 1=BUY, 2=SELL
signals = pd.Series(model_predictions, index=test_data.index)

# Backtest ausfuehren
result = backtester.run(
    data=test_data,
    signals=signals,
    initial_capital=10000.0
)

# Ergebnis anzeigen
print(result.summary())
```

### Walk-Forward im Backtester

Der InternalBacktester hat bereits `run_walk_forward()`:

```python
# Walk-Forward Analyse
results = backtester.run_walk_forward(
    data=df,
    signals=signals,
    initial_capital=10000.0,
    train_size=0.7,      # 70% Training (nicht im Backtest verwendet)
    test_size=0.3,       # 30% Test
    n_splits=5           # 5 Zeitfenster
)

# Ergebnisse aggregieren
total_return = sum(r.total_return_pct for r in results)
avg_win_rate = np.mean([r.win_rate for r in results])
```

### Metriken-Klasse (erweitert)

`PerformanceMetrics` bietet zusaetzliche Analysen:

```python
from btcusd_analyzer.backtesting.metrics import PerformanceMetrics

# Risiko-Metriken
sharpe = PerformanceMetrics.sharpe_ratio(returns)
sortino = PerformanceMetrics.sortino_ratio(returns)
calmar = PerformanceMetrics.calmar_ratio(returns)

# Drawdown-Analyse
max_dd = PerformanceMetrics.max_drawdown(equity_curve)
dd_duration = PerformanceMetrics.max_drawdown_duration(equity_curve)
dd_series = PerformanceMetrics.drawdown_series(equity_curve)

# Trade-Statistiken
win_rate = PerformanceMetrics.win_rate(pnls)
profit_factor = PerformanceMetrics.profit_factor(pnls)
expectancy = PerformanceMetrics.expectancy(pnls)
rr_ratio = PerformanceMetrics.risk_reward_ratio(pnls)

# Benchmark-Vergleich (z.B. Buy & Hold)
comparison = PerformanceMetrics.benchmark_comparison(
    strategy_returns=strategy_returns,
    benchmark_returns=buyhold_returns
)
# comparison['alpha'], comparison['beta'], comparison['outperformance']
```

### Integration Training → Backtest

Nach dem Training wird automatisch ein Backtest auf Test-Daten angeboten:

```
TrainingWindow
      │
      ▼
[Training abgeschlossen]
      │
      ▼
"Backtest auf Test-Daten starten?" [Ja] [Nein]
      │
      ▼ (Ja)
BacktestWindow oeffnen mit:
- model: trainiertes Modell
- test_data: X_test, Y_test
- source_data: Original-OHLCV fuer Preise
```

### GUI: Backtest-Report Anzeige

Nach Backtest wird ein Report angezeigt:

```
+------------------------------------------------------------------+
|                      Backtest Ergebnis                            |
+------------------------------------------------------------------+
| Trades: 47          | Win Rate: 63.8%    | Profit Factor: 1.85   |
| Total P/L: $2,847   | Return: 28.5%      | Max Drawdown: 8.2%    |
| Sharpe: 1.42        | Sortino: 2.15      | Calmar: 3.47          |
+------------------------------------------------------------------+
|                      Equity-Kurve                                 |
|  $13k ─────────────────────────────────╱──────                   |
|  $12k ─────────────────────────────╱───                          |
|  $11k ─────────────────────────╱───                              |
|  $10k ─────────────────────────                                  |
|       Jan    Feb    Mar    Apr    Mai    Jun                      |
+------------------------------------------------------------------+
| Trade-Liste                                                       |
| # | Entry      | Exit       | P/L     | Type | Duration          |
| 1 | 2025-01-15 | 2025-01-18 | +$234   | LONG | 3d                |
| 2 | 2025-01-20 | 2025-01-22 | -$89    | LONG | 2d                |
| ...                                                               |
+------------------------------------------------------------------+
```

---

## Betroffene Dateien

| Datei | Aenderung |
|-------|-----------|
| `src/btcusd_analyzer/training/labeler.py` | +250 Zeilen (Enum, Config, Methoden, Auto-Label) |
| `src/btcusd_analyzer/data/processor.py` | +60 Zeilen (Volumen + Zeit-Features) |
| `src/btcusd_analyzer/gui/prepare_data_window.py` | Refactoring (~500 Zeilen) - 3 Tabs |
| `src/btcusd_analyzer/gui/training_window.py` | +50 Zeilen (Backtest-Dialog nach Training) |
| `src/btcusd_analyzer/backtesting/backtester.py` | Vorhanden - ggf. kleine Anpassungen |
| `requirements.txt` | +1 Zeile (scipy) |

---

## Implementierungsreihenfolge

### Phase 1: Backend

1. **labeler.py** - Neue Labeling-Methoden
   - LabelingMethod Enum
   - LabelingConfig Dataclass
   - Neue Extrema-Methoden (ZigZag, Peaks, Fractals, Pivots)
   - Offset-Funktion (apply_extrema_offset)
   - Auto-Label Funktion (auto_generate_labels)
   - generate_labels() mit Config-Unterstuetzung refactoren

2. **processor.py** - Neue Features
   - Volumen-Features: _calc_volume, _calc_relativevolume
   - Zeit-Features: _calc_hour_sin, _calc_hour_cos

### Phase 2: GUI - PrepareDataWindow

3. **prepare_data_window.py** - Tab-Struktur
   - Tab-Widget mit 3 Tabs einfuegen
   - **Tab 1: Labeling**
     - Methoden-Dropdown
     - Parameter (lookforward, threshold, etc.)
     - Auto-Modus Checkbox + Zielanzahl
     - Offset SpinBox
     - "Labels generieren" Button
   - **Tab 2: Features**
     - Feature-Checkboxen (Preis, Volumen, Zeit)
     - Normalisierung Dropdown
     - Feature-Info Anzeige
   - **Tab 3: Samples**
     - Lookback SpinBox
     - HOLD-Samples Gruppe
     - Dataset-Split Slider (Train/Val/Test)
     - "Vorschau" + "Generieren" Buttons
   - Pipeline-Validierung (Invalidierung bei Upstream-Aenderung)

### Phase 3: Training + Backtest Integration

4. **training_window.py** - Backtest-Integration
   - training_data mit Test-Indizes speichern
   - Nach Training: "Backtest starten?" Dialog
   - Backtest auf X_test ausfuehren
   - BacktestResult anzeigen (Summary + Equity-Kurve)

5. **Backtest-Report Dialog** (optional neues Widget)
   - Metriken-Tabelle
   - Equity-Kurve Plot
   - Trade-Liste

### Phase 4: Integration + Test

6. **requirements.txt**
   - scipy hinzufuegen

7. **Verifikation**
   - Kompletter Workflow testen:
     1. Daten laden
     2. Labels generieren (Tab 1)
     3. Features waehlen (Tab 2)
     4. Samples + Split (Tab 3)
     5. Training starten
     6. Backtest auf Test-Daten
     7. Report pruefen

---

## Verifikation

1. **Tab-Navigation:** Zwischen Tabs wechseln
2. **Labeling-Tab:** Methode waehlen, Parameter einstellen, Labels generieren
3. **Chart:** BUY/SELL Marker erscheinen nach Label-Generierung
4. **Samples-Tab:** Warnung wenn keine Labels vorhanden
5. **Training:** Mit generierten Daten trainieren
