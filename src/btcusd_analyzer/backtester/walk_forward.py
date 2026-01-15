"""
Walk-Forward Engine - Rolling Backtest mit verschiedenen Modi

Implementiert eine flexible Walk-Forward Analyse mit:
- Vier Modi: Inference Only, Inference+Live, Retrain/Split, Live-Simulation
- Anchored und Rolling Walk-Forward
- Purged/Embargo Gap zur Vermeidung von Look-Ahead Bias
- Integration mit Backtrader fuer realistische Simulation
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Tuple, Dict, Any, Callable
import copy
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..core.logger import get_logger
from ..data.processor import FeatureProcessor
from ..training.normalizer import ZScoreNormalizer
from ..training.sequence import SequenceGenerator, compute_class_weights
from ..training.labeler import DailyExtremaLabeler


logger = get_logger()


# =============================================================================
# Enums fuer Konfiguration
# =============================================================================

class BacktestMode(Enum):
    """Die vier verfuegbaren Backtest-Modi."""
    INFERENCE_ONLY = "inference_only"           # Session-Modell, Batch
    INFERENCE_LIVE = "inference_live"           # Session-Modell, Bar-by-Bar
    RETRAIN_PER_SPLIT = "retrain_per_split"     # Neutraining pro Split, Batch
    LIVE_SIMULATION = "live_simulation"         # Neutraining + Bar-by-Bar


class WalkForwardType(Enum):
    """Walk-Forward Typ."""
    ROLLING = "rolling"      # Train-Fenster verschiebt sich
    ANCHORED = "anchored"    # Train-Start fix, waechst


class BacktestEngine(Enum):
    """Backtest Engine Auswahl."""
    SIMPLE = "simple"          # Eigener schneller Backtester
    BACKTRADER = "backtrader"  # Professionelle Engine


class OptimizerType(Enum):
    """Optimizer-Typ fuer Training."""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"


class SchedulerType(Enum):
    """Learning Rate Scheduler Typ."""
    NONE = "none"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    STEP_LR = "step_lr"
    COSINE_ANNEALING = "cosine_annealing"
    ONE_CYCLE = "one_cycle"


class ClassWeightMode(Enum):
    """Class Weight Berechnung."""
    AUTO = "auto"           # Automatisch aus Klassenverteilung
    MANUAL = "manual"       # Manuell definiert
    NONE = "none"           # Keine Gewichtung


# =============================================================================
# Konfiguration Dataclass
# =============================================================================

@dataclass
class WalkForwardConfig:
    """
    Konfiguration fuer Walk-Forward Analyse.

    Enthaelt alle Parameter fuer:
    - Zeitraum und Fenster-Konfiguration
    - Modus-Auswahl (4 Modi)
    - Training-Parameter (Experten-Modus)
    - Backtest-Engine (Simple/Backtrader)
    - Erweiterte Features (Embargo Gap, Ensemble, etc.)
    """

    # ==========================================
    # ZEITRAUM
    # ==========================================
    start_date: Optional[datetime] = None  # None = Anfang der Daten
    end_date: Optional[datetime] = None    # None = Ende der Daten

    # ==========================================
    # FENSTER-KONFIGURATION
    # ==========================================
    train_size: int = 500                  # Anzahl Bars fuer Training
    test_size: int = 100                   # Anzahl Bars fuer Test
    step_size: int = 100                   # Schritt zwischen Splits

    # ==========================================
    # WALK-FORWARD TYP
    # ==========================================
    walk_forward_type: WalkForwardType = WalkForwardType.ROLLING

    # ==========================================
    # PURGED/EMBARGO GAP
    # ==========================================
    embargo_gap: int = 0                   # Bars zwischen Train/Test (0 = aus)
    purge_overlap: int = 0                 # Letzte N Bars aus Train entfernen

    # ==========================================
    # MODUS
    # ==========================================
    mode: BacktestMode = BacktestMode.INFERENCE_ONLY

    # ==========================================
    # BACKTEST ENGINE
    # ==========================================
    backtest_engine: BacktestEngine = BacktestEngine.SIMPLE
    initial_capital: float = 10000.0

    # Backtrader-spezifische Parameter
    bt_slippage_perc: float = 0.0005      # 0.05% Slippage
    bt_slippage_fixed: float = 0.0        # Oder fixer Betrag
    bt_commission_perc: float = 0.001     # 0.1% Commission
    bt_commission_fixed: float = 0.0      # Oder fix pro Trade
    bt_margin: float = 1.0                # 1.0 = kein Hebel
    bt_trade_on_close: bool = True        # Trade bei Bar-Close

    # ==========================================
    # TRAINING PARAMETER (fuer Retrain-Modi)
    # ==========================================
    epochs: int = 50                       # Max Epochen pro Split
    batch_size: int = 32                   # Batch-Groesse
    learning_rate: float = 0.001           # Initiale Lernrate
    validation_split: float = 0.2          # Train/Val Aufteilung

    # --- Early Stopping ---
    early_stopping: bool = True
    patience: int = 5
    min_delta: float = 0.001

    # --- Optimizer ---
    optimizer: OptimizerType = OptimizerType.ADAMW
    weight_decay: float = 0.01
    momentum: float = 0.9
    betas: Tuple[float, float] = (0.9, 0.999)

    # --- Learning Rate Scheduler ---
    scheduler: SchedulerType = SchedulerType.REDUCE_ON_PLATEAU
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
    scheduler_t_max: int = 50
    scheduler_eta_min: float = 1e-6

    # --- Gradient Clipping ---
    gradient_clipping: bool = True
    max_grad_norm: float = 1.0

    # --- Class Weights ---
    class_weight_mode: ClassWeightMode = ClassWeightMode.AUTO
    manual_class_weights: Optional[List[float]] = None

    # --- Modell-Architektur (optional ueberschreiben) ---
    hidden_size: Optional[int] = None
    num_layers: Optional[int] = None
    dropout: Optional[float] = None
    bidirectional: bool = True

    # --- Regularisierung ---
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0

    # ==========================================
    # ENSEMBLE
    # ==========================================
    create_ensemble: bool = False
    ensemble_top_n: int = 5
    ensemble_method: str = "voting"        # "voting" oder "averaging"

    # ==========================================
    # HYPERPARAMETER-VARIATION
    # ==========================================
    hyperparam_variation: bool = False
    variation_percent: float = 0.2
    vary_learning_rate: bool = True
    vary_dropout: bool = True
    vary_hidden_size: bool = False

    # ==========================================
    # AUSFUEHRUNG
    # ==========================================
    max_parallel: int = 0                  # 0 = Auto (GPU-basiert)
    timeout_per_split: int = 300           # 5 Min Timeout pro Split

    # ==========================================
    # REPRODUZIERBARKEIT
    # ==========================================
    random_seed: Optional[int] = 42


# =============================================================================
# Ergebnis Dataclasses
# =============================================================================

@dataclass
class TradeRecord:
    """Einzelner Trade mit allen Details."""
    trade_id: int
    split_index: int
    entry_time: datetime
    exit_time: Optional[datetime]
    signal: str                            # 'BUY' oder 'SELL'
    entry_price: float
    exit_price: Optional[float]
    size: float
    pnl: float
    pnl_pct: float
    duration: Optional[timedelta]
    prediction: int                        # Model-Prediction (0/1/2)
    confidence: float                      # Softmax-Confidence
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class EquityPoint:
    """Einzelner Punkt der Equity-Kurve."""
    datetime: datetime
    equity: float
    cash: float
    position_value: float
    position: str                          # 'LONG', 'SHORT', 'FLAT'
    drawdown: float
    drawdown_pct: float
    cumulative_return: float
    split_index: int


@dataclass
class SplitResult:
    """Ergebnis eines einzelnen Splits."""
    split_index: int
    train_range: Tuple[int, int]
    test_range: Tuple[int, int]

    # Klassifikations-Metriken
    accuracy: float
    f1_score: float
    precision: float = 0.0
    recall: float = 0.0

    # Trading-Metriken
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    num_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0

    # Training-Info (nur bei Retrain-Modi)
    training_epochs: Optional[int] = None
    best_val_accuracy: Optional[float] = None

    # Trade-Liste
    trades: List[TradeRecord] = field(default_factory=list)


@dataclass
class WalkForwardResult:
    """Gesamtergebnis der Walk-Forward Analyse."""
    config: WalkForwardConfig
    splits: List[SplitResult]
    mode: str
    timestamp: datetime

    # Aggregierte Metriken
    total_return: float = 0.0
    avg_return: float = 0.0
    std_return: float = 0.0
    avg_accuracy: float = 0.0
    std_accuracy: float = 0.0
    avg_sharpe: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    duration_sec: float = 0.0
    parallel_workers: int = 1

    # Detaillierte Daten fuer Excel-Export
    all_trades: List[TradeRecord] = field(default_factory=list)
    equity_curve: List[EquityPoint] = field(default_factory=list)

    # Robustheits-Metriken
    robustness_score: Optional[float] = None

    # Benchmark-Vergleich
    benchmark_return: Optional[float] = None  # Buy & Hold


# =============================================================================
# Rolling Normalizer (fuer Live-Simulation)
# =============================================================================

class RollingNormalizer:
    """
    Normalisiert nur basierend auf vergangenen Daten.
    Verhindert Look-Ahead Bias in der LIVE-Simulation.
    """

    def __init__(self, lookback: int = 100, epsilon: float = 1e-8):
        self.lookback = lookback
        self.epsilon = epsilon
        self.history: List[np.ndarray] = []

    def reset(self):
        """Zuruecksetzen fuer neuen Split."""
        self.history = []

    def update_and_normalize(self, new_values: np.ndarray) -> np.ndarray:
        """
        Fuegt neuen Wert hinzu und normalisiert NUR basierend auf Vergangenheit.

        Args:
            new_values: Feature-Vektor der aktuellen Bar [n_features]

        Returns:
            Normalisierter Feature-Vektor
        """
        # History aktualisieren
        self.history.append(new_values.copy())
        if len(self.history) > self.lookback:
            self.history.pop(0)

        # Nicht genug History? Erste Bars mit 0 normalisieren
        if len(self.history) < 2:
            return np.zeros_like(new_values)

        # Statistik NUR aus Vergangenheit berechnen!
        history_arr = np.array(self.history[:-1])  # Ohne aktuelle Bar
        mean = history_arr.mean(axis=0)
        std = history_arr.std(axis=0) + self.epsilon

        # Aktuelle Bar normalisieren
        return (new_values - mean) / std

    def get_stats(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Gibt aktuelle Mean/Std zurueck (fuer Debugging)."""
        if len(self.history) < 2:
            return None, None
        history_arr = np.array(self.history[:-1])
        return history_arr.mean(axis=0), history_arr.std(axis=0)


# =============================================================================
# Walk-Forward Engine
# =============================================================================

class WalkForwardEngine:
    """
    Hauptklasse fuer Walk-Forward Analyse.

    Unterstuetzt vier Modi:
    1. INFERENCE_ONLY: Session-Modell, Batch-Verarbeitung
    2. INFERENCE_LIVE: Session-Modell, Bar-by-Bar
    3. RETRAIN_PER_SPLIT: Neutraining pro Split, Batch
    4. LIVE_SIMULATION: Neutraining + Bar-by-Bar

    Verwendung:
        engine = WalkForwardEngine(data, model, model_info, config)
        result = engine.run(progress_callback=my_callback)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        model: nn.Module,
        model_info: Dict[str, Any],
        config: WalkForwardConfig
    ):
        """
        Initialisiert die Engine.

        Args:
            data: DataFrame mit OHLCV-Daten
            model: Trainiertes BILSTM-Modell (aus Session)
            model_info: Modell-Metadaten (features, lookback, etc.)
            config: Walk-Forward Konfiguration
        """
        logger.debug("[WalkForward] === Engine Initialisierung ===")
        logger.debug(f"[WalkForward] Daten: {len(data)} Zeilen, Spalten: {list(data.columns)}")
        logger.debug(f"[WalkForward] Model-Info: {model_info}")
        logger.debug(f"[WalkForward] Config: {config}")

        self.data = data.copy()
        self.model_info = model_info
        self.config = config

        # Aus model_info extrahieren
        self.features = model_info.get('features', [])
        self.lookback = model_info.get('lookback_size', 50)
        self.num_classes = model_info.get('num_classes', 3)

        logger.debug(f"[WalkForward] Features: {self.features}")
        logger.debug(f"[WalkForward] Lookback: {self.lookback}, Num Classes: {self.num_classes}")

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.debug(f"[WalkForward] Device: {self.device}")

        # Modell auf CPU kopieren (vermeidet CUDA-Threading-Probleme)
        # deepcopy erstellt eine unabhaengige Kopie des Modells
        logger.debug("[WalkForward] Kopiere Modell mit deepcopy...")
        self.model = copy.deepcopy(model)
        logger.debug("[WalkForward] Verschiebe Modell auf CPU...")
        self.model.to('cpu')
        logger.debug("[WalkForward] Modell erfolgreich auf CPU kopiert")

        # Abbruch-Flag
        self._cancel_requested = False
        self._lock = threading.Lock()

        # Processor und Generator
        logger.debug(f"[WalkForward] Erstelle FeatureProcessor mit {len(self.features)} Features")
        self.processor = FeatureProcessor(features=self.features)
        self.seq_gen = SequenceGenerator(
            lookback=self.lookback,
            lookforward=0,
            normalize=True
        )
        logger.debug("[WalkForward] === Engine Initialisierung abgeschlossen ===")

    def request_cancel(self):
        """Abbruch anfordern."""
        with self._lock:
            self._cancel_requested = True
        logger.warning("[WalkForward] Abbruch angefordert...")

    def is_cancelled(self) -> bool:
        """Pruefen ob Abbruch angefordert."""
        with self._lock:
            return self._cancel_requested

    def run(self, progress_callback: Optional[Callable] = None) -> WalkForwardResult:
        """
        Fuehrt die Walk-Forward Analyse aus.

        Args:
            progress_callback: Optional Callback(current, total, message)

        Returns:
            WalkForwardResult mit allen Ergebnissen
        """
        logger.debug("[WalkForward] === run() gestartet ===")
        start_time = time.time()
        self._cancel_requested = False

        # Reproduzierbarkeit
        if self.config.random_seed is not None:
            logger.debug(f"[WalkForward] Setze Random Seed: {self.config.random_seed}")
            torch.manual_seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)

        # Splits generieren
        logger.debug("[WalkForward] Generiere Splits...")
        splits = self.generate_splits()
        logger.info(f"[WalkForward] {len(splits)} Splits generiert")

        if len(splits) == 0:
            logger.error("[WalkForward] Keine Splits - zu wenig Daten")
            return self._create_empty_result(start_time)

        # Modus-spezifische Verarbeitung
        mode = self.config.mode
        logger.info(f"[WalkForward] Starte im Modus: {mode.value}")
        logger.debug(f"[WalkForward] Model auf Device: {next(self.model.parameters()).device}")

        if mode == BacktestMode.INFERENCE_ONLY:
            max_workers = self._calculate_max_parallel()
            results = self._run_inference_only(splits, max_workers, progress_callback)

        elif mode == BacktestMode.INFERENCE_LIVE:
            results = self._run_inference_live(splits, progress_callback)

        elif mode == BacktestMode.RETRAIN_PER_SPLIT:
            max_workers = self._calculate_max_parallel_training()
            results = self._run_retrain_parallel(splits, max_workers, progress_callback)

        elif mode == BacktestMode.LIVE_SIMULATION:
            results = self._run_live_simulation(splits, progress_callback)

        else:
            raise ValueError(f"Unbekannter Modus: {mode}")

        # Ergebnisse aggregieren
        duration = time.time() - start_time
        return self._aggregate_results(results, duration)

    def generate_splits(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Generiert Train/Test Splits basierend auf Konfiguration.

        Returns:
            Liste von Tuples: ((train_start, train_end), (test_start, test_end))
        """
        # Zeitraum filtern
        data = self._filter_date_range()
        n_samples = len(data)

        splits = []
        train_size = self.config.train_size
        test_size = self.config.test_size
        step_size = self.config.step_size
        gap = self.config.embargo_gap
        purge = self.config.purge_overlap

        if self.config.walk_forward_type == WalkForwardType.ROLLING:
            # Rolling: Train-Fenster verschiebt sich
            i = 0
            while i + train_size + gap + test_size <= n_samples:
                train_range = (i, i + train_size - purge)
                test_start = i + train_size + gap
                test_range = (test_start, test_start + test_size)
                splits.append((train_range, test_range))
                i += step_size

        else:  # ANCHORED
            # Anchored: Train-Start fix bei 0
            train_end = train_size
            while train_end + gap + test_size <= n_samples:
                train_range = (0, train_end - purge)
                test_start = train_end + gap
                test_range = (test_start, test_start + test_size)
                splits.append((train_range, test_range))
                train_end += step_size

        logger.debug(f"[WalkForward] {len(splits)} Splits generiert "
                    f"(Type: {self.config.walk_forward_type.value}, "
                    f"Train: {train_size}, Test: {test_size}, Step: {step_size})")

        return splits

    def _filter_date_range(self) -> pd.DataFrame:
        """Filtert Daten auf konfigurierten Zeitraum."""
        data = self.data.copy()

        # Datetime-Index sicherstellen
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'DateTime' in data.columns:
                data.set_index('DateTime', inplace=True)
            elif 'Date' in data.columns:
                data.set_index('Date', inplace=True)

        # Zeitraum filtern
        if self.config.start_date:
            data = data[data.index >= self.config.start_date]
        if self.config.end_date:
            data = data[data.index <= self.config.end_date]

        return data

    def _calculate_max_parallel(self) -> int:
        """Berechnet max parallele Workers fuer Inference."""
        if self.config.max_parallel > 0:
            return self.config.max_parallel

        # PyTorch GPU-Inference ist nicht threadsicher - daher sequenziell
        # Parallelisierung nur bei CPU oder expliziter Anforderung
        return 1

    def _calculate_max_parallel_training(self) -> int:
        """Berechnet max parallele Workers fuer Training."""
        if self.config.max_parallel > 0:
            return self.config.max_parallel

        if not torch.cuda.is_available():
            return 1

        total_mem = torch.cuda.get_device_properties(0).total_memory
        available = total_mem - (2 * 1024**3)  # 2GB Reserve
        per_split = 800 * 1024**2  # 800MB pro Split (Training)

        return min(8, max(1, int(available / per_split)))

    # =========================================================================
    # Modus 1: INFERENCE_ONLY
    # =========================================================================

    def _run_inference_only(
        self,
        splits: List[Tuple],
        max_workers: int,
        callback: Optional[Callable]
    ) -> List[SplitResult]:
        """
        INFERENCE ONLY: Session-Modell fuer alle Splits verwenden.
        Batch-Verarbeitung, schnellster Modus.
        """
        logger.info(f"[WalkForward] Inference Only mit {max_workers} Workers")
        logger.debug(f"[WalkForward] Device: {self.device}, CUDA verfuegbar: {torch.cuda.is_available()}")

        # CUDA synchronisieren falls GPU verwendet wird
        if torch.cuda.is_available():
            logger.debug("[WalkForward] CUDA sync vor Model-Transfer...")
            torch.cuda.synchronize()
            logger.debug("[WalkForward] CUDA sync abgeschlossen")

        # Modell auf GPU laden
        logger.debug(f"[WalkForward] Verschiebe Model auf {self.device}...")
        self.model.to(self.device)
        self.model.eval()
        logger.debug("[WalkForward] Model auf Device verschoben und in eval() Modus")

        # Nochmals synchronisieren nach Model-Transfer
        if torch.cuda.is_available():
            logger.debug("[WalkForward] CUDA sync nach Model-Transfer...")
            torch.cuda.synchronize()
            logger.debug("[WalkForward] CUDA sync abgeschlossen")

        results = []
        completed = 0

        # Sequenziell oder parallel
        logger.debug(f"[WalkForward] Starte {len(splits)} Splits (sequenziell: {max_workers <= 1})")
        if max_workers <= 1:
            for i, split in enumerate(splits):
                if self.is_cancelled():
                    logger.debug("[WalkForward] Abbruch erkannt, stoppe Verarbeitung")
                    break
                logger.debug(f"[WalkForward] === Starte Split {i} ===")
                result = self._process_inference_split(i, split)
                logger.debug(f"[WalkForward] Split {i} abgeschlossen: {result.num_trades} Trades")
                results.append(result)
                completed += 1
                if callback:
                    callback(completed, len(splits), f"Split {completed}/{len(splits)}")
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._process_inference_split, i, split): i
                    for i, split in enumerate(splits)
                }

                for future in as_completed(futures):
                    if self.is_cancelled():
                        break
                    split_idx = futures[future]
                    try:
                        result = future.result(timeout=self.config.timeout_per_split)
                        results.append(result)
                    except Exception as e:
                        logger.error(f"[WalkForward] Split {split_idx} fehlgeschlagen: {e}")

                    completed += 1
                    if callback:
                        callback(completed, len(splits), f"Split {completed}/{len(splits)}")

        # Nach Index sortieren
        results.sort(key=lambda r: r.split_index)
        return results

    def _process_inference_split(self, split_idx: int, split_info: Tuple) -> SplitResult:
        """Verarbeitet einen Split im Inference-Only Modus."""
        logger.debug(f"[Split {split_idx}] _process_inference_split gestartet")
        train_range, test_range = split_info
        logger.debug(f"[Split {split_idx}] Train: {train_range}, Test: {test_range}")

        # Test-Daten extrahieren
        logger.debug(f"[Split {split_idx}] Filtere Daten...")
        data = self._filter_date_range()
        test_data = data.iloc[test_range[0]:test_range[1]]
        logger.debug(f"[Split {split_idx}] Test-Daten: {len(test_data)} Zeilen, Spalten: {list(test_data.columns)}")

        # Features berechnen
        logger.debug(f"[Split {split_idx}] Berechne Features...")
        try:
            feature_df = self.processor.process(test_data)
            logger.debug(f"[Split {split_idx}] Features berechnet: {feature_df.shape}")
        except Exception as e:
            logger.error(f"[Split {split_idx}] Fehler bei Feature-Berechnung: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

        logger.debug(f"[Split {split_idx}] Extrahiere Feature-Matrix...")
        feature_matrix = self.processor.get_feature_matrix(feature_df)
        logger.debug(f"[Split {split_idx}] Feature-Matrix: {feature_matrix.shape}")

        # Normalisierung
        logger.debug(f"[Split {split_idx}] Normalisiere...")
        normalizer = ZScoreNormalizer()
        normalized = normalizer.fit_transform(feature_matrix)
        logger.debug(f"[Split {split_idx}] Normalisiert: {normalized.shape}")

        # Sequenzen erstellen
        logger.debug(f"[Split {split_idx}] Erstelle Sequenzen (Lookback: {self.lookback})...")
        sequences = []
        for i in range(self.lookback, len(normalized)):
            seq = normalized[i - self.lookback:i]
            sequences.append(seq)

        if len(sequences) == 0:
            logger.warning(f"[Split {split_idx}] Keine Sequenzen - zu wenig Daten")
            return self._create_empty_split_result(split_idx, train_range, test_range)

        sequences = np.array(sequences, dtype=np.float32)
        logger.debug(f"[Split {split_idx}] {len(sequences)} Sequenzen erstellt, Shape: {sequences.shape}")

        # Predictions
        logger.debug(f"[Split {split_idx}] Starte Inference auf {self.device}...")
        try:
            with torch.no_grad():
                x = torch.FloatTensor(sequences).to(self.device)
                logger.debug(f"[Split {split_idx}] Tensor erstellt: {x.shape}, Device: {x.device}")
                logits = self.model(x)
                logger.debug(f"[Split {split_idx}] Logits: {logits.shape}")
                probs = torch.softmax(logits, dim=1)
                predictions = logits.argmax(dim=1).cpu().numpy()
                confidences = probs.max(dim=1).values.cpu().numpy()
                logger.debug(f"[Split {split_idx}] Predictions: {len(predictions)}, Unique: {np.unique(predictions)}")
        except Exception as e:
            logger.error(f"[Split {split_idx}] Fehler bei Inference: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

        # Metriken berechnen
        logger.debug(f"[Split {split_idx}] Berechne Metriken...")
        result = self._calculate_split_metrics(
            split_idx, train_range, test_range,
            predictions, confidences, test_data
        )
        logger.debug(f"[Split {split_idx}] Metriken berechnet: Return={result.total_return:.2%}, Trades={result.num_trades}")
        return result

    # =========================================================================
    # Modus 2: INFERENCE_LIVE
    # =========================================================================

    def _run_inference_live(
        self,
        splits: List[Tuple],
        callback: Optional[Callable]
    ) -> List[SplitResult]:
        """
        INFERENCE + LIVE: Session-Modell mit Bar-by-Bar Verarbeitung.
        Rolling Normalisierung verhindert Look-Ahead Bias.
        """
        logger.info("[WalkForward] Inference + Live-Simulation")

        # CUDA synchronisieren falls GPU verwendet wird
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.model.to(self.device)
        self.model.eval()

        # Nochmals synchronisieren nach Model-Transfer
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        results = []
        for i, split in enumerate(splits):
            if self.is_cancelled():
                break

            result = self._process_inference_live_split(i, split)
            results.append(result)

            if callback:
                callback(i + 1, len(splits), f"Split {i + 1}/{len(splits)} (Live)")

        return results

    def _process_inference_live_split(self, split_idx: int, split_info: Tuple) -> SplitResult:
        """Verarbeitet einen Split mit Bar-by-Bar Live-Simulation."""
        train_range, test_range = split_info

        data = self._filter_date_range()
        test_data = data.iloc[test_range[0]:test_range[1]]

        # Rolling Normalizer
        rolling_norm = RollingNormalizer(lookback=self.lookback)

        predictions = []
        confidences = []
        feature_buffer = []

        for i in range(len(test_data)):
            # Neue Bar "empfangen"
            new_bar = test_data.iloc[i:i + 1]

            # Buffer aktualisieren
            feature_buffer.append(new_bar)
            if len(feature_buffer) > self.lookback:
                feature_buffer.pop(0)

            # Genuegend Daten?
            if len(feature_buffer) < self.lookback:
                predictions.append(1)  # HOLD
                confidences.append(0.0)
                continue

            # Features berechnen
            buffer_df = pd.concat(feature_buffer, ignore_index=True)
            feature_df = self.processor.process(buffer_df)
            feature_matrix = self.processor.get_feature_matrix(feature_df)

            # Rolling Normalisierung
            raw_sequence = feature_matrix[-self.lookback:]
            normalized_sequence = []
            for row in raw_sequence:
                norm_row = rolling_norm.update_and_normalize(row)
                normalized_sequence.append(norm_row)
            sequence = np.array(normalized_sequence, dtype=np.float32)

            # Prediction
            with torch.no_grad():
                x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)
                pred = logits.argmax(dim=1).item()
                conf = probs.max(dim=1).values.item()

            predictions.append(pred)
            confidences.append(conf)

        predictions = np.array(predictions)
        confidences = np.array(confidences)

        return self._calculate_split_metrics(
            split_idx, train_range, test_range,
            predictions, confidences, test_data
        )

    # =========================================================================
    # Modus 3: RETRAIN_PER_SPLIT
    # =========================================================================

    def _run_retrain_parallel(
        self,
        splits: List[Tuple],
        max_workers: int,
        callback: Optional[Callable]
    ) -> List[SplitResult]:
        """
        RETRAIN PER SPLIT: Neutraining pro Split, Batch-Inference.
        """
        logger.info(f"[WalkForward] Retrain per Split mit {max_workers} Workers")

        results = []
        completed = 0

        # Bei Training immer sequenziell um GPU-Konflikte zu vermeiden
        for i, split in enumerate(splits):
            if self.is_cancelled():
                break

            result = self._process_retrain_split(i, split)
            results.append(result)
            completed += 1

            if callback:
                callback(completed, len(splits),
                        f"Split {completed}/{len(splits)} (Train: {result.training_epochs} Ep)")

        return results

    def _process_retrain_split(self, split_idx: int, split_info: Tuple) -> SplitResult:
        """Trainiert neues Modell und fuehrt Inference durch."""
        train_range, test_range = split_info

        data = self._filter_date_range()
        train_data = data.iloc[train_range[0]:train_range[1]]
        test_data = data.iloc[test_range[0]:test_range[1]]

        # Optional: Hyperparameter-Variation
        config = self._get_varied_config(split_idx)

        # Neues Modell trainieren
        model, training_info = self._train_model_for_split(train_data, config)

        # CUDA synchronisieren vor Model-Transfer
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Inference auf Test-Daten
        model.to(self.device)
        model.eval()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Features und Sequenzen
        feature_df = self.processor.process(test_data)
        feature_matrix = self.processor.get_feature_matrix(feature_df)

        normalizer = ZScoreNormalizer()
        normalized = normalizer.fit_transform(feature_matrix)

        sequences = []
        for i in range(self.lookback, len(normalized)):
            seq = normalized[i - self.lookback:i]
            sequences.append(seq)

        if len(sequences) == 0:
            result = self._create_empty_split_result(split_idx, train_range, test_range)
            result.training_epochs = training_info.get('epochs', 0)
            result.best_val_accuracy = training_info.get('best_val_acc', 0.0)
            return result

        sequences = np.array(sequences, dtype=np.float32)

        with torch.no_grad():
            x = torch.FloatTensor(sequences).to(self.device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1).cpu().numpy()
            confidences = probs.max(dim=1).values.cpu().numpy()

        result = self._calculate_split_metrics(
            split_idx, train_range, test_range,
            predictions, confidences, test_data
        )
        result.training_epochs = training_info.get('epochs', 0)
        result.best_val_accuracy = training_info.get('best_val_acc', 0.0)

        # GPU Memory freigeben
        del model
        torch.cuda.empty_cache()

        return result

    def _train_model_for_split(
        self,
        train_data: pd.DataFrame,
        config: WalkForwardConfig
    ) -> Tuple[nn.Module, Dict]:
        """Trainiert ein neues Modell fuer einen Split."""
        from ..models import BiLSTMClassifier

        # Labels generieren
        labeler = DailyExtremaLabeler(
            method=self.model_info.get('label_method', 'future_return')
        )
        labels = labeler.generate(train_data)

        # Features
        feature_df = self.processor.process(train_data)
        feature_matrix = self.processor.get_feature_matrix(feature_df)

        # Sequenzen
        seq_gen = SequenceGenerator(
            lookback=self.lookback,
            lookforward=0,
            normalize=True
        )
        X, y = seq_gen.generate(feature_matrix, labels)

        if len(X) == 0:
            # Leeres Modell zurueckgeben
            model = BiLSTMClassifier(
                input_size=len(self.features),
                hidden_size=config.hidden_size or self.model_info.get('hidden_size', 100),
                num_layers=config.num_layers or self.model_info.get('num_layers', 2),
                num_classes=self.num_classes,
                dropout=config.dropout or self.model_info.get('dropout', 0.2),
                bidirectional=config.bidirectional
            )
            return model, {'epochs': 0, 'best_val_acc': 0.0}

        # Train/Val Split
        split_idx = int(len(X) * (1 - config.validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Class Weights
        if config.class_weight_mode == ClassWeightMode.AUTO:
            class_weights = compute_class_weights(y_train, num_classes=self.num_classes)
            class_weights = class_weights.to(self.device)
        elif config.class_weight_mode == ClassWeightMode.MANUAL and config.manual_class_weights:
            class_weights = torch.FloatTensor(config.manual_class_weights).to(self.device)
        else:
            class_weights = None

        # DataLoader
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
            batch_size=config.batch_size,
            pin_memory=True
        )

        # Modell erstellen
        model = BiLSTMClassifier(
            input_size=len(self.features),
            hidden_size=config.hidden_size or self.model_info.get('hidden_size', 100),
            num_layers=config.num_layers or self.model_info.get('num_layers', 2),
            num_classes=self.num_classes,
            dropout=config.dropout or self.model_info.get('dropout', 0.2),
            bidirectional=config.bidirectional
        ).to(self.device)

        # Optimizer
        optimizer = self._create_optimizer(model, config)

        # Scheduler
        scheduler = self._create_scheduler(optimizer, config)

        # Loss
        if config.label_smoothing > 0:
            criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=config.label_smoothing
            )
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Training Loop
        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0

        for epoch in range(config.epochs):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()

                if config.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                optimizer.step()

            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    logits = model(batch_x)
                    preds = logits.argmax(dim=1)
                    correct += (preds == batch_y).sum().item()
                    total += len(batch_y)

            val_acc = correct / total if total > 0 else 0.0

            # Scheduler Step
            if scheduler and config.scheduler == SchedulerType.REDUCE_ON_PLATEAU:
                scheduler.step(val_acc)
            elif scheduler:
                scheduler.step()

            # Early Stopping Check
            if val_acc > best_val_acc + config.min_delta:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if config.early_stopping and patience_counter >= config.patience:
                logger.debug(f"[Split] Early Stopping nach {epoch + 1} Epochen")
                break

        # Bestes Modell laden
        if best_model_state:
            model.load_state_dict(best_model_state)

        epochs_completed = epoch + 1
        return model, {'epochs': epochs_completed, 'best_val_acc': best_val_acc}

    def _create_optimizer(self, model: nn.Module, config: WalkForwardConfig):
        """Erstellt Optimizer basierend auf Config."""
        params = model.parameters()

        if config.optimizer == OptimizerType.ADAM:
            return torch.optim.Adam(params, lr=config.learning_rate, betas=config.betas)
        elif config.optimizer == OptimizerType.ADAMW:
            return torch.optim.AdamW(
                params, lr=config.learning_rate, betas=config.betas,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == OptimizerType.SGD:
            return torch.optim.SGD(
                params, lr=config.learning_rate, momentum=config.momentum,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == OptimizerType.RMSPROP:
            return torch.optim.RMSprop(params, lr=config.learning_rate, momentum=config.momentum)
        else:
            raise ValueError(f"Unbekannter Optimizer: {config.optimizer}")

    def _create_scheduler(self, optimizer, config: WalkForwardConfig):
        """Erstellt LR Scheduler basierend auf Config."""
        if config.scheduler == SchedulerType.NONE:
            return None
        elif config.scheduler == SchedulerType.REDUCE_ON_PLATEAU:
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=config.scheduler_factor,
                patience=config.scheduler_patience
            )
        elif config.scheduler == SchedulerType.STEP_LR:
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma
            )
        elif config.scheduler == SchedulerType.COSINE_ANNEALING:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.scheduler_t_max, eta_min=config.scheduler_eta_min
            )
        else:
            return None

    def _get_varied_config(self, split_index: int) -> WalkForwardConfig:
        """Erstellt variierte Config fuer einen Split."""
        if not self.config.hyperparam_variation:
            return self.config

        rng = np.random.RandomState(
            (self.config.random_seed or 0) + split_index
        )
        var = self.config.variation_percent

        varied = copy.deepcopy(self.config)

        if self.config.vary_learning_rate:
            factor = 1.0 + rng.uniform(-var, var)
            varied.learning_rate = self.config.learning_rate * factor

        if self.config.vary_dropout and varied.dropout:
            factor = 1.0 + rng.uniform(-var, var)
            varied.dropout = min(0.5, max(0.0, varied.dropout * factor))

        if self.config.vary_hidden_size and varied.hidden_size:
            factor = 1.0 + rng.uniform(-var, var)
            varied.hidden_size = int(varied.hidden_size * factor)

        return varied

    # =========================================================================
    # Modus 4: LIVE_SIMULATION
    # =========================================================================

    def _run_live_simulation(
        self,
        splits: List[Tuple],
        callback: Optional[Callable]
    ) -> List[SplitResult]:
        """
        LIVE SIMULATION: Neutraining + Bar-by-Bar.
        Realistischster Modus ohne Look-Ahead Bias.
        """
        logger.info("[WalkForward] Live-Simulation (Retrain + Bar-by-Bar)")

        results = []
        for i, split in enumerate(splits):
            if self.is_cancelled():
                break

            result = self._process_live_simulation_split(i, split)
            results.append(result)

            if callback:
                callback(i + 1, len(splits),
                        f"Split {i + 1}/{len(splits)} (Live-Sim)")

        return results

    def _process_live_simulation_split(self, split_idx: int, split_info: Tuple) -> SplitResult:
        """Trainiert Modell und fuehrt Bar-by-Bar Simulation durch."""
        train_range, test_range = split_info

        data = self._filter_date_range()
        train_data = data.iloc[train_range[0]:train_range[1]]
        test_data = data.iloc[test_range[0]:test_range[1]]

        config = self._get_varied_config(split_idx)

        # Modell trainieren
        model, training_info = self._train_model_for_split(train_data, config)

        # CUDA synchronisieren vor Model-Transfer
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        model.to(self.device)
        model.eval()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Bar-by-Bar Simulation
        rolling_norm = RollingNormalizer(lookback=self.lookback)

        predictions = []
        confidences = []
        feature_buffer = []

        for i in range(len(test_data)):
            new_bar = test_data.iloc[i:i + 1]
            feature_buffer.append(new_bar)
            if len(feature_buffer) > self.lookback:
                feature_buffer.pop(0)

            if len(feature_buffer) < self.lookback:
                predictions.append(1)
                confidences.append(0.0)
                continue

            buffer_df = pd.concat(feature_buffer, ignore_index=True)
            feature_df = self.processor.process(buffer_df)
            feature_matrix = self.processor.get_feature_matrix(feature_df)

            raw_sequence = feature_matrix[-self.lookback:]
            normalized_sequence = []
            for row in raw_sequence:
                norm_row = rolling_norm.update_and_normalize(row)
                normalized_sequence.append(norm_row)
            sequence = np.array(normalized_sequence, dtype=np.float32)

            with torch.no_grad():
                x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                pred = logits.argmax(dim=1).item()
                conf = probs.max(dim=1).values.item()

            predictions.append(pred)
            confidences.append(conf)

        predictions = np.array(predictions)
        confidences = np.array(confidences)

        result = self._calculate_split_metrics(
            split_idx, train_range, test_range,
            predictions, confidences, test_data
        )
        result.training_epochs = training_info.get('epochs', 0)
        result.best_val_accuracy = training_info.get('best_val_acc', 0.0)

        del model
        torch.cuda.empty_cache()

        return result

    # =========================================================================
    # Metriken-Berechnung
    # =========================================================================

    def _calculate_split_metrics(
        self,
        split_idx: int,
        train_range: Tuple[int, int],
        test_range: Tuple[int, int],
        predictions: np.ndarray,
        confidences: np.ndarray,
        test_data: pd.DataFrame
    ) -> SplitResult:
        """Berechnet alle Metriken fuer einen Split."""
        logger.debug(f"[Split {split_idx}] _calculate_split_metrics gestartet")
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        # Labels aus Test-Daten generieren
        logger.debug(f"[Split {split_idx}] Generiere Labels...")
        labeler = DailyExtremaLabeler(
            method=self.model_info.get('label_method', 'future_return')
        )
        true_labels = labeler.generate(test_data)
        logger.debug(f"[Split {split_idx}] Labels generiert: {len(true_labels)}, Unique: {np.unique(true_labels)}")

        # Nur gueltige Predictions (nach Lookback)
        valid_start = self.lookback
        if valid_start >= len(true_labels):
            logger.warning(f"[Split {split_idx}] Lookback ({valid_start}) >= Labels ({len(true_labels)})")
            return self._create_empty_split_result(split_idx, train_range, test_range)

        true_labels = true_labels[valid_start:valid_start + len(predictions)]
        if len(true_labels) != len(predictions):
            min_len = min(len(true_labels), len(predictions))
            logger.debug(f"[Split {split_idx}] Anpassung: Labels {len(true_labels)} != Predictions {len(predictions)}, min={min_len}")
            true_labels = true_labels[:min_len]
            predictions = predictions[:min_len]
            confidences = confidences[:min_len]

        logger.debug(f"[Split {split_idx}] Finale Groessen: Labels={len(true_labels)}, Predictions={len(predictions)}")

        # Klassifikations-Metriken
        logger.debug(f"[Split {split_idx}] Berechne Klassifikations-Metriken...")
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        logger.debug(f"[Split {split_idx}] Accuracy={accuracy:.2%}, F1={f1:.3f}")

        # Trading-Simulation
        logger.debug(f"[Split {split_idx}] Starte Trading-Simulation...")
        trading_result = self._simulate_trading(predictions, confidences, test_data, split_idx)
        logger.debug(f"[Split {split_idx}] Trading-Simulation abgeschlossen: {trading_result['num_trades']} Trades")

        return SplitResult(
            split_index=split_idx,
            train_range=train_range,
            test_range=test_range,
            accuracy=accuracy,
            f1_score=f1,
            precision=precision,
            recall=recall,
            total_return=trading_result['total_return'],
            sharpe_ratio=trading_result['sharpe_ratio'],
            max_drawdown=trading_result['max_drawdown'],
            num_trades=trading_result['num_trades'],
            win_rate=trading_result['win_rate'],
            profit_factor=trading_result['profit_factor'],
            trades=trading_result['trades']
        )

    def _simulate_trading(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        test_data: pd.DataFrame,
        split_idx: int
    ) -> Dict:
        """Simuliert Trading basierend auf Predictions."""
        if self.config.backtest_engine == BacktestEngine.BACKTRADER:
            return self._simulate_with_backtrader(predictions, confidences, test_data, split_idx)
        else:
            return self._simulate_simple(predictions, confidences, test_data, split_idx)

    def _simulate_simple(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        test_data: pd.DataFrame,
        split_idx: int
    ) -> Dict:
        """Einfache Trading-Simulation ohne Backtrader."""
        logger.debug(f"[Split {split_idx}] _simulate_simple gestartet")
        logger.debug(f"[Split {split_idx}] Predictions: {len(predictions)}, Test-Daten: {len(test_data)}")
        logger.debug(f"[Split {split_idx}] Num Classes: {self.num_classes}")

        # Signal-Mapping basierend auf num_classes:
        # 2 Klassen: 0=BUY, 1=SELL
        # 3 Klassen: 0=HOLD, 1=BUY, 2=SELL
        if self.num_classes == 2:
            BUY_SIGNAL = 0
            SELL_SIGNAL = 1
        else:  # 3 Klassen
            BUY_SIGNAL = 1
            SELL_SIGNAL = 2
        logger.debug(f"[Split {split_idx}] Signal-Mapping: BUY={BUY_SIGNAL}, SELL={SELL_SIGNAL}")

        # Prediction-Verteilung loggen
        unique, counts = np.unique(predictions, return_counts=True)
        logger.debug(f"[Split {split_idx}] Prediction-Verteilung: {dict(zip(unique, counts))}")

        capital = self.config.initial_capital
        position = 0  # 0=flat, 1=long, -1=short
        entry_price = 0.0
        entry_idx = 0

        trades = []
        returns = []
        equity_history = [capital]

        # Offset fuer Lookback
        offset = self.lookback if len(predictions) < len(test_data) else 0
        prices = test_data['Close'].values[offset:offset + len(predictions)]
        logger.debug(f"[Split {split_idx}] Offset: {offset}, Preise: {len(prices)}")

        if len(prices) != len(predictions):
            logger.debug(f"[Split {split_idx}] Preis-Anpassung: {len(prices)} -> {len(predictions)}")
            prices = prices[:len(predictions)]

        trade_id = 0

        for i, (pred, conf, price) in enumerate(zip(predictions, confidences, prices)):
            dt = test_data.index[offset + i] if hasattr(test_data.index, '__getitem__') else None

            if pred == BUY_SIGNAL and position <= 0:  # BUY Signal
                if position < 0:  # Close Short
                    pnl = entry_price - price
                    pnl_pct = pnl / entry_price if entry_price > 0 else 0
                    capital += pnl * abs(position)
                    trades.append(TradeRecord(
                        trade_id=trade_id,
                        split_index=split_idx,
                        entry_time=test_data.index[entry_idx] if entry_idx < len(test_data.index) else dt,
                        exit_time=dt,
                        signal='SELL',
                        entry_price=entry_price,
                        exit_price=price,
                        size=abs(position),
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        duration=None,
                        prediction=2,
                        confidence=conf
                    ))
                    trade_id += 1
                    returns.append(pnl_pct)

                # Open Long
                position = 1
                entry_price = price
                entry_idx = offset + i

            elif pred == SELL_SIGNAL and position >= 0:  # SELL Signal
                if position > 0:  # Close Long
                    pnl = price - entry_price
                    pnl_pct = pnl / entry_price if entry_price > 0 else 0
                    capital += pnl * position
                    trades.append(TradeRecord(
                        trade_id=trade_id,
                        split_index=split_idx,
                        entry_time=test_data.index[entry_idx] if entry_idx < len(test_data.index) else dt,
                        exit_time=dt,
                        signal='BUY',
                        entry_price=entry_price,
                        exit_price=price,
                        size=position,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        duration=None,
                        prediction=0,
                        confidence=conf
                    ))
                    trade_id += 1
                    returns.append(pnl_pct)

                # Open Short
                position = -1
                entry_price = price
                entry_idx = offset + i

            equity_history.append(capital)

        # Metriken berechnen
        total_return = (capital - self.config.initial_capital) / self.config.initial_capital
        num_trades = len(trades)
        logger.debug(f"[Split {split_idx}] Simulation beendet: {num_trades} Trades")

        if len(returns) > 0:
            avg_return = np.mean(returns)
            std_return = np.std(returns) if len(returns) > 1 else 0.001
            sharpe_ratio = avg_return / std_return * np.sqrt(252) if std_return > 0 else 0.0
            win_rate = sum(1 for r in returns if r > 0) / len(returns)
            wins = sum(r for r in returns if r > 0)
            losses = abs(sum(r for r in returns if r < 0))
            profit_factor = wins / losses if losses > 0 else float('inf')
            logger.debug(f"[Split {split_idx}] Returns: {len(returns)}, Avg={avg_return:.4f}, Sharpe={sharpe_ratio:.3f}")
        else:
            sharpe_ratio = 0.0
            win_rate = 0.0
            profit_factor = 0.0
            logger.debug(f"[Split {split_idx}] Keine Returns (keine abgeschlossenen Trades)")

        # Max Drawdown
        peak = equity_history[0]
        max_dd = 0.0
        for eq in equity_history:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        logger.debug(f"[Split {split_idx}] Ergebnis: Return={total_return:.2%}, DD={max_dd:.2%}, Trades={num_trades}")

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'trades': trades
        }

    def _simulate_with_backtrader(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        test_data: pd.DataFrame,
        split_idx: int
    ) -> Dict:
        """Trading-Simulation mit Backtrader-Engine."""
        from .backtrader_engine import BacktraderEngine

        # Predictions als Signal-Spalte hinzufuegen
        test_df = test_data.copy()
        offset = self.lookback if len(predictions) < len(test_data) else 0
        test_df = test_df.iloc[offset:offset + len(predictions)]
        test_df['signal'] = predictions

        # Backtrader Engine
        engine = BacktraderEngine(test_df, model=None, model_info=self.model_info)
        engine.prepare_data()

        result = engine.run_backtest(
            initial_capital=self.config.initial_capital,
            commission=self.config.bt_commission_perc,
            slippage=self.config.bt_slippage_perc,
            stake_pct=95.0,
            allow_short=True
        )

        # Trades konvertieren
        trades = []
        for i, t in enumerate(result.trades):
            trades.append(TradeRecord(
                trade_id=i,
                split_index=split_idx,
                entry_time=t.get('entry_date'),
                exit_time=t.get('exit_date'),
                signal='BUY' if t.get('size', 0) > 0 else 'SELL',
                entry_price=t.get('entry_price', 0),
                exit_price=t.get('exit_price', 0),
                size=abs(t.get('size', 0)),
                pnl=t.get('pnl', 0),
                pnl_pct=t.get('pnl_pct', 0),
                duration=None,
                prediction=0,
                confidence=0.0,
                commission=t.get('commission', 0)
            ))

        return {
            'total_return': result.total_return_pct / 100,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown_pct / 100,
            'num_trades': result.total_trades,
            'win_rate': result.win_rate / 100,
            'profit_factor': result.profit_factor,
            'trades': trades
        }

    # =========================================================================
    # Ergebnis-Aggregation
    # =========================================================================

    def _aggregate_results(
        self,
        results: List[SplitResult],
        duration: float
    ) -> WalkForwardResult:
        """Aggregiert alle Split-Ergebnisse."""
        logger.debug(f"[WalkForward] _aggregate_results: {len(results)} Splits, Dauer: {duration:.1f}s")

        if len(results) == 0:
            logger.warning("[WalkForward] Keine Ergebnisse zum Aggregieren")
            return self._create_empty_result(time.time() - duration)

        # Metriken sammeln
        accuracies = [r.accuracy for r in results]
        returns = [r.total_return for r in results]
        sharpes = [r.sharpe_ratio for r in results]
        drawdowns = [r.max_drawdown for r in results]
        logger.debug(f"[WalkForward] Accuracies: {[f'{a:.2%}' for a in accuracies]}")
        logger.debug(f"[WalkForward] Returns: {[f'{r:.2%}' for r in returns]}")

        # Alle Trades sammeln
        all_trades = []
        for r in results:
            all_trades.extend(r.trades)
        logger.debug(f"[WalkForward] Gesamt-Trades: {len(all_trades)}")

        # Aggregierte Metriken
        total_return = np.prod([1 + r for r in returns]) - 1
        total_trades = sum(r.num_trades for r in results)
        logger.debug(f"[WalkForward] Total Return: {total_return:.2%}, Total Trades: {total_trades}")

        # Win Rate gesamt
        winning_trades = sum(1 for t in all_trades if t.pnl > 0)
        win_rate = winning_trades / len(all_trades) if len(all_trades) > 0 else 0.0

        # Profit Factor gesamt
        total_wins = sum(t.pnl for t in all_trades if t.pnl > 0)
        total_losses = abs(sum(t.pnl for t in all_trades if t.pnl < 0))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

        # Robustheits-Score
        robustness = None
        if self.config.hyperparam_variation and len(accuracies) > 1:
            robustness = np.std(accuracies) / np.mean(accuracies)

        return WalkForwardResult(
            config=self.config,
            splits=results,
            mode=self.config.mode.value,
            timestamp=datetime.now(),
            total_return=total_return,
            avg_return=np.mean(returns),
            std_return=np.std(returns),
            avg_accuracy=np.mean(accuracies),
            std_accuracy=np.std(accuracies),
            avg_sharpe=np.mean(sharpes),
            max_drawdown=max(drawdowns) if drawdowns else 0.0,
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            duration_sec=duration,
            parallel_workers=self._calculate_max_parallel(),
            all_trades=all_trades,
            robustness_score=robustness
        )

    def _create_empty_result(self, start_time: float) -> WalkForwardResult:
        """Erstellt leeres Ergebnis."""
        return WalkForwardResult(
            config=self.config,
            splits=[],
            mode=self.config.mode.value,
            timestamp=datetime.now(),
            duration_sec=time.time() - start_time
        )

    def _create_empty_split_result(
        self,
        split_idx: int,
        train_range: Tuple[int, int],
        test_range: Tuple[int, int]
    ) -> SplitResult:
        """Erstellt leeres Split-Ergebnis."""
        return SplitResult(
            split_index=split_idx,
            train_range=train_range,
            test_range=test_range,
            accuracy=0.0,
            f1_score=0.0
        )
