"""
BTCUSD Analyzer - Haupteinstiegspunkt
"""

import sys
from pathlib import Path


def main():
    """Haupteinstiegspunkt fuer die Anwendung."""

    # Basis-Verzeichnis bestimmen (btcusd_analyzer_python)
    # __file__ -> main.py -> btcusd_analyzer -> src -> btcusd_analyzer_python
    base_dir = Path(__file__).parent.parent.parent

    # Logger initialisieren
    from btcusd_analyzer.core.logger import get_logger
    logger = get_logger('btcusd_analyzer', log_dir=base_dir / 'log')
    logger.info('BTCUSD Analyzer gestartet')
    logger.info(f'Log-Datei: {logger.get_log_file_path()}')

    # GPU-Status pruefen
    from btcusd_analyzer.utils.helpers import get_gpu_info
    gpu_info = get_gpu_info()
    if gpu_info['cuda_available']:
        logger.success(f'GPU verfuegbar: {gpu_info["devices"][0]["name"]}')
    else:
        logger.warning('Keine GPU verfuegbar - CPU wird verwendet')

    # PyQt6 GUI starten
    try:
        from PyQt6.QtWidgets import QApplication
        from btcusd_analyzer.gui.main_window import MainWindow

        app = QApplication(sys.argv)
        app.setApplicationName('BTCUSD Analyzer')
        app.setOrganizationName('BTCUSD')

        # Dark Theme
        app.setStyle('Fusion')

        window = MainWindow(base_dir)
        window.show()

        sys.exit(app.exec())

    except ImportError as e:
        logger.error(f'PyQt6 nicht installiert: {e}')
        logger.info('Starte im CLI-Modus...')
        cli_mode(base_dir, logger)


def cli_mode(base_dir: Path, logger):
    """Kommandozeilen-Modus (falls keine GUI verfuegbar)."""
    print('\n=== BTCUSD Analyzer (CLI) ===\n')
    print('Optionen:')
    print('  1. Daten laden')
    print('  2. Modell trainieren')
    print('  3. Backtest durchfuehren')
    print('  4. Beenden')

    while True:
        try:
            choice = input('\nAuswahl: ').strip()

            if choice == '1':
                _cli_load_data(base_dir, logger)
            elif choice == '2':
                _cli_train_model(base_dir, logger)
            elif choice == '3':
                _cli_backtest(base_dir, logger)
            elif choice == '4':
                print('Auf Wiedersehen!')
                break
            else:
                print('Ungueltige Auswahl')

        except KeyboardInterrupt:
            print('\nAbgebrochen')
            break
        except Exception as e:
            logger.error(f'Fehler: {e}')


def _cli_load_data(base_dir: Path, logger):
    """Laedt Daten im CLI-Modus."""
    from btcusd_analyzer.data.reader import CSVReader

    reader = CSVReader(base_dir / 'Daten_csv')
    df, filepath = reader.load_last_csv()

    if df is not None:
        reader.log_data_info(df, filepath)
        print(f'\n{len(df)} Datensaetze geladen')
    else:
        print('Keine Daten gefunden')


def _cli_train_model(base_dir: Path, logger):
    """Trainiert ein Modell im CLI-Modus."""
    from btcusd_analyzer.core.config import Config
    from btcusd_analyzer.data.reader import CSVReader
    from btcusd_analyzer.data.processor import FeatureProcessor
    from btcusd_analyzer.training.labeler import DailyExtremaLabeler
    from btcusd_analyzer.training.sequence import SequenceGenerator
    from btcusd_analyzer.models.bilstm import BiLSTMClassifier
    from btcusd_analyzer.trainer.trainer import Trainer
    from btcusd_analyzer.trainer.callbacks import EarlyStopping, ModelCheckpoint

    config = Config(base_dir)

    # Daten laden
    reader = CSVReader(config.paths.data_dir)
    df, _ = reader.load_last_csv()

    if df is None:
        print('Keine Daten gefunden')
        return

    # Features generieren
    processor = FeatureProcessor(config.training.features)
    df = processor.process(df)
    features = processor.get_feature_matrix(df)

    # Labels generieren
    labeler = DailyExtremaLabeler(
        lookforward=config.training.lookforward,
        threshold_pct=2.0
    )
    labels = labeler.generate_labels(df)

    # Klassenverteilung anzeigen
    dist = labeler.get_class_distribution(labels)
    print('\nKlassenverteilung:')
    for cls, info in dist.items():
        print(f'  {cls}: {info["count"]} ({info["percentage"]:.1f}%)')

    # Sequenzen generieren
    seq_gen = SequenceGenerator(
        lookback=config.training.lookback,
        lookforward=config.training.lookforward
    )

    train_loader, val_loader = seq_gen.create_dataloaders(
        features, labels,
        batch_size=config.training.batch_size,
        validation_split=config.training.validation_split
    )

    # Modell erstellen
    model = BiLSTMClassifier(
        input_size=config.training.input_size,
        hidden_size=config.training.hidden_size,
        num_layers=config.training.num_layers,
        num_classes=config.training.num_classes,
        dropout=config.training.dropout
    )

    print(f'\nModell: {model.name}')
    print(f'Parameter: {model.count_parameters():,}')

    # Callbacks
    session_dir = config.paths.get_session_dir('training')
    callbacks = [
        EarlyStopping(patience=config.training.patience),
        ModelCheckpoint(
            filepath=str(session_dir / '{epoch:02d}_{val_accuracy:.1f}.pt'),
            save_best_only=True
        )
    ]

    # Training
    trainer = Trainer(model, callbacks=callbacks)

    class_weights = labeler.get_label_weights(labels)

    history = trainer.train(
        train_loader, val_loader,
        epochs=config.training.epochs,
        learning_rate=config.training.learning_rate,
        class_weights=class_weights
    )

    print(f'\nBeste Validation Accuracy: {history.best_val_accuracy:.2f}%')
    print(f'Ergebnisse in: {session_dir}')


def _cli_backtest(base_dir: Path, logger):
    """Fuehrt Backtest im CLI-Modus durch."""
    print('Backtest noch nicht implementiert im CLI-Modus')


if __name__ == '__main__':
    main()
