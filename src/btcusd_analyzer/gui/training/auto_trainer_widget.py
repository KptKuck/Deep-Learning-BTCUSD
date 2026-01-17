"""
Auto-Trainer Widget - UI und Logik fuer automatisches Modell-Training.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal

import torch


class AutoTrainerWidget(QWidget):
    """
    Widget fuer Auto-Trainer Ergebnisse und Detail-Ansicht.

    Zeigt:
    - Ergebnis-Tabelle mit allen getesteten Modellen
    - Detail-Ansicht mit Trainingsverlauf
    - Precision/Recall pro Klasse
    """

    # Signals
    adopt_model = pyqtSignal(object)  # AutoTrainResult
    log_message = pyqtSignal(str, str)  # message, level

    def __init__(self, parent=None):
        super().__init__(parent)
        self.results: List = []
        self._setup_ui()

    def _setup_ui(self):
        """Erstellt die UI."""
        layout = QVBoxLayout(self)

        # Splitter fuer Tabelle und Detail
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Linke Seite: Ergebnis-Tabelle
        splitter.addWidget(self._create_results_table())

        # Rechte Seite: Detail-Ansicht
        splitter.addWidget(self._create_detail_panel())

        splitter.setSizes([600, 400])
        layout.addWidget(splitter)

        # Buttons
        self._create_buttons(layout)

    def _create_results_table(self) -> QWidget:
        """Erstellt die Ergebnis-Tabelle."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(8)
        self.results_table.setHorizontalHeaderLabels([
            'Rang', 'Modell', 'Config', 'Train ACC', 'Val ACC', 'F1', 'Epochen', 'Parameter'
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.itemSelectionChanged.connect(self._on_result_selected)
        layout.addWidget(self.results_table)

        return widget

    def _create_detail_panel(self) -> QWidget:
        """Erstellt das Detail-Panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 0, 0, 0)

        # Header
        header = QLabel("Trainingsverlauf")
        header.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(header)

        # Modell-Info
        self.model_label = QLabel("Modell: -")
        layout.addWidget(self.model_label)

        self.config_label = QLabel("Config: -")
        layout.addWidget(self.config_label)

        # Metriken
        metrics_group = QGroupBox("Metriken")
        metrics_layout = QGridLayout(metrics_group)

        metrics_layout.addWidget(QLabel("Best Epoch:"), 0, 0)
        self.best_epoch_label = QLabel("-")
        metrics_layout.addWidget(self.best_epoch_label, 0, 1)

        metrics_layout.addWidget(QLabel("Trainingszeit:"), 1, 0)
        self.time_label = QLabel("-")
        metrics_layout.addWidget(self.time_label, 1, 1)

        metrics_layout.addWidget(QLabel("F1-Score:"), 2, 0)
        self.f1_label = QLabel("-")
        metrics_layout.addWidget(self.f1_label, 2, 1)

        layout.addWidget(metrics_group)

        # Trainingsverlauf
        history_group = QGroupBox("Verlauf pro Epoche")
        history_layout = QVBoxLayout(history_group)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels([
            'Epoche', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc'
        ])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.history_table.setMaximumHeight(200)
        history_layout.addWidget(self.history_table)

        layout.addWidget(history_group)

        # Precision/Recall pro Klasse
        class_group = QGroupBox("Precision/Recall pro Klasse")
        class_layout = QVBoxLayout(class_group)

        self.class_table = QTableWidget()
        self.class_table.setColumnCount(3)
        self.class_table.setHorizontalHeaderLabels(['Klasse', 'Precision', 'Recall'])
        self.class_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.class_table.setMaximumHeight(100)
        class_layout.addWidget(self.class_table)

        layout.addWidget(class_group)
        layout.addStretch()

        return widget

    def _create_buttons(self, layout: QVBoxLayout):
        """Erstellt die Buttons."""
        btn_layout = QHBoxLayout()

        self.adopt_btn = QPushButton("Bestes Modell uebernehmen")
        self.adopt_btn.clicked.connect(self._on_adopt_clicked)
        self.adopt_btn.setEnabled(False)
        btn_layout.addWidget(self.adopt_btn)

        self.export_btn = QPushButton("Ergebnisse exportieren")
        self.export_btn.clicked.connect(self._export_results)
        self.export_btn.setEnabled(False)
        btn_layout.addWidget(self.export_btn)

        layout.addLayout(btn_layout)

    def _on_result_selected(self):
        """Zeigt Details zum ausgewaehlten Ergebnis."""
        if not self.results:
            return

        selected = self.results_table.selectedItems()
        if not selected:
            return

        row = selected[0].row()
        if row >= len(self.results):
            return

        result = self.results[row]
        self._show_result_details(result)

    def _show_result_details(self, result):
        """Zeigt Details eines Ergebnisses."""
        # Modell-Info
        self.model_label.setText(f"Modell: {result.model_type}")

        if 'hidden_sizes' in result.config:
            config_str = f"hidden_sizes: {result.config['hidden_sizes']}"
        elif 'd_model' in result.config:
            config_str = f"d_model: {result.config.get('d_model')}, nhead: {result.config.get('nhead', 4)}"
        else:
            config_str = str(result.config)
        self.config_label.setText(f"Config: {config_str}")

        # Metriken
        self.best_epoch_label.setText(f"{result.best_epoch}")
        self.time_label.setText(f"{result.training_time:.1f}s")
        self.f1_label.setText(f"{result.f1_score:.4f}")

        # Trainingsverlauf
        history = result.training_history
        train_loss = history.get('train_loss', [])
        self.history_table.setRowCount(len(train_loss))

        for i in range(len(train_loss)):
            self.history_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.history_table.setItem(i, 1, QTableWidgetItem(f"{train_loss[i]:.4f}"))
            self.history_table.setItem(i, 2, QTableWidgetItem(f"{history['train_acc'][i]:.2%}"))
            self.history_table.setItem(i, 3, QTableWidgetItem(f"{history['val_loss'][i]:.4f}"))
            self.history_table.setItem(i, 4, QTableWidgetItem(f"{history['val_acc'][i]:.2%}"))

        self.history_table.scrollToBottom()

        # Precision/Recall
        self.class_table.setRowCount(len(result.precision))
        for i, (class_name, prec) in enumerate(result.precision.items()):
            rec = result.recall.get(class_name, 0.0)
            self.class_table.setItem(i, 0, QTableWidgetItem(class_name))
            self.class_table.setItem(i, 1, QTableWidgetItem(f"{prec:.4f}"))
            self.class_table.setItem(i, 2, QTableWidgetItem(f"{rec:.4f}"))

    def display_results(self, results: List):
        """Zeigt Auto-Training Ergebnisse an."""
        self.results = results
        self.results_table.setRowCount(0)

        for i, result in enumerate(results):
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)

            # Config-String
            if 'hidden_sizes' in result.config:
                config_str = str(result.config['hidden_sizes'])
            elif 'd_model' in result.config:
                config_str = f"d={result.config.get('d_model')}, h={result.config.get('nhead', 4)}"
            else:
                config_str = str(result.config)[:20]

            self.results_table.setItem(row, 0, QTableWidgetItem(str(i + 1)))
            self.results_table.setItem(row, 1, QTableWidgetItem(result.model_type))
            self.results_table.setItem(row, 2, QTableWidgetItem(config_str))
            self.results_table.setItem(row, 3, QTableWidgetItem(f"{result.train_acc:.2%}"))
            self.results_table.setItem(row, 4, QTableWidgetItem(f"{result.val_acc:.2%}"))
            self.results_table.setItem(row, 5, QTableWidgetItem(f"{result.f1_score:.3f}"))
            self.results_table.setItem(row, 6, QTableWidgetItem(f"{result.epochs_trained}"))
            self.results_table.setItem(row, 7, QTableWidgetItem(f"{result.num_parameters:,}"))

            # Beste Zeile hervorheben
            if i == 0:
                for col in range(8):
                    item = self.results_table.item(row, col)
                    if item:
                        item.setBackground(Qt.GlobalColor.darkGreen)

        # Buttons aktivieren
        self.adopt_btn.setEnabled(len(results) > 0)
        self.export_btn.setEnabled(len(results) > 0)

        # Erste Zeile selektieren
        if results:
            self.results_table.selectRow(0)

    def _on_adopt_clicked(self):
        """Uebernimmt das beste Modell."""
        if not self.results:
            return

        selected = self.results_table.selectedItems()
        if selected:
            row = selected[0].row()
            result = self.results[row]
        else:
            result = self.results[0]

        self.adopt_model.emit(result)

    def _export_results(self):
        """Exportiert Ergebnisse als JSON."""
        if not self.results:
            return

        from PyQt6.QtWidgets import QFileDialog
        import json

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Ergebnisse exportieren", "auto_trainer_results.json", "JSON (*.json)"
        )

        if not filepath:
            return

        export_data = []
        for r in self.results:
            export_data.append({
                'rank': r.rank,
                'model_type': r.model_type,
                'config': r.config,
                'train_acc': r.train_acc,
                'val_acc': r.val_acc,
                'f1_score': r.f1_score,
                'epochs_trained': r.epochs_trained,
                'best_epoch': r.best_epoch,
                'training_time': r.training_time,
                'precision': r.precision,
                'recall': r.recall
            })

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        self.log_message.emit(f"Ergebnisse exportiert: {filepath}", 'INFO')

    def clear(self):
        """Leert alle Anzeigen."""
        self.results = []
        self.results_table.setRowCount(0)
        self.history_table.setRowCount(0)
        self.class_table.setRowCount(0)
        self.model_label.setText("Modell: -")
        self.config_label.setText("Config: -")
        self.best_epoch_label.setText("-")
        self.time_label.setText("-")
        self.f1_label.setText("-")
        self.adopt_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
