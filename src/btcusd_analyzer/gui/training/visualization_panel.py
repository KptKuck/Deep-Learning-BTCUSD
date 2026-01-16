"""
Visualization Panel - Plots, Metriken-Tabelle und Log fuer Training.
"""

from typing import Dict, List
from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QTextEdit, QProgressBar
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from ..styles import COLORS


class VisualizationPanel(QWidget):
    """
    Visualisierungs-Panel fuer Training.

    Enthaelt:
    - Live Loss/Accuracy Plots
    - Metriken-Tabelle
    - Log-Ausgabe
    - Fortschrittsanzeige
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self._setup_ui()

    def _setup_ui(self):
        """Erstellt die UI-Komponenten."""
        layout = QVBoxLayout(self)

        # Tabs
        self.tabs = QTabWidget()

        # Tab 1: Live-Plot
        plot_tab = self._create_plot_tab()
        self.tabs.addTab(plot_tab, "Training-Verlauf")

        # Tab 2: Metriken-Tabelle
        metrics_tab = self._create_metrics_tab()
        self.tabs.addTab(metrics_tab, "Metriken")

        # Tab 3: Log
        log_tab = self._create_log_tab()
        self.tabs.addTab(log_tab, "Log")

        layout.addWidget(self.tabs)

        # Fortschrittsanzeige
        self._create_progress_section(layout)

    def _create_plot_tab(self) -> QWidget:
        """Erstellt den Plot-Tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure

            self.figure = Figure(figsize=(8, 6), facecolor=COLORS['bg_primary'])
            self.canvas = FigureCanvas(self.figure)

            self.ax_loss = self.figure.add_subplot(211)
            self.ax_acc = self.figure.add_subplot(212)

            self._has_matplotlib = True
            self._setup_plot_style()
            layout.addWidget(self.canvas)

        except ImportError:
            layout.addWidget(QLabel("matplotlib nicht installiert - pip install matplotlib"))
            self._has_matplotlib = False

        return widget

    def _setup_plot_style(self):
        """Konfiguriert den Plot-Style."""
        if not self._has_matplotlib:
            return

        for ax in [self.ax_loss, self.ax_acc]:
            ax.set_facecolor(COLORS['bg_secondary'])
            ax.tick_params(colors=COLORS['text_secondary'])
            for spine in ax.spines.values():
                spine.set_color(COLORS['border'])

        self.ax_loss.set_title('Loss', color=COLORS['text_primary'])
        self.ax_loss.set_xlabel('Epoch', color=COLORS['text_secondary'])
        self.ax_loss.set_ylabel('Loss', color=COLORS['text_secondary'])

        self.ax_acc.set_title('Accuracy', color=COLORS['text_primary'])
        self.ax_acc.set_xlabel('Epoch', color=COLORS['text_secondary'])
        self.ax_acc.set_ylabel('Accuracy (%)', color=COLORS['text_secondary'])

        self.figure.tight_layout()

    def _create_metrics_tab(self) -> QWidget:
        """Erstellt den Metriken-Tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(5)
        self.metrics_table.setHorizontalHeaderLabels([
            'Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc'
        ])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        layout.addWidget(self.metrics_table)
        return widget

    def _create_log_tab(self) -> QWidget:
        """Erstellt den Log-Tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['bg_secondary']};
                color: {COLORS['text_primary']};
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
            }}
        """)
        layout.addWidget(self.log_text)

        clear_btn = QPushButton("Log leeren")
        clear_btn.clicked.connect(lambda: self.log_text.clear())
        layout.addWidget(clear_btn)

        return widget

    def _create_progress_section(self, layout: QVBoxLayout):
        """Erstellt die Fortschrittsanzeige."""
        group = QGroupBox("Fortschritt")
        group_layout = QVBoxLayout(group)

        # Epoch + Zeit
        top_layout = QHBoxLayout()

        self.epoch_label = QLabel("Epoch: 0 / 0")
        self.epoch_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        top_layout.addWidget(self.epoch_label)
        top_layout.addStretch()

        self.time_label = QLabel("Zeit: --:--")
        top_layout.addWidget(self.time_label)
        group_layout.addLayout(top_layout)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        group_layout.addWidget(self.progress_bar)

        # Aktuelle Metriken
        metrics_layout = QHBoxLayout()

        self.train_loss_label = QLabel("Train Loss: -")
        self.train_acc_label = QLabel("Train Acc: -")
        self.val_loss_label = QLabel("Val Loss: -")
        self.val_acc_label = QLabel("Val Acc: -")

        for label in [self.train_loss_label, self.train_acc_label,
                      self.val_loss_label, self.val_acc_label]:
            label.setStyleSheet(f"color: {COLORS['text_secondary']};")
            metrics_layout.addWidget(label)

        group_layout.addLayout(metrics_layout)
        layout.addWidget(group)

    def update_epoch(self, epoch: int, train_loss: float, train_acc: float,
                     val_loss: float, val_acc: float, total_epochs: int):
        """Aktualisiert die Anzeige nach einer Epoche."""
        # History aktualisieren
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc * 100)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc * 100)

        # Labels aktualisieren
        self.epoch_label.setText(f"Epoch: {epoch} / {total_epochs}")
        self.train_loss_label.setText(f"Train Loss: {train_loss:.4f}")
        self.train_acc_label.setText(f"Train Acc: {train_acc*100:.1f}%")
        self.val_loss_label.setText(f"Val Loss: {val_loss:.4f}")
        self.val_acc_label.setText(f"Val Acc: {val_acc*100:.1f}%")

        # Progress Bar
        progress = int((epoch / total_epochs) * 100)
        self.progress_bar.setValue(progress)

        # Tabelle aktualisieren
        self._add_metrics_row(epoch, train_loss, train_acc, val_loss, val_acc)

        # Plot aktualisieren
        self._update_plot()

    def _add_metrics_row(self, epoch: int, train_loss: float, train_acc: float,
                         val_loss: float, val_acc: float):
        """Fuegt eine Zeile zur Metriken-Tabelle hinzu."""
        row = self.metrics_table.rowCount()
        self.metrics_table.insertRow(row)

        self.metrics_table.setItem(row, 0, QTableWidgetItem(str(epoch)))
        self.metrics_table.setItem(row, 1, QTableWidgetItem(f"{train_loss:.4f}"))
        self.metrics_table.setItem(row, 2, QTableWidgetItem(f"{train_acc*100:.1f}%"))
        self.metrics_table.setItem(row, 3, QTableWidgetItem(f"{val_loss:.4f}"))
        self.metrics_table.setItem(row, 4, QTableWidgetItem(f"{val_acc*100:.1f}%"))

        # Scroll zur letzten Zeile
        self.metrics_table.scrollToBottom()

    def _update_plot(self):
        """Aktualisiert den Plot."""
        if not self._has_matplotlib:
            return

        epochs = list(range(1, len(self.history['train_loss']) + 1))

        # Loss Plot
        self.ax_loss.clear()
        if epochs:
            self.ax_loss.plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
            self.ax_loss.plot(epochs, self.history['val_loss'], 'r-', label='Validation', linewidth=2)
            self.ax_loss.legend(facecolor=COLORS['bg_tertiary'], labelcolor=COLORS['text_primary'])
        self.ax_loss.set_title('Loss', color=COLORS['text_primary'])
        self.ax_loss.set_facecolor(COLORS['bg_secondary'])

        # Accuracy Plot
        self.ax_acc.clear()
        if epochs:
            self.ax_acc.plot(epochs, self.history['train_acc'], 'b-', label='Train', linewidth=2)
            self.ax_acc.plot(epochs, self.history['val_acc'], 'r-', label='Validation', linewidth=2)
            self.ax_acc.legend(facecolor=COLORS['bg_tertiary'], labelcolor=COLORS['text_primary'])
        self.ax_acc.set_title('Accuracy', color=COLORS['text_primary'])
        self.ax_acc.set_facecolor(COLORS['bg_secondary'])

        self.figure.tight_layout()
        self.canvas.draw()

    def log(self, message: str, level: str = 'INFO'):
        """Fuegt eine Log-Nachricht hinzu."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f'[{timestamp}] [{level}] {message}')

    def set_time(self, elapsed_seconds: int):
        """Setzt die Zeitanzeige."""
        minutes = elapsed_seconds // 60
        seconds = elapsed_seconds % 60
        self.time_label.setText(f"Zeit: {minutes:02d}:{seconds:02d}")

    def reset(self):
        """Setzt alle Anzeigen zurueck."""
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.epoch_label.setText("Epoch: 0 / 0")
        self.time_label.setText("Zeit: --:--")
        self.progress_bar.setValue(0)
        self.train_loss_label.setText("Train Loss: -")
        self.train_acc_label.setText("Train Acc: -")
        self.val_loss_label.setText("Val Loss: -")
        self.val_acc_label.setText("Val Acc: -")
        self.metrics_table.setRowCount(0)

        if self._has_matplotlib:
            self.ax_loss.clear()
            self.ax_acc.clear()
            self._setup_plot_style()
            self.canvas.draw()

    def switch_to_tab(self, index: int):
        """Wechselt zum angegebenen Tab."""
        self.tabs.setCurrentIndex(index)

    def add_auto_trainer_tab(self, widget: QWidget):
        """Fuegt Auto-Trainer Tab hinzu."""
        self.tabs.addTab(widget, "Auto-Trainer")
