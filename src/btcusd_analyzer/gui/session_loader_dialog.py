"""
SessionLoaderDialog - Dialog zum Laden einer Session
"""

from pathlib import Path
from typing import Optional, List

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from ..core.session_manager import SessionManager


class SessionLoaderDialog(QDialog):
    """Dialog zur Auswahl einer Session zum Laden."""

    def __init__(self, log_dir: Path, parent=None):
        super().__init__(parent)
        self.log_dir = Path(log_dir)
        self.selected_session: Optional[Path] = None

        self._init_ui()
        self._load_sessions()

    def _init_ui(self):
        """Initialisiert die UI."""
        self.setWindowTitle('Session laden')
        self.setMinimumSize(700, 400)
        self.setStyleSheet('''
            QDialog {
                background-color: #262626;
            }
            QLabel {
                color: #cccccc;
            }
            QTableWidget {
                background-color: #1a1a1a;
                color: #cccccc;
                border: 1px solid #333;
                gridline-color: #333;
            }
            QTableWidget::item:selected {
                background-color: #3a5a8a;
            }
            QHeaderView::section {
                background-color: #333;
                color: #cccccc;
                padding: 5px;
                border: 1px solid #444;
            }
            QPushButton {
                background-color: #444;
                color: white;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #555;
            }
            QPushButton:disabled {
                background-color: #333;
                color: #666;
            }
        ''')

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Titel
        title = QLabel('Verfuegbare Sessions')
        title.setFont(QFont('Segoe UI', 12, QFont.Weight.Bold))
        title.setStyleSheet('color: #4de6ff;')
        layout.addWidget(title)

        # Info
        info = QLabel('Sessions mit Trainingsdaten und/oder Modellen:')
        info.setStyleSheet('color: #888;')
        layout.addWidget(info)

        # Tabelle
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            'Session', 'Training', 'Backtest', 'Modell', 'Accuracy'
        ])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.doubleClicked.connect(self._on_double_click)
        layout.addWidget(self.table)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.cancel_btn = QPushButton('Abbrechen')
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)

        self.load_btn = QPushButton('Laden')
        self.load_btn.setStyleSheet('''
            QPushButton {
                background-color: #3a7a5a;
            }
            QPushButton:hover {
                background-color: #4a9a6a;
            }
        ''')
        self.load_btn.clicked.connect(self._on_load)
        btn_layout.addWidget(self.load_btn)

        layout.addLayout(btn_layout)

    def _load_sessions(self):
        """Laedt alle verfuegbaren Sessions."""
        sessions = SessionManager.list_sessions(self.log_dir)

        self.table.setRowCount(len(sessions))
        self._sessions = []

        for row, session in enumerate(sessions):
            self._sessions.append(session['session_dir'])

            # Session-Name
            name_item = QTableWidgetItem(session['session_name'])
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, name_item)

            # Training-Daten
            train_item = QTableWidgetItem('Ja' if session['has_training_data'] else '-')
            train_item.setFlags(train_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            train_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if session['has_training_data']:
                train_item.setForeground(Qt.GlobalColor.green)
            self.table.setItem(row, 1, train_item)

            # Backtest-Daten
            bt_item = QTableWidgetItem('Ja' if session['has_backtest_data'] else '-')
            bt_item.setFlags(bt_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            bt_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if session['has_backtest_data']:
                bt_item.setForeground(Qt.GlobalColor.green)
            self.table.setItem(row, 2, bt_item)

            # Modell
            model_item = QTableWidgetItem('Ja' if session['has_model'] else '-')
            model_item.setFlags(model_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            model_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if session['has_model']:
                model_item.setForeground(Qt.GlobalColor.green)
            self.table.setItem(row, 3, model_item)

            # Accuracy
            acc = session.get('model_accuracy', 0)
            acc_text = f'{acc:.1f}%' if acc > 0 else '-'
            acc_item = QTableWidgetItem(acc_text)
            acc_item.setFlags(acc_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            acc_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if acc >= 70:
                acc_item.setForeground(Qt.GlobalColor.green)
            elif acc >= 60:
                acc_item.setForeground(Qt.GlobalColor.yellow)
            self.table.setItem(row, 4, acc_item)

        if not sessions:
            # Keine Sessions gefunden
            self.table.setRowCount(1)
            item = QTableWidgetItem('Keine Sessions mit Daten gefunden')
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setForeground(Qt.GlobalColor.gray)
            self.table.setItem(0, 0, item)
            self.load_btn.setEnabled(False)

    def _on_double_click(self):
        """Doppelklick laedt Session."""
        self._on_load()

    def _on_load(self):
        """Laedt die ausgewaehlte Session."""
        row = self.table.currentRow()
        if row >= 0 and row < len(self._sessions):
            self.selected_session = Path(self._sessions[row])
            self.accept()

    def get_selected_session(self) -> Optional[Path]:
        """Gibt die ausgewaehlte Session zurueck."""
        return self.selected_session
