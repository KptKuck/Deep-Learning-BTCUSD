"""
SessionManagerWindow - Fenster zur Verwaltung aller Sessions.

Bietet eine Uebersicht aller Sessions mit:
- Filterung nach Status (trained/prepared)
- Sortierung nach verschiedenen Kriterien
- Loeschen von Sessions
- Statistiken
- Manuelle Migration
"""

from pathlib import Path
from typing import Optional
import shutil

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QComboBox, QGroupBox, QMessageBox, QCheckBox, QFrame
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor

from ..core.session_database import SessionDatabase
from ..core.session_manager import SessionManager


class SessionManagerWindow(QDialog):
    """Fenster zur Verwaltung aller Sessions."""

    def __init__(self, data_dir: Path, log_dir: Path, parent=None):
        super().__init__(parent)
        self.data_dir = Path(data_dir)
        self.log_dir = Path(log_dir)
        self.db = SessionDatabase(self.data_dir)

        self._init_ui()
        self._load_sessions()
        self._update_statistics()

    def _init_ui(self):
        """Initialisiert die UI."""
        self.setWindowTitle('Session Manager')
        self.setMinimumSize(900, 600)
        self.setStyleSheet('''
            QDialog {
                background-color: #262626;
            }
            QLabel {
                color: #cccccc;
            }
            QGroupBox {
                color: #cccccc;
                border: 1px solid #444;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QTableWidget {
                background-color: #1a1a1a;
                color: #cccccc;
                border: 1px solid #333;
                gridline-color: #333;
                alternate-background-color: #222;
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
            QComboBox {
                background-color: #333;
                color: #cccccc;
                border: 1px solid #444;
                border-radius: 3px;
                padding: 4px 8px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #333;
                color: #cccccc;
                selection-background-color: #3a5a8a;
            }
            QCheckBox {
                color: #cccccc;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
        ''')

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Titel
        title = QLabel('Session Manager')
        title.setFont(QFont('Segoe UI', 14, QFont.Weight.Bold))
        title.setStyleSheet('color: #4de6ff;')
        layout.addWidget(title)

        # Statistiken
        stats_group = QGroupBox('Statistiken')
        stats_layout = QHBoxLayout(stats_group)

        self.stats_total = QLabel('Gesamt: -')
        self.stats_trained = QLabel('Trained: -')
        self.stats_prepared = QLabel('Prepared: -')
        self.stats_avg_acc = QLabel('Avg Accuracy: -')
        self.stats_max_acc = QLabel('Max Accuracy: -')

        for label in [self.stats_total, self.stats_trained, self.stats_prepared,
                      self.stats_avg_acc, self.stats_max_acc]:
            label.setStyleSheet('color: #aaa; padding: 5px 15px;')
            stats_layout.addWidget(label)

        stats_layout.addStretch()
        layout.addWidget(stats_group)

        # Filter-Leiste
        filter_layout = QHBoxLayout()

        filter_layout.addWidget(QLabel('Status:'))
        self.status_filter = QComboBox()
        self.status_filter.addItems(['Alle', 'Trained', 'Prepared'])
        self.status_filter.currentTextChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self.status_filter)

        filter_layout.addSpacing(20)

        filter_layout.addWidget(QLabel('Sortierung:'))
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(['Datum (neueste)', 'Datum (aelteste)', 'Accuracy (beste)', 'Accuracy (schlechteste)'])
        self.sort_combo.currentTextChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self.sort_combo)

        filter_layout.addStretch()

        self.refresh_btn = QPushButton('Aktualisieren')
        self.refresh_btn.clicked.connect(self._refresh)
        filter_layout.addWidget(self.refresh_btn)

        layout.addLayout(filter_layout)

        # Tabelle
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            'Session', 'Status', 'Features', 'Samples', 'Accuracy', 'Erstellt', 'Aktionen'
        ])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)

        # Buttons unten
        btn_layout = QHBoxLayout()

        self.migrate_btn = QPushButton('Migration ausfuehren')
        self.migrate_btn.setToolTip('Scannt log/-Ordner nach neuen Sessions und fuegt sie zur DB hinzu')
        self.migrate_btn.clicked.connect(self._run_migration)
        btn_layout.addWidget(self.migrate_btn)

        self.delete_selected_btn = QPushButton('Ausgewaehlte loeschen')
        self.delete_selected_btn.setStyleSheet('''
            QPushButton {
                background-color: #742a2a;
            }
            QPushButton:hover {
                background-color: #943a3a;
            }
        ''')
        self.delete_selected_btn.clicked.connect(self._delete_selected)
        btn_layout.addWidget(self.delete_selected_btn)

        btn_layout.addStretch()

        self.close_btn = QPushButton('Schliessen')
        self.close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self.close_btn)

        layout.addLayout(btn_layout)

    def _load_sessions(self):
        """Laedt alle Sessions aus der DB."""
        # Filter ermitteln
        status_filter = self.status_filter.currentText().lower()
        if status_filter == 'alle':
            status_filter = None

        # Sessions laden
        sessions = self.db.list_sessions(status=status_filter)

        # Sortierung anwenden
        sort_mode = self.sort_combo.currentText()
        if sort_mode == 'Datum (neueste)':
            sessions = sorted(sessions, key=lambda x: x.get('created_at', ''), reverse=True)
        elif sort_mode == 'Datum (aelteste)':
            sessions = sorted(sessions, key=lambda x: x.get('created_at', ''))
        elif sort_mode == 'Accuracy (beste)':
            sessions = sorted(sessions, key=lambda x: x.get('model_accuracy', 0), reverse=True)
        elif sort_mode == 'Accuracy (schlechteste)':
            sessions = sorted(sessions, key=lambda x: x.get('model_accuracy', 0))

        # Tabelle fuellen
        self.table.setRowCount(len(sessions))
        self._sessions = sessions

        for row, session in enumerate(sessions):
            # Session-Name
            name_item = QTableWidgetItem(session.get('id', '-'))
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, name_item)

            # Status
            status = session.get('status', '-')
            status_item = QTableWidgetItem(status.capitalize() if status else '-')
            status_item.setFlags(status_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if status == 'trained':
                status_item.setForeground(QColor('#33b34d'))
            elif status == 'prepared':
                status_item.setForeground(QColor('#e6b333'))
            self.table.setItem(row, 1, status_item)

            # Features
            num_features = session.get('num_features', 0)
            features_item = QTableWidgetItem(str(num_features))
            features_item.setFlags(features_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            features_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 2, features_item)

            # Samples
            num_samples = session.get('num_samples', 0)
            samples_item = QTableWidgetItem(f'{num_samples:,}' if num_samples else '-')
            samples_item.setFlags(samples_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            samples_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 3, samples_item)

            # Accuracy
            acc = session.get('model_accuracy', 0)
            acc_text = f'{acc:.1f}%' if acc > 0 else '-'
            acc_item = QTableWidgetItem(acc_text)
            acc_item.setFlags(acc_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            acc_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if acc >= 70:
                acc_item.setForeground(QColor('#33b34d'))
            elif acc >= 60:
                acc_item.setForeground(QColor('#e6b333'))
            self.table.setItem(row, 4, acc_item)

            # Erstellt
            created = session.get('created_at', '')
            if created:
                # ISO-Format kuerzen
                created = created.replace('T', ' ')[:16]
            created_item = QTableWidgetItem(created)
            created_item.setFlags(created_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            created_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 5, created_item)

            # Aktionen-Button
            delete_btn = QPushButton('Loeschen')
            delete_btn.setStyleSheet('''
                QPushButton {
                    background-color: #5c2a2a;
                    padding: 3px 8px;
                    font-size: 11px;
                }
                QPushButton:hover {
                    background-color: #7c3a3a;
                }
            ''')
            delete_btn.clicked.connect(lambda checked, r=row: self._delete_session(r))
            self.table.setCellWidget(row, 6, delete_btn)

        if not sessions:
            self.table.setRowCount(1)
            item = QTableWidgetItem('Keine Sessions gefunden')
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setForeground(Qt.GlobalColor.gray)
            self.table.setItem(0, 0, item)

    def _update_statistics(self):
        """Aktualisiert die Statistik-Anzeige."""
        stats = self.db.get_statistics()

        self.stats_total.setText(f'Gesamt: {stats["total_sessions"]}')
        self.stats_trained.setText(f'Trained: {stats["trained_sessions"]}')
        self.stats_prepared.setText(f'Prepared: {stats["prepared_sessions"]}')

        if stats['avg_accuracy'] > 0:
            self.stats_avg_acc.setText(f'Avg Accuracy: {stats["avg_accuracy"]:.1f}%')
            self.stats_avg_acc.setStyleSheet('color: #33b34d; padding: 5px 15px;')
        else:
            self.stats_avg_acc.setText('Avg Accuracy: -')
            self.stats_avg_acc.setStyleSheet('color: #aaa; padding: 5px 15px;')

        if stats['max_accuracy'] > 0:
            self.stats_max_acc.setText(f'Max Accuracy: {stats["max_accuracy"]:.1f}%')
            self.stats_max_acc.setStyleSheet('color: #33b34d; padding: 5px 15px;')
        else:
            self.stats_max_acc.setText('Max Accuracy: -')
            self.stats_max_acc.setStyleSheet('color: #aaa; padding: 5px 15px;')

    def _on_filter_changed(self):
        """Wird aufgerufen wenn Filter geaendert wird."""
        self._load_sessions()

    def _refresh(self):
        """Aktualisiert die Anzeige."""
        self._load_sessions()
        self._update_statistics()

    def _run_migration(self):
        """Fuehrt die Migration von Session-Ordnern durch."""
        count = self.db.migrate_from_folders(self.log_dir)

        if count > 0:
            QMessageBox.information(
                self, 'Migration',
                f'{count} Session(s) wurden zur Datenbank hinzugefuegt.'
            )
        else:
            QMessageBox.information(
                self, 'Migration',
                'Keine neuen Sessions gefunden.'
            )

        self._refresh()

    def _delete_session(self, row: int):
        """Loescht eine einzelne Session."""
        if row < 0 or row >= len(self._sessions):
            return

        session = self._sessions[row]
        session_id = session.get('id', '')
        session_path = session.get('path', '')

        # Bestaetigung
        reply = QMessageBox.question(
            self, 'Session loeschen',
            f'Session "{session_id}" wirklich loeschen?\n\n'
            f'Dies entfernt die Session aus der Datenbank.\n'
            f'Der Ordner auf der Festplatte bleibt erhalten.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.db.delete_session(session_id)
            self._refresh()

    def _delete_selected(self):
        """Loescht die ausgewaehlte Session."""
        row = self.table.currentRow()
        if row >= 0:
            self._delete_session(row)
        else:
            QMessageBox.warning(self, 'Hinweis', 'Bitte zuerst eine Session auswaehlen.')
