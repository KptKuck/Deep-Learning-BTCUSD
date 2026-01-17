"""
SessionManagerWindow - Fenster zur Verwaltung aller Sessions.

Bietet eine Uebersicht aller Sessions mit:
- Laden von Sessions
- Filterung nach Status (trained/prepared)
- Sortierung nach verschiedenen Kriterien
- Loeschen von Sessions
- Statistiken
- Manuelle Migration
- Ordner im Explorer oeffnen
"""

import os
import subprocess
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QComboBox, QGroupBox, QMessageBox, QWidget, QSplitter,
    QTextEdit, QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor

from ..core.session_database import SessionDatabase
from ..core.session_manager import SessionManager


class SessionManagerWindow(QDialog):
    """Fenster zur Verwaltung aller Sessions."""

    # Signal wenn eine Session geladen werden soll
    session_load_requested = pyqtSignal(str)  # session_path

    def __init__(self, data_dir: Path, log_dir: Path, parent=None, current_session: str = None):
        super().__init__(parent)
        self.data_dir = Path(data_dir)
        self.log_dir = Path(log_dir)
        self.db = SessionDatabase(self.data_dir)
        self.current_session = current_session  # Aktuell geladene Session
        self.selected_session_path: Optional[str] = None  # Fuer Rueckgabe

        self._init_ui()
        self._load_sessions()
        self._update_statistics()

    def _init_ui(self):
        """Initialisiert die UI."""
        self.setWindowTitle('1.1 Session Manager')

        # Fenstergroesse: Breite fix, Hoehe 95% des Bildschirms
        screen = QApplication.primaryScreen()
        if screen:
            screen_height = screen.availableGeometry().height()
            window_height = int(screen_height * 0.95)
        else:
            window_height = 900

        self.setMinimumSize(1100, 600)
        self.resize(1100, window_height)
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
            QTextEdit {
                background-color: #1a1a1a;
                color: #cccccc;
                border: 1px solid #333;
                font-family: Consolas, monospace;
                font-size: 11px;
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
        self.stats_storage = QLabel('Speicher: -')

        for label in [self.stats_total, self.stats_trained, self.stats_prepared,
                      self.stats_avg_acc, self.stats_max_acc, self.stats_storage]:
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

        # Splitter fuer Tabelle und Details (vertikal)
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Tabelle (oben)
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        table_layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            'Session', 'Status', 'Features', 'Samples', 'Accuracy', 'Groesse', 'Erstellt'
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
        self.table.doubleClicked.connect(self._on_double_click)
        self.table.selectionModel().selectionChanged.connect(self._on_selection_changed)
        table_layout.addWidget(self.table)

        splitter.addWidget(table_widget)

        # Details Panel (unten) - deutlich groesser
        details_widget = QWidget()
        details_layout = QVBoxLayout(details_widget)
        details_layout.setContentsMargins(0, 5, 0, 0)

        # Details Header mit Buttons
        details_header = QHBoxLayout()

        details_label = QLabel('Session Details')
        details_label.setFont(QFont('Segoe UI', 11, QFont.Weight.Bold))
        details_label.setStyleSheet('color: #4de6ff;')
        details_header.addWidget(details_label)

        details_header.addStretch()

        self.open_folder_btn = QPushButton('Ordner oeffnen')
        self.open_folder_btn.setToolTip('Oeffnet den Session-Ordner im Explorer')
        self.open_folder_btn.clicked.connect(self._open_folder)
        self.open_folder_btn.setEnabled(False)
        details_header.addWidget(self.open_folder_btn)

        self.delete_btn = QPushButton('Loeschen')
        self.delete_btn.setStyleSheet('''
            QPushButton {
                background-color: #5c2a2a;
            }
            QPushButton:hover {
                background-color: #7c3a3a;
            }
        ''')
        self.delete_btn.clicked.connect(self._delete_selected)
        self.delete_btn.setEnabled(False)
        details_header.addWidget(self.delete_btn)

        details_layout.addLayout(details_header)

        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setPlaceholderText('Session auswaehlen um Details anzuzeigen...')
        self.details_text.setMinimumHeight(200)
        details_layout.addWidget(self.details_text)

        splitter.addWidget(details_widget)

        # Splitter-Groessen setzen (45% Tabelle, 55% Details)
        splitter.setSizes([350, 450])

        layout.addWidget(splitter, 1)  # stretch=1 damit Splitter den Platz fuellt

        # Buttons unten
        btn_layout = QHBoxLayout()

        self.migrate_btn = QPushButton('Migration')
        self.migrate_btn.setToolTip('Scannt log/-Ordner nach neuen Sessions')
        self.migrate_btn.clicked.connect(self._run_migration)
        btn_layout.addWidget(self.migrate_btn)

        btn_layout.addStretch()

        self.load_btn = QPushButton('Session laden')
        self.load_btn.setStyleSheet('''
            QPushButton {
                background-color: #3a7a5a;
                padding: 8px 20px;
            }
            QPushButton:hover {
                background-color: #4a9a6a;
            }
            QPushButton:disabled {
                background-color: #333;
            }
        ''')
        self.load_btn.clicked.connect(self._load_selected_session)
        self.load_btn.setEnabled(False)
        btn_layout.addWidget(self.load_btn)

        self.close_btn = QPushButton('Schliessen')
        self.close_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.close_btn)

        layout.addLayout(btn_layout)

    def _get_folder_size(self, path: str) -> int:
        """
        Berechnet die Groesse eines Ordners in Bytes.

        Args:
            path: Pfad zum Ordner

        Returns:
            Groesse in Bytes, 0 bei Fehler
        """
        try:
            folder = Path(path)
            if not folder.exists():
                return 0
            total = 0
            for f in folder.rglob('*'):
                if f.is_file():
                    total += f.stat().st_size
            return total
        except Exception:
            return 0

    def _format_size(self, size_bytes: int) -> str:
        """
        Formatiert Bytes in lesbare Groesse.

        Args:
            size_bytes: Groesse in Bytes

        Returns:
            Formatierter String (z.B. "12.5 MB")
        """
        if size_bytes == 0:
            return '-'
        elif size_bytes < 1024:
            return f'{size_bytes} B'
        elif size_bytes < 1024 * 1024:
            return f'{size_bytes / 1024:.1f} KB'
        elif size_bytes < 1024 * 1024 * 1024:
            return f'{size_bytes / (1024 * 1024):.1f} MB'
        else:
            return f'{size_bytes / (1024 * 1024 * 1024):.2f} GB'

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
            session_id = session.get('id', '-')
            is_current = (session_id == self.current_session)

            # Session-Name (mit Markierung fuer aktive Session)
            name_text = session_id
            if is_current:
                name_text = f"* {session_id}"
            name_item = QTableWidgetItem(name_text)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if is_current:
                name_item.setForeground(QColor('#4de6ff'))
                name_item.setToolTip('Aktuell geladene Session')
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

            # Groesse
            session_path = session.get('path', '')
            folder_size = self._get_folder_size(session_path)
            size_text = self._format_size(folder_size)
            size_item = QTableWidgetItem(size_text)
            size_item.setFlags(size_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            size_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            # Grosse Sessions (>50MB) orange markieren
            if folder_size > 50 * 1024 * 1024:
                size_item.setForeground(QColor('#e6b333'))
            self.table.setItem(row, 5, size_item)

            # Erstellt
            created = session.get('created_at', '')
            if created:
                created = created.replace('T', ' ')[:16]
            created_item = QTableWidgetItem(created)
            created_item.setFlags(created_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            created_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 6, created_item)

        if not sessions:
            self.table.setRowCount(1)
            item = QTableWidgetItem('Keine Sessions gefunden')
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setForeground(Qt.GlobalColor.gray)
            self.table.setItem(0, 0, item)

        # Details zuruecksetzen
        self.details_text.clear()
        self._update_button_states()

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

        # Gesamtspeicher berechnen
        total_storage = 0
        all_sessions = self.db.list_sessions()
        for session in all_sessions:
            session_path = session.get('path', '')
            if session_path:
                total_storage += self._get_folder_size(session_path)

        if total_storage > 0:
            storage_color = '#e6b333' if total_storage > 500 * 1024 * 1024 else '#aaa'
            self.stats_storage.setText(f'Speicher: {self._format_size(total_storage)}')
            self.stats_storage.setStyleSheet(f'color: {storage_color}; padding: 5px 15px;')
        else:
            self.stats_storage.setText('Speicher: -')
            self.stats_storage.setStyleSheet('color: #aaa; padding: 5px 15px;')

    def _on_filter_changed(self):
        """Wird aufgerufen wenn Filter geaendert wird."""
        self._load_sessions()

    def _on_selection_changed(self):
        """Wird aufgerufen wenn Auswahl sich aendert."""
        self._update_button_states()
        self._update_details()

    def _update_button_states(self):
        """Aktualisiert den Zustand der Buttons."""
        row = self.table.currentRow()
        has_selection = row >= 0 and row < len(self._sessions) if hasattr(self, '_sessions') else False

        self.load_btn.setEnabled(has_selection)
        self.delete_btn.setEnabled(has_selection)
        self.open_folder_btn.setEnabled(has_selection)

    def _update_details(self):
        """Aktualisiert das Details-Panel."""
        row = self.table.currentRow()
        if row < 0 or not hasattr(self, '_sessions') or row >= len(self._sessions):
            self.details_text.clear()
            return

        session = self._sessions[row]

        # Details formatieren
        details = []
        details.append(f"<b style='color: #4de6ff;'>Session:</b> {session.get('id', '-')}")
        details.append(f"<b>Status:</b> {session.get('status', '-').capitalize()}")
        details.append(f"<b>Pfad:</b> {session.get('path', '-')}")
        details.append("")

        # Features
        features = session.get('features', [])
        if features:
            details.append(f"<b style='color: #4de6ff;'>Features ({len(features)}):</b>")
            for f in features:
                details.append(f"  - {f}")
        else:
            details.append(f"<b>Features:</b> {session.get('num_features', 0)}")

        details.append("")
        details.append(f"<b>Samples:</b> {session.get('num_samples', 0):,}")

        # Modell-Info
        acc = session.get('model_accuracy', 0)
        if acc > 0:
            details.append("")
            details.append(f"<b style='color: #4de6ff;'>Modell:</b>")
            details.append(f"  Accuracy: <span style='color: #33b34d;'>{acc:.2f}%</span>")
            details.append(f"  Version: {session.get('model_version', '-')}")
            details.append(f"  Typ: {session.get('model_type', '-')}")

        # Daten-Status
        details.append("")
        details.append(f"<b style='color: #4de6ff;'>Daten:</b>")
        details.append(f"  Training-Daten: {'Ja' if session.get('has_training_data') else 'Nein'}")
        details.append(f"  Backtest-Daten: {'Ja' if session.get('has_backtest_data') else 'Nein'}")
        details.append(f"  Modell: {'Ja' if session.get('has_model') else 'Nein'}")

        # Speicherplatz
        session_path = session.get('path', '')
        folder_size = self._get_folder_size(session_path)
        if folder_size > 0:
            details.append("")
            details.append(f"<b style='color: #4de6ff;'>Speicher:</b>")
            size_color = '#e6b333' if folder_size > 50 * 1024 * 1024 else '#aaa'
            details.append(f"  Ordnergroesse: <span style='color: {size_color};'>{self._format_size(folder_size)}</span>")

        # Zeitstempel
        details.append("")
        created = session.get('created_at', '')
        updated = session.get('updated_at', '')
        if created:
            details.append(f"<b>Erstellt:</b> {created.replace('T', ' ')[:19]}")
        if updated and updated != created:
            details.append(f"<b>Aktualisiert:</b> {updated.replace('T', ' ')[:19]}")

        self.details_text.setHtml('<br>'.join(details))

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

    def _on_double_click(self):
        """Doppelklick laedt die Session."""
        self._load_selected_session()

    def _load_selected_session(self):
        """Laedt die ausgewaehlte Session."""
        row = self.table.currentRow()
        if row < 0 or row >= len(self._sessions):
            return

        session = self._sessions[row]
        session_path = session.get('path', '')

        if not session_path:
            QMessageBox.warning(self, 'Fehler', 'Session-Pfad nicht gefunden.')
            return

        # Pfad speichern und Dialog schliessen
        self.selected_session_path = session_path
        self.accept()

    def _open_folder(self):
        """Oeffnet den Session-Ordner im Explorer."""
        row = self.table.currentRow()
        if row < 0 or row >= len(self._sessions):
            return

        session = self._sessions[row]
        session_path = session.get('path', '')

        if session_path and Path(session_path).exists():
            # Windows Explorer oeffnen
            subprocess.Popen(['explorer', session_path])
        else:
            QMessageBox.warning(self, 'Fehler', f'Ordner existiert nicht:\n{session_path}')

    def _delete_selected(self):
        """Loescht die ausgewaehlte Session."""
        row = self.table.currentRow()
        if row < 0 or row >= len(self._sessions):
            QMessageBox.warning(self, 'Hinweis', 'Bitte zuerst eine Session auswaehlen.')
            return

        session = self._sessions[row]
        session_id = session.get('id', '')

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

    def get_selected_session(self) -> Optional[str]:
        """Gibt den Pfad der ausgewaehlten Session zurueck."""
        return self.selected_session_path
