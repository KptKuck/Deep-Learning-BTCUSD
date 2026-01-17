"""
Profiling Dialog - Zeigt cProfile-Ergebnisse in einer interaktiven Tabelle.

Features:
- Sortierbare Tabelle nach allen Spalten
- Export als .prof Datei (fuer snakeviz, etc.)
- Farbige Hervorhebung von langsamen Funktionen
"""

import pstats
import io
from pathlib import Path
from typing import Optional
from datetime import datetime

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QHeaderView, QFileDialog, QApplication,
    QAbstractItemView, QGroupBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont


class ProfilingDialog(QDialog):
    """
    Dialog zur Anzeige von cProfile-Ergebnissen.

    Zeigt eine sortierbare Tabelle mit:
    - Funktionsname
    - Anzahl Aufrufe
    - Totale Zeit (nur diese Funktion)
    - Kumulative Zeit (inkl. Unterfunktionen)
    - Zeit pro Aufruf
    """

    def __init__(self, profiler, parent=None, title: str = "Profiling Ergebnisse"):
        super().__init__(parent)
        self.profiler = profiler
        self.stats_data = []

        self._init_ui(title)
        self._load_stats()

    def _init_ui(self, title: str):
        """Initialisiert die UI."""
        self.setWindowTitle(f"4.5 - {title}")

        # Fenstergroesse relativ zum Bildschirm
        screen = QApplication.primaryScreen()
        if screen:
            screen_rect = screen.availableGeometry()
            window_width = int(screen_rect.width() * 0.7)
            window_height = int(screen_rect.height() * 0.8)
        else:
            window_width, window_height = 1000, 700

        self.setMinimumSize(800, 500)
        self.resize(window_width, window_height)

        self.setStyleSheet('''
            QDialog {
                background-color: #262626;
            }
            QLabel {
                color: #cccccc;
            }
            QTableWidget {
                background-color: #1e1e1e;
                color: #cccccc;
                gridline-color: #3a3a3a;
                selection-background-color: #3a5a7a;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QHeaderView::section {
                background-color: #3a3a3a;
                color: #ffffff;
                padding: 6px;
                border: 1px solid #4a4a4a;
                font-weight: bold;
            }
            QPushButton {
                background-color: #3a5a7a;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #4a6a8a;
            }
            QGroupBox {
                color: #4de6ff;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
            }
        ''')

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Header mit Zusammenfassung
        summary_group = QGroupBox("Zusammenfassung")
        summary_layout = QHBoxLayout(summary_group)

        self.total_calls_label = QLabel("Aufrufe: -")
        self.total_calls_label.setFont(QFont('Segoe UI', 10))
        summary_layout.addWidget(self.total_calls_label)

        self.total_time_label = QLabel("Gesamtzeit: -")
        self.total_time_label.setFont(QFont('Segoe UI', 10))
        summary_layout.addWidget(self.total_time_label)

        self.functions_label = QLabel("Funktionen: -")
        self.functions_label.setFont(QFont('Segoe UI', 10))
        summary_layout.addWidget(self.functions_label)

        summary_layout.addStretch()
        layout.addWidget(summary_group)

        # Tabelle
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            'Funktion', 'Datei', 'Aufrufe', 'Total (ms)', 'Kumulativ (ms)', 'ms/Aufruf'
        ])

        # Tabellen-Einstellungen
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)

        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setVisible(False)

        layout.addWidget(self.table)

        # Buttons
        btn_layout = QHBoxLayout()

        self.export_btn = QPushButton("Als .prof exportieren")
        self.export_btn.setToolTip("Speichert Profiling-Daten fuer snakeviz, etc.")
        self.export_btn.clicked.connect(self._export_prof)
        btn_layout.addWidget(self.export_btn)

        self.export_text_btn = QPushButton("Als Text exportieren")
        self.export_text_btn.clicked.connect(self._export_text)
        btn_layout.addWidget(self.export_text_btn)

        btn_layout.addStretch()

        self.snakeviz_hint = QLabel(
            "Tipp: pip install snakeviz && snakeviz datei.prof"
        )
        self.snakeviz_hint.setStyleSheet("color: #808080; font-size: 9px;")
        btn_layout.addWidget(self.snakeviz_hint)

        btn_layout.addStretch()

        self.close_btn = QPushButton("Schliessen")
        self.close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self.close_btn)

        layout.addLayout(btn_layout)

    def _load_stats(self):
        """Laedt die Profiling-Statistiken in die Tabelle."""
        if not self.profiler:
            return

        # Stats-Objekt erstellen
        stream = io.StringIO()
        stats = pstats.Stats(self.profiler, stream=stream)
        stats.strip_dirs()

        # Daten sammeln
        self.stats_data = []
        total_calls = 0
        total_time = 0.0

        for (filename, line, func), (cc, nc, tt, ct, callers) in stats.stats.items():
            total_calls += nc
            total_time += tt

            self.stats_data.append({
                'func': func,
                'file': f"{filename}:{line}",
                'calls': nc,
                'total_time': tt * 1000,  # In ms
                'cumulative_time': ct * 1000,  # In ms
                'per_call': (tt / nc * 1000) if nc > 0 else 0
            })

        # Nach kumulativer Zeit sortieren
        self.stats_data.sort(key=lambda x: x['cumulative_time'], reverse=True)

        # Zusammenfassung aktualisieren
        self.total_calls_label.setText(f"Aufrufe: {total_calls:,}")
        self.total_time_label.setText(f"Gesamtzeit: {total_time*1000:.1f} ms")
        self.functions_label.setText(f"Funktionen: {len(self.stats_data):,}")

        # Tabelle fuellen
        self.table.setRowCount(len(self.stats_data))

        # Maximale kumulative Zeit fuer Farbgebung
        max_cum_time = max((d['cumulative_time'] for d in self.stats_data), default=1)

        for row, data in enumerate(self.stats_data):
            # Funktionsname
            func_item = QTableWidgetItem(data['func'])
            self.table.setItem(row, 0, func_item)

            # Datei
            file_item = QTableWidgetItem(data['file'])
            file_item.setForeground(QColor('#808080'))
            self.table.setItem(row, 1, file_item)

            # Aufrufe
            calls_item = QTableWidgetItem()
            calls_item.setData(Qt.ItemDataRole.DisplayRole, data['calls'])
            calls_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, 2, calls_item)

            # Total Zeit
            total_item = QTableWidgetItem()
            total_item.setData(Qt.ItemDataRole.DisplayRole, round(data['total_time'], 2))
            total_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, 3, total_item)

            # Kumulative Zeit (mit Farbgebung)
            cum_item = QTableWidgetItem()
            cum_item.setData(Qt.ItemDataRole.DisplayRole, round(data['cumulative_time'], 2))
            cum_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            # Farbgebung: Rot fuer langsame Funktionen
            intensity = min(data['cumulative_time'] / max_cum_time, 1.0)
            if intensity > 0.5:
                # Rot-Orange fuer Top-Funktionen
                red = int(255 * min(intensity * 1.5, 1.0))
                green = int(128 * (1 - intensity))
                cum_item.setForeground(QColor(red, green, 50))
            elif intensity > 0.1:
                # Gelb fuer mittlere
                cum_item.setForeground(QColor(230, 180, 50))

            self.table.setItem(row, 4, cum_item)

            # Zeit pro Aufruf
            per_call_item = QTableWidgetItem()
            per_call_item.setData(Qt.ItemDataRole.DisplayRole, round(data['per_call'], 4))
            per_call_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, 5, per_call_item)

        # Nach kumulativer Zeit sortieren (Spalte 4, absteigend)
        self.table.sortItems(4, Qt.SortOrder.DescendingOrder)

    def _export_prof(self):
        """Exportiert die Profiling-Daten als .prof Datei."""
        if not self.profiler:
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_name = f"backtest_profile_{timestamp}.prof"

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Profiling exportieren",
            default_name,
            "Profile Dateien (*.prof)"
        )

        if filepath:
            self.profiler.dump_stats(filepath)
            # Auch im Log vermerken
            if self.parent():
                self.parent()._log(f"Profiling exportiert: {filepath}", 'SUCCESS')

    def _export_text(self):
        """Exportiert die Profiling-Daten als Text."""
        if not self.profiler:
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_name = f"backtest_profile_{timestamp}.txt"

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Profiling als Text exportieren",
            default_name,
            "Text Dateien (*.txt)"
        )

        if filepath:
            stream = io.StringIO()
            stats = pstats.Stats(self.profiler, stream=stream)
            stats.strip_dirs()
            stats.sort_stats('cumulative')
            stats.print_stats(100)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(stream.getvalue())

            if self.parent():
                self.parent()._log(f"Profiling-Text exportiert: {filepath}", 'SUCCESS')
