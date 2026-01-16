"""
TimeRange Dialog - Dialog zur Auswahl eines neuen Zeitraums fuer den Backtest.
"""

from typing import Optional, Tuple
from datetime import datetime, timedelta

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QDateTimeEdit,
    QFileDialog, QComboBox, QMessageBox, QWidget
)
from PyQt6.QtCore import Qt, QDateTime
from PyQt6.QtGui import QFont

import pandas as pd

from ..styles import StyleFactory


class TimeRangeDialog(QDialog):
    """
    Dialog zur Auswahl eines neuen Zeitraums fuer den Backtest.

    Ermoeglicht:
    - Auswahl einer anderen CSV-Datei
    - Auswahl von Start- und Enddatum
    - Vorschau der verfuegbaren Daten
    """

    def __init__(self, parent=None, current_file: str = None,
                 current_data: pd.DataFrame = None):
        super().__init__(parent)
        self.setWindowTitle("Zeitraum aendern")
        self.setMinimumSize(500, 400)
        self.setModal(True)

        # Aktuelle Werte
        self.current_file = current_file
        self.selected_file = current_file
        self.loaded_data: Optional[pd.DataFrame] = None
        self.data_start: Optional[datetime] = None
        self.data_end: Optional[datetime] = None

        # Ergebnis
        self.result_data: Optional[pd.DataFrame] = None
        self.result_start: Optional[datetime] = None
        self.result_end: Optional[datetime] = None

        self._setup_ui()
        self.setStyleSheet(self._dialog_style())

        # Session-Daten als Default verwenden
        if current_data is not None and len(current_data) > 0:
            self._use_session_data(current_data)
        elif current_file:
            self._load_file(current_file)

    def _setup_ui(self):
        """Erstellt die UI-Komponenten."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Titel
        title = QLabel("Neuen Zeitraum waehlen")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Datei-Auswahl
        layout.addWidget(self._create_file_group())

        # Zeitraum-Auswahl
        layout.addWidget(self._create_timerange_group())

        # Info-Anzeige
        layout.addWidget(self._create_info_group())

        # Buttons
        layout.addWidget(self._create_button_row())

    def _create_file_group(self) -> QGroupBox:
        """Erstellt die Datei-Auswahl Gruppe."""
        group = QGroupBox("Datenquelle")
        group.setStyleSheet(StyleFactory.group_style(title_color=(0.3, 0.7, 1.0)))
        layout = QHBoxLayout(group)

        self.file_label = QLabel(self.current_file or "Keine Datei ausgewaehlt")
        self.file_label.setStyleSheet("color: #aaa;")
        self.file_label.setWordWrap(True)
        layout.addWidget(self.file_label, stretch=1)

        browse_btn = QPushButton("Durchsuchen...")
        browse_btn.setStyleSheet(StyleFactory.button_style((0.4, 0.6, 0.8)))
        browse_btn.clicked.connect(self._browse_file)
        layout.addWidget(browse_btn)

        return group

    def _create_timerange_group(self) -> QGroupBox:
        """Erstellt die Zeitraum-Auswahl Gruppe."""
        group = QGroupBox("Zeitraum")
        group.setStyleSheet(StyleFactory.group_style(title_color=(0.9, 0.7, 0.3)))
        layout = QGridLayout(group)
        layout.setSpacing(10)

        # Start-Datum
        layout.addWidget(QLabel("Start:"), 0, 0)
        self.start_edit = QDateTimeEdit()
        self.start_edit.setDisplayFormat("dd.MM.yyyy HH:mm")
        self.start_edit.setCalendarPopup(True)
        self.start_edit.setStyleSheet(self._datetime_style())
        self.start_edit.dateTimeChanged.connect(self._on_range_changed)
        layout.addWidget(self.start_edit, 0, 1)

        # Ende-Datum
        layout.addWidget(QLabel("Ende:"), 1, 0)
        self.end_edit = QDateTimeEdit()
        self.end_edit.setDisplayFormat("dd.MM.yyyy HH:mm")
        self.end_edit.setCalendarPopup(True)
        self.end_edit.setStyleSheet(self._datetime_style())
        self.end_edit.dateTimeChanged.connect(self._on_range_changed)
        layout.addWidget(self.end_edit, 1, 1)

        # Schnellauswahl
        layout.addWidget(QLabel("Schnellauswahl:"), 2, 0)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Gesamter Zeitraum",
            "Letzte Woche",
            "Letzter Monat",
            "Letzte 3 Monate",
            "Letztes Jahr",
            "Erste Haelfte",
            "Zweite Haelfte"
        ])
        self.preset_combo.setStyleSheet(self._combo_style())
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        layout.addWidget(self.preset_combo, 2, 1)

        return group

    def _create_info_group(self) -> QGroupBox:
        """Erstellt die Info-Anzeige Gruppe."""
        group = QGroupBox("Daten-Info")
        group.setStyleSheet(StyleFactory.group_style(title_color=(0.5, 0.9, 0.5)))
        layout = QGridLayout(group)
        layout.setColumnStretch(1, 1)

        info_labels = [
            ("Verfuegbar:", "available_label", "-"),
            ("Ausgewaehlt:", "selected_label", "-"),
            ("Datenpunkte:", "datapoints_label", "-"),
            ("Preisbereich:", "price_range_label", "-"),
        ]

        for row, (text, attr, default) in enumerate(info_labels):
            layout.addWidget(QLabel(text), row, 0)
            label = QLabel(default)
            label.setStyleSheet("color: white;")
            setattr(self, attr, label)
            layout.addWidget(label, row, 1)

        return group

    def _create_button_row(self) -> QWidget:
        """Erstellt die Button-Zeile."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 10, 0, 0)

        # Abbrechen
        cancel_btn = QPushButton("Abbrechen")
        cancel_btn.setStyleSheet(StyleFactory.button_style((0.5, 0.5, 0.5)))
        cancel_btn.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        cancel_btn.clicked.connect(self.reject)
        layout.addWidget(cancel_btn)

        layout.addStretch()

        # Uebernehmen
        self.apply_btn = QPushButton("Uebernehmen")
        self.apply_btn.setStyleSheet(StyleFactory.button_style((0.2, 0.7, 0.3)))
        self.apply_btn.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.apply_btn.clicked.connect(self._apply)
        self.apply_btn.setEnabled(False)
        layout.addWidget(self.apply_btn)

        return widget

    def _browse_file(self):
        """Oeffnet den Datei-Dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "CSV-Datei auswaehlen",
            "",
            "CSV-Dateien (*.csv);;Alle Dateien (*.*)"
        )
        if file_path:
            self._load_file(file_path)

    def _use_session_data(self, df: pd.DataFrame):
        """Verwendet die aktuellen Session-Daten als Ausgangspunkt."""
        self.loaded_data = df

        # Datums-Bereich ermitteln
        if hasattr(df.index, 'min') and hasattr(df.index[0], 'year'):
            self.data_start = df.index.min().to_pydatetime()
            self.data_end = df.index.max().to_pydatetime()
        elif 'DateTime' in df.columns:
            dt_col = pd.to_datetime(df['DateTime'])
            self.data_start = dt_col.min().to_pydatetime()
            self.data_end = dt_col.max().to_pydatetime()
        else:
            # Fallback
            self.data_start = datetime(2020, 1, 1)
            self.data_end = datetime(2020, 1, 1) + timedelta(hours=len(df))

        # UI aktualisieren
        if self.current_file:
            self.file_label.setText(self.current_file.split('/')[-1].split('\\')[-1])
            self.file_label.setToolTip(self.current_file)
        else:
            self.file_label.setText("Session-Daten (kein Dateipfad)")

        # DateTimeEdit-Bereich setzen
        self.start_edit.setDateTimeRange(
            QDateTime(self.data_start),
            QDateTime(self.data_end)
        )
        self.end_edit.setDateTimeRange(
            QDateTime(self.data_start),
            QDateTime(self.data_end)
        )

        # Auf aktuellen Zeitraum setzen (gesamte Session)
        self.start_edit.setDateTime(QDateTime(self.data_start))
        self.end_edit.setDateTime(QDateTime(self.data_end))

        # Info aktualisieren
        self._update_info()
        self.apply_btn.setEnabled(True)

    def _load_file(self, file_path: str):
        """Laedt eine CSV-Datei und aktualisiert die UI."""
        try:
            from ...data.reader import CSVReader

            reader = CSVReader()
            df = reader.read(file_path)

            if df is None or len(df) == 0:
                QMessageBox.warning(self, "Fehler", "Datei konnte nicht geladen werden.")
                return

            self.selected_file = file_path
            self.loaded_data = df

            # Datums-Bereich ermitteln
            if hasattr(df.index, 'min') and hasattr(df.index[0], 'year'):
                self.data_start = df.index.min().to_pydatetime()
                self.data_end = df.index.max().to_pydatetime()
            elif 'DateTime' in df.columns:
                dt_col = pd.to_datetime(df['DateTime'])
                self.data_start = dt_col.min().to_pydatetime()
                self.data_end = dt_col.max().to_pydatetime()
            else:
                # Fallback: Index als Datenpunkte
                self.data_start = datetime(2020, 1, 1)
                self.data_end = datetime(2020, 1, 1) + timedelta(hours=len(df))

            # UI aktualisieren
            self.file_label.setText(file_path.split('/')[-1].split('\\')[-1])
            self.file_label.setToolTip(file_path)

            # DateTimeEdit-Bereich setzen
            self.start_edit.setDateTimeRange(
                QDateTime(self.data_start),
                QDateTime(self.data_end)
            )
            self.end_edit.setDateTimeRange(
                QDateTime(self.data_start),
                QDateTime(self.data_end)
            )

            # Auf gesamten Zeitraum setzen
            self.start_edit.setDateTime(QDateTime(self.data_start))
            self.end_edit.setDateTime(QDateTime(self.data_end))

            # Info aktualisieren
            self._update_info()
            self.apply_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.warning(self, "Fehler", f"Fehler beim Laden: {e}")

    def _on_range_changed(self):
        """Handler fuer Zeitraum-Aenderungen."""
        self._update_info()

    def _on_preset_changed(self, index: int):
        """Handler fuer Schnellauswahl-Aenderungen."""
        if self.loaded_data is None or self.data_start is None or self.data_end is None:
            return

        total_seconds = (self.data_end - self.data_start).total_seconds()

        if index == 0:  # Gesamter Zeitraum
            start = self.data_start
            end = self.data_end
        elif index == 1:  # Letzte Woche
            end = self.data_end
            candidate = end - timedelta(days=7)
            start = max(self.data_start, candidate)
        elif index == 2:  # Letzter Monat
            end = self.data_end
            candidate = end - timedelta(days=30)
            start = max(self.data_start, candidate)
        elif index == 3:  # Letzte 3 Monate
            end = self.data_end
            candidate = end - timedelta(days=90)
            start = max(self.data_start, candidate)
        elif index == 4:  # Letztes Jahr
            end = self.data_end
            candidate = end - timedelta(days=365)
            start = max(self.data_start, candidate)
        elif index == 5:  # Erste Haelfte
            start = self.data_start
            end = self.data_start + timedelta(seconds=total_seconds / 2)
        elif index == 6:  # Zweite Haelfte
            start = self.data_start + timedelta(seconds=total_seconds / 2)
            end = self.data_end
        else:
            return

        self.start_edit.setDateTime(QDateTime(start))
        self.end_edit.setDateTime(QDateTime(end))

    def _update_info(self):
        """Aktualisiert die Info-Anzeige."""
        if self.loaded_data is None:
            return

        # Verfuegbarer Zeitraum
        self.available_label.setText(
            f"{self.data_start.strftime('%d.%m.%Y')} - {self.data_end.strftime('%d.%m.%Y')}"
        )

        # Ausgewaehlter Zeitraum
        start_dt = self.start_edit.dateTime().toPyDateTime()
        end_dt = self.end_edit.dateTime().toPyDateTime()
        self.selected_label.setText(
            f"{start_dt.strftime('%d.%m.%Y %H:%M')} - {end_dt.strftime('%d.%m.%Y %H:%M')}"
        )

        # Datenpunkte im ausgewaehlten Bereich
        filtered = self._filter_data(start_dt, end_dt)
        if filtered is not None:
            self.datapoints_label.setText(f"{len(filtered):,}")

            if 'Low' in filtered.columns and 'High' in filtered.columns:
                price_min = filtered['Low'].min()
                price_max = filtered['High'].max()
                self.price_range_label.setText(f"${price_min:,.2f} - ${price_max:,.2f}")
        else:
            self.datapoints_label.setText("0")
            self.price_range_label.setText("-")

    def _filter_data(self, start_dt: datetime, end_dt: datetime) -> Optional[pd.DataFrame]:
        """Filtert die Daten nach Zeitraum."""
        if self.loaded_data is None:
            return None

        df = self.loaded_data

        # Nach Index oder DateTime-Spalte filtern
        if hasattr(df.index, 'to_pydatetime'):
            mask = (df.index >= start_dt) & (df.index <= end_dt)
            return df[mask]
        elif 'DateTime' in df.columns:
            dt_col = pd.to_datetime(df['DateTime'])
            mask = (dt_col >= start_dt) & (dt_col <= end_dt)
            return df[mask]

        return df

    def _apply(self):
        """Wendet die Auswahl an."""
        if self.loaded_data is None:
            return

        start_dt = self.start_edit.dateTime().toPyDateTime()
        end_dt = self.end_edit.dateTime().toPyDateTime()

        filtered = self._filter_data(start_dt, end_dt)
        if filtered is None or len(filtered) == 0:
            QMessageBox.warning(self, "Fehler", "Keine Daten im ausgewaehlten Zeitraum.")
            return

        self.result_data = filtered.copy()
        self.result_start = start_dt
        self.result_end = end_dt

        self.accept()

    def get_result(self) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Gibt die ausgewaehlten Daten und den Dateipfad zurueck."""
        return self.result_data, self.selected_file

    def _dialog_style(self) -> str:
        """Gibt das Dialog-Stylesheet zurueck."""
        return """
            QDialog { background-color: #262626; }
            QLabel { color: #b3b3b3; }
            QGroupBox { color: white; }
        """

    def _datetime_style(self) -> str:
        """Gibt das DateTimeEdit-Stylesheet zurueck."""
        return """
            QDateTimeEdit {
                background-color: #333;
                color: white;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px;
                min-width: 180px;
            }
            QDateTimeEdit::drop-down {
                background-color: #444;
                border: none;
                width: 25px;
            }
            QDateTimeEdit::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #aaa;
            }
        """

    def _combo_style(self) -> str:
        """Gibt das ComboBox-Stylesheet zurueck."""
        return """
            QComboBox {
                background-color: #333;
                color: white;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px;
                min-width: 150px;
            }
            QComboBox::drop-down {
                background-color: #444;
                border: none;
                width: 25px;
            }
            QComboBox QAbstractItemView {
                background-color: #333;
                color: white;
                selection-background-color: #4da8da;
            }
        """
