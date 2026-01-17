"""
SaveConfirmDialog - Dialog zur Bestaetigung beim Ueberschreiben

Zeigt dem Benutzer:
- Was existiert bereits
- Was sich aendern wuerde
- Warnungen (z.B. Modell wird ungueltig)

Optionen:
- Abbrechen: Nichts tun
- Neue Session: Neue Session erstellen statt ueberschreiben
- Ueberschreiben: Bestehende Daten ersetzen
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QGroupBox,
    QStyle,
)

from ..core.save_manager import OverwriteAction, SaveCheckResult
from .styles import COLORS, StyleFactory


class SaveConfirmDialog(QDialog):
    """
    Dialog zur Bestaetigung beim Ueberschreiben von Session-Daten.

    Zeigt:
    - Was existiert bereits (existing_data)
    - Was sich aendern wuerde (changes)
    - Warnungen (warnings)

    Gibt zurueck:
    - OverwriteAction.CANCEL: Dialog abgebrochen
    - OverwriteAction.OVERWRITE: Ueberschreiben bestaetigt
    - OverwriteAction.NEW_SESSION: Neue Session erstellen
    """

    def __init__(self, check_result: SaveCheckResult, parent=None):
        """
        Initialisiert den Dialog.

        Args:
            check_result: Ergebnis der SaveManager-Pruefung
            parent: Parent-Widget
        """
        super().__init__(parent)
        self.check_result = check_result  # Nicht 'result' - das ist eine QDialog-Methode!
        self.action = OverwriteAction.CANCEL

        self._init_ui()

    def _init_ui(self):
        """Initialisiert die UI-Komponenten."""
        self.setWindowTitle("Speichern bestaetigen")
        self.setMinimumWidth(450)
        self.setModal(True)

        # Dark Theme
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {COLORS['bg_primary']};
                color: {COLORS['text_primary']};
            }}
            QLabel {{
                color: {COLORS['text_primary']};
            }}
            QGroupBox {{
                background-color: {COLORS['bg_tertiary']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                margin-top: 12px;
                padding-top: 10px;
                font-weight: bold;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: {COLORS['accent']};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Header mit Warnung-Icon
        header_layout = QHBoxLayout()

        # Icon
        icon_label = QLabel()
        style = self.style()
        if style is not None:
            icon = style.standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning)
            icon_label.setPixmap(icon.pixmap(32, 32))
        header_layout.addWidget(icon_label)

        # Titel
        title_label = QLabel("<b>Daten existieren bereits!</b>")
        title_label.setStyleSheet(f"""
            font-size: 16px;
            color: {COLORS['warning']};
        """)
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        layout.addLayout(header_layout)

        # Vorhandene Daten
        if self.check_result.existing_data:
            existing_group = QGroupBox("Vorhandene Daten:")
            existing_layout = QVBoxLayout(existing_group)

            for item in self.check_result.existing_data:
                item_label = QLabel(f"  - {item}")
                item_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
                existing_layout.addWidget(item_label)

            layout.addWidget(existing_group)

        # Aenderungen
        if self.check_result.changes:
            changes_group = QGroupBox("Aenderungen:")
            changes_layout = QVBoxLayout(changes_group)

            for change in self.check_result.changes:
                change_label = QLabel(f"  - {change}")
                change_label.setStyleSheet(f"color: {COLORS['info']};")
                changes_layout.addWidget(change_label)

            layout.addWidget(changes_group)

        # Warnungen (rot hervorgehoben)
        if self.check_result.warnings:
            for warning in self.check_result.warnings:
                warning_label = QLabel(f"! {warning}")
                warning_label.setStyleSheet(f"""
                    color: {COLORS['error']};
                    font-weight: bold;
                    padding: 8px;
                    background-color: rgba(204, 77, 51, 0.2);
                    border-radius: 4px;
                """)
                warning_label.setWordWrap(True)
                layout.addWidget(warning_label)

        # Spacer
        layout.addSpacing(20)

        # Buttons
        button_layout = QHBoxLayout()

        # Abbrechen (links)
        btn_cancel = QPushButton("Abbrechen")
        btn_cancel.setStyleSheet(StyleFactory.button_style((0.4, 0.4, 0.4)))
        btn_cancel.clicked.connect(self._on_cancel)
        button_layout.addWidget(btn_cancel)

        button_layout.addStretch()

        # Neue Session (mitte-rechts)
        btn_new = QPushButton("Neue Session")
        btn_new.setStyleSheet(StyleFactory.button_style((0.3, 0.6, 0.9)))
        btn_new.setToolTip("Erstellt eine neue Session statt zu ueberschreiben")
        btn_new.clicked.connect(self._on_new_session)
        button_layout.addWidget(btn_new)

        # Ueberschreiben (rechts, rot)
        btn_overwrite = QPushButton("Ueberschreiben")
        btn_overwrite.setStyleSheet(StyleFactory.button_style((0.8, 0.3, 0.2)))
        btn_overwrite.setToolTip("Ersetzt die vorhandenen Daten")
        btn_overwrite.clicked.connect(self._on_overwrite)
        button_layout.addWidget(btn_overwrite)

        layout.addLayout(button_layout)

    def _on_cancel(self):
        """Handler fuer Abbrechen-Button."""
        self.action = OverwriteAction.CANCEL
        self.reject()

    def _on_new_session(self):
        """Handler fuer Neue-Session-Button."""
        self.action = OverwriteAction.NEW_SESSION
        self.accept()

    def _on_overwrite(self):
        """Handler fuer Ueberschreiben-Button."""
        self.action = OverwriteAction.OVERWRITE
        self.accept()

    def get_action(self) -> OverwriteAction:
        """Gibt die gewaehlte Aktion zurueck."""
        return self.action


def ask_save_confirmation(
    check_result: SaveCheckResult,
    parent=None
) -> OverwriteAction:
    """
    Hilfsfunktion zum Anzeigen des Dialogs.

    Args:
        check_result: Ergebnis der SaveManager-Pruefung
        parent: Parent-Widget

    Returns:
        OverwriteAction mit der Benutzer-Entscheidung
    """
    dialog = SaveConfirmDialog(check_result, parent)
    dialog.exec()
    return dialog.get_action()
