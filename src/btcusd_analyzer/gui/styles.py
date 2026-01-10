"""
GUI Styles - Dark Theme fuer PyQt6
"""

# Dark Theme Farben (entsprechend MATLAB GUI)
COLORS = {
    'bg_primary': '#262626',      # Haupt-Hintergrund
    'bg_secondary': '#1a1a1a',    # Sekundaerer Hintergrund
    'bg_tertiary': '#333333',     # Tertiärer Hintergrund (Panels)

    'text_primary': '#ffffff',     # Primaerer Text
    'text_secondary': '#aaaaaa',   # Sekundaerer Text
    'text_disabled': '#666666',    # Deaktivierter Text

    'accent': '#4da8da',          # Akzentfarbe (Blau)
    'accent_hover': '#5db8ea',    # Akzent Hover

    'success': '#33b34d',         # Erfolg (Gruen)
    'warning': '#e6b333',         # Warnung (Orange)
    'error': '#cc4d33',           # Fehler (Rot)
    'neutral': '#808080',         # Neutral (Grau)

    'border': '#404040',          # Rahmenfarbe
    'border_focus': '#4da8da',    # Rahmen bei Fokus

    # Trading-Modi
    'testnet_bg': '#1a2e1a',
    'testnet_border': '#33cc33',
    'testnet_text': '#66ff66',

    'live_bg': '#3d1a1a',
    'live_border': '#ff3333',
    'live_text': '#ff6666',
}


def get_stylesheet() -> str:
    """Gibt das komplette Dark Theme Stylesheet zurueck."""
    return f"""
        /* === Globale Styles === */
        QMainWindow, QDialog, QWidget {{
            background-color: {COLORS['bg_primary']};
            color: {COLORS['text_primary']};
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 13px;
        }}

        /* === Labels === */
        QLabel {{
            color: {COLORS['text_primary']};
            padding: 2px;
        }}

        QLabel[class="header"] {{
            font-size: 16px;
            font-weight: bold;
            color: {COLORS['accent']};
        }}

        QLabel[class="secondary"] {{
            color: {COLORS['text_secondary']};
        }}

        /* === Buttons === */
        QPushButton {{
            background-color: {COLORS['bg_tertiary']};
            color: {COLORS['text_primary']};
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
            padding: 8px 16px;
            font-weight: bold;
            min-height: 30px;
        }}

        QPushButton:hover {{
            background-color: {COLORS['accent']};
            border-color: {COLORS['accent']};
        }}

        QPushButton:pressed {{
            background-color: {COLORS['accent_hover']};
        }}

        QPushButton:disabled {{
            background-color: {COLORS['bg_secondary']};
            color: {COLORS['text_disabled']};
            border-color: {COLORS['bg_secondary']};
        }}

        QPushButton[class="success"] {{
            background-color: {COLORS['success']};
            border-color: {COLORS['success']};
        }}

        QPushButton[class="warning"] {{
            background-color: {COLORS['warning']};
            border-color: {COLORS['warning']};
            color: #000000;
        }}

        QPushButton[class="danger"] {{
            background-color: {COLORS['error']};
            border-color: {COLORS['error']};
        }}

        /* === Input Fields === */
        QLineEdit, QTextEdit, QPlainTextEdit {{
            background-color: {COLORS['bg_secondary']};
            color: {COLORS['text_primary']};
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
            padding: 6px;
        }}

        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
            border-color: {COLORS['border_focus']};
        }}

        QLineEdit:disabled {{
            background-color: {COLORS['bg_tertiary']};
            color: {COLORS['text_disabled']};
        }}

        /* === Spin Boxes === */
        QSpinBox, QDoubleSpinBox {{
            background-color: {COLORS['bg_secondary']};
            color: {COLORS['text_primary']};
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
            padding: 4px;
        }}

        QSpinBox::up-button, QDoubleSpinBox::up-button,
        QSpinBox::down-button, QDoubleSpinBox::down-button {{
            background-color: {COLORS['bg_tertiary']};
            border: none;
            width: 20px;
        }}

        /* === Combo Boxes === */
        QComboBox {{
            background-color: {COLORS['bg_secondary']};
            color: {COLORS['text_primary']};
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
            padding: 6px;
            min-height: 30px;
        }}

        QComboBox:hover {{
            border-color: {COLORS['accent']};
        }}

        QComboBox::drop-down {{
            border: none;
            width: 30px;
        }}

        QComboBox QAbstractItemView {{
            background-color: {COLORS['bg_secondary']};
            color: {COLORS['text_primary']};
            selection-background-color: {COLORS['accent']};
            border: 1px solid {COLORS['border']};
        }}

        /* === Group Boxes === */
        QGroupBox {{
            background-color: {COLORS['bg_tertiary']};
            border: 1px solid {COLORS['border']};
            border-radius: 6px;
            margin-top: 12px;
            padding-top: 10px;
            font-weight: bold;
        }}

        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 0 5px;
            color: {COLORS['accent']};
        }}

        /* === Tab Widget === */
        QTabWidget::pane {{
            background-color: {COLORS['bg_tertiary']};
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
        }}

        QTabBar::tab {{
            background-color: {COLORS['bg_secondary']};
            color: {COLORS['text_secondary']};
            border: 1px solid {COLORS['border']};
            padding: 8px 16px;
            margin-right: 2px;
        }}

        QTabBar::tab:selected {{
            background-color: {COLORS['bg_tertiary']};
            color: {COLORS['text_primary']};
            border-bottom-color: {COLORS['bg_tertiary']};
        }}

        QTabBar::tab:hover:!selected {{
            background-color: {COLORS['bg_primary']};
        }}

        /* === Progress Bar === */
        QProgressBar {{
            background-color: {COLORS['bg_secondary']};
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
            text-align: center;
            color: {COLORS['text_primary']};
            height: 20px;
        }}

        QProgressBar::chunk {{
            background-color: {COLORS['accent']};
            border-radius: 3px;
        }}

        /* === Scroll Bars === */
        QScrollBar:vertical {{
            background-color: {COLORS['bg_secondary']};
            width: 12px;
            border-radius: 6px;
        }}

        QScrollBar::handle:vertical {{
            background-color: {COLORS['border']};
            border-radius: 6px;
            min-height: 30px;
        }}

        QScrollBar::handle:vertical:hover {{
            background-color: {COLORS['accent']};
        }}

        QScrollBar:horizontal {{
            background-color: {COLORS['bg_secondary']};
            height: 12px;
            border-radius: 6px;
        }}

        QScrollBar::handle:horizontal {{
            background-color: {COLORS['border']};
            border-radius: 6px;
            min-width: 30px;
        }}

        QScrollBar::add-line, QScrollBar::sub-line {{
            width: 0px;
            height: 0px;
        }}

        /* === Tables === */
        QTableWidget, QTableView {{
            background-color: {COLORS['bg_secondary']};
            color: {COLORS['text_primary']};
            gridline-color: {COLORS['border']};
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
        }}

        QTableWidget::item, QTableView::item {{
            padding: 4px;
        }}

        QTableWidget::item:selected, QTableView::item:selected {{
            background-color: {COLORS['accent']};
        }}

        QHeaderView::section {{
            background-color: {COLORS['bg_tertiary']};
            color: {COLORS['text_primary']};
            padding: 6px;
            border: none;
            border-right: 1px solid {COLORS['border']};
            border-bottom: 1px solid {COLORS['border']};
            font-weight: bold;
        }}

        /* === Splitter === */
        QSplitter::handle {{
            background-color: {COLORS['border']};
        }}

        QSplitter::handle:horizontal {{
            width: 2px;
        }}

        QSplitter::handle:vertical {{
            height: 2px;
        }}

        /* === Status Bar === */
        QStatusBar {{
            background-color: {COLORS['bg_secondary']};
            color: {COLORS['text_secondary']};
            border-top: 1px solid {COLORS['border']};
        }}

        /* === Menu Bar === */
        QMenuBar {{
            background-color: {COLORS['bg_secondary']};
            color: {COLORS['text_primary']};
            border-bottom: 1px solid {COLORS['border']};
        }}

        QMenuBar::item:selected {{
            background-color: {COLORS['accent']};
        }}

        QMenu {{
            background-color: {COLORS['bg_secondary']};
            color: {COLORS['text_primary']};
            border: 1px solid {COLORS['border']};
        }}

        QMenu::item:selected {{
            background-color: {COLORS['accent']};
        }}

        /* === Tool Tips === */
        QToolTip {{
            background-color: {COLORS['bg_secondary']};
            color: {COLORS['text_primary']};
            border: 1px solid {COLORS['border']};
            padding: 4px;
        }}

        /* === Checkboxes === */
        QCheckBox {{
            color: {COLORS['text_primary']};
            spacing: 8px;
        }}

        QCheckBox::indicator {{
            width: 18px;
            height: 18px;
            border: 1px solid {COLORS['border']};
            border-radius: 3px;
            background-color: {COLORS['bg_secondary']};
        }}

        QCheckBox::indicator:checked {{
            background-color: {COLORS['accent']};
            border-color: {COLORS['accent']};
        }}

        /* === Radio Buttons === */
        QRadioButton {{
            color: {COLORS['text_primary']};
            spacing: 8px;
        }}

        QRadioButton::indicator {{
            width: 18px;
            height: 18px;
            border: 1px solid {COLORS['border']};
            border-radius: 9px;
            background-color: {COLORS['bg_secondary']};
        }}

        QRadioButton::indicator:checked {{
            background-color: {COLORS['accent']};
            border-color: {COLORS['accent']};
        }}

        /* === Sliders === */
        QSlider::groove:horizontal {{
            background-color: {COLORS['bg_secondary']};
            height: 6px;
            border-radius: 3px;
        }}

        QSlider::handle:horizontal {{
            background-color: {COLORS['accent']};
            width: 16px;
            height: 16px;
            margin: -5px 0;
            border-radius: 8px;
        }}

        QSlider::handle:horizontal:hover {{
            background-color: {COLORS['accent_hover']};
        }}
    """


# Testnet/Live Mode spezifische Styles
TESTNET_STYLE = f"""
    background-color: {COLORS['testnet_bg']};
    border: 2px solid {COLORS['testnet_border']};
"""

LIVE_STYLE = f"""
    background-color: {COLORS['live_bg']};
    border: 2px solid {COLORS['live_border']};
"""

TESTNET_BANNER_STYLE = f"""
    background-color: #2d5a2d;
    color: {COLORS['testnet_text']};
    font-size: 16px;
    font-weight: bold;
    padding: 10px;
    border-radius: 4px;
"""

LIVE_BANNER_STYLE = f"""
    background-color: #8b0000;
    color: #ffffff;
    font-size: 18px;
    font-weight: bold;
    padding: 10px;
    border-radius: 4px;
"""
