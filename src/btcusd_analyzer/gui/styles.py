"""
GUI Styles - Dark Theme fuer PyQt6
"""

# Dark Theme Farben (entsprechend MATLAB GUI)
# MATLAB verwendet RGB-Werte im Bereich 0-1, hier umgerechnet auf Hex

COLORS = {
    # Hintergruende (MATLAB: [0.15, 0.15, 0.15], [0.18, 0.18, 0.18], [0.1, 0.1, 0.1])
    'bg_primary': '#262626',      # Haupt-Hintergrund (0.15 * 255 = 38)
    'bg_secondary': '#1a1a1a',    # Sekundaerer Hintergrund / Input (0.1 * 255 = 26)
    'bg_tertiary': '#2e2e2e',     # Panel-Hintergrund (0.18 * 255 = 46)
    'bg_panel': '#333333',        # GroupBox-Hintergrund (0.2 * 255 = 51)

    # Text
    'text_primary': '#ffffff',     # Primaerer Text
    'text_secondary': '#aaaaaa',   # Sekundaerer Text
    'text_disabled': '#666666',    # Deaktivierter Text

    # Akzent (MATLAB: [0.3, 0.6, 0.9] = Blau)
    'accent': '#4d99e6',          # Akzentfarbe (Blau)
    'accent_hover': '#5da9f6',    # Akzent Hover

    # Status-Farben (MATLAB)
    'success': '#33b34d',         # Erfolg/BUY (0.2, 0.7, 0.3)
    'warning': '#e6b333',         # Warnung (0.9, 0.7, 0.2)
    'error': '#cc4d33',           # Fehler/SELL (0.8, 0.3, 0.2)
    'neutral': '#808080',         # Neutral/HOLD (0.5, 0.5, 0.5)
    'info': '#33cccc',            # Info (0.2, 0.8, 0.8)
    'training': '#9933cc',        # Training (0.6, 0.2, 0.8)

    # Signal-Farben (Trading)
    'buy': '#33b34d',             # BUY Signal (Gruen)
    'sell': '#cc4d33',            # SELL Signal (Rot)
    'hold': '#808080',            # HOLD Signal (Grau)
    'long': '#33b34d',            # LONG Position (Gruen)
    'short': '#cc4d33',           # SHORT Position (Rot)

    # UI-Elemente
    'border': '#404040',          # Rahmenfarbe
    'border_focus': '#4d99e6',    # Rahmen bei Fokus
    'grid': '#4d4d4d',            # Chart-Grid

    # Trading-Modi
    'testnet_bg': '#1a2e1a',
    'testnet_border': '#33cc33',
    'testnet_text': '#66ff66',

    'live_bg': '#3d1a1a',
    'live_border': '#ff3333',
    'live_text': '#ff6666',
}

# MATLAB RGB-Farben als Tupel (0-1 Range) fuer direkte Verwendung
MATLAB_COLORS = {
    # Hintergruende
    'bg_main': (0.15, 0.15, 0.15),
    'bg_panel': (0.18, 0.18, 0.18),
    'bg_input': (0.1, 0.1, 0.1),
    'bg_group': (0.2, 0.2, 0.2),

    # Signale
    'buy': (0.2, 0.7, 0.3),
    'sell': (0.8, 0.3, 0.2),
    'hold': (0.5, 0.5, 0.5),

    # Buttons
    'success': (0.2, 0.7, 0.3),
    'primary': (0.3, 0.6, 0.9),
    'warning': (0.9, 0.7, 0.2),
    'error': (0.8, 0.3, 0.2),
    'neutral': (0.5, 0.5, 0.5),
    'info': (0.2, 0.8, 0.8),
    'training': (0.6, 0.2, 0.8),

    # Spezielle Button-Farben
    'step': (0.5, 0.5, 0.7),
    'close': (0.4, 0.4, 0.4),
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


# =============================================================================
# StyleFactory - Zentrale Style-Generierung
# =============================================================================

class StyleFactory:
    """
    Factory-Klasse fuer konsistente Style-Generierung.
    Ersetzt duplizierte _button_style() und _group_style() Methoden.
    """

    @staticmethod
    def rgb_to_hex(color: tuple) -> str:
        """Konvertiert RGB-Tuple (0-1 Range) zu Hex-String."""
        r, g, b = [int(c * 255) for c in color]
        return f'#{r:02x}{g:02x}{b:02x}'

    @staticmethod
    def hex_to_rgb(hex_color: str) -> tuple:
        """Konvertiert Hex-String zu RGB-Tuple (0-255)."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    @staticmethod
    def button_style(color: tuple, padding: str = '8px 12px') -> str:
        """
        Generiert Button-Stylesheet aus RGB-Tuple (0-1 Range).

        Args:
            color: RGB-Tuple im Bereich 0-1, z.B. (0.3, 0.6, 0.9)
            padding: CSS padding, default '8px 12px'

        Returns:
            Stylesheet-String fuer QPushButton
        """
        r, g, b = [int(c * 255) for c in color]
        r_h, g_h, b_h = [min(255, int(c * 255 * 1.2)) for c in color]
        r_p, g_p, b_p = [int(c * 255 * 0.8) for c in color]

        return f'''
            QPushButton {{
                background-color: rgb({r}, {g}, {b});
                color: white;
                border: none;
                border-radius: 4px;
                padding: {padding};
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: rgb({r_h}, {g_h}, {b_h});
            }}
            QPushButton:pressed {{
                background-color: rgb({r_p}, {g_p}, {b_p});
            }}
            QPushButton:disabled {{
                background-color: rgb(80, 80, 80);
                color: rgb(120, 120, 120);
            }}
        '''

    @staticmethod
    def button_style_hex(hex_color: str, padding: str = '8px 12px') -> str:
        """
        Generiert Button-Stylesheet aus Hex-Farbe.

        Args:
            hex_color: Hex-Farbe, z.B. '#4da8da'
            padding: CSS padding, default '8px 12px'

        Returns:
            Stylesheet-String fuer QPushButton
        """
        r, g, b = StyleFactory.hex_to_rgb(hex_color)
        r_h, g_h, b_h = [min(255, int(c * 1.2)) for c in (r, g, b)]
        r_p, g_p, b_p = [int(c * 0.8) for c in (r, g, b)]

        return f'''
            QPushButton {{
                background-color: rgb({r}, {g}, {b});
                color: white;
                border: none;
                border-radius: 4px;
                padding: {padding};
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: rgb({r_h}, {g_h}, {b_h});
            }}
            QPushButton:pressed {{
                background-color: rgb({r_p}, {g_p}, {b_p});
            }}
            QPushButton:disabled {{
                background-color: rgb(80, 80, 80);
                color: rgb(120, 120, 120);
            }}
        '''

    @staticmethod
    def group_style(title_color: tuple = None, hex_color: str = None) -> str:
        """
        Generiert GroupBox-Stylesheet mit farbigem Titel.

        Args:
            title_color: RGB-Tuple (0-1 Range) fuer Titelfarbe
            hex_color: Alternativ Hex-Farbe fuer Titel

        Returns:
            Stylesheet-String fuer QGroupBox
        """
        if hex_color:
            color_str = hex_color
        elif title_color:
            r, g, b = [int(c * 255) for c in title_color]
            color_str = f'rgb({r}, {g}, {b})'
        else:
            color_str = COLORS['accent']

        return f'''
            QGroupBox {{
                background-color: rgb(51, 51, 51);
                border: 1px solid rgb(68, 68, 68);
                border-radius: 4px;
                margin-top: 12px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: {color_str};
                font-weight: bold;
            }}
        '''

    @staticmethod
    def tab_style() -> str:
        """Generiert Tab-Widget Stylesheet."""
        return f'''
            QTabWidget::pane {{
                border: 1px solid #333;
                background-color: #2a2a2a;
            }}
            QTabBar::tab {{
                background: #333;
                color: #aaa;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-weight: bold;
            }}
            QTabBar::tab:selected {{
                background: {COLORS['accent']};
                color: white;
            }}
            QTabBar::tab:hover:!selected {{
                background: #444;
            }}
            QTabBar::tab:disabled {{
                background: #222;
                color: #555;
            }}
        '''

    @staticmethod
    def spinbox_style() -> str:
        """Generiert SpinBox-Stylesheet."""
        return f'''
            QSpinBox, QDoubleSpinBox {{
                background-color: #3a3a3a;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 3px;
                color: white;
            }}
            QSpinBox:focus, QDoubleSpinBox:focus {{
                border-color: {COLORS['accent']};
            }}
        '''

    @staticmethod
    def combobox_style() -> str:
        """Generiert ComboBox-Stylesheet."""
        return f'''
            QComboBox {{
                background-color: #3a3a3a;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px;
                color: white;
            }}
            QComboBox:hover {{
                border-color: {COLORS['accent']};
            }}
            QComboBox::drop-down {{
                border: none;
            }}
        '''

    @staticmethod
    def window_style() -> str:
        """Generiert Basis-Fenster-Stylesheet."""
        return '''
            QMainWindow {
                background-color: #262626;
            }
            QWidget {
                color: white;
            }
            QLabel {
                color: white;
            }
            QScrollArea {
                background-color: #2e2e2e;
            }
        '''


# Shortcut-Funktionen fuer einfacheren Zugriff
def button_style(color: tuple, padding: str = '8px 12px') -> str:
    """Shortcut fuer StyleFactory.button_style()"""
    return StyleFactory.button_style(color, padding)


def button_style_hex(hex_color: str, padding: str = '8px 12px') -> str:
    """Shortcut fuer StyleFactory.button_style_hex()"""
    return StyleFactory.button_style_hex(hex_color, padding)


def group_style(title_color: tuple = None, hex_color: str = None) -> str:
    """Shortcut fuer StyleFactory.group_style()"""
    return StyleFactory.group_style(title_color, hex_color)
