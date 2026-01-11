"""
Erstellt einen Flowchart der Projektstruktur als PNG.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Figur erstellen
fig, ax = plt.subplots(1, 1, figsize=(20, 16))
ax.set_xlim(0, 20)
ax.set_ylim(0, 16)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('#1a1a1a')
ax.set_facecolor('#1a1a1a')

# Farben
COLORS = {
    'main': '#4da8da',      # Blau - Einstiegspunkt
    'core': '#9b59b6',      # Lila - Core
    'gui': '#33b34d',       # Gruen - GUI
    'data': '#e6b333',      # Orange - Data
    'models': '#cc4d33',    # Rot - Models
    'training': '#3498db',  # Hellblau - Training
    'trading': '#e74c3c',   # Rot - Trading
    'backtest': '#1abc9c',  # Tuerkis - Backtesting
    'web': '#f39c12',       # Gold - Web
    'utils': '#95a5a6',     # Grau - Utils
}

def draw_box(ax, x, y, w, h, text, color, fontsize=8):
    """Zeichnet eine Box mit Text."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=color,
        edgecolor='white',
        linewidth=1.5,
        alpha=0.9
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, color='white', fontweight='bold',
            wrap=True)

def draw_module_group(ax, x, y, w, h, title, files, color):
    """Zeichnet eine Modulgruppe mit Dateien."""
    # Hintergrund
    bg = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.2",
        facecolor=color,
        edgecolor='white',
        linewidth=2,
        alpha=0.3
    )
    ax.add_patch(bg)

    # Titel
    ax.text(x + w/2, y + h - 0.3, title, ha='center', va='top',
            fontsize=11, color='white', fontweight='bold')

    # Dateien
    file_y = y + h - 0.7
    for f in files:
        ax.text(x + 0.15, file_y, f"  {f}", ha='left', va='top',
                fontsize=7, color='#cccccc', family='monospace')
        file_y -= 0.35

def draw_arrow(ax, start, end, color='white', style='->'):
    """Zeichnet einen Pfeil."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=1.5,
                               connectionstyle='arc3,rad=0.1'))

# === LAYOUT ===

# Titel
ax.text(10, 15.5, 'BTCUSD Analyzer - Projektstruktur', ha='center', va='center',
        fontsize=18, color='white', fontweight='bold')
ax.text(10, 15.0, 'Modul-Abhaengigkeiten und Datenfluss', ha='center', va='center',
        fontsize=10, color='#aaaaaa')

# === MAIN (Einstiegspunkt) ===
draw_box(ax, 9, 13.5, 2, 0.8, 'main.py\n(Einstiegspunkt)', COLORS['main'], 9)

# === CORE ===
draw_module_group(ax, 0.5, 11, 3.5, 2.2, 'core/',
                  ['logger.py', 'config.py'], COLORS['core'])

# === GUI ===
draw_module_group(ax, 4.5, 10, 4, 3.5, 'gui/',
                  ['main_window.py', 'training_window.py',
                   'prepare_data_window.py', 'backtest_window.py',
                   'trading_window.py', 'visualize_data_window.py',
                   'styles.py', 'widgets/'], COLORS['gui'])

# === DATA ===
draw_module_group(ax, 9, 10, 3, 2.5, 'data/',
                  ['reader.py', 'processor.py', 'downloader.py'], COLORS['data'])

# === MODELS ===
draw_module_group(ax, 12.5, 10, 3.5, 2.8, 'models/',
                  ['base.py', 'factory.py', 'bilstm.py',
                   'gru.py', 'cnn.py', 'cnn_lstm.py'], COLORS['models'])

# === TRAINING ===
draw_module_group(ax, 16.5, 10, 3, 2.5, 'training/',
                  ['labeler.py', 'sequence.py', 'normalizer.py'], COLORS['training'])

# === TRAINER ===
draw_module_group(ax, 0.5, 6, 3.5, 2.5, 'trainer/',
                  ['trainer.py', 'callbacks.py'], COLORS['training'])

# === BACKTESTING ===
draw_module_group(ax, 4.5, 5.5, 4, 3, 'backtesting/',
                  ['backtester.py', 'base.py', 'metrics.py',
                   'adapters/factory.py', 'adapters/vectorbt.py',
                   'adapters/backtrader.py'], COLORS['backtest'])

# === TRADING ===
draw_module_group(ax, 9, 5.5, 4, 3.2, 'trading/',
                  ['live_trader.py', 'binance_client.py',
                   'order_manager.py', 'risk_manager.py',
                   'websocket_handler.py', 'api_config.py'], COLORS['trading'])

# === OPTIMIZATION ===
draw_module_group(ax, 13.5, 6, 3, 2, 'optimization/',
                  ['optuna_tuner.py'], COLORS['core'])

# === WEB ===
draw_module_group(ax, 13.5, 3, 3, 2.5, 'web/',
                  ['server.py', 'routes.py'], COLORS['web'])

# === UTILS ===
draw_module_group(ax, 17, 6, 2.5, 2, 'utils/',
                  ['helpers.py'], COLORS['utils'])

# === PFEILE (Abhaengigkeiten) ===

# main -> GUI
draw_arrow(ax, (10, 13.5), (6.5, 13.5), COLORS['gui'])

# main -> core
draw_arrow(ax, (9, 13.9), (4, 12.5), COLORS['core'])

# GUI -> Data
draw_arrow(ax, (8.5, 11.5), (9, 11.5), COLORS['data'])

# GUI -> Models
draw_arrow(ax, (8.5, 11), (12.5, 11), COLORS['models'])

# GUI -> Training
draw_arrow(ax, (8.5, 10.5), (16.5, 11), COLORS['training'])

# GUI -> Trainer
draw_arrow(ax, (5.5, 10), (3, 8.5), COLORS['training'])

# GUI -> Backtesting
draw_arrow(ax, (5.5, 10), (6, 8.5), COLORS['backtest'])

# GUI -> Trading
draw_arrow(ax, (7.5, 10), (10.5, 8.7), COLORS['trading'])

# Models -> base
draw_arrow(ax, (14, 10), (14, 9.5), 'white', '->')

# Trainer -> Models
draw_arrow(ax, (4, 7), (12.5, 10.5), COLORS['models'])

# Trainer -> Training
draw_arrow(ax, (4, 7.5), (16.5, 11.5), COLORS['training'])

# Data -> Training
draw_arrow(ax, (12, 11), (16.5, 11.5), COLORS['training'])

# Backtesting -> Models
draw_arrow(ax, (8.5, 7), (12.5, 10), COLORS['models'])

# Trading -> Models
draw_arrow(ax, (11, 8.7), (13, 10), COLORS['models'])

# Web -> Trading
draw_arrow(ax, (13.5, 4.5), (13, 5.5), COLORS['trading'])

# Optimization -> Trainer
draw_arrow(ax, (13.5, 7), (4, 7.5), COLORS['training'])

# === LEGENDE ===
legend_y = 2
legend_x = 0.5
ax.text(legend_x, legend_y + 1.5, 'Legende:', fontsize=10, color='white', fontweight='bold')

legend_items = [
    ('Einstiegspunkt', COLORS['main']),
    ('Core (Logger, Config)', COLORS['core']),
    ('GUI (PyQt6)', COLORS['gui']),
    ('Data (CSV, Features)', COLORS['data']),
    ('Models (LSTM, GRU, CNN)', COLORS['models']),
    ('Training (Labels, Sequences)', COLORS['training']),
    ('Backtesting', COLORS['backtest']),
    ('Trading (Binance)', COLORS['trading']),
    ('Web (Flask)', COLORS['web']),
    ('Utils', COLORS['utils']),
]

for i, (label, color) in enumerate(legend_items):
    row = i // 2
    col = i % 2
    lx = legend_x + col * 4.5
    ly = legend_y - row * 0.4
    ax.add_patch(plt.Rectangle((lx, ly - 0.1), 0.3, 0.25, facecolor=color, edgecolor='white'))
    ax.text(lx + 0.4, ly, label, fontsize=8, color='white', va='center')

# === DATENFLUSS-INFO ===
ax.text(17.5, 2.5, 'Datenfluss:', fontsize=10, color='white', fontweight='bold')
ax.text(17.5, 2.0, '1. CSV laden (data/)', fontsize=8, color='#aaaaaa')
ax.text(17.5, 1.6, '2. Features berechnen', fontsize=8, color='#aaaaaa')
ax.text(17.5, 1.2, '3. Labels generieren', fontsize=8, color='#aaaaaa')
ax.text(17.5, 0.8, '4. Sequenzen erstellen', fontsize=8, color='#aaaaaa')
ax.text(17.5, 0.4, '5. Modell trainieren', fontsize=8, color='#aaaaaa')

# Speichern
plt.tight_layout()
plt.savefig('c:/Work/MatLab/Deep-Learning-BTCUSD/docs/project_structure.png',
            dpi=150, facecolor='#1a1a1a', edgecolor='none',
            bbox_inches='tight', pad_inches=0.3)
plt.close()
print("Flowchart gespeichert: docs/project_structure.png")
