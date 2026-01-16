"""
Trade-Statistik Dialog - Detaillierte Analyse der Trades.
"""

from typing import List, Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QWidget,
    QGroupBox, QLabel, QPushButton, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

import pandas as pd

from ..styles import get_stylesheet, StyleFactory


class TradeStatisticsDialog(QDialog):
    """Dialog fuer detaillierte Trade-Statistiken."""

    def __init__(self, parent, trades: list, data, equity_curve: list,
                 initial_capital: float, current_equity: float):
        super().__init__(parent)
        self.trades = trades
        self.data = data
        self.equity_curve = equity_curve
        self.initial_capital = initial_capital
        self.current_equity = current_equity

        self.setWindowTitle("Trade-Statistik")
        self.setMinimumSize(900, 700)
        self.setStyleSheet(get_stylesheet())

        self._setup_ui()

    def _setup_ui(self):
        """Erstellt die UI-Komponenten."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Tab-Widget fuer verschiedene Statistik-Bereiche
        tabs = QTabWidget()
        tabs.setStyleSheet(self._tab_style())

        # Tab 1: Uebersicht
        tabs.addTab(self._create_overview_tab(), "Uebersicht")

        # Tab 2: Trade-Liste
        tabs.addTab(self._create_trades_tab(), "Trade-Liste")

        # Tab 3: Long/Short Analyse
        tabs.addTab(self._create_long_short_tab(), "Long/Short")

        # Tab 4: Zeit-Analyse
        tabs.addTab(self._create_time_analysis_tab(), "Zeit-Analyse")

        layout.addWidget(tabs)

        # Export Button
        export_btn = QPushButton("Export als CSV")
        export_btn.setStyleSheet(StyleFactory.button_style_hex('#4da8da', padding='8px 20px'))
        export_btn.clicked.connect(self._export_csv)
        layout.addWidget(export_btn)

    def _create_overview_tab(self) -> QWidget:
        """Erstellt den Uebersicht-Tab."""
        from PyQt6.QtWidgets import QWidget
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # Linke Spalte: Konto
        left_group = QGroupBox("Konto-Uebersicht")
        left_group.setStyleSheet(self._group_style('#33b34d'))
        left_layout = QGridLayout(left_group)

        stats = self._calculate_account_stats()
        account_rows = [
            ("Startkapital:", f"${stats['start_capital']:,.2f}"),
            ("Aktuelles Kapital:", f"${stats['current_capital']:,.2f}"),
            ("Gesamt P/L:", f"${stats['total_pnl']:,.2f}"),
            ("Gesamt P/L %:", f"{stats['total_pnl_pct']:+.2f}%"),
            ("Max. Equity:", f"${stats['max_equity']:,.2f}"),
            ("Min. Equity:", f"${stats['min_equity']:,.2f}"),
            ("Max. Drawdown:", f"${stats['max_drawdown']:,.2f} ({stats['max_drawdown_pct']:.2f}%)"),
        ]

        for row, (label, value) in enumerate(account_rows):
            left_layout.addWidget(QLabel(label), row, 0)
            val_label = QLabel(value)
            val_label.setStyleSheet("color: white; font-weight: bold;")
            if 'P/L' in label:
                color = '#33cc33' if stats['total_pnl'] >= 0 else '#cc3333'
                val_label.setStyleSheet(f"color: {color}; font-weight: bold;")
            left_layout.addWidget(val_label, row, 1)

        layout.addWidget(left_group)

        # Rechte Spalte: Trade-Statistik
        right_group = QGroupBox("Trade-Statistik")
        right_group.setStyleSheet(self._group_style('#e6b333'))
        right_layout = QGridLayout(right_group)

        trade_stats = self._calculate_trade_stats()
        trade_rows = [
            ("Anzahl Trades:", str(trade_stats['total_trades'])),
            ("Gewinner:", f"{trade_stats['winners']} ({trade_stats['win_rate']:.1f}%)"),
            ("Verlierer:", f"{trade_stats['losers']} ({100-trade_stats['win_rate']:.1f}%)"),
            ("Profit Factor:", f"{trade_stats['profit_factor']:.2f}"),
            ("Avg. Trade:", f"${trade_stats['avg_trade']:,.2f}"),
            ("Avg. Gewinn:", f"${trade_stats['avg_win']:,.2f}"),
            ("Avg. Verlust:", f"${trade_stats['avg_loss']:,.2f}"),
            ("Groesster Gewinn:", f"${trade_stats['max_win']:,.2f}"),
            ("Groesster Verlust:", f"${trade_stats['max_loss']:,.2f}"),
            ("Win Streak:", str(trade_stats['max_win_streak'])),
            ("Loss Streak:", str(trade_stats['max_loss_streak'])),
        ]

        for row, (label, value) in enumerate(trade_rows):
            right_layout.addWidget(QLabel(label), row, 0)
            val_label = QLabel(value)
            val_label.setStyleSheet("color: white; font-weight: bold;")
            right_layout.addWidget(val_label, row, 1)

        layout.addWidget(right_group)

        return widget

    def _create_trades_tab(self) -> QWidget:
        """Erstellt den Trade-Liste Tab."""
        from PyQt6.QtWidgets import QWidget
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Tabelle
        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(9)
        self.trades_table.setHorizontalHeaderLabels([
            '#', 'Typ', 'Entry Zeit', 'Entry Preis', 'Exit Zeit', 'Exit Preis',
            'P/L', 'P/L %', 'Dauer'
        ])

        # Header-Styling
        header = self.trades_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setStyleSheet("QHeaderView::section { background-color: #333; color: white; padding: 5px; }")

        self.trades_table.setStyleSheet("""
            QTableWidget {
                background-color: #1a1a1a;
                color: white;
                gridline-color: #333;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QTableWidget::item:selected {
                background-color: #4da8da;
            }
        """)

        # Daten einfuegen
        self.trades_table.setRowCount(len(self.trades))
        for row, trade in enumerate(self.trades):
            self._add_trade_row(row, trade)

        layout.addWidget(self.trades_table)

        return widget

    def _add_trade_row(self, row: int, trade: dict):
        """Fuegt eine Trade-Zeile in die Tabelle ein."""
        entry_idx = trade.get('entry_index', 0) - 1
        exit_idx = trade.get('exit_index', 0) - 1
        pnl = trade.get('pnl', 0)
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        trade_type = trade.get('position', 'LONG')

        # Zeit-Informationen
        entry_time = ""
        exit_time = ""
        duration = ""
        if self.data is not None and entry_idx >= 0 and exit_idx >= 0:
            if entry_idx < len(self.data):
                entry_dt = self._get_datetime(entry_idx)
                entry_time = entry_dt.strftime('%Y-%m-%d %H:%M') if entry_dt else ""
            if exit_idx < len(self.data):
                exit_dt = self._get_datetime(exit_idx)
                exit_time = exit_dt.strftime('%Y-%m-%d %H:%M') if exit_dt else ""
            # Echte Zeitdauer berechnen
            if entry_dt and exit_dt:
                time_diff = exit_dt - entry_dt
                duration = self._format_duration(time_diff)
            else:
                duration_bars = exit_idx - entry_idx
                duration = f"{duration_bars} Bars"

        # P/L Prozent
        pnl_pct = (pnl / entry_price * 100) if entry_price > 0 else 0

        # Farbe basierend auf P/L
        color = QColor('#33cc33') if pnl >= 0 else QColor('#cc3333')

        items = [
            str(row + 1),
            trade_type,
            entry_time,
            f"${entry_price:,.2f}",
            exit_time,
            f"${exit_price:,.2f}",
            f"${pnl:,.2f}",
            f"{pnl_pct:+.2f}%",
            duration
        ]

        for col, text in enumerate(items):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if col in [6, 7]:  # P/L Spalten
                item.setForeground(color)
            self.trades_table.setItem(row, col, item)

    def _create_long_short_tab(self) -> QWidget:
        """Erstellt den Long/Short Analyse Tab."""
        from PyQt6.QtWidgets import QWidget
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Tabelle fuer Long/Short Vergleich
        table = QTableWidget()
        table.setColumnCount(4)
        table.setRowCount(7)
        table.setHorizontalHeaderLabels(['Metrik', 'Long', 'Short', 'Gesamt'])
        table.setVerticalHeaderLabels([''] * 7)

        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setStyleSheet("QHeaderView::section { background-color: #333; color: white; padding: 5px; }")

        table.setStyleSheet("""
            QTableWidget {
                background-color: #1a1a1a;
                color: white;
                gridline-color: #333;
            }
        """)

        # Statistiken berechnen
        long_stats = self._calculate_type_stats('LONG')
        short_stats = self._calculate_type_stats('SHORT')
        total_stats = self._calculate_trade_stats()

        rows = [
            ('Trades', long_stats['count'], short_stats['count'], total_stats['total_trades']),
            ('Gewinner', long_stats['winners'], short_stats['winners'], total_stats['winners']),
            ('Win-Rate', f"{long_stats['win_rate']:.1f}%", f"{short_stats['win_rate']:.1f}%", f"{total_stats['win_rate']:.1f}%"),
            ('P/L', f"${long_stats['pnl']:,.2f}", f"${short_stats['pnl']:,.2f}", f"${total_stats['total_pnl']:,.2f}"),
            ('Avg. Trade', f"${long_stats['avg_trade']:,.2f}", f"${short_stats['avg_trade']:,.2f}", f"${total_stats['avg_trade']:,.2f}"),
            ('Avg. Dauer', f"{long_stats['avg_duration']:.1f} Bars", f"{short_stats['avg_duration']:.1f} Bars", f"{total_stats['avg_duration']:.1f} Bars"),
            ('Profit Factor', f"{long_stats['profit_factor']:.2f}", f"{short_stats['profit_factor']:.2f}", f"{total_stats['profit_factor']:.2f}"),
        ]

        for row, (metric, long_val, short_val, total_val) in enumerate(rows):
            table.setItem(row, 0, QTableWidgetItem(metric))
            table.setItem(row, 1, QTableWidgetItem(str(long_val)))
            table.setItem(row, 2, QTableWidgetItem(str(short_val)))
            table.setItem(row, 3, QTableWidgetItem(str(total_val)))

        layout.addWidget(table)

        return widget

    def _create_time_analysis_tab(self) -> QWidget:
        """Erstellt den Zeit-Analyse Tab."""
        from PyQt6.QtWidgets import QWidget
        widget = QWidget()
        layout = QVBoxLayout(widget)

        time_stats = self._calculate_time_stats()

        # Zeit-Statistiken Gruppe
        time_group = QGroupBox("Zeit-Statistiken")
        time_group.setStyleSheet(self._group_style('#4da8da'))
        time_layout = QGridLayout(time_group)

        time_rows = [
            ("Backtest-Zeitraum:", time_stats['period']),
            ("Erster Trade:", time_stats['first_trade']),
            ("Letzter Trade:", time_stats['last_trade']),
            ("Trading-Tage:", str(time_stats['trading_days'])),
            ("Trades pro Tag:", f"{time_stats['trades_per_day']:.2f}"),
            ("Avg. Trade-Dauer:", time_stats['avg_duration']),
            ("Min. Trade-Dauer:", time_stats['min_duration']),
            ("Max. Trade-Dauer:", time_stats['max_duration']),
            ("Laengster Gewinn-Trade:", time_stats['longest_win']),
            ("Laengster Verlust-Trade:", time_stats['longest_loss']),
            ("Zeit im Markt:", time_stats['total_time_in_market']),
            ("Zeit im Markt (%):", time_stats['time_in_market_pct']),
        ]

        for row, (label, value) in enumerate(time_rows):
            time_layout.addWidget(QLabel(label), row, 0)
            val_label = QLabel(value)
            val_label.setStyleSheet("color: white; font-weight: bold;")
            time_layout.addWidget(val_label, row, 1)

        layout.addWidget(time_group)

        # Performance nach Tageszeit (wenn Daten vorhanden)
        if self.data is not None and len(self.trades) > 0:
            hourly_group = QGroupBox("Performance nach Stunde (Entry)")
            hourly_group.setStyleSheet(self._group_style('#b19cd9'))
            hourly_layout = QVBoxLayout(hourly_group)

            hourly_table = QTableWidget()
            hourly_stats = self._calculate_hourly_stats()

            hourly_table.setColumnCount(4)
            hourly_table.setRowCount(len(hourly_stats))
            hourly_table.setHorizontalHeaderLabels(['Stunde', 'Trades', 'Win-Rate', 'P/L'])

            header = hourly_table.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            header.setStyleSheet("QHeaderView::section { background-color: #333; color: white; }")

            hourly_table.setStyleSheet("""
                QTableWidget { background-color: #1a1a1a; color: white; gridline-color: #333; }
            """)

            for row, (hour, stats) in enumerate(sorted(hourly_stats.items())):
                hourly_table.setItem(row, 0, QTableWidgetItem(f"{hour:02d}:00"))
                hourly_table.setItem(row, 1, QTableWidgetItem(str(stats['count'])))
                hourly_table.setItem(row, 2, QTableWidgetItem(f"{stats['win_rate']:.1f}%"))
                pnl_item = QTableWidgetItem(f"${stats['pnl']:,.2f}")
                pnl_item.setForeground(QColor('#33cc33' if stats['pnl'] >= 0 else '#cc3333'))
                hourly_table.setItem(row, 3, pnl_item)

            hourly_layout.addWidget(hourly_table)
            layout.addWidget(hourly_group)

        layout.addStretch()
        return widget

    def _get_datetime(self, idx: int):
        """Holt DateTime aus dem DataFrame (Index oder Spalte)."""
        if self.data is None or idx < 0 or idx >= len(self.data):
            return None
        try:
            # Versuche Index (wenn DateTime als Index gesetzt)
            dt = self.data.index[idx]
            if hasattr(dt, 'strftime'):
                return dt
            # Versuche DateTime-Spalte
            if 'DateTime' in self.data.columns:
                dt_val = self.data['DateTime'].iloc[idx]
                # Falls String, in Timestamp konvertieren
                if isinstance(dt_val, str):
                    return pd.to_datetime(dt_val)
                if hasattr(dt_val, 'strftime'):
                    return dt_val
        except:
            pass
        return None

    def _format_duration(self, td) -> str:
        """Formatiert eine Zeitdauer lesbar."""
        total_seconds = int(td.total_seconds())
        if total_seconds < 0:
            return "0m"

        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        minutes = (total_seconds % 3600) // 60

        if days > 0:
            return f"{days}d {hours}h"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"

    def _calculate_account_stats(self) -> dict:
        """Berechnet Konto-Statistiken."""
        max_equity = max(self.equity_curve) if self.equity_curve else self.initial_capital
        min_equity = min(self.equity_curve) if self.equity_curve else self.initial_capital

        # Max Drawdown berechnen
        max_drawdown = 0
        peak = self.initial_capital
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            drawdown = peak - eq
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        total_pnl = self.current_equity - self.initial_capital
        total_pnl_pct = (total_pnl / self.initial_capital * 100) if self.initial_capital > 0 else 0
        max_drawdown_pct = (max_drawdown / peak * 100) if peak > 0 else 0

        return {
            'start_capital': self.initial_capital,
            'current_capital': self.current_equity,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'max_equity': max_equity,
            'min_equity': min_equity,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
        }

    def _calculate_trade_stats(self) -> dict:
        """Berechnet Trade-Statistiken."""
        if not self.trades:
            return {
                'total_trades': 0, 'winners': 0, 'losers': 0, 'win_rate': 0,
                'profit_factor': 0, 'avg_trade': 0, 'avg_win': 0, 'avg_loss': 0,
                'max_win': 0, 'max_loss': 0, 'max_win_streak': 0, 'max_loss_streak': 0,
                'total_pnl': 0, 'avg_duration': 0
            }

        pnls = [t.get('pnl', 0) for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        # Streaks berechnen
        win_streak = 0
        loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        for pnl in pnls:
            if pnl > 0:
                win_streak += 1
                loss_streak = 0
                max_win_streak = max(max_win_streak, win_streak)
            else:
                loss_streak += 1
                win_streak = 0
                max_loss_streak = max(max_loss_streak, loss_streak)

        # Durchschnittliche Dauer
        durations = []
        for t in self.trades:
            entry_idx = t.get('entry_index', 0)
            exit_idx = t.get('exit_index', 0)
            if entry_idx > 0 and exit_idx > 0:
                durations.append(exit_idx - entry_idx)

        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = (total_wins / total_losses) if total_losses > 0 else float('inf') if total_wins > 0 else 0

        return {
            'total_trades': len(self.trades),
            'winners': len(wins),
            'losers': len(losses),
            'win_rate': (len(wins) / len(self.trades) * 100) if self.trades else 0,
            'profit_factor': profit_factor if profit_factor != float('inf') else 999.99,
            'avg_trade': sum(pnls) / len(pnls) if pnls else 0,
            'avg_win': sum(wins) / len(wins) if wins else 0,
            'avg_loss': sum(losses) / len(losses) if losses else 0,
            'max_win': max(pnls) if pnls else 0,
            'max_loss': min(pnls) if pnls else 0,
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
            'total_pnl': sum(pnls),
            'avg_duration': sum(durations) / len(durations) if durations else 0,
        }

    def _calculate_type_stats(self, trade_type: str) -> dict:
        """Berechnet Statistiken fuer einen Trade-Typ (LONG/SHORT)."""
        type_trades = [t for t in self.trades if t.get('position') == trade_type]

        if not type_trades:
            return {
                'count': 0, 'winners': 0, 'win_rate': 0, 'pnl': 0,
                'avg_trade': 0, 'avg_duration': 0, 'profit_factor': 0
            }

        pnls = [t.get('pnl', 0) for t in type_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        durations = []
        for t in type_trades:
            entry_idx = t.get('entry_index', 0)
            exit_idx = t.get('exit_index', 0)
            if entry_idx > 0 and exit_idx > 0:
                durations.append(exit_idx - entry_idx)

        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = (total_wins / total_losses) if total_losses > 0 else float('inf') if total_wins > 0 else 0

        return {
            'count': len(type_trades),
            'winners': len(wins),
            'win_rate': (len(wins) / len(type_trades) * 100) if type_trades else 0,
            'pnl': sum(pnls),
            'avg_trade': sum(pnls) / len(pnls) if pnls else 0,
            'avg_duration': sum(durations) / len(durations) if durations else 0,
            'profit_factor': profit_factor if profit_factor != float('inf') else 999.99,
        }

    def _calculate_time_stats(self) -> dict:
        """Berechnet Zeit-bezogene Statistiken mit echten Zeitdauern."""
        if not self.trades or self.data is None:
            return {
                'period': '-', 'first_trade': '-', 'last_trade': '-',
                'trading_days': 0, 'trades_per_day': 0,
                'avg_duration': '-', 'min_duration': '-', 'max_duration': '-',
                'longest_win': '-', 'longest_loss': '-',
                'total_time_in_market': '-', 'time_in_market_pct': '-'
            }

        # Zeitraum
        start_dt = self._get_datetime(0)
        end_dt = self._get_datetime(len(self.data) - 1)
        period_start = start_dt.strftime('%Y-%m-%d') if start_dt else '-'
        period_end = end_dt.strftime('%Y-%m-%d') if end_dt else '-'
        period = f"{period_start} bis {period_end}"

        # Erster und letzter Trade
        first_trade = '-'
        last_trade = '-'
        if self.trades:
            first_idx = self.trades[0].get('entry_index', 0) - 1
            last_idx = self.trades[-1].get('exit_index', 0) - 1
            first_dt = self._get_datetime(first_idx)
            last_dt = self._get_datetime(last_idx)
            if first_dt:
                first_trade = first_dt.strftime('%Y-%m-%d %H:%M')
            if last_dt:
                last_trade = last_dt.strftime('%Y-%m-%d %H:%M')

        # Echte Zeitdauern berechnen
        durations = []  # in Sekunden
        win_durations = []
        loss_durations = []
        total_time_in_market = 0

        for t in self.trades:
            entry_idx = t.get('entry_index', 0) - 1
            exit_idx = t.get('exit_index', 0) - 1
            pnl = t.get('pnl', 0)

            entry_dt = self._get_datetime(entry_idx)
            exit_dt = self._get_datetime(exit_idx)

            if entry_dt and exit_dt:
                dur_seconds = (exit_dt - entry_dt).total_seconds()
                durations.append(dur_seconds)
                total_time_in_market += dur_seconds
                if pnl > 0:
                    win_durations.append(dur_seconds)
                else:
                    loss_durations.append(dur_seconds)

        # Trading-Tage (einzigartige Tage)
        trading_days = set()
        for t in self.trades:
            entry_idx = t.get('entry_index', 0) - 1
            entry_dt = self._get_datetime(entry_idx)
            if entry_dt:
                trading_days.add(entry_dt.strftime('%Y-%m-%d'))

        # Gesamtzeit des Backtests
        total_backtest_time = 0
        if start_dt and end_dt:
            total_backtest_time = (end_dt - start_dt).total_seconds()

        # Zeit im Markt als Prozent
        time_in_market_pct = (total_time_in_market / total_backtest_time * 100) if total_backtest_time > 0 else 0

        # Hilfsfunktion fuer Zeitformatierung
        def format_seconds(secs):
            if secs == 0:
                return "0m"
            td = pd.Timedelta(seconds=secs)
            return self._format_duration(td)

        return {
            'period': period,
            'first_trade': first_trade,
            'last_trade': last_trade,
            'trading_days': len(trading_days),
            'trades_per_day': len(self.trades) / len(trading_days) if trading_days else 0,
            'avg_duration': format_seconds(sum(durations) / len(durations)) if durations else '-',
            'min_duration': format_seconds(min(durations)) if durations else '-',
            'max_duration': format_seconds(max(durations)) if durations else '-',
            'longest_win': format_seconds(max(win_durations)) if win_durations else '-',
            'longest_loss': format_seconds(max(loss_durations)) if loss_durations else '-',
            'total_time_in_market': format_seconds(total_time_in_market),
            'time_in_market_pct': f"{time_in_market_pct:.1f}%",
        }

    def _calculate_hourly_stats(self) -> dict:
        """Berechnet Statistiken nach Stunde."""
        hourly = {}

        for t in self.trades:
            entry_idx = t.get('entry_index', 0) - 1
            if entry_idx >= 0 and entry_idx < len(self.data):
                try:
                    hour = self.data.index[entry_idx].hour
                except:
                    continue

                if hour not in hourly:
                    hourly[hour] = {'count': 0, 'wins': 0, 'pnl': 0}

                hourly[hour]['count'] += 1
                hourly[hour]['pnl'] += t.get('pnl', 0)
                if t.get('pnl', 0) > 0:
                    hourly[hour]['wins'] += 1

        # Win-Rate berechnen
        for hour in hourly:
            count = hourly[hour]['count']
            hourly[hour]['win_rate'] = (hourly[hour]['wins'] / count * 100) if count > 0 else 0

        return hourly

    def _export_csv(self):
        """Exportiert die Trades als CSV."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Trades exportieren", "trades_export.csv", "CSV Files (*.csv)"
        )

        if filename:
            try:
                import csv
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['#', 'Typ', 'Entry Zeit', 'Entry Preis', 'Exit Zeit',
                                    'Exit Preis', 'P/L', 'P/L %', 'Dauer (Bars)'])

                    for i, trade in enumerate(self.trades):
                        entry_idx = trade.get('entry_index', 0) - 1
                        exit_idx = trade.get('exit_index', 0) - 1
                        entry_time = str(self.data.index[entry_idx]) if entry_idx >= 0 and entry_idx < len(self.data) else ''
                        exit_time = str(self.data.index[exit_idx]) if exit_idx >= 0 and exit_idx < len(self.data) else ''
                        entry_price = trade.get('entry_price', 0)
                        pnl = trade.get('pnl', 0)
                        pnl_pct = (pnl / entry_price * 100) if entry_price > 0 else 0
                        duration = exit_idx - entry_idx if entry_idx >= 0 and exit_idx >= 0 else 0

                        writer.writerow([
                            i + 1, trade.get('position', ''), entry_time,
                            f"{trade.get('entry_price', 0):.2f}", exit_time,
                            f"{trade.get('exit_price', 0):.2f}",
                            f"{pnl:.2f}", f"{pnl_pct:.2f}", duration
                        ])

            except Exception as e:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Export Fehler", f"Fehler beim Export: {e}")

    def _tab_style(self) -> str:
        """Gibt das Tab-Stylesheet zurueck."""
        return '''
            QTabWidget::pane { border: 1px solid #4d4d4d; background-color: #262626; }
            QTabBar::tab { background-color: #333; color: #b3b3b3; padding: 8px 20px;
                          margin-right: 2px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background-color: #4da8da; color: white; }
            QTabBar::tab:hover:!selected { background-color: #444; }
        '''

    def _group_style(self, color: str) -> str:
        """Gibt das GroupBox-Stylesheet zurueck."""
        return f'''
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {color};
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                color: {color};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
            QLabel {{ color: #b3b3b3; }}
        '''
