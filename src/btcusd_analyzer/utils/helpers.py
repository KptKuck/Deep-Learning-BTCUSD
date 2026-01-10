"""
Helper Funktionen - Allgemeine Hilfsfunktionen
"""

from datetime import datetime
from typing import Optional, Union

import numpy as np


def format_number(value: float, decimals: int = 2) -> str:
    """
    Formatiert eine Zahl mit Tausender-Trennzeichen.

    Args:
        value: Zu formatierende Zahl
        decimals: Anzahl Dezimalstellen

    Returns:
        Formatierte Zeichenkette
    """
    return f'{value:,.{decimals}f}'


def format_currency(value: float, currency: str = '$', decimals: int = 2) -> str:
    """
    Formatiert einen Waehrungsbetrag.

    Args:
        value: Betrag
        currency: Waehrungssymbol
        decimals: Anzahl Dezimalstellen

    Returns:
        Formatierte Zeichenkette (z.B. '$1,234.56')
    """
    sign = '-' if value < 0 else ''
    return f'{sign}{currency}{abs(value):,.{decimals}f}'


def format_percentage(value: float, decimals: int = 2, include_sign: bool = True) -> str:
    """
    Formatiert einen Prozentwert.

    Args:
        value: Prozentwert (z.B. 12.5 fuer 12.5%)
        decimals: Anzahl Dezimalstellen
        include_sign: Vorzeichen bei positiven Werten anzeigen

    Returns:
        Formatierte Zeichenkette (z.B. '+12.50%')
    """
    sign = '+' if value > 0 and include_sign else ''
    return f'{sign}{value:.{decimals}f}%'


def format_duration(seconds: float) -> str:
    """
    Formatiert eine Dauer in lesbares Format.

    Args:
        seconds: Dauer in Sekunden

    Returns:
        Formatierte Zeichenkette (z.B. '1h 23m 45s')
    """
    if seconds < 60:
        return f'{seconds:.1f}s'
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f'{minutes}m {secs}s'
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f'{hours}h {minutes}m'


def format_timestamp(
    timestamp: Union[datetime, np.datetime64],
    format_str: str = '%Y-%m-%d %H:%M:%S'
) -> str:
    """
    Formatiert einen Zeitstempel.

    Args:
        timestamp: Zeitstempel
        format_str: Format-String

    Returns:
        Formatierte Zeichenkette
    """
    if isinstance(timestamp, np.datetime64):
        timestamp = timestamp.astype('datetime64[s]').astype(datetime)
    return timestamp.strftime(format_str)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Begrenzt einen Wert auf einen Bereich.

    Args:
        value: Eingabewert
        min_val: Minimum
        max_val: Maximum

    Returns:
        Begrenzter Wert
    """
    return max(min_val, min(max_val, value))


def round_to_tick(price: float, tick_size: float = 0.01) -> float:
    """
    Rundet einen Preis auf die naechste Tick-Groesse.

    Args:
        price: Eingabepreis
        tick_size: Tick-Groesse

    Returns:
        Gerundeter Preis
    """
    return round(price / tick_size) * tick_size


def calculate_position_size(
    capital: float,
    risk_per_trade: float,
    entry_price: float,
    stop_loss_price: float
) -> float:
    """
    Berechnet die Positionsgroesse basierend auf Risikomanagement.

    Args:
        capital: Verfuegbares Kapital
        risk_per_trade: Risiko pro Trade in Prozent (z.B. 0.02 fuer 2%)
        entry_price: Einstiegspreis
        stop_loss_price: Stop-Loss Preis

    Returns:
        Positionsgroesse (Anzahl Einheiten)
    """
    risk_amount = capital * risk_per_trade
    price_risk = abs(entry_price - stop_loss_price)

    if price_risk == 0:
        return 0

    return risk_amount / price_risk


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """
    Berechnet einen gleitenden Durchschnitt.

    Args:
        data: Eingabedaten
        window: Fenstergroesse

    Returns:
        Array mit gleitendem Durchschnitt
    """
    if len(data) < window:
        return np.full_like(data, np.nan)

    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def exponential_moving_average(data: np.ndarray, span: int) -> np.ndarray:
    """
    Berechnet einen exponentiell gewichteten gleitenden Durchschnitt.

    Args:
        data: Eingabedaten
        span: EMA Span

    Returns:
        Array mit EMA
    """
    alpha = 2 / (span + 1)
    ema = np.zeros_like(data)
    ema[0] = data[0]

    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

    return ema


def get_gpu_info() -> dict:
    """
    Gibt Informationen ueber verfuegbare GPUs zurueck.

    Returns:
        Dictionary mit GPU-Informationen
    """
    info = {
        'cuda_available': False,
        'device_count': 0,
        'devices': []
    }

    try:
        import torch
        info['cuda_available'] = torch.cuda.is_available()

        if info['cuda_available']:
            info['device_count'] = torch.cuda.device_count()

            for i in range(info['device_count']):
                props = torch.cuda.get_device_properties(i)
                info['devices'].append({
                    'name': props.name,
                    'total_memory_gb': props.total_memory / (1024 ** 3),
                    'compute_capability': f'{props.major}.{props.minor}'
                })

    except ImportError:
        pass

    return info


def validate_dataframe(df, required_columns: list) -> tuple:
    """
    Validiert einen DataFrame auf erforderliche Spalten.

    Args:
        df: DataFrame
        required_columns: Liste erforderlicher Spalten

    Returns:
        Tuple aus (is_valid, missing_columns)
    """
    if df is None:
        return False, required_columns

    missing = [col for col in required_columns if col not in df.columns]
    return len(missing) == 0, missing
