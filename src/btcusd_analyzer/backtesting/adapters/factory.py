"""
Backtester Factory - Erstellt Backtester-Instanzen
"""

from typing import Dict, List, Type

from ..base import BacktesterInterface
from ..backtester import InternalBacktester


class BacktesterFactory:
    """
    Factory fuer Backtester-Instanzen.

    Ermoeglicht dynamische Erstellung verschiedener Backtester
    basierend auf verfuegbaren Installationen.

    Usage:
        backtester = BacktesterFactory.create('internal')
        backtester = BacktesterFactory.create('vectorbt')  # Falls installiert
    """

    _registry: Dict[str, Type[BacktesterInterface]] = {
        'internal': InternalBacktester,
    }

    @classmethod
    def register(cls, name: str, backtester_class: Type[BacktesterInterface]):
        """
        Registriert einen Backtester.

        Args:
            name: Name des Backtesters
            backtester_class: Backtester-Klasse
        """
        cls._registry[name.lower()] = backtester_class

    @classmethod
    def create(cls, name: str = 'internal', **kwargs) -> BacktesterInterface:
        """
        Erstellt einen Backtester.

        Args:
            name: Name des Backtesters ('internal', 'vectorbt', etc.)
            **kwargs: Parameter fuer den Backtester

        Returns:
            Backtester-Instanz

        Raises:
            ValueError: Wenn Backtester nicht verfuegbar
        """
        name_lower = name.lower()

        # Lazy Loading fuer optionale Adapter
        if name_lower == 'vectorbt' and name_lower not in cls._registry:
            try:
                from .vectorbt import VectorBTAdapter
                cls.register('vectorbt', VectorBTAdapter)
            except ImportError:
                raise ValueError('VectorBT nicht installiert: pip install vectorbt')

        if name_lower == 'backtrader' and name_lower not in cls._registry:
            try:
                from .backtrader import BacktraderAdapter
                cls.register('backtrader', BacktraderAdapter)
            except ImportError:
                raise ValueError('Backtrader nicht installiert: pip install backtrader')

        if name_lower == 'backtestingpy' and name_lower not in cls._registry:
            try:
                from .backtestingpy import BacktestingPyAdapter
                cls.register('backtestingpy', BacktestingPyAdapter)
            except ImportError:
                raise ValueError('Backtesting.py nicht installiert: pip install backtesting')

        if name_lower not in cls._registry:
            available = ', '.join(cls.available())
            raise ValueError(f'Unbekannter Backtester: {name}. Verfuegbar: {available}')

        backtester = cls._registry[name_lower]()

        if kwargs:
            backtester.set_params(**kwargs)

        return backtester

    @classmethod
    def available(cls) -> List[str]:
        """
        Liste verfuegbarer Backtester.

        Returns:
            Liste von Backtester-Namen
        """
        available = ['internal']  # Immer verfuegbar

        try:
            import vectorbt
            available.append('vectorbt')
        except ImportError:
            pass

        try:
            import backtrader
            available.append('backtrader')
        except ImportError:
            pass

        try:
            import backtesting
            available.append('backtestingpy')
        except ImportError:
            pass

        return available

    @classmethod
    def is_available(cls, name: str) -> bool:
        """
        Prueft ob ein Backtester verfuegbar ist.

        Args:
            name: Name des Backtesters

        Returns:
            True wenn verfuegbar
        """
        return name.lower() in cls.available()
