"""Web Module - Status-Dashboard Server"""

from .server import StatusServer, ExtendedStatusServer, AppState
from .routes import APIRoutes, WebSocketRoutes

__all__ = [
    'StatusServer',
    'ExtendedStatusServer',
    'AppState',
    'APIRoutes',
    'WebSocketRoutes'
]
