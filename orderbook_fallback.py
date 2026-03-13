# orderbook_fallback.py — proxy de compatibilidade
# Modulo movido para orderbook_core/orderbook_fallback.py
from orderbook_core.orderbook_fallback import *  # noqa: F401,F403
from orderbook_core.orderbook_fallback import (  # noqa: F401
    OrderBookFallback,
    FallbackConfig,
    get_fallback_instance,
    fetch_with_fallback,
)
