# orderbook_core/exceptions.py
# Mantido para compatibilidade - exceções movidas para common/exceptions.py
from common.exceptions import (  # noqa: F401
    OrderBookError,
    InvalidUpdateError,
    OrderBookConnectionError,
    OrderBookTimeoutError,
)
