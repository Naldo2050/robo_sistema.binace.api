# orderbook_core/exceptions.py


class OrderBookError(Exception):
    """Base exception for order book operations."""
    pass


class InvalidUpdateError(OrderBookError):
    """Raised when order book update is invalid."""
    pass


class OrderBookConnectionError(OrderBookError):
    """Raised when order book connection fails."""
    pass


class OrderBookTimeoutError(OrderBookError):
    """Raised when order book operation times out."""
    pass