# risk_management/exceptions.py


class RiskLimitExceeded(Exception):
    """Raised when risk limits are exceeded."""
    pass


class PositionLimitError(Exception):
    """Raised when position limits are exceeded."""
    pass


class DailyLossLimitError(Exception):
    """Raised when daily loss limits are exceeded."""
    pass


class RiskManagementError(Exception):
    """Base exception for risk management errors."""
    pass