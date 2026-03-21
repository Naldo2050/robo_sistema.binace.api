# risk_management/exceptions.py
# Mantido para compatibilidade - exceções movidas para common/exceptions.py
from common.exceptions import (  # noqa: F401
    RiskLimitExceeded,
    PositionLimitError,
    DailyLossLimitError,
    RiskManagementError,
)
