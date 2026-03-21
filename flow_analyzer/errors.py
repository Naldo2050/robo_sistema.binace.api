# flow_analyzer/errors.py
# Mantido para compatibilidade - exceções movidas para common/exceptions.py
from common.exceptions import (  # noqa: F401
    FlowAnalyzerError,
    TradeValidationError,
    FlowConfigurationError as ConfigurationError,
    TimeBudgetExceededError,
    InvariantViolationError,
    CircuitBreakerOpenError,
    FlowDataQualityError as DataQualityError,
    AbsorptionGuardError,
)
