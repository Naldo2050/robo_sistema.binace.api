# flow_analyzer/errors.py
"""
Exceções customizadas do FlowAnalyzer.

Hierarquia:
    FlowAnalyzerError (base)
    ├── TradeValidationError
    ├── ConfigurationError
    ├── TimeBudgetExceededError
    ├── InvariantViolationError
    ├── CircuitBreakerOpenError
    └── DataQualityError
"""

from typing import Optional, Dict, Any


class FlowAnalyzerError(Exception):
    """
    Exceção base para todos os erros do FlowAnalyzer.
    
    Attributes:
        message: Descrição do erro
        context: Dados contextuais para debugging
        recoverable: Se o erro é recuperável
    """
    
    def __init__(
        self, 
        message: str, 
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True
    ):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.recoverable = recoverable
    
    def __str__(self) -> str:
        base = self.message
        if self.context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            base = f"{base} [{ctx_str}]"
        return base
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa exceção para logging estruturado."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "context": self.context,
            "recoverable": self.recoverable,
        }


class TradeValidationError(FlowAnalyzerError):
    """Levantada quando um trade falha na validação."""
    
    def __init__(
        self, 
        reason: str, 
        trade_data: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=f"Trade validation failed: {reason}",
            context={"reason": reason, "trade": trade_data},
            recoverable=True
        )
        self.reason = reason
        self.trade_data = trade_data


class ConfigurationError(FlowAnalyzerError):
    """Levantada quando há erro de configuração."""
    
    def __init__(self, parameter: str, value: Any, expected: str):
        super().__init__(
            message=f"Invalid configuration for '{parameter}': {value}. Expected: {expected}",
            context={"parameter": parameter, "value": value, "expected": expected},
            recoverable=False
        )
        self.parameter = parameter
        self.value = value
        self.expected = expected


class TimeBudgetExceededError(FlowAnalyzerError):
    """Levantada quando o time budget é excedido."""
    
    def __init__(self, operation: str, elapsed_ms: float, budget_ms: float):
        super().__init__(
            message=f"Time budget exceeded in '{operation}': {elapsed_ms:.1f}ms > {budget_ms:.1f}ms",
            context={
                "operation": operation,
                "elapsed_ms": elapsed_ms,
                "budget_ms": budget_ms
            },
            recoverable=True
        )
        self.operation = operation
        self.elapsed_ms = elapsed_ms
        self.budget_ms = budget_ms


class InvariantViolationError(FlowAnalyzerError):
    """Levantada quando uma invariante matemática é violada."""
    
    def __init__(
        self, 
        invariant: str, 
        expected: Any, 
        actual: Any,
        tolerance: Optional[float] = None
    ):
        msg = f"Invariant violation: {invariant}. Expected={expected}, Actual={actual}"
        if tolerance is not None:
            msg += f" (tolerance={tolerance})"
        
        super().__init__(
            message=msg,
            context={
                "invariant": invariant,
                "expected": expected,
                "actual": actual,
                "tolerance": tolerance
            },
            recoverable=True
        )
        self.invariant = invariant
        self.expected = expected
        self.actual = actual
        self.tolerance = tolerance


class CircuitBreakerOpenError(FlowAnalyzerError):
    """Levantada quando o circuit breaker está aberto."""
    
    def __init__(self, operation: str, failures: int, recovery_time_remaining_ms: float):
        super().__init__(
            message=f"Circuit breaker open for '{operation}'. Failures: {failures}",
            context={
                "operation": operation,
                "failures": failures,
                "recovery_time_remaining_ms": recovery_time_remaining_ms
            },
            recoverable=True
        )
        self.operation = operation
        self.failures = failures
        self.recovery_time_remaining_ms = recovery_time_remaining_ms


class DataQualityError(FlowAnalyzerError):
    """Levantada quando a qualidade dos dados está abaixo do aceitável."""
    
    def __init__(
        self, 
        metric: str, 
        value: float, 
        threshold: float,
        direction: str = "below"
    ):
        super().__init__(
            message=f"Data quality issue: {metric} is {value:.2f} ({direction} threshold {threshold:.2f})",
            context={
                "metric": metric,
                "value": value,
                "threshold": threshold,
                "direction": direction
            },
            recoverable=True
        )
        self.metric = metric
        self.value = value
        self.threshold = threshold


class AbsorptionGuardError(FlowAnalyzerError):
    """Levantada quando há inconsistência na classificação de absorção."""
    
    def __init__(self, delta: float, label: str, eps: float):
        super().__init__(
            message=f"Absorption guard triggered: delta={delta:.4f}, label='{label}', eps={eps}",
            context={"delta": delta, "label": label, "eps": eps},
            recoverable=True
        )
        self.delta = delta
        self.label = label
        self.eps = eps