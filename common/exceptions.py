# common/exceptions.py
"""
Hierarquia unificada de excecoes do sistema.

Todas as excecoes do projeto derivam de BotBaseError, permitindo
captura generica com `except BotBaseError` quando necessario.

Hierarquia:
    BotBaseError
    ├── ConfigurationError
    ├── DataQualityError
    ├── ExternalServiceError
    │   ├── APIConnectionError
    │   ├── APITimeoutError
    │   └── RateLimitError
    ├── PipelineError
    ├── AIAnalysisError
    │   ├── AIRateLimitError
    │   └── ModelTimeoutError
    ├── AIRunnerError
    ├── RiskManagementError
    │   ├── RiskLimitExceeded
    │   ├── PositionLimitError
    │   └── DailyLossLimitError
    ├── OrderBookError
    │   ├── InvalidUpdateError
    │   ├── OrderBookConnectionError
    │   └── OrderBookTimeoutError
    └── FlowAnalyzerError
        ├── TradeValidationError
        ├── FlowConfigurationError
        ├── TimeBudgetExceededError
        ├── InvariantViolationError
        ├── CircuitBreakerOpenError
        ├── FlowDataQualityError
        └── AbsorptionGuardError
"""

from typing import Optional, Dict, Any


# ============================================================
# Base
# ============================================================

class BotBaseError(Exception):
    """Raiz de todas as excecoes do sistema."""
    pass


# ============================================================
# Configuracao e Dados (shared)
# ============================================================

class ConfigurationError(BotBaseError):
    """Configuracao invalida ou ausente."""
    pass


class DataQualityError(BotBaseError):
    """Dados invalidos, corrompidos ou insuficientes."""
    pass


# ============================================================
# Servicos Externos
# ============================================================

class ExternalServiceError(BotBaseError):
    """Falha em servico externo (Binance, Groq, FRED, yFinance)."""
    pass


class APIConnectionError(ExternalServiceError):
    """Falha de conexao com API externa."""
    pass


class APITimeoutError(ExternalServiceError):
    """Timeout em chamada de API externa."""
    pass


class RateLimitError(ExternalServiceError):
    """Rate limit excedido em API externa."""
    pass


# ============================================================
# Pipeline
# ============================================================

class PipelineError(BotBaseError):
    """Falha no pipeline de processamento de dados."""
    pass


# ============================================================
# AI Runner
# ============================================================

class AIAnalysisError(BotBaseError):
    """Raised when AI analysis fails."""
    pass


class AIRateLimitError(AIAnalysisError):
    """Raised when AI rate limit is exceeded.

    Note: Renomeado de RateLimitError (ai_runner) para evitar conflito
    com common.RateLimitError(ExternalServiceError).
    Re-exportado como RateLimitError em ai_runner/exceptions.py.
    """
    pass


class ModelTimeoutError(AIAnalysisError):
    """Raised when model operation times out."""
    pass


class AIRunnerError(BotBaseError):
    """Base exception for AI runner operations."""
    pass


# ============================================================
# Risk Management
# ============================================================

class RiskManagementError(BotBaseError):
    """Base exception for risk management errors."""
    pass


class RiskLimitExceeded(RiskManagementError):
    """Raised when risk limits are exceeded."""
    pass


class PositionLimitError(RiskManagementError):
    """Raised when position limits are exceeded."""
    pass


class DailyLossLimitError(RiskManagementError):
    """Raised when daily loss limits are exceeded."""
    pass


# ============================================================
# OrderBook
# ============================================================

class OrderBookError(BotBaseError):
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


# ============================================================
# FlowAnalyzer
# ============================================================

class FlowAnalyzerError(BotBaseError):
    """
    Excecao base para todos os erros do FlowAnalyzer.

    Attributes:
        message: Descricao do erro
        context: Dados contextuais para debugging
        recoverable: Se o erro e recuperavel
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
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
        """Serializa excecao para logging estruturado."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "context": self.context,
            "recoverable": self.recoverable,
        }


class TradeValidationError(FlowAnalyzerError):
    """Levantada quando um trade falha na validacao."""

    def __init__(
        self,
        reason: str,
        trade_data: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"Trade validation failed: {reason}",
            context={"reason": reason, "trade": trade_data},
            recoverable=True,
        )
        self.reason = reason
        self.trade_data = trade_data


class FlowConfigurationError(FlowAnalyzerError):
    """Levantada quando ha erro de configuracao no FlowAnalyzer.

    Note: Renomeado de ConfigurationError (flow_analyzer) para evitar conflito
    com common.ConfigurationError(BotBaseError).
    Re-exportado como ConfigurationError em flow_analyzer/errors.py.
    """

    def __init__(self, parameter: str, value: Any, expected: str):
        super().__init__(
            message=f"Invalid configuration for '{parameter}': {value}. Expected: {expected}",
            context={"parameter": parameter, "value": value, "expected": expected},
            recoverable=False,
        )
        self.parameter = parameter
        self.value = value
        self.expected = expected


class TimeBudgetExceededError(FlowAnalyzerError):
    """Levantada quando o time budget e excedido."""

    def __init__(self, operation: str, elapsed_ms: float, budget_ms: float):
        super().__init__(
            message=f"Time budget exceeded in '{operation}': {elapsed_ms:.1f}ms > {budget_ms:.1f}ms",
            context={
                "operation": operation,
                "elapsed_ms": elapsed_ms,
                "budget_ms": budget_ms,
            },
            recoverable=True,
        )
        self.operation = operation
        self.elapsed_ms = elapsed_ms
        self.budget_ms = budget_ms


class InvariantViolationError(FlowAnalyzerError):
    """Levantada quando uma invariante matematica e violada."""

    def __init__(
        self,
        invariant: str,
        expected: Any,
        actual: Any,
        tolerance: Optional[float] = None,
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
                "tolerance": tolerance,
            },
            recoverable=True,
        )
        self.invariant = invariant
        self.expected = expected
        self.actual = actual
        self.tolerance = tolerance


class CircuitBreakerOpenError(FlowAnalyzerError):
    """Levantada quando o circuit breaker esta aberto."""

    def __init__(self, operation: str, failures: int, recovery_time_remaining_ms: float):
        super().__init__(
            message=f"Circuit breaker open for '{operation}'. Failures: {failures}",
            context={
                "operation": operation,
                "failures": failures,
                "recovery_time_remaining_ms": recovery_time_remaining_ms,
            },
            recoverable=True,
        )
        self.operation = operation
        self.failures = failures
        self.recovery_time_remaining_ms = recovery_time_remaining_ms


class FlowDataQualityError(FlowAnalyzerError):
    """Levantada quando a qualidade dos dados esta abaixo do aceitavel.

    Note: Renomeado de DataQualityError (flow_analyzer) para evitar conflito
    com common.DataQualityError(BotBaseError).
    Re-exportado como DataQualityError em flow_analyzer/errors.py.
    """

    def __init__(
        self,
        metric: str,
        value: float,
        threshold: float,
        direction: str = "below",
    ):
        super().__init__(
            message=f"Data quality issue: {metric} is {value:.2f} ({direction} threshold {threshold:.2f})",
            context={
                "metric": metric,
                "value": value,
                "threshold": threshold,
                "direction": direction,
            },
            recoverable=True,
        )
        self.metric = metric
        self.value = value
        self.threshold = threshold


class AbsorptionGuardError(FlowAnalyzerError):
    """Levantada quando ha inconsistencia na classificacao de absorcao."""

    def __init__(self, delta: float, label: str, eps: float):
        super().__init__(
            message=f"Absorption guard triggered: delta={delta:.4f}, label='{label}', eps={eps}",
            context={"delta": delta, "label": label, "eps": eps},
            recoverable=True,
        )
        self.delta = delta
        self.label = label
        self.eps = eps
