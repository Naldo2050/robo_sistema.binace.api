"""
Protocolos para desacoplar componentes de IA e quebrar imports circulares.

Os contratos aqui refletem a superfície realmente usada pelo sistema atual:
- `AIAnalyzer` é síncrono em `analyze(...)`
- `build_compact_payload(...)` é uma função chamável
- summary builders seguem `builder(payload) -> dict`
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class AIAnalyzerProtocol(Protocol):
    """Contrato mínimo consumido pelo orquestrador para análise de IA."""

    def analyze(
        self,
        event_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analisa o evento e retorna o resultado estruturado."""
        ...

    def close(self) -> None:
        """Fecha recursos síncronos associados ao analisador."""
        ...

    async def aclose(self) -> None:
        """Fecha recursos assíncronos associados ao analisador."""
        ...


@runtime_checkable
class SummaryBuilderProtocol(Protocol):
    """Interface para funções `build_*_summary`."""

    def __call__(
        self,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Constrói uma seção resumida do payload."""
        ...


@runtime_checkable
class BuildCompactPayloadProtocol(Protocol):
    """Interface para a função principal `build_compact_payload`."""

    def __call__(
        self,
        event_data: Dict[str, Any],
        builders: Any = None,
    ) -> Dict[str, Any]:
        """Constrói o payload compactado para a IA."""
        ...


@runtime_checkable
class LockProtocol(Protocol):
    """Contrato mínimo para locks/semaphores usados com `with`."""

    def __enter__(self) -> Any:
        ...

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool | None:
        ...


@runtime_checkable
class RateLimiterProtocol(Protocol):
    """Contrato mínimo para o rate limiter usado pelo runner."""

    def acquire(self) -> Any:
        ...


@runtime_checkable
class HealthMonitorProtocol(Protocol):
    """Contrato mínimo de heartbeat usado pelo runner de IA."""

    def heartbeat(self, module_name: str) -> Any:
        ...


@runtime_checkable
class PredictorProtocol(Protocol):
    """Contrato mínimo para engines de inferência/ML."""

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        ...


@runtime_checkable
class FeatureCalculatorProtocol(Protocol):
    """Contrato mínimo para o calculador de features usado no overlay de ML."""

    history_count: int

    def compute(self) -> Dict[str, Any]:
        ...


@runtime_checkable
class EventSaverProtocol(Protocol):
    """Contrato mínimo para persistência de eventos de IA."""

    def save_event(self, event: Dict[str, Any]) -> Any:
        ...


@runtime_checkable
class BotAIRuntimeProtocol(Protocol):
    """
    Superfície mínima do bot consumida por `market_orchestrator.ai.ai_runner`.

    O objetivo é explicitar o contrato real sem acoplar o runner ao
    `EnhancedMarketBot` completo.
    """

    symbol: str
    should_stop: bool

    ai_runner: Any
    ai_analyzer: AIAnalyzerProtocol | None
    ai_initialization_attempted: bool
    ai_test_passed: bool
    ai_thread_pool: List[Any]
    max_ai_threads: int
    ml_engine: PredictorProtocol | None

    _ai_init_lock: LockProtocol
    _ai_pool_lock: LockProtocol
    ai_semaphore: LockProtocol
    ai_rate_limiter: RateLimiterProtocol

    health_monitor: HealthMonitorProtocol
    event_saver: EventSaverProtocol | None
    feature_calc: FeatureCalculatorProtocol | None


SummaryFn = SummaryBuilderProtocol
SummaryBuilderMap = Dict[str, SummaryBuilderProtocol]
