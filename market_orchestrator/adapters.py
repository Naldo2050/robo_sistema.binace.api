from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from .protocols import OrchestratorControlProtocol

if TYPE_CHECKING:
    from .market_orchestrator import EnhancedMarketBot
    from .orchestrator import MarketOrchestrator


class MarketOrchestratorAdapter(OrchestratorControlProtocol):
    """
    Adapter do orchestrator leve para um contrato comum.

    Não altera o comportamento atual. Só normaliza a superfície usada por
    monitoramento, testes e futuras etapas de consolidação.
    """

    def __init__(self, orchestrator: "MarketOrchestrator") -> None:
        self._orchestrator = orchestrator

    @property
    def symbol(self) -> str:
        return self._orchestrator.symbol

    async def start_runtime(self) -> None:
        self._orchestrator.start()

    async def stop_runtime(self) -> None:
        self._orchestrator.stop()

    def snapshot_state(self) -> Dict[str, Any]:
        status = self._orchestrator.get_status()
        health = self._orchestrator.health_check()
        metrics = self._orchestrator.get_metrics()
        return {
            "kind": "market_orchestrator",
            "symbol": self.symbol,
            "is_running": status.get("is_running", False),
            "state": status.get("state", {}),
            "health": health,
            "metrics": metrics,
        }


class EnhancedMarketBotAdapter(OrchestratorControlProtocol):
    """
    Adapter do orchestrator principal para o mesmo contrato comum.

    O objetivo aqui é preparar uma camada estável de leitura/controle antes de
    qualquer tentativa de unificação estrutural entre as duas implementações.
    """

    def __init__(self, bot: "EnhancedMarketBot") -> None:
        self._bot = bot

    @property
    def symbol(self) -> str:
        return self._bot.symbol

    async def start_runtime(self) -> None:
        await self._bot.run()

    async def stop_runtime(self) -> None:
        await self._bot.shutdown()

    def snapshot_state(self) -> Dict[str, Any]:
        return {
            "kind": "enhanced_market_bot",
            "symbol": self.symbol,
            "is_running": not bool(getattr(self._bot, "should_stop", False)),
            "state": {
                "should_stop": bool(getattr(self._bot, "should_stop", False)),
                "warming_up": bool(getattr(self._bot, "warming_up", False)),
                "window_count": int(getattr(self._bot, "window_count", 0)),
                "ai_ready": bool(
                    getattr(self._bot, "ai_analyzer", None)
                    and getattr(self._bot, "ai_test_passed", False)
                ),
                "has_ai_runner": getattr(self._bot, "ai_runner", None) is not None,
                "connection_manager": getattr(self._bot, "connection_manager", None)
                is not None,
            },
            "health": {
                "health_monitor": getattr(self._bot, "health_monitor", None) is not None,
                "event_bus": getattr(self._bot, "event_bus", None) is not None,
                "context_collector": getattr(self._bot, "context_collector", None)
                is not None,
            },
            "metrics": {
                "ai_threads": len(getattr(self._bot, "ai_thread_pool", [])),
                "max_ai_threads": int(getattr(self._bot, "max_ai_threads", 0)),
                "window_ms": int(getattr(self._bot, "window_ms", 0)),
            },
        }


def adapt_orchestrator_runtime(runtime: Any) -> OrchestratorControlProtocol:
    """
    Retorna um adapter padronizado para qualquer runtime conhecido.

    Esta factory permite começar a migrar consumidores para contratos comuns
    sem acoplá-los diretamente às implementações concretas.
    """
    if hasattr(runtime, "process_market_data") and hasattr(runtime, "health_check"):
        return MarketOrchestratorAdapter(runtime)

    if hasattr(runtime, "run") and hasattr(runtime, "shutdown"):
        return EnhancedMarketBotAdapter(runtime)

    raise TypeError(
        "Unsupported orchestrator runtime for adaptation: "
        f"{type(runtime).__name__}"
    )
