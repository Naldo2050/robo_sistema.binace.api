# market_orchestrator/__init__.py
"""
Init leve para evitar imports pesados durante testes focados.
EnhancedMarketBot é carregado sob demanda via __getattr__.
"""

__all__ = [
    "EnhancedMarketBot",
    "adapt_orchestrator_runtime",
    "EnhancedMarketBotAdapter",
    "MarketOrchestratorAdapter",
]


def __getattr__(name):
    if name == "EnhancedMarketBot":
        from .market_orchestrator import EnhancedMarketBot

        return EnhancedMarketBot
    if name in {
        "adapt_orchestrator_runtime",
        "EnhancedMarketBotAdapter",
        "MarketOrchestratorAdapter",
    }:
        from .adapters import (
            EnhancedMarketBotAdapter,
            MarketOrchestratorAdapter,
            adapt_orchestrator_runtime,
        )

        exported = {
            "adapt_orchestrator_runtime": adapt_orchestrator_runtime,
            "EnhancedMarketBotAdapter": EnhancedMarketBotAdapter,
            "MarketOrchestratorAdapter": MarketOrchestratorAdapter,
        }
        return exported[name]
    raise AttributeError(f"module 'market_orchestrator' has no attribute {name}")
