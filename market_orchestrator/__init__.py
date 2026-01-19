# market_orchestrator/__init__.py
"""
Init leve para evitar imports pesados durante testes focados.
EnhancedMarketBot é carregado sob demanda via __getattr__.
"""

__all__ = ["EnhancedMarketBot"]


def __getattr__(name):
    if name == "EnhancedMarketBot":
        from .market_orchestrator import EnhancedMarketBot

        return EnhancedMarketBot
    raise AttributeError(f"module 'market_orchestrator' has no attribute {name}")
