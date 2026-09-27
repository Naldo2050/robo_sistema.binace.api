"""
orderbook_analyzer — pacote de análise de orderbook.
OrderBookAnalyzer (produção, v2.2.0) vem de .core.
SimplifiedOrderBookAnalyzer (legado/testes) vem de .legacy_simplified.
"""
from .core import (
    OrderBookAnalyzer,
    _to_float_list,
    _sum_depth_usd,
    _simulate_market_impact,
)
from .directional_liquidity import build_directional_liquidity
from .spread_tracker import SpreadTracker

__all__ = [
    "OrderBookAnalyzer",
    "SpreadTracker",
    "SimplifiedOrderBookAnalyzer",
    "_to_float_list",
    "_sum_depth_usd",
    "_simulate_market_impact",
    "build_directional_liquidity",
]


def __getattr__(name: str):
    # Import lazy: .legacy_simplified só é carregado quando solicitado.
    # Assim o DeprecationWarning dispara apenas para quem importa o caminho
    # legado (.legacy_simplified ou .analyzer) diretamente — nunca no boot
    # do pacote via orderbook_analyzer.
    if name == "SimplifiedOrderBookAnalyzer":
        from .legacy_simplified import SimplifiedOrderBookAnalyzer
        return SimplifiedOrderBookAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
