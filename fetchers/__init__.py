# fetchers/__init__.py
"""
Pacote de fetchers de dados externos.

Contém: ContextCollector, FREDFetcher, MacroDataProvider,
FundingAggregator, OnchainFetcher.
"""

from .context_collector import ContextCollector  # noqa: F401
from .fred_fetcher import FREDFetcher  # noqa: F401
from .funding_aggregator import FundingAggregator  # noqa: F401
from .macro_data_provider import MacroDataProvider  # noqa: F401
from .onchain_fetcher import OnchainFetcher  # noqa: F401

__all__ = [
    "ContextCollector",
    "FREDFetcher",
    "FundingAggregator",
    "MacroDataProvider",
    "OnchainFetcher",
]
