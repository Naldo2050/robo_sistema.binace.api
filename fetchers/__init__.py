# fetchers/__init__.py
"""
Pacote de fetchers de dados externos.

Contém: FREDFetcher, ContextCollector, MacroDataProvider, MacroService.
"""

from fetchers.fred_fetcher import FREDFetcher
from fetchers.context_collector import ContextCollector

__all__ = [
    "FREDFetcher",
    "ContextCollector",
]
