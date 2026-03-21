# data_processing/__init__.py
"""
Pacote de processamento e validacao de dados.

Contém: DataHandler, DataEnricher, DataValidator, DataQualityValidator, FeatureStore.
"""

from data_processing.data_enricher import DataEnricher
from data_processing.data_validator import DataValidator
from data_processing.feature_store import FeatureStore

__all__ = [
    "DataEnricher",
    "DataValidator",
    "FeatureStore",
]
