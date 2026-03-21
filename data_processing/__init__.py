# data_processing/__init__.py
"""
Pacote de processamento e validação de dados.

Contém: DataEnricher, DataValidator, DataQualityValidator,
FeatureStore, EnrichmentIntegrator, DataHandler.
"""

from .data_enricher import DataEnricher  # noqa: F401
from .data_handler import create_absorption_event, create_exhaustion_event  # noqa: F401
from .data_quality_validator import DataQualityValidator  # noqa: F401
from .data_validator import DataValidator  # noqa: F401
from .enrichment_integrator import (  # noqa: F401
    build_analysis_trigger_event,
    enrich_analysis_trigger_event,
)
from .feature_store import FeatureStore  # noqa: F401

__all__ = [
    "DataEnricher",
    "DataQualityValidator",
    "DataValidator",
    "FeatureStore",
    "build_analysis_trigger_event",
    "create_absorption_event",
    "create_exhaustion_event",
    "enrich_analysis_trigger_event",
]
