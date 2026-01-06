# data_pipeline/__init__.py
"""
Data Pipeline Package v3.2.1

Pacote modular para processamento de dados de trading com:
- Validação vetorizada de trades
- Sistema adaptativo de thresholds
- Cache inteligente com TTL
- Logging granular
- Fallback automático
- ML features encapsulado
"""

from .config import PipelineConfig
from .logging_utils import PipelineLogger, setup_pipeline_logging
from .pipeline import DataPipeline

__version__ = "3.2.1"
__all__ = [
    "DataPipeline",
    "PipelineConfig",
    "PipelineLogger",
    "setup_pipeline_logging",
]