"""
payload_sections — Summary builders para o payload compacto da IA.

Cada módulo transforma dados brutos de um domínio em uma conclusão
curta e interpretada, reduzindo ambiguidade para o modelo.

Uso:
    from market_orchestrator.ai.payload_sections import (
        build_flow_summary,
        build_sr_summary,
        build_regime_summary,
        build_institutional_summary,
        build_quality_summary,
    )
"""

from .flow_summary import build_flow_summary
from .sr_summary import build_sr_summary
from .regime_summary import build_regime_summary
from .institutional_summary import build_institutional_summary
from .quality_summary import build_quality_summary

__all__ = [
    "build_flow_summary",
    "build_sr_summary",
    "build_regime_summary",
    "build_institutional_summary",
    "build_quality_summary",
]
