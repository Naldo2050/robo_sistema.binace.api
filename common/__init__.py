# common/__init__.py
"""
Pacote de utilitários comuns.

Contém: exceptions, format_utils, ml_features, technical_indicators,
ai_throttler, report_generator.
"""

from .exceptions import BotBaseError  # noqa: F401
from .format_utils import (  # noqa: F401
    format_delta,
    format_large_number,
    format_percent,
    format_price,
    format_quantity,
    format_time_seconds,
)
from .ml_features import (  # noqa: F401
    calculate_cross_asset_features,
    generate_ml_features,
)
from .report_generator import generate_ai_analysis_report  # noqa: F401
from .ai_throttler import SmartAIThrottler  # noqa: F401

__all__ = [
    "BotBaseError",
    "SmartAIThrottler",
    "calculate_cross_asset_features",
    "format_delta",
    "format_large_number",
    "format_percent",
    "format_price",
    "format_quantity",
    "format_time_seconds",
    "generate_ai_analysis_report",
    "generate_ml_features",
]
