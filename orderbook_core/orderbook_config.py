# orderbook_config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class OrderBookConfig:
    """
    Configuração estruturada do OrderBookAnalyzer.

    Esta classe NÃO importa `config.py` diretamente.
    Ela apenas define a estrutura. Os valores são
    preenchidos pelo chamador (ex.: orderbook_analyzer.py)
    com base em `config` e/ou defaults.
    """

    depth_levels: List[int]
    spread_tight_threshold_bps: float
    spread_avg_windows_min: List[int]

    critical_imbalance: float
    min_dominant_usd: float
    min_ratio_dom: float

    request_timeout: float
    retry_delay: float
    max_retries: int
    max_requests_per_min: int

    cache_ttl: float
    max_stale: float

    min_depth_usd: float
    allow_partial: bool
    use_fallback: bool
    fallback_max_age: float
    emergency_mode: bool