from __future__ import annotations

# tests/conftest.py
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Desabilitar métricas do Prometheus para evitar duplicação em testes
os.environ["ORDERBOOK_METRICS_ENABLED"] = "0"

import time
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import pytest

# Limpar registry do Prometheus antes dos testes
try:
    from prometheus_client import CollectorRegistry
    # Usar um registry limpo para os testes
    import prometheus_client
    prometheus_client.registry.REGISTRY = CollectorRegistry()
except ImportError:
    pass  # prometheus_client não disponível


@dataclass
class FakeTimeManager:
    now: int  # epoch_ms fixo

    def now_ms(self) -> int:
        return int(self.now)

    def build_time_index(
        self,
        epoch_ms: int,
        include_local: bool = False,
        timespec: str = "seconds",
    ) -> Dict[str, Any]:
        # suficiente para o analyzer (ny/utc podem ser None)
        return {
            "timestamp_ny": None,
            "timestamp_utc": None,
        }


def make_valid_snapshot(epoch_ms: int) -> Dict[str, Any]:
    # bids em ordem decrescente, asks em ordem crescente
    bids: List[Tuple[float, float]] = [
        (100.0, 5.0),
        (99.5, 5.0),
        (99.0, 5.0),
        (98.5, 5.0),
        (98.0, 5.0),
    ]
    asks: List[Tuple[float, float]] = [
        (100.5, 5.0),
        (101.0, 5.0),
        (101.5, 5.0),
        (102.0, 5.0),
        (102.5, 5.0),
    ]
    return {
        "E": int(epoch_ms),
        "bids": bids,
        "asks": asks,
    }


@pytest.fixture
def fixed_now_ms() -> int:
    # um timestamp fixo para evitar flakiness
    return 1_700_000_000_000


@pytest.fixture
def tm(fixed_now_ms: int) -> FakeTimeManager:
    return FakeTimeManager(now=fixed_now_ms)