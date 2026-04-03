
import asyncio
import sys
import os
import time
from datetime import timezone, timedelta

# Adicionar raiz do projeto ao path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pytest
from unittest.mock import patch, MagicMock

try:
    from zoneinfo import ZoneInfo
    _TZ_UTC = ZoneInfo("UTC")
    _TZ_NY = ZoneInfo("America/New_York")
    _TZ_SP = ZoneInfo("America/Sao_Paulo")
except Exception:
    _TZ_UTC = timezone.utc
    _TZ_NY = timezone(timedelta(hours=-4))
    _TZ_SP = timezone(timedelta(hours=-3))


# ══════════════════════════════════════════════════════════════════
# FIX: Prometheus Registry duplicado entre testes
# ══════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def clean_prometheus_registry():
    """
    Limpa métricas duplicadas do Prometheus entre testes.
    Evita: ValueError: Duplicated timeseries in CollectorRegistry
    """
    try:
        from prometheus_client import REGISTRY

        collectors_before = set(REGISTRY._names_to_collectors.keys())

        yield

        collectors_after = set(REGISTRY._names_to_collectors.keys())
        new_names = collectors_after - collectors_before

        seen = set()
        for name in new_names:
            try:
                collector = REGISTRY._names_to_collectors.get(name)
                if collector and id(collector) not in seen:
                    seen.add(id(collector))
                    REGISTRY.unregister(collector)
            except Exception:
                pass

    except ImportError:
        yield


# ══════════════════════════════════════════════════════════════════
# FIXTURE: TimeManager mockado (sem rede)
# ══════════════════════════════════════════════════════════════════

@pytest.fixture
def tm():
    """TimeManager com sync inicial mockado — sem chamadas de rede."""
    from monitoring.time_manager import TimeManager

    with patch.object(TimeManager, "_initialize_sync"):
        instance = object.__new__(TimeManager)
        instance._initialized = True
        instance.sync_interval_seconds = 1800
        instance.max_acceptable_offset_ms = 600
        instance.max_init_attempts = 3
        instance.num_sync_samples = 5
        instance.server_time_offset_ms = 0
        instance.last_sync_mono = time.monotonic()
        instance.sync_attempts = 0
        instance.sync_failures = 0
        instance.last_successful_sync_ms = int(time.time() * 1000)
        instance.last_offset_ms = 0
        instance.best_rtt_ms = 50
        instance.last_rtt_ms = 50
        instance.time_sync_status = "ok"
        instance.auto_corrections = 0
        instance._correction_attempts = 0
        instance._last_offset_history = []
        instance._lock = __import__("threading").Lock()
        instance._sync_needed = False
        instance._sync_lock = asyncio.Lock()
        instance.tz_utc = _TZ_UTC
        instance.tz_ny = _TZ_NY
        instance.tz_sp = _TZ_SP
        if not hasattr(instance, "CRITICAL_OFFSET_MS"):
            instance.CRITICAL_OFFSET_MS = 60000
        if not hasattr(instance, "WARNING_OFFSET_MS"):
            instance.WARNING_OFFSET_MS = 30000
        if not hasattr(instance, "MAX_CORRECTION_ATTEMPTS"):
            instance.MAX_CORRECTION_ATTEMPTS = 3
        yield instance


# ══════════════════════════════════════════════════════════════════
# HELPER: snapshot de orderbook válido para testes
# ══════════════════════════════════════════════════════════════════

def make_valid_snapshot(timestamp_ms: int) -> dict:
    """
    Cria um snapshot de orderbook válido (dict) com depth suficiente para passar
    na validação de min_depth_usd padrão (>1000 USD por lado).

    Retorna dict no formato que _validate_snapshot espera:
    {"E": ts_ms, "lastUpdateId": ..., "symbol": ..., "bids": [...], "asks": [...]}
    """
    # ~84000 USDT × 2 BTC por nível × 5 níveis = ~840k USD por lado
    base_price = 84_000.0
    bids = [
        [base_price - i * 10, 2.0]
        for i in range(5)
    ]
    asks = [
        [base_price + 10 + i * 10, 2.0]
        for i in range(5)
    ]
    return {
        "E": timestamp_ms,
        "lastUpdateId": 1_000_000,
        "symbol": "BTCUSDT",
        "bids": bids,
        "asks": asks,
    }