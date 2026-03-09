
import sys
import os

# Adicionar raiz do projeto ao path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pytest
from unittest.mock import patch, MagicMock


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