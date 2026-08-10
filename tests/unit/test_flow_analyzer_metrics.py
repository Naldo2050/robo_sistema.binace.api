# tests/unit/test_flow_analyzer_metrics.py
"""
Testes das métricas Prometheus do FlowAnalyzer.

Cobrem o fix de wiring: CVD, whale_delta, flow_trades_count, trades_total
e trades_invalid_total agora são atualizados durante process_trade (antes
apenas record_ooo era chamado).

Protocolo de prova: cada teste processa trades reais via process_trade e
lê o REGISTRY padrão do prometheus_client (o mesmo que o servidor /metrics
expõe em produção). Se o wiring for revertido, os valores ficam em 0.0/None
e os testes falham.
"""

from decimal import Decimal
import time

import pytest

pytest.importorskip("prometheus_client")
from prometheus_client import REGISTRY

from flow_analyzer import FlowAnalyzer
from flow_analyzer.prometheus_metrics import PrometheusMetrics

CVD = "flow_analyzer_cvd"
WHALE_DELTA = "flow_analyzer_whale_delta"
FLOW_TRADES_COUNT = "flow_analyzer_flow_trades_count"
TRADES_TOTAL = "flow_analyzer_trades_total"
TRADES_INVALID_TOTAL = "flow_analyzer_trades_invalid_total"
OOO_TOTAL = "flow_analyzer_ooo_total"


def _unregister_flow_analyzer_metrics() -> None:
    """Remove todos os collectors flow_analyzer_* do REGISTRY global."""
    for name in list(REGISTRY._names_to_collectors):
        if name.startswith("flow_analyzer_"):
            try:
                REGISTRY.unregister(REGISTRY._names_to_collectors[name])
            except Exception:
                pass


def _sample(metric_name: str, labels: dict = None) -> float:
    """Valor da série com os labels (0.0 se a série nunca foi emitida)."""
    value = REGISTRY.get_sample_value(metric_name, labels)
    return value if value is not None else 0.0


def _counter_total(metric_name: str) -> float:
    """Soma de todas as séries de um Counter (sem depender dos labels).

    Nota: compara pelo nome da SÉRIE (ex: ..._total), pois o prometheus_client
    moderno normaliza o nome da família (remove o sufixo _total).
    """
    total = 0.0
    for family in REGISTRY.collect():
        for sample in family.samples:
            if sample.name == metric_name:
                total += sample.value
    return total


@pytest.fixture
def analyzer():
    """Analyzer com métricas Prometheus ativas e REGISTRY limpo."""
    _unregister_flow_analyzer_metrics()
    inst = FlowAnalyzer()
    if inst._prometheus is None:
        inst._prometheus = PrometheusMetrics()
    inst.whale_threshold = Decimal("5.0")
    yield inst
    _unregister_flow_analyzer_metrics()


class TestFlowAnalyzerPrometheusMetrics:
    """Métricas Prometheus refletem o processamento real de trades."""

    def test_cvd_whale_and_flow_trades_count_are_updated(self, analyzer):
        """Gauges de estado agregado refletem trades processados."""
        now = int(time.time() * 1000)
        analyzer.process_trade(
            {'p': 50000.0, 'q': 1.5, 'T': now, 'm': False}
        )
        analyzer.process_trade(
            {'p': 50000.0, 'q': 0.4, 'T': now + 100, 'm': True}
        )
        analyzer.process_trade(
            {'p': 50000.0, 'q': 10.0, 'T': now + 200, 'm': False}
        )

        assert _sample(CVD) == 11.1
        assert _sample(WHALE_DELTA) == 10.0
        assert _sample(FLOW_TRADES_COUNT) == 3

    def test_trades_total_and_invalid_counters_are_updated(self, analyzer):
        """Counters de trades válidos e inválidos refletem o fluxo."""
        now = int(time.time() * 1000)
        analyzer.process_trade(
            {'p': 50000.0, 'q': 1.5, 'T': now, 'm': False}
        )
        analyzer.process_trade(
            {'p': 50000.0, 'q': 0.4, 'T': now + 100, 'm': True}
        )
        analyzer.process_trade({'invalid': 'data'})

        assert _sample(TRADES_TOTAL, {"side": "buy", "sector": "whale"}) == 1.0
        assert _sample(TRADES_TOTAL, {"side": "sell", "sector": "retail"}) == 1.0
        assert _counter_total(TRADES_INVALID_TOTAL) == 1.0

    def test_record_ooo_still_works(self, analyzer):
        """record_ooo continua sendo chamado em trades out-of-order."""
        now = int(time.time() * 1000)
        analyzer.process_trade(
            {'p': 50000.0, 'q': 1.0, 'T': now, 'm': False}
        )
        analyzer.process_trade(
            {
                'p': 50000.0, 'q': 0.2, 'T': now + 100, 'm': True,
                'T_raw': now - 1000,
            }
        )

        assert _sample(OOO_TOTAL) == 1.0
        assert _sample(CVD) == 0.8
