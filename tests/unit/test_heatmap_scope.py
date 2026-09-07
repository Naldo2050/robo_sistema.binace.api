# tests/unit/test_heatmap_scope.py
"""
Regressão (Liquidity Heatmap — escopo rolling vs janela):
o heatmap acumula rolling window de N trades (default 2000), mas o log o
chamava de "Janela #N" e o payload da IA não declarava o escopo — corrupção
silenciosa de contexto (a IA lia Vol/Trades rolling como se fossem da janela).

Contrato exigido (sem mudar o algoritmo nem restringir à janela):
  - FlowAnalyzer -> fluxo_continuo.liquidity_heatmap carrega
    scope_type="rolling_trades" + scope_size=<window_size efetivo> (lido da
    instância, nunca hardcoded).
  - build_compact_payload emite payload["liq_scope"] com o mesmo escopo.
  - O escopo sobrevive ao guardrail (whitelist) e ao groq summary
    (passthrough) até o payload FINAL enviado ao modelo.
  - O log humano rotula como rolling, não "Janela #N".
"""

import logging

import pytest

from flow_analyzer import FlowAnalyzer
from market_analysis.liquidity_heatmap import LiquidityHeatmap
from market_orchestrator.ai.payload_builder_compact import build_compact_payload
from market_orchestrator.ai.llm_payload_guardrail import guardrail_rewrap
from market_orchestrator.ai.analyzer_qwen import AIAnalyzer


class _FakeClock:
    def __init__(self, start_ms=1_700_000_000_000):
        self._now = start_ms

    def now_ms(self):
        return self._now


def _flow_with_heatmap(window_size=13):
    flow = FlowAnalyzer(time_manager=_FakeClock())
    flow.liquidity_heatmap = LiquidityHeatmap(
        window_size=window_size,
        cluster_threshold_pct=0.005,
        min_trades_per_cluster=3,
        update_interval_ms=0,
    )
    base_ms = 1_700_000_000_000
    for i in range(10):
        flow.liquidity_heatmap.add_trade(
            price=79421.0 + (i % 3),
            volume=0.5,
            side="buy" if i % 2 == 0 else "sell",
            timestamp_ms=base_ms + i * 1000,
        )
    return flow


def _event_with_heatmap(hm):
    return {
        "symbol": "BTCUSDT",
        "tipo_evento": "ANALYSIS_TRIGGER",
        "preco_fechamento": 79421.2,
        "epoch_ms": 1_700_000_060_000,
        "fluxo_continuo": {"liquidity_heatmap": hm},
    }


def test_producer_emits_scope_from_instance():
    """scope_size vem da instância (13), nunca hardcoded."""
    flow = _flow_with_heatmap(window_size=13)
    hm = flow._get_heatmap_data(1_700_000_060_000)
    assert hm["scope_type"] == "rolling_trades"
    assert hm["scope_size"] == 13
    assert hm["clusters"], "esperava ao menos 1 cluster com 10 trades"


def test_scope_survives_to_final_llm_payload():
    """Cadeia completa até o payload FINAL enviado ao modelo."""
    flow = _flow_with_heatmap(window_size=13)
    hm = flow._get_heatmap_data(1_700_000_060_000)

    compact = build_compact_payload(_event_with_heatmap(hm))
    assert compact.get("liq"), "clusters deveriam gerar payload['liq']"
    assert compact.get("liq_scope") == {
        "scope_type": "rolling_trades",
        "scope_size": 13,
    }

    rewrapped = guardrail_rewrap(compact)
    assert rewrapped["ai_payload"].get("liq_scope") == {
        "scope_type": "rolling_trades",
        "scope_size": 13,
    }, "guardrail descartou liq_scope (whitelist?)"

    final = AIAnalyzer._build_groq_payload_summary(rewrapped["ai_payload"])
    assert final.get("liq_scope") == {
        "scope_type": "rolling_trades",
        "scope_size": 13,
    }, "groq summary descartou liq_scope (passthrough?)"
    assert final.get("liq"), "clusters devem chegar ao payload final"


def test_log_labels_rolling_not_window(caplog):
    """O log humano não deve chamar o heatmap rolling de 'Janela #N'."""
    from market_orchestrator.market_orchestrator import EnhancedMarketBot

    flow = _flow_with_heatmap(window_size=13)
    hm = flow._get_heatmap_data(1_700_000_060_000)
    bot = type("FakeBot", (), {"window_count": 7})()

    with caplog.at_level(logging.INFO, logger="root"):
        EnhancedMarketBot._log_liquidity_heatmap(bot, {"liquidity_heatmap": hm})

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "rolling(13 trades)" in text
    # O rótulo antigo chamava o dado rolling de "Janela #N" (sem escopo)
    assert "HEATMAP - Janela #7:" not in text
