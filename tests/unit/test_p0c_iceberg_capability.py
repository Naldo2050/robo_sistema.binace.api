# tests/unit/test_p0c_iceberg_capability.py — P0-C: capability enforcement p/ iceberg.
#
# Contrato (fonte de verdade: market_orchestrator.capabilities, nomes reais):
# - CONTINUOUS_TRADES_WS=True, POINT_IN_TIME_L2_SNAPSHOT=True (REST ~60s)
# - CONTINUOUS_L2=False, ORDER_CANCEL_TRACKING=False, QUEUE_REPLENISHMENT=False
# - ICEBERG_DETECTION_SUPPORTED=False, SPOOFING_DETECTION_SUPPORTED=False,
#   HIDDEN_ORDERS_SUPPORTED=False
# Com ICEBERG_DETECTION_SUPPORTED=False, nenhuma heurística de snapshots pode
# ser apresentada como iceberg confirmado/recarregado. Telemetria heurística
# permanece, explicitamente UNCONFIRMED. Orderbook (walls, imbalance, slippage,
# depth, market impact) intacto.

import pytest

from market_orchestrator import capabilities as cap
from orderbook_analyzer import OrderBookAnalyzer
from orderbook_analyzer.core import iceberg_detection_confirmed
from tests.conftest import make_valid_snapshot


# ── F. Capabilities (pin do contrato) ────────────────────────────────────────

def test_f_capability_contract_pins_iceberg_unsupported():
    """Se alguém ligar ICEBERG sem L2 contínuo, este teste quebra de propósito:
    mudar a capability exige atualizar o contrato (e este arquivo)."""
    assert cap.CONTINUOUS_TRADES_WS is True
    assert cap.POINT_IN_TIME_L2_SNAPSHOT is True
    assert cap.CONTINUOUS_L2 is False
    assert cap.ORDER_CANCEL_TRACKING is False
    assert cap.QUEUE_REPLENISHMENT_TRACKING is False
    assert cap.ICEBERG_DETECTION_SUPPORTED is False
    assert cap.SPOOFING_DETECTION_SUPPORTED is False
    assert cap.HIDDEN_ORDERS_SUPPORTED is False
    assert iceberg_detection_confirmed() is False


# ── A. Nenhum claim confirmado com ICEBERG=False ─────────────────────────────

def test_a_labels_never_emit_reload_alert_when_unconfirmed(tm):
    oba = OrderBookAnalyzer(symbol="BTCUSDT", time_manager=tm)
    _, alertas, _, _ = oba._build_labels_and_alerts(
        imbalance=0.1, iceberg=False, spread_bps=5.0,
        ratio=1.1, bid_usd=100000.0, ask_usd=90000.0)
    assert not any("Iceberg" in a for a in alertas)


@pytest.mark.asyncio
async def test_a_event_fail_closed_even_when_heuristic_fires(tm):
    """Dois snapshots REST: o 2º recarrega nível (raw dispara), mas o evento
    sai confirmado=False/0.0/UNSUPPORTED com heurística preservada."""
    oba = OrderBookAnalyzer(symbol="BTCUSDT", time_manager=tm)
    now = tm.now_ms()

    snap1 = make_valid_snapshot(now)
    evt1 = await oba.analyze(current_snapshot=snap1, event_epoch_ms=now)
    assert evt1["is_valid"] is True

    snap2 = make_valid_snapshot(now)
    snap2["bids"] = [[float(p), float(q)] for p, q in snap2["bids"]]
    snap2["bids"][0][1] = snap2["bids"][0][1] * 5.0  # recarga 5x no top bid
    evt2 = await oba.analyze(current_snapshot=snap2, event_epoch_ms=now)

    assert evt2["is_valid"] is True
    # Confirmado: sempre false/zero.
    assert evt2["iceberg_reloaded"] is False
    assert evt2["iceberg_score"] == 0.0
    assert evt2["iceberg_status"] == "UNSUPPORTED"
    # Nenhum alerta confirmado de recarga.
    assert not any("Iceberg" in a for a in (evt2.get("alertas_liquidez") or []))
    # Heurística: o raw disparou e foi preservado como não-confirmatório.
    heur = evt2["iceberg_heuristic"]
    assert heur["validity"] == "UNCONFIRMED"
    assert heur["reason"] == "CONTINUOUS_L2_UNAVAILABLE"
    assert heur["value"] > 0.0


# ── B. Heurística não alimenta direção ───────────────────────────────────────

def test_b_heuristic_algorithm_intact_but_non_directional(tm):
    """Algoritmo _compute_iceberg inalterado (regressão em test_orderbook_helpers
    continua verde); aqui prova-se que o raw dispara no fixture do teste A."""
    oba = OrderBookAnalyzer(symbol="BTCUSDT", time_manager=tm)
    oba.prev_snapshot = {"bids": [(84000.0, 2.0)], "asks": [(84010.0, 2.0)]}
    raw, raw_score = oba._compute_iceberg([(84000.0, 10.0)], [(84010.0, 2.0)])
    assert raw is True and raw_score > 0.0


@pytest.mark.asyncio
async def test_b_heuristic_does_not_feed_bias(tm):
    """bias_score permanece função pura de imbalance+ratio (linhas do produtor);
    campos iceberg não entram no cálculo."""
    oba = OrderBookAnalyzer(symbol="BTCUSDT", time_manager=tm)
    now = tm.now_ms()
    snap2 = make_valid_snapshot(now)
    snap2["bids"] = [[float(p), float(q)] for p, q in snap2["bids"]]
    snap2["bids"][0][1] *= 5.0
    await oba.analyze(current_snapshot=make_valid_snapshot(now), event_epoch_ms=now)
    evt2 = await oba.analyze(current_snapshot=snap2, event_epoch_ms=now)
    imb = evt2["flow_imbalance"]
    ratio = evt2["volume_ratio"]
    expected = 0.5 + (imb * 0.3)
    if ratio and ratio > 0:
        import math
        ratio_adj = min(1.0, max(-1.0, (ratio - 1.0) / 2.0))
        if math.isfinite(ratio_adj):
            expected += ratio_adj * 0.2
    expected = min(1.0, max(0.0, expected))
    assert evt2["orderbook_data"]["consolidated_bias_score"] == pytest.approx(
        round(expected, 4))
    assert evt2["iceberg_heuristic"]["validity"] == "UNCONFIRMED"


# ── C. Regressão: resto do orderbook idêntico ────────────────────────────────

@pytest.mark.asyncio
async def test_c_walls_imbalance_slippage_intact(tm):
    oba = OrderBookAnalyzer(symbol="BTCUSDT", time_manager=tm)
    now = tm.now_ms()
    evt = await oba.analyze(
        current_snapshot=make_valid_snapshot(now), event_epoch_ms=now)
    assert evt["is_valid"] is True
    # Imbalance = (bid-ask)/(bid+ask) do snapshot simétrico do fixture.
    assert evt["flow_imbalance"] == pytest.approx(0.0, abs=0.05)
    # Estrutura walls/slippage/market-impact intacta (detecção de walls em si
    # coberta por test_orderbook_helpers::test_detect_walls_simple; o fixture
    # simétrico legitimamente não tem parede dominante).
    assert isinstance(evt["walls"]["bids"], list)
    assert isinstance(evt["walls"]["asks"], list)
    assert "100k" in (evt.get("market_impact_buy") or {})
    assert "100k" in (evt.get("market_impact_sell") or {})
    assert evt["iceberg_status"] == "UNSUPPORTED"
    assert evt["iceberg_heuristic"]["validity"] == "UNCONFIRMED"


# ── D. Payload/legend ────────────────────────────────────────────────────────

def test_d_payload_omits_iceberg_and_legend_explains():
    from market_orchestrator.ai.payload_builder_compact import build_compact_payload
    from common.ai_field_legend import FIELD_LEGEND

    evt = {"symbol": "BTCUSDT",
           "orderbook_data": {"bid_depth_usd": 1.0, "ask_depth_usd": 1.0,
                              "imbalance": 0.0},
           "iceberg_reloaded": False, "iceberg_score": 0.0,
           "iceberg_status": "UNSUPPORTED",
           "iceberg_heuristic": {"value": 1.0, "validity": "UNCONFIRMED",
                                 "reason": "CONTINUOUS_L2_UNAVAILABLE"}}
    payload = build_compact_payload(evt)
    assert "iceberg" not in payload
    assert "UNSUPPORTED" in FIELD_LEGEND
    assert "UNCONFIRMED" in FIELD_LEGEND


# ── E. Busca estática: alerta só existe atrás do gate ────────────────────────

def test_e_reload_alert_exists_only_behind_capability_gate():
    import pathlib
    src = pathlib.Path("orderbook_analyzer/core.py").read_text(encoding="utf-8")
    assert src.count("Iceberg possivelmente recarregando") == 1
    # O único uso está no produtor cujo caller aplica o gate P0-C...
    assert "iceberg_detection_confirmed" in src
    # ...e o compact gateia a outra ponta (defesa em profundidade, pré-existente).
    compact = pathlib.Path(
        "market_orchestrator/ai/payload_builder_compact.py").read_text(
            encoding="utf-8")
    assert "ICEBERG_DETECTION_SUPPORTED" in compact
