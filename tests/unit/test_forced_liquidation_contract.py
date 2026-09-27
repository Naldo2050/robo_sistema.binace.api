# tests/unit/test_forced_liquidation_contract.py
# -*- coding: utf-8 -*-
"""
Testes unitários rigorosos para P2-D — Binance Forced Liquidations Telemetry Contract v1.

Cobre:
1. SELL forceOrder -> liquidated LONG
2. BUY forceOrder -> liquidated SHORT
3. filled qty vs original qty
4. average execution price com fallback documentado
5. observed notional (filled_qty * average_price)
6. partial event (notional não computável)
7. invalid side / non-finite / malformed
8. deterministic dedup (event_id estável)
9. duplicate event not double counted
10. long aggregation
11. short aggregation
12. zero observed on healthy stream
13. missing on unhealthy stream
14. event-time window boundaries ([window_start, window_end))
15. no future heatmap fields
16. no bullish/bearish/reversal strings
17. Evidence counts_as_vote=false, direction=UNKNOWN, calibration=NOT_APPLICABLE
18. Confluence ignores liquidation fields
19. RFC 8259 JSON compliance
20. Benchmark do parser e agregador
"""

import json
import time
import pytest

from fetchers.binance_liquidation_stream import (
    STREAM_CAPABILITY,
    ForcedLiquidationEvent,
    LargerObservedSide,
    LiquidatedPositionSide,
    LiquidationValidity,
    LiquidationWindowAggregator,
    LiquidationWindowSummary,
    OrderSide,
    liquidation_summary_to_evidence,
    parse_force_order_payload,
)
from institutional.confluence_shadow import (
    ReconcilerStatus,
    reconcile,
)
from institutional.evidence import (
    Evidence,
    EvidenceCalibration,
    EvidenceDirection,
    EvidenceFamily,
    EvidenceType,
    EvidenceValidity,
)


# ── 1. SEMÂNTICA DO SIDE (SELL -> LONG, BUY -> SHORT) ─────────────────────────

def test_sell_force_order_means_liquidated_long():
    """Uma ordem de venda forçada (SELL) é gerada pelo fechamento de um LONG liquidado."""
    raw = {
        "e": "forceOrder",
        "E": 1720000005000,
        "o": {
            "s": "BTCUSDT",
            "S": "SELL",
            "o": "LIMIT",
            "f": "IOC",
            "q": "1.500",
            "p": "62000.00",
            "ap": "61990.50",
            "X": "FILLED",
            "l": "1.500",
            "z": "1.500",
            "T": 1720000005000,
        }
    }
    event = parse_force_order_payload(raw)

    assert event.validity == LiquidationValidity.VALID
    assert event.order_side == OrderSide.SELL.value
    assert event.liquidated_position_side == LiquidatedPositionSide.LONG.value
    assert event.filled_qty == 1.5
    assert event.average_price == 61990.50
    assert event.observed_notional_usd == round(1.5 * 61990.50, 2)
    assert event.capability == STREAM_CAPABILITY


def test_buy_force_order_means_liquidated_short():
    """Uma ordem de compra forçada (BUY) é gerada pelo fechamento de um SHORT liquidado."""
    raw = {
        "e": "forceOrder",
        "E": 1720000010000,
        "o": {
            "s": "BTCUSDT",
            "S": "BUY",
            "o": "LIMIT",
            "f": "IOC",
            "q": "0.750",
            "p": "63100.00",
            "ap": "63105.00",
            "X": "FILLED",
            "l": "0.750",
            "z": "0.750",
            "T": 1720000010000,
        }
    }
    event = parse_force_order_payload(raw)

    assert event.validity == LiquidationValidity.VALID
    assert event.order_side == OrderSide.BUY.value
    assert event.liquidated_position_side == LiquidatedPositionSide.SHORT.value
    assert event.filled_qty == 0.75
    assert event.average_price == 63105.00
    assert event.observed_notional_usd == round(0.75 * 63105.00, 2)


# ── 2. FILLED QTY VS ORIGINAL QTY E PREÇO DE EXECUÇÃO ─────────────────────────

def test_filled_qty_prioritized_over_orig_qty():
    """Prioriza z (filled acumulado) mesmo se q (original) for diferente."""
    raw = {
        "e": "forceOrder",
        "E": 1720000015000,
        "o": {
            "s": "BTCUSDT",
            "S": "SELL",
            "q": "10.000",      # Original submetido
            "z": "2.500",       # Executado observado
            "ap": "60000.00",
            "p": "60000.00",
            "X": "PARTIALLY_FILLED",
            "T": 1720000015000,
        }
    }
    event = parse_force_order_payload(raw)

    assert event.original_qty == 10.0
    assert event.filled_qty == 2.5
    assert event.observed_notional_usd == 150_000.0


def test_limit_price_fallback_when_avg_price_zero():
    """Fallback documentado para p (preço limite) quando ap (average price) é 0 ou ausente."""
    raw = {
        "e": "forceOrder",
        "E": 1720000020000,
        "o": {
            "s": "BTCUSDT",
            "S": "BUY",
            "q": "1.000",
            "z": "1.000",
            "ap": "0",          # average price zero
            "p": "61500.00",    # fallback
            "X": "FILLED",
            "T": 1720000020000,
        }
    }
    event = parse_force_order_payload(raw)

    assert event.validity == LiquidationValidity.VALID
    assert event.average_price == 61500.00
    assert event.observed_notional_usd == 61500.00
    assert "LIMIT_PRICE_FALLBACK" in event.reason


# ── 3. CASOS DE ERRO E VALIDADE (PARTIAL / INVALID / NON-FINITE) ──────────────

def test_partial_event_when_notional_cannot_be_calculated():
    """Evento real onde faltam preço ou quantidade resulta em PARTIAL (sem quebrar)."""
    raw = {
        "e": "forceOrder",
        "E": 1720000025000,
        "o": {
            "s": "BTCUSDT",
            "S": "SELL",
            "q": None,
            "z": None,
            "p": None,
            "ap": None,
            "T": 1720000025000,
        }
    }
    event = parse_force_order_payload(raw)

    assert event.validity == LiquidationValidity.PARTIAL
    assert event.observed_notional_usd is None
    assert "NOTIONAL_CANNOT_BE_CALCULATED" in event.reason


def test_invalid_side_or_corrupted_data():
    """Lado desconhecido ou dados corrompidos resultam em INVALID."""
    # Lado inválido
    raw_bad_side = {
        "o": {"s": "BTCUSDT", "S": "UNKNOWN_SIDE", "q": "1.0", "p": "60000", "T": 1720000000}
    }
    ev1 = parse_force_order_payload(raw_bad_side)
    assert ev1.validity == LiquidationValidity.INVALID
    assert "INVALID_SIDE" in ev1.reason

    # Non-finite NaN
    raw_nan = {
        "o": {"s": "BTCUSDT", "S": "SELL", "q": float("nan"), "p": "60000", "T": 1720000000}
    }
    ev2 = parse_force_order_payload(raw_nan)
    assert ev2.validity == LiquidationValidity.INVALID

    # Quantidade negativa
    raw_neg = {
        "o": {"s": "BTCUSDT", "S": "SELL", "z": "-1.5", "p": "60000", "T": 1720000000}
    }
    ev3 = parse_force_order_payload(raw_neg)
    assert ev3.validity == LiquidationValidity.INVALID


# ── 4. DEDUPLICAÇÃO DETERMINÍSTICA ────────────────────────────────────────────

def test_deterministic_dedup_and_no_double_counting():
    """Eventos repetidos com o mesmo id físico não podem ser contados duas vezes."""
    agg = LiquidationWindowAggregator(symbol="BTCUSDT")

    raw_event = {
        "e": "forceOrder",
        "E": 1720000030000,
        "o": {
            "s": "BTCUSDT",
            "S": "SELL",
            "q": "2.000",
            "z": "2.000",
            "ap": "60000.00",
            "p": "60000.00",
            "X": "FILLED",
            "T": 1720000030000,
        }
    }

    # Primeira inserção: aceita
    ok1 = agg.add_event(raw_event)
    assert ok1 is True

    # Segunda inserção (duplicata exata de rede / reconnect): rejeitada
    ok2 = agg.add_event(raw_event)
    assert ok2 is False

    summary = agg.summarize_window(1720000000000, 1720000060000, stream_healthy=True)
    assert summary.event_count == 1
    assert summary.long_liquidated_notional_usd == 120_000.0
    assert summary.total_liquidated_notional_usd == 120_000.0


# ── 5. AGREGAÇÃO POR JANELA TEMPORAL ──────────────────────────────────────────

def test_window_aggregation_long_and_short():
    """Agrega eventos separando long_notional e short_notional contemporâneos."""
    agg = LiquidationWindowAggregator(symbol="BTCUSDT")

    t_base = 1720000000000

    # Evento 1: Long liquidado (SELL order) 1.0 BTC @ 60,000 = $60,000
    agg.add_event({
        "o": {"s": "BTCUSDT", "S": "SELL", "z": "1.0", "ap": "60000.0", "X": "FILLED", "T": t_base + 10_000}
    })
    # Evento 2: Long liquidado (SELL order) 2.0 BTC @ 60,000 = $120,000
    agg.add_event({
        "o": {"s": "BTCUSDT", "S": "SELL", "z": "2.0", "ap": "60000.0", "X": "FILLED", "T": t_base + 20_000}
    })
    # Evento 3: Short liquidado (BUY order) 0.5 BTC @ 62,000 = $31,000
    agg.add_event({
        "o": {"s": "BTCUSDT", "S": "BUY", "z": "0.5", "ap": "62000.0", "X": "FILLED", "T": t_base + 30_000}
    })
    # Evento 4: Fora da janela (t_base + 70s): não deve entrar
    agg.add_event({
        "o": {"s": "BTCUSDT", "S": "SELL", "z": "5.0", "ap": "60000.0", "X": "FILLED", "T": t_base + 70_000}
    })

    summary = agg.summarize_window(t_base, t_base + 60_000, stream_healthy=True)

    assert summary.event_count == 3
    assert summary.long_liquidated_qty == 3.0
    assert summary.short_liquidated_qty == 0.5
    assert summary.long_liquidated_notional_usd == 180_000.0
    assert summary.short_liquidated_notional_usd == 31_000.0
    assert summary.total_liquidated_notional_usd == 211_000.0
    assert summary.larger_observed_side == LargerObservedSide.LONG.value
    assert summary.latest_event_ms == t_base + 30_000
    assert summary.validity == LiquidationValidity.VALID


# ── 6. HEALTHY STREAM ZERO VS UNHEALTHY STREAM MISSING ────────────────────────

def test_zero_observed_on_healthy_stream_vs_missing_on_unhealthy():
    """Garante a distinção fundamental entre zero observado e desconexão."""
    agg = LiquidationWindowAggregator(symbol="BTCUSDT")
    t0 = 1720000000000
    t1 = t0 + 60_000

    # Cenário A: Stream saudável e zero liquidações na janela (calmaria do mercado)
    healthy_summary = agg.summarize_window(t0, t1, stream_healthy=True)
    assert healthy_summary.validity == LiquidationValidity.VALID
    assert healthy_summary.event_count == 0
    assert healthy_summary.long_liquidated_notional_usd == 0.0
    assert healthy_summary.short_liquidated_notional_usd == 0.0
    assert healthy_summary.total_liquidated_notional_usd == 0.0
    assert healthy_summary.reason == "ZERO_OBSERVED_HEALTHY_STREAM"
    assert healthy_summary.larger_observed_side == LargerObservedSide.NONE.value

    # Cenário B: Stream desconectado / não saudável
    unhealthy_summary = agg.summarize_window(t0, t1, stream_healthy=False)
    assert unhealthy_summary.validity == LiquidationValidity.UNKNOWN
    assert unhealthy_summary.event_count is None
    assert unhealthy_summary.long_liquidated_notional_usd is None
    assert unhealthy_summary.short_liquidated_notional_usd is None
    assert unhealthy_summary.total_liquidated_notional_usd is None
    assert unhealthy_summary.reason == "STREAM_UNHEALTHY_OR_DISCONNECTED"


# ── 7. AUSÊNCIA DE CAMPOS FUTUROS E STRINGS DIRECIONAIS ───────────────────────

def test_no_future_heatmap_and_no_directional_strings():
    """Verifica que o contrato não inventa heatmaps futuros nem strings de sinal."""
    agg = LiquidationWindowAggregator(symbol="BTCUSDT")
    summary = agg.summarize_window(1720000000000, 1720000060000, stream_healthy=True)
    d = summary.to_dict()

    # Nenhuma chave de predição de níveis futuros ou clusters magnet
    forbidden_keys = [
        "liquidation_heatmap",
        "predicted_levels",
        "magnet_clusters",
        "coinglass",
        "coinmetrics",
        "reversal_probability",
    ]
    for k in forbidden_keys:
        assert k not in d

    # larger_observed_side deve ser puramente descritivo (LONG/SHORT/EQUAL/NONE)
    assert d["larger_observed_side"] in ("LONG", "SHORT", "EQUAL", "NONE")
    assert "bullish" not in str(d).lower()
    assert "bearish" not in str(d).lower()
    assert "reversal" not in str(d).lower()


# ── 8. EVIDENCE ADAPTERS V1 E DESCARTE DE VOTO ────────────────────────────────

def test_evidence_contract_non_voting_and_direction_unknown():
    """Evidências de liquidação são estritamente NON_VOTING e direction=UNKNOWN."""
    summary = LiquidationWindowSummary(
        window_start_ms=1720000000000,
        window_end_ms=1720000060000,
        event_count=2,
        long_liquidated_qty=1.0,
        short_liquidated_qty=0.0,
        long_liquidated_notional_usd=60_000.0,
        short_liquidated_notional_usd=0.0,
        total_liquidated_notional_usd=60_000.0,
        latest_event_ms=1720000030000,
        validity=LiquidationValidity.VALID,
    )

    evidences = liquidation_summary_to_evidence(summary, symbol="BTCUSDT")
    assert len(evidences) == 4

    expected_fids = [
        "derivatives.liquidations.long_observed_notional",
        "derivatives.liquidations.short_observed_notional",
        "derivatives.liquidations.total_observed_notional",
        "derivatives.liquidations.event_count",
    ]
    actual_fids = [ev.metadata["field_id"] for ev in evidences]
    assert actual_fids == expected_fids

    for ev in evidences:
        assert ev.family == EvidenceFamily.DERIVATIVES
        assert ev.evidence_type == EvidenceType.SLOW_CONTEXT
        assert ev.counts_as_vote is False
        assert ev.direction == EvidenceDirection.UNKNOWN
        assert ev.calibration == EvidenceCalibration.NOT_APPLICABLE
        assert ev.validity == EvidenceValidity.VALID
        assert ev.provenance["capability"] == STREAM_CAPABILITY


def test_confluence_shadow_ignores_liquidation_evidences():
    """O Confluence Shadow descarta evidências de liquidação por não estarem na whitelist."""
    summary = LiquidationWindowSummary(
        window_start_ms=1720000000000,
        window_end_ms=1720000060000,
        event_count=5,
        long_liquidated_qty=2.0,
        short_liquidated_qty=0.0,
        long_liquidated_notional_usd=120_000.0,
        short_liquidated_notional_usd=0.0,
        total_liquidated_notional_usd=120_000.0,
        validity=LiquidationValidity.VALID,
    )
    evidences = liquidation_summary_to_evidence(summary, symbol="BTCUSDT")

    reconciler = reconcile
    result = reconciler(
        evidences=evidences,
        symbol="BTCUSDT",
        observation_open_ms=1720000000000,
        observation_close_ms=1720000060000,
        causal_anchor_ms=1720000060000,
    )

    # Nenhuma evidência de liquidação é aceita na whitelist direcional
    # Como não há outras evidências primárias direcionais, o status deve ser INSUFFICIENT_DATA
    assert result.status == ReconcilerStatus.INSUFFICIENT_DATA
    assert len(result.valid_directional_evidence) == 0


# ── 9. SERIALIZAÇÃO DETERMINÍSTICA RFC 8259 ───────────────────────────────────

def test_rfc8259_serialization_json_compliance():
    """Valida serialização JSON de eventos e resumos sem floats anômalos."""
    raw = {
        "e": "forceOrder",
        "E": 1720000000000,
        "o": {
            "s": "BTCUSDT",
            "S": "SELL",
            "z": "1.0",
            "ap": "60000.0",
            "T": 1720000000000,
        }
    }
    event = parse_force_order_payload(raw)
    ev_dict = event.to_dict()

    dumped = json.dumps(ev_dict)
    loaded = json.loads(dumped)
    assert loaded["symbol"] == "BTCUSDT"
    assert loaded["liquidated_position_side"] == "LONG"

    agg = LiquidationWindowAggregator(symbol="BTCUSDT")
    agg.add_event(event)
    summary = agg.summarize_window(1720000000000, 1720000060000, stream_healthy=True)

    sum_dict = summary.to_dict()
    dumped_sum = json.dumps(sum_dict)
    loaded_sum = json.loads(dumped_sum)
    assert loaded_sum["event_count"] == 1
    assert loaded_sum["long_liquidated_notional_usd"] == 60_000.0


# ── 10. BENCHMARK PARSER & AGGREGATOR ─────────────────────────────────────────

def test_benchmark_parser_and_aggregator():
    """Mede a latência e throughput do parser e do agregador."""
    payloads = [
        {
            "e": "forceOrder",
            "E": 1720000000000 + i * 100,
            "o": {
                "s": "BTCUSDT",
                "S": "SELL" if i % 2 == 0 else "BUY",
                "q": "1.0",
                "z": "1.0",
                "ap": "60000.0",
                "T": 1720000000000 + i * 100,
            }
        }
        for i in range(1000)
    ]

    t0 = time.perf_counter()
    parsed_events = [parse_force_order_payload(p) for p in payloads]
    t_parse = time.perf_counter() - t0

    agg = LiquidationWindowAggregator(symbol="BTCUSDT")
    t1 = time.perf_counter()
    for ev in parsed_events:
        agg.add_event(ev)
    summary = agg.summarize_window(1720000000000, 1720000100000, stream_healthy=True)
    t_agg = time.perf_counter() - t1

    mean_parse_us = (t_parse / len(payloads)) * 1_000_000
    mean_agg_us = (t_agg / len(payloads)) * 1_000_000

    print(f"\n[BENCHMARK] Parse 1000 events: {t_parse*1000:.2f} ms ({mean_parse_us:.2f} µs/op)")
    print(f"[BENCHMARK] Aggregate 1000 events: {t_agg*1000:.2f} ms ({mean_agg_us:.2f} µs/op)")

    assert summary.event_count == 1000
    # Desempenho sub-milissegundo por operação no hot-path
    assert mean_parse_us < 100.0
    assert mean_agg_us < 100.0
