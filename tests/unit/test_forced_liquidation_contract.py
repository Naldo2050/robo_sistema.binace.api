# tests/unit/test_forced_liquidation_contract.py
# -*- coding: utf-8 -*-
"""
Testes unitários rigorosos para P2-D1.1 — Forced Liquidation Semantic Hardening.

Cobre com exatidão:
1. ap válido -> observed notional (notional_quality=OBSERVED_EXECUTION, validity=VALID)
2. ap ausente/zero + p válido -> estimated notional, observed_notional_usd=None (notional_quality=ESTIMATED_LIMIT_PRICE, validity=PARTIAL)
3. Aggregator: observed total exclui estimates (somam estritamente observed_notional_usd)
4. SELL forceOrder -> liquidated LONG
5. BUY forceOrder -> liquidated SHORT
6. filled qty vs original qty
7. partial event (notional não computável)
8. invalid side / non-finite / malformed / negative price or qty
9. deterministic dedup (event_id estável) & duplicate snapshot not double counted
10. documented collision risk (sem UUID aleatório)
11. long aggregation & short aggregation
12. healthy silence (stream_healthy=True / CONNECTED + 0 events) -> zero observed (0.0)
13. connection disconnected (stream_healthy=False / DISCONNECTED) -> UNKNOWN (None)
14. exact window boundaries ([start, end): start entra, end pertence à próxima janela)
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
    NotionalQuality,
    OrderSide,
    StreamConnectionStatus,
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
    assert event.estimated_notional_usd is None
    assert event.notional_quality == NotionalQuality.OBSERVED_EXECUTION.value
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
    assert event.estimated_notional_usd is None
    assert event.notional_quality == NotionalQuality.OBSERVED_EXECUTION.value


# ── 2. PRICE FALLBACK: OBSERVED (ap) VS ESTIMATED (p) ─────────────────────────

def test_ap_valid_gives_observed_execution_notional():
    """ap válido gera observed_notional_usd com qualidade OBSERVED_EXECUTION."""
    raw = {
        "e": "forceOrder",
        "E": 1720000015000,
        "o": {
            "s": "BTCUSDT",
            "S": "SELL",
            "q": "2.000",
            "z": "2.000",
            "ap": "60500.00",
            "p": "60400.00",
            "X": "FILLED",
            "T": 1720000015000,
        }
    }
    event = parse_force_order_payload(raw)

    assert event.validity == LiquidationValidity.VALID
    assert event.average_price == 60500.00
    assert event.limit_price == 60400.00
    assert event.observed_notional_usd == 121000.00
    assert event.estimated_notional_usd is None
    assert event.notional_quality == NotionalQuality.OBSERVED_EXECUTION.value


def test_ap_missing_or_zero_gives_estimated_limit_price_not_observed():
    """ap ausente ou 0 com p válido gera estimated_notional_usd; observed_notional_usd fica None."""
    raw = {
        "e": "forceOrder",
        "E": 1720000020000,
        "o": {
            "s": "BTCUSDT",
            "S": "BUY",
            "q": "1.000",
            "z": "1.000",
            "ap": "0",          # average price ausente/zero
            "p": "61500.00",    # fallback preço limite
            "X": "FILLED",
            "T": 1720000020000,
        }
    }
    event = parse_force_order_payload(raw)

    # Não mente que foi observado: validity=PARTIAL
    assert event.validity == LiquidationValidity.PARTIAL
    assert event.average_price is None
    assert event.limit_price == 61500.00
    # observed_notional_usd NUNCA é preenchido como se fosse observado!
    assert event.observed_notional_usd is None
    # estimated_notional_usd recebe o valor
    assert event.estimated_notional_usd == 61500.00
    assert event.notional_quality == NotionalQuality.ESTIMATED_LIMIT_PRICE.value
    assert "LIMIT_PRICE_FALLBACK" in event.reason


def test_aggregator_observed_totals_exclude_estimates():
    """O agregador soma estritamente observed_notional_usd nos totais observados."""
    agg = LiquidationWindowAggregator(symbol="BTCUSDT")
    t0 = 1720000000000

    # Evento 1: 1.0 BTC observado @ 60,000 (ap válido) -> observed = 60,000
    agg.add_event({
        "o": {"s": "BTCUSDT", "S": "SELL", "z": "1.0", "ap": "60000.0", "p": "60000.0", "X": "FILLED", "T": t0 + 10_000}
    })
    # Evento 2: 2.0 BTC estimado @ 50,000 (ap zero, fallback p) -> estimated = 100,000
    agg.add_event({
        "o": {"s": "BTCUSDT", "S": "SELL", "z": "2.0", "ap": "0", "p": "50000.0", "X": "FILLED", "T": t0 + 20_000}
    })

    summary = agg.summarize_window(t0, t0 + 60_000, stream_healthy=True)

    # Total observado contém SOMENTE os 60.000 observados
    assert summary.long_liquidated_notional_usd == 60_000.0
    assert summary.total_liquidated_notional_usd == 60_000.0

    # Total estimado contém separadamente os 100.000 estimados
    assert summary.long_estimated_notional_usd == 100_000.0
    assert summary.total_estimated_notional_usd == 100_000.0

    # Validade reflete a presença de evento parcial na janela
    assert summary.validity == LiquidationValidity.PARTIAL
    assert "WINDOW_CONTAINS_PARTIAL_EVENTS" in summary.reason


# ── 3. FILLED QTY VS ORIGINAL QTY ─────────────────────────────────────────────

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


# ── 4. CASOS DE ERRO E VALIDADE (PARTIAL / INVALID / NON-FINITE) ──────────────

def test_partial_event_when_notional_cannot_be_calculated():
    """Evento real onde faltam preço ou quantidade resulta em PARTIAL."""
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
    assert event.estimated_notional_usd is None
    assert event.notional_quality == NotionalQuality.NONE.value
    assert "NOTIONAL_CANNOT_BE_CALCULATED" in event.reason


def test_invalid_side_or_corrupted_data():
    """Lado desconhecido, valores não finitos ou negativos resultam em INVALID."""
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
    assert "NEGATIVE_PRICE_OR_QTY" in ev3.reason


# ── 5. DEDUPLICAÇÃO DETERMINÍSTICA E RISCO RESIDUAL DOCUMENTADO ───────────────

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

    # Segunda inserção (duplicata exata de rede / reconnect / replay): rejeitada
    ok2 = agg.add_event(raw_event)
    assert ok2 is False

    summary = agg.summarize_window(1720000000000, 1720000060000, stream_healthy=True)
    assert summary.event_count == 1
    assert summary.long_liquidated_notional_usd == 120_000.0
    assert summary.total_liquidated_notional_usd == 120_000.0


def test_documented_collision_risk_deterministic():
    """Garante que o event_id é estritamente determinístico (sem random/UUID)."""
    raw = {
        "e": "forceOrder",
        "E": 1720000030000,
        "o": {
            "s": "BTCUSDT",
            "S": "SELL",
            "q": "1.000",
            "z": "1.000",
            "ap": "60000.00",
            "p": "60000.00",
            "X": "FILLED",
            "T": 1720000030000,
        }
    }
    ev1 = parse_force_order_payload(raw)
    ev2 = parse_force_order_payload(raw)

    # Identificador idêntico para idempotência em replay
    assert ev1.event_id == ev2.event_id
    assert ev1.event_id.startswith("BTCUSDT_1720000030000_SELL_")


# ── 6. LIMITES DE JANELA EXATOS [window_start_ms, window_end_ms) ──────────────

def test_exact_window_boundaries_inclusion_and_exclusion():
    """
    Contrato estrito:
    - trade_time == window_start_ms => ENTRA na janela atual.
    - trade_time == window_end_ms   => EXCLUÍDO da janela atual (pertence à próxima).
    """
    agg = LiquidationWindowAggregator(symbol="BTCUSDT")
    t_start = 1720000000000
    t_end = t_start + 60_000

    # Evento no boundary inicial exato (t_start): DEVE ENTRAR
    agg.add_event({
        "o": {"s": "BTCUSDT", "S": "SELL", "z": "1.0", "ap": "60000.0", "X": "FILLED", "T": t_start}
    })
    # Evento dentro da janela: DEVE ENTRAR
    agg.add_event({
        "o": {"s": "BTCUSDT", "S": "SELL", "z": "2.0", "ap": "60000.0", "X": "FILLED", "T": t_start + 30_000}
    })
    # Evento no boundary final exato (t_end): NÃO DEVE ENTRAR na janela [t_start, t_end)
    agg.add_event({
        "o": {"s": "BTCUSDT", "S": "SELL", "z": "4.0", "ap": "60000.0", "X": "FILLED", "T": t_end}
    })

    summary_current = agg.summarize_window(t_start, t_end, stream_healthy=True)
    # Contém apenas o evento de t_start e o intermediário (1.0 + 2.0 = 3.0 BTC)
    assert summary_current.event_count == 2
    assert summary_current.long_liquidated_qty == 3.0
    assert summary_current.long_liquidated_notional_usd == 180_000.0

    # Na próxima janela [t_end, t_end + 60_000), o evento de t_end entra perfeitamente
    summary_next = agg.summarize_window(t_end, t_end + 60_000, stream_healthy=True)
    assert summary_next.event_count == 1
    assert summary_next.long_liquidated_qty == 4.0
    assert summary_next.long_liquidated_notional_usd == 240_000.0


# ── 7. STREAM HEALTH CONTRACT: CONNECTED VS DISCONNECTED ──────────────────────

def test_stream_health_contract_connected_silence_vs_disconnected():
    """
    forceOrder é event-sparse.
    Silêncio em stream CONNECTED => ZERO_OBSERVED (0.0).
    Stream DISCONNECTED => UNKNOWN (None).
    """
    agg = LiquidationWindowAggregator(symbol="BTCUSDT")
    t0 = 1720000000000
    t1 = t0 + 60_000

    # Cenário A: Camada de transporte está CONNECTED e nenhum evento ocorreu (mercado quieto)
    healthy_summary = agg.summarize_window(
        t0, t1, connection_status=StreamConnectionStatus.CONNECTED
    )
    assert healthy_summary.validity == LiquidationValidity.VALID
    assert healthy_summary.connection_status == "CONNECTED"
    assert healthy_summary.event_count == 0
    assert healthy_summary.long_liquidated_notional_usd == 0.0
    assert healthy_summary.short_liquidated_notional_usd == 0.0
    assert healthy_summary.total_liquidated_notional_usd == 0.0
    assert healthy_summary.reason == "ZERO_OBSERVED_HEALTHY_STREAM"

    # Cenário B: Camada de transporte está DISCONNECTED
    disconnected_summary = agg.summarize_window(
        t0, t1, connection_status=StreamConnectionStatus.DISCONNECTED
    )
    assert disconnected_summary.validity == LiquidationValidity.UNKNOWN
    assert disconnected_summary.connection_status == "DISCONNECTED"
    assert disconnected_summary.event_count is None
    assert disconnected_summary.long_liquidated_notional_usd is None
    assert disconnected_summary.short_liquidated_notional_usd is None
    assert disconnected_summary.total_liquidated_notional_usd is None
    assert disconnected_summary.reason == "STREAM_DISCONNECTED"

    # Cenário C: Camada de transporte está RECONNECTING
    reconnecting_summary = agg.summarize_window(
        t0, t1, connection_status=StreamConnectionStatus.RECONNECTING
    )
    assert reconnecting_summary.validity == LiquidationValidity.UNKNOWN
    assert reconnecting_summary.reason == "STREAM_RECONNECTING"


# ── 8. AUSÊNCIA DE CAMPOS FUTUROS E STRINGS DIRECIONAIS ───────────────────────

def test_no_future_heatmap_and_no_directional_strings():
    """Verifica que o contrato não inventa heatmaps futuros nem strings de sinal."""
    agg = LiquidationWindowAggregator(symbol="BTCUSDT")
    summary = agg.summarize_window(1720000000000, 1720000060000, stream_healthy=True)
    d = summary.to_dict()

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

    assert d["larger_observed_side"] in ("LONG", "SHORT", "EQUAL", "NONE")
    assert "bullish" not in str(d).lower()
    assert "bearish" not in str(d).lower()
    assert "reversal" not in str(d).lower()


# ── 9. EVIDENCE ADAPTERS V1 E DESCARTE DE VOTO ────────────────────────────────

def test_evidence_contract_non_voting_and_direction_unknown():
    """Evidências de liquidação usam exclusivamente totais observados, NON_VOTING e direction=UNKNOWN."""
    summary = LiquidationWindowSummary(
        window_start_ms=1720000000000,
        window_end_ms=1720000060000,
        event_count=2,
        long_liquidated_qty=1.0,
        short_liquidated_qty=0.0,
        long_liquidated_notional_usd=60_000.0,
        short_liquidated_notional_usd=0.0,
        total_liquidated_notional_usd=60_000.0,
        long_estimated_notional_usd=25_000.0,
        total_estimated_notional_usd=25_000.0,
        latest_event_ms=1720000030000,
        validity=LiquidationValidity.VALID,
    )

    evidences = liquidation_summary_to_evidence(summary, symbol="BTCUSDT")
    assert len(evidences) == 4

    ev_long = next(e for e in evidences if e.metadata["field_id"] == "derivatives.liquidations.long_observed_notional")
    # Usa EXCLUSIVAMENTE o total observado (60.000), nunca somado com o estimado
    assert ev_long.value == 60_000.0
    assert ev_long.metadata["estimated_notional_usd"] == 25_000.0
    assert ev_long.family == EvidenceFamily.DERIVATIVES
    assert ev_long.evidence_type == EvidenceType.SLOW_CONTEXT
    assert ev_long.counts_as_vote is False
    assert ev_long.direction == EvidenceDirection.UNKNOWN
    assert ev_long.calibration == EvidenceCalibration.NOT_APPLICABLE


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

    result = reconcile(
        evidences=evidences,
        symbol="BTCUSDT",
        observation_open_ms=1720000000000,
        observation_close_ms=1720000060000,
        causal_anchor_ms=1720000060000,
    )

    assert result.status == ReconcilerStatus.INSUFFICIENT_DATA
    assert len(result.valid_directional_evidence) == 0


# ── 10. SERIALIZAÇÃO DETERMINÍSTICA RFC 8259 ──────────────────────────────────

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
    assert loaded["notional_quality"] == NotionalQuality.OBSERVED_EXECUTION.value

    agg = LiquidationWindowAggregator(symbol="BTCUSDT")
    agg.add_event(event)
    summary = agg.summarize_window(1720000000000, 1720000060000, stream_healthy=True)

    sum_dict = summary.to_dict()
    dumped_sum = json.dumps(sum_dict)
    loaded_sum = json.loads(dumped_sum)
    assert loaded_sum["event_count"] == 1
    assert loaded_sum["long_liquidated_notional_usd"] == 60_000.0
    assert loaded_sum["connection_status"] == "CONNECTED"


# ── 11. BENCHMARK PARSER & AGGREGATOR ─────────────────────────────────────────

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
    assert mean_parse_us < 100.0
    assert mean_agg_us < 100.0
