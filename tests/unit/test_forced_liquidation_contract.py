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
    BinanceLiquidationListener,
    ConnectionCoverageStatus,
    ConnectionIntervalTracker,
    ForcedLiquidationEvent,
    LargerObservedSide,
    LiquidatedPositionSide,
    LiquidationMetrics,
    LiquidationValidity,
    LiquidationWindowAggregator,
    LiquidationWindowSummary,
    NotionalQuality,
    OrderSide,
    StreamConnectionStatus,
    get_liquidation_window_context,
    is_agg_trade_message,
    is_force_order_message,
    liquidation_summary_to_evidence,
    parse_force_order_payload,
)
from market_orchestrator.market_orchestrator import parse_trade_message
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


# ── 12. P2-D2: ROUTING CRUZADO (aggTrade vs forceOrder) ───────────────────────

def test_cross_routing_agg_trade_vs_force_order():
    """
    Testes de routing cruzado obrigatórios:
    - forceOrder payload identificado por event type e/ou stream name.
    - NUNCA passar forceOrder pelo parser de aggTrade.
    - NUNCA passar aggTrade pelo parser de liquidation.
    """
    force_payload = {
        "e": "forceOrder",
        "E": 1720000000000,
        "o": {
            "s": "BTCUSDT",
            "S": "SELL",
            "o": "LIMIT",
            "f": "IOC",
            "q": "0.500",
            "p": "62000.0",
            "ap": "61950.0",
            "X": "FILLED",
            "l": "0.500",
            "z": "0.500",
            "T": 1720000000000,
        }
    }
    agg_payload = {
        "e": "aggTrade",
        "E": 1720000000100,
        "s": "BTCUSDT",
        "a": 1234567,
        "p": "62000.0",
        "q": "0.100",
        "f": 100,
        "l": 101,
        "T": 1720000000100,
        "m": True,
    }

    # 1. Guards de tipo
    assert is_force_order_message(force_payload) is True
    assert is_force_order_message(agg_payload) is False

    assert is_agg_trade_message(agg_payload) is True
    assert is_agg_trade_message(force_payload) is False

    # 2. forceOrder no parser de trade do orchestrator:
    # Como p, q, T estão dentro de 'o' e não na raiz, parse_trade_message rejeita (retorna None)
    trade_from_force = parse_trade_message(force_payload)
    assert trade_from_force is None

    # 3. aggTrade no listener de liquidação: descartado imediatamente
    listener = BinanceLiquidationListener(symbol="BTCUSDT")
    ev = listener.handle_raw_message(agg_payload)
    assert ev is None
    assert len(listener.aggregator._events) == 0

    # 4. forceOrder no listener de liquidação: aceito e parseado
    ev_force = listener.handle_raw_message(force_payload)
    assert ev_force is not None
    assert ev_force.validity == LiquidationValidity.VALID
    assert ev_force.liquidated_position_side == "LONG"
    assert len(listener.aggregator._events) == 1


# ── 13. P2-D2: CONNECTION COVERAGE (FULL, PARTIAL, NONE) ──────────────────────

def test_connection_coverage_full_partial_none():
    """
    Garante cobertura temporal e semântica de zero vs missing:
    - FULL + zero eventos => VALID ZERO_OBSERVED (0.0).
    - PARTIAL + zero eventos => PARTIAL (não assumir zero completo, notionals=None).
    - PARTIAL + eventos => PARTIAL (soma eventos parciais disponíveis).
    - NONE => UNKNOWN (notionals=None).
    """
    agg = LiquidationWindowAggregator(symbol="BTCUSDT")

    # 1. FULL coverage + 0 eventos
    sum_full_zero = agg.summarize_window(
        1000, 2000,
        stream_healthy=True,
        connection_status=StreamConnectionStatus.CONNECTED,
        connection_coverage_status=ConnectionCoverageStatus.FULL,
    )
    assert sum_full_zero.validity == LiquidationValidity.VALID
    assert sum_full_zero.reason == "ZERO_OBSERVED_HEALTHY_STREAM"
    assert sum_full_zero.event_count == 0
    assert sum_full_zero.total_liquidated_notional_usd == 0.0
    assert sum_full_zero.connection_coverage_status == "FULL"

    # 2. PARTIAL coverage + 0 eventos: NÃO declarar zero completo
    sum_part_zero = agg.summarize_window(
        1000, 2000,
        stream_healthy=True,
        connection_status=StreamConnectionStatus.CONNECTED,
        connection_coverage_status=ConnectionCoverageStatus.PARTIAL,
    )
    assert sum_part_zero.validity == LiquidationValidity.PARTIAL
    assert sum_part_zero.reason == "PARTIAL_CONNECTION_COVERAGE_ZERO_OBSERVED"
    assert sum_part_zero.event_count == 0
    assert sum_part_zero.total_liquidated_notional_usd is None  # Não afirma 0.0
    assert sum_part_zero.connection_coverage_status == "PARTIAL"

    # 3. PARTIAL coverage com evento presente
    ev = ForcedLiquidationEvent(
        symbol="BTCUSDT",
        event_time_ms=1500,
        trade_time_ms=1500,
        order_side="SELL",
        liquidated_position_side="LONG",
        original_qty=1.0,
        filled_qty=1.0,
        average_price=60000.0,
        limit_price=60000.0,
        observed_notional_usd=60000.0,
        estimated_notional_usd=None,
        notional_quality=NotionalQuality.OBSERVED_EXECUTION.value,
        validity=LiquidationValidity.VALID,
        event_id="EV_PARTIAL_1",
    )
    assert agg.add_event(ev) is True
    sum_part_ev = agg.summarize_window(
        1000, 2000,
        stream_healthy=True,
        connection_status=StreamConnectionStatus.CONNECTED,
        connection_coverage_status=ConnectionCoverageStatus.PARTIAL,
    )
    assert sum_part_ev.validity == LiquidationValidity.PARTIAL
    assert "PARTIAL_CONNECTION_COVERAGE" in sum_part_ev.reason
    assert sum_part_ev.event_count == 1
    assert sum_part_ev.total_liquidated_notional_usd == 60000.0

    # 4. NONE coverage (janela totalmente coberta por desconexão)
    sum_none = agg.summarize_window(
        1000, 2000,
        stream_healthy=False,
        connection_status=StreamConnectionStatus.DISCONNECTED,
        connection_coverage_status=ConnectionCoverageStatus.NONE,
    )
    assert sum_none.validity == LiquidationValidity.UNKNOWN
    assert sum_none.event_count is None
    assert sum_none.total_liquidated_notional_usd is None
    assert sum_none.connection_coverage_status == "NONE"


def test_connection_interval_tracker_timeline():
    """Valida o cálculo temporal determinístico do ConnectionIntervalTracker."""
    tracker = ConnectionIntervalTracker(initial_status=StreamConnectionStatus.DISCONNECTED)

    # Inicia desconectado em t=0
    tracker.record_status(StreamConnectionStatus.DISCONNECTED, timestamp_ms=0)
    # Conecta em t=1000
    tracker.record_status(StreamConnectionStatus.CONNECTED, timestamp_ms=1000)
    # Permanece conectado até t=5000
    tracker.record_status(StreamConnectionStatus.DISCONNECTED, timestamp_ms=5000)

    # Janela [1500, 4500): 100% coberta por CONNECTED -> FULL
    assert tracker.evaluate_coverage(1500, 4500) == ConnectionCoverageStatus.FULL

    # Janela [500, 2000): Conectou no meio -> PARTIAL
    assert tracker.evaluate_coverage(500, 2000) == ConnectionCoverageStatus.PARTIAL

    # Janela [4000, 6000): Desconectou no meio -> PARTIAL
    assert tracker.evaluate_coverage(4000, 6000) == ConnectionCoverageStatus.PARTIAL

    # Janela [6000, 8000): Totalmente desconectado -> NONE
    assert tracker.evaluate_coverage(6000, 8000) == ConnectionCoverageStatus.NONE


# ── 14. P2-D2: PRUNING & BOUNDED MEMORY ───────────────────────────────────────

def test_pruning_and_bounded_memory_growth():
    """
    Valida política de pruning e retenção causal bounded:
    - prune_events_older_than remove eventos passados após materialização da janela.
    - _seen_event_ids tem teto estrito (max_seen_events) e não cresce infinitamente.
    """
    agg = LiquidationWindowAggregator(symbol="BTCUSDT", max_seen_events=10)

    # Insere 15 eventos com timestamps incrementais
    for i in range(15):
        ev = ForcedLiquidationEvent(
            symbol="BTCUSDT",
            event_time_ms=1000 + i * 100,
            trade_time_ms=1000 + i * 100,
            order_side="SELL",
            liquidated_position_side="LONG",
            original_qty=0.1,
            filled_qty=0.1,
            average_price=60000.0,
            limit_price=60000.0,
            observed_notional_usd=6000.0,
            estimated_notional_usd=None,
            notional_quality=NotionalQuality.OBSERVED_EXECUTION.value,
            validity=LiquidationValidity.VALID,
            event_id=f"EV_{i}",
        )
        agg.add_event(ev)

    # Capacidade de deduplicação mantida no teto max_seen_events
    assert len(agg._seen_event_ids) == 10
    # Total de eventos em memória antes do pruning
    assert len(agg._events) == 15

    # Poda eventos anteriores a 2000 ms (ou seja, trade_time < 2000)
    pruned_count = agg.prune_events_older_than(2000)
    assert pruned_count == 10
    assert len(agg._events) == 5
    for e in agg._events:
        assert e.trade_time_ms >= 2000

    # Deduplicação continua protegendo contra IDs recentes
    last_ev = agg._events[-1]
    assert agg.add_event(last_ev) is False


# ── 15. P2-D2: RECONNECT DUPLICATE SNAPSHOT PROTECTION ────────────────────────

def test_reconnect_duplicate_snapshot_protection():
    """
    Garante que snapshot repetido recebido após reconnect não seja contado duas vezes.
    """
    listener = BinanceLiquidationListener(symbol="BTCUSDT")

    payload = {
        "e": "forceOrder",
        "E": 1720000000000,
        "o": {
            "s": "BTCUSDT",
            "S": "SELL",
            "o": "LIMIT",
            "f": "IOC",
            "q": "2.0",
            "p": "60000.0",
            "ap": "60000.0",
            "X": "FILLED",
            "l": "2.0",
            "z": "2.0",
            "T": 1720000000000,
        }
    }

    # 1. Recebe pela primeira vez
    ev1 = listener.handle_raw_message(payload)
    assert ev1 is not None
    assert len(listener.aggregator._events) == 1

    # 2. Simula queda e reconexão do listener
    listener.tracker.record_status(StreamConnectionStatus.DISCONNECTED)
    listener.tracker.record_status(StreamConnectionStatus.RECONNECTING)
    listener.tracker.record_status(StreamConnectionStatus.CONNECTED)

    # 3. Binance re-envia o mesmo snapshot de forceOrder
    ev2 = listener.handle_raw_message(payload)
    assert ev2 is not None
    # Não duplica na lista de eventos agregados
    assert len(listener.aggregator._events) == 1

    summary = listener.aggregator.summarize_window(
        1720000000000, 1720000010000, stream_healthy=True
    )
    assert summary.event_count == 1
    assert summary.total_liquidated_notional_usd == 120_000.0


# ── 16. P2-D2: METRICS SINGLETON & CONTEXT GETTER ─────────────────────────────

def test_metrics_singleton_and_context_getter():
    """
    Valida singleton de métricas (sem erro de duplicação) e getter de contexto interno.
    """
    m1 = LiquidationMetrics.get_instance()
    m2 = LiquidationMetrics.get_instance()
    assert m1 is m2

    # Métricas funcionam sem exceptions
    m1.set_connected(True, "BTCUSDT")
    m1.record_received("BTCUSDT")
    m1.record_validity(LiquidationValidity.VALID, "BTCUSDT")
    m1.record_deduplicated("BTCUSDT")
    m1.record_reconnect("BTCUSDT")

    # Context getter não altera trading/LLM
    agg = LiquidationWindowAggregator(symbol="BTCUSDT")
    summary = agg.summarize_window(1000, 2000, stream_healthy=True)
    ctx = get_liquidation_window_context(summary)

    assert ctx["is_telemetry_only"] is True
    assert ctx["counts_as_vote"] is False
    assert ctx["direction"] == "UNKNOWN"
    assert "liquidation_summary" in ctx


# ── 17. P2-D2: GRACEFUL SHUTDOWN ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_graceful_shutdown_listener():
    """Valida que o listener encerra de forma limpa e com status DISCONNECTED."""
    listener = BinanceLiquidationListener(symbol="BTCUSDT")
    listener.tracker.record_status(StreamConnectionStatus.CONNECTED)
    assert listener.is_connected is True

    await listener.stop()
    assert listener.is_connected is False
    assert listener.tracker.current_status == StreamConnectionStatus.DISCONNECTED


# ── 18. P2-D2: BENCHMARK DISPATCH + PARSE + AGGREGATE (p50, p95, p99) ─────────

def test_benchmark_dispatch_parse_aggregate_percentiles():
    """
    Mede a latência end-to-end de dispatch + parse + dedup + aggregate
    em lote de 2.000 mensagens e reporta p50, p95 e p99 em microssegundos.
    """
    import numpy as np

    listener = BinanceLiquidationListener(symbol="BTCUSDT")
    latencies_us = []

    for i in range(2000):
        raw_msg = json.dumps({
            "e": "forceOrder",
            "E": 1720000000000 + i * 50,
            "o": {
                "s": "BTCUSDT",
                "S": "SELL" if i % 2 == 0 else "BUY",
                "o": "LIMIT",
                "f": "IOC",
                "q": "0.250",
                "p": "61000.0",
                "ap": "61000.0",
                "X": "FILLED",
                "l": "0.250",
                "z": "0.250",
                "T": 1720000000000 + i * 50,
            }
        })
        t0 = time.perf_counter_ns()
        ev = listener.handle_raw_message(raw_msg)
        t1 = time.perf_counter_ns()
        latencies_us.append((t1 - t0) / 1000.0)
        assert ev is not None

    arr = np.array(latencies_us)
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))

    print(f"\n[BENCHMARK END-TO-END] 2000 events dispatch+parse+aggregate:")
    print(f"  p50: {p50:.2f} µs")
    print(f"  p95: {p95:.2f} µs")
    print(f"  p99: {p99:.2f} µs")

    assert p50 < 100.0
    assert p99 < 500.0
    assert len(listener.aggregator._events) == 2000


