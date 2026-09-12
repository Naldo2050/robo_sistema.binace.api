"""P00 regression: compressor must preserve compact schema (builder→compressor→guardrail).

Escopo estrito P00. Não toca P01/fórmulas/thresholds/BUY-SELL.
"""
import copy
import json
import math

import pytest

from market_orchestrator.ai.payload_builder_compact import build_compact_payload
from market_orchestrator.ai.payload_compressor import compress_payload
from market_orchestrator.ai.llm_payload_guardrail import ensure_safe_llm_payload


def _compact_sample():
    """Payload compacto sintético cobrindo seções exigidas (sem raw_event)."""
    return {
        "symbol": "BTCUSDT",
        "epoch_ms": 1789158900000,
        "trigger": "AT",
        "price": {"c": 77338.8, "o": 77325.0, "h": 77339.0, "l": 77325.0, "vw": 77332.0},
        "regime": {"cs": "MIX", "cf": 0.5, "mode": "MR"},
        "flow": {
            "d1": "+341K", "delta": 4.404, "vol": 9.032, "buy_pct": 74,
            "d5": "-3.3M", "d15": "-5.4M",  # numeradores P01 (incorretos por desenho) preservados
            "cvd_4h": -70.0, "imb": 0.49, "bsr": 2.9,
        },
        "ob": {"b": "3.0M", "a": "320K", "imb": 0.81, "bias": "BUY", "t5": 0.69},
        "tf": {"15m": {"t": "DN", "rsi": 42}},
        "sr": {"r1": [77400, 52]},
        "qual": {"lat": "ACCE", "liq": "NORMAL"},
        "vwap": {"svw": 77330.0, "dist": 0.0001},
        "onchain": {"st": "fresh", "mempool_sz": 30073.0, "fees_fast": 0},
        "mkt": "fut_perp",
    }


def _size(payload):
    return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def test_p00_compact_sections_preserved_through_compressor_and_guardrail():
    compact = _compact_sample()
    before = copy.deepcopy(compact)
    compressed = compress_payload(compact, max_bytes=6144)
    # Não muta original
    assert compact == before
    for key in ("price", "flow", "ob", "regime", "tf", "sr", "qual", "vwap", "onchain"):
        assert key in compressed, f"P00 FAIL: {key} removido pelo compressor"
    assert compressed["price"]["c"] == 77338.8
    assert compressed["flow"]["d15"] == "-5.4M"
    assert compressed["ob"]["imb"] == 0.81
    # Proibidas continuam proibidas
    for forbidden in ("raw_event", "contextual_snapshot", "historical_vp", "observability", "enriched_snapshot"):
        assert forbidden not in compressed
    assert _size(compressed) <= 6144

    guarded = ensure_safe_llm_payload(compressed)
    assert guarded is not None, "P00 FAIL: guardrail abortou payload compacto válido"
    for key in ("price", "flow", "ob", "regime", "tf"):
        assert key in guarded, f"P00 FAIL: {key} removido pelo guardrail"


def test_p00_builder_to_guardrail_interface_preserves_market_data():
    event = {
        "symbol": "BTCUSDT",
        "epoch_ms": 1789158900000,
        "tipo_evento": "ANALYSIS_TRIGGER",
        "descricao": "Evento automático para análise da IA",
        "preco_fechamento": 77338.8,
        "delta": 4.404,
        "volume_total": 9.032,
        "volume_compra": 6.718,
        "volume_venda": 2.314,
        "fluxo_continuo": {
            "cvd": -70.002,
            "sector_flow": {"whale": {"delta": -46.618}, "retail": {"delta": -10.761}},
            "order_flow": {
                "net_flow_1m": 340574.3071,
                "net_flow_5m": -3300000.0,
                "net_flow_15m": -5400000.0,  # numerador P01 (intencionalmente incorreto)
                "flow_imbalance": 0.4876,
                "aggressive_buy_pct": 74.0,
                "buy_sell_ratio": {"buy_sell_ratio": 2.9, "ratios": {"current": 2.9, "imbalance_1m": 0.4876, "imbalance_5m": -4.7154, "imbalance_15m": -7.7549}},
            },
        },
        "orderbook_data": {"bid_depth_usd": 3013161.43, "ask_depth_usd": 319913.43, "flow_imbalance": 0.808, "imbalance": 0.808},
        "ml_features": {"microstructure": {"trade_intensity_v2": 3.5, "tick_rule_sum": 79}},
        "multi_tf": {"15m": {"tendencia": "Baixa", "rsi_short": 42, "macd": 1.0, "macd_signal": 0.5, "adx": 25, "atr": 10.0}},
        "institutional_analytics": {"quality": {"latency": {"latency_ms": 100, "latency_category": "ACCEPTABLE"}, "calendar": {"expected_liquidity": "NORMAL"}}},
    }
    built = build_compact_payload(event)
    assert "price" in built and "flow" in built and "ob" in built
    # P01: numerador incorreto deve sobreviver (NÃO corrigir aqui)
    assert built["flow"].get("d15") is not None
    compressed = compress_payload(built, max_bytes=6144)
    assert "price" in compressed and "flow" in compressed and "ob" in compressed
    assert compressed["flow"].get("d15") == built["flow"].get("d15"), "P00 não pode mascarar P01"
    guarded = ensure_safe_llm_payload(compressed)
    assert guarded is not None
    assert "price" in guarded and "flow" in guarded and "ob" in guarded
    assert "raw_event" not in guarded and "contextual_snapshot" not in guarded and "historical_vp" not in guarded


def test_p00_nan_inf_never_leak_as_invalid_json():
    compact = _compact_sample()
    compact["flow"]["imb"] = float("nan")
    compact["price"]["c"] = float("inf")
    guarded = ensure_safe_llm_payload(compress_payload(compact, max_bytes=6144))
    assert guarded is not None
    blob = json.dumps(guarded, ensure_ascii=False, allow_nan=False)
    assert "NaN" not in blob and "Infinity" not in blob
    # Política canônica: non-finite → None (não 0); sem crash
    assert guarded["flow"]["imb"] is None
