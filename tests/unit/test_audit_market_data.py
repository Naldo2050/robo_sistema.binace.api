# -*- coding: utf-8 -*-
"""
TESTES SINTÉTICOS DO AUDITOR OFFLINE DOS DADOS DO MERCADO (audit_market_data.py).

Garante que o auditor detecta com precisão:
- Evento perfeito → PASS
- Delta errado → BLOCKER
- Flow imbalance errado → BLOCKER
- Timestamp desalinhado → BLOCKER
- NaN em payload → BLOCKER
- Macro null → PASS (permitido)
- Stale marcado live → BLOCKER
- Suporte acima do preço → BLOCKER
- VP status error com 70% fabricado → BLOCKER
- Arredondamento pequeno → PASS
"""

import json
import pytest
from scripts.diagnostics.audit_market_data import (
    is_close,
    check_flow,
    check_orderbook,
    check_temporal,
    check_source_quality,
    check_value_profile,
    check_support_resistance,
    check_pivots,
    check_derivatives,
    check_macro,
    check_ai_quality,
    BLOCKER,
    HIGH,
    PASS,
    FAIL,
    SKIP
)


def get_perfect_event():
    """Retorna um payload sintético totalmente válido e consistente."""
    return {
        "tipo_evento": "ANALYSIS_TRIGGER",
        "symbol": "BTCUSDT",
        "epoch_ms": 1786545600000,
        "timestamp_utc": "2026-08-12T14:40:00.000Z",
        "janela_numero": 1,
        "event_id": "test_perfect_123",
        "preco_fechamento": 63500.0,
        "volume_total": 10.0,
        "volume_compra": 6.0,
        "volume_venda": 4.0,
        "delta": 2.0,
        "buy_notional_usdt": 381000.0,
        "sell_notional_usdt": 254000.0,
        "fluxo_continuo": {
            "order_flow": {
                "total_volume_btc": 10.0,
                "buy_volume_btc": 6.0,
                "sell_volume_btc": 4.0,
                "flow_imbalance": 0.2,  # (6-4)/(6+4) = 2/10 = 0.2
                "aggressive_buy_pct": 60.0,
                "aggressive_sell_pct": 40.0,
                "net_flow_1m": 127000.0,  # 381000 - 254000
                "computation_window_min": 1,
                "buy_sell_ratio": {
                    "current": 1.5,  # 6 / 4 = 1.5
                    "buy_volume": 6.0,
                    "sell_volume": 4.0
                }
            }
        },
        "orderbook_data": {
            "bid": 63499.0,
            "ask": 63501.0,
            "mid": 63500.0,
            "spread": 2.0,
            "spread_bps": 0.31496,
            "bid_depth_usd": 1000000.0,
            "ask_depth_usd": 500000.0,
            "imbalance": 0.333333,  # (1m - 0.5m) / 1.5m = 0.3333
            "volume_ratio": 2.0,
            "data_source": "binance_futures"
        },
        "order_book_depth": {
            "L1": {"bids": 100000.0, "asks": 50000.0, "flow_imbalance": 0.3333},
            "L5": {"bids": 300000.0, "asks": 150000.0, "flow_imbalance": 0.3333},
            "L10": {"bids": 500000.0, "asks": 250000.0, "flow_imbalance": 0.3333},
            "L25": {"bids": 1000000.0, "asks": 500000.0, "flow_imbalance": 0.3333}
        },
        "quality": {
            "latency": {
                "is_acceptable": 1,
                "latency_ms": 150
            }
        },
        "data_reliability": {
            "latency_acceptable": 1
        },
        "profile_analysis": {
            "va_volume_pct": {
                "status": "success",
                "value_area_volume_pct": 68.2,
                "volume_in_va": 6.82,
                "total_volume": 10.0
            }
        },
        "val": 63400.0,
        "vah": 63600.0,
        "poc_price": 63500.0,
        "immediate_support": [63400.0, 63300.0],
        "support_strength": [80.0, 70.0],
        "immediate_resistance": [63600.0, 63700.0],
        "resistance_strength": [85.0, 75.0],
        "pivots": {
            "daily": {
                "high": 64000.0,
                "low": 63000.0,
                "close": 63500.0,
                "pivot": 63500.0,
                "r1": 64000.0,  # 2*63500 - 63000 = 64000
                "s1": 63000.0,  # 2*63500 - 64000 = 63000
                "r2": 64500.0,  # 63500 + 1000 = 64500
                "s2": 62500.0   # 63500 - 1000 = 62500
            }
        },
        "derivatives": {
            "ETHUSDT": {
                "long_short_ratio": 2.0,
                "longs_usd": 2000000.0,
                "shorts_usd": 1000000.0,
                "open_interest_usd": 3000000.0,
                "funding_rate_percent": 0.01
            }
        },
        "market_environment": {
            "dxy_return": -0.05,
            "vix": 15.2,
            "macro_score": None  # Null é permitido
        }
    }


def test_perfect_event_passes():
    ev = get_perfect_event()
    results = []
    results.extend(check_flow(ev))
    results.extend(check_orderbook(ev))
    results.extend(check_temporal(ev))
    results.extend(check_source_quality(ev))
    results.extend(check_value_profile(ev))
    results.extend(check_support_resistance(ev))
    results.extend(check_pivots(ev))
    results.extend(check_derivatives(ev))
    results.extend(check_macro(ev))

    # Nenhum resultado deve ter status FAIL
    failures = [r for r in results if r["status"] == FAIL]
    assert len(failures) == 0, f"Evento perfeito falhou em: {failures}"


def test_wrong_delta_blocker():
    ev = get_perfect_event()
    ev["delta"] = 999.0  # Incompatível com compra=6 e venda=4 (esperado delta=2)

    results = check_flow(ev)
    fail_delta = [r for r in results if r["check"] == "flow.delta_conservation" and r["status"] == FAIL]
    assert len(fail_delta) == 1
    assert fail_delta[0]["severity"] == BLOCKER


def test_wrong_flow_imbalance_blocker():
    ev = get_perfect_event()
    ev["fluxo_continuo"]["order_flow"]["flow_imbalance"] = -0.99  # Esperado +0.2

    results = check_flow(ev)
    fail_imb = [r for r in results if r["check"] == "flow.imbalance_formula" and r["status"] == FAIL]
    assert len(fail_imb) == 1
    assert fail_imb[0]["severity"] == BLOCKER


def test_wrong_timestamp_blocker():
    ev = get_perfect_event()
    ev["epoch_ms"] = 1786545600000
    ev["timestamp_utc"] = "2026-08-12T20:00:00.000Z"  # Diferença de 5.33 horas!

    results = check_temporal(ev)
    fail_ts = [r for r in results if r["check"] == "temporal.utc_epoch_alignment" and r["status"] == FAIL]
    assert len(fail_ts) == 1
    assert fail_ts[0]["severity"] == BLOCKER


def test_nan_in_payload_blocker():
    ev = get_perfect_event()
    ev["derivatives"]["ETHUSDT"]["long_short_ratio"] = float("nan")

    results = check_derivatives(ev)
    fail_nan = [r for r in results if "no_nan_inf" in r["check"] and r["status"] == FAIL]
    assert len(fail_nan) == 1
    assert fail_nan[0]["severity"] == BLOCKER


def test_null_macro_allowed():
    ev = get_perfect_event()
    ev["market_environment"]["us10y"] = None
    ev["market_environment"]["gold"] = None

    results = check_macro(ev)
    failures = [r for r in results if r["status"] == FAIL]
    assert len(failures) == 0


def test_stale_marked_live_blocker():
    ev = get_perfect_event()
    ev["source"] = {"exchange": "stale_connection_rest"}
    ev["orderbook_quality"] = "live"

    results = check_source_quality(ev)
    fail_src = [r for r in results if r["check"] == "source.degraded_not_marked_live" and r["status"] == FAIL]
    assert len(fail_src) == 1
    assert fail_src[0]["severity"] == BLOCKER


def test_support_above_price_blocker():
    ev = get_perfect_event()
    ev["preco_fechamento"] = 63500.0
    ev["immediate_support"] = [64000.0]  # Suporte 64000 > Preço 63500!

    results = check_support_resistance(ev)
    fail_sup = [r for r in results if "immediate_support" in r["check"] and r["status"] == FAIL]
    assert len(fail_sup) == 1
    assert fail_sup[0]["severity"] == BLOCKER


def test_vp_error_70_pct_blocker():
    ev = get_perfect_event()
    ev["profile_analysis"]["va_volume_pct"]["status"] = "error"
    ev["profile_analysis"]["va_volume_pct"]["value_area_volume_pct"] = 70.0

    results = check_value_profile(ev)
    fail_vp = [r for r in results if r["check"] == "value_profile.fabricated_pct_on_error" and r["status"] == FAIL]
    assert len(fail_vp) == 1
    assert fail_vp[0]["severity"] == BLOCKER


def test_small_rounding_passes():
    ev = get_perfect_event()
    # Diferença minúscula de float de 63500.000001
    ev["orderbook_data"]["mid"] = 63500.000001

    results = check_orderbook(ev)
    failures = [r for r in results if r["status"] == FAIL]
    assert len(failures) == 0


def test_sector_flow_j1_not_comparable():
    """J1: sector_flow é apenas 1.1x maior que order_flow. Deve ser NOT_COMPARABLE e não FAIL/HIGH."""
    ev = get_perfect_event()
    ev["fluxo_continuo"] = {
        "order_flow": {
            "buy_volume_btc": 3.36775,
            "sell_volume_btc": 4.21339
        },
        "sector_flow": {
            "retail": {"buy": 3.7914, "sell": 3.71108, "delta": 0.08032},
            "mid": {"buy": 0.0, "sell": 0.51116, "delta": -0.51116},
            "whale": {"buy": 0.0, "sell": 0.0, "delta": 0.0}
        },
        "cvd": -0.43084
    }

    from scripts.diagnostics.audit_market_data import check_sector_flow, NOT_COMPARABLE
    results = check_sector_flow(ev)

    rec_check = [r for r in results if r["check"] == "sector_flow.reconciliation"]
    assert len(rec_check) == 1
    assert rec_check[0]["status"] == NOT_COMPARABLE
    assert rec_check[0]["severity"] == "NONE"
    assert "sector_flow is cumulative" in rec_check[0]["details"]

    failures = [r for r in results if r["status"] == FAIL]
    assert len(failures) == 0

