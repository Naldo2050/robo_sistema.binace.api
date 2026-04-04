"""
Golden snapshot test — trava a estrutura canônica do payload compacto.

Objetivo: detectar drift estrutural silencioso.
  - adição de chave inesperada
  - remoção de chave obrigatória
  - mudança de tipo de valor
  - mudança de formato de número compacto

O snapshot NÃO trava valores exatos que mudam naturalmente
(como epoch_ms ou vwap).

Trava:
  - presença e ausência de chaves por seção
  - tipos dos valores
  - formato de campos compactos (ex: "+16K", "BU", "V_HI")
  - estrutura aninhada

Se o builder mudar intencionalmente, atualize EXPECTED_KEYS e
EXPECTED_TYPES para refletir o novo contrato.
"""

import json
import copy
import pytest

import build_compact_payload as bcp


@pytest.fixture(autouse=True)
def reset_static_cache():
    bcp._last_static_ctx = {}
    bcp._last_static_ts = 0.0
    yield
    bcp._last_static_ctx = {}
    bcp._last_static_ts = 0.0


def make_canonical_event() -> dict:
    return {
        "symbol": "BTCUSDT",
        "epoch_ms": 1775173800000,
        "tipo_evento": "ANALYSIS_TRIGGER",
        "descricao": "Evento automático para análise da IA",
        "preco_fechamento": 66875,
        "delta": 0.123,
        "volume_total": 1.5,
        "volume_compra": 0.9,
        "contextual_snapshot": {
            "ohlc": {
                "open": 66884,
                "high": 66890,
                "low": 66856,
                "close": 66875,
                "vwap": 66879,
            }
        },
        "market_context": {
            "trading_session": "NY",
            "session_phase": "ACTIVE",
        },
        "market_environment": {
            "volatility_regime": "HIGH",
            "market_structure": "RANGE_BOUND",
        },
        "historical_vp": {
            "daily": {
                "status": "success",
                "poc": 66444,
                "val": 66388,
                "vah": 67771,
            }
        },
        "external_markets": {
            "VIX": {"preco_atual": 23.9},
            "FEAR_GREED": {"preco_atual": 50},
            "DXY": {"preco_atual": 104.2},
            "SP500": {"preco_atual": 669.1},
        },
        "derivatives": {
            "BTCUSDT": {
                "long_short_ratio": 1.83,
                "open_interest": 90101.906,
                "funding_rate_percent": -0.01,
                "longs_usd": 3893635557.19,
                "shorts_usd": 2129881055.29,
            },
            "ETHUSDT": {"long_short_ratio": 1.66},
        },
        "ml_features": {
            "cross_asset": {
                "btc_eth_corr_7d": 0.92,
                "btc_dxy_corr_30d": -0.41,
            },
            "microstructure": {
                "trade_intensity_v2": 12.3,
                "tick_rule_sum": 10,
                "order_book_slope": -0.25,
            },
        },
        "multi_tf": {
            "15m": {
                "tendencia": "Baixa",
                "rsi_short": 45,
                "macd": 27,
                "macd_signal": 41,
                "adx": 11,
                "atr": 127,
                "regime": "Range",
            },
            "1h": {
                "tendencia": "Baixa",
                "rsi_short": 50,
                "macd": -151,
                "macd_signal": -241,
                "adx": 21,
                "atr": 384,
                "regime": "Range",
            },
            "4h": {
                "tendencia": "Baixa",
                "rsi_short": 43,
                "macd": -178,
                "macd_signal": -64,
                "adx": 27,
                "atr": 968,
                "regime": "Manipulação",
            },
            "1d": {
                "tendencia": "Baixa",
                "rsi_short": 40,
                "macd": -829,
                "macd_signal": -584,
                "adx": 21,
                "atr": 2525,
                "regime": "Acumulação",
            },
        },
        "volatility_metrics": {
            "bbw": 0.18,
            "atr_pct": 0.57,
        },
        "fluxo_continuo": {
            "cvd": 0.2,
            "sector_flow": {
                "whale": {"delta": 0.4},
                "retail": {"delta": -0.2},
            },
            "order_flow": {
                "net_flow_1m": 16000,
                "net_flow_5m": 12000,
                "net_flow_15m": 8000,
                "flow_imbalance": 0.18,
                "aggressive_buy_pct": 59,
                "buy_sell_ratio": {"buy_sell_ratio": 1.44},
            },
            "absorption_analysis": {
                "current_absorption": {
                    "buyer_strength": 5.9,
                    "seller_exhaustion": 1.8,
                    "continuation_probability": 0.22,
                }
            },
        },
        "orderbook_data": {
            "bid_depth_usd": 303000,
            "ask_depth_usd": 1700000,
            "imbalance": -0.7,
        },
        "order_book_depth": {
            "L5": {"imbalance": -0.96},
        },
        "market_impact": {
            "slippage_matrix": {
                "100k_usd": {"buy": 0.05, "sell": 0.06},
            }
        },
        "institutional_analytics": {
            "profile_analysis": {
                "profile_shape": {"shape": "b"},
                "poor_extremes": {
                    "action_bias": "expect_retest_both",
                    "poor_high": {"detected": 1},
                    "poor_low": {"detected": 1, "volume_ratio": 5.145},
                },
                "va_volume_pct": {
                    "breakout_risk": "VERY_HIGH",
                    "compression_signal": 1,
                },
            },
            "quality": {
                "calendar": {"expected_liquidity": "NORMAL"},
                "latency": {
                    "latency_ms": 820,
                    "latency_category": "OK",
                },
            },
            "flow_analysis": {
                "passive_aggressive": {
                    "composite": {
                        "signal": "buy_absorption",
                        "conviction": "MEDIUM",
                    }
                },
                "whale_accumulation": {
                    "score": 18,
                    "classification": "MILD_ACCUMULATION",
                },
            },
            "sr_analysis": {
                "defense_zones": {
                    "status": "success",
                    "sell_defense": [
                        {"center": 66931, "strength": 66, "source_count": 5},
                        {"center": 67771, "strength": 52, "source_count": 3},
                    ],
                    "buy_defense": [
                        {"center": 66839, "strength": 58, "source_count": 4},
                        {"center": 66410, "strength": 62, "source_count": 4},
                    ],
                    "defense_asymmetry": {"bias": "neutral"},
                }
            },
        },
        "technical_indicators_extended": {
            "cci_signal": "NEUTRAL",
            "stochastic": {"k": 51},
            "williams_r": {"value": -58},
            "garch_forecast_1h": 0.0055,
            "hurst_exponent": 0.341,
            "shannon_entropy": 3.461,
            "fractal_dimension": 0.619,
            "kalman_filter": {
                "kalman_price": 66938.24,
                "deviation_pct": -0.0815,
                "trend_direction": "DOWN",
            },
            "regression_channel": {
                "slope_per_bar": -2.8103,
                "position_in_channel": 0.4995,
                "deviation_from_trend": -0.06,
            },
            "monte_carlo": {
                "prob_up": 0.426,
                "p10": 66756.3,
                "p90": 66976.79,
            },
            "dominant_cycles": {
                "dominant_cycles": [66.7, 100.0, 40.0],
            },
        },
        "pattern_recognition": {
            "fibonacci_levels": {
                "high": 67100,
                "low": 66731.08,
                "38.2": 66872.0,
                "61.8": 66959.07,
            },
            "smart_money": {
                "fair_value_gaps": [
                    {"type": "BULLISH"},
                    {"type": "BEARISH"},
                ],
                "market_structure": {
                    "structure": "BEARISH",
                    "bos_detected": 0,
                },
            },
        },
        "alerts": {
            "active_alerts": [
                {"type": "SUPPORT_TEST",    "severity": "HIGH", "level": 66839},
                {"type": "RESISTANCE_TEST", "severity": "HIGH", "level": 66931},
            ]
        },
        "data_reliability": {
            "latency_acceptable": 1,
            "onchain_coverage": "full",
        },
    }


# ────────────────────────────────────────────────────────────
# CONTRATO DE CHAVES
# ────────────────────────────────────────────────────────────

# Chaves obrigatórias no payload raiz
REQUIRED_ROOT_KEYS = {
    "symbol", "epoch_ms", "trigger",
    "price", "regime", "flow", "ob", "tf", "sr", "ctx", "summary",
}

# Chaves que NÃO devem aparecer no payload raiz
FORBIDDEN_ROOT_KEYS = {
    "raw_event",
    "fluxo_continuo",
    "institutional_analytics",
    "technical_indicators_extended",
    "multi_tf",
    "orderbook_data",
    "order_book_depth",
    "pattern_recognition",
    "historical_vp",
    "market_impact",
    "data_reliability",
}

# Chaves obrigatórias em cada seção
REQUIRED_PRICE_KEYS     = {"c"}
REQUIRED_REGIME_KEYS    = {"cs", "cf", "v", "mode"}
REQUIRED_FLOW_KEYS      = {"d1", "imb"}
REQUIRED_OB_KEYS        = {"b", "a", "imb", "bias"}
REQUIRED_CTX_KEYS       = {"ses", "poc", "val", "vah", "lsr"}
REQUIRED_SR_KEYS        = {"r1", "s1"}

# Chaves opcionais que quando presentes devem ter o formato certo
OPTIONAL_FLOW_KEYS      = {"d5", "d15", "cvd", "ab", "bsr", "pa", "conv",
                           "abs_buy_str", "abs_sell_exh", "abs_cont",
                           "sf_w", "sf_r", "ti", "trs", "obs",
                           "delta", "vol", "buy_pct"}
OPTIONAL_PRICE_KEYS     = {"o", "h", "l", "vw", "sh", "auc",
                           "ph", "pl", "brk_risk"}
OPTIONAL_EXT_KEYS       = {"cci", "stoch", "wr", "garch", "hurst",
                           "entropy", "fd", "kalman", "reg", "mc",
                           "cycles", "smc", "fib"}


def test_snapshot_required_root_keys_present():
    payload = bcp.build_compact_payload(make_canonical_event())
    missing = REQUIRED_ROOT_KEYS - set(payload.keys())
    assert not missing, f"Chaves obrigatórias ausentes na raiz: {missing}"


def test_snapshot_forbidden_root_keys_absent():
    payload = bcp.build_compact_payload(make_canonical_event())
    present = FORBIDDEN_ROOT_KEYS & set(payload.keys())
    assert not present, (
        f"Chaves proibidas encontradas na raiz (dados brutos não compactados): {present}"
    )


def test_snapshot_price_section_structure():
    payload = bcp.build_compact_payload(make_canonical_event())
    price = payload["price"]

    missing = REQUIRED_PRICE_KEYS - set(price.keys())
    assert not missing, f"Chaves obrigatórias ausentes em price: {missing}"

    unknown = set(price.keys()) - REQUIRED_PRICE_KEYS - OPTIONAL_PRICE_KEYS
    assert not unknown, f"Chaves inesperadas em price: {unknown}"

    assert isinstance(price["c"], int), "price.c deve ser int"


def test_snapshot_regime_section_structure():
    payload = bcp.build_compact_payload(make_canonical_event())
    regime = payload["regime"]

    missing = REQUIRED_REGIME_KEYS - set(regime.keys())
    assert not missing, f"Chaves obrigatórias ausentes em regime: {missing}"

    assert regime["cs"] in {"BULL", "BEAR", "MIX"}
    assert isinstance(regime["cf"], float)
    assert regime["v"] in {"L", "M", "H", "LOW", "MED", "HIG"}
    assert regime["mode"] in {"MR", "RB", "TRD", "BRK"}


def test_snapshot_flow_section_structure():
    payload = bcp.build_compact_payload(make_canonical_event())
    flow = payload["flow"]

    missing = REQUIRED_FLOW_KEYS - set(flow.keys())
    assert not missing, f"Chaves obrigatórias ausentes em flow: {missing}"

    unknown = set(flow.keys()) - REQUIRED_FLOW_KEYS - OPTIONAL_FLOW_KEYS
    assert not unknown, f"Chaves inesperadas em flow: {unknown}"

    assert isinstance(flow["d1"], str)
    assert flow["d1"].startswith(("+", "-")) or flow["d1"] == "0"
    assert isinstance(flow["imb"], float)


def test_snapshot_orderbook_section_structure():
    payload = bcp.build_compact_payload(make_canonical_event())
    ob = payload["ob"]

    missing = REQUIRED_OB_KEYS - set(ob.keys())
    assert not missing, f"Chaves obrigatórias ausentes em ob: {missing}"

    assert ob["bias"] in {"BUY", "SELL", "NEUT"}
    assert isinstance(ob["b"], str)
    assert isinstance(ob["a"], str)
    assert isinstance(ob["imb"], float)


def test_snapshot_ctx_section_structure():
    payload = bcp.build_compact_payload(make_canonical_event())
    ctx = payload["ctx"]

    missing = REQUIRED_CTX_KEYS - set(ctx.keys())
    assert not missing, f"Chaves obrigatórias ausentes em ctx: {missing}"

    assert isinstance(ctx["ses"], str)
    assert isinstance(ctx["poc"], int)
    assert isinstance(ctx["val"], int)
    assert isinstance(ctx["vah"], int)
    assert isinstance(ctx["lsr"], float)


def test_snapshot_sr_section_structure():
    payload = bcp.build_compact_payload(make_canonical_event())
    sr = payload["sr"]

    missing = REQUIRED_SR_KEYS - set(sr.keys())
    assert not missing, f"Chaves obrigatórias ausentes em sr: {missing}"

    for key in ("r1", "s1"):
        assert isinstance(sr[key], list), f"sr.{key} deve ser list"
        assert len(sr[key]) == 2, f"sr.{key} deve ter [price, strength]"
        assert isinstance(sr[key][0], int), f"sr.{key}[0] deve ser int"
        assert isinstance(sr[key][1], (int, float)), f"sr.{key}[1] deve ser numérico"

    for dist_key in ("r1_dist", "s1_dist"):
        assert dist_key in sr, f"{dist_key} obrigatório em sr"
        assert isinstance(sr[dist_key], int), f"sr.{dist_key} deve ser int"


def test_snapshot_tf_section_has_all_four_timeframes():
    payload = bcp.build_compact_payload(make_canonical_event())
    tf = payload["tf"]

    for name in ("15m", "1h", "4h", "1d"):
        assert name in tf, f"Timeframe ausente em tf: {name}"
        assert tf[name]["t"] in {"UP", "DN", "SW", "NE"}
        assert isinstance(tf[name]["rsi"], int)
        assert isinstance(tf[name]["adx"], (int, float))


def test_snapshot_compact_number_format_consistency():
    payload = bcp.build_compact_payload(make_canonical_event())
    flow = payload["flow"]

    for field in ("d1", "d5", "d15"):
        if field in flow:
            val = flow[field]
            assert isinstance(val, str), f"flow.{field} deve ser str"
            assert (
                val == "0"
                or val[0] in ("+", "-")
            ), f"flow.{field} deve ter sinal: '{val}'"
            assert not val.endswith("K.") and not val.endswith("M."), (
                f"flow.{field} tem formato inválido: '{val}'"
            )


def test_snapshot_brk_risk_values_are_canonical():
    payload = bcp.build_compact_payload(make_canonical_event())
    brk = payload["price"].get("brk_risk")

    if brk is not None:
        assert brk in {"HI", "V_HI", "MOD"}, (
            f"brk_risk deve ser HI/V_HI/MOD, recebeu: '{brk}'"
        )
        assert brk != "VERY", "brk_risk truncado para 'VERY' — bug de truncamento"
        assert brk != "VERY_HIGH", "brk_risk não compactado — deve ser 'V_HI'"


def test_snapshot_fvg_last_is_not_ambiguous():
    payload = bcp.build_compact_payload(make_canonical_event())
    smc = payload.get("ext", {}).get("smc", {})

    if "fvg_last" in smc:
        assert smc["fvg_last"] in {"BU", "BE"}, (
            f"fvg_last ambíguo: '{smc['fvg_last']}' — deve ser 'BU' ou 'BE'"
        )
        assert smc["fvg_last"] != "B", (
            "fvg_last='B' é ambíguo — não distingue BULLISH de BEARISH"
        )


def test_snapshot_structure_is_serializable_to_json():
    payload = bcp.build_compact_payload(make_canonical_event())

    try:
        serialized = json.dumps(payload, ensure_ascii=False)
        reparsed = json.loads(serialized)
    except (TypeError, ValueError) as exc:
        pytest.fail(f"Payload não é serializável como JSON: {exc}")

    assert reparsed["symbol"] == "BTCUSDT"
    assert reparsed["trigger"] == "AT"


# ── Testes adicionais: contrato da seção summary ───────────────────────

REQUIRED_SUMMARY_KEYS = {"flow", "sr", "regime", "institutional", "quality"}

REQUIRED_SUMMARY_FLOW_KEYS        = {"bias", "type", "actor", "conf", "note"}
REQUIRED_SUMMARY_SR_KEYS          = {"nearest", "compressed", "conf_bias", "note"}
REQUIRED_SUMMARY_REGIME_KEYS      = {"label", "strategies", "avoid", "duration", "note"}
REQUIRED_SUMMARY_INSTITUTIONAL_KEYS = {
    "auction_state", "whale_bias", "profile_bias", "unfinished", "note"
}
REQUIRED_SUMMARY_QUALITY_KEYS     = {"reliable", "confidence_cap", "issues", "note"}


def test_snapshot_summary_section_present():
    payload = bcp.build_compact_payload(make_canonical_event())

    if "summary" not in payload:
        pytest.skip("summary não disponível neste ambiente")

    missing = REQUIRED_SUMMARY_KEYS - set(payload["summary"].keys())
    assert not missing, f"Builders ausentes no summary: {missing}"


def test_snapshot_summary_flow_structure():
    payload = bcp.build_compact_payload(make_canonical_event())

    if "summary" not in payload:
        pytest.skip("summary não disponível neste ambiente")

    flow_s = payload["summary"]["flow"]
    missing = REQUIRED_SUMMARY_FLOW_KEYS - set(flow_s.keys())
    assert not missing, f"Chaves ausentes em summary.flow: {missing}"

    assert flow_s["bias"] in {"BUY", "SELL", "NEUTRAL"}
    assert flow_s["type"] in {"absorption", "aggressive", "passive", "mixed"}
    assert flow_s["actor"] in {"whale", "retail", "mixed", "unknown"}
    assert flow_s["conf"] in {"H", "M", "L"}
    assert isinstance(flow_s["note"], str) and len(flow_s["note"]) > 5


def test_snapshot_summary_sr_structure():
    payload = bcp.build_compact_payload(make_canonical_event())

    if "summary" not in payload:
        pytest.skip("summary não disponível neste ambiente")

    sr_s = payload["summary"]["sr"]
    missing = REQUIRED_SUMMARY_SR_KEYS - set(sr_s.keys())
    assert not missing, f"Chaves ausentes em summary.sr: {missing}"

    assert sr_s["nearest"] in {"support", "resistance", "equidistant", "unknown"}
    assert isinstance(sr_s["compressed"], bool)
    assert sr_s["conf_bias"] in {"BUY", "SELL", "NEUTRAL"}
    assert isinstance(sr_s["note"], str) and len(sr_s["note"]) > 5


def test_snapshot_summary_regime_structure():
    payload = bcp.build_compact_payload(make_canonical_event())

    if "summary" not in payload:
        pytest.skip("summary não disponível neste ambiente")

    reg_s = payload["summary"]["regime"]
    missing = REQUIRED_SUMMARY_REGIME_KEYS - set(reg_s.keys())
    assert not missing, f"Chaves ausentes em summary.regime: {missing}"

    assert isinstance(reg_s["label"], str)
    assert isinstance(reg_s["strategies"], list)
    assert isinstance(reg_s["avoid"], list)
    assert isinstance(reg_s["duration"], str)
    assert isinstance(reg_s["note"], str) and len(reg_s["note"]) > 5


def test_snapshot_summary_institutional_structure():
    payload = bcp.build_compact_payload(make_canonical_event())

    if "summary" not in payload:
        pytest.skip("summary não disponível neste ambiente")

    inst_s = payload["summary"]["institutional"]
    missing = REQUIRED_SUMMARY_INSTITUTIONAL_KEYS - set(inst_s.keys())
    assert not missing, f"Chaves ausentes em summary.institutional: {missing}"

    assert inst_s["whale_bias"] in {"ACCUMULATING", "DISTRIBUTING", "NEUTRAL"}
    assert inst_s["profile_bias"] in {"BULLISH", "BEARISH", "NEUTRAL"}
    assert isinstance(inst_s["unfinished"], list)
    assert isinstance(inst_s["note"], str) and len(inst_s["note"]) > 5


def test_snapshot_summary_quality_structure():
    payload = bcp.build_compact_payload(make_canonical_event())

    if "summary" not in payload:
        pytest.skip("summary não disponível neste ambiente")

    qual_s = payload["summary"]["quality"]
    missing = REQUIRED_SUMMARY_QUALITY_KEYS - set(qual_s.keys())
    assert not missing, f"Chaves ausentes em summary.quality: {missing}"

    assert isinstance(qual_s["reliable"], bool)
    assert isinstance(qual_s["confidence_cap"], float)
    assert 0.0 <= qual_s["confidence_cap"] <= 1.0
    assert isinstance(qual_s["issues"], list)
    assert isinstance(qual_s["note"], str) and len(qual_s["note"]) > 5


def test_snapshot_payload_with_summary_still_serializable():
    payload = bcp.build_compact_payload(make_canonical_event())

    try:
        serialized = json.dumps(payload, ensure_ascii=False)
        reparsed = json.loads(serialized)
    except (TypeError, ValueError) as exc:
        pytest.fail(f"Payload com summary não é serializável: {exc}")

    if "summary" in reparsed:
        assert isinstance(reparsed["summary"], dict)


def test_snapshot_payload_with_summary_below_hard_limit():
    payload = bcp.build_compact_payload(make_canonical_event())
    size = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    assert size < 6144, (
        f"Payload com summary ultrapassou hard limit: {size} bytes"
    )
