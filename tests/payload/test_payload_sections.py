"""
Testes dos summary builders de payload_sections.

Cobre:
  - flow_summary
  - sr_summary
  - regime_summary
  - institutional_summary
  - quality_summary

Para cada builder:
  - caso nominal (dados completos)
  - caso sem dados (fallback)
  - casos de borda semânticos
"""

import pytest
from market_orchestrator.ai.payload_sections import (
    build_flow_summary,
    build_sr_summary,
    build_regime_summary,
    build_institutional_summary,
    build_quality_summary,
)


# ═══════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════

def make_compact_payload() -> dict:
    return {
        "symbol": "BTCUSDT",
        "epoch_ms": 1775173800000,
        "trigger": "AT",
        "price": {
            "c": 66875,
            "o": 66884,
            "h": 66890,
            "sh": "b",
            "auc": "expect_retest_low",
            "pl": 1,
            "brk_risk": "V_HI",
        },
        "regime": {
            "cs": "BEAR",
            "cf": 1.0,
            "v": "H",
            "dom": "4h",
            "mode": "MR",
            "bbw": 0.18,
            "atr%": 0.57,
        },
        "flow": {
            "d1": "+16K",
            "d5": "-8K",
            "d15": "-12K",
            "imb": 0.18,
            "ab": 59,
            "bsr": 1.44,
            "pa": "buy_absorption",
            "conv": "M",
            "abs_buy_str": 5.9,
            "abs_sell_exh": 1.8,
            "abs_cont": 0.22,
            "sf_w": 0.4,
            "sf_r": -0.2,
            "ti": 12.3,
            "trs": 10,
            "delta": 0.123,
            "vol": 1.5,
            "buy_pct": 60,
        },
        "ob": {
            "b": "303K",
            "a": "1.7M",
            "imb": -0.7,
            "bias": "SELL",
            "t5": -0.96,
            "slip_b": 5.0,
            "slip_s": 6.0,
        },
        "tf": {
            "15m": {"t": "DN", "rsi": 45, "adx": 11, "atr": 127, "r": "RNG"},
            "1h":  {"t": "DN", "rsi": 50, "adx": 21, "atr": 384, "r": "RNG"},
            "4h":  {"t": "DN", "rsi": 43, "adx": 27, "atr": 968, "r": "MNP"},
            "1d":  {"t": "DN", "rsi": 40, "adx": 21, "atr": 2525, "r": "ACC"},
        },
        "sr": {
            "r1": [66931, 66],
            "r1_dist": 30,
            "r1_conf": 5,
            "r2": [67771, 52],
            "r2_dist": 896,
            "r2_conf": 3,
            "s1": [66839, 58],
            "s1_dist": 800,
            "s1_conf": 4,
            "s2": [66410, 62],
            "s2_dist": 465,
            "def_bias": "neutral",
        },
        "w": {
            "s": 18,
            "c": "MA",
        },
        "ctx": {
            "ses": "NY",
            "poc": 66444,
            "val": 66388,
            "vah": 67771,
            "lsr": 1.83,
            "oi": 90,
            "fr": -0.01,
        },
        "qual": {
            "lat": "OK",
        },
        "alerts": [
            {"type": "SUPPORT_TEST",    "sev": "H", "lvl": 66839},
            {"type": "RESISTANCE_TEST", "sev": "H", "lvl": 66931},
        ],
    }


# ═══════════════════════════════════════════════════════════════════
# FLOW SUMMARY
# ═══════════════════════════════════════════════════════════════════

class TestFlowSummary:

    def test_nominal_returns_required_keys(self):
        result = build_flow_summary(make_compact_payload())
        for key in ("bias", "type", "actor", "conf", "note"):
            assert key in result, f"Chave ausente: {key}"

    def test_nominal_buy_absorption_bias(self):
        result = build_flow_summary(make_compact_payload())
        assert result["bias"] == "BUY"
        assert result["type"] == "absorption"

    def test_nominal_whale_actor_detected(self):
        result = build_flow_summary(make_compact_payload())
        assert result["actor"] == "whale"

    def test_nominal_conviction_medium(self):
        result = build_flow_summary(make_compact_payload())
        assert result["conf"] == "M"

    def test_nominal_note_is_non_empty_string(self):
        result = build_flow_summary(make_compact_payload())
        assert isinstance(result["note"], str)
        assert len(result["note"]) > 10

    def test_divergence_d1_vs_d5_sets_reversal_signal(self):
        payload = make_compact_payload()
        payload["flow"]["d1"] = "+16K"
        payload["flow"]["d5"] = "-8K"
        result = build_flow_summary(payload)
        assert result.get("reversal_signal") is True

    def test_no_divergence_when_both_positive(self):
        payload = make_compact_payload()
        payload["flow"]["d1"] = "+16K"
        payload["flow"]["d5"] = "+8K"
        result = build_flow_summary(payload)
        assert result.get("reversal_signal") is not True

    def test_fallback_when_no_flow(self):
        result = build_flow_summary({})
        assert result["bias"] == "NEUTRAL"
        assert result["conf"] == "L"
        assert "note" in result

    def test_sell_absorption_sets_sell_bias(self):
        payload = make_compact_payload()
        payload["flow"]["pa"] = "sell_absorption"
        payload["flow"]["abs_buy_str"] = 0.5
        payload["flow"]["abs_sell_exh"] = 7.5
        result = build_flow_summary(payload)
        assert result["bias"] == "SELL"
        assert result["type"] == "absorption"

    def test_high_buyer_strength_upgrades_conviction(self):
        payload = make_compact_payload()
        payload["flow"]["abs_buy_str"] = 8.5
        payload["flow"]["abs_sell_exh"] = 0.5
        result = build_flow_summary(payload)
        assert result["conf"] == "H"

    def test_retail_actor_when_only_retail_active(self):
        payload = make_compact_payload()
        payload["flow"]["sf_w"] = 0.0
        payload["flow"]["sf_r"] = -0.45
        result = build_flow_summary(payload)
        assert result["actor"] == "retail"

    def test_unknown_actor_when_both_inactive(self):
        payload = make_compact_payload()
        payload["flow"]["sf_w"] = 0.0
        payload["flow"]["sf_r"] = 0.0
        result = build_flow_summary(payload)
        assert result["actor"] == "unknown"


# ═══════════════════════════════════════════════════════════════════
# SR SUMMARY
# ═══════════════════════════════════════════════════════════════════

class TestSrSummary:

    def test_nominal_returns_required_keys(self):
        result = build_sr_summary(make_compact_payload())
        for key in ("nearest", "compressed", "conf_bias", "note"):
            assert key in result, f"Chave ausente: {key}"

    def test_nominal_resistance_is_nearest(self):
        result = build_sr_summary(make_compact_payload())
        assert result["nearest"] == "resistance"

    def test_nominal_not_compressed(self):
        result = build_sr_summary(make_compact_payload())
        assert result["compressed"] is False

    def test_neutral_def_bias(self):
        result = build_sr_summary(make_compact_payload())
        assert result["conf_bias"] == "NEUTRAL"

    def test_dist_atr_present_when_atr_available(self):
        result = build_sr_summary(make_compact_payload())
        assert "r1_dist_atr" in result
        assert "s1_dist_atr" in result
        assert result["r1_dist_atr"] == round(30 / 384, 2)
        assert result["s1_dist_atr"] == round(800 / 384, 2)

    def test_compression_detected_when_gap_tiny(self):
        payload = make_compact_payload()
        payload["sr"]["r1_dist"] = 30
        payload["sr"]["s1_dist"] = 25
        result = build_sr_summary(payload)
        assert result["compressed"] is True
        assert "gap_usd" in result

    def test_s1_near_flag_when_support_very_close(self):
        payload = make_compact_payload()
        payload["sr"]["s1_dist"] = 5
        result = build_sr_summary(payload)
        assert result.get("s1_near") is True

    def test_r1_near_flag_when_resistance_very_close(self):
        payload = make_compact_payload()
        payload["sr"]["r1_dist"] = 5
        result = build_sr_summary(payload)
        assert result.get("r1_near") is True

    def test_fallback_when_no_sr(self):
        result = build_sr_summary({})
        assert result["nearest"] == "unknown"
        assert result["compressed"] is False
        assert "note" in result

    def test_support_nearest_when_s1_closer(self):
        payload = make_compact_payload()
        payload["sr"]["r1_dist"] = 900
        payload["sr"]["s1_dist"] = 36
        result = build_sr_summary(payload)
        assert result["nearest"] == "support"

    def test_note_is_non_empty_string(self):
        result = build_sr_summary(make_compact_payload())
        assert isinstance(result["note"], str)
        assert len(result["note"]) > 10


# ═══════════════════════════════════════════════════════════════════
# REGIME SUMMARY
# ═══════════════════════════════════════════════════════════════════

class TestRegimeSummary:

    def test_nominal_returns_required_keys(self):
        result = build_regime_summary(make_compact_payload())
        for key in ("label", "strategies", "avoid", "duration", "note"):
            assert key in result, f"Chave ausente: {key}"

    def test_nominal_mode_mr_label(self):
        result = build_regime_summary(make_compact_payload())
        assert result["label"] == "Mean Reversion"

    def test_nominal_strategies_are_list(self):
        result = build_regime_summary(make_compact_payload())
        assert isinstance(result["strategies"], list)
        assert len(result["strategies"]) > 0

    def test_nominal_avoid_are_list(self):
        result = build_regime_summary(make_compact_payload())
        assert isinstance(result["avoid"], list)
        assert len(result["avoid"]) > 0

    def test_bear_consensus_removes_buy_strategies_in_mr(self):
        result = build_regime_summary(make_compact_payload())
        for strategy in result["strategies"]:
            assert "comprar" not in strategy.lower(), (
                f"Estratégia de compra incluída em BEAR MR: '{strategy}'"
            )

    def test_bull_consensus_removes_sell_strategies_in_trending(self):
        payload = make_compact_payload()
        payload["regime"]["mode"] = "TRD"
        payload["regime"]["cs"] = "BULL"
        result = build_regime_summary(payload)
        for strategy in result["strategies"]:
            assert "vender" not in strategy.lower(), (
                f"Estratégia de venda incluída em BULL TRD: '{strategy}'"
            )

    def test_low_bbw_triggers_bbw_note(self):
        payload = make_compact_payload()
        payload["regime"]["bbw"] = 0.07
        result = build_regime_summary(payload)
        assert "bbw_note" in result
        assert "comprimidas" in result["bbw_note"].lower()

    def test_high_bbw_triggers_expansion_note(self):
        payload = make_compact_payload()
        payload["regime"]["bbw"] = 0.55
        result = build_regime_summary(payload)
        assert "bbw_note" in result
        assert "expandidas" in result["bbw_note"].lower()

    def test_high_volatility_triggers_vol_note(self):
        result = build_regime_summary(make_compact_payload())
        assert "vol_note" in result
        assert "alta" in result["vol_note"].lower()

    def test_fallback_when_no_regime(self):
        result = build_regime_summary({})
        assert result["label"] == "Indeterminado"
        assert result["strategies"] == []
        assert result["avoid"] == []

    def test_note_is_non_empty_string(self):
        result = build_regime_summary(make_compact_payload())
        assert isinstance(result["note"], str)
        assert len(result["note"]) > 10

    def test_breakout_mode_label(self):
        payload = make_compact_payload()
        payload["regime"]["mode"] = "BRK"
        result = build_regime_summary(payload)
        assert result["label"] == "Breakout"
        assert "confirmação" in " ".join(result["strategies"]).lower()


# ═══════════════════════════════════════════════════════════════════
# INSTITUTIONAL SUMMARY
# ═══════════════════════════════════════════════════════════════════

class TestInstitutionalSummary:

    def test_nominal_returns_required_keys(self):
        result = build_institutional_summary(make_compact_payload())
        for key in ("auction_state", "whale_bias", "profile_bias", "unfinished", "note"):
            assert key in result, f"Chave ausente: {key}"

    def test_nominal_shape_b_is_bullish(self):
        result = build_institutional_summary(make_compact_payload())
        assert result["profile_bias"] == "BULLISH"

    def test_nominal_whale_ma_is_accumulating(self):
        result = build_institutional_summary(make_compact_payload())
        assert result["whale_bias"] == "ACCUMULATING"

    def test_nominal_poor_low_in_unfinished(self):
        result = build_institutional_summary(make_compact_payload())
        assert "low" in result["unfinished"]

    def test_nominal_auction_state_populated(self):
        result = build_institutional_summary(make_compact_payload())
        assert isinstance(result["auction_state"], str)
        assert len(result["auction_state"]) > 5

    def test_alignment_bull_when_all_signals_bullish(self):
        payload = make_compact_payload()
        payload["price"]["sh"] = "b"
        payload["w"]["c"] = "MA"
        payload["flow"]["pa"] = "buy_absorption"
        result = build_institutional_summary(payload)
        assert result["alignment"] == "BULL_ALIGNED"

    def test_alignment_conflicted_when_signals_diverge(self):
        payload = make_compact_payload()
        payload["price"]["sh"] = "p"
        payload["w"]["c"] = "MA"
        result = build_institutional_summary(payload)
        assert result["alignment"] == "CONFLICTED"

    def test_brk_risk_vhi_appears_in_note(self):
        result = build_institutional_summary(make_compact_payload())
        assert "muito alto" in result["note"].lower() or "alto" in result["note"].lower()

    def test_no_unfinished_when_no_poor_extremes(self):
        payload = make_compact_payload()
        payload["price"].pop("ph", None)
        payload["price"].pop("pl", None)
        result = build_institutional_summary(payload)
        assert result["unfinished"] == []

    def test_distribution_classification(self):
        payload = make_compact_payload()
        payload["w"]["c"] = "SD"
        result = build_institutional_summary(payload)
        assert result["whale_bias"] == "DISTRIBUTING"

    def test_fallback_when_no_price_or_whale(self):
        result = build_institutional_summary({})
        assert result["whale_bias"] == "NEUTRAL"
        assert result["profile_bias"] == "NEUTRAL"
        assert result["unfinished"] == []


# ═══════════════════════════════════════════════════════════════════
# QUALITY SUMMARY
# ═══════════════════════════════════════════════════════════════════

class TestQualitySummary:

    def test_nominal_ok_data_is_reliable(self):
        result = build_quality_summary(make_compact_payload())
        assert result["reliable"] is True
        assert result["confidence_cap"] == 1.0
        assert result["issues"] == []

    def test_degraded_latency_reduces_confidence(self):
        payload = make_compact_payload()
        payload["qual"] = {"lat": "DEGR", "ms": 3622}
        result = build_quality_summary(payload)
        assert result["reliable"] is False
        assert result["confidence_cap"] == 0.7
        assert any("degradada" in i.lower() for i in result["issues"])

    def test_critical_latency_caps_at_04(self):
        payload = make_compact_payload()
        payload["qual"] = {"lat": "CRIT", "ms": 8000}
        result = build_quality_summary(payload)
        assert result["confidence_cap"] == 0.4
        assert result["reliable"] is False

    def test_reduced_liquidity_reduces_confidence(self):
        payload = make_compact_payload()
        payload["qual"] = {"lat": "OK", "liq": "RED"}
        result = build_quality_summary(payload)
        assert result["confidence_cap"] == 0.8
        assert any("reduzida" in i.lower() for i in result["issues"])

    def test_cached_ctx_reduces_confidence(self):
        payload = make_compact_payload()
        payload["ctx"]["cached"] = True
        result = build_quality_summary(payload)
        assert result["confidence_cap"] < 1.0
        assert any("cache" in i.lower() for i in result["issues"])

    def test_combined_degraded_and_reduced_liquidity(self):
        payload = make_compact_payload()
        payload["qual"] = {"lat": "DEGR", "ms": 3622, "liq": "RED"}
        result = build_quality_summary(payload)
        assert result["confidence_cap"] == 0.7
        assert len(result["issues"]) >= 2

    def test_note_includes_confidence_percentage(self):
        payload = make_compact_payload()
        payload["qual"] = {"lat": "DEGR", "ms": 3622}
        result = build_quality_summary(payload)
        assert "70%" in result["note"]

    def test_reliable_note_is_positive(self):
        result = build_quality_summary(make_compact_payload())
        assert "plena" in result["note"].lower() or "real" in result["note"].lower()

    def test_fallback_when_no_qual(self):
        result = build_quality_summary({})
        assert result["reliable"] is True
        assert result["confidence_cap"] == 1.0
        assert result["issues"] == []
