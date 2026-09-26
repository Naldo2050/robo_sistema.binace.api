# tests/unit/test_p0d2_regime_failclosed.py — P0-D2: regime fail-closed.
#
# Contrato (sem pesos/thresholds novos, sem mínimo de evidências, sem calibrar):
# - Defaults ADX 25.0 / profile RANGE / structure RANGE_BOUND NÃO votam.
# - Fallback 0.33/0.50/0.17 removido do caminho semântico.
# - NaN/±Inf nunca votam (INVALID) e nunca serializam.
# - 0 votos => INSUFFICIENT_DATA/UNKNOWN/nulls. >=1 voto => PARTIAL +
#   UNCALIBRATED_HEURISTIC (1.0 = 100% dos votos disponíveis).
# - Matemática/pesos dos inputs observados preservados bit a bit.

import json
import math
from pathlib import Path

import pytest

from institutional.enricher import _build_regime_probabilities as regime

ROOT = Path(__file__).resolve().parents[2]


def _full_tf():
    return {tf: {"status": "FULL", "is_temporal_coverage_valid": True}
            for tf in ("1m", "5m", "15m")}


def _valid_trend_event(trend="accelerating_buying"):
    return {"fluxo_continuo": {
        "order_flow": {"buy_sell_ratio": {
            "flow_trend": trend,
            "imbalance_validity": {
                "1m": {"validity": "VALID", "reason": None},
                "5m": {"validity": "VALID", "reason": None}}}},
        "flow_window_integrity": _full_tf()}}


# ── A. evento vazio ──────────────────────────────────────────────────────────

def test_a_empty_event_is_insufficient():
    out = regime({})
    assert out["status"] == "INSUFFICIENT_DATA"
    assert out["current_regime"] == "UNKNOWN"
    assert out["regime_probabilities"] is None
    assert out["regime_change_probability"] is None
    assert out["expected_regime_duration"] is None
    assert out["avg_adx"] is None
    assert out["calibration_status"] == "UNCALIBRATED_HEURISTIC"
    assert out["evidence_count"] == 0
    for key in ("adx", "profile_shape", "market_structure", "flow_trend",
                "whale", "orderbook"):
        assert out["evidence"][key]["status"] == "INSUFFICIENT_DATA"
        assert out["evidence"][key]["observed"] is False
    # Antes: MEAN_REVERTING 1.0 (0.15+0.25+0.20 normalizado). Nunca mais.
    assert out["current_regime"] != "MEAN_REVERTING"


# ── B..G. uma única evidência real (defaults alheios não votam) ─────────────

def test_b_only_real_adx():
    out = regime({"multi_tf": {"15m": {"adx": 60}, "1h": {"adx": 60},
                               "4h": {"adx": 60}}})
    assert out["status"] == "PARTIAL"
    assert out["regime_probabilities"] == {"trending": 1.0, "mean_reverting": 0.0,
                                           "breakout": 0.0}
    assert out["current_regime"] == "TRENDING"
    assert out["avg_adx"] == 60.0
    assert out["evidence"]["adx"] == {"status": "VALID", "observed": True}
    assert out["evidence"]["profile_shape"]["status"] == "INSUFFICIENT_DATA"
    assert out["evidence"]["market_structure"]["status"] == "INSUFFICIENT_DATA"


def test_c_only_real_profile():
    out = regime({"institutional_analytics": {"profile_analysis": {
        "profile_shape": {"shape": "B", "trading_signal": "BREAKOUT_EXPECTED"}}}})
    assert out["status"] == "PARTIAL"
    assert out["regime_probabilities"]["breakout"] == 1.0
    assert out["current_regime"] == "BREAKOUT"
    assert out["expected_regime_duration"] == "5m-30m"


def test_d_only_real_structure():
    out = regime({"market_environment": {"market_structure": "TRENDING"}})
    assert out["status"] == "PARTIAL"
    assert out["regime_probabilities"]["trending"] == 1.0
    assert out["current_regime"] == "TRENDING"


def test_e_only_real_whale():
    out = regime({"institutional_analytics": {"flow_analysis": {
        "whale_accumulation": {"score": 40}}}})
    assert out["status"] == "PARTIAL"
    assert out["regime_probabilities"]["breakout"] == 1.0
    assert out["evidence"]["whale"] == {"status": "VALID", "observed": True}


def test_f_only_real_orderbook():
    out = regime({"orderbook_data": {"imbalance": 0.9}})
    assert out["status"] == "PARTIAL"
    probs = out["regime_probabilities"]
    assert probs["breakout"] == pytest.approx(round(0.25 / 0.35, 3))
    assert probs["trending"] == pytest.approx(round(0.10 / 0.35, 3))
    assert probs["mean_reverting"] == 0.0
    assert out["evidence"]["orderbook"] == {"status": "VALID", "observed": True}


def test_g_only_valid_flow_trend():
    out = regime(_valid_trend_event("accelerating_buying"))
    assert out["status"] == "PARTIAL"
    assert out["regime_probabilities"]["trending"] == 1.0
    assert out["evidence"]["flow_trend"] == {"status": "VALID", "observed": True}


def test_h_partial_flow_does_not_vote():
    ev = _valid_trend_event("accelerating_buying")
    ev["fluxo_continuo"]["flow_window_integrity"]["5m"] = {
        "status": "WARMING_UP", "is_temporal_coverage_valid": False}
    ev["fluxo_continuo"]["order_flow"]["buy_sell_ratio"][
        "imbalance_validity"]["5m"] = {"validity": "PARTIAL",
                                       "reason": "WARMING_UP"}
    out = regime(ev)
    # Único input era flow PARTIAL => zero votos => INSUFFICIENT.
    assert out["status"] == "INSUFFICIENT_DATA"
    assert out["current_regime"] == "UNKNOWN"
    assert out["regime_probabilities"] is None


# ── I. non-finite ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("event", [
    {"multi_tf": {"15m": {"adx": float("nan")}}},
    {"multi_tf": {"15m": {"adx": float("inf")}}},
    {"institutional_analytics": {"flow_analysis": {
        "whale_accumulation": {"score": float("-inf")}}}},
    {"orderbook_data": {"imbalance": float("nan")}},
])
def test_i_nonfinite_never_votes(event):
    out = regime(event)
    assert out["status"] == "INSUFFICIENT_DATA"
    assert out["regime_probabilities"] is None
    assert out["avg_adx"] is None
    from common.json_safe import json_dumps_rfc8259, sanitize_json_safe
    json_dumps_rfc8259(sanitize_json_safe(out))  # nunca non-finite no JSON


def test_i_nonfinite_status_marked_invalid():
    out = regime({"multi_tf": {"15m": {"adx": float("inf")}},
                  "orderbook_data": {"imbalance": 0.9}})
    assert out["evidence"]["adx"]["status"] == "INVALID"
    # Só o orderbook vota: t=0.10, b=0.25.
    assert out["status"] == "PARTIAL"
    assert out["current_regime"] == "BREAKOUT"


# ── J. replay 15 janelas ─────────────────────────────────────────────────────

def test_j_replay_defaults_only_become_insufficient():
    path = ROOT / "fallback_events" / "eventos_20260307.json"
    events = [json.loads(line) for line in
              path.read_text(encoding="utf-8").splitlines() if line.strip()]
    insufficient = 0
    for e in events:
        out = regime(e)
        # Conta janelas cujo único suporte anterior eram os 3 defaults.
        ev = out["evidence"]
        if (ev["adx"]["status"] == "INSUFFICIENT_DATA"
                and ev["profile_shape"]["status"] == "INSUFFICIENT_DATA"
                and ev["market_structure"]["status"] == "INSUFFICIENT_DATA"):
            # Sem defaults, resta algo real?
            real_left = [k for k, v in ev.items()
                         if v["status"] == "VALID"]
            if not real_left:
                insufficient += 1
                assert out["status"] == "INSUFFICIENT_DATA"
                assert out["current_regime"] == "UNKNOWN"
    # 9 janelas do replay eram MEAN_REVERTING 1.0 só por defaults (auditoria).
    assert insufficient == 9


# ── K. FULL legacy fixture: matemática preservada ────────────────────────────

def test_k_full_inputs_math_unchanged():
    ev = {"multi_tf": {"15m": {"adx": 60}, "1h": {"adx": 60}, "4h": {"adx": 60}},
          "institutional_analytics": {
              "profile_analysis": {"profile_shape": {
                  "shape": "B", "trading_signal": "BREAKOUT_EXPECTED"}},
              "flow_analysis": {"whale_accumulation": {"score": 40}}},
          "orderbook_data": {"imbalance": 0.9},
          "market_environment": {"market_structure": "TRENDING"}}
    ev["fluxo_continuo"] = _valid_trend_event(
        "accelerating_buying")["fluxo_continuo"]
    out = regime(ev)
    # Matemática legada: T=0.4+0.10+0.15+0.20=0.85; M=0; B=0.35+0.20+0.25=0.80.
    assert out["status"] == "PARTIAL"
    assert out["regime_probabilities"]["trending"] == pytest.approx(round(0.85 / 1.65, 3))
    assert out["regime_probabilities"]["mean_reverting"] == 0.0
    assert out["regime_probabilities"]["breakout"] == pytest.approx(round(0.80 / 1.65, 3))
    assert out["current_regime"] == "TRENDING"
    assert out["tie_detected"] is False
    assert out["selection_method"] == "ARGMAX_HEURISTIC"
    assert out["avg_adx"] == 60.0
    assert out["expected_regime_duration"] == "2h-8h"
    assert out["evidence_count"] == 6


def test_k_tie_detected():
    ev = {"orderbook_data": {"imbalance": 0.9},
          "market_environment": {"market_structure": "TRENDING"}}
    # T=0.10+0.20=0.30; B=0.25 → sem empate; usa whale p/ forçar empate:
    # ob 0.9 (T+0.10,B+0.25) + whale 40?? B+0.20 => T=0.10,B=0.45. Sem empate.
    # Empate real: adx>50 (+0.4T) + profile BREAKOUT (+0.35B)... T=0.4,B=0.35.
    # T==B exato: structure TRENDING (+0.2T) + ob>=0.5 (+0.1B) + whale>=15
    # (+0.1B): T=0.2,B=0.2 => empate.
    ev = {"market_environment": {"market_structure": "TRENDING"},
          "orderbook_data": {"imbalance": 0.6},
          "institutional_analytics": {"flow_analysis": {
              "whale_accumulation": {"score": 20}}}}
    out = regime(ev)
    assert out["regime_probabilities"]["trending"] == \
        out["regime_probabilities"]["breakout"] == 0.5
    assert out["tie_detected"] is True
    assert out["selection_method"] == "ARGMAX_HEURISTIC"


# ── L. compact/summary UNKNOWN ───────────────────────────────────────────────

def test_l_compact_mode_unk_and_no_mr_strategies():
    from market_orchestrator.ai.payload_builder_compact import build_compact_payload
    from market_orchestrator.ai.payload_sections.regime_summary import (
        build_regime_summary)

    event = {"symbol": "BTCUSDT", "tipo_evento": "ANALYSIS_TRIGGER",
             "preco_fechamento": 65000.0, "epoch_ms": 1700000000000,
             "ml_features": {}, "multi_tf": {},
             "market_environment": {"market_structure": "RANGE_BOUND"},
             "regime_analysis": {"status": "INSUFFICIENT_DATA",
                                 "current_regime": "UNKNOWN",
                                 "regime_probabilities": None}}
    payload = build_compact_payload(event)
    assert payload["regime"]["mode"] == "UNK"
    summary = build_regime_summary(payload)
    assert summary["label"] == "Indeterminado"
    assert summary["strategies"] == []
    assert summary["avoid"] == []
    flat = json.dumps(summary).lower()
    assert "mean reversion" not in flat and "fade extremos" not in flat


def test_l_legacy_missing_key_keeps_fallback():
    # Eventos legados sem status mantêm o fallback market_environment.
    from market_orchestrator.ai.payload_builder_compact import build_compact_payload
    event = {"symbol": "BTCUSDT", "tipo_evento": "ANALYSIS_TRIGGER",
             "preco_fechamento": 65000.0, "epoch_ms": 1700000000000,
             "ml_features": {}, "multi_tf": {},
             "market_environment": {"market_structure": "RANGE_BOUND"}}
    payload = build_compact_payload(event)
    assert payload["regime"]["mode"] == "RB"


# ── M. RFC8259 ───────────────────────────────────────────────────────────────

def test_m_no_nonfinite_serialized():
    from common.json_safe import json_dumps_rfc8259, sanitize_json_safe
    for ev in ({},
               {"multi_tf": {"15m": {"adx": float("inf")}}},
               {"orderbook_data": {"imbalance": float("-inf")}},
               {"multi_tf": {"15m": {"adx": 60}}}):
        out = regime(ev)
        text = json_dumps_rfc8259(sanitize_json_safe(out))
        assert "NaN" not in text and "Infinity" not in text
        for v in (out["regime_probabilities"] or {}).values():
            assert math.isfinite(v)
