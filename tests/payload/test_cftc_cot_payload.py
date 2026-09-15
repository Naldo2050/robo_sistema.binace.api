# tests/payload/test_cftc_cot_payload.py
# -*- coding: utf-8 -*-
"""
Payload CFTC COT (P6): chave 'cftc' independente, flag-gated, sem claims.

Garante: pos x cftc separados; flag OFF reproduz baseline; UNSUPPORTED/
UNAVAILABLE/INVALID somem; PARTIAL/STALE visíveis; sem NaN/Inf; sem
"smart money"; seção <= 400 bytes.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, ".")

from market_orchestrator.ai.payload_builder_compact import (
    _build_cftc_cot,
    build_compact_payload,
)


def _snap(**over):
    base = {
        "schema_version": 1,
        "source": "cftc_cot",
        "report_family": "TFF",
        "report_scope": "futures_only",
        "symbol": "BTCUSDT",
        "market_and_exchange_name": "BITCOIN - CHICAGO MERCANTILE EXCHANGE",
        "cftc_contract_market_code": "133741",
        "report_as_of_date": "2026-09-08",
        "age_reference_seconds": 350000.0,
        "status": "AVAILABLE",
        "is_available": True,
        "is_stale": False,
        "positions": {
            "dealer": {"net": 2943},
            "asset_manager": {"net": 3743},
            "leveraged": {"net": -7892},
            "other": {"net": 669},
            "nonreportable": {"net": 537},
        },
        "open_interest": {"total": 21083, "change_wow": 1386},
        "error_code": None,
    }
    base.update(over)
    return base


def _event(extra=None):
    ev = {
        "symbol": "BTCUSDT",
        "tipo_evento": "ANALYSIS_TRIGGER",
        "preco_fechamento": 77500.0,
        "institutional_analytics": {
            "positioning": {
                "global_account_ratio": 1.28,
                "top_position_ratio": 2.07,
                "regime": "TOP_LONG_DIVERGENCE",
                "is_available": True,
            }
        },
    }
    if extra:
        ev.update(extra)
    return ev


def test_flag_off_omits_cftc(monkeypatch):
    monkeypatch.setenv("ENABLE_CFTC_COT_CONTEXT", "0")
    assert _build_cftc_cot({"cftc_cot": _snap()}) == {}


def test_flag_off_baseline_identical(monkeypatch):
    monkeypatch.setenv("ENABLE_CFTC_COT_CONTEXT", "0")
    base = build_compact_payload(_event())
    with_cftc = build_compact_payload(_event({"cftc_cot": _snap()}))
    assert "cftc" not in base and "cftc" not in with_cftc
    # builder possui cache temporal (ctx/summary variam entre chamadas);
    # a garantia P6 é: mesmas chaves e mesmo 'pos', sem 'cftc'.
    assert set(base) == set(with_cftc)
    assert base.get("pos") == with_cftc.get("pos")


def test_flag_on_available_section(monkeypatch):
    monkeypatch.setenv("ENABLE_CFTC_COT_CONTEXT", "1")
    out = _build_cftc_cot({"cftc_cot": _snap()})
    assert out["st"] == "AVAILABLE"
    assert out["fam"] == "TFF" and out["asof"] == "2026-09-08"
    assert out["oi"] == 21083 and out["oi_wow"] == 1386
    assert out["net_lev"] == -7892
    assert len(json.dumps(out, allow_nan=False)) <= 400


def test_unavailable_statuses_omitted(monkeypatch):
    monkeypatch.setenv("ENABLE_CFTC_COT_CONTEXT", "1")
    for st in ("UNSUPPORTED", "UNAVAILABLE", "INVALID", ""):
        assert _build_cftc_cot({"cftc_cot": _snap(status=st)}) == {}, st


def test_partial_and_stale_visible(monkeypatch):
    monkeypatch.setenv("ENABLE_CFTC_COT_CONTEXT", "1")
    assert _build_cftc_cot({"cftc_cot": _snap(status="PARTIAL")})["st"] == "PARTIAL"
    assert _build_cftc_cot({"cftc_cot": _snap(status="STALE")})["st"] == "STALE"


def test_pos_and_cftc_separated(monkeypatch):
    monkeypatch.setenv("ENABLE_CFTC_COT_CONTEXT", "1")
    compact = build_compact_payload(_event({"cftc_cot": _snap()}))
    assert "pos" in compact and "cftc" in compact
    assert compact["pos"]["ga"] == 1.28
    assert compact["cftc"]["oi"] == 21083
    assert "smart" not in json.dumps(compact).lower()
    json.dumps(compact, allow_nan=False)


def test_no_nan_inf(monkeypatch):
    monkeypatch.setenv("ENABLE_CFTC_COT_CONTEXT", "1")
    snap = _snap()
    snap["positions"]["leveraged"]["net"] = float("nan")
    snap["open_interest"]["total"] = float("inf")
    out = _build_cftc_cot({"cftc_cot": snap})
    assert "net_lev" not in out and "oi" not in out
    json.dumps(out, allow_nan=False)
