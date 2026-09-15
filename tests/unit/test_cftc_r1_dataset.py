# tests/unit/test_cftc_r1_dataset.py
# -*- coding: utf-8 -*-
"""
R1.8 — Dataset histórico CFTC. Determinístico, sem rede (séries sintéticas).
Cobre: normalização, duplicata, semana faltante, OI zero, string numérica,
NaN/Inf, categoria ausente, percentis (insuficiente/26/52/156),
standard/micro separados, RFC 8259.
"""
import json
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "../../scripts/analytics")))

import cftc_r1_build_dataset as r1


def _tue(n):
    return (date(2024, 1, 2) + timedelta(days=7 * n)).isoformat()


def _row(n, oi="10000", lev_long="3000", lev_short="1000", **over):
    r = {
        "id": f"test{n}", "market_and_exchange_names": "BITCOIN - CHICAGO MERCANTILE EXCHANGE",
        "report_date_as_yyyy_mm_dd": _tue(n) + "T00:00:00.000",
        "contract_market_name": "BITCOIN", "cftc_contract_market_code": "133741",
        "open_interest_all": oi,
        "dealer_positions_long_all": "1000", "dealer_positions_short_all": "500",
        "dealer_positions_spread_all": "0",
        "asset_mgr_positions_long": "2000", "asset_mgr_positions_short": "1500",
        "asset_mgr_positions_spread": "0",
        "lev_money_positions_long": lev_long, "lev_money_positions_short": lev_short,
        "lev_money_positions_spread": "0",
        "other_rept_positions_long": "100", "other_rept_positions_short": "50",
        "other_rept_positions_spread": "0",
        "tot_rept_positions_long_all": "6100", "tot_rept_positions_short": "3050",
        "nonrept_positions_long_all": "3900", "nonrept_positions_short_all": "6950",
        "contract_units": "(5 Bitcoins)", "futonly_or_combined": "FutOnly",
    }
    r.update(over)
    return r


def test_normalize_string_numbers_and_sums():
    rec = r1.normalize_row("133741", _row(0))
    assert rec["open_interest"] == 10000
    assert rec["positions"]["leveraged"] == {
        "long": 3000, "short": 1000, "spreading": 0}
    assert rec["quality_flags"] == []  # 6100+3900=10000 fecha


def test_duplicate_keeps_first_and_flags():
    recs = [r1.normalize_row("133741", _row(0)),
            r1.normalize_row("133741", _row(0))]
    val = r1.validate_series("133741", recs)
    assert val["n_duplicates"] == 1
    assert val["duplicate_asofs"] == [_tue(0)]
    assert val["n_weeks"] == 1


def test_missing_week_gap_no_forward_fill():
    recs = [r1.normalize_row("133741", _row(n)) for n in (0, 2)]
    val = r1.validate_series("133741", recs)
    assert val["n_gaps"] == 1 and val["gaps"][0]["missing_weeks"] == 1
    feats = r1.derive_features(recs)
    assert feats[1]["leveraged_net_change_1w"] is None
    assert feats[1]["gap_before"] is True


def test_oi_zero_nulls_shares():
    rec = r1.normalize_row("133741", _row(0, oi="0"))
    assert rec["open_interest"] == 0 and "oi_zero" in rec["quality_flags"]
    feats = r1.derive_features([rec])
    assert feats[0]["leveraged_net"] == 2000
    assert feats[0]["leveraged_net_share_oi"] is None


def test_nan_inf_and_negative_rejected():
    rec = r1.normalize_row("133741", _row(0, lev_long="NaN"))
    assert rec["positions"]["leveraged"]["long"] is None
    rec2 = r1.normalize_row("133741", _row(0, lev_short="Infinity"))
    assert rec2["positions"]["leveraged"]["short"] is None
    rec3 = r1.normalize_row("133741", _row(0, oi="-5"))
    assert rec3["open_interest"] is None


def test_missing_category_partial():
    r = _row(0)
    del r["asset_mgr_positions_long"]
    rec = r1.normalize_row("133741", r)
    assert rec["positions"]["asset_manager"]["long"] is None
    assert any("asset_mgr_positions_long" in fl for fl in rec["quality_flags"])


def test_percentile_insufficient_history():
    recs = [r1.normalize_row("133741", _row(n,
                lev_long=str(3000 + n * 10))) for n in range(25)]
    feats = r1.derive_features(recs)
    assert feats[-1]["leveraged_net_share_pct26"] is None
    assert feats[-1]["leveraged_net_share_pct52"] is None


def test_percentile_exact_windows():
    recs = [r1.normalize_row("133741", _row(n,
                lev_long=str(3000 + n * 10))) for n in range(156)]
    feats = r1.derive_features(recs)
    last = feats[-1]
    # série estritamente crescente -> percentil 1.0 em todas as janelas
    assert last["leveraged_net_share_pct26"] == 1.0
    assert last["leveraged_net_share_pct52"] == 1.0
    assert last["leveraged_net_share_pct156"] == 1.0
    # primeira linha: sem histórico
    assert feats[0]["leveraged_net_share_pct26"] is None
    # WoW da segunda linha
    assert feats[1]["leveraged_net_change_1w"] == 10


def test_standard_micro_never_mixed():
    micro = _row(0)
    micro["cftc_contract_market_code"] = "133742"
    rec = r1.normalize_row("133741", micro)
    assert "contract_code_mismatch" in rec["quality_flags"]


def test_rfc8259_strict():
    recs = [r1.normalize_row("133741", _row(n)) for n in range(30)]
    feats = r1.derive_features(recs)
    blob = json.dumps({"records": recs, "feats": feats}, allow_nan=False)
    assert "NaN" not in blob and "Infinity" not in blob


def test_research_provenance_shape():
    assert r1.REPORT_FAMILY == "TFF" and r1.REPORT_SCOPE == "futures_only"
    assert r1.R1_SCHEMA_VERSION == 1
    assert set(r1.CODES) == {"133741", "133742", "146021", "146022"}
    rec = r1.normalize_row("133741", _row(0))
    # research_history nunca fabrica first_seen nem promete point-in-time
    assert "first_seen_at" not in rec
    assert rec["schema_version"] == 1
