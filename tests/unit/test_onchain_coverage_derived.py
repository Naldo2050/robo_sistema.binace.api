# tests/unit/test_onchain_coverage_derived.py
"""PFIX-LOW: data_reliability.onchain_coverage deriva de field_status (P04).

Contrato:
  fonte exclusiva: advanced_analysis.onchain_field_status (quando presente);
  usable = VALID/REAL_ZERO; denominador = SUPPORTED_ONCHAIN_FIELDS (15);
  labels: full (15/15) / partial (1..14) / none (0/15) / unknown (sem fs);
  freshness (onchain_status) é ortogonal: não altera coverage.

Escopo estrito PFIX-LOW (sem P02/orderbook/CVD/payload/dedup/indicadores).
"""
import copy

from fetchers.onchain_fetcher import (
    SUPPORTED_ONCHAIN_FIELDS,
    USABLE_ONCHAIN_STATES,
)
from institutional.enricher import _derive_onchain_coverage, enrich_signal


def _fs_all(state="VALID"):
    return {f: state for f in SUPPORTED_ONCHAIN_FIELDS}


def _event(fs=None, status="fresh"):
    adv = {"onchain_metrics": {"mempool_size": 30073.0},
           "onchain_status": status,
           "onchain_age_seconds": 5.0,
           "onchain_source": "blockchain.info+mempool.space"}
    if fs is not None:
        adv["onchain_field_status"] = dict(fs)
    return {
        "symbol": "BTCUSDT",
        "tipo_evento": "ANALYSIS_TRIGGER",
        "preco_fechamento": 80000.0,
        "epoch_ms": 1789158300000,
        "raw_event": {
            "preco_fechamento": 80000.0,
            "volume_total": 1.0,
            "symbol": "BTCUSDT",
            "advanced_analysis": adv,
        },
    }


# Denominador canônico: 15, sem paid/metadata/difficulty_adjustment.
def test_supported_fields_contract():
    assert len(SUPPORTED_ONCHAIN_FIELDS) == 15
    assert USABLE_ONCHAIN_STATES == {"VALID", "REAL_ZERO"}
    for paid in ("exchange_netflow", "whale_transactions",
                 "exchange_reserves", "sopr"):
        assert paid not in SUPPORTED_ONCHAIN_FIELDS
    for meta in ("data_source", "is_real_data", "difficulty_adjustment"):
        assert meta not in SUPPORTED_ONCHAIN_FIELDS


# Paridade: _merge_metrics sempre emite exatamente o conjunto suportado.
def test_merge_emits_exactly_supported_fields():
    from fetchers.onchain_fetcher import OnchainFetcher
    f = OnchainFetcher()
    b_names = ("hash_rate_eh", "difficulty", "n_tx_24h",
               "miners_revenue_btc_24h", "unconfirmed_txs",
               "minutes_between_blocks", "total_btc_sent_24h",
               "total_fees_btc_24h", "trade_volume_btc_24h")
    blockchain = {n: 10.0 for n in b_names}
    blockchain["_status"] = {n: "VALID" for n in (
        "hash_rate_eh", "difficulty", "n_tx_24h",
        "miners_revenue_btc_24h", "unconfirmed_txs",
        "minutes_between_blocks", "total_btc_sent_24h",
        "total_fees_btc_24h", "trade_volume_btc_24h", "stats")}
    mempool = {
        "mempool": {"count": 100, "vsize": 5000000, "total_fee": 100000},
        "fees": {"fastest_sat_vb": 5, "half_hour_sat_vb": 3,
                 "hour_sat_vb": 1, "economy_sat_vb": 1},
        "difficulty_adjustment": {"progress_pct": 15.0},
        "_status": {"mempool.count": "VALID",
                    "mempool.vsize_bytes": "VALID",
                    "mempool.total_fee_btc": "VALID",
                    "fees.fastest_sat_vb": "VALID",
                    "fees.half_hour_sat_vb": "VALID",
                    "fees.hour_sat_vb": "VALID",
                    "fees.economy_sat_vb": "VALID"},
    }
    merged = f._merge_metrics(blockchain, mempool)
    assert set(merged["_field_status"]) == set(SUPPORTED_ONCHAIN_FIELDS)


# A) 15 VALID -> FULL 100.
def test_a_all_valid_full_100():
    label, pct, usable, total = _derive_onchain_coverage(_fs_all("VALID"))
    assert (label, pct, usable, total) == ("full", 100.0, 15, 15)


# B) 14 VALID + 1 REAL_ZERO -> FULL 100 (REAL_ZERO == válido).
def test_b_real_zero_equals_valid():
    fs = _fs_all("VALID")
    fs["fees_fastest_sat_vb"] = "REAL_ZERO"
    assert _derive_onchain_coverage(fs) == ("full", 100.0, 15, 15)


# C) caso LIVE pós-P04: 6 VALID + 2 REAL_ZERO + 6 API_ERROR + 1 MISSING.
def test_c_live_8_of_15_partial_53_3():
    fs = {
        "difficulty": "VALID", "active_addresses": "VALID",
        "hash_rate": "REAL_ZERO", "miner_flows": "REAL_ZERO",
        "mempool_size": "VALID", "minutes_between_blocks": "VALID",
        "total_btc_sent_24h": "VALID", "trade_volume_btc_24h": "VALID",
        "mempool_vsize_mb": "API_ERROR", "mempool_total_fee_btc": "API_ERROR",
        "fees_fastest_sat_vb": "API_ERROR",
        "fees_half_hour_sat_vb": "API_ERROR",
        "fees_hour_sat_vb": "API_ERROR",
        "fees_economy_sat_vb": "API_ERROR",
        "total_fees_btc_24h": "MISSING",
    }
    label, pct, usable, total = _derive_onchain_coverage(fs)
    assert label == "partial"
    assert usable == 8 and total == 15
    assert pct == 53.3
    assert 0 <= pct <= 100


# D) 1/15 -> PARTIAL ~6.7.
def test_d_single_valid_partial():
    fs = {f: "API_ERROR" for f in SUPPORTED_ONCHAIN_FIELDS}
    fs["mempool_size"] = "VALID"
    label, pct, usable, total = _derive_onchain_coverage(fs)
    assert label == "partial"
    assert (usable, total, pct) == (1, 15, 6.7)


# E) 0/15 -> NONE 0 (nunca FULL nem PARTIAL).
def test_e_zero_usable_none():
    for bad in ("API_ERROR", "MISSING"):
        label, pct, usable, _ = _derive_onchain_coverage(_fs_all(bad))
        assert label == "none"
        assert pct == 0.0 and usable == 0


# F) sem field_status -> UNKNOWN, pct None (pct omitido no evento).
def test_f_legacy_unknown():
    for fs in (None, {}, "fresh"):
        label, pct, _, _ = _derive_onchain_coverage(fs)
        assert label == "unknown" and pct is None
    out = enrich_signal(_event(fs=None))
    assert out["data_reliability"]["onchain_coverage"] == "unknown"
    assert "onchain_coverage_pct" not in out["data_reliability"]


# G) fresh vs stale, mesma field_status -> mesma coverage (ortogonalidade).
def test_g_freshness_orthogonal():
    fs = _fs_all("VALID")
    fs["fees_fastest_sat_vb"] = "API_ERROR"
    got = set()
    for status in ("fresh", "stale", "warming_up", "unavailable"):
        out = enrich_signal(_event(fs=fs, status=status))
        dr = out["data_reliability"]
        got.add((dr["onchain_coverage"], dr["onchain_coverage_pct"]))
        # freshness do evento continua intacta e independente
        assert out["raw_event"]["advanced_analysis"]["onchain_status"] == status
    assert got == {("partial", 93.3)}


# H) monotonicidade por campo.
def test_h_monotonicity():
    base = _fs_all("VALID")
    _, pct_base, _, _ = _derive_onchain_coverage(base)

    worse = dict(base, difficulty="API_ERROR")
    _, pct_worse, _, _ = _derive_onchain_coverage(worse)
    assert pct_worse < pct_base  # VALID -> API_ERROR nunca aumenta

    better = dict(worse, difficulty="VALID")
    _, pct_better, _, _ = _derive_onchain_coverage(better)
    assert pct_better >= pct_worse  # API_ERROR -> VALID nunca diminui

    up = dict(base)
    up["difficulty"] = "MISSING"
    up2 = dict(up, difficulty="REAL_ZERO")
    assert _derive_onchain_coverage(up2)[1] > _derive_onchain_coverage(up)[1]
    assert _derive_onchain_coverage(up)[1] < pct_base  # REAL_ZERO->MISSING cai


# I) paid fields não afetam o denominador.
def test_i_paid_fields_ignored():
    fs = _fs_all("API_ERROR")
    fs["mempool_size"] = "VALID"
    fs_plus = dict(fs, exchange_netflow="VALID",
                    whale_transactions="VALID",
                    exchange_reserves="VALID", sopr="VALID")
    assert _derive_onchain_coverage(fs_plus) == _derive_onchain_coverage(fs)


# J) metadata/difficulty_adjustment não afetam o denominador.
def test_j_metadata_ignored():
    fs = _fs_all("API_ERROR")
    fs["mempool_size"] = "VALID"
    fs_plus = dict(fs, data_source="blockchain.info+mempool.space",
                    is_real_data=True, difficulty_adjustment={"progress_pct": 1.0})
    assert _derive_onchain_coverage(fs_plus) == _derive_onchain_coverage(fs)


# Evento fim-a-fim: caso LIVE com status fresh -> partial 53.3 (não "full").
def test_event_live_case_partial_despite_fresh():
    fs = {
        "difficulty": "VALID", "active_addresses": "VALID",
        "hash_rate": "REAL_ZERO", "miner_flows": "REAL_ZERO",
        "mempool_size": "VALID", "minutes_between_blocks": "VALID",
        "total_btc_sent_24h": "VALID", "trade_volume_btc_24h": "VALID",
        "mempool_vsize_mb": "API_ERROR", "mempool_total_fee_btc": "API_ERROR",
        "fees_fastest_sat_vb": "API_ERROR",
        "fees_half_hour_sat_vb": "API_ERROR",
        "fees_hour_sat_vb": "API_ERROR",
        "fees_economy_sat_vb": "API_ERROR",
        "total_fees_btc_24h": "MISSING",
    }
    out = enrich_signal(_event(fs=copy.deepcopy(fs), status="fresh"))
    dr = out["data_reliability"]
    assert dr["onchain_coverage"] == "partial"
    assert dr["onchain_coverage_pct"] == 53.3
