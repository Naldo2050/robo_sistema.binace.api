# tests/unit/test_crossasset_metadata.py
"""
F5-C metadata/migração — method/n/instrument viajam com a feature.

Regra: linhas antigas (sem as chaves) = positional_v1; linhas novas =
shared_session_returns_v2. Nunca misturar em treino futuro. Cross-asset
segue BLOQUEADO para treinamento ML.
"""

import pandas as pd

from common.ml_features import _map_correlations_to_features


def _v2_values():
    return {
        "status": "ok",
        "btc_dxy_corr_30d": -0.45,
        "btc_dxy_corr_90d": -0.44,
        "btc_ndx_corr_30d": 0.12,
        "btc_eth_corr_7d": 0.93,
        "correlation_method": "shared_session_returns_v2",
        "correlation_contract_version": 2,
        "btc_dxy_corr_30d_n": 30,
        "btc_dxy_corr_90d_n": 63,
        "btc_ndx_corr_30d_n": 29,
        "btc_eth_corr_7d_n": 167,
        "btc_eth_corr_30d_n": 700,
        "btc_dxy_instrument": "DX-Y.NYB",
        "nasdaq_instrument": "QQQ",
        "nasdaq_role": "nasdaq_proxy",
        "btc_eth_instrument": "BINANCE:BTCUSDT/ETHUSDT_1h",
    }


def test_mapping_carries_method_n_instrument():
    feats = _map_correlations_to_features(_v2_values())
    assert feats["cross_asset_method"] == "shared_session_returns_v2"
    assert feats["cross_asset_contract_version"] == 2
    assert feats["btc_dxy_corr_30d_n"] == 30
    assert feats["btc_ndx_corr_30d_n"] == 29
    assert feats["btc_dxy_instrument"] == "DX-Y.NYB"
    assert feats["nasdaq_instrument"] == "QQQ"
    assert feats["nasdaq_role"] == "nasdaq_proxy"
    # valores continuam mapeados
    assert feats["btc_dxy_corr_30d"] == -0.45


def test_legacy_values_default_to_positional_v1():
    feats = _map_correlations_to_features({"btc_dxy_corr_30d": 0.16})
    assert feats["cross_asset_contract_version"] == 1  # legado
    assert feats["cross_asset_method"] is None
    assert "btc_dxy_corr_30d_n" not in feats


def test_payload_cross_preserves_method(tmp_path=None):
    from market_orchestrator.ai.payload_builder_compact import (
        build_compact_payload,
    )

    ml = {"cross_asset": _map_correlations_to_features(_v2_values()),
          "cross_asset_status": "fresh",
          "cross_asset_age_seconds": 10.0}
    event = {"symbol": "BTCUSDT", "tipo_evento": "ANALYSIS_TRIGGER",
             "preco_fechamento": 79000.0, "epoch_ms": 1700000060000,
             "ml_features": ml}
    compact = build_compact_payload(event)
    cross = compact["cross"]
    assert cross["st"] == "fresh"
    assert cross["method"] == "shared_session_returns_v2"
    assert cross["n"] == 29  # mínimo entre os N presentes
    assert cross["inst_ndx"] == "QQQ"
    assert cross["inst_dxy"] == "DX-Y.NYB"


def test_payload_cross_omits_method_for_legacy():
    from market_orchestrator.ai.payload_builder_compact import (
        build_compact_payload,
    )

    ml = {"cross_asset": _map_correlations_to_features(
        {"btc_dxy_corr_30d": 0.16}),
        "cross_asset_status": "fresh",
        "cross_asset_age_seconds": 10.0}
    event = {"symbol": "BTCUSDT", "tipo_evento": "ANALYSIS_TRIGGER",
             "preco_fechamento": 79000.0, "epoch_ms": 1700000060000,
             "ml_features": ml}
    compact = build_compact_payload(event)
    assert compact["cross"]["st"] == "fresh"
    assert "method" not in compact["cross"]
    assert "n" not in compact["cross"]


def test_legend_documents_method():
    from common.ai_field_legend import FIELD_LEGEND
    assert "shared_session_returns_v2" in FIELD_LEGEND
    assert "positional_v1" in FIELD_LEGEND
    assert "PROXY" in FIELD_LEGEND.upper()


def test_featurestore_roundtrip_preserves_version(tmp_path):
    from data_processing.feature_store import FeatureStore

    feats = _map_correlations_to_features(_v2_values())
    flat = {k: v for k, v in feats.items()
            if isinstance(v, (int, float)) or isinstance(v, str)}
    store = FeatureStore(base_dir=str(tmp_path))
    store.save_features("W1", {**flat,
                               "cross_asset_status": "fresh",
                               "cross_asset_age_seconds": 10.0})
    store._flush()
    parts = list(tmp_path.rglob("*.parquet"))
    assert parts, "nenhum parquet gravado"
    df = pd.read_parquet(parts[0])
    assert "cross_asset_contract_version" in df.columns
    assert "cross_asset_method" in df.columns
    assert "btc_dxy_corr_30d_n" in df.columns
    assert "nasdaq_instrument" in df.columns
    row = df.iloc[0]
    assert int(row["cross_asset_contract_version"]) == 2
    assert str(row["cross_asset_method"]) == "shared_session_returns_v2"
    # filtro de migração: só v2 entra em dataset futuro
    assert str(row["cross_asset_method"]) != "positional_v1"
