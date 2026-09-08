# tests/unit/test_dominance_change_missing.py
"""
B-P0-4: btc_dominance_change_7d sem medição permanece missing (2 camadas).

Produtor (cross_asset_correlations) hardcodava 0.0; consumidor
(ml_features) defaultava get(..., 0.0). 0.0 constante ia para dataset,
FeatureStore e payload como "variação 0%" observada.

Contrato:
  - sem medição: chave ausente no produtor; NaN no mapeamento (como os
    demais campos); nunca 0.0;
  - 0.0 REAL explicitamente fornecido: permanece 0.0;
  - FeatureStore/Parquet/dataset aceitam missing sem virar zero;
  - payload não inventa variação 0%.
"""

import pandas as pd

from common.ml_features import _map_correlations_to_features


def test_producer_omits_unmeasured_key(monkeypatch):
    """Camada 1: produtor não emite a chave sem medição (sem rede)."""
    import market_analysis.cross_asset_correlations as ca

    monkeypatch.setattr(
        ca, "get_btc_eth_correlations", lambda now_utc=None, stop_event=None: {"status": "ok"}
    )
    monkeypatch.setattr(
        ca, "get_btc_macro_correlations", lambda now_utc=None, stop_event=None: {"status": "ok"}
    )
    monkeypatch.setattr(
        ca, "_run_async_safely", lambda coro, timeout=30.0: {"gold": 4000.0}
    )
    out = ca.get_enhanced_cross_asset_correlations()
    assert "btc_dominance_change_7d" not in out, "0.0 hardcoded no produtor"


def test_mapping_missing_is_nan_not_zero():
    """Camada 2: ausente => NaN (como os demais), nunca 0.0."""
    import math

    feats = _map_correlations_to_features({})
    assert math.isnan(feats["btc_dominance_change_7d"]), "ausente virou 0.0"


def test_real_zero_is_preserved():
    """0.0 medido de verdade continua 0.0."""
    feats = _map_correlations_to_features({"btc_dominance_change_7d": 0.0})
    assert feats["btc_dominance_change_7d"] == 0.0


def test_featurestore_roundtrip_keeps_missing(tmp_path):
    """Parquet guarda missing como null, nunca 0."""
    from data_processing.feature_store import FeatureStore

    feats = _map_correlations_to_features({})
    store = FeatureStore(base_dir=str(tmp_path))
    store.save_features("W1", feats)
    store._flush()
    parts = list(tmp_path.rglob("*.parquet"))
    assert parts, "nenhum parquet gravado"
    df = pd.read_parquet(parts[0])
    val = df["btc_dominance_change_7d"].iloc[0]
    assert not (val == 0.0), "missing virou 0 no parquet"
    assert pd.isna(val), "missing deve ser null"
