# tests/unit/test_ml_imbalance_missing_contract.py
"""
B-P0-2: flow_imbalance ausente nunca vira +/-1.0 artificial.

Premissa verificada: flow_imbalance NÃO está em
model_metadata_latest.json (9 features: price/returns/bb/rsi/volume_ratio),
logo missing permanece missing (contrato FeatureStore/dataset: coluna
ausente, nunca extremo forjado).

Contrato:
  - chave ausente (+ net != 0) => chave omitida (sem derivação extrema);
  - +/-1.0 observado legitimamente => preservado exatamente;
  - 0.0 legítimo => preservado;
  - NaN/Inf presente => omitido (JSON final limpo via allow_nan=False).
"""

import json

from common.ml_features import calculate_microstructure_features

REPO_METADATA = "ml/models/model_metadata_latest.json"


def _no_sentinels(obj):
    text = json.dumps(obj, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text
    return text


def test_missing_imbalance_stays_missing():
    micro = calculate_microstructure_features(
        {}, {"order_flow": {"net_flow_1m": -5.0}}, df=None
    )
    assert "flow_imbalance" not in micro, "ausência virou extremo"
    _no_sentinels(micro)


def test_legit_extreme_preserved():
    for v in (1.0, -1.0, 0.0, 0.42):
        micro = calculate_microstructure_features(
            {}, {"order_flow": {"flow_imbalance": v}}, df=None
        )
        assert micro["flow_imbalance"] == v, f"{v} legítimo alterado"
    _no_sentinels(micro)


def test_nonfinite_present_omitted():
    for bad in (float("nan"), float("inf"), float("-inf"), None):
        micro = calculate_microstructure_features(
            {}, {"order_flow": {"flow_imbalance": bad}}, df=None
        )
        assert "flow_imbalance" not in micro, f"{bad} vazou"
        _no_sentinels(micro)


def test_model_does_not_require_flow_imbalance():
    meta = json.loads(open(REPO_METADATA, encoding="utf-8").read())
    assert "flow_imbalance" not in set(meta["feature_names"])
