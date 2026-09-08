# tests/golden/test_gw_ml.py — Golden ML provenance (sem treinar, sem rede).
#
# Pipeline real: LiveFeatureCalculator -> provenance_fields ->
# DatasetCollector -> parquet -> _apply_provenance_gate; elegibilidade e
# quant via shapes reais. P1-C parado para fórmula/retreino.

import pandas as pd
import pytest

from market_orchestrator.ai.payload_builder_compact import _build_quant
from ml.dataset_collector import DatasetCollector, provenance_fields
from ml.feature_calculator import LiveFeatureCalculator
from ml.hybrid_decision import HybridDecisionMaker, is_eligible_prediction
from ml.inference_engine import sanitize_dmatrix_values

from .conftest import FakeClock  # noqa: F401  (clock congelado via fixture abaixo)


def _feed(n, start=100.0):
    calc = LiveFeatureCalculator()
    for i in range(n):
        calc.update(price=start + i * 0.5 + (i % 3) * 0.1, volume=1.0 + (i % 2))
    return calc


def _collect_rows(tmp_path, monkeypatch, feats, price, ts, pred, n=16):
    import ml.dataset_collector as dc

    monkeypatch.setattr(dc, "BUFFER_PATH", tmp_path / "buf.parquet")
    col = DatasetCollector(flush_every=1000)
    for i in range(n):
        row = dict(feats)
        row.update(provenance_fields(pred))
        col.collect_window(features=row, price_close=price + i,
                           timestamp=ts + i)
    col._flush_to_disk()
    return pd.read_parquet(tmp_path / "buf.parquet")


def _eligible_pred(prob_up=0.75):
    return {"status": "ok", "prob_up": prob_up, "confidence": 0.8,
            "valid_for_futures": True, "ml_stale": False,
            "_ml_usable": True, "_features_real_count": 9}


def test_mlgw1_ready_end_to_end(frozen_state, tmp_path, monkeypatch):
    calc = _feed(25)
    out = calc.compute()
    assert out["_features_real_count"] == 9
    assert out["_ml_usable"] is True
    assert out["_features_default_list"] == []
    pred = {"status": "ok", "prob_up": 0.62, "confidence": 0.7,
            "valid_for_futures": True, "ml_stale": False,
            "_ml_usable": out["_ml_usable"],
            "_features_real_count": out["_features_real_count"]}
    assert is_eligible_prediction(pred) is True
    feats = {k: out[k] for k in ("price_close", "return_1", "return_5",
                                 "return_10", "bb_upper", "bb_lower",
                                 "bb_width", "rsi", "volume_ratio")}
    df = _collect_rows(tmp_path, monkeypatch, feats, out["price_close"],
                       1_700_000_000_000, pred)
    assert (df["feature_ready"] == True).all()  # noqa: E712
    assert (df["feature_schema_version"] == 1).all()
    from ml.train_model import ModelTrainer
    trainer = ModelTrainer.__new__(ModelTrainer)
    kept, report = trainer._apply_provenance_gate(df)
    assert len(kept) == len(df) and report["schema_mismatch"] == 0


def test_mlgw2_warmup_stored_but_gated(frozen_state, tmp_path, monkeypatch):
    # 2 closes = 4/9 reais < 5: usable False (5 é o mínimo de participação).
    calc = _feed(2)
    out = calc.compute()
    assert out["_ml_usable"] is False
    assert out["_features_real_count"] == 4
    pred = {"status": "ok", "prob_up": 0.6, "ml_stale": True,
            "valid_for_futures": False, "_ml_usable": False,
            "_features_real_count": out["_features_real_count"]}
    assert is_eligible_prediction(pred) is False
    feats = {k: out[k] for k in ("price_close", "return_1", "rsi")}
    df = _collect_rows(tmp_path, monkeypatch, feats, out["price_close"],
                       1_700_000_000_000, pred)
    assert (df["feature_ready"] == False).all()  # noqa: E712
    from ml.train_model import ModelTrainer
    trainer = ModelTrainer.__new__(ModelTrainer)
    kept, _ = trainer._apply_provenance_gate(df)
    assert len(kept) == 0  # armazenada p/ auditoria, fora do treino


def test_mlgw3_legacy_auditable_not_trainable(frozen_state, tmp_path, monkeypatch):
    import ml.dataset_collector as dc

    monkeypatch.setattr(dc, "BUFFER_PATH", tmp_path / "buf.parquet")
    col = DatasetCollector(flush_every=1000)
    for i in range(16):
        col.collect_window(features={"price_close": 100.0 + i, "rsi": 50.0},
                           price_close=100.0 + i,
                           timestamp=1_700_000_000_000 + i)
    col._flush_to_disk()
    df = pd.read_parquet(tmp_path / "buf.parquet")
    assert "feature_ready" not in df.columns  # auditável...
    from ml.train_model import ModelTrainer
    trainer = ModelTrainer.__new__(ModelTrainer)
    kept, report = trainer._apply_provenance_gate(df)
    assert len(kept) == 0 and report["legacy_missing_column"] is True


def test_mlgw4_nonfinite_policy(frozen_state):
    calc = LiveFeatureCalculator()
    for _ in range(25):
        calc.update(price=100.0, volume=1.0)
    out = calc.compute()
    assert out["return_1"] == 0.0  # flat observado (real, não fallback)
    vec = [out["price_close"], out["return_1"], float("inf"),
           float("-inf"), float("nan")]
    clean = sanitize_dmatrix_values(vec)
    assert clean[0] == 100.0 and clean[1] == 0.0
    assert all(v != float("inf") and v != float("-inf") for v in clean[2:])
    assert clean[4] != clean[4]  # NaN preservado como missing nativo


def test_mlgw5_not_for_futures_never_evidence(frozen_state):
    pred = {"status": "ok", "prob_up": 0.93, "confidence": 0.9,
            "valid_for_futures": False, "ml_stale": True,
            "_ml_usable": True, "_features_real_count": 9}
    assert is_eligible_prediction(pred) is False
    q = _build_quant({"ml_prediction": pred})
    assert "pu" not in q and q["ml_stale"] is True
    HybridDecisionMaker._instance = None
    HybridDecisionMaker._initialized = False
    maker = HybridDecisionMaker()
    maker._last_model_prob = None
    res = maker.fuse_decisions(pred, {"action": "sell", "confidence": 0.8,
                                      "sentiment": "bearish",
                                      "rationale": "t"})
    assert res.model_prob_up is None
    assert res.action == "sell"


def test_mlgw6_eligible_neutral_stays_neutral(frozen_state):
    pred = _eligible_pred(prob_up=0.5)
    pred["confidence"] = 0.0
    assert is_eligible_prediction(pred) is True
    q = _build_quant({"ml_prediction": pred})
    assert q["pu"] == 0.5 and q["ml_stale"] is False
    q2 = _build_quant({})
    assert "pu" not in q2  # fallback jamais confundido com neutra real
