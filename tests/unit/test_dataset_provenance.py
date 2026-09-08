# tests/unit/test_dataset_provenance.py — P1-C1: provenance no dataset + gate.
#
# Valor numérico de return_* inalterado (0.0); corrigidos provenance e
# elegibilidade. Dataset legado sem a coluna NUNCA vira ready.

import pandas as pd
import pytest

from ml.dataset_collector import DatasetCollector, provenance_fields
from ml.returns_validator import ReturnsValidator


def test_empty_warmup_invalid_value_kept():
    ret, valid, reason = ReturnsValidator.validate_return(100.0, [], 1)
    assert (ret, valid, reason) == (0.0, False, "EMPTY")
    ret, valid, reason = ReturnsValidator.validate_return(101.0, [100.0], 1)
    assert (ret, valid, reason) == (0.0, False, "WARMUP")


def test_provenance_fields_derivation():
    # feature_ready = qualidade dos DADOS (_ml_usable); staleness do MODELO
    # viaja junto mas não força not-ready (senão o retreino seria inalcançável
    # com o modelo spot atual). Staleness bloqueia INFERÊNCIA (P1-C3).
    ok = {"_ml_usable": True, "_features_real_count": 9,
          "ml_stale": False, "valid_for_futures": True}
    full = provenance_fields(ok)
    assert full["feature_ready"] is True
    assert full["feature_schema_version"] == 1
    assert full["features_valid_count"] == 9
    assert provenance_fields({})["feature_ready"] is False
    assert provenance_fields(None)["feature_ready"] is False
    assert provenance_fields({**ok, "ml_stale": True})["feature_ready"] is True
    assert provenance_fields({"_ml_usable": False})["feature_ready"] is False


def _rows(pred, n=16, start=100.0):
    feats = {"price_close": start, **provenance_fields(pred)}
    return [(dict(feats, price_close=start + i), start + i, 1_700_000_000_000 + i)
            for i in range(n)]


def test_roundtrip_collector_parquet_loader(tmp_path, monkeypatch):
    import ml.dataset_collector as dc

    monkeypatch.setattr(dc, "BUFFER_PATH", tmp_path / "buf.parquet")
    col = DatasetCollector(flush_every=1)
    ok = {"_ml_usable": True, "_features_real_count": 9,
          "ml_stale": False, "valid_for_futures": True}
    for feats, price, ts in _rows(ok):
        col.collect_window(features=feats, price_close=price, timestamp=ts)
    for feats, price, ts in _rows({}):
        col.collect_window(features=feats, price_close=price + 1000,
                           timestamp=ts + 10_000_000)
    df = pd.read_parquet(tmp_path / "buf.parquet")
    assert "feature_ready" in df.columns
    assert int((df["feature_ready"] == True).sum()) == 16  # noqa: E712
    assert int((df["feature_ready"] == True).sum()) < len(df)

    from ml.train_model import ModelTrainer
    trainer = ModelTrainer.__new__(ModelTrainer)  # gate puro, sem I/O
    kept, report = trainer._apply_provenance_gate(df)
    assert report["eligible_rows"] == 16
    assert report["legacy_missing_column"] is False
    assert report["schema_mismatch_rows"] == 0
    assert report["unversioned_rows"] == 0
    assert len(kept) == 16
    assert (kept["feature_ready"] == True).all()  # noqa: E712


@pytest.mark.parametrize("ready,ver,keep", [
    (True, 1, True),          # ready + v1 -> KEEP
    (True, 99, False),        # versão incompatível -> DROP
    (True, None, False),      # versão ausente -> DROP unversioned
    (True, float("nan"), False),
    (False, 1, False),        # not ready -> DROP
    (None, 1, False),
    (1, 1, False),            # int 1 NÃO é True explícito
    ("True", 1, False),       # string NÃO é True explícito
])
def test_gate_per_row_types(ready, ver, keep):
    import numpy as np

    from ml.train_model import ModelTrainer

    df = pd.DataFrame([{"price_close": 100.0, "feature_ready": ready,
                        "feature_schema_version": ver}])
    trainer = ModelTrainer.__new__(ModelTrainer)
    kept, report = trainer._apply_provenance_gate(df)
    assert (len(kept) == 1) is keep
    assert (report["eligible_rows"] == 1) is keep


def test_gate_mixed_frame_keeps_only_ready_v1():
    import numpy as np

    from ml.train_model import ModelTrainer

    df = pd.DataFrame([
        {"price_close": 100.0, "feature_ready": True, "feature_schema_version": 1},
        {"price_close": 101.0, "feature_ready": True, "feature_schema_version": None},
        {"price_close": 102.0, "feature_ready": True, "feature_schema_version": 99},
        {"price_close": 103.0, "feature_ready": False, "feature_schema_version": 1},
    ])
    trainer = ModelTrainer.__new__(ModelTrainer)
    kept, report = trainer._apply_provenance_gate(df)
    assert report["input_rows"] == 4
    assert report["eligible_rows"] == 1
    assert report["unversioned_rows"] == 1
    assert report["schema_mismatch_rows"] == 1
    assert report["not_ready_rows"] == 3
    assert list(kept["price_close"]) == [100.0]


def test_gate_numpy_bool_accepted():
    import numpy as np

    from ml.train_model import ModelTrainer

    df = pd.DataFrame([{"price_close": 100.0,
                        "feature_ready": np.bool_(True),
                        "feature_schema_version": np.int64(1)}])
    trainer = ModelTrainer.__new__(ModelTrainer)
    kept, _ = trainer._apply_provenance_gate(df)
    assert len(kept) == 1


def test_legacy_loader_aborts_clearly(tmp_path):
    """Loader real com dataset legado => None (abort claro, sem relaxar)."""
    import shutil

    from ml.train_model import ModelTrainer

    shutil.copy("ml/datasets/training_dataset.parquet",
                tmp_path / "training_dataset.parquet")
    trainer = ModelTrainer.__new__(ModelTrainer)
    trainer.features_dir = tmp_path
    trainer.config = {"data": {"max_file_size_mb": 100, "chunk_size": 10000}}
    assert trainer.load_and_validate_data() is None


def test_legacy_dataset_aborts_clearly():
    df = pd.read_parquet("ml/datasets/training_dataset.parquet")
    assert "feature_ready" not in df.columns
    assert len(df) == 30

    from ml.train_model import ModelTrainer
    trainer = ModelTrainer.__new__(ModelTrainer)
    kept, report = trainer._apply_provenance_gate(df)
    assert report == {"input_rows": 30, "eligible_rows": 0,
                      "not_ready_rows": 30, "unversioned_rows": 0,
                      "schema_mismatch_rows": 0, "legacy_missing_column": True}
    assert len(kept) == 0
    assert len(kept) < trainer._min_ready_rows()  # treino aborta, sem relaxar
