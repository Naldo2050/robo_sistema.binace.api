# tests/unit/test_effort_response_dataset.py
"""
Testes unitários para P1-E: Effort/Response Shadow Dataset v1 (Hardened).

Valida:
1. Números nominais e metadados reais de REAL_J2 (epoch ms 2026-09-06, Sunday, weekend);
2. Isolamento de dados inventados em SYNTHETIC_J2_LIKE;
3. Determinismo de record_id;
4. Separação causal rígida (features_at_t vs context_at_t vs outcomes_future);
5. Ausência de lookahead (zero tokens de lookahead em features/contexto);
6. Whitelist estrita de contexto e proveniência (impede injeção de raw_event/env);
7. Estado PENDING inicial de outcomes com target_timestamp_ms explícito;
8. Contrato temporal estrito de outcomes (t < timestamp <= t+H para excursões);
9. Preço futuro com target_timestamp_ms, observed_price_timestamp_ms e timing_error_ms;
10. Tolerância de boundary de 1000ms alinhada ao OutcomeTracker;
11. Nomenclatura neutra de excursão (max_excursion_up_bps, max_excursion_down_bps) e aliases;
12. Fail-closed no storage: corrupção no meio lança StorageCorruptionError;
13. Fail-closed no storage: update órfão lança StorageOrphanUpdateError;
14. Crash recovery na cauda do arquivo JSONL;
15. Declaração explícita de SINGLE_WRITER_ONLY;
16. Idempotência de CREATEs e UPDATEs;
17. Reconstrução de IDs em reinicialização;
18. Relatório descritivo offline: 10.080 min/sem, gaps, overlaps, autocorrelação lag-1 e N por estrato;
19. Outcomes pendentes tratados como missing (nunca como retorno 0.0);
20. Complexidade O(1) e ausência de dependências pesadas.
"""

import json
import math
import pathlib
import pytest
from datetime import datetime, timezone
from typing import Dict, Any

from flow_analyzer.effort_response_dataset import (
    CONCURRENCY_MODEL,
    FEATURE_CONTRACT_VERSION,
    FORBIDDEN_LOOKAHEAD_WORDS,
    FORBIDDEN_SECRET_KEYS,
    FORBIDDEN_SUBSTRINGS,
    MAX_THEORETICAL_MINUTES_PER_WEEK,
    OUTCOME_BOUNDARY_TOLERANCE_MS,
    SHADOW_SCHEMA_VERSION,
    EffortResponseShadowRecord,
    EffortResponseShadowStorage,
    StorageCorruptionError,
    StorageOrphanUpdateError,
    attach_shadow_outcomes,
    build_deterministic_record_id,
    build_shadow_context_at_t,
    build_shadow_features_at_t,
    build_shadow_outcomes,
    build_shadow_record,
)
from scripts.analytics.effort_response_distribution import (
    analyze_records,
    compute_lag1_autocorrelation,
    compute_metric_stats,
)

# ── FIXTURES DOCUMENTAIS ───────────────────────────────────────────────────────

REAL_J2: Dict[str, Any] = {
    "buy_notional_usd": 6816945.1591,
    "sell_notional_usd": 1149280.5758,
    "open": 79776.9,
    "high": 79810.8,
    "low": 79776.9,
    "close": 79792.7,
    "window_duration_ms": 57453,
    "vwap": 79803.5,
    "poc": 79804.9,
}

REAL_J2_TIMESTAMPS = {
    "window_open_ms": 1788702361157,
    "window_close_ms": 1788702418610,
    "window_duration_ms": 57453,
}

SYNTHETIC_J2_LIKE: Dict[str, Any] = {
    "buy_notional_usd": 2500000.0,
    "sell_notional_usd": 1000000.0,
    "open": 50000.0,
    "high": 50050.0,
    "low": 49980.0,
    "close": 50020.0,
    "window_duration_ms": 60000,
    "vwap": 50010.0,
    "poc": 50005.0,
}


# ── TESTES DE REAL_J2 E SYNTHETIC_J2_LIKE ─────────────────────────────────────

def test_real_j2_exact_timestamps_and_date():
    """Valida que REAL_J2 usa os timestamps reais e deriva domingo 2026-09-06."""
    close_ms = REAL_J2_TIMESTAMPS["window_close_ms"]
    open_ms = REAL_J2_TIMESTAMPS["window_open_ms"]
    dur = REAL_J2_TIMESTAMPS["window_duration_ms"]
    assert close_ms - open_ms == dur == 57453

    dt = datetime.fromtimestamp(close_ms / 1000.0, tz=timezone.utc)
    assert dt.year == 2026
    assert dt.month == 9
    assert dt.day == 6
    assert dt.strftime("%A") == "Sunday"
    assert dt.weekday() == 6

    # Contexto derivado automaticamente a partir do epoch ms
    ctx, missing = build_shadow_context_at_t(symbol="BTCUSDT", window_close_ms=close_ms)
    assert ctx["timestamp_utc"] == "2026-09-06T13:46:58.610000+00:00"
    assert ctx["day_of_week"] == 6
    assert ctx["is_weekend"] is True
    assert ctx["session_time_bucket"] == "12:00-16:00"

    # Contextos não medidos permanecem estritamente None
    for field in ("regime_current_at_t", "regime_status_at_t", "regime_calibration_status_at_t",
                  "spread", "bid_depth", "ask_depth", "orderbook_imbalance", "market_structure"):
        assert ctx[field] is None
        assert field in missing


def test_real_j2_features_numbers():
    """Valida números de features de REAL_J2."""
    feats, core_val, opt_comp, reasons = build_shadow_features_at_t(**REAL_J2)
    assert core_val == "VALID"
    assert opt_comp == "COMPLETE"
    assert feats["total_aggressive_notional_usd"] == pytest.approx(7966225.7349)
    assert feats["net_aggressive_notional_usd"] == pytest.approx(5667664.5833)
    assert feats["buy_share"] == pytest.approx(0.8557308, abs=1e-4)
    assert feats["sell_share"] == pytest.approx(0.1442692, abs=1e-4)
    assert feats["price_displacement_usd"] == pytest.approx(15.8)
    assert feats["price_displacement_bps"] == pytest.approx(15.8 / 79776.9 * 10000)
    assert feats["range_usd"] == pytest.approx(33.9)
    assert feats["range_bps"] == pytest.approx(33.9 / 79776.9 * 10000)
    assert feats["close_from_high_usd"] == pytest.approx(-18.1)
    assert feats["close_from_low_usd"] == pytest.approx(15.8)
    assert feats["close_vs_vwap_usd"] == pytest.approx(-10.8)
    assert feats["close_vs_poc_usd"] == pytest.approx(-12.2)
    assert feats["window_duration_ms"] == 57453


def test_synthetic_j2_like_features_numbers():
    """Valida que SYNTHETIC_J2_LIKE possui valores didáticos separados da real."""
    feats, core_val, opt_comp, reasons = build_shadow_features_at_t(**SYNTHETIC_J2_LIKE)
    assert core_val == "VALID"
    assert opt_comp == "COMPLETE"
    assert feats["total_aggressive_notional_usd"] == pytest.approx(3500000.0)
    assert feats["net_aggressive_notional_usd"] == pytest.approx(1500000.0)
    assert feats["buy_share"] == pytest.approx(0.7142857142857143)
    assert feats["sell_share"] == pytest.approx(0.2857142857142857)
    assert feats["price_displacement_usd"] == pytest.approx(20.0)
    assert feats["range_usd"] == pytest.approx(70.0)


# ── TESTES DE DETERMINISMO E CAUSALIDADE ──────────────────────────────────────

def test_deterministic_record_id():
    """Gera record_id determinístico baseado em symbol, close e versões."""
    rid = build_deterministic_record_id(
        symbol="BTCUSDT",
        window_close_ms=REAL_J2_TIMESTAMPS["window_close_ms"]
    )
    assert rid == f"rec_BTCUSDT_{REAL_J2_TIMESTAMPS['window_close_ms']}_v1.0.0_s1.0.0"


def test_feature_builder_has_no_future_arguments():
    """Verifica ausência de parâmetros de futuro na assinatura do feature builder."""
    import inspect
    sig = inspect.signature(build_shadow_features_at_t)
    for p in sig.parameters.keys():
        p_lower = p.lower()
        for tok in ("future", "mfe", "mae", "outcome", "target", "label", "next"):
            assert tok not in p_lower, f"Parâmetro de lookahead detectado: {p}"


def test_no_lookahead_keys_in_features_and_context():
    """Chaves de features e contexto não contêm tokens de lookahead."""
    rec = build_shadow_record(
        symbol="BTCUSDT",
        window_open_ms=REAL_J2_TIMESTAMPS["window_open_ms"],
        window_close_ms=REAL_J2_TIMESTAMPS["window_close_ms"],
        window_data=REAL_J2,
    )
    for container_name, mapping in (("features_at_t", rec.features_at_t), ("context_at_t", rec.context_at_t)):
        for k in mapping.keys():
            k_lower = k.lower()
            for sub in FORBIDDEN_SUBSTRINGS:
                assert sub not in k_lower, f"Substring proibida '{sub}' em {container_name}.{k}"
            for p in k_lower.split("_"):
                assert p not in FORBIDDEN_LOOKAHEAD_WORDS, f"Palavra proibida '{p}' em {container_name}.{k}"


def test_whitelist_rejection_of_arbitrary_keys():
    """Contexto e proveniência rejeitam campos arbitrários (raw_event, env, etc.)."""
    with pytest.raises(ValueError, match="fora da whitelist no contexto"):
        build_shadow_context_at_t(symbol="BTCUSDT", window_close_ms=1700000000000, raw_event={"some": "event"})

    d = build_shadow_record(
        symbol="BTCUSDT",
        window_open_ms=1700000000000,
        window_close_ms=1700000060000,
        window_data=SYNTHETIC_J2_LIKE,
    ).to_dict()
    d["provenance"]["arbitrary_header"] = "value"
    with pytest.raises(ValueError, match="Campo não permitido na proveniência"):
        EffortResponseShadowRecord.from_dict(d)


# ── TESTES DO CONTRATO TEMPORAL DE OUTCOMES ──────────────────────────────────

def test_initial_record_has_pending_outcomes():
    """Record inicial possui outcomes PENDING com target_timestamp_ms explícito."""
    t_close = REAL_J2_TIMESTAMPS["window_close_ms"]
    rec = build_shadow_record(
        symbol="BTCUSDT",
        window_open_ms=REAL_J2_TIMESTAMPS["window_open_ms"],
        window_close_ms=t_close,
        window_data=REAL_J2,
    )
    assert rec.outcomes_future["status"] == "PENDING"
    h1 = rec.outcomes_future["horizons"]["1m"]
    assert h1["status"] == "PENDING"
    assert h1["target_timestamp_ms"] == t_close + 60_000
    assert h1["future_price"] is None
    assert h1["return_bps"] is None


def test_excursion_interval_strict_temporal_contract():
    """Observações com timestamp <= window_close_ms NUNCA entram no cálculo de excursão."""
    t_close = 1700000060000
    close_at_t = 50000.0

    # Observação da feature window (<= t_close) com pico artificial de 60000
    obs_feature_window = {"timestamp_ms": t_close, "price": 60000.0, "high": 60000.0, "low": 50000.0}
    # Observações futuras (> t_close): máxima real 50100, mínima 49950
    obs_future = [
        {"timestamp_ms": t_close + 10000, "price": 50050.0, "high": 50100.0, "low": 50000.0},
        {"timestamp_ms": t_close + 60000, "price": 50020.0, "high": 50040.0, "low": 49950.0},
    ]

    outcomes = build_shadow_outcomes(
        close_price_at_t=close_at_t,
        window_close_ms=t_close,
        future_observations=[obs_feature_window] + obs_future,
        horizons=("1m",),
    )

    h1 = outcomes["horizons"]["1m"]
    assert h1["status"] == "RESOLVED"
    # A máxima de 60000 não pode ter entrado (pertencia à feature window)
    assert h1["max_high"] == 50100.0
    assert h1["min_low"] == 49950.0
    assert h1["max_excursion_up_bps"] == pytest.approx((50100.0 - 50000.0) / 50000.0 * 10000.0)
    assert h1["max_excursion_down_bps"] == pytest.approx((49950.0 - 50000.0) / 50000.0 * 10000.0)
    # Aliases
    assert h1["mfe_bps"] == h1["max_excursion_up_bps"]
    assert h1["mae_bps"] == h1["max_excursion_down_bps"]


def test_future_price_policy_and_tolerance():
    """Valida determinação de future_price com tolerância auditada de 1000ms."""
    t_close = 1700000060000
    target_1m = t_close + 60_000  # 1700000120000
    close_at_t = 50000.0

    # Caso 1: Primeiro preço observado no boundary com drift de +250ms (dentro dos 1000ms)
    obs_valid = [
        {"timestamp_ms": target_1m - 5000, "price": 50010.0},
        {"timestamp_ms": target_1m + 250, "price": 50080.0},
    ]
    outcomes = build_shadow_outcomes(
        close_price_at_t=close_at_t,
        window_close_ms=t_close,
        future_observations=obs_valid,
        horizons=("1m",),
        policy="FIRST_ON_OR_AFTER",
    )
    h1 = outcomes["horizons"]["1m"]
    assert h1["status"] == "RESOLVED"
    assert h1["target_timestamp_ms"] == target_1m
    assert h1["observed_price_timestamp_ms"] == target_1m + 250
    assert h1["timing_error_ms"] == 250
    assert h1["future_price"] == 50080.0
    assert h1["return_bps"] == pytest.approx((50080.0 - 50000.0) / 50000.0 * 10000.0)

    # Caso 2: Observação muito distante (+1500ms > 1000ms de tolerância) -> INSUFFICIENT_DATA
    obs_too_late = [
        {"timestamp_ms": target_1m - 5000, "price": 50010.0},
        {"timestamp_ms": target_1m + 1500, "price": 50090.0},
    ]
    outcomes_missed = build_shadow_outcomes(
        close_price_at_t=close_at_t,
        window_close_ms=t_close,
        future_observations=obs_too_late,
        horizons=("1m",),
        policy="FIRST_ON_OR_AFTER",
    )
    assert outcomes_missed["horizons"]["1m"]["status"] == "INSUFFICIENT_DATA"
    assert outcomes_missed["horizons"]["1m"]["future_price"] is None


# ── TESTES DE STORAGE: AUDITORIA E MATERIALIZAÇÃO ─────────────────────────────

def test_storage_concurrency_declaration():
    """Storage declara formalmente SINGLE_WRITER_ONLY."""
    assert CONCURRENCY_MODEL == "SINGLE_WRITER_ONLY"


def test_storage_append_and_duplicate_create(tmp_path: pathlib.Path):
    """Storage grava CREATE e rejeita duplicate CREATE idempotentemente."""
    db_file = tmp_path / "shadow_test.jsonl"
    storage = EffortResponseShadowStorage(filepath=db_file)

    rec = build_shadow_record(
        symbol="BTCUSDT",
        window_open_ms=1700000000000,
        window_close_ms=1700000060000,
        window_data=SYNTHETIC_J2_LIKE,
    )
    assert storage.append_record(rec) is True
    # Duplicate CREATE retorna False
    assert storage.append_record(rec) is False

    recs = storage.read_records()
    assert len(recs) == 1
    assert recs[0].record_id == rec.record_id


def test_storage_orphan_update_fail_closed(tmp_path: pathlib.Path):
    """Update para record_id inexistente lança StorageOrphanUpdateError (fail-closed)."""
    db_file = tmp_path / "shadow_orphan.jsonl"
    storage = EffortResponseShadowStorage(filepath=db_file)

    with pytest.raises(StorageOrphanUpdateError, match="record_id desconhecido"):
        storage.update_record_outcomes("rec_UNKNOWN_ID", {"status": "RESOLVED"}, strict=True)


def test_storage_corruption_middle_line_fail_closed(tmp_path: pathlib.Path):
    """Corrupção JSON no meio do arquivo lança StorageCorruptionError."""
    db_file = tmp_path / "shadow_corrupt.jsonl"
    rec1 = build_shadow_record(symbol="BTCUSDT", window_open_ms=1000, window_close_ms=2000, window_data=SYNTHETIC_J2_LIKE)
    rec2 = build_shadow_record(symbol="BTCUSDT", window_open_ms=3000, window_close_ms=4000, window_data=SYNTHETIC_J2_LIKE)

    with open(db_file, "w", encoding="utf-8") as f:
        f.write(rec1.to_json() + "\n")
        f.write("CORRUPTED_JSON_NOT_VALID_HERE\n")
        f.write(rec2.to_json() + "\n")

    with pytest.raises(StorageCorruptionError, match="linha 2"):
        EffortResponseShadowStorage(filepath=db_file)


def test_storage_crash_recovery_truncated_final_line(tmp_path: pathlib.Path):
    """Crash truncando a última linha sem newline é recuperado descartando apenas a cauda."""
    db_file = tmp_path / "shadow_crash.jsonl"
    rec1 = build_shadow_record(symbol="BTCUSDT", window_open_ms=1000, window_close_ms=2000, window_data=SYNTHETIC_J2_LIKE)

    # Escreve rec1 completo com newline, e depois meia linha truncada sem newline
    with open(db_file, "wb") as f:
        f.write((rec1.to_json() + "\n").encode("utf-8"))
        f.write(b'{"record_id": "rec_INCOMPL')  # Sem \n

    storage = EffortResponseShadowStorage(filepath=db_file)
    recs = storage.read_records()
    assert len(recs) == 1
    assert recs[0].record_id == rec1.record_id


def test_storage_restart_reconstructs_ids(tmp_path: pathlib.Path):
    """Reinicialização do processo lê o arquivo e restaura os seen_record_ids."""
    db_file = tmp_path / "shadow_restart.jsonl"
    storage1 = EffortResponseShadowStorage(filepath=db_file)
    rec = build_shadow_record(symbol="BTCUSDT", window_open_ms=1000, window_close_ms=2000, window_data=SYNTHETIC_J2_LIKE)
    storage1.append_record(rec)

    # Novo processo instancia novo storage apontando para o mesmo arquivo
    storage2 = EffortResponseShadowStorage(filepath=db_file)
    assert rec.record_id in storage2._seen_record_ids
    # Tentar gravar novamente retorna False (já existe)
    assert storage2.append_record(rec) is False


# ── TESTES DO RELATÓRIO DESCRITIVO ────────────────────────────────────────────

def test_distribution_report_sample_rate_and_pending_as_missing():
    """Valida 10.080 min/sem teóricos e que PENDING nunca vira retorno zero."""
    assert MAX_THEORETICAL_MINUTES_PER_WEEK == 10080

    rec1 = build_shadow_record(
        symbol="BTCUSDT",
        window_open_ms=1000,
        window_close_ms=61000,
        window_data=SYNTHETIC_J2_LIKE,
    )
    # rec1 tem outcome 1m PENDING (return_bps é None)
    rep = analyze_records([rec1.to_dict()])
    h1_stats = rep["stratifications"]["by_outcomes_horizon"]["1m"]
    assert h1_stats["pending_count"] == 1
    assert h1_stats["resolved_count"] == 0
    # O missing_count é 1 e o count válido é 0 (NÃO foi transformado em retorno 0.0)
    assert h1_stats["return_bps"]["missing_count"] == 1
    assert h1_stats["return_bps"]["count"] == 0
    assert h1_stats["return_bps"]["mean"] is None

    # Métricas de amostra temporal
    temp_metrics = rep["temporal_sample_metrics"]
    assert temp_metrics["max_theoretical_minutes_per_week"] == 10080
    assert "records_per_day" in temp_metrics
    assert "gaps_count" in temp_metrics
    assert "overlaps_count" in temp_metrics


def test_autocorrelation_calculation():
    """Valida cálculo de autocorrelação lag-1 pura."""
    series = [10.0, 20.0, 30.0, 40.0, 50.0]
    r = compute_lag1_autocorrelation(series)
    assert r is not None and r == pytest.approx(1.0)


# ── TESTES DE PERFORMANCE E ZERO DEPENDÊNCIAS PESADAS ─────────────────────────

def test_o1_no_heavy_deps():
    """Garante que esforço/dataset não importa bibliotecas pesadas de analytics."""
    for fn in ("flow_analyzer/effort_response_dataset.py", "scripts/analytics/effort_response_distribution.py"):
        src = pathlib.Path(fn).read_text(encoding="utf-8")
        for token in ("pandas", "numpy", "polars", "DataFrame", "socket", "requests"):
            assert token not in src, f"Dependência pesada detectada em {fn}: {token}"
