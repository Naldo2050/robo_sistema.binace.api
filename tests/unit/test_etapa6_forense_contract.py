# -*- coding: utf-8 -*-
"""
ETAPA 6 — Contrato forense macro/intermarket (bug fixes confirmados).

Cobre os 3 BUG_CONFIRMED da auditoria forense:
  1. FRED disk cache: `ts` (epoch UTC) e `updated` (ISO-8601) devem
     representar o MESMO instante, com timezone explícito, independente
     do fuso local da máquina.
  2. Persistência: NaN/+Inf/-Inf -> JSON null (RFC 8259) em
     EventStore (SQLite events.payload) e EventSaver (eventos JSONL/
     fallback). Nunca literal NaN/Infinity; nunca fabricar 0.
  3. AI payload/guardrail: non-finite nunca chega ao prompt/LLM
     (_safe_price/_safe_round/_safe_int/eth7/dxy30/ensure_safe_llm_payload).

NÃO altera: provider FRED/Yahoo, tickers, TTL, séries, fórmulas, cache
de dados, arquivos históricos.
"""

import json
import logging
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from common.json_safe import (
    is_non_finite_number,
    sanitize_json_safe,
    json_dumps_rfc8259,
)


def _strict_json_loads(text: str):
    """Parser JSON estrito: NaN/Infinity literals são rejeitados."""
    return json.loads(
        text,
        parse_constant=lambda c: (_ for _ in ()).throw(
            ValueError(f"literal não-RFC8259 no JSON: {c}")
        ),
    )


# ============================================================
# SANITIZADOR CANÔNICO (common/json_safe.py)
# ============================================================

def test_is_non_finite_number_detects_all_forms():
    assert is_non_finite_number(float("nan")) is True
    assert is_non_finite_number(float("inf")) is True
    assert is_non_finite_number(float("-inf")) is True
    assert is_non_finite_number(0.0) is False
    assert is_non_finite_number(3.14) is False
    assert is_non_finite_number(-7) is False
    assert is_non_finite_number(None) is False
    assert is_non_finite_number("nan") is False
    assert is_non_finite_number(True) is False


def test_sanitize_scalars():
    assert sanitize_json_safe(float("nan")) is None
    assert sanitize_json_safe(float("inf")) is None
    assert sanitize_json_safe(float("-inf")) is None
    assert sanitize_json_safe(None) is None
    assert sanitize_json_safe(0.0) == 0.0
    assert sanitize_json_safe(0) == 0
    assert sanitize_json_safe(5) == 5
    assert sanitize_json_safe(True) is True
    assert sanitize_json_safe("abc") == "abc"


def test_sanitize_nested_structures():
    payload = {
        "ml_features": {
            "cross_asset": {
                "us2y_yield": float("nan"),
                "us10y_change_1d": float("nan"),
                "gold_change_1d": None,
                "btc_dominance_change_7d": 0.0,
                "vix_current": 15.16,
                "tags": ["nan", float("inf"), 1, 2.5],
            }
        },
        "nested": {"a": {"b": float("-inf")}},
    }
    out = sanitize_json_safe(payload)
    ca = out["ml_features"]["cross_asset"]
    assert ca["us2y_yield"] is None
    assert ca["us10y_change_1d"] is None
    assert ca["gold_change_1d"] is None
    assert ca["btc_dominance_change_7d"] == 0.0
    assert ca["vix_current"] == 15.16
    assert ca["tags"] == ["nan", None, 1, 2.5]
    assert out["nested"]["a"]["b"] is None


def test_sanitize_does_not_mutate_input():
    original = {"k": [float("nan"), 1]}
    snapshot = {"k": [float("nan"), 1]}
    out = sanitize_json_safe(original)
    # entradas intactas (NaN != NaN em dict equality; comparar por dump)
    assert json.dumps(original, sort_keys=True, allow_nan=True) == json.dumps(
        snapshot, sort_keys=True, allow_nan=True
    )
    assert math.isnan(original["k"][0]) and original["k"][1] == 1
    assert out is not original
    assert out["k"][0] is None


def test_json_serialization_is_rfc8259():
    payload = {
        "a": float("nan"),
        "b": float("inf"),
        "c": float("-inf"),
        "d": None,
        "e": 0.0,
        "f": 3.14,
        "g": [1, float("nan"), {"x": float("inf")}],
    }
    text = json.dumps(sanitize_json_safe(payload), ensure_ascii=False)
    assert "nan" not in text.lower()
    assert "infinity" not in text.lower()
    data = _strict_json_loads(text)
    assert data["a"] is None and data["b"] is None and data["c"] is None
    assert data["d"] is None
    assert data["e"] == 0.0 and data["f"] == 3.14
    assert data["g"][1] is None and data["g"][2]["x"] is None
    # dumps estrito aceita o resultado sanitizado
    json_dumps_rfc8259(sanitize_json_safe(payload))


def test_numpy_non_finite_if_available():
    np = pytest.importorskip("numpy")
    assert is_non_finite_number(np.float64("nan")) is True
    assert is_non_finite_number(np.float64("inf")) is True
    assert sanitize_json_safe(np.float64("nan")) is None
    assert sanitize_json_safe(np.float64(3.14)) == 3.14


# ============================================================
# BUG 1 — FRED DISK CACHE TIMESTAMP (updated timezone-aware)
# ============================================================

def _make_fred_fetcher(tmp_path):
    from fetchers.fred_fetcher import FREDFetcher

    fetcher = FREDFetcher.__new__(FREDFetcher)
    fetcher._disk_cache = {}
    fetcher._disk_cache_ttl = 86400  # mesmo TTL do __init__ (24h)
    fetcher._disk_cache_path = tmp_path / "fred_cache.json"
    return fetcher


def test_fred_disk_cache_updated_is_timezone_aware(tmp_path):
    fetcher = _make_fred_fetcher(tmp_path)
    fetcher._set_disk_cache("TNX", 4.688)

    entry = fetcher._disk_cache["TNX"]
    assert entry["value"] == 4.688

    # `ts` continua epoch Unix UTC
    ts = entry["ts"]
    assert isinstance(ts, float) and ts > 0

    # `updated` é ISO-8601 com timezone explícito (+00:00)
    updated = entry["updated"]
    assert updated.endswith("+00:00"), updated
    dt = datetime.fromisoformat(updated)
    assert dt.tzinfo is not None
    assert dt.utcoffset() == timedelta(0)

    # ts e updated representam o MESMO instante (tolerância de 1s)
    assert abs(dt.timestamp() - ts) < 1.0
    assert dt == datetime.fromtimestamp(ts, tz=timezone.utc)


def test_fred_disk_cache_parsing_independent_of_local_tz(tmp_path):
    fetcher = _make_fred_fetcher(tmp_path)
    fetcher._set_disk_cache("TNX", 4.688)
    entry = fetcher._disk_cache["TNX"]

    dt = datetime.fromisoformat(entry["updated"])
    # offset UTC explícito -> instante idêntico em qualquer fuso local
    assert dt == datetime.fromtimestamp(entry["ts"], tz=timezone.utc)
    # e o arquivo em disco também é RFC 8259 / strict-parsable
    on_disk = _strict_json_loads(
        (tmp_path / "fred_cache.json").read_text(encoding="utf-8")
    )
    assert on_disk["TNX"]["value"] == 4.688
    assert on_disk["TNX"]["updated"].endswith("+00:00")


def test_fred_disk_cache_legacy_naive_updated_still_readable(tmp_path):
    fetcher = _make_fred_fetcher(tmp_path)
    # Registro antigo (updated naive, sem timezone) — leitura via `ts` deve
    # continuar funcionando; compatibilidade com caches já escritos.
    fetcher._disk_cache["TNX"] = {
        "value": 4.2,
        "ts": time.time(),
        "updated": "2026-08-10T11:25:46.576340",
    }
    assert fetcher._get_from_disk_cache("TNX") == 4.2


# ============================================================
# BUG 2 — PERSISTÊNCIA JSON RFC 8259 (NaN/±Inf -> null)
# ============================================================

@pytest.fixture
def non_finite_event():
    return {
        "epoch_ms": 1786371600000,
        "tipo_evento": "ANALYSIS_TRIGGER",
        "symbol": "BTCUSDT",
        "window_id": "w1",
        "is_signal": True,
        "ml_features": {
            "cross_asset": {
                "us2y_yield": float("nan"),
                "us10y_change_1d": float("nan"),
                "gold_change_1d": None,
                "btc_dominance_change_7d": 0.0,
                "vix_current": 15.16,
            }
        },
        "nested": {
            "pos_inf": float("inf"),
            "neg_inf": float("-inf"),
            "zero": 0.0,
            "ok": 3.14,
        },
        "tags": [float("nan"), None, 1, "nan"],
    }


def test_event_store_persists_rfc8259(tmp_path, non_finite_event):
    from database.event_store import EventStore

    store = EventStore(db_path=str(tmp_path / "events.db"))
    store.save_event(non_finite_event)

    rows = store.get_recent_events(limit=10)
    assert len(rows) == 1
    saved = rows[0]

    ca = saved["ml_features"]["cross_asset"]
    assert ca["us2y_yield"] is None
    assert ca["us10y_change_1d"] is None
    assert ca["gold_change_1d"] is None
    assert ca["btc_dominance_change_7d"] == 0.0
    assert ca["vix_current"] == 15.16
    assert saved["nested"]["pos_inf"] is None
    assert saved["nested"]["neg_inf"] is None
    assert saved["nested"]["zero"] == 0.0
    assert saved["nested"]["ok"] == 3.14
    assert saved["tags"][0] is None
    assert saved["tags"][2] == 1
    assert saved["tags"][3] == "nan"  # string não é number


def test_event_store_db_has_no_non_finite_literal(tmp_path, non_finite_event):
    from database.event_store import EventStore

    store = EventStore(db_path=str(tmp_path / "events.db"))
    store.save_event(non_finite_event)

    # Valida o payload cru no SQLite com parser estrito
    conn = store._get_conn()
    payload_raw = conn.execute(
        "SELECT payload FROM events ORDER BY id DESC LIMIT 1"
    ).fetchone()[0]
    data = _strict_json_loads(payload_raw)
    assert data["ml_features"]["cross_asset"]["us2y_yield"] is None
    conn.close()


def test_event_saver_jsonl_persists_rfc8259(tmp_path, non_finite_event):
    import events.event_saver as es_mod
    from events.event_saver import EventSaver

    saver = EventSaver.__new__(EventSaver)
    saver.logger = logging.getLogger("tests.etapa6.jsonl")
    saver.write_jsonl = True
    saver.history_file = tmp_path / "events.jsonl"
    saver.max_jsonl_bytes = 2_000_000

    saver._save_to_jsonl(non_finite_event)

    line = saver.history_file.read_text(encoding="utf-8").splitlines()[0]
    data = _strict_json_loads(line)
    ca = data["ml_features"]["cross_asset"]
    assert ca["us2y_yield"] is None
    assert ca["us10y_change_1d"] is None
    assert ca["btc_dominance_change_7d"] == 0.0
    assert data["nested"]["pos_inf"] is None
    assert data["nested"]["ok"] == 3.14


def test_event_saver_jsonl_non_analysis_trigger_also_sanitized(tmp_path):
    import logging as _logging
    from events.event_saver import EventSaver

    saver = EventSaver.__new__(EventSaver)
    saver.logger = _logging.getLogger("tests.etapa6.jsonl.plain")
    saver.write_jsonl = True
    saver.history_file = tmp_path / "events.jsonl"

    plain_event = {
        "tipo_evento": "OUTRO_TIPO",
        "epoch_ms": 1786371600000,
        "symbol": "BTCUSDT",
        "valor": float("nan"),
        "zero": 0.0,
    }
    saver._save_to_jsonl(plain_event)
    data = _strict_json_loads(
        saver.history_file.read_text(encoding="utf-8").splitlines()[0]
    )
    assert data["valor"] is None
    assert data["zero"] == 0.0


def test_event_saver_fallback_json_sanitized(tmp_path, non_finite_event):
    from events.event_saver import EventSaver

    saver = EventSaver.__new__(EventSaver)
    saver.logger = logging.getLogger("tests.etapa6.fallback")

    saver._save_fallback(non_finite_event, "json")

    stamp = datetime.now().strftime("%Y%m%d")
    fallback_file = Path("fallback_events") / f"eventos_{stamp}.json"
    try:
        data = _strict_json_loads(fallback_file.read_text(encoding="utf-8"))
        assert data["ml_features"]["cross_asset"]["us2y_yield"] is None
        assert data["nested"]["neg_inf"] is None
    finally:
        fallback_file.unlink(missing_ok=True)


# ============================================================
# BUG 3 — AI PAYLOAD / GUARDRAIL (non-finite fora do LLM)
# ============================================================

def test_safe_price_non_finite_returns_none():
    from market_orchestrator.ai.payload_builder_compact import _safe_price

    assert _safe_price({"VIX": {"preco_atual": float("nan")}}, "VIX") is None
    assert _safe_price({"VIX": {"preco_atual": float("inf")}}, "VIX") is None
    assert _safe_price({"VIX": {"preco_atual": float("-inf")}}, "VIX") is None
    assert _safe_price({"DXY": {"preco_atual": None}}, "DXY") is None
    assert _safe_price({"DXY": {"preco_atual": 0}}, "DXY") is None
    assert _safe_price({"DXY": {"preco_atual": 103.456}}, "DXY") == 103.46


def test_safe_round_non_finite_returns_none():
    from market_orchestrator.ai.payload_builder_compact import _safe_round

    assert _safe_round(float("nan")) is None
    assert _safe_round(float("inf")) is None
    assert _safe_round(float("-inf")) is None
    assert _safe_round(None) is None
    assert _safe_round(3.14159, 2) == 3.14
    assert _safe_round(65000.4) == 65000


def test_safe_int_non_finite_returns_none():
    from market_orchestrator.ai.payload_builder_compact import _safe_int

    assert _safe_int({"FG": {"preco_atual": float("nan")}}, "FG") is None
    assert _safe_int({"FG": {"preco_atual": float("inf")}}, "FG") is None
    assert _safe_int({"FG": {"preco_atual": None}}, "FG") is None
    assert _safe_int({"FG": {"preco_atual": 67}}, "FG") == 67


def test_static_context_omits_non_finite_corr_and_ext():
    from market_orchestrator.ai.payload_builder_compact import _build_static_context

    event = {
        "market_context": {"trading_session": "ny", "session_phase": "open"},
        "external_markets": {
            "DXY": {"preco_atual": float("nan")},
            "VIX": {"preco_atual": float("inf")},
            "TNX": {"preco_atual": 4.688},
        },
        "ml_features": {
            "cross_asset": {
                "btc_eth_corr_7d": float("nan"),
                "btc_dxy_corr_30d": float("-inf"),
            }
        },
    }
    ctx = _build_static_context(event)
    assert "dxy" not in ctx
    assert "vix" not in ctx
    assert "eth7" not in ctx
    assert "dxy30" not in ctx
    assert ctx["tnx"] == 4.69
    assert json_dumps_rfc8259(ctx)  # não lança e nada non-finite


def test_static_context_preserves_finite_corr():
    from market_orchestrator.ai.payload_builder_compact import _build_static_context

    event = {
        "market_context": {"trading_session": "ny", "session_phase": "open"},
        "external_markets": {"DXY": {"preco_atual": 103.456}},
        "ml_features": {
            "cross_asset": {
                "btc_eth_corr_7d": 0.8261,
                "btc_dxy_corr_30d": 0.0922,
            }
        },
    }
    ctx = _build_static_context(event)
    assert ctx["dxy"] == 103.46
    assert ctx["eth7"] == 0.8
    assert ctx["dxy30"] == 0.09


def test_guardrail_strips_non_finite_before_llm():
    from market_orchestrator.ai.llm_payload_guardrail import ensure_safe_llm_payload

    payload = {
        "symbol": "BTCUSDT",
        "epoch_ms": 1786371600000,
        "price": {"c": 64742.1},
        "ctx": {
            "dxy": float("nan"),
            "tnx": float("inf"),
            "us2y": float("-inf"),
            "vix": 15.16,
        },
    }
    out = ensure_safe_llm_payload(payload)
    assert out is not None

    # nenhum valor non-finite no payload final
    json.dumps(out, ensure_ascii=False, allow_nan=False)
    # parser estrito aceita o payload final
    data = _strict_json_loads(json.dumps(out, ensure_ascii=False))
    assert data["ctx"]["dxy"] is None
    assert data["ctx"]["tnx"] is None
    assert data["ctx"]["us2y"] is None
    assert data["ctx"]["vix"] == 15.16
    # finitos intocados
    assert data["price"]["c"] == 64742.1


# ============================================================
# E2E: evento com NaN -> persistência null -> payload IA sem literal
# ============================================================

def test_e2e_nan_event_null_in_storage_and_clean_ai_payload(tmp_path, non_finite_event):
    from database.event_store import EventStore
    from market_orchestrator.ai.llm_payload_guardrail import ensure_safe_llm_payload

    store = EventStore(db_path=str(tmp_path / "events_e2e.db"))
    store.save_event(non_finite_event)

    rows = store.get_recent_events(limit=10)
    assert len(rows) == 1
    saved = rows[0]
    # Persistência: campos NaN -> null
    ca = saved["ml_features"]["cross_asset"]
    assert ca["us2y_yield"] is None
    assert ca["us10y_change_1d"] is None

    # Payload IA construído a partir do evento persistido
    ai_payload = ensure_safe_llm_payload(saved)
    assert ai_payload is not None
    # nenhum valor non-finite no payload final (string "nan" é legítima)
    json.dumps(ai_payload, ensure_ascii=False, allow_nan=False)
    _strict_json_loads(json.dumps(ai_payload, ensure_ascii=False))
