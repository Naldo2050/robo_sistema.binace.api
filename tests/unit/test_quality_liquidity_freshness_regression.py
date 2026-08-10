# tests/unit/test_quality_liquidity_freshness_regression.py
# -*- coding: utf-8 -*-
"""
ETAPA 3 — Auditoria de qualidade de liquidez, freshness, cache e fallback.

Cobre o padrão de falha da ETAPA 2 no eixo de LIQUIDEZ e no eixo de
FONTE DO ORDERBOOK (stale/cache/fallback/emergency):

  - qual.get("liq", "NORMAL")       -> dado ausente promovido a "NORMAL"
  - _resolve_liquidity (cap or 1.0) -> categoria desconhecida -> cap 1.0
  - orderbook_quality: data_source "stale"/"fallback_rest"/"circuit_open"/
    "external"/"unknown" eram colapsados para "live" (market_orchestrator)
  - enricher: orderbook_quality ausente -> default "live" (sem penalidade)
  - a fonte do orderbook não chegava ao resumo de qualidade da IA

Ajuste final (pós-audit): mapeamento estritamente fail-closed no
market_orchestrator — SOMENTE match exato com "live" produz "live";
origem desconhecida/não-mapeada cai no else -> tier "unknown". Default do
enricher: "unknown" (rótulo honesto de ausência) com MESMO peso -1.5 do
tier degradado ("cache" mantém -1.5 quando é origem real).

Casos (FASE 8):
  A  — liquidez NORMAL conhecida -> sem penalização
  B  — liquidez ruim (REDUCED/LOW) -> issue + cap conforme contrato
  C  — liquidez ausente -> FAIL-CLOSED (nunca NORMAL/confiança plena)
  D  — dado live/fresh -> não marcado como stale
  E  — dado stale -> não promovido a live
  F  — is_stale ausente no compressor v3 -> código legado, não ativo
  G  — cache disponível mas não usado != cache usado
  H  — stale fallback disponível mas não usado != stale usado
  I  — fallback REST usado -> degradado (não live)
  I2 — fonte desconhecida/nova -> FAIL-CLOSED (nunca live, tier degradado)
  J  — emergency mode -> propagado como emergency
  K  — caminho ativo = build_compact_payload (ai_runner)
  L  — fonte degradada chega ao summary da IA (qual.src)
  M  — fonte live -> sem issue de freshness no summary
  N  — enricher: ausência de orderbook_quality -> default "unknown" com
       o MESMO peso -1.5 do tier degradado (sem diferença de score vs "cache")
"""
from __future__ import annotations

import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import market_orchestrator.market_orchestrator as mo
import market_orchestrator.ai.payload_builder_compact as bcp
from data_processing.data_handler import NY_TZ
from institutional.enricher import enrich_signal
from market_orchestrator.ai.payload_builder_compact import build_compact_payload
from market_orchestrator.ai.payload_sections.quality_summary import (
    build_quality_summary,
)

# ---------------------------------------------------------------------------
# HELPERS — RESUMO DE QUALIDADE
# ---------------------------------------------------------------------------


def _summary(qual: dict | None = None, ctx: dict | None = None) -> dict:
    return build_quality_summary({"qual": qual or {}, "ctx": ctx or {}})


def _event_base() -> dict:
    return {
        "tipo_evento": "ANALYSIS_TRIGGER",
        "epoch_ms": 1786142460000,
        "preco_fechamento": 64856.34,
        "fluxo_continuo": {},
        "orderbook_data": {},
        "ml_features": {},
        "multi_tf": {},
        "historical_vp": {},
        "derivatives": {},
        "market_context": {},
        "market_environment": {},
    }


def _event_full_healthy() -> dict:
    """Evento completo: latência GOOD + calendário com liquidez NORMAL."""
    ev = _event_base()
    ev["institutional_analytics"] = {
        "quality": {
            "latency": {
                "latency_ms": 300,
                "latency_category": "GOOD",
                "data_freshness": "REAL_TIME",
                "is_acceptable": True,
                "is_stale": False,
            },
            "calendar": {
                "expected_liquidity": "NORMAL",
                "is_us_holiday": 0,
            },
        }
    }
    return ev


# ---------------------------------------------------------------------------
# HELPERS — CAMINHO REAL DO ORDERBOOK (mo._enrich_signal)
# ---------------------------------------------------------------------------


class _FakeTimeManager:
    tz_utc = timezone.utc

    def from_timestamp_ms(self, epoch_ms: int, tz) -> datetime:
        return datetime.fromtimestamp(epoch_ms / 1000.0, tz=tz)

    def now_utc_iso(self, timespec: str = "seconds") -> str:
        return datetime.now(self.tz_utc).isoformat(timespec=timespec)


@dataclass
class _LevelsStub:
    last_event: Optional[Dict[str, Any]] = None

    def add_from_event(self, evt: Dict[str, Any]) -> None:
        self.last_event = evt


@dataclass
class _EventSaverStub:
    saved_events: List[Dict[str, Any]] = field(default_factory=list)

    def save_event(self, evt: Dict[str, Any]) -> None:
        self.saved_events.append(evt)


@dataclass
class _EventBusStub:
    published: List[Dict[str, Any]] = field(default_factory=list)

    def publish(self, topic: str, evt: Dict[str, Any]) -> None:
        self.published.append({"topic": topic, "event": evt})


@dataclass
class _FakeBot:
    symbol: str = "BTCUSDT"
    window_count: int = 5
    time_manager: Any = field(default_factory=_FakeTimeManager)
    ny_tz = NY_TZ
    institutional_analytics: Any = None
    orderbook_fetch_failures: int = 0
    volume_history: deque = field(default_factory=lambda: deque(maxlen=100))
    volatility_history: deque = field(default_factory=lambda: deque(maxlen=100))
    levels: Any = field(default_factory=_LevelsStub)
    event_bus: Any = field(default_factory=_EventBusStub)
    event_saver: Any = field(default_factory=_EventSaverStub)
    _sent_triggers: set = field(default_factory=set)
    _ai_pool_lock: threading.Lock = field(default_factory=threading.Lock)

    def _validate_flow_metrics(self, flow_metrics: Dict[str, Any], valid_window_data: List[Dict[str, Any]]) -> bool:
        return True

    def _build_institutional_event(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        return {"wrapped": signal.copy()}

    def _log_event(self, evt: Dict[str, Any]) -> None:
        pass


def _ob_event_with_source(source: str, *, has_cached: bool = False, has_stale: bool = False) -> dict:
    return {
        "is_valid": True,
        "orderbook_data": {
            "bid_depth_usd": 1000.0,
            "ask_depth_usd": 800.0,
            "imbalance": 0.2,
            "mid": 100.0,
            "spread": 0.5,
            "spread_percent": 0.005,
        },
        "spread_metrics": {
            "mid": 100.0,
            "spread": 0.5,
            "spread_percent": 0.005,
            "bid_depth_usd": 1000.0,
            "ask_depth_usd": 800.0,
        },
        "order_book_depth": {"L5": {"bids": 1000.0, "asks": 800.0, "imbalance": 0.2}},
        "spread_analysis": {"current_spread_bps": 50.0},
        "depth_metrics": {
            "bid_liquidity_top5": 1000.0,
            "ask_liquidity_top5": 800.0,
            "depth_imbalance": 0.2,
        },
        "market_impact_buy": {"100k": {"move_usd": 1.0, "bps": 10.0}},
        "market_impact_sell": {"100k": {"move_usd": 1.5, "bps": 15.0}},
        "data_quality": {
            "is_valid": True,
            "data_source": source,
            "age_seconds": 0.1,
        },
        "health_stats": {
            "has_cached_data": has_cached,
            "has_stale_data": has_stale,
            "cache_hits": 0,
            "stale_data_uses": 0,
            "cache_age_seconds": None if not has_cached else 0.2,
            "stale_age_seconds": None if not has_stale else 5.0,
        },
    }


def _enrich_signal_with_ob_source(source: str, *, has_cached: bool = False, has_stale: bool = False) -> dict:
    """Executa o caminho REAL de produção: mo._enrich_signal + event bus."""
    bot = _FakeBot()

    def _accept(sig: Dict[str, Any]) -> Dict[str, Any]:
        return {"validated": True}

    import market_orchestrator.market_orchestrator as _mo

    _orig_validator = None
    try:
        _orig_validator = _mo.validator.validate_and_clean
    except Exception:
        pass

    import market_orchestrator.market_orchestrator as _mo2
    monkey = pytest.MonkeyPatch()
    monkey.setattr(_mo2.validator, "validate_and_clean", _accept)
    monkey.setattr(_mo2, "adicionar_memoria_evento", lambda *a, **k: None)
    monkey.setattr(_mo2, "obter_memoria_eventos", lambda n=4: [])

    base = {
        "tipo_evento": "ABSORÇÃO",
        "resultado_da_batalha": "Demanda Forte",
        "descricao": "Evento de teste",
        "delta": 10.0,
        "volume_total": 100.0,
        "volume_compra": 70.0,
        "volume_venda": 30.0,
        "ativo": "BTCUSDT",
    }

    try:
        _mo2.EnhancedMarketBot._enrich_signal(
            bot,
            base,
            {"dummy": True},
            {"dummy_flow": True},
            total_buy_volume=70.0,
            total_sell_volume=30.0,
            macro_context={"market_context": {}, "market_environment": {}},
            close_ms=1_700_000_000_000,
            ml_payload={"price_features": {}, "volume_features": {}, "microstructure": {}},
            enriched_snapshot={"ohlc": {"close": 100.0}, "volume_total": 100.0, "delta_fechamento": 10.0},
            contextual_snapshot={},
            ob_event=_ob_event_with_source(source, has_cached=has_cached, has_stale=has_stale),
            valid_window_data=[{"p": 100.0, "q": 1.0, "T": 1234567890}],
            support_resistance={},
            defense_zones_data={},
        )
    finally:
        monkey.undo()
        if _orig_validator is not None:
            try:
                _mo.validator.validate_and_clean = _orig_validator
            except Exception:
                pass

    assert bot.event_bus.published, "nenhum evento publicado"
    return bot.event_bus.published[0]["event"]


# ---------------------------------------------------------------------------
# CASO A — LIQUIDEZ NORMAL CONHECIDA
# ---------------------------------------------------------------------------


def test_caso_a_normal_liquidity_not_penalized():
    s = _summary({"lat": "GOOD", "liq": "NORMAL"})
    assert s["reliable"] is True
    assert s["confidence_cap"] == 1.0
    assert s["issues"] == []


# ---------------------------------------------------------------------------
# CASO B — LIQUIDEZ RUIM (REDUCED canônico do produtor)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("liq,cap", [("REDUCED", 0.8), ("LOW", 0.7), ("VERY_LOW", 0.5)])
def test_caso_b_bad_liquidity_propagates_issue_and_cap(liq, cap):
    s = _summary({"lat": "GOOD", "liq": liq})
    assert s["confidence_cap"] == cap
    assert s["reliable"] is False
    assert any("liquidez" in i.lower() for i in s["issues"])


# ---------------------------------------------------------------------------
# CASO C — LIQUIDEZ AUSENTE -> FAIL-CLOSED
# ---------------------------------------------------------------------------


def test_caso_c_missing_liquidity_is_fail_closed():
    s = _summary({"lat": "GOOD"})
    assert s["confidence_cap"] < 1.0
    assert s["reliable"] is False
    assert any("liquidez desconhecida" in i.lower() for i in s["issues"])


def test_caso_c_missing_qual_flags_both_axes():
    s = _summary({})
    assert any("latência desconhecida" in i.lower() for i in s["issues"])
    assert any("liquidez desconhecida" in i.lower() for i in s["issues"])


def test_caso_c_builder_emits_explicit_normal_liquidity():
    # Reseta cache de ctx estático (padrão do repo — ver
    # test_latency_reliability_regression.py CASO D): outro teste pode
    # ter deixado _last_static_ctx preenchido e tornar o payload "cached".
    bcp._last_static_ctx = {}
    bcp._last_static_ts = 0
    payload = build_compact_payload(_event_full_healthy())
    assert payload["qual"]["liq"] == "NORMAL"
    assert payload["summary"]["quality"]["reliable"] is True
    assert payload["summary"]["quality"]["confidence_cap"] == 1.0


# ---------------------------------------------------------------------------
# CASO D — LIVE NÃO É STALE; G/H — DISPONIBILIDADE ≠ USO
# ---------------------------------------------------------------------------


def test_caso_d_live_source_keeps_live_quality_even_with_available_slots():
    evt = _enrich_signal_with_ob_source("live", has_cached=True, has_stale=True)
    assert evt["orderbook_quality"] == "live"
    assert evt["orderbook_data"]["data_source"] == "live"


# ---------------------------------------------------------------------------
# CASO E — STALE NÃO PODE SER PROMOVIDO A LIVE
# ---------------------------------------------------------------------------


def test_caso_e_stale_source_is_not_live():
    evt = _enrich_signal_with_ob_source("stale", has_stale=True)
    assert evt["orderbook_quality"] != "live"


# ---------------------------------------------------------------------------
# CASO F — COMPRESSOR v3 É LEGADO; CAMINHO ATIVO É build_compact_payload
# ---------------------------------------------------------------------------


def test_caso_k_active_payload_path_is_build_compact_payload():
    from market_orchestrator.ai.ai_runner import AIRunner

    runner = AIRunner.create()
    assert runner._payload_builder is build_compact_payload


def test_caso_f_compressor_v3_not_imported_by_production():
    # Prova em subprocesso com interpretador limpo: sys.modules é global e
    # qualquer outro teste pode importar o v3 (poluição de ordem).
    import subprocess
    import sys as _sys

    root = str(Path(__file__).resolve().parents[2])
    code = (
        "import market_orchestrator.ai.ai_runner; import sys; "
        "print('market_orchestrator.ai.payload_compressor_v3' in sys.modules)"
    )
    out = subprocess.run(
        [_sys.executable, "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


# ---------------------------------------------------------------------------
# CASO I — FALLBACK REST / CIRCUIT OPEN -> DEGRADADO (não live)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source", ["fallback_rest", "circuit_open", "unknown", "external"])
def test_caso_i_non_live_sources_are_not_labeled_live(source):
    evt = _enrich_signal_with_ob_source(source)
    assert evt["orderbook_quality"] != "live"


def test_caso_i2_unmapped_new_source_is_fail_closed():
    # Fonte desconhecida/futura: NÃO pode virar "live" (fail-closed estrito
    # do mapeamento: só match exato com "live" produz "live"). Deve cair na
    # tier degradada padrão E preservar a origem real no payload da IA.
    src = "weird_new_source_v2"
    evt = _enrich_signal_with_ob_source(src)
    assert evt["orderbook_quality"] != "live"
    assert evt["orderbook_quality"] == "unknown"
    assert evt["orderbook_data"]["data_source"] == src
    bcp._last_static_ctx = {}
    bcp._last_static_ts = 0
    payload = build_compact_payload(evt)
    assert payload["qual"].get("src") == src
    q = payload["summary"]["quality"]
    assert q["reliable"] is False
    assert q["confidence_cap"] <= 0.8


# ---------------------------------------------------------------------------
# CASO J — EMERGENCY
# ---------------------------------------------------------------------------


def test_caso_j_emergency_source_maps_to_emergency():
    evt = _enrich_signal_with_ob_source("emergency")
    assert evt["orderbook_quality"] == "emergency"


# ---------------------------------------------------------------------------
# SCORES — orderbook ausente não pode render reliability plena
# ---------------------------------------------------------------------------


def test_caso_k_missing_orderbook_quality_penalizes_reliability():
    ev = _event_base()
    ev.pop("orderbook_data", None)
    enrich_signal(ev)
    assert ev["reliability_score"] < 10.0


def test_caso_n_enricher_default_unknown_has_same_weight_as_cache():
    # Ausência total de orderbook_quality -> default "unknown" com o MESMO
    # peso -1.5 do tier degradado: score idêntico ao de "cache" (origem
    # real) e ao de "unknown" explícito — diferença apenas de rótulo.
    ev_cache = _event_base()
    ev_cache["orderbook_quality"] = "cache"
    enrich_signal(ev_cache)
    ev_unknown = _event_base()
    ev_unknown["orderbook_quality"] = "unknown"
    enrich_signal(ev_unknown)
    ev_missing = _event_base()
    enrich_signal(ev_missing)
    assert ev_cache["reliability_score"] == 8.5
    assert ev_unknown["reliability_score"] == 8.5
    assert ev_missing["reliability_score"] == 8.5


def test_caso_k2_live_orderbook_quality_keeps_full_reliability():
    ev = _event_base()
    ev["orderbook_quality"] = "live"
    enrich_signal(ev)
    assert ev["reliability_score"] == 10.0


# ---------------------------------------------------------------------------
# CASO L — FONTE DEGRADADA CHEGA AO SUMMARY DA IA
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src,cap_expected,keyword",
    [
        ("stale", 0.8, "stale"),
        ("fallback_rest", 0.8, "fallback"),
        ("circuit_open", 0.8, "circuit"),
        ("cache", 0.9, "cache"),
        ("emergency", 0.5, "indisponível"),
        ("external", 0.8, "externa"),
        ("unknown", 0.8, "desconhecida"),
    ],
)
def test_caso_l_degraded_source_reaches_ai_summary(src, cap_expected, keyword):
    s = _summary({"lat": "GOOD", "liq": "NORMAL", "src": src})
    assert s["reliable"] is False
    assert s["confidence_cap"] <= cap_expected
    assert any(keyword in i.lower() for i in s["issues"])


def test_caso_l_builder_propagates_stale_source_into_qual():
    bcp._last_static_ctx = {}
    bcp._last_static_ts = 0
    ev = _event_full_healthy()
    ev["orderbook_data"] = {"data_source": "stale", "bid_depth_usd": 1000.0}
    payload = build_compact_payload(ev)
    assert payload["qual"].get("src") == "stale"
    q = payload["summary"]["quality"]
    assert q["reliable"] is False
    assert any("stale" in i.lower() for i in q["issues"])


# ---------------------------------------------------------------------------
# CASO M — FONTE LIVE: SEM ISSUE DE FRESHNESS
# ---------------------------------------------------------------------------


def test_caso_m_live_source_no_freshness_issue():
    s = _summary({"lat": "GOOD", "liq": "NORMAL", "src": "live"})
    assert s["reliable"] is True
    assert s["confidence_cap"] == 1.0
    assert not any("stale" in i.lower() or "fallback" in i.lower() for i in s["issues"])
