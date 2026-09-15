# institutional_enricher.py
# -*- coding: utf-8 -*-
"""
Enriquecedor Institucional de Sinais - Onda 1 + Onda 2

Campos implementados (sem API paga):
ONDA 1 (impacto máximo, baixo esforço):
  - metadata: sequence_id, exchange_timestamp, data_quality_score, completeness_pct,
               reliability_score, primary_exchange, data_feed_type
  - price_data: bid/ask explícitos, tick_direction, twap, preços anteriores (1h/4h/1d/1w)
  - support_resistance: pivot_points (classic + camarilla), immediate_support/resistance arrays
  - spread_analysis: spread_volatility
  - market_context: is_holiday
  - market_impact: slippage 1k/10k_usd
  - derivatives: funding_rate BTC (complementa ETH que já existia)

ONDA 2 (impacto alto, médio esforço):
  - volume_profile: value_area_volume_pct, single_prints, hvn_nodes[], lvn_nodes[]
  - alerts: sistema de alertas estruturado (SUPPORT_TEST, VOLUME_SPIKE, DIVERGENCE, etc.)
  - technical_indicators: stochastic, williams_r, cci (quando não vêm do TA institucional)
  - order_flow: passive_buy_pct, passive_sell_pct explícitos
  - whale_activity: large_orders_1h (filtro de trades grandes), iceberg_activity
  - volatility_metrics: realized_vol_24h, realized_vol_7d, volatility_percentile
  - defense_zones: no_mans_land zone calculada
  - fibonacci_levels: campo padronizado na raiz do evento
"""

from __future__ import annotations

import logging
import time
import math
from collections import deque
from datetime import datetime, timezone, date
from typing import Any, Dict, List, Optional, Tuple

from fetchers.onchain_fetcher import SUPPORTED_ONCHAIN_FIELDS, USABLE_ONCHAIN_STATES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Holidays (mercados tradicionais US - impactam liquidez BTC em certos dias)
# ---------------------------------------------------------------------------
_US_HOLIDAYS_2025_2026 = {
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17), date(2025, 5, 26),
    date(2025, 6, 19), date(2025, 7, 4), date(2025, 9, 1), date(2025, 11, 27),
    date(2025, 12, 25),
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 5, 25),
    date(2026, 6, 19), date(2026, 7, 4), date(2026, 9, 7), date(2026, 11, 26),
    date(2026, 12, 25),
}

try:
    import holidays as _holidays_lib
    _HOLIDAYS_LIB_OK = True
except ImportError:
    _HOLIDAYS_LIB_OK = False


def _is_holiday(dt: datetime) -> bool:
    """Verifica se a data é feriado em mercados tradicionais (US)."""
    d = dt.date() if hasattr(dt, "date") else dt
    if _HOLIDAYS_LIB_OK:
        try:
            us_h = _holidays_lib.US(years=d.year)
            return d in us_h
        except Exception:
            pass
    return d in _US_HOLIDAYS_2025_2026


# ---------------------------------------------------------------------------
# Buffer de estado — mantido em memória durante a sessão
# ---------------------------------------------------------------------------
class _SessionState:
    """Estado de sessão para cálculos que precisam de histórico."""

    def __init__(self, maxlen: int = 500):
        # Preços (close) e seus timestamps — para períodos anteriores
        self.price_history: deque = deque(maxlen=maxlen)       # (epoch_ms, price)
        # Spreads históricos — para volatilidade do spread
        self.spread_history: deque = deque(maxlen=300)          # spread_bps values
        # Volumes horários — para realized_vol rolling
        self.vol_history: deque = deque(maxlen=maxlen)          # (epoch_ms, realized_vol_1h)
        # OHLC por janela — para stochastic/williams_r reais (mínimo 14 candles)
        self.ohlc_history: deque = deque(maxlen=200)            # dicts {open, high, low, close}
        # Contador de sequência
        self.sequence_counter: int = 0
        # Trades grandes detectados (para large_orders_1h)
        self.large_trades: deque = deque(maxlen=200)            # dicts
        # Histórico de alerts para evitar repetição
        self.last_alert_ts: Dict[str, float] = {}


_STATE = _SessionState()

# Threshold para "ordem grande" (em BTC) — ajustável
_LARGE_ORDER_THRESHOLD_BTC = 1.0
# Cooldown de alertas (segundos) — mesmo tipo não dispara de novo antes disso
_ALERT_COOLDOWN_SEC = 120


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def _safe_div(a, b, default=0.0):
    try:
        if b == 0:
            return default
        return a / b
    except Exception:
        return default


def _get_nested(d: dict, *keys, default=None):
    """Acesso seguro a dicts aninhados."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is None:
            return default
    return cur


def _count_present_fields(event: dict, required_fields: list) -> float:
    """Retorna % de campos presentes no evento (0-100)."""
    if not required_fields:
        return 100.0
    present = sum(1 for f in required_fields if _get_nested(event, f) is not None)
    return round(present / len(required_fields) * 100, 1)


# ---------------------------------------------------------------------------
# P05: presença != validade. completeness = % ponderado de features UTILIZÁVEIS.
# ---------------------------------------------------------------------------
# Contrato:
#   PRESENCE  = container/chave existe (is not None).
#   VALIDITY  = dado existe E é semanticamente utilizável (regras por feature).
#   FRESHNESS = idade dentro do TTL (status fresh/stale/unavailable/warming_up).
#   SOURCE HEALTH = fonte respondeu (P04 field_status VALID/REAL_ZERO vs
#                 MISSING/API_ERROR).
#   FALLBACK/WARMUP = cache/fallback ou cobertura parcial (não equivale a FULL).
# Pesos preservam a escala histórica (soma 10; REQUIRED w2, CONDITIONAL w1,
# OPTIONAL w0 — rastreado sem penalizar). Container vazio ({}), 0 inválido,
# None, STALE/is_valid=0/is_available=0, WARMING_UP sem FULL => não utilizável.
_USABILITY_WEIGHTS = {
    "price": 2, "flow_1m": 2, "orderbook": 2,
    "flow_5m": 1, "flow_15m": 1, "session_vwap": 1, "onchain": 1,
    "positioning": 0, "options": 0,
}
_USABILITY_TOTAL_WEIGHT = 10


def _is_usable_number(v) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v))


def _flow_window_usable(event: dict, window_min: int) -> tuple:
    """(usable: bool, reason: str|None) para flow_{1,5,15}m."""
    key = f"{window_min}m"
    integrity = _get_nested(event, "fluxo_continuo", "flow_window_integrity") or {}
    info = integrity.get(key)
    if isinstance(info, dict):
        if info.get("status") == "FULL" or info.get("is_temporal_coverage_valid"):
            return True, None
        return False, f"flow_{window_min}m_{str(info.get('status', 'unknown')).lower()}"
    # Legado (sem bloco de integridade): presença de volumes da janela.
    of = _get_nested(event, "fluxo_continuo", "order_flow") or {}
    if window_min == min(1, 5, 15):
        if of.get("buy_volume_btc") is not None and of.get("sell_volume_btc") is not None:
            return True, None
        return False, "flow_1m_missing_volumes"
    if of.get(f"net_flow_{window_min}m") is not None:
        return True, None
    return False, f"flow_{window_min}m_missing"


def _score_usability(event: dict):
    """Retorna (components: dict, reasons: list[str], earned: float).

    components[name] = {"weight": w, "usable": bool, "required": bool}.
    Monotônico: tornar válido→missing nunca aumenta; inválido→válido nunca reduz.
    REAL_ZERO conta como válido (regra por validade, nunca `if value`).
    """
    components: dict = {}
    reasons: list = []

    def _put(name: str, weight: int, usable: bool, reason=None, required=False,
             inform=False):
        components[name] = {"weight": weight, "usable": bool(usable),
                            "required": bool(required)}
        if reason and (not usable or inform):
            reasons.append(reason)

    # price (REQUIRED): número finito > 0 (mesma cadeia do close efetivo).
    _px = _get_nested(event, "preco_fechamento")
    if _px is None:
        _px = _get_nested(event, "contextual_snapshot", "ohlc", "close")
    _put("price", 2, _is_usable_number(_px) and _px > 0,
         None if (_is_usable_number(_px) and _px > 0) else "price_missing_or_invalid",
         required=True)

    # flow 1m/5m/15m.
    _u1, _r1 = _flow_window_usable(event, 1)
    _put("flow_1m", 2, _u1, _r1, required=True)
    _u5, _r5 = _flow_window_usable(event, 5)
    _put("flow_5m", 1, _u5, _r5)
    _u15, _r15 = _flow_window_usable(event, 15)
    _put("flow_15m", 1, _u15, _r15)

    # orderbook (REQUIRED): is_valid + profundidades + fonte não-emergency.
    ob = event.get("orderbook_data") or {}
    if "is_valid" in ob:
        _ob_ok = bool(ob.get("is_valid"))
        _ob_reason = None if _ob_ok else "orderbook_invalid"
    else:
        _ob_ok = ob.get("mid") is not None  # legado: sem flag, usa presença do mid
        _ob_reason = None if _ob_ok else "orderbook_missing"
    if _ob_ok:
        _bid = ob.get("bid_depth_usd")
        _ask = ob.get("ask_depth_usd")
        if ((_bid is not None or _ask is not None)
                and not ((_bid or 0) > 0 and (_ask or 0) > 0)):
            _ob_ok, _ob_reason = False, "orderbook_zero_depth"
    if _ob_ok:
        _src = ob.get("data_source") or ob.get("source") or ""
        if str(_src).lower() == "emergency":
            _ob_ok, _ob_reason = False, "orderbook_emergency_source"
    _put("orderbook", 2, _ob_ok, _ob_reason, required=True)

    # session_vwap (CONDITIONAL): is_valid + status fora de STALE/ERROR.
    sv = _get_nested(event, "institutional_analytics", "session_vwap") or {}
    if "is_valid" in sv or "status" in sv:
        _sv_ok = (bool(sv.get("is_valid"))
                  and str(sv.get("status", "")).upper() not in ("STALE", "ERROR", "INVALID"))
        _sv_reason = None if _sv_ok else "session_vwap_stale_or_invalid"
    elif sv:
        _sv_ok, _sv_reason = True, None  # legado: seção presente sem flags
    else:
        _sv_ok, _sv_reason = False, "session_vwap_missing"
    _put("session_vwap", 1, _sv_ok, _sv_reason)

    # onchain (CONDITIONAL, consome P04): algum campo VALID/REAL_ZERO.
    _aa = _get_nested(event, "raw_event", "advanced_analysis") or {}
    _fs = _aa.get("onchain_field_status")
    if isinstance(_fs, dict) and _fs:
        _valid_fields = [k for k, s in _fs.items() if s in ("VALID", "REAL_ZERO")]
        _oc_ok = bool(_valid_fields)
        if _oc_ok:
            _n_bad = sum(1 for s in _fs.values() if s in ("MISSING", "API_ERROR"))
            _oc_reason = f"onchain_partial({len(_valid_fields)}/{len(_fs)} valid)" if _n_bad else None
        else:
            _oc_reason = "onchain_unavailable"
    else:
        # Legado (pré-P04, sem field_status): status fresh + ≥1 valor real.
        # Limitação documentada: 0 fabricado pré-P04 é indistinguível aqui;
        # o replay de auditoria usa external_apis.jsonl (UNKNOWN se ambíguo).
        _ocm = _aa.get("onchain_metrics") or {}
        _real_vals = [k for k in ("mempool_size", "mempool_vsize_mb",
                                  "fees_fastest_sat_vb", "fees_half_hour_sat_vb",
                                  "fees_hour_sat_vb", "difficulty", "active_addresses")
                      if _ocm.get(k) is not None]
        _oc_ok = bool(_real_vals) and _aa.get("onchain_status") == "fresh"
        _oc_reason = None if _oc_ok else "onchain_unavailable_or_unverifiable"
    _put("onchain", 1, _oc_ok, _oc_reason, inform=True)

    # positioning (OPTIONAL w0): rastreado, nunca penaliza.
    _pos = _get_nested(event, "institutional_analytics", "positioning") or {}
    _pos_ok = bool(_pos.get("is_available"))
    _put("positioning", 0, _pos_ok,
         None if _pos_ok else "positioning_unavailable")

    # options (OPTIONAL w0): rastreado, nunca penaliza.
    _has_opt = bool(_get_nested(event, "data_reliability", "has_options_data"))
    if not _has_opt:
        _has_opt = bool((_aa.get("options_metrics") or {}).get("is_real_data"))
    _put("options", 0, _has_opt,
         None if _has_opt else "options_unavailable")

    earned = sum(c["weight"] for c in components.values() if c["usable"])
    return components, reasons, earned


# ---------------------------------------------------------------------------
# PFIX-LOW: coverage onchain deriva de field_status (P04), nunca de freshness.
# ---------------------------------------------------------------------------
# Contrato:
#   fonte exclusiva: advanced_analysis.onchain_field_status (quando presente).
#   usable = status in USABLE_ONCHAIN_STATES; denominador = SUPPORTED_ONCHAIN_FIELDS.
#   labels em minúsculas (estilo existente de data_reliability).
#   freshness (onchain_status/age) é ORTOGONAL: não entra aqui, não é alterada aqui.
def _derive_onchain_coverage(field_status) -> Tuple[str, Optional[float], int, int]:
    """Retorna (label, pct|None, usable_count, total_supported).

    field_status ausente/vazio => ("unknown", None, 0, total).
    Itera sobre SUPPORTED_ONCHAIN_FIELDS (não sobre as chaves recebidas):
    chaves paid/metadata extras nunca alteram o denominador.
    """
    total = len(SUPPORTED_ONCHAIN_FIELDS)
    if not isinstance(field_status, dict) or not field_status:
        return "unknown", None, 0, total
    usable = sum(1 for f in SUPPORTED_ONCHAIN_FIELDS
                 if field_status.get(f) in USABLE_ONCHAIN_STATES)
    pct = round(usable / total * 100, 1) if total else 0.0
    if usable >= total:
        label = "full"
    elif usable > 0:
        label = "partial"
    else:
        label = "none"
    return label, pct, usable, total


# ---------------------------------------------------------------------------
# ONDA 1 — Campos de Metadados e Qualidade
# ---------------------------------------------------------------------------

def _extract_exchange_timestamp(event: dict, valid_window_data: Optional[List[Dict]] = None) -> Optional[str]:
    """
    Extrai o timestamp real da exchange (Binance) a partir do campo 'T'
    do último trade da janela, ou do campo 'E' do evento WebSocket.
    Retorna ISO-8601 UTC string.
    """
    # Tentar do último trade da janela (campo 'T' = trade timestamp da Binance)
    if valid_window_data:
        for trade in reversed(valid_window_data):
            t_ms = trade.get("T") or trade.get("E")
            if t_ms and isinstance(t_ms, (int, float)) and t_ms > 0:
                try:
                    dt = datetime.fromtimestamp(int(t_ms) / 1000, tz=timezone.utc)
                    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")
                except Exception:
                    pass

    # Fallback: open_time/close_time do contextual_snapshot
    snap = event.get("contextual_snapshot", {}) or {}
    ohlc = snap.get("ohlc", {}) or {}
    close_time = ohlc.get("close_time")
    if close_time and isinstance(close_time, (int, float)) and close_time > 0:
        try:
            dt = datetime.fromtimestamp(int(close_time) / 1000, tz=timezone.utc)
            return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        except Exception:
            pass

    return None


def _estimate_cross_exchange_variance(event: dict) -> Optional[float]:
    """
    Estima a variância de preço cross-exchange usando dados disponíveis.

    Sem APIs de outras exchanges, usamos a divergência entre:
    - orderbook mid price vs VWAP vs close price
    como proxy da dispersão de preço entre venues.

    Em mercados eficientes (BTC spot), a variância real entre
    Binance/Coinbase/Kraken fica tipicamente entre 0.01-0.10%.
    Esta estimativa fornece um upper-bound razoável.
    """
    ob = event.get("orderbook_data", {}) or {}
    snap = event.get("contextual_snapshot", {}) or {}
    ohlc = snap.get("ohlc", {}) or {}

    mid = ob.get("mid", 0) or 0
    vwap = ohlc.get("vwap", 0) or 0
    close = event.get("preco_fechamento", 0) or ohlc.get("close", 0) or 0

    prices = [p for p in (mid, vwap, close) if p > 0]
    if len(prices) < 2:
        return None

    avg = sum(prices) / len(prices)
    if avg <= 0:
        return None

    # Max deviation como proxy de cross-exchange variance
    max_dev = max(abs(p - avg) for p in prices)
    variance_pct = round(max_dev / avg * 100, 4)

    # Clamp: em BTC líquido, variance > 0.5% seria anomalia
    return min(variance_pct, 0.50)


def _build_metadata_fields(
    event: dict,
    epoch_ms: int,
    valid_window_data: Optional[List[Dict]] = None,
) -> dict:
    """
    Retorna campos de metadata faltantes:
    sequence_id, exchange_timestamp, data_quality_score, completeness_pct,
    reliability_score, primary_exchange, backup_exchanges,
    data_feed_type, cross_exchange_variance_pct
    """
    _STATE.sequence_counter += 1

    # P05: completeness = % ponderado de features UTILIZÁVEIS (presence != validity).
    # Substitui a contagem de containers (dict {} contava como presente).
    # Pesos: soma 10 (escala histórica preservada em espírito; ver _USABILITY_WEIGHTS).
    _components, _reasons, _earned = _score_usability(event)
    completeness = round(_earned / _USABILITY_TOTAL_WEIGHT * 100, 1)

    # Reliability: operacional (pipeline health) + penalidade documentada de
    # validade quando a seção existe mas está inválida/stale. Ausência de seção
    # NÃO penaliza reliability (é papel da completeness).
    reliability = 10.0
    # FIX (ETAPA 3): default fail-closed com rótulo honesto. Antes
    # `event.get("orderbook_quality", "live")` promovia ausência de
    # informação de qualidade do orderbook a "live" (sem penalidade).
    # "cache" foi o primeiro default escolhido, mas tem conotação de
    # origem real; "unknown" comunica corretamente AUSÊNCIA de
    # informação. Mesma penalidade do tier degradado (peso -1.5, sem
    # peso novo): ausência não vira KNOWN_GOOD nem falsifica provenance
    # (a IA só recebe qual["src"] quando a origem é conhecida).
    ob_quality = event.get("orderbook_quality", "unknown")
    if ob_quality == "emergency":
        reliability -= 3.0
    elif ob_quality in ("cache", "unknown"):
        reliability -= 1.5

    # FIX (ETAPA 2): penalidade derivada dos flags canônicos de latência
    # (time_manager.track_data_latency: is_acceptable = ms < 5000,
    # is_stale = ms > 15000). Antes usava thresholds próprios (8000/15000)
    # que divergiam do canônico — latência POOR/inaceitável (5000-8000ms)
    # renderia reliability_score=10.0 (JANELA 1). Mesmos pesos (0.5/1.0),
    # mesma fonte de verdade, sem inventar thresholds novos.
    latency_info = _get_nested(event, "institutional_analytics", "quality", "latency") or {}
    if latency_info:
        if latency_info.get("is_stale"):
            reliability -= 1.0
        elif not latency_info.get("is_acceptable"):
            reliability -= 0.5

    anomaly_sev = _get_nested(event, "institutional_analytics", "quality", "anomalies", "max_severity", default="NONE")
    if anomaly_sev == "HIGH":
        reliability -= 2.0
    elif anomaly_sev == "MEDIUM":
        reliability -= 0.5

    # P05: session_vwap presente-mas-inválida/stale reduz reliability (-0.5).
    # Ausente => sem penalidade aqui (completeness já debita). Peso preservado
    # na ordem das penalidades operacionais existentes; sem thresholds novos.
    _sv_rel = _get_nested(event, "institutional_analytics", "session_vwap") or {}
    if "is_valid" in _sv_rel or "status" in _sv_rel:
        if (not _sv_rel.get("is_valid")
                or str(_sv_rel.get("status", "")).upper() in ("STALE", "ERROR", "INVALID")):
            reliability -= 0.5
            _reasons.append("reliability:session_vwap_stale_or_invalid")

    reliability = max(0.0, min(10.0, reliability))

    # Data quality score: combinação de completeness + reliability
    data_quality_score = round((completeness / 100 * 5) + (reliability / 10 * 5), 2)

    # Exchange timestamp (real da Binance)
    exchange_timestamp = _extract_exchange_timestamp(event, valid_window_data)

    # Cross-exchange variance (estimativa sem API externa)
    cross_exchange_variance = _estimate_cross_exchange_variance(event)

    result = {
        "sequence_id": _STATE.sequence_counter,
        "primary_exchange": "BINANCE",
        "backup_exchanges": ["COINBASE", "KRAKEN", "OKX"],
        "data_feed_type": "WEBSOCKET_L2",
        "data_quality_score": data_quality_score,
        "completeness_pct": completeness,
        "reliability_score": round(reliability, 2),
        # P05: componentes explícitos — explicam o score (seção 11).
        "quality_components": _components,
        "quality_reasons": list(_reasons),
    }

    if exchange_timestamp:
        result["exchange_timestamp"] = exchange_timestamp

    if cross_exchange_variance is not None:
        result["cross_exchange_variance_pct"] = cross_exchange_variance

    return result


# ---------------------------------------------------------------------------
# ONDA 1 — Preços Anteriores e Bid/Ask explícitos
# ---------------------------------------------------------------------------

def _update_price_history(epoch_ms: int, price: float) -> None:
    """Adiciona preço atual ao histórico."""
    _STATE.price_history.append((epoch_ms, price))


def _get_price_n_hours_ago(hours: float, current_epoch_ms: int) -> Optional[float]:
    """Busca preço mais próximo de N horas atrás no histórico."""
    target_ms = current_epoch_ms - int(hours * 3600 * 1000)
    best = None
    best_diff = float("inf")
    for ep, pr in _STATE.price_history:
        diff = abs(ep - target_ms)
        if diff < best_diff:
            best_diff = diff
            best = pr
    # Só retorna se existir dado razoavelmente próximo (±30min)
    if best_diff < 1800 * 1000:
        return best
    return None


def _build_price_fields(event: dict, current_price: float, epoch_ms: int) -> dict:
    """
    Campos de preço faltantes:
    - bid/ask explícitos
    - tick_direction
    - twap (aproximado via vwap histórico)
    - previous_periods: 1h_ago, 4h_ago, 1d_ago, 1w_ago
    """
    result = {}

    # Bid/Ask explícitos (mid ± spread/2)
    ob = event.get("orderbook_data", {}) or {}
    mid = ob.get("mid", current_price)
    spread_raw = ob.get("spread", 0.1)  # em USD
    result["bid"] = round(mid - spread_raw / 2, 2)
    result["ask"] = round(mid + spread_raw / 2, 2)

    # Tick direction (+1 subiu, -1 desceu, 0 neutro)
    prev_price = _get_price_n_hours_ago(1 / 60, epoch_ms)  # ~1 minuto atrás
    if prev_price is not None and prev_price != 0:
        result["tick_direction"] = 1 if current_price > prev_price else (-1 if current_price < prev_price else 0)
    else:
        result["tick_direction"] = 0

    # TWAP: usa histórico de preços dos últimos 30 períodos (5min × 30 = 2.5h)
    recent_prices = [pr for _, pr in list(_STATE.price_history)[-30:] if pr > 0]
    if recent_prices:
        twap = round(sum(recent_prices) / len(recent_prices), 2)
        result["twap"] = twap
    else:
        snap = event.get("contextual_snapshot", {}) or {}
        result["twap"] = snap.get("ohlc", {}).get("vwap", current_price)

    # Preços de períodos anteriores
    previous_periods = {}
    for label, hours in [("1h_ago", 1), ("4h_ago", 4), ("1d_ago", 24), ("1w_ago", 168)]:
        p = _get_price_n_hours_ago(hours, epoch_ms)
        if p is not None:
            previous_periods[label] = round(p, 2)
    if previous_periods:
        result["previous_periods"] = previous_periods

    return result


# ---------------------------------------------------------------------------
# ONDA 1 — Pivot Points e Suporte/Resistência Imediatos
# ---------------------------------------------------------------------------

def _is_vp_degenerate(vp: dict) -> bool:
    """Detecta VP degenerado onde poc == vah == val (dados insuficientes)."""
    poc = vp.get("poc", 0)
    vah = vp.get("vah", 0)
    val = vp.get("val", 0)
    if not (poc and vah and val):
        return True
    # Diferença < 0.01% indica degeneração
    if poc > 0 and abs(vah - val) / poc < 0.0001:
        return True
    return False


# Sources derivadas do mesmo snapshot REST L2 (sem independência estrutural).
# Espelha NON_STRUCTURAL_SOURCES de support_resistance/defense_zones.py sem
# importar o módulo (evita acoplamento; manter sincronizado).
_NON_STRUCTURAL_SOURCES = frozenset({
    "orderbook_bid_wall",
    "orderbook_ask_wall",
    "orderbook_cluster",
    "depth_asymmetry",
})


def _is_wall_only_zone(zone: object) -> bool:
    """Compat: wall-only clássico (ver _is_non_structural_zone)."""
    return _is_non_structural_zone(zone)


def _is_non_structural_zone(zone: object) -> bool:
    """Verdadeiro se a zona não tem fonte estrutural independente.

    Cobre wall-only (liquidity_only) e heurísticas projetadas puras
    (projected_only, ex.: depth_asymmetry isolado). Zonas sem flags nem
    sources (fixtures legados) retornam False — nunca bloquear por falta
    de metadados.
    """
    if not isinstance(zone, dict):
        return False
    if zone.get("liquidity_only") is True or zone.get("projected_only") is True:
        return True
    if zone.get("has_structural_confluence") is True:
        return False
    sources = zone.get("sources")
    if isinstance(sources, list) and sources:
        return all(s in _NON_STRUCTURAL_SOURCES for s in sources)
    return False


def _build_pivot_points(event: dict) -> dict:
    """
    Calcula pivot points clássicos (daily, weekly, monthly).
    - Usa VP histórico quando disponível e não degenerado
    - Para VP degenerado (poc=vah=val), usa OHLC do multi_tf do período correto
    - Cada TF usa seus próprios dados (não replica daily para weekly/monthly)
    Retorna também immediate_support/resistance arrays (sem duplicatas).
    """
    historical_vp = event.get("historical_vp", {}) or {}
    multi_tf = event.get("multi_tf", {}) or {}
    current_price = float(event.get("preco_fechamento", 0) or 0)
    # FIX pivot_points (auditoria 2026-08-09): pivots clássicos do período
    # anterior COMPLETO (calculados por context_collector._calculate_pivots
    # → daily_pivot iloc[-2], fix 75bd3ec) são a fonte PRIMÁRIA. O campo
    # "pivots" é injetado no sinal pelo market_orchestrator a partir de
    # macro_context["pivots"]. Fallback: VP intraday parcial (marcado).
    classic_pivots = event.get("pivots") or {}

    pivots = {}
    support_levels = []
    resistance_levels = []
    support_strength = []
    resistance_strength = []

    # Mapeamento: período VP → timeframe multi_tf para fallback OHLC
    tf_map = {
        "daily":   ("1d", 1.0),
        "weekly":  ("4h", 0.9),   # 4h dá visão semanal razoável
        "monthly": ("1d", 0.8),   # 1d dá visão mensal
    }

    for period, (tf_key, weight) in tf_map.items():
        vp = historical_vp.get(period, {}) or {}
        degenerate = _is_vp_degenerate(vp)

        h = l = c = 0.0
        source_tag = "vp_fallback"
        classic_levels: dict = {}

        # 1) Fonte PRIMÁRIA: pivots clássicos (período anterior completo).
        #    Validação: pivot > 0; reconstrução de H/L via inversão exata das
        #    fórmulas clássicas (s1 = 2p - h → h = 2p - s1 ; r1 = 2p - l →
        #    l = 2p - r1) para manter o formato vah/val/poc do campo.
        classic_data = classic_pivots.get(period) or {}
        classic_valid = (
            isinstance(classic_data, dict)
            and classic_data.get("pivot") is not None
            and float(classic_data.get("pivot") or 0) > 0
        )
        if classic_valid:
            p_c = float(classic_data["pivot"])
            # OHLC originais do período completo (propagados por daily_pivot,
            # fix pivot_points 2026-08-09) — sem drift de arredondamento.
            h = classic_data.get("high")
            l = classic_data.get("low")
            c = classic_data.get("close")
            if h is not None and l is not None and c is not None:
                h, l, c = float(h), float(l), float(c)
            else:
                # Compatibilidade: dados legados sem high/low/close —
                # reconstruir via inversão exata das fórmulas clássicas.
                s1_c = classic_data.get("s1")
                r1_c = classic_data.get("r1")
                h = float(2 * p_c - s1_c) if s1_c is not None else None
                l = float(2 * p_c - r1_c) if r1_c is not None else None
                c = float(3 * p_c - h - l) if (h is not None and l is not None) else p_c
            if h and l and c and h > l:
                source_tag = "classic"
                classic_levels = {
                    "r1": classic_data.get("r1"), "s1": classic_data.get("s1"),
                    "r2": classic_data.get("r2"), "s2": classic_data.get("s2"),
                    "r3": classic_data.get("r3"), "s3": classic_data.get("s3"),
                }
            else:
                h = l = c = 0.0

        if source_tag != "classic":
            # 2) Fallback: VP intraday parcial (dia corrente 00:00Z→agora)
            if not degenerate and vp.get("vah") and vp.get("val") and vp.get("poc"):
                # VP válido — usar diretamente
                h, l, c = float(vp["vah"]), float(vp["val"]), float(vp["poc"])
            else:
                # VP degenerado ou ausente — usar OHLC real do multi_tf
                tf_data = multi_tf.get(tf_key, {}) or {}
                preco_tf = float(tf_data.get("preco_atual", 0) or 0)
                atr_tf = float(tf_data.get("atr", 0) or 0)
                ema_tf = float(tf_data.get("mme_21", 0) or 0)

                if preco_tf > 0 and atr_tf > 0:
                    # Estimar H/L/C a partir de ATR: H ≈ EMA + ATR, L ≈ EMA - ATR
                    ref = ema_tf if ema_tf > 0 else preco_tf
                    h = round(ref + atr_tf, 2)
                    l = round(ref - atr_tf, 2)
                    c = round(preco_tf, 2)
                elif preco_tf > 0:
                    # Sem ATR — usar ±0.5% como estimativa
                    pct = {"daily": 0.005, "weekly": 0.015, "monthly": 0.04}.get(period, 0.01)
                    h = round(preco_tf * (1 + pct), 2)
                    l = round(preco_tf * (1 - pct), 2)
                    c = round(preco_tf, 2)
                else:
                    continue  # Sem dados suficientes — pular este TF

        if h <= 0 or l <= 0 or c <= 0 or h <= l:
            continue

        if source_tag == "classic":
            pivot = round(float(classic_data["pivot"]), 2)
            pivot_data: dict = {
                "pivot": pivot,
                "r1": classic_levels["r1"],
                "r2": classic_levels["r2"],
                "r3": classic_levels["r3"],
                "s1": classic_levels["s1"],
                "s2": classic_levels["s2"],
                "s3": classic_levels["s3"],
                "vah": round(h, 2),
                "val": round(l, 2),
                "poc": round(c, 2),
                "source": "classic",
            }
            if classic_pivots.get("calculated_at_ms"):
                pivot_data["calculated_at_ms"] = classic_pivots["calculated_at_ms"]
        else:
            pivot = round((h + l + c) / 3, 2)
            rng = h - l

            pivot_data = {
                "pivot": pivot,
                "r1": round(2 * pivot - l, 2),
                "r2": round(pivot + rng, 2),
                "r3": round(pivot + 2 * rng, 2),
                "s1": round(2 * pivot - h, 2),
                "s2": round(pivot - rng, 2),
                "s3": round(l - 2 * (h - pivot), 2),
                "vah": round(h, 2),
                "val": round(l, 2),
                "poc": round(c, 2),
                # Rastreabilidade: fallback é VP intraday PARCIAL (não é
                # pivot clássico fixo — muda a cada ciclo de atualização)
                "source": "multi_tf_fallback" if degenerate else "vp_fallback",
            }

            if degenerate:
                pivot_data["vp_status"] = "insufficient_data"

        pivots[period] = pivot_data

        # Coletar candidatos a S/R
        if current_price > 0:
            for level, label in [(h, "R"), (pivot, "P"), (l, "S")]:
                dist_pct = abs(level - current_price) / current_price * 100
                if dist_pct <= 5.0:
                    # Escala 0-100: weight(0-1) * 100 * proximity_factor
                    strength = round(weight * 100 * (1 - dist_pct / 10), 1)
                    lvl_rounded = round(level, 2)
                    if level > current_price:
                        resistance_levels.append(lvl_rounded)
                        resistance_strength.append(min(100.0, strength))
                    else:
                        support_levels.append(lvl_rounded)
                        support_strength.append(min(100.0, strength))

    # Adicionar defense zones como S/R
    ia = event.get("institutional_analytics", {}) or {}
    dz = _get_nested(ia, "sr_analysis", "defense_zones") or {}
    buy_def = dz.get("buy_defense", []) or []
    sell_def = dz.get("sell_defense", []) or []

    for zone in buy_def[:3]:
        center = zone.get("center")
        strength = zone.get("strength", 0)  # Já em escala 0-100
        # Guarda wall-only (fix 2026-09): zona de snapshot liquidity sem fonte
        # estrutural independente NÃO vira suporte imediato — wall snapshot-only
        # não é S/R autônomo. Com confluência estrutural, a promoção segue.
        if _is_non_structural_zone(zone):
            continue
        # Guarda de lado: defesa buy só vira suporte imediato se o centro
        # estiver abaixo/igual ao preço (consistente com o filtro h/pivot/l
        # acima). Zona buy acima do preço (voto majoritário) não contamina
        # immediate_support.
        if center and center <= current_price:
            support_levels.append(round(center, 2))
            support_strength.append(min(100.0, round(strength, 1)))

    for zone in sell_def[:3]:
        center = zone.get("center")
        strength = zone.get("strength", 0)  # Já em escala 0-100
        # Guarda wall-only (fix 2026-09): idem para resistência imediata.
        if _is_non_structural_zone(zone):
            continue
        # Guarda de lado: defesa sell só vira resistência imediata se o
        # centro estiver acima do preço (consistente com h/pivot/l).
        if center and center > current_price:
            resistance_levels.append(round(center, 2))
            resistance_strength.append(min(100.0, round(strength, 1)))

    # Deduplicar: manter o de maior strength quando há duplicatas
    def _dedup_levels(levels, strengths):
        """Remove duplicatas mantendo a maior strength para cada nível."""
        best = {}
        for lvl, st in zip(levels, strengths):
            if lvl not in best or st > best[lvl]:
                best[lvl] = st
        return list(best.keys()), list(best.values())

    support_levels, support_strength = _dedup_levels(support_levels, support_strength)
    resistance_levels, resistance_strength = _dedup_levels(resistance_levels, resistance_strength)

    # Ordenar: suportes decrescente, resistências crescente
    if support_levels and current_price > 0:
        combined_s = sorted(zip(support_levels, support_strength), key=lambda x: -x[0])
        if combined_s:
            support_levels = [p[0] for p in combined_s[:5]]
            support_strength = [p[1] for p in combined_s[:5]]

    if resistance_levels and current_price > 0:
        combined_r = sorted(zip(resistance_levels, resistance_strength), key=lambda x: x[0])
        if combined_r:
            resistance_levels = [p[0] for p in combined_r[:5]]
            resistance_strength = [p[1] for p in combined_r[:5]]

    result: dict = {}
    if pivots:
        result["pivot_points"] = pivots
    if support_levels:
        result["immediate_support"] = support_levels
        result["support_strength"] = support_strength
    if resistance_levels:
        result["immediate_resistance"] = resistance_levels
        result["resistance_strength"] = resistance_strength

    return result


# ---------------------------------------------------------------------------
# ONDA 1 — Fibonacci padronizado na raiz
# ---------------------------------------------------------------------------

def _build_fibonacci(event: dict, current_price: float) -> dict:
    """
    Retorna fibonacci_levels padronizado usando swing high/low da sessão.
    Prioriza pattern_recognition existente ou calcula dos dados de candles.
    """
    # Verificar se já tem fibonacci em pattern_recognition
    pr = event.get("pattern_recognition", {}) or {}
    fib_existing = pr.get("fibonacci_levels", {}) or {}
    if fib_existing and fib_existing.get("high") and fib_existing.get("low"):
        return {"fibonacci_levels": fib_existing}

    # Calcular do contextual_snapshot ou historical_vp
    snap = event.get("contextual_snapshot", {}) or {}
    ohlc = snap.get("ohlc", {}) or {}
    high = ohlc.get("high") or event.get("historical_vp", {}).get("daily", {}).get("vah")
    low = ohlc.get("low") or event.get("historical_vp", {}).get("daily", {}).get("val")

    if not (high and low and high > low):
        return {}

    rng = high - low

    # FIX 6B: Fibonacci is useless on tiny ranges (e.g. $28 on $70k BTC = 0.04%)
    # Minimum 0.3% range required (~$210 at $70k)
    if low > 0 and (rng / low) < 0.003:
        return {}

    fib = {
        "high": round(high, 2),
        "low": round(low, 2),
        "23.6": round(low + rng * 0.236, 2),
        "38.2": round(low + rng * 0.382, 2),
        "50.0": round(low + rng * 0.500, 2),
        "61.8": round(low + rng * 0.618, 2),
        "78.6": round(low + rng * 0.786, 2),
    }
    return {"fibonacci_levels": fib}


# ---------------------------------------------------------------------------
# ONDA 1 — Spread Volatility
# ---------------------------------------------------------------------------

def _update_spread_history(spread_bps: float) -> None:
    _STATE.spread_history.append(spread_bps)


def _build_spread_volatility() -> Optional[float]:
    """Desvio padrão dos spreads no período."""
    if len(_STATE.spread_history) < 3:
        return None
    spreads = list(_STATE.spread_history)
    n = len(spreads)
    mean = sum(spreads) / n
    var = sum((x - mean) ** 2 for x in spreads) / n
    return round(math.sqrt(var), 4)


# ---------------------------------------------------------------------------
# ONDA 1 — Slippage 1k/10k (interpolação do OB existente)
# ---------------------------------------------------------------------------

def _build_slippage_small(event: dict) -> dict:
    """
    Interpola slippage para 1k e 10k USD a partir do order book depth.
    Usa interpolação linear baseada na liquidez disponível em L1/L5.
    """
    ob = event.get("orderbook_data", {}) or {}
    ob_depth = event.get("order_book_depth", {}) or {}
    mi = event.get("market_impact", {}) or {}
    slippage_existing = mi.get("slippage_matrix", {}) or {}

    # Se já tem 100k/1m, interpolamos
    s100k = slippage_existing.get("100k_usd", {}) or {}
    s1m = slippage_existing.get("1m_usd", {}) or {}

    result = {}
    for side in ("buy", "sell"):
        val_100k = s100k.get(side)
        val_1m = s1m.get(side)

        if val_100k is not None and val_100k > 0:
            # Slippage ~linear com tamanho abaixo de 100k
            # 1k = ~1% de 100k em tamanho → slippage proporcional
            slip_1k = round(val_100k * 0.01, 4)
            slip_10k = round(val_100k * 0.10, 4)
        else:
            # Fallback: estima pelo bid_depth L1
            L1 = ob_depth.get("L1", {}) or {}
            depth_side = L1.get("bids" if side == "buy" else "asks", 100000)
            # Slippage para 1k = 1000 / depth * spread_factor
            spread_bps = ob.get("spread", 0.1) / ob.get("mid", 68000) * 10000
            slip_1k = round(1000 / max(depth_side, 1000) * spread_bps, 4)
            slip_10k = round(10000 / max(depth_side, 1000) * spread_bps, 4)

        result[f"1k_usd_{side}"] = slip_1k
        result[f"10k_usd_{side}"] = slip_10k

    return {
        "1k_usd":  {"buy": result.get("1k_usd_buy", 0.0),  "sell": result.get("1k_usd_sell", 0.0)},
        "10k_usd": {"buy": result.get("10k_usd_buy", 0.0), "sell": result.get("10k_usd_sell", 0.0)},
    }


# ---------------------------------------------------------------------------
# ONDA 1 — Funding Rate BTC (via derivatives já presentes no evento)
# ---------------------------------------------------------------------------

def _parse_valid_funding_value(val: Any) -> Optional[float]:
    """
    Valida e converte valor de funding:
    - Rejeita None, bool (True/False) e strings não-numéricas.
    - Rejeita valores não-finitos: NaN, +Inf, -Inf.
    - Retorna float finito ou None.
    """
    if val is None or isinstance(val, bool):
        return None
    try:
        f = float(val)
        if math.isfinite(f):
            return f
    except (ValueError, TypeError):
        pass
    return None


def _build_btc_funding(event: dict) -> Optional[float]:
    """
    Extrai funding rate do BTC em formato PERCENTUAL (ex: 0.01 para 0.01%).
    Retorna SEMPRE em percentual finito. Nunca retorna fração nem NaN/Inf.

    Precedência canônica determinística:
    1. funding_rate_percent [%]      (se numérico finito)
    2. funding_rate_pct     [%]      (se numérico finito)
    3. funding_rate         [fração] -> convertido ×100 (se numérico finito)
    """
    deriv = event.get("derivatives", {}) or {}
    btc_d = deriv.get("BTCUSDT", {}) or {}

    val_percent = _parse_valid_funding_value(btc_d.get("funding_rate_percent"))
    val_pct = _parse_valid_funding_value(btc_d.get("funding_rate_pct"))

    # 1. Preferir funding_rate_percent se válido e finito
    if val_percent is not None:
        if val_pct is not None and abs(val_percent - val_pct) > 1e-6:
            logging.getLogger(__name__).warning(
                "Inconsistência em derivativos BTC: funding_rate_percent=%.6f != funding_rate_pct=%.6f. "
                "Usando funding_rate_percent.", val_percent, val_pct
            )
        return val_percent

    # 2. Fallback para funding_rate_pct se válido e finito
    if val_pct is not None:
        return val_pct

    # 3. Fallback para funding_rate (fração decimal) -> converte para percentual
    val_raw = _parse_valid_funding_value(btc_d.get("funding_rate"))
    if val_raw is not None:
        return val_raw * 100.0

    return None


# ---------------------------------------------------------------------------
# ONDA 2 — Volume Profile avançado (value_area_volume_pct, HVN/LVN, single prints)
# ---------------------------------------------------------------------------

def _build_volume_profile_advanced(event: dict, current_price: float) -> dict:
    """
    Adiciona campos avançados ao volume profile:
    - value_area_volume_pct
    - single_prints (estimativa por bins vazios)
    - hvn_nodes[] / lvn_nodes[] estruturados
    """
    result: dict = {}
    historical_vp = event.get("historical_vp", {}) or {}

    for period in ("daily", "weekly", "monthly"):
        vp = historical_vp.get(period, {}) or {}
        vah = vp.get("vah")
        val = vp.get("val")
        poc = vp.get("poc")

        if not (vah and val and poc):
            continue

        # Value Area Volume % — padrão institucional: ~68% (1 desvio padrão)
        # Estimativa: VA_range / session_range * escala para ~68%
        session_high = event.get("contextual_snapshot", {}).get("ohlc", {}).get("high", vah)
        session_low = event.get("contextual_snapshot", {}).get("ohlc", {}).get("low", val)
        session_range = max(session_high, vah) - min(session_low, val)
        va_range = vah - val
        raw_ratio = _safe_div(va_range, session_range, 0.68)
        # Escalar para manter ~68% como centro (padrão institucional)
        # ratio próximo de 1.0 → ~68%, ratio < 0.5 → < 50% (disperso)
        va_pct = round(raw_ratio * 68.0, 1)
        # Clamp entre 30% e 70% (padrão institucional, não 95%)
        va_pct = max(30.0, min(70.0, va_pct))

        # HVN/LVN a partir do poc e dos extremos
        # HVN: poc ± 0.5 ATR (proxy de nodo de alto volume)
        atr = 0.0
        multi_tf = event.get("multi_tf", {}) or {}
        for tf in ("1h", "15m", "4h"):
            atr_val = multi_tf.get(tf, {}).get("atr", 0) or 0
            if atr_val > 0:
                atr = atr_val
                break

        if atr == 0:
            atr = abs(vah - val) * 0.15

        hvn_nodes = []
        lvn_nodes = []

        # POC region = HVN primário (escala 0-100)
        hvn_nodes.append({
            "price": round(poc, 2),
            "volume": vp.get("poc_volume", 0),
            "strength": 95,
            "timeframe": period,
            "type": "poc",
        })

        # VAH/VAL são nodos de volume médio (borda da value area)
        hvn_nodes.append({
            "price": round(vah, 2),
            "volume": 0,
            "strength": 70,
            "timeframe": period,
            "type": "vah",
        })

        # LVN: áreas acima do VAH e abaixo do VAL (regiões de baixo volume)
        if current_price > 0:
            lvn_above = round(vah + atr * 0.5, 2)
            lvn_below = round(val - atr * 0.5, 2)
            # Strength normalizado: distância relativa ao preço, escala 0-100
            dist_above_pct = abs(lvn_above - current_price) / current_price * 100 if current_price > 0 else 0
            dist_below_pct = abs(lvn_below - current_price) / current_price * 100 if current_price > 0 else 0
            lvn_nodes.append({
                "price": lvn_above,
                "volume": 0,
                "strength": round(min(100.0, dist_above_pct * 20), 1),
                "timeframe": period,
                "type": "lvn_above_vah",
            })
            lvn_nodes.append({
                "price": lvn_below,
                "volume": 0,
                "strength": round(min(100.0, dist_below_pct * 20), 1),
                "timeframe": period,
                "type": "lvn_below_val",
            })

        key = f"vp_{period}_advanced"
        result[key] = {
            "value_area_volume_pct": va_pct,
            "hvn_nodes": hvn_nodes,
            "lvn_nodes": lvn_nodes,
        }

    # Single prints (aproximação via clusters vazios na heatmap)
    heatmap = _get_nested(event, "fluxo_continuo", "liquidity_heatmap") or {}
    clusters = heatmap.get("clusters", []) or []
    single_prints = []
    for cl in clusters:
        # Cluster com imbalance extremo e volume baixo → potencial single print
        imb = abs(cl.get("imbalance_ratio", 0) or 0)
        total_vol = cl.get("total_volume", 0) or 0
        if imb > 0.4 and total_vol < 2.0:
            single_prints.append(round(cl.get("center", 0), 2))
    if single_prints:
        result["single_prints"] = single_prints

    return result


# ---------------------------------------------------------------------------
# ONDA 2 — Volatilidade consolidada
# ---------------------------------------------------------------------------

def _update_vol_history(epoch_ms: int, realized_vol: float) -> None:
    _STATE.vol_history.append((epoch_ms, realized_vol))


def _build_volatility_metrics(event: dict, epoch_ms: int) -> dict:
    """
    Consolida volatilidade de múltiplas fontes:
    - realized_vol_24h, realized_vol_7d
    - volatility_percentile (vs histórico de sessão)
    """
    multi_tf = event.get("multi_tf", {}) or {}
    result: dict = {}

    # Coletar vols por TF
    vols_by_tf = {}
    for tf in ("15m", "1h", "4h", "1d"):
        v = multi_tf.get(tf, {}).get("realized_vol")
        if v and v > 0:
            vols_by_tf[tf] = v

    # 24h approximation: usa 1d se disponível, senão 4h * sqrt(6)
    if "1d" in vols_by_tf:
        result["realized_vol_24h"] = round(vols_by_tf["1d"], 6)
    elif "4h" in vols_by_tf:
        result["realized_vol_24h"] = round(vols_by_tf["4h"] * math.sqrt(6), 6)

    # 7d approximation: 1d * sqrt(7)
    if "1d" in vols_by_tf:
        result["realized_vol_7d"] = round(vols_by_tf["1d"] * math.sqrt(7), 6)

    # Volatility regime
    vol_regime = event.get("market_environment", {}).get("volatility_regime", "NORMAL")
    result["volatility_regime"] = vol_regime

    # Volatility percentile (vs histórico da sessão corrente)
    current_vol = vols_by_tf.get("1h") or vols_by_tf.get("4h")
    if current_vol:
        _update_vol_history(epoch_ms, current_vol)

    # Mínimo 20 janelas para percentil estatisticamente significativo.
    # Com menos dados, qualquer valor pode ser o máximo (percentil=100) por acidente.
    if current_vol and len(_STATE.vol_history) >= 20:
        hist_vols = [v for _, v in _STATE.vol_history]
        below = sum(1 for v in hist_vols if v <= current_vol)
        pct = round(below / len(hist_vols) * 100, 1)
        result["volatility_percentile"] = pct
    elif current_vol:
        result["volatility_percentile"] = None  # histórico insuficiente (<20 janelas)
    elif vol_regime == "LOW":
        result["volatility_percentile"] = 25.0
    elif vol_regime == "HIGH":
        result["volatility_percentile"] = 75.0
    else:
        result["volatility_percentile"] = 50.0

    return result


# ---------------------------------------------------------------------------
# ONDA 2 — Passive buy/sell pct explícitos
# ---------------------------------------------------------------------------

def _build_passive_flow(event: dict) -> dict:
    """
    Calcula passive_buy_pct e passive_sell_pct.

    Lógica: se agressores vendem muito, quem está absorvendo (passivo) são
    os compradores no book.  Portanto passive_buy_pct deve ser alto quando
    aggressive_sell_pct é alto — refletindo que as limit orders de compra
    estão absorvendo a pressão vendedora.

    Fonte primária: orderbook depth (bid_depth vs ask_depth).
    Fallback: inversão do aggressive pct.
    """
    # P1-B: presença explícita + finiteness + aggressive_status.
    # 0.0/50.0/100.0 reais preservados (`or 50.0` os destruía e o gate
    # `!= 50` abaixo estava morto: missing já chegava como 50.0).
    def _finite_pct(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        fv = float(value)
        return fv if math.isfinite(fv) else None

    flow = _get_nested(event, "fluxo_continuo", "order_flow") or {}
    status = flow.get("aggressive_status")
    agg_buy = _finite_pct(flow.get("aggressive_buy_pct"))
    agg_sell = _finite_pct(flow.get("aggressive_sell_pct"))

    # Fonte primária: orderbook depth (quem tem mais liquidez passiva).
    # Independente do agressivo (ver test_B2_*): vale mesmo sem pcts.
    ob = event.get("orderbook_data", {}) or {}
    bid_depth = ob.get("bid_depth_usd", 0) or 0
    ask_depth = ob.get("ask_depth_usd", 0) or 0
    depth_total = bid_depth + ask_depth

    if depth_total > 0:
        # Bids são passive buyers, asks são passive sellers
        passive_buy_pct = round(bid_depth / depth_total * 100, 1)
        passive_sell_pct = round(100 - passive_buy_pct, 1)
        return {
            "passive_buy_pct": passive_buy_pct,
            "passive_sell_pct": passive_sell_pct,
        }
    if agg_buy is None or agg_sell is None:
        return {}
    if status is not None and status != "observed":
        return {}
    # Fallback: passive = inversão do aggressive observado
    # (ou legado com pcts finitos e sem status).
    # Se agg_sell=85%, passive_buy=85% (compradores absorvendo)
    return {
        "passive_buy_pct": round(agg_sell, 1),
        "passive_sell_pct": round(agg_buy, 1),
    }


# ---------------------------------------------------------------------------
# ONDA 2 — Whale: large orders e iceberg detection
# ---------------------------------------------------------------------------

def _register_large_trades(valid_window_data: List[Dict], current_price: float, epoch_ms: int) -> None:
    """Filtra e armazena trades grandes para large_orders_1h.

    NOTA DE PROVENIÊNCIA:
    Trata-se de executed aggressive trades (aggTrades >= 5.0 BTC),
    NÃO ordens limites descansando no livro (resting limit orders).
    """
    for trade in valid_window_data:
        qty = float(trade.get("q", 0) or 0)
        if qty >= _LARGE_ORDER_THRESHOLD_BTC:
            ts = int(trade.get("T", epoch_ms))
            price = float(trade.get("p", current_price))
            is_sell = bool(trade.get("m", False))
            _STATE.large_trades.append({
                "size": round(qty, 4),
                "price": round(price, 2),
                "side": "SELL" if is_sell else "BUY",
                "timestamp_ms": ts,
            })


def _build_whale_activity(event: dict, epoch_ms: int, current_price: float) -> dict:
    """
    Retorna large_orders_1h, iceberg_activity, hidden_orders_detected.

    NOTA DE PROVENIÊNCIA & CAPABILITY:
    - large_orders_1h: executed aggressive trades (aggTrades >= 5.0 BTC).
    - iceberg_activity: heurística experimental de fragmentação de trades (clusters).
      NÃO representa detecção de iceberg em orderbook e NÃO alimenta o payload de IA.
    - hidden_orders_detected: heurística experimental baseada em desequilíbrio sem absorção.
      NÃO suportada como evidência de ordens ocultas reais; NÃO alimentar IA.
    """
    result: dict = {}

    # large_orders_1h: filtrar por última hora + deduplicar (executed trades)
    cutoff_1h = epoch_ms - 3600 * 1000
    large_1h_raw = [t for t in _STATE.large_trades if t.get("timestamp_ms", 0) >= cutoff_1h]
    if large_1h_raw:
        seen = set()
        large_1h = []
        for t in large_1h_raw:
            key = (t.get("size"), t.get("price"), t.get("timestamp_ms"))
            if key not in seen:
                seen.add(key)
                large_1h.append(t)
        result["large_orders_1h"] = large_1h[-20:]  # Últimas 20 ordens grandes executadas (dedup)

    # Iceberg detection: mesmos preços com muitas execuções fragmentadas
    # Heurística: cluster com trades_count alto + avg_trade_size muito baixo
    # (NOTA: trade fragmentation pattern, não L2 iceberg replenishment)
    heatmap = _get_nested(event, "fluxo_continuo", "liquidity_heatmap") or {}
    clusters = heatmap.get("clusters", []) or []
    iceberg_detected = False
    for cl in clusters:
        trades_count = cl.get("trades_count", 0) or 0
        avg_size = cl.get("avg_trade_size", 1) or 1
        total_vol = cl.get("total_volume", 0) or 0
        # Sinal de fragmentação: muitos trades pequenos no mesmo nível com volume total relevante
        if trades_count > 500 and avg_size < 0.01 and total_vol > 5.0:
            iceberg_detected = True
            break

    result["iceberg_activity"] = iceberg_detected

    # Hidden orders: heurística experimental baseada em absorção neutra com delta forte (não para IA)
    abs_data = _get_nested(event, "fluxo_continuo", "absorption_analysis", "current_absorption") or {}
    abs_index = abs_data.get("index", 0) or 0
    flow_imb = abs(abs_data.get("flow_imbalance", 0) or 0)
    hidden_orders = 0
    if abs_index < 0.05 and flow_imb > 0.15:
        # Absorção mínima com desequilíbrio grande → sugestão heurística (não comprovada)
        hidden_orders = 1
    result["hidden_orders_detected"] = hidden_orders

    return result


# ---------------------------------------------------------------------------
# ONDA 2 — Defense Zones: No-Man's Land
# ---------------------------------------------------------------------------

def _build_no_mans_land(event: dict) -> Optional[dict]:
    """
    Calcula a zona neutra entre buy_defense e sell_defense.
    """
    ia = event.get("institutional_analytics", {}) or {}
    dz = _get_nested(ia, "sr_analysis", "defense_zones") or {}

    buy_def = dz.get("buy_defense", []) or []
    sell_def = dz.get("sell_defense", []) or []

    if not buy_def or not sell_def:
        return None

    # Melhor buy defense (mais perto do preço = maior distância negativa)
    best_buy = max(buy_def, key=lambda z: z.get("center", 0))
    best_sell = min(sell_def, key=lambda z: z.get("center", float("inf")))

    buy_top = best_buy.get("range_high", best_buy.get("center", 0))
    sell_bottom = best_sell.get("range_low", best_sell.get("center", 0))

    if buy_top and sell_bottom and sell_bottom > buy_top:
        return {
            "start": round(buy_top, 2),
            "end": round(sell_bottom, 2),
            "width": round(sell_bottom - buy_top, 2),
        }
    return None


# ---------------------------------------------------------------------------
# ONDA 2 — Sistema de Alertas Estruturado
# ---------------------------------------------------------------------------

def _should_fire_alert(alert_type: str, now_ts: float) -> bool:
    """Verifica cooldown por tipo de alerta."""
    last = _STATE.last_alert_ts.get(alert_type, 0)
    if now_ts - last >= _ALERT_COOLDOWN_SEC:
        _STATE.last_alert_ts[alert_type] = now_ts
        return True
    return False


def _build_alerts(event: dict, current_price: float, epoch_ms: int) -> dict:
    """
    Sistema de alertas estruturado.
    Tipos: SUPPORT_TEST, RESISTANCE_TEST, VOLUME_SPIKE, DEPTH_DIVERGENCE,
           WHALE_DISTRIBUTION, WHALE_ACCUMULATION, RSI_DIVERGENCE,
           FUNDING_EXTREME, OB_ANOMALY
    """
    alerts = []
    now_ts = epoch_ms / 1000

    # --- SUPPORT_TEST ---
    support_levels = event.get("immediate_support", []) or []
    for level in support_levels[:3]:
        dist_pct = abs(current_price - level) / current_price * 100
        if dist_pct <= 0.3 and _should_fire_alert("SUPPORT_TEST", now_ts):
            alerts.append({
                "type": "SUPPORT_TEST",
                "level": level,
                "severity": "HIGH" if dist_pct <= 0.1 else "MEDIUM",
                "probability": round(max(0.5, 1 - dist_pct * 2), 2),
                "action": "MONITOR_CLOSELY",
                "description": f"Preço testando suporte em {level:.2f} (dist: {dist_pct:.2f}%)",
            })

    # --- RESISTANCE_TEST ---
    resistance_levels = event.get("immediate_resistance", []) or []
    for level in resistance_levels[:3]:
        dist_pct = abs(current_price - level) / current_price * 100
        if dist_pct <= 0.3 and _should_fire_alert("RESISTANCE_TEST", now_ts):
            alerts.append({
                "type": "RESISTANCE_TEST",
                "level": level,
                "severity": "HIGH" if dist_pct <= 0.1 else "MEDIUM",
                "probability": round(max(0.5, 1 - dist_pct * 2), 2),
                "action": "MONITOR_CLOSELY",
                "description": f"Preço testando resistência em {level:.2f} (dist: {dist_pct:.2f}%)",
            })

    # --- VOLUME_SPIKE ---
    snap = event.get("contextual_snapshot", {}) or {}
    vol_sma_ratio = _get_nested(event, "ml_features", "volume_features", "volume_sma_ratio", default=1.0) or 1.0
    if vol_sma_ratio >= 3.0 and _should_fire_alert("VOLUME_SPIKE", now_ts):
        alerts.append({
            "type": "VOLUME_SPIKE",
            "threshold_exceeded": round(vol_sma_ratio, 2),
            "severity": "HIGH" if vol_sma_ratio >= 5 else "MEDIUM",
            "probability": min(0.95, round(vol_sma_ratio / 10, 2)),
            "action": "PREPARE_ENTRY" if vol_sma_ratio >= 5 else "MONITOR",
            "description": f"Volume {vol_sma_ratio:.1f}x acima da média",
        })

    # --- DEPTH_DIVERGENCE (orderbook extremamente assimétrico) ---
    ob = event.get("orderbook_data", {}) or {}
    ob_imb = ob.get("imbalance", 0) or 0
    if abs(ob_imb) >= 0.7 and _should_fire_alert("DEPTH_DIVERGENCE", now_ts):
        direction = "ASK_HEAVY" if ob_imb < 0 else "BID_HEAVY"
        alerts.append({
            "type": "DEPTH_DIVERGENCE",
            "level": round(ob_imb, 3),
            "severity": "HIGH" if abs(ob_imb) >= 0.85 else "MEDIUM",
            "probability": round(min(0.9, abs(ob_imb)), 2),
            "action": "PREPARE_SHORT" if ob_imb < 0 else "PREPARE_LONG",
            "description": f"Orderbook {direction}: imbalance={ob_imb:.3f}",
        })

    # --- WHALE_DISTRIBUTION / ACCUMULATION ---
    ia = event.get("institutional_analytics", {}) or {}
    whale = _get_nested(ia, "flow_analysis", "whale_accumulation") or {}
    whale_score = whale.get("score", 0) or 0
    if whale_score <= -25 and _should_fire_alert("WHALE_DISTRIBUTION", now_ts):
        alerts.append({
            "type": "WHALE_DISTRIBUTION",
            "level": whale_score,
            "severity": "HIGH" if whale_score <= -35 else "MEDIUM",
            "probability": min(0.9, round(abs(whale_score) / 50, 2)),
            "action": "AVOID_LONG",
            "description": f"Sinal de distribuição de whales (score={whale_score})",
        })
    elif whale_score >= 25 and _should_fire_alert("WHALE_ACCUMULATION", now_ts):
        alerts.append({
            "type": "WHALE_ACCUMULATION",
            "level": whale_score,
            "severity": "HIGH" if whale_score >= 35 else "MEDIUM",
            "probability": min(0.9, round(whale_score / 50, 2)),
            "action": "PREPARE_LONG",
            "description": f"Sinal de acumulação de whales (score={whale_score})",
        })

    # --- RSI_DIVERGENCE (preço faz novo low mas RSI não) ---
    multi_tf = event.get("multi_tf", {}) or {}
    tf_1h = multi_tf.get("1h", {}) or {}
    rsi_1h = tf_1h.get("rsi_short")
    if rsi_1h is not None:
        if rsi_1h < 15 and _should_fire_alert("RSI_OVERSOLD", now_ts):
            alerts.append({
                "type": "RSI_OVERSOLD",
                "level": rsi_1h,
                "severity": "HIGH" if rsi_1h < 10 else "MEDIUM",
                "probability": round(max(0.5, (20 - rsi_1h) / 20), 2),
                "action": "WATCH_REVERSAL",
                "description": f"RSI 1h em zona extremamente sobrevendida ({rsi_1h:.1f})",
            })
        elif rsi_1h > 85 and _should_fire_alert("RSI_OVERBOUGHT", now_ts):
            alerts.append({
                "type": "RSI_OVERBOUGHT",
                "level": rsi_1h,
                "severity": "MEDIUM",
                "probability": round(max(0.5, (rsi_1h - 80) / 20), 2),
                "action": "WATCH_REVERSAL",
                "description": f"RSI 1h em zona extremamente sobrecomprada ({rsi_1h:.1f})",
            })

    # --- FUNDING_EXTREME ---
    deriv = event.get("derivatives", {}) or {}
    btc_deriv = deriv.get("BTCUSDT", {}) or {}
    funding = btc_deriv.get("funding_rate_percent") or btc_deriv.get("funding_rate_btc_pct")
    if funding is not None and abs(float(funding)) >= 0.05:
        if _should_fire_alert("FUNDING_EXTREME", now_ts):
            direction = "ALTA" if float(funding) > 0 else "BAIXA"
            alerts.append({
                "type": "FUNDING_EXTREME",
                "level": funding,
                "severity": "MEDIUM",
                "probability": min(0.8, abs(float(funding)) / 0.1),
                "action": f"PRESSÃO_{direction}",
                "description": f"Funding rate extremo: {funding}% — pressão de {direction}",
            })

    # --- BREAKOUT_SIGNAL (profile B com poor extremes) ---
    prof = _get_nested(ia, "profile_analysis", "profile_shape") or {}
    if prof.get("trading_signal") == "BREAKOUT_EXPECTED" and _should_fire_alert("BREAKOUT_SIGNAL", now_ts):
        alerts.append({
            "type": "BREAKOUT_SIGNAL",
            "severity": "MEDIUM",
            "probability": 0.65,
            "action": "PREPARE_BREAKOUT",
            "description": f"Perfil tipo B detectado — breakout iminente",
        })

    return {
        "active_alerts": alerts,
        "alert_count": len(alerts),
        "max_severity": (
            "HIGH" if any(a.get("severity") == "HIGH" for a in alerts)
            else "MEDIUM" if alerts else "NONE"
        ),
    }


# ---------------------------------------------------------------------------
# ONDA 2 — Indicadores técnicos extras (CCI)
# ---------------------------------------------------------------------------

def _build_cci(event: dict) -> Optional[float]:
    """
    CCI usando multi_tf. CCI = (price - MA) / (0.015 * MeanDev)
    Aproximação com dados disponíveis.
    """
    multi_tf = event.get("multi_tf", {}) or {}
    tf = multi_tf.get("1h", {}) or {}
    price = tf.get("preco_atual", 0) or 0
    ema = tf.get("mme_21", 0) or 0
    atr = tf.get("atr", 0) or 0

    if not (price and ema and atr):
        return None

    # Proxy: mean deviation ≈ ATR / 4
    mean_dev = atr / 4
    if mean_dev == 0:
        return None

    cci = (price - ema) / (0.015 * mean_dev)
    return round(cci, 2)


# ---------------------------------------------------------------------------
# ONDA 3 — Price Targets Probabilísticos
# ---------------------------------------------------------------------------

def _build_price_targets_probabilistic(event: dict, current_price: float) -> dict:
    """
    Gera price_targets estruturados para 5m, 15m e 1h usando:
    - ATR dos TFs correspondentes
    - VP levels (VAH, VAL, POC)
    - Defense zones
    - Probabilidade baseada no modelo ML já existente (_model_prob_up)
    """
    if current_price <= 0:
        return {}

    multi_tf = event.get("multi_tf", {}) or {}
    historical_vp = event.get("historical_vp", {}) or {}
    ia = event.get("institutional_analytics", {}) or {}

    # Probabilidade base do modelo ML
    prob_up = 0.5
    ai_result = event.get("ai_result", {}) or {}
    if "_model_prob_up" in ai_result:
        prob_up = float(ai_result["_model_prob_up"])
    elif "quant" in event.get("ai_payload", {}):
        prob_up = float(event["ai_payload"]["quant"].get("prob_up", 0.5))

    prob_down = 1.0 - prob_up

    # ATR por timeframe
    atrs = {}
    for tf in ("15m", "1h", "4h"):
        atr = multi_tf.get(tf, {}).get("atr")
        if atr and atr > 0:
            atrs[tf] = float(atr)

    if not atrs:
        return {}

    atr_15m = atrs.get("15m", current_price * 0.002)
    atr_1h = atrs.get("1h", current_price * 0.005)
    atr_4h = atrs.get("4h", current_price * 0.010)

    # Targets por timeframe
    # Bull target: próximo nível de resistência ou +ATR
    # Bear target: próximo nível de suporte ou -ATR
    vp_daily = historical_vp.get("daily", {}) or {}
    vah = vp_daily.get("vah", current_price + atr_1h * 2)
    val = vp_daily.get("val", current_price - atr_1h * 2)
    poc = vp_daily.get("poc", current_price)

    # Defense zones como targets
    dz = _get_nested(ia, "sr_analysis", "defense_zones") or {}
    sell_def = dz.get("sell_defense", []) or []
    buy_def = dz.get("buy_defense", []) or []

    sell_target = sell_def[0].get("center") if sell_def else None
    buy_target = buy_def[0].get("center") if buy_def else None

    def _bull_target(atr_mult: float) -> float:
        candidates = [current_price + atr_mult * atr_15m]
        if sell_target and sell_target > current_price:
            candidates.append(sell_target)
        if vah and vah > current_price:
            candidates.append(vah)
        return round(min(candidates, key=lambda x: x - current_price if x > current_price else float("inf")), 2)  # type: ignore[arg-type, operator, return-value]

    def _bear_target(atr_mult: float) -> float:
        candidates = [current_price - atr_mult * atr_15m]
        if buy_target and buy_target < current_price:
            candidates.append(buy_target)
        if val and val < current_price:
            candidates.append(val)
        return round(max(candidates, key=lambda x: x if x < current_price else -float("inf")), 2)  # type: ignore[arg-type, operator, return-value]

    # Confidence ajustada por convicção do modelo
    model_conf = float(ai_result.get("_model_confidence") or 0.08)
    base_conf = min(0.8, max(0.35, 0.5 + model_conf * 3))

    short_term = {
        "5m": {
            "bull": _bull_target(0.3),
            "bear": _bear_target(0.3),
            "confidence": round(base_conf * 0.7, 2),
        },
        "15m": {
            "bull": _bull_target(0.7),
            "bear": _bear_target(0.7),
            "confidence": round(base_conf * 0.6, 2),
        },
        "1h": {
            "bull": round(
                min(vah, current_price + atr_1h * 1.2)
                if vah > current_price
                else current_price + atr_1h * 1.2,
                2,
            ),
            "bear": round(
                max(val, current_price - atr_1h * 1.2)
                if val < current_price
                else current_price - atr_1h * 1.2,
                2,
            ),
            "confidence": round(base_conf * 0.5, 2),
        },
    }

    # Distribuição de probabilidade por movimento em bps
    # Baseado em prob_up do modelo + regime de volatilidade
    vol_regime = event.get("market_environment", {}).get("volatility_regime", "NORMAL")
    vol_factor = {"LOW": 0.7, "NORMAL": 1.0, "HIGH": 1.5, "EXTREME": 2.0}.get(vol_regime, 1.0)

    prob_distribution = {
        "up_10_bps": round(prob_up * 0.5 * vol_factor, 3),
        "up_25_bps": round(prob_up * 0.3 * vol_factor, 3),
        "flat":      round(0.2, 3),
        "down_10_bps": round(prob_down * 0.5 * vol_factor, 3),
        "down_25_bps": round(prob_down * 0.3 * vol_factor, 3),
    }

    return {
        "short_term": short_term,
        "probability_distribution": prob_distribution,
        "base_prob_up": round(prob_up, 4),
        "model_confidence": round(model_conf, 4),
    }


# ---------------------------------------------------------------------------
# ONDA 3 — Regime Detection Probabilístico (sem VIX/API paga)
# ---------------------------------------------------------------------------

def _build_regime_probabilities(event: dict) -> dict:
    """
    Calcula probabilidades de regime (trending, mean_reverting, breakout)
    usando dados disponíveis no evento sem API paga:
    - ADX (force of trend)
    - Profile shape (B=breakout, D=range)
    - Whale score
    - Flow acceleration
    - OB imbalance
    """
    multi_tf = event.get("multi_tf", {}) or {}
    ia = event.get("institutional_analytics", {}) or {}
    market_env = event.get("market_environment", {}) or {}

    # ADX médio (força de tendência)
    adx_vals = []
    for tf in ("15m", "1h", "4h"):
        adx = multi_tf.get(tf, {}).get("adx")
        if adx and adx > 0:
            adx_vals.append(float(adx))
    avg_adx = sum(adx_vals) / len(adx_vals) if adx_vals else 25.0

    # Profile shape
    prof_shape = _get_nested(ia, "profile_analysis", "profile_shape", "shape", default="D") or "D"
    prof_signal = _get_nested(ia, "profile_analysis", "profile_shape", "trading_signal", default="RANGE") or "RANGE"

    # Whale score
    whale_score = abs(_get_nested(ia, "flow_analysis", "whale_accumulation", "score", default=0) or 0)

    # OB imbalance
    ob_imb = abs(event.get("orderbook_data", {}).get("imbalance", 0) or 0)

    # Flow trend (aceleração de venda/compra)
    flow_trend = _get_nested(event, "fluxo_continuo", "order_flow", "buy_sell_ratio", "flow_trend", default="") or ""

    # Score para cada regime
    trending_score = 0.0
    mean_rev_score = 0.0
    breakout_score = 0.0

    # ADX: > 40 → trending; < 20 → range; 20-40 → mixed
    if avg_adx > 50:
        trending_score += 0.4
    elif avg_adx > 35:
        trending_score += 0.25
    elif avg_adx < 20:
        mean_rev_score += 0.3
    else:
        mean_rev_score += 0.15

    # Profile shape
    if prof_signal == "BREAKOUT_EXPECTED":
        breakout_score += 0.35
    elif prof_signal == "RANGE":
        mean_rev_score += 0.25
    elif prof_signal in ("TREND_UP", "TREND_DOWN"):
        trending_score += 0.25

    # Whale score: alta acumulação/distribuição → breakout potencial
    if whale_score >= 30:
        breakout_score += 0.20
    elif whale_score >= 15:
        breakout_score += 0.10

    # OB imbalance extremo → possível breakout
    if ob_imb >= 0.8:
        breakout_score += 0.25
        trending_score += 0.10
    elif ob_imb >= 0.5:
        breakout_score += 0.10

    # Flow trend
    if "accel" in flow_trend.lower():
        trending_score += 0.15
    if "reversal" in flow_trend.lower():
        mean_rev_score += 0.15

    # Market structure
    structure = market_env.get("market_structure", "RANGE_BOUND")
    if structure == "TRENDING":
        trending_score += 0.20
    elif structure == "RANGE_BOUND":
        mean_rev_score += 0.20

    # Normalizar para somar 1.0
    total = trending_score + mean_rev_score + breakout_score
    if total == 0:
        trending_score = 0.33
        mean_rev_score = 0.50
        breakout_score = 0.17
        total = 1.0

    trending_prob = round(trending_score / total, 3)
    mean_rev_prob = round(mean_rev_score / total, 3)
    breakout_prob = round(1.0 - trending_prob - mean_rev_prob, 3)
    breakout_prob = max(0.0, breakout_prob)

    # Regime dominante
    probs = {
        "trending": trending_prob,
        "mean_reverting": mean_rev_prob,
        "breakout": breakout_prob,
    }
    current_regime = max(probs, key=probs.get)  # type: ignore[arg-type]

    # Regime change probability: baseado em instabilidade dos sinais
    regime_change_prob = round(breakout_prob + (0.5 - abs(0.5 - trending_prob)) * 0.3, 3)
    regime_change_prob = min(0.95, max(0.05, regime_change_prob))

    # Expected duration
    if current_regime == "trending":
        duration_est = "30m-2h" if avg_adx < 50 else "2h-8h"
    elif current_regime == "mean_reverting":
        duration_est = "15m-1h"
    else:
        duration_est = "5m-30m"

    return {
        "current_regime": current_regime.upper(),
        "regime_probabilities": probs,
        "regime_change_probability": regime_change_prob,
        "expected_regime_duration": duration_est,
        "avg_adx": round(avg_adx, 1),
    }


# ---------------------------------------------------------------------------
# ONDA 2 — Active Patterns (swing detection + padrões geométricos básicos)
# ---------------------------------------------------------------------------

def _build_active_patterns(event: dict, current_price: float) -> list:
    """
    Detecta padrões gráficos básicos usando swing highs/lows do histórico de preços.
    Padrões: ASCENDING_TRIANGLE, DESCENDING_TRIANGLE, SYMMETRICAL_TRIANGLE,
             RISING_WEDGE, FALLING_WEDGE.
    Retorna lista de dicts com type, completion, target_price, stop_loss, confidence.
    """
    patterns: list = []
    history = list(_STATE.price_history)  # [(epoch_ms, price), ...]

    if len(history) < 6:
        return patterns

    # Extrair série de preços e OHLC do evento para swing detection
    prices = [p for _, p in history]
    snap = event.get("contextual_snapshot", {}) or {}
    ohlc = snap.get("ohlc", {}) or {}
    session_high = ohlc.get("high", max(prices))
    session_low = ohlc.get("low", min(prices))
    vp_daily = (event.get("historical_vp", {}) or {}).get("daily", {}) or {}
    vah = vp_daily.get("vah", session_high)
    val = vp_daily.get("val", session_low)

    # --- Encontrar swing highs e lows (simplificado: 3 pontos) ---
    n = len(prices)
    swing_highs = []
    swing_lows = []
    for i in range(1, n - 1):
        if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
            swing_highs.append(prices[i])
        if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
            swing_lows.append(prices[i])

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return patterns

    # Últimos 2 swing highs e lows
    h1, h2 = swing_highs[-2], swing_highs[-1]
    l1, l2 = swing_lows[-2], swing_lows[-1]

    range_pct = (session_high - session_low) / current_price if current_price > 0 else 0.01

    # --- ASCENDING TRIANGLE: highs planos + lows sobindo ---
    high_flat = abs(h2 - h1) / current_price < 0.002  # highs dentro de 0.2%
    lows_rising = l2 > l1
    if high_flat and lows_rising and h2 > current_price:
        target = round(h2 + (h2 - l2), 2)
        stop = round(l2 * 0.998, 2)
        completion = round(min(0.95, (current_price - l2) / max(h2 - l2, 1)), 2)
        patterns.append({
            "type": "ASCENDING_TRIANGLE",
            "completion": completion,
            "target_price": target,
            "stop_loss": stop,
            "confidence": round(0.55 + completion * 0.2, 2),
            "bias": "BULLISH",
        })

    # --- DESCENDING TRIANGLE: lows planos + highs descendo ---
    low_flat = abs(l2 - l1) / current_price < 0.002
    highs_falling = h2 < h1
    if low_flat and highs_falling and l2 < current_price:
        target = round(l2 - (h2 - l2), 2)
        stop = round(h2 * 1.002, 2)
        completion = round(min(0.95, (h2 - current_price) / max(h2 - l2, 1)), 2)
        patterns.append({
            "type": "DESCENDING_TRIANGLE",
            "completion": completion,
            "target_price": target,
            "stop_loss": stop,
            "confidence": round(0.55 + completion * 0.2, 2),
            "bias": "BEARISH",
        })

    # --- SYMMETRICAL TRIANGLE: highs descendo + lows subindo (convergência) ---
    converging = highs_falling and lows_rising
    if converging and not high_flat and not low_flat:
        midpoint = (h2 + l2) / 2
        target_bull = round(h2 + (h1 - l1) * 0.75, 2)
        target_bear = round(l2 - (h1 - l1) * 0.75, 2)
        completion = round(min(0.90, 1 - (h2 - l2) / max(h1 - l1, 1)), 2)
        patterns.append({
            "type": "SYMMETRICAL_TRIANGLE",
            "completion": completion,
            "target_price": target_bull if current_price > midpoint else target_bear,
            "stop_loss": round(l2 * 0.997 if current_price > midpoint else h2 * 1.003, 2),
            "confidence": round(0.45 + completion * 0.25, 2),
            "bias": "BULLISH" if current_price > midpoint else "BEARISH",
        })

    # --- RISING WEDGE: highs e lows subindo mas convergindo (bearish) ---
    both_rising = h2 > h1 and l2 > l1
    rate_high = (h2 - h1) / max(h1, 1)
    rate_low = (l2 - l1) / max(l1, 1)
    if both_rising and rate_low > rate_high and range_pct < 0.03:
        target = round(l1, 2)
        stop = round(h2 * 1.002, 2)
        completion = round(min(0.90, rate_high / max(rate_low, 0.0001)), 2)
        patterns.append({
            "type": "RISING_WEDGE",
            "completion": completion,
            "target_price": target,
            "stop_loss": stop,
            "confidence": round(0.50 + completion * 0.2, 2),
            "bias": "BEARISH",
        })

    # --- FALLING WEDGE: highs e lows descendo mas convergindo (bullish) ---
    both_falling = h2 < h1 and l2 < l1
    rate_high_fall = (h1 - h2) / max(h1, 1)
    rate_low_fall = (l1 - l2) / max(l1, 1)
    if both_falling and rate_high_fall > rate_low_fall and range_pct < 0.03:
        target = round(h1, 2)
        stop = round(l2 * 0.998, 2)
        completion = round(min(0.90, rate_low_fall / max(rate_high_fall, 0.0001)), 2)
        patterns.append({
            "type": "FALLING_WEDGE",
            "completion": completion,
            "target_price": target,
            "stop_loss": stop,
            "confidence": round(0.50 + completion * 0.2, 2),
            "bias": "BULLISH",
        })

    return patterns[:3]  # Máximo 3 padrões por janela


# ---------------------------------------------------------------------------
# ONDA 2 — Stochastic e Williams %R (pull do institutional_analytics ou proxy)
# ---------------------------------------------------------------------------

def _calc_stochastic_real(k_period: int = 14, d_period: int = 3) -> Optional[Dict[str, Any]]:
    """Calcula Stochastic %K e %D reais a partir do OHLC history da sessão."""
    ohlc = list(_STATE.ohlc_history)
    if len(ohlc) < k_period:
        return None

    highs = [c["high"] for c in ohlc]
    lows = [c["low"] for c in ohlc]
    closes = [c["close"] for c in ohlc]

    # %K = 100 * (Close - Lowest Low) / (Highest High - Lowest Low)
    hh = max(highs[-k_period:])
    ll = min(lows[-k_period:])
    denom = hh - ll
    if denom <= 0:
        return None

    k_val = round(100.0 * (closes[-1] - ll) / denom, 2)
    k_val = max(0.0, min(100.0, k_val))

    # %D = SMA dos últimos d_period valores de %K
    k_values = []
    for i in range(min(d_period, len(closes))):
        idx = len(closes) - 1 - i
        if idx < k_period - 1:
            break
        start = idx - k_period + 1
        period_hh = max(highs[start:idx + 1])
        period_ll = min(lows[start:idx + 1])
        period_denom = period_hh - period_ll
        if period_denom > 0:
            k_values.append(100.0 * (closes[idx] - period_ll) / period_denom)
        else:
            k_values.append(50.0)

    d_val = round(sum(k_values) / len(k_values), 2) if k_values else k_val
    d_val = max(0.0, min(100.0, d_val))

    if k_val < 20 and d_val < 20:
        signal = "OVERSOLD"
    elif k_val > 80 and d_val > 80:
        signal = "OVERBOUGHT"
    elif k_val > d_val:
        signal = "BULLISH"
    elif k_val < d_val:
        signal = "BEARISH"
    else:
        signal = "NEUTRAL"

    return {"k": k_val, "d": d_val, "signal": signal, "source": "real"}


def _calc_williams_r_real(period: int = 14) -> Optional[Dict[str, Any]]:
    """Calcula Williams %R real a partir do OHLC history da sessão."""
    ohlc = list(_STATE.ohlc_history)
    if len(ohlc) < period:
        return None

    highs = [c["high"] for c in ohlc[-period:]]
    lows = [c["low"] for c in ohlc[-period:]]
    close = ohlc[-1]["close"]

    hh = max(highs)
    ll = min(lows)
    denom = hh - ll
    if denom <= 0:
        return None

    wr_val = round(-100.0 * (hh - close) / denom, 2)
    wr_val = max(-100.0, min(0.0, wr_val))

    if wr_val > -20:
        zone = "overbought"
    elif wr_val < -80:
        zone = "oversold"
    else:
        zone = "neutral"

    return {
        "value": wr_val,
        "overbought": wr_val > -20,
        "oversold": wr_val < -80,
        "zone": zone,
        "source": "real",
    }


def _inject_advanced_indicators(event: dict) -> None:
    """
    Injeta Hurst, Shannon Entropy, Kalman, Regression Channel, Fourier Cycles,
    Fractal Dimension, Monte Carlo e SMC (FVG, BOS) em technical_indicators_extended
    e pattern_recognition. Lê de institutional_analytics.technical_extras.
    """
    ia = event.get("institutional_analytics", {}) or {}
    tech_extras = ia.get("technical_extras", {}) or {}
    if not tech_extras:
        return

    ti = event.setdefault("technical_indicators_extended", {})

    # Métricas escalares
    for key in ("hurst_exponent", "shannon_entropy", "fractal_dimension"):
        val = tech_extras.get(key)
        if val is not None:
            ti.setdefault(key, val)

    # Dicts compactos
    for key in ("kalman_filter", "regression_channel", "dominant_cycles", "monte_carlo"):
        val = tech_extras.get(key)
        if val and isinstance(val, dict):
            ti.setdefault(key, val)

    # SMC → pattern_recognition
    fvg = tech_extras.get("fair_value_gaps")
    bos = tech_extras.get("market_structure")
    if fvg is not None or bos is not None:
        pr = event.setdefault("pattern_recognition", {})
        sm = pr.setdefault("smart_money", {})
        if fvg is not None:
            sm.setdefault("fair_value_gaps", fvg)
        if bos is not None:
            sm.setdefault("market_structure", bos)


def _inject_stochastic_williams(event: dict) -> None:
    """
    Injeta stochastic (k, d, signal) e williams_r no technical_indicators_extended.

    Prioridade:
      1. InstitutionalAnalyticsEngine (technical_extras) — cálculo real com 14+ candles
      2. OHLC history da sessão (cálculo real com fórmulas reais)
      3. ATR-based range estimation
      4. Proxy RSI como último recurso
    """
    ti = event.setdefault("technical_indicators_extended", {})

    ia = event.get("institutional_analytics", {}) or {}
    tech_extras = ia.get("technical_extras", {}) or {}

    multi_tf = event.get("multi_tf", {}) or {}
    tf_1h = multi_tf.get("1h", {}) or {}
    current_price = float(
        event.get("preco_fechamento", 0)
        or _get_nested(event, "contextual_snapshot", "ohlc", "close", default=0)
        or 0
    )

    # Stochastic
    if "stochastic" not in ti and "stoch_rsi" not in ti:
        # P1: institutional_analytics
        stoch = tech_extras.get("stoch_rsi")
        if stoch and isinstance(stoch, dict) and "k" in stoch and "error" not in stoch:
            ti["stochastic"] = {
                "k": stoch["k"],
                "d": stoch["d"],
                "signal": (
                    "OVERBOUGHT" if stoch["k"] > 80
                    else "OVERSOLD" if stoch["k"] < 20
                    else "NEUTRAL"
                ),
                "source": "real",
            }
        else:
            # P2: OHLC history real
            real_stoch = _calc_stochastic_real()
            if real_stoch:
                ti["stochastic"] = real_stoch
            elif current_price > 0 and tf_1h.get("atr"):
                # P3: ATR estimation
                atr = float(tf_1h["atr"])
                ema_1h = float(tf_1h.get("mme_21", current_price))
                estimated_hh = ema_1h + 1.5 * atr
                estimated_ll = ema_1h - 1.5 * atr
                denom = estimated_hh - estimated_ll
                if denom > 0:
                    k_val = round(100 * (current_price - estimated_ll) / denom, 2)
                    k_val = max(0.0, min(100.0, k_val))
                    rsi_d = tf_1h.get("rsi_long")
                    d_val = round(rsi_d, 2) if rsi_d is not None else k_val
                    d_val = max(0.0, min(100.0, d_val))
                    ti["stochastic"] = {
                        "k": k_val,
                        "d": d_val,
                        "signal": (
                            "OVERBOUGHT" if k_val > 80
                            else "OVERSOLD" if k_val < 20
                            else "NEUTRAL"
                        ),
                        "source": "atr_estimated",
                    }
            else:
                # P4: RSI proxy
                rsi_k = tf_1h.get("rsi_short")
                rsi_d = tf_1h.get("rsi_long")
                if rsi_k is not None:
                    ti["stochastic"] = {
                        "k": round(rsi_k, 2),
                        "d": round(rsi_d, 2) if rsi_d is not None else round(rsi_k, 2),
                        "signal": (
                            "OVERBOUGHT" if rsi_k > 80
                            else "OVERSOLD" if rsi_k < 20
                            else "NEUTRAL"
                        ),
                        "source": "rsi_proxy",
                    }

    # Williams %R
    if "williams_r" not in ti:
        # P1: institutional_analytics
        wr = tech_extras.get("williams_r")
        if wr and isinstance(wr, dict) and "value" in wr and "error" not in wr:
            wr.setdefault("source", "real")
            ti["williams_r"] = wr
        else:
            # P2: OHLC history real
            real_wr = _calc_williams_r_real()
            if real_wr:
                ti["williams_r"] = real_wr
            elif current_price > 0 and tf_1h.get("atr"):
                # P3: ATR estimation
                atr = float(tf_1h["atr"])
                ema_1h = float(tf_1h.get("mme_21", current_price))
                estimated_hh = ema_1h + 1.5 * atr
                estimated_ll = ema_1h - 1.5 * atr
                denom = estimated_hh - estimated_ll
                if denom > 0:
                    wr_val = round(-100 * (estimated_hh - current_price) / denom, 2)
                    wr_val = max(-100.0, min(0.0, wr_val))
                    ti["williams_r"] = {
                        "value": wr_val,
                        "overbought": wr_val > -20,
                        "oversold": wr_val < -80,
                        "zone": (
                            "overbought" if wr_val > -20
                            else "oversold" if wr_val < -80
                            else "neutral"
                        ),
                        "source": "atr_estimated",
                    }
            else:
                # P4: RSI proxy
                rsi = tf_1h.get("rsi_short")
                if rsi is not None:
                    wr_proxy = round(-100 * (1 - rsi / 100), 2)
                    ti["williams_r"] = {
                        "value": wr_proxy,
                        "overbought": wr_proxy > -20,
                        "oversold": wr_proxy < -80,
                        "zone": (
                            "overbought" if wr_proxy > -20
                            else "oversold" if wr_proxy < -80
                            else "neutral"
                        ),
                        "source": "rsi_proxy",
                    }


# ---------------------------------------------------------------------------
# ONDA 2 — GARCH forecast (estimativa EWMA sem lib arch)
# ---------------------------------------------------------------------------

def _build_garch_forecast(event: dict) -> Optional[float]:
    """
    Estima a volatilidade da próxima hora usando EWMA (lambda=0.94) sobre
    o histórico de vols de sessão. Equivalente a um GARCH(1,1) simplificado.
    Retorna como fração (ex: 0.0065 = 0.65% volatilidade estimada em 1h).
    """
    if len(_STATE.vol_history) < 5:
        # Sem histórico suficiente — usa vol 1h do multi_tf como fallback
        multi_tf = event.get("multi_tf", {}) or {}
        tf_1h = multi_tf.get("1h", {}) or {}
        vol_1h = tf_1h.get("realized_vol")
        if vol_1h and vol_1h > 0:
            # Escala ligeiramente para cima (GARCH tende a mean-revert)
            return round(float(vol_1h) * 1.05, 6)
        return None

    # EWMA com lambda=0.94
    lam = 0.94
    vols = [v for _, v in _STATE.vol_history]

    # Variância inicial = variância da série
    n = len(vols)
    mean_v = sum(vols) / n
    var_ewma = sum((v - mean_v) ** 2 for v in vols) / n

    # Atualizar recursivamente com EWMA
    for v in vols:
        var_ewma = lam * var_ewma + (1 - lam) * v ** 2

    garch_vol = round(math.sqrt(max(var_ewma, 1e-12)), 6)
    return garch_vol


# ---------------------------------------------------------------------------
# ONDA 2 — ML Features: volatility_1h e order_book_slope
# ---------------------------------------------------------------------------

def _inject_ml_features(event: dict) -> None:
    """
    Injeta campos faltantes nas ml_features:
    - volatility_1h: vol realizada do TF 1h
    - order_book_slope: inclinação linear do livro de ofertas (price vs cum_vol)
    """
    ml = event.setdefault("ml_features", {})

    # volatility_1h
    pf = ml.setdefault("price_features", {})
    if "volatility_1h" not in pf:
        multi_tf = event.get("multi_tf", {}) or {}
        vol_1h = multi_tf.get("1h", {}).get("realized_vol")
        if vol_1h and vol_1h > 0:
            pf["volatility_1h"] = round(float(vol_1h), 6)

    # order_book_slope via order_book_depth (regressão linear simplificada)
    micro = ml.setdefault("microstructure", {})
    # FIX: Sempre recalcular order_book_slope a partir de dados atuais (não usar cache)
    # ou validar que os dados de order_book_depth mudaram desde o último cálculo
    ob_depth = event.get("order_book_depth", {}) or {}
    ob_depth_changed = (
        "order_book_slope" not in micro 
        or micro.get("order_book_slope") == 0.0
    )
    
    # Se temos dados de depth atuais, validar se mudaram
    if ob_depth and ob_depth_changed is False:
        # Verificar se depth atual é diferente do anterior (detectar cache obsoleto)
        current_l1_asks = ob_depth.get("L1", {}).get("asks", 0) or 0
        micro_cached_slope = micro.get("order_book_slope", 0.0)
        # Se o slope está exatamente igual ao valor anterior, pode ser cache
        if micro_cached_slope != 0.0 and current_l1_asks == 0:
            # Dados de depth não foram atualizados, não recalcular
            pass
        elif ob_depth != {}:
            # Há dados novos, recalcular mesmo que slope exista
            ob_depth_changed = True
    
    if ob_depth and (ob_depth_changed or "order_book_slope" not in micro):
        # Extrair pontos (nível, volume cumulativo)
        levels = []
        for lvl_key in ("L1", "L5", "L10", "L25"):
            lvl = ob_depth.get(lvl_key, {}) or {}
            asks = lvl.get("asks", 0) or 0
            bids = lvl.get("bids", 0) or 0
            if asks > 0 or bids > 0:
                levels.append((asks - bids))  # desequilíbrio cumulativo
        if len(levels) >= 2:
            # Slope = variação do desequilíbrio por nível de profundidade
            n = len(levels)
            xs = list(range(1, n + 1))
            x_mean = sum(xs) / n
            y_mean = sum(levels) / n
            num = sum((xs[i] - x_mean) * (levels[i] - y_mean) for i in range(n))
            den = sum((xs[i] - x_mean) ** 2 for i in range(n))
            slope_raw = round(num / den, 4) if den != 0 else 0.0
            # Normalizar pelo preço para comparabilidade entre ativos/períodos
            current_price = float(
                event.get("preco_fechamento", 0)
                or _get_nested(event, "contextual_snapshot", "ohlc", "close", default=0)
                or 0
            )
            if current_price > 0:
                new_slope = round(slope_raw / current_price, 6)
            else:
                new_slope = slope_raw
            
            # Detectar mudança gra and and log para auditoria
            old_slope = micro.get("order_book_slope", None)
            if old_slope is not None and old_slope != 0.0 and abs(new_slope - old_slope) > 0.01:
                logging.debug(f"order_book_slope recalculado: {old_slope:.6f} -> {new_slope:.6f}")
            
            micro["order_book_slope"] = new_slope
        else:
            micro.setdefault("order_book_slope", 0.0)
    else:
        micro.setdefault("order_book_slope", 0.0)


# ---------------------------------------------------------------------------
# FUNÇÃO PRINCIPAL — enrich_signal
# ---------------------------------------------------------------------------

def enrich_signal(
    event: dict,
    valid_window_data: Optional[List[Dict]] = None,
) -> dict:
    """
    Enriquece o sinal com todos os campos faltantes (Ondas 1 e 2).

    Args:
        event: O dicionário do sinal já montado pelo sistema principal.
        valid_window_data: Lista de trades da janela (para large_orders detection).

    Returns:
        O mesmo dict `event` modificado in-place com novos campos adicionados.
        Nunca substitui campos já existentes.
    """
    try:
        epoch_ms = int(event.get("epoch_ms", 0) or time.time() * 1000)
        current_price = float(
            event.get("preco_fechamento", 0)
            or _get_nested(event, "contextual_snapshot", "ohlc", "close", default=0)
            or 0
        )

        if current_price <= 0:
            logger.debug("[InstitutionalEnricher] Preço inválido, enriquecimento ignorado")
            return event

        dt_utc = datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc)

        # ------------------------------------------------------------------
        # 1. Atualiza históricos de sessão
        # ------------------------------------------------------------------
        _update_price_history(epoch_ms, current_price)

        # Registrar OHLC da janela para cálculo real de stochastic/williams_r
        _ohlc = _get_nested(event, "contextual_snapshot", "ohlc") or {}
        if not _ohlc:
            _ohlc = _get_nested(event, "enriched_snapshot", "ohlc") or {}
        if _ohlc and _ohlc.get("high") and _ohlc.get("low"):
            _STATE.ohlc_history.append({
                "open": float(_ohlc.get("open", _ohlc.get("close", current_price))),
                "high": float(_ohlc.get("high", current_price)),
                "low": float(_ohlc.get("low", current_price)),
                "close": float(_ohlc.get("close", current_price)),
            })

        # FIX 3.7: spread consolidado em orderbook_data (antes era spread_analysis separado)
        spread_bps = (
            _get_nested(event, "orderbook_data", "spread_bps", default=0)
            or _get_nested(event, "spread_analysis", "current_spread_bps", default=0)
            or 0
        )
        if spread_bps > 0:
            _update_spread_history(spread_bps)

        if valid_window_data:
            _register_large_trades(valid_window_data, current_price, epoch_ms)

        # ------------------------------------------------------------------
        # ONDA 1: Metadados
        # ------------------------------------------------------------------
        meta = _build_metadata_fields(event, epoch_ms)
        # Injeta no evento sem sobrescrever campos existentes
        event.setdefault("sequence_id", meta["sequence_id"])
        event.setdefault("primary_exchange", meta["primary_exchange"])
        event.setdefault("data_feed_type", meta["data_feed_type"])
        event.setdefault("data_quality_score", meta["data_quality_score"])
        event.setdefault("completeness_pct", meta["completeness_pct"])
        event.setdefault("reliability_score", meta["reliability_score"])
        event.setdefault("quality_components", meta["quality_components"])
        event.setdefault("quality_reasons", meta["quality_reasons"])

        # ------------------------------------------------------------------
        # ONDA 1: is_holiday no market_context
        # ------------------------------------------------------------------
        mc = event.setdefault("market_context", {})
        mc.setdefault("is_holiday", _is_holiday(dt_utc))

        # ------------------------------------------------------------------
        # ONDA 1: Bid/Ask, tick_direction, twap, previous_periods
        # ------------------------------------------------------------------
        price_fields = _build_price_fields(event, current_price, epoch_ms)
        event.setdefault("bid", price_fields.get("bid"))
        event.setdefault("ask", price_fields.get("ask"))
        event.setdefault("tick_direction", price_fields.get("tick_direction"))
        event.setdefault("twap", price_fields.get("twap"))
        if "previous_periods" in price_fields:
            event.setdefault("previous_periods", price_fields["previous_periods"])

        # ------------------------------------------------------------------
        # ONDA 1: Pivot Points + Suporte/Resistência imediatos
        # ------------------------------------------------------------------
        sr_data = _build_pivot_points(event)
        if sr_data.get("pivot_points"):
            event.setdefault("pivot_points", sr_data["pivot_points"])
        if sr_data.get("immediate_support"):
            event.setdefault("immediate_support", sr_data["immediate_support"])
            event.setdefault("support_strength", sr_data["support_strength"])
        if sr_data.get("immediate_resistance"):
            event.setdefault("immediate_resistance", sr_data["immediate_resistance"])
            event.setdefault("resistance_strength", sr_data["resistance_strength"])

        # ------------------------------------------------------------------
        # ONDA 1: Fibonacci padronizado
        # ------------------------------------------------------------------
        fib_data = _build_fibonacci(event, current_price)
        if fib_data.get("fibonacci_levels"):
            event.setdefault("fibonacci_levels", fib_data["fibonacci_levels"])

        # ------------------------------------------------------------------
        # ONDA 1: Spread volatility
        # FIX 3.4: Consolidado em orderbook_data (antes era spread_analysis separado)
        # ------------------------------------------------------------------
        spread_vol = _build_spread_volatility()
        if spread_vol is not None:
            ob = event.setdefault("orderbook_data", {})
            ob.setdefault("spread_volatility", spread_vol)

        # ------------------------------------------------------------------
        # ONDA 1: Slippage 1k/10k
        # ------------------------------------------------------------------
        small_slippage = _build_slippage_small(event)
        mi = event.setdefault("market_impact", {})
        sm = mi.setdefault("slippage_matrix", {})
        sm.setdefault("1k_usd", small_slippage["1k_usd"])
        sm.setdefault("10k_usd", small_slippage["10k_usd"])

        # ------------------------------------------------------------------
        # ONDA 1: Funding rate BTC explícito (contrato canônico percentual)
        # ------------------------------------------------------------------
        funding_btc = _build_btc_funding(event)
        if funding_btc is not None:
            deriv = event.setdefault("derivatives", {})
            btc_d = deriv.setdefault("BTCUSDT", {})
            btc_d.setdefault("funding_rate_percent", funding_btc)

        # ------------------------------------------------------------------
        # ONDA 2: Volume Profile avançado
        # FIX 3.3: REMOVIDO — dados já existem em historical_vp (fonte canônica).
        # volume_profile_advanced duplicava POC/VAH/VAL e não era consumido
        # por build_compact_payload nem ai_analyzer_qwen.
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # ONDA 2: Volatilidade consolidada
        # ------------------------------------------------------------------
        vol_metrics = _build_volatility_metrics(event, epoch_ms)
        if vol_metrics:
            event.setdefault("volatility_metrics", vol_metrics)

        # ------------------------------------------------------------------
        # ONDA 2: Passive flow explícito
        # ------------------------------------------------------------------
        passive_flow = _build_passive_flow(event)
        flow_root = event.setdefault("order_flow_extended", {})
        # B8-FIX: chaves opcionais (ausência permanece ausência).
        if "passive_buy_pct" in passive_flow:
            flow_root.setdefault(
                "passive_buy_pct",
                passive_flow["passive_buy_pct"],
            )
        if "passive_sell_pct" in passive_flow:
            flow_root.setdefault(
                "passive_sell_pct",
                passive_flow["passive_sell_pct"],
            )

        # ------------------------------------------------------------------
        # ONDA 2: Whale Activity (large orders + iceberg)
        # FIX 4.2: Se whale volume total é 0, forçar iceberg/hidden = 0
        # ------------------------------------------------------------------
        whale_act = _build_whale_activity(event, epoch_ms, current_price)
        if whale_act:
            existing_whale = event.get("whale_activity", {}) or {}
            merged_whale = {**whale_act, **existing_whale}
            # FIX 4.2: Consistência — sem volume de whale, sem detecção
            fc = event.get("fluxo_continuo", {})
            whale_buy = fc.get("whale_buy_volume", 0) or 0
            whale_sell = fc.get("whale_sell_volume", 0) or 0
            if whale_buy + whale_sell == 0:
                merged_whale["iceberg_activity"] = 0
                merged_whale["hidden_orders_detected"] = 0
            event["whale_activity"] = merged_whale

        # ------------------------------------------------------------------
        # ONDA 2: No-Man's Land
        # ------------------------------------------------------------------
        nml = _build_no_mans_land(event)
        if nml:
            ia = event.get("institutional_analytics", {}) or {}
            sr = ia.get("sr_analysis", {}) or {}
            dz = sr.get("defense_zones", {}) or {}
            dz.setdefault("no_mans_land", nml)

        # ------------------------------------------------------------------
        # ONDA 2: CCI
        # ------------------------------------------------------------------
        cci_val = _build_cci(event)
        if cci_val is not None:
            ti = event.setdefault("technical_indicators_extended", {})
            ti.setdefault("cci_1h", cci_val)
            # Sinal CCI
            if cci_val > 100:
                ti.setdefault("cci_signal", "OVERBOUGHT")
            elif cci_val < -100:
                ti.setdefault("cci_signal", "OVERSOLD")
            else:
                ti.setdefault("cci_signal", "NEUTRAL")

        # ------------------------------------------------------------------
        # ONDA 2: Stochastic e Williams %R (via institutional_analytics ou cálculo direto)
        # ------------------------------------------------------------------
        _inject_stochastic_williams(event)

        # ------------------------------------------------------------------
        # ONDA 2: Indicadores avançados (#23-28, #18, SMC)
        # ------------------------------------------------------------------
        _inject_advanced_indicators(event)

        # ------------------------------------------------------------------
        # ONDA 2: Active Patterns (padrões gráficos básicos)
        # ------------------------------------------------------------------
        active_pats = _build_active_patterns(event, current_price)
        if active_pats:
            pr = event.setdefault("pattern_recognition", {})
            pr.setdefault("active_patterns", active_pats)

        # ------------------------------------------------------------------
        # ONDA 2: GARCH forecast aproximado
        # ------------------------------------------------------------------
        garch = _build_garch_forecast(event)
        if garch is not None:
            ti = event.setdefault("technical_indicators_extended", {})
            ti.setdefault("garch_forecast_1h", garch)

        # ------------------------------------------------------------------
        # ONDA 2: ML features faltantes (volatility_1h, order_book_slope)
        # ------------------------------------------------------------------
        _inject_ml_features(event)

        # ------------------------------------------------------------------
        # ONDA 1: backup_exchanges no evento
        # ------------------------------------------------------------------
        event.setdefault("backup_exchanges", ["COINBASE", "KRAKEN", "OKX"])

        # ------------------------------------------------------------------
        # ONDA 2: Alertas estruturados (SEMPRE ao final para ter todos os campos)
        # ------------------------------------------------------------------
        alert_data = _build_alerts(event, current_price, epoch_ms)
        event.setdefault("alerts", alert_data)

        # ------------------------------------------------------------------
        # ONDA 3: Price Targets Probabilísticos
        # FIX 4.3: Só incluir se model_confidence >= 0.20 (abaixo disso é ruído)
        # ------------------------------------------------------------------
        price_targets = _build_price_targets_probabilistic(event, current_price)
        if price_targets and price_targets.get("model_confidence", 0) >= 0.20:
            event.setdefault("price_targets", price_targets)

        # ------------------------------------------------------------------
        # ONDA 3: Regime Detection Probabilístico
        # ------------------------------------------------------------------
        regime_analysis = _build_regime_probabilities(event)
        if regime_analysis:
            event.setdefault("regime_analysis", regime_analysis)

        # ------------------------------------------------------------------
        # FIX 4.5: Data reliability flags
        # FIX (ETAPA 2): data_reliability é RESUMO — deve derivar da fonte
        # canônica de latência (institutional_analytics.quality.latency, que
        # vem de time_manager.track_data_latency). Antes lia
        # event["quality"] no TOP-LEVEL, que NUNCA é populado (a qualidade
        # vive em institutional_analytics.quality), e com default fail-open
        # `_latency.get("is_acceptable", 1)` — resultado: latency_acceptable
        # SEMPRE true, mesmo com latency_ms=7312/POOR (JANELAS 1 e 4).
        # Novo contrato: latência desconhecida -> False (fail-closed).
        # ------------------------------------------------------------------
        _aa = _get_nested(event, "raw_event", "advanced_analysis") or {}
        _latency = _get_nested(event, "institutional_analytics", "quality", "latency") or {}
        _latency_known = bool(_latency) and _latency.get("latency_ms") is not None
        # PFIX-LOW: coverage deriva EXCLUSIVAMENTE de onchain_field_status
        # (P04). Freshness (fresh/stale/...) responde "quão recente é o
        # snapshot"; coverage responde "quanto do conjunto suportado está
        # utilizável". Ortogonais por desenho: mesma field_status com status
        # fresh ou stale gera a MESMA coverage.
        _fs_cov = _aa.get("onchain_field_status")
        _oc_coverage, _oc_coverage_pct, _, _ = _derive_onchain_coverage(_fs_cov)
        event["data_reliability"] = {
            "has_options_data": bool(_aa.get("options_metrics", {}).get("is_real_data")),
            "onchain_coverage": _oc_coverage,
            "latency_acceptable": bool(_latency.get("is_acceptable")) if _latency_known else False,
            "price_targets_available": "price_targets" in event,
        }
        if _oc_coverage_pct is not None:
            event["data_reliability"]["onchain_coverage_pct"] = _oc_coverage_pct

    except Exception as exc:
        logger.error(
            "[InstitutionalEnricher] Erro no enrich_signal: %s",
            exc,
            exc_info=True,
        )

    return event
