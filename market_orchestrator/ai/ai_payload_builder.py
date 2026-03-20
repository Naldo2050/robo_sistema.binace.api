# market_orchestrator/ai/ai_payload_builder.py
# -*- coding: utf-8 -*-
"""
Construtor de Payload para Análise de IA.

Este módulo é responsável por padronizar e organizar os dados brutos e métricas
do sistema em um formato estruturado e semântico para consumo pelos modelos de IA.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone
from pathlib import Path
import logging
import json

_ML_EXTREME_THRESHOLD_HIGH = 0.95
_ML_EXTREME_THRESHOLD_LOW = 0.05
import hashlib
import os
from functools import lru_cache

import yaml

# Import do otimizador de payload (localizado em src/utils/)
from src.utils.ai_payload_optimizer import AIPayloadOptimizer, compact_historical_vp

from market_orchestrator.ai.ai_enrichment_context import build_enriched_ai_context
from market_orchestrator.ai.payload_compressor import compress_payload
from market_orchestrator.ai.payload_section_cache import SectionCache, canonical_ref, is_fresh
from market_orchestrator.ai.payload_metrics_aggregator import append_metric_line


def _check_in_range(price, low, high):
    """Helper simples para verificar se preço está em range."""
    if price is None or low is None or high is None:
        return None
    return low <= price <= high


def _strip_empty(d: dict) -> dict:
    """Remove recursivamente chaves com valor None, {}, [] ou string vazia."""
    if not isinstance(d, dict):
        return d
    cleaned = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, dict):
            v = _strip_empty(v)
            if not v:  # dict vazio após limpeza
                continue
        if isinstance(v, list) and len(v) == 0:
            continue
        if isinstance(v, str) and v == "":
            continue
        cleaned[k] = v
    return cleaned


def _safe_epoch_ms(signal: dict) -> int:
    """
    Extrai epoch_ms do signal garantindo que retorna SEMPRE um INT válido.
    
    Problema resolvido:
        - signal["timestamp"] pode ser STRING "2026-03-09 21:32:00-04:00"
        - Python avalia strings não-vazias como TRUTHY
        - O 'or' encadeado retornava a STRING em vez do fallback correto
        - O compressor não conseguia usar STRING como int → epoch_ms virava None
    
    Estratégia de busca (ordem de prioridade):
        1. Campos diretos que são int/float (epoch_ms, timestamp_ms, etc.)
        2. Dentro de fluxo_continuo.time_index.epoch_ms
        3. Converter timestamp string ISO para int
        4. Fallback: timestamp atual em ms
    
    Args:
        signal (dict): O dicionário do sinal/evento bruto
        
    Returns:
        int: Timestamp em milissegundos, NUNCA None
    """
    # ── 1. Campos diretos com valor numérico válido ──────────────────
    # Considera válido: número > 1_000_000_000_000 (= ano 2001 em ms)
    # Isso filtra zeros, negativos e timestamps em segundos por engano
    for key in ("epoch_ms", "timestamp_ms", "window_close_ms", "window_id"):
        val = signal.get(key)
        if isinstance(val, (int, float)) and val > 1_000_000_000_000:
            return int(val)

    # ── 2. Dentro de fluxo_continuo.time_index ──────────────────────
    # O fluxo_continuo sempre tem time_index.epoch_ms correto
    fc = signal.get("fluxo_continuo", {})
    if isinstance(fc, dict):
        ti = fc.get("time_index", {})
        if isinstance(ti, dict):
            val = ti.get("epoch_ms")
            if isinstance(val, (int, float)) and val > 1_000_000_000_000:
                return int(val)

    # ── 3. Converter timestamp string ISO para int ───────────────────
    # Tenta converter strings como "2026-03-10T01:32:00.000Z"
    for key in ("timestamp_utc", "timestamp"):
        val = signal.get(key)
        if isinstance(val, str) and val:
            try:
                # Normaliza formato Z → +00:00 para fromisoformat()
                dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
                result = int(dt.timestamp() * 1000)
                if result > 1_000_000_000_000:
                    logging.debug(
                        "EPOCH_MS_FROM_STRING key=%s val=%s result=%s",
                        key, val, result
                    )
                    return result
            except Exception:
                pass  # Tenta próximo campo

    # ── 4. Fallback: timestamp atual ─────────────────────────────────
    # Só chega aqui se NENHUM campo de timestamp estiver disponível
    fallback = int(datetime.now(timezone.utc).timestamp() * 1000)
    logging.warning(
        "EPOCH_MS_FALLBACK_NOW signal_keys=%s fallback_ms=%s",
        list(signal.keys()) if isinstance(signal, dict) else "N/A",
        fallback
    )
    return fallback


def _inject_institutional_analytics(
    ai_payload: Dict[str, Any],
    signal: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Mescla dados do InstitutionalAnalyticsEngine nas seções
    existentes do payload, de forma comprimida.
    
    Adiciona ~80-120 tokens extras ao payload com informações de
    altíssimo valor analítico.
    """
    ia = signal.get("institutional_analytics")
    if not ia or not isinstance(ia, dict) or ia.get("status") != "ok":
        return ai_payload

    # ═══════════════════════════════════════════════════════
    # 1. TECHNICAL INDICATORS — StochRSI, WilliamsR, Candles
    # ═══════════════════════════════════════════════════════
    tech = ai_payload.get("technical_indicators")
    if isinstance(tech, dict):
        extras = ia.get("technical_extras", {})

        # StochRSI
        sr = extras.get("stoch_rsi", {})
        if sr and "error" not in sr:
            tech["stoch_rsi_k"] = sr.get("k")
            tech["stoch_rsi_d"] = sr.get("d")
            tech["stoch_rsi_signal"] = sr.get("crossover", "none")

        # Williams %R
        wr = extras.get("williams_r", {})
        if wr and "error" not in wr:
            tech["williams_r"] = wr.get("value")
            tech["williams_r_zone"] = wr.get("zone")

        # Candlestick patterns
        candles = ia.get("candlestick_patterns", {})
        if candles and candles.get("patterns_detected", 0) > 0:
            patterns = candles.get("patterns", [])
            # Acesso seguro com verificação de tipo
            p0 = patterns[0] if isinstance(patterns, list) and patterns else None
            p0 = p0 if isinstance(p0, dict) else {}
            tech["candle_pattern"] = p0.get("name")
            tech["candle_pattern_type"] = p0.get("type")
            tech["candle_pattern_conf"] = p0.get("confidence")
            if len(patterns) > 1:
                p1 = patterns[1] if isinstance(patterns[1], dict) else {}
                tech["candle_pattern_2"] = p1.get("name")

    # ═══════════════════════════════════════════════════════
    # 2. PRICE CONTEXT — TWAP, PoorH/L, Shape, RefPrices
    # ═══════════════════════════════════════════════════════
    price = ai_payload.get("price_context")
    if isinstance(price, dict):
        extras = ia.get("technical_extras", {})
        profile = ia.get("profile_analysis", {})
        sr_data = ia.get("sr_analysis", {})

        # TWAP vs VWAP
        twap = extras.get("twap_analysis", {})
        if twap and "error" not in twap:
            price["twap"] = twap.get("twap")
            price["twap_vwap_div"] = twap.get("divergence_pct")
            price["twap_signal"] = twap.get("signal")

        # Poor High/Low
        poor = profile.get("poor_extremes", {})
        if poor and poor.get("status") == "success":
            ph = poor.get("poor_high", {})
            pl = poor.get("poor_low", {})
            if ph.get("detected") or pl.get("detected"):
                price["poor_high"] = ph.get("detected", False)
                price["poor_low"] = pl.get("detected", False)
                price["auction_bias"] = poor.get("action_bias")

        # Profile Shape
        shape = profile.get("profile_shape", {})
        if shape and shape.get("status") == "success":
            price["profile_shape"] = shape.get("shape")
            price["profile_signal"] = shape.get("trading_signal")

        # Value Area Volume %
        va_pct = profile.get("va_volume_pct", {})
        if va_pct and "error" not in va_pct:
            pct = va_pct.get("value_area_volume_pct")
            if pct:
                price["va_volume_pct"] = pct
                price["va_compression"] = va_pct.get("compression_signal", False)

        # Reference Prices (prev day close + distance)
        refs = sr_data.get("reference_prices", {})
        if refs and refs.get("status") == "ok":
            ref_prices = refs.get("reference_prices", {})
            summary = refs.get("summary", {})

            prev_day = ref_prices.get("prev_day", {})
            if prev_day:
                price["prev_day_close"] = prev_day.get("close")
                price["above_prev_day"] = prev_day.get("above_prev_close")
                price["dist_prev_day_pct"] = prev_day.get("distance_from_close_pct")

            prev_week = ref_prices.get("prev_week", {})
            if prev_week:
                price["prev_week_close"] = prev_week.get("close")
                price["above_prev_week"] = prev_week.get("above_prev_close")

    # ═══════════════════════════════════════════════════════
    # 3. FLOW CONTEXT — BuySellRatio, Passive/Agg, Whale
    # ═══════════════════════════════════════════════════════
    flow = ai_payload.get("flow_context")
    if isinstance(flow, dict):
        flow_data = ia.get("flow_analysis", {})

        # Buy/Sell Ratio
        bsr = flow_data.get("buy_sell_ratio", {})
        if bsr and "error" not in bsr:
            flow["buy_sell_ratio"] = bsr.get("buy_sell_ratio")
            flow["pressure"] = bsr.get("pressure")
            flow["flow_trend"] = bsr.get("flow_trend")

        # Passive/Aggressive
        pa = flow_data.get("passive_aggressive", {})
        if pa and pa.get("status") == "success":
            composite = pa.get("composite", {})
            flow["passive_agg_signal"] = composite.get("signal")
            flow["passive_agg_conviction"] = composite.get("conviction")

        # Whale Accumulation Score
        whale = flow_data.get("whale_accumulation", {})
        if whale and whale.get("status") == "success":
            flow["whale_score"] = whale.get("score")
            flow["whale_class"] = whale.get("classification")
            flow["whale_bias"] = whale.get("bias")

        # Absorption Zones summary
        abs_zones = flow_data.get("absorption_zones", {})
        if abs_zones and abs_zones.get("status") == "success":
            strongest = abs_zones.get("strongest_zone")
            if strongest:
                flow["top_absorption_zone"] = strongest.get("center")
                flow["top_absorption_side"] = strongest.get("dominant_side")
                flow["absorption_zones_count"] = abs_zones.get("total_zones", 0)

    # ═══════════════════════════════════════════════════════
    # 4. ORDERBOOK CONTEXT — Spread Percentile
    # ═══════════════════════════════════════════════════════
    ob = ai_payload.get("orderbook_context")
    if isinstance(ob, dict):
        quality = ia.get("quality", {})
        sp = quality.get("spread_percentile", {})
        if sp and sp.get("status") == "ok":
            ob["spread_percentile"] = sp.get("spread_percentile")
            ob["liquidity_signal"] = sp.get("liquidity_signal")

    # ═══════════════════════════════════════════════════════
    # 5. MACRO CONTEXT — Calendar
    # ═══════════════════════════════════════════════════════
    macro = ai_payload.get("macro_context")
    if isinstance(macro, dict):
        quality = ia.get("quality", {})
        cal = quality.get("calendar", {})
        if cal and "error" not in cal:
            if cal.get("liquidity_warning"):
                macro["holiday_warning"] = True
                macro["expected_liquidity"] = cal.get("expected_liquidity")
                macro["holiday_name"] = cal.get("holiday_name")
            macro["day_of_week"] = cal.get("day_of_week")

    # ═══════════════════════════════════════════════════════
    # 6. NEW: SR CONTEXT (compact)
    # ═══════════════════════════════════════════════════════
    sr_data = ia.get("sr_analysis", {})
    sr_context = {}

    # S/R Strength — derivar nearest de levels[] (sem duplicar arrays)
    sr_str = sr_data.get("sr_strength", {})
    if sr_str and sr_str.get("status") == "success":
        levels = sr_str.get("levels", [])
        # Derivar nearest support/resistance direto de levels[] por type
        nearest_sup = next((l for l in levels if l.get("type") == "support"), None)
        nearest_res = next((l for l in levels if l.get("type") == "resistance"), None)
        # Fallback: nearest_support/nearest_resistance antigos (backward compat)
        nearest_sup = nearest_sup or sr_str.get("nearest_support")
        nearest_res = nearest_res or sr_str.get("nearest_resistance")
        if nearest_sup:
            sr_context["nearest_support"] = nearest_sup.get("price")
            sr_context["support_strength"] = nearest_sup.get("strength")
            sr_context["support_sources"] = nearest_sup.get("confluence_count", 1)
        if nearest_res:
            sr_context["nearest_resistance"] = nearest_res.get("price")
            sr_context["resistance_strength"] = nearest_res.get("strength")
            sr_context["resistance_sources"] = nearest_res.get("confluence_count", 1)

    # Defense Zones — strongest buy/sell defense
    dz = sr_data.get("defense_zones", {})
    if dz and dz.get("status") == "success":
        sb = dz.get("strongest_buy")
        ss = dz.get("strongest_sell")
        if sb:
            sr_context["buy_defense_price"] = sb.get("center")
            sr_context["buy_defense_strength"] = sb.get("strength")
            sr_context["buy_defense_type"] = sb.get("type")
        if ss:
            sr_context["sell_defense_price"] = ss.get("center")
            sr_context["sell_defense_strength"] = ss.get("strength")
            sr_context["sell_defense_type"] = ss.get("type")
        asym = dz.get("defense_asymmetry", {})
        if asym:
            sr_context["defense_bias"] = asym.get("bias")

    # No-Man's Land
    profile = ia.get("profile_analysis", {})
    nml = profile.get("no_mans_land", {})
    if nml and nml.get("status") == "success":
        if nml.get("price_in_no_mans_land"):
            sr_context["in_no_mans_land"] = True
            sr_context["nml_warning"] = nml.get("warning")
        nearest_nml = nml.get("nearest_no_mans_land")
        if nearest_nml:
            sr_context["nearest_nml_range"] = [
                nearest_nml.get("range_low"),
                nearest_nml.get("range_high"),
            ]
            sr_context["nearest_nml_risk"] = nearest_nml.get("risk")

    # HVN/LVN Strength — derivar de scored_hvns[0] (sem campo duplicado)
    vns = profile.get("volume_node_strength", {})
    if vns and vns.get("status") == "success":
        scored_hvns = vns.get("scored_hvns", [])
        strongest_hvn = scored_hvns[0] if scored_hvns else vns.get("strongest_hvn")
        if strongest_hvn:
            sr_context["strongest_hvn"] = strongest_hvn.get("price")
            sr_context["strongest_hvn_str"] = strongest_hvn.get("strength")
            sr_context["hvn_multi_tf"] = strongest_hvn.get("multi_tf_confluence", False)

    if sr_context:
        ai_payload["sr_context"] = sr_context

    # ═══════════════════════════════════════════════════════
    # 7. ANOMALY ALERT — Só se risco elevado
    # ═══════════════════════════════════════════════════════
    quality = ia.get("quality", {})
    anomalies = quality.get("anomalies", {})
    if anomalies and anomalies.get("risk_elevated"):
        alert_list = anomalies.get("anomalies", [])
        critical = [a for a in alert_list if a.get("severity") in ("CRITICAL", "HIGH")]
        if critical:
            ai_payload["anomaly_alert"] = {
                "count": len(critical),
                "types": [a["type"] for a in critical[:3]],
                "max_severity": anomalies.get("max_severity"),
                "summary": anomalies.get("summary"),
            }

    return ai_payload


# Configuração de payload (feature flags)
DEFAULT_LLM_PAYLOAD_CONFIG = {
    "v2_enabled": True,
    "max_bytes": 6144,
    "section_budgets_enabled": True,
    "section_cache_enabled": True,
    "cache_ttls_s": {
        "macro_context": 1800,
        "cross_asset_context": 1800,
    },
    "guardrail_hard_enabled": True,
}

_FLAGS_LOGGED = False
_METRICS_PATH = Path("logs") / "payload_metrics.jsonl"


@lru_cache(maxsize=1)
def get_llm_payload_config() -> Dict[str, Any]:
    cfg = dict(DEFAULT_LLM_PAYLOAD_CONFIG)
    config_path = Path("config/model_config.yaml")
    try:
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            llm_cfg = (loaded or {}).get("llm_payload") or {}
            if isinstance(llm_cfg, dict):
                cfg.update(llm_cfg)
    except Exception as e:
        logging.warning("Não foi possível carregar config llm_payload: %s", e)
    return cfg


def _append_payload_metric(metric: Dict[str, Any]) -> None:
    append_metric_line(metric, str(_METRICS_PATH))


# Cross-asset features são calculadas UMA VEZ em generate_ml_features()
# e lidas do payload — não importar/recalcular aqui.

# Import para análise de regime
try:
    from src.data.macro_data_provider import MacroDataProvider
    from src.analysis.regime_detector import EnhancedRegimeDetector
except ImportError as e:
    MacroDataProvider = None
    EnhancedRegimeDetector = None
    print(f"⚠️ Não foi possível importar MacroDataProvider ou EnhancedRegimeDetector: {e}")

def add_enriched_context_to_ai_payload(ai_payload: Dict[str, Any], raw_event: Dict[str, Any]) -> Dict[str, Any]:
    """
    [V5] REMOVIDO: Injeção redundante de raw_event desativada para evitar bloat de 60KB.
    O ai_payload construído em build_ai_input já contém as informações necessárias.
    """
    return ai_payload

def build_ai_input(
    symbol: str,
    signal: Dict[str, Any],
    enriched: Dict[str, Any],
    flow_metrics: Dict[str, Any],
    historical_profile: Dict[str, Any],
    macro_context: Dict[str, Any],
    market_environment: Dict[str, Any],
    orderbook_data: Dict[str, Any],
    ml_features: Dict[str, Any],
    ml_prediction: Optional[Dict[str, Any]] = None,
    pivots: Optional[Dict[str, Any]] = None,
    skip_v2_compress: bool = False,
) -> Dict[str, Any]:
    """
    Constrói um dicionário estruturado e limpo para o analisador de IA.
    
    Organiza os dados em seções contextuais (Preço, Fluxo, Orderbook, Macro, etc.)
    e mantém chaves de compatibilidade na raiz para não quebrar templates existentes.

    Args:
        symbol (str): Símbolo do ativo (ex: BTCUSDT).
        signal (dict): O evento/sinal base (contém tipo, descrição, timestamps).
        enriched (dict): Dados enriquecidos do pipeline (OHLC, métricas básicas).
        flow_metrics (dict): Métricas do FlowAnalyzer (CVD, Whales, Heatmap).
        historical_profile (dict): Volume Profile histórico (POC, VAH, VAL).
        macro_context (dict): Contexto de sessão, horários, feriados.
        market_environment (dict): Regime de mercado, correlações.
        orderbook_data (dict): Snapshot e métricas do livro de ofertas.
        ml_features (dict): Features quantitativas para ML.
        ml_prediction (Optional[dict]): Previsão do modelo ML (injetada pelo ai_runner).

    Returns:
        dict: Payload completo e organizado para a IA.
    """

    cfg = get_llm_payload_config()
    global _FLAGS_LOGGED
    if not _FLAGS_LOGGED:
        logging.info(
            "PAYLOAD_FEATURE_FLAGS v2_enabled=%s max_bytes=%s cache_enabled=%s budgets_enabled=%s guardrail_enabled=%s",
            cfg.get("v2_enabled", True),
            cfg.get("max_bytes", 6144),
            cfg.get("section_cache_enabled", True),
            cfg.get("section_budgets_enabled", True),
            cfg.get("guardrail_hard_enabled", True),
        )
        _FLAGS_LOGGED = True

    # Garante que ml_features sempre seja um dicionário
    if not isinstance(ml_features, dict):
        ml_features = {}
    else:
        ml_features = ml_features or {}

    # 1. Contexto de Preço (Price Context)
    ohlc = enriched.get("ohlc", {})
    vp_daily = historical_profile.get("daily", {})

    # Compacta HVNs/LVNs para reduzir custo sem perder niveis relevantes
    try:
        current_price_for_vp = signal.get("preco_fechamento", ohlc.get("close"))
        current_price_for_vp = float(current_price_for_vp) if current_price_for_vp is not None else None
    except Exception:
        current_price_for_vp = None

    vp_compact_daily: Dict[str, Any] = {}
    try:
        if isinstance(historical_profile, dict):
            vp_compact = compact_historical_vp(
                historical_profile,
                current_price=current_price_for_vp,
                pct_range=0.05,
                max_levels=5,
                timeframes=("daily",),
            )
            vp_compact_daily = vp_compact.get("daily", {}) if isinstance(vp_compact, dict) else {}
    except Exception:
        vp_compact_daily = {}
    
    # Cálculos de Price Action (Candle)
    pa_metrics = {
        "candle_range_pct": 0.0,
        "candle_body_pct": 0.0,
        "upper_shadow_pct": 0.0,
        "lower_shadow_pct": 0.0,
        "close_position": 0.5
    }
    
    try:
        op = ohlc.get("open")
        hi = ohlc.get("high")
        lo = ohlc.get("low")
        cl = ohlc.get("close")
        
        if all(x is not None and x > 0 for x in [op, hi, lo, cl]):
            # Range total
            rng = hi - lo
            if op > 0:
                pa_metrics["candle_range_pct"] = (rng / op) * 100
                
            # Corpo
            body = abs(cl - op)
            if op > 0:
                pa_metrics["candle_body_pct"] = (body / op) * 100
            
            # Posição do fechamento (0.0 = Low, 1.0 = High)
            if rng > 0:
                pa_metrics["close_position"] = (cl - lo) / rng
            
            # Sombras
            upper_shadow = hi - max(op, cl)
            lower_shadow = min(op, cl) - lo
            if rng > 0: # percentual do range total
                 pa_metrics["upper_shadow_pct"] = (upper_shadow / rng) * 100
                 pa_metrics["lower_shadow_pct"] = (lower_shadow / rng) * 100
                 
    except Exception:
        pass # Mantém defaults seguros

    volume_profile_daily = {
        "poc": vp_daily.get("poc"),
        "vah": vp_daily.get("vah"),
        "val": vp_daily.get("val"),
        "in_value_area": _check_in_range(ohlc.get("close"), vp_daily.get("val"), vp_daily.get("vah")),
    }
    hvns_nearby = vp_compact_daily.get("hvns_nearby")
    if isinstance(hvns_nearby, list) and hvns_nearby:
        volume_profile_daily["hvns_nearby"] = hvns_nearby
    lvns_nearby = vp_compact_daily.get("lvns_nearby")
    if isinstance(lvns_nearby, list) and lvns_nearby:
        volume_profile_daily["lvns_nearby"] = lvns_nearby

    price_context = {
        "current_price": signal.get("preco_fechamento", ohlc.get("close")),
        "ohlc": {
            "open": ohlc.get("open"),
            "high": ohlc.get("high"),
            "low": ohlc.get("low"),
            "close": ohlc.get("close"),
            "vwap": ohlc.get("vwap")
        },
        "price_action": pa_metrics, # 🆕 Bloco de Price Action explícito
        "volume_profile_daily": volume_profile_daily,
        "volatility": {
            "atr": macro_context.get("atr"),  # Se disponível no macro
            "volatility_regime": market_environment.get("volatility_regime")
        }
    }

    # 2. Contexto de Fluxo (Flow Context)
    order_flow = flow_metrics.get("order_flow", {})
    whale_flow = {
        "whale_delta": flow_metrics.get("whale_delta", 0),
        "whale_buy_vol": flow_metrics.get("whale_buy_volume", 0),
        "whale_sell_vol": flow_metrics.get("whale_sell_volume", 0)
    }
    
    flow_context = {
        "net_flow": order_flow.get("net_flow_1m"),
        "cvd_accumulated": flow_metrics.get("cvd"),
        "flow_imbalance": order_flow.get("flow_imbalance"),
        "aggressive_buyers": order_flow.get("aggressive_buy_pct"),
        # aggressive_sellers omitido (= 100 - aggressive_buyers)
        "whale_activity": whale_flow,
        "absorption_type": flow_metrics.get("tipo_absorcao", "Neutra")
    }

    # 3. Contexto de Orderbook (Liquidez)
    # Tenta normalizar dados que podem vir de estruturas diferentes
    spread_metrics = signal.get("spread_metrics", {})
    # 🆕 Tenta extrair depth completo se disponível
    depth_metrics = orderbook_data.get("depth_metrics", {})
    
    ob_context = {
        "bid_depth_usd": orderbook_data.get("bid_depth_usd") or spread_metrics.get("bid_depth_usd"),
        "ask_depth_usd": orderbook_data.get("ask_depth_usd") or spread_metrics.get("ask_depth_usd"),
        "imbalance": orderbook_data.get("imbalance"),
        "spread_percent": orderbook_data.get("spread_percent") or spread_metrics.get("spread_percent"),
        "market_impact_score": orderbook_data.get("pressure"), # Proxy se existir
        "walls_detected": len(signal.get("order_book_depth", {})) > 0, # Simplificação
        # 🆕 Métricas de profundidade explícitas
        "depth_metrics": {
             "bid_liquidity_top5": depth_metrics.get("bid_liquidity_top5", 0),
             "ask_liquidity_top5": depth_metrics.get("ask_liquidity_top5", 0),
             "depth_imbalance": depth_metrics.get("depth_imbalance", 0)
        }
    }

    # 🆕 3.5. Indicadores Técnicos (Technical Indicators)
    # Extrai indicadores chave do contexto MTF (prioriza 1h ou 4h para trend, 15m para tático)
    mtf = (
        macro_context.get("mtf_trends")
        or macro_context.get("multi_timeframe_trends")
        or macro_context.get("multi_tf")
        or signal.get("multi_tf")
        or signal.get("multi_timeframe_trends")
        or {}
    )
    # Tenta pegar contexto de 1h para indicadores 'padrão', fallback para 4h/15m
    tf_tech = mtf.get("1h") or mtf.get("4h") or mtf.get("15m") or {}

    rsi_value = tf_tech.get("rsi_short") or tf_tech.get("rsi") or tf_tech.get("rsi_14")
    macd_line = tf_tech.get("macd") or tf_tech.get("macd_line")
    macd_signal = tf_tech.get("macd_signal") or tf_tech.get("signal")
    adx_value = tf_tech.get("adx") or tf_tech.get("adx_14")
    stoch_value = tf_tech.get("stoch_k") or tf_tech.get("stoch") or tf_tech.get("stoch_14")
    
    technical_indicators = {
        "rsi": rsi_value, # 14 period default
        "macd": {
            "line": macd_line,
            "signal": macd_signal,
            "histogram": round((macd_line or 0) - (macd_signal or 0), 4)
        },
        "adx": adx_value,
        "stoch": stoch_value, # Se existir no MTF
        "pivots": pivots or {}
    }

    # 4. Contexto Cross-Asset
    # Usa cross_asset já calculado em generate_ml_features() — não recalcular
    cross_asset_data = ml_features.get("cross_asset", {})

    cross_asset_context = {}
    if cross_asset_data:
        cross_asset_context = {
            "btc_eth_correlations": {
                "short_term_7d": cross_asset_data.get("btc_eth_corr_7d"),
                "long_term_30d": cross_asset_data.get("btc_eth_corr_30d"),
            },
            "btc_dxy_correlations": {
                "medium_term_30d": cross_asset_data.get("btc_dxy_corr_30d"),
                "long_term_90d": cross_asset_data.get("btc_dxy_corr_90d"),
            },
            "btc_ndx_correlations": {
                "medium_term_30d": cross_asset_data.get("btc_ndx_corr_30d"),
            },
            "dxy_momentum": {
                "return_5d": cross_asset_data.get("dxy_return_5d"),
                "return_20d": cross_asset_data.get("dxy_return_20d"),
            },
        }
    else:
        cross_asset_context = {
            "btc_eth_correlations": {
                "short_term_7d": cross_asset_data.get("btc_eth_corr_7d"),
                "long_term_30d": cross_asset_data.get("btc_eth_corr_30d"),
            },
            "btc_dxy_correlations": {
                "medium_term_30d": cross_asset_data.get("btc_dxy_corr_30d"),
                "long_term_90d": cross_asset_data.get("btc_dxy_corr_90d"),
            },
            "btc_ndx_correlations": {
                "medium_term_30d": cross_asset_data.get("btc_ndx_corr_30d"),
            },
            "dxy_momentum": {
                "return_5d": cross_asset_data.get("dxy_return_5d"),
                "return_20d": cross_asset_data.get("dxy_return_20d"),
            },
        }

    # 5. Contexto Macro e Regime
    macro_full_context = {
        "session": macro_context.get("trading_session"),
        "phase": macro_context.get("session_phase"),
        "multi_timeframe_trends": mtf,
        "regime": {
            "structure": market_environment.get("market_structure"),
            "trend": market_environment.get("trend_direction"),
            "sentiment": market_environment.get("risk_sentiment")
        },
        "correlations": {
            "sp500": market_environment.get("correlation_spy"),
            "dxy": market_environment.get("correlation_dxy")
        }
    }

    # 6. Análise de Regime com EnhancedRegimeDetector
    regime_analysis = {}
    if MacroDataProvider and EnhancedRegimeDetector:
        try:
            # Inicializa o detector de regime (stateful)
            regime_detector = EnhancedRegimeDetector()

            # Extrai dados reais de mercados externos do contexto do coletor
            _ext = macro_context.get("external", {})
            _cross = ml_features.get("cross_asset", {}) if isinstance(ml_features, dict) else {}

            def _ext_price(key: str):
                """Retorna preco_atual do mercado externo ou None."""
                entry = _ext.get(key)
                if isinstance(entry, dict):
                    return entry.get("preco_atual")
                return None

            _dxy_val   = _ext_price("DXY")
            _tnx_val   = _ext_price("TNX")
            _gold_val  = _ext_price("GOLD")
            _wti_val   = _ext_price("WTI")
            _sp500_val = _ext_price("SP500")
            _vix_val   = _ext_price("VIX")

            # Fear & Greed real (alternative.me)
            _fng_entry = _ext.get("FEAR_GREED", {})
            _fng_val   = _fng_entry.get("preco_atual") if isinstance(_fng_entry, dict) else None

            # yield_spread: 10Y - 2Y (usa TNX como proxy do 10Y quando disponível)
            _t10y = _tnx_val
            _t2y  = None  # não coletado diretamente
            _yield_spread = (_t10y - _t2y) if (_t10y and _t2y) else None

            # dxy_momentum: usa retorno 5d do cross_asset já calculado
            _dxy_mom = _cross.get("dxy_return_5d") or _cross.get("dxy_momentum")

            macro_data = {
                "vix": _vix_val,           # real via yfinance ^VIX (delay 15min, suficiente)
                "fear_greed_index": _fng_val,  # real via alternative.me (gratuito)
                "treasury_10y": _t10y,
                "treasury_2y": _t2y,
                "dxy": _dxy_val,
                "gold": _gold_val,
                "oil": _wti_val,
                "btc_dominance": None,     # requer CoinGecko/CMC — omitido
                "eth_dominance": None,
                "usdt_dominance": None,
                "yield_spread": _yield_spread,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            # Dados de cross_asset reais para correlações
            cross_asset_features = {
                "correlation_spy": market_environment.get("correlation_spy"),
                "btc_dxy_corr_30d": market_environment.get("correlation_dxy") or _cross.get("btc_dxy_corr_30d"),
                "dxy_momentum": _dxy_mom,
            }
            
            current_price_data = {
                "btc_price": price_context.get("current_price"),
                "eth_price": None  # Adicionar se disponível
            }
            
            # Analisa o regime
            regime = regime_detector.detect_regime(
                macro_data=macro_data,
                cross_asset_features=cross_asset_features,
                current_price_data=current_price_data
            )
            
            # Adiciona a análise de regime ao payload
            regime_analysis = {
                "market_regime": regime.market_regime.value,
                "correlation_regime": regime.correlation_regime.value,
                "volatility_regime": regime.volatility_regime.value,
                "regime_confidence": regime.regime_confidence,
                "regime_stability": regime.regime_stability,
                "risk_score": regime.risk_score,
                "fear_greed_proxy": regime.fear_greed_proxy,
                "regime_change_warning": regime.regime_change_warning,
                "divergence_alert": regime.divergence_alert,
                "primary_driver": regime.primary_driver,
                "signals_summary": regime.signals_summary
            }
        except Exception as e:
            logging.debug("Erro ao analisar regime: %s", e)
            regime_analysis = {}

    # 6. Metadados do Sinal
    signal_metadata = {
        "type": signal.get("tipo_evento"),
        "battle_result": signal.get("resultado_da_batalha"),
        "severity": signal.get("severity", "INFO"),
        "window_id": signal.get("janela_numero"),
        "timestamp_utc": signal.get("timestamp_utc"),
        "description": signal.get("descricao")
    }

    # 7. Volume Profile compacto para os 3 timeframes (RESTAURADO - essencial para IA)
    vp_all_tf = {}
    if isinstance(historical_profile, dict):
        for tf_name in ("daily", "weekly", "monthly"):
            tf_vp = historical_profile.get(tf_name, {})
            if tf_vp and tf_vp.get("poc"):
                vp_all_tf[tf_name] = {
                    "poc": tf_vp.get("poc"),
                    "vah": tf_vp.get("vah"),
                    "val": tf_vp.get("val"),
                }
    # Adicionar hvns/lvns nearby do daily (ja compactados)
    if vp_compact_daily:
        hvns = vp_compact_daily.get("hvns_nearby")
        if isinstance(hvns, list) and hvns:
            vp_all_tf.setdefault("daily", {})["hvns_nearby"] = hvns[:3]
        lvns = vp_compact_daily.get("lvns_nearby")
        if isinstance(lvns, list) and lvns:
            vp_all_tf.setdefault("daily", {})["lvns_nearby"] = lvns[:3]

    # 8. Estrutura Final
    # CORREÇÃO: usa _safe_epoch_ms() para garantir int válido
    # Problema anterior: signal["timestamp"] é STRING e retornava
    # como truthy no 'or' encadeado, quebrando o epoch_ms no LLM
    _epoch_ms_safe = _safe_epoch_ms(signal)
    
    ai_payload = {
        "symbol": symbol,
        "timestamp": signal.get("timestamp"),
        "epoch_ms": _epoch_ms_safe,
        "signal_metadata": signal_metadata,
        "price_context": price_context,
        "volume_profile": vp_all_tf,
        "flow_context": flow_context,
        "orderbook_context": ob_context,
        "technical_indicators": technical_indicators,
        "cross_asset_context": cross_asset_context,
        "macro_context": macro_full_context,
        "ml_features": ml_features,
        "historical_stats": signal.get("historical_confidence", {})
    }

    # Adiciona a análise de regime ao payload
    if regime_analysis:
        ai_payload["regime_analysis"] = regime_analysis

    # === CAMPOS DE COMPATIBILIDADE (Legacy Support) ===
    # Campos escalares leves para templates Jinja/f-strings existentes
    ai_payload["tipo_evento"] = signal.get("tipo_evento")
    ai_payload["ativo"] = symbol
    ai_payload["descricao"] = signal.get("descricao")
    ai_payload["delta"] = signal.get("delta")
    ai_payload["volume_total"] = signal.get("volume_total")
    ai_payload["preco_fechamento"] = signal.get("preco_fechamento")
    # REMOVIDOS para redução de custo (dados já estão em seções estruturadas):
    # - fluxo_continuo  → já resumido em flow_context
    # - historical_vp   → já compactado em price_context.volume_profile_daily (+ FORBIDDEN pelo guardrail)
    # - orderbook_data  → já resumido em orderbook_context
    # multi_tf mantido pois compressor qwen usa diretamente
    ai_payload["multi_tf"] = mtf
    ai_payload["event_history"] = signal.get("event_history", [])  # Memória injetada

    # === SEÇÃO DE INTELIGÊNCIA QUANTITATIVA ===
    # Se houver previsão ML, adiciona ao contexto
    quant_context: Dict[str, Any] = {}

    # Usa a previsão passada como parâmetro, com fallback para o próprio sinal (compatibilidade)
    ml_prediction = ml_prediction or signal.get("ml_prediction") or {}

    if ml_prediction and ml_prediction.get("status") == "ok":
        prob = ml_prediction.get("prob_up", 0.5)
        confidence = ml_prediction.get("confidence", 0.0)

        # FIX 6: flag extreme predictions (HybridDecision descarta mas LLM recebia sem aviso)
        is_extreme = (
            ml_prediction.get("extreme_filtered", False)
            or float(prob) > _ML_EXTREME_THRESHOLD_HIGH
            or float(prob) < _ML_EXTREME_THRESHOLD_LOW
        )

        quant_context = {
            "model_probability_up": float(prob),
            "confidence_score": float(confidence),
        }

        if is_extreme:
            quant_context["unreliable"] = True
            quant_context["unreliable_reason"] = "extreme_probability"
            logging.info(
                "QUANT_EXTREME_FLAGGED prob_up=%.4f confidence=%.4f → marked unreliable",
                prob, confidence,
            )

    # Adiciona ao payload principal
    ai_payload["quant_model"] = quant_context
    ai_payload["ml_str"] = ""  # Templates constroem ml_str proprio via qwen_patch2

    # 🔧 ENRICHMENT: Adiciona contextos de targets/opções/on-chain/risco
    enriched_ctx = build_enriched_ai_context(signal)
    ai_payload.update(enriched_ctx)
    
    # NOVO: mesclar contextos enriquecidos
    ai_payload = add_enriched_context_to_ai_payload(ai_payload, signal)

    def _apply_section_cache(payload: Dict[str, Any]) -> Dict[str, Any]:
        cache_path = os.getenv("PAYLOAD_SECTION_CACHE_PATH", "logs/payload_section_cache.json")
        cache = SectionCache(cache_path)
        cache_meta = payload.setdefault("_section_cache", {})
        now_ms = payload.get("epoch_ms") or int(datetime.now(timezone.utc).timestamp() * 1000)
        cacheable = {
            "macro_context": (cfg.get("cache_ttls_s") or {}).get("macro_context", 3600),
            "cross_asset_context": (cfg.get("cache_ttls_s") or {}).get("cross_asset_context", 3600),
        }

        for section_name, ttl_s in cacheable.items():
            section = payload.get(section_name)
            if not isinstance(section, dict):
                continue

            ref_new = canonical_ref(section)
            cache_key = f"{symbol}:{section_name}:v2"
            entry = cache.get(cache_key)
            if entry and entry.get("ref") == ref_new and is_fresh(entry.get("saved_at_ms"), ttl_s, now_ms):
                age_s = int(max(0, now_ms - entry.get("saved_at_ms", now_ms)) / 1000)
                # Não substituir os dados do payload; apenas registra meta de cache.
                cache_meta[section_name] = {"hit": True, "ref": entry["ref"], "age_s": age_s}
                logging.info(
                    "CACHE_HIT section=%s ref=%s age_s=%s",
                    section_name,
                    entry["ref"],
                    age_s,
                )
                _append_payload_metric(
                    {
                        "cache_hit": True,
                        "section": section_name,
                        "ref": entry["ref"],
                        "age_s": age_s,
                    }
                )
            else:
                cache.set(cache_key, ref_new, now_ms, section)
                cache_meta[section_name] = {"hit": False, "ref": ref_new, "age_s": 0}
                logging.info(
                    "CACHE_MISS section=%s ref=%s",
                    section_name,
                    ref_new,
                )
                _append_payload_metric({"cache_hit": False, "section": section_name, "ref": ref_new})

        return payload

    def _build_decision_features_hash(payload: Dict[str, Any]) -> str:
        """Gera hash estável de campos chave para canary local."""
        base = {
            "symbol": payload.get("symbol"),
            "epoch_ms": payload.get("epoch_ms"),
            "price": (payload.get("price_context") or {}).get("current_price"),
            "net_flow": (payload.get("flow_context") or {}).get("net_flow"),
            "cvd": (payload.get("flow_context") or {}).get("cvd_accumulated"),
            "imbalance": (payload.get("orderbook_context") or {}).get("imbalance"),
            "volatility_regime": (payload.get("macro_context") or {}).get("regime", {}).get("trend"),
            "action_bias": (payload.get("quant_model") or {}).get("action_bias"),
        }
        encoded = json.dumps(base, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _validate_payload_v2(payload: Dict[str, Any]) -> None:
        """
        Valida estrutura e tipos do payload v2 antes de enviar ao LLM.
        
        CORREÇÃO aplicada:
            - Antes: aceitava epoch_ms como STRING (bug silencioso)
            - Agora: verifica tipo int/float E valor mínimo plausível
            - Antes: não logava detalhes do erro
            - Agora: loga tipo e valor recebido para debug
        """
        # ── Verificação 1: É um dicionário? ─────────────────────────
        if not isinstance(payload, dict):
            raise ValueError(
                f"payload_v2 inválido: esperado dict, "
                f"recebi {type(payload).__name__}"
            )
        
        # ── Verificação 2: Symbol presente? ─────────────────────────
        if not payload.get("symbol"):
            raise ValueError(
                "payload_v2 inválido: campo 'symbol' é obrigatório "
                "e não pode ser vazio"
            )
        
        # ── Verificação 3: epoch_ms é int válido? ───────────────────
        # CRÍTICO: antes aceitava STRING como truthy - bug silencioso!
        epoch = payload.get("epoch_ms")
        
        if epoch is None:
            raise ValueError(
                "payload_v2 inválido: 'epoch_ms' é None - "
                "timestamp obrigatório para rastreabilidade"
            )
        
        if not isinstance(epoch, (int, float)):
            raise ValueError(
                f"payload_v2 inválido: 'epoch_ms' deve ser int ou float, "
                f"recebi type={type(epoch).__name__} value='{epoch}' - "
                f"STRING de timestamp não é aceita aqui"
            )
        
        if int(epoch) < 1_000_000_000_000:
            raise ValueError(
                f"payload_v2 inválido: 'epoch_ms' valor suspeito={epoch} - "
                f"esperado timestamp em milissegundos (> 1_000_000_000_000)"
            )
        
        # ── Verificação 4: Preço atual presente? ────────────────────
        price_ctx = payload.get("price_context") or {}
        if price_ctx.get("current_price") is None:
            raise ValueError(
                "payload_v2 inválido: price_context.current_price é obrigatório - "
                "o LLM não pode analisar sem preço atual"
            )
        
        # ── Log de sucesso para rastreabilidade ─────────────────────
        logging.debug(
            "PAYLOAD_V2_VALID symbol=%s epoch_ms=%s price=%s",
            payload.get("symbol"),
            epoch,
            price_ctx.get("current_price")
        )

    # === INSTITUTIONAL ANALYTICS (23 módulos) ===
    ai_payload = _inject_institutional_analytics(ai_payload, signal)

    # === MULTI-TIMEFRAME (compactado) ===
    _multi_tf_raw = None
    if isinstance(signal, dict):
        # 0) tenta direto no signal (fallback mais simples)
        _multi_tf_raw = signal.get("multi_tf")

        # 1) tenta raw_event direto
        if not isinstance(_multi_tf_raw, dict):
            _raw = signal.get("raw_event", {})
            if isinstance(_raw, dict):
                _multi_tf_raw = _raw.get("multi_tf")

                # 2) tenta raw_event.raw_event (caso aninhado)
                if not isinstance(_multi_tf_raw, dict):
                    _raw2 = _raw.get("raw_event", {})
                    if isinstance(_raw2, dict):
                        _multi_tf_raw = _raw2.get("multi_tf")

                # 3) tenta contextual_snapshot em ambos níveis
                if not isinstance(_multi_tf_raw, dict):
                    _cs = _raw.get("contextual_snapshot", {})
                    if isinstance(_cs, dict):
                        _multi_tf_raw = _cs.get("multi_tf")

                if not isinstance(_multi_tf_raw, dict):
                    _raw2 = _raw.get("raw_event", {})
                    if isinstance(_raw2, dict):
                        _cs2 = _raw2.get("contextual_snapshot", {})
                        if isinstance(_cs2, dict):
                            _multi_tf_raw = _cs2.get("multi_tf")

    if not isinstance(_multi_tf_raw, dict):
        _multi_tf_raw = None
    
    if isinstance(_multi_tf_raw, dict) and _multi_tf_raw:
        tf_compact = {}
        for tf_key, tf_data in _multi_tf_raw.items():
            if isinstance(tf_data, dict):
                # CORREÇÃO: tenta múltiplas chaves pois o signal usa
                # "trend" (inglês) mas o legado usava "tendencia" (PT)
                # Ambas as formas são suportadas com fallback
                tf_entry = {
                    # Tendência: signal usa "trend", legado usa "tendencia"
                    "trend": (
                        tf_data.get("trend")        # ✅ chave real no signal
                        or tf_data.get("tendencia") # fallback legado PT
                    ),
                    # Preço atual do timeframe
                    "price": (
                        tf_data.get("price")
                        or tf_data.get("preco_atual")
                    ),
                    # MME21
                    "mme21": tf_data.get("mme_21"),
                    # RSI: tenta múltiplas variações de nome
                    "rsi": (
                        tf_data.get("rsi")
                        or tf_data.get("rsi_short")
                        or tf_data.get("rsi_14")
                    ),
                    # MACD line
                    "macd": (
                        tf_data.get("macd")
                        or tf_data.get("macd_line")
                    ),
                    # MACD signal
                    "macd_s": (
                        tf_data.get("macd_signal")
                        or tf_data.get("macd_s")
                        or tf_data.get("signal")
                    ),
                    # ADX
                    "adx": (
                        tf_data.get("adx")
                        or tf_data.get("adx_14")
                    ),
                    # ATR
                    "atr": tf_data.get("atr"),
                    # Regime do timeframe
                    "regime": (
                        tf_data.get("regime")
                        or tf_data.get("market_regime")
                    ),
                }
                # Remove campos None para não poluir payload
                tf_entry = {
                    k: v for k, v in tf_entry.items() 
                    if v is not None
                }
                # Só adiciona se tem pelo menos trend
                # (evita entradas vazias no payload)
                if tf_entry.get("trend"):
                    tf_compact[tf_key] = tf_entry
                elif tf_entry:
                    # Tem dados mas sem trend - adiciona mesmo assim
                    tf_compact[tf_key] = tf_entry
        
        if tf_compact:
            ai_payload["multi_tf"] = tf_compact
            logging.debug(
                "MULTI_TF_COMPACT timeframes=%s",
                list(tf_compact.keys())
            )
        else:
            logging.debug(
                "MULTI_TF_EMPTY raw_keys=%s",
                list(_multi_tf_raw.keys())
            )

    # === COMPRESSÃO / V2 ===
    v1_bytes = len(json.dumps(ai_payload, ensure_ascii=False).encode("utf-8"))
    action_bias_v1 = (ai_payload.get("quant_model") or {}).get("action_bias")
    # Se V3 vai comprimir no runner, pular V2 para evitar dupla compressão
    v2_enabled = bool(cfg.get("v2_enabled", True)) and not skip_v2_compress
    max_bytes = int(cfg.get("max_bytes", 6144) or 6144)

    # Verifica se multi_tf está presente antes da compressão (debug apenas)
    _logger = logging.getLogger(__name__)
    pre_has_multi_tf = "multi_tf" in ai_payload
    if _logger.isEnabledFor(logging.DEBUG):
        _logger.debug("PRE_COMPRESS has_multi_tf=%s keys=%s",
                      pre_has_multi_tf, sorted(ai_payload.keys()))

    if v2_enabled:
        try:
            payload_v2 = compress_payload(ai_payload, max_bytes=max_bytes)
            
            # ── CORREÇÃO: Restaurar epoch_ms se compressor descartou ──
            # O compressor pode eliminar epoch_ms durante otimização de bytes
            # Aqui garantimos que o valor original sempre seja preservado
            if payload_v2.get("epoch_ms") is None:
                payload_v2["epoch_ms"] = _epoch_ms_safe
                logging.warning(
                    "EPOCH_MS_RESTORED_AFTER_COMPRESSION "
                    "restored_value=%s symbol=%s",
                    _epoch_ms_safe,
                    symbol
                )
            # Verifica se epoch_ms é int válido após compressão
            elif not isinstance(payload_v2.get("epoch_ms"), (int, float)):
                logging.warning(
                    "EPOCH_MS_WRONG_TYPE_AFTER_COMPRESSION "
                    "type=%s value=%s restoring=%s",
                    type(payload_v2.get("epoch_ms")).__name__,
                    payload_v2.get("epoch_ms"),
                    _epoch_ms_safe
                )
                payload_v2["epoch_ms"] = _epoch_ms_safe
            
            _validate_payload_v2(payload_v2)
            # Confirma limite de bytes
            size_bytes = len(json.dumps(payload_v2, ensure_ascii=False).encode("utf-8"))
            if size_bytes > max_bytes:
                raise ValueError(f"payload_v2 excedeu limite de bytes: {size_bytes}")
            decision_hash = _build_decision_features_hash(payload_v2)
            payload_v2["decision_features_hash"] = decision_hash
            action_bias_v2 = (payload_v2.get("quant_model") or {}).get("action_bias")
            ai_payload = payload_v2
            logging.info(
                "PAYLOAD_V2_CANARY v1_bytes=%s v2_bytes=%s decision_hash=%s decision_consistent=%s action_v1=%s action_v2=%s",
                v1_bytes,
                size_bytes,
                decision_hash,
                action_bias_v1 == action_bias_v2,
                action_bias_v1,
                action_bias_v2,
            )
        except Exception as e:
            logging.error(f"Fallback para payload v1 (compressão falhou): {e}", exc_info=True)
            logging.info("PAYLOAD_V1_ONLY v1_bytes=%s", v1_bytes)
            _append_payload_metric({"fallback_v1": True, "payload_bytes": v1_bytes, "error": str(e)})
    else:
        logging.info("PAYLOAD_V1_FORCED v1_bytes=%s", v1_bytes)

    # Verifica se multi_tf sobreviveu à compressão (debug + alerta se drop)
    post_has_multi_tf = "multi_tf" in ai_payload
    post_has_tf = "tf" in ai_payload  # forma compactada
    if _logger.isEnabledFor(logging.DEBUG):
        _logger.debug("POST_COMPRESS has_multi_tf=%s has_tf=%s keys=%s",
                      post_has_multi_tf, post_has_tf, sorted(ai_payload.keys()))

    # Só alerta se multi_tf existia antes e sumiu depois (possível bug)
    if pre_has_multi_tf and not (post_has_multi_tf or post_has_tf):
        _logger.warning(
            "COMPRESSOR_DROPPED_MULTI_TF pre_has_multi_tf=%s post_has_multi_tf=%s post_has_tf=%s",
            pre_has_multi_tf, post_has_multi_tf, post_has_tf
        )

    # Aplica cache de seções apenas para v2
    if ai_payload.get("_v") == 2 and cfg.get("section_cache_enabled", True):
        ai_payload = _apply_section_cache(ai_payload)

    try:
        optimized_payload = AIPayloadOptimizer.optimize(ai_payload)
        if isinstance(optimized_payload, dict) and optimized_payload:
            ai_payload = optimized_payload
    except Exception as e:
        logging.debug("Falha ao otimizar payload da IA: %s", e, exc_info=True)

    # Limpar campos nulos/vazios antes de retornar
    ai_payload = _strip_empty(ai_payload)

    # Remover metadados internos que não ajudam o LLM na análise
    # NOTA: _v é usado pelo guardrail e deep compression para identificar
    # payload já comprimido - NÃO remover aqui, é verificado downstream
    _internal_keys = [
        "decision_features_hash",  # hash interno
        "_section_cache",          # metadado de cache
    ]
    for _k in _internal_keys:
        ai_payload.pop(_k, None)

    # Remover total_features do quant_model (não ajuda LLM)
    _qm = ai_payload.get("quant_model")
    if isinstance(_qm, dict):
        _qm.pop("total_features", None)
        _qm.pop("features_used", None)

    return ai_payload


def build_payload_with_cross_asset(
    symbol: str,
    event_type: str,
    timestamp: int,
    base_payload: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Constrói um payload com contexto cross-asset adicionado.

    Esta função é um wrapper simplificado para adicionar cross_asset_context
    a um payload base existente, focando especificamente nas correlações
    cross-asset para análise da IA.

    Args:
        symbol (str): Símbolo do ativo (ex: BTCUSDT).
        event_type (str): Tipo do evento (ex: AI_ANALYSIS).
        timestamp (int): Timestamp em milissegundos.
        base_payload (Dict[str, Any]): Payload base com dados do mercado.

    Returns:
        Dict[str, Any]: Payload atualizado com cross_asset_context.
    """
    # Copia o payload base
    final_payload = base_payload.copy()

    # Adiciona metadados básicos se não existirem
    if "symbol" not in final_payload:
        final_payload["symbol"] = symbol
    if "event_type" not in final_payload:
        final_payload["event_type"] = event_type
    if "timestamp" not in final_payload:
        final_payload["timestamp"] = timestamp

    # Usa cross_asset já presente no payload (calculado uma vez em generate_ml_features)
    existing_cross = (
        base_payload.get("ml_features", {}).get("cross_asset", {})
        or base_payload.get("cross_asset", {})
    )

    cross_asset_context = {}
    if existing_cross:
        cross_asset_context = {
            "features": {
                "btc_eth_corr_7d": existing_cross.get("btc_eth_corr_7d"),
                "btc_eth_corr_30d": existing_cross.get("btc_eth_corr_30d"),
                "btc_dxy_corr_30d": existing_cross.get("btc_dxy_corr_30d"),
                "btc_ndx_corr_30d": existing_cross.get("btc_ndx_corr_30d"),
            },
            "correlations": {
                "btc_eth": {
                    "short_term_7d": existing_cross.get("btc_eth_corr_7d"),
                    "long_term_30d": existing_cross.get("btc_eth_corr_30d"),
                    "relationship": "cripto_major_pair"
                },
                "btc_dxy": {
                    "medium_term_30d": existing_cross.get("btc_dxy_corr_30d"),
                    "long_term_90d": existing_cross.get("btc_dxy_corr_90d"),
                    "relationship": "inverse_usd_strength"
                },
                "btc_ndx": {
                    "medium_term_30d": existing_cross.get("btc_ndx_corr_30d"),
                    "relationship": "tech_risk_correlation"
                }
            },
            "market_context": {
                "dxy_momentum": {
                    "return_5d": existing_cross.get("dxy_return_5d"),
                    "return_20d": existing_cross.get("dxy_return_20d"),
                }
            }
        }
    else:
        cross_asset_context = {
            "features": {},
            "correlations": {},
            "market_context": {}
        }

    # Adiciona o contexto cross-asset ao payload
    final_payload["cross_asset_context"] = cross_asset_context

    return final_payload
