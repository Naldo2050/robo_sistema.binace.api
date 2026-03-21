# market_orchestrator/ai/payload_compressor_v3.py
"""
Payload Compressor V3.1 -- Compressor Inteligente para LLM API
==============================================================
Comprime dados redundantes/verbosos mantendo 100% da qualidade analítica.
Estratégia:
  - Remove campos duplicados e redundantes
  - Abrevia labels longos (strings descritivas)
  - Mantém TODOS os dados numéricos críticos
  - Preserva dados institucionais completos
  - Adiciona campos que estavam sendo perdidos

Economia estimada: ~65% tokens sem perda de qualidade analítica.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════
# MAPEAMENTOS DE COMPRESSÃO (strings longas → abreviações)
# ══════════════════════════════════════════════════════════════════

REGIME_MAP = {
    # Português
    "Alta": "UP", "Baixa": "DOWN", "Lateral": "SIDE",
    "alta": "UP", "baixa": "DOWN", "lateral": "SIDE",
    "Acumulação": "ACCUM", "Manipulação": "MANIP",
    "Distribuição": "DIST", "Expansão": "EXPAN", "Range": "RANGE",
    # Encoding corrompido (fallback)
    "Acumulação": "ACCUM", "Manipulação": "MANIP",
    "Distribuição": "DIST", "Expansão": "EXPAN",
    # CORREÇÃO BUG2: inglês que o signal usa diretamente
    "neutral": "NEUT", "Neutral": "NEUT",
    "bullish": "UP",   "Bullish": "UP",
    "bearish": "DOWN", "Bearish": "DOWN",
    "trending": "TREND", "ranging": "RANGE",
    "breakout": "BREAK", "reversal": "REV",
    "accumulation": "ACCUM", "distribution": "DIST",
    # Abreviações que o compressor v1 usa
    "UP": "UP", "DOWN": "DOWN", "SIDE": "SIDE",
    "NEUT": "NEUT", "TREND": "TREND",
}

FLOW_TREND_MAP = {
    "accelerating_selling": "accel_sell",
    "accelerating_buying": "accel_buy",
    "decelerating_selling": "decel_sell",
    "decelerating_buying": "decel_buy",
    "short_term_reversal_to_sell": "rev_sell",
    "short_term_reversal_to_buy": "rev_buy",
    "increasing_buying": "inc_buy",
    "increasing_selling": "inc_sell",
    "stable": "stable",
}

ABSORPTION_MAP = {
    "Neutra": "NEUTRAL",
    "Forte Compradora": "STRONG_BUY",
    "Forte Vendedora": "STRONG_SELL",
    "Compradora": "BUY",
    "Vendedora": "SELL",
    "STRONG_ABSORPTION": "STRONG_ABS",
    "WEAK_ABSORPTION": "WEAK_ABS",
    "NO_ABSORPTION": "NO_ABS",
}

WHALE_CLASS_MAP = {
    "MILD_DISTRIBUTION": "MILD_DIST",
    "STRONG_DISTRIBUTION": "STR_DIST",
    "MILD_ACCUMULATION": "MILD_ACC",
    "STRONG_ACCUMULATION": "STR_ACC",
    "NEUTRAL": "NEUTRAL",
}

WHALE_BIAS_MAP = {
    "DISTRIBUTING": "DIST",
    "ACCUMULATING": "ACC",
    "NEUTRAL": "NEUT",
}

SEVERITY_MAP = {
    "HIGH": "H", "MEDIUM": "M", "LOW": "L", "CRITICAL": "C",
}

PRESSURE_MAP = {
    "STRONG_SELL": "STR_SELL",
    "STRONG_BUY": "STR_BUY",
    "SELL": "SELL",
    "BUY": "BUY",
    "NEUTRAL": "NEUT",
}


# ══════════════════════════════════════════════════════════════════
# FUNÇÕES AUXILIARES
# ══════════════════════════════════════════════════════════════════

def _r(value: Any, precision_type: str = "ratio") -> Any:
    """Arredonda valor com precisão adequada ao tipo."""
    if value is None:
        return None
    try:
        v = float(value)
    except (ValueError, TypeError):
        return value
    precisions = {
        "price": 1,
        "ratio": 2,
        "percent": 1,
        "indicator": 0,
        "score": 0,
        "volume_usd": 0,
    }
    decimals = precisions.get(precision_type, 2)
    return int(round(v)) if decimals == 0 else round(v, decimals)


def _safe_get(data: dict, *keys: str, default=None) -> Any:
    """Navega em dicionários aninhados com segurança."""
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
        if current is None:
            return default
    return current


def _get_institutional(payload: dict) -> dict:
    """Extrai institutional_analytics de qualquer nível do payload."""
    inst = payload.get("institutional_analytics", {})
    if not inst:
        inst = payload.get("raw_event", {}).get("institutional_analytics", {})
    return inst if isinstance(inst, dict) else {}


def _get_fluxo(payload: dict) -> dict:
    """Extrai fluxo_continuo de qualquer nível do payload."""
    fc = payload.get("fluxo_continuo", {})
    if not fc:
        fc = payload.get("raw_event", {}).get("fluxo_continuo", {})
    return fc if isinstance(fc, dict) else {}


# ══════════════════════════════════════════════════════════════════
# COMPRESSORES POR SEÇÃO
# ══════════════════════════════════════════════════════════════════

def _compress_price(payload: dict) -> dict:
    """
    Comprime dados de preço mantendo todos os dados analíticos críticos.
    Inclui: OHLC, VWAP, profile shape, poor extremes, auction status.
    """
    result: Dict[str, Any] = {}

    # Buscar preço em múltiplos locais
    price_close = (
        payload.get("preco_fechamento")
        or payload.get("anchor_price")
        or _safe_get(payload, "raw_event", "preco_fechamento")
        or _safe_get(payload, "contextual_snapshot", "ohlc", "close")
        or payload.get("price", {}).get("c") if isinstance(payload.get("price"), dict) else None
    )

    # Se payload já comprimido (price é dict com c)
    price_data = payload.get("price", {})
    if isinstance(price_data, dict) and price_data.get("c"):
        result.update({k: v for k, v in price_data.items() if v is not None})
        return result

    # Construir do zero
    ohlc = _safe_get(payload, "contextual_snapshot", "ohlc") or {}

    result["c"] = _r(price_close or ohlc.get("close"), "price")
    result["o"] = _r(ohlc.get("open"), "price")
    result["h"] = _r(ohlc.get("high"), "price")
    result["l"] = _r(ohlc.get("low"), "price")
    result["vwap"] = _r(ohlc.get("vwap"), "price")

    # Dados institucionais do perfil
    inst = _get_institutional(payload)
    profile = _safe_get(inst, "profile_analysis", "profile_shape") or {}
    poor = _safe_get(inst, "profile_analysis", "poor_extremes") or {}

    if profile:
        result["shape"] = profile.get("shape", "")
        result["signal"] = profile.get("trading_signal", "")

    if poor:
        result["poor_high"] = 1 if poor.get("poor_high", {}).get("detected") else 0
        result["poor_low"] = 1 if poor.get("poor_low", {}).get("detected") else 0
        result["auction"] = poor.get("action_bias", "")
        # Preços dos poor extremes
        if poor.get("poor_high", {}).get("detected"):
            result["poor_high_px"] = _r(poor["poor_high"].get("price"), "price")
        if poor.get("poor_low", {}).get("detected"):
            result["poor_low_px"] = _r(poor["poor_low"].get("price"), "price")

    # Snapshot adicional
    snap = payload.get("contextual_snapshot", {})
    if snap:
        result["poc"] = _r(snap.get("poc_price"), "price")
        result["dwell_loc"] = snap.get("dwell_location", "")
        result["trades_ps"] = _r(snap.get("trades_per_second"), "ratio")

    return {k: v for k, v in result.items() if v is not None and v != ""}


def _compress_volume_profile(payload: dict) -> Optional[dict]:
    """
    Comprime Volume Profile mantendo POC, VAL, VAH para todos os timeframes.

    CORREÇÃO BUG4:
        Antes ignorava "historical_vp" que contém dados daily/weekly/monthly.
        O payload real tem:
            "historical_vp": {
                "daily":   {poc, vah, val, status}
                "weekly":  {poc, vah, val, status}
                "monthly": {poc, vah, val, status}
            }
        Agora busca em todos os locais possíveis e retorna
        VP estruturado por timeframe para análise multi-tf.
    """
    result: Dict[str, Any] = {}

    # ── Prioridade 1: historical_vp completo (daily/weekly/monthly) ──
    hvp = payload.get("historical_vp", {})
    if isinstance(hvp, dict) and hvp:
        for tf_name in ("daily", "weekly", "monthly"):
            tf_vp = hvp.get(tf_name, {})
            if isinstance(tf_vp, dict) and tf_vp.get("poc"):
                result[tf_name] = {
                    "poc": _r(tf_vp.get("poc"), "price"),
                    "vah": _r(tf_vp.get("vah"), "price"),
                    "val": _r(tf_vp.get("val"), "price"),
                }
        if result:
            return result

    # ── Prioridade 2: volume_profile já estruturado ──────────────────
    vp = payload.get("vp") or payload.get("volume_profile") or {}
    if isinstance(vp, dict):
        # Se tem subchaves daily/weekly/monthly
        if vp.get("daily") or vp.get("weekly") or vp.get("monthly"):
            for tf_name in ("daily", "weekly", "monthly"):
                tf_vp = vp.get(tf_name, {})
                if isinstance(tf_vp, dict) and tf_vp.get("poc"):
                    result[tf_name] = {
                        "poc": _r(tf_vp.get("poc"), "price"),
                        "vah": _r(tf_vp.get("vah"), "price"),
                        "val": _r(tf_vp.get("val"), "price"),
                    }
            if result:
                return result
        # Se tem poc direto (VP simples sem timeframe)
        if vp.get("poc"):
            result["daily"] = {
                "poc": _r(vp.get("poc"), "price"),
                "vah": _r(vp.get("vah"), "price"),
                "val": _r(vp.get("val"), "price"),
            }
            return result

    # ── Prioridade 3: contextual_snapshot ────────────────────────────
    snap = payload.get("contextual_snapshot", {})
    if isinstance(snap, dict) and snap.get("poc_price"):
        result["current"] = {
            "poc": _r(snap.get("poc_price"), "price"),
            "poc_vol": _r(snap.get("poc_volume"), "ratio"),
            "poc_pct": _r(snap.get("poc_percentage"), "percent"),
        }
        return result

    return result if result else None


def _compress_regime(payload: dict) -> dict:
    """Comprime análise de regime de mercado."""
    result: Dict[str, Any] = {}

    # Regime analysis
    regime = payload.get("regime_analysis", {})
    if regime:
        result["market"] = regime.get("current_regime", "")
        probs = regime.get("regime_probabilities", {})
        if probs:
            result["prob_trend"] = _r(probs.get("trending"), "ratio")
            result["prob_rev"] = _r(probs.get("mean_reverting"), "ratio")
            result["prob_break"] = _r(probs.get("breakout"), "ratio")
        result["change_prob"] = _r(regime.get("regime_change_probability"), "ratio")
        result["duration"] = regime.get("expected_regime_duration", "")

    # Volatility
    vol = payload.get("volatility_metrics", {})
    if vol:
        result["vol_regime"] = vol.get("volatility_regime", "")
        result["vol_pct"] = _r(vol.get("volatility_percentile"), "percent")

    # Cross-asset macro regime
    ml_features = payload.get("ml_features", {})
    cross = ml_features.get("cross_asset", {})
    if cross:
        result["macro"] = cross.get("macro_regime", "")
        result["corr_regime"] = cross.get("correlation_regime", "")

    return {k: v for k, v in result.items() if v is not None and v != ""}


def _compress_orderbook(payload: dict) -> dict:
    """
    Comprime orderbook mantendo dados de profundidade por nível.
    CRÍTICO: Mantém depth por nível (L1, L5, L10, L25) para análise institucional.
    """
    result: Dict[str, Any] = {}

    # Dados básicos
    ob = payload.get("orderbook_data", payload.get("ob", {}))
    if isinstance(ob, dict) and ob.get("bid") is not None:
        # Já comprimido
        result.update({k: v for k, v in ob.items() if v is not None})
        return result

    if isinstance(ob, dict):
        result["mid"] = _r(ob.get("mid"), "price")
        result["bid"] = _r(ob.get("bid_depth_usd"), "volume_usd")
        result["ask"] = _r(ob.get("ask_depth_usd"), "volume_usd")
        result["imb"] = _r(ob.get("imbalance"), "ratio")
        result["spread"] = _r(ob.get("spread"), "price")

        # Depth metrics (IMPORTANTE para análise)
        depth = ob.get("depth_metrics", {})
        if depth:
            result["top5_bid"] = _r(depth.get("bid_liquidity_top5"), "volume_usd")
            result["top5_ask"] = _r(depth.get("ask_liquidity_top5"), "volume_usd")
            result["top5_imb"] = _r(depth.get("depth_imbalance"), "ratio")

    # Order book depth por nível (L1, L5, L10, L25)
    depth_levels = payload.get("order_book_depth", {})
    if depth_levels:
        levels = {}
        for level in ["L1", "L5", "L10", "L25"]:
            ld = depth_levels.get(level, {})
            if ld:
                levels[level] = {
                    "b": _r(ld.get("bids"), "volume_usd"),
                    "a": _r(ld.get("asks"), "volume_usd"),
                    "i": _r(ld.get("imbalance"), "ratio"),
                }
        if levels:
            result["depth_levels"] = levels
        result["depth_ratio"] = _r(depth_levels.get("total_depth_ratio"), "ratio")

    # Market impact (liquidez)
    impact = payload.get("market_impact", {})
    if impact:
        result["liq_score"] = _r(impact.get("liquidity_score"), "ratio")
        result["exec_qual"] = impact.get("execution_quality", "")[:5]
        slippage = impact.get("slippage_matrix", {})
        if slippage:
            result["slip_100k"] = slippage.get("100k_usd", {})
            result["slip_1m"] = slippage.get("1m_usd", {})

    # Spread analysis
    spread = payload.get("spread_analysis", {})
    if spread:
        result["spread_bps"] = _r(spread.get("current_spread_bps"), "ratio")

    return {k: v for k, v in result.items() if v is not None}


def _compress_flow(payload: dict) -> dict:
    """
    Comprime flow mantendo TODOS os dados críticos:
    - Fluxo por janela (1m, 5m, 15m)
    - CVD, imbalance, pressão
    - ABSORÇÃO completa (buyer_strength, seller_exhaustion)
    - Participant analysis (retail sentiment)
    - Liquidity heatmap
    """
    result: Dict[str, Any] = {}
    fc = _get_fluxo(payload)
    order_flow = fc.get("order_flow", {})

    # Se payload já comprimido
    flow_data = payload.get("flow", {})
    if isinstance(flow_data, dict) and flow_data.get("net_1m") is not None:
        result.update(flow_data)
    else:
        # Fluxo por janela
        result["net_1m"] = _r(order_flow.get("net_flow_1m"), "volume_usd")
        result["net_5m"] = _r(order_flow.get("net_flow_5m"), "volume_usd")
        result["net_15m"] = _r(order_flow.get("net_flow_15m"), "volume_usd")

        # CVD
        result["cvd"] = _r(fc.get("cvd"), "ratio")

        # Imbalance
        result["imb"] = _r(order_flow.get("flow_imbalance"), "ratio")

        # Agressão
        result["agg_buy"] = _r(order_flow.get("aggressive_buy_pct"), "percent")
        result["agg_sell"] = _r(order_flow.get("aggressive_sell_pct"), "percent")

        # Buy/Sell ratio
        bsr = order_flow.get("buy_sell_ratio", {})
        if isinstance(bsr, dict):
            result["bsr"] = _r(bsr.get("buy_sell_ratio"), "ratio")
            result["pressure"] = PRESSURE_MAP.get(
                bsr.get("pressure", ""), bsr.get("pressure", ""))
            result["trend"] = FLOW_TREND_MAP.get(
                bsr.get("flow_trend", ""), bsr.get("flow_trend", ""))

        # Volumes
        result["buy_vol"] = _r(order_flow.get("buy_volume_btc"), "ratio")
        result["sell_vol"] = _r(order_flow.get("sell_volume_btc"), "ratio")

    # ═══ ABSORÇÃO COMPLETA (CRÍTICO) ═══
    absorption = _safe_get(fc, "absorption_analysis", "current_absorption") or {}
    if absorption:
        result["abs_idx"] = _r(absorption.get("index"), "ratio")
        result["abs_class"] = absorption.get("classification", "")
        result["abs_label"] = ABSORPTION_MAP.get(
            absorption.get("label", ""), absorption.get("label", ""))
        result["buyer_str"] = _r(absorption.get("buyer_strength"), "ratio")
        result["seller_exh"] = _r(absorption.get("seller_exhaustion"), "ratio")
        result["cont_prob"] = _r(absorption.get("continuation_probability"), "ratio")
        result["abs_delta"] = _r(absorption.get("delta_usd"), "volume_usd")

    # ═══ PASSIVE/AGGRESSIVE ANALYSIS (CRÍTICO) ═══
    inst = _get_institutional(payload)
    pa = _safe_get(inst, "flow_analysis", "passive_aggressive") or {}
    if pa:
        aggressive = pa.get("aggressive", {})
        passive = pa.get("passive", {})
        composite = pa.get("composite", {})

        if aggressive:
            result["pa_agg_buy"] = _r(aggressive.get("buy_pct"), "percent")
            result["pa_agg_sell"] = _r(aggressive.get("sell_pct"), "percent")
            result["pa_agg_dom"] = aggressive.get("dominance", "")[:6]

        if passive:
            result["pa_pass_dom"] = passive.get("dominance", "")[:6]

        if composite:
            result["pa_signal"] = composite.get("signal", "")
            result["pa_conv"] = composite.get("conviction", "")[:4]
            result["pa_agree"] = composite.get("agreement", 0)

    # ═══ PARTICIPANT ANALYSIS ═══
    participants = fc.get("participant_analysis", {})
    if participants:
        retail = participants.get("retail", {})
        if retail:
            result["retail_dir"] = retail.get("direction", "")[:4]
            result["retail_sent"] = retail.get("sentiment", "")[:4]
            result["retail_score"] = _r(retail.get("composite_score"), "ratio")
            result["retail_imb"] = _r(retail.get("imbalance"), "ratio")

    # ═══ LIQUIDITY HEATMAP ═══
    heatmap = fc.get("liquidity_heatmap", {})
    if heatmap:
        clusters = heatmap.get("clusters", [])
        if clusters:
            # Pegar cluster principal
            c1 = clusters[0]
            result["cluster_px"] = _r(c1.get("center"), "price")
            result["cluster_vol"] = _r(c1.get("total_volume"), "ratio")
            result["cluster_imb"] = _r(c1.get("imbalance_ratio"), "ratio")
            result["cluster_trades"] = c1.get("trades_count", 0)

        resistances = heatmap.get("resistances", [])
        if resistances:
            result["heatmap_res"] = [_r(r, "price") for r in resistances[:3]]

    # ═══ TIPO DE ABSORÇÃO ═══
    tipo_abs = fc.get("tipo_absorcao", "")
    if tipo_abs:
        result["tipo_abs"] = ABSORPTION_MAP.get(tipo_abs, tipo_abs)

    return {k: v for k, v in result.items() if v is not None and v != ""}


def _compress_whale(payload: dict) -> Optional[dict]:
    """
    Comprime dados de whale mantendo TODOS os componentes:
    - Score e classificação
    - Componentes detalhados (flow, depth, absorption)
    - Tendência histórica
    """
    result: Dict[str, Any] = {}

    # Buscar em múltiplos caminhos
    inst = _get_institutional(payload)
    whale = (
        _safe_get(inst, "flow_analysis", "whale_accumulation")
        or payload.get("whale_analysis")
        or payload.get("whale")
        or {}
    )

    # Se já comprimido
    if isinstance(whale, dict) and "score" in whale and "class" in whale:
        result.update(whale)
        return {k: v for k, v in result.items() if v is not None}

    if not isinstance(whale, dict) or not whale:
        return None

    # Score e classificação
    score = whale.get("score")
    if score is not None:
        result["score"] = _r(score, "score")

    cls = whale.get("classification")
    if cls:
        result["class"] = WHALE_CLASS_MAP.get(cls, cls)

    bias = whale.get("bias")
    if bias:
        result["bias"] = WHALE_BIAS_MAP.get(bias, bias)

    # ═══ COMPONENTES DETALHADOS (CRÍTICO) ═══
    components = whale.get("components", {})
    if components:
        # Flow component
        flow_comp = components.get("flow", {})
        if flow_comp:
            result["flow_score"] = _r(flow_comp.get("score"), "ratio")
            detail = flow_comp.get("detail", {})
            if detail:
                result["flow_div"] = detail.get("divergence", "")[:6]
                result["retail_delta"] = _r(detail.get("retail_delta"), "ratio")

        # Depth component
        depth_comp = components.get("depth", {})
        if depth_comp:
            result["depth_score"] = _r(depth_comp.get("score"), "ratio")
            detail = depth_comp.get("detail", {})
            if detail:
                result["depth_bid"] = _r(detail.get("bid_depth"), "volume_usd")
                result["depth_ask"] = _r(detail.get("ask_depth"), "volume_usd")
                result["depth_ratio"] = _r(detail.get("ratio"), "ratio")

        # Absorption component
        abs_comp = components.get("absorption", {})
        if abs_comp:
            result["abs_score"] = _r(abs_comp.get("score"), "ratio")
            detail = abs_comp.get("detail", {})
            if detail:
                result["abs_buyer_str"] = _r(detail.get("buyer_strength"), "ratio")
                result["abs_seller_exh"] = _r(detail.get("seller_exhaustion"), "ratio")
                result["abs_net"] = _r(detail.get("net_absorption"), "ratio")

    # Tendência histórica
    trend = whale.get("trend", {})
    if trend:
        result["trend_dir"] = trend.get("direction", "")[:10]
        result["trend_avg"] = _r(trend.get("avg_score"), "score")
        result["trend_samples"] = trend.get("samples", 0)

    return {k: v for k, v in result.items() if v is not None and v != ""}


def _compress_institutional(payload: dict) -> Optional[dict]:
    """
    NOVA SEÇÃO: Dados institucionais completos.
    - Profile analysis (shape, poor extremes)
    - Buy/Sell ratio institucional
    - Absorption zones
    - Calendar quality
    """
    inst = _get_institutional(payload)
    if not inst:
        return None

    result: Dict[str, Any] = {}

    # ═══ PROFILE ANALYSIS ═══
    profile = _safe_get(inst, "profile_analysis", "profile_shape") or {}
    poor = _safe_get(inst, "profile_analysis", "poor_extremes") or {}

    if profile:
        result["shape"] = profile.get("shape", "")
        result["shape_impl"] = profile.get("implication", "")[:50]
        result["trade_signal"] = profile.get("trading_signal", "")
        dist = profile.get("distribution", {})
        if dist:
            result["dist_upper"] = _r(dist.get("upper_third_pct"), "percent")
            result["dist_middle"] = _r(dist.get("middle_third_pct"), "percent")
            result["dist_lower"] = _r(dist.get("lower_third_pct"), "percent")
            result["dominant_zone"] = dist.get("dominant_zone", "")

    if poor and poor.get("status") == "success":
        result["action_bias"] = poor.get("action_bias", "")
        poor_h = poor.get("poor_high", {})
        poor_l = poor.get("poor_low", {})
        if poor_h.get("detected"):
            result["poor_h_px"] = _r(poor_h.get("price"), "price")
            result["poor_h_vol"] = _r(poor_h.get("volume_ratio"), "ratio")
        if poor_l.get("detected"):
            result["poor_l_px"] = _r(poor_l.get("price"), "price")
            result["poor_l_vol"] = _r(poor_l.get("volume_ratio"), "ratio")

    # ═══ BUY/SELL RATIO INSTITUCIONAL ═══
    bsr = _safe_get(inst, "flow_analysis", "buy_sell_ratio") or {}
    if bsr:
        ratios = bsr.get("ratios", {})
        if ratios:
            result["bsr_1m"] = _r(ratios.get("imbalance_1m"), "ratio")
            result["bsr_5m"] = _r(ratios.get("imbalance_5m"), "ratio")
            result["bsr_15m"] = _r(ratios.get("imbalance_15m"), "ratio")

        sector = bsr.get("sector_ratios", {})
        if sector:
            result["sector_retail"] = _r(sector.get("retail"), "ratio")
            result["sector_mid"] = _r(sector.get("mid"), "ratio")
            result["sector_whale"] = _r(sector.get("whale"), "ratio")

    # ═══ ABSORPTION ZONES ═══
    abs_zones = _safe_get(inst, "flow_analysis", "absorption_zones") or {}
    if abs_zones and abs_zones.get("status") != "no_events":
        result["abs_zones"] = abs_zones

    # ═══ QUALIDADE DO DADO ═══
    quality = inst.get("quality", {})
    if quality:
        calendar = quality.get("calendar", {})
        if calendar:
            result["is_weekend"] = calendar.get("is_weekend", 0)
            result["liq_warning"] = calendar.get("liquidity_warning", 0)
            result["expected_liq"] = calendar.get("expected_liquidity", "")

        latency = quality.get("latency", {})
        if latency:
            result["data_fresh"] = latency.get("data_freshness", "")
            result["is_stale"] = latency.get("is_stale", 0)

    return {k: v for k, v in result.items() if v is not None and v != ""}


def _compress_defense_zones(payload: dict) -> Optional[dict]:
    """
    NOVA SEÇÃO: Zonas de defesa institucional.
    Dados críticos para identificar S/R institucionais.
    """
    inst = _get_institutional(payload)
    defense = _safe_get(inst, "sr_analysis", "defense_zones") or {}

    if not defense or defense.get("status") != "success":
        return None

    result: Dict[str, Any] = {}

    # Zona de venda mais forte
    sell = defense.get("strongest_sell", {})
    if sell:
        result["sell_ctr"] = _r(sell.get("center"), "price")
        result["sell_low"] = _r(sell.get("range_low"), "price")
        result["sell_high"] = _r(sell.get("range_high"), "price")
        result["sell_str"] = _r(sell.get("strength"), "score")
        result["sell_dist_pct"] = _r(sell.get("distance_pct"), "ratio")
        result["sell_sources"] = sell.get("sources", [])

    # Zona de compra mais forte
    buy = defense.get("strongest_buy", {})
    if buy:
        result["buy_ctr"] = _r(buy.get("center"), "price")
        result["buy_low"] = _r(buy.get("range_low"), "price")
        result["buy_high"] = _r(buy.get("range_high"), "price")
        result["buy_str"] = _r(buy.get("strength"), "score")
        result["buy_dist_pct"] = _r(buy.get("distance_pct"), "ratio")
        result["buy_sources"] = buy.get("sources", [])

    # Total de zonas e assimetria
    result["total_zones"] = defense.get("total_zones", 0)
    asym = defense.get("defense_asymmetry", {})
    if asym:
        result["def_bias"] = asym.get("bias", "")
        result["def_desc"] = asym.get("description", "")[:60]

    return {k: v for k, v in result.items() if v is not None and v != ""}


def _compress_alerts(payload: dict) -> Optional[dict]:
    """
    NOVA SEÇÃO: Alertas e anomalias ativas.
    CRÍTICO: Informa a IA sobre condições extremas detectadas.
    """
    result: Dict[str, Any] = {}

    # Alertas ativos
    alerts = payload.get("alerts", {})
    if not alerts:
        alerts = payload.get("raw_event", {}).get("alerts", {})

    if alerts and isinstance(alerts, dict):
        active = alerts.get("active_alerts", [])
        if active:
            result["active"] = [
                {
                    "type": a.get("type", "")[:20],
                    "sev": SEVERITY_MAP.get(a.get("severity", ""), a.get("severity", "")),
                    "prob": _r(a.get("probability"), "ratio"),
                    "action": a.get("action", "")[:15],
                    "desc": a.get("description", "")[:60],
                }
                for a in active[:5]  # Máximo 5 alertas
            ]
            result["count"] = alerts.get("alert_count", len(active))
            result["max_sev"] = alerts.get("max_severity", "")

    # Anomalias detectadas
    inst = _get_institutional(payload)
    anomalies = _safe_get(inst, "quality", "anomalies") or {}

    if anomalies and anomalies.get("anomalies_detected"):
        result["anomaly_count"] = anomalies.get("count", 0)
        result["anomaly_max_sev"] = SEVERITY_MAP.get(
            anomalies.get("max_severity", ""), anomalies.get("max_severity", ""))
        result["risk_elevated"] = anomalies.get("risk_elevated", 0)
        anoms = anomalies.get("anomalies", [])
        if anoms:
            result["anomalies"] = [
                {
                    "type": a.get("type", "")[:25],
                    "sev": SEVERITY_MAP.get(a.get("severity", ""), ""),
                    "val": _r(a.get("value"), "ratio"),
                    "desc": a.get("description", "")[:50],
                }
                for a in anoms[:3]
            ]

    return result if result else None


def _compress_fibonacci(payload: dict) -> Optional[dict]:
    """
    NOVA SEÇÃO: Níveis de Fibonacci.
    Níveis críticos para identificar zonas de suporte/resistência.
    """
    fib = payload.get("fibonacci_levels", {})
    if not fib:
        return None

    result: Dict[str, Any] = {}

    result["high"] = _r(fib.get("high"), "price")
    result["low"] = _r(fib.get("low"), "price")
    result["r236"] = _r(fib.get("23.6"), "price")
    result["r382"] = _r(fib.get("38.2"), "price")
    result["r500"] = _r(fib.get("50.0"), "price")
    result["r618"] = _r(fib.get("61.8"), "price")
    result["r786"] = _r(fib.get("78.6"), "price")

    return {k: v for k, v in result.items() if v is not None}


def _compress_sr_levels(payload: dict) -> Optional[dict]:
    """
    NOVA SEÇÃO: Suportes e resistências imediatos.
    """
    result: Dict[str, Any] = {}

    res = payload.get("immediate_resistance", [])
    res_str = payload.get("resistance_strength", [])
    sup = payload.get("immediate_support", [])
    sup_str = payload.get("support_strength", [])

    if res:
        result["res"] = [_r(r, "price") for r in res[:3]]
    if res_str:
        result["res_str"] = [_r(s, "score") for s in res_str[:3]]
    if sup:
        result["sup"] = [_r(s, "price") for s in sup[:3]]
    if sup_str:
        result["sup_str"] = [_r(s, "score") for s in sup_str[:3]]

    # Order flow estendido
    ofe = payload.get("order_flow_extended", {})
    if ofe:
        result["pass_buy_pct"] = _r(ofe.get("passive_buy_pct"), "percent")
        result["pass_sell_pct"] = _r(ofe.get("passive_sell_pct"), "percent")

    return result if result else None


def _compress_onchain(payload: dict) -> Optional[dict]:
    """
    Comprime métricas on-chain relevantes.
    Remove dados redundantes mas mantém indicadores críticos.
    """
    # Buscar onchain em múltiplos locais
    raw = payload.get("raw_event", {})
    adv = raw.get("advanced_analysis", {})
    onchain = adv.get("onchain_metrics", {})

    if not onchain or not isinstance(onchain, dict):
        return None

    if onchain.get("is_real_data", 0) == 0:
        return None

    result: Dict[str, Any] = {}

    # Dados relevantes para análise
    result["active_addr"] = onchain.get("active_addresses")
    result["mempool_sz"] = onchain.get("mempool_size")
    result["fees_fast"] = onchain.get("fees_fastest_sat_vb")
    result["trade_vol_24h"] = _r(onchain.get("trade_volume_btc_24h"), "ratio")
    result["btc_sent_24h"] = _r(onchain.get("total_btc_sent_24h"), "ratio")

    # Difficulty adjustment
    diff_adj = onchain.get("difficulty_adjustment", {})
    if diff_adj:
        result["diff_change_pct"] = _r(diff_adj.get("estimated_change_pct"), "ratio")
        result["diff_progress"] = _r(diff_adj.get("progress_pct"), "percent")

    result["min_per_block"] = _r(onchain.get("minutes_between_blocks"), "ratio")

    return {k: v for k, v in result.items() if v is not None}


def _compress_derivatives(payload: dict) -> Optional[dict]:
    """
    Comprime dados de derivativos.

    CORREÇÃO BUG3:
        Antes buscava "derivatives_context" que nunca existe no payload real.
        O payload real tem "derivatives" com BTCUSDT/ETHUSDT como chaves.
        Agora tenta múltiplos nomes de campo para compatibilidade.
    """
    derivs = (
        payload.get("derivatives")          # ✅ campo real no signal
        or payload.get("derivatives_context") # fallback legado
        or payload.get("deriv")              # forma comprimida
        or {}
    )
    if not derivs or not isinstance(derivs, dict):
        return None

    result: Dict[str, Any] = {}

    btc = derivs.get("BTCUSDT", {})
    if btc and isinstance(btc, dict):
        oi = btc.get("open_interest")
        if oi is not None:
            result["btc_oi"] = _r(oi, "score")
        lsr = btc.get("long_short_ratio")
        if lsr is not None:
            result["btc_lsr"] = _r(lsr, "ratio")
    else:
        oi = derivs.get("btc_oi")
        if oi is not None:
            result["btc_oi"] = _r(oi, "score")
        lsr = derivs.get("btc_lsr")
        if lsr is not None:
            result["btc_lsr"] = _r(lsr, "ratio")

    eth = derivs.get("ETHUSDT", {})
    if eth and isinstance(eth, dict):
        lsr = eth.get("long_short_ratio")
        if lsr is not None:
            result["eth_lsr"] = _r(lsr, "ratio")

    return result if result else None


def _compress_cross_asset(payload: dict) -> Optional[dict]:
    """Comprime dados cross-asset mantendo correlações críticas."""
    ml = payload.get("ml_features", {})
    cross_ml = ml.get("cross_asset", {})

    ca = payload.get("cross_asset_context", payload.get("cross", {}))

    result: Dict[str, Any] = {}

    if cross_ml:
        result["eth_7d"] = _r(cross_ml.get("btc_eth_corr_7d"), "ratio")
        result["eth_30d"] = _r(cross_ml.get("btc_eth_corr_30d"), "ratio")
        result["dxy_30d"] = _r(cross_ml.get("btc_dxy_corr_30d"), "ratio")
        result["dxy_90d"] = _r(cross_ml.get("btc_dxy_corr_90d"), "ratio")
        result["ndx_30d"] = _r(cross_ml.get("btc_ndx_corr_30d"), "ratio")
        result["dxy_r5d"] = _r(cross_ml.get("dxy_return_5d"), "ratio")
        result["dxy_r20d"] = _r(cross_ml.get("dxy_return_20d"), "ratio")
        result["vix"] = _r(cross_ml.get("vix_current"), "ratio")
        result["us10y"] = _r(cross_ml.get("us10y_yield"), "ratio")
        result["btc_dom"] = _r(cross_ml.get("btc_dominance"), "ratio")
        result["eth_dom"] = _r(cross_ml.get("eth_dominance"), "ratio")
        result["gold"] = _r(cross_ml.get("gold_price"), "price")
        result["oil"] = _r(cross_ml.get("oil_price"), "price")
        result["macro"] = cross_ml.get("macro_regime", "")
        result["corr_regime"] = cross_ml.get("correlation_regime", "")

    elif ca and isinstance(ca, dict):
        result["eth_7d"] = _r(ca.get("eth_7d"), "ratio")
        result["eth_30d"] = _r(ca.get("eth_30d"), "ratio")
        result["dxy_30d"] = _r(ca.get("dxy_30d"), "ratio")
        result["dxy_90d"] = _r(ca.get("dxy_90d"), "ratio")

    return {k: v for k, v in result.items() if v is not None and v != ""} or None


def _compress_timeframes(payload: dict) -> Optional[dict]:
    """Comprime dados de múltiplos timeframes."""
    mtf = (
        payload.get("multi_tf")
        or payload.get("tf")
        or _safe_get(payload, "macro_context", "multi_timeframe_trends")
        or {}
    )

    if not isinstance(mtf, dict) or not mtf:
        return None

    result: Dict[str, Any] = {}

    for key, val in mtf.items():
        if not isinstance(val, dict):
            continue
        tf_data: Dict[str, Any] = {}

        tend = val.get("tendencia", val.get("trend", val.get("t", "")))
        tf_data["t"] = REGIME_MAP.get(tend, tend)

        ema = val.get("mme_21", val.get("ema_21", val.get("ema")))
        if ema is not None:
            tf_data["ema"] = _r(ema, "price")

        rsi = val.get("rsi_short", val.get("rsi"))
        if rsi is not None:
            tf_data["rsi"] = _r(rsi, "percent")

        macd = val.get("macd")
        macd_s = val.get("macd_signal")
        if isinstance(macd, list):
            tf_data["macd"] = [_r(macd[0], "indicator"),
                               _r(macd[1] if len(macd) > 1 else 0, "indicator")]
        elif macd is not None:
            tf_data["macd"] = [_r(macd, "indicator"), _r(macd_s, "indicator")]

        adx = val.get("adx")
        if adx is not None:
            tf_data["adx"] = _r(adx, "percent")

        regime = val.get("regime", val.get("reg", ""))
        tf_data["reg"] = REGIME_MAP.get(regime, regime)

        result[key] = {k: v for k, v in tf_data.items() if v is not None}

    return result if result else None


def _compress_quant(payload: dict) -> Optional[dict]:
    """Comprime dados do modelo quantitativo."""
    qm = payload.get("quant_model", payload.get("quant", {}))
    if not qm or not isinstance(qm, dict):
        return None

    prob = qm.get("model_probability_up") or qm.get("prob_up")
    conf = qm.get("confidence_score") or qm.get("confidence")

    if prob is None and conf is None:
        return None

    result: Dict[str, Any] = {}
    if prob is not None:
        result["prob_up"] = _r(prob, "ratio")
    if conf is not None:
        result["conf"] = _r(conf, "ratio")

    return result


# ══════════════════════════════════════════════════════════════════
# FUNÇÃO PRINCIPAL
# ══════════════════════════════════════════════════════════════════

def compress_payload_v3(payload: dict) -> dict:
    """
    Comprime o ai_payload completo para envio ao LLM.

    Mantém TODOS os dados analíticos críticos:
    - Price completo com profile e poor extremes
    - Flow com absorção completa
    - Whale com componentes detalhados
    - OrderBook com depth por nível
    - Dados institucionais completos
    - Defense zones (S/R institucional)
    - Fibonacci levels
    - Alertas e anomalias
    - On-chain metrics
    - Cross-asset com macro regime

    Args:
        payload: ai_payload original ou parcialmente comprimido

    Returns:
        Payload comprimido sem perda de qualidade analítica
    """
    if not isinstance(payload, dict):
        return payload

    c: Dict[str, Any] = {}

    # ── Metadata ──
    c["symbol"] = payload.get("symbol", payload.get("ativo", "BTCUSDT"))
    window = payload.get("window") or _safe_get(
        payload, "signal_metadata", "window_id")
    if window is not None:
        c["window"] = window

    # CORREÇÃO BUG1: epoch_ms com fallback robusto
    # Antes: payload.get("epoch_ms") → None se ausente
    # Agora: tenta múltiplos campos garantindo sempre um int válido
    _epoch = None
    # Tenta campos diretos como int válido
    for _ek in ("epoch_ms", "timestamp_ms", "window_close_ms"):
        _v = payload.get(_ek)
        if isinstance(_v, (int, float)) and int(_v) > 1_000_000_000_000:
            _epoch = int(_v)
            break
    # Tenta dentro de signal_metadata
    if _epoch is None:
        _sm = payload.get("signal_metadata", {})
        if isinstance(_sm, dict):
            _v = _sm.get("epoch_ms") or _sm.get("timestamp_ms")
            if isinstance(_v, (int, float)) and int(_v) > 1_000_000_000_000:
                _epoch = int(_v)
    # Tenta dentro de fluxo_continuo.time_index
    if _epoch is None:
        _fc = payload.get("fluxo_continuo", {})
        if isinstance(_fc, dict):
            _ti = _fc.get("time_index", {})
            if isinstance(_ti, dict):
                _v = _ti.get("epoch_ms")
                if isinstance(_v, (int, float)) and int(_v) > 1_000_000_000_000:
                    _epoch = int(_v)
    # Fallback: agora em ms
    if _epoch is None:
        import time as _time
        _epoch = int(_time.time() * 1000)
        logger.warning(
            "COMPRESS_V3_EPOCH_FALLBACK symbol=%s usando_now=%s",
            c.get("symbol"), _epoch
        )
    c["epoch_ms"] = _epoch

    trigger = (
        payload.get("trigger")
        or payload.get("tipo_evento")
        or _safe_get(payload, "signal_metadata", "type")
    )
    if trigger:
        c["trigger"] = trigger

    # ── Preço e Profile ──
    c["price"] = _compress_price(payload)

    # ── Volume Profile ──
    vp = _compress_volume_profile(payload)
    if vp:
        c["vp"] = vp

    # ── Regime de Mercado ──
    regime = _compress_regime(payload)
    if regime:
        c["regime"] = regime

    # ── OrderBook com Depth ──
    ob = _compress_orderbook(payload)
    if ob:
        c["ob"] = ob

    # ── Flow com Absorção Completa ──
    flow = _compress_flow(payload)
    if flow:
        c["flow"] = flow

    # ── Whale com Componentes ──
    whale = _compress_whale(payload)
    if whale:
        c["whale"] = whale

    # ── Dados Institucionais ──
    institutional = _compress_institutional(payload)
    if institutional:
        c["inst"] = institutional

    # ── Zonas de Defesa (S/R Institucional) ──
    defense = _compress_defense_zones(payload)
    if defense:
        c["defense"] = defense

    # ── Fibonacci Levels ──
    fib = _compress_fibonacci(payload)
    if fib:
        c["fib"] = fib

    # ── Suportes e Resistências ──
    sr = _compress_sr_levels(payload)
    if sr:
        c["sr"] = sr

    # ── Alertas e Anomalias ──
    alerts = _compress_alerts(payload)
    if alerts:
        c["alerts"] = alerts

    # ── On-chain ──
    onchain = _compress_onchain(payload)
    if onchain:
        c["onchain"] = onchain

    # ── Derivativos ──
    deriv = _compress_derivatives(payload)
    if deriv:
        c["deriv"] = deriv

    # ── Cross-Asset ──
    cross = _compress_cross_asset(payload)
    if cross:
        c["cross"] = cross

    # ── Timeframes ──
    tf = _compress_timeframes(payload)
    if tf:
        c["tf"] = tf

    # ── Quant Model ──
    quant = _compress_quant(payload)
    if quant:
        c["quant"] = quant

    # Validação final
    _validate_compressed(c, payload)

    return c


def _validate_compressed(compressed: dict, original: dict) -> None:
    """
    Valida que dados essenciais sobreviveram à compressão.

    CORREÇÃO BUG5:
        Adicionada validação e recuperação de epoch_ms.
        Antes: epoch_ms podia ser None no resultado final.
        Agora: tenta recuperar de múltiplos campos do original.
    """
    warnings = []

    # ── Validação 1: epoch_ms obrigatório ────────────────────────────
    epoch = compressed.get("epoch_ms")
    if epoch is None or not isinstance(epoch, (int, float)):
        warnings.append("epoch_ms MISSING or INVALID")
        # Tenta recuperar do original
        for _ek in ("epoch_ms", "timestamp_ms", "window_close_ms"):
            _v = original.get(_ek)
            if isinstance(_v, (int, float)) and int(_v) > 1_000_000_000_500:
                compressed["epoch_ms"] = int(_v)
                warnings[-1] += f" (RECOVERED from {_ek})"
                break
        # Se ainda None → usa now
        if compressed.get("epoch_ms") is None:
            import time as _time
            compressed["epoch_ms"] = int(_time.time() * 1000)
            warnings[-1] += " (FALLBACK=now)"

    # ── Validação 2: price.c obrigatório ─────────────────────────────
    price = compressed.get("price", {})
    if not price.get("c"):
        warnings.append("price.c MISSING")
        for key in ("preco_fechamento", "anchor_price", "current_price"):
            if original.get(key):
                compressed.setdefault("price", {})["c"] = _r(
                    original[key], "price"
                )
                warnings[-1] += " (RECOVERED)"
                break

    # ── Validação 3: seções críticas ─────────────────────────────────
    if "flow" not in compressed:
        warnings.append("flow MISSING")
    if "whale" not in compressed:
        warnings.append("whale MISSING")
    if "ob" not in compressed:
        warnings.append("ob MISSING")
    if "alerts" not in compressed:
        warnings.append("alerts MISSING - sem alertas ativos")

    if warnings:
        logger.warning(
            "COMPRESS_V3_VALIDATION warnings=%s original_keys=%s",
            warnings,
            list(original.keys()),
        )

    if "flow" not in compressed:
        warnings.append("flow MISSING")

    if "whale" not in compressed:
        warnings.append("whale MISSING")

    if "ob" not in compressed:
        warnings.append("ob MISSING")

    if "alerts" not in compressed:
        warnings.append("alerts MISSING - sem alertas ativos")

    if warnings:
        logger.warning(
            "COMPRESS_V3_VALIDATION: %s | Original keys: %s",
            warnings,
            list(original.keys()),
        )