"""
build_compact_payload.py — Construtor de Payload Compactado v3.1

Constrói payload otimizado para envio à IA (LLM).
Redução de ~50-60% em tokens vs v1, com 100% da qualidade analítica.

Changelog v3.1 (2026-03-14):
  - Keys LONGAS compatíveis com pipeline (price, flow, regime, etc.)
  - _v = 2 para compatibilidade com ai_analyzer_qwen.py
  - symbol incluído no payload para detecção pelo GUARDRAIL_REWRAP
  - Números com sinal explícito (+280K / -34K)
  - Mercados externos incluídos no ctx (DXY, SP500, GOLD, etc.)
  - Forçar ctx para eventos importantes (Absorção, Exaustão, etc.)
  - TFs nulos filtrados
  - Summary removido (100% redundante)
  - Redundâncias whale/flow/regime removidas
  - Cache de contexto estático (5 min)
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

logger = logging.getLogger(__name__)

# ============================================================
# SUMMARY BUILDERS — interpretação pré-processada para a IA
# ============================================================
if TYPE_CHECKING:
    from common.ai_protocols import SummaryBuilderMap

_BUILDERS: "SummaryBuilderMap | None" = None
_BUILDERS_IMPORT_FAILED = False


def _load_summary_builders() -> "SummaryBuilderMap":
    """
    Carrega os summary builders sob demanda para evitar import circular.
    """
    from market_orchestrator.ai.payload_sections import (
        build_flow_summary,
        build_institutional_summary,
        build_quality_summary,
        build_regime_summary,
        build_sr_summary,
    )

    return {
        "flow": build_flow_summary,
        "sr": build_sr_summary,
        "regime": build_regime_summary,
        "institutional": build_institutional_summary,
        "quality": build_quality_summary,
    }


def _get_builders() -> "SummaryBuilderMap":
    """Retorna builders cacheados; vazio se `payload_sections` falhar."""
    global _BUILDERS, _BUILDERS_IMPORT_FAILED
    if _BUILDERS is not None:
        return _BUILDERS
    if _BUILDERS_IMPORT_FAILED:
        return {}

    try:
        _BUILDERS = _load_summary_builders()
    except ImportError:
        _BUILDERS_IMPORT_FAILED = True
        logger.warning(
            "payload_sections não disponível — "
            "payload será enviado sem seção summary"
        )
        return {}

    return _BUILDERS

# ============================================================
# CONSTANTES E MAPAS
# ============================================================

TRIGGER_ABBREV: dict[str, str] = {
    "ANALYSIS_TRIGGER": "AT",
    "Teste de Conexão": "TEST",
    "Absorção": "ABS",
    "Exaustão": "EXH",
    "Breakout": "BRK",
    "Whale Activity": "WHL",
    "Divergência": "DIV",
    "Reversão": "REV",
    "Volume Spike": "VSPK",
    "Momentum Shift": "MOM",
}

TREND_ABBREV: dict[str, str] = {
    "Baixa": "DN",
    "Alta": "UP",
    "Lateral": "SW",
    "Neutro": "SW",
    "NE": "NE",
}

REGIME_ABBREV: dict[str, str] = {
    "Range": "RNG",
    "Acumulação": "ACC",
    "Tendência": "TRD",
    "Manipulação": "MNP",
    "RANGE": "RNG",
    "ACCUM": "ACC",
    "TRENDING": "TRD",
    "RANGE_BOUND": "RB",
    "BREAKOUT": "BRK",
}

SESSION_ABBREV: dict[str, str] = {
    "NY_OVERLAP_ACTIVE": "NY_OVL",
    "NY_MORNING_ACTIVE": "NY_AM",
    "NY_AFTERNOON_ACTIVE": "NY_PM",
    "NY_ACTIVE": "NY",
    "LONDON_ACTIVE": "LDN",
    "LONDON_CLOSE_ACTIVE": "LDN_CL",
    "ASIA_ACTIVE": "ASIA",
    "ASIA_CLOSE_ACTIVE": "ASIA_CL",
    "OFF_HOURS_ACTIVE": "OFF",
}

# Eventos que SEMPRE recebem ctx completo (não usar cache)
IMPORTANT_EVENTS: set[str] = {
    "Absorção", "Exaustão", "Breakout", "Whale Activity",
    "Divergência", "Reversão", "Volume Spike", "Momentum Shift",
    "ABS", "EXH", "BRK", "WHL", "DIV", "REV", "VSPK", "MOM",
}

# ============================================================
# CACHE DE CONTEXTO ESTÁTICO
# ============================================================

_last_static_ctx: dict = {}
_last_static_ts: float = 0.0
_STATIC_INTERVAL: float = 300.0  # 5 minutos


def _should_send_static(ctx: dict) -> bool:
    """Determina se o contexto estático deve ser reenviado."""
    global _last_static_ctx, _last_static_ts
    now = time.time()

    # Primeira vez: sempre enviar
    if _last_static_ts == 0.0:
        _last_static_ctx = ctx.copy()
        _last_static_ts = now
        return True

    # Expirou: reenviar
    if now - _last_static_ts > _STATIC_INTERVAL:
        _last_static_ctx = ctx.copy()
        _last_static_ts = now
        return True

    # Mudou significativamente: reenviar
    if ctx != _last_static_ctx:
        for key in ["ses", "fg", "lsr"]:
            if ctx.get(key) != _last_static_ctx.get(key):
                _last_static_ctx = ctx.copy()
                _last_static_ts = now
                return True
        vix_diff = abs((ctx.get("vix") or 0) - (_last_static_ctx.get("vix") or 0))
        if vix_diff > 1.0:
            _last_static_ctx = ctx.copy()
            _last_static_ts = now
            return True

    return False


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def compact_number(value: Optional[float], force_sign: bool = True) -> str:
    """
    Converte números grandes para notação compacta COM SINAL EXPLÍCITO.

    Exemplos:
        280000    → "+280K"
        -34000    → "-34K"
        1599170   → "+1.6M"
        -4593927  → "-4.6M"
        0.55      → "+0.55"
        0         → "0"
    """
    if value is None or value == 0:
        return "0"

    abs_val = abs(value)
    sign = "+" if value > 0 else "-"

    if abs_val >= 1_000_000:
        formatted = f"{abs_val / 1_000_000:.1f}M"
    elif abs_val >= 1_000:
        formatted = f"{abs_val / 1_000:.0f}K"
    elif abs_val >= 1:
        formatted = f"{abs_val:.0f}"
    else:
        formatted = f"{abs_val:.2f}"

    if force_sign:
        return f"{sign}{formatted}"
    elif value < 0:
        return f"-{formatted}"
    return formatted


def _safe_round(value: Any, decimals: int = 0) -> Optional[float]:
    """Arredonda com segurança, retorna None se inválido."""
    if value is None:
        return None
    try:
        v = float(value)
        if decimals == 0:
            return int(round(v))
        return round(v, decimals)
    except (ValueError, TypeError):
        return None


def _safe_price(ext: dict, key: str, decimals: int = 2) -> Optional[float]:
    """Extrai preço de external_markets com segurança."""
    data = ext.get(key, {})
    if isinstance(data, dict):
        val = data.get("preco_atual", None)
    elif isinstance(data, (int, float)):
        val = data
    else:
        val = None

    if val is None or val == 0:
        return None

    return round(float(val), decimals)


def _safe_int(ext: dict, key: str) -> Optional[int]:
    """Extrai valor inteiro de external_markets."""
    data = ext.get(key, {})
    if isinstance(data, dict):
        val = data.get("preco_atual", None)
    elif isinstance(data, (int, float)):
        val = data
    else:
        val = None

    if val is None:
        return None
    return int(val)


# ============================================================
# CONSTRUTORES DE SEÇÕES
# ============================================================

def _build_price(event_data: dict) -> dict:
    """Constrói seção de preço."""
    ohlc = (
        event_data.get("contextual_snapshot", {}).get("ohlc", {})
        or {}
    )

    close = (
        event_data.get("preco_fechamento")
        or event_data.get("raw_event", {}).get("preco_fechamento")
        or ohlc.get("close")
        or 0
    )

    price: dict[str, Any] = {
        "c": _safe_round(close),
    }

    if ohlc:
        o = _safe_round(ohlc.get("open"))
        h = _safe_round(ohlc.get("high"))
        l_ = _safe_round(ohlc.get("low"))
        vw = _safe_round(ohlc.get("vwap"))
        if o and o != price["c"]:
            price["o"] = o
        if h:
            price["h"] = max(h, price["c"])  # FIX 1: sempre incluir h, garantir h >= c
        if l_ and l_ != price["c"]:
            price["l"] = l_
        if vw:
            price["vw"] = vw

    ia = event_data.get("institutional_analytics", {})
    pa = ia.get("profile_analysis", {})

    shape = pa.get("profile_shape", {}).get("shape")
    if shape:
        price["sh"] = shape

    pe = pa.get("poor_extremes", {})
    if pe:
        auction = pe.get("action_bias")
        if auction:
            price["auc"] = auction

        ph = pe.get("poor_high", {})
        if isinstance(ph, dict) and ph.get("detected"):
            price["ph"] = 1

        pl = pe.get("poor_low", {})
        if isinstance(pl, dict) and pl.get("detected"):
            price["pl"] = 1

    # FIX 4a: breakout_risk do Volume Profile compression
    va_pct = pa.get("va_volume_pct", {})
    if isinstance(va_pct, dict):
        brk_risk = va_pct.get("breakout_risk")
        compression = va_pct.get("compression_signal", 0)
        if brk_risk and brk_risk in ("HIGH", "VERY_HIGH"):
            price["brk_risk"] = {"HIGH": "HI", "VERY_HIGH": "V_HI"}.get(brk_risk, brk_risk[:4])
        elif compression:
            price["brk_risk"] = "MOD"

    return price


def _calculate_regime_consensus(event_data: dict) -> dict:
    """
    Calcula consenso de regime ponderado por timeframe + macro.

    ANTES: 12+ campos conflitantes (regime_analysis, market_environment,
           multi_tf regimes, cross_asset, fear_greed) → LLM flip-flop.
    AGORA: 1 campo 'consensus' com confidence e conflitos explícitos.

    Pesos: 1d=4, 4h=3, 1h=2, 15m=1 (TFs maiores dominam).
    Fear&Greed extremo (<20 ou >80) adiciona 2 votos.
    """
    votes_bull = 0
    votes_bear = 0
    total_weight = 0

    # --- Votos dos timeframes (tendência) ---
    weights = {"1d": 4, "4h": 3, "1h": 2, "15m": 1}
    multi_tf = event_data.get("multi_tf", {})
    dominant_tf = ""
    max_adx = 0

    for tf_name, weight in weights.items():
        tf_data = multi_tf.get(tf_name, {})
        if not tf_data:
            continue
        tendencia = tf_data.get("tendencia", "")
        if tendencia in ("Alta",):
            votes_bull += weight
            total_weight += weight
        elif tendencia in ("Baixa",):
            votes_bear += weight
            total_weight += weight

        adx = tf_data.get("adx", 0) or 0
        if adx > max_adx:
            max_adx = adx
            dominant_tf = tf_name

    # --- Fear & Greed extremo ---
    ext = event_data.get("external_markets", {})
    fg_entry = ext.get("FEAR_GREED", {})
    fg_val = fg_entry.get("preco_atual", 50) if isinstance(fg_entry, dict) else 50
    try:
        fg_val = int(fg_val)
    except (TypeError, ValueError):
        fg_val = 50

    if fg_val < 20:
        votes_bear += 2
        total_weight += 2
    elif fg_val > 80:
        votes_bull += 2
        total_weight += 2

    # --- Volatility regime (informativo, não vota) ---
    env = event_data.get("market_environment", {})
    vol_regime = env.get("volatility_regime", "")
    vol_short = vol_regime[0] if vol_regime in ("LOW", "MEDIUM", "HIGH") else vol_regime[:3]

    # FIX 3c: usar regime_analysis.current_regime como fonte primária de mode
    _regime_to_mode: dict[str, str] = {
        "MEAN_REVERTING": "MR",
        "MEAN_REVERSION": "MR",
        "TRENDING":       "TRD",
        "BREAKOUT":       "BRK",
        "RANGE":          "RB",
        "RANGE_BOUND":    "RB",
    }

    regime_analysis = event_data.get("regime_analysis", {})
    current_regime = regime_analysis.get("current_regime", "")
    mode_short = _regime_to_mode.get(current_regime.upper(), "") if current_regime else ""

    # fallback: inferir do market_environment se regime_analysis ausente
    if not mode_short:
        ms = str(env.get("market_structure", "")).upper()
        mode_short = _regime_to_mode.get(ms, ms[:3] if ms else "")

    # --- Consensus calculation ---
    total_weight = max(total_weight, 1)
    bull_pct = round(votes_bull / total_weight * 100)
    bear_pct = round(votes_bear / total_weight * 100)

    if votes_bull > votes_bear * 1.5:
        consensus = "BULL"
    elif votes_bear > votes_bull * 1.5:
        consensus = "BEAR"
    else:
        consensus = "MIX"

    conf = round(abs(votes_bull - votes_bear) / total_weight, 2)

    result: dict[str, Any] = {
        "cs": consensus,
        "cf": conf,
    }

    if vol_short:
        result["v"] = vol_short

    if mode_short:
        result["mode"] = mode_short

    if dominant_tf:
        result["dom"] = dominant_tf

    # Flag conflict explicitly when strong disagreement exists
    if votes_bull > 0 and votes_bear > 0:
        result["bull%"] = bull_pct
        result["bear%"] = bear_pct

    vol_metrics = event_data.get("volatility_metrics", {})
    bbw = vol_metrics.get("bbw")
    atr_pct = vol_metrics.get("atr_pct")
    if bbw is not None:
        result["bbw"] = round(bbw, 2)
    if atr_pct is not None:
        result["atr%"] = round(atr_pct, 2)

    return result


def _build_regime(event_data: dict) -> dict:
    """
    Constrói seção de regime usando CONSENSO ponderado.

    ANTES: 5+ campos crus (v, tr, st, rgm, chg) que se contradiziam.
    AGORA: consensus (cs) + confidence (cf) + volatility (v) + dominant TF.
    """
    return _calculate_regime_consensus(event_data)


def _build_flow(event_data: dict) -> dict:
    """Constrói seção de fluxo (sem pressure/pa_signal/pa_conv redundantes)."""
    fluxo = event_data.get("fluxo_continuo", {})
    of = fluxo.get("order_flow", {})
    bsr_data = of.get("buy_sell_ratio", {})

    flow: dict[str, Any] = {
        "d1": compact_number(of.get("net_flow_1m", 0)),
    }

    delta = event_data.get("delta")
    vol = event_data.get("volume_total")
    vol_buy = event_data.get("volume_compra")
    if delta is not None:
        flow["delta"] = round(delta, 3)
    if vol:
        flow["vol"] = round(vol, 3)
    if vol_buy and vol:
        flow["buy_pct"] = round(vol_buy / vol * 100)
        
    micro = event_data.get("ml_features", {}).get("microstructure", {})
    if micro:
        ti = micro.get("trade_intensity_v2")
        trs = micro.get("tick_rule_sum")
        obs_slope = micro.get("order_book_slope")
        if ti is not None:
            flow["ti"] = round(ti, 1)
        if trs is not None:
            flow["trs"] = int(trs)
        if obs_slope is not None and abs(obs_slope - 1.0) > 0.01:
            flow["obs"] = round(obs_slope, 3)

    sf = fluxo.get("sector_flow", {})
    if sf:
        for cat in ("whale", "retail"):
            d = sf.get(cat, {}).get("delta")
            if d is not None and abs(d) > 0.001:
                flow[f"sf_{cat[0]}"] = round(d, 3)

    d5 = of.get("net_flow_5m")
    d15 = of.get("net_flow_15m")
    if d5 is not None:
        flow["d5"] = compact_number(d5)
    if d15 is not None:
        flow["d15"] = compact_number(d15)

    cvd = fluxo.get("cvd", 0)
    if cvd:
        flow["cvd"] = round(cvd, 1)

    imb = of.get("flow_imbalance", 0)
    flow["imb"] = round(imb, 2)

    ab = of.get("aggressive_buy_pct")
    if ab is not None:
        flow["ab"] = round(ab)

    bsr = bsr_data.get("buy_sell_ratio")
    if bsr is not None:
        flow["bsr"] = round(bsr, 2)

    # Passive/Aggressive composite (T6)
    ia = event_data.get("institutional_analytics", {})
    pa_data = ia.get("flow_analysis", {}).get("passive_aggressive", {})
    pa_comp = pa_data.get("composite") or pa_data.get("composite_signal")
    conv_map = {"HIGH": "H", "MEDIUM": "M", "LOW": "L", "STRONG": "H", "WEAK": "L"}
    if isinstance(pa_comp, dict):
        # Caso normal: composite é dict com signal/conviction
        signal = pa_comp.get("signal", "")
        if signal:
            flow["pa"] = signal[:10]
        conv = pa_comp.get("conviction", "")
        if conv:
            flow["conv"] = conv_map.get(str(conv).upper(), str(conv)[0])
    elif isinstance(pa_comp, str) and pa_comp:
        flow["pa"] = pa_comp[:10]
        pa_conv = pa_data.get("conviction") or pa_data.get("conviction_level")
        if pa_conv:
            flow["conv"] = conv_map.get(str(pa_conv).upper(), str(pa_conv)[0])

    # FIX 4b: absorption_strength (buyer_strength / seller_exhaustion)
    abs_data = fluxo.get("absorption_analysis", {}).get("current_absorption", {})
    if abs_data:
        buyer_str = abs_data.get("buyer_strength")
        seller_exh = abs_data.get("seller_exhaustion")
        cont_prob = abs_data.get("continuation_probability")
        if buyer_str is not None and seller_exh is not None:
            flow["abs_buy_str"] = round(buyer_str, 1)
            flow["abs_sell_exh"] = round(seller_exh, 1)
        if cont_prob is not None and cont_prob > 0.1:
            flow["abs_cont"] = round(cont_prob, 2)

    return flow


def _build_orderbook(event_data: dict) -> dict:
    """Constrói seção de orderbook (números compactos)."""
    ob_data = event_data.get("orderbook_data", {})
    depth = event_data.get("order_book_depth", {})

    bid = ob_data.get("bid_depth_usd", 0)
    ask = ob_data.get("ask_depth_usd", 0)

    imb_val = round(ob_data.get("flow_imbalance", 0), 2)
    ob: dict[str, Any] = {
        "b": compact_number(bid, force_sign=False),
        "a": compact_number(ask, force_sign=False),
        "imb": imb_val,
        "bias": "BUY" if imb_val > 0.1 else "SELL" if imb_val < -0.1 else "NEUT",
    }

    t5_imb = depth.get("L5", {}).get("flow_imbalance")
    if t5_imb is not None:
        ob["t5"] = round(t5_imb, 2)

    # FIX #8: Extrair spread_percent do orderbook_data
    spread_percent = ob_data.get("spread_percent")
    if spread_percent is None or spread_percent == 0:
        # Tentar calcular se dados dispon?veis
        mid = ob_data.get("mid")
        spread = ob_data.get("spread")
        if mid and mid > 0 and spread is not None and spread > 0:
            spread_percent = (spread / mid) * 100.0

    if spread_percent is not None and spread_percent > 0:
        ob["spread_pct"] = round(spread_percent, 4)

    mi = event_data.get("market_impact", {}).get("slippage_matrix", {})
    s100 = mi.get("100k_usd", {})
    if s100:
        ob["slip_b"] = round(s100.get("buy", 0) * 100, 2)
        ob["slip_s"] = round(s100.get("sell", 0) * 100, 2)

    return ob


_ML_EXTREME_HIGH: float = 0.95
_ML_EXTREME_LOW: float = 0.05

_WHALE_CLS_MAP: dict[str, str] = {
    "MILD_ACCUMULATION": "MA",
    "STRONG_ACCUMULATION": "SA",
    "MILD_DISTRIBUTION": "MD",
    "STRONG_DISTRIBUTION": "SD",
    "NEUTRAL": "N",
}


_WHALE_DIV_MAP: dict[str, str] = {
    "smart_money_accumulation": "smart_accu",
    "smart_money_distribution": "smart_dist",
    "retail_accumulation": "retail_accu",
    "retail_distribution": "retail_dist",
    "divergence_bullish": "div_bull",
    "divergence_bearish": "div_bear",
    "neutral": "neut",
}


def _build_whale(event_data: dict) -> dict:
    """Retorna whale score + classificação + divergência (T3)."""
    ia = event_data.get("institutional_analytics", {})
    fa = ia.get("flow_analysis", {})
    wa = fa.get("whale_accumulation", {})

    score = wa.get("score", 0)
    score = int(score) if isinstance(score, (int, float)) else 0
    if score == 0:
        return {}

    result: dict[str, Any] = {"s": score}

    cls = wa.get("classification")
    if cls:
        result["c"] = _WHALE_CLS_MAP.get(cls, cls[:2])

    div = wa.get("smart_money_divergence") or wa.get("divergence_type")
    if div:
        result["div"] = _WHALE_DIV_MAP.get(str(div).lower(), str(div)[:12])

    return result


def _build_quant(event_data: dict) -> dict:
    """Constrói seção quantitativa (ML) — FIX 2: flag extreme predictions."""
    ml = event_data.get("ml_prediction", {}) or event_data.get("quant_prediction", {})
    if not ml:
        return {"pu": 0.5, "c": 0.0}

    prob_up = ml.get("prob_up", 0.5)
    confidence = ml.get("confidence", 0)

    is_extreme = (
        ml.get("extreme_filtered", False)
        or prob_up > _ML_EXTREME_HIGH
        or prob_up < _ML_EXTREME_LOW
    )

    quant: dict[str, Any] = {
        "pu": round(prob_up, 2),
        "c": round(confidence, 2),
    }

    if is_extreme:
        quant["unreliable"] = True
        quant["reason"] = "extreme_prob"

    return quant


def _build_timeframes(event_data: dict) -> dict:
    """
    Constrói seção de timeframes.
    FILTRA timeframes sem dados reais (tendência NE, RSI null).
    """
    multi_tf = event_data.get("multi_tf", {})
    tf_section: dict[str, dict] = {}

    for tf_name in ["15m", "1h", "4h", "1d"]:
        tf = multi_tf.get(tf_name, {})
        if not tf:
            continue

        tendencia = tf.get("tendencia")
        rsi = tf.get("rsi_short")

        if not tendencia or tendencia == "NE" or rsi is None:
            continue

        tf_entry: dict[str, Any] = {
            "t": TREND_ABBREV.get(tendencia, tendencia[:3]),
        }

        if rsi is not None:
            tf_entry["rsi"] = round(rsi)

        macd = tf.get("macd")
        macd_signal = tf.get("macd_signal")
        if macd is not None and macd_signal is not None:
            tf_entry["macd"] = [round(macd), round(macd_signal)]

        adx = tf.get("adx")
        if adx is not None:
            tf_entry["adx"] = round(adx)

        atr = tf.get("atr")
        if atr is not None:
            tf_entry["atr"] = round(atr)

        regime = tf.get("regime", "")
        if regime:
            tf_entry["r"] = REGIME_ABBREV.get(regime, regime[:3])

        tf_section[tf_name] = tf_entry

    return tf_section


def _build_ext_indicators(event_data: dict) -> dict:
    """
    Constrói seção de indicadores estendidos (sobrecompra/sobrevenda).
    Fonte: event_data["technical_indicators_extended"]

    Inclui: CCI, Stochastic K, Williams %R, GARCH forecast.
    Crítico para evitar BUY em condições de OVERBOUGHT.
    """
    ext_ind = event_data.get("technical_indicators_extended", {})
    ext: dict[str, Any] = {}


    # CCI — Commodity Channel Index
    cci_signal = ext_ind.get("cci_signal")
    if cci_signal:
        if cci_signal == "OVERBOUGHT":
            ext["cci"] = "OB"
        elif cci_signal == "OVERSOLD":
            ext["cci"] = "OS"
        else:
            ext["cci"] = "N"

    # Stochastic %K + signal
    stochastic = ext_ind.get("stochastic", {})
    stoch_k = stochastic.get("k") if isinstance(stochastic, dict) else None
    if stoch_k is not None:
        ext["stoch"] = round(stoch_k)
        # FIX 3b: incluir signal se não neutro
        stoch_sig = (
            stochastic.get("signal", "")
            if isinstance(stochastic, dict) else ""
        )
        if stoch_sig and stoch_sig.upper() not in ("NEUTRAL", ""):
            _stoch_sig_map = {
                "OVERBOUGHT": "OB",
                "OVERSOLD":   "OS",
            }
            ext["stoch_sig"] = _stoch_sig_map.get(
                stoch_sig.upper(), stoch_sig[:2].upper()
            )

    # Williams %R
    williams = ext_ind.get("williams_r", {})
    wr_value = williams.get("value") if isinstance(williams, dict) else None
    if wr_value is not None:
        ext["wr"] = round(wr_value)
    elif isinstance(williams, dict) and "zone" in williams:
        # Fallback: inferir valor aproximado da zone quando value ausente
        zone = williams.get("zone", "neutral")
        ext["wr"] = -10 if zone == "overbought" else -90 if zone == "oversold" else -50

    # GARCH forecast de volatilidade 1h
    garch = ext_ind.get("garch_forecast_1h")
    if garch is not None:
        ext["garch"] = round(float(garch), 2)

    # Hurst Exponent (#23): H>0.5=trending, H<0.5=mean-rev
    hurst = ext_ind.get("hurst_exponent")
    if hurst is not None:
        ext["hurst"] = round(float(hurst), 2)

    # Shannon Entropy (#25): alta=ruído, baixa=previsível
    entropy = ext_ind.get("shannon_entropy")
    if entropy is not None:
        ext["entropy"] = round(float(entropy), 2)

    # Fractal Dimension (#24): <1.5=trending, >1.5=ruído
    frac = ext_ind.get("fractal_dimension")
    if frac is not None:
        ext["fd"] = round(float(frac), 2)

    # Kalman Filter (#27): preço suavizado vs raw
    kalman = ext_ind.get("kalman_filter")
    if kalman and isinstance(kalman, dict):
        ext["kalman"] = {
            "kp": kalman.get("kalman_price"),
            "dev": kalman.get("deviation_pct"),
            "dir": kalman.get("trend_direction"),
        }

    # Regression Channel (#28): posição no canal
    reg = ext_ind.get("regression_channel")
    if reg and isinstance(reg, dict):
        ext["reg"] = {
            "sl": reg.get("slope_per_bar"),
            "pos": reg.get("position_in_channel"),
            "dev": reg.get("deviation_from_trend"),
        }

    # Monte Carlo (#18): prob_up e percentis
    mc = ext_ind.get("monte_carlo")
    if mc and isinstance(mc, dict):
        ext["mc"] = {
            "pu": mc.get("prob_up"),
            "p10": mc.get("p10"),
            "p90": mc.get("p90"),
        }

    # Dominant Cycles (#26): períodos dominantes
    cyc = ext_ind.get("dominant_cycles")
    cyc_list = (cyc or {}).get("dominant_cycles") if isinstance(cyc, dict) else None
    if cyc_list:
        ext["cycles"] = list(cyc_list)[:2]  # top 2 ciclos

    # Smart Money: FVG + BOS (do pattern_recognition)
    pr = event_data.get("pattern_recognition", {})
    sm = pr.get("smart_money", {}) if isinstance(pr, dict) else {}
    if sm:
        fvg_list = sm.get("fair_value_gaps", [])
        bos = sm.get("market_structure", {})
        smc: dict[str, Any] = {}
        if fvg_list:
            smc["fvg"] = len(fvg_list)
            smc["fvg_last"] = fvg_list[-1].get("type", "")[:2]  # BU/BE
        if bos and isinstance(bos, dict):
            smc["struct"] = bos.get("structure", "")[:4]
            smc["bos"] = bos.get("bos_detected", False)
        if smc:
            ext["smc"] = smc

    # Fibonacci retracement levels (#10)
    # Fonte: event_data["fibonacci_levels"] ou pattern_recognition
    pr = event_data.get("pattern_recognition", {})
    fib = event_data.get("fibonacci_levels") or (pr.get("fibonacci_levels", {}) if isinstance(pr, dict) else {})
    if fib and isinstance(fib, dict):
        sh = fib.get("swing_high") or fib.get("high")
        sl = fib.get("swing_low") or fib.get("low")
        if sh and sl and float(sh) > float(sl) > 0:
            fib_entry: dict[str, Any] = {
                "hi": round(float(sh)),
                "lo": round(float(sl)),
            }
            for lvl in ("38.2", "61.8"):
                v = fib.get(lvl)
                if v is not None:
                    fib_entry[lvl.replace(".", "")] = round(float(v))
            ext["fib"] = fib_entry

    # FIX 3b: Candlestick patterns (de institutional_analytics)
    ia = event_data.get("institutional_analytics", {})
    candles = ia.get("candlestick_patterns", {})
    if candles and isinstance(candles, dict):
        dominant = candles.get("dominant_signal", "")
        patterns = candles.get("patterns", [])
        max_conf = candles.get("max_confidence", 0) or 0

        if dominant and dominant != "neutral" and max_conf >= 0.65:
            cand: dict[str, Any] = {
                "sig":  dominant[:4],
                "conf": round(float(max_conf), 2),
            }
            if patterns and isinstance(patterns, list) and len(patterns) > 0:
                top = patterns[0]
                if isinstance(top, dict):
                    cand["name"] = str(top.get("name", ""))[:12]
            ext["candle"] = cand

    return ext


def _build_alerts(event_data: dict) -> list:
    """
    Retorna alertas HIGH/CRITICAL ativos (máx 3) — P2.
    Fonte: event_data["alerts"]["active_alerts"]
    """
    alerts_data = event_data.get("alerts", {})
    active = alerts_data.get("active_alerts", [])
    if not active:
        return []

    high = [a for a in active if a.get("severity") in ("HIGH", "CRITICAL")]
    result = []
    for alert in high[:3]:
        sev = alert.get("severity", "")
        entry: dict[str, Any] = {
            "type": str(alert.get("type", "UNKNOWN"))[:15],
            "sev": sev[0] if sev else "?",
        }
        lvl = alert.get("level")
        if lvl is not None:
            entry["lvl"] = round(float(lvl))
        result.append(entry)
    return result


def _build_sr(event_data: dict) -> dict:
    """
    Retorna top 2 suportes e resistências com confluência — P6.
    FIX 3.5: Fonte canônica agora é defense_zones (mais completo que sr_strength).
    """
    sr_analysis = (
        event_data.get("institutional_analytics", {}).get("sr_analysis", {})
    )
    if not sr_analysis:
        return {}

    dz = sr_analysis.get("defense_zones", {})
    if not dz or dz.get("status") == "error":
        return {}

    sr: dict[str, Any] = {}

    close = event_data.get("preco_fechamento", 0)

    # sell_defense = zonas de resistência (acima do preço)
    for i, zone in enumerate(dz.get("sell_defense", [])[:2]):
        price_r = zone.get("center")
        if price_r is not None:
            key = f"r{i+1}"
            sr[key] = [round(float(price_r)), round(zone.get("strength", 0), 1)]
            if close:
                sr[f"{key}_dist"] = abs(round(float(close) - float(price_r)))
            sources = zone.get("source_count", zone.get("sources", 0))
            if sources and sources >= 3:
                sr[f"{key}_conf"] = sources

    # buy_defense = zonas de suporte (abaixo do preço)
    for i, zone in enumerate(dz.get("buy_defense", [])[:2]):
        price_s = zone.get("center")
        if price_s is not None:
            key = f"s{i+1}"
            sr[key] = [round(float(price_s)), round(zone.get("strength", 0), 1)]
            if close:
                sr[f"{key}_dist"] = abs(round(float(close) - float(price_s)))
            sources = zone.get("source_count", zone.get("sources", 0))
            if sources and sources >= 3:
                sr[f"{key}_conf"] = sources

    defense = dz.get("defense_asymmetry", {})
    bias = defense.get("bias")
    if bias:
        sr["def_bias"] = str(bias)[:10]

    return sr


def _build_static_context(event_data: dict) -> dict:
    """
    Constrói contexto estático — dados que mudam lentamente.
    Enviado a cada 5 min OU quando evento importante acontece.
    """
    ctx_market = event_data.get("market_context", {})
    ext = event_data.get("external_markets", {})
    vp = event_data.get("historical_vp", {}).get("daily", {})
    deriv = event_data.get("derivatives", {})
    ml_cross = event_data.get("ml_features", {}).get("cross_asset", {})

    session_raw = (
        f"{ctx_market.get('trading_session', '')}_"
        f"{ctx_market.get('session_phase', '')}"
    )
    ses = SESSION_ABBREV.get(session_raw, session_raw[:8])

    ctx: dict[str, Any] = {
        "ses": ses,
    }

    # --- Mercados Externos ---
    dxy = _safe_price(ext, "DXY")
    if dxy:
        ctx["dxy"] = dxy

    tnx = _safe_price(ext, "TNX")
    if tnx:
        ctx["tnx"] = tnx

    spx = _safe_price(ext, "SP500")
    if spx:
        ctx["spy"] = spx  # FIX 3: ticker é SPY (ETF ~$669), não índice S&P 500 (~$5700)

    ndx = _safe_price(ext, "NASDAQ")
    if ndx:
        ctx["ndx"] = ndx

    gold = _safe_price(ext, "GOLD", decimals=0)
    if gold:
        ctx["gold"] = int(gold)

    wti = _safe_price(ext, "WTI")
    if wti:
        ctx["wti"] = wti

    vix = _safe_price(ext, "VIX", decimals=1)
    if vix:
        ctx["vix"] = vix

    fg = _safe_int(ext, "FEAR_GREED")
    if fg is not None:
        ctx["fg"] = fg

    # --- Volume Profile Diário (só se dados válidos) ---
    if vp.get("status") == "success":
        poc = vp.get("poc")
        if poc:
            ctx["poc"] = int(poc)
        val_ = vp.get("val")
        if val_:
            ctx["val"] = int(val_)
        vah = vp.get("vah")
        if vah:
            ctx["vah"] = int(vah)

    # --- Derivativos ---
    btc_deriv = deriv.get("BTCUSDT", {})
    eth_deriv = deriv.get("ETHUSDT", {})

    lsr = btc_deriv.get("long_short_ratio")
    if lsr:
        ctx["lsr"] = round(lsr, 2)

    eth_lsr = eth_deriv.get("long_short_ratio")
    if eth_lsr:
        ctx["eth_lsr"] = round(eth_lsr, 2)

    oi = btc_deriv.get("open_interest")
    if oi and oi > 0:
        ctx["oi"] = round(oi / 1000)

    fr = btc_deriv.get("funding_rate_percent")
    if fr is not None:
        ctx["fr"] = round(fr, 4) if fr is not None else 0.0

    longs = btc_deriv.get("longs_usd")
    shorts = btc_deriv.get("shorts_usd")
    if longs:
        ctx["longs"] = compact_number(longs)
    if shorts:
        ctx["shorts"] = compact_number(shorts)

    # --- Correlações ---
    eth7 = ml_cross.get("btc_eth_corr_7d")
    if eth7:
        ctx["eth7"] = round(eth7, 1)

    dxy30 = ml_cross.get("btc_dxy_corr_30d")
    if dxy30:
        ctx["dxy30"] = round(dxy30, 2)

    return ctx


# ============================================================
# BUILDERS DOS GAPS CRÍTICOS
# ============================================================

def _build_ofi(event_data: dict) -> dict:
    """
    Order Flow Imbalance — pressão líquida no livro.
    Fonte 1: institutional_analytics.order_flow_imbalance
    Fonte 2: fluxo_continuo.order_flow.flow_imbalance (mais confiável)
    Fonte 3: ml_features.microstructure.flow_imbalance
    """
    ia = event_data.get("institutional_analytics", {})
    ofi_raw = ia.get("order_flow_imbalance", {})

    if ofi_raw and isinstance(ofi_raw, dict):
        score = ofi_raw.get("score") or ofi_raw.get("ofi_score")
        direction = ofi_raw.get("direction", "")
        if score is not None:
            return {
                "score": round(float(score), 3),
                "dir": str(direction)[:4].upper() or "N",
            }

    # FIX 3e: usar order_flow como fonte secundária (mais rico que microstructure)
    fluxo = event_data.get("fluxo_continuo", {})
    of = fluxo.get("order_flow", {})
    fi_of = of.get("flow_imbalance")

    if fi_of is not None:
        direction = "BUY" if fi_of > 0.05 else "SELL" if fi_of < -0.05 else "NEU"
        return {
            "score": round(float(fi_of), 3),
            "dir": direction,
            "src": "order_flow",
        }

    # Fallback final: microstructure
    micro = event_data.get("ml_features", {}).get("microstructure", {})
    fi = micro.get("flow_imbalance")
    if fi is not None:
        direction = "BUY" if fi > 0.05 else "SELL" if fi < -0.05 else "NEU"
        return {
            "score": round(float(fi), 3),
            "dir": direction,
            "src": "micro",
        }

    return {}


def _build_vwap_context(event_data: dict) -> dict:
    """
    Desvio do VWAP — posição relativa ao fair value intraday.
    Fontes: institutional/vwap_twap, contextual_snapshot.ohlc.vwap
    """
    ia = event_data.get("institutional_analytics", {})
    vwap_data = ia.get("vwap_twap", {})

    if vwap_data and isinstance(vwap_data, dict):
        dev_pct = vwap_data.get("deviation_pct") or vwap_data.get("vwap_deviation_pct")
        signal = vwap_data.get("signal", "")
        if dev_pct is not None:
            side = "above" if float(dev_pct) > 0 else "below"
            return {
                "dev": round(float(dev_pct), 3),
                "side": side,
                "sig": str(signal)[:10] if signal else side,
            }

    # Fallback: calcular via ohlc.vwap e preco_fechamento
    close = event_data.get("preco_fechamento", 0) or 0
    ohlc = event_data.get("contextual_snapshot", {}).get("ohlc", {})
    vwap = ohlc.get("vwap", 0) or 0

    if close and vwap and vwap > 0:
        dev = (close - vwap) / vwap * 100
        side = "above" if dev > 0 else "below"
        signal = "premium" if dev > 0.1 else "discount" if dev < -0.1 else "fair"
        return {
            "dev": round(dev, 3),
            "side": side,
            "sig": signal,
            "src": "ohlc",
        }

    return {}


def _build_iceberg(event_data: dict) -> dict:
    """
    Detecção de ordens iceberg — liquidez oculta.
    Fonte: institutional_analytics ou whale_activity
    """
    ia = event_data.get("institutional_analytics", {})
    iceberg_data = ia.get("iceberg_detector", {})

    if iceberg_data and isinstance(iceberg_data, dict):
        detected = iceberg_data.get("detected", False)
        if detected:
            return {
                "det": 1,
                "side": str(iceberg_data.get("side", ""))[:4].upper(),
                "sz": str(iceberg_data.get("estimated_size", ""))[:6],
            }
        return {}

    # Fallback: whale_activity do payload bruto
    wa = event_data.get("whale_activity", {})
    if isinstance(wa, dict) and wa.get("iceberg_activity"):
        return {"det": 1, "src": "whale_activity"}

    return {}


def _build_liquidity_clusters(event_data: dict) -> list:
    """
    Top 2 clusters de liquidez — ímãs de preço de curto prazo.
    Fonte: fluxo_continuo.liquidity_heatmap
    """
    fluxo = event_data.get("fluxo_continuo", {})
    heatmap = fluxo.get("liquidity_heatmap", {})
    clusters = heatmap.get("clusters", [])

    if not clusters:
        return []

    close = event_data.get("preco_fechamento", 0) or 0
    result = []

    sorted_clusters = sorted(
        clusters,
        key=lambda c: abs((c.get("center") or 0) - close)
    )

    for c in sorted_clusters[:2]:
        center = c.get("center")
        imb = c.get("imbalance_ratio") or c.get("imbalance", 0)
        if center is None:
            continue
        side = "sell" if (imb or 0) < 0 else "buy"
        entry: dict[str, Any] = {
            "p": round(float(center)),
            "side": side,
        }
        vol = c.get("total_volume")
        if vol:
            entry["vol"] = round(float(vol), 2)
        result.append(entry)

    return result


def _build_smart_money_score(event_data: dict) -> dict:
    """
    Smart Money Score — footprint institucional agregado.
    Fonte: institutional_analytics.smart_money
    """
    ia = event_data.get("institutional_analytics", {})
    sm_data = ia.get("smart_money", {})

    if not sm_data or not isinstance(sm_data, dict):
        return {}

    score = sm_data.get("score") or sm_data.get("smart_money_score")
    signal = sm_data.get("signal") or sm_data.get("direction", "")

    if score is None:
        return {}

    return {
        "score": round(float(score), 3),
        "sig": str(signal)[:10] if signal else "",
    }


def _build_cvd_divergence(event_data: dict) -> dict:
    """
    Divergência CVD vs preço — confirmação ou negação do movimento.
    Fonte: institutional_analytics.cvd ou fluxo_continuo.cvd
    """
    ia = event_data.get("institutional_analytics", {})
    cvd_data = ia.get("cvd", {})

    if cvd_data and isinstance(cvd_data, dict):
        div = cvd_data.get("divergence") or cvd_data.get("price_cvd_divergence", {})
        if div and isinstance(div, dict):
            detected = div.get("detected", False)
            if detected:
                return {
                    "det": 1,
                    "type": str(div.get("type", ""))[:12],
                    "bars": div.get("bars", 0),
                }
        return {}

    # Fallback: usar CVD bruto + tendência de preço para inferir
    fluxo = event_data.get("fluxo_continuo", {})
    cvd_val = fluxo.get("cvd", 0) or 0
    multi_tf = event_data.get("multi_tf", {})
    trend_1h = multi_tf.get("1h", {}).get("tendencia", "")

    if abs(cvd_val) > 0.5:
        price_up = trend_1h in ("Alta",)
        cvd_up = cvd_val > 0
        diverging = price_up != cvd_up

        if diverging:
            div_type = "bearish_div" if price_up and not cvd_up else "bullish_div"
            return {
                "det": 1,
                "type": div_type,
                "src": "inferred",
            }

    return {}


def _build_mean_reversion_score(event_data: dict) -> dict:
    """
    Score de mean reversion — quão esticado está o preço.
    Fonte: institutional_analytics.mean_reversion
    """
    ia = event_data.get("institutional_analytics", {})
    mr_data = ia.get("mean_reversion", {})

    if mr_data and isinstance(mr_data, dict):
        score = mr_data.get("score") or mr_data.get("reversion_score")
        signal = mr_data.get("signal", "")
        if score is not None:
            return {
                "score": round(float(score), 3),
                "sig": str(signal)[:15] if signal else "",
            }

    # Fallback: inferir via Hurst + posição no regression channel
    # FIX 3d: threshold reduzido de 0.3 para 0.2, hurst de 0.45 para 0.48
    ext = event_data.get("technical_indicators_extended", {})
    hurst = ext.get("hurst_exponent")
    reg = ext.get("regression_channel", {})
    pos = reg.get("position_in_channel") if isinstance(reg, dict) else None

    if hurst is not None and pos is not None:
        h = float(hurst)
        p = float(pos)

        if h < 0.48 and abs(p - 0.5) > 0.2:
            strength = round((0.5 - h) * abs(p - 0.5) * 4, 3)
            if p < 0.35:
                sig = "stretched_bear"
            elif p > 0.65:
                sig = "stretched_bull"
            else:
                sig = "mild_stretch"

            return {
                "score": strength,
                "sig": sig,
                "src": "inferred",
            }

    return {}


# ============================================================
# ESCALA GRADUAL DE COMPACTAÇÃO DO SUMMARY
# ============================================================

_SUMMARY_THRESHOLDS = [
    (3500, "full"),             # Limite auditado de qualidade (Prompt E)
    (4500, "truncate_notes"),    # Compactação leve
    (5500, "essentials_only"),   # Compactação agressiva
    (6144, "remove"),            # Segurança do pipeline (Hard Limit)
]

_SUMMARY_ESSENTIAL_KEYS: dict[str, set] = {
    "flow":          {"bias", "type", "conf", "actor"},
    "sr":            {"nearest", "compressed", "conf_bias"},
    "regime":        {"label", "strategies"},
    "institutional": {"whale_bias", "profile_bias", "unfinished"},
    "quality":       {"reliable", "confidence_cap", "issues"},
}

_SUMMARY_REQUIRED_KEYS: dict[str, set] = {
    "flow":          {"bias", "type", "actor", "conf", "note"},
    "sr":            {"nearest", "compressed", "conf_bias", "note"},
    "regime":        {"label", "strategies", "avoid", "duration", "note"},
    "institutional": {"auction_state", "whale_bias", "profile_bias", "unfinished", "note"},
    "quality":       {"reliable", "confidence_cap", "issues", "note"},
}


def _compact_summary_by_level(summary: dict, level: str) -> dict:
    """Compacta o summary progressivamente conforme o nível."""
    if level == "full":
        return summary

    if level == "truncate_notes":
        # Mantém todos os campos obrigatórios, remove extras desconhecidos,
        # trunca notas a 100 chars e listas a 2 itens
        compacted = {}
        for key, section in summary.items():
            required = _SUMMARY_REQUIRED_KEYS.get(key, {"note"})
            compacted[key] = {
                k: (v[:100] if isinstance(v, str) and k == "note" else
                    v[:2] if isinstance(v, list) else v)
                for k, v in section.items()
                if k in required
            }
        return compacted

    if level == "essentials_only":
        return {
            key: {
                k: (v[:2] if isinstance(v, list) else v)
                for k, v in section.items()
                if k in _SUMMARY_ESSENTIAL_KEYS.get(key, set())
            }
            for key, section in summary.items()
        }

    return {}


def _determine_compaction_level(size: int) -> str:
    """Determina o nível de compactação baseado no tamanho."""
    for threshold, level in _SUMMARY_THRESHOLDS:
        if size <= threshold:
            return level
    return "remove"


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def _build_summary_section(
    payload: dict,
    builders: "SummaryBuilderMap | None" = None,
) -> dict:
    """
    Constrói seção de resumos interpretativos.
    Cada builder é isolado — falha em um não afeta os outros.
    """
    resolved = builders if builders is not None else _get_builders()
    if not resolved:
        return {}

    summary: dict[str, Any] = {}

    for name, builder in resolved.items():
        try:
            result = builder(payload)
            if result:
                summary[name] = result
        except Exception as exc:
            logger.warning(
                "SUMMARY_BUILD_ERROR: builder=%s error=%s",
                name,
                str(exc),
            )

    return summary


def build_compact_payload(
    event_data: dict,
    builders: "SummaryBuilderMap | None" = None,
) -> Dict[str, Any]:
    """
    Construtor Principal de Payload Compactado.
 para envio à LLM.

    IMPORTANTE: Usa keys LONGAS (price, flow, regime, etc.) para
    compatibilidade com o pipeline existente (ai_analyzer_qwen.py).
    Os VALORES são compactos (números abreviados, nulos filtrados).

    Returns:
        dict: Payload compactado (~200 tokens)
    """
    # --- TRIGGER ---
    trigger_raw = event_data.get("tipo_evento", "")
    trigger = TRIGGER_ABBREV.get(trigger_raw, trigger_raw[:6])

    # --- SEÇÕES ---
    price = _build_price(event_data)
    regime = _build_regime(event_data)
    flow = _build_flow(event_data)
    ob = _build_orderbook(event_data)
    whale = _build_whale(event_data)
    quant = _build_quant(event_data)
    tf_section = _build_timeframes(event_data)
    ext_indicators = _build_ext_indicators(event_data)
    alerts = _build_alerts(event_data)
    sr = _build_sr(event_data)
    static_ctx = _build_static_context(event_data)

    # ═══════════════════════════════════════════════════════════
    # MONTAR PAYLOAD COM KEYS LONGAS                ← CORRIGIDO
    # O pipeline (ai_analyzer_qwen.py linha 3499)
    # procura "price" no event_data para detectar
    # payload flat e empacotar como ai_payload.
    # ═══════════════════════════════════════════════════════════
    payload: dict[str, Any] = {
        "symbol": event_data.get("symbol", "BTCUSDT"),  # ← ADICIONADO
        "epoch_ms": event_data.get("epoch_ms") or int(time.time() * 1000),
        "trigger": trigger,                              # ← ERA "t"
        "price": price,                                  # ← ERA "p"
    }

    if regime:
        payload["regime"] = regime                       # ← ERA "r"

    ia_quality = event_data.get("institutional_analytics", {}).get("quality", {})
    reliability = event_data.get("data_reliability", {})

    latency_data = (ia_quality.get("latency", {}) or {})
    calendar_data = (ia_quality.get("calendar", {}) or {})

    lat = latency_data.get("latency_ms") or reliability.get("latency_ms")
    cat = latency_data.get("latency_category") or reliability.get("latency_category", "OK")
    # FIX 3a: preservar valor completo de liquidez (não truncar)
    liq = calendar_data.get("expected_liquidity") or reliability.get("expected_liquidity", "NORMAL")
    # FIX 3a: capturar feriado
    is_holiday = calendar_data.get("is_us_holiday", 0)
    holiday_name = calendar_data.get("holiday_name", "")

    if lat or cat != "OK" or liq != "NORMAL" or is_holiday:
        qual: dict[str, Any] = {}
        if cat and cat != "OK":
            qual["lat"] = cat[:4]
        # FIX 3a: não truncar liq — preservar VERY_LOW intacto
        if liq and liq != "NORMAL":
            qual["liq"] = liq
        if lat and lat > 2000:
            qual["ms"] = round(lat)
        # FIX 3a: incluir feriado no payload
        if is_holiday and holiday_name:
            qual["holiday"] = holiday_name[:25]
        if qual:
            payload["qual"] = qual

    # ═══════════════════════════════════════════════════════════
    # SEÇÕES OBRIGATÓRIAS: sempre incluir, mesmo que vazias.
    # IA sem flow/ob/tf/sr decide às cegas → pior que payload
    # um pouco maior com stubs indicando "no data".
    # ═══════════════════════════════════════════════════════════
    payload["flow"] = flow if flow else {}
    payload["ob"] = ob if ob else {}
    payload["tf"] = tf_section if tf_section else {"_": "no_data"}
    payload["sr"] = sr if sr else {"_": "no_data"}

    if whale:
        payload["w"] = whale

    if quant and quant.get("pu", 0.5) != 0.5:
        payload["quant"] = quant                         # ← ERA "q"

    if ext_indicators:
        payload["ext"] = ext_indicators

    if alerts:
        payload["alerts"] = alerts

    # Validação: avisar se seções obrigatórias estão com stub
    _required = {"flow", "ob", "tf", "sr"}
    _stub = {s for s in _required if isinstance(payload.get(s), dict) and "_" in payload[s]}
    if _stub:
        logger.warning(
            "PAYLOAD_INCOMPLETE: seções com stub (dados insuficientes): %s",
            sorted(_stub),
        )

    # --- CONTEXTO ESTÁTICO ---
    force_ctx = (
        trigger_raw in IMPORTANT_EVENTS
        or trigger in IMPORTANT_EVENTS
    )

    global _last_static_ctx, _last_static_ts

    if force_ctx or _should_send_static(static_ctx):
        payload["ctx"] = static_ctx
        ctx_status = "SENT" + (" (forced)" if force_ctx else "")
        # Atualizar cache
        _last_static_ctx = static_ctx
        _last_static_ts = time.time()
    else:
        # P3: ctx CACHED — enviar versão mínima com campos essenciais
        if _last_static_ctx:
            mini_ctx: dict[str, Any] = {
                "cached": True,
                "ses": _last_static_ctx.get("ses"),
            }
            # Incluir campos críticos para continuidade analítica
            for _key in ("lsr", "oi", "dxy30", "vix", "fg"):
                if _key in _last_static_ctx:
                    mini_ctx[_key] = _last_static_ctx[_key]
            payload["ctx"] = mini_ctx
        else:
            # Fallback (não deveria acontecer se cache funciona)
            payload["ctx"] = {"cached": True}
        ctx_status = "CACHED(mini)"

    # ═══════════════════════════════════════════════════════════
    # DADOS CRÍTICOS FALTANTES — gaps identificados na auditoria
    # Injetados ANTES do summary para que os builders possam usá-los
    # ═══════════════════════════════════════════════════════════

    ofi = _build_ofi(event_data)
    if ofi:
        payload["ofi"] = ofi

    vwap_ctx = _build_vwap_context(event_data)
    if vwap_ctx:
        payload["vwap"] = vwap_ctx

    iceberg = _build_iceberg(event_data)
    if iceberg:
        payload["iceberg"] = iceberg

    liq_clusters = _build_liquidity_clusters(event_data)
    if liq_clusters:
        payload["liq"] = liq_clusters

    sm_score = _build_smart_money_score(event_data)
    if sm_score:
        payload["sm"] = sm_score

    cvd_div = _build_cvd_divergence(event_data)
    if cvd_div:
        payload["cvd_div"] = cvd_div

    mr_score = _build_mean_reversion_score(event_data)
    if mr_score:
        payload["mr"] = mr_score

    # ═══════════════════════════════════════════════════════════
    # SUMMARY BUILDERS
    # ═══════════════════════════════════════════════════════════
    try:
        summary = _build_summary_section(payload, builders=builders)
    except TypeError:
        # Retrocompatibilidade com mocks que não aceitam o argumento builders=
        summary = _build_summary_section(payload)

    if summary:
        payload["summary"] = summary

    # ═══════════════════════════════════════════════════════════
    # ESCALA GRADUAL DE COMPACTAÇÃO
    # ═══════════════════════════════════════════════════════════
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    payload_size = len(payload_json)

    compaction_level = _determine_compaction_level(payload_size)

    if compaction_level == "remove":
        logger.error(
            "PAYLOAD_OVER_HARD_LIMIT: %d bytes — removendo summary",
            payload_size,
        )
        payload.pop("summary", None)
    elif compaction_level != "full":
        logger.warning(
            "PAYLOAD_COMPACTION: %d bytes — nível=%s",
            payload_size,
            compaction_level,
        )
        # Remover seção ext (indicadores secundários) — salva ~350 bytes
        payload.pop("ext", None)

        # Compactar TFs: remover macd/atr/r mas MANTER adx (crítico)
        if "tf" in payload:
            for tf_name in payload["tf"]:
                if isinstance(payload["tf"][tf_name], dict):
                    for k in ["macd", "atr", "r"]:
                        payload["tf"][tf_name].pop(k, None)

        if "summary" in payload:
            payload["summary"] = _compact_summary_by_level(
                payload["summary"], compaction_level
            )
            payload["summary"]["_compacted"] = compaction_level

    # Recalcular após compactação
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    payload_size = len(payload_json)
    estimated_tokens = payload_size // 4
    tf_keys = list(tf_section.keys()) if tf_section else "NONE"
    summary_keys = list(summary.keys()) if summary else "NONE"
    gaps_added = [k for k in ("ofi", "vwap", "iceberg", "liq", "sm", "cvd_div", "mr")
                  if k in payload]

    logger.info(
        "BUILD_COMPACT: OK | price=%s | size=%d chars | ~%d tokens | "
        "tf=%s | summary=%s | gaps=%s | compaction=%s | ctx=%s",
        price.get("c"),
        payload_size,
        estimated_tokens,
        tf_keys,
        summary_keys,
        gaps_added,
        compaction_level,
        ctx_status,
    )

    logger.debug("COMPACT_PAYLOAD_DEBUG: %s", payload_json)

    # ═══════════════════════════════════════════════════════════
    # DEBUG: Instrumentação de tamanho por seção (TEMPORÁRIA)
    # ═══════════════════════════════════════════════════════════
    if os.environ.get("PAYLOAD_DEBUG_SIZES") == "1":
        section_sizes = {}
        for key, value in payload.items():
            section_json = json.dumps({key: value}, ensure_ascii=False, separators=(",", ":"))
            section_sizes[key] = len(section_json)
        
        sorted_sections = sorted(section_sizes.items(), key=lambda x: x[1], reverse=True)
        
        print("\n" + "="*80)
        print(f"PAYLOAD_DEBUG_SIZES: total={payload_size} bytes")
        print(f"Sections by size (top 15 of {len(section_sizes)}):")
        for i, (section, size) in enumerate(sorted_sections[:15], 1):
            pct = (size / payload_size * 100) if payload_size > 0 else 0
            print(f"  {i:2d}. {section:15s} {size:6d} bytes ({pct:5.1f}%)")
        print("="*80)

    return payload


# ============================================================
# PAYLOAD WRAPPER (compatibilidade com sistema existente)
# ============================================================

def build_compact_payload_for_llm(
    event_data: dict,
    symbol: str = "BTCUSDT",
    window: int = 0,
    epoch_ms: int = 0,
    builders: "SummaryBuilderMap | None" = None,
) -> Dict[str, Any]:
    """
    Wrapper que adiciona metadados necessários para o pipeline LLM.
    """
    compact = build_compact_payload(event_data, builders=builders)

    wrapped = {
        "symbol": compact.pop("symbol", symbol),
        "window": window,
        "epoch_ms": epoch_ms or int(time.time() * 1000),
        "trigger": compact.pop("trigger", "AT"),
        "price": compact.pop("price", {}),
    }

    # Copiar restante com keys longas (já estão longas agora)
    for key in list(compact.keys()):
        wrapped[key] = compact.pop(key)

    # Metadados do evento original
    tipo_evento = event_data.get("tipo_evento", "")
    if tipo_evento:
        wrapped["tipo_evento"] = tipo_evento
    desc = event_data.get("descricao", "")
    if desc:
        wrapped["descricao"] = desc
    ativo = event_data.get("symbol", "")
    if ativo and ativo != wrapped.get("symbol"):
        wrapped["ativo"] = ativo

    return wrapped


# ============================================================
# DADOS OPCIONAIS AUSENTES - LOG
# ============================================================

def _log_missing_data(event_data: dict, trigger: str) -> None:
    """Loga quais dados opcionais estão ausentes."""
    missing = []
    if not event_data.get("historical_vp", {}).get("daily", {}).get("poc"):
        missing.append("VP")
    if not event_data.get("multi_tf"):
        missing.append("TF")
    if not event_data.get("institutional_analytics"):
        missing.append("IA")
    if not event_data.get("derivatives"):
        missing.append("DERIV")
    if not event_data.get("external_markets"):
        missing.append("EXT_MKT")
    if not event_data.get("ml_features", {}).get("cross_asset"):
        missing.append("CROSS")

    if missing:
        logger.info(
            f"BUILD_COMPACT: DADOS OPCIONAIS AUSENTES {missing} | "
            f"trigger={trigger}"
        )
