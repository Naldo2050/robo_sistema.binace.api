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

import json
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

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
        if isinstance(pl, dict):
            vol_ratio = pl.get("volume_ratio", 1.0)
            if vol_ratio and vol_ratio < 0.5:
                price["pl"] = 1

    # FIX 4a: breakout_risk do Volume Profile compression
    va_pct = pa.get("va_volume_pct", {})
    if isinstance(va_pct, dict):
        brk_risk = va_pct.get("breakout_risk")
        compression = va_pct.get("compression_signal", 0)
        if brk_risk and brk_risk in ("HIGH", "VERY_HIGH"):
            price["brk_risk"] = brk_risk[:4]
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

    if dominant_tf:
        result["dom"] = dominant_tf

    # Flag conflict explicitly when strong disagreement exists
    if votes_bull > 0 and votes_bear > 0:
        result["bull%"] = bull_pct
        result["bear%"] = bear_pct

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

    imb_val = round(ob_data.get("imbalance", 0), 2)
    ob: dict[str, Any] = {
        "b": compact_number(bid, force_sign=False),
        "a": compact_number(ask, force_sign=False),
        "imb": imb_val,
        "bias": "BUY" if imb_val > 0.1 else "SELL" if imb_val < -0.1 else "NEUT",
    }

    t5_imb = depth.get("L5", {}).get("imbalance")
    if t5_imb is not None:
        ob["t5"] = round(t5_imb, 2)

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
    if not ext_ind:
        return {}

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

    # Stochastic %K
    stochastic = ext_ind.get("stochastic", {})
    stoch_k = stochastic.get("k") if isinstance(stochastic, dict) else None
    if stoch_k is not None:
        ext["stoch"] = round(stoch_k)

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
        ext["garch"] = round(float(garch), 4)

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

    # sell_defense = zonas de resistência (acima do preço)
    for i, zone in enumerate(dz.get("sell_defense", [])[:2]):
        price_r = zone.get("center")
        if price_r is not None:
            key = f"r{i+1}"
            sr[key] = [round(float(price_r)), round(zone.get("strength", 0), 1)]
            sources = zone.get("source_count", zone.get("sources", 0))
            if sources and sources >= 3:
                sr[f"{key}_conf"] = sources

    # buy_defense = zonas de suporte (abaixo do preço)
    for i, zone in enumerate(dz.get("buy_defense", [])[:2]):
        price_s = zone.get("center")
        if price_s is not None:
            key = f"s{i+1}"
            sr[key] = [round(float(price_s)), round(zone.get("strength", 0), 1)]
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

    # --- Volume Profile Diário ---
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

    # --- Correlações ---
    eth7 = ml_cross.get("btc_eth_corr_7d")
    if eth7:
        ctx["eth7"] = round(eth7, 1)

    dxy30 = ml_cross.get("btc_dxy_corr_30d")
    if dxy30:
        ctx["dxy30"] = round(dxy30, 2)

    return ctx


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def build_compact_payload(event_data: dict) -> dict:
    """
    Constrói payload compactado para envio à LLM.

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
        "trigger": trigger,                              # ← ERA "t"
        "price": price,                                  # ← ERA "p"
    }

    if regime:
        payload["regime"] = regime                       # ← ERA "r"

    if flow:
        payload["flow"] = flow                           # ← ERA "f"

    if ob:
        payload["ob"] = ob

    if whale:
        payload["w"] = whale

    if quant and quant.get("pu", 0.5) != 0.5:
        payload["quant"] = quant                         # ← ERA "q"

    if tf_section:
        payload["tf"] = tf_section

    if ext_indicators:
        payload["ext"] = ext_indicators

    if alerts:
        payload["alerts"] = alerts

    if sr:
        payload["sr"] = sr

    # --- CONTEXTO ESTÁTICO ---
    force_ctx = (
        trigger_raw in IMPORTANT_EVENTS
        or trigger in IMPORTANT_EVENTS
    )

    if force_ctx or _should_send_static(static_ctx):
        payload["ctx"] = static_ctx
        ctx_status = "SENT" + (" (forced)" if force_ctx else "")
    else:
        # P3: ctx CACHED — enviar versão mínima para IA não ficar "cega"
        if _last_static_ctx:
            payload["ctx"] = {
                "cached": True,
                "ses": _last_static_ctx.get("ses"),
                "fg": _last_static_ctx.get("fg"),
                "vix": _last_static_ctx.get("vix"),
                "poc": _last_static_ctx.get("poc"),
                "val": _last_static_ctx.get("val"),
                "vah": _last_static_ctx.get("vah"),
            }
            # Remover keys None para compactar
            payload["ctx"] = {k: v for k, v in payload["ctx"].items() if v is not None}
        else:
            payload["ctx"] = {"cached": True}
        ctx_status = "CACHED(mini)"

    # --- LOG ---
    sections = len(payload)
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    payload_size = len(payload_json)
    estimated_tokens = payload_size // 4
    tf_keys = list(tf_section.keys()) if tf_section else "NONE"

    logger.info(
        f"BUILD_COMPACT: OK | price={price.get('c')} | "
        f"sections={sections} | size={payload_size} chars | "
        f"~{estimated_tokens} tokens | tf={tf_keys} | ctx={ctx_status}"
    )

    logger.debug(f"COMPACT_PAYLOAD_DEBUG: {payload_json}")

    # ═══════════════════════════════════════════════════════════
    # NÃO injetar em event_data aqui.                ← CORRIGIDO
    # O pipeline (ai_analyzer_qwen.py linha 3499)
    # detecta "price" no event_data FLAT e faz o
    # empacotamento como ai_payload automaticamente.
    # Basta que o CALLER coloque o payload flat no
    # event_data, o que já acontece no fluxo normal.
    # ═══════════════════════════════════════════════════════════

    return payload


# ============================================================
# PAYLOAD WRAPPER (compatibilidade com sistema existente)
# ============================================================

def build_compact_payload_for_llm(
    event_data: dict,
    symbol: str = "BTCUSDT",
    window: int = 0,
    epoch_ms: int = 0,
) -> dict:
    """
    Wrapper que adiciona metadados necessários para o pipeline LLM.
    """
    compact = build_compact_payload(event_data)

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