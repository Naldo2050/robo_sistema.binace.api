# market_orchestrator/ai/payload_compressor_v3.py
"""
Payload Compressor V3 -- Compressor Final para LLM API
=====================================================
Transforma o ai_payload verboso em formato compacto otimizado para tokens.
Economia: ~59% de tokens por chamada sem perda de qualidade analitica.

ADAPTATIVO: funciona com payload V1 (chaves originais como price_context,
flow_context) E com payload V2 (chaves ja parcialmente comprimidas).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# -- Mapeamentos de compressao --

REGIME_MAP = {
    "Alta": "UP", "Baixa": "DOWN", "Lateral": "SIDE",
    "alta": "UP", "baixa": "DOWN", "lateral": "SIDE",
    "Acumulação": "ACCUM", "Manipulação": "MANIP",
    "Distribuição": "DIST", "Expansão": "EXPAN", "Range": "RANGE",
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
    "Neutra": "NEUTRAL", "Forte Compradora": "STRONG_BUY",
    "Forte Vendedora": "STRONG_SELL", "Compradora": "BUY", "Vendedora": "SELL",
}

WHALE_CLASS_MAP = {
    "MILD_DISTRIBUTION": "MILD_DIST", "STRONG_DISTRIBUTION": "STR_DIST",
    "MILD_ACCUMULATION": "MILD_ACC", "STRONG_ACCUMULATION": "STR_ACC",
    "NEUTRAL": "NEUTRAL",
}

WHALE_BIAS_MAP = {"DISTRIBUTING": "DIST", "ACCUMULATING": "ACC", "NEUTRAL": "NEUT"}

DRIVER_MAP = {
    "MIXED_SIGNALS": "MIXED", "RISK_ON_MOMENTUM": "RISK_ON",
    "RISK_OFF_FLIGHT": "RISK_OFF", "CRYPTO_NATIVE": "CRYPTO",
}


def _r(value: Any, precision_type: str = "ratio") -> Any:
    """Arredonda valor com precisao adequada ao tipo."""
    if value is None:
        return None
    try:
        v = float(value)
    except (ValueError, TypeError):
        return value
    precisions = {
        "price": 1, "ratio": 2, "percent": 1,
        "indicator": 0, "score": 0, "volume_usd": 0,
    }
    decimals = precisions.get(precision_type, 2)
    if decimals == 0:
        return int(round(v))
    return round(v, decimals)


def _clean_vol(regime: str) -> str:
    """Limpa regime de volatilidade: LOW_VOL -> LOW."""
    if not regime:
        return ""
    return regime.replace("_VOL", "")


def compress_payload_v3(payload: dict) -> dict:
    """
    Comprime o ai_payload completo para envio ao LLM.

    ADAPTATIVO: funciona com chaves originais (price_context, flow_context,
    orderbook_context, regime_analysis, macro_context) E com chaves ja
    parcialmente comprimidas (price, flow, ob, regime, tf).

    Args:
        payload: ai_payload original do builder (V1) ou ja comprimido (V2)

    Returns:
        Payload comprimido (~450 tokens vs ~1,100 original)
    """
    if not isinstance(payload, dict):
        return payload

    c: Dict[str, Any] = {}

    # -- Metadata minimo --
    c["symbol"] = payload.get("symbol", payload.get("ativo", "BTCUSDT"))
    window = payload.get("window") or payload.get("signal_metadata", {}).get("window_id")
    if window is not None:
        c["window"] = window
    c["epoch_ms"] = payload.get("epoch_ms")
    trigger = payload.get("trigger") or payload.get("tipo_evento") or payload.get("signal_metadata", {}).get("type")
    if trigger:
        c["trigger"] = trigger

    # -- Price --
    c["price"] = _compress_price(payload)

    # -- Volume Profile --
    vp = _compress_volume_profile(payload)
    if vp:
        c["vp"] = vp

    # -- Regime (merge regime_analysis + macro_context) --
    regime = _compress_regime(payload)
    if regime and any(v for v in regime.values() if v):
        c["regime"] = regime

    # -- Orderbook --
    ob = _compress_orderbook(payload)
    if ob and any(v is not None for v in ob.values()):
        c["ob"] = ob

    # -- Flow --
    flow = _compress_flow(payload)
    if flow and any(v is not None for v in flow.values()):
        c["flow"] = flow

    # -- Whale --
    whale = _compress_whale(payload)
    if whale:
        c["whale"] = whale

    # -- Derivatives --
    deriv = _compress_derivatives(payload)
    if deriv:
        c["deriv"] = deriv

    # -- Cross Asset compacto --
    cross = _compress_cross_asset(payload)
    if cross:
        c["cross"] = cross

    # -- Timeframes compactos --
    tf = _compress_timeframes(payload)
    if tf:
        c["tf"] = tf

    # -- Quant compacto --
    quant = _compress_quant(payload)
    if quant:
        c["quant"] = quant

    # -- SR Context (ja compacto do builder) --
    sr = payload.get("sr_context")
    if isinstance(sr, dict) and sr:
        c["sr"] = sr

    # -- Anomaly alert --
    anomaly = payload.get("anomaly_alert")
    if isinstance(anomaly, dict) and anomaly:
        c["anomaly"] = anomaly

    # -- Event history (memoria) --
    hist = payload.get("event_history")
    if isinstance(hist, list) and hist:
        c["history"] = hist

    # -- Limpar None/vazios --
    c = {k: v for k, v in c.items() if v is not None and v != {} and v != []}

    # -- Validacao de dados essenciais --
    _validate_compressed(c, payload)

    return c


# -- PRICE (ADAPTATIVO: price_context OU price) --

def _compress_price(p: dict) -> dict:
    # Formato A: price_context (V1 original)
    pc = p.get("price_context", {})
    ohlc = pc.get("ohlc", {})

    # Formato B: price (V2 ja comprimido ou passthrough)
    existing = p.get("price", {})
    if isinstance(existing, (int, float)):
        # price eh um numero scalar, nao dict
        return {"c": _r(existing, "price")}

    # Buscar current price em multiplos locais
    current = (
        pc.get("current_price")
        or (existing.get("c") if isinstance(existing, dict) else None)
        or (existing.get("current") if isinstance(existing, dict) else None)
        or p.get("preco_fechamento")
        or p.get("anchor_price")
        or ohlc.get("close")
    )

    result: Dict[str, Any] = {
        "c": _r(current, "price"),
    }

    # OHLC - buscar em ohlc (V1) ou existing (V2)
    o = ohlc.get("open") or (existing.get("o") if isinstance(existing, dict) else None)
    h = ohlc.get("high") or (existing.get("h") if isinstance(existing, dict) else None)
    low = ohlc.get("low") or (existing.get("l") if isinstance(existing, dict) else None)
    vwap = ohlc.get("vwap") or (existing.get("vwap") if isinstance(existing, dict) else None)

    if o is not None:
        result["o"] = _r(o, "price")
    if h is not None:
        result["h"] = _r(h, "price")
    if low is not None:
        result["l"] = _r(low, "price")
    if vwap is not None:
        result["vwap"] = _r(vwap, "price")

    # Price action - buscar em pc (V1) ou existing (V2)
    pa = pc.get("price_action", {})
    cp = pa.get("close_position") or (existing.get("close_pos") if isinstance(existing, dict) else None)
    if cp is not None:
        result["close_pos"] = _r(cp, "ratio")

    # Profile shape/auction
    shape = pc.get("profile_shape") or (existing.get("shape") if isinstance(existing, dict) else None)
    if shape:
        result["shape"] = shape
    signal = pc.get("profile_signal") or (existing.get("signal") if isinstance(existing, dict) else None)
    if signal:
        result["signal"] = signal
    if pc.get("poor_high") or (isinstance(existing, dict) and existing.get("poor_high")):
        result["poor_high"] = True
    if pc.get("poor_low") or (isinstance(existing, dict) and existing.get("poor_low")):
        result["poor_low"] = True
    auction = pc.get("auction_bias") or (existing.get("auction") if isinstance(existing, dict) else None)
    if auction and auction != "neutral":
        result["auction"] = auction

    return result


# -- VOLUME PROFILE (ADAPTATIVO) --

def _compress_volume_profile(p: dict) -> Optional[dict]:
    pc = p.get("price_context", {})
    vpd = pc.get("volume_profile_daily", {})
    hvp = p.get("historical_vp") or p.get("volume_profile") or {}

    # Formato V2: vp ja comprimido
    existing_vp = p.get("vp", {})

    result: Dict[str, Any] = {}

    # Daily - buscar em vpd (V1), hvp (V1), ou existing_vp (V2)
    daily_data = vpd if vpd.get("poc") else hvp.get("daily", {})
    daily_v2 = existing_vp.get("d", {}) if isinstance(existing_vp, dict) else {}

    if daily_data and daily_data.get("poc"):
        result["d"] = {
            "poc": daily_data.get("poc"),
            "vah": daily_data.get("vah"),
            "val": daily_data.get("val"),
        }
        hvns = vpd.get("hvns_nearby", [])
        lvns = vpd.get("lvns_nearby", [])
        if hvns:
            result["hvn"] = hvns[:3]
        if lvns:
            result["lvn"] = lvns[:3]
        in_va = vpd.get("in_value_area") or daily_data.get("in_value_area")
        if in_va is not None:
            result["in_va"] = bool(in_va)
    elif daily_v2 and daily_v2.get("poc"):
        result["d"] = daily_v2
        if existing_vp.get("hvn"):
            result["hvn"] = existing_vp["hvn"][:3]
        if existing_vp.get("lvn"):
            result["lvn"] = existing_vp["lvn"][:3]
        if existing_vp.get("in_va") is not None:
            result["in_va"] = bool(existing_vp["in_va"])

    # Weekly
    weekly = hvp.get("weekly", {})
    weekly_v2 = existing_vp.get("w", {}) if isinstance(existing_vp, dict) else {}
    if weekly and weekly.get("poc"):
        result["w"] = {"poc": weekly["poc"], "vah": weekly.get("vah"), "val": weekly.get("val")}
    elif weekly_v2 and weekly_v2.get("poc"):
        result["w"] = weekly_v2

    # Monthly
    monthly = hvp.get("monthly", {})
    monthly_v2 = existing_vp.get("m", {}) if isinstance(existing_vp, dict) else {}
    if monthly and monthly.get("poc"):
        result["m"] = {"poc": monthly["poc"], "vah": monthly.get("vah"), "val": monthly.get("val")}
    elif monthly_v2 and monthly_v2.get("poc"):
        result["m"] = monthly_v2

    return result if result else None


# -- REGIME (ADAPTATIVO: regime_analysis + macro_context OU regime) --

def _compress_regime(p: dict) -> dict:
    # Formato A (V1): regime_analysis + macro_context
    ra = p.get("regime_analysis", {})
    mc = p.get("macro_context", {})
    mc_regime = mc.get("regime", {})

    # Formato B (V2): regime ja comprimido
    existing = p.get("regime", {})
    if isinstance(existing, str):
        existing = {}

    session = mc.get("session", existing.get("session", ""))
    phase = mc.get("phase", "")
    if phase and "_" not in str(session):
        session = f"{session}_{phase}"

    day = mc.get("day_of_week", existing.get("day", ""))
    if isinstance(day, str) and len(day) > 3:
        day = day[:3]

    signals = ra.get("signals_summary", {})
    vix = signals.get("vix") or existing.get("vix")

    result: Dict[str, Any] = {
        "market": ra.get("market_regime") or existing.get("market"),
        "vol": _clean_vol(ra.get("volatility_regime", "") or existing.get("vol", "")),
        "trend": mc_regime.get("trend", "") or existing.get("trend", ""),
        "structure": mc_regime.get("structure", "") or existing.get("structure", ""),
        "sentiment": mc_regime.get("sentiment", "") or existing.get("sentiment", ""),
        "confidence": _r(
            ra.get("regime_confidence") or existing.get("confidence"), "ratio"
        ),
        "fear_greed": _r(
            ra.get("fear_greed_proxy") or existing.get("fear_greed"), "ratio"
        ),
        "session": session,
    }
    if day:
        result["day"] = day
    if vix is not None:
        result["vix"] = vix

    return result


# -- ORDERBOOK (ADAPTATIVO: orderbook_context OU ob) --

def _compress_orderbook(p: dict) -> dict:
    # Formato A (V1)
    ob = p.get("orderbook_context", {})
    # Formato B (V2)
    if not ob:
        ob = p.get("ob", {})

    dm = ob.get("depth_metrics", {})

    bid = ob.get("bid_depth_usd") or ob.get("bid")
    ask = ob.get("ask_depth_usd") or ob.get("ask")
    imb = ob.get("imbalance") or ob.get("imb")
    top5 = dm.get("depth_imbalance") or ob.get("top5_imb")
    walls = ob.get("walls_detected") or ob.get("walls")

    result = {
        "bid": _r(bid, "volume_usd"),
        "ask": _r(ask, "volume_usd"),
        "imb": _r(imb, "ratio"),
        "top5_imb": _r(top5, "ratio"),
    }
    if walls is not None:
        result["walls"] = bool(walls)

    return result


# -- FLOW (ADAPTATIVO: flow_context OU flow) --

def _compress_flow(p: dict) -> dict:
    # Formato A (V1)
    fc = p.get("flow_context", {})
    # Formato B (V2): se flow_context esta vazio/ausente, tentar flow
    if not fc or len(fc) <= 1:
        fc_alt = p.get("flow", {})
        if isinstance(fc_alt, dict) and len(fc_alt) > len(fc):
            fc = fc_alt

    net = fc.get("net_flow") or fc.get("net")
    cvd = fc.get("cvd_accumulated") or fc.get("cvd")
    imb = fc.get("flow_imbalance") or fc.get("imb")
    agg = fc.get("aggressive_buyers") or fc.get("agg_buy")
    bsr = fc.get("buy_sell_ratio") or fc.get("bsr")
    pressure = fc.get("pressure", "")
    trend_raw = fc.get("flow_trend") or fc.get("trend") or ""
    absorption_raw = fc.get("absorption_type") or fc.get("absorption") or ""
    pa_sig = fc.get("passive_agg_signal") or fc.get("pa_signal")
    pa_conv = fc.get("passive_agg_conviction") or fc.get("pa_conv")

    result: Dict[str, Any] = {
        "net": _r(net, "volume_usd"),
        "cvd": _r(cvd, "ratio"),
        "imb": _r(imb, "ratio"),
        "agg_buy": _r(agg, "percent"),
        "pressure": pressure if pressure else None,
        "trend": FLOW_TREND_MAP.get(trend_raw, trend_raw) if trend_raw else None,
        "absorption": ABSORPTION_MAP.get(absorption_raw, absorption_raw) if absorption_raw else None,
    }
    if bsr is not None:
        result["bsr"] = _r(bsr, "ratio")
    if pa_sig:
        result["pa_signal"] = pa_sig
    if pa_conv:
        result["pa_conv"] = pa_conv

    return {k: v for k, v in result.items() if v is not None}


# -- WHALE (ADAPTATIVO: flow_context.whale_* OU whale) --

def _compress_whale(p: dict) -> Optional[dict]:
    # Formato A (V1): whale data dentro de flow_context
    fc = p.get("flow_context", {})
    # Formato B (V2): bloco separado
    wh = p.get("whale", {})

    score = fc.get("whale_score") or (wh.get("score") if isinstance(wh, dict) else None)
    wclass = fc.get("whale_class") or (wh.get("class") if isinstance(wh, dict) else None)
    bias = fc.get("whale_bias") or (wh.get("bias") if isinstance(wh, dict) else None)

    if not any([score, wclass, bias]):
        return None

    result: Dict[str, Any] = {}
    if score is not None:
        result["score"] = _r(score, "score")
    if wclass:
        result["class"] = WHALE_CLASS_MAP.get(wclass, wclass)
    if bias:
        result["bias"] = WHALE_BIAS_MAP.get(bias, bias)
    return result if result else None


# -- DERIVATIVES --

def _compress_derivatives(p: dict) -> Optional[dict]:
    derivs = p.get("derivatives", p.get("deriv", {}))
    if not derivs or not isinstance(derivs, dict):
        return None

    result: Dict[str, Any] = {}

    # Formato A (V1): derivs.BTCUSDT.open_interest
    btc = derivs.get("BTCUSDT", {})
    if btc and isinstance(btc, dict):
        oi = btc.get("open_interest")
        if oi is not None:
            result["btc_oi"] = _r(oi, "score")
        lsr = btc.get("long_short_ratio")
        if lsr is not None:
            result["btc_lsr"] = _r(lsr, "ratio")
    else:
        # Formato B (V2): derivs.btc_oi diretamente
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
    else:
        lsr = derivs.get("eth_lsr")
        if lsr is not None:
            result["eth_lsr"] = _r(lsr, "ratio")

    return result if result else None


# -- CROSS ASSET (compacto, sem strings descritivas) --

def _compress_cross_asset(p: dict) -> Optional[dict]:
    ca = p.get("cross_asset_context", p.get("cross", {}))
    if not ca or not isinstance(ca, dict) or "error" in ca:
        return None

    # Formato A (V1): nested dicts
    eth = ca.get("btc_eth_correlations", {})
    dxy = ca.get("btc_dxy_correlations", {})
    ndx = ca.get("btc_ndx_correlations", {})
    mom = ca.get("dxy_momentum", {})

    if eth or dxy or ndx or mom:
        result = {
            "eth_7d": _r(eth.get("short_term_7d"), "ratio"),
            "eth_30d": _r(eth.get("long_term_30d"), "ratio"),
            "dxy_30d": _r(dxy.get("medium_term_30d"), "ratio"),
            "dxy_90d": _r(dxy.get("long_term_90d"), "ratio"),
            "ndx_30d": _r(ndx.get("medium_term_30d"), "ratio"),
            "dxy_ret5d": _r(mom.get("return_5d"), "ratio"),
            "dxy_ret20d": _r(mom.get("return_20d"), "ratio"),
        }
    else:
        # Formato B (V2): chaves flat ja comprimidas
        result = {
            "eth_7d": _r(ca.get("eth_7d"), "ratio"),
            "eth_30d": _r(ca.get("eth_30d"), "ratio"),
            "dxy_30d": _r(ca.get("dxy_30d"), "ratio"),
            "dxy_90d": _r(ca.get("dxy_90d"), "ratio"),
            "ndx_30d": _r(ca.get("ndx_30d"), "ratio"),
            "dxy_ret5d": _r(ca.get("dxy_ret5d"), "ratio"),
            "dxy_ret20d": _r(ca.get("dxy_ret20d"), "ratio"),
        }

    return {k: v for k, v in result.items() if v is not None} or None


# -- TIMEFRAMES (ADAPTATIVO: multi_tf OU tf) --

def _compress_timeframes(p: dict) -> Optional[dict]:
    # Formato A (V1): multi_tf
    mtf = p.get("multi_tf")
    # Formato B (V2): tf (alias ja renomeado)
    if not mtf:
        mtf = p.get("tf")
    # Formato C: dentro de macro_context
    if not mtf:
        mtf = p.get("macro_context", {}).get("multi_timeframe_trends", {})

    if not isinstance(mtf, dict) or not mtf:
        return None

    result: Dict[str, Any] = {}
    for key, val in mtf.items():
        if not isinstance(val, dict):
            continue
        tf_data: Dict[str, Any] = {}

        # Tendencia compacta - buscar em multiplos nomes
        tend = val.get("tendencia", val.get("trend", val.get("t", "")))
        tf_data["t"] = REGIME_MAP.get(tend, tend)

        # EMA
        ema = val.get("mme_21", val.get("ema_21", val.get("mme21", val.get("ema"))))
        if ema is not None:
            tf_data["ema"] = _r(ema, "price")

        # RSI
        rsi = val.get("rsi_short", val.get("rsi"))
        if rsi is not None:
            tf_data["rsi"] = _r(rsi, "percent")

        # MACD como array [line, signal]
        macd = val.get("macd")
        macd_s = val.get("macd_signal", val.get("macd_s"))
        if isinstance(macd, list):
            # Ja eh array
            tf_data["macd"] = [_r(macd[0], "indicator"),
                               _r(macd[1] if len(macd) > 1 else 0, "indicator")]
        elif macd is not None:
            tf_data["macd"] = [_r(macd, "indicator"), _r(macd_s, "indicator")]

        # ADX
        adx = val.get("adx")
        if adx is not None:
            tf_data["adx"] = _r(adx, "percent")

        # ATR
        atr = val.get("atr")
        if atr is not None:
            tf_data["atr"] = _r(atr, "indicator")

        # Regime compacto
        regime = val.get("regime", val.get("reg", ""))
        tf_data["reg"] = REGIME_MAP.get(regime, regime)

        result[key] = tf_data

    return result if result else None


# -- QUANT (ADAPTATIVO: quant_model OU quant) --

def _compress_quant(p: dict) -> Optional[dict]:
    qm = p.get("quant_model", p.get("quant", {}))
    if not qm or not isinstance(qm, dict):
        return None

    # Formato A (V1): model_probability_up, confidence_score
    prob = qm.get("model_probability_up") or qm.get("prob_up")
    conf = qm.get("confidence_score") or qm.get("confidence")
    if prob is None and conf is None:
        return None

    result: Dict[str, Any] = {}
    if prob is not None:
        result["prob_up"] = _r(prob, "ratio")
    if conf is not None:
        result["confidence"] = _r(conf, "ratio")
    return result


# -- VALIDACAO --

def _validate_compressed(compressed: dict, original: dict) -> None:
    """
    Valida que dados essenciais sobreviveram a compressao.
    Se faltarem, tenta recuperar do payload original e loga warning.
    """
    warnings = []

    # Preco eh obrigatorio
    price = compressed.get("price", {})
    if not price.get("c"):
        warnings.append("price.c MISSING")
        # Tentar recuperar de qualquer lugar
        for key in ("preco_fechamento", "anchor_price", "current_price"):
            if key in original and original[key]:
                compressed.setdefault("price", {})["c"] = _r(original[key], "price")
                warnings[-1] += " (RECOVERED)"
                break

    # TF
    if "tf" not in compressed:
        warnings.append("tf MISSING")

    # Regime
    regime = compressed.get("regime", {})
    if not regime or not regime.get("market"):
        warnings.append("regime.market MISSING")

    # Orderbook
    ob = compressed.get("ob", {})
    if not ob or (ob.get("bid") is None and ob.get("ask") is None):
        warnings.append("ob MISSING")

    # Flow
    flow = compressed.get("flow", {})
    if not flow or (flow.get("net") is None and flow.get("cvd") is None):
        warnings.append("flow.net/cvd MISSING")

    if warnings:
        logger.warning(
            "COMPRESS_V3_VALIDATION: Dados essenciais faltando apos compressao! "
            "Missing: %s | Original keys: %s",
            warnings,
            list(original.keys()),
        )
