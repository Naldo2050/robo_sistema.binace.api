"""
build_compact_payload.py
========================
Constroi o ai_payload compacto DIRETAMENTE do event_data.
NAO depende de nenhum builder ou compressor existente.
NAO importa o que outros modulos fazem.

Uso no ai_runner.py:
    from build_compact_payload import build_compact_payload

    # Substituir TUDO que constroi/comprime o payload por:
    ai_payload = build_compact_payload(event_data)
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _r(val: Any, dec: int = 2) -> Any:
    if val is None:
        return None
    if not isinstance(val, (int, float)):
        return val
    return int(round(val)) if dec == 0 else round(val, dec)


_REG = {
    "Alta": "UP", "Baixa": "DOWN", "Lateral": "SIDE",
    "Acumulacao": "ACCUM", "Manipulacao": "MANIP",
    "Distribuicao": "DIST", "Expansao": "EXPAN", "Range": "RANGE",
}

_FT = {
    "accelerating_selling": "accel_sell",
    "accelerating_buying": "accel_buy",
    "short_term_reversal_to_sell": "reversal_sell",
    "short_term_reversal_to_buy": "reversal_buy",
}

_ABS = {
    "Neutra": "NEUTRAL", "Forte Compradora": "STRONG_BUY",
    "Forte Vendedora": "STRONG_SELL",
    "Absorcao de Compra": "BUY_ABS", "Absorcao de Venda": "SELL_ABS",
}


def build_compact_payload(event_data: dict) -> dict:
    """
    Constroi payload compacto para LLM direto do event_data
    (o dicionario ANALYSIS_TRIGGER completo).

    Busca dados em TODOS os locais possiveis do evento.
    Nao depende de builder nem de compressor externo.
    """
    # ======================================
    # FONTES DE DADOS (buscar em multiplos locais)
    # ======================================
    raw = event_data.get("raw_event", {})
    cs = event_data.get("contextual_snapshot", {})
    es = event_data.get("enriched_snapshot", {})
    fc = event_data.get("fluxo_continuo", {})
    ob = event_data.get("orderbook_data", {})
    me = event_data.get("market_environment", {})
    mc = event_data.get("market_context", {})
    ia = event_data.get("institutional_analytics", {})
    derivs = event_data.get("derivatives", {})
    ml = event_data.get("ml_features", {})

    # OHLC: buscar em contextual_snapshot ou enriched_snapshot
    ohlc = cs.get("ohlc", es.get("ohlc", {}))

    # Multi TF: buscar em raw_event
    multi_tf = raw.get("multi_tf", event_data.get("multi_tf", {}))

    # Historical VP: buscar em raw_event ou evento
    hvp = raw.get("historical_vp", event_data.get("historical_vp", {}))

    # Flow order_flow
    of = fc.get("order_flow", {})
    bsr = of.get("buy_sell_ratio", {})

    # Institutional
    pa = ia.get("profile_analysis", {})
    fa = ia.get("flow_analysis", {})
    pag = fa.get("passive_aggressive", {})
    whale = fa.get("whale_accumulation", {})
    poor = pa.get("poor_extremes", {})
    shape_info = pa.get("profile_shape", {})

    # Cross asset do ml_features
    ca = ml.get("cross_asset", {}) if isinstance(ml, dict) else {}

    # ======================================
    # PRECO
    # ======================================
    current_price = (
        event_data.get("preco_fechamento")
        or raw.get("preco_fechamento")
        or ohlc.get("close")
        or raw.get("advanced_analysis", {}).get("price")
    )

    price = {"c": _r(current_price, 1)}
    if ohlc.get("open"):
        price["o"] = _r(ohlc["open"], 1)
    if ohlc.get("high"):
        price["h"] = _r(ohlc["high"], 1)
    if ohlc.get("low"):
        price["l"] = _r(ohlc["low"], 1)
    if ohlc.get("vwap"):
        price["vwap"] = _r(ohlc["vwap"], 1)

    # Profile shape
    shape = shape_info.get("shape")
    if shape:
        price["shape"] = shape
        signal = shape_info.get("trading_signal")
        if signal:
            price["signal"] = signal

    # Poor extremes
    if poor.get("poor_high", {}).get("detected"):
        price["poor_high"] = True
    if poor.get("poor_low", {}).get("detected"):
        price["poor_low"] = True
    auction = poor.get("action_bias")
    if auction and auction != "neutral":
        price["auction"] = auction

    # ======================================
    # REGIME (market_environment + market_context)
    # ======================================
    session = mc.get("trading_session", "")
    phase = mc.get("session_phase", "")
    session_str = f"{session}_{phase}" if phase else session

    dow = mc.get("day_of_week", "")
    day_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    day_str = day_names.get(dow, str(dow)) if isinstance(dow, int) else str(dow)[:3]

    regime = {
        "vol": me.get("volatility_regime", ""),
        "trend": me.get("trend_direction", ""),
        "structure": me.get("market_structure", ""),
        "sentiment": me.get("risk_sentiment", ""),
        "session": session_str,
        "day": day_str,
    }
    regime = {k: v for k, v in regime.items() if v}

    # ======================================
    # VOLUME PROFILE
    # ======================================
    vp = {}
    for tf_key in ("daily", "weekly", "monthly"):
        src = hvp.get(tf_key, {})
        if src and src.get("poc"):
            vp[tf_key] = {
                "poc": src["poc"],
                "vah": src.get("vah"),
                "val": src.get("val"),
            }

    # ======================================
    # ORDERBOOK
    # ======================================
    dm = ob.get("depth_metrics", {})
    ob_out = {}
    if ob.get("bid_depth_usd") is not None:
        ob_out = {
            "bid": _r(ob["bid_depth_usd"], 0),
            "ask": _r(ob.get("ask_depth_usd"), 0),
            "imb": _r(ob.get("imbalance"), 2),
            "top5_imb": _r(dm.get("depth_imbalance"), 2),
        }

    # ======================================
    # FLOW
    # ======================================
    flow_trend = bsr.get("flow_trend", "")
    absorption = fc.get("tipo_absorcao", "")
    composite = pag.get("composite", {})

    flow = {
        "net_1m": _r(of.get("net_flow_1m"), 0),
        "net_5m": _r(of.get("net_flow_5m"), 0),
        "net_15m": _r(of.get("net_flow_15m"), 0),
        "cvd": _r(fc.get("cvd"), 2),
        "imb": _r(of.get("flow_imbalance"), 2),
        "agg_buy": _r(of.get("aggressive_buy_pct"), 1),
        "bsr": _r(bsr.get("buy_sell_ratio"), 2),
        "pressure": bsr.get("pressure"),
        "trend": _FT.get(flow_trend, flow_trend),
        "absorption": _ABS.get(absorption, absorption.upper() if absorption else None),
        "pa_signal": composite.get("signal"),
        "pa_conv": composite.get("conviction"),
    }
    flow = {k: v for k, v in flow.items() if v is not None}

    # ======================================
    # WHALE
    # ======================================
    whale_out = {}
    if whale.get("score") is not None:
        whale_out = {
            "score": _r(whale["score"], 0),
            "class": whale.get("classification", ""),
            "bias": whale.get("bias", ""),
        }

    # ======================================
    # DERIVATIVES
    # ======================================
    deriv_out = {}
    btc_d = derivs.get("BTCUSDT", {})
    eth_d = derivs.get("ETHUSDT", {})
    if btc_d.get("long_short_ratio") is not None:
        deriv_out = {
            "btc_oi": _r(btc_d.get("open_interest"), 0),
            "btc_lsr": _r(btc_d["long_short_ratio"], 2),
            "eth_lsr": _r(eth_d.get("long_short_ratio"), 2),
        }

    # ======================================
    # CROSS ASSET (do ml_features)
    # ======================================
    cross = {}
    if ca:
        cross = {
            "eth_7d": _r(ca.get("btc_eth_corr_7d"), 2),
            "eth_30d": _r(ca.get("btc_eth_corr_30d"), 2),
            "dxy_30d": _r(ca.get("btc_dxy_corr_30d"), 2),
            "dxy_90d": _r(ca.get("btc_dxy_corr_90d"), 2),
            "ndx_30d": _r(ca.get("btc_ndx_corr_30d"), 2),
            "dxy_r5d": _r(ca.get("dxy_return_5d"), 2),
            "dxy_r20d": _r(ca.get("dxy_return_20d"), 2),
        }
        cross = {k: v for k, v in cross.items() if v is not None}

    # ======================================
    # TIMEFRAMES
    # ======================================
    tf_out = {}
    for tf_key, tf_val in multi_tf.items():
        if not isinstance(tf_val, dict):
            continue
        trend = tf_val.get("tendencia", tf_val.get("trend", ""))
        regime_str = tf_val.get("regime", "")
        macd_v = tf_val.get("macd", tf_val.get("macd_signal", 0))
        macd_s = tf_val.get("macd_signal", tf_val.get("macd_s", 0))

        tf_out[tf_key] = {
            "t": _REG.get(trend, trend[:2].upper() if trend else "?"),
            "ema": _r(tf_val.get("mme_21", tf_val.get("mme21")), 0),
            "rsi": _r(tf_val.get("rsi_short", tf_val.get("rsi")), 1),
            "macd": [_r(macd_v, 0), _r(macd_s, 0)],
            "adx": _r(tf_val.get("adx"), 1),
            "atr": _r(tf_val.get("atr"), 0),
            "reg": _REG.get(regime_str, regime_str[:4].upper() if regime_str else "?"),
        }

    # ======================================
    # MONTAR RESULTADO FINAL
    # ======================================
    result = {
        "symbol": event_data.get("symbol", "BTCUSDT"),
        "window": event_data.get("janela_numero"),
        "epoch_ms": event_data.get("epoch_ms"),
        "trigger": event_data.get("tipo_evento", "UNKNOWN"),
        "price": price,
    }

    if regime:
        result["regime"] = regime
    if vp:
        result["vp"] = vp
    if ob_out:
        result["ob"] = ob_out
    if flow:
        result["flow"] = flow
    if whale_out:
        result["whale"] = whale_out
    if deriv_out:
        result["deriv"] = deriv_out
    if cross:
        result["cross"] = cross
    if tf_out:
        result["tf"] = tf_out

    # Quant model (do ml_features se disponivel)
    result["quant"] = {
        "prob_up": 0.5,  # placeholder - sera preenchido pelo ai_runner
        "conf": 0.0,
    }

    # ======================================
    # VALIDACAO
    # ======================================
    missing = []
    if not result.get("price", {}).get("c"):
        missing.append("PRICE")
    if not result.get("tf"):
        missing.append("TF")
    if not result.get("vp"):
        missing.append("VP")
    if not result.get("ob"):
        missing.append("OB")
    if not result.get("flow"):
        missing.append("FLOW")

    if missing:
        logger.error(
            "BUILD_COMPACT: DADOS FALTANDO %s | event_keys=%s | raw_keys=%s",
            missing, list(event_data.keys())[:10], list(raw.keys())[:10]
        )
    else:
        import json
        size = len(json.dumps(result, separators=(",", ":")))
        logger.info(
            "BUILD_COMPACT: OK | price=%s | sections=%d | size=%d chars | ~%d tokens",
            result["price"].get("c"), len(result), size, size // 4
        )

    return result
