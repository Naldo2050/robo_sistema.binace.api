# market_orchestrator/ai/ai_payload_builder.py
# -*- coding: utf-8 -*-
"""
Construtor de Payload para Análise de IA.

Este módulo é responsável por padronizar e organizar os dados brutos e métricas
do sistema em um formato estruturado e semântico para consumo pelos modelos de IA.
"""

from typing import Dict, Any, Optional
from datetime import datetime

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
    ml_prediction: Optional[Dict[str, Any]] = None
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

    # Garante que ml_features sempre seja um dicionário
    if not isinstance(ml_features, dict):
        ml_features = {}
    else:
        ml_features = ml_features or {}

    # 1. Contexto de Preço (Price Context)
    ohlc = enriched.get("ohlc", {})
    vp_daily = historical_profile.get("daily", {})
    
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
        "volume_profile_daily": {
            "poc": vp_daily.get("poc"),
            "vah": vp_daily.get("vah"),
            "val": vp_daily.get("val"),
            "in_value_area": _check_in_range(ohlc.get("close"), vp_daily.get("val"), vp_daily.get("vah"))
        },
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
        "net_flow": order_flow.get("net_flow_1m"),  # Delta da janela
        "cvd_accumulated": flow_metrics.get("cvd"),
        "flow_imbalance": order_flow.get("flow_imbalance"),
        "aggressive_buyers": order_flow.get("aggressive_buy_pct"),
        "aggressive_sellers": order_flow.get("aggressive_sell_pct"),
        "whale_activity": whale_flow,
        "liquidity_clusters_count": len(flow_metrics.get("liquidity_heatmap", {}).get("clusters", [])),
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

    # 4. Contexto Macro e Regime
    macro_full_context = {
        "session": macro_context.get("trading_session"),
        "phase": macro_context.get("session_phase"),
        "multi_timeframe_trends": macro_context.get("mtf_trends", {}),
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

    # 5. Metadados do Sinal
    signal_metadata = {
        "type": signal.get("tipo_evento"),
        "battle_result": signal.get("resultado_da_batalha"),
        "severity": signal.get("severity", "INFO"),
        "window_id": signal.get("janela_numero"),
        "timestamp_utc": signal.get("timestamp_utc"),
        "description": signal.get("descricao")
    }

    # 6. Estrutura Final
    ai_payload = {
        "symbol": symbol,
        "timestamp": signal.get("timestamp"),
        "signal_metadata": signal_metadata,
        "price_context": price_context,
        "flow_context": flow_context,
        "orderbook_context": ob_context,
        "macro_context": macro_full_context,
        "ml_features": ml_features, # Repassa features brutas para análise quantitativa da IA
        "historical_stats": signal.get("historical_confidence", {})
    }

    # === CAMPOS DE COMPATIBILIDADE (Legacy Support) ===
    # Mantém chaves na raiz para não quebrar templates Jinja/f-strings existentes no ai_analyzer_qwen.py
    # que esperam acessar payload['delta'], payload['orderbook_data'], etc.
    ai_payload["tipo_evento"] = signal.get("tipo_evento")
    ai_payload["ativo"] = symbol
    ai_payload["descricao"] = signal.get("descricao")
    ai_payload["delta"] = signal.get("delta")
    ai_payload["volume_total"] = signal.get("volume_total")
    ai_payload["preco_fechamento"] = signal.get("preco_fechamento")
    ai_payload["orderbook_data"] = orderbook_data
    ai_payload["fluxo_continuo"] = flow_metrics
    ai_payload["historical_vp"] = historical_profile
    ai_payload["multi_tf"] = macro_context.get("mtf_trends", {})
    ai_payload["event_history"] = signal.get("event_history", []) # Se houver memória injetada

    # === SEÇÃO DE INTELIGÊNCIA QUANTITATIVA ===
    # Se houver previsão ML, adiciona ao contexto
    quant_context = {}

    # Usa a previsão passada como parâmetro, com fallback para o próprio sinal (compatibilidade)
    ml_prediction = ml_prediction or signal.get("ml_prediction") or {}

    if ml_prediction and ml_prediction.get("status") == "ok":
        prob = ml_prediction.get("prob_up", 0.5)
        confidence = ml_prediction.get("confidence", 0.0)

        # Traduz probabilidade para texto
        if prob > 0.75:
            sentiment = "BULLISH FORTE (Alta Probabilidade)"
            action_bias = "compra"
        elif prob > 0.60:
            sentiment = "BULLISH (Moderado)"
            action_bias = "compra"
        elif prob < 0.25:
            sentiment = "BEARISH FORTE (Alta Probabilidade)"
            action_bias = "venda"
        elif prob < 0.40:
            sentiment = "BEARISH (Moderado)"
            action_bias = "venda"
        else:
            sentiment = "NEUTRO / INDEFINIDO"
            action_bias = "aguardar"

        quant_context = {
            "model_probability_up": float(prob),
            "model_probability_down": 1.0 - float(prob),
            "model_sentiment": sentiment,
            "action_bias": action_bias,
            "confidence_score": float(confidence),
            "features_used": ml_prediction.get("features_used", 0),
            "total_features": ml_prediction.get("total_features", 0),
        }

    # Adiciona ao payload principal
    ai_payload["quant_model"] = quant_context

    # String formatada para templates legacy
    if quant_context:
        prob_pct = quant_context['model_probability_up'] * 100
        confidence_pct = quant_context['confidence_score'] * 100

        # Adiciona ao contexto de ML (para compatibilidade)
        ai_payload["ml_str"] = (
            f"\n🤖 **INTELIGÊNCIA QUANTITATIVA (XGBoost)**\n"
            f"   📈 Probabilidade de Alta: {prob_pct:.1f}%\n"
            f"   📉 Probabilidade de Baixa: {(100-prob_pct):.1f}%\n"
            f"   🎯 Viés Matemático: {quant_context['model_sentiment']}\n"
            f"   📊 Confiança do Modelo: {confidence_pct:.1f}%\n"
            f"   🔍 Features: {quant_context['features_used']}/{quant_context['total_features']}\n"
        )
    else:
        # Fallback para compatibilidade
        if "ml_str" not in ai_payload:
            ai_payload["ml_str"] = ""

    return ai_payload

def _check_in_range(price, low, high):
    """Helper simples para verificar se preço está em range."""
    if price is None or low is None or high is None:
        return None
    return low <= price <= high