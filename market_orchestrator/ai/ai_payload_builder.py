# market_orchestrator/ai/ai_payload_builder.py
# -*- coding: utf-8 -*-
"""
Construtor de Payload para Análise de IA.

Este módulo é responsável por padronizar e organizar os dados brutos e métricas
do sistema em um formato estruturado e semântico para consumo pelos modelos de IA.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import asyncio
import sys
from pathlib import Path

# Adiciona o diretório raiz do projeto ao PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent.parent))

from market_orchestrator.ai.ai_enrichment_context import build_enriched_ai_context

# Import para correlações cross-asset
try:
    from cross_asset_correlations import get_cross_asset_features
except ImportError as e:
    get_cross_asset_features = None

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
    Adiciona contextos enriquecidos ao ai_payload.
    
    Esta função mescla contextos adicionais do raw_event no ai_payload
    para garantir que todas as informações relevantes estejam disponíveis
    para a análise da IA.
    
    Args:
        ai_payload (Dict[str, Any]): O payload principal da IA
        raw_event (Dict[str, Any]): O evento bruto com dados originais
        
    Returns:
        Dict[str, Any]: O ai_payload atualizado com contextos enriquecidos
    """
    # Verifica se raw_event contém dados válidos
    if not raw_event or not isinstance(raw_event, dict):
        return ai_payload
    
    # Extrai contextos enriquecidos do raw_event
    enriched_contexts = {}
    
    # Adiciona contexto raw_event se presente
    if "raw_event" in raw_event:
        enriched_contexts["raw_event_context"] = raw_event["raw_event"]
    
    # Adiciona contexto avançado se presente
    if "advanced_analysis" in raw_event:
        enriched_contexts["advanced_analysis_context"] = raw_event["advanced_analysis"]
    
    # Adiciona outros contextos específicos do evento
    for key in ["features_window_id", "enriched_snapshot", "contextual_snapshot"]:
        if key in raw_event:
            enriched_contexts[key] = raw_event[key]
    
    # Mescla os contextos no ai_payload
    if enriched_contexts:
        ai_payload["enriched_contexts"] = enriched_contexts
    
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
    pivots: Optional[Dict[str, Any]] = None
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

    # 🆕 3.5. Indicadores Técnicos (Technical Indicators)
    # Extrai indicadores chave do contexto MTF (prioriza 1h ou 4h para trend, 15m para tático)
    mtf = macro_context.get("mtf_trends", {})
    # Tenta pegar contexto de 1h para indicadores 'padrão', fallback para 15m
    tf_tech = mtf.get("1h", {}) if "1h" in mtf else mtf.get("15m", {})
    
    technical_indicators = {
        "rsi": tf_tech.get("rsi_short"), # 14 period default
        "macd": {
            "line": tf_tech.get("macd"),
            "signal": tf_tech.get("macd_signal"),
            "histogram": round((tf_tech.get("macd", 0) or 0) - (tf_tech.get("macd_signal", 0) or 0), 4)
        },
        "adx": tf_tech.get("adx"),
        "stoch": tf_tech.get("stoch_k"), # Se existir no MTF
        "pivots": pivots or {}
    }

    # 4. Contexto Cross-Asset (🆕)
    # Extrai correlações das ml_features se disponíveis
    cross_asset_data = ml_features.get("cross_asset", {})
    
    # Calcula correlações em tempo real para o payload da IA
    cross_asset_context = {}
    if symbol == "BTCUSDT" and get_cross_asset_features is not None:
        try:
            # Calcula correlações em tempo real para a IA
            correlations = get_cross_asset_features(datetime.utcnow())
            
            if correlations.get("status") == "ok":
                cross_asset_context = {
                    "btc_eth_correlations": {
                        "short_term_7d": correlations.get("btc_eth_corr_7d"),
                        "long_term_30d": correlations.get("btc_eth_corr_30d"),
                        "relationship": "cripto_major_pair",
                        "timeframe": "1h data"
                    },
                    "btc_dxy_correlations": {  # 🆕 Foco especial (inversa esperada)
                        "medium_term_30d": correlations.get("btc_dxy_corr_30d"),
                        "long_term_90d": correlations.get("btc_dxy_corr_90d"),
                        "relationship": "inverse_usd_strength",
                        "interpretation": "BTC tende a se mover inversamente ao DXY",
                        "timeframe": "daily data"
                    },
                    "btc_ndx_correlations": {
                        "medium_term_30d": correlations.get("btc_ndx_corr_30d"),
                        "relationship": "tech_risk_correlation",
                        "timeframe": "daily data"
                    },
                    "dxy_momentum": {
                        "return_5d": correlations.get("dxy_return_5d"),
                        "return_20d": correlations.get("dxy_return_20d"),
                        "momentum": correlations.get("dxy_momentum")
                    },
                    "cross_asset_sentiment": {
                        "dxy_inverse_strength": correlations.get("btc_dxy_inverse_strength", 0),
                        "correlation_stability": correlations.get("btc_dxy_correlation_stability", 0),
                        "crypto_leadership": correlations.get("btc_eth_corr_30d", 0)
                    }
                }
        except Exception as e:
            cross_asset_context = {
                "error": f"Falha ao calcular correlações: {str(e)}",
                "fallback_data": cross_asset_data
            }
    else:
        # Usa dados das ml_features como fallback
        cross_asset_context = {
            "btc_eth_correlations": {
                "short_term_7d": cross_asset_data.get("btc_eth_corr_7d"),
                "long_term_30d": cross_asset_data.get("btc_eth_corr_30d"),
                "relationship": "cripto_major_pair"
            },
            "btc_dxy_correlations": {
                "medium_term_30d": cross_asset_data.get("btc_dxy_corr_30d"),
                "long_term_90d": cross_asset_data.get("btc_dxy_corr_90d"),
                "relationship": "inverse_usd_strength"
            },
            "btc_ndx_correlations": {
                "medium_term_30d": cross_asset_data.get("btc_ndx_corr_30d"),
                "relationship": "tech_risk_correlation"
            },
            "dxy_momentum": {
                "return_5d": cross_asset_data.get("dxy_return_5d"),
                "return_20d": cross_asset_data.get("dxy_return_20d")
            }
        }

    # 5. Contexto Macro e Regime
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

    # 6. Análise de Regime com EnhancedRegimeDetector
    regime_analysis = {}
    if MacroDataProvider and EnhancedRegimeDetector:
        try:
            # Inicializa o detector de regime (stateful)
            regime_detector = EnhancedRegimeDetector()
            
            # Cria dados macro mock para evitar chamadas assíncronas
            macro_data = {
                "vix": 12.5,
                "treasury_10y": 4.5,
                "treasury_2y": 3.8,
                "dxy": 105.2,
                "gold": 1950.0,
                "oil": 85.0,
                "btc_dominance": 45.0,
                "eth_dominance": 18.0,
                "usdt_dominance": 5.0,
                "yield_spread": 0.7,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Dados de exemplo para cross_asset_features e current_price_data
            cross_asset_features = {
                "correlation_spy": market_environment.get("correlation_spy"),
                "btc_dxy_corr_30d": market_environment.get("correlation_dxy"),
                "dxy_momentum": 0.3
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
            print(f"Erro ao analisar regime: {e}")
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

    # 7. Estrutura Final
    ai_payload = {
        "symbol": symbol,
        "timestamp": signal.get("timestamp"),
        "signal_metadata": signal_metadata,
        "price_context": price_context,
        "flow_context": flow_context,
        "orderbook_context": ob_context,
        "technical_indicators": technical_indicators,
        "cross_asset_context": cross_asset_context,  # 🆕 Contexto cross-asset
        "macro_context": macro_full_context,
        "ml_features": ml_features, # Repassa features brutas para análise quantitativa da IA
        "historical_stats": signal.get("historical_confidence", {})
    }

    # Adiciona a análise de regime ao payload
    if regime_analysis:
        ai_payload["regime_analysis"] = regime_analysis

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

    # 🔧 ENRICHMENT: Adiciona contextos de targets/opções/on-chain/risco
    enriched_ctx = build_enriched_ai_context(signal)
    ai_payload.update(enriched_ctx)
    
    # NOVO: mesclar contextos enriquecidos
    ai_payload = add_enriched_context_to_ai_payload(ai_payload, signal)

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

    # Calcula contexto cross-asset
    cross_asset_context = {}
    if symbol == "BTCUSDT" and get_cross_asset_features is not None:
        try:
            # Calcula correlações em tempo real
            correlations = get_cross_asset_features(datetime.utcnow())

            if correlations.get("status") == "ok":
                cross_asset_context = {
                    "features": {
                        "btc_eth_corr_7d": correlations.get("btc_eth_corr_7d"),
                        "btc_eth_corr_30d": correlations.get("btc_eth_corr_30d"),
                        "btc_dxy_corr_7d": correlations.get("btc_dxy_corr_7d"),
                        "btc_dxy_corr_30d": correlations.get("btc_dxy_corr_30d"),
                        "btc_ndx_corr_7d": correlations.get("btc_ndx_corr_7d"),
                        "btc_ndx_corr_30d": correlations.get("btc_ndx_corr_30d")
                    },
                    "correlations": {
                        "btc_eth": {
                            "short_term_7d": correlations.get("btc_eth_corr_7d"),
                            "long_term_30d": correlations.get("btc_eth_corr_30d"),
                            "relationship": "cripto_major_pair"
                        },
                        "btc_dxy": {
                            "medium_term_30d": correlations.get("btc_dxy_corr_30d"),
                            "long_term_90d": correlations.get("btc_dxy_corr_90d"),
                            "relationship": "inverse_usd_strength"
                        },
                        "btc_ndx": {
                            "medium_term_30d": correlations.get("btc_ndx_corr_30d"),
                            "relationship": "tech_risk_correlation"
                        }
                    },
                    "market_context": {
                        "dxy_momentum": {
                            "return_5d": correlations.get("dxy_return_5d"),
                            "return_20d": correlations.get("dxy_return_20d"),
                            "momentum": correlations.get("dxy_momentum")
                        }
                    }
                }
        except Exception as e:
            cross_asset_context = {
                "error": f"Falha ao calcular correlações: {str(e)}",
                "features": {}
            }
    else:
        # Fallback vazio
        cross_asset_context = {
            "features": {},
            "correlations": {},
            "market_context": {}
        }

    # Adiciona o contexto cross-asset ao payload
    final_payload["cross_asset_context"] = cross_asset_context

    return final_payload

def _check_in_range(price, low, high):
    """Helper simples para verificar se preço está em range."""
    if price is None or low is None or high is None:
        return None
    return low <= price <= high