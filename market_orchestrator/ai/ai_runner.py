# ai/ai_runner.py
# -*- coding: utf-8 -*-

"""
Lógica de inicialização e execução da IA do EnhancedMarketBot.

Extraído dos métodos _initialize_ai_async e _run_ai_analysis_threaded
do arquivo market_orchestrator.py original, adaptados para funções
que recebem `bot` como argumento.
"""

import threading
import logging
import time
import json
from typing import Any, Dict

import config
from ai_analyzer_qwen import AIAnalyzer  # arquivo na raiz do projeto
from ..utils.price_fetcher import get_current_price
from export_signals import create_chart_signal_from_event, export_signal_to_csv
from format_utils import (
    format_price,
    format_large_number,
    format_delta,
)

from orderbook_core.structured_logging import StructuredLogger
from orderbook_core.tracing_utils import TracerWrapper

# [AI_PAYLOAD_BUILDER] Importa o novo construtor de payload
from .ai_payload_builder import build_ai_input
from src.utils.ai_payload_optimizer import AIPayloadOptimizer

# [DEDUP + COMPRESS V3] Pipeline de otimização
from .raw_event_deduplicator import deduplicate_event
from .payload_compressor_v3 import compress_payload_v3

# [HYBRID_DECISION] Importa módulo de decisão híbrida
try:
    from ml.hybrid_decision import (
        fuse_decisions,
        decision_to_ai_result,
        HybridDecisionMaker,
    )
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False
    fuse_decisions = None
    decision_to_ai_result = None


def initialize_ai_async(bot) -> None:
    """
    Inicializa a IA em uma thread separada, exatamente como no método
    EnhancedMarketBot._initialize_ai_async original.

    v2.3.x: agora passa o HealthMonitor do bot para o AIAnalyzer,
    permitindo heartbeat periódico do módulo 'ai'.
    """

    # Usa logger/tracer do bot se existirem, senão cria locais
    slog = getattr(
        bot,
        "slog",
        StructuredLogger("ai_runner", getattr(bot, "symbol", "UNKNOWN")),
    )
    tracer = getattr(
        bot,
        "tracer",
        TracerWrapper(
            service_name="enhanced_market_bot",
            component="ai",
            symbol=getattr(bot, "symbol", "UNKNOWN"),
        ),
    )

    def ai_init_worker() -> None:
        with tracer.start_span(
            "ai_init",
            {"symbol": getattr(bot, "symbol", "UNKNOWN")},
        ):
            try:
                with bot._ai_init_lock:
                    if bot.ai_initialization_attempted:
                        return
                    bot.ai_initialization_attempted = True

                logging.info("=" * 30 + " INICIALIZANDO IA " + "=" * 30)
                logging.info("🧠 Tentando inicializar AI Analyzer...")

                try:
                    slog.info("ai_init_start")
                except Exception:
                    pass

                # Integração com HealthMonitor:
                try:
                    hm = getattr(bot, "health_monitor", None)
                except Exception:
                    hm = None

                bot.ai_analyzer = AIAnalyzer(
                    health_monitor=hm,
                    module_name="ai",
                )

                # ============================================
                # Inicializa Motor de Inferência Quantitativa
                # ============================================
                try:
                    from ml.inference_engine import MLInferenceEngine
                    bot.ml_engine = MLInferenceEngine()
                    logging.info("🤖 Motor de Inferência Quantitativa (XGBoost) inicializado")

                    # Teste rápido do ML Engine
                    test_result = bot.ml_engine.predict({
                        "delta": 0.5,
                        "volume_total": 10000,
                        "fluxo_continuo": {"microstructure": {"tick_rule_sum": 0.2}}
                    })

                    if test_result.get("status") == "ok":
                        logging.info(f"✅ ML Engine testado: {test_result.get('prob_up', 0):.1%}")
                    else:
                        logging.warning(f"⚠️ ML Engine teste falhou: {test_result.get('status')}")

                    try:
                        slog.info(
                            "ml_engine_initialized",
                            status=test_result.get("status"),
                        )
                    except Exception:
                        pass

                except Exception as e:
                    logging.error(f"❌ Falha ao inicializar ML Engine: {e}", exc_info=True)
                    bot.ml_engine = None
                    try:
                        slog.error(
                            "ml_engine_init_error",
                            error=str(e),
                        )
                    except Exception:
                        pass

                logging.info(
                    "✅ Módulo da IA carregado. Realizando teste de análise..."
                )

                current_price = get_current_price(bot.symbol)

                test_event = {
                    "tipo_evento": "Teste de Conexão",
                    "ativo": bot.symbol,
                    "descricao": (
                        "Teste inicial do sistema de análise "
                        "para garantir operacionalidade."
                    ),
                    "delta": 150.5,
                    "volume_total": 50000,
                    "preco_fechamento": current_price,
                    "orderbook_data": {
                        "bid_depth_usd": 1000000,
                        "ask_depth_usd": 800000,
                        "bid_usd": 1000000,
                        "ask_usd": 800000,
                        "imbalance": 0.2,
                        "mid": current_price,
                        "spread": 0.10,
                        "spread_percent": 0.0001,
                    },
                    "spread_metrics": {
                        "bid_depth_usd": 1000000,
                        "ask_depth_usd": 800000,
                    },
                }

                ai_payload = None
                try:
                    from datetime import datetime, timezone
                    import time

                    now_ms = int(time.time() * 1000)
                    test_event["symbol"] = bot.symbol
                    test_event["epoch_ms"] = now_ms
                    test_event["timestamp_utc"] = datetime.now(timezone.utc).isoformat()

                    # opcionais p/ evitar missing flow/tf no teste:
                    test_event.setdefault("fluxo_continuo", {
                        "cvd": 0.0,
                        "order_flow": {
                            "net_flow_1m": 0.0,
                            "flow_imbalance": 0.0,
                            "aggressive_buy_pct": 50.0,
                            "aggressive_sell_pct": 50.0,
                        },
                    })
                    test_event.setdefault("multi_tf", {
                        "1h": {"tendencia": "Neutra", "preco_atual": current_price}
                    })

                    ml_feats = test_event.get("ml_features") or {}
                    if not ml_feats and getattr(bot, "ml_engine", None):
                        try:
                            ml_feats = bot.ml_engine.extract_ml_features(test_event)  # type: ignore
                        except Exception as e:
                            logging.debug("Falha ao extrair ml_features via MLInferenceEngine: %s", e, exc_info=True)
                            ml_feats = {}

                    ai_payload = build_ai_input(
                        symbol=bot.symbol,
                        signal=test_event,
                        enriched=test_event.get("enriched_snapshot", {}),
                        flow_metrics=test_event.get("fluxo_continuo", {}),
                        historical_profile=test_event.get("historical_vp", {}),
                        macro_context=test_event.get("market_context", {}),
                        market_environment=test_event.get("market_environment", {}),
                        orderbook_data=test_event.get("orderbook_data", {}),
                        ml_features=ml_feats,
                        ml_prediction={},
                        pivots=test_event.get("pivots", {}),
                    )

                    _payload_bytes = len(json.dumps(ai_payload, ensure_ascii=False).encode("utf-8"))
                    logging.info("PAYLOAD_READY_FOR_LLM bytes=%d keys=%s v=%s",
                                 _payload_bytes, list(ai_payload.keys())[:8], ai_payload.get("_v", 2))

                except Exception as e:
                    logging.warning("Falha ao construir ai_payload para teste de conexão: %s", e, exc_info=True)

                analysis_input = ai_payload if isinstance(ai_payload, dict) and ai_payload else test_event
                analysis = bot.ai_analyzer.analyze(analysis_input)

                min_chars = getattr(config, "AI_TEST_MIN_CHARS", 10)

                if analysis and len(analysis.get("raw_response", "")) >= min_chars:
                    bot.ai_test_passed = True
                    logging.info("✅ Teste da IA bem-sucedido!")
                    logging.info("═" * 25 + " RESULTADO DO TESTE DA IA " + "═" * 25)
                    logging.info(analysis.get("raw_response", ""))
                    logging.info("═" * 75)

                    try:
                        slog.info(
                            "ai_init_success",
                            test_response_len=len(analysis.get("raw_response", "")),
                        )
                    except Exception:
                        pass
                else:
                    bot.ai_test_passed = True
                    logging.warning(
                        "⚠️ Teste da IA retornou resultado inesperado. "
                        "Prosseguindo em modo fallback."
                    )
                    logging.warning(f"Resultado recebido: {analysis}")
                    logging.info("═" * 75)

                    try:
                        slog.warning(
                            "ai_init_unexpected_result",
                            test_response_len=len(
                                (analysis or {}).get("raw_response", "")
                            ),
                        )
                    except Exception:
                        pass

            except Exception as e:
                bot.ai_analyzer = None
                bot.ai_test_passed = False

                logging.error("=" * 30 + " ERRO NA IA " + "=" * 30)
                logging.error(
                    f"❌ Falha crítica ao inicializar a IA: {e}",
                    exc_info=True,
                )
                logging.error("═" * 75)

                try:
                    slog.error(
                        "ai_init_error",
                        error=str(e),
                    )
                except Exception:
                    pass

    threading.Thread(target=ai_init_worker, daemon=True).start()


def run_ai_analysis_threaded(bot, event_data: Dict[str, Any]) -> None:
    """
    Executa a análise da IA em uma thread separada, com:
    - rate limiter
    - semaphore
    - pool de threads limitado
    - logs detalhados
    - otimização para pular análise em condições de baixo volume/volatilidade lateral

    Equivalente ao método EnhancedMarketBot._run_ai_analysis_threaded original.
    """

    if not bot.ai_analyzer or not bot.ai_test_passed or bot.should_stop:
        if bot.ai_analyzer and not bot.ai_test_passed:
            logging.warning(
                "⚠️ Análise da IA ignorada: sistema não passou no teste inicial."
            )
        return

    # ============================================
    # [AI_OPTIMIZATION] Otimização para economizar tokens de IA
    # ============================================
    # Pula análise da IA se volatility_regime for 'SIDEWAYS' e volume estiver baixo
    try:
        regime_analysis = event_data.get("regime_analysis", {})
        volatility_regime = regime_analysis.get("volatility_regime", "").upper()

        # Verifica se está em regime lateral
        if volatility_regime == "SIDEWAYS":
            # Verifica volume baixo
            volume_total = event_data.get("volume_total", 0)
            volume_threshold = getattr(config, "AI_SKIP_VOLUME_THRESHOLD", 100000)  # 100k USD default

            if volume_total < volume_threshold:
                logging.info(
                    f"🤖 IA pulada: regime SIDEWAYS + volume baixo "
                    f"({format_large_number(volume_total)} < {format_large_number(volume_threshold)})"
                )

                # Usa logger estruturado se disponível
                slog = getattr(
                    bot,
                    "slog",
                    StructuredLogger("ai_runner", getattr(bot, "symbol", "UNKNOWN")),
                )
                try:
                    slog.info(
                        "ai_analysis_skipped",
                        reason="sideways_low_volume",
                        volatility_regime=volatility_regime,
                        volume_total=volume_total,
                        volume_threshold=volume_threshold,
                        tipo_evento=event_data.get("tipo_evento"),
                    )
                except Exception:
                    pass

                return

    except Exception as e:
        logging.debug(f"Erro na verificação de otimização da IA: {e}")
        # Continua normalmente se houver erro na verificação

    logging.debug(
        "🔍 Evento recebido para análise da IA: %s",
        event_data.get("tipo_evento", "N/A"),
    )

    # Usa logger/tracer do bot se existirem, senão cria locais
    slog = getattr(
        bot,
        "slog",
        StructuredLogger("ai_runner", getattr(bot, "symbol", "UNKNOWN")),
    )
    tracer = getattr(
        bot,
        "tracer",
        TracerWrapper(
            service_name="enhanced_market_bot",
            component="ai",
            symbol=getattr(bot, "symbol", "UNKNOWN"),
        ),
    )

    try:
        slog.info(
            "ai_analysis_scheduled",
            tipo_evento=event_data.get("tipo_evento"),
            resultado_da_batalha=event_data.get("resultado_da_batalha"),
        )
    except Exception:
        pass

    # ============================================
    # [EXPORT_SIGNALS] Exportação de Sinais para CSV
    # ============================================
    try:
        # Extrai dados necessários para criar o sinal
        enriched_snapshot = event_data.get("enriched_snapshot", {})
        historical_profile = event_data.get("historical_vp", {})
        market_environment = event_data.get("market_environment", {})
        orderbook_data = event_data.get("orderbook_data", {})
        
        # Cria o sinal para exportação
        signal = create_chart_signal_from_event(
            event_data=event_data,
            symbol=bot.symbol,
            exchange="BINANCE",  # Pode ser parametrizado depois
            enriched_snapshot=enriched_snapshot,
            historical_profile=historical_profile,
            market_environment=market_environment,
            orderbook_data=orderbook_data
        )
        
        # Exporta o sinal para CSV
        export_signal_to_csv(signal)
        
        try:
            slog.info(
                "signal_exported",
                symbol=signal.symbol,
                event_type=signal.event_type,
                side=signal.side,
                strength=signal.strength,
            )
        except Exception:
            pass
            
    except Exception as e:
        logging.debug(
            f"Falha ao exportar sinal para CSV: {e}",
            exc_info=True,
        )
        try:
            slog.warning(
                "signal_export_error",
                error=str(e),
                tipo_evento=event_data.get("tipo_evento"),
            )
        except Exception:
            pass

    def _print_ai_report_clean(report_text: str) -> None:
        if not report_text:
            return

        header = "ANÁLISE PROFISSIONAL DA IA"
        start = (report_text or "")[:200].upper()
        sep = "═" * 75

        if header in start:
            logging.info("\n" + report_text.rstrip())
        else:
            logging.info(
                "\n" + "═" * 25 + " " + header + " " + "═" * 25
            )
            logging.info(report_text)

        logging.info(sep)

    def ai_worker() -> None:
        with tracer.start_span(
            "ai_analysis",
            {
                "tipo_evento": event_data.get("tipo_evento"),
                "resultado_da_batalha": event_data.get("resultado_da_batalha"),
            },
        ):
            try:
                bot.ai_rate_limiter.acquire()

                with bot.ai_semaphore:
                    logging.info(
                        "🧠 IA iniciando análise para evento: %s",
                        event_data.get("resultado_da_batalha", "N/A"),
                    )

                    # ============================================
                    # [INTELIGÊNCIA HÍBRIDA] Inferência Quantitativa
                    # ============================================
                    ml_prediction = {}
                    if hasattr(bot, 'ml_engine') and bot.ml_engine:
                        try:
                            ml_prediction = bot.ml_engine.predict(event_data)

                            if ml_prediction.get("status") == "ok":
                                prob = ml_prediction.get("prob_up", 0.5)
                                confidence = ml_prediction.get("confidence", 0.0)

                                if prob > 0.6:
                                    bias = "📈 ALTISTA"
                                elif prob < 0.4:
                                    bias = "📉 BAIXISTA"
                                else:
                                    bias = "⚖️  NEUTRO"

                                logging.info(
                                    f"🤖 ML Prediction: {bias} "
                                    f"(Prob: {prob:.1%}, Conf: {confidence:.1%})"
                                )

                                event_data["ml_prediction"] = ml_prediction

                                try:
                                    slog.info(
                                        "ml_prediction_done",
                                        prob_up=prob,
                                        confidence=confidence,
                                    )
                                except Exception:
                                    pass

                            else:
                                logging.warning(f"⚠️ ML Engine retornou status: {ml_prediction.get('status')}")
                        except Exception as e:
                            logging.error(f"❌ Erro na inferência ML: {e}", exc_info=True)
                            ml_prediction = {"status": "error", "msg": str(e)}
                    else:
                        logging.debug("🤖 ML Engine não disponível - usando apenas IA Generativa")

                    # Heartbeat extra
                    try:
                        bot.health_monitor.heartbeat("ai")
                    except Exception:
                        pass

                    try:
                        logging.debug(
                            "📊 Dados do evento para IA: %s",
                            {
                                "tipo": event_data.get("tipo_evento"),
                                "delta": format_delta(event_data.get("delta")),
                                "volume": format_large_number(
                                    event_data.get("volume_total")
                                ),
                                "preco": format_price(
                                    event_data.get("preco_fechamento")
                                ),
                            },
                        )
                    except Exception:
                        pass

                    # [RAW_EVENT_DEDUP] Deduplicar evento ANTES de construir payload
                    # NOTA: NÃO reatribuir event_data aqui — ele é capturado
                    # da closure externa e qualquer `event_data = ...` faria
                    # Python tratá-lo como variável local em todo ai_worker(),
                    # causando UnboundLocalError na linha do tracer.start_span.
                    try:
                        _deduped = deduplicate_event(event_data, deep_copy=False)
                        _dedup_stats = getattr(_deduped, '_dedup_stats', None)
                        event_data.clear()
                        event_data.update(_deduped)
                        logging.info(
                            "RAW_EVENT_DEDUP: deduplicacao aplicada, keys=%d",
                            len(event_data),
                        )
                    except Exception as _dedup_err:
                        logging.warning("Dedup falhou (não-crítico): %s", _dedup_err)

                    # [AI_PAYLOAD_BUILDER] Construção do payload
                    try:
                        ai_payload = event_data

                        # [COMPRESS_V3] Comprimir evento original para LLM (~59% economia)
                        try:
                            _pre_bytes = len(json.dumps(event_data, ensure_ascii=False).encode("utf-8"))

                            # DEBUG: logar chaves antes da compressão V3
                            logging.info(
                                "DEBUG_PAYLOAD_KEYS_BEFORE_V3: %s",
                                json.dumps(list(event_data.keys())),
                            )
                            logging.info(
                                "DEBUG_PAYLOAD_TYPES_BEFORE_V3: %s",
                                json.dumps({k: type(v).__name__ for k, v in event_data.items()}),
                            )

                            ai_payload = compress_payload_v3(event_data)

                            if ml_prediction and ml_prediction.get("status") == "ok":
                                ai_payload["quant"] = {
                                    "prob_up": round(ml_prediction.get("prob_up", 0.5), 2),
                                    "conf": round(ml_prediction.get("confidence", 0), 2),
                                }

                            _post_bytes = len(json.dumps(ai_payload, ensure_ascii=False).encode("utf-8"))
                            logging.info(
                                "PAYLOAD_COMPRESSED_V3 before=%d after=%d saved=%d%% keys=%s",
                                _pre_bytes,
                                _post_bytes,
                                int((1 - _post_bytes / max(_pre_bytes, 1)) * 100),
                                list(ai_payload.keys())[:10],
                            )
                        except Exception as _comp_err:
                            logging.debug("Compress V3 falhou (usando original): %s", _comp_err)

                        event_data["ai_payload"] = ai_payload

                    except Exception as e:
                        logging.debug(
                            f"Falha ao construir ai_payload: {e}",
                            exc_info=True,
                        )

                    analysis_result = bot.ai_analyzer.analyze(event_data)

                    if analysis_result and not bot.should_stop:
                        try:
                            raw_response = analysis_result.get("raw_response", "")
                            _print_ai_report_clean(raw_response)
                            logging.info("✅ Análise da IA concluída com sucesso")

                            try:
                                slog.info(
                                    "ai_analysis_done",
                                    tipo_evento=event_data.get("tipo_evento"),
                                    resultado_da_batalha=event_data.get("resultado_da_batalha"),
                                )
                            except Exception:
                                pass

                            # [AI_EVENT_SAVE] Salva evento de análise da IA
                            try:
                                ai_payload = event_data.get("ai_payload", {})
                                symbol = event_data.get("ativo") or event_data.get("symbol") or bot.symbol
                                anchor_ts_ms = (
                                    event_data.get("epoch_ms")
                                    or event_data.get("timestamp_ms")
                                    or int(time.time() * 1000)
                                )
                                anchor_price = event_data.get("preco_fechamento") or event_data.get("preco_atual")
                                anchor_window_id = event_data.get("window_id") or event_data.get("janela_numero")

                                ai_result_json = analysis_result.get("structured")
                                if ai_result_json is None:
                                    try:
                                        ai_result_json = json.loads(analysis_result.get("raw_response", "{}"))
                                    except:
                                        ai_result_json = {
                                            "raw_response": analysis_result.get("raw_response", "")
                                        }

                                # [HYBRID_DECISION]
                                if HYBRID_AVAILABLE and getattr(config, "HYBRID_ENABLED", True) and fuse_decisions and decision_to_ai_result:
                                    try:
                                        ml_pred = ml_prediction if ml_prediction.get("status") == "ok" else None
                                        hybrid_result = fuse_decisions(ml_pred, ai_result_json)
                                        ai_result_json = decision_to_ai_result(hybrid_result)

                                        logging.info(
                                            f"🧠 Decisão Final: {hybrid_result.action.upper()} "
                                            f"(conf={hybrid_result.confidence:.0%}, source={hybrid_result.source})"
                                        )

                                        try:
                                            slog.info(
                                                "ai_hybrid_decision",
                                                action=hybrid_result.action,
                                                confidence=float(hybrid_result.confidence),
                                                source=hybrid_result.source,
                                            )
                                        except Exception:
                                            pass

                                    except Exception as e:
                                        logging.warning(
                                            f"⚠️ Erro na fusão híbrida, usando IA pura: {e}"
                                        )

                                # Filtro de confiança
                                if isinstance(ai_result_json, dict):
                                    action = ai_result_json.get("action", "wait")
                                    confidence = ai_result_json.get("confidence", 0.0)
                                    if confidence < 0.7:
                                        ai_result_json["action"] = "wait"

                                # Usar o payload otimizado (formato compacto) ao salvar o evento
                                # O ai_payload já foi otimizado pelo AIPayloadOptimizer.optimize() acima
                                ai_event = {
                                    "tipo_evento": "AI_ANALYSIS",
                                    "symbol": symbol,
                                    "timestamp_ms": anchor_ts_ms,
                                    "anchor_price": anchor_price,
                                    "anchor_window_id": anchor_window_id,
                                    "ai_result": ai_result_json,
                                    "ai_payload": ai_payload,  # Usa o payload já otimizado
                                }

                                if hasattr(bot, "event_saver") and bot.event_saver:
                                    bot.event_saver.save_event(ai_event)

                            except Exception as e:
                                logging.debug(
                                    f"Falha ao salvar evento de análise da IA: {e}",
                                    exc_info=True,
                                )

                        except Exception as e:
                            logging.error(
                                f"❌ Erro ao processar resposta da IA: {e}",
                                exc_info=True,
                            )

            except Exception as e:
                logging.error(
                    f"❌ Erro na thread de análise da IA: {e}",
                    exc_info=True,
                )
                try:
                    slog.error(
                        "ai_analysis_error",
                        error=str(e),
                        tipo_evento=event_data.get("tipo_evento"),
                    )
                except Exception:
                    pass
            finally:
                with bot._ai_pool_lock:
                    try:
                        current_thread = threading.current_thread()
                        bot.ai_thread_pool = [
                            t
                            for t in bot.ai_thread_pool
                            if t is not current_thread and t.is_alive()
                        ]
                    except Exception as e:
                        logging.debug(f"Erro ao limpar thread pool: {e}")

    logging.debug("🔧 Criando thread para análise da IA...")
    t = threading.Thread(target=ai_worker, daemon=True)

    with bot._ai_pool_lock:
        bot.ai_thread_pool = [th for th in bot.ai_thread_pool if th.is_alive()]

        if len(bot.ai_thread_pool) >= bot.max_ai_threads:
            logging.warning("⚠️ Thread pool da IA cheio, aguardando...")
            bot.ai_thread_pool[0].join(timeout=5.0)
            bot.ai_thread_pool = [
                th for th in bot.ai_thread_pool if th.is_alive()
            ]

        bot.ai_thread_pool.append(t)

    t.start()
