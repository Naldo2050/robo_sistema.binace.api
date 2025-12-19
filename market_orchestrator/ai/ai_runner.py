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
from format_utils import (
    format_price,
    format_large_number,
    format_delta,
)

# [AI_PAYLOAD_BUILDER] Importa o novo construtor de payload
from .ai_payload_builder import build_ai_input

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

    def ai_init_worker() -> None:
        try:
            with bot._ai_init_lock:
                if bot.ai_initialization_attempted:
                    return
                bot.ai_initialization_attempted = True

            logging.info("=" * 30 + " INICIALIZANDO IA " + "=" * 30)
            logging.info("🧠 Tentando inicializar AI Analyzer...")

            # Integração com HealthMonitor:
            # Se o bot tiver um health_monitor, passamos para o AIAnalyzer.
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

            except Exception as e:
                logging.error(f"❌ Falha ao inicializar ML Engine: {e}", exc_info=True)
                bot.ml_engine = None

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

            analysis = bot.ai_analyzer.analyze(test_event)

            min_chars = getattr(config, "AI_TEST_MIN_CHARS", 10)

            if analysis and len(analysis.get("raw_response", "")) >= min_chars:
                bot.ai_test_passed = True
                logging.info("✅ Teste da IA bem-sucedido!")

                logging.info("═" * 25 + " RESULTADO DO TESTE DA IA " + "═" * 25)
                logging.info(analysis.get("raw_response", ""))
                logging.info("═" * 75)
            else:
                bot.ai_test_passed = True
                logging.warning(
                    "⚠️ Teste da IA retornou resultado inesperado. "
                    "Prosseguindo em modo fallback."
                )
                logging.warning(f"Resultado recebido: {analysis}")
                logging.info("═" * 75)

        except Exception as e:
            bot.ai_analyzer = None
            bot.ai_test_passed = False

            logging.error("=" * 30 + " ERRO NA IA " + "=" * 30)
            logging.error(
                f"❌ Falha crítica ao inicializar a IA: {e}",
                exc_info=True,
            )
            logging.error("═" * 75)

    threading.Thread(target=ai_init_worker, daemon=True).start()


def run_ai_analysis_threaded(bot, event_data: Dict[str, Any]) -> None:
    """
    Executa a análise da IA em uma thread separada, com:
    - rate limiter
    - semaphore
    - pool de threads limitado
    - logs detalhados

    Equivalente ao método EnhancedMarketBot._run_ai_analysis_threaded original.
    """

    if not bot.ai_analyzer or not bot.ai_test_passed or bot.should_stop:
        if bot.ai_analyzer and not bot.ai_test_passed:
            logging.warning(
                "⚠️ Análise da IA ignorada: sistema não passou no teste inicial."
            )
        return

    logging.debug(
        "🔍 Evento recebido para análise da IA: %s",
        event_data.get("tipo_evento", "N/A"),
    )

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
                        # Executa previsão do modelo XGBoost
                        ml_prediction = bot.ml_engine.predict(event_data)

                        if ml_prediction.get("status") == "ok":
                            prob = ml_prediction.get("prob_up", 0.5)
                            confidence = ml_prediction.get("confidence", 0.0)

                            # Log da previsão quantitativa
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

                            # Injeta no event_data para uso no builder
                            event_data["ml_prediction"] = ml_prediction

                        else:
                            logging.warning(f"⚠️ ML Engine retornou status: {ml_prediction.get('status')}")

                    except Exception as e:
                        logging.error(f"❌ Erro na inferência ML: {e}", exc_info=True)
                        ml_prediction = {"status": "error", "msg": str(e)}
                else:
                    logging.debug("🤖 ML Engine não disponível - usando apenas IA Generativa")

                # Heartbeat extra (além do heartbeat periódico do AIAnalyzer)
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

                # [AI_PAYLOAD_BUILDER] Construção do payload estruturado para IA
                try:
                    enriched = event_data.get("enriched_snapshot", {})
                    flow_metrics = event_data.get("fluxo_continuo", {})
                    historical_profile = event_data.get("historical_vp", {})
                    macro_ctx = event_data.get("market_context", {})
                    market_env = event_data.get("market_environment", {})
                    # [PIVOTS] Extrai pivots do evento (vindos do ContextCollector)
                    pivots = event_data.get("pivots", {})
                    ob_data = event_data.get("orderbook_data", {})
                    ml_feats = event_data.get("ml_features") or {}

                    # Se não houver ml_features explícitas, tenta extrair via MLInferenceEngine
                    if not ml_feats and getattr(bot, "ml_engine", None):
                        try:
                            ml_feats = bot.ml_engine.extract_ml_features(event_data)
                        except Exception as e:
                            logging.debug(
                                f"Falha ao extrair ml_features via MLInferenceEngine: {e}",
                                exc_info=True,
                            )
                            ml_feats = {}

                    if not ml_feats:
                        logging.warning(
                            "⚠️ Nenhuma ml_feature disponível para este evento; "
                            "IA Generativa operará com menos contexto quantitativo."
                        )

                    ai_payload = build_ai_input(
                        symbol=bot.symbol,
                        signal=event_data,
                        enriched=enriched,
                        flow_metrics=flow_metrics,
                        historical_profile=historical_profile,
                        macro_context=macro_ctx,
                        market_environment=market_env,
                        orderbook_data=ob_data,
                        ml_features=ml_feats,
                        ml_prediction=ml_prediction,
                        pivots=pivots, # [NEW ARGUMENT]
                    )

                    # Anexa ao evento original, sem mudar o formato que a IA já espera
                    event_data["ai_payload"] = ai_payload

                except Exception as e:
                    logging.debug(
                        f"Falha ao construir ai_payload: {e}",
                        exc_info=True,
                    )

                analysis_result = bot.ai_analyzer.analyze(event_data)

                if analysis_result and not bot.should_stop:
                    try:
                        raw_response = analysis_result.get(
                            "raw_response", ""
                        )
                        _print_ai_report_clean(raw_response)
                        logging.info("✅ Análise da IA concluída com sucesso")

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

                            # ai_result: usa structured se disponível, senão raw_response parseada
                            ai_result_json = analysis_result.get("structured")
                            if ai_result_json is None:
                                try:
                                    ai_result_json = json.loads(analysis_result.get("raw_response", "{}"))
                                except:
                                    ai_result_json = {"raw_response": analysis_result.get("raw_response", "")}

                            # ============================================
                            # [HYBRID_DECISION] Fusão de Decisão Híbrida
                            # ============================================
                            if HYBRID_AVAILABLE and getattr(config, "HYBRID_ENABLED", True):
                                try:
                                    # Obtém previsão do modelo (já calculada anteriormente)
                                    ml_pred = ml_prediction if ml_prediction.get("status") == "ok" else None
                                    
                                    # Faz fusão de decisões
                                    hybrid_result = fuse_decisions(ml_pred, ai_result_json)
                                    
                                    # Converte para formato compatível com AITradeAnalysis
                                    ai_result_json = decision_to_ai_result(hybrid_result)
                                    
                                    logging.info(
                                        f"🧠 Decisão Final: {hybrid_result.action.upper()} "
                                        f"(conf={hybrid_result.confidence:.0%}, source={hybrid_result.source})"
                                    )
                                    
                                except Exception as e:
                                    logging.warning(f"⚠️ Erro na fusão híbrida, usando IA pura: {e}")
                                    # Mantém ai_result_json original
                            
                            # Filtro de confiança: se < 0.7, força action para "wait"
                            if isinstance(ai_result_json, dict):
                                action = ai_result_json.get("action", "wait")
                                confidence = ai_result_json.get("confidence", 0.0)
                                if confidence < 0.7:
                                    ai_result_json["action"] = "wait"

                            ai_event = {
                                "tipo_evento": "AI_ANALYSIS",
                                "symbol": symbol,
                                "timestamp_ms": anchor_ts_ms,
                                "anchor_price": anchor_price,
                                "anchor_window_id": anchor_window_id,
                                "ai_result": ai_result_json,
                                "ai_payload": {
                                    "price_context": ai_payload.get("price_context", {}),
                                    "flow_context": ai_payload.get("flow_context", {}),
                                    "orderbook_context": ai_payload.get("orderbook_context", {}),
                                    "macro_context": ai_payload.get("macro_context", {}),
                                    "liquidity_heatmap": ai_payload.get("fluxo_continuo", {}).get("liquidity_heatmap", {}),
                                },
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