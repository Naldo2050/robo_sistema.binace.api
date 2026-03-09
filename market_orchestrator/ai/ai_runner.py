# ai/ai_runner.py
# -*- coding: utf-8 -*-

"""
LÃ³gica de inicializaÃ§Ã£o e execuÃ§Ã£o da IA do EnhancedMarketBot.

ExtraÃ­do dos mÃ©todos _initialize_ai_async e _run_ai_analysis_threaded
do arquivo market_orchestrator.py original, adaptados para funÃ§Ãµes
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

# [BUILD_COMPACT] Payload compacto direto do event_data
from build_compact_payload import build_compact_payload

# [DEDUP] Deduplicador de eventos
from .raw_event_deduplicator import deduplicate_event

# [HYBRID_DECISION] Importa mÃ³dulo de decisÃ£o hÃ­brida
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
    Inicializa a IA em uma thread separada, exatamente como no mÃ©todo
    EnhancedMarketBot._initialize_ai_async original.

    v2.3.x: agora passa o HealthMonitor do bot para o AIAnalyzer,
    permitindo heartbeat periÃ³dico do mÃ³dulo 'ai'.
    """

    # Usa logger/tracer do bot se existirem, senÃ£o cria locais
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
                logging.info("AI init: tentando inicializar AI Analyzer...")

                try:
                    slog.info("ai_init_start")
                except Exception:
                    pass

                # IntegraÃ§Ã£o com HealthMonitor:
                try:
                    hm = getattr(bot, "health_monitor", None)
                except Exception:
                    hm = None

                bot.ai_analyzer = AIAnalyzer(
                    health_monitor=hm,
                    module_name="ai",
                )

                # ============================================
                # Inicializa Motor de InferÃªncia Quantitativa
                # ============================================
                try:
                    from ml.inference_engine import MLInferenceEngine
                    bot.ml_engine = MLInferenceEngine()
                    logging.info("ML engine: XGBoost inicializado")

                    # Teste rÃ¡pido do ML Engine
                    test_result = bot.ml_engine.predict({
                        "delta": 0.5,
                        "volume_total": 10000,
                        "fluxo_continuo": {"microstructure": {"tick_rule_sum": 0.2}}
                    })

                    if test_result.get("status") == "ok":
                        logging.info(f"ML engine testado: {test_result.get('prob_up', 0):.1%}")
                    else:
                        logging.warning(f"âš ï¸ ML Engine teste falhou: {test_result.get('status')}")

                    try:
                        slog.info(
                            "ml_engine_initialized",
                            status=test_result.get("status"),
                        )
                    except Exception:
                        pass

                except Exception as e:
                    logging.error(f"âŒ Falha ao inicializar ML Engine: {e}", exc_info=True)
                    bot.ml_engine = None
                    try:
                        slog.error(
                            "ml_engine_init_error",
                            error=str(e),
                        )
                    except Exception:
                        pass

                logging.info(
                    "Modulo da IA carregado. Realizando teste de analise..."
                )

                current_price = get_current_price(bot.symbol)

                test_event = {
                    "tipo_evento": "Teste de ConexÃ£o",
                    "ativo": bot.symbol,
                    "descricao": (
                        "Teste inicial do sistema de anÃ¡lise "
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

                    ai_payload = build_compact_payload(test_event)

                    _payload_bytes = len(json.dumps(ai_payload, ensure_ascii=False).encode("utf-8"))
                    logging.info("PAYLOAD_READY_FOR_LLM bytes=%d keys=%s v=%s",
                                 _payload_bytes, list(ai_payload.keys())[:8], ai_payload.get("_v", 2))

                except Exception as e:
                    logging.warning("Falha ao construir ai_payload para teste de conexÃ£o: %s", e, exc_info=True)

                analysis_input = dict(test_event)
                if isinstance(ai_payload, dict) and ai_payload:
                    analysis_input["ai_payload"] = ai_payload
                analysis = bot.ai_analyzer.analyze(analysis_input)

                min_chars = getattr(config, "AI_TEST_MIN_CHARS", 10)
                analysis_ok = bool((analysis or {}).get("success"))
                analysis_fallback = bool((analysis or {}).get("is_fallback"))

                if analysis_ok and len(analysis.get("raw_response", "")) >= min_chars:
                    bot.ai_test_passed = True
                    logging.info("Teste da IA bem-sucedido")
                    logging.info("=" * 25 + " RESULTADO DO TESTE DA IA " + "=" * 25)
                    logging.info(analysis.get("raw_response", ""))
                    logging.info("=" * 75)

                    try:
                        slog.info(
                            "ai_init_success",
                            test_response_len=len(analysis.get("raw_response", "")),
                        )
                    except Exception:
                        pass
                elif analysis_fallback and len((analysis or {}).get("raw_response", "")) >= min_chars:
                    bot.ai_test_passed = True
                    logging.warning(
                        "Teste da IA concluído com fallback estruturado | reason=%s",
                        (analysis or {}).get("fallback_reason") or "unknown",
                    )
                    logging.info("=" * 25 + " RESULTADO DO TESTE DA IA " + "=" * 25)
                    logging.info(analysis.get("raw_response", ""))
                    logging.info("=" * 75)

                    try:
                        slog.warning(
                            "ai_init_fallback",
                            test_response_len=len((analysis or {}).get("raw_response", "")),
                            fallback_reason=(analysis or {}).get("fallback_reason"),
                        )
                    except Exception:
                        pass
                else:
                    bot.ai_test_passed = True
                    logging.warning(
                        "âš ï¸ Teste da IA retornou resultado inesperado. "
                        "Prosseguindo em modo fallback."
                    )
                    logging.warning(f"Resultado recebido: {analysis}")
                    logging.info("â•" * 75)

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
                    f"âŒ Falha crÃ­tica ao inicializar a IA: {e}",
                    exc_info=True,
                )
                logging.error("â•" * 75)

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
    Executa a anÃ¡lise da IA em uma thread separada, com:
    - rate limiter
    - semaphore
    - pool de threads limitado
    - logs detalhados
    - otimizaÃ§Ã£o para pular anÃ¡lise em condiÃ§Ãµes de baixo volume/volatilidade lateral

    Equivalente ao mÃ©todo EnhancedMarketBot._run_ai_analysis_threaded original.
    """

    if not bot.ai_analyzer or not bot.ai_test_passed or bot.should_stop:
        if bot.ai_analyzer and not bot.ai_test_passed:
            logging.warning(
                "âš ï¸ AnÃ¡lise da IA ignorada: sistema nÃ£o passou no teste inicial."
            )
        return

    # ============================================
    # [AI_OPTIMIZATION] OtimizaÃ§Ã£o para economizar tokens de IA
    # ============================================
    # Pula anÃ¡lise da IA se volatility_regime for 'SIDEWAYS' e volume estiver baixo
    try:
        regime_analysis = event_data.get("regime_analysis", {})
        volatility_regime = regime_analysis.get("volatility_regime", "").upper()

        # Verifica se estÃ¡ em regime lateral
        if volatility_regime == "SIDEWAYS":
            # Verifica volume baixo
            volume_total = event_data.get("volume_total", 0)
            volume_threshold = getattr(config, "AI_SKIP_VOLUME_THRESHOLD", 100000)  # 100k USD default

            if volume_total < volume_threshold:
                logging.info(
                    f"ðŸ¤– IA pulada: regime SIDEWAYS + volume baixo "
                    f"({format_large_number(volume_total)} < {format_large_number(volume_threshold)})"
                )

                # Usa logger estruturado se disponÃ­vel
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
        logging.debug(f"Erro na verificaÃ§Ã£o de otimizaÃ§Ã£o da IA: {e}")
        # Continua normalmente se houver erro na verificaÃ§Ã£o

    logging.debug(
        "ðŸ” Evento recebido para anÃ¡lise da IA: %s",
        event_data.get("tipo_evento", "N/A"),
    )

    # Usa logger/tracer do bot se existirem, senÃ£o cria locais
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
    # [EXPORT_SIGNALS] ExportaÃ§Ã£o de Sinais para CSV
    # ============================================
    try:
        # Extrai dados necessÃ¡rios para criar o sinal
        enriched_snapshot = event_data.get("enriched_snapshot", {})
        historical_profile = event_data.get("historical_vp", {})
        market_environment = event_data.get("market_environment", {})
        orderbook_data = event_data.get("orderbook_data", {})
        
        # Cria o sinal para exportaÃ§Ã£o
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

        header = "ANÃLISE PROFISSIONAL DA IA"
        start = (report_text or "")[:200].upper()
        sep = "â•" * 75

        if header in start:
            logging.info("\n" + report_text.rstrip())
        else:
            logging.info(
                "\n" + "â•" * 25 + " " + header + " " + "â•" * 25
            )
            logging.info(report_text)

        logging.info(sep)

    def _print_ai_report_json(report_payload: Dict[str, Any]) -> None:
        if not isinstance(report_payload, dict) or not report_payload:
            return

        header = "ANALISE PROFISSIONAL DA IA"
        sep = "=" * 79
        logging.info("")
        logging.info(sep)
        logging.info(header)
        logging.info(sep)
        logging.info(
            json.dumps(report_payload, ensure_ascii=False, separators=(",", ":"))
        )
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
                    event_label = event_data.get("resultado_da_batalha")
                    if not event_label or str(event_label).strip().upper() in {"N/A", "NONE", "NULL"}:
                        event_label = event_data.get("tipo_evento") or "UNKNOWN"
                    logging.info(
                        "AI analysis start | event=%s",
                        event_label,
                    )

                    # ============================================
                    # [INTELIGÃŠNCIA HÃBRIDA] InferÃªncia Quantitativa
                    # ============================================
                    ml_prediction = {}
                    if hasattr(bot, 'ml_engine') and bot.ml_engine:
                        try:
                            ml_prediction = bot.ml_engine.predict(event_data)

                            if ml_prediction.get("status") == "ok":
                                prob = ml_prediction.get("prob_up", 0.5)
                                confidence = ml_prediction.get("confidence", 0.0)

                                if prob > 0.6:
                                    bias = "ðŸ“ˆ ALTISTA"
                                elif prob < 0.4:
                                    bias = "ðŸ“‰ BAIXISTA"
                                else:
                                    bias = "âš–ï¸  NEUTRO"

                                logging.info(
                                    f"ML prediction: {bias} "
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
                                logging.warning(f"âš ï¸ ML Engine retornou status: {ml_prediction.get('status')}")
                        except Exception as e:
                            logging.error(f"âŒ Erro na inferÃªncia ML: {e}", exc_info=True)
                            ml_prediction = {"status": "error", "msg": str(e)}
                    else:
                        logging.debug("ðŸ¤– ML Engine nÃ£o disponÃ­vel - usando apenas IA Generativa")

                    # Heartbeat extra
                    try:
                        bot.health_monitor.heartbeat("ai")
                    except Exception:
                        pass

                    try:
                        logging.debug(
                            "ðŸ“Š Dados do evento para IA: %s",
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
                    # NOTA: deep_copy=True para nÃ£o corromper o evento original.
                    # NÃ£o fazemos clear()+update() pois build_compact_payload
                    # lÃª direto do event_data original â€” dedup Ã© sÃ³ para o LLM.
                    try:
                        _deduped = deduplicate_event(event_data, deep_copy=True)
                        logging.info(
                            "RAW_EVENT_DEDUP: deduplicacao aplicada, keys=%d",
                            len(_deduped),
                        )
                    except Exception as _dedup_err:
                        logging.warning("Dedup falhou (nÃ£o-crÃ­tico): %s", _dedup_err)
                        _deduped = event_data

                    # [BUILD_COMPACT] Payload compacto direto do event_data
                    try:
                        ai_payload = build_compact_payload(event_data)

                        # Preencher quant model com ML prediction real
                        if ml_prediction.get("status") == "ok":
                            ai_payload["quant"] = {
                                "prob_up": round(float(ml_prediction.get("prob_up", 0.5)), 2),
                                "conf": round(float(ml_prediction.get("confidence", 0.0)), 2),
                            }

                        event_data["ai_payload"] = ai_payload

                    except Exception as e:
                        logging.error(
                            "ERRO_BUILD_COMPACT: %s",
                            e,
                            exc_info=True,
                        )

                    analysis_result = bot.ai_analyzer.analyze(event_data)

                    if analysis_result and not bot.should_stop:
                        try:
                            final_structured = analysis_result.get("structured") or {}
                            analysis_status = analysis_result.get("status", "unknown")
                            analysis_success = bool(analysis_result.get("success"))
                            analysis_fallback = bool(analysis_result.get("is_fallback"))
                            _print_ai_report_json(final_structured)
                            if analysis_success:
                                logging.info("✅ Análise da IA concluída com JSON válido")
                            else:
                                logging.warning(
                                    "Análise da IA concluída com fallback estruturado | reason=%s",
                                    analysis_result.get("fallback_reason") or "unknown",
                                )

                            try:
                                slog.info(
                                    "ai_analysis_done",
                                    status=analysis_status,
                                    success=analysis_success,
                                    fallback=analysis_fallback,
                                    fallback_reason=analysis_result.get("fallback_reason"),
                                    tipo_evento=event_data.get("tipo_evento"),
                                    resultado_da_batalha=event_data.get("resultado_da_batalha"),
                                )
                            except Exception:
                                pass

                            # [AI_EVENT_SAVE] Salva evento de anÃ¡lise da IA
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

                                ai_result_json = analysis_result.get("structured") or {}

                                # [HYBRID_DECISION]
                                if HYBRID_AVAILABLE and getattr(config, "HYBRID_ENABLED", True) and fuse_decisions and decision_to_ai_result:
                                    try:
                                        ml_pred = ml_prediction if ml_prediction.get("status") == "ok" else None
                                        hybrid_result = fuse_decisions(ml_pred, ai_result_json)
                                        ai_result_json = decision_to_ai_result(hybrid_result)

                                        logging.info(
                                            "Hybrid final decision: %s (conf=%s, source=%s)",
                                            (hybrid_result.action or "wait").upper(),
                                            f"{(hybrid_result.confidence or 0):.0%}",
                                            hybrid_result.source or "unknown",
                                        )

                                        try:
                                            slog.info(
                                                "ai_hybrid_decision",
                                                action=hybrid_result.action,
                                                confidence=float(hybrid_result.confidence),
                                                source=hybrid_result.source,
                                                llm_fallback=bool(getattr(hybrid_result, "llm_is_fallback", False)),
                                            )
                                        except Exception:
                                            pass

                                    except Exception as e:
                                        logging.warning(
                                            f"âš ï¸ Erro na fusÃ£o hÃ­brida, usando IA pura: {e}"
                                        )

                                # Filtro de confianÃ§a
                                if isinstance(ai_result_json, dict):
                                    action = ai_result_json.get("action", "wait")
                                    confidence = ai_result_json.get("confidence", 0.0)
                                    if confidence < 0.7:
                                        ai_result_json["action"] = "wait"

                                # Usar o payload otimizado (formato compacto) ao salvar o evento
                                # O ai_payload jÃ¡ foi otimizado pelo AIPayloadOptimizer.optimize() acima
                                ai_event = {
                                    "tipo_evento": "AI_ANALYSIS",
                                    "symbol": symbol,
                                    "timestamp_ms": anchor_ts_ms,
                                    "anchor_price": anchor_price,
                                    "anchor_window_id": anchor_window_id,
                                    "ai_result": ai_result_json,
                                    "ai_payload": ai_payload,  # Usa o payload jÃ¡ otimizado
                                }

                                if hasattr(bot, "event_saver") and bot.event_saver:
                                    bot.event_saver.save_event(ai_event)

                            except Exception as e:
                                logging.debug(
                                    f"Falha ao salvar evento de anÃ¡lise da IA: {e}",
                                    exc_info=True,
                                )

                        except Exception as e:
                            logging.error(
                                f"âŒ Erro ao processar resposta da IA: {e}",
                                exc_info=True,
                            )

            except Exception as e:
                logging.error(
                    f"âŒ Erro na thread de anÃ¡lise da IA: {e}",
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

    logging.debug("ðŸ”§ Criando thread para anÃ¡lise da IA...")
    t = threading.Thread(target=ai_worker, daemon=True)

    with bot._ai_pool_lock:
        bot.ai_thread_pool = [th for th in bot.ai_thread_pool if th.is_alive()]

        if len(bot.ai_thread_pool) >= bot.max_ai_threads:
            logging.warning("âš ï¸ Thread pool da IA cheio, aguardando...")
            bot.ai_thread_pool[0].join(timeout=5.0)
            bot.ai_thread_pool = [
                th for th in bot.ai_thread_pool if th.is_alive()
            ]

        bot.ai_thread_pool.append(t)

    t.start()

