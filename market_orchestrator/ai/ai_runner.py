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
from trading.export_signals import create_chart_signal_from_event, export_signal_to_csv
from common.format_utils import (
    format_price,
    format_large_number,
    format_delta,
)

from orderbook_core.structured_logging import StructuredLogger
from orderbook_core.tracing_utils import TracerWrapper

# [BUILD_COMPACT] Payload compacto direto do event_data
from build_compact_payload import build_compact_payload

# [THROTTLE] Controle de frequencia de chamadas IA (v3 singleton)
try:
    from common.ai_throttler import get_throttler
    _ai_throttler = get_throttler(
        min_interval=180,
        hard_min_interval=60,
        daily_token_budget=85_000,
        max_calls_per_hour=10,
    )
except ImportError:
    _ai_throttler = None

# [DEDUP] Deduplicador de eventos
from .raw_event_deduplicator import deduplicate_event

# [HYBRID_DECISION] Importa módulo de decisão híbrida
try:
    from ml.hybrid_decision import (
        fuse_decisions,
        decision_to_ai_result,
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
                logging.info("AI init: tentando inicializar AI Analyzer...")

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
                    logging.info("ML engine: XGBoost inicializado")

                    # Teste rápido do ML Engine
                    test_result = bot.ml_engine.predict({
                        "delta": 0.5,
                        "volume_total": 10000,
                        "fluxo_continuo": {"microstructure": {"tick_rule_sum": 0.2}}
                    })

                    _ml_status = test_result.get("status", "unknown")
                    if _ml_status == "ok":
                        logging.info(f"ML engine testado: {test_result.get('prob_up', 0):.1%}")
                    elif _ml_status == "hybrid_disabled":
                        logging.info("ML Engine: hybrid mode desabilitado (usando LLM only)")
                    else:
                        logging.warning(f"⚠️ ML Engine teste falhou: {_ml_status}")

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

                # ── Health check sem consumir tokens ──────────────
                # Antigo: enviava "Teste de Conexão" ao Groq,
                # gastava tokens e frequentemente tomava 429.
                # Agora: apenas verifica se API key existe e o
                # analyzer carregou. A primeira análise real
                # serve como teste efetivo.
                _api_key = getattr(config, "GROQ_API_KEY", None)
                if bot.ai_analyzer and _api_key:
                    bot.ai_test_passed = True
                    logging.info(
                        "AI configurada (API key presente). "
                        "Será testada na primeira análise real."
                    )
                    try:
                        slog.info("ai_init_ready", mode="deferred_test")
                    except Exception:
                        pass
                else:
                    bot.ai_test_passed = False
                    logging.warning(
                        "AI: analyzer=%s, api_key=%s — análise desabilitada",
                        bool(bot.ai_analyzer),
                        bool(_api_key),
                    )

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
                    # [INTELIGÊNCIA HÍBRIDA] Inferência Quantitativa
                    # ============================================
                    # Injetar features do WindowState (fonte única de verdade)
                    # para que o ML engine encontre bb_width, rsi, etc.
                    # diretamente na raiz do event_data
                    try:
                        from core.state_manager import StateManager
                        _ws = StateManager.instance().current
                        if _ws is not None:
                            _ws_feats = _ws.get_ml_features()
                            for _k, _v in _ws_feats.items():
                                if _k not in event_data or event_data.get(_k) in (None, 0, 0.0):
                                    event_data[_k] = _v
                            logging.debug(
                                "WindowState ML features injected: price=%.2f bb_w=%.6f rsi=%.1f",
                                _ws_feats.get('price_close', 0),
                                _ws_feats.get('bb_width', 0),
                                _ws_feats.get('rsi', 0),
                            )
                    except Exception as _ws_err:
                        logging.debug("WindowState inject falhou (nao-critico): %s", _ws_err)

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

                            elif ml_prediction.get("status") == "hybrid_disabled":
                                logging.debug("ML skip: hybrid_disabled")
                            else:
                                logging.warning(f"⚠️ ML Engine retornou status: {ml_prediction.get('status')}")
                        except Exception as e:
                            logging.error(f"❌ Erro na inferência ML: {e}", exc_info=True)
                            ml_prediction = {"status": "error", "msg": str(e)}
                    else:
                        logging.debug("🤖 ML Engine não disponível - usando apenas IA Generativa")

                    # ── Coleta contínua de dados para retreino (independe de ML ativo) ──
                    try:
                        from ml.dataset_collector import get_dataset_collector
                        _price = float(
                            event_data.get("preco_fechamento")
                            or event_data.get("enriched_snapshot", {}).get("ohlc", {}).get("close", 0)
                            or 0
                        )
                        if _price > 0:
                            _ml_feats = event_data.get("ml_features", {})
                            _flat_feats = {}
                            for _sec, _vals in _ml_feats.items():
                                if isinstance(_vals, dict):
                                    for _k, _v in _vals.items():
                                        if isinstance(_v, (int, float)):
                                            _flat_feats[f"{_sec}.{_k}"] = _v
                            get_dataset_collector().collect_window(
                                features=_flat_feats,
                                price_close=_price,
                            )
                    except Exception as _dc_err:
                        logging.debug("DatasetCollector: %s", _dc_err)

                    # FIX 5B: Overlay real window-level returns into ml_features.price_features.
                    # common/ml_features.py computes returns from individual ticks (~1e-7 scale),
                    # but the XGBoost model uses window closes (~1e-4 scale).
                    # Copy the real features from feature_calc to keep the event consistent.
                    try:
                        if hasattr(bot, 'feature_calc') and bot.feature_calc.history_count >= 2:
                            _real_fc = bot.feature_calc.compute()
                            _pf = event_data.setdefault("ml_features", {}).setdefault("price_features", {})
                            _r1 = _real_fc.get("return_1")
                            if _r1 is not None:
                                _pf["returns_1"] = _r1
                            _r5 = _real_fc.get("return_5")
                            if _r5 is not None and _r5 != 0:
                                _pf["returns_5"] = _r5
                            _r10 = _real_fc.get("return_10")
                            if _r10 is not None and _r10 != 0:
                                _pf["returns_10"] = _r10
                    except Exception:
                        pass

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
                    # NOTA: deep_copy=True para não corromper o evento original.
                    # Não fazemos clear()+update() pois build_compact_payload
                    # lê direto do event_data original — dedup é só para o LLM.
                    try:
                        _deduped = deduplicate_event(event_data, deep_copy=True)
                        logging.debug(
                            "RAW_EVENT_DEDUP: deduplicacao aplicada, keys=%d",
                            len(_deduped),
                        )
                    except Exception as _dedup_err:
                        logging.warning("Dedup falhou (não-crítico): %s", _dedup_err)
                        _deduped = event_data

                    # [BUILD_COMPACT] Payload compacto direto do event_data
                    try:
                        ai_payload = build_compact_payload(event_data)

                        # Preencher quant model com ML prediction real
                        if ml_prediction.get("status") == "ok":
                            ai_payload["quant"] = {
                                "pu": round(float(ml_prediction.get("prob_up", 0.5)), 2),
                                "c": round(float(ml_prediction.get("confidence", 0.0)), 2),
                            }

                        event_data["ai_payload"] = ai_payload

                    except Exception as e:
                        logging.error(
                            "ERRO_BUILD_COMPACT: %s",
                            e,
                            exc_info=True,
                        )
                        # Payload de emergência para que guardrail não falhe
                        price = (
                            event_data.get("preco_fechamento")
                            or event_data.get("contextual_snapshot", {}).get("ohlc", {}).get("close")
                            or 0
                        )
                        epoch = event_data.get("epoch_ms") or int(time.time() * 1000)
                        emergency_payload = {
                            "symbol": event_data.get("symbol", "BTCUSDT"),
                            "trigger": "EMERGENCY",
                            "price": {"c": price},
                            "epoch_ms": epoch,
                            "_emergency": True,
                        }
                        event_data["ai_payload"] = emergency_payload
                        logging.warning(
                            "EMERGENCY_PAYLOAD: build_compact falhou, usando payload mínimo (price=%s)",
                            price,
                        )

                    # [THROTTLE] Verificar se vale a pena chamar a IA (v3)
                    _throttled = False
                    if _ai_throttler is not None:
                        _should = _ai_throttler.should_call_ai(
                            event_type=event_data.get("tipo_evento", "ANALYSIS_TRIGGER"),
                            delta=event_data.get("delta", 0.0),
                            volume=event_data.get("volume", 0.0),
                            avg_volume=event_data.get("avg_volume", 10.0),
                            window_count=event_data.get("window_count", 99),
                        )
                        if not _should:
                            _throttled = True

                    if _throttled:
                        analysis_result = None
                        logging.info(
                            "IA nao chamada (throttled) | status=%s",
                            _ai_throttler.get_status() if _ai_throttler else "N/A",
                        )
                    else:
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

                                ai_result_json = analysis_result.get("structured") or {}

                                # [HYBRID_DECISION]
                                if HYBRID_AVAILABLE and getattr(config, "HYBRID_ENABLED", True) and fuse_decisions and decision_to_ai_result:
                                    try:
                                        ml_pred = ml_prediction if ml_prediction.get("status") == "ok" else None
                                        # Obter bias_info do monitor
                                        _bias_info = None
                                        try:
                                            from ml.bias_monitor import get_bias_monitor
                                            _bias_info = get_bias_monitor().get_confidence_adjustment()
                                        except Exception:
                                            pass
                                        hybrid_result = fuse_decisions(ml_pred, ai_result_json, bias_info=_bias_info)
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

