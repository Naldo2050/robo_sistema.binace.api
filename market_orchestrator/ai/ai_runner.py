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
from typing import Any, Dict

import config
from ai_analyzer_qwen import AIAnalyzer  # arquivo na raiz do projeto
from ..utils.price_fetcher import get_current_price
from format_utils import (
    format_price,
    format_large_number,
    format_delta,
)


def initialize_ai_async(bot) -> None:
    """
    Inicializa a IA em uma thread separada, exatamente como no método
    EnhancedMarketBot._initialize_ai_async original.
    """

    def ai_init_worker() -> None:
        try:
            with bot._ai_init_lock:
                if bot.ai_initialization_attempted:
                    return
                bot.ai_initialization_attempted = True

            logging.info("=" * 30 + " INICIALIZANDO IA " + "=" * 30)
            logging.info("🧠 Tentando inicializar AI Analyzer...")

            bot.ai_analyzer = AIAnalyzer()

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

                analysis_result = bot.ai_analyzer.analyze(event_data)

                if analysis_result and not bot.should_stop:
                    try:
                        raw_response = analysis_result.get(
                            "raw_response", ""
                        )
                        _print_ai_report_clean(raw_response)
                        logging.info("✅ Análise da IA concluída com sucesso")
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