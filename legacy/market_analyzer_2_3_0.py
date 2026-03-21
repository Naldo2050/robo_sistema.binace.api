# market_analyzer.py v2.3.0 — SUPER-CORRECTED (ASYNC ORDERBOOK SAFE)
# DEPRECATED: use DataPipeline (data_pipeline.py) + EnhancedMarketBot (market_orchestrator.py)
"""
DEPRECATED — ESTE MÓDULO ESTÁ EM MODO LEGADO.

Substituição recomendada:

  • Validação / enrich / contexto / ML:
      - data_pipeline.DataPipeline  (v3.2.1)
  • Orquestração de streaming + janelas + sinais:
      - market_orchestrator.EnhancedMarketBot
  • Entry point da aplicação:
      - main.py

Este arquivo permanece apenas para compatibilidade temporária com código legado.
Toda lógica nova deve usar o DataPipeline + EnhancedMarketBot.

----------------------------------------------------------------------
DESCRIÇÃO ORIGINAL

Market Analyzer com integração COMPLETA, validação robusta e precisão máxima.

📌 CORREÇÕES v2.3.0:
  ✅ Tolerância correta (1e-8 BTC em vez de 0.001 BTC)
  ✅ Validação de timestamps em clusters do heatmap
  ✅ Correção automática de age_ms negativo
  ✅ Validação de first_seen <= last_seen
  ✅ Logs informativos sem poluir console
  ✅ Integração com data_validator para validação final
  ✅ Precisão máxima em TODOS os cálculos
  ✅ Contadores de correções detalhados
  ✅ Integração segura com OrderBookAnalyzer assíncrono (loop dedicado ou externo)

📌 Componentes integrados:
  • RobustConnectionManager  — WebSocket robusto
  • EnhancedMarketAnalyzer   — análise e validação completa
  • Validação de timestamps  — previne erros críticos
  • Data validator           — validação final de eventos

📌 Nota:
  Este módulo já NÃO define o EnhancedMarketBot. O bot principal está em:
    - market_orchestrator.EnhancedMarketBot
    - entry point: main.py

Autor: Sistema de Trading Institucional
Data: 2025-01-11
Versão: 2.3.0 (LEGADO / DEPRECATED)
"""

from __future__ import annotations

import logging
import random
import socket
import ssl
import threading
import time
import asyncio
import warnings
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd

# Config e componentes do projeto
import config
from orderbook_analyzer import OrderBookAnalyzer
from flow_analyzer import FlowAnalyzer
from ml_features import generate_ml_features
from monitoring.time_manager import TimeManager

# 🆕 Importa validador com tratamento de erro
try:
    from data_processing.data_validator import DataValidator
    DATA_VALIDATOR_AVAILABLE = True
except ImportError:
    DATA_VALIDATOR_AVAILABLE = False
    logging.warning("⚠️ DataValidator não disponível. Continuando sem validação final.")

# 🆕 Importa IntegrationValidator com tratamento de erro
try:
    from integration_validator import IntegrationValidator
    INTEGRATION_VALIDATOR_AVAILABLE = True
except ImportError:
    INTEGRATION_VALIDATOR_AVAILABLE = False
    logging.warning("⚠️ IntegrationValidator não disponível. Continuando sem validação integrada.")

try:
    import websocket  # type: ignore
except Exception as e:
    raise RuntimeError("O pacote 'websocket-client' é obrigatório. pip install websocket-client") from e

SCHEMA_VERSION = "2.3.0"

logger = logging.getLogger("market_analyzer")

# Emite aviso global de depreciação no import deste módulo
warnings.warn(
    "market_analyzer.EnhancedMarketAnalyzer está DEPRECATED. "
    "Use DataPipeline (data_pipeline.DataPipeline) + "
    "EnhancedMarketBot (market_orchestrator.EnhancedMarketBot).",
    DeprecationWarning,
    stacklevel=2,
)


# ========================================================================
# HELPER FUNCTIONS
# ========================================================================

def _decimal_round(value: float, decimals: int = 8) -> float:
    """
    Arredonda usando Decimal para evitar erros de float.
    
    🆕 CORREÇÃO: Trata None e valores inválidos
    """
    if value is None:
        return 0.0
    
    try:
        d = Decimal(str(value))
        quantize_str = '0.' + '0' * decimals
        return float(d.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP))
    except (ValueError, TypeError, Exception) as e:
        logger.warning(f"⚠️ Erro ao arredondar {value}: {e}. Retornando 0.0")
        return 0.0


def _validate_volume_consistency(
    volume_total: float,
    volume_compra: float,
    volume_venda: float,
    window_id: str = "UNKNOWN"
) -> Tuple[bool, float]:
    """
    Valida consistência de volumes e retorna flag + volume corrigido.
    
    🆕 CORREÇÃO: Tolerância correta (1e-8 BTC em vez de 0.001 BTC)
    """
    try:
        vt = _decimal_round(volume_total, decimals=8)
        vc = _decimal_round(volume_compra, decimals=8)
        vv = _decimal_round(volume_venda, decimals=8)
        
        expected_total = _decimal_round(vc + vv, decimals=8)
        discrepancy = abs(vt - expected_total)
        
        tolerance = 1e-8
        
        if discrepancy > tolerance:
            if discrepancy > 0.0001:
                logger.error(
                    f"🔴 DISCREPÂNCIA SIGNIFICATIVA DE VOLUME (janela {window_id}):\n"
                    f"   volume_compra: {vc:.8f} BTC\n"
                    f"   volume_venda: {vv:.8f} BTC\n"
                    f"   Soma calculada: {expected_total:.8f} BTC\n"
                    f"   Total reportado: {vt:.8f} BTC\n"
                    f"   DIFERENÇA: {discrepancy:.8f} BTC\n"
                    f"   ---\n"
                    f"   ✅ Usando soma calculada (mais confiável)"
                )
            else:
                logger.debug(
                    f"⚠️ Pequena discrepância de volume (janela {window_id}): "
                    f"{discrepancy:.8f} BTC. Corrigindo silenciosamente."
                )
            
            return False, expected_total
        
        return True, vt
        
    except Exception as e:
        logger.error(f"Erro ao validar volumes: {e}")
        return False, _decimal_round(volume_compra + volume_venda, decimals=8)


def _validate_and_fix_cluster_timestamps(
    cluster: Dict[str, Any],
    reference_ts_ms: int,
    window_id: str = "UNKNOWN"
) -> Dict[str, Any]:
    """
    🆕 Valida e corrige timestamps em clusters do heatmap.
    
    Garante:
    - first_seen_ms <= last_seen_ms
    - age_ms >= 0
    - Timestamps positivos
    """
    try:
        first_seen = cluster.get('first_seen_ms')
        last_seen = cluster.get('last_seen_ms')
        age_ms = cluster.get('age_ms')
        
        if first_seen is not None and last_seen is not None:
            first = int(first_seen)
            last = int(last_seen)
            
            if first <= 0 or last <= 0:
                logger.warning(
                    f"⚠️ Cluster com timestamps inválidos (janela {window_id}): "
                    f"first={first}, last={last}. Usando reference."
                )
                cluster['first_seen_ms'] = reference_ts_ms
                cluster['last_seen_ms'] = reference_ts_ms
            
            elif first > last:
                logger.warning(
                    f"⚠️ Cluster com timestamps invertidos (janela {window_id}): "
                    f"first_seen ({first}) > last_seen ({last}). Invertendo."
                )
                cluster['first_seen_ms'] = last
                cluster['last_seen_ms'] = first
        
        if age_ms is not None:
            age = int(age_ms)
            
            if age < 0:
                last_seen = cluster.get('last_seen_ms')
                if last_seen and reference_ts_ms:
                    recalculated_age = reference_ts_ms - int(last_seen)
                    if recalculated_age >= 0:
                        cluster['age_ms'] = recalculated_age
                        logger.debug(
                            f"✅ age_ms corrigido em cluster (janela {window_id}): "
                            f"{age} → {recalculated_age}"
                        )
                    else:
                        cluster['age_ms'] = 0
                        logger.warning(
                            f"⚠️ age_ms negativo irrecuperável (janela {window_id}). "
                            f"Zerado."
                        )
                else:
                    cluster['age_ms'] = 0
        
        return cluster
        
    except Exception as e:
        logger.error(f"Erro ao validar timestamps de cluster: {e}")
        return cluster


# ========================================================================
# ROBUST CONNECTION MANAGER (COM RECONEXÃO MELHORADA)
# ========================================================================

class RobustConnectionManager:
    """
    Gerenciador robusto de conexão WebSocket com reconexão automática.
    
    AVISO: Esta implementação está em modo legado neste módulo.
    O RobustConnectionManager atualmente em uso pelo bot está em:
        - market_orchestrator.RobustConnectionManager
    """
    def __init__(
        self,
        stream_url: str,
        symbol: str,
        max_reconnect_attempts: int = 15,
        initial_delay: float = 1.0,
        max_delay: float = 120.0,
        backoff_factor: float = 2.0,
        heartbeat_interval: int = 30,
        heartbeat_timeout: int = 120,
    ) -> None:
        self.stream_url = stream_url
        self.symbol = symbol
        self.max_reconnect_attempts = max_reconnect_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout

        self.current_delay = initial_delay
        self.reconnect_count = 0
        self.is_connected = False
        self.last_message_time: Optional[datetime] = None
        self.connection_start_time: Optional[datetime] = None

        self.heartbeat_thread: Optional[threading.Thread] = None
        self.should_stop = False

        self.on_message_callback = None
        self.on_open_callback = None
        self.on_close_callback = None
        self.on_error_callback = None

        self.total_messages_received = 0
        self.total_reconnects = 0
        self.total_errors = 0

        self.ws: Optional[websocket.WebSocketApp] = None

        logger.info(
            "🔌 [LEGACY] ConnectionManager v2.3.0 inicializado: %s | max_reconnects=%d | backoff %.1f..%.1fs",
            symbol, max_reconnect_attempts, initial_delay, max_delay
        )

    def set_callbacks(self, on_message=None, on_open=None, on_close=None, on_error=None) -> None:
        self.on_message_callback = on_message
        self.on_open_callback = on_open
        self.on_close_callback = on_close
        self.on_error_callback = on_error

    def _test_connection(self) -> bool:
        """Testa host/porta antes de abrir o WebSocket."""
        try:
            parsed_url = urlparse(self.stream_url)
            host = parsed_url.hostname
            port = parsed_url.port or (443 if parsed_url.scheme == "wss" else 80)
            if not host:
                return False

            logger.debug("🔍 Testando conectividade: %s:%s", host, port)

            with socket.create_connection((host, port), timeout=5) as sock:
                if parsed_url.scheme == "wss":
                    context = ssl.create_default_context()
                    with context.wrap_socket(sock, server_hostname=host):
                        return True
                return True

        except socket.timeout:
            logger.error("❌ Timeout ao testar conexão")
            return False
        except OSError as e:
            logger.error("❌ Erro de socket: %s", e)
            return False
        except Exception as e:
            logger.error("❌ Erro ao testar conexão: %s", e)
            return False

    def _on_message(self, ws, message):
        """Handler interno para delegar ao callback e atualizar heartbeat."""
        try:
            self.last_message_time = datetime.now(timezone.utc)
            self.total_messages_received += 1
            if self.on_message_callback:
                self.on_message_callback(ws, message)
        except Exception as e:
            logger.error("❌ Erro no processamento da mensagem: %s", e, exc_info=True)

    def _on_open(self, ws):
        self.is_connected = True
        self.reconnect_count = 0
        self.current_delay = self.initial_delay
        self.connection_start_time = datetime.now(timezone.utc)
        self.last_message_time = self.connection_start_time

        logger.info("✅ Conexão estabelecida com %s (tentativa=%d)", self.symbol, self.total_reconnects + 1)
        self._start_monitoring_threads()
        if self.on_open_callback:
            self.on_open_callback(ws)

    def _on_close(self, ws, close_status_code, close_msg):
        self.is_connected = False
        logger.warning("🔌 Conexão fechada: code=%s, msg=%s", close_status_code, close_msg)
        self._stop_monitoring_threads()
        if self.on_close_callback:
            self.on_close_callback(ws, close_status_code, close_msg)

    def _on_error(self, ws, error):
        self.total_errors += 1
        logger.error("❌ Erro WebSocket: %s", error)
        if self.on_error_callback:
            self.on_error_callback(ws, error)

    def _start_monitoring_threads(self) -> None:
        self.should_stop = False
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_monitor, daemon=True)
        self.heartbeat_thread.start()

    def _stop_monitoring_threads(self) -> None:
        self.should_stop = True
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=1.0)

    def _heartbeat_monitor(self) -> None:
        while not self.should_stop and self.is_connected:
            time.sleep(self.heartbeat_interval)
            if not self.last_message_time:
                continue
            time_since_last = (datetime.now(timezone.utc) - self.last_message_time).total_seconds()
            if time_since_last > self.heartbeat_timeout:
                logger.warning("⚠️ Stale connection: %.0fs sem mensagens — fechando socket p/ reconectar", time_since_last)
                try:
                    if self.ws:
                        self.ws.close()
                except Exception:
                    pass
                self.is_connected = False
                break

    def connect(self) -> None:
        """Tenta conectar com retry/backoff até should_stop ou atingir o limite."""
        while self.reconnect_count < self.max_reconnect_attempts and not self.should_stop:
            try:
                if not self._test_connection():
                    raise ConnectionError("Falha no teste de conectividade")

                logger.info(
                    "🔄 Tentativa %d/%d | delay atual: %.1fs",
                    self.reconnect_count + 1, self.max_reconnect_attempts, self.current_delay
                )

                self.ws = websocket.WebSocketApp(
                    self.stream_url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_open,
                )

                self.ws.run_forever(ping_interval=self.heartbeat_interval, ping_timeout=10)

                if self.should_stop:
                    break

            except KeyboardInterrupt:
                logger.info("⏹️ Interrompido pelo usuário")
                self.should_stop = True
                break

            except Exception as e:
                self.reconnect_count += 1
                self.total_reconnects += 1
                logger.error("❌ Erro na conexão (%d/%d): %s", self.reconnect_count, self.max_reconnect_attempts, e)

                if self.reconnect_count < self.max_reconnect_attempts and not self.should_stop:
                    jitter = random.uniform(0, 0.1 * self.current_delay)
                    sleep_time = min(self.current_delay + jitter, self.max_delay)
                    logger.info("⏳ Aguardando %.1fs antes de reconectar...", sleep_time)
                    time.sleep(sleep_time)
                    self.current_delay = min(self.current_delay * self.backoff_factor, self.max_delay)

        if self.reconnect_count >= self.max_reconnect_attempts:
            logger.critical("💀 Falha após %d tentativas. Verifique a rede/stream.", self.max_reconnect_attempts)

        self._stop_monitoring_threads()

    def disconnect(self) -> None:
        """Sinaliza parada e fecha o socket para destravar o run_forever."""
        logger.info("🛑 Desconectando...")
        self.should_stop = True
        try:
            if self.ws:
                self.ws.close()
        except Exception:
            pass

    def get_stats(self) -> Dict[str, Any]:
        uptime = None
        if self.connection_start_time:
            uptime = (datetime.now(timezone.utc) - self.connection_start_time).total_seconds()
        return {
            "is_connected": self.is_connected,
            "total_messages": self.total_messages_received,
            "total_reconnects": self.total_reconnects,
            "total_errors": self.total_errors,
            "uptime_seconds": uptime,
            "reconnect_count": self.reconnect_count,
            "current_delay": self.current_delay,
        }


# ========================================================================
# ENHANCED MARKET ANALYZER (LEGACY)
# ========================================================================

@dataclass
class AnalyzerStats:
    total_windows: int = 0
    valid_events: int = 0
    invalid_events: int = 0
    volume_corrections: int = 0
    timestamp_corrections: int = 0  # 🆕


class EnhancedMarketAnalyzer:
    """
    [LEGACY / DEPRECATED]
    
    Analisador de mercado COMPLETO com validação robusta v2.3.0.
    
    AVISO IMPORTANTE:
      - Esta classe está em modo LEGADO.
      - A arquitetura oficial agora é:
          • DataPipeline (data_pipeline.py) para validação/enrich/ML
          • EnhancedMarketBot (market_orchestrator.py) para orquestração
      - Esta classe permanece apenas para compatibilidade com código antigo.
    
    🆕 CORREÇÕES v2.3.0 (mantidas para compatibilidade):
      - Validação de timestamps em clusters do heatmap
      - Correção automática de age_ms negativo
      - Validação de first_seen <= last_seen
      - Tolerância correta (1e-8 BTC)
      - Integração com data_validator
      - Logs informativos sem poluir console
      - Integração segura com OrderBookAnalyzer assíncrono
        • Pode usar loop dedicado interno
        • OU reutilizar um loop externo (ex.: do EnhancedMarketBot)
    """

    def __init__(
        self,
        symbol: str,
        time_manager: Optional[TimeManager] = None,
        flow_analyzer: Optional[FlowAnalyzer] = None,
        orderbook_analyzer: Optional[OrderBookAnalyzer] = None,
        async_loop: Optional[asyncio.AbstractEventLoop] = None,
        validator: Optional[Any] = None,      # 🆕 Tipo flexível
        data_validator: Optional[Any] = None, # 🆕 Tipo flexível
    ) -> None:
        # Aviso de depreciação no uso da classe
        logger.warning(
            "DEPRECATION: EnhancedMarketAnalyzer é legado. "
            "Toda lógica nova deve usar DataPipeline + EnhancedMarketBot. "
            "Este componente será removido em futura versão."
        )

        self.symbol = symbol
        self.time_manager = time_manager or TimeManager()
        self.flow_analyzer = flow_analyzer or FlowAnalyzer(time_manager=self.time_manager)

        # Controle de propriedade (evita fechar recursos externos)
        self._external_orderbook = orderbook_analyzer is not None
        self.orderbook_analyzer = orderbook_analyzer or OrderBookAnalyzer(
            symbol=symbol,
            time_manager=self.time_manager,
            cache_ttl_seconds=1.0,
            max_stale_seconds=30.0,
            rate_limit_threshold=10,
        )
        self._owns_orderbook_analyzer = not self._external_orderbook

        # 🆕 Inicialização condicional dos validadores
        if INTEGRATION_VALIDATOR_AVAILABLE:
            self.validator = validator or IntegrationValidator()
        else:
            self.validator = None
            
        if DATA_VALIDATOR_AVAILABLE:
            self.data_validator = data_validator or DataValidator()
        else:
            self.data_validator = None

        # 🆕 Loop asyncio:
        # - Se async_loop for fornecido, reutiliza (ex.: loop do EnhancedMarketBot)
        # - Caso contrário, cria loop próprio + thread
        if async_loop is not None:
            self._async_loop = async_loop
            self._async_loop_thread: Optional[threading.Thread] = None
            self._owns_loop = False
            logger.info(
                "🔁 [LEGACY] EnhancedMarketAnalyzer reutilizando event loop externo "
                "para OrderBookAnalyzer (symbol=%s)", symbol
            )

            if self._external_orderbook:
                logger.info(
                    "🔁 [LEGACY] EnhancedMarketAnalyzer reutilizando OrderBookAnalyzer externo "
                    "(symbol=%s)", symbol
                )
        else:
            self._async_loop = asyncio.new_event_loop()
            self._async_loop_thread = threading.Thread(
                target=self._run_async_loop,
                name=f"ema_orderbook_loop_{symbol}",
                daemon=True,
            )
            self._async_loop_thread.start()
            self._owns_loop = True
            logger.info(
                "🧵 [LEGACY] EnhancedMarketAnalyzer criou loop assíncrono dedicado "
                "para OrderBookAnalyzer (symbol=%s)", symbol
            )

        # Aviso de uso potencialmente perigoso:
        if self._external_orderbook and async_loop is None:
            logger.warning(
                "⚠️ [LEGACY] EnhancedMarketAnalyzer recebeu um OrderBookAnalyzer EXTERNO "
                "sem async_loop. Será criado um event loop próprio.\n"
                "   Se esse OrderBookAnalyzer for compartilhado com outro componente "
                "(ex.: EnhancedMarketBot), injete o MESMO event loop para evitar "
                "várias loops usando a mesma sessão HTTP."
            )

        self.stats = AnalyzerStats()
        self.last_event: Optional[Dict[str, Any]] = None

        logger.info("=" * 72)
        logger.info("⚠️ EnhancedMarketAnalyzer v%s inicializado em modo LEGADO", SCHEMA_VERSION)
        logger.info("   Symbol:           %s", symbol)
        logger.info("   Schema Version:   %s", SCHEMA_VERSION)
        logger.info("   Components:       FlowAnalyzer, OrderBook, ML (LEGACY PATH)")
        if self.validator:
            logger.info("   Integration Validator: ATIVO (LEGACY)")
        if self.data_validator:
            logger.info("   Data Validator:   ATIVO (LEGACY)")
        logger.info("   Features:         Volume Validation, Timestamp Validation, Precision Control")
        if self._owns_loop:
            logger.info("   Async Loop:       loop dedicado interno (LEGACY)")
        else:
            logger.info("   Async Loop:       loop externo reutilizado")
        if self._owns_orderbook_analyzer:
            logger.info("   OrderBook:        instância própria (LEGACY)")
        else:
            logger.info("   OrderBook:        instância externa reutilizada")
        logger.info("   AVISO:            PREFIRA DataPipeline + EnhancedMarketBot")
        logger.info("=" * 72)

    # ---------------- helpers ----------------

    def _run_async_loop(self) -> None:
        """
        Loop de evento dedicado ao OrderBookAnalyzer.

        Roda em uma thread separada e permite que as corotinas do
        OrderBookAnalyzer sejam executadas via asyncio.run_coroutine_threadsafe
        sem recriar event loops.
        """
        asyncio.set_event_loop(self._async_loop)
        try:
            self._async_loop.run_forever()
        finally:
            # Encerramento gracioso (similar ao EnhancedMarketBot)
            try:
                try:
                    pending = asyncio.all_tasks()
                except TypeError:
                    pending = asyncio.Task.all_tasks()
            except Exception:
                pending = []

            for task in pending:
                task.cancel()

            if pending:
                try:
                    self._async_loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                except Exception:
                    pass

            try:
                self._async_loop.run_until_complete(
                    self._async_loop.shutdown_asyncgens()
                )
            except Exception:
                pass

            self._async_loop.close()

    def _run_orderbook_analyze(
        self,
        event_epoch_ms: int,
        window_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Executa OrderBookAnalyzer.analyze() no loop asyncio associado.

        - Se o EnhancedMarketAnalyzer criou um loop próprio, usa esse loop.
        - Se recebeu um loop externo (ex.: do EnhancedMarketBot), reutiliza esse loop.
        """
        if event_epoch_ms <= 0:
            logger.error("❌ event_epoch_ms inválido: %s", event_epoch_ms)
            return None

        loop = getattr(self, "_async_loop", None)
        if loop is None:
            logger.error("❌ Loop assíncrono do OrderBookAnalyzer não inicializado")
            return None
        if loop.is_closed():
            logger.error("❌ Loop assíncrono do OrderBookAnalyzer já foi fechado")
            return None
        if not loop.is_running() and not self._owns_loop:
            # Loop externo existe, mas aparentemente não está rodando
            logger.error(
                "❌ Loop assíncrono externo não está em execução "
                "(symbol=%s, window_id=%s)", self.symbol, window_id
            )
            return None

        try:
            coro = self.orderbook_analyzer.analyze(
                current_snapshot=None,
                event_epoch_ms=event_epoch_ms,
                window_id=window_id,
            )

            future = asyncio.run_coroutine_threadsafe(coro, loop)

            try:
                return future.result(timeout=5.0)
            except FutureTimeoutError:
                logger.error(
                    "⏱️ Timeout ao buscar orderbook (async loop) - "
                    "cancelando coroutine pendente"
                )
                future.cancel()
                return None

        except Exception as e:
            logger.error(
                "❌ Erro ao buscar orderbook (async loop): %s",
                e,
                exc_info=True,
            )
            return None

    @staticmethod
    def _sanitize_trades(trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normaliza e filtra trades inválidos; aceita payload bruto ou {'data': {...}}."""
        clean: List[Dict[str, Any]] = []
        for t in trades or []:
            try:
                if "T" not in t and "data" in t and isinstance(t["data"], dict):
                    t = t["data"]

                T = int(t.get("T", 0))
                p = float(t.get("p", 0) or 0)
                q = float(t.get("q", 0) or 0)
                if T <= 0 or p <= 0 or q <= 0:
                    continue

                m = t.get("m", None)
                if isinstance(m, str):
                    m = m.strip().lower() in {"true", "t", "1", "sell", "ask", "s"}

                clean.append({"T": T, "p": p, "q": q, "m": m})
            except Exception:
                continue
        return clean

    def process_trades(self, trades: List[Dict[str, Any]]) -> None:
        for trade in self._sanitize_trades(trades):
            try:
                self.flow_analyzer.process_trade(trade)
            except Exception as e:
                logger.debug("Erro ao processar trade no FlowAnalyzer: %s", e)

    # ---------------- API principal ----------------

    def analyze_window(self, window_data: List[Dict[str, Any]], window_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        [LEGACY] Analisa uma janela de trades e retorna um evento integrado.
        
        🆕 v2.3.0: Validação completa de timestamps e volumes
        🆕 v2.3.x: Integração segura com OrderBookAnalyzer assíncrono
        
        AVISO:
            Esta função continua funcionando para compatibilidade,
            mas a orquestração oficial de janelas hoje é feita por:
                - market_orchestrator.EnhancedMarketBot
                - data_pipeline.DataPipeline
        """
        self.stats.total_windows += 1

        clean_window = self._sanitize_trades(window_data)
        if len(clean_window) < 2:
            logger.warning("⚠️ [LEGACY] Janela %s inválida: %d trades válidos", window_id, len(clean_window))
            self.stats.invalid_events += 1
            return None

        try:
            last_trade_ts = clean_window[-1]["T"]

            # Atualiza estado do fluxo
            self.process_trades(clean_window)

            # Métricas de fluxo
            flow_metrics = self.flow_analyzer.get_flow_metrics(reference_epoch_ms=last_trade_ts)

            # Orderbook validado (via loop assíncrono associado)
            wid = window_id or "UNKNOWN"
            orderbook_event = self._run_orderbook_analyze(
                event_epoch_ms=last_trade_ts,
                window_id=wid,
            )

            if not orderbook_event or not orderbook_event.get("is_valid", False):
                logger.error(
                    "❌ [LEGACY] Orderbook inválido (janela %s): %s",
                    wid,
                    (orderbook_event or {}).get("erro", "unknown"),
                )
                self.stats.invalid_events += 1
                return None

            orderbook_data = orderbook_event.get("orderbook_data", {}) or {}

            # ML features (Pandas apenas aqui, opcional)
            ml_features: Dict[str, Any] = {}
            if getattr(config, "ENABLE_ML_FEATURES", True):
                df_window = pd.DataFrame(clean_window)
                ml_features = generate_ml_features(
                    df=df_window,
                    orderbook_data=orderbook_data,
                    flow_metrics=flow_metrics,
                    lookback_windows=[1, 5, 15],
                    volume_ma_window=20,
                )

            # ============================================================
            # 🆕 MAPEAMENTO E VALIDAÇÃO DE VOLUMES (v2.3.0)
            # ============================================================
            order_flow_data = flow_metrics.get("order_flow", {})
            
            volume_compra_btc = _decimal_round(order_flow_data.get("buy_volume_btc", 0.0), decimals=8)
            volume_venda_btc = _decimal_round(order_flow_data.get("sell_volume_btc", 0.0), decimals=8)
            volume_total_calculated = _decimal_round(volume_compra_btc + volume_venda_btc, decimals=8)
            reported_total = _decimal_round(order_flow_data.get("total_volume_btc", 0.0), decimals=8)
            
            if reported_total > 0:
                is_consistent, volume_total_btc = _validate_volume_consistency(
                    volume_total=reported_total,
                    volume_compra=volume_compra_btc,
                    volume_venda=volume_venda_btc,
                    window_id=wid,
                )
                if not is_consistent:
                    self.stats.volume_corrections += 1
            else:
                volume_total_btc = volume_total_calculated
            
            volume_compra_usd = _decimal_round(order_flow_data.get("buy_volume", 0.0), decimals=2)
            volume_venda_usd = _decimal_round(order_flow_data.get("sell_volume", 0.0), decimals=2)

            # ============================================================
            # 🆕 VALIDAÇÃO DE TIMESTAMPS EM CLUSTERS (v2.3.0)
            # ============================================================
            liquidity_heatmap = flow_metrics.get("liquidity_heatmap", {})
            if "clusters" in liquidity_heatmap:
                corrected_clusters = []
                for cluster in liquidity_heatmap["clusters"]:
                    original_cluster = dict(cluster) if isinstance(cluster, dict) else cluster
                    corrected_cluster = _validate_and_fix_cluster_timestamps(
                        cluster=cluster,
                        reference_ts_ms=last_trade_ts,
                        window_id=wid,
                    )
                    corrected_clusters.append(corrected_cluster)
                    
                    if corrected_cluster != original_cluster:
                        self.stats.timestamp_corrections += 1
                
                liquidity_heatmap["clusters"] = corrected_clusters
                flow_metrics["liquidity_heatmap"] = liquidity_heatmap

            # ============================================================
            # CRIAÇÃO DO EVENTO (FORMATO LEGADO)
            # ============================================================
            window_duration_ms = clean_window[-1]["T"] - clean_window[0]["T"]
            event: Dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "tipo_evento": "MarketAnalysis",
                "ativo": self.symbol,
                "window_id": window_id,
                
                "time_index": self.time_manager.build_time_index(
                    last_trade_ts, 
                    include_local=True, 
                    timespec="milliseconds"
                ),
                
                "cvd": flow_metrics.get("cvd", 0.0),
                "whale_buy_volume": flow_metrics.get("whale_buy_volume", 0.0),
                "whale_sell_volume": flow_metrics.get("whale_sell_volume", 0.0),
                "whale_delta": flow_metrics.get("whale_delta", 0.0),
                "order_flow": flow_metrics.get("order_flow", {}),
                "tipo_absorcao": flow_metrics.get("tipo_absorcao", "Neutra"),
                "participant_analysis": flow_metrics.get("participant_analysis", {}),
                "bursts": flow_metrics.get("bursts", {}),
                "sector_flow": flow_metrics.get("sector_flow", {}),
                
                "volume_compra": volume_compra_btc,
                "volume_venda": volume_venda_btc,
                "volume_total": volume_total_btc,
                "volume_compra_usd": volume_compra_usd,
                "volume_venda_usd": volume_venda_usd,
                
                "orderbook_data": orderbook_data,
                "orderbook_event": orderbook_event,
                
                "ml_features": ml_features,
                
                "liquidity_heatmap": liquidity_heatmap,
                
                "trades_count": len(clean_window),
                "window_duration_ms": int(window_duration_ms) if window_duration_ms > 0 else 0,
            }

            # ============================================================
            # 🆕 VALIDAÇÃO FINAL COM DATA VALIDATOR (v2.3.0)
            # ============================================================
            if self.data_validator:
                try:
                    validated_event = self.data_validator.validate_and_clean(event)
                    
                    if validated_event is None:
                        logger.error("❌ [LEGACY] Evento rejeitado pelo DataValidator (janela %s)", window_id)
                        self.stats.invalid_events += 1
                        return None
                    
                    event = validated_event
                    
                except Exception as e:
                    logger.error(f"❌ [LEGACY] Erro no DataValidator: {e}. Usando evento sem validação final.")
            else:
                logger.debug("⏭️ [LEGACY] DataValidator não disponível - pulando validação final")

            # Validação integrada (IntegrationValidator)
            if self.validator:
                validation = self.validator.validate_event(event)
                event["validation"] = validation
                event["is_valid"] = bool(validation.get("is_valid", False))
                event["should_skip"] = bool(validation.get("should_skip", False))

                if event["should_skip"]:
                    self.stats.invalid_events += 1
                    logger.error("❌ [LEGACY] EVENTO INVÁLIDO (janela %s): %s", window_id, validation.get("validation_summary"))
                    for issue in validation.get("critical_issues", []):
                        logger.error("   🔴 %s", issue)
                    for issue in validation.get("issues", []):
                        logger.warning("   ⚠️ %s", issue)
                    return None
            else:
                event["is_valid"] = True
                event["should_skip"] = False
                event["validation"] = {
                    "is_valid": True,
                    "should_skip": False,
                    "validation_summary": "Sem validação - aceito por padrão (LEGACY)",
                    "issues": [],
                    "critical_issues": [],
                    "warnings": []
                }

            self.stats.valid_events += 1
            self.last_event = event

            if self.validator:
                for warning in event.get("validation", {}).get("warnings", []):
                    logger.debug("⚡ %s", warning)

            of = event.get("order_flow", {})
            logger.info(
                "✅ [LEGACY] Janela %s válida: %d trades, delta=%.2f, cvd=%.2f",
                window_id, len(clean_window), of.get("net_flow_1m", 0.0), event.get("cvd", 0.0)
            )
            return event

        except Exception as e:
            logger.exception("❌ [LEGACY] Erro ao processar janela %s: %s", window_id, e)
            self.stats.invalid_events += 1
            return None

    # ---------------- métricas/diagnóstico ----------------

    def get_stats(self) -> Dict[str, Any]:
        valid_rate = 100.0 * self.stats.valid_events / max(1, self.stats.total_windows)
        return {
            "total_windows": self.stats.total_windows,
            "valid_events": self.stats.valid_events,
            "invalid_events": self.stats.invalid_events,
            "volume_corrections": self.stats.volume_corrections,
            "timestamp_corrections": self.stats.timestamp_corrections,
            "valid_rate_pct": round(valid_rate, 2),
            "flow_analyzer_stats": self.flow_analyzer.get_stats(),
            "orderbook_analyzer_stats": self.orderbook_analyzer.get_stats(),
            "validator_stats": self.validator.get_stats() if self.validator else {"status": "não disponível (LEGACY)"},
            "data_validator_stats": self.data_validator.get_correction_stats() if self.data_validator else {"status": "não disponível (LEGACY)"},
            "mode": "LEGACY",
        }

    def diagnose(self) -> Dict[str, Any]:
        stats = self.get_stats()
        logger.info("🔍 DIAGNÓSTICO DO MARKET ANALYZER v%s (LEGACY)", SCHEMA_VERSION)
        logger.info("-" * 72)
        logger.info("📊 Janelas: total=%d | válidas=%d (%.2f%%) | inválidas=%d",
                    stats["total_windows"], stats["valid_events"], stats["valid_rate_pct"], stats["invalid_events"])
        logger.info("🔧 Correções de volume: %d", stats["volume_corrections"])
        logger.info("⏰ Correções de timestamp: %d", stats["timestamp_corrections"])
        logger.info("🌊 Flow Analyzer: %s", stats["flow_analyzer_stats"])
        logger.info("📚 OrderBook Analyzer: %s", stats["orderbook_analyzer_stats"])
        if self.validator:
            logger.info("✅ Integration Validator: %s", stats["validator_stats"])
        if self.data_validator:
            logger.info("🛡️ Data Validator: %s", stats["data_validator_stats"])
        logger.info("   AVISO: Este diagnóstico refere-se ao caminho LEGACY.")
        logger.info("-" * 72)
        return stats

    def close(self, timeout: float = 2.0) -> None:
        """
        Fecha recursos assíncronos (ClientSession do OrderBookAnalyzer
        + loop asyncio dedicado, se forem de propriedade desta instância).
        """
        loop = getattr(self, "_async_loop", None)
        if not loop or loop.is_closed():
            return

        # Fecha OrderBookAnalyzer apenas se ele for "nosso"
        try:
            if self._owns_orderbook_analyzer and hasattr(self.orderbook_analyzer, "close"):
                fut = asyncio.run_coroutine_threadsafe(
                    self.orderbook_analyzer.close(),
                    loop,
                )
                try:
                    fut.result(timeout=timeout)
                except FutureTimeoutError:
                    logger.debug(
                        "Timeout ao fechar OrderBookAnalyzer; "
                        "cancelando tarefa pendente"
                    )
                    fut.cancel()
                except Exception:
                    pass
        except Exception as e:
            logger.debug("Falha ao fechar OrderBookAnalyzer: %s", e)

        # Para o loop apenas se ele for interno
        if getattr(self, "_owns_loop", False):
            try:
                loop.call_soon_threadsafe(loop.stop)
            except Exception:
                pass

            try:
                if hasattr(self, "_async_loop_thread") and self._async_loop_thread:
                    self._async_loop_thread.join(timeout=timeout)
            except Exception:
                pass


# ========================================================================
# EXECUÇÃO DIRETA (somente informação / LEGACY)
# ========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info(
        "Este módulo (market_analyzer.py) está DEPRECADO e permanece apenas para compatibilidade.\n"
        "Use DataPipeline (data_pipeline.DataPipeline) + EnhancedMarketBot em market_orchestrator.py.\n"
        "O entry point da aplicação continua sendo main.py."
    )