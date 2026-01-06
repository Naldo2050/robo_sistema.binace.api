# market_orchestrator/connection/robust_connection.py
# -*- coding: utf-8 -*-

"""
Gerenciador robusto de conexão WebSocket usando AIOHTTP (Asyncio).

Substitui a versão antiga baseada em 'websocket-client' (threading).
Gerencia reconexão automática, heartbeats e backoff exponencial.
"""

import logging
import asyncio
import json
import socket
from datetime import datetime, timezone
from typing import Optional, Callable, Any

import aiohttp
from aiohttp import ClientWebSocketResponse, WSMsgType

import config
import time
import threading

from orderbook_core.structured_logging import StructuredLogger
from orderbook_core.tracing_utils import TracerWrapper

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter thread-safe.

    - acquire(): versão síncrona (bloqueia a thread que chamou)
    - acquire_async(): versão assíncrona (para uso opcional em corrotinas)
    """

    def __init__(self, max_calls: int, period_seconds: float) -> None:
        self.max_calls = max_calls
        self.period = period_seconds
        self.timestamps: list[float] = []
        self._lock = threading.Lock()

    def _prune_and_get_sleep(self, now: float) -> float:
        """
        Remove timestamps antigos e calcula quanto tempo precisamos esperar
        para liberar a próxima chamada.
        """
        with self._lock:
            # mantém apenas chamadas dentro da janela
            self.timestamps = [
                t for t in self.timestamps if now - t < self.period
            ]

            if len(self.timestamps) >= self.max_calls:
                sleep_time = self.timestamps[0] + self.period - now
            else:
                sleep_time = 0.0

        return max(0.0, sleep_time)

    def acquire(self) -> None:
        """Versão síncrona: bloqueia a thread até poder prosseguir."""
        while True:
            now = time.monotonic()
            sleep_time = self._prune_and_get_sleep(now)
            if sleep_time <= 0:
                break
            time.sleep(sleep_time)

        with self._lock:
            self.timestamps.append(time.monotonic())

    async def acquire_async(self) -> None:
        """Versão assíncrona opcional (para uso futuro em corrotinas)."""
        while True:
            now = time.monotonic()
            sleep_time = self._prune_and_get_sleep(now)
            if sleep_time <= 0:
                break
            await asyncio.sleep(sleep_time)

        with self._lock:
            self.timestamps.append(time.monotonic())


class RobustConnectionManager:
    """
    Gerenciador de WebSocket assíncrono (aiohttp).
    Mantém a conexão viva e reconecta automaticamente em caso de queda.
    """

    def __init__(
        self,
        stream_url: str,
        symbol: str,
        max_reconnect_attempts: int = 15,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 1.5,
    ) -> None:
        self.stream_url = stream_url
        self.symbol = symbol
        self.max_reconnect_attempts = max_reconnect_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor

        self.current_delay = initial_delay
        self.reconnect_count = 0
        self.is_connected = False
        self.should_stop = False

        # Estatísticas
        self.total_messages_received = 0
        self.total_reconnects = 0
        self.last_message_time: Optional[datetime] = None
        self.connection_start_time: Optional[datetime] = None

        # Callbacks (podem ser async ou sync)
        self.on_message_callback: Optional[Callable] = None
        self.on_open_callback: Optional[Callable] = None
        self.on_close_callback: Optional[Callable] = None
        self.on_error_callback: Optional[Callable] = None
        self.on_reconnect_callback: Optional[Callable] = None
        self.external_heartbeat_cb: Optional[Callable] = None

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[ClientWebSocketResponse] = None

        # Logging estruturado e tracing para a conexão
        self.slog = StructuredLogger("robust_connection", self.symbol)
        self.tracer = TracerWrapper(
            service_name="enhanced_market_bot",
            component="connection",
            symbol=self.symbol,
        )

    def set_callbacks(
        self,
        on_message=None,
        on_open=None,
        on_close=None,
        on_error=None,
        on_reconnect=None,
    ) -> None:
        self.on_message_callback = on_message
        self.on_open_callback = on_open
        self.on_close_callback = on_close
        self.on_error_callback = on_error
        self.on_reconnect_callback = on_reconnect

    def set_heartbeat_cb(self, cb) -> None:
        self.external_heartbeat_cb = cb

    async def connect(self) -> None:
        """
        Inicia o loop de conexão principal.
        Bloqueia a execução enquanto a conexão estiver ativa.
        """
        self.should_stop = False

        # Configurações de timeout e ping/pong
        ping_interval = getattr(config, "WS_PING_INTERVAL", 20)
        ping_timeout = getattr(config, "WS_PING_TIMEOUT", 10)
        connect_timeout = getattr(config, "ORDERBOOK_REQUEST_TIMEOUT", 10.0)
        logger.info(f"🚀 Iniciando conexão Async (aiohttp) para {self.symbol}... (ping_interval={ping_interval}s, ping_timeout={ping_timeout}s)")

        with self.tracer.start_span(
            "ws_connect_loop",
            {
                "symbol": self.symbol,
                "max_reconnect_attempts": self.max_reconnect_attempts,
                "ping_interval": ping_interval,
            },
        ):
            while not self.should_stop:
                try:
                    async with aiohttp.ClientSession() as session:
                        self._session = session

                        # Tentativa de conexão
                        try:
                            try:
                                self.slog.info(
                                    "ws_connect_attempt",
                                    reconnect_count=self.reconnect_count,
                                    current_delay=self.current_delay,
                                )
                            except Exception:
                                pass

                            async with session.ws_connect(
                                self.stream_url,
                                heartbeat=ping_interval,
                                autoping=True,
                                timeout=connect_timeout
                            ) as ws:
                                self._ws = ws
                                await self._handle_connection_success(ws)

                                # Loop de mensagens
                                async for msg in ws:
                                    if self.should_stop:
                                        break
                                    await self._handle_message(msg)

                                # Se saiu do loop, a conexão fechou
                                if not self.should_stop:
                                    logger.warning("🔌 Conexão fechada pelo servidor.")
                                    await self._handle_reconnect()

                        except (aiohttp.ClientError, socket.gaierror) as e:
                            logger.error(f"❌ Erro de conexão/rede: {e}")
                            await self._handle_reconnect()
                        except Exception as e:
                            logger.error(f"❌ Erro inesperado no WebSocket: {e}", exc_info=True)
                            await self._handle_reconnect()

                except Exception as e:
                    logger.critical(f"💀 Erro crítico na sessão aiohttp: {e}")
                    await asyncio.sleep(self.initial_delay)

                if self.reconnect_count >= self.max_reconnect_attempts:
                    logger.critical("⛔ Máximo de tentativas de reconexão atingido. Parando.")
                    try:
                        self.slog.error(
                            "ws_max_reconnect_reached",
                            reconnect_count=self.reconnect_count,
                            max_reconnect_attempts=self.max_reconnect_attempts,
                        )
                    except Exception:
                        pass
                    break

    async def disconnect(self) -> None:
        """Fecha a conexão graciosamente."""
        logger.info("🛑 Desconectando...")
        try:
            self.slog.info("ws_disconnect_called")
        except Exception:
            pass
        self.should_stop = True
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()

    # ============================================================
    # HANDLERS INTERNOS
    # ============================================================

    async def _handle_connection_success(self, ws) -> None:
        """Chamado quando a conexão é estabelecida com sucesso."""
        self.is_connected = True
        self.reconnect_count = 0
        self.current_delay = self.initial_delay
        self.connection_start_time = datetime.now(timezone.utc)
        
        logger.info(f"✅ Conexão estabelecida (Async) | {self.symbol}")

        try:
            self.slog.info(
                "ws_connected",
                reconnect_count=self.reconnect_count,
                connection_start_time=self.connection_start_time.isoformat(),
            )
        except Exception:
            pass
        
        if self.on_open_callback:
            # Suporta callbacks sync e async
            if asyncio.iscoroutinefunction(self.on_open_callback):
                await self.on_open_callback(ws)
            else:
                self.on_open_callback(ws)

    async def _handle_message(self, msg) -> None:
        """Processa mensagens recebidas com tratamento robusto de JSON."""
        if msg.type == WSMsgType.TEXT:
            self.last_message_time = datetime.now(timezone.utc)
            self.total_messages_received += 1
            
            # Chama heartbeat externo se configurado
            if self.external_heartbeat_cb:
                try:
                    self.external_heartbeat_cb()
                except Exception as hb_err:
                    logger.warning(f"Falha no heartbeat externo: {hb_err}")

            if self.on_message_callback:
                try:
                    # Validação básica do JSON antes de passar para o callback
                    try:
                        data = json.loads(msg.data)
                        if not isinstance(data, dict):
                            raise ValueError("Mensagem não é um objeto JSON válido")
                        
                        # Verifica se é uma mensagem de OrderBook válida
                        if 'e' in data and data['e'] == 'depthUpdate':
                            # Mensagem de OrderBook válida
                            pass
                        elif 'result' in data or 'error' in data:
                            # Mensagem de controle da Binance
                            logger.info(f"Mensagem de controle da Binance: {data}")
                            if 'error' in data:
                                logger.error(f"Erro da Binance: {data['error']}")
                                return
                            
                    except json.JSONDecodeError as json_err:
                        logger.error(f"Erro ao decodificar JSON: {json_err}. Mensagem: {msg.data[:200]}")
                        # Implementar lógica de fallback ou reconexão aqui
                        await self._handle_reconnect()
                        return
                    except ValueError as val_err:
                        logger.error(f"JSON inválido: {val_err}. Mensagem: {msg.data[:200]}")
                        return
                    except Exception as parse_err:
                        logger.error(f"Erro ao parsear mensagem: {parse_err}. Mensagem: {msg.data[:200]}")
                        return

                    if asyncio.iscoroutinefunction(self.on_message_callback):
                        await self.on_message_callback(self._ws, msg.data)
                    else:
                        # Executa callback síncrono (cuidado para não bloquear o loop)
                        self.on_message_callback(self._ws, msg.data)
                        
                except Exception as e:
                    logger.error(f"Erro no callback on_message: {e}", exc_info=True)
                    # Em caso de erro no callback, tentamos reconectar
                    await self._handle_reconnect()
        
        elif msg.type == WSMsgType.ERROR:
            logger.error(f"Erro no WebSocket: {msg.data}")
            if self.on_error_callback:
                if asyncio.iscoroutinefunction(self.on_error_callback):
                    await self.on_error_callback(self._ws, msg.data)
                else:
                    self.on_error_callback(self._ws, msg.data)
        
        elif msg.type == WSMsgType.PONG:
            # Mensagem de pong recebida - conexão está ativa
            logger.debug("🏓 Pong recebido - conexão WebSocket ativa")
            self.last_message_time = datetime.now(timezone.utc)

    async def _handle_reconnect(self) -> None:
        """Gerencia a lógica de backoff e reconexão."""
        self.is_connected = False
        self.reconnect_count += 1
        self.total_reconnects += 1

        try:
            self.slog.warning(
                "ws_reconnect_scheduled",
                reconnect_count=self.reconnect_count,
                delay_seconds=self.current_delay,
                total_reconnects=self.total_reconnects,
            )
        except Exception:
            pass
        
        if self.on_close_callback:
             # Passamos códigos genéricos pois aiohttp abstrai isso no loop
            if asyncio.iscoroutinefunction(self.on_close_callback):
                await self.on_close_callback(self._ws, -1, "Connection Lost")
            else:
                self.on_close_callback(self._ws, -1, "Connection Lost")

        if self.should_stop:
            return

        if self.reconnect_count <= self.max_reconnect_attempts:
            logger.info(
                f"⏳ Reconectando em {self.current_delay:.1f}s "
                f"(Tentativa {self.reconnect_count}/{self.max_reconnect_attempts})..."
            )
            await asyncio.sleep(self.current_delay)
            
            # Backoff exponencial com jitter melhorado
            import random
            jitter = random.uniform(0, 0.2 * self.current_delay)
            self.current_delay = min(
                self.current_delay * self.backoff_factor + jitter,
                self.max_delay
            )
            
            # Log do próximo delay
            logger.debug(f"Próximo delay de reconexão: {self.current_delay:.1f}s")
            
            if self.on_reconnect_callback:
                if asyncio.iscoroutinefunction(self.on_reconnect_callback):
                    await self.on_reconnect_callback()
                else:
                    self.on_reconnect_callback()

    def get_stats(self) -> dict[str, Any]:
        """
        Retorna estatísticas básicas da conexão, para debug/observabilidade.
        """
        now = datetime.now(timezone.utc)
        conn_uptime = None
        if self.connection_start_time:
            conn_uptime = (now - self.connection_start_time).total_seconds()
        last_msg_age = None
        if self.last_message_time:
            last_msg_age = (now - self.last_message_time).total_seconds()

        return {
            "symbol": self.symbol,
            "is_connected": self.is_connected,
            "should_stop": self.should_stop,
            "total_messages_received": self.total_messages_received,
            "total_reconnects": self.total_reconnects,
            "reconnect_count": self.reconnect_count,
            "current_delay": self.current_delay,
            "connection_start_time": self.connection_start_time.isoformat()
            if self.connection_start_time
            else None,
            "last_message_time": self.last_message_time.isoformat()
            if self.last_message_time
            else None,
            "connection_uptime_sec": conn_uptime,
            "last_message_age_sec": last_msg_age,
        }