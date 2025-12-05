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
        
        # Configurações de timeout
        ping_interval = getattr(config, "WS_PING_INTERVAL", 20)
        # aiohttp usa heartbeat para enviar pings automáticos
        
        logger.info(f"🚀 Iniciando conexão Async (aiohttp) para {self.symbol}...")

        while not self.should_stop:
            try:
                async with aiohttp.ClientSession() as session:
                    self._session = session
                    
                    # Tentativa de conexão
                    try:
                        async with session.ws_connect(
                            self.stream_url,
                            heartbeat=ping_interval,
                            autoping=True
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
                break

    async def disconnect(self) -> None:
        """Fecha a conexão graciosamente."""
        logger.info("🛑 Desconectando...")
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
        
        if self.on_open_callback:
            # Suporta callbacks sync e async
            if asyncio.iscoroutinefunction(self.on_open_callback):
                await self.on_open_callback(ws)
            else:
                self.on_open_callback(ws)

    async def _handle_message(self, msg) -> None:
        """Processa mensagens recebidas."""
        if msg.type == WSMsgType.TEXT:
            self.last_message_time = datetime.now(timezone.utc)
            self.total_messages_received += 1
            
            # Chama heartbeat externo se configurado
            if self.external_heartbeat_cb:
                self.external_heartbeat_cb()

            if self.on_message_callback:
                try:
                    if asyncio.iscoroutinefunction(self.on_message_callback):
                        await self.on_message_callback(self._ws, msg.data)
                    else:
                        # Executa callback síncrono (cuidado para não bloquear o loop)
                        self.on_message_callback(self._ws, msg.data)
                except Exception as e:
                    logger.error(f"Erro no callback on_message: {e}")
        
        elif msg.type == WSMsgType.ERROR:
            logger.error(f"Erro no WebSocket: {msg.data}")
            if self.on_error_callback:
                if asyncio.iscoroutinefunction(self.on_error_callback):
                    await self.on_error_callback(self._ws, msg.data)
                else:
                    self.on_error_callback(self._ws, msg.data)

    async def _handle_reconnect(self) -> None:
        """Gerencia a lógica de backoff e reconexão."""
        self.is_connected = False
        self.reconnect_count += 1
        self.total_reconnects += 1
        
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
            
            # Backoff exponencial com jitter
            import random
            jitter = random.uniform(0, 0.1 * self.current_delay)
            self.current_delay = min(
                self.current_delay * self.backoff_factor + jitter, 
                self.max_delay
            )
            
            if self.on_reconnect_callback:
                if asyncio.iscoroutinefunction(self.on_reconnect_callback):
                    await self.on_reconnect_callback()
                else:
                    self.on_reconnect_callback()