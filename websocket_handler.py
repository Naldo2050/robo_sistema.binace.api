# websocket_handler.py
"""
Handler de WebSocket com reconexão robusta e backoff exponencial.
"""

import asyncio
import aiohttp
import logging
import random
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class RobustWebSocketHandler:
    """
    Handler de WebSocket com reconexão robusta e backoff exponencial
    
    Funcionalidades:
    - Reconexão automática com backoff exponencial
    - Período de aquecimento pós-reconexão
    - heartbeat configurável
    - Métricas de reconexão
    """
    
    def __init__(
        self,
        url: str,
        on_message: Callable,
        max_reconnect_attempts: int = 25,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        ping_interval: float = 20.0,
        ping_timeout: float = 10.0
    ):
        self.url = url
        self.on_message = on_message
        self.max_reconnect_attempts = max_reconnect_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        
        self.reconnect_count = 0
        self.total_reconnects = 0
        self.is_connected = False
        self.is_warming_up = False
        self.warmup_windows_remaining = 0
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._running = False
    
    def _calculate_backoff_delay(self) -> float:
        """
        Calcula delay com backoff exponencial e jitter.
        
        Returns:
            Delay em segundos para próxima tentativa de reconexão
        """
        delay = min(
            self.initial_delay * (2 ** self.reconnect_count),
            self.max_delay
        )
        # Adicionar jitter de até 20%
        jitter = delay * random.uniform(-0.2, 0.2)
        return max(0.5, delay + jitter)  # Mínimo de 0.5s
    
    async def connect(self):
        """
        Estabelece conexão com retry automático.
        
        Raises:
            ConnectionError: Se atingir máximo de tentativas de reconexão
        """
        self._running = True
        
        while self._running and self.reconnect_count < self.max_reconnect_attempts:
            try:
                logger.info(
                    f"🔌 Conectando ao WebSocket... "
                    f"(tentativa {self.reconnect_count + 1}/{self.max_reconnect_attempts})"
                )
                
                if self._session is None or self._session.closed:
                    timeout = aiohttp.ClientTimeout(total=30)
                    self._session = aiohttp.ClientSession(timeout=timeout)
                
                self._ws = await self._session.ws_connect(
                    self.url,
                    heartbeat=self.ping_interval,
                    receive_timeout=self.ping_timeout
                )
                
                self.is_connected = True
                self.reconnect_count = 0  # Reset contador após sucesso
                self.total_reconnects += 1
                
                logger.info(
                    f"✅ Conexão WebSocket estabelecida "
                    f"(total reconexões: {self.total_reconnects})"
                )
                
                # Iniciar período de aquecimento
                await self._start_warmup()
                
                # Processar mensagens
                await self._message_loop()
                
            except aiohttp.ClientError as e:
                logger.error(f"❌ Erro de conexão WebSocket: {e}")
                await self._handle_disconnect()
                
            except asyncio.CancelledError:
                logger.info("⏹️ Conexão WebSocket cancelada")
                break
                
            except Exception as e:
                logger.error(f"❌ Erro inesperado na conexão: {e}", exc_info=True)
                await self._handle_disconnect()
        
        if self.reconnect_count >= self.max_reconnect_attempts:
            logger.critical(
                "💀 Número máximo de tentativas de reconexão atingido! "
                f"({self.max_reconnect_attempts})"
            )
            raise ConnectionError("Max reconnect attempts exceeded")
    
    async def _message_loop(self):
        """Loop de processamento de mensagens"""
        try:
            async for msg in self._ws:
                if not self._running:
                    break
                    
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        await self.on_message(msg.data)
                    except Exception as e:
                        logger.error(f"Erro processando mensagem: {e}")
                        
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"Erro WebSocket: {self._ws.exception()}")
                    break
                    
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                    logger.warning("🔌 WebSocket fechado pelo servidor")
                    break
                    
                elif msg.type == aiohttp.WSMsgType.PING:
                    await self._ws.pong(msg.data)
                    
                elif msg.type == aiohttp.WSMsgType.PONG:
                    logger.debug("💓 heartbeat PONG recebido")
                    
        except Exception as e:
            logger.error(f"Erro no message loop: {e}")
    
    async def _handle_disconnect(self):
        """
        Trata desconexão com backoff exponencial.
        """
        self.is_connected = False
        
        if self._running:  # Só reconecta se ainda deve estar rodando
            self.reconnect_count += 1
            
            delay = self._calculate_backoff_delay()
            
            logger.warning(
                f"⏳ Reconectando em {delay:.1f}s "
                f"(tentativa {self.reconnect_count}/{self.max_reconnect_attempts})"
            )
            
            await asyncio.sleep(delay)
    
    async def _start_warmup(self):
        """
        Inicia período de aquecimento pós-reconexão.
        """
        if self.total_reconnects > 1:  # Não na primeira conexão
            self.is_warming_up = True
            self.warmup_windows_remaining = 3
            
            logger.warning(
                f"🔄 RECONEXÃO DETECTADA - Iniciando período de aquecimento..."
            )
            logger.info(
                f"⏳ Aguardando {self.warmup_windows_remaining} janelas "
                "para estabilizar dados..."
            )
        else:
            logger.info("✅ Conexão inicial estabelecida - Sistema pronto!")
    
    def complete_warmup_window(self):
        """
        Marca uma janela de aquecimento como concluída.
        """
        if self.is_warming_up and self.warmup_windows_remaining > 0:
            self.warmup_windows_remaining -= 1
            
            logger.info(
                f"⏳ AQUECIMENTO: Janela processada "
                f"({3 - self.warmup_windows_remaining}/3)"
            )
            
            if self.warmup_windows_remaining == 0:
                self.is_warming_up = False
                logger.info("✅ AQUECIMENTO CONCLUÍDO - Sistema pronto!")
    
    def should_skip_analysis(self) -> bool:
        """
        Verifica se deve pular análise (durante aquecimento).
        
        Returns:
            True se deve pular análise, False caso contrário
        """
        return self.is_warming_up
    
    async def close(self):
        """
        Fecha conexão graciosamente.
        """
        self._running = False
        
        if self._ws and not self._ws.closed:
            try:
                await self._ws.close()
            except Exception as e:
                logger.debug(f"Erro ao fechar WebSocket: {e}")
        
        if self._session and not self._session.closed:
            try:
                await self._session.close()
            except Exception as e:
                logger.debug(f"Erro ao fechar sessão: {e}")
        
        logger.info("🔌 WebSocket fechado")
    
    def get_stats(self) -> dict:
        """
        Retorna estatísticas da conexão.
        
        Returns:
            Dicionário com estatísticas
        """
        return {
            "url": self.url,
            "is_connected": self.is_connected,
            "is_warming_up": self.is_warming_up,
            "warmup_windows_remaining": self.warmup_windows_remaining,
            "reconnect_count": self.reconnect_count,
            "total_reconnects": self.total_reconnects,
            "max_reconnect_attempts": self.max_reconnect_attempts,
            "running": self._running,
        }


async def test_connection():
    """Teste básico da conexão WebSocket."""
    import config
    
    async def dummy_handler(message):
        print(f"Recebido: {message[:100]}...")
    
    handler = RobustWebSocketHandler(
        url=config.STREAM_URL,
        on_message=dummy_handler,
        max_reconnect_attempts=3,
        initial_delay=0.5,
        max_delay=5.0
    )
    
    try:
        await asyncio.wait_for(handler.connect(), timeout=10.0)
    except asyncio.TimeoutError:
        logger.info("Teste超时 - conexão funcionando normalmente")
    except ConnectionError as e:
        logger.error(f"Teste falhou: {e}")
    finally:
        await handler.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_connection())
