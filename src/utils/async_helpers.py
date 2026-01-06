"""
Helpers para executar código async em contextos síncronos (threads).
"""
import asyncio
import logging
from typing import Any, Coroutine, Optional
from concurrent.futures import ThreadPoolExecutor
import functools

logger = logging.getLogger(__name__)

# Executor global para tasks async
executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="async_helper")


def run_async_in_thread(coro: Coroutine) -> Any:
    """
    Executa coroutine de forma segura em qualquer contexto (thread ou main).
    
    Resolve o problema: "There is no current event loop in thread"
    
    Args:
        coro: Coroutine a executar
        
    Returns:
        Resultado da coroutine
        
    Usage:
        result = run_async_in_thread(some_async_function())
    """
    try:
        # Tentar obter loop existente
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop is not None and loop.is_running():
            # Já tem um loop rodando - usar run_coroutine_threadsafe
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            try:
                return future.result(timeout=30)
            except concurrent.futures.TimeoutError:
                logger.error("⚠️ Timeout executando coroutine em loop existente")
                future.cancel()
                return None
        else:
            # Não tem loop ou não está rodando - criar novo
            try:
                # Python 3.7+
                return asyncio.run(coro)
            except RuntimeError:
                # Fallback: criar loop manualmente
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()
                    
    except Exception as e:
        logger.error(f"❌ Erro executando async em thread: {e}")
        return None


def async_to_sync(func):
    """
    Decorator para converter função async em sync.
    Útil para chamar funções async de contextos síncronos.
    
    Usage:
        @async_to_sync
        async def my_async_function():
            ...
        
        # Agora pode chamar como função normal:
        result = my_async_function()
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        coro = func(*args, **kwargs)
        return run_async_in_thread(coro)
    return wrapper


class AsyncBridge:
    """
    Bridge para executar código async em threads sem event loop.
    Mantém um loop dedicado em thread separada.
    """
    
    _instance = None
    _loop: Optional[asyncio.AbstractEventLoop] = None
    _thread = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._start_loop()
    
    def _start_loop(self):
        """Inicia loop em thread dedicada"""
        import threading
        
        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()
        
        self._thread = threading.Thread(target=run_loop, daemon=True, name="AsyncBridge")
        self._thread.start()
        
        # Aguardar loop estar pronto
        import time
        for _ in range(50):  # 5 segundos max
            if self._loop is not None:
                break
            time.sleep(0.1)
        
        logger.info("✅ AsyncBridge iniciado com loop dedicado")
    
    def run(self, coro: Coroutine, timeout: float = 30) -> Any:
        """
        Executa coroutine no loop dedicado.
        
        Args:
            coro: Coroutine a executar
            timeout: Timeout em segundos
            
        Returns:
            Resultado da coroutine
        """
        if self._loop is None:
            logger.error("❌ AsyncBridge loop não está pronto")
            return None
        
        import concurrent.futures
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logger.error(f"⚠️ AsyncBridge timeout após {timeout}s")
            future.cancel()
            return None
        except Exception as e:
            logger.error(f"❌ AsyncBridge erro: {e}")
            return None
    
    def stop(self):
        """Para o loop"""
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)


# Instância global
_async_bridge: Optional[AsyncBridge] = None


def get_async_bridge() -> AsyncBridge:
    """Retorna instância global do AsyncBridge"""
    global _async_bridge
    if _async_bridge is None:
        _async_bridge = AsyncBridge()
    return _async_bridge


def run_async_in_thread_safe(coro: Coroutine, default=None, timeout: int = 30) -> Any:
    """
    Executa coroutine com tratamento de erro robusto.
    
    Args:
        coro: Coroutine a executar
        default: Valor padrão se falhar
        timeout: Timeout em segundos
        
    Returns:
        Resultado ou default se falhar
    """
    try:
        result = run_async_in_thread(coro)
        
        if result is None:
            logger.warning("⚠️ Coroutine retornou None")
            return default
        
        return result
        
    except asyncio.TimeoutError:
        logger.error(f"⚠️ Timeout ({timeout}s) executando coroutine")
        return default
    except Exception as e:
        logger.error(f"❌ Erro executando coroutine: {e}")
        return default