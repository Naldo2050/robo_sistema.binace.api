"""
Helpers para executar código async em contextos síncronos (threads).
"""
import asyncio
import logging
from typing import TYPE_CHECKING, Any, Coroutine, Optional
from concurrent.futures import ThreadPoolExecutor
import functools

# TYPE_CHECKING: Help Pylance recognize fredapi module
if TYPE_CHECKING:
    # Import type stub for Pylance
    from types_fredapi import Fred

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


# ═══════════════════════════════════════════════════════════════════════════════
# FUNÇÕES DE EXECUTOR PARA CHAMADAS SÍNCRONAS (yfinance, FRED)
# ═══════════════════════════════════════════════════════════════════════════════

def run_yfinance_in_executor(ticker_symbol: str, period: str = "1d") -> Any:
    """
    Executa chamada yfinance em thread separada para não bloquear o event loop.
    
    CRÍTICO: yfinance.download() e yf.Ticker().history() são operações síncronas
    que podem bloquear o event loop por até 1 segundo, causando latência de 300s.
    
    Args:
        ticker_symbol: Símbolo do ticker (ex: '^VIX', 'DX-Y.NYB', 'CL=F')
        period: Período de dados (default: 1d)
        
    Returns:
        Valor do último fechamento ou None se falhar
    """
    import yfinance as yf
    
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period=period)
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
        return None
    except Exception as e:
        logger.warning(f"⚠️ Erro no yfinance para {ticker_symbol}: {e}")
        return None


def run_yfinance_download_in_executor(symbol: str, period: str = "1d") -> Any:
    """
    Executa yfinance.download() em thread separada.
    
    Útil para downloads de múltiplos tickers de uma vez.
    
    Args:
        symbol: Símbolo ou lista de símbolos
        period: Período de dados
        
    Returns:
        DataFrame com dados ou None se falhar
    """
    import yfinance as yf
    
    try:
        df = yf.download(symbol, period=period, progress=False)
        return df
    except Exception as e:
        logger.warning(f"⚠️ Erro no yfinance.download para {symbol}: {e}")
        return None


def run_fred_get_series_in_executor(series_id: str, api_key: str) -> Any:
    """
    Executa fred.get_series() em thread separada para não bloquear o event loop.
    
    Args:
        series_id: ID da série do FRED (ex: 'GDP', 'CPIAUCSL')
        api_key: API key do FRED
        
    Returns:
        Series com dados ou None se falhar
    """
    try:
        # Importar o módulo FRED de forma segura
        # Usar import local para evitar problemas com analisadores de código
        fredapi_module = __import__('fredapi')  # type: ignore[attr-defined]
        fred = fredapi_module.Fred(api_key=api_key)  # type: ignore[attr-defined]
        series = fred.get_series(series_id)
        return series
    except ImportError:
        logger.warning(f"⚠️ Módulo fredapi não disponível para {series_id}")
        return None
    except Exception as e:
        logger.warning(f"⚠️ Erro no FRED get_series para {series_id}: {e}")
        return None


async def async_yfinance_fetch(ticker_symbol: str, period: str = "1d") -> Any:
    """
    Wrapper async para fetching de dados yfinance de forma não-bloqueante.
    
    Usage:
        value = await async_yfinance_fetch("^VIX", "1d")
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, 
        lambda: run_yfinance_in_executor(ticker_symbol, period)
    )


async def async_fred_fetch(series_id: str, api_key: str) -> Any:
    """
    Wrapper async para fetching de dados FRED de forma não-bloqueante.
    
    Usage:
        series = await async_fred_fetch("GDP", fred_api_key)
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: run_fred_get_series_in_executor(series_id, api_key)
    )