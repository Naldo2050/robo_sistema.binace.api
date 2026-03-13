# utils/async_helpers.py

import asyncio
import logging
from typing import Callable, Any, Optional, TypeVar
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')

async def safe_async_call(
    coro_factory: Callable[[], Any],
    timeout_seconds: float = 30.0,
    default_value: Any = None,
    operation_name: str = "async_operation"
) -> Any:
    """
    Executa uma coroutine de forma segura com timeout

    IMPORTANTE: coro_factory deve ser uma FUNÇÃO que RETORNA a coroutine,
    não a coroutine em si. Isso evita o erro 'cannot reuse already awaited coroutine'.

    Uso correto:
        # ERRADO:
        result = await safe_async_call(my_async_function())

        # CORRETO:
        result = await safe_async_call(lambda: my_async_function())
        # ou
        result = await safe_async_call(my_async_function)  # se não precisa de args

    Args:
        coro_factory: Função que retorna a coroutine a ser executada
        timeout_seconds: Timeout em segundos
        default_value: Valor padrão se falhar
        operation_name: Nome da operação para logging

    Returns:
        Resultado da coroutine ou default_value se falhar
    """
    try:
        # Criar nova coroutine a cada chamada
        coro = coro_factory()

        # Executar com timeout
        result = await asyncio.wait_for(coro, timeout=timeout_seconds)
        return result

    except asyncio.TimeoutError:
        logger.warning(f"⚠️ Timeout em {operation_name} após {timeout_seconds}s")
        return default_value

    except Exception as e:
        logger.error(f"❌ Erro em {operation_name}: {type(e).__name__}: {e}")
        return default_value


async def run_with_retry(
    coro_factory: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 1.0,
    timeout_seconds: float = 30.0,
    operation_name: str = "async_operation"
) -> Optional[Any]:
    """
    Executa coroutine com retry e backoff exponencial

    Args:
        coro_factory: Função que retorna a coroutine
        max_retries: Número máximo de tentativas
        base_delay: Delay base entre tentativas
        timeout_seconds: Timeout por tentativa
        operation_name: Nome para logging

    Returns:
        Resultado ou None se todas tentativas falharem
    """
    import random

    for attempt in range(max_retries):
        try:
            # Criar nova coroutine
            coro = coro_factory()

            # Executar com timeout
            result = await asyncio.wait_for(coro, timeout=timeout_seconds)

            if attempt > 0:
                logger.info(f"✅ {operation_name} bem-sucedido na tentativa {attempt + 1}")

            return result

        except asyncio.TimeoutError:
            logger.warning(
                f"⚠️ Timeout em {operation_name} "
                f"(tentativa {attempt + 1}/{max_retries})"
            )

        except Exception as e:
            logger.warning(
                f"⚠️ Erro em {operation_name}: {e} "
                f"(tentativa {attempt + 1}/{max_retries})"
            )

        # Se não é última tentativa, aguardar antes de retry
        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            await asyncio.sleep(delay)

    logger.error(f"❌ {operation_name} falhou após {max_retries} tentativas")
    return None


def create_coroutine_factory(func: Callable, *args, **kwargs) -> Callable[[], Any]:
    """
    Cria uma factory function para coroutine com argumentos pré-definidos

    Uso:
        factory = create_coroutine_factory(my_async_func, arg1, arg2, kwarg1=value)
        result = await safe_async_call(factory)
    """
    def factory():
        return func(*args, **kwargs)
    return factory


def run_async_in_thread(coro) -> Any:
    """
    Executa uma coroutine em uma nova thread usando asyncio.run_in_executor.

    IMPORTANTE: Passa a coroutine DIRETAMENTE, não uma factory.
    Isso evita o erro 'cannot reuse already awaited coroutine'.

    Uso correto:
        result = run_async_in_thread(my_async_function())

    Args:
        coro: Coroutine a ser executada (já instanciada)

    Returns:
        Resultado da coroutine
    """
    import asyncio
    import concurrent.futures

    try:
        # Cria um novo event loop em uma thread separada
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result(timeout=30.0)  # Timeout de 30 segundos

    except concurrent.futures.TimeoutError:
        logger.error("Timeout ao executar coroutine em thread separada")
        raise
    except Exception as e:
        logger.error(f"Erro ao executar coroutine em thread: {e}")
        raise