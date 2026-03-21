#!/usr/bin/env python3
"""
AsyncBridge - Ponte para operações assíncronas
"""

import asyncio
from typing import Any, Callable, Coroutine, Optional, Dict
from concurrent.futures import ThreadPoolExecutor


class AsyncBridge:
    """Ponte para operações assíncronas"""
    
    def __init__(self):
        """Inicializa a ponte assíncrona"""
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._loop = None
    
    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """Define o event loop"""
        self._loop = loop
    
    async def run_in_executor(self, func: Callable, *args, **kwargs) -> Any:
        """Executa uma função síncrona em um executor assíncrono"""
        if self._loop is None:
            self._loop = asyncio.get_event_loop()
        
        return await self._loop.run_in_executor(
            self._executor, 
            lambda: func(*args, **kwargs)
        )
    
    async def run_coroutine(self, coro: Coroutine) -> Any:
        """Executa uma coroutine"""
        return await coro
    
    def create_task(self, coro: Coroutine) -> asyncio.Task:
        """Cria uma task assíncrona"""
        if self._loop is None:
            self._loop = asyncio.get_event_loop()
        
        return self._loop.create_task(coro)
    
    async def gather_tasks(self, *tasks: asyncio.Task) -> list:
        """Agrupa múltiplas tasks"""
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def shutdown(self):
        """Encerra o executor"""
        if self._executor:
            self._executor.shutdown(wait=True)