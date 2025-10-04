# event_bus.py

import time
import threading
from collections import deque
from typing import Dict, Any, Callable
import logging
import hashlib

class EventBus:
    def __init__(self, max_queue_size=1000, deduplication_window=30):
        """
        max_queue_size: Tamanho máximo da fila de eventos
        deduplication_window: Tempo em segundos para deduplicação (30s padrão)
        """
        self._handlers = {}
        self._queue = deque(maxlen=max_queue_size)
        self._lock = threading.Lock()
        self._processing = False
        self._thread = None
        self._stop = False
        self._dedup_cache = {}
        self._dedup_window = deduplication_window
        self._logger = logging.getLogger("EventBus")
        
        # Iniciar thread de processamento
        self.start()

    def _generate_event_id(self, event: Dict) -> str:
        """Gera ID único para deduplicação"""
        # Campos mais relevantes para identificação única
        key = f"{event.get('timestamp', '')}|{event.get('delta', '')}|{event.get('volume_total', '')}|{event.get('preco_fechamento', '')}"
        return hashlib.md5(key.encode()).hexdigest()

    def _is_duplicate(self, event: Dict) -> bool:
        """Verifica se evento é duplicado"""
        event_id = self._generate_event_id(event)
        current_time = time.time()
        
        # Limpar cache antigo
        expired_keys = [k for k, t in self._dedup_cache.items() if current_time - t > self._dedup_window]
        for k in expired_keys:
            del self._dedup_cache[k]
        
        # Verificar duplicado
        if event_id in self._dedup_cache:
            return True
            
        # Registrar novo evento
        self._dedup_cache[event_id] = current_time
        return False

    def subscribe(self, event_type: str, handler: Callable):
        with self._lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)

    def publish(self, event_type: str, event_data: Dict):
        with self._lock:
            # Ignorar eventos duplicados
            if self._is_duplicate(event_data):
                self._logger.debug(f"Evento duplicado ignorado: {event_type}")
                return
                
            # Adicionar à fila
            self._queue.append((event_type, event_data))
            
            # Log de debug
            self._logger.debug(f"Evento publicado: {event_type}")

    def _process_queue(self):
        while not self._stop:
            try:
                if self._queue:
                    with self._lock:
                        event_type, event_data = self._queue.popleft()
                    
                    # Processar evento
                    self._dispatch(event_type, event_data)
                else:
                    time.sleep(0.01)  # Pequena pausa para reduzir uso de CPU
            except Exception as e:
                self._logger.error(f"Erro no processamento de eventos: {e}")

    def _dispatch(self, event_type: str, event_data: Dict):
        """Envia evento para todos os handlers registrados"""
        if event_type in self._handlers:
            for handler in self._handlers[event_type]:
                try:
                    handler(event_data)
                except Exception as e:
                    self._logger.error(f"Erro no handler para {event_type}: {e}")
        else:
            self._logger.debug(f"Nenhum handler para {event_type}")

    def start(self):
        if not self._thread or not self._thread.is_alive():
            self._stop = False
            self._thread = threading.Thread(target=self._process_queue, daemon=True)
            self._thread.start()
            self._logger.info("EventBus iniciado")

    def shutdown(self):
        self._stop = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._logger.info("EventBus desligado")