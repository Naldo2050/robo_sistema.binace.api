import threading
import queue
import logging
from typing import Callable, Any

class EventBus:
    def __init__(self, max_queue_size=1000):
        """
        Gerencia eventos de forma assíncrona entre módulos.
        - max_queue_size: tamanho máximo da fila de eventos (evita estouro de memória)
        """
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.subscribers = {}  # { event_type: [callbacks] }
        self._lock = threading.Lock()
        self._running = True

        # Thread worker que processa eventos em background
        self.worker = threading.Thread(target=self._process_events, daemon=True)
        self.worker.start()

        logging.info("✅ EventBus inicializado. Processamento assíncrono ativado.")

    def subscribe(self, event_type: str, callback: Callable[[dict], Any]):
        """Inscreve um callback para um tipo de evento."""
        with self._lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(callback)
            logging.debug(f"🔔 Inscrição: {callback.__qualname__} em '{event_type}'")

    def publish(self, event_type: str, event_data: dict):
        """Publica um evento na fila para processamento assíncrono."""
        try:
            self.queue.put_nowait((event_type, event_data))
        except queue.Full:
            logging.warning(f"⚠️ Fila cheia. Evento '{event_type}' descartado.")
        except Exception as e:
            logging.error(f"❌ Falha ao publicar evento: {e}")

    def _process_events(self):
        """Processa eventos da fila em loop, chamando os subscribers registrados."""
        while self._running:
            try:
                event_type, event_data = self.queue.get(timeout=1)
                if event_type == "__shutdown__":
                    break

                with self._lock:
                    callbacks = self.subscribers.get(event_type, [])

                for callback in callbacks:
                    try:
                        callback(event_data)
                    except Exception as e:
                        logging.error(f"❌ Erro no callback {callback.__qualname__} para evento '{event_type}': {e}")

                self.queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"❌ Erro no worker do EventBus: {e}")

    def shutdown(self):
        """Encerra o EventBus de forma limpa."""
        self._running = False
        try:
            self.queue.put_nowait(("__shutdown__", {}))
        except queue.Full:
            pass
        if self.worker.is_alive():
            self.worker.join(timeout=5)
        logging.info("🛑 EventBus encerrado.")