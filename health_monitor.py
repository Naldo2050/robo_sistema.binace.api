import time
import threading
import logging

class HealthMonitor:
    def __init__(self, max_silence_seconds=60, check_interval_seconds=10):
        """
        Monitora a saúde dos módulos principais do sistema.
        - max_silence_seconds: tempo máximo sem heartbeat antes de alertar
        - check_interval_seconds: intervalo de verificação do monitor
        """
        self.max_silence = max_silence_seconds
        self.last_heartbeat = {}
        self.alerted = set()  # módulos já alertados
        self._lock = threading.Lock()

        # Inicia thread de verificação
        self._stop_event = threading.Event()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        logging.info(f"✅ HealthMonitor iniciado. Alerta após {max_silence_seconds}s de silêncio.")

    def heartbeat(self, module_name: str):
        """Registra que um módulo está vivo."""
        with self._lock:
            self.last_heartbeat[module_name] = time.time()
            if module_name in self.alerted:
                self.alerted.remove(module_name)
                logging.info(f"💚 Módulo {module_name} voltou ao normal.")

    def _monitor_loop(self):
        """Verifica periodicamente se algum módulo está travado."""
        while not self._stop_event.is_set():
            time.sleep(10)  # checa a cada 10s
            now = time.time()
            with self._lock:
                for module, last_beat in self.last_heartbeat.items():
                    silence = now - last_beat
                    if silence > self.max_silence and module not in self.alerted:
                        logging.critical(f"💀 Módulo '{module}' sem heartbeat há {silence:.0f}s!")
                        self.alerted.add(module)
                        # Aqui você pode adicionar: enviar Telegram, tocar alerta sonoro, reiniciar módulo, etc.

    def stop(self):
        """Para o monitor de saúde."""
        self._stop_event.set()
        if self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        logging.info("🛑 HealthMonitor parado.")