# health_monitor.py
import time
import threading
import logging

from config import HEALTH_CHECK_TIMEOUT, HEALTH_CHECK_CRITICAL, HEALTH_CHECK_INTERVAL


class HealthMonitor:
    def __init__(
        self,
        max_silence_seconds: int = HEALTH_CHECK_TIMEOUT,
        check_interval_seconds: int = HEALTH_CHECK_INTERVAL,
        critical_silence_seconds: int = HEALTH_CHECK_CRITICAL,
    ):
        """
        Monitora a saúde dos módulos principais do sistema.

        - max_silence_seconds: tempo sem heartbeat para logar WARNING
        - critical_silence_seconds: tempo sem heartbeat para logar CRITICAL
        - check_interval_seconds: intervalo de verificação do monitor
        """
        self.warn_silence = max_silence_seconds
        self.critical_silence = max(critical_silence_seconds, self.warn_silence)
        self.check_interval = max(1, check_interval_seconds)

        self.last_heartbeat: dict[str, float] = {}
        # guarda nível já alertado: "warning" ou "critical"
        self.alerted_level: dict[str, str] = {}
        self._lock = threading.Lock()

        # Inicia thread de verificação
        self._stop_event = threading.Event()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        logging.info(
            "✅ HealthMonitor iniciado. Warning após %ss, Critical após %ss de silêncio.",
            self.warn_silence,
            self.critical_silence,
        )

    def heartbeat(self, module_name: str):
        """Registra que um módulo está vivo."""
        now = time.time()
        with self._lock:
            self.last_heartbeat[module_name] = now

            # Se já tinha sido alertado, limpa o estado e loga recovery
            if module_name in self.alerted_level:
                logging.info("💚 Módulo %s voltou ao normal.", module_name)
                self.alerted_level.pop(module_name, None)

    def _monitor_loop(self):
        """Verifica periodicamente se algum módulo está travado."""
        while not self._stop_event.is_set():
            time.sleep(self.check_interval)
            now = time.time()

            with self._lock:
                for module, last_beat in list(self.last_heartbeat.items()):
                    silence = now - last_beat
                    level = self.alerted_level.get(module)

                    # Primeiro WARNING, depois CRITICAL
                    if silence >= self.critical_silence and level != "critical":
                        logging.critical(
                            "💀 Módulo '%s' sem heartbeat há %.0fs!",
                            module,
                            silence,
                        )
                        self.alerted_level[module] = "critical"

                    elif silence >= self.warn_silence and level is None:
                        logging.warning(
                            "⚠️ Módulo '%s' sem heartbeat há %.0fs!",
                            module,
                            silence,
                        )
                        self.alerted_level[module] = "warning"

    def stop(self):
        """Para o monitor de saúde."""
        self._stop_event.set()
        if self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        logging.info("🛑 HealthMonitor parado.")