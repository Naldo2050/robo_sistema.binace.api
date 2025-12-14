# health_monitor.py
import time
import threading
import logging

from config import HEALTH_CHECK_TIMEOUT, HEALTH_CHECK_CRITICAL, HEALTH_CHECK_INTERVAL

# Tenta importar monitor OCI, falha graciosamente se não existir
try:
    from infrastructure.oci.monitoring import OCIMonitor
except ImportError:
    OCIMonitor = None

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
        
        # OCI Monitoring
        self.oci_monitor = None
        if OCIMonitor:
            try:
                self.oci_monitor = OCIMonitor()
                logging.info(f"☁️ OCI Monitoring habilitado: {self.oci_monitor.enabled}")
            except Exception as e:
                logging.warning(f"⚠️ Falha ao iniciar OCI Monitor: {e}")

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
        """Verifica periodicamente se algum módulo está travado e envia métricas."""
        last_metric_push = 0
        metric_push_interval = 60  # Enviar métricas a cada 60s

        while not self._stop_event.is_set():
            time.sleep(self.check_interval)
            now = time.time()
            
            # 1. Verificação de Liveness (Logs locais)
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

            # 2. Envio de Métricas OCI (se habilitado)
            if self.oci_monitor and self.oci_monitor.enabled:
                if now - last_metric_push >= metric_push_interval:
                    try:
                        self._push_oci_metrics(now)
                        last_metric_push = now
                    except Exception as e:
                        logging.error(f"❌ Erro no envio de métricas OCI: {e}")

    def _push_oci_metrics(self, now):
        """Coleta e envia métricas customizadas para OCI."""
        # Métricas de Sistema
        metrics = self.oci_monitor.collect_system_metrics()
        
        # Status Geral (1 = OK, 0 = Algum Erro Crítico)
        status_val = 1
        critical_count = 0
        warning_count = 0
        
        with self._lock:
            for lvl in self.alerted_level.values():
                if lvl == "critical":
                    status_val = 0
                    critical_count += 1
                elif lvl == "warning":
                    warning_count += 1
                    
            # Lag dos módulos (latência de heartbeat)
            for module, last_beat in self.last_heartbeat.items():
                lag = now - last_beat
                metrics[f"{module}_Lag"] = lag
        
        metrics["BotHealthStatus"] = status_val
        metrics["ActiveCriticalAlerts"] = critical_count
        metrics["ActiveWarningAlerts"] = warning_count
        
        self.oci_monitor.post_metrics(metrics)

    def stop(self):
        """Para o monitor de saúde."""
        self._stop_event.set()
        if self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        logging.info("🛑 HealthMonitor parado.")