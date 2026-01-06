# utils/heartbeat_manager.py

import asyncio
import time
import threading
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class HeartbeatManager:
    """
    Gerenciador de heartbeat robusto para módulos críticos.
    
    Funcionalidades:
    - Heartbeat automático em thread separada
    - Monitoramento de silêncio (alertas quando模块 para de responder)
    - Integração com HealthMonitor existente (opcional)
    """

    def __init__(
        self,
        module_name: str,
        warning_threshold: int = 60,
        critical_threshold: int = 120,
        heartbeat_interval: int = 10,
        auto_beat_interval: int = 30,
        health_monitor=None,
    ):
        self.module_name = module_name
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.heartbeat_interval = heartbeat_interval
        self.auto_beat_interval = auto_beat_interval
        self.health_monitor = health_monitor

        self.last_heartbeat = time.time()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._auto_beat_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._alerted_level = None  # "warning" ou "critical"

    def beat(self):
        """Registra um heartbeat"""
        with self._lock:
            self.last_heartbeat = time.time()
            # Se já tinha sido alertado, limpa o estado
            if self._alerted_level is not None:
                logger.info(f"💚 Módulo '{self.module_name}' voltou ao normal.")
                self._alerted_level = None
            # Also beat to health_monitor if available
            if self.health_monitor:
                try:
                    self.health_monitor.heartbeat(self.module_name)
                except Exception:
                    pass

    def get_silence_seconds(self) -> float:
        """Retorna segundos desde último heartbeat"""
        return time.time() - self.last_heartbeat

    async def start(self):
        """Inicia monitoramento de heartbeat em background"""
        self._running = True
        
        # Inicia loop de monitoramento async
        self._task = asyncio.create_task(self._monitor_loop())
        
        # Inicia thread de heartbeat automático
        self._auto_beat_thread = threading.Thread(
            target=self._auto_beat_loop,
            daemon=True,
            name=f"heartbeat_auto_{self.module_name}"
        )
        self._auto_beat_thread.start()
        
        logger.info(f"💓 HeartbeatManager iniciado para '{self.module_name}' "
                   f"(auto-beat a cada {self.auto_beat_interval}s)")

    async def stop(self):
        """Para monitoramento"""
        self._running = False
        
        # Para thread de auto-beat
        if self._auto_beat_thread:
            self._auto_beat_thread.join(timeout=2.0)
        
        # Cancela task async
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"🛑 HeartbeatManager parado para '{self.module_name}'")

    def _auto_beat_loop(self):
        """Thread que faz heartbeat automaticamente"""
        while self._running:
            try:
                time.sleep(self.auto_beat_interval)
                if self._running:
                    self.beat()
            except Exception as e:
                logger.error(f"Erro no auto-beat: {e}")
                time.sleep(5)

    async def _monitor_loop(self):
        """Loop de monitoramento de silêncio"""
        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                silence = self.get_silence_seconds()
                
                with self._lock:
                    # Já está em nível crítico
                    if self._alerted_level == "critical":
                        continue
                    
                    # Promove warning para critical
                    if self._alerted_level == "warning" and silence > self.critical_threshold:
                        logger.critical(
                            f"💀 CRÍTICO: Módulo '{self.module_name}' sem heartbeat há {silence:.0f}s!"
                        )
                        self._alerted_level = "critical"
                    # Primeiro alerta (warning)
                    elif silence > self.warning_threshold and self._alerted_level is None:
                        logger.warning(
                            f"⚠️ Módulo '{self.module_name}' sem heartbeat há {silence:.0f}s"
                        )
                        self._alerted_level = "warning"
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erro no heartbeat monitor: {e}")
                await asyncio.sleep(5)

    def get_stats(self) -> dict:
        """
        Retorna snapshot do estado atual.
        """
        silence = self.get_silence_seconds()
        with self._lock:
            alerted = self._alerted_level
        
        return {
            "module_name": self.module_name,
            "last_heartbeat_ts": self.last_heartbeat,
            "silence_seconds": silence,
            "alert_level": alerted,
            "running": self._running,
            "warning_threshold": self.warning_threshold,
            "critical_threshold": self.critical_threshold,
        }
