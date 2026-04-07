# -*- coding: utf-8 -*-
"""
clock_sync.py - v1.0.0
----------------------------------------
Sincronização de relógio com servidor Binance.

Fornece offset entre o relógio local e o horário do servidor Binance,
com atualização periódica e compensação automática de drift.

Interface compatível com:
  - get_clock_sync()
  - is_synced()
  - get_offset_seconds()
  - get_server_time_ms()
  - stop()
"""

import os
import time
import threading
import requests
import logging
from datetime import datetime, timedelta, timezone

_BINANCE_TIME_URL = "https://api.binance.com/api/v3/time"
_SYNC_INTERVAL = 60.0  # segundos
_TIMEOUT = 5.0


class ClockSync:
    def __init__(self):
        self._offset_sec = 0.0
        self._is_synced = False
        self._last_sync = None
        self._stop_event = threading.Event()
        self._initial_sync_event = threading.Event()
        self.logger = logging.getLogger(__name__)

        if os.getenv("BOT_TEST_MODE") == "1":
            self._is_synced = True
            self._offset_sec = 0.0
            self._last_sync = time.time()
            self._initial_sync_event.set()
            self.logger.info("🕐 [TEST_MODE] ClockSync: Rede desabilitada, usando tempo local.")
            # Inicializa outros campos para evitar AttributeError se acessados
            self._offset_history = [0.0]
            self._max_history = 10
            self._sync_count = 1
            self._max_offset_warning = 0.6
            self._last_warning_offset = None
            return

        self._thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._thread.start()
        
        # Histórico de offsets para detecção de drift
        self._offset_history = []
        self._max_history = 10
        self._sync_count = 0
        
        # Tolerância máxima de offset (600ms = 0.6s) antes de avisar
        self._max_offset_warning = 0.6
        self._last_warning_offset = None
        
        self.logger.info("🕐 ClockSync iniciado (sincronizando com Binance)")

    def _fetch_server_time(self):
        """Obtém o horário do servidor Binance (em ms)."""
        if os.getenv("BOT_TEST_MODE") == "1":
            return 0.0

        try:
            t0 = time.time()
            response = requests.get(_BINANCE_TIME_URL, timeout=_TIMEOUT)
            response.raise_for_status()
            t1 = time.time()

            data = response.json()
            server_time_ms = int(data["serverTime"])
            server_time = server_time_ms / 1000.0

            # Round trip latency estimada
            latency = (t1 - t0) / 2.0
            adjusted_local_time = t0 + latency

            offset = server_time - adjusted_local_time
            return offset
        except Exception as e:
            self.logger.warning(f"⚠️ Falha ao obter hora do servidor: {e}")
            return None

    def wait_for_sync(self, timeout: float = 5.0) -> bool:
        """Aguarda a primeira sincronização completar (blocking)."""
        return self._initial_sync_event.wait(timeout=timeout)

    def _sync_loop(self):
        """Loop de sincronização periódica."""
        # ✅ Sincronização inicial imediata
        offset = self._fetch_server_time()
        if offset is not None:
            self._update_offset(offset, is_initial=True)

        while not self._stop_event.wait(_SYNC_INTERVAL):
            offset = self._fetch_server_time()
            if offset is not None:
                self._update_offset(offset, is_initial=False)

    def _update_offset(self, offset: float, is_initial: bool = False):
        """Atualiza offset com rastreamento de drift e avisos."""
        self._sync_count += 1
        self._offset_sec = offset
        self._is_synced = True
        self._last_sync = time.time()
        
        # Rastrear histórico para detecção de drift
        self._offset_history.append(offset)
        if len(self._offset_history) > self._max_history:
            self._offset_history.pop(0)
        
        # Detectar drift
        if len(self._offset_history) >= 3:
            offsets = self._offset_history
            drift_trend = offsets[-1] - offsets[0]
            avg_offset = sum(offsets) / len(offsets)
            
            # Se drift é significativo (cresce), advertir
            if abs(drift_trend) > 0.05 and self._sync_count % 5 == 0:
                self.logger.warning(
                    f"⚠️ ClockSync DRIFT detectado: offset={offset:+.3f}s, "
                    f"trend={drift_trend:+.3f}s (últimas {len(offsets)} sincronizações)"
                )
        
        # Avisar se offset excede limite
        if abs(offset) > self._max_offset_warning:
            if self._last_warning_offset is None or abs(offset - self._last_warning_offset) > 0.1:
                self.logger.warning(
                    f"⚠️ ClockSync OFFSET DEGRADADO: {abs(offset*1000):+.0f}ms "
                    f"(limite: {self._max_offset_warning*1000:.0f}ms) - "
                    f"Sistema operando em modo DEGRADADO"
                )
                self._last_warning_offset = offset
        
        if is_initial:
            self._initial_sync_event.set()
            self.logger.info(f"[ClockSync] Sincronização inicial concluída. Offset: {offset:+.3f}s")
        else:
            self.logger.debug(f"[ClockSync] Offset atualizado: {offset:+.3f}s (sync #{self._sync_count})")

    def is_synced(self):
        """Retorna True se o relógio foi sincronizado com sucesso."""
        return self._is_synced

    def get_offset_seconds(self):
        """Retorna o offset atual (servidor - local)."""
        return self._offset_sec

    def get_server_time_ms(self):
        """Retorna timestamp do servidor em milissegundos."""
        now = time.time() + self._offset_sec
        return int(now * 1000)

    def get_stats(self) -> dict:
        """Retorna estatísticas de sincronização para diagnóstico."""
        if not self._offset_history:
            return {"status": "not_synced"}
        
        offsets = self._offset_history
        return {
            "is_synced": self._is_synced,
            "current_offset_ms": int(self._offset_sec * 1000),
            "max_offset_allowed_ms": int(self._max_offset_warning * 1000),
            "sync_count": self._sync_count,
            "offset_history_ms": [int(o * 1000) for o in offsets],
            "offset_stdev_ms": int((sum((o - sum(offsets)/len(offsets))**2 for o in offsets) / len(offsets)) ** 0.5 * 1000) if len(offsets) > 1 else 0,
            "status": "healthy" if abs(self._offset_sec) < self._max_offset_warning else "degraded",
        }

    def stop(self):
        """Interrompe o loop de sincronização."""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.logger.info("🛑 ClockSync parado com segurança")


# Singleton global
_clock_sync_instance = None
_lock = threading.Lock()


def get_clock_sync():
    """Obtém instância única de ClockSync (thread-safe)."""
    global _clock_sync_instance
    with _lock:
        if _clock_sync_instance is None:
            _clock_sync_instance = ClockSync()
        return _clock_sync_instance
