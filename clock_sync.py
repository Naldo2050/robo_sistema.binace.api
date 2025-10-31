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
        self._thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._thread.start()
        self.logger = logging.getLogger(__name__)
        self.logger.info("🕐 ClockSync iniciado (sincronizando com Binance)")

    def _fetch_server_time(self):
        """Obtém o horário do servidor Binance (em ms)."""
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

    def _sync_loop(self):
        """Loop de sincronização periódica."""
        while not self._stop_event.wait(_SYNC_INTERVAL):
            offset = self._fetch_server_time()
            if offset is not None:
                self._offset_sec = offset
                self._is_synced = True
                self._last_sync = time.time()
                self.logger.debug(f"[ClockSync] Offset atualizado: {offset:+.3f}s")

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
