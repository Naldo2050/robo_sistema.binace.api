# connection/robust_connection.py
# -*- coding: utf-8 -*-

"""
Gerenciador robusto de conexão WebSocket + RateLimiter.

Código equivalente ao que existia em market_orchestrator.py original,
com pequenas limpezas (remoção de duplicações) mas mesma lógica.
"""

import logging
import time
import socket
import random
import threading
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse
from collections import deque

import websocket
import config


class RateLimiter:
    """Rate limiter thread-safe para controle de requisições."""

    def __init__(self, max_calls: int, period_seconds: float) -> None:
        self.max_calls = max_calls
        self.period = period_seconds
        self.calls: deque[float] = deque()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        """Adquire permissão para fazer chamada (bloqueia se necessário)."""
        while True:
            sleep_time = 0.0

            with self.lock:
                now = time.time()

                # Remove chamadas antigas
                while self.calls and self.calls[0] < now - self.period:
                    self.calls.popleft()

                # Se atingiu limite, calcula tempo restante
                if len(self.calls) >= self.max_calls:
                    sleep_time = max(0.0, self.calls[0] + self.period - now)
                else:
                    self.calls.append(now)
                    return

            if sleep_time > 0:
                time.sleep(sleep_time)


class RobustConnectionManager:
    """Gerenciador robusto de conexão WebSocket com reconnect automático."""

    def __init__(
        self,
        stream_url: str,
        symbol: str,
        max_reconnect_attempts: int = 10,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 1.5,
    ) -> None:
        self.stream_url = stream_url
        self.symbol = symbol
        self.max_reconnect_attempts = max_reconnect_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor

        self.current_delay = initial_delay
        self.reconnect_count = 0
        self.is_connected = False

        self.last_message_time: Optional[datetime] = None
        self.last_successful_message_time: Optional[datetime] = None
        self.connection_start_time: Optional[datetime] = None

        # Evento para parar conexões e heartbeat
        self.stop_event = threading.Event()

        # Thread de heartbeat
        self.heartbeat_thread: Optional[threading.Thread] = None

        # Callbacks externos
        self.on_message_callback = None
        self.on_open_callback = None
        self.on_close_callback = None
        self.on_error_callback = None
        self.on_reconnect_callback = None

        self.total_messages_received = 0
        self.total_reconnects = 0

        self.external_heartbeat_cb = None

        # Referência para o WebSocket atual
        self._ws: Optional[websocket.WebSocketApp] = None

    # ============================================================
    #   CALLBACKS / CONFIGURAÇÃO
    # ============================================================

    def set_callbacks(
        self,
        on_message=None,
        on_open=None,
        on_close=None,
        on_error=None,
        on_reconnect=None,
    ) -> None:
        self.on_message_callback = on_message
        self.on_open_callback = on_open
        self.on_close_callback = on_close
        self.on_error_callback = on_error
        self.on_reconnect_callback = on_reconnect

    def set_heartbeat_cb(self, cb) -> None:
        self.external_heartbeat_cb = cb

    # ============================================================
    #   TESTE DE CONECTIVIDADE TCP
    # ============================================================

    def _test_connection(self) -> bool:
        """Testa conectividade TCP antes de tentar WebSocket."""
        try:
            parsed = urlparse(self.stream_url)
            host = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == "wss" else 80)

            socket.getaddrinfo(host, port)

            with socket.create_connection((host, port), timeout=3):
                return True
        except Exception as e:
            logging.error(f"Erro ao testar conexão TCP: {e}")
            return False

    # ============================================================
    #   CALLBACKS INTERNOS DO WEBSOCKET
    # ============================================================

    def _on_message(self, ws, message: str) -> None:
        """Callback interno para mensagens recebidas."""
        try:
            self.last_message_time = datetime.now(timezone.utc)
            self.total_messages_received += 1

            if self.on_message_callback:
                self.on_message_callback(ws, message)

            self.last_successful_message_time = self.last_message_time

            # Relaxa o delay quando o fluxo volta ao normal
            if self.current_delay > self.initial_delay:
                self.current_delay = max(
                    self.initial_delay,
                    self.current_delay * 0.9,
                )
        except Exception as e:
            logging.error(f"Erro no processamento da mensagem: {e}")

    def _on_open(self, ws) -> None:
        """Callback interno para conexão aberta."""
        self.is_connected = True

        if self.reconnect_count > 0 and self.on_reconnect_callback:
            try:
                self.on_reconnect_callback()
            except Exception as e:
                logging.error(f"Erro no callback de reconexão: {e}")

        self.reconnect_count = 0
        self.current_delay = self.initial_delay

        now = datetime.now(timezone.utc)
        self.connection_start_time = now
        self.last_message_time = now
        self.last_successful_message_time = now

        logging.info(f"✅ Conexão estabelecida com {self.symbol}")

        self._start_heartbeat()

        if self.on_open_callback:
            self.on_open_callback(ws)

    def _on_close(self, ws, close_status_code, close_msg) -> None:
        self.is_connected = False
        logging.warning(
            f"🔌 Conexão fechada - Código: {close_status_code}, Msg: {close_msg}"
        )

        self._stop_heartbeat()

        if self.on_close_callback:
            self.on_close_callback(ws, close_status_code, close_msg)

    def _on_error(self, ws, error) -> None:
        logging.error(f"❌ Erro WebSocket: {error}")
        if self.on_error_callback:
            self.on_error_callback(ws, error)

    # ============================================================
    #   HEARTBEAT
    # ============================================================

    def _start_heartbeat(self) -> None:
        """Inicia o thread de heartbeat, usando stop_event em vez de bool."""
        self.stop_event.clear()

        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            return

        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_monitor,
            daemon=True,
        )
        self.heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        """Sinaliza parada ao heartbeat e tenta aguardar seu término."""
        self.stop_event.set()
        t = getattr(self, "heartbeat_thread", None)
        if t and t.is_alive():
            t.join(timeout=1.0)

    def _heartbeat_monitor(self) -> None:
        """Monitora a saúde da conexão."""
        while self.is_connected and not self.stop_event.is_set():

            if self.stop_event.wait(20.0):
                break

            if not self.is_connected:
                break

            if self.last_message_time:
                gap = (datetime.now(timezone.utc) - self.last_message_time).total_seconds()
                if gap > 60:
                    logging.warning(f"⚠️ Sem mensagens há {gap:.0f}s. Forçando reconexão.")
                    self.is_connected = False
                    break

            if self.last_successful_message_time:
                valid_gap = (
                    datetime.now(timezone.utc)
                    - self.last_successful_message_time
                ).total_seconds()
                if valid_gap > 120:
                    logging.critical(
                        f"💀 SEM MENSAGENS VÁLIDAS HÁ {valid_gap:.0f}s!"
                    )

    # ============================================================
    #   CICLO DE CONEXÃO + RECONEXÃO
    # ============================================================

    def _calculate_next_delay(self) -> float:
        delay = min(self.current_delay * self.backoff_factor, self.max_delay)
        jitter = delay * 0.2 * (random.random() - 0.5)
        self.current_delay = max(self.initial_delay, delay + jitter)
        return self.current_delay

    def connect(self) -> None:
        ping_interval = getattr(config, "WS_PING_INTERVAL", 15)
        ping_timeout = getattr(config, "WS_PING_TIMEOUT", 8)

        self.stop_event.clear()

        while (
            self.reconnect_count < self.max_reconnect_attempts
            and not self.stop_event.is_set()
        ):
            try:
                if self.external_heartbeat_cb:
                    try:
                        self.external_heartbeat_cb()
                    except Exception:
                        pass

                if not self._test_connection():
                    raise ConnectionError("Falha no teste de conectividade")

                if self.reconnect_count > 0:
                    logging.info(
                        f"🔄 Tentativa de reconexão "
                        f"{self.reconnect_count + 1}/{self.max_reconnect_attempts}"
                    )

                ws = websocket.WebSocketApp(
                    self.stream_url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_open,
                )

                self._ws = ws

                ws.run_forever(
                    ping_interval=ping_interval,
                    ping_timeout=ping_timeout,
                    skip_utf8_validation=True,
                )

                self._ws = None

                if self.stop_event.is_set():
                    break

            except KeyboardInterrupt:
                logging.info("⏹️ Interrompido pelo usuário")
                self.stop_event.set()
                break

            except Exception as e:
                self.reconnect_count += 1
                self.total_reconnects += 1

                logging.error(
                    f"❌ Erro na conexão "
                    f"({self.reconnect_count}/{self.max_reconnect_attempts}): {e}"
                )

                if self.reconnect_count < self.max_reconnect_attempts:
                    delay = self._calculate_next_delay()
                    logging.info(f"⏳ Reconectando em {delay:.1f}s...")

                    t0 = time.time()
                    while (
                        time.time() - t0 < delay
                        and not self.stop_event.is_set()
                    ):
                        if self.external_heartbeat_cb:
                            try:
                                self.external_heartbeat_cb()
                            except Exception:
                                pass
                        time.sleep(1.0)
                else:
                    logging.error(
                        "💀 Máximo de tentativas atingido. Encerrando."
                    )
                    break

        self._stop_heartbeat()

    # ============================================================
    #   DESCONECTAR
    # ============================================================

    def disconnect(self) -> None:
        logging.info("🛑 Iniciando desconexão...")
        self.stop_event.set()

        ws = getattr(self, "_ws", None)
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

        # No código original havia closes duplicados; aqui um único
        # close explícito já é suficiente para manter o comportamento.