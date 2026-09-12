# audit_live/forensic_writer.py
# -*- coding: utf-8 -*-
"""Writer JSONL robusto para captura forense.

Requisitos do autorizador:
- NUNCA `except: pass` silencioso.
- Falha de escrita NÃO derruba o bot, mas incrementa contadores
  audit_writer_errors / dropped_audit_records com tipo, timestamp e arquivo.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ForensicWriter:
    """Writer NÃO-BLOQUEANTE: fila + worker dedicado + contabilidade explícita.

    Mecanismo concreto:
    - append_jsonl() faz APENAS enqueue não-bloqueante (put_nowait) no thread caller.
    - json.dumps + open/write ocorrem SOMENTE no worker dedicado (_worker_loop).
    - Fila: queue.Queue(maxsize=20000).
    - Fila cheia -> put_nowait levanta Full -> incrementa dropped_audit_records +
      audit_writer_errors{queue_full} + warning (nunca bloqueia o WS).
    - flush()/close(): sinaliza shutdown, worker drena até fila vazia, fecha.
    """

    def __init__(self, maxsize: int = 20000) -> None:
        import queue as _queue

        self._queue: Any = _queue.Queue(maxsize=maxsize)
        self._maxsize = maxsize
        self.max_depth = 0
        self.records_enqueued = 0
        self.records_written = 0
        self._worker_started = False
        self._shutdown = False
        self._worker_thread: threading.Thread | None = None
        self._file_locks: Dict[str, threading.Lock] = {}
        self._file_locks_guard = threading.Lock()
        # Contadores exigidos (seção 2 da autorização)
        self.audit_writer_errors = 0
        self.dropped_audit_records = 0
        self.last_error: Dict[str, Any] | None = None
        self._counter_lock = threading.Lock()

    def _ensure_worker(self) -> None:
        if self._worker_started:
            return
        with self._counter_lock:
            if self._worker_started:
                return
            self._worker_started = True
        t = threading.Thread(target=self._worker_loop, name="forensic-writer", daemon=True)
        t.start()
        self._worker_thread = t

    def _file_lock(self, path: str) -> threading.Lock:
        with self._file_locks_guard:
            lock = self._file_locks.get(path)
            if lock is None:
                lock = threading.Lock()
                self._file_locks[path] = lock
            return lock

    def _record_error(self, error_type: str, file_path: str, exc: BaseException) -> None:
        with self._counter_lock:
            self.audit_writer_errors += 1
            self.dropped_audit_records += 1
            self.last_error = {
                "error_type": error_type,
                "file": file_path,
                "timestamp_ms": int(time.time() * 1000),
                "detail": str(exc)[:500],
            }
        # NÃO esconder: warning explícito com exc_info
        logger.warning(
            "FORENSIC_WRITER_ERROR type=%s file=%s dropped_total=%d err=%s",
            error_type,
            file_path,
            self.dropped_audit_records,
            exc,
            exc_info=True,
        )

    def append_jsonl(self, path: str, record: Dict[str, Any]) -> bool:
        """Enqueue não-bloqueante. json.dumps + I/O ocorrem no worker, NÃO no caller."""
        self._ensure_worker()
        if self._shutdown:
            self._record_error("shutdown_enqueued", path, RuntimeError("writer closed"))
            return False
        try:
            parent = os.path.dirname(path)
            if parent and not os.path.exists(parent):
                try:
                    os.makedirs(parent, exist_ok=True)
                except Exception as e_mkdir:
                    self._record_error("mkdir_failed", path, e_mkdir)
                    return False
            # Sem serialização aqui: enfileira referência (hooks criam dict novo por chamada)
            self._queue.put_nowait((path, record))
            with self._counter_lock:
                self.records_enqueued += 1
                qd = self._queue.qsize()
                if qd > self.max_depth:
                    self.max_depth = qd
            return True
        except Exception as e_full:
            # Fila cheia (queue.Full) ou outro erro de enqueue
            try:
                import queue as _q

                etype = "queue_full" if isinstance(e_full, _q.Full) else "enqueue_failed"
            except Exception:
                etype = "enqueue_failed"
            self._record_error(etype, path, e_full)
            return False

    def _worker_loop(self) -> None:
        import queue as _queue

        while True:
            try:
                item = self._queue.get(timeout=0.5)
            except _queue.Empty:
                if self._shutdown:
                    return
                continue
            if item is None:  # sentinela de shutdown
                try:
                    self._queue.task_done()
                except Exception:
                    pass
                return
            path, record = item
            try:
                line = json.dumps(record, ensure_ascii=False, default=str)
            except Exception as e_ser:
                self._record_error("serialize_failed", str(path), e_ser)
                try:
                    self._queue.task_done()
                except Exception:
                    pass
                continue
            try:
                lock = self._file_lock(str(path))
                with lock:
                    with open(str(path), "a", encoding="utf-8") as f:
                        f.write(line + "\n")
                with self._counter_lock:
                    self.records_written += 1
            except Exception as e_write:
                self._record_error("write_failed", str(path), e_write)
            finally:
                try:
                    self._queue.task_done()
                except Exception as e_td:
                    logger.warning("FORENSIC task_done falhou: %s", e_td, exc_info=True)

    def flush(self, timeout: float = 10.0) -> bool:
        """Bloqueia até fila vazia (join) ou timeout. Usado no encerramento limpo."""
        import time as _time

        start = _time.time()
        while True:
            try:
                pending = self._queue.unfinished_tasks
            except Exception as e_q:
                logger.warning("FORENSIC flush qsize falhou: %s", e_q, exc_info=True)
                return False
            if pending <= 0:
                return True
            if _time.time() - start > timeout:
                logger.warning("FORENSIC flush timeout pending=%d", pending)
                return False
            _time.sleep(0.05)

    def close(self, timeout: float = 15.0) -> bool:
        """Drena fila e encerra worker. Deve ser chamado antes de gerar manifest/sha."""
        self._shutdown = True
        try:
            try:
                self._queue.put_nowait(None)
            except Exception as e_s:
                logger.warning("FORENSIC close sentinel falhou: %s", e_s, exc_info=True)
        except Exception as e:
            logger.warning("FORENSIC close falhou: %s", e, exc_info=True)
            return False
        ok = self.flush(timeout=timeout)
        try:
            if self._worker_thread is not None:
                self._worker_thread.join(timeout=5.0)
        except Exception as e_j:
            logger.warning("FORENSIC worker join falhou: %s", e_j, exc_info=True)
            return False
        return ok

    def snapshot_counters(self) -> Dict[str, Any]:
        try:
            qd = self._queue.qsize()
        except Exception as e_q:
            logger.warning("FORENSIC qsize snapshot falhou: %s", e_q, exc_info=True)
            qd = -1
        with self._counter_lock:
            return {
                "audit_writer_errors": self.audit_writer_errors,
                "dropped_audit_records": self.dropped_audit_records,
                "last_error": self.last_error,
                "queue_maxsize": self._maxsize,
                "queue_depth": qd,
                "queue_max_depth": self.max_depth,
                "records_enqueued": self.records_enqueued,
                "records_written": self.records_written,
            }


# Singleton compartilhado pelos hooks
WRITER = ForensicWriter()
