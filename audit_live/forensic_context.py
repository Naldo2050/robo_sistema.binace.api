# audit_live/forensic_context.py
# -*- coding: utf-8 -*-
"""Contexto global da captura forense (contadores, run_id, diretório).

NÃO altera lógica de negócio. Apenas estado observacional thread-safe.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
import uuid
from typing import Any, Dict, List, Optional


def _getenv(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def forensic_enabled() -> bool:
    return _getenv("FORENSIC_CAPTURE", "0") == "1"


class ForensicContext:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.capture_run_id: str = _getenv("FORENSIC_RUN_ID", "")
        self.session_id: str = f"sess_{uuid.uuid4().hex[:12]}"
        self.base_dir: str = _getenv("FORENSIC_DIR", "")
        self.started_at_utc: str = ""
        # Contadores seção 6 / 16
        self.counters: Dict[str, int] = {
            "raw_messages_received": 0,
            "aggtrades_received": 0,
            "normalized_trades": 0,
            "invalid_trades": 0,
            "trade_buffer_drops": 0,
            "reconnections": 0,
            "orderbook_updates": 0,
            "orderbook_resync_count": 0,
            "windows_generated": 0,
            "analysis_triggers_generated": 0,
            "payloads_generated": 0,
            "llm_requests_sent": 0,
            "external_api_errors": 0,
        }
        # Conjuntos para integridade de IDs (limitados para não estourar RAM)
        self._seen_a: set = set()
        self._max_tracked_ids = 500000
        self.duplicate_aggtrade_ids: int = 0
        self.out_of_order_count: int = 0
        self._max_a_seen: Optional[int] = None
        self._max_T_seen: Optional[int] = None
        self._id_gaps_observed: int = 0
        self._id_gaps_sample: List[Dict[str, Any]] = []
        self._duplicates_sample: List[int] = []

    def ensure_init(self, base_dir: Optional[str] = None) -> str:
        with self._lock:
            if not self.capture_run_id:
                # Re-lê o env aqui (não só no import): cobre import-before-env.
                self.capture_run_id = _getenv("FORENSIC_RUN_ID", "") or ""
            if not self.capture_run_id:
                ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
                self.capture_run_id = f"live_{ts}_{uuid.uuid4().hex[:6]}"
            if base_dir:
                self.base_dir = base_dir
            if not self.base_dir:
                self.base_dir = os.path.join("dados", "audit", self.capture_run_id)
            if not self.started_at_utc:
                self.started_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                # Registra flush/close no exit para drenar fila antes de gerar manifest/sha
                try:
                    import atexit as _atexit

                    if not getattr(self, "_atexit_registered", False):
                        self._atexit_registered = True
                        _atexit.register(self._atexit_flush)
                except Exception as _fe:
                    try:
                        import logging as _lg

                        _lg.warning("FORENSIC atexit register falhou: %s", _fe, exc_info=True)
                    except Exception:
                        pass
            return self.base_dir

    def _atexit_flush(self) -> None:
        """Drena fila + escreve manifest_base. Nunca levanta."""
        try:
            from audit_live.forensic_writer import WRITER as _w

            try:
                _w.flush(timeout=15.0)
            except Exception as _fe:
                import logging as _lg

                _lg.warning("FORENSIC atexit flush falhou: %s", _fe, exc_info=True)
            try:
                import json as _json

                base = self.build_manifest(writer_snapshot=_w.snapshot_counters())
                with open(self.path("manifest_base.json"), "w", encoding="utf-8") as _f:
                    _json.dump(base, _f, ensure_ascii=False, indent=2)
            except Exception as _fe2:
                import logging as _lg2

                _lg2.warning("FORENSIC atexit manifest falhou: %s", _fe2, exc_info=True)
            try:
                _w.close(timeout=10.0)
            except Exception as _fe3:
                import logging as _lg3

                _lg3.warning("FORENSIC atexit close falhou: %s", _fe3, exc_info=True)
        except Exception as _fe0:
            try:
                import logging as _lg0

                _lg0.warning("FORENSIC atexit outer falhou: %s", _fe0, exc_info=True)
            except Exception:
                pass

    def path(self, filename: str) -> str:
        base = self.base_dir or os.path.join("dados", "audit", self.capture_run_id or "live_manual")
        return os.path.join(base, filename)

    def inc(self, key: str, delta: int = 1) -> None:
        with self._lock:
            self.counters[key] = self.counters.get(key, 0) + delta

    def observe_aggtrade_id(self, a: Any, T: Any) -> Dict[str, Any]:
        """Observa integridade de sequência. NUNCA descarta. Retorna flags."""
        flags: Dict[str, Any] = {"duplicate": False, "gap": False, "ooo_time": False}
        try:
            a_int = int(a) if a is not None else None
        except Exception:
            a_int = None
        try:
            T_int = int(T) if T is not None else None
        except Exception:
            T_int = None
        with self._lock:
            if a_int is not None:
                if a_int in self._seen_a:
                    self.duplicate_aggtrade_ids += 1
                    flags["duplicate"] = True
                    if len(self._duplicates_sample) < 100:
                        self._duplicates_sample.append(a_int)
                else:
                    if len(self._seen_a) < self._max_tracked_ids:
                        self._seen_a.add(a_int)
                if self._max_a_seen is not None and a_int < self._max_a_seen:
                    self.out_of_order_count += 1
                    flags["gap"] = True  # fora de ordem de IDs
                if self._max_a_seen is None or a_int > self._max_a_seen:
                    # gap observado (NÃO assumir perda local — ver seção 6)
                    if self._max_a_seen is not None and a_int > self._max_a_seen + 1:
                        self._id_gaps_observed += 1
                        if len(self._id_gaps_sample) < 100:
                            self._id_gaps_sample.append(
                                {"prev_max_a": self._max_a_seen, "cur_a": a_int,
                                 "gap_size": a_int - self._max_a_seen - 1}
                            )
                    self._max_a_seen = a_int
            if T_int is not None:
                if self._max_T_seen is not None and T_int < self._max_T_seen:
                    flags["ooo_time"] = True
                if self._max_T_seen is None or T_int > self._max_T_seen:
                    self._max_T_seen = T_int
        return flags

    def snapshot_integrity(self) -> Dict[str, Any]:
        with self._lock:
            seen = len(self._seen_a)
            return {
                "unique_aggtrade_ids": seen,
                "duplicate_aggtrade_ids": self.duplicate_aggtrade_ids,
                "id_gaps_observed": self._id_gaps_observed,
                "id_gaps_sample": list(self._id_gaps_sample[:20]),
                "duplicates_sample": list(self._duplicates_sample[:20]),
                "out_of_order_count": self.out_of_order_count,
                "max_a_seen": self._max_a_seen,
                "max_T_seen": self._max_T_seen,
            }

    def snapshot_counters(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.counters)

    def build_manifest(
        self,
        extra: Optional[Dict[str, Any]] = None,
        writer_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        import config
        import config.settings as settings

        def _git(cmd: List[str]) -> str:
            try:
                out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=5)
                return out.decode("utf-8", errors="replace").strip()
            except Exception as e:
                return f"UNAVAILABLE:{e}"

        manifest: Dict[str, Any] = {
            "capture_run_id": self.capture_run_id,
            "git_commit": _git(["git", "rev-parse", "HEAD"]),
            "git_status": _git(["git", "status", "--porcelain"]),
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "symbol": getattr(config, "SYMBOL", "BTCUSDT"),
            "market_type": getattr(settings, "MARKET_TYPE", "binance_futures_perp"),
            "trade_stream": getattr(config, "STREAM_URL", ""),
            "orderbook_stream": getattr(settings, "ORDERBOOK_WS_ENDPOINT", ""),
            "OBSERVATION_MODE": os.getenv("OBSERVATION_MODE", ""),
            "EXECUTION_ENABLED": bool(getattr(settings, "EXECUTION_ENABLED", False)),
            "HYBRID_ENABLED": bool(getattr(settings, "HYBRID_ENABLED", False)),
            "AI_ENABLED": os.getenv("AI_ENABLED", "false"),
            "FORENSIC_NO_LLM": os.getenv("FORENSIC_NO_LLM", ""),
            "session_id": self.session_id,
            "config_hash": self._config_hash(),
        }
        manifest.update(self.snapshot_counters())
        manifest.update(self.snapshot_integrity())
        manifest["audit_writer_errors"] = (writer_snapshot or {}).get("audit_writer_errors", 0)
        manifest["audit_writer_drops"] = (writer_snapshot or {}).get("dropped_audit_records", 0)
        manifest["audit_writer_last_error"] = (writer_snapshot or {}).get("last_error")
        if extra:
            manifest.update(extra)
        return manifest

    def _config_hash(self) -> str:
        try:
            import config
            items = sorted(
                (k, str(v))
                for k, v in vars(config).items()
                if not k.startswith("__") and not callable(v)
            )
            blob = json.dumps(items, ensure_ascii=False, default=str).encode("utf-8")
            return hashlib.sha256(blob).hexdigest()[:16]
        except Exception:
            return "UNAVAILABLE"


CTX = ForensicContext()
