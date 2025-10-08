import time
import requests
import logging
import random
from threading import Lock
from datetime import timezone, datetime, timedelta
from typing import Optional, Dict, Any, Tuple

# Fuso horários (com fallback caso ZoneInfo não esteja disponível)
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    TZ_UTC = ZoneInfo("UTC")
    TZ_NY = ZoneInfo("America/New_York")
    TZ_SP = ZoneInfo("America/Sao_Paulo")
    _ZONEINFO_OK = True
except Exception as e:
    logging.warning(f"ZoneInfo indisponível ({e}). Usando offsets fixos (sem DST).")
    TZ_UTC = timezone.utc
    # Fallbacks fixos (atenção: sem DST dinâmico)
    TZ_NY = timezone(timedelta(hours=-4))
    TZ_SP = timezone(timedelta(hours=-3))
    _ZONEINFO_OK = False


class TimeManager:
    """
    Gerencia sincronização de tempo com a Binance e fornece utilitários para timestamps
    consistentes (UTC/NY/SP), evitando duplicidade de chaves e offsets incorretos.

    Atualizações:
    - Usa time.monotonic() para decidir re-sync (imune a ajustes do SO).
    - Sincroniza com múltiplas amostras e escolhe a de menor RTT (aprox. NTP).
    - Não zera o offset em falha; mantém o último válido (status 'degraded').
    - timespec padrão em 'milliseconds' para alinhar com epoch_ms.
    - Métodos auxiliares: time_index_from_epoch, validate_triplet, get_sync_stats,
      now_utc_iso/now_ny_iso/now_sp_iso, to_epoch_ms_from_iso, parse_any_ts.
    - Thread-safe com Lock.
    """

    BINANCE_TIME_URL = "https://fapi.binance.com/fapi/v1/time"

    def __init__(self, sync_interval_minutes: int = 30):
        self.server_time_offset_ms: int = 0                   # offset = server_ms − local_ms
        self.last_sync_mono: float = 0.0                      # relógio monotônico da última sync
        self.sync_interval_seconds: int = sync_interval_minutes * 60

        # Telemetria
        self.sync_attempts: int = 0
        self.sync_failures: int = 0
        self.last_successful_sync_ms: Optional[int] = None
        self.last_offset_ms: int = 0
        self.best_rtt_ms: Optional[int] = None
        self.last_rtt_ms: Optional[int] = None
        self.time_sync_status: str = "init"                   # init|ok|degraded|failed

        # Expor TZs como atributos para fácil reuso
        self.tz_utc = TZ_UTC
        self.tz_ny = TZ_NY
        self.tz_sp = TZ_SP

        self._lock = Lock()

        # 🔹 DEBUG: Log inicial antes da sincronização
        local_ms_before = int(time.time() * 1000)
        logging.info(f"🕐 TimeManager inicializando... Local time (ms): {local_ms_before}")
        
        # Primeira sincronização (não fatal se falhar)
        self._sync_with_binance()
        
        # 🔹 DEBUG: Log após sincronização
        local_ms_after = int(time.time() * 1000)
        synced_ms = self.now()
        offset = synced_ms - local_ms_after
        
        logging.info(f"🕐 TimeManager sincronizado:")
        logging.info(f"   Local time (ms):  {local_ms_after}")
        logging.info(f"   Synced time (ms): {synced_ms}")
        logging.info(f"   Offset: {offset} ms ({offset/1000:.2f} segundos)")
        
        # 🔹 ALERTA: Offset muito grande
        if abs(offset) > 3600000:  # > 1 hora
            logging.critical(f"⚠️ OFFSET CRÍTICO DETECTADO: {offset/3600000:.2f} horas!")
            logging.critical(f"   Isso causará problemas nos timestamps!")
        elif abs(offset) > 60000:  # > 1 minuto
            logging.warning(f"⚠️ Offset grande detectado: {offset/1000:.1f} segundos")

    # -----------------------------
    # Sincronização com a Binance (múltiplas amostras)
    # -----------------------------
    def _sample_server_time(self, timeout: Tuple[float, float] = (2.0, 3.0)) -> Optional[Dict[str, int]]:
        """
        Faz uma amostra do serverTime e retorna dict com:
        - server_ms, send_ms, recv_ms, rtt_ms, offset_ms_estimado
        """
        try:
            send_ms = int(time.time() * 1000)
            resp = requests.get(self.BINANCE_TIME_URL, timeout=timeout)
            recv_ms = int(time.time() * 1000)
            resp.raise_for_status()
            server_ms = int(resp.json().get("serverTime"))
            rtt_ms = max(0, recv_ms - send_ms)
            # Aproximação de NTP: offset = server − (send + rtt/2)
            est_local_at_server = send_ms + (rtt_ms // 2)
            offset_ms = server_ms - est_local_at_server
            
            # 🔹 DEBUG: Log da amostra
            logging.debug(f"   Amostra: server={server_ms}, local={send_ms}, rtt={rtt_ms}ms, offset={offset_ms}ms")
            
            return {
                "server_ms": server_ms,
                "send_ms": send_ms,
                "recv_ms": recv_ms,
                "rtt_ms": rtt_ms,
                "offset_ms": offset_ms,
            }
        except Exception as e:
            logging.debug(f"Falha em amostra de tempo: {e}")
            return None

    def _sync_with_binance(self) -> None:
        """
        Sincroniza com N amostras e escolhe a de menor RTT.
        Em falha: mantém último offset (status 'degraded' se houver offset anterior).
        """
        samples = []
        tries = 5
        self.sync_attempts += 1
        
        logging.info(f"🔄 Iniciando sincronização com Binance ({tries} tentativas)...")

        for i in range(tries):
            sample = self._sample_server_time()
            if sample:
                samples.append(sample)
            else:
                self.sync_failures += 1
            # pequeno jitter para evitar picos
            time.sleep(0.05 + random.uniform(0, 0.1))

        if not samples:
            with self._lock:
                # não zera offset; mantém o último
                self.time_sync_status = "degraded" if self.last_successful_sync_ms is not None else "failed"
                # Atualiza o marcador monotônico mesmo sem sucesso para evitar loop de sync
                self.last_sync_mono = time.monotonic()
            logging.warning("Falha persistente ao sincronizar com Binance. Mantendo offset anterior.")
            return

        best = min(samples, key=lambda s: s["rtt_ms"])
        
        # 🔹 DEBUG: Mostra a melhor amostra
        logging.info(f"✅ Melhor amostra: RTT={best['rtt_ms']}ms, Offset={best['offset_ms']}ms")
        
        with self._lock:
            old_offset = self.server_time_offset_ms
            self.server_time_offset_ms = int(best["offset_ms"])
            self.last_offset_ms = self.server_time_offset_ms
            self.last_rtt_ms = int(best["rtt_ms"])
            self.best_rtt_ms = self.last_rtt_ms if self.best_rtt_ms is None else min(self.best_rtt_ms, self.last_rtt_ms)
            self.last_successful_sync_ms = int(best["recv_ms"])
            self.last_sync_mono = time.monotonic()
            self.time_sync_status = "ok"
            
            # 🔹 DEBUG: Se offset mudou muito
            if abs(old_offset - self.server_time_offset_ms) > 1000:
                logging.warning(f"⚠️ Offset mudou significativamente: {old_offset}ms → {self.server_time_offset_ms}ms")

        # Alertas de drift
        if abs(self.server_time_offset_ms) > 5000:
            logging.critical(f"DRIFT DE TEMPO DETECTADO: {self.server_time_offset_ms} ms (RTT {self.last_rtt_ms} ms)")
            logging.critical("Relógio local fora de sincronia. Considere ativar NTP/sincronização automática.")
        elif abs(self.server_time_offset_ms) > 1000:
            logging.warning(f"Desvio de tempo acima de 1s: {self.server_time_offset_ms} ms (RTT {self.last_rtt_ms} ms)")
        else:
            logging.info(f"TimeManager sincronizado. Offset: {self.server_time_offset_ms} ms (RTT {self.last_rtt_ms} ms)")

    def _should_sync(self) -> bool:
        """
        Verifica se é hora de sincronizar novamente (com base no relógio monotônico).
        """
        try:
            return (time.monotonic() - self.last_sync_mono) > self.sync_interval_seconds
        except Exception:
            return True

    # -----------------------------
    # "Agora" sincronizado
    # -----------------------------
    def now(self) -> int:
        """
        Retorna o timestamp atual em milissegundos, ajustado pelo offset do servidor.
        Em caso de falha de sync, usa o último offset conhecido (status 'degraded').
        """
        try:
            if self._should_sync():
                self._sync_with_binance()
            with self._lock:
                offset = int(self.server_time_offset_ms)
            return int(time.time() * 1000) + offset
        except Exception as e:
            logging.error(f"Erro ao obter timestamp: {e}")
            return int(time.time() * 1000)

    # Alias mais explícito
    def now_ms(self) -> int:
        return self.now()

    # 🔹 NOVO: Método para forçar sincronização
    def force_sync(self) -> Dict[str, Any]:
        """Força uma sincronização imediata com a Binance e retorna estatísticas."""
        logging.info("🔄 Forçando sincronização com Binance...")
        self._sync_with_binance()
        return self.get_sync_stats()

    # -----------------------------
    # ISO helpers
    # -----------------------------
    def now_iso(self, tz=TZ_UTC, timespec: str = "milliseconds") -> str:
        try:
            ms = self.now()
            return self.format_timestamp(ms, tz=tz, timespec=timespec)
        except Exception as e:
            logging.error(f"Erro ao formatar timestamp ISO: {e}")
            return datetime.now(tz).isoformat(timespec="seconds")

    def now_utc_iso(self, timespec: str = "milliseconds") -> str:
        return self.now_iso(tz=self.tz_utc, timespec=timespec)

    def now_ny_iso(self, timespec: str = "milliseconds") -> str:
        return self.now_iso(tz=self.tz_ny, timespec=timespec)

    def now_sp_iso(self, timespec: str = "milliseconds") -> str:
        return self.now_iso(tz=self.tz_sp, timespec=timespec)

    # -----------------------------
    # Time index builders
    # -----------------------------
    def iso_triplet(self, ts_ms: Optional[int] = None, timespec: str = "milliseconds") -> Dict[str, str]:
        """
        Retorna um dicionário com as 3 representações ISO (todas derivadas do mesmo epoch):
        - timestamp_utc (+00:00)
        - timestamp_ny (America/New_York)
        - timestamp_sp (America/Sao_Paulo)
        """
        if ts_ms is None:
            ts_ms = self.now()
        return {
            "timestamp_utc": self.format_timestamp(ts_ms, tz=self.tz_utc, timespec=timespec),
            "timestamp_ny": self.format_timestamp(ts_ms, tz=self.tz_ny, timespec=timespec),
            "timestamp_sp": self.format_timestamp(ts_ms, tz=self.tz_sp, timespec=timespec),
        }

    def build_time_index(self, ts_ms: Optional[int] = None, include_local: bool = True,
                         timespec: str = "milliseconds") -> Dict[str, Any]:
        """
        Constrói o payload padrão de tempo para anexar em eventos, evitando duplicatas.
        - Sempre inclui epoch_ms e timestamp_utc
        - Opcionalmente inclui timestamp_ny e timestamp_sp
        """
        if ts_ms is None:
            ts_ms = self.now()

        payload = {
            "epoch_ms": int(ts_ms),
            "timestamp_utc": self.format_timestamp(ts_ms, tz=self.tz_utc, timespec=timespec),
        }

        if include_local:
            payload["timestamp_ny"] = self.format_timestamp(ts_ms, tz=self.tz_ny, timespec=timespec)
            payload["timestamp_sp"] = self.format_timestamp(ts_ms, tz=self.tz_sp, timespec=timespec)

        return payload

    # Alias explícito com nome semântico
    def time_index_from_epoch(self, epoch_ms: int, include_local: bool = True,
                              timespec: str = "milliseconds") -> Dict[str, Any]:
        return self.build_time_index(epoch_ms, include_local=include_local, timespec=timespec)

    def attach_timestamps(self, data: Dict[str, Any], ts_ms: Optional[int] = None,
                          include_local: bool = True, overwrite: bool = True,
                          timespec: str = "milliseconds") -> Dict[str, Any]:
        """
        Injeta campos de tempo padronizados no dicionário 'data'.
        - overwrite=True: sobrescreve valores existentes (evita inconsistência/duplicidade).
        Retorna o próprio dicionário (mutação in-place) para uso fluente.
        """
        if ts_ms is None:
            ts_ms = self.now()

        idx = self.build_time_index(ts_ms, include_local=include_local, timespec=timespec)

        for k, v in idx.items():
            if overwrite or (k not in data):
                data[k] = v

        return data

    # -----------------------------
    # Utilitários
    # -----------------------------
    def calc_age_ms(self, recent_ts_ms: int, reference_ts_ms: Optional[int] = None) -> int:
        """
        Calcula age_ms a partir de um timestamp 'recent_ts_ms' até 'reference_ts_ms'
        (padrão: agora). Não retorna negativo.
        """
        if reference_ts_ms is None:
            reference_ts_ms = self.now()
        try:
            age = int(reference_ts_ms) - int(recent_ts_ms)
            return max(0, age)
        except Exception as e:
            logging.error(f"Erro ao calcular age_ms: {e}")
            return 0

    @staticmethod
    def format_timestamp(ts_ms: int, tz=TZ_UTC, timespec: str = "milliseconds") -> str:
        """
        Formata timestamp (ms) em ISO 8601 no timezone informado.
        Fallback para 'seconds' caso o ambiente não suporte o timespec fornecido.
        """
        try:
            if not isinstance(ts_ms, (int, float)) or ts_ms < 0:
                raise ValueError(f"Timestamp inválido: {ts_ms}")
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=tz)
            try:
                return dt.isoformat(timespec=timespec)
            except Exception:
                # Fallback para ambientes sem suporte a 'milliseconds'
                return dt.isoformat(timespec="seconds")
        except Exception as e:
            logging.error(f"Erro ao formatar timestamp {ts_ms}: {e}")
            return datetime.now(tz).isoformat(timespec="seconds")

    def validate_triplet(self, epoch_ms: int, timestamp_utc: str,
                         timestamp_ny: Optional[str] = None,
                         timestamp_sp: Optional[str] = None,
                         tol_seconds: float = 2.0) -> Dict[str, Any]:
        """
        Valida se os timestamps ISO batem com o epoch_ms dentro de uma tolerância.
        Retorna diffs em ms e um ok geral.
        """
        def _to_epoch(iso_str: str) -> Optional[int]:
            try:
                iso_str = iso_str.replace("Z", "+00:00")
                dt = datetime.fromisoformat(iso_str)
                return int(dt.timestamp() * 1000)
            except Exception:
                return None

        diffs = {}
        ok = True

        utc_epoch = _to_epoch(timestamp_utc)
        if utc_epoch is None:
            ok = False
            diffs["utc_error"] = "parse_fail"
        else:
            diffs["utc_diff_ms"] = abs(utc_epoch - int(epoch_ms))
            ok = ok and (diffs["utc_diff_ms"] <= tol_seconds * 1000)

        if timestamp_ny:
            ny_epoch = _to_epoch(timestamp_ny)
            if ny_epoch is None:
                ok = False
                diffs["ny_error"] = "parse_fail"
            else:
                diffs["ny_diff_ms"] = abs(ny_epoch - int(epoch_ms))
                ok = ok and (diffs["ny_diff_ms"] <= tol_seconds * 1000)

        if timestamp_sp:
            sp_epoch = _to_epoch(timestamp_sp)
            if sp_epoch is None:
                ok = False
                diffs["sp_error"] = "parse_fail"
            else:
                diffs["sp_diff_ms"] = abs(sp_epoch - int(epoch_ms))
                ok = ok and (diffs["sp_diff_ms"] <= tol_seconds * 1000)

        return {"ok": ok, "diffs": diffs}

    @staticmethod
    def to_epoch_ms_from_iso(iso_str: str) -> Optional[int]:
        """
        Converte string ISO 8601 para epoch em ms. Retorna None em falha.
        """
        try:
            iso_str = iso_str.replace("Z", "+00:00")
            dt = datetime.fromisoformat(iso_str)
            return int(dt.timestamp() * 1000)
        except Exception:
            return None

    @staticmethod
    def parse_any_ts(value: Any) -> Optional[int]:
        """
        Tenta interpretar 'value' como epoch_ms (int/float/str) ou ISO 8601.
        Retorna epoch_ms ou None.
        """
        try:
            if value is None:
                return None
            if isinstance(value, (int, float)):
                v = int(value)
                return v if v >= 0 else None
            if isinstance(value, str):
                v = value.strip()
                if v.isdigit():
                    return int(v)
                # Tenta ISO
                v = v.replace("Z", "+00:00")
                dt = datetime.fromisoformat(v)
                return int(dt.timestamp() * 1000)
            return None
        except Exception:
            return None

    # -----------------------------
    # Telemetria
    # -----------------------------
    def get_sync_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "status": self.time_sync_status,
                "server_time_offset_ms": int(self.server_time_offset_ms),
                "last_offset_ms": int(self.last_offset_ms),
                "last_rtt_ms": self.last_rtt_ms,
                "best_rtt_ms": self.best_rtt_ms,
                "last_successful_sync_ms": self.last_successful_sync_ms,
                "sync_attempts": self.sync_attempts,
                "sync_failures": self.sync_failures,
                "zoneinfo_ok": _ZONEINFO_OK,
            }