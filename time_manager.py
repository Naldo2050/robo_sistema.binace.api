import time
import requests
import logging
import random
import numpy as np
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

# ═══════════════════════════════════════════════════════════════
# CALENDÁRIO DE FERIADOS — Mercados TradFi (afeta liquidez crypto)
# Atualizar anualmente. Crypto opera 24/7 mas liquidez cai
# significativamente em feriados dos EUA.
# ═══════════════════════════════════════════════════════════════
US_MARKET_HOLIDAYS = {
    2025: {
        "2025-01-01": "New Year's Day",
        "2025-01-20": "MLK Day",
        "2025-02-17": "Presidents Day",
        "2025-04-18": "Good Friday",
        "2025-05-26": "Memorial Day",
        "2025-06-19": "Juneteenth",
        "2025-07-04": "Independence Day",
        "2025-09-01": "Labor Day",
        "2025-11-27": "Thanksgiving",
        "2025-11-28": "Black Friday (early close)",
        "2025-12-25": "Christmas",
    },
    2026: {
        "2026-01-01": "New Year's Day",
        "2026-01-19": "MLK Day",
        "2026-02-16": "Presidents Day",
        "2026-04-03": "Good Friday",
        "2026-05-25": "Memorial Day",
        "2026-06-19": "Juneteenth",
        "2026-07-03": "Independence Day (observed)",
        "2026-09-07": "Labor Day",
        "2026-11-26": "Thanksgiving",
        "2026-11-27": "Black Friday (early close)",
        "2026-12-25": "Christmas",
    },
    2027: {
        "2027-01-01": "New Year's Day",
        "2027-01-18": "MLK Day",
        "2027-02-15": "Presidents Day",
        "2027-03-26": "Good Friday",
        "2027-05-31": "Memorial Day",
        "2027-06-18": "Juneteenth (observed)",
        "2027-07-05": "Independence Day (observed)",
        "2027-09-06": "Labor Day",
        "2027-11-25": "Thanksgiving",
        "2027-11-26": "Black Friday (early close)",
        "2027-12-24": "Christmas (observed)",
    },
}


class TimeManager:
    """
    Gerencia sincronização de tempo com a Binance e fornece utilitários para timestamps
    consistentes (UTC/NY/SP), evitando duplicidade de chaves e offsets incorretos.

    Melhorias v2.1.2 (CORREÇÃO CRÍTICA):
    - ✅ Adicionado método from_timestamp_ms() que estava faltando
    - ✅ Conversão de epoch_ms para datetime em timezone específico
    
    Melhorias v2.1.1 (mantidas):
    - ✅ Previne loop infinito de re-sincronização
    - ✅ Detecta e aceita latência de rede estável (< 1000ms)
    - ✅ Limite de 3 tentativas de correção automática
    - ✅ Histórico de offsets para detectar estabilidade
    - ✅ Logs informativos sem poluir console
    - ✅ Recomendação de ajuste de configuração
    """

    BINANCE_TIME_URL = "https://fapi.binance.com/fapi/v1/time"
    MAX_ACCEPTABLE_RTT_MS = 2000  # Rejeita amostras com RTT > 2s
    CRITICAL_OFFSET_MS = 3600000  # 1 hora
    WARNING_OFFSET_MS = 60000     # 1 minuto
    MAX_CORRECTION_ATTEMPTS = 3   # Limite de tentativas de correção

    # ------------------ SINGLETON GLOBAL ------------------
    _instance: Optional["TimeManager"] = None
    _instance_lock: Lock = Lock()

    def __new__(cls, *args, **kwargs):
        """
        Implementa padrão Singleton: qualquer chamada a TimeManager() retorna
        SEMPRE a mesma instância global.
        """
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    # ------------------------------------------------------

    def __init__(self, 
                 sync_interval_minutes: int = 30,
                 max_init_attempts: int = 3,
                 max_acceptable_offset_ms: int = 600,
                 num_sync_samples: int = 5):
        """
        Inicializa o TimeManager com sincronização robusta.
        
        Args:
            sync_interval_minutes: Intervalo entre re-sincronizações automáticas
            max_init_attempts: Máximo de tentativas na inicialização
            max_acceptable_offset_ms: Offset máximo aceitável (padrão 600ms)
            num_sync_samples: Número de amostras por sincronização
        """
        # --- GUARDA DE SINGLETON: evita reexecutar __init__ ---
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        # ------------------------------------------------------

        # Configuração
        self.sync_interval_seconds: int = sync_interval_minutes * 60
        self.max_acceptable_offset_ms: int = max_acceptable_offset_ms
        self.max_init_attempts: int = max_init_attempts
        self.num_sync_samples: int = num_sync_samples
        
        # Estado de sincronização
        self.server_time_offset_ms: int = 0
        self.last_sync_mono: float = 0.0
        
        # Telemetria
        self.sync_attempts: int = 0
        self.sync_failures: int = 0
        self.last_successful_sync_ms: Optional[int] = None
        self.last_offset_ms: int = 0
        self.best_rtt_ms: Optional[int] = None
        self.last_rtt_ms: Optional[int] = None
        self.time_sync_status: str = "init"  # init|ok|degraded|failed
        self.auto_corrections: int = 0

        # Controle de loop infinito
        self._correction_attempts = 0
        self._last_offset_history = []

        # Timezones
        self.tz_utc = TZ_UTC
        self.tz_ny = TZ_NY
        self.tz_sp = TZ_SP

        # Thread safety
        self._lock = Lock()

        # Inicialização com retry
        self._initialize_sync()

    # ========================================================================
    # INICIALIZAÇÃO
    # ========================================================================
    
    def _initialize_sync(self) -> None:
        """Executa sincronização inicial com múltiplas tentativas e validação."""
        local_ms_before = int(time.time() * 1000)
        logging.info("=" * 80)
        logging.info("🕐 TIMEMANAGER v2.1.2 - INICIALIZANDO")
        logging.info("=" * 80)
        logging.info(f"   Tempo local:     {local_ms_before} ms")
        logging.info(f"   Timezone UTC:    {self.tz_utc}")
        logging.info(f"   Timezone NY:     {self.tz_ny}")
        logging.info(f"   Timezone SP:     {self.tz_sp}")
        logging.info(f"   ZoneInfo:        {'✅ Disponível' if _ZONEINFO_OK else '⚠️ Indisponível (usando offsets fixos)'}")
        logging.info(f"   Max offset:      {self.max_acceptable_offset_ms} ms")
        logging.info(f"   Sync samples:    {self.num_sync_samples}")
        logging.info(f"   Sync interval:   {self.sync_interval_seconds} s ({self.sync_interval_seconds // 60} min)")
        logging.info("=" * 80)
        
        success = False
        for attempt in range(self.max_init_attempts):
            try:
                logging.info(f"🔄 Tentativa {attempt + 1}/{self.max_init_attempts}")
                self._sync_with_binance()
                
                if self.time_sync_status == "ok":
                    success = True
                    logging.info(f"✅ Sincronização bem-sucedida na tentativa {attempt + 1}")
                    break
                else:
                    logging.warning(f"⚠️ Tentativa {attempt + 1} falhou - Status: {self.time_sync_status}")
                    if attempt < self.max_init_attempts - 1:
                        backoff = 2 ** attempt
                        logging.info(f"   Aguardando {backoff}s antes da próxima tentativa...")
                        time.sleep(backoff)
                        
            except Exception as e:
                logging.error(f"❌ Erro na tentativa {attempt + 1}: {e}")
                if attempt < self.max_init_attempts - 1:
                    backoff = 2 ** attempt
                    logging.info(f"   Aguardando {backoff}s antes da próxima tentativa...")
                    time.sleep(backoff)
        
        if not success:
            logging.critical("=" * 80)
            logging.critical("⛔ FALHA CRÍTICA NA SINCRONIZAÇÃO")
            logging.critical("=" * 80)
            logging.critical("   Não foi possível sincronizar com a Binance após múltiplas tentativas")
            logging.critical("   O sistema continuará usando tempo local (NÃO RECOMENDADO para trading)")
            logging.critical("   ")
            logging.critical("   AÇÕES RECOMENDADAS:")
            logging.critical("   1. Verifique sua conexão com a internet")
            logging.critical("   2. Verifique se a Binance está acessível")
            logging.critical("   3. Sincronize o relógio do sistema:")
            logging.critical("      - Windows: w32tm /resync")
            logging.critical("      - Linux:   sudo ntpdate pool.ntp.org")
            logging.critical("      - macOS:   sudo sntp -sS pool.ntp.org")
            logging.critical("=" * 80)
            self.time_sync_status = "failed"
        
        # Diagnóstico completo
        logging.info("")
        self.diagnose()
        logging.info("=" * 80)

    # ========================================================================
    # SINCRONIZAÇÃO COM BINANCE
    # ========================================================================
    
    def _sample_server_time(self, timeout: Tuple[float, float] = (2.0, 3.0)) -> Optional[Dict[str, int]]:
        """
        Coleta uma amostra do tempo do servidor Binance.
        
        Returns:
            Dict com server_ms, send_ms, recv_ms, rtt_ms, offset_ms ou None em falha
            
        Rejeita amostras com RTT > MAX_ACCEPTABLE_RTT_MS
        """
        try:
            send_ms = int(time.time() * 1000)
            resp = requests.get(self.BINANCE_TIME_URL, timeout=timeout)
            recv_ms = int(time.time() * 1000)
            
            resp.raise_for_status()
            server_ms = int(resp.json().get("serverTime"))
            rtt_ms = max(0, recv_ms - send_ms)
            
            if rtt_ms > self.MAX_ACCEPTABLE_RTT_MS:
                logging.warning(
                    f"⚠️ Amostra rejeitada: RTT muito alto "
                    f"({rtt_ms}ms > {self.MAX_ACCEPTABLE_RTT_MS}ms)"
                )
                return None
            
            est_local_at_server = send_ms + (rtt_ms / 2.0)
            offset_ms = int(server_ms - est_local_at_server)
            
            logging.debug(
                f"   ✓ Amostra OK: RTT={rtt_ms}ms, Offset={offset_ms}ms, "
                f"Server={server_ms}, Local={send_ms}"
            )
            
            return {
                "server_ms": server_ms,
                "send_ms": send_ms,
                "recv_ms": recv_ms,
                "rtt_ms": rtt_ms,
                "offset_ms": offset_ms,
            }
            
        except requests.Timeout:
            logging.debug("   ✗ Timeout na requisição")
            return None
        except requests.RequestException as e:
            logging.debug(f"   ✗ Erro de rede: {e}")
            return None
        except Exception as e:
            logging.debug(f"   ✗ Erro inesperado: {e}")
            return None

    def _sync_with_binance(self) -> None:
        """
        Sincroniza com N amostras e escolhe a de menor RTT (melhor qualidade).
        Em falha total: mantém último offset válido (status 'degraded').
        """
        samples = []
        self.sync_attempts += 1
        
        logging.info(f"🔄 Iniciando sincronização com Binance ({self.num_sync_samples} amostras)...")

        for i in range(self.num_sync_samples):
            sample = self._sample_server_time()
            if sample:
                samples.append(sample)
            else:
                self.sync_failures += 1
            
            if i < self.num_sync_samples - 1:
                time.sleep(0.05 + random.uniform(0, 0.1))

        if not samples:
            with self._lock:
                if self.last_successful_sync_ms is not None:
                    self.time_sync_status = "degraded"
                    logging.warning(
                        "⚠️ Falha na sincronização. Mantendo offset anterior: "
                        f"{self.server_time_offset_ms}ms (modo degradado)"
                    )
                else:
                    self.time_sync_status = "failed"
                    logging.error("❌ Falha na sincronização e sem offset anterior disponível")
                
                self.last_sync_mono = time.monotonic()
            return

        best = min(samples, key=lambda s: s["rtt_ms"])
        
        if len(samples) > 1:
            rtts = [s["rtt_ms"] for s in samples]
            offsets = [s["offset_ms"] for s in samples]
            logging.debug("   📊 Estatísticas das amostras:")
            logging.debug(f"      RTTs:    {rtts} (média: {np.mean(rtts):.1f}ms, std: {np.std(rtts):.1f}ms)")
            logging.debug(f"      Offsets: {offsets} (média: {np.mean(offsets):.1f}ms, std: {np.std(offsets):.1f}ms)")
        
        logging.info(
            f"✅ Melhor amostra selecionada: "
            f"RTT={best['rtt_ms']}ms, Offset={best['offset_ms']}ms"
        )
        
        with self._lock:
            old_offset = self.server_time_offset_ms
            self.server_time_offset_ms = int(best["offset_ms"])
            self.last_offset_ms = self.server_time_offset_ms
            self.last_rtt_ms = int(best["rtt_ms"])
            self.best_rtt_ms = (
                self.last_rtt_ms if self.best_rtt_ms is None 
                else min(self.best_rtt_ms, self.last_rtt_ms)
            )
            self.last_successful_sync_ms = int(best["recv_ms"])
            self.last_sync_mono = time.monotonic()
            self.time_sync_status = "ok"
            
            offset_change = abs(old_offset - self.server_time_offset_ms)
            if offset_change > 100 and old_offset != 0:
                logging.warning(
                    f"⚠️ Offset mudou significativamente: "
                    f"{old_offset}ms → {self.server_time_offset_ms}ms "
                    f"(Δ={offset_change}ms)"
                )

        self._validate_offset()

    def _should_sync(self) -> bool:
        """Verifica se é hora de sincronizar novamente."""
        try:
            elapsed = time.monotonic() - self.last_sync_mono
            return elapsed > self.sync_interval_seconds
        except Exception:
            return True

    # ========================================================================
    # VALIDAÇÃO E NTP
    # ========================================================================
    
    def _validate_offset(self) -> None:
        """
        Previne loop infinito de re-sincronização.
        
        Aceita offsets estáveis até 1000ms como "aceitáveis para uso"
        mesmo que acima do ideal de 600ms.
        """
        with self._lock:
            offset_abs = abs(self.server_time_offset_ms)
            
            self._last_offset_history.append(offset_abs)
            if len(self._last_offset_history) > 10:
                self._last_offset_history.pop(0)
        
        if offset_abs > self.CRITICAL_OFFSET_MS:
            logging.critical("=" * 80)
            logging.critical(f"⛔ OFFSET CRÍTICO: {offset_abs/1000:.1f}s")
            logging.critical("=" * 80)
            logging.critical("   🔧 AÇÃO NECESSÁRIA:")
            logging.critical("   1. PARE o bot")
            logging.critical("   2. Sincronize o relógio do sistema:")
            logging.critical("      - Windows: w32tm /resync")
            logging.critical("      - Linux:   sudo ntpdate pool.ntp.org")
            logging.critical("      - macOS:   sudo sntp -sS pool.ntp.org")
            logging.critical("=" * 80)
            
            logging.info("🔄 Tentando sincronização NTP automática...")
            if self._try_system_ntp_sync():
                logging.info("✅ NTP bem-sucedido. Re-sincronizando com Binance...")
                self._sync_with_binance()
            else:
                logging.error("❌ NTP falhou. Intervenção manual necessária.")
        
        elif offset_abs > self.WARNING_OFFSET_MS:
            logging.warning("=" * 80)
            logging.warning(f"⚠️ OFFSET ALTO: {offset_abs/1000:.1f}s")
            logging.warning("=" * 80)
            logging.warning("   Comandos para sincronizar:")
            logging.warning("   - Windows: w32tm /resync")
            logging.warning("   - Linux:   sudo ntpdate pool.ntp.org")
            logging.warning("=" * 80)
            
            logging.info("🔄 Tentando NTP automático...")
            if self._try_system_ntp_sync():
                logging.info("✅ NTP bem-sucedido. Re-sincronizando...")
                self._sync_with_binance()
        
        elif offset_abs > self.max_acceptable_offset_ms:
            is_stable = self._is_offset_stable(offset_abs)
            
            if offset_abs <= 1000 and is_stable:
                if self._correction_attempts == 0:
                    logging.warning(
                        f"⚠️ Offset {offset_abs}ms > {self.max_acceptable_offset_ms}ms "
                        f"mas ESTÁVEL e < 1000ms"
                    )
                    logging.warning("   Isso parece ser LATÊNCIA DE REDE (não erro de relógio).")
                    logging.warning("   ✅ ACEITANDO offset. Sistema operará normalmente.")
                    logging.warning(
                        f"   💡 Para reduzir avisos, considere ajustar "
                        f"'max_acceptable_offset_ms' para {offset_abs + 100}ms no config."
                    )
                self._correction_attempts = 0
                return
            
            if self._correction_attempts >= self.MAX_CORRECTION_ATTEMPTS:
                if offset_abs <= 1000:
                    logging.warning(
                        f"⚠️ Offset {offset_abs}ms não corrigível após "
                        f"{self.MAX_CORRECTION_ATTEMPTS} tentativas."
                    )
                    logging.warning("   Provável causa: LATÊNCIA DE REDE (não erro de relógio).")
                    logging.warning("   ✅ ACEITANDO offset. Sistema operará normalmente.")
                    logging.warning(
                        f"   💡 Recomendação: Ajuste 'max_acceptable_offset_ms' "
                        f"para {offset_abs + 100}ms no config.py"
                    )
                    self._correction_attempts = 0
                    return
                else:
                    logging.error(
                        f"❌ Offset {offset_abs}ms muito alto e não corrigível."
                    )
                    logging.error("   Sistema continuará mas pode haver problemas.")
                    return
            
            self._correction_attempts += 1
            logging.warning(
                f"⚠️ Offset {offset_abs}ms > {self.max_acceptable_offset_ms}ms "
                f"(tentativa {self._correction_attempts}/{self.MAX_CORRECTION_ATTEMPTS})"
            )
            logging.info("🔄 Tentando re-sincronização...")
            
            old_offset = self.server_time_offset_ms
            self._sync_with_binance()
            
            new_offset_abs = abs(self.server_time_offset_ms)
            
            if new_offset_abs <= self.max_acceptable_offset_ms:
                logging.info(
                    f"✅ Offset corrigido: {old_offset}ms → {self.server_time_offset_ms}ms"
                )
                self.auto_corrections += 1
                self._correction_attempts = 0
            else:
                logging.warning(
                    f"⚠️ Re-sync não melhorou: {old_offset}ms → {self.server_time_offset_ms}ms"
                )
        
        else:
            logging.info(f"✅ Offset dentro do limite aceitável: {offset_abs}ms")
            self._correction_attempts = 0
    
    def _is_offset_stable(self, current_offset: int) -> bool:
        """Verifica se o offset é estável (não está aumentando)."""
        if len(self._last_offset_history) < 3:
            return False
        
        recent = self._last_offset_history[-3:]
        max_offset = max(recent)
        min_offset = min(recent)
        variation = max_offset - min_offset
        
        return variation < 50

    def _try_system_ntp_sync(self) -> bool:
        """Tenta sincronizar o relógio do sistema usando NTP."""
        import platform
        import subprocess
        
        try:
            system = platform.system()
            
            if system == "Windows":
                logging.info("🔄 Tentando sincronizar com NTP (Windows)...")
                result = subprocess.run(
                    ["w32tm", "/resync"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    logging.info("✅ Sincronização NTP do Windows bem-sucedida")
                    return True
                else:
                    logging.warning(f"⚠️ Falha na sincronização NTP: {result.stderr}")
            
            elif system == "Linux":
                logging.info("🔄 Tentando sincronizar com NTP (Linux)...")
                commands = [
                    ["chronyc", "makestep"],
                    ["sudo", "ntpdate", "-u", "pool.ntp.org"],
                    ["timedatectl", "set-ntp", "true"],
                ]
                
                for cmd in commands:
                    try:
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        if result.returncode == 0:
                            logging.info(f"✅ Sincronização NTP bem-sucedida: {' '.join(cmd)}")
                            return True
                    except FileNotFoundError:
                        continue
                    except Exception as e:
                        logging.debug(f"   Falha em {' '.join(cmd)}: {e}")
                        continue
            
            elif system == "Darwin":  # macOS
                logging.info("🔄 Tentando sincronizar com NTP (macOS)...")
                result = subprocess.run(
                    ["sudo", "sntp", "-sS", "pool.ntp.org"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    logging.info("✅ Sincronização NTP do macOS bem-sucedida")
                    return True
        
        except subprocess.TimeoutExpired:
            logging.error("❌ Timeout ao executar comando NTP")
        except Exception as e:
            logging.error(f"❌ Erro ao tentar sincronizar NTP do sistema: {e}")
        
        logging.warning("⚠️ Sincronização NTP automática falhou")
        return False

    # ========================================================================
    # TIMESTAMP "AGORA"
    # ========================================================================
    
    def now(self) -> int:
        """Retorna o timestamp atual em milissegundos, ajustado pelo offset da Binance."""
        try:
            if self._should_sync():
                logging.debug("⏰ Tempo de re-sincronização automática")
                self._sync_with_binance()
            
            with self._lock:
                offset = int(self.server_time_offset_ms)
            
            return int(time.time() * 1000) + offset
            
        except Exception as e:
            logging.error(f"❌ Erro ao obter timestamp: {e}")
            return int(time.time() * 1000)

    def now_ms(self) -> int:
        """Alias explícito para now()."""
        return self.now()

    def force_sync(self) -> Dict[str, Any]:
        """Força uma sincronização imediata com a Binance."""
        logging.info("🔄 Forçando sincronização com Binance...")
        self._sync_with_binance()
        return self.get_sync_stats()

    # ========================================================================
    # MÉTODO from_timestamp_ms()
    # ========================================================================
    
    def from_timestamp_ms(self, ts_ms: int, tz=None) -> datetime:
        """Converte timestamp em milissegundos para objeto datetime."""
        if tz is None:
            tz = self.tz_utc
        
        try:
            ts_sec = ts_ms / 1000.0
            dt = datetime.fromtimestamp(ts_sec, tz=tz)
            return dt
        except Exception as e:
            logging.error(f"❌ Erro ao converter timestamp {ts_ms}: {e}")
            return datetime.now(tz)

    # ========================================================================
    # ISO HELPERS
    # ========================================================================
    
    def now_iso(self, tz=TZ_UTC, timespec: str = "milliseconds") -> str:
        """Retorna timestamp atual em formato ISO 8601."""
        try:
            ms = self.now()
            return self.format_timestamp(ms, tz=tz, timespec=timespec)
        except Exception as e:
            logging.error(f"❌ Erro ao formatar timestamp ISO: {e}")
            return datetime.now(tz).isoformat(timespec="seconds")

    def now_utc_iso(self, timespec: str = "milliseconds") -> str:
        """Timestamp atual em UTC (ISO 8601)."""
        return self.now_iso(tz=self.tz_utc, timespec=timespec)

    def now_ny_iso(self, timespec: str = "milliseconds") -> str:
        """Timestamp atual em New York (ISO 8601)."""
        return self.now_iso(tz=self.tz_ny, timespec=timespec)

    def now_sp_iso(self, timespec: str = "milliseconds") -> str:
        """Timestamp atual em São Paulo (ISO 8601)."""
        return self.now_iso(tz=self.tz_sp, timespec=timespec)

    # ========================================================================
    # TIME INDEX BUILDERS
    # ========================================================================
    
    def iso_triplet(self, ts_ms: Optional[int] = None, timespec: str = "milliseconds") -> Dict[str, str]:
        """Retorna dicionário com timestamps ISO em 3 timezones."""
        if ts_ms is None:
            ts_ms = self.now()
        
        return {
            "timestamp_utc": self.format_timestamp(ts_ms, tz=self.tz_utc, timespec=timespec),
            "timestamp_ny": self.format_timestamp(ts_ms, tz=self.tz_ny, timespec=timespec),
            "timestamp_sp": self.format_timestamp(ts_ms, tz=self.tz_sp, timespec=timespec),
        }

    def build_time_index(self, 
                        ts_ms: Optional[int] = None, 
                        include_local: bool = True,
                        timespec: str = "milliseconds") -> Dict[str, Any]:
        """Constrói payload padrão de tempo para eventos."""
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

    def time_index_from_epoch(self, 
                             epoch_ms: int, 
                             include_local: bool = True,
                             timespec: str = "milliseconds") -> Dict[str, Any]:
        """Alias semântico para build_time_index."""
        return self.build_time_index(epoch_ms, include_local=include_local, timespec=timespec)

    def attach_timestamps(self, 
                         data: Dict[str, Any], 
                         ts_ms: Optional[int] = None,
                         include_local: bool = True, 
                         overwrite: bool = True,
                         timespec: str = "milliseconds") -> Dict[str, Any]:
        """Injeta campos de tempo padronizados no dicionário (in-place)."""
        if ts_ms is None:
            ts_ms = self.now()

        idx = self.build_time_index(ts_ms, include_local=include_local, timespec=timespec)

        for k, v in idx.items():
            if overwrite or (k not in data):
                data[k] = v

        return data

    # ========================================================================
    # UTILITÁRIOS
    # ========================================================================
    
    def calc_age_ms(self, recent_ts_ms: int, reference_ts_ms: Optional[int] = None) -> int:
        """Calcula idade (age_ms) de um timestamp até referência (padrão: agora)."""
        if reference_ts_ms is None:
            reference_ts_ms = self.now()
        
        try:
            recent = int(recent_ts_ms)
            reference = int(reference_ts_ms)
            
            if recent <= 0 or reference <= 0:
                logging.warning(
                    f"⚠️ Timestamp inválido ao calcular age: "
                    f"recent={recent}, reference={reference}"
                )
                return 0
            
            age = reference - recent
            
            if age < 0:
                logging.warning(
                    f"⚠️ Idade negativa detectada: {age}ms "
                    f"(recent={recent} > reference={reference}). "
                    f"Retornando 0."
                )
                return 0
            
            return age
            
        except Exception as e:
            logging.error(f"❌ Erro ao calcular age_ms: {e}")
            return 0

    @staticmethod
    def format_timestamp(ts_ms: int, tz=TZ_UTC, timespec: str = "milliseconds") -> str:
        """Formata timestamp (ms) em ISO 8601 no timezone especificado."""
        try:
            if not isinstance(ts_ms, (int, float)) or ts_ms < 0:
                raise ValueError(f"Timestamp inválido: {ts_ms}")
            
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=tz)
            
            try:
                return dt.isoformat(timespec=timespec)
            except TypeError:
                return dt.isoformat(timespec="seconds")
                
        except Exception as e:
            logging.error(f"❌ Erro ao formatar timestamp {ts_ms}: {e}")
            return datetime.now(tz).isoformat(timespec="seconds")

    def validate_triplet(self, 
                        epoch_ms: int, 
                        timestamp_utc: str,
                        timestamp_ny: Optional[str] = None,
                        timestamp_sp: Optional[str] = None,
                        tol_seconds: float = 2.0) -> Dict[str, Any]:
        """Valida se timestamps ISO correspondem ao epoch_ms."""
        def _to_epoch(iso_str: str) -> Optional[int]:
            try:
                iso_str = iso_str.replace("Z", "+00:00")
                dt = datetime.fromisoformat(iso_str)
                return int(dt.timestamp() * 1000)
            except Exception:
                return None

        diffs = {}
        ok = True
        tol_ms = tol_seconds * 1000

        utc_epoch = _to_epoch(timestamp_utc)
        if utc_epoch is None:
            ok = False
            diffs["utc_error"] = "parse_fail"
        else:
            diffs["utc_diff_ms"] = abs(utc_epoch - int(epoch_ms))
            ok = ok and (diffs["utc_diff_ms"] <= tol_ms)

        if timestamp_ny:
            ny_epoch = _to_epoch(timestamp_ny)
            if ny_epoch is None:
                ok = False
                diffs["ny_error"] = "parse_fail"
            else:
                diffs["ny_diff_ms"] = abs(ny_epoch - int(epoch_ms))
                ok = ok and (diffs["ny_diff_ms"] <= tol_ms)

        if timestamp_sp:
            sp_epoch = _to_epoch(timestamp_sp)
            if sp_epoch is None:
                ok = False
                diffs["sp_error"] = "parse_fail"
            else:
                diffs["sp_diff_ms"] = abs(sp_epoch - int(epoch_ms))
                ok = ok and (diffs["sp_diff_ms"] <= tol_ms)

        return {"ok": ok, "diffs": diffs}

    @staticmethod
    def to_epoch_ms_from_iso(iso_str: str) -> Optional[int]:
        """Converte string ISO 8601 para epoch em ms."""
        try:
            iso_str = iso_str.replace("Z", "+00:00")
            dt = datetime.fromisoformat(iso_str)
            return int(dt.timestamp() * 1000)
        except Exception:
            return None

    @staticmethod
    def parse_any_ts(value: Any) -> Optional[int]:
        """Tenta interpretar value como epoch_ms (int/float/str) ou ISO 8601."""
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
                
                v = v.replace("Z", "+00:00")
                dt = datetime.fromisoformat(v)
                return int(dt.timestamp() * 1000)
            
            return None
            
        except Exception:
            return None

    # ========================================================================
    # DIAGNÓSTICO E TELEMETRIA
    # ========================================================================
    
    def diagnose(self) -> Dict[str, Any]:
        """Executa diagnóstico completo do sistema de tempo."""
        logging.info("🔍 DIAGNÓSTICO DO SISTEMA DE TEMPO")
        logging.info("-" * 80)
        
        local_time_ms = int(time.time() * 1000)
        synced_time_ms = self.now()
        current_offset = synced_time_ms - local_time_ms
        
        logging.info("⏰ Timestamps:")
        logging.info(f"   Local time (ms):  {local_time_ms}")
        logging.info(f"   Synced time (ms): {synced_time_ms}")
        logging.info(f"   Current offset:   {current_offset} ms ({current_offset/1000:.2f}s)")
        
        now_utc = datetime.now(self.tz_utc)
        now_ny = datetime.now(self.tz_ny)
        now_sp = datetime.now(self.tz_sp)
        
        logging.info("")
        logging.info("🌍 Timezones:")
        logging.info(f"   UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        logging.info(f"   NY:  {now_ny.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        logging.info(f"   SP:  {now_sp.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        ny_offset = now_ny.utcoffset()
        sp_offset = now_sp.utcoffset()
        utc_offset_ny = (ny_offset.total_seconds() / 3600) if ny_offset is not None else 0
        utc_offset_sp = (sp_offset.total_seconds() / 3600) if sp_offset is not None else 0
        
        logging.info(f"   Offset NY vs UTC: {utc_offset_ny:+.1f} horas")
        logging.info(f"   Offset SP vs UTC: {utc_offset_sp:+.1f} horas")
        
        ny_ok = utc_offset_ny in [-5, -4]
        sp_ok = utc_offset_sp == -3
        
        if ny_ok:
            logging.info(f"   ✅ Offset NY correto: {utc_offset_ny:+.1f} horas")
        else:
            logging.error(f"   ❌ Offset NY incorreto: {utc_offset_ny:+.1f} horas (esperado: -5 ou -4)")
        
        if sp_ok:
            logging.info(f"   ✅ Offset SP correto: {utc_offset_sp:+.1f} horas")
        else:
            logging.error(f"   ❌ Offset SP incorreto: {utc_offset_sp:+.1f} horas (esperado: -3)")
        
        stats = self.get_sync_stats()
        
        logging.info("")
        logging.info("📊 Estatísticas de Sincronização:")
        logging.info(f"   Status:                {stats['status']}")
        logging.info(f"   Server offset:         {stats['server_time_offset_ms']} ms")
        logging.info(f"   Last RTT:              {stats['last_rtt_ms']} ms")
        logging.info(f"   Best RTT:              {stats['best_rtt_ms']} ms")
        logging.info(f"   Sync attempts:         {stats['sync_attempts']}")
        logging.info(f"   Sync failures:         {stats['sync_failures']}")
        logging.info(f"   Auto corrections:      {stats['auto_corrections']}")
        
        if stats['sync_attempts'] > 0:
            success_rate = ((stats['sync_attempts'] - stats['sync_failures']) / stats['sync_attempts']) * 100
            logging.info(f"   Success rate:          {success_rate:.1f}%")
        
        logging.info(f"   ZoneInfo available:    {'✅ Yes' if stats['zoneinfo_ok'] else '⚠️ No (using fixed offsets)'}")
        
        logging.info("-" * 80)
        
        diag = {
            "local_time_ms": local_time_ms,
            "synced_time_ms": synced_time_ms,
            "current_offset_ms": current_offset,
            "sync_stats": stats,
            "timezones": {
                "utc": now_utc.isoformat(),
                "ny": now_ny.isoformat(),
                "sp": now_sp.isoformat(),
                "ny_offset_hours": utc_offset_ny,
                "sp_offset_hours": utc_offset_sp,
                "ny_ok": ny_ok,
                "sp_ok": sp_ok,
            },
            "zoneinfo_available": _ZONEINFO_OK,
        }
        
        return diag

    def get_sync_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de sincronização."""
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
                "auto_corrections": self.auto_corrections,
                "zoneinfo_ok": _ZONEINFO_OK,
            }

    def get_market_calendar_context(self, dt: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Retorna contexto de calendário de mercado.
        Crypto opera 24/7 mas liquidez cai em feriados TradFi.
        """
        if dt is None:
            dt = datetime.now(TZ_NY)
        
        date_str = dt.strftime("%Y-%m-%d")
        year = dt.year
        
        year_holidays = US_MARKET_HOLIDAYS.get(year, {})
        
        is_holiday = date_str in year_holidays
        holiday_name = year_holidays.get(date_str)
        is_weekend = dt.weekday() >= 5
        day_of_week = dt.strftime("%A")
        
        # Verificar se amanhã é feriado
        tomorrow = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
        is_pre_holiday = tomorrow in year_holidays
        
        # Verificar se ontem foi feriado
        yesterday = (dt - timedelta(days=1)).strftime("%Y-%m-%d")
        is_post_holiday = yesterday in year_holidays
        
        # Classificar liquidez esperada
        if is_holiday:
            expected_liquidity = "VERY_LOW"
        elif is_weekend and (is_pre_holiday or is_post_holiday):
            expected_liquidity = "VERY_LOW"
        elif is_weekend:
            expected_liquidity = "LOW"
        elif is_pre_holiday:
            expected_liquidity = "REDUCED"
        elif is_post_holiday:
            expected_liquidity = "REDUCED"
        elif day_of_week == "Sunday":
            expected_liquidity = "LOW"
        else:
            expected_liquidity = "NORMAL"
        
        return {
            "day_of_week": day_of_week,
            "day_of_week_num": dt.weekday(),
            "is_us_holiday": is_holiday,
            "holiday_name": holiday_name,
            "is_weekend": is_weekend,
            "is_pre_holiday": is_pre_holiday,
            "is_post_holiday": is_post_holiday,
            "expected_liquidity": expected_liquidity,
            "liquidity_warning": expected_liquidity in ("VERY_LOW", "LOW", "REDUCED"),
        }

    def track_data_latency(self, exchange_timestamp_ms: int) -> Dict[str, Any]:
        """
        Calcula e classifica a latência entre o timestamp da exchange
        e o timestamp local (ajustado com offset Binance).
        """
        local_now_ms = self.now()
        latency_ms = local_now_ms - exchange_timestamp_ms
        
        # Proteger contra timestamps futuros (clock skew)
        if latency_ms < 0:
            latency_ms = abs(latency_ms)
        
        if latency_ms < 50:
            category = "EXCELLENT"
            freshness = "REAL_TIME"
        elif latency_ms < 200:
            category = "GOOD"
            freshness = "REAL_TIME"
        elif latency_ms < 500:
            category = "ACCEPTABLE"
            freshness = "NEAR_REAL_TIME"
        elif latency_ms < 2000:
            category = "DEGRADED"
            freshness = "NEAR_REAL_TIME"
        elif latency_ms < 10000:
            category = "POOR"
            freshness = "DELAYED"
        else:
            category = "CRITICAL"
            freshness = "STALE"
        
        return {
            "latency_ms": round(latency_ms),
            "latency_category": category,
            "data_freshness": freshness,
            "is_acceptable": latency_ms < 2000,
            "is_stale": latency_ms > 10000,
        }

    def __repr__(self) -> str:
        """Representação string do TimeManager."""
        with self._lock:
            return (
                f"TimeManager(status={self.time_sync_status}, "
                f"offset={self.server_time_offset_ms}ms, "
                f"rtt={self.last_rtt_ms}ms)"
            )


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    tm = TimeManager(
        sync_interval_minutes=30,
        max_init_attempts=3,
        max_acceptable_offset_ms=600,
        num_sync_samples=5
    )
    
    print("\n📅 Exemplos de uso:")
    print(f"Epoch ms:     {tm.now()}")
    print(f"UTC ISO:      {tm.now_utc_iso()}")
    print(f"NY ISO:       {tm.now_ny_iso()}")
    print(f"SP ISO:       {tm.now_sp_iso()}")
    
    print("\n🆕 Testando from_timestamp_ms():")
    ts_ms = tm.now()
    dt_utc = tm.from_timestamp_ms(ts_ms)
    dt_ny = tm.from_timestamp_ms(ts_ms, tz=tm.tz_ny)
    print(f"   Timestamp: {ts_ms}")
    print(f"   UTC:       {dt_utc.isoformat()}")
    print(f"   NY:        {dt_ny.isoformat()}")
    
    print("\n📊 Time index:")
    idx = tm.build_time_index()
    for k, v in idx.items():
        print(f"   {k}: {v}")
    
    print("\n🔄 Forçando nova sincronização...")
    tm.force_sync()
    
    print("\n✅ TimeManager v2.1.2 testado com sucesso!")