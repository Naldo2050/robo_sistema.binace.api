import time
import requests
import logging
import random
from datetime import timezone, datetime

class TimeManager:
    def __init__(self, sync_interval_minutes=30):
        self.server_time_offset_ms = 0
        self.last_sync_time = 0
        self.sync_interval_seconds = sync_interval_minutes * 60
        self._sync_with_binance()

    def _sync_with_binance(self):
        """Sincroniza com o servidor da Binance e calcula o offset em milissegundos."""
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                # 🔹 CORRIGIDO: REMOVIDOS ESPAÇOS FINAIS
                response = requests.get("https://fapi.binance.com/fapi/v1/time", timeout=5)
                response.raise_for_status()
                server_time_ms = response.json().get("serverTime")
                
                if server_time_ms is None:
                    raise ValueError("serverTime ausente na resposta da Binance")
                
                local_time_ms = int(time.time() * 1000)
                self.server_time_offset_ms = server_time_ms - local_time_ms
                self.last_sync_time = time.time()
                
                # 🔹 NOVO: Alerta se o offset for muito grande
                if abs(self.server_time_offset_ms) > 5000:  # 5 segundos
                    logging.critical(f"💀 DRIFT DE TEMPO DETECTADO: {self.server_time_offset_ms}ms!")
                    logging.critical("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    logging.critical("!!! ATENÇÃO: O RELÓGIO DO SEU COMPUTADOR ESTÁ FORA DE SINCRONIA. !!!")
                    logging.critical("!!! ATIVE A SINCRONIZAÇÃO AUTOMÁTICA DE HORA NAS CONFIGURAÇÕES DO SEU SISTEMA. !!!")
                    logging.critical("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                
                logging.info(f"✅ TimeManager sincronizado. Offset: {self.server_time_offset_ms}ms")
                return
                
            except requests.exceptions.RequestException as e:
                logging.warning(f"Erro de requisição ao sincronizar com Binance (tentativa {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logging.info(f"Aguardando {delay:.1f}s antes de retry...")
                    time.sleep(delay)
            except Exception as e:
                logging.warning(f"Erro ao sincronizar com Binance (tentativa {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logging.info(f"Aguardando {delay:.1f}s antes de retry...")
                    time.sleep(delay)
        
        # 🔹 Fallback: usa tempo local
        logging.warning("⚠️ Falha persistente ao sincronizar com Binance. Usando tempo local.")
        self.server_time_offset_ms = 0
        self.last_sync_time = time.time()

    def _should_sync(self):
        """Verifica se é hora de sincronizar novamente."""
        return time.time() - self.last_sync_time > self.sync_interval_seconds

    def now(self):
        """Retorna o timestamp atual em milissegundos, sincronizado com Binance."""
        try:
            if self._should_sync():
                self._sync_with_binance()
            return int(time.time() * 1000) + self.server_time_offset_ms
        except Exception as e:
            logging.error(f"Erro ao obter timestamp: {e}")
            # 🔹 Fallback: retorna tempo local em caso de erro
            return int(time.time() * 1000)

    def now_iso(self, tz=timezone.utc):
        """Retorna o timestamp formatado em ISO."""
        try:
            ms = self.now()
            return self.format_timestamp(ms, tz)
        except Exception as e:
            logging.error(f"Erro ao formatar timestamp ISO: {e}")
            # 🔹 Fallback: retorna timestamp local
            return datetime.now(tz).isoformat(timespec="seconds")

    @staticmethod
    def format_timestamp(ts_ms: int, tz=timezone.utc):
        """Formata timestamp (ms) em string ISO."""
        try:
            if not isinstance(ts_ms, (int, float)) or ts_ms < 0:
                raise ValueError(f"Timestamp inválido: {ts_ms}")
                
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=tz)
            return dt.isoformat(timespec="seconds")
        except Exception as e:
            logging.error(f"Erro ao formatar timestamp {ts_ms}: {e}")
            # 🔹 Fallback: retorna timestamp atual
            return datetime.now(tz).isoformat(timespec="seconds")