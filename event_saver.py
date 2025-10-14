# -*- coding: utf-8 -*-
# event_saver.py - v4.2 PROFISSIONAL - Sistema institucional com janelas numeradas e timestamps NY/SP corretos
import json
from pathlib import Path
import platform
import logging
import threading
import time
import atexit
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Union
import re

# Tratamento robusto de timezones
TIMEZONE_AVAILABLE = False
UTC_TZ = None
NY_TZ = None
SP_TZ = None

# Tenta importar zoneinfo (Python 3.9+)
try:
    from zoneinfo import ZoneInfo
    UTC_TZ = ZoneInfo("UTC")
    NY_TZ = ZoneInfo("America/New_York")
    SP_TZ = ZoneInfo("America/Sao_Paulo")
    TIMEZONE_AVAILABLE = True
except ImportError:
    # Tenta pytz como fallback
    try:
        import pytz
        UTC_TZ = pytz.UTC
        NY_TZ = pytz.timezone("America/New_York")
        SP_TZ = pytz.timezone("America/Sao_Paulo")
        TIMEZONE_AVAILABLE = True
    except ImportError:
        # Fallback final: usa timezone offset fixo
        logging.warning("Nenhuma biblioteca de timezone disponível (zoneinfo ou pytz). Usando offsets fixos.")
        UTC_TZ = timezone.utc
        # NY: UTC-5 (EST) ou UTC-4 (EDT) - vamos usar -5 como padrão
        NY_TZ = timezone(timedelta(hours=-5))
        # SP: UTC-3 (BRT)
        SP_TZ = timezone(timedelta(hours=-3))

# Tratamento de numpy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logging.info("NumPy não disponível - usando verificação básica para NaN/Inf")

# Import seguro do time_manager
try:
    from time_manager import TimeManager
except ImportError:
    logging.info("TimeManager não disponível - usando implementação básica")
    class TimeManager:
        def now_iso(self, tz=None):
            if tz:
                return datetime.now(tz).isoformat()
            return datetime.now().isoformat()

# Import seguro de format_utils com fallback
try:
    from format_utils import (
        format_price,
        format_quantity,
        format_percent,
        format_large_number,
        format_delta,
        format_time_seconds,
        format_scientific
    )
    HAS_FORMAT_UTILS = True
except ImportError:
    HAS_FORMAT_UTILS = False
    logging.info("format_utils não disponível - usando formatação básica")
    # Implementação básica de fallback
    def format_price(value):
        """Formata preço com 2-4 casas decimais."""
        if value is None:
            return "null"
        try:
            if abs(value) >= 1000:
                return f"${value:,.2f}"
            else:
                return f"${value:.4f}"
        except:
            return str(value)

    def format_quantity(value):
        """Formata quantidade com notação apropriada."""
        if value is None:
            return "null"
        try:
            if abs(value) >= 1_000_000:
                return f"{value/1_000_000:.2f}M"
            elif abs(value) >= 1_000:
                return f"{value/1_000:.2f}K"
            else:
                return f"{value:.3f}"
        except:
            return str(value)

    def format_percent(value):
        """Formata percentual."""
        if value is None:
            return "null"
        try:
            return f"{value:.2f}%"
        except:
            return str(value)

    def format_large_number(value):
        """Formata números grandes com separadores."""
        if value is None:
            return "null"
        try:
            if abs(value) >= 1_000_000_000:
                return f"{value/1_000_000_000:.2f}B"
            elif abs(value) >= 1_000_000:
                return f"{value/1_000_000:.2f}M"
            elif abs(value) >= 1_000:
                return f"{value/1_000:.2f}K"
            else:
                return f"{value:,.2f}"
        except:
            return str(value)

    def format_delta(value):
        """Formata delta com sinal."""
        if value is None:
            return "null"
        try:
            sign = "+" if value > 0 else ""
            return f"{sign}{value:.3f}"
        except:
            return str(value)

    def format_time_seconds(value):
        """Formata tempo em segundos."""
        if value is None:
            return "null"
        try:
            return f"{value:.1f}s"
        except:
            return str(value)

    def format_scientific(value):
        """Formata notação científica."""
        if value is None:
            return "null"
        try:
            if abs(value) < 0.0001:
                return f"{value:.2e}"
            else:
                return f"{value:.6f}"
        except:
            return str(value)

# Diretório de dados
DATA_DIR = Path("dados")
DATA_DIR.mkdir(exist_ok=True)

class EventSaver:
    """
    Classe responsável por salvar e formatar eventos de trading.
    Implementa buffer assíncrono, formatação inteligente e múltiplos formatos de saída.
    """
    def __init__(self, sound_alert: bool = True):
        self.sound_alert = sound_alert
        self.snapshot_file = DATA_DIR / "eventos-fluxo.json"
        self.history_file = DATA_DIR / "eventos_fluxo.jsonl"
        self.visual_log_file = DATA_DIR / "eventos_visuais.log"
        self.last_window_id = None
        self.time_manager = TimeManager()
        self._window_counter = 0  # Contador global de janelas
        
        # Controle de cabeçalho por minuto
        self._last_logged_block = None  # chave: "YYYY-mm-dd HH:MM|context"
        self._seen_in_block = set()
        
        # Buffer de escrita + thread de flush
        self._write_buffer: List[Dict] = []
        self._buffer_lock = threading.Lock()
        self._flush_interval = 5  # segundos
        self._stop_event = threading.Event()
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Encerra limpo no exit
        atexit.register(self.stop)
        
        # Verificação inicial de horário
        self._debug_timezone_check()

    def _debug_timezone_check(self):
        """Verifica se os timezones estão funcionando corretamente."""
        try:
            now_utc = datetime.now(UTC_TZ)
            if TIMEZONE_AVAILABLE:
                now_ny = self._convert_timezone(now_utc, NY_TZ)
                now_sp = self._convert_timezone(now_utc, SP_TZ)
                self.logger.info("🕐 Verificação de Timezone:")
                self.logger.info(f"   UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                self.logger.info(f"   NY:  {now_ny.strftime('%Y-%m-%d %H:%M:%S')} (EST/EDT)")
                self.logger.info(f"   SP:  {now_sp.strftime('%Y-%m-%d %H:%M:%S')} (BRT)")
                
                # Calcula offsets
                if hasattr(now_ny, 'utcoffset') and now_ny.utcoffset():
                    utc_offset_ny = now_ny.utcoffset().total_seconds() / 3600
                    self.logger.info(f"   Offset NY vs UTC: {utc_offset_ny:.1f} horas")
                if hasattr(now_sp, 'utcoffset') and now_sp.utcoffset():
                    utc_offset_sp = now_sp.utcoffset().total_seconds() / 3600
                    self.logger.info(f"   Offset SP vs UTC: {utc_offset_sp:.1f} horas")
            else:
                self.logger.warning("⚠️ Usando offsets fixos de timezone (NY: -5h, SP: -3h)")
        except Exception as e:
            self.logger.error(f"Erro ao verificar timezones: {e}")

    def _convert_timezone(self, dt: datetime, target_tz) -> datetime:
        """Converte datetime para timezone alvo de forma segura."""
        try:
            # Se já tem timezone
            if dt.tzinfo is not None:
                # Método para pytz
                if hasattr(target_tz, 'normalize'):
                    return target_tz.normalize(dt.astimezone(target_tz))
                # Método padrão
                else:
                    return dt.astimezone(target_tz)
            else:
                # Assume UTC se não tem timezone
                if UTC_TZ == timezone.utc:
                    # Usando timezone padrão
                    dt_utc = dt.replace(tzinfo=UTC_TZ)
                    return dt_utc.astimezone(target_tz)
                else:
                    # Usando pytz ou zoneinfo
                    if hasattr(UTC_TZ, 'localize'):
                        dt_utc = UTC_TZ.localize(dt)
                    else:
                        dt_utc = dt.replace(tzinfo=UTC_TZ)
                    if hasattr(target_tz, 'normalize'):
                        return target_tz.normalize(dt_utc.astimezone(target_tz))
                    else:
                        return dt_utc.astimezone(target_tz)
        except Exception as e:
            self.logger.debug(f"Erro na conversão de timezone: {e}")
            return dt

    @staticmethod
    def _parse_iso8601(ts: str) -> datetime:
        """Parseia timestamp ISO-8601 com suporte a 'Z' (UTC) e offset."""
        if not ts:
            raise ValueError("Timestamp vazio")
        try:
            # Normaliza 'Z' -> '+00:00' para compatibilidade
            if ts.endswith("Z"):
                ts = ts[:-1] + "+00:00"
            # Tenta parse direto
            dt = datetime.fromisoformat(ts)
            # Garante que tem timezone
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC_TZ)
            return dt
        except Exception as e:
            # Fallback: tenta formatos alternativos
            try:
                # Remove microsegundos se houver
                base = ts.split(".")[0]
                # Tenta identificar offset
                if "+" in base:
                    base, offset = base.split("+")
                    ts_clean = f"{base}+{offset}"
                elif "-" in base and base.count("-") > 2:
                    # Pode ter offset negativo
                    parts = base.split("-")
                    base = "-".join(parts[:3])
                    offset = "-".join(parts[3:])
                    ts_clean = f"{base}-{offset}"
                else:
                    # Assume UTC
                    ts_clean = base + "+00:00"
                return datetime.fromisoformat(ts_clean)
            except Exception:
                # Último fallback: tenta parsing manual
                try:
                    # Formato comum: 2024-01-01T12:00:00.000Z
                    dt_str = ts.replace("Z", "").replace("T", " ").split(".")[0]
                    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                    return dt.replace(tzinfo=UTC_TZ)
                except:
                    raise ValueError(f"Não foi possível parsear timestamp: {ts}") from e

    def stop(self):
        """Para a thread de flush e realiza um flush final."""
        if not self._stop_event.is_set():
            self._stop_event.set()
            try:
                if self._flush_thread.is_alive():
                    self._flush_thread.join(timeout=2.0)
            except Exception as e:
                self.logger.error(f"Erro ao parar thread de flush: {e}")
            # Flush final
            with self._buffer_lock:
                buffer_copy = self._write_buffer.copy()
                self._write_buffer.clear()
            if buffer_copy:
                self._flush_buffer(buffer_copy)
                self.logger.info(f"Flush final realizado: {len(buffer_copy)} eventos")

    def _is_nan_or_inf(self, value) -> bool:
        """Verifica se um valor é NaN ou Inf de forma segura."""
        if value is None:
            return False
        if HAS_NUMPY:
            try:
                return np.isnan(value) or np.isinf(value)
            except (TypeError, ValueError):
                return False
        else:
            # Fallback sem numpy
            try:
                if isinstance(value, float):
                    return value != value or value in (float('inf'), float('-inf'))
                return False
            except:
                return False

    def _clean_numeric_value(self, value: Any, field_type: str = "generic") -> Optional[Union[int, float]]:
        """
        Limpa e arredonda um valor numérico.
        Retorna None para valores inválidos.
        """
        if value is None or value == '':
            return None
        if isinstance(value, str):
            if value.lower() in ['n/a', 'none', 'null', 'nan', 'inf', '-inf']:
                return None
            # Remove vírgulas de milhares
            value = value.replace(',', '')
        try:
            # Converte para float
            num_value = float(value)
            # Verifica NaN/Inf
            if self._is_nan_or_inf(num_value):
                return None
            # Define precisão baseada no tipo
            precision_map = {
                "price": 4,
                "quantity": 3,
                "percent": 2,
                "delta": 3,
                "scientific": 8,
                "generic": 4
            }
            precision = precision_map.get(field_type, 4)
            rounded_value = round(num_value, precision)
            # Retorna como int se não houver parte decimal
            if rounded_value == int(rounded_value) and field_type != "scientific":
                return int(rounded_value)
            return rounded_value
        except (ValueError, TypeError, OverflowError):
            return None

    def _clean_event_data(self, event: Dict) -> Dict:
        """
        Limpa recursivamente todos os campos numéricos do evento.
        """
        if not isinstance(event, dict):
            return event
        cleaned = {}
        
        # Categorização de campos por tipo
        field_types = {
            'price': {'preco', 'price', 'close', 'open', 'high', 'low', 'poc', 'val', 'vah', 
                     'level', 'bid_price', 'ask_price', 'wall_price', 'anchor'},
            'quantity': {'volume', 'quantity', 'size', 'count', 'trades', 'interest'},
            'delta': {'delta', 'flow', 'imbalance', 'net'},
            'percent': {'pct', 'percent', 'ratio', 'prob', 'rate'},
            'scientific': {'volatility', 'returns', 'slope', 'momentum', 'correlation', 'gradient'},
            'timestamp': {'epoch_ms', 'timestamp_ms', 'window_ms', 'time_ms', '_ms', '_ts'}
        }

        def get_field_type(key: str) -> str:
            """Determina o tipo do campo baseado no nome."""
            key_lower = key.lower()
            # Timestamps têm prioridade
            if any(ts in key_lower for ts in field_types['timestamp']):
                return 'timestamp'
            for field_type, keywords in field_types.items():
                if field_type != 'timestamp':
                    if any(kw in key_lower for kw in keywords):
                        return field_type
            return 'generic'

        # Processa cada campo
        for key, value in event.items():
            field_type = get_field_type(key)
            
            # Timestamps não são processados como números
            if field_type == 'timestamp':
                cleaned[key] = value
            # Processa valores numéricos
            elif isinstance(value, (int, float)):
                clean_value = self._clean_numeric_value(value, field_type)
                if clean_value is not None:
                    cleaned[key] = clean_value
                # Omite valores None/NaN/Inf
            # Strings que podem ser números
            elif isinstance(value, str) and value.replace('.', '').replace('-', '').replace('+', '').replace(',', '').isdigit():
                clean_value = self._clean_numeric_value(value, field_type)
                if clean_value is not None:
                    cleaned[key] = clean_value
            # Dicionários aninhados - otimiza arrays para economizar espaço
            elif isinstance(value, dict):
                cleaned_dict = self._clean_event_data(value)
                if cleaned_dict:  # Só adiciona se não estiver vazio
                    cleaned[key] = cleaned_dict
            # Listas - otimiza para economizar espaço
            elif isinstance(value, list):
                cleaned_list = []
                for item in value:
                    if isinstance(item, dict):
                        clean_item = self._clean_event_data(item)
                        if clean_item:
                            cleaned_list.append(clean_item)
                    elif isinstance(item, (int, float)):
                        clean_value = self._clean_numeric_value(item, field_type)
                        if clean_value is not None:
                            cleaned_list.append(clean_value)  # CORRIGIDO: era clean_item
                    elif item is not None:
                        cleaned_list.append(item)
                
                # Otimiza listas grandes (como hvns) para economizar espaço
                if key.lower() in ['hvns', 'lvns', 'historical_volumes'] and len(cleaned_list) > 10:
                    # Converte para string compacta se for uma lista de números
                    if all(isinstance(x, (int, float)) for x in cleaned_list):
                        cleaned[key] = cleaned_list
                    else:
                        cleaned[key] = cleaned_list
                elif cleaned_list:  # Só adiciona se não estiver vazia
                    cleaned[key] = cleaned_list
            # Outros valores (strings, booleans, etc)
            elif value is not None:
                cleaned[key] = value

        return cleaned

    def _flush_loop(self):
        """Thread que periodicamente esvazia o buffer."""
        while not self._stop_event.wait(self._flush_interval):
            with self._buffer_lock:
                if not self._write_buffer:
                    continue
                buffer_copy = self._write_buffer.copy()
                self._write_buffer.clear()
            if buffer_copy:
                self._flush_buffer(buffer_copy)

    def _flush_buffer(self, events: List[Dict]):
        """Escreve eventos em lote nos arquivos."""
        for event in events:
            try:
                # Limpa dados
                cleaned_event = self._clean_event_data(event)
                if cleaned_event:
                    # Salva em diferentes formatos
                    self._save_to_json(cleaned_event)
                    self._save_to_jsonl(cleaned_event)
                    self._add_visual_log_entry(cleaned_event)
            except Exception as e:
                self.logger.error(f"Erro ao processar evento no flush: {e}")

    def _save_to_json(self, event: Dict):
        """Salva evento em arquivo JSON com retry."""
        max_retries = 3
        retry_delay = 0.5
        for attempt in range(max_retries):
            try:
                # Lê eventos existentes
                events = []
                if self.snapshot_file.exists():
                    try:
                        with open(self.snapshot_file, "r", encoding="utf-8") as f:
                            content = f.read().strip()
                            if content:
                                events = json.loads(content)
                    except (json.JSONDecodeError, IOError) as e:
                        self.logger.warning(f"Arquivo JSON corrompido, recriando: {e}")
                        events = []

                # Adiciona novo evento
                events.append(event)
                
                # Limita tamanho do snapshot
                max_events = 1000
                if len(events) > max_events:
                    events = events[-max_events:]

                # Salva atomicamente
                temp_file = self.snapshot_file.with_suffix('.tmp')
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(events, f, indent=2, ensure_ascii=False, default=str)
                # Move atomicamente
                temp_file.replace(self.snapshot_file)
                return  # Sucesso
            except Exception as e:
                self.logger.error(f"Erro ao salvar JSON (tentativa {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    # Fallback final
                    self._save_fallback(event, "json")

    def _save_to_jsonl(self, event: Dict):
        """Salva evento em arquivo JSONL com retry."""
        max_retries = 3
        retry_delay = 0.5
        for attempt in range(max_retries):
            try:
                with open(self.history_file, "a", encoding="utf-8") as f:
                    json_line = json.dumps(event, ensure_ascii=False, default=str)
                    f.write(json_line + "\n")
                    f.flush()
                return  # Sucesso
            except Exception as e:
                self.logger.error(f"Erro ao salvar JSONL (tentativa {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    # Fallback final
                    self._save_fallback(event, "jsonl")

    def _save_fallback(self, event: Dict, format_type: str):
        """Salva em diretório de fallback quando o principal falha."""
        try:
            fallback_dir = Path("./fallback_events")
            fallback_dir.mkdir(exist_ok=True)
            if format_type == "json":
                fallback_file = fallback_dir / f"eventos_{datetime.now().strftime('%Y%m%d')}.json"
                events = []
                if fallback_file.exists():
                    try:
                        with open(fallback_file, "r", encoding="utf-8") as f:
                            events = json.load(f)
                    except:
                        events = []
                events.append(event)
                with open(fallback_file, "w", encoding="utf-8") as f:
                    json.dump(events, f, indent=2, ensure_ascii=False, default=str)
            else:  # jsonl
                fallback_file = fallback_dir / f"eventos_{datetime.now().strftime('%Y%m%d')}.jsonl"
                with open(fallback_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            self.logger.warning(f"⚠️ Evento salvo em fallback: {fallback_file}")
        except Exception as e:
            self.logger.critical(f"💀 FALHA TOTAL ao salvar em fallback: {e}")

    def save_event(self, event: Dict):
        """
        API pública para salvar um evento.
        Adiciona timestamps, contexto e bufferiza para escrita assíncrona.
        """
        if not isinstance(event, dict):
            self.logger.error("Evento inválido: não é um dicionário")
            return

        try:
            # Determina contexto temporal
            epoch_ms = event.get("epoch_ms")
            timestamp = event.get("timestamp")
            dt = None
            if epoch_ms:
                try:
                    dt = datetime.fromtimestamp(int(epoch_ms) / 1000, tz=UTC_TZ)
                except (ValueError, TypeError, OSError) as e:
                    self.logger.error(f"epoch_ms inválido {epoch_ms}: {e}")
            elif timestamp:
                try:
                    dt = self._parse_iso8601(timestamp)
                except Exception as e:
                    self.logger.error(f"timestamp inválido {timestamp}: {e}")

            if dt:
                now = datetime.now(UTC_TZ)
                # Determina se é histórico ou real-time
                time_diff = abs((dt - now).total_seconds())
                if time_diff > 86400:  # > 24 horas
                    event["data_context"] = "historical"
                else:
                    event["data_context"] = "real_time"
                
                # Adiciona timestamps em diferentes timezones - FORMATADO PARA BRASIL
                event["timestamp_utc"] = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
                
                if TIMEZONE_AVAILABLE:
                    dt_ny = self._convert_timezone(dt, NY_TZ)
                    dt_sp = self._convert_timezone(dt, SP_TZ)
                    event["timestamp_ny"] = dt_ny.strftime("%Y-%m-%d %H:%M:%S EST/EDT")
                    event["timestamp_sp"] = dt_sp.strftime("%Y-%m-%d %H:%M:%S BRT")
                else:
                    # Offsets fixos
                    dt_ny = dt.replace(tzinfo=UTC_TZ).astimezone(NY_TZ)
                    dt_sp = dt.replace(tzinfo=UTC_TZ).astimezone(SP_TZ)
                    event["timestamp_ny"] = dt_ny.strftime("%Y-%m-%d %H:%M:%S") + " (EST)"
                    event["timestamp_sp"] = dt_sp.strftime("%Y-%m-%d %H:%M:%S") + " (BRT)"
                
                # Garante timestamp ISO se não existir
                if "timestamp" not in event:
                    event["timestamp"] = dt.isoformat(timespec="milliseconds")
            else:
                event["data_context"] = "unknown"
                self.logger.warning("Evento sem timestamp válido")

            # Enriquece eventos simples com contexto mínimo institucional
            if event.get("tipo_evento") == "Alerta" and "context" in event:
                context = event["context"]
                enriched = {
                    "price_data": {
                        "current": {
                            "last": context.get("price"),
                            "volume": context.get("volume")
                        }
                    },
                    "volatility_metrics": {
                        "realized_vol_24h": context.get("volatility")
                    },
                    "market_context": {
                        "trading_session": "NY_OVERLAP",
                        "session_phase": "ACTIVE"
                    }
                }
                event.update(enriched)

            # Detecta nova janela - melhoria para detectar janelas de tempo real
            window_id = event.get("window_id") or event.get("candle_id_ms") or event.get("epoch_ms")
            if dt and window_id:
                # Cria ID de janela baseado no minuto
                window_time = dt.replace(second=0, microsecond=0)
                window_key = window_time.strftime("%Y%m%d_%H%M")
                
                if window_key != self.last_window_id:
                    self._window_counter += 1
                    event["janela_numero"] = self._window_counter
                    self._add_visual_separator(event)
                    self.last_window_id = window_key

            # Bufferiza evento
            with self._buffer_lock:
                self._write_buffer.append(event)

            # Alerta sonoro para sinais
            if self.sound_alert and event.get("is_signal", False):
                self._play_sound()

        except Exception as e:
            self.logger.error(f"Erro crítico ao processar evento: {e}", exc_info=True)

    def _add_visual_separator(self, event: Dict):
        """Adiciona separador visual para nova janela de tempo."""
        try:
            epoch_ms = event.get("epoch_ms") or event.get("window_close_ms")
            if epoch_ms:
                try:
                    dt_utc = datetime.fromtimestamp(int(epoch_ms) / 1000, tz=UTC_TZ)
                    if TIMEZONE_AVAILABLE:
                        dt_ny = self._convert_timezone(dt_utc, NY_TZ)
                        dt_sp = self._convert_timezone(dt_utc, SP_TZ)
                        timestamp_utc = dt_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
                        timestamp_ny = dt_ny.strftime("%Y-%m-%d %H:%M:%S EST/EDT")
                        timestamp_sp = dt_sp.strftime("%Y-%m-%d %H:%M:%S BRT")
                    else:
                        dt_ny = dt_utc.replace(tzinfo=UTC_TZ).astimezone(NY_TZ)
                        dt_sp = dt_utc.replace(tzinfo=UTC_TZ).astimezone(SP_TZ)
                        timestamp_utc = dt_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
                        timestamp_ny = dt_ny.strftime("%Y-%m-%d %H:%M:%S") + " (EST)"
                        timestamp_sp = dt_sp.strftime("%Y-%m-%d %H:%M:%S") + " (BRT)"
                except:
                    timestamp_utc = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
                    timestamp_ny = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " (EST)"
                    timestamp_sp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " (BRT)"
            else:
                timestamp_utc = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
                timestamp_ny = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " (EST)"
                timestamp_sp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " (BRT)"

            # Obtém o número da janela
            window_num = event.get("janela_numero", "NOVA")
            
            # Formata janela de tempo
            open_ms = event.get("window_open_ms") or event.get("candle_open_time_ms")
            close_ms = event.get("window_close_ms") or event.get("candle_close_time_ms")
            
            def format_window_time(ms):
                if not ms or ms <= 0:
                    return "N/A"
                try:
                    dt = datetime.fromtimestamp(ms / 1000, tz=UTC_TZ)
                    if TIMEZONE_AVAILABLE:
                        dt_sp = self._convert_timezone(dt, SP_TZ)
                    else:
                        dt_sp = dt.replace(tzinfo=UTC_TZ).astimezone(SP_TZ)
                    return dt_sp.strftime("%H:%M:%S")
                except:
                    return "N/A"

            start_time = format_window_time(open_ms)
            end_time = format_window_time(close_ms)
            
            # CORREÇÃO: Formato exato conforme solicitado
            separator = f"\n{'='*100}\n"
            separator += f"🗓️  JANELA {window_num}\n"
            separator += f"⏰ {timestamp_utc}\n"
            separator += f"📍 NY: {timestamp_ny.split(' ')[1].split('(')[0]} {timestamp_ny.split(' ')[2] if '(' in timestamp_ny else 'EST/EDT'}\n"
            separator += f"📍 São Paulo: {timestamp_sp.split(' ')[1].split('(')[0]} {timestamp_sp.split(' ')[2] if '(' in timestamp_sp else 'BRT'}\n"
            separator += f"📊 Contexto: {event.get('data_context', 'real_time')}\n"
            separator += f"{'='*100}\n"
            
            with open(self.visual_log_file, "a", encoding="utf-8") as f:
                f.write(separator)
                f.flush()
        except Exception as e:
            self.logger.error(f"Erro ao adicionar separador: {e}")

    def _format_value_for_display(self, value: Any, key: str = "") -> str:
        """Formata valor para exibição legível."""
        if value is None:
            return "null"
        if not HAS_FORMAT_UTILS:
            # Formatação básica se format_utils não estiver disponível
            if isinstance(value, bool):
                return str(value).lower()
            elif isinstance(value, (int, float)):
                return f"{value:,.4f}" if isinstance(value, float) else str(value)
            else:
                return str(value)
        
        key_lower = key.lower()
        try:
            if isinstance(value, bool):
                return str(value).lower()
            elif isinstance(value, (int, float)):
                # Usa funções apropriadas baseadas no tipo
                if any(x in key_lower for x in ['price', 'preco', 'poc', 'val', 'vah']):
                    return format_price(value)
                elif any(x in key_lower for x in ['volume', 'quantity', 'size']):
                    return format_large_number(value)
                elif 'delta' in key_lower:
                    return format_delta(value)
                elif any(x in key_lower for x in ['pct', 'percent', 'ratio']):
                    return format_percent(value)
                elif any(x in key_lower for x in ['volatility', 'returns', 'slope']):
                    return format_scientific(value)
                else:
                    return str(value)
            else:
                return str(value)
        except:
            return str(value)

    def _add_visual_log_entry(self, event: Dict):
        """Adiciona entrada formatada ao log visual."""
        try:
            # Determina bloco de minuto
            epoch_ms = event.get("epoch_ms")
            timestamp = event.get("timestamp")
            dt = None
            if epoch_ms:
                try:
                    dt = datetime.fromtimestamp(int(epoch_ms) / 1000, tz=UTC_TZ)
                except:
                    pass
            elif timestamp:
                try:
                    dt = self._parse_iso8601(timestamp)
                except:
                    pass

            if dt:
                minute_key = dt.strftime("%Y-%m-%d %H:%M")
                context = event.get("data_context", "unknown")
                minute_block = f"{minute_key}|{context}"
                
                # Evita duplicatas no mesmo bloco
                event_key = (
                    str(event.get("timestamp")),
                    str(event.get("tipo_evento")),
                    str(event.get("volume_total"))
                )
                if event_key in self._seen_in_block:
                    return
                self._seen_in_block.add(event_key)

            # Prepara evento limpo para log
            clean_event = self._prepare_visual_event(event)
            
            # Escreve no log com formatação otimizada
            with open(self.visual_log_file, "a", encoding="utf-8") as f:
                # Converte para JSON e formata para economizar espaço
                json_str = json.dumps(clean_event, indent=2, ensure_ascii=False, default=str)
                
                # Otimiza arrays longos para economizar espaço
                json_str = self._optimize_json_display(json_str)
                
                f.write(json_str + "\n")
                f.flush()
        except Exception as e:
            self.logger.error(f"Erro ao adicionar entrada visual: {e}")

    def _optimize_json_display(self, json_str: str) -> str:
        """Otimiza a exibição de JSON para economizar espaço."""
        # Otimiza arrays longos de números
        def optimize_arrays(match):
            array_content = match.group(1)
            numbers = [x.strip() for x in array_content.split(',')]
            if len(numbers) > 10 and all(re.match(r'^[\d\.\-]+$', n) for n in numbers):
                # Mostra primeiro e últimos 3 elementos
                if len(numbers) > 6:
                    optimized = numbers[:3] + ['...'] + numbers[-3:]
                    return '[' + ', '.join(optimized) + ']'
            return match.group(0)
        
        # Substitui arrays longos
        json_str = re.sub(r'\[([^\[\]]*)\]', optimize_arrays, json_str)
        return json_str

    def _prepare_visual_event(self, event: Dict) -> Dict:
        """Prepara evento para visualização removendo redundâncias."""
        clean = dict(event)
        
        # Remove campos de timezone redundantes
        for field in ['time_ny', 'time_sp', 'time_utc']:
            clean.pop(field, None)
        
        # Remove campos internos
        for field in ['_id', '_rev', '_key']:
            clean.pop(field, None)
        
        # Remove duplicações conhecidas
        if 'contextual_snapshot' in clean and 'flow_metrics' in clean.get('contextual_snapshot', {}):
            if 'flow_metrics' in clean:
                del clean['contextual_snapshot']['flow_metrics']
        
        # Remove campos zerados ou nulos recursivamente
        def remove_empty(obj):
            if isinstance(obj, dict):
                return {k: remove_empty(v) for k, v in obj.items() 
                       if v is not None and v != 0 and v != "" and v != []}
            elif isinstance(obj, list):
                return [remove_empty(item) for item in obj if item is not None]
            return obj
        
        clean = remove_empty(clean)
        return clean

    def _play_sound(self):
        """Reproduz alerta sonoro multiplataforma."""
        try:
            system = platform.system()
            if system == "Windows":
                try:
                    import winsound
                    winsound.Beep(1000, 500)
                except ImportError:
                    print("\n🔔 ALERTA: Sinal detectado! 🔔\n")
            elif system == "Darwin":  # macOS
                try:
                    import subprocess
                    subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"], 
                                 capture_output=True, timeout=2)
                except:
                    print("\n🔔 ALERTA: Sinal detectado! 🔔\n")
            elif system == "Linux":
                try:
                    import subprocess
                    # Tenta diferentes comandos de áudio
                    for cmd in [
                        ["paplay", "/usr/share/sounds/freedesktop/stereo/bell.oga"],
                        ["aplay", "/usr/share/sounds/alsa/Front_Center.wav"],
                        ["speaker-test", "-t", "sine", "-f", "1000", "-l", "1"]
                    ]:
                        try:
                            subprocess.run(cmd, capture_output=True, timeout=1)
                            break
                        except:
                            continue
                except:
                    print("\n🔔 ALERTA: Sinal detectado! 🔔\n")
            else:
                print("\n🔔 ALERTA: Sinal detectado! 🔔\n")
        except Exception as e:
            self.logger.debug(f"Som não disponível: {e}")
            print("\n🔔 ALERTA: Sinal detectado! 🔔\n")

# Instância global (opcional)
_global_saver = None

def get_event_saver() -> EventSaver:
    """Retorna instância global do EventSaver."""
    global _global_saver
    if _global_saver is None:
        _global_saver = EventSaver()
    return _global_saver

# Para instalar as dependências opcionais (caso necessário):
def install_dependencies():
    """Helper para instalar dependências opcionais."""
    import subprocess
    import sys
    packages = []
    # Verifica pytz
    try:
        import pytz
    except ImportError:
        packages.append("pytz")
    # Verifica numpy
    try:
        import numpy
    except ImportError:
        packages.append("numpy")
    
    if packages:
        print(f"Instalando dependências opcionais: {', '.join(packages)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
        print("✅ Dependências instaladas com sucesso!")
    else:
        print("✅ Todas as dependências já estão instaladas!")

if __name__ == "__main__":
    # Teste básico
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Verifica/instala dependências
    print("Verificando dependências...")
    try:
        install_dependencies()
    except Exception as e:
        print(f"⚠️ Não foi possível instalar dependências automaticamente: {e}")
        print("Execute manualmente: pip install pytz numpy")
    
    print("\nIniciando teste do EventSaver...")
    saver = EventSaver(sound_alert=False)
    
    # Evento de teste com estrutura institucional completa
    test_event = {
        "metadata": {
            "timestamp_utc": "2025-09-15T01:05:00.000Z",
            "timestamp_unix": 1726359900000,
            "sequence_id": 1663456789012,
            "exchange_timestamp": "2025-09-15T01:05:00.123Z",
            "latency_ms": 12.5,
            "data_quality_score": 0.98,
            "completeness_pct": 99.2,
            "reliability_score": 9.1
        },
        "data_source": {
            "primary_exchange": "BINANCE",
            "backup_exchanges": ["COINBASE", "KRAKEN"],
            "data_feed_type": "WEBSOCKET_L2",
            "validation_passed": True,
            "cross_exchange_variance_pct": 0.05,
            "anomaly_detected": False
        },
        "market_context": {
            "trading_session": "NY_OVERLAP",
            "session_phase": "ACTIVE",
            "time_to_session_close": 7200,
            "day_of_week": 1,
            "is_holiday": False,
            "market_hours_type": "EXTENDED"
        },
        "price_data": {
            "current": {
                "last": 114994.05,
                "bid": 114994.04,
                "ask": 114994.06,
                "mid": 114994.05,
                "spread_bps": 0.17,
                "tick_direction": 1
            },
            "session": {
                "open": 114850.00,
                "high": 115100.00,
                "low": 114750.00,
                "close": 114994.05,
                "vwap": 114925.33,
                "twap": 114940.12
            }
        },
        "volume_profile": {
            "poc_price": 114875.50,
            "poc_volume": 1250.5,
            "vah": 115025.00,
            "val": 114725.00,
            "value_area_volume_pct": 68.5,
            "profile_shape": "NORMAL",
            "historical_vp": {
                "daily": {
                    "poc": 114243,
                    "vah": 115134,
                    "val": 113643,
                    "hvns": [113643, 113826, 113921, 113983, 114050, 114120, 114200, 114250, 114300, 114350, 114400, 114450, 114500, 114550, 114600, 114650, 114700, 114750, 114800, 114850]
                }
            }
        },
        "order_flow": {
            "net_flow_1m": -125000,
            "net_flow_5m": 340000,
            "net_flow_15m": -89000,
            "aggressive_buy_pct": 45.2,
            "aggressive_sell_pct": 54.8,
            "buy_sell_ratio": 0.84
        },
        "whale_activity": {
            "whale_net_position": "LONG",
            "whale_accumulation_score": 7.2,
            "large_orders_1h": [
                {"size": 15.5, "price": 114875.00, "side": "BUY", "timestamp": "01:03:45"},
                {"size": 22.1, "price": 114950.00, "side": "SELL", "timestamp": "01:04:12"}
            ]
        },
        "is_signal": True,
        "epoch_ms": int(time.time() * 1000),
        "tipo_evento": "institucional_snapshot"
    }
    
    print("Salvando evento de teste...")
    saver.save_event(test_event)
    
    # Aguarda flush
    print("Aguardando flush do buffer...")
    time.sleep(6)
    
    print(f"\n✅ Teste concluído!")
    print(f"📁 Verifique os arquivos em: {DATA_DIR.absolute()}")
    
    # Para thread
    saver.stop()