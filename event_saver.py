# event_saver.py - Ponto central de serialização com formatação limpa (VERSÃO FINAL)
import json
from pathlib import Path
import platform
import logging
import threading
import time
import atexit
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from time_manager import TimeManager
import numpy as np  # 🔹 ADICIONADO IMPORT DO NUMPY

# 🔹 IMPORTA UTILITÁRIOS DE FORMATAÇÃO
from format_utils import (
    format_price,
    format_quantity,
    format_percent,
    format_large_number,
    format_delta,
    format_time_seconds,
    format_scientific
)

NY_TZ = ZoneInfo("America/New_York")
SP_TZ = ZoneInfo("America/Sao_Paulo")
UTC_TZ = ZoneInfo("UTC")

DATA_DIR = Path("dados")
DATA_DIR.mkdir(exist_ok=True)


class EventSaver:
    def __init__(self, sound_alert=True):
        self.sound_alert = sound_alert
        self.snapshot_file = DATA_DIR / "eventos-fluxo.json"
        self.history_file = DATA_DIR / "eventos_fluxo.jsonl"
        self.visual_log_file = DATA_DIR / "eventos_visuais.log"
        self.last_window_id = None
        self.time_manager = TimeManager()

        # Controle de cabeçalho por minuto (não repetir)
        self._last_logged_block = None  # chave: "YYYY-mm-dd HH:MM|context"
        # Anti-duplicado simples no LOG por bloco (opcional)
        self._seen_in_block = set()

        # Buffer de escrita + thread de flush
        self._write_buffer = []
        self._buffer_lock = threading.Lock()
        self._flush_interval = 5  # segundos
        self._stop_event = threading.Event()
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

        # Encerra limpo no exit
        atexit.register(self.stop)
        
        # 🔹 DEBUG: Verificação inicial de horário
        self._debug_timezone_check()

    def _debug_timezone_check(self):
        """Verifica se os timezones estão funcionando corretamente."""
        try:
            now_utc = datetime.now(UTC_TZ)
            now_ny = now_utc.astimezone(NY_TZ)
            now_sp = now_utc.astimezone(SP_TZ)
            
            logging.info("🕐 Verificação de Timezone:")
            logging.info(f"   UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            logging.info(f"   NY:  {now_ny.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            logging.info(f"   SP:  {now_sp.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            
            # 🔹 CORREÇÃO: Usa utcoffset() ao invés de diferença de tempo
            utc_offset_ny = now_ny.utcoffset().total_seconds() / 3600 if now_ny.utcoffset() else 0
            utc_offset_sp = now_sp.utcoffset().total_seconds() / 3600 if now_sp.utcoffset() else 0
            
            logging.info(f"   Offset NY vs UTC: {utc_offset_ny:.1f} horas")
            logging.info(f"   Offset SP vs UTC: {utc_offset_sp:.1f} horas")
            
            # NY deve estar -4 ou -5 horas de UTC (EDT ou EST)
            # SP deve estar -3 horas de UTC
            if not (-5.5 < utc_offset_ny < -3.5):
                logging.warning(f"⚠️ Offset NY incomum: {utc_offset_ny:.1f} horas")
            else:
                logging.info(f"✅ Offset NY correto: {utc_offset_ny:.1f} horas")
                
            if not (-3.5 < utc_offset_sp < -2.5):
                logging.warning(f"⚠️ Offset SP incomum: {utc_offset_sp:.1f} horas")
            else:
                logging.info(f"✅ Offset SP correto: {utc_offset_sp:.1f} horas")
                
        except Exception as e:
            logging.error(f"Erro ao verificar timezones: {e}")

    # ---------- Utilidades internas ----------

    @staticmethod
    def _parse_iso8601(ts: str) -> datetime:
        """Aceita ISO-8601 com 'Z' (UTC) e com offset."""
        try:
            # Normaliza 'Z' -> +00:00 para compatibilidade ampla
            if ts.endswith("Z"):
                ts = ts[:-1] + "+00:00"
            dt = datetime.fromisoformat(ts)
            
            # 🔹 DEBUG: Verificação de timestamp
            now = datetime.now(UTC_TZ)
            diff = abs((dt - now).total_seconds())
            
            if diff > 86400:  # > 24 horas
                logging.warning(f"⚠️ Timestamp com diferença de {diff/3600:.1f} horas do horário atual")
                logging.warning(f"   Timestamp: {dt}")
                logging.warning(f"   Agora:     {now}")
            
            return dt
        except Exception as e:
            logging.error(f"Erro ao parsear ISO8601 '{ts}': {e}")
            # Último recurso: tenta sem microsegundos
            try:
                base, _, offset = ts.partition("+")
                if base and offset:
                    base = base.split(".")[0]
                    return datetime.fromisoformat(base + "+" + offset)
            except Exception:
                raise

    def stop(self):
        """Para a thread de flush e realiza um flush final."""
        if not self._stop_event.is_set():
            self._stop_event.set()
            try:
                if self._flush_thread.is_alive():
                    self._flush_thread.join(timeout=1.5)
            except Exception:
                pass
            # Flush final do que sobrou
            with self._buffer_lock:
                buffer_copy = self._write_buffer.copy()
                self._write_buffer.clear()
            if buffer_copy:
                self._flush_buffer(buffer_copy)

    # ---------- LIMPEZA E FORMATAÇÃO DE DADOS ----------

    def _clean_numeric_value(self, value, field_type="generic"):
        """
        Limpa e arredonda um valor numérico, retornando-o como float ou int.
        A formatação para exibição é feita em outra função.
        """
        if value is None or value == '' or (isinstance(value, str) and value.lower() in ['n/a', 'none', 'null']):
            return None
        
        try:
            # Converte para float, removendo vírgulas de strings
            if isinstance(value, str):
                value = float(value.replace(',', ''))
            else:
                value = float(value)

            if np.isnan(value) or np.isinf(value):
                return None

            # Aplica arredondamento baseado no tipo
            precision = 4 # Default
            if field_type == "price":
                precision = 4
            elif field_type == "quantity":
                precision = 3
            elif field_type == "percent":
                precision = 4
            elif field_type == "delta":
                precision = 3
            elif field_type == "scientific":
                # Para garantir que não seja notação científica no JSON
                return float(f"{value:.8f}")

            rounded_value = round(value, precision)
            
            # Retorna como int se não houver parte fracionária
            if rounded_value == int(rounded_value):
                return int(rounded_value)

            return rounded_value
                
        except (ValueError, TypeError):
            return value  # Retorna original se não for numérico

    def _clean_event_data(self, event: dict) -> dict:
        """
        Limpa todos os campos numéricos do evento antes de salvar.
        Aplica formatação adequada para cada tipo de campo.
        """
        if not isinstance(event, dict):
            return event
        
        cleaned = {}
        
        # Definição dos tipos de campos
        price_fields = {
            'preco_fechamento', 'preco_abertura', 'preco_maximo', 'preco_minimo',
            'price', 'close', 'open', 'high', 'low', 'poc', 'val', 'vah',
            'level', 'nearest_hvn', 'nearest_lvn', 'anchor_price',
            'bid_price', 'ask_price', 'bid_wall_price', 'ask_wall_price'
        }
        
        quantity_fields = {
            'volume_total', 'volume', 'quantity', 'total_volume',
            'buy_volume', 'sell_volume', 'trades_count', 'open_interest',
            'bid_size', 'ask_size', 'bid_wall_size', 'ask_wall_size'
        }
        
        delta_fields = {
            'delta', 'delta_fechamento', 'flow_imbalance', 'net_flow',
            'net_flow_1m', 'net_flow_5m', 'net_flow_15m'
        }
        
        percent_fields = {
            'imbalance_ratio', 'buy_sell_pressure', 'volume_sma_ratio',
            'long_prob', 'short_prob', 'neutral_prob', 'spread_percent',
            'aggressive_buy_pct', 'aggressive_sell_pct', 'passive_buy_pct', 'passive_sell_pct',
            'funding_rate_percent', 'volume_pct'
        }
        
        scientific_fields = {
            'volatility_1', 'volatility_5', 'volatility_15',
            'returns_1', 'returns_5', 'returns_15',
            'order_book_slope', 'microstructure', 'momentum_score',
            'tick_rule_sum', 'liquidity_gradient', 'long_short_ratio'
        }
        
        # Processa cada campo
        for key, value in event.items():
            # Determina o tipo do campo
            field_type = "generic"
            key_lower = key.lower()
            
            if key in price_fields or 'price' in key_lower or 'poc' in key_lower or 'val' in key_lower or 'vah' in key_lower:
                field_type = "price"
            elif key in quantity_fields or 'volume' in key_lower or 'size' in key_lower or 'count' in key_lower:
                field_type = "quantity"
            elif key in delta_fields or 'delta' in key_lower or 'flow' in key_lower:
                field_type = "delta"
            elif key in percent_fields or 'pct' in key_lower or 'percent' in key_lower or 'ratio' in key_lower or 'prob' in key_lower:
                field_type = "percent"
            elif key in scientific_fields or 'volatility' in key_lower or 'returns' in key_lower or 'slope' in key_lower:
                field_type = "scientific"
            
            # Processa o valor
            if isinstance(value, (int, float)):
                cleaned[key] = self._clean_numeric_value(value, field_type)
            elif isinstance(value, str) and value.replace('.', '').replace('-', '').replace('+', '').isdigit():
                cleaned[key] = self._clean_numeric_value(value, field_type)
            elif isinstance(value, dict):
                # Recursivo para dicionários aninhados
                cleaned[key] = self._clean_event_data(value)
            elif isinstance(value, list):
                # Processa listas
                cleaned_list = []
                for item in value:
                    if isinstance(item, dict):
                        cleaned_list.append(self._clean_event_data(item))
                    elif isinstance(item, (int, float)):
                        # Lista de números (ex: HVNs, LVNs)
                        if key in ['hvns', 'lvns', 'single_prints'] or 'price' in key_lower:
                            cleaned_list.append(self._clean_numeric_value(item, "price"))
                        else:
                            cleaned_list.append(self._clean_numeric_value(item, field_type))
                    else:
                        cleaned_list.append(item)
                cleaned[key] = cleaned_list
            else:
                # Mantém valores não numéricos como estão
                cleaned[key] = value
        
        return cleaned

    # ---------- Loop/flush ----------

    def _flush_loop(self):
        """Thread que esvazia o buffer periodicamente."""
        while not self._stop_event.is_set():
            time.sleep(self._flush_interval)
            with self._buffer_lock:
                if not self._write_buffer:
                    continue
                buffer_copy = self._write_buffer.copy()
                self._write_buffer.clear()
            self._flush_buffer(buffer_copy)

    def _flush_buffer(self, events: list):
        """Escreve eventos em lote."""
        for event in events:
            # Limpa dados antes de salvar
            cleaned_event = self._clean_event_data(event)
            
            # Salva em snapshot
            self._save_to_json(cleaned_event)
            # Salva em JSONL
            self._save_to_jsonl(cleaned_event)
            # Adiciona ao log visual
            self._add_visual_log_entry(cleaned_event)

    # ---------- Persistência JSON/JSONL ----------

    def _save_to_json(self, event: dict):
        """Salva evento em arquivo JSON com retry e fallback."""
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                events = []
                if self.snapshot_file.exists():
                    with open(self.snapshot_file, "r", encoding="utf-8") as f:
                        content = f.read()
                        if content:
                            events = json.loads(content)
                
                # Validação do evento
                if not isinstance(event, dict):
                    logging.error("Evento inválido: não é um dicionário")
                    return
                    
                events.append(event)
                
                with open(self.snapshot_file, "w", encoding="utf-8") as f:
                    json.dump(events, f, indent=4, ensure_ascii=False, default=str)
                return  # Sucesso
                
            except json.JSONDecodeError as e:
                logging.error(f"Erro de decodificação JSON (tentativa {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
            except PermissionError as e:
                logging.error(f"Erro de permissão ao salvar snapshot (tentativa {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
            except Exception as e:
                logging.error(f"Erro ao salvar snapshot (tentativa {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
        
        # Fallback: salva em diretório alternativo
        try:
            fallback_dir = Path("./fallback_events")
            fallback_dir.mkdir(exist_ok=True)
            fallback_file = fallback_dir / "eventos-fluxo.json"
            
            # Lê conteúdo existente do fallback
            fallback_events = []
            if fallback_file.exists():
                try:
                    with open(fallback_file, "r", encoding="utf-8") as f:
                        content = f.read()
                        if content:
                            fallback_events = json.loads(content)
                except Exception:
                    pass
            
            # Adiciona novo evento
            if isinstance(event, dict):
                fallback_events.append(event)
            
            # Salva no fallback
            with open(fallback_file, "w", encoding="utf-8") as f:
                json.dump(fallback_events, f, indent=4, ensure_ascii=False, default=str)
                
            logging.warning(f"⚠️ Salvamento fallback usado para snapshot: {fallback_file}")
            
        except Exception as e2:
            logging.critical(f"💀 FALHA TOTAL DE PERSISTÊNCIA para snapshot: {e2}")

    def _save_to_jsonl(self, event: dict):
        """Salva evento em arquivo JSONL com retry e fallback."""
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                if not isinstance(event, dict):
                    logging.error("Evento inválido: não é um dicionário")
                    return
                    
                with open(self.history_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
                return  # Sucesso
                
            except PermissionError as e:
                logging.error(f"Erro de permissão ao salvar JSONL (tentativa {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
            except Exception as e:
                logging.error(f"Erro ao salvar JSONL (tentativa {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
        
        # Fallback: salva em diretório alternativo
        try:
            fallback_dir = Path("./fallback_events")
            fallback_dir.mkdir(exist_ok=True)
            fallback_file = fallback_dir / "eventos_fluxo.jsonl"
            
            with open(fallback_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
                
            logging.warning(f"⚠️ Salvamento fallback usado para JSONL: {fallback_file}")
            
        except Exception as e2:
            logging.critical(f"💀 FALHA TOTAL DE PERSISTÊNCIA para JSONL: {e2}")

    # ---------- API Pública ----------

    def save_event(self, event: dict):
        """Salva evento com validação, horários SP/NY e contexto histórico vs real_time."""
        try:
            # Validação básica do evento
            if not isinstance(event, dict):
                logging.error("Tentativa de salvar evento inválido: não é um dicionário")
                return

            # 🔹 PRIORIZA epoch_ms SE DISPONÍVEL
            epoch_ms = event.get("epoch_ms")
            ts = event.get("timestamp")
            
            if epoch_ms:
                # Usa epoch_ms para gerar timestamps consistentes
                try:
                    dt = datetime.fromtimestamp(int(epoch_ms) / 1000, tz=UTC_TZ)
                    now = datetime.now(UTC_TZ)
                    
                    # Critério: se estiver mais de 1 dia à frente do relógio → histórico
                    if dt > now + timedelta(days=1):
                        event["data_context"] = "historical"
                    else:
                        event["data_context"] = "real_time"
                    
                    # Converte para SP/NY
                    event["time_ny"] = dt.astimezone(NY_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
                    event["time_sp"] = dt.astimezone(SP_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
                    
                    # Adiciona timestamp_utc se não existir
                    if "timestamp" not in event:
                        event["timestamp"] = dt.astimezone(UTC_TZ).isoformat(timespec="milliseconds")
                        
                except Exception as e:
                    logging.error(f"Erro ao processar epoch_ms: {e}")
                    event["data_context"] = "unknown"
                    
            elif ts:
                # Fallback: usa timestamp string
                try:
                    dt = self._parse_iso8601(ts)
                    now = datetime.now(UTC_TZ)

                    if dt > now + timedelta(days=1):
                        event["data_context"] = "historical"
                    else:
                        event["data_context"] = "real_time"

                    event["time_ny"] = dt.astimezone(NY_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
                    event["time_sp"] = dt.astimezone(SP_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
                except Exception as e:
                    logging.error(f"Erro ao converter timestamp do evento: {e}")
                    event["data_context"] = "unknown"
            else:
                event["data_context"] = "unknown"
                logging.warning("Evento sem epoch_ms nem timestamp - usando 'unknown'")

            # Adiciona separador para nova janela (prioriza window_id; fallback para legado)
            window_id = event.get("window_id") or event.get("candle_id_ms")
            if window_id and window_id != self.last_window_id:
                self._add_visual_separator(event)
                self.last_window_id = window_id

            # Bufferiza o evento
            with self._buffer_lock:
                self._write_buffer.append(event)

            # Alerta sonoro (não bufferizado)
            if self.sound_alert and event.get("is_signal", False):
                self._play_sound()
                
        except Exception as e:
            logging.error(f"Erro ao processar evento para salvamento: {e}")

    # ---------- Saídas visuais ----------

    def _add_visual_separator(self, event: dict):
        """Adiciona um separador visual no arquivo de log."""
        try:
            # 🔹 USA epoch_ms DO EVENTO SE DISPONÍVEL
            epoch_ms = event.get("epoch_ms") or event.get("window_close_ms")
            
            if epoch_ms:
                try:
                    dt_utc = datetime.fromtimestamp(int(epoch_ms) / 1000, tz=UTC_TZ)
                    timestamp_ny = dt_utc.astimezone(NY_TZ).isoformat(timespec="seconds")
                except Exception:
                    timestamp_ny = self.time_manager.now_iso(tz=NY_TZ)
            else:
                timestamp_ny = self.time_manager.now_iso(tz=NY_TZ)
            
            open_ms = event.get("window_open_ms")
            close_ms = event.get("window_close_ms")

            # Compatibilidade com eventos legados
            if open_ms is None:
                open_ms = event.get("candle_open_time_ms")
            if close_ms is None:
                close_ms = event.get("candle_close_time_ms")

            def _fmt(ms):
                if not ms or ms <= 0:
                    return "N/A"
                return datetime.fromtimestamp(ms / 1000, tz=NY_TZ).isoformat(timespec="seconds")

            start_time = _fmt(open_ms)
            end_time = _fmt(close_ms)
            
            separator = f"\n{timestamp_ny} | --- INÍCIO DE NOVA JANELA --- | {start_time} --> {end_time}\n"
            with open(self.visual_log_file, "a", encoding="utf-8") as f:
                f.write(separator)
        except Exception as e:
            logging.error(f"Erro ao adicionar separador visual: {e}")
            # Fallback: salva em diretório alternativo
            try:
                fallback_dir = Path("./fallback_events")
                fallback_dir.mkdir(exist_ok=True)
                fallback_file = fallback_dir / "eventos_visuais.log"
                with open(fallback_file, "a", encoding="utf-8") as f:
                    f.write(f"Erro ao adicionar separador: {e}\n")
            except Exception as e2:
                logging.critical(f"💀 FALHA TOTAL ao salvar separador visual: {e2}")

    def _format_value_for_display(self, value, key=""):
        """Formata valor para exibição no log visual usando as funções do format_utils."""
        if value is None:
            return "null"
        
        # Determina o tipo baseado no nome da chave
        key_lower = key.lower()
        
        try:
            if isinstance(value, (int, float)):
                # Preços
                if any(x in key_lower for x in ['price', 'preco', 'poc', 'val', 'vah', 'level']):
                    return format_price(value)
                # Quantidades/Volumes
                elif any(x in key_lower for x in ['volume', 'quantity', 'size', 'count']):
                    return format_large_number(value)
                # Deltas
                elif 'delta' in key_lower or 'flow' in key_lower:
                    return format_delta(value)
                # Percentuais
                elif any(x in key_lower for x in ['pct', 'percent', 'ratio', 'prob']):
                    return format_percent(value)
                # Científicos
                elif any(x in key_lower for x in ['volatility', 'returns', 'slope', 'momentum']):
                    return format_scientific(value)
                else:
                    return str(value)
            else:
                return str(value)
        except:
            return str(value)

    def _compact_json_log(self, data, indent=2, parent_key=""):
        """Função customizada para serializar JSON com formatação adequada de números."""
        
        # 🔹 LISTA DE CAMPOS QUE DEVEM PERMANECER COMO NÚMEROS INTEIROS (sem formatação)
        timestamp_fields = {
            'epoch_ms', 'window_open_ms', 'window_close_ms', 
            'window_duration_ms', 'T', 'E', 'tradeTime',
            'candle_open_time_ms', 'candle_close_time_ms',
            'last_successful_sync_ms', 'age_ms', 'timestamp_ms'
        }
        
        if isinstance(data, dict):
            # Formata cada item do dicionário
            formatted_items = []
            for key, value in data.items():
                # 🔹 CORREÇÃO: Não formatar campos de timestamp/epoch
                if key in timestamp_fields or key.endswith('_ms') or key.endswith('_ts'):
                    # Mantém valores de timestamp como números puros
                    if isinstance(value, (int, float)):
                        formatted_items.append(f'{" " * (indent + 2)}"{key}": {int(value)}')
                    else:
                        formatted_items.append(f'{" " * (indent + 2)}"{key}": {json.dumps(value)}')
                
                # Para campos booleanos especiais
                elif key in ['is_signal', 'units_check_passed'] or key.startswith('is_'):
                    if isinstance(value, bool):
                        formatted_items.append(f'{" " * (indent + 2)}"{key}": {str(value).lower()}')
                    else:
                        formatted_items.append(f'{" " * (indent + 2)}"{key}": {json.dumps(value)}')
                
                # Para listas de números (hvns, lvns, etc.)
                elif isinstance(value, list) and all(isinstance(i, (int, float)) for i in value):
                    # Formata cada número da lista apropriadamente
                    if any(x in key.lower() for x in ['hvn', 'lvn', 'price', 'level']):
                        formatted_list = [format_price(v) for v in value]
                    else:
                        formatted_list = [str(v) for v in value]
                    list_str = "[" + ", ".join(formatted_list) + "]"
                    formatted_items.append(f'{" " * (indent + 2)}"{key}": {list_str}')
                
                else:
                    # Recursivamente formata outros valores
                    value_str = self._compact_json_log(value, indent + 2, parent_key=key)
                    formatted_items.append(f'{" " * (indent + 2)}"{key}": {value_str}')
            
            return "{\n" + ",\n".join(formatted_items) + f"\n{' ' * indent}}}"
        
        elif isinstance(data, list):
            # Para listas de objetos, mantém formatação padrão
            if any(isinstance(i, dict) for i in data):
                return json.dumps(data, ensure_ascii=False, default=str, indent=indent)
            # Para listas simples de números
            elif all(isinstance(i, (int, float)) for i in data):
                if any(x in parent_key.lower() for x in ['hvn', 'lvn', 'price', 'level']):
                    formatted = [format_price(v) for v in data]
                else:
                    formatted = [str(v) for v in data]
                return "[" + ", ".join(formatted) + "]"
            else:
                return json.dumps(data, ensure_ascii=False, default=str)
        
        elif isinstance(data, (int, float)):
            # 🔹 CORREÇÃO PRINCIPAL: Não formatar timestamps
            if parent_key in timestamp_fields or parent_key.endswith('_ms') or parent_key.endswith('_ts'):
                return str(int(data))  # Retorna como número puro, sem formatação
            
            # Para outros valores booleanos numéricos (0/1)
            elif parent_key in ['is_signal'] or parent_key.startswith('is_'):
                return str(bool(data)).lower()
            
            # Para outros números, formata para exibição
            else:
                return f'"{self._format_value_for_display(data, parent_key)}"'
        
        elif isinstance(data, bool):
            return str(data).lower()
        
        elif isinstance(data, str):
            # Para strings, usa escape JSON padrão
            return json.dumps(data, ensure_ascii=False)
        
        else:
            # Para outros tipos, usa a representação padrão
            return json.dumps(data, ensure_ascii=False, default=str)

    def _add_visual_log_entry(self, event: dict):
        """Log visual amigável com formatação adequada e JSON válido."""
        try:
            # 🔹 PRIORIZA epoch_ms PARA GERAR HORÁRIOS
            epoch_ms = event.get("epoch_ms")
            ts = event.get("timestamp")
            
            minute_block = None
            utc_header = ny_header = sp_header = None
            context = event.get("data_context", "unknown")

            if epoch_ms:
                try:
                    dt = datetime.fromtimestamp(int(epoch_ms) / 1000, tz=UTC_TZ)
                    minute_key = dt.strftime("%Y-%m-%d %H:%M")
                    minute_block = f"{minute_key}|{context}"

                    utc_header = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
                    ny_header = dt.astimezone(NY_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
                    sp_header = dt.astimezone(SP_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
                except Exception as e:
                    logging.error(f"Erro ao processar epoch_ms no log visual: {e}")
                    
            elif ts:
                try:
                    dt = self._parse_iso8601(ts)
                    minute_key = dt.astimezone(UTC_TZ).strftime("%Y-%m-%d %H:%M")
                    minute_block = f"{minute_key}|{context}"

                    utc_header = dt.astimezone(UTC_TZ).strftime("%Y-%m-%d %H:%M:%S UTC")
                    ny_header = dt.astimezone(NY_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
                    sp_header = dt.astimezone(SP_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
                except Exception as e:
                    logging.error(f"Erro ao interpretar timestamp no log visual: {e}")

            with open(self.visual_log_file, "a", encoding="utf-8") as f:
                if minute_block and minute_block != self._last_logged_block:
                    f.write("\n" + "="*150 + "\n")
                    if utc_header: f.write(f"HORÁRIO UTC: {utc_header}\n")
                    if ny_header: f.write(f"HORÁRIO NY:  {ny_header}\n")
                    if sp_header: f.write(f"HORÁRIO SP:  {sp_header}\n")
                    f.write(f"CONTEXT: {context}\n")
                    f.write("------------------------------\n")
                    self._last_logged_block = minute_block
                    self._seen_in_block.clear()

                dedupe_key = (
                    str(event.get("timestamp")),
                    str(event.get("tipo_evento")),
                    str(event.get("resultado_da_batalha")),
                )
                if dedupe_key in self._seen_in_block:
                    return
                self._seen_in_block.add(dedupe_key)

                clean = dict(event)
                clean.pop("time_ny", None)
                clean.pop("time_sp", None)
                if "timestamp" in clean:
                    clean["timestamp_utc"] = clean.pop("timestamp")

                # CORREÇÃO: Usar json.dump para garantir um JSON válido e f.flush() para evitar truncamento.
                log_string = json.dumps(clean, indent=4, ensure_ascii=False, default=str)
                f.write(log_string + "\n")
                f.flush() # Força a escrita para o disco

        except Exception as e:
            logging.error(f"Erro ao adicionar entrada visual: {e}")
            try:
                fallback_dir = Path("./fallback_events")
                fallback_dir.mkdir(exist_ok=True)
                fallback_file = fallback_dir / "eventos_visuais.log"
                with open(fallback_file, "a", encoding="utf-8") as f:
                    f.write(f"Erro ao adicionar entrada: {e}\nEvento: {json.dumps(event, ensure_ascii=False, default=str)}\n")
                    f.flush()
            except Exception as e2:
                logging.critical(f"💀 FALHA TOTAL ao salvar entrada visual: {e2}")

    # ---------- Alerta sonoro ----------

    def _play_sound(self):
        """Reproduz um som de alerta dependendo do sistema operacional."""
        try:
            if platform.system() == "Windows":
                import winsound
                winsound.Beep(1000, 500)
            elif platform.system() == "Darwin":
                import os
                os.system('afplay /System/Library/Sounds/Glass.aiff')
            else:
                import os
                os.system('paplay /usr/share/sounds/freedesktop/stereo/bell.oga &')
        except Exception as e:
            logging.warning(f"[ALERTA] Não foi possível reproduzir som: {e}")
            print("\n🔔 ALERTA SONORO: Evento detectado! 🔔\n")