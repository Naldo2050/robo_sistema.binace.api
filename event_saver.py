# event_saver.py
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

    # ---------- Utilidades internas ----------

    @staticmethod
    def _parse_iso8601(ts: str) -> datetime:
        """
        Aceita ISO-8601 com 'Z' (UTC) e com offset.
        """
        try:
            # Normaliza 'Z' -> +00:00 para compatibilidade ampla
            if ts.endswith("Z"):
                ts = ts[:-1] + "+00:00"
            return datetime.fromisoformat(ts)
        except Exception:
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
            # Salva em snapshot
            self._save_to_json(event)
            # Salva em JSONL
            self._save_to_jsonl(event)
            # Adiciona ao log visual
            self._add_visual_log_entry(event)

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

            # Conversão de timestamp + contexto
            ts = event.get("timestamp")
            if ts:
                try:
                    dt = self._parse_iso8601(ts)  # timezone-aware
                    now = datetime.now(UTC_TZ)

                    # Critério simples: se estiver mais de 1 dia à frente do relógio → histórico
                    if dt > now + timedelta(days=1):
                        event["data_context"] = "historical"
                    else:
                        event["data_context"] = "real_time"

                    # Converte para SP/NY (para salvar nos JSONs)
                    event["time_ny"] = dt.astimezone(NY_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
                    event["time_sp"] = dt.astimezone(SP_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
                except Exception as e:
                    logging.error(f"Erro ao converter timestamp do evento: {e}")
                    event["data_context"] = "unknown"
            else:
                event["data_context"] = "unknown"

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

    def _add_visual_log_entry(self, event: dict):
        """
        Log visual amigável:
        - Cabeçalho (UTC, NY, SP, CONTEXT) 1x por minuto+contexto.
        - Não repetir time_ny/time_sp dentro do corpo do evento.
        - Exibir o timestamp do evento como 'timestamp_utc' para não confundir.
        - Filtro anti-duplicado opcional por bloco (timestamp+tipo_evento+descricao).
        """
        try:
            ts = event.get("timestamp")
            minute_block = None
            utc_header = ny_header = sp_header = None
            context = event.get("data_context", "unknown")

            if ts:
                try:
                    dt = self._parse_iso8601(ts)  # aware
                    # Bloco: minuto em UTC para ser consistente
                    minute_key = dt.astimezone(UTC_TZ).strftime("%Y-%m-%d %H:%M")
                    minute_block = f"{minute_key}|{context}"

                    # Cabeçalho com os 3 fusos (para acabar com a dúvida)
                    utc_header = dt.astimezone(UTC_TZ).strftime("%Y-%m-%d %H:%M:%S UTC")
                    ny_header = dt.astimezone(NY_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
                    sp_header = dt.astimezone(SP_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
                except Exception as e:
                    logging.error(f"Erro ao interpretar timestamp no log visual: {e}")

            with open(self.visual_log_file, "a", encoding="utf-8") as f:
                # Se mudou bloco (minuto+contexto), imprime cabeçalho e reseta anti-duplicado
                if minute_block and minute_block != self._last_logged_block:
                    f.write("\n" + "="*150 + "\n")
                    if utc_header:
                        f.write(f"HORÁRIO UTC: {utc_header}\n")
                    if ny_header:
                        f.write(f"HORÁRIO NY:  {ny_header}\n")
                    if sp_header:
                        f.write(f"HORÁRIO SP:  {sp_header}\n")
                    f.write(f"CONTEXT: {context}\n")
                    f.write("------------------------------\n")
                    self._last_logged_block = minute_block
                    self._seen_in_block.clear()

                # Anti-duplicado simples no LOG por minuto+contexto (opcional)
                dedupe_key = (
                    str(event.get("timestamp")),
                    str(event.get("tipo_evento")),
                    str(event.get("resultado_da_batalha")),
                    str(event.get("descricao")),
                )
                if dedupe_key in self._seen_in_block:
                    return
                self._seen_in_block.add(dedupe_key)

                # JSON visual limpo: remove time_ny/time_sp e renomeia timestamp -> timestamp_utc
                clean = dict(event)
                clean.pop("time_ny", None)
                clean.pop("time_sp", None)
                if "timestamp" in clean:
                    clean["timestamp_utc"] = clean.pop("timestamp")

                f.write(json.dumps(clean, ensure_ascii=False, default=str, indent=2) + "\n")
        except Exception as e:
            logging.error(f"Erro ao adicionar entrada visual: {e}")
            # Fallback: salva em diretório alternativo
            try:
                fallback_dir = Path("./fallback_events")
                fallback_dir.mkdir(exist_ok=True)
                fallback_file = fallback_dir / "eventos_visuais.log"
                with open(fallback_file, "a", encoding="utf-8") as f:
                    f.write(f"Erro ao adicionar entrada: {e}\nEvento: {json.dumps(event, ensure_ascii=False, default=str)}\n")
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
