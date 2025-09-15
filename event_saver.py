import json
from pathlib import Path
import platform
import logging
import threading
import time
import random
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from time_manager import TimeManager

NY_TZ = ZoneInfo("America/New_York")

DATA_DIR = Path("dados")
DATA_DIR.mkdir(exist_ok=True)

class EventSaver:
    def __init__(self, sound_alert=True):
        self.sound_alert = sound_alert
        self.snapshot_file = DATA_DIR / "eventos-fluxo.json"
        self.history_file = DATA_DIR / "eventos_fluxo.jsonl"
        self.visual_log_file = DATA_DIR / "eventos_visuais.log"
        self.last_candle_id = None
        self.time_manager = TimeManager()

        # 🔹 NOVO: buffer de escrita + thread de flush (Fase 2)
        self._write_buffer = []
        self._buffer_lock = threading.Lock()
        self._flush_interval = 5  # segundos
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

    def _flush_loop(self):
        """Thread que esvazia o buffer periodicamente."""
        while True:
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
        
        # 🔹 Fallback: salva em diretório alternativo
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
        
        # 🔹 Fallback: salva em diretório alternativo
        try:
            fallback_dir = Path("./fallback_events")
            fallback_dir.mkdir(exist_ok=True)
            fallback_file = fallback_dir / "eventos_fluxo.jsonl"
            
            with open(fallback_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
                
            logging.warning(f"⚠️ Salvamento fallback usado para JSONL: {fallback_file}")
            
        except Exception as e2:
            logging.critical(f"💀 FALHA TOTAL DE PERSISTÊNCIA para JSONL: {e2}")

    def save_event(self, event: dict):
        """Salva evento com validação e fallback."""
        try:
            # Validação básica do evento
            if not isinstance(event, dict):
                logging.error("Tentativa de salvar evento inválido: não é um dicionário")
                return
                
            # Adiciona separador para nova janela
            candle_id = event.get("candle_id_ms")
            if candle_id and candle_id != self.last_candle_id:
                self._add_visual_separator(event)
                self.last_candle_id = candle_id
            
            # Bufferiza o evento
            with self._buffer_lock:
                self._write_buffer.append(event)
            
            # Alerta sonoro (não bufferizado)
            if self.sound_alert and event.get("is_signal", False):
                self._play_sound()
                
        except Exception as e:
            logging.error(f"Erro ao processar evento para salvamento: {e}")

    def _add_visual_separator(self, event: dict):
        """Adiciona um separador visual no arquivo de log."""
        try:
            timestamp_ny = self.time_manager.now_iso(tz=NY_TZ)
            
            start_time = datetime.fromtimestamp(event.get('candle_open_time_ms', 0) / 1000, tz=NY_TZ).isoformat(timespec="seconds")
            end_time = datetime.fromtimestamp(event.get('candle_close_time_ms', 0) / 1000, tz=NY_TZ).isoformat(timespec="seconds")
            
            separator = f"\n{timestamp_ny} | --- INÍCIO DE NOVA JANELA --- | {start_time} --> {end_time}\n"
            with open(self.visual_log_file, "a", encoding="utf-8") as f:
                f.write(separator)
        except Exception as e:
            logging.error(f"Erro ao adicionar separador visual: {e}")
            # 🔹 Fallback: salva em diretório alternativo
            try:
                fallback_dir = Path("./fallback_events")
                fallback_dir.mkdir(exist_ok=True)
                fallback_file = fallback_dir / "eventos_visuais.log"
                with open(fallback_file, "a", encoding="utf-8") as f:
                    f.write(f"Erro ao adicionar separador: {e}\n")
            except Exception as e2:
                logging.critical(f"💀 FALHA TOTAL ao salvar separador visual: {e2}")

    def _add_visual_log_entry(self, event: dict):
        """Adiciona uma entrada formatada e detalhada no arquivo de log visual."""
        try:
            with open(self.visual_log_file, "a", encoding="utf-8") as f:
                f.write("{\n")
                for key, value in event.items():
                    if isinstance(value, (int, float)):
                        f.write(f"  \"{key}\": {value},\n")
                    elif isinstance(value, str):
                        # Escapa caracteres especiais na string
                        escaped_value = value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                        f.write(f"  \"{key}\": \"{escaped_value}\",\n")
                    else:
                        # Converte qualquer outro tipo para string
                        f.write(f"  \"{key}\": {json.dumps(value, ensure_ascii=False, default=str)},\n")
                f.write("}\n")
        except Exception as e:
            logging.error(f"Erro ao adicionar entrada visual: {e}")
            # 🔹 Fallback: salva em diretório alternativo
            try:
                fallback_dir = Path("./fallback_events")
                fallback_dir.mkdir(exist_ok=True)
                fallback_file = fallback_dir / "eventos_visuais.log"
                with open(fallback_file, "a", encoding="utf-8") as f:
                    f.write(f"Erro ao adicionar entrada: {e}\nEvento: {json.dumps(event, ensure_ascii=False, default=str)}\n")
            except Exception as e2:
                logging.critical(f"💀 FALHA TOTAL ao salvar entrada visual: {e2}")

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