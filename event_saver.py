import json
from pathlib import Path
import platform
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

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

    def save_event(self, event: dict):
        # 1. Adiciona um separador ao log visual no início de uma nova janela.
        candle_id = event.get("candle_id_ms")
        if candle_id and candle_id != self.last_candle_id:
            self._add_visual_separator(event)
            self.last_candle_id = candle_id
        
        # 2. Salva o evento em formato JSON.
        events = []
        if self.snapshot_file.exists():
            try:
                with open(self.snapshot_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content:
                        events = json.loads(content)
            except (json.JSONDecodeError, FileNotFoundError):
                logging.warning(f"Arquivo {self.snapshot_file} corrompido ou vazio. Iniciando um novo.")
                events = []
            except Exception as e:
                logging.error(f"Erro ao ler snapshot: {e}")
                events = []

        events.append(event)
        with open(self.snapshot_file, "w", encoding="utf-8") as f:
            json.dump(events, f, indent=4, ensure_ascii=False)

        # 3. Salva o evento no arquivo JSONL.
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

        # 4. Adiciona o evento formatado no log visual.
        self._add_visual_log_entry(event)

        # 5. Alerta sonoro.
        if self.sound_alert and event.get("is_signal", False):
            self._play_sound()

    def _add_visual_separator(self, event: dict):
        """Adiciona um separador visual no arquivo de log."""
        timestamp_ny = datetime.now(NY_TZ).isoformat(timespec="seconds")
        
        start_time = datetime.fromtimestamp(event.get('candle_open_time_ms', 0) / 1000, tz=NY_TZ).isoformat(timespec="seconds")
        end_time = datetime.fromtimestamp(event.get('candle_close_time_ms', 0) / 1000, tz=NY_TZ).isoformat(timespec="seconds")
        
        separator = f"\n{timestamp_ny} | --- INÍCIO DE NOVA JANELA --- | {start_time} --> {end_time}\n"
        with open(self.visual_log_file, "a", encoding="utf-8") as f:
            f.write(separator)

    def _add_visual_log_entry(self, event: dict):
        """Adiciona uma entrada formatada e detalhada no arquivo de log visual."""
        with open(self.visual_log_file, "a", encoding="utf-8") as f:
            f.write("{\n")
            for key, value in event.items():
                if isinstance(value, (int, float)):
                    f.write(f"  \"{key}\": {value},\n")
                elif isinstance(value, str):
                    f.write(f"  \"{key}\": \"{value}\",\n")
                else:
                    f.write(f"  \"{key}\": {value},\n")
            f.write("}\n")

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