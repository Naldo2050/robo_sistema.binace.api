# teste_separador.py - Debug de separadores
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch


def main() -> int:
    temp_dir = os.environ.get("LEGACY_SEPARADOR_TEMP_DIR")
    if not temp_dir:
        print(
            "[ABORTADO] Defina LEGACY_SEPARADOR_TEMP_DIR com um diretório "
            "temporário explícito; nenhuma escrita operacional será feita."
        )
        return 2

    temp_path = Path(temp_dir).expanduser().resolve()
    if not temp_path.is_dir():
        print(f"[ABORTADO] Diretório temporário inexistente: {temp_path}")
        return 2

    from database.event_store import EventStore
    from events import event_saver
    from events.event_saver import EventSaver

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    old_cwd = os.getcwd()
    os.chdir(temp_path)
    saver = None
    try:
        with patch.object(event_saver, "DATA_DIR", temp_path), \
             patch.object(event_saver, "HAS_CLOCK_SYNC", False), \
             patch.object(
                 event_saver,
                 "EventStore",
                 lambda: EventStore(db_path=str(temp_path / "trading_bot.db")),
             ), \
             patch.object(
                 event_saver.config,
                 "EVENT_SAVER_JSONL_ARCHIVE_DIR",
                 str(temp_path / "archive"),
                 create=True,
             ):
            print("TESTE DE SEPARADORES\n")
            print("=" * 80)

            saver = EventSaver(sound_alert=False)

            for i in range(5):
                epoch_base = int(datetime.now().timestamp() * 1000)
                epoch_minuto = (epoch_base // 60000) * 60000
                epoch_minuto += i * 60000

                evento = {
                    "tipo_evento": f"TESTE_JANELA_{i + 1}",
                    "is_signal": True,
                    "epoch_ms": epoch_minuto,
                    "window_id": epoch_minuto,
                    "price_data": {
                        "current": {
                            "last": 110000 + i * 100,
                            "volume": 100 + i,
                        }
                    },
                }

                print(f"Salvando evento {i + 1} (epoch: {epoch_minuto})...")
                saver.save_event(evento)
                time.sleep(0.5)

            print("\nAguardando flush...")
            time.sleep(7)

            print("\nEstatisticas:")
            stats = saver.get_stats()
            for k, v in stats.items():
                if isinstance(v, dict):
                    print(f"  {k}:")
                    for k2, v2 in v.items():
                        print(f"    {k2}: {v2}")
                else:
                    print(f"  {k}: {v}")

            print("\nTeste concluido!")
            print("=" * 80)
    finally:
        if saver is not None:
            try:
                saver.stop()
            except Exception:
                pass
        os.chdir(old_cwd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
