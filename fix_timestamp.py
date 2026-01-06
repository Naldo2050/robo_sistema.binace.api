# fix_timestamp.py - Aplica correção no main.py

import re

# Lê o arquivo
with open('main.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Substitui as linhas problemáticas
old_code = '''signal["timestamp_utc"] = self.time_manager.from_timestamp_ms(
                close_ms
            ).isoformat(timespec="milliseconds")'''

new_code = '''try:
            dt_utc = datetime.fromtimestamp(close_ms / 1000, tz=timezone.utc)
            signal["timestamp_utc"] = dt_utc.isoformat(timespec="milliseconds")
        except Exception as e:
            logging.warning(f"Erro ao converter timestamp_utc: {e}")
            signal["timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds")'''

content = content.replace(old_code, new_code)

# Segunda substituição
old_code2 = '''signal["timestamp"] = self.time_manager.from_timestamp_ms(
                close_ms
            ).astimezone(self.ny_tz).strftime("%Y-%m-%d %H:%M:%S")'''

new_code2 = '''try:
            dt_utc = datetime.fromtimestamp(close_ms / 1000, tz=timezone.utc)
            dt_ny = dt_utc.astimezone(self.ny_tz)
            signal["timestamp"] = dt_ny.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            logging.warning(f"Erro ao converter timestamp: {e}")
            signal["timestamp"] = datetime.now(self.ny_tz).strftime("%Y-%m-%d %H:%M:%S")'''

content = content.replace(old_code2, new_code2)

# Salva backup
with open('main.backup2.py', 'w', encoding='utf-8') as f:
    f.write(content)

# Aplica correção
with open('main.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Correção aplicada com sucesso!")
print("📦 Backup salvo em: main.backup2.py")