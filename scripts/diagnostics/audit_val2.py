import sqlite3
import json
import re

with open('logs/collect_2h_20260906_203537.log', 'r', encoding='utf-8', errors='replace') as f:
    text = f.read()

count_ml = len(re.findall(r'ML neutralizado', text))
print(f'Log count ML neutralizado: {count_ml}')

conn = sqlite3.connect('dados/trading_bot.db')
cur = conn.cursor()
cur.execute('SELECT id, payload FROM events')
rows = cur.fetchall()

v_futures = 0
v_stale = 0
ml_present_count = 0

for r in rows:
    p_str = r[1] if isinstance(r[1], str) else json.dumps(r[1])
    if 'valid_for_futures' in p_str:
        ml_present_count += 1
    if '"valid_for_futures": true' in p_str.lower() or '"valid_for_futures":true' in p_str.lower():
        v_futures += 1
    if '"ml_stale": false' in p_str.lower() or '"ml_stale":false' in p_str.lower():
        v_stale += 1

print(f'Total eventos SQLite: {len(rows)}')
print(f'Eventos com metadados ML: {ml_present_count}')
print(f'Violacoes valid_for_futures=True: {v_futures}')
print(f'Violacoes ml_stale=False: {v_stale}')
