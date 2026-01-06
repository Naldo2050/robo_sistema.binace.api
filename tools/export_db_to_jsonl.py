import json
import sqlite3
from pathlib import Path

DB = Path("dados/trading_bot.db")
OUTDIR = Path("exports_jsonl")
LIMIT = 5000  # últimos N registros por tabela (ajuste se quiser)

OUTDIR.mkdir(parents=True, exist_ok=True)

con = sqlite3.connect(DB)
con.row_factory = sqlite3.Row
cur = con.cursor()

tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

for t in tables:
    out = OUTDIR / f"{t}.jsonl"
    try:
        # tenta pegar os últimos registros (rowid costuma existir na maioria das tabelas)
        rows = cur.execute(f'SELECT * FROM "{t}" ORDER BY rowid DESC LIMIT ?', (LIMIT,)).fetchall()
        rows = list(reversed(rows))
    except Exception:
        rows = cur.execute(f'SELECT * FROM "{t}" LIMIT ?', (LIMIT,)).fetchall()

    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            d = dict(r)
            f.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")

    print(f"OK: {t} -> {out} ({len(rows)} linhas)")