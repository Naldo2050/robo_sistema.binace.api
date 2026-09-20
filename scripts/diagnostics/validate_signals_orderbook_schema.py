#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/diagnostics/validate_signals_orderbook_schema.py

Validação 5: Paridade de schema orderbook_data em sinais reais persistidos no SQLite.
Schema canônico confirmado:
  events(id, timestamp_ms, event_type, symbol, window_id, is_signal, payload, created_at)
"""
import sys
import os
import json
import sqlite3
import argparse

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def validate_signals_schema(db_path: str = "dados/trading_bot.db"):
    print("=" * 80)
    print("VALIDAÇÃO 5 — PARIDADE DE SCHEMA ORDERBOOK_DATA EM SINAIS REAIS")
    print(f"Banco avaliado: {db_path}")
    print("=" * 80)

    if not os.path.exists(db_path):
        print(f"❌ Banco {db_path} não encontrado!")
        return 1

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        # Consulta contra as colunas reais confirmadas no SQLite: event_type e payload
        cur.execute("SELECT event_type, payload FROM events WHERE is_signal = 1")
        rows = cur.fetchall()
    except Exception as e:
        print(f"❌ Erro ao consultar tabela 'events': {e}")
        conn.close()
        return 1
    conn.close()

    print(f"Total de sinais reais encontrados (is_signal = 1): {len(rows)}")

    if len(rows) == 0:
        print("ℹ️ Nenhum sinal com is_signal = 1 encontrado no banco de dados.")
        print("Aguardando execução da sessão contínua para captura de sinais reais de produção.")
        return 0

    faltando = []
    campos_esperados = {"source", "timestamps", "snapshot_offset_ms", "source_type"}

    for tipo, raw in rows:
        d = json.loads(raw) if isinstance(raw, str) else (raw or {})
        ob = d.get("orderbook_data") or {}
        if not isinstance(ob, dict):
            faltando.append((tipo, {"orderbook_data_ausente"}))
            continue

        faltantes = campos_esperados - set(ob.keys())
        if faltantes:
            faltando.append((tipo, faltantes))

    print(f"Sinais analisados: {len(rows)}")
    print(f"Sinais com campos faltantes no bloco orderbook_data: {len(faltando)}")
    for t, f in faltando[:20]:
        print(f"  • Tipo: {t} | Campos ausentes: {f}")

    if len(faltando) == 0:
        print("✅ CONFORME: 100% dos sinais possuem o bloco orderbook_data completo!")
        return 0
    else:
        print("❌ INCONFORME: Existem sinais com campos ausentes no orderbook_data.")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validação de paridade do schema orderbook_data.")
    parser.add_argument("pos_db", nargs="?", default=None, help="Caminho posicional do banco")
    parser.add_argument("--db", default=None, help="Caminho do banco SQLite")
    args = parser.parse_args()

    target_db = args.db or args.pos_db or "dados/trading_bot.db"
    sys.exit(validate_signals_schema(target_db))
