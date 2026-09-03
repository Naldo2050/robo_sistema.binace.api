# scripts/diagnostics/audit_data_attrition.py
# -*- coding: utf-8 -*-
"""
Auditoria de Attrition e Origem Amostral dos Datasets — Fase V1.1.
Rastreia exatamente onde e por que registros são filtrados.
"""

import json
import os
import sqlite3
import pandas as pd

def audit_attrition():
    print("=" * 80)
    print("AUDITORIA DE ATTRITION AMOSTRAL (FASE V1.1)")
    print("=" * 80)

    db_path = "dados/trading_bot.db"
    jsonl_path = "dados/eventos_fluxo.jsonl"

    # 1. Tabela events no SQLite
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM events")
    total_events = cur.fetchone()[0]

    cur.execute("SELECT event_type, count(*) FROM events GROUP BY event_type")
    events_by_type = dict(cur.fetchall())

    cur.execute("SELECT count(*) FROM positioning_shadow_dataset")
    total_pos = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM signal_outcomes")
    total_outcomes = cur.fetchone()[0]
    conn.close()

    # 2. JSONL
    total_jsonl = 0
    jsonl_by_type = {}
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    total_jsonl += 1
                    try:
                        d = json.loads(line)
                        t = d.get("tipo_evento") or d.get("event_type") or "unknown"
                        jsonl_by_type[t] = jsonl_by_type.get(t, 0) + 1
                    except Exception:
                        pass

    # 3. Filtragem pelo extrator de features
    valid_payload_rows = 0
    missing_price_rows = 0
    has_pos_count = 0
    has_vwap_count = 0
    has_ms_count = 0

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT timestamp_ms, payload FROM events ORDER BY timestamp_ms ASC")
    for ts_ms, payload_str in cur.fetchall():
        if not payload_str:
            continue
        try:
            p = json.loads(payload_str)
            price_dict = p.get("price") or p.get("p") or {}
            price = float(price_dict.get("c") or p.get("preco_fechamento") or 0.0)
            if price > 0:
                valid_payload_rows += 1
                if p.get("pos"):
                    has_pos_count += 1
                if p.get("vwap"):
                    has_vwap_count += 1
                if p.get("ms"):
                    has_ms_count += 1
            else:
                missing_price_rows += 1
        except Exception:
            missing_price_rows += 1
    conn.close()

    print(f"\n1. CONTAGEM BRUTA POR FONTE (RAW ROWS):")
    print(f"   - Tabela 'events' (SQLite):                  {total_events} linhas")
    print(f"     * Por tipo: {events_by_type}")
    print(f"   - Arquivo 'eventos_fluxo.jsonl':            {total_jsonl} linhas")
    print(f"     * Por tipo: {jsonl_by_type}")
    print(f"   - Tabela 'positioning_shadow_dataset':       {total_pos} linhas")
    print(f"   - Tabela 'signal_outcomes':                  {total_outcomes} linhas")

    print(f"\n2. ATTRITION TABLE (FILTRAGEM DE EXTRAÇÃO):")
    print(f"   - Linhas lidas em events:                    {total_events}")
    print(f"   - Descartadas por falta de preço/fechamento: {missing_price_rows} (alertas/logs sem OHLC)")
    print(f"   - Linhas com preço válido (Extraídas):       {valid_payload_rows}")
    print(f"   - Linhas com seção 'pos' (Positioning):      {has_pos_count}")
    print(f"   - Linhas com seção 'vwap' (Session VWAP):    {has_vwap_count}")
    print(f"   - Linhas com seção 'ms' (Market Structure):  {has_ms_count}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    audit_attrition()
