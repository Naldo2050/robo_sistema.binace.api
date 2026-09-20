#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/diagnostics/analyze_orderbook_sync_session.py

Script de análise estatística de alta resolução para validação pós-coleta de 2h:
1. Distribuição completa de snapshot_offset_ms (p10, p50, p90, p99, max, min)
2. Taxa percentual de eventos com source="live_sync" vs source="cache_bg"
3. Verificação de conformidade do timeout de 1500ms:
   - Identifica se algum evento marcado como live_sync excedeu 1500ms (vazamento de timeout)
   - Avalia latência de acionamento do fallback
"""
import argparse
import os
import sys
import json
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def analyze_session_orderbook_sync(
    db_path: str = "dados/trading_bot.db",
    session_filter: str = None,
    max_timeout_ms: float = 1500.0
):
    print("=" * 80)
    print("ANÁLISE ESTATÍSTICA DE SNAPSHOT DO ORDERBOOK (SESSÃO DE 2 HORAS)")
    print("=" * 80)

    if not os.path.exists(db_path):
        print(f"❌ Banco de dados {db_path} não encontrado!")
        print("Este script deve ser executado após a conclusão da coleta oficial (Item 8).")
        return 1

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        # Schema confirmado: events(id, timestamp_ms, event_type, symbol, window_id, is_signal, payload, created_at)
        query = "SELECT id, timestamp_ms, event_type, window_id, payload FROM events"
        params = []
        if session_filter:
            query += " WHERE window_id LIKE ? OR created_at LIKE ?"
            params.extend([f"%{session_filter}%", f"%{session_filter}%"])
        query += " ORDER BY timestamp_ms ASC"

        cursor.execute(query, params)
        rows = cursor.fetchall()
    except Exception as e:
        print(f"❌ Erro ao consultar tabela 'events': {e}")
        conn.close()
        return 1
    conn.close()

    print(f"Total de eventos brutos carregados: {len(rows)}")

    if len(rows) == 0:
        print("ℹ️ A tabela 'events' existe no schema confirmado, mas contém 0 registros.")
        print("Aguardando execução da coleta contínua de 2h para gerar massa de dados.")
        return 0

    records = []
    for r in rows:
        ev_id, ts_ms, ev_type, win_id, payload_raw = r
        payload = json.loads(payload_raw) if isinstance(payload_raw, str) else (payload_raw or {})

        ob = payload.get("orderbook_data") or {}
        if not isinstance(ob, dict):
            continue

        offset_ms = ob.get("snapshot_offset_ms")
        if offset_ms is None:
            offset_ms = payload.get("snapshot_offset_ms")

        source = ob.get("source") or ob.get("source_type") or payload.get("source")
        if isinstance(source, dict):
            source = source.get("stream")

        if offset_ms is not None:
            records.append({
                "id": ev_id,
                "window_id": win_id,
                "event_type": ev_type,
                "timestamp_ms": ts_ms,
                "offset_ms": float(offset_ms),
                "source": str(source or "unknown"),
            })

    if not records:
        print("⚠️ Nenhum evento com snapshot_offset_ms encontrado na base.")
        return 1

    df = pd.DataFrame(records)
    total_events = len(df)
    print(f"Eventos analisados com bloco orderbook: {total_events}\n")

    # 1. Fontes de orderbook
    source_counts = df["source"].value_counts()
    print("── Distribuição de Origem do Snapshot ──")
    for src, cnt in source_counts.items():
        pct = (cnt / total_events) * 100.0
        print(f"  • {src}: {cnt} eventos ({pct:.2f}%)")

    live_count = df[df["source"] == "live_sync"].shape[0]
    pct_live = (live_count / total_events) * 100.0

    # 2. Distribuição Estatística de snapshot_offset_ms (Foco em live_sync para auditoria de SLA)
    live_df = df[df["source"] == "live_sync"]
    offsets = live_df["offset_ms"] if len(live_df) > 0 else df["offset_ms"]
    label_dist = "live_sync" if len(live_df) > 0 else "global"

    print(f"\n── Distribuição Estatística de snapshot_offset_ms ({label_dist}) (ms) ──")
    print(f"  • Contagem: {len(offsets)}")
    print(f"  • Média ± DesvPad: {offsets.mean():.1f} ± {offsets.std():.1f} ms")
    print(f"  • Mínimo: {offsets.min():.1f} ms")
    print(f"  • p10:    {offsets.quantile(0.10):.1f} ms")
    print(f"  • p25:    {offsets.quantile(0.25):.1f} ms")
    print(f"  • p50 (Mediana): {offsets.quantile(0.50):.1f} ms")
    print(f"  • p75:    {offsets.quantile(0.75):.1f} ms")
    print(f"  • p90:    {offsets.quantile(0.90):.1f} ms")
    print(f"  • p95:    {offsets.quantile(0.95):.1f} ms")
    print(f"  • p99:    {offsets.quantile(0.99):.1f} ms")
    print(f"  • Máximo: {offsets.max():.1f} ms")

    # 3. Teste de Conformidade de Timeout (Leaked Latency Check)
    leaked_live = df[(df["source"] == "live_sync") & (df["offset_ms"] > max_timeout_ms)]
    print("\n── Verificação de Integridade do Timeout (1500 ms) ──")
    if len(leaked_live) == 0:
        print(f"  ✅ CONFORME: 0 eventos 'live_sync' excederam o timeout limite de {max_timeout_ms:.0f}ms.")
    else:
        print(f"  ❌ VAZAMENTO DETECTADO: {len(leaked_live)} eventos 'live_sync' têm offset > {max_timeout_ms:.0f}ms!")
        print("  Amostras fora de conformidade:")
        print(leaked_live[["window_id", "event_type", "offset_ms", "source"]].head(10).to_string(index=False))

    # 4. Avaliação do Fallback (cache_bg)
    bg_events = df[df["source"] == "cache_bg"]
    if len(bg_events) > 0:
        print(f"\n  ℹ️ Eventos em fallback (cache_bg): {len(bg_events)} ({len(bg_events)/total_events*100:.1f}%)")
        print(f"     Idade do cache (offset positivo): p50={bg_events['offset_ms'].median():.1f}ms, p90={bg_events['offset_ms'].quantile(0.90):.1f}ms, min={bg_events['offset_ms'].min():.1f}ms, max={bg_events['offset_ms'].max():.1f}ms")

    print("\n" + "=" * 80)
    live_p90 = offsets.quantile(0.90) if len(offsets) > 0 else 0.0
    verdict = "APROVADO" if (pct_live >= 80.0 and len(leaked_live) == 0 and live_p90 <= max_timeout_ms) else "REPROVADO / REQUER INVESTIGAÇÃO"
    print(f"Veredito de Validação Estatística do Snapshot: {verdict}")
    print("=" * 80 + "\n")
    return 0 if verdict.startswith("APROVADO") else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Análise estatística de sincronização do orderbook.")
    parser.add_argument("pos_db", nargs="?", default=None, help="Caminho posicional para o banco de dados")
    parser.add_argument("--db", default=None, help="Caminho do banco SQLite (ex: dados/trading_bot.db)")
    parser.add_argument("--session", default=None, help="Filtro de período/sessão (ex: 2026-09-08)")
    parser.add_argument("--timeout", type=float, default=1500.0, help="Timeout limite em ms (default 1500)")

    args = parser.parse_args()
    db_file = args.db or args.pos_db or "dados/trading_bot.db"

    sys.exit(analyze_session_orderbook_sync(
        db_path=db_file,
        session_filter=args.session,
        max_timeout_ms=args.timeout,
    ))
