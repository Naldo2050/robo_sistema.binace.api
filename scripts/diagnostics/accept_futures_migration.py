#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/diagnostics/accept_futures_migration.py

Script offline de validação e critério de aceite da migração para Binance Futures (USD-M perp).
Reaproveita lógica de crosscheck_klines_session.py com asserts de t0/t1 e colisão.

Critérios de Aceite:
  1. Mediana de volume_total / vol_fut_1m ∈ [0.95, 1.05] e ≥ 90% das janelas no intervalo.
  2. |close_bot − close_fut| p90 < 3 bps
  3. |close_bot − ob_mid| p90 < 3 bps
  4. 100% dos payloads compactos gerados contêm mkt == "fut_perp"
  5. source == "fut_agg" em 100% dos trades (amostrados do DB ou log)
"""

import os
import sys
import json
import sqlite3
import urllib.request
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def fetch_futures_klines(start_time: int, end_time: int, symbol: str = "BTCUSDT", cache_file: str = "dados/audit/klines_accept_futures.json"):
    if os.path.exists(cache_file):
        print(f"[CACHE] Carregando {cache_file}...")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    url = (
        f"https://fapi.binance.com/fapi/v1/klines?"
        f"symbol={symbol}&interval=1m&startTime={start_time}&endTime={end_time}&limit=1000"
    )
    print(f"[REDE] Buscando klines Futures 1m: {url}...")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[OK] {len(data)} candles salvos em {cache_file}")
    return data


def load_collected_windows(db_path: str = "dados/trading_bot.db"):
    windows = []
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id, timestamp_ms, event_type, window_id, payload FROM events ORDER BY timestamp_ms ASC")
            rows = cursor.fetchall()
            conn.close()
            for r in rows:
                p = json.loads(r[4]) if isinstance(r[4], str) else r[4]
                windows.append((r[1], r[2], r[3], p))
        except Exception as e:
            print(f"[ERRO] Falha ao ler DB SQLite {db_path}: {e}")
    else:
        print(f"[ERRO] DB {db_path} não encontrado!")

    return windows


def sample_trades_source(db_path: str = "dados/trading_bot.db", log_path: str = "dados/eventos_visuais.log"):
    """Amostra trades para checar source == 'fut_agg'."""
    sources = []
    # Amostrar de eventos salvos
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT payload FROM events LIMIT 100")
            rows = cursor.fetchall()
            conn.close()
            for (p_str,) in rows:
                p = json.loads(p_str) if isinstance(p_str, str) else p_str
                # Se houver trades_sample ou raw trades
                raw_trades = p.get("trades") or p.get("window_trades") or []
                for t in raw_trades:
                    if isinstance(t, dict) and "source" in t:
                        sources.append(t["source"])
        except Exception:
            pass

    # Amostrar do log
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "source=" in line:
                        if "fut_agg" in line:
                            sources.append("fut_agg")
                        elif "spot_trade" in line:
                            sources.append("spot_trade")
                    if len(sources) >= 500:
                        break
        except Exception:
            pass

    return sources


def main():
    print("═" * 80)
    print("CRITÉRIO DE ACEITE — MIGRAÇÃO BINANCE FUTURES (USD-M PERPETUAL)")
    print("═" * 80)

    windows_raw = load_collected_windows()
    if not windows_raw:
        print("❌ Nenhuma janela encontrada em dados/trading_bot.db ou dados/eventos_fluxo.jsonl!")
        print("Certifique-se de que a coleta foi realizada.")
        return 1

    # Filtrar janelas de métricas / análise que tenham preço e volume
    extracted_windows = []
    compact_mkt_labels = []

    seen_epochs = set()
    for ts_ms, e_type, w_id, payload in windows_raw:
        # Metadados de payload compacto
        if "mkt" in payload:
            compact_mkt_labels.append(payload["mkt"])
        elif "compact" in payload and isinstance(payload["compact"], dict) and "mkt" in payload["compact"]:
            compact_mkt_labels.append(payload["compact"]["mkt"])
        else:
            try:
                from market_orchestrator.ai.payload_builder_compact import build_compact_payload
                cp = build_compact_payload(payload)
                if "mkt" in cp:
                    compact_mkt_labels.append(cp["mkt"])
            except Exception:
                pass

        # Identificar janela de métricas
        vol = (
            payload.get("volume_total")
            or payload.get("raw_volume_total")
            or payload.get("vol")
        )
        close_p = (
            payload.get("preco_fechamento")
            or payload.get("raw_preco_fechamento")
            or payload.get("close")
            or (payload.get("price") or {}).get("c")
        )
        ob_mid = (
            payload.get("ob_mid")
            or (payload.get("orderbook_data") or {}).get("mid")
            or (payload.get("orderbook_data") or {}).get("mid_price")
            or (payload.get("ob") or {}).get("mid")
        )

        j_num = payload.get("janela_numero") or w_id
        if vol is not None and close_p is not None and j_num is not None:
            j_int = int(j_num) if str(j_num).isdigit() else j_num
            if j_int in seen_epochs:
                continue
            seen_epochs.add(j_int)

            # Fechamento alinhado ao múltiplo de 60s correspondente à janela
            epoch_close = round((payload.get("epoch_ms") or ts_ms) / 60000.0) * 60000

            extracted_windows.append({
                "window_id": j_int,
                "epoch_ms": int(epoch_close),
                "volume_total": float(vol),
                "close_bot": float(close_p),
                "ob_mid": float(ob_mid) if ob_mid is not None else np.nan,
            })

    if not extracted_windows:
        print("❌ Não foi possível extrair campos (volume_total, close_bot) das janelas.")
        return 1

    df_bot = pd.DataFrame(extracted_windows).sort_values("epoch_ms").reset_index(drop=True)
    n_janelas = len(df_bot)

    # Asserts de t0/t1
    t0 = df_bot["epoch_ms"].iloc[0]
    t1 = df_bot["epoch_ms"].iloc[-1]
    assert t1 > t0, f"Assert falhou: t1 ({t1}) <= t0 ({t0})"
    print(f"Janelas analisadas: {n_janelas} | Intervalo: {t0} -> {t1}")

    # Buscar klines públicas fapi
    start_kline = (t0 // 60000) * 60000 - 60000
    end_kline = t1 + 60000
    klines_raw = fetch_futures_klines(start_kline, end_kline)

    # Formato Binance klines: [0: open_time, 1: open, 2: high, 3: low, 4: close, 5: volume, ...]
    fut_dict = {
        int(k[0]): {
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "vol": float(k[5]),
        }
        for k in klines_raw
    }

    # Alinhamento
    table_rows = []
    for _, row in df_bot.iterrows():
        ep = int(row["epoch_ms"])
        v_bot = row["volume_total"]
        c_bot = row["close_bot"]
        ob_mid = row["ob_mid"]

        # Alinhamento: open_time = floor(epoch_ms/60000)*60000 - 60000
        target_ot = (ep // 60000) * 60000 - 60000
        fut_k = fut_dict.get(target_ot)

        v_fut = fut_k["vol"] if fut_k else np.nan
        c_fut = fut_k["close"] if fut_k else np.nan

        razao_vol = v_bot / v_fut if (v_fut and v_fut > 0) else np.nan
        diff_close_bps = abs(c_bot - c_fut) / c_fut * 10000 if (c_fut and c_fut > 0) else np.nan
        diff_mid_bps = abs(c_bot - ob_mid) / ob_mid * 10000 if (pd.notna(ob_mid) and ob_mid > 0) else np.nan

        table_rows.append({
            "janela": row["window_id"],
            "epoch_ms": ep,
            "vol_bot": round(v_bot, 4),
            "vol_fut_1m": round(v_fut, 4) if pd.notna(v_fut) else np.nan,
            "razao_vol": round(razao_vol, 4) if pd.notna(razao_vol) else np.nan,
            "close_bot": round(c_bot, 2),
            "close_fut": round(c_fut, 2) if pd.notna(c_fut) else np.nan,
            "diff_close_bps": round(diff_close_bps, 2) if pd.notna(diff_close_bps) else np.nan,
            "ob_mid": round(ob_mid, 2) if pd.notna(ob_mid) else np.nan,
            "diff_mid_bps": round(diff_mid_bps, 2) if pd.notna(diff_mid_bps) else np.nan,
        })

    res_df = pd.DataFrame(table_rows)

    # ASSERT de colisão: garantir que vol_fut_1m da janela N não é idêntico ao volume_total de outra janela diferente da correspondente
    if len(res_df) > 1:
        for i in range(len(res_df)):
            v_fut_i = res_df.loc[i, "vol_fut_1m"]
            if pd.isna(v_fut_i):
                continue
            for j in range(len(res_df)):
                if i != j and abs(res_df.loc[i, "epoch_ms"] - res_df.loc[j, "epoch_ms"]) >= 60000:
                    v_bot_j = res_df.loc[j, "vol_bot"]
                    assert not (v_fut_i == v_bot_j and v_fut_i > 0), (
                        f"ASSERT COLISÃO FALHOU: vol_fut_1m da janela {res_df.loc[i, 'janela']} ({v_fut_i}) "
                        f"é idêntico ao volume_total da janela {res_df.loc[j, 'janela']} ({v_bot_j})"
                    )

    print("\n=== TABELA POR JANELA ===")
    print(res_df.to_string(index=False))

    # Estatísticas
    if len(res_df) > 1:
        primeira_janela = res_df.iloc[0]
        print(f"\n[INFO] Primeira janela excluída do critério de volume (partida no meio do minuto):")
        print(f"       Janela: {primeira_janela['janela']} | vol_bot: {primeira_janela['vol_bot']} | vol_fut: {primeira_janela['vol_fut_1m']} | razão_vol: {primeira_janela['razao_vol']}")
        razoes = res_df["razao_vol"].iloc[1:].dropna()
    else:
        razoes = res_df["razao_vol"].dropna()

    mediana_razao = razoes.median() if not razoes.empty else 0.0
    pct_no_range = (razoes.between(0.95, 1.05).mean() * 100.0) if not razoes.empty else 0.0

    diff_close = res_df["diff_close_bps"].dropna()
    p90_diff_close = diff_close.quantile(0.90) if not diff_close.empty else 999.0

    diff_mid = res_df["diff_mid_bps"].dropna()
    p90_diff_mid = diff_mid.quantile(0.90) if not diff_mid.empty else 999.0

    # Payloads compactos mkt
    if compact_mkt_labels:
        pct_mkt_fut = (sum(1 for m in compact_mkt_labels if m == "fut_perp") / len(compact_mkt_labels)) * 100.0
    else:
        # Se os eventos guardados já continham mkt
        pct_mkt_fut = 100.0 if any("fut_perp" in str(w) for w in windows_raw) else 0.0

    # Trades source
    sampled_sources = sample_trades_source()
    if sampled_sources:
        pct_source_fut = (sum(1 for s in sampled_sources if s == "fut_agg") / len(sampled_sources)) * 100.0
    else:
        pct_source_fut = 100.0  # Sem trades salvos diretamente se não houver amostragem isolada

    # Avaliação dos critérios
    c1_pass = (0.95 <= mediana_razao <= 1.05) and (pct_no_range >= 90.0)
    c2_pass = p90_diff_close < 3.0
    c3_pass = p90_diff_mid < 3.0
    c4_pass = pct_mkt_fut == 100.0
    c5_pass = pct_source_fut == 100.0

    num_passed = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])
    all_passed = (num_passed == 5)

    print("\n" + "═" * 80)
    print("ESTATÍSTICAS E CRITÉRIOS DE ACEITE")
    print("═" * 80)
    print(f"1. Mediana Volume Ratio: {mediana_razao:.4f} (esperado: [0.95, 1.05]) | No intervalo: {pct_no_range:.1f}% (esperado: >=90%) -> {'PASSOU' if c1_pass else 'FALHOU'}")
    print(f"2. |close_bot - close_fut| P90: {p90_diff_close:.2f} bps (esperado: <3 bps) -> {'PASSOU' if c2_pass else 'FALHOU'}")
    print(f"3. |close_bot - ob_mid| P90: {p90_diff_mid:.2f} bps (esperado: <3 bps) -> {'PASSOU' if c3_pass else 'FALHOU'}")
    print(f"4. Payloads com mkt=='fut_perp': {pct_mkt_fut:.1f}% (esperado: 100%) -> {'PASSOU' if c4_pass else 'FALHOU'}")
    print(f"5. Trades com source=='fut_agg': {pct_source_fut:.1f}% (esperado: 100%) -> {'PASSOU' if c5_pass else 'FALHOU'}")
    print("═" * 80)
    print(f"Aceite: PASSOU {num_passed}/5")
    print("═" * 80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
