#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/diagnostics/calc_r4b_stats.py
Calcula as métricas completas da R4b para todas as 75 janelas.
"""

import os
import json
import math
import pandas as pd
import numpy as np

# 1. Carregar klines
with open("dados/audit/klines_spot_s1.json", "r") as f:
    spot_klines = json.load(f)
with open("dados/audit/klines_fut_s1.json", "r") as f:
    fut_klines = json.load(f)

spot_by_ot = {int(k[0]): {"close": float(k[4]), "vol": float(k[5])} for k in spot_klines}
fut_by_ot = {int(k[0]): {"close": float(k[4]), "vol": float(k[5])} for k in fut_klines}

# 2. Carregar windows_flat.csv
df = pd.read_csv("dados/audit/windows_flat.csv")
s1 = df[df["meta_session"] == 1].copy()
s1_valid = (
    s1.dropna(subset=["meta_janela_numero"])
    .drop_duplicates(subset=["meta_janela_numero"])
    .sort_values("meta_janela_numero")
    .reset_index(drop=True)
)

rows = []
for idx, r in s1_valid.iterrows():
    wkey = str(r["meta_window_key"])
    ep = int(r["meta_epoch_ms"])
    ep_utc = str(r["meta_timestamp_utc"])
    v_bot = float(r["raw_volume_total"])
    c_bot = float(r["raw_preco_fechamento"])
    ob_mid = float(r["ob_mid"]) if pd.notna(r["ob_mid"]) else np.nan

    # Alinhamento conforme passo 3:
    # open_time == floor(meta_epoch_ms/60000)*60000 - 60000
    target_ot = (ep // 60000) * 60000 - 60000

    spot_k = spot_by_ot.get(target_ot)
    fut_k = fut_by_ot.get(target_ot)

    if spot_k is None or fut_k is None:
        # Busca sobreposição temporal máxima
        w_start, w_end = ep - 60000, ep
        best_ot, best_ov = None, -1
        for ot in spot_by_ot:
            ov = max(0, min(w_end, ot + 60000) - max(w_start, ot))
            if ov > best_ov:
                best_ov, best_ot = ov, ot
        target_ot = best_ot
        spot_k = spot_by_ot[target_ot]
        fut_k = fut_by_ot[target_ot]

    v_spot = spot_k["vol"]
    c_spot = spot_k["close"]
    v_fut = fut_k["vol"]
    c_fut = fut_k["close"]

    r_spot = v_bot / v_spot if v_spot > 0 else np.nan
    r_fut = v_bot / v_fut if v_fut > 0 else np.nan

    # Basis por janela = (ob_mid - close_fut) / close_fut * 1e4 (bps)
    basis_bps = ((ob_mid - c_fut) / c_fut * 1e4) if pd.notna(ob_mid) else np.nan

    # Diferença close_bot vs close_spot em bps: (close_bot - close_spot) / close_spot * 1e4
    diff_spot_bps = ((c_bot - c_spot) / c_spot * 1e4)

    rows.append({
        "window_key": wkey,
        "epoch_utc": ep_utc,
        "volume_total": round(v_bot, 5),
        "vol_spot_1m": round(v_spot, 5),
        "vol_fut_1m": round(v_fut, 3),
        "razao_spot": round(r_spot, 6),
        "razao_fut": round(r_fut, 6),
        "close_bot": round(c_bot, 2),
        "close_spot": round(c_spot, 2),
        "close_fut": round(c_fut, 2),
        "ob_mid": round(ob_mid, 2) if pd.notna(ob_mid) else np.nan,
        "basis_fut_bps": round(basis_bps, 4) if pd.notna(basis_bps) else np.nan,
        "diff_spot_bps": round(diff_spot_bps, 4),
    })

res_df = pd.DataFrame(rows)

print("=== AMOSTRA DAS PRIMEIRAS 5 JANELAS ===")
print(res_df.head(5).to_string(index=False))

print("\n=== AMOSTRA DAS JANELAS 1:20 A 1:25 ===")
print(res_df.iloc[19:25].to_string(index=False))

# Estatísticas Passo 5
r_spot_s = res_df["razao_spot"].dropna()
r_fut_s = res_df["razao_fut"].dropna()
basis_s = res_df["basis_fut_bps"].dropna()
diff_spot_s = res_df["diff_spot_bps"].dropna()

print("\n=== ESTATÍSTICAS PASSO 5 ===")
print(f"Razão SPOT: Mediana={r_spot_s.median():.6f}, P10={r_spot_s.quantile(0.10):.6f}, P90={r_spot_s.quantile(0.90):.6f}")
print(f"Razão FUT:  Mediana={r_fut_s.median():.6f}, P10={r_fut_s.quantile(0.10):.6f}, P90={r_fut_s.quantile(0.90):.6f}")

print(f"\nBasis Futures (ob_mid - close_fut)/close_fut * 1e4:")
print(f"  Mediana (p50):   {basis_s.median():.4f} bps")
print(f"  P10:             {basis_s.quantile(0.10):.4f} bps")
print(f"  P90:             {basis_s.quantile(0.90):.4f} bps")
print(f"  |Basis| p50:     {basis_s.abs().median():.4f} bps")

print(f"\nDiferença Spot (close_bot - close_spot)/close_spot * 1e4:")
print(f"  |Diff| p50:      {diff_spot_s.abs().median():.4f} bps")
print(f"  |Diff| p90:      {diff_spot_s.abs().quantile(0.90):.4f} bps")
print(f"  Diff p50 (com sinal): {diff_spot_s.median():.4f} bps")
print(f"  Diff p90 (com sinal): {diff_spot_s.quantile(0.90):.4f} bps")

res_df.to_csv("dados/audit/r4b_tabela_75_janelas.csv", index=False)
print("\nSalvo em dados/audit/r4b_tabela_75_janelas.csv")
