#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/diagnostics/run_r4b_full.py
Executa os passos 1 a 5 da correção R4b com todas as travas e asserts.
"""

import os
import json
import math
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# 1. Carregar windows_flat.csv
csv_path = "dados/audit/windows_flat.csv"
df = pd.read_csv(csv_path)

# Filtrar sessão 1
s1 = df[df["meta_session"] == 1].copy()

# Deduplicar as 75 janelas por meta_janela_numero (1.0 a 75.0)
s1_valid = (
    s1.dropna(subset=["meta_janela_numero"])
    .drop_duplicates(subset=["meta_janela_numero"])
    .sort_values("meta_janela_numero")
    .reset_index(drop=True)
)

print(f"Total janelas na Sessão 1 deduplicadas: {len(s1_valid)}")
assert len(s1_valid) == 75, f"Esperava 75 janelas, encontrou {len(s1_valid)}"

# Obter t0 e t1 de todas as janelas da sessão 1 (incluindo janela preliminar se houver)
epochs_all_s1 = s1["meta_epoch_ms"].astype("int64")
t0 = int(epochs_all_s1.min())
t1 = int(epochs_all_s1.max())

dt0 = datetime.fromtimestamp(t0 / 1000, tz=timezone.utc)
dt1 = datetime.fromtimestamp(t1 / 1000, tz=timezone.utc)

print(f"\n--- PASSO 1: TIMESTAMP ASSERT ---")
print(f"t0: {t0} ({dt0.strftime('%Y-%m-%dT%H:%M:%SZ')})")
print(f"t1: {t1} ({dt1.strftime('%Y-%m-%dT%H:%M:%SZ')})")

t0_bound_low = int(datetime(2026, 9, 1, 23, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
t0_bound_high = int(datetime(2026, 9, 1, 23, 30, 0, tzinfo=timezone.utc).timestamp() * 1000)
assert t0_bound_low <= t0 <= t0_bound_high, (
    f"ASSERT 1 FALHOU: t0={t0} fora de [2026-09-01T23:00Z, 2026-09-01T23:30Z]"
)
print("[OK] ASSERT 1: t0 dentro do intervalo esperado.")

# 2. Carregar klines já baixadas e salvas em dados/audit/
spot_file = "dados/audit/klines_spot_s1.json"
fut_file = "dados/audit/klines_fut_s1.json"

with open(spot_file, "r", encoding="utf-8") as f:
    spot_klines = json.load(f)
with open(fut_file, "r", encoding="utf-8") as f:
    fut_klines = json.load(f)

first_spot = spot_klines[0]
first_fut = fut_klines[0]
assert abs(first_spot[0] - t0) <= 120_000, "ASSERT 2 FALHOU (SPOT)"
assert abs(first_fut[0] - t0) <= 120_000, "ASSERT 2 FALHOU (FUT)"
print("\n--- PASSO 2: KLINES ASSERT ---")
print(f"[OK] ASSERT 2: open_time do primeiro candle SPOT ({first_spot[0]}) e FUT ({first_fut[0]}) a <= 120s de t0.")

# Mapear klines por open_time (int)
# Formato kline: [open_time, open, high, low, close, volume, close_time, ...]
spot_by_ot = {int(k[0]): {"close": float(k[4]), "vol": float(k[5])} for k in spot_klines}
fut_by_ot = {int(k[0]): {"close": float(k[4]), "vol": float(k[5])} for k in fut_klines}

# Coletar todos os volumes_totais do bot para o ASSERT do Passo 4
all_bot_volumes = s1_valid["raw_volume_total"].astype(float).tolist()

# 3 & 4. Alinhamento e Construção da Tabela das 75 Janelas
print("\n--- PASSO 3 & 4: ALINHAMENTO E CONSTRUÇÃO DA TABELA ---")
table_rows = []
collision_count = 0

for idx, row in s1_valid.iterrows():
    wkey = str(row["meta_window_key"])
    ep = int(row["meta_epoch_ms"])
    ep_utc = str(row["meta_timestamp_utc"])
    vol_bot = float(row["raw_volume_total"])
    close_bot = float(row["raw_preco_fechamento"])
    ob_mid = float(row["ob_mid"]) if pd.notna(row["ob_mid"]) else np.nan

    # Critério de Alinhamento:
    # A janela fecha em ep (epoch_ms). O candle de 1m que acabou de fechar abre 60s antes do fechamento.
    # Se ep for minuto cheio (ex: 1788305340000, rem = 0):
    # candle_open_time = floor(ep / 60000) * 60000 - 60000.
    # Se ep não for minuto cheio (tem fração de segundos, duração de ~58.4s):
    # A janela cobre o intervalo [ep - duracao_ms, ep].
    # O candle de 1m com maior sobreposição temporal é aquele cujo intervalo [ot, ot + 60000]
    # mais se sobrepõe a [ep - duracao, ep]. Como a janela fecha em ep, o minuto de referência é floor(ep / 60000) * 60000 - 60000
    # ou floor(ep / 60000) * 60000 dependendo do segundo de fechamento.
    # Seguindo estritamente a fórmula solicitada:
    # candle cujo open_time == floor(meta_epoch_ms / 60000) * 60000 - 60000
    target_ot = math.floor(ep / 60000) * 60000 - 60000

    # Verificar se target_ot existe nas klines
    spot_k = spot_by_ot.get(target_ot)
    fut_k = fut_by_ot.get(target_ot)

    if spot_k is None or fut_k is None:
        # Se não encontrou no minuto exato - 60s, busca por sobreposição máxima
        # Janela [ep - 60000, ep]
        best_ot = None
        best_overlap = -1
        w_start = ep - 60000
        w_end = ep
        for ot in spot_by_ot.keys():
            c_start = ot
            c_end = ot + 60000
            overlap = max(0, min(w_end, c_end) - max(w_start, c_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_ot = ot
        target_ot = best_ot
        spot_k = spot_by_ot[target_ot]
        fut_k = fut_by_ot[target_ot]

    vol_spot = spot_k["vol"]
    close_spot = spot_k["close"]
    vol_fut = fut_k["vol"]
    close_fut = fut_k["close"]

    # ASSERT por linha: vol_spot_1m != volume_total de qualquer janela do bot (tolerância 1e-6)
    is_collision = any(abs(vol_spot - b_vol) < 1e-6 for b_vol in all_bot_volumes)
    if is_collision:
        collision_count += 1

    razao_spot = vol_bot / vol_spot if vol_spot > 0 else np.nan
    razao_fut = vol_bot / vol_fut if vol_fut > 0 else np.nan

    # Basis por janela = (ob_mid - close_fut) / close_fut * 1e4 (bps)
    basis_fut_bps = ((ob_mid - close_fut) / close_fut * 1e4) if pd.notna(ob_mid) else np.nan
    # Diff close_bot vs close_spot em bps: (close_bot - close_spot) / close_spot * 1e4
    diff_spot_bps = ((close_bot - close_spot) / close_spot * 1e4)

    table_rows.append({
        "window_key": wkey,
        "epoch_utc": ep_utc,
        "volume_total": vol_bot,
        "vol_spot_1m": vol_spot,
        "vol_fut_1m": vol_fut,
        "razao_spot": razao_spot,
        "razao_fut": razao_fut,
        "close_bot": close_bot,
        "close_spot": close_spot,
        "close_fut": close_fut,
        "ob_mid": ob_mid,
        "basis_fut_bps": basis_fut_bps,
        "diff_spot_bps": diff_spot_bps,
        "candle_ot": target_ot,
    })

print(f"Colisões de vol_spot_1m com volume_total do bot: {collision_count}")
assert collision_count <= 2, f"ASSERT 4 FALHOU: {collision_count} linhas coincidiram com volume do bot (> 2). Coluna contaminada!"
print("[OK] ASSERT 4: vol_spot_1m é independente dos volumes do bot (0 colisões ou <= 2).")

res_df = pd.DataFrame(table_rows)

# 5. Estatísticas
print("\n--- PASSO 5: ESTATÍSTICAS ---")
r_spot = res_df["razao_spot"].dropna()
r_fut = res_df["razao_fut"].dropna()

med_spot = r_spot.median()
p10_spot = r_spot.quantile(0.10)
p90_spot = r_spot.quantile(0.90)

med_fut = r_fut.median()
p10_fut = r_fut.quantile(0.10)
p90_fut = r_fut.quantile(0.90)

print(f"Razão SPOT  -> Mediana: {med_spot:.4f}, P10: {p10_spot:.4f}, P90: {p90_spot:.4f}")
print(f"Razão FUT   -> Mediana: {med_fut:.4f}, P10: {p10_fut:.4f}, P90: {p90_fut:.4f}")

basis_fut = res_df["basis_fut_bps"].dropna()
basis_p50 = basis_fut.median()
basis_p10 = basis_fut.quantile(0.10)
basis_p90 = basis_fut.quantile(0.90)
basis_abs_p50 = basis_fut.abs().median()

print(f"\nBasis Futures (ob_mid - close_fut)/close_fut * 1e4:")
print(f"  Mediana (p50): {basis_p50:.4f} bps")
print(f"  P10:           {basis_p10:.4f} bps")
print(f"  P90:           {basis_p90:.4f} bps")
print(f"  |Basis| p50:   {basis_abs_p50:.4f} bps")

diff_spot = res_df["diff_spot_bps"].dropna()
diff_spot_p50 = diff_spot.abs().median()
diff_spot_p90 = diff_spot.abs().quantile(0.90)

print(f"\nDiferença Spot (close_bot - close_spot)/close_spot * 1e4:")
print(f"  |Diff| Mediana (p50): {diff_spot_p50:.4f} bps")
print(f"  |Diff| P90:           {diff_spot_p90:.4f} bps")

# Salvar tabela gerada em csv para inclusão e auditoria
res_df.to_csv("dados/audit/r4b_aligned_windows.csv", index=False)
print("\nTabela completa salva em dados/audit/r4b_aligned_windows.csv")
