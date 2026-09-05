#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/diagnostics/crosscheck_klines_session.py

Etapa B da Rodada 4:
Busca klines públicas de 1m da Binance (Spot e Futuros) para o intervalo da Sessão 1
(1788305340000 a 1788309840000) e confronta com os dados extraídos das 75 janelas.
Salva respostas cruas em dados/audit/klines_spot_s1.json e dados/audit/klines_fut_s1.json.
"""

import os
import json
import urllib.request
import pandas as pd
import numpy as np


def fetch_klines(url: str, outfile: str):
    if os.path.exists(outfile):
        print(f"[CACHE] Carregando {outfile} existente...")
        with open(outfile, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"[REDE] Buscando {url}...")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[OK] Salvo em {outfile} ({len(data)} candles)")
    return data


def main():
    start_time = 1788305340000  # 2026-09-01T23:29:00Z
    end_time = 1788309840000    # 2026-09-02T00:44:00Z
    symbol = "BTCUSDT"

    url_spot = (
        f"https://api.binance.com/api/v3/klines?"  # SPOT intencional porque compara spot vs fut na auditoria retroativa
        f"symbol={symbol}&interval=1m&startTime={start_time}&endTime={end_time}&limit=100"
    )
    url_fut = (
        f"https://fapi.binance.com/fapi/v1/klines?"
        f"symbol={symbol}&interval=1m&startTime={start_time}&endTime={end_time}&limit=100"
    )

    spot_file = "dados/audit/klines_spot_s1.json"
    fut_file = "dados/audit/klines_fut_s1.json"

    spot_klines = fetch_klines(url_spot, spot_file)
    fut_klines = fetch_klines(url_fut, fut_file)

    # Converter para dicionários indexados por open_time (int)
    # Formato kline Binance: [open_time, open, high, low, close, volume, close_time, quote_vol, trades, taker_base, taker_quote, ignore]
    spot_dict = {
        k[0]: {
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "vol": float(k[5]),
            "trades": int(k[8]),
        }
        for k in spot_dict_raw
    } if (spot_dict_raw := spot_klines) else {}

    fut_dict = {
        k[0]: {
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "vol": float(k[5]),
            "trades": int(k[8]),
        }
        for k in fut_dict_raw
    } if (fut_dict_raw := fut_klines) else {}

    # Carregar windows_flat.csv
    csv_file = "dados/audit/windows_flat.csv"
    df = pd.read_csv(csv_file)
    s1 = df[df["meta_session"] == 1].dropna(subset=["meta_janela_numero"]).drop_duplicates(subset=["meta_janela_numero"]).sort_values("meta_janela_numero").reset_index(drop=True)

    records = []
    for _, row in s1.iterrows():
        j_num = int(row["meta_janela_numero"])
        epoch = int(row["meta_epoch_ms"])
        vol_bot = row["raw_volume_total"]
        close_bot = row["raw_preco_fechamento"]
        mid_bot = row["ob_mid"]
        lat_ms = row["ob_latency_ms"]

        # Alinhar com candle spot e fut
        spot_k = spot_dict.get(epoch)
        fut_k = fut_dict.get(epoch)

        vol_spot = spot_k["vol"] if spot_k else np.nan
        close_spot = spot_k["close"] if spot_k else np.nan

        vol_fut = fut_k["vol"] if fut_k else np.nan
        close_fut = fut_k["close"] if fut_k else np.nan

        razao_spot = vol_bot / vol_spot if (vol_spot and vol_spot > 0) else np.nan
        razao_fut = vol_bot / vol_fut if (vol_fut and vol_fut > 0) else np.nan

        basis_bps = ((mid_bot - close_spot) / close_spot * 1e4) if (close_spot and pd.notna(mid_bot)) else np.nan

        records.append({
            "janela": j_num,
            "epoch_ms": epoch,
            "volume_total": vol_bot,
            "vol_spot_1m": vol_spot,
            "vol_fut_1m": vol_fut,
            "razao_spot": razao_spot,
            "razao_fut": razao_fut,
            "close_janela": close_bot,
            "close_spot": close_spot,
            "close_fut": close_fut,
            "mid_book": mid_bot,
            "latency_ms": lat_ms,
            "basis_bps": basis_bps,
        })

    res_df = pd.DataFrame(records)
    print("\n=== AMOSTRA ALINHAMENTO POR MINUTO (10 JANELAS) ===")
    sample_cols = [
        "janela", "volume_total", "vol_spot_1m", "vol_fut_1m", 
        "razao_spot", "razao_fut", "close_janela", "close_spot", "mid_book"
    ]
    sample_indices = [0, 9, 19, 20, 23, 29, 39, 49, 59, 74]
    valid_samples = [i for i in sample_indices if i < len(res_df)]
    print(res_df.iloc[valid_samples][sample_cols].to_string(index=False))

    # B3: Estatísticas de razão de volume
    v_spot = res_df["razao_spot"].dropna()
    v_fut = res_df["razao_fut"].dropna()

    print("\n=== B3: RAZÃO DE VOLUME (BOT / CANDLE EXTERNO) ===")
    print(f"Razão SPOT  -> Mediana: {v_spot.median():.4f}, P10: {v_spot.quantile(0.10):.4f}, P90: {v_spot.quantile(0.90):.4f}, Média: {v_spot.mean():.4f}")
    print(f"Razão FUT   -> Mediana: {v_fut.median():.4f}, P10: {v_fut.quantile(0.10):.4f}, P90: {v_fut.quantile(0.90):.4f}, Média: {v_fut.mean():.4f}")

    # B4: Basis e controle de resíduo
    basis = res_df["basis_bps"].dropna()
    mean_basis = basis.mean()
    std_basis = basis.std()
    same_sign_pct = (np.sign(basis) == np.sign(mean_basis)).mean() * 100

    print("\n=== B4: BASIS SPOT x FUTURES ===")
    print(f"Basis (bps) -> Média: {mean_basis:.4f} bps, Desvio: {std_basis:.4f} bps, % Mesmo sinal: {same_sign_pct:.1f}%")

    # Correlação close - mid com latency
    valid_b4 = res_df.dropna(subset=["close_janela", "mid_book", "latency_ms"]).copy()
    diff_abs = (valid_b4["close_janela"] - valid_b4["mid_book"]).abs()
    corr_raw = diff_abs.corr(valid_b4["latency_ms"])

    # Controlando pelo basis médio
    # diff = close - mid; basis = mid - close_spot
    # resíduo: |(close_janela - mid_book) - (-mean_basis_usd)|
    mean_basis_usd = (valid_b4["mid_book"] - valid_b4["close_spot"]).mean()
    resid = (valid_b4["close_janela"] - valid_b4["mid_book"]) - (-mean_basis_usd)
    corr_controlled = resid.abs().corr(valid_b4["latency_ms"])

    print(f"Divergência Close x Mid USD média: {diff_abs.mean():.2f} USD")
    print(f"Basis médio USD: {mean_basis_usd:.2f} USD")
    print(f"Correlação bruta |close - mid| vs latency_ms: {corr_raw:.4f}")
    print(f"Correlação controlada por basis |resid| vs latency_ms: {corr_controlled:.4f}")


if __name__ == "__main__":
    main()
