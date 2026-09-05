#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/diagnostics/audit_sr_recurrence.py

Módulo independente para análise de recorrência e persistência de:
  - Walls de Order Book (Top-3 Bid e Ask agrupadas com tolerância ±0.05%)
  - Defense Zones (Top-5 agrupadas por níveis de confluência)
Lê diretamente de dados/audit/windows_flat.csv.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

CSV_PATH = Path("dados/audit/windows_flat.csv")


def cluster_walls(walls_list: List[Dict[str, Any]], tol_pct: float = 0.0005) -> List[Dict[str, Any]]:
    """Agrupa ordens/walls por proximidade de preço (padrão ±0.05%)."""
    if not walls_list:
        return []
    sorted_walls = sorted(walls_list, key=lambda x: x["price"])
    clusters = []
    current_cluster = [sorted_walls[0]]
    for w in sorted_walls[1:]:
        center = np.mean([x["price"] for x in current_cluster])
        if abs(w["price"] - center) / center <= tol_pct:
            current_cluster.append(w)
        else:
            clusters.append(current_cluster)
            current_cluster = [w]
    if current_cluster:
        clusters.append(current_cluster)

    result = []
    for c in clusters:
        unique_windows = sorted(list(set(x["wkey"] for x in c)))
        avg_price = np.mean([x["price"] for x in c])
        max_qty = max(x["qty"] for x in c)
        avg_qty = np.mean([x["qty"] for x in c])
        side = c[0]["side"]
        first_w = c[0]["wkey"]
        last_w = c[-1]["wkey"]
        result.append({
            "avg_price": round(avg_price, 2),
            "n_windows": len(unique_windows),
            "side": side,
            "max_qty": round(max_qty, 3),
            "avg_qty": round(avg_qty, 3),
            "first_win": first_w,
            "last_win": last_w,
            "windows": unique_windows
        })
    return [r for r in result if r["n_windows"] >= 3]


def run_sr_recurrence_audit(csv_path: Path = CSV_PATH) -> Dict[str, Any]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo {csv_path} não encontrado!")

    df = pd.read_csv(csv_path)
    # Filtra janelas analíticas (ANALYSIS_TRIGGER e Exaustão) da sessão 1
    df_ana = df[df["meta_tipo_evento"].isin(["ANALYSIS_TRIGGER", "Exaustão"])].copy()

    # 1. Extração de Walls Top-3
    wall_bids = []
    wall_asks = []

    for idx, r in df_ana.iterrows():
        wkey = r["meta_window_key"]
        for i in range(3):
            p_b = r.get(f"ob_wall_bid_{i}_price")
            q_b = r.get(f"ob_wall_bid_{i}_qty")
            if pd.notna(p_b) and pd.notna(q_b) and q_b > 0:
                wall_bids.append({"price": float(p_b), "qty": float(q_b), "side": "bid", "wkey": wkey})

            p_a = r.get(f"ob_wall_ask_{i}_price")
            q_a = r.get(f"ob_wall_ask_{i}_qty")
            if pd.notna(p_a) and pd.notna(q_a) and q_a > 0:
                wall_asks.append({"price": float(p_a), "qty": float(q_a), "side": "ask", "wkey": wkey})

    bids_clusters = cluster_walls(wall_bids)
    asks_clusters = cluster_walls(wall_asks)
    top_walls = sorted(bids_clusters + asks_clusters, key=lambda x: x["n_windows"], reverse=True)

    # 2. Persistência de Defense Zones
    # Níveis de referência de J21 da R1/R2
    ref_zones = [77356.18, 77547.91, 77680.88, 78458.00, 78607.82]
    dz_persistence = {lvl: [] for lvl in ref_zones}

    for idx, r in df_ana.iterrows():
        wkey = r["meta_window_key"]
        for i in range(5):
            p_dz = r.get(f"sr_dz_{i}_price")
            s_dz = r.get(f"sr_dz_{i}_strength")
            side_dz = r.get(f"sr_dz_{i}_side")
            if pd.notna(p_dz):
                for ref_lvl in ref_zones:
                    if abs(p_dz - ref_lvl) / ref_lvl <= 0.001:  # 0.1% tol
                        dz_persistence[ref_lvl].append({
                            "wkey": wkey,
                            "price": p_dz,
                            "strength": s_dz,
                            "side": side_dz
                        })

    print("=" * 80)
    print("RELATÓRIO DE RECORRÊNCIA DE SUPORTE E RESISTÊNCIA (E1/E2)")
    print("=" * 80)
    print(f"\n1. WALLS RECORRENTES (>= 3 Janelas, tol ±0.05%) - Total clusters: {len(top_walls)}")
    print(f"{'Preço Médio':<12} | {'Lado':<5} | {'Janelas':<8} | {'Max Qty':<10} | {'Média Qty':<10} | {'Primeira / Última Janela'}")
    print("-" * 75)
    for w in top_walls[:12]:
        print(f"${w['avg_price']:<11.2f} | {w['side']:<5} | {w['n_windows']:<8} | {w['max_qty']:<10.3f} | {w['avg_qty']:<10.3f} | {w['first_win']} -> {w['last_win']}")

    print(f"\n2. PERSISTÊNCIA DAS DEFENSE ZONES (Referência J21)")
    print(f"{'Nível Ref.':<12} | {'Janelas':<8} | {'Strength Min-Max':<18} | {'Lados Observados'}")
    print("-" * 65)
    for lvl in ref_zones:
        occs = dz_persistence[lvl]
        unique_wins = set(x["wkey"] for x in occs)
        strengths = [x["strength"] for x in occs if pd.notna(x["strength"])]
        s_min = min(strengths) if strengths else 0
        s_max = max(strengths) if strengths else 0
        sides = set(x["side"] for x in occs)
        print(f"${lvl:<11.2f} | {len(unique_wins):<8} | {s_min:.0f} - {s_max:.0f} {'':<11} | {', '.join(sides)}")
    print("=" * 80)

    return {
        "top_walls": top_walls,
        "dz_persistence": dz_persistence
    }


if __name__ == "__main__":
    run_sr_recurrence_audit(CSV_PATH)
