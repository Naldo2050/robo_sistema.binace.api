# -*- coding: utf-8 -*-
"""
T3 — Quantificar o erro da Value Area: sorted-desc (código atual) vs POC-outward (teórico).
Apenas análise numérica — não altera código de produção.
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from support_resistance.volume_profile import VolumeProfileAnalyzer


def sorted_desc_va(bins_vol, total, pct=0.70):
    """Replica o método do código: ordena bins por volume desc, acumula até pct, range = min/max centers."""
    target = total * pct
    order = sorted(range(len(bins_vol)), key=lambda i: -bins_vol[i])
    chosen = []
    acc = 0.0
    for i in order:
        chosen.append(i)
        acc += bins_vol[i]
        if acc >= target:
            break
    return min(chosen), max(chosen), acc / total * 100


def poc_outward_va(centers, bins_vol, total, pct=0.70):
    """Método teórico: expande do bin POC para os vizinhos mais volumosos, range contíguo."""
    target = total * pct
    poc_idx = int(np.argmax(bins_vol))
    chosen = [poc_idx]
    acc = bins_vol[poc_idx]
    lo, hi = poc_idx, poc_idx
    while acc < target and (lo > 0 or hi < len(bins_vol) - 1):
        left = bins_vol[lo - 1] if lo > 0 else -1
        right = bins_vol[hi + 1] if hi < len(bins_vol) - 1 else -1
        if right >= left:
            hi += 1
            chosen.append(hi)
            acc += bins_vol[hi]
        else:
            lo -= 1
            chosen.append(lo)
            acc += bins_vol[lo]
    return lo, hi, acc / total * 100


def show_case(name, centers, bins_vol):
    total = sum(bins_vol)
    sd_lo, sd_hi, sd_pct = sorted_desc_va(bins_vol, total)
    po_lo, po_hi, po_pct = poc_outward_va(centers, bins_vol, total)
    va_low_code = centers[sd_lo]
    va_high_code = centers[sd_hi]
    va_low_theo = centers[po_lo]
    va_high_theo = centers[po_hi]
    print(f"\n=== {name} ===")
    print(f"bins: {[(centers[i], bins_vol[i]) for i in range(len(centers))]}  total={total}")
    print(f"  CODIGO  (sorted-desc): VAL={va_low_code} VAH={va_high_code} acumulado={sd_pct:.1f}%  bins={list(range(sd_lo, sd_hi+1))}")
    print(f"  TEORICO (POC-outward): VAL={va_low_theo} VAH={va_high_theo} acumulado={po_pct:.1f}%  bins={list(range(po_lo, po_hi+1))}")
    print(f"  DIFERENCA VAH: {va_high_code - va_high_theo:.4f}  ({abs(va_high_code - va_high_theo)/centers[poc_outward_va(centers, bins_vol, total)[1]]*100:.2f}% do preco)")


# ── Caso 1: teste da auditoria anterior (3 price levels, 30 pontos) ──
print("=" * 70)
print("CASO 1 — teste commitado (audit_support_resistance_test.py):")
prices = [49900.0] * 10 + [49905.0] * 10 + [49910.0] * 10
vols = [5.0] * 10 + [2.0] * 10 + [8.0] * 10
vpa = VolumeProfileAnalyzer(pd.Series(prices), pd.Series(vols))
prof = vpa.calculate_profile()
code_val = prof["value_area"]["low"]
code_vah = prof["value_area"]["high"]
# POC-outward manual: POC=49909.9 (80); vizinhos abaixo: 49900.1 (50) > 49905.1 (20)
# -> pega 49900.1 -> cum 130/150 = 86.7% >= 105 -> VA = [49900.1, 49909.9]
print(f"  CODIGO:  VAL={code_val} VAH={code_vah}")
print(f"  TEORICO: VAL=49900.1 VAH=49909.9  -> DIFERENCA: 0.0 (metodos coincidem quando POC e no extremo)")

# ── Caso 2: POC central + picos distantes (pior caso) ──
centers2 = np.array([490.0, 495.0, 500.0, 505.0, 510.0])
bins2 = np.array([50.0, 20.0, 60.0, 15.0, 55.0])
show_case("CASO 2 — POC central, picos distantes (POC=500, vol 60)", centers2, bins2)

# ── Caso 3: mesmo com o VolumeProfileAnalyzer real (50 bins, 30 pontos) ──
np.random.seed(42)
p3 = np.concatenate([
    np.random.normal(500.0, 1.0, 40) * 1.0,
    np.random.normal(506.0, 1.0, 25),
])
v3 = np.random.uniform(5, 30, len(p3))
vpa3 = VolumeProfileAnalyzer(pd.Series(p3), pd.Series(v3))
prof3 = vpa3.calculate_profile()
bins_p = prof3.get("bins") or prof3.get("profile_bins") or []
print(f"\n=== CASO 3 — dados sinteticos bimodais via VolumeProfileAnalyzer ===")
print(f"  CODIGO:  VAL={prof3['value_area']['low']:.4f} VAH={prof3['value_area']['high']:.4f}")
print(f"  volume_area_volume_pct: {vpa3.calculate_value_area_volume_pct(prof3).get('value_area_volume_pct')}%")

# ── Erro de classificacao in_value_area ──
print(f"\n=== IMPACTO EM 'in_value_area' (ai_payload_builder.py:566) ===")
print(f"  No CASO 2, se preco=507.0 (dentro do VAH bugado 510, fora do correto 505):")
print(f"    com bug (sorted-desc): in_value_area=True  | correto (POC-outward): in_value_area=False")
print(f"  -> classificacao de zona divergente com o metodo teorico")
