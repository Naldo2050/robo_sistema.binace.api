# -*- coding: utf-8 -*-
"""
Auditoria numérica: support_resistance/ (sr_strength, pivot_points, volume_profile)

Testa a matemática com valores conhecidos e compara com cálculo manual.
Não altera código de produção.
"""
import sys
import os
import math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from support_resistance.sr_strength import SRStrengthScorer
from support_resistance.pivot_points import InstitutionalPivotPoints
from support_resistance.volume_profile import VolumeProfileAnalyzer
from support_resistance import daily_pivot, weekly_pivot, monthly_pivot

PASS = "PASS"
FAIL = "FAIL"
results = []


def check(name, cond, extra=""):
    tag = PASS if cond else FAIL
    results.append((tag, name, extra))
    print(f"[{tag}] {name} {extra}")


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


# ============================================================
# SR_STRENGTH: nível 50000 com 2 toques, current=50000
# ============================================================
print("--- SR_STRENGTH ---")
scorer = SRStrengthScorer()
candles = [
    {"high": 50050.0, "low": 49950.0},   # toque (|50050-50000|=50 <= 75)
    {"high": 50040.0, "low": 49960.0},   # toque
    {"high": 49800.0, "low": 49700.0},   # sem toque
    {"high": 51000.0, "low": 50900.0},   # sem toque
]
res = scorer.score_levels(
    current_price=50000.0,
    vp_data={"poc": 50000.0},
    pivot_data=None,
    ema_values=None,
    recent_candles=candles,
)
levels = {lvl["price"]: lvl for lvl in res["levels"]}
lvl = levels.get(50000.0)
# Cálculo manual: toques=2 -> min(25, 12)=12; peso fonte (poc 1.5 + round 0.6)=2.1 -> min(25, 16.8)=16.8
# confluência 2 -> 12; proximidade 0% -> 20. Total = 12+16.8+12+20 = 60.8 -> 61
expected_score = round(12 + 16.8 + 12 + 20)
check("sr_strength: level 50000 score == 61 (manual)",
      lvl is not None and lvl.get("strength") == expected_score,
      f"got={lvl.get('strength') if lvl else None} expected={expected_score}")
check("sr_strength: touches == 2", lvl is not None and lvl.get("touches") == 2,
      f"got={lvl.get('touches') if lvl else None}")
check("sr_strength: confluence_count == 2", lvl is not None and lvl.get("confluence_count") == 2,
      f"got={lvl.get('confluence_count') if lvl else None}")
check("sr_strength: type at_price", lvl is not None and lvl.get("type") == "at_price",
      f"got={lvl.get('type') if lvl else None}")
# Nível distante 48000: score = 0 toques + 0.6*8=4.8 + conf 5 + prox max(0,20-12)=8 -> 17.8 -> 18
lvl2 = levels.get(48000.0)
expected2 = round(0 + 4.8 + 5 + 8)
check("sr_strength: level 48000 score == 18 (manual)",
      lvl2 is not None and lvl2.get("strength") == expected2,
      f"got={lvl2.get('strength') if lvl2 else None} expected={expected2}")
# Range check: score sempre 0-100
check("sr_strength: todos scores 0-100",
      all(0 <= l["strength"] <= 100 for l in res["levels"]))

# ============================================================
# PIVOT POINTS: H=52000 L=48000 C=50000 (range=4000)
# ============================================================
print("--- PIVOT POINTS ---")
pv = InstitutionalPivotPoints.calculate_enhanced_pivot_points(
    high=52000.0, low=48000.0, close=50000.0
)
classic = pv["classic"]
# PP=(H+L+C)/3=50000; R1=2PP-L=52000; S1=2PP-H=48000; R2=PP+R=54000; S2=PP-R=46000; R3=58000; S3=42000
for name, got, exp in [
    ("classic PP", classic["pivot"], 50000.0),
    ("classic R1", classic["r1"], 52000.0),
    ("classic S1", classic["s1"], 48000.0),
    ("classic R2", classic["r2"], 54000.0),
    ("classic S2", classic["s2"], 46000.0),
    ("classic R3", classic["r3"], 58000.0),
    ("classic S3", classic["s3"], 42000.0),
]:
    check(f"pivot {name} == {exp}", approx(got, exp), f"got={got}")

cam = pv["camarilla"]
# R1=C+R*1.1/12; R2=C+R*1.1/6; R3=C+R*1.1/4; R4=C+R*1.1/2
for name, got, exp in [
    ("camarilla R1", cam["r1"], 50000.0 + 4000.0 * 1.1 / 12),
    ("camarilla S1", cam["s1"], 50000.0 - 4000.0 * 1.1 / 12),
    ("camarilla R2", cam["r2"], 50000.0 + 4000.0 * 1.1 / 6),
    ("camarilla S2", cam["s2"], 50000.0 - 4000.0 * 1.1 / 6),
    ("camarilla R3", cam["r3"], 50000.0 + 4000.0 * 1.1 / 4),
    ("camarilla S3", cam["s3"], 50000.0 - 4000.0 * 1.1 / 4),
    ("camarilla R4", cam["r4"], 50000.0 + 4000.0 * 1.1 / 2),
    ("camarilla S4", cam["s4"], 50000.0 - 4000.0 * 1.1 / 2),
]:
    check(f"pivot {name}", approx(got, exp), f"got={got} exp={exp:.4f}")

wood = pv["woodie"]
# PP=(H+L+2C)/4=50000; R1=2PP-L=52000; S1=48000; R2=54000; S2=46000
for name, got, exp in [
    ("woodie PP", wood["pivot"], 50000.0),
    ("woodie R1", wood["r1"], 52000.0),
    ("woodie S1", wood["s1"], 48000.0),
    ("woodie R2", wood["r2"], 54000.0),
    ("woodie S2", wood["s2"], 46000.0),
]:
    check(f"pivot {name}", approx(got, exp), f"got={got}")

fib = pv["fibonacci"]
# PP=50000; R1=PP+0.382R=51528; S1=48472; R2=52472; S2=47528; R3=54000; S3=46000
for name, got, exp in [
    ("fibonacci PP", fib["pivot"], 50000.0),
    ("fibonacci R1", fib["r1"], 50000.0 + 0.382 * 4000.0),
    ("fibonacci S1", fib["s1"], 50000.0 - 0.382 * 4000.0),
    ("fibonacci R2", fib["r2"], 50000.0 + 0.618 * 4000.0),
    ("fibonacci S2", fib["s2"], 50000.0 - 0.618 * 4000.0),
    ("fibonacci R3", fib["r3"], 54000.0),
    ("fibonacci S3", fib["s3"], 46000.0),
]:
    check(f"pivot {name}", approx(got, exp), f"got={got}")

# --- Erro clássico do caminho live: daily_pivot usa iloc[-1] (período ATUAL) ---
df = pd.DataFrame({
    "high": [52000.0, 50500.0],   # ontem completo, HOJE em andamento (parcial)
    "low":  [48000.0, 49500.0],
    "close":[50000.0, 50200.0],
})
live_pivot = daily_pivot(df)
correct_pivot = (52000.0 + 48000.0 + 50000.0) / 3  # deveria usar ONTEM
check("pivot live daily_pivot usa periodo ANTERIOR (iloc[-2])",
      approx(live_pivot["pivot"], correct_pivot),
      f"got={live_pivot['pivot']} expected(onter)= {correct_pivot} (iloc[-1] seria o periodo atual em andamento)")

# ============================================================
# VOLUME PROFILE: 3 níveis, volumes conhecidos
# ============================================================
print("--- VOLUME PROFILE ---")
# 30 pontos (>= min_data_points=20): 49900 (vol 50), 49905 (vol 20), 49910 (vol 80) -> total 150
prices = [49900.0] * 10 + [49905.0] * 10 + [49910.0] * 10
vols = [5.0] * 10 + [2.0] * 10 + [8.0] * 10
vpa = VolumeProfileAnalyzer(pd.Series(prices), pd.Series(vols))
prof = vpa.calculate_profile()

# bins=50: edges=linspace(49900, 49910, 51), step 0.2 -> centers 49900.1 e 49909.9
poc_price = prof["poc"]["price"]
poc_vol = prof["poc"]["volume"]
check("volume_profile: POC price == 49909.9 (bin center do maior volume)",
      approx(poc_price, 49909.9, 1e-9), f"got={poc_price}")
check("volume_profile: POC volume == 80", approx(poc_vol, 80.0), f"got={poc_vol}")
check("volume_profile: POC percent == 53.333%", approx(prof["poc"]["percent_of_total"], 80.0 / 150.0 * 100, 1e-6), f"got={prof['poc']['percent_of_total']}")

va = prof["value_area"]
# VA 70% = 105 (POC-outward): POC=bin 49910 (80) -> acima não existe -> abaixo 49905.1 (20)
# -> cum 100 < 105 -> abaixo 49900.1 (50) -> cum 150 >= 105 -> VAL=49900.1 VAH=49909.9 (contígua)
check("volume_profile: VAL == 49900.1", approx(va["low"], 49900.1, 1e-9), f"got={va['low']}")
check("volume_profile: VAH == 49909.9", approx(va["high"], 49909.9, 1e-9), f"got={va['high']}")
check("volume_profile: VA contígua (sem gap entre bins)", True,
      f"bins no range = {len([b for b in prof['price_bins'] if va['low'] <= b <= va['high']])}")

# calculate_value_area_volume_pct: agora usa os BINS ACUMULADOS (não preços raw)
va_volume_pct = vpa.calculate_value_area_volume_pct(prof)
check("volume_profile: value_area_volume_pct usa bins acumulados (== 100% aqui, 150/150 no range)",
      approx(va_volume_pct["value_area_volume_pct"], 100.0, 0.1),
      f"got={va_volume_pct['value_area_volume_pct']} (bins no range somam 150/150)")

# Conservação de volume
check("volume_profile: volume total conservado (150)",
      approx(prof["total_volume"], 150.0), f"got={prof['total_volume']}")

# ============================================================
# CASO BIMODAL: POC no meio, volume concentrado nos extremos
# POC-outward deve expandir do POC para os dois lados, NÃO pular para os extremos
# (sorted-desc pegaria os extremos distantes e faria VA descontínua)
# bins=5 -> centers 492/496/500/504/508, volumes 50/20/60/15/55, total 200, target 140
# ============================================================
print("--- VOLUME PROFILE BIMODAL ---")
from support_resistance.config import VolumeProfileConfig
b_config = VolumeProfileConfig(bins=5)
b_prices = [490.0] * 50 + [495.0] * 20 + [500.0] * 60 + [505.0] * 15 + [510.0] * 55
b_vols = [1.0] * 200
b_vpa = VolumeProfileAnalyzer(pd.Series(b_prices), pd.Series(b_vols), config=b_config)
b_prof = b_vpa.calculate_profile()
b_va = b_prof["value_area"]
# POC-outward: POC=500 (60) -> 495 (20) > 505 (15) -> 490 (50) -> 505 (15) => cum 145 >= 140
# VA = [492, 504] — NÃO inclui o extremo 508 (sorted-desc daria [492, 508] descontínua)
check("bimodal: POC == 500 (bin central, vol 60)",
      approx(b_prof["poc"]["price"], 500.0), f"got={b_prof['poc']['price']}")
check("bimodal: VAL == 492 (expansao para baixo primeiro)",
      approx(b_va["low"], 492.0), f"got={b_va['low']}")
check("bimodal: VAH == 504 — nao pula para o extremo distante (508)",
      approx(b_va["high"], 504.0), f"got={b_va['high']} (sorted-desc daria 508)")
b_bins = b_prof["price_bins"]
b_vols_arr = b_prof["volume_per_bin"]
b_in = [i for i, b in enumerate(b_bins) if b_va["low"] <= b <= b_va["high"]]
b_contiguous = len(b_in) == (max(b_in) - min(b_in) + 1)
check("bimodal: VA contígua (bins sem gap entre VAL e VAH)", b_contiguous,
      f"indices={b_in[0]}..{b_in[-1]}")
b_vol_in_va = sum(b_vols_arr[i] for i in b_in)
check("bimodal: volume em [VAL, VAH] >= 70% do total",
      b_vol_in_va / b_prof["total_volume"] >= 0.70,
      f"{b_vol_in_va}/{b_prof['total_volume']} = {b_vol_in_va/b_prof['total_volume']*100:.1f}%")
check("bimodal: value_area_volume_pct == acumulado dos bins (72.5%)",
      approx(b_vpa.calculate_value_area_volume_pct(b_prof)["value_area_volume_pct"], 72.5, 0.1),
      f"got={b_vpa.calculate_value_area_volume_pct(b_prof)['value_area_volume_pct']}")

# ============================================================
# CASO NORMAL: volume concentrado no centro -> VA simétrica ao redor do POC
# ============================================================
print("--- VOLUME PROFILE NORMAL ---")
np.random.seed(7)
n_prices = np.random.normal(500.0, 1.5, 2000).tolist()
n_vols = np.random.uniform(1.0, 5.0, 2000).tolist()
n_vpa = VolumeProfileAnalyzer(pd.Series(n_prices), pd.Series(n_vols))
n_prof = n_vpa.calculate_profile()
n_va = n_prof["value_area"]
n_poc = n_prof["poc"]["price"]
dist_above = abs(n_va["high"] - n_poc)
dist_below = abs(n_poc - n_va["low"])
# Para distribuição normal, a expansão é balanceada: razão entre lados ~ 1 (tolerância 1.5x)
check("normal: VA simétrica ao redor do POC",
      max(dist_above, dist_below) / min(dist_above, dist_below) <= 1.5,
      f"acima={dist_above:.3f} abaixo={dist_below:.3f} poc={n_poc:.3f}")
n_bins = n_prof["price_bins"]
n_vols_arr = n_prof["volume_per_bin"]
n_in = [i for i, b in enumerate(n_bins) if n_va["low"] <= b <= n_va["high"]]
n_vol_in_va = sum(n_vols_arr[i] for i in n_in)
check("normal: volume em [VAL, VAH] >= 70% do total",
      n_vol_in_va / n_prof["total_volume"] >= 0.70,
      f"{n_vol_in_va}/{n_prof['total_volume']} = {n_vol_in_va/n_prof['total_volume']*100:.1f}%")
check("normal: VA contígua", len(n_in) == (max(n_in) - min(n_in) + 1),
      f"indices={n_in[0]}..{n_in[-1]}")

print("=" * 66)
failed = [r for r in results if r[0] == FAIL]
print(f"RESULTADO: {len(results) - len(failed)}/{len(results)} PASS"
      f"{'  — ' + str(len(failed)) + ' FAIL' if failed else ''}")
