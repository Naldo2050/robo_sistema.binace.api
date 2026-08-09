# -*- coding: utf-8 -*-
"""
Auditoria numérica: support_resistance/ (sr_strength, pivot_points, volume_profile)

Testa a matemática com valores conhecidos e compara com cálculo manual.
Não altera código de produção.
"""
import sys
import os
import math
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
# VA 70% = 105: sorted desc -> bin 49910 (80) + bin 49900 (50) = 130 >= 105 -> VAL=49900.1 VAH=49909.9
check("volume_profile: VAL == 49900.1", approx(va["low"], 49900.1, 1e-9), f"got={va['low']}")
check("volume_profile: VAH == 49909.9", approx(va["high"], 49909.9, 1e-9), f"got={va['high']}")

# Demonstração do método VA (sorted-desc, NÃO POC-outward): bin do meio (49905, vol 20)
# NÃO entra na VA (acumulado 80+50=130 >= 105) apesar de estar entre os dois bins escolhidos -> VA descontínua
# E calculate_value_area_volume_pct refaz o mask sobre os preços ORIGINAIS com as bordas dos bins
# (VAL/VAH são centers): trades exatamente nas bordas ficam fora -> só o bin interno conta: 20/150 = 13.3%
va_volume_pct = vpa.calculate_value_area_volume_pct(prof)
expected_edge_pct = 20.0 / 150.0 * 100.0
check("volume_profile: value_area_volume_pct sensível a bordas de bin "
      "(trades em bordas ficam fora do range VAL/VAH)",
      approx(va_volume_pct["value_area_volume_pct"], expected_edge_pct, 0.1),
      f"got={va_volume_pct['value_area_volume_pct']} (trades nas bordas; acumulado VA real=130/150={130/150*100:.1f}%, range teo=100%)")

# Conservação de volume
check("volume_profile: volume total conservado (150)",
      approx(prof["total_volume"], 150.0), f"got={prof['total_volume']}")

print("=" * 66)
failed = [r for r in results if r[0] == FAIL]
print(f"RESULTADO: {len(results) - len(failed)}/{len(results)} PASS"
      f"{'  — ' + str(len(failed)) + ' FAIL' if failed else ''}")
