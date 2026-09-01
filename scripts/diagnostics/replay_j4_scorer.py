"""ETAPA 5B FASE 9 - Replay do SRStrengthScorer com dados J4 (nao commitado).

Compara candidatos com/sem H/L/C de pivot_classic no scorer standalone.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import pandas as pd
from support_resistance.sr_strength import SRStrengthScorer

PRICE = 64742.1
pivots = {
    "pivot": 65035.38, "r1": 65340.67, "s1": 64596.29,
    "r2": 65779.76, "s2": 64291.0, "r3": 66085.05, "s3": 63851.91,
    "high": 65474.46, "low": 64730.08, "close": 64901.59,
}
vp_data = {"poc": 0, "vah": 0, "val": 0, "hvns": [64737.0]}

candles = pd.DataFrame({
    "high": [PRICE + 100, 65474.0, 65475.0, 64730.0, 64731.0, 64900.0, 65475.5],
    "low": [PRICE - 100, 64730.5, 64729.5, 64729.0, 65473.0, 64800.0, 64729.0],
    "close": [PRICE] * 7,
})


def tab(levels):
    return [{
        "price": l.get("price"),
        "primary_source": l.get("primary_source"),
        "confluences": l.get("confluences"),
        "c_count": l.get("confluence_count"),
        "touches": l.get("touches"),
        "dist_pct": l.get("distance_pct"),
        "strength": l.get("strength"),
        "type": l.get("type"),
    } for l in levels]


scorer = SRStrengthScorer()
r_with = scorer.score_levels(
    current_price=PRICE, vp_data=vp_data, pivot_data={"classic": pivots},
    recent_candles=candles,
)
pivots_no_hlc = {k: v for k, v in pivots.items() if k not in ("high", "low", "close")}
r_without = scorer.score_levels(
    current_price=PRICE, vp_data=vp_data, pivot_data={"classic": pivots_no_hlc},
    recent_candles=candles,
)

print("=== COM H/L/C ===")
for row in tab(r_with["levels"]):
    print(json.dumps(row, ensure_ascii=False))
print("=== SEM H/L/C ===")
for row in tab(r_without["levels"]):
    print(json.dumps(row, ensure_ascii=False))
