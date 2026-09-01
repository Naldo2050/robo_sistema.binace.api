"""ETAPA 5B FASE 3 - Captura do payload compacto real (secao sr) com dados J4."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from market_orchestrator.ai.payload_builder_compact import build_compact_payload
from support_resistance.defense_zones import DefenseZoneDetector

PRICE = 64742.1
res = DefenseZoneDetector().detect(
    current_price=PRICE,
    orderbook_data={"bid_depth_usd": 50000, "ask_depth_usd": 100000, "imbalance": -0.25},
    vp_data={"poc": 0, "vah": 0, "val": 0, "hvns": [64737.0]},
)

event = {
    "preco_fechamento": PRICE,
    "pivots": {"daily": {
        "pivot": 65035.38, "r1": 65340.67, "s1": 64596.29,
        "r2": 65779.76, "s2": 64291.0, "r3": 66085.05, "s3": 63851.91,
        "high": 65474.46, "low": 64730.08, "close": 64901.59,
    }},
    "historical_vp": {"daily": {"poc": 64689, "vah": 65133, "val": 64520, "status": "success"}},
    "institutional_analytics": {"sr_analysis": {"defense_zones": res}},
}

payload = build_compact_payload(event)
sr = payload.get("sr", {})
ctx = payload.get("ctx", {})
print("=== payload['sr'] ===")
print(json.dumps(sr, ensure_ascii=False, indent=1))
print("=== payload['ctx'] ===")
print(json.dumps(ctx, ensure_ascii=False, indent=1))
print("=== payload['price'] ===")
print(json.dumps(payload.get("price", {}), ensure_ascii=False))
