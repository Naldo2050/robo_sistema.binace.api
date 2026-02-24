# ═══════════════════════════════════════
# Teste #4 — S/R Strength Scoring
# ═══════════════════════════════════════
from support_resistance.sr_strength import SRStrengthScorer

scorer = SRStrengthScorer()

result = scorer.score_levels(
    current_price=64892,
    vp_data={
        "poc": 64888, "vah": 66055, "val": 64683,
        "hvns": [64428, 64888, 66818],
    },
    pivot_data={
        "classic": {"PP": 64850, "R1": 65100, "R2": 65400, "S1": 64600, "S2": 64300},
    },
    ema_values={
        "ema_21_1h": 65418,
        "ema_21_4h": 66695,
        "ema_21_1d": 70411,
    },
    weekly_vp={"poc": 68110, "vah": 69125, "val": 66875},
    monthly_vp={"poc": 77540, "vah": 84505, "val": 73139},
)

print(f"S/R Strength: {result['status']}")
print(f"Total levels: {result['total_levels_found']}")
assert result["status"] == "success"
assert len(result["levels"]) > 0

# Verificar que POC tem score alto (confluência com PP)
poc_levels = [l for l in result["levels"] if "poc" in l.get("primary_source", "")]
if poc_levels:
    print(f"  POC strength: {poc_levels[0]['strength']}")

if result["nearest_support"]:
    print(f"  Nearest support: {result['nearest_support']['price']} (str={result['nearest_support']['strength']})")
if result["nearest_resistance"]:
    print(f"  Nearest resistance: {result['nearest_resistance']['price']} (str={result['nearest_resistance']['strength']})")
print("✅ #4 S/R Strength OK")


# ═══════════════════════════════════════
# Teste #5 — Defense Zones
# ═══════════════════════════════════════
from support_resistance.defense_zones import DefenseZoneDetector

detector = DefenseZoneDetector()

zones = detector.detect(
    current_price=64892,
    orderbook_data={
        "bid_depth_usd": 552161.86,
        "ask_depth_usd": 477276.84,
        "imbalance": 0.073,
        "depth_metrics": {"depth_imbalance": -0.172},
    },
    vp_data={
        "poc": 64888, "vah": 66055, "val": 64683,
        "hvns": [64428, 64888, 66818],
    },
    pivot_data={
        "classic": {"PP": 64850, "R1": 65100, "S1": 64600},
    },
    absorption_events=[
        {"price": 64700, "type": "Absorção de Compra", "strength": 0.65},
        {"price": 64690, "type": "Absorção de Compra", "strength": 0.45},
        {"price": 65050, "type": "Absorção de Venda", "strength": 0.55},
    ],
    ema_values={"ema_21_1h": 65418, "ema_21_4h": 66695},
)

print(f"\nDefense Zones: {zones['status']}")
print(f"Total zones: {zones['total_zones']}")
print(f"Buy defense: {len(zones['buy_defense'])}")
print(f"Sell defense: {len(zones['sell_defense'])}")
assert zones["status"] == "success"

if zones["strongest_buy"]:
    sb = zones["strongest_buy"]
    print(f"  Strongest buy: {sb['center']} (str={sb['strength']}, sources={sb['source_count']})")
if zones["strongest_sell"]:
    ss = zones["strongest_sell"]
    print(f"  Strongest sell: {ss['center']} (str={ss['strength']}, sources={ss['source_count']})")

print(f"  Asymmetry: {zones['defense_asymmetry']['bias']}")
print("✅ #5 Defense Zones OK")


# ═══════════════════════════════════════
# Teste #14 — Passive/Aggressive Flow
# ═══════════════════════════════════════
from flow_analyzer.aggregates import analyze_passive_aggressive_flow

pa_result = analyze_passive_aggressive_flow(
    flow_data={
        "aggressive_buy_pct": 44.95,
        "aggressive_sell_pct": 55.05,
        "buy_volume_btc": 3.578,
        "sell_volume_btc": 4.383,
        "flow_imbalance": -0.101,
    },
    orderbook_data={
        "bid_depth_usd": 552161.86,
        "ask_depth_usd": 477276.84,
        "imbalance": 0.073,
    },
)

print(f"\nPassive/Aggressive: {pa_result['status']}")
print(f"  Aggressive dominance: {pa_result['aggressive']['dominance']}")
print(f"  Passive dominance: {pa_result['passive']['dominance']}")
print(f"  Composite signal: {pa_result['composite']['signal']}")
print(f"  Interpretation: {pa_result['composite']['interpretation']}")
assert pa_result["status"] == "success"
assert pa_result["composite"]["signal"] in (
    "strong_bullish", "strong_bearish", "buy_absorption",
    "sell_absorption", "neutral_balanced", "mixed", "passive_unknown"
)
print("✅ #14 Passive/Aggressive OK")


# ═══════════════════════════════════════
# Teste #17 — Absorption Zone Mapping
# ═══════════════════════════════════════
from flow_analyzer.absorption import AbsorptionZoneMapper
import time

mapper = AbsorptionZoneMapper()

now_ms = int(time.time() * 1000)

# Simular vários eventos de absorção na mesma zona
mapper.record_event(price=64700, classification="Absorção de Compra",
                   index=0.65, timestamp_ms=now_ms - 3600000, volume_usd=500000)
mapper.record_event(price=64710, classification="Absorção de Compra",
                   index=0.55, timestamp_ms=now_ms - 1800000, volume_usd=350000)
mapper.record_event(price=64695, classification="Absorção de Compra",
                   index=0.70, timestamp_ms=now_ms - 600000, volume_usd=600000)

# Absorção em zona diferente
mapper.record_event(price=65050, classification="Absorção de Venda",
                   index=0.45, timestamp_ms=now_ms - 900000, volume_usd=400000)
mapper.record_event(price=65060, classification="Absorção de Venda",
                   index=0.50, timestamp_ms=now_ms - 300000, volume_usd=450000)

zones = mapper.get_zones(current_price=64892)

print(f"\nAbsorption Zones: {zones['status']}")
print(f"Total zones: {zones['total_zones']}")
print(f"Total events: {zones['total_events']}")
print(f"Buy zones: {zones['buy_zone_count']}")
print(f"Sell zones: {zones['sell_zone_count']}")
assert zones["status"] == "success"
assert zones["total_events"] == 5
assert zones["total_zones"] >= 2

if zones["strongest_zone"]:
    sz = zones["strongest_zone"]
    print(f"  Strongest zone: {sz['center']} ({sz['dominant_side']}, "
          f"events={sz['event_count']}, reliability={sz['reliability']})")

# Resumo
summary = mapper.get_summary()
print(f"  Summary: {summary}")
assert summary["status"] == "ok"
print("✅ #17 Absorption Zone Mapping OK")


print("\n🎉 DIA 4 — TODOS OS TESTES PASSARAM!")