"""Medição do efeito do dedup em dados reais (auditoria ETAPA 5).

Reproduz o pipeline real do dia 2026-08-10 14:16 (memory/levels_BTCUSDT.json):
- vp_data do historical profiler (bins de $1; HVNs reais)
- sr_levels = output real do SRStrengthScorer sobre o MESMO vp_data
- pivot_data real (macro_context["pivots"]) + ema_values reais

Mede:
  A) sinais por rota (direta vs sr_level_*)
  B) colapsos do dedup ATUAL (tolerância = zone_width 0.15%)
  C) colapsos do dedup NOVO (identidade: (fonte canônica, preço tick-compatível))
  D) estrutura da zona final (source_count vs signals_in_zone)
"""
import sys
import json

sys.path.insert(0, ".")

from support_resistance.defense_zones import DefenseZoneDetector
from support_resistance.sr_strength import SRStrengthScorer

CURRENT_PRICE = 64892.0

VP_DAILY = {
    "poc": 64689.0,
    "vah": 65130.05,
    "val": 64520.0,
    "hvns": [
        64545.33, 64564.0, 64591.0, 64638.0, 64667.33,
        64689.0, 64737.0, 65001.99, 65024.81,
    ],
    "status": "success",
}

PIVOT_DATA = {
    "daily": {
        "pivot": 65035.38, "r1": 65340.67, "s1": 64596.29,
        "r2": 65779.76, "s2": 64291.0, "r3": 66085.05, "s3": 63851.91,
        "high": 65474.46, "low": 64730.08, "close": 64901.59,
    },
    "weekly": {
        "pivot": 63751.0, "r1": 64461.0, "s1": 63362.0,
        "high": 64461.0, "low": 63362.0, "close": 63751.0,
    },
    "calculated_at_ms": 1754300000000,
}

EMAS = {
    "ema_21_15m": 64850.5,
    "ema_21_1h": 64870.0,
    "ema_21_4h": 64890.0,
    "ema_21_1d": 64900.0,
}


def identity_dedup(signals, current_price):
    """Proposta: dedup por IDENTIDADE — (fonte canônica, preço tick-compatível).
    Sem colapso por proximidade: bins distintos da mesma fonte permanecem."""
    d = DefenseZoneDetector()
    buckets = {}
    for sig in signals:
        key = d._canonical_source(sig.get("source", "unknown"))
        buckets.setdefault(key, []).append(sig)
    deduped = []
    for key, bucket in buckets.items():
        groups = {}
        for sig in bucket:
            tick = round(sig["price"], 2)
            groups.setdefault(tick, []).append(sig)
        for tick, g in groups.items():
            best = dict(max(g, key=lambda s: (s.get("strength", 0), s["price"])))
            best["source"] = key
            deduped.append(best)
    return deduped


def build_signals():
    d = DefenseZoneDetector()
    signals = []
    signals += d._extract_vp_defense(VP_DAILY, CURRENT_PRICE)
    scorer = SRStrengthScorer()
    sr = scorer.score_levels(
        current_price=CURRENT_PRICE,
        vp_data=VP_DAILY,
        pivot_data=PIVOT_DATA,
        ema_values=EMAS,
        recent_candles=None,
    )
    sr_levels = [l for l in sr.get("levels", []) if l.get("strength", 0) > 30]
    signals += d._extract_sr_defense(sr_levels, CURRENT_PRICE)
    signals += d._extract_pivot_defense(PIVOT_DATA, CURRENT_PRICE)
    signals += d._extract_ema_defense(EMAS, CURRENT_PRICE)
    return signals, sr_levels


def main():
    signals, sr_levels = build_signals()

    print(f"current_price={CURRENT_PRICE}  tolerance_0.15%={CURRENT_PRICE*0.0015:.2f} USD")
    print(f"sr_levels do scorer (>30): {len(sr_levels)}")
    print(f"sinais totais (2 rotas + pivots + emas): {len(signals)}")
    print(f"  via vp direto: {len([s for s in signals if not s['source'].startswith('sr_level_')])}")
    print(f"  via sr_level_: {len([s for s in signals if s['source'].startswith('sr_level_')])}")
    print()

    d = DefenseZoneDetector()

    print("== A) Sinais por fonte bruta ==")
    from collections import Counter
    for src, n in sorted(Counter(s["source"] for s in signals).items()):
        prices = sorted(round(s["price"], 2) for s in signals if s["source"] == src)
        print(f"  {src:28s} x{n}  {prices}")

    print()
    print("== B) Dedup ATUAL (tolerância 0.15% por fonte canônica) ==")
    old = d._dedupe_signals([dict(s) for s in signals], CURRENT_PRICE)
    print(f"  {len(signals)} -> {len(old)} sinais  (colapsados: {len(signals)-len(old)})")
    for sig in sorted(old, key=lambda s: s["price"]):
        print(f"    {sig['source']:24s} {sig['price']:>10.2f}  str={sig['strength']}")

    print()
    print("== C) Dedup NOVO (identidade: fonte canônica + preço tick-compatível) ==")
    new = identity_dedup([dict(s) for s in signals], CURRENT_PRICE)
    print(f"  {len(signals)} -> {len(new)} sinais  (colapsados: {len(signals)-len(new)})")
    for sig in sorted(new, key=lambda s: s["price"]):
        print(f"    {sig['source']:24s} {sig['price']:>10.2f}  str={sig['strength']}")

    print()
    print("== D) Zonas finais com o dedup NOVO ==")
    zones = d._cluster_signals(new, CURRENT_PRICE)
    for z in sorted(zones, key=lambda z: z["center"]):
        print(f"    center={z['center']:>9.2f} side={z['side']:4s} "
              f"source_count={z['source_count']} signals={z['signals_in_zone']} "
              f"sources={sorted(z['sources'])}  strength={z['strength']}")


if __name__ == "__main__":
    main()
