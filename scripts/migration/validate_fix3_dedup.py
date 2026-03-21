"""
Valida que duplicações foram removidas sem quebrar funcionalidade.
Executar: python scripts/migration/validate_fix3_dedup.py [evento.json]
"""

import json
import sys


def check_deduplication(event: dict) -> tuple[list, list]:
    issues = []
    passed = []

    # 1. advanced_analysis não deve estar em contextual_snapshot
    cs = event.get("contextual_snapshot", {})
    if "advanced_analysis" in cs:
        aa_size = len(json.dumps(cs["advanced_analysis"]))
        issues.append(f"advanced_analysis AINDA em contextual_snapshot ({aa_size:,} bytes)")
    else:
        passed.append("advanced_analysis removido de contextual_snapshot")

    # Mas deve existir em raw_event
    if "advanced_analysis" in event.get("raw_event", {}):
        passed.append("advanced_analysis mantido em raw_event")

    # 2. buy_sell_ratio não deve estar em institutional_analytics
    ia = event.get("institutional_analytics", {}).get("flow_analysis", {})
    if "buy_sell_ratio" in ia:
        issues.append("buy_sell_ratio AINDA em institutional_analytics.flow_analysis")
    else:
        passed.append("buy_sell_ratio removido de institutional_analytics")

    # Mas deve existir em fluxo_continuo
    fc = event.get("fluxo_continuo", {}).get("order_flow", {})
    if "buy_sell_ratio" in fc:
        passed.append("buy_sell_ratio mantido em fluxo_continuo.order_flow")

    # 3. volume_profile_advanced não deve existir
    if "volume_profile_advanced" in event:
        vpa_size = len(json.dumps(event["volume_profile_advanced"]))
        issues.append(f"volume_profile_advanced AINDA existe ({vpa_size:,} bytes)")
    else:
        passed.append("volume_profile_advanced removido")

    # historical_vp deve existir
    if "historical_vp" in event:
        passed.append("historical_vp mantido (fonte canônica)")

    # 4. spread_analysis e orderbook_data_quality não devem existir separados
    if "spread_analysis" in event:
        issues.append("spread_analysis AINDA existe como seção separada")
    else:
        passed.append("spread_analysis removido (dados em orderbook_data)")

    if "orderbook_data_quality" in event:
        issues.append("orderbook_data_quality AINDA existe como seção separada")
    else:
        passed.append("orderbook_data_quality removido (dados em orderbook_data)")

    # Mas orderbook_data deve ter spread_bps e is_valid
    ob = event.get("orderbook_data", {})
    if "spread_bps" not in ob and "spread_percent" not in ob:
        issues.append("orderbook_data sem spread (deveria ter sido movido de spread_analysis)")

    # 5. sr_analysis.sr_strength não deve existir
    sr = event.get("institutional_analytics", {}).get("sr_analysis", {})
    if "sr_strength" in sr:
        levels_count = len(sr["sr_strength"].get("levels", []))
        sr_size = len(json.dumps(sr["sr_strength"]))
        issues.append(f"sr_strength AINDA existe ({levels_count} levels, {sr_size:,} bytes)")
    else:
        passed.append("sr_strength removido (defense_zones eh canonico)")

    # defense_zones deve existir
    if "defense_zones" in sr:
        passed.append("defense_zones mantido")

    # 6. price_targets não deve estar em raw_event.advanced_analysis
    raw_aa = event.get("raw_event", {}).get("advanced_analysis", {})
    if "price_targets" in raw_aa:
        issues.append("price_targets AINDA em raw_event.advanced_analysis")
    else:
        passed.append("price_targets removido de raw_event.advanced_analysis")

    # price_targets root deve existir
    if "price_targets" in event:
        passed.append("price_targets mantido no root")

    # Tamanho total
    total_bytes = len(json.dumps(event))
    passed.append(f"Tamanho total do evento: {total_bytes:,} bytes")

    return issues, passed


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "dados/last_event.json"

    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
        if "\n" in content:
            content = content.strip().split("\n")[-1]
        event = json.loads(content)

    print("=" * 60)
    print("VALIDACAO FIX 3 - DEDUPLICACAO")
    print("=" * 60)

    issues, passed = check_deduplication(event)

    for p in passed:
        print(f"  OK: {p}")

    if issues:
        print(f"\n  PROBLEMAS:")
        for i in issues:
            print(f"  FAIL: {i}")

    print(f"\n{'='*60}")
    if not issues:
        print("DEDUPLICACAO COMPLETA")
    else:
        print(f"{len(issues)} duplicacoes restantes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
