"""
Valida que dados falsos foram removidos do payload.
Executar: python scripts/migration/validate_fix4_fake_data.py [evento.json]
"""

import json
import sys


def check_fake_data(event: dict) -> tuple[list, list]:
    issues = []
    passed = []

    # 1. options_metrics não deve existir se is_real_data=False
    raw_aa = event.get("raw_event", {}).get("advanced_analysis", {})
    options = raw_aa.get("options_metrics", {})
    if options and not options.get("is_real_data"):
        issues.append("options_metrics com is_real_data=False ainda presente")
    elif not options:
        passed.append("options_metrics removido (sem dados reais)")
    else:
        passed.append("options_metrics presente com dados reais")

    # 2. whale_activity deve ser consistente
    whale = event.get("whale_activity", {})
    fc = event.get("fluxo_continuo", {})
    whale_buy = fc.get("whale_buy_volume", 0) or 0
    whale_sell = fc.get("whale_sell_volume", 0) or 0

    if whale.get("iceberg_activity", 0) == 1 and (whale_buy + whale_sell) == 0:
        issues.append("whale_activity diz detected=1 mas volume total=0")
    else:
        passed.append("whale_activity consistente com volumes")

    # 3. price_targets
    pt = event.get("price_targets", {})
    if pt:
        conf = pt.get("model_confidence", 1)
        if conf < 0.20:
            issues.append(f"price_targets com model_confidence={conf} (< 0.20)")
        else:
            passed.append(f"price_targets com confidence={conf} (aceitavel)")
    else:
        passed.append("price_targets removido (baixa confianca)")

    # 4. onchain com zeros removidos
    onchain = raw_aa.get("onchain_metrics", {})
    if onchain:
        zero_fields = {k: v for k, v in onchain.items()
                       if v == 0 and k in ["hash_rate", "exchange_netflow",
                                            "whale_transactions", "exchange_reserves", "sopr"]}
        if zero_fields:
            issues.append(f"onchain_metrics: {len(zero_fields)} campos zerados presentes: "
                          f"{list(zero_fields.keys())}")
        else:
            passed.append("onchain_metrics: campos zerados removidos")

    # 5. data_reliability flag
    dr = event.get("data_reliability", {})
    if dr:
        passed.append(f"data_reliability presente: {json.dumps(dr)}")
    else:
        issues.append("campo data_reliability nao encontrado no evento")

    # Tamanho
    total = len(json.dumps(event))
    passed.append(f"Tamanho total: {total:,} bytes")

    return issues, passed


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "dados/last_event.json"

    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
        if "\n" in content:
            content = content.strip().split("\n")[-1]
        event = json.loads(content)

    print("=" * 60)
    print("VALIDACAO FIX 4 - DADOS FALSOS")
    print("=" * 60)

    issues, passed = check_fake_data(event)

    for p in passed:
        print(f"  OK: {p}")

    if issues:
        print(f"\n  PROBLEMAS:")
        for i in issues:
            print(f"  FAIL: {i}")

    print(f"\n{'='*60}")
    if not issues:
        print("DADOS FALSOS REMOVIDOS")
    else:
        print(f"{len(issues)} problemas restantes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
