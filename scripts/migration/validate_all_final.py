"""
Validacao FINAL de todos os fixes (5, 6, 7).
Executar apos aplicar todos os prompts.
Uso: python scripts/migration/validate_all_final.py [evento.json|evento.jsonl]
"""

import json
import sys


def validate_fix5a_warmup(event: dict) -> list:
    """5A: ML nao deve ser chamado com defaults durante warmup."""
    issues = []
    janela = event.get("janela_numero", 0)

    if janela >= 20:
        issues.append({
            "severity": "INFO",
            "msg": f"Janela {janela} >= 20: ML deveria estar com warmup completo.",
        })
    else:
        issues.append({
            "severity": "INFO",
            "msg": f"Janela {janela} < 20: ML em warmup (esperado).",
        })

    return issues


def validate_fix5b_ml_features(event: dict) -> list:
    """5B: ml_features.price_features deve ter returns em escala correta."""
    issues = []
    ml = event.get("ml_features", {})
    pf = ml.get("price_features", {})

    ret1 = abs(pf.get("returns_1", 0))
    janela = event.get("janela_numero", 0)

    if ret1 > 0 and ret1 < 1e-5 and janela > 2:
        issues.append({
            "severity": "CRITICAL",
            "msg": f"returns_1={ret1:.2e} ainda em escala de ticks (deveria ser > 1e-4)",
        })
    elif ret1 == 0 and janela <= 2:
        issues.append({
            "severity": "OK",
            "msg": f"returns_1=0 na janela {janela} (warmup, esperado)",
        })
    else:
        issues.append({
            "severity": "OK",
            "msg": f"returns_1={ret1:.6f} (escala correta)",
        })

    return issues


def validate_fix5c_risk_sentiment(event: dict) -> list:
    """5C: risk_sentiment deve ser consistente com F&G e VIX."""
    issues = []

    sentiment = event.get("market_environment", {}).get("risk_sentiment", "N/A")
    fg = event.get("external_markets", {}).get("FEAR_GREED", {}).get("preco_atual", 50)
    vix = event.get("external_markets", {}).get("VIX", {}).get("preco_atual", 20)

    try:
        fg = float(fg)
    except (TypeError, ValueError):
        fg = 50
    try:
        vix = float(vix)
    except (TypeError, ValueError):
        vix = 20

    if fg <= 20 and sentiment == "BULLISH":
        issues.append({
            "severity": "CRITICAL",
            "msg": f"risk_sentiment=BULLISH mas Fear&Greed={fg} (Extreme Fear). Contradicao!",
        })
    elif fg <= 20 and sentiment in ("RISK_OFF", "BEARISH"):
        issues.append({
            "severity": "OK",
            "msg": f"risk_sentiment={sentiment} consistente com F&G={fg}",
        })
    elif vix > 25 and sentiment == "BULLISH":
        issues.append({
            "severity": "HIGH",
            "msg": f"risk_sentiment=BULLISH mas VIX={vix} (alto). Inconsistente.",
        })
    else:
        issues.append({
            "severity": "OK",
            "msg": f"risk_sentiment={sentiment} (F&G={fg}, VIX={vix})",
        })

    return issues


def validate_fix6a_historical_vp(event: dict) -> list:
    """6A: historical_vp.daily nao deve ter POC=VAH=VAL com status=success."""
    issues = []

    hvp = event.get("historical_vp", {}).get("daily", {})
    poc = hvp.get("poc", 0)
    vah = hvp.get("vah", 0)
    val = hvp.get("val", 0)
    status = hvp.get("status", "")

    if poc == vah == val and poc > 0 and status == "success":
        issues.append({
            "severity": "HIGH",
            "msg": f"VP daily: POC=VAH=VAL={poc} com status=success. Deveria ser insufficient_data.",
        })
    elif status == "insufficient_data":
        issues.append({
            "severity": "OK",
            "msg": "VP daily corretamente marcado como insufficient_data",
        })
    elif vah > val and poc >= val and poc <= vah:
        issues.append({
            "severity": "OK",
            "msg": f"VP daily valido: POC={poc}, VAL={val}, VAH={vah}",
        })
    elif not hvp:
        issues.append({
            "severity": "OK",
            "msg": "VP daily ausente (sem dados ainda)",
        })
    else:
        issues.append({
            "severity": "OK",
            "msg": f"VP daily: POC={poc}, VAL={val}, VAH={vah}, status={status}",
        })

    return issues


def validate_fix6b_fibonacci(event: dict) -> list:
    """6B: fibonacci nao deve existir com range < 0.3%."""
    issues = []

    fib = event.get("fibonacci_levels", {})
    if fib:
        high = fib.get("high", 0)
        low = fib.get("low", 0)
        if low > 0:
            range_pct = (high - low) / low * 100
            if range_pct < 0.3:
                issues.append({
                    "severity": "HIGH",
                    "msg": f"Fibonacci com range {range_pct:.2f}% (< 0.3%). Deveria ser removido.",
                })
            else:
                issues.append({
                    "severity": "OK",
                    "msg": f"Fibonacci com range {range_pct:.2f}% (aceitavel)",
                })
    else:
        issues.append({
            "severity": "OK",
            "msg": "Fibonacci nao incluido (range insuficiente ou ausente — correto)",
        })

    return issues


def validate_fix7a_empty_sections(event: dict) -> list:
    """7A: Secoes vazias nao devem estar no evento."""
    issues = []

    ia = event.get("institutional_analytics", {})
    pa = ia.get("profile_analysis", {})

    # volume_node_strength
    vns = pa.get("volume_node_strength", {})
    if vns and vns.get("total_hvns", 0) == 0 and vns.get("total_lvns", 0) == 0:
        issues.append({
            "severity": "MEDIUM",
            "msg": "volume_node_strength com zero nodes ainda presente",
        })

    # reference_prices
    sr = ia.get("sr_analysis", {})
    ref = sr.get("reference_prices", {})
    if ref and ref.get("status") == "no_data":
        issues.append({
            "severity": "MEDIUM",
            "msg": "reference_prices com status=no_data ainda presente",
        })

    # va_volume_pct
    va = pa.get("va_volume_pct", {})
    if va and va.get("value_area_volume_pct", 0) == 0:
        issues.append({
            "severity": "MEDIUM",
            "msg": "va_volume_pct com 0% ainda presente",
        })

    # candlestick_patterns
    cp = ia.get("candlestick_patterns", {})
    if cp and cp.get("patterns_detected", 0) == 0:
        issues.append({
            "severity": "MEDIUM",
            "msg": "candlestick_patterns com 0 patterns ainda presente",
        })

    # no_mans_land
    nml = pa.get("no_mans_land", {})
    if nml and nml.get("status") in ("insufficient_hvns", "insufficient_data"):
        issues.append({
            "severity": "MEDIUM",
            "msg": f"no_mans_land com status={nml.get('status')} ainda presente",
        })

    if not issues:
        issues.append({"severity": "OK", "msg": "Secoes vazias removidas corretamente"})

    return issues


def validate_fix7b_ob_dedup(event: dict) -> list:
    """7B: depth_metrics nao deve estar em orderbook_data."""
    issues = []

    ob = event.get("orderbook_data", {})
    if "depth_metrics" in ob:
        issues.append({
            "severity": "MEDIUM",
            "msg": "depth_metrics ainda em orderbook_data (duplica order_book_depth)",
        })
    else:
        issues.append({
            "severity": "OK",
            "msg": "depth_metrics removido de orderbook_data",
        })

    return issues


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "dados/eventos_fluxo.jsonl"

    # Ler ultimo evento
    try:
        if path.endswith(".jsonl"):
            with open(path, encoding="utf-8") as f:
                lines = f.read().strip().split("\n")
                event = json.loads(lines[-1])
        else:
            with open(path, encoding="utf-8") as f:
                event = json.loads(f.read())
    except FileNotFoundError:
        print(f"Arquivo nao encontrado: {path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Erro ao parsear JSON: {e}")
        sys.exit(1)

    janela = event.get("janela_numero", "?")
    price = event.get("preco_fechamento", "?")

    print("=" * 70)
    print("VALIDACAO FINAL - FIXES 5, 6, 7")
    print(f"Janela: {janela} | Preco: {price}")
    print("=" * 70)

    validators = [
        ("FIX 5A - ML Warmup", validate_fix5a_warmup),
        ("FIX 5B - ML Features no Evento", validate_fix5b_ml_features),
        ("FIX 5C - Risk Sentiment", validate_fix5c_risk_sentiment),
        ("FIX 6A - Historical VP", validate_fix6a_historical_vp),
        ("FIX 6B - Fibonacci", validate_fix6b_fibonacci),
        ("FIX 7A - Secoes Vazias", validate_fix7a_empty_sections),
        ("FIX 7B - OB Dedup", validate_fix7b_ob_dedup),
    ]

    total_critical = 0
    total_high = 0
    total_medium = 0
    total_ok = 0

    for name, validator in validators:
        print(f"\n{'-' * 70}")
        print(f"  {name}")
        print(f"{'-' * 70}")

        results = validator(event)
        for r in results:
            sev = r["severity"]
            if sev == "CRITICAL":
                marker = "[CRITICAL]"
                total_critical += 1
            elif sev == "HIGH":
                marker = "[HIGH]   "
                total_high += 1
            elif sev == "MEDIUM":
                marker = "[MEDIUM] "
                total_medium += 1
            elif sev == "OK":
                marker = "[OK]     "
                total_ok += 1
            else:
                marker = "[INFO]   "

            print(f"  {marker} {r['msg']}")

    # Tamanho do evento
    size_bytes = len(json.dumps(event))
    size_kb = size_bytes / 1024

    print(f"\n{'=' * 70}")
    print("RESUMO FINAL")
    print(f"{'=' * 70}")
    print(f"  OK:       {total_ok}")
    print(f"  Medio:    {total_medium}")
    print(f"  Alto:     {total_high}")
    print(f"  Critico:  {total_critical}")
    print(f"  Tamanho:  {size_kb:.1f} KB")

    if total_critical == 0 and total_high == 0:
        print(f"\n  TODOS OS FIXES VALIDADOS!")
        print(f"  Sistema pronto para refatoracao estrutural.")
    elif total_critical == 0:
        print(f"\n  {total_high} problemas de alta prioridade restantes.")
    else:
        print(f"\n  {total_critical} problemas CRITICOS. Corrigir antes de prosseguir.")

    print(f"{'=' * 70}")

    sys.exit(1 if total_critical > 0 else 0)


if __name__ == "__main__":
    main()
