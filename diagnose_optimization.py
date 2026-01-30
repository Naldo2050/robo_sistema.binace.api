import argparse, json
from pathlib import Path
from collections import defaultdict

FORBIDDEN = {"observability","processing_times_ms","circuit_breaker","memory","volume_nodes","single_prints","enriched_snapshot"}
TARGET_TYPES = {"ANALYSIS_TRIGGER","AI_ANALYSIS"}

def has_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return True
        return any(has_key(v, key) for v in obj.values())
    if isinstance(obj, list):
        return any(has_key(i, key) for i in obj)
    return False

def iter_events(path: Path):
    with path.open("r", encoding="utf-8") as f:
        first = f.readline()
        f.seek(0)
        if first.strip().startswith("{"):
            # JSONL
            for line in f:
                line=line.strip()
                if not line: 
                    continue
                yield json.loads(line)
        else:
            data = json.load(f)
            if isinstance(data, list):
                for e in data: 
                    yield e
            elif isinstance(data, dict):
                yield data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="dados/eventos_fluxo.jsonl")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    p = Path(args.file)
    if not p.exists():
        raise SystemExit(f"Arquivo não encontrado: {p}")

    sizes = defaultdict(list)
    forbidden_counts = defaultdict(int)
    counts = defaultdict(int)

    for evt in iter_events(p):
        t = evt.get("tipo_evento")
        if t not in TARGET_TYPES:
            continue

        s = json.dumps(evt, separators=(",",":")).encode("utf-8")
        sizes[t].append(len(s))
        counts[t] += 1

        for k in FORBIDDEN:
            if has_key(evt, k):
                forbidden_counts[k] += 1

    print("DIAGNOSTICO")
    print("="*60)
    total = sum(counts.values())
    print(f"Arquivo: {p}")
    print(f"Eventos analisados: {total}")
    for t in ["ANALYSIS_TRIGGER","AI_ANALYSIS"]:
        if counts[t]:
            avg = sum(sizes[t]) / len(sizes[t])
            print(f"- {t}: {counts[t]} | avg_bytes={avg:,.0f} | min={min(sizes[t]):,} | max={max(sizes[t]):,}")

    print("\nCampos proibidos (presenca em eventos analisados):")
    for k,v in sorted(forbidden_counts.items(), key=lambda x: -x[1]):
        if v:
            print(f"- {k}: {v}x")

    if args.verbose:
        print("\nRecomendacao rapida:")
        if counts["ANALYSIS_TRIGGER"] and (sum(sizes["ANALYSIS_TRIGGER"])/len(sizes["ANALYSIS_TRIGGER"])) > 5000:
            print("- ANALYSIS_TRIGGER ainda grande: otimize antes de logar/salvar OU nao logue o evento completo.")
        if forbidden_counts["observability"]:
            print("- Remover observability do que for salvo/logado.")

if __name__ == "__main__":
    main()
