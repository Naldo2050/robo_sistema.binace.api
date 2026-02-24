from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.ai_payload_optimizer import AIPayloadOptimizer


def _json_bytes(obj: Any) -> int:
    try:
        return len(json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    except Exception:
        return 0


def _percentile(values: List[int], pct: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    idx = int(round((pct / 100) * (len(values) - 1)))
    idx = min(max(idx, 0), len(values) - 1)
    return float(values[idx])


def _read_events(path: Path) -> List[Dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    stripped = raw.lstrip()
    if not stripped:
        return []

    if path.suffix.lower() == ".jsonl":
        events: List[Dict[str, Any]] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                events.append(obj)
        return events

    first = stripped[0]
    if first in "[{":
        loaded = json.loads(raw)
        if isinstance(loaded, list):
            return [x for x in loaded if isinstance(x, dict)]
        if isinstance(loaded, dict):
            return [loaded]
        return []

    # fallback: tenta como JSONL
    events = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events


def _summarize_sizes(name: str, values: List[int]) -> None:
    if not values:
        print(f"{name}: (vazio)")
        return
    p50 = _percentile(values, 50)
    p90 = _percentile(values, 90)
    p95 = _percentile(values, 95)
    print(
        f"{name}: n={len(values)} min={min(values)} p50={p50:.0f} p90={p90:.0f} p95={p95:.0f} max={max(values)} avg={statistics.mean(values):.0f}"
    )


def _export_optimized(events: Iterable[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        for ev in events:
            optimized = AIPayloadOptimizer.optimize(ev)
            record = {
                "tipo_evento": ev.get("tipo_evento"),
                "resultado_da_batalha": ev.get("resultado_da_batalha"),
                "descricao": ev.get("descricao"),
                "symbol": ev.get("symbol") or ev.get("ativo") or optimized.get("symbol"),
                "epoch_ms": ev.get("epoch_ms") or ev.get("timestamp_ms") or optimized.get("ts"),
                "ai_payload": optimized,
            }
            fp.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audita tamanho/estima tokens de eventos JSON/JSONL e sugere onde estao os maiores custos."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="dados/eventos-fluxo.json",
        help="Arquivo .json (lista/objeto) ou .jsonl (1 evento por linha).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Processa apenas os N primeiros eventos (0 = todos).")
    parser.add_argument("--top", type=int, default=10, help="Mostra os N maiores eventos.")
    parser.add_argument(
        "--export-jsonl",
        default="",
        help="Se definido, exporta eventos compactados para este caminho (.jsonl).",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"Arquivo nao encontrado: {path}")

    events = _read_events(path)
    if args.limit and args.limit > 0:
        events = events[: args.limit]

    print(f"arquivo: {path}")
    print(f"eventos: {len(events)}")

    total_sizes: List[int] = []
    raw_sizes: List[int] = []
    ai_payload_sizes: List[int] = []
    optimized_sizes: List[int] = []
    raw_share: List[float] = []

    top_level_keys = Counter()
    raw_top_keys = Counter()
    raw_subkey_sizes: defaultdict[str, List[int]] = defaultdict(list)

    ranked: List[Tuple[int, int, str]] = []

    for idx, ev in enumerate(events):
        total_b = _json_bytes(ev)
        total_sizes.append(total_b)
        ranked.append((total_b, idx, str(ev.get("tipo_evento") or "N/A")))
        top_level_keys.update(ev.keys())

        raw = ev.get("raw_event")
        if isinstance(raw, dict):
            raw_b = _json_bytes(raw)
            raw_sizes.append(raw_b)
            if total_b:
                raw_share.append(raw_b / total_b)
            raw_top_keys.update(raw.keys())
            for k, v in raw.items():
                raw_subkey_sizes[k].append(_json_bytes(v))

            optimized = AIPayloadOptimizer.optimize(ev)
            optimized_sizes.append(_json_bytes(optimized))

        ai_payload = ev.get("ai_payload")
        if isinstance(ai_payload, dict):
            ai_payload_sizes.append(_json_bytes(ai_payload))

    print("\n**Tamanhos (bytes, JSON minificado)**")
    _summarize_sizes("evento_total", total_sizes)
    _summarize_sizes("raw_event", raw_sizes)
    _summarize_sizes("ai_payload", ai_payload_sizes)
    _summarize_sizes("otimizado(AIPayloadOptimizer)", optimized_sizes)

    if raw_share:
        ratios_sorted = sorted(raw_share)
        p50 = statistics.median(ratios_sorted) * 100
        p95 = ratios_sorted[int(0.95 * (len(ratios_sorted) - 1))] * 100
        print(
            f"\nraw_event participacao: avg={statistics.mean(ratios_sorted)*100:.1f}% p50={p50:.1f}% p95={p95:.1f}%"
        )

    print("\n**Top-level keys (frequencia)**")
    for k, c in top_level_keys.most_common(25):
        print(f"- {k}: {c}")

    if raw_top_keys:
        print("\n**raw_event keys (frequencia)**")
        for k, c in raw_top_keys.most_common(25):
            print(f"- {k}: {c}")

    ranked.sort(reverse=True)
    print(f"\n**Maiores eventos (top {args.top})**")
    for b, idx, tipo in ranked[: max(0, int(args.top))]:
        print(f"- idx={idx} tipo={tipo} bytes={b}")

    if raw_subkey_sizes:
        print("\n**raw_event subkeys por tamanho medio (top 20)**")
        avg_rows: List[Tuple[float, str, int, int]] = []
        for k, vals in raw_subkey_sizes.items():
            if not vals:
                continue
            avg_rows.append((statistics.mean(vals), k, max(vals), len(vals)))
        avg_rows.sort(reverse=True)
        for avg, k, mx, n in avg_rows[:20]:
            print(f"- {k}: avg={avg:.0f} max={mx} n={n}")

    if args.export_jsonl:
        out_path = Path(args.export_jsonl)
        _export_optimized(events, out_path)
        print(f"\nexport: {out_path} (jsonl)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
