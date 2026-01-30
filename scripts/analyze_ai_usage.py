from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _mean(values: List[float]) -> Optional[float]:
    return statistics.mean(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audita uso/qualidade das decisoes da IA a partir de eventos AI_ANALYSIS (json/jsonl)."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="dados/eventos_fluxo.jsonl",
        help="Arquivo .json (lista/objeto) ou .jsonl (1 evento por linha).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Processa apenas os N primeiros eventos (0 = todos).")
    parser.add_argument("--out", default="", help="Se definido, salva um resumo em JSON neste caminho.")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"Arquivo nao encontrado: {path}")

    events = _read_events(path)
    if args.limit and args.limit > 0:
        events = events[: args.limit]

    ai_events = [e for e in events if e.get("tipo_evento") == "AI_ANALYSIS" and isinstance(e.get("ai_payload"), dict)]
    if not ai_events:
        print("Nenhum evento AI_ANALYSIS encontrado.")
        return 0

    action_counts = Counter()
    sentiment_counts = Counter()
    confidences: List[float] = []

    dist_poc_by_action: defaultdict[str, List[float]] = defaultdict(list)
    dist_vah_by_action: defaultdict[str, List[float]] = defaultdict(list)
    dist_val_by_action: defaultdict[str, List[float]] = defaultdict(list)

    hvn_present = 0
    lvn_present = 0

    for ev in ai_events:
        ai_result = ev.get("ai_result") if isinstance(ev.get("ai_result"), dict) else {}
        action = ai_result.get("action") or "unknown"
        sentiment = ai_result.get("sentiment") or "unknown"
        conf = _safe_float(ai_result.get("confidence"))

        action_counts[str(action)] += 1
        sentiment_counts[str(sentiment)] += 1
        if conf is not None:
            confidences.append(conf)

        payload = ev.get("ai_payload") if isinstance(ev.get("ai_payload"), dict) else {}
        price_ctx = payload.get("price_context") if isinstance(payload.get("price_context"), dict) else {}
        cur_price = _safe_float(price_ctx.get("current_price"))

        vp = price_ctx.get("volume_profile_daily") if isinstance(price_ctx.get("volume_profile_daily"), dict) else {}
        poc = _safe_float(vp.get("poc"))
        vah = _safe_float(vp.get("vah"))
        val = _safe_float(vp.get("val"))

        if isinstance(vp.get("hvns_nearby"), list) and vp.get("hvns_nearby"):
            hvn_present += 1
        if isinstance(vp.get("lvns_nearby"), list) and vp.get("lvns_nearby"):
            lvn_present += 1

        if cur_price and cur_price > 0:
            if poc is not None:
                dist_poc_by_action[str(action)].append(abs(cur_price - poc) / cur_price)
            if vah is not None:
                dist_vah_by_action[str(action)].append(abs(cur_price - vah) / cur_price)
            if val is not None:
                dist_val_by_action[str(action)].append(abs(cur_price - val) / cur_price)

    total = len(ai_events)

    print("=== AUDITORIA DE USO/QUALIDADE (AI_ANALYSIS) ===")
    print(f"Arquivo: {path}")
    print(f"Eventos AI_ANALYSIS: {total}")

    print("\nAcoes:")
    for act, c in action_counts.most_common():
        print(f"- {act}: {c} ({c/total:.1%})")

    print("\nSentimentos:")
    for s, c in sentiment_counts.most_common():
        print(f"- {s}: {c} ({c/total:.1%})")

    if confidences:
        print(f"\nConfianca: media={statistics.mean(confidences):.4f} min={min(confidences):.4f} max={max(confidences):.4f}")

    def _print_dist(title: str, by_action: defaultdict[str, List[float]]) -> None:
        print(f"\n{title} (abs(preco-nivel)/preco):")
        for act, vals in sorted(by_action.items(), key=lambda kv: -len(kv[1])):
            if not vals:
                continue
            print(f"- {act}: n={len(vals)} media={statistics.mean(vals):.2%} p95={sorted(vals)[int(0.95*(len(vals)-1))]:.2%}")

    _print_dist("Distancia ao POC", dist_poc_by_action)
    _print_dist("Distancia ao VAH", dist_vah_by_action)
    _print_dist("Distancia ao VAL", dist_val_by_action)

    print(f"\nVP extras presentes: hvns_nearby={hvn_present}/{total} ({hvn_present/total:.1%}) | lvns_nearby={lvn_present}/{total} ({lvn_present/total:.1%})")

    summary: Dict[str, Any] = {
        "file": str(path),
        "total_ai_analysis": total,
        "actions": dict(action_counts),
        "sentiments": dict(sentiment_counts),
        "avg_confidence": _mean(confidences),
        "vp_nearby_presence": {
            "hvns_nearby_rate": hvn_present / total if total else 0.0,
            "lvns_nearby_rate": lvn_present / total if total else 0.0,
        },
        "avg_dist_pct": {
            "poc": {k: _mean(v) for k, v in dist_poc_by_action.items()},
            "vah": {k: _mean(v) for k, v in dist_vah_by_action.items()},
            "val": {k: _mean(v) for k, v in dist_val_by_action.items()},
        },
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nResumo salvo em: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

