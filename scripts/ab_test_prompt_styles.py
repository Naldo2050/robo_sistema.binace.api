from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ai_analyzer_qwen as mod
from ai_payload_optimizer import AIPayloadOptimizer


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


def _percentile(values: List[int], pct: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    idx = int(round((pct / 100) * (len(values) - 1)))
    idx = min(max(idx, 0), len(values) - 1)
    return float(values[idx])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compara custo (tokens estimados) entre prompt legacy vs compact (sem chamadas a LLM)."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="dados/eventos_fluxo.jsonl",
        help="Arquivo .json (lista/objeto) ou .jsonl (1 evento por linha).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Processa apenas os N primeiros eventos (0 = todos).")
    parser.add_argument("--sample", action="store_true", help="Imprime 1 exemplo de prompt de cada estilo.")
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

    # Evita inicializacao de provedores externos ao instanciar AIAnalyzer
    def _fake_init(self):
        self.mode = None
        self.enabled = True
        self.client = None
        self.client_async = None

    mod.AIAnalyzer._initialize_api = _fake_init  # type: ignore[method-assign]
    analyzer = mod.AIAnalyzer(health_monitor=None)

    legacy_tokens: List[int] = []
    compact_tokens: List[int] = []

    legacy_sys = mod.SYSTEM_PROMPT_LEGACY
    compact_sys = mod.SYSTEM_PROMPT

    sample_payload: Optional[Dict[str, Any]] = None

    for ev in ai_events:
        payload = ev.get("ai_payload")
        if not isinstance(payload, dict):
            continue
        sample_payload = sample_payload or payload

        legacy_user = analyzer._build_structured_prompt_legacy(payload)
        compact_user = analyzer._build_structured_prompt(payload)

        legacy_total = AIPayloadOptimizer._estimate_tokens(legacy_sys) + AIPayloadOptimizer._estimate_tokens(legacy_user)
        compact_total = AIPayloadOptimizer._estimate_tokens(compact_sys) + AIPayloadOptimizer._estimate_tokens(compact_user)

        legacy_tokens.append(int(legacy_total))
        compact_tokens.append(int(compact_total))

    if not legacy_tokens or not compact_tokens:
        print("Nao foi possivel estimar tokens (lista vazia).")
        return 0

    def _summ(name: str, vals: List[int]) -> None:
        p50 = _percentile(vals, 50)
        p95 = _percentile(vals, 95)
        print(
            f"{name}: n={len(vals)} min={min(vals)} p50={p50:.0f} p95={p95:.0f} max={max(vals)} avg={statistics.mean(vals):.0f}"
        )

    print("=== A/B (PROMPT STYLES) - TOKENS ESTIMADOS ===")
    print(f"Arquivo: {path}")
    print(f"Eventos AI_ANALYSIS: {len(legacy_tokens)}")
    _summ("legacy_total", legacy_tokens)
    _summ("compact_total", compact_tokens)
    avg_legacy = statistics.mean(legacy_tokens)
    avg_compact = statistics.mean(compact_tokens)
    if avg_legacy > 0:
        print(f"\nReducao media: {(1 - (avg_compact / avg_legacy)) * 100:.1f}%")

    if args.sample and sample_payload is not None:
        print("\n--- SAMPLE (legacy user prompt) ---")
        print(analyzer._build_structured_prompt_legacy(sample_payload))
        print("\n--- SAMPLE (compact user prompt) ---")
        print(analyzer._build_structured_prompt(sample_payload))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
