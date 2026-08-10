import argparse, json
from pathlib import Path

FIELDS_REMOVE = {
    "observability","processing_times_ms","circuit_breaker","memory",
    "data_quality","valid_rate_pct","flow_trades_count","processing_time_ms",
    "metadata","burst_window_ms","last_reset_ms","config_version",
    "ui_sum_ok","invariants_ok","features_window_id","_log_id",
    "enriched_snapshot","volume_nodes","single_prints",
}
VP_REMOVE = {"single_prints","volume_nodes"}  # hvns/lvns ainda ficam; opcional remover também

def iter_events(path: Path):
    with path.open("r", encoding="utf-8") as f:
        first = f.readline()
        f.seek(0)
        if first.strip().startswith("{"):
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

def recursive_remove(obj):
    if isinstance(obj, dict):
        out = {}
        for k,v in obj.items():
            if k in FIELDS_REMOVE:
                continue
            out[k] = recursive_remove(v)
        return out
    if isinstance(obj, list):
        return [recursive_remove(x) for x in obj]
    return obj

def clean_event(evt):
    if not isinstance(evt, dict):
        return evt
    return recursive_remove(evt)

def get_in(d, path):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

def set_in(d, path, value):
    cur = d
    for k in path[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[path[-1]] = value

def simplify_historical_vp(evt):
    # tenta achar historical_vp em locais comuns
    locations = [
        ["historical_vp"],
        ["raw_event","historical_vp"],
        ["contextual_snapshot","historical_vp"],
        ["raw_event","raw_event","historical_vp"],
    ]

    for loc in locations:
        vp = get_in(evt, loc)
        if isinstance(vp, dict):
            new_vp = {}
            for tf in ["daily", "weekly", "monthly"]:
                tf_vp = vp.get(tf)
                if not isinstance(tf_vp, dict):
                    continue

                for k in VP_REMOVE:
                    tf_vp.pop(k, None)

                if tf == "daily":
                    slim = {
                        "poc": tf_vp.get("poc"),
                        "vah": tf_vp.get("vah"),
                        "val": tf_vp.get("val"),
                        "hvns_nearby": tf_vp.get("hvns_nearby"),
                        "lvns_nearby": tf_vp.get("lvns_nearby"),
                    }
                    if "status" in tf_vp:
                        slim["status"] = tf_vp.get("status")
                else:
                    slim = {
                        "poc": tf_vp.get("poc"),
                        "vah": tf_vp.get("vah"),
                        "val": tf_vp.get("val"),
                    }
                    if "status" in tf_vp:
                        slim["status"] = tf_vp.get("status")

                slim = {k: v for k, v in slim.items() if v is not None}
                new_vp[tf] = slim
            set_in(evt, loc, new_vp)
    return evt

def strip_profile_bins(evt):
    """
    ETAPA 4: remove price_bins/volume_per_bin do historical_vp
    (daily/weekly/monthly) na representação PERSISTIDA do evento.

    Os bins são necessários apenas internamente
    (historical_profiler -> institutional_analytics -> value_area_volume_pct);
    persisti-los inflaria o payload em ~100 KB/evento sem uso downstream.

    Não muta o evento original: retorna cópia profunda SOMENTE quando há bins
    a remover; caso contrário retorna o mesmo objeto (custo zero).
    """
    if not isinstance(evt, dict):
        return evt

    locations = [
        ["historical_vp"],
        ["raw_event", "historical_vp"],
        ["contextual_snapshot", "historical_vp"],
        ["raw_event", "raw_event", "historical_vp"],
    ]

    def _has_bins(vp):
        if not isinstance(vp, dict):
            return False
        for tf in ("daily", "weekly", "monthly"):
            tf_vp = vp.get(tf)
            if isinstance(tf_vp, dict) and ("price_bins" in tf_vp or "volume_per_bin" in tf_vp):
                return True
        return False

    if not any(_has_bins(get_in(evt, loc)) for loc in locations):
        return evt

    import copy
    out = copy.deepcopy(evt)
    for loc in locations:
        vp = get_in(out, loc)
        if not isinstance(vp, dict):
            continue
        for tf in ("daily", "weekly", "monthly"):
            tf_vp = vp.get(tf)
            if isinstance(tf_vp, dict):
                tf_vp.pop("price_bins", None)
                tf_vp.pop("volume_per_bin", None)
    return out

def remove_enriched_snapshot(evt):
    def pop_key(obj, key: str):
        if isinstance(obj, dict):
            obj.pop(key, None)
            for v in obj.values():
                pop_key(v, key)
        elif isinstance(obj, list):
            for v in obj:
                pop_key(v, key)

    if isinstance(evt, dict):
        pop_key(evt, "enriched_snapshot")
    return evt

def optimize(evt):
    if evt.get("tipo_evento") != "ANALYSIS_TRIGGER":
        return evt
    evt = recursive_remove(evt)
    evt = simplify_historical_vp(evt)
    evt = remove_enriched_snapshot(evt)
    return evt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="dados/eventos_fluxo.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    p = Path(args.file)
    if not p.exists():
        raise SystemExit(f"Arquivo não encontrado: {p}")

    events = list(iter_events(p))
    before = sum(len(json.dumps(e, separators=(",",":")).encode("utf-8")) for e in events)
    opt_events = [optimize(dict(e)) if isinstance(e, dict) else e for e in events]
    after = sum(len(json.dumps(e, separators=(",",":")).encode("utf-8")) for e in opt_events)

    red = (before - after) / before * 100 if before else 0
    print(f"Antes:  {before:,} bytes")
    print(f"Depois: {after:,} bytes")
    print(f"Reducao: {before-after:,} bytes ({red:.1f}%)")

    if args.dry_run:
        print("DRY-RUN: nenhum arquivo alterado.")
        return

    out = Path(args.output) if args.output else p
    with out.open("w", encoding="utf-8") as f:
        for e in opt_events:
            f.write(json.dumps(e, separators=(",",":")) + "\n")
    print(f"Salvo em: {out}")

if __name__ == "__main__":
    main()
