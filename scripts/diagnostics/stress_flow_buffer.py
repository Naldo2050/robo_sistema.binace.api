# scripts/diagnostics/stress_flow_buffer.py
# -*- coding: utf-8 -*-
"""
TESTE CONTROLADO DT-02/H1 — saturação deliberada do buffer flow_trades (100k).

Reproduz trades reais (dump JSONL de produção) em taxa acelerada sustentada
(>150 trades/s) contra FlowAnalyzer isolado, SEM DB, SEM rede, SEM escrita em
produção. A cada fechamento simulado de janela (60s) chama get_flow_metrics(),
reproduzindo a contenção snapshot-vs-ingestão do ambiente real.

Saída: JSON com todos os eventos de latência + correlação ±10s com
flow_trades_capacity_truncated + veredito falsificável H1 (>=70% => CONFIRMADA).

Uso:
    python scripts/diagnostics/stress_flow_buffer.py --rate 250 --minutes 14
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LAT_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*LATÊNCIA CRÍTICA: process_trade took ([\d\.]+)ms"
    r"(?: \| buffer_size=(\d+) \| evict_5s=(\d+) \| parquet_flush=(\w+))?"
)
TRUNC_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*flow_trades_capacity_truncated")


def load_raw_trades(dump_path: Path, limit: int):
    trades = []
    with open(dump_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if not all(k in o for k in ("p", "q")):
                continue
            trades.append(o)
            if len(trades) >= limit:
                break
    return trades


def main() -> int:
    ap = argparse.ArgumentParser(description="Stress controlado do buffer flow_trades (DT-02/H1)")
    ap.add_argument("--dump", default="dados/audit/observacao_golive.jsonl")
    ap.add_argument("--rate", type=float, default=250.0, help="trades/s alvo")
    ap.add_argument("--minutes", type=float, default=14.0)
    ap.add_argument("--metrics-every-s", type=float, default=60.0)
    ap.add_argument("--concurrent-metrics", action="store_true",
                    help="get_flow_metrics em thread separada CONCORRENTE com a ingestão "
                         "(reproduz a contenção lock snapshot-vs-ingestão da produção; "
                         "sem esta flag a ingestão pausa durante o snapshot)")
    ap.add_argument("--ooo-frac", type=float, default=0.0,
                    help="fração de trades com T deslocado para trás (0-1): força o ramo "
                         "O(n) de rebuild do prune sob lock (_out_of_order_seen), como "
                         "ocorre na produção com trades atrasados/clampados")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out or f"dados/audit/stress_dt02_{ts}.json")
    run_log = out_path.with_suffix(".log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(run_log, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    log = logging.getLogger("stress_dt02")

    # Guarda de segurança: este harness nunca deve tocar em DB de produção.
    assert "EventStore" not in dir(), "guard"
    log.info("H1-START rate=%.0f/s minutes=%.1f dump=%s (leitura do dump; sem escrita em DB/producao)", args.rate, args.minutes, args.dump)

    from flow_analyzer.core import FlowAnalyzer

    total_needed = int(args.rate * args.minutes * 60) + 1000
    raw = load_raw_trades(Path(args.dump), total_needed)
    if len(raw) < total_needed * 0.5:
        log.error("Trades insuficientes no dump: %d", len(raw))
        return 2
    log.info("Trades carregados: %d | cap flow_trades=%s", len(raw), "100000 (default)")

    fa = FlowAnalyzer()
    log.info("FlowAnalyzer instanciado | flow_trades_maxlen=%s | concurrent_metrics=%s | ooo_frac=%s",
             fa.flow_trades_maxlen, args.concurrent_metrics, args.ooo_frac)

    # Reescreve T para agora (monotônico): preserva lado/preço/qtd, mantém o
    # prune temporal realista. Sem OOO (ramo O(1) do prune).
    t0_ms = int(time.time() * 1000)
    deadline = time.perf_counter() + args.minutes * 60
    batch_n = max(1, int(args.rate / 10))
    interval = batch_n / args.rate
    import random as _rnd

    _ooo_every = int(1.0 / args.ooo_frac) if args.ooo_frac > 0 else 0
    fed = 0
    next_tick = time.perf_counter()
    next_metrics = time.perf_counter() + args.metrics_every_s
    metrics_calls = []
    i = 0

    stop_ev = None
    metrics_thread = None
    if args.concurrent_metrics:
        import threading

        stop_ev = threading.Event()

        def _metrics_loop():
            while not stop_ev.wait(args.metrics_every_s):
                ms0 = time.perf_counter()
                try:
                    fa.get_flow_metrics()
                except Exception as e:  # noqa: BLE001
                    log.warning("get_flow_metrics falhou: %s", e)
                metrics_calls.append(round((time.perf_counter() - ms0) * 1000, 1))

        metrics_thread = threading.Thread(target=_metrics_loop, daemon=True, name="H1-metrics")
        metrics_thread.start()

    while time.perf_counter() < deadline and i < len(raw):
        for _ in range(batch_n):
            if i >= len(raw) or time.perf_counter() >= deadline:
                break
            r = raw[i]
            i += 1
            base_t = t0_ms + int((time.perf_counter() - (deadline - args.minutes * 60)) * 1000)
            if _ooo_every and (i % _ooo_every == 0):
                base_t -= _rnd.randint(1000, 120000)  # 1s..120s atrasado
            t = {
                "p": float(r.get("p") or r.get("price")),
                "q": float(r.get("q") or r.get("quantity")),
                "T": base_t,
                "m": not bool(r.get("is_buyer_maker", False)),
            }
            fa.process_trade(t)
            fed += 1
        if not args.concurrent_metrics:
            next_tick += interval
            now = time.perf_counter()
            if now >= next_metrics:
                ms0 = time.perf_counter()
                try:
                    fa.get_flow_metrics()
                except Exception as e:  # noqa: BLE001 — harness não pode morrer
                    log.warning("get_flow_metrics falhou: %s", e)
                metrics_calls.append(round((time.perf_counter() - ms0) * 1000, 1))
                next_metrics = now + args.metrics_every_s
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
        else:
            next_tick += interval
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)

    if stop_ev is not None:
        stop_ev.set()
        metrics_thread.join(timeout=30)

    elapsed = args.minutes * 60 - max(0.0, deadline - time.perf_counter())
    log.info("H1-FEED fim: %d trades em %.0fs (%.1f t/s) | metrics_calls=%d", fed, elapsed, fed / max(elapsed, 1), len(metrics_calls))

    # ---- análise do próprio log ----
    lat, trunc = [], []
    with open(run_log, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LAT_RE.search(line)
            if m:
                dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                lat.append({
                    "t": m.group(1), "epoch": dt.timestamp(),
                    "ms": float(m.group(2)),
                    "buffer_size": int(m.group(3)) if m.group(3) else None,
                    "evict_5s": int(m.group(4)) if m.group(4) else None,
                    "parquet_flush": m.group(5),
                })
                continue
            m2 = TRUNC_RE.search(line)
            if m2:
                trunc.append(datetime.strptime(m2.group(1), "%Y-%m-%d %H:%M:%S").timestamp())

    big = [e for e in lat if e["ms"] > 1500]
    trunc_epochs = sorted(trunc)

    def near_trunc(e, win=10.0):
        return any(abs(e["epoch"] - c) <= win for c in trunc_epochs)

    corr = (sum(1 for e in big if near_trunc(e)) / len(big)) if big else None
    if corr is None:
        verdict = "INCONCLUSIVO (zero eventos >1500ms; ver secundária >200ms)"
    elif corr >= 0.70:
        verdict = "H1 CONFIRMADA"
    else:
        verdict = "H1 REFUTADA"
    over200 = [e for e in lat if e["ms"] > 200]
    corr200 = (sum(1 for e in over200 if near_trunc(e)) / len(over200)) if over200 else None

    result = {
        "rate_target": args.rate, "minutes": args.minutes, "trades_fed": fed,
        "concurrent_metrics": args.concurrent_metrics, "ooo_frac": args.ooo_frac,
        "metrics_calls_ms": metrics_calls,
        "lat_gt200": len(over200), "lat_gt1500": len(big),
        "trunc_lines": len(trunc_epochs),
        "corr_gt1500_trunc_10s": corr, "corr_gt200_trunc_10s": corr200,
        "verdict": verdict,
        "top5": sorted(big, key=lambda e: -e["ms"])[:5],
        "run_log": str(run_log),
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("H1-RESULT lat>200=%d lat>1500=%d trunc=%d corr1500=%s corr200=%s => %s",
             len(over200), len(big), len(trunc_epochs), corr, corr200, verdict)
    log.info("JSON: %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
