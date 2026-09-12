# tools/audit_shadow.py
# -*- coding: utf-8 -*-
"""Shadow calculator independente — NÃO importa lógica de produção.

PROIBIDO importar (verificação em runtime):
  flow_analyzer, data_pipeline, market_orchestrator, institutional, orderbook_analyzer

Permitido: stdlib + json + decimal + datetime + hashlib + argparse.
Entrada: dados/audit/live_<run>/raw_aggtrades.jsonl (RAW Binance, campos a/p/q/T/m).
Regra BUY/SELL (seção 8):
  m=true  -> aggressive SELL
  m=false -> aggressive BUY
  m=None  -> UNKNOWN (não conta como BUY; registra separado)

Calcula por janela tumbling (epoch-60000, epoch] + rolling 1m/5m/15m,
compara com produção (windows.jsonl / analysis_triggers) sem corrigir nada.
"""
from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal, getcontext
from typing import Any, Dict, List

getcontext().prec = 28

FORBIDDEN = ("flow_analyzer", "data_pipeline", "market_orchestrator", "institutional", "orderbook_analyzer")


def assert_no_forbidden_imports() -> None:
    bad = [m for m in sys.modules if any(m == f or m.startswith(f + ".") for f in FORBIDDEN)]
    if bad:
        raise RuntimeError(f"SHADOW VIOLATION: módulos proibidos importados: {bad}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shadow calculator forense (stdlib only)")
    p.add_argument("--input", required=True, help="raw_aggtrades.jsonl")
    p.add_argument("--windows", required=False, default=None, help="windows.jsonl produção (opcional)")
    p.add_argument("--triggers", required=False, default=None, help="analysis_triggers.jsonl (opcional)")
    p.add_argument("--out", required=True, help="divergences.jsonl saída")
    p.add_argument("--epoch-ms", required=False, default=None, help="epoch alvo único (opcional)")
    return p.parse_args()


def load_raw(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def classify(m: Any) -> str:
    if m is True:
        return "SELL"
    if m is False:
        return "BUY"
    return "UNKNOWN"


def calc_window(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(trades)
    if n == 0:
        return {"trade_count": 0}
    prices = [Decimal(str(t["p"])) for t in trades]
    qtys = [Decimal(str(t["q"])) for t in trades]
    o = float(prices[0])
    h = float(max(prices))
    low = float(min(prices))
    c = float(prices[-1])
    buy_b = Decimal("0")
    sell_b = Decimal("0")
    buy_q = Decimal("0")
    sell_q = Decimal("0")
    unknown_b = Decimal("0")
    for t in trades:
        side = classify(t.get("m"))
        q = Decimal(str(t["q"]))
        pq = Decimal(str(t["p"])) * q
        if side == "BUY":
            buy_b += q
            buy_q += pq
        elif side == "SELL":
            sell_b += q
            sell_q += pq
        else:
            unknown_b += q
    total_b = buy_b + sell_b
    delta_b = buy_b - sell_b
    total_q = buy_q + sell_q
    delta_q = buy_q - sell_q
    if total_b > 0:
        imb = float(delta_b / total_b)
        buy_pct = float(buy_b / total_b * 100)
        sell_pct = float(sell_b / total_b * 100)
        vwap = float(total_q / total_b)
    else:
        imb, buy_pct, sell_pct, vwap = 0.0, 0.0, 0.0, 0.0
    ratio = float(buy_b / sell_b) if sell_b > 0 else (float("inf") if buy_b > 0 else 0.0)
    avg = float(total_b / n) if n else 0.0
    tmin = min(int(t["T"]) for t in trades)
    tmax = max(int(t["T"]) for t in trades)
    dur = (tmax - tmin) / 1000.0
    tps = (n / dur) if dur > 0 else 0.0
    return {
        "trade_count": n,
        "open": o, "high": h, "low": low, "close": c,
        "buy_base": float(buy_b), "sell_base": float(sell_b),
        "total_base": float(total_b), "delta_base": float(delta_b),
        "buy_quote": float(buy_q), "sell_quote": float(sell_q),
        "total_quote": float(total_q), "delta_quote": float(delta_q),
        "unknown_base": float(unknown_b),
        "buy_pct": buy_pct, "sell_pct": sell_pct,
        "buy_sell_ratio": ratio, "trade_flow_imbalance": imb,
        "VWAP": vwap, "avg_trade_size": avg, "trades_per_second": tps,
        "first_T": tmin, "last_T": tmax,
    }


def main() -> int:
    assert_no_forbidden_imports()
    args = parse_args()
    raw = load_raw(args.input)
    # Filtra só registros com a/p/q/T válidos (m pode ser None -> UNKNOWN)
    valid = []
    for r in raw:
        try:
            if r.get("p") is None or r.get("q") is None or r.get("T") is None:
                continue
            float(r["p"]); float(r["q"]); int(r["T"])
            valid.append(r)
        except Exception:
            continue
    valid.sort(key=lambda x: int(x["T"]))
    out_lines: List[Dict[str, Any]] = []
    # Se epoch único: janela (epoch-60k, epoch]
    if args.epoch_ms is not None:
        epoch = int(args.epoch_ms)
        w = [t for t in valid if (epoch - 60000) < int(t["T"]) <= epoch]
        out_lines.append({"epoch_ms": epoch, "shadow": calc_window(w)})
    else:
        # Deriva epochs por minuto a partir dos dados (tumbling UTC)
        if valid:
            t0 = (int(valid[0]["T"]) // 60000) * 60000
            t1 = (int(valid[-1]["T"]) // 60000) * 60000
            epoch = t0 + 60000
            while epoch <= t1 + 60000:
                w = [t for t in valid if (epoch - 60000) < int(t["T"]) <= epoch]
                if w:
                    out_lines.append({"epoch_ms": epoch, "shadow": calc_window(w)})
                    # Rolling 5m/15m + imbalance correto [-1,1]
                    for xm in (1, 5, 15):
                        wr = [t for t in valid if (epoch - xm * 60000) < int(t["T"]) <= epoch]
                        c = calc_window(wr)
                        out_lines[-1][f"shadow_{xm}m"] = {
                            "buy": c.get("buy_base"), "sell": c.get("sell_base"),
                            "total": c.get("total_base"), "delta": c.get("delta_base"),
                            "imbalance_correct": c.get("trade_flow_imbalance"),
                        }
                epoch += 60000
    # Compara com produção se fornecida (sem corrigir)
    prod_by_epoch: Dict[int, Any] = {}
    if args.windows:
        try:
            with open(args.windows, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        j = json.loads(line)
                        ep = int(j.get("window_end_ms", j.get("epoch_ms", 0)))
                        prod_by_epoch[ep] = j
                    except Exception:
                        continue
        except FileNotFoundError:
            pass
    with open(args.out, "w", encoding="utf-8") as out:
        for rec in out_lines:
            ep = rec["epoch_ms"]
            prod = prod_by_epoch.get(ep)
            if prod is not None:
                rec["production_ref"] = {
                    "normalized_trade_count": prod.get("normalized_trade_count"),
                    "open": prod.get("open"), "high": prod.get("high"),
                    "low": prod.get("low"), "close": prod.get("close"),
                }
            out.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    print(f"SHADOW OK: raw={len(raw)} valid={len(valid)} windows={len(out_lines)} out={args.out}")
    assert_no_forbidden_imports()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
