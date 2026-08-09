#!/usr/bin/env python3
"""
measure_out_of_order_trades.py — Medição da frequência real de trades com
timestamp fora de ordem (T < last_T) no pipeline do bot.

Replica EXATAMENTE a lógica de market_orchestrator/market_orchestrator.py:

  linha 654:  T = trade.get("T") or trade.get("E") or trade.get("tradeTime")
  linha 733:  last_T = self._last_trade_ts_ms
  linha 734:  if last_T is not None and T < last_T:   -> clamp (linha 741: T = last_T)
  linha 743:  else: self._last_trade_ts_ms = T

O script NÃO modifica código de produção. Dois modos:

  1. --db (padrão): inspeciona dados/trading_bot.db procurando trades brutos
     persistidos. O schema atual só contém 'events' (eventos de janela) e
     'signal_outcomes' — se não houver trades brutos, use o modo live.

  2. --live: conecta no MESMO stream do bot
     (wss://stream.binance.com:9443/ws/{symbol}@trade) e aplica a mesma
     normalização de T, contando:
       - total de trades e % que sofreriam clamp
       - distribuição do delta (last_T - T) em ms
       - streak de clamps consecutivos (sintoma de fluxos interleaved)
       - T duplicados (proxy de entrega duplicada / segunda conexão)
       - taxa de mensagens por segundo (2x a taxa típica => 2 conexões)

Uso:
    python scripts/diagnostics/measure_out_of_order_trades.py --db
    python scripts/diagnostics/measure_out_of_order_trades.py --live --duration 900 --out out.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter

if sys.stdout and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

DB_DEFAULT = os.path.join(REPO_DIR, "dados", "trading_bot.db")
STREAM_DEFAULT = "wss://stream.binance.com:9443/ws/{symbol}@trade"

DELTA_BUCKETS_MS = [
    ("<1ms", 1),
    ("1-10ms", 10),
    ("10-100ms", 100),
    ("100ms-1s", 1000),
    ("1-10s", 10_000),
    ("10-60s", 60_000),
]


def _extract_t(raw: dict):
    trade = raw.get("data", raw)
    p = trade.get("p") or trade.get("P") or trade.get("price")
    q = trade.get("q") or trade.get("Q") or trade.get("quantity")
    T = trade.get("T") or trade.get("E") or trade.get("tradeTime")
    if (p is None or q is None or T is None) and isinstance(trade.get("k"), dict):
        k = trade["k"]
        if p is None:
            p = k.get("c")
        if q is None:
            q = k.get("v")
        if T is None:
            T = k.get("T") or raw.get("E")
    return p, q, T


def _pct(values: list[int], p: float):
    import math

    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return round(s[f] * (c - k) + s[c] * (k - f), 3)


# ─────────────────────────────────────────────────────────────────────────────
# Modo 1: inspeção do banco de dados
# ─────────────────────────────────────────────────────────────────────────────

def inspect_db(db_path: str) -> dict:
    import sqlite3

    report = {"db_path": db_path, "raw_trades_encontrados": False, "tabelas": {}}
    con = sqlite3.connect(db_path)
    try:
        for (t,) in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall():
            try:
                cols = [r[1] for r in con.execute(f"PRAGMA table_info({t})")]
                cnt = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                report["tabelas"][t] = {"colunas": cols, "rows": cnt}
            except Exception as e:
                report["tabelas"][t] = {"erro": str(e)}
        if "events" in report["tabelas"]:
            rows = con.execute("SELECT payload FROM events").fetchall()
            com_t = sum(1 for (p,) in rows if p and '"T"' in p)
            report["events_com_campo_T"] = com_t
            report["total_events"] = len(rows)
            if com_t:
                report["raw_trades_encontrados"] = True
    finally:
        con.close()
    return report


def run_db_mode(args) -> int:
    rep = inspect_db(args.db)
    print(f"== DB: {rep['db_path']} ==")
    for t, info in rep["tabelas"].items():
        print(f"  tabela {t}: colunas={info.get('colunas')} rows={info.get('rows')}")
    if rep.get("events_com_campo_T") is not None:
        print(
            f"  eventos com campo 'T' no payload: "
            f"{rep['events_com_campo_T']}/{rep['total_events']}"
        )
    if rep["raw_trades_encontrados"]:
        print("RESULTADO: trades brutos ENCONTRADOS no DB.")
        print("IMPORTANTE: o DB só contém a visão do clamp; a medição precisa do")
        print("timestamp original (T pré-clamp), que não é persistido. Use --live.")
        return 0
    print("RESULTADO: trades brutos NÃO são persistidos neste DB.")
    print("Use o modo live para medir em tempo real:")
    print("  python scripts/diagnostics/measure_out_of_order_trades.py --live")
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# Modo 2: medição live no mesmo stream do bot
# ─────────────────────────────────────────────────────────────────────────────

class OutOfOrderMeasurer:
    def __init__(self):
        self.total = 0
        self.clamped = 0
        self.deltas_ms: list[int] = []
        self.streak = 0
        self.max_streak = 0
        self.dup_T_total = 0
        self.dup_T_distinct = 0
        self._seen_T: Counter = Counter()
        self._last_T = None
        self._last_raw_T = None
        self._max_forward_gap_ms = 0
        self._start = None
        self._last_report = 0

    def process(self, raw: dict) -> None:
        p, q, T = _extract_t(raw)
        if p is None or q is None or T is None:
            return
        T = int(T)
        if T <= 0:
            return

        self.total += 1
        self._seen_T[T] += 1

        last_T = self._last_T
        if last_T is not None and T < last_T:
            self.clamped += 1
            delta = last_T - T
            self.deltas_ms.append(delta)
            self.streak += 1
            self.max_streak = max(self.max_streak, self.streak)
            T = last_T
        else:
            if last_T is not None:
                gap = T - last_T
                if gap > self._max_forward_gap_ms:
                    self._max_forward_gap_ms = gap
            self.streak = 0
            self._last_T = T

    def progress(self) -> str:
        pct = 100.0 * self.clamped / self.total if self.total else 0.0
        return (
            f"trades={self.total} clamp={self.clamped} ({pct:.3f}%) "
            f"max_streak={self.max_streak} dups={self.dup_T_total}"
        )

    def finalize(self) -> dict:
        n = len(self.deltas_ms)
        pct = 100.0 * self.clamped / self.total if self.total else 0.0
        for t, c in self._seen_T.items():
            if c > 1:
                self.dup_T_total += c - 1
                self.dup_T_distinct += 1

        buckets = {}
        if n:
            lo = 0
            for label, lim in DELTA_BUCKETS_MS:
                cnt = sum(1 for d in self.deltas_ms if lo <= d < lim)
                buckets[label] = {"count": cnt, "pct": 100.0 * cnt / n}
                lo = lim
            over = sum(1 for d in self.deltas_ms if d >= lo)
            buckets[">60s"] = {"count": over, "pct": 100.0 * over / n}

        dist = {}
        if n:
            for p in (0.0, 25.0, 50.0, 75.0, 95.0, 99.0, 100.0):
                dist[str(p)] = _pct(self.deltas_ms, p)

        return {
            "total_trades": self.total,
            "clamped": self.clamped,
            "pct_afetados": round(pct, 4),
            "delta_ms": {
                "min": min(self.deltas_ms) if n else None,
                "p25": dist.get("25.0"),
                "mediana": dist.get("50.0"),
                "p75": dist.get("75.0"),
                "p95": dist.get("95.0"),
                "p99": dist.get("99.0"),
                "max": max(self.deltas_ms) if n else None,
            },
            "buckets_delta_ms": buckets,
            "max_streak_clamps_consecutivos": self.max_streak,
            "T_duplicados_total_mensagens": self.dup_T_total,
            "T_duplicados_distintos": self.dup_T_distinct,
            "max_gap_forward_entre_T_ms": self._max_forward_gap_ms,
            "taxa_msgs_por_seg": round(
                self.total / (time.time() - self._start), 2
            ) if self._start else 0.0,
        }


async def run_live(args) -> int:
    import aiohttp

    url = args.url or STREAM_DEFAULT.format(symbol=args.symbol.lower())
    m = OutOfOrderMeasurer()
    m._start = time.time()
    deadline = time.time() + args.duration
    conn_ok = False

    print(f"== LIVE: conectando em {url} por {args.duration}s ==")
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        ) as session:
            async with session.ws_connect(
                url, heartbeat=20, autoping=True
            ) as ws:
                conn_ok = True
                print("Conexão estabelecida. Medindo...")
                async for msg in ws:
                    if time.time() >= deadline:
                        break
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            raw = json.loads(msg.data)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(raw, dict):
                            m.process(raw)
                    now = time.time()
                    if now - m._last_report >= 15:
                        m._last_report = now
                        remaining = max(0, int(deadline - now))
                        print(f"  [{remaining:>4}s restantes] {m.progress()}")
    except Exception as e:
        print(f"ERRO na conexão: {e}", file=sys.stderr)
        if not conn_ok:
            return 2

    rep = m.finalize()
    rep["duracao_seg"] = round(time.time() - m._start, 1)
    rep["stream_url"] = url

    _print_report(rep)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        print(f"\nRelatório JSON salvo em: {args.out}")
    return 0


def _print_report(rep: dict) -> None:
    d = rep["delta_ms"]
    print("\n" + "=" * 60)
    print("RELATÓRIO — Trades com timestamp fora de ordem (clamp T<last_T)")
    print("=" * 60)
    print(f"Stream: {rep['stream_url']}")
    print(f"Duração: {rep['duracao_seg']:.0f}s | "
          f"trades recebidos: {rep['total_trades']}")
    print(f"% de trades afetados: {rep['clamped']} "
          f"({rep['pct_afetados']:.4f}%)")
    print(f"Delta (last_T - T) em ms: min={d['min']} p25={d['p25']} "
          f"mediana={d['mediana']} p75={d['p75']} p95={d['p95']} "
          f"p99={d['p99']} max={d['max']}")
    print("Buckets do delta:")
    for label, b in rep["buckets_delta_ms"].items():
        print(f"  {label:>10s}: {b['count']:>7d} ({b['pct']:.3f}%)")
    print(f"Streak máximo de clamps consecutivos: "
          f"{rep['max_streak_clamps_consecutivos']}")
    print(f"T duplicados: {rep['T_duplicados_total_mensagens']} mensagens "
          f"({rep['T_duplicados_distintos']} trades distintos)")
    print(f"Gap forward máximo entre T consecutivos: "
          f"{rep['max_gap_forward_entre_T_ms']} ms")
    print(f"Taxa: {rep['taxa_msgs_por_seg']} msg/s "
          f"(típico BTC spot @trade: ~2-6 msg/s)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mede a frequência real de trades fora de ordem (T<last_T)"
    )
    parser.add_argument("--db", default=DB_DEFAULT,
                        help="Caminho do trading_bot.db para inspeção")
    parser.add_argument("--live", action="store_true",
                        help="Mede ao vivo no stream @trade real")
    parser.add_argument("--duration", type=float, default=900.0,
                        help="Duração da medição em segundos (padrão 900 = 15 min)")
    parser.add_argument("--symbol", default="BTCUSDT",
                        help="Símbolo (padrão BTCUSDT)")
    parser.add_argument("--url", default=None,
                        help="URL do stream (padrão: mesma do bot)")
    parser.add_argument("--out", default=None,
                        help="Caminho para salvar o relatório JSON")
    args = parser.parse_args()

    if args.live:
        return asyncio.run(run_live(args))
    return run_db_mode(args)


if __name__ == "__main__":
    sys.exit(main())
