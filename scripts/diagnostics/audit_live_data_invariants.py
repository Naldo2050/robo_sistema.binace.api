# -*- coding: utf-8 -*-
"""
AUDITOR DE INVARIANTES - DADOS LIVE (data-readiness / shadow observation).

SCRIPT DIAGNOSTICO APENAS. NAO modifica codigo de producao. NAO corrige nada.

Entrada:
  - dados/trading_bot.db (SQLite, fonte canonica) e/ou dados/eventos_fluxo.jsonl
  - logs/shadow_run_meta.json (delimita janelas novas da observacao shadow)

Saida:
  - por invariante: PASS / FAIL / SKIP / NOT_APPLICABLE / NOT_COMPARABLE
    com window, field, expected, actual, difference, severity.
  - scorecard final (FASE D) + criterio GO/NO-GO.

Uso:
  python scripts/diagnostics/audit_live_data_invariants.py
  python scripts/diagnostics/audit_live_data_invariants.py --all
  python scripts/diagnostics/audit_live_data_invariants.py --start-ms 1786545800000 --end-ms 1786550400000
"""

import argparse
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

# =============================================================================
# SEVERIDADES (FASE C)
# =============================================================================
BLOCKER = "BLOCKER"
HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"

NAN_INF = ("nan", "inf", "-inf", "+inf", "-nan", "+nan")

EVENT_AT = "ANALYSIS_TRIGGER"
EVENT_AI = "AI_ANALYSIS"

RTOL = 0.01          # tolerancia relativa default (1%)
ABS_TOL = 1e-6


def _num(x):
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _fmt(x, nd=6):
    if x is None:
        return "None"
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def _rel_diff(a, b):
    a, b = _num(a), _num(b)
    if a is None or b is None:
        return None
    denom = max(abs(a), abs(b))
    if denom == 0:
        return 0.0 if a == b else float("inf")
    return abs(a - b) / denom


class Invariant:
    __slots__ = ("id", "group", "name", "severity", "critical_math", "fn", "details")

    def __init__(self, iid, group, name, fn, severity=HIGH, critical_math=False, details=""):
        self.id = iid
        self.group = group
        self.name = name
        self.fn = fn
        self.severity = severity
        self.critical_math = critical_math
        self.details = details


class Outcome:
    __slots__ = ("status", "expected", "actual", "difference", "detail", "inv")

    def __init__(self, status, expected=None, actual=None, difference=None, detail="", inv=None):
        self.status = status          # PASS/FAIL/SKIP/NOT_APPLICABLE/NOT_COMPARABLE
        self.expected = expected
        self.actual = actual
        self.difference = difference
        self.detail = detail
        self.inv = inv


def near(a, b, rtol=RTOL, atol=ABS_TOL, msg=""):
    """Compara numeros com tolerancia. Retorna (ok, diff)."""
    a, b = _num(a), _num(b)
    if a is None or b is None:
        return False, None
    if math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
        return True, abs(a - b)
    return False, abs(a - b)


# =============================================================================
# CARGA DE DADOS
# =============================================================================

def load_events(db_path):
    out = []
    if db_path and Path(db_path).exists():
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        for row in cur.execute("SELECT event_type, timestamp_ms, payload FROM events"):
            etype, ts, payload = row
            try:
                ev = json.loads(payload)
            except Exception as exc:
                ev = {"_payload_parse_error": str(exc), "tipo_evento": etype,
                      "epoch_ms": ts, "_raw_payload": str(payload)[:200]}
            out.append(ev)
        conn.close()
    return out


def load_jsonl(jsonl_path):
    out = []
    if jsonl_path and Path(jsonl_path).exists():
        for line in Path(jsonl_path).read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                out.append({"tipo_evento": "PARSE_ERROR", "_raw": line[:300]})
    return out


def window_bounds(meta_path, start_ms, end_ms):
    if start_ms is not None and end_ms is not None:
        return int(start_ms), int(end_ms)
    if meta_path and Path(meta_path).exists():
        try:
            meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
            s = float(meta.get("start_unix", 0)) * 1000
            e = float(meta.get("end_unix", 0)) * 1000
            if not e:
                e = parse_iso_to_ms(meta.get("end_utc")) or 0
            if s and e:
                return int(s), int(e)
        except Exception:
            pass
    return None, None


def is_new_window(ev, start_ms, end_ms):
    ts = ev.get("epoch_ms") or ev.get("timestamp_ms")
    if ts is None:
        return False
    try:
        ts = int(ts)
    except (TypeError, ValueError):
        return False
    if start_ms is not None and ts < start_ms:
        return False
    if end_ms is not None and ts > end_ms:
        return False
    return True


# =============================================================================
# PARSERS
# =============================================================================

def parse_iso_to_ms(s):
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


def resolve(d, *path):
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def deep_scan_float_strings(obj, path="", hits=None):
    """Procura NaN/Inf em strings/numeros. Retorna lista de (path, token)."""
    if hits is None:
        hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            deep_scan_float_strings(v, f"{path}.{k}" if path else str(k), hits)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            deep_scan_float_strings(v, f"{path}[{i}]", hits)
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            hits.append((path, repr(obj)))
    elif isinstance(obj, str):
        low = obj.strip().lower()
        if low in NAN_INF:
            hits.append((path, obj))
    return hits


# =============================================================================
# REGISTRO DE INVARIANTES
# =============================================================================

INVARIANTS = []


def register(group, name, fn, severity=HIGH, critical=False, details=""):
    iid = f"{group}:{name}"
    inv = Invariant(iid, group, name, fn, severity=severity,
                    critical_math=critical, details=details)
    INVARIANTS.append(inv)
    return inv


# ---------------------------------------------------------------------------
# GRUPO 1 - VOLUME / FLOW
# ---------------------------------------------------------------------------
def g1_volume_sum(ev, ctx):
    vt = _num(ev.get("volume_total"))
    vb = _num(ev.get("volume_compra"))
    vs = _num(ev.get("volume_venda"))
    if vt is None or vb is None or vs is None:
        return Outcome("SKIP", detail="campos volume_total/compra/venda ausentes")
    ok, d = near(vb + vs, vt, rtol=0.002, atol=1e-6)
    if not ok:
        return Outcome("FAIL", vb + vs, vt, _fmt(d), "volume_total != compra+venda")
    return Outcome("PASS", f"{vb}+{vs}", vt)


def g1_delta(ev, ctx):
    dl = _num(ev.get("delta"))
    vb = _num(ev.get("volume_compra"))
    vs = _num(ev.get("volume_venda"))
    if dl is None or vb is None or vs is None:
        return Outcome("SKIP", detail="campos delta/compra/venda ausentes")
    ok, d = near(vb - vs, dl, rtol=0.002, atol=1e-6)
    if not ok:
        return Outcome("FAIL", vb - vs, dl, _fmt(d), "delta != compra-venda")
    return Outcome("PASS", f"{vb}-{vs}", dl)


def g1_ob_btc_sum(ev, ctx):
    of = ev.get("fluxo_continuo", {}).get("order_flow") or {}
    tt = _num(of.get("total_volume_btc"))
    bb = _num(of.get("buy_volume_btc"))
    ss = _num(of.get("sell_volume_btc"))
    if tt is None or bb is None or ss is None:
        return Outcome("SKIP", detail="total/buy/sell_volume_btc ausentes")
    ok, d = near(bb + ss, tt, rtol=0.002, atol=1e-6)
    if not ok:
        return Outcome("FAIL", bb + ss, tt, _fmt(d), "total_volume_btc != buy+sell")
    return Outcome("PASS", f"{bb}+{ss}", tt)


def g1_flow_imbalance(ev, ctx):
    of = ev.get("fluxo_continuo", {}).get("order_flow") or {}
    imb = _num(of.get("flow_imbalance"))
    bb = _num(of.get("buy_volume_btc"))
    ss = _num(of.get("sell_volume_btc"))
    if imb is None or bb is None or ss is None:
        return Outcome("SKIP", detail="flow_imbalance/buy/sell_volume_btc ausentes")
    denom = bb + ss
    if denom <= 0:
        return Outcome("SKIP", detail="denominador <= 0")
    exp = (bb - ss) / denom
    ok, d = near(exp, imb, rtol=0.02, atol=0.01)
    if not ok:
        return Outcome("FAIL", exp, imb, _fmt(d), "flow_imbalance != (buy-sell)/(buy+sell)")
    return Outcome("PASS", exp, imb)


def g1_agg_pcts(ev, ctx):
    of = ev.get("fluxo_continuo", {}).get("order_flow") or {}
    ab = _num(of.get("aggressive_buy_pct"))
    asl = _num(of.get("aggressive_sell_pct"))
    if ab is None or asl is None:
        return Outcome("SKIP", detail="aggressive pcts ausentes")
    s = ab + asl
    ok, d = near(s, 100.0, rtol=0.002, atol=0.2)
    if not ok:
        return Outcome("FAIL", 100.0, s, _fmt(d), "buy_pct+sell_pct != 100")
    return Outcome("PASS", 100.0, s)


def g1_buy_sell_ratio(ev, ctx):
    of = ev.get("fluxo_continuo", {}).get("order_flow") or {}
    bsr = of.get("buy_sell_ratio")
    if not isinstance(bsr, dict):
        return Outcome("SKIP", detail="buy_sell_ratio ausente")
    val = _num(bsr.get("buy_sell_ratio"))
    bb = _num(of.get("buy_volume_btc"))
    ss = _num(of.get("sell_volume_btc"))
    if val is None or bb is None or ss is None or ss <= 0:
        return Outcome("SKIP", detail="sell<=0 ou campos ausentes")
    exp = bb / ss
    ok, d = near(exp, val, rtol=0.02, atol=0.01)
    if not ok:
        return Outcome("FAIL", exp, val, _fmt(d), "buy_sell_ratio != buy/sell")
    return Outcome("PASS", exp, val)


def g1_net_flow(ev, ctx):
    """net_flow_1m deve ser compativel com contrato corrigido: buy_volume - sell_volume (USD).
    Se metadata provar janela diferente (computation_window_min != 1), NOT_COMPARABLE."""
    of = ev.get("fluxo_continuo", {}).get("order_flow") or {}
    nf = _num(of.get("net_flow_1m"))
    bb = _num(of.get("buy_volume"))
    ss = _num(of.get("sell_volume"))
    cw = of.get("computation_window_min")
    if nf is None or bb is None or ss is None:
        return Outcome("SKIP", detail="net_flow_1m/buy_volume/sell_volume ausentes")
    if cw not in (1, "1", 1.0, None):
        return Outcome("NOT_COMPARABLE", bb - ss, nf,
                       detail=f"computation_window_min={cw} != 1 (universo temporal diferente)")
    exp = bb - ss
    ok, d = near(exp, nf, rtol=0.01, atol=10.0)
    if not ok:
        return Outcome("FAIL", exp, nf, _fmt(d),
                       "net_flow_1m != buy_volume-sell_volume (contrato corrigido)",
                       inv=ctx.get("_inv"))
    return Outcome("PASS", exp, nf)


register("volume/flow", "g1_volume_sum", g1_volume_sum, MEDIUM, True)
register("volume/flow", "g1_delta", g1_delta, HIGH, True)
register("volume/flow", "g1_ob_btc_sum", g1_ob_btc_sum, HIGH, True)
register("volume/flow", "g1_flow_imbalance", g1_flow_imbalance, HIGH, True)
register("volume/flow", "g1_agg_pcts", g1_agg_pcts, MEDIUM, True,
         "aggressive_buy_pct + aggressive_sell_pct ~ 100")
register("volume/flow", "g1_buy_sell_ratio", g1_buy_sell_ratio, MEDIUM, True)
register("volume/flow", "g1_net_flow", g1_net_flow, HIGH, True,
         "net_flow_1m compativel com contrato (buy-sell USD); NOT_COMPARABLE se janela != 1min")

# ---------------------------------------------------------------------------
# GRUPO 2 - SECTOR FLOW
# ---------------------------------------------------------------------------
def g2_sector_reconcile(ev, ctx):
    fc = ev.get("fluxo_continuo", {}) or {}
    sf = fc.get("sector_flow") or {}
    of = fc.get("order_flow") or {}
    if not isinstance(sf, dict) or not sf:
        return Outcome("SKIP", detail="sector_flow ausente (dict vazio)")
    buy_total = sum(_num(v.get("buy")) or 0 for v in sf.values())
    sell_total = sum(_num(v.get("sell")) or 0 for v in sf.values())
    ref_buy = _num(of.get("buy_volume_btc"))
    ref_sell = _num(of.get("sell_volume_btc"))
    uni = of.get("computation_window_min")
    if uni not in (1, "1", 1.0) and uni is not None:
        return Outcome("NOT_COMPARABLE", f"buy={_fmt(buy_total)},sell={_fmt(sell_total)}",
                       f"buy_btc={_fmt(ref_buy)},sell_btc={_fmt(ref_sell)}",
                       detail=f"computation_window_min={uni} -> universo temporal diferente")
    if ref_buy is None or ref_sell is None:
        return Outcome("SKIP", detail="buy/sell_volume_btc ausentes")
    ok_b, db_ = near(buy_total, ref_buy, rtol=0.01, atol=1e-6)
    ok_s, ds_ = near(sell_total, ref_sell, rtol=0.01, atol=1e-6)
    if ok_b and ok_s:
        # universo == janela corrente (mesmo fechamento): valida semantica interna
        for name, data in sf.items():
            b = _num(data.get("buy"))
            s = _num(data.get("sell"))
            dl = _num(data.get("delta"))
            if b is None or s is None:
                return Outcome("FAIL", "buy/sell numericos", f"{name}: {data}",
                               detail="bucket sem buy/sell numericos")
            if dl is not None:
                ok, d = near(b - s, dl, rtol=0.005, atol=1e-6)
                if not ok:
                    return Outcome("FAIL", b - s, dl, _fmt(d),
                                   f"{name}: delta != buy-sell")
        missing = set(("retail", "mid", "whale")) - set(sf.keys())
        if missing:
            return Outcome("FAIL", "retail/mid/whale presentes",
                           f"ausentes: {sorted(missing)}",
                           detail="bucket faltante na semantica atual (limite superior None)")
        return Outcome("PASS", f"sum={buy_total}/{sell_total}", f"{ref_buy}/{ref_sell}")
    # universo != janela corrente: evidencias de universo temporal diferente.
    # NUNCA chamar FAIL automaticamente sem confirmar o universo (spec GRUPO 2).
    evidence = []
    accumulated = buy_total >= 1.5 * ref_buy or sell_total >= 1.5 * ref_sell
    if accumulated:
        evidence.append("totais dos buckets >= 1.5x o fechamento da janela "
                        "(acumulador/rolling)")
    non_mono = ctx.get("hist_flow_btc")
    if non_mono and len(non_mono) >= 4:
        buys = [r[1] for r in non_mono]
        if sum(1 for i in range(1, len(buys)) if buys[i] < buys[i - 1]) == 0 \
                and buys[-1] > 0:
            evidence.append("buckets crescem monotonicamente (universo acumulado)")
    if accumulated or evidence:
        return Outcome("NOT_COMPARABLE",
                       f"buy={_fmt(buy_total)},sell={_fmt(sell_total)}",
                       f"janela: buy={_fmt(ref_buy)},sell={_fmt(ref_sell)}",
                       detail="; ".join(evidence) + " | meta: universos temporais distintos")
    if ctx.get("hist_flow_btc") is not None:
        tail = ctx["hist_flow_btc"]
        matched = None
        for n in range(2, len(tail) + 1):
            blk = tail[-n:]
            cb = sum(r[1] for r in blk)
            cs = sum(r[2] for r in blk)
            if near(cb, buy_total, rtol=0.15, atol=0.5)[0] \
                    and near(cs, sell_total, rtol=0.15, atol=0.5)[0]:
                matched = (n, cb, cs)
                break
        if matched is not None:
            return Outcome("NOT_COMPARABLE",
                           f"buy={_fmt(buy_total)},sell={_fmt(sell_total)}",
                           f"bloco({matched[0]} janelas) buy={_fmt(matched[1])},sell={_fmt(matched[2])}",
                           detail=f"universe=rolling ({matched[0]} janelas); "
                                  "meta: sector_flow nao corresponde ao fechamento da janela")
    return Outcome("FAIL", f"buy={_fmt(buy_total)},sell={_fmt(sell_total)}",
                   f"buy={_fmt(ref_buy)},sell={_fmt(ref_sell)}",
                   f"diff buy={_fmt(db_)},sell={_fmt(ds_)}",
                   "soma dos buckets não reconcilia com a janela nem com bloco; "
                   "sem evidencia de universo acumulado -> investigar antes de concluir")


register("sector-flow", "g2_sector_reconcile", g2_sector_reconcile, HIGH, True,
         "somatórios sector-flow reconciliam com volume; NOT_COMPARABLE se universo difere")

# ---------------------------------------------------------------------------
# GRUPO 3 - ORDERBOOK
# ---------------------------------------------------------------------------
def g3_bid_ask(ev, ctx):
    bid = _num(ev.get("bid"))
    ask = _num(ev.get("ask"))
    if bid is None or ask is None:
        return Outcome("SKIP", detail="bid/ask ausentes")
    if bid >= ask:
        return Outcome("FAIL", "bid < ask", f"bid={bid},ask={ask}", _fmt(bid - ask))
    return Outcome("PASS", f"{bid} < {ask}")


def g3_mid_spread(ev, ctx):
    ob = ev.get("orderbook_data") or {}
    mid = _num(ob.get("mid"))
    spread = _num(ob.get("spread"))
    bid = _num(ev.get("bid"))
    ask = _num(ev.get("ask"))
    if mid is not None and bid is not None and ask is not None:
        exp = (bid + ask) / 2
        ok, d = near(exp, mid, rtol=0.001, atol=0.05)
        if not ok:
            return Outcome("FAIL", exp, mid, _fmt(d), "mid != (bid+ask)/2 usando bid/ask top-level")
    if mid is None or spread is None:
        return Outcome("SKIP", detail="mid/spread ausentes")
    # spread vs bid/ask top-level (mesma janela; mesma captura se latencia baixa)
    if bid is not None and ask is not None:
        exp_sp = ask - bid
        ok, d = near(exp_sp, spread, rtol=0.05, atol=0.11)
        if not ok:
            return Outcome("FAIL", exp_sp, spread, _fmt(d),
                           "spread != ask-bid (possivel captura de timestamps diferentes)")
    spb = _num(ob.get("spread_bps"))
    if spb is not None and mid and mid > 0:
        exp_bps = spread / mid * 10000
        ok, d = near(exp_bps, spb, rtol=0.05, atol=0.01)
        if not ok:
            return Outcome("FAIL", exp_bps, spb, _fmt(d), "spread_bps != spread/mid*10000")
    return Outcome("PASS", f"mid={_fmt(mid)}, spread={_fmt(spread)}")


def g3_depth_imbalance(ev, ctx):
    ob = ev.get("orderbook_data") or {}
    imb = _num(ob.get("imbalance"))
    bd = _num(ob.get("bid_depth_usd"))
    ad = _num(ob.get("ask_depth_usd"))
    if imb is None or bd is None or ad is None:
        return Outcome("SKIP", detail="imbalance/depths ausentes")
    denom = bd + ad
    if denom <= 0:
        return Outcome("SKIP", detail="bid+ask depth <= 0")
    exp = (bd - ad) / denom
    ok, d = near(exp, imb, rtol=0.02, atol=0.01)
    if not ok:
        return Outcome("FAIL", exp, imb, _fmt(d), "imbalance != (bid_depth-ask_depth)/(bid+ask)")
    return Outcome("PASS", exp, imb)


def g3_volume_ratio(ev, ctx):
    ob = ev.get("orderbook_data") or {}
    vr = _num(ob.get("volume_ratio"))
    bd = _num(ob.get("bid_depth_usd"))
    ad = _num(ob.get("ask_depth_usd"))
    if vr is None or bd is None or ad is None or ad <= 0:
        return Outcome("SKIP", detail="volume_ratio/depths ausentes ou ask<=0")
    exp = bd / ad
    ok, d = near(exp, vr, rtol=0.02, atol=0.01)
    if not ok:
        return Outcome("FAIL", exp, vr, _fmt(d), "volume_ratio != bid_depth/ask_depth")
    return Outcome("PASS", exp, vr)


def g3_depth_levels(ev, ctx):
    obd = ev.get("order_book_depth") or {}
    bad = []
    for lvl in ("L1", "L5", "L10", "L25"):
        d = obd.get(lvl)
        if not isinstance(d, dict):
            continue
        b = _num(d.get("bids"))
        a = _num(d.get("asks"))
        if b is None or a is None:
            continue
        if b < 0 or a < 0:
            bad.append(f"{lvl}: bids={b},asks={a} <0")
        fi = _num(d.get("flow_imbalance"))
        if fi is not None and not -1.0 - 1e-9 <= fi <= 1.0 + 1e-9:
            bad.append(f"{lvl}: flow_imbalance {fi} fora de [-1,1]")
    if not obd:
        return Outcome("SKIP", detail="order_book_depth ausente")
    if bad:
        return Outcome("FAIL", "valores nao-negativos e imb em [-1,1]", "; ".join(bad))
    return Outcome("PASS", "L1/L5/L10/L25 nao-negativos")


def g3_market_impact(ev, ctx):
    mi = ev.get("market_impact") or {}
    sm = mi.get("slippage_matrix") or {}
    if not sm:
        return Outcome("SKIP", detail="slippage_matrix ausente")
    sizes = ["1k_usd", "10k_usd", "100k_usd", "1m_usd"]
    prev_b, prev_s = None, None
    bad = []
    for sz in sizes:
        band = sm.get(sz)
        if not isinstance(band, dict):
            continue
        for side in ("buy", "sell"):
            v = band.get(side)
            if v is None:
                # nivel ausente: valido apenas se execucao nao modelada naquele tamanho
                # (depth insuficiente) - NAO classificado como erro; NOT_APPLICABLE
                continue
            vf = _num(v)
            if vf is None:
                bad.append(f"{sz}.{side}=NaN/Inf/non-numeric")
                continue
            if vf < 0:
                bad.append(f"{sz}.{side}<0")
        pb = _num(band.get("buy"))
        ps = _num(band.get("sell"))
        if prev_b is not None and pb is not None and pb < prev_b:
            bad.append(f"slippage buy nao monotona {sz}({pb}) < {prev_b}")
        if prev_s is not None and ps is not None and ps < prev_s:
            bad.append(f"slippage sell nao monotona {sz}({ps}) < {prev_s}")
        if pb is not None:
            prev_b = pb
        if ps is not None:
            prev_s = ps
    lq = _num(mi.get("liquidity_score"))
    if lq is not None and not 0 <= lq <= 10:
        bad.append(f"liquidity_score {lq} fora de [0,10]")
    if bad:
        return Outcome("FAIL", "impacto=0..10, monotono, sem NaN/Inf", "; ".join(bad))
    return Outcome("PASS", "market_impact consistente (niveis ausentes = execucao nao modelada)")


def g3_source_liveness(ev, ctx):
    ob = ev.get("orderbook_data") or {}
    src = str(ob.get("data_source", "")).lower()
    oq = str(ev.get("orderbook_quality", "")).lower()
    degraded = ("stale", "cache", "fallback", "emergency", "unknown")
    if src in degraded and oq == "live":
        return Outcome("FAIL", "fonte degradada nao classifica como live",
                       f"data_source={src}, orderbook_quality={oq}", severity_hint="live_" + src)
    if src not in ("live", "") and oq == "live":
        return Outcome("FAIL", "fonte != live mas quality=live", f"data_source={src}")
    return Outcome("PASS", f"data_source={src}, quality={oq}")


register("orderbook", "g3_bid_ask", g3_bid_ask, HIGH, True)
register("orderbook", "g3_mid_spread", g3_mid_spread, HIGH, True)
register("orderbook", "g3_depth_imbalance", g3_depth_imbalance, HIGH, True)
register("orderbook", "g3_volume_ratio", g3_volume_ratio, MEDIUM, True)
register("orderbook", "g3_depth_levels", g3_depth_levels, MEDIUM, False)
register("orderbook", "g3_market_impact", g3_market_impact, MEDIUM, False)
register("orderbook", "g3_source_liveness", g3_source_liveness, BLOCKER, False,
         "stale/cache/fallback/emergency/unknown nunca classificados como live")

# ---------------------------------------------------------------------------
# GRUPO 4 - TEMPORAL
# ---------------------------------------------------------------------------
def g4_ts_consistency(ev, ctx):
    ep = ev.get("epoch_ms")
    t_utc = parse_iso_to_ms(ev.get("timestamp_utc"))
    t_ny = parse_iso_to_ms(ev.get("timestamp_ny"))
    t_sp = parse_iso_to_ms(ev.get("timestamp_sp"))
    refs = [("epoch_ms", ep), ("timestamp_utc", t_utc), ("timestamp_ny", t_ny),
            ("timestamp_sp", t_sp)]
    present = [(n, v) for n, v in refs if v is not None]
    if not present:
        return Outcome("SKIP", detail="sem timestamps")
    base_name, base = present[0]
    bad = []
    for n, v in present[1:]:
        d = abs(v - base)
        if d > 60_000:
            bad.append(f"{n} difere {base_name} por {d}ms")
    if bad:
        return Outcome("FAIL", "mesmos instante (±60s)", "; ".join(bad))
    # exchange/window nao pode estar absurdamente a frente do evento de criacao
    # (evento gravado depois do close da janela)
    etype = ev.get("tipo_evento")
    if etype == EVENT_AI and base is not None:
        created = ev.get("epoch_ms")
        if created is not None and _num(created) is not None and _num(created) < base - 60_000:
            return Outcome("FAIL", "criacao >= janela ancorada",
                           f"created={created}", _fmt(base - created),
                           "evento AI com criacao anterior ao close da janela")
    return Outcome("PASS", f"{base_name}={base}")


def g4_latency_equiv(ev, ctx):
    q = ev.get("institutional_analytics", {}).get("quality", {}).get("latency", {})
    dr = ev.get("data_reliability", {}) or {}
    a1 = q.get("is_acceptable")
    a2 = dr.get("latency_acceptable")
    if a1 is None or a2 is None:
        return Outcome("SKIP", detail="is_acceptable/latency_acceptable ausentes (um deles)")
    if a1 != a2:
        return Outcome("FAIL", str(a1), str(a2),
                       detail="quality.latency.is_acceptable != data_reliability.latency_acceptable")
    return Outcome("PASS", str(a1), str(a2))


def g4_latency_ranges(ev, ctx):
    """Consistencia interna: is_acceptable/is_stale vs categoria."""
    q = ev.get("institutional_analytics", {}).get("quality", {}).get("latency", {})
    if not q:
        return Outcome("SKIP", detail="latency ausente")
    acc = q.get("is_acceptable")
    stale = q.get("is_stale")
    cat = str(q.get("latency_category", "")).upper()
    ms = _num(q.get("latency_ms"))
    if stale is True and acc is True:
        return Outcome("FAIL", "stale nao pode ser acceptable",
                       f"is_stale={stale}, is_acceptable={acc}")
    if ms is not None and ms < 0:
        return Outcome("FAIL", "latency_ms >= 0", ms)
    if cat and acc is True and cat in ("POOR", "DELAYED", "STALE", "CRITICAL"):
        return Outcome("FAIL", "categoria compativel com acceptable",
                       f"cat={cat}, acc={acc}")
    return Outcome("PASS", f"ms={_fmt(ms)}, cat={cat}, acc={acc}")


def g4_window_continuity(ev, ctx):
    """Janelas consecutivas: proxima janela ~60s depois (janela de 1min)."""
    prev = ctx.get("prev_epoch_ms")
    if prev is None:
        return Outcome("SKIP", detail="primeira janela")
    cur = ev.get("epoch_ms")
    if cur is None or prev is None:
        return Outcome("SKIP", detail="epoch_ms ausente")
    d = cur - prev
    if not (50_000 <= d <= 75_000):
        return Outcome("FAIL", "gap ~60s ±15s", f"d={d}ms",
                       detail=f"janelas nao consecutivas (prev={prev})")
    return Outcome("PASS", "~60s", d)


register("temporal", "g4_ts_consistency", g4_ts_consistency, HIGH, False)
register("temporal", "g4_latency_equiv", g4_latency_equiv, HIGH, False)
register("temporal", "g4_latency_ranges", g4_latency_ranges, MEDIUM, False)
register("temporal", "g4_window_continuity", g4_window_continuity, MEDIUM, False)

# ---------------------------------------------------------------------------
# GRUPO 5 - VALUE PROFILE
# ---------------------------------------------------------------------------
def _vp_hist(ev, tf="daily"):
    return (ev.get("historical_vp") or {}).get(tf) or {}


def g5_va_pct_bounds(ev, ctx):
    pa = ev.get("institutional_analytics", {}).get("profile_analysis", {}) or {}
    va = pa.get("va_volume_pct") or {}
    status = str(va.get("status", "missing")).lower()
    if status in ("missing", "insufficient_data", "error"):
        return Outcome("SKIP", detail=f"va_volume_pct status={status}")
    if status != "success":
        return Outcome("SKIP", detail=f"status={status}")
    pct = _num(va.get("value_area_volume_pct"))
    if pct is None:
        return Outcome("FAIL", "value_area_volume_pct numerico", va.get("value_area_volume_pct"),
                       detail="success sem pct")
    if not 0 <= pct <= 100:
        return Outcome("FAIL", "0 <= pct <= 100", pct)
    vi = _num(va.get("volume_in_va"))
    tv = _num(va.get("total_volume"))
    if vi is not None and tv is not None and vi > tv + 1e-6:
        return Outcome("FAIL", "volume_in_va <= total_volume", f"{vi} > {tv}")
    return Outcome("PASS", f"pct={pct}, vi={_fmt(vi)}, tv={_fmt(tv)}")


def g5_vp_levels(ev, ctx):
    for tf in ("daily", "weekly", "monthly"):
        vp = _vp_hist(ev, tf)
        if not vp:
            continue
        val, vah, poc = _num(vp.get("val")), _num(vp.get("vah")), _num(vp.get("poc"))
        if val is not None and vah is not None and val > vah:
            return Outcome("FAIL", "VAL <= VAH", f"{tf}: val={val} > vah={vah}")
        if poc is not None and val is not None and vah is not None:
            if not (val - 1e-6 <= poc <= vah + 1e-6):
                return Outcome("FAIL", "VAL <= POC <= VAH", f"{tf}: poc={poc} fora de [val,vah]")
        st = str(vp.get("status", "")).lower()
        if st == "success":
            hits = deep_scan_float_strings(vp)
            if hits:
                return Outcome("FAIL", "sem NaN/Inf", f"{tf}: {hits[:3]}")
    return Outcome("PASS", "VP levels consistentes (VAL<=VAH, POC dentro)")


def g5_va_status_contract(ev, ctx):
    """status=error nao fabrica pct plausivel; insufficient_data nao gera sinais positivos."""
    pa = ev.get("institutional_analytics", {}).get("profile_analysis", {}) or {}
    va = pa.get("va_volume_pct") or {}
    status = str(va.get("status", "")).lower()
    if status == "error":
        pct = _num(va.get("value_area_volume_pct"))
        if pct is not None:
            return Outcome("FAIL", "status=error sem pct fabricado", pct,
                           detail="erro nao pode conter pct plausivel")
        return Outcome("PASS", "status=error sem pct fabricado")
    if status == "insufficient_data":
        if va.get("breakout_risk") in ("HIGH", "MODERATE"):
            return Outcome("FAIL", "insufficient_data sem breakout/compression positivo",
                           va.get("breakout_risk"))
        if va.get("compression_signal") is True:
            return Outcome("FAIL", "compression_signal não deve ser True em insufficient_data",
                           True)
        return Outcome("PASS", "insufficient_data sem sinais positivos")
    return Outcome("SKIP", detail=f"status={status or 'missing'}")


def g5_va_recompute(ev, ctx):
    """Se bins internos existirem, recalcular VA independentemente (amostra)."""
    vp = _vp_hist(ev, "daily")
    pb = vp.get("price_bins") or vp.get("bins")
    vpb = vp.get("volume_per_bin") or vp.get("volumes")
    if pb is None or vpb is None:
        return Outcome("SKIP", detail="bins internos ausentes (sem dados p/ recomputo)")
    try:
        bins = [float(x) for x in pb]
        vols = [float(x) for x in vpb]
    except Exception:
        return Outcome("SKIP", detail="bins nao numericos")
    if len(bins) != len(vols) or not bins:
        return Outcome("SKIP", detail="dimensao bins != volumes")
    total = sum(vols)
    if total <= 0:
        return Outcome("SKIP", detail="volume total <= 0")
    pairs = sorted(zip(bins, vols))
    # VA = 70% central (config VP_VALUE_AREA_PERCENT=0.7)
    target = total * 0.70
    n = len(pairs)
    best = None
    for i in range(n):
        acc, j = 0.0, i
        while j < n and acc < target:
            acc += pairs[j][1]
            j += 1
        if j <= n and j > i:
            span = pairs[j - 1][0] - pairs[i][0]
            if best is None or span < best[0]:
                best = (span, pairs[i][0], pairs[j - 1][0])
    if best is None:
        return Outcome("SKIP", detail="sem janela central valida")
    lo, hi = best[1], best[2]
    val, vah = _num(vp.get("val")), _num(vp.get("vah"))
    if val is None or vah is None:
        return Outcome("SKIP", detail="val/vah ausentes")
    tol = 0.05 * max(abs(hi - lo), 1.0)
    if abs(val - lo) > tol or abs(vah - hi) > tol:
        return Outcome("FAIL", f"val={lo}, vah={hi} (recomputo)", f"val={val}, vah={vah}",
                       _fmt(max(abs(val - lo), abs(vah - hi))),
                       "VA recomputado difere do reportado")
    return Outcome("PASS", f"val={_fmt(lo)}, vah={_fmt(hi)} (recomputo)", f"{val}/{vah}")


register("value-profile", "g5_va_pct_bounds", g5_va_pct_bounds, HIGH, True)
register("value-profile", "g5_vp_levels", g5_vp_levels, HIGH, True)
register("value-profile", "g5_va_status_contract", g5_va_status_contract, HIGH, False)
register("value-profile", "g5_va_recompute", g5_va_recompute, MEDIUM, False)

# ---------------------------------------------------------------------------
# GRUPO 6 - S/R
# ---------------------------------------------------------------------------
def g6_sr_position(ev, ctx):
    price = _num(ev.get("preco_fechamento"))
    sups = ev.get("immediate_support") or []
    ress = ev.get("immediate_resistance") or []
    if price is None:
        return Outcome("SKIP", detail="preco_fechamento ausente")
    bad = []
    for s in sups:
        sf = _num(s)
        if sf is not None and sf > price + 1e-6:
            bad.append(f"support {sf} > price {price}")
    for r in ress:
        rf = _num(r)
        if rf is not None and rf < price - 1e-6:
            bad.append(f"resistance {rf} < price {price}")
    if not sups and not ress:
        return Outcome("SKIP", detail="sem supports/resistances")
    if bad:
        return Outcome("FAIL", f"support<=price<=resistance", "; ".join(bad))
    return Outcome("PASS", f"price={price}")


def g6_defense_position(ev, ctx):
    dz = ev.get("institutional_analytics", {}).get("sr_analysis", {}).get("defense_zones") or {}
    price = _num(ev.get("preco_fechamento"))
    if price is None or not dz:
        return Outcome("SKIP", detail="defense_zones ausentes")
    bad = []
    for z in dz.get("buy_defense", []) or []:
        c = _num(z.get("center"))
        if c is not None and c > price + 1e-6:
            bad.append(f"buy_defense center {c} > price {price}")
    for z in dz.get("sell_defense", []) or []:
        c = _num(z.get("center"))
        if c is not None and c < price - 1e-6:
            bad.append(f"sell_defense center {c} < price {price}")
    if bad:
        return Outcome("FAIL", "buy center<=price<=sell center", "; ".join(bad))
    return Outcome("PASS", f"defenses ok (price={price})")


def g6_defense_metrics(ev, ctx):
    dz = ev.get("institutional_analytics", {}).get("sr_analysis", {}).get("defense_zones") or {}
    bad = []
    for side in ("buy_defense", "sell_defense"):
        for z in dz.get(side, []) or []:
            srcs = z.get("sources") or []
            sc = _num(z.get("source_count"))
            if sc is not None and len(set(srcs)) < sc:
                bad.append(f"{side}: source_count={sc} > fontes distintas {len(set(srcs))}")
            siz = _num(z.get("signals_in_zone"))
            if sc is not None and siz is not None and siz < sc:
                bad.append(f"{side}: signals_in_zone={siz} < source_count={sc}")
            st = _num(z.get("strength"))
            if st is not None and not 0 <= st <= 100:
                bad.append(f"{side}: strength {st} fora de [0,100]")
    if bad:
        return Outcome("FAIL", "source_count/strength consistentes", "; ".join(bad))
    return Outcome("PASS", "defense metrics consistentes")


register("sr", "g6_sr_position", g6_sr_position, HIGH, False)
register("sr", "g6_defense_position", g6_defense_position, HIGH, False)
register("sr", "g6_defense_metrics", g6_defense_metrics, MEDIUM, False)

# ---------------------------------------------------------------------------
# GRUPO 7 - PIVOTS
# ---------------------------------------------------------------------------
def g7_classic_pivots(ev, ctx):
    pivs = ev.get("pivots") or {}
    bad = []
    for tf in ("daily", "weekly", "monthly"):
        pv = pivs.get(tf) or {}
        h, l, c = _num(pv.get("high")), _num(pv.get("low")), _num(pv.get("close"))
        if h is None or l is None or c is None or h <= l:
            continue
        P = (h + l + c) / 3
        R1, S1 = 2 * P - l, 2 * P - h
        R2, S2 = P + (h - l), P - (h - l)
        for name, exp, got in (("pivot", P, _num(pv.get("pivot"))),
                               ("r1", R1, _num(pv.get("r1"))),
                               ("s1", S1, _num(pv.get("s1"))),
                               ("r2", R2, _num(pv.get("r2"))),
                               ("s2", S2, _num(pv.get("s2")))):
            if got is None or exp is None:
                continue
            ok, d = near(exp, got, rtol=0.002, atol=1e-6)
            if not ok:
                bad.append(f"{tf}.{name}: esperado {exp:.4f} obtido {got:.4f} (d={d:.4f})")
    if not bad and not pivs:
        return Outcome("SKIP", detail="pivots ausente")
    if bad:
        return Outcome("FAIL", "formulas classicas P,R1,S1,R2,S2", "; ".join(bad))
    return Outcome("PASS", "pivots classicos recalcularam iguais")


def g7_pivot_points_legacy(ev, ctx):
    """pivot_points.vah/val/poc sao alias H/L/C (classic source) - NAO sao Volume Profile."""
    pp = ev.get("pivot_points") or {}
    hist = (ev.get("historical_vp") or {}).get("daily") or {}
    bad = []
    for tf, d in pp.items():
        if not isinstance(d, dict):
            continue
        src = str(d.get("source", ""))
        if src != "classic":
            continue
        vah, val, poc = _num(d.get("vah")), _num(d.get("val")), _num(d.get("poc"))
        if vah is not None and val is not None and val > vah:
            bad.append(f"pivot_points.{tf}: val > vah")
    hv_vah, hv_val = _num(hist.get("vah")), _num(hist.get("val"))
    if hv_vah is not None and hv_val is not None and hv_val > hv_vah:
        bad.append(f"historical_vp.daily: val > vah")
    if bad:
        return Outcome("FAIL", "ordem de niveis", "; ".join(bad))
    return Outcome("PASS", "legacy aliases nao interpretados como VP (val<=vah ok)")


register("pivots", "g7_classic_pivots", g7_classic_pivots, HIGH, True)
register("pivots", "g7_pivot_points_legacy", g7_pivot_points_legacy, MEDIUM, False,
         "legacy pivot_points.vah/val/poc NAO e Volume Profile (alias classic)")

# ---------------------------------------------------------------------------
# GRUPO 8 - DERIVATIVES
# ---------------------------------------------------------------------------
def g8_oi_plausibility(ev, ctx):
    der = (ev.get("derivatives") or {}).get("BTCUSDT") or {}
    oi = _num(der.get("open_interest"))
    oi_usd = _num(der.get("open_interest_usd"))
    price = _num(ev.get("preco_fechamento"))
    if oi is None or oi_usd is None or price is None:
        return Outcome("SKIP", detail="open_interest/open_interest_usd/price ausentes")
    exp = oi * price
    ok, d = near(exp, oi_usd, rtol=0.05, atol=1)
    if not ok:
        return Outcome("FAIL", exp, oi_usd, _fmt(d),
                       "open_interest_usd != open_interest*price (contrato) ou OI nao em contratos")
    return Outcome("PASS", exp, oi_usd)


def g8_long_short_reconcile(ev, ctx):
    der = (ev.get("derivatives") or {}).get("BTCUSDT") or {}
    oi_usd = _num(der.get("open_interest_usd"))
    longs = _num(der.get("longs_usd"))
    shorts = _num(der.get("shorts_usd"))
    lsr = _num(der.get("long_short_ratio"))
    if longs is None or shorts is None or oi_usd is None:
        return Outcome("SKIP", detail="longs/shorts/oi_usd ausentes")
    ok, d = near(longs + shorts, oi_usd, rtol=0.05, atol=1)
    if not ok:
        return Outcome("FAIL", longs + shorts, oi_usd, _fmt(d),
                       "longs_usd+shorts_usd != open_interest_usd")
    if lsr is not None and shorts > 0:
        exp = longs / shorts
        ok2, d2 = near(exp, lsr, rtol=0.05, atol=0.01)
        if not ok2:
            return Outcome("FAIL", exp, lsr, _fmt(d2), "long_short_ratio != longs/shorts")
    return Outcome("PASS", f"longs+shorts={longs + shorts}", f"oi={oi_usd}, lsr={lsr}")


def g8_funding_unit(ev, ctx):
    der = (ev.get("derivatives") or {}).get("BTCUSDT") or {}
    fr = der.get("funding_rate_percent")
    if fr is None:
        return Outcome("SKIP", detail="funding_rate_percent ausente")
    f = _num(fr)
    if f is None:
        return Outcome("FAIL", "numeric funding", fr, detail="NaN/Inf em funding")
    if abs(f) >= 100:
        return Outcome("FAIL", "campo *_percent em percentual",
                       f, detail="valor >=100 indica fração bruta armazenada em campo percentual")
    return Outcome("PASS", f"{f}% (~{f*10:.1f} bps)")


register("derivatives", "g8_oi_plausibility", g8_oi_plausibility, HIGH, True)
register("derivatives", "g8_long_short_reconcile", g8_long_short_reconcile, HIGH, True)
register("derivatives", "g8_funding_unit", g8_funding_unit, MEDIUM, False,
         "detecta apenas erro de unidade, nao direcao de mercado")

# ---------------------------------------------------------------------------
# GRUPO 9 - MACRO
# ---------------------------------------------------------------------------
def g9_no_nan_inf(ev, ctx):
    hits = deep_scan_float_strings(ev)
    if hits:
        return Outcome("FAIL", "nenhum NaN/Inf", f"{len(hits)} ocorrencias",
                       "; ".join(f"{p}={v}" for p, v in hits[:5]),
                       "NaN/Inf persistido em JSON")
    return Outcome("PASS", "sem NaN/Inf no payload")


def g9_macro_status_classification(ev, ctx):
    """Missing deve ser null/None/status apropriado; distinguir categorias.
    Registra classificacao, nao assume que null = bug."""
    findings = []
    macro_blocks = {
        "external_markets": ev.get("external_markets"),
        "multi_tf": ev.get("multi_tf"),
        "historical_vp": ev.get("historical_vp"),
        "derivatives": ev.get("derivatives"),
        "market_environment": ev.get("market_environment"),
    }
    for name, blk in macro_blocks.items():
        if blk is None:
            findings.append((name, "MISSING", "null"))
        elif isinstance(blk, dict) and not blk:
            findings.append((name, "MISSING/EMPTY(optional)", "{}"))
        elif isinstance(blk, dict) and all(
            (v is None or (isinstance(v, dict) and not v)
             or str(v.get("status", "")).lower() in ("insufficient_data", "insufficient_history"))
            for v in blk.values() if isinstance(v, dict)):
            findings.append((name, "INSUFFICIENT_HISTORY", "status insuficiente"))
        else:
            findings.append((name, "PRESENT", list(blk.keys())[:4]))
    desc = "; ".join(f"{n}={st}" for n, st, _ in findings)
    return Outcome("PASS", "classificacao registrada", desc,
                   detail="null/{} nao tratado como bug; rastrear fonte via status")


register("macro", "g9_no_nan_inf", g9_no_nan_inf, HIGH, False)
register("macro", "g9_macro_status_classification", g9_macro_status_classification, LOW, False)

# ---------------------------------------------------------------------------
# GRUPO 10 - IA PAYLOAD
# ---------------------------------------------------------------------------
def _ai_sample(ev, ctx):
    return ev  # retorna o proprio evento AI_ANALYSIS


def build_at_index(events):
    idx = {}
    for ev in events:
        if ev.get("tipo_evento") == EVENT_AT:
            jn = ev.get("janela_numero")
            ep = ev.get("epoch_ms")
            if jn is not None:
                idx.setdefault(int(jn), ev)
            if ep is not None:
                idx.setdefault(f"ep:{ep}", ev)
    return idx


def g10_anchor(ev, ctx):
    at_idx = ctx.get("at_index", {})
    ap = ev.get("ai_payload") or {}
    anchor_wid = ev.get("anchor_window_id")
    anchor = None
    if anchor_wid is not None:
        anchor = at_idx.get(int(anchor_wid))
    if anchor is None:
        ep = ap.get("epoch_ms")
        if ep is not None:
            anchor = at_idx.get(f"ep:{ep}")
    if anchor is None:
        return Outcome("SKIP", detail="sem ANALYSIS_TRIGGER ancorado (window id ou epoch)")
    ctx["_anchor"] = anchor
    return None


def g10_price(ev, ctx):
    anchor = ctx.get("_anchor")
    ap = ev.get("ai_payload") or {}
    if anchor is None or not ap.get("price"):
        return Outcome("SKIP", detail="sem anchor/price")
    pc = _num(ap["price"].get("c"))
    ref = _num(anchor.get("preco_fechamento"))
    if pc is None or ref is None:
        return Outcome("SKIP", detail="price.c/preco ausentes")
    ok, d = near(pc, ref, rtol=0.01, atol=ref * 0.01)
    if not ok:
        return Outcome("FAIL", ref, pc, _fmt(d), "price compacto != preco da janela ancorada")
    return Outcome("PASS", ref, pc)


def g10_flow(ev, ctx):
    anchor = ctx.get("_anchor")
    ap = ev.get("ai_payload") or {}
    of = (anchor or {}).get("fluxo_continuo", {}).get("order_flow") or {}
    fl = ap.get("flow") or {}
    bad = []

    def chk(desc, exp, got, rt=0.05, at=0.02):
        if got is None:
            return
        ok, d = near(exp, got, rtol=rt, abs_tol=at)
        if not ok:
            bad.append(f"{desc}: exp={_fmt(exp)} got={_fmt(got)} d={_fmt(d)}")

    tb = _num(of.get("total_volume_btc"))
    imb = _num(of.get("flow_imbalance"))
    if tb is not None:
        chk("flow.vol", tb, _num(fl.get("vol")), rt=0.03, at=0.5)
    if imb is not None:
        chk("flow.imb/flow.delta", imb, _num(fl.get("imb")) if fl.get("imb") is not None else _num(fl.get("delta")), rt=0.15, at=0.05)
    bsr = _num(of.get("buy_sell_ratio", {}).get("buy_sell_ratio")) if isinstance(of.get("buy_sell_ratio"), dict) else None
    if bsr is not None:
        chk("flow.bsr", bsr, _num(fl.get("bsr")), rt=0.15, at=0.05)
    if not bad:
        return Outcome("PASS", "flow compacto coerente com anchor")
    return Outcome("FAIL", "flow compativel", "; ".join(bad[:4]))


def g10_ob(ev, ctx):
    anchor = ctx.get("_anchor")
    ap = ev.get("ai_payload") or {}
    ob = (anchor or {}).get("orderbook_data") or {}
    obp = ap.get("ob") or {}
    bad = []
    imb = _num(ob.get("imbalance"))
    if imb is not None and obp.get("imb") is not None:
        ok, d = near(imb, _num(obp.get("imb")), rtol=0.15, atol=0.05)
        if not ok:
            bad.append(f"ob.imb: exp={_fmt(imb)} got={_fmt(obp.get('imb'))}")
    bias = obp.get("bias")
    if bias:
        sent = ev.get("ai_result", {}).get("sentiment")
        if bias == "SELL" and sent == "bullish":
            bad.append(f"ob.bias={bias} vs sentiment={sent} (possivel inversao)")
        if bias == "BUY" and sent == "bearish":
            bad.append(f"ob.bias={bias} vs sentiment={sent} (possivel inversao)")
    if not bad:
        return Outcome("PASS", f"ob compacto coerente (bias={bias})")
    return Outcome("FAIL", "orderbook bias coerente", "; ".join(bad))


def g10_sr(ev, ctx):
    anchor = ctx.get("_anchor")
    ap = ev.get("ai_payload") or {}
    srp = ap.get("sr") or {}
    price = _num((anchor or {}).get("preco_fechamento"))
    if price is None:
        return Outcome("SKIP", detail="sem preco anchor")
    bad = []
    s1 = srp.get("s1")
    r1 = srp.get("r1")
    if isinstance(s1, list) and len(s1) >= 1:
        p = _num(s1[0])
        if p is not None and p > price * 1.01:
            bad.append(f"sr.s1={p} acima do preço {price}")
    if isinstance(r1, list) and len(r1) >= 1:
        p = _num(r1[0])
        if p is not None and p < price * 0.99:
            bad.append(f"sr.r1={p} abaixo do preço {price}")
    if not bad:
        return Outcome("PASS", f"s1={s1[0] if isinstance(s1, list) and s1 else None}, r1={r1[0] if isinstance(r1, list) and r1 else None}")
    return Outcome("FAIL", "S/R compacto coerente", "; ".join(bad))


def g10_quality(ev, ctx):
    anchor = ctx.get("_anchor")
    ap = ev.get("ai_payload") or {}
    q = anchor.get("institutional_analytics", {}).get("quality", {}).get("latency", {}) if anchor else {}
    qp = ap.get("qual") or {}
    bad = []
    lat = q.get("latency_category")
    if lat and qp.get("lat") and lat != qp.get("lat"):
        bad.append(f"qual.lat={qp.get('lat')} vs quality.latency_category={lat}")
    acc = q.get("is_acceptable")
    summ = ap.get("summary", {}).get("quality", {})
    rel = summ.get("reliable")
    if acc is False and rel is True:
        bad.append("latency inaceitavel mas summary.quality.reliable=True (ausencia virou healthy)")
    if acc is False and rel is not False:
        bad.append(f"latency inaceitavel, reliable={rel}")
    if acc is False and ev.get("ai_result", {}).get("confidence", 0) > 0.95:
        bad.append(f"confidence {ev['ai_result']['confidence']} alta com fonte degradada")
    if not bad:
        return Outcome("PASS", f"lat={lat}, rel={rel}")
    return Outcome("FAIL", "quality/source degradada preservada", "; ".join(bad))


def g10_no_nan(ev, ctx):
    hits = deep_scan_float_strings(ev.get("ai_payload") or {}) + \
        deep_scan_float_strings(ev.get("ai_result") or {})
    if hits:
        return Outcome("FAIL", "sem NaN/Inf", f"{len(hits)}", "; ".join(f"{p}" for p, _ in hits[:4]))
    return Outcome("PASS", "sem NaN/Inf no payload IA")


def g10_direction(ev, ctx):
    ar = ev.get("ai_result") or {}
    ap = ev.get("ai_payload") or {}
    sent = str(ar.get("sentiment", "")).lower()
    flow = ap.get("flow") or {}
    ob = ap.get("ob") or {}
    bad = []
    delta_imb = _num(flow.get("delta"))
    if delta_imb is not None and abs(delta_imb) > 0.3:
        if sent in ("bullish", "bearish") and "wait" not in str(ar.get("action", "")).lower():
            exp_side = "bullish" if delta_imb > 0 else "bearish"
            if sent != exp_side:
                bad.append(f"sentiment={sent} vs flow.delta={delta_imb} (sinal invertido?)")
    bias = ob.get("bias")
    if bias and bias == "SELL" and sent == "bullish":
        bad.append(f"sentiment={sent} vs ob.bias=SELL")
    if not bad:
        return Outcome("PASS", f"sentiment={sent}, action={ar.get('action')}")
    return Outcome("FAIL", "sinal/direcao nao invertido", "; ".join(bad))


register("ai-payload", "g10_anchor", lambda ev, ctx: g10_anchor(ev, ctx) or Outcome("PASS"), LOW, False)
register("ai-payload", "g10_price", g10_price, HIGH, True)
register("ai-payload", "g10_flow", g10_flow, HIGH, True)
register("ai-payload", "g10_ob", g10_ob, MEDIUM, False)
register("ai-payload", "g10_sr", g10_sr, MEDIUM, False)
register("ai-payload", "g10_quality", g10_quality, HIGH, False)
register("ai-payload", "g10_no_nan", g10_no_nan, HIGH, False)
register("ai-payload", "g10_direction", g10_direction, HIGH, True)


# =============================================================================
# EXECUCAO
# =============================================================================

def run_audit(events, start_ms, end_ms, sample_ai=None, include_pre_filter=False):
    at_events = []
    ai_events = []
    other_counts = {}
    seen = {}
    for ev in events:
        t = ev.get("tipo_evento")
        if include_pre_filter or is_new_window(ev, start_ms, end_ms):
            # dedupe por (tipo_evento, epoch_ms): DB (payload completo) tem prioridade
            # sobre JSONL (trimmed_by_guardian); mesma janela pode aparecer nos dois.
            key = (t, ev.get("epoch_ms") or ev.get("timestamp_ms"))
            prev = seen.get(key)
            if prev is not None:
                if len(ev) > len(prev):
                    seen[key] = ev
                continue
            seen[key] = ev
    for ev in seen.values():
        t = ev.get("tipo_evento")
        if t == EVENT_AT:
            at_events.append(ev)
        elif t == EVENT_AI:
            ai_events.append(ev)
        else:
            other_counts[t] = other_counts.get(t, 0) + 1

    at_events.sort(key=lambda e: e.get("epoch_ms") or e.get("timestamp_ms") or 0)
    ai_events.sort(key=lambda e: e.get("epoch_ms") or e.get("timestamp_ms") or 0)

    at_index = build_at_index(at_events)

    rows = []
    prev_epoch = None
    hist_flow_btc = []
    for ev in at_events:
        ctx = {"prev_epoch_ms": prev_epoch, "at_index": at_index, "hist_flow_btc": hist_flow_btc}
        of = (ev.get("fluxo_continuo") or {}).get("order_flow") or {}
        if of.get("buy_volume_btc") is not None:
            hist_flow_btc.append((ev.get("epoch_ms"), _num(of.get("buy_volume_btc")) or 0,
                                  _num(of.get("sell_volume_btc")) or 0))
        for inv in INVARIANTS:
            if inv.group == "ai-payload":
                continue
            try:
                outcome = inv.fn(ev, ctx)
            except Exception as exc:
                outcome = Outcome("FAIL", detail=f"excecao no auditor: {exc}")
            if outcome is None:
                outcome = Outcome("SKIP", detail="retorno None")
            outcome.inv = inv
            rows.append(_row(ev, inv, outcome))
        prev_epoch = ev.get("epoch_ms") or prev_epoch

    ai_sample = ai_events
    if sample_ai and len(ai_sample) > sample_ai:
        step = len(ai_sample) / sample_ai
        ai_sample = [ai_sample[int(i * step)] for i in range(sample_ai)]
    for ev in ai_sample:
        ctx = {"at_index": at_index}
        for inv in INVARIANTS:
            if inv.group != "ai-payload":
                continue
            try:
                outcome = inv.fn(ev, ctx)
            except Exception as exc:
                outcome = Outcome("FAIL", detail=f"excecao no auditor: {exc}")
            outcome.inv = inv
            rows.append(_row(ev, inv, outcome))

    return rows, len(at_events), len(ai_events), other_counts


def _row(ev, inv, outcome):
    w = ev.get("janela_numero") or ev.get("anchor_window_id") or \
        ev.get("epoch_ms") or ev.get("timestamp_ms")
    return {
        "window": str(w),
        "invariant": inv.id,
        "group": inv.group,
        "name": inv.name,
        "status": outcome.status,
        "expected": _fmt(outcome.expected),
        "actual": _fmt(outcome.actual),
        "difference": _fmt(outcome.difference),
        "severity": inv.severity if outcome.status == "FAIL" else "",
        "critical_math": inv.critical_math,
        "detail": outcome.detail,
    }


def scorecard(rows):
    groups = {}
    for r in rows:
        g = r["group"]
        groups.setdefault(g, {"PASS": 0, "FAIL": 0, "SKIP": 0, "NOT_APPLICABLE": 0,
                              "NOT_COMPARABLE": 0, "total": 0})
        st = r["status"]
        groups[g][st] = groups[g].get(st, 0) + 1
        groups[g]["total"] += 1
    return groups


def print_report(rows, n_at, n_ai, other_counts, start_ms, end_ms, groups):
    print("=" * 100)
    print("DATA ACCEPTANCE SCORECARD - AUDITORIA DE INVARIANTES (dados novos)")
    print("=" * 100)
    print(f"Janelas ANALYSIS_TRIGGER auditadas: {n_at}  |  AI_ANALYSIS: {n_ai}"
          f"  |  outros eventos (Exaustão/Absorção/Alerta): {sum(other_counts.values())}")
    print(f"Filtro temporal: start_ms={start_ms} end_ms={end_ms}")
    print("-" * 100)
    order = ["volume/flow", "sector-flow", "orderbook", "temporal", "value-profile",
             "sr", "pivots", "derivatives", "macro", "ai-payload"]
    labels = {"volume/flow": "Flow", "sector-flow": "Sector Flow", "orderbook": "Orderbook",
              "temporal": "Temporal", "value-profile": "VP", "sr": "S/R", "pivots": "Pivots",
              "derivatives": "Derivatives", "macro": "Macro", "ai-payload": "AI Payload"}
    for g in order:
        s = groups.get(g, {})
        tot = s.get("total", 0)
        p = s.get("PASS", 0)
        print(f"{labels[g]:<14}: {p}/{tot} PASS"
              f"  (FAIL={s.get('FAIL', 0)}, SKIP={s.get('SKIP', 0)},"
              f" NA={s.get('NOT_APPLICABLE', 0)}, NC={s.get('NOT_COMPARABLE', 0)})")

    fails = [r for r in rows if r["status"] == "FAIL"]
    sev_count = {"BLOCKER": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for r in fails:
        sev_count[r["severity"]] = sev_count.get(r["severity"], 0) + 1
    print("-" * 100)
    print(f"BLOCKERS : {sev_count['BLOCKER']}")
    print(f"HIGH     : {sev_count['HIGH']}")
    print(f"MEDIUM   : {sev_count['MEDIUM']}")
    print(f"LOW      : {sev_count['LOW']}")

    crit = [r for r in rows if r.get("critical_math")]
    crit_ok = [r for r in crit if r["status"] == "PASS"]
    crit_eval = [r for r in crit if r["status"] in ("PASS", "FAIL")]
    rate = (len(crit_ok) / len(crit_eval)) * 100 if crit_eval else 0
    print("-" * 100)
    print(f"Invariantes matematicos criticos: {len(crit_ok)}/{len(crit_eval)} PASS "
          f"({rate:.2f}%)  [alvo >= 99%]")
    blockers = [r for r in fails if r["severity"] == "BLOCKER"]
    go = (len(blockers) == 0) and (rate >= 99.0)
    print(f"GO/NO-GO: {'GO (paper/shadow extensivo)' if go else 'NO-GO -> avaliar antes de go-live'}")
    if n_at < 30:
        print(f"ATENCAO: menos de 30 janelas novas ({n_at}). Amostra insuficiente para conclusao firme.")
    print("=" * 100)

    if fails:
        print("\n----- FALHAS (evidencia) -----")
        for r in fails:
            print(f"[{r['severity']}] {r['invariant']} | janela={r['window']}")
            print(f"      expected={r['expected']} actual={r['actual']}"
                  f" diff={r['difference']}")
            if r["detail"]:
                print(f"      detail: {r['detail']}")
    else:
        print("\nSem falhas registradas.")

    print("\n----- RESUMO POR INVARIANTE -----")
    by_inv = {}
    for r in rows:
        by_inv.setdefault(r["invariant"], {}).setdefault(r["status"], 0)
        by_inv[r["invariant"]][r["status"]] = by_inv[r["invariant"]][r["status"]] + 1
    for inv in sorted(by_inv):
        s = by_inv[inv]
        print(f"  {inv:<42} {json.dumps(s, ensure_ascii=False)}")


def main():
    ap = argparse.ArgumentParser(description="Auditor de invariantes de dados live (shadow)")
    ap.add_argument("--db", default=str(REPO / "dados" / "trading_bot.db"))
    ap.add_argument("--jsonl", default=str(REPO / "dados" / "eventos_fluxo.jsonl"))
    ap.add_argument("--meta", default=str(REPO / "logs" / "shadow_run_meta.json"))
    ap.add_argument("--all", action="store_true", help="auditar TODOS os eventos (sem filtro temporal)")
    ap.add_argument("--start-ms", type=int, default=None)
    ap.add_argument("--end-ms", type=int, default=None)
    ap.add_argument("--sample-ai", type=int, default=None,
                    help="numero maximo de janelas AI_ANALYSIS auditadas (default: todas)")
    ap.add_argument("--report", default=str(REPO / "logs" / "audit_live_data_invariants_report.json"))
    args = ap.parse_args()

    events = load_events(args.db)
    events += load_jsonl(args.jsonl)

    start_ms, end_ms = window_bounds(args.meta, args.start_ms, args.end_ms)
    if args.all:
        start_ms = end_ms = None

    rows, n_at, n_ai, other = run_audit(events, start_ms, end_ms,
                                        sample_ai=args.sample_ai,
                                        include_pre_filter=args.all)
    groups = scorecard(rows)
    print_report(rows, n_at, n_ai, other, start_ms, end_ms, groups)

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    report = {
        "start_ms": start_ms,
        "end_ms": end_ms,
        "windows_at": n_at,
        "windows_ai": n_ai,
        "rows": rows,
        "groups": groups,
    }
    Path(args.report).write_text(json.dumps(report, indent=1, ensure_ascii=False),
                                 encoding="utf-8")
    print(f"\nRelatorio JSON: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())