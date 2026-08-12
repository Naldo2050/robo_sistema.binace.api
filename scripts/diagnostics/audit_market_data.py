# -*- coding: utf-8 -*-
"""
AUDITOR OFFLINE DOS DADOS REAIS DO MERCADO.

Ferramenta rápida, offline e independente para auditagem de integridade
e invariantes dos dados coletados pelo robô (dados/trading_bot.db, dados/eventos_fluxo.jsonl).

NÃO acessa Binance.
NÃO acessa internet.
NÃO chama IA.
NÃO inicia WebSocket.
NÃO executa trades.
NÃO altera código de produção.
"""

import argparse
import json
import math
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# =============================================================================
# SEVERIDADES & STATUS
# =============================================================================
BLOCKER = "BLOCKER"
HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"
NOT_COMPARABLE = "NOT_COMPARABLE"

DEGRADED_SOURCES = {
    "stale", "fallback_rest", "circuit_open", "external",
    "unknown", "error", "emergency", "fallback"
}


# =============================================================================
# FUNÇÃO CENTRAL DE TOLERÂNCIA
# =============================================================================
def is_close(a, b, rel_tol=0.01, abs_tol=1e-5):
    """
    Verifica se a e b são aproximadamente iguais.
    Suporta float, int e strings numéricas.
    Retorna True/False. Se um dos valores for inválido/None/NaN/Inf, retorna False.
    """
    if a is None or b is None:
        return False
    try:
        fa = float(a)
        fb = float(b)
    except (TypeError, ValueError):
        return False

    if math.isnan(fa) or math.isnan(fb) or math.isinf(fa) or math.isinf(fb):
        return False

    diff = abs(fa - fb)
    if diff <= abs_tol:
        return True

    denom = max(abs(fa), abs(fb))
    if denom == 0:
        return True

    return (diff / denom) <= rel_tol


def safe_float(val):
    """Converte valor para float seguro. Retorna None se None/NaN/Inf/inválido."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def is_nan_or_inf(val):
    """Retorna True se val é float NaN ou Inf ou string rep 'nan'/'inf'."""
    if val is None:
        return False
    if isinstance(val, float):
        return math.isnan(val) or math.isinf(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("nan", "-nan", "+nan", "inf", "-inf", "+inf", "infinity", "-infinity"):
            return True
    return False


# =============================================================================
# CARREGAMENTO DE DADOS
# =============================================================================
def load_sqlite_events(db_path, symbol_filter=None, event_type_filter=None, since_id=None, since_epoch_ms=None):
    """Carrega eventos do banco SQLite."""
    if not os.path.exists(db_path):
        return []
    
    events = []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        query = "SELECT id, timestamp_ms, event_type, symbol, window_id, payload, created_at FROM events WHERE 1=1"
        params = []

        if symbol_filter:
            query += " AND symbol = ?"
            params.append(symbol_filter)
        if event_type_filter:
            query += " AND event_type = ?"
            params.append(event_type_filter)
        if since_id is not None:
            query += " AND id >= ?"
            params.append(since_id)
        if since_epoch_ms is not None:
            query += " AND timestamp_ms >= ?"
            params.append(since_epoch_ms)

        query += " ORDER BY id ASC"
        cursor.execute(query, params)
        rows = cursor.fetchall()

        for row in rows:
            rec_id, ts_ms, ev_type, sym, win_id, payload_raw, created_at = row
            try:
                payload = json.loads(payload_raw) if isinstance(payload_raw, str) else (payload_raw or {})
            except Exception:
                payload = {}

            payload["_source_db_id"] = rec_id
            payload["_source"] = "sqlite"
            if "event_id" not in payload and win_id:
                payload["event_id"] = str(win_id)
            if "tipo_evento" not in payload:
                payload["tipo_evento"] = ev_type
            if "symbol" not in payload:
                payload["symbol"] = sym
            if "epoch_ms" not in payload:
                payload["epoch_ms"] = ts_ms

            events.append(payload)
        conn.close()
    except Exception as e:
        sys.stderr.write(f"Erro ao carregar SQLite ({db_path}): {e}\n")
    return events


def load_jsonl_events(jsonl_path, symbol_filter=None, event_type_filter=None, since_epoch_ms=None):
    """Carrega eventos do arquivo JSONL."""
    if not os.path.exists(jsonl_path):
        return []
    
    events = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue

                payload["_source_line"] = line_no
                payload["_source"] = "jsonl"

                ev_type = payload.get("tipo_evento") or payload.get("event_type")
                sym = payload.get("symbol") or payload.get("ativo")
                ep_ms = payload.get("epoch_ms") or payload.get("window_id") or payload.get("timestamp_ms")

                if symbol_filter and sym != symbol_filter:
                    continue
                if event_type_filter and ev_type != event_type_filter:
                    continue
                if since_epoch_ms is not None and ep_ms is not None and safe_float(ep_ms) < since_epoch_ms:
                    continue

                events.append(payload)
    except Exception as e:
        sys.stderr.write(f"Erro ao carregar JSONL ({jsonl_path}): {e}\n")
    return events


# =============================================================================
# VERIFICAÇÕES INDIVIDUAIS (12 CATEGORIAS)
# =============================================================================

def make_check_result(window, event_id, check_name, status, severity, expected, actual, difference=None, details=""):
    """Cria dicionário padronizado de verificação."""
    diff_str = None
    if difference is not None:
        diff_str = str(difference)
    elif expected is not None and actual is not None:
        fe, fa = safe_float(expected), safe_float(actual)
        if fe is not None and fa is not None:
            diff_str = f"{abs(fe - fa):.6g}"

    return {
        "window": window,
        "event_id": str(event_id) if event_id is not None else "N/A",
        "check": check_name,
        "status": status,
        "severity": severity if status == FAIL else "NONE",
        "expected": expected,
        "actual": actual,
        "difference": diff_str,
        "details": details
    }


def check_flow(ev):
    """1 — FLOW Checks"""
    results = []
    win = ev.get("janela_numero") or ev.get("window_id") or ev.get("epoch_ms")
    eid = ev.get("event_id") or ev.get("_source_db_id")

    ctx = ev.get("contextual_snapshot") or ev.get("enriched_snapshot") or ev
    vol_tot = safe_float(ctx.get("volume_total") or ctx.get("volume_total_btc"))
    vol_compra = safe_float(ctx.get("volume_compra") or ctx.get("volume_compra_btc"))
    vol_venda = safe_float(ctx.get("volume_venda") or ctx.get("volume_venda_btc"))
    delta = safe_float(ctx.get("delta") or ctx.get("delta_fechamento"))

    # volume_total ≈ volume_compra + volume_venda
    if vol_tot is not None and vol_compra is not None and vol_venda is not None:
        exp_sum = vol_compra + vol_venda
        if is_close(exp_sum, vol_tot, rel_tol=0.01, abs_tol=1e-4):
            results.append(make_check_result(win, eid, "flow.volume_conservation", PASS, LOW, exp_sum, vol_tot))
        else:
            results.append(make_check_result(win, eid, "flow.volume_conservation", FAIL, BLOCKER, exp_sum, vol_tot,
                                              details="volume_total != volume_compra + volume_venda"))

    # delta ≈ volume_compra - volume_venda
    if delta is not None and vol_compra is not None and vol_venda is not None:
        exp_delta = vol_compra - vol_venda
        if is_close(exp_delta, delta, rel_tol=0.01, abs_tol=1e-4):
            results.append(make_check_result(win, eid, "flow.delta_conservation", PASS, LOW, exp_delta, delta))
        else:
            results.append(make_check_result(win, eid, "flow.delta_conservation", FAIL, BLOCKER, exp_delta, delta,
                                              details="delta != volume_compra - volume_venda"))

    # order_flow.total_volume_btc ≈ buy_volume_btc + sell_volume_btc
    of = ev.get("fluxo_continuo", {}).get("order_flow") or ev.get("order_flow") or {}
    tot_btc = safe_float(of.get("total_volume_btc"))
    buy_btc = safe_float(of.get("buy_volume_btc"))
    sell_btc = safe_float(of.get("sell_volume_btc"))
    if tot_btc is not None and buy_btc is not None and sell_btc is not None:
        exp_btc = buy_btc + sell_btc
        if is_close(exp_btc, tot_btc, rel_tol=0.01, abs_tol=1e-4):
            results.append(make_check_result(win, eid, "flow.btc_volume_conservation", PASS, LOW, exp_btc, tot_btc))
        else:
            results.append(make_check_result(win, eid, "flow.btc_volume_conservation", FAIL, BLOCKER, exp_btc, tot_btc,
                                              details="order_flow.total_volume_btc != buy_btc + sell_btc"))

    # flow_imbalance esperado: (buy_btc - sell_btc) / (buy_btc + sell_btc)
    flow_imb = safe_float(of.get("flow_imbalance"))
    if flow_imb is not None and buy_btc is not None and sell_btc is not None and (buy_btc + sell_btc) > 0:
        exp_imb = (buy_btc - sell_btc) / (buy_btc + sell_btc)
        if is_close(exp_imb, flow_imb, rel_tol=0.02, abs_tol=1e-3):
            results.append(make_check_result(win, eid, "flow.imbalance_formula", PASS, LOW, exp_imb, flow_imb))
        else:
            results.append(make_check_result(win, eid, "flow.imbalance_formula", FAIL, BLOCKER, exp_imb, flow_imb,
                                              details="flow_imbalance != (buy_btc-sell_btc)/(buy_btc+sell_btc)"))

    # aggressive_buy_pct + aggressive_sell_pct ≈ 100
    agg_buy = safe_float(of.get("aggressive_buy_pct"))
    agg_sell = safe_float(of.get("aggressive_sell_pct"))
    if agg_buy is not None and agg_sell is not None:
        if (tot_btc or vol_tot or 1) == 0:
            results.append(make_check_result(win, eid, "flow.aggressive_pct_sum", SKIP, LOW, 100.0, 0,
                                              details="Volume total zero, pcts ignorados"))
        else:
            exp_pct_sum = 100.0
            act_pct_sum = agg_buy + agg_sell
            if is_close(exp_pct_sum, act_pct_sum, rel_tol=0.01, abs_tol=0.1):
                results.append(make_check_result(win, eid, "flow.aggressive_pct_sum", PASS, LOW, exp_pct_sum, act_pct_sum))
            else:
                results.append(make_check_result(win, eid, "flow.aggressive_pct_sum", FAIL, MEDIUM, exp_pct_sum, act_pct_sum,
                                                  details="aggressive_buy_pct + aggressive_sell_pct != 100"))

    # buy_sell_ratio.current ≈ buy / sell quando sell > 0
    bsr = of.get("buy_sell_ratio") or {}
    curr_r = safe_float(bsr.get("current") if isinstance(bsr, dict) else None)
    if curr_r is None and isinstance(bsr, dict):
        curr_r = safe_float(bsr.get("ratios", {}).get("current") or bsr.get("buy_sell_ratio"))

    bs_buy = safe_float(bsr.get("buy_volume")) if isinstance(bsr, dict) else None
    bs_sell = safe_float(bsr.get("sell_volume")) if isinstance(bsr, dict) else None
    if bs_buy is None:
        bs_buy = buy_btc or vol_compra
    if bs_sell is None:
        bs_sell = sell_btc or vol_venda

    if curr_r is not None and bs_buy is not None and bs_sell is not None and bs_sell > 0:
        exp_bsr = bs_buy / bs_sell
        if is_close(exp_bsr, curr_r, rel_tol=0.05, abs_tol=1e-2):
            results.append(make_check_result(win, eid, "flow.buy_sell_ratio", PASS, LOW, exp_bsr, curr_r))
        else:
            results.append(make_check_result(win, eid, "flow.buy_sell_ratio", FAIL, MEDIUM, exp_bsr, curr_r,
                                              details="buy_sell_ratio != buy_volume / sell_volume"))

    # net_flow_1m: validar conforme CONTRATO ATUAL (buy_volume - sell_volume USD quando janela == 1m)
    nf_1m = safe_float(of.get("net_flow_1m"))
    buy_usd = safe_float(of.get("buy_volume")) or safe_float(ev.get("buy_notional_usdt"))
    sell_usd = safe_float(of.get("sell_volume")) or safe_float(ev.get("sell_notional_usdt"))
    cw = of.get("computation_window_min") or 1

    if nf_1m is not None:
        if cw not in (1, "1", 1.0, None):
            results.append(make_check_result(win, eid, "flow.net_flow_1m", SKIP, LOW, None, nf_1m,
                                              details=f"computation_window_min={cw} != 1m (universo temporal diferente)"))
        elif buy_usd is not None and sell_usd is not None:
            exp_nf = buy_usd - sell_usd
            if is_close(exp_nf, nf_1m, rel_tol=0.02, abs_tol=10.0):
                results.append(make_check_result(win, eid, "flow.net_flow_1m", PASS, LOW, exp_nf, nf_1m))
            else:
                results.append(make_check_result(win, eid, "flow.net_flow_1m", FAIL, HIGH, exp_nf, nf_1m,
                                                  details="net_flow_1m != buy_volume - sell_volume (USD)"))

    return results


def check_sector_flow(ev):
    """2 — SECTOR FLOW Checks"""
    results = []
    win = ev.get("janela_numero") or ev.get("window_id") or ev.get("epoch_ms")
    eid = ev.get("event_id") or ev.get("_source_db_id")

    fc = ev.get("fluxo_continuo") or {}
    sf = fc.get("sector_flow") or {}
    retail = sf.get("retail") or {}
    mid = sf.get("mid") or {}
    whale = sf.get("whale") or {}

    if not retail or not mid or not whale:
        return results

    r_buy, r_sell, r_delta = safe_float(retail.get("buy")), safe_float(retail.get("sell")), safe_float(retail.get("delta"))
    m_buy, m_sell, m_delta = safe_float(mid.get("buy")), safe_float(mid.get("sell")), safe_float(mid.get("delta"))
    w_buy, w_sell, w_delta = safe_float(whale.get("buy")), safe_float(whale.get("sell")), safe_float(whale.get("delta"))

    # Regra Atual: sector_flow é acumulativo/contínuo desde a inicialização e não representa a janela isolada de 1m do order_flow.
    # Reconciliação direta de volume absoluto acumulado com janela 1m é sempre NOT_COMPARABLE / SKIP.
    of = fc.get("order_flow") or {}
    of_buy = safe_float(of.get("buy_volume_btc")) or safe_float(ev.get("volume_compra_btc"))
    tot_sec_buy = (r_buy or 0.0) + (m_buy or 0.0) + (w_buy or 0.0)

    if of_buy is not None:
        results.append(make_check_result(
            win, eid, "sector_flow.reconciliation", NOT_COMPARABLE, LOW, of_buy, tot_sec_buy,
            details="sector_flow is cumulative; order_flow is windowed"
        ))

    # Validações internas dos deltas setoriais (retail, mid, whale)
    if r_buy is not None and r_sell is not None and r_delta is not None:
        exp_r_d = r_buy - r_sell
        if is_close(exp_r_d, r_delta, rel_tol=0.01, abs_tol=1e-3):
            results.append(make_check_result(win, eid, "sector_flow.retail_delta_internal", PASS, LOW, exp_r_d, r_delta))
        else:
            results.append(make_check_result(win, eid, "sector_flow.retail_delta_internal", FAIL, HIGH, exp_r_d, r_delta,
                                              details="retail.delta != retail.buy - retail.sell"))

    if m_buy is not None and m_sell is not None and m_delta is not None:
        exp_m_d = m_buy - m_sell
        if is_close(exp_m_d, m_delta, rel_tol=0.01, abs_tol=1e-3):
            results.append(make_check_result(win, eid, "sector_flow.mid_delta_internal", PASS, LOW, exp_m_d, m_delta))
        else:
            results.append(make_check_result(win, eid, "sector_flow.mid_delta_internal", FAIL, HIGH, exp_m_d, m_delta,
                                              details="mid.delta != mid.buy - mid.sell"))

    if w_buy is not None and w_sell is not None and w_delta is not None:
        exp_w_d = w_buy - w_sell
        if is_close(exp_w_d, w_delta, rel_tol=0.01, abs_tol=1e-3):
            results.append(make_check_result(win, eid, "sector_flow.whale_delta_internal", PASS, LOW, exp_w_d, w_delta))
        else:
            results.append(make_check_result(win, eid, "sector_flow.whale_delta_internal", FAIL, HIGH, exp_w_d, w_delta,
                                              details="whale.delta != whale.buy - whale.sell"))

    # Checar se a soma dos deltas setoriais bate com CVD
    cvd_val = safe_float(fc.get("cvd"))
    if cvd_val is not None and r_delta is not None and m_delta is not None and w_delta is not None:
        exp_cvd = r_delta + m_delta + w_delta
        if is_close(exp_cvd, cvd_val, rel_tol=0.01, abs_tol=1e-3):
            results.append(make_check_result(win, eid, "sector_flow.cvd_sum_internal", PASS, LOW, exp_cvd, cvd_val))
        else:
            results.append(make_check_result(win, eid, "sector_flow.cvd_sum_internal", FAIL, HIGH, exp_cvd, cvd_val,
                                              details="retail.delta + mid.delta + whale.delta != cvd"))

    # Checar se trade qty válido ficou fora de bucket por teto whale
    data_q = ev.get("data_quality") or {}
    inv_trades = safe_float(data_q.get("invalid_trades"))
    if inv_trades is not None and inv_trades > 0:
        results.append(make_check_result(win, eid, "sector_flow.invalid_trades", FAIL, HIGH, 0, inv_trades,
                                          details="Trades considerados inválidos/descartados por estouro de teto"))
    elif inv_trades is not None:
        results.append(make_check_result(win, eid, "sector_flow.invalid_trades", PASS, LOW, 0, 0))

    return results


def check_orderbook(ev):
    """3 — ORDERBOOK Checks"""
    results = []
    win = ev.get("janela_numero") or ev.get("window_id") or ev.get("epoch_ms")
    eid = ev.get("event_id") or ev.get("_source_db_id")

    ob = ev.get("orderbook_data") or ev.get("order_book") or {}
    bid = safe_float(ob.get("bid") or ev.get("bid"))
    ask = safe_float(ob.get("ask") or ev.get("ask"))
    mid = safe_float(ob.get("mid"))
    spread = safe_float(ob.get("spread"))
    spread_bps = safe_float(ob.get("spread_bps"))
    bid_depth = safe_float(ob.get("bid_depth_usd"))
    ask_depth = safe_float(ob.get("ask_depth_usd"))
    imb = safe_float(ob.get("imbalance"))
    vol_ratio = safe_float(ob.get("volume_ratio"))

    # bid < ask
    if bid is not None and ask is not None:
        if bid < ask:
            results.append(make_check_result(win, eid, "orderbook.bid_less_than_ask", PASS, LOW, f"<{ask}", bid))
        else:
            results.append(make_check_result(win, eid, "orderbook.bid_less_than_ask", FAIL, BLOCKER, f"<{ask}", bid,
                                              details="Orderbook cruzado: bid >= ask"))

    # mid ≈ (bid + ask)/2
    if mid is not None and bid is not None and ask is not None:
        exp_mid = (bid + ask) / 2.0
        if is_close(exp_mid, mid, rel_tol=0.001, abs_tol=1e-3):
            results.append(make_check_result(win, eid, "orderbook.mid_formula", PASS, LOW, exp_mid, mid))
        else:
            results.append(make_check_result(win, eid, "orderbook.mid_formula", FAIL, HIGH, exp_mid, mid,
                                              details="mid != (bid + ask) / 2"))

    # spread ≈ ask - bid
    if spread is not None and bid is not None and ask is not None:
        exp_spread = ask - bid
        if is_close(exp_spread, spread, rel_tol=0.01, abs_tol=1e-4):
            results.append(make_check_result(win, eid, "orderbook.spread_formula", PASS, LOW, exp_spread, spread))
        else:
            results.append(make_check_result(win, eid, "orderbook.spread_formula", FAIL, HIGH, exp_spread, spread,
                                              details="spread != ask - bid"))

    # spread_bps ≈ (spread / mid) * 10000
    if spread_bps is not None and spread is not None and mid is not None and mid > 0:
        exp_bps = (spread / mid) * 10000.0
        if is_close(exp_bps, spread_bps, rel_tol=0.02, abs_tol=1e-2):
            results.append(make_check_result(win, eid, "orderbook.spread_bps_formula", PASS, LOW, exp_bps, spread_bps))
        else:
            results.append(make_check_result(win, eid, "orderbook.spread_bps_formula", FAIL, MEDIUM, exp_bps, spread_bps,
                                              details="spread_bps != (spread / mid) * 10000"))

    # depth >= 0
    if bid_depth is not None:
        if bid_depth >= 0:
            results.append(make_check_result(win, eid, "orderbook.bid_depth_non_negative", PASS, LOW, ">=0", bid_depth))
        else:
            results.append(make_check_result(win, eid, "orderbook.bid_depth_non_negative", FAIL, BLOCKER, ">=0", bid_depth,
                                              details="bid_depth_usd < 0"))

    if ask_depth is not None:
        if ask_depth >= 0:
            results.append(make_check_result(win, eid, "orderbook.ask_depth_non_negative", PASS, LOW, ">=0", ask_depth))
        else:
            results.append(make_check_result(win, eid, "orderbook.ask_depth_non_negative", FAIL, BLOCKER, ">=0", ask_depth,
                                              details="ask_depth_usd < 0"))

    # imbalance ≈ (bid_depth_usd - ask_depth_usd) / (bid_depth_usd + ask_depth_usd)
    if imb is not None and bid_depth is not None and ask_depth is not None and (bid_depth + ask_depth) > 0:
        exp_imb = (bid_depth - ask_depth) / (bid_depth + ask_depth)
        if is_close(exp_imb, imb, rel_tol=0.02, abs_tol=1e-3):
            results.append(make_check_result(win, eid, "orderbook.imbalance_formula", PASS, LOW, exp_imb, imb))
        else:
            results.append(make_check_result(win, eid, "orderbook.imbalance_formula", FAIL, HIGH, exp_imb, imb,
                                              details="imbalance != (bid_depth-ask_depth)/(bid_depth+ask_depth)"))

    # volume_ratio ≈ bid_depth / ask_depth
    if vol_ratio is not None and bid_depth is not None and ask_depth is not None and ask_depth > 0:
        exp_ratio = bid_depth / ask_depth
        if is_close(exp_ratio, vol_ratio, rel_tol=0.02, abs_tol=1e-2):
            results.append(make_check_result(win, eid, "orderbook.volume_ratio_formula", PASS, LOW, exp_ratio, vol_ratio))
        else:
            results.append(make_check_result(win, eid, "orderbook.volume_ratio_formula", FAIL, MEDIUM, exp_ratio, vol_ratio,
                                              details="volume_ratio != bid_depth / ask_depth"))

    # L1 / L5 / L10 / L25 depth validation
    ob_depth = ev.get("order_book_depth") or {}
    for level in ("L1", "L5", "L10", "L25"):
        lvl_data = ob_depth.get(level)
        if isinstance(lvl_data, dict):
            lbids = safe_float(lvl_data.get("bids"))
            lasks = safe_float(lvl_data.get("asks"))
            limb = safe_float(lvl_data.get("flow_imbalance") or lvl_data.get("imbalance"))

            if lbids is not None and lbids < 0:
                results.append(make_check_result(win, eid, f"orderbook.{level}_bids_valid", FAIL, BLOCKER, ">=0", lbids))
            if lasks is not None and lasks < 0:
                results.append(make_check_result(win, eid, f"orderbook.{level}_asks_valid", FAIL, BLOCKER, ">=0", lasks))
            if limb is not None:
                if -1.0001 <= limb <= 1.0001:
                    results.append(make_check_result(win, eid, f"orderbook.{level}_imbalance_range", PASS, LOW, "[-1, 1]", limb))
                else:
                    results.append(make_check_result(win, eid, f"orderbook.{level}_imbalance_range", FAIL, HIGH, "[-1, 1]", limb,
                                                      details=f"{level} imbalance fora do intervalo [-1, 1]"))

    return results


def check_temporal(ev):
    """4 — TEMPORAL Checks"""
    results = []
    win = ev.get("janela_numero") or ev.get("window_id") or ev.get("epoch_ms")
    eid = ev.get("event_id") or ev.get("_source_db_id")

    ep_ms = safe_float(ev.get("epoch_ms") or ev.get("timestamp_ms"))
    ts_utc = ev.get("timestamp_utc") or ev.get("timestamp")

    if ep_ms is not None and ts_utc and isinstance(ts_utc, str):
        try:
            # Parse ISO UTC string
            iso_str = ts_utc.replace("Z", "+00:00")
            dt = datetime.fromisoformat(iso_str)
            parsed_ms = dt.timestamp() * 1000.0
            if abs(parsed_ms - ep_ms) <= 1000.0:  # 1 seg precisão
                results.append(make_check_result(win, eid, "temporal.utc_epoch_alignment", PASS, LOW, ep_ms, parsed_ms))
            else:
                results.append(make_check_result(win, eid, "temporal.utc_epoch_alignment", FAIL, BLOCKER, ep_ms, parsed_ms,
                                                  details=f"epoch_ms ({ep_ms}) não corresponde a timestamp_utc ({ts_utc})"))
        except Exception as e:
            results.append(make_check_result(win, eid, "temporal.utc_epoch_alignment", FAIL, HIGH, "ISO format", ts_utc,
                                              details=f"Erro ao converter timestamp_utc: {e}"))

    # quality.latency.is_acceptable vs data_reliability.latency_acceptable
    q_lat = ev.get("quality", {}).get("latency", {})
    q_acc = q_lat.get("is_acceptable") if isinstance(q_lat, dict) else None

    dr = ev.get("data_reliability") or {}
    dr_acc = dr.get("latency_acceptable") if isinstance(dr, dict) else None

    if q_acc is not None and dr_acc is not None:
        q_bool = bool(q_acc)
        dr_bool = bool(dr_acc)
        if q_bool == dr_bool:
            results.append(make_check_result(win, eid, "temporal.latency_acceptance_agreement", PASS, LOW, q_bool, dr_bool))
        else:
            results.append(make_check_result(win, eid, "temporal.latency_acceptance_agreement", FAIL, BLOCKER, q_bool, dr_bool,
                                              details="quality.latency.is_acceptable e data_reliability.latency_acceptable discordam!"))

    return results


def check_source_quality(ev):
    """5 — SOURCE / QUALITY Checks"""
    results = []
    win = ev.get("janela_numero") or ev.get("window_id") or ev.get("epoch_ms")
    eid = ev.get("event_id") or ev.get("_source_db_id")

    src_obj = ev.get("source") or {}
    src_name = ""
    if isinstance(src_obj, dict):
        src_name = str(src_obj.get("exchange") or src_obj.get("type") or src_obj.get("name") or "").lower()
    elif isinstance(src_obj, str):
        src_name = src_obj.lower()

    data_src = str(ev.get("orderbook_data", {}).get("data_source") or ev.get("data_source") or "").lower()
    ob_qual = str(ev.get("orderbook_quality") or ev.get("quality_tier") or "").lower()

    # Se a fonte é degraded/stale/fallback, NÃO pode aparecer como "live"
    is_degraded = any(deg in src_name or deg in data_src for deg in DEGRADED_SOURCES)

    if is_degraded:
        if ob_qual == "live" or data_src == "live":
            results.append(make_check_result(win, eid, "source.degraded_not_marked_live", FAIL, BLOCKER, "not live", ob_qual or data_src,
                                              details=f"Origem degradada ({src_name}/{data_src}) foi incorretamente classificada como tier 'live'"))
        else:
            results.append(make_check_result(win, eid, "source.degraded_not_marked_live", PASS, LOW, "degraded/fallback", ob_qual or data_src))
    else:
        results.append(make_check_result(win, eid, "source.degraded_not_marked_live", PASS, LOW, "normal", ob_qual or data_src or "ok"))

    return results


def check_value_profile(ev):
    """6 — VALUE PROFILE Checks"""
    results = []
    win = ev.get("janela_numero") or ev.get("window_id") or ev.get("epoch_ms")
    eid = ev.get("event_id") or ev.get("_source_db_id")

    prof = ev.get("profile_analysis") or {}
    va_info = prof.get("va_volume_pct") or prof.get("volume_profile") or {}
    vp_status = str(va_info.get("status") or prof.get("status") or "").lower()

    # Checar NaN / Inf em tudo do profile_analysis
    for key, val in va_info.items():
        if is_nan_or_inf(val):
            results.append(make_check_result(win, eid, f"value_profile.no_nan_inf_{key}", FAIL, BLOCKER, "Finite number", str(val),
                                              details=f"NaN/Inf detectado no Value Profile ({key})"))

    va_pct = safe_float(va_info.get("value_area_volume_pct"))
    vol_va = safe_float(va_info.get("volume_in_va"))
    vol_tot = safe_float(va_info.get("total_volume") or ev.get("volume_total"))

    val_price = safe_float(ev.get("val") or ev.get("pivot_points", {}).get("daily", {}).get("val"))
    vah_price = safe_float(ev.get("vah") or ev.get("pivot_points", {}).get("daily", {}).get("vah"))
    poc_price = safe_float(ev.get("poc_price") or ev.get("pivot_points", {}).get("daily", {}).get("poc"))

    if vp_status == "success":
        # 0 <= value_area_volume_pct <= 100
        if va_pct is not None:
            if 0.0 <= va_pct <= 100.0:
                results.append(make_check_result(win, eid, "value_profile.va_pct_range", PASS, LOW, "[0, 100]", va_pct))
            else:
                results.append(make_check_result(win, eid, "value_profile.va_pct_range", FAIL, BLOCKER, "[0, 100]", va_pct,
                                                  details="value_area_volume_pct fora do intervalo [0, 100]"))

        # volume_in_va <= total_volume
        if vol_va is not None and vol_tot is not None and vol_tot > 0:
            if vol_va <= vol_tot * 1.001:
                results.append(make_check_result(win, eid, "value_profile.vol_in_va_le_total", PASS, LOW, f"<={vol_tot}", vol_va))
            else:
                results.append(make_check_result(win, eid, "value_profile.vol_in_va_le_total", FAIL, BLOCKER, f"<={vol_tot}", vol_va,
                                                  details="volume_in_va maior que total_volume"))

        # VAL <= VAH
        if val_price is not None and vah_price is not None:
            if val_price <= vah_price:
                results.append(make_check_result(win, eid, "value_profile.val_le_vah", PASS, LOW, f"<={vah_price}", val_price))
            else:
                results.append(make_check_result(win, eid, "value_profile.val_le_vah", FAIL, BLOCKER, f"<={vah_price}", val_price,
                                                  details="VAL > VAH no Value Profile"))

        # VAL <= POC <= VAH (se POC disponível no mesmo contrato VP)
        if val_price is not None and vah_price is not None and poc_price is not None:
            if val_price <= poc_price <= vah_price:
                results.append(make_check_result(win, eid, "value_profile.val_poc_vah_order", PASS, LOW, f"{val_price}<={poc_price}<={vah_price}", "ok"))
            else:
                # Pode ocorrer se POC for histórico de timeframe diferente, registramos como MEDIUM se não coincidir
                results.append(make_check_result(win, eid, "value_profile.val_poc_vah_order", FAIL, MEDIUM, f"[{val_price}, {vah_price}]", poc_price,
                                                  details="POC fora do intervalo [VAL, VAH]"))

    elif vp_status == "error":
        # Não aceitar pct fabricado de 70%
        if va_pct == 70 or va_pct == 70.0:
            results.append(make_check_result(win, eid, "value_profile.fabricated_pct_on_error", FAIL, BLOCKER, "Not 70%", va_pct,
                                              details="Status error com percentual fabricado de 70%"))
        else:
            results.append(make_check_result(win, eid, "value_profile.fabricated_pct_on_error", PASS, LOW, "No fabricated 70%", va_pct))

    elif vp_status == "insufficient_data":
        comp_sig = va_info.get("compression_signal")
        brk_risk = str(va_info.get("breakout_risk") or "").upper()

        if comp_sig is True or comp_sig == 1:
            results.append(make_check_result(win, eid, "value_profile.insufficient_data_compression", FAIL, HIGH, False, comp_sig,
                                              details="insufficient_data não pode gerar compression_signal=True"))
        if brk_risk in ("HIGH", "VERY_HIGH"):
            results.append(make_check_result(win, eid, "value_profile.insufficient_data_breakout_risk", FAIL, HIGH, "LOW/MEDIUM", brk_risk,
                                              details=f"insufficient_data com breakout_risk={brk_risk}"))

    return results


def check_support_resistance(ev):
    """7 — SUPPORT / RESISTANCE Checks"""
    results = []
    win = ev.get("janela_numero") or ev.get("window_id") or ev.get("epoch_ms")
    eid = ev.get("event_id") or ev.get("_source_db_id")

    curr_price = safe_float(ev.get("preco_fechamento") or ev.get("last_price") or ev.get("price") or ev.get("tick_context_out", {}).get("last_price"))
    imm_sup = ev.get("immediate_support") or []
    imm_res = ev.get("immediate_resistance") or []
    sup_str = ev.get("support_strength") or []
    res_str = ev.get("resistance_strength") or []

    if curr_price is not None:
        # immediate_support <= current_price
        for idx, sup in enumerate(imm_sup):
            fsup = safe_float(sup)
            if fsup is not None:
                if fsup <= curr_price * 1.0001:  # margem float
                    results.append(make_check_result(win, eid, f"sr.immediate_support_{idx}", PASS, LOW, f"<={curr_price}", fsup))
                else:
                    results.append(make_check_result(win, eid, f"sr.immediate_support_{idx}", FAIL, BLOCKER, f"<={curr_price}", fsup,
                                                      details=f"Suporte de execução ({fsup}) acima do preço atual ({curr_price})"))

        # immediate_resistance >= current_price
        for idx, res in enumerate(imm_res):
            fres = safe_float(res)
            if fres is not None:
                if fres >= curr_price * 0.9999:
                    results.append(make_check_result(win, eid, f"sr.immediate_resistance_{idx}", PASS, LOW, f">={curr_price}", fres))
                else:
                    results.append(make_check_result(win, eid, f"sr.immediate_resistance_{idx}", FAIL, BLOCKER, f">={curr_price}", fres,
                                                      details=f"Resistência de execução ({fres}) abaixo do preço atual ({curr_price})"))

    # Defense zones: center vs price
    sr_an = ev.get("sr_analysis") or {}
    def_zones = sr_an.get("defense_zones") or {}
    buy_def = def_zones.get("buy_defense") or []
    sell_def = def_zones.get("sell_defense") or []

    if curr_price is not None:
        for idx, zone in enumerate(buy_def):
            ctr = safe_float(zone.get("center"))
            if ctr is not None:
                if ctr <= curr_price * 1.001:
                    results.append(make_check_result(win, eid, f"sr.buy_defense_center_{idx}", PASS, LOW, f"<={curr_price}", ctr))
                else:
                    results.append(make_check_result(win, eid, f"sr.buy_defense_center_{idx}", FAIL, BLOCKER, f"<={curr_price}", ctr,
                                                      details=f"Buy defense center ({ctr}) acima do preço ({curr_price})"))

        for idx, zone in enumerate(sell_def):
            ctr = safe_float(zone.get("center"))
            if ctr is not None:
                if ctr >= curr_price * 0.999:
                    results.append(make_check_result(win, eid, f"sr.sell_defense_center_{idx}", PASS, LOW, f">={curr_price}", ctr))
                else:
                    results.append(make_check_result(win, eid, f"sr.sell_defense_center_{idx}", FAIL, BLOCKER, f">={curr_price}", ctr,
                                                      details=f"Sell defense center ({ctr}) abaixo do preço ({curr_price})"))

    # 0 <= strength <= 100
    all_strengths = (sup_str or []) + (res_str or [])
    for idx, st in enumerate(all_strengths):
        fst = safe_float(st)
        if fst is not None:
            if 0.0 <= fst <= 100.0:
                results.append(make_check_result(win, eid, f"sr.strength_range_{idx}", PASS, LOW, "[0, 100]", fst))
            else:
                results.append(make_check_result(win, eid, f"sr.strength_range_{idx}", FAIL, HIGH, "[0, 100]", fst,
                                                  details="Força S/R fora do intervalo [0, 100]"))

    # signals_in_zone >= source_count
    all_zones = (buy_def or []) + (sell_def or [])
    for idx, zone in enumerate(all_zones):
        sig_cnt = safe_float(zone.get("signals_in_zone"))
        src_cnt = safe_float(zone.get("source_count"))
        if sig_cnt is not None and src_cnt is not None:
            if sig_cnt >= src_cnt:
                results.append(make_check_result(win, eid, f"sr.signals_ge_sources_{idx}", PASS, LOW, f">={src_cnt}", sig_cnt))
            else:
                results.append(make_check_result(win, eid, f"sr.signals_ge_sources_{idx}", FAIL, MEDIUM, f">={src_cnt}", sig_cnt,
                                                  details="signals_in_zone menor que source_count"))

    return results


def check_pivots(ev):
    """8 — PIVOTS Checks"""
    results = []
    win = ev.get("janela_numero") or ev.get("window_id") or ev.get("epoch_ms")
    eid = ev.get("event_id") or ev.get("_source_db_id")

    pivots = ev.get("pivots") or ev.get("pivot_points") or {}
    daily_pivots = pivots.get("daily") or {}

    high = safe_float(daily_pivots.get("high"))
    low = safe_float(daily_pivots.get("low"))
    close = safe_float(daily_pivots.get("close"))

    p_val = safe_float(daily_pivots.get("pivot"))
    r1_val = safe_float(daily_pivots.get("r1"))
    s1_val = safe_float(daily_pivots.get("s1"))
    r2_val = safe_float(daily_pivots.get("r2"))
    s2_val = safe_float(daily_pivots.get("s2"))

    if high is not None and low is not None and close is not None:
        exp_p = (high + low + close) / 3.0
        exp_r1 = (2 * exp_p) - low
        exp_s1 = (2 * exp_p) - high
        exp_r2 = exp_p + (high - low)
        exp_s2 = exp_p - (high - low)

        if p_val is not None:
            if is_close(exp_p, p_val, rel_tol=0.005, abs_tol=1e-2):
                results.append(make_check_result(win, eid, "pivots.classic_pivot", PASS, LOW, exp_p, p_val))
            else:
                results.append(make_check_result(win, eid, "pivots.classic_pivot", FAIL, HIGH, exp_p, p_val,
                                                  details="Pivô central P != (H+L+C)/3"))

        if r1_val is not None:
            if is_close(exp_r1, r1_val, rel_tol=0.005, abs_tol=1e-2):
                results.append(make_check_result(win, eid, "pivots.classic_r1", PASS, LOW, exp_r1, r1_val))
            else:
                results.append(make_check_result(win, eid, "pivots.classic_r1", FAIL, HIGH, exp_r1, r1_val,
                                                  details="R1 != 2P - L"))

        if s1_val is not None:
            if is_close(exp_s1, s1_val, rel_tol=0.005, abs_tol=1e-2):
                results.append(make_check_result(win, eid, "pivots.classic_s1", PASS, LOW, exp_s1, s1_val))
            else:
                results.append(make_check_result(win, eid, "pivots.classic_s1", FAIL, HIGH, exp_s1, s1_val,
                                                  details="S1 != 2P - H"))

        if r2_val is not None:
            if is_close(exp_r2, r2_val, rel_tol=0.005, abs_tol=1e-2):
                results.append(make_check_result(win, eid, "pivots.classic_r2", PASS, LOW, exp_r2, r2_val))
            else:
                results.append(make_check_result(win, eid, "pivots.classic_r2", FAIL, HIGH, exp_r2, r2_val,
                                                  details="R2 != P + (H-L)"))

        if s2_val is not None:
            if is_close(exp_s2, s2_val, rel_tol=0.005, abs_tol=1e-2):
                results.append(make_check_result(win, eid, "pivots.classic_s2", PASS, LOW, exp_s2, s2_val))
            else:
                results.append(make_check_result(win, eid, "pivots.classic_s2", FAIL, HIGH, exp_s2, s2_val,
                                                  details="S2 != P - (H-L)"))

    return results


def check_derivatives(ev):
    """9 — DERIVATIVES Checks"""
    results = []
    win = ev.get("janela_numero") or ev.get("window_id") or ev.get("epoch_ms")
    eid = ev.get("event_id") or ev.get("_source_db_id")

    derivs = ev.get("derivatives") or {}
    eth_d = derivs.get("ETHUSDT") or (derivs if "long_short_ratio" in derivs else {})

    lsr = safe_float(eth_d.get("long_short_ratio"))
    longs_usd = safe_float(eth_d.get("longs_usd"))
    shorts_usd = safe_float(eth_d.get("shorts_usd"))
    oi_usd = safe_float(eth_d.get("open_interest_usd"))

    # Checar NaN / Inf
    for k, v in eth_d.items():
        if is_nan_or_inf(v):
            results.append(make_check_result(win, eid, f"derivatives.no_nan_inf_{k}", FAIL, BLOCKER, "Finite or null", str(v),
                                              details=f"NaN/Inf detectado em derivados ({k})"))

    # long_short_ratio ≈ longs_usd / shorts_usd
    if lsr is not None and longs_usd is not None and shorts_usd is not None and shorts_usd > 0:
        exp_lsr = longs_usd / shorts_usd
        if is_close(exp_lsr, lsr, rel_tol=0.02, abs_tol=1e-2):
            results.append(make_check_result(win, eid, "derivatives.lsr_formula", PASS, LOW, exp_lsr, lsr))
        else:
            results.append(make_check_result(win, eid, "derivatives.lsr_formula", FAIL, HIGH, exp_lsr, lsr,
                                              details="long_short_ratio != longs_usd / shorts_usd"))

    # longs_usd + shorts_usd ≈ open_interest_usd (quando aplicável ao contrato)
    if longs_usd is not None and shorts_usd is not None and oi_usd is not None:
        exp_oi = longs_usd + shorts_usd
        if is_close(exp_oi, oi_usd, rel_tol=0.05, abs_tol=1e-1):
            results.append(make_check_result(win, eid, "derivatives.oi_sum_reconciliation", PASS, LOW, exp_oi, oi_usd))
        else:
            results.append(make_check_result(win, eid, "derivatives.oi_sum_reconciliation", FAIL, MEDIUM, exp_oi, oi_usd,
                                              details="longs_usd + shorts_usd != open_interest_usd"))

    # Funding rate
    funding = eth_d.get("funding_rate_percent") or eth_d.get("funding_rate")
    if funding is not None:
        ff = safe_float(funding)
        if ff is not None:
            results.append(make_check_result(win, eid, "derivatives.funding_rate_registered", PASS, LOW, "value recorded", f"{ff}%",
                                              details=f"Funding rate registrado: {ff}%"))

    return results


def check_macro(ev):
    """10 — MACRO Checks (verificação recursiva de NaN/Inf e contagem null/missing)"""
    results = []
    win = ev.get("janela_numero") or ev.get("window_id") or ev.get("epoch_ms")
    eid = ev.get("event_id") or ev.get("_source_db_id")

    macro_data = ev.get("external_markets") or ev.get("market_environment") or ev.get("macro") or {}

    counts = {"present": 0, "null": 0, "missing": 0, "nan_inf": 0}

    def _inspect(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _inspect(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(obj, list):
            for i, elem in enumerate(obj):
                _inspect(elem, f"{prefix}[{i}]")
        else:
            if obj is None:
                counts["null"] += 1
            elif is_nan_or_inf(obj):
                counts["nan_inf"] += 1
                results.append(make_check_result(win, eid, f"macro.no_nan_inf_{prefix}", FAIL, BLOCKER, "Finite or null", str(obj),
                                                  details=f"NaN/Inf detectado no campo macro ({prefix})"))
            else:
                counts["present"] += 1

    _inspect(macro_data)

    if counts["nan_inf"] == 0:
        results.append(make_check_result(win, eid, "macro.no_nan_inf_check", PASS, LOW, 0, 0,
                                          details=f"Campos macro ok. Presentes: {counts['present']}, Nulls: {counts['null']}"))

    return results


def check_ai_quality(ev, ai_events_map=None):
    """11 — AI QUALITY Checks (correlaciona AI_ANALYSIS com ANALYSIS_TRIGGER)"""
    results = []
    win = ev.get("janela_numero") or ev.get("window_id") or ev.get("epoch_ms")
    eid = ev.get("event_id") or ev.get("_source_db_id")

    ev_type = ev.get("tipo_evento") or ev.get("event_type")
    if ev_type != "ANALYSIS_TRIGGER":
        return results

    # Buscar AI_ANALYSIS correspondente por epoch_ms ou anchor_window_id
    ep_ms = safe_float(ev.get("epoch_ms"))
    ai_ev = None
    if ai_events_map and ep_ms is not None:
        ai_ev = ai_events_map.get(ep_ms)

    if ai_ev:
        ai_res = ai_ev.get("ai_result") or {}
        ai_conf = safe_float(ai_res.get("confidence"))

        q_lat = ev.get("quality", {}).get("latency", {})
        is_acceptable = q_lat.get("is_acceptable")

        # Se latência foi inaceitável (is_acceptable=0), AI não pode ter confiança total (ex: > 0.9) sem aviso/cap
        if is_acceptable == 0 or is_acceptable is False:
            if ai_conf is not None and ai_conf > 0.9:
                results.append(make_check_result(win, eid, "ai_quality.latency_confidence_cap", FAIL, BLOCKER, "<=0.9", ai_conf,
                                                  details="Janela com latência inaceitável (POOR) descrita com confiança total (>90%) pela IA"))
            else:
                results.append(make_check_result(win, eid, "ai_quality.latency_confidence_cap", PASS, LOW, "<=0.9", ai_conf or 0))

    return results


def check_cross_file(sqlite_events, jsonl_events):
    """12 — CROSS-FILE Checks (compara persistência SQLite vs JSONL)"""
    results = []
    
    # Mapear JSONL por event_id
    jsonl_map = {}
    for ev in jsonl_events:
        eid = ev.get("event_id")
        if eid:
            jsonl_map[str(eid)] = ev

    for sq_ev in sqlite_events:
        eid = sq_ev.get("event_id")
        if not eid or str(eid) not in jsonl_map:
            continue

        js_ev = jsonl_map[str(eid)]
        win = sq_ev.get("janela_numero") or sq_ev.get("window_id") or sq_ev.get("epoch_ms")

        # Comparar campos críticos
        sq_ep = safe_float(sq_ev.get("epoch_ms"))
        js_ep = safe_float(js_ev.get("epoch_ms"))

        sq_px = safe_float(sq_ev.get("preco_fechamento") or sq_ev.get("last_price"))
        js_px = safe_float(js_ev.get("preco_fechamento") or js_ev.get("last_price"))

        sq_vol = safe_float(sq_ev.get("volume_total") or sq_ev.get("volume_total_btc"))
        js_vol = safe_float(js_ev.get("volume_total") or js_ev.get("volume_total_btc"))

        sq_delta = safe_float(sq_ev.get("delta") or sq_ev.get("delta_fechamento"))
        js_delta = safe_float(js_ev.get("delta") or js_ev.get("delta_fechamento"))

        divergences = []
        if sq_ep is not None and js_ep is not None and not is_close(sq_ep, js_ep, rel_tol=0.0, abs_tol=1.0):
            divergences.append(f"epoch_ms ({sq_ep} vs {js_ep})")
        if sq_px is not None and js_px is not None and not is_close(sq_px, js_px, rel_tol=0.001, abs_tol=1e-2):
            divergences.append(f"price ({sq_px} vs {js_px})")
        if sq_vol is not None and js_vol is not None and not is_close(sq_vol, js_vol, rel_tol=0.01, abs_tol=1e-3):
            divergences.append(f"volume ({sq_vol} vs {js_vol})")
        if sq_delta is not None and js_delta is not None and not is_close(sq_delta, js_delta, rel_tol=0.01, abs_tol=1e-3):
            divergences.append(f"delta ({sq_delta} vs {js_delta})")

        if divergences:
            results.append(make_check_result(win, eid, "cross_file.persistence_alignment", FAIL, BLOCKER, "Identical persistence",
                                              ", ".join(divergences), details=f"Divergência de persistência entre SQLite e JSONL no event_id {eid}"))
        else:
            results.append(make_check_result(win, eid, "cross_file.persistence_alignment", PASS, LOW, "Identical", "Identical"))

    return results


# =============================================================================
# EXECUTOR PRINCIPAL DA AUDITORIA
# =============================================================================
def run_audit(db_path, jsonl_path, log_path=None, last_n=None, run_all=False,
              symbol_filter=None, event_type_filter=None, since_id=None, since_epoch_ms=None):
    """Executa a auditoria completa e retorna placar + resultados."""
    start_time = datetime.now()

    # 1. Carregar eventos
    sqlite_events = load_sqlite_events(db_path, symbol_filter, event_type_filter, since_id, since_epoch_ms)
    jsonl_events = load_jsonl_events(jsonl_path, symbol_filter, event_type_filter, since_epoch_ms)

    # Combinar ou focar nos eventos de trigger
    all_events = sqlite_events + jsonl_events

    # Mapear eventos AI_ANALYSIS por epoch_ms/anchor_window_id
    ai_events_map = {}
    for ev in all_events:
        ev_type = ev.get("tipo_evento") or ev.get("event_type")
        if ev_type == "AI_ANALYSIS":
            anchor_id = safe_float(ev.get("anchor_window_id") or ev.get("timestamp_ms") or ev.get("epoch_ms"))
            if anchor_id:
                ai_events_map[anchor_id] = ev

    # Filtrar por tipo ANALYSIS_TRIGGER como padrão principal para auditoria de janelas
    trigger_events = [ev for ev in all_events if (ev.get("tipo_evento") or ev.get("event_type")) == "ANALYSIS_TRIGGER"]

    if not trigger_events:
        trigger_events = all_events  # Fallback se não houver triggers explícitos

    # Filtrar last N
    if last_n and not run_all and len(trigger_events) > last_n:
        trigger_events = trigger_events[-last_n:]

    all_check_results = []
    audited_event_meta = []

    # Processar cada evento
    for ev in trigger_events:
        win = ev.get("janela_numero") or ev.get("window_id") or ev.get("epoch_ms")
        eid = ev.get("event_id") or ev.get("_source_db_id")
        ep_ms = ev.get("epoch_ms") or ev.get("timestamp_ms")
        ts_utc = ev.get("timestamp_utc") or ev.get("timestamp")

        audited_event_meta.append({
            "event_id": str(eid) if eid is not None else "N/A",
            "sequence_id": ev.get("sequence_id") or ev.get("_source_db_id") or "N/A",
            "epoch_ms": ep_ms,
            "timestamp": ts_utc
        })

        all_check_results.extend(check_flow(ev))
        all_check_results.extend(check_sector_flow(ev))
        all_check_results.extend(check_orderbook(ev))
        all_check_results.extend(check_temporal(ev))
        all_check_results.extend(check_source_quality(ev))
        all_check_results.extend(check_value_profile(ev))
        all_check_results.extend(check_support_resistance(ev))
        all_check_results.extend(check_pivots(ev))
        all_check_results.extend(check_derivatives(ev))
        all_check_results.extend(check_macro(ev))
        all_check_results.extend(check_ai_quality(ev, ai_events_map))

    # Cross-file check
    all_check_results.extend(check_cross_file(sqlite_events, jsonl_events))

    # Montar scorecard
    category_summary = {}
    severity_counts = {BLOCKER: 0, HIGH: 0, MEDIUM: 0, LOW: 0}

    for res in all_check_results:
        chk_name = res["check"]
        cat = chk_name.split(".")[0].upper()
        if cat not in category_summary:
            category_summary[cat] = {"PASS": 0, "FAIL": 0, "SKIP": 0, "NOT_COMPARABLE": 0, "TOTAL": 0}

        st = res["status"]
        if st in category_summary[cat]:
            category_summary[cat][st] += 1
        category_summary[cat]["TOTAL"] += 1

        sev = res["severity"]
        if st == FAIL and sev in severity_counts:
            severity_counts[sev] += 1

    # Decisão Final
    if severity_counts[BLOCKER] > 0:
        decision = "FAIL"
    elif severity_counts[HIGH] > 0 or severity_counts[MEDIUM] > 0:
        decision = "PASS_WITH_WARNINGS"
    else:
        decision = "PASS"

    end_time = datetime.now()
    duration_sec = (end_time - start_time).total_seconds()

    report_payload = {
        "metadata": {
            "audited_at": datetime.now(timezone.utc).isoformat(),
            "total_events_audited": len(audited_event_meta),
            "execution_time_seconds": round(duration_sec, 3),
            "db_path": db_path,
            "jsonl_path": jsonl_path
        },
        "audited_events": audited_event_meta,
        "scorecard": {
            "decision": decision,
            "severity_counts": severity_counts,
            "category_summary": category_summary
        },
        "checks": all_check_results
    }

    return report_payload


# =============================================================================
# FORMATADORES DE SAÍDA (TERMINAL, JSON, MARKDOWN)
# =============================================================================
def print_terminal_summary(report):
    """Imprime o scorecard final no terminal de forma legível."""
    meta = report["metadata"]
    sc = report["scorecard"]

    print("\n============================================================")
    print("           AUDITOR OFFLINE DOS DADOS REAIS - RESULTADO      ")
    print("============================================================")
    print(f" Eventos Auditados : {meta['total_events_audited']}")
    print(f" Tempo de Execução : {meta['execution_time_seconds']}s")
    print(f" Decisão Final     : {sc['decision']}")
    print("------------------------------------------------------------")
    print(" SCORECARD POR CATEGORIA:")

    for cat, data in sc["category_summary"].items():
        total = data["TOTAL"]
        passed = data["PASS"]
        pct = (passed / total * 100.0) if total > 0 else 0
        print(f"  {cat:<20} PASS {passed:>3}/{total:<3} ({pct:>5.1f}%) | FAIL: {data['FAIL']} | SKIP: {data['SKIP']}")

    print("------------------------------------------------------------")
    print(" CONTAGEM DE FALHAS POR SEVERIDADE:")
    sev = sc["severity_counts"]
    print(f"  BLOCKER : {sev['BLOCKER']}")
    print(f"  HIGH    : {sev['HIGH']}")
    print(f"  MEDIUM  : {sev['MEDIUM']}")
    print(f"  LOW     : {sev['LOW']}")
    print("============================================================\n")


def generate_markdown_report(report, md_path):
    """Gera relatório complementar em Markdown."""
    meta = report["metadata"]
    sc = report["scorecard"]

    md = []
    md.append("# Relatório da Auditoria Offline dos Dados Reais\n")
    md.append(f"- **Data da Análise**: {meta['audited_at']}")
    md.append(f"- **Eventos Auditados**: {meta['total_events_audited']}")
    md.append(f"- **Tempo de Execução**: {meta['execution_time_seconds']}s")
    md.append(f"- **Decisão Final**: `{sc['decision']}`\n")

    md.append("## Placar (Scorecard)\n")
    md.append("| Categoria | Pass | Fail | Skip | Not Comp | Total | Approval Rate |")
    md.append("|---|---|---|---|---|---|---|")

    for cat, data in sc["category_summary"].items():
        tot = data["TOTAL"]
        p = data["PASS"]
        pct = (p / tot * 100.0) if tot > 0 else 0.0
        md.append(f"| {cat} | {p} | {data['FAIL']} | {data['SKIP']} | {data['NOT_COMPARABLE']} | {tot} | {pct:.1f}% |")

    md.append("\n## Severidade das Falhas Encontradas\n")
    sev = sc["severity_counts"]
    md.append(f"- **BLOCKER**: {sev['BLOCKER']}")
    md.append(f"- **HIGH**: {sev['HIGH']}")
    md.append(f"- **MEDIUM**: {sev['MEDIUM']}")
    md.append(f"- **LOW**: {sev['LOW']}\n")

    md.append("## Eventos Auditados\n")
    md.append("| Event ID | Sequence ID | Epoch MS | Timestamp |")
    md.append("|---|---|---|---|")
    for ev in report["audited_events"][:50]:  # Limitar a 50 na exibição do md
        md.append(f"| `{ev['event_id']}` | `{ev['sequence_id']}` | `{ev['epoch_ms']}` | `{ev['timestamp']}` |")

    if len(report["audited_events"]) > 50:
        md.append(f"\n*... e mais {len(report['audited_events']) - 50} eventos.*")

    md.append("\n## Detalhe das Verificações com Falha\n")
    failed_checks = [c for c in report["checks"] if c["status"] == FAIL]
    if not failed_checks:
        md.append("Nenhuma falha encontrada!\n")
    else:
        md.append("| Window | Event ID | Check | Severity | Expected | Actual | Details |")
        md.append("|---|---|---|---|---|---|---|")
        for fc in failed_checks[:100]:
            md.append(f"| {fc['window']} | `{fc['event_id']}` | `{fc['check']}` | **{fc['severity']}** | `{fc['expected']}` | `{fc['actual']}` | {fc['details']} |")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


# =============================================================================
# CLI ENTRYPOINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Auditor Offline dos Dados Reais do Mercado")
    parser.add_argument("--db", default="dados/trading_bot.db", help="Caminho do banco SQLite")
    parser.add_argument("--jsonl", default="dados/eventos_fluxo.jsonl", help="Caminho do arquivo JSONL")
    parser.add_argument("--log", default="dados/eventos_visuais.log", help="Caminho opcional do log visual")
    parser.add_argument("--last", type=int, default=None, help="Analisar apenas os últimos N eventos ANALYSIS_TRIGGER (ex: 5, 10, 50)")
    parser.add_argument("--all", action="store_true", help="Analisar todos os eventos")
    parser.add_argument("--event-type", default=None, help="Filtro por tipo de evento (ex: ANALYSIS_TRIGGER)")
    parser.add_argument("--symbol", default=None, help="Filtro por símbolo (ex: BTCUSDT)")
    parser.add_argument("--since-id", type=int, default=None, help="Filtro por ID inicial no banco SQLite")
    parser.add_argument("--since-epoch-ms", type=float, default=None, help="Filtro por epoch_ms inicial")
    parser.add_argument("--output-json", default="dados/audit_market_data_report.json", help="Caminho para salvar o relatório JSON")
    parser.add_argument("--output-md", default="dados/audit_market_data_report.md", help="Caminho para salvar o relatório Markdown")

    args = parser.parse_args()

    # Se nenhum filtro de escopo for passado (--last N ou --all), usar --last 10 como padrão amigável
    last_n = args.last
    if not last_n and not args.all:
        last_n = 10

    report = run_audit(
        db_path=args.db,
        jsonl_path=args.jsonl,
        log_path=args.log,
        last_n=last_n,
        run_all=args.all,
        symbol_filter=args.symbol,
        event_type_filter=args.event_type,
        since_id=args.since_id,
        since_epoch_ms=args.since_epoch_ms
    )

    # Imprimir no terminal
    print_terminal_summary(report)

    # Salvar JSON obrigatório
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Relatório JSON salvo em: {args.output_json}")

    # Salvar Markdown
    if args.output_md:
        os.makedirs(os.path.dirname(args.output_md), exist_ok=True)
        generate_markdown_report(report, args.output_md)
        print(f"Relatório Markdown salvo em: {args.output_md}")


if __name__ == "__main__":
    main()
