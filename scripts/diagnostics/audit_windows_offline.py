#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/diagnostics/audit_windows_offline.py

Script 100% offline para auditoria e achatamento de janelas a partir de dados/trading_bot.db.
Extrai metadados de sessão/janela e achata colunas numéricas com prefixos padronizados:
  - meta_  : metadados de rastreio (row_id, session, janela_numero, window_key, tipo_evento, timestamps)
  - raw_   : dados brutos de volume, trades, preços, duração (com registro de proveniência de json-path)
  - ob_    : order book (mid, spread, depths, imbalances, scores, latência, timestamps de exchange, walls top-3)
  - flow_  : fluxo contínuo (cvd, whale delta/volumes, sector flow, bursts, scores)
  - sr_    : suporte/resistência, volume profile e defense zones top-5
  - inst_  : métricas institucionais e qualidade de dados
  - ml_    : features e probabilidades de machine learning

JSON-PATHS MAPEADOS NO PAYLOAD:
  - raw_preco_fechamento: payload.preco_fechamento -> raw_event.preco_fechamento -> contextual_snapshot.ohlc.close
  - raw_duracao_segundos: payload.duration_s -> payload.window_duration_ms / 1000.0 -> contextual_snapshot.ohlc.(close_time-open_time) / 1000.0
  - ob_latency_ms / category: institutional_analytics.quality.latency.latency_ms / latency_category
  - ob_exchange_ms: raw_event.orderbook_data.timestamps.exchange_ms -> orderbook_data.timestamps.exchange_ms
  - ob_age_ms: epoch_ms - ob_exchange_ms
  - ob_l1_bid_usd / ask_usd: order_book_depth.L1.bids / asks -> raw_event.orderbook_data.order_book_depth.L1.bids / asks
  - ob_spread_percentile: institutional_analytics.quality.spread_percentile.spread_percentile -> raw_event.orderbook_data.spread_analysis.spread_percentile
  - flow_cvd: fluxo_continuo.cvd -> raw_event.advanced_analysis.flow_metrics.cvd
  - flow_whale_score: institutional_analytics.flow_analysis.whale_accumulation.score
  - flow_iceberg_activity: whale_activity.iceberg_activity
  - data_quality.*: fluxo_continuo.data_quality.* -> data_quality.*
  - walls top-3: raw_event.orderbook_data.walls.bids[:3] / asks[:3]
  - defense_zones top-5: institutional_analytics.sr_analysis.defense_zones (buy_defense + sell_defense)
"""

import sys
import io
import os
import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audit_windows_offline")

DB_PATH = Path("dados/trading_bot.db")
OUTPUT_DIR = Path("dados/audit")


def first_not_none(*vals: Any) -> Optional[Any]:
    """Retorna o primeiro valor que não seja None. Preserva 0 e 0.0."""
    for v in vals:
        if v is not None:
            return v
    return None


def safe_float(val: Any) -> Optional[float]:
    """Converte valor para float seguro sem descartar 0.0."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val.replace(",", "").strip())
        except (ValueError, TypeError):
            return None
    return None


def safe_int(val: Any) -> Optional[int]:
    """Converte valor para int seguro sem descartar 0."""
    if val is None:
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    if isinstance(val, str):
        try:
            return int(float(val.replace(",", "").strip()))
        except (ValueError, TypeError):
            return None
    return None


def extract_flat_window(row_id: int, db_event_type: str, db_ts_ms: int, payload: Dict[str, Any], session: int) -> Dict[str, Any]:
    """
    Achata os dados de um evento/janela em um dicionário tabular com prefixos:
    meta_, raw_, ob_, flow_, sr_, inst_, ml_.
    """
    janela_numero = payload.get("janela_numero")
    tipo_evento = first_not_none(payload.get("tipo_evento"), db_event_type)
    epoch_ms = first_not_none(payload.get("epoch_ms"), db_ts_ms)
    timestamp_utc = payload.get("timestamp_utc") or ""
    event_id = payload.get("event_id") or f"db_row_{row_id}"
    symbol = payload.get("symbol") or "BTCUSDT"

    win_str = str(janela_numero) if janela_numero is not None else "none"
    window_key = f"{session}:{win_str}"

    row: Dict[str, Any] = {
        # Metadados
        "meta_row_id": row_id,
        "meta_session": session,
        "meta_janela_numero": janela_numero,
        "meta_window_key": window_key,
        "meta_tipo_evento": tipo_evento,
        "meta_event_id": event_id,
        "meta_symbol": symbol,
        "meta_epoch_ms": epoch_ms,
        "meta_timestamp_utc": timestamp_utc,
    }

    # 1. RAW_ (Volume, trades, preços, duração)
    raw_event = payload.get("raw_event") or {}
    ctx_snap = payload.get("contextual_snapshot") or {}
    ohlc = ctx_snap.get("ohlc") or {}

    preco_fechamento = first_not_none(
        payload.get("preco_fechamento"),
        raw_event.get("preco_fechamento"),
        ctx_snap.get("preco_fechamento"),
        ohlc.get("close")
    )
    volume_total = first_not_none(
        payload.get("volume_total"),
        raw_event.get("volume_total"),
        ctx_snap.get("volume_total")
    )
    volume_compra = first_not_none(
        payload.get("volume_compra"),
        raw_event.get("volume_compra"),
        ctx_snap.get("volume_compra")
    )
    volume_venda = first_not_none(
        payload.get("volume_venda"),
        raw_event.get("volume_venda"),
        ctx_snap.get("volume_venda")
    )
    delta = first_not_none(
        payload.get("delta"),
        raw_event.get("delta")
    )
    num_trades = first_not_none(
        ctx_snap.get("num_trades"),
        payload.get("num_trades")
    )

    # Determinação da duração e registro de proveniência (json-path)
    duracao_val = None
    duracao_source = "missing"
    if payload.get("duration_s") is not None:
        duracao_val = safe_float(payload.get("duration_s"))
        duracao_source = "payload.duration_s"
    elif payload.get("window_duration_ms") is not None:
        duracao_val = safe_float(payload.get("window_duration_ms")) / 1000.0
        duracao_source = "payload.window_duration_ms"
    elif ohlc.get("open_time") is not None and ohlc.get("close_time") is not None:
        ot = safe_float(ohlc.get("open_time"))
        ct = safe_float(ohlc.get("close_time"))
        if ot is not None and ct is not None and ct >= ot:
            duracao_val = round((ct - ot) / 1000.0, 3)
            duracao_source = "contextual_snapshot.ohlc.(close_time-open_time)"
    elif payload.get("duracao_segundos") is not None:
        duracao_val = safe_float(payload.get("duracao_segundos"))
        duracao_source = "payload.duracao_segundos"
    elif ctx_snap.get("duracao_segundos") is not None:
        duracao_val = safe_float(ctx_snap.get("duracao_segundos"))
        duracao_source = "contextual_snapshot.duracao_segundos"

    vt = safe_float(volume_total)
    nt = safe_int(num_trades)
    ds = safe_float(duracao_val)

    btc_per_second = (vt / ds) if (vt is not None and ds is not None and ds > 0) else None
    btc_per_trade = (vt / nt) if (vt is not None and nt is not None and nt > 0) else None

    row["raw_preco_fechamento"] = safe_float(preco_fechamento)
    row["raw_volume_total"] = vt
    row["raw_volume_compra"] = safe_float(volume_compra)
    row["raw_volume_venda"] = safe_float(volume_venda)
    row["raw_delta"] = safe_float(delta)
    row["raw_num_trades"] = nt
    row["raw_duracao_segundos"] = ds
    row["raw_duracao_source"] = duracao_source
    row["raw_btc_per_second"] = btc_per_second
    row["raw_btc_per_trade"] = btc_per_trade
    row["raw_open"] = safe_float(ohlc.get("open"))
    row["raw_high"] = safe_float(ohlc.get("high"))
    row["raw_low"] = safe_float(ohlc.get("low"))
    row["raw_close"] = safe_float(ohlc.get("close"))
    row["raw_vwap"] = safe_float(ohlc.get("vwap"))

    # 2. OB_ (Orderbook)
    ob = first_not_none(payload.get("orderbook_data"), raw_event.get("orderbook_data")) or {}
    raw_ob = raw_event.get("orderbook_data") or {}

    row["ob_mid"] = safe_float(first_not_none(ob.get("mid"), raw_ob.get("mid")))
    row["ob_bid"] = safe_float(first_not_none(payload.get("bid"), ob.get("bid"), raw_ob.get("bid")))
    row["ob_ask"] = safe_float(first_not_none(payload.get("ask"), ob.get("ask"), raw_ob.get("ask")))
    row["ob_spread"] = safe_float(first_not_none(ob.get("spread"), raw_ob.get("spread")))
    row["ob_spread_percent"] = safe_float(first_not_none(ob.get("spread_percent"), raw_ob.get("spread_percent")))
    row["ob_spread_bps"] = safe_float(first_not_none(ob.get("spread_bps"), raw_ob.get("spread_bps")))
    row["ob_bid_depth_usd"] = safe_float(first_not_none(ob.get("bid_depth_usd"), raw_ob.get("bid_depth_usd")))
    row["ob_ask_depth_usd"] = safe_float(first_not_none(ob.get("ask_depth_usd"), raw_ob.get("ask_depth_usd")))
    row["ob_imbalance"] = safe_float(first_not_none(ob.get("imbalance"), raw_ob.get("imbalance")))
    row["ob_flow_imbalance"] = safe_float(first_not_none(ob.get("flow_imbalance"), raw_ob.get("flow_imbalance")))
    row["ob_volume_ratio"] = safe_float(first_not_none(ob.get("volume_ratio"), raw_ob.get("volume_ratio")))
    row["ob_bias_score"] = safe_float(first_not_none(ob.get("bias_score"), raw_ob.get("bias_score")))
    row["ob_consolidated_bias_score"] = safe_float(first_not_none(ob.get("consolidated_bias_score"), raw_ob.get("consolidated_bias_score")))

    # Latência do snapshot do book
    inst_ana = payload.get("institutional_analytics") or {}
    quality_obj = inst_ana.get("quality") or {}
    latency_obj = quality_obj.get("latency") or {}
    row["ob_latency_ms"] = safe_float(first_not_none(latency_obj.get("latency_ms"), ob.get("latency_ms"), raw_ob.get("latency_ms")))
    row["ob_latency_category"] = str(first_not_none(latency_obj.get("latency_category"), ob.get("latency_category"), raw_ob.get("latency_category"), ""))

    # Timestamps de exchange do orderbook e ob_age_ms
    ts_ob = first_not_none(raw_ob.get("timestamps"), ob.get("timestamps")) or {}
    exchange_ms = safe_int(first_not_none(ts_ob.get("exchange_ms"), raw_event.get("timestamp_utc"), raw_event.get("timestamp")))
    row["ob_exchange_ms"] = exchange_ms
    if epoch_ms is not None and exchange_ms is not None:
        row["ob_age_ms"] = int(epoch_ms - exchange_ms)
    else:
        row["ob_age_ms"] = None

    # L1 / Depth
    ob_depth = first_not_none(payload.get("order_book_depth"), raw_ob.get("order_book_depth"), ob.get("order_book_depth")) or {}
    l1 = ob_depth.get("L1") or {}
    row["ob_l1_bid_usd"] = safe_float(l1.get("bids"))
    row["ob_l1_ask_usd"] = safe_float(l1.get("asks"))
    row["ob_l1_flow_imbalance"] = safe_float(l1.get("flow_imbalance"))
    row["ob_total_depth_ratio"] = safe_float(ob_depth.get("total_depth_ratio"))

    # Spread percentile
    sp_obj = quality_obj.get("spread_percentile") or {}
    row["ob_spread_percentile"] = safe_float(first_not_none(
        sp_obj.get("spread_percentile") if isinstance(sp_obj, dict) else None,
        raw_ob.get("spread_analysis", {}).get("spread_percentile"),
        ob.get("spread_percentile")
    ))

    # Walls top-3 bid e ask
    walls = first_not_none(raw_ob.get("walls"), ob.get("walls")) or {}
    b_walls = walls.get("bids") or []
    a_walls = walls.get("asks") or []

    for i in range(3):
        if i < len(b_walls) and isinstance(b_walls[i], dict):
            row[f"ob_wall_bid_{i}_price"] = safe_float(b_walls[i].get("price"))
            row[f"ob_wall_bid_{i}_qty"] = safe_float(first_not_none(b_walls[i].get("qty"), b_walls[i].get("quantity")))
        else:
            row[f"ob_wall_bid_{i}_price"] = None
            row[f"ob_wall_bid_{i}_qty"] = None

        if i < len(a_walls) and isinstance(a_walls[i], dict):
            row[f"ob_wall_ask_{i}_price"] = safe_float(a_walls[i].get("price"))
            row[f"ob_wall_ask_{i}_qty"] = safe_float(first_not_none(a_walls[i].get("qty"), a_walls[i].get("quantity")))
        else:
            row[f"ob_wall_ask_{i}_price"] = None
            row[f"ob_wall_ask_{i}_qty"] = None

    # 3. FLOW_ (Fluxo contínuo, CVD, Whale, Absorção, Sector)
    fluxo = payload.get("fluxo_continuo") or {}
    row["flow_cvd"] = safe_float(first_not_none(fluxo.get("cvd"), raw_event.get("advanced_analysis", {}).get("flow_metrics", {}).get("cvd")))
    row["flow_whale_buy"] = safe_float(fluxo.get("whale_buy_volume"))
    row["flow_whale_sell"] = safe_float(fluxo.get("whale_sell_volume"))
    row["flow_whale_delta"] = safe_float(fluxo.get("whale_delta"))

    sec_flow = fluxo.get("sector_flow")
    if isinstance(sec_flow, dict):
        row["flow_retail_delta"] = safe_float(sec_flow.get("retail", {}).get("delta"))
        row["flow_mid_delta"] = safe_float(sec_flow.get("mid", {}).get("delta"))
        row["flow_whale_sector_delta"] = safe_float(sec_flow.get("whale", {}).get("delta"))
    else:
        row["flow_retail_delta"] = None
        row["flow_mid_delta"] = None
        row["flow_whale_sector_delta"] = None

    bursts = fluxo.get("bursts")
    row["flow_bursts_count"] = len(bursts) if isinstance(bursts, list) else 0

    heatmap = fluxo.get("liquidity_heatmap") or {}
    clusters = heatmap.get("clusters")
    row["flow_heatmap_clusters_count"] = len(clusters) if isinstance(clusters, list) else 0

    flow_ana = inst_ana.get("flow_analysis") or {}
    whale_acc = flow_ana.get("whale_accumulation") or {}
    row["flow_whale_score"] = safe_float(first_not_none(whale_acc.get("score"), flow_ana.get("whale_score")))
    row["flow_whale_bias"] = str(whale_acc.get("bias", ""))

    whale_act = payload.get("whale_activity") or {}
    row["flow_iceberg_activity"] = bool(whale_act.get("iceberg_activity", False))
    row["flow_hidden_orders_count"] = safe_int(whale_act.get("hidden_orders_detected"))

    # 4. SR_ (Support & Resistance, Volume Profile, Defense Zones)
    sr_ana = inst_ana.get("sr_analysis") or {}
    h_vp = payload.get("historical_vp") or {}
    daily_vp = h_vp.get("daily") or {}

    poc = safe_float(daily_vp.get("poc"))
    val = safe_float(daily_vp.get("val"))
    vah = safe_float(daily_vp.get("vah"))
    row["sr_poc"] = poc
    row["sr_val"] = val
    row["sr_vah"] = vah

    pf = row["raw_preco_fechamento"]
    if pf is not None:
        row["sr_dist_to_poc_bps"] = round(((pf - poc) / pf) * 10000, 2) if poc else None
        row["sr_dist_to_val_bps"] = round(((pf - val) / pf) * 10000, 2) if val else None
        row["sr_dist_to_vah_bps"] = round(((pf - vah) / pf) * 10000, 2) if vah else None
    else:
        row["sr_dist_to_poc_bps"] = None
        row["sr_dist_to_val_bps"] = None
        row["sr_dist_to_vah_bps"] = None

    # Defense zones top-5 (ordenadas por strength decrescente)
    dz_obj = sr_ana.get("defense_zones") or {}
    all_dz = []
    for side, key in [("buy", "buy_defense"), ("sell", "sell_defense")]:
        zones = dz_obj.get(key) or []
        for z in zones:
            if isinstance(z, dict):
                center = safe_float(z.get("center"))
                strength = safe_float(z.get("strength"))
                sources = z.get("sources") or []
                sources_str = ",".join(sources) if isinstance(sources, list) else str(sources)
                all_dz.append({
                    "price": center,
                    "side": side,
                    "strength": strength if strength is not None else 0.0,
                    "sources": sources_str
                })
    all_dz.sort(key=lambda x: x["strength"], reverse=True)

    for i in range(5):
        if i < len(all_dz):
            row[f"sr_dz_{i}_price"] = all_dz[i]["price"]
            row[f"sr_dz_{i}_side"] = all_dz[i]["side"]
            row[f"sr_dz_{i}_strength"] = all_dz[i]["strength"]
            row[f"sr_dz_{i}_sources"] = all_dz[i]["sources"]
        else:
            row[f"sr_dz_{i}_price"] = None
            row[f"sr_dz_{i}_side"] = None
            row[f"sr_dz_{i}_strength"] = None
            row[f"sr_dz_{i}_sources"] = None

    # 5. INST_ (Data Quality e Institutional Status)
    row["inst_status"] = str(inst_ana.get("status", ""))
    row["inst_quality_score"] = safe_float(payload.get("data_quality_score"))
    row["inst_completeness_pct"] = safe_float(payload.get("completeness_pct"))
    row["inst_reliability_score"] = safe_float(payload.get("reliability_score"))

    # data_quality.* (total_trades_processed, invalid_trades, valid_rate_pct, latency)
    dq = first_not_none(fluxo.get("data_quality"), payload.get("data_quality")) or {}
    row["data_quality_total_trades"] = safe_int(dq.get("total_trades_processed"))
    row["data_quality_invalid_trades"] = safe_int(dq.get("invalid_trades"))
    row["data_quality_valid_rate_pct"] = safe_float(dq.get("valid_rate_pct"))
    row["data_quality_latency_ms"] = row["ob_latency_ms"]

    # 6. ML_ (Machine Learning)
    ml = payload.get("ml_features") or {}
    vol_metrics = payload.get("volatility_metrics") or {}
    regime_ana = payload.get("regime_analysis") or {}

    row["ml_prob_up"] = safe_float(ml.get("prob_up"))
    row["ml_prob_down"] = safe_float(ml.get("prob_down"))
    row["ml_volatility_regime"] = str(first_not_none(vol_metrics.get("volatility_regime"), regime_ana.get("current_regime"), ""))
    row["ml_volatility_percentile"] = safe_float(vol_metrics.get("volatility_percentile"))

    return row


def build_audit_dataset(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Lê todas as linhas de events no SQLite e monta o DataFrame achatado."""
    if not db_path.exists():
        raise FileNotFoundError(f"Banco de dados {db_path} não encontrado!")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT id, event_type, timestamp_ms, payload FROM events ORDER BY id ASC")

    rows = []
    session = 1
    prev_win = None

    for row_id, ev_type, ts_ms, payload_str in cursor.fetchall():
        try:
            payload = json.loads(payload_str)
        except Exception as e:
            logger.warning(f"Erro ao parsear payload do row_id {row_id}: {e}")
            continue

        win = payload.get("janela_numero")
        if win is not None:
            if prev_win is not None and win < prev_win:
                session += 1
                logger.info(f"Nova sessão detectada: {session} (janela regrediu de {prev_win} para {win})")
            prev_win = win

        flat_row = extract_flat_window(row_id, ev_type, ts_ms, payload, session)
        rows.append(flat_row)

    conn.close()

    df = pd.DataFrame(rows)
    logger.info(f"Dataset achatado gerado com {len(df)} registros e {len(df.columns)} colunas.")
    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_audit_dataset(DB_PATH)

    parquet_path = OUTPUT_DIR / "windows_flat.parquet"
    csv_path = OUTPUT_DIR / "windows_flat.csv"

    try:
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Salvo: {parquet_path}")
    except Exception as e:
        logger.warning(f"Falha ao salvar parquet: {e}")

    df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info(f"Salvo: {csv_path}")

    print("\n" + "=" * 80)
    print("RESUMO DO ACHATAMENTO DE JANELAS:")
    print(f"Total de registros: {len(df)}")
    print(f"Sessões encontradas: {df['meta_session'].unique().tolist()}")
    print(f"Tipos de evento: {df['meta_tipo_evento'].value_counts().to_dict()}")
    print("Colunas por prefixo:")
    for prefix in ["meta_", "raw_", "ob_", "flow_", "sr_", "inst_", "data_quality_", "ml_"]:
        cols = [c for c in df.columns if c.startswith(prefix)]
        print(f"  - {prefix}: {len(cols)} colunas ({', '.join(cols[:4])}...)")
    print("=" * 80)

    # Verificação estrita de confronto com a R1 nas janelas 1:21 e 1:24
    print("\nCONFRONTO COM A R1 (Janelas 1:21 e 1:24):")
    cols_check = [
        "meta_row_id", "meta_tipo_evento", "meta_window_key",
        "raw_duracao_segundos", "raw_duracao_source",
        "ob_latency_ms", "ob_latency_category",
        "flow_cvd", "raw_preco_fechamento", "ob_mid"
    ]
    df_check = df[df["meta_window_key"].isin(["1:21", "1:24"])][cols_check]
    print(df_check.to_string(index=False))


if __name__ == "__main__":
    main()
