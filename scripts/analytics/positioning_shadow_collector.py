# scripts/analytics/positioning_shadow_collector.py
# -*- coding: utf-8 -*-
"""
Coletor e Persistidor de Shadow Dataset de Posicionamento Binance.
Fase P1.1B.

Garante:
1. Persistência de observações brutas e normalizadas sem qualquer alteração na decisão de trade.
2. Registro estrito de proveniência (source_timestamp_ms, age_seconds, is_stale, cache_hit).
3. Zero lookahead: timestamps da feature são explicitamente <= observation_timestamp.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, ".")

from fetchers.binance_positioning_fetcher import BinancePositioningFetcher, BinancePositioningSnapshot
from institutional.crypto_cot import CryptoCOT, PositioningRegime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PositioningShadowCollector")

DEFAULT_DB_PATH = "dados/trading_bot.db"


def init_shadow_db(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Cria tabela estruturada para o shadow dataset de posicionamento."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS positioning_shadow_dataset (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp_ms INTEGER NOT NULL,
        symbol TEXT NOT NULL,
        price REAL,
        global_account_ratio REAL,
        top_account_ratio REAL,
        top_position_ratio REAL,
        global_long_pct REAL,
        global_short_pct REAL,
        top_long_account_pct REAL,
        top_short_account_pct REAL,
        top_long_position_pct REAL,
        top_short_position_pct REAL,
        open_interest REAL,
        open_interest_usd REAL,
        oi_delta_1h REAL,
        oi_delta_4h REAL,
        funding_rate REAL,
        top_account_vs_global REAL,
        top_position_vs_global REAL,
        positioning_regime TEXT NOT NULL,
        source_timestamp_ms INTEGER,
        age_seconds REAL,
        is_stale INTEGER NOT NULL,
        cache_hit INTEGER NOT NULL,
        reasons_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_pos_shadow_ts_sym 
    ON positioning_shadow_dataset(timestamp_ms, symbol);
    """)
    conn.commit()
    return conn


def record_shadow_observation(
    snapshot: BinancePositioningSnapshot,
    analysis_regime: str,
    reasons: List[str],
    current_price: Optional[float] = None,
    funding_rate: Optional[float] = None,
    cache_hit: bool = False,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """Grava uma observação pontual no shadow dataset."""
    conn = init_shadow_db(db_path)
    cur = conn.cursor()
    
    obs_ms = int(snapshot.observed_at * 1000)
    
    cur.execute("""
    INSERT INTO positioning_shadow_dataset (
        timestamp_ms, symbol, price,
        global_account_ratio, top_account_ratio, top_position_ratio,
        global_long_pct, global_short_pct,
        top_long_account_pct, top_short_account_pct,
        top_long_position_pct, top_short_position_pct,
        open_interest, open_interest_usd,
        oi_delta_1h, oi_delta_4h,
        funding_rate, top_account_vs_global, top_position_vs_global,
        positioning_regime, source_timestamp_ms, age_seconds,
        is_stale, cache_hit, reasons_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        obs_ms,
        snapshot.symbol,
        current_price,
        snapshot.global_account_ratio,
        snapshot.top_account_ratio,
        snapshot.top_position_ratio,
        snapshot.global_long_account_pct,
        snapshot.global_short_account_pct,
        snapshot.top_long_account_pct,
        snapshot.top_short_account_pct,
        snapshot.top_long_position_pct,
        snapshot.top_short_position_pct,
        snapshot.open_interest,
        snapshot.open_interest_usd,
        snapshot.oi_delta_1h,
        snapshot.oi_delta_4h,
        funding_rate,
        snapshot.top_account_vs_global,
        snapshot.top_position_vs_global,
        analysis_regime,
        snapshot.source_timestamp,
        snapshot.age_seconds,
        1 if snapshot.is_stale else 0,
        1 if cache_hit else 0,
        json.dumps(reasons, ensure_ascii=False),
    ))
    
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


async def collect_single_shadow_sample(
    symbol: str = "BTCUSDT",
    fetcher: Optional[BinancePositioningFetcher] = None,
    current_price: Optional[float] = None,
    funding_rate: Optional[float] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> Dict[str, Any]:
    """Coleta e persiste uma amostra de sombra em runtime."""
    if fetcher is None:
        fetcher = BinancePositioningFetcher()
        
    cached_before = fetcher.get_cached(symbol) is not None
    snapshot = await fetcher.fetch_positioning(symbol)
    
    cot = CryptoCOT()
    analysis = cot.analyze(snapshot, funding_rate=funding_rate, symbol=symbol)
    
    row_id = record_shadow_observation(
        snapshot=snapshot,
        analysis_regime=analysis.regime.value,
        reasons=analysis.reasons,
        current_price=current_price,
        funding_rate=funding_rate,
        cache_hit=cached_before,
        db_path=db_path,
    )
    
    return {
        "id": row_id,
        "symbol": symbol,
        "observed_at": snapshot.observed_at,
        "regime": analysis.regime.value,
        "global_account_ratio": snapshot.global_account_ratio,
        "top_position_ratio": snapshot.top_position_ratio,
        "oi_delta_1h": snapshot.oi_delta_1h,
        "is_stale": snapshot.is_stale,
    }


async def run_positioning_shadow_daemon(
    interval_seconds: float = 300.0,
    symbol: str = "BTCUSDT",
    db_path: str = DEFAULT_DB_PATH,
    stop_event: Optional[asyncio.Event] = None,
):
    """Loop contínuo assíncrono para coleta de posicionamento Binance USDM a cada 5m."""
    fetcher = BinancePositioningFetcher(cache_ttl=interval_seconds)
    logger.info(f"Iniciando daemon de coleta de Positioning (intervalo={interval_seconds}s, symbol={symbol})...")
    while True:
        if stop_event and stop_event.is_set():
            logger.info("Encerrando daemon de posicionamento.")
            break
        try:
            res = await collect_single_shadow_sample(symbol=symbol, fetcher=fetcher, db_path=db_path)
            logger.info(
                f"[POSITIONING SHADOW] Amostra ID #{res['id']} gravada. "
                f"Regime={res['regime']}, GA={res['global_account_ratio']}, "
                f"TP={res['top_position_ratio']}, stale={res['is_stale']}"
            )
        except Exception as e:
            logger.warning(f"[POSITIONING SHADOW] Erro na coleta periódica: {e}")

        try:
            if stop_event:
                await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
                break
            else:
                await asyncio.sleep(interval_seconds)
        except asyncio.TimeoutError:
            pass


if __name__ == "__main__":
    async def main():
        print("Coletando amostra de teste para o shadow dataset...")
        res = await collect_single_shadow_sample(symbol="BTCUSDT")
        print(f"Amostra gravada com sucesso: ID #{res['id']} | Regime: {res['regime']}")
        
    asyncio.run(main())

