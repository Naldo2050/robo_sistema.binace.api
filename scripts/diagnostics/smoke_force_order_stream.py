#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/diagnostics/smoke_force_order_stream.py

P2-D2 — Binance forceOrder Live Smoke Test.

Objetivo:
- Conectar à stream pública real Binance USD-M Futures (<symbol>@forceOrder).
- Confirmar handshake e estado CONNECTED.
- Se eventos chegarem, validar schema real e serialização RFC 8259.
- Se nenhum evento chegar no tempo estipulado, reportar:
  CONNECTED / ZERO EVENTS OBSERVED
  e isso é considerado SUCESSO de transporte (exit code 0).
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
import sys
import time
from typing import List

# Garante raiz do repositório no sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Fix encoding Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fetchers.binance_liquidation_stream import (
    BinanceLiquidationListener,
    ConnectionCoverageStatus,
    ConnectionIntervalTracker,
    ForcedLiquidationEvent,
    LiquidationValidity,
    LiquidationWindowAggregator,
    StreamConnectionStatus,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("smoke_force_order")


async def run_smoke(symbol: str, duration_sec: float) -> int:
    logger.info(f"=== INICIANDO LIVE SMOKE: {symbol.upper()}@forceOrder ===")
    logger.info(f"Duração configurada: {duration_sec}s")

    aggregator = LiquidationWindowAggregator(symbol=symbol)
    tracker = ConnectionIntervalTracker()
    received_events: List[ForcedLiquidationEvent] = []

    def on_event(ev: ForcedLiquidationEvent) -> None:
        received_events.append(ev)
        logger.info(
            f"⚡ EVENTO RECEBIDO: {ev.symbol} | side={ev.order_side} | "
            f"liquidated={ev.liquidated_position_side} | "
            f"notional_obs=${ev.observed_notional_usd} | "
            f"notional_est=${ev.estimated_notional_usd} | "
            f"quality={ev.notional_quality} | validity={ev.validity.value}"
        )

    listener = BinanceLiquidationListener(
        symbol=symbol,
        aggregator=aggregator,
        tracker=tracker,
        on_event_callback=on_event,
    )

    logger.info(f"Conectando ao endpoint: {listener.stream_url}")
    start_time = time.time()
    task = listener.start()

    # 1. Aguarda handshake e status CONNECTED (máximo 8s)
    connected = False
    connect_timeout = 8.0
    while time.time() - start_time < connect_timeout:
        if listener.is_connected:
            connected = True
            break
        await asyncio.sleep(0.1)

    if not connected:
        logger.error(f"❌ FALHA DE HANDSHAKE: não foi possível conectar em {connect_timeout}s.")
        await listener.stop()
        return 1

    handshake_latency_ms = (time.time() - start_time) * 1000.0
    logger.info(f"✅ HANDSHAKE CONFIRMADO! Estado: CONNECTED (latência de conexão: {handshake_latency_ms:.1f}ms)")

    # 2. Escuta durante duration_sec
    logger.info(f"Escutando stream por {duration_sec}s...")
    listen_start = time.time()
    while time.time() - listen_start < duration_sec:
        await asyncio.sleep(0.2)

    listen_end_ms = int(time.time() * 1000)
    listen_start_ms = int(listen_start * 1000)

    # 3. Encerramento gracioso
    logger.info("Encerrando conexão WebSocket graciosamente...")
    await listener.stop()

    # 4. Avaliação de cobertura e resumo
    coverage = tracker.evaluate_coverage(listen_start_ms, listen_end_ms)
    logger.info(f"Cobertura temporal do transporte na janela de teste: {coverage.value}")

    summary = aggregator.summarize_window(
        window_start_ms=listen_start_ms,
        window_end_ms=listen_end_ms,
        stream_healthy=True,
        connection_status=StreamConnectionStatus.CONNECTED,
        connection_coverage_status=coverage,
    )

    print("\n" + "=" * 60)
    print("RELATÓRIO DE RESULTADO DO SMOKE TEST")
    print("=" * 60)
    print(f"Status Transporte : {summary.connection_status}")
    print(f"Coverage Status   : {summary.connection_coverage_status}")
    print(f"Eventos Recebidos : {len(received_events)}")
    print(f"Summary Event Cnt : {summary.event_count}")
    print(f"Summary Validity  : {summary.validity.value}")
    print(f"Summary Reason    : {summary.reason}")

    if received_events:
        print("\n--- AMOSTRA DO PRIMEIRO EVENTO OBSERVADO ---")
        ev0 = received_events[0]
        ev_dict = ev0.to_dict()
        print(json.dumps(ev_dict, indent=2))
        print("Schema real validado com sucesso!")
    else:
        print("\n[RESULTADO] CONNECTED / ZERO EVENTS OBSERVED")
        print("Natureza event-sparse confirmada. Sucesso operacional do transporte.")

    print("=" * 60 + "\n")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Smoke Test para Binance forceOrder stream.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Símbolo Futures (default: BTCUSDT)")
    parser.add_argument("--duration", type=float, default=5.0, help="Duração da escuta em segundos (default: 5.0)")
    args = parser.parse_args()

    exit_code = asyncio.run(run_smoke(symbol=args.symbol, duration_sec=args.duration))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
