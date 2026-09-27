# scripts/analytics/benchmark_effort_response_transport.py
"""
Script de medição e benchmark institucional para P1-F Etapa 2M.

Mede:
1. DTO build time (microssegundos)
2. submit_nowait latency no hot path (microssegundos)
3. Tamanho serializado em bytes (CREATE vs RESOLVED)
4. Throughput do writer (records/segundo)
5. Overhead hot path: flag OFF vs flag ON (enqueue-only)
"""
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from flow_analyzer.effort_response_dataset import build_shadow_record
from flow_analyzer.effort_response_transport import (
    EffortResponseSnapshotDTO,
    ShadowAsyncTransport,
)


def run_benchmark():
    print("=== P1-F ETAPA 2M: BENCHMARK SUITE ===")

    # 1. DTO build time
    n_iterations = 10_000
    t0 = time.perf_counter()
    for i in range(n_iterations):
        dto = EffortResponseSnapshotDTO(
            symbol="BTCUSDT",
            causal_anchor_ms=1788702420000 + i * 60_000,
            observation_open_ms=1788702361157 + i * 60_000,
            observation_close_ms=1788702418610 + i * 60_000,
            buy_notional_usd=6816945.1591,
            sell_notional_usd=1149280.5758,
            open=79776.9,
            high=79810.8,
            low=79776.9,
            close=79792.7,
            window_duration_ms=57453,
            vwap=79803.5,
            poc=79804.9,
            context_data={"regime_current_at_t": "TRENDING_EXPANSION"},
        )
    t1 = time.perf_counter()
    dto_build_us = ((t1 - t0) / n_iterations) * 1_000_000.0
    print(f"1. DTO Build Time: {dto_build_us:.3f} us por snapshot (n={n_iterations})")

    # 2. Tamanho serializado em disco (CREATE vs UPDATE)
    window_data = {
        "buy_notional_usd": 6816945.1591,
        "sell_notional_usd": 1149280.5758,
        "open": 79776.9,
        "high": 79810.8,
        "low": 79776.9,
        "close": 79792.7,
        "window_duration_ms": 57453,
        "vwap": 79803.5,
        "poc": 79804.9,
    }
    rec = build_shadow_record(
        symbol="BTCUSDT",
        window_open_ms=1788702361157,
        window_close_ms=1788702418610,
        window_data=window_data,
        causal_anchor_ms=1788702420000,
        observation_open_ms=1788702361157,
        observation_close_ms=1788702418610,
    )
    raw_json_create = json.dumps(rec.to_dict(), separators=(",", ":"))
    size_create_bytes = len(raw_json_create.encode("utf-8"))
    print(f"2a. Tamanho serializado CREATE: {size_create_bytes} bytes (~{size_create_bytes/1024:.2f} KB)")

    # 3. submit_nowait Latency (Enqueue-only sem I/O no hot path)
    temp_dir = tempfile.mkdtemp(prefix="bench_shadow_")
    bench_file = Path(temp_dir) / "bench.jsonl"
    transport_on = ShadowAsyncTransport(filepath=bench_file, queue_capacity=20_000, enabled=True, start_worker=False)

    t0 = time.perf_counter()
    for i in range(n_iterations):
        dto = EffortResponseSnapshotDTO(
            symbol="BTCUSDT",
            causal_anchor_ms=1788702420000 + i * 60_000,
            observation_open_ms=1788702361157 + i * 60_000,
            observation_close_ms=1788702418610 + i * 60_000,
            buy_notional_usd=6816945.1591,
            sell_notional_usd=1149280.5758,
            open=79776.9,
            high=79810.8,
            low=79776.9,
            close=79792.7,
            window_duration_ms=57453,
        )
        transport_on.submit_nowait(dto)
    t1 = time.perf_counter()
    submit_latency_us = ((t1 - t0) / n_iterations) * 1_000_000.0
    print(f"3. submit_nowait Latency (hot path): {submit_latency_us:.3f} us por chamada")

    # 4. Overhead Flag OFF vs Flag ON
    transport_off = ShadowAsyncTransport(filepath=bench_file, enabled=False)
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        transport_off.submit_nowait(dto)
    t1 = time.perf_counter()
    off_latency_us = ((t1 - t0) / n_iterations) * 1_000_000.0
    print(f"4. Flag OFF submit_nowait: {off_latency_us:.3f} us por chamada (overhead praticamente nulo)")

    # 5. Throughput do Writer Thread (I/O em background)
    writer_file = Path(temp_dir) / "writer_bench.jsonl"
    transport_writer = ShadowAsyncTransport(filepath=writer_file, queue_capacity=5000, enabled=True, start_worker=True)

    t_writer_start = time.perf_counter()
    n_writer = 2000
    for i in range(n_writer):
        dto = EffortResponseSnapshotDTO(
            symbol="BTCUSDT",
            causal_anchor_ms=1788702420000 + i * 60_000,
            observation_open_ms=1788702361157 + i * 60_000,
            observation_close_ms=1788702418610 + i * 60_000,
            buy_notional_usd=6816945.1591,
            sell_notional_usd=1149280.5758,
            open=79776.9,
            high=79810.8,
            low=79776.9,
            close=79792.7,
            window_duration_ms=57453,
        )
        transport_writer.submit_nowait(dto)

    transport_writer.flush(timeout=10.0)
    t_writer_end = time.perf_counter()
    duration_s = t_writer_end - t_writer_start
    throughput = n_writer / duration_s
    print(f"5. Single-Writer Throughput: {throughput:.1f} records/s ({duration_s:.3f}s para {n_writer} records)")

    transport_writer.close(timeout=1.0)
    transport_on.close(timeout=1.0)
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("=== BENCHMARK CONCLUÍDO ===")


if __name__ == "__main__":
    run_benchmark()
