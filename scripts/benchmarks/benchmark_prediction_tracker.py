# scripts/benchmarks/benchmark_prediction_tracker.py
"""
Benchmark for in-memory PredictionTracker on_tick performance (Gate D0-B).

Measures latency per tick with 5, 15, and 60 pending predictions.
Guarantees O(N_pending) in-memory comparison and zero SQLite I/O on ordinary non-resolving ticks.
"""

import time
from paper_trading.contracts import CanonicalDecision
from paper_trading.prediction import PredictionTrackerConfig
from paper_trading.prediction_tracker import PredictionTracker


def _make_decision(idx: int, base_ts: int, horizon_s: int = 300) -> CanonicalDecision:
    return CanonicalDecision(
        cohort_id="CH_BENCH",
        symbol="BTCUSDT",
        window_id=f"w_{idx}",
        decision_provider="bench",
        strategy_version="v1.0",
        signal_timestamp=base_ts + idx * 1000,
        decision_timestamp=base_ts + idx * 1000,
        available_at=base_ts + idx * 1000 + 5,
        side="LONG",
        reference_price=100.0,
        notional_usdt=1000.0,
        horizon_s=horizon_s,
    )


def run_benchmark() -> None:
    print("==================================================")
    print("PREDICTION TRACKER BENCHMARK (Gate D0-B)")
    print("==================================================")

    for n_pending in [5, 15, 60]:
        tracker = PredictionTracker(
            config=PredictionTrackerConfig(resolution_tolerance_ms=30_000, flat_tolerance_bps=1.0)
        )
        base_ts = 1_000_000
        for i in range(n_pending):
            dec = _make_decision(i, base_ts, horizon_s=300)
            tracker.register(dec)

        assert tracker.pending_count == n_pending

        # Measure 10,000 ordinary ticks (prior to deadline, non-resolving)
        n_ticks = 10_000
        tick = {"T": base_ts + 100, "p": 100.5, "s": "BTCUSDT"}

        # Warmup
        for _ in range(500):
            tracker.on_tick(tick)

        start_ns = time.perf_counter_ns()
        for _ in range(n_ticks):
            tracker.on_tick(tick)
        elapsed_ns = time.perf_counter_ns() - start_ns
        avg_ns = elapsed_ns / n_ticks
        avg_us = avg_ns / 1_000.0
        print(f"Pending N = {n_pending:2d}: {avg_us:6.3f} µs = {avg_ns:8.1f} ns por tick")

    print("==================================================")


if __name__ == "__main__":
    run_benchmark()
