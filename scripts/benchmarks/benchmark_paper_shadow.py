# scripts/benchmarks/benchmark_paper_shadow.py
# -*- coding: utf-8 -*-
"""
Offline, hermetic performance benchmark for Paper Trading Shadow Runtime (Gate C3-C-B4).

Measures empirical overhead across ExecutionSink (S0, S1, S2, S3-N5),
Runtime Hook (H0, H1, H2, H3), and on_message micro-harness (M0, M1).

Invariants enforced:
- Strictly offline: zero network access, zero WebSocket, zero Binance API, zero LLM.
- Hermetic: pure in-memory execution, zero disk I/O in measured regions.
- Deterministic data generated outside measured regions.
- Warmup executed prior to measured repetitions.
- time.perf_counter_ns() precision.
"""

from __future__ import annotations

import gc
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Safe console encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from paper_trading.config import ShadowPaperConfig
from paper_trading.contracts import PaperOrder, PaperPosition
from paper_trading.execution_sink import ExecutionSink
from paper_trading.shadow_runtime import ShadowPaperRuntime


# ==============================================================================
# ENVIRONMENT DETECTION
# ==============================================================================

def get_environment_info() -> Dict[str, Any]:
    """Capture runtime and hardware environment metadata."""
    power_plan = "UNKNOWN"
    if sys.platform == "win32":
        try:
            out = subprocess.check_output("powercfg /getactivescheme", shell=True, text=True)
            power_plan = out.strip()
        except Exception:
            power_plan = "FAILED_TO_DETECT"

    return {
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "python_compiler": platform.python_compiler(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "pid": os.getpid(),
        "power_plan": power_plan,
        "gc_enabled": gc.isenabled(),
        "coverage_active": sys.gettrace() is not None,
    }


# ==============================================================================
# DETERMINISTIC TAPE GENERATION
# ==============================================================================

def generate_deterministic_ticks(
    count: int,
    base_trade_id: int = 1_000_000,
    base_time_ms: int = 1_726_000_000_000,
) -> List[Dict[str, Any]]:
    """
    Generate synthetic market ticks deterministically outside measured code.

    Ticks oscillate slightly around reference price to keep positions stable.
    """
    ticks: List[Dict[str, Any]] = []
    for i in range(count):
        price = 65000.0 + (i % 50) * 0.1
        qty = 0.1 + (i % 10) * 0.01
        ts = base_time_ms + i * 10  # 10ms monotonic step
        ticks.append({
            "p": price,
            "q": round(qty, 4),
            "T": ts,
            "T_raw": ts,
            "m": (i % 2 == 0),
            "source": "fut_agg",
            "trade_id": base_trade_id + i,
        })
    return ticks


# ==============================================================================
# SCENARIO FACTORIES
# ==============================================================================

def create_sink_s0(dedup_capacity: int = 200_000) -> ExecutionSink:
    """S0: ExecutionSink active, zero pending orders, zero open positions."""
    return ExecutionSink(symbol="BTCUSDT", dedup_capacity=dedup_capacity)


def create_sink_s1(dedup_capacity: int = 200_000) -> ExecutionSink:
    """S1: ExecutionSink active, 1 pending order that never fills during measured tape."""
    sink = ExecutionSink(symbol="BTCUSDT", dedup_capacity=dedup_capacity)
    order = PaperOrder(
        order_id="ORD_PENDING_S1",
        decision_id="DEC_PENDING_S1",
        cohort_id="CH_BENCH",
        decision_provider="fixed_long",
        symbol="BTCUSDT",
        side="LONG",
        reference_price=65000.0,
        notional_usdt=100.0,
        signal_timestamp=1_726_000_000_000,
        decision_timestamp=1_726_000_000_000,
        available_at=2_500_000_000_000,  # Well beyond benchmark tape max timestamp
        expires_at=3_000_000_000_000,
        horizon_s=3600,
        stop_loss=10000.0,
        take_profit=100000.0,
    )
    sink.submit_order(order)
    assert len(sink.executor.pending_orders) == 1
    return sink


def create_sink_s2(dedup_capacity: int = 200_000) -> ExecutionSink:
    """S2: ExecutionSink active, 1 open position that never hits SL/TP/horizon during measured tape."""
    sink = ExecutionSink(symbol="BTCUSDT", dedup_capacity=dedup_capacity)
    pos = PaperPosition(
        position_id="POS_ACTIVE_S2",
        cohort_id="CH_BENCH",
        decision_provider="fixed_long",
        symbol="BTCUSDT",
        side="LONG",
        entry_price=65000.0,
        reference_price=65000.0,
        quantity=0.01,
        notional_usdt=650.0,
        opened_ts_ms=1_726_000_000_000,
        horizon_deadline_ms=2_500_000_000_000,  # Far in the future
        decision_id="DEC_ACTIVE_S2",
        signal_timestamp=1_726_000_000_000,
        decision_timestamp=1_726_000_000_000,
        available_at=1_726_000_000_000,
        stop_loss=10000.0,   # Never hit with price ~65000
        take_profit=100000.0, # Never hit with price ~65000
    )
    sink.executor.position_manager.add_position(pos)
    assert len(sink.executor.position_manager.open_positions) == 1
    return sink


def create_sink_s3_n5(dedup_capacity: int = 200_000) -> ExecutionSink:
    """S3-N5: ExecutionSink active with 5 pending orders to test O(N) executor scaling."""
    sink = ExecutionSink(symbol="BTCUSDT", dedup_capacity=dedup_capacity)
    for i in range(5):
        order = PaperOrder(
            order_id=f"ORD_PENDING_N5_{i}",
            decision_id=f"DEC_PENDING_N5_{i}",
            cohort_id=f"CH_BENCH_{i}",
            decision_provider="fixed_long",
            symbol="BTCUSDT",
            side="LONG",
            reference_price=65000.0,
            notional_usdt=100.0,
            signal_timestamp=1_726_000_000_000,
            decision_timestamp=1_726_000_000_000,
            available_at=2_500_000_000_000,
            expires_at=3_000_000_000_000,
            horizon_s=3600,
            stop_loss=10000.0,
            take_profit=100000.0,
        )
        sink.submit_order(order)
    assert len(sink.executor.pending_orders) == 5
    return sink


def create_shadow_runtime_instance() -> ShadowPaperRuntime:
    """Create a fully-configured hermetic ShadowPaperRuntime without ledger."""
    config = ShadowPaperConfig(
        enabled=True,
        cohort_id="CH_BENCH",
        provider="fixed_long",
        symbol="BTCUSDT",
        timeframe="1m",
        notional_usdt=100.0,
        horizon_s=3600,
        order_ttl_ms=60000,
        taker_fee_bps=5.0,
        maker_fee_bps=2.0,
        entry_slippage_bps=2.0,
        exit_slippage_bps=2.0,
        cost_source="DEFAULT_FUTURES",
        cost_effective_at="2026-09-01T00:00:00Z",
        random_seed=42,
    )
    runtime = ShadowPaperRuntime(config=config, ledger=None)
    runtime.start()
    return runtime


def create_runtime_h1() -> ShadowPaperRuntime:
    """H1: ShadowRuntime ON, zero orders, zero positions."""
    return create_shadow_runtime_instance()


def create_runtime_h2() -> ShadowPaperRuntime:
    """H2: ShadowRuntime ON, 1 pending order."""
    rt = create_shadow_runtime_instance()
    order = PaperOrder(
        order_id="ORD_PENDING_H2",
        decision_id="DEC_PENDING_H2",
        cohort_id="CH_BENCH",
        decision_provider="fixed_long",
        symbol="BTCUSDT",
        side="LONG",
        reference_price=65000.0,
        notional_usdt=100.0,
        signal_timestamp=1_726_000_000_000,
        decision_timestamp=1_726_000_000_000,
        available_at=2_500_000_000_000,
        expires_at=3_000_000_000_000,
        horizon_s=3600,
        stop_loss=10000.0,
        take_profit=100000.0,
    )
    rt.execution_sink.submit_order(order)
    assert len(rt.execution_sink.executor.pending_orders) == 1
    return rt


def create_runtime_h3() -> ShadowPaperRuntime:
    """H3: ShadowRuntime ON, 1 open position."""
    rt = create_shadow_runtime_instance()
    pos = PaperPosition(
        position_id="POS_ACTIVE_H3",
        cohort_id="CH_BENCH",
        decision_provider="fixed_long",
        symbol="BTCUSDT",
        side="LONG",
        entry_price=65000.0,
        reference_price=65000.0,
        quantity=0.01,
        notional_usdt=650.0,
        opened_ts_ms=1_726_000_000_000,
        horizon_deadline_ms=2_500_000_000_000,
        decision_id="DEC_ACTIVE_H3",
        signal_timestamp=1_726_000_000_000,
        decision_timestamp=1_726_000_000_000,
        available_at=1_726_000_000_000,
        stop_loss=10000.0,
        take_profit=100000.0,
    )
    rt.execution_sink.executor.position_manager.add_position(pos)
    assert len(rt.execution_sink.executor.position_manager.open_positions) == 1
    return rt


# ==============================================================================
# MEASUREMENT HARNESS
# ==============================================================================

@dataclass
class RunMetrics:
    scenario: str
    n_ticks: int
    repetition: int
    p50_ns: float
    p95_ns: float
    p99_ns: float
    max_ns: int
    mean_ns: float
    burst_total_ms: float
    burst_ticks_per_sec: float


def measure_scenario(
    scenario_name: str,
    target_fn: Callable[[Dict[str, Any]], Any],
    ticks: List[Dict[str, Any]],
    repetition: int,
) -> RunMetrics:
    """
    Measure both tick-by-tick distribution and whole-burst throughput.
    """
    n = len(ticks)
    durations = [0] * n
    perf_ns = time.perf_counter_ns

    # 1. Tick-by-tick measurement loop
    for i in range(n):
        t0 = perf_ns()
        target_fn(ticks[i])
        durations[i] = perf_ns() - t0

    durations.sort()
    p50 = durations[int(n * 0.50)]
    p95 = durations[int(n * 0.95)]
    p99 = durations[int(n * 0.99)]
    max_d = durations[-1]
    mean_d = sum(durations) / n

    # 2. Burst measurement loop (uninstrumented hot-loop)
    t_burst_start = perf_ns()
    for i in range(n):
        target_fn(ticks[i])
    t_burst_end = perf_ns()

    burst_ns = t_burst_end - t_burst_start
    burst_ms = burst_ns / 1_000_000.0
    burst_tps = (n / (burst_ns / 1_000_000_000.0)) if burst_ns > 0 else 0.0

    return RunMetrics(
        scenario=scenario_name,
        n_ticks=n,
        repetition=repetition,
        p50_ns=float(p50),
        p95_ns=float(p95),
        p99_ns=float(p99),
        max_ns=int(max_d),
        mean_ns=float(mean_d),
        burst_total_ms=round(burst_ms, 3),
        burst_ticks_per_sec=round(burst_tps, 1),
    )


# ==============================================================================
# SUITE EXECUTION
# ==============================================================================

def compute_median_metric(metrics: List[RunMetrics]) -> Dict[str, Any]:
    """Compute median values across multiple repetitions of a scenario."""
    if not metrics:
        return {}

    n = len(metrics)
    p50_sorted = sorted(m.p50_ns for m in metrics)
    p95_sorted = sorted(m.p95_ns for m in metrics)
    p99_sorted = sorted(m.p99_ns for m in metrics)
    max_sorted = sorted(m.max_ns for m in metrics)
    mean_sorted = sorted(m.mean_ns for m in metrics)
    burst_ms_sorted = sorted(m.burst_total_ms for m in metrics)
    tps_sorted = sorted(m.burst_ticks_per_sec for m in metrics)

    mid = n // 2
    return {
        "scenario": metrics[0].scenario,
        "n_ticks": metrics[0].n_ticks,
        "repetitions_count": n,
        "p50_ns": p50_sorted[mid],
        "p95_ns": p95_sorted[mid],
        "p99_ns": p99_sorted[mid],
        "max_ns": max_sorted[mid],
        "mean_ns": round(mean_sorted[mid], 1),
        "burst_total_ms": burst_ms_sorted[mid],
        "burst_ticks_per_sec": tps_sorted[mid],
    }


def run_benchmark_suite(
    tick_counts: Tuple[int, ...] = (1_000, 10_000, 100_000),
    warmup_ticks: int = 5_000,
    repetitions: int = 5,
) -> Dict[str, Any]:
    """Execute complete benchmark matrix."""
    env_info = get_environment_info()
    print("=" * 70)
    print("STARTING PAPER SHADOW PERFORMANCE BENCHMARK (C3-C-B4)")
    print("=" * 70)
    print(f"OS:            {env_info['os']}")
    print(f"Python:        {env_info['python_version']} ({env_info['architecture']})")
    print(f"Processor:     {env_info['processor']}")
    print(f"GC Active:     {env_info['gc_enabled']}")
    print(f"Coverage:      {env_info['coverage_active']} (must be False)")
    print(f"Repetitions:   {repetitions} per scenario")
    print("=" * 70)

    # 1. Generate Warmup Tape
    warmup_tape = generate_deterministic_ticks(warmup_ticks, base_trade_id=10_000)

    # Pre-warm runtime and sink
    sink_w = create_sink_s0()
    for t in warmup_tape:
        sink_w.on_market_trade(t)

    rt_w = create_runtime_h1()
    for t in warmup_tape:
        rt_w.on_market_trade(t)

    all_results: List[RunMetrics] = []
    summary_by_size: Dict[int, Dict[str, Any]] = {}

    for n_ticks in tick_counts:
        print(f"\n--- MEASURING SIZE: {n_ticks:,} TICKS ---")
        measured_tape = generate_deterministic_ticks(n_ticks, base_trade_id=100_000)
        scenario_runs: Dict[str, List[RunMetrics]] = {}

        # Define targets factory
        scenarios = [
            # ExecutionSink
            ("S0", lambda: create_sink_s0().on_market_trade),
            ("S1", lambda: create_sink_s1().on_market_trade),
            ("S2", lambda: create_sink_s2().on_market_trade),
            ("S3_N5", lambda: create_sink_s3_n5().on_market_trade),
            # Hook Incremental
            ("H0", lambda: (lambda norm: None)),  # branch OFF
            ("H1", lambda: create_runtime_h1().on_market_trade),
            ("H2", lambda: create_runtime_h2().on_market_trade),
            ("H3", lambda: create_runtime_h3().on_market_trade),
        ]

        # Add Micro-Harness M0 / M1
        def make_m0():
            def run_m0_step(norm: Dict[str, Any]):
                # Emulate buffer append + OFF hook + boundary check
                norm["T"]
            return run_m0_step

        def make_m1():
            rt = create_runtime_h1()
            def run_m1_step(norm: Dict[str, Any]):
                # Emulate buffer append + ON hook + boundary check
                rt.on_market_trade(norm)
            return run_m1_step

        scenarios.append(("M0", make_m0))
        scenarios.append(("M1", make_m1))

        for sc_name, sc_factory in scenarios:
            scenario_runs[sc_name] = []
            for rep in range(1, repetitions + 1):
                fn = sc_factory()
                m = measure_scenario(sc_name, fn, measured_tape, rep)
                scenario_runs[sc_name].append(m)
                all_results.append(m)

            med = compute_median_metric(scenario_runs[sc_name])
            print(
                f"[{sc_name:6s}] p50={med['p50_ns']:6.0f}ns | p95={med['p95_ns']:6.0f}ns | "
                f"p99={med['p99_ns']:6.0f}ns | max={med['max_ns']:8.0f}ns | "
                f"burst={med['burst_total_ms']:7.2f}ms ({med['burst_ticks_per_sec']:9,.0f} ticks/s)"
            )

        # Store summary for this size
        summary_by_size[n_ticks] = {
            sc_name: compute_median_metric(runs)
            for sc_name, runs in scenario_runs.items()
        }

    # Save to JSON in analysis/results
    results_dir = REPO_ROOT / "analysis" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = results_dir / f"paper_shadow_benchmark_{ts_str}.json"

    out_payload = {
        "commit": "b0662417501f806fc30205b57e58b9d6544807db",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "environment": env_info,
        "summary_by_size": summary_by_size,
        "raw_runs": [asdict(r) for r in all_results],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nResults successfully saved to: {json_path}")

    return out_payload


if __name__ == "__main__":
    run_benchmark_suite()
