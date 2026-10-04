# tests/unit/paper_trading/test_execution_capacity_c3_4.py
"""
Unit test suite for Gate C3.4 / C3.4b: Atomic Enforcement of max_open_positions
using pending_orders as natural capacity reservations.

Enforces:
- Invariant under ExecutionSink._lock:
  len(open_positions) + len(pending_orders) <= max_open_positions
- Precedence of DUPLICATE_DECISION and POSITION_OPEN before POSITION_LIMIT_EXCEEDED.
- Zero state duplication: no artificial _reservations table.
- Atomic state transition during fill: effective capacity remains constant.
- Clean release of capacity upon order TTL expiry.
- Thread concurrency safety without lock inversions or deadlocks.
- Strict positive integer semantics for max_open_positions (> 0).
- Factually derived risk_position_limit_active flag from the actual sink in use.
- Fail-fast validation against capacity limit mismatches between RiskConfig and injected ExecutionSink.
"""

from __future__ import annotations

import concurrent.futures
import inspect
import threading
import time
from typing import List, Optional

import pytest

from paper_trading.config import ShadowPaperConfig
from paper_trading.contracts import PaperOrder, PaperPosition, Rejection
from paper_trading.cost_model import CostModel
from paper_trading.execution_sink import ExecutionSink, SubmitStatus
from paper_trading.executor import ExecutorConfig, PaperExecutor
from paper_trading.positions import PositionManager
from paper_trading.shadow_runtime import ShadowPaperRuntime
import paper_trading.shadow_runtime as shadow_runtime_module
from risk_management.risk_manager import RiskConfig, RiskManager


def _build_order(
    symbol: str = "BTCUSDT",
    order_id: str = "ord_test_001",
    decision_id: str = "dec_test_001",
    cohort_id: str = "CH_TEST",
    provider: str = "prov_a",
    side: str = "LONG",
    reference_price: float = 50_000.0,
    notional_usdt: float = 1_000.0,
    signal_timestamp: int = 1_000,
    decision_timestamp: int = 1_000,
    available_at: int = 1_100,
    expires_at: int = 6_100,
    horizon_s: int = 60,
    stop_loss: Optional[float] = 49_000.0,
    take_profit: Optional[float] = 52_000.0,
) -> PaperOrder:
    return PaperOrder(
        order_id=order_id,
        decision_id=decision_id,
        cohort_id=cohort_id,
        decision_provider=provider,
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        reference_price=reference_price,
        notional_usdt=notional_usdt,
        signal_timestamp=signal_timestamp,
        decision_timestamp=decision_timestamp,
        available_at=available_at,
        expires_at=expires_at,
        horizon_s=horizon_s,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )


def test_1_max_open_positions_single_slot_concurrency():
    """1. max=1, duas ordens concorrentes em símbolos distintos: exatamente 1 aceita, 1 rejeitada."""
    executor = PaperExecutor(config=ExecutorConfig(max_open_positions=1))
    sink = ExecutionSink(symbol="BTCUSDT", executor=executor)

    order1 = _build_order(symbol="BTCUSDT", order_id="ord_1", decision_id="dec_1")
    order2 = _build_order(symbol="ETHUSDT", order_id="ord_2", decision_id="dec_2")

    results = []

    def _submit(ord_item: PaperOrder):
        return sink.submit_order(ord_item)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futs = [pool.submit(_submit, order1), pool.submit(_submit, order2)]
        for f in futs:
            results.append(f.result())

    statuses = [r.status for r in results]
    assert statuses.count(SubmitStatus.ACCEPTED) == 1
    assert statuses.count(SubmitStatus.REJECTED) == 1

    rej_result = [r for r in results if r.status == SubmitStatus.REJECTED][0]
    assert rej_result.rejection is not None
    assert rej_result.rejection.reason == "POSITION_LIMIT_EXCEEDED"

    snap = sink.get_capacity_snapshot()
    assert snap["effective_positions"] == 1
    assert snap["available_capacity"] == 0


def test_2_max_open_positions_n_slots():
    """2. max=N, N ordens aceitas, N+1 rejeitada."""
    n = 3
    executor = PaperExecutor(config=ExecutorConfig(max_open_positions=n))
    sink = ExecutionSink(symbol="BTCUSDT", executor=executor)

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT"]
    orders = [
        _build_order(symbol=symbols[i], order_id=f"ord_{i}", decision_id=f"dec_{i}")
        for i in range(4)
    ]

    # Submete as 3 primeiras
    for i in range(3):
        res = sink.submit_order(orders[i])
        assert res.status == SubmitStatus.ACCEPTED
        snap = sink.get_capacity_snapshot()
        assert snap["effective_positions"] == i + 1

    # Submete a 4ª (estouro de capacidade)
    res_overflow = sink.submit_order(orders[3])
    assert res_overflow.status == SubmitStatus.REJECTED
    assert res_overflow.rejection is not None
    assert res_overflow.rejection.reason == "POSITION_LIMIT_EXCEEDED"

    snap_final = sink.get_capacity_snapshot()
    assert snap_final["effective_positions"] == 3
    assert snap_final["available_capacity"] == 0


def test_3_pending_order_counts_as_capacity():
    """3. Uma pending order consome capacidade imediatamente na admissão."""
    executor = PaperExecutor(config=ExecutorConfig(max_open_positions=2))
    sink = ExecutionSink(symbol="BTCUSDT", executor=executor)

    assert sink.get_capacity_snapshot()["available_capacity"] == 2

    order1 = _build_order(symbol="BTCUSDT", order_id="ord_1", decision_id="dec_1")
    res1 = sink.submit_order(order1)
    assert res1.status == SubmitStatus.ACCEPTED

    snap = sink.get_capacity_snapshot()
    assert snap["pending_orders"] == 1
    assert snap["open_positions"] == 0
    assert snap["effective_positions"] == 1
    assert snap["available_capacity"] == 1


def test_4_open_position_counts_as_capacity():
    """4. Uma open position conta como capacidade."""
    pos_mgr = PositionManager()
    pos = PaperPosition(
        position_id="pos_1",
        cohort_id="CH_TEST",
        decision_provider="prov_a",
        symbol="BTCUSDT",
        side="LONG",
        entry_price=50000.0,
        reference_price=50000.0,
        quantity=0.02,
        notional_usdt=1000.0,
        opened_ts_ms=1000,
        horizon_deadline_ms=61000,
        decision_id="dec_pos_1",
        signal_timestamp=1000,
        decision_timestamp=1000,
        available_at=1100,
    )
    pos_mgr.add_position(pos)

    executor = PaperExecutor(position_manager=pos_mgr, config=ExecutorConfig(max_open_positions=1))
    sink = ExecutionSink(symbol="BTCUSDT", executor=executor)

    snap = sink.get_capacity_snapshot()
    assert snap["open_positions"] == 1
    assert snap["pending_orders"] == 0
    assert snap["effective_positions"] == 1
    assert snap["available_capacity"] == 0

    # Nova ordem para outro símbolo é rejeitada por limite de capacidade
    order2 = _build_order(symbol="ETHUSDT", order_id="ord_2", decision_id="dec_2")
    res2 = sink.submit_order(order2)
    assert res2.status == SubmitStatus.REJECTED
    assert res2.rejection is not None
    assert res2.rejection.reason == "POSITION_LIMIT_EXCEEDED"


def test_5_fill_conserves_effective_capacity():
    """5. Transição PENDING -> OPEN conserva rigorosamente a capacidade efetiva."""
    executor = PaperExecutor(config=ExecutorConfig(max_open_positions=2))
    sink = ExecutionSink(symbol="BTCUSDT", executor=executor)

    order1 = _build_order(
        symbol="BTCUSDT",
        order_id="ord_1",
        decision_id="dec_1",
        available_at=1100,
        expires_at=6100,
    )
    sink.submit_order(order1)

    snap_before = sink.get_capacity_snapshot()
    assert snap_before["open_positions"] == 0
    assert snap_before["pending_orders"] == 1
    assert snap_before["effective_positions"] == 1

    # Market tick que preenche a ordem (T=1200 >= available_at=1100)
    norm_trade = {
        "p": 50000.0,
        "q": 0.5,
        "T": 1200,
        "T_raw": 1200,
        "m": False,
        "source": "aggTrade",
        "trade_id": 99991,
    }
    trade_res = sink.on_market_trade(norm_trade)
    assert trade_res.events is not None
    assert len(trade_res.events.fills) == 1
    assert len(trade_res.events.opened_positions) == 1

    snap_after = sink.get_capacity_snapshot()
    assert snap_after["open_positions"] == 1
    assert snap_after["pending_orders"] == 0
    assert snap_after["effective_positions"] == 1
    assert snap_after["available_capacity"] == 1


def test_6_expiry_releases_capacity():
    """6. Expiração de ordem libera capacidade para novas ordens."""
    executor = PaperExecutor(config=ExecutorConfig(max_open_positions=1))
    sink = ExecutionSink(symbol="BTCUSDT", executor=executor)

    order1 = _build_order(
        symbol="BTCUSDT",
        order_id="ord_1",
        decision_id="dec_1",
        available_at=1100,
        expires_at=5000,
    )
    sink.submit_order(order1)
    assert sink.get_capacity_snapshot()["effective_positions"] == 1

    # Tick com T=5100 > expires_at=5000
    norm_trade = {
        "p": 50000.0,
        "q": 0.5,
        "T": 5100,
        "T_raw": 5100,
        "m": False,
        "source": "aggTrade",
        "trade_id": 99992,
    }
    trade_res = sink.on_market_trade(norm_trade)
    assert trade_res.events is not None
    assert len(trade_res.events.rejections) == 1
    assert trade_res.events.rejections[0].reason == "EXPIRED_NO_MARKET_DATA"

    snap_after_expiry = sink.get_capacity_snapshot()
    assert snap_after_expiry["pending_orders"] == 0
    assert snap_after_expiry["effective_positions"] == 0
    assert snap_after_expiry["available_capacity"] == 1

    # Novo slot agora está disponível
    order2 = _build_order(symbol="ETHUSDT", order_id="ord_2", decision_id="dec_2")
    res2 = sink.submit_order(order2)
    assert res2.status == SubmitStatus.ACCEPTED
    assert sink.get_capacity_snapshot()["effective_positions"] == 1


def test_7_submit_rejected_does_not_consume_capacity():
    """7. Uma ordem rejeitada na validação estrutural não consome capacidade."""
    executor = PaperExecutor(config=ExecutorConfig(max_open_positions=1))
    sink = ExecutionSink(symbol="BTCUSDT", executor=executor)

    # Ordem com side inválido (NEUTRAL)
    bad_order = _build_order(symbol="BTCUSDT", order_id="ord_bad", side="NEUTRAL")  # type: ignore[arg-type]
    res = sink.submit_order(bad_order)
    assert res.status == SubmitStatus.REJECTED
    assert res.rejection is not None
    assert res.rejection.reason == "NO_DIRECTION"

    snap = sink.get_capacity_snapshot()
    assert snap["effective_positions"] == 0
    assert snap["available_capacity"] == 1


def test_8_duplicate_submit_does_not_alter_capacity():
    """8. Submissão duplicada com mesmo order_id preserva DUPLICATE_DECISION e não duplica capacidade."""
    executor = PaperExecutor(config=ExecutorConfig(max_open_positions=2))
    sink = ExecutionSink(symbol="BTCUSDT", executor=executor)

    order1 = _build_order(symbol="BTCUSDT", order_id="ord_same", decision_id="dec_1")
    res1 = sink.submit_order(order1)
    assert res1.status == SubmitStatus.ACCEPTED
    assert sink.get_capacity_snapshot()["effective_positions"] == 1

    # Tenta submeter exatamente a mesma ordem novamente
    res_dup = sink.submit_order(order1)
    assert res_dup.status == SubmitStatus.REJECTED
    assert res_dup.rejection is not None
    assert res_dup.rejection.reason == "DUPLICATE_DECISION"

    assert sink.get_capacity_snapshot()["effective_positions"] == 1
    assert sink.get_capacity_snapshot()["available_capacity"] == 1


def test_9_precedence_position_open_over_capacity_limit():
    """9. Conflito do mesmo símbolo tem precedência histórica (POSITION_OPEN) sobre POSITION_LIMIT_EXCEEDED."""
    executor = PaperExecutor(config=ExecutorConfig(max_open_positions=1))
    sink = ExecutionSink(symbol="BTCUSDT", executor=executor)

    order1 = _build_order(symbol="BTCUSDT", order_id="ord_1", decision_id="dec_1")
    res1 = sink.submit_order(order1)
    assert res1.status == SubmitStatus.ACCEPTED

    # Ordem para o MESMO símbolo (BTCUSDT) quando o limite (1) já está atingido
    order_same_sym = _build_order(symbol="BTCUSDT", order_id="ord_2", decision_id="dec_2")
    res_same_sym = sink.submit_order(order_same_sym)
    assert res_same_sym.status == SubmitStatus.REJECTED
    assert res_same_sym.rejection is not None
    # Precedência: POSITION_OPEN é validado antes de POSITION_LIMIT_EXCEEDED!
    assert res_same_sym.rejection.reason == "POSITION_OPEN"


def test_10_precedence_duplicate_decision_over_capacity_limit():
    """10. DUPLICATE_DECISION tem precedência antes da verificação de capacidade."""
    executor = PaperExecutor(config=ExecutorConfig(max_open_positions=1))
    sink = ExecutionSink(symbol="BTCUSDT", executor=executor)

    order1 = _build_order(symbol="BTCUSDT", order_id="ord_1", decision_id="dec_1")
    sink.submit_order(order1)

    # Submete mesma ordem novamente quando limite já está lotado
    res_dup = sink.submit_order(order1)
    assert res_dup.status == SubmitStatus.REJECTED
    assert res_dup.rejection is not None
    assert res_dup.rejection.reason == "DUPLICATE_DECISION"


def test_11_snapshot_consistency_under_high_concurrency():
    """11. Snapshot de capacidade permanece rigorosamente consistente sob concorrência de threads."""
    executor = PaperExecutor(config=ExecutorConfig(max_open_positions=5))
    sink = ExecutionSink(symbol="BTCUSDT", executor=executor)

    accepted_count = 0
    rejected_count = 0
    lock = threading.Lock()

    def _worker(idx: int):
        nonlocal accepted_count, rejected_count
        order = _build_order(
            symbol=f"SYM_{idx}USDT",
            order_id=f"ord_{idx}",
            decision_id=f"dec_{idx}",
        )
        res = sink.submit_order(order)
        snap = sink.get_capacity_snapshot()
        # Invariante universal em todo momento
        assert snap["effective_positions"] <= 5
        with lock:
            if res.status == SubmitStatus.ACCEPTED:
                accepted_count += 1
            else:
                rejected_count += 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        futs = [pool.submit(_worker, i) for i in range(20)]
        concurrent.futures.wait(futs)

    assert accepted_count == 5
    assert rejected_count == 15
    final_snap = sink.get_capacity_snapshot()
    assert final_snap["effective_positions"] == 5
    assert final_snap["available_capacity"] == 0


def test_12_legacy_unlimited_behavior_preserved():
    """12. max_open_positions=None preserva comportamento anterior irrestrito."""
    executor = PaperExecutor(config=ExecutorConfig(max_open_positions=None))
    sink = ExecutionSink(symbol="BTCUSDT", executor=executor)

    for i in range(25):
        order = _build_order(
            symbol=f"SYM_{i}USDT",
            order_id=f"ord_{i}",
            decision_id=f"dec_{i}",
        )
        res = sink.submit_order(order)
        assert res.status == SubmitStatus.ACCEPTED

    snap = sink.get_capacity_snapshot()
    assert snap["effective_positions"] == 25
    assert snap["max_open_positions"] is None
    assert snap["available_capacity"] is None


def test_13_shadow_runtime_propagates_risk_config():
    """13. ShadowPaperRuntime propaga exatamente RiskConfig.max_open_positions para o executor."""
    custom_risk_config = RiskConfig(max_open_positions=4)
    rm = RiskManager(custom_risk_config)

    runtime_config = ShadowPaperConfig(
        enabled=True,
        cohort_id="CH_PROPAGATION_TEST",
        symbol="BTCUSDT",
        strategy_version="v1.0",
        provider="fixed_long",
        order_ttl_ms=5000,
    )

    runtime = ShadowPaperRuntime(
        config=runtime_config,
        risk_manager=rm,
    )

    # Identidade tripla comprovada
    assert rm.config.max_open_positions == 4
    assert runtime.execution_sink.executor.config.max_open_positions == 4
    snap = runtime.execution_sink.get_capacity_snapshot()
    assert snap["max_open_positions"] == 4
    assert snap["available_capacity"] == 4

    status = runtime.get_status()
    assert status["risk_position_limit_active"] is True
    assert status["risk_daily_loss_active"] is False


def test_14_lock_ordering_and_no_deadlock():
    """14. Ordem estrita de locks e ausência de bloqueios em cascata."""
    custom_risk_config = RiskConfig(max_open_positions=2)
    rm = RiskManager(custom_risk_config)
    runtime_config = ShadowPaperConfig(
        enabled=True,
        cohort_id="CH_LOCK_TEST",
        symbol="BTCUSDT",
        strategy_version="v1.0",
        provider="fixed_long",
    )
    runtime = ShadowPaperRuntime(config=runtime_config, risk_manager=rm)

    # get_capacity_snapshot não toca em Runtime._lock
    snap = runtime.execution_sink.get_capacity_snapshot()
    assert snap["max_open_positions"] == 2

    # Verifica que chamar get_capacity_snapshot enquanto segura Runtime._lock não gera deadlock
    with runtime._lock:
        snap_inside = runtime.execution_sink.get_capacity_snapshot()
        assert snap_inside["max_open_positions"] == 2


def test_15_invalid_config_raises_value_error_for_zero_and_negative():
    """15. max_open_positions deve ser estritamente positivo (>0) ou None. Zero e negativos falham."""
    with pytest.raises(ValueError, match="max_open_positions must be a positive integer"):
        PaperExecutor(config=ExecutorConfig(max_open_positions=0))

    with pytest.raises(ValueError, match="max_open_positions must be a positive integer"):
        PaperExecutor(config=ExecutorConfig(max_open_positions=-1))

    with pytest.raises(ValueError, match="max_open_positions must be a positive integer"):
        PaperExecutor(config=ExecutorConfig(max_open_positions=True))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="max_open_positions must be a positive integer"):
        PaperExecutor(config=ExecutorConfig(max_open_positions="10"))  # type: ignore[arg-type]


def test_16_runtime_with_injected_sink_correct_limit_active():
    """16. Runtime com sink injetado que possui limite coincidente define active=True."""
    custom_risk = RiskConfig(max_open_positions=3)
    rm = RiskManager(custom_risk)

    executor = PaperExecutor(config=ExecutorConfig(max_open_positions=3))
    sink = ExecutionSink(symbol="BTCUSDT", executor=executor)

    cfg = ShadowPaperConfig(
        enabled=True,
        cohort_id="CH_INJECTED_OK",
        symbol="BTCUSDT",
        strategy_version="v1.0",
        provider="fixed_long",
    )
    runtime = ShadowPaperRuntime(
        config=cfg,
        risk_manager=rm,
        execution_sink=sink,
    )

    assert runtime.is_risk_position_limit_active is True
    assert runtime.get_status()["risk_position_limit_active"] is True


def test_17_runtime_with_injected_sink_none_limit_inactive():
    """17. Runtime com sink injetado e limite None: flag NÃO pode ser True."""
    custom_risk = RiskConfig(max_open_positions=5)
    rm = RiskManager(custom_risk)

    executor = PaperExecutor(config=ExecutorConfig(max_open_positions=None))
    sink = ExecutionSink(symbol="BTCUSDT", executor=executor)

    cfg = ShadowPaperConfig(
        enabled=True,
        cohort_id="CH_INJECTED_NONE",
        symbol="BTCUSDT",
        strategy_version="v1.0",
        provider="fixed_long",
    )
    runtime = ShadowPaperRuntime(
        config=cfg,
        risk_manager=rm,
        execution_sink=sink,
    )

    assert runtime.is_risk_position_limit_active is False
    assert runtime.get_status()["risk_position_limit_active"] is False


def test_18_runtime_with_injected_sink_mismatch_fails_fast():
    """18. Runtime com sink injetado e limite divergente de RiskConfig falha rápido (fail-fast)."""
    custom_risk = RiskConfig(max_open_positions=10)
    rm = RiskManager(custom_risk)

    # Sink configurado com 5 enquanto RiskConfig possui 10
    executor = PaperExecutor(config=ExecutorConfig(max_open_positions=5))
    sink = ExecutionSink(symbol="BTCUSDT", executor=executor)

    cfg = ShadowPaperConfig(
        enabled=True,
        cohort_id="CH_INJECTED_MISMATCH",
        symbol="BTCUSDT",
        strategy_version="v1.0",
        provider="fixed_long",
    )

    with pytest.raises(ValueError, match="Capacity limit mismatch: RiskConfig has 10 but injected ExecutionSink executor has 5"):
        ShadowPaperRuntime(
            config=cfg,
            risk_manager=rm,
            execution_sink=sink,
        )


def test_19_audit_no_insecure_direct_callers_in_runtime():
    """19. Prova por introspecção que ShadowPaperRuntime delega exclusivamente via ExecutionSink.submit_order."""
    src = inspect.getsource(shadow_runtime_module.ShadowPaperRuntime.on_signal)
    assert "execution_sink.submit_order" in src
    assert "executor.submit_order" not in src
