# tests/unit/test_outcome_tracker_direction_aware.py
"""
Testes unitários para o cálculo direction-aware de probabilidades históricas
no OutcomeTracker.

Cobre:
1. Dataset LONG simétrico (6 UP, 3 DOWN, 1 FLAT) => win_rate=60.0%, directional_win_rate=66.7%
2. Dataset SHORT simétrico (3 UP, 6 DOWN, 1 FLAT) => win_rate=60.0%, directional_win_rate=66.7%
3. Dataset UNKNOWN / NEUTRAL => win_rate=None, directional_win_rate=None
4. Preservação integral de prob_up, prob_down, prob_flat, avg_return_pct, avg_win_pct, avg_loss_pct.
"""

import pytest
import sqlite3
from pathlib import Path
from trading.outcome_tracker import OutcomeTracker


@pytest.fixture
def temp_tracker(tmp_path):
    db_file = tmp_path / "test_outcomes.db"
    tracker = OutcomeTracker(db_path=str(db_file))
    return tracker, str(db_file)


def _seed_dataset(db_file: str, event_type: str, battle_result: str, outcomes: list):
    """
    outcomes: lista de tuplas (direction, pct)
    """
    with sqlite3.connect(db_file) as conn:
        for idx, (direction, pct) in enumerate(outcomes):
            epoch_ms = 1700000000000 + idx * 60_000
            conn.execute(
                """INSERT INTO signal_outcomes
                (signal_epoch_ms, event_type, battle_result, entry_price, symbol,
                 outcome_15m_pct, outcome_direction_15m)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (epoch_ms, event_type, battle_result, 100.0, "BTCUSDT", pct, direction),
            )


# ==============================================================================
# 1. TESTE SIMÉTRICO: LONG DATASET (6 UP, 3 DOWN, 1 FLAT)
# ==============================================================================

def test_outcome_tracker_long_symmetric(temp_tracker):
    tracker, db_file = temp_tracker

    # 6 UP (+1.0%), 3 DOWN (-1.0%), 1 FLAT (0.0%)
    outcomes = (
        [("UP", 1.0)] * 6
        + [("DOWN", -1.0)] * 3
        + [("FLAT", 0.0)] * 1
    )
    _seed_dataset(db_file, "Absorção", "Absorção de Venda", outcomes)

    res = tracker.get_historical_probability(
        event_type="Absorção",
        battle_result="Absorção de Venda",
        window="15m",
        min_samples=10,
    )

    assert res["status"] == "ok"
    assert res["samples"] == 10
    assert res["signal_side"] == "LONG"

    # Probabilidades de mercado brutas preservadas
    assert res["prob_up"] == 0.6
    assert res["prob_down"] == 0.3
    assert res["prob_flat"] == 0.1

    # Métricas direction-aware
    assert res["prob_win"] == 0.6
    assert res["prob_loss"] == 0.3
    assert res["win_rate"] == 60.0  # 6 / (6+3+1) * 100
    assert res["directional_win_rate"] == 66.7  # 6 / (6+3) * 100

    # Médias preservadas
    assert res["avg_win_pct"] == 1.0
    assert res["avg_loss_pct"] == -1.0


# ==============================================================================
# 2. TESTE SIMÉTRICO: SHORT DATASET (3 UP, 6 DOWN, 1 FLAT)
# ==============================================================================

def test_outcome_tracker_short_symmetric(temp_tracker):
    tracker, db_file = temp_tracker

    # 3 UP (+1.0%), 6 DOWN (-1.0%), 1 FLAT (0.0%)
    outcomes = (
        [("UP", 1.0)] * 3
        + [("DOWN", -1.0)] * 6
        + [("FLAT", 0.0)] * 1
    )
    _seed_dataset(db_file, "Absorção", "Absorção de Compra", outcomes)

    res = tracker.get_historical_probability(
        event_type="Absorção",
        battle_result="Absorção de Compra",
        window="15m",
        min_samples=10,
    )

    assert res["status"] == "ok"
    assert res["samples"] == 10
    assert res["signal_side"] == "SHORT"

    # Probabilidades de mercado brutas preservadas (descrevem movimento do ativo)
    assert res["prob_up"] == 0.3
    assert res["prob_down"] == 0.6
    assert res["prob_flat"] == 0.1

    # Métricas direction-aware (comprovando simetria perfeita: 60% win_rate)
    assert res["prob_win"] == 0.6  # 6 DOWN é WIN para SHORT
    assert res["prob_loss"] == 0.3  # 3 UP é LOSS para SHORT
    assert res["win_rate"] == 60.0  # 6 / (6+3+1) * 100
    assert res["directional_win_rate"] == 66.7  # 6 / (6+3) * 100

    # Médias brutas preservadas intactas
    assert res["avg_win_pct"] == 1.0
    assert res["avg_loss_pct"] == -1.0


# ==============================================================================
# 3. TESTE DE EVENTOS DESCONHECIDOS / NEUTROS (NÃO ASSUME LONG)
# ==============================================================================

def test_outcome_tracker_unknown_side(temp_tracker):
    tracker, db_file = temp_tracker

    outcomes = [("UP", 1.0)] * 10
    _seed_dataset(db_file, "TipoCustomizado", "RótuloDesconhecido", outcomes)

    res = tracker.get_historical_probability(
        event_type="TipoCustomizado",
        battle_result="RótuloDesconhecido",
        window="15m",
        min_samples=10,
    )

    assert res["status"] == "unknown_signal_side"
    assert res["signal_side"] == "UNKNOWN"
    assert res["prob_up"] == 1.0
    assert res["prob_down"] == 0.0
    # Nunca inventa win_rate
    assert res["prob_win"] is None
    assert res["prob_loss"] is None
    assert res["win_rate"] is None
    assert res["directional_win_rate"] is None


def test_outcome_tracker_neutral_side(temp_tracker):
    tracker, db_file = temp_tracker

    outcomes = [("UP", 1.0)] * 10
    _seed_dataset(db_file, "OrderBook", "Equilíbrio", outcomes)

    res = tracker.get_historical_probability(
        event_type="OrderBook",
        battle_result="Equilíbrio",
        window="15m",
        min_samples=10,
    )

    assert res["status"] == "neutral_signal"
    assert res["signal_side"] == "NEUTRAL"
    assert res["prob_up"] == 1.0
    assert res["win_rate"] is None
    assert res["directional_win_rate"] is None


# ==============================================================================
# 4. TESTE DE INTERPRETAÇÃO DO REGISTRO REAL (READ-ONLY)
# ==============================================================================

def test_real_database_record_interpretation():
    """
    Demonstra a semântica correta no caso real presente em dados/trading_bot.db:
    battle_result: 'Absorção de Compra'
    outcome_direction_5m: 'UP'

    Antes da correção: tratado erroneamente como WIN porque UP era tratado como vitória universal.
    Após a correção: infer_signal_side infere SHORT e classify_outcome resulta em LOSS.
    """
    from common.signal_direction import infer_signal_side, classify_outcome

    event_type = "Absorção"
    battle_result = "Absorção de Compra"
    outcome_direction = "UP"

    side = infer_signal_side(event_type=event_type, battle_result=battle_result)
    assert side == "SHORT"

    result = classify_outcome(signal_side=side, outcome_direction=outcome_direction)
    assert result == "LOSS"
