"""
Contrato boundary-only fail-closed do OutcomeTracker.

Um horizonte (5m/15m/30m/60m) só é preenchido quando o epoch da avaliação
está dentro do boundary temporal daquele horizonte:

    target_ms = signal_epoch_ms + horizon_min * 60_000
    0 <= current_epoch_ms - target_ms <= OUTCOME_BOUNDARY_TOLERANCE_MS

Boundary perdido => NULL permanente nesta versão (fail-closed).
Todos os testes usam SQLite temporário (tmp_path); produção nunca é aberta.
"""

import sqlite3

import pytest

from trading.outcome_tracker import (
    OUTCOME_BOUNDARY_TOLERANCE_MS,
    OutcomeTracker,
)

# Epochs reais da execução auditada (2026-08-23), todos % 60_000 == 0
T0 = 1787444520000              # 00:22:00 (boundary do sinal)
ENTRY = 77104.29


def make_tracker(tmp_path):
    return OutcomeTracker(db_path=str(tmp_path / "outcomes_test.db"))


def register(tmp_path_tracker, signal_epoch_ms=T0, entry_price=ENTRY,
             event_type="Absorção", battle="Absorção de Compra"):
    tracker = tmp_path_tracker
    tracker.register_signal({
        "epoch_ms": signal_epoch_ms,
        "tipo_evento": event_type,
        "resultado_da_batalha": battle,
        "preco_fechamento": entry_price,
        "symbol": "BTCUSDT",
    })
    return tracker


def fetch_row(db_path):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    row = conn.execute(
        """SELECT outcome_5m_pct, outcome_direction_5m,
                  outcome_15m_pct, outcome_30m_pct, outcome_60m_pct,
                  evaluated_at
           FROM signal_outcomes WHERE id = 1"""
    ).fetchone()
    conn.close()
    keys = ("pct_5m", "dir_5m", "pct_15m", "pct_30m", "pct_60m", "evaluated_at")
    return dict(zip(keys, row))


# ---------------------------------------------------------------------------
# A) boundary exato preenche
# ---------------------------------------------------------------------------

def test_a_exact_boundary_fills_5m(tmp_path):
    tracker = register(make_tracker(tmp_path))
    eval_ms = T0 + 300_000                      # 00:27:00, drift = 0
    tracker.evaluate_pending_outcomes(77193.70, eval_ms)

    row = fetch_row(tracker.db_path)
    expected = round((77193.70 - ENTRY) / ENTRY * 100, 4)
    assert row["pct_5m"] == pytest.approx(expected)
    assert row["dir_5m"] == "UP"
    assert row["evaluated_at"] == eval_ms


# ---------------------------------------------------------------------------
# B) um instante antes do boundary não grava
# ---------------------------------------------------------------------------

def test_b_before_boundary_stays_null(tmp_path):
    tracker = register(make_tracker(tmp_path))
    tracker.evaluate_pending_outcomes(77193.70, T0 + 299_999)   # 00:26:59

    row = fetch_row(tracker.db_path)
    assert row["pct_5m"] is None
    assert row["dir_5m"] is None


# ---------------------------------------------------------------------------
# C) dentro da tolerância (+500ms) grava
# ---------------------------------------------------------------------------

def test_c_within_tolerance_fills(tmp_path):
    assert OUTCOME_BOUNDARY_TOLERANCE_MS == 1000
    tracker = register(make_tracker(tmp_path))
    tracker.evaluate_pending_outcomes(77193.70, T0 + 300_500)   # 00:27:00.500

    row = fetch_row(tracker.db_path)
    assert row["pct_5m"] is not None
    assert row["evaluated_at"] == T0 + 300_500


# ---------------------------------------------------------------------------
# D) acima da tolerância (+1001ms) não grava
# ---------------------------------------------------------------------------

def test_d_above_tolerance_rejected(tmp_path):
    tracker = register(make_tracker(tmp_path))
    tracker.evaluate_pending_outcomes(77244.23, T0 + 300_000 + 1_001)

    row = fetch_row(tracker.db_path)
    assert row["pct_5m"] is None


# ---------------------------------------------------------------------------
# E) candle seguinte (+60s) NUNCA preenche
# ---------------------------------------------------------------------------

def test_e_next_candle_never_fills(tmp_path):
    tracker = register(make_tracker(tmp_path))
    tracker.evaluate_pending_outcomes(77244.23, T0 + 360_000)   # 00:28:00

    row = fetch_row(tracker.db_path)
    assert row["pct_5m"] is None
    assert row["pct_5m"] != 0.1815                              # regressão do bug


# ---------------------------------------------------------------------------
# F/G/H) horizontes independentes; perdido permanece NULL
# ---------------------------------------------------------------------------

def test_f_15m_fills_and_5m_stays_null(tmp_path):
    tracker = register(make_tracker(tmp_path))
    tracker.evaluate_pending_outcomes(77350.00, T0 + 900_000)   # 00:37:00

    row = fetch_row(tracker.db_path)
    expected = round((77350.00 - ENTRY) / ENTRY * 100, 4)
    assert row["pct_15m"] == pytest.approx(expected)
    assert row["pct_5m"] is None                                # boundary perdido
    assert row["pct_30m"] is None
    assert row["pct_60m"] is None


def test_g_30m_fills(tmp_path):
    tracker = register(make_tracker(tmp_path))
    tracker.evaluate_pending_outcomes(77400.00, T0 + 1_800_000)  # 00:52:00

    row = fetch_row(tracker.db_path)
    expected = round((77400.00 - ENTRY) / ENTRY * 100, 4)
    assert row["pct_30m"] == pytest.approx(expected)
    assert row["pct_5m"] is None


def test_h_60m_fills(tmp_path):
    tracker = register(make_tracker(tmp_path))
    tracker.evaluate_pending_outcomes(77500.00, T0 + 3_600_000)  # 01:22:00

    row = fetch_row(tracker.db_path)
    expected = round((77500.00 - ENTRY) / ENTRY * 100, 4)
    assert row["pct_60m"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# I) idempotência: já preenchido não sobrescreve
# ---------------------------------------------------------------------------

def test_i_no_overwrite_after_filled(tmp_path):
    tracker = register(make_tracker(tmp_path))
    first_eval = T0 + 300_000
    tracker.evaluate_pending_outcomes(77193.70, first_eval)
    before = fetch_row(tracker.db_path)

    tracker.evaluate_pending_outcomes(77999.99, first_eval)     # mesma janela
    tracker.evaluate_pending_outcomes(77999.99, T0 + 300_700)   # ainda na tolerância

    after = fetch_row(tracker.db_path)
    assert after["pct_5m"] == before["pct_5m"]
    assert after["dir_5m"] == before["dir_5m"]
    assert after["evaluated_at"] == before["evaluated_at"] == first_eval


# ---------------------------------------------------------------------------
# J) pré-filtro SQL inclui idade exatamente +5m
# ---------------------------------------------------------------------------

def test_j_prefilter_includes_exact_five_minutes(tmp_path):
    tracker = register(make_tracker(tmp_path))
    # Se o pré-filtro continuasse estrito (<), este caso falharia.
    tracker.evaluate_pending_outcomes(77193.70, T0 + 300_000)
    assert fetch_row(tracker.db_path)["pct_5m"] is not None


# ---------------------------------------------------------------------------
# K) epoch não alinhado: warning e nenhuma normalização silenciosa
# ---------------------------------------------------------------------------

def test_k_unaligned_epoch_warns_and_never_fills(tmp_path, caplog):
    import logging

    unaligned = T0 + 12345
    tracker = make_tracker(tmp_path)
    with caplog.at_level(logging.WARNING, logger="OutcomeTracker"):
        tracker.register_signal({
            "epoch_ms": unaligned,
            "tipo_evento": "Exaustão",
            "resultado_da_batalha": "Exaustão de Venda",
            "preco_fechamento": ENTRY,
            "symbol": "BTCUSDT",
        })

    assert any("nao esta alinhado" in r.message for r in caplog.records)

    # Nenhum boundary alinhado produz drift dentro da tolerância
    for offset in range(1, 8):
        tracker.evaluate_pending_outcomes(77500.00, T0 + offset * 300_000)

    conn = sqlite3.connect(f"file:{tracker.db_path}?mode=ro", uri=True)
    stored_epoch = conn.execute(
        "SELECT signal_epoch_ms FROM signal_outcomes").fetchone()[0]
    pcts = conn.execute(
        "SELECT outcome_5m_pct, outcome_15m_pct FROM signal_outcomes"
    ).fetchone()
    conn.close()

    assert stored_epoch == unaligned            # sem normalização silenciosa
    assert pcts[0] is None and pcts[1] is None


# ---------------------------------------------------------------------------
# L) NULL não entra no denominador da probabilidade histórica
# ---------------------------------------------------------------------------

def test_l_probability_denominator_excludes_null(tmp_path):
    tracker = make_tracker(tmp_path)

    register(tracker, signal_epoch_ms=T0)
    tracker.evaluate_pending_outcomes(77193.70, T0 + 300_000)   # avaliado UP

    register(tracker, signal_epoch_ms=T0 + 60_000)              # nunca avaliado

    prob = tracker.get_historical_probability(
        event_type="Absorção",
        battle_result="Absorção de Compra",
        window="5m",
        min_samples=1,
    )

    assert prob["status"] == "ok"
    assert prob["samples"] == 1                                 # NULL fora do denominador
    assert prob["details"].get("DOWN") is None                  # NULL não vira LOSS
    assert prob["prob_up"] == 1.0


# ---------------------------------------------------------------------------
# Tolerância exata na borda do intervalo
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("offset_ms,expect_fill", [
    (300_999, True),     # +999ms  => aceita
    (301_000, True),     # +1000ms => aceita (borda inclusiva)
    (301_001, False),    # +1001ms => rejeita
    (360_000, False),    # +60000ms (candle seguinte) => rejeita
])
def test_exact_tolerance_edges(tmp_path, offset_ms, expect_fill):
    tracker = register(make_tracker(tmp_path))
    tracker.evaluate_pending_outcomes(77193.70, T0 + offset_ms)
    row = fetch_row(tracker.db_path)
    if expect_fill:
        assert row["pct_5m"] is not None
        assert row["pct_5m"] == pytest.approx(
            round((77193.70 - ENTRY) / ENTRY * 100, 4))
    else:
        assert row["pct_5m"] is None


# ---------------------------------------------------------------------------
# Cada boundary preenche SOMENTE o próprio horizonte
# ---------------------------------------------------------------------------

def test_each_boundary_fills_only_its_own_horizon(tmp_path):
    tracker = register(make_tracker(tmp_path))

    stamps = [
        (300_000,   "pct_5m"),    # +5m
        (900_000,   "pct_15m"),   # +15m
        (1_800_000, "pct_30m"),   # +30m
        (3_600_000, "pct_60m"),   # +60m
    ]
    cols = [col for _, col in stamps]
    seen = []
    for offset, col in stamps:
        before = fetch_row(tracker.db_path)
        tracker.evaluate_pending_outcomes(77200.00 + offset, T0 + offset)
        after = fetch_row(tracker.db_path)

        for c in cols:
            if c == col:
                assert after[c] is not None, f"{col} deveria ter sido preenchido"
                seen.append(c)
            elif c in seen:
                assert after[c] == before[c], f"{c} não pode ser sobrescrito"
            else:
                # boundary posterior NÃO preenche horizontes anteriores perdidos
                assert after[c] is None, (
                    f"{c} foi preenchido por boundary de {col} (look-back proibido)")


# ---------------------------------------------------------------------------
# evaluated_at = última avaliação que preencheu algum horizonte
# ---------------------------------------------------------------------------

def test_evaluated_at_tracks_last_filling_evaluation(tmp_path):
    tracker = register(make_tracker(tmp_path))

    tracker.evaluate_pending_outcomes(77193.70, T0 + 300_000)     # 5m
    row = fetch_row(tracker.db_path)
    assert row["evaluated_at"] == T0 + 300_000

    pct_5m_before = row["pct_5m"]
    tracker.evaluate_pending_outcomes(77350.00, T0 + 900_000)     # 15m
    row = fetch_row(tracker.db_path)
    assert row["evaluated_at"] == T0 + 900_000                    # avançou
    assert row["pct_5m"] == pct_5m_before                         # 5m intacto


# ---------------------------------------------------------------------------
# Anti-churn: sinal antigo com tudo NULL sai da varredura sem efeito colateral
# ---------------------------------------------------------------------------

def test_antichurn_old_missed_signal_not_rescanned(tmp_path):
    tracker = make_tracker(tmp_path)

    two_days_old = T0 - 2 * 24 * 3_600_000
    tracker.register_signal({
        "epoch_ms": two_days_old,
        "tipo_evento": "Absorção",
        "resultado_da_batalha": "Absorção de Compra",
        "preco_fechamento": ENTRY,
        "symbol": "BTCUSDT",
    })

    # Chamada atual: stamp válido para o sinal RECENTE deve continuar funcionando
    register(tracker, signal_epoch_ms=T0)
    tracker.evaluate_pending_outcomes(77193.70, T0 + 300_000)

    conn = sqlite3.connect(f"file:{tracker.db_path}?mode=ro", uri=True)
    rows = {
        ts: (p5,)
        for ts, p5 in conn.execute(
            "SELECT signal_epoch_ms, outcome_5m_pct FROM signal_outcomes")
    }
    conn.close()

    assert rows[T0][0] is not None, "sinal válido não pode ser afetado pelo anti-churn"
    assert rows[two_days_old][0] is None, "sinal antigo missed permanece NULL"
    # ...e uma chamada posterior qualquer não o ressuscita:
    tracker.evaluate_pending_outcomes(77500.00, T0 + 900_000)
    conn = sqlite3.connect(f"file:{tracker.db_path}?mode=ro", uri=True)
    old = conn.execute(
        "SELECT outcome_5m_pct FROM signal_outcomes WHERE signal_epoch_ms=?",
        (two_days_old,)).fetchone()[0]
    conn.close()
    assert old is None


# ---------------------------------------------------------------------------
# Regressão do caso real auditado (P1)
# ---------------------------------------------------------------------------

class TestRealCaseRegression:
    def test_target_close_wins_over_late_close(self, tmp_path):
        tracker = register(make_tracker(tmp_path))

        # boundary correto (+5m): close 77193.70 => ~ +0.116%
        tracker.evaluate_pending_outcomes(77193.70, T0 + 300_000)
        row = fetch_row(tracker.db_path)
        expected = round((77193.70 - ENTRY) / ENTRY * 100, 4)
        assert row["pct_5m"] == pytest.approx(expected)
        first_eval = row["evaluated_at"]

        # chamada tardia (+6m, close 77244.23) NÃO pode virar +0.1815%
        tracker.evaluate_pending_outcomes(77244.23, T0 + 360_000)
        row = fetch_row(tracker.db_path)
        assert row["pct_5m"] == pytest.approx(expected)
        assert row["evaluated_at"] == first_eval

    def test_if_5m_missed_then_6m_cannot_fill(self, tmp_path):
        tracker = register(make_tracker(tmp_path))

        # só existe a chamada tardia de +6m
        tracker.evaluate_pending_outcomes(77244.23, T0 + 360_000)
        row = fetch_row(tracker.db_path)
        assert row["pct_5m"] is None
