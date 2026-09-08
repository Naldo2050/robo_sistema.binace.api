"""Testes para Crypto COT — contrato real de produção (stateless).

Contrato: CryptoCOT().analyze(positioning_data, funding_rate=None, symbol)
  -> CryptoCOTAnalysis (regime + reasons + campos, fail-closed UNKNOWN).

Removidos explicitamente (comportamento NÃO existe mais; sem adaptação à força):
- data_points/latest/add_data(timestamp, funding, oi, ratio, ...) — sem acumulador;
- analyze() sem args e result.signals/confidence/source — sem lista de sinais
  (context-only devolve regime+reasons; ver to_legacy_analysis_result);
- CryptoCOT(oi_change_threshold_pct=...) — sem esse threshold (constantes de
  classe OI_EXPANSION_1H/4H); reset() — stateless, nada a resetar.
Regimes via snapshot já cobertos em test_binance_positioning_p1_1.py;
aqui: superfície dict + limites numéricos + fail-closed + serialização.
"""

import pytest

from fetchers.binance_positioning_fetcher import BinancePositioningSnapshot
from institutional.crypto_cot import CryptoCOT, PositioningRegime


def _dict(**over):
    base = {
        "global_account_ratio": 1.10,
        "top_account_ratio": 1.15,
        "top_position_ratio": 1.20,
        "global_long_account_pct": 52.0,
        "global_short_account_pct": 48.0,
        "open_interest": 100000.0,
        "open_interest_usd": 6500000000.0,
        "oi_delta_1h": 0.01,
        "oi_delta_4h": 0.02,
        "funding_rate": 0.0001,
        "is_available": True,
        "is_stale": False,
    }
    base.update(over)
    return base


def test_stateless_no_accumulation():
    """Sem add_data/latest/data_points/reset: analyzes independentes."""
    cot = CryptoCOT()
    assert not hasattr(cot, "data_points")
    assert not hasattr(cot, "latest")
    assert not hasattr(cot, "add_data")
    assert not hasattr(cot, "reset")
    r1 = cot.analyze(_dict(global_account_ratio=2.5, top_position_ratio=2.4))
    r2 = cot.analyze(_dict())
    assert r1.regime == PositioningRegime.CROWDED_LONG
    assert r2.regime == PositioningRegime.NEUTRAL


def test_valid_dict_full():
    cot = CryptoCOT()
    res = cot.analyze(_dict(), symbol="BTCUSDT")
    assert res.regime == PositioningRegime.NEUTRAL
    assert res.is_available is True
    assert res.is_stale is False
    assert res.global_account_ratio == 1.10
    assert res.open_interest == 100000.0
    assert res.funding_rate == 0.0001
    assert res.reasons


def test_none_is_unknown_fail_closed():
    res = CryptoCOT().analyze(None)
    assert res.regime == PositioningRegime.UNKNOWN
    assert res.is_available is False


def test_invalid_type_is_unknown_fail_closed():
    for bad in (42, "posicionado", [1.2], (1.2,)):
        res = CryptoCOT().analyze(bad)
        assert res.regime == PositioningRegime.UNKNOWN, bad
        assert res.is_available is False


def test_stale_is_unknown_fail_closed():
    res = CryptoCOT().analyze(_dict(is_stale=True, age_seconds=9999))
    assert res.regime == PositioningRegime.UNKNOWN
    assert res.is_stale is True
    assert res.is_available is False


def test_unavailable_is_unknown_fail_closed():
    res = CryptoCOT().analyze(_dict(is_available=False))
    assert res.regime == PositioningRegime.UNKNOWN
    assert res.is_available is False


def test_partial_missing_ratios():
    res = CryptoCOT().analyze(_dict(global_account_ratio=None))
    assert res.regime == PositioningRegime.PARTIAL
    assert res.is_available is True
    res2 = CryptoCOT().analyze(_dict(top_position_ratio=None))
    assert res2.regime == PositioningRegime.PARTIAL


def test_numeric_threshold_boundaries():
    cot = CryptoCOT()
    # crowding inclusivo: >= 2.0 / <= 0.5
    assert cot.analyze(_dict(global_account_ratio=2.0,
                             top_position_ratio=2.0)).regime == PositioningRegime.CROWDED_LONG
    assert cot.analyze(_dict(global_account_ratio=0.5,
                             top_position_ratio=0.5)).regime == PositioningRegime.CROWDED_SHORT
    # squeeze exige funding ALÉM de 0.0003 E crowding (limite exato não dispara)
    assert cot.analyze(_dict(global_account_ratio=2.5, top_position_ratio=2.4,
                             funding_rate=0.0003)).regime != PositioningRegime.SQUEEZE_RISK
    assert cot.analyze(_dict(global_account_ratio=2.5, top_position_ratio=2.4,
                             funding_rate=0.0005)).regime == PositioningRegime.SQUEEZE_RISK
    assert cot.analyze(_dict(global_account_ratio=0.4, top_position_ratio=0.4,
                             funding_rate=-0.0005)).regime == PositioningRegime.SQUEEZE_RISK
    # divergência além de 0.40 (limite exato não dispara)
    assert cot.analyze(_dict(global_account_ratio=1.0,
                             top_position_ratio=1.40)).regime == PositioningRegime.NEUTRAL
    assert cot.analyze(_dict(global_account_ratio=1.0,
                             top_position_ratio=1.41)).regime == PositioningRegime.TOP_LONG_DIVERGENCE
    # OI expansion: |1h| >= 0.05 (limite exato dispara), senão 4h >= 0.10
    assert cot.analyze(_dict(oi_delta_1h=0.05, oi_delta_4h=0.0)).regime == PositioningRegime.OI_EXPANSION
    assert cot.analyze(_dict(oi_delta_1h=0.049,
                             oi_delta_4h=0.11)).regime == PositioningRegime.OI_EXPANSION


def test_funding_rate_kwarg_priority():
    cot = CryptoCOT()
    # kwarg vence o dict
    res = cot.analyze(_dict(funding_rate=0.0001,
                            global_account_ratio=2.5, top_position_ratio=2.4),
                      funding_rate=0.0005)
    assert res.funding_rate == 0.0005
    assert res.regime == PositioningRegime.SQUEEZE_RISK
    # kwarg None cai para o dict
    res2 = cot.analyze(_dict(funding_rate=0.0002))
    assert res2.funding_rate == 0.0002


def test_to_dict_and_legacy_serialization():
    cot = CryptoCOT()
    res = cot.analyze(_dict())
    d = res.to_dict()
    assert d["regime"] == "NEUTRAL"
    assert d["funding_rate"] == 0.0001
    assert d["symbol"] == "BTCUSDT"
    legacy = cot.to_legacy_analysis_result(res)
    assert legacy.source == "crypto_cot"
    assert legacy.confidence == 0.0  # context-only
    assert legacy.metrics["funding_rate"] == 0.0001


def test_snapshot_input_accepted():
    snap = BinancePositioningSnapshot(
        symbol="BTCUSDT", period="5m", observed_at=1_700_000_000.0,
        global_account_ratio=1.05, top_account_ratio=1.10,
        top_position_ratio=1.15, open_interest=100000.0,
        is_available=True, is_stale=False,
    )
    res = CryptoCOT().analyze(snap)
    assert res.regime == PositioningRegime.NEUTRAL
    assert res.is_available is True
