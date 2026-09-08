# tests/unit/test_regime_corr_missing.py
"""
P1-A: corr ausente/nonfinite/insuficiente NUNCA vira 0.0 observado.

0.0 real => numérico (CRYPTO_NATIVE). None/ausente/NaN/±Inf/n<min =>
CorrelationRegime.UNKNOWN (sem "neutral" inventado).
"""

import pytest

from market_analysis.regime_detector import (
    CorrelationRegime,
    EnhancedRegimeDetector,
)


def _corr(cross):
    return EnhancedRegimeDetector()._analyze_correlation_regime(cross)


def test_1_real_positive_stays_numeric():
    assert _corr({"btc_dxy_corr_30d": 0.5}) == CorrelationRegime.MACRO_CORRELATED


def test_2_real_negative_stays_numeric():
    # dxy forte sozinho segue o ramo legado (default MACRO_CORRELATED);
    # o ponto é continuar numérico, não UNKNOWN.
    assert _corr({"btc_dxy_corr_30d": -0.5}) == CorrelationRegime.MACRO_CORRELATED


def test_3_real_zero_stays_numeric():
    assert _corr({"btc_dxy_corr_30d": 0.0}) == CorrelationRegime.CRYPTO_NATIVE


def test_4_none_is_unknown():
    assert _corr({"btc_dxy_corr_30d": None}) == CorrelationRegime.UNKNOWN


def test_5_missing_key_is_unknown():
    assert _corr({}) == CorrelationRegime.UNKNOWN


def test_6_nan_is_unknown():
    assert _corr({"btc_dxy_corr_30d": float("nan")}) == CorrelationRegime.UNKNOWN


def test_7_pos_inf_is_unknown():
    assert _corr({"btc_dxy_corr_30d": float("inf")}) == CorrelationRegime.UNKNOWN


def test_8_neg_inf_is_unknown():
    assert _corr({"btc_dxy_corr_30d": float("-inf")}) == CorrelationRegime.UNKNOWN


def test_9_insufficient_n_is_unknown():
    assert _corr({"btc_dxy_corr_30d": 0.5,
                  "btc_dxy_corr_30d_n": 5}) == CorrelationRegime.UNKNOWN
    # n suficiente mantém o numérico
    assert _corr({"btc_dxy_corr_30d": 0.5,
                  "btc_dxy_corr_30d_n": 30}) == CorrelationRegime.MACRO_CORRELATED


def test_10_insufficient_unavailable_is_unknown():
    # NaN = insufficient computado (n < min); chave ausente = unavailable.
    # Nunca podem virar zero observado. Nomes honestos: este teste NÃO
    # reproduz stale-com-valores (ver test_11).
    assert _corr({"btc_dxy_corr_30d": float("nan")}) == CorrelationRegime.UNKNOWN
    assert _corr({"correlation_spy": None}) == CorrelationRegime.UNKNOWN


def test_11_stale_usable_with_values_stays_numeric():
    # stale_usable (E3-B): snapshot stale preserva values completos e o
    # builder encaminha {correlation_spy: None, btc_dxy_corr_30d: <float>,
    # btc_dxy_corr_30d_n: <int>} — freshness/status NÃO descem ao detector.
    # Com valor finito + n>=min, stale continua evidência numérica.
    assert _corr({"correlation_spy": None, "btc_dxy_corr_30d": -0.45,
                  "btc_dxy_corr_30d_n": 30}) == CorrelationRegime.MACRO_CORRELATED
    assert _corr({"correlation_spy": None, "btc_dxy_corr_30d": 0.05,
                  "btc_dxy_corr_30d_n": 30}) == CorrelationRegime.CRYPTO_NATIVE


def test_json_safe_nan_does_not_recreate_zero():
    # json_safe: NaN -> None; o detector não pode refazer o 0 via `or 0`
    # (NaN é truthy, None é falsy: ambos os ramos antigos fabricavam regime).
    from common.json_safe import sanitize_json_safe
    clean = sanitize_json_safe({"btc_dxy_corr_30d": float("nan")})
    assert clean == {"btc_dxy_corr_30d": None}
    assert _corr(clean) == CorrelationRegime.UNKNOWN
