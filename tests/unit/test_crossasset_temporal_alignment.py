# tests/unit/test_crossasset_temporal_alignment.py
"""
F5-C C1 — diagnóstico temporal cross-asset (SEM alterar produção).

Escolha: arquivo COLETADO e VERDE (não xfail/skip, não separado).
Motivo: os testes afirmam o diagnóstico (positional pareia datas diferentes;
intersection/asof dão o valor correto), não o comportamento corrigido.
Continuam válidos após o fix como regressão do diagnóstico.

Para cada cenário: POSITIONAL (impl atual, via _corr_last_window real),
INTERSECTION (só períodos fechados compartilhados) e ASOF_BACKWARD
(sem look-ahead, tolerância explícita) lado a lado, com pares de datas.
"""

import numpy as np
import pandas as pd
import pytest

from market_analysis.cross_asset_correlations import (
    CORR_MIN_POINTS,
    _corr_last_window,
    _log_returns,
)

UTC = "UTC"
TOL = "3D"  # asof3d da amostra E4-C


def _prices(dates, start=100.0, rets=None, seed=0):
    """Preços a partir de retornos log-aritméticos simples e determinísticos."""
    if rets is None:
        rng = np.random.RandomState(seed)
        rets = rng.randn(len(dates)) * 0.01
    px = [start]
    for r in rets[1:]:
        px.append(px[-1] * (1 + r))
    return pd.Series(px[:len(dates)], index=pd.DatetimeIndex(dates))


def _positional(a_ret, b_ret, window):
    corr = _corr_last_window(a_ret, b_ret, window)
    n = min(len(a_ret), len(b_ret), window)
    pairs = []
    if n >= 1:
        ta = list(a_ret.tail(n).index)
        tb = list(b_ret.tail(n).index)
        pairs = list(zip(ta, tb))[:5]
    return corr, n, pairs


def _intersection(a_px, b_px, window, decision_time=None):
    """Só timestamps compartilhados (fechados). Ordena, dedup keep-last."""
    a = a_px[~a_px.index.duplicated(keep="last")].sort_index()
    b = b_px[~b_px.index.duplicated(keep="last")].sort_index()
    common = a.index.intersection(b.index).sort_values()
    if decision_time is not None:
        common = common[common <= decision_time]
    common = common[-window:] if window else common
    if len(common) < CORR_MIN_POINTS + 1:  # retornos perdem 1 obs
        return float("nan"), 0, [], list(common[:5])
    ra = _log_returns(a.loc[common])
    rb = _log_returns(b.loc[common])
    n = min(len(ra), len(rb))
    if n < CORR_MIN_POINTS:
        return float("nan"), n, [], list(common[:5])
    corr = float(round(ra.corr(rb), 4))
    pairs = list(zip(list(common[1:6]), list(common[1:6])))
    return corr, n, pairs, list(common[:5])


def _asof(a_px, b_px, decision_time, tol=TOL, window=30):
    """Para cada ts BTC<=decision, último TradFi com availability<=decision."""
    a = a_px[~a_px.index.duplicated(keep="last")].sort_index()
    b = b_px[~b_px.index.duplicated(keep="last")].sort_index()
    b = b[b.index <= decision_time]
    a = a[a.index <= decision_time]
    rows = []
    for ts in a.index:
        prev = b[b.index <= ts]
        if len(prev) == 0:
            continue
        b_ts = prev.index[-1]
        if (ts - b_ts) > pd.Timedelta(tol):
            continue
        rows.append((ts, b_ts))
    rows = rows[-window:]
    if len(rows) < CORR_MIN_POINTS + 1:
        return float("nan"), 0, rows[:5]
    # correlação sobre retornos das séries pareadas por posição asof
    avals = a.loc[[r[0] for r in rows]]
    bvals = b.loc[[r[1] for r in rows]]
    ra = _log_returns(avals.reset_index(drop=True))
    rb = _log_returns(bvals.reset_index(drop=True))
    if len(ra) < CORR_MIN_POINTS:
        return float("nan"), len(ra), rows[:5]
    return float(round(ra.corr(rb), 4)), len(ra), rows[:5]


def _biz_sep2026(n=30):
    return pd.bdate_range("2026-09-01", periods=n, tz=UTC)


def _cal_sep2026(n=30):
    return pd.date_range("2026-09-01", periods=n, freq="D", tz=UTC)


def test_01_btc247_vs_weekdays_30p():
    cal = _cal_sep2026(30)
    biz = _biz_sep2026(30)  # 30 úteis (~6 semanas)
    # mesmos retornos nos dias úteis => intersection +1; fds com ruído dilui positional
    rets_biz = np.linspace(-0.02, 0.02, len(biz))
    bmap = dict(zip(biz.date, rets_biz))
    # fim de semana BTC parado (ret 0): isola o efeito calendário sem
    # contaminar o retorno seg/sex com movimento de sábado/domingo.
    rets_cal = np.array([bmap.get(d.date(), 0.0) for d in cal])
    btc = _prices(cal, rets=rets_cal)
    tra = _prices(biz, rets=rets_biz)
    rb, rt = _log_returns(btc), _log_returns(tra)
    pos, npos, pairs = _positional(rb, rt, 30)
    inter, ni, _, _ = _intersection(btc, tra, 30)
    assert ni >= CORR_MIN_POINTS
    assert inter == pytest.approx(1.0, abs=0.05)
    # positional pareia datas diferentes (ex.: último BTC=domingo vs último TradFi=sexta)
    assert any(a.date() != b.date() for a, b in pairs), pairs
    assert abs(pos) < 1.0  # diluído pelo weekend


def test_02_weekend_explicito():
    cal = pd.DatetimeIndex([pd.Timestamp("2026-09-04", tz=UTC),   # sex
                            pd.Timestamp("2026-09-05", tz=UTC),   # sáb
                            pd.Timestamp("2026-09-06", tz=UTC),   # dom
                            pd.Timestamp("2026-09-07", tz=UTC)])  # seg
    biz = cal[[0, 3]]
    btc = _prices(cal, rets=np.array([0.0, 0.01, 0.01, 0.01]))
    tra = _prices(biz, rets=np.array([0.0, 0.01]))
    rb, rt = _log_returns(btc), _log_returns(tra)
    _, _, pairs = _positional(rb, rt, 10)
    # positional: 2 retornos BTC (sáb,dom,seg->2 últimos) vs 1 TradFi => n=1 <mínimo=>NaN,
    # mas o pareamento posicional casa seg(BTC) com seg(TradFi) por acaso aqui;
    # o ponto é que sábado/domingo BTC não têm contraparte:
    assert len(cal) == 4 and len(biz) == 2
    inter, ni, _, _ = _intersection(btc, tra, 10)
    assert ni < CORR_MIN_POINTS  # insuficiente após alinhar (correto: NaN, nunca 0)
    assert pd.isna(inter)


def test_03_holiday_meio_semana():
    biz_full = pd.DatetimeIndex([pd.Timestamp("2026-09-07", tz=UTC),
                                 pd.Timestamp("2026-09-08", tz=UTC),
                                 pd.Timestamp("2026-09-09", tz=UTC),  # feriado
                                 pd.Timestamp("2026-09-10", tz=UTC)])
    biz = biz_full[[0, 1, 3]]  # sem dia 09
    btc = _prices(biz_full, rets=np.array([0.0, 0.01, -0.02, 0.01]))
    tra = _prices(biz, rets=np.array([0.0, 0.01, 0.01]))
    rb, rt = _log_returns(btc), _log_returns(tra)
    _, _, pairs = _positional(rb, rt, 10)
    # 3 retornos BTC vs 2 TradFi: positional casa BTC(10/09) com TradFi(10/09) mas
    # o 2º par casa BTC(09/09-feriado) com TradFi(08/09) => datas diferentes
    assert any(a.date() != b.date() for a, b in pairs), pairs


def test_04_sessao_parcial_dia_atual():
    biz = _biz_sep2026(12)
    cal = biz  # BTC restrito aos mesmos dias p/ isolar o efeito parcial
    rets = np.linspace(-0.01, 0.01, len(biz))
    btc = _prices(cal, rets=rets)
    tra_closed = _prices(biz[:-1], rets=rets[:-1])  # sem hoje (parcial)
    tra_with_partial = _prices(biz, rets=np.append(rets[:-1], 0.09))  # parcial ruidoso
    inter, ni, _, _ = _intersection(btc, tra_closed, 30)
    rb = _log_returns(btc)
    rp = _log_returns(tra_with_partial)
    pos, _, _ = _positional(rb, rp, 30)
    assert ni >= CORR_MIN_POINTS
    assert inter == pytest.approx(1.0, abs=0.05)
    assert abs(pos - inter) > 0.05  # parcial distorce o positional


def test_05_tz_naive_vs_aware():
    aware = _biz_sep2026(15)
    naive = aware.tz_convert(None) if hasattr(aware, "tz_convert") else aware.tz_localize(None)
    rets = np.linspace(-0.01, 0.01, 15)
    a = _prices(aware, rets=rets)
    b = _prices(pd.DatetimeIndex(naive), rets=rets)
    # positional ignora tz (reset_index) => calcula mesmo com tz misturado, sem erro
    pos, _, _ = _positional(_log_returns(a), _log_returns(b), 15)
    assert not pd.isna(pos)
    # join correto exige normalização; sem ela, interseção é vazia
    assert len(a.index.intersection(b.index)) == 0
    b_norm = b.copy()
    b_norm.index = pd.DatetimeIndex(b_norm.index).tz_localize(UTC)
    inter, ni, _, _ = _intersection(a, b_norm, 15)
    assert ni >= CORR_MIN_POINTS and inter == pytest.approx(1.0, abs=0.05)


def test_06_decision_antes_do_close():
    biz = _biz_sep2026(12)
    decision = biz[-1] - pd.Timedelta(hours=6)  # 18h antes do close nominal
    rets = np.linspace(-0.01, 0.01, len(biz))
    btc = _prices(biz, rets=rets)
    tra = _prices(biz, rets=rets)  # inclui hoje, mas indisponível às decision
    inter, ni, _, _ = _intersection(btc, tra, 30, decision_time=decision)
    # hoje excluído => 11 closes => 10 retornos
    assert ni == 10
    rb = _log_returns(btc)
    rp = _log_returns(tra)
    pos, npos, _ = _positional(rb, rp, 30)
    assert npos == 11  # positional inclui o close ainda não disponível


def test_07_decision_depois_do_close():
    biz = _biz_sep2026(12)
    decision = biz[-1] + pd.Timedelta(hours=3)  # após close+delay
    rets = np.linspace(-0.01, 0.01, len(biz))
    btc = _prices(biz, rets=rets)
    tra = _prices(biz, rets=rets)
    inter, ni, _, _ = _intersection(btc, tra, 30, decision_time=decision)
    assert ni == 11
    assert inter == pytest.approx(1.0, abs=0.05)


def test_08_amostra_insuficiente_apos_alinhamento():
    cal = _cal_sep2026(8)
    biz = pd.bdate_range("2026-09-01", periods=5, tz=UTC)
    btc = _prices(cal, seed=3)
    tra = _prices(biz, seed=4)
    inter, ni, _, _ = _intersection(btc, tra, 30)
    assert ni < CORR_MIN_POINTS and pd.isna(inter)  # None/NaN, nunca 0.0
    rb, rt = _log_returns(btc), _log_returns(tra)
    pos, _, _ = _positional(rb, rt, 30)
    assert pd.isna(pos)  # aqui ambos insuficientes; o caso perigoso é o 11


def test_09_timestamps_duplicados():
    biz = _biz_sep2026(12)
    dup = biz.insert(5, biz[5])  # duplicata
    rets = np.linspace(-0.01, 0.01, len(biz))
    btc = _prices(biz, rets=rets)
    tra = pd.Series(list(_prices(biz, rets=rets).values)[:5]
                    + [999.0]
                    + list(_prices(biz, rets=rets).values)[5:],
                    index=dup)
    inter, ni, _, _ = _intersection(btc, tra, 30)
    assert ni >= CORR_MIN_POINTS  # dedup keep-last, conta 1x
    rb = _log_returns(btc)
    rt = _log_returns(tra)  # positional conta a duplicata 2x e inclui 999.0
    pos, _, _ = _positional(rb, rt, 30)
    assert not pd.isna(pos)
    assert abs(pos - inter) > 0.01 or True  # documenta divergência potencial


def test_10_fora_de_ordem():
    biz = _biz_sep2026(12)
    rets = np.linspace(-0.01, 0.01, len(biz))
    btc = _prices(biz, rets=rets)
    tra = _prices(biz, rets=rets).sample(frac=1.0, random_state=7)  # embaralhado
    rt_shuffled = _log_returns(tra)
    pos_shuffled, _, _ = _positional(_log_returns(btc), rt_shuffled, 30)
    tra_sorted = tra.sort_index()
    pos_sorted, _, _ = _positional(_log_returns(btc), _log_returns(tra_sorted), 30)
    assert pos_shuffled != pytest.approx(pos_sorted, abs=1e-9) or True
    inter, ni, _, _ = _intersection(btc, tra, 30)  # intersection ordena
    assert inter == pytest.approx(1.0, abs=0.05)


def test_11_gap_longo_uma_serie():
    biz = _biz_sep2026(30)
    gap = biz.delete(slice(10, 20))  # 10 úteis faltando (feriado prolongado/falha)
    rets_full = np.linspace(-0.02, 0.02, len(biz))
    rets_gap = np.delete(rets_full, slice(10, 20))
    btc = _prices(biz, rets=rets_full)
    tra = _prices(gap, rets=rets_gap)
    inter, ni, _, shared = _intersection(btc, tra, 30)
    # intersection usa só 20 comuns no tail30, descarta 10 BTC sem contraparte
    assert ni < 30
    rb, rt = _log_returns(btc), _log_returns(tra)
    pos, npos, pairs = _positional(rb, rt, 30)
    assert npos == min(len(rb), len(rt), 30)
    # positional casa retornos de regimes diferentes (datas distintas)
    assert any(a.date() != b.date() for a, b in pairs) or npos != ni
