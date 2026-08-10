# tests/unit/test_flow_consistency_regression.py
# -*- coding: utf-8 -*-
"""
Testes de regressao para a ETAPA DE FLOW (auditoria forense 2026-08-10).

Cobrem:
  A) Bucket boundaries: todo qty valido pertence a EXATAMENTE um bucket
     (whale aberto em [1.0, +inf) — antes 9999.0 ficava sem classificacao).
  B) Lifecycle do sector_flow: acumulador "desde o ultimo reset" (nao
     per-window). Duas janelas acumulam; reset zera.
  C) Contrato de deduplicacao: NAO ha dedup por trade id — o mesmo trade
     entregue 2x infla metricas (contrato documentado).
  D) Invariante: net_flow_1m == buy_volume - sell_volume, mesmo quando a
     analise roda com atraso (reference no passado + trades novos chegando),
     que era a causa dos valores J2/J4.
  E) Membership 1m/5m/15m: exatamente quais trades entram em cada janela,
     inclusive com trade fora da janela de analise (lag).
  G) Warmup/metadata: metadata.num_trades continua sendo a contagem da
     janela menor (1m), para nao inflar trade_intensity_v2 do ml_features.
"""

from decimal import Decimal

import pytest

from flow_analyzer import FlowAnalyzer

BASE = 1_750_000_000_000  # epoch ms


class FakeClock:
    """TimeManager fake: clock controlavel + metodos usados pelo analyzer."""

    def __init__(self, start_ms=BASE):
        self._now = start_ms

    def now_ms(self):
        return self._now

    def build_time_index(self, ts_ms, include_local=False, timespec="milliseconds"):
        return {"timestamp_utc": str(ts_ms), "epoch_ms": ts_ms}

    def format_timestamp(self, ts_ms):
        return str(ts_ms)

    def from_timestamp_ms(self, ts_ms, tz=None):
        return None


def _mk_analyzer(start_ms=BASE):
    return FlowAnalyzer(time_manager=FakeClock(start_ms))


def _trade(ts, qty, price=65000.0, buy=True):
    return {"p": price, "q": qty, "T": ts, "m": (not buy)}


def _feed(analyzer, trades, clock_now):
    """Alimenta trades avançando o clock junto (como o websocket real)."""
    for t in trades:
        analyzer.time_manager._now = t["T"]
        analyzer.process_trade(t)
    analyzer.time_manager._now = clock_now


def _sf(metrics, sector):
    return metrics["sector_flow"][sector]


def _of(metrics):
    return metrics["order_flow"]


# ============================================================================
# A) BUCKET BOUNDARIES
# ============================================================================

class TestBucketBoundaries:
    """Todo qty valido deve cair em exatamente um bucket."""

    @pytest.mark.parametrize("qty,sector", [
        (0.0, "retail"),
        (0.4999, "retail"),
        (0.5, "mid"),
        (0.9999, "mid"),
        (1.0, "whale"),
        (9998.999, "whale"),
        (9999.0, "whale"),      # antes: nenhum bucket (bug de cobertura)
        (9999.001, "whale"),    # antes: nenhum bucket
        (100000.0, "whale"),    # antes: nenhum bucket
    ])
    def test_qty_classified_exactly_once(self, qty, sector):
        a = _mk_analyzer()
        _feed(a, [_trade(BASE, qty, buy=True)], BASE)
        m = a.get_flow_metrics(reference_epoch_ms=BASE)
        for s in ("retail", "mid", "whale"):
            val = _sf(m, s)
            if s == sector:
                assert float(val["buy"]) == pytest.approx(qty, abs=1e-6), (
                    f"qty={qty} deveria estar em '{sector}' (tem buy={val['buy']})"
                )
            else:
                assert float(val["buy"]) == 0.0, (
                    f"qty={qty} nao deveria estar em '{s}'"
                )

    def test_whale_open_ended_sector_matches_cvd(self):
        """qty >= 9999 conta no sector whale e no CVD (não pode sumir)."""
        a = _mk_analyzer()
        _feed(a, [
            _trade(BASE, 9999.0, buy=True),
            _trade(BASE + 100, 1.5, buy=False),
        ], BASE + 100)
        m = a.get_flow_metrics(reference_epoch_ms=BASE + 100)
        whale = _sf(m, "whale")
        assert float(whale["buy"]) == pytest.approx(9999.0, abs=1e-6)
        assert float(whale["sell"]) == pytest.approx(1.5, abs=1e-6)
        assert float(whale["delta"]) == pytest.approx(9997.5, abs=1e-6)
        assert float(m["cvd"]) == pytest.approx(9997.5, abs=1e-6)


# ============================================================================
# B) SECTOR_FLOW LIFECYCLE (acumulador desde o ultimo reset)
# ============================================================================

class TestSectorFlowLifecycle:
    """Contrato: sector_flow acumula desde o ultimo reset (4h por padrao)."""

    def test_accumulates_across_windows_no_reset(self):
        a = _mk_analyzer()
        # Janela 1
        w1 = [_trade(BASE + i * 100, 0.2, buy=(i % 2 == 0)) for i in range(10)]
        _feed(a, w1, BASE + 60_000)
        m1 = a.get_flow_metrics(reference_epoch_ms=BASE + 60_000)
        buy1 = sum(_sf(m1, "retail")[k] for k in ("buy",)) + _sf(m1, "mid")["buy"] + _sf(m1, "whale")["buy"]
        sell1 = _sf(m1, "retail")["sell"] + _sf(m1, "mid")["sell"] + _sf(m1, "whale")["sell"]
        assert float(buy1) == pytest.approx(1.0, abs=1e-6)
        assert float(sell1) == pytest.approx(1.0, abs=1e-6)

        # Janela 2 (sem reset entre janelas)
        w2 = [_trade(BASE + 60_000 + i * 100, 0.3, buy=(i % 2 == 0)) for i in range(10)]
        _feed(a, w2, BASE + 120_000)
        m2 = a.get_flow_metrics(reference_epoch_ms=BASE + 120_000)
        # acumulou: janela1 + janela2
        buy2 = _sf(m2, "retail")["buy"] + _sf(m2, "mid")["buy"] + _sf(m2, "whale")["buy"]
        sell2 = _sf(m2, "retail")["sell"] + _sf(m2, "mid")["sell"] + _sf(m2, "whale")["sell"]
        assert float(buy2) == pytest.approx(1.0 + 1.5, abs=1e-6)
        assert float(sell2) == pytest.approx(1.0 + 1.5, abs=1e-6)
        # last_reset_ms disponivel no payload
        assert m2.get("last_reset_ms") is not None

    def test_reset_clears_sector_flow(self):
        a = _mk_analyzer()
        _feed(a, [_trade(BASE, 1.0, buy=True)], BASE)
        m1 = a.get_flow_metrics(reference_epoch_ms=BASE)
        assert float(_sf(m1, "whale")["buy"]) == 1.0

        a._reset_metrics()
        m2 = a.get_flow_metrics(reference_epoch_ms=BASE + 100)
        for s in ("retail", "mid", "whale"):
            assert float(_sf(m2, s)["buy"]) == 0.0
            assert float(_sf(m2, s)["sell"]) == 0.0
            assert float(_sf(m2, s)["delta"]) == 0.0


# ============================================================================
# C) CONTRATO DE DEDUPLICACAO (nao existe dedup por trade id)
# ============================================================================

class TestNoDedupContract:
    """Documenta o contrato: mesmo trade entregue 2x infla metricas."""

    def test_duplicate_trade_doubles_metrics(self):
        a = _mk_analyzer()
        t = _trade(BASE, 2.0, buy=True)
        _feed(a, [t, t], BASE)  # mesmo trade duas vezes (mesmo T, p, q, m)
        m = a.get_flow_metrics(reference_epoch_ms=BASE)
        # CVD e sector dobram (nao ha deduplicacao)
        assert float(m["cvd"]) == pytest.approx(4.0, abs=1e-6)
        assert float(_sf(m, "whale")["buy"]) == pytest.approx(4.0, abs=1e-6)


# ============================================================================
# D) net_flow_1m == buy_volume - sell_volume (com lag de processamento)
# ============================================================================

class TestNetFlowConsistency:
    """Invariante de populacao: net_flow_w deve vir da MESMA janela de
    buy/sell, mesmo quando a analise roda depois de novos trades chegarem."""

    def _scenario(self):
        a = _mk_analyzer()
        # Minuto 1 (janela de analise [BASE, BASE+60s]): 12 trades espalhados
        w1 = [
            _trade(BASE + i * 5000, 0.1 + 0.05 * (i % 4), buy=(i % 2 == 0))
            for i in range(12)
        ]
        # Minuto 2 (ja chegou quando a analise do minuto 1 roda): 5 trades.
        # Comeca ESTRITAMENTE depois do close (BASE+60s): trade com ts ==
        # now_ms pertence a janela fechada (limite inclusivo do analyzer).
        w2 = [
            _trade(BASE + 61_000 + i * 5000, 0.2, buy=(i % 2 == 0))
            for i in range(5)
        ]
        _feed(a, w1 + w2, BASE + 60_000 + 40_000)
        m = a.get_flow_metrics(reference_epoch_ms=BASE + 60_000)
        return m, w1, w2

    def test_net_flow_1m_matches_buy_minus_sell(self):
        m, w1, _w2 = self._scenario()
        of = _of(m)
        buy = float(of["buy_volume"])
        sell = float(of["sell_volume"])
        nf1 = float(of["net_flow_1m"])
        # Invariante: net_flow_1m == buy - sell (mesma populacao)
        assert nf1 == pytest.approx(buy - sell, abs=0.05), (
            f"net_flow_1m={nf1} vs buy-sell={buy - sell}"
        )

    def test_flow_imbalance_equals_imbalance_1m(self):
        """J4: flow_imbalance=0.6167 vs imbalance_1m=0.056 — populacoes
        diferentes. Com a correcao, as duas formulas usam os MESMOS trades."""
        m, _w1, _w2 = self._scenario()
        of = _of(m)
        imb = of.get("flow_imbalance")
        imb_1m = (of.get("buy_sell_ratio") or {}).get("ratios", {}).get("imbalance_1m")
        assert imb is not None, "flow_imbalance deveria existir (>= 5 trades)"
        assert imb_1m is not None, "imbalance_1m deveria existir"
        assert float(imb_1m) == pytest.approx(float(imb), abs=1e-3), (
            f"imbalance_1m={imb_1m} vs flow_imbalance={imb}"
        )

    def test_net_flow_1m_does_not_include_next_window_trades(self):
        m, w1, w2 = self._scenario()
        of = _of(m)
        nf1 = float(of["net_flow_1m"])
        # w2 (proxima janela) NAO pode entrar no net_flow_1m da janela 1
        w1_net = sum(
            (q * 65000.0) if buy else (-q * 65000.0) for t in w1
            for (q, buy) in [(t["q"], not t["m"])]
        )
        w2_net = sum(
            (q * 65000.0) if buy else (-q * 65000.0) for t in w2
            for (q, buy) in [(t["q"], not t["m"])]
        )
        assert nf1 == pytest.approx(w1_net, abs=1.0)
        assert nf1 != pytest.approx(w1_net + w2_net, abs=1.0)


# ============================================================================
# E) MEMBERSHIP 1m/5m/15m
# ============================================================================

class TestWindowMembership:
    """Exatamente quais trades entram em 1m/5m/15m, com lag de analise."""

    def test_membership_with_analysis_lag(self):
        a = _mk_analyzer()
        trades = []
        # A: dentro de 1m/5m/15m (6 trades)
        A = [_trade(BASE - 30_000 + i * 1000, 0.1, buy=(i % 2 == 0)) for i in range(6)]
        # B: dentro de 5m/15m, fora de 1m (3 trades)
        B = [_trade(BASE - 90_000 + i * 1000, 0.2, buy=True) for i in range(3)]
        # C: dentro de 5m/15m, fora de 1m (2 trades)
        C = [_trade(BASE - 240_000 + i * 1000, 0.3, buy=False) for i in range(2)]
        # D: dentro de 15m, fora de 1m/5m (1 trade)
        D = [_trade(BASE - 600_000, 0.4, buy=True)]
        # E: fora de 15m (nao deve entrar em nenhuma janela)
        E = [_trade(BASE - 960_000, 0.5, buy=False)]
        # F: trade da proxima janela (ja chegou quando a analise roda)
        F = [_trade(BASE + 30_000, 0.6, buy=True) for _ in range(2)]

        # Feed em ordem CRESCENTE de ts (como o websocket real) para que o
        # caminho de agregação (cache) fique ATIVO e o bug de população apareça.
        all_t = sorted(A + B + C + D + E + F, key=lambda x: x["T"])
        _feed(a, all_t, BASE + 60_000)
        m = a.get_flow_metrics(reference_epoch_ms=BASE)
        of = _of(m)

        def net(ts_list):
            return sum(
                (t["q"] * 65000.0) if (not t["m"]) else (-t["q"] * 65000.0)
                for t in ts_list
            )

        assert float(of["net_flow_1m"]) == pytest.approx(net(A), abs=1.0), (
            f"1m={of['net_flow_1m']} esperado {net(A)}"
        )
        assert float(of["net_flow_5m"]) == pytest.approx(net(A + B + C), abs=1.0), (
            f"5m={of['net_flow_5m']} esperado {net(A + B + C)}"
        )
        assert float(of["net_flow_15m"]) == pytest.approx(net(A + B + C + D), abs=1.0), (
            f"15m={of['net_flow_15m']} esperado {net(A + B + C + D)}"
        )
        # E e F nao podem aparecer em nenhuma janela da analise
        assert abs(float(of["net_flow_15m"]) - net(A + B + C + D + F)) > 1.0, (
            "trade da proxima janela (F) vazou para net_flow_15m"
        )
        assert abs(float(of["net_flow_15m"]) - net(A + B + C + D + E)) > 1.0, (
            "trade fora de 15m (E) vazou para net_flow_15m"
        )

    def test_warmup_60s_history_no_fake_coverage(self):
        """So 60s de historico: 5m/15m refletem apenas o que existe
        (nao podem incluir trades inexistentes), e 1m continua correto."""
        a = _mk_analyzer()
        w = [_trade(BASE + i * 1000, 0.1, buy=(i % 2 == 0)) for i in range(60)]
        _feed(a, w, BASE + 60_000)
        m = a.get_flow_metrics(reference_epoch_ms=BASE + 60_000)
        of = _of(m)
        expected_1m = sum(
            (t["q"] * 65000.0) if (not t["m"]) else (-t["q"] * 65000.0) for t in w
        )
        # Com 60s de historia, 5m == 15m == 1m (mesmo conjunto de trades)
        assert float(of["net_flow_1m"]) == pytest.approx(expected_1m, abs=1.0)
        assert float(of["net_flow_5m"]) == pytest.approx(expected_1m, abs=1.0)
        assert float(of["net_flow_15m"]) == pytest.approx(expected_1m, abs=1.0)
        # E a contagem de trades da janela menor confere
        assert m["metadata"]["num_trades"] == 60


# ============================================================================
# G) metadata.num_trades (contrato com ml_features.trade_intensity_v2)
# ============================================================================

class TestMetadataNumTrades:
    """num_trades deve ser a contagem da janela MENOR (1m), nao do buffer."""

    def test_num_trades_counts_only_last_minute(self):
        a = _mk_analyzer()
        # 150s de historia a 10 trades/s
        w = [_trade(BASE + i * 100, 0.1, buy=(i % 2 == 0)) for i in range(1500)]
        _feed(a, w, BASE + 150_000)
        m = a.get_flow_metrics(reference_epoch_ms=BASE + 150_000)
        assert m["metadata"]["num_trades"] == 600, (
            f"num_trades={m['metadata']['num_trades']} (esperado 600: ultimos 60s)"
        )
