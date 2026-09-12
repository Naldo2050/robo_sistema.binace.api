# tests/unit/test_session_vwap_coverage.py
# -*- coding: utf-8 -*-
"""Session VWAP: alimentação incremental + bootstrap de sessão + cobertura.

Regressão do bug HISTORY_NOT_UPDATED (linhas de janela com volume 0.0 eram
rejeitadas; tracker congelava no prefetch) e PARTIAL_SESSION (limit=200).
Sem rede: Binance mockado em todos os testes.
"""
import sys
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from institutional.session_vwap import (
    SessionVWAPTracker,
    SessionVWAPStatus,
    get_utc_session_start_ms,
)

DAY = 1789171200000  # 2026-09-12T00:00:00Z (fictício, só aritmética UTC)


def _candle(ts, o=77000.0, h=77100.0, low=76900.0, c=77050.0, v=10.0):
    return {"open_time": ts, "high": h, "low": low, "close": c, "volume": v}


class TestIncrementalFeeding(unittest.TestCase):
    """Cada candle fechado válido avança last_candle (nunca congela em T0)."""

    def test_last_candle_advances_per_closed_candle(self):
        tr = SessionVWAPTracker(symbol="BTCUSDT")
        tr.reset_session(DAY)
        t0 = DAY
        tr.update_candle(t0, 77000.0, 76900.0, 77050.0, 10.0)
        self.assertEqual(tr._last_candle_ms, t0)
        t1 = t0 + 60000
        tr.update_candle(t1, 77100.0, 77000.0, 77050.0, 12.0)
        self.assertEqual(tr._last_candle_ms, t1)
        t2 = t1 + 60000
        tr.update_candle(t2, 77200.0, 77100.0, 77150.0, 8.0)
        self.assertEqual(tr._last_candle_ms, t2)
        self.assertEqual(tr._bars_count, 3)

    def test_zero_volume_row_rejected_but_counts_coverage(self):
        tr = SessionVWAPTracker(symbol="BTCUSDT")
        tr.reset_session(DAY)
        tr.update_candle(DAY, 77000.0, 76900.0, 77050.0, 10.0)
        vwap_before = tr.current_vwap
        # minuto vazio genuíno: não move o valor, mas conta na cobertura
        tr.update_candle(DAY + 60000, 77010.0, 76910.0, 77060.0, 0.0)
        self.assertEqual(tr.current_vwap, vwap_before)
        self.assertEqual(tr._bars_count, 1)
        snap = tr.get_snapshot(current_price=77060.0, now_ms=DAY + 3 * 60000)
        self.assertEqual(snap.coverage_status, "PARTIAL")
        self.assertEqual(snap.missing_bars, 1)  # só o minuto corrente aberto falta... ver abaixo


class TestWeightedMath(unittest.TestCase):
    """Weighted != arithmetic mean (volumes diferentes)."""

    def test_weighted_average(self):
        tr = SessionVWAPTracker(symbol="BTCUSDT")
        tr.reset_session(DAY)
        # tp1 = 100 (v=1), tp2 = 200 (v=3) -> (100+600)/4 = 175 != 150
        tr.update_candle(DAY, 100.0, 100.0, 100.0, 1.0)
        tr.update_candle(DAY + 60000, 200.0, 200.0, 200.0, 3.0)
        self.assertEqual(tr.current_vwap, 175.0)


class TestGapDetection(unittest.TestCase):
    def test_gap_000102_missing(self):
        tr = SessionVWAPTracker(symbol="BTCUSDT")
        tr.reset_session(DAY)
        for i in (0, 1, 3):  # falta 00:02
            tr.update_candle(DAY + i * 60000, 100.0, 99.0, 99.5, 5.0)
        snap = tr.get_snapshot(current_price=99.5, now_ms=DAY + 5 * 60000)
        # minutos fechados 00:00..00:04 => 5 esperados, 3 recebidos
        self.assertEqual(snap.expected_bars, 5)
        self.assertEqual(snap.missing_bars, 2)
        self.assertEqual(snap.coverage_pct, 60.0)
        self.assertEqual(snap.coverage_status, "PARTIAL")
        self.assertEqual(snap.status, SessionVWAPStatus.PARTIAL)
        self.assertTrue(snap.is_valid)

    def test_full_session_when_complete(self):
        tr = SessionVWAPTracker(symbol="BTCUSDT")
        tr.reset_session(DAY)
        for i in range(10):
            tr.update_candle(DAY + i * 60000, 100.0, 99.0, 99.5, 5.0)
        snap = tr.get_snapshot(current_price=99.5, now_ms=DAY + 10 * 60000)
        self.assertEqual(snap.expected_bars, 10)
        self.assertEqual(snap.missing_bars, 0)
        self.assertEqual(snap.coverage_status, "FULL")
        self.assertEqual(snap.status, SessionVWAPStatus.VALID)
        self.assertEqual(snap.first_candle_ms, DAY)


class TestDedup(unittest.TestCase):
    def test_same_candle_twice_no_double_count(self):
        tr = SessionVWAPTracker(symbol="BTCUSDT")
        tr.reset_session(DAY)
        tr.update_candle(DAY, 100.0, 99.0, 99.5, 10.0)
        vol_before = tr._sum_vol
        tr.update_candle(DAY, 200.0, 200.0, 200.0, 100.0)  # mesmo ts, valores absurdos
        self.assertEqual(tr._sum_vol, vol_before)
        self.assertEqual(tr._bars_count, 1)


class TestRollover(unittest.TestCase):
    def test_midnight_resets_and_restarts_coverage(self):
        tr = SessionVWAPTracker(symbol="BTCUSDT")
        tr.reset_session(DAY)
        tr.update_candle(DAY + 23 * 3600000 + 59 * 60000, 100.0, 99.0, 99.5, 5.0)
        day2 = DAY + 86400000
        tr.update_candle(day2, 200.0, 199.0, 199.5, 7.0)
        self.assertEqual(tr.session_start_ms, day2)
        self.assertEqual(tr._bars_count, 1)
        self.assertEqual(tr._sum_vol, 7.0)  # sem vazamento do dia anterior
        snap = tr.get_snapshot(current_price=199.5, now_ms=day2 + 2 * 60000)
        self.assertEqual(snap.first_candle_ms, day2)
        self.assertEqual(snap.coverage_status, "PARTIAL")


class TestSchemaMismatchRegression(unittest.TestCase):
    """OHLC sem volume interno + volume_total real => history com volume real."""

    def test_window_row_carries_base_volume(self):
        # Simula _update_histories: ohlc de calculate_ohlc (SEM chave volume)
        # + volume_total de volume_metrics (base BTC). O tracker deve aceitar.
        tr = SessionVWAPTracker(symbol="BTCUSDT")
        tr.reset_session(DAY)
        ohlc = {"open": 77000.0, "high": 77100.0, "low": 76900.0, "close": 77050.0,
                "open_time": DAY, "close_time": DAY + 59999, "vwap": 77040.0}
        volume_total = 12.5  # base BTC, mesma candle
        row = {"timestamp": DAY, "open_time": DAY, "close_time": DAY + 59999,
               "open": ohlc["open"], "high": ohlc["high"], "low": ohlc["low"],
               "close": ohlc["close"],
               "volume": float(volume_total) if volume_total else float(ohlc.get("volume", 0.0)),
               "timeframe": "1m", "is_closed": True}
        self.assertGreater(row["volume"], 0.0)
        tr.update_candle(row["open_time"], row["high"], row["low"], row["close"], row["volume"])
        self.assertEqual(tr._last_candle_ms, DAY)
        self.assertEqual(tr._bars_count, 1)


class TestBootstrapPagination(unittest.TestCase):
    """Prefetch pagina desde 00:00 UTC; exclui candle aberto; conta requests."""

    def _klines(self, start, n, base_price=77000.0):
        out = []
        for i in range(n):
            ts = start + i * 60000
            out.append([ts, str(base_price), str(base_price + 10), str(base_price - 10),
                        str(base_price + 5), str(5.0 + i % 3), ts + 59999,
                        "0", 10, "0", "0", "0"])
        return out

    def _run_prefetch(self, now_ms, pages):
        """pages: lista de (page_start, klines). Retorna (bot, calls)."""
        import asyncio
        from types import SimpleNamespace
        from collections import deque
        import market_orchestrator.market_orchestrator as mo

        def _ctx_for(data):
            m = MagicMock()
            m.status = 200
            m.json = AsyncMock(return_value=data)
            ctx = MagicMock()
            ctx.__aenter__ = AsyncMock(return_value=m)
            ctx.__aexit__ = AsyncMock(return_value=None)
            return ctx

        class FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return None

            def get(self, url, params=None, timeout=None):
                calls.append(dict(params))
                start = params["startTime"]
                data = []
                for page_start, page in pages:
                    if page_start <= start < page_start + len(page) * 60000:
                        data = page
                        break
                return _ctx_for(data)

        calls = []
        bot = SimpleNamespace(
            symbol="BTCUSDT",
            pattern_ohlc_history=deque(maxlen=200),
            institutional_analytics=SimpleNamespace(session_vwap_tracker=None),
        )
        with patch("aiohttp.ClientSession", return_value=FakeSession()):
            with patch("time.time", return_value=now_ms / 1000.0):
                asyncio.run(mo.EnhancedMarketBot._prefetch_ohlc_history(bot))
        return bot, calls

    def test_early_morning_single_request(self):
        s0 = get_utc_session_start_ms(DAY)
        now_ms = s0 + 5 * 60000 + 30000  # 00:05:30 UTC
        pages = [(s0, self._klines(s0, 5))]
        bot, calls = self._run_prefetch(now_ms, pages)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["startTime"], s0)
        self.assertEqual(len(bot.pattern_ohlc_history), 5)

    def test_late_day_two_requests(self):
        s0 = get_utc_session_start_ms(DAY)
        now_ms = s0 + (23 * 60 + 30) * 60000  # 23:30 UTC
        pages = [(s0, self._klines(s0, 1000)),
                 (s0 + 1000 * 60000, self._klines(s0 + 1000 * 60000, 410))]
        bot, calls = self._run_prefetch(now_ms, pages)
        self.assertEqual(len(calls), 2)
        # deque guarda só as últimas 200; tracker recebe a sessão toda
        self.assertEqual(len(bot.pattern_ohlc_history), 200)

    def test_open_candle_excluded(self):
        s0 = get_utc_session_start_ms(DAY)
        now_ms = s0 + 10 * 60000 + 30000  # 00:10:30, candle 00:10 aberto
        pages = [(s0, self._klines(s0, 11))]  # inclui 00:10 aberto
        bot, calls = self._run_prefetch(now_ms, pages)
        ts_list = [r["timestamp"] for r in bot.pattern_ohlc_history]
        self.assertNotIn(s0 + 10 * 60000, ts_list)
        self.assertIn(s0 + 9 * 60000, ts_list)

    def test_partial_failure_not_full(self):
        import asyncio
        from types import SimpleNamespace
        from collections import deque
        import market_orchestrator.market_orchestrator as mo
        s0 = get_utc_session_start_ms(DAY)
        now_ms = s0 + 60 * 60000
        bot = SimpleNamespace(symbol="BTCUSDT",
                              pattern_ohlc_history=deque(maxlen=200),
                              institutional_analytics=SimpleNamespace(
                                  session_vwap_tracker=None))

        class BadSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return None

            def get(self, url, params=None, timeout=None):
                m = MagicMock()
                m.status = 500
                m.json = AsyncMock(return_value=None)
                ctx = MagicMock()
                ctx.__aenter__ = AsyncMock(return_value=m)
                ctx.__aexit__ = AsyncMock(return_value=None)
                return ctx

        with patch("aiohttp.ClientSession", return_value=BadSession()):
            with patch("time.time", return_value=now_ms / 1000.0):
                asyncio.run(mo.EnhancedMarketBot._prefetch_ohlc_history(bot))
        self.assertEqual(len(bot.pattern_ohlc_history), 0)


class TestPipelinePending(unittest.TestCase):
    """S3: missing == {last_closed} exatamente => PIPELINE_PENDING."""

    def test_only_last_closed_missing_is_pending(self):
        tr = SessionVWAPTracker(symbol="BTCUSDT")
        tr.reset_session(DAY)
        # 00:00..00:04 presentes; snapshot com last_closed=00:05 -> falta só 00:05
        for i in range(5):
            tr.update_candle(DAY + i * 60000, 100.0, 99.0, 99.5, 5.0)
        snap = tr.get_snapshot(current_price=99.5, now_ms=DAY + 6 * 60000)
        self.assertEqual(snap.missing_bars, 1)
        self.assertEqual(snap.missing_minutes, [DAY + 5 * 60000])
        self.assertEqual(snap.coverage_status, "PARTIAL")
        self.assertEqual(snap.status, SessionVWAPStatus.PIPELINE_PENDING)
        self.assertTrue(snap.is_valid)

    def test_old_missing_is_partial(self):
        tr = SessionVWAPTracker(symbol="BTCUSDT")
        tr.reset_session(DAY)
        # falta 00:01 (antiga); 00:05 presente
        for i in (0, 2, 3, 4, 5):
            tr.update_candle(DAY + i * 60000, 100.0, 99.0, 99.5, 5.0)
        snap = tr.get_snapshot(current_price=99.5, now_ms=DAY + 6 * 60000)
        self.assertEqual(snap.missing_minutes, [DAY + 60000])
        self.assertEqual(snap.status, SessionVWAPStatus.PARTIAL)
        self.assertTrue(snap.is_valid)

    def test_gap_plus_pending_is_partial(self):
        tr = SessionVWAPTracker(symbol="BTCUSDT")
        tr.reset_session(DAY)
        for i in (0, 2, 3, 4):  # faltam 00:01 (gap) e 00:05 (pending)
            tr.update_candle(DAY + i * 60000, 100.0, 99.0, 99.5, 5.0)
        snap = tr.get_snapshot(current_price=99.5, now_ms=DAY + 6 * 60000)
        self.assertEqual(snap.missing_bars, 2)
        self.assertEqual(snap.status, SessionVWAPStatus.PARTIAL)

    def test_false_positive_single_old_gap(self):
        # missing_count==1 NÃO basta: falta antiga com last_closed presente => PARTIAL
        tr = SessionVWAPTracker(symbol="BTCUSDT")
        tr.reset_session(DAY)
        for i in (0, 2, 3, 4, 5):  # 00:01 ausente, 00:05 presente
            tr.update_candle(DAY + i * 60000, 100.0, 99.0, 99.5, 5.0)
        snap = tr.get_snapshot(current_price=99.5, now_ms=DAY + 6 * 60000)
        self.assertEqual(snap.missing_bars, 1)
        self.assertNotEqual(snap.status, SessionVWAPStatus.PIPELINE_PENDING)
        self.assertEqual(snap.status, SessionVWAPStatus.PARTIAL)

    def test_cycle_proof_pending_then_incorporated(self):
        tr = SessionVWAPTracker(symbol="BTCUSDT")
        tr.reset_session(DAY)
        for i in range(5):
            tr.update_candle(DAY + i * 60000, 100.0, 99.0, 99.5, 5.0)
        s_n = tr.get_snapshot(current_price=99.5, now_ms=DAY + 6 * 60000)
        self.assertEqual(s_n.status, SessionVWAPStatus.PIPELINE_PENDING)
        # ciclo N+1: a candle entra; pendente passa a ser a seguinte
        tr.update_candle(DAY + 5 * 60000, 100.0, 99.0, 99.5, 5.0)
        s_n1 = tr.get_snapshot(current_price=99.5, now_ms=DAY + 7 * 60000)
        self.assertEqual(s_n1.status, SessionVWAPStatus.PIPELINE_PENDING)
        self.assertEqual(s_n1.missing_minutes, [DAY + 6 * 60000])
        self.assertEqual(s_n1.bars_count, 6)
        # ausência verdadeira em N+1 => PARTIAL (não acumula como pending)
        s_gap = tr.get_snapshot(current_price=99.5, now_ms=DAY + 9 * 60000)
        # faltam 00:07 e 00:08 (02 ausentes, um deles é last_closed)
        self.assertEqual(s_gap.status, SessionVWAPStatus.PARTIAL)


class TestUnavailableAndStale(unittest.TestCase):
    def test_no_closed_minute_yet_unavailable(self):
        tr = SessionVWAPTracker(symbol="BTCUSDT")
        tr.reset_session(DAY)
        snap = tr.get_snapshot(current_price=100.0, now_ms=DAY + 30000)
        self.assertEqual(snap.coverage_status, "UNAVAILABLE")
        self.assertEqual(snap.expected_bars, 0)

    def test_frozen_tracker_goes_stale(self):
        tr = SessionVWAPTracker(symbol="BTCUSDT")
        tr.reset_session(DAY)
        for i in range(10):
            tr.update_candle(DAY + i * 60000, 100.0, 99.0, 99.5, 5.0)
        snap = tr.get_snapshot(current_price=99.5, now_ms=DAY + 3600 * 1000)
        self.assertEqual(snap.status, SessionVWAPStatus.STALE)
        self.assertFalse(snap.is_valid)

    def test_bootstrap_prime_reaches_full(self):
        # Simula o prime do prefetch: batch completo da sessão até o último fechado.
        tr = SessionVWAPTracker(symbol="BTCUSDT")
        tr.reset_session(DAY)
        tr.update_batch([
            {"open_time": DAY + i * 60000, "high": 100.0, "low": 99.0,
             "close": 99.5, "volume": 5.0}
            for i in range(120)
        ])
        snap = tr.get_snapshot(current_price=99.5, now_ms=DAY + 120 * 60000 + 30000)
        self.assertEqual(snap.coverage_status, "FULL")
        self.assertEqual(snap.status, SessionVWAPStatus.VALID)
        self.assertEqual(snap.bars_count, 120)


class TestPayloadCoverageToken(unittest.TestCase):
    def test_pipeline_pending_reaches_compact_payload(self):
        from market_orchestrator.ai.payload_builder_compact import _build_vwap_context
        ctx = _build_vwap_context({"institutional_analytics": {"session_vwap": {
            "is_valid": True, "session_vwap": 77400.0, "distance_fraction": 0.001,
            "side": "ABOVE", "coverage_status": "PIPELINE_PENDING", "coverage_pct": 99.3,
        }}})
        self.assertEqual(ctx["m"], "session_utc")
        self.assertEqual(ctx["cov"], "PIPELINE_PENDING")
        self.assertEqual(ctx["cov_pct"], 99.3)

    def test_full_token_preserved(self):
        from market_orchestrator.ai.payload_builder_compact import _build_vwap_context
        ctx = _build_vwap_context({"institutional_analytics": {"session_vwap": {
            "is_valid": True, "session_vwap": 77400.0, "distance_fraction": 0.001,
            "side": "ABOVE", "coverage_status": "FULL", "coverage_pct": 100.0,
        }}})
        self.assertEqual(ctx["cov"], "FULL")


if __name__ == "__main__":
    unittest.main(verbosity=2)
