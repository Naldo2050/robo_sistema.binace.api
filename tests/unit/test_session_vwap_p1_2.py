# tests/unit/test_session_vwap_p1_2.py
# -*- coding: utf-8 -*-
"""
Suíte de Testes Automatizados para Session VWAP Canônico Ancorado em UTC 00:00:00.
Fase P1.2 (Arquitetura Context-Only).

Valida:
1. Cálculo matemático exato com valores conhecidos à mão.
2. Rollover de sessão em UTC 00:00:00 e independência de fuso horário local.
3. Validação defensiva (Volume zero, NaN, Inf, barras fora de ordem/duplicadas).
4. Equivalência exata: Implementação Incremental vs Batch de Referência.
5. Proteção contra corrupção pós-restart (Recovery desde UTC 00:00).
6. Integração de proveniência com InstitutionalAnalyticsEngine e payload compacto.
"""

import json
import math
import os
import sys
import time
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, ".")

from institutional.session_vwap import (
    SessionVWAPTracker,
    SessionVWAPSnapshot,
    SessionVWAPStatus,
    get_utc_session_start_ms,
)
from market_orchestrator.analysis.institutional_analytics import InstitutionalAnalyticsEngine
from market_orchestrator.ai.payload_builder_compact import build_compact_payload, _build_vwap_context
from market_orchestrator.ai.analyzer_qwen import AIAnalyzer


class TestSessionVWAPMathAndLogic(unittest.TestCase):
    """Testes unitários matemáticos e de fronteira de sessão UTC."""

    def setUp(self):
        self.tracker = SessionVWAPTracker(symbol="BTCUSDT")

    def test_hand_calculated_math(self):
        """
        Teste com resultado analítico exato conhecido à mão:
        Candle 1: Typical Price = (100 + 100 + 100)/3 = 100, Volume = 2 -> PV = 200
        Candle 2: Typical Price = (110 + 110 + 110)/3 = 110, Volume = 1 -> PV = 110
        Total PV = 310, Total Vol = 3 -> VWAP = 310 / 3 = 103.333333...
        """
        session_start = get_utc_session_start_ms()
        self.tracker.reset_session(session_start)

        t1 = session_start + 60000  # 00:01 UTC
        self.tracker.update_candle(t1, high=100.0, low=100.0, close=100.0, volume=2.0)
        self.assertEqual(self.tracker.current_vwap, 100.0)

        t2 = session_start + 120000  # 00:02 UTC
        self.tracker.update_candle(t2, high=110.0, low=110.0, close=110.0, volume=1.0)
        self.assertEqual(self.tracker.current_vwap, 103.33)
        self.assertAlmostEqual(self.tracker._sum_pv / self.tracker._sum_vol, 310.0 / 3.0, places=9)

    def test_utc_session_boundary_and_rollover(self):
        """
        Garante que às 23:59:59.999 UTC o VWAP acumula normalmente, e às 00:00:00.000 UTC
        do dia seguinte reseta os acumuladores sem contaminação do dia anterior.
        """
        day1_00_utc = 1788307200000  # Exemplo UTC 00:00:00.000
        self.tracker.reset_session(day1_00_utc)

        # Barra do dia 1 às 23:59 UTC
        t_2359 = day1_00_utc + (23 * 3600 + 59 * 60) * 1000
        self.tracker.update_candle(t_2359, high=80000.0, low=80000.0, close=80000.0, volume=10.0)
        self.assertEqual(self.tracker.current_vwap, 80000.0)
        self.assertEqual(self.tracker._bars_count, 1)

        # Barra do dia 2 às 00:00:00 UTC (virada de sessão)
        day2_00_utc = day1_00_utc + 86400000
        self.tracker.update_candle(day2_00_utc, high=70000.0, low=70000.0, close=70000.0, volume=5.0)

        # Deve ter resetado e acumulado apenas a barra do dia 2
        self.assertEqual(self.tracker.session_start_ms, day2_00_utc)
        self.assertEqual(self.tracker.current_vwap, 70000.0)
        self.assertEqual(self.tracker._bars_count, 1)
        self.assertEqual(self.tracker._sum_vol, 5.0)

    def test_defensive_data_validation(self):
        """Rejeita volume zero, NaN, Inf e dados fora de ordem."""
        session_start = get_utc_session_start_ms()
        self.tracker.reset_session(session_start)

        t1 = session_start + 60000
        self.tracker.update_candle(t1, high=100.0, low=98.0, close=99.0, volume=10.0)
        expected_vwap = round((99.0 * 10.0) / 10.0, 2)
        self.assertEqual(self.tracker.current_vwap, expected_vwap)

        # Volume zero -> ignorado
        self.tracker.update_candle(t1 + 60000, high=105.0, low=103.0, close=104.0, volume=0.0)
        self.assertEqual(self.tracker.current_vwap, expected_vwap)

        # NaN / Inf -> ignorado
        self.tracker.update_candle(t1 + 120000, high=float("nan"), low=90.0, close=95.0, volume=10.0)
        self.tracker.update_candle(t1 + 180000, high=100.0, low=90.0, close=float("inf"), volume=10.0)
        self.assertEqual(self.tracker.current_vwap, expected_vwap)

        # Barra duplicada (mesmo timestamp) -> ignorada
        self.tracker.update_candle(t1, high=200.0, low=200.0, close=200.0, volume=100.0)
        self.assertEqual(self.tracker.current_vwap, expected_vwap)

    def test_distance_fraction_and_side(self):
        """Verifica o cálculo de distance_fraction e classificação de side."""
        session_start = get_utc_session_start_ms()
        self.tracker.reset_session(session_start)
        self.tracker.update_candle(session_start + 60000, high=100.0, low=100.0, close=100.0, volume=10.0)

        # Preço 100.50 (+0.50% acima) -> ABOVE
        snap_above = self.tracker.get_snapshot(current_price=100.50)
        self.assertEqual(snap_above.session_vwap, 100.0)
        self.assertEqual(snap_above.distance_fraction, 0.005)
        self.assertEqual(snap_above.side, "ABOVE")

        # Preço 99.20 (-0.80% abaixo) -> BELOW
        snap_below = self.tracker.get_snapshot(current_price=99.20)
        self.assertEqual(snap_below.distance_fraction, -0.008)
        self.assertEqual(snap_below.side, "BELOW")

        # Preço 100.02 (+0.02% dentro da banda neutra 5 bps) -> AT
        snap_at = self.tracker.get_snapshot(current_price=100.02)
        self.assertEqual(snap_at.side, "AT")


class TestSessionVWAPBatchVsIncremental(unittest.TestCase):
    """Teste de Concordância Rigoroso: Incremental vs Batch de Referência."""

    def test_incremental_equals_batch(self):
        """
        Processa 200 candles sintéticos de forma incremental e compara com o cálculo em lote.
        Tolerância: erro absoluto < 1e-9.
        """
        session_start = get_utc_session_start_ms()
        tracker = SessionVWAPTracker(symbol="BTCUSDT")
        tracker.reset_session(session_start)

        candles = []
        batch_pv = 0.0
        batch_vol = 0.0

        for i in range(200):
            ts = session_start + (i + 1) * 60000
            # Preços variando entre 70000 e 78000
            h = 75000.0 + (math.sin(i * 0.1) * 2000.0) + 50.0
            l = 75000.0 + (math.sin(i * 0.1) * 2000.0) - 50.0
            c = 75000.0 + (math.sin(i * 0.1) * 2000.0)
            v = 10.0 + (i % 7) * 2.5

            typical = (h + l + c) / 3.0
            batch_pv += typical * v
            batch_vol += v

            candles.append({"t": ts, "h": h, "l": l, "c": c, "v": v})
            tracker.update_candle(ts, h, l, c, v)

        expected_batch_vwap = batch_pv / batch_vol
        actual_incremental_vwap = tracker._sum_pv / tracker._sum_vol

        self.assertAlmostEqual(actual_incremental_vwap, expected_batch_vwap, places=9)
        self.assertEqual(round(actual_incremental_vwap, 2), round(expected_batch_vwap, 2))


class TestSessionVWAPRecovery(unittest.IsolatedAsyncioTestCase):
    """Teste de Restart e Reconstrução da Sessão."""

    async def test_restart_recovery_equivalence(self):
        """
        Execução Contínua vs Restart no meio do dia:
        1. Processa 300 barras continuamente -> VWAP_cont.
        2. Simula restart: novo tracker às 15:00 UTC que carrega as 300 barras via rebuild -> VWAP_restart.
        3. Ambos devem produzir exatamente o mesmo valor.
        """
        session_start = get_utc_session_start_ms()

        # Mock de 300 klines da Binance
        mock_klines = []
        for i in range(300):
            ts = session_start + i * 60000
            o = 76000.0 + (i * 5.0)
            h = o + 20.0
            l = o - 20.0
            c = o + 10.0
            v = 15.0 + (i % 5)
            # Binance kline format: [open_time, open, high, low, close, volume, ...]
            mock_klines.append([ts, str(o), str(h), str(l), str(c), str(v), ts + 59999, "0", 100, "0", "0", "0"])

        # Tracker Contínuo
        continuous_tracker = SessionVWAPTracker(symbol="BTCUSDT")
        continuous_tracker.reset_session(session_start)
        continuous_tracker.update_batch(mock_klines)
        cont_vwap = continuous_tracker.current_vwap

        # Tracker Reiniciado com Rebuild
        restarted_tracker = SessionVWAPTracker(symbol="BTCUSDT")
        
        # Simula a resposta HTTP da Binance via mock_session
        from unittest.mock import MagicMock
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_klines)
        
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_resp
        mock_ctx.__aexit__.return_value = None
        mock_session = MagicMock()
        mock_session.get.return_value = mock_ctx

        success = await restarted_tracker.rebuild_from_binance(session=mock_session)
        # Se agora for no meio/fim do dia, 300 barras podem ser parciais em relação ao horário de agora,
        # mas se simulamos elapsed compatível:
        self.assertEqual(restarted_tracker.current_vwap, cont_vwap)
        self.assertEqual(restarted_tracker._bars_count, 300)

    async def test_full_day_1440_candles_pagination_rebuild(self):
        """Testa reconstrução de 1440 candles (24h completas) através de 2 páginas de 1000 + 440 candles."""
        now_ms = int(time.time() * 1000)
        session_start = get_utc_session_start_ms(now_ms)

        page1 = []
        for i in range(1000):
            ts = session_start + i * 60000
            page1.append([ts, "75000", "75100", "74900", "75050", "10", ts + 59999, "0", 10, "0", "0", "0"])

        page2 = []
        for i in range(1000, 1440):
            ts = session_start + i * 60000
            page2.append([ts, "75050", "75200", "75000", "75150", "10", ts + 59999, "0", 10, "0", "0", "0"])

        from unittest.mock import MagicMock
        mock_resp1 = AsyncMock()
        mock_resp1.status = 200
        mock_resp1.json = AsyncMock(return_value=page1)

        mock_resp2 = AsyncMock()
        mock_resp2.status = 200
        mock_resp2.json = AsyncMock(return_value=page2)

        mock_ctx1 = AsyncMock()
        mock_ctx1.__aenter__.return_value = mock_resp1
        mock_ctx1.__aexit__.return_value = None

        mock_ctx2 = AsyncMock()
        mock_ctx2.__aenter__.return_value = mock_resp2
        mock_ctx2.__aexit__.return_value = None

        mock_session = MagicMock()
        mock_session.get.side_effect = [mock_ctx1, mock_ctx2]

        tracker = SessionVWAPTracker(symbol="BTCUSDT")
        end_of_day_ms = session_start + (23 * 3600 + 59 * 60 + 59) * 1000  # 23:59:59 UTC
        success = await tracker.rebuild_from_binance(session=mock_session, now_ms=end_of_day_ms)
        self.assertTrue(success)
        self.assertEqual(tracker._bars_count, 1440)
        self.assertEqual(tracker._status, SessionVWAPStatus.VALID)
        self.assertIsNotNone(tracker.current_vwap)


class TestSessionVWAPIntegrationAndPayload(unittest.TestCase):
    """Testes de integração com InstitutionalAnalyticsEngine e Payload Compacto."""

    def test_engine_and_payload_provenance(self):
        """Verifica se session_vwap flui corretamente até o payload compacto e Groq summary."""
        engine = InstitutionalAnalyticsEngine(symbol="BTCUSDT")
        now_ms = int(time.time() * 1000)
        session_start = get_utc_session_start_ms(now_ms)

        # Injeta uma barra recente (10s atrás) no tracker do engine
        engine.session_vwap_tracker.reset_session(session_start)
        engine.session_vwap_tracker.update_candle(
            now_ms - 10000, high=77500.0, low=77300.0, close=77400.0, volume=100.0
        )

        res = engine.compute_all(current_price=77600.0)
        self.assertIn("session_vwap", res)
        self.assertEqual(res["session_vwap"]["status"], "VALID")
        self.assertTrue(res["session_vwap"]["is_valid"])
        self.assertEqual(res["session_vwap"]["method"], "ohlcv_1m_typical_price")

        # Injeta no evento e constrói payload
        event = {
            "symbol": "BTCUSDT",
            "tipo_evento": "ANALYSIS_TRIGGER",
            "preco_fechamento": 77600.0,
            "institutional_analytics": res,
        }
        compact = build_compact_payload(event)
        self.assertIn("vwap", compact)
        vwap_sec = compact["vwap"]
        self.assertEqual(vwap_sec["svw"], 77400.0)
        self.assertEqual(vwap_sec["side"], "above")
        self.assertEqual(vwap_sec["m"], "session_utc")
        self.assertIsInstance(vwap_sec["dist"], float)

        # Groq summary compressor preserva a seção vwap
        groq_summary = AIAnalyzer._build_groq_payload_summary(compact)
        self.assertIn("vwap", groq_summary)
        self.assertEqual(groq_summary["vwap"]["svw"], 77400.0)

        # Serialização RFC 8259 estrita
        serialized = json.dumps(groq_summary, allow_nan=False, ensure_ascii=False)
        self.assertIn('"vwap"', serialized)
        reparsed = json.loads(serialized)
        self.assertEqual(reparsed["vwap"]["svw"], 77400.0)
        self.assertEqual(reparsed["vwap"]["side"], "above")
        self.assertEqual(reparsed["vwap"]["m"], "session_utc")


if __name__ == "__main__":
    unittest.main(verbosity=2)
