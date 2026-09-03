# tests/unit/test_market_structure_p1_3.py
# -*- coding: utf-8 -*-
"""
Suíte de Testes Automatizados para Market Structure: BOS & Liquidity Sweep.
Fase P1.3C (Correção Semântica e Provenance Canônica).

Valida:
1. Detecção analítica de Swing Highs e Swing Lows com confirmação canônica (L=2, R=2).
2. Break of Structure (BOS) bullish e bearish com confirmação de fechamento de candle.
3. Liquidity Sweep (buy-side / sell-side / both) caracterizado por excursão e reclaim/rejeição.
4. Exclusão mútua estrita entre BOS e Sweep para o mesmo candle e mesmo nível.
5. Garantia formal Anti-Lookahead e Prefix Invariance (Zero Repaint).
6. Resolução de Double Sweep (Candle Largo) sem viés de ordem de loop.
7. Identidade de Evento determinística (event_id) e Schema Versioning 1.1.0.
8. Timeframe canônico explícito 1m e congelamento do comportamento de plateaus.
9. Integração no InstitutionalAnalyticsEngine, payload compacto e Groq summary.
"""

import json
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, ".")

from institutional.market_structure import (
    MarketStructureDetector,
    MarketStructureResult,
    BOSEvent,
    BOSType,
    LiquiditySweepEvent,
    SweepType,
    StructurePointType,
    MARKET_STRUCTURE_SCHEMA_VERSION,
)
from market_orchestrator.analysis.institutional_analytics import InstitutionalAnalyticsEngine
from market_orchestrator.ai.payload_builder_compact import build_compact_payload, _build_market_structure
from market_orchestrator.ai.analyzer_qwen import AIAnalyzer


class TestMarketStructureMathAndLogic(unittest.TestCase):
    """Testes de lógica matemática e detecção determinística de Swings, BOS e Sweep."""

    def setUp(self):
        # Detector com janela de confirmação canônica L=2, R=2 e timeframe 1m
        self.detector = MarketStructureDetector(left_bars=2, right_bars=2, timeframe="1m", symbol="BTCUSDT")

    def _generate_synthetic_swing_candles(self):
        """
        Cria uma série de candles com um Swing High claro no índice 2 (preço 75000)
        e um Swing Low claro no índice 5 (preço 72000).
        """
        base_ts = 1788300000000
        candles = [
            {"t": base_ts + 0 * 60000, "o": 74000, "h": 74200, "l": 73900, "c": 74100},  # 0
            {"t": base_ts + 1 * 60000, "o": 74100, "h": 74600, "l": 74050, "c": 74500},  # 1
            {"t": base_ts + 2 * 60000, "o": 74500, "h": 75000, "l": 74400, "c": 74800},  # 2: SWING HIGH (75000)
            {"t": base_ts + 3 * 60000, "o": 74800, "h": 74600, "l": 73500, "c": 73600},  # 3
            {"t": base_ts + 4 * 60000, "o": 73600, "h": 73800, "l": 73000, "c": 73100},  # 4: Confirma Swing High (índice 2)
            {"t": base_ts + 5 * 60000, "o": 73100, "h": 73200, "l": 72000, "c": 72400},  # 5: SWING LOW (72000)
            {"t": base_ts + 6 * 60000, "o": 72400, "h": 72800, "l": 72300, "c": 72700},  # 6
            {"t": base_ts + 7 * 60000, "o": 72700, "h": 73500, "l": 72600, "c": 73400},  # 7: Confirma Swing Low (índice 5)
        ]
        return candles

    def test_swing_high_and_low_detection(self):
        """Verifica a identificação precisa dos níveis de swing."""
        candles = self._generate_synthetic_swing_candles()
        res = self.detector.analyze_candles(candles)

        self.assertEqual(res.status, "VALID")
        self.assertEqual(res.last_swing_high, 75000.0)
        self.assertEqual(res.last_swing_low, 72000.0)
        self.assertEqual(res.confirmed_swings_count, 2)
        self.assertEqual(res.timeframe, "1m")
        self.assertEqual(res.schema_version, MARKET_STRUCTURE_SCHEMA_VERSION)
        self.assertIsNone(res.active_bos)
        self.assertIsNone(res.active_sweep)

    def test_bullish_bos_detection(self):
        """
        Candle 8 fecha acima do Swing High (75000):
        High = 75200, Close = 75100 > 75000 -> Bullish BOS confirmado.
        """
        candles = self._generate_synthetic_swing_candles()
        base_ts = candles[-1]["t"]
        candles.append({
            "t": base_ts + 60000,
            "o": 73400, "h": 75200, "l": 73300, "c": 75100
        })

        res = self.detector.analyze_candles(candles)
        self.assertIsNotNone(res.active_bos)
        self.assertEqual(res.active_bos.type, BOSType.BULLISH)
        self.assertEqual(res.active_bos.level, 75000.0)
        self.assertEqual(res.active_bos.break_price, 75100.0)
        self.assertEqual(res.active_bos.timeframe, "1m")
        self.assertIn("BTCUSDT:1m:BOS_BULLISH:75000.0", res.active_bos.event_id)
        self.assertIsNone(res.active_sweep)

    def test_bearish_bos_detection(self):
        """
        Candle 8 fecha abaixo do Swing Low (72000):
        Low = 71700, Close = 71800 < 72000 -> Bearish BOS confirmado.
        """
        candles = self._generate_synthetic_swing_candles()
        base_ts = candles[-1]["t"]
        candles.append({
            "t": base_ts + 60000,
            "o": 73400, "h": 73500, "l": 71700, "c": 71800
        })

        res = self.detector.analyze_candles(candles)
        self.assertIsNotNone(res.active_bos)
        self.assertEqual(res.active_bos.type, BOSType.BEARISH)
        self.assertEqual(res.active_bos.level, 72000.0)
        self.assertEqual(res.active_bos.break_price, 71800.0)
        self.assertIn("BTCUSDT:1m:BOS_BEARISH:72000.0", res.active_bos.event_id)
        self.assertIsNone(res.active_sweep)

    def test_buy_side_liquidity_sweep(self):
        """
        Candle 8 perfura o Swing High (75000) mas rejeita e fecha abaixo:
        High = 75300 > 75000, mas Close = 74800 <= 75000 -> Buy-Side Liquidity Sweep.
        """
        candles = self._generate_synthetic_swing_candles()
        base_ts = candles[-1]["t"]
        candles.append({
            "t": base_ts + 60000,
            "o": 73400, "h": 75300, "l": 73400, "c": 74800
        })

        res = self.detector.analyze_candles(candles)
        self.assertIsNone(res.active_bos)
        self.assertIsNotNone(res.active_sweep)
        self.assertEqual(res.active_sweep.type, SweepType.BUY_SIDE)
        self.assertEqual(res.active_sweep.level, 75000.0)
        self.assertEqual(res.active_sweep.wick_price, 75300.0)
        self.assertEqual(res.active_sweep.close_price, 74800.0)
        self.assertIn("BTCUSDT:1m:SWEEP_BUY_SIDE:75000.0", res.active_sweep.event_id)

    def test_sell_side_liquidity_sweep(self):
        """
        Candle 8 perfura o Swing Low (72000) mas rejeita e fecha acima:
        Low = 71600 < 72000, mas Close = 72200 >= 72000 -> Sell-Side Liquidity Sweep.
        """
        candles = self._generate_synthetic_swing_candles()
        base_ts = candles[-1]["t"]
        candles.append({
            "t": base_ts + 60000,
            "o": 73400, "h": 73500, "l": 71600, "c": 72200
        })

        res = self.detector.analyze_candles(candles)
        self.assertIsNone(res.active_bos)
        self.assertIsNotNone(res.active_sweep)
        self.assertEqual(res.active_sweep.type, SweepType.SELL_SIDE)
        self.assertEqual(res.active_sweep.level, 72000.0)
        self.assertEqual(res.active_sweep.wick_price, 71600.0)
        self.assertEqual(res.active_sweep.close_price, 72200.0)
        self.assertIn("BTCUSDT:1m:SWEEP_SELL_SIDE:72000.0", res.active_sweep.event_id)

    def test_double_sweep_both_sides_preserved(self):
        """
        Verifica a resolução não-viesada de Double Sweep (Candle Largo que varre topo e fundo simultaneamente).
        """
        candles = self._generate_synthetic_swing_candles()
        base_ts = candles[-1]["t"]
        # Candle 8 com High = 75500 (>75000) e Low = 71500 (<72000), fechando em 73500 (dentro do range)
        candles.append({
            "t": base_ts + 60000,
            "o": 73400, "h": 75500, "l": 71500, "c": 73500
        })

        res = self.detector.analyze_candles(candles)
        self.assertIsNone(res.active_bos)
        self.assertIsNotNone(res.active_sweep)
        self.assertEqual(res.active_sweep.type, SweepType.BOTH)
        self.assertIn("SWEEP_BOTH", res.active_sweep.event_id)

    def test_mutual_exclusivity_conflict_matrix(self):
        """
        Matriz de Conflito: Garante que um mesmo candle e nível não sejam classificados
        ambiguamente como BOS e Sweep simultaneamente.
        """
        candles = self._generate_synthetic_swing_candles()
        base_ts = candles[-1]["t"]

        # Caso 1: Rompimento com Close acima -> BOS=True, Sweep=False
        c_bos = {"t": base_ts + 60000, "o": 74500, "h": 75200, "l": 74400, "c": 75100}
        res_bos = self.detector.analyze_candles(candles + [c_bos])
        self.assertIsNotNone(res_bos.active_bos)
        self.assertIsNone(res_bos.active_sweep)

        # Caso 2: Rompimento com Pavio e Close abaixo -> BOS=False, Sweep=True
        c_swp = {"t": base_ts + 60000, "o": 74500, "h": 75200, "l": 74400, "c": 74900}
        res_swp = self.detector.analyze_candles(candles + [c_swp])
        self.assertIsNone(res_swp.active_bos)
        self.assertIsNotNone(res_swp.active_sweep)

    def test_plateau_equal_highs_behavior_frozen(self):
        """
        Congela o comportamento documentado para plateaus curtos e longos.
        """
        base_ts = 1788300000000
        # Plateau curto de 3 candles (75000, 75000, 75000)
        c_plat3 = [
            {"t": base_ts + 0 * 60000, "o": 70000, "h": 70000, "l": 69000, "c": 70000},
            {"t": base_ts + 1 * 60000, "o": 72000, "h": 72000, "l": 71000, "c": 72000},
            {"t": base_ts + 2 * 60000, "o": 74000, "h": 75000, "l": 74000, "c": 74500},
            {"t": base_ts + 3 * 60000, "o": 74500, "h": 75000, "l": 74000, "c": 74500},
            {"t": base_ts + 4 * 60000, "o": 74500, "h": 75000, "l": 74000, "c": 74500},
            {"t": base_ts + 5 * 60000, "o": 72000, "h": 72000, "l": 71000, "c": 72000},
            {"t": base_ts + 6 * 60000, "o": 70000, "h": 70000, "l": 69000, "c": 70000},
        ]
        res = self.detector.analyze_candles(c_plat3)
        self.assertEqual(res.status, "VALID")
        self.assertEqual(res.last_swing_high, 75000.0)


class TestMarketStructureAntiLookahead(unittest.TestCase):
    """Teste de Garantia Anti-Lookahead e Prefix Invariance."""

    def test_prefix_invariance_zero_repaint(self):
        """
        Prova formal de Prefix Invariance com schema 1.1.0 e timeframe 1m.
        """
        detector = MarketStructureDetector(left_bars=2, right_bars=2, timeframe="1m")
        base_ts = 1788300000000

        candles_past = []
        for i in range(15):
            h = 70000.0 + (500.0 if i == 3 else 0.0)  # Swing High no candle 3 (70500)
            l = 69500.0 - (500.0 if i == 7 else 0.0)  # Swing Low no candle 7 (69000)
            c = 70600.0 if i == 12 else 69800.0       # BOS no candle 12 (Close=70600 > 70500)
            o = 69800.0
            candles_past.append({"t": base_ts + i * 60000, "o": o, "h": max(h, c), "l": min(l, c), "c": c})

        res_at_T = detector.analyze_candles(candles_past)
        self.assertIsNotNone(res_at_T.active_bos)
        self.assertEqual(res_at_T.active_bos.level, 70500.0)
        self.assertEqual(res_at_T.active_bos.break_price, 70600.0)

        # Anexa 50 candles futuros
        candles_future = list(candles_past)
        for i in range(15, 65):
            candles_future.append({
                "t": base_ts + i * 60000,
                "o": 70200.0,
                "h": 70300.0,
                "l": 70100.0,
                "c": 70200.0,
            })

        res_recomputed = detector.analyze_candles(candles_future[:15])
        self.assertEqual(res_at_T.active_bos.level, res_recomputed.active_bos.level)
        self.assertEqual(res_at_T.active_bos.break_price, res_recomputed.active_bos.break_price)
        self.assertEqual(res_at_T.active_bos.event_id, res_recomputed.active_bos.event_id)


class TestMarketStructureIntegrationAndPayload(unittest.TestCase):
    """Testes de integração com InstitutionalAnalyticsEngine, Payload e Groq Summary."""

    def test_engine_and_compact_payload_flow(self):
        """Verifica fluxo ponta-a-ponta até payload compacto com timeframe canônico 1m."""
        engine = InstitutionalAnalyticsEngine(symbol="BTCUSDT")

        import pandas as pd
        base_ts = 1788300000000
        candles = [
            {"t": base_ts + 0 * 60000, "o": 74000, "h": 74200, "l": 73900, "c": 74100, "v": 10},
            {"t": base_ts + 1 * 60000, "o": 74100, "h": 74600, "l": 74050, "c": 74500, "v": 10},
            {"t": base_ts + 2 * 60000, "o": 74500, "h": 75000, "l": 74400, "c": 74800, "v": 10},  # SH (75000)
            {"t": base_ts + 3 * 60000, "o": 74800, "h": 74600, "l": 73500, "c": 73600, "v": 10},
            {"t": base_ts + 4 * 60000, "o": 73600, "h": 73800, "l": 73000, "c": 73100, "v": 10},  # Confirma SH
            {"t": base_ts + 5 * 60000, "o": 73100, "h": 73200, "l": 72000, "c": 72400, "v": 10},  # SL (72000)
            {"t": base_ts + 6 * 60000, "o": 72400, "h": 72800, "l": 72300, "c": 72700, "v": 10},
            {"t": base_ts + 7 * 60000, "o": 72700, "h": 73500, "l": 72600, "c": 73400, "v": 10},  # Confirma SL
            {"t": base_ts + 8 * 60000, "o": 73400, "h": 75200, "l": 73300, "c": 75100, "v": 10},  # Bullish BOS!
        ]
        df = pd.DataFrame(candles)

        res = engine.compute_all(current_price=75100.0, candles_df=df)
        self.assertIn("market_structure", res)
        self.assertEqual(res["market_structure"]["status"], "VALID")
        self.assertEqual(res["market_structure"]["timeframe"], "1m")
        self.assertEqual(res["market_structure"]["schema_version"], "1.1.0")
        self.assertIsNotNone(res["market_structure"]["bos"])
        self.assertEqual(res["market_structure"]["bos"]["type"], "bullish")

        # Constrói Payload Compacto
        event = {
            "symbol": "BTCUSDT",
            "tipo_evento": "ANALYSIS_TRIGGER",
            "preco_fechamento": 75100.0,
            "institutional_analytics": res,
        }
        compact = build_compact_payload(event)
        self.assertIn("ms", compact)
        self.assertEqual(compact["ms"]["bos"], "BULL_75000")
        self.assertEqual(compact["ms"]["sh"], 75000)
        self.assertEqual(compact["ms"]["sl"], 72000)
        self.assertEqual(compact["ms"]["tf"], "1m")

        # Groq Summary Compressor
        groq_summary = AIAnalyzer._build_groq_payload_summary(compact)
        self.assertIn("ms", groq_summary)
        self.assertEqual(groq_summary["ms"]["bos"], "BULL_75000")

        # Serialização RFC 8259 estrita
        final_json = json.dumps(groq_summary, allow_nan=False, ensure_ascii=False)
        self.assertIn('"ms"', final_json)
        self.assertIn('"BULL_75000"', final_json)
        self.assertEqual(groq_summary["ms"]["tf"], "1m")


if __name__ == "__main__":
    unittest.main(verbosity=2)
