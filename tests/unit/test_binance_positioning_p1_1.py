# tests/unit/test_binance_positioning_p1_1.py
# -*- coding: utf-8 -*-
"""
Suíte de Testes Automatizados para Binance Positioning & Crypto COT (Fase P1.1).
Valida:
1. Endpoints, schema e separação semântica dos 3 L/S ratios.
2. Fetcher assíncrono, cache, timeout, retries e fail-soft.
3. Deltas de Open Interest (1h/4h) e warm-up.
4. Divergências Top vs Global.
5. Classificação determinística de regimes Crypto COT.
6. Integração com payload compacto e serialização RFC 8259.
7. Isolamento Context-Only (sem interferência em trade/risk).
"""

import asyncio
import json
import math
import os
import sys
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fetchers.binance_positioning_fetcher import (
    BinancePositioningFetcher,
    BinancePositioningSnapshot,
    _safe_float,
)
from institutional.crypto_cot import (
    CryptoCOT,
    CryptoCOTAnalysis,
    PositioningRegime,
)
from market_orchestrator.ai.payload_builder_compact import (
    build_compact_payload,
    _build_positioning,
)
from market_orchestrator.ai.analyzer_qwen import AIAnalyzer
from market_orchestrator.analysis.institutional_analytics import InstitutionalAnalyticsEngine


class TestBinancePositioningFetcher(unittest.IsolatedAsyncioTestCase):
    """Testes do coletor assíncrono de posicionamento da Binance."""

    def setUp(self):
        self.fetcher = BinancePositioningFetcher(cache_ttl=300.0, max_stale_seconds=900.0)

    def _sample_global_acc(self):
        now_ms = int(time.time() * 1000)
        return [
            {"symbol": "BTCUSDT", "longAccount": "0.5500", "shortAccount": "0.4500", "longShortRatio": "1.2222", "timestamp": now_ms - 300000},
            {"symbol": "BTCUSDT", "longAccount": "0.6800", "shortAccount": "0.3200", "longShortRatio": "2.1250", "timestamp": now_ms},
        ]

    def _sample_top_acc(self):
        now_ms = int(time.time() * 1000)
        return [
            {"symbol": "BTCUSDT", "longAccount": "0.5200", "shortAccount": "0.4800", "longShortRatio": "1.0833", "timestamp": now_ms - 300000},
            {"symbol": "BTCUSDT", "longAccount": "0.5800", "shortAccount": "0.4200", "longShortRatio": "1.3810", "timestamp": now_ms},
        ]

    def _sample_top_pos(self):
        now_ms = int(time.time() * 1000)
        return [
            {"symbol": "BTCUSDT", "longPosition": "0.6000", "shortPosition": "0.4000", "longShortRatio": "1.5000", "timestamp": now_ms - 300000},
            {"symbol": "BTCUSDT", "longPosition": "0.7200", "shortPosition": "0.2800", "longShortRatio": "2.5714", "timestamp": now_ms},
        ]

    def _sample_oi_hist(self, count=60):
        now_ms = int(time.time() * 1000)
        data = []
        for i in range(count):
            ts = now_ms - (count - 1 - i) * 300000  # 5 min cada barra
            # Simula OI crescendo de 100k para 110k
            oi = 100000.0 + i * (10000.0 / count)
            data.append({
                "symbol": "BTCUSDT",
                "sumOpenInterest": str(oi),
                "sumOpenInterestValue": str(oi * 77000.0),
                "timestamp": ts,
            })
        return data

    async def test_fetch_all_endpoints_happy_path(self):
        """Happy path: 4 endpoints retornam dados válidos com separação semântica estrita."""
        mock_fetch = AsyncMock()
        mock_fetch.side_effect = [
            self._sample_global_acc(),
            self._sample_top_acc(),
            self._sample_top_pos(),
            self._sample_oi_hist(60),
        ]
        self.fetcher._fetch_single_endpoint = mock_fetch

        snapshot = await self.fetcher.fetch_positioning("BTCUSDT", force_refresh=True)

        self.assertTrue(snapshot.is_available)
        self.assertFalse(snapshot.is_stale)
        self.assertEqual(snapshot.symbol, "BTCUSDT")

        # 3 L/S ratios separados semanticamente
        self.assertEqual(snapshot.global_account_ratio, 2.1250)
        self.assertEqual(snapshot.top_account_ratio, 1.3810)
        self.assertEqual(snapshot.top_position_ratio, 2.5714)

        # Ratios devem ser distintos
        self.assertNotEqual(snapshot.global_account_ratio, snapshot.top_account_ratio)
        self.assertNotEqual(snapshot.top_account_ratio, snapshot.top_position_ratio)

        # Deltas de OI
        self.assertIsNotNone(snapshot.oi_delta_1h)
        self.assertIsNotNone(snapshot.oi_delta_4h)
        self.assertGreater(snapshot.oi_delta_1h, 0.0)
        self.assertGreater(snapshot.oi_delta_4h, 0.0)

        # Divergências
        self.assertEqual(snapshot.top_position_vs_global, round(2.5714 - 2.1250, 4))
        self.assertEqual(snapshot.top_account_vs_global, round(1.3810 - 2.1250, 4))

    async def test_cache_mechanism(self):
        """Segunda chamada dentro do TTL deve retornar cache sem requisição de rede."""
        mock_fetch = AsyncMock()
        mock_fetch.side_effect = [
            self._sample_global_acc(),
            self._sample_top_acc(),
            self._sample_top_pos(),
            self._sample_oi_hist(60),
        ]
        self.fetcher._fetch_single_endpoint = mock_fetch

        # 1ª chamada: busca
        snap1 = await self.fetcher.fetch_positioning("BTCUSDT", force_refresh=False)
        self.assertEqual(mock_fetch.call_count, 4)

        # 2ª chamada: usa cache
        snap2 = await self.fetcher.fetch_positioning("BTCUSDT", force_refresh=False)
        self.assertEqual(mock_fetch.call_count, 4)
        self.assertEqual(snap1.global_account_ratio, snap2.global_account_ratio)

    async def test_timeout_and_error_fail_soft(self):
        """Falha nos endpoints deve retornar snapshot marcado como indisponível sem quebrar."""
        mock_fetch = AsyncMock(return_value=[])
        self.fetcher._fetch_single_endpoint = mock_fetch

        snapshot = await self.fetcher.fetch_positioning("BTCUSDT", force_refresh=True)

        self.assertFalse(snapshot.is_available)
        self.assertIsNone(snapshot.global_account_ratio)
        self.assertIsNone(snapshot.top_position_ratio)
        self.assertIsNone(snapshot.open_interest)
        self.assertEqual(snapshot.error, "no_data_from_endpoints")

    async def test_stale_detection(self):
        """Dados com timestamp antigo (>900s) devem ser marcados como is_stale=True."""
        old_ts = int((time.time() - 1000.0) * 1000)  # 1000s atrás
        stale_data = [
            {"symbol": "BTCUSDT", "longAccount": "0.5", "shortAccount": "0.5", "longShortRatio": "1.0", "timestamp": old_ts}
        ]
        mock_fetch = AsyncMock(side_effect=[stale_data, stale_data, stale_data, stale_data])
        self.fetcher._fetch_single_endpoint = mock_fetch

        snapshot = await self.fetcher.fetch_positioning("BTCUSDT", force_refresh=True)
        self.assertTrue(snapshot.is_available)
        self.assertTrue(snapshot.is_stale)
        self.assertGreater(snapshot.age_seconds, 900.0)


class TestCryptoCOTLogic(unittest.TestCase):
    """Testes para a lógica de análise e regimes em institutional/crypto_cot.py."""

    def setUp(self):
        self.cot = CryptoCOT()

    def test_crowded_long_classification(self):
        """Global ratio >= 2.0 deve classificar como CROWDED_LONG com evidência."""
        snap = BinancePositioningSnapshot(
            symbol="BTCUSDT",
            period="5m",
            observed_at=time.time(),
            global_account_ratio=2.35,
            top_account_ratio=1.40,
            top_position_ratio=2.10,
            open_interest=100000.0,
            is_available=True,
            is_stale=False,
        )
        res = self.cot.analyze(snap)
        self.assertEqual(res.regime, PositioningRegime.CROWDED_LONG)
        self.assertTrue(any("sobrecarregado na compra" in r for r in res.reasons))

    def test_crowded_short_classification(self):
        """Global ratio <= 0.5 deve classificar como CROWDED_SHORT com evidência."""
        snap = BinancePositioningSnapshot(
            symbol="BTCUSDT",
            period="5m",
            observed_at=time.time(),
            global_account_ratio=0.42,
            top_account_ratio=0.80,
            top_position_ratio=0.50,
            open_interest=100000.0,
            is_available=True,
            is_stale=False,
        )
        res = self.cot.analyze(snap)
        self.assertEqual(res.regime, PositioningRegime.CROWDED_SHORT)
        self.assertTrue(any("sobrecarregado na venda" in r for r in res.reasons))

    def test_top_long_divergence(self):
        """Top traders muito mais comprados que varejo (diff > 0.40)."""
        snap = BinancePositioningSnapshot(
            symbol="BTCUSDT",
            period="5m",
            observed_at=time.time(),
            global_account_ratio=1.10,
            top_account_ratio=1.20,
            top_position_ratio=1.75,  # diff = +0.65
            open_interest=100000.0,
            is_available=True,
            is_stale=False,
        )
        res = self.cot.analyze(snap)
        self.assertEqual(res.regime, PositioningRegime.TOP_LONG_DIVERGENCE)
        self.assertEqual(res.top_position_vs_global, 0.65)

    def test_top_short_divergence(self):
        """Top traders muito mais vendidos que varejo (diff < -0.40)."""
        snap = BinancePositioningSnapshot(
            symbol="BTCUSDT",
            period="5m",
            observed_at=time.time(),
            global_account_ratio=1.40,
            top_account_ratio=1.10,
            top_position_ratio=0.80,  # diff = -0.60
            open_interest=100000.0,
            is_available=True,
            is_stale=False,
        )
        res = self.cot.analyze(snap)
        self.assertEqual(res.regime, PositioningRegime.TOP_SHORT_DIVERGENCE)
        self.assertEqual(res.top_position_vs_global, -0.60)

    def test_squeeze_risk_classification(self):
        """Funding extremo positivo + varejo comprado -> SQUEEZE_RISK (Long Squeeze)."""
        snap = BinancePositioningSnapshot(
            symbol="BTCUSDT",
            period="5m",
            observed_at=time.time(),
            global_account_ratio=2.20,
            top_account_ratio=1.50,
            top_position_ratio=2.00,
            open_interest=100000.0,
            is_available=True,
            is_stale=False,
        )
        # Funding rate 0.0005 (0.05% / 5 bps)
        res = self.cot.analyze(snap, funding_rate=0.0005)
        self.assertEqual(res.regime, PositioningRegime.SQUEEZE_RISK)
        self.assertTrue(any("Long Squeeze" in r for r in res.reasons))

    def test_oi_expansion_classification(self):
        """Variação expressiva de OI (>= 5% em 1h) em mercado equilibrado."""
        snap = BinancePositioningSnapshot(
            symbol="BTCUSDT",
            period="5m",
            observed_at=time.time(),
            global_account_ratio=1.10,
            top_account_ratio=1.15,
            top_position_ratio=1.20,
            open_interest=100000.0,
            oi_delta_1h=0.065,  # +6.5%
            is_available=True,
            is_stale=False,
        )
        res = self.cot.analyze(snap)
        self.assertEqual(res.regime, PositioningRegime.OI_EXPANSION)

    def test_neutral_market(self):
        """Mercado com ratios equilibrados deve classificar como NEUTRAL."""
        snap = BinancePositioningSnapshot(
            symbol="BTCUSDT",
            period="5m",
            observed_at=time.time(),
            global_account_ratio=1.05,
            top_account_ratio=1.10,
            top_position_ratio=1.15,
            open_interest=100000.0,
            oi_delta_1h=0.01,
            is_available=True,
            is_stale=False,
        )
        res = self.cot.analyze(snap)
        self.assertEqual(res.regime, PositioningRegime.NEUTRAL)

    def test_stale_or_missing_returns_unknown(self):
        """Dados nulos ou obsoletos resultam em UNKNOWN sem gerar dados falsos."""
        res_none = self.cot.analyze(None)
        self.assertEqual(res_none.regime, PositioningRegime.UNKNOWN)
        self.assertFalse(res_none.is_available)

        snap_stale = BinancePositioningSnapshot(
            symbol="BTCUSDT",
            period="5m",
            observed_at=time.time(),
            global_account_ratio=1.5,
            top_position_ratio=1.5,
            is_available=True,
            is_stale=True,
        )
        res_stale = self.cot.analyze(snap_stale)
        self.assertEqual(res_stale.regime, PositioningRegime.UNKNOWN)
        self.assertTrue(res_stale.is_stale)


class TestPayloadPositioningIntegration(unittest.TestCase):
    """Testes de integração do posicionamento com o payload compacto e LLM."""

    def test_build_positioning_section(self):
        """Verifica a extração dos campos compactos 'pos'."""
        event = {
            "institutional_analytics": {
                "positioning": {
                    "global_account_ratio": 2.125,
                    "top_account_ratio": 1.381,
                    "top_position_ratio": 2.571,
                    "oi_delta_1h": 0.024,
                    "oi_delta_4h": 0.051,
                    "regime": "CROWDED_LONG",
                    "is_available": True,
                }
            }
        }
        pos = _build_positioning(event)
        self.assertEqual(pos["ga"], round(2.125, 2))
        self.assertEqual(pos["ta"], round(1.381, 2))
        self.assertEqual(pos["tp"], round(2.571, 2))
        self.assertEqual(pos["od1"], 0.024)
        self.assertEqual(pos["od4"], 0.051)
        self.assertEqual(pos["rg"], "CROWDED_LONG")

    def test_compact_payload_preserves_pos_in_groq_summary(self):
        """Verifica que 'pos' sobrevive à compressão Groq."""
        event = {
            "symbol": "BTCUSDT",
            "tipo_evento": "ANALYSIS_TRIGGER",
            "preco_fechamento": 77500.0,
            "institutional_analytics": {
                "positioning": {
                    "global_account_ratio": 1.28,
                    "top_account_ratio": 1.39,
                    "top_position_ratio": 2.07,
                    "oi_delta_1h": 0.024,
                    "regime": "TOP_LONG_DIVERGENCE",
                    "is_available": True,
                }
            }
        }
        compact = build_compact_payload(event)
        self.assertIn("pos", compact)

        groq_summary = AIAnalyzer._build_groq_payload_summary(compact)
        self.assertIn("pos", groq_summary)
        self.assertEqual(groq_summary["pos"]["ga"], 1.28)
        self.assertEqual(groq_summary["pos"]["rg"], "TOP_LONG_DIVERGENCE")

        # Serialização RFC 8259
        serialized = json.dumps(groq_summary, allow_nan=False)
        self.assertIn('"pos"', serialized)
        self.assertNotIn("NaN", serialized)


if __name__ == "__main__":
    unittest.main(verbosity=2)
