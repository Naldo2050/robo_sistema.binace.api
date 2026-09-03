# tests/payload/test_positioning_provenance_p1_1b.py
# -*- coding: utf-8 -*-
"""
Teste de Proveniência Ponta-a-Ponta e Integridade de Dados — Fase P1.1B.
Valida toda a cadeia:
Mock Binance HTTP -> Fetcher -> Canonical Model -> CryptoCOT ->
InstitutionalAnalyticsEngine -> Orchestrator Data -> Compact Payload ->
Groq Summary -> Guardrail -> Final JSON RFC 8259.

Asserções Obrigatórias:
1. Sem perda semântica entre as camadas.
2. Sem mudança de unidade ou escala espúria (sem *100 ou /100 indevido).
3. Sem stringificação de números (od1/od4 numéricos canônicos).
4. Sem duplicação de Funding Rate em 'pos' (Funding apenas em price.fr).
5. Sem NaN, Inf ou booleans como números.
6. Sem defaults falsos (ausência não vira 0.0 ou 1.0 ou NEUTRAL).
7. Dados obsoletos (stale) classificados estritamente como UNKNOWN.
"""

import json
import math
import os
import sys
import time
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, ".")

from fetchers.binance_positioning_fetcher import BinancePositioningFetcher, BinancePositioningSnapshot, _safe_float
from institutional.crypto_cot import CryptoCOT, CryptoCOTAnalysis, PositioningRegime
from market_orchestrator.analysis.institutional_analytics import InstitutionalAnalyticsEngine
from market_orchestrator.ai.payload_builder_compact import build_compact_payload, _build_positioning
from market_orchestrator.ai.analyzer_qwen import AIAnalyzer


class TestPositioningProvenanceP1_1B(unittest.IsolatedAsyncioTestCase):
    """Suíte de validação de proveniência e integridade semântica ponta-a-ponta."""

    def setUp(self):
        self.fetcher = BinancePositioningFetcher(cache_ttl=300.0, max_stale_seconds=900.0)
        self.engine = InstitutionalAnalyticsEngine("BTCUSDT")

    def _sample_mock_responses(self, now_ms=None):
        if now_ms is None:
            now_ms = int(time.time() * 1000)
            
        g_acc = [
            {"symbol": "BTCUSDT", "longAccount": "0.5616", "shortAccount": "0.4384", "longShortRatio": "1.2810", "timestamp": now_ms}
        ]
        t_acc = [
            {"symbol": "BTCUSDT", "longAccount": "0.5811", "shortAccount": "0.4189", "longShortRatio": "1.3872", "timestamp": now_ms}
        ]
        t_pos = [
            {"symbol": "BTCUSDT", "longPosition": "0.6742", "shortPosition": "0.3258", "longShortRatio": "2.0695", "timestamp": now_ms}
        ]
        
        # Histórico de OI com 60 barras (5m cada)
        oi_hist = []
        for i in range(60):
            ts = now_ms - (59 - i) * 300000
            # OI inicial 100k -> 1h atrás 102k -> atual 105k
            base_oi = 100000.0 + (i * 84.74)
            oi_hist.append({
                "symbol": "BTCUSDT",
                "sumOpenInterest": str(base_oi),
                "sumOpenInterestValue": str(base_oi * 77500.0),
                "timestamp": ts,
            })
        return g_acc, t_acc, t_pos, oi_hist

    async def test_end_to_end_provenance_pipeline(self):
        """Cadeia completa: Mock -> Fetcher -> COT -> Engine -> Payload -> Groq -> JSON RFC 8259."""
        now_ms = int(time.time() * 1000)
        g_acc, t_acc, t_pos, oi_hist = self._sample_mock_responses(now_ms)
        
        # 1. Mock Fetcher
        mock_fetch = AsyncMock(side_effect=[g_acc, t_acc, t_pos, oi_hist])
        self.fetcher._fetch_single_endpoint = mock_fetch
        
        snapshot = await self.fetcher.fetch_positioning("BTCUSDT", force_refresh=True)
        self.assertTrue(snapshot.is_available)
        self.assertFalse(snapshot.is_stale)
        self.assertEqual(snapshot.global_account_ratio, 1.2810)
        self.assertEqual(snapshot.top_account_ratio, 1.3872)
        self.assertEqual(snapshot.top_position_ratio, 2.0695)
        self.assertIsInstance(snapshot.oi_delta_1h, float)
        self.assertIsInstance(snapshot.oi_delta_4h, float)

        # 2. Crypto COT Layer
        cot = CryptoCOT()
        canonical_funding = 0.0001  # 0.01% / 1 bp
        analysis = cot.analyze(snapshot, funding_rate=canonical_funding)
        self.assertEqual(analysis.regime, PositioningRegime.TOP_LONG_DIVERGENCE)
        self.assertTrue(any("Top Traders" in r for r in analysis.reasons))

        # 3. InstitutionalAnalyticsEngine
        inst_res = self.engine.compute_all(
            current_price=77500.0,
            positioning_data=snapshot.to_dict(),
            derivatives_data={"BTCUSDT": {"funding_rate": canonical_funding}}
        )
        self.assertIn("positioning", inst_res)
        self.assertEqual(inst_res["positioning"]["regime"], "TOP_LONG_DIVERGENCE")

        # 4. Orchestrator Signal to Compact Payload
        signal_event = {
            "symbol": "BTCUSDT",
            "tipo_evento": "ANALYSIS_TRIGGER",
            "preco_fechamento": 77500.0,
            "derivatives": {"BTCUSDT": {"funding_rate": canonical_funding}},
            "institutional_analytics": inst_res,
        }
        compact_payload = build_compact_payload(signal_event)
        
        # 5. Validação de Seções no Payload Compacto
        self.assertIn("pos", compact_payload)
        self.assertIn("price", compact_payload)
        
        pos_sec = compact_payload["pos"]
        self.assertEqual(pos_sec["ga"], 1.28)
        self.assertEqual(pos_sec["ta"], 1.39)
        self.assertEqual(pos_sec["tp"], 2.07)
        self.assertIsInstance(pos_sec["od1"], float)
        self.assertIsInstance(pos_sec["od4"], float)
        self.assertEqual(pos_sec["rg"], "TOP_LONG_DIVERGENCE")
        
        # Asserção de Não-Duplicação de Funding Rate
        self.assertEqual(compact_payload["price"]["fr"], 0.0001)
        self.assertNotIn("fr", pos_sec)
        self.assertNotIn("funding_rate", pos_sec)

        # 6. Groq Summary Compressor
        groq_summary = AIAnalyzer._build_groq_payload_summary(compact_payload)
        self.assertIn("pos", groq_summary)
        self.assertEqual(groq_summary["pos"]["ga"], 1.28)
        self.assertEqual(groq_summary["pos"]["tp"], 2.07)
        self.assertEqual(groq_summary["p"]["fr"], 0.0001)

        # 7. Serialização Estrita JSON RFC 8259 (sem NaN, Inf ou strings inválidas)
        final_json = json.dumps(groq_summary, allow_nan=False, ensure_ascii=False)
        self.assertNotIn("NaN", final_json)
        self.assertIn('"pos"', final_json)
        self.assertIn('"fr"', final_json)
        # Recarregar para garantir conformidade RFC 8259
        reparsed = json.loads(final_json)
        self.assertEqual(reparsed["p"]["fr"], 0.0001)
        self.assertEqual(reparsed["pos"]["ga"], 1.28)
        self.assertEqual(reparsed["pos"]["tp"], 2.07)

    def test_stale_data_fails_soft_to_unknown(self):
        """Dados obsoletos (>900s) devem produzir regime UNKNOWN e não propagar ao LLM como dados ativos."""
        old_ts = int((time.time() - 1200.0) * 1000)
        stale_snap = BinancePositioningSnapshot(
            symbol="BTCUSDT",
            period="5m",
            observed_at=time.time(),
            global_account_ratio=1.5,
            top_position_ratio=1.5,
            source_timestamp=old_ts,
            age_seconds=1200.0,
            is_stale=True,
            is_available=True,
        )
        cot = CryptoCOT()
        analysis = cot.analyze(stale_snap)
        self.assertEqual(analysis.regime, PositioningRegime.UNKNOWN)
        self.assertTrue(analysis.is_stale)
        
        # Payload builder deve ignorar dados quando is_available=False ou is_stale sem dados válidos
        event = {
            "institutional_analytics": {
                "positioning": {
                    "is_available": False,
                    "is_stale": True,
                    "regime": "UNKNOWN"
                }
            }
        }
        pos = _build_positioning(event)
        self.assertEqual(pos, {})

    def test_numeric_edge_cases_safety(self):
        """Valida que NaN, Inf, None e Booleans não poluem a seção 'pos'."""
        event_dirty = {
            "institutional_analytics": {
                "positioning": {
                    "is_available": True,
                    "global_account_ratio": float("nan"),
                    "top_account_ratio": float("inf"),
                    "top_position_ratio": True,  # booleano
                    "oi_delta_1h": None,
                    "oi_delta_4h": 0.0,  # zero legítimo
                    "regime": "NEUTRAL",  # regime neutro não deve poluir chave rg
                }
            }
        }
        pos = _build_positioning(event_dirty)
        self.assertNotIn("ga", pos)
        self.assertNotIn("ta", pos)
        self.assertNotIn("tp", pos)
        self.assertNotIn("od1", pos)
        self.assertNotIn("rg", pos)  # NEUTRAL é suprimido para economizar tokens
        self.assertEqual(pos.get("od4"), 0.0)  # zero float legítimo é preservado


if __name__ == "__main__":
    unittest.main(verbosity=2)
