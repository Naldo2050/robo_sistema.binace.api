# tests/payload/test_funding_rate_pipeline_p0.py
# -*- coding: utf-8 -*-
"""
Testes de regressão e contrato para Funding Rate e integridade do payload (Fase P0).
Valida a sobrevivência do Funding Rate canônico (fração decimal) através de:
builder -> compressor -> guardrail -> JSON RFC 8259 final.
"""

import json
import math
import os
import sys
import unittest
from typing import Any, Dict

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from market_orchestrator.ai.payload_builder_compact import (
    build_compact_payload,
    _extract_canonical_funding_rate,
)
from market_orchestrator.ai.analyzer_qwen import AIAnalyzer


class TestFundingRatePipelineP0(unittest.TestCase):
    """Testes unitários e de integração para Funding Rate na Fase P0."""

    def _make_base_event(self) -> Dict[str, Any]:
        """Gera um evento base válido para testes de payload."""
        return {
            "symbol": "BTCUSDT",
            "tipo_evento": "ANALYSIS_TRIGGER",
            "resultado_da_batalha": "COMPRA",
            "epoch_ms": 1788309780000,
            "preco_fechamento": 77500.0,
            "contextual_snapshot": {
                "ohlc": {"open": 77450.0, "high": 77600.0, "low": 77400.0, "close": 77500.0, "vwap": 77480.0}
            },
            "window_metrics": {
                "total_volume": 120.0,
                "buy_volume": 80.0,
                "sell_volume": 40.0,
                "delta": 40.0,
                "trade_count": 1500,
                "flow_imbalance": 0.33,
            },
            "orderbook_data": {
                "bids": [[77490.0, 10.0]],
                "asks": [[77510.0, 8.0]],
                "bid_depth_usd": 500000.0,
                "ask_depth_usd": 400000.0,
                "imbalance": 0.11,
                "spread_bps": 2.5,
                "data_source": "live",
            },
            "support_resistance": {
                "poc": 77450.0,
                "val": 77300.0,
                "vah": 77600.0,
            },
            "regime_analysis": {
                "regime": "TRENDING",
                "probabilities": {"trending": 0.8, "mean_reverting": 0.1, "breakout": 0.1},
                "adx": 35.0,
            },
            "derivatives": {
                "BTCUSDT": {}
            },
        }

    # ============================================================
    # TESTES DE EXTRAÇÃO CANÔNICA (_extract_canonical_funding_rate)
    # ============================================================

    def test_funding_positivo_fracao_direta(self):
        """Funding positivo em fração decimal direta (ex: 0.0001 = 0.01%)."""
        event = self._make_base_event()
        event["derivatives"]["BTCUSDT"]["funding_rate"] = 0.0001
        
        fr = _extract_canonical_funding_rate(event)
        self.assertEqual(fr, 0.0001)

        compact = build_compact_payload(event)
        self.assertEqual(compact.get("price", {}).get("fr"), 0.0001)

        groq_summary = AIAnalyzer._build_groq_payload_summary(compact)
        self.assertEqual(groq_summary.get("p", {}).get("fr"), 0.0001)

    def test_funding_negativo_fracao_direta(self):
        """Funding negativo em fração decimal direta (ex: -0.00025 = -0.025%)."""
        event = self._make_base_event()
        event["derivatives"]["BTCUSDT"]["funding_rate"] = -0.00025
        
        fr = _extract_canonical_funding_rate(event)
        self.assertEqual(fr, -0.00025)

        compact = build_compact_payload(event)
        self.assertEqual(compact.get("price", {}).get("fr"), -0.00025)

        groq_summary = AIAnalyzer._build_groq_payload_summary(compact)
        self.assertEqual(groq_summary.get("p", {}).get("fr"), -0.00025)

    def test_funding_zero(self):
        """Funding zero exato (0.0) deve ser preservado e não tratado como ausente."""
        event = self._make_base_event()
        event["derivatives"]["BTCUSDT"]["funding_rate"] = 0.0
        
        fr = _extract_canonical_funding_rate(event)
        self.assertEqual(fr, 0.0)

        compact = build_compact_payload(event)
        self.assertEqual(compact.get("price", {}).get("fr"), 0.0)

        groq_summary = AIAnalyzer._build_groq_payload_summary(compact)
        self.assertEqual(groq_summary.get("p", {}).get("fr"), 0.0)

    def test_funding_converte_de_percentual(self):
        """Funding vindo como percentual (0.01 representando 0.01%) deve virar 0.0001."""
        event = self._make_base_event()
        event["derivatives"]["BTCUSDT"]["funding_rate_percent"] = 0.01
        
        fr = _extract_canonical_funding_rate(event)
        self.assertEqual(fr, 0.0001)

        compact = build_compact_payload(event)
        self.assertEqual(compact.get("price", {}).get("fr"), 0.0001)

        groq_summary = AIAnalyzer._build_groq_payload_summary(compact)
        self.assertEqual(groq_summary.get("p", {}).get("fr"), 0.0001)

    def test_funding_none_ausente(self):
        """Funding ausente não deve injetar 'fr' na seção de preço."""
        event = self._make_base_event()
        # Sem dados de derivativos
        event["derivatives"] = {}

        fr = _extract_canonical_funding_rate(event)
        self.assertIsNone(fr)

        compact = build_compact_payload(event)
        self.assertNotIn("fr", compact.get("price", {}))

        groq_summary = AIAnalyzer._build_groq_payload_summary(compact)
        self.assertNotIn("fr", groq_summary.get("p", {}))

    def test_funding_extremo_valido(self):
        """Valores nos limites válidos (+0.05 e -0.05) devem ser aceitos."""
        event_max = self._make_base_event()
        event_max["derivatives"]["BTCUSDT"]["funding_rate"] = 0.05
        self.assertEqual(_extract_canonical_funding_rate(event_max), 0.05)

        event_min = self._make_base_event()
        event_min["derivatives"]["BTCUSDT"]["funding_rate"] = -0.05
        self.assertEqual(_extract_canonical_funding_rate(event_min), -0.05)

    def test_funding_extremo_invalido_rejeitado(self):
        """Valores fora dos limites (ex: 0.10 = 10%) devem ser rejeitados (None)."""
        event = self._make_base_event()
        event["derivatives"]["BTCUSDT"]["funding_rate"] = 0.10
        self.assertIsNone(_extract_canonical_funding_rate(event))

    def test_funding_non_finite_rejeitado(self):
        """NaN, +Inf e -Inf devem ser rejeitados de forma segura."""
        event_nan = self._make_base_event()
        event_nan["derivatives"]["BTCUSDT"]["funding_rate"] = float("nan")
        self.assertIsNone(_extract_canonical_funding_rate(event_nan))

        event_inf = self._make_base_event()
        event_inf["derivatives"]["BTCUSDT"]["funding_rate"] = float("inf")
        self.assertIsNone(_extract_canonical_funding_rate(event_inf))

        event_ninf = self._make_base_event()
        event_ninf["derivatives"]["BTCUSDT"]["funding_rate"] = float("-inf")
        self.assertIsNone(_extract_canonical_funding_rate(event_ninf))

    def test_funding_boolean_rejeitado(self):
        """Booleanos (True/False) não podem ser interpretados como 1.0 ou 0.0."""
        event = self._make_base_event()
        event["derivatives"]["BTCUSDT"]["funding_rate"] = True
        self.assertIsNone(_extract_canonical_funding_rate(event))

    # ============================================================
    # TESTE END-TO-END DO PAYLOAD E SERIALIZAÇÃO RFC 8259
    # ============================================================

    def test_e2e_payload_pipeline_and_rfc8259_serialization(self):
        """Valida que o payload final com Funding é serializável em JSON RFC 8259 estrito."""
        event = self._make_base_event()
        event["derivatives"]["BTCUSDT"]["funding_rate"] = 0.0001
        event["context_collector"] = {
            "derivatives": {"open_interest": 125000.0}
        }

        # 1. Builder
        compact = build_compact_payload(event)
        self.assertIn("price", compact)
        self.assertEqual(compact["price"]["fr"], 0.0001)

        # 2. Compressor Groq
        groq_summary = AIAnalyzer._build_groq_payload_summary(compact)
        self.assertIn("p", groq_summary)
        self.assertEqual(groq_summary["p"]["fr"], 0.0001)

        # 3. Serialização RFC 8259
        serialized_json = json.dumps(groq_summary, allow_nan=False)
        self.assertIsInstance(serialized_json, str)
        self.assertNotIn("NaN", serialized_json)
        self.assertNotIn("Infinity", serialized_json)

        # 4. Deserialização idêntica
        parsed = json.loads(serialized_json)
        self.assertEqual(parsed["p"]["fr"], 0.0001)


if __name__ == "__main__":
    unittest.main(verbosity=2)
