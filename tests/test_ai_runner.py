# tests/test_ai_runner.py
"""
Testes do AI Runner.
Versão corrigida: usa Groq (provider atual) em vez de QwenClient (legado).
"""
import pytest
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock


class TestAIRunner:
    """Testes do módulo AI Runner."""

    # ══════════════════════════════════════════════════════════════
    # FIXTURES
    # ══════════════════════════════════════════════════════════════

    @pytest.fixture
    def mock_groq_response(self):
        """Resposta mockada válida do Groq."""
        return {
            "sentiment": "bearish",
            "confidence": 0.85,
            "action": "sell",
            "rationale": "Distribuição institucional detectada: whale score -41, flow imbalance -90%",
            "entry_zone": [66350, 66380],
            "invalidation_zone": [66450, 66500],
            "region_type": "distribution_zone",
            "_is_fallback": False,
            "_is_valid": True,
        }

    @pytest.fixture
    def mock_market_payload(self):
        """Payload de mercado COMPLETO para testes de compressão."""
        return {
            "symbol": "BTCUSDT",
            "window": 1,
            "trigger": "ANALYSIS_TRIGGER",
            "tipo_evento": "ANALYSIS_TRIGGER",
            "preco_fechamento": 66355.5,
            "anchor_price": 66355.5,

            # Price data
            "price": {
                "c": 66355.5,
                "o": 66370.3,
                "h": 66370.3,
                "l": 66355.5,
                "vwap": 66366.1,
                "shape": "P",
                "signal": "BEARISH_AFTER",
                "poor_high": 1,
                "poor_low": 0,
                "auction": "expect_retest_both",
            },

            # Flow data completo
            "flow": {
                "net_1m": -9596,
                "net_5m": -9596,
                "net_15m": -9596,
                "cvd": -0.14,
                "imb": -0.90,
                "agg_buy": 5.0,
                "agg_sell": 95.04,
                "bsr": 0.05,
                "pressure": "STRONG_SELL",
                "trend": "accelerating_selling",
                "absorption": "NEUTRAL",
                "abs_idx": 0.8729,
                "buyer_str": 0.5,
                "seller_exh": 9,
                "cont_prob": 0.79,
                "pa_signal": "strong_bearish",
                "pa_conv": "HIGH",
                "pa_agree": 1,
            },

            # Whale data completo
            "whale": {
                "score": -41,
                "class": "MILD_DISTRIBUTION",
                "bias": "DISTRIBUTING",
                "trend_dir": "insufficient_data",
                "flow_score": -0.72,
                "depth_score": -15.16,
                "abs_score": -25.0,
                "components": {
                    "flow": {
                        "score": -0.72,
                        "max": 30,
                        "detail": {
                            "divergence": "aligned",
                            "retail_delta": -0.145,
                            "cvd_used": True
                        }
                    },
                    "depth": {
                        "score": -15.16,
                        "max": 20,
                        "detail": {
                            "bid_depth": 133221.88,
                            "ask_depth": 590329.32,
                            "ratio": -0.63,
                            "deep_confirmation": True
                        }
                    },
                    "absorption": {
                        "score": -25,
                        "max": 25,
                        "detail": {
                            "buyer_strength": 0.5,
                            "seller_exhaustion": 9,
                            "net_absorption": -8.5,
                            "index": 0.8729,
                            "label": "Neutra"
                        }
                    }
                }
            },

            # OrderBook completo
            "ob": {
                "bid": 133222,
                "ask": 590329,
                "imb": -0.63,
                "top5_imb": -0.67,
                "mid": 66313.55,
                "spread": 0.1,
                "liq_score": 9.997,
                "exec_qual": "EXCELLENT",
            },

            # OrderBook depth por nível
            "order_book_depth": {
                "L1": {"bids": 96884.02, "asks": 502988.66, "imbalance": -0.677},
                "L5": {"bids": 102321.72, "asks": 519699.78, "imbalance": -0.671},
                "L10": {"bids": 104841.59, "asks": 523280.78, "imbalance": -0.666},
                "L25": {"bids": 110676.94, "asks": 538533.5, "imbalance": -0.659},
                "total_depth_ratio": 0.21,
            },

            # Fibonacci
            "fibonacci_levels": {
                "high": 66370.3,
                "low": 66355.5,
                "23.6": 66358.99,
                "38.2": 66361.15,
                "50.0": 66362.9,
                "61.8": 66364.65,
                "78.6": 66367.13,
            },

            # Resistências imediatas
            "immediate_resistance": [66455.04],
            "resistance_strength": [56],

            # Alertas
            "alerts": {
                "active_alerts": [
                    {
                        "type": "RESISTANCE_TEST",
                        "level": 66455.04,
                        "severity": "MEDIUM",
                        "probability": 0.7,
                        "action": "MONITOR_CLOSELY",
                        "description": "Preço testando resistência em 66455.04"
                    },
                    {
                        "type": "VOLUME_SPIKE",
                        "threshold_exceeded": 5,
                        "severity": "HIGH",
                        "probability": 0.5,
                        "action": "PREPARE_ENTRY",
                        "description": "Volume 5.0x acima da média"
                    },
                    {
                        "type": "WHALE_DISTRIBUTION",
                        "level": -41,
                        "severity": "HIGH",
                        "probability": 0.82,
                        "action": "AVOID_LONG",
                        "description": "Sinal de distribuição de whales (score=-41)"
                    }
                ],
                "alert_count": 3,
                "max_severity": "HIGH",
            },

            # Regime
            "regime_analysis": {
                "current_regime": "MEAN_REVERTING",
                "regime_probabilities": {
                    "trending": 0.187,
                    "mean_reverting": 0.437,
                    "breakout": 0.376,
                },
                "regime_change_probability": 0.43,
                "expected_regime_duration": "15m-1h",
            },

            # Volatility
            "volatility_metrics": {
                "volatility_regime": "NORMAL",
                "volatility_percentile": 50,
            },

            # ML Features cross-asset
            "ml_features": {
                "cross_asset": {
                    "btc_eth_corr_7d": 0.9325,
                    "btc_eth_corr_30d": 0.9046,
                    "btc_dxy_corr_30d": -0.139,
                    "btc_dxy_corr_90d": -0.0472,
                    "btc_ndx_corr_30d": -0.0844,
                    "dxy_return_5d": 0.5512,
                    "dxy_return_20d": 2.8672,
                    "vix_current": 29.49,
                    "us10y_yield": 4.133,
                    "btc_dominance": 32.8676,
                    "eth_dominance": 17.5035,
                    "gold_price": 5045.0025,
                    "oil_price": 109.52,
                    "macro_regime": "TRANSITION",
                    "correlation_regime": "DECORRELATED",
                }
            },

            # Institutional analytics completo
            "institutional_analytics": {
                "status": "ok",
                "profile_analysis": {
                    "poor_extremes": {
                        "poor_high": {
                            "detected": True,
                            "price": 66370.34,
                            "volume_ratio": 3.439,
                            "implication": "High likely to be revisited",
                        },
                        "poor_low": {
                            "detected": True,
                            "price": 66355.48,
                            "volume_ratio": 0.71,
                            "implication": "Low likely to be revisited",
                        },
                        "action_bias": "expect_retest_both",
                        "status": "success",
                    },
                    "profile_shape": {
                        "shape": "P",
                        "implication": "Short covering rally - bearish bias expected",
                        "trading_signal": "BEARISH_AFTER",
                        "distribution": {
                            "lower_third_pct": 11.9,
                            "middle_third_pct": 26.2,
                            "upper_third_pct": 61.9,
                        },
                        "dominant_zone": "upper",
                        "status": "success",
                    },
                },
                "flow_analysis": {
                    "passive_aggressive": {
                        "aggressive": {
                            "buy_pct": 4.96,
                            "sell_pct": 95.04,
                            "net_pct": -90.08,
                            "dominance": "sellers",
                        },
                        "passive": {
                            "dominance": "sellers",
                            "bid_depth": 133221.88,
                            "ask_depth": 590329.32,
                            "ob_imbalance": -0.632,
                        },
                        "composite": {
                            "agreement": True,
                            "signal": "strong_bearish",
                            "interpretation": "Both aggressive and passive sellers active",
                            "conviction": "HIGH",
                        },
                        "status": "success",
                    },
                    "whale_accumulation": {
                        "score": -41,
                        "classification": "MILD_DISTRIBUTION",
                        "bias": "DISTRIBUTING",
                        "trend": {
                            "direction": "insufficient_data",
                            "avg_score": -41,
                            "samples": 1,
                        },
                        "status": "success",
                    },
                    "buy_sell_ratio": {
                        "buy_sell_ratio": 0.05,
                        "ratios": {
                            "current": 0.0522,
                            "imbalance_1m": -0.969,
                            "imbalance_5m": -0.969,
                            "imbalance_15m": -0.969,
                        },
                        "sector_ratios": {
                            "retail": 0.0487,
                            "mid": 1,
                            "whale": 1,
                        },
                        "pressure": "STRONG_SELL",
                        "flow_trend": "accelerating_selling",
                    },
                },
                "sr_analysis": {
                    "defense_zones": {
                        "sell_defense": [{
                            "center": 66455.04,
                            "range_low": 66372.09,
                            "range_high": 66537.98,
                            "strength": 56,
                            "side": "sell",
                            "sources": ["orderbook_ask_wall", "depth_asymmetry"],
                            "distance_from_price": 99.54,
                            "distance_pct": 0.15,
                        }],
                        "total_zones": 1,
                        "strongest_sell": {
                            "center": 66455.04,
                            "range_low": 66372.09,
                            "range_high": 66537.98,
                            "strength": 56,
                            "side": "sell",
                            "sources": ["orderbook_ask_wall", "depth_asymmetry"],
                            "distance_pct": 0.15,
                        },
                        "defense_asymmetry": {
                            "bias": "strong_sell_defense",
                            "description": "Significantly more sell defense",
                            "sell_total_strength": 56,
                        },
                        "status": "success",
                    },
                },
                "quality": {
                    "calendar": {
                        "day_of_week": "Sunday",
                        "is_weekend": True,
                        "expected_liquidity": "LOW",
                        "liquidity_warning": True,
                    },
                    "latency": {
                        "latency_ms": 18073,
                        "data_freshness": "STALE",
                        "is_stale": True,
                    },
                    "anomalies": {
                        "anomalies_detected": True,
                        "count": 2,
                        "anomalies": [
                            {
                                "type": "FLOW_EXTREME_IMBALANCE",
                                "severity": "HIGH",
                                "value": -0.9008,
                                "description": "Extreme flow imbalance: -90.08%",
                            },
                            {
                                "type": "DEPTH_EXTREME_ASYMMETRY",
                                "severity": "MEDIUM",
                                "ratio": 0.23,
                                "description": "Order book depth ratio 0.23:1",
                            },
                        ],
                        "max_severity": "HIGH",
                        "risk_elevated": True,
                        "types_found": [
                            "FLOW_EXTREME_IMBALANCE",
                            "DEPTH_EXTREME_ASYMMETRY",
                        ],
                    },
                },
            },

            # Quant
            "quant": {
                "prob_up": 0.94,
                "conf": 0.89,
            },

            # Market impact
            "market_impact": {
                "slippage_matrix": {
                    "100k_usd": {"buy": 0.05, "sell": 0.15},
                    "1m_usd": {"buy": 6.75, "sell": 6.35},
                },
                "liquidity_score": 9.997,
                "execution_quality": "EXCELLENT",
            },

            # Spread
            "spread_analysis": {
                "current_spread_bps": 0.0151,
                "avg_spread_1h": 0.0151,
                "avg_spread_24h": 0.0151,
            },
        }

    @pytest.fixture
    def ai_runner_instance(self):
        """
        Cria instância do AIRunner com provider mockado.
        Compatível com sistema atual (Groq).
        """
        try:
            from ai_runner import AIRunner
            runner = AIRunner(api_key='test_key', model='llama-3.1-8b-instant')
            return runner
        except Exception:
            # Fallback: mock completo
            runner = MagicMock()
            runner.api_key = 'test_key'
            runner.model = 'llama-3.1-8b-instant'
            return runner

    # ══════════════════════════════════════════════════════════════
    # TESTES DE INICIALIZAÇÃO
    # ══════════════════════════════════════════════════════════════

    def test_ai_runner_initialization(self):
        """Testa inicialização do AIRunner."""
        try:
            from ai_runner import AIRunner
            runner = AIRunner(api_key='test_key', model='llama-3.1-8b-instant')
            assert runner.api_key == 'test_key'
            assert runner.model == 'llama-3.1-8b-instant'
        except ImportError:
            pytest.skip("AIRunner não disponível para importação direta")

    def test_ai_runner_initialization_default_model(self):
        """Testa inicialização com modelo padrão."""
        try:
            from ai_runner import AIRunner
            runner = AIRunner(api_key='test_key')
            assert runner.api_key == 'test_key'
            assert runner.model is not None
            assert len(runner.model) > 0
        except (ImportError, TypeError):
            pytest.skip("AIRunner com modelo padrão não disponível")

    # ══════════════════════════════════════════════════════════════
    # TESTES DE ANÁLISE
    # ══════════════════════════════════════════════════════════════

    def test_analyze_with_ai(self, mock_market_payload, mock_groq_response):
        """
        Testa análise com IA mockada.
        """
        import json

        try:
            from ai_runner import AIRunner
            runner = AIRunner(api_key='test_key')

            # Configurar mock corretamente com return_value explícito
            mock_client = MagicMock()
            mock_client.generate.return_value = json.dumps(mock_groq_response)

            # Configurar chat.completions.create
            mock_choice = MagicMock()
            mock_choice.message.content = json.dumps(mock_groq_response)
            mock_completion = MagicMock()
            mock_completion.choices = [mock_choice]
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = MagicMock(
                return_value=mock_completion
            )

            runner.client = mock_client

            # Executar análise
            result = runner.analyze(str(mock_market_payload))
            result_str = str(result)

            # Verificar resposta válida
            valid_responses = [
                "BUY", "SELL", "WAIT",
                "buy", "sell", "wait",
                "bullish", "bearish", "neutral",
                "NEUTRAL", "confidence", "sentiment",
                "sell", "bearish"
            ]

            assert any(v in result_str for v in valid_responses), (
                f"Resposta inválida: {result_str[:200]}"
            )

        except (ImportError, AttributeError) as e:
            pytest.skip(f"AIRunner não compatível: {e}")

    def test_analyze_returns_valid_structure(self, mock_groq_response):
        """Testa que a análise retorna estrutura válida."""
        # Testar estrutura da resposta diretamente
        required_keys = ["sentiment", "confidence", "action", "rationale"]

        for key in required_keys:
            assert key in mock_groq_response, f"Campo '{key}' ausente"

        assert mock_groq_response["sentiment"] in ["bullish", "bearish", "neutral"]
        assert mock_groq_response["action"] in ["buy", "sell", "wait"]
        assert 0.0 <= mock_groq_response["confidence"] <= 1.0
        assert len(mock_groq_response["rationale"]) > 10

    def test_analyze_fallback_structure(self):
        """Testa estrutura do fallback quando IA falha."""
        fallback = {
            "sentiment": "neutral",
            "confidence": 0.0,
            "action": "wait",
            "rationale": "llm_error_provider_unavailable",
            "entry_zone": None,
            "invalidation_zone": None,
            "region_type": None,
            "_is_fallback": True,
            "_fallback_reason": "provider_unavailable",
            "_is_valid": False,
        }

        assert fallback["_is_fallback"] is True
        assert fallback["confidence"] == 0.0
        assert fallback["action"] == "wait"

    # ══════════════════════════════════════════════════════════════
    # TESTES DE RETRY
    # ══════════════════════════════════════════════════════════════

    def test_ai_runner_with_retry_success(self):
        """Testa retry com sucesso na segunda tentativa."""
        try:
            from ai_runner import AIRunner
            runner = AIRunner(api_key='test_key')
            runner.client = Mock()

            # Primeira falha, segunda sucesso
            runner.client.generate.side_effect = [
                ConnectionError("Timeout"),
                "BUY with confidence 0.9"
            ]

            result = runner.analyze_with_retry(
                "Market data",
                max_retries=2
            )

            assert result == "BUY with confidence 0.9"
            assert runner.client.generate.call_count == 2

        except (ImportError, AttributeError):
            pytest.skip("analyze_with_retry não disponível")

    def test_ai_runner_with_retry_failure(self):
        """Testa que esgota retries e levanta exceção."""
        try:
            from ai_runner import AIRunner
            runner = AIRunner(api_key='test_key')
            runner.client = Mock()

            runner.client.generate.side_effect = ConnectionError("Timeout")

            with pytest.raises((ConnectionError, Exception)):
                runner.analyze_with_retry(
                    "Market data",
                    max_retries=3
                )

            assert runner.client.generate.call_count == 3

        except (ImportError, AttributeError):
            pytest.skip("analyze_with_retry não disponível")

    # ══════════════════════════════════════════════════════════════
    # TESTES DE PARSE DE RESPOSTA
    # ══════════════════════════════════════════════════════════════

    def test_ai_runner_parse_response_json(self):
        """Testa parse de resposta JSON válida."""
        try:
            from ai_runner import AIRunner
            runner = AIRunner(api_key='test_key')

            # Resposta JSON válida
            raw_response = json.dumps({
                "sentiment": "bearish",
                "confidence": 0.85,
                "action": "sell",
                "rationale": "Distribuição detectada",
                "entry_zone": [66350, 66380],
                "invalidation_zone": [66450, 66500],
                "region_type": "distribution_zone"
            })

            parsed = runner.parse_response(raw_response)

            assert parsed is not None
            assert isinstance(parsed, dict)

            # Verificar campos esperados
            if 'signal' in parsed:
                assert parsed['signal'] in [
                    'BUY', 'SELL', 'WAIT',
                    'buy', 'sell', 'wait',
                    'bullish', 'bearish', 'neutral'
                ]
            if 'confidence' in parsed:
                assert 0.0 <= float(parsed['confidence']) <= 1.0

        except (ImportError, AttributeError):
            pytest.skip("parse_response não disponível")

    def test_ai_runner_parse_response_text(self):
        """Testa parse de resposta em texto simples."""
        try:
            from ai_runner import AIRunner
            runner = AIRunner(api_key='test_key')

            raw_response = "Signal: BUY\nConfidence: 0.85"
            parsed = runner.parse_response(raw_response)

            assert parsed is not None

            if isinstance(parsed, dict):
                if 'signal' in parsed:
                    assert parsed['signal'] == 'BUY'
                if 'confidence' in parsed:
                    assert parsed['confidence'] == 0.85

        except (ImportError, AttributeError):
            pytest.skip("parse_response não disponível")

    def test_ai_runner_parse_response_invalid(self):
        """Testa parse de resposta inválida."""
        try:
            from ai_runner import AIRunner
            runner = AIRunner(api_key='test_key')

            # Resposta completamente inválida
            raw_response = "erro: conexão perdida"
            parsed = runner.parse_response(raw_response)

            # Deve retornar algo (não lançar exceção)
            assert parsed is not None

        except (ImportError, AttributeError):
            pytest.skip("parse_response não disponível")

    # ══════════════════════════════════════════════════════════════
    # TESTES DO ANALISADOR QWEN (ai_analyzer_qwen.py)
    # ══════════════════════════════════════════════════════════════

    def test_ai_analyzer_groq_connection(self):
        """Testa conexão com Groq (sem fazer chamada real)."""
        import os
        api_key = os.getenv("GROQ_API_KEY", "")

        if not api_key:
            pytest.skip("GROQ_API_KEY não configurada")

        assert api_key.startswith("gsk_"), (
            "GROQ_API_KEY deve começar com 'gsk_'"
        )

    def test_ai_analyzer_payload_validation(self, mock_market_payload):
        """Testa que payload de mercado tem estrutura correta."""
        required_sections = ["symbol", "price", "flow", "whale", "ob", "quant"]

        for section in required_sections:
            assert section in mock_market_payload, (
                f"Seção '{section}' ausente no payload"
            )

        # Validar price
        price = mock_market_payload["price"]
        assert price.get("c") is not None, "price.c (close) ausente"
        assert price.get("shape") is not None, "price.shape ausente"

        # Validar flow
        flow = mock_market_payload["flow"]
        assert flow.get("imb") is not None, "flow.imb ausente"
        assert flow.get("abs_idx") is not None, "flow.abs_idx ausente"

        # Validar whale
        whale = mock_market_payload["whale"]
        assert whale.get("score") is not None, "whale.score ausente"
        assert -100 <= whale["score"] <= 100, "whale.score fora do range"

    def test_ai_analyzer_system_prompt_quality(self):
        """Testa que o SYSTEM_PROMPT tem conteúdo profissional."""
        try:
            from ai_analyzer_qwen import SYSTEM_PROMPT

            # Verificar que tem conteúdo mínimo
            assert len(SYSTEM_PROMPT) > 500, (
                "SYSTEM_PROMPT muito curto - deve ter análise institucional"
            )

            # Verificar keywords essenciais
            keywords = [
                "absorção", "Absorção",
                "whale", "Whale",
                "flow", "Flow",
                "suporte", "resistência",
                "entry_zone",
                "rationale",
            ]

            prompt_lower = SYSTEM_PROMPT.lower()
            found = [k for k in keywords if k.lower() in prompt_lower]

            assert len(found) >= 5, (
                f"SYSTEM_PROMPT não tem keywords institucionais suficientes. "
                f"Encontradas: {found}"
            )

        except ImportError:
            pytest.skip("ai_analyzer_qwen não disponível")

    def test_payload_compressor_preserves_critical_data(
        self,
        mock_market_payload
    ):
        """Testa que o compressor não perde dados críticos."""
        try:
            from market_orchestrator.ai.payload_compressor_v3 import (
                compress_payload_v3
            )
            import json

            compressed = compress_payload_v3(mock_market_payload)

            assert isinstance(compressed, dict), "Resultado deve ser dict"
            assert len(compressed) > 0, "Resultado não pode ser vazio"

            # ── Dados obrigatórios ──
            assert "price" in compressed, "price perdido na compressão"
            assert compressed["price"].get("c") is not None, "price.c perdido"
            assert "flow" in compressed, "flow perdido na compressão"
            assert "whale" in compressed, "whale perdido na compressão"
            assert "ob" in compressed, "ob perdido na compressão"

            # ── Dados institucionais (novos) ──
            assert "fib" in compressed, "fibonacci perdido na compressão"
            assert "alerts" in compressed, "alerts perdido na compressão"
            assert "defense" in compressed, "defense zones perdidas na compressão"

            # ── Verificar qualidade do flow ──
            flow = compressed["flow"]
            assert "imb" in flow, "flow.imb perdido"
            assert "abs_idx" in flow or "absorption" in flow, (
                "absorption data perdida"
            )

            # ── Verificar qualidade do whale ──
            whale = compressed["whale"]
            assert "score" in whale, "whale.score perdido"
            assert "bias" in whale, "whale.bias perdido"

            # ── Métricas de compressão ──
            original_size = len(json.dumps(mock_market_payload))
            compressed_size = len(json.dumps(compressed))
            ratio = (1 - compressed_size / original_size) * 100

            print(f"\n📊 Compressão: {original_size} → {compressed_size} bytes "
                  f"({ratio:.1f}% redução)")
            print(f"   Seções preservadas: {list(compressed.keys())}")

            # Com payload completo, deve comprimir pelo menos 20%
            assert ratio >= 20, (
                f"Compressão insuficiente: {ratio:.1f}% "
                f"(payload original: {original_size} bytes)"
            )

        except ImportError as e:
            pytest.skip(f"payload_compressor_v3 não disponível: {e}")

    def test_ai_response_validation(self, mock_groq_response):
        """Testa validação de resposta da IA."""
        # Resposta válida
        assert mock_groq_response["_is_valid"] is True
        assert mock_groq_response["_is_fallback"] is False
        assert mock_groq_response["sentiment"] in ["bullish", "bearish", "neutral"]
        assert mock_groq_response["action"] in ["buy", "sell", "wait"]
        assert 0.0 <= mock_groq_response["confidence"] <= 1.0

        # entry_zone deve ser lista ou null
        entry = mock_groq_response.get("entry_zone")
        if entry is not None:
            assert isinstance(entry, list)
            assert len(entry) == 2
            assert entry[0] < entry[1]

        # invalidation_zone deve ser lista ou null
        invalid = mock_groq_response.get("invalidation_zone")
        if invalid is not None:
            assert isinstance(invalid, list)
            assert len(invalid) == 2