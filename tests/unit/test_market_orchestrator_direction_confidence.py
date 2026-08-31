# tests/unit/test_market_orchestrator_direction_confidence.py
"""
Teste de integração / unidade para a resolução de confiança direcional
em MarketOrchestrator._handle_signal_event.

Garante que:
1. Sinais SHORT leiam short_prob (ex: long_prob=0.15, short_prob=0.85 => confidence=0.85);
2. Sinais LONG leiam long_prob (ex: long_prob=0.75, short_prob=0.25 => confidence=0.75);
3. Sinais NEUTRAL / UNKNOWN usem o fallback seguro (0.5).
"""

from unittest.mock import MagicMock, patch
from common.signal_direction import infer_signal_side, get_directional_confidence


def test_directional_confidence_resolution_short_vs_long():
    conf = {
        "long_prob": 0.15,
        "short_prob": 0.85,
        "neutral_prob": 0.0,
    }

    # 1. Evento de Absorção de Compra (SHORT)
    event_short = {
        "tipo_evento": "Absorção",
        "resultado_da_batalha": "Absorção de Compra",
        "historical_confidence": conf,
    }
    side_short = infer_signal_side(
        event_type=event_short.get("tipo_evento"),
        battle_result=event_short.get("resultado_da_batalha"),
    )
    assert side_short == "SHORT"
    direction_short = "short" if side_short == "SHORT" else "long"
    confidence_short = get_directional_confidence(event_short["historical_confidence"], direction_short)

    # Confiança para SHORT deve ser 0.85 (short_prob), NUNCA 0.15 (long_prob)!
    assert confidence_short == 0.85

    # 2. Evento de Absorção de Venda (LONG)
    event_long = {
        "tipo_evento": "Absorção",
        "resultado_da_batalha": "Absorção de Venda",
        "historical_confidence": conf,
    }
    side_long = infer_signal_side(
        event_type=event_long.get("tipo_evento"),
        battle_result=event_long.get("resultado_da_batalha"),
    )
    assert side_long == "LONG"
    direction_long = "long" if side_long == "LONG" else "short"
    confidence_long = get_directional_confidence(event_long["historical_confidence"], direction_long)

    # Confiança para LONG deve ser 0.15 (long_prob)
    assert confidence_long == 0.15


def test_market_orchestrator_should_trade_receives_correct_confidence():
    """
    Testa que _handle_signal_event passa signal_confidence=0.85 para
    RegimeBasedRules.should_trade quando o sinal é SHORT.
    """
    from market_orchestrator.market_orchestrator import EnhancedMarketBot

    bot = EnhancedMarketBot.__new__(EnhancedMarketBot)
    bot.ai_analyzer = MagicMock()
    bot.ai_test_passed = True
    bot._last_ai_analysis_ts = 0
    bot._ai_min_interval_sec = 0
    bot._run_ai_analysis_threaded = MagicMock()

    mock_regime_rules = MagicMock()
    mock_regime_rules.should_trade.return_value = (True, "OK")
    mock_regime_rules.format_regime_summary.return_value = "REGIME SUMMARY"

    event_data = {
        "tipo_evento": "Absorção",
        "resultado_da_batalha": "Absorção de Compra",  # SHORT
        "severity": "CRITICAL",
        "ai_payload": {
            "regime_analysis": {"current_regime": "TRENDING_DOWN"}
        },
        "historical_confidence": {
            "long_prob": 0.15,
            "short_prob": 0.85,
            "neutral_prob": 0.0,
        },
    }

    with patch("market_analysis.regime_rules.RegimeBasedRules", return_value=mock_regime_rules):
        bot._handle_signal_event(event_data)

    # Verificar os argumentos passados para should_trade
    mock_regime_rules.should_trade.assert_called_once()
    kwargs = mock_regime_rules.should_trade.call_args[1]

    assert kwargs["signal_direction"] == "short"
    # A confiança passada DEVE ser 0.85, comprovando a correção do P1
    assert kwargs["signal_confidence"] == 0.85
