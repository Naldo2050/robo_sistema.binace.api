"""
Testes unitários para a integração das RegimeBasedRules.
"""
import pytest
from src.rules.regime_rules import RegimeBasedRules, TradeRecommendation


def test_regime_rules_initialization():
    """Testa a inicialização das RegimeBasedRules."""
    regime_rules = RegimeBasedRules()
    assert regime_rules is not None
    assert regime_rules.base_position_size == 1.0
    assert regime_rules.base_stop_pct == 0.02
    assert regime_rules.base_target_pct == 0.04


def test_get_regime_adjustment_risk_on_low_vol():
    """Testa ajustes para RISK_ON + LOW_VOL."""
    regime_rules = RegimeBasedRules()
    regime_analysis = {
        "market_regime": "RISK_ON",
        "volatility_regime": "LOW_VOL",
        "correlation_regime": "MACRO_CORRELATED",
        "risk_score": 0.5,
        "fear_greed_proxy": 0.4,
        "regime_change_warning": False,
        "divergence_alert": False,
        "primary_driver": "MIXED_SIGNALS"
    }
    adjustment = regime_rules.get_regime_adjustment(regime_analysis)
    assert adjustment.position_size_multiplier == 1.25
    assert adjustment.stop_loss_multiplier == 0.8
    assert adjustment.take_profit_multiplier == 1.2
    assert adjustment.min_confidence_required == 0.55
    assert adjustment.allowed_directions == ["long", "short"]
    assert adjustment.aggressive_entries is True
    assert adjustment.scale_in_allowed is True
    assert adjustment.scale_out_required is False
    assert adjustment.extra_confirmation_needed is False
    assert adjustment.regime_warning is None


def test_get_regime_adjustment_risk_off_high_vol():
    """Testa ajustes para RISK_OFF + HIGH_VOL."""
    regime_rules = RegimeBasedRules()
    regime_analysis = {
        "market_regime": "RISK_OFF",
        "volatility_regime": "HIGH_VOL",
        "correlation_regime": "MACRO_CORRELATED",
        "risk_score": 0.3,
        "fear_greed_proxy": 0.2,
        "regime_change_warning": False,
        "divergence_alert": False,
        "primary_driver": "MIXED_SIGNALS"
    }
    adjustment = regime_rules.get_regime_adjustment(regime_analysis)
    assert adjustment.position_size_multiplier == 0.5
    assert adjustment.stop_loss_multiplier == 1.5
    assert adjustment.take_profit_multiplier == 0.8
    assert adjustment.min_confidence_required == 0.75
    assert adjustment.allowed_directions == ["short"]
    assert adjustment.aggressive_entries is False
    assert adjustment.scale_in_allowed is False
    assert adjustment.scale_out_required is True
    assert adjustment.extra_confirmation_needed is True
    assert adjustment.regime_warning == "⚠️ RISK_OFF + HIGH_VOL: Operar com extrema cautela"


def test_should_trade_blocked_by_regime():
    """Testa se o trade é bloqueado pelo regime."""
    regime_rules = RegimeBasedRules()
    regime_analysis = {
        "market_regime": "RISK_OFF",
        "volatility_regime": "HIGH_VOL",
        "correlation_regime": "MACRO_CORRELATED",
        "risk_score": 0.3,
        "fear_greed_proxy": 0.2,
        "regime_change_warning": False,
        "divergence_alert": False,
        "primary_driver": "MIXED_SIGNALS"
    }
    should_trade, reason = regime_rules.should_trade(
        regime_analysis=regime_analysis,
        signal_direction="long",
        signal_confidence=0.6
    )
    assert should_trade is False
    assert "Direção long não permitida no regime atual" in reason


def test_should_trade_allowed():
    """Testa se o trade é permitido pelo regime."""
    regime_rules = RegimeBasedRules()
    regime_analysis = {
        "market_regime": "RISK_ON",
        "volatility_regime": "LOW_VOL",
        "correlation_regime": "MACRO_CORRELATED",
        "risk_score": 0.5,
        "fear_greed_proxy": 0.4,
        "regime_change_warning": False,
        "divergence_alert": False,
        "primary_driver": "MIXED_SIGNALS"
    }
    should_trade, reason = regime_rules.should_trade(
        regime_analysis=regime_analysis,
        signal_direction="long",
        signal_confidence=0.6
    )
    assert should_trade is True
    assert reason == "OK"


def test_calculate_position_size():
    """Testa o cálculo do tamanho da posição ajustado pelo regime."""
    regime_rules = RegimeBasedRules()
    regime_analysis = {
        "market_regime": "RISK_ON",
        "volatility_regime": "LOW_VOL",
        "correlation_regime": "MACRO_CORRELATED",
        "risk_score": 0.5,
        "fear_greed_proxy": 0.4,
        "regime_change_warning": False,
        "divergence_alert": False,
        "primary_driver": "MIXED_SIGNALS"
    }
    position_size = regime_rules.calculate_position_size(
        regime_analysis=regime_analysis,
        base_size=1.0,
        account_balance=10000.0
    )
    assert position_size == 1.25


def test_calculate_stop_loss():
    """Testa o cálculo do stop loss ajustado pelo regime."""
    regime_rules = RegimeBasedRules()
    regime_analysis = {
        "market_regime": "RISK_ON",
        "volatility_regime": "LOW_VOL",
        "correlation_regime": "MACRO_CORRELATED",
        "risk_score": 0.5,
        "fear_greed_proxy": 0.4,
        "regime_change_warning": False,
        "divergence_alert": False,
        "primary_driver": "MIXED_SIGNALS"
    }
    stop_loss = regime_rules.calculate_stop_loss(
        regime_analysis=regime_analysis,
        entry_price=100.0,
        direction="long",
        atr=1.0
    )
    assert stop_loss == 100.0 - (1.5 * 0.8)


def test_calculate_take_profit():
    """Testa o cálculo do take profit ajustado pelo regime."""
    regime_rules = RegimeBasedRules()
    regime_analysis = {
        "market_regime": "RISK_ON",
        "volatility_regime": "LOW_VOL",
        "correlation_regime": "MACRO_CORRELATED",
        "risk_score": 0.5,
        "fear_greed_proxy": 0.4,
        "regime_change_warning": False,
        "divergence_alert": False,
        "primary_driver": "MIXED_SIGNALS"
    }
    take_profit = regime_rules.calculate_take_profit(
        regime_analysis=regime_analysis,
        entry_price=100.0,
        direction="long",
        atr=1.0
    )
    assert take_profit == 100.0 + (3.0 * 1.2)


def test_get_trade_recommendation():
    """Testa a recomendação de trading baseada no regime."""
    regime_rules = RegimeBasedRules()
    regime_analysis = {
        "market_regime": "RISK_ON",
        "volatility_regime": "LOW_VOL",
        "correlation_regime": "MACRO_CORRELATED",
        "risk_score": 0.5,
        "fear_greed_proxy": 0.4,
        "regime_change_warning": False,
        "divergence_alert": False,
        "primary_driver": "MIXED_SIGNALS"
    }
    recommendation = regime_rules.get_trade_recommendation(
        regime_analysis=regime_analysis,
        flow_signal="bullish",
        technical_signal="bullish",
        confidence=0.8
    )
    assert recommendation == TradeRecommendation.STRONG_LONG


def test_format_regime_summary():
    """Testa a formatação do resumo do regime."""
    regime_rules = RegimeBasedRules()
    regime_analysis = {
        "market_regime": "RISK_ON",
        "volatility_regime": "LOW_VOL",
        "correlation_regime": "MACRO_CORRELATED",
        "risk_score": 0.5,
        "fear_greed_proxy": 0.4,
        "regime_change_warning": False,
        "divergence_alert": False,
        "primary_driver": "MIXED_SIGNALS"
    }
    summary = regime_rules.format_regime_summary(regime_analysis)
    assert "RISK_ON" in summary
    assert "LOW_VOL" in summary
    assert "MACRO_CORRELATED" in summary
    assert "Position Size:" in summary
    assert "Stop Multiplier:" in summary
    assert "Target Multiplier:" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])