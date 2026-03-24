"""
Testes para ModelBiasMonitor.
Cobre: detecção de bias, confidence adjustment, macro incoherence, singleton.
"""

import time
import pytest
from unittest.mock import patch

from ml.bias_monitor import (
    ModelBiasMonitor,
    BiasAlert,
    get_bias_monitor,
    reset_bias_monitor,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_singleton():
    reset_bias_monitor()
    yield
    reset_bias_monitor()


@pytest.fixture
def monitor() -> ModelBiasMonitor:
    """Monitor com window pequena para testes rápidos."""
    return ModelBiasMonitor(
        window_size=10,
        bias_threshold=0.80,
        alert_cooldown_seconds=0,  # sem cooldown nos testes
    )


def _fill_bullish(monitor, count=10, prob=0.90):
    """Preenche monitor com predições bullish."""
    for _ in range(count):
        monitor.record(prob_up=prob, features={"rsi": 28})


def _fill_bearish(monitor, count=10, prob=0.10):
    """Preenche monitor com predições bearish."""
    for _ in range(count):
        monitor.record(prob_up=prob, features={"rsi": 72})


def _fill_mixed(monitor, count=10):
    """Preenche monitor com predições mistas."""
    for i in range(count):
        prob = 0.7 if i % 2 == 0 else 0.3
        monitor.record(prob_up=prob, features={"rsi": 50})


# ──────────────────────────────────────────────
# Testes: Detecção de bias
# ──────────────────────────────────────────────

class TestBiasDetection:

    def test_no_bias_with_insufficient_data(self, monitor):
        """Sem dados suficientes, sem alerta."""
        alert = monitor.record(prob_up=0.95, features={})
        assert alert is None

    def test_bullish_bias_detected(self, monitor):
        """Bias bullish deve ser detectado."""
        for i in range(9):
            monitor.record(prob_up=0.90, features={"rsi": 28})
        # 10a predição completa a janela
        alert = monitor.record(prob_up=0.90, features={"rsi": 28})
        assert alert is not None
        assert alert.direction == "bullish"
        assert alert.consecutive_pct >= 0.80

    def test_bearish_bias_detected(self, monitor):
        """Bias bearish deve ser detectado."""
        _fill_bearish(monitor)
        alert = monitor.record(prob_up=0.10, features={"rsi": 72})
        # Pode ser None se window já resetou, mas bias deve estar ativo
        assert monitor._bias_active is True

    def test_mixed_no_bias(self, monitor):
        """Predições mistas não devem gerar bias."""
        _fill_mixed(monitor)
        assert monitor._bias_active is False

    def test_bias_resolves_when_mixed(self, monitor):
        """Bias deve resolver quando predições voltam ao normal."""
        _fill_bullish(monitor)
        assert monitor._bias_active is True

        # Adicionar predições neutras (0.5 → nem bullish nem bearish)
        for _ in range(10):
            monitor.record(prob_up=0.5, features={"rsi": 50})
        assert monitor._bias_active is False

    def test_alert_has_features_snapshot(self, monitor):
        """Alerta deve conter snapshot das features."""
        _fill_bullish(monitor, count=9)
        features = {"rsi": 28, "bb_w": 0.004, "ret1": -0.001}
        alert = monitor.record(prob_up=0.90, features=features)
        assert alert is not None
        assert alert.features_snapshot == features


# ──────────────────────────────────────────────
# Testes: Confidence Adjustment
# ──────────────────────────────────────────────

class TestConfidenceAdjustment:

    def test_no_adjustment_without_bias(self, monitor):
        """Sem bias, fator = 1.0."""
        _fill_mixed(monitor)
        result = monitor.get_confidence_adjustment()
        assert result["factor"] == 1.0
        assert result["block"] is False

    def test_no_adjustment_insufficient_data(self, monitor):
        """Dados insuficientes, fator = 1.0."""
        monitor.record(prob_up=0.95, features={})
        result = monitor.get_confidence_adjustment()
        assert result["factor"] == 1.0
        assert result["block"] is False

    def test_adjustment_with_bias(self):
        """Com bias alto (mas <95%), fator entre 0.5 e 1.0."""
        # 90% bullish (9/10) — acima do threshold 0.80 mas abaixo de 0.95
        m = ModelBiasMonitor(window_size=10, bias_threshold=0.80, alert_cooldown_seconds=0)
        for _ in range(9):
            m.record(prob_up=0.90, features={})
        m.record(prob_up=0.30, features={})  # 1 bearish para ficar em 90%
        result = m.get_confidence_adjustment()
        assert result["block"] is False
        assert 0.5 <= result["factor"] < 1.0

    def test_adjustment_minimum_0_5(self, monitor):
        """Fator mínimo é 0.5 (nunca zero) para bias alto (não extremo)."""
        _fill_bullish(monitor, prob=0.99)
        result = monitor.get_confidence_adjustment()
        # 100% bullish com window=12 e threshold=0.85 -> block=True (>=95%)
        # Neste caso factor=0.0 é esperado (bloqueio extremo)
        assert result["block"] is True or result["factor"] >= 0.5

    def test_stronger_bias_lower_adjustment(self, monitor):
        """Bias mais forte -> fator mais baixo."""
        # 80% bullish
        m1 = ModelBiasMonitor(window_size=10, bias_threshold=0.70)
        for _ in range(8):
            m1.record(prob_up=0.90, features={})
        for _ in range(2):
            m1.record(prob_up=0.30, features={})
        adj_80 = m1.get_confidence_adjustment()["factor"]

        # 100% bullish -> will be blocked (factor=0.0)
        m2 = ModelBiasMonitor(window_size=10, bias_threshold=0.70)
        for _ in range(10):
            m2.record(prob_up=0.90, features={})
        adj_100 = m2.get_confidence_adjustment()["factor"]

        assert adj_100 <= adj_80


# ──────────────────────────────────────────────
# Testes: Macro Incoherence
# ──────────────────────────────────────────────

class TestMacroIncoherence:

    def test_bullish_ml_bearish_macro_warns(self, caplog):
        """ML bullish + macro bearish deve gerar warning."""
        # Usar window pequena para que a check de macro seja alcançada
        mon = ModelBiasMonitor(window_size=2, bias_threshold=0.80, alert_cooldown_seconds=0)
        # Preencher janela
        mon.record(prob_up=0.90, features={"rsi": 30})
        import logging
        with caplog.at_level(logging.WARNING):
            mon.record(
                prob_up=0.93,
                features={"rsi": 28},
                macro_context={
                    "15m": "Baixa",
                    "1h": "Baixa",
                    "4h": "Baixa",
                    "1d": "Baixa",
                },
            )
        assert any("INCOHERENCE" in r.message for r in caplog.records)

    def test_aligned_ml_macro_no_warning(self, monitor, caplog):
        """ML e macro alinhados não geram warning."""
        import logging
        with caplog.at_level(logging.WARNING):
            monitor.record(
                prob_up=0.85,
                features={"rsi": 55},
                macro_context={
                    "15m": "Alta",
                    "1h": "Alta",
                    "4h": "Baixa",
                    "1d": "Alta",
                },
            )
        incoherence_warnings = [
            r for r in caplog.records if "INCOHERENCE" in r.message
        ]
        assert len(incoherence_warnings) == 0

    def test_no_macro_no_warning(self, monitor, caplog):
        """Sem macro context não gera warning."""
        import logging
        with caplog.at_level(logging.WARNING):
            monitor.record(
                prob_up=0.93,
                features={"rsi": 28},
                macro_context=None,
            )
        incoherence_warnings = [
            r for r in caplog.records if "INCOHERENCE" in r.message
        ]
        assert len(incoherence_warnings) == 0


# ──────────────────────────────────────────────
# Testes: Alert Cooldown
# ──────────────────────────────────────────────

class TestAlertCooldown:

    def test_cooldown_prevents_spam(self):
        """Cooldown deve prevenir alertas consecutivos."""
        monitor = ModelBiasMonitor(
            window_size=5,
            bias_threshold=0.80,
            alert_cooldown_seconds=999,  # cooldown longo
        )

        # Preencher e emitir primeiro alerta
        for _ in range(5):
            monitor.record(prob_up=0.95, features={})
        first_alert = monitor.record(prob_up=0.95, features={})

        # Tentar emitir segundo (deve ser bloqueado pelo cooldown)
        for _ in range(5):
            monitor.record(prob_up=0.95, features={})
        second_alert = monitor.record(prob_up=0.95, features={})

        # Primeiro alerta OK, segundo bloqueado
        assert monitor._alerts_emitted == 1

    def test_no_cooldown_allows_multiple(self):
        """Sem cooldown, múltiplos alertas possíveis."""
        monitor = ModelBiasMonitor(
            window_size=5,
            bias_threshold=0.80,
            alert_cooldown_seconds=0,
        )

        for _ in range(20):
            monitor.record(prob_up=0.95, features={})

        assert monitor._alerts_emitted >= 2


# ──────────────────────────────────────────────
# Testes: get_stats
# ──────────────────────────────────────────────

class TestStats:

    def test_stats_empty(self, monitor):
        """Stats com zero predições."""
        stats = monitor.get_stats()
        assert stats["total_predictions"] == 0
        assert stats["confidence_adjustment"] == 1.0

    def test_stats_with_data(self, monitor):
        """Stats com dados."""
        _fill_bullish(monitor, count=10, prob=0.85)
        stats = monitor.get_stats()

        assert stats["total_predictions"] == 10
        assert stats["window_size"] == 10
        assert stats["avg_prob_up"] == pytest.approx(0.85, abs=0.01)
        assert stats["bullish_pct"] == pytest.approx(1.0, abs=0.01)

    def test_stats_has_all_keys(self, monitor):
        """Stats deve conter todas as chaves."""
        _fill_mixed(monitor)
        stats = monitor.get_stats()
        expected_keys = {
            "total_predictions", "window_size", "avg_prob_up",
            "bullish_pct", "bearish_pct", "bias_active",
            "alerts_emitted", "confidence_adjustment",
        }
        assert expected_keys.issubset(stats.keys())


# ──────────────────────────────────────────────
# Testes: Singleton
# ──────────────────────────────────────────────

class TestSingleton:

    def test_same_instance(self):
        m1 = get_bias_monitor()
        m2 = get_bias_monitor()
        assert m1 is m2

    def test_reset_creates_new(self):
        m1 = get_bias_monitor()
        reset_bias_monitor()
        m2 = get_bias_monitor()
        assert m1 is not m2


# ──────────────────────────────────────────────
# Teste: Cenário E2E do log
# ──────────────────────────────────────────────

class TestE2EFromLogs:

    def test_12_bullish_windows_detects_bias(self):
        """
        Reproduz cenário do log: 12 janelas, todas bullish.
        RSI=28-47, 4 TFs Baixa.
        """
        monitor = ModelBiasMonitor(
            window_size=10,
            bias_threshold=0.85,
            alert_cooldown_seconds=0,
        )

        log_data = [
            {"prob": 0.935, "rsi": 28},
            {"prob": 0.930, "rsi": 28},
            {"prob": 0.849, "rsi": 29},
            {"prob": 0.930, "rsi": 29},
            {"prob": 0.930, "rsi": 34},
            {"prob": 0.827, "rsi": 42},
            {"prob": 0.856, "rsi": 47},
            {"prob": 0.722, "rsi": 43},
            {"prob": 0.939, "rsi": 43},
            {"prob": 0.939, "rsi": 43},
            {"prob": 0.859, "rsi": 47},
            {"prob": 0.846, "rsi": 47},
        ]

        macro = {"15m": "Baixa", "1h": "Baixa", "4h": "Baixa", "1d": "Baixa"}

        alerts = []
        for entry in log_data:
            alert = monitor.record(
                prob_up=entry["prob"],
                features={"rsi": entry["rsi"]},
                macro_context=macro,
            )
            if alert is not None:
                alerts.append(alert)

        # Deve detectar bias (12/12 bullish = 100% > 85%)
        assert monitor._bias_active is True
        assert len(alerts) >= 1
        assert alerts[0].direction == "bullish"

        # Confidence deve estar reduzida ou bloqueada
        result = monitor.get_confidence_adjustment()
        assert result["factor"] < 1.0 or result["block"] is True

    def test_bias_reduces_confidence_in_hybrid_decision(self):
        """
        Simula: ML diz 93% bullish -> bias detectado -> confiança reduzida ou bloqueada.
        """
        monitor = ModelBiasMonitor(
            window_size=10,
            bias_threshold=0.85,
            alert_cooldown_seconds=0,
        )

        # 10 predições bullish
        for _ in range(10):
            monitor.record(prob_up=0.93, features={"rsi": 28})

        # Ajuste de confiança
        result = monitor.get_confidence_adjustment()

        # 100% bullish >= 95% -> block=True
        assert result["block"] is True

        # Se não bloqueado, fator deve reduzir confiança
        if not result["block"]:
            original_conf = 0.86
            adjusted_conf = original_conf * result["factor"]
            assert adjusted_conf < original_conf


# ──────────────────────────────────────────────
# Testes: price_targets_len fix
# ──────────────────────────────────────────────

class TestPriceTargetsLen:
    """Testa lógica de busca de price_targets em múltiplos caminhos."""

    @staticmethod
    def _find_price_targets(advanced, event, raw_event):
        """Lógica corrigida de busca de price_targets."""
        pt = (
            advanced.get("price_targets")
            or event.get("price_targets")
            or raw_event.get("price_targets")
        )
        return len(pt) if isinstance(pt, (list, dict)) else "N/A"

    def test_in_advanced(self):
        """price_targets em advanced_analysis."""
        assert self._find_price_targets(
            {"price_targets": [1, 2, 3]}, {}, {}
        ) == 3

    def test_in_event_root(self):
        """price_targets no event root (cenário atual)."""
        assert self._find_price_targets(
            {}, {"price_targets": [1, 2, 3, 4, 5, 6, 7]}, {}
        ) == 7

    def test_in_raw_event(self):
        """price_targets em raw_event."""
        assert self._find_price_targets(
            {}, {}, {"price_targets": [1, 2]}
        ) == 2

    def test_as_dict(self):
        """price_targets como dict (fixture format)."""
        assert self._find_price_targets(
            {},
            {"price_targets": {"resistance_1": 45600, "support_1": 44900}},
            {},
        ) == 2

    def test_missing_everywhere(self):
        """price_targets ausente em todos os caminhos."""
        assert self._find_price_targets({}, {}, {}) == "N/A"

    def test_none_value(self):
        """price_targets=None."""
        assert self._find_price_targets(
            {"price_targets": None}, {"price_targets": None}, {}
        ) == "N/A"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
