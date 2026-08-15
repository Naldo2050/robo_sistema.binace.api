# tests/unit/test_volatility_alert_type.py
"""
Testes do contrato de tipo do alerta de volatilidade (alert_engine).

Cobrem o fix: estado EXPANDED deve emitir type="VOLATILITY_EXPANSION"
e estado COMPRESSED deve emitir type="VOLATILITY_SQUEEZE".

Todos os testes chamam o caminho REAL de alert_engine
(detect_volatility_squeeze / generate_alerts), sem fakes de detector.

Contrato de cooldown (market_orchestrator.py L2098-2105):
o bot usa alert.get("type") como chave do cooldown — portanto os dois
tipos devem ser chaves distintas para não se confundirem.
"""

import os.path
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from trading.alert_engine import (
    detect_volatility_squeeze,
    generate_alerts,
    format_alert_message,
)

# Série de volatilidades que ativa os dois ramos do detector:
# 9 pontos baixos + ponto alto => low_thresh ≈ base, high_thresh ≈ topo.
SQUEEZE_VOLS = [1e-5] * 9 + [1.0]
EXPANSION_VOLS = [1e-5] * 9 + [1.0]


class TestVolatilityAlertType:
    """Contrato de tipo por estado."""

    def test_a_expanded_emits_volatility_expansion(self):
        """Estado EXPANDED -> type VOLATILITY_EXPANSION (caminho real)."""
        alert = detect_volatility_squeeze(
            current_vol=2.0, recent_vols=EXPANSION_VOLS
        )
        assert alert is not None
        assert alert["type"] == "VOLATILITY_EXPANSION"
        assert alert["volatility_state"] == "EXPANDED"

    def test_b_compressed_emits_volatility_squeeze(self):
        """Estado COMPRESSED -> type VOLATILITY_SQUEEZE (caminho real)."""
        alert = detect_volatility_squeeze(
            current_vol=1e-6, recent_vols=SQUEEZE_VOLS
        )
        assert alert is not None
        assert alert["type"] == "VOLATILITY_SQUEEZE"
        assert alert["volatility_state"] == "COMPRESSED"

    def test_c_description_and_message_consistent_with_type(self):
        """Descrição e mensagem formatada coerentes com o type."""
        exp = detect_volatility_squeeze(
            current_vol=2.0, recent_vols=EXPANSION_VOLS
        )
        sqz = detect_volatility_squeeze(
            current_vol=1e-6, recent_vols=SQUEEZE_VOLS
        )

        assert exp["description"].startswith("EXPANSION:")
        assert sqz["description"].startswith("SQUEEZE:")

        msg_exp = format_alert_message(exp)
        msg_sqz = format_alert_message(sqz)

        # EXPANSION não pode cair no fallback genérico ("⚠️ Alerta:")
        assert "VOLATILITY EXPANSION" in msg_exp
        assert not msg_exp.startswith("⚠️ Alerta:")
        assert "VOLATILITY SQUEEZE" in msg_sqz

    def test_d_numeric_fields_unchanged(self):
        """
        Campos numéricos/comportamento não são alterados pelo fix de type:
        intensidade, probabilidade, severity e ação seguem as fórmulas
        originais do detector.
        """
        exp = detect_volatility_squeeze(
            current_vol=2.0, recent_vols=EXPANSION_VOLS
        )
        sqz = detect_volatility_squeeze(
            current_vol=1e-6, recent_vols=SQUEEZE_VOLS
        )

        # EXPANSION: intensity=(2.0-1.0)/1.0=1.0, prob=0.4+0.55*1.0, CRITICAL
        assert exp["intensity"] == pytest.approx(1.0, abs=1e-9)
        assert exp["probability"] == pytest.approx(0.95, abs=1e-9)
        assert exp["severity"] == "CRITICAL"
        assert exp["action"] == "EXPECT_REVERSION_IMMINENT"
        assert exp["volatility_current"] == pytest.approx(2.0)
        # high_thresh real de numpy (percentil 90 com interpolação linear)
        assert exp["volatility_threshold"] == pytest.approx(0.100009, abs=1e-5)

        # SQUEEZE: intensity=1-(1e-6/1e-5)=0.9, prob=0.4+0.55*0.9, CRITICAL
        assert sqz["intensity"] == pytest.approx(0.9, abs=1e-6)
        assert sqz["probability"] == pytest.approx(0.895, abs=1e-6)
        assert sqz["severity"] == "CRITICAL"
        assert sqz["action"] == "PREPARE_FOR_BREAKOUT_IMMINENT"
        assert sqz["volatility_current"] == pytest.approx(1e-6)
        assert sqz["volatility_threshold"] == pytest.approx(1e-5, abs=1e-9)

    def test_e_cooldown_contract_separates_types(self):
        """
        Contrato do cooldown (market_orchestrator.py L2098-2105):
        a chave é alert.get("type"). Como os tipos agora diferem,
        expansion e squeeze possuem chaves (e cooldowns) separados.
        """
        exp = detect_volatility_squeeze(
            current_vol=2.0, recent_vols=EXPANSION_VOLS
        )
        sqz = detect_volatility_squeeze(
            current_vol=1e-6, recent_vols=SQUEEZE_VOLS
        )

        # Espelha a lógica real do bot:
        #   atype = alert.get("type", "GENERIC")
        #   if now - last_ts.get(atype, 0.0) < cooldown: skip
        #   last_ts[atype] = now
        def bot_cooldown_check(alert, last_ts, cooldown_sec, now):
            atype = alert.get("type", "GENERIC")
            last = last_ts.get(atype, 0.0)
            if now - last < cooldown_sec:
                return False
            last_ts[atype] = now
            return True

        last_ts = {}
        now = 1000.0

        assert bot_cooldown_check(sqz, last_ts, 30.0, now) is True
        # squeeze repetido dentro do cooldown -> bloqueado
        assert bot_cooldown_check(sqz, last_ts, 30.0, now + 5) is False
        # expansion NÃO é bloqueado pelo cooldown do squeeze (chave distinta)
        assert bot_cooldown_check(exp, last_ts, 30.0, now + 5) is True
        # expansion repetido dentro do cooldown -> bloqueado
        assert bot_cooldown_check(exp, last_ts, 30.0, now + 10) is False


class TestGenerateAlertsRealPath:
    """generate_alerts com o detector real (sem volume spike)."""

    def _volume_no_spike(self):
        # ratio atual/média = 100/550 = 0.18 << 3.0 -> sem VOLUME_SPIKE
        return 100.0, [5000.0, 6000.0]

    def test_generate_alerts_expansion_real(self):
        current_volume, avg_history = self._volume_no_spike()
        alerts = generate_alerts(
            price=100.0,
            support_resistance={},
            current_volume=current_volume,
            average_volume=sum(avg_history) / len(avg_history),
            current_volatility=2.0,
            recent_volatilities=list(EXPANSION_VOLS),
        )
        vol_alerts = [a for a in alerts if "VOLATILITY" in a["type"]]
        assert len(vol_alerts) == 1
        assert vol_alerts[0]["type"] == "VOLATILITY_EXPANSION"

    def test_generate_alerts_squeeze_real(self):
        current_volume, avg_history = self._volume_no_spike()
        alerts = generate_alerts(
            price=100.0,
            support_resistance={},
            current_volume=current_volume,
            average_volume=sum(avg_history) / len(avg_history),
            current_volatility=1e-6,
            recent_volatilities=list(SQUEEZE_VOLS),
        )
        vol_alerts = [a for a in alerts if "VOLATILITY" in a["type"]]
        assert len(vol_alerts) == 1
        assert vol_alerts[0]["type"] == "VOLATILITY_SQUEEZE"