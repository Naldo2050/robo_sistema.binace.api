"""
Testes para invariantes de dados: delta, volume, quality metrics.
Cobre bugs #7 (delta or 0.0), #8 (quality rate), #9 (volume arredondamento).
"""

import pytest
import logging


# ──────────────────────────────────────────────
# Testes: Bug #7 — Delta 0.0 tratado como falsy
# ──────────────────────────────────────────────

class TestDeltaFalsy:
    """
    Bug: `delta_fechamento or recalc` trata 0.0 como falsy.
    Fix: usar `is not None` em vez de `or`.
    """

    @staticmethod
    def _calc_delta_buggy(enriched_data: dict) -> float:
        """Lógica ANTIGA (bugada)."""
        return (
            enriched_data.get("delta_fechamento")
            or (
                float(enriched_data.get("volume_compra", 0))
                - float(enriched_data.get("volume_venda", 0))
            )
        )

    @staticmethod
    def _calc_delta_fixed(enriched_data: dict) -> float:
        """Lógica NOVA (corrigida)."""
        return (
            enriched_data.get("delta_fechamento")
            if enriched_data.get("delta_fechamento") is not None
            else (
                float(enriched_data.get("volume_compra", 0))
                - float(enriched_data.get("volume_venda", 0))
            )
        )

    def test_delta_zero_is_valid(self):
        """delta_fechamento=0.0 é legítimo (mercado equilibrado)."""
        data = {
            "delta_fechamento": 0.0,
            "volume_compra": 5.0,
            "volume_venda": 3.0,
        }
        # Bug: or trata 0.0 como falsy → recalcula como 2.0
        assert self._calc_delta_buggy(data) == 2.0  # ERRADO

        # Fix: is not None preserva 0.0
        assert self._calc_delta_fixed(data) == 0.0  # CORRETO

    def test_delta_none_recalculates(self):
        """delta_fechamento=None deve recalcular."""
        data = {
            "delta_fechamento": None,
            "volume_compra": 18.36,
            "volume_venda": 9.64,
        }
        result = self._calc_delta_fixed(data)
        assert abs(result - 8.72) < 0.01

    def test_delta_missing_recalculates(self):
        """delta_fechamento ausente deve recalcular."""
        data = {
            "volume_compra": 10.0,
            "volume_venda": 7.5,
        }
        result = self._calc_delta_fixed(data)
        assert abs(result - 2.5) < 0.01

    def test_delta_positive_preserved(self):
        """delta_fechamento positivo deve ser preservado."""
        data = {
            "delta_fechamento": 5.5,
            "volume_compra": 99.0,
            "volume_venda": 1.0,
        }
        assert self._calc_delta_fixed(data) == 5.5

    def test_delta_negative_preserved(self):
        """delta_fechamento negativo deve ser preservado."""
        data = {
            "delta_fechamento": -3.2,
            "volume_compra": 1.0,
            "volume_venda": 99.0,
        }
        assert self._calc_delta_fixed(data) == -3.2

    def test_delta_no_volumes_returns_zero(self):
        """Sem delta nem volumes, deve retornar 0.0."""
        data = {}
        result = self._calc_delta_fixed(data)
        assert result == 0.0

    def test_delta_zero_vs_none_distinction(self):
        """0.0 e None devem ter comportamentos diferentes."""
        data_zero = {"delta_fechamento": 0.0, "volume_compra": 10, "volume_venda": 5}
        data_none = {"delta_fechamento": None, "volume_compra": 10, "volume_venda": 5}

        # 0.0 → preservar
        assert self._calc_delta_fixed(data_zero) == 0.0
        # None → recalcular
        assert self._calc_delta_fixed(data_none) == 5.0

    def test_delta_false_values_table(self):
        """Tabela de valores falsy vs None."""
        cases = [
            # (delta_fechamento, expected_with_fix)
            (0.0, 0.0),         # falsy mas válido
            (0, 0),             # int zero, válido
            (None, 5.0),        # None → recalcular
            (1.5, 1.5),         # positivo
            (-2.0, -2.0),       # negativo
        ]
        for delta_val, expected in cases:
            data = {
                "delta_fechamento": delta_val,
                "volume_compra": 10.0,
                "volume_venda": 5.0,
            }
            result = self._calc_delta_fixed(data)
            assert result == expected, (
                f"delta_fechamento={delta_val!r}: "
                f"expected={expected}, got={result}"
            )


# ──────────────────────────────────────────────
# Testes: Bug #9 — Volume arredondamento float
# ──────────────────────────────────────────────

class TestVolumeRounding:
    """
    Bug: Volume diff de 0.00001 BTC gera alerta (arredondamento float).
    Fix: Tolerância adequada (0.0001 para volume, 0.01 para delta).
    """

    TOLERANCE_BTC = 1e-4     # 0.0001 BTC
    TOLERANCE_DELTA = 1e-2   # 0.01 BTC

    @staticmethod
    def validate_volume(vol_buy, vol_sell, vol_total, tolerance=1e-4):
        """Validação de volume com tolerância."""
        vol_sum = vol_buy + vol_sell
        diff = abs(vol_sum - vol_total)

        if diff > 0.01:
            return "ERROR", diff
        elif diff > tolerance:
            return "WARNING", diff
        else:
            return "OK", diff

    @staticmethod
    def validate_delta(vol_buy, vol_sell, stored_delta, tolerance=1e-2):
        """Validação de delta com correção automática."""
        calculated = vol_buy - vol_sell
        diff = abs(calculated - stored_delta)

        if diff > tolerance:
            return "CORRECTED", calculated
        else:
            return "OK", stored_delta

    def test_exact_volume_ok(self):
        """Volume exato não gera alerta."""
        status, diff = self.validate_volume(6.0, 4.0, 10.0)
        assert status == "OK"

    def test_tiny_rounding_ok(self):
        """Diferença de 0.00001 (float rounding) é OK."""
        # Caso típico de arredondamento float: diff ~1e-15
        status, diff = self.validate_volume(6.14450000, 7.92940000, 14.07390000)
        assert status == "OK", f"Diff {diff} não deveria gerar alerta"

    def test_log_scenario_volume_0001_is_warning(self):
        """Caso do log: diff=0.0001 BTC gera WARNING (> tolerance) mas NÃO ERROR."""
        # 6.1445 + 7.9293 = 14.0738, total=14.0739 → diff=0.0001
        status, diff = self.validate_volume(6.14450000, 7.92930000, 14.07390000)
        # 0.0001 > TOLERANCE_BTC(1e-4) por float imprecision → WARNING, não ERROR
        assert status == "WARNING"
        assert diff < 0.01  # NÃO é erro grave

    def test_small_rounding_ok(self):
        """Diferença de 0.00005 é OK (< 0.0001)."""
        status, diff = self.validate_volume(5.0, 5.0, 10.00005)
        assert status == "OK"

    def test_medium_diff_warning(self):
        """Diferença de 0.001 gera warning (> 0.0001, < 0.01)."""
        status, diff = self.validate_volume(5.0, 5.0, 10.001)
        assert status == "WARNING"

    def test_large_diff_error(self):
        """Diferença de 0.1 gera erro."""
        status, diff = self.validate_volume(5.0, 5.0, 10.1)
        assert status == "ERROR"

    def test_delta_match_ok(self):
        """Delta correto não gera correção."""
        status, value = self.validate_delta(18.36, 9.64, 8.72)
        assert status == "OK"
        assert abs(value - 8.72) < 0.001

    def test_delta_zero_stored_but_nonzero_calculated(self):
        """
        Bug #7: delta armazenado 0.0 mas calculado 8.72.
        Deve corrigir automaticamente.
        """
        status, value = self.validate_delta(18.36, 9.64, 0.0)
        assert status == "CORRECTED"
        assert abs(value - 8.72) < 0.01

    def test_delta_small_diff_accepted(self):
        """Diferença < 0.01 em delta é aceitável."""
        status, value = self.validate_delta(10.0, 5.0, 5.005)
        assert status == "OK"

    def test_float_precision_edge_cases(self):
        """Casos de borda de precisão float."""
        cases = [
            # (buy, sell, total, expected_status)
            (0.1, 0.2, 0.3, "OK"),                  # Clássico float 0.1+0.2
            (1/3, 2/3, 1.0, "OK"),                   # Frações
            (0.00001, 0.00002, 0.00003, "OK"),       # Muito pequeno
            (100000.1, 100000.2, 200000.3, "OK"),    # Muito grande
        ]
        for buy, sell, total, expected in cases:
            status, diff = self.validate_volume(buy, sell, total)
            assert status == expected, (
                f"buy={buy}, sell={sell}, total={total}: "
                f"status={status} (expected={expected}), diff={diff}"
            )


# ──────────────────────────────────────────────
# Testes: Bug #8 — Quality rate inclui whale corrections
# ──────────────────────────────────────────────

class TestQualityRate:
    """
    Bug: Whale corrections contam no correction_rate → sempre > 15%.
    Fix: Excluir whale corrections do rate; usar threshold de 20%.
    """

    EXPECTED_CORRECTIONS = frozenset({
        "reconciled_whale_volume",
        "recalculated_whale_delta",
        "reconciled_whale_count",
    })

    WARNING_THRESHOLD = 20.0
    ERROR_THRESHOLD = 35.0

    @classmethod
    def calc_problematic_rate(
        cls, total_events, corrections_by_type
    ) -> tuple:
        """
        Calcula rate excluindo whale corrections.
        Returns: (rate_pct, problematic_count, expected_count)
        """
        expected_count = sum(
            count for ctype, count in corrections_by_type.items()
            if ctype in cls.EXPECTED_CORRECTIONS
        )
        total_corrections = sum(corrections_by_type.values())
        problematic = max(0, total_corrections - expected_count)
        rate = (problematic / total_events * 100) if total_events > 0 else 0
        return rate, problematic, expected_count

    def test_whale_corrections_excluded(self):
        """Whale corrections NÃO devem contar no rate."""
        corrections = {
            "recalculated_delta": 2,          # problemático
            "reconciled_whale_volume": 3,     # esperado
            "recalculated_whale_delta": 1,    # esperado
        }
        rate, problematic, expected = self.calc_problematic_rate(30, corrections)
        assert problematic == 2
        assert expected == 4
        assert abs(rate - 6.67) < 0.1  # 2/30 = 6.67%

    def test_all_whale_rate_zero(self):
        """Se todas são whale corrections, rate = 0."""
        corrections = {
            "reconciled_whale_volume": 5,
            "recalculated_whale_delta": 3,
        }
        rate, problematic, expected = self.calc_problematic_rate(30, corrections)
        assert problematic == 0
        assert rate == 0.0

    def test_no_corrections_rate_zero(self):
        """Sem correções, rate = 0."""
        rate, problematic, expected = self.calc_problematic_rate(30, {})
        assert rate == 0.0
        assert problematic == 0
        assert expected == 0

    def test_scenario_from_logs(self):
        """
        Cenário real do log:
        - 28 eventos totais
        - 5 recalculated_delta (problemático)
        - 1 recalculated_whale_delta (esperado)
        - 1 reconciled_whale_volume (esperado)

        Antes (bugado): (5+1+1)/28 = 25% → WARNING
        Depois (fix):   5/28 = 17.9% → OK (< 20%)
        """
        corrections = {
            "recalculated_delta": 5,
            "recalculated_whale_delta": 1,
            "reconciled_whale_volume": 1,
        }
        rate, problematic, expected = self.calc_problematic_rate(28, corrections)
        assert problematic == 5
        assert expected == 2
        assert abs(rate - 17.86) < 0.1
        # Com threshold de 20%, NÃO gera warning
        assert rate < self.WARNING_THRESHOLD

    def test_scenario_from_logs_window_22(self):
        """
        Janela 22: 23 eventos, 4 recalculated_delta.
        Antes: 4/23 = 17.4% → WARNING (> 15%)
        Depois: 4/23 = 17.4% → OK (< 20%)
        """
        corrections = {"recalculated_delta": 4}
        rate, problematic, _ = self.calc_problematic_rate(23, corrections)
        assert abs(rate - 17.39) < 0.1
        assert rate < self.WARNING_THRESHOLD

    def test_high_problematic_rate_triggers_warning(self):
        """Rate > 20% (sem whales) deve gerar warning."""
        corrections = {"recalculated_delta": 8}
        rate, _, _ = self.calc_problematic_rate(30, corrections)
        assert abs(rate - 26.67) < 0.1
        assert rate > self.WARNING_THRESHOLD

    def test_very_high_rate_triggers_error(self):
        """Rate > 35% deve gerar error."""
        corrections = {"recalculated_delta": 15}
        rate, _, _ = self.calc_problematic_rate(30, corrections)
        assert rate == 50.0
        assert rate > self.ERROR_THRESHOLD

    def test_zero_events_safe(self):
        """Zero eventos não deve causar divisão por zero."""
        rate, problematic, expected = self.calc_problematic_rate(
            0, {"recalculated_delta": 5}
        )
        assert rate == 0.0

    def test_mixed_corrections_realistic(self):
        """Cenário realista com mix de correções."""
        corrections = {
            "recalculated_delta": 3,
            "reconciled_whale_volume": 2,
            "recalculated_whale_delta": 1,
            "reconciled_whale_count": 1,
            "fixed_timestamp": 1,  # outro tipo problemático
        }
        rate, problematic, expected = self.calc_problematic_rate(50, corrections)
        # Problemáticos: recalculated_delta(3) + fixed_timestamp(1) = 4
        # Esperados: whale_vol(2) + whale_delta(1) + whale_count(1) = 4
        assert problematic == 4
        assert expected == 4
        assert abs(rate - 8.0) < 0.1  # 4/50 = 8%


# ──────────────────────────────────────────────
# Testes: Integração delta → quality
# ──────────────────────────────────────────────

class TestDeltaQualityIntegration:
    """Testa o fluxo completo: delta calculado → validado → quality rate."""

    @staticmethod
    def _simulate_window(
        delta_fechamento, vol_compra, vol_venda, whale_corrections=0
    ):
        """Simula uma janela completa de dados."""
        # 1. Calcular delta (lógica corrigida)
        delta = (
            delta_fechamento
            if delta_fechamento is not None
            else (float(vol_compra) - float(vol_venda))
        )

        # 2. Validar invariante
        delta_calc = float(vol_compra) - float(vol_venda)
        delta_ok = abs(delta_calc - delta) < 0.01

        # 3. Se não bate, corrigir
        if not delta_ok:
            delta = delta_calc
            correction_type = "recalculated_delta"
        else:
            correction_type = None

        return {
            "delta": delta,
            "delta_ok": delta_ok,
            "correction_type": correction_type,
        }

    def test_normal_window_no_correction(self):
        """Janela normal: delta bate com volumes."""
        result = self._simulate_window(
            delta_fechamento=5.0,
            vol_compra=10.0,
            vol_venda=5.0,
        )
        assert result["delta_ok"] is True
        assert result["correction_type"] is None
        assert result["delta"] == 5.0

    def test_zero_delta_preserved(self):
        """delta=0.0 legítimo não é corrigido."""
        result = self._simulate_window(
            delta_fechamento=0.0,
            vol_compra=5.0,
            vol_venda=5.0,
        )
        assert result["delta_ok"] is True
        assert result["delta"] == 0.0

    def test_none_delta_recalculated(self):
        """delta=None é recalculado a partir dos volumes."""
        result = self._simulate_window(
            delta_fechamento=None,
            vol_compra=18.36,
            vol_venda=9.64,
        )
        assert abs(result["delta"] - 8.72) < 0.01

    def test_wrong_delta_corrected(self):
        """delta incorreto é corrigido e registrado."""
        result = self._simulate_window(
            delta_fechamento=0.0,  # errado (deveria ser 8.72)
            vol_compra=18.36,
            vol_venda=9.64,
        )
        # delta=0.0 is not None, então é preservado inicialmente
        # Mas validação detecta que 0.0 != 8.72 → corrige
        assert result["correction_type"] == "recalculated_delta"
        assert abs(result["delta"] - 8.72) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
