# -*- coding: utf-8 -*-
"""
ETAPA 4B — FAIL-CLOSED DO VOLUME PROFILE (2026-08-10)

Escopo estrito: support_resistance/volume_profile.calculate_value_area_volume_pct
e seus consumidores de payload (ai_payload_builder, payload_builder_compact,
institutional_summary).

Objetivo: nenhuma exceção real pode virar um dado financeiro plausível.
Antes do fix (fail-open): except fabricava volume_in_va=70 / total_volume=100
→ value_area_volume_pct=70%, indistinguível de cálculo real.

Contrato novo (fail-closed), schema uniforme de 7 chaves em TODO ramo:
    status:              "success" | "insufficient_data" | "error"
    value_area_volume_pct: pct real | 0.0
    interpretation:      real | "insufficient_data" | "UNKNOWN"
    breakout_risk:       real | "UNKNOWN"
    volume_in_va:        real | 0.0
    total_volume:        real | 0.0
    compression_signal:  bool real | False

Cobertura:
    A) exceção forçada → NUNCA 70% fabricado
    B) estado erro explicitamente identificável (status + schema uniforme)
    C) IA/payload: erro não vira VERY_HIGH/HIGH compression nem falso breakout
    D) caminho saudável: dataset real ~70% continua retornando ~70%
    E) insufficient_data semanticamente diferente do cálculo saudável
    F) sem NaN/Infinity em nenhum estado
"""
import math
import unittest
from unittest import mock

import pandas as pd

from support_resistance.volume_profile import VolumeProfileAnalyzer


def _vpa(prices=(100.0, 101.0), vols=(1.0, 1.0)):
    return VolumeProfileAnalyzer(pd.Series(list(prices)), pd.Series(list(vols)))


def _profile_error_bins():
    """Bins com tipo inválido (contrato violado) → np.asarray(dtype=float64) levanta."""
    return {
        "value_area": {"low": 100.0, "high": 102.0},
        "price_bins": ["100.0", "abc", "102.0"],
        "volume_per_bin": [10.0, 10.0, 10.0],
    }


def _profile_sucesso_70():
    """10 bins de volume 10; VA cobre 7 bins → 70/100 = 70% real."""
    return {
        "value_area": {"low": 101.0, "high": 107.0},
        "price_bins": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        "volume_per_bin": [10.0] * 10,
    }


class TestErroForcadoNaoFabrica70(unittest.TestCase):
    """A) Exceção forçada em operação interna → status error, NUNCA 70 fabricado."""

    def test_bins_invalidos_nao_retornam_70(self):
        r = _vpa().calculate_value_area_volume_pct(_profile_error_bins())
        self.assertEqual(r["status"], "error")
        self.assertNotEqual(r["value_area_volume_pct"], 70.0)  # antes: 70.0 fabricado
        self.assertEqual(r["value_area_volume_pct"], 0.0)
        self.assertEqual(r["interpretation"], "UNKNOWN")
        self.assertEqual(r["breakout_risk"], "UNKNOWN")
        self.assertFalse(r["compression_signal"])

    def test_calculate_profile_falhando_retorna_status_error(self):
        vpa = _vpa()
        with mock.patch.object(vpa, "calculate_profile", side_effect=RuntimeError("boom")):
            r = vpa.calculate_value_area_volume_pct(profile=None)
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["value_area_volume_pct"], 0.0)
        self.assertEqual(r["interpretation"], "UNKNOWN")
        self.assertEqual(r["breakout_risk"], "UNKNOWN")
        self.assertFalse(r["compression_signal"])


class TestContratoDeEstado(unittest.TestCase):
    """B) Os três estados são explicitamente identificáveis e têm schema uniforme."""

    REQUIRED_KEYS = (
        "status", "value_area_volume_pct", "interpretation",
        "breakout_risk", "volume_in_va", "total_volume", "compression_signal",
    )

    def _estado_erro(self):
        return _vpa().calculate_value_area_volume_pct(_profile_error_bins())

    def _estado_insufficiente(self):
        return _vpa(prices=(100.0,), vols=(1.0,)).calculate_value_area_volume_pct({})

    def _estado_sucesso(self):
        return _vpa().calculate_value_area_volume_pct(_profile_sucesso_70())

    def test_estados_distintos_e_rotulados(self):
        e, i, s = self._estado_erro(), self._estado_insufficiente(), self._estado_sucesso()
        self.assertEqual(
            (e["status"], i["status"], s["status"]),
            ("error", "insufficient_data", "success"),
        )
        self.assertEqual(len({e["status"], i["status"], s["status"]}), 3)
        self.assertEqual(
            (e["interpretation"], i["interpretation"], s["interpretation"]),
            ("UNKNOWN", "insufficient_data", "normal"),
        )

    def test_schema_uniforme_em_todos_os_ramos(self):
        for r in (self._estado_erro(), self._estado_insufficiente(), self._estado_sucesso()):
            for key in self.REQUIRED_KEYS:
                self.assertIn(key, r, f"chave ausente no estado {r.get('status')}: {key}")

    def test_sem_nan_infinity_em_nenhum_estado(self):
        for r in (self._estado_erro(), self._estado_insufficiente(), self._estado_sucesso()):
            for key in ("value_area_volume_pct", "volume_in_va", "total_volume"):
                self.assertTrue(
                    math.isfinite(r[key]),
                    f"{key} não finito no estado {r.get('status')}: {r[key]}",
                )


class TestCaminhoSaudavel70Intacto(unittest.TestCase):
    """D) O 70% legítimo (cálculo real) continua retornando ~70% — a correção
    de erro NÃO pode quebrar o valor real."""

    def test_70_legitimo_success(self):
        r = _vpa().calculate_value_area_volume_pct(_profile_sucesso_70())
        self.assertEqual(r["status"], "success")
        self.assertEqual(r["value_area_volume_pct"], 70.0)
        self.assertEqual(r["total_volume"], 100.0)
        self.assertEqual(r["volume_in_va"], 70.0)
        self.assertEqual(r["interpretation"], "normal")
        self.assertEqual(r["breakout_risk"], "LOW")
        self.assertFalse(r["compression_signal"])

    def test_90_legitimo_compression_sinaliza(self):
        profile = {
            "value_area": {"low": 100.0, "high": 108.0},
            "price_bins": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            "volume_per_bin": [10.0] * 10,
        }
        r = _vpa().calculate_value_area_volume_pct(profile)
        self.assertEqual(r["status"], "success")
        self.assertEqual(r["value_area_volume_pct"], 90.0)
        self.assertTrue(r["compression_signal"])
        self.assertEqual(r["breakout_risk"], "VERY_HIGH")


class TestInsufficientSemanticamenteDistinto(unittest.TestCase):
    """E) insufficient_data é semanticamente diferente de cálculo saudável:
    nunca gera sinal de compression nem breakout."""

    def test_insufficient_data_sem_sinal(self):
        r = _vpa(prices=(100.0,), vols=(1.0,)).calculate_value_area_volume_pct({})
        self.assertEqual(r["status"], "insufficient_data")
        self.assertEqual(r["value_area_volume_pct"], 0.0)
        self.assertEqual(r["interpretation"], "insufficient_data")
        self.assertEqual(r["breakout_risk"], "UNKNOWN")
        self.assertFalse(r["compression_signal"])
        s = _vpa().calculate_value_area_volume_pct(_profile_sucesso_70())
        self.assertNotEqual(r["status"], s["status"])
        self.assertNotEqual(r["interpretation"], s["interpretation"])
        self.assertNotEqual(r["breakout_risk"], s["breakout_risk"])


class TestConsumidoresNaoPropagamErro(unittest.TestCase):
    """C) IA/payload: status=error/pct=0/UNKNOWN/False NÃO gera:
    falso breakout, HIGH/VERY_HIGH compression, crash, KeyError, comparação inválida."""

    @staticmethod
    def _estado_erro():
        return {
            "status": "error",
            "value_area_volume_pct": 0.0,
            "interpretation": "UNKNOWN",
            "breakout_risk": "UNKNOWN",
            "volume_in_va": 0.0,
            "total_volume": 0.0,
            "compression_signal": False,
        }

    def test_ai_payload_builder_ignora_erro(self):
        from market_orchestrator.ai.ai_payload_builder import _inject_institutional_analytics
        ai_payload = {"price_context": {}}
        signal = {
            "institutional_analytics": {
                "status": "ok",
                "profile_analysis": {"va_volume_pct": self._estado_erro()},
            }
        }
        out = _inject_institutional_analytics(ai_payload, signal)
        price = out["price_context"]
        self.assertNotIn("va_volume_pct", price)
        self.assertNotIn("va_compression", price)

    def test_ai_payload_builder_controlo_sucesso(self):
        from market_orchestrator.ai.ai_payload_builder import _inject_institutional_analytics
        ai_payload = {"price_context": {}}
        signal = {
            "institutional_analytics": {
                "status": "ok",
                "profile_analysis": {
                    "va_volume_pct": {
                        "status": "success",
                        "value_area_volume_pct": 70.0,
                        "interpretation": "normal",
                        "breakout_risk": "LOW",
                        "compression_signal": False,
                    }
                },
            }
        }
        out = _inject_institutional_analytics(ai_payload, signal)
        price = out["price_context"]
        self.assertEqual(price["va_volume_pct"], 70.0)
        self.assertIs(price["va_compression"], False)

    def test_compact_price_erro_nao_gera_brk_risk(self):
        from market_orchestrator.ai.payload_builder_compact import _build_price
        event = {
            "institutional_analytics": {
                "profile_analysis": {"va_volume_pct": self._estado_erro()}
            }
        }
        price = _build_price(event)
        self.assertNotIn("brk_risk", price)
        self.assertNotIn("va_compression", price)

    def test_compact_price_erro_nao_estoura_com_estado_insufficient(self):
        from market_orchestrator.ai.payload_builder_compact import _build_price
        event = {
            "institutional_analytics": {
                "profile_analysis": {
                    "va_volume_pct": {
                        "status": "insufficient_data",
                        "value_area_volume_pct": 0.0,
                        "interpretation": "insufficient_data",
                        "breakout_risk": "UNKNOWN",
                        "compression_signal": False,
                    }
                }
            }
        }
        price = _build_price(event)
        self.assertNotIn("brk_risk", price)

    def test_compact_price_controlo_very_high(self):
        from market_orchestrator.ai.payload_builder_compact import _build_price
        event = {
            "institutional_analytics": {
                "profile_analysis": {
                    "va_volume_pct": {
                        "status": "success",
                        "breakout_risk": "VERY_HIGH",
                        "compression_signal": True,
                    }
                }
            }
        }
        price = _build_price(event)
        self.assertEqual(price["brk_risk"], "V_HI")

    def test_institutional_summary_sem_falso_breakout(self):
        from market_orchestrator.ai.payload_sections.institutional_summary import build_institutional_summary
        payload = {"price": {"c": 100.0}, "w": {"s": 0, "c": "N"}, "flow": {}}
        summary = build_institutional_summary(payload)
        self.assertNotIn("breakout", summary["note"].lower())
        self.assertIsNotNone(summary["note"])


class TestInjecaoAlternativaDeErro(unittest.TestCase):
    """8) A política de ERROR fecha para QUALQUER exceção interna — não só
    para input string ("abc"). Os testes verificam a POLÍTICA, não a origem."""

    def test_objeto_inconvertivel_no_volume_bin_retorna_error(self):
        profile = {
            "value_area": {"low": 100.0, "high": 102.0},
            "price_bins": [100.0, 101.0, 102.0],
            "volume_per_bin": [10.0, object(), 10.0],
        }
        r = _vpa().calculate_value_area_volume_pct(profile)
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["value_area_volume_pct"], 0.0)
        self.assertEqual(r["interpretation"], "UNKNOWN")
        self.assertEqual(r["breakout_risk"], "UNKNOWN")
        self.assertFalse(r["compression_signal"])

    def test_mock_operacao_interna_lancando_retorna_error(self):
        profile = {
            "value_area": {"low": 100.0, "high": 102.0},
            "price_bins": [100.0, 101.0, 102.0],
            "volume_per_bin": [10.0, 10.0, 10.0],
        }
        vpa = _vpa()
        with mock.patch("numpy.asarray", side_effect=OverflowError("agregacao simulada")):
            r = vpa.calculate_value_area_volume_pct(profile)
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["value_area_volume_pct"], 0.0)
        self.assertEqual(r["interpretation"], "UNKNOWN")
        self.assertEqual(r["breakout_risk"], "UNKNOWN")
        self.assertFalse(r["compression_signal"])

    def test_resultado_nao_finito_retorna_error(self):
        # NaN num bin passa na conversão float64, mas o resultado não é finito:
        # também não pode virar success com percentual inválido.
        profile = {
            "value_area": {"low": 100.0, "high": 102.0},
            "price_bins": [100.0, 101.0, 102.0],
            "volume_per_bin": [10.0, float("nan"), 10.0],
        }
        r = _vpa().calculate_value_area_volume_pct(profile)
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["value_area_volume_pct"], 0.0)
        self.assertEqual(r["breakout_risk"], "UNKNOWN")
        self.assertFalse(r["compression_signal"])
        self.assertTrue(math.isfinite(r["value_area_volume_pct"]))

    def test_volume_total_zero_retorna_insufficient_data(self):
        # Antes: bins zerados → success com 0% fabricado (indistinguível).
        profile = {
            "value_area": {"low": 100.0, "high": 102.0},
            "price_bins": [100.0, 101.0, 102.0],
            "volume_per_bin": [0.0, 0.0, 0.0],
        }
        r = _vpa().calculate_value_area_volume_pct(profile)
        self.assertEqual(r["status"], "insufficient_data")
        self.assertEqual(r["value_area_volume_pct"], 0.0)
        self.assertEqual(r["interpretation"], "insufficient_data")
        self.assertEqual(r["breakout_risk"], "UNKNOWN")
        self.assertFalse(r["compression_signal"])


class TestObservabilidadeDoErro(unittest.TestCase):
    """7) O erro é observável: logger.warning é chamado, sem arrays completos,
    com operação e tipo da exceção — sem depender da string completa do erro."""

    def test_warning_emitido_no_caminho_de_bins_sem_arrays_completos(self):
        profile = {
            "value_area": {"low": 100.0, "high": 102.0},
            "price_bins": ["100.0", "abc", "102.0"],
            "volume_per_bin": [10.0, 10.0, 10.0],
        }
        with self.assertLogs("support_resistance.volume_profile", level="WARNING") as logs:
            r = _vpa().calculate_value_area_volume_pct(profile)
        self.assertEqual(r["status"], "error")
        self.assertTrue(logs.output, "esperava pelo menos um warning")
        joined = " ".join(logs.output)
        self.assertIn("value area", joined)      # operação
        self.assertIn("ValueError", joined)      # tipo da exceção
        self.assertNotIn("abc", joined)          # nada de arrays completos
        self.assertNotIn("[10.0", joined)        # volumes de bins não vazados

    def test_warning_emitido_quando_calculate_profile_falha(self):
        vpa = _vpa()
        with mock.patch.object(vpa, "calculate_profile", side_effect=RuntimeError("boom")):
            with self.assertLogs("support_resistance.volume_profile", level="WARNING") as logs:
                r = vpa.calculate_value_area_volume_pct(profile=None)
        self.assertEqual(r["status"], "error")
        self.assertTrue(any("calculate_profile" in m for m in logs.output))


if __name__ == "__main__":
    unittest.main()
