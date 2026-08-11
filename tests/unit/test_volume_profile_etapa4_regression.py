# -*- coding: utf-8 -*-
"""
ETAPA 4 — AUDITORIA MATEMÁTICA DO VOLUME PROFILE (2026-08-10)

Testes sintéticos que fixam o CONTRATO MATEMÁTICO real dos produtores:
  A) volume_profile.calculate_value_area_volume_pct — caminho com BINS
     (a métrica é: volume_dentro_da_VA / volume_total, NUNCA função da posição
     do preço atual).
  B) REG-HIST (bug): vp_profile SEM bins + instância dummy [price]/[1.0]
     → reproduz o comportamento observado em produção (total_volume == 1 e
     pct ∈ {0, 100} conforme o preço atual está dentro da VA).
  C) market_analysis.historical_profiler — deve expor price_bins/volume_per_bin
     (conservação de volume) para que o caminho A seja alcançável.
  D) detect_no_mans_land — contrato de gaps (bins $1 densos → 0 zonas).
  E) classify_profile_shape — terços de preço + bimodal.
  F) score_volume_nodes — volume_score == 15 sem dados de volume (contrato atual).
  G) Invariantes globais.
"""
import unittest

import numpy as np
import pandas as pd

from support_resistance.volume_profile import VolumeProfileAnalyzer
from market_analysis.dynamic_volume_profile import DynamicVolumeProfile
from market_analysis.historical_profiler import HistoricalVolumeProfiler


def _vpa_from(profile, prices=(70000.0,), vols=(1.0,)):
    return VolumeProfileAnalyzer(pd.Series(list(prices)), pd.Series(list(vols)))


class TestVaPctCaminhoBins(unittest.TestCase):
    """A) Caminho com price_bins/volume_per_bin — a métrica REAL."""

    def _profile(self, volume_scale=1.0):
        return {
            "poc": {"price": 102.0, "volume": 10.0 * volume_scale, "percent_of_total": 20.0},
            "value_area": {"low": 100.4, "high": 102.8},
            "price_bins": [100.4, 101.2, 102.0, 102.8, 103.6],
            "volume_per_bin": [10.0 * volume_scale] * 5,
        }

    def test_va_pct_mede_volume_na_va(self):
        r = _vpa_from({}).calculate_value_area_volume_pct(self._profile())
        # 4 bins em [100.4, 102.8] de 5 → 40/50
        self.assertEqual(r["value_area_volume_pct"], 80.0)
        self.assertEqual(r["total_volume"], 50.0)
        self.assertEqual(r["volume_in_va"], 40.0)
        self.assertIn(r["interpretation"], ("slightly_compressed", "compressed"))

    def test_va_pct_invariante_escala(self):
        r = _vpa_from({}).calculate_value_area_volume_pct(self._profile(volume_scale=10.0))
        self.assertEqual(r["value_area_volume_pct"], 80.0)
        self.assertEqual(r["total_volume"], 500.0)

    def test_va_pct_conserva_volume(self):
        r = _vpa_from({}).calculate_value_area_volume_pct(self._profile())
        self.assertAlmostEqual(r["volume_in_va"] + r["volume_in_va"] / 4.0, r["total_volume"])

    def test_va_pct_nao_depende_da_instancia(self):
        # Mesmo profile, instâncias com dados diferentes → mesmo resultado (via bins)
        r1 = _vpa_from({}, prices=(100.0,), vols=(1.0,)).calculate_value_area_volume_pct(self._profile())
        r2 = _vpa_from({}, prices=(99999.0, 100000.0), vols=(5.0, 7.0)).calculate_value_area_volume_pct(self._profile())
        self.assertEqual(r1["value_area_volume_pct"], r2["value_area_volume_pct"])


class TestVaPctBugDummyRegHist(unittest.TestCase):
    """B) REG-HIST — o caminho sem bins + instância dummy deve declarar
    insuficiência, NUNCA fabricar 0%/100% (bug observado em produção:
    total_volume==1 e pct função da posição do preço, não do volume)."""

    def _vp_profile_institucional(self, current_price, val=63988.9, vah=65051.3):
        # Formato montado por institutional_analytics._compute_profile_analysis
        return {
            "poc": {"price": 64689.0, "volume": 0, "percent_of_total": 0},
            "value_area": {"low": val, "high": vah},
            "volume_nodes": {"hvn": [64100.0, 64689.0], "lvn": [63950.0, 65000.0], "hvn_levels": []},
            "current_position": {"price": current_price},
        }

    def test_dummy_sem_bins_retorna_insufficient_data(self):
        # 1 ponto dummy [64732.2]/[1.0] não pode mais gerar 100.0 fabricado
        vpa = VolumeProfileAnalyzer(pd.Series([64732.2]), pd.Series([1.0]))
        r = vpa.calculate_value_area_volume_pct(self._vp_profile_institucional(64732.2))
        self.assertEqual(r["interpretation"], "insufficient_data")
        self.assertEqual(r["breakout_risk"], "UNKNOWN")
        self.assertEqual(r["value_area_volume_pct"], 0.0)
        self.assertEqual(r["total_volume"], 0.0)
        self.assertFalse(r["compression_signal"])

    def test_dummy_nao_depende_mais_da_posicao_do_preco(self):
        # Antes do fix: preço dentro → 100.0, fora → 0.0 (bug). Agora: insuficiente.
        inside = VolumeProfileAnalyzer(pd.Series([64732.2]), pd.Series([1.0]))
        outside = VolumeProfileAnalyzer(pd.Series([65500.0]), pd.Series([1.0]))
        r_in = inside.calculate_value_area_volume_pct(self._vp_profile_institucional(64732.2))
        r_out = outside.calculate_value_area_volume_pct(self._vp_profile_institucional(65500.0))
        self.assertEqual(r_in["interpretation"], r_out["interpretation"])
        self.assertEqual(r_in["interpretation"], "insufficient_data")
        self.assertNotEqual(r_in["value_area_volume_pct"], 100.0)  # antes fabricava 100
        self.assertEqual(r_out["total_volume"], 0.0)  # antes fabricava 1.0

    def test_error_em_calculo_de_bins_retorna_status_error(self):
        profile = {
            "value_area": {"low": 100.0, "high": 102.0},
            "price_bins": ["100.0", "abc", "102.0"],
            "volume_per_bin": [10.0, 10.0, 10.0],
        }
        vpa = VolumeProfileAnalyzer(pd.Series([100.0, 101.0]), pd.Series([1.0, 1.0]))
        r = vpa.calculate_value_area_volume_pct(profile)
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["interpretation"], "UNKNOWN")
        self.assertEqual(r["breakout_risk"], "UNKNOWN")
        self.assertEqual(r["value_area_volume_pct"], 0.0)
        self.assertFalse(r["compression_signal"])
        self.assertEqual(r["volume_in_va"], 0.0)
        self.assertEqual(r["total_volume"], 0.0)

    def test_fallback_legitimo_com_dados_reais(self):
        # Fallback continua válido quando a instância tem dados reais (>=2 pontos)
        vpa = VolumeProfileAnalyzer(pd.Series([100.0, 101.0, 102.0]), pd.Series([1.0, 2.0, 3.0]))
        profile = {"value_area": {"low": 100.0, "high": 102.0}}
        r = vpa.calculate_value_area_volume_pct(profile)
        self.assertEqual(r["total_volume"], 6.0)
        self.assertEqual(r["value_area_volume_pct"], 100.0)  # 100% legítimo: todo volume na VA


class TestProfilerExpõeBins(unittest.TestCase):
    """C) historical_profiler deve expor os bins já calculados ($1) para o vp_profile."""

    def _df(self):
        return pd.DataFrame({
            "p": [64000.4, 64001.2, 64100.0, 64100.8, 64500.0, 64500.6, 64600.0],
            "q": [1.5, 2.5, 10.0, 8.0, 6.0, 6.5, 4.0],
        })

    def setUp(self):
        self.profiler = HistoricalVolumeProfiler("BTC/USDT", num_days=1, value_area_percent=0.70)

    def test_perfil_retorna_bins_e_conserva_volume(self):
        vp = self.profiler._calculate_profile(self._df(), "daily")
        self.assertEqual(vp["status"], "success")
        self.assertIn("price_bins", vp)
        self.assertIn("volume_per_bin", vp)
        self.assertEqual(len(vp["price_bins"]), len(vp["volume_per_bin"]))
        self.assertAlmostEqual(sum(vp["volume_per_bin"]), self._df()["q"].sum())

    def test_vp_profile_institucional_com_bins_da_va_real(self):
        # Integração: formato do institutional_analytics ENRIQUECIDO com bins do profiler
        vp = self.profiler._calculate_profile(self._df(), "daily")
        vp_profile = {
            "poc": {"price": vp["poc"], "volume": 0, "percent_of_total": 0},
            "value_area": {"low": vp["val"], "high": vp["vah"]},
            "volume_nodes": {"hvn": vp["hvns"], "lvn": vp["lvns"], "hvn_levels": []},
            "price_bins": vp["price_bins"],
            "volume_per_bin": vp["volume_per_bin"],
            "current_position": {"price": 64500.0},
        }
        vpa = VolumeProfileAnalyzer(pd.Series([64500.0]), pd.Series([1.0]))
        r = vpa.calculate_value_area_volume_pct(vp_profile)
        # VA POC-outward a 70% do volume total → pct real próximo de 70 (bins $1
        # discretos podem saltar para 75-85% — aceita LOW/MEDIUM, nunca 0/100)
        self.assertGreater(r["value_area_volume_pct"], 55.0)
        self.assertLess(r["value_area_volume_pct"], 95.0)
        self.assertGreater(r["total_volume"], 1.0)
        self.assertNotIn(r["value_area_volume_pct"], (0.0, 100.0))
        self.assertIn(r["breakout_risk"], ("LOW", "MEDIUM"))


class TestNoMansLand(unittest.TestCase):
    """D) detect_no_mans_land — contrato de gaps."""

    def test_hvns_densos_bins_1_sem_zonas(self):
        profile = {"volume_nodes": {"hvn": [64000.0, 64001.0, 64002.0, 64003.0], "lvn": []}}
        r = _vpa_from({}).detect_no_mans_land(profile, current_price=64001.5)
        self.assertEqual(r["status"], "success")
        self.assertEqual(r["total_zones"], 0)

    def test_gap_grande_com_lvn_gera_zona(self):
        profile = {"volume_nodes": {"hvn": [64000.0, 65000.0], "lvn": [64500.0]}}
        r = _vpa_from({}).detect_no_mans_land(profile, current_price=64500.0)
        self.assertEqual(r["total_zones"], 1)
        zone = r["zones"][0]
        self.assertTrue(zone["lvn_confirmed"])
        self.assertTrue(r["price_in_no_mans_land"])

    def test_gap_grande_sem_lvn_ainda_gera_zona(self):
        # Contrato atual: a zona é criada mesmo sem LVN (lvn_confirmed é flag informativo)
        profile = {"volume_nodes": {"hvn": [64000.0, 65000.0], "lvn": []}}
        r = _vpa_from({}).detect_no_mans_land(profile, current_price=65000.0)
        self.assertEqual(r["total_zones"], 1)
        self.assertFalse(r["zones"][0]["lvn_confirmed"])


class TestClassifyShape(unittest.TestCase):
    """E) classify_profile_shape — terços de preço + bimodal."""

    def _df(self, prices, vols):
        return pd.DataFrame({"p": prices, "q": vols})

    def test_lower_45_gera_b(self):
        df = self._df([100.0] * 50 + [110.0] * 20 + [120.0] * 30, [1.0] * 100)
        r = DynamicVolumeProfile("TEST").classify_profile_shape(df)
        self.assertEqual(r["shape"], "b")
        self.assertAlmostEqual(r["distribution"]["lower_third_pct"] / 100.0, 0.5, delta=0.01)

    def test_middle_50_gera_D(self):
        df = self._df([100.0] * 20 + [110.0] * 60 + [120.0] * 20, [1.0] * 100)
        r = DynamicVolumeProfile("TEST").classify_profile_shape(df)
        self.assertEqual(r["shape"], "D")

    def test_bimodal_direto_40_20_40_gera_B(self):
        # B direto: lower>0.32, upper>0.32, middle<0.30
        df = self._df(
            [100.0, 101.0, 102.0, 103.0] * 8 + [104.0, 105.0, 106.0, 107.0] * 4 + [108.0, 109.0, 110.0, 111.0] * 8,
            [3.0] * 80,
        )
        r = DynamicVolumeProfile("TEST").classify_profile_shape(df)
        self.assertEqual(r["shape"], "B")

    def test_clusters_nas_bordas_nao_disparam_bimodal(self):
        # LIMITAÇÃO REAL (EXPECTED_BEHAVIOR): o detector de picos varre
        # range(1, len-1) e nunca vê picos nas bordas do histograma →
        # clusters nas bordas do range caem em "P" (upper > 0.45 primeiro).
        df = self._df(
            list(range(100, 104)) * 10 + list(range(110, 114)) * 10,
            [3.0] * 80,
        )
        r = DynamicVolumeProfile("TEST").classify_profile_shape(df)
        self.assertEqual(r["shape"], "P")
        self.assertEqual(r["distribution"]["lower_third_pct"], 50.0)
        self.assertEqual(r["distribution"]["upper_third_pct"], 50.0)

    def test_distribuicao_soma_100(self):
        df = self._df([100.0] * 30 + [110.0] * 40 + [120.0] * 30, [1.0] * 100)
        r = DynamicVolumeProfile("TEST").classify_profile_shape(df)
        d = r["distribution"]
        self.assertAlmostEqual(d["lower_third_pct"] + d["middle_third_pct"] + d["upper_third_pct"], 100.0, delta=0.2)


class TestScoreVolumeNodes(unittest.TestCase):
    """F) score_volume_nodes — volume_score==15 sem dados de volume (contrato atual)."""

    def _vp_profile(self):
        return {
            "poc": {"price": 64689.0, "volume": 0, "percent_of_total": 0},
            "value_area": {"low": 63988.9, "high": 65051.3},
            "volume_nodes": {
                "hvn": [64100.0, 64689.0, 64900.0],
                "lvn": [63950.0, 65000.0],
                "hvn_levels": [],
            },
            "current_position": {"price": 64689.0},
        }

    def test_volume_score_15_sem_dados(self):
        r = _vpa_from({}).score_volume_nodes(self._vp_profile(), current_price=64689.0)
        self.assertEqual(r["status"], "success")
        self.assertEqual(r["total_hvns"], 3)
        self.assertEqual(r["total_lvns"], 2)
        for node in r["scored_hvns"] + r["scored_lvns"]:
            self.assertEqual(node["volume_score"], 15.0)

    def test_strength_dentro_dos_limites(self):
        r = _vpa_from({}).score_volume_nodes(self._vp_profile(), current_price=64689.0)
        for node in r["scored_hvns"] + r["scored_lvns"]:
            self.assertGreaterEqual(node["strength"], 0)
            self.assertLessEqual(node["strength"], 100)


class TestInvariantesGlobais(unittest.TestCase):
    """G) Invariantes matemáticas dos produtores."""

    def test_va_pct_intervalo(self):
        cases = [
            {"price_bins": [10.0, 11.0], "volume_per_bin": [5.0, 5.0],
             "value_area": {"low": 10.0, "high": 11.0}},
            {"price_bins": [10.0, 11.0, 12.0], "volume_per_bin": [1.0, 8.0, 1.0],
             "value_area": {"low": 11.0, "high": 11.0}},
            {},
        ]
        for c in cases:
            r = _vpa_from({}).calculate_value_area_volume_pct(c)
            self.assertGreaterEqual(r["value_area_volume_pct"], 0.0)
            self.assertLessEqual(r["value_area_volume_pct"], 100.0)

    def test_profiler_val_poc_vah_ordenado(self):
        profiler = HistoricalVolumeProfiler("BTC/USDT", num_days=1, value_area_percent=0.70)
        df = pd.DataFrame({"p": [100.0] * 3 + [101.0] * 7 + [102.0] * 5 + [103.0] * 4, "q": [1.0] * 19})
        vp = profiler._calculate_profile(df, "daily")
        if vp["status"] == "success":
            self.assertLessEqual(vp["val"], vp["poc"])
            self.assertLessEqual(vp["poc"], vp["vah"])


class TestPersistenciaSemBins(unittest.TestCase):
    """H) Contrato interno vs persistido: bins ficam no cálculo, saem do evento salvo."""

    @staticmethod
    def _tf(vp):
        return {
            "poc": vp.get("poc"), "vah": vp.get("vah"), "val": vp.get("val"),
            "status": vp.get("status", "success"),
        }

    def _evento_com_bins(self):
        daily = {
            "poc": 64100.0, "vah": 64140.0, "val": 64060.0, "status": "success",
            "hvns": [64100.0], "lvns": [63900.0],
            "price_bins": [64060.0, 64070.0, 64080.0, 64100.0, 64120.0, 64140.0],
            "volume_per_bin": [1.0, 2.0, 4.0, 8.0, 3.0, 2.0],
        }
        weekly = self._tf(daily)
        weekly["price_bins"] = [64000.0, 64100.0]
        weekly["volume_per_bin"] = [20.0, 80.0]
        monthly = {"poc": 64000.0, "vah": 64500.0, "val": 63600.0, "status": "success"}
        return {
            "tipo_evento": "Exaustão",
            "historical_vp": {"daily": daily, "weekly": weekly, "monthly": monthly},
        }

    def test_persistido_sem_bins_em_todos_os_timeframes(self):
        from data_processing.fix_optimization import strip_profile_bins
        sanitizado = strip_profile_bins(self._evento_com_bins())
        for tf in ("daily", "weekly", "monthly"):
            vp = sanitizado["historical_vp"][tf]
            self.assertNotIn("price_bins", vp)
            self.assertNotIn("volume_per_bin", vp)
            self.assertIn("poc", vp)
            self.assertIn("vah", vp)
            self.assertIn("val", vp)

    def test_persistido_sem_bins_em_locais_aninhados(self):
        from data_processing.fix_optimization import strip_profile_bins
        ev = self._evento_com_bins()
        ev["raw_event"] = {"historical_vp": ev["historical_vp"]}
        ev["contextual_snapshot"] = {"historical_vp": ev["historical_vp"]}
        sanitizado = strip_profile_bins(ev)
        for loc in ("historical_vp", "raw_event.historical_vp", "contextual_snapshot.historical_vp"):
            cur = sanitizado
            for part in loc.split("."):
                cur = cur[part]
            self.assertNotIn("price_bins", cur["daily"])
            self.assertNotIn("volume_per_bin", cur["daily"])

    def test_original_nao_mutado(self):
        from data_processing.fix_optimization import strip_profile_bins
        ev = self._evento_com_bins()
        bins_antes = ev["historical_vp"]["daily"]["price_bins"]
        strip_profile_bins(ev)
        self.assertIn("price_bins", ev["historical_vp"]["daily"])
        self.assertEqual(ev["historical_vp"]["daily"]["price_bins"], bins_antes)
        self.assertIn("volume_per_bin", ev["historical_vp"]["daily"])

    def test_sem_bins_retorna_mesmo_objeto(self):
        from data_processing.fix_optimization import strip_profile_bins
        ev = {"historical_vp": {"daily": {"poc": 1.0, "vah": 2.0, "val": 0.5}}}
        self.assertIs(strip_profile_bins(ev), ev)

    def test_event_saver_sanitiza_antes_de_persistir(self):
        # Contrato de regressão: save_event DEVE chamar strip_profile_bins antes de
        # persistir (SQLite/jsonl/visual/fallback). Sem instanciar o saver (evita
        # escrita real em arquivos/DB).
        import inspect
        import events.event_saver as es_mod
        src = inspect.getsource(es_mod.EventSaver.save_event)
        self.assertIn("strip_profile_bins", src)
        self.assertLess(
            src.index("strip_profile_bins"),
            src.index("janela_numero"),
        )


if __name__ == "__main__":
    unittest.main()
