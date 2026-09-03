# tests/unit/test_feature_evaluator_diagnostics.py
# -*- coding: utf-8 -*-
"""
Suíte de Testes Automatizados "Test the Tester" — Fase V1.2.
Valida o próprio avaliador estatístico contra datasets sintéticos com verdade conhecida (Ground Truth):
1. Feature Fortemente Preditiva: Deve ser detectada (Δ AUC > 0.05, p < 0.05).
2. Ruído Puro (White Noise): NÃO deve ser promovido (Δ AUC <= 0.005 ou p > 0.05).
3. Feature Duplicada do Baseline: Não deve apresentar ganho incremental (Δ AUC ≈ 0.0).
4. Feature Constante (Zero Variance): Deve ser tratada com segurança sem crash e Δ AUC ≈ 0.0.
5. StandardScaler: Garante que features com micro-escala (ex: 1e-4) recebam pesos apropriados.
6. Permutação com Correção Finita: Garante p-valor no intervalo (0, 1] e nunca exatamente zero finito.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, ".")

from scripts.analytics.hardened_feature_evaluator import HardenedFeatureEvaluator


class TestFeatureEvaluatorDiagnostics(unittest.TestCase):
    """Testes de integridade matemática e defensiva do avaliador estatístico."""

    def setUp(self):
        self.evaluator = HardenedFeatureEvaluator(random_seed=42)
        np.random.seed(42)

        # Constrói dataset sintético base de 600 barras
        base_ts = 1788300000000
        n_bars = 600
        timestamps = np.array([base_ts + i * 60000 for i in range(n_bars)])

        # Feature base fracamente preditiva
        x_base1 = np.random.normal(0, 1.0, n_bars)
        x_base2 = np.random.normal(0, 1.0, n_bars)

        # Target forward 15m
        latent_signal = 0.2 * x_base1 + np.random.normal(0, 0.9, n_bars)
        y_target = (latent_signal > 0).astype(int)

        self.df = pd.DataFrame({
            "timestamp_ms": timestamps,
            "flow_base1": x_base1,
            "flow_base2": x_base2,
            "target_dir_15m": y_target,
            "fwd_ret_15m": latent_signal * 0.002,
        })
        self.base_cols = ["flow_base1", "flow_base2"]

    def test_strong_predictive_feature_detected(self):
        """Caso A: Feature fortemente preditiva deve ser detectada com alto Δ AUC e baixo p-valor."""
        df_test = self.df.copy()
        # Injeta feature candidata fortemente correlacionada com target
        df_test["strong_feature"] = self.df["target_dir_15m"].values * 2.0 + np.random.normal(0, 0.5, len(self.df))

        model_cols = self.base_cols + ["strong_feature"]
        res = self.evaluator.evaluate_cohort(
            df_test, self.base_cols, model_cols, "COHORT_STRONG_SIGNAL", horizon_bars=15, n_permutations=50
        )

        self.assertGreater(res.delta_oos_auc, 0.05)
        self.assertLess(res.null_permutation_p_val, 0.05)
        self.assertIn(res.verdict, ["STRONG_INCREMENTAL_EVIDENCE", "PROMISING"])

    def test_pure_noise_rejected(self):
        """Caso B: Ruído puro não deve ser promovido como evidência preditiva."""
        df_test = self.df.copy()
        df_test["noise_feature"] = np.random.normal(0, 1.0, len(self.df))

        model_cols = self.base_cols + ["noise_feature"]
        res = self.evaluator.evaluate_cohort(
            df_test, self.base_cols, model_cols, "COHORT_PURE_NOISE", horizon_bars=15, n_permutations=50
        )

        self.assertLess(res.delta_oos_auc, 0.02)
        self.assertGreaterEqual(res.null_permutation_p_val, 0.05)
        self.assertIn(res.verdict, ["INCREMENTAL_VALUE_UNKNOWN", "NO_EVIDENCE"])

    def test_duplicated_baseline_feature_no_gain(self):
        """Caso C: Duplicação exata de feature do baseline não deve gerar ganho espúrio."""
        df_test = self.df.copy()
        df_test["dup_feature"] = df_test["flow_base1"].values

        model_cols = self.base_cols + ["dup_feature"]
        res = self.evaluator.evaluate_cohort(
            df_test, self.base_cols, model_cols, "COHORT_DUPLICATE", horizon_bars=15, n_permutations=30
        )

        self.assertAlmostEqual(res.delta_oos_auc, 0.0, delta=0.01)

    def test_constant_feature_safe_handling(self):
        """Caso D: Feature constante (variância zero) não deve causar crash numérico ou divisão por zero."""
        df_test = self.df.copy()
        df_test["const_feature"] = 1.0

        model_cols = self.base_cols + ["const_feature"]
        res = self.evaluator.evaluate_cohort(
            df_test, self.base_cols, model_cols, "COHORT_CONST", horizon_bars=15, n_permutations=30
        )

        self.assertAlmostEqual(res.delta_oos_auc, 0.0, delta=0.01)

    def test_standard_scaler_enables_micro_scale_learning(self):
        """Caso G: Feature com micro-escala (ex: 1e-4) deve ser treinada sem ser anulada pela regularização L2."""
        df_test = self.df.copy()
        # Injeta sinal em micro-escala (1e-4) com target
        df_test["micro_scale_signal"] = (self.df["target_dir_15m"].values * 0.0002) + np.random.normal(0, 0.00005, len(self.df))

        model_cols = self.base_cols + ["micro_scale_signal"]
        res = self.evaluator.evaluate_cohort(
            df_test, self.base_cols, model_cols, "COHORT_MICRO_SCALE", horizon_bars=15, n_permutations=50
        )

        # Deve aprender sem que a L2 zere o coeficiente
        self.assertGreater(res.delta_oos_auc, 0.02)

    def test_future_leakage_detected(self):
        """Caso E (Future Leakage): Evaluator deve detectar violação anti-lookahead e rejeitar dataset."""
        df_test = self.df.copy()
        # Injeta feature artificial que vaza diretamente o target futuro (lookahead violado)
        df_test["future_leaked_target"] = self.df["target_dir_15m"].values * 3.0

        model_cols = self.base_cols + ["future_leaked_target"]
        res = self.evaluator.evaluate_cohort(
            df_test, self.base_cols, model_cols, "COHORT_FUTURE_LEAKAGE", horizon_bars=15, n_permutations=20
        )

        # Deve REJEITAR explicitamente por leakage, não aprovar por AUC
        self.assertTrue(res.leakage_detected)
        self.assertEqual(res.verdict, "REJECTED_FUTURE_LEAKAGE")
        self.assertGreater(len(res.leakage_reasons), 0)

    def test_high_autocorrelation_effective_n(self):
        """Caso F (High Autocorrelation): Evaluator deve reportar dependência temporal e não inflar effective N."""
        df_test = self.df.copy()
        # Injeta série cumulativa com persistência extrema (random walk integrado, rho_1 >= 0.90)
        df_test["high_ar_feature"] = np.cumsum(np.random.normal(0, 1.0, len(self.df)))

        model_cols = self.base_cols + ["high_ar_feature"]
        res = self.evaluator.evaluate_cohort(
            df_test, self.base_cols, model_cols, "COHORT_HIGH_AUTOCORRELATION", horizon_bars=15, n_permutations=20
        )

        # Deve acusar dependência temporal
        self.assertTrue(res.temporal_dependency_detected)
        self.assertIsNotNone(res.autocorrelation_lag1)
        self.assertGreaterEqual(res.autocorrelation_lag1, 0.70)
        # N_eff deve ser severamente penalizado em relação a raw_n (não inflado)
        self.assertIsNotNone(res.effective_n)
        self.assertLess(res.effective_n, res.raw_n // 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

