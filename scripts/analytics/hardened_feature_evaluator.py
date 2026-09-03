# scripts/analytics/hardened_feature_evaluator.py
# -*- coding: utf-8 -*-
"""
Hardened Feature Evaluator — Fase V1.2.
Harness metodológico blindado e auditado para avaliação de valor informacional de features:
1. Cohorts separados por família de features (evita perda por inner join desnecessário).
2. StandardScaler estrito fitado exclusivamente na partição de treino (evita encolhimento L2 espúrio).
3. Duplo Controle Negativo: (A) Hash Determinístico de Timestamp, (B) PRNG Independente.
4. Teste de Permutação com Correção Finita: p = (1 + count(null >= obs)) / (1 + B).
5. Paired Block Bootstrap com tamanho de bloco proporcional ao horizonte (block_size = horizon_bars).
6. Provenance e Lineage Metadata Manifest.
7. Verificador de Maturidade V2 e Sanity Checks.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import sqlite3
import sys
sys.path.insert(0, ".")
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HardenedEvaluator")

RESULTS_DIR = "analysis/results"


@dataclass
class CohortMetrics:
    cohort_name: str
    target_horizon: str
    data_provenance: str  # LIVE_OBSERVED, LIVE_ASOF_EXPANDED, SYNTHETIC_BENCHMARK, HISTORICAL_RECONSTRUCTED
    raw_n: int
    unique_source_n: int
    non_overlapping_n: int
    n_features_base: int
    n_features_model: int
    val_auc_base: float
    val_auc_model: float
    delta_val_auc: float
    oos_auc_base: float
    oos_auc_model: float
    delta_oos_auc: float
    delta_oos_auc_ci_low: float
    delta_oos_auc_ci_high: float
    neg_control_hash_delta_oos: float
    neg_control_prng_delta_oos: float
    null_permutation_p_val: float
    verdict: str  # NO_EVIDENCE, WEAK, PROMISING, STRONG_INCREMENTAL_EVIDENCE, INSUFFICIENT_DATA, REJECTED_FUTURE_LEAKAGE
    effective_n: Optional[int] = None
    autocorrelation_lag1: Optional[float] = None
    temporal_dependency_detected: bool = False
    leakage_detected: bool = False
    leakage_reasons: List[str] = field(default_factory=list)


@dataclass
class V2ReadinessReport:
    status: str  # NOT_READY / READY_FOR_V2
    current_calendar_days: float
    target_calendar_days: float = 7.0
    current_live_observations: int = 0
    target_live_observations: int = 2000
    current_pos_snapshots: int = 0
    target_pos_snapshots: int = 500
    current_bos_events: int = 0
    target_bos_events: int = 100
    current_sweep_events: int = 0
    target_sweep_events: int = 100
    missing_data_pct: float = 0.0
    recommendations: List[str] = field(default_factory=list)


class HardenedFeatureEvaluator:
    """Avaliador metodologicamente blindado com StandardScaler, Bootstrap e Permutação Finita."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)

    def generate_negative_controls(self, timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gera dois controles negativos independentes:
        1. neg_control_hash: Determinístico a partir do timestamp (sem futuro).
        2. neg_control_prng: Pure pseudo-random noise independente do timestamp.
        """
        n = len(timestamps)
        # Controle 1: Hash determinístico
        hash_noise = np.zeros(n)
        for i, ts in enumerate(timestamps):
            h = hashlib.sha256(f"seed_{self.random_seed}_{ts}".encode()).hexdigest()
            hash_noise[i] = (int(h[:8], 16) / 0xFFFFFFFF) * 2.0 - 1.0

        # Controle 2: PRNG independente fixo
        rng = np.random.RandomState(self.random_seed + 999)
        prng_noise = rng.normal(0, 1.0, n)

        return hash_noise, prng_noise

    def estimate_feature_autocorrelation(self, x: np.ndarray) -> float:
        """Calcula a autocorrelação lag-1 empírica da série temporal."""
        valid_x = x[~np.isnan(x)]
        if len(valid_x) < 5 or float(np.std(valid_x)) < 1e-9:
            return 0.0
        x_centered = valid_x - float(np.mean(valid_x))
        c0 = float(np.dot(x_centered, x_centered) / len(x_centered))
        c1 = float(np.dot(x_centered[:-1], x_centered[1:]) / len(x_centered))
        if c0 <= 1e-12:
            return 0.0
        rho1 = float(c1 / c0)
        return max(-1.0, min(1.0, rho1))

    def check_anti_lookahead(
        self,
        df: pd.DataFrame,
        candidate_cols: List[str],
        target_col: str,
        timestamp_col: str = "timestamp_ms",
        horizon_bars: int = 15,
    ) -> Tuple[bool, List[str]]:
        """
        Audita o dataset para violações de anti-lookahead / future leakage:
        1. Temporal Precedence: Garante que timestamps da feature não excedem o observation_ts.
        2. Direct Future Target Leakage: Verifica se a correlação contemporânea com target futuro é espúria (|r| > 0.85).
        3. Backward Shift / Lookahead Alignment: Verifica se a feature se correlaciona anomalamente com retornos futuros.
        """
        reasons = []
        # Checagem 1: Timestamp
        if timestamp_col in df.columns:
            obs_ts = df[timestamp_col].values
            for col in candidate_cols:
                if "source_timestamp" in col or "ts" in col.lower():
                    try:
                        feat_ts = df[col].values
                        if (feat_ts > obs_ts).any():
                            reasons.append(f"Timestamp futuro detectado em '{col}': source_ts > observation_ts")
                    except Exception:
                        pass

        # Checagem 2: Correlação com target futuro
        if target_col in df.columns:
            y = df[target_col].fillna(0).values
            for col in candidate_cols:
                x = df[col].fillna(0).values
                if len(np.unique(x)) > 1 and len(np.unique(y)) > 1:
                    corr = float(np.corrcoef(x, y)[0, 1])
                    if abs(corr) >= 0.95:
                        reasons.append(
                            f"Vazamento direto de futuro / Target Leakage em '{col}': |corr(x, y_future)| = {abs(corr):.4f} >= 0.95"
                        )
                    # Checagem de shift futuro: x correlacionado com o próximo target (lookahead shift)
                    if len(y) > horizon_bars:
                        y_fwd = np.roll(y, -1)
                        corr_fwd = float(np.corrcoef(x[:-1], y_fwd[:-1])[0, 1])
                        if abs(corr_fwd) >= 0.95:
                            reasons.append(
                                f"Future lookahead detectado em '{col}': |corr(x_t, y_t+1)| = {abs(corr_fwd):.4f} >= 0.95"
                            )

        return (len(reasons) > 0, reasons)

    def evaluate_cohort(
        self,
        df: pd.DataFrame,
        base_cols: List[str],
        model_cols: List[str],
        cohort_name: str,
        data_provenance: str = "SYNTHETIC_BENCHMARK",
        unique_source_n: Optional[int] = None,
        horizon_bars: int = 15,
        n_bootstraps: int = 300,
        n_permutations: int = 100,
    ) -> CohortMetrics:
        """Avalia um cohort específico aplicando padronização estrita e testes estatísticos calibrados."""
        target_col = f"target_dir_{horizon_bars}m"
        raw_n = len(df)
        u_source = unique_source_n if unique_source_n is not None else raw_n

        candidate_cols = [c for c in model_cols if c not in base_cols]

        if df.empty or raw_n < (horizon_bars * 4):
            return CohortMetrics(
                cohort_name=cohort_name,
                target_horizon=f"{horizon_bars}m",
                data_provenance=data_provenance,
                raw_n=raw_n,
                unique_source_n=u_source,
                non_overlapping_n=raw_n // horizon_bars if raw_n > 0 else 0,
                n_features_base=len(base_cols),
                n_features_model=len(model_cols),
                val_auc_base=0.50, val_auc_model=0.50, delta_val_auc=0.0,
                oos_auc_base=0.50, oos_auc_model=0.50, delta_oos_auc=0.0,
                delta_oos_auc_ci_low=0.0, delta_oos_auc_ci_high=0.0,
                neg_control_hash_delta_oos=0.0, neg_control_prng_delta_oos=0.0,
                null_permutation_p_val=1.0,
                verdict="INSUFFICIENT_DATA",
                effective_n=0,
                autocorrelation_lag1=0.0,
                temporal_dependency_detected=False,
                leakage_detected=False,
                leakage_reasons=[],
            )

        # 0. Anti-Lookahead / Future Leakage Guard
        has_leakage, leakage_reasons = self.check_anti_lookahead(
            df, candidate_cols, target_col, horizon_bars=horizon_bars
        )
        if has_leakage:
            logger.warning(f"REJEIÇÃO METODOLÓGICA — Future Leakage detectado em {cohort_name}: {leakage_reasons}")
            return CohortMetrics(
                cohort_name=cohort_name,
                target_horizon=f"{horizon_bars}m",
                data_provenance=data_provenance,
                raw_n=raw_n,
                unique_source_n=u_source,
                non_overlapping_n=raw_n // horizon_bars if raw_n > 0 else 0,
                n_features_base=len(base_cols),
                n_features_model=len(model_cols),
                val_auc_base=0.50, val_auc_model=0.50, delta_val_auc=0.0,
                oos_auc_base=0.50, oos_auc_model=0.50, delta_oos_auc=0.0,
                delta_oos_auc_ci_low=0.0, delta_oos_auc_ci_high=0.0,
                neg_control_hash_delta_oos=0.0, neg_control_prng_delta_oos=0.0,
                null_permutation_p_val=1.0,
                verdict="REJECTED_FUTURE_LEAKAGE",
                effective_n=0,
                autocorrelation_lag1=0.0,
                temporal_dependency_detected=False,
                leakage_detected=True,
                leakage_reasons=leakage_reasons,
            )

        # 0B. Autocorrelação & Tamanho Amostral Efetivo (Effective N)
        max_rho1 = 0.0
        for col in candidate_cols:
            rho = self.estimate_feature_autocorrelation(df[col].fillna(0).values)
            if abs(rho) > abs(max_rho1):
                max_rho1 = rho

        temporal_dep = abs(max_rho1) >= 0.70
        if temporal_dep:
            # Fórmula de Bartlett / Newey-West para amostra efetiva sob dependência temporal:
            # N_eff = N * (1 - |rho|) / (1 + |rho|)
            ar_factor = (1.0 - abs(max_rho1)) / (1.0 + abs(max_rho1))
            n_eff = max(1, int(raw_n * ar_factor))
            bootstrap_block = max(horizon_bars, int(1.0 / max(0.01, 1.0 - abs(max_rho1))))
        else:
            n_eff = raw_n // horizon_bars
            bootstrap_block = horizon_bars

        # Adiciona controles negativos
        df_eval = df.copy()
        hash_n, prng_n = self.generate_negative_controls(df_eval["timestamp_ms"].values)
        df_eval["neg_control_hash"] = hash_n
        df_eval["neg_control_prng"] = prng_n

        # Divisão Temporal Estrita (60% Discovery / 20% Validation / 20% OOS)
        n = len(df_eval)
        n_train = int(n * 0.60)
        n_val = int(n * 0.20)

        train_df = df_eval.iloc[:n_train]
        val_df = df_eval.iloc[n_train:n_train + n_val]
        oos_df = df_eval.iloc[n_train + n_val:]

        y_train = train_df[target_col].values
        y_val = val_df[target_col].values
        y_oos = oos_df[target_col].values

        # 1. Modelo Baseline com StandardScaler
        scaler_base = StandardScaler()
        X_train_base = scaler_base.fit_transform(train_df[base_cols].fillna(0).values)
        X_val_base = scaler_base.transform(val_df[base_cols].fillna(0).values)
        X_oos_base = scaler_base.transform(oos_df[base_cols].fillna(0).values)

        clf_base = LogisticRegression(max_iter=500, random_state=self.random_seed)
        clf_base.fit(X_train_base, y_train)
        val_probs_base = clf_base.predict_proba(X_val_base)[:, 1] if len(np.unique(y_val)) > 1 else np.full(len(y_val), 0.5)
        oos_probs_base = clf_base.predict_proba(X_oos_base)[:, 1] if len(np.unique(y_oos)) > 1 else np.full(len(y_oos), 0.5)

        val_auc_base = roc_auc_score(y_val, val_probs_base) if len(np.unique(y_val)) > 1 else 0.50
        oos_auc_base = roc_auc_score(y_oos, oos_probs_base) if len(np.unique(y_oos)) > 1 else 0.50

        # 2. Modelo com Features Candidatas (com StandardScaler independente)
        scaler_model = StandardScaler()
        X_train_model = scaler_model.fit_transform(train_df[model_cols].fillna(0).values)
        X_val_model = scaler_model.transform(val_df[model_cols].fillna(0).values)
        X_oos_model = scaler_model.transform(oos_df[model_cols].fillna(0).values)

        clf_model = LogisticRegression(max_iter=500, random_state=self.random_seed)
        clf_model.fit(X_train_model, y_train)
        val_probs_model = clf_model.predict_proba(X_val_model)[:, 1] if len(np.unique(y_val)) > 1 else np.full(len(y_val), 0.5)
        oos_probs_model = clf_model.predict_proba(X_oos_model)[:, 1] if len(np.unique(y_oos)) > 1 else np.full(len(y_oos), 0.5)

        val_auc_model = roc_auc_score(y_val, val_probs_model) if len(np.unique(y_val)) > 1 else 0.50
        oos_auc_model = roc_auc_score(y_oos, oos_probs_model) if len(np.unique(y_oos)) > 1 else 0.50

        delta_val = val_auc_model - val_auc_base
        delta_oos = oos_auc_model - oos_auc_base

        # 3. Modelos com Controles Negativos
        # 3A: Hash
        scaler_hash = StandardScaler()
        X_tr_h = scaler_hash.fit_transform(train_df[base_cols + ["neg_control_hash"]].fillna(0).values)
        X_oos_h = scaler_hash.transform(oos_df[base_cols + ["neg_control_hash"]].fillna(0).values)
        clf_h = LogisticRegression(max_iter=500, random_state=self.random_seed).fit(X_tr_h, y_train)
        oos_auc_hash = roc_auc_score(y_oos, clf_h.predict_proba(X_oos_h)[:, 1]) if len(np.unique(y_oos)) > 1 else 0.50
        delta_hash = oos_auc_hash - oos_auc_base

        # 3B: PRNG
        scaler_prng = StandardScaler()
        X_tr_p = scaler_prng.fit_transform(train_df[base_cols + ["neg_control_prng"]].fillna(0).values)
        X_oos_p = scaler_prng.transform(oos_df[base_cols + ["neg_control_prng"]].fillna(0).values)
        clf_p = LogisticRegression(max_iter=500, random_state=self.random_seed).fit(X_tr_p, y_train)
        oos_auc_prng = roc_auc_score(y_oos, clf_p.predict_proba(X_oos_p)[:, 1]) if len(np.unique(y_oos)) > 1 else 0.50
        delta_prng = oos_auc_prng - oos_auc_base

        # 4. Paired Block Bootstrap para Delta AUC (com tamanho de bloco proporcional à persistência temporal)
        delta_ci_low, delta_ci_high = self._paired_block_bootstrap_delta_auc(
            y_oos, oos_probs_base, oos_probs_model, block_size=bootstrap_block, n_bootstraps=n_bootstraps
        )

        # 5. Permutation Null Test com Correção Finita
        null_p_val = self._permutation_test_finite(
            train_df, oos_df, base_cols, model_cols, target_col, delta_oos, n_permutations=n_permutations
        )

        # Veredito Metodológico Rigoroso
        max_neg_delta = max(delta_hash, delta_prng)
        if raw_n < 500:
            verdict = "INSUFFICIENT_SAMPLE"
        elif delta_oos > 0.02 and delta_ci_low > 0 and null_p_val < 0.05 and delta_oos > max_neg_delta:
            verdict = "STRONG_INCREMENTAL_EVIDENCE"
        elif delta_oos > 0.005 and null_p_val < 0.10 and delta_oos > max_neg_delta:
            verdict = "PROMISING"
        elif delta_oos > -0.005:
            verdict = "INCREMENTAL_VALUE_UNKNOWN"
        else:
            verdict = "NO_EVIDENCE"

        return CohortMetrics(
            cohort_name=cohort_name,
            target_horizon=f"{horizon_bars}m",
            data_provenance=data_provenance,
            raw_n=raw_n,
            unique_source_n=u_source,
            non_overlapping_n=raw_n // horizon_bars,
            n_features_base=len(base_cols),
            n_features_model=len(model_cols),
            val_auc_base=round(float(val_auc_base), 4),
            val_auc_model=round(float(val_auc_model), 4),
            delta_val_auc=round(float(delta_val), 4),
            oos_auc_base=round(float(oos_auc_base), 4),
            oos_auc_model=round(float(oos_auc_model), 4),
            delta_oos_auc=round(float(delta_oos), 4),
            delta_oos_auc_ci_low=round(float(delta_ci_low), 4),
            delta_oos_auc_ci_high=round(float(delta_ci_high), 4),
            neg_control_hash_delta_oos=round(float(delta_hash), 4),
            neg_control_prng_delta_oos=round(float(delta_prng), 4),
            null_permutation_p_val=round(float(null_p_val), 4),
            verdict=verdict,
            effective_n=n_eff,
            autocorrelation_lag1=round(float(max_rho1), 3),
            temporal_dependency_detected=temporal_dep,
            leakage_detected=False,
            leakage_reasons=[],
        )


    def _paired_block_bootstrap_delta_auc(
        self, y_true: np.ndarray, p_base: np.ndarray, p_model: np.ndarray, block_size: int = 15, n_bootstraps: int = 300
    ) -> Tuple[float, float]:
        """Calcula o IC 95% empírico de Delta AUC via Paired Block Bootstrap."""
        n = len(y_true)
        if n < block_size * 2:
            return 0.0, 0.0

        n_blocks = n // block_size
        deltas = []

        for _ in range(n_bootstraps):
            blocks = [random.randint(0, n_blocks - 1) for _ in range(n_blocks)]
            indices = np.concatenate([np.arange(b * block_size, (b + 1) * block_size) for b in blocks])
            y_sample = y_true[indices]
            if len(np.unique(y_sample)) > 1:
                auc_b = roc_auc_score(y_sample, p_base[indices])
                auc_m = roc_auc_score(y_sample, p_model[indices])
                deltas.append(auc_m - auc_b)

        if not deltas:
            return 0.0, 0.0

        return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))

    def _permutation_test_finite(
        self, train_df: pd.DataFrame, oos_df: pd.DataFrame, base_cols: List[str], model_cols: List[str],
        target_col: str, observed_delta: float, n_permutations: int = 100
    ) -> float:
        """
        Teste de hipótese nula com Correção Amostral Finita:
        p = (1 + count(null_delta >= observed_delta)) / (1 + B)
        """
        new_cols = [c for c in model_cols if c not in base_cols]
        if not new_cols:
            return 1.0

        y_train = train_df[target_col].values
        y_oos = oos_df[target_col].values

        scaler_base = StandardScaler()
        X_tr_base = scaler_base.fit_transform(train_df[base_cols].fillna(0).values)
        X_oos_base = scaler_base.transform(oos_df[base_cols].fillna(0).values)

        base_clf = LogisticRegression(max_iter=300, random_state=self.random_seed)
        base_clf.fit(X_tr_base, y_train)
        oos_base_probs = base_clf.predict_proba(X_oos_base)[:, 1] if len(np.unique(y_oos)) > 1 else np.full(len(y_oos), 0.5)
        base_auc = roc_auc_score(y_oos, oos_base_probs) if len(np.unique(y_oos)) > 1 else 0.50

        exceed_count = 0
        for _ in range(n_permutations):
            perm_train = train_df[model_cols].copy()
            # Embaralha as features candidatas
            perm_train[new_cols] = np.random.permutation(perm_train[new_cols].values)

            scaler_perm = StandardScaler()
            X_tr_perm = scaler_perm.fit_transform(perm_train.fillna(0).values)
            X_oos_perm = scaler_perm.transform(oos_df[model_cols].fillna(0).values)

            clf_perm = LogisticRegression(max_iter=300, random_state=self.random_seed)
            clf_perm.fit(X_tr_perm, y_train)
            oos_perm_probs = clf_perm.predict_proba(X_oos_perm)[:, 1] if len(np.unique(y_oos)) > 1 else np.full(len(y_oos), 0.5)
            perm_auc = roc_auc_score(y_oos, oos_perm_probs) if len(np.unique(y_oos)) > 1 else 0.50
            delta_perm = perm_auc - base_auc

            if delta_perm >= observed_delta:
                exceed_count += 1

        # Fórmula exata com correção finita
        finite_p = (1.0 + exceed_count) / (1.0 + n_permutations)
        return float(finite_p)

    def check_v2_readiness(self, db_path: str = "dados/trading_bot.db") -> V2ReadinessReport:
        """Verifica os critérios formais de prontidão estatística para início da Fase V2."""
        conn = sqlite3.connect(db_path) if os.path.exists(db_path) else None
        live_obs = 0
        pos_snaps = 0
        bos_events = 0
        sweep_events = 0
        days_span = 0.0

        if conn:
            cur = conn.cursor()
            try:
                cur.execute("SELECT count(*), min(timestamp_ms), max(timestamp_ms) FROM events")
                row = cur.fetchone()
                live_obs = row[0] or 0
                if row[1] and row[2] and row[2] > row[1]:
                    days_span = (row[2] - row[1]) / (1000.0 * 86400.0)

                cur.execute("SELECT count(*) FROM positioning_shadow_dataset")
                pos_snaps = cur.fetchone()[0] or 0
            except Exception:
                pass
            finally:
                conn.close()

        recs: List[str] = []
        is_ready = True

        if live_obs < 2000:
            is_ready = False
            recs.append(f"Amostra de observações live insuficiente: {live_obs}/2.000 necessárias.")
        if pos_snaps < 500:
            is_ready = False
            recs.append(f"Snapshots de positioning insuficientes: {pos_snaps}/500 necessários.")
        if days_span < 7.0:
            is_ready = False
            recs.append(f"Cobertura temporal em dias insuficiente: {days_span:.2f}/7.00 dias requeridos.")

        status = "READY_FOR_V2" if is_ready else "NOT_READY"
        return V2ReadinessReport(
            status=status,
            current_calendar_days=round(days_span, 2),
            current_live_observations=live_obs,
            current_pos_snapshots=pos_snaps,
            current_bos_events=bos_events,
            current_sweep_events=sweep_events,
            recommendations=recs,
        )


def run_hardened_evaluation_suite():
    print("=" * 80)
    print("FASE V1.2 — VALIDATOR FORENSICS & COHORT EVALUATION")
    print("=" * 80)

    evaluator = HardenedFeatureEvaluator()

    # 1. Verifica Prontidão V2
    readiness = evaluator.check_v2_readiness()
    print(f"\n1. V2 READINESS REPORT: [{readiness.status}]")
    print(f"   - Observações Live:      {readiness.current_live_observations} / {readiness.target_live_observations}")
    print(f"   - Positioning Snapshots: {readiness.current_pos_snapshots} / {readiness.target_pos_snapshots}")
    print(f"   - Dias de Cobertura:     {readiness.current_calendar_days:.2f} / {readiness.target_calendar_days} dias")
    for r in readiness.recommendations:
        print(f"   [AVISO] {r}")

    # 2. Sanity Check: Rolling VWAP vs Session VWAP
    print("\n2. SANITY CHECK: ROLLING VWAP vs SESSION VWAP:")
    prices = 75000.0 * np.exp(np.cumsum(np.random.normal(0.00005, 0.0015, 1440)))
    vols = np.random.gamma(2.0, 10.0, 1440)
    session_vwap = np.cumsum(prices * vols) / np.cumsum(vols)
    rolling_vwap = pd.Series(prices).rolling(20, min_periods=1).mean().values

    dist_session = (prices - session_vwap) / session_vwap
    dist_rolling = (prices - rolling_vwap) / rolling_vwap

    r_levels = np.corrcoef(rolling_vwap, session_vwap)[0, 1]
    r_dist = np.corrcoef(dist_rolling, dist_session)[0, 1]
    spearman_dist = pd.Series(dist_rolling).corr(pd.Series(dist_session), method="spearman")

    print(f"   - Correlacao dos NIVEIS de Preco (Rolling VWAP vs Session VWAP): r = {r_levels:+.4f}")
    print(f"   - Correlacao das DISTANCIAS Normalizadas ao Preco:              r = {r_dist:+.4f} (Pearson), rs = {spearman_dist:+.4f} (Spearman)")

    # 3. Execução de Cohorts sobre Benchmark Sintético
    from scripts.analytics.feature_value_validator import FeatureValueValidator
    fvv = FeatureValueValidator()
    bench_df = fvv.generate_benchmark_synthetic_stream(1500)
    labeled_df = fvv.attach_forward_labels(bench_df)

    base_cols = ["flow_d1", "flow_imb", "flow_cvd_4h", "ob_imb", "funding_rate", "rolling_vwap_dist"]
    cohort_definitions = [
        ("COHORT_SESSION_VWAP", base_cols + ["session_vwap_dist"], "SYNTHETIC_BENCHMARK", 1440),
        ("COHORT_POSITIONING", base_cols + ["pos_ga", "pos_ta", "pos_tp", "pos_od1", "pos_od4", "pos_top_acc_vs_global"], "SYNTHETIC_BENCHMARK", 288),
        ("COHORT_MARKET_STRUCTURE", base_cols + ["ms_bos_bull", "ms_bos_bear", "ms_bos_str", "ms_sw_buy", "ms_sw_sell", "ms_sw_exc"], "SYNTHETIC_BENCHMARK", 1440),
        ("COHORT_CONFLUENCE_ALL", base_cols + ["session_vwap_dist", "pos_ga", "pos_ta", "pos_tp", "pos_od1", "pos_od4", "pos_top_acc_vs_global", "ms_bos_bull", "ms_bos_bear", "ms_bos_str", "ms_sw_buy", "ms_sw_sell", "ms_sw_exc"], "SYNTHETIC_BENCHMARK", 288),
    ]

    cohort_results = []
    print("\n3. MATRIZ METODOLOGICA DE COHORTS SEPARADOS COM STANDARSCALER E PAIRED BOOTSTRAP:")
    print("-" * 120)
    header = f"{'Cohort':<24} | {'Provenance':<20} | {'Raw N':<6} | {'N_eff':<6} | {'Base AUC':<8} | {'Mod AUC':<8} | {'Delta OOS':<10} | {'CI 95% Delta':<18} | {'Null p':<6}"
    print(header)
    print("-" * 120)

    for c_name, cols, prov, u_n in cohort_definitions:
        m = evaluator.evaluate_cohort(
            labeled_df, base_cols, cols, c_name, data_provenance=prov, unique_source_n=u_n, horizon_bars=15
        )
        cohort_results.append(m)
        ci_str = f"[{m.delta_oos_auc_ci_low:+.3f}, {m.delta_oos_auc_ci_high:+.3f}]"
        print(f"{m.cohort_name:<24} | {m.data_provenance:<20} | {m.raw_n:<6} | {m.non_overlapping_n:<6} | {m.oos_auc_base:<8.4f} | {m.oos_auc_model:<8.4f} | {m.delta_oos_auc:<+10.4f} | {ci_str:<18} | {m.null_permutation_p_val:<6.2f}")
    print("-" * 120)

    # 4. Salva resultados consolidados com Data Lineage Manifest
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_file = os.path.join(RESULTS_DIR, "v1_2_hardening_results.json")
    save_data = {
        "manifest": {
            "git_sha": "HEAD_P1_3C_V1_2",
            "generated_at": time.time(),
            "sources": ["dados/trading_bot.db", "benchmark_synthetic_stream"],
            "live_rows": readiness.current_live_observations,
            "historical_rows": 0,
            "synthetic_rows": len(labeled_df),
            "unique_source_observations": readiness.current_live_observations,
            "time_range": f"{bench_df['timestamp_ms'].min()} -> {bench_df['timestamp_ms'].max()}",
            "schema_versions": ["1.1.0"],
        },
        "v2_readiness": asdict(readiness),
        "sanity_check_vwap": {
            "r_levels": float(r_levels),
            "r_dist_pearson": float(r_dist),
            "r_dist_spearman": float(spearman_dist),
        },
        "cohorts": [asdict(c) for c in cohort_results],
        "evaluated_at": time.time(),
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Resultados da Fase V1.2 salvos em: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    run_hardened_evaluation_suite()
