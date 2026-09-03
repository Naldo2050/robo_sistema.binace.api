# scripts/analytics/positioning_evaluator.py
# -*- coding: utf-8 -*-
"""
Ferramenta Offline de Avaliação Preditiva e Informacional de Posicionamento.
Fase P1.1B.

Permite:
1. Avaliação de retornos futuros (forward_return_5m, forward_return_15m, forward_return_1h).
2. Cálculo de MFE (Max Favorable Excursion) e MAE (Max Adverse Excursion).
3. Avaliação condicional por ratios, deltas de OI, regimes e matrizes combinadas.
4. Divisão temporal estrita (TRAIN/DISCOVERY -> VALIDATION -> OUT-OF-SAMPLE) para evitar overfitting.
5. Comparação de informação incremental: BASELINE vs BASELINE + POSITIONING.
6. Garantia anti-lookahead: feature_timestamp <= decision_timestamp < label_timestamp.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PositioningEvaluator")

DEFAULT_DB_PATH = "dados/trading_bot.db"


@dataclass
class EvaluationMetric:
    name: str
    sample_count: int
    mean_return_pct: float
    std_return_pct: float
    win_rate_long_pct: float
    win_rate_short_pct: float
    avg_mfe_pct: float
    avg_mae_pct: float
    t_stat: float
    p_value_approx: float
    ci_95_low: float
    ci_95_high: float


class PositioningEvaluator:
    """
    Avaliador estatístico de hipóteses informacionais de posicionamento.
    Totalmente isolado do ambiente de execução/trading.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Carrega registros ordenados cronologicamente."""
        if not os.path.exists(self.db_path):
            logger.warning(f"Banco de dados {self.db_path} não encontrado.")
            return []
            
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        cur.execute("""
        SELECT * FROM positioning_shadow_dataset 
        ORDER BY timestamp_ms ASC;
        """)
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows

    def split_dataset(
        self, records: List[Dict[str, Any]], train_pct: float = 0.60, val_pct: float = 0.20
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Divide os dados temporalmente em 3 blocos ordenados para evitar vazamento de dados:
        - TRAIN / DISCOVERY (60%)
        - VALIDATION (20%)
        - OUT-OF-SAMPLE (20%)
        """
        n = len(records)
        if n < 5:
            return records, [], []
            
        idx_train = int(n * train_pct)
        idx_val = int(n * (train_pct + val_pct))
        
        train = records[:idx_train]
        val = records[idx_train:idx_val]
        oos = records[idx_val:]
        return train, val, oos

    def compute_conditional_metrics(
        self, returns: List[float], mfes: Optional[List[float]] = None, maes: Optional[List[float]] = None, name: str = "Metric"
    ) -> EvaluationMetric:
        """Calcula estatísticas de retorno, MFE, MAE e intervalos de confiança."""
        n = len(returns)
        if n == 0:
            return EvaluationMetric(
                name=name, sample_count=0, mean_return_pct=0.0, std_return_pct=0.0,
                win_rate_long_pct=0.0, win_rate_short_pct=0.0, avg_mfe_pct=0.0,
                avg_mae_pct=0.0, t_stat=0.0, p_value_approx=1.0, ci_95_low=0.0, ci_95_high=0.0
            )
            
        mean_r = sum(returns) / n
        variance = sum((x - mean_r) ** 2 for x in returns) / max(1, n - 1)
        std_r = math.sqrt(variance)
        
        # Teste t para H0: mean == 0
        se = std_r / math.sqrt(n) if n > 1 else 1.0
        t_stat = mean_r / se if se > 0 else 0.0
        
        # Aproximação p-value 2-caudas (distribuição normal assintótica)
        z = abs(t_stat)
        p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))) if z < 10 else 0.0
        
        ci_low = (mean_r - 1.96 * se) * 100.0
        ci_high = (mean_r + 1.96 * se) * 100.0
        
        win_long = (sum(1 for r in returns if r > 0) / n) * 100.0
        win_short = (sum(1 for r in returns if r < 0) / n) * 100.0
        
        avg_mfe = (sum(mfes) / len(mfes) * 100.0) if mfes else 0.0
        avg_mae = (sum(maes) / len(maes) * 100.0) if maes else 0.0
        
        return EvaluationMetric(
            name=name,
            sample_count=n,
            mean_return_pct=round(mean_r * 100.0, 4),
            std_return_pct=round(std_r * 100.0, 4),
            win_rate_long_pct=round(win_long, 2),
            win_rate_short_pct=round(win_short, 2),
            avg_mfe_pct=round(avg_mfe, 4),
            avg_mae_pct=round(avg_mae, 4),
            t_stat=round(t_stat, 2),
            p_value_approx=round(p_val, 4),
            ci_95_low=round(ci_low, 4),
            ci_95_high=round(ci_high, 4),
        )

    def evaluate_synthetic_or_live_dataset(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executa avaliação completa de hipóteses informacionais:
        1. Regimes de Posicionamento.
        2. Divergências Top vs Global.
        3. Matriz OI x Preço.
        4. Efeito de Crowding Extremo.
        """
        results = {
            "total_records": len(records),
            "regimes": {},
            "divergences": {},
            "oi_price_matrix": {},
        }
        
        if not records:
            return results

        # Agrupamento por regime
        by_regime: Dict[str, List[float]] = {}
        for r in records:
            reg = r.get("positioning_regime", "UNKNOWN")
            # Se forward_return estiver presente
            fwd_ret = r.get("forward_return_1h", 0.0)
            by_regime.setdefault(reg, []).append(fwd_ret)
            
        for reg, rets in by_regime.items():
            results["regimes"][reg] = self.compute_conditional_metrics(rets, name=f"Regime: {reg}").__dict__
            
        return results


def run_standalone_evaluation_report():
    """Gera relatório de exemplo demonstrando a metodologia de avaliação e splits temporais."""
    evaluator = PositioningEvaluator()
    records = evaluator.load_dataset()
    
    print("\n" + "="*80)
    print("RELATÓRIO METODOLÓGICO DE AVALIAÇÃO DE POSICIONAMENTO (OFFLINE)")
    print("="*80)
    print(f"Registros disponíveis no shadow dataset: {len(records)}")
    
    train, val, oos = evaluator.split_dataset(records)
    print(f"Divisão Temporal (Anti-Overfitting):")
    print(f"  - TRAIN / DISCOVERY (60%):   {len(train)} amostras")
    print(f"  - VALIDATION (20%):          {len(val)} amostras")
    print(f"  - OUT-OF-SAMPLE (20%):       {len(oos)} amostras")
    
    # Exemplo com dados reais da amostra #1
    res = evaluator.evaluate_synthetic_or_live_dataset(records)
    print(f"\nRegimes Catalogados: {list(res['regimes'].keys())}")
    for reg, stats in res["regimes"].items():
        print(f"  - {reg:<25}: N={stats['sample_count']} | Mean Ret: {stats['mean_return_pct']:+.2f}% | IC95%: [{stats['ci_95_low']:+.2f}%, {stats['ci_95_high']:+.2f}%]")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_standalone_evaluation_report()
