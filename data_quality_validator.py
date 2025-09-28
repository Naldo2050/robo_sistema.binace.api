import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

class DataQualityValidator:
    """
    Valida a qualidade de uma janela de dados de trade (aggTrade).
    """
    def __init__(self, thresholds: Dict[str, Any] = None):
        """
        Inicializa o validador com thresholds personalizáveis.

        Args:
            thresholds (dict): Dicionário com limiares para as validações.
        """
        default_thresholds = {
            "max_price_jump_std": 5.0,      # Desvio padrão máximo para saltos de preço
            "max_volume_jump_std": 10.0,     # Desvio padrão máximo para picos de volume
            "min_completeness_pct": 0.90,   # Percentual mínimo de dados esperados na janela
            "max_zero_volume_pct": 0.10,    # Percentual máximo de trades com volume zero
            "max_time_gap_seconds": 10      # Gap máximo em segundos entre trades consecutivos
        }
        self.thresholds = thresholds or default_thresholds
        logging.info("✅ DataQualityValidator inicializado.")

    def validate_window(self, df_window: pd.DataFrame, window_duration_seconds: int) -> Dict[str, Any]:
        """
        Executa todas as validações para uma janela de dados e retorna um score de qualidade.

        Args:
            df_window (pd.DataFrame): DataFrame com os dados da janela ('p', 'q', 'T').
            window_duration_seconds (int): Duração esperada da janela em segundos.

        Returns:
            dict: Dicionário com o score de qualidade, uma flag de validade e os problemas encontrados.
        """
        if df_window.empty or len(df_window) < 2:
            return {
                "is_valid": False,
                "quality_score": 0,
                "issues": ["DataFrame vazio ou com menos de 2 trades"],
                "metrics": {}
            }

        issues = []
        score = 100
        metrics = {}

        # 1. Validação de Saltos de Preço (Price Jumps)
        price_returns = df_window['p'].pct_change().dropna()
        if not price_returns.empty:
            price_std = price_returns.std()
            price_jumps = price_returns[price_returns.abs() > self.thresholds["max_price_jump_std"] * price_std]
            if not price_jumps.empty:
                issues.append(f"Detectados {len(price_jumps)} saltos de preço anormais.")
                score -= 25
            metrics["price_return_std"] = price_std

        # 2. Validação de Picos de Volume (Volume Spikes)
        volume_std = df_window['q'].std()
        if volume_std > 0:
            volume_spikes = df_window[df_window['q'] > self.thresholds["max_volume_jump_std"] * volume_std]
            if not volume_spikes.empty:
                issues.append(f"Detectados {len(volume_spikes)} picos de volume anormais.")
                score -= 15
            metrics["volume_std"] = volume_std

        # 3. Validação de Gaps Temporais
        time_gaps = df_window['T'].diff().dropna() / 1000  # em segundos
        large_gaps = time_gaps[time_gaps > self.thresholds["max_time_gap_seconds"]]
        if not large_gaps.empty:
            issues.append(f"Detectados {len(large_gaps)} gaps temporais > {self.thresholds['max_time_gap_seconds']}s.")
            score -= 30
        metrics["max_time_gap_seconds"] = time_gaps.max() if not time_gaps.empty else 0

        # 4. Validação de Volume Zero
        zero_volume_trades = df_window[df_window['q'] == 0]
        zero_vol_pct = len(zero_volume_trades) / len(df_window) if len(df_window) > 0 else 0
        if zero_vol_pct > self.thresholds["max_zero_volume_pct"]:
            issues.append(f"Percentual de trades com volume zero ({zero_vol_pct:.1%}) excedeu o limite.")
            score -= 10
        metrics["zero_volume_pct"] = zero_vol_pct

        # Score final
        final_score = max(0, score)

        return {
            "is_valid": final_score > 70,  # Considera a janela válida se o score for > 70
            "quality_score": final_score,
            "issues": issues,
            "metrics": metrics
        }