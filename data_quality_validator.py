# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

class DataQualityValidator:
    """
    Valida qualidade de dados de trades E métricas calculadas.
    """
    def __init__(self, thresholds: Dict[str, Any] = None):
        default_thresholds = {
            # Thresholds originais (trades)
            "max_price_jump_std": 5.0,
            "max_volume_jump_std": 10.0,
            "min_completeness_pct": 0.90,
            "max_zero_volume_pct": 0.10,
            "max_time_gap_seconds": 10,
            
            # 🆕 NOVOS: Validação de features calculadas
            "min_orderbook_depth_usd": 10000,      # Mínimo $10k de liquidez
            "max_volume_sma_ratio": 500,           # Cap em 5x a média
            "min_value_area_range": 0.001,         # VAH - VAL > 0.1% do preço
            "max_flow_delta_divergence": 0.5,      # Máx divergência flow vs delta
            "min_tick_rule_variance": 0.01,        # tick_rule não pode ser sempre 0
        }
        self.thresholds = thresholds or default_thresholds
        logging.info("✅ DataQualityValidator inicializado (versão expandida).")

    def validate_window(self, df_window: pd.DataFrame, window_duration_seconds: int) -> Dict[str, Any]:
        """Validação original de trades (mantém código existente)"""
        # ... código original mantido ...
        pass

    # 🆕 NOVA FUNÇÃO: Valida métricas calculadas
    def validate_metrics(self, metrics: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """
        Valida métricas calculadas (orderbook, features, flow, etc.)
        
        Args:
            metrics: Dicionário com todas as métricas calculadas
            current_price: Preço atual para validações relativas
            
        Returns:
            dict: Resultado da validação com flags específicas
        """
        issues = []
        flags = []
        score = 100

        # ========================================
        # 1. VALIDAÇÃO DE ORDERBOOK
        # ========================================
        bid_depth = metrics.get('bid_depth_usd', 0)
        ask_depth = metrics.get('ask_depth_usd', 0)
        
        if bid_depth == 0 or ask_depth == 0:
            issues.append("❌ ORDERBOOK ZERADO (bid_depth ou ask_depth = $0)")
            flags.append("INVALID_ORDERBOOK")
            score -= 50  # Crítico
        elif bid_depth < self.thresholds['min_orderbook_depth_usd']:
            issues.append(f"⚠️ Liquidez baixa: bid_depth = ${bid_depth:,.0f}")
            flags.append("LOW_LIQUIDITY")
            score -= 20

        # ========================================
        # 2. VALIDAÇÃO DE VALUE AREA
        # ========================================
        VAL = metrics.get('VAL', 0)
        VAH = metrics.get('VAH', 0)
        
        if VAL == 0 or VAH == 0:
            issues.append("❌ VALUE AREA ZERADA (VAL ou VAH = $0.00)")
            flags.append("INVALID_VALUE_AREA")
            score -= 30
        elif VAH > 0 and VAL > 0:
            va_range_pct = (VAH - VAL) / current_price
            if va_range_pct < self.thresholds['min_value_area_range']:
                issues.append(f"⚠️ Value Area muito estreita: {va_range_pct:.3%}")
                flags.append("NARROW_VALUE_AREA")
                score -= 10

        # ========================================
        # 3. VALIDAÇÃO DE TICK RULE
        # ========================================
        tick_rule_sum = metrics.get('tick_rule_sum', 0)
        delta = metrics.get('delta', 0)
        
        if tick_rule_sum == 0 and delta != 0:
            issues.append("❌ TICK_RULE_SUM = 0 mas delta ≠ 0 (lógica quebrada)")
            flags.append("TICK_RULE_BROKEN")
            score -= 25

        # ========================================
        # 4. VALIDAÇÃO DE VOLUME RATIO
        # ========================================
        volume_sma_ratio = metrics.get('volume_sma_ratio', 0)
        
        if volume_sma_ratio > self.thresholds['max_volume_sma_ratio']:
            issues.append(f"❌ VOLUME_SMA_RATIO absurdo: {volume_sma_ratio:.1f}%")
            flags.append("VOLUME_RATIO_EXTREME")
            score -= 15

        # ========================================
        # 5. VALIDAÇÃO DE CONSISTÊNCIA FLOW vs DELTA
        # ========================================
        flow_imbalance = metrics.get('flow_imbalance', 0)
        
        if delta != 0 and flow_imbalance != 0:
            # Delta positivo deve ter flow positivo (e vice-versa)
            if np.sign(delta) != np.sign(flow_imbalance):
                divergence = abs(delta - flow_imbalance * abs(delta))
                if divergence > self.thresholds['max_flow_delta_divergence'] * abs(delta):
                    issues.append(f"⚠️ Divergência flow vs delta: {divergence:.2f}")
                    flags.append("FLOW_DELTA_MISMATCH")
                    score -= 20

        # ========================================
        # 6. VALIDAÇÃO DE WHALE METRICS
        # ========================================
        whale_buy = metrics.get('whale_buy_volume', 0)
        whale_sell = metrics.get('whale_sell_volume', 0)
        whale_delta_reported = metrics.get('whale_delta', 0)
        
        if whale_buy > 0 or whale_sell > 0:
            whale_delta_calculated = whale_buy - whale_sell
            if abs(whale_delta_calculated - whale_delta_reported) > 0.01:
                issues.append("⚠️ Whale delta inconsistente (buy/sell não batem)")
                flags.append("WHALE_DELTA_MISMATCH")
                score -= 10

        # ========================================
        # 7. VALIDAÇÃO DE LIQUIDATION HEATMAP
        # ========================================
        liq_heatmap = metrics.get('liquidation_heatmap', {})
        open_interest = metrics.get('open_interest', 0)
        
        if not liq_heatmap and open_interest > 0:
            issues.append("⚠️ Liquidation heatmap vazio mas OI existe")
            flags.append("MISSING_LIQUIDATION_DATA")
            score -= 5  # Baixa prioridade

        # ========================================
        # RESULTADO FINAL
        # ========================================
        final_score = max(0, score)
        
        return {
            "is_valid": final_score >= 70 and "INVALID_ORDERBOOK" not in flags,
            "quality_score": final_score,
            "issues": issues,
            "flags": flags,  # 🆕 Flags específicos para cada problema
            "critical_failures": [f for f in flags if f.startswith("INVALID_")],
        }

    # 🆕 VALIDAÇÃO COMPLETA (trades + métricas)
    def validate_full_context(self, df_window: pd.DataFrame, metrics: Dict[str, Any], 
                             current_price: float, window_duration_seconds: int) -> Dict[str, Any]:
        """
        Valida TUDO: trades + métricas calculadas.
        
        Returns:
            dict: Resultado consolidado com todas as validações
        """
        # Valida trades
        trades_validation = self.validate_window(df_window, window_duration_seconds)
        
        # Valida métricas
        metrics_validation = self.validate_metrics(metrics, current_price)
        
        # Consolida
        return {
            "is_valid": trades_validation["is_valid"] and metrics_validation["is_valid"],
            "trades_quality": trades_validation["quality_score"],
            "metrics_quality": metrics_validation["quality_score"],
            "overall_quality": (trades_validation["quality_score"] + metrics_validation["quality_score"]) / 2,
            "all_issues": trades_validation["issues"] + metrics_validation["issues"],
            "flags": metrics_validation.get("flags", []),
            "critical_failures": metrics_validation.get("critical_failures", []),
        }