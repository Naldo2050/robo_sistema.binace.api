# integration_validator.py v1.0.0

"""
Validador de integração que rejeita dados inválidos ANTES de análise.

🎯 USO: Chamar ANTES de enviar dados para IA ou processar eventos.
"""

import logging
from typing import Dict, Any, List, Optional


class IntegrationValidator:
    """
    Valida dados integrados de múltiplas fontes antes de análise.
    
    REJEITA eventos com:
      - Orderbook zerado
      - Value Area zerada
      - Volumes zero mas razões existem (contradição)
      - tick_rule_sum sempre zero
    """
    
    def __init__(self):
        self.validation_failures = 0
        self.total_validations = 0
        
        logging.info("✅ IntegrationValidator inicializado")
    
    def validate_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida evento completo antes de análise.
        
        Args:
            event: Evento com dados de orderbook, flow, etc.
        
        Returns:
            Dict com:
              - is_valid: bool
              - should_skip: bool
              - issues: List[str]
              - critical_issues: List[str]
        """
        self.total_validations += 1
        
        issues: List[str] = []
        critical_issues: List[str] = []
        warnings: List[str] = []
        
        # ========================================
        # 1. VALIDA ORDERBOOK
        # ========================================
        orderbook_data = event.get('orderbook_data', {})
        
        bid_depth = orderbook_data.get('bid_depth_usd', 0)
        ask_depth = orderbook_data.get('ask_depth_usd', 0)
        ob_imbalance = orderbook_data.get('imbalance', 0)
        
        if bid_depth == 0 or ask_depth == 0:
            critical_issues.append(
                f"❌ ORDERBOOK ZERADO: bid=${bid_depth}, ask=${ask_depth}"
            )
        
        if bid_depth == 0 and ask_depth == 0 and ob_imbalance != 0:
            issues.append(
                f"⚠️ Imbalance ({ob_imbalance}) existe mas depths zerados"
            )
        
        # ========================================
        # 2. VALIDA VALUE AREA
        # ========================================
        val = event.get('VAL', 0) or event.get('value_area_low', 0)
        vah = event.get('VAH', 0) or event.get('value_area_high', 0)
        
        if val == 0 or vah == 0:
            warnings.append(
                f"⚡ VALUE AREA ZERADA: VAL=${val}, VAH=${vah}"
            )
        
        # ========================================
        # 3. VALIDA ORDER FLOW
        # ========================================
        order_flow = event.get('order_flow', {})
        flow_data = event.get('flow', {})
        
        # Procura buy/sell volumes
        buy_vol = (
            order_flow.get('buy_volume', 0) or
            flow_data.get('buy_volume', 0) or
            event.get('buy_volume', 0)
        )
        
        sell_vol = (
            order_flow.get('sell_volume', 0) or
            flow_data.get('sell_volume', 0) or
            event.get('sell_volume', 0)
        )
        
        delta = (
            order_flow.get('delta', 0) or
            order_flow.get('net_flow_1m', 0) or
            flow_data.get('delta', 0) or
            event.get('delta', 0)
        )
        
        buy_sell_ratio = (
            order_flow.get('buy_sell_ratio') or
            flow_data.get('buy_sell_ratio')
        )
        
        aggressive_buy_pct = order_flow.get('aggressive_buy_pct')
        aggressive_sell_pct = order_flow.get('aggressive_sell_pct')
        
        # VALIDAÇÃO: Volumes zero mas razão/percentuais existem
        if buy_vol == 0 and sell_vol == 0:
            if buy_sell_ratio is not None and buy_sell_ratio != 0:
                issues.append(
                    f"⚠️ CONTRADIÇÃO: buy/sell volumes zero mas razão = {buy_sell_ratio}"
                )
            
            if aggressive_buy_pct is not None or aggressive_sell_pct is not None:
                issues.append(
                    f"⚠️ CONTRADIÇÃO: volumes zero mas percentuais = "
                    f"{aggressive_buy_pct}% / {aggressive_sell_pct}%"
                )
        
        # VALIDAÇÃO: Delta existe mas volumes zero
        if delta != 0 and buy_vol == 0 and sell_vol == 0:
            warnings.append(
                f"⚡ Delta={delta} mas buy/sell volumes zero (pode ser net_flow em USD)"
            )
        
        # ========================================
        # 4. VALIDA ML FEATURES
        # ========================================
        ml_features = event.get('ml_features', {})
        microestrutura = event.get('microestrutura', {})
        
        # Procura tick_rule_sum
        tick_rule = (
            ml_features.get('tick_rule_sum', 0) or
            microestrutura.get('tick_rule_sum', 0) or
            order_flow.get('tick_rule_sum', 0)
        )
        
        # VALIDAÇÃO: tick_rule_sum sempre zero mas delta varia
        if tick_rule == 0 and abs(delta) > 100:
            warnings.append(
                f"⚡ tick_rule_sum=0 mas delta={delta} (pode estar quebrado)"
            )
        
        # order_book_slope sempre zero
        order_book_slope = (
            ml_features.get('order_book_slope', 0) or
            microestrutura.get('order_book_slope', 0)
        )
        
        if order_book_slope == 0 and (bid_depth > 0 or ask_depth > 0):
            warnings.append(
                "⚡ order_book_slope=0 mas orderbook tem dados"
            )
        
        # ========================================
        # 5. VALIDA FLOW IMBALANCE
        # ========================================
        flow_imbalance = (
            order_flow.get('flow_imbalance') or
            ml_features.get('flow_imbalance') or
            microestrutura.get('flow_imbalance')
        )
        
        if flow_imbalance is None and abs(delta) > 100:
            warnings.append(
                "⚡ flow_imbalance ausente mas delta existe"
            )
        
        # ========================================
        # RESULTADO
        # ========================================
        has_critical = len(critical_issues) > 0
        has_issues = len(issues) > 0
        
        is_valid = not has_critical
        should_skip = has_critical or (has_issues and len(issues) >= 3)
        
        if not is_valid:
            self.validation_failures += 1
        
        return {
            "is_valid": is_valid,
            "should_skip": should_skip,
            "issues": issues,
            "critical_issues": critical_issues,
            "warnings": warnings,
            "validation_summary": self._format_summary(
                critical_issues, 
                issues, 
                warnings
            ),
        }
    
    def _format_summary(
        self, 
        critical: List[str], 
        issues: List[str], 
        warnings: List[str]
    ) -> str:
        """Formata resumo legível."""
        parts = []
        
        if critical:
            parts.append(f"🔴 CRÍTICO: {len(critical)} problema(s)")
        if issues:
            parts.append(f"⚠️ ISSUES: {len(issues)} problema(s)")
        if warnings:
            parts.append(f"⚡ AVISOS: {len(warnings)}")
        
        if not parts:
            return "✅ Dados válidos"
        
        return " | ".join(parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de validação."""
        failure_rate = 100 * self.validation_failures / max(1, self.total_validations)
        
        return {
            "total_validations": self.total_validations,
            "validation_failures": self.validation_failures,
            "failure_rate_pct": round(failure_rate, 2),
        }


# ========================================
# TESTE
# ========================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    validator = IntegrationValidator()
    
    # Teste com evento problemático
    event_ruim = {
        "delta": -20.00,
        "buy_volume": 0,
        "sell_volume": 0,
        "order_flow": {
            "buy_sell_ratio": 0.36,
            "aggressive_buy_pct": 26.47,
            "aggressive_sell_pct": 73.53,
            "tick_rule_sum": 0.0000,
        },
        "orderbook_data": {
            "bid_depth_usd": 0,
            "ask_depth_usd": 0,
            "imbalance": 0.00,
        },
        "VAL": 0.00,
        "VAH": 0.00,
        "ml_features": {
            "flow_imbalance": -0.360,
            "order_book_slope": 0.0000,
            "momentum_score": -0.05440,
            "tick_rule_sum": 0.0000,
        },
    }
    
    print("\n" + "="*80)
    print("🧪 VALIDANDO EVENTO PROBLEMÁTICO")
    print("="*80 + "\n")
    
    result = validator.validate_event(event_ruim)
    
    print(f"is_valid: {result['is_valid']}")
    print(f"should_skip: {result['should_skip']}")
    print(f"\nResumo: {result['validation_summary']}\n")
    
    if result['critical_issues']:
        print("🔴 CRÍTICOS:")
        for issue in result['critical_issues']:
            print(f"  {issue}")
    
    if result['issues']:
        print("\n⚠️ ISSUES:")
        for issue in result['issues']:
            print(f"  {issue}")
    
    if result['warnings']:
        print("\n⚡ AVISOS:")
        for warning in result['warnings']:
            print(f"  {warning}")
    
    print("\n" + "="*80)
    print(f"Stats: {validator.get_stats()}")
    print("="*80 + "\n")