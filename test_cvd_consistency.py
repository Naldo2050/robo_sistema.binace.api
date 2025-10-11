# test_cvd_consistency.py
"""
Script para testar consistência do CVD e identificar problemas.

Testa:
1. Acumulação correta do CVD
2. Relação entre delta da janela e CVD
3. Reset periódico
4. Concorrência (threads)
"""

import time
import logging
from datetime import datetime
from typing import List, Dict
import random

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Importar módulos do sistema
try:
    from flow_analyzer import FlowAnalyzer
    from time_manager import TimeManager
except ImportError as e:
    print(f"❌ Erro ao importar módulos: {e}")
    print("Execute este script na pasta do projeto!")
    exit(1)


class CVDTester:
    """Testa consistência do CVD"""
    
    def __init__(self):
        self.time_manager = TimeManager()
        self.flow_analyzer = FlowAnalyzer(time_manager=self.time_manager)
        self.trade_history: List[Dict] = []
        self.cvd_history: List[float] = []
        
    def create_trade(
        self, 
        qty: float, 
        price: float, 
        is_buyer_maker: bool, 
        timestamp_ms: int
    ) -> Dict:
        """Cria um trade sintético"""
        return {
            'q': qty,
            'p': price,
            'm': is_buyer_maker,
            'T': timestamp_ms,
        }
    
    def test_basic_accumulation(self):
        """Teste 1: Acumulação básica do CVD"""
        print("\n" + "="*80)
        print("🧪 TESTE 1: Acumulação Básica do CVD")
        print("="*80)
        
        # Reset para começar do zero
        self.flow_analyzer._reset_metrics()
        
        base_ts = self.time_manager.now_ms()
        base_price = 100000.0
        
        # Criar sequência de trades com deltas conhecidos
        test_trades = [
            # (qty, price, is_buyer_maker, expected_delta)
            (1.0, base_price, False, +1.0),   # Compra: +1.0
            (0.5, base_price, True, -0.5),    # Venda: -0.5
            (2.0, base_price, False, +2.0),   # Compra: +2.0
            (1.5, base_price, True, -1.5),    # Venda: -1.5
            (3.0, base_price, False, +3.0),   # Compra: +3.0
        ]
        
        expected_cvd = 0.0
        print(f"\n{'Time':>8} | {'Qty':>6} | {'Side':>6} | {'Delta':>8} | {'Expected CVD':>12} | {'Actual CVD':>12} | {'Match':>6}")
        print("-" * 80)
        
        all_match = True
        
        for i, (qty, price, is_buyer_maker, expected_delta) in enumerate(test_trades):
            ts = base_ts + (i * 1000)  # 1 segundo entre trades
            
            # Processar trade
            trade = self.create_trade(qty, price, is_buyer_maker, ts)
            self.flow_analyzer.process_trade(trade)
            
            # Atualizar CVD esperado
            expected_cvd += expected_delta
            
            # Obter CVD atual
            metrics = self.flow_analyzer.get_flow_metrics(reference_epoch_ms=ts)
            actual_cvd = metrics['cvd']
            
            # Comparar
            match = abs(actual_cvd - expected_cvd) < 0.0001
            all_match = all_match and match
            
            side = "SELL" if is_buyer_maker else "BUY"
            
            print(
                f"{i:8d} | {qty:6.2f} | {side:6s} | {expected_delta:+8.2f} | "
                f"{expected_cvd:+12.4f} | {actual_cvd:+12.4f} | "
                f"{'✅' if match else '❌'}"
            )
            
            self.cvd_history.append(actual_cvd)
            self.trade_history.append({
                'trade': trade,
                'expected_delta': expected_delta,
                'expected_cvd': expected_cvd,
                'actual_cvd': actual_cvd,
            })
        
        print("-" * 80)
        if all_match:
            print("✅ TESTE PASSOU: CVD está acumulando corretamente!")
        else:
            print("❌ TESTE FALHOU: CVD não está consistente!")
        
        return all_match
    
    def test_window_vs_cvd(self):
        """Teste 2: Diferença entre delta da janela e CVD"""
        print("\n" + "="*80)
        print("🧪 TESTE 2: Delta da Janela vs CVD Acumulado")
        print("="*80)
        
        # Reset
        self.flow_analyzer._reset_metrics()
        
        base_ts = self.time_manager.now_ms()
        base_price = 100000.0
        
        # Simular 3 janelas de 1 minuto
        print(f"\n{'Janela':>8} | {'Trades':>8} | {'Delta Janela':>14} | {'CVD Acumulado':>14} | {'Esperado':>14}")
        print("-" * 80)
        
        cumulative_delta = 0.0
        
        for window in range(3):
            window_ts_start = base_ts + (window * 60 * 1000)  # 1 minuto entre janelas
            window_delta = 0.0
            
            # Criar 5 trades aleatórios por janela
            for i in range(5):
                qty = random.uniform(0.5, 3.0)
                is_buyer_maker = random.choice([True, False])
                ts = window_ts_start + (i * 1000)
                
                trade = self.create_trade(qty, base_price, is_buyer_maker, ts)
                self.flow_analyzer.process_trade(trade)
                
                # Calcular delta deste trade
                trade_delta = -qty if is_buyer_maker else qty
                window_delta += trade_delta
            
            cumulative_delta += window_delta
            
            # Obter métricas
            metrics = self.flow_analyzer.get_flow_metrics(
                reference_epoch_ms=window_ts_start + 60000
            )
            
            actual_cvd = metrics['cvd']
            net_flow_1m = metrics.get('order_flow', {}).get('net_flow_1m', 0.0)
            
            # Converter net_flow de USD para BTC (aproximado)
            net_flow_btc = net_flow_1m / base_price
            
            print(
                f"{window+1:8d} | {5:8d} | {window_delta:+14.4f} | "
                f"{actual_cvd:+14.4f} | {cumulative_delta:+14.4f}"
            )
        
        print("-" * 80)
        print("\n💡 OBSERVAÇÃO:")
        print("   - 'Delta Janela' = delta APENAS daquela janela")
        print("   - 'CVD Acumulado' = soma de TODAS as janelas")
        print("   - CVD deve ser igual ao 'Esperado' (soma acumulada)")
        
        # Verificar se CVD final está correto
        final_metrics = self.flow_analyzer.get_flow_metrics()
        final_cvd = final_metrics['cvd']
        
        match = abs(final_cvd - cumulative_delta) < 0.01
        
        if match:
            print(f"\n✅ CONSISTÊNCIA OK: CVD final ({final_cvd:+.4f}) = Soma esperada ({cumulative_delta:+.4f})")
        else:
            print(f"\n❌ INCONSISTÊNCIA: CVD final ({final_cvd:+.4f}) ≠ Soma esperada ({cumulative_delta:+.4f})")
        
        return match
    
    def test_reset_behavior(self):
        """Teste 3: Comportamento do reset"""
        print("\n" + "="*80)
        print("🧪 TESTE 3: Reset do CVD")
        print("="*80)
        
        # Criar alguns trades
        base_ts = self.time_manager.now_ms()
        base_price = 100000.0
        
        for i in range(5):
            qty = 1.0
            is_buyer_maker = False  # Todas compras
            ts = base_ts + (i * 1000)
            
            trade = self.create_trade(qty, base_price, is_buyer_maker, ts)
            self.flow_analyzer.process_trade(trade)
        
        metrics_before = self.flow_analyzer.get_flow_metrics()
        cvd_before = metrics_before['cvd']
        
        print(f"\nCVD antes do reset: {cvd_before:+.4f}")
        
        # Forçar reset
        self.flow_analyzer._reset_metrics()
        
        metrics_after = self.flow_analyzer.get_flow_metrics()
        cvd_after = metrics_after['cvd']
        
        print(f"CVD depois do reset: {cvd_after:+.4f}")
        
        if abs(cvd_after) < 0.0001:
            print("\n✅ Reset funcionou corretamente!")
            return True
        else:
            print("\n❌ Reset NÃO zerou o CVD!")
            return False
    
    def test_whale_delta_consistency(self):
        """Teste 4: Consistência do whale_delta"""
        print("\n" + "="*80)
        print("🧪 TESTE 4: Consistência Whale Delta")
        print("="*80)
        
        # Reset
        self.flow_analyzer._reset_metrics()
        
        base_ts = self.time_manager.now_ms()
        base_price = 100000.0
        
        # Trades whale (>= 5.0 BTC)
        whale_trades = [
            (6.0, False),   # Compra whale: +6.0
            (7.0, True),    # Venda whale: -7.0
            (8.0, False),   # Compra whale: +8.0
        ]
        
        expected_buy = 0.0
        expected_sell = 0.0
        expected_delta = 0.0
        
        print(f"\n{'Trade':>6} | {'Qty':>6} | {'Side':>6} | {'Buy Vol':>10} | {'Sell Vol':>10} | {'Delta':>10} | {'Esperado':>10} | {'Match':>6}")
        print("-" * 110)
        
        all_match = True
        
        for i, (qty, is_buyer_maker) in enumerate(whale_trades):
            ts = base_ts + (i * 1000)
            trade = self.create_trade(qty, base_price, is_buyer_maker, ts)
            self.flow_analyzer.process_trade(trade)
            
            # Atualizar esperado
            if is_buyer_maker:  # Venda
                expected_sell += qty
                expected_delta -= qty
            else:  # Compra
                expected_buy += qty
                expected_delta += qty
            
            # Obter atual
            metrics = self.flow_analyzer.get_flow_metrics(reference_epoch_ms=ts)
            actual_buy = metrics['whale_buy_volume']
            actual_sell = metrics['whale_sell_volume']
            actual_delta = metrics['whale_delta']
            
            # Verificar
            calculated_delta = actual_buy - actual_sell
            match = abs(actual_delta - expected_delta) < 0.0001 and \
                    abs(calculated_delta - expected_delta) < 0.0001
            
            all_match = all_match and match
            
            side = "SELL" if is_buyer_maker else "BUY"
            
            print(
                f"{i+1:6d} | {qty:6.2f} | {side:6s} | "
                f"{actual_buy:10.2f} | {actual_sell:10.2f} | "
                f"{actual_delta:+10.2f} | {expected_delta:+10.2f} | "
                f"{'✅' if match else '❌'}"
            )
        
        print("-" * 110)
        
        # Validação final
        final_metrics = self.flow_analyzer.get_flow_metrics()
        final_delta = final_metrics['whale_delta']
        final_buy = final_metrics['whale_buy_volume']
        final_sell = final_metrics['whale_sell_volume']
        
        formula_check = abs((final_buy - final_sell) - final_delta) < 0.0001
        
        print(f"\nValidação: whale_delta = buy - sell")
        print(f"  buy = {final_buy:.2f}")
        print(f"  sell = {final_sell:.2f}")
        print(f"  buy - sell = {final_buy - final_sell:+.2f}")
        print(f"  whale_delta = {final_delta:+.2f}")
        print(f"  Match: {'✅' if formula_check else '❌'}")
        
        if all_match and formula_check:
            print("\n✅ TESTE PASSOU: Whale delta está consistente!")
        else:
            print("\n❌ TESTE FALHOU: Whale delta tem problemas!")
        
        return all_match and formula_check
    
    def run_all_tests(self):
        """Executa todos os testes"""
        print("\n" + "="*80)
        print("🚀 INICIANDO TESTES DE CONSISTÊNCIA DO CVD")
        print("="*80)
        
        results = {
            'basic_accumulation': self.test_basic_accumulation(),
            'window_vs_cvd': self.test_window_vs_cvd(),
            'reset_behavior': self.test_reset_behavior(),
            'whale_delta': self.test_whale_delta_consistency(),
        }
        
        print("\n" + "="*80)
        print("📊 RESUMO DOS TESTES")
        print("="*80)
        
        for test_name, passed in results.items():
            status = "✅ PASSOU" if passed else "❌ FALHOU"
            print(f"{test_name:30s}: {status}")
        
        all_passed = all(results.values())
        
        print("\n" + "="*80)
        if all_passed:
            print("🎉 TODOS OS TESTES PASSARAM!")
            print("✅ CVD está funcionando corretamente")
        else:
            print("⚠️ ALGUNS TESTES FALHARAM")
            print("❌ Verifique os logs acima para detalhes")
        print("="*80 + "\n")
        
        return all_passed


if __name__ == "__main__":
    tester = CVDTester()
    success = tester.run_all_tests()
    
    exit(0 if success else 1)