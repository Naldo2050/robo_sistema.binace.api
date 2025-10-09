#!/usr/bin/env python3
"""
Script de teste para validar correções v2.0.1

Testa:
1. Extração de orderbook da IA
2. Cálculo de buy_sell_ratio
3. Validação de dados
4. Consistência de volumes

Uso:
    python test_corrections.py
"""

import sys
import logging
import time  # <-- CORREÇÃO: import adicionado aqui
from typing import Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)

# Contador de testes
passed = 0
failed = 0
warnings = 0


def log_result(test_name: str, passed_test: bool, message: str = ""):
    """Log resultado do teste."""
    global passed, failed
    
    if passed_test:
        passed += 1
        logging.info(f"✅ {test_name}: PASSOU")
        if message:
            logging.info(f"   {message}")
    else:
        failed += 1
        logging.error(f"❌ {test_name}: FALHOU")
        if message:
            logging.error(f"   {message}")


def log_warning(test_name: str, message: str):
    """Log warning."""
    global warnings
    warnings += 1
    logging.warning(f"⚠️ {test_name}: WARNING")
    logging.warning(f"   {message}")


def test_orderbook_extraction():
    """Testa extração de orderbook."""
    print("\n" + "="*80)
    print("TESTE 1: Extração de Orderbook")
    print("="*80)
    
    try:
        from ai_analyzer_qwen import AIAnalyzer
        
        analyzer = AIAnalyzer()
        
        # Caso 1: Dados em orderbook_data
        event1 = {
            'orderbook_data': {
                'bid_depth_usd': 1000000.0,
                'ask_depth_usd': 900000.0,
            }
        }
        
        ob1 = analyzer._extract_orderbook_data(event1)
        test1 = (ob1.get('bid_depth_usd') == 1000000.0)
        log_result(
            "Extração de orderbook_data",
            test1,
            f"bid_usd={ob1.get('bid_depth_usd')}"
        )
        
        # Caso 2: Dados em spread_metrics
        event2 = {
            'spread_metrics': {
                'bid_depth_usd': 2000000.0,
                'ask_depth_usd': 1800000.0,
            }
        }
        
        ob2 = analyzer._extract_orderbook_data(event2)
        test2 = (ob2.get('bid_depth_usd') == 2000000.0)
        log_result(
            "Extração de spread_metrics",
            test2,
            f"bid_usd={ob2.get('bid_depth_usd')}"
        )
        
        # Caso 3: Dados zerados (inválidos)
        event3 = {
            'orderbook_data': {
                'bid_depth_usd': 0.0,
                'ask_depth_usd': 0.0,
            }
        }
        
        ob3 = analyzer._extract_orderbook_data(event3)
        test3 = (len(ob3) == 0)  # Deve retornar vazio
        log_result(
            "Rejeição de dados zerados",
            test3,
            "Retornou dict vazio (correto)" if test3 else f"Retornou: {ob3}"
        )
        
    except Exception as e:
        log_result("Teste de extração", False, str(e))


def test_flow_analyzer():
    """Testa FlowAnalyzer."""
    print("\n" + "="*80)
    print("TESTE 2: FlowAnalyzer - Buy/Sell Ratio")
    print("="*80)
    
    try:
        from flow_analyzer import FlowAnalyzer
        
        fa = FlowAnalyzer()
        
        # Simula trades
        now_ms = int(time.time() * 1000)
        
        trades = [
            {'q': 1.0, 'p': 50000.0, 'T': now_ms, 'm': False},      # Buy
            {'q': 2.0, 'p': 50010.0, 'T': now_ms + 100, 'm': True}, # Sell
            {'q': 1.5, 'p': 50005.0, 'T': now_ms + 200, 'm': False},# Buy
        ]
        
        for trade in trades:
            fa.process_trade(trade)
        
        metrics = fa.get_flow_metrics(reference_epoch_ms=now_ms + 300)
        
        of = metrics.get('order_flow', {})
        
        # Verifica volumes
        buy_vol = of.get('buy_volume', 0)
        sell_vol = of.get('sell_volume', 0)
        ratio = of.get('buy_sell_ratio')
        
        # Valores esperados:
        # Buy: 1.0*50000 + 1.5*50005 = 50000 + 75007.5 = 125007.5
        # Sell: 2.0*50010 = 100020
        # Ratio: 125007.5 / 100020 ≈ 1.25
        
        test1 = (buy_vol > 0 and sell_vol > 0)
        log_result(
            "Volumes calculados",
            test1,
            f"buy=${buy_vol:,.2f}, sell=${sell_vol:,.2f}"
        )
        
        test2 = (ratio is not None and ratio > 0)
        log_result(
            "Ratio calculado",
            test2,
            f"ratio={ratio:.4f}" if ratio else "ratio=None"
        )
        
        # Testa caso sem trades (ratio deve ser None)
        fa2 = FlowAnalyzer()
        metrics2 = fa2.get_flow_metrics()
        ratio2 = metrics2.get('order_flow', {}).get('buy_sell_ratio')
        
        test3 = (ratio2 is None)
        log_result(
            "Ratio None quando sem trades",
            test3,
            f"ratio={ratio2}" if not test3 else "ratio=None (correto)"
        )
        
    except Exception as e:
        log_result("Teste FlowAnalyzer", False, str(e))


def test_orderbook_analyzer():
    """Testa OrderBookAnalyzer."""
    print("\n" + "="*80)
    print("TESTE 3: OrderBookAnalyzer - Evento Completo")
    print("="*80)
    
    try:
        from orderbook_analyzer import OrderBookAnalyzer
        
        oba = OrderBookAnalyzer("BTCUSDT", market_type="futures")
        
        # Simula snapshot válido
        snap = {
            'bids': [(50000.0, 1.0), (49999.0, 2.0)],
            'asks': [(50001.0, 1.5), (50002.0, 2.5)],
            'E': int(time.time() * 1000),  # Agora 'time' está definido globalmente
        }
        
        event = oba.analyze(current_snapshot=snap)
        
        # Verifica campos obrigatórios
        test1 = ('orderbook_data' in event)
        log_result(
            "Campo orderbook_data presente",
            test1
        )
        
        if test1:
            ob = event['orderbook_data']
            
            test2 = (ob.get('bid_depth_usd', 0) > 0)
            log_result(
                "bid_depth_usd válido",
                test2,
                f"bid_depth_usd={ob.get('bid_depth_usd')}"
            )
            
            test3 = (ob.get('ask_depth_usd', 0) > 0)
            log_result(
                "ask_depth_usd válido",
                test3,
                f"ask_depth_usd={ob.get('ask_depth_usd')}"
            )
            
            test4 = ('is_valid' in ob)
            log_result(
                "Flag is_valid presente",
                test4,
                f"is_valid={ob.get('is_valid')}"
            )
        
        # Testa snapshot inválido (vazio)
        snap_invalid = {
            'bids': [],
            'asks': [],
        }
        
        event_invalid = oba.analyze(current_snapshot=snap_invalid)
        
        test5 = (event_invalid.get('is_valid') == False)
        log_result(
            "Snapshot inválido detectado",
            test5,
            f"is_valid={event_invalid.get('is_valid')}"
        )
        
        test6 = (event_invalid.get('orderbook_data', {}).get('is_valid') == False)
        log_result(
            "orderbook_data.is_valid=False",
            test6
        )
        
    except Exception as e:
        log_result("Teste OrderBookAnalyzer", False, str(e))


def test_data_consistency():
    """Testa consistência de dados."""
    print("\n" + "="*80)
    print("TESTE 4: Consistência de Dados")
    print("="*80)
    
    try:
        from flow_analyzer import FlowAnalyzer
        
        fa = FlowAnalyzer()
        
        # Caso: Delta significativo com volumes válidos
        now_ms = int(time.time() * 1000)
        
        trades = [
            {'q': 5.0, 'p': 50000.0, 'T': now_ms, 'm': False},  # Buy grande
            {'q': 0.1, 'p': 50010.0, 'T': now_ms + 100, 'm': True},  # Sell pequeno
        ]
        
        for trade in trades:
            fa.process_trade(trade)
        
        metrics = fa.get_flow_metrics(reference_epoch_ms=now_ms + 200)
        of = metrics.get('order_flow', {})
        
        delta = of.get('net_flow_1m', 0)
        buy_vol = of.get('buy_volume', 0)
        sell_vol = of.get('sell_volume', 0)
        ratio = of.get('buy_sell_ratio')
        
        # Delta deve ser positivo (mais compra)
        test1 = (delta > 0)
        log_result(
            "Delta positivo para mais compras",
            test1,
            f"delta={delta:+,.2f}"
        )
        
        # Volumes devem existir
        test2 = (buy_vol > 0 and sell_vol > 0)
        log_result(
            "Volumes consistentes com trades",
            test2,
            f"buy=${buy_vol:,.0f}, sell=${sell_vol:,.0f}"
        )
        
        # Ratio deve existir
        test3 = (ratio is not None and ratio > 1.0)  # Mais buy que sell
        log_result(
            "Ratio consistente (buy > sell)",
            test3,
            f"ratio={ratio:.4f}" if ratio else "ratio=None"
        )
        
        # Verifica que ratio reflete volumes
        if buy_vol > 0 and sell_vol > 0 and ratio:
            expected_ratio = buy_vol / sell_vol
            diff = abs(ratio - expected_ratio) / expected_ratio
            
            test4 = (diff < 0.01)  # < 1% de diferença
            log_result(
                "Ratio calculado corretamente",
                test4,
                f"calculado={ratio:.4f}, esperado={expected_ratio:.4f}, diff={diff*100:.2f}%"
            )
        
    except Exception as e:
        log_result("Teste de consistência", False, str(e))


def test_ai_analyzer_validation():
    """Testa validação da IA."""
    print("\n" + "="*80)
    print("TESTE 5: Validação da IA")
    print("="*80)
    
    try:
        from ai_analyzer_qwen import AIAnalyzer
        
        analyzer = AIAnalyzer()
        
        # Caso 1: Evento com orderbook válido
        event_valid = {
            'tipo_evento': 'OrderBook',
            'ativo': 'BTCUSDT',
            'orderbook_data': {
                'bid_depth_usd': 1000000.0,
                'ask_depth_usd': 900000.0,
                'imbalance': 0.05,
                'mid': 50000.0,
            },
            'flow_metrics': {
                'order_flow': {
                    'net_flow_1m': 123456.78,
                    'flow_imbalance': 0.10,
                    'buy_volume': 600000.0,
                    'sell_volume': 500000.0,
                    'buy_sell_ratio': 1.20,
                }
            },
        }
        
        prompt1 = analyzer._create_prompt(event_valid)
        
        # Verifica se prompt contém dados do orderbook
        test1 = ('Bid Depth:' in prompt1 or 'bids=' in prompt1.lower())
        log_result(
            "Prompt inclui dados de orderbook",
            test1
        )
        
        test2 = ('1,000,000' in prompt1 or '1000000' in prompt1)
        log_result(
            "Prompt mostra bid_depth correto",
            test2
        )
        
        # Caso 2: Evento com orderbook zerado
        event_invalid = {
            'tipo_evento': 'OrderBook',
            'ativo': 'BTCUSDT',
            'orderbook_data': {
                'bid_depth_usd': 0.0,
                'ask_depth_usd': 0.0,
            },
            'flow_metrics': {
                'order_flow': {
                    'net_flow_1m': -50000.0,
                    'flow_imbalance': -0.25,
                }
            },
        }
        
        prompt2 = analyzer._create_prompt(event_invalid)
        
        # Verifica se prompt marca como indisponível
        test3 = ('INDISPONÍVEL' in prompt2.upper() or 'ZERADO' in prompt2.upper())
        log_result(
            "Prompt marca orderbook inválido",
            test3
        )
        
        test4 = ('net_flow' in prompt2.lower() or 'flow_imbalance' in prompt2.lower())
        log_result(
            "Prompt sugere usar fluxo alternativo",
            test4
        )
        
    except Exception as e:
        log_result("Teste validação IA", False, str(e))


def print_summary():
    """Imprime resumo dos testes."""
    print("\n" + "="*80)
    print("RESUMO DOS TESTES")
    print("="*80)
    
    total = passed + failed
    
    print(f"\n✅ Passou:   {passed}/{total} ({100*passed/total:.1f}%)" if total > 0 else "")
    print(f"❌ Falhou:   {failed}/{total} ({100*failed/total:.1f}%)" if total > 0 else "")
    print(f"⚠️  Warnings: {warnings}")
    
    if failed == 0:
        print("\n🎉 TODAS AS CORREÇÕES ESTÃO FUNCIONANDO!")
        print("   Pode rodar o sistema com segurança.")
    elif failed <= 2:
        print("\n⚠️ ALGUMAS CORREÇÕES PRECISAM DE AJUSTES")
        print("   Revise os testes que falharam acima.")
    else:
        print("\n🚨 MUITOS TESTES FALHARAM")
        print("   Verifique se os arquivos foram substituídos corretamente.")
    
    print("\n" + "="*80 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    print("\n" + "="*80)
    print("SCRIPT DE TESTE - CORREÇÕES v2.0.1")
    print("="*80)
    print("\nIniciando testes...\n")
    
    try:
        test_orderbook_extraction()
        test_flow_analyzer()
        test_orderbook_analyzer()
        test_data_consistency()
        test_ai_analyzer_validation()
        
        success = print_summary()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Testes interrompidos pelo usuário\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Erro fatal nos testes: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)