#!/usr/bin/env python3
"""
Script para verificar se o PATCH foi implementado corretamente.
Verifica se:
1. O parâmetro ob_limit_fetch existe na assinatura do __init__
2. O atributo self.ob_limit_fetch é inicializado corretamente
3. O atributo self.wall_std é inicializado corretamente
"""

from orderbook_analyzer import OrderBookAnalyzer
import inspect

def test_patch_implementation():
    """Testa se o PATCH foi implementado corretamente."""
    
    print("Verificando implementacao do PATCH...")
    
    # 1. Verificar assinatura do __init__
    init_signature = inspect.signature(OrderBookAnalyzer.__init__)
    params = list(init_signature.parameters.keys())
    
    print(f"Parametros do __init__: {params}")
    
    # Verificar se ob_limit_fetch está na assinatura
    if 'ob_limit_fetch' in params:
        print("[OK] Parametro 'ob_limit_fetch' encontrado na assinatura do __init__")
    else:
        print("[ERRO] Parametro 'ob_limit_fetch' NAO encontrado na assinatura do __init__")
        return False
    
    # 2. Verificar inicialização dos atributos
    try:
        # Criar instância com parâmetros específicos
        analyzer = OrderBookAnalyzer(
            symbol="BTCUSDT",
            ob_limit_fetch=150,  # Valor customizado para teste
            wall_std_dev_factor=2.5  # Valor customizado para teste
        )
        
        print("[OK] Instancia criada com sucesso")
        
        # Verificar se o atributo ob_limit_fetch existe e tem o valor correto
        if hasattr(analyzer, 'ob_limit_fetch'):
            ob_value = analyzer.ob_limit_fetch
            print(f"[OK] Atributo 'ob_limit_fetch' existe: {ob_value}")
            if ob_value == 150:
                print("[OK] Valor de 'ob_limit_fetch' esta correto (150)")
            else:
                print(f"[ERRO] Valor incorreto de 'ob_limit_fetch': {ob_value} (esperado: 150)")
                return False
        else:
            print("[ERRO] Atributo 'ob_limit_fetch' NAO existe")
            return False
        
        # Verificar se o atributo wall_std existe e tem o valor correto
        if hasattr(analyzer, 'wall_std'):
            wall_std_value = analyzer.wall_std
            print(f"[OK] Atributo 'wall_std' existe: {wall_std_value}")
            if wall_std_value == 2.5:
                print("[OK] Valor de 'wall_std' esta correto (2.5)")
            else:
                print(f"[ERRO] Valor incorreto de 'wall_std': {wall_std_value} (esperado: 2.5)")
                return False
        else:
            print("[ERRO] Atributo 'wall_std' NAO existe")
            return False
        
        # Verificar se outros atributos necessários existem
        required_attrs = ['top_n', 'alert_threshold', 'symbol']
        for attr in required_attrs:
            if hasattr(analyzer, attr):
                print(f"[OK] Atributo '{attr}' existe: {getattr(analyzer, attr)}")
            else:
                print(f"[ERRO] Atributo '{attr}' NAO existe")
                return False
        
        print("\n[SUCCESS] PATCH IMPLEMENTADO COM SUCESSO!")
        print("   - [OK] Parametro ob_limit_fetch na assinatura")
        print("   - [OK] Atributo self.ob_limit_fetch inicializado")
        print("   - [OK] Atributo self.wall_std inicializado")
        print("   - [OK] Todos os atributos necessarios presentes")
        
        return True
        
    except Exception as e:
        print(f"[ERRO] Erro ao criar instancia: {e}")
        return False

def test_usage_example():
    """Exemplo de uso do PATCH."""
    print("\n" + "="*50)
    print("Exemplo de uso do PATCH:")
    print("="*50)
    
    # Exemplo de uso
    analyzer = OrderBookAnalyzer(
        symbol="BTCUSDT",
        ob_limit_fetch=200,  # Agora funciona!
        wall_std_dev_factor=3.5,
        top_n_levels=25
    )
    
    print(f"Symbol: {analyzer.symbol}")
    print(f"ob_limit_fetch: {analyzer.ob_limit_fetch}")
    print(f"wall_std: {analyzer.wall_std}")
    print(f"top_n: {analyzer.top_n}")
    print(f"alert_threshold: {analyzer.alert_threshold}")
    
    # Verificar que pode ser usado em _fetch_orderbook
    print(f"\nO atributo pode ser usado em _fetch_orderbook:")
    print(f"   limit = self.ob_limit_fetch  # = {analyzer.ob_limit_fetch}")
    
    return True

if __name__ == "__main__":
    print("VERIFICACAO DO PATCH - OrderBookAnalyzer")
    print("="*50)
    
    success = test_patch_implementation()
    if success:
        test_usage_example()
        print("\n[SUCCESS] TODAS AS VERIFICACOES PASSARAM!")
    else:
        print("\n[ERRO] FALHAS NA VERIFICACAO!")
        exit(1)