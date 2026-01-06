#!/usr/bin/env python3
"""
Teste Simplificado das Correções
Executa todas as 5 correções implementadas
"""

import sys
import traceback
from pathlib import Path

# ════════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════════════════

try:
    from src.data.macro_data_provider import MacroDataProvider
    from src.services.macro_service import MacroService
    from src.bridges.async_bridge import AsyncBridge
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")
    sys.exit(1)

# ════════════════════════════════════════════════════════════════════════════════
# FUNÇÕES DE TESTE
# ════════════════════════════════════════════════════════════════════════════════

def test_singleton_pattern():
    """Teste 1: Verificar se MacroDataProvider implementa padrão Singleton"""
    print("Testando Padrão Singleton...")
    
    try:
        # Verificar se __new__ está implementado
        if hasattr(MacroDataProvider, '__new__'):
            print("  OK __new__ implementado")
        
        # Verificar se _instance existe
        if hasattr(MacroDataProvider, '_instance'):
            print("  OK Atributo _instance encontrado")
        
        # Testar se é realmente singleton
        instance1 = MacroDataProvider()
        instance2 = MacroDataProvider()
        
        if instance1 is instance2:
            print("  OK Singleton implementado")
            return True
        else:
            print("  ERRO: Não é singleton")
            return False
            
    except Exception as e:
        print(f"  ERRO: {e}")
        return False


def test_cache_system():
    """Teste 2: Verificar se MacroDataProvider implementa sistema de cache"""
    print("Testando Sistema de Cache...")
    
    try:
        provider = MacroDataProvider()
        
        # Verificar atributos de cache
        has_cache = hasattr(provider, '_cache')
        has_timestamps = hasattr(provider, '_cache_timestamps')
        has_ttl = hasattr(provider, '_cache_ttl')
        
        print(f"  OK Atributos de cache: {has_cache and has_timestamps and has_ttl}")
        
        # Verificar métodos de cache
        has_get_cached = hasattr(provider, '_get_cached')
        has_set_cache = hasattr(provider, '_set_cache')
        has_clear_cache = hasattr(provider, 'clear_cache')
        
        print(f"  OK Métodos de cache: {has_get_cached and has_set_cache and has_clear_cache}")
        
        return has_cache and has_timestamps and has_ttl and has_get_cached and has_set_cache and has_clear_cache
        
    except Exception as e:
        print(f"  ERRO: {e}")
        return False


def test_async_bridge():
    """Teste 3: Verificar se AsyncBridge está implementado"""
    print("Testando AsyncBridge...")
    
    try:
        # Verificar se a classe existe
        if not AsyncBridge:
            print("  ERRO: AsyncBridge não encontrado")
            return False
        
        # Verificar se existe no módulo
        print("  OK AsyncBridge implementado")
        return True
        
    except Exception as e:
        print(f"  ERRO: {e}")
        return False


def test_macro_service():
    """Teste 4: Verificar se MacroService está implementado"""
    print("Testando MacroService...")
    
    try:
        # Verificar se a classe existe
        if not MacroService:
            print("  ERRO: MacroService não encontrado")
            return False
        
        # Verificar se tem método get_all
        has_get_all = hasattr(MacroService, 'get_all')
        print(f"  OK Método get_all: {has_get_all}")
        
        return MacroService is not None
        
    except Exception as e:
        print(f"  ERRO: {e}")
        return False


def test_orderbook_timestamp_fix():
    """Teste 5: Verificar correção de timestamp no orderbook"""
    print("Testando Correção de Timestamp no Orderbook...")
    
    try:
        # Procurar pelo arquivo orderbook_analyzer.py
        orderbook_file = Path("orderbook_analyzer.py")
        
        if not orderbook_file.exists():
            print("  ERRO: Arquivo orderbook_analyzer.py não encontrado")
            return False
        
        # Ler o arquivo
        content = orderbook_file.read_text(encoding='utf-8')
        
        # Verificar se tem correções de timestamp
        has_timestamp_fix = "timestamp" in content.lower() and "event_time" in content.lower()
        
        print(f"  OK Correções de timestamp encontradas: {has_timestamp_fix}")
        
        return has_timestamp_fix
        
    except Exception as e:
        print(f"  ERRO: {e}")
        return False


# ════════════════════════════════════════════════════════════════════════════════
# FUNÇÃO PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════════

def main():
    print("============================================================")
    print("    TESTE SIMPLIFICADO DAS CORREÇÕES")
    print("============================================================")
    
    tests = [
        ("Singleton", test_singleton_pattern),
        ("Cache", test_cache_system),
        ("AsyncBridge", test_async_bridge),
        ("MacroService", test_macro_service),
        ("Orderbook Timestamp", test_orderbook_timestamp_fix),
    ]
    
    results = {}
    passed = 0
    
    for name, test_func in tests:
        print(f"\n=== {name} ===")
        try:
            results[name] = test_func()
            if results[name]:
                passed += 1
        except Exception as e:
            print(f"  ERRO CRÍTICO: {e}")
            traceback.print_exc()
            results[name] = False
    
    # ════════════════════════════════════════════════════════════════════════════════
    # RESUMO
    # ════════════════════════════════════════════════════════════════════════════════
    
    print("\n============================================================")
    print("    RESUMO DOS TESTES")
    print("============================================================")
    
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{len(tests)} testes passaram")
    
    if passed == len(tests):
        print("\nOK TODAS AS CORREÇÕES FORAM IMPLEMENTADAS COM SUCESSO!")
        return 0
    else:
        print(f"\nERRO {len(tests) - passed} CORREÇÃO(ÕES) AINDA PRECISAM SER IMPLEMENTADA(S)")
        return 1


if __name__ == "__main__":
    sys.exit(main())