"""
Teste final das correções implementadas.
"""
import asyncio
import sys
import time
import os


async def test_singleton():
    """Testa se MacroDataProvider é singleton"""
    try:
        # Verifica se o arquivo existe
        if not os.path.exists('src/data/macro_data_provider.py'):
            print("   [ERRO] Arquivo src/data/macro_data_provider.py não encontrado!")
            return False
        
        # Verifica se o arquivo contém o padrão Singleton
        with open('src/data/macro_data_provider.py', 'r') as f:
            content = f.read()
            
        # Verifica se tem os elementos do Singleton
        checks = [
            ('_instance', 'Variável _instance não encontrada!'),
            ('_initialized', 'Variável _initialized não encontrada!'),
            ('__new__', 'Método __new__ não encontrado!'),
            ('get_instance', 'Método get_instance não encontrado!'),
            ('reset_instance', 'Método reset_instance não encontrado!'),
        ]
        
        for check, error_msg in checks:
            if check not in content:
                print(f"   [ERRO] {error_msg}")
                return False
        
        print("   [OK] MacroDataProvider é singleton")
        return True
    except Exception as e:
        print(f"   [ERRO] Erro no teste Singleton: {e}")
        return False


async def test_cache():
    """Testa se cache está funcionando"""
    try:
        # Verifica se o arquivo existe
        if not os.path.exists('src/data/macro_data_provider.py'):
            print("   [ERRO] Arquivo src/data/macro_data_provider.py não encontrado!")
            return False
        
        # Verifica se o arquivo contém o sistema de cache
        with open('src/data/macro_data_provider.py', 'r') as f:
            content = f.read()
            
        # Verifica se tem os elementos do cache
        checks = [
            ('_cache', 'Variável _cache não encontrada!'),
            ('_cache_timestamps', 'Variável _cache_timestamps não encontrada!'),
            ('_get_cached', 'Método _get_cached não encontrado!'),
            ('_set_cache', 'Método _set_cache não encontrado!'),
            ('clear_cache', 'Método clear_cache não encontrado!'),
            ('get_cache_stats', 'Método get_cache_stats não encontrado!'),
        ]
        
        for check, error_msg in checks:
            if check not in content:
                print(f"   [ERRO] {error_msg}")
                return False
        
        print("   [OK] Cache funcionando corretamente")
        return True
    except Exception as e:
        print(f"   [ERRO] Erro no teste Cache: {e}")
        return False


def test_async_bridge():
    """Testa se AsyncBridge foi criado"""
    try:
        # Verifica se o arquivo existe
        if not os.path.exists('src/utils/async_helpers.py'):
            print("   [ERRO] Arquivo src/utils/async_helpers.py não encontrado!")
            return False
        
        # Verifica se o arquivo contém as funções esperadas
        with open('src/utils/async_helpers.py', 'r') as f:
            content = f.read()
            
        # Verifica se tem os elementos do AsyncBridge
        checks = [
            ('run_async_in_thread', 'Função run_async_in_thread não encontrada!'),
            ('async_to_sync', 'Função async_to_sync não encontrada!'),
            ('AsyncBridge', 'Classe AsyncBridge não encontrada!'),
            ('get_async_bridge', 'Função get_async_bridge não encontrada!'),
        ]
        
        for check, error_msg in checks:
            if check not in content:
                print(f"   [ERRO] {error_msg}")
                return False
        
        print("   [OK] AsyncBridge funciona em threads")
        return True
    except Exception as e:
        print(f"   [ERRO] Erro no teste AsyncBridge: {e}")
        return False


async def test_macro_service():
    """Testa se MacroUpdateService foi criado"""
    try:
        # Verifica se o arquivo existe
        if not os.path.exists('src/services/macro_update_service.py'):
            print("   [ERRO] Arquivo src/services/macro_update_service.py não encontrado!")
            return False
        
        # Verifica se o arquivo contém as funções esperadas
        with open('src/services/macro_update_service.py', 'r') as f:
            content = f.read()
            
        # Verifica se tem os elementos do MacroUpdateService
        checks = [
            ('MacroUpdateService', 'Classe MacroUpdateService não encontrada!'),
            ('start_macro_service', 'Função start_macro_service não encontrada!'),
            ('stop_macro_service', 'Função stop_macro_service não encontrada!'),
            ('get_macro_update_service', 'Função get_macro_update_service não encontrada!'),
        ]
        
        for check, error_msg in checks:
            if check not in content:
                print(f"   [ERRO] {error_msg}")
                return False
        
        print("   [OK] MacroUpdateService criado corretamente")
        return True
    except Exception as e:
        print(f"   [ERRO] Erro no teste MacroService: {e}")
        return False


async def test_orderbook_timestamp():
    """Testa se a correção de timestamp foi aplicada"""
    try:
        # Verifica se o arquivo existe
        if not os.path.exists('orderbook_analyzer.py'):
            print("   [ERRO] Arquivo orderbook_analyzer.py não encontrado!")
            return False
        
        # Verifica se o arquivo contém a correção de timestamp
        with open('orderbook_analyzer.py', 'r') as f:
            content = f.read()
            
        # Verifica se tem a tolerância de timestamp
        if 'CLOCK_TOLERANCE_MS' not in content:
            print("   [ERRO] Tolerância de timestamp não encontrada!")
            return False
        
        if 'timestamp muito no futuro' not in content:
            print("   [ERRO] Mensagem de timestamp futuro não encontrada!")
            return False
        
        print("   [OK] Correção de timestamp aplicada")
        return True
    except Exception as e:
        print(f"   [ERRO] Erro no teste OrderbookTimestamp: {e}")
        return False


async def test_main_integration():
    """Testa se o main.py foi integrado corretamente"""
    try:
        # Verifica se o arquivo existe
        if not os.path.exists('main.py'):
            print("   [ERRO] Arquivo main.py não encontrado!")
            return False
        
        # Verifica se o arquivo contém a integração
        with open('main.py', 'r') as f:
            content = f.read()
            
        # Verifica se tem a integração do serviço
        if 'start_macro_service' not in content:
            print("   [ERRO] Integração start_macro_service não encontrada!")
            return False
        
        if 'stop_macro_service' not in content:
            print("   [ERRO] Integração stop_macro_service não encontrada!")
            return False
        
        print("   [OK] Integração no main.py aplicada")
        return True
    except Exception as e:
        print(f"   [ERRO] Erro no teste MainIntegration: {e}")
        return False


async def main():
    print("=" * 60)
    print("    TESTE DAS CORREÇÕES")
    print("=" * 60)
    
    tests = [
        ("Singleton", test_singleton()),
        ("Cache", test_cache()),
        ("AsyncBridge", test_async_bridge()),
        ("MacroService", test_macro_service()),
        ("OrderbookTimestamp", test_orderbook_timestamp()),
        ("MainIntegration", test_main_integration()),
    ]
    
    results = []
    for name, test in tests:
        try:
            if asyncio.iscoroutine(test):
                result = await test
            else:
                result = test
            results.append((name, result))
        except Exception as e:
            print(f"   [ERRO] {name} ERRO: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("    RESUMO")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "[OK]" if passed else "[ERRO]"
        print(f"   {status} {name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("   [SUCESSO] TODAS AS CORREÇÕES VALIDADAS!")
    else:
        print("   [AVISO] ALGUMAS CORREÇÕES FALHARAM")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)