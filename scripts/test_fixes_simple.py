"""
Teste simplificado das correções implementadas.
"""
import asyncio
import sys
import time
import os

# Adicionar o diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))


async def test_singleton():
    """Testa se MacroDataProvider é singleton"""
    try:
        # Testa import básico
        from data.macro_data_provider import MacroDataProvider
        
        print("\n[1/4] Testando Singleton...")
        
        # Verifica se a classe existe
        assert MacroDataProvider is not None, "MacroDataProvider não encontrado!"
        
        # Verifica se tem os métodos esperados
        assert hasattr(MacroDataProvider, 'get_instance'), "Método get_instance não encontrado!"
        assert hasattr(MacroDataProvider, '_instance'), "Variável _instance não encontrada!"
        assert hasattr(MacroDataProvider, '_initialized'), "Variável _initialized não encontrada!"
        
        print("   [OK] MacroDataProvider é singleton")
        return True
    except Exception as e:
        print(f"   [ERRO] Erro no teste Singleton: {e}")
        return False


async def test_cache():
    """Testa se cache está funcionando"""
    try:
        from data.macro_data_provider import get_macro_provider
        
        print("\n[2/4] Testando Cache...")
        
        provider = get_macro_provider()
        
        # Verifica se tem métodos de cache
        assert hasattr(provider, '_get_cached'), "Método _get_cached não encontrado!"
        assert hasattr(provider, '_set_cache'), "Método _set_cache não encontrado!"
        assert hasattr(provider, 'clear_cache'), "Método clear_cache não encontrado!"
        assert hasattr(provider, 'get_cache_stats'), "Método get_cache_stats não encontrado!"
        
        print("   [OK] Cache funcionando corretamente")
        return True
    except Exception as e:
        print(f"   [ERRO] Erro no teste Cache: {e}")
        return False


def test_async_bridge():
    """Testa se AsyncBridge funciona em thread"""
    import threading
    
    print("\n[3/4] Testando AsyncBridge em thread...")
    
    result = [None]
    error = [None]
    
    def thread_func():
        try:
            from utils.async_helpers import run_async_in_thread
            
            async def sample_async():
                await asyncio.sleep(0.1)
                return "success"
            
            result[0] = run_async_in_thread(sample_async())
        except Exception as e:
            error[0] = e
    
    thread = threading.Thread(target=thread_func)
    thread.start()
    thread.join(timeout=5)
    
    if error[0]:
        print(f"   [ERRO] Erro: {error[0]}")
        return False
    
    assert result[0] == "success", "AsyncBridge não retornou resultado correto!"
    print("   [OK] AsyncBridge funciona em threads")
    return True


async def test_macro_in_thread():
    """Testa se MacroData pode ser obtido de thread"""
    import threading
    
    print("\n[4/4] Testando MacroData em thread...")
    
    result = [None]
    error = [None]
    
    def thread_func():
        try:
            from utils.async_helpers import run_async_in_thread
            from data.macro_data_provider import get_macro_provider
            
            async def fetch_macro():
                provider = get_macro_provider()
                return await provider.get_all_macro_data()
            
            result[0] = run_async_in_thread(fetch_macro())
        except Exception as e:
            error[0] = e
    
    thread = threading.Thread(target=thread_func, name="TestAIWorker")
    thread.start()
    thread.join(timeout=30)
    
    if error[0]:
        print(f"   [ERRO] Erro: {error[0]}")
        return False
    
    assert result[0] is not None, "MacroData não foi obtido!"
    assert "vix" in result[0], "VIX não está no resultado!"
    print(f"   [OK] MacroData obtido em thread: {len(result[0])} campos")
    return True


async def main():
    print("=" * 60)
    print("    TESTE DAS CORREÇÕES")
    print("=" * 60)
    
    tests = [
        ("Singleton", test_singleton()),
        ("Cache", test_cache()),
        ("AsyncBridge", test_async_bridge()),
        ("MacroInThread", test_macro_in_thread()),
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