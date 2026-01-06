"""
Testa se as correções foram implementadas corretamente.
"""
import asyncio
import sys
import time
import os

# Adicionar o diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_singleton():
    """Testa se MacroDataProvider é singleton"""
    try:
        from src.data.macro_data_provider import MacroDataProvider, get_macro_provider
        
        print("\n[1/4] Testando Singleton...")
        
        p1 = MacroDataProvider()
        p2 = MacroDataProvider()
        p3 = get_macro_provider()
        
        assert p1 is p2, "MacroDataProvider não é singleton!"
        assert p2 is p3, "get_macro_provider não retorna singleton!"
        
        print("   [OK] MacroDataProvider é singleton")
        return True
    except Exception as e:
        print(f"   [ERRO] Erro no teste Singleton: {e}")
        return False


async def test_cache():
    """Testa se cache está funcionando"""
    try:
        from src.data.macro_data_provider import get_macro_provider
        
        print("\n[2/4] Testando Cache...")
        
        provider = get_macro_provider()
        
        # Primeira chamada - deve buscar da API
        start = time.time()
        data1 = await provider.get_all_macro_data()
        time1 = time.time() - start
        
        # Segunda chamada - deve vir do cache (muito mais rápida)
        start = time.time()
        data2 = await provider.get_all_macro_data()
        time2 = time.time() - start
        
        print(f"   Primeira chamada: {time1:.3f}s")
        print(f"   Segunda chamada (cache): {time2:.3f}s")
        
        assert time2 < time1 * 0.5, "Cache não está funcionando!"
        assert data1 == data2, "Cache retornou dados diferentes!"
        
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
            from src.data.macro_data_provider import get_macro_provider
            
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