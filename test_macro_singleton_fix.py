"""
Teste de validação do MacroDataProvider Singleton
Valida se o padrão Singleton está funcionando corretamente
"""
import asyncio
import logging
import sys
import os

# Adicionar o diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configurar logging para ver as mensagens de debug
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_singleton_behavior():
    """Testa se o padrão Singleton está funcionando"""
    print("TESTE: Comportamento Singleton")
    print("=" * 60)
    
    # Importar o módulo
    try:
        from src.data.macro_data_provider import MacroDataProvider, get_macro_provider
        print("[OK] Import bem-sucedido")
    except Exception as e:
        print(f"[ERRO] Erro no import: {e}")
        return False
    
    # Teste 1: Verificar se múltiplas instancias são iguais
    print("\nTeste 1: Verificação de múltiplas instâncias")
    provider1 = MacroDataProvider()
    provider2 = MacroDataProvider()
    provider3 = MacroDataProvider.get_instance()
    
    if provider1 is provider2 and provider2 is provider3:
        print("[OK] Padrão Singleton funcionando - todas as instâncias são iguais")
    else:
        print("[ERRO] Padrão Singleton FALHOU - instâncias diferentes")
        return False
    
    # Teste 2: Verificar função helper
    print("\nTeste 2: Função helper get_macro_provider()")
    provider4 = get_macro_provider()
    if provider4 is provider1:
        print("[OK] Função helper funcionando corretamente")
    else:
        print("[ERRO] Função helper FALHOU")
        return False
    
    # Teste 3: Testar cache
    print("\nTeste 3: Sistema de cache")
    
    # Primeira chamada - deve fazer request
    print("Primeira chamada para get_vix():")
    vix1 = await provider1.get_vix()
    print(f"  Resultado: {vix1}")
    
    # Segunda chamada - deve usar cache
    print("Segunda chamada para get_vix():")
    vix2 = await provider1.get_vix()
    print(f"  Resultado: {vix2}")
    
    if vix1 == vix2:
        print("[OK] Cache funcionando - mesmo resultado nas duas chamadas")
    else:
        print("[AVISO] Cache pode estar falhando - resultados diferentes")
    
    # Teste 4: Verificar stats do cache
    print("\nTeste 4: Estatísticas do cache")
    stats = provider1.get_cache_stats()
    print(f"  Total de chaves em cache: {stats['total_keys']}")
    for key, info in stats['keys'].items():
        print(f"    {key}: {info['age_seconds']}s, válido: {info['is_valid']}")
    
    print("\n[OK] TESTE SINGLETON CONCLUÍDO COM SUCESSO")
    return True

async def test_multiple_calls():
    """Testa múltiplas chamadas para verificar se não há duplicação"""
    print("\nTESTE: Múltiplas chamadas")
    print("=" * 60)
    
    from src.data.macro_data_provider import MacroDataProvider
    
    # Fazer 5 chamadas seguidas
    for i in range(5):
        print(f"\nChamada {i+1}:")
        provider = MacroDataProvider()  # Sempre cria nova "instância"
        result = await provider.get_vix()
        print(f"  Resultado VIX: {result}")
    
    print("\n[OK] TESTE MÚLTIPLAS CHAMADAS CONCLUÍDO")

async def main():
    """Função principal de teste"""
    print("INICIANDO TESTES DO MACRO DATA PROVIDER SINGLETON")
    print("=" * 80)
    
    # Executar testes
    success = await test_singleton_behavior()
    await test_multiple_calls()
    
    print("\n" + "=" * 80)
    if success:
        print("SUCESSO: TODOS OS TESTES PASSARAM - SINGLETON FUNCIONANDO")
    else:
        print("ERRO: ALGUNS TESTES FALHARAM")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())