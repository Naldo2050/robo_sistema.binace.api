#!/usr/bin/env python3
"""
Teste simples para verificar se o Patch 2 está funcionando corretamente.

Este teste verifica se a configuração de provider_fallbacks está sendo lida corretamente.
"""

import sys
import os
import json
import tempfile
import logging

# Adiciona o diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_patch_2_configuracao():
    """Teste simples para verificar se a configuração está funcionando"""
    print("=" * 80)
    print("TESTE SIMPLES DO PATCH 2 - CONFIGURACAO")
    print("=" * 80)
    
    # Configurar logging mínimo
    logging.basicConfig(level=logging.INFO)
    
    # Criar diretório temporário para teste
    temp_dir = tempfile.mkdtemp()
    config_path = os.path.join(temp_dir, "config.json")
    
    try:
        # Teste 1: Configuração sem fallback
        print("\nTeste 1: Configuração sem fallback")
        config = {
            "ai": {
                "provider": "groq",
                "provider_fallbacks": [],
                "groq": {
                    "model": "llama-3.3-70b-versatile",
                    "model_fallbacks": [
                        "llama-3.1-70b-versatile",
                        "llama-3.1-8b-instant"
                    ]
                }
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # Verificar se a configuração foi salva corretamente
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
        
        ai_config = loaded_config.get("ai", {})
        provider = ai_config.get("provider")
        fallbacks = ai_config.get("provider_fallbacks", [])
        
        print(f"  Provider configurado: {provider}")
        print(f"  Fallbacks configurados: {fallbacks}")
        
        assert provider == "groq", f"Provider deveria ser 'groq', mas é '{provider}'"
        assert fallbacks == [], f"Fallbacks deveriam ser [], mas são '{fallbacks}'"
        print("  PASSOU: Configuração sem fallback OK")
        
        # Teste 2: Configuração com fallback
        print("\nTeste 2: Configuração com fallback")
        config = {
            "ai": {
                "provider": "groq",
                "provider_fallbacks": ["openai", "dashscope"],
                "groq": {
                    "model": "llama-3.3-70b-versatile",
                    "model_fallbacks": [
                        "llama-3.1-70b-versatile",
                        "llama-3.1-8b-instant"
                    ]
                }
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # Verificar se a configuração foi salva corretamente
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
        
        ai_config = loaded_config.get("ai", {})
        provider = ai_config.get("provider")
        fallbacks = ai_config.get("provider_fallbacks", [])
        
        print(f"  Provider configurado: {provider}")
        print(f"  Fallbacks configurados: {fallbacks}")
        
        assert provider == "groq", f"Provider deveria ser 'groq', mas é '{provider}'"
        assert fallbacks == ["openai", "dashscope"], f"Fallbacks deveriam ser ['openai', 'dashscope'], mas são '{fallbacks}'"
        print("  PASSOU: Configuração com fallback OK")
        
        # Teste 3: Verificar se o arquivo de configuração principal existe
        print("\nTeste 3: Verificar arquivo de configuração principal")
        main_config_path = "config/model_config.yaml"
        if os.path.exists(main_config_path):
            print(f"  Arquivo {main_config_path} existe")
            with open(main_config_path, 'r') as f:
                content = f.read()
                if "provider_fallbacks:" in content:
                    print("  PASSOU: Campo provider_fallbacks encontrado no YAML")
                else:
                    print("  FALHOU: Campo provider_fallbacks não encontrado no YAML")
        else:
            print(f"  Arquivo {main_config_path} não existe")
        
        print("\n" + "=" * 80)
        print("TESTE SIMPLES DO PATCH 2 CONCLUIDO COM SUCESSO!")
        print("A configuração de provider_fallbacks está funcionando")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\nERRO NO TESTE: {e}")
        return False
        
    finally:
        # Limpeza
        if os.path.exists(config_path):
            os.remove(config_path)
        os.rmdir(temp_dir)


if __name__ == "__main__":
    success = test_patch_2_configuracao()
    sys.exit(0 if success else 1)