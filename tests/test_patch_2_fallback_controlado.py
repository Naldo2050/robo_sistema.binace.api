#!/usr/bin/env python3
"""
Teste para verificar se o Patch 2 está funcionando corretamente.

Este teste verifica:
1. Se provider=groq e todos os modelos Groq falharem, o sistema vai para MOCK
2. Não faz fallback automático para OpenAI sem configuração explícita
3. Só faz fallback se provider_fallbacks estiver configurado
"""

import sys
import os
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import logging

# Adiciona o diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_analyzer_qwen import AIAnalyzer


class TestPatch2FallbackControlado(unittest.TestCase):
    """Testes para o Patch 2 - Fallback Controlado"""
    
    def setUp(self):
        """Configuração para cada teste"""
        # Capturar logs para verificação
        self.logs = []
        
        # Mock do logger
        self.mock_logger = MagicMock()
        
        # Configurar tempfile para config.json
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.json")
        
    def tearDown(self):
        """Limpeza após cada teste"""
        # Remove arquivo temporário
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        os.rmdir(self.temp_dir)
        
    def _create_test_config(self, provider="groq", provider_fallbacks=None):
        """Cria arquivo de configuração de teste"""
        config = {
            "ai": {
                "provider": provider,
                "provider_fallbacks": provider_fallbacks or [],
                "groq": {
                    "model": "llama-3.3-70b-versatile",
                    "model_fallbacks": [
                        "llama-3.1-70b-versatile",
                        "llama-3.1-8b-instant"
                    ]
                }
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f)
            
    def _mock_groq_failure(self):
        """Mock para fazer Groq falhar em todos os modelos"""
        def mock_chat_completions_create(*args, **kwargs):
            raise Exception("Modelo Groq não disponível ou chave inválida")
            
        return patch('openai.OpenAI.chat.completions.create', side_effect=mock_chat_completions_create)
        
    def _mock_openai_success(self):
        """Mock para fazer OpenAI funcionar"""
        def mock_chat_completions_create(*args, **kwargs):
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = "OpenAI funcionando"
            return response
            
        return patch('openai.OpenAI.chat.completions.create', side_effect=mock_chat_completions_create)
        
    def test_patch_2_groq_fail_sem_fallback_vai_para_mock(self):
        """Teste: Groq falha e não há fallback configurado → deve ir para MOCK"""
        print("\nTestando: Groq falha sem fallback configurado")
        
        # Configurar sem fallback
        self._create_test_config(provider="groq", provider_fallbacks=[])
        
        # Simular falha do Groq
        with self._mock_groq_failure():
            # Criar analyzer
            analyzer = AIAnalyzer()
            
            # Deve estar em modo MOCK
            self.assertIsNone(analyzer.mode, "Modo deve ser None (mock)")
            self.assertTrue(analyzer.enabled, "Deve estar enabled")
            
        print("PASSOU: Groq falhou e foi para MOCK (sem fallback automático)")
        
    def test_patch_2_groq_fail_com_fallback_openai(self):
        """Teste: Groq falha mas há fallback configurado para OpenAI → deve tentar OpenAI"""
        print("\nTestando: Groq falha com fallback configurado para OpenAI")
        
        # Configurar com fallback para OpenAI
        self._create_test_config(provider="groq", provider_fallbacks=["openai"])
        
        # Simular falha do Groq e sucesso do OpenAI
        with self._mock_groq_failure():
            with self._mock_openai_success():
                # Criar analyzer
                analyzer = AIAnalyzer()
                
                # Deve estar em modo OpenAI (fallback)
                self.assertEqual(analyzer.mode, "openai", "Modo deve ser 'openai' (fallback)")
                self.assertTrue(analyzer.enabled, "Deve estar enabled")
                
        print("PASSOU: Groq falhou e fez fallback para OpenAI (configurado)")
        
    def test_patch_2_groq_funciona_normal(self):
        """Teste: Groq funciona normalmente → deve usar Groq"""
        print("\nTestando: Groq funciona normalmente")
        
        # Configurar sem fallback (não deve ser necessário)
        self._create_test_config(provider="groq", provider_fallbacks=[])
        
        # Simular sucesso do Groq
        def mock_groq_success(*args, **kwargs):
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = "Groq funcionando"
            return response
            
        with patch('openai.OpenAI.chat.completions.create', side_effect=mock_groq_success):
            # Criar analyzer
            analyzer = AIAnalyzer()
            
            # Deve estar em modo Groq
            self.assertEqual(analyzer.mode, "groq", "Modo deve ser 'groq'")
            self.assertTrue(analyzer.enabled, "Deve estar enabled")
            
        print("PASSOU: Groq funcionou normalmente")
        
    def test_patch_2_provider_nao_groq_vai_para_openai(self):
        """Teste: Provider não é Groq → deve tentar OpenAI normalmente"""
        print("\nTestando: Provider não é Groq")
        
        # Configurar provider como openai
        self._create_test_config(provider="openai", provider_fallbacks=[])
        
        # Simular sucesso do OpenAI
        with self._mock_openai_success():
            # Criar analyzer
            analyzer = AIAnalyzer()
            
            # Deve estar em modo OpenAI
            self.assertEqual(analyzer.mode, "openai", "Modo deve ser 'openai'")
            self.assertTrue(analyzer.enabled, "Deve estar enabled")
            
        print("PASSOU: Provider OpenAI funcionou normalmente")
        
    def test_patch_2_multiple_fallbacks(self):
        """Teste: Múltiplos fallbacks configurados"""
        print("\nTestando: Múltiplos fallbacks configurados")
        
        # Configurar com múltiplos fallbacks
        self._create_test_config(provider="groq", provider_fallbacks=["openai", "dashscope"])
        
        # Simular falha do Groq e sucesso do OpenAI
        with self._mock_groq_failure():
            with self._mock_openai_success():
                # Criar analyzer
                analyzer = AIAnalyzer()
                
                # Deve estar em modo OpenAI (primeiro fallback que funcionou)
                self.assertEqual(analyzer.mode, "openai", "Modo deve ser 'openai' (primeiro fallback)")
                self.assertTrue(analyzer.enabled, "Deve estar enabled")
                
        print("PASSOU: Múltiplos fallbacks funcionaram (OpenAI foi o primeiro)")


def run_patch2_tests():
    """Executa os testes do Patch 2"""
    print("=" * 80)
    print("TESTES DO PATCH 2 - FALLBACK CONTROLADO")
    print("=" * 80)
    
    # Configurar logging mínimo
    logging.basicConfig(level=logging.WARNING)
    
    # Executar testes
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPatch2FallbackControlado)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print("TODOS OS TESTES DO PATCH 2 PASSARAM!")
        print("O fallback automático foi controlado com sucesso")
    else:
        print("ALGUNS TESTES FALHARAM")
        print(f"   Falhas: {len(result.failures)}")
        print(f"   Erros: {len(result.errors)}")
    print("=" * 80)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_patch2_tests()
    sys.exit(0 if success else 1)