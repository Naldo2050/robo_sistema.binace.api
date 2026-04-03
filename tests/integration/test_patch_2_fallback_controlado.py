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


def _make_failing_client():
    """Retorna instância _OpenAI onde chat.completions.create sempre falha."""
    instance = MagicMock()
    instance.chat.completions.create.side_effect = Exception("API error — connection refused")
    return instance


def _make_succeeding_client():
    """Retorna instância _OpenAI onde chat.completions.create sempre tem sucesso."""
    instance = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "OK response"
    instance.chat.completions.create.return_value = response
    return instance


def _openai_factory_fail_groq_succeed_openai(*args, **kwargs):
    """
    Cria um cliente que:
    - Falha o ping se base_url contém 'groq' (Groq client)
    - Tem sucesso se não contém 'groq' (OpenAI client)
    """
    if 'groq' in str(kwargs.get('base_url', '')):
        return _make_failing_client()
    return _make_succeeding_client()


def _openai_factory_all_succeed(*args, **kwargs):
    return _make_succeeding_client()


def _openai_factory_all_fail(*args, **kwargs):
    return _make_failing_client()


class TestPatch2FallbackControlado(unittest.TestCase):
    """Testes para o Patch 2 - Fallback Controlado"""

    def _patch_config(self, provider="groq", provider_fallbacks=None):
        """Injeta config diretamente no AIAnalyzer._load_config."""
        cfg = {
            "ai": {
                "provider": provider,
                "provider_fallbacks": provider_fallbacks or [],
                "groq": {
                    "model": "llama-3.1-8b-instant",
                    "model_fallbacks": ["llama-3.1-8b-instant"],
                }
            }
        }

        def fake_load_config(self_inner):
            self_inner.config = cfg

        return patch.object(AIAnalyzer, '_load_config', fake_load_config)

    def test_patch_2_groq_fail_sem_fallback_vai_para_mock(self):
        """Teste: Groq falha e não há fallback configurado → deve ir para MOCK"""
        print("\nTestando: Groq falha sem fallback configurado")

        with self._patch_config(provider="groq", provider_fallbacks=[]):
            with patch('ai_analyzer_qwen._OpenAI', side_effect=_openai_factory_all_fail):
                analyzer = AIAnalyzer()

        self.assertIsNone(analyzer.mode, "Modo deve ser None (mock)")
        self.assertTrue(analyzer.enabled, "Deve estar enabled")
        print("PASSOU: Groq falhou e foi para MOCK (sem fallback automático)")

    def test_patch_2_groq_fail_com_fallback_openai(self):
        """Teste: Groq falha mas há fallback para OpenAI → deve tentar OpenAI"""
        print("\nTestando: Groq falha com fallback configurado para OpenAI")

        with self._patch_config(provider="groq", provider_fallbacks=["openai"]):
            with patch('ai_analyzer_qwen._OpenAI', side_effect=_openai_factory_fail_groq_succeed_openai):
                analyzer = AIAnalyzer()

        # OpenAI path não faz ping — apenas cria cliente → mode = "openai"
        self.assertEqual(analyzer.mode, "openai", "Modo deve ser 'openai' (fallback)")
        self.assertTrue(analyzer.enabled, "Deve estar enabled")
        print("PASSOU: Groq falhou e fez fallback para OpenAI (configurado)")

    def test_patch_2_groq_funciona_normal(self):
        """Teste: Groq funciona normalmente → deve usar Groq"""
        print("\nTestando: Groq funciona normalmente")

        with self._patch_config(provider="groq", provider_fallbacks=[]):
            with patch('ai_analyzer_qwen._OpenAI', side_effect=_openai_factory_all_succeed):
                analyzer = AIAnalyzer()

        self.assertEqual(analyzer.mode, "groq", "Modo deve ser 'groq'")
        self.assertTrue(analyzer.enabled, "Deve estar enabled")
        print("PASSOU: Groq funcionou normalmente")

    def test_patch_2_provider_nao_groq_vai_para_openai(self):
        """Teste: Provider não é Groq → deve usar OpenAI normalmente"""
        print("\nTestando: Provider não é Groq")

        with self._patch_config(provider="openai", provider_fallbacks=[]):
            with patch('ai_analyzer_qwen._OpenAI', side_effect=_openai_factory_all_succeed):
                analyzer = AIAnalyzer()

        # Com provider="openai", vai direto para _try_initialize_openai → mode = "openai"
        self.assertEqual(analyzer.mode, "openai", "Modo deve ser 'openai'")
        self.assertTrue(analyzer.enabled, "Deve estar enabled")
        print("PASSOU: Provider OpenAI funcionou normalmente")

    def test_patch_2_multiple_fallbacks(self):
        """Teste: Múltiplos fallbacks configurados → primeiro que funcionar"""
        print("\nTestando: Múltiplos fallbacks configurados")

        with self._patch_config(provider="groq", provider_fallbacks=["openai", "dashscope"]):
            with patch('ai_analyzer_qwen._OpenAI', side_effect=_openai_factory_fail_groq_succeed_openai):
                analyzer = AIAnalyzer()

        # Groq falha → tenta OpenAI (primeiro fallback) → OpenAI succeeds → mode = "openai"
        self.assertEqual(analyzer.mode, "openai", "Modo deve ser 'openai' (primeiro fallback)")
        self.assertTrue(analyzer.enabled, "Deve estar enabled")
        print("PASSOU: Múltiplos fallbacks funcionaram (OpenAI foi o primeiro)")


def run_patch2_tests():
    """Executa os testes do Patch 2"""
    print("=" * 80)
    print("TESTES DO PATCH 2 - FALLBACK CONTROLADO")
    print("=" * 80)

    logging.basicConfig(level=logging.WARNING)

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPatch2FallbackControlado)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_patch2_tests()
    sys.exit(0 if success else 1)
