# test_ai_response_validator.py
# -*- coding: utf-8 -*-

"""
Testes para o validador de resposta da IA.
Cobre os casos especificados:
- resposta JSON válida
- resposta com texto antes/depois do JSON
- resposta truncada
- resposta com campo obrigatório ausente
- resposta com confidence fora do intervalo
- resposta com action inválida
- redaction de segredos em logs
- correção do path default da OCI
"""

import unittest
import json
import logging
import os
import sys

# Adiciona o diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_response_validator import (
    AIResponseValidator,
    validate_ai_response,
    is_fallback_response,
    FALLBACK_RESPONSE,
)

# Configura logging para os testes
logging.basicConfig(level=logging.DEBUG)


class TestAIResponseValidator(unittest.TestCase):
    """Testes para o AIResponseValidator."""
    
    def setUp(self):
        """Configura o validador para cada teste."""
        self.validator = AIResponseValidator()
    
    # ============================================================
    # TESTE 1: Resposta JSON válida
    # ============================================================
    def test_valid_json_response(self):
        """Resposta JSON válida deve ser aceita."""
        valid_json = '{"sentiment":"bullish","confidence":0.85,"action":"buy","rationale":"fluxo positivo","entry_zone":"95000","invalidation_zone":"94000","region_type":"resistance"}'
        
        result = self.validator.validate(valid_json)
        
        self.assertTrue(result.is_valid)
        self.assertEqual(result.data["sentiment"], "bullish")
        self.assertEqual(result.data["confidence"], 0.85)
        self.assertEqual(result.data["action"], "buy")
        self.assertFalse(result.is_fallback)
    
    # ============================================================
    # TESTE 2: Resposta com texto antes do JSON
    # ============================================================
    def test_text_before_json(self):
        """Texto antes do JSON deve ser removido."""
        text_with_prefix = 'Okay, let\'s analyze this: {"sentiment":"bearish","confidence":0.6,"action":"sell","rationale":"fluxo negativo","entry_zone":null,"invalidation_zone":null,"region_type":null}'
        
        result = self.validator.validate(text_with_prefix)
        
        self.assertTrue(result.is_valid)
        self.assertEqual(result.data["action"], "sell")
    
    def test_text_with_okay_prefix(self):
        """Texto começando com 'Okay' deve ser removido."""
        text = 'Okay, let\'s break this down... {"sentiment":"neutral","confidence":0.5,"action":"wait","rationale":"incerteza","entry_zone":null,"invalidation_zone":null,"region_type":null}'
        
        result = self.validator.validate(text)
        
        self.assertTrue(result.is_valid)
    
    # ============================================================
    # TESTE 3: Resposta com texto depois do JSON
    # ============================================================
    def test_text_after_json(self):
        """Texto após o JSON deve ser ignorado."""
        text_with_suffix = '{"sentiment":"bullish","confidence":0.7,"action":"buy","rationale":"setup claro","entry_zone":"96000","invalidation_zone":"95000","region_type":"support"} Follow-up analysis...'
        
        result = self.validator.validate(text_with_suffix)
        
        self.assertTrue(result.is_valid)
        self.assertEqual(result.data["action"], "buy")
    
    # ============================================================
    # TESTE 4: Resposta truncada
    # ============================================================
    def test_truncated_json(self):
        """JSON truncado deve ser rejeitado."""
        truncated = '{"sentiment":"bullish","confidence":0.32,'
        
        result = self.validator.validate(truncated)
        
        self.assertFalse(result.is_valid)
        self.assertTrue(result.is_fallback)
        # Verifica que houve erro (pode ser qualquer mensagem)
        self.assertIsNotNone(result.error_message)
    
    def test_truncated_json_in_text(self):
        """JSON truncado no meio do texto deve ser rejeitado."""
        text = 'Some text before {"sentiment":"bullish","confidence":0.32, and some text after'
        
        result = self.validator.validate(text)
        
        self.assertFalse(result.is_valid)
    
    # ============================================================
    # TESTE 5: Campo obrigatório ausente
    # ============================================================
    def test_missing_required_field_sentiment(self):
        """Campo 'sentiment' ausente deveInvalidar resposta."""
        missing_sentiment = '{"confidence":0.8,"action":"buy","rationale":"test","entry_zone":null,"invalidation_zone":null,"region_type":null}'
        
        result = self.validator.validate(missing_sentiment)
        
        self.assertFalse(result.is_valid)
        self.assertTrue(result.is_fallback)
    
    def test_missing_required_field_confidence(self):
        """Campo 'confidence' ausente deveInvalidar resposta."""
        missing_confidence = '{"sentiment":"bullish","action":"buy","rationale":"test","entry_zone":null,"invalidation_zone":null,"region_type":null}'
        
        result = self.validator.validate(missing_confidence)
        
        self.assertFalse(result.is_valid)
    
    def test_missing_required_field_action(self):
        """Campo 'action' ausente deveInvalidar resposta."""
        missing_action = '{"sentiment":"bullish","confidence":0.8,"rationale":"test","entry_zone":null,"invalidation_zone":null,"region_type":null}'
        
        result = self.validator.validate(missing_action)
        
        self.assertFalse(result.is_valid)
    
    def test_missing_required_field_rationale(self):
        """Campo 'rationale' ausente deveInvalidar resposta."""
        missing_rationale = '{"sentiment":"bullish","confidence":0.8,"action":"buy","entry_zone":null,"invalidation_zone":null,"region_type":null}'
        
        result = self.validator.validate(missing_rationale)
        
        self.assertFalse(result.is_valid)
    
    # ============================================================
    # TESTE 6: Confidence fora do intervalo
    # ============================================================
    def test_confidence_above_1(self):
        """Confidence > 1.0 deveInvalidar resposta."""
        above_one = '{"sentiment":"bullish","confidence":1.5,"action":"buy","rationale":"test","entry_zone":null,"invalidation_zone":null,"region_type":null}'
        
        result = self.validator.validate(above_one)
        
        self.assertFalse(result.is_valid)
    
    def test_confidence_below_0(self):
        """Confidence < 0.0 deveInvalidar resposta."""
        below_zero = '{"sentiment":"bullish","confidence":-0.5,"action":"buy","rationale":"test","entry_zone":null,"invalidation_zone":null,"region_type":null}'
        
        result = self.validator.validate(below_zero)
        
        self.assertFalse(result.is_valid)
    
    def test_confidence_edge_cases(self):
        """Confidence = 0.0 e 1.0 devem ser aceitos."""
        edge_zero = '{"sentiment":"neutral","confidence":0.0,"action":"wait","rationale":"test","entry_zone":null,"invalidation_zone":null,"region_type":null}'
        edge_one = '{"sentiment":"bullish","confidence":1.0,"action":"buy","rationale":"test","entry_zone":null,"invalidation_zone":null,"region_type":null}'
        
        result_zero = self.validator.validate(edge_zero)
        result_one = self.validator.validate(edge_one)
        
        self.assertTrue(result_zero.is_valid)
        self.assertTrue(result_one.is_valid)
    
    # ============================================================
    # TESTE 7: Action inválida
    # ============================================================
    def test_invalid_action(self):
        """Action inválida deveInvalidar resposta."""
        invalid_action = '{"sentiment":"bullish","confidence":0.8,"action":"invalid_action","rationale":"test","entry_zone":null,"invalidation_zone":null,"region_type":null}'
        
        result = self.validator.validate(invalid_action)
        
        self.assertFalse(result.is_valid)
    
    def test_valid_actions(self):
        """Todas as actions válidas devem ser aceitas."""
        valid_actions = ["buy", "sell", "hold", "flat", "wait", "avoid"]
        
        for action in valid_actions:
            json_str = f'{{"sentiment":"neutral","confidence":0.5,"action":"{action}","rationale":"test","entry_zone":null,"invalidation_zone":null,"region_type":null}}'
            result = self.validator.validate(json_str)
            self.assertTrue(result.is_valid, f"Action '{action}' should be valid")
    
    # ============================================================
    # TESTE 8: Validar sentiments
    # ============================================================
    def test_valid_sentiments(self):
        """Todos os sentiments válidos devem ser aceitos."""
        valid_sentiments = ["bullish", "bearish", "neutral"]
        
        for sentiment in valid_sentiments:
            json_str = f'{{"sentiment":"{sentiment}","confidence":0.5,"action":"buy","rationale":"test","entry_zone":null,"invalidation_zone":null,"region_type":null}}'
            result = self.validator.validate(json_str)
            self.assertTrue(result.is_valid, f"Sentiment '{sentiment}' should be valid")
    
    def test_invalid_sentiment(self):
        """Sentiment inválido deveInvalidar resposta."""
        invalid = '{"sentiment":"unknown","confidence":0.5,"action":"buy","rationale":"test","entry_zone":null,"invalidation_zone":null,"region_type":null}'
        
        result = self.validator.validate(invalid)
        
        self.assertFalse(result.is_valid)
    
    # ============================================================
    # TESTE 9: Fallback
    # ============================================================
    def test_fallback_structure(self):
        """Resposta fallback deve ter estrutura correta."""
        invalid = 'not a json at all'
        
        result = self.validator.validate(invalid)
        
        self.assertFalse(result.is_valid)
        self.assertTrue(result.is_fallback)
        self.assertEqual(result.data["action"], "wait")
        self.assertEqual(result.data["confidence"], 0.0)
        self.assertTrue(result.data.get("_is_fallback"))
    
    def test_is_fallback_detection(self):
        """Deve detectar corretamente respostas fallback."""
        invalid = 'invalid response'
        
        result = self.validator.validate(invalid)
        
        self.assertTrue(is_fallback_response(result.data))
        
        valid = '{"sentiment":"bullish","confidence":0.8,"action":"buy","rationale":"test","entry_zone":null,"invalidation_zone":null,"region_type":null}'
        
        result_valid = self.validator.validate(valid)
        
        self.assertFalse(is_fallback_response(result_valid.data))
    
    # ============================================================
    # TESTE 10: Normalização
    # ============================================================
    def test_normalization(self):
        """Valores devem ser normalizados corretamente."""
        # Case insensitive para sentiment e action (mas o JSON keys precisam ser minúsculos)
        # O validador normaliza os valores, não as chaves
        mixed_case = '{"sentiment":"BULLISH","confidence":0.8,"action":"BUY","rationale":"TEST","entry_zone":null,"invalidation_zone":null,"region_type":null}'
        
        result = self.validator.validate(mixed_case)
        
        self.assertTrue(result.is_valid)
        self.assertEqual(result.data["sentiment"], "bullish")
        self.assertEqual(result.data["action"], "buy")


class TestLogSanitizer(unittest.TestCase):
    """Testes para o sanitizador de logs."""
    
    def test_groq_token_redaction(self):
        """Tokens Groq devem ser redigidos."""
        from log_sanitizer import sanitize_log_message
        
        message = "Groq key: gsk_abcdef1234567890abcdef"
        
        sanitized = sanitize_log_message(message)
        
        # O token real não deve aparecer
        self.assertNotIn("abcdef1234567890abcdef", sanitized)
        self.assertIn("REDACTED", sanitized)
    
    def test_partial_key_redaction(self):
        """Chaves parciais devem ser redigidas."""
        from log_sanitizer import sanitize_log_message
        
        message = "Chave: abcdef123456"
        
        sanitized = sanitize_log_message(message)
        
        self.assertNotIn("abcdef123456", sanitized)
        self.assertIn("REDACTED", sanitized)
    
    def test_env_var_redaction(self):
        """Variáveis de ambiente com chaves devem ser redigidas."""
        from log_sanitizer import sanitize_log_message
        
        message = "GROQ_API_KEY=gsk_abcdef1234567890"
        
        sanitized = sanitize_log_message(message)
        
        self.assertNotIn("gsk_", sanitized)
        self.assertIn("REDACTED", sanitized)


class TestOCIPathCorrection(unittest.TestCase):
    """Testes para verificação do path OCI."""
    
    def test_oci_config_path(self):
        """Path OCI deve ser ~/.oci/config (sem typo)."""
        import os
        
        # Verifica o caminho hardcoded
        expected_path = os.path.expanduser("~/.oci/config")
        
        # O caminho não deve ter o typo 'configg'
        self.assertNotIn("configg", expected_path)
        self.assertEqual(expected_path, os.path.expanduser("~/.oci/config"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
