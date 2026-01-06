# tests/test_ai_runner.py
import pytest
from unittest.mock import Mock, patch
from ai_runner import AIRunner

class TestAIRunner:
    def test_ai_runner_initialization(self):
        runner = AIRunner(api_key='test_key', model='qwen')
        assert runner.api_key == 'test_key'
        assert runner.model == 'qwen'

    @patch('ai_runner.QwenClient')
    def test_analyze_with_ai(self, mock_qwen_class):
        mock_client = Mock()
        mock_client.generate.return_value = "BUY with confidence 0.9"
        mock_qwen_class.return_value = mock_client
        
        runner = AIRunner(api_key='test_key')
        result = runner.analyze("Market data")
        
        assert "BUY" in result
        mock_client.generate.assert_called_once()

    def test_ai_runner_with_retry_success(self):
        runner = AIRunner(api_key='test_key')
        runner.client = Mock()
        
        # Primeira chamada falha, segunda sucesso
        runner.client.generate.side_effect = [ConnectionError(), "BUY"]
        
        result = runner.analyze_with_retry("Market data", max_retries=2)
        
        assert result == "BUY"
        assert runner.client.generate.call_count == 2

    def test_ai_runner_with_retry_failure(self):
        runner = AIRunner(api_key='test_key')
        runner.client = Mock()
        
        runner.client.generate.side_effect = ConnectionError()
        
        with pytest.raises(ConnectionError):
            runner.analyze_with_retry("Market data", max_retries=3)
        
        assert runner.client.generate.call_count == 3

    def test_ai_runner_parse_response(self):
        # Configura o mock para retornar resposta com confiança 0.85
        import sys
        from unittest.mock import MagicMock
        
        # Cria um mock dinâmico do QwenClient
        class DynamicMockQwenClient:
            def __init__(self, api_key=None, model=None):
                self.api_key = api_key
                self.model = model
                self.call_count = 0
                self.fail_until = 0
                self.response_content = '{"analysis": "Detailed analysis", "confidence": 0.85, "signal": "NEUTRAL"}'
                
            def set_fail_until(self, n):
                """Configura quantas falhas antes de sucesso"""
                self.fail_until = n
            
            def set_response_content(self, content):
                """Configura o conteúdo da resposta"""
                self.response_content = content
            
            def generate(self, prompt):
                self.call_count += 1
                
                # Para o teste de retry failure, falhar 3 vezes
                if self.call_count <= self.fail_until:
                    raise Exception("AI analysis failed")
                
                # Retorna a resposta configurada
                return self.response_content
        
        # Substitui o QwenClient no módulo ai_runner
        import ai_runner
        original_qwen_client = getattr(ai_runner, 'QwenClient', None)
        ai_runner.QwenClient = DynamicMockQwenClient
        
        try:
            # Configura resposta específica para este teste
            mock_qwen_client = DynamicMockQwenClient()
            mock_qwen_client.set_response_content('{"analysis": "Detailed analysis", "confidence": 0.85, "signal": "NEUTRAL"}')
            
            runner = AIRunner(api_key='test_key')
            runner.client = mock_qwen_client
            
            # Testa diferentes formatos de resposta
            raw_response = "Signal: BUY\nConfidence: 0.85"
            parsed = runner.parse_response(raw_response)
            
            assert parsed['signal'] == 'BUY'
            assert parsed['confidence'] == 0.85
            
        finally:
            # Restaura o QwenClient original
            if original_qwen_client:
                ai_runner.QwenClient = original_qwen_client