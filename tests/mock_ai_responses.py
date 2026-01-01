"""
Mocks para respostas da IA usados nos testes.
"""

# Resposta mockada do QwenClient
MOCK_QWEN_RESPONSE = {
    "choices": [
        {
            "message": {
                "content": "Análise de teste: o mercado mostra sinais de consolidação."
            }
        }
    ]
}

# Resposta mockada para a análise de IA
MOCK_AI_ANALYSIS = {
    "content": "Análise de teste concluída com sucesso.",
    "confidence": 0.85,
    "signal": "NEUTRAL",
    "reasoning": "Teste de análise de mercado.",
    "timestamp": 1234567890
}

# Mock do cliente Qwen
class MockQwenClient:
    def __init__(self, *args, **kwargs):
        # Mock the chat completions interface like OpenAI
        self.chat = type('obj', (object,), {
            'completions': type('obj', (object,), {
                'create': self._create_completion
            })
        })
    
    def _create_completion(self, **kwargs):
        """Mock completion creation that returns proper OpenAI-style response"""
        return type('obj', (object,), {
            'choices': [type('obj', (object,), {
                'message': type('obj', (object,), {
                    'content': "Análise de teste: o mercado mostra sinais de consolidação."
                })
            })],
            'usage': type('obj', (object,), {
                'prompt_tokens': 100,
                'completion_tokens': 50,
                'total_tokens': 150
            })
        })
    
    def call(self, messages, **kwargs):
        """Legacy method for backward compatibility"""
        return type('obj', (object,), {
            'choices': [type('obj', (object,), {
                'message': type('obj', (object,), {
                    'content': "Análise de teste: o mercado mostra sinais de consolidação."
                })
            })]
        })