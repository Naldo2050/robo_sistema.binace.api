# tests/mock_qwen.py
"""
Mock do QwenClient para testes.
"""

class MockQwenClient:
    def __init__(self, api_key=None, model=None):
        pass
    
    def call(self, messages, **kwargs):
        # Retorna uma resposta mockada no formato que os testes esperam
        class MockResponse:
            class Choice:
                class Message:
                    content = "Análise de teste gerada por mock"
                message = Message()
            choices = [Choice()]
        
        return MockResponse()