# tests/fix_qwen_import.py
"""
Fix para importação do QwenClient nos testes.
Este arquivo deve ser importado ANTES de qualquer teste.
"""

import sys
import types

# Cria um módulo 'qwen' fake
fake_qwen_module = types.ModuleType('qwen')

# Cria a classe QwenClient dentro do módulo fake
class QwenClient:
    def __init__(self, api_key=None, model=None):
        self.api_key = api_key
        self.model = model
    
    def call(self, messages, **kwargs):
        class MockResponse:
            def __init__(self):
                self.choices = [self.MockChoice()]
            
            class MockChoice:
                class MockMessage:
                    content = '{"analysis": "Test", "confidence": 0.85, "signal": "NEUTRAL"}'
                message = MockMessage()
        
        return MockResponse()

# Adiciona a classe ao módulo
fake_qwen_module.QwenClient = QwenClient

# Injeta o módulo fake no sys.modules
sys.modules['qwen'] = fake_qwen_module

# Verifica se ai_runner existe e injeta também
if 'ai_runner' in sys.modules:
    sys.modules['ai_runner'].QwenClient = QwenClient