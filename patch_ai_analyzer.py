#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para aplicar patches de seguranca e validacao JSON no ai_analyzer_qwen.py
"""

import re
import os
import sys

# Configura encoding para UTF-8
try:
    sys.stdout.reconfigure(encoding='utf-8')  # type: ignore[attr-defined]
except AttributeError:
    pass  # Python < 3.7

def apply_patches():
    filepath = "ai_analyzer_qwen.py"
    
    # Le o arquivo com encoding UTF-8
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    original_content = content
    
    # 1. Atualiza a versao no cabecalho
    content = re.sub(
        r'# ai_analyzer_qwen\.py v[\d.]+ - [^\n]+',
        '# ai_analyzer_qwen.py v2.6.0 - JSON ESTRITO COM VALIDACAO RIGIDA',
        content,
        count=1
    )
    
    # 2. Remove o log que expoe a chave Groq
    # Padrao: logging.info(f"... GroqCloud ATIVO ... Chave: {groq_key[:10]}...{groq_key[-4:]}")
    content = re.sub(
        r'logging\.info\(\s*f"[^"]*GroqCloud ATIVO[^"]*Chave:[^"]*\{groq_key[^}]*\}[^"]*"\s*\)',
        'logging.info(f"GroqCloud ATIVO | Modelo: {self.model_name}")',
        content
    )
    
    # 3. Aplica patches no fred_fetcher.py
    fred_filepath = "fred_fetcher.py"
    if os.path.exists(fred_filepath):
        with open(fred_filepath, "r", encoding="utf-8") as f:
            fred_content = f.read()
        
        # Padrao: logger.info(f"... FRED API inicializada ... Key: {self.api_key[:8]}...")
        fred_content = re.sub(
            r'logger\.info\(f"[^"]*FRED API inicializada[^"]*Key:[^"]*\{self\.api_key[^}]*\}[^"]*"\s*\)',
            'logger.info("FRED API inicializada")',
            fred_content
        )
        
        with open(fred_filepath, "w", encoding="utf-8") as f:
            f.write(fred_content)
        print("Patched " + fred_filepath)
    
    # 4. Adiciona import do validador
    validator_import = '''
# LLM Response Validator - Validacao JSON estrita
try:
    from market_orchestrator.ai.llm_response_validator import (
        validate_llm_response,
        sanitize_for_log,
        FALLBACK_RESPONSE,
    )
    LLM_VALIDATOR_AVAILABLE = True
except ImportError:
    LLM_VALIDATOR_AVAILABLE = False
    validate_llm_response = None
    sanitize_for_log = lambda text, max_len=200: (text or "")[:max_len]
    FALLBACK_RESPONSE = {
        "sentiment": "neutral",
        "confidence": 0.0,
        "action": "wait",
        "rationale": "validator_not_available",
        "entry_zone": None,
        "invalidation_zone": None,
        "region_type": None,
        "_fallback": True,
    }
'''
    
    # Insere apos os imports do payload_metrics_aggregator
    import_marker = "except ImportError:\n    def _summarize_metrics(*args: Any, **kwargs: Any) -> Dict[str, Any]:\n        return {}"
    
    if import_marker in content and "LLM_VALIDATOR_AVAILABLE" not in content:
        content = content.replace(
            import_marker,
            import_marker + "\n" + validator_import
        )
    
    # Salva o arquivo modificado
    if content != original_content:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print("Patched " + filepath)
    else:
        print("No changes needed for " + filepath)

if __name__ == "__main__":
    apply_patches()
