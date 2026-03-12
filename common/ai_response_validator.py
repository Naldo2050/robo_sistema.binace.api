# ai_response_validator.py
# -*- coding: utf-8 -*-

"""
Validador de resposta da IA.

Este módulo valida rigidamente as respostas JSON retornadas pelo LLM
antes de marcar como sucesso. Garante que:
- A resposta seja JSON válido
- Todos os campos obrigatórios estejam presentes
- Os valores estejam nos intervalos esperados
- Texto extra fora do JSON invalide a resposta
- Respostas truncadas sejam detectadas e rejeitadas
"""

import json
import re
import logging
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("ai_response_validator")

# Constantes para valores válidos
VALID_SENTIMENTS = {"bullish", "bearish", "neutral"}
VALID_ACTIONS = {"buy", "sell", "hold", "flat", "wait", "avoid"}

# Fallback estruturado para respostas inválidas
MAX_RATIONALE_CHARS = 160

FALLBACK_RESPONSE = {
    "sentiment": "neutral",
    "confidence": 0.0,
    "action": "wait",
    "rationale": "llm_error_unknown",
    "entry_zone": None,
    "invalidation_zone": None,
    "region_type": None,
    "_is_fallback": True,
    "_fallback_reason": "unknown",
    "_validation_error": None,
}


def _normalize_reason(reason: Optional[str]) -> str:
    """Normaliza o identificador de motivo de fallback."""
    text = str(reason or "unknown").strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def build_fallback_response(reason: Optional[str], *, error_key: str = "_validation_error") -> Dict[str, Any]:
    """Constroi o fallback estruturado único do pipeline de IA."""
    normalized_reason = _normalize_reason(reason)
    fallback = deepcopy(FALLBACK_RESPONSE)
    fallback["rationale"] = f"llm_error_{normalized_reason}"[:MAX_RATIONALE_CHARS]
    fallback["_fallback_reason"] = normalized_reason
    fallback[error_key] = normalized_reason
    return fallback


@dataclass
class ValidationResult:
    """Resultado da validação."""
    is_valid: bool
    data: Dict[str, Any]
    error_message: Optional[str]
    is_fallback: bool


class AIResponseValidator:
    """
    Validador de respostas JSON do LLM.
    
    Regras de validação:
    - JSON válido e bem formado
    - Campos obrigatórios: sentiment, confidence, action, rationale
    - confidence entre 0.0 e 1.0
    - sentiment em {bullish, bearish, neutral}
    - action em {buy, sell, hold, flat, wait, avoid}
    - rationale não vazio
    - Sem texto extra antes ou depois do JSON
    - Detecção de JSON truncado
    """
    
    REQUIRED_FIELDS = {"sentiment", "confidence", "action", "rationale"}
    OPTIONAL_FIELDS = {"entry_zone", "invalidation_zone", "region_type"}
    
    def __init__(self, enable_retry_prompt: bool = True, max_retries: int = 1):
        """
        Inicializa o validador.
        
        Args:
            enable_retry_prompt: Se True, pode gerar prompt mais restritivo para retry
            max_retries: Número máximo de tentativas de re-validação
        """
        self.enable_retry_prompt = enable_retry_prompt
        self.max_retries = max_retries
    
    def validate(self, raw_response: str) -> ValidationResult:
        """
        Valida uma resposta crua do LLM.
        
        Args:
            raw_response: Resposta textual do LLM
            
        Returns:
            ValidationResult com dados validados ou fallback
        """
        # Limpa a resposta
        cleaned = self._clean_response(raw_response)
        
        # Tenta extrair JSON
        parsed = self._extract_json(cleaned)
        
        if parsed is None:
            return ValidationResult(
                is_valid=False,
                data=self._get_fallback_response("json_parse_error"),
                error_message="Resposta não contém JSON válido",
                is_fallback=True,
            )
        
        # Valida campos obrigatórios
        validation_error = self._validate_fields(parsed)
        if validation_error:
            return ValidationResult(
                is_valid=False,
                data=self._get_fallback_response(validation_error),
                error_message=validation_error,
                is_fallback=True,
            )
        
        # Normaliza valores
        normalized = self._normalize_values(parsed)
        
        return ValidationResult(
            is_valid=True,
            data=normalized,
            error_message=None,
            is_fallback=False,
        )
    
    def _clean_response(self, response: str) -> str:
        """Limpa a resposta removendo markdown e caracteres indesejados."""
        if not isinstance(response, str):
            return ""
        
        text = response.strip()
        
        # Remove blocos de código markdown
        if text.startswith("```"):
            # Remove inicio e fim de bloco markdown
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        
        # Remove tags de pensamento
        text = re.sub(r"<THINK>.*?</THINK>", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = text.replace("<think>", "").replace("</think>", "")
        
        # Remove texto antes do JSON (como "Okay, let's break this down...")
        # Detecta e remove frases comuns de texto livre
        text = self._remove_free_text_prefix(text)
        
        return text.strip()
    
    def _remove_free_text_prefix(self, text: str) -> str:
        """Remove texto livre antes do JSON."""
        # Padrões de texto livre que precedem JSON
        free_text_patterns = [
            r"^(Okay|Ok|Alright|Let me|Let's|First|I will|I need to|I'll|Looking at|Analyzing|Based on|The data shows|Given the).*?[\n\r]+",
            r"^(Hmm|Well|So|Now)[\s:,\-]*.*?[\n\r]+",
            r"^Here['']s (the|my).*?[\n\r]+",
            r"^Sure[,\s]+.*?[\n\r]+",
            r"^Certainly[,\s]+.*?[\n\r]+",
            r"^Below is.*?[\n\r]+",
        ]
        
        # Se começa com { não precisa limpar
        if text.startswith("{"):
            return text
        
        # Tenta encontrar JSON no meio do texto
        json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        # Remove padrões de texto livre
        for pattern in free_text_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
        
        return text
    
    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extrai JSON do texto, detectando truncamento."""
        if not text:
            return None
        
        # Tenta parse direto
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        
        # Tenta encontrar JSON entre chaves
        start = text.find("{")
        end = text.rfind("}")
        
        if start == -1 or end == -1 or end <= start:
            return None
        
        json_str = text[start:end + 1]
        
        # Detecta truncamento: se o JSON termina abruptamente
        if self._is_truncated(json_str):
            logger.warning(f"JSON truncado detectado: {json_str[:100]}...")
            return None
        
        try:
            obj = json.loads(json_str)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        
        return None
    
    def _is_truncated(self, json_str: str) -> bool:
        """Detecta se o JSON parece estar truncado."""
        # Sinais de truncamento
        truncated_indicators = [
            # Termina no meio de uma string
            json_str.rstrip().endswith('"'),
            # Termina no meio de uma chave
            json_str.rstrip().endswith('":'),
            # Termina no meio de um valor
            json_str.rstrip().endswith(',') or json_str.rstrip().endswith(':'),
            # Tem menos de 30 caracteres (muito curto para um JSON válido)
            len(json_str) < 30,
        ]
        
        # Conta chaves desbalanceadas
        open_braces = json_str.count("{")
        close_braces = json_str.count("}")
        
        if open_braces != close_braces:
            return True
        
        return any(truncated_indicators)
    
    def _validate_fields(self, data: Dict[str, Any]) -> Optional[str]:
        """Valida campos obrigatórios e seus valores."""
        # Verifica campos obrigatórios
        missing = self.REQUIRED_FIELDS - set(data.keys())
        if missing:
            return f"Campos obrigatórios faltando: {missing}"
        
        # Valida sentiment
        sentiment = data.get("sentiment", "").lower()
        if sentiment not in VALID_SENTIMENTS:
            return f"sentiment inválido: '{sentiment}'. Deve ser um de: {VALID_SENTIMENTS}"
        
        # Valida confidence
        confidence = data.get("confidence")
        if confidence is None:
            return "confidence não pode ser null"
        
        try:
            confidence = float(confidence)
            if not (0.0 <= confidence <= 1.0):
                return f"confidence fora do intervalo [0,1]: {confidence}"
        except (ValueError, TypeError):
            return f"confidence inválido: {confidence}"
        
        # Valida action
        action = data.get("action", "").lower()
        if action not in VALID_ACTIONS:
            return f"action inválida: '{action}'. Deve ser um de: {VALID_ACTIONS}"
        
        # Valida rationale (não vazio)
        rationale = data.get("rationale", "")
        if not rationale or not isinstance(rationale, str) or len(rationale.strip()) == 0:
            return "rationale não pode ser vazio"
        
        return None
    
    def _normalize_values(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normaliza valores para tipos corretos."""
        normalized = dict(data)
        
        # Normaliza sentiment
        normalized["sentiment"] = data.get("sentiment", "neutral").lower()
        
        # Normaliza confidence para float
        try:
            normalized["confidence"] = float(data.get("confidence", 0.0))
        except (ValueError, TypeError):
            normalized["confidence"] = 0.0
        
        # Normaliza action
        normalized["action"] = data.get("action", "wait").lower()
        
        # Normaliza rationale
        normalized["rationale"] = str(data.get("rationale", "")).strip()[:MAX_RATIONALE_CHARS]

        # Preserva campos opcionais
        for field in self.OPTIONAL_FIELDS:
            normalized[field] = data.get(field)

        # Marca como não-fallback
        normalized["_is_fallback"] = False
        normalized["_fallback_reason"] = None
        normalized["_validation_error"] = None

        return normalized
    
    def _get_fallback_response(self, error_type: str) -> Dict[str, Any]:
        """Retorna o fallback estruturado."""
        return build_fallback_response(error_type)
    
    def get_retry_prompt(self) -> str:
        """
        Retorna um prompt mais restritivo para retry.
        
        Este prompt instrui o LLM a responder APENAS com JSON,
        sem texto adicional.
        """
        return """Responda APENAS com JSON válido, sem texto antes ou depois.

Formato obrigatório:
{"sentiment":"bullish|bearish|neutral","confidence":0.0-1.0,"action":"buy|sell|hold|flat|wait|avoid","rationale":"texto","entry_zone":null,"invalidation_zone":null,"region_type":null}

Não inclua explicações, raciocínio ou formatação markdown. Apenas o JSON."""
    
    def log_validation_failure(self, raw_response: str, error: str) -> None:
        """Loga falha de validação de forma segura (sem expor segredos)."""
        # Não logue a resposta bruta completa
        preview = raw_response[:200] if raw_response else "(vazio)"
        preview = re.sub(r"\s+", " ", preview).strip()

        logger.warning(
            f"AI_RESPONSE_INVALID | erro={error} | tamanho={len(raw_response)} | preview={preview}..."
        )


def validate_ai_response(raw_response: str) -> Tuple[Dict[str, Any], bool]:
    """
    Função de conveniência para validar resposta da IA.
    
    Args:
        raw_response: Resposta crua do LLM
        
    Returns:
        Tupla (dados_validados, is_valid)
    """
    validator = AIResponseValidator()
    result = validator.validate(raw_response)
    
    if not result.is_valid:
        validator.log_validation_failure(raw_response, result.error_message or "unknown")
    
    return result.data, result.is_valid


def is_fallback_response(data: Dict[str, Any]) -> bool:
    """Verifica se a resposta é um fallback (resultado de erro)."""
    return data.get("_is_fallback", False) is True
