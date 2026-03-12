# market_orchestrator/ai/llm_response_validator.py
# -*- coding: utf-8 -*-

"""
Validador de resposta LLM com saída JSON estrita.

Este módulo garante que:
1. A resposta seja JSON válido
2. Contenha todos os campos obrigatórios
3. Os valores estejam nos ranges permitidos
4. Não haja texto extra fora do JSON
5. Logs sanitizados sem segredos
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("LLMValidator")

# ==============================================================================
# CONSTANTES E SCHEMA
# ==============================================================================

# Campos obrigatórios e seus valores permitidos
REQUIRED_FIELDS = {
    "sentiment": {"bullish", "bearish", "neutral"},
    "action": {"buy", "sell", "hold", "flat", "wait", "avoid"},
}

# Campos numéricos com range
NUMERIC_FIELDS = {
    "confidence": (0.0, 1.0),
}

# Mapeamento de actions inválidas para válidas
# O LLM às vezes usa termos financeiros padrão que não estão
# na lista de actions do sistema → mapeamos em vez de rejeitar
ACTION_ALIASES = {
    "short":     "sell",   # posição vendida → sell
    "long":      "buy",    # posição comprada → buy
    "close":     "flat",   # fechar posição → flat
    "exit":      "flat",   # sair da posição → flat
    "neutral":   "wait",   # neutro → wait
    "pass":      "wait",   # passar → wait
    "skip":      "avoid",  # pular → avoid
    "strong_buy": "buy",   # compra forte → buy
    "strong_sell": "sell", # venda forte → sell
}

# Campos opcionais de string
OPTIONAL_STRING_FIELDS = ["rationale", "entry_zone", "invalidation_zone", "region_type"]

# Fallback estruturado para respostas inválidas
FALLBACK_RESPONSE = {
    "sentiment": "neutral",
    "confidence": 0.0,
    "action": "wait",
    "rationale": "invalid_llm_output",
    "entry_zone": None,
    "invalidation_zone": None,
    "region_type": None,
    "_fallback": True,      # mantido para compatibilidade
    "_is_fallback": True,   # CORREÇÃO: ai_runner verifica esta chave
    "_is_valid": False,     # ADICIONADO: consistência com ai_runner
}


@dataclass
class ValidationResult:
    """Resultado da validação de resposta LLM."""
    valid: bool
    parsed: Dict[str, Any]
    error_reason: Optional[str] = None
    is_fallback: bool = False
    raw_preview: Optional[str] = None


def sanitize_for_log(text: str, max_len: int = 200) -> str:
    """
    Sanitiza texto para log removendo:
    - Chaves de API (gsk_*, sk-*, etc.)
    - Tokens longos
    - Caracteres problemáticos
    """
    if not isinstance(text, str):
        return ""
    
    # Trunca
    result = text[:max_len]
    
    # Remove padrões de segredos
    patterns_to_redact = [
        (r'gsk_[A-Za-z0-9_]+', 'gsk_***REDACTED***'),
        (r'sk-[A-Za-z0-9_\-]+', 'sk-***REDACTED***'),
        (r'Key:\s*[A-Za-z0-9_\-]{8,}', 'Key: ***REDACTED***'),
        (r'Chave:\s*[A-Za-z0-9_\-]{8,}', 'Chave: ***REDACTED***'),
        (r'token["\']?\s*[:=]\s*["\']?[A-Za-z0-9_\-]{10,}', 'token: ***REDACTED***'),
        (r'api_key["\']?\s*[:=]\s*["\']?[A-Za-z0-9_\-]{10,}', 'api_key: ***REDACTED***'),
    ]
    
    for pattern, replacement in patterns_to_redact:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    return result


def extract_json_from_response(text: str) -> Optional[str]:
    """
    Extrai JSON de uma resposta que pode conter:
    - JSON puro
    - JSON dentro de markdown (```json ... ```)
    - JSON precedido/sucedido por texto
    
    Retorna None se não encontrar JSON válido.
    """
    if not isinstance(text, str):
        return None
    
    text = text.strip()
    if not text:
        return None
    
    # Remove blocos de thinking/razonamento
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'```thinking.*?```', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'```thought.*?```', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Tenta parse direto primeiro
    text_stripped = text.strip()
    if text_stripped.startswith('{'):
        # Encontra o fechamento do JSON
        brace_count = 0
        in_string = False
        escape = False
        end_pos = -1
        
        for i, char in enumerate(text_stripped):
            if escape:
                escape = False
                continue
            if char == '\\' and in_string:
                escape = True
                continue
            if char == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i
                    break
        
        if end_pos > 0:
            json_str = text_stripped[:end_pos + 1]
            return json_str
    
    # Tenta encontrar JSON dentro de markdown
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        return json_match.group(1)
    
    # Tenta encontrar qualquer objeto JSON na resposta
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    
    return None


def validate_json_structure(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Valida se o JSON tem todos os campos obrigatórios e valores válidos.
    
    Returns:
        (valid, error_reason)
    """
    if not isinstance(data, dict):
        return False, "response_not_dict"
    
    # Verifica campos obrigatórios
    for field, allowed_values in REQUIRED_FIELDS.items():
        if field not in data:
            return False, f"missing_field:{field}"
        
        value = data.get(field)
        if value is None:
            return False, f"null_field:{field}"
        
        # Normaliza para lowercase
        str_value = str(value).lower().strip()
        
        # CORREÇÃO BUG1: tenta mapear aliases antes de rejeitar
        # Ex: "short" → "sell", "long" → "buy", "close" → "flat"
        if str_value not in allowed_values:
            if field == "action":
                mapped = ACTION_ALIASES.get(str_value)
                if mapped and mapped in allowed_values:
                    logger.info(
                        "ACTION_ALIAS_MAPPED original=%s mapped=%s",
                        str_value,
                        mapped
                    )
                    str_value = mapped
                else:
                    return False, f"invalid_value:{field}={str_value}"
            else:
                return False, f"invalid_value:{field}={str_value}"
        
        # Atualiza o valor normalizado
        data[field] = str_value
    
    # Verifica campos numéricos
    for field, (min_val, max_val) in NUMERIC_FIELDS.items():
        if field not in data:
            return False, f"missing_field:{field}"
        
        value = data.get(field)
        try:
            # Verifica se value é conversível para float antes de converter
            if value is None or not isinstance(value, (int, float, str)):
                return False, f"invalid_numeric:{field}={value}"
            num_value = float(value)

            # CORREÇÃO BUG5: clamp em vez de rejeitar
            # confidence=1.1 ou confidence=-0.05 são erros pequenos do LLM
            # Rejeitar completamente causa fallback desnecessário
            # Clamp para o range válido é mais robusto
            if num_value < min_val or num_value > max_val:
                clamped = max(min_val, min(max_val, num_value))
                logger.info(
                    "NUMERIC_CLAMPED field=%s original=%.4f clamped=%.4f",
                    field,
                    num_value,
                    clamped
                )
                num_value = clamped

            data[field] = round(num_value, 4)

        except (TypeError, ValueError):
            return False, f"invalid_numeric:{field}={value}"
    
    # CORREÇÃO BUG2: normaliza entry_zone e invalidation_zone
    # O LLM retorna formatos variados - normalizamos para lista [min, max]
    for zone_field in ("entry_zone", "invalidation_zone"):
        raw_zone = data.get(zone_field)
        
        if raw_zone is None or raw_zone == "null":
            # null é válido → mantém None
            data[zone_field] = None
            continue
        
        # Já é lista com 2 números → valida e mantém
        if isinstance(raw_zone, list):
            try:
                # Extrai números com conversão segura e tipagem explícita
                nums: list[float] = []
                for x in raw_zone:
                    if isinstance(x, (int, float)):
                        nums.append(float(x))
                    elif isinstance(x, str):
                        try:
                            nums.append(float(x))
                        except ValueError:
                            pass
                    # None e outros tipos são ignorados silenciosamente
                if len(nums) >= 2:
                    data[zone_field] = [
                        round(min(nums), 2),
                        round(max(nums), 2)
                    ]
                elif len(nums) == 1:
                    # Lista com 1 número → usa como ponto único
                    data[zone_field] = [
                        round(nums[0], 2),
                        round(nums[0], 2)
                    ]
                else:
                    data[zone_field] = None
            except (TypeError, ValueError):
                data[zone_field] = None
            continue
        
        # É número único → converte para lista
        if isinstance(raw_zone, (int, float)):
            data[zone_field] = [round(float(raw_zone), 2)] * 2
            continue
        
        # É string → tenta extrair números
        if isinstance(raw_zone, str):
            # Ignora strings literais do prompt como "preco|null"
            if "|" in raw_zone or raw_zone.lower() in (
                "null", "none", "n/a", "preco", "price"
            ):
                data[zone_field] = None
                continue
            
            # Tenta extrair números da string "68425-69153" ou "68425, 69153"
            nums_found = re.findall(r'\d+(?:\.\d+)?', raw_zone)
            if len(nums_found) >= 2:
                data[zone_field] = [
                    round(float(nums_found[0]), 2),
                    round(float(nums_found[-1]), 2)
                ]
            elif len(nums_found) == 1:
                data[zone_field] = [round(float(nums_found[0]), 2)] * 2
            else:
                data[zone_field] = None
            continue
        
        # Qualquer outro tipo → None
        data[zone_field] = None
    
    # Verifica rationale (deve ser string não vazia)
    rationale = data.get("rationale", "")
    if rationale is None or (isinstance(rationale, str) and not rationale.strip()):
        # rationale vazio é aceito, mas logamos warning
        logger.debug("rationale vazio ou ausente")
    
    return True, None


def is_truncated_json(text: str) -> bool:
    """
    Detecta se o JSON está truncado.

    CORREÇÃO BUG3:
        Antes: "not text.endswith('}')" causava falso positivo
        quando JSON válido tinha newline ou espaço no final.
        Ex: '{"action":"buy"}\n' → endswith('}') = False → truncado!

        Agora: strip() antes de verificar + verifica balanço de chaves
        sem depender do último caractere.
    """
    if not isinstance(text, str):
        return False

    # Remove whitespace das bordas para comparação
    text_stripped = text.strip()

    if not text_stripped:
        return False

    # ── Indicadores claros de truncamento ───────────────────────────
    # Esses casos são inequívocos: JSON cortado no meio
    hard_truncation = (
        text_stripped.endswith(',')   # vírgula no final = campo cortado
        or text_stripped.endswith(':') # dois pontos no final = valor ausente
        or text_stripped.endswith('{"') # abre chave + abre string = truncado
    )
    if hard_truncation:
        return True

    # ── Verifica balanço de estruturas ──────────────────────────────
    # Conta chaves/colchetes não balanceados
    open_braces = text_stripped.count('{') - text_stripped.count('}')
    open_brackets = text_stripped.count('[') - text_stripped.count(']')

    if open_braces > 0 or open_brackets > 0:
        return True

    # ── Verifica aspas (número ímpar = string aberta) ───────────────
    # Conta aspas fora de strings escapadas
    quote_count = 0
    in_escape = False
    for char in text_stripped:
        if in_escape:
            in_escape = False
            continue
        if char == '\\':
            in_escape = True
            continue
        if char == '"':
            quote_count += 1

    if quote_count % 2 != 0:
        return True

    # ── Verifica se contém pelo menos um JSON completo ───────────────
    # Tenta fazer parse para confirmar que é JSON válido
    # Se falhar, pode estar truncado
    if text_stripped.startswith('{'):
        try:
            import json as _json
            _json.loads(text_stripped)
            return False  # Parse OK → não truncado
        except _json.JSONDecodeError as e:
            # Erro de parse pode indicar truncamento
            # Mas apenas se o erro for no final
            error_pos = getattr(e, 'pos', 0)
            if error_pos >= len(text_stripped) - 5:
                return True
            # Erro no meio = formato inválido mas não necessariamente truncado

    return False


def has_extra_text(text: str) -> bool:
    """
    Verifica se há texto extra fora do JSON.
    """
    if not isinstance(text, str):
        return False
    
    text = text.strip()
    
    # Extrai JSON
    json_str = extract_json_from_response(text)
    if json_str is None:
        return True  # Sem JSON = texto extra (ou só texto)
    
    # Remove o JSON encontrado e verifica sobra
    remaining = text.replace(json_str, '', 1).strip()
    
    # Remove markdown code blocks
    remaining = re.sub(r'```\w*\n?', '', remaining).strip()
    
    # Se sobrou algo significativo (> 10 chars não whitespace)
    if len(remaining) > 10:
        return True
    
    return False


def validate_llm_response(
    response: str,
    strict: bool = False,  # CORREÇÃO BUG4: False por padrão
    # Antes: True → rejeitava respostas válidas com texto extra após JSON
    # Agora: False → extrai JSON e ignora texto extra
    # Use strict=True apenas em testes unitários
    log_errors: bool = True,
) -> ValidationResult:
    """
    Valida resposta do LLM e retorna resultado estruturado.
    
    Args:
        response: Texto bruto da resposta do LLM
        strict: Se True, rejeita texto extra fora do JSON
        log_errors: Se True, loga erros de validação
    
    Returns:
        ValidationResult com parsed JSON ou fallback
    """
    if not isinstance(response, str):
        if log_errors:
            logger.warning("ai_response_invalid: response_not_string")
        return ValidationResult(
            valid=False,
            parsed=FALLBACK_RESPONSE.copy(),
            error_reason="response_not_string",
            is_fallback=True,
            raw_preview=None,
        )
    
    response = response.strip()
    
    if not response:
        if log_errors:
            logger.warning("ai_response_invalid: empty_response")
        return ValidationResult(
            valid=False,
            parsed=FALLBACK_RESPONSE.copy(),
            error_reason="empty_response",
            is_fallback=True,
            raw_preview=None,
        )
    
    # Detecta truncamento
    if is_truncated_json(response):
        preview = sanitize_for_log(response, 150)
        if log_errors:
            logger.warning(f"ai_response_invalid: truncated_json | preview={preview}")
        return ValidationResult(
            valid=False,
            parsed=FALLBACK_RESPONSE.copy(),
            error_reason="truncated_json",
            is_fallback=True,
            raw_preview=preview,
        )
    
    # Detecta texto livre (padrões comuns de LLM "thinking")
    free_text_patterns = [
        r'^okay[,\s]',
        r'^let\s+me\s+',
        r'^first[,\s]',
        r'^looking\s+at',
        r'^analyzing',
        r'^here\s+is',
        r'^i\s+will',
        r'^the\s+analysis',
    ]
    
    response_lower = response.lower()
    for pattern in free_text_patterns:
        if re.match(pattern, response_lower):
            preview = sanitize_for_log(response, 150)
            if log_errors:
                logger.warning(f"ai_response_invalid: free_text_mode | preview={preview}")
            return ValidationResult(
                valid=False,
                parsed=FALLBACK_RESPONSE.copy(),
                error_reason="free_text_mode",
                is_fallback=True,
                raw_preview=preview,
            )
    
    # Extrai JSON
    json_str = extract_json_from_response(response)
    if json_str is None:
        preview = sanitize_for_log(response, 150)
        if log_errors:
            logger.warning(f"ai_response_invalid: no_json_found | preview={preview}")
        return ValidationResult(
            valid=False,
            parsed=FALLBACK_RESPONSE.copy(),
            error_reason="no_json_found",
            is_fallback=True,
            raw_preview=preview,
        )
    
    # Verifica texto extra (se strict mode)
    if strict and has_extra_text(response):
        preview = sanitize_for_log(response, 150)
        if log_errors:
            logger.warning(f"ai_response_invalid: extra_text_outside_json | preview={preview}")
        return ValidationResult(
            valid=False,
            parsed=FALLBACK_RESPONSE.copy(),
            error_reason="extra_text_outside_json",
            is_fallback=True,
            raw_preview=preview,
        )
    
    # Parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        preview = sanitize_for_log(response, 150)
        if log_errors:
            logger.warning(f"ai_response_invalid: json_parse_error | error={str(e)[:50]} | preview={preview}")
        return ValidationResult(
            valid=False,
            parsed=FALLBACK_RESPONSE.copy(),
            error_reason=f"json_parse_error:{str(e)[:50]}",
            is_fallback=True,
            raw_preview=preview,
        )
    
    # Valida estrutura
    valid, error_reason = validate_json_structure(data)
    
    if not valid:
        preview = sanitize_for_log(response, 150)
        if log_errors:
            logger.warning(f"ai_response_invalid: {error_reason} | preview={preview}")
        return ValidationResult(
            valid=False,
            parsed=FALLBACK_RESPONSE.copy(),
            error_reason=error_reason,
            is_fallback=True,
            raw_preview=preview,
        )
    
    # Sucesso - normaliza campos opcionais
    result = dict(data)
    
    for field in OPTIONAL_STRING_FIELDS:
        if field not in result:
            result[field] = None
    
    # Log de sucesso (sem dados sensíveis)
    logger.info(f"ai_response_valid: action={result.get('action')} confidence={result.get('confidence')}")
    
    return ValidationResult(
        valid=True,
        parsed=result,
        error_reason=None,
        is_fallback=False,
        raw_preview=None,
    )


def validate_and_parse_llm_response(
    response: str,
    fallback_on_invalid: bool = True,
    log_errors: bool = True,
) -> Tuple[Dict[str, Any], bool, Optional[str]]:
    """
    Função de conveniência que retorna (parsed_data, is_valid, error_reason).
    
    Mantém compatibilidade com código existente que espera tuplas.
    """
    result = validate_llm_response(response, strict=True, log_errors=log_errors)
    
    if result.valid or fallback_on_invalid:
        return result.parsed, result.valid, result.error_reason
    else:
        return {}, False, result.error_reason


# ==============================================================================
# LOGGING SEGURO
# ==============================================================================

class SecretRedactingLogger:
    """
    Wrapper de logger que automaticamente redaz segredos de mensagens.
    """
    
    # Padrões de segredos para redacionar
    SECRET_PATTERNS = [
        (r'gsk_[A-Za-z0-9_]+', 'gsk_***'),
        (r'sk-[A-Za-z0-9_\-]+', 'sk-***'),
        (r'Key:\s*[A-Za-z0-9_\-]{8,}', 'Key: ***'),
        (r'Chave:\s*[A-Za-z0-9_\-]{8,}', 'Chave: ***'),
        (r'api[_-]?key["\']?\s*[:=]\s*["\']?[A-Za-z0-9_\-]{8,}', 'api_key: ***'),
        (r'token["\']?\s*[:=]\s*["\']?[A-Za-z0-9_\-]{8,}', 'token: ***'),
        (r'Bearer\s+[A-Za-z0-9_\-\.]+', 'Bearer ***'),
        (r'["\']?[A-Za-z0-9_\-\.\+]+@["\']?\s*[:=]', 'email: ***'),
    ]
    
    def __init__(self, logger_instance: logging.Logger):
        self._logger = logger_instance
    
    def _redact(self, message: str) -> str:
        """Remove segredos da mensagem."""
        result = str(message)
        for pattern, replacement in self.SECRET_PATTERNS:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return result
    
    def info(self, msg: str, *args, **kwargs):
        self._logger.info(self._redact(msg), *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        self._logger.warning(self._redact(msg), *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        self._logger.error(self._redact(msg), *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs):
        self._logger.debug(self._redact(msg), *args, **kwargs)


def create_safe_logger(name: str) -> SecretRedactingLogger:
    """Cria um logger com redação automática de segredos."""
    return SecretRedactingLogger(logging.getLogger(name))