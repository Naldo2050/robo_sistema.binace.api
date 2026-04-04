# Fase 5 — Pipeline de IA e ML

> Data: 2026-03-31 | Branch: audit/2026-03-31

---

## 5.1 Cadeia Completa de Payload → LLM → Resposta

### Dois Paths de Construção

| Path | Módulos | Usado em | Status |
|---|---|---|---|
| **Principal** | `build_compact_payload.py` → `market_orchestrator/ai/ai_runner.py` | Produção | Ativo |
| **Legado** | `ai_payload_builder.py` → `payload_compressor_v3.py` | Testes + fallback | Ativo |

### Verificações do Path Principal

| Verificação | Status | Evidência |
|---|---|---|
| Throttler integrado | ✅ Implementado | `SmartAIThrottler` em `market_orchestrator/ai/ai_runner.py` |
| Guardrail integrado | ✅ Implementado | `ensure_safe_llm_payload` em `ai_runner.py` |
| System prompt 8B vs 70B | ✅ Corrigido | `_MODELS_WITHOUT_JSON_MODE` set (sessão 2026-03-11) |
| Temperature llama | ✅ Corrigido | 0.3 (era 1.0) |
| Validação de resposta LLM | ✅ | `llm_response_validator.py` + `ai_response_validator.py` |
| Fallback quando rate-limited | ❓ A verificar | Throttler tem min interval mas sem cache de resposta anterior |

### Problema Crítico: `get_cross_asset_features` ausente

| Arquivo | Linha | Problema | Impacto |
|---|---|---|---|
| `tests/payload/conftest.py:28` | 28 | `monkeypatch.setattr(builder, "get_cross_asset_features", ...)` | **TODOS os 73 testes de payload falham** |
| `market_orchestrator/ai/ai_payload_builder.py` | 773 | `cross_asset_features = {...}` — dict literal, não função | Refatoração removeu o atributo sem atualizar testes |

O módulo `ai_payload_builder.py` não expõe `get_cross_asset_features` como atributo de nível de módulo. O `conftest.py` do diretório `tests/payload/` tenta monkeypatch este atributo e falha, causando erro em TODOS os 73 testes do diretório `tests/payload/`.

---

## 5.2 Machine Learning Pipeline

### FEATURE_MAP (ml/inference_engine.py)

| Verificação | Status | Detalhes |
|---|---|---|
| Bollinger Bands corrigido | ✅ v3 | Aliases diretos para `bb_upper/bb_lower/bb_width` |
| Fallback VAH/VAL removido | ✅ | Comentário documenta: "REMOVIDO fallback VAH/VAL (semanticamente errado)" |
| RSI anomalia corrigida | ✅ | Fallback para `multi_tf` antes de usar 50.0 |
| RSI fallback final | ✅ 50.0 | Neutral quando não há dados |
| BB quando histórico < 20 | ✅ Documentado | NaN intencional (XGBoost trata) |

### Status dos Problemas Conhecidos

| Issue | Status | Observação |
|---|---|---|
| RSI=100.0 do fallback ML | ✅ Corrigido | Fallback corrigido para 50.0 |
| Bollinger fallback errado | ✅ Corrigido | FEATURE_MAP v3 |
| System prompt 8B vs 70B | ✅ Corrigido | `_MODELS_WITHOUT_JSON_MODE` |
| macro_data hardcoded (VIX=12.5) | ✅ Corrigido | Usa VIX real do yFinance |
| Default model "qwen-plus" | ✅ Corrigido | → "llama-3.1-8b-instant" |

### Avaliação do Path AI

O path de AI principal (`market_orchestrator/ai/ai_runner.py`) está bem implementado com todos os fixes documentados aplicados. O path legado (`ai_runner/ai_runner.py`) é mais simples, sem guardrail/throttler — adequado para testes unitários mas não para produção.

### Aviso: payload_compressor_v3 com dados incompletos

Durante os testes de integração, observado warning:
```
COMPRESS_V3_VALIDATION warnings=['price.c MISSING (RECOVERED)', 'flow MISSING', 'whale MISSING', 'ob MISSING']
Original keys: ['symbol', 'epoch_ms', 'preco_fechamento', 'multi_tf']
```
Indica que alguns payloads chegam ao compressor sem seções obrigatórias (flow, whale, ob). O compressor recupera `price.c` mas não as demais seções.
