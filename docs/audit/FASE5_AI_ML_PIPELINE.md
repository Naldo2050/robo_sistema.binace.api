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

---

## Backlog: institutional/cvd.py não integrado ao caminho live
- Instanciado em `event_bridge.py:61`, nunca recebe update/process_trade
- Possui testes unitários (`tests/unit/test_institutional_cvd.py`) mas não afeta produção
- Decisão pendente: integrar ao event_bridge OU remover (dead weight)
- Não bloqueante — CVD ativo real está em `flow_analyzer/core.py` (validado)

---

## Auditoria 2026-08-08: falso sinal de divergência CVD após reset do FlowAnalyzer

### Problema
O reset periódico do CVD (4h) zerava o `cvd` acumulado, mas o fallback de
`_build_cvd_divergence` comparava esse CVD recém-zerado com a tendência de 1h
inteira. Resultado: logo após o reset, vendas leves geravam "bearish_div"
falso com `src=inferred` (confirmado no cenário FAIL do script de auditoria).

### Correção (3 camadas)

| Camada | O que faz | Onde |
|---|---|---|
| 1 | Supressão de `cvd_div` enquanto o CVD ainda aquece após o reset (`CVD_DIV_WARMUP_SECONDS`, default 300s) | `market_orchestrator/ai/payload_builder_compact.py` |
| 2 | Comparação do CVD com a **variação de preço no mesmo período** desde o reset (`price_at_reset` capturado no reset), não com o trend_1h inteiro | `flow_analyzer/core.py` + `payload_builder_compact.py` |
| 2.3 | Período pós-reset < `CVD_DIV_MIN_PERIOD_SECONDS` (10s) também suprime (referência de preço não confiável) | `payload_builder_compact.py` |

### Mudanças
- `flow_analyzer/core.py`: `_price_at_reset` capturado em `_reset_metrics()`; exposto na raiz de `get_flow_metrics()` como `last_reset_ms` e `price_at_reset`
- `common/payload_optimizer_config.py`: `last_reset_ms`/`price_at_reset` movidos para `FIELDS_TO_KEEP_INTERNAL` (usados no cálculo interno antes da compressão; não vão ao payload final da IA)
- `config/settings.py`: `CVD_DIV_WARMUP_SECONDS = 300`, `CVD_DIV_MIN_PERIOD_SECONDS = 10`

### Validação
- `scripts/diagnostics/audit_cvd_reset_divergence_test.py`: **PASS** (cenário que antes dava FAIL agora é suprimido; divergência real no mesmo período continua detectada como `bearish_div`)
- `tests/payload/`: 265 passed
- `tests/unit/test_institutional_cvd.py`: 16 passed
- `tests/unit/test_flow_analyzer.py` + `tests/integration/test_optimization.py`: 40 passed
- Consumidores de `cvd_div` (payload builder, `ai_payload_types`, `analyzer_qwen`) intactos — contrato `{det, type}` preservado
