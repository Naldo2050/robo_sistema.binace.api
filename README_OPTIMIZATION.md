# Guia Completo de Otimização de Payload para IA

## Objetivo

Reduzir o tamanho do payload JSON enviado para a API de IA (ex.: ~8.000 → ~2.700 tokens / ~66%) **sem perder qualidade** dos dados necessários para decisão.

Este pacote adiciona dois níveis de otimização:

- **Evento otimizado**: remove/simplifica campos do evento inteiro (boa para armazenar/logar/transportar).
- **Payload minimalista**: constrói um `ai_payload` pequeno e focado (bom para enviar ao LLM).

## Arquivos

- `payload_optimizer_config.py`: regras do que remover/manter/arredondar/filtrar.
- `optimize_ai_payload.py`: funções de otimização + builder do payload.
- `test_optimization.py`: valida redução, preservação e estrutura.
- `tests/fixtures/sample_analysis_trigger.json`: fixture de exemplo usada nos testes.

## Como Usar

### 1) Otimizar um evento (ANALYSIS_TRIGGER, etc.)

```python
from optimize_ai_payload import optimize_event_for_ai

optimized_event = optimize_event_for_ai(original_event)
```

O otimizador aplica:

1. Remove campos de debug/observabilidade e flags internas
2. Compacta `historical_vp` (mantém `poc/vah/val` + `hvns_nearby/lvns_nearby`)
3. Remove duplicações (`enriched_snapshot`, `fluxo_continuo` quando duplicado)
4. Arredonda floats (2–4 casas conforme tipo)
5. Limita listas grandes (ex.: clusters)
6. Valida campos obrigatórios (configurável)

### 2) Construir payload minimalista para IA

```python
from optimize_ai_payload import build_optimized_ai_payload

ai_payload = build_optimized_ai_payload(original_event)
```

Estrutura resultante:

- `price_context`: preço atual, OHLC, posição do close, VP diário
- `flow_context`: net flow, CVD, imbalance, agressão
- `orderbook_context`: profundidades, imbalance, pressure, depth_metrics
- `macro_context`: sessão, fase, regime

## O que foi otimizado

### Removido (impacto ~0% na IA)

- `observability`, `processing_times_ms`, `metadata` e flags internas
- timestamps redundantes (`timestamp_ny`, `timestamp_sp`, `timestamp`)
- estruturas duplicadas (ex.: `enriched_snapshot`)

### Simplificado

- `historical_vp`: arrays grandes → **apenas níveis próximos** (`hvns_nearby/lvns_nearby`)
- números: precisão reduzida por tipo de campo (configurável)
- clusters: limita para os mais relevantes (por volume)

## Configuração

Edite `payload_optimizer_config.py`:

- `FIELDS_TO_REMOVE`: chaves removidas em qualquer nível (recursivo).
- `ROUNDING_CONFIG`: regras de arredondamento por tipo de campo.
- `FILTER_CONFIG`: filtros e limites (VP nearby, clusters).
- `SAFETY_CONFIG`: campos obrigatórios, abortar se faltar, backup/log.

## Testes

Rodar apenas os testes desta feature:

```bash
pytest --no-cov -q test_optimization.py
```

## Fase 2 (Integração no envio para IA)

Checklist sugerido:

1. Localizar onde o evento é convertido em `ai_payload` (buscar por `AI_ANALYSIS`, `ai_payload`, `call_*ai*`).
2. Importar:

```python
from optimize_ai_payload import build_optimized_ai_payload
```

3. Substituir o payload enviado:

- Antes: `ai_payload = event["ai_payload"]` (ou equivalente)
- Depois: `ai_payload = build_optimized_ai_payload(event)`

4. Em modo debug, validar log:

- `📊 Otimização completa: X → Y bytes (-Z%)`
