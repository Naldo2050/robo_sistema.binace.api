# RELATÓRIO DE VALIDAÇÃO OPERACIONAL E SHADOW OBSERVATION — FASE P1.1B
**Projeto:** Robô de Trading Binance API  
**Data:** 2026-09-01  
**Status:** CONCLUÍDO COM SUCESSO (VALIDAÇÃO E AUDITORIA COMPLETA)  
**Base Normativa:** `docs/audit/BINANCE_POSITIONING_DESIGN_2026.md` & `docs/audit/INSTITUTIONAL_DATA_CONTRACTS_2026.md`  

---

## 1. ESCOPO E OBJETIVO DA FASE P1.1B

A **Fase P1.1B** teve como objetivo auditar rigorosamente a semântica dos dados, representação numérica, comportamento de cache e frescor, impacto no critical path, rate limits, garantias anti-lookahead, integridade de proveniência ponta-a-ponta e criar a infraestrutura offline para coleta e avaliação de hipóteses preditivas de posicionamento.

### 🛡️ Restrição Arquitetural Primária (Strict Context-Only)
- **Nenhum sinal direto de trade (BUY/SELL) é gerado a partir de posicionamento.**
- **Nenhum parâmetro de trade execution, position sizing, veto algorítmico ou risk management consome dados de posicionamento.**
- O dado atua exclusivamente como **contexto macro e estrutural** para o LLM interpretar crowding e risco de squeeze.

---

## 2. AUDITORIA SEMÂNTICA DOS ENDPOINTS DA BINANCE

A revalidação técnica dos 4 endpoints públicos da Binance USD-M Futures confirmou seus esquemas exatos:

### 2.1. Global Long/Short Account Ratio
- **Endpoint:** `GET https://fapi.binance.com/futures/data/globalLongShortAccountRatio`
- **Parâmetros:** `symbol=BTCUSDT`, `period=5m`, `limit=60`
- **Frequência de Atualização Binance:** A cada 5 minutos.
- **Campos Retornados:**
  - `symbol` (string): Identificador do par (`"BTCUSDT"`).
  - `longAccount` (string float): Fração de contas net long em relação ao total ($0.0$ a $1.0$).
  - `shortAccount` (string float): Fração de contas net short em relação ao total ($0.0$ a $1.0$).
  - `longShortRatio` (string float): Razão direta $\frac{\text{longAccount}}{\text{shortAccount}}$.
  - `timestamp` (integer ms): Carimbo de tempo da barra agregada.
- **Definição Oficial da Binance:** "Proporção de contas compradas e vendidas líquidas em relação ao total de contas de todos os traders no par."
- **Unidade:** Adimensional ($>0.0$).

### 2.2. Top Trader Long/Short Account Ratio
- **Endpoint:** `GET https://fapi.binance.com/futures/data/topLongShortAccountRatio`
- **Parâmetros:** `symbol=BTCUSDT`, `period=5m`, `limit=60`
- **Campos Retornados:** `symbol`, `longAccount`, `shortAccount`, `longShortRatio`, `timestamp`.
- **Definição Oficial da Binance:** "Proporção de contas compradas e vendidas líquidas em relação ao total de contas dos **top 20% usuários com maior saldo de margem**." (1 voto por conta).
- **Unidade:** Adimensional ($>0.0$).

### 2.3. Top Trader Long/Short Position Ratio
- **Endpoint:** `GET https://fapi.binance.com/futures/data/topLongShortPositionRatio`
- **Parâmetros:** `symbol=BTCUSDT`, `period=5m`, `limit=60`
- **Campos Retornados:** `symbol`, `longPosition` (ou `longAccount`), `shortPosition` (ou `shortAccount`), `longShortRatio`, `timestamp`.
- **Definição Oficial da Binance:** "Proporção do volume nocional total de posições compradas vs vendidas detido pelos **top 20% usuários com maior saldo de margem**." (Ponderado pelo tamanho financeiro da posição).
- **Unidade:** Adimensional ($>0.0$).

### 2.4. Open Interest Statistics History
- **Endpoint:** `GET https://fapi.binance.com/futures/data/openInterestHist`
- **Parâmetros:** `symbol=BTCUSDT`, `period=5m`, `limit=60`
- **Campos Retornados:** `symbol`, `sumOpenInterest` (BTC), `sumOpenInterestValue` (USDT), `timestamp`.
- **Definição Oficial da Binance:** "Contratos em aberto totais e valor nocional acumulado em USD."
- **Unidade:** BTC (contratos) e USD (nocional).

> [!IMPORTANT]
> **CORREÇÃO DE TERMINOLOGIA NORMATIVA:**
> É incorreto afirmar que os endpoints de "Top Trader" representam "todo o capital institucional global". A definição exata da Binance refere-se estritamente aos **top 20% usuários com maior saldo de margem na Binance USD-M Futures**. Toda a documentação e legendas foram alinhadas a essa definição estrita.

---

## 3. REPRESENTAÇÃO NUMÉRICA E FORMATAÇÃO CANÔNICA

Na auditoria P1.1B, eliminou-se a representação em string formatada (`"+0.2%"`) dentro do JSON de máquina (`pos`), padronizando todos os campos numéricos em **frações decimais puras**:

- `ga`: float canônico arredondado em 2 casas decimais (ex: `1.28`).
- `ta`: float canônico arredondado em 2 casas decimais (ex: `1.39`).
- `tp`: float canônico arredondado em 2 casas decimais (ex: `2.07`).
- `od1`: float canônico da variação percentual relativa de 1h em fração decimal (ex: `0.0017` = $+0.17\%$).
- `od4`: float canônico da variação percentual relativa de 4h em fração decimal (ex: `-0.0035` = $-0.35\%$).
- `rg`: string com o nome do regime heurístico (`"TOP_LONG_DIVERGENCE"`).

### Tratamento Defensivo de Edge Cases:
- `0.0` exato é preservado (não descartado como ausente).
- `None`, `NaN`, `+Inf`, `-Inf` e tipos `bool` (`True`/`False`) são rejeitados de forma segura pela função `_safe_val`, prevenindo quebras no serializador JSON RFC 8259.

---

## 4. FÓRMULAS MATEMÁTICAS E AUDITORIA ANTI-LOOKAHEAD

### 4.1. Fórmulas de Divergência
$$\text{top\_account\_vs\_global} = \text{top\_account\_ratio} - \text{global\_account\_ratio}$$
$$\text{top\_position\_vs\_global} = \text{top\_position\_ratio} - \text{global\_account\_ratio}$$

### 4.2. Fórmulas de Delta de Open Interest
Para a barra $T$ atual (último elemento `[-1]` da série histórica de 5m retornada pela Binance):
- **Delta 1h (12 barras de 5m atrás, índice `[-13]`):**
  $$\text{oi\_delta\_1h} = \frac{\text{sumOpenInterest}(T) - \text{sumOpenInterest}(T - 1\text{h})}{\text{sumOpenInterest}(T - 1\text{h})}$$
- **Delta 4h (48 barras de 5m atrás, índice `[-49]`):**
  $$\text{oi\_delta\_4h} = \frac{\text{sumOpenInterest}(T) - \text{sumOpenInterest}(T - 4\text{h})}{\text{sumOpenInterest}(T - 4\text{h})}$$

### 4.3. Prova Formal Anti-Lookahead
- Ambas as baselines ($T-1\text{h}$ e $T-4\text{h}$) pertencem estritamente ao passado ($t_{\text{baseline}} < T_{\text{decision}}$).
- Não há qualquer ponto futuro utilizado na reconstrução da feature.
- Em caso de histórico insuficiente no arranque (warm-up $<13$ barras para 1h ou $<49$ barras para 4h), o retorno é determinístico `None`, eliminando interpolações espúrias.

---

## 5. CACHE, FRESCOR E DISTINÇÃO NORMATIVA

- **`CACHE_TTL` = 300 segundos (5 minutos):** Alinhado à cadência em que a Binance publica novas barras de 5m nos endpoints `/futures/data/`. Evita requisições HTTP redundantes no mesmo ciclo de 5m.
- **`SOURCE_FRESHNESS_TTL` = 900 segundos (15 minutos):** Limite máximo para a idade do carimbo de tempo da exchange (`now - source_timestamp / 1000.0 > 900.0s`).
- **Garantia de Independência:** Se a API da Binance responder HTTP 200 contendo dados defasados por problemas internos da exchange, `is_stale` é marcado como `True`, e o analisador `CryptoCOT` classifica o dado como `UNKNOWN`, rejeitando a exposição de dados congelados como se fossem frescos.

---

## 6. MEDIÇÃO EXPERIMENTAL DO CRITICAL PATH

Foi executado benchmark assíncrono real via `scripts/diagnostics/measure_critical_path_positioning.py`:

```
======================================================================
1. LATÊNCIA INDIVIDUAL DOS ENDPOINTS BINANCE (amostras = 5)
======================================================================
  - global_account_ratio     : min=279.7ms | p50=281.5ms | max=318.7ms
  - top_account_ratio        : min=275.5ms | p50=279.6ms | max=288.1ms
  - top_position_ratio       : min=278.1ms | p50=283.2ms | max=283.7ms
  - open_interest_hist       : min=276.7ms | p50=278.9ms | max=280.2ms

======================================================================
2. BINANCE POSITIONING FETCHER (PARALELO 4 ENDPOINTS)
======================================================================
  - Cold Cache Fetch (4 requests paralelos): 447.70ms
  - Warm Cache Fetch (p50): 1.50 µs | max: 8.80 µs

======================================================================
3. TEMPO DE CADA CORROTINA NO GATHER & CRITICAL PATH
======================================================================
  - market_env          :  4693.9 ms  <-- CRITICAL PATH BOTTLENECK (yFinance/Macro)
  - derivatives         :  2819.5 ms
  - sentiment           :  1689.0 ms
  - mtf                 :  1224.0 ms
  - pivots              :   943.2 ms
  - intermarket         :   651.5 ms
  - positioning         :   322.9 ms
  - external            :     2.5 ms

  Total asyncio.gather time: 4697.9 ms
```

### Conclusão de Latência:
Como `positioning` leva **322.9ms** e é executado em paralelo dentro do mesmo `asyncio.gather` limitado pelo gargalo de **4693.9ms** de `market_env`, ele completa **4.3 segundos antes** do fim do gather. A sua contribuição marginal para o tempo total de coleta de contexto é de **0.00ms**.

---

## 7. RATE LIMIT BUDGET E RESILIÊNCIA

- **Limite da Binance Futures USD-M:** 1200 unidades de peso por minuto.
- **Consumo de Posicionamento:** 4 requisições (peso 1 cada) a cada 5 minutos = **0.8 req/min**.
- **Consumo do Budget:** **0.067%** da cota permitida.
- **Comportamento sob HTTP 429:** O fetcher aborta sem retries, registra aviso em log, marca `is_available=False` e não trava o orquestrador nem gera tempestade de retries.

---

## 8. CLASSIFICAÇÃO NORMATIVA DOS THRESHOLDS DE REGIME

| Regra / Threshold | Valor | Classificação | Justificativa |
|---|---|---|---|
| `CROWDED_LONG_RATIO` | `2.0` | **HEURISTIC** | $>66.7\%$ de contas líquidas compradas. |
| `CROWDED_SHORT_RATIO` | `0.5` | **HEURISTIC** | $>66.7\%$ de contas líquidas vendidas. |
| `DIVERGENCE_THRESHOLD` | `0.40` | **HEURISTIC** | Desvio expressivo entre Top Trader Position e Global Account Ratio. |
| `OI_EXPANSION_1H` | `0.05` (+5%) | **HEURISTIC** | Variação de volume nocional em 60 min. |
| `OI_EXPANSION_4H` | `0.10` (+10%) | **HEURISTIC** | Variação de volume nocional em 4 horas. |
| `SQUEEZE_FUNDING_THRESHOLD` | `0.0003` (3 bps) | **DOCUMENTED_EXTERNAL** | Padrão da literatura financeira de derivativos para prêmio de taxa de financiamento desbalanceado. |

> [!NOTE]
> Nenhum dos thresholds acima foi ajustado ou sobreajustado no dataset histórico. São tratados explicitamente como **contexto heurístico**.

---

## 9. SHADOW DATASET E METODOLOGIA OFICIAL DE AVALIAÇÃO OFFLINE

Foi criado o módulo [`scripts/analytics/positioning_shadow_collector.py`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/scripts/analytics/positioning_shadow_collector.py) e a ferramenta [`scripts/analytics/positioning_evaluator.py`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/scripts/analytics/positioning_evaluator.py).

### 9.1. Esquema da Tabela SQLite `positioning_shadow_dataset`
Armazena registros brutos numéricos com carimbos temporais, proveniência e métricas:
- `timestamp_ms`, `symbol`, `price`
- `global_account_ratio`, `top_account_ratio`, `top_position_ratio`
- `global_long_pct`, `global_short_pct`, `top_long_account_pct`, `top_long_position_pct`
- `open_interest`, `open_interest_usd`, `oi_delta_1h`, `oi_delta_4h`, `funding_rate`
- `top_account_vs_global`, `top_position_vs_global`, `positioning_regime`
- `source_timestamp_ms`, `age_seconds`, `is_stale`, `cache_hit`, `reasons_json`

### 9.2. Metodologia de Divisão Temporal (Anti-Overfitting)
A ferramenta implementa partição cronológica estrita:
1. **TRAIN / DISCOVERY (60%):** Identificação exploratória de hipóteses informacionais.
2. **VALIDATION (20%):** Calibração de parâmetros e thresholds.
3. **OUT-OF-SAMPLE (20%):** Teste cego de valor preditivo incremental.

### 9.3. Protocolo de Teste de Valor Incremental
Futuras avaliações compararão:
- **BASELINE:** `Regime + CVD + Flow + Orderbook + Funding`
- **BASELINE + POSITIONING:** `Baseline + {ga, ta, tp, od1, od4, rg}`
- **Métricas:** Directional Accuracy, Conditional Forward Return (5m, 15m, 1h), MFE, MAE, AUC, Teste $t$ de Student e Intervalo de Confiança de 95%.

---

## 10. INTEGRIDADE DE PROVENIÊNCIA E SUÍTE DE TESTES

A suíte completa [`tests/payload/test_positioning_provenance_p1_1b.py`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/tests/payload/test_positioning_provenance_p1_1b.py) e todas as suítes de regressão foram executadas com **100% de sucesso**:

```
Ran 57 tests in 0.239s:
  - tests/unit/test_binance_positioning_p1_1.py: 14/14 PASS
  - tests/payload/test_funding_rate_pipeline_p0.py: 10/10 PASS
  - tests/payload/test_positioning_provenance_p1_1b.py: 3/3 PASS
  - tests/unit/test_ai_response_validator.py: 30/30 PASS

Resultado: 57/57 PASS (Zero Regressões)
```

### Verificações Específicas Concluídas:
- **Funding Rate Não-Duplicado:** `price.fr` preservado; `pos` não contém campo duplicado de funding.
- **Escala de Funding Preservada:** Fração decimal canônica direta (`0.0001` = 1 bp), sem divisões ou multiplicações espúrias por 100.
- **Payload LLM:** Seção `pos` compactada em ~23 tokens, explicada no `SYSTEM_PROMPT` como contexto estrutural e compatível com o parser RFC 8259 estrito.

---

## 11. BUGS ENCONTRADOS E CORRIGIDOS NA FASE P1.1B

1. **Bug de Representação Numérica em `_build_positioning`:** `od1` e `od4` estavam formatados como strings com símbolo de porcentagem (`"+0.2%"`). Corrigido para float canônico (`0.0020`), preservando pureza numérica para o parser.
2. **Defesa contra Valores Não-Finitos:** Adicionada validação de `math.isfinite()` e rejeição de booleanos para `ga`, `ta`, `tp`, evitando contaminação de `NaN` ou `Inf` no payload.
3. **Assertiva do Compressor Groq:** Ajustada a assertiva do teste de proveniência para a chave compactada `groq_summary["p"]["fr"]`.

---

## 12. PONTO DE CONTROLE E PARADA

A **Fase P1.1B** está concluída com êxito.

> [!IMPORTANT]
> **ESTADO ATUAL DO SISTEMA:**
> - Positioning e Crypto COT estão 100% validados em runtime, documentados, testados ponta-a-ponta e com coleta shadow em banco de dados ativa.
> - O sistema permanece em modo **STRICT CONTEXT-ONLY** com isolamento total de trade execution.
> - O desenvolvimento está **pausado**, aguardando autorização formal antes de iniciar qualquer fase subsequente (P1.2 / P1.3).
