# AUDITORIA FORENSE PÓS-FIX (EXECUÇÃO REAL/SHADOW 1H)
**Data da Auditoria:** 2026-08-31T21:18:00-03:00 (2026-09-01T00:18:00Z)  
**Modo de Execução:** READ-ONLY (Nenhuma alteração em código ou banco de dados)  
**Status Final:** **READY**

---

## 1. DELIMITAÇÃO DO RUN

A execução foi isolada com base no ciclo de vida dos processos e no banco SQLite:

- **Início do Run (`run_start_utc`):** `2026-08-31T23:02:02Z` (Local: `20:02:02`)
- **Fim do Run (`run_end_utc`):** `2026-09-01T00:06:09Z` (Local: `21:06:09`)
- **Duração Total (`duration`):** 3.847,0 segundos (64,12 minutos / ~1h04m)
- **Primeiro Evento:** Tipo `Absorção` (Absorção de Compra), `window_id=1788217379962`, `timestamp_utc=2026-08-31T23:03:00.000Z` (`created_at=2026-08-31 23:03:08`)
- **Último Evento:** Tipo `Alerta` (Fim/Shutdown), `timestamp_utc=2026-09-01T00:06:05.603Z` (`created_at=2026-09-01 00:06:09`)
- **Quantidade Total de Eventos:** 86 eventos no fluxo (`eventos_fluxo.jsonl`) e 86 eventos no SQLite (`trading_bot.db -> events`)
- **Quantidade de Sinais (`is_signal=True`):** 4 sinais (`Absorção` x2, `Exaustão` x2)
- **Quantidade de `ANALYSIS_TRIGGER`:** 61 triggers de 1 minuto
- **Quantidade de Análises IA (`AI_ANALYSIS`):** 11 chamadas

---

## 2. OUTCOMETRACKER — BOUNDARY P1

Auditoria individual de cada sinal e horizonte registrado na tabela `signal_outcomes` durante o run:

| ID | Sinal | Lado | Preço Entrada | Horizonte | Target Epoch UTC | Preço Boundary | Drift Real | Status |
|---|---|---|---|---|---|---|---|---|
| **1** | Absorção de Compra | SHORT | 78.422,11 | 5m | 2026-08-31T23:08:00Z | 78.480,00 (+0.0738%) | 0 ms | **PASS** |
| **1** | Absorção de Compra | SHORT | 78.422,11 | 15m | 2026-08-31T23:18:00Z | 78.488,20 (+0.0843%) | 0 ms | **PASS** |
| **1** | Absorção de Compra | SHORT | 78.422,11 | 30m | 2026-08-31T23:33:00Z | 78.580,60 (+0.2021%) | 0 ms | **PASS** |
| **1** | Absorção de Compra | SHORT | 78.422,11 | 60m | 2026-09-01T00:03:00Z | 78.608,00 (+0.2370%) | 0 ms | **PASS** |
| **2** | Absorção de Compra | SHORT | 78.448,04 | 5m | 2026-08-31T23:18:00Z | 78.488,20 (+0.0512%) | 0 ms | **PASS** |
| **2** | Absorção de Compra | SHORT | 78.448,04 | 15m | 2026-08-31T23:28:00Z | 78.575,80 (+0.1629%) | 0 ms | **PASS** |
| **2** | Absorção de Compra | SHORT | 78.448,04 | 30m | 2026-08-31T23:43:00Z | 78.570,00 (+0.1555%) | 0 ms | **PASS** |
| **2** | Absorção de Compra | SHORT | 78.448,04 | 60m | 2026-09-01T00:13:00Z | *(Após término)* | N/A | **NULL (Não vencido)** |
| **3** | Exaustão de Compra | SHORT | 78.527,28 | 5m | 2026-08-31T23:43:00Z | 78.570,00 (+0.0544%) | 0 ms | **PASS** |
| **3** | Exaustão de Compra | SHORT | 78.527,28 | 15m | 2026-08-31T23:53:00Z | 78.573,40 (+0.0587%) | 0 ms | **PASS** |
| **3** | Exaustão de Compra | SHORT | 78.527,28 | 30m | 2026-09-01T00:08:00Z | *(Após término)* | N/A | **NULL (Não vencido)** |
| **3** | Exaustão de Compra | SHORT | 78.527,28 | 60m | 2026-09-01T00:38:00Z | *(Após término)* | N/A | **NULL (Não vencido)** |
| **4** | Exaustão de Venda | LONG | 78.558,01 | 5m | 2026-08-31T23:49:00Z | 78.555,30 (-0.0034%) | 0 ms | **PASS** |
| **4** | Exaustão de Venda | LONG | 78.558,01 | 15m | 2026-08-31T23:59:00Z | 78.564,00 (+0.0076%) | 0 ms | **PASS** |
| **4** | Exaustão de Venda | LONG | 78.558,01 | 30m | 2026-09-01T00:14:00Z | *(Após término)* | N/A | **NULL (Não vencido)** |
| **4** | Exaustão de Venda | LONG | 78.558,01 | 60m | 2026-09-01T00:44:00Z | *(Após término)* | N/A | **NULL (Não vencido)** |

### Verificações Específicas do Boundary:
- **Padrão antigo de drift (+1 min / 60s):** 0 ocorrências detectadas.
- **Drift em relação ao fechamento exato:** 0 ms em 100% dos horizontes elegíveis.
- **Horizontes com target após o fim do run:** Permaneceram estritamente `NULL` (sem preenchimento prematuro ou indevido).

---

## 3. SEMÂNTICA DIRECTION-AWARE — P1

Avaliação da classificação do lado e resultado dos sinais:

- **Sinal 1 (`Absorção de Compra`):** Inferido como **SHORT**.
  - 5m: Movimento `UP` (+0.0738%) $\rightarrow$ **LOSS**
  - 15m: Movimento `UP` (+0.0843%) $\rightarrow$ **LOSS**
  - 30m: Movimento `UP` (+0.2021%) $\rightarrow$ **LOSS**
  - 60m: Movimento `UP` (+0.2370%) $\rightarrow$ **LOSS**
- **Sinal 2 (`Absorção de Compra`):** Inferido como **SHORT**.
  - 5m: Movimento `UP` (+0.0512%) $\rightarrow$ **LOSS**
  - 15m: Movimento `UP` (+0.1629%) $\rightarrow$ **LOSS**
  - 30m: Movimento `UP` (+0.1555%) $\rightarrow$ **LOSS**
- **Sinal 3 (`Exaustão de Compra`):** Inferido como **SHORT**.
  - 5m: Movimento `UP` (+0.0544%) $\rightarrow$ **LOSS**
  - 15m: Movimento `UP` (+0.0587%) $\rightarrow$ **LOSS**
- **Sinal 4 (`Exaustão de Venda`):** Inferido como **LONG**.
  - 5m: Movimento `FLAT` (-0.0034%) $\rightarrow$ **FLAT**
  - 15m: Movimento `FLAT` (+0.0076%) $\rightarrow$ **FLAT**

**Auditoria de Inversões:**
- `SHORT + UP = WIN`: **0** ocorrências
- `SHORT + DOWN = LOSS`: **0** ocorrências
- **Resultado:** **PASS** (Zero inversões semânticas detectadas).

---

## 4. GATE DIRECIONAL

- Sinais emitidos no run: **3 SHORT**, **1 LONG**, **0 NEUTRAL**.
- Sinais SHORT observados e testados no fluxo: **SIM (3)**.
- Sinais LONG observados e testados no fluxo: **SIM (1)**.
- `RegimeBasedRules` / `get_directional_confidence`: Sinais SHORT consumiram `short_prob` e sinais LONG consumiram `long_prob`, com fallback seguro e tipagem finita (`0.0 <= val <= 1.0`).

---

## 5. OUTCOMES E AMOSTRAGEM

- **Total de Signal Outcomes:** 4
  - SHORT: 3 (75%)
  - LONG: 1 (25%)
  - NEUTRAL / UNKNOWN: 0 (0%)
- **Taxa de Preenchimento por Horizonte:**
  - **5m:** 4 elegíveis, 4 preenchidos (100%), 0 perdidos, 0 futuros.
  - **15m:** 4 elegíveis, 4 preenchidos (100%), 0 perdidos, 0 futuros.
  - **30m:** 2 elegíveis, 2 preenchidos (100%), 0 perdidos, 2 futuros (`NULL`).
  - **60m:** 1 elegível, 1 preenchido (100%), 0 perdidos, 3 futuros (`NULL`).

---

## 6. FLOW TEMPORAL & FLOW.Q

- **Estados de integridade observados ao longo da evolução temporal:**
  - `1m`: Inicia em `warm` (cobertura 77,1%), atinge `full` (99,9%) ao final da 1ª janela completa.
  - `5m`: Inicia em `warm` (cobertura 15,4%), atinge `full` (99,9%) no minuto 5.
  - `15m`: Inicia em `warm` (cobertura 5,1%), atinge `full` (99,1%) no minuto 15.
- **Propagação para os 11 Payloads IA:**
  - 100% dos payloads contêm o objeto `flow.q` com sub-janelas `1m`, `5m` e `15m`.
  - Mapeamento de status: `warm` $\rightarrow$ `s: "warm"`, `full` $\rightarrow$ `s: "full"`.
  - Cobertura efetiva `c` propagada numericamente (ex: `77.1`, `99.9`).
  - Ocorrências de `d15` presente com source truncado mas `q` ausente: **0** (**PASS**).

---

## 7. MATEMÁTICA FORENSE

- **FLOW:**
  - $BuyVolume + SellVolume = TotalVolume$: 100% consistente dentro da precisão de 3 casas decimais.
  - $BuyVolume - SellVolume = Delta$: 100% consistente. O guardrail de invariante atuou preventivamente em 2 janelas intermediárias corrigindo discrepâncias antes da emissão de eventos.
  - $Buy\% + Sell\% \approx 100$: Consistente.
- **ORDERBOOK:**
  - `bid_depth >= 0`, `ask_depth >= 0`, `spread >= 0`: 100% PASS.
  - `imbalance = (bid - ask) / (bid + ask)`: Verificado e consistente em todos os níveis de profundidade (L1, L5, L10, L25).
- **CVD:**
  - Soma e continuidade mantidas sem saltos artificiais.
- **FUNDING:**
  - Unidade percentual confirmada (% período / % anualizada), sem ocorrência de dupla conversão $(\times 100 \times 100)$.
- **VOLUME PROFILE:**
  - $VAL \le POC \le VAH$: 100% PASS em todos os eventos com perfil calculado.

---

## 8. CROSS-MODULE CONSISTENCY

- Comparação entre módulos para os mesmos boundaries:
  - Preço Âncora (`anchor_price`) vs Preço do Payload (`price.c`): diferença máxima observada de 0,40 USDT, decorrente de arredondamento inteiro de compactação para economia de tokens (`78422.11` $\rightarrow$ `78422`).
  - Sem contradições de polaridade entre Flow, Orderbook, Volume Profile e Regime.

---

## 9. DUPLICAÇÃO

- **Duplicatas Exatas de Eventos:** 0 (86 assinaturas únicas).
- **Duplicatas Semânticas:** 0.
- **Triggers Duplicados no mesmo minuto:** 0.
- **Chamadas de IA repetidas para o mesmo timestamp:** 0 (11 chamadas com timestamps únicos).
- **Taxa de Duplicação Geral:** **0,00%**.

---

## 10. PAYLOAD & TOKENS

Métricas calculadas sobre todos os 11 payloads de IA gerados no run:

- **Tamanho do Payload (Bytes):**
  - Mínimo: 2.773 bytes
  - Máximo: 3.818 bytes
  - Média: 3.446,4 bytes
  - Mediana (p50): 3.670,0 bytes
  - Percentil 95 (p95): 3.818,0 bytes
- **Tokens Estimados:**
  - Mínimo: 688 tokens
  - Máximo: 946 tokens
  - Média: 854,4 tokens
  - Mediana (p50): 911 tokens
  - Percentil 95 (p95): 946 tokens

### Distribuição Média por Seção:
- `summary`: 1.323,8 bytes (331,0 tokens / 38,4%)
- `ext`: 429,0 bytes (107,2 tokens / 12,4%)
- `flow`: 400,1 bytes (100,0 tokens / 11,6%)
- `tf`: 262,6 bytes (65,7 tokens / 7,6%)
- `sr`: 185,0 bytes (46,2 tokens / 5,4%)
- `ctx`: 167,9 bytes (42,0 tokens / 4,9%)
- `ob`: 119,9 bytes (30,0 tokens / 3,5%)
- `price`: 111,9 bytes (28,0 tokens / 3,2%)
- `alerts`, `regime`, `vwap`, `ofi`, `qual`, `liq`, `iceberg`, `w`: somam ~450 bytes (~13%)

### Oportunidades de Otimização (P2):
- A seção `summary` duplica em prosa informações já estruturadas em `flow`, `sr`, `regime` e `institutional`, gerando ~300 tokens de redundância semântica.
- `tf` envia stub `{"_": "no_data"}` quando as klines de prazos maiores falham por timeout de rede externa.

---

## 11. AUDITORIA DE FLOW.Q

- `flow.q` ausente: **0 / 11** (0%)
- `flow.q` completo: **11 / 11** (100%)
- Subchaves auditadas: `1m`, `5m`, `15m` com status válidos (`warm`, `full`) e coberturas entre 5,1% e 100,0%.
- Sem ocorrência de `NaN`, `Inf` ou estruturas malformadas.

---

## 12. EVENT SIMILARITY

- Módulo acionado durante o run com proteção `compatible_samples < 3` $\rightarrow$ `historical_win_rate = None`.
- Separação direcional de lado oposto confirmada no contrato canônico.

---

## 13. DADOS MACRO

- `dados/fred_cache.json`:
  - `TNX`: valor `4.73`, atualizado em `2026-08-31T23:58:49.549609+00:00`.
  - Freshness mantida dentro da janela de validade (< 1h).
  - Sem valores nulos, infinitos ou `NaN`.

---

## 14. LOG HEALTH & DIAGNÓSTICO DE INCIDENTES

- **Distribuição de Logs no Run Atual (64 min):**
  - **CRITICAL:** 2 ocorrências (avisos do health monitor sobre silêncio temporário dos módulos `orderbook` e `ws_error` durante a reconexão automática do WebSocket às 20:08 local).
  - **ERROR:** 10 ocorrências (5 timeouts na API REST klines de 15m/1h do `ContextCollector`; 2 alertas de silêncio; 1 timeout de PONG do WebSocket com reconexão bem-sucedida; 1 latência pontual).
  - **WARNING:** 120 ocorrências (59 conversões informativas de funding; 8 avisos de silêncio; 7 latências de dados; 7 alertas de clock drift em torno de +0.2s; 3 compactações de payload; 2 correções automáticas de invariante de delta; 1 reconexão).
- **Comparativo Pós-Fix:**
  - Zero ocorrências de "database is locked".
  - Zero crashes ou interrupções de processo.
  - Zero divergências de boundary.

---

## 15. AVALIAÇÃO DE PRONTIDÃO E NOTAS (0–100)

| Dimensão | Nota (0–100) | Observações |
|---|:---:|---|
| **Matemática** | 98 | Invariante corrigiu deltas zerados automaticamente; volumes e OB consistentes |
| **Temporal** | 100 | Transição warm $\rightarrow$ full e cálculo de cobertura temporal impecáveis |
| **Flow** | 99 | CVD, deltas, volumes e ratios íntegros |
| **Orderbook** | 96 | Níveis L1 a L25 íntegros; fallback em cache operou corretamente na reconexão |
| **S/R** | 98 | Níveis e distâncias íntegros |
| **Institutional** | 97 | Scores de whale e leilão consistentes |
| **Macro** | 92 | Cache FRED funcional; timeouts externos pontuais em klines 15m/1h |
| **OutcomeTracker** | 100 | Drift = 0 ms em 100% dos horizontes; boundaries futuros permaneceram NULL |
| **Direction-aware** | 100 | Semântica SHORT/LONG estrita; zero inversões |
| **Payload IA** | 98 | Schema e tipagem 100% conformes |
| **Tokens** | 88 | Payloads compactos (~854 tokens médios), porém com ~30% de redundância em prosa |
| **Persistência** | 100 | SQLite WAL sem locks; integridade relacional total |

### Classificação de Severidade:
- **P0 (Crítico/Corrupção):** **0**
- **P1 (Semântica Incorreta/Decisão):** **0**
- **P2 (Custo/Redundância/Observabilidade):** **2** (Redundância no `summary` do payload IA; timeouts de REST klines)
- **P3 (Cosmético/Log):** **2** (Avisos de funding rate no log; arredondamento de float no preço)

### Classificação Final: **READY**
O sistema atende a todos os critérios de prontidão técnica pós-fix: 0 P0 e 0 P1.
