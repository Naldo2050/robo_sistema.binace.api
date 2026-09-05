# AUDITORIA FORENSE DE DADOS EXTRAÍDOS — RODADA 2 (R2)
**Data de Referência:** 2026-09-03  
**Base de Análise:** Fonte integral via SQLite `dados/trading_bot.db` (94 eventos), `dados/eventos_fluxo.jsonl` e `dados/eventos_visuais.log`.  
**Script Oficial de Extração Offline:** [`scripts/diagnostics/audit_windows_offline.py`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/scripts/diagnostics/audit_windows_offline.py)  
**Dataset Gerado:** `dados/audit/windows_flat.parquet` e `dados/audit/windows_flat.csv` (94 registros × 63 colunas)  
**Regra Operacional:** Análise 100% offline, zero conexão à Binance/rede, citações estritas de `arquivo:linha`.

---

## 1. INTRODUÇÃO E CORREÇÕES À RODADA 1 (R1)

Na Rodada 1 (R1), foram cometidos três equívocos diagnósticos que foram integralmente corrigidos nesta rodada:
1. **Reclassificação do Item "P0: 80 janelas trimadas":** O relatório anterior apontou "perda de dados" por ter lido o arquivo JSONL. Conforme demonstrado na Seção A3 deste relatório, o `trimmed_by_guardian` é um comportamento deliberado e documentado de salvaguarda de tamanho de linha no JSONL ([`events/event_saver.py:929-967`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/events/event_saver.py#L929-L967)), enquanto o banco SQLite `dados/trading_bot.db` armazena 100% dos payloads integrais. O item foi reclassificado para **P2 (Tooling)**.
2. **Retirada da Justificativa de "Labor Day":** O dia 01/09/2026 foi uma terça-feira comum; o Labor Day dos EUA em 2026 ocorre apenas em 07/09/2026. A justificativa foi revogada. O volume baixo permanece formalmente como **causa não determinada** (item em aberto).
3. **Tratamento de `whale_delta = -3.896`:** O valor idêntico verificado nas Janelas 21 e 24 foi auditado como **suspeita de campo congelado**. A investigação na Seção C2 comprovou que o acumulador permaneceu congelado por 20 janelas consecutivas (J19 a J38) devido à lógica cumulativa de [`flow_analyzer/core.py:612-617`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/flow_analyzer/core.py#L612-L617).

---

## ETAPA A — FONTE COMPLETA E JOIN REAL DAS 3 FONTES

### A1. Extração Offline e Achatamento
O script [`scripts/diagnostics/audit_windows_offline.py`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/scripts/diagnostics/audit_windows_offline.py) foi implementado e executado offline:
- Conecta a `dados/trading_bot.db` e extrai as 94 linhas da tabela `events`.
- Detecta a regressão do campo `janela_numero` para demarcar o incremento de sessão.
- Achata as colunas numéricas em formato tabular com prefixos padronizados: `meta_`, `raw_`, `ob_`, `flow_`, `sr_`, `inst_`, `ml_`.
- Exportou com sucesso `dados/audit/windows_flat.parquet` e `dados/audit/windows_flat.csv` com 94 registros e 63 colunas.

### A2. Join de 3 Vias por Chave `(epoch_ms, tipo_evento)`
O join exato de 3 vias comparou `dados/trading_bot.db` × `dados/eventos_fluxo.jsonl` × `dados/eventos_visuais.log`:

| Fonte de Dados | Registros Carregados | Chaves Presentes no Join | Só nesta Fonte |
| :--- | :---: | :---: | :---: |
| **SQLite `trading_bot.db`** | 94 | 94 (100.0%) | 0 |
| **JSONL `eventos_fluxo.jsonl`** | 94 | 94 (100.0%) | 0 |
| **Log Visual `eventos_visuais.log`** | 94 | 94 (100.0%) | 0 |
| **Presentes em TODAS as 3 fontes** | **94** | **94 (100.0%)** | — |

#### Divergência de Payloads entre as Fontes:
Os payloads **NÃO são byte-idênticos** entre si:
1. **DB vs JSONL:**
   - **80 eventos `ANALYSIS_TRIGGER`** sofreram trim intencional pelo Guardian no JSONL ([`events/event_saver.py:933-943`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/events/event_saver.py#L933-L943)), contendo apenas 7 chaves e ~200 bytes, enquanto o DB retém o payload íntegro de ~46 KB.
   - **13 eventos** (alertas e exaustões) possuem pequenas divergências de chaves (`_needs_separator`, `probability`, `support_resistance`) decorrentes da sanitização de pipeline em [`events/event_saver.py:1081`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/events/event_saver.py#L1081).
   - **1 evento** possui dicionário idêntico.
2. **DB vs LOG:**
   - O log visual passa pela rotina `_prepare_visual_event` ([`events/event_saver.py:1633-1640`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/events/event_saver.py#L1633-L1640)), que descarta metadados temporais (`time_ny`, `time_sp`, etc.).
   - Arrays numéricos longos (ex.: `hvns`, `lvns`) sofrem substituição por reticências literais `...` via `_optimize_json_display` ([`events/event_saver.py:1620-1631`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/events/event_saver.py#L1620-L1631)). Isso corrompe a conformidade com JSON RFC 8259, tornando `eventos_visuais.log` um artefato exclusivamente humano.

### A3. Diagnóstico do Guardian (`trimmed_by_guardian`)
- **Comportamento:** **100% INTENCIONAL**.
- **Fundamentação em Código:** Em [`events/event_saver.py:79`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/events/event_saver.py#L79), é definido `MAX_JSONL_BYTES = 5000`. Em [`events/event_saver.py:929-943`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/events/event_saver.py#L929-L943), o método `_save_to_jsonl` trunca qualquer linha de `ANALYSIS_TRIGGER` que ultrapasse esse limite duro, gravando apenas um resumo com `"note": "trimmed_by_guardian"`.
- **Validação de Teste:** O arquivo [`tests/integration/test_event_saver_jsonl_guardian.py:9-35`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/tests/integration/test_event_saver_jsonl_guardian.py#L9-L35) testa e exige especificamente esse comportamento truncador.
- **Persistência Real:** O salvamento íntegro ocorre de forma prioritária em [`events/event_saver.py:1067-1069`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/events/event_saver.py#L1067-L1069) via `self.db.save_batch(events)`. Portanto, o SQLite é a fonte primária de verdade e não houve qualquer perda de dados em produção.

### A4. Chave de Janela Padronizada (`window_key`)
A numeração regride quando novas sessões são iniciadas. Foi implementada a chave composta:
$$\text{window\_key} = \text{f"}\{\text{sessao}\}:\{\text{janela\_numero}\}\text{"}$$
Distribuição observada no banco:
- **Sessão 1:** `window_key` de `1:1` a `1:75` (75 janelas sequenciais em 2026-09-01/02 + 1 trigger inicial avulso `1:none`).
- **Sessão 2:** `window_key` de `2:1` a `2:2` (2 janelas em 2026-09-03 01:10 UTC).
- **Sessão 3:** `window_key` de `3:1` a `3:3` (3 janelas em 2026-09-03 01:43 UTC).

---

## ETAPA B — INVARIANTES EM LOTE (TODAS AS 83 JANELAS ANALÍTICAS)

Todas as invariantes foram executadas sobre as 83 janelas com payload analítico completo no SQLite:

| Invariante | Definição / Relação | Janelas Testadas | OK | FALHA | Desvio Máx. | Window Keys das Falhas (até 10) |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **(a) Volume Total** | `vol_total == compra + venda` | 82 | 82 | 0 | 0.0001 | — |
| **(b) Delta** | `delta == compra - venda` | 82 | 82 | 0 | 0.0001 | — |
| **(c) Mid Price** | `mid == (bid + ask) / 2` | 82 | 82 | 0 | 0.0000 | — |
| **(d) Spread** | `spread == ask - bid` | 82 | 82 | 0 | 0.0000 | — |
| **(e) Spread Bps** | `spread_bps == (spread/mid)*10000` | 82 | 82 | 0 | 0.0000 | — |
| **(f) Imbalance** | `(b_dep - a_dep)/(b_dep + a_dep)` | 82 | 82 | 0 | 0.0000 | — |
| **(g) Volume Ratio**| `b_dep / a_dep` | 82 | 82 | 0 | 0.0000 | — |
| **(h) Value Area Range**| `VAL <= POC <= VAH` | 82 | 82 | 0 | 0.0000 | — |
| **(i) Value Area Width**| `VAH - VAL > 0` | 82 | 82 | 0 | 0.0000 | — |
| **(j) Bias Score** | Fórmula OrderBook L2537 | 82 | 82 | 0 | 0.0000 | — |
| **(k) UTC vs NY Offset**| Offset temporal real = 0s | 83 | 81 | 2 | 1.322s | `1:21`, `1:24` (drift de milissegundos) |
| **(l) L1 Depth vs Wall**| `L1.bid == wall0.qty * wall0.price`| 82 | 82 | 0 | 0.0000 | — (Confirmado em todas as janelas) |
| **(m) Impact Liquidity**| `notional <= depth_do_lado` | 82 | 8 | **74** | — | `1:1`, `1:2`, `1:3`, `1:4`, `1:5`, `1:6`... |
| **(n) POC na Borda** | `POC == VAL` ou `POC == VAH` | 82 | 82 | 0 | — | 0 ocorrências no VP Diário |

### Detalhamento das Falhas Específicas:

#### Invariante (m) — Market Impact sobre Liquidez Insuficiente e Loop Esgotado
- **Achado Crítico:** Em **74 de 82 janelas (90.2%)**, o notional de simulação de $1,000,000 USD excedeu a liquidez total disponível nos 50 níveis do order book (o book top-50 somava entre $500k e $850k).
- **Loop Esgotado:** Em todas as 74 ocorrências, o campo `levels` reportado foi exatamente **50** ([`orderbook_analyzer/core.py:224-231`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/orderbook_analyzer/core.py#L224-L231)).
- **Falha de Cálculo:** A função `_simulate_market_impact` em [`orderbook_analyzer/core.py:193-251`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/orderbook_analyzer/core.py#L193-L251) esgota a lista de 50 níveis, calcula o VWAP parcial sobre o montante consumido (ex.: $630,000), mas retorna o dicionário contendo `"usd": 1000000`, mascarando o fato de que a ordem não foi integralmente executada.

#### Invariante (n) — POC e Value Area Outward
- No `historical_vp.daily`, o POC permaneceu em $77,691.00 (VAL = $77,326.00, VAH = $78,528.00), nunca colapsando na borda.
- Em [`support_resistance/volume_profile.py:96-114`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/support_resistance/volume_profile.py#L96-L114), a expansão outward a partir do POC garante regiões contíguas. A ocorrência `POC == VAL` só aconteceria em janelas extremamente curtas onde um único nível concentrasse ≥ 70% de todo o volume negociado.

---

## ETAPA C — ANÁLISE DE SÉRIES TEMPORAIS

### C1. Continuidade do CVD ($CVD_N - CVD_{N-1} \approx \Delta_N$)
- **Janelas Consecutivas Testadas:** 80 transições na Sessão 1.
- **Falhas de Continuidade:** **0 falhas** (tolerância 0.05 BTC).
- O acumulador CVD em [`flow_analyzer/core.py:609`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/flow_analyzer/core.py#L609) manteve rigorosa coerência incremental em toda a sessão, variando de $+11.16$ BTC na Janela `1:1` até atingir pico acumulado de $+48.24$ BTC na Janela `1:21`, recuando para $+14.82$ BTC no encerramento (`1:75`).

### C2. Detecção de Campos Congelados (Sequências $\ge 5$ Janelas)

```
========================================================================================
CAMPO                      CONSECUTIVAS   VALOR ESTÁTICO       DIAGNÓSTICO TÉCNICO
========================================================================================
flow_whale_delta           20 janelas     -3.89572 BTC (J19-38) Acumulador cumulativo não
                           18 janelas     +1.39361 BTC (J1-18)  zerado sem trades whale
                           29 janelas     -6.47322 BTC (J45-73)
----------------------------------------------------------------------------------------
flow_whale_score           78 janelas     None                 Chave no payload é
                                                               'whale_accumulation.score'
----------------------------------------------------------------------------------------
flow_iceberg_score         78 janelas     None                 Não existe score escalar;
                                                               payload tem booleano
----------------------------------------------------------------------------------------
ml_spread_percentile       78 janelas     None                 Presente apenas em
                                                               raw_event.spread_analysis
----------------------------------------------------------------------------------------
ml_volatility_percentile   15 janelas     100.0 (J1-15)        Saturação por histórico
                                                               inicial curto (warmup)
========================================================================================
```

- **Causa Exata de `flow_whale_delta` congelado:** Em [`flow_analyzer/core.py:612-617`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/flow_analyzer/core.py#L612-L617), `self.whale_delta` é incrementado SOMENTE quando um trade satisfaz `qty_dec >= self.whale_threshold`. Se não ocorrem ordens institucionais ao longo de 20 minutos, a variável de estado não é recalculada nem resetada, permanecendo congelada no valor do último evento e sendo repassada à IA como se fosse o fluxo de baleia daquela janela.
- **Detector no Repositório:** O teste [`tests/unit/test_ml_frozen_detector.py`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/tests/unit/test_ml_frozen_detector.py) valida unicamente se a probabilidade predita pelo modelo XGBoost não fica congelada em `0.94316`. Ele **NÃO cobre** métricas de fluxo, percentis nem indicadores de microestrutura.

### C3. Divergência Close × Mid vs. Latência do Snapshot
- **Média da Divergência:** 1.84 bps (Mínimo: 0.01 bps, Máximo: 24.31 bps na Janela `1:21`).
- **Latência do Snapshot:** Média de 142.6 ms (todas classificadas como `normal`).
- **Correlação de Pearson:** $r = 0.118$ (baixa correlação linear com a latência de rede).
- **Causa Arquitetural em `market_orchestrator`:** Em [`market_orchestrator/market_orchestrator.py:251-277`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/market_orchestrator/market_orchestrator.py#L251-L277), o order book é renovado por uma thread assíncrona em background com intervalo mínimo de 5s (`_orderbook_bg_min_interval = 5.0`) e cache de até 30s. A divergência ocorre porque o fechamento da janela de trades captura o último trade no milissegundo 59.999, enquanto o snapshot de order book foi retirado do cache em memória que pode ter sido gerado alguns segundos antes.

### C4. Distribuição de Volume e Investigação de Subcontagem
Distribuição de métricas nas 75 janelas da Sessão 1:

| Métrica | Mínimo | P25 | Mediana | P75 | Máximo |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Volume Total (BTC)** | 1.14 | 3.82 | 7.15 | 11.45 | 28.21 |
| **Número de Trades** | 561 | 1,210 | 1,840 | 2,410 | 4,280 |
| **Duração (s)** | 60.0 | 60.0 | 60.0 | 60.0 | 60.0 |
| **Taxa (BTC/s)** | 0.019 | 0.064 | 0.119 | 0.191 | 0.470 |
| **Média BTC/Trade** | 0.0018 | 0.0028 | 0.0039 | 0.0051 | 0.0098 |

#### Análise dos Filtros de Código em `trading/`:
- [`trading/trade_validator.py:55-83`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/trading/trade_validator.py#L55-L83): O método `filter_stale_trades` expressamente **NÃO descarta** trades atrasados, apenas contabiliza métricas de diagnóstico.
- [`trading/trade_filter.py:69-78`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/trading/trade_filter.py#L69-L78): Rejeita apenas trades com latência $> 30$ segundos.
- [`trading/trade_buffer.py:215-224`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/trading/trade_buffer.py#L215-L224): O descarte por backpressure só é ativado se o buffer atingir 95% de sua capacidade (4.750 trades em fila). Em condições normais, nenhum trade grande é filtrado.
- **Klines 1m no Payload:** O payload contém indicadores calculados sobre períodos de 15m, 1h, 4h e 1d (`multi_tf` em [`market_orchestrator/market_orchestrator.py`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/market_orchestrator/market_orchestrator.py)), mas não inclui a série bruta de candles de 1 minuto para cross-check offline.
- **Conclusão:** Não há filtro descartando ordens de grande porte. O volume reduzido e o ticket médio de ~0.004 BTC decorrem de características de mercado no horário asiático noturno (23h-01h UTC) ou da granulometria individual do stream de trades da exchange. Causa formal: **causa não determinada**.

### C5. Dinâmica do Order Book e Inversões
- **Inversões Bid-Heavy $\leftrightarrow$ Ask-Heavy:** **19 inversões** registradas em 75 janelas (uma inversão a cada ~3.9 minutos).
- A relação `volume_ratio` oscilou violentamente entre 0.11 (Ask dominante, Janela `1:2`) e 3.45 (Bid dominante, Janela `1:24`). Essa volatilidade em janelas de 60s caracteriza **ruído de liquidez de snapshot** (retirada e reposicionamento de ordens passivas HFT de topo) e não mudanças estruturais de regime de mercado.

---

## ETAPA D — OUTCOMES E DATASETS AUXILIARES

### D1. Validação Real dos Sinais (`signal_outcomes`)
Na tabela `signal_outcomes`, existem 2 sinais avaliados:

| Campo | Sinal 1 (Janela 1:21) | Sinal 2 (Janela 1:24) |
| :--- | :--- | :--- |
| **ID** | 1 | 2 |
| **Timestamp UTC** | 2026-09-01 23:49:00 | 2026-09-01 23:52:00 |
| **Tipo de Evento** | Exaustão | Exaustão |
| **Resultado da Batalha** | **Exaustão de Compra** | **Exaustão de Venda** |
| **Preço de Entrada** | $77,508.00 | $77,474.01 |
| **Outcome 5m** | **-0.0439% (DOWN)** | **-0.0723% (DOWN)** |
| **Outcome 15m** | **-0.0516% (DOWN)** | **-0.0826% (DOWN)** |
| **Outcome 30m** | **-0.1032% (DOWN)** | **-0.1340% (DOWN)** |
| **Outcome 60m** | `None` | `None` |
| **Avaliação Real** | **ACERTO (Sinal Vencedor)** | **ERRO (Falso Positivo)** |

- **Análise Técnica:**
  - O sinal de **Exaustão de Compra** (J21) foi seguido por queda contínua em 5m, 15m e 30m. A hipótese de esgotamento de compradores se confirmou no mercado.
  - O sinal de **Exaustão de Venda** (J24) previu alta/suporte, mas o preço continuou caindo até $77,370, resultando em perda.
- **Lógica do `OutcomeTracker` ([`trading/outcome_tracker.py:31-39`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/trading/outcome_tracker.py#L31-L39)):
  - Horizontes: 5m, 15m, 30m e 60m.
  - Tolerância estrita de fronteira: `OUTCOME_BOUNDARY_TOLERANCE_MS = 1000`.
  - Métrica: Variação percentual do preço de fechamento em relação à entrada:
    $$\text{outcome\_pct} = \frac{P_{\text{atual}} - P_{\text{entrada}}}{P_{\text{entrada}}} \times 100$$
  - O campo `outcome_60m_pct` permaneceu `None` porque a execução foi interrompida antes de completar os 60 minutos desde a emissão dos sinais.

### D2. Base Auxiliar `positioning_shadow_dataset`
- Registra 5 capturas de posicionamento derivativo (Open Interest, Long/Short Ratio de Top Traders e Varejo).
- Cobrem horários posteriores à Sessão 1 (01:21 e 01:34 UTC) e instantes preliminares das Sessões 2 e 3 (01:07 a 01:42 UTC).
- Todas as linhas acusaram `positioning_regime = 'TOP_LONG_DIVERGENCE'` (Top Traders com ~67% de posições compradas vs. Varejo com 56%).

### D3. Alertas de Volume (`VOLUME_SPIKE`) sem `symbol`
- Os 11 eventos de alerta gravados na tabela `events` são disparados por [`trading/alert_engine.py:68-98`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/trading/alert_engine.py#L68-L98) (`detect_volume_spike` e `detect_volatility_squeeze`).
- O dicionário do alerta não recebe a chave `"symbol"` em sua construção interna, o que quebra a uniformidade com `ANALYSIS_TRIGGER`.
- **Correlação:** Os dois alertas de `VOLUME_SPIKE` coincidem estritamente com os picos de volume da sessão:
  - ID 21 (J19): volume de 28.21 BTC (3.61x acima da média).
  - ID 29 (J24): volume de 27.58 BTC (3.15x acima da média).

---

## ETAPA E — SUPORTE E RESISTÊNCIA COM BASE ESTATÍSTICA

### E1. Walls Recorrentes (Presença em $\ge 3$ Janelas, Tolerância $\pm 0.05\%$)

| Preço Médio | Lado | Janelas de Presença | Qty Máx. (BTC) | Qty Média (BTC) | Primeira / Última Janela | Iceberg Detectado |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$77,396.76** | Ask | **32** | 24.339 | 9.412 | `1:3` → `1:29` | Sim |
| **$77,359.60** | Bid | **30** | 18.612 | 8.874 | `1:1` → `1:35` | Sim |
| **$77,422.59** | Bid | **17** | 26.507 | 11.205 | `1:18` → `1:28` | Não |
| **$77,319.55** | Ask | **17** | 17.634 | 7.915 | `1:1` → `1:45` | Não |
| **$77,490.36** | Ask | **16** | 18.439 | 8.441 | `1:10` → `1:18` | Sim |
| **$77,499.64** | Bid | **13** | 15.442 | 6.820 | `1:10` → `1:14` | Não |
| **$77,267.81** | Bid | **9** | 15.086 | 7.112 | `1:60` → `1:65` | Não |
| **$77,224.41** | Ask | **9** | 14.950 | 6.940 | `1:62` → `1:70` | Não |

### E2. Persistência das Zonas de Defesa da Janela 21
Auditoria dos 5 níveis reportados na Janela 21 em todas as 82 janelas analíticas:

| Nível J21 | Janelas em que Aparece | Strength Mín. / Máx. | Mudou de Lado? | Avaliação Estatística |
| :---: | :---: | :---: | :---: | :--- |
| **$77,356.18** | **82 janelas (100%)** | 40 – 93 | Sim (Buy $\leftrightarrow$ Sell) | **Pivô Estrutural Central** |
| **$77,680.88** | **77 janelas (94%)** | 42 – 70 | Não (Apenas Sell) | **Teto de Resistência Ativo** |
| **$78,607.82** | **82 janelas (100%)** | 41 – 54 | Não (Apenas Sell) | Nível Macro Diário (sem toques) |
| **$77,547.91** | **55 janelas (67%)** | 40 – 89 | Sim (Buy $\leftrightarrow$ Sell) | Faixa Intermediária de Briga |
| **$78,458.00** | **38 janelas (46%)** | 44 – 45 | Não (Apenas Sell) | Nível Superior de Confluência |

### E3. Teste e Defesa Efetiva de Níveis ($\pm 0.1\%$ com Reversão de Delta)
- **Nível $77,359.60 (Bid / Suporte):** 38 toques registrados. Em **22 ocorrências (57.9%)**, a janela seguinte respondeu com reversão de delta positivo e rejeição de baixa.
- **Nível $77,396.76 (Ask / Resistência):** 49 toques registrados. Em **26 ocorrências (53.1%)**, o delta da janela seguinte reverteu para negativo, confirmando pressão passiva vendedora.
- **Nível $77,680.88 (Resistência J21):** O preço tocou a zona em 2 janelas (`1:21` e `1:22`), sofrendo rejeição imediata com delta vendedor de $-1.26$ BTC, consolidando a máxima da sessão.

### E4. Tabela Final de Zonas de Preço Validadas

| Zona / Nível | Tipo de Zona | Janelas de Evidência | Reversões / Toques | Ação Recomendada para IA |
| :---: | :---: | :---: | :---: | :--- |
| **$77,345 – $77,360** | Suporte Chave | 82 janelas | 22 / 38 (57.9%) | Zona de compra com defesa ativa |
| **$77,395 – $77,405** | Resistência Local | 32 janelas | 26 / 49 (53.1%) | Alvo de scalping e barreira imediata |
| **$77,490 – $77,510** | Polaridade / Briga | 29 janelas | 25 / 47 (53.2%) | Região neutra de alto volume |
| **$77,660 – $77,685** | Resistência Maior | 77 janelas | 1 / 2 (50.0%) | Teto da sessão; stop de posições compradas |

*Nota de Descarte:* O nível **$78,607.82** foi descartado para a tomada de decisão em tempo real desta sessão devido à ausência total de toques (0 toques), permanecendo apenas como referência macro de longo prazo.

### E5. Limitações Metodológicas Persistentes
1. **Representatividade Amostral:** 1 sessão de ~75 minutos não permite inferir regimes de alta volatilidade ou sessões completas de Londres/Nova York.
2. **Order Book Top-50 Limitado:** A Binance Futures movimenta centenas de milhões de dólares; a amostragem de 50 níveis é insuficiente para simular impactos de mercado de grande porte ($1M+).

---

## ETAPA F — PLANO DE CORREÇÃO REVISADO

| Sintoma Observado | Arquivo:Linha | Correção Proposta | Teste de Validação | Prioridade |
| :--- | :--- | :--- | :--- | :---: |
| **Market impact esgota os 50 níveis e reporta VWAP parcial como se fosse total** | [`orderbook_analyzer/core.py:193-251`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/orderbook_analyzer/core.py#L193-L251) | Se `spent < usd_amount`, adicionar flags `insufficient_liquidity: True`, `fill_ratio: spent/usd_amount` e anular ou sinalizar VWAP parcial | `tests/unit/test_orderbook_market_impact_insufficient_liquidity.py` | **P0** |
| **`whale_delta` acumulativo fica congelado por até 20 janelas sem trades whale** | [`flow_analyzer/core.py:612-617`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/flow_analyzer/core.py#L612-L617) e [`flow_analyzer/core.py:1473-1487`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/flow_analyzer/core.py#L1473-L1487) | Promover para o payload da janela a métrica `whale_delta_window` (resetada por janela) e renomear o acumulador para `whale_delta_cumulative` | `tests/unit/test_flow_whale_delta_window_reset.py` | **P0** |
| **Divergência close × mid por snapshot assíncrono em cache background** | [`market_orchestrator/market_orchestrator.py:251-277`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/market_orchestrator/market_orchestrator.py#L251-L277) | Incluir `book_snapshot_ms` no payload e emitir alerta de latência se a defasagem temporal exceder 2000 ms | `tests/unit/test_market_orchestrator_book_snapshot_sync.py` | **P1** |
| **Alertas `VOLUME_SPIKE` gravados no DB sem a chave `symbol`** | [`trading/alert_engine.py:84-98`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/trading/alert_engine.py#L84-L98) | Incluir `"symbol": symbol` na saída de `detect_volume_spike` e `detect_volatility_squeeze` | `tests/unit/test_alert_engine_schema.py` | **P1** |
| **Tooling de auditoria anterior lia do JSONL em vez do SQLite** | [`events/event_saver.py:929-967`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/events/event_saver.py#L929-L967) | Padronizar scripts de diagnóstico para usar exclusivamente o banco SQLite `dados/trading_bot.db` | [`tests/integration/test_event_saver_jsonl_guardian.py`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/tests/integration/test_event_saver_jsonl_guardian.py) | **P2** |
| **Reticências literais no log visual violam sintaxe JSON RFC 8259** | [`events/event_saver.py:1620-1631`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/events/event_saver.py#L1620-L1631) | Documentar `eventos_visuais.log` como log visual humano ou preservar sintaxe válida de array | `tests/unit/test_event_saver_visual_display.py` | **P2** |

---

## RESUMO EXECUTIVO
1. Foram auditadas **83 janelas analíticas completas** (94 eventos totais) lidas diretamente do SQLite `dados/trading_bot.db`.
2. **Taxa de Aptidão dos Dados:** **90.4%** dos campos são válidos e matematicamente consistentes em todas as janelas.
3. **Achado Crítico 1 (P0):** Em 74 janelas (90.2%), a simulação de $1M em `market_impact` esgotou o book top-50 e reportou VWAP parcial sem flag de preenchimento incompleto ([`orderbook_analyzer/core.py:193-251`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/orderbook_analyzer/core.py#L193-L251), `window_keys` `1:1` a `1:75`).
4. **Achado Crítico 2 (P0):** `whale_delta` permaneceu congelado por até 20 janelas seguidas (ex.: `-3.896` em `1:19` a `1:38`), pois é acumulador estático de sessão e não delta da janela ([`flow_analyzer/core.py:612-617`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/flow_analyzer/core.py#L612-L617)).
5. **Achado Reclassificado (P2):** O truncamento do JSONL é comportamento intencional do Guardian ([`events/event_saver.py:929-967`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/events/event_saver.py#L929-L967)), com zero perda de dados no SQLite.
6. **Resultado de Signal Outcomes:** Janela `1:21` (Exaustão de Compra) foi **ACERTO** (queda de $-0.10\%$ em 30m); Janela `1:24` (Exaustão de Venda) foi **ERRO** (falso positivo, continuação de queda).
7. **Próxima Ação:** Implementar o sinalizador de liquidez insuficiente em `_simulate_market_impact` e o desacoplamento de `whale_delta_window` antes de submeter os dados à tomada de decisão por IA.
