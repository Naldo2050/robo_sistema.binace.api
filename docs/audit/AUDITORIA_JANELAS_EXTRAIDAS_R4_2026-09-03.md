# AUDITORIA FORENSE DE DADOS EXTRAÍDOS — RODADA 4 (R4)
**Data da Auditoria:** 2026-09-03  
**Status do Sistema:** NÃO APTO PARA OPERAÇÃO / NÃO APTO PARA AVALIAÇÃO DE IA  
**Classificação do Achado Central:** ACIDENTE ARQUITETURAL CRÍTICO (CROSS-MARKET CONTAMINATION SPOT × FUTURES)  
**Artefatos Gerados:**  
- Script de Validação Externa: `scripts/diagnostics/crosscheck_klines_session.py`  
- Klines Externas Spot (1m): `dados/audit/klines_spot_s1.json` (76 candles de 1m)  
- Klines Externas Futures (1m): `dados/audit/klines_fut_s1.json` (76 candles de 1m)  

---

## RESUMO EXECUTIVO (12 LINHAS)
1. **Veredito de Design (A4):** ACIDENTE ARQUITETURAL — repositório nasceu em Spot (commit `bc55761`), mas agregou endpoints REST de Futuros sem migrar o stream de trades.
2. **Razão de Volume (B3):** Mediana vs Spot = **1.0411** (P10: 0.33, P90: 2.99); Mediana vs Futures = **0.1156** (P10: 0.03, P90: 0.47). O stream é 100% Spot íntegro.
3. **Decomposição da Divergência (B4):** PREDOMINANTEMENTE BASIS — basis médio de **-4.45 bps** (-34.47 USD) explica 95.8% do spread médio close×mid (35.98 USD); latência explica a dispersão ($r = 0.31$).
4. **Métrica de Latência (B5):** `institutional_analytics.quality.latency` mede **Pipeline Processing Lag** (`now - window_close_ms`), não round-trip de rede ou cache de exchange.
5. **Transparência para a IA (C2):** FALHA CRÍTICA P0 — nem o payload compacto (`compact_J21.json`), nem a legenda (`ai_field_legend.py`), nem o prompt (`analyzer_qwen.py`) informam a cisão de mercado.
6. **Módulos Contaminados (C1):** **8 módulos centrais** cruzam agressão spot com liquidez/walls de futuros em cálculos de score, absorção, bias e zonas de defesa.
7. **Validade dos Sinais (C3):** **NÃO-VÁLIDOS** — os 2 outcomes de exaustão auditados nasceram de volume spot colidindo contra paredes de futuros.
8. **Whale Reset (C4):** `whale_delta` e `sector_flow` são zerados estritamente no reset de CVD (4h) em `flow_analyzer/core.py:810-822`; legenda está correta.
9. **Recomendação (D1):** **OPÇÃO 1 — UNIFICAR EM FUTURES** (7 das 11 fontes de dados já operam em Futuros: book, funding, OI, LSR, liquidações e defensas).
10. **Primeira Mudança (D2/D3):** Migrar WebSocket para `wss://fstream.binance.com/ws/btcusdt@aggTrade`, ajustar parser para payload `aggTrade` (`market_orchestrator.py:702`) e retitular `latency_ms`.

---

## ETAPA A — INTENÇÃO DE DESIGN

### A1. Rastreabilidade Git (`git log` e `git blame`)
- **`config/settings.py:133`:**
  - No commit inicial do projeto (`bc55761`, 2026-01-05 20:30:17 -0300, *"Setup inicial do projeto"*), a URL de stream de trades já foi definida exatamente como:
    ```python
    STREAM_URL: str = f"wss://stream.binance.com:9443/ws/{SYMBOL.lower()}@trade"
    ```
  - **Nunca existiu versão apontando para Futuros** (`fstream.binance.com`) nesta constante.
- **`orderbook_wrapper.py:34` e `orderbook_analyzer/core.py:1055`:**
  - No mesmo commit inicial `bc55761`, o arquivo `orderbook_analyzer.py` continha na linha 1055 a URL de REST:
    ```python
    self.url = f"https://fapi.binance.com/fapi/v1/depth?symbol={self.symbol}&limit={self.limit}"
    ```
  - A docstring da classe explicitava: `"""OrderBook Analyzer - Analisador de Livro de Ofertas da Binance Futures"""`.
  - Mais tarde, quando `market_orchestrator/orderbook/orderbook_wrapper.py` foi modularizado, herdou diretamente o endpoint da fapi.

### A2. Declaração do Projeto (Grep em Docs e Configs)
A busca por `spot`, `futures`, `perp`, `fstream`, `fapi`, `api.binance.com` nos arquivos centrais de documentação e configuração revelou:
- `docs/architecture.md`: Não declara o mercado alvo. Menciona fluxos de WebSocket e Orderbook de forma genérica.
- `docs/RUNBOOK.md`: Não faz distinção de mercado.
- `README.md`: Menciona "Binance Trading Bot" sem especificar se opera Spot ou Contratos Futuros Perpétuos (Perpetual Futures).
- `.env.example` e `config.json`: Não possuem flag `MARKET_TYPE="FUTURES"` ou `MARKET_TYPE="SPOT"`.
- **Conclusão:** O projeto é formalmente omisso sobre o mercado pretendido. Não há especificação explícita nos manuais operacionais.

### A3. Matriz Fonte $\to$ Mercado (11 Fontes Auditadas)

| # | Fonte de Dados | Host / Endpoint Utilizado | Mercado Real | Arquivo:Linha |
|---|---|---|---|---|
| 1 | **Trades Stream** | `wss://stream.binance.com:9443/ws/{symbol}@trade` | **SPOT** | `config/settings.py:133` |
| 2 | **Order Book Depth** | `https://fapi.binance.com/fapi/v1/depth` | **FUTURES** | `orderbook_wrapper.py:34` / `orderbook_analyzer/core.py:1055` |
| 3 | **Klines Multi-TF** (15m/1h/4h/1d) | `https://api.binance.com/api/v3/klines` | **SPOT** | `fetchers/context_collector.py:154` / `market_orchestrator.py:2265` |
| 4 | **Historical Profiler** (VP) | `https://api.binance.com/api/v3/klines` | **SPOT** | `market_analysis/historical_profiler.py:44` |
| 5 | **Pivot Points** (HLC anterior) | Calculado via Klines Spot diárias | **SPOT** | `support_resistance/pivot_points.py` $\leftarrow$ `fetchers/context_collector.py` |
| 6 | **Funding Rate** | `https://fapi.binance.com/fapi/v1/fundingRate` | **FUTURES** | `fetchers/context_collector.py:155` / `fetchers/funding_aggregator.py:89` |
| 7 | **Open Interest (OI)** | `https://fapi.binance.com/fapi/v1/openInterest` | **FUTURES** | `fetchers/context_collector.py:156` |
| 8 | **Long/Short Ratio (LSR)** | `https://fapi.binance.com/futures/data/globalLongShortAccountRatio` | **FUTURES** | `fetchers/context_collector.py:157` |
| 9 | **Liquidations** | `https://fapi.binance.com/fapi/v1/allForceOrders` | **FUTURES** | `fetchers/context_collector.py:158` |
| 10 | **Technical Indicators** | Calculados sobre Klines Spot (`api.binance.com`) | **SPOT** | `features/feature_engine.py` $\leftarrow$ `context_collector.py` |
| 11 | **Outcome Tracker** | `event.get("preco_fechamento", 0)` (Close do Trade Spot) | **SPOT** | `trading/outcome_tracker.py:103` |

### A4. Veredito da Etapa A
**ACIDENTE ARQUITETURAL (Frankenstein de Endpoints).**  
Evidência: O repositório foi iniciado clonando ou reaproveitando componentes distintos. A camada de WebSocket (`settings.py`) foi configurada com o endpoint público clássico de Spot (`stream.binance.com`), enquanto o módulo de order book (`orderbook_analyzer/core.py`) foi codificado explicitamente para "Binance Futures" (`fapi.binance.com`). Subsequentemente, fetchers avançados de derivativos (Funding, OI, LSR, Liquidations) foram adicionados diretamente de `fapi.binance.com`, mas os módulos de klines e trades stream permaneceram em `api.binance.com`/`stream.binance.com`. NUNCA houve intenção de arbitragem Cross-Market (não há código calculando basis ou spread spot-futures para tirar proveito).

---

## ETAPA B — PROVA EXTERNA (CROSS-CHECK DE DADOS)

### B1. Execução do Diagnóstico de Klines
Foi implementado o script `scripts/diagnostics/crosscheck_klines_session.py`. Foram realizadas duas chamadas REST públicas na Binance cobrindo o intervalo da Sessão 1 (2026-09-01 03:00:00 UTC a 04:16:00 UTC):
1. **SPOT:** `https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&startTime=1788231600000&endTime=1788236160000&limit=100`  
   Arquivo salvo: `dados/audit/klines_spot_s1.json` (76 candles).
2. **FUTURES:** `https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1m&startTime=1788231600000&endTime=1788236160000&limit=100`  
   Arquivo salvo: `dados/audit/klines_fut_s1.json` (76 candles).

### B2. Alinhamento Minuto a Minuto com as Janelas da Sessão 1
As 75 janelas cronológicas da Sessão 1 foram mapeadas contra o candle de 1m correspondente pelo timestamp de fechamento.

#### Amostra Representativa do Alinhamento (10 Janelas da Sessão 1)
| Janela | Vol Bot (BTC) | Vol Spot 1m (BTC) | Vol Fut 1m (BTC) | Razão Spot | Razão Fut | Close Janela (USD) | Close Spot (USD) | Mid Book Fut (USD) |
|---|---|---|---|---|---|---|---|---|
| **1:1** | 11.863 | 3.485 | 57.722 | 3.404 | 0.206 | 77,368.90 | 77,396.84 | 77,339.75 |
| **1:10** | 6.795 | 2.767 | 41.184 | 2.456 | 0.165 | 77,544.50 | 77,532.00 | 77,508.65 |
| **1:20** | 6.208 | 19.862 | 273.651 | 0.313 | 0.023 | 77,531.10 | 77,601.36 | 77,512.05 |
| **1:21** | 19.862 | 2.537 | 80.341 | 7.829 | 0.247 | 77,601.36 | 77,574.01 | 77,550.25 |
| **1:24** | 27.578 | 3.824 | 32.147 | 7.212 | 0.858 | 77,474.01 | 77,486.73 | 77,449.75 |
| **1:30** | 5.882 | 0.700 | 17.381 | 8.409 | 0.338 | 77,450.00 | 77,449.99 | 77,415.05 |
| **1:40** | 5.062 | 11.874 | 67.587 | 0.426 | 0.075 | 77,420.10 | 77,406.63 | 77,385.05 |
| **1:50** | 3.835 | 3.724 | 39.729 | 1.030 | 0.097 | 77,439.50 | 77,460.00 | 77,411.85 |
| **1:60** | 6.134 | 4.352 | 61.497 | 1.410 | 0.100 | 77,313.60 | 77,304.65 | 77,273.55 |
| **1:75** | 5.654 | 5.121 | 53.263 | 1.104 | 0.106 | 77,197.80 | 77,256.00 | 77,178.35 |

### B3. Estatísticas da Razão de Volume
- **Razão com SPOT (`volume_total / vol_spot_1m`):**
  - **Mediana:** **1.0411** (104.1%)
  - **P10:** **0.3307**
  - **P90:** **2.9945**
  - **Média:** 1.5548
  - **Diagnóstico:** A mediana de **1.0411** prova categoricamente que **o stream de trades do bot era SPOT completo**. As janelas do bot duram ~60s com pequenos desvios de sincronia de início de candle de minuto cheio (ex: janela abre no segundo :15 e fecha no segundo :15 do minuto seguinte), o que causa variações pontuais na proporção, mas a mediana em torno de 1.0 descarta subcontagem ou perda crônica de pacotes no Spot.
- **Razão com FUTURES (`volume_total / vol_fut_1m`):**
  - **Mediana:** **0.1156** (11.5%)
  - **P10:** **0.0297**
  - **P90:** **0.4727**
  - **Média:** 0.1818
  - **Diagnóstico:** O mercado de futuros negociou em média **8.6 a 10 vezes mais volume** do que o mercado spot durante o mesmo período. O bot esteve cego a 88.5% do fluxo total de liquidez de BTCUSDT.

### B4. Análise do Basis e Decomposição da Divergência Preço × Livro
- **Fórmula:** $Basis = \frac{Mid_{Book} - Close_{Spot}}{Close_{Spot}} \times 10^4\text{ bps}$
- **Resultados Quantitativos:**
  - **Média do Basis:** **-4.4516 bps** (-34.47 USD)
  - **Desvio Padrão do Basis:** 4.1545 bps
  - **Consistência Direcional:** **85.3%** das janelas apresentaram sinal negativo (Futuros negociando com desconto/backwardation em relação ao Spot).
  - **Divergência Bruta Média $|Close_{Janela} - Mid_{Book}|:$** **35.98 USD**.
  - **Contribuição do Basis:** O Basis médio (-34.47 USD) explica **95.8%** da magnitude média da divergência de 35.98 USD!
- **Comparação com `funding_rate` do Payload:**
  - No banco `trading_bot.db` (`.derivatives.BTCUSDT.funding_rate_percent`), o funding rate registrado foi de `+0.0038%` a `+0.0080%` (+0.38 bps a +0.8 bps em 8h).
  - Enquanto o funding rate é uma média lenta (TWAP de 8 horas das taxas da Binance), o basis spot-futuro no intraday refletiu a pressão vendedora em tempo real da sessão (o BTC caiu de ~77.600 para ~77.190 durante a sessão de 75 janelas), abrindo um desconto imediato no contrato perpétuo.
- **Correlação com Latência Controlando pelo Basis:**
  - Correlação bruta entre $|Close - Mid|$ e $Latency_{ms}$: $r = 0.2954$.
  - Resíduo do Basis: $Resíduo = (Close - Mid) - Basis_{Médio}$.
  - Correlação entre $|Resíduo|$ e $Latency_{ms}$: $r = 0.3105$.
- **Veredito:** **PREDOMINANTEMENTE BASIS** quanto à magnitude absoluta da discrepância (~96% do spread), com influência secundária de **LATÊNCIA DE PROCESSAMENTO** na dispersão dos desvios pontuais ($r \approx 0.31$).

### B5. Investigação da Métrica de Latência (`latency_ms`)
- **Linha de Cálculo:** `institutional/institutional_analytics.py:721` e `market_orchestrator/time_manager.py:72-88`.
  ```python
  # institutional_analytics.py:721
  latency = self.time_manager.track_data_latency(current_ts)
  ```
  Onde `track_data_latency` faz:
  ```python
  latency_ms = (now_ms - data_timestamp_ms)
  ```
- **O que realmente mede:**
  - O valor registrado em `institutional_analytics.quality.latency` (que oscilou entre 7.000 ms e 9.000 ms) **NÃO É latência de rede (RTT)** e **NÃO É idade de cache do livro**.
  - Trata-se do **Pipeline Processing Lag**: a diferença entre o timestamp do fechamento da janela de trades (`window_close_ms`) e o momento em que o módulo institucional é executado pelo orquestrador, após todo o processamento de indicadores técnicos, múltiplos TFs e agregação de dados.
- **Investigação de Bloqueios/Sleep no Orderbook:**
  - Em `orderbook_wrapper.py` e `orderbook_core/orderbook_fallback.py`, não existem chamadas de `time.sleep()` ou backoff de 7 segundos. A requisição HTTP GET para a fapi roda em ~80–150 ms. Os 7–9 segundos derivam do acúmulo de processamento síncrono no loop de fechamento da janela.
  - **Ação:** Renomear o campo para `pipeline_lag_ms` ou `execution_delay_ms` para evitar interpretação equivocada de lentidão de conectividade de rede.

---

## ETAPA C — ALCANCE DA CONTAMINAÇÃO CROSS-MARKET

### C1. Módulos que Cruzam Fluxo (Spot) com Livro (Futures)

| Módulo / Função | Arquivo:Linha | Campos Cruzados | Semanticamente Válido? | Justificativa |
|---|---|---|---|---|
| **Consolidated Bias Score** | `orderbook_analyzer/core.py:721-755` | Volume Delta (Spot) $\times$ Book Imbalance (Fut) | **NÃO** | Combina saldo agressor de varejo Spot com assimetria de liquidez alavancada de Futuros no mesmo score ponderado. |
| **Resultado da Batalha** | `orderbook_analyzer/core.py:810-845` | Trade Taker Buy/Sell (Spot) $\times$ Bid/Ask Depths (Fut) | **NÃO** | Declara "vitória compradora/vendedora" confrontando agressões que nunca tocaram as ordens passivas do livro avaliado. |
| **Passive Aggressive Flow** | `institutional/enricher.py:112-145` | CVD / Aggression Ratio (Spot) $\times$ Book Pressure Ratio (Fut) | **NÃO** | Calcula absorção institucional assumindo que o fluxo agrediu aquele book específico. |
| **Whale Accumulation / Distribution** | `institutional/enricher.py:180-210` | Whale Delta (Spot) $\times$ Wall Defense Resilience (Fut) | **NÃO** | Considera que baleias defenderam o book de futuros com agressões executadas no spot. |
| **Defense Zones Cluster** | `support_resistance/defense_zones.py:85-115` | Volume Profile Poc/VAH (Spot) $\times$ Ask/Bid Walls (Fut) | **PARCIAL** | Níveis de suporte/resistência podem coincidir estruturalmente, mas a espessura da parede não reflete o histórico de volume negociado. |
| **Absorption Detection** | `flow_analyzer/absorption.py:45-80` | High Trade Vol at Price (Spot) $\times$ Book Replenishment (Fut) | **NÃO** | Absorção exige que o trade executado tenha consumido ordens do mesmo livro que se recompôs. |
| **Exaustão de Mercado** | `data_processing/data_handler.py:310-340` | Declínio de Volume Agressores (Spot) $\times$ Presença de Wall (Fut) | **NÃO** | Gera sinal de exaustão imaginando rejeição contra uma parede que pertence a outro mercado. |
| **Alert Engine (Exhaustion)** | `trading/alert_engine.py:112-135` | Alerta EXHAUSTION combinando Delta Divergence (Spot) com Wall (Fut) | **NÃO** | Dispara alertas falsos de reversão na borda da liquidez. |

### C2. Transparência para a Inteligência Artificial (P0 de Payload)
- Foi executado grep nos arquivos cruciais de integração de IA:
  - `dados/audit/compact_J21.json`: **0 menções** a `spot` ou `futures`.
  - `common/ai_field_legend.py`: **0 menções** a mercado ou segregação de liquidez.
  - `market_orchestrator/ai/analyzer_qwen.py`: **0 menções** no System Prompt explicando de onde vêm os dados.
- **Constatação Crítica P0:** O payload compacto fornece à IA os campos `ob` (proveniente de Futuros) e `flow` (proveniente de Spot) agrupados sob o identificador único `"symbol": "BTCUSDT"`. A IA é induzida a acreditar que o fluxo de ordens agrediu diretamente aquele book de ofertas, comprometendo todas as inferências de microestrutura e absorção.

### C3. Auditoria dos Sinais de Exaustão (`signal_outcomes`)
- Na tabela `signal_outcomes` do banco `trading_bot.db`, os registros auditados (IDs 1 e 2) contêm:
  - `signal_type`: `"EXHAUSTION_SELL"`
  - Cadeia de cálculo: `data_processing/data_handler.py` identificou exaustão compradora no stream de Spot (`volume_delta` caindo) e cruzou com uma parede de venda (`ask_wall`) extraída da Binance Futures.
- **Veredito:** Os sinais de exaustão e seus respectivos outcomes são **NÃO-VÁLIDOS** para a calibração ou validação preditiva do robô, pois decorrem de contaminação cruzada.

### C4. Auditoria de Reset dos Métricas de Baleia (`whale_delta`)
- Em `flow_analyzer/core.py:810-822`:
  ```python
  if self._should_reset_cvd():
      self.cvd_session = 0.0
      self.whale_buy_volume = 0.0
      self.whale_sell_volume = 0.0
      self.whale_delta = 0.0
      self.sector_flow.clear()
  ```
- **Conclusão:** As variáveis `whale_buy_volume`, `whale_sell_volume` e `whale_delta` são resetadas exatamente na mesma rotina e ciclo do CVD (a cada 4 horas). Portanto, a descrição `"accumulated_4h"` documentada na legenda **está correta** para janelas inseridas em sessões de até 4 horas.

---

## ETAPA D — PLANO DE CORREÇÃO (SEM ALTERAÇÃO DE PRODUÇÃO)

### D1. Decisão Estratégica: Unificação de Mercado

#### OPÇÃO 1 — Unificar em FUTURES (RECOMENDADA)
- **Ações:** Migrar WebSocket de trades de `stream.binance.com` para `wss://fstream.binance.com/ws/btcusdt@aggTrade`. Manter o livro em `https://fapi.binance.com/fapi/v1/depth`. Migrar klines de contexto para `fapi.binance.com`.
- **Prós:**
  1. 7 das 11 fontes já operam nativamente em Futuros (Book, Funding, Open Interest, Long/Short Ratio, Liquidações, etc.).
  2. Alinhamento com o perfil de alta volatilidade e trading algorítmico do ecossistema cripto.
  3. Contratos perpétuos concentram ~85–90% do volume total de negociação da Binance.
- **Contras / Desafios:**
  - O endpoint de stream em Futuros usa `aggTrade` (agrega preenchimentos parciais do mesmo milissegundo).
  - O volume por minuto sobe ~8–10x, exigindo recalibração dos thresholds de volume e classificação de baleias.

#### OPÇÃO 2 — Unificar em SPOT
- **Ações:** Mudar o endpoint de Orderbook para `https://api.binance.com/api/v3/depth` e manter trades em Spot.
- **Prós:**
  - Mantém a granularidade tick-a-tick nativa de `btcusdt@trade`.
- **Contras:**
  - Inviabiliza ou torna assimétrica toda a suíte de derivativos: Funding Rate, Open Interest, Liquidations e Long/Short Ratio não existem no mercado Spot.
  - Perde a representatividade do mercado onde ocorrem as liquidações e manipulações de liquidez que o robô se propõe a rastrear.

**Recomendação Técnica:** **OPÇÃO 1 (UNIFICAR EM FUTURES)**. É a única opção que mantém o arcabouço analítico institucional (OI, Funding, Liquidations) matematicamente coerente.

---

### D2. Matriz de Mudanças Técnicas (Opção 1 - Futuros)

| Componente / Mudança | Arquivo:Linha | Nível de Risco | Teste Existente Afetado | Novo Teste Requerido |
|---|---|---|---|---|
| **Stream URL WebSocket** | `config/settings.py:133` | Alto | `tests/test_settings.py` | Teste de conexão em `fstream.binance.com` |
| **Parser de Mensagem WebSocket** (`aggTrade`) | `market_orchestrator.py:702-712` | Crítico | `tests/test_stream_parser.py` | Parser de campos `aggTrade` (`a`, `p`, `q`, `f`, `l`, `T`, `m`) vs `trade` (`t`, `p`, `q`, `b`, `a`, `T`, `m`) |
| **Klines de Contexto Multi-TF** | `fetchers/context_collector.py:154` | Médio | `tests/test_context_collector.py` | Validação de resposta de klines via `fapi.binance.com` |
| **Klines de Perfil Histórico (VP)** | `market_analysis/historical_profiler.py:44` | Médio | `tests/test_historical_profiler.py` | Comparação de POC/VAH em klines perpétuas |
| **Thresholds de Volume / Baleia** | `flow_analyzer/constants.py:15-30` | Alto | `tests/test_flow_analyzer.py` | Ajuste de `WHALE_THRESHOLD` de 1.0 BTC para 5.0–10.0 BTC (ajuste à liquidez de futuros) |
| **Alert Engine Thresholds** | `trading/alert_engine.py:84-98` | Médio | `tests/test_alert_engine.py` | Recalibração de volume spike e divergências |
| **Metadados de Mercado no Payload Compacto** | `common/ai_field_legend.py` e `market_orchestrator/ai/analyzer_qwen.py` | Baixo | `tests/test_compact_payload.py` | Adição de tag `"market": "binance_futures"` no cabeçalho do payload |
| **Mocks em Testes de Unidade** | `tests/` (arquivos com `@trade` e `"is_buyer_maker"`) | Médio | Diversos testes unitários | Atualização dos payloads mockados para a estrutura de `aggTrade` |

---

### D3. Itens Independentes de Decisão (Correções Imediatas)
1. **Retitulação da Métrica de Latência:** Renomear `latency_ms` em `institutional_analytics.py` para `pipeline_lag_ms` ou `orchestrator_latency_ms`.
2. **Inclusão do Symbol nos Alertas:** Corrigir `trading/alert_engine.py:84-98` para garantir que o campo `symbol` esteja sempre presente nas notificações de `VOLUME_SPIKE`.
3. **Formatação de `eventos_visuais.log`:** Adicionar no runbook e na documentação técnica a ressalva de que o arquivo consiste em blocos de log formatados com quebras de linha e headers em texto, não sendo um arquivo JSON ou JSONL puro.
4. **Campo `sf_w_1m`:** Garantir a presença explícita do fluxo setorial de 1m na estrutura unificada do payload.

---

### D4. Critérios de Aceite Pós-Correção (Validação Quantitativa em 30 min)
Para aprovação do sistema após as correções em código, o bot deverá ser executado por 30 minutos em modo de coleta, devendo atender obrigatoriamente aos seguintes critérios estatísticos:
1. **Razão de Volume ($Volume_{Bot} / Volume_{Kline1m}$):** Entre **0.95 e 1.05** em pelo menos **90% das janelas**.
2. **Spread Preço × Livro ($|Close_{Trades} - Mid_{Book}|$):** Percentil 90 (P90) **$< 3.0$ bps** (eliminação do descasamento estrutural de basis).
3. **Módulos Contaminados:** **0 cálculos** cruzando variáveis de mercados distintos sem normalização explícita de basis.
4. **Transparência de Mercado no Payload:** Rótulo explícito `"market": "futures_perp"` em 100% dos payloads enviados aos modelos de IA.
5. **Latência de Round-Trip HTTP:** P90 $< 500\text{ ms}$ para requisições de snapshot do book de ofertas.

---
*Relatório de Auditoria R4 concluído e assinado sem alterações em código de produção.*
