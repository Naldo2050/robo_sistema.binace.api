# CONTRATOS DE DADOS INSTITUCIONAIS (DATA PROVENANCE CONTRACTS)
**Projeto:** Robô de Trading Binance API  
**Data:** 2026-09-01  
**Status:** ESPECIFICAÇÃO NORMATIVA (READ-ONLY)  
**Versão:** 1.0.0  

---

## 1. ESCOPO E FINALIDADE

Este documento estabelece o **Contrato de Proveniência e Integridade Semântica de Dados** para todas as features e capacidades institucionais da plataforma.

Cada feature institucional integrada ou planejada deve aderir estritamente ao contrato definido abaixo para prevenir corrupção silenciosa, descasamento de unidades, regressões de latência e alucinações da IA.

---

## 2. ESQUEMA PADRÃO DO CONTRATO

Cada entrada define:
- `name`: Nome canônico da feature no pipeline interno.
- `semantic_definition`: Significado financeiro e microestrutural exato.
- `source`: Origem primária do dado (Stream WS / REST / Cálculo interno).
- `unit`: Unidade canônica e escala normalizada.
- `valid_range`: Limites matemáticos aceitáveis (fora desse range = anomalia).
- `timestamp_source`: Origem do carimbo de tempo (Exchange / Local monotonic).
- `ttl`: Tempo máximo de vida útil em segundos antes de ser marcado como *stale*.
- `fallback_policy`: Comportamento determinístico quando a fonte primária falhar.
- `nullable`: Se o campo pode ser omitido/nulo (`True`/`False`).
- `payload_name`: Nome da chave no JSON compactado v3.1 enviado ao LLM.

---

## 3. TABELA DE CONTRATOS DE DADOS INSTITUCIONAIS

### 3.1. CVD (Cumulative Volume Delta)
- **`name`:** `cumulative_volume_delta`
- **`semantic_definition`:** Soma acumulada do volume comprador agressivo menos o volume vendedor agressivo ($\sum (\text{Vol}_{\text{buy}} - \text{Vol}_{\text{sell}})$).
- **`source`:** WebSocket Binance Futures (`wss://stream.binance.com:9443/ws/btcusdt@trade`), campo `m` (isBuyerMaker).
- **`unit`:** BTC (ativo base).
- **`valid_range`:** $[-\infty, +\infty]$ (normalmente $[-500.0, +500.0]$ BTC por janela de 15m).
- **`timestamp_source`:** `trade.T` (timestamp da exchange).
- **`ttl`:** 1.0 segundo (tempo real contínuo).
- **`fallback_policy`:** `delta_validator.py` reconstrói o delta via invariante $\text{buy} + \text{sell} = \text{total}$. Se ausente, emitir `0.0` com flag `flow:insufficient_trades`.
- **`nullable`:** `False`
- **`payload_name`:** `flow.d` (janela 1m), `flow.cvd_4h` (acumulado).

---

### 3.2. Trade Flow Imbalance
- **`name`:** `trade_flow_imbalance`
- **`semantic_definition`:** Razão normalizada entre delta e volume total $(\text{Vol}_{\text{buy}} - \text{Vol}_{\text{sell}}) / \text{Vol}_{\text{total}}$.
- **`source`:** `FlowAnalyzer` (`flow_analyzer/metrics.py:45`).
- **`unit`:** Adimensional $[-1.0, +1.0]$.
- **`valid_range`:** $[-1.0, +1.0]$.
- **`timestamp_source`:** Janela de fechamento `window.close_ms`.
- **`ttl`:** 60.0 segundos.
- **`fallback_policy`:** Se volume total $\le 0$, retornar `0.0`.
- **`nullable`:** `False`
- **`payload_name`:** `flow.imb`

---

### 3.3. L2 Order Flow Imbalance (L2 OFI)
- **`name`:** `l2_order_flow_imbalance`
- **`semantic_definition`:** Variação líquida de ordens passivas nos melhores níveis de bid e ask entre snapshots consecutivos do livro ($\Delta \text{Bid} - \Delta \text{Ask}$).
- **`source`:** `OrderBookAnalyzer` via WebSocket Depth (`@depth20@100ms`).
- **`unit`:** USD ou BTC por segundo.
- **`valid_range`:** $[-\infty, +\infty]$.
- **`timestamp_source`:** `depth.E` (timestamp da exchange).
- **`ttl`:** 5.0 segundos.
- **`fallback_policy`:** Se o livro estiver desatualizado ou em reconexão, retornar `null` (não emitir valor falso).
- **`nullable`:** `True`
- **`payload_name`:** `ob.ofi`

---

### 3.4. Funding Rate
- **`name`:** `funding_rate`
- **`semantic_definition`:** Taxa periódica de financiamento paga entre posições compradas e vendidas no contrato perpétuo da Binance Futures.
- **`source`:** REST Binance Futures `GET /fapi/v1/premiumIndex?symbol=BTCUSDT` (`lastFundingRate`).
- **`unit`:** Fração decimal canônica (ex: `0.0001` representa 0.01% ou 1 bp).
- **`valid_range`:** $[-0.05, +0.05]$ ($-5.0\%$ a $+5.0\%$).
- **`timestamp_source`:** `time` retornado pela API da Binance.
- **`ttl`:** 300 segundos (5 minutos).
- **`fallback_policy`:** Usar último valor válido em cache por até 30 minutos; se expirar, consultar TwelveData/FRED (`test_funding_rate_fallback.py`).
- **`nullable`:** `False`
- **`payload_name`:** `price.fr`

---

### 3.5. Open Interest (OI)
- **`name`:** `open_interest`
- **`semantic_definition`:** Número total de contratos de futuros em aberto (posições não liquidadas).
- **`source`:** REST Binance Futures `GET /fapi/v1/openInterest?symbol=BTCUSDT`.
- **`unit`:** Quantidade em BTC (ou USD notional correspondente).
- **`valid_range`:** $[1000.0, 500000.0]$ BTC.
- **`timestamp_source`:** `time` retornado pela API.
- **`ttl`:** 300 segundos (5 minutos).
- **`fallback_policy`:** Cache local de 15 minutos; se falhar, emitir `null` e marcar `derivatives:oi_unavailable`.
- **`nullable`:** `True`
- **`payload_name`:** `tf.oi` ou `ctx.oi`

---

### 3.6. VWAP (Volume Weighted Average Price)
- **`name`:** `vwap`
- **`semantic_definition`:** Preço médio ponderado pelo volume transacionado desde a ancoragem de sessão ($\sum (P \times V) / \sum V$).
- **`source`:** `common/technical_indicators.py:450` calculado sobre candles/trades da sessão.
- **`unit`:** USD (preço).
- **`valid_range`:** $[0.5 \times P_{\text{close}}, 2.0 \times P_{\text{close}}]$.
- **`timestamp_source`:** `window.close_ms`.
- **`ttl`:** 60 segundos.
- **`fallback_policy`:** Se volume total $= 0$, retornar `price_close`.
- **`nullable`:** `False`
- **`payload_name`:** `p.vw` ou `price.vwap`

---

### 3.7. Volume Profile: POC, VAH, VAL
- **`name`:** `volume_profile_metrics` (`poc`, `vah`, `val`)
- **`semantic_definition`:** 
  - `poc`: Preço com maior concentração de volume na sessão.
  - `vah`: Preço limite superior da Área de Valor (70% do volume).
  - `val`: Preço limite inferior da Área de Valor (70% do volume).
- **`source`:** `support_resistance/volume_profile.py:19` (calculado sobre histórico diário/semanal).
- **`unit`:** USD (preço).
- **`valid_range`:** $\text{VAL} < \text{POC} < \text{VAH}$ (Invariante Estrito).
- **`timestamp_source`:** Timestamp de fechamento do dia UTC.
- **`ttl`:** 3600 segundos (1 hora com recálculo na virada do dia UTC).
- **`fallback_policy`:** Se histórico for insuficiente, calcular sobre últimas 24h de candles de 1m.
- **`nullable`:** `False`
- **`payload_name`:** `sr.poc`, `sr.vah`, `sr.val` (ou `ctx.poc`, `ctx.val`, `ctx.vah`)

---

### 3.8. GARCH Volatility Forecast
- **`name`:** `garch_volatility_forecast`
- **`semantic_definition`:** Projeção da volatilidade condicional anualizada para a próxima hora via modelo autoregressivo heterocedástico GARCH(1,1).
- **`source`:** `common/technical_indicators.py:500`.
- **`unit`:** Desvio-padrão de retornos (adimensional contínuo, ex: `0.0025`).
- **`valid_range`:** $[0.00001, 0.50]$.
- **`timestamp_source`:** `window.close_ms`.
- **`ttl`:** 300 segundos.
- **`fallback_policy`:** Desvio padrão histórico dos últimos 50 retornos.
- **`nullable`:** `False`
- **`payload_name`:** `ext.garch`

---

### 3.9. Hurst Exponent
- **`name`:** `hurst_exponent`
- **`semantic_definition`:** Medida de persistência/memória de longo prazo da série temporal via análise R/S ($H < 0.5$: mean-reverting; $H \approx 0.5$: passeio aleatório; $H > 0.5$: persistente/tendência).
- **`source`:** `common/technical_indicators.py:282`.
- **`unit`:** Adimensional $[0.0, 1.0]$.
- **`valid_range`:** $[0.05, 0.95]$.
- **`timestamp_source`:** `window.close_ms`.
- **`ttl`:** 300 segundos.
- **`fallback_policy`:** Retornar `0.50` (passeio aleatório neutro) se amostra $< 50$ pontos.
- **`nullable`:** `False`
- **`payload_name`:** `ext.hurst`

---

### 3.10. Shannon Entropy
- **`name`:** `shannon_entropy`
- **`semantic_definition`:** Grau de incerteza informacional ou dispersão da distribuição de retornos de preços ($H = -\sum p_i \log_2 p_i$).
- **`source`:** `common/technical_indicators.py:309`.
- **`unit`:** Bits de informação (escala típica $1.0$ a $5.0$).
- **`valid_range`:** $[0.0, 6.0]$.
- **`timestamp_source`:** `window.close_ms`.
- **`ttl`:** 300 segundos.
- **`fallback_policy`:** Retornar `null` se dados insuficientes.
- **`nullable`:** `True`
- **`payload_name`:** `ext.entropy`

---

### 3.11. Kalman Filter Estimated Price
- **`name`:** `kalman_price`
- **`semantic_definition`:** Estado latente estimado do preço verdadeiro eliminando ruído gaussiano de microestrutura via filtro de Kalman 1D.
- **`source`:** `common/technical_indicators.py:330`.
- **`unit`:** USD (preço).
- **`valid_range`:** $[0.8 \times P_{\text{close}}, 1.2 \times P_{\text{close}}]$.
- **`timestamp_source`:** `window.close_ms`.
- **`ttl`:** 60 segundos.
- **`fallback_policy`:** Retornar `price_close` em caso de divergência numérica.
- **`nullable`:** `False`
- **`payload_name`:** `ext.kalman`

---

### 3.12. Whale Activity Score
- **`name`:** `whale_score`
- **`semantic_definition`:** Índice ponderado de acumulação ($>+30$) ou distribuição ($<-30$) de grandes ordens de mercado ($\ge 1.0$ BTC).
- **`source`:** `flow_analyzer/whale_score.py:39`.
- **`unit`:** Escala discreta $[-100, +100]$.
- **`valid_range`:** $[-100, +100]$.
- **`timestamp_source`:** `window.close_ms`.
- **`ttl`:** 300 segundos.
- **`fallback_policy`:** Retornar `0` (neutro) se não houver grandes ordens na janela.
- **`nullable`:** `False`
- **`payload_name`:** `w.s`

---

### 3.13. Fear & Greed Index
- **`name`:** `fear_and_greed_index`
- **`semantic_definition`:** Índice de sentimento macro de mercado compilado por Alternative.me.
- **`source`:** REST `https://api.alternative.me/fng/`.
- **`unit`:** Inteiro $[0, 100]$.
- **`valid_range`:** $[0, 100]$.
- **`timestamp_source`:** Timestamp retornado pela API.
- **`ttl`:** 1800 segundos (30 minutos).
- **`fallback_policy`:** Retornar último valor em cache por até 24 horas; se indisponível, omitir.
- **`nullable`:** `True`
- **`payload_name`:** `ctx.fng`

---

### 3.14. Market Regime
- **`name`:** `market_regime`
- **`semantic_definition`:** Classificação do estado da microestrutura em `BREAKOUT`, `TRENDING`, `RANGE_BOUND`, `MEAN_REVERTING`.
- **`source`:** `market_analysis/regime_detector.py:20`.
- **`unit`:** Categórico (`enum`).
- **`valid_range`:** `{"BREAKOUT", "TRENDING", "RANGE_BOUND", "MEAN_REVERTING", "UNKNOWN"}`.
- **`timestamp_source`:** `window.close_ms`.
- **`ttl`:** 300 segundos.
- **`fallback_policy`:** `RANGE_BOUND` com confiança $0.30$.
- **`nullable`:** `False`
- **`payload_name`:** `regime.reg` (ou `r.cs`)

---

### 3.15. Global Long/Short Account Ratio
- **`name`:** `global_account_ratio`
- **`semantic_definition`:** Razão entre número de contas compradas e vendidas de todos os traders na Binance Futures USD-M.
- **`source`:** REST Binance Futures `GET /futures/data/globalLongShortAccountRatio?symbol=BTCUSDT&period=5m&limit=30` (`longShortRatio`).
- **`unit`:** Adimensional ($>0.0$, ex: `1.28`).
- **`valid_range`:** $[0.01, 50.0]$.
- **`timestamp_source`:** `timestamp` retornado pela API da Binance.
- **`ttl`:** 300 segundos (5 minutos, TTL cache) / 900 segundos (stale cutoff).
- **`fallback_policy`:** Emitir `null` e marcar `is_available=False` (nunca converter ausência em `1.0` ou `0.0`).
- **`nullable`:** `True`
- **`payload_name`:** `pos.ga`

---

### 3.16. Top Trader Long/Short Account Ratio
- **`name`:** `top_account_ratio`
- **`semantic_definition`:** Razão entre número de contas compradas e vendidas entre os top 20% traders por volume/margem.
- **`source`:** REST Binance Futures `GET /futures/data/topLongShortAccountRatio?symbol=BTCUSDT&period=5m&limit=30` (`longShortRatio`).
- **`unit`:** Adimensional ($>0.0$, ex: `1.39`).
- **`valid_range`:** $[0.01, 50.0]$.
- **`timestamp_source`:** `timestamp` retornado pela API da Binance.
- **`ttl`:** 300 segundos (5 minutos) / 900 segundos (stale cutoff).
- **`fallback_policy`:** Emitir `null` e marcar `is_available=False`.
- **`nullable`:** `True`
- **`payload_name`:** `pos.ta`

---

### 3.17. Top Trader Long/Short Position Ratio
- **`name`:** `top_position_ratio`
- **`semantic_definition`:** Razão entre volume financeiro nocional total de posições compradas vs vendidas detido pelos top 20% traders.
- **`source`:** REST Binance Futures `GET /futures/data/topLongShortPositionRatio?symbol=BTCUSDT&period=5m&limit=30` (`longShortRatio`).
- **`unit`:** Adimensional ($>0.0$, ex: `2.07`).
- **`valid_range`:** $[0.01, 50.0]$.
- **`timestamp_source`:** `timestamp` retornado pela API da Binance.
- **`ttl`:** 300 segundos (5 minutos) / 900 segundos (stale cutoff).
- **`fallback_policy`:** Emitir `null` e marcar `is_available=False`.
- **`nullable`:** `True`
- **`payload_name`:** `pos.tp`

---

### 3.18. Open Interest Deltas (1h & 4h)
- **`name`:** `oi_delta_1h`, `oi_delta_4h`
- **`semantic_definition`:** Variação percentual relativa do volume nocional do Open Interest em relação a 1 hora (12 barras de 5m) e 4 horas (48 barras de 5m).
- **`source`:** REST Binance Futures `GET /futures/data/openInterestHist?symbol=BTCUSDT&period=5m&limit=60`.
- **`unit`:** Fração decimal (ex: `+0.024` = +2.4%).
- **`valid_range`:** $[-0.90, +10.0]$.
- **`timestamp_source`:** `timestamp` retornado pela API da Binance.
- **`ttl`:** 300 segundos.
- **`fallback_policy`:** Em caso de histórico insuficiente (< 13 barras para 1h ou < 49 barras para 4h), emitir `null` (não emitir `0.0`).
- **`nullable`:** `True`
- **`payload_name`:** `pos.od1`, `pos.od4`

---

### 3.19. Crypto COT Positioning Regime
- **`name`:** `positioning_regime`
- **`semantic_definition`:** Classificação determinística e explicável da estrutura macro/institucional de posicionamento (`CROWDED_LONG`, `CROWDED_SHORT`, `TOP_LONG_DIVERGENCE`, `TOP_SHORT_DIVERGENCE`, `OI_EXPANSION`, `SQUEEZE_RISK`, `NEUTRAL`, `UNKNOWN`, `PARTIAL`).
- **`source`:** `institutional/crypto_cot.py` (`CryptoCOT.analyze`).
- **`unit`:** Categórico (`PositioningRegime` enum) + `reasons: list[str]`.
- **`valid_range`:** Enum válido.
- **`timestamp_source`:** `observed_at` monotonic / UTC.
- **`ttl`:** 300 segundos.
- **`fallback_policy`:** Se dados nulos ou obsoletos, classificar como `UNKNOWN` ou `PARTIAL`.
- **`nullable`:** `False`
- **`payload_name`:** `pos.rg`

---

### 3.20. Canonical Session VWAP (Ancorado em UTC 00:00:00)
- **`name`:** `session_vwap`, `session_vwap_distance`
- **`semantic_definition`:** Preço médio ponderado por volume acumulado desde UTC 00:00:00.000 do dia atual. Serve como benchmark institucional de execução e localização relativa de preço.
- **`source`:** `institutional/session_vwap.py` (`SessionVWAPTracker`).
- **`method`:** `"ohlcv_1m_typical_price"`, usando Typical Price $(H + L + C) / 3$ e Volume de cada barra de 1 minuto da Binance Futures.
- **`unit`:** USD para `session_vwap` / Fração decimal para `distance_fraction` $((P - \text{VWAP}) / \text{VWAP})$.
- **`valid_range`:** Preço $> 0.0$ / Distância $[-0.50, +0.50]$.
- **`anchor`:** `UTC_00_00_00` (rollover diário automático em 00:00 UTC).
- **`recovery_policy`:** Em caso de restart intraday, reconstrói acumuladores via REST Binance Futures (`/fapi/v1/klines?symbol=BTCUSDT&interval=1m&startTime={00_utc_ms}`).
- **`timestamp_source`:** `timestamp_ms` da barra de 1m mais recente.
- **`ttl`:** 300 segundos (`_MAX_STALE_SECONDS`).
- **`fallback_policy`:** Em warm-up ou ausência de dados, emitir `status="WARMING_UP"` / `status="ERROR"` e `is_valid=False`.
- **`nullable`:** `True`
- **`payload_name`:** `vwap.svw`, `vwap.dist`, `vwap.side`, `vwap.m`

---

### 3.21. Break of Structure (BOS)
- **`name`:** `market_structure.bos`
- **`semantic_definition`:** Rompimento estrutural confirmado por fechamento de candle acima de um Swing High prévio (`bullish`) ou abaixo de um Swing Low prévio (`bearish`).
- **`source`:** `institutional/market_structure.py` (`MarketStructureDetector`).
- **`schema_version`:** `1.1.0`
- **`swing_definition`:** Pivot confirmado com $L=2$ barras à esquerda e $R=2$ barras à direita. Ativado apenas em $t \ge \text{center} + R$.
- **`unit`:** String categórica / Preço numérico em USD (ex: `"BULL_75000"`).
- **`event_id_format`:** `{symbol}:{timeframe}:BOS_{TYPE}:{level}:{swing_ts}:{confirmed_ts}:{schema_version}`
- **`anti_lookahead_guarantee`:** Zero repaint comprovado via Prefix Invariance.
- **`timeframe`:** `"1m"` (canônico do fluxo de market data).
- **`nullable`:** `True`
- **`payload_name`:** `ms.bos`, `ms.b_str`, `ms.tf`

---

### 3.22. Liquidity Sweep
- **`name`:** `market_structure.sweep`
- **`semantic_definition`:** Excursão de preço além de um nível de Swing prévio seguida de rejeição/reclaim no fechamento do mesmo candle (wick além do nível, mas close aquém).
- **`source`:** `institutional/market_structure.py` (`MarketStructureDetector`).
- **`schema_version`:** `1.1.0`
- **`types`:** `buy_side` (sweep de Swing High / stops de shorts), `sell_side` (sweep de Swing Low / stops de longs) e `both` (candle largo que varre ambos os lados).
- **`unit`:** String categórica / Preço numérico em USD (ex: `"BUY_75000"`, `"BOTH_75000"`).
- **`event_id_format`:** `{symbol}:{timeframe}:SWEEP_{TYPE}:{level}:{swing_ts}:{confirmed_ts}:{schema_version}`
- **`anti_lookahead_guarantee`:** Zero repaint comprovado. Mutuamente exclusivo com BOS no mesmo candle e nível.
- **`timeframe`:** `"1m"` (canônico do fluxo de market data).
- **`nullable`:** `True`
- **`payload_name`:** `ms.sw`, `ms.sw_exc`, `ms.tf`

---

## 4. MATRIZ DE TESTES E VALIDAÇÃO DE CONTRATO

| Feature | Teste de Regressão Unitário | Teste de Invariante de Range | Validação em Runtime |
|---|---|---|---|
| **CVD** | `tests/unit/test_flow_analyzer_metrics.py` | `test_delta_validator.py` | `data_quality_validator.py` |
| **Orderbook** | `tests/unit/test_orderbook_analyzer.py` | `_assert_sorted` em `core.py` | `SpreadTracker` |
| **Funding Rate** | `tests/payload/test_funding_rate_pipeline_p0.py` | Range $[-0.05, 0.05]$ | `context_collector.py` |
| **Volume Profile** | `tests/unit/test_volume_profile_etapa4_regression.py` | Invariante `VAL < POC < VAH` | `InstitutionalAnalyticsEngine` |
| **Whale Score** | `tests/unit/test_institutional_whale.py` | Range $[-100, 100]$ | `enricher.py` |
| **Regime Rules** | `tests/unit/test_regime_rules.py` | Enum válido | `RegimeBasedRules` |
| **Binance Positioning (P1.1)** | `tests/unit/test_binance_positioning_p1_1.py` | 3 L/S Ratios > 0, deltas com warm-up | `BinancePositioningFetcher` |
| **Crypto COT (P1.1)** | `tests/unit/test_binance_positioning_p1_1.py` | Regimes determinísticos + reasons | `CryptoCOT` |
| **Session VWAP (P1.2)** | `tests/unit/test_session_vwap_p1_2.py` | Batch == Incremental, UTC 00:00 rollover, Restart recovery | `SessionVWAPTracker` |
| **Market Structure (P1.3)** | `tests/unit/test_market_structure_p1_3.py` | Prefix Invariance, Exclusão Mútua BOS vs Sweep | `MarketStructureDetector` |

