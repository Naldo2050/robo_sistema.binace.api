# Dívida Técnica de Testes (TEST_DEBT)

Este documento registra os testes atualmente quebrados no repositório, identificando o commit exato de regressão (via `git log -S`) e a causa raiz técnica.

> **Status:** Registrado para resolução futura. Nenhuma correção de lógica executada nesta fase.

---

## 1. Testes de `CryptoCOT` (`tests/unit/test_institutional_cot.py`)

* **Commit que quebrou:** `e8bb9d5` (*freeze(o1): production shadow observation baseline (P0-P1.3C, V1-V1.2, test-the-tester, O1 tools)*)
* **Causa Raiz Geral:** O commit `e8bb9d5` reestruturou a classe `CryptoCOT` (`institutional/crypto_cot.py`) de um modelo de rastreamento com estado em memória (`add_data()`, `reset()`, etc.) para um analisador estatutário/funcional sem estado baseado no contrato `positioning_data` e no enum `PositioningRegime`. Os testes unitários originais não foram migrados para a nova interface.

| # | Teste Unitário | Commit | Causa Específica |
|---|---|---|---|
| 1 | `TestCOTInit::test_default` | `e8bb9d5` | `AttributeError: 'CryptoCOT' object has no attribute 'data_points'` (propriedades `data_points` e `latest` removidas do novo design). |
| 2 | `TestCOTAddData::test_add_single` | `e8bb9d5` | `AttributeError: 'CryptoCOT' object has no attribute 'add_data'` (método de acúmulo temporal `add_data` foi descontinuado). |
| 3 | `TestCOTFundingAnalysis::test_extreme_positive_funding` | `e8bb9d5` | `AttributeError: 'CryptoCOT' object has no attribute 'add_data'`. |
| 4 | `TestCOTFundingAnalysis::test_extreme_negative_funding` | `e8bb9d5` | `AttributeError: 'CryptoCOT' object has no attribute 'add_data'`. |
| 5 | `TestCOTFundingAnalysis::test_normal_funding_no_signal` | `e8bb9d5` | `AttributeError: 'CryptoCOT' object has no attribute 'add_data'`. |
| 6 | `TestCOTOIAnalysis::test_oi_confirming_uptrend` | `e8bb9d5` | `TypeError: CryptoCOT.__init__() got an unexpected keyword argument 'oi_change_threshold_pct'`. |
| 7 | `TestCOTSqueeze::test_short_squeeze_conditions` | `e8bb9d5` | `AttributeError: 'CryptoCOT' object has no attribute 'add_data'`. |
| 8 | `TestCOTSqueeze::test_long_squeeze_conditions` | `e8bb9d5` | `AttributeError: 'CryptoCOT' object has no attribute 'add_data'`. |
| 9 | `TestCOTAnalysis::test_analyze_empty` | `e8bb9d5` | `TypeError: CryptoCOT.analyze() missing 1 required positional argument: 'positioning_data'`. |
| 10 | `TestCOTAnalysis::test_analyze_complete` | `e8bb9d5` | `AttributeError: 'CryptoCOT' object has no attribute 'add_data'`. |
| 11 | `TestCOTReset::test_reset` | `e8bb9d5` | `AttributeError: 'CryptoCOT' object has no attribute 'add_data'` (método `reset` e armazenamento interno foram descontinuados). |

---

## 2. Teste de Estrutura do Snapshot Price (`tests/payload/test_build_compact_payload_snapshot.py`)

* **Commit que quebrou:** `e8bb9d5` (*freeze(o1): production shadow observation baseline (P0-P1.3C, V1-V1.2, test-the-tester, O1 tools)*)
* **Causa Raiz:** O commit `e8bb9d5` incluiu a chave `fr` (*funding rate*) dentro da seção `price` do payload compacto (`market_orchestrator/ai/payload_builder_compact.py`), porém não adicionou `'fr'` ao conjunto de chaves permitidas (`OPTIONAL_PRICE_KEYS`) em `tests/payload/test_build_compact_payload_snapshot.py`.

| # | Teste Unitário | Commit | Causa Específica |
|---|---|---|---|
| 12 | `test_snapshot_price_section_structure` | `e8bb9d5` | `AssertionError: Chaves inesperadas em price: {'fr'}` (chave `fr` adicionada no builder sem atualizar `OPTIONAL_PRICE_KEYS`). |

---

## 3. Calibração de Thresholds de Fluxo e Volume Futures (Rodada 2026-09-05)

* **Motivação:** No mercado de Binance Futures (USD-M `btcusdt@aggTrade`), a distribuição de tamanho de ordens e volume é significativamente mais elevada que em Spot. Manter limiares legados de Spot (whale trade >= 1.0 BTC, spike por multiplicador puro sem piso p95) distorcia métricas e saturava alertas.
* **Base Empírica:** 
  - `dados/audit/aggtrades_sample.json`: amostra de 1h (2026-09-04 14:00–15:00 UTC) com p99 = 2.0 BTC.
  - `dados/audit/klines_fut_28d.json`: 28 dias de klines 1m (40.321 velas) gerando `config/volume_baseline_fut.json` (p95 global = 367.6 BTC/minuto e tabela horária 0..23 UTC).

### Ajustes de Código e Testes Atualizados

| Componente / Arquivo | Ajuste Realizado | Testes Unitários Ajustados | Motivo do Ajuste no Teste |
|---|---|---|---|
| `flow_analyzer/constants.py` | `DEFAULT_WHALE_TRADE_THRESHOLD`: 1.0 → 2.0 BTC.<br>`ORDER_SIZE_BUCKETS`: retail (0, 0.2), mid (0.2, 2.0), whale (2.0, None). | `tests/unit/test_flow_consistency_regression.py` | Fixtures de teste utilizavam 0.99 BTC para retail e 1.5 BTC para whale; atualizado para 0.1999 BTC (retail) e 2.5 BTC / 2.0 BTC (whale). |
| `flow_analyzer/constants.py` | Buckets de tamanho de ordem atualizados. | `tests/unit/test_flow_analyzer_metrics.py` | Trades de teste usavam 1.5 BTC para testar contagem de whale; atualizado para 2.5 BTC (whale) e 0.1 BTC (retail). |
| `trading/alert_engine.py` | Gate duplo para `VOLUME_SPIKE`: `ratio >= threshold` E `current_volume >= p95_hourly[hour_utc]`. | `tests/unit/test_volume_spike_dual_gate.py` (Novo) | Validação do gate duplo: volume abaixo do p95 bloqueia o alerta mesmo com ratio alto; volume acima do p95 dispara normalmente. |
| `orderbook_wrapper.py` & `data_handler.py` | Snapshot síncrono com timeout 1.5s e schema padronizado (`timestamps`, `source`, `snapshot_offset_ms`). | `tests/unit/test_orderbook_sync_snapshot.py`<br>`tests/unit/test_signal_orderbook_schema.py` (Novos) | Validação de fallback síncrono para cache background e paridade de schema entre sinais e triggers. |

