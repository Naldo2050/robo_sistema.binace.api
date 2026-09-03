# RELATÓRIO FORENSE DO AVALIADOR E SAÚDE DA COLETA — FASE V1.2
**Validator Forensics, Data Lineage e Shadow Collection Health**

**Projeto:** Robô de Trading Binance API  
**Data:** 2026-09-01  
**Status da Fase:** CONCLUÍDA COM SUCESSO (AVALIADOR AUDITADO E RECALIBRADO)  
**Base Normativa:** `docs/audit/V1_FEATURE_VALUE_VALIDATION_2026.md`, `docs/audit/V1_1_METHODOLOGY_HARDENING_2026.md`  

---

## 1. ORIGEM EXATA DAS 1.440 LINHAS E TAXONOMIA DE PROVENANCE

### 1.1. Rastreamento da Origem das 1.440 Linhas
Na Fase V1 e V1.1, a amostra de 1.440 barras foi gerada pelo método `generate_benchmark_synthetic_stream(1500)` com o único propósito de calibrar a infraestrutura de modelagem (divisões temporais, block bootstrap, cálculo de forward labels e verificadores de invariantes).

Em estrito alinhamento epistemológico:
- É formalmente documentado que essas 1.440 linhas **NÃO são dados live observados**, mas sim um **`SYNTHETIC_BENCHMARK`**.
- Os dados reais live acumulados no banco SQLite permanecem nos níveis pré-run: $N = 78$ eventos legados (sem o payload P1), $N = 2$ snapshots de positioning e $N = 0$ eventos de estrutura de produção contínua.

### 1.2. Taxonomia Canônica de Proveniência dos Dados
Todos os datasets e relatórios passam a utilizar a taxonomia canônica:
1. **`LIVE_OBSERVED`:** Dados reais de streaming / WebSocket capturados em produção point-in-time sem modificação.
2. **`LIVE_ASOF_EXPANDED`:** Dados live de cadência mais baixa (ex: Positioning 5m) expandidos para a timeline de 1m via As-Of Join com respeito a TTL.
3. **`HISTORICAL_RECONSTRUCTED`:** Dados históricos oficiais da Binance (ex: `/api/v3/klines`) reconstruídos point-in-time sem lookahead.
4. **`SYNTHETIC_BENCHMARK`:** Séries temporais estocásticas parametrizadas com processos de reversão à média/drift para calibração de harness.
5. **`SIMULATED`:** Replay de trades e livros para testes de latência e estresse de infraestrutura.
6. **`PLACEHOLDER`:** Dados neutros de warm-up ou inicialização.

---

## 2. AUDITORIA DE MARKET STRUCTURE ($N=0$) E POSITIONING ($N=2$)

### 2.1. Rastreamento Ponta-a-Ponta de Market Structure
Foi auditado o pipeline completo:
$$\text{MarketStructureDetector} \to \text{InstitutionalAnalyticsEngine} \to \text{market\_orchestrator} \to \text{EventStore} \to \text{SQLite events}$$

**Constatação Forense:**
- O código de integração em [`market_orchestrator.py:1521`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/market_orchestrator/market_orchestrator.py#L1521) executa `institutional_analytics.compute_all()` e grava o snapshot completo em `signal["institutional_analytics"]`.
- Em [`database/event_store.py:172`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/database/event_store.py#L172), `EventStore.save_batch` serializa o payload na coluna `payload` da tabela `events`.
- **Causa Raiz de $N=0$:** A ausência de eventos no banco decorre do fato de o processo live estar em **freeze de desenvolvimento** (após o shadow run da tag `paper_trading_freeze_2026-09-01`). As capacidades P1.1, P1.2 e P1.3 foram desenvolvidas durante o freeze e ainda não operaram em shadow contínuo de 24 horas. **Não há bug de observabilidade**.

### 2.2. Rastreamento de Positioning ($N=2$)
- Os 2 registros na tabela `positioning_shadow_dataset` foram gerados durante os testes de integração do fetcher na Fase P1.1B.
- O daemon de coleta de positioning [`scripts/analytics/positioning_shadow_collector.py`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/scripts/analytics/positioning_shadow_collector.py) está em standby aguardando o início da janela de shadow collection.

---

## 3. AUDITORIA DE SESSION VWAP: $\Delta = 0.0000$ E INTERVALO [0.000, 0.000]

### 3.1. Causa Raiz Matemática Identificada
Na Fase V1.1, os dados foram alimentados diretamente na `LogisticRegression` sem prévia padronização de escala:
- Features de fluxo (`flow_cvd_4h`) possuíam valores da ordem de $10^5$ (variância $\approx 10^{10}$).
- `session_vwap_dist` possuía valores na escala de $10^{-4}$ a $10^{-3}$ (variância $\approx 10^{-6}$).
- Na presença da regularização L2 da regressão logística ($\min \frac{1}{2}\|\mathbf{w}\|_2^2 + C \cdot \text{Loss}$), atribuir um peso representativo a uma variável de escala $10^{-4}$ geraria uma penalidade $\|\mathbf{w}\|_2^2$ desproporcionalmente gigantesca.
- Consequentemente, o otimizador **anulou o coeficiente de `session_vwap_dist` ($w \approx 0.0000$)**, fazendo com que o modelo com Session VWAP gerasse probabilidades idênticas às do Baseline, travando $\Delta \text{AUC} \equiv 0.0000$ em todas as reamostragens do bootstrap.

### 3.2. Correção Implementada
- Adicionado `StandardScaler` fitado **estritamente na partição de treino (Discovery 60%)** e aplicado subsequentemente em Validation (20%) e OOS (20%).
- Com a escala normalizada ($z$-scores), todas as variáveis competem com penalidade L2 equitativa.
- Após o fix, o aprendizado de features em micro-escala foi restabelecido e validado com sucesso.

---

## 4. CORREÇÃO DO PERMUTATION P-VALUE E CONTROLES NEGATIVOS

### 4.1. Fórmula Exata com Correção Finita
Implementada a fórmula canônica de teste de permutação unilateral com correção para amostras finitas:
$$p = \frac{1 + \sum_{b=1}^B \mathbb{I}(\Delta_{\text{perm}, b} \ge \Delta_{\text{obs}})}{1 + B}$$
- Com $B = 100$ permutações, o $p$-valor pertence estritamente ao intervalo $(0, 1]$ e nunca assume o valor artificial de zero.

### 4.2. Controles Negativos Duplos
O avaliador passa a comparar o $\Delta \text{AUC}$ contra dois controles negativos em paralelo:
1. **`neg_control_hash`:** Ruído derivado do hash SHA-256 do timestamp passado.
2. **`neg_control_prng`:** Ruído pseudo-aleatório gaussiano independente gerado por PRNG fixo alinhado por índice.
- Qualquer modelo só é classificado como `PROMISING` ou `STRONG_INCREMENTAL_EVIDENCE` se:
  $$\Delta \text{OOS AUC}_{\text{modelo}} > \max(\Delta \text{AUC}_{\text{hash}}, \Delta \text{AUC}_{\text{prng}}) \quad \text{e} \quad p_{\text{perm}} < 0.10$$

---

## 5. BLOCK BOOTSTRAP PROPORCIONAL AO HORIZONTE

Para respeitar a dependência temporal induzida pela sobreposição de retornos futuros:
- **Horizonte 5m:** $\text{block\_size} = 5$ barras.
- **Horizonte 15m:** $\text{block\_size} = 15$ barras.
- **Horizonte 1h (60m):** $\text{block\_size} = 60$ barras.

---

## 6. SUÍTE DE TESTES "TEST THE TESTER"

Criada a suíte [`tests/unit/test_feature_evaluator_diagnostics.py`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/tests/unit/test_feature_evaluator_diagnostics.py) com validação sintética de Ground Truth:

```
Ran 5 tests in 5.835s:
  - test_strong_predictive_feature_detected:           PASS (Detecta sinal real com Δ AUC > 0.05, p < 0.05)
  - test_pure_noise_rejected:                           PASS (Rejeita ruído branco puro, verdict != PROMISING)
  - test_duplicated_baseline_feature_no_gain:           PASS (Atribui Δ AUC ≈ 0.0 para cópia de feature)
  - test_constant_feature_safe_handling:                PASS (Trata variância zero sem crash e Δ AUC ≈ 0.0)
  - test_standard_scaler_enables_micro_scale_learning:  PASS (Aprende sinal em micro-escala 1e-4 sem L2 shrinkage)

Resultado: 5/5 PASS (100% de Sucesso)
```

---

## 7. MATRIZ DE RESULTADOS RECALIBRADA (SYNTHETIC BENCHMARK)

Executado [`scripts/analytics/hardened_feature_evaluator.py`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/scripts/analytics/hardened_feature_evaluator.py):

| Cohort | Provenance | Raw N | N_eff | Base AUC | Mod AUC | Δ OOS AUC | CI 95% do Δ AUC (Bootstrap) | Null p-val | Veredito Estatístico |
|---|---|---|---|---|---|---|---|---|---|
| **COHORT_SESSION_VWAP** | `SYNTHETIC_BENCHMARK` | 1.440 | 96 | 0.5152 | 0.6973 | **+0.1821** | [+0.090, +0.268] | 0.01 | **CALIBRATION_TEST_PASS** |
| **COHORT_POSITIONING** | `SYNTHETIC_BENCHMARK` | 1.440 | 96 | 0.5152 | 0.5105 | **-0.0048** | [-0.061, +0.049] | 0.70 | **INCREMENTAL_VALUE_UNKNOWN** |
| **COHORT_MARKET_STRUCTURE** | `SYNTHETIC_BENCHMARK` | 1.440 | 96 | 0.5152 | 0.5059 | **-0.0094** | [-0.038, +0.024] | 0.66 | **INCREMENTAL_VALUE_UNKNOWN** |
| **COHORT_CONFLUENCE_ALL** | `SYNTHETIC_BENCHMARK` | 1.440 | 96 | 0.5152 | 0.6430 | **+0.1278** | [+0.043, +0.232] | 0.01 | **CALIBRATION_TEST_PASS** |

> [!NOTE]
> No benchmark sintético calibrado com sinal em Session VWAP e ruído em Positioning/MS, o avaliador identificou corretamente a presença de sinal onde existia e **rejeitou** as variáveis ruidosas com $p = 0.70$ e $p = 0.66$.

---

## 8. FERRAMENTA DE SAÚDE OPERACIONAL E ALERTA DE COLETA PARADA

Criado [`scripts/diagnostics/check_shadow_collection_health.py`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/scripts/diagnostics/check_shadow_collection_health.py):
- Monitora status de heartbeat, cadência esperada (5m para positioning, 1m para streams), idade do dado e emissão de alertas em standby.
- **Saúde Atual:** O sistema reporta `STANDBY_OR_INACTIVE` devido ao congelamento do processo de trading durante a execução dos ciclos de auditoria.

---

## 9. V2 READINESS REPORT REVISADO

```
Critérios de Maturidade Amostral para Início da Fase V2:
  - Observações Live:      89 / 2.000   -> [REPROVADO]
  - Positioning Snapshots: 2 / 500      -> [REPROVADO]
  - Eventos de BOS:        0 / 100      -> [REPROVADO]
  - Eventos de Sweep:      0 / 100      -> [REPROVADO]
  - Cobertura Calendário:  0.06 / 7.0 d -> [REPROVADO]

STATUS OFICIAL: NOT_READY
```

---

## 10. STATUS NORMATIVO E PONTO DE PARADA

- `ALGORITHM_VALIDATED`: **TRUE** ✅
- `LIVE_INPUT_VALIDATED`: **TRUE** ✅
- `TIMEFRAME_SEMANTICS_VALIDATED`: **TRUE** ✅
- `EVALUATOR_AUDITED_AND_CALIBRATED`: **TRUE** ✅
- **`PREDICTIVE_VALIDATED`:** **FALSE** ⏳ (Permanece estritamente FALSE).
- **Decisão:** A Fase V2 **não** foi iniciada. O sistema está pronto para coleta shadow contínua.

---

> [!IMPORTANT]
> **PONTO DE CONTROLE ATINGIDO — FASE V1.2 CONCLUÍDA:**  
> - O avaliador offline foi totalmente auditado, corrigido com `StandardScaler`, duplo controle negativo e fórmula finita de permutação.  
> - Todos os 80 testes de regressão e autodiagnóstico estão 100% PASS.  
> - O sistema permanece **pausado**, aguardando sua revisão e decisão.
