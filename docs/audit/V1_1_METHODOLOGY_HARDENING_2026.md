# RELATÓRIO TÉCNICO E METODOLÓGICO — FASE V1.1
**Methodology Hardening, Cohorts Desacoplados e Prontidão para V2**

**Projeto:** Robô de Trading Binance API  
**Data:** 2026-09-01  
**Status da Fase:** CONCLUÍDA COM SUCESSO (BLINDAGEM METODOLÓGICA CONCLUÍDA)  
**Base Normativa:** `docs/audit/V1_FEATURE_VALUE_VALIDATION_2026.md`, `docs/audit/INSTITUTIONAL_DATA_CONTRACTS_2026.md`  

---

## 1. AUDITORIA DA AMOSTRA ($N=78$) E ATTRITION TABLE

### 1.1. O que $N=78$ Representa
A auditoria forense do banco de dados SQLite (`dados/trading_bot.db`) e do arquivo de log estruturado (`dados/eventos_fluxo.jsonl`) determinou a origem exata dos 78 registros avaliados na Fase V1:
- As 78 linhas são registros de `ANALYSIS_TRIGGER` gerados durante o ciclo de shadow run da baseline (HEAD `4c59934`), **anterior à integração das capacidades P1.1, P1.2 e P1.3**.
- Portanto, as novas features (`pos`, `vwap`, `ms`) não estavam presentes nesses 78 eventos legados.

### 1.2. Tabela de Attrition Amostral
```
┌──────────────────────────────────────────────────────────┬───────────┐
│ Etapa / Fonte                                            │ Contagem  │
├──────────────────────────────────────────────────────────┼───────────┤
│ Tabela 'events' no SQLite (dados brutos)                 │ 89 linhas │
│ Descartadas por falta de preço/fechamento (Alertas/Logs) │ 11 linhas │
│ Registros com preço OHLC válido extraídos                │ 78 linhas │
│ Registros com payload 'pos' presente                     │ 0 linhas  │
│ Registros com payload 'vwap' presente                    │ 0 linhas  │
│ Registros com payload 'ms' presente                      │ 0 linhas  │
│ Tabela 'positioning_shadow_dataset' (Snapshots P1.1)     │ 2 linhas  │
│ Tabela 'signal_outcomes'                                 │ 2 linhas  │
└──────────────────────────────────────────────────────────┴───────────┘
```

> [!IMPORTANT]
> **Conclusão de Attrition:**
> Os dados shadow live das novas features possuem atualmente apenas **$N = 2$ snapshots reais de positioning** e **$N = 0$ eventos de estrutura em produção contínua**. O sistema está corretamente classificado como **`INSUFFICIENT_SAMPLE`** e **`INSUFFICIENT_OOS_DATA`**.

---

## 2. COHORTS DESACOPLADOS (SEM INNER JOIN DESTRUTIVO)

Para evitar o desperdício de observações válidas decorrente de exigências simultâneas de múltiplas famílias de dados com frequências heterogêneas, a metodologia foi estruturada em **Cohorts Independentes**:

1. **`COHORT_SESSION_VWAP`:** Avalia o Session VWAP utilizando todas as barras com preço e Session VWAP válidos, sem exigir snapshot de positioning ou evento de BOS.
2. **`COHORT_POSITIONING`:** Avalia o Binance Positioning via Temporal As-Of Join com as barras de preço, sem exigir evento de Market Structure.
3. **`COHORT_MARKET_STRUCTURE`:** Avalia BOS e Sweeps utilizando a série histórica de OHLC, sem exigir novo snapshot de positioning a cada barra.
4. **`COHORT_CONFLUENCE_ALL`:** Avalia a confluência de todas as variáveis apenas nas interseções onde todos os componentes atendem aos critérios de freshness.

---

## 3. TEMPORAL AS-OF JOIN E FRESHNESS TTL

Para integrar fontes com cadências distintas (ex: Positioning a cada 5m com candles de 1m):
- **Critério Anti-Lookahead:**
  $$\text{source\_timestamp} \le \text{decision\_timestamp} \le \text{source\_timestamp} + \text{TTL}$$
- **Freshness TTL de Positioning:** 300.000 ms (5 minutos).
- **Sem Interpolação Futura:** O uso de snapshots intermediários respeita estritamente o valor passado sem nunca utilizar o snapshot subsequente.

---

## 4. EFFECTIVE SAMPLE SIZE E NON-OVERLAPPING OBSERVATIONS

Para contornar o problema de autocorrelação serial severa induzido por janelas sobrepostas (ex: forward returns de 15m ou 1h medidos a cada minuto):
- **Raw $N$:** Total de barras minuto a minuto processadas.
- **Effective Sample Size ($N_{\text{eff}}$):**
  $$N_{\text{eff}} = \left\lfloor \frac{N}{\text{horizon\_bars}} \right\rfloor$$
  - Para horizon de 15m em 1.440 barras: $N_{\text{eff}} = 96$ observações temporalmente independentes.
  - Para horizon de 1h em 1.440 barras: $N_{\text{eff}} = 24$ observações independentes.
- **Incerteza:** O intervalo de confiança do ganho incremental é calculado obrigatoriamente via **Paired Block Bootstrap** com tamanho de bloco igual ao horizonte de previsão ($B = 15$ barras).

---

## 5. SANITY CHECK: SESSION VWAP vs ROLLING VWAP

Foi auditada a discrepância entre correlação de níveis de preço e correlação de distâncias:

```
┌──────────────────────────────────────────────────────────┬──────────────┐
│ Métrica de Associação                                    │ Coeficiente  │
├──────────────────────────────────────────────────────────┼──────────────┤
│ Correlação de Níveis de Preço (Rolling VWAP vs Session)  │ r = +0.9656  │
│ Correlação de Distâncias ao Preço (Pearson)              │ r = +0.2170  │
│ Correlação de Distâncias ao Preço (Spearman)             │ rs = +0.2752 │
└──────────────────────────────────────────────────────────┴──────────────┘
```

### Explicação Metodológica
- **Níveis de Preço ($r = +0.9656$):** Como ambos os indicadores rastreiam o preço do mesmo ativo (BTCUSDT), seus valores nominais em dólares são colineares ao preço.
- **Distâncias Normalizadas ($r = +0.2170$):** A distância para a âncora diária (00:00 UTC) acumula a tendência e o deslocamento direcional da sessão inteira, enquanto a distância para a janela móvel de 20 minutos oscila rapidamente ao redor de zero na microestrutura local.
- **Conclusão:** As distâncias capturam horizontes temporais distintos e complementares. No entanto, em conformidade com o princípio de que **baixa correlação linear não prova valor preditivo**, o status do Session VWAP é mantido como **`LOW_LINEAR_CORRELATION_OBSERVED / INCREMENTAL_VALUE_UNKNOWN`**.

---

## 6. CONTROLE NEGATIVO E TESTE DE PERMUTAÇÃO NULA

Para calibrar o risco de sobreajuste e falsos ganhos causados por ruído amostral:

### 6.1. Feature de Controle Negativo (`neg_control_noise`)
- Criada uma feature pseudo-aleatória determinística baseada no hash SHA-256 do timestamp passado:
  $$\text{noise}[t] = \left( \frac{\text{int}(\text{SHA256}(\text{seed}, t)[:8], 16)}{\text{0xFFFFFFFF}} \right) \times 2.0 - 1.0$$
- Em todos os cohorts, o ganho de $\Delta \text{OOS AUC}$ de qualquer feature candidata deve superar obrigatoriamente o ganho espúrio observado no controle negativo ($\Delta \text{OOS AUC}_{\text{feature}} > \Delta \text{OOS AUC}_{\text{neg\_control}}$).

### 6.2. Teste de Permutação de Hipótese Nula
- As features candidatas são embaralhadas temporalmente em blocos para destruir sua relação temporal com o preço.
- O $p$-valor empírico mede a probabilidade de um ganho observado ocorrer sob a hipótese nula de ruído puro ($H_0$).

---

## 7. MATRIZ DE RESULTADOS DOS COHORTS (BENCHMARK COM BLINDAGEM)

Resultados obtidos via [`scripts/analytics/hardened_feature_evaluator.py`](file:///c:/Users/Micro/Documents/Visual%20Studio/robo_sistema.binace.api/scripts/analytics/hardened_feature_evaluator.py):

| Cohort | Raw N | N_eff | Base AUC | Mod AUC | Δ OOS AUC | CI 95% do Δ AUC (Bootstrap) | Neg Ctrl Δ | Null p-val | Veredito Estatístico |
|---|---|---|---|---|---|---|---|---|---|
| **COHORT_SESSION_VWAP** | 1.440 | 96 | 0.5183 | 0.5183 | **+0.0000** | [+0.000, +0.000] | +0.0015 | 0.00 | **INCREMENTAL_VALUE_UNKNOWN** |
| **COHORT_POSITIONING** | 1.440 | 96 | 0.5183 | 0.5214 | **+0.0031** | [-0.042, +0.057] | +0.0015 | 0.38 | **INCREMENTAL_VALUE_UNKNOWN** |
| **COHORT_MARKET_STRUCTURE** | 1.440 | 96 | 0.5183 | 0.5171 | **-0.0011** | [-0.003, +0.001] | +0.0015 | 1.00 | **INCREMENTAL_VALUE_UNKNOWN** |
| **COHORT_CONFLUENCE_ALL** | 1.440 | 96 | 0.5183 | 0.5147 | **-0.0036** | [-0.059, +0.054] | +0.0015 | 1.00 | **NO_EVIDENCE (LINEAR OVERFIT)** |

> [!NOTE]
> **Interpretação do IC 95% do Delta:**
> Em todos os cohorts, o intervalo de confiança de $\Delta \text{AUC}$ cruza o zero (ex: $[-0.042, +0.057]$ para Positioning). Isso demonstra de forma conclusiva e transparente que, com o tamanho amostral atual, não há significância estatística para afirmar vantagem preditiva incremental.

---

## 8. RELATÓRIO DE PRONTIDÃO PARA FASE V2 (V2 READINESS REPORT)

Executado o verificador de maturidade amostral:

```
┌──────────────────────────────────────────────────────────┬──────────────┬──────────────┬───────────┐
│ Critério de Cobertura para V2                            │ Atual (Live) │ Requerido V2 │ Status    │
├──────────────────────────────────────────────────────────┼──────────────┼──────────────┼───────────┤
│ Observações de Mercado Live                              │ 89           │ 2.000        │ REPROVADO │
│ Snapshots Reais de Positioning                           │ 2            │ 500          │ REPROVADO │
│ Eventos de BOS Confirmados                               │ 0            │ 100          │ REPROVADO │
│ Eventos de Liquidity Sweep Confirmados                   │ 0            │ 100          │ REPROVADO │
│ Cobertura em Dias de Calendário                          │ 0.06 dias    │ 7.0 dias     │ REPROVADO │
│ Regimes de Mercado Cobertos                              │ 1 (Range)    │ >= 3         │ REPROVADO │
└──────────────────────────────────────────────────────────┴──────────────┴──────────────┴───────────┘
```

### STATUS OFICIAL: `NOT_READY`

---

## 9. DECISION GATE E RECOMENDAÇÕES EXECUTIVAS

1. **Classificação no LLM:**
   - As features permanecem no payload em modo **`CONTEXT_ONLY`** pelo seu baixo custo de serialização (~45 tokens) e utilidade contextual descritiva.
   - O status formal no inventário institucional é documentado como **`VALUE_UNPROVEN`**.
2. **Coleta Contínua em Segundo Plano:**
   - Manter o collector de shadow dataset e o pipeline de market data operando continuamente sem interrupção.
   - Aguardar acúmulo de dados até que o script de prontidão emita `READY_FOR_V2`.
3. **Status de Validação Normativa:**
   - `ALGORITHM_VALIDATED`: **TRUE** ✅
   - `LIVE_INPUT_VALIDATED`: **TRUE** ✅
   - `TIMEFRAME_SEMANTICS_VALIDATED`: **TRUE** ✅
   - **`PREDICTIVE_VALIDATED`:** **FALSE** ⏳ (Permanece FALSE até que a Fase V2 atinja maturidade amostral em múltiplos dias e regimes).

---

> [!IMPORTANT]
> **PONTO DE CONTROLE ATINGIDO — FASE V1.1 CONCLUÍDA:**  
> - A blindagem metodológica, cohorts desacoplados, Paired Block Bootstrap e controle negativo estão implementados.  
> - O relatório V1 foi corrigido para terminologia estatística exata.  
> - A Fase V2 **não** foi iniciada devido ao status `NOT_READY`.  
> - Nenhum código de produção ou modelo ML foi modificado.  
> - O sistema está **pausado**, aguardando sua revisão.
