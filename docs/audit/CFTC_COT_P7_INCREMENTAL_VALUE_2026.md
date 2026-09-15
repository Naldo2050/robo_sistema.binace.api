# CFTC/CME COT — P7 Incremental Value (2026-09-15)

## 1. Estado da evidência

- Shadow: N=1 coleta × 2 símbolos (BTC/ETH AVAILABLE, P4). Histórico point-in-time próprio: 0 semanas.
- Histórico público disponível para backtest futuro: BTC desde 2018-04-10 (~440 semanas TFF), ETH desde 2021-04-06 — coleta histórica ainda não executada (requer ingestão paginada + first_seen sintético via calendário conservador).
- Payload: seção `cftc` ≤132 bytes no caso típico (medido: compact 1099→1308 chars com pos+cftc em evento mínimo de teste; delta atribuível ao `cftc` ≈ 130-210 chars). Sem bloat.

## 2. Métricas derivadas — posição

Defensáveis quando houver histórico: net por categoria, WoW net change, share do OI, ΔOI (implementados como brutos em `CftcCot.analyze`). **Não implementados**: percentis 26/52/156, COT Index, divergência preço×posicionamento (exigem min_periods + semanas sem forward-fill através de período desconhecido — regra P2 §5).

## 3. Ablação A/B/C/D

- A (baseline) e B (Binance positioning): operacionais há meses.
- C (CFTC) e D (combinado): pendentes de ≥26 semanas de shadow contínuo. Com ~52 obs/ano, qualquer teste de retorno/Sharpe/win-rate hoje seria ruído — **não executado por disciplina, não por omissão**.

## 4. Classificação

**INSUFFICIENT_SAMPLE** (com teto **KEEP_CONTEXT_ONLY**). Sem evidência de valor incremental ou redundância com Binance positioning neste estágio. Retornar a esta fase após 26+ semanas de `cftc_cot_shadow_dataset` contínuo + ingestão histórica point-in-time.

Resultado `PROMOTE_TO_TRADING_SIGNAL`: **explicitamente não permitido** sem fase futura específica de validação.
