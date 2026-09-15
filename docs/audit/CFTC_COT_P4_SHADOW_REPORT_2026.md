# CFTC/CME COT — P4 Shadow Report (2026-09-15)

Coleta shadow sem payload, sem sinal, sem execução. Tabela: `cftc_cot_shadow_dataset` em `dados/trading_bot.db` (gitignored).

## 1. Observações reais

| collected_at (UTC) | symbol | code | asof | status | OI | rev | latency |
|---|---|---|---|---|---|---|---|
| 2026-09-15T01:56:12Z | BTCUSDT | 133741 | 2026-09-08 | AVAILABLE | 21083 | 0 | 0.78s |
| 2026-09-15T01:56:12Z | ETHUSDT | 146021 | 2026-09-08 | AVAILABLE | 26564 | 0 | 2.02s |

Nota de higiene: duas linhas anteriores (coleta de debug com bug de ordenação ASC que retornou 2018/2021) foram removidas do dataset; o bug (`fetch_latest` sem DESC) foi corrigido e coberto por coleta válida acima.

## 2. Métricas (janela: 1 coleta × 2 símbolos)

- coverage_rate (símbolos com linha oficial): 2/2 = 1.0
- available_rate: 2/2 = 1.0
- partial_rate / stale_rate / revision_rate: 0.0
- fetch_error_rate / schema_error_rate: 0.0
- latência publicação→observação: relatório de 2026-09-08 (sexta 2026-09-11 15:30 ET) observado 2026-09-15T01:56Z ≈ 3.4 dias (coleta manual, não contínua — latência operacional real será medida com o updater em produção).
- schema drift: nenhum (87 colunas TFF estáveis; todos os campos P2 presentes).
- missing fields: nenhum nos 2 contratos.

## 3. Limitações honestas

- **N=1 coleta**: múltiplas publicações reais exigem semanas. O gate P4 é **condicional**: schema + first_seen + versionamento provados por teste (P3/P5); estabilidade multissemana pendente e delegada ao updater contínuo + re-coletas antes de P6 em produção.
- Micro contratos (`133742/146022`) não coletados nesta rodada (fora do roteamento atual do bot); mapa pronto.
- Revisão real ainda não observada (CFTC declara não republicar histórico — P1 §16); versionamento exercitado com fixture `tff_btc_revision.json` (P3, revision 1 preserva first_seen).

## 4. GATE P4

- schema instável? Não (0 drift).
- first_seen confiável? Sim (raw cache append-only + `first_seen_at` preservado em revisão).
- contrato ambíguo? Não (P2 fechado).
- revisões versionadas? Sim (hash + revision, teste P3 verde).
- cobertura insuficiente? Para BTC/ETH CME: suficiente (2/2). Para universo total: por desenho só CME cripto.

**GATE P4: PASS (condicional a re-coletas semanais antes de P6 produtivo).**
