# Flake registrado — test_event_bus_metrics (não corrigido nesta etapa)

**Data:** 2026-08-10
**Etapa:** 3 (auditoria de qualidade de dados) — registrado, NÃO corrigido
**Decisão:** meta em etapa futura dedicada a concorrência/observabilidade

## Teste

- Arquivo: `tests/unit/test_event_bus_metrics.py`
- Função: `test_publish_without_handler_increments_counter_and_warns`
- Padrão: publica evento sem handler → espera worker assíncrono processar
  (polling `bus._queue`) → lê contador Prometheus GLOBAL
  (`trading_event_bus_events_without_handler_total`, labels
  `event_type=metrics_void`) e asserta `== 1`/`== 2` + WARNING em caplog.

## Sintoma

- Falha INTERMITENTE em execuções full-suite (`tests/unit tests/payload
  tests/integration` juntos), sob carga total (coverage sobre ~42k linhas).
- NÃO reproduzível isoladamente: 5/5 sucesso isolado; sucesso também em
  `tests/unit` sozinho, `unit+payload`, `unit+integration` e no baseline
  (HEAD ad5dd57, stash completo).
- Observado 2x em runs full-suite com as mudanças da ETAPA 3 presentes;
  passou na rodada final reconciliada (1676 passed / 3 skipped / 0 failed).
- LACUNA DE EVIDÊNCIA registrada: o texto exato da assertion falhada nunca
  foi capturado (saída truncada nas execuções) — classificação baseada no
  padrão do teste e na reprodutibilidade, não no erro literal.

## Hipótese

- Race entre o dispatch ASSÍNCRONO do worker do EventBus e a leitura do
  contador Prometheus global (e/ou captura do WARNING via caplog) logo após
  o esvaziamento da fila (`_wait_processed` faz polling de `bus._queue`,
  que pode estar vazia antes de o worker incrementar o contador).
- Sob carga total (GIL disputado por 1580+ testes), a janela entre o
  enqueue e o incremento cresce e a assertion é lida antes do incremento.

## Atribuição

- NÃO atribuído a nenhuma mudança das ETAPAS 1, 2 ou 3: módulos
  `events/event_bus.py` e `prometheus_client` não foram tocados nessas
  etapas; o teste falha apenas sob carga, nunca isolado.
- Sem mudanças nesta auditoria; sem alterar o teste nesta etapa.

## Meta futura

- Retomar em etapa dedicada a concorrência/observabilidade: tornar a
  assertion determinística (sincronização com o worker ou espera ativa do
  contador com timeout), sem mudar a semântica do teste.
