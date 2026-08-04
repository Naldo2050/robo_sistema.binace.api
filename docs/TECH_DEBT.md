# Tech Debt

## Audit de `except` silenciosos no código de produção

- **Data:** 2026-08-04
- **Status:** aberto
- **Contexto:** `_update_histories` em `market_orchestrator/market_orchestrator.py:1768` usava
  `except Exception: pass`, mascarando silenciosamente qualquer erro real ao registrar OHLC
  (mesmo padrão de "falha que não aparece em log nenhum" que a auditoria da IA encontrou).
  Essa ocorrência específica já foi corrigida (agora loga `{e!r}` em nível WARNING).
- **Escopo restante:** auditoria geral de excepts silenciosos/degradação oculta em produção:
  - `market_orchestrator/market_orchestrator.py`: 45+ `except Exception:` baleados (sem log).
  - `ai_analyzer_qwen.py`, `fetchers/*`, `events/*`, `monitoring/*`, `ml/*`: dezenas de
    ocorrências (`except Exception:` com `pass` ou corpo vazio).
  - Critério proposto: cada `except Exception:` deve logar o tipo/contexto da exceção
    (pelo menos `{e!r}`), exceto onde o silêncio é deliberado e documentado.
  - Atenção: `market_orchestrator.py:1753` loga em DEBUG — invisível em produção;
    avaliar se volatilidade deveria ser WARNING.
- **Como medir:** `rg -U "except[^:]*:\s*\n\s*pass"` nos pacotes de produção.

## "Alerta" de evento sem handler: métrica instrumentada vs. alerta operacional

- **Data:** 2026-08-04
- **Status:** aguardando decisão de produto (não é bug)
- **Contexto:** o item da auditoria pedia "criar alerta quando um evento é publicado sem
  nenhum handler". O commit `ec62b6d` entregou: métrica
  `trading_event_bus_events_without_handler_total{event_type}` + WARNING no log. Isso é
  "dado exposto para configurar regra depois", não um alerta operacional ativo.
- **Infra disponível no projeto:**
  - Prometheus: `main.py:280-285` serve `/metrics` no registry default, mas **nenhum
    arquivo de regras de alerta** (`*.rules`, `prometheus*.yml`) existe no repo.
  - AlertManager próprio: `trading/alert_manager.py` (enum `AlertType`, callbacks de
    notificação via `add_notification_callback`), porém `integrate_with_alert_manager()`
    (`monitoring/metrics_collector.py:627`) **nunca é chamado** no código de produção, e
    não há tipo `EVENT_NO_HANDLER` no enum.
- **Opções (decisão de produto):**
  1. Fechar como instrumentado (estado atual) e configurar regra no Prometheus externo.
  2. Wiring interno: novo `AlertType.EVENT_NO_HANDLER` + threshold + chamada de
     `integrate_with_alert_manager` na inicialização do bot.
  3. Criar arquivo de alerting rule (`prometheus/rules.yml`) para a métrica.

## Dashboard mínimo do EventBus (Grafana ou equivalente)

- **Data:** 2026-08-04
- **Status:** backlog
- **Escopo:** painel cruzando `trading_event_bus_events_delivered_total` vs
  `trading_event_bus_events_without_handler_total` (e `handler_errors_total`) por
  `event_type`; indicador de entrega "saudável" por minuto e alerta visual quando
  `without_handler` cresce. Sem isso, os dados expostos em `/metrics` ficam sem
  visualização (meio caminho andado operacionalmente).
