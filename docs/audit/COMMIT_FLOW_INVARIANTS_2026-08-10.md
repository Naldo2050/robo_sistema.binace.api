# COMMIT — Correção de Invariantes do Flow Analyzer

## Mensagem sugerida (uma linha)

```
fix(flow): align closed-window order_flow metrics with num_trades/net_flow_1m/imbalance; open-ended whale bucket no longer drops qty>=9999 trades (net_flow_1m == buy-sell now holds in every window)
```

## Mensagem detalhada (git commit -m + body)

```
fix(flow): align closed-window order_flow with num_trades/net_flow/imbalance; fix open-ended whale bucket

Problema:
- num_trades contava o buffer (até 15m) em vez da janela usada no order_flow
- net_flow_1m/flow_imbalance eram lidos do RollingAggregate (janela flutuante),
  divergindo de buy_volume - sell_volume e de imbalance_1m (janela fechada)
- _create_snapshot copiava na janela máxima, mas o order_flow era calculado
  na janela da requisição (1m), retornando séries inconsistentes
- trades com qty >= 9999 não casavam NENHUM bucket (whale aberto (1.0, None)),
  sumindo do sector_flow e do cvd quando o sector era None

Correção:
- num_trades, net_flow_1m e flow_imbalance passam a ser calculados da mesma
  janela fechada do order_flow (_calc_from_trades)
- _create_snapshot copia na janela máxima configurada (max_window_min);
  janelas maiores que o buffer usam os trades disponíveis, com
  data_quality.flow_trades_count refletindo a cobertura real
- classificação de buckets trata maxv is None como bucket aberto
  (qty >= minv) e exige qty >= minv (qty=0.0 não casa retail)

Verificação:
- 19 testes novos de regressão (tests/unit/test_flow_consistency_regression.py)
- Suíte completa: 1622 passed, 3 skipped (baseline 1603)
- Replay sintético: net_flow_1m == buy-sell e flow_imbalance == imbalance_1m
  em 6/6 janelas; sem regressão de performance (5.74 ms/trade vs 5.73)
```

## Arquivos do commit

```
flow_analyzer/core.py
flow_analyzer/constants.py
tests/unit/test_flow_consistency_regression.py
docs/audit/RELATORIO_FLOW_INVARIANTS_2026-08-10.md
```
