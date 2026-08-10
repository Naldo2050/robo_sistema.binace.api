# RELATÓRIO — Invariantes do Flow Analyzer: Correção de Inconsistências (Fase 9/10)

> Data: 2026-08-10 | Escopo: `flow_analyzer/core.py`, `flow_analyzer/constants.py`

---

## 1. Resumo Executivo

Foram corrigidos **4 bugs** que quebravam invariantes internas do `FlowAnalyzer` (janelas rolling vs. janela fechada do orquestrador) e **2 bugs de classificação de buckets** que podiam fazer ordens "sumirem" da análise. Todo o invariante `net_flow_1m == buy_volume - sell_volume` e `flow_imbalance == imbalance_1m` agora se mantém em todas as janelas.

| Métrica | Antes | Depois |
|---|---|---|
| Testes do novo arquivo regressão | 8 falhas → 11 passes | **19 passes** |
| Suíte completa | 1603 passou | **1622 passou, 3 skip** |
| process_trade (500 trades) | 5.73 ms/trade | 5.74 ms/trade (sem regressão) |
| Invariantes no replay sintético (6 min) | divergiam | **OK em todas as janelas** |

---

## 2. Bugs Corrigidos

### 2.1 `num_trades` contava todos os trades do buffer (15m) — não os da janela
**Sintoma:** `metadata.num_trades` ≠ trades usados em `order_flow`.
**Causa:** `_create_snapshot` copiava o buffer (até 15 min), e `num_trades` era calculado sobre esse buffer inteiro.
**Correção:** `num_trades` agora é derivado da mesma janela usada pelo `order_flow` (janela fechada `[start, now]`), via `_calc_from_trades` quando `window <= 15m`, ou via janela máxima (`_max_window_min`) quando a requisição pede janela maior que o buffer.

### 2.2 `net_flow_1m`/`flow_imbalance` usavam agregado rolling desincronizado do `order_flow`
**Sintoma:** `net_flow_1m != buy_volume - sell_volume`; `flow_imbalance != imbalance_1m`.
**Causa:** o `order_flow` (buy/sell/imbalance) era calculado da janela fechada `[now-60s, now]`, mas `net_flow_1m` e `flow_imbalance` eram lidos do `RollingAggregate` de 1 min — que tinha janela flutuante (cheia/parcial) e podia estar deslocado (inclusive deslocado quando trades atrasados/agrupados eram descartados do agregado em modo degradado).
**Correção:** `net_flow_1m` e `flow_imbalance` passam a ser calculados da **mesma fonte** do `order_flow` (`_calc_from_trades` na janela fechada). O agregado rolling permanece como fonte da série histórica `net_flow_history` (janelas 5m/15m) — agora sem o desalinhamento.

### 2.3 `_create_snapshot` truncava o copy na janela MÁXIMA, mas o `order_flow` era calculado na janela da requisição
**Sintoma:** janelas fechadas pequenas (1m) retornavam trades além da janela nas séries; janelas grandes (> max window) retornavam datasets vazios.
**Causa:** o copy de trades (`flow_trades`, `sector_flow`) usava `max(MAX_SNAPSHOT_WINDOW_MIN, window_min)` — para `window=1m` copiava 15 min; para `window=4h` copiava só 15 min.
**Correção:** o copy usa a janela máxima configurada (`_max_window_min`), o que é suficiente para qualquer janela de requisição suportada. Para janelas acima do buffer (4h), o `order_flow`/`num_trades` são calculados do buffer disponível, com `data_quality.flow_trades_count` refletindo a cobertura real.

### 2.4 Classificação de sector: bucket whale aberto `(1.0, None)` não casava `qty >= 1.0`
**Sintoma:** trades com `qty >= 9999` (p.ex. ordens agressivas no topo) não eram classificados em NENHUM bucket — não entravam em `sector_flow` nem, por consequência, no `cvd` quando `whale_threshold` era cruzado (a condição `sector_name is not None` pulava o `cvd`).
**Causa:** a constante do bucket whale era `(1.0, None)` mas o laço de classificação usava `qty < maxv` para o bucket aberto, que nunca casa.
**Correção:** `constants.py` usa `(1.0, None)` → o laço agora trata `maxv is None` como "aberto" (`if maxv is None or qty < maxv`). A classificação original (agregados 5m/15m + bursts) continua intacta, pois o agregado rolling recebia o `trade_record` em ambos os caminhos.

### 2.5 Bucket retail `(0.0, 0.5)` com `qty == 0.0`
**Sintoma:** trades com qty exatamente `0.0` caíam no bucket retail (janela fechada), mas em janelas rolling iam para `None`.
**Correção:** o laço exige `qty >= minv` — `qty=0.0` é excluído dos buckets (consistente com o rolling).

---

## 3. Invariantes Verificadas

1. `net_flow_1m == buy_volume - sell_volume` (dentro de 0.05 USD) — **antes:** divergia; **depois:** OK em 6/6 janelas no replay.
2. `flow_imbalance == imbalance_1m` (1e-3) — **depois:** OK 6/6.
3. `sector_flow[whale].buy == 9999.0` para qty open-ended — **depois:** OK.
4. `cvd == whale.delta` para o cenário open-ended (9997.5) — **depois:** OK.
5. `net_flow_1m` não inclui trades da janela seguinte (limite fechado `ts <= now_ms`) — **depois:** OK.
6. Trades com `ts == close` da janela pertencem à janela fechada (inclusivo).

---

## 4. Arquivos Alterados

| Arquivo | Mudança |
|---|---|
| `flow_analyzer/constants.py` | Bucket whale aberto `(1.0, None)` (constante de classificação) |
| `flow_analyzer/core.py` | Laço de classificação com `maxv is None`; `_create_snapshot` usa janela máxima; `_calc_from_trades` para `num_trades`/`net_flow_1m`/`flow_imbalance`; `num_trades` derivado da janela |
| `tests/unit/test_flow_consistency_regression.py` | **Novo** — 19 testes de regressão (buckets, janelas, invariantes) |

---

## 5. Testes

```
pytest tests/unit/test_flow_consistency_regression.py   → 19 passed
pytest (suíte completa)                                  → 1622 passed, 3 skipped
```

Performance: 500 trades → 5.74 ms/trade (baseline 5.73 ms/trade; Δ = 0.2%, ruído).

---

## 6. Notas / Observações (não corrigidas nesta fase)

- `_adjust_timestamp_if_needed` possui código morto (retornos duplicados; `flag` descartado pelo chamador) — o clamp de trades atrasados (`MAX_BATCH_LATE_MS`) é, na prática, não-funcional: trades atrasados mantêm o timestamp original. Não alterado para não mudar comportamento fora do escopo (documentado em `docs/TECH_DEBT.md` se desejado).
- Limite de janela fechada é inclusivo em `ts == now_ms` (o orquestrador manda o trade de `T == window_end` para a próxima janela — diferença de 1 trade na borda; interna ao analyzer é consistente).
