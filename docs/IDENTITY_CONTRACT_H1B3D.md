# Contrato de Identidade H1-B3-D — Exaustão v2 e Zona v2

Fase 5.11 · Gate H1-B3-D · Contrato derivado de H1-B3-C-H1 (veredito A).

## 1. Objetivo

Tornar determinística e verificável a identidade (`event_id`) de:

- **Exaustão** — antes gerada com delta fixo `0.0` (v1), agora com o delta
  efetivo da janela;
- **Zona** — antes herdada do evento base (Absorção/Exaustão/trigger),
  agora própria, derivada do snapshot da ocorrência.

Nenhuma identidade histórica persistida foi alterada. Nenhum evento
existente foi reescrito. As fórmulas v2 convivem com a v1 congelada.

## 2. Fórmulas

Exaustão (`exhaustion-v2`):

```
event_id = SHA256("v2|{window_open_ms}|{symbol}|Exaustão|{delta:.8f}")
```

Zona (`zone-v2`):

```
event_id = SHA256("Zona|{symbol}|{window_key}|{kind}|{timeframe}|"
                  "{anchor:.2f}|{low:.2f}|{high:.2f}|{last_touched}")
```

Componentes de Zona:

- `symbol`: `ativo` do evento (ou `symbol`, caso do trigger) — obrigatório;
- `window_key`: `window_id` → `features_window_id` → `str(epoch_ms)` — obrigatório;
- `kind` / `timeframe`: vocabulário do LevelRegistry (POC/VAL/VAH/HVN/LVN/...);
- `anchor` / `low` / `high`: geometria da zona, 2 casas decimais;
- `last_touched`: ISO com milissegundos, definido pelo registry no toque
  (`LevelRegistry.check_price`).

Campos **nunca** usados como componente: `touch_count`, `janela_numero`,
identidade/UUID do evento base, `session_id`.

O `base_event_id` é preenchido na Zona **somente** quando o evento base
possui identidade determinística (SHA-256 hexdigest minúsculo de 64 chars).
UUIDs técnicos de 8 chars do EventSaver não são promovidos.

## 3. Validação e normalização

Implementação pura em `common/identity_contract.py` (sem I/O, sem relógio,
sem UUID, sem mutação de entradas):

- texto: `strip`, não vazio, sem `|` nem caracteres de controle;
- numérico: `Real` finito; `bool` rejeitado;
- inteiro: `Integral` (compatível com numpy int); `bool` rejeitado;
- `f"{x:.2f}"` / `f"{x:.8f}"`: ponto decimal independente de locale
  (comportamento idêntico em Windows/Linux, Python 3.8+);
- qualquer componente inválido ⇒ `IdentityContractError` (fail-closed).
  Erros nunca são convertidos em IDs aleatórios.

Rejeição no produtor = evento **não emitido** (Zona) ou evento `Erro`
sem identidade (Exaustão não finita), sempre com log ERROR explícito.

## 4. Limitações declaradas

1. **Formatação decimal ≠ aritmética monetária decimal exata.**
   `f"{delta:.8f}"` de um float64 não é conversão decimal exata;
   `|delta| < 5e-9` formata como `"0.00000000"` e `-0.00000000` para
   negativos minúsculos. O prefixo `"v2|"` garante que isso nunca
   colida textualmente com a identidade v1 (delta fixo `0.0`).
2. **`last_touched` é derivado do relógio de execução.** A identidade de
   Zona é reproduzível para um snapshot imutável com `last_touched`
   preservado (ex.: replay do payload), mas **não** é necessariamente
   reproduzível se a ocorrência for reconstruída em outro processamento
   (novo toque ⇒ novo `last_touched` ⇒ nova identidade).
3. **Sem idempotência cross-run absoluta.** Deduplicação continua sendo
   por identidade + hash canônico no EventStore; reemissão byte-idêntica
   é deduplicada; payload divergente com mesmo id ⇒ fail-closed
   (`IntegrityCollisionError`).
4. **Preservação forense de rejeitados é parcial.** Eventos de Exaustão
   com delta não finito são rejeitados com log explícito, mas o
   encaminhamento físico a quarentena depende de H1-B2-Q2 (bloqueado).
   Até lá, não há promessa de preservação forense integral.
5. **Determinismo numérico entre ambientes:** somas numpy podem variar
   entre versões de biblioteca (drift de última casa); identidades devem
   ser comparadas a partir do payload persistido, não recalculadas de
   agregados re-derivados.

## 5. Separação de camadas

| Camada | Responsabilidade | Estado em H1-B3-D |
|---|---|---|
| Produtores (`data_handler`, `market_orchestrator`) | compor identidade determinística | alterado (escopo autorizado) |
| `common/identity_contract` | funções puras | novo |
| EventSaver | uuid8 técnico p/ anônimos, poda, JSONL | inalterado (débitos → Q3) |
| EventStore | dedup uuid/hash, fail-closed | inalterado (política → Q2) |
| Quarentena | rota de rejeitados | fora de escopo (Q2) |
| LevelRegistry | estado de toques | inalterado |

## 6. Compatibilidade histórica

- 545 eventos históricos (Absorção) permanecem intocados, com ids v1;
- JSONL/DB: zero reescrita; coerção numérica do espelho JSONL permanece
  (débito documentado → Q3);
- `_mk_event_id` permanece inalterado para a Absorção;
- Absorção não usa identidade v2 nesta fase.

## 7. Testes

- `tests/unit/test_identity_contract_h1b3d.py` — funções puras;
- `tests/unit/test_exhaustion_identity_v2.py` — produtor Exaustão;
- `tests/unit/test_zone_identity_v2.py` — `_check_zone_touches` com stubs;
- `tests/events/test_zone_identity_sqlite_h1b3d.py` — coexistência base+Zona
  e colisões em SQLite exclusivamente em `tmp_path`.

## 8. Não-objetivos

- Não homologa H1-B3; não libera H1-B2-Q2; não altera schema, UNIQUE,
  EventStore/EventSaver/reconciliador/validador/quarentena;
- Não ativa retry financeiro nem recuperação automática;
- Não altera `dados/`, `fallback_events/`, `memory/` nem fixtures.
