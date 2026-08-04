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
