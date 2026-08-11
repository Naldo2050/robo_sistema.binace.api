# ETAPA 6 — Auditoria: Proveniência Macro / Freshness / NaN / FRED / yfinance / Cache

Data: 2026-08-11 · Escopo: documentação/classificação + patch 1+2+3 (aprovado)
Sem commit (working tree mantido para revisão).

---

## FASE 1 — Baseline

- HEAD limpo; suíte completa pré-patch: **1762 passed / 3 skipped / 0 failed**, coverage 46.53%.
- Instrumento: `python -m pytest --tb=line -p no:warnings --no-header -q -rs -rf -o timeout=300`.

## FASE 2 — Tickers e proveniência (FASE 3/12 da ETAPA)

| Campo | Provider | Ticker | TTL | Notas |
|---|---|---|---|---|
| VIX | Yahoo | `^VIX` (5d) | 60s (yf 900s) | vix_current=15.16 OK |
| Treasury 10Y | TwelveData→Yahoo | `TNX`→`^TNX` | 300s | us10y_yield=4.688 OK; FRED **não** alimenta provider |
| DXY | Yahoo (fonte de verdade) | `DX-Y.NYB` | 600s | dxy_return_5d/20d = −0.1211/−1.1601 |
| SP500 / Gold | TwelveData (`^GSPC`/`XAU/USD`) | — | 600s | **None sem chave API** → `float("nan")` no ml_features |
| Oil | Yahoo | `CL=F` (clamp 10–250) | 60s | oil_price=80.64 OK |
| BTC dominance | CoinGecko `/global` → fallback Binance volume share | — | 120s | **16.77 = volume share Binance**, não CoinGecko |
| ETH dominance | Binance volume share | — | 120s | 7.49 |
| 2Y / spread / USDT dominance | **nunca buscados** | — | — | us2y*/usdt_dominance = NaN |

- `all_macro`: TTL 900s, 8 fetches paralelos, timeout 8s (`macro_data_provider.py:957`).
- Gate `_prefetch_market_data` = `ENABLE_ALPHAVANTAGE=True` (settings.py:119).

## FASE 3 — Matriz de causas por campo (evento J1-J4, 2026-08-10 ~14:16–14:19 UTC)

| Campo | Valor observado | Causa | Status pós-patch |
|---|---|---|---|
| vix_current / us10y_yield / gold_price / oil_price | 15.16 / 4.688 / 4329.24 / 80.64 | dados reais de provider | inalterado |
| btc_dominance | 16.772727 | fallback volume share Binance | inalterado |
| btc_dominance_change_7d | 0.0 | **fabricado** (cross_asset_correlations.py:657) | 0.0 preservado (não é non-finite; fora do escopo de sanitização) |
| us10y_change_1d / us2y_yield / us2y_change_1d / usdt_dominance | NaN | `None` hardcoded no provider (:1012-1020) → default `float("nan")` (ml_features :566/:574) | **NaN → null** na persistência e fora do payload IA |
| vix/gold/oil_change_1d, btc_vix/gold/oil/yields_corr_30d | null | placeholders `None` hardcoded (:580/:617/:622/:633/:646/:650/:654) | null preservado |
| corr 7d/30d/90d (eth/dxy/ndx) | 0.8261/0.8602/0.0922/0.1075/−0.334 | correlações reais | inalterado |
| stability/inverse_strength/dxy_momentum | 0.0153/0.09985/−1.03896 | proxies derivados (|30d−90d|, |média|, 20d−5d %) | inalterado (semântica documentada) |
| macro_regime / correlation_regime | TRANSITION / DECORRELATED | regime cross-asset | inalterado |

### Inconsistência de unidade (não é bug runtime)

- `build_cross_asset_context` (:712-744) compara `dxy_return_5d > 0.005` como decimal,
  mas produção emite **%** (−0.12). Módulo está **fora de produção** (consumidor produtivo
  único = `get_all_correlations` via ml_features.py:511). Dívida de atenção registrada.

## FASE 4 — Cadeia de sanitização (FASE 8)

- **Não existe sanitizador canônico antes desta ETAPA.** `_clean_event_data`/`_is_nan_or_inf`
  (event_saver) só atuam no **visual log** (`_prepare_visual_event`); `fix_optimization.clean_event`
  só remove campos; `buffer.py:332-334` é checksum/dedup. `event_store.save_batch`/`save_event`
  persistiam `NaN/Infinity` sem sanitização → payloads JSON não-RFC 8259 em `events.payload`.

## FASE 5 — Bug FRED (confirmado)

- `dados/fred_cache.json`: `ts` epoch UTC 14:25:46 vs `updated` naive local UTC−3 11:25:46
  (mesmo instante, formatos divergentes). Causa: `datetime.now()` (naive local) em
  `fred_fetcher._set_disk_cache`. `updated` não tem consumidor em produção (informativo).
- Cache FRED: memória 300s, disco 86400s, failure 3600s.

## FASE 6 — Bugs confirmados (patch 1+2+3)

| Bug | Severidade | Impacto |
|---|---|---|
| 1. FRED `updated` naive (sem timezone) | Baixa | ambigüidade de instante em cache de 24h; interpretação dependente de fuso |
| 2. Persistência JSON não-RFC 8259 (NaN/±Inf) | **Alta** | `events.payload` (SQLite) e jsonl podem conter literais inválidos; parsers estritos quebram; análises consomem lixo |
| 3. NaN pode chegar ao LLM | **Alta** | `_safe_price` deixava NaN passar (NaN é truthy); `eth7`/`dxy30` idem; guardrail sem non-finite → risco de preço 0/NaN na decisão |

## FASE 7 — Patch 1+2+3 (implementado)

1. **FRED**: `_set_disk_cache` → `datetime.now(timezone.utc)`; `ts` epoch UTC, `updated` ISO-8601 `+00:00`; leitura compatível com registros antigos naive (via `ts`).
2. **Sanitizador canônico** novo `common/json_safe.py`: `sanitize_json_safe` (recursivo, NaN/±Inf→`null`, `None`/`0` preservados, não muta entrada, não arredonda), `is_non_finite_number` (float/np.floating), `json_dumps_rfc8259` (`allow_nan=False`). Aplicado no menor chokepoint: `event_store.save_event`/`save_batch` (SQLite) e `event_saver` (`_save_to_jsonl`, `_save_fallback` jsonl+json).
3. **AI payload**: `payload_builder_compact._safe_round/_safe_price/_safe_int` e `eth7`/`dxy30` com `math.isfinite` (contrato preservado: `0` continua omitido como antes); `llm_payload_guardrail.ensure_safe_llm_payload` sanitiza na entrada + `logging.debug("GUARDRAIL_NON_FINITE_SANITIZED")`.

Diff total: **5 arquivos, +66/−14** (+2 arquivos novos não contados: `common/json_safe.py`, `tests/unit/test_etapa6_forense_contract.py`).

## FASE 8 — Contratos (tests/unit/test_etapa6_forense_contract.py, 21 testes)

- FRED: `updated` timezone-aware (+00:00), `ts`==instante (tolerância 1s), leitura de cache legado naive, arquivo strict-parsable.
- Sanitizador: NaN/±Inf→None escalares e aninhados; None/0 preservados; não muta entrada; numpy; RFC 8259 (`json.dumps` com `allow_nan=False`); parser estrito (`parse_constant` rejeita literais) — string `"nan"` é legítima.
- EventStore: RFC 8259 no round-trip SQLite; sem literal non-finite no BLOB; valores finitos intactos.
- EventSaver: jsonl (incl. não-ANALYSIS_TRIGGER) e fallback json sanitizados.
- Builder: `_safe_price/_safe_round/_safe_int` non-finite→None; `_build_static_context` omite non-finite (dxy/vix/eth7/dxy30) e preserva finitos (tnx/dxy/eth7/dxy30 com rounding histórico).
- Guardrail: payload final sem non-finite, finitos intocados.
- E2E: evento NaN → persistência `null` → payload IA limpo.
- **21/21 PASSED** (antes e depois do patch; com `--no-cov` devido ao fail-under=10 do addopts).

## FASE 9 — Validação

- Suítes dirigidas: **122/122** (event_saver_jsonl_guardian, volume_profile_etapa4_regression, cross_asset, updated_correlations, macro_cache_validator_fix, yfinance_cache, data_invariants) e **289/289** (tests/payload + test_patch_guardrail + test_sr_etapa5b_contract + test_ai_llm_fallback_flow).
- Suíte completa: **1783 passed / 3 skipped / 0 failed** (baseline 1762/3/0; +21 novos, 0 falhas), coverage 46.53% → 46.69%.
- `git diff --check`: limpo.
- Replay J1-J4 (scripts/diagnostics/replay_etapa6_j1j4.py): **PASS** — SQLite/JSONL RFC 8259, NaN→null, 0.0 fabricado preservado, payload IA sem non-finite.

## FASE 10 — Restrições respeitadas

- Nada fabricado (sem 0/último válido/forward-fill); upstream (providers, cross_asset_correlations, ml_features) **intocado**; bancos históricos não alterados; NaN nunca vira preço 0 no caminho IA; séries/providers/tickers/TTLs/nomes preservados.

## Anexos (scripts de diagnóstico, sem commit)

- `scripts/diagnostics/replay_etapa6_j1j4.py` — replay J1-J4 (matriz FIELD/VALUE/SOURCE/STATUS + persistência RFC 8259 + payload IA).
- `common/json_safe.py` — sanitizador canônico (novo, parte do patch).
