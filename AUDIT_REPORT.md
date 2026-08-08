
# AUDIT_REPORT.md — Levantamento do Sistema de Trading


Gerado em 2026-08-07 por script de auditoria. Objetivo: organizar dados brutos para análise externa por IA especializada.


**AVISO DE REDAÇÃO**: chaves/segredos foram substituídos por ***REDACTED***.


# SEÇÃO 1 — PAYLOAD ENVIADO À IA (análise de custo)


## 1.1 logs/last_llm_payload.json (conteúdo completo)


```json
{
  "symbol": "BTCUSDT",
  "price": 76538.3,
  "delta": 64.96,
  "volume": 123.64,
  "signal_type": "Exaustão de Compra",
  "window": 6,
  "tipo_evento": "ANALYSIS_TRIGGER",
  "resultado_da_batalha": "COMPRA",
  "volume_total": 123.64,
  "preco_fechamento": 76538.3,
  "epoch_ms": 1785871613908,
  "_dump_meta": {
    "payload_bytes_final": 273,
    "timestamp_utc": "2026-08-04T19:26:53.917876+00:00",
    "flags": {
      "v2_enabled": true,
      "max_bytes": 6144,
      "guardrail_hard_enabled": true
    }
  }
}
```


## 1.2 logs/payload_metrics.jsonl (últimas 30 linhas)


```jsonl
{"payload_bytes": 246, "leak_blocked": false, "bytes_after": 246, "payload_root_name": "clean_payload"}
{"payload_bytes": 273, "keys_top_level": ["symbol", "price", "delta", "volume", "signal_type", "window", "tipo_evento", "resultado_da_batalha", "volume_total", "preco_fechamento", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1777425161761, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": true, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": true, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": true, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": true, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": true, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": true, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": true, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 145, "leak_blocked": false, "bytes_after": 145, "payload_root_name": "clean_payload"}
{"payload_bytes": 172, "keys_top_level": ["tipo_evento", "ativo", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785857634696, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785857634735, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785857634753, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785857634821, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785857634833, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 145, "leak_blocked": false, "bytes_after": 145, "payload_root_name": "clean_payload"}
{"payload_bytes": 172, "keys_top_level": ["tipo_evento", "ativo", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785857768362, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785857768415, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785857768437, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785857768535, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785857768553, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 134}
{"cache_hit": true, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 135}
{"cache_hit": false, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 145, "leak_blocked": false, "bytes_after": 145, "payload_root_name": "clean_payload"}
{"payload_bytes": 172, "keys_top_level": ["tipo_evento", "ativo", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785857922414, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785857922455, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785857922474, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785857922555, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785857922571, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 288}
{"cache_hit": true, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 288}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 294}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "8873e263cbdd6c3ce6075f4ff823824b4b924d24a1db54e85239076b0e37893a"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "a65c3abcd2b25c0035919f640471c2c36be3e5a89932103e8a76453704b59d8f"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 91, "leak_blocked": false, "bytes_after": 91, "payload_root_name": "clean_payload"}
{"payload_bytes": 131, "leak_blocked": true, "bytes_after": 84, "payload_root_name": "event"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 246, "leak_blocked": false, "bytes_after": 246, "payload_root_name": "clean_payload"}
{"payload_bytes": 273, "keys_top_level": ["symbol", "price", "delta", "volume", "signal_type", "window", "tipo_evento", "resultado_da_batalha", "volume_total", "preco_fechamento", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785871613908, "counts": {}}
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "8873e263cbdd6c3ce6075f4ff823824b4b924d24a1db54e85239076b0e37893a"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "a65c3abcd2b25c0035919f640471c2c36be3e5a89932103e8a76453704b59d8f"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 91, "leak_blocked": false, "bytes_after": 91, "payload_root_name": "clean_payload"}
{"payload_bytes": 131, "leak_blocked": true, "bytes_after": 84, "payload_root_name": "event"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 406}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 406}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "8873e263cbdd6c3ce6075f4ff823824b4b924d24a1db54e85239076b0e37893a"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "a65c3abcd2b25c0035919f640471c2c36be3e5a89932103e8a76453704b59d8f"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 91, "leak_blocked": false, "bytes_after": 91, "payload_root_name": "clean_payload"}
{"payload_bytes": 131, "leak_blocked": true, "bytes_after": 84, "payload_root_name": "event"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 454}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 454}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "8873e263cbdd6c3ce6075f4ff823824b4b924d24a1db54e85239076b0e37893a"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "a65c3abcd2b25c0035919f640471c2c36be3e5a89932103e8a76453704b59d8f"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 91, "leak_blocked": false, "bytes_after": 91, "payload_root_name": "clean_payload"}
{"payload_bytes": 131, "leak_blocked": true, "bytes_after": 84, "payload_root_name": "event"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 796}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 796}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "8873e263cbdd6c3ce6075f4ff823824b4b924d24a1db54e85239076b0e37893a"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "a65c3abcd2b25c0035919f640471c2c36be3e5a89932103e8a76453704b59d8f"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 91, "leak_blocked": false, "bytes_after": 91, "payload_root_name": "clean_payload"}
{"payload_bytes": 131, "leak_blocked": true, "bytes_after": 84, "payload_root_name": "event"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 51431, "leak_blocked": true, "bytes_after": 3625, "payload_root_name": "event"}
{"payload_bytes": 3715, "keys_top_level": ["symbol", "epoch_ms", "trigger", "price", "regime", "qual", "flow", "ob", "tf", "sr", "w", "ext", "alerts", "ctx", "ofi", "vwap", "liq", "summary", "tipo_evento", "descricao"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785874980000, "counts": {}}
{"payload_bytes": 52037, "leak_blocked": true, "bytes_after": 3714, "payload_root_name": "event"}
{"payload_bytes": 3804, "keys_top_level": ["symbol", "epoch_ms", "trigger", "price", "regime", "qual", "flow", "ob", "tf", "sr", "w", "ext", "alerts", "ctx", "ofi", "vwap", "iceberg", "liq", "mr", "summary", "tipo_evento", "descricao"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785875280000, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": true, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 145, "leak_blocked": false, "bytes_after": 145, "payload_root_name": "clean_payload"}
{"payload_bytes": 172, "keys_top_level": ["tipo_evento", "ativo", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785881581113, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785881581125, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785881581134, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785881582914, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785881582921, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "8873e263cbdd6c3ce6075f4ff823824b4b924d24a1db54e85239076b0e37893a"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "a65c3abcd2b25c0035919f640471c2c36be3e5a89932103e8a76453704b59d8f"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 91, "leak_blocked": false, "bytes_after": 91, "payload_root_name": "clean_payload"}
{"payload_bytes": 131, "leak_blocked": true, "bytes_after": 84, "payload_root_name": "event"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": false, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 145, "leak_blocked": false, "bytes_after": 145, "payload_root_name": "clean_payload"}
{"payload_bytes": 172, "keys_top_level": ["tipo_evento", "ativo", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785881750223, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785881750235, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785881750242, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785881751968, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785881751974, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 169}
{"cache_hit": true, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 169}
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "8873e263cbdd6c3ce6075f4ff823824b4b924d24a1db54e85239076b0e37893a"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "a65c3abcd2b25c0035919f640471c2c36be3e5a89932103e8a76453704b59d8f"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 91, "leak_blocked": false, "bytes_after": 91, "payload_root_name": "clean_payload"}
{"payload_bytes": 131, "leak_blocked": true, "bytes_after": 84, "payload_root_name": "event"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": false, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 145, "leak_blocked": false, "bytes_after": 145, "payload_root_name": "clean_payload"}
{"payload_bytes": 172, "keys_top_level": ["tipo_evento", "ativo", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785881929432, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785881929444, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785881929453, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785881931116, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785881931122, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 338}
{"cache_hit": true, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 338}
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "8873e263cbdd6c3ce6075f4ff823824b4b924d24a1db54e85239076b0e37893a"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "a65c3abcd2b25c0035919f640471c2c36be3e5a89932103e8a76453704b59d8f"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 91, "leak_blocked": false, "bytes_after": 91, "payload_root_name": "clean_payload"}
{"payload_bytes": 131, "leak_blocked": true, "bytes_after": 84, "payload_root_name": "event"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785882126817, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785882126832, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785882126853, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785882126858, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 145, "leak_blocked": false, "bytes_after": 145, "payload_root_name": "clean_payload"}
{"payload_bytes": 172, "keys_top_level": ["tipo_evento", "ativo", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785882146934, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785882146946, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785882146955, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785882146977, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785882146984, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 554}
{"cache_hit": true, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 554}
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "8873e263cbdd6c3ce6075f4ff823824b4b924d24a1db54e85239076b0e37893a"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "a65c3abcd2b25c0035919f640471c2c36be3e5a89932103e8a76453704b59d8f"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 91, "leak_blocked": false, "bytes_after": 91, "payload_root_name": "clean_payload"}
{"payload_bytes": 131, "leak_blocked": true, "bytes_after": 84, "payload_root_name": "event"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": false, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 145, "leak_blocked": false, "bytes_after": 145, "payload_root_name": "clean_payload"}
{"payload_bytes": 172, "keys_top_level": ["tipo_evento", "ativo", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785882850404, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785882850417, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785882850426, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785882850451, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785882850456, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 1257}
{"cache_hit": true, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 1257}
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "8873e263cbdd6c3ce6075f4ff823824b4b924d24a1db54e85239076b0e37893a"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "a65c3abcd2b25c0035919f640471c2c36be3e5a89932103e8a76453704b59d8f"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 91, "leak_blocked": false, "bytes_after": 91, "payload_root_name": "clean_payload"}
{"payload_bytes": 131, "leak_blocked": true, "bytes_after": 84, "payload_root_name": "event"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": false, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 145, "leak_blocked": false, "bytes_after": 145, "payload_root_name": "clean_payload"}
{"payload_bytes": 172, "keys_top_level": ["tipo_evento", "ativo", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785883437296, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785883437307, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785883437317, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785883437341, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785883437347, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "8873e263cbdd6c3ce6075f4ff823824b4b924d24a1db54e85239076b0e37893a"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "a65c3abcd2b25c0035919f640471c2c36be3e5a89932103e8a76453704b59d8f"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 91, "leak_blocked": false, "bytes_after": 91, "payload_root_name": "clean_payload"}
{"payload_bytes": 131, "leak_blocked": true, "bytes_after": 84, "payload_root_name": "event"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": false, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 145, "leak_blocked": false, "bytes_after": 145, "payload_root_name": "clean_payload"}
{"payload_bytes": 172, "keys_top_level": ["tipo_evento", "ativo", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785886693531, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785886693544, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785886693555, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785886693582, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785886693587, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "8873e263cbdd6c3ce6075f4ff823824b4b924d24a1db54e85239076b0e37893a"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "a65c3abcd2b25c0035919f640471c2c36be3e5a89932103e8a76453704b59d8f"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 91, "leak_blocked": false, "bytes_after": 91, "payload_root_name": "clean_payload"}
{"payload_bytes": 131, "leak_blocked": true, "bytes_after": 84, "payload_root_name": "event"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": false, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 145, "leak_blocked": false, "bytes_after": 145, "payload_root_name": "clean_payload"}
{"payload_bytes": 172, "keys_top_level": ["tipo_evento", "ativo", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785888496259, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785888496299, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785888496313, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785888496376, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785888496390, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "8873e263cbdd6c3ce6075f4ff823824b4b924d24a1db54e85239076b0e37893a"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "a65c3abcd2b25c0035919f640471c2c36be3e5a89932103e8a76453704b59d8f"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 91, "leak_blocked": false, "bytes_after": 91, "payload_root_name": "clean_payload"}
{"payload_bytes": 131, "leak_blocked": true, "bytes_after": 84, "payload_root_name": "event"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": false, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 145, "leak_blocked": false, "bytes_after": 145, "payload_root_name": "clean_payload"}
{"payload_bytes": 172, "keys_top_level": ["tipo_evento", "ativo", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785895980711, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785895980793, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785895980840, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785895980971, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785895980984, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "8873e263cbdd6c3ce6075f4ff823824b4b924d24a1db54e85239076b0e37893a"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "a65c3abcd2b25c0035919f640471c2c36be3e5a89932103e8a76453704b59d8f"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 91, "leak_blocked": false, "bytes_after": 91, "payload_root_name": "clean_payload"}
{"payload_bytes": 131, "leak_blocked": true, "bytes_after": 84, "payload_root_name": "event"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": false, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 145, "leak_blocked": false, "bytes_after": 145, "payload_root_name": "clean_payload"}
{"payload_bytes": 172, "keys_top_level": ["tipo_evento", "ativo", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785898065780, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785898065824, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785898065838, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785898065903, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785898065914, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "8873e263cbdd6c3ce6075f4ff823824b4b924d24a1db54e85239076b0e37893a"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "a65c3abcd2b25c0035919f640471c2c36be3e5a89932103e8a76453704b59d8f"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 91, "leak_blocked": false, "bytes_after": 91, "payload_root_name": "clean_payload"}
{"payload_bytes": 131, "leak_blocked": true, "bytes_after": 84, "payload_root_name": "event"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": false, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 145, "leak_blocked": false, "bytes_after": 145, "payload_root_name": "clean_payload"}
{"payload_bytes": 172, "keys_top_level": ["tipo_evento", "ativo", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785899866651, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785899866692, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785899866707, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785899866770, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785899866780, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "8873e263cbdd6c3ce6075f4ff823824b4b924d24a1db54e85239076b0e37893a"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "a65c3abcd2b25c0035919f640471c2c36be3e5a89932103e8a76453704b59d8f"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 91, "leak_blocked": false, "bytes_after": 91, "payload_root_name": "clean_payload"}
{"payload_bytes": 131, "leak_blocked": true, "bytes_after": 84, "payload_root_name": "event"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": false, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 145, "leak_blocked": false, "bytes_after": 145, "payload_root_name": "clean_payload"}
{"payload_bytes": 172, "keys_top_level": ["tipo_evento", "ativo", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785959223522, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785959223539, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785959223547, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785959223568, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785959223573, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "8873e263cbdd6c3ce6075f4ff823824b4b924d24a1db54e85239076b0e37893a"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "a65c3abcd2b25c0035919f640471c2c36be3e5a89932103e8a76453704b59d8f"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 91, "leak_blocked": false, "bytes_after": 91, "payload_root_name": "clean_payload"}
{"payload_bytes": 131, "leak_blocked": true, "bytes_after": 84, "payload_root_name": "event"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": false, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 145, "leak_blocked": false, "bytes_after": 145, "payload_root_name": "clean_payload"}
{"payload_bytes": 172, "keys_top_level": ["tipo_evento", "ativo", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785974450349, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785974450393, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785974450409, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785974450467, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785974450478, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "8873e263cbdd6c3ce6075f4ff823824b4b924d24a1db54e85239076b0e37893a"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "a65c3abcd2b25c0035919f640471c2c36be3e5a89932103e8a76453704b59d8f"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 91, "leak_blocked": false, "bytes_after": 91, "payload_root_name": "clean_payload"}
{"payload_bytes": 131, "leak_blocked": true, "bytes_after": 84, "payload_root_name": "event"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": false, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 145, "leak_blocked": false, "bytes_after": 145, "payload_root_name": "clean_payload"}
{"payload_bytes": 172, "keys_top_level": ["tipo_evento", "ativo", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785976237535, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785976237585, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785976237602, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785976237674, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785976237685, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 1787}
{"cache_hit": true, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 1787}
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "8873e263cbdd6c3ce6075f4ff823824b4b924d24a1db54e85239076b0e37893a"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "a65c3abcd2b25c0035919f640471c2c36be3e5a89932103e8a76453704b59d8f"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 91, "leak_blocked": false, "bytes_after": 91, "payload_root_name": "clean_payload"}
{"payload_bytes": 131, "leak_blocked": true, "bytes_after": 84, "payload_root_name": "event"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": false, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": true, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": true, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 145, "leak_blocked": false, "bytes_after": 145, "payload_root_name": "clean_payload"}
{"payload_bytes": 172, "keys_top_level": ["tipo_evento", "ativo", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785984434608, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785984434650, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785984434665, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785984434725, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785984434736, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 145, "leak_blocked": false, "bytes_after": 145, "payload_root_name": "clean_payload"}
{"payload_bytes": 172, "keys_top_level": ["tipo_evento", "ativo", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785984553169, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785984553187, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785984553195, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785984553222, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785984553229, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 117}
{"cache_hit": true, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 117}
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "8873e263cbdd6c3ce6075f4ff823824b4b924d24a1db54e85239076b0e37893a"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "a65c3abcd2b25c0035919f640471c2c36be3e5a89932103e8a76453704b59d8f"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 91, "leak_blocked": false, "bytes_after": 91, "payload_root_name": "clean_payload"}
{"payload_bytes": 131, "leak_blocked": true, "bytes_after": 84, "payload_root_name": "event"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": false, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 145, "leak_blocked": false, "bytes_after": 145, "payload_root_name": "clean_payload"}
{"payload_bytes": 172, "keys_top_level": ["tipo_evento", "ativo", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785985630377, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785985630396, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785985630406, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785985630448, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1785985630453, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 1195}
{"cache_hit": true, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 1195}
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "8873e263cbdd6c3ce6075f4ff823824b4b924d24a1db54e85239076b0e37893a"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "a65c3abcd2b25c0035919f640471c2c36be3e5a89932103e8a76453704b59d8f"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 91, "leak_blocked": false, "bytes_after": 91, "payload_root_name": "clean_payload"}
{"payload_bytes": 131, "leak_blocked": true, "bytes_after": 84, "payload_root_name": "event"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 48174, "leak_blocked": true, "bytes_after": 3409, "payload_root_name": "event"}
{"payload_bytes": 3499, "keys_top_level": ["symbol", "epoch_ms", "trigger", "price", "regime", "qual", "flow", "ob", "tf", "sr", "w", "ext", "ctx", "ofi", "vwap", "liq", "summary", "tipo_evento", "descricao"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786054020000, "counts": {}}
{"payload_bytes": 49925, "leak_blocked": true, "bytes_after": 3670, "payload_root_name": "event"}
{"payload_bytes": 3789, "keys_top_level": ["symbol", "epoch_ms", "trigger", "price", "regime", "qual", "flow", "ob", "tf", "sr", "w", "ext", "alerts", "ctx", "ofi", "vwap", "iceberg", "liq", "summary", "tipo_evento", "descricao", "ativo"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786054380000, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": true, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": true, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": true, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": true, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 145, "leak_blocked": false, "bytes_after": 145, "payload_root_name": "clean_payload"}
{"payload_bytes": 172, "keys_top_level": ["tipo_evento", "ativo", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786064979299, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786064979349, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786064979364, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786064979503, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786064979515, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "8873e263cbdd6c3ce6075f4ff823824b4b924d24a1db54e85239076b0e37893a"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "a65c3abcd2b25c0035919f640471c2c36be3e5a89932103e8a76453704b59d8f"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 91, "leak_blocked": false, "bytes_after": 91, "payload_root_name": "clean_payload"}
{"payload_bytes": 131, "leak_blocked": true, "bytes_after": 84, "payload_root_name": "event"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": false, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 145, "leak_blocked": false, "bytes_after": 145, "payload_root_name": "clean_payload"}
{"payload_bytes": 172, "keys_top_level": ["tipo_evento", "ativo", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786065461495, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786065461539, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786065461552, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786065461613, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786065461624, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 481}
{"cache_hit": true, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 481}
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "8873e263cbdd6c3ce6075f4ff823824b4b924d24a1db54e85239076b0e37893a"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "a65c3abcd2b25c0035919f640471c2c36be3e5a89932103e8a76453704b59d8f"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 91, "leak_blocked": false, "bytes_after": 91, "payload_root_name": "clean_payload"}
{"payload_bytes": 131, "leak_blocked": true, "bytes_after": 84, "payload_root_name": "event"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": false, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": true, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 145, "leak_blocked": false, "bytes_after": 145, "payload_root_name": "clean_payload"}
{"payload_bytes": 172, "keys_top_level": ["tipo_evento", "ativo", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786066633262, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786066633306, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786066633320, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786066633379, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786066633389, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 1653}
{"cache_hit": true, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 1653}
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "8873e263cbdd6c3ce6075f4ff823824b4b924d24a1db54e85239076b0e37893a"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "a65c3abcd2b25c0035919f640471c2c36be3e5a89932103e8a76453704b59d8f"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 91, "leak_blocked": false, "bytes_after": 91, "payload_root_name": "clean_payload"}
{"payload_bytes": 131, "leak_blocked": true, "bytes_after": 84, "payload_root_name": "event"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": false, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 145, "leak_blocked": false, "bytes_after": 145, "payload_root_name": "clean_payload"}
{"payload_bytes": 172, "keys_top_level": ["tipo_evento", "ativo", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786067098317, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786067098362, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786067098377, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786067098436, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786067098446, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "8873e263cbdd6c3ce6075f4ff823824b4b924d24a1db54e85239076b0e37893a"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "a65c3abcd2b25c0035919f640471c2c36be3e5a89932103e8a76453704b59d8f"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 91, "leak_blocked": false, "bytes_after": 91, "payload_root_name": "clean_payload"}
{"payload_bytes": 131, "leak_blocked": true, "bytes_after": 84, "payload_root_name": "event"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": false, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 145, "leak_blocked": false, "bytes_after": 145, "payload_root_name": "clean_payload"}
{"payload_bytes": 172, "keys_top_level": ["tipo_evento", "ativo", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786067499967, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786067500010, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786067500024, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786067500083, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786067500093, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 402}
{"cache_hit": true, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 402}
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "8873e263cbdd6c3ce6075f4ff823824b4b924d24a1db54e85239076b0e37893a"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "a65c3abcd2b25c0035919f640471c2c36be3e5a89932103e8a76453704b59d8f"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 91, "leak_blocked": false, "bytes_after": 91, "payload_root_name": "clean_payload"}
{"payload_bytes": 131, "leak_blocked": true, "bytes_after": 84, "payload_root_name": "event"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 1315}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "8873e263cbdd6c3ce6075f4ff823824b4b924d24a1db54e85239076b0e37893a"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "a65c3abcd2b25c0035919f640471c2c36be3e5a89932103e8a76453704b59d8f"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 91, "leak_blocked": false, "bytes_after": 91, "payload_root_name": "clean_payload"}
{"payload_bytes": 131, "leak_blocked": true, "bytes_after": 84, "payload_root_name": "event"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 145, "leak_blocked": false, "bytes_after": 145, "payload_root_name": "clean_payload"}
{"payload_bytes": 172, "keys_top_level": ["tipo_evento", "ativo", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786069127955, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786069128021, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786069128036, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786069128098, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786069128109, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 145, "leak_blocked": false, "bytes_after": 145, "payload_root_name": "clean_payload"}
{"payload_bytes": 172, "keys_top_level": ["tipo_evento", "ativo", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786069234382, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786069234426, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786069234442, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786069234592, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786069234603, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "8873e263cbdd6c3ce6075f4ff823824b4b924d24a1db54e85239076b0e37893a"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "a65c3abcd2b25c0035919f640471c2c36be3e5a89932103e8a76453704b59d8f"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 91, "leak_blocked": false, "bytes_after": 91, "payload_root_name": "clean_payload"}
{"payload_bytes": 131, "leak_blocked": true, "bytes_after": 84, "payload_root_name": "event"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": false, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": true, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 145, "leak_blocked": false, "bytes_after": 145, "payload_root_name": "clean_payload"}
{"payload_bytes": 172, "keys_top_level": ["tipo_evento", "ativo", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786070178040, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786070178083, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786070178098, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786070178158, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786070178168, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 943}
{"cache_hit": true, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 943}
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "8873e263cbdd6c3ce6075f4ff823824b4b924d24a1db54e85239076b0e37893a"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "a65c3abcd2b25c0035919f640471c2c36be3e5a89932103e8a76453704b59d8f"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 91, "leak_blocked": false, "bytes_after": 91, "payload_root_name": "clean_payload"}
{"payload_bytes": 131, "leak_blocked": true, "bytes_after": 84, "payload_root_name": "event"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 51366, "leak_blocked": true, "bytes_after": 3737, "payload_root_name": "event"}
{"payload_bytes": 3827, "keys_top_level": ["symbol", "epoch_ms", "trigger", "price", "regime", "qual", "flow", "ob", "tf", "sr", "w", "ext", "alerts", "ctx", "ofi", "vwap", "liq", "cvd_div", "mr", "summary", "tipo_evento", "descricao"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786141980000, "counts": {}}
{"payload_bytes": 51967, "leak_blocked": true, "bytes_after": 3774, "payload_root_name": "event"}
{"payload_bytes": 3864, "keys_top_level": ["symbol", "epoch_ms", "trigger", "price", "regime", "qual", "flow", "ob", "tf", "sr", "w", "ext", "ctx", "ofi", "vwap", "iceberg", "liq", "cvd_div", "mr", "summary", "tipo_evento", "descricao"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786142280000, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 145, "leak_blocked": false, "bytes_after": 145, "payload_root_name": "clean_payload"}
{"payload_bytes": 172, "keys_top_level": ["tipo_evento", "ativo", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786147235502, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786147235521, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786147235533, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786147235563, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786147235570, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "8873e263cbdd6c3ce6075f4ff823824b4b924d24a1db54e85239076b0e37893a"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "a65c3abcd2b25c0035919f640471c2c36be3e5a89932103e8a76453704b59d8f"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 91, "leak_blocked": false, "bytes_after": 91, "payload_root_name": "clean_payload"}
{"payload_bytes": 131, "leak_blocked": true, "bytes_after": 84, "payload_root_name": "event"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": false, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 145, "leak_blocked": false, "bytes_after": 145, "payload_root_name": "clean_payload"}
{"payload_bytes": 172, "keys_top_level": ["tipo_evento", "ativo", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786147455065, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786147455083, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786147455092, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786147455121, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786147455126, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 219}
{"cache_hit": true, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 219}
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "8873e263cbdd6c3ce6075f4ff823824b4b924d24a1db54e85239076b0e37893a"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "a65c3abcd2b25c0035919f640471c2c36be3e5a89932103e8a76453704b59d8f"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 91, "leak_blocked": false, "bytes_after": 91, "payload_root_name": "clean_payload"}
{"payload_bytes": 131, "leak_blocked": true, "bytes_after": 84, "payload_root_name": "event"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": false, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 145, "leak_blocked": false, "bytes_after": 145, "payload_root_name": "clean_payload"}
{"payload_bytes": 172, "keys_top_level": ["tipo_evento", "ativo", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786147632112, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786147632134, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786147632138, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786147632174, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786147632181, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 396}
{"cache_hit": true, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 396}
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "8873e263cbdd6c3ce6075f4ff823824b4b924d24a1db54e85239076b0e37893a"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "a65c3abcd2b25c0035919f640471c2c36be3e5a89932103e8a76453704b59d8f"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 91, "leak_blocked": false, "bytes_after": 91, "payload_root_name": "clean_payload"}
{"payload_bytes": 131, "leak_blocked": true, "bytes_after": 84, "payload_root_name": "event"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": false, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 145, "leak_blocked": false, "bytes_after": 145, "payload_root_name": "clean_payload"}
{"payload_bytes": 172, "keys_top_level": ["tipo_evento", "ativo", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786147836680, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786147836697, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786147836706, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786147836724, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786147836741, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 600}
{"cache_hit": true, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 600}
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "8873e263cbdd6c3ce6075f4ff823824b4b924d24a1db54e85239076b0e37893a"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "a65c3abcd2b25c0035919f640471c2c36be3e5a89932103e8a76453704b59d8f"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 91, "leak_blocked": false, "bytes_after": 91, "payload_root_name": "clean_payload"}
{"payload_bytes": 131, "leak_blocked": true, "bytes_after": 84, "payload_root_name": "event"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": false, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 145, "leak_blocked": false, "bytes_after": 145, "payload_root_name": "clean_payload"}
{"payload_bytes": 172, "keys_top_level": ["tipo_evento", "ativo", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786148028316, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786148028342, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786148028351, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786148028373, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786148028373, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 791}
{"cache_hit": true, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 791}
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "8873e263cbdd6c3ce6075f4ff823824b4b924d24a1db54e85239076b0e37893a"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "a65c3abcd2b25c0035919f640471c2c36be3e5a89932103e8a76453704b59d8f"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 91, "leak_blocked": false, "bytes_after": 91, "payload_root_name": "clean_payload"}
{"payload_bytes": 131, "leak_blocked": true, "bytes_after": 84, "payload_root_name": "event"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"cache_hit": false, "section": "macro_context", "ref": "17bc8650f22cea0b345f4455318acc08919024445e9aa316edb201903162b28c"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 162, "leak_blocked": false, "bytes_after": 162, "payload_root_name": "clean_payload"}
{"payload_bytes": 9294, "leak_blocked": true, "bytes_after": 162, "payload_root_name": "event"}
{"payload_bytes": 75, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 145, "leak_blocked": false, "bytes_after": 145, "payload_root_name": "clean_payload"}
{"payload_bytes": 172, "keys_top_level": ["tipo_evento", "ativo", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786148187020, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786148187038, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786148187047, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786148187075, "counts": {}}
{"payload_bytes": 173, "leak_blocked": false, "bytes_after": 173, "payload_root_name": "clean_payload"}
{"payload_bytes": 200, "keys_top_level": ["tipo_evento", "ativo", "symbol", "delta", "volume_total", "preco_fechamento", "resultado_da_batalha", "epoch_ms"], "schema_version": "v1", "symbol": "BTCUSDT", "epoch_ms": 1786148187079, "counts": {}}
{"cache_hit": false, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 950}
{"cache_hit": true, "section": "macro_context", "ref": "fdc87e4532c300f8aba2f6636c5451a6b267855b93946f14656e18705600311b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 950}
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": true, "section": "macro_context", "ref": "498bcc8d56f68f526707a228e91a0f760c1426a30445b3c9f07413fdb5d2b23b", "age_s": 0}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"cache_hit": false, "section": "macro_context", "ref": "8873e263cbdd6c3ce6075f4ff823824b4b924d24a1db54e85239076b0e37893a"}
{"cache_hit": false, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c"}
{"cache_hit": false, "section": "macro_context", "ref": "a65c3abcd2b25c0035919f640471c2c36be3e5a89932103e8a76453704b59d8f"}
{"cache_hit": true, "section": "cross_asset_context", "ref": "6379f5f56ae42877850482b4dc4b715f72a734b01ef851b7c26c9275b7fb2d7c", "age_s": 0}
{"payload_bytes": 91, "leak_blocked": false, "bytes_after": 91, "payload_root_name": "clean_payload"}
{"payload_bytes": 131, "leak_blocked": true, "bytes_after": 84, "payload_root_name": "event"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
```


## 1.3 scripts/audit_json_payload_costs.py — saída da execução


Comando: `python scripts/audit_json_payload_costs.py dados/eventos-fluxo.json --top 5`


```text
arquivo: dados\eventos-fluxo.json
eventos: 18

**Tamanhos (bytes, JSON minificado)**
evento_total: n=18 min=705 p50=21261 p90=22098 p95=22412 max=41385 avg=18332
raw_event: n=13 min=934 p50=937 p90=944 p95=944 max=1235 avg=961
ai_payload: n=2 min=3401 p50=3401 p90=3447 p95=3447 max=3447 avg=3424
otimizado(AIPayloadOptimizer): n=13 min=777 p50=809 p90=824 p95=824 max=874 avg=814

raw_event participacao: avg=4.5% p50=4.4% p95=4.6%

**Top-level keys (frequencia)**
- tipo_evento: 18
- timestamp: 18
- epoch_ms: 18
- data_context: 18
- event_id: 18
- timestamp_utc: 18
- timestamp_ny: 18
- timestamp_sp: 18
- descricao: 16
- resultado_da_batalha: 16
- janela_numero: 16
- market_context: 16
- volatility_metrics: 16
- symbol: 15
- is_signal: 14
- delta: 14
- volume_total: 14
- volume_compra: 14
- volume_venda: 14
- preco_fechamento: 14
- ml_features: 14
- orderbook_data: 14
- historical_vp: 14
- multi_tf: 14
- pattern_recognition: 14

**raw_event keys (frequencia)**
- delta: 13
- volume_total: 13
- volume_compra: 13
- volume_venda: 13
- preco_fechamento: 13
- advanced_analysis: 13
- orderbook_data: 1
- timestamp_utc: 1

**Maiores eventos (top 5)**
- idx=14 tipo=Exaustão bytes=41385
- idx=15 tipo=ANALYSIS_TRIGGER bytes=22412
- idx=10 tipo=ANALYSIS_TRIGGER bytes=22098
- idx=8 tipo=ANALYSIS_TRIGGER bytes=21975
- idx=6 tipo=ANALYSIS_TRIGGER bytes=21802

**raw_event subkeys por tamanho medio (top 20)**
- advanced_analysis: avg=814 max=843 n=13
- orderbook_data: avg=215 max=215 n=1
- timestamp_utc: avg=13 max=13 n=1
- preco_fechamento: avg=6 max=7 n=13
- delta: avg=6 max=7 n=13
- volume_total: avg=5 max=6 n=13
- volume_venda: avg=5 max=5 n=13
- volume_compra: avg=5 max=5 n=13
```


## 1.4 Tamanho atual do payload (bytes/tokens)


Métricas observadas (medidas em 2026-08-07):

- `logs/last_llm_payload.json` (dump antigo 04/08): **539 bytes no arquivo**, campo `_dump_meta.payload_bytes_final: 273` → ~68 tokens (273/4).
- `payload_metrics.jsonl` (n=561 linhas com `payload_bytes`): min=23, max=52_037, avg=1_588, **p50=173**, p95=9_294 bytes.
- Payloads v1 reais (schema v1, keys longas): 3_827 e 3_864 bytes → ~960-970 tokens.
- Eventos brutos (antes do compact): p50=21_261 bytes; evento Exaustão máximo = 41_385 bytes.
- Eventos com guardrail `leak_blocked` (51_366 → 3_737 bytes; 51_967 → 3_774): bloqueio e compactação de ~93%.
- Amostra de 2 `ai_payload` no arquivo de eventos: avg 3_424 bytes (~850 tokens).
- Payload compactado v3.1 (`build_compact_payload`): ~200 tokens (docstring do próprio código).

## 1.5 Quais seções compõem o payload final


Cadeia real de construção (confirmada por grep de imports):

- **Construtor ativo**: `market_orchestrator/ai/payload_builder_compact.py` → `build_compact_payload()` (chamado por `market_orchestrator/ai/ai_runner.py:82`).
- **Seções incluídas sempre** (payload_builder_compact.py:1466-1525): `symbol`, `epoch_ms`, `trigger`, `price`, `flow`, `ob`, `tf`, `sr` (obrigatórias, com stub `{"_": "no_data"}`).
- **Seções condicionais**: `regime`, `qual`, `w` (whale), `quant`, `ext`, `alerts`, `ctx` (com cache/mini-ctx), `ofi`, `vwap`, `iceberg`, `liq`, `sm`, `cvd_div`, `mr`, `summary`.
- **`summary`** = resultado dos summary builders de `market_orchestrator/ai/payload_sections/`: `flow_summary`, `sr_summary`, `regime_summary`, `institutional_summary`, `quality_summary` (todos carregados em payload_builder_compact.py:44-58).
- **`skill_bridge.py` NÃO é importado** pelo `payload_sections/__init__.py` (linhas 17-21 só importam os 5 builders acima). É código preparado para uso futuro (dead code por ora).
- **`payload_compressor_v3.py` NÃO é usado no runtime** — só referenciado em comentários do analyzer_qwen e em testes (`tests/integration/test_patch_compressor_v3.py`, `tests/integration/test_ai_runner.py`).
- **`ai_payload_builder.py`** só é usado para `get_llm_payload_config()` (config de llm_payload: v2_enabled, max_bytes=6144, guardrail, budgets) — importado por analyzer_qwen.py:336.

## 1.6 market_orchestrator/ai/ai_payload_builder.py (primeiras 50 linhas)


```python
# market_orchestrator/ai/ai_payload_builder.py
# -*- coding: utf-8 -*-
"""
Construtor de Payload para Análise de IA.

Este módulo é responsável por padronizar e organizar os dados brutos e métricas
do sistema em um formato estruturado e semântico para consumo pelos modelos de IA.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone
from pathlib import Path
import logging
import json

_ML_EXTREME_THRESHOLD_HIGH = 0.95
_ML_EXTREME_THRESHOLD_LOW = 0.05
import hashlib
import os
from functools import lru_cache

import yaml

# Import do otimizador de payload (localizado em src/utils/)
from common.ai_payload_optimizer import AIPayloadOptimizer, compact_historical_vp

from market_orchestrator.ai.ai_enrichment_context import build_enriched_ai_context
from market_orchestrator.ai.payload_compressor import compress_payload
from market_orchestrator.ai.payload_section_cache import SectionCache, canonical_ref, is_fresh
from market_orchestrator.ai.payload_metrics_aggregator import append_metric_line


def _check_in_range(price, low, high):
    """Helper simples para verificar se preço está em range."""
    if price is None or low is None or high is None:
        return None
    return low <= price <= high


def _strip_empty(d: dict) -> dict:
    """Remove recursivamente chaves com valor None, {}, [] ou string vazia."""
    if not isinstance(d, dict):
        return d
    cleaned = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, dict):
            v = _strip_empty(v)
            if not v:  # dict vazio após limpeza
[... TRUNCADO: 1398 linhas no total, mostradas 50 ...]
```


## 1.7 market_orchestrator/ai/payload_compressor_v3.py (primeiras 50 linhas)


```python
# market_orchestrator/ai/payload_compressor_v3.py
"""
Payload Compressor V3.1 -- Compressor Inteligente para LLM API
==============================================================
Comprime dados redundantes/verbosos mantendo 100% da qualidade analítica.
Estratégia:
  - Remove campos duplicados e redundantes
  - Abrevia labels longos (strings descritivas)
  - Mantém TODOS os dados numéricos críticos
  - Preserva dados institucionais completos
  - Adiciona campos que estavam sendo perdidos

Economia estimada: ~65% tokens sem perda de qualidade analítica.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════
# MAPEAMENTOS DE COMPRESSÃO (strings longas → abreviações)
# ══════════════════════════════════════════════════════════════════

REGIME_MAP = {
    # Português
    "Alta": "UP", "Baixa": "DOWN", "Lateral": "SIDE",
    "alta": "UP", "baixa": "DOWN", "lateral": "SIDE",
    "Acumulação": "ACCUM", "Manipulação": "MANIP",
    "Distribuição": "DIST", "Expansão": "EXPAN", "Range": "RANGE",
    # Encoding corrompido (fallback)
    "Acumulação": "ACCUM", "Manipulação": "MANIP",
    "Distribuição": "DIST", "Expansão": "EXPAN",
    # CORREÇÃO BUG2: inglês que o signal usa diretamente
    "neutral": "NEUT", "Neutral": "NEUT",
    "bullish": "UP",   "Bullish": "UP",
    "bearish": "DOWN", "Bearish": "DOWN",
    "trending": "TREND", "ranging": "RANGE",
    "breakout": "BREAK", "reversal": "REV",
    "accumulation": "ACCUM", "distribution": "DIST",
    # Abreviações que o compressor v1 usa
    "UP": "UP", "DOWN": "DOWN", "SIDE": "SIDE",
    "NEUT": "NEUT", "TREND": "TREND",
}

FLOW_TREND_MAP = {
    "accelerating_selling": "accel_sell",
    "accelerating_buying": "accel_buy",
    "decelerating_selling": "decel_sell",
[... TRUNCADO: 1219 linhas no total, mostradas 50 ...]
```


## 1.8 market_orchestrator/ai/payload_sections/ (primeiras 50 linhas de cada)


### payload_sections/flow_summary.py


```python
"""
flow_summary — Resume microestrutura e fluxo de ordens em conclusão operacional.

Transforma:
    flow.pa, flow.abs_*, flow.imb, flow.d1/d5/d15,
    flow.sf_w/r, flow.ti, flow.trs

Em:
    flow_summary: {
        "bias":    "BUY" | "SELL" | "NEUTRAL",
        "type":    "absorption" | "aggressive" | "passive" | "mixed",
        "actor":   "whale" | "retail" | "mixed" | "unknown",
        "conf":    "H" | "M" | "L",
        "note":    str  # frase curta interpretada
    }
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_PA_SIGNAL_BIAS: dict[str, str] = {
    "buy_absorp":       "BUY",
    "sell_absorp":      "SELL",
    "buy_aggres":       "BUY",
    "sell_aggre":       "SELL",
    "buy_passiv":       "BUY",
    "sell_passi":       "SELL",
    "buy_absorption":   "BUY",
    "sell_absorption":  "SELL",
    "buy_aggressive":   "BUY",
    "sell_aggressive":  "SELL",
    "buy_passive":      "BUY",
    "sell_passive":     "SELL",
    "neutral":          "NEUTRAL",
}

_CONVICTION_MAP: dict[str, str] = {
    "HIGH":   "H",
    "MEDIUM": "M",
    "LOW":    "L",
    "H":      "H",
    "M":      "M",
    "L":      "L",
}
[... TRUNCADO: 248 linhas no total, mostradas 50 ...]
```


### payload_sections/sr_summary.py


```python
"""
sr_summary — Resume suportes e resistências em contexto operacional.

Transforma:
    sr.r1, sr.r1_dist, sr.r1_conf
    sr.s1, sr.s1_dist, sr.s1_conf
    sr.def_bias

Em:
    sr_summary: {
        "nearest":    "support" | "resistance" | "equidistant",
        "compressed": bool,       # preço entre níveis muito próximos
        "conf_bias":  "BUY" | "SELL" | "NEUTRAL",
        "r1_dist_atr": float,     # distância da resistência em ATRs
        "s1_dist_atr": float,     # distância do suporte em ATRs
        "note":       str
    }
"""

from __future__ import annotations

from typing import Any


_BIAS_MAP: dict[str, str] = {
    "buyers":  "BUY",
    "sellers": "SELL",
    "neutral": "NEUTRAL",
    "buy":     "BUY",
    "sell":    "SELL",
}

# Compressão: preço está entre S/R com gap < 0.5% do preço
_COMPRESSION_THRESHOLD_PCT = 0.005

# Muito próximo: < 0.15% do preço
_NEAR_THRESHOLD_PCT = 0.0015


def build_sr_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Gera resumo interpretado de suporte/resistência.

    Args:
        payload: payload já construído pelo build_compact_payload()

    Returns:
        dict com nearest, compressed, conf_bias, distâncias em ATR e note
    """
    sr = payload.get("sr", {})
[... TRUNCADO: 219 linhas no total, mostradas 50 ...]
```


### payload_sections/regime_summary.py


```python
"""
regime_summary — Resume regime de mercado com estratégias recomendadas e proibidas.

Transforma:
    regime.cs, regime.cf, regime.v, regime.mode
    regime.bbw, regime.atr%
    tf.*.r (regime por timeframe)

Em:
    regime_summary: {
        "label":      str,    # descrição legível
        "strategies": list,   # o que fazer neste regime
        "avoid":      list,   # o que evitar
        "duration":   str,    # expectativa de duração
        "note":       str
    }
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Constantes por modo de mercado
# ---------------------------------------------------------------------------

_MODE_LABEL: dict[str, str] = {
    "MR":  "Mean Reversion",
    "RB":  "Range Bound",
    "TRD": "Trending",
    "BRK": "Breakout",
}

_MODE_STRATEGIES: dict[str, list[str]] = {
    "MR": [
        "fade extremos",
        "vender resistência",
        "comprar suporte",
        "aguardar absorção nos extremos",
    ],
    "RB": [
        "operar no range",
        "comprar VAL, vender VAH",
        "reduzir exposição fora do POC",
        "aguardar catalisador para breakout",
    ],
    "TRD": [
        "seguir tendência dominante",
        "comprar pullbacks no uptrend",
[... TRUNCADO: 219 linhas no total, mostradas 50 ...]
```


### payload_sections/institutional_summary.py


```python
"""
institutional_summary — Resume análise institucional em conclusão acionável.

Transforma:
    price.sh (profile shape), price.auc (auction bias)
    price.ph / price.pl (poor extremes)
    price.brk_risk (breakout risk do Value Area)
    w.s, w.c (whale score e classificação)
    flow.pa, flow.conv

Em:
    institutional_summary: {
        "auction_state": str,
        "whale_bias":    "ACCUMULATING" | "DISTRIBUTING" | "NEUTRAL",
        "profile_bias":  "BULLISH" | "BEARISH" | "NEUTRAL",
        "unfinished":    list[str],   # "low" | "high"
        "note":          str
    }
"""

from __future__ import annotations

from typing import Any


_SHAPE_BIAS: dict[str, str] = {
    "b":  "BULLISH",  # b-shape: long liquidation, bullish after
    "p":  "BEARISH",  # p-shape: short covering, bearish after
    "D":  "NEUTRAL",  # double distribution
    "I":  "NEUTRAL",  # thin, indeterminate
}

_WHALE_CLS_BIAS: dict[str, str] = {
    "MA": "ACCUMULATING",
    "SA": "ACCUMULATING",
    "MD": "DISTRIBUTING",
    "SD": "DISTRIBUTING",
    "N":  "NEUTRAL",
}

_AUCTION_MAP: dict[str, str] = {
    "expect_retest_low":  "Leilão incompleto — mínima deve ser revisitada",
    "expect_retest_high": "Leilão incompleto — máxima deve ser revisitada",
    "expect_retest_both": "Leilão incompleto em ambos os extremos",
    "balanced":           "Leilão equilibrado",
    "accept_higher":      "Mercado aceitando preços mais altos",
    "accept_lower":       "Mercado aceitando preços mais baixos",
}


[... TRUNCADO: 185 linhas no total, mostradas 50 ...]
```


### payload_sections/quality_summary.py


```python
"""
quality_summary — Resume qualidade dos dados e impacto na confiança da análise.

Transforma:
    qual.lat, qual.liq, qual.ms, qual.holiday
    ctx.cached

Em:
    quality_summary: {
        "reliable":       bool,
        "confidence_cap": float,
        "issues":         list[str],
        "note":           str
    }
"""

from __future__ import annotations

from typing import Any


_LATENCY_CAPS: dict[str, float] = {
    "OK":   1.0,
    "NEAR": 0.9,
    "DEGR": 0.7,
    "CRIT": 0.4,
}

_LIQUIDITY_CAPS: dict[str, float] = {
    "NORMAL":   1.0,
    "RED":      0.8,
    "LOW":      0.7,
    "VERY_LOW": 0.5,
    "VERY":     0.5,   # fallback para truncamento
    "VER":      0.5,   # fallback para truncamento [:3]
}

_LIQUIDITY_LABELS: dict[str, str] = {
    "NORMAL":   "normal",
    "RED":      "reduzida",
    "LOW":      "baixa",
    "VERY_LOW": "muito baixa",
    "VERY":     "muito baixa",
    "VER":      "muito baixa",
}


def _resolve_liquidity(liq_raw: str) -> tuple[float, str]:
    """Resolve cap e label de liquidez independente de truncamento."""
    key = liq_raw.upper()
[... TRUNCADO: 166 linhas no total, mostradas 50 ...]
```


### payload_sections/skill_bridge.py


```python
"""
skill_bridge.py — Ponte entre o sistema de payload e o futuro framework de skills.

Este módulo define:
  1. O contrato que uma skill analítica deve seguir
  2. Um registry simples de skills disponíveis
  3. A função que o payload builder usará para ativar skills por contexto

Quando o framework de skills/ for criado, este módulo será o ponto
de integração — sem precisar alterar build_compact_payload.py.

Filosofia:
  - Skills são read-only (só leem, não executam ordens)
  - Skills retornam dicts compactos — mesmo formato dos gaps
  - Skills podem ser ativadas por regime, trigger ou contexto
  - Skills com erro são ignoradas silenciosamente
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ============================================================
# CONTRATO DE SKILL
# ============================================================

@runtime_checkable
class AnalyticalSkill(Protocol):
    """
    Protocolo que toda skill analítica deve implementar.

    Uma skill:
      - recebe o payload compacto já montado
      - retorna um dict compacto com sua análise
      - nunca lança exceção para o chamador
      - nunca modifica o payload recebido
    """

    @property
    def name(self) -> str:
        """Identificador único da skill."""
        ...

    @property
    def version(self) -> str:
        """Versão da skill."""
[... TRUNCADO: 366 linhas no total, mostradas 50 ...]
```


# SEÇÃO 2 — EVENTOS DE FLUXO


## 2.1 Comparação eventos-fluxo.json vs eventos_fluxo.jsonl

- `dados/eventos-fluxo.json`: **23.021 linhas**, JSON array formatado (pretty-printed), 18 eventos.
- `dados/eventos_fluxo.jsonl`: **18 linhas**, 1 evento por linha.
- **Mesmos 18 eventos (mesmos epoch_ms, 100% contidos)** — verificado: todos os epoch_ms do .jsonl existem no .json. Último registro de ambos: `epoch_ms=1786142580000` = 2026-08-07T22:43:00Z.
- Diferença: o .jsonl contém registros **truncados pelo guardian** (`"note":"trimmed_by_guardian"`) para os ANALYSIS_TRIGGER puros, e 1 registro completo (Exaustão) + 1 AI_ANALYSIS com `ai_payload` completo (~4,9 KB). O .json tem tudo completo.
- **SINALIZAÇÃO: os dois arquivos são redundantes** (mesmo propósito e mesmo conteúdo; formatos diferentes). Suspeita: o .jsonl é um artefato de log/export compacto e o .json o armazenamento canônico.

## 2.2 dados/eventos_fluxo.jsonl — últimas 150 linhas


```jsonl
{"tipo_evento":"ANALYSIS_TRIGGER","symbol":"BTCUSDT","epoch_ms":1786141860000,"janela_numero":1,"event_id":"6630d1e4","timestamp_utc":"2026-08-07T22:31:00.000Z","note":"trimmed_by_guardian"}
{"tipo_evento":"ANALYSIS_TRIGGER","symbol":"BTCUSDT","epoch_ms":1786141920000,"janela_numero":2,"event_id":"871e4b9e","timestamp_utc":"2026-08-07T22:32:00.000Z","note":"trimmed_by_guardian"}
{"tipo_evento":"ANALYSIS_TRIGGER","symbol":"BTCUSDT","epoch_ms":1786141980000,"janela_numero":3,"event_id":"3dc6793e","timestamp_utc":"2026-08-07T22:33:00.000Z","note":"trimmed_by_guardian"}
{"tipo_evento":"AI_ANALYSIS","symbol":"BTCUSDT","timestamp_ms":1786141980000,"anchor_price":64889.5,"anchor_window_id":3,"ai_result":{"sentiment":"neutral","confidence":0.65,"action":"wait","rationale":"O preço está em queda nos 15 min, mas nos 1 h e 4 h a tendência ainda é de alta, indicando uma retração dentro de um movimento de alta. O fluxo está dominado po","region_type":"retracement","_is_fallback":0,"_is_valid":1},"ai_payload":{"symbol":"BTCUSDT","epoch_ms":1786141980000,"trigger":"AT","price":{"c":64890,"o":64925,"h":64925,"l":64888,"vw":64909,"sh":"P","auc":"expect_retest_both","ph":1,"pl":1,"brk_risk":"V_HI"},"regime":{"cs":"BULL","cf":0.8,"v":"NOR","mode":"MR","dom":"4h","bull%":90,"bear%":10},"qual":{"lat":"POOR","ms":6984},"flow":{"d1":"+238K","delta":-11.031,"vol":19.309,"buy_pct":21,"ti":27.7,"trs":-239,"obs":-1.487,"sf_w":-1,"sf_r":-7.832,"d5":"-734K","d15":"-734K","cvd":-11.3,"imb":-0.57,"ab":21,"bsr":0.27,"pa":"sell_absor","conv":"M","abs_buy_str":2.1,"abs_sell_exh":5.7},"ob":{"b":"2.0M","a":"989K","imb":0.34,"bias":"BUY","t5":0.38,"spread_pct":0,"slip_b":5,"slip_s":5},"tf":{"15m":{"t":"DN","rsi":43,"macd":[15,17],"adx":25,"atr":66,"r":"RNG"},"1h":{"t":"UP","rsi":55,"macd":[103,99],"adx":25,"atr":220,"r":"RNG"},"4h":{"t":"UP","rsi":65,"macd":[279,258],"adx":30,"atr":492,"r":"RNG"},"1d":{"t":"UP","rsi":59,"macd":[71,37],"adx":15,"atr":1370,"r":"MNP"}},"sr":{"r1":[65037,55],"r1_dist":148,"r1_conf":3,"r2":[65346,53],"r2_dist":456,"r2_conf":3,"s1":[64481,62],"s1_dist":408,"s1_conf":4,"s2":[64836,61],"s2_dist":53,"s2_conf":4,"def_bias":"slight_sel"},"w":{"s":-13,"c":"N"},"ext":{"cci":"OB","stoch":86,"stoch_sig":"OB","wr":-42,"garch":0,"hurst":0.36,"entropy":2.97,"fd":0.62,"kalman":{"kp":64942.82,"dev":-0.0274,"dir":"DOWN"},"reg":{"sl":-1.5658,"pos":0.7697,"dev":24.56},"mc":{"pu":0.526,"p10":64834.57,"p90":64956.23},"cycles":[100,40],"smc":{"struct":"BEAR","bos":0}},"alerts":[{"type":"SUPPORT_TEST","sev":"H","lvl":64836},{"type":"RESISTANCE_TEST","sev":"H","lvl":64945}],"ctx":{"ses":"NY","poc":65046,"val":64522,"vah":65346,"lsr":1.1,"eth_lsr":2.07,"oi":107,"fr":0.0063,"longs":"+3634.5M","shorts":"+3305.6M","eth7":0.8,"dxy30":-0.06},"ofi":{"score":-0.571,"dir":"SELL","src":"order_flow"},"vwap":{"dev":-0.031,"side":"below","sig":"fair","src":"ohlc"},"liq":[{"p":64907,"side":"sell","vol":20.51}],"cvd_div":{"det":1,"type":"bearish_div","src":"inferred"},"mr":{"score":0.155,"sig":"stretched_bull","src":"inferred"},"summary":{"flow":{"bias":"SELL","type":"mixed","actor":"retail","conf":"M","note":"Fluxo misto sem dominância clara. (varejo vendedor). — divergência 1m vs 5m detectada. [imbalance extremo de venda]","reversal_signal":1},"sr":{"nearest":"resistance","compressed":0,"conf_bias":"NEUTRAL","note":"Resistência mais próxima em 65037 (força 55, confluência 3 fontes, dist 148 pts (0.7 ATR))","r1_dist_atr":0.67,"s1_dist_atr":1.85},"regime":{"label":"Mean Reversion","strategies":["fade extremos","comprar suporte","aguardar absorção nos extremos"],"avoid":["perseguir momentum","entrar no meio do range","operar breakouts sem confirmação"],"duration":"15m – 2h tipicamente","note":"Regime Mean Reversion com consenso de alta (confiança alta: 80%). dominado pelo 4h. Favorece reversão para equilíbrio — não perseguir altas."},"institutional":{"auction_state":"Leilão incompleto em ambos os extremos","whale_bias":"NEUTRAL","profile_bias":"NEUTRAL","unfinished":["low","high"],"alignment":"NEUTRAL","note":"Leilão incompleto em ambos os extremos. Extremo(s) incompleto(s): low e high — reteste esperado. Risco de breakout da Value Area muito alto."},"quality":{"reliable":1,"confidence_cap":1,"note":"Dados em tempo real sem anomalias. Análise com confiança plena."}},"tipo_evento":"ANALYSIS_TRIGGER","descricao":"Evento automático para análise da IA"},"epoch_ms":1786141988998,"event_id":"cdefe2ce","data_context":"real_time","timestamp_utc":"2026-08-07T22:33:08.998+00:00","timestamp_ny":"2026-08-07T18:33:08.998-04:00","timestamp_sp":"2026-08-07T19:33:08.998-03:00","timestamp":"2026-08-07T22:33:08.998+00:00"}
{"tipo_evento":"ANALYSIS_TRIGGER","symbol":"BTCUSDT","epoch_ms":1786142040000,"janela_numero":4,"event_id":"fd9eb5bd","timestamp_utc":"2026-08-07T22:34:00.000Z","note":"trimmed_by_guardian"}
{"tipo_evento":"ANALYSIS_TRIGGER","symbol":"BTCUSDT","epoch_ms":1786142100000,"janela_numero":5,"event_id":"9692644e","timestamp_utc":"2026-08-07T22:35:00.000Z","note":"trimmed_by_guardian"}
{"tipo_evento":"ANALYSIS_TRIGGER","symbol":"BTCUSDT","epoch_ms":1786142160000,"janela_numero":6,"event_id":"bc8b572b","timestamp_utc":"2026-08-07T22:36:00.000Z","note":"trimmed_by_guardian"}
{"tipo_evento":"ANALYSIS_TRIGGER","symbol":"BTCUSDT","epoch_ms":1786142220000,"janela_numero":7,"event_id":"894cbe83","timestamp_utc":"2026-08-07T22:37:00.000Z","note":"trimmed_by_guardian"}
{"tipo_evento":"ANALYSIS_TRIGGER","symbol":"BTCUSDT","epoch_ms":1786142280000,"janela_numero":8,"event_id":"46907d43","timestamp_utc":"2026-08-07T22:38:00.000Z","note":"trimmed_by_guardian"}
{"tipo_evento":"AI_ANALYSIS","symbol":"BTCUSDT","timestamp_ms":1786142280000,"anchor_price":64846,"anchor_window_id":8,"ai_result":{"sentiment":"bearish","confidence":0.66,"action":"wait","rationale":"Os 15m apontam queda com fluxo de venda intenso e desequilíbrio de ordem negativo, enquanto o 4h ainda indica tendência de alta, sugerindo uma retração dentro d","region_type":"retração","_is_fallback":0,"_is_valid":1},"ai_payload":{"symbol":"BTCUSDT","epoch_ms":1786142280000,"trigger":"AT","price":{"c":64846,"o":64873,"h":64873,"vw":64863,"sh":"P","auc":"expect_retest_high","ph":1,"brk_risk":"V_HI"},"regime":{"cs":"BULL","cf":0.8,"v":"NOR","mode":"BRK","dom":"4h","bull%":90,"bear%":10},"qual":{"lat":"POOR","ms":11450},"flow":{"d1":"-221K","delta":-8.128,"vol":8.708,"buy_pct":3,"ti":20.6,"trs":-223,"obs":-2.838,"sf_w":-4,"sf_r":-18.859,"d5":"-943K","d15":"-1.7M","cvd":-25.8,"imb":-0.93,"ab":3,"bsr":0.03,"pa":"sell_absor","conv":"M","abs_buy_str":0.3,"abs_sell_exh":9.3,"abs_cont":0.33},"ob":{"b":"3.6M","a":"308K","imb":0.84,"bias":"BUY","t5":0.87,"spread_pct":0,"slip_b":5,"slip_s":5},"tf":{"15m":{"t":"DN","rsi":38,"macd":[13,17],"adx":24,"atr":70,"r":"RNG"},"1h":{"t":"UP","rsi":53,"macd":[100,98],"adx":24,"atr":224,"r":"RNG"},"4h":{"t":"UP","rsi":62,"macd":[275,257],"adx":30,"atr":495,"r":"RNG"},"1d":{"t":"UP","rsi":59,"macd":[71,37],"adx":15,"atr":1370,"r":"MNP"}},"sr":{"r1":[65030,54],"r1_dist":184,"r1_conf":3,"r2":[65346,53],"r2_dist":500,"r2_conf":3,"s1":[64480,62],"s1_dist":366,"s1_conf":4,"s2":[64823,62],"s2_dist":23,"s2_conf":4,"def_bias":"slight_sel"},"w":{"s":-37,"c":"MD"},"ext":{"cci":"OB","stoch":0,"stoch_sig":"OS","wr":-100,"garch":0,"hurst":0.33,"entropy":2.93,"fd":0.62,"kalman":{"kp":64933.92,"dev":-0.094,"dir":"DOWN"},"reg":{"sl":-1.1398,"pos":0.1289,"dev":-24.4},"mc":{"pu":0.466,"p10":64785.73,"p90":64904.98},"cycles":[100,40],"smc":{"fvg":1,"fvg_last":"BE","struct":"BEAR","bos":0}},"ctx":{"ses":"NY","fg":29,"poc":65046,"val":64522,"vah":65346,"lsr":1.1,"eth_lsr":2.07,"oi":107,"fr":0.0063,"longs":"+3631.4M","shorts":"+3302.8M","eth7":0.8,"dxy30":-0.06},"ofi":{"score":-0.933,"dir":"SELL","src":"order_flow"},"vwap":{"dev":-0.027,"side":"below","sig":"fair","src":"ohlc"},"iceberg":{"det":1,"src":"whale_activity"},"liq":[{"p":64867,"side":"sell","vol":10.58}],"cvd_div":{"det":1,"type":"bearish_div","src":"inferred"},"mr":{"score":0.248,"sig":"stretched_bear","src":"inferred"},"summary":{"flow":{"bias":"SELL","type":"mixed","actor":"retail","conf":"M","note":"Fluxo misto sem dominância clara. (varejo vendedor). prob. continuação 33%. [imbalance extremo de venda]"},"sr":{"nearest":"resistance","compressed":0,"conf_bias":"NEUTRAL","note":"Resistência mais próxima em 65030 (força 54, confluência 3 fontes, dist 184 pts (0.8 ATR))","r1_dist_atr":0.82,"s1_dist_atr":1.63},"regime":{"label":"Breakout","strategies":["aguardar confirmação de rompimento","entrar no reteste do nível rompido","usar stop apertado acima/abaixo do nível","monitorar volume de confirmação"],"avoid":["entrar antes da confirmação","ignorar falsos rompimentos","operar range enquanto houver BRK ativo"],"duration":"minutos a horas — confirmar rápido","note":"Regime Breakout com consenso de alta (confiança alta: 80%). dominado pelo 4h."},"institutional":{"auction_state":"Leilão incompleto — máxima deve ser revisitada","whale_bias":"DISTRIBUTING","profile_bias":"NEUTRAL","unfinished":["high"],"alignment":"BEAR_ALIGNED","note":"Leilão incompleto — máxima deve ser revisitada. Extremo(s) incompleto(s): high — reteste esperado. Risco de breakout da Value Area muito alto. Sinais institucionais alinhados para baixa."},"quality":{"reliable":1,"confidence_cap":1,"note":"Dados em tempo real sem anomalias. Análise com confiança plena."}},"tipo_evento":"ANALYSIS_TRIGGER","descricao":"Evento automático para análise da IA"},"epoch_ms":1786142293862,"event_id":"85c52eed","data_context":"real_time","timestamp_utc":"2026-08-07T22:38:13.862+00:00","timestamp_ny":"2026-08-07T18:38:13.862-04:00","timestamp_sp":"2026-08-07T19:38:13.862-03:00","timestamp":"2026-08-07T22:38:13.862+00:00"}
{"tipo_evento":"ANALYSIS_TRIGGER","symbol":"BTCUSDT","epoch_ms":1786142340000,"janela_numero":9,"event_id":"8b08362f","timestamp_utc":"2026-08-07T22:39:00.000Z","note":"trimmed_by_guardian"}
{"tipo_evento":"ANALYSIS_TRIGGER","symbol":"BTCUSDT","epoch_ms":1786142400000,"janela_numero":10,"event_id":"ab1845a1","timestamp_utc":"2026-08-07T22:40:00.000Z","note":"trimmed_by_guardian"}
{"tipo_evento":"ANALYSIS_TRIGGER","symbol":"BTCUSDT","epoch_ms":1786142460000,"janela_numero":11,"event_id":"8a04173a","timestamp_utc":"2026-08-07T22:41:00.000Z","note":"trimmed_by_guardian"}
{"tipo_evento":"Alerta","resultado_da_batalha":"VOLATILITY_SQUEEZE","descricao":"Tipo: VOLATILITY_SQUEEZE","timestamp":"2026-08-07T22:41:22+00:00","severity":"LOW","probability":0.4,"action":"WATCH_FOR_EXPANSION","context":{"price":64872,"volume":0.396,"average_volume":4.046,"volatility":6e-08},"data_context":"real_time","janela_numero":11,"epoch_ms":1786142482499,"event_id":"a2e84d90","timestamp_utc":"2026-08-07T22:41:22.499+00:00","timestamp_ny":"2026-08-07T18:41:22.499-04:00","timestamp_sp":"2026-08-07T19:41:22.499-03:00","price_data":{"current":{"last":64872,"volume":0.396}},"volatility_metrics":{"realized_vol_24h":0},"market_context":{"trading_session":"NY_OVERLAP","session_phase":"ACTIVE"}}
{"is_signal":1,"tipo_evento":"Exaustão","resultado_da_batalha":"Exaustão de Venda","descricao":"Pico de venda 11.11 vs média 4.05","ativo":"BTCUSDT","window_open_ms":1786142460293,"window_close_ms":1786142519917,"window_duration_ms":59624,"window_id":1786142519917,"volume_total_btc":11.105,"volume_compra_btc":2.746,"volume_venda_btc":8.359,"buy_notional_usdt":178096.2612,"sell_notional_usdt":542212.7234,"total_notional_usdt":720308.9847,"volume_total":11.105,"volume_compra":2.746,"volume_venda":8.359,"preco_abertura":64872,"preco_maxima":64872,"preco_minima":64846,"preco_fechamento":64856.34,"ohlc":{"open":64872,"high":64872,"low":64846,"close":64856.34},"delta_minimo":-8.094,"delta_maximo":0.026,"delta_fechamento":-5.613,"reversao_desde_minimo":2.4813,"reversao_desde_maximo":5.6381,"dwell_price":64871.35,"dwell_seconds":39.379,"dwell_location":"High","trades_per_second":26.181,"avg_trade_size":0.007,"layer":"signal","data_context":"real_time","source":{"exchange":"binance_futures","stream":"trades"},"poc_price":64871.278,"vah":64871.278,"val":64861.1665,"hvns":[64871.278],"lvns":[64846.7215,64848.1665,64851.0555,64852.5,64853.9445,64855.389,64856.8335,64858.278,64859.722,64861.1665,64862.611,64864.0555,64865.5,64868.389,64869.8335],"vpd_params":{"dynamic_bins":18,"value_area_pct":0.65,"hvn_sensitivity":1.5,"lvn_sensitivity":1.5,"volatility_factor":0.5,"whale_factor":1.4502,"trend_factor":1.3},"fluxo_continuo":{"cvd":-28.5699,"whale_buy_volume":0,"whale_sell_volume":5,"whale_delta":-5,"bursts":{"count":8,"max_burst_volume":7.558},"sector_flow":{"retail":{"buy":12.0048,"sell":32.5694,"delta":-20.565},"mid":{"buy":1.522,"sell":4.5274,"delta":-3.005},"whale":{"buy":0,"sell":5,"delta":-5}},"timestamp":"2026-08-07T22:42:00.000+00:00","time_index":{"epoch_ms":1786142520000,"timestamp_utc":"2026-08-07T22:42:00.000+00:00","timestamp_ny":"2026-08-07T18:42:00.000-04:00","timestamp_sp":"2026-08-07T19:42:00.000-03:00"},"metadata":{"burst_window_ms":200,"in_burst":0,"last_reset_ms":1786141813933,"config_version":1,"num_trades":1574,"window_sec":60},"order_flow":{"net_flow_1m":160810.2968,"absorcao_1m":"Neutra","buy_volume":178096.26,"sell_volume":542212.72,"total_volume":720308.98,"ui_sum_ok":1,"buy_volume_btc":2.746,"sell_volume_btc":8.359,"total_volume_btc":11.105,"whale_buy_volume_window":0,"whale_sell_volume_window":1,"whale_delta_window":-1,"flow_imbalance":-0.5055,"aggressive_buy_pct":24.72,"aggressive_sell_pct":75.28,"net_flow_5m":-196915.5833,"absorcao_5m":"Neutra","net_flow_15m":-1854091.4037,"absorcao_15m":"Neutra","computation_window_min":1,"available_windows_min":[1,5,15],"buy_sell_ratio":{"buy_sell_ratio":0.33,"ratios":{"current":0.3285,"imbalance_1m":0.223,"imbalance_5m":-0.273,"imbalance_15m":-2.574},"sector_ratios":{"retail":0.3686,"mid":0.3362,"whale":0},"pressure":"STRONG_SELL","flow_trend":"short_term_reversal_to_buy","buy_volume":2.746,"sell_volume":8.359}},"tipo_absorcao":"Neutra","participant_analysis":{"retail":{"volume_pct":85.9,"direction":"SELL","sentiment":"BEARISH","composite_score":-0.713,"imbalance":-0.424},"mid":{"volume_pct":5.09,"direction":"SELL","sentiment":"BEARISH","composite_score":-0.421,"imbalance":-1},"whale":{"volume_pct":9,"direction":"SELL","sentiment":"BEARISH","composite_score":-0.436,"imbalance":-1}},"liquidity_heatmap":{"clusters":[{"center":64862.4746,"low":64846,"high":64872,"width":26,"total_volume":13.606,"buy_volume":3.607,"sell_volume":9.999,"imbalance":-6.391,"imbalance_ratio":-0.47,"trades_count":2000,"avg_trade_size":0.007,"recent_timestamp":1786142526893,"recent_ts_ms":1786142526893,"last_seen_ms":1786142526893,"first_seen_ms":1786142315761,"age_ms":265.0,"cluster_duration_ms":211132,"price_std":10.0286,"volume_std":0.037,"bin_threshold_usd":194.5874}],"resistances":[64862.4746],"clusters_count":1},"absorption_analysis":{"current_absorption":{"index":0.1129,"classification":"WEAK_ABSORPTION","label":"Neutra","buyer_strength":2.5,"seller_exhaustion":5.1,"continuation_probability":0.1,"delta_usd":160810.297,"total_volume_usd":720308.98,"flow_imbalance":-0.5055,"window_min":1}},"data_quality":{"total_trades_processed":7554,"invalid_trades":0,"valid_rate_pct":100,"flow_trades_count":1574,"processing_time_ms":8.323100000325212},"observability":{"processing_times_ms":{"p50":0.05870000040886225,"p95":0.12250000008862116,"p99":0.20599999970727367,"max":0.3475999997135659,"min":0.04579999995257822,"avg":0.07280430000537308,"count":1000,"total_recorded":7554},"memory":{"flow_trades_size":7554,"flow_trades_capacity":100000},"circuit_breaker":{"state":"CLOSED","failures":0,"successes":7554,"time_in_state_ms":713639,"recovery_remaining_ms":0,"threshold":5}},"invariants_ok":1},"historical_vp":{"daily":{"poc":65046,"vah":65346,"val":64522,"hvns":[64304,64306,64314,64320,64322,64354,64376,64394,64425,64512,64810,64815,64858,64873,64880,64909,64918,64926,64935,64940,64993,65004,65009,65015,65026,65033,65044,65046,65080,65113,65114,65127,65129,65149,65154,65157,65166,65182,65190,65196,65209,65213,65214,65223,65231,65232,65252,65255,65300,65301,65302,65346],"lvns":[64179,64180,64182,64183,64187,64190,64195,64208,64212,64213,64219,64220,64223,64225,64230,64246,64248,64251,64252,64255,64257,64258,64262,64266,64274,64275,64278,64280,64283,64285,64287,64295,64297,64300,64305,64307,64309,64313,64327,64331,64339,64348,64355,64356,64361,64363,64366,64379,64380,64381,64382,64384,64386,64390,64391,64396,64399,64404,64423,64436,64440,64447,64448,64450,64460,64462,64480,64485,64500,64516,64523,64598,64658,64692,64722,64745,64766,64775,64816,64843,64857,64861,64862,64883,64887,64903,64911,64947,64964,64976,64978,64991,65001,65008,65012,65022,65077,65090,65103,65121,65215],"single_prints":[64223,64255,64266,64297,64305,64307,64309,64313,64321,64327,64339,64348,64361,64366,64386,64399,64423,64431,64480,64485,64516,64598,64658,64722,64812,64816,64843,64857,64903,64911,64954,64964,65025,65045,65121,65200],"volume_nodes":{"hvn_nodes":[[64304.0,67.73988,3.06],[64306.0,53.94228,2.44],[64314.0,53.796510000000005,2.43],[64320.0,60.9738,2.76],[64322.0,72.71476000000001,3.29],[64354.0,40.32035,1.82],[64376.0,71.47123,3.23],[64394.0,40.0858,1.81],[64425.0,49.071,2.22],[64512.0,44.82474,2.03],[64810.0,41.61074,1.88],[64815.0,61.08252,2.76],[64858.0,38.58834,1.75],[64873.0,134.24998,6.07],[64880.0,90.04688,4.07],[64909.0,78.25269,3.54],[64918.0,46.53939,2.11],[64926.0,50.178749999999994,2.27],[64935.0,98.15797,4.44],[64940.0,62.34289,2.82],[64993.0,41.937740000000005,1.9],[65004.0,47.78778,2.16],[65009.0,67.11474,3.04],[65015.0,39.15253,1.77],[65026.0,52.98003,2.4],[65033.0,65.19254,2.95],[65044.0,38.59124,1.75],[65046.0,221.02348999999998,10.0],[65080.0,42.25639,1.91],[65113.0,41.04726,1.86],[65114.0,48.194379999999995,2.18],[65127.0,57.41659,2.6],[65129.0,58.98461,2.67],[65149.0,101.77942,4.6],[65154.0,45.32618,2.05],[65157.0,66.65561,3.02],[65166.0,54.13736,2.45],[65182.0,52.497510000000005,2.38],[65190.0,39.62336,1.79],[65196.0,64.47973,2.92],[65209.0,69.27777,3.13],[65213.0,70.67875,3.2],[65214.0,56.5937,2.56],[65223.0,74.28292,3.36],[65231.0,64.74432,2.93],[65232.0,49.31932,2.23],[65252.0,39.63037,1.79],[65255.0,51.04381,2.31],[65300.0,66.03468000000001,2.99],[65301.0,39.67111,1.79],[65302.0,46.37914,2.1],[65346.0,106.18444,4.8]],"lvn_nodes":[[64179.0,4.24474,9.81],[64180.0,3.57926,9.84],[64182.0,3.93413,9.82],[64183.0,3.2761,9.85],[64187.0,2.65409,9.88],[64190.0,3.81962,9.83],[64195.0,3.79467,9.83],[64208.0,3.58096,9.84],[64212.0,3.2321,9.85],[64213.0,2.06124,9.91],[64219.0,2.99334,9.86],[64220.0,4.7497799999999994,9.79],[64223.0,1.25377,9.94],[64225.0,4.80321,9.78],[64230.0,2.07791,9.91],[64246.0,4.45712,9.8],[64248.0,4.68392,9.79],[64251.0,2.9631999999999996,9.87],[64252.0,3.84494,9.83],[64255.0,1.54673,9.93],[64257.0,3.76443,9.83],[64258.0,3.39069,9.85],[64262.0,3.82717,9.83],[64266.0,1.58008,9.93],[64274.0,3.04509,9.86],[64275.0,4.49153,9.8],[64278.0,3.77121,9.83],[64280.0,4.16666,9.81],[64283.0,1.95633,9.91],[64285.0,3.97216,9.82],[64287.0,3.73426,9.83],[64295.0,5.00548,9.77],[64297.0,1.31265,9.94],[64300.0,5.24599,9.76],[64305.0,3.44122,9.84],[64307.0,1.23179,9.94],[64309.0,0.87349,9.96],[64313.0,2.66159,9.88],[64327.0,0.37452,9.98],[64331.0,4.25856,9.81],[64339.0,1.99728,9.91],[64348.0,3.22897,9.85],[64355.0,3.81043,9.83],[64356.0,1.80703,9.92],[64361.0,1.68926,9.92],[64363.0,5.2804,9.76],[64366.0,3.22755,9.85],[64379.0,4.31004,9.8],[64380.0,5.408720000000001,9.76],[64381.0,3.79399,9.83],[64382.0,3.95661,9.82],[64384.0,4.48611,9.8],[64386.0,2.59853,9.88],[64390.0,3.76553,9.83],[64391.0,4.80344,9.78],[64396.0,3.69423,9.83],[64399.0,1.22663,9.94],[64404.0,3.26433,9.85],[64423.0,2.76864,9.87],[64436.0,3.07818,9.86],[64440.0,3.07764,9.86],[64447.0,3.48233,9.84],[64448.0,3.40636,9.85],[64450.0,5.24821,9.76],[64460.0,2.27524,9.9],[64462.0,4.40812,9.8],[64480.0,1.65009,9.93],[64485.0,1.33693,9.94],[64500.0,5.05463,9.77],[64516.0,2.90787,9.87],[64523.0,4.3961,9.8],[64598.0,2.47809,9.89],[64658.0,2.83861,9.87],[64692.0,4.57211,9.79],[64722.0,2.44665,9.89],[64745.0,4.53843,9.79],[64766.0,3.34199,9.85],[64775.0,4.6803,9.79],[64816.0,1.00341,9.95],[64843.0,4.86568,9.78],[64857.0,3.70827,9.83],[64861.0,2.0873,9.91],[64862.0,2.10681,9.9],[64883.0,5.38641,9.76],[64887.0,3.47893,9.84],[64903.0,5.41496,9.76],[64911.0,2.83296,9.87],[64947.0,3.2699,9.85],[64964.0,1.97222,9.91],[64976.0,5.37939,9.76],[64978.0,3.04878,9.86],[64991.0,3.6763,9.83],[65001.0,2.57905,9.88],[65008.0,5.49886,9.75],[65012.0,4.59123,9.79],[65022.0,4.61114,9.79],[65077.0,5.05843,9.77],[65090.0,5.45375,9.75],[65103.0,3.8176,9.83],[65121.0,3.82624,9.83],[65215.0,4.62879,9.79]]},"status":"success"},"weekly":{"poc":62610,"vah":63567,"val":62314,"hvns":[62399,62418,62598,62600,62610,62621,62635,62640,62702,62706,62835,62878,62930,62942,62954,62965,62982,62986,62992,63000,63080,63082,63085,63093,63100,63110,63115,63116,63133,63152,63175,63243,63321,63342,63351,63394,63405,63425,63428,63460,63462,63494,63496,63502,63506,63520,63526,63530,63532,63543,63550,63567,63620,63685,63704,63724,63740,63751,63760,63766,63774,63783,63792,63800,63802,63807,63808,63810,63816,63818,63820,63821,63823,63828,63843,63853,63880,63884,63886,63967,63994,64162],"lvns":[62506,62515,62524,62530,62550,62558,62578,62596,62630,62646,62675,62687,62692,62708,62709,62728,62765,62771,62778,62788,62799,62808,62813,62815,62819,62822,62831,62833,62842,62844,62849,62855,62858,62868,62922,62928,62938,62941,62951,62956,62962,62967,62973,62985,62988,62989,62996,63002,63010,63018,63019,63027,63043,63047,63053,63055,63067,63079,63081,63083,63094,63095,63099,63102,63109,63113,63123,63125,63130,63136,63140,63142,63146,63156,63160,63171,63187,63197,63202,63208,63236,63242,63267,63278,63286,63290,63310,63333,63334,63359,63375,63382,63427,63431,63435,63437,63443,63449,63453,63454,63455,63471,63510,63517,63525,63534,63536,63570,63588,63635,63647,63667,63690,63716,63778,63805,63825,63862],"single_prints":[62506,62550,62558,62578,62596,62630,62743,62771,62778,62833,62928,62938,62941,62956,62996,63010,63047,63079,63081,63083,63109,63113,63125,63136,63156,63171,63197,63202,63208,63236,63242,63262,63310,63375,63427,63468,63471,63489,63510,63517,63525,63534,63647,63690,63801,63805,63822,63825,63882,63885],"volume_nodes":{"hvn_nodes":[[62399.0,294.0842,7.51],[62418.0,128.04348,3.27],[62598.0,138.68552,3.54],[62600.0,249.31916999999999,6.37],[62610.0,391.61894,10.0],[62621.0,132.8681,3.39],[62635.0,130.53661,3.33],[62640.0,200.77168,5.13],[62702.0,140.06781,3.58],[62706.0,147.41717,3.76],[62835.0,232.32138,5.93],[62878.0,115.05104,2.94],[62930.0,274.81345,7.02],[62942.0,121.04401,3.09],[62954.0,207.38574,5.3],[62965.0,117.56890000000001,3.0],[62982.0,298.54864,7.62],[62986.0,113.17351,2.89],[62992.0,117.21755999999999,2.99],[63000.0,212.59551,5.43],[63080.0,213.15154,5.44],[63082.0,178.86463,4.57],[63085.0,118.8479,3.03],[63093.0,154.94474,3.96],[63100.0,202.53757,5.17],[63110.0,110.50667,2.82],[63115.0,126.95239000000001,3.24],[63116.0,164.00229000000002,4.19],[63133.0,129.26478,3.3],[63152.0,124.84576,3.19],[63175.0,153.45308,3.92],[63243.0,136.26963,3.48],[63321.0,153.14067,3.91],[63342.0,160.6609,4.1],[63351.0,154.70452,3.95],[63394.0,339.08465,8.66],[63405.0,110.79841,2.83],[63425.0,110.65537,2.83],[63428.0,296.327,7.57],[63460.0,117.50828,3.0],[63462.0,164.40156,4.2],[63494.0,223.82197000000002,5.72],[63496.0,202.61972,5.17],[63502.0,173.24966,4.42],[63506.0,217.70113,5.56],[63520.0,130.48484,3.33],[63526.0,114.08404999999999,2.91],[63530.0,252.12511999999998,6.44],[63532.0,208.66274,5.33],[63543.0,129.59888,3.31],[63550.0,136.52242999999999,3.49],[63567.0,260.76083,6.66],[63620.0,143.02633,3.65],[63685.0,117.49553,3.0],[63704.0,124.31002,3.17],[63724.0,132.0072,3.37],[63740.0,191.88948000000002,4.9],[63751.0,351.59321,8.98],[63760.0,135.92831,3.47],[63766.0,385.92627,9.85],[63774.0,115.91514000000001,2.96],[63783.0,184.86056,4.72],[63792.0,132.65683,3.39],[63800.0,162.23109,4.14],[63802.0,147.22476,3.76],[63807.0,124.32736,3.17],[63808.0,133.92018,3.42],[63810.0,142.04885000000002,3.63],[63816.0,228.01736,5.82],[63818.0,139.10197,3.55],[63820.0,195.84261,5.0],[63821.0,150.26853,3.84],[63823.0,168.58154,4.3],[63828.0,117.00525,2.99],[63843.0,146.45026,3.74],[63853.0,152.85647,3.9],[63880.0,191.22533,4.88],[63884.0,212.5999,5.43],[63886.0,203.08616999999998,5.19],[63967.0,240.55014,6.14],[63994.0,161.9911,4.14],[64162.0,249.07463,6.36]],"lvn_nodes":[[62506.0,15.06991,9.62],[62515.0,8.51603,9.78],[62524.0,10.80844,9.72],[62530.0,15.5374,9.6],[62550.0,16.39797,9.58],[62558.0,10.04577,9.74],[62578.0,6.06561,9.85],[62596.0,8.37368,9.79],[62630.0,6.28167,9.84],[62646.0,13.35453,9.66],[62675.0,12.59401,9.68],[62687.0,14.76282,9.62],[62692.0,16.13337,9.59],[62708.0,11.99753,9.69],[62709.0,12.23336,9.69],[62728.0,14.5949,9.63],[62765.0,12.88563,9.67],[62771.0,4.98646,9.87],[62778.0,6.59977,9.83],[62788.0,13.95636,9.64],[62799.0,15.6845,9.6],[62808.0,11.65946,9.7],[62813.0,11.5303,9.71],[62815.0,7.71603,9.8],[62819.0,12.27771,9.69],[62822.0,9.22183,9.76],[62831.0,15.30756,9.61],[62833.0,4.23598,9.89],[62842.0,10.97243,9.72],[62844.0,9.62836,9.75],[62849.0,16.0898,9.59],[62855.0,14.15592,9.64],[62858.0,10.93725,9.72],[62868.0,10.62331,9.73],[62922.0,11.09328,9.72],[62928.0,4.97574,9.87],[62938.0,15.75122,9.6],[62941.0,13.72219,9.65],[62951.0,7.98267,9.8],[62956.0,12.97513,9.67],[62962.0,10.38756,9.73],[62967.0,13.81257,9.65],[62973.0,12.40873,9.68],[62985.0,8.94355,9.77],[62988.0,11.32377,9.71],[62989.0,6.19079,9.84],[62996.0,12.18857,9.69],[63002.0,9.22411,9.76],[63010.0,5.49879,9.86],[63018.0,15.81364,9.6],[63019.0,15.72474,9.6],[63027.0,16.13079,9.59],[63043.0,11.46813,9.71],[63047.0,9.1053,9.77],[63053.0,13.21409,9.66],[63055.0,12.07385,9.69],[63067.0,13.47955,9.66],[63079.0,14.0804,9.64],[63081.0,8.07513,9.79],[63083.0,13.77433,9.65],[63094.0,9.06442,9.77],[63095.0,9.31716,9.76],[63099.0,10.58062,9.73],[63102.0,13.536460000000002,9.65],[63109.0,9.65566,9.75],[63113.0,9.25243,9.76],[63123.0,16.26607,9.58],[63125.0,8.92207,9.77],[63130.0,8.19909,9.79],[63136.0,4.85736,9.88],[63140.0,14.88344,9.62],[63142.0,7.55169,9.81],[63146.0,13.33089,9.66],[63156.0,9.22189,9.76],[63160.0,9.50751,9.76],[63171.0,7.96165,9.8],[63187.0,10.15174,9.74],[63197.0,8.70547,9.78],[63202.0,7.77674,9.8],[63208.0,10.95911,9.72],[63236.0,13.19127,9.66],[63242.0,12.14649,9.69],[63267.0,13.00713,9.67],[63278.0,11.08196,9.72],[63286.0,15.56654,9.6],[63290.0,13.27022,9.66],[63310.0,8.40344,9.79],[63333.0,12.76331,9.67],[63334.0,12.38795,9.68],[63359.0,13.83613,9.65],[63375.0,10.80791,9.72],[63382.0,13.90372,9.64],[63427.0,12.04044,9.69],[63431.0,8.77182,9.78],[63435.0,4.82027,9.88],[63437.0,14.91107,9.62],[63443.0,11.22618,9.71],[63449.0,10.28971,9.74],[63453.0,15.82072,9.6],[63454.0,12.32164,9.69],[63455.0,10.32202,9.74],[63471.0,12.92255,9.67],[63510.0,12.01509,9.69],[63517.0,11.12329,9.72],[63525.0,12.50591,9.68],[63534.0,8.31125,9.79],[63536.0,9.94024,9.75],[63570.0,15.43729,9.61],[63588.0,10.69317,9.73],[63635.0,13.08835,9.67],[63647.0,13.05105,9.67],[63667.0,13.70476,9.65],[63690.0,14.17659,9.64],[63716.0,9.76123,9.75],[63778.0,12.59372,9.68],[63805.0,15.71189,9.6],[63825.0,13.26533,9.66],[63862.0,10.59599,9.73]]},"status":"success"},"monthly":{"poc":63554,"vah":64214,"val":61789,"hvns":[62102,62318,62332,62500,62556,62568,62613,62618,62804,62813,62827,62828,62840,62932,62979,62985,63028,63039,63098,63100,63207,63391,63412,63554,63618,63732,63750,63796,63828,63830,63872,63890,63915,63918,63932,63933,63944,63959,63960,63962,63968,63983,63986,64008,64015,64024,64026,64033,64070,64090,64095,64100,64110,64112,64127,64144,64161,64162,64176,64200,64218,64234,64381,64499,64542,64601,64650,64705,64730,64734,64744,64768,64782,64788,64930,65038,65118,65164,65313,65380],"lvns":[62091,62211,62222,62290,62325,62330,62470,62573,62600,62645,62664,62690,62699,62714,62745,62841,62845,62910,63183,63229,63230,63256,63286,63290,63313,63317,63344,63376,63496,63800,63827,63841,63848,63865,63874,63887,63900,63910,63911,63926,63927,63929,63952,63967,63994,63995,63999,64001,64010,64016,64023,64046,64066,64098,64114,64119,64123,64126,64133,64134,64141,64142,64148,64150,64154,64159,64163,64168,64172,64180,64181,64190,64196,64198,64199,64207,64208,64223,64224,64231,64236,64264,64266,64270,64271,64272,64286,64323,64328,64353,64354,64356,64367,64382,64415,64448,64487,64491,64521,64607,64612,64616,64620,64642,64644,64654,64667,64694,64700,64706,64712,64720,64724,64776,64779,64794,64822,64827,64871,64880,64897,64914,64968,65010],"single_prints":[62714,62839,62841,62980,63183,63800,63874,63916,63929,63961,63967,63984,64016,64028,64092,64098,64114,64119,64123,64126,64139,64145,64163,64231,64236,64367,64382,64521,64776,64779,64783],"volume_nodes":{"hvn_nodes":[[62102.0,575.95733,3.29],[62318.0,860.81376,4.91],[62332.0,803.51943,4.59],[62500.0,574.93311,3.28],[62556.0,676.48671,3.86],[62568.0,459.3002,2.62],[62613.0,845.51977,4.83],[62618.0,531.49493,3.03],[62804.0,862.1336200000001,4.92],[62813.0,1462.2678,8.35],[62827.0,594.58482,3.39],[62828.0,593.17147,3.39],[62840.0,440.29827,2.51],[62932.0,505.36055,2.88],[62979.0,676.64333,3.86],[62985.0,454.94104,2.6],[63028.0,585.19921,3.34],[63039.0,625.2501,3.57],[63098.0,455.73539,2.6],[63100.0,1099.6865400000002,6.28],[63207.0,725.4131,4.14],[63391.0,445.05931,2.54],[63412.0,686.01088,3.92],[63554.0,1751.82343,10.0],[63618.0,561.31296,3.2],[63732.0,605.96554,3.46],[63750.0,885.29583,5.05],[63796.0,671.93807,3.84],[63828.0,1701.50539,9.71],[63830.0,481.65060000000005,2.75],[63872.0,656.2278699999999,3.75],[63890.0,698.42926,3.99],[63915.0,534.5503,3.05],[63918.0,767.68037,4.38],[63932.0,439.11439,2.51],[63933.0,458.51282,2.62],[63944.0,653.86347,3.73],[63959.0,510.0945,2.91],[63960.0,993.38154,5.67],[63962.0,1545.57447,8.82],[63968.0,584.0457,3.33],[63983.0,450.50937,2.57],[63986.0,522.29,2.98],[64008.0,986.83627,5.63],[64015.0,448.38387,2.56],[64024.0,491.27680999999995,2.8],[64026.0,462.05033000000003,2.64],[64033.0,1373.57785,7.84],[64070.0,619.87259,3.54],[64090.0,553.29849,3.16],[64095.0,724.6339800000001,4.14],[64100.0,913.9593,5.22],[64110.0,1089.3103099999998,6.22],[64112.0,801.9954799999999,4.58],[64127.0,1191.74837,6.8],[64144.0,1283.16102,7.32],[64161.0,1021.83717,5.83],[64162.0,560.45903,3.2],[64176.0,538.9005999999999,3.08],[64200.0,562.30178,3.21],[64218.0,481.80739,2.75],[64234.0,522.65958,2.98],[64381.0,438.95163,2.51],[64499.0,441.03357,2.52],[64542.0,634.28436,3.62],[64601.0,558.88683,3.19],[64650.0,711.15812,4.06],[64705.0,747.33687,4.27],[64730.0,479.09281,2.73],[64734.0,592.06674,3.38],[64744.0,1162.36379,6.64],[64768.0,586.32417,3.35],[64782.0,1147.23165,6.55],[64788.0,777.9184,4.44],[64930.0,803.55245,4.59],[65038.0,473.54582,2.7],[65118.0,490.01949,2.8],[65164.0,722.65086,4.13],[65313.0,570.80715,3.26],[65380.0,540.92679,3.09]],"lvn_nodes":[[62091.0,47.56872,9.73],[62211.0,60.48586,9.65],[62222.0,58.80519,9.66],[62290.0,40.82141,9.77],[62325.0,42.61666,9.76],[62330.0,31.65868,9.82],[62470.0,58.32244,9.67],[62573.0,44.59119,9.75],[62600.0,52.07317,9.7],[62645.0,58.61862,9.67],[62664.0,55.27942,9.68],[62690.0,58.73291,9.66],[62699.0,61.42336,9.65],[62714.0,56.64061,9.68],[62745.0,62.84646,9.64],[62841.0,55.98002,9.68],[62845.0,48.55009,9.72],[62910.0,56.67036,9.68],[63183.0,35.35897,9.8],[63229.0,54.73419,9.69],[63230.0,59.82032,9.66],[63256.0,38.27367,9.78],[63286.0,45.96775,9.74],[63290.0,64.22612,9.63],[63313.0,53.73242,9.69],[63317.0,47.78479,9.73],[63344.0,49.56841,9.72],[63376.0,46.93081,9.73],[63496.0,61.96078,9.65],[63800.0,51.30674,9.71],[63827.0,33.56492,9.81],[63841.0,52.08453,9.7],[63848.0,28.89956,9.84],[63865.0,52.62745,9.7],[63874.0,55.37344,9.68],[63887.0,56.61556,9.68],[63900.0,56.90165,9.68],[63910.0,32.36485,9.82],[63911.0,25.48414,9.85],[63926.0,43.89435,9.75],[63927.0,27.57956,9.84],[63929.0,31.95227,9.82],[63952.0,30.4872,9.83],[63967.0,57.98451,9.67],[63994.0,42.3891,9.76],[63995.0,61.42126,9.65],[63999.0,40.01549,9.77],[64001.0,54.8111,9.69],[64010.0,61.83091,9.65],[64016.0,34.50971,9.8],[64023.0,57.56183,9.67],[64046.0,63.57302,9.64],[64066.0,57.63537,9.67],[64098.0,51.60599,9.71],[64114.0,52.29812,9.7],[64119.0,29.2789,9.83],[64123.0,38.45855,9.78],[64126.0,28.95436,9.83],[64133.0,30.67605,9.82],[64134.0,56.16519,9.68],[64141.0,35.80722,9.8],[64142.0,63.27429,9.64],[64148.0,26.18415,9.85],[64150.0,34.85346,9.8],[64154.0,43.50225,9.75],[64159.0,52.59504,9.7],[64163.0,43.86638,9.75],[64168.0,30.12177,9.83],[64172.0,28.51267,9.84],[64180.0,63.66604,9.64],[64181.0,43.91014,9.75],[64190.0,51.1383,9.71],[64196.0,53.54202,9.69],[64198.0,56.94656,9.67],[64199.0,40.67753,9.77],[64207.0,47.77097,9.73],[64208.0,33.20408,9.81],[64223.0,56.30652,9.68],[64224.0,47.74338,9.73],[64231.0,51.26858,9.71],[64236.0,47.81361,9.73],[64264.0,44.39749,9.75],[64266.0,31.93245,9.82],[64270.0,27.09318,9.85],[64271.0,52.15961,9.7],[64272.0,63.68604,9.64],[64286.0,31.09328,9.82],[64323.0,57.84002,9.67],[64328.0,39.83597,9.77],[64353.0,55.11732,9.69],[64354.0,29.91619,9.83],[64356.0,23.80451,9.86],[64367.0,30.2385,9.83],[64382.0,25.77959,9.85],[64415.0,63.86359,9.64],[64448.0,46.88938,9.73],[64487.0,50.28309,9.71],[64491.0,39.66976,9.77],[64521.0,37.19094,9.79],[64607.0,54.89064,9.69],[64612.0,58.63468,9.67],[64616.0,45.15167,9.74],[64620.0,47.93147,9.73],[64642.0,64.26272,9.63],[64644.0,57.28331,9.67],[64654.0,49.55733,9.72],[64667.0,47.124,9.73],[64694.0,53.87083,9.69],[64700.0,31.37178,9.82],[64706.0,35.30312,9.8],[64712.0,57.1625,9.67],[64720.0,26.92108,9.85],[64724.0,60.79525,9.65],[64776.0,55.2937,9.68],[64779.0,20.34292,9.88],[64794.0,62.80996,9.64],[64822.0,62.4548,9.64],[64827.0,43.72451,9.75],[64871.0,52.90181,9.7],[64880.0,52.70478,9.7],[64897.0,44.58118,9.75],[64914.0,45.52568,9.74],[64968.0,50.61019,9.71],[65010.0,36.97205,9.79]]},"status":"success"}},"timestamp":"2026-08-07T22:42:00.000Z","epoch_ms":1786142520000,"timestamp_utc":"2026-08-07T22:42:00.000Z","timestamp_ny":"2026-08-07T18:42:00.000-04:00","timestamp_sp":"2026-08-07T19:42:00.000-03:00","event_id":"1b84c3d82fa62cbac78d6789338a640d8c527cf2f0432ca5f569cad2af3a88b8","trades_count":1561,"duration_s":59.62,"tick_context_out":{"last_price":64856.34,"last_m":0},"delta":-5.613,"multi_tf":{"15m":{"tendencia":"Baixa","preco_atual":64871.99,"mme_21":64925.41,"atr":73.63,"regime":"Range","rsi_short":37.88,"rsi_long":44.06,"macd":12.6535,"macd_signal":16.6562,"adx":23.83,"realized_vol":0.0014},"1h":{"tendencia":"Alta","preco_atual":64871.99,"mme_21":64788.8,"atr":227.14,"regime":"Range","rsi_short":52.6,"rsi_long":54.35,"macd":100.3548,"macd_signal":98.3005,"adx":23.82,"realized_vol":0.0023},"4h":{"tendencia":"Alta","preco_atual":64872,"mme_21":64460.35,"atr":498.88,"regime":"Range","rsi_short":61.94,"rsi_long":61.09,"macd":274.9477,"macd_signal":256.7109,"adx":30.01,"realized_vol":0.0058},"1d":{"tendencia":"Alta","preco_atual":64872,"mme_21":64144.77,"atr":1369.69,"regime":"Manipulação","rsi_short":58.86,"rsi_long":55.63,"macd":67.1517,"macd_signal":35.9228,"adx":15.05,"realized_vol":0.0184}},"pattern_recognition":{"smart_money":{"fair_value_gaps":[{"type":"BEARISH","top":64925,"bottom":64892.3,"gap_size":32.7,"gap_pct":0.05}],"market_structure":{"structure":"BEARISH","bos_detected":0,"last_swing_high":64925,"last_swing_low":64909.55,"higher_highs":0,"higher_lows":0}}},"janela_numero":12,"whale_sell_volume":5,"derivatives":{"BTCUSDT":{"funding_rate_percent":0.01,"open_interest":106945.765,"open_interest_usd":6934687958.97,"long_short_ratio":1.1,"longs_usd":3631669164.51,"shorts_usd":3303018794.46},"ETHUSDT":{"funding_rate_percent":0,"open_interest":2279164.179,"open_interest_usd":4360633657.11,"long_short_ratio":2.07,"longs_usd":2939074192.69,"shorts_usd":1421559464.42}},"market_context":{"trading_session":"NY","session_phase":"ACTIVE","time_to_session_close":11964,"day_of_week":4,"is_holiday":0,"market_hours_type":"EXTENDED"},"market_environment":{"volatility_regime":"NORMAL","trend_direction":"UP","market_structure":"RANGE_BOUND","liquidity_environment":"NORMAL","risk_sentiment":"BULLISH","correlation_spy":0.4056,"correlation_dxy":-0.0854,"correlation_gold":0.2379},"features_window_id":1786142520000,"ml_features":{"price_features":{"returns_1":0.0,"volatility_1":0.0,"returns_5":1.5e-07,"volatility_5":1.2e-07,"returns_15":0.0,"volatility_15":1e-07,"momentum_score":-0.70710678,"volatility_1h":0.0023},"volume_features":{"volume_sma_ratio":0.739,"volume_momentum":-0.511,"buy_sell_pressure":-0.5054,"liquidity_gradient":-0.51112327},"microstructure":{"order_book_slope":-4.385968,"flow_imbalance":-0.5055,"tick_rule_sum":-94,"trade_intensity":40,"trade_intensity_v2":26.2333},"cross_asset":{"btc_eth_corr_7d":0.8212,"btc_eth_corr_30d":0.8599,"btc_dxy_corr_30d":-0.0605,"btc_dxy_corr_90d":-0.067,"btc_ndx_corr_30d":0.4147,"dxy_return_5d":-0.3561,"dxy_return_20d":-1.6548,"btc_dxy_correlation_stability":0.0065,"btc_dxy_inverse_strength":0.0638,"dxy_momentum":-1.29867555,"vix_current":14.9,"us10y_yield":4.66,"btc_dominance":17.0377,"btc_dominance_change_7d":0,"eth_dominance":9.0966,"gold_price":4342.3515,"oil_price":77.08,"macro_regime":"RISK_ON","correlation_regime":"DECORRELATED"},"data_quality":{"has_price_features":1,"has_volume_features":1,"has_microstructure":1,"has_cross_asset":1,"is_valid":1}},"contextual_snapshot":{"symbol":"BTCUSDT","ohlc":{"open":64872,"high":64872,"low":64846,"close":64856.3,"open_time":1786142460293,"close_time":1786142519917,"vwap":64863.1},"volume_total":11.105,"volume_total_usdt":720309,"volume_compra":2.746,"volume_venda":8.359,"num_trades":1561,"delta_minimo":-8.094,"delta_maximo":0.025,"delta_fechamento":-5.613,"reversao_desde_minimo":2.48,"reversao_desde_maximo":5.64,"poc_price":64871.4,"poc_volume":3.54,"poc_percentage":31.9,"dwell_price":64871.4,"dwell_seconds":39,"dwell_location":"High","trades_per_second":26.18,"avg_trade_size":0.007},"enriched_snapshot":{"symbol":"BTCUSDT","ohlc":{"open":64872,"high":64872,"low":64846,"close":64856.3,"open_time":1786142460293,"close_time":1786142519917,"vwap":64863.1},"volume_total":11.105,"volume_total_usdt":720309,"volume_compra":2.746,"volume_venda":8.359,"num_trades":1561,"delta_minimo":-8.094,"delta_maximo":0.025,"delta_fechamento":-5.613,"reversao_desde_minimo":2.48,"reversao_desde_maximo":5.64,"poc_price":64871.4,"poc_volume":3.54,"poc_percentage":31.9,"dwell_price":64871.4,"dwell_seconds":39,"dwell_location":"High","trades_per_second":26.18,"avg_trade_size":0.007},"orderbook_data":{"mid":64827.85,"spread":0.1,"spread_percent":0,"bid_depth_usd":4099382.22,"ask_depth_usd":571474.73,"imbalance":0.755,"flow_imbalance":0.7553,"volume_ratio":7.173,"pressure":0.7553,"consolidated_bias_score":0.9266,"spread_bps":0.0154,"is_valid":1,"data_source":"live","spread_volatility":0.0},"order_book_depth":{"L1":{"bids":2417104.52,"asks":285696.56,"flow_imbalance":0.7886},"L5":{"bids":2501445.35,"asks":289132.44,"flow_imbalance":0.7928},"L10":{"bids":2779876.48,"asks":294967.03,"flow_imbalance":0.8081},"L25":{"bids":3287850.83,"asks":299116.14,"flow_imbalance":0.8332},"total_depth_ratio":10.99},"orderbook_quality":"live","market_impact":{"slippage_matrix":{"100k_usd":{"buy":0.05,"sell":0.05},"1m_usd":{"buy":5.65,"sell":0.05}},"liquidity_score":9.9985,"execution_quality":"EXCELLENT"},"institutional_analytics":{"status":"ok","computed_at_ms":1786142528064,"technical_extras":{"stoch_rsi":{"k":58.57,"d":39.05,"overbought":0,"oversold":0,"crossover":"none"},"williams_r":{"value":-67.09,"overbought":0,"oversold":0,"zone":"neutral","source":"real"},"hurst_exponent":0.3246,"shannon_entropy":2.9163,"kalman_filter":{"kalman_price":64925.85,"raw_price":64872,"deviation_pct":-0.08,"trend_direction":"DOWN"},"regression_channel":{"slope_per_bar":-1.1318,"trend_price":64888.64,"upper_1sd":64905.32,"lower_1sd":64871.95,"upper_2sd":64922.01,"lower_2sd":64855.26,"deviation_from_trend":-16.64,"position_in_channel":0.2508},"dominant_cycles":{"dominant_cycles":[100,40,33.3],"cycle_strengths":[3118.06,1966.69,1010.51]},"fractal_dimension":0.6199,"monte_carlo":{"median_price":64849.52,"p10":64792.93,"p25":64821.18,"p75":64882.56,"p90":64912.74,"prob_up":0.43,"horizon_bars":12},"fair_value_gaps":[{"type":"BEARISH","top":64925,"bottom":64892.3,"gap_size":32.7,"gap_pct":0.05}],"market_structure":{"structure":"BEARISH","bos_detected":0,"last_swing_high":64925,"last_swing_low":64909.55,"higher_highs":0,"higher_lows":0}},"profile_analysis":{"poor_extremes":{"poor_high":{"detected":1,"price":64872,"volume_ratio":3.961,"implication":"High likely to be revisited - unfinished auction"},"poor_low":{"detected":1,"price":64846,"volume_ratio":0.799,"implication":"Low likely to be revisited - unfinished auction"},"excess_high":0,"excess_low":0,"session_high":64872,"session_low":64846,"action_bias":"expect_retest_both","status":"success"},"profile_shape":{"shape":"P","implication":"Short covering rally - bearish bias expected","trading_signal":"BEARISH_AFTER","distribution":{"lower_third_pct":30.3,"middle_third_pct":8.3,"upper_third_pct":61.4},"dominant_zone":"upper","status":"success"},"no_mans_land":{"zones":[{"range_low":64512,"range_high":64810,"gap_size":298,"gap_size_pct":0.462,"risk":"LOW","lvn_confirmed":1,"lvns_in_zone":9,"nearest_hvn_below":64512,"nearest_hvn_above":64810,"distance_from_price":46.34,"distance_pct":0.07,"direction":"below"}],"price_in_no_mans_land":0,"nearest_no_mans_land":{"range_low":64512,"range_high":64810,"gap_size":298,"gap_size_pct":0.462,"risk":"LOW","lvn_confirmed":1,"lvns_in_zone":9,"nearest_hvn_below":64512,"nearest_hvn_above":64810,"distance_from_price":46.34,"distance_pct":0.07,"direction":"below"},"total_zones":1,"max_gap_pct":0.46,"status":"success"},"va_volume_pct":{"value_area_volume_pct":100,"interpretation":"extremely_compressed","breakout_risk":"VERY_HIGH","volume_in_va":1,"total_volume":1,"compression_signal":1},"volume_node_strength":{"scored_hvns":[{"price":64304,"strength":71,"volume_score":15,"proximity_score":25.7,"multi_tf_confluence":1,"confluence_sources":["weekly","monthly"],"in_value_area":0},{"price":64306,"strength":71,"volume_score":15,"proximity_score":25.8,"multi_tf_confluence":1,"confluence_sources":["weekly","monthly"],"in_value_area":0},{"price":64314,"strength":71,"volume_score":15,"proximity_score":25.8,"multi_tf_confluence":1,"confluence_sources":["weekly","monthly"],"in_value_area":0},{"price":64320,"strength":71,"volume_score":15,"proximity_score":25.9,"multi_tf_confluence":1,"confluence_sources":["weekly","monthly"],"in_value_area":0},{"price":64322,"strength":71,"volume_score":15,"proximity_score":25.9,"multi_tf_confluence":1,"confluence_sources":["weekly","monthly"],"in_value_area":0}],"scored_lvns":[{"price":64404,"strength":65,"volume_score":15,"proximity_score":26.5,"multi_tf_confluence":1,"confluence_sources":["monthly"],"in_value_area":0},{"price":64423,"strength":65,"volume_score":15,"proximity_score":26.7,"multi_tf_confluence":1,"confluence_sources":["monthly"],"in_value_area":0},{"price":64436,"strength":65,"volume_score":15,"proximity_score":26.8,"multi_tf_confluence":1,"confluence_sources":["monthly"],"in_value_area":0},{"price":64440,"strength":65,"volume_score":15,"proximity_score":26.8,"multi_tf_confluence":1,"confluence_sources":["monthly"],"in_value_area":0},{"price":64447,"strength":65,"volume_score":15,"proximity_score":26.8,"multi_tf_confluence":1,"confluence_sources":["monthly"],"in_value_area":0}],"total_hvns":52,"total_lvns":101,"avg_hvn_strength":66.2,"avg_lvn_strength":63.1,"status":"success"}},"flow_analysis":{"passive_aggressive":{"aggressive":{"buy_pct":24.72,"sell_pct":75.28,"net_pct":-50.56,"dominance":"sellers","buy_volume":2.746,"sell_volume":8.359},"passive":{"dominance":"buyers","inference":"from_orderbook_depth","bid_depth":4099382.22,"ask_depth":571474.73,"bid_ratio":0.88,"ob_imbalance":0.755},"composite":{"agreement":0,"signal":"sell_absorption","interpretation":"Aggressive sellers hitting passive buy walls - potential reversal or breakdown","conviction":"MEDIUM"},"status":"success"},"whale_accumulation":{"score":-21,"classification":"MILD_DISTRIBUTION","bias":"DISTRIBUTING","components":{"flow":{"score":-30,"max":30,"detail":{"divergence":"aligned","whale_delta":-5,"mid_delta":-3.005,"retail_delta":-20.565,"primary_delta":-5}},"depth":{"score":15.11,"max":20,"detail":{"bid_depth":4099382.22,"ask_depth":571474.73,"ratio":0.76,"deep_confirmation":0}},"absorption":{"score":-7.8,"max":25,"detail":{"buyer_strength":2.5,"seller_exhaustion":5.1,"net_absorption":-2.6,"index":0.1129,"label":"Neutra"}},"derivatives":{"score":1.49,"max":25,"detail":{"long_short_ratio":1.1,"lsr_score":1.49}}},"trend":{"direction":"stable","avg_score":-25.5,"recent_avg":-29.8,"momentum":4.5,"samples":12,"score_range":{"min":-49,"max":14}},"status":"success"},"absorption_zones":{"total_zones":0,"total_events":0,"buy_zone_count":0,"sell_zone_count":0,"status":"no_events"}},"sr_analysis":{"defense_zones":{"buy_defense":[{"center":64479.84,"range_low":64376.36,"range_high":64570.64,"strength":62,"side":"buy","sources":["sr_level_ema_21_4h","vp_hvn","ema_ema_21_4h","vp_val"],"source_count":4,"signals_in_zone":5,"type":"confluence","distance_from_price":376.5,"distance_pct":0.58},{"center":64831.72,"range_low":64740.16,"range_high":64928.64,"strength":61,"side":"buy","sources":["orderbook_bid_wall","vp_hvn","ema_ema_21_1h","sr_level_ema_21_1h"],"source_count":4,"signals_in_zone":8,"type":"confluence","distance_from_price":24.62,"distance_pct":0.04},{"center":64162.07,"range_low":64096.13,"range_high":64228.02,"strength":46,"side":"buy","sources":["ema_ema_21_1d","sr_level_vah_monthly"],"source_count":2,"signals_in_zone":2,"type":"cluster","distance_from_price":694.27,"distance_pct":1.07},{"center":64336.25,"range_low":64255.36,"range_high":64442.64,"strength":40,"side":"buy","sources":["vp_hvn","sr_level_hvn_daily"],"source_count":2,"signals_in_zone":9,"type":"cluster","distance_from_price":520.09,"distance_pct":0.8}],"sell_defense":[{"center":65037.38,"range_low":64960.36,"range_high":65128.64,"strength":55,"side":"sell","sources":["sr_level_poc_daily","vp_poc","vp_hvn"],"source_count":3,"signals_in_zone":9,"type":"confluence","distance_from_price":181.04,"distance_pct":0.28},{"center":65346,"range_low":65297.36,"range_high":65394.64,"strength":53,"side":"sell","sources":["sr_level_vah_daily","vp_hvn","vp_vah"],"source_count":3,"signals_in_zone":3,"type":"confluence","distance_from_price":489.66,"distance_pct":0.76},{"center":64944.5,"range_low":64860.36,"range_high":65052.64,"strength":49,"side":"sell","sources":["vp_hvn","ema_ema_21_15m","sr_level_hvn_daily"],"source_count":3,"signals_in_zone":9,"type":"confluence","distance_from_price":88.16,"distance_pct":0.14},{"center":65157.17,"range_low":65064.36,"range_high":65257.64,"strength":41,"side":"sell","sources":["vp_hvn","sr_level_hvn_daily"],"source_count":2,"signals_in_zone":13,"type":"cluster","distance_from_price":300.83,"distance_pct":0.46},{"center":65252.3,"range_low":65164.36,"range_high":65350.64,"strength":40,"side":"sell","sources":["vp_hvn","sr_level_hvn_daily"],"source_count":2,"signals_in_zone":11,"type":"cluster","distance_from_price":395.96,"distance_pct":0.61}],"total_zones":9,"strongest_buy":{"center":64479.84,"range_low":64376.36,"range_high":64570.64,"strength":62,"side":"buy","sources":["sr_level_ema_21_4h","vp_hvn","ema_ema_21_4h","vp_val"],"source_count":4,"signals_in_zone":5,"type":"confluence","distance_from_price":376.5,"distance_pct":0.58},"strongest_sell":{"center":65037.38,"range_low":64960.36,"range_high":65128.64,"strength":55,"side":"sell","sources":["sr_level_poc_daily","vp_poc","vp_hvn"],"source_count":3,"signals_in_zone":9,"type":"confluence","distance_from_price":181.04,"distance_pct":0.28},"defense_asymmetry":{"ratio":0.88,"bias":"slight_sell_defense","description":"Slightly more sell defense","buy_total_strength":209,"sell_total_strength":238},"status":"success"}},"quality":{"calendar":{"day_of_week":"Friday","day_of_week_num":4,"is_us_holiday":0,"is_weekend":0,"is_pre_holiday":0,"is_post_holiday":0,"expected_liquidity":"NORMAL","liquidity_warning":0},"latency":{"latency_ms":8225,"latency_category":"POOR","data_freshness":"DELAYED","is_acceptable":0,"is_stale":0},"spread_percentile":{"status":"ok","current_spread":0.1,"spread_percentile":0,"spread_mean":0.1,"spread_median":0.1,"spread_std":0,"spread_min":0.1,"spread_max":0.1,"spread_z_score":-0.9574,"is_tight":1,"is_wide":0,"is_anomalous":0,"liquidity_signal":"EXCELLENT","samples":12,"window_minutes":1440},"anomalies":{"anomalies_detected":1,"count":2,"anomalies":[{"type":"FLOW_EXTREME_IMBALANCE","severity":"MEDIUM","value":-0.5055,"direction":"SELL","description":"Extreme flow imbalance: -50.55% toward sellers"},{"type":"DEPTH_EXTREME_ASYMMETRY","severity":"MEDIUM","ratio":7.17,"bid_depth":4099382.22,"ask_depth":571474.73,"direction":"BID_HEAVY","description":"Order book depth ratio 7.17:1 is extreme"}],"max_severity":"MEDIUM","risk_elevated":0,"types_found":["DEPTH_EXTREME_ASYMMETRY","FLOW_EXTREME_IMBALANCE"],"summary":"2 anomalies detected (max severity: MEDIUM)"}},"candlestick_patterns":{"patterns_detected":1,"patterns":[{"name":"doji","type":"neutral","confidence":0.65,"candles_used":1,"implication":"Indecision - watch next candle for direction"}],"dominant_signal":"neutral","max_confidence":0.65,"bullish_count":0,"bearish_count":0,"neutral_count":1}},"sequence_id":12,"primary_exchange":"BINANCE","data_feed_type":"WEBSOCKET_L2","data_quality_score":9.5,"completeness_pct":100,"reliability_score":9,"bid":64827.8,"ask":64827.9,"tick_direction":-1,"twap":64881.49,"pivot_points":{"daily":{"pivot":64971.33,"r1":65420.66,"r2":65795.33,"r3":66619.33,"s1":64596.66,"s2":64147.33,"s3":63323.33,"vah":65346,"val":64522,"poc":65046},"weekly":{"pivot":62830.33,"r1":63346.66,"r2":64083.33,"r3":65336.33,"s1":62093.66,"s2":61577.33,"s3":60324.33,"vah":63567,"val":62314,"poc":62610},"monthly":{"pivot":63185.67,"r1":64582.34,"r2":65610.67,"r3":68035.67,"s1":62157.34,"s2":60760.67,"s3":58335.67,"vah":64214,"val":61789,"poc":63554}},"immediate_support":[64831.72,64522,64479.84,64214,64162.07],"support_strength":[61,94.8,62,72.1,46],"immediate_resistance":[64944.5,64971.33,65037.38,65346],"resistance_strength":[49,98.2,55,92.5],"volatility_metrics":{"realized_vol_24h":0.0184,"realized_vol_7d":0.0487,"volatility_regime":"NORMAL"},"order_flow_extended":{"passive_buy_pct":87.8,"passive_sell_pct":12.2},"whale_activity":{"large_orders_1h":[{"size":1,"price":64896.34,"side":"SELL","timestamp_ms":1786141945725},{"size":1,"price":64883.14,"side":"SELL","timestamp_ms":1786142044496},{"size":1,"price":64870.63,"side":"SELL","timestamp_ms":1786142264045},{"size":1,"price":64853.03,"side":"SELL","timestamp_ms":1786142272654},{"size":1,"price":64867.43,"side":"SELL","timestamp_ms":1786142499673}],"iceberg_activity":1,"hidden_orders_detected":0},"technical_indicators_extended":{"cci_1h":97.67,"cci_signal":"NEUTRAL","stochastic":{"k":58.57,"d":39.05,"signal":"NEUTRAL","source":"real"},"williams_r":{"value":-67.09,"overbought":0,"oversold":0,"zone":"neutral","source":"real"},"hurst_exponent":0.3246,"shannon_entropy":2.9163,"fractal_dimension":0.6199,"kalman_filter":{"kalman_price":64925.85,"raw_price":64872,"deviation_pct":-0.08,"trend_direction":"DOWN"},"regression_channel":{"slope_per_bar":-1.1318,"trend_price":64888.64,"upper_1sd":64905.32,"lower_1sd":64871.95,"upper_2sd":64922.01,"lower_2sd":64855.26,"deviation_from_trend":-16.64,"position_in_channel":0.2508},"dominant_cycles":{"dominant_cycles":[100,40,33.3],"cycle_strengths":[3118.06,1966.69,1010.51]},"monte_carlo":{"median_price":64849.52,"p10":64792.93,"p25":64821.18,"p75":64882.56,"p90":64912.74,"prob_up":0.43,"horizon_bars":12},"garch_forecast_1h":0.0017},"backup_exchanges":["COINBASE","KRAKEN","OKX"],"alerts":{"active_alerts":[{"type":"DEPTH_DIVERGENCE","level":0.755,"severity":"MEDIUM","probability":0.76,"action":"PREPARE_LONG","description":"Orderbook BID_HEAVY: imbalance=0.755"}],"alert_count":1,"max_severity":"MEDIUM"},"regime_analysis":{"current_regime":"MEAN_REVERTING","regime_probabilities":{"trending":0,"mean_reverting":0.714,"breakout":0.286},"regime_change_probability":0.29,"expected_regime_duration":"15m-1h","avg_adx":25.9},"data_reliability":{"has_options_data":0,"onchain_coverage":"partial","latency_acceptable":1,"price_targets_available":0}}
{"tipo_evento":"ANALYSIS_TRIGGER","symbol":"BTCUSDT","epoch_ms":1786142519917,"janela_numero":12,"event_id":"a92dec97","timestamp_utc":"2026-08-07T22:42:00.000Z","note":"trimmed_by_guardian"}
{"tipo_evento":"Alerta","resultado_da_batalha":"VOLATILITY_SQUEEZE","descricao":"Tipo: VOLATILITY_SQUEEZE","timestamp":"2026-08-07T22:42:09+00:00","severity":"LOW","probability":0.4,"action":"WATCH_FOR_NORMALIZATION","context":{"price":64856.3,"volume":11.105,"average_volume":4.635,"volatility":1.2e-07},"data_context":"real_time","janela_numero":12,"epoch_ms":1786142529331,"event_id":"4cdd3c4d","timestamp_utc":"2026-08-07T22:42:09.331+00:00","timestamp_ny":"2026-08-07T18:42:09.331-04:00","timestamp_sp":"2026-08-07T19:42:09.331-03:00","price_data":{"current":{"last":64856.3,"volume":11.105}},"volatility_metrics":{"realized_vol_24h":0},"market_context":{"trading_session":"NY_OVERLAP","session_phase":"ACTIVE"}}
{"tipo_evento":"ANALYSIS_TRIGGER","symbol":"BTCUSDT","epoch_ms":1786142580000,"janela_numero":13,"event_id":"16e36cc3","timestamp_utc":"2026-08-07T22:43:00.000Z","note":"trimmed_by_guardian"}
```


## 2.3 wc -l


```text
eventos-fluxo.json: 23021 linhas (pretty JSON, 18 eventos)
eventos_fluxo.jsonl: 18 linhas (18 eventos)
mesmos epoch_ms nos dois arquivos
```


# SEÇÃO 3 — BANCO DE DADOS (dados/trading_bot.db)


## 3.1 .tables + COUNT(*) por tabela


```sql
events: 18 linhas
signal_outcomes: 0 linhas
sqlite_sequence: 1 linhas
```


## 3.2 .schema (todas as tabelas)


```sql
CREATE TABLE events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp_ms INTEGER NOT NULL,
                        event_type TEXT NOT NULL,
                        symbol TEXT,
                        window_id TEXT,
                        is_signal BOOLEAN DEFAULT 0,
                        payload JSON NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )

CREATE TABLE signal_outcomes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        signal_epoch_ms INTEGER NOT NULL,
                        event_type TEXT NOT NULL,
                        battle_result TEXT,
                        entry_price REAL NOT NULL,
                        symbol TEXT DEFAULT 'BTCUSDT',
                        context_json TEXT,
                        outcome_5m_pct REAL,
                        outcome_15m_pct REAL,
                        outcome_30m_pct REAL,
                        outcome_60m_pct REAL,
                        outcome_direction_5m TEXT,
                        outcome_direction_15m TEXT,
                        outcome_direction_30m TEXT,
                        outcome_direction_60m TEXT,
                        evaluated_at INTEGER,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )

CREATE TABLE sqlite_sequence(name,seq)
```


## 3.3 Amostra: events (20 mais recentes, metadados + tamanho do payload JSON)


```sql
SELECT id,timestamp_ms,event_type,symbol,window_id,is_signal,payload,created_at FROM events ORDER BY rowid DESC LIMIT 20;

id=18 | timestamp_ms=1786142580000 | event_type=ANALYSIS_TRIGGER | symbol=BTCUSDT | window_id= | is_signal=1 | payload_len=25282 | created_at=2026-08-07 22:43:10
id=17 | timestamp_ms=1786142529331 | event_type=Alerta | symbol= | window_id= | is_signal=0 | payload_len=894 | created_at=2026-08-07 22:42:10
id=16 | timestamp_ms=1786142519917 | event_type=ANALYSIS_TRIGGER | symbol=BTCUSDT | window_id= | is_signal=1 | payload_len=26006 | created_at=2026-08-07 22:42:10
id=15 | timestamp_ms=1786142520000 | event_type=Exaustão | symbol=BTCUSDT | window_id=1786142519917 | is_signal=1 | payload_len=49079 | created_at=2026-08-07 22:42:10
id=14 | timestamp_ms=1786142482499 | event_type=Alerta | symbol= | window_id= | is_signal=0 | payload_len=887 | created_at=2026-08-07 22:41:24
id=13 | timestamp_ms=1786142460000 | event_type=ANALYSIS_TRIGGER | symbol=BTCUSDT | window_id= | is_signal=1 | payload_len=24858 | created_at=2026-08-07 22:41:24
id=12 | timestamp_ms=1786142400000 | event_type=ANALYSIS_TRIGGER | symbol=BTCUSDT | window_id= | is_signal=1 | payload_len=24563 | created_at=2026-08-07 22:40:09
id=11 | timestamp_ms=1786142340000 | event_type=ANALYSIS_TRIGGER | symbol=BTCUSDT | window_id= | is_signal=1 | payload_len=25714 | created_at=2026-08-07 22:39:09
id=10 | timestamp_ms=1786142293862 | event_type=AI_ANALYSIS | symbol=BTCUSDT | window_id= | is_signal=0 | payload_len=4849 | created_at=2026-08-07 22:38:14
id=9 | timestamp_ms=1786142280000 | event_type=ANALYSIS_TRIGGER | symbol=BTCUSDT | window_id= | is_signal=1 | payload_len=25508 | created_at=2026-08-07 22:38:14
id=8 | timestamp_ms=1786142220000 | event_type=ANALYSIS_TRIGGER | symbol=BTCUSDT | window_id= | is_signal=1 | payload_len=25058 | created_at=2026-08-07 22:37:08
id=7 | timestamp_ms=1786142160000 | event_type=ANALYSIS_TRIGGER | symbol=BTCUSDT | window_id= | is_signal=1 | payload_len=25300 | created_at=2026-08-07 22:36:28
id=6 | timestamp_ms=1786142100000 | event_type=ANALYSIS_TRIGGER | symbol=BTCUSDT | window_id= | is_signal=1 | payload_len=25368 | created_at=2026-08-07 22:35:08
id=5 | timestamp_ms=1786142040000 | event_type=ANALYSIS_TRIGGER | symbol=BTCUSDT | window_id= | is_signal=1 | payload_len=23896 | created_at=2026-08-07 22:34:08
id=4 | timestamp_ms=1786141988998 | event_type=AI_ANALYSIS | symbol=BTCUSDT | window_id= | is_signal=0 | payload_len=4815 | created_at=2026-08-07 22:33:13
id=3 | timestamp_ms=1786141980000 | event_type=ANALYSIS_TRIGGER | symbol=BTCUSDT | window_id= | is_signal=1 | payload_len=24906 | created_at=2026-08-07 22:33:08
id=2 | timestamp_ms=1786141920000 | event_type=ANALYSIS_TRIGGER | symbol=BTCUSDT | window_id= | is_signal=1 | payload_len=23783 | created_at=2026-08-07 22:32:07
id=1 | timestamp_ms=1786141860000 | event_type=ANALYSIS_TRIGGER | symbol=BTCUSDT | window_id= | is_signal=1 | payload_len=24645 | created_at=2026-08-07 22:31:17
```


## 3.4 Amostra: events — payload COMPLETO do registro AI_ANALYSIS (id=10, o que é enviado à IA)


(payload completo de 4.849 bytes — evento AI_ANALYSIS com ai_result + ai_payload; registros ANALYSIS_TRIGGER têm 24-49 KB e estão representados pelos metadados acima + seção 1)


```json
{"tipo_evento": "AI_ANALYSIS", "symbol": "BTCUSDT", "timestamp_ms": 1786142280000, "anchor_price": 64846.0, "anchor_window_id": 8, "ai_result": {"sentiment": "bearish", "confidence": 0.66, "action": "wait", "rationale": "Os 15m apontam queda com fluxo de venda intenso e desequil\u00edbrio de ordem negativo, enquanto o 4h ainda indica tend\u00eancia de alta, sugerindo uma retra\u00e7\u00e3o dentro d", "entry_zone": null, "invalidation_zone": null, "region_type": "retra\u00e7\u00e3o", "_is_fallback": false, "_fallback_reason": null, "_validation_error": null, "_is_valid": true}, "ai_payload": {"symbol": "BTCUSDT", "epoch_ms": 1786142280000, "trigger": "AT", "price": {"c": 64846, "o": 64873, "h": 64873, "vw": 64863, "sh": "P", "auc": "expect_retest_high", "ph": 1, "brk_risk": "V_HI"}, "regime": {"cs": "BULL", "cf": 0.8, "v": "NOR", "mode": "BRK", "dom": "4h", "bull%": 90, "bear%": 10}, "qual": {"lat": "POOR", "ms": 11450}, "flow": {"d1": "-221K", "delta": -8.128, "vol": 8.708, "buy_pct": 3, "ti": 20.6, "trs": -223, "obs": -2.838, "sf_w": -4.0, "sf_r": -18.859, "d5": "-943K", "d15": "-1.7M", "cvd": -25.8, "imb": -0.93, "ab": 3, "bsr": 0.03, "pa": "sell_absor", "conv": "M", "abs_buy_str": 0.3, "abs_sell_exh": 9.3, "abs_cont": 0.33}, "ob": {"b": "3.6M", "a": "308K", "imb": 0.84, "bias": "BUY", "t5": 0.87, "spread_pct": 0.0002, "slip_b": 5.0, "slip_s": 5.0}, "tf": {"15m": {"t": "DN", "rsi": 38, "macd": [13, 17], "adx": 24, "atr": 70, "r": "RNG"}, "1h": {"t": "UP", "rsi": 53, "macd": [100, 98], "adx": 24, "atr": 224, "r": "RNG"}, "4h": {"t": "UP", "rsi": 62, "macd": [275, 257], "adx": 30, "atr": 495, "r": "RNG"}, "1d": {"t": "UP", "rsi": 59, "macd": [71, 37], "adx": 15, "atr": 1370, "r": "MNP"}}, "sr": {"r1": [65030, 54], "r1_dist": 184, "r1_conf": 3, "r2": [65346, 53], "r2_dist": 500, "r2_conf": 3, "s1": [64480, 62], "s1_dist": 366, "s1_conf": 4, "s2": [64823, 62], "s2_dist": 23, "s2_conf": 4, "def_bias": "slight_sel"}, "w": {"s": -37, "c": "MD"}, "ext": {"cci": "OB", "stoch": 0, "stoch_sig": "OS", "wr": -100, "garch": 0.0, "hurst": 0.33, "entropy": 2.93, "fd": 0.62, "kalman": {"kp": 64933.92, "dev": -0.094, "dir": "DOWN"}, "reg": {"sl": -1.1398, "pos": 0.1289, "dev": -24.4}, "mc": {"pu": 0.466, "p10": 64785.73, "p90": 64904.98}, "cycles": [100.0, 40.0], "smc": {"fvg": 1, "fvg_last": "BE", "struct": "BEAR", "bos": 0}}, "ctx": {"ses": "NY", "fg": 29, "poc": 65046, "val": 64522, "vah": 65346, "lsr": 1.1, "eth_lsr": 2.07, "oi": 107, "fr": 0.0063, "longs": "+3631.4M", "shorts": "+3302.8M", "eth7": 0.8, "dxy30": -0.06}, "ofi": {"score": -0.933, "dir": "SELL", "src": "order_flow"}, "vwap": {"dev": -0.027, "side": "below", "sig": "fair", "src": "ohlc"}, "iceberg": {"det": 1, "src": "whale_activity"}, "liq": [{"p": 64867, "side": "sell", "vol": 10.58}], "cvd_div": {"det": 1, "type": "bearish_div", "src": "inferred"}, "mr": {"score": 0.248, "sig": "stretched_bear", "src": "inferred"}, "summary": {"flow": {"bias": "SELL", "type": "mixed", "actor": "retail", "conf": "M", "note": "Fluxo misto sem domin\u00e2ncia clara. (varejo vendedor). prob. continua\u00e7\u00e3o 33%. [imbalance extremo de venda]"}, "sr": {"nearest": "resistance", "compressed": false, "conf_bias": "NEUTRAL", "note": "Resist\u00eancia mais pr\u00f3xima em 65030 (for\u00e7a 54, conflu\u00eancia 3 fontes, dist 184 pts (0.8 ATR))", "r1_dist_atr": 0.82, "s1_dist_atr": 1.63}, "regime": {"label": "Breakout", "strategies": ["aguardar confirma\u00e7\u00e3o de rompimento", "entrar no reteste do n\u00edvel rompido", "usar stop apertado acima/abaixo do n\u00edvel", "monitorar volume de confirma\u00e7\u00e3o"], "avoid": ["entrar antes da confirma\u00e7\u00e3o", "ignorar falsos rompimentos", "operar range enquanto houver BRK ativo"], "duration": "minutos a horas \u2014 confirmar r\u00e1pido", "note": "Regime Breakout com consenso de alta (confian\u00e7a alta: 80%). dominado pelo 4h."}, "institutional": {"auction_state": "Leil\u00e3o incompleto \u2014 m\u00e1xima deve ser revisitada", "whale_bias": "DISTRIBUTING", "profile_bias": "NEUTRAL", "unfinished": ["high"], "alignment": "BEAR_ALIGNED", "note": "Leil\u00e3o incompleto \u2014 m\u00e1xima deve ser revisitada. Extremo(s) incompleto(s): high \u2014 reteste esperado. Risco de breakout da Value Area muito alto. Sinais institucionais alinhados para baixa."}, "quality": {"reliable": true, "confidence_cap": 1.0, "issues": [], "note": "Dados em tempo real sem anomalias. An\u00e1lise com confian\u00e7a plena."}}, "tipo_evento": "ANALYSIS_TRIGGER", "descricao": "Evento autom\u00e1tico para an\u00e1lise da IA"}, "epoch_ms": 1786142293862, "event_id": "85c52eed", "data_context": "real_time", "timestamp_utc": "2026-08-07T22:38:13.862+00:00", "timestamp_ny": "2026-08-07T18:38:13.862-04:00", "timestamp_sp": "2026-08-07T19:38:13.862-03:00", "timestamp": "2026-08-07T22:38:13.862+00:00"}
```


## 3.5 signal_outcomes — vazio (0 linhas)


```sql
SELECT COUNT(*) FROM signal_outcomes;  -- 0
SELECT * FROM signal_outcomes ORDER BY rowid DESC LIMIT 20;  -- sem registros
```


# SEÇÃO 4 — CONFIGURAÇÕES ATIVAS


## 4.1 config.json (completo)


```json
{
  "SYMBOL": "BTCUSDT",
  "WINDOW_SIZE_MINUTES": 1,
  "VOL_FACTOR_EXH": 2.5,
  "HISTORY_SIZE": 100,
  "DELTA_STD_DEV_FACTOR": 2.0,
  "CONTEXT_SMA_PERIOD": 20,
  "LIQUIDITY_FLOW_ALERT_PERCENTAGE": 0.15,
  "WALL_STD_DEV_FACTOR": 3.0,
  "ai": {
    "payload_compression": true,
    "provider": "groq",
    "groq": {
      "base_url": "https://api.groq.com/openai/v1",
      "model": "openai/gpt-oss-120b"
    },
    "dashscope": {
      "enabled": false
    }
  }
}
```


## 4.2 config/model_config.yaml (completo)


```yaml
﻿ai:
  provider: groq
  provider_fallbacks: []  # Lista de providers para fallback automatico (so usar se necessario)
  groq:
    model: openai/gpt-oss-120b   # Modelo principal — 120B para qualidade de analise
    model_fallbacks:
      - llama-3.1-8b-instant         # Fallback 1: rapido/barato (se 70B falhar por rate limit)
      - mixtral-8x7b-32768           # Fallback 2: contexto longo (32K)

model:
  # Parametros do target
  lookahead_windows: 15         # Quantas janelas a frente prever
  min_return_threshold: 0.002   # 0.2% minimo para considerar "COMPRA"

  # Divisao dos dados
  test_size: 0.2                # 20% para teste
  validation_size: 0.1          # 10% para validacao
  random_state: 42              # Semente para reprodutibilidade

xgboost:
  # HiperparÇ½metros do XGBoost
  n_estimators: 500             # Numero de arvores
  learning_rate: 0.05           # Taxa de aprendizado
  max_depth: 6                  # Profundidade maxima das arvores
  subsample: 0.8                # Amostragem de linhas
  colsample_bytree: 0.8         # Amostragem de colunas
  eval_metric: "logloss"        # Metrica de avaliacao
  n_jobs: -1                    # Usar todos os cores da CPU
  random_state: 42              # Semente
  
features:
  # Configuracoes das features
  required_columns:             # Colunas obrigatorias
    - "price_close"
    - "volume"
  
  drop_columns:                 # Colunas para remover
    - "window_id"
    - "saved_at"
    - "symbol"
    - "timestamp"
    - "timestamp_utc"
    - "epoch_ms"
  
  # Limites de qualidade
  max_null_percentage: 0.3      # Maximo 30% de valores nulos
  correlation_threshold: 0.95   # Remover features com correlacao > 95%

llm_payload:
  v2_enabled: true
  max_bytes: 6144
  section_budgets_enabled: true
  section_cache_enabled: true
  cache_ttls_s:
    macro_context: 1800
    cross_asset_context: 1800
  guardrail_hard_enabled: true
  tripwires:
    fallback_rate_max: 0.10          # 10% (era 5% — guardrail causava fallbacks falsos antes do fix)
    abort_rate_max: 0.12             # 12% (era 2% — primeiras janelas abortam por dados incompletos)
    guardrail_block_rate_max: 0.50   # 50% (todo analyze() com raw_event bloqueia)
    bytes_p95_max: 90000             # 90KB (payloads reais chegam a 80KB)
    cache_hit_rate_min:
      macro_context: 0.0             # 0% (cache macro demora a aquecer, não deve disparar tripwire)
      cross_asset_context: 0.0       # 0% (idem — cache cross_asset pode nunca ter hit nas primeiras janelas)

sampling:
  # Balanceamento de classes
  use_smote: false              # Usar SMOTE para oversampling?
  smote_ratio: 0.5              # Balanceamento (1.0 = 50/50)

paths:
  # Diretorios
  features_dir: "features"
  models_dir: "ml/models"
  logs_dir: "ml/logs"
```


## 4.3 .env.example (completo)


```text
# =============================================================================
# .env.example - Template de variaveis de ambiente
# Copie este arquivo para .env e preencha com suas credenciais
# NUNCA commite o .env com credenciais reais!
# =============================================================================

# ===== Binance API =====
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here

# ===== AI / LLM Provider (Groq) =====
GROQ_API_KEY=your_groq_api_key_here

# ===== OpenAI (opcional, alternativa ao Groq) =====
OPENAI_API_KEY=your_openai_api_key_here

# ===== Alpha Vantage (dados de mercado) =====
ALPHAVANTAGE_API_KEY=your_alphavantage_api_key_here

# ===== FRED API (dados macroeconomicos) =====
FRED_API_KEY=your_fred_api_key_here
```


## 4.4 Variáveis do .env real — apenas NOMES (valores redigidos)


```text
Nomes de variáveis presentes no .env real (ordem alfabética):
AI_ENABLED
AI_MAX_PER_HOUR
AI_PROVIDER
AI_THROTTLE_INTERVAL
ALPHAVANTAGE_API_KEY
BINANCE_API_KEY
BINANCE_API_SECRET
DEFAULT_SYMBOL
FEATURE_ML_READY
FRED_API_KEY
GROQ_API_KEY
GROQ_MODEL
LOG_LEVEL
OPENAI_API_KEY
TWELVEDATA_API_KEY

Variáveis presentes no .env real e NÃO documentadas no .env.example:
AI_ENABLED, AI_MAX_PER_HOUR, AI_PROVIDER, AI_THROTTLE_INTERVAL, DEFAULT_SYMBOL,
FEATURE_ML_READY, GROQ_MODEL, LOG_LEVEL, TWELVEDATA_API_KEY
```


## 4.5 Parâmetros de runtime (janelas, thresholds, throttler, buffers)


**AI Throttler** — `market_orchestrator/ai/ai_runner.py:41-46` (sobrescreve defaults do singleton):


```python
_ai_throttler = get_throttler(
    min_interval=60,          # soft min (s) — pode ser bypassed
    hard_min_interval=30,     # hard min (s) — nunca bypassed
    daily_token_budget=85_000,
    max_calls_per_hour=10,
)
```


**Defaults do SmartAIThrottler** — `common/ai_throttler.py:52-63` (usados se get_throttler() sem kwargs, ex: analyzer_qwen.py:3456):


```python
min_interval: float = 180.0          # soft min (s)
hard_min_interval: float = 60.0      # hard min (s)
significant_imb_change: float = 0.5
daily_token_budget: int = 50_000
tokens_per_call_estimate: int = 2_500
max_calls_per_hour: int = 6
base_cooldown_429: float = 120.0
max_cooldown_429: float = 1800.0
ALWAYS_PROCESS = {"Exaustão", "Absorção", "whale_detected", "regime_change", "LARGE_TRADE"}
```


**FlowAnalyzer** — `flow_analyzer/constants.py` (completo):


```python
# flow_analyzer/constants.py
"""
Constantes e configurações do FlowAnalyzer.

Este módulo centraliza todas as constantes, magic numbers e valores
de configuração para facilitar manutenção e testes.
"""

from decimal import Decimal
from typing import Dict, Tuple

# ==============================================================================
# VERSÃO
# ==============================================================================
VERSION = "2.4.0"

# ==============================================================================
# CONFIGURAÇÕES PADRÃO (fallback se config.py não disponível)
# ==============================================================================
DEFAULT_NET_FLOW_WINDOWS_MIN = [1, 5, 15]
DEFAULT_ABSORCAO_DELTA_EPS = 1.0
DEFAULT_ABSORCAO_GUARD_MODE = "warn"
DEFAULT_FLOW_TRADES_MAXLEN = 100_000
DEFAULT_FLOW_LOG_PERF = False
DEFAULT_FLOW_LOG_DETAILED = False
DEFAULT_FLOW_TIME_BUDGET_MS = 500.0
DEFAULT_FLOW_CACHE_ENABLED = True
DEFAULT_WHALE_TRADE_THRESHOLD = 1.0  # reduzido de 5.0 → detecta trades ~$68K+ (top ~0.5%)
DEFAULT_CVD_RESET_INTERVAL_HOURS = 4

# ==============================================================================
# TIMESTAMPS E SINCRONIZAÇÃO
# ==============================================================================
TIMESTAMP_JITTER_TOLERANCE_MS = 2000
LATE_TRADE_THRESHOLD_MS = 120000  # Trade atrasado > 120s
MAX_LATE_TRADE_MS = 5000  # Trade muito atrasado > 5s
MAX_BATCH_LATE_MS = 30000  # Tolerância de atraso para trades em batch (30s)

# ==============================================================================
# BURST DETECTION
# ==============================================================================
DEFAULT_BURST_WINDOW_MS = 200
DEFAULT_BURST_COOLDOWN_MS = 200
BURST_END_THRESHOLD_RATIO = 0.5  # Burst termina quando volume < 50% do threshold

# ==============================================================================
# LIQUIDITY HEATMAP
# ==============================================================================
DEFAULT_LHM_WINDOW_SIZE = 2000
DEFAULT_LHM_CLUSTER_THRESHOLD_PCT = 0.003
DEFAULT_LHM_MIN_TRADES_PER_CLUSTER = 5
DEFAULT_LHM_UPDATE_INTERVAL_MS = 100

# ==============================================================================
# ABSORÇÃO
# ==============================================================================
DEFAULT_ABSORCAO_ATR_MULTIPLIER = 0.5
DEFAULT_ABSORCAO_VOL_MULTIPLIER = 1.0
DEFAULT_ABSORCAO_MIN_PCT_TOLERANCE = 0.001
DEFAULT_ABSORCAO_MAX_PCT_TOLERANCE = 0.01
DEFAULT_ABSORCAO_FALLBACK_PCT_TOLERANCE = 0.002

# Thresholds para classificação de absorção
ABSORPTION_INTENSITY_THRESHOLD = 0.15
ABSORPTION_IMBALANCE_THRESHOLD = 0.15

# ==============================================================================
# PRECISÃO NUMÉRICA
# ==============================================================================
DECIMAL_PRECISION_BTC = 8
DECIMAL_PRECISION_USD = 2
DECIMAL_CENT = Decimal('0.01')
DECIMAL_ZERO = Decimal('0')
DECIMAL_TOLERANCE_BTC = Decimal('1e-6')
UI_TOLERANCE_USD = Decimal('0.02')

# ==============================================================================
# PERFORMANCE E OBSERVABILIDADE
# ==============================================================================
PERF_MONITOR_WINDOW_SIZE = 1000
LAZY_LOG_INTERVAL_MS = 1000
HIGH_LATENCY_P99_MS = 500.0  # antes: 100ms, muito agressivo com warmup/macro fetch
MEMORY_USAGE_WARNING_RATIO = 0.9

# ==============================================================================
# PARTICIPANT ANALYSIS
# ==============================================================================
DEFAULT_ORDER_SIZE_BUCKETS: Dict[str, Tuple[float, float]] = {
    "retail": (0.0, 0.5),
    "mid": (0.5, 1.0),
    "whale": (1.0, 9999.0),
}

# Pesos para composite score
PARTICIPANT_IMBALANCE_WEIGHT = 0.4
PARTICIPANT_PARTICIPATION_WEIGHT = 0.4
PARTICIPANT_FREQUENCY_WEIGHT = 0.2
MAX_TRADES_PER_SECOND = 10.0  # Para normalização de frequência

# ==============================================================================
# CIRCUIT BREAKER
# ==============================================================================
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
CIRCUIT_BREAKER_RECOVERY_TIME_MS = 30_000

# ==============================================================================
# ERROR TRACKING
# ==============================================================================
MAX_ERROR_KEYS = 100  # Limite de tipos de erro rastreados

# ==============================================================================
# ROLLING AGGREGATES
# ==============================================================================
MAX_TRADES_PER_MINUTE_ESTIMATE = 600  # ~10 trades/segundo
MAX_AGGREGATE_TRADES = 10_000  # Limite absoluto por janela
```


**Janelas de tempo do bot** — `config.json`: `WINDOW_SIZE_MINUTES: 1`, `HISTORY_SIZE: 100`.


**llm_payload** — `config/model_config.yaml`: `v2_enabled: true`, `max_bytes: 6144`, `section_budgets_enabled: true`, `section_cache_enabled: true`, `cache_ttls_s: {macro_context: 1800, cross_asset_context: 1800}`, `guardrail_hard_enabled: true`; tripwires: fallback_rate_max=0.10, abort_rate_max=0.12, guardrail_block_rate_max=0.50, bytes_p95_max=90000, cache_hit_rate_min=0.0 (ambos).


**Modelo/fallbacks** — `model_config.yaml`: modelo principal `openai/gpt-oss-120b` (Groq); fallbacks `llama-3.1-8b-instant`, `mixtral-8x7b-32768`. `config.json` ai.provider=groq, dashscope disabled.


# SEÇÃO 5 — CÁLCULOS E FÓRMULAS


## 5.1 common/technical_indicators.py (funções de cálculo completas)


```python
import numpy as np
import pandas as pd
from typing import Optional
from common.twap_validator import TWAPValidator

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) for a price series.
    Returns a pandas Series with RSI values.
    """
    series = series.astype(float)
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    loss = down.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = gain / loss
    rsi = 100 - 100 / (1 + rs)
    return rsi

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series]:
    """
    Compute the MACD line and signal line for a price series.
    Returns a tuple (macd_line, signal_line).
    """
    series = series.astype(float)
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Helper to compute True Range, used in ATR and ADX calculations.
    """
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr

def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Compute the Average Directional Index (ADX) for given high, low, close series.
    Returns a pandas Series with ADX values.
    """
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)
    # Compute directional movements
    up_move = high.diff()
    down_move = low.shift() - low
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    # True range
    tr = _true_range(high, low, close)
    # Smooth TR, plus_dm, minus_dm
    atr = tr.rolling(window=window, min_periods=window).sum().shift()  # to align
    plus_di = 100 * (plus_dm.rolling(window=window, min_periods=window).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window=window, min_periods=window).sum() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=window, min_periods=window).mean()
    return adx

def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> tuple[pd.Series, pd.Series]:
    """
    Compute the Stochastic Oscillator %K and %D lines.
    Returns a tuple (%K, %D).
    """
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)
    lowest_low = low.rolling(window=k_window, min_periods=k_window).min()
    highest_high = high.rolling(window=k_window, min_periods=k_window).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_window, min_periods=d_window).mean()
    return k, d

def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """
    Compute the Commodity Channel Index (CCI).
    Returns a pandas Series with CCI values.
    """
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)
    typical_price = (high + low + close) / 3.0
    sma_tp = typical_price.rolling(window=window, min_periods=window).mean()
    mean_dev = typical_price.rolling(window=window, min_periods=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (typical_price - sma_tp) / (0.015 * mean_dev)
    return cci

def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Williams %R - Oscilador de momentum.
    Varia de -100 a 0.
    Acima de -20 = overbought, abaixo de -80 = oversold.
    """
    highest_high = high.rolling(window=window).max()
    lowest_low = low.rolling(window=window).min()
    denom = highest_high - lowest_low
    denom = denom.replace(0, float('nan'))
    wr = ((highest_high - close) / denom) * -100
    return wr.fillna(-50)

def stochastic_rsi(
    series: pd.Series,
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_smooth: int = 3,
    d_smooth: int = 3
) -> tuple[pd.Series, pd.Series]:
    """
    Stochastic RSI - RSI aplicado sobre o próprio RSI.
    Mais sensível que RSI ou Stochastic isolados.
    Retorna (K, D) onde ambos variam de 0 a 100.
    K > 80 = overbought, K < 20 = oversold.
    Crossover K/D gera sinais.
    """
    rsi_values = rsi(series, window=rsi_period)
    min_rsi = rsi_values.rolling(window=stoch_period).min()
    max_rsi = rsi_values.rolling(window=stoch_period).max()
    denom = max_rsi - min_rsi
    denom = denom.replace(0, float('nan'))
    stoch_rsi_raw = ((rsi_values - min_rsi) / denom) * 100
    stoch_rsi_raw = stoch_rsi_raw.fillna(50)
    k = stoch_rsi_raw.rolling(window=k_smooth).mean()
    d = k.rolling(window=d_smooth).mean()
    return k, d


def twap(close_prices: pd.Series) -> float:
    """
    TWAP (Time-Weighted Average Price).
    Preço médio ponderado pelo tempo — cada barra tem peso igual.
    Divergência TWAP vs VWAP indica absorção institucional:
      - TWAP > VWAP → preço ficou mais tempo em cima (absorção compradora)
      - TWAP < VWAP → preço ficou mais tempo embaixo (absorção vendedora)

    Args:
        close_prices: Série de preços de fechamento da sessão/período

    Returns:
        Valor TWAP (float)
    """
    if close_prices.empty:
        return 0.0
    return float(close_prices.mean())



def twap_validated(
    closes: np.ndarray,
    volumes: np.ndarray,
    low: float,
    high: float,
    symbol: str = "UNKNOWN"
) -> dict:
    """
    FIX #2: Calcula TWAP com validação de bounds e fallback para VWAP.
    
    Garante que low <= TWAP <= high (dentro de tolerância de 1%).
    Se TWAP violar bounds, usa VWAP como fallback.
    
    Args:
        closes: Array de preços de fechamento
        volumes: Array de volumes
        low: Baixa do período
        high: Alta do período
        symbol: Símbolo para logging
        
    Returns:
        Dict com TWAP, VWAP, validação e valor final
    """
    return TWAPValidator.validate_twap_with_fallback(closes, volumes, low, high, symbol)

def twap_vwap_analysis(
    close_prices: pd.Series,
    volumes: pd.Series,
    current_vwap: Optional[float] = None
) -> dict:
    """
    Calcula TWAP e compara com VWAP para detectar absorção.

    Args:
        close_prices: Série de preços de fechamento
        volumes: Série de volumes correspondentes
        current_vwap: VWAP pré-calculado (opcional, calcula se não fornecido)

    Returns:
        Dict com TWAP, VWAP, divergência e sinal
    """
    if close_prices.empty or len(close_prices) < 2:
        return {
            "twap": 0.0,
            "vwap": 0.0,
            "divergence_pct": 0.0,
            "signal": "insufficient_data",
        }

    twap_value = float(close_prices.mean())

    if current_vwap is not None and current_vwap > 0:
        vwap_value = current_vwap
    else:
        # Calcular VWAP
        total_volume = volumes.sum()
        if total_volume > 0:
            vwap_value = float((close_prices * volumes).sum() / total_volume)
        else:
            vwap_value = twap_value

    # Divergência
    if vwap_value > 0:
        divergence_pct = round((twap_value - vwap_value) / vwap_value * 100, 6)
    else:
        divergence_pct = 0.0

    # Classificação do sinal
    if abs(divergence_pct) < 0.005:
        signal = "neutral"
    elif divergence_pct > 0.02:
        signal = "strong_buy_absorption"
    elif divergence_pct > 0:
        signal = "slight_buy_absorption"
    elif divergence_pct < -0.02:
        signal = "strong_sell_absorption"
    else:
        signal = "slight_sell_absorption"

    return {
        "twap": round(twap_value, 2),
        "vwap": round(vwap_value, 2),
        "divergence_pct": divergence_pct,
        "signal": signal,
        "interpretation": (
            "Price spent more time at higher levels (buy absorption)"
            if divergence_pct > 0
            else "Price spent more time at lower levels (sell absorption)"
            if divergence_pct < 0
            else "Price time-balanced (no absorption detected)"
        ),
    }


def realized_volatility(series: pd.Series, window: int = 30) -> pd.Series:
    """
    Compute the realized volatility of a series of prices.
    Volatility is the standard deviation of log returns over the specified window.
    Returns a pandas Series of annualized volatility.
    """
    series = series.astype(float)
    log_returns = pd.Series(np.log(series / series.shift()), index=series.index)
    vol = log_returns.rolling(window=window, min_periods=window).std(ddof=0)
    # annualize assuming 365 days and 24*60/5 minute bars: adjust if sampling different frequency
    return vol * np.sqrt(window)

def detect_regime(adx_series: pd.Series, vol_series: pd.Series, adx_threshold: float = 25, vol_threshold: float = 0.02) -> pd.Series:
    """
    Simple regime detection combining ADX and volatility.
    If ADX > adx_threshold and volatility < vol_threshold: Trending.
    If ADX <= adx_threshold and volatility < vol_threshold: Range-bound.
    If volatility >= vol_threshold: High-volatility regime.
    Returns a pandas Series of strings.
    """
    def classify_row(row):
        if row['vol'] >= vol_threshold:
            return 'High Volatility'
        if row['adx'] > adx_threshold:
            return 'Trending'
        return 'Range'
    df = pd.DataFrame({'adx': adx_series, 'vol': vol_series})
    regime = df.apply(classify_row, axis=1)
    return regime


# ==============================================================================
# MÉTODOS INSTITUCIONAIS AVANÇADOS
# ==============================================================================

def hurst_exponent(prices: list, max_lag: int = 50) -> Optional[float]:
    """
    Hurst Exponent via R/S analysis.
    H > 0.5 = trending (persistente)
    H < 0.5 = mean-reverting
    H ≈ 0.5 = random walk
    """
    if len(prices) < max_lag * 2:
        return None
    try:
        ts = np.array(prices, dtype=float)
        lags = range(2, min(max_lag, len(ts) // 2))
        tau = []
        for lag in lags:
            diffs = ts[lag:] - ts[:-lag]
            std = float(np.std(diffs))
            tau.append(std if std > 1e-12 else 1e-12)
        if len(tau) < 2:
            return None
        x = np.log(list(lags)[:len(tau)])
        y = np.log(tau)
        reg = np.polyfit(x, y, 1)
        return round(float(reg[0]), 4)
    except Exception:
        return None


def shannon_entropy(returns: list, bins: int = 20) -> Optional[float]:
    """
    Entropia de Shannon dos retornos.
    Baixa = mercado previsível, Alta = ruído/incerteza.
    """
    if len(returns) < 30:
        return None
    try:
        arr = np.array(returns, dtype=float)
        hist, _ = np.histogram(arr, bins=bins)
        hist = hist[hist > 0]
        probs = hist / float(hist.sum())
        return round(float(-np.sum(probs * np.log2(probs))), 4)
    except Exception:
        return None


def simple_kalman_filter(prices: list, Q: float = 1e-5, R: float = 0.01) -> Optional[dict]:
    """
    Kalman filter escalar para suavização de preço.
    Q = ruído do processo, R = ruído de medição.
    """
    if len(prices) < 10:
        return None
    try:
        x = float(prices[0])
        P = 1.0
        for z in prices:
            P_pred = P + Q
            K = P_pred / (P_pred + R)
            x = x + K * (float(z) - x)
            P = (1.0 - K) * P_pred
        raw = float(prices[-1])
        deviation_pct = round((raw - x) / x * 100, 4) if x > 0 else 0.0
        trend_dir = "UP" if raw > x * 1.0001 else "DOWN" if raw < x * 0.9999 else "FLAT"
        return {
            "kalman_price": round(x, 2),
            "raw_price": round(raw, 2),
            "deviation_pct": deviation_pct,
            "trend_direction": trend_dir,
        }
    except Exception:
        return None


def regression_channel(prices: list, window: int = 50) -> Optional[dict]:
    """
    Canal de regressão linear com bandas ±1σ e ±2σ.
    """
    if len(prices) < window:
        return None
    try:
        y = np.array(prices[-window:], dtype=float)
        x = np.arange(window, dtype=float)
        coeffs = np.polyfit(x, y, 1)
        trend_line = np.polyval(coeffs, x)
        residuals = y - trend_line
        std = float(np.std(residuals))
        current_trend = float(trend_line[-1])
        pos = float((prices[-1] - (current_trend - 2 * std)) / (4 * std)) if std > 0 else 0.5
        return {
            "slope_per_bar": round(float(coeffs[0]), 4),
            "trend_price": round(current_trend, 2),
            "upper_1sd": round(current_trend + std, 2),
            "lower_1sd": round(current_trend - std, 2),
            "upper_2sd": round(current_trend + 2 * std, 2),
            "lower_2sd": round(current_trend - 2 * std, 2),
            "deviation_from_trend": round(float(prices[-1]) - current_trend, 2),
            "position_in_channel": round(max(0.0, min(1.0, pos)), 4),
        }
    except Exception:
        return None


def dominant_cycles(prices: list, min_period: int = 5, max_period: int = 100) -> Optional[dict]:
    """
    Ciclos dominantes via FFT. Encontra os 3 períodos com maior amplitude.
    """
    if len(prices) < max_period * 2:
        return None
    try:
        arr = np.array(prices, dtype=float)
        fft_vals = np.fft.fft(arr - np.mean(arr))
        n = len(arr)
        freqs = np.fft.fftfreq(n)
        magnitudes = np.abs(fft_vals[1: n // 2])
        periods = 1.0 / np.abs(freqs[1: n // 2] + 1e-12)
        mask = (periods >= min_period) & (periods <= max_period)
        if not np.any(mask):
            return None
        m_masked = magnitudes[mask]
        p_masked = periods[mask]
        top_idx = np.argsort(m_masked)[-3:][::-1]
        return {
            "dominant_cycles": [round(float(p_masked[i]), 1) for i in top_idx],
            "cycle_strengths": [round(float(m_masked[i]), 2) for i in top_idx],
        }
    except Exception:
        return None


def fractal_dimension(prices: list) -> Optional[float]:
    """
    Dimensão fractal (método de Higuchi aproximado).
    < 1.5 = trending, > 1.5 = ruído/mean-reverting.
    """
    if len(prices) < 50:
        return None
    try:
        arr = np.array(prices, dtype=float)
        N = len(arr)
        max_k = min(N // 4, 50)
        lengths = []
        for k in range(1, max_k):
            segs = [arr[m::k] for m in range(k) if len(arr[m::k]) > 1]
            if not segs:
                continue
            Lk = np.mean([
                np.sum(np.abs(np.diff(s))) * (N - 1) / (((N - m) // k) * k + 1e-12)
                for m, s in enumerate(segs)
            ])
            if Lk > 0:
                lengths.append((np.log(1.0 / k), np.log(Lk)))
        if len(lengths) < 2:
            return None
        x, y = zip(*lengths)
        return round(float(np.polyfit(x, y, 1)[0]), 4)
    except Exception:
        return None


def monte_carlo_forecast(
    returns: list,
    current_price: float,
    n_sims: int = 1000,
    horizon: int = 12,
) -> Optional[dict]:
    """
    Monte Carlo: simula n_sims cenários de preço para 'horizon' passos.
    Retorna percentis e probabilidade de alta.
    """
    if len(returns) < 30 or current_price <= 0:
        return None
    try:
        ret = np.array(returns, dtype=float)
        mu = float(np.mean(ret))
        sigma = float(np.std(ret))
        rng = np.random.default_rng(seed=42)
        sims = rng.normal(mu, sigma, (n_sims, horizon))
        final_prices = current_price * np.exp(np.cumsum(sims, axis=1)[:, -1])
        return {
            "median_price": round(float(np.median(final_prices)), 2),
            "p10": round(float(np.percentile(final_prices, 10)), 2),
            "p25": round(float(np.percentile(final_prices, 25)), 2),
            "p75": round(float(np.percentile(final_prices, 75)), 2),
            "p90": round(float(np.percentile(final_prices, 90)), 2),
            "prob_up": round(float(np.mean(final_prices > current_price)), 4),
            "horizon_bars": horizon,
        }
    except Exception:
        return None
```


## 5.2 institutional/cvd.py (completo)


```python
# institutional/cvd.py
"""
Cumulative Volume Delta (CVD)

Acumula a diferença entre volume de compra agressiva e venda agressiva.
Divergências entre CVD e preço antecipam reversões.

Método #3 do Arsenal Institucional.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from institutional.base import (
    AnalysisResult,
    InsufficientDataError,
    InvalidParameterError,
    MarketRegime,
    Side,
    Signal,
    SignalStrength,
    Trade,
)


@dataclass
class CVDBar:
    """Barra de CVD."""
    timestamp: float
    buy_volume: float
    sell_volume: float
    delta: float
    cumulative_delta: float
    price_open: float
    price_close: float
    price_high: float
    price_low: float
    trade_count: int


@dataclass
class CVDDivergence:
    """Divergência detectada entre CVD e preço."""
    divergence_type: str  # "bullish" or "bearish"
    price_direction: str  # "up" or "down"
    cvd_direction: str    # "up" or "down"
    strength: float       # 0.0 a 1.0
    bars_count: int       # quantas barras compõem a divergência
    start_timestamp: float
    end_timestamp: float


class CVDAnalyzer:
    """
    Analisador de Cumulative Volume Delta.

    Calcula delta de volume (compra - venda agressiva),
    acumula ao longo do tempo e detecta divergências com preço.
    """

    def __init__(
        self,
        bar_interval_seconds: float = 60.0,
        max_bars: int = 1000,
        divergence_lookback: int = 10,
        divergence_threshold: float = 0.6,
    ):
        if bar_interval_seconds <= 0:
            raise InvalidParameterError(
                f"bar_interval_seconds must be > 0, got {bar_interval_seconds}"
            )
        if max_bars < 10:
            raise InvalidParameterError(
                f"max_bars must be >= 10, got {max_bars}"
            )

        self.bar_interval_seconds = bar_interval_seconds
        self.max_bars = max_bars
        self.divergence_lookback = divergence_lookback
        self.divergence_threshold = divergence_threshold

        self.bars: deque[CVDBar] = deque(maxlen=max_bars)
        self.cumulative_delta: float = 0.0

        # Buffer para barra atual
        self._current_bar_start: float = 0.0
        self._bar_initialized: bool = False
        self._current_buy_vol: float = 0.0
        self._current_sell_vol: float = 0.0
        self._current_price_open: float = 0.0
        self._current_price_close: float = 0.0
        self._current_price_high: float = 0.0
        self._current_price_low: float = float("inf")
        self._current_trade_count: int = 0
        self._total_trades_processed: int = 0

    @property
    def total_trades_processed(self) -> int:
        return self._total_trades_processed

    @property
    def bar_count(self) -> int:
        return len(self.bars)

    def process_trade(self, trade: Trade) -> Optional[CVDBar]:
        """
        Processa um trade e retorna CVDBar se uma barra foi completada.
        """
        self._total_trades_processed += 1

        completed_bar = None

        # Inicializar barra se necessário
        if not self._bar_initialized:
            self._current_bar_start = trade.timestamp
            self._current_price_open = trade.price
            self._current_price_high = trade.price
            self._current_price_low = trade.price
            self._bar_initialized = True
        else:
            # Verificar se barra completou ANTES de adicionar o trade
            elapsed = trade.timestamp - self._current_bar_start
            if elapsed >= self.bar_interval_seconds:
                completed_bar = self._close_bar(trade.timestamp)
                # Iniciar nova barra com este trade
                self._current_price_open = trade.price
                self._current_price_high = trade.price
                self._current_price_low = trade.price
                self._bar_initialized = True

        # Atualizar preço
        self._current_price_close = trade.price
        self._current_price_high = max(self._current_price_high, trade.price)
        self._current_price_low = min(self._current_price_low, trade.price)
        self._current_trade_count += 1

        # Acumular volume por lado
        if trade.side == Side.BUY:
            self._current_buy_vol += trade.quantity
        elif trade.side == Side.SELL:
            self._current_sell_vol += trade.quantity

        return completed_bar

    def process_trades(self, trades: list[Trade]) -> list[CVDBar]:
        """Processa lista de trades, retorna barras completadas."""
        completed: list[CVDBar] = []
        for trade in trades:
            bar = self.process_trade(trade)
            if bar is not None:
                completed.append(bar)
        return completed

    def _close_bar(self, timestamp: float) -> CVDBar:
        """Fecha a barra atual e inicia nova."""
        delta = self._current_buy_vol - self._current_sell_vol
        self.cumulative_delta += delta

        bar = CVDBar(
            timestamp=self._current_bar_start,
            buy_volume=self._current_buy_vol,
            sell_volume=self._current_sell_vol,
            delta=delta,
            cumulative_delta=self.cumulative_delta,
            price_open=self._current_price_open,
            price_close=self._current_price_close,
            price_high=self._current_price_high,
            price_low=self._current_price_low,
            trade_count=self._current_trade_count,
        )

        self.bars.append(bar)
        self._reset_current_bar(timestamp)
        return bar

    def _reset_current_bar(self, timestamp: float) -> None:
        """Reseta buffer da barra atual."""
        self._current_bar_start = timestamp
        self._bar_initialized = False
        self._current_buy_vol = 0.0
        self._current_sell_vol = 0.0
        self._current_price_open = 0.0
        self._current_price_close = 0.0
        self._current_price_high = 0.0
        self._current_price_low = float("inf")
        self._current_trade_count = 0

    def get_current_delta(self) -> float:
        """Delta da barra atual (ainda não fechada)."""
        return self._current_buy_vol - self._current_sell_vol

    def get_current_cvd(self) -> float:
        """CVD total incluindo barra atual não fechada."""
        return self.cumulative_delta + self.get_current_delta()

    def detect_divergences(
        self,
        lookback: Optional[int] = None,
    ) -> list[CVDDivergence]:
        """
        Detecta divergências entre CVD e preço.

        Divergência bullish: preço faz lower low, CVD faz higher low
        Divergência bearish: preço faz higher high, CVD faz lower high
        """
        lookback = lookback or self.divergence_lookback

        if len(self.bars) < lookback + 1:
            return []

        recent_bars = list(self.bars)[-lookback:]
        divergences: list[CVDDivergence] = []

        # Calcular direções
        price_start = recent_bars[0].price_close
        price_end = recent_bars[-1].price_close
        cvd_start = recent_bars[0].cumulative_delta
        cvd_end = recent_bars[-1].cumulative_delta

        price_change = (price_end - price_start) / price_start if price_start else 0
        cvd_change = cvd_end - cvd_start

        price_direction = "up" if price_change > 0 else "down"
        cvd_direction = "up" if cvd_change > 0 else "down"

        # Divergência: preço e CVD vão em direções opostas
        if price_direction != cvd_direction:
            # Calcular força da divergência
            strength = min(abs(price_change) * 100, 1.0)

            if price_direction == "down" and cvd_direction == "up":
                div_type = "bullish"
            else:
                div_type = "bearish"

            divergences.append(
                CVDDivergence(
                    divergence_type=div_type,
                    price_direction=price_direction,
                    cvd_direction=cvd_direction,
                    strength=strength,
                    bars_count=lookback,
                    start_timestamp=recent_bars[0].timestamp,
                    end_timestamp=recent_bars[-1].timestamp,
                )
            )

        return divergences

    def analyze(self) -> AnalysisResult:
        """Análise completa do CVD atual."""
        result = AnalysisResult(
            source="cvd_analyzer",
            timestamp=time.time(),
        )

        if len(self.bars) < 3:
            result.confidence = 0.0
            return result

        recent = list(self.bars)[-5:] if len(self.bars) >= 5 else list(self.bars)

        # Métricas
        current_delta = self.get_current_delta()
        total_buy_vol = sum(b.buy_volume for b in recent)
        total_sell_vol = sum(b.sell_volume for b in recent)
        avg_delta = sum(b.delta for b in recent) / len(recent)

        result.metrics = {
            "cumulative_delta": self.cumulative_delta,
            "current_bar_delta": current_delta,
            "avg_delta_5bars": avg_delta,
            "total_buy_volume": total_buy_vol,
            "total_sell_volume": total_sell_vol,
            "buy_sell_ratio": (
                total_buy_vol / total_sell_vol if total_sell_vol > 0 else 0
            ),
            "bar_count": len(self.bars),
        }

        # Divergências
        divergences = self.detect_divergences()
        for div in divergences:
            signal_dir = Side.BUY if div.divergence_type == "bullish" else Side.SELL

            strength = SignalStrength.STRONG
            if div.strength < 0.3:
                strength = SignalStrength.WEAK
            elif div.strength < 0.6:
                strength = SignalStrength.MODERATE

            result.signals.append(
                Signal(
                    timestamp=div.end_timestamp,
                    signal_type=f"cvd_divergence_{div.divergence_type}",
                    direction=signal_dir,
                    strength=strength,
                    price=recent[-1].price_close,
                    confidence=div.strength,
                    source="cvd_analyzer",
                    description=(
                        f"CVD {div.divergence_type} divergence: "
                        f"price {div.price_direction}, "
                        f"CVD {div.cvd_direction} "
                        f"over {div.bars_count} bars"
                    ),
                )
            )

        # Sinal de pressão dominante
        if total_buy_vol > 0 or total_sell_vol > 0:
            ratio = total_buy_vol / max(total_sell_vol, 1e-10)
            if ratio > 3.0:
                result.signals.append(
                    Signal(
                        timestamp=time.time(),
                        signal_type="cvd_buy_pressure",
                        direction=Side.BUY,
                        strength=SignalStrength.STRONG,
                        price=recent[-1].price_close,
                        confidence=min(ratio / 5.0, 1.0),
                        source="cvd_analyzer",
                        description=f"Strong buy pressure: ratio {ratio:.1f}x",
                    )
                )
            elif ratio < 0.33:
                result.signals.append(
                    Signal(
                        timestamp=time.time(),
                        signal_type="cvd_sell_pressure",
                        direction=Side.SELL,
                        strength=SignalStrength.STRONG,
                        price=recent[-1].price_close,
                        confidence=min(1.0 / max(ratio, 1e-10) / 5.0, 1.0),
                        source="cvd_analyzer",
                        description=(
                            f"Strong sell pressure: ratio {1/max(ratio,1e-10):.1f}x"
                        ),
                    )
                )

        result.confidence = max(
            (s.confidence for s in result.signals), default=0.0
        )

        return result

    def reset(self) -> None:
        """Reseta todo o estado."""
        self.bars.clear()
        self.cumulative_delta = 0.0
        self._reset_current_bar(0.0)
        self._total_trades_processed = 0
```


## 5.3 institutional/order_flow_imbalance.py (completo)


```python
# institutional/order_flow_imbalance.py
"""
Order Flow Imbalance (OFI)

Calcula desequilíbrio entre volume comprador e vendedor
em cada barra de preço. Quando um lado domina 300%+,
sinaliza movimento iminente.

Método #5 do Arsenal Institucional.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from institutional.base import (
    AnalysisResult,
    InvalidParameterError,
    Side,
    Signal,
    SignalStrength,
    Trade,
)


@dataclass
class OFIBar:
    """Barra de Order Flow Imbalance."""
    timestamp: float
    buy_volume: float
    sell_volume: float
    buy_count: int
    sell_count: int
    imbalance_ratio: float  # buy/sell ratio
    price_open: float
    price_close: float
    price_change_pct: float
    dominant_side: Side

    @property
    def net_volume(self) -> float:
        return self.buy_volume - self.sell_volume

    @property
    def total_volume(self) -> float:
        return self.buy_volume + self.sell_volume


@dataclass
class OFIAlert:
    """Alerta de desequilíbrio."""
    timestamp: float
    ratio: float
    side: Side
    volume: float
    price: float
    severity: str  # "extreme", "strong", "moderate"


class OrderFlowImbalanceAnalyzer:
    """
    Analisador de desequilíbrio de fluxo de ordens.

    Monitora ratio compra/venda em tempo real.
    Ratio >= 3.0 indica desequilíbrio forte.
    Ratio >= 5.0 indica desequilíbrio extremo.
    """

    # Thresholds de desequilíbrio
    MODERATE_THRESHOLD = 2.0
    STRONG_THRESHOLD = 3.0
    EXTREME_THRESHOLD = 5.0

    def __init__(
        self,
        window_seconds: float = 30.0,
        max_history: int = 500,
        alert_cooldown_seconds: float = 60.0,
    ):
        if window_seconds <= 0:
            raise InvalidParameterError(
                f"window_seconds must be > 0, got {window_seconds}"
            )

        self.window_seconds = window_seconds
        self.max_history = max_history
        self.alert_cooldown = alert_cooldown_seconds

        self.bars: deque[OFIBar] = deque(maxlen=max_history)
        self.alerts: list[OFIAlert] = []

        # Buffer da janela atual
        self._window_start: float = 0.0
        self._window_initialized: bool = False
        self._buy_vol: float = 0.0
        self._sell_vol: float = 0.0
        self._buy_count: int = 0
        self._sell_count: int = 0
        self._price_open: float = 0.0
        self._price_close: float = 0.0
        self._last_alert_time: float = 0.0
        self._total_trades: int = 0

    @property
    def total_trades_processed(self) -> int:
        return self._total_trades

    def process_trade(self, trade: Trade) -> Optional[OFIBar]:
        """Processa trade, retorna OFIBar se janela fechou."""
        self._total_trades += 1

        completed_bar = None

        if not self._window_initialized:
            self._window_start = trade.timestamp
            self._price_open = trade.price
            self._window_initialized = True
        else:
            # Verificar se janela fechou ANTES de adicionar o trade
            elapsed = trade.timestamp - self._window_start
            if elapsed >= self.window_seconds:
                completed_bar = self._close_window(trade.timestamp)

        # Acumular
        if trade.side == Side.BUY:
            self._buy_vol += trade.quantity
            self._buy_count += 1
        elif trade.side == Side.SELL:
            self._sell_vol += trade.quantity
            self._sell_count += 1

        self._price_close = trade.price

        return completed_bar

    def _close_window(self, timestamp: float) -> OFIBar:
        """Fecha janela e calcula métricas."""
        sell_vol = max(self._sell_vol, 1e-10)
        buy_vol = max(self._buy_vol, 1e-10)

        imbalance_ratio = self._buy_vol / sell_vol if self._sell_vol > 0 else (
            float("inf") if self._buy_vol > 0 else 1.0
        )

        if self._buy_vol > self._sell_vol:
            dominant = Side.BUY
        elif self._sell_vol > self._buy_vol:
            dominant = Side.SELL
        else:
            dominant = Side.UNKNOWN

        price_change = 0.0
        if self._price_open > 0:
            price_change = (
                (self._price_close - self._price_open) / self._price_open
            ) * 100

        bar = OFIBar(
            timestamp=self._window_start,
            buy_volume=self._buy_vol,
            sell_volume=self._sell_vol,
            buy_count=self._buy_count,
            sell_count=self._sell_count,
            imbalance_ratio=imbalance_ratio,
            price_open=self._price_open,
            price_close=self._price_close,
            price_change_pct=price_change,
            dominant_side=dominant,
        )

        self.bars.append(bar)

        # Verificar se gera alerta
        self._check_alert(bar)

        # Reset
        self._window_start = timestamp
        self._window_initialized = False
        self._buy_vol = 0.0
        self._sell_vol = 0.0
        self._buy_count = 0
        self._sell_count = 0
        self._price_open = self._price_close
        # _price_close mantém o último valor

        return bar

    def _check_alert(self, bar: OFIBar) -> None:
        """Verifica se barra gera alerta."""
        ratio = bar.imbalance_ratio
        inverse_ratio = 1.0 / ratio if ratio > 0 else float("inf")

        effective_ratio = max(ratio, inverse_ratio)
        if effective_ratio == float("inf"):
            effective_ratio = 100.0

        if effective_ratio < self.MODERATE_THRESHOLD:
            return

        # Cooldown
        if (bar.timestamp - self._last_alert_time) < self.alert_cooldown:
            return

        if effective_ratio >= self.EXTREME_THRESHOLD:
            severity = "extreme"
        elif effective_ratio >= self.STRONG_THRESHOLD:
            severity = "strong"
        else:
            severity = "moderate"

        alert = OFIAlert(
            timestamp=bar.timestamp,
            ratio=effective_ratio,
            side=bar.dominant_side,
            volume=bar.total_volume,
            price=bar.price_close,
            severity=severity,
        )

        self.alerts.append(alert)
        self._last_alert_time = bar.timestamp

    def get_current_imbalance(self) -> dict:
        """Imbalance da janela atual (não fechada)."""
        sell_vol = max(self._sell_vol, 1e-10)
        ratio = self._buy_vol / sell_vol

        if self._buy_vol > self._sell_vol:
            dominant = Side.BUY
        elif self._sell_vol > self._buy_vol:
            dominant = Side.SELL
        else:
            dominant = Side.UNKNOWN

        return {
            "buy_volume": self._buy_vol,
            "sell_volume": self._sell_vol,
            "ratio": ratio,
            "dominant_side": dominant.value,
            "buy_count": self._buy_count,
            "sell_count": self._sell_count,
        }

    def get_trend_strength(self, lookback: int = 5) -> dict:
        """
        Calcula força da tendência baseado em barras recentes.

        Retorna score de -1.0 (venda extrema) a +1.0 (compra extrema).
        """
        if len(self.bars) < lookback:
            return {"score": 0.0, "direction": "neutral", "confidence": 0.0}

        recent = list(self.bars)[-lookback:]

        buy_dominant_count = sum(
            1 for b in recent if b.dominant_side == Side.BUY
        )
        sell_dominant_count = sum(
            1 for b in recent if b.dominant_side == Side.SELL
        )

        avg_ratio = sum(b.imbalance_ratio for b in recent) / len(recent)

        if buy_dominant_count > sell_dominant_count:
            score = buy_dominant_count / lookback
            direction = "bullish"
        elif sell_dominant_count > buy_dominant_count:
            score = -(sell_dominant_count / lookback)
            direction = "bearish"
        else:
            score = 0.0
            direction = "neutral"

        return {
            "score": score,
            "direction": direction,
            "avg_ratio": avg_ratio,
            "buy_dominant_bars": buy_dominant_count,
            "sell_dominant_bars": sell_dominant_count,
            "confidence": abs(score),
        }

    def analyze(self) -> AnalysisResult:
        """Análise completa."""
        result = AnalysisResult(
            source="ofi_analyzer",
            timestamp=time.time(),
        )

        if len(self.bars) < 2:
            result.confidence = 0.0
            return result

        trend = self.get_trend_strength()
        current = self.get_current_imbalance()
        last_bar = self.bars[-1]

        result.metrics = {
            "current_ratio": current["ratio"],
            "current_dominant": current["dominant_side"],
            "trend_score": trend["score"],
            "trend_direction": trend["direction"],
            "avg_ratio": trend["avg_ratio"],
            "recent_alerts": len([
                a for a in self.alerts
                if time.time() - a.timestamp < 300
            ]),
            "bar_count": len(self.bars),
        }

        # Sinais baseados em tendência
        if abs(trend["score"]) > 0.6:
            direction = Side.BUY if trend["score"] > 0 else Side.SELL
            result.signals.append(
                Signal(
                    timestamp=time.time(),
                    signal_type="ofi_trend",
                    direction=direction,
                    strength=SignalStrength.STRONG if abs(trend["score"]) > 0.8 else SignalStrength.MODERATE,
                    price=last_bar.price_close,
                    confidence=abs(trend["score"]),
                    source="ofi_analyzer",
                    description=(
                        f"OFI trend {trend['direction']}: "
                        f"score={trend['score']:.2f}"
                    ),
                )
            )

        # Sinais de alertas recentes
        recent_alerts = [
            a for a in self.alerts
            if time.time() - a.timestamp < 120
        ]
        for alert in recent_alerts:
            result.signals.append(
                Signal(
                    timestamp=alert.timestamp,
                    signal_type=f"ofi_imbalance_{alert.severity}",
                    direction=alert.side,
                    strength=(
                        SignalStrength.STRONG if alert.severity == "extreme"
                        else SignalStrength.MODERATE
                    ),
                    price=alert.price,
                    confidence=min(alert.ratio / 10.0, 1.0),
                    source="ofi_analyzer",
                    description=(
                        f"{alert.severity.upper()} imbalance: "
                        f"{alert.ratio:.1f}x {alert.side.value}"
                    ),
                )
            )

        result.confidence = max(
            (s.confidence for s in result.signals), default=0.0
        )

        return result

    def reset(self) -> None:
        """Reseta estado."""
        self.bars.clear()
        self.alerts.clear()
        self._window_start = 0.0
        self._window_initialized = False
        self._buy_vol = 0.0
        self._sell_vol = 0.0
        self._buy_count = 0
        self._sell_count = 0
        self._price_open = 0.0
        self._price_close = 0.0
        self._last_alert_time = 0.0
        self._total_trades = 0
```


## 5.4 institutional/vwap_twap.py (completo)


```python
# institutional/vwap_twap.py
"""
VWAP (Volume Weighted Average Price) e TWAP (Time Weighted Average Price)

VWAP: Benchmark institucional de execução.
TWAP: Preço médio ao longo do tempo.

Métodos #11 e #12 do Arsenal Institucional.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from institutional.base import (
    AnalysisResult,
    InvalidParameterError,
    Side,
    Signal,
    SignalStrength,
)


@dataclass
class VWAPBand:
    """Bandas de desvio padrão ao redor do VWAP."""
    vwap: float
    upper_1sd: float
    lower_1sd: float
    upper_2sd: float
    lower_2sd: float
    upper_3sd: float
    lower_3sd: float


@dataclass
class PriceVolume:
    """Par preço-volume."""
    timestamp: float
    price: float
    volume: float


class VWAPCalculator:
    """
    Calculador de VWAP com bandas de desvio padrão.

    VWAP = Σ(Preço × Volume) / Σ(Volume)

    Institucional usa como:
    - Benchmark de execução (comprou abaixo = boa execução)
    - Suporte/resistência dinâmico
    - Desvios padrão = zonas de sobrecompra/sobrevenda
    """

    def __init__(
        self,
        anchor_period: str = "session",
        max_data_points: int = 5000,
    ):
        self.anchor_period = anchor_period
        self.max_points = max_data_points

        self._data: deque[PriceVolume] = deque(maxlen=max_data_points)
        self._cumulative_pv: float = 0.0  # Σ(price * volume)
        self._cumulative_vol: float = 0.0  # Σ(volume)
        self._cumulative_pv2: float = 0.0  # Σ(price² * volume) para desvio

    @property
    def vwap(self) -> float:
        """VWAP atual."""
        if self._cumulative_vol > 0:
            return self._cumulative_pv / self._cumulative_vol
        return 0.0

    @property
    def data_points(self) -> int:
        return len(self._data)

    def add_candle(
        self,
        timestamp: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> float:
        """
        Adiciona candle e retorna VWAP atualizado.
        Usa preço típico = (High + Low + Close) / 3
        """
        typical_price = (high + low + close) / 3.0
        return self.add_price_volume(timestamp, typical_price, volume)

    def add_price_volume(
        self,
        timestamp: float,
        price: float,
        volume: float,
    ) -> float:
        """Adiciona par preço-volume e retorna VWAP."""
        self._data.append(PriceVolume(timestamp, price, volume))
        self._cumulative_pv += price * volume
        self._cumulative_vol += volume
        self._cumulative_pv2 += (price ** 2) * volume

        return self.vwap

    def get_bands(self) -> VWAPBand:
        """
        Calcula VWAP com bandas de desvio padrão.

        Desvio padrão ponderado pelo volume.
        """
        vwap_val = self.vwap
        if self._cumulative_vol <= 0:
            return VWAPBand(0, 0, 0, 0, 0, 0, 0)

        # Variância ponderada
        variance = (
            self._cumulative_pv2 / self._cumulative_vol
        ) - (vwap_val ** 2)

        # Proteção contra variância negativa (arredondamento)
        variance = max(variance, 0.0)
        std_dev = variance ** 0.5

        return VWAPBand(
            vwap=vwap_val,
            upper_1sd=vwap_val + std_dev,
            lower_1sd=vwap_val - std_dev,
            upper_2sd=vwap_val + 2 * std_dev,
            lower_2sd=vwap_val - 2 * std_dev,
            upper_3sd=vwap_val + 3 * std_dev,
            lower_3sd=vwap_val - 3 * std_dev,
        )

    def get_deviation(self, current_price: float) -> dict:
        """
        Calcula desvio do preço atual em relação ao VWAP.

        Retorna distância em %, desvios padrão e zona.
        """
        bands = self.get_bands()
        vwap_val = bands.vwap

        if vwap_val <= 0:
            return {
                "deviation_pct": 0.0,
                "std_devs": 0.0,
                "zone": "no_data",
            }

        deviation_pct = ((current_price - vwap_val) / vwap_val) * 100
        std_dev = bands.upper_1sd - vwap_val

        if std_dev > 0:
            std_devs = (current_price - vwap_val) / std_dev
        else:
            std_devs = 0.0

        # Zona
        if abs(std_devs) < 1:
            zone = "fair_value"
        elif abs(std_devs) < 2:
            zone = "extended" if std_devs > 0 else "oversold"
        elif abs(std_devs) < 3:
            zone = "overbought" if std_devs > 0 else "deeply_oversold"
        else:
            zone = "extreme_overbought" if std_devs > 0 else "extreme_oversold"

        return {
            "deviation_pct": deviation_pct,
            "std_devs": std_devs,
            "zone": zone,
            "vwap": vwap_val,
            "bands": {
                "upper_1sd": bands.upper_1sd,
                "lower_1sd": bands.lower_1sd,
                "upper_2sd": bands.upper_2sd,
                "lower_2sd": bands.lower_2sd,
            },
        }

    def reset(self) -> None:
        """Reseta para nova sessão."""
        self._data.clear()
        self._cumulative_pv = 0.0
        self._cumulative_vol = 0.0
        self._cumulative_pv2 = 0.0


class TWAPCalculator:
    """
    Calculador de TWAP.

    TWAP = Média simples do preço ao longo do tempo.
    Usado para benchmark de execução e como
    suporte/resistência simples.
    """

    def __init__(self, max_data_points: int = 5000):
        self.max_points = max_data_points
        self._prices: deque[PriceVolume] = deque(maxlen=max_data_points)
        self._cumulative_price: float = 0.0

    @property
    def twap(self) -> float:
        """TWAP atual."""
        if self._prices:
            return self._cumulative_price / len(self._prices)
        return 0.0

    @property
    def data_points(self) -> int:
        return len(self._prices)

    def add_price(self, timestamp: float, price: float) -> float:
        """Adiciona preço e retorna TWAP."""
        self._prices.append(PriceVolume(timestamp, price, 0))
        self._cumulative_price += price
        return self.twap

    def get_deviation(self, current_price: float) -> dict:
        """Calcula desvio do preço em relação ao TWAP."""
        twap_val = self.twap
        if twap_val <= 0:
            return {"deviation_pct": 0.0, "twap": 0.0}

        deviation_pct = ((current_price - twap_val) / twap_val) * 100

        return {
            "deviation_pct": deviation_pct,
            "twap": twap_val,
            "above_twap": current_price > twap_val,
        }

    def reset(self) -> None:
        self._prices.clear()
        self._cumulative_price = 0.0


class VWAPTWAPAnalyzer:
    """
    Analisador combinado VWAP + TWAP.
    Gera sinais quando preço desvia significativamente.
    """

    def __init__(self, max_data_points: int = 5000):
        self.vwap = VWAPCalculator(max_data_points=max_data_points)
        self.twap = TWAPCalculator(max_data_points=max_data_points)

    def add_candle(
        self,
        timestamp: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> dict:
        """Adiciona candle e retorna métricas atuais."""
        vwap_val = self.vwap.add_candle(timestamp, high, low, close, volume)
        twap_val = self.twap.add_price(timestamp, close)

        return {
            "vwap": vwap_val,
            "twap": twap_val,
            "price": close,
            "vwap_deviation": self.vwap.get_deviation(close),
            "twap_deviation": self.twap.get_deviation(close),
        }

    def analyze(self, current_price: float) -> AnalysisResult:
        """Análise completa VWAP + TWAP."""
        result = AnalysisResult(
            source="vwap_twap_analyzer",
            timestamp=time.time(),
        )

        vwap_dev = self.vwap.get_deviation(current_price)
        twap_dev = self.twap.get_deviation(current_price)
        bands = self.vwap.get_bands()

        result.metrics = {
            "vwap": bands.vwap,
            "twap": self.twap.twap,
            "price": current_price,
            "vwap_deviation_pct": vwap_dev["deviation_pct"],
            "vwap_std_devs": vwap_dev["std_devs"],
            "vwap_zone": vwap_dev.get("zone", "unknown"),
            "twap_deviation_pct": twap_dev["deviation_pct"],
        }

        # Sinais baseados em VWAP
        std_devs = vwap_dev.get("std_devs", 0)
        if abs(std_devs) >= 2:
            if std_devs >= 2:
                direction = Side.SELL  # Overbought — probabilidade de reverter
                desc = f"Price {std_devs:.1f} std devs ABOVE VWAP — overbought"
            else:
                direction = Side.BUY  # Oversold
                desc = f"Price {abs(std_devs):.1f} std devs BELOW VWAP — oversold"

            strength = (
                SignalStrength.STRONG if abs(std_devs) >= 3
                else SignalStrength.MODERATE
            )

            result.signals.append(
                Signal(
                    timestamp=time.time(),
                    signal_type="vwap_deviation",
                    direction=direction,
                    strength=strength,
                    price=current_price,
                    confidence=min(abs(std_devs) / 4.0, 1.0),
                    source="vwap_twap_analyzer",
                    description=desc,
                )
            )

        # Sinal de convergência VWAP/TWAP
        if bands.vwap > 0 and self.twap.twap > 0:
            vt_diff = abs(bands.vwap - self.twap.twap) / bands.vwap * 100
            if vt_diff > 0.5:
                # VWAP e TWAP divergem = mercado desequilibrado
                if bands.vwap > self.twap.twap:
                    direction = Side.BUY  # Volume-weighted price is higher
                    desc = "VWAP > TWAP — volume concentrated at higher prices"
                else:
                    direction = Side.SELL
                    desc = "VWAP < TWAP — volume concentrated at lower prices"

                result.signals.append(
                    Signal(
                        timestamp=time.time(),
                        signal_type="vwap_twap_divergence",
                        direction=direction,
                        strength=SignalStrength.WEAK,
                        price=current_price,
                        confidence=min(vt_diff / 2.0, 1.0),
                        source="vwap_twap_analyzer",
                        description=desc,
                    )
                )

        result.confidence = max(
            (s.confidence for s in result.signals), default=0.0
        )

        return result

    def reset(self) -> None:
        self.vwap.reset()
        self.twap.reset()
```


## 5.5 institutional/whale_detector.py (completo)


```python
"""
Whale Detector — Detecção de trades de baleias.

Monitora trades em tempo real e identifica
operações de grande volume (whales).

Substituto gratuito dos métodos #34 (Dark Pool) e #35 (Whale Alerts).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from institutional.base import (
    AnalysisResult,
    InvalidParameterError,
    Side,
    Signal,
    SignalStrength,
    Trade,
)


@dataclass
class WhaleEvent:
    """Evento de baleia detectado."""
    timestamp: float
    price: float
    quantity: float
    value_usd: float
    side: Side
    category: str  # "whale", "mega_whale", "institutional"
    percentile: float  # Percentil em relação ao volume médio
    impact_score: float  # 0-1, impacto estimado


@dataclass
class WhaleActivity:
    """Resumo da atividade de baleias em janela."""
    total_whale_volume: float
    total_whale_count: int
    buy_whale_volume: float
    sell_whale_volume: float
    buy_whale_count: int
    sell_whale_count: int
    net_whale_flow: float  # buy - sell
    largest_trade: float
    avg_whale_size: float


class WhaleDetector:
    """
    Detector de trades de baleias.

    Monitora volume de cada trade e categoriza:
    - Retail: abaixo do threshold
    - Whale: acima do threshold (top 1% do volume)
    - Mega Whale: acima de 5x o threshold
    - Institutional: acima de 10x o threshold

    Adaptativo: ajusta thresholds baseado no volume recente.
    """

    def __init__(
        self,
        whale_threshold_usd: float = 100_000.0,
        mega_whale_multiplier: float = 5.0,
        institutional_multiplier: float = 10.0,
        adaptive: bool = True,
        adaptive_percentile: float = 99.0,
        volume_history_size: int = 10000,
        max_events: int = 1000,
    ):
        if whale_threshold_usd <= 0:
            raise InvalidParameterError("whale_threshold_usd must be > 0")

        self.base_threshold = whale_threshold_usd
        self.mega_multiplier = mega_whale_multiplier
        self.institutional_multiplier = institutional_multiplier
        self.adaptive = adaptive
        self.adaptive_percentile = adaptive_percentile

        self._volume_history: deque[float] = deque(maxlen=volume_history_size)
        self._events: deque[WhaleEvent] = deque(maxlen=max_events)
        self._total_trades: int = 0
        self._total_volume: float = 0.0
        self._adaptive_threshold: float = whale_threshold_usd

    @property
    def threshold(self) -> float:
        if self.adaptive and self._adaptive_threshold > 0:
            return self._adaptive_threshold
        return self.base_threshold

    @property
    def events(self) -> list[WhaleEvent]:
        return list(self._events)

    @property
    def total_trades(self) -> int:
        return self._total_trades

    def _update_adaptive_threshold(self) -> None:
        """Atualiza threshold adaptativo baseado no histórico."""
        if len(self._volume_history) < 100:
            return

        sorted_vols = sorted(self._volume_history)
        idx = int(len(sorted_vols) * self.adaptive_percentile / 100)
        idx = min(idx, len(sorted_vols) - 1)

        new_threshold = sorted_vols[idx]
        # Não deixar cair abaixo de metade do base
        self._adaptive_threshold = max(new_threshold, self.base_threshold * 0.5)

    def process_trade(self, trade: Trade) -> Optional[WhaleEvent]:
        """
        Processa trade e retorna WhaleEvent se for baleia.
        """
        self._total_trades += 1
        self._total_volume += trade.value_usd
        self._volume_history.append(trade.value_usd)

        # Atualizar threshold periodicamente
        if self.adaptive and self._total_trades % 1000 == 0:
            self._update_adaptive_threshold()

        # Verificar se é whale
        threshold = self.threshold
        if trade.value_usd < threshold:
            return None

        # Categorizar
        if trade.value_usd >= threshold * self.institutional_multiplier:
            category = "institutional"
        elif trade.value_usd >= threshold * self.mega_multiplier:
            category = "mega_whale"
        else:
            category = "whale"

        # Calcular percentil
        if self._volume_history:
            sorted_vols = sorted(self._volume_history)
            count_below = sum(1 for v in sorted_vols if v <= trade.value_usd)
            percentile = (count_below / len(sorted_vols)) * 100
        else:
            percentile = 99.0

        # Impact score
        avg_vol = self._total_volume / max(self._total_trades, 1)
        impact = min(trade.value_usd / max(avg_vol * 10, 1), 1.0)

        event = WhaleEvent(
            timestamp=trade.timestamp,
            price=trade.price,
            quantity=trade.quantity,
            value_usd=trade.value_usd,
            side=trade.side,
            category=category,
            percentile=percentile,
            impact_score=impact,
        )

        self._events.append(event)
        return event

    def get_whale_activity(
        self,
        window_seconds: float = 300.0,
    ) -> WhaleActivity:
        """Resumo da atividade de baleias na janela."""
        now = time.time()
        recent = [
            e for e in self._events
            if now - e.timestamp <= window_seconds
        ]

        buy_events = [e for e in recent if e.side == Side.BUY]
        sell_events = [e for e in recent if e.side == Side.SELL]

        buy_vol = sum(e.value_usd for e in buy_events)
        sell_vol = sum(e.value_usd for e in sell_events)
        total_vol = buy_vol + sell_vol

        return WhaleActivity(
            total_whale_volume=total_vol,
            total_whale_count=len(recent),
            buy_whale_volume=buy_vol,
            sell_whale_volume=sell_vol,
            buy_whale_count=len(buy_events),
            sell_whale_count=len(sell_events),
            net_whale_flow=buy_vol - sell_vol,
            largest_trade=max((e.value_usd for e in recent), default=0),
            avg_whale_size=total_vol / max(len(recent), 1),
        )

    def get_whale_pressure(self, window_seconds: float = 300.0) -> dict:
        """
        Calcula pressão de baleias (buy vs sell).

        Retorna score de -1.0 (pressão vendedora) a +1.0 (pressão compradora).
        """
        activity = self.get_whale_activity(window_seconds)

        total = activity.buy_whale_volume + activity.sell_whale_volume
        if total == 0:
            return {
                "pressure_score": 0.0,
                "direction": "neutral",
                "confidence": 0.0,
            }

        score = (activity.buy_whale_volume - activity.sell_whale_volume) / total

        if score > 0.3:
            direction = "bullish"
        elif score < -0.3:
            direction = "bearish"
        else:
            direction = "neutral"

        return {
            "pressure_score": score,
            "direction": direction,
            "confidence": abs(score),
            "buy_volume": activity.buy_whale_volume,
            "sell_volume": activity.sell_whale_volume,
            "whale_count": activity.total_whale_count,
        }

    def analyze(self) -> AnalysisResult:
        """Análise completa de atividade de baleias."""
        result = AnalysisResult(
            source="whale_detector",
            timestamp=time.time(),
        )

        activity = self.get_whale_activity(300)
        pressure = self.get_whale_pressure(300)

        result.metrics = {
            "total_trades_analyzed": self._total_trades,
            "whale_events_5min": activity.total_whale_count,
            "whale_volume_5min": activity.total_whale_volume,
            "buy_whale_volume": activity.buy_whale_volume,
            "sell_whale_volume": activity.sell_whale_volume,
            "net_whale_flow": activity.net_whale_flow,
            "whale_pressure": pressure["pressure_score"],
            "largest_trade": activity.largest_trade,
            "current_threshold": self.threshold,
        }

        # Sinal de pressão de baleias
        if abs(pressure["pressure_score"]) > 0.3 and activity.total_whale_count >= 2:
            direction = Side.BUY if pressure["pressure_score"] > 0 else Side.SELL

            result.signals.append(
                Signal(
                    timestamp=time.time(),
                    signal_type="whale_pressure",
                    direction=direction,
                    strength=(
                        SignalStrength.STRONG if abs(pressure["pressure_score"]) > 0.6
                        else SignalStrength.MODERATE
                    ),
                    price=self._events[-1].price if self._events else 0,
                    confidence=abs(pressure["pressure_score"]),
                    source="whale_detector",
                    description=(
                        f"Whale {pressure['direction']} pressure: "
                        f"score={pressure['pressure_score']:.2f}, "
                        f"{activity.total_whale_count} whale trades"
                    ),
                    metadata={
                        "buy_volume": activity.buy_whale_volume,
                        "sell_volume": activity.sell_whale_volume,
                        "whale_count": activity.total_whale_count,
                    },
                )
            )

        # Sinal de trade individual muito grande
        recent_events = [
            e for e in self._events
            if time.time() - e.timestamp < 60
        ]
        for event in recent_events:
            if event.category in ("mega_whale", "institutional"):
                result.signals.append(
                    Signal(
                        timestamp=event.timestamp,
                        signal_type=f"whale_{event.category}",
                        direction=event.side,
                        strength=SignalStrength.STRONG,
                        price=event.price,
                        confidence=event.impact_score,
                        source="whale_detector",
                        description=(
                            f"{event.category.upper()} {event.side.value}: "
                            f"${event.value_usd:,.0f} "
                            f"({event.quantity:.4f} @ {event.price:,.2f})"
                        ),
                    )
                )

        result.confidence = max(
            (s.confidence for s in result.signals), default=0.0
        )

        return result

    def reset(self) -> None:
        """Reseta detector."""
        self._volume_history.clear()
        self._events.clear()
        self._total_trades = 0
        self._total_volume = 0.0
        self._adaptive_threshold = self.base_threshold
```


## 5.6 flow_analyzer/whale_score.py (completo)


```python
"""
Whale Accumulation Score — Detector de acumulação/distribuição institucional.

Score composto de -100 (distribuição forte) a +100 (acumulação forte)
baseado em múltiplas fontes:

  1. Whale/Mid flow direction               → -30 a +30 pontos
  2. Order book depth asymmetry              → -20 a +20 pontos
  3. Absorption pattern bias                 → -25 a +25 pontos
  4. Derivatives context (OI + LSR)          → -25 a +25 pontos

Classificações:
  +50 a +100 → STRONG_ACCUMULATION
  +20 a +49  → MILD_ACCUMULATION
  -19 a +19  → NEUTRAL
  -49 a -20  → MILD_DISTRIBUTION
  -100 a -50 → STRONG_DISTRIBUTION

Uso:
    calculator = WhaleAccumulationCalculator()
    result = calculator.calculate(
        sector_flow={"mid": {"delta": -1.66}, "retail": {"delta": 2.27}},
        orderbook_data={"bid_depth_usd": 552161, "ask_depth_usd": 477276},
        absorption_data={"buyer_strength": 4.5, "seller_exhaustion": 1.0},
        derivatives_data={"BTCUSDT": {"long_short_ratio": 2.42, "open_interest": 79425}},
    )
    print(result["score"])            # ex: 28
    print(result["classification"])   # ex: "MILD_ACCUMULATION"
"""

import logging
import time
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


class WhaleAccumulationCalculator:
    """
    Calcula score de acumulação/distribuição de whales.
    
    Combina sinais de múltiplas fontes em um score único.
    Mantém histórico para detectar tendências de acumulação ao longo do tempo.
    """

    def __init__(self, history_window: int = 30):
        """
        Args:
            history_window: Quantos scores anteriores manter para média móvel.
        """
        self._history: deque = deque(maxlen=history_window)
        self._last_score = 0
        self._last_calc_ms = 0

    def calculate(
        self,
        sector_flow: Optional[dict] = None,
        orderbook_data: Optional[dict] = None,
        absorption_data: Optional[dict] = None,
        derivatives_data: Optional[dict] = None,
        onchain_data: Optional[dict] = None,
        cvd: Optional[float] = None,
    ) -> dict:
        """
        Calcula o Whale Accumulation Score.
        
        Args:
            sector_flow: Fluxo por setor.
                Espera: {
                    "whale": {"buy": x, "sell": x, "delta": x},  # se disponível
                    "mid": {"buy": x, "sell": x, "delta": x},
                    "retail": {"buy": x, "sell": x, "delta": x},
                }
            orderbook_data: Dados do order book.
                Espera: {"bid_depth_usd": x, "ask_depth_usd": x, "imbalance": x}
            absorption_data: Dados de absorção atual.
                Espera: {
                    "index": x, "classification": "...",
                    "buyer_strength": x, "seller_exhaustion": x,
                    "continuation_probability": x,
                }
                OU: {"current_absorption": {...}} (nested)
            derivatives_data: Dados de derivativos.
                Espera: {"BTCUSDT": {"long_short_ratio": x, "open_interest": x, "open_interest_usd": x}}
            onchain_data: Dados on-chain (se disponível).
                Espera: {"exchange_netflow": x, "whale_transactions": x, "funding_rates": {...}}
            cvd: Cumulative Volume Delta acumulado.
            
        Returns:
            Dict com score (-100 a +100), classificação e componentes.
        """
        components = {}
        score: float = 0.0

        # ═══════════════════════════════════════════
        # 1. WHALE / MID FLOW DIRECTION (-30 a +30)
        # ═══════════════════════════════════════════
        flow_score: float = 0.0
        flow_detail: dict = {}

        if sector_flow and isinstance(sector_flow, dict):
            # Priorizar whale, fallback para mid
            whale_data = sector_flow.get("whale", {})
            mid_data = sector_flow.get("mid", {})
            retail_data = sector_flow.get("retail", {})

            # Delta do whale/mid (quem move o mercado)
            whale_delta: float = 0.0
            if isinstance(whale_data, dict):
                whale_delta = float(whale_data.get("delta", 0))

            mid_delta: float = 0.0
            if isinstance(mid_data, dict):
                mid_delta = float(mid_data.get("delta", 0))

            retail_delta: float = 0.0
            if isinstance(retail_data, dict):
                retail_delta = float(retail_data.get("delta", 0))

            # Usar whale se disponível, senão mid
            primary_delta = whale_delta if whale_delta != 0 else mid_delta
            
            # Normalizar: clamp entre -30 e +30
            # Delta é em BTC, escalar por fator
            flow_score = max(-30, min(30, primary_delta * 10))

            # Divergência smart money vs retail (sinal forte)
            # Se whales compram e retail vende = acumulação silenciosa
            smart_delta = whale_delta + mid_delta
            if smart_delta > 0 and retail_delta < 0:
                flow_score = min(30, flow_score + 10)  # Bonus: smart money buying while retail sells
                flow_detail["divergence"] = "smart_accumulation"
            elif smart_delta < 0 and retail_delta > 0:
                flow_score = max(-30, flow_score - 10)  # Smart money distributing
                flow_detail["divergence"] = "smart_distribution"
            else:
                flow_detail["divergence"] = "aligned"

            flow_detail["whale_delta"] = round(whale_delta, 4)
            flow_detail["mid_delta"] = round(mid_delta, 4)
            flow_detail["retail_delta"] = round(retail_delta, 4)
            flow_detail["primary_delta"] = round(primary_delta, 4)

        # CVD como fallback/complemento
        if cvd is not None and flow_score == 0:
            flow_score = max(-15, min(15, cvd * 5))
            flow_detail["cvd_used"] = True

        components["flow"] = {
            "score": round(flow_score, 2),
            "max": 30,
            "detail": flow_detail,
        }
        score += flow_score

        # ═══════════════════════════════════════════
        # 2. ORDER BOOK DEPTH ASYMMETRY (-20 a +20)
        # ═══════════════════════════════════════════
        depth_score: float = 0.0
        depth_detail = {}

        if orderbook_data and isinstance(orderbook_data, dict):
            bid_depth = float(orderbook_data.get("bid_depth_usd", 0))
            ask_depth = float(orderbook_data.get("ask_depth_usd", 0))
            ob_imbalance = orderbook_data.get("imbalance", None)

            total_depth = bid_depth + ask_depth
            if total_depth > 0:
                depth_ratio = (bid_depth - ask_depth) / total_depth
                depth_score = depth_ratio * 20  # -20 a +20

                depth_detail["bid_depth"] = round(bid_depth, 2)
                depth_detail["ask_depth"] = round(ask_depth, 2)
                depth_detail["ratio"] = round(depth_ratio, 4)

                # Depth metrics mais detalhados
                depth_metrics = orderbook_data.get("depth_metrics", {})
                if isinstance(depth_metrics, dict):
                    deep_imb = depth_metrics.get("depth_imbalance", 0)
                    if isinstance(deep_imb, (int, float)):
                        # Confirmar com depth mais profundo
                        if (depth_ratio > 0 and deep_imb > 0) or (depth_ratio < 0 and deep_imb < 0):
                            depth_score *= 1.2  # Confirmação = boost
                            depth_detail["deep_confirmation"] = True
                        else:
                            depth_detail["deep_confirmation"] = False

            depth_score = max(-20, min(20, depth_score))

        components["depth"] = {
            "score": round(depth_score, 2),
            "max": 20,
            "detail": depth_detail,
        }
        score += depth_score

        # ═══════════════════════════════════════════
        # 3. ABSORPTION PATTERN BIAS (-25 a +25)
        # ═══════════════════════════════════════════
        abs_score: float = 0.0
        abs_detail: dict = {}

        if absorption_data and isinstance(absorption_data, dict):
            # Suportar formato nested ou flat
            abs_inner = absorption_data.get("current_absorption", absorption_data)
            
            if isinstance(abs_inner, dict):
                buyer_str = float(abs_inner.get("buyer_strength", 0))
                seller_exh = float(abs_inner.get("seller_exhaustion", 0))
                abs_index = float(abs_inner.get("index", 0))
                classification = str(abs_inner.get("classification", ""))
                label = str(abs_inner.get("label", ""))

                # buyer_strength alto = compradores fortes = acumulação
                # seller_exhaustion alto = vendedores cansados = acumulação
                net_absorption = buyer_str - seller_exh

                # Escalar: valores típicos são 0-10
                if net_absorption > 0:
                    abs_score = min(25, net_absorption * 3)
                else:
                    abs_score = max(-25, net_absorption * 3)

                # Boost se absorção é STRONG
                if "STRONG" in classification.upper():
                    if "COMPRA" in label.upper() or "BUY" in label.upper():
                        abs_score = min(25, abs_score + 8)
                    elif "VENDA" in label.upper() or "SELL" in label.upper():
                        abs_score = max(-25, abs_score - 8)

                abs_detail["buyer_strength"] = buyer_str
                abs_detail["seller_exhaustion"] = seller_exh
                abs_detail["net_absorption"] = round(net_absorption, 2)
                abs_detail["index"] = abs_index
                abs_detail["label"] = label

        components["absorption"] = {
            "score": round(abs_score, 2),
            "max": 25,
            "detail": abs_detail,
        }
        score += abs_score

        # ═══════════════════════════════════════════
        # 4. DERIVATIVES CONTEXT (-25 a +25)
        # ═══════════════════════════════════════════
        deriv_score: float = 0.0
        deriv_detail: dict = {}

        if derivatives_data and isinstance(derivatives_data, dict):
            # Buscar dados de BTCUSDT
            btc_deriv = derivatives_data.get("BTCUSDT", derivatives_data)

            if isinstance(btc_deriv, dict):
                lsr = float(btc_deriv.get("long_short_ratio", 1.0))
                oi = float(btc_deriv.get("open_interest", 0))
                oi_usd = float(btc_deriv.get("open_interest_usd", 0))

                # Long/Short Ratio
                # LSR > 2 = muito mais longs = posicionamento bullish
                # LSR < 0.5 = muito mais shorts = posicionamento bearish
                # Mas cuidado: LSR extremo pode indicar crowded trade
                if lsr > 1:
                    # Normalizar: LSR 1→0pts, LSR 2→15pts, LSR 3→20pts
                    lsr_score = min(20, (lsr - 1) * 15)
                else:
                    # LSR 1→0pts, LSR 0.5→-15pts, LSR 0.3→-20pts
                    lsr_score = max(-20, (lsr - 1) * 20)

                deriv_score += lsr_score
                deriv_detail["long_short_ratio"] = lsr
                deriv_detail["lsr_score"] = round(lsr_score, 2)

                # Funding rates (se disponível via onchain)
                if onchain_data and isinstance(onchain_data, dict):
                    funding = onchain_data.get("funding_rates", {})
                    if isinstance(funding, dict) and funding:
                        avg_funding = sum(float(v) for v in funding.values()) / len(funding)
                        # Funding positivo = longs pagam shorts = bullish positioning
                        funding_score = max(-5, min(5, avg_funding * 10000))
                        deriv_score += funding_score
                        deriv_detail["avg_funding"] = round(avg_funding, 6)
                        deriv_detail["funding_score"] = round(funding_score, 2)

                deriv_score = max(-25, min(25, deriv_score))

        # On-chain exchange netflow como bônus
        if onchain_data and isinstance(onchain_data, dict):
            netflow = onchain_data.get("exchange_netflow", 0)
            if isinstance(netflow, (int, float)) and netflow != 0:
                # Netflow negativo = saída de exchanges = acumulação
                # Netflow positivo = entrada em exchanges = distribuição
                netflow_bonus = max(-5, min(5, -netflow * 0.02))
                deriv_score = max(-25, min(25, deriv_score + netflow_bonus))
                deriv_detail["exchange_netflow"] = netflow
                deriv_detail["netflow_signal"] = "accumulation" if netflow < 0 else "distribution"

        components["derivatives"] = {
            "score": round(deriv_score, 2),
            "max": 25,
            "detail": deriv_detail,
        }
        score += deriv_score

        # ═══════════════════════════════════════════
        # SCORE FINAL E CLASSIFICAÇÃO
        # ═══════════════════════════════════════════
        score = max(-100, min(100, round(score)))

        if score >= 50:
            classification = "STRONG_ACCUMULATION"
        elif score >= 20:
            classification = "MILD_ACCUMULATION"
        elif score >= -19:
            classification = "NEUTRAL"
        elif score >= -49:
            classification = "MILD_DISTRIBUTION"
        else:
            classification = "STRONG_DISTRIBUTION"

        # Bias simplificado
        if score > 10:
            bias = "ACCUMULATING"
        elif score < -10:
            bias = "DISTRIBUTING"
        else:
            bias = "NEUTRAL"

        # Registrar no histórico
        now_ms = int(time.time() * 1000)
        self._history.append({"score": score, "ts": now_ms})
        self._last_score = score
        self._last_calc_ms = now_ms

        # Tendência (comparar com histórico)
        trend = self._calculate_trend()

        return {
            "score": score,
            "classification": classification,
            "bias": bias,
            "components": components,
            "trend": trend,
            "status": "success",
        }

    def _calculate_trend(self) -> dict:
        """Calcula tendência do score ao longo do tempo."""
        if len(self._history) < 3:
            return {
                "direction": "insufficient_data",
                "avg_score": self._last_score,
                "samples": len(self._history),
            }

        scores = [h["score"] for h in self._history]
        avg_score = sum(scores) / len(scores)
        recent_avg = sum(scores[-5:]) / min(5, len(scores))

        # Tendência
        if recent_avg > avg_score + 5:
            direction = "increasing_accumulation"
        elif recent_avg < avg_score - 5:
            direction = "increasing_distribution"
        else:
            direction = "stable"

        # Momentum: diferença entre último e média
        momentum = self._last_score - avg_score

        return {
            "direction": direction,
            "avg_score": round(avg_score, 1),
            "recent_avg": round(recent_avg, 1),
            "momentum": round(momentum, 1),
            "samples": len(self._history),
            "score_range": {"min": min(scores), "max": max(scores)},
        }

    def get_last_score(self) -> int:
        """Retorna último score calculado."""
        return self._last_score

    def get_history_summary(self) -> dict:
        """Retorna resumo do histórico de scores."""
        if not self._history:
            return {"status": "empty", "samples": 0}

        scores = [h["score"] for h in self._history]
        return {
            "status": "ok",
            "samples": len(scores),
            "current": scores[-1],
            "avg": round(sum(scores) / len(scores), 1),
            "min": min(scores),
            "max": max(scores),
            "std": round(
                (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)) ** 0.5, 1
            ) if len(scores) > 1 else 0,
            "trend": self._calculate_trend()["direction"],
        }

    def reset(self) -> None:
        """Limpa histórico."""
        self._history.clear()
        self._last_score = 0
```


## 5.7 support_resistance/sr_strength.py (completo)


```python
"""
S/R Strength Scoring — Pontuação de força de Suportes e Resistências.

Pontua cada nível de S/R de 0-100 baseado em:
  1. Toques históricos (quantas vezes o preço testou o nível)      → 0-25 pts
  2. Volume acumulado no nível (do Volume Profile)                  → 0-25 pts
  3. Confluência com outros indicadores (pivots, EMA, round numbers)→ 0-30 pts
  4. Recência (mais recente = mais forte)                          → 0-20 pts

Uso:
    scorer = SRStrengthScorer()
    levels = scorer.score_levels(
        current_price=64892,
        vp_data={"poc": 64888, "vah": 66055, "val": 64683, "hvns": [...], "lvns": [...]},
        pivot_data={"classic": {"PP": 64850, "R1": 65100, ...}},
        ema_values={"ema_21_1h": 65418, "ema_21_4h": 66695},
        recent_candles=df_candles,
    )
    print(levels)  # Lista de níveis com strength score
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SRStrengthScorer:
    """
    Calcula força de níveis de suporte e resistência.
    
    Combina múltiplas fontes de dados para produzir um score
    único por nível. Usado para priorizar quais S/R são mais
    confiáveis para stops, entries e targets.
    """

    def __init__(self, touch_tolerance_pct: float = 0.15, round_number_interval: int = 1000):
        """
        Args:
            touch_tolerance_pct: % de tolerância para considerar um "toque" no nível.
                                 0.15 = preço a 0.15% do nível conta como toque.
            round_number_interval: Intervalo para números redondos (1000 = 64000, 65000...).
        """
        self._touch_tolerance_pct = touch_tolerance_pct
        self._round_number_interval = round_number_interval

    def score_levels(
        self,
        current_price: float,
        vp_data: Optional[dict] = None,
        pivot_data: Optional[dict] = None,
        ema_values: Optional[dict] = None,
        recent_candles=None,
        weekly_vp: Optional[dict] = None,
        monthly_vp: Optional[dict] = None,
    ) -> dict:
        """
        Pontua todos os níveis de S/R identificáveis.
        
        Args:
            current_price: Preço atual do ativo.
            vp_data: Volume Profile diário.
                     Espera: {"poc": float, "vah": float, "val": float,
                              "hvns": [float], "lvns": [float]}
            pivot_data: Pivot Points calculados.
                        Espera: {"classic": {"PP": x, "R1": x, ...}, "fibonacci": {...}, ...}
            ema_values: EMAs de diferentes timeframes.
                        Espera: {"ema_21_15m": x, "ema_21_1h": x, "ema_21_4h": x, "ema_21_1d": x}
            recent_candles: DataFrame ou lista de candles recentes para contagem de toques.
                           Espera colunas/chaves: high, low, close
            weekly_vp: Volume Profile semanal (para confluência multi-TF).
                       Espera: {"poc": float, "vah": float, "val": float}
            monthly_vp: Volume Profile mensal.
                        Espera: {"poc": float, "vah": float, "val": float}
                        
        Returns:
            Dict com:
              - levels: Lista de níveis pontuados [{price, type, source, strength, ...}]
                        (cada item tem "type": "support" ou "resistance")
              - total_levels_found: Total de níveis encontrados
              - status: "success", "no_candidates" ou "invalid_price"
        """
        if current_price <= 0:
            return {"levels": [], "status": "invalid_price"}

        # 1. Coletar todos os candidatos a S/R
        candidates = self._collect_candidates(
            current_price, vp_data, pivot_data, ema_values, weekly_vp, monthly_vp
        )

        if not candidates:
            return {"levels": [], "status": "no_candidates"}

        # 2. Mesclar candidatos próximos (evitar duplicatas)
        merged = self._merge_nearby_levels(candidates, current_price)

        # 3. Pontuar cada nível
        scored = []
        for level in merged:
            score = self._calculate_score(
                level, current_price, recent_candles
            )
            level["strength"] = score
            scored.append(level)

        # 4. Ordenar por strength
        scored.sort(key=lambda x: x["strength"], reverse=True)

        # levels[] já contém campo "type" ("support"/"resistance") em cada item.
        # Não duplicar em arrays separados — consumidores podem filtrar por type.
        return {
            "levels": scored[:20],  # Top 20 (cada item tem "type": "support" ou "resistance")
            "total_levels_found": len(scored),
            "status": "success",
        }

    def _collect_candidates(
        self, current_price, vp_data, pivot_data, ema_values, weekly_vp, monthly_vp
    ) -> list:
        """Coleta todos os candidatos a S/R de múltiplas fontes."""
        candidates = []
        has_other_sources = False

        # --- Volume Profile Diário ---
        if vp_data and isinstance(vp_data, dict):
            has_other_sources = True
            poc = vp_data.get("poc", 0) or vp_data.get("poc_price", 0)
            vah = vp_data.get("vah", 0)
            val = vp_data.get("val", 0)

            if poc > 0:
                candidates.append({"price": poc, "source": "poc_daily", "source_weight": 1.5})
            if vah > 0:
                candidates.append({"price": vah, "source": "vah_daily", "source_weight": 1.2})
            if val > 0:
                candidates.append({"price": val, "source": "val_daily", "source_weight": 1.2})

            for hvn in (vp_data.get("hvns", []) or []):
                if hvn and hvn > 0:
                    candidates.append({"price": hvn, "source": "hvn_daily", "source_weight": 0.8})

        # --- Volume Profile Semanal ---
        if weekly_vp and isinstance(weekly_vp, dict):
            has_other_sources = True
            for key, src in [("poc", "poc_weekly"), ("vah", "vah_weekly"), ("val", "val_weekly")]:
                val_w = weekly_vp.get(key, 0)
                if val_w and val_w > 0:
                    candidates.append({"price": val_w, "source": src, "source_weight": 1.8})

        # --- Volume Profile Mensal ---
        if monthly_vp and isinstance(monthly_vp, dict):
            has_other_sources = True
            for key, src in [("poc", "poc_monthly"), ("vah", "vah_monthly"), ("val", "val_monthly")]:
                val_m = monthly_vp.get(key, 0)
                if val_m and val_m > 0:
                    candidates.append({"price": val_m, "source": src, "source_weight": 2.0})

        # --- Pivot Points ---
        if pivot_data and isinstance(pivot_data, dict):
            has_other_sources = True
            for method_name, method_levels in pivot_data.items():
                if not isinstance(method_levels, dict):
                    continue
                weight = 1.3 if method_name == "classic" else 1.0
                for level_name, level_price in method_levels.items():
                    if isinstance(level_price, (int, float)) and level_price > 0:
                        candidates.append({
                            "price": level_price,
                            "source": f"pivot_{method_name}_{level_name}",
                            "source_weight": weight,
                        })

        # --- EMAs ---
        if ema_values and isinstance(ema_values, dict):
            has_other_sources = True
            ema_weights = {
                "ema_21_15m": 0.5,
                "ema_21_1h": 0.8,
                "ema_21_4h": 1.2,
                "ema_21_1d": 1.5,
                "mme_21": 1.0,  # nome alternativo
            }
            for ema_name, ema_price in ema_values.items():
                if isinstance(ema_price, (int, float)) and ema_price > 0:
                    weight = ema_weights.get(ema_name, 0.8)
                    candidates.append({
                        "price": ema_price,
                        "source": ema_name,
                        "source_weight": weight,
                    })

        # --- Números Redondos ---
        interval = self._round_number_interval
        if current_price > 0 and interval > 0 and has_other_sources:
            # 3 acima e 3 abaixo
            base = int(current_price / interval) * interval
            for offset in range(-3, 4):
                round_price = base + (offset * interval)
                if round_price > 0:
                    candidates.append({
                        "price": round_price,
                        "source": "round_number",
                        "source_weight": 0.6,
                    })

        return candidates

    def _merge_nearby_levels(self, candidates: list, current_price: float) -> list:
        """
        Mescla candidatos que estão muito próximos (dentro de tolerance_pct).
        Mantém o com maior source_weight e acumula confluences.
        """
        if not candidates:
            return []

        tolerance = current_price * (self._touch_tolerance_pct / 100)
        candidates_sorted = sorted(candidates, key=lambda c: c["price"])

        merged = []
        used = set()

        for i, candidate in enumerate(candidates_sorted):
            if i in used:
                continue

            group = [candidate]
            used.add(i)

            for j in range(i + 1, len(candidates_sorted)):
                if j in used:
                    continue
                if abs(candidates_sorted[j]["price"] - candidate["price"]) <= tolerance:
                    group.append(candidates_sorted[j])
                    used.add(j)
                else:
                    break  # Sorted, so no more nearby

            # Mesclar grupo
            best = max(group, key=lambda g: g["source_weight"])
            confluences = list(set(g["source"] for g in group))
            avg_price = sum(g["price"] for g in group) / len(group)

            merged.append({
                "price": round(avg_price, 2),
                "primary_source": best["source"],
                "confluences": confluences,
                "confluence_count": len(confluences),
                "max_source_weight": best["source_weight"],
                "sum_source_weight": sum(g["source_weight"] for g in group),
            })

        return merged

    def _calculate_score(self, level: dict, current_price: float, recent_candles=None) -> int:
        """
        Calcula score final de 0-100 para um nível.
        
        Componentes:
          1. Toques históricos  → 0-25 pontos
          2. Peso da fonte      → 0-25 pontos
          3. Confluência        → 0-30 pontos
          4. Proximidade        → 0-20 pontos
        """
        score = 0
        level_price = level["price"]

        # 1. TOQUES HISTÓRICOS (0-25 pontos)
        if recent_candles is not None:
            touches = self._count_touches(level_price, recent_candles)
            score += min(25, touches * 6)  # 4+ toques = 24-25 pts
            level["touches"] = touches
        else:
            # Sem dados de candles, dar score parcial baseado em confluência
            score += 8
            level["touches"] = None

        # 2. PESO DA FONTE (0-25 pontos)
        # source_weight vai de 0.5 (fraco) a 2.0 (forte)
        source_weight = level.get("sum_source_weight", level.get("max_source_weight", 1.0))
        weight_score = min(25, source_weight * 8)
        score += weight_score

        # 3. CONFLUÊNCIA (0-30 pontos)
        confluence_count = level.get("confluence_count", 1)
        # 1 confluência = 5pts, 2 = 12pts, 3 = 20pts, 4+ = 25-30pts
        if confluence_count >= 5:
            conf_score = 30
        elif confluence_count >= 4:
            conf_score = 25
        elif confluence_count >= 3:
            conf_score = 20
        elif confluence_count >= 2:
            conf_score = 12
        else:
            conf_score = 5
        score += conf_score

        # 4. PROXIMIDADE ao preço atual (0-20 pontos)
        # Mais próximo = mais relevante imediatamente
        if current_price > 0 and level_price > 0:
            distance_pct = abs(level_price - current_price) / current_price * 100
            proximity_score = max(0, 20 - (distance_pct * 3))
            score += proximity_score
            level["distance_pct"] = round(distance_pct, 4)
        else:
            level["distance_pct"] = None

        # Tipo: suporte ou resistência
        if level_price < current_price:
            level["type"] = "support"
        elif level_price > current_price:
            level["type"] = "resistance"
        else:
            level["type"] = "at_price"

        return min(round(score), 100)

    def _count_touches(self, level_price: float, candles, lookback: int = 100) -> int:
        """
        Conta quantas vezes o preço tocou um nível nos últimos N candles.
        
        Um "toque" = o high ou low do candle está dentro de tolerance_pct do nível.
        """
        tolerance = level_price * (self._touch_tolerance_pct / 100)
        touches = 0

        try:
            # Se é DataFrame pandas
            if hasattr(candles, 'iterrows'):
                data = candles.tail(lookback)
                for _, row in data.iterrows():
                    high = float(row.get("high", row.get("h", 0)))
                    low = float(row.get("low", row.get("l", 0)))
                    if high > 0 and low > 0:
                        if abs(high - level_price) <= tolerance or abs(low - level_price) <= tolerance:
                            touches += 1
                        elif low <= level_price <= high:
                            touches += 1

            # Se é lista de dicts
            elif isinstance(candles, list):
                data = candles[-lookback:]
                for candle in data:
                    if isinstance(candle, dict):
                        high = float(candle.get("high", candle.get("h", 0)))
                        low = float(candle.get("low", candle.get("l", 0)))
                        if high > 0 and low > 0:
                            if abs(high - level_price) <= tolerance or abs(low - level_price) <= tolerance:
                                touches += 1
                            elif low <= level_price <= high:
                                touches += 1

        except Exception as e:
            logger.debug(f"Touch counting error: {e}")

        return touches

    def quick_score(
        self,
        level_price: float,
        current_price: float,
        source: str = "unknown",
        confluence_count: int = 1,
    ) -> int:
        """
        Pontuação rápida sem dados históricos completos.
        Útil para scoring em tempo real de níveis individuais.
        """
        score = 0

        # Peso base por tipo de fonte
        source_weights = {
            "poc_daily": 20, "poc_weekly": 22, "poc_monthly": 25,
            "vah_daily": 15, "val_daily": 15,
            "vah_weekly": 18, "val_weekly": 18,
            "pivot_classic_PP": 18, "pivot_classic_R1": 14, "pivot_classic_S1": 14,
            "ema_21_1d": 16, "ema_21_4h": 14,
            "round_number": 10,
            "hvn_daily": 12,
        }
        score += source_weights.get(source, 10)

        # Confluência
        score += min(30, confluence_count * 8)

        # Proximidade
        if current_price > 0:
            dist_pct = abs(level_price - current_price) / current_price * 100
            score += max(0, int(20 - dist_pct * 3))

        return min(score, 100)
```


## 5.8 flow_analyzer/absorption.py (completo)


```python
# flow_analyzer/absorption.py
"""
Lógica de absorção do FlowAnalyzer.

Absorção ocorre quando:
- Grande volume de um lado (ex: vendas)
- Preço não se move significativamente
- Indica que o outro lado (compradores) está absorvendo a pressão

Tipos:
- Absorção de Compra: Vendedores agressivos, compradores absorvem
- Absorção de Venda: Compradores agressivos, vendedores absorvem
- Neutra: Sem absorção significativa
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

from .constants import (
    DEFAULT_ABSORCAO_DELTA_EPS,
    DEFAULT_ABSORCAO_ATR_MULTIPLIER,
    DEFAULT_ABSORCAO_VOL_MULTIPLIER,
    DEFAULT_ABSORCAO_MIN_PCT_TOLERANCE,
    DEFAULT_ABSORCAO_MAX_PCT_TOLERANCE,
    DEFAULT_ABSORCAO_FALLBACK_PCT_TOLERANCE,
    ABSORPTION_INTENSITY_THRESHOLD,
    ABSORPTION_IMBALANCE_THRESHOLD,
)
from .validation import validate_ohlc, guard_absorcao
from .utils import lazy_log, decimal_round, clamp


# ==============================================================================
# ABSORPTION CLASSIFIER
# ==============================================================================

@dataclass
class AbsorptionConfig:
    """Configuração para classificação de absorção."""
    
    eps: float = DEFAULT_ABSORCAO_DELTA_EPS
    atr_multiplier: float = DEFAULT_ABSORCAO_ATR_MULTIPLIER
    vol_multiplier: float = DEFAULT_ABSORCAO_VOL_MULTIPLIER
    min_pct_tolerance: float = DEFAULT_ABSORCAO_MIN_PCT_TOLERANCE
    max_pct_tolerance: float = DEFAULT_ABSORCAO_MAX_PCT_TOLERANCE
    fallback_pct_tolerance: float = DEFAULT_ABSORCAO_FALLBACK_PCT_TOLERANCE
    intensity_threshold: float = ABSORPTION_INTENSITY_THRESHOLD
    imbalance_threshold: float = ABSORPTION_IMBALANCE_THRESHOLD


class AbsorptionClassifier:
    """
    Classificador de absorção com contexto de volatilidade.
    
    Usa OHLC e delta para determinar se há absorção e de que tipo.
    Suporta tolerância dinâmica baseada em ATR ou volatilidade.
    
    Example:
        >>> classifier = AbsorptionClassifier()
        >>> classifier.update_volatility(atr=500.0)
        >>> label = classifier.classify(
        ...     delta_btc=-10.0,
        ...     open_p=50000, high_p=50100, low_p=49900, close_p=50050
        ... )
        >>> print(label)  # "Absorção de Compra"
    """
    
    def __init__(self, config: Optional[AbsorptionConfig] = None):
        self.config = config or AbsorptionConfig()
        self._atr_price: Optional[float] = None
        self._price_volatility: Optional[float] = None
    
    def update_volatility(
        self,
        atr: Optional[float] = None,
        price_volatility: Optional[float] = None
    ) -> None:
        """
        Atualiza contexto de volatilidade.
        
        Args:
            atr: Average True Range
            price_volatility: Volatilidade de preço (desvio padrão)
        """
        if isinstance(atr, (int, float)) and atr > 0:
            self._atr_price = float(atr)
        if isinstance(price_volatility, (int, float)) and price_volatility > 0:
            self._price_volatility = float(price_volatility)
    
    def _calculate_tolerance(self, base_price: float) -> float:
        """
        Calcula tolerância de preço dinâmica.
        
        Prioridade:
        1. ATR-based
        2. Volatility-based
        3. Fallback fixo
        
        Args:
            base_price: Preço base para cálculo percentual
            
        Returns:
            Tolerância como percentual do preço
        """
        if base_price <= 0:
            return self.config.fallback_pct_tolerance
        
        pct_tolerance: Optional[float] = None
        
        # ATR-based
        if self._atr_price is not None and self._atr_price > 0:
            pct_tolerance = (self.config.atr_multiplier * self._atr_price) / base_price
        
        # Volatility-based fallback
        if pct_tolerance is None and self._price_volatility is not None:
            if self._price_volatility > 0:
                pct_tolerance = (
                    self.config.vol_multiplier * self._price_volatility
                ) / base_price
        
        # Fallback fixo
        if pct_tolerance is None:
            pct_tolerance = self.config.fallback_pct_tolerance
        
        # Clamp
        return clamp(
            pct_tolerance,
            self.config.min_pct_tolerance,
            self.config.max_pct_tolerance
        )
    
    def classify(
        self,
        delta_btc: float,
        open_p: float,
        high_p: float,
        low_p: float,
        close_p: float,
        eps: Optional[float] = None,
    ) -> str:
        """
        Classifica absorção baseado em delta e OHLC.
        
        Lógica:
        - Delta negativo + preço não caiu muito = Absorção de Compra
          (compradores absorveram pressão vendedora)
        - Delta positivo + preço não subiu muito = Absorção de Venda
          (vendedores absorveram pressão compradora)
        
        Args:
            delta_btc: Delta de volume (compras - vendas)
            open_p: Preço de abertura
            high_p: Preço máximo
            low_p: Preço mínimo
            close_p: Preço de fechamento
            eps: Epsilon para delta neutro (usa config se None)
            
        Returns:
            "Absorção de Compra", "Absorção de Venda", ou "Neutra"
        """
        if eps is None:
            eps = self.config.eps
        
        try:
            # Validação básica
            if not all(isinstance(x, (int, float)) for x in 
                      [delta_btc, open_p, high_p, low_p, close_p, eps]):
                return "Neutra"
            
            if not validate_ohlc(open_p, high_p, low_p, close_p):
                return "Neutra"
            
            # Range do candle
            candle_range = high_p - low_p
            if candle_range <= 0:
                candle_range = 0.0001  # Evita divisão por zero
            
            # Posição do fechamento no range
            # close_pos_compra: quanto mais alto, mais força compradora
            close_pos_compra = (close_p - low_p) / candle_range
            # close_pos_venda: quanto mais baixo, mais força vendedora
            close_pos_venda = (high_p - close_p) / candle_range
            
            # Tolerância dinâmica
            base_price = close_p if close_p > 0 else open_p
            pct_tolerance = self._calculate_tolerance(base_price)
            
            # Bounds
            lower_bound = open_p * (1.0 - pct_tolerance)
            upper_bound = open_p * (1.0 + pct_tolerance)
            
            # Classificação
            # Absorção de Compra:
            # - Delta negativo (mais vendas)
            # - Mas preço não caiu (fechou acima do lower bound)
            # - Fechamento na metade superior do candle
            if (delta_btc < -abs(eps) and 
                close_p >= lower_bound and 
                close_pos_compra > 0.5):
                return "Absorção de Compra"
            
            # Absorção de Venda:
            # - Delta positivo (mais compras)
            # - Mas preço não subiu (fechou abaixo do upper bound)
            # - Fechamento na metade inferior do candle
            if (delta_btc > abs(eps) and 
                close_p <= upper_bound and 
                close_pos_venda > 0.5):
                return "Absorção de Venda"
            
            return "Neutra"
            
        except Exception as e:
            if lazy_log.should_log("absorption_classify_error"):
                logging.warning(f"Erro em classify: {e}")
            return "Neutra"
    
    def classify_simple(self, delta: float, eps: Optional[float] = None) -> str:
        """
        Classificador simples de absorção apenas por delta.
        
        Não considera OHLC, útil para análise rápida.
        
        Args:
            delta: Delta de volume
            eps: Epsilon para delta neutro
            
        Returns:
            "Absorção de Compra", "Absorção de Venda", ou "Neutra"
        """
        if eps is None:
            eps = self.config.eps
        
        try:
            d = float(delta)
        except (TypeError, ValueError):
            return "Neutra"
        
        if d < -eps:
            return "Absorção de Compra"
        if d > eps:
            return "Absorção de Venda"
        return "Neutra"
    
    @staticmethod
    def map_aggression_to_label(aggression_side: str) -> str:
        """
        Mapeia lado de agressão para rótulo de absorção.
        
        Args:
            aggression_side: "buy" ou "sell"
            
        Returns:
            Rótulo de absorção correspondente
        """
        side = (aggression_side or "").strip().lower()
        if side == "buy":
            return "Absorção de Compra"
        if side == "sell":
            return "Absorção de Venda"
        return "Absorção"


# ==============================================================================
# ABSORPTION ANALYSIS
# ==============================================================================

@dataclass
class AbsorptionAnalysis:
    """Resultado de análise de absorção."""
    
    index: float  # 0.0 a 1.0
    classification: str  # NONE, WEAK, MODERATE, STRONG
    label: str  # Absorção de Compra/Venda/Neutra
    buyer_strength: float  # 0-10
    seller_strength: float  # 0-10
    seller_exhaustion: float  # 0-10
    continuation_probability: float  # 0.0 a 1.0
    
    # Dados de suporte
    delta_usd: float
    total_volume_usd: float
    flow_imbalance: float
    window_min: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'index': self.index,
            'classification': self.classification,
            'label': self.label,
            'buyer_strength': self.buyer_strength,
            'seller_exhaustion': self.seller_exhaustion,
            'continuation_probability': self.continuation_probability,
            'delta_usd': self.delta_usd,
            'total_volume_usd': self.total_volume_usd,
            'flow_imbalance': self.flow_imbalance,
            'window_min': self.window_min,
        }


class AbsorptionAnalyzer:
    """
    Analisador avançado de absorção.
    
    Calcula índices de absorção, força de compradores/vendedores,
    e probabilidade de continuação.
    """
    
    def __init__(self, config: Optional[AbsorptionConfig] = None):
        self.config = config or AbsorptionConfig()
    
    def analyze(
        self,
        delta_usd: float,
        total_volume_usd: float,
        flow_imbalance: float,
        buy_pct: float,
        sell_pct: float,
        absorption_label: str,
        window_min: int,
    ) -> Optional[AbsorptionAnalysis]:
        """
        Analisa absorção a partir de métricas de flow.
        
        Args:
            delta_usd: Net flow em USD
            total_volume_usd: Volume total em USD
            flow_imbalance: Imbalance (-1 a 1)
            buy_pct: Percentual de compras
            sell_pct: Percentual de vendas
            absorption_label: Rótulo de absorção
            window_min: Janela de tempo em minutos
            
        Returns:
            AbsorptionAnalysis ou None se dados insuficientes
        """
        if total_volume_usd <= 0:
            return None
        
        try:
            # Índice de absorção: combinação de delta relativo e imbalance
            rel_delta = min(1.0, abs(delta_usd) / total_volume_usd)
            abs_flow = min(1.0, abs(flow_imbalance))
            absorption_index = decimal_round(rel_delta * abs_flow, decimals=4)
            
            # Classificação
            if absorption_index >= 0.7:
                classification = "STRONG_ABSORPTION"
            elif absorption_index >= 0.4:
                classification = "MODERATE_ABSORPTION"
            elif absorption_index > 0.1:
                classification = "WEAK_ABSORPTION"
            else:
                classification = "NONE"
            
            # Força de compradores/vendedores
            total_pct = buy_pct + sell_pct
            if total_pct > 0:
                buy_intensity = buy_pct / total_pct
            else:
                buy_intensity = 0.5
            
            buyer_strength = decimal_round(buy_intensity * 10, decimals=1)
            seller_strength = decimal_round((1 - buy_intensity) * 10, decimals=1)
            
            # Seller exhaustion baseado no tipo de absorção
            if "Compra" in absorption_label:
                # Absorção de compra = compradores absorvendo vendas
                seller_exhaustion = buyer_strength
            elif "Venda" in absorption_label:
                # Absorção de venda = vendedores absorvendo compras
                seller_exhaustion = seller_strength
            else:
                seller_exhaustion = decimal_round(abs_flow * 10, decimals=1)
            
            # Probabilidade de continuação
            continuation_probability = decimal_round(absorption_index * 0.9, decimals=2)
            
            return AbsorptionAnalysis(
                index=absorption_index,
                classification=classification,
                label=absorption_label,
                buyer_strength=buyer_strength,
                seller_strength=seller_strength,
                seller_exhaustion=seller_exhaustion,
                continuation_probability=continuation_probability,
                delta_usd=delta_usd,
                total_volume_usd=total_volume_usd,
                flow_imbalance=flow_imbalance,
                window_min=window_min,
            )
            
        except Exception as e:
            if lazy_log.should_log("absorption_analyze_error"):
                logging.debug(f"Erro em analyze: {e}")
            return None
    
    def refine_label_with_intensity(
        self,
        base_label: str,
        delta_btc: float,
        total_btc: float,
        flow_imbalance: float,
        eps: float,
    ) -> str:
        """
        Refina rótulo de absorção com base em intensidade.
        
        Só mantém rótulo de absorção se intensidade e imbalance
        forem significativos.
        
        Args:
            base_label: Rótulo original
            delta_btc: Delta em BTC
            total_btc: Volume total em BTC
            flow_imbalance: Imbalance (-1 a 1)
            eps: Epsilon
            
        Returns:
            Rótulo refinado
        """
        try:
            if total_btc <= 0:
                return "Neutra"
            
            intensidade = abs(delta_btc) / total_btc
            
            # Precisa de intensidade E imbalance significativos
            if (intensidade >= self.config.intensity_threshold and 
                abs(flow_imbalance) >= self.config.imbalance_threshold):
                
                if delta_btc < -eps:
                    return "Absorção de Compra"
                elif delta_btc > eps:
                    return "Absorção de Venda"
            
            return "Neutra"
            
        except Exception:
             return base_label


class AbsorptionZoneMapper:
    """
    Mapeia zonas onde absorções significativas ocorreram.
    
    Mantém histórico de eventos de absorção com timestamp e preço,
    agrupa por zona de preço, e identifica zonas recorrentes.
    
    Zonas com múltiplas absorções são defesas fortes e confiáveis.
    
    Uso:
        mapper = AbsorptionZoneMapper()
        mapper.record_event(price=64800, classification="Absorção de Compra",
                           index=0.65, timestamp_ms=1771888200000)
        zones = mapper.get_zones(current_price=64892)
    """

    def __init__(
        self,
        zone_tolerance_pct: float = 0.15,
        max_history_hours: int = 24,
        min_index_threshold: float = 0.1,
    ):
        """
        Args:
            zone_tolerance_pct: % de tolerância para agrupar eventos na mesma zona.
            max_history_hours: Máximo de horas de histórico a manter.
            min_index_threshold: Índice mínimo de absorção para registrar.
        """
        self._tolerance_pct = zone_tolerance_pct
        self._max_history_ms = max_history_hours * 3600 * 1000
        self._min_index = min_index_threshold
        self._events: list = []

    def record_event(
        self,
        price: float,
        classification: str,
        index: float = 0,
        timestamp_ms: Optional[int] = None,
        buyer_strength: float = 0,
        seller_exhaustion: float = 0,
        volume_usd: float = 0,
    ) -> None:
        """
        Registra um evento de absorção.
        
        Args:
            price: Preço onde a absorção ocorreu.
            classification: Tipo ("Absorção de Compra", "Absorção de Venda",
                            "STRONG_ABSORPTION", etc.)
            index: Índice de absorção (0-1).
            timestamp_ms: Timestamp em ms.
            buyer_strength: Força do comprador (0-10).
            seller_exhaustion: Exaustão do vendedor (0-10).
            volume_usd: Volume em USD durante a absorção.
        """
        import time

        if price <= 0:
            return

        if index < self._min_index:
            # Absorção muito fraca — só registrar se tiver classificação real
            if "Neutra" in classification or "NONE" in classification.upper():
                return

        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        # Normalizar classificação
        classification_upper = classification.upper()
        if "COMPRA" in classification_upper or "BUY" in classification_upper:
            side = "buy"
        elif "VENDA" in classification_upper or "SELL" in classification_upper:
            side = "sell"
        else:
            side = "neutral"

        self._events.append({
            "price": price,
            "side": side,
            "classification": classification,
            "index": index,
            "timestamp_ms": timestamp_ms,
            "buyer_strength": buyer_strength,
            "seller_exhaustion": seller_exhaustion,
            "volume_usd": volume_usd,
        })

        # Cleanup de eventos antigos (lazy)
        if len(self._events) > 500:
            self._cleanup()

    def _cleanup(self) -> None:
        """Remove eventos fora da janela temporal."""
        import time
        now_ms = int(time.time() * 1000)
        cutoff = now_ms - self._max_history_ms
        self._events = [e for e in self._events if e["timestamp_ms"] >= cutoff]

    def get_zones(self, current_price: float = 0, top_n: int = 10) -> dict:
        """
        Retorna zonas de absorção mapeadas.
        
        Args:
            current_price: Preço atual para calcular distâncias.
            top_n: Número máximo de zonas a retornar.
            
        Returns:
            Dict com zonas agrupadas, lado dominante e métricas.
        """
        import time

        self._cleanup()

        if not self._events:
            return {
                "zones": [],
                "total_zones": 0,
                "total_events": 0,
                "buy_zone_count": 0,
                "sell_zone_count": 0,
                "status": "no_events",
            }

        # Agrupar eventos por zona de preço
        zones_map: dict = {}

        for event in self._events:
            # Encontrar zona existente ou criar nova
            assigned = False
            for zone_key, zone_data in zones_map.items():
                if abs(event["price"] - zone_data["center"]) / zone_data["center"] * 100 < self._tolerance_pct:
                    zone_data["events"].append(event)
                    # Atualizar center como média ponderada
                    total_events = len(zone_data["events"])
                    zone_data["center"] = sum(e["price"] for e in zone_data["events"]) / total_events
                    assigned = True
                    break

            if not assigned:
                zone_key = round(event["price"], 0)
                zones_map[zone_key] = {
                    "center": event["price"],
                    "events": [event],
                }

        # Construir resultado
        zones_result = []
        for zone_key, zone_data in zones_map.items():
            events = zone_data["events"]
            center = zone_data["center"]

            # Contagem por lado
            buy_events = [e for e in events if e["side"] == "buy"]
            sell_events = [e for e in events if e["side"] == "sell"]

            # Métricas
            total_index = sum(e["index"] for e in events)
            avg_index = total_index / len(events) if events else 0
            max_index = max((e["index"] for e in events), default=0)
            total_volume = sum(e["volume_usd"] for e in events)

            # Último evento
            latest = max(events, key=lambda e: e["timestamp_ms"])
            oldest = min(events, key=lambda e: e["timestamp_ms"])

            # Lado dominante
            if len(buy_events) > len(sell_events):
                dominant_side = "buy_defense"
            elif len(sell_events) > len(buy_events):
                dominant_side = "sell_defense"
            else:
                dominant_side = "contested"

            # Distância ao preço atual
            distance = 0
            distance_pct = 0
            direction = "unknown"
            if current_price > 0:
                distance = abs(center - current_price)
                distance_pct = (distance / current_price) * 100
                direction = "below" if center < current_price else "above"

            zones_result.append({
                "center": round(center, 2),
                "range_low": round(min(e["price"] for e in events), 2),
                "range_high": round(max(e["price"] for e in events), 2),
                "event_count": len(events),
                "buy_events": len(buy_events),
                "sell_events": len(sell_events),
                "dominant_side": dominant_side,
                "total_strength": round(total_index, 4),
                "avg_strength": round(avg_index, 4),
                "max_strength": round(max_index, 4),
                "total_volume_usd": round(total_volume, 2),
                "last_event_ms": latest["timestamp_ms"],
                "last_classification": latest["classification"],
                "zone_age_ms": latest["timestamp_ms"] - oldest["timestamp_ms"],
                "distance_from_price": round(distance, 2),
                "distance_pct": round(distance_pct, 4),
                "direction": direction,
                "reliability": (
                    "HIGH" if len(events) >= 5 and avg_index > 0.4
                    else "MEDIUM" if len(events) >= 3 or avg_index > 0.3
                    else "LOW"
                ),
            })

        # Ordenar por força total (mais forte primeiro)
        zones_result.sort(key=lambda z: z["total_strength"], reverse=True)
        zones_result = zones_result[:top_n]

        buy_zones = [z for z in zones_result if z["dominant_side"] == "buy_defense"]
        sell_zones = [z for z in zones_result if z["dominant_side"] == "sell_defense"]

        return {
            "zones": zones_result,
            "total_zones": len(zones_result),
            "total_events": len(self._events),
            "buy_zone_count": len(buy_zones),
            "sell_zone_count": len(sell_zones),
            "strongest_zone": zones_result[0] if zones_result else None,
            "status": "success",
        }

    def get_summary(self) -> dict:
        """Resumo rápido sem cálculos pesados."""
        if not self._events:
            return {"status": "empty", "total_events": 0}

        buy_events = sum(1 for e in self._events if e["side"] == "buy")
        sell_events = sum(1 for e in self._events if e["side"] == "sell")

        return {
            "status": "ok",
            "total_events": len(self._events),
            "buy_absorptions": buy_events,
            "sell_absorptions": sell_events,
            "avg_index": round(
                sum(e["index"] for e in self._events) / len(self._events), 4
            ),
            "dominant_side": "buy" if buy_events > sell_events else "sell" if sell_events > buy_events else "balanced",
        }

    def reset(self) -> None:
        """Limpa todo o histórico."""
        self._events.clear()


# ==============================================================================
# FUNÇÕES DE CONVENIÊNCIA
# ==============================================================================

# Instância global para uso simples
_default_classifier = AbsorptionClassifier()
_default_analyzer = AbsorptionAnalyzer()


def classify_absorption(
    delta_btc: float,
    open_p: float,
    high_p: float,
    low_p: float,
    close_p: float,
    eps: float = DEFAULT_ABSORCAO_DELTA_EPS,
    atr: Optional[float] = None,
    price_volatility: Optional[float] = None,
) -> str:
    """
    Função de conveniência para classificar absorção.
    
    Args:
        delta_btc: Delta de volume
        open_p, high_p, low_p, close_p: OHLC
        eps: Epsilon para delta neutro
        atr: ATR para tolerância dinâmica
        price_volatility: Volatilidade para tolerância dinâmica
        
    Returns:
        Rótulo de absorção
    """
    classifier = AbsorptionClassifier()
    classifier.update_volatility(atr=atr, price_volatility=price_volatility)
    return classifier.classify(delta_btc, open_p, high_p, low_p, close_p, eps)


def classify_absorption_simple(delta: float, eps: float = DEFAULT_ABSORCAO_DELTA_EPS) -> str:
    """
    Classificador simples apenas por delta.
    
    Args:
        delta: Delta de volume
        eps: Epsilon
        
    Returns:
        Rótulo de absorção
    """
    return _default_classifier.classify_simple(delta, eps)
```


# SEÇÃO 6 — REDUNDÂNCIAS DE CÓDIGO / CHAMADAS DUPLICADAS


## 6.1 Contagem grep — ai_runner.run / client.futures_ / client.get_


Comando equivalente rodado: `Select-String -Pattern 'ai_runner\.run|client\.futures_|client\.get_'` em todos os .py de `market_orchestrator/` + `main.py`.


```text
TOTAL de ocorrências: 0

O padrão literal não existe no código atual:
- Chamadas à IA são feitas via bot.ai_analyzer.analyze(event_data) (ai_runner.py:659) e bot.ai_analyzer._a_call_openai_text/_call_openai_compatible (analyzer_qwen.py).
- Chamadas à Binance são via WebSocket (monitoring/websocket_handler.py) e aiohttp direto (market_orchestrator.py:2149-2155 para klines de pré-carga).
```


## 6.2 Chamadas à IA dentro do mesmo ciclo (potencial de custo)


`_call_openai_compatible` (analyzer_qwen.py:3407-3533): **até 6 chamadas por análise** — `max_retries=3` × `strict_json_modes=[True, False]` (2 modos). Cada tentativa reenvia o prompt inteiro (payload + system prompt).


`_a_call_openai_text` (analyzer_qwen.py:3363-3405): itera `_groq_model_candidates` (modelo principal + 2 fallbacks) = até 3 chamadas sequenciais em caso de erro/rate-limit.


Dois pontos de entrada disparam `run_ai_analysis_threaded` no orchestrator:

- `_handle_signal_event` (market_orchestrator.py:928-979) → filtro `_is_important_event_for_ai` + cooldown `_ai_min_interval_sec` → `_run_ai_analysis_threaded`.
- `_handle_zone_touch_event` (market_orchestrator.py:981-995) → ignora cooldown (comentário 'BYPASS DE COOLDOWN') → `_run_ai_analysis_threaded`.
- Há também gate do throttler v3 em ai_runner.py:639-657 (should_call_ai) antes de `analyze()`.

## 6.3 Módulos de eventos — proxy vs módulos reais


Não há `event_bus.py`/`event_saver.py`/`event_memory.py` na raiz do projeto. Existe apenas a versão real em `events/`:

- `events/event_bus.py`, `events/event_saver.py`, `events/event_memory.py`, `events/event_similarity.py`, `events/event_stats_model.py`.
- Imports ativos encontrados: `market_orchestrator.py:53 (events.event_memory)`, `:62 (events.event_similarity)`, `:76 (events.event_saver)`, `:84 (events.event_bus)`, `:2123 (event_bus=)`. **Sem proxies/re-exports duplicados.**
- Usos de `event_saver.save_event` em 3 pontos: market_orchestrator.py:1512, 1696, 2063 — persistência em DB + eventos de IA.
- Também existe `tools/export_db_to_jsonl.py` e `events/event_store.py` (referenciado em alguns testes/scripts).

## 6.4 Outras duplicações/redundâncias identificadas

- **eventos-fluxo.json vs eventos_fluxo.jsonl**: conteúdo idêntico, formatos diferentes (ver Seção 2.1).
- **3 construtores de payload coexistem**: `payload_builder_compact.py` (ativo), `ai_payload_builder.py` (só get_llm_payload_config), `payload_compressor_v3.py` (só testes). Também existe `payload_compressor.py` (v1).
- **Arquivos .bak**: `ai_payload_builder.py.bak`, `llm_payload_guardrail.py.bak`, `llm_response_validator.py.bak`, `payload_compressor_v3.py.bak` no mesmo diretório.
- **legacy/**: `market_analyzer.py`, `market_analyzer_2_3_0.py`, `support_resistance_legacy.py`, `patch_ai_analyzer.py`, `ai_analyzer_qwen_patch2.py`, `ai_analyzer_disabled.py`, `data_pipeline_legacy..py` (nota: nome com '..').
- **institutional/**: `absorption_detector.py` + `flow_analyzer/absorption.py` — dois módulos de absorção em pastas diferentes (uso a verificar).
- **whale detection duplicada**: `institutional/whale_detector.py` (WhaleDetector) + `flow_analyzer/whale_score.py` (WhaleAccumulationCalculator) + `flow_analyzer/core.py` (whale threshold) — 3 implementações de detecção de baleia.
- **orderbook_analyzer/**: `analyzer.py`, `core.py`, `legacy_simplified.py`, `config/settings.py` + raiz `orderbook_analyzer.py`.
- **Throttler configurado 2x com valores diferentes**: ai_runner.py usa min=60/hard=30/budget=85k/max=10; defaults do dataclass são min=180/hard=60/budget=50k/max=6. `analyzer_qwen.py` usa `get_throttler()` sem kwargs → pode pegar o singleton JÁ configurado por ai_runner (primeira inicialização vence).
- **Cooldown duplo**: `_ai_min_interval_sec` no orchestrator (linhas 896, 946, 988) + `SmartAIThrottler` no ai_runner (camadas independentes).

# SEÇÃO 7 — LOGS DE EXECUÇÃO RECENTE


## 7.1 dados/eventos_visuais.log — últimas 100 linhas


(Arquivo com JSON pretty-printed por evento; as 100 últimas linhas são o fim do evento mais recente — event_id=16e36cc3, 2026-08-07T19:43:00-03:00)


```json

----------------------------------------------------------------------------------------------------
 # Janela 1
----------------------------------------------------------------------------------------------------
EVENTO: ANALYSIS_TRIGGER | SYMBOL: BTCUSDT | JANELA: 1
UTC: 2026-08-07 22:31:00 UTC
NY:  2026-08-07 18:31:00 EST/EDT
SP:  2026-08-07 19:31:00 BRT
----------------------------------------------------------------------------------------------------
{
  "is_signal": 1,
  "tipo_evento": "ANALYSIS_TRIGGER",
  "descricao": "Evento automático para análise da IA",
  "symbol": "BTCUSDT",
  "raw_event": {
    "delta": -0.552,
    "volume_total": 0.773,
    "volume_compra": 0.111,
    "volume_venda": 0.662,
    "preco_fechamento": 64925,
    "advanced_analysis": {
      "symbol": "BTCUSDT",
      "price": 64925,
      "volume": 0.773,
      "adaptive_thresholds": {
        "current_volatility": 0.03,
        "volatility_factor": 2.0,
        "absorption_threshold": 0.3,
        "flow_threshold": 0.2
      },
      "onchain_metrics": {
        "difficulty": 126.23,
        "active_addresses": 717974,
        "mempool_size": 87009,
        "mempool_vsize_mb": 43.57,
        "mempool_total_fee_btc": 0.1575,
        "fees_fastest_sat_vb": 5,
        "fees_half_hour_sat_vb": 4,
        "fees_hour_sat_vb": 3,
        "fees_economy_sat_vb": 2,
        "difficulty_adjustment": {
          "progress_pct": 93.4,
          "estimated_change_pct": 0.75,
          "remaining_blocks": 133,
          "remaining_time_ms": 79249114,
          "previous_retarget_pct": -0.74
        },
        "minutes_between_blocks": 8.4423,
        "total_btc_sent_24h": 807619.08,
        "trade_volume_btc_24h": 3265.64,
        "data_source": "blockchain.info+mempool.space",
        "is_real_data": 1,
        "requires_paid_api": [
          "exchange_netflow",
          "whale_transactions",
          "exchange_reserves",
          "sopr"
        ]
      }
    }
  },
  "resultado_da_batalha": "N/A",
  "delta": -0.552,
  "volume_total": 0.773,
  "volume_compra": 0.111,
  "volume_venda": 0.662,
  "preco_fechamento": 64925,
  "timestamp": "2026-08-07T22:31:13Z",
  "epoch_ms": 1786141860000,
  "ml_features": {
    "price_features": {
      "returns_1": 0.0,
      "volatility_1": 0.0,
      "returns_5": -1.5e-07,
      "volatility_5": 1.2e-07,
      "returns_15": 0.0,
      "volatility_15": 1.3e-07,
      "momentum_score": 0.70710678,
      "volatility_1h": 0.0023
    },
    "volume_features": {
      "volume_sma_ratio": 5,
      "volume_momentum": 7.924,
      "buy_sell_pressure": -0.7134,
      "liquidity_gradient": 7.9238963
    },
    "microstructure": {
      "order_book_slope": 1.925663,
      "flow_imbalance": -0.7134,
      "tick_rule_sum": -1,
      "trade_intensity": 0,
      "trade_intensity_v2": 1.1333
    },
    "cross_asset": {
      "btc_eth_corr_7d": 0.821,
      "btc_eth_corr_30d": 0.8598,
      "btc_dxy_corr_30d": -0.0625,
      "btc_dxy_corr_90d": -0.0677,
      "btc_ndx_corr_30d": 0.4156,
      "dxy_return_5d": -0.3561,
      "dxy_return_20d": -1.6548,
      "btc_dxy_correlation_stability": 0.0052,
      "btc_dxy_inverse_strength": 0.0651,
      "dxy_momentum": -1.29867555,
      "vix_current": 14.9,
      "us10y_yield": 4.66,
      "btc_dominance": 17.0254,
      "btc_dominance_change_7d": 0,
      "eth_dominance": 9.1117,
      "gold_price": 4342.3475,
      "oil_price": 77.08,
      "macro_regime": "RISK_ON",
      "correlation_regime": "DECORRELATED"
    }
  },
  "orderbook_data": {
    "mid": 64895.65,
    "spread": 0.1,
    "spread_percent": 0,
    "bid_depth_usd": 1223162.4,
    "ask_depth_usd": 1324540.04,
    "imbalance": -0.04,
    "flow_imbalance": -0.0398,
    "volume_ratio": 0.923,
    "pressure": -0.0398,
    "consolidated_bias_score": 0.4804,
    "spread_bps": 0.0154,
    "is_valid": 1,
    "data_source": "live"
  },
  "historical_vp": {
    "daily": {
      "poc": 65046,
      "vah": 65346,
      "val": 64522,
      "status": "success"
    },
    "weekly": {
      "poc": 62610,
      "vah": 63567,
      "val": 62314,
      "status": "success"
    },
    "monthly": {
      "poc": 63554,
      "vah": 64214,
      "val": 61789,
      "status": "success"
    }
  },
  "multi_tf": {
    "15m": {
      "tendencia": "Baixa",
      "preco_atual": 64924.96,
      "mme_21": 64930.22,
      "atr": 63.1,
      "regime": "Range",
      "rsi_short": 46.61,
      "rsi_long": 49.31,
      "macd": 16.879,
      "macd_signal": 17.5013,
      "adx": 25.4,
      "realized_vol": 0.0014
    },
    "1h": {
      "tendencia": "Alta",
      "preco_atual": 64924.96,
      "mme_21": 64793.62,
      "atr": 219.83,
      "regime": "Range",
      "rsi_short": 57.11,
      "rsi_long": 57.27,
      "macd": 104.5803,
      "macd_signal": 99.1456,
      "adx": 24.96,
      "realized_vol": 0.0023
    },
    "4h": {
      "tendencia": "Alta",
      "preco_atual": 64924.96,
      "mme_21": 64465.17,
      "atr": 491.56,
      "regime": "Range",
      "rsi_short": 64.6,
      "rsi_long": 62.72,
      "macd": 279.1724,
      "macd_signal": 257.5559,
      "adx": 30.01,
      "realized_vol": 0.0058
    },
    "1d": {
      "tendencia": "Alta",
      "preco_atual": 64924.96,
      "mme_21": 64149.58,
      "atr": 1369.69,
      "regime": "Manipulação",
      "rsi_short": 59.35,
      "rsi_long": 55.96,
      "macd": 71.3764,
      "macd_signal": 36.7677,
      "adx": 15.05,
      "realized_vol": 0.0184
    }
  },
  "data_context": "real_time",
  "price_targets": [
    {
      "level": 66872.75,
      "confidence": 0.3,
      "source": "fallback_r1",
      "weight": 0.1
    },
    {
      "level": 68820.5,
      "confidence": 0.2,
      "source": "fallback_r2",
      "weight": 0.05
    },
    {
      "level": 62977.25,
      "confidence": 0.3,
      "source": "fallback_s1",
      "weight": 0.1
    },
    {
      "level": 61029.5,
      "confidence": 0.2,
      "source": "fallback_s2",
      "weight": 0.05
    },
    {
      "level": 64925,
      "confidence": 0.4,
      "source": "fallback_poc",
      "weight": 0.15
    }
  ],
  "pattern_recognition": {
    "smart_money": {
      "market_structure": {
        "structure": "BEARISH",
        "bos_detected": 0,
        "last_swing_high": 64947.7,
        "last_swing_low": 64909.55,
        "higher_highs": 0,
        "higher_lows": 0
      }
    }
  },
  "janela_numero": 1,
  "timestamp_utc": "2026-08-07T22:31:00.000Z",
  "derivatives": {
    "BTCUSDT": {
      "funding_rate_percent": 0.01,
      "open_interest": 106941.874,
      "open_interest_usd": 6940097829.41,
      "long_short_ratio": 1.1,
      "longs_usd": 3634502292.66,
      "shorts_usd": 3305595536.75
    },
    "ETHUSDT": {
      "funding_rate_percent": 0,
      "open_interest": 2279582.358,
      "open_interest_usd": 4363895691.21,
      "long_short_ratio": 2.07,
      "longs_usd": 2940391094.65,
      "shorts_usd": 1423504596.56
    }
  },
  "fluxo_continuo": {
    "cvd": -0.5515,
    "whale_buy_volume": 0,
    "whale_sell_volume": 0,
    "whale_delta": 0,
    "bursts": {
      "count": 0,
      "max_burst_volume": 0
    },
    "sector_flow": {
      "retail": {
        "buy": 0.1122,
        "sell": 0.6638,
        "delta": -0.552
      },
      "mid": {
        "buy": 0,
        "sell": 0,
        "delta": 0
      },
      "whale": {
        "buy": 0,
        "sell": 0,
        "delta": 0
      }
    },
    "timestamp": "2026-08-07T22:31:00.000+00:00",
    "time_index": {
      "epoch_ms": 1786141860000,
      "timestamp_utc": "2026-08-07T22:31:00.000+00:00",
      "timestamp_ny": "2026-08-07T18:31:00.000-04:00",
      "timestamp_sp": "2026-08-07T19:31:00.000-03:00"
    },
    "order_flow": {
      "net_flow_1m": -35808.0621,
      "absorcao_1m": "Neutra",
      "buy_volume": 7193.69,
      "sell_volume": 43010.84,
      "total_volume": 50204.52,
      "buy_volume_btc": 0.111,
      "sell_volume_btc": 0.662,
      "total_volume_btc": 0.773,
      "whale_buy_volume_window": 0,
      "whale_sell_volume_window": 0,
      "whale_delta_window": 0,
      "flow_imbalance": -0.7134,
      "aggressive_buy_pct": 14.33,
      "aggressive_sell_pct": 85.67,
      "net_flow_5m": -35808.0621,
      "absorcao_5m": "Neutra",
      "net_flow_15m": -35808.0621,
      "absorcao_15m": "Neutra",
      "computation_window_min": 1,
      "available_windows_min": [
        1,
        5,
        15
      ],
      "buy_sell_ratio": {
        "buy_sell_ratio": 0.17,
        "ratios": {
          "current": 0.1673,
          "imbalance_1m": -0.713,
          "imbalance_5m": -0.713,
          "imbalance_15m": -0.713
        },
        "sector_ratios": {
          "retail": 0.1691,
          "mid": 1,
          "whale": 1
        },
        "pressure": "STRONG_SELL",
        "flow_trend": "accelerating_selling",
        "buy_volume": 0.111,
        "sell_volume": 0.662
      }
    },
    "tipo_absorcao": "Neutra",
    "participant_analysis": {
      "retail": {
        "volume_pct": 100,
        "direction": "SELL",
        "sentiment": "BEARISH",
        "composite_score": -0.706,
        "imbalance": -0.713
      }
    },
    "liquidity_heatmap": {
      "clusters": [
        {
          "center": 64924.9662,
          "low": 64924.96,
          "high": 64924.97,
          "width": 0.01,
          "total_volume": 0.776,
          "buy_volume": 0.112,
          "sell_volume": 0.664,
          "imbalance": -0.552,
          "imbalance_ratio": -0.711,
          "trades_count": 68,
          "avg_trade_size": 0.011,
          "recent_timestamp": 1786141865559,
          "recent_ts_ms": 1786141865559,
          "last_seen_ms": 1786141865559,
          "first_seen_ms": 1786141818682,
          "age_ms": 138.0,
          "cluster_duration_ms": 46877,
          "price_std": 0.0049,
          "volume_std": 0.048,
          "bin_threshold_usd": 194.7749
        }
      ],
      "resistances": [
        64924.9662
      ],
      "clusters_count": 1
    },
    "absorption_analysis": {
      "current_absorption": {
        "index": 0.5088,
        "classification": "MODERATE_ABSORPTION",
        "label": "Neutra",
        "buyer_strength": 1.4,
        "seller_exhaustion": 7.1,
        "continuation_probability": 0.46,
        "delta_usd": -35808.062,
        "total_volume_usd": 50204.52,
        "flow_imbalance": -0.7134,
        "window_min": 1
      }
    }
  },
  "market_context": {
    "trading_session": "NY",
    "session_phase": "ACTIVE",
    "time_to_session_close": 12573,
    "day_of_week": 4,
    "is_holiday": 0,
    "market_hours_type": "EXTENDED"
  },
  "market_environment": {
    "volatility_regime": "NORMAL",
    "trend_direction": "UP",
    "market_structure": "RANGE_BOUND",
    "liquidity_environment": "NORMAL",
    "risk_sentiment": "BULLISH",
    "correlation_spy": 0.406,
    "correlation_dxy": -0.0867,
    "correlation_gold": 0.2401
  },
  "contextual_snapshot": {
    "symbol": "BTCUSDT",
    "ohlc": {
      "open": 64925,
      "high": 64925,
      "low": 64925,
      "close": 64925,
      "open_time": 1786141818682,
      "close_time": 1786141859925,
      "vwap": 64925
    },
    "volume_total": 0.773,
    "volume_total_usdt": 50205,
    "volume_compra": 0.111,
    "volume_venda": 0.662,
    "num_trades": 62,
    "delta_minimo": -0.552,
    "delta_maximo": 0.019,
    "delta_fechamento": -0.552,
    "reversao_desde_minimo": 0,
    "reversao_desde_maximo": 0.57,
    "poc_price": 64925,
    "poc_volume": 0.66,
    "poc_percentage": 85.7,
    "dwell_price": 64925,
    "dwell_seconds": 40,
    "dwell_location": "Low",
    "trades_per_second": 1.5,
    "avg_trade_size": 0.012
  },
  "order_book_depth": {
    "L1": {
      "bids": 269381.64,
      "asks": 480552.66,
      "flow_imbalance": -0.2816
    },
    "L5": {
      "bids": 366724.66,
      "asks": 989011.27,
      "flow_imbalance": -0.459
    },
    "L10": {
      "bids": 376264.21,
      "asks": 1038202.62,
      "flow_imbalance": -0.468
    },
    "L25": {
      "bids": 481521.87,
      "asks": 1096221.21,
      "flow_imbalance": -0.3896
    },
    "total_depth_ratio": 0.44
  },
  "orderbook_quality": "live",
  "market_impact": {
    "slippage_matrix": {
      "100k_usd": {
        "buy": 0.05,
        "sell": 0.05
      },
      "1m_usd": {
        "buy": 0.55,
        "sell": 5.15
      }
    },
    "liquidity_score": 9.9985,
    "execution_quality": "EXCELLENT"
  },
  "institutional_analytics": {
    "status": "ok",
    "computed_at_ms": 1786141873153,
    "technical_extras": {
      "stoch_rsi": {
        "k": 58.2,
        "d": 58.2,
        "overbought": 0,
        "oversold": 0,
        "crossover": "none"
      },
      "williams_r": {
        "value": -59.61,
        "overbought": 0,
        "oversold": 0,
        "zone": "neutral",
        "source": "real"
      },
      "hurst_exponent": 0.3677,
      "shannon_entropy": 3.0047,
      "kalman_filter": {
        "kalman_price": 64943.96,
        "raw_price": 64924.96,
        "deviation_pct": -0.03,
        "trend_direction": "DOWN"
      },
      "regression_channel": {
        "slope_per_bar": -2.0863,
        "trend_price": 64893.36,
        "upper_1sd": 64920.52,
        "lower_1sd": 64866.2,
        "upper_2sd": 64947.67,
        "lower_2sd": 64839.04,
        "deviation_from_trend": 31.6,
        "position_in_channel": 0.7909
      },
      "dominant_cycles": {
        "dominant_cycles": [
          100,
          40,
          33.3
        ],
        "cycle_strengths": [
          3314.59,
          1999.72,
          1182.19
        ]
      },
      "fractal_dimension": 0.6171,
      "monte_carlo": {
        "median_price": 64926.67,
        "p10": 64868.4,
        "p25": 64897.49,
        "p75": 64960.68,
        "p90": 64991.76,
        "prob_up": 0.52,
        "horizon_bars": 12
      },
      "market_structure": {
        "structure": "BEARISH",
        "bos_detected": 0,
        "last_swing_high": 64947.7,
        "last_swing_low": 64909.55,
        "higher_highs": 0,
        "higher_lows": 0
      }
    },
    "profile_analysis": {
      "poor_extremes": {
        "poor_high": {
          "detected": 1,
          "price": 64924.97,
          "volume_ratio": 1.433,
          "implication": "High likely to be revisited - unfinished auction"
        },
        "poor_low": {
          "detected": 1,
          "price": 64924.96,
          "volume_ratio": 8.567,
          "implication": "Low likely to be revisited - unfinished auction"
        },
        "excess_high": 0,
        "excess_low": 0,
        "session_high": 64924.97,
        "session_low": 64924.96,
        "action_bias": "expect_retest_both",
        "status": "success"
      },
      "profile_shape": {
        "shape": "b",
        "implication": "Long liquidation - bullish bias expected",
        "trading_signal": "BULLISH_AFTER",
        "distribution": {
          "lower_third_pct": 85.7,
          "middle_third_pct": 0,
          "upper_third_pct": 14.3
        },
        "dominant_zone": "lower",
        "status": "success"
      },
      "no_mans_land": {
        "zones": [
          {
            "range_low": 64512,
            "range_high": 64810,
            "gap_size": 298,
            "gap_size_pct": 0.462,
            "risk": "LOW",
            "lvn_confirmed": 1,
            "lvns_in_zone": 9,
            "nearest_hvn_below": 64512,
            "nearest_hvn_above": 64810,
            "distance_from_price": 115,
            "distance_pct": 0.18,
            "direction": "below"
          }
        ],
        "price_in_no_mans_land": 0,
        "nearest_no_mans_land": {
          "range_low": 64512,
          "range_high": 64810,
          "gap_size": 298,
          "gap_size_pct": 0.462,
          "risk": "LOW",
          "lvn_confirmed": 1,
          "lvns_in_zone": 9,
          "nearest_hvn_below": 64512,
          "nearest_hvn_above": 64810,
          "distance_from_price": 115,
          "distance_pct": 0.18,
          "direction": "below"
        },
        "total_zones": 1,
        "max_gap_pct": 0.46,
        "status": "success"
      },
      "va_volume_pct": {
        "value_area_volume_pct": 100,
        "interpretation": "extremely_compressed",
        "breakout_risk": "VERY_HIGH",
        "volume_in_va": 1,
        "total_volume": 1,
        "compression_signal": 1
      },
      "volume_node_strength": {
        "scored_hvns": [
          {
            "price": 64354,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64304,
            "strength": 70,
            "volume_score": 15,
            "proximity_score": 25.2,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64306,
            "strength": 70,
            "volume_score": 15,
            "proximity_score": 25.2,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64314,
            "strength": 70,
            "volume_score": 15,
            "proximity_score": 25.3,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64320,
            "strength": 70,
            "volume_score": 15,
            "proximity_score": 25.3,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          }
        ],
        "scored_lvns": [
          {
            "price": 64480,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64485,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64500,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64516,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.9,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64348,
            "strength": 64,
            "volume_score": 15,
            "proximity_score": 25.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          }
        ],
        "total_hvns": 52,
        "total_lvns": 101,
        "avg_hvn_strength": 66.4,
        "avg_lvn_strength": 62.8,
        "status": "success"
      }
    },
    "flow_analysis": {
      "passive_aggressive": {
        "aggressive": {
          "buy_pct": 14.33,
          "sell_pct": 85.67,
          "net_pct": -71.34,
          "dominance": "sellers",
          "buy_volume": 0.111,
          "sell_volume": 0.662
        },
        "passive": {
          "dominance": "balanced",
          "inference": "from_orderbook_depth",
          "bid_depth": 1223162.4,
          "ask_depth": 1324540.04,
          "bid_ratio": 0.48,
          "ob_imbalance": -0.04
        },
        "composite": {
          "signal": "mixed",
          "interpretation": "Mixed signals between aggressive and passive flow",
          "conviction": "LOW"
        },
        "status": "success"
      },
      "whale_accumulation": {
        "score": -19,
        "classification": "NEUTRAL",
        "bias": "DISTRIBUTING",
        "components": {
          "flow": {
            "score": -2.76,
            "max": 30,
            "detail": {
              "divergence": "aligned",
              "whale_delta": 0,
              "mid_delta": 0,
              "retail_delta": -0.551,
              "primary_delta": 0,
              "cvd_used": 1
            }
          },
          "depth": {
            "score": -0.8,
            "max": 20,
            "detail": {
              "bid_depth": 1223162.4,
              "ask_depth": 1324540.04,
              "ratio": -0.04,
              "deep_confirmation": 0
            }
          },
          "absorption": {
            "score": -17.1,
            "max": 25,
            "detail": {
              "buyer_strength": 1.4,
              "seller_exhaustion": 7.1,
              "net_absorption": -5.7,
              "index": 0.5088,
              "label": "Neutra"
            }
          },
          "derivatives": {
            "score": 1.49,
            "max": 25,
            "detail": {
              "long_short_ratio": 1.1,
              "lsr_score": 1.49
            }
          }
        },
        "trend": {
          "direction": "insufficient_data",
          "avg_score": -19,
          "samples": 1
        },
        "status": "success"
      },
      "absorption_zones": {
        "total_zones": 0,
        "total_events": 0,
        "buy_zone_count": 0,
        "sell_zone_count": 0,
        "status": "no_events"
      }
    },
    "sr_analysis": {
      "defense_zones": {
        "buy_defense": [
          {
            "center": 64481.04,
            "range_low": 64376.31,
            "range_high": 64570.69,
            "strength": 62,
            "side": "buy",
            "sources": [
              "sr_level_ema_21_4h",
              "vp_hvn",
              "ema_ema_21_4h",
              "vp_val"
            ],
            "source_count": 4,
            "signals_in_zone": 5,
            "type": "confluence",
            "distance_from_price": 443.96,
            "distance_pct": 0.68
          },
          {
            "center": 64838.27,
            "range_low": 64744.93,
            "range_high": 64928.69,
            "strength": 49,
            "side": "buy",
            "sources": [
              "vp_hvn",
              "ema_ema_21_1h",
              "sr_level_ema_21_1h"
            ],
            "source_count": 3,
            "signals_in_zone": 7,
            "type": "confluence",
            "distance_from_price": 86.73,
            "distance_pct": 0.13
          },
          {
            "center": 64165.68,
            "range_low": 64100.89,
            "range_high": 64230.48,
            "strength": 46,
            "side": "buy",
            "sources": [
              "ema_ema_21_1d",
              "sr_level_vah_monthly"
            ],
            "source_count": 2,
            "signals_in_zone": 2,
            "type": "cluster",
            "distance_from_price": 759.32,
            "distance_pct": 1.17
          },
          {
            "center": 64336.25,
            "range_low": 64255.31,
            "range_high": 64442.69,
            "strength": 40,
            "side": "buy",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 9,
            "type": "cluster",
            "distance_from_price": 588.75,
            "distance_pct": 0.91
          }
        ],
        "sell_defense": [
          {
            "center": 65037.38,
            "range_low": 64960.31,
            "range_high": 65128.69,
            "strength": 55,
            "side": "sell",
            "sources": [
              "sr_level_poc_daily",
              "vp_poc",
              "vp_hvn"
            ],
            "source_count": 3,
            "signals_in_zone": 9,
            "type": "confluence",
            "distance_from_price": 112.38,
            "distance_pct": 0.17
          },
          {
            "center": 65346,
            "range_low": 65297.31,
            "range_high": 65394.69,
            "strength": 53,
            "side": "sell",
            "sources": [
              "sr_level_vah_daily",
              "vp_hvn",
              "vp_vah"
            ],
            "source_count": 3,
            "signals_in_zone": 3,
            "type": "confluence",
            "distance_from_price": 421,
            "distance_pct": 0.65
          },
          {
            "center": 64945.09,
            "range_low": 64860.31,
            "range_high": 65052.69,
            "strength": 49,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "ema_ema_21_15m",
              "sr_level_hvn_daily"
            ],
            "source_count": 3,
            "signals_in_zone": 9,
            "type": "confluence",
            "distance_from_price": 20.09,
            "distance_pct": 0.03
          },
          {
            "center": 65157.17,
            "range_low": 65064.31,
            "range_high": 65257.69,
            "strength": 41,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 13,
            "type": "cluster",
            "distance_from_price": 232.17,
            "distance_pct": 0.36
          },
          {
            "center": 65252.3,
            "range_low": 65164.31,
            "range_high": 65350.69,
            "strength": 40,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 11,
            "type": "cluster",
            "distance_from_price": 327.3,
            "distance_pct": 0.5
          }
        ],
        "total_zones": 9,
        "strongest_buy": {
          "center": 64481.04,
          "range_low": 64376.31,
          "range_high": 64570.69,
          "strength": 62,
          "side": "buy",
          "sources": [
            "sr_level_ema_21_4h",
            "vp_hvn",
            "ema_ema_21_4h",
            "vp_val"
          ],
          "source_count": 4,
          "signals_in_zone": 5,
          "type": "confluence",
          "distance_from_price": 443.96,
          "distance_pct": 0.68
        },
        "strongest_sell": {
          "center": 65037.38,
          "range_low": 64960.31,
          "range_high": 65128.69,
          "strength": 55,
          "side": "sell",
          "sources": [
            "sr_level_poc_daily",
            "vp_poc",
            "vp_hvn"
          ],
          "source_count": 3,
          "signals_in_zone": 9,
          "type": "confluence",
          "distance_from_price": 112.38,
          "distance_pct": 0.17
        },
        "defense_asymmetry": {
          "ratio": 0.83,
          "bias": "slight_sell_defense",
          "description": "Slightly more sell defense",
          "buy_total_strength": 197,
          "sell_total_strength": 238
        },
        "status": "success"
      }
    },
    "quality": {
      "calendar": {
        "day_of_week": "Friday",
        "day_of_week_num": 4,
        "is_us_holiday": 0,
        "is_weekend": 0,
        "is_pre_holiday": 0,
        "is_post_holiday": 0,
        "expected_liquidity": "NORMAL",
        "liquidity_warning": 0
      },
      "latency": {
        "latency_ms": 13366,
        "latency_category": "POOR",
        "data_freshness": "DELAYED",
        "is_acceptable": 0,
        "is_stale": 0
      },
      "spread_percentile": {
        "status": "insufficient_data",
        "samples": 1,
        "min_required": 5,
        "current_spread": 0.1,
        "spread_percentile": 50,
        "liquidity_signal": "UNKNOWN"
      },
      "anomalies": {
        "anomalies_detected": 1,
        "count": 1,
        "anomalies": [
          {
            "type": "FLOW_EXTREME_IMBALANCE",
            "severity": "HIGH",
            "value": -0.7134,
            "direction": "SELL",
            "description": "Extreme flow imbalance: -71.34% toward sellers"
          }
        ],
        "max_severity": "HIGH",
        "risk_elevated": 1,
        "types_found": [
          "FLOW_EXTREME_IMBALANCE"
        ],
        "summary": "1 anomalies detected (max severity: HIGH)"
      }
    },
    "candlestick_patterns": {
      "patterns_detected": 2,
      "patterns": [
        {
          "name": "pin_bar",
          "type": "bearish",
          "confidence": 0.72,
          "candles_used": 1,
          "implication": "Strong bearish rejection - higher prices rejected"
        },
        {
          "name": "gravestone_doji",
          "type": "bearish",
          "confidence": 0.65,
          "candles_used": 1,
          "implication": "Indecision - watch next candle for direction"
        }
      ],
      "dominant_signal": "bearish",
      "max_confidence": 0.72,
      "bullish_count": 0,
      "bearish_count": 2,
      "neutral_count": 0
    }
  },
  "sequence_id": 1,
  "primary_exchange": "BINANCE",
  "data_feed_type": "WEBSOCKET_L2",
  "data_quality_score": 8.75,
  "completeness_pct": 100,
  "reliability_score": 7.5,
  "bid": 64895.6,
  "ask": 64895.7,
  "tick_direction": 0,
  "twap": 64925,
  "pivot_points": {
    "daily": {
      "pivot": 64971.33,
      "r1": 65420.66,
      "r2": 65795.33,
      "r3": 66619.33,
      "s1": 64596.66,
      "s2": 64147.33,
      "s3": 63323.33,
      "vah": 65346,
      "val": 64522,
      "poc": 65046
    },
    "weekly": {
      "pivot": 62830.33,
      "r1": 63346.66,
      "r2": 64083.33,
      "r3": 65336.33,
      "s1": 62093.66,
      "s2": 61577.33,
      "s3": 60324.33,
      "vah": 63567,
      "val": 62314,
      "poc": 62610
    },
    "monthly": {
      "pivot": 63185.67,
      "r1": 64582.34,
      "r2": 65610.67,
      "r3": 68035.67,
      "s1": 62157.34,
      "s2": 60760.67,
      "s3": 58335.67,
      "vah": 64214,
      "val": 61789,
      "poc": 63554
    }
  },
  "immediate_support": [
    64838.27,
    64522,
    64481.04,
    64214,
    64165.68
  ],
  "support_strength": [
    49,
    93.8,
    62,
    71.2,
    46
  ],
  "immediate_resistance": [
    64945.09,
    64971.33,
    65037.38,
    65346
  ],
  "resistance_strength": [
    49,
    99.3,
    55,
    93.5
  ],
  "volatility_metrics": {
    "realized_vol_24h": 0.0184,
    "realized_vol_7d": 0.0487,
    "volatility_regime": "NORMAL"
  },
  "order_flow_extended": {
    "passive_buy_pct": 48,
    "passive_sell_pct": 52
  },
  "whale_activity": {
    "iceberg_activity": 0,
    "hidden_orders_detected": 0
  },
  "technical_indicators_extended": {
    "cci_1h": 159.32,
    "cci_signal": "OVERBOUGHT",
    "stochastic": {
      "k": 58.2,
      "d": 58.2,
      "signal": "NEUTRAL",
      "source": "real"
    },
    "williams_r": {
      "value": -59.61,
      "overbought": 0,
      "oversold": 0,
      "zone": "neutral",
      "source": "real"
    },
    "hurst_exponent": 0.3677,
    "shannon_entropy": 3.0047,
    "fractal_dimension": 0.6171,
    "kalman_filter": {
      "kalman_price": 64943.96,
      "raw_price": 64924.96,
      "deviation_pct": -0.03,
      "trend_direction": "DOWN"
    },
    "regression_channel": {
      "slope_per_bar": -2.0863,
      "trend_price": 64893.36,
      "upper_1sd": 64920.52,
      "lower_1sd": 64866.2,
      "upper_2sd": 64947.67,
      "lower_2sd": 64839.04,
      "deviation_from_trend": 31.6,
      "position_in_channel": 0.7909
    },
    "dominant_cycles": {
      "dominant_cycles": [
        100,
        40,
        33.3
      ],
      "cycle_strengths": [
        3314.59,
        1999.72,
        1182.19
      ]
    },
    "monte_carlo": {
      "median_price": 64926.67,
      "p10": 64868.4,
      "p25": 64897.49,
      "p75": 64960.68,
      "p90": 64991.76,
      "prob_up": 0.52,
      "horizon_bars": 12
    },
    "garch_forecast_1h": 0.0024
  },
  "backup_exchanges": [
    "COINBASE",
    "KRAKEN",
    "OKX"
  ],
  "alerts": {
    "active_alerts": [
      {
        "type": "SUPPORT_TEST",
        "level": 64838.27,
        "severity": "MEDIUM",
        "probability": 0.73,
        "action": "MONITOR_CLOSELY",
        "description": "Preço testando suporte em 64838.27 (dist: 0.13%)"
      },
      {
        "type": "RESISTANCE_TEST",
        "level": 64945.09,
        "severity": "HIGH",
        "probability": 0.94,
        "action": "MONITOR_CLOSELY",
        "description": "Preço testando resistência em 64945.09 (dist: 0.03%)"
      },
      {
        "type": "VOLUME_SPIKE",
        "threshold_exceeded": 5,
        "severity": "HIGH",
        "probability": 0.5,
        "action": "PREPARE_ENTRY",
        "description": "Volume 5.0x acima da média"
      }
    ],
    "alert_count": 3,
    "max_severity": "HIGH"
  },
  "regime_analysis": {
    "current_regime": "MEAN_REVERTING",
    "regime_probabilities": {
      "trending": 0.25,
      "mean_reverting": 0.583,
      "breakout": 0.167
    },
    "regime_change_probability": 0.24,
    "expected_regime_duration": "15m-1h",
    "avg_adx": 26.8
  },
  "data_reliability": {
    "has_options_data": 0,
    "onchain_coverage": "full",
    "latency_acceptable": 1,
    "price_targets_available": 1
  },
  "event_id": "6630d1e4",
  "timestamp_ny": "2026-08-07T18:31:00.000-04:00",
  "timestamp_sp": "2026-08-07T19:31:00.000-03:00",
  "_log_id": "6630d1e4"
}

----------------------------------------------------------------------------------------------------
 # Janela 2
----------------------------------------------------------------------------------------------------
EVENTO: ANALYSIS_TRIGGER | SYMBOL: BTCUSDT | JANELA: 2
UTC: 2026-08-07 22:32:00 UTC
NY:  2026-08-07 18:32:00 EST/EDT
SP:  2026-08-07 19:32:00 BRT
----------------------------------------------------------------------------------------------------
{
  "is_signal": 1,
  "tipo_evento": "ANALYSIS_TRIGGER",
  "descricao": "Evento automático para análise da IA",
  "symbol": "BTCUSDT",
  "raw_event": {
    "delta": 0.036,
    "volume_total": 0.149,
    "volume_compra": 0.093,
    "volume_venda": 0.057,
    "preco_fechamento": 64925,
    "advanced_analysis": {
      "symbol": "BTCUSDT",
      "price": 64925,
      "volume": 0.149,
      "adaptive_thresholds": {
        "current_volatility": 0.03,
        "volatility_factor": 2.0,
        "absorption_threshold": 0.3,
        "flow_threshold": 0.2
      },
      "onchain_metrics": {
        "difficulty": 126.23,
        "active_addresses": 717974,
        "mempool_size": 87009,
        "mempool_vsize_mb": 43.57,
        "mempool_total_fee_btc": 0.1575,
        "fees_fastest_sat_vb": 5,
        "fees_half_hour_sat_vb": 4,
        "fees_hour_sat_vb": 3,
        "fees_economy_sat_vb": 2,
        "difficulty_adjustment": {
          "progress_pct": 93.4,
          "estimated_change_pct": 0.75,
          "remaining_blocks": 133,
          "remaining_time_ms": 79249114,
          "previous_retarget_pct": -0.74
        },
        "minutes_between_blocks": 8.4423,
        "total_btc_sent_24h": 807619.08,
        "trade_volume_btc_24h": 3265.64,
        "data_source": "blockchain.info+mempool.space",
        "is_real_data": 1,
        "requires_paid_api": [
          "exchange_netflow",
          "whale_transactions",
          "exchange_reserves",
          "sopr"
        ]
      }
    }
  },
  "resultado_da_batalha": "N/A",
  "delta": 0.036,
  "volume_total": 0.149,
  "volume_compra": 0.093,
  "volume_venda": 0.057,
  "preco_fechamento": 64925,
  "timestamp": "2026-08-07T22:32:05Z",
  "epoch_ms": 1786141920000,
  "ml_features": {
    "price_features": {
      "returns_1": 1.5e-07,
      "volatility_1": 0.0,
      "returns_5": 1.5e-07,
      "volatility_5": 1.2e-07,
      "returns_15": 0.0,
      "volatility_15": 1.3e-07,
      "momentum_score": -1.41421356,
      "volatility_1h": 0.0023
    },
    "volume_features": {
      "volume_sma_ratio": 2.32,
      "volume_momentum": 1.828,
      "buy_sell_pressure": 0.2435,
      "liquidity_gradient": 1.82752435
    },
    "microstructure": {
      "order_book_slope": 0.064298,
      "flow_imbalance": 0.2435,
      "tick_rule_sum": 0,
      "trade_intensity": 0,
      "trade_intensity_v2": 1.4333
    },
    "cross_asset": {
      "btc_eth_corr_7d": 0.821,
      "btc_eth_corr_30d": 0.8598,
      "btc_dxy_corr_30d": -0.0625,
      "btc_dxy_corr_90d": -0.0677,
      "btc_ndx_corr_30d": 0.4156,
      "dxy_return_5d": -0.3561,
      "dxy_return_20d": -1.6548,
      "btc_dxy_correlation_stability": 0.0052,
      "btc_dxy_inverse_strength": 0.0651,
      "dxy_momentum": -1.29867555,
      "vix_current": 14.9,
      "us10y_yield": 4.66,
      "btc_dominance": 17.0254,
      "btc_dominance_change_7d": 0,
      "eth_dominance": 9.1117,
      "gold_price": 4342.3475,
      "oil_price": 77.08,
      "macro_regime": "RISK_ON",
      "correlation_regime": "DECORRELATED"
    }
  },
  "orderbook_data": {
    "mid": 64895.65,
    "spread": 0.1,
    "spread_percent": 0,
    "bid_depth_usd": 1425768.55,
    "ask_depth_usd": 1034391.61,
    "imbalance": 0.159,
    "flow_imbalance": 0.1591,
    "volume_ratio": 1.378,
    "pressure": 0.1591,
    "consolidated_bias_score": 0.5856,
    "spread_bps": 0.0154,
    "is_valid": 1,
    "data_source": "live"
  },
  "historical_vp": {
    "daily": {
      "poc": 65046,
      "vah": 65346,
      "val": 64522,
      "status": "success"
    },
    "weekly": {
      "poc": 62610,
      "vah": 63567,
      "val": 62314,
      "status": "success"
    },
    "monthly": {
      "poc": 63554,
      "vah": 64214,
      "val": 61789,
      "status": "success"
    }
  },
  "multi_tf": {
    "15m": {
      "tendencia": "Baixa",
      "preco_atual": 64924.96,
      "mme_21": 64930.22,
      "atr": 63.1,
      "regime": "Range",
      "rsi_short": 46.61,
      "rsi_long": 49.31,
      "macd": 16.879,
      "macd_signal": 17.5013,
      "adx": 25.4,
      "realized_vol": 0.0014
    },
    "1h": {
      "tendencia": "Alta",
      "preco_atual": 64924.96,
      "mme_21": 64793.62,
      "atr": 219.83,
      "regime": "Range",
      "rsi_short": 57.11,
      "rsi_long": 57.27,
      "macd": 104.5803,
      "macd_signal": 99.1456,
      "adx": 24.96,
      "realized_vol": 0.0023
    },
    "4h": {
      "tendencia": "Alta",
      "preco_atual": 64924.96,
      "mme_21": 64465.17,
      "atr": 491.56,
      "regime": "Range",
      "rsi_short": 64.6,
      "rsi_long": 62.72,
      "macd": 279.1724,
      "macd_signal": 257.5559,
      "adx": 30.01,
      "realized_vol": 0.0058
    },
    "1d": {
      "tendencia": "Alta",
      "preco_atual": 64924.96,
      "mme_21": 64149.58,
      "atr": 1369.69,
      "regime": "Manipulação",
      "rsi_short": 59.35,
      "rsi_long": 55.96,
      "macd": 71.3764,
      "macd_signal": 36.7677,
      "adx": 15.05,
      "realized_vol": 0.0184
    }
  },
  "data_context": "real_time",
  "price_targets": [
    {
      "level": 66872.75,
      "confidence": 0.3,
      "source": "fallback_r1",
      "weight": 0.1
    },
    {
      "level": 68820.5,
      "confidence": 0.2,
      "source": "fallback_r2",
      "weight": 0.05
    },
    {
      "level": 62977.25,
      "confidence": 0.3,
      "source": "fallback_s1",
      "weight": 0.1
    },
    {
      "level": 61029.5,
      "confidence": 0.2,
      "source": "fallback_s2",
      "weight": 0.05
    },
    {
      "level": 64925,
      "confidence": 0.4,
      "source": "fallback_poc",
      "weight": 0.15
    }
  ],
  "pattern_recognition": {
    "smart_money": {
      "market_structure": {
        "structure": "BEARISH",
        "bos_detected": 0,
        "last_swing_high": 64947.7,
        "last_swing_low": 64909.55,
        "higher_highs": 0,
        "higher_lows": 0
      }
    }
  },
  "janela_numero": 2,
  "timestamp_utc": "2026-08-07T22:32:00.000Z",
  "derivatives": {
    "BTCUSDT": {
      "funding_rate_percent": 0.01,
      "open_interest": 106941.874,
      "open_interest_usd": 6940097829.41,
      "long_short_ratio": 1.1,
      "longs_usd": 3634502292.66,
      "shorts_usd": 3305595536.75
    },
    "ETHUSDT": {
      "funding_rate_percent": 0,
      "open_interest": 2279582.358,
      "open_interest_usd": 4363895691.21,
      "long_short_ratio": 2.07,
      "longs_usd": 2940391094.65,
      "shorts_usd": 1423504596.56
    }
  },
  "fluxo_continuo": {
    "cvd": -0.4776,
    "whale_buy_volume": 0,
    "whale_sell_volume": 0,
    "whale_delta": 0,
    "bursts": {
      "count": 0,
      "max_burst_volume": 0
    },
    "sector_flow": {
      "retail": {
        "buy": 0.3016,
        "sell": 0.7792,
        "delta": -0.478
      },
      "mid": {
        "buy": 0,
        "sell": 0,
        "delta": 0
      },
      "whale": {
        "buy": 0,
        "sell": 0,
        "delta": 0
      }
    },
    "timestamp": "2026-08-07T22:32:00.000+00:00",
    "time_index": {
      "epoch_ms": 1786141920000,
      "timestamp_utc": "2026-08-07T22:32:00.000+00:00",
      "timestamp_ny": "2026-08-07T18:32:00.000-04:00",
      "timestamp_sp": "2026-08-07T19:32:00.000-03:00"
    },
    "order_flow": {
      "net_flow_1m": 4718.0987,
      "absorcao_1m": "Neutra",
      "buy_volume": 6028.93,
      "sell_volume": 3667.61,
      "total_volume": 9696.54,
      "buy_volume_btc": 0.093,
      "sell_volume_btc": 0.056,
      "total_volume_btc": 0.149,
      "whale_buy_volume_window": 0,
      "whale_sell_volume_window": 0,
      "whale_delta_window": 0,
      "flow_imbalance": 0.2435,
      "aggressive_buy_pct": 62.18,
      "aggressive_sell_pct": 37.82,
      "net_flow_5m": -31005.5609,
      "absorcao_5m": "Neutra",
      "net_flow_15m": -31005.5609,
      "absorcao_15m": "Neutra",
      "computation_window_min": 1,
      "available_windows_min": [
        1,
        5,
        15
      ],
      "buy_sell_ratio": {
        "buy_sell_ratio": 1.64,
        "ratios": {
          "current": 1.6438,
          "imbalance_1m": 0.487,
          "imbalance_5m": -3.198,
          "imbalance_15m": -3.198
        },
        "sector_ratios": {
          "retail": 0.3871,
          "mid": 1,
          "whale": 1
        },
        "pressure": "MODERATE_BUY",
        "flow_trend": "short_term_reversal_to_buy",
        "buy_volume": 0.093,
        "sell_volume": 0.057
      }
    },
    "tipo_absorcao": "Neutra",
    "participant_analysis": {
      "retail": {
        "volume_pct": 100,
        "direction": "BUY",
        "sentiment": "BULLISH",
        "composite_score": 0.517,
        "imbalance": 0.244
      }
    },
    "liquidity_heatmap": {
      "clusters": [
        {
          "center": 64924.967,
          "low": 64924.96,
          "high": 64924.97,
          "width": 0.01,
          "total_volume": 1.081,
          "buy_volume": 0.302,
          "sell_volume": 0.779,
          "imbalance": -0.478,
          "imbalance_ratio": -0.442,
          "trades_count": 148,
          "avg_trade_size": 0.007,
          "recent_timestamp": 1786141924663,
          "recent_ts_ms": 1786141924663,
          "last_seen_ms": 1786141924663,
          "first_seen_ms": 1786141818682,
          "age_ms": 141.0,
          "cluster_duration_ms": 105981,
          "price_std": 0.0046,
          "volume_std": 0.034,
          "bin_threshold_usd": 194.7749
        }
      ],
      "resistances": [
        64924.967
      ],
      "clusters_count": 1
    },
    "absorption_analysis": {
      "current_absorption": {
        "index": 0.1185,
        "classification": "WEAK_ABSORPTION",
        "label": "Neutra",
        "buyer_strength": 6.2,
        "seller_exhaustion": 2.4,
        "continuation_probability": 0.11,
        "delta_usd": 4718.099,
        "total_volume_usd": 9696.54,
        "flow_imbalance": 0.2435,
        "window_min": 1
      }
    }
  },
  "market_context": {
    "trading_session": "NY",
    "session_phase": "ACTIVE",
    "time_to_session_close": 12573,
    "day_of_week": 4,
    "is_holiday": 0,
    "market_hours_type": "EXTENDED"
  },
  "market_environment": {
    "volatility_regime": "NORMAL",
    "trend_direction": "UP",
    "market_structure": "RANGE_BOUND",
    "liquidity_environment": "NORMAL",
    "risk_sentiment": "BULLISH",
    "correlation_spy": 0.406,
    "correlation_dxy": -0.0867,
    "correlation_gold": 0.2401
  },
  "contextual_snapshot": {
    "symbol": "BTCUSDT",
    "ohlc": {
      "open": 64925,
      "high": 64925,
      "low": 64925,
      "close": 64925,
      "open_time": 1786141861129,
      "close_time": 1786141919283,
      "vwap": 64925
    },
    "volume_total": 0.149,
    "volume_total_usdt": 9697,
    "volume_compra": 0.093,
    "volume_venda": 0.057,
    "num_trades": 58,
    "delta_minimo": 0,
    "delta_maximo": 0.051,
    "delta_fechamento": 0.036,
    "reversao_desde_minimo": 0.04,
    "reversao_desde_maximo": 0.02,
    "poc_price": 64925,
    "poc_volume": 0.09,
    "poc_percentage": 62.2,
    "dwell_price": 64925,
    "dwell_seconds": 58,
    "dwell_location": "High",
    "trades_per_second": 1,
    "avg_trade_size": 0.003
  },
  "order_book_depth": {
    "L1": {
      "bids": 424871.49,
      "asks": 474647.15,
      "flow_imbalance": -0.0553
    },
    "L5": {
      "bids": 624684.54,
      "asks": 711711.51,
      "flow_imbalance": -0.0651
    },
    "L10": {
      "bids": 634743.26,
      "asks": 737605.21,
      "flow_imbalance": -0.075
    },
    "L25": {
      "bids": 732084.16,
      "asks": 790496.76,
      "flow_imbalance": -0.0384
    },
    "total_depth_ratio": 0.93
  },
  "orderbook_quality": "live",
  "market_impact": {
    "slippage_matrix": {
      "100k_usd": {
        "buy": 0.05,
        "sell": 0.05
      },
      "1m_usd": {
        "buy": 5.15,
        "sell": 4.95
      }
    },
    "liquidity_score": 9.9985,
    "execution_quality": "EXCELLENT"
  },
  "institutional_analytics": {
    "status": "ok",
    "computed_at_ms": 1786141925798,
    "technical_extras": {
      "stoch_rsi": {
        "k": 72.13,
        "d": 62.84,
        "overbought": 0,
        "oversold": 0,
        "crossover": "none"
      },
      "williams_r": {
        "value": -59.5,
        "overbought": 0,
        "oversold": 0,
        "zone": "neutral",
        "source": "real"
      },
      "hurst_exponent": 0.362,
      "shannon_entropy": 2.9921,
      "kalman_filter": {
        "kalman_price": 64943.38,
        "raw_price": 64925,
        "deviation_pct": -0.03,
        "trend_direction": "DOWN"
      },
      "regression_channel": {
        "slope_per_bar": -1.8405,
        "trend_price": 64896.61,
        "upper_1sd": 64922.2,
        "lower_1sd": 64871.02,
        "upper_2sd": 64947.79,
        "lower_2sd": 64845.43,
        "deviation_from_trend": 28.39,
        "position_in_channel": 0.7774
      },
      "dominant_cycles": {
        "dominant_cycles": [
          100,
          40,
          33.3
        ],
        "cycle_strengths": [
          3232.71,
          1969.6,
          1117.36
        ]
      },
      "fractal_dimension": 0.6187,
      "monte_carlo": {
        "median_price": 64925.85,
        "p10": 64867.73,
        "p25": 64896.75,
        "p75": 64959.78,
        "p90": 64990.78,
        "prob_up": 0.51,
        "horizon_bars": 12
      },
      "market_structure": {
        "structure": "BEARISH",
        "bos_detected": 0,
        "last_swing_high": 64947.7,
        "last_swing_low": 64909.55,
        "higher_highs": 0,
        "higher_lows": 0
      }
    },
    "profile_analysis": {
      "poor_extremes": {
        "poor_high": {
          "detected": 1,
          "price": 64924.97,
          "volume_ratio": 6.218,
          "implication": "High likely to be revisited - unfinished auction"
        },
        "poor_low": {
          "detected": 1,
          "price": 64924.96,
          "volume_ratio": 3.782,
          "implication": "Low likely to be revisited - unfinished auction"
        },
        "excess_high": 0,
        "excess_low": 0,
        "session_high": 64924.97,
        "session_low": 64924.96,
        "action_bias": "expect_retest_both",
        "status": "success"
      },
      "profile_shape": {
        "shape": "P",
        "implication": "Short covering rally - bearish bias expected",
        "trading_signal": "BEARISH_AFTER",
        "distribution": {
          "lower_third_pct": 37.8,
          "middle_third_pct": 0,
          "upper_third_pct": 62.2
        },
        "dominant_zone": "upper",
        "status": "success"
      },
      "no_mans_land": {
        "zones": [
          {
            "range_low": 64512,
            "range_high": 64810,
            "gap_size": 298,
            "gap_size_pct": 0.462,
            "risk": "LOW",
            "lvn_confirmed": 1,
            "lvns_in_zone": 9,
            "nearest_hvn_below": 64512,
            "nearest_hvn_above": 64810,
            "distance_from_price": 115,
            "distance_pct": 0.18,
            "direction": "below"
          }
        ],
        "price_in_no_mans_land": 0,
        "nearest_no_mans_land": {
          "range_low": 64512,
          "range_high": 64810,
          "gap_size": 298,
          "gap_size_pct": 0.462,
          "risk": "LOW",
          "lvn_confirmed": 1,
          "lvns_in_zone": 9,
          "nearest_hvn_below": 64512,
          "nearest_hvn_above": 64810,
          "distance_from_price": 115,
          "distance_pct": 0.18,
          "direction": "below"
        },
        "total_zones": 1,
        "max_gap_pct": 0.46,
        "status": "success"
      },
      "va_volume_pct": {
        "value_area_volume_pct": 100,
        "interpretation": "extremely_compressed",
        "breakout_risk": "VERY_HIGH",
        "volume_in_va": 1,
        "total_volume": 1,
        "compression_signal": 1
      },
      "volume_node_strength": {
        "scored_hvns": [
          {
            "price": 64354,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64304,
            "strength": 70,
            "volume_score": 15,
            "proximity_score": 25.2,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64306,
            "strength": 70,
            "volume_score": 15,
            "proximity_score": 25.2,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64314,
            "strength": 70,
            "volume_score": 15,
            "proximity_score": 25.3,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64320,
            "strength": 70,
            "volume_score": 15,
            "proximity_score": 25.3,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          }
        ],
        "scored_lvns": [
          {
            "price": 64480,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64485,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64500,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64516,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.9,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64348,
            "strength": 64,
            "volume_score": 15,
            "proximity_score": 25.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          }
        ],
        "total_hvns": 52,
        "total_lvns": 101,
        "avg_hvn_strength": 66.4,
        "avg_lvn_strength": 62.8,
        "status": "success"
      }
    },
    "flow_analysis": {
      "passive_aggressive": {
        "aggressive": {
          "buy_pct": 62.18,
          "sell_pct": 37.82,
          "net_pct": 24.36,
          "dominance": "buyers",
          "buy_volume": 0.093,
          "sell_volume": 0.057
        },
        "passive": {
          "dominance": "buyers",
          "inference": "from_orderbook_depth",
          "bid_depth": 1425768.55,
          "ask_depth": 1034391.61,
          "bid_ratio": 0.58,
          "ob_imbalance": 0.159
        },
        "composite": {
          "agreement": 1,
          "signal": "strong_bullish",
          "interpretation": "Both aggressive and passive buyers active - strong upward trend",
          "conviction": "HIGH"
        },
        "status": "success"
      },
      "whale_accumulation": {
        "score": 14,
        "classification": "NEUTRAL",
        "bias": "ACCUMULATING",
        "components": {
          "flow": {
            "score": -2.39,
            "max": 30,
            "detail": {
              "divergence": "aligned",
              "whale_delta": 0,
              "mid_delta": 0,
              "retail_delta": -0.478,
              "primary_delta": 0,
              "cvd_used": 1
            }
          },
          "depth": {
            "score": 3.18,
            "max": 20,
            "detail": {
              "bid_depth": 1425768.55,
              "ask_depth": 1034391.61,
              "ratio": 0.16,
              "deep_confirmation": 0
            }
          },
          "absorption": {
            "score": 11.4,
            "max": 25,
            "detail": {
              "buyer_strength": 6.2,
              "seller_exhaustion": 2.4,
              "net_absorption": 3.8,
              "index": 0.1185,
              "label": "Neutra"
            }
          },
          "derivatives": {
            "score": 1.49,
            "max": 25,
            "detail": {
              "long_short_ratio": 1.1,
              "lsr_score": 1.49
            }
          }
        },
        "trend": {
          "direction": "insufficient_data",
          "avg_score": 14,
          "samples": 2
        },
        "status": "success"
      },
      "absorption_zones": {
        "total_zones": 0,
        "total_events": 0,
        "buy_zone_count": 0,
        "sell_zone_count": 0,
        "status": "no_events"
      }
    },
    "sr_analysis": {
      "defense_zones": {
        "buy_defense": [
          {
            "center": 64481.04,
            "range_low": 64376.31,
            "range_high": 64570.69,
            "strength": 62,
            "side": "buy",
            "sources": [
              "sr_level_ema_21_4h",
              "vp_hvn",
              "ema_ema_21_4h",
              "vp_val"
            ],
            "source_count": 4,
            "signals_in_zone": 5,
            "type": "confluence",
            "distance_from_price": 443.96,
            "distance_pct": 0.68
          },
          {
            "center": 64841,
            "range_low": 64744.93,
            "range_high": 64928.69,
            "strength": 59,
            "side": "buy",
            "sources": [
              "vp_hvn",
              "ema_ema_21_1h",
              "sr_level_ema_21_1h",
              "orderbook_bid_wall"
            ],
            "source_count": 4,
            "signals_in_zone": 8,
            "type": "confluence",
            "distance_from_price": 84,
            "distance_pct": 0.13
          },
          {
            "center": 64165.68,
            "range_low": 64100.89,
            "range_high": 64230.48,
            "strength": 46,
            "side": "buy",
            "sources": [
              "ema_ema_21_1d",
              "sr_level_vah_monthly"
            ],
            "source_count": 2,
            "signals_in_zone": 2,
            "type": "cluster",
            "distance_from_price": 759.32,
            "distance_pct": 1.17
          },
          {
            "center": 64336.25,
            "range_low": 64255.31,
            "range_high": 64442.69,
            "strength": 40,
            "side": "buy",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 9,
            "type": "cluster",
            "distance_from_price": 588.75,
            "distance_pct": 0.91
          }
        ],
        "sell_defense": [
          {
            "center": 65037.38,
            "range_low": 64960.31,
            "range_high": 65128.69,
            "strength": 55,
            "side": "sell",
            "sources": [
              "sr_level_poc_daily",
              "vp_poc",
              "vp_hvn"
            ],
            "source_count": 3,
            "signals_in_zone": 9,
            "type": "confluence",
            "distance_from_price": 112.38,
            "distance_pct": 0.17
          },
          {
            "center": 65346,
            "range_low": 65297.31,
            "range_high": 65394.69,
            "strength": 53,
            "side": "sell",
            "sources": [
              "sr_level_vah_daily",
              "vp_hvn",
              "vp_vah"
            ],
            "source_count": 3,
            "signals_in_zone": 3,
            "type": "confluence",
            "distance_from_price": 421,
            "distance_pct": 0.65
          },
          {
            "center": 64945.09,
            "range_low": 64860.31,
            "range_high": 65052.69,
            "strength": 49,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "ema_ema_21_15m",
              "sr_level_hvn_daily"
            ],
            "source_count": 3,
            "signals_in_zone": 9,
            "type": "confluence",
            "distance_from_price": 20.09,
            "distance_pct": 0.03
          },
          {
            "center": 65157.17,
            "range_low": 65064.31,
            "range_high": 65257.69,
            "strength": 41,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 13,
            "type": "cluster",
            "distance_from_price": 232.17,
            "distance_pct": 0.36
          },
          {
            "center": 65252.3,
            "range_low": 65164.31,
            "range_high": 65350.69,
            "strength": 40,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 11,
            "type": "cluster",
            "distance_from_price": 327.3,
            "distance_pct": 0.5
          }
        ],
        "total_zones": 9,
        "strongest_buy": {
          "center": 64481.04,
          "range_low": 64376.31,
          "range_high": 64570.69,
          "strength": 62,
          "side": "buy",
          "sources": [
            "sr_level_ema_21_4h",
            "vp_hvn",
            "ema_ema_21_4h",
            "vp_val"
          ],
          "source_count": 4,
          "signals_in_zone": 5,
          "type": "confluence",
          "distance_from_price": 443.96,
          "distance_pct": 0.68
        },
        "strongest_sell": {
          "center": 65037.38,
          "range_low": 64960.31,
          "range_high": 65128.69,
          "strength": 55,
          "side": "sell",
          "sources": [
            "sr_level_poc_daily",
            "vp_poc",
            "vp_hvn"
          ],
          "source_count": 3,
          "signals_in_zone": 9,
          "type": "confluence",
          "distance_from_price": 112.38,
          "distance_pct": 0.17
        },
        "defense_asymmetry": {
          "ratio": 0.87,
          "bias": "slight_sell_defense",
          "description": "Slightly more sell defense",
          "buy_total_strength": 207,
          "sell_total_strength": 238
        },
        "status": "success"
      }
    },
    "quality": {
      "calendar": {
        "day_of_week": "Friday",
        "day_of_week_num": 4,
        "is_us_holiday": 0,
        "is_weekend": 0,
        "is_pre_holiday": 0,
        "is_post_holiday": 0,
        "expected_liquidity": "NORMAL",
        "liquidity_warning": 0
      },
      "latency": {
        "latency_ms": 6024,
        "latency_category": "POOR",
        "data_freshness": "DELAYED",
        "is_acceptable": 0,
        "is_stale": 0
      },
      "spread_percentile": {
        "status": "insufficient_data",
        "samples": 2,
        "min_required": 5,
        "current_spread": 0.1,
        "spread_percentile": 50,
        "liquidity_signal": "UNKNOWN"
      },
      "anomalies": {
        "anomalies_detected": 0,
        "count": 0,
        "max_severity": "NONE",
        "risk_elevated": 0,
        "summary": "No anomalies detected"
      }
    },
    "candlestick_patterns": {
      "patterns_detected": 1,
      "patterns": [
        {
          "name": "doji",
          "type": "neutral",
          "confidence": 0.65,
          "candles_used": 1,
          "implication": "Indecision - watch next candle for direction"
        }
      ],
      "dominant_signal": "neutral",
      "max_confidence": 0.65,
      "bullish_count": 0,
      "bearish_count": 0,
      "neutral_count": 1
    }
  },
  "sequence_id": 2,
  "primary_exchange": "BINANCE",
  "data_feed_type": "WEBSOCKET_L2",
  "data_quality_score": 10,
  "completeness_pct": 100,
  "reliability_score": 10,
  "bid": 64895.6,
  "ask": 64895.7,
  "tick_direction": 0,
  "twap": 64925,
  "pivot_points": {
    "daily": {
      "pivot": 64971.33,
      "r1": 65420.66,
      "r2": 65795.33,
      "r3": 66619.33,
      "s1": 64596.66,
      "s2": 64147.33,
      "s3": 63323.33,
      "vah": 65346,
      "val": 64522,
      "poc": 65046
    },
    "weekly": {
      "pivot": 62830.33,
      "r1": 63346.66,
      "r2": 64083.33,
      "r3": 65336.33,
      "s1": 62093.66,
      "s2": 61577.33,
      "s3": 60324.33,
      "vah": 63567,
      "val": 62314,
      "poc": 62610
    },
    "monthly": {
      "pivot": 63185.67,
      "r1": 64582.34,
      "r2": 65610.67,
      "r3": 68035.67,
      "s1": 62157.34,
      "s2": 60760.67,
      "s3": 58335.67,
      "vah": 64214,
      "val": 61789,
      "poc": 63554
    }
  },
  "immediate_support": [
    64841,
    64522,
    64481.04,
    64214,
    64165.68
  ],
  "support_strength": [
    59,
    93.8,
    62,
    71.2,
    46
  ],
  "immediate_resistance": [
    64945.09,
    64971.33,
    65037.38,
    65346
  ],
  "resistance_strength": [
    49,
    99.3,
    55,
    93.5
  ],
  "volatility_metrics": {
    "realized_vol_24h": 0.0184,
    "realized_vol_7d": 0.0487,
    "volatility_regime": "NORMAL"
  },
  "order_flow_extended": {
    "passive_buy_pct": 58,
    "passive_sell_pct": 42
  },
  "whale_activity": {
    "iceberg_activity": 0,
    "hidden_orders_detected": 0
  },
  "technical_indicators_extended": {
    "cci_1h": 159.32,
    "cci_signal": "OVERBOUGHT",
    "stochastic": {
      "k": 72.13,
      "d": 62.84,
      "signal": "NEUTRAL",
      "source": "real"
    },
    "williams_r": {
      "value": -59.5,
      "overbought": 0,
      "oversold": 0,
      "zone": "neutral",
      "source": "real"
    },
    "hurst_exponent": 0.362,
    "shannon_entropy": 2.9921,
    "fractal_dimension": 0.6187,
    "kalman_filter": {
      "kalman_price": 64943.38,
      "raw_price": 64925,
      "deviation_pct": -0.03,
      "trend_direction": "DOWN"
    },
    "regression_channel": {
      "slope_per_bar": -1.8405,
      "trend_price": 64896.61,
      "upper_1sd": 64922.2,
      "lower_1sd": 64871.02,
      "upper_2sd": 64947.79,
      "lower_2sd": 64845.43,
      "deviation_from_trend": 28.39,
      "position_in_channel": 0.7774
    },
    "dominant_cycles": {
      "dominant_cycles": [
        100,
        40,
        33.3
      ],
      "cycle_strengths": [
        3232.71,
        1969.6,
        1117.36
      ]
    },
    "monte_carlo": {
      "median_price": 64925.85,
      "p10": 64867.73,
      "p25": 64896.75,
      "p75": 64959.78,
      "p90": 64990.78,
      "prob_up": 0.51,
      "horizon_bars": 12
    },
    "garch_forecast_1h": 0.0024
  },
  "backup_exchanges": [
    "COINBASE",
    "KRAKEN",
    "OKX"
  ],
  "alerts": {
    "alert_count": 0,
    "max_severity": "NONE"
  },
  "regime_analysis": {
    "current_regime": "MEAN_REVERTING",
    "regime_probabilities": {
      "trending": 0,
      "mean_reverting": 1,
      "breakout": 0
    },
    "regime_change_probability": 0.05,
    "expected_regime_duration": "15m-1h",
    "avg_adx": 26.8
  },
  "data_reliability": {
    "has_options_data": 0,
    "onchain_coverage": "full",
    "latency_acceptable": 1,
    "price_targets_available": 1
  },
  "event_id": "871e4b9e",
  "timestamp_ny": "2026-08-07T18:32:00.000-04:00",
  "timestamp_sp": "2026-08-07T19:32:00.000-03:00",
  "_log_id": "871e4b9e"
}

----------------------------------------------------------------------------------------------------
 # Janela 3
----------------------------------------------------------------------------------------------------
EVENTO: ANALYSIS_TRIGGER | SYMBOL: BTCUSDT | JANELA: 3
UTC: 2026-08-07 22:33:00 UTC
NY:  2026-08-07 18:33:00 EST/EDT
SP:  2026-08-07 19:33:00 BRT
----------------------------------------------------------------------------------------------------
{
  "is_signal": 1,
  "tipo_evento": "ANALYSIS_TRIGGER",
  "descricao": "Evento automático para análise da IA",
  "symbol": "BTCUSDT",
  "raw_event": {
    "delta": -11.031,
    "volume_total": 19.309,
    "volume_compra": 4.139,
    "volume_venda": 15.17,
    "preco_fechamento": 64889.5,
    "advanced_analysis": {
      "symbol": "BTCUSDT",
      "price": 64889.5,
      "volume": 19.309,
      "adaptive_thresholds": {
        "current_volatility": 0.03,
        "volatility_factor": 2.0,
        "absorption_threshold": 0.3,
        "flow_threshold": 0.2
      },
      "onchain_metrics": {
        "difficulty": 126.23,
        "active_addresses": 717974,
        "mempool_size": 87009,
        "mempool_vsize_mb": 43.57,
        "mempool_total_fee_btc": 0.1575,
        "fees_fastest_sat_vb": 5,
        "fees_half_hour_sat_vb": 4,
        "fees_hour_sat_vb": 3,
        "fees_economy_sat_vb": 2,
        "difficulty_adjustment": {
          "progress_pct": 93.4,
          "estimated_change_pct": 0.75,
          "remaining_blocks": 133,
          "remaining_time_ms": 79249114,
          "previous_retarget_pct": -0.74
        },
        "minutes_between_blocks": 8.4423,
        "total_btc_sent_24h": 807619.08,
        "trade_volume_btc_24h": 3265.64,
        "data_source": "blockchain.info+mempool.space",
        "is_real_data": 1,
        "requires_paid_api": [
          "exchange_netflow",
          "whale_transactions",
          "exchange_reserves",
          "sopr"
        ]
      }
    }
  },
  "resultado_da_batalha": "N/A",
  "delta": -11.031,
  "volume_total": 19.309,
  "volume_compra": 4.139,
  "volume_venda": 15.17,
  "preco_fechamento": 64889.5,
  "timestamp": "2026-08-07T22:33:06Z",
  "epoch_ms": 1786141980000,
  "ml_features": {
    "price_features": {
      "returns_1": -1.5e-07,
      "volatility_1": 0.0,
      "returns_5": -1.5e-07,
      "volatility_5": 1.2e-07,
      "returns_15": 3.39e-06,
      "volatility_15": 8.5e-07,
      "momentum_score": 1.41421356,
      "volatility_1h": 0.0023
    },
    "volume_features": {
      "volume_sma_ratio": 0.637,
      "volume_momentum": -0.354,
      "buy_sell_pressure": -0.5713,
      "liquidity_gradient": -0.35379983
    },
    "microstructure": {
      "order_book_slope": -1.48711,
      "flow_imbalance": -0.5714,
      "tick_rule_sum": -239,
      "trade_intensity": 15,
      "trade_intensity_v2": 27.6833
    },
    "cross_asset": {
      "btc_eth_corr_7d": 0.821,
      "btc_eth_corr_30d": 0.8598,
      "btc_dxy_corr_30d": -0.0625,
      "btc_dxy_corr_90d": -0.0677,
      "btc_ndx_corr_30d": 0.4156,
      "dxy_return_5d": -0.3561,
      "dxy_return_20d": -1.6548,
      "btc_dxy_correlation_stability": 0.0052,
      "btc_dxy_inverse_strength": 0.0651,
      "dxy_momentum": -1.29867555,
      "vix_current": 14.9,
      "us10y_yield": 4.66,
      "btc_dominance": 17.0254,
      "btc_dominance_change_7d": 0,
      "eth_dominance": 9.1117,
      "gold_price": 4342.3475,
      "oil_price": 77.08,
      "macro_regime": "RISK_ON",
      "correlation_regime": "DECORRELATED"
    }
  },
  "orderbook_data": {
    "mid": 64863.25,
    "spread": 0.1,
    "spread_percent": 0,
    "bid_depth_usd": 1986941.37,
    "ask_depth_usd": 989392.9,
    "imbalance": 0.335,
    "flow_imbalance": 0.3352,
    "volume_ratio": 2.008,
    "pressure": 0.3352,
    "consolidated_bias_score": 0.7014,
    "spread_bps": 0.0154,
    "is_valid": 1,
    "data_source": "live",
    "spread_volatility": 0.0
  },
  "historical_vp": {
    "daily": {
      "poc": 65046,
      "vah": 65346,
      "val": 64522,
      "status": "success"
    },
    "weekly": {
      "poc": 62610,
      "vah": 63567,
      "val": 62314,
      "status": "success"
    },
    "monthly": {
      "poc": 63554,
      "vah": 64214,
      "val": 61789,
      "status": "success"
    }
  },
  "multi_tf": {
    "15m": {
      "tendencia": "Baixa",
      "preco_atual": 64903.32,
      "mme_21": 64928.25,
      "atr": 65.98,
      "regime": "Range",
      "rsi_short": 42.6,
      "rsi_long": 47.02,
      "macd": 15.1528,
      "macd_signal": 17.156,
      "adx": 25.05,
      "realized_vol": 0.0014
    },
    "1h": {
      "tendencia": "Alta",
      "preco_atual": 64903.32,
      "mme_21": 64791.65,
      "atr": 219.83,
      "regime": "Range",
      "rsi_short": 55.18,
      "rsi_long": 56.04,
      "macd": 102.8541,
      "macd_signal": 98.8004,
      "adx": 24.96,
      "realized_vol": 0.0023
    },
    "4h": {
      "tendencia": "Alta",
      "preco_atual": 64924.96,
      "mme_21": 64465.17,
      "atr": 491.56,
      "regime": "Range",
      "rsi_short": 64.6,
      "rsi_long": 62.72,
      "macd": 279.1724,
      "macd_signal": 257.5559,
      "adx": 30.01,
      "realized_vol": 0.0058
    },
    "1d": {
      "tendencia": "Alta",
      "preco_atual": 64924.96,
      "mme_21": 64149.58,
      "atr": 1369.69,
      "regime": "Manipulação",
      "rsi_short": 59.35,
      "rsi_long": 55.96,
      "macd": 71.3764,
      "macd_signal": 36.7677,
      "adx": 15.05,
      "realized_vol": 0.0184
    }
  },
  "data_context": "real_time",
  "price_targets": [
    {
      "level": 66836.18,
      "confidence": 0.3,
      "source": "fallback_r1",
      "weight": 0.1
    },
    {
      "level": 68782.87,
      "confidence": 0.2,
      "source": "fallback_r2",
      "weight": 0.05
    },
    {
      "level": 62942.81,
      "confidence": 0.3,
      "source": "fallback_s1",
      "weight": 0.1
    },
    {
      "level": 60996.13,
      "confidence": 0.2,
      "source": "fallback_s2",
      "weight": 0.05
    },
    {
      "level": 64889.5,
      "confidence": 0.4,
      "source": "fallback_poc",
      "weight": 0.15
    }
  ],
  "pattern_recognition": {
    "smart_money": {
      "market_structure": {
        "structure": "BEARISH",
        "bos_detected": 0,
        "last_swing_high": 64947.7,
        "last_swing_low": 64909.55,
        "higher_highs": 0,
        "higher_lows": 0
      }
    }
  },
  "janela_numero": 3,
  "timestamp_utc": "2026-08-07T22:33:00.000Z",
  "derivatives": {
    "BTCUSDT": {
      "funding_rate_percent": 0.01,
      "open_interest": 106941.874,
      "open_interest_usd": 6940097829.41,
      "long_short_ratio": 1.1,
      "longs_usd": 3634502292.66,
      "shorts_usd": 3305595536.75
    },
    "ETHUSDT": {
      "funding_rate_percent": 0,
      "open_interest": 2279582.358,
      "open_interest_usd": 4363895691.21,
      "long_short_ratio": 2.07,
      "longs_usd": 2940391094.65,
      "shorts_usd": 1423504596.56
    }
  },
  "fluxo_continuo": {
    "cvd": -11.3021,
    "whale_buy_volume": 0,
    "whale_sell_volume": 1,
    "whale_delta": -1,
    "bursts": {
      "count": 3,
      "max_burst_volume": 7.558
    },
    "sector_flow": {
      "retail": {
        "buy": 3.6105,
        "sell": 11.4429,
        "delta": -7.832
      },
      "mid": {
        "buy": 0.9922,
        "sell": 3.4619,
        "delta": -2.47
      },
      "whale": {
        "buy": 0,
        "sell": 1,
        "delta": -1
      }
    },
    "timestamp": "2026-08-07T22:33:00.000+00:00",
    "time_index": {
      "epoch_ms": 1786141980000,
      "timestamp_utc": "2026-08-07T22:33:00.000+00:00",
      "timestamp_ny": "2026-08-07T18:33:00.000-04:00",
      "timestamp_sp": "2026-08-07T19:33:00.000-03:00"
    },
    "order_flow": {
      "net_flow_1m": 238062.0028,
      "absorcao_1m": "Neutra",
      "buy_volume": 268590.45,
      "sell_volume": 984777.75,
      "total_volume": 1253368.2,
      "buy_volume_btc": 4.139,
      "sell_volume_btc": 15.17,
      "total_volume_btc": 19.309,
      "whale_buy_volume_window": 0,
      "whale_sell_volume_window": 1,
      "whale_delta_window": -1,
      "flow_imbalance": -0.5714,
      "aggressive_buy_pct": 21.43,
      "aggressive_sell_pct": 78.57,
      "net_flow_5m": -733790.4331,
      "absorcao_5m": "Neutra",
      "net_flow_15m": -733790.4331,
      "absorcao_15m": "Neutra",
      "computation_window_min": 1,
      "available_windows_min": [
        1,
        5,
        15
      ],
      "buy_sell_ratio": {
        "buy_sell_ratio": 0.27,
        "ratios": {
          "current": 0.2728,
          "imbalance_1m": 0.19,
          "imbalance_5m": -0.586,
          "imbalance_15m": -0.586
        },
        "sector_ratios": {
          "retail": 0.3155,
          "mid": 0.2866,
          "whale": 0
        },
        "pressure": "STRONG_SELL",
        "flow_trend": "short_term_reversal_to_buy",
        "buy_volume": 4.139,
        "sell_volume": 15.17
      }
    },
    "tipo_absorcao": "Neutra",
    "participant_analysis": {
      "retail": {
        "volume_pct": 71.75,
        "direction": "SELL",
        "sentiment": "BEARISH",
        "composite_score": -0.705,
        "imbalance": -0.546
      },
      "mid": {
        "volume_pct": 23.07,
        "direction": "SELL",
        "sentiment": "BEARISH",
        "composite_score": -0.316,
        "imbalance": -0.554
      },
      "whale": {
        "volume_pct": 5.18,
        "direction": "SELL",
        "sentiment": "BEARISH",
        "composite_score": -0.421,
        "imbalance": -1
      }
    },
    "liquidity_heatmap": {
      "clusters": [
        {
          "center": 64906.8539,
          "low": 64888,
          "high": 64924.97,
          "width": 36.97,
          "total_volume": 20.508,
          "buy_volume": 4.603,
          "sell_volume": 15.905,
          "imbalance": -11.302,
          "imbalance_ratio": -0.551,
          "trades_count": 1781,
          "avg_trade_size": 0.012,
          "recent_timestamp": 1786141985439,
          "recent_ts_ms": 1786141985439,
          "last_seen_ms": 1786141985439,
          "first_seen_ms": 1786141818682,
          "age_ms": 249.0,
          "cluster_duration_ms": 166757,
          "price_std": 15.3559,
          "volume_std": 0.063,
          "bin_threshold_usd": 194.7206
        }
      ],
      "resistances": [
        64906.8539
      ],
      "clusters_count": 1
    },
    "absorption_analysis": {
      "current_absorption": {
        "index": 0.1085,
        "classification": "WEAK_ABSORPTION",
        "label": "Neutra",
        "buyer_strength": 2.1,
        "seller_exhaustion": 5.7,
        "continuation_probability": 0.1,
        "delta_usd": 238062.003,
        "total_volume_usd": 1253368.2,
        "flow_imbalance": -0.5714,
        "window_min": 1
      }
    }
  },
  "market_context": {
    "trading_session": "NY",
    "session_phase": "ACTIVE",
    "time_to_session_close": 12573,
    "day_of_week": 4,
    "is_holiday": 0,
    "market_hours_type": "EXTENDED"
  },
  "market_environment": {
    "volatility_regime": "NORMAL",
    "trend_direction": "UP",
    "market_structure": "RANGE_BOUND",
    "liquidity_environment": "NORMAL",
    "risk_sentiment": "BULLISH",
    "correlation_spy": 0.406,
    "correlation_dxy": -0.0867,
    "correlation_gold": 0.2401
  },
  "contextual_snapshot": {
    "symbol": "BTCUSDT",
    "ohlc": {
      "open": 64925,
      "high": 64925,
      "low": 64888,
      "close": 64889.5,
      "open_time": 1786141920382,
      "close_time": 1786141978699,
      "vwap": 64909.4
    },
    "volume_total": 19.309,
    "volume_total_usdt": 1253368,
    "volume_compra": 4.139,
    "volume_venda": 15.17,
    "num_trades": 1533,
    "delta_minimo": -14.808,
    "delta_maximo": 0.056,
    "delta_fechamento": -11.031,
    "reversao_desde_minimo": 3.78,
    "reversao_desde_maximo": 11.09,
    "poc_price": 64924,
    "poc_volume": 4.85,
    "poc_percentage": 25.1,
    "dwell_price": 64888.9,
    "dwell_seconds": 33,
    "dwell_location": "Low",
    "trades_per_second": 26.29,
    "avg_trade_size": 0.013
  },
  "order_book_depth": {
    "L1": {
      "bids": 628978.45,
      "asks": 358953.5,
      "flow_imbalance": 0.2733
    },
    "L5": {
      "bids": 809686.51,
      "asks": 366218.25,
      "flow_imbalance": 0.3771
    },
    "L10": {
      "bids": 874094.46,
      "asks": 378801.94,
      "flow_imbalance": 0.3953
    },
    "L25": {
      "bids": 1153838.86,
      "asks": 579429.16,
      "flow_imbalance": 0.3314
    },
    "total_depth_ratio": 1.99
  },
  "orderbook_quality": "live",
  "market_impact": {
    "slippage_matrix": {
      "100k_usd": {
        "buy": 0.05,
        "sell": 0.05
      },
      "1m_usd": {
        "buy": 5.95,
        "sell": 2.35
      }
    },
    "liquidity_score": 9.9985,
    "execution_quality": "EXCELLENT"
  },
  "institutional_analytics": {
    "status": "ok",
    "computed_at_ms": 1786141986760,
    "technical_extras": {
      "stoch_rsi": {
        "k": 86.06,
        "d": 72.13,
        "overbought": 1,
        "oversold": 0,
        "crossover": "none"
      },
      "williams_r": {
        "value": -41.52,
        "overbought": 0,
        "oversold": 0,
        "zone": "neutral",
        "source": "real"
      },
      "hurst_exponent": 0.3565,
      "shannon_entropy": 2.9668,
      "kalman_filter": {
        "kalman_price": 64942.82,
        "raw_price": 64925,
        "deviation_pct": -0.03,
        "trend_direction": "DOWN"
      },
      "regression_channel": {
        "slope_per_bar": -1.5658,
        "trend_price": 64900.44,
        "upper_1sd": 64923.2,
        "lower_1sd": 64877.68,
        "upper_2sd": 64945.96,
        "lower_2sd": 64854.92,
        "deviation_from_trend": 24.56,
        "position_in_channel": 0.7697
      },
      "dominant_cycles": {
        "dominant_cycles": [
          100,
          40,
          33.3
        ],
        "cycle_strengths": [
          3165.78,
          1936.09,
          1056.1
        ]
      },
      "fractal_dimension": 0.6197,
      "monte_carlo": {
        "median_price": 64892.04,
        "p10": 64834.57,
        "p25": 64863.26,
        "p75": 64925.58,
        "p90": 64956.23,
        "prob_up": 0.53,
        "horizon_bars": 12
      },
      "market_structure": {
        "structure": "BEARISH",
        "bos_detected": 0,
        "last_swing_high": 64947.7,
        "last_swing_low": 64909.55,
        "higher_highs": 0,
        "higher_lows": 0
      }
    },
    "profile_analysis": {
      "poor_extremes": {
        "poor_high": {
          "detected": 1,
          "price": 64924.97,
          "volume_ratio": 3.105,
          "implication": "High likely to be revisited - unfinished auction"
        },
        "poor_low": {
          "detected": 1,
          "price": 64888,
          "volume_ratio": 2.268,
          "implication": "Low likely to be revisited - unfinished auction"
        },
        "excess_high": 0,
        "excess_low": 0,
        "session_high": 64924.97,
        "session_low": 64888,
        "action_bias": "expect_retest_both",
        "status": "success"
      },
      "profile_shape": {
        "shape": "P",
        "implication": "Short covering rally - bearish bias expected",
        "trading_signal": "BEARISH_AFTER",
        "distribution": {
          "lower_third_pct": 30.8,
          "middle_third_pct": 17.8,
          "upper_third_pct": 51.4
        },
        "dominant_zone": "upper",
        "status": "success"
      },
      "no_mans_land": {
        "zones": [
          {
            "range_low": 64512,
            "range_high": 64810,
            "gap_size": 298,
            "gap_size_pct": 0.462,
            "risk": "LOW",
            "lvn_confirmed": 1,
            "lvns_in_zone": 9,
            "nearest_hvn_below": 64512,
            "nearest_hvn_above": 64810,
            "distance_from_price": 79.5,
            "distance_pct": 0.12,
            "direction": "below"
          }
        ],
        "price_in_no_mans_land": 0,
        "nearest_no_mans_land": {
          "range_low": 64512,
          "range_high": 64810,
          "gap_size": 298,
          "gap_size_pct": 0.462,
          "risk": "LOW",
          "lvn_confirmed": 1,
          "lvns_in_zone": 9,
          "nearest_hvn_below": 64512,
          "nearest_hvn_above": 64810,
          "distance_from_price": 79.5,
          "distance_pct": 0.12,
          "direction": "below"
        },
        "total_zones": 1,
        "max_gap_pct": 0.46,
        "status": "success"
      },
      "va_volume_pct": {
        "value_area_volume_pct": 100,
        "interpretation": "extremely_compressed",
        "breakout_risk": "VERY_HIGH",
        "volume_in_va": 1,
        "total_volume": 1,
        "compression_signal": 1
      },
      "volume_node_strength": {
        "scored_hvns": [
          {
            "price": 64306,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.5,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64314,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64320,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64322,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64354,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.9,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          }
        ],
        "scored_lvns": [
          {
            "price": 64436,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.5,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64440,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.5,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64447,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64448,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64450,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          }
        ],
        "total_hvns": 52,
        "total_lvns": 101,
        "avg_hvn_strength": 66.3,
        "avg_lvn_strength": 63,
        "status": "success"
      }
    },
    "flow_analysis": {
      "passive_aggressive": {
        "aggressive": {
          "buy_pct": 21.43,
          "sell_pct": 78.57,
          "net_pct": -57.14,
          "dominance": "sellers",
          "buy_volume": 4.139,
          "sell_volume": 15.17
        },
        "passive": {
          "dominance": "buyers",
          "inference": "from_orderbook_depth",
          "bid_depth": 1986941.37,
          "ask_depth": 989392.9,
          "bid_ratio": 0.67,
          "ob_imbalance": 0.335
        },
        "composite": {
          "agreement": 0,
          "signal": "sell_absorption",
          "interpretation": "Aggressive sellers hitting passive buy walls - potential reversal or breakdown",
          "conviction": "MEDIUM"
        },
        "status": "success"
      },
      "whale_accumulation": {
        "score": -13,
        "classification": "NEUTRAL",
        "bias": "DISTRIBUTING",
        "components": {
          "flow": {
            "score": -10,
            "max": 30,
            "detail": {
              "divergence": "aligned",
              "whale_delta": -1,
              "mid_delta": -2.47,
              "retail_delta": -7.832,
              "primary_delta": -1
            }
          },
          "depth": {
            "score": 6.7,
            "max": 20,
            "detail": {
              "bid_depth": 1986941.37,
              "ask_depth": 989392.9,
              "ratio": 0.34,
              "deep_confirmation": 0
            }
          },
          "absorption": {
            "score": -10.8,
            "max": 25,
            "detail": {
              "buyer_strength": 2.1,
              "seller_exhaustion": 5.7,
              "net_absorption": -3.6,
              "index": 0.1085,
              "label": "Neutra"
            }
          },
          "derivatives": {
            "score": 1.49,
            "max": 25,
            "detail": {
              "long_short_ratio": 1.1,
              "lsr_score": 1.49
            }
          }
        },
        "trend": {
          "direction": "stable",
          "avg_score": -6,
          "recent_avg": -6,
          "momentum": -7.0,
          "samples": 3,
          "score_range": {
            "min": -19,
            "max": 14
          }
        },
        "status": "success"
      },
      "absorption_zones": {
        "total_zones": 0,
        "total_events": 0,
        "buy_zone_count": 0,
        "sell_zone_count": 0,
        "status": "no_events"
      }
    },
    "sr_analysis": {
      "defense_zones": {
        "buy_defense": [
          {
            "center": 64481.04,
            "range_low": 64376.33,
            "range_high": 64570.67,
            "strength": 62,
            "side": "buy",
            "sources": [
              "sr_level_ema_21_4h",
              "vp_hvn",
              "ema_ema_21_4h",
              "vp_val"
            ],
            "source_count": 4,
            "signals_in_zone": 5,
            "type": "confluence",
            "distance_from_price": 408.46,
            "distance_pct": 0.63
          },
          {
            "center": 64836.28,
            "range_low": 64742.98,
            "range_high": 64928.67,
            "strength": 61,
            "side": "buy",
            "sources": [
              "vp_hvn",
              "ema_ema_21_1h",
              "sr_level_ema_21_1h",
              "orderbook_bid_wall"
            ],
            "source_count": 4,
            "signals_in_zone": 8,
            "type": "confluence",
            "distance_from_price": 53.22,
            "distance_pct": 0.08
          },
          {
            "center": 64165.68,
            "range_low": 64100.91,
            "range_high": 64230.46,
            "strength": 46,
            "side": "buy",
            "sources": [
              "ema_ema_21_1d",
              "sr_level_vah_monthly"
            ],
            "source_count": 2,
            "signals_in_zone": 2,
            "type": "cluster",
            "distance_from_price": 723.82,
            "distance_pct": 1.12
          },
          {
            "center": 64336.25,
            "range_low": 64255.33,
            "range_high": 64442.67,
            "strength": 40,
            "side": "buy",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 9,
            "type": "cluster",
            "distance_from_price": 553.25,
            "distance_pct": 0.85
          }
        ],
        "sell_defense": [
          {
            "center": 65037.38,
            "range_low": 64960.33,
            "range_high": 65128.67,
            "strength": 55,
            "side": "sell",
            "sources": [
              "sr_level_poc_daily",
              "vp_poc",
              "vp_hvn"
            ],
            "source_count": 3,
            "signals_in_zone": 9,
            "type": "confluence",
            "distance_from_price": 147.88,
            "distance_pct": 0.23
          },
          {
            "center": 65346,
            "range_low": 65297.33,
            "range_high": 65394.67,
            "strength": 53,
            "side": "sell",
            "sources": [
              "sr_level_vah_daily",
              "vp_hvn",
              "vp_vah"
            ],
            "source_count": 3,
            "signals_in_zone": 3,
            "type": "confluence",
            "distance_from_price": 456.5,
            "distance_pct": 0.7
          },
          {
            "center": 64944.85,
            "range_low": 64860.33,
            "range_high": 65052.67,
            "strength": 49,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "ema_ema_21_15m",
              "sr_level_hvn_daily"
            ],
            "source_count": 3,
            "signals_in_zone": 9,
            "type": "confluence",
            "distance_from_price": 55.35,
            "distance_pct": 0.09
          },
          {
            "center": 65157.17,
            "range_low": 65064.33,
            "range_high": 65257.67,
            "strength": 41,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 13,
            "type": "cluster",
            "distance_from_price": 267.67,
            "distance_pct": 0.41
          },
          {
            "center": 65252.3,
            "range_low": 65164.33,
            "range_high": 65350.67,
            "strength": 40,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 11,
            "type": "cluster",
            "distance_from_price": 362.8,
            "distance_pct": 0.56
          }
        ],
        "total_zones": 9,
        "strongest_buy": {
          "center": 64481.04,
          "range_low": 64376.33,
          "range_high": 64570.67,
          "strength": 62,
          "side": "buy",
          "sources": [
            "sr_level_ema_21_4h",
            "vp_hvn",
            "ema_ema_21_4h",
            "vp_val"
          ],
          "source_count": 4,
          "signals_in_zone": 5,
          "type": "confluence",
          "distance_from_price": 408.46,
          "distance_pct": 0.63
        },
        "strongest_sell": {
          "center": 65037.38,
          "range_low": 64960.33,
          "range_high": 65128.67,
          "strength": 55,
          "side": "sell",
          "sources": [
            "sr_level_poc_daily",
            "vp_poc",
            "vp_hvn"
          ],
          "source_count": 3,
          "signals_in_zone": 9,
          "type": "confluence",
          "distance_from_price": 147.88,
          "distance_pct": 0.23
        },
        "defense_asymmetry": {
          "ratio": 0.88,
          "bias": "slight_sell_defense",
          "description": "Slightly more sell defense",
          "buy_total_strength": 209,
          "sell_total_strength": 238
        },
        "status": "success"
      }
    },
    "quality": {
      "calendar": {
        "day_of_week": "Friday",
        "day_of_week_num": 4,
        "is_us_holiday": 0,
        "is_weekend": 0,
        "is_pre_holiday": 0,
        "is_post_holiday": 0,
        "expected_liquidity": "NORMAL",
        "liquidity_warning": 0
      },
      "latency": {
        "latency_ms": 6984,
        "latency_category": "POOR",
        "data_freshness": "DELAYED",
        "is_acceptable": 0,
        "is_stale": 0
      },
      "spread_percentile": {
        "status": "insufficient_data",
        "samples": 3,
        "min_required": 5,
        "current_spread": 0.1,
        "spread_percentile": 50,
        "liquidity_signal": "UNKNOWN"
      },
      "anomalies": {
        "anomalies_detected": 1,
        "count": 1,
        "anomalies": [
          {
            "type": "FLOW_EXTREME_IMBALANCE",
            "severity": "MEDIUM",
            "value": -0.5714,
            "direction": "SELL",
            "description": "Extreme flow imbalance: -57.14% toward sellers"
          }
        ],
        "max_severity": "MEDIUM",
        "risk_elevated": 0,
        "types_found": [
          "FLOW_EXTREME_IMBALANCE"
        ],
        "summary": "1 anomalies detected (max severity: MEDIUM)"
      }
    },
    "candlestick_patterns": {
      "patterns_detected": 1,
      "patterns": [
        {
          "name": "doji",
          "type": "neutral",
          "confidence": 0.65,
          "candles_used": 1,
          "implication": "Indecision - watch next candle for direction"
        }
      ],
      "dominant_signal": "neutral",
      "max_confidence": 0.65,
      "bullish_count": 0,
      "bearish_count": 0,
      "neutral_count": 1
    }
  },
  "sequence_id": 3,
  "primary_exchange": "BINANCE",
  "data_feed_type": "WEBSOCKET_L2",
  "data_quality_score": 9.75,
  "completeness_pct": 100,
  "reliability_score": 9.5,
  "bid": 64863.2,
  "ask": 64863.3,
  "tick_direction": -1,
  "twap": 64913.17,
  "pivot_points": {
    "daily": {
      "pivot": 64971.33,
      "r1": 65420.66,
      "r2": 65795.33,
      "r3": 66619.33,
      "s1": 64596.66,
      "s2": 64147.33,
      "s3": 63323.33,
      "vah": 65346,
      "val": 64522,
      "poc": 65046
    },
    "weekly": {
      "pivot": 62830.33,
      "r1": 63346.66,
      "r2": 64083.33,
      "r3": 65336.33,
      "s1": 62093.66,
      "s2": 61577.33,
      "s3": 60324.33,
      "vah": 63567,
      "val": 62314,
      "poc": 62610
    },
    "monthly": {
      "pivot": 63185.67,
      "r1": 64582.34,
      "r2": 65610.67,
      "r3": 68035.67,
      "s1": 62157.34,
      "s2": 60760.67,
      "s3": 58335.67,
      "vah": 64214,
      "val": 61789,
      "poc": 63554
    }
  },
  "immediate_support": [
    64836.28,
    64522,
    64481.04,
    64214,
    64165.68
  ],
  "support_strength": [
    61,
    94.3,
    62,
    71.7,
    46
  ],
  "immediate_resistance": [
    64944.85,
    64971.33,
    65037.38,
    65346
  ],
  "resistance_strength": [
    49,
    98.7,
    55,
    93
  ],
  "volatility_metrics": {
    "realized_vol_24h": 0.0184,
    "realized_vol_7d": 0.0487,
    "volatility_regime": "NORMAL"
  },
  "order_flow_extended": {
    "passive_buy_pct": 66.8,
    "passive_sell_pct": 33.2
  },
  "whale_activity": {
    "large_orders_1h": [
      {
        "size": 1,
        "price": 64896.34,
        "side": "SELL",
        "timestamp_ms": 1786141945725
      }
    ],
    "iceberg_activity": 0,
    "hidden_orders_detected": 0
  },
  "technical_indicators_extended": {
    "cci_1h": 135.46,
    "cci_signal": "OVERBOUGHT",
    "stochastic": {
      "k": 86.06,
      "d": 72.13,
      "signal": "OVERBOUGHT",
      "source": "real"
    },
    "williams_r": {
      "value": -41.52,
      "overbought": 0,
      "oversold": 0,
      "zone": "neutral",
      "source": "real"
    },
    "hurst_exponent": 0.3565,
    "shannon_entropy": 2.9668,
    "fractal_dimension": 0.6197,
    "kalman_filter": {
      "kalman_price": 64942.82,
      "raw_price": 64925,
      "deviation_pct": -0.03,
      "trend_direction": "DOWN"
    },
    "regression_channel": {
      "slope_per_bar": -1.5658,
      "trend_price": 64900.44,
      "upper_1sd": 64923.2,
      "lower_1sd": 64877.68,
      "upper_2sd": 64945.96,
      "lower_2sd": 64854.92,
      "deviation_from_trend": 24.56,
      "position_in_channel": 0.7697
    },
    "dominant_cycles": {
      "dominant_cycles": [
        100,
        40,
        33.3
      ],
      "cycle_strengths": [
        3165.78,
        1936.09,
        1056.1
      ]
    },
    "monte_carlo": {
      "median_price": 64892.04,
      "p10": 64834.57,
      "p25": 64863.26,
      "p75": 64925.58,
      "p90": 64956.23,
      "prob_up": 0.53,
      "horizon_bars": 12
    },
    "garch_forecast_1h": 0.0024
  },
  "backup_exchanges": [
    "COINBASE",
    "KRAKEN",
    "OKX"
  ],
  "alerts": {
    "active_alerts": [
      {
        "type": "SUPPORT_TEST",
        "level": 64836.28,
        "severity": "HIGH",
        "probability": 0.84,
        "action": "MONITOR_CLOSELY",
        "description": "Preço testando suporte em 64836.28 (dist: 0.08%)"
      },
      {
        "type": "RESISTANCE_TEST",
        "level": 64944.85,
        "severity": "HIGH",
        "probability": 0.83,
        "action": "MONITOR_CLOSELY",
        "description": "Preço testando resistência em 64944.85 (dist: 0.09%)"
      }
    ],
    "alert_count": 2,
    "max_severity": "HIGH"
  },
  "regime_analysis": {
    "current_regime": "MEAN_REVERTING",
    "regime_probabilities": {
      "trending": 0,
      "mean_reverting": 1,
      "breakout": 0
    },
    "regime_change_probability": 0.05,
    "expected_regime_duration": "15m-1h",
    "avg_adx": 26.7
  },
  "data_reliability": {
    "has_options_data": 0,
    "onchain_coverage": "full",
    "latency_acceptable": 1,
    "price_targets_available": 1
  },
  "event_id": "3dc6793e",
  "timestamp_ny": "2026-08-07T18:33:00.000-04:00",
  "timestamp_sp": "2026-08-07T19:33:00.000-03:00",
  "_log_id": "3dc6793e"
}
----------------------------------------------------------------------------------------------------
EVENTO: AI_ANALYSIS | SYMBOL: BTCUSDT
UTC: 2026-08-07 22:33:08 UTC
NY:  2026-08-07 18:33:08 EST/EDT
SP:  2026-08-07 19:33:08 BRT
----------------------------------------------------------------------------------------------------
{
  "tipo_evento": "AI_ANALYSIS",
  "symbol": "BTCUSDT",
  "timestamp_ms": 1786141980000,
  "anchor_price": 64889.5,
  "anchor_window_id": 3,
  "ai_result": {
    "sentiment": "neutral",
    "confidence": 0.65,
    "action": "wait",
    "rationale": "O preço está em queda nos 15 min, mas nos 1 h e 4 h a tendência ainda é de alta, indicando uma retração dentro de um movimento de alta. O fluxo está dominado po",
    "region_type": "retracement",
    "_is_fallback": 0,
    "_is_valid": 1
  },
  "ai_payload": {
    "symbol": "BTCUSDT",
    "epoch_ms": 1786141980000,
    "trigger": "AT",
    "price": {
      "c": 64890,
      "o": 64925,
      "h": 64925,
      "l": 64888,
      "vw": 64909,
      "sh": "P",
      "auc": "expect_retest_both",
      "ph": 1,
      "pl": 1,
      "brk_risk": "V_HI"
    },
    "regime": {
      "cs": "BULL",
      "cf": 0.8,
      "v": "NOR",
      "mode": "MR",
      "dom": "4h",
      "bull%": 90,
      "bear%": 10
    },
    "qual": {
      "lat": "POOR",
      "ms": 6984
    },
    "flow": {
      "d1": "+238K",
      "delta": -11.031,
      "vol": 19.309,
      "buy_pct": 21,
      "ti": 27.7,
      "trs": -239,
      "obs": -1.487,
      "sf_w": -1,
      "sf_r": -7.832,
      "d5": "-734K",
      "d15": "-734K",
      "cvd": -11.3,
      "imb": -0.57,
      "ab": 21,
      "bsr": 0.27,
      "pa": "sell_absor",
      "conv": "M",
      "abs_buy_str": 2.1,
      "abs_sell_exh": 5.7
    },
    "ob": {
      "b": "2.0M",
      "a": "989K",
      "imb": 0.34,
      "bias": "BUY",
      "t5": 0.38,
      "spread_pct": 0,
      "slip_b": 5,
      "slip_s": 5
    },
    "tf": {
      "15m": {
        "t": "DN",
        "rsi": 43,
        "macd": [
          15,
          17
        ],
        "adx": 25,
        "atr": 66,
        "r": "RNG"
      },
      "1h": {
        "t": "UP",
        "rsi": 55,
        "macd": [
          103,
          99
        ],
        "adx": 25,
        "atr": 220,
        "r": "RNG"
      },
      "4h": {
        "t": "UP",
        "rsi": 65,
        "macd": [
          279,
          258
        ],
        "adx": 30,
        "atr": 492,
        "r": "RNG"
      },
      "1d": {
        "t": "UP",
        "rsi": 59,
        "macd": [
          71,
          37
        ],
        "adx": 15,
        "atr": 1370,
        "r": "MNP"
      }
    },
    "sr": {
      "r1": [
        65037,
        55
      ],
      "r1_dist": 148,
      "r1_conf": 3,
      "r2": [
        65346,
        53
      ],
      "r2_dist": 456,
      "r2_conf": 3,
      "s1": [
        64481,
        62
      ],
      "s1_dist": 408,
      "s1_conf": 4,
      "s2": [
        64836,
        61
      ],
      "s2_dist": 53,
      "s2_conf": 4,
      "def_bias": "slight_sel"
    },
    "w": {
      "s": -13,
      "c": "N"
    },
    "ext": {
      "cci": "OB",
      "stoch": 86,
      "stoch_sig": "OB",
      "wr": -42,
      "garch": 0,
      "hurst": 0.36,
      "entropy": 2.97,
      "fd": 0.62,
      "kalman": {
        "kp": 64942.82,
        "dev": -0.0274,
        "dir": "DOWN"
      },
      "reg": {
        "sl": -1.5658,
        "pos": 0.7697,
        "dev": 24.56
      },
      "mc": {
        "pu": 0.526,
        "p10": 64834.57,
        "p90": 64956.23
      },
      "cycles": [
        100,
        40
      ],
      "smc": {
        "struct": "BEAR",
        "bos": 0
      }
    },
    "alerts": [
      {
        "type": "SUPPORT_TEST",
        "sev": "H",
        "lvl": 64836
      },
      {
        "type": "RESISTANCE_TEST",
        "sev": "H",
        "lvl": 64945
      }
    ],
    "ctx": {
      "ses": "NY",
      "poc": 65046,
      "val": 64522,
      "vah": 65346,
      "lsr": 1.1,
      "eth_lsr": 2.07,
      "oi": 107,
      "fr": 0.0063,
      "longs": "+3634.5M",
      "shorts": "+3305.6M",
      "eth7": 0.8,
      "dxy30": -0.06
    },
    "ofi": {
      "score": -0.571,
      "dir": "SELL",
      "src": "order_flow"
    },
    "vwap": {
      "dev": -0.031,
      "side": "below",
      "sig": "fair",
      "src": "ohlc"
    },
    "liq": [
      {
        "p": 64907,
        "side": "sell",
        "vol": 20.51
      }
    ],
    "cvd_div": {
      "det": 1,
      "type": "bearish_div",
      "src": "inferred"
    },
    "mr": {
      "score": 0.155,
      "sig": "stretched_bull",
      "src": "inferred"
    },
    "summary": {
      "flow": {
        "bias": "SELL",
        "type": "mixed",
        "actor": "retail",
        "conf": "M",
        "note": "Fluxo misto sem dominância clara. (varejo vendedor). — divergência 1m vs 5m detectada. [imbalance extremo de venda]",
        "reversal_signal": 1
      },
      "sr": {
        "nearest": "resistance",
        "compressed": 0,
        "conf_bias": "NEUTRAL",
        "note": "Resistência mais próxima em 65037 (força 55, confluência 3 fontes, dist 148 pts (0.7 ATR))",
        "r1_dist_atr": 0.67,
        "s1_dist_atr": 1.85
      },
      "regime": {
        "label": "Mean Reversion",
        "strategies": [
          "fade extremos",
          "comprar suporte",
          "aguardar absorção nos extremos"
        ],
        "avoid": [
          "perseguir momentum",
          "entrar no meio do range",
          "operar breakouts sem confirmação"
        ],
        "duration": "15m – 2h tipicamente",
        "note": "Regime Mean Reversion com consenso de alta (confiança alta: 80%). dominado pelo 4h. Favorece reversão para equilíbrio — não perseguir altas."
      },
      "institutional": {
        "auction_state": "Leilão incompleto em ambos os extremos",
        "whale_bias": "NEUTRAL",
        "profile_bias": "NEUTRAL",
        "unfinished": [
          "low",
          "high"
        ],
        "alignment": "NEUTRAL",
        "note": "Leilão incompleto em ambos os extremos. Extremo(s) incompleto(s): low e high — reteste esperado. Risco de breakout da Value Area muito alto."
      },
      "quality": {
        "reliable": 1,
        "confidence_cap": 1,
        "note": "Dados em tempo real sem anomalias. Análise com confiança plena."
      }
    },
    "tipo_evento": "ANALYSIS_TRIGGER",
    "descricao": "Evento automático para análise da IA"
  },
  "epoch_ms": 1786141988998,
  "event_id": "cdefe2ce",
  "data_context": "real_time",
  "timestamp_utc": "2026-08-07T22:33:08.998+00:00",
  "timestamp_ny": "2026-08-07T18:33:08.998-04:00",
  "timestamp_sp": "2026-08-07T19:33:08.998-03:00",
  "timestamp": "2026-08-07T22:33:08.998+00:00",
  "_log_id": "cdefe2ce"
}

----------------------------------------------------------------------------------------------------
 # Janela 4
----------------------------------------------------------------------------------------------------
EVENTO: ANALYSIS_TRIGGER | SYMBOL: BTCUSDT | JANELA: 4
UTC: 2026-08-07 22:34:00 UTC
NY:  2026-08-07 18:34:00 EST/EDT
SP:  2026-08-07 19:34:00 BRT
----------------------------------------------------------------------------------------------------
{
  "is_signal": 1,
  "tipo_evento": "ANALYSIS_TRIGGER",
  "descricao": "Evento automático para análise da IA",
  "symbol": "BTCUSDT",
  "raw_event": {
    "delta": 0.333,
    "volume_total": 0.444,
    "volume_compra": 0.388,
    "volume_venda": 0.056,
    "preco_fechamento": 64892.3,
    "advanced_analysis": {
      "symbol": "BTCUSDT",
      "price": 64892.3,
      "volume": 0.444,
      "adaptive_thresholds": {
        "current_volatility": 0.03,
        "volatility_factor": 2.0,
        "absorption_threshold": 0.3,
        "flow_threshold": 0.2
      },
      "onchain_metrics": {
        "difficulty": 126.23,
        "active_addresses": 717974,
        "mempool_size": 87009,
        "mempool_vsize_mb": 43.57,
        "mempool_total_fee_btc": 0.1575,
        "fees_fastest_sat_vb": 5,
        "fees_half_hour_sat_vb": 4,
        "fees_hour_sat_vb": 3,
        "fees_economy_sat_vb": 2,
        "difficulty_adjustment": {
          "progress_pct": 93.4,
          "estimated_change_pct": 0.75,
          "remaining_blocks": 133,
          "remaining_time_ms": 79249114,
          "previous_retarget_pct": -0.74
        },
        "minutes_between_blocks": 8.4423,
        "total_btc_sent_24h": 807619.08,
        "trade_volume_btc_24h": 3265.64,
        "data_source": "blockchain.info+mempool.space",
        "is_real_data": 1,
        "requires_paid_api": [
          "exchange_netflow",
          "whale_transactions",
          "exchange_reserves",
          "sopr"
        ]
      }
    }
  },
  "resultado_da_batalha": "N/A",
  "delta": 0.333,
  "volume_total": 0.444,
  "volume_compra": 0.388,
  "volume_venda": 0.056,
  "preco_fechamento": 64892.3,
  "timestamp": "2026-08-07T22:34:07Z",
  "epoch_ms": 1786142040000,
  "ml_features": {
    "price_features": {
      "returns_1": 0.0,
      "volatility_1": 0.0,
      "returns_5": 0.0,
      "volatility_5": 1e-07,
      "returns_15": 0.0,
      "volatility_15": 8e-08,
      "momentum_score": 0.0,
      "volatility_1h": 0.0023
    },
    "volume_features": {
      "volume_sma_ratio": 0.796,
      "volume_momentum": -0.015,
      "buy_sell_pressure": 0.7489,
      "liquidity_gradient": -0.01546369
    },
    "microstructure": {
      "order_book_slope": 1.951782,
      "flow_imbalance": 0.7488,
      "tick_rule_sum": 17,
      "trade_intensity": 20,
      "trade_intensity_v2": 13.6667
    },
    "cross_asset": {
      "btc_eth_corr_7d": 0.821,
      "btc_eth_corr_30d": 0.8598,
      "btc_dxy_corr_30d": -0.0625,
      "btc_dxy_corr_90d": -0.0677,
      "btc_ndx_corr_30d": 0.4156,
      "dxy_return_5d": -0.3561,
      "dxy_return_20d": -1.6548,
      "btc_dxy_correlation_stability": 0.0052,
      "btc_dxy_inverse_strength": 0.0651,
      "dxy_momentum": -1.29867555,
      "vix_current": 14.9,
      "us10y_yield": 4.66,
      "btc_dominance": 17.0254,
      "btc_dominance_change_7d": 0,
      "eth_dominance": 9.1117,
      "gold_price": 4342.3475,
      "oil_price": 77.08,
      "macro_regime": "RISK_ON",
      "correlation_regime": "DECORRELATED"
    }
  },
  "orderbook_data": {
    "mid": 64842.65,
    "spread": 0.1,
    "spread_percent": 0,
    "bid_depth_usd": 759424.97,
    "ask_depth_usd": 1481953.09,
    "imbalance": -0.322,
    "flow_imbalance": -0.3224,
    "volume_ratio": 0.512,
    "pressure": -0.3224,
    "consolidated_bias_score": 0.3545,
    "spread_bps": 0.0154,
    "is_valid": 1,
    "data_source": "live",
    "spread_volatility": 0.0
  },
  "historical_vp": {
    "daily": {
      "poc": 65046,
      "vah": 65346,
      "val": 64522,
      "status": "success"
    },
    "weekly": {
      "poc": 62610,
      "vah": 63567,
      "val": 62314,
      "status": "success"
    },
    "monthly": {
      "poc": 63554,
      "vah": 64214,
      "val": 61789,
      "status": "success"
    }
  },
  "multi_tf": {
    "15m": {
      "tendencia": "Baixa",
      "preco_atual": 64903.32,
      "mme_21": 64928.25,
      "atr": 65.98,
      "regime": "Range",
      "rsi_short": 42.6,
      "rsi_long": 47.02,
      "macd": 15.1528,
      "macd_signal": 17.156,
      "adx": 25.05,
      "realized_vol": 0.0014
    },
    "1h": {
      "tendencia": "Alta",
      "preco_atual": 64903.32,
      "mme_21": 64791.65,
      "atr": 219.83,
      "regime": "Range",
      "rsi_short": 55.18,
      "rsi_long": 56.04,
      "macd": 102.8541,
      "macd_signal": 98.8004,
      "adx": 24.96,
      "realized_vol": 0.0023
    },
    "4h": {
      "tendencia": "Alta",
      "preco_atual": 64924.96,
      "mme_21": 64465.17,
      "atr": 491.56,
      "regime": "Range",
      "rsi_short": 64.6,
      "rsi_long": 62.72,
      "macd": 279.1724,
      "macd_signal": 257.5559,
      "adx": 30.01,
      "realized_vol": 0.0058
    },
    "1d": {
      "tendencia": "Alta",
      "preco_atual": 64924.96,
      "mme_21": 64149.58,
      "atr": 1369.69,
      "regime": "Manipulação",
      "rsi_short": 59.35,
      "rsi_long": 55.96,
      "macd": 71.3764,
      "macd_signal": 36.7677,
      "adx": 15.05,
      "realized_vol": 0.0184
    }
  },
  "data_context": "real_time",
  "price_targets": [
    {
      "level": 66839.07,
      "confidence": 0.3,
      "source": "fallback_r1",
      "weight": 0.1
    },
    {
      "level": 68785.84,
      "confidence": 0.2,
      "source": "fallback_r2",
      "weight": 0.05
    },
    {
      "level": 62945.53,
      "confidence": 0.3,
      "source": "fallback_s1",
      "weight": 0.1
    },
    {
      "level": 60998.76,
      "confidence": 0.2,
      "source": "fallback_s2",
      "weight": 0.05
    },
    {
      "level": 64892.3,
      "confidence": 0.4,
      "source": "fallback_poc",
      "weight": 0.15
    }
  ],
  "pattern_recognition": {
    "smart_money": {
      "market_structure": {
        "structure": "BEARISH",
        "bos_detected": 0,
        "last_swing_high": 64947.7,
        "last_swing_low": 64909.55,
        "higher_highs": 0,
        "higher_lows": 0
      }
    }
  },
  "janela_numero": 4,
  "timestamp_utc": "2026-08-07T22:34:00.000Z",
  "derivatives": {
    "BTCUSDT": {
      "funding_rate_percent": 0.01,
      "open_interest": 106941.874,
      "open_interest_usd": 6940097829.41,
      "long_short_ratio": 1.1,
      "longs_usd": 3634502292.66,
      "shorts_usd": 3305595536.75
    },
    "ETHUSDT": {
      "funding_rate_percent": 0,
      "open_interest": 2279582.358,
      "open_interest_usd": 4363895691.21,
      "long_short_ratio": 2.07,
      "longs_usd": 2940391094.65,
      "shorts_usd": 1423504596.56
    }
  },
  "fluxo_continuo": {
    "cvd": -17.4227,
    "whale_buy_volume": 0,
    "whale_sell_volume": 2,
    "whale_delta": -2,
    "bursts": {
      "count": 4,
      "max_burst_volume": 7.558
    },
    "sector_flow": {
      "retail": {
        "buy": 3.7492,
        "sell": 16.2023,
        "delta": -12.453
      },
      "mid": {
        "buy": 0.9922,
        "sell": 3.9619,
        "delta": -2.97
      },
      "whale": {
        "buy": 0,
        "sell": 2,
        "delta": -2
      }
    },
    "timestamp": "2026-08-07T22:34:00.000+00:00",
    "time_index": {
      "epoch_ms": 1786142040000,
      "timestamp_utc": "2026-08-07T22:34:00.000+00:00",
      "timestamp_ny": "2026-08-07T18:34:00.000-04:00",
      "timestamp_sp": "2026-08-07T19:34:00.000-03:00"
    },
    "order_flow": {
      "net_flow_1m": -324036.0293,
      "absorcao_1m": "Neutra",
      "buy_volume": 25195.9,
      "sell_volume": 3618.4,
      "total_volume": 28814.3,
      "buy_volume_btc": 0.388,
      "sell_volume_btc": 0.056,
      "total_volume_btc": 0.444,
      "whale_buy_volume_window": 0,
      "whale_sell_volume_window": 0,
      "whale_delta_window": 0,
      "flow_imbalance": 0.7488,
      "aggressive_buy_pct": 87.44,
      "aggressive_sell_pct": 12.56,
      "net_flow_5m": -1130934.1376,
      "absorcao_5m": "Neutra",
      "net_flow_15m": -1130934.1376,
      "absorcao_15m": "Neutra",
      "computation_window_min": 1,
      "available_windows_min": [
        1,
        5,
        15
      ],
      "buy_sell_ratio": {
        "buy_sell_ratio": 6.96,
        "ratios": {
          "current": 6.9634,
          "imbalance_1m": -11.246,
          "imbalance_5m": -39.249,
          "imbalance_15m": -39.249
        },
        "sector_ratios": {
          "retail": 0.2314,
          "mid": 0.2504,
          "whale": 0
        },
        "pressure": "STRONG_BUY",
        "flow_trend": "accelerating_selling",
        "buy_volume": 0.388,
        "sell_volume": 0.056
      }
    },
    "tipo_absorcao": "Neutra",
    "participant_analysis": {
      "retail": {
        "volume_pct": 100,
        "direction": "BUY",
        "sentiment": "BULLISH",
        "composite_score": 0.757,
        "imbalance": 0.749
      }
    },
    "liquidity_heatmap": {
      "clusters": [
        {
          "center": 64895.5777,
          "low": 64872.67,
          "high": 64924.96,
          "width": 52.29,
          "total_volume": 21.51,
          "buy_volume": 4.421,
          "sell_volume": 17.089,
          "imbalance": -12.668,
          "imbalance_ratio": -0.589,
          "trades_count": 2000,
          "avg_trade_size": 0.011,
          "recent_timestamp": 1786142046947,
          "recent_ts_ms": 1786142046947,
          "last_seen_ms": 1786142046947,
          "first_seen_ms": 1786141934732,
          "age_ms": 287.0,
          "cluster_duration_ms": 112215,
          "price_std": 12.7631,
          "volume_std": 0.06,
          "bin_threshold_usd": 194.6867
        }
      ],
      "resistances": [
        64895.5777
      ],
      "clusters_count": 1
    },
    "absorption_analysis": {
      "current_absorption": {
        "index": 0.7488,
        "classification": "STRONG_ABSORPTION",
        "label": "Neutra",
        "buyer_strength": 8.7,
        "seller_exhaustion": 7.5,
        "continuation_probability": 0.67,
        "delta_usd": -324036.029,
        "total_volume_usd": 28814.3,
        "flow_imbalance": 0.7488,
        "window_min": 1
      }
    }
  },
  "market_context": {
    "trading_session": "NY",
    "session_phase": "ACTIVE",
    "time_to_session_close": 12573,
    "day_of_week": 4,
    "is_holiday": 0,
    "market_hours_type": "EXTENDED"
  },
  "market_environment": {
    "volatility_regime": "NORMAL",
    "trend_direction": "UP",
    "market_structure": "RANGE_BOUND",
    "liquidity_environment": "NORMAL",
    "risk_sentiment": "BULLISH",
    "correlation_spy": 0.406,
    "correlation_dxy": -0.0867,
    "correlation_gold": 0.2401
  },
  "contextual_snapshot": {
    "symbol": "BTCUSDT",
    "ohlc": {
      "open": 64889.5,
      "high": 64892.3,
      "low": 64889.5,
      "close": 64892.3,
      "open_time": 1786141980221,
      "close_time": 1786142038424,
      "vwap": 64891.2
    },
    "volume_total": 0.444,
    "volume_total_usdt": 28814,
    "volume_compra": 0.388,
    "volume_venda": 0.056,
    "num_trades": 173,
    "delta_minimo": 0.076,
    "delta_maximo": 0.333,
    "delta_fechamento": 0.333,
    "reversao_desde_minimo": 0.26,
    "reversao_desde_maximo": 0,
    "poc_price": 64892.2,
    "poc_volume": 0.22,
    "poc_percentage": 49.6,
    "dwell_price": 64892.2,
    "dwell_seconds": 58,
    "dwell_location": "High",
    "trades_per_second": 2.97,
    "avg_trade_size": 0.003
  },
  "order_book_depth": {
    "L1": {
      "bids": 384840.83,
      "asks": 498380.99,
      "flow_imbalance": -0.1286
    },
    "L5": {
      "bids": 581766.89,
      "asks": 663146.57,
      "flow_imbalance": -0.0654
    },
    "L10": {
      "bids": 584036.35,
      "asks": 691029.2,
      "flow_imbalance": -0.0839
    },
    "L25": {
      "bids": 608870.29,
      "asks": 1136058.14,
      "flow_imbalance": -0.3021
    },
    "total_depth_ratio": 0.54
  },
  "orderbook_quality": "live",
  "market_impact": {
    "slippage_matrix": {
      "100k_usd": {
        "buy": 0.05,
        "sell": 0.05
      },
      "1m_usd": {
        "buy": 2.25,
        "sell": 5.65
      }
    },
    "liquidity_score": 9.9985,
    "execution_quality": "EXCELLENT"
  },
  "institutional_analytics": {
    "status": "ok",
    "computed_at_ms": 1786142047897,
    "technical_extras": {
      "stoch_rsi": {
        "k": 66.67,
        "d": 74.95,
        "overbought": 0,
        "oversold": 0,
        "crossover": "bearish"
      },
      "williams_r": {
        "value": -95.95,
        "overbought": 0,
        "oversold": 1,
        "zone": "oversold",
        "source": "real"
      },
      "hurst_exponent": 0.3509,
      "shannon_entropy": 2.9894,
      "kalman_filter": {
        "kalman_price": 64941.17,
        "raw_price": 64889.5,
        "deviation_pct": -0.08,
        "trend_direction": "DOWN"
      },
      "regression_channel": {
        "slope_per_bar": -1.4927,
        "trend_price": 64899.7,
        "upper_1sd": 64921.78,
        "lower_1sd": 64877.62,
        "upper_2sd": 64943.87,
        "lower_2sd": 64855.54,
        "deviation_from_trend": -10.2,
        "position_in_channel": 0.3845
      },
      "dominant_cycles": {
        "dominant_cycles": [
          100,
          40,
          33.3
        ],
        "cycle_strengths": [
          3108.43,
          1899.12,
          996.49
        ]
      },
      "fractal_dimension": 0.6225,
      "monte_carlo": {
        "median_price": 64891.67,
        "p10": 64833.37,
        "p25": 64862.48,
        "p75": 64925.7,
        "p90": 64956.79,
        "prob_up": 0.49,
        "horizon_bars": 12
      },
      "market_structure": {
        "structure": "BEARISH",
        "bos_detected": 0,
        "last_swing_high": 64947.7,
        "last_swing_low": 64909.55,
        "higher_highs": 0,
        "higher_lows": 0
      }
    },
    "profile_analysis": {
      "poor_extremes": {
        "poor_high": {
          "detected": 1,
          "price": 64892.32,
          "volume_ratio": 4.961,
          "implication": "High likely to be revisited - unfinished auction"
        },
        "poor_low": {
          "detected": 1,
          "price": 64889.47,
          "volume_ratio": 2.891,
          "implication": "Low likely to be revisited - unfinished auction"
        },
        "excess_high": 0,
        "excess_low": 0,
        "session_high": 64892.32,
        "session_low": 64889.47,
        "action_bias": "expect_retest_both",
        "status": "success"
      },
      "profile_shape": {
        "shape": "P",
        "implication": "Short covering rally - bearish bias expected",
        "trading_signal": "BEARISH_AFTER",
        "distribution": {
          "lower_third_pct": 36.1,
          "middle_third_pct": 9.2,
          "upper_third_pct": 54.7
        },
        "dominant_zone": "upper",
        "status": "success"
      },
      "no_mans_land": {
        "zones": [
          {
            "range_low": 64512,
            "range_high": 64810,
            "gap_size": 298,
            "gap_size_pct": 0.462,
            "risk": "LOW",
            "lvn_confirmed": 1,
            "lvns_in_zone": 9,
            "nearest_hvn_below": 64512,
            "nearest_hvn_above": 64810,
            "distance_from_price": 82.3,
            "distance_pct": 0.13,
            "direction": "below"
          }
        ],
        "price_in_no_mans_land": 0,
        "nearest_no_mans_land": {
          "range_low": 64512,
          "range_high": 64810,
          "gap_size": 298,
          "gap_size_pct": 0.462,
          "risk": "LOW",
          "lvn_confirmed": 1,
          "lvns_in_zone": 9,
          "nearest_hvn_below": 64512,
          "nearest_hvn_above": 64810,
          "distance_from_price": 82.3,
          "distance_pct": 0.13,
          "direction": "below"
        },
        "total_zones": 1,
        "max_gap_pct": 0.46,
        "status": "success"
      },
      "va_volume_pct": {
        "value_area_volume_pct": 100,
        "interpretation": "extremely_compressed",
        "breakout_risk": "VERY_HIGH",
        "volume_in_va": 1,
        "total_volume": 1,
        "compression_signal": 1
      },
      "volume_node_strength": {
        "scored_hvns": [
          {
            "price": 64314,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.5,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64320,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64322,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64354,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.9,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64304,
            "strength": 70,
            "volume_score": 15,
            "proximity_score": 25.5,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          }
        ],
        "scored_lvns": [
          {
            "price": 64440,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.5,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64447,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64448,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64450,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64460,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          }
        ],
        "total_hvns": 52,
        "total_lvns": 101,
        "avg_hvn_strength": 66.4,
        "avg_lvn_strength": 63,
        "status": "success"
      }
    },
    "flow_analysis": {
      "passive_aggressive": {
        "aggressive": {
          "buy_pct": 87.44,
          "sell_pct": 12.56,
          "net_pct": 74.88,
          "dominance": "buyers",
          "buy_volume": 0.388,
          "sell_volume": 0.056
        },
        "passive": {
          "dominance": "sellers",
          "inference": "from_orderbook_depth",
          "bid_depth": 759424.97,
          "ask_depth": 1481953.09,
          "bid_ratio": 0.34,
          "ob_imbalance": -0.322
        },
        "composite": {
          "agreement": 0,
          "signal": "buy_absorption",
          "interpretation": "Aggressive buyers hitting passive sell walls - potential reversal or breakout",
          "conviction": "MEDIUM"
        },
        "status": "success"
      },
      "whale_accumulation": {
        "score": -21,
        "classification": "MILD_DISTRIBUTION",
        "bias": "DISTRIBUTING",
        "components": {
          "flow": {
            "score": -20,
            "max": 30,
            "detail": {
              "divergence": "aligned",
              "whale_delta": -2,
              "mid_delta": -2.97,
              "retail_delta": -12.453,
              "primary_delta": -2
            }
          },
          "depth": {
            "score": -6.45,
            "max": 20,
            "detail": {
              "bid_depth": 759424.97,
              "ask_depth": 1481953.09,
              "ratio": -0.32,
              "deep_confirmation": 0
            }
          },
          "absorption": {
            "score": 3.6,
            "max": 25,
            "detail": {
              "buyer_strength": 8.7,
              "seller_exhaustion": 7.5,
              "net_absorption": 1.2,
              "index": 0.7488,
              "label": "Neutra"
            }
          },
          "derivatives": {
            "score": 1.49,
            "max": 25,
            "detail": {
              "long_short_ratio": 1.1,
              "lsr_score": 1.49
            }
          }
        },
        "trend": {
          "direction": "stable",
          "avg_score": -9.8,
          "recent_avg": -9.8,
          "momentum": -11.2,
          "samples": 4,
          "score_range": {
            "min": -21,
            "max": 14
          }
        },
        "status": "success"
      },
      "absorption_zones": {
        "total_zones": 0,
        "total_events": 0,
        "buy_zone_count": 0,
        "sell_zone_count": 0,
        "status": "no_events"
      }
    },
    "sr_analysis": {
      "defense_zones": {
        "buy_defense": [
          {
            "center": 64481.04,
            "range_low": 64376.33,
            "range_high": 64570.67,
            "strength": 62,
            "side": "buy",
            "sources": [
              "sr_level_ema_21_4h",
              "vp_hvn",
              "ema_ema_21_4h",
              "vp_val"
            ],
            "source_count": 4,
            "signals_in_zone": 5,
            "type": "confluence",
            "distance_from_price": 411.26,
            "distance_pct": 0.63
          },
          {
            "center": 64837.94,
            "range_low": 64742.98,
            "range_high": 64928.67,
            "strength": 49,
            "side": "buy",
            "sources": [
              "vp_hvn",
              "ema_ema_21_1h",
              "sr_level_ema_21_1h"
            ],
            "source_count": 3,
            "signals_in_zone": 7,
            "type": "confluence",
            "distance_from_price": 54.36,
            "distance_pct": 0.08
          },
          {
            "center": 64165.68,
            "range_low": 64100.91,
            "range_high": 64230.46,
            "strength": 46,
            "side": "buy",
            "sources": [
              "ema_ema_21_1d",
              "sr_level_vah_monthly"
            ],
            "source_count": 2,
            "signals_in_zone": 2,
            "type": "cluster",
            "distance_from_price": 726.62,
            "distance_pct": 1.12
          },
          {
            "center": 64336.25,
            "range_low": 64255.33,
            "range_high": 64442.67,
            "strength": 40,
            "side": "buy",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 9,
            "type": "cluster",
            "distance_from_price": 556.05,
            "distance_pct": 0.86
          }
        ],
        "sell_defense": [
          {
            "center": 64946.08,
            "range_low": 64860.33,
            "range_high": 65052.67,
            "strength": 59,
            "side": "sell",
            "sources": [
              "orderbook_ask_wall",
              "vp_hvn",
              "ema_ema_21_15m",
              "sr_level_hvn_daily"
            ],
            "source_count": 4,
            "signals_in_zone": 10,
            "type": "confluence",
            "distance_from_price": 53.78,
            "distance_pct": 0.08
          },
          {
            "center": 65037.38,
            "range_low": 64960.33,
            "range_high": 65128.67,
            "strength": 55,
            "side": "sell",
            "sources": [
              "sr_level_poc_daily",
              "vp_poc",
              "vp_hvn"
            ],
            "source_count": 3,
            "signals_in_zone": 9,
            "type": "confluence",
            "distance_from_price": 145.08,
            "distance_pct": 0.22
          },
          {
            "center": 65346,
            "range_low": 65297.33,
            "range_high": 65394.67,
            "strength": 53,
            "side": "sell",
            "sources": [
              "sr_level_vah_daily",
              "vp_hvn",
              "vp_vah"
            ],
            "source_count": 3,
            "signals_in_zone": 3,
            "type": "confluence",
            "distance_from_price": 453.7,
            "distance_pct": 0.7
          },
          {
            "center": 65157.17,
            "range_low": 65064.33,
            "range_high": 65257.67,
            "strength": 41,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 13,
            "type": "cluster",
            "distance_from_price": 264.87,
            "distance_pct": 0.41
          },
          {
            "center": 65252.3,
            "range_low": 65164.33,
            "range_high": 65350.67,
            "strength": 40,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 11,
            "type": "cluster",
            "distance_from_price": 360,
            "distance_pct": 0.55
          }
        ],
        "total_zones": 9,
        "strongest_buy": {
          "center": 64481.04,
          "range_low": 64376.33,
          "range_high": 64570.67,
          "strength": 62,
          "side": "buy",
          "sources": [
            "sr_level_ema_21_4h",
            "vp_hvn",
            "ema_ema_21_4h",
            "vp_val"
          ],
          "source_count": 4,
          "signals_in_zone": 5,
          "type": "confluence",
          "distance_from_price": 411.26,
          "distance_pct": 0.63
        },
        "strongest_sell": {
          "center": 64946.08,
          "range_low": 64860.33,
          "range_high": 65052.67,
          "strength": 59,
          "side": "sell",
          "sources": [
            "orderbook_ask_wall",
            "vp_hvn",
            "ema_ema_21_15m",
            "sr_level_hvn_daily"
          ],
          "source_count": 4,
          "signals_in_zone": 10,
          "type": "confluence",
          "distance_from_price": 53.78,
          "distance_pct": 0.08
        },
        "defense_asymmetry": {
          "ratio": 0.79,
          "bias": "slight_sell_defense",
          "description": "Slightly more sell defense",
          "buy_total_strength": 197,
          "sell_total_strength": 248
        },
        "status": "success"
      }
    },
    "quality": {
      "calendar": {
        "day_of_week": "Friday",
        "day_of_week_num": 4,
        "is_us_holiday": 0,
        "is_weekend": 0,
        "is_pre_holiday": 0,
        "is_post_holiday": 0,
        "expected_liquidity": "NORMAL",
        "liquidity_warning": 0
      },
      "latency": {
        "latency_ms": 8123,
        "latency_category": "POOR",
        "data_freshness": "DELAYED",
        "is_acceptable": 0,
        "is_stale": 0
      },
      "spread_percentile": {
        "status": "insufficient_data",
        "samples": 4,
        "min_required": 5,
        "current_spread": 0.1,
        "spread_percentile": 50,
        "liquidity_signal": "UNKNOWN"
      },
      "anomalies": {
        "anomalies_detected": 1,
        "count": 1,
        "anomalies": [
          {
            "type": "FLOW_EXTREME_IMBALANCE",
            "severity": "HIGH",
            "value": 0.7488,
            "direction": "BUY",
            "description": "Extreme flow imbalance: 74.88% toward buyers"
          }
        ],
        "max_severity": "HIGH",
        "risk_elevated": 1,
        "types_found": [
          "FLOW_EXTREME_IMBALANCE"
        ],
        "summary": "1 anomalies detected (max severity: HIGH)"
      }
    }
  },
  "sequence_id": 4,
  "primary_exchange": "BINANCE",
  "data_feed_type": "WEBSOCKET_L2",
  "data_quality_score": 8.75,
  "completeness_pct": 100,
  "reliability_score": 7.5,
  "bid": 64842.6,
  "ask": 64842.7,
  "tick_direction": 1,
  "twap": 64907.95,
  "pivot_points": {
    "daily": {
      "pivot": 64971.33,
      "r1": 65420.66,
      "r2": 65795.33,
      "r3": 66619.33,
      "s1": 64596.66,
      "s2": 64147.33,
      "s3": 63323.33,
      "vah": 65346,
      "val": 64522,
      "poc": 65046
    },
    "weekly": {
      "pivot": 62830.33,
      "r1": 63346.66,
      "r2": 64083.33,
      "r3": 65336.33,
      "s1": 62093.66,
      "s2": 61577.33,
      "s3": 60324.33,
      "vah": 63567,
      "val": 62314,
      "poc": 62610
    },
    "monthly": {
      "pivot": 63185.67,
      "r1": 64582.34,
      "r2": 65610.67,
      "r3": 68035.67,
      "s1": 62157.34,
      "s2": 60760.67,
      "s3": 58335.67,
      "vah": 64214,
      "val": 61789,
      "poc": 63554
    }
  },
  "immediate_support": [
    64837.94,
    64522,
    64481.04,
    64214,
    64165.68
  ],
  "support_strength": [
    49,
    94.3,
    62,
    71.6,
    46
  ],
  "immediate_resistance": [
    64946.08,
    64971.33,
    65037.38,
    65346
  ],
  "resistance_strength": [
    59,
    98.8,
    55,
    93
  ],
  "volatility_metrics": {
    "realized_vol_24h": 0.0184,
    "realized_vol_7d": 0.0487,
    "volatility_regime": "NORMAL"
  },
  "order_flow_extended": {
    "passive_buy_pct": 33.9,
    "passive_sell_pct": 66.1
  },
  "whale_activity": {
    "large_orders_1h": [
      {
        "size": 1,
        "price": 64896.34,
        "side": "SELL",
        "timestamp_ms": 1786141945725
      }
    ],
    "iceberg_activity": 0,
    "hidden_orders_detected": 0
  },
  "technical_indicators_extended": {
    "cci_1h": 135.46,
    "cci_signal": "OVERBOUGHT",
    "stochastic": {
      "k": 66.67,
      "d": 74.95,
      "signal": "NEUTRAL",
      "source": "real"
    },
    "williams_r": {
      "value": -95.95,
      "overbought": 0,
      "oversold": 1,
      "zone": "oversold",
      "source": "real"
    },
    "hurst_exponent": 0.3509,
    "shannon_entropy": 2.9894,
    "fractal_dimension": 0.6225,
    "kalman_filter": {
      "kalman_price": 64941.17,
      "raw_price": 64889.5,
      "deviation_pct": -0.08,
      "trend_direction": "DOWN"
    },
    "regression_channel": {
      "slope_per_bar": -1.4927,
      "trend_price": 64899.7,
      "upper_1sd": 64921.78,
      "lower_1sd": 64877.62,
      "upper_2sd": 64943.87,
      "lower_2sd": 64855.54,
      "deviation_from_trend": -10.2,
      "position_in_channel": 0.3845
    },
    "dominant_cycles": {
      "dominant_cycles": [
        100,
        40,
        33.3
      ],
      "cycle_strengths": [
        3108.43,
        1899.12,
        996.49
      ]
    },
    "monte_carlo": {
      "median_price": 64891.67,
      "p10": 64833.37,
      "p25": 64862.48,
      "p75": 64925.7,
      "p90": 64956.79,
      "prob_up": 0.49,
      "horizon_bars": 12
    },
    "garch_forecast_1h": 0.0024
  },
  "backup_exchanges": [
    "COINBASE",
    "KRAKEN",
    "OKX"
  ],
  "alerts": {
    "alert_count": 0,
    "max_severity": "NONE"
  },
  "regime_analysis": {
    "current_regime": "MEAN_REVERTING",
    "regime_probabilities": {
      "trending": 0.25,
      "mean_reverting": 0.583,
      "breakout": 0.167
    },
    "regime_change_probability": 0.24,
    "expected_regime_duration": "15m-1h",
    "avg_adx": 26.7
  },
  "data_reliability": {
    "has_options_data": 0,
    "onchain_coverage": "full",
    "latency_acceptable": 1,
    "price_targets_available": 1
  },
  "event_id": "fd9eb5bd",
  "timestamp_ny": "2026-08-07T18:34:00.000-04:00",
  "timestamp_sp": "2026-08-07T19:34:00.000-03:00",
  "_log_id": "fd9eb5bd"
}

----------------------------------------------------------------------------------------------------
 # Janela 5
----------------------------------------------------------------------------------------------------
EVENTO: ANALYSIS_TRIGGER | SYMBOL: BTCUSDT | JANELA: 5
UTC: 2026-08-07 22:35:00 UTC
NY:  2026-08-07 18:35:00 EST/EDT
SP:  2026-08-07 19:35:00 BRT
----------------------------------------------------------------------------------------------------
{
  "is_signal": 1,
  "tipo_evento": "ANALYSIS_TRIGGER",
  "descricao": "Evento automático para análise da IA",
  "symbol": "BTCUSDT",
  "raw_event": {
    "delta": -5.711,
    "volume_total": 7.186,
    "volume_compra": 0.738,
    "volume_venda": 6.448,
    "preco_fechamento": 64882,
    "advanced_analysis": {
      "symbol": "BTCUSDT",
      "price": 64882,
      "volume": 7.186,
      "adaptive_thresholds": {
        "current_volatility": 0.03,
        "volatility_factor": 2.0,
        "absorption_threshold": 0.3,
        "flow_threshold": 0.2
      },
      "onchain_metrics": {
        "difficulty": 126.23,
        "active_addresses": 717974,
        "mempool_size": 87009,
        "mempool_vsize_mb": 43.57,
        "mempool_total_fee_btc": 0.1575,
        "fees_fastest_sat_vb": 5,
        "fees_half_hour_sat_vb": 4,
        "fees_hour_sat_vb": 3,
        "fees_economy_sat_vb": 2,
        "difficulty_adjustment": {
          "progress_pct": 93.4,
          "estimated_change_pct": 0.75,
          "remaining_blocks": 133,
          "remaining_time_ms": 79249114,
          "previous_retarget_pct": -0.74
        },
        "minutes_between_blocks": 8.4423,
        "total_btc_sent_24h": 807619.08,
        "trade_volume_btc_24h": 3265.64,
        "data_source": "blockchain.info+mempool.space",
        "is_real_data": 1,
        "requires_paid_api": [
          "exchange_netflow",
          "whale_transactions",
          "exchange_reserves",
          "sopr"
        ]
      }
    }
  },
  "resultado_da_batalha": "N/A",
  "delta": -5.711,
  "volume_total": 7.186,
  "volume_compra": 0.738,
  "volume_venda": 6.448,
  "preco_fechamento": 64882,
  "timestamp": "2026-08-07T22:35:07Z",
  "epoch_ms": 1786142100000,
  "ml_features": {
    "price_features": {
      "returns_1": -1.5e-07,
      "volatility_1": 0.0,
      "returns_5": -1.5e-07,
      "volatility_5": 1.2e-07,
      "returns_15": -1.5e-07,
      "volatility_15": 1e-07,
      "momentum_score": 0.0,
      "volatility_1h": 0.0023
    },
    "volume_features": {
      "volume_sma_ratio": 0.389,
      "volume_momentum": -0.766,
      "buy_sell_pressure": -0.7947,
      "liquidity_gradient": -0.76574287
    },
    "microstructure": {
      "order_book_slope": 4.126738,
      "flow_imbalance": -0.7947,
      "tick_rule_sum": -110,
      "trade_intensity": 20,
      "trade_intensity_v2": 18.8667
    },
    "cross_asset": {
      "btc_eth_corr_7d": 0.821,
      "btc_eth_corr_30d": 0.8598,
      "btc_dxy_corr_30d": -0.0625,
      "btc_dxy_corr_90d": -0.0677,
      "btc_ndx_corr_30d": 0.4156,
      "dxy_return_5d": -0.3561,
      "dxy_return_20d": -1.6548,
      "btc_dxy_correlation_stability": 0.0052,
      "btc_dxy_inverse_strength": 0.0651,
      "dxy_momentum": -1.29867555,
      "vix_current": 14.9,
      "us10y_yield": 4.66,
      "btc_dominance": 17.0254,
      "btc_dominance_change_7d": 0,
      "eth_dominance": 9.1117,
      "gold_price": 4342.3475,
      "oil_price": 77.08,
      "macro_regime": "RISK_ON",
      "correlation_regime": "DECORRELATED"
    }
  },
  "orderbook_data": {
    "mid": 64853.35,
    "spread": 0.1,
    "spread_percent": 0,
    "bid_depth_usd": 1851195.64,
    "ask_depth_usd": 2778618.98,
    "imbalance": -0.2,
    "flow_imbalance": -0.2003,
    "volume_ratio": 0.666,
    "pressure": -0.2003,
    "consolidated_bias_score": 0.4065,
    "spread_bps": 0.0154,
    "is_valid": 1,
    "data_source": "live",
    "spread_volatility": 0.0
  },
  "historical_vp": {
    "daily": {
      "poc": 65046,
      "vah": 65346,
      "val": 64522,
      "status": "success"
    },
    "weekly": {
      "poc": 62610,
      "vah": 63567,
      "val": 62314,
      "status": "success"
    },
    "monthly": {
      "poc": 63554,
      "vah": 64214,
      "val": 61789,
      "status": "success"
    }
  },
  "multi_tf": {
    "15m": {
      "tendencia": "Baixa",
      "preco_atual": 64872.68,
      "mme_21": 64925.47,
      "atr": 70.07,
      "regime": "Range",
      "rsi_short": 37.97,
      "rsi_long": 44.12,
      "macd": 12.7086,
      "macd_signal": 16.6672,
      "adx": 23.54,
      "realized_vol": 0.0014
    },
    "1h": {
      "tendencia": "Alta",
      "preco_atual": 64872.68,
      "mme_21": 64788.86,
      "atr": 223.59,
      "regime": "Range",
      "rsi_short": 52.66,
      "rsi_long": 54.39,
      "macd": 100.4098,
      "macd_signal": 98.3115,
      "adx": 24.35,
      "realized_vol": 0.0023
    },
    "4h": {
      "tendencia": "Alta",
      "preco_atual": 64924.96,
      "mme_21": 64465.17,
      "atr": 491.56,
      "regime": "Range",
      "rsi_short": 64.6,
      "rsi_long": 62.72,
      "macd": 279.1724,
      "macd_signal": 257.5559,
      "adx": 30.01,
      "realized_vol": 0.0058
    },
    "1d": {
      "tendencia": "Alta",
      "preco_atual": 64924.96,
      "mme_21": 64149.58,
      "atr": 1369.69,
      "regime": "Manipulação",
      "rsi_short": 59.35,
      "rsi_long": 55.96,
      "macd": 71.3764,
      "macd_signal": 36.7677,
      "adx": 15.05,
      "realized_vol": 0.0184
    }
  },
  "data_context": "real_time",
  "price_targets": [
    {
      "level": 66828.46,
      "confidence": 0.3,
      "source": "fallback_r1",
      "weight": 0.1
    },
    {
      "level": 68774.92,
      "confidence": 0.2,
      "source": "fallback_r2",
      "weight": 0.05
    },
    {
      "level": 62935.54,
      "confidence": 0.3,
      "source": "fallback_s1",
      "weight": 0.1
    },
    {
      "level": 60989.08,
      "confidence": 0.2,
      "source": "fallback_s2",
      "weight": 0.05
    },
    {
      "level": 64882,
      "confidence": 0.4,
      "source": "fallback_poc",
      "weight": 0.15
    }
  ],
  "pattern_recognition": {
    "smart_money": {
      "fair_value_gaps": [
        {
          "type": "BEARISH",
          "top": 64925,
          "bottom": 64892.3,
          "gap_size": 32.7,
          "gap_pct": 0.05
        }
      ],
      "market_structure": {
        "structure": "BEARISH",
        "bos_detected": 0,
        "last_swing_high": 64947.7,
        "last_swing_low": 64909.55,
        "higher_highs": 0,
        "higher_lows": 0
      }
    }
  },
  "janela_numero": 5,
  "timestamp_utc": "2026-08-07T22:35:00.000Z",
  "derivatives": {
    "BTCUSDT": {
      "funding_rate_percent": 0.01,
      "open_interest": 106941.874,
      "open_interest_usd": 6940097829.41,
      "long_short_ratio": 1.1,
      "longs_usd": 3634502292.66,
      "shorts_usd": 3305595536.75
    },
    "ETHUSDT": {
      "funding_rate_percent": 0,
      "open_interest": 2279582.358,
      "open_interest_usd": 4363895691.21,
      "long_short_ratio": 2.07,
      "longs_usd": 2940391094.65,
      "shorts_usd": 1423504596.56
    }
  },
  "fluxo_continuo": {
    "cvd": -16.872,
    "whale_buy_volume": 0,
    "whale_sell_volume": 2,
    "whale_delta": -2,
    "bursts": {
      "count": 4,
      "max_burst_volume": 7.558
    },
    "sector_flow": {
      "retail": {
        "buy": 4.5312,
        "sell": 16.4335,
        "delta": -11.902
      },
      "mid": {
        "buy": 0.9922,
        "sell": 3.9619,
        "delta": -2.97
      },
      "whale": {
        "buy": 0,
        "sell": 2,
        "delta": -2
      }
    },
    "timestamp": "2026-08-07T22:35:00.000+00:00",
    "time_index": {
      "epoch_ms": 1786142100000,
      "timestamp_utc": "2026-08-07T22:35:00.000+00:00",
      "timestamp_ny": "2026-08-07T18:35:00.000-04:00",
      "timestamp_sp": "2026-08-07T19:35:00.000-03:00"
    },
    "order_flow": {
      "net_flow_1m": 35702.285,
      "absorcao_1m": "Neutra",
      "buy_volume": 47867.43,
      "sell_volume": 418412.71,
      "total_volume": 466280.14,
      "buy_volume_btc": 0.738,
      "sell_volume_btc": 6.448,
      "total_volume_btc": 7.186,
      "whale_buy_volume_window": 0,
      "whale_sell_volume_window": 1,
      "whale_delta_window": -1,
      "flow_imbalance": -0.7947,
      "aggressive_buy_pct": 10.27,
      "aggressive_sell_pct": 89.73,
      "net_flow_5m": -1095205.2548,
      "absorcao_5m": "Neutra",
      "net_flow_15m": -1095205.2548,
      "absorcao_15m": "Neutra",
      "computation_window_min": 1,
      "available_windows_min": [
        1,
        5,
        15
      ],
      "buy_sell_ratio": {
        "buy_sell_ratio": 0.11,
        "ratios": {
          "current": 0.1144,
          "imbalance_1m": 0.077,
          "imbalance_5m": -2.349,
          "imbalance_15m": -2.349
        },
        "sector_ratios": {
          "retail": 0.2757,
          "mid": 0.2504,
          "whale": 0
        },
        "pressure": "STRONG_SELL",
        "flow_trend": "short_term_reversal_to_buy",
        "buy_volume": 0.738,
        "sell_volume": 6.448
      }
    },
    "tipo_absorcao": "Neutra",
    "participant_analysis": {
      "retail": {
        "volume_pct": 79.13,
        "direction": "SELL",
        "sentiment": "BEARISH",
        "composite_score": -0.813,
        "imbalance": -0.74
      },
      "mid": {
        "volume_pct": 6.96,
        "direction": "SELL",
        "sentiment": "BEARISH",
        "composite_score": -0.428,
        "imbalance": -1
      },
      "whale": {
        "volume_pct": 13.92,
        "direction": "SELL",
        "sentiment": "BEARISH",
        "composite_score": -0.456,
        "imbalance": -1
      }
    },
    "liquidity_heatmap": {
      "clusters": [
        {
          "center": 64886.1283,
          "low": 64872.67,
          "high": 64903.31,
          "width": 30.64,
          "total_volume": 14.4,
          "buy_volume": 5.093,
          "sell_volume": 9.307,
          "imbalance": -4.213,
          "imbalance_ratio": -0.293,
          "trades_count": 2000,
          "avg_trade_size": 0.007,
          "recent_timestamp": 1786142106296,
          "recent_ts_ms": 1786142106296,
          "last_seen_ms": 1786142106296,
          "first_seen_ms": 1786141945722,
          "age_ms": 289.0,
          "cluster_duration_ms": 160574,
          "price_std": 8.438,
          "volume_std": 0.049,
          "bin_threshold_usd": 194.6584
        }
      ],
      "resistances": [
        64886.1283
      ],
      "clusters_count": 1
    },
    "absorption_analysis": {
      "current_absorption": {
        "index": 0.0608,
        "classification": "NONE",
        "label": "Neutra",
        "buyer_strength": 1,
        "seller_exhaustion": 7.9,
        "continuation_probability": 0.05,
        "delta_usd": 35702.285,
        "total_volume_usd": 466280.14,
        "flow_imbalance": -0.7947,
        "window_min": 1
      }
    }
  },
  "market_context": {
    "trading_session": "NY",
    "session_phase": "ACTIVE",
    "time_to_session_close": 12573,
    "day_of_week": 4,
    "is_holiday": 0,
    "market_hours_type": "EXTENDED"
  },
  "market_environment": {
    "volatility_regime": "NORMAL",
    "trend_direction": "UP",
    "market_structure": "RANGE_BOUND",
    "liquidity_environment": "NORMAL",
    "risk_sentiment": "BULLISH",
    "correlation_spy": 0.406,
    "correlation_dxy": -0.0867,
    "correlation_gold": 0.2401
  },
  "contextual_snapshot": {
    "symbol": "BTCUSDT",
    "ohlc": {
      "open": 64892.3,
      "high": 64892.3,
      "low": 64872.7,
      "close": 64882,
      "open_time": 1786142040352,
      "close_time": 1786142099283,
      "vwap": 64885.1
    },
    "volume_total": 7.186,
    "volume_total_usdt": 466280,
    "volume_compra": 0.738,
    "volume_venda": 6.448,
    "num_trades": 1122,
    "delta_minimo": -6.366,
    "delta_maximo": 0,
    "delta_fechamento": -5.711,
    "reversao_desde_minimo": 0.66,
    "reversao_desde_maximo": 5.71,
    "poc_price": 64891.8,
    "poc_volume": 2.4,
    "poc_percentage": 33.4,
    "dwell_price": 64882,
    "dwell_seconds": 55,
    "dwell_location": "Mid",
    "trades_per_second": 19.04,
    "avg_trade_size": 0.006
  },
  "order_book_depth": {
    "L1": {
      "bids": 472521.14,
      "asks": 1274109.9,
      "flow_imbalance": -0.4589
    },
    "L5": {
      "bids": 513962.28,
      "asks": 1986590.47,
      "flow_imbalance": -0.5889
    },
    "L10": {
      "bids": 515453.88,
      "asks": 2016228.82,
      "flow_imbalance": -0.5928
    },
    "L25": {
      "bids": 603390.67,
      "asks": 2288100.54,
      "flow_imbalance": -0.5826
    },
    "total_depth_ratio": 0.26
  },
  "orderbook_quality": "live",
  "market_impact": {
    "slippage_matrix": {
      "100k_usd": {
        "buy": 0.05,
        "sell": 0.05
      },
      "1m_usd": {
        "buy": 0.05,
        "sell": 4.55
      }
    },
    "liquidity_score": 9.9985,
    "execution_quality": "EXCELLENT"
  },
  "institutional_analytics": {
    "status": "ok",
    "computed_at_ms": 1786142107263,
    "technical_extras": {
      "stoch_rsi": {
        "k": 37.52,
        "d": 63.41,
        "overbought": 0,
        "oversold": 0,
        "crossover": "none"
      },
      "williams_r": {
        "value": -88.38,
        "overbought": 0,
        "oversold": 1,
        "zone": "oversold",
        "source": "real"
      },
      "hurst_exponent": 0.3459,
      "shannon_entropy": 2.9762,
      "kalman_filter": {
        "kalman_price": 64939.66,
        "raw_price": 64892.3,
        "deviation_pct": -0.07,
        "trend_direction": "DOWN"
      },
      "regression_channel": {
        "slope_per_bar": -1.3796,
        "trend_price": 64899.83,
        "upper_1sd": 64920.57,
        "lower_1sd": 64879.08,
        "upper_2sd": 64941.32,
        "lower_2sd": 64858.33,
        "deviation_from_trend": -7.53,
        "position_in_channel": 0.4093
      },
      "dominant_cycles": {
        "dominant_cycles": [
          100,
          40,
          33.3
        ],
        "cycle_strengths": [
          3064.92,
          1864.31,
          946.12
        ]
      },
      "fractal_dimension": 0.6238,
      "monte_carlo": {
        "median_price": 64880.87,
        "p10": 64822.67,
        "p25": 64851.73,
        "p75": 64914.84,
        "p90": 64945.88,
        "prob_up": 0.49,
        "horizon_bars": 12
      },
      "fair_value_gaps": [
        {
          "type": "BEARISH",
          "top": 64925,
          "bottom": 64892.3,
          "gap_size": 32.7,
          "gap_pct": 0.05
        }
      ],
      "market_structure": {
        "structure": "BEARISH",
        "bos_detected": 0,
        "last_swing_high": 64947.7,
        "last_swing_low": 64909.55,
        "higher_highs": 0,
        "higher_lows": 0
      }
    },
    "profile_analysis": {
      "poor_extremes": {
        "poor_high": {
          "detected": 1,
          "price": 64892.32,
          "volume_ratio": 3.368,
          "implication": "High likely to be revisited - unfinished auction"
        },
        "poor_low": {
          "detected": 1,
          "price": 64872.67,
          "volume_ratio": 1.255,
          "implication": "Low likely to be revisited - unfinished auction"
        },
        "excess_high": 0,
        "excess_low": 0,
        "session_high": 64892.32,
        "session_low": 64872.67,
        "action_bias": "expect_retest_both",
        "status": "success"
      },
      "profile_shape": {
        "shape": "B",
        "implication": "Double distribution - breakout imminent",
        "trading_signal": "BREAKOUT_EXPECTED",
        "distribution": {
          "lower_third_pct": 25.9,
          "middle_third_pct": 20.3,
          "upper_third_pct": 53.8
        },
        "dominant_zone": "upper",
        "status": "success"
      },
      "no_mans_land": {
        "zones": [
          {
            "range_low": 64512,
            "range_high": 64810,
            "gap_size": 298,
            "gap_size_pct": 0.462,
            "risk": "LOW",
            "lvn_confirmed": 1,
            "lvns_in_zone": 9,
            "nearest_hvn_below": 64512,
            "nearest_hvn_above": 64810,
            "distance_from_price": 72,
            "distance_pct": 0.11,
            "direction": "below"
          }
        ],
        "price_in_no_mans_land": 0,
        "nearest_no_mans_land": {
          "range_low": 64512,
          "range_high": 64810,
          "gap_size": 298,
          "gap_size_pct": 0.462,
          "risk": "LOW",
          "lvn_confirmed": 1,
          "lvns_in_zone": 9,
          "nearest_hvn_below": 64512,
          "nearest_hvn_above": 64810,
          "distance_from_price": 72,
          "distance_pct": 0.11,
          "direction": "below"
        },
        "total_zones": 1,
        "max_gap_pct": 0.46,
        "status": "success"
      },
      "va_volume_pct": {
        "value_area_volume_pct": 100,
        "interpretation": "extremely_compressed",
        "breakout_risk": "VERY_HIGH",
        "volume_in_va": 1,
        "total_volume": 1,
        "compression_signal": 1
      },
      "volume_node_strength": {
        "scored_hvns": [
          {
            "price": 64304,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.5,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64306,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64314,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64320,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64322,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          }
        ],
        "scored_lvns": [
          {
            "price": 64436,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64440,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64447,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64448,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64450,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          }
        ],
        "total_hvns": 52,
        "total_lvns": 101,
        "avg_hvn_strength": 66.3,
        "avg_lvn_strength": 63,
        "status": "success"
      }
    },
    "flow_analysis": {
      "passive_aggressive": {
        "aggressive": {
          "buy_pct": 10.27,
          "sell_pct": 89.73,
          "net_pct": -79.46,
          "dominance": "sellers",
          "buy_volume": 0.738,
          "sell_volume": 6.448
        },
        "passive": {
          "dominance": "sellers",
          "inference": "from_orderbook_depth",
          "bid_depth": 1851195.64,
          "ask_depth": 2778618.98,
          "bid_ratio": 0.4,
          "ob_imbalance": -0.2
        },
        "composite": {
          "agreement": 1,
          "signal": "strong_bearish",
          "interpretation": "Both aggressive and passive sellers active - strong downward trend",
          "conviction": "HIGH"
        },
        "status": "success"
      },
      "whale_accumulation": {
        "score": -43,
        "classification": "MILD_DISTRIBUTION",
        "bias": "DISTRIBUTING",
        "components": {
          "flow": {
            "score": -20,
            "max": 30,
            "detail": {
              "divergence": "aligned",
              "whale_delta": -2,
              "mid_delta": -2.97,
              "retail_delta": -11.902,
              "primary_delta": -2
            }
          },
          "depth": {
            "score": -4.01,
            "max": 20,
            "detail": {
              "bid_depth": 1851195.64,
              "ask_depth": 2778618.98,
              "ratio": -0.2,
              "deep_confirmation": 0
            }
          },
          "absorption": {
            "score": -20.7,
            "max": 25,
            "detail": {
              "buyer_strength": 1,
              "seller_exhaustion": 7.9,
              "net_absorption": -6.9,
              "index": 0.0608,
              "label": "Neutra"
            }
          },
          "derivatives": {
            "score": 1.49,
            "max": 25,
            "detail": {
              "long_short_ratio": 1.1,
              "lsr_score": 1.49
            }
          }
        },
        "trend": {
          "direction": "stable",
          "avg_score": -16.4,
          "recent_avg": -16.4,
          "momentum": -26.6,
          "samples": 5,
          "score_range": {
            "min": -43,
            "max": 14
          }
        },
        "status": "success"
      },
      "absorption_zones": {
        "total_zones": 0,
        "total_events": 0,
        "buy_zone_count": 0,
        "sell_zone_count": 0,
        "status": "no_events"
      }
    },
    "sr_analysis": {
      "defense_zones": {
        "buy_defense": [
          {
            "center": 64481.04,
            "range_low": 64376.34,
            "range_high": 64570.66,
            "strength": 62,
            "side": "buy",
            "sources": [
              "sr_level_ema_21_4h",
              "vp_hvn",
              "ema_ema_21_4h",
              "vp_val"
            ],
            "source_count": 4,
            "signals_in_zone": 5,
            "type": "confluence",
            "distance_from_price": 400.96,
            "distance_pct": 0.62
          },
          {
            "center": 64837.48,
            "range_low": 64740.2,
            "range_high": 64928.66,
            "strength": 49,
            "side": "buy",
            "sources": [
              "vp_hvn",
              "ema_ema_21_1h",
              "sr_level_ema_21_1h"
            ],
            "source_count": 3,
            "signals_in_zone": 7,
            "type": "confluence",
            "distance_from_price": 44.52,
            "distance_pct": 0.07
          },
          {
            "center": 64165.68,
            "range_low": 64100.92,
            "range_high": 64230.45,
            "strength": 46,
            "side": "buy",
            "sources": [
              "ema_ema_21_1d",
              "sr_level_vah_monthly"
            ],
            "source_count": 2,
            "signals_in_zone": 2,
            "type": "cluster",
            "distance_from_price": 716.32,
            "distance_pct": 1.1
          },
          {
            "center": 64336.25,
            "range_low": 64255.34,
            "range_high": 64442.66,
            "strength": 40,
            "side": "buy",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 9,
            "type": "cluster",
            "distance_from_price": 545.75,
            "distance_pct": 0.84
          }
        ],
        "sell_defense": [
          {
            "center": 64944.74,
            "range_low": 64860.34,
            "range_high": 65052.66,
            "strength": 59,
            "side": "sell",
            "sources": [
              "orderbook_ask_wall",
              "vp_hvn",
              "ema_ema_21_15m",
              "sr_level_hvn_daily"
            ],
            "source_count": 4,
            "signals_in_zone": 10,
            "type": "confluence",
            "distance_from_price": 62.74,
            "distance_pct": 0.1
          },
          {
            "center": 65037.38,
            "range_low": 64960.34,
            "range_high": 65128.66,
            "strength": 55,
            "side": "sell",
            "sources": [
              "sr_level_poc_daily",
              "vp_poc",
              "vp_hvn"
            ],
            "source_count": 3,
            "signals_in_zone": 9,
            "type": "confluence",
            "distance_from_price": 155.38,
            "distance_pct": 0.24
          },
          {
            "center": 65346,
            "range_low": 65297.34,
            "range_high": 65394.66,
            "strength": 53,
            "side": "sell",
            "sources": [
              "sr_level_vah_daily",
              "vp_hvn",
              "vp_vah"
            ],
            "source_count": 3,
            "signals_in_zone": 3,
            "type": "confluence",
            "distance_from_price": 464,
            "distance_pct": 0.72
          },
          {
            "center": 65157.17,
            "range_low": 65064.34,
            "range_high": 65257.66,
            "strength": 41,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 13,
            "type": "cluster",
            "distance_from_price": 275.17,
            "distance_pct": 0.42
          },
          {
            "center": 65252.3,
            "range_low": 65164.34,
            "range_high": 65350.66,
            "strength": 40,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 11,
            "type": "cluster",
            "distance_from_price": 370.3,
            "distance_pct": 0.57
          }
        ],
        "total_zones": 9,
        "strongest_buy": {
          "center": 64481.04,
          "range_low": 64376.34,
          "range_high": 64570.66,
          "strength": 62,
          "side": "buy",
          "sources": [
            "sr_level_ema_21_4h",
            "vp_hvn",
            "ema_ema_21_4h",
            "vp_val"
          ],
          "source_count": 4,
          "signals_in_zone": 5,
          "type": "confluence",
          "distance_from_price": 400.96,
          "distance_pct": 0.62
        },
        "strongest_sell": {
          "center": 64944.74,
          "range_low": 64860.34,
          "range_high": 65052.66,
          "strength": 59,
          "side": "sell",
          "sources": [
            "orderbook_ask_wall",
            "vp_hvn",
            "ema_ema_21_15m",
            "sr_level_hvn_daily"
          ],
          "source_count": 4,
          "signals_in_zone": 10,
          "type": "confluence",
          "distance_from_price": 62.74,
          "distance_pct": 0.1
        },
        "defense_asymmetry": {
          "ratio": 0.79,
          "bias": "slight_sell_defense",
          "description": "Slightly more sell defense",
          "buy_total_strength": 197,
          "sell_total_strength": 248
        },
        "status": "success"
      }
    },
    "quality": {
      "calendar": {
        "day_of_week": "Friday",
        "day_of_week_num": 4,
        "is_us_holiday": 0,
        "is_weekend": 0,
        "is_pre_holiday": 0,
        "is_post_holiday": 0,
        "expected_liquidity": "NORMAL",
        "liquidity_warning": 0
      },
      "latency": {
        "latency_ms": 7488,
        "latency_category": "POOR",
        "data_freshness": "DELAYED",
        "is_acceptable": 0,
        "is_stale": 0
      },
      "spread_percentile": {
        "status": "ok",
        "current_spread": 0.1,
        "spread_percentile": 0,
        "spread_mean": 0.1,
        "spread_median": 0.1,
        "spread_std": 0,
        "spread_min": 0.1,
        "spread_max": 0.1,
        "spread_z_score": 0,
        "is_tight": 1,
        "is_wide": 0,
        "is_anomalous": 0,
        "liquidity_signal": "EXCELLENT",
        "samples": 5,
        "window_minutes": 1440
      },
      "anomalies": {
        "anomalies_detected": 1,
        "count": 1,
        "anomalies": [
          {
            "type": "FLOW_EXTREME_IMBALANCE",
            "severity": "HIGH",
            "value": -0.7947,
            "direction": "SELL",
            "description": "Extreme flow imbalance: -79.47% toward sellers"
          }
        ],
        "max_severity": "HIGH",
        "risk_elevated": 1,
        "types_found": [
          "FLOW_EXTREME_IMBALANCE"
        ],
        "summary": "1 anomalies detected (max severity: HIGH)"
      }
    }
  },
  "sequence_id": 5,
  "primary_exchange": "BINANCE",
  "data_feed_type": "WEBSOCKET_L2",
  "data_quality_score": 9,
  "completeness_pct": 100,
  "reliability_score": 8,
  "bid": 64853.3,
  "ask": 64853.4,
  "tick_direction": -1,
  "twap": 64902.76,
  "pivot_points": {
    "daily": {
      "pivot": 64971.33,
      "r1": 65420.66,
      "r2": 65795.33,
      "r3": 66619.33,
      "s1": 64596.66,
      "s2": 64147.33,
      "s3": 63323.33,
      "vah": 65346,
      "val": 64522,
      "poc": 65046
    },
    "weekly": {
      "pivot": 62830.33,
      "r1": 63346.66,
      "r2": 64083.33,
      "r3": 65336.33,
      "s1": 62093.66,
      "s2": 61577.33,
      "s3": 60324.33,
      "vah": 63567,
      "val": 62314,
      "poc": 62610
    },
    "monthly": {
      "pivot": 63185.67,
      "r1": 64582.34,
      "r2": 65610.67,
      "r3": 68035.67,
      "s1": 62157.34,
      "s2": 60760.67,
      "s3": 58335.67,
      "vah": 64214,
      "val": 61789,
      "poc": 63554
    }
  },
  "immediate_support": [
    64837.48,
    64522,
    64481.04,
    64214,
    64165.68
  ],
  "support_strength": [
    49,
    94.5,
    62,
    71.8,
    46
  ],
  "immediate_resistance": [
    64944.74,
    64971.33,
    65037.38,
    65346
  ],
  "resistance_strength": [
    59,
    98.6,
    55,
    92.8
  ],
  "volatility_metrics": {
    "realized_vol_24h": 0.0184,
    "realized_vol_7d": 0.0487,
    "volatility_regime": "NORMAL"
  },
  "order_flow_extended": {
    "passive_buy_pct": 40,
    "passive_sell_pct": 60
  },
  "whale_activity": {
    "large_orders_1h": [
      {
        "size": 1,
        "price": 64896.34,
        "side": "SELL",
        "timestamp_ms": 1786141945725
      },
      {
        "size": 1,
        "price": 64883.14,
        "side": "SELL",
        "timestamp_ms": 1786142044496
      }
    ],
    "iceberg_activity": 1,
    "hidden_orders_detected": 0
  },
  "technical_indicators_extended": {
    "cci_1h": 99.97,
    "cci_signal": "NEUTRAL",
    "stochastic": {
      "k": 37.52,
      "d": 63.41,
      "signal": "NEUTRAL",
      "source": "real"
    },
    "williams_r": {
      "value": -88.38,
      "overbought": 0,
      "oversold": 1,
      "zone": "oversold",
      "source": "real"
    },
    "hurst_exponent": 0.3459,
    "shannon_entropy": 2.9762,
    "fractal_dimension": 0.6238,
    "kalman_filter": {
      "kalman_price": 64939.66,
      "raw_price": 64892.3,
      "deviation_pct": -0.07,
      "trend_direction": "DOWN"
    },
    "regression_channel": {
      "slope_per_bar": -1.3796,
      "trend_price": 64899.83,
      "upper_1sd": 64920.57,
      "lower_1sd": 64879.08,
      "upper_2sd": 64941.32,
      "lower_2sd": 64858.33,
      "deviation_from_trend": -7.53,
      "position_in_channel": 0.4093
    },
    "dominant_cycles": {
      "dominant_cycles": [
        100,
        40,
        33.3
      ],
      "cycle_strengths": [
        3064.92,
        1864.31,
        946.12
      ]
    },
    "monte_carlo": {
      "median_price": 64880.87,
      "p10": 64822.67,
      "p25": 64851.73,
      "p75": 64914.84,
      "p90": 64945.88,
      "prob_up": 0.49,
      "horizon_bars": 12
    },
    "garch_forecast_1h": 0.0012
  },
  "backup_exchanges": [
    "COINBASE",
    "KRAKEN",
    "OKX"
  ],
  "alerts": {
    "active_alerts": [
      {
        "type": "SUPPORT_TEST",
        "level": 64837.48,
        "severity": "HIGH",
        "probability": 0.86,
        "action": "MONITOR_CLOSELY",
        "description": "Preço testando suporte em 64837.48 (dist: 0.07%)"
      },
      {
        "type": "RESISTANCE_TEST",
        "level": 64944.74,
        "severity": "HIGH",
        "probability": 0.81,
        "action": "MONITOR_CLOSELY",
        "description": "Preço testando resistência em 64944.74 (dist: 0.10%)"
      },
      {
        "type": "WHALE_DISTRIBUTION",
        "level": -43,
        "severity": "HIGH",
        "probability": 0.86,
        "action": "AVOID_LONG",
        "description": "Sinal de distribuição de whales (score=-43)"
      },
      {
        "type": "BREAKOUT_SIGNAL",
        "severity": "MEDIUM",
        "probability": 0.65,
        "action": "PREPARE_BREAKOUT",
        "description": "Perfil tipo B detectado — breakout iminente"
      }
    ],
    "alert_count": 4,
    "max_severity": "HIGH"
  },
  "regime_analysis": {
    "current_regime": "BREAKOUT",
    "regime_probabilities": {
      "trending": 0,
      "mean_reverting": 0.476,
      "breakout": 0.524
    },
    "regime_change_probability": 0.52,
    "expected_regime_duration": "5m-30m",
    "avg_adx": 26
  },
  "data_reliability": {
    "has_options_data": 0,
    "onchain_coverage": "full",
    "latency_acceptable": 1,
    "price_targets_available": 1
  },
  "event_id": "9692644e",
  "timestamp_ny": "2026-08-07T18:35:00.000-04:00",
  "timestamp_sp": "2026-08-07T19:35:00.000-03:00",
  "_log_id": "9692644e"
}

----------------------------------------------------------------------------------------------------
 # Janela 6
----------------------------------------------------------------------------------------------------
EVENTO: ANALYSIS_TRIGGER | SYMBOL: BTCUSDT | JANELA: 6
UTC: 2026-08-07 22:36:00 UTC
NY:  2026-08-07 18:36:00 EST/EDT
SP:  2026-08-07 19:36:00 BRT
----------------------------------------------------------------------------------------------------
{
  "is_signal": 1,
  "tipo_evento": "ANALYSIS_TRIGGER",
  "descricao": "Evento automático para análise da IA",
  "symbol": "BTCUSDT",
  "raw_event": {
    "delta": -0.759,
    "volume_total": 1.145,
    "volume_compra": 0.193,
    "volume_venda": 0.952,
    "preco_fechamento": 64872.9,
    "advanced_analysis": {
      "symbol": "BTCUSDT",
      "price": 64872.9,
      "volume": 1.145,
      "adaptive_thresholds": {
        "current_volatility": 0.03,
        "volatility_factor": 2.0,
        "absorption_threshold": 0.3,
        "flow_threshold": 0.2
      },
      "onchain_metrics": {
        "difficulty": 126.23,
        "active_addresses": 717974,
        "mempool_size": 87732,
        "mempool_vsize_mb": 43.83,
        "mempool_total_fee_btc": 0.1697,
        "fees_fastest_sat_vb": 5,
        "fees_half_hour_sat_vb": 4,
        "fees_hour_sat_vb": 3,
        "fees_economy_sat_vb": 2,
        "difficulty_adjustment": {
          "progress_pct": 93.4,
          "estimated_change_pct": 0.72,
          "remaining_blocks": 133,
          "remaining_time_ms": 79270394,
          "previous_retarget_pct": -0.74
        },
        "minutes_between_blocks": 8.4423,
        "total_btc_sent_24h": 807619.08,
        "trade_volume_btc_24h": 3265.64,
        "data_source": "blockchain.info+mempool.space",
        "is_real_data": 1,
        "requires_paid_api": [
          "exchange_netflow",
          "whale_transactions",
          "exchange_reserves",
          "sopr"
        ]
      }
    }
  },
  "resultado_da_batalha": "N/A",
  "delta": -0.759,
  "volume_total": 1.145,
  "volume_compra": 0.193,
  "volume_venda": 0.952,
  "preco_fechamento": 64872.9,
  "timestamp": "2026-08-07T22:36:23Z",
  "epoch_ms": 1786142160000,
  "ml_features": {
    "price_features": {
      "returns_1": -1.5e-07,
      "volatility_1": 0.0,
      "returns_5": -1.5e-07,
      "volatility_5": 1.2e-07,
      "returns_15": -1.5e-07,
      "volatility_15": 1.2e-07,
      "momentum_score": 0.0,
      "volatility_1h": 0.0023
    },
    "volume_features": {
      "volume_sma_ratio": 2.148,
      "volume_momentum": -0.594,
      "buy_sell_pressure": -0.6629,
      "liquidity_gradient": -0.59381177
    },
    "microstructure": {
      "order_book_slope": 5.666204,
      "flow_imbalance": -0.6629,
      "tick_rule_sum": -52,
      "trade_intensity": 20,
      "trade_intensity_v2": 9.0167
    },
    "cross_asset": {
      "btc_eth_corr_7d": 0.8212,
      "btc_eth_corr_30d": 0.8599,
      "btc_dxy_corr_30d": -0.0603,
      "btc_dxy_corr_90d": -0.0669,
      "btc_ndx_corr_30d": 0.4146,
      "dxy_return_5d": -0.3561,
      "dxy_return_20d": -1.6548,
      "btc_dxy_correlation_stability": 0.0066,
      "btc_dxy_inverse_strength": 0.0636,
      "dxy_momentum": -1.29867555,
      "vix_current": 14.9,
      "us10y_yield": 4.66,
      "btc_dominance": 17.0386,
      "btc_dominance_change_7d": 0,
      "eth_dominance": 9.0982,
      "gold_price": 4342.3475,
      "oil_price": 77.08,
      "macro_regime": "RISK_ON",
      "correlation_regime": "DECORRELATED"
    }
  },
  "orderbook_data": {
    "mid": 64842.65,
    "spread": 0.1,
    "spread_percent": 0,
    "bid_depth_usd": 483385.34,
    "ask_depth_usd": 4276334.39,
    "imbalance": -0.797,
    "flow_imbalance": -0.7969,
    "volume_ratio": 0.113,
    "pressure": -0.7969,
    "consolidated_bias_score": 0.1722,
    "spread_bps": 0.0154,
    "is_valid": 1,
    "data_source": "live",
    "spread_volatility": 0.0
  },
  "historical_vp": {
    "daily": {
      "poc": 65046,
      "vah": 65346,
      "val": 64522,
      "status": "success"
    },
    "weekly": {
      "poc": 62610,
      "vah": 63567,
      "val": 62314,
      "status": "success"
    },
    "monthly": {
      "poc": 63554,
      "vah": 64214,
      "val": 61789,
      "status": "success"
    }
  },
  "multi_tf": {
    "15m": {
      "tendencia": "Baixa",
      "preco_atual": 64872.68,
      "mme_21": 64925.47,
      "atr": 70.07,
      "regime": "Range",
      "rsi_short": 37.97,
      "rsi_long": 44.12,
      "macd": 12.7086,
      "macd_signal": 16.6672,
      "adx": 23.54,
      "realized_vol": 0.0014
    },
    "1h": {
      "tendencia": "Alta",
      "preco_atual": 64872.68,
      "mme_21": 64788.86,
      "atr": 223.59,
      "regime": "Range",
      "rsi_short": 52.66,
      "rsi_long": 54.39,
      "macd": 100.4098,
      "macd_signal": 98.3115,
      "adx": 24.35,
      "realized_vol": 0.0023
    },
    "4h": {
      "tendencia": "Alta",
      "preco_atual": 64872.89,
      "mme_21": 64460.43,
      "atr": 495.32,
      "regime": "Range",
      "rsi_short": 61.98,
      "rsi_long": 61.12,
      "macd": 275.0187,
      "macd_signal": 256.7251,
      "adx": 30.01,
      "realized_vol": 0.0058
    },
    "1d": {
      "tendencia": "Alta",
      "preco_atual": 64924.96,
      "mme_21": 64149.58,
      "atr": 1369.69,
      "regime": "Manipulação",
      "rsi_short": 59.35,
      "rsi_long": 55.96,
      "macd": 71.3764,
      "macd_signal": 36.7677,
      "adx": 15.05,
      "realized_vol": 0.0184
    }
  },
  "data_context": "real_time",
  "price_targets": [
    {
      "level": 66819.09,
      "confidence": 0.3,
      "source": "fallback_r1",
      "weight": 0.1
    },
    {
      "level": 68765.27,
      "confidence": 0.2,
      "source": "fallback_r2",
      "weight": 0.05
    },
    {
      "level": 62926.71,
      "confidence": 0.3,
      "source": "fallback_s1",
      "weight": 0.1
    },
    {
      "level": 60980.53,
      "confidence": 0.2,
      "source": "fallback_s2",
      "weight": 0.05
    },
    {
      "level": 64872.9,
      "confidence": 0.4,
      "source": "fallback_poc",
      "weight": 0.15
    }
  ],
  "pattern_recognition": {
    "smart_money": {
      "fair_value_gaps": [
        {
          "type": "BEARISH",
          "top": 64925,
          "bottom": 64892.3,
          "gap_size": 32.7,
          "gap_pct": 0.05
        }
      ],
      "market_structure": {
        "structure": "BEARISH",
        "bos_detected": 0,
        "last_swing_high": 64947.7,
        "last_swing_low": 64909.55,
        "higher_highs": 0,
        "higher_lows": 0
      }
    }
  },
  "janela_numero": 6,
  "timestamp_utc": "2026-08-07T22:36:00.000Z",
  "derivatives": {
    "BTCUSDT": {
      "funding_rate_percent": 0.01,
      "open_interest": 106934.462,
      "open_interest_usd": 6934209575.49,
      "long_short_ratio": 1.1,
      "longs_usd": 3631418636.94,
      "shorts_usd": 3302790938.55
    },
    "ETHUSDT": {
      "funding_rate_percent": 0,
      "open_interest": 2279900.074,
      "open_interest_usd": 4362909179.41,
      "long_short_ratio": 2.07,
      "longs_usd": 2940607898.43,
      "shorts_usd": 1422301280.98
    }
  },
  "fluxo_continuo": {
    "cvd": -17.7173,
    "whale_buy_volume": 0,
    "whale_sell_volume": 2,
    "whale_delta": -2,
    "bursts": {
      "count": 4,
      "max_burst_volume": 7.558
    },
    "sector_flow": {
      "retail": {
        "buy": 4.6757,
        "sell": 17.4233,
        "delta": -12.748
      },
      "mid": {
        "buy": 0.9922,
        "sell": 3.9619,
        "delta": -2.97
      },
      "whale": {
        "buy": 0,
        "sell": 2,
        "delta": -2
      }
    },
    "timestamp": "2026-08-07T22:36:00.000+00:00",
    "time_index": {
      "epoch_ms": 1786142160000,
      "timestamp_utc": "2026-08-07T22:36:00.000+00:00",
      "timestamp_ny": "2026-08-07T18:36:00.000-04:00",
      "timestamp_sp": "2026-08-07T19:36:00.000-03:00"
    },
    "order_flow": {
      "net_flow_1m": -54836.8041,
      "absorcao_1m": "Neutra",
      "buy_volume": 12523.97,
      "sell_volume": 61780.58,
      "total_volume": 74304.54,
      "buy_volume_btc": 0.193,
      "sell_volume_btc": 0.952,
      "total_volume_btc": 1.145,
      "whale_buy_volume_window": 0,
      "whale_sell_volume_window": 0,
      "whale_delta_window": 0,
      "flow_imbalance": -0.6629,
      "aggressive_buy_pct": 16.85,
      "aggressive_sell_pct": 83.15,
      "net_flow_5m": -838082.1684,
      "absorcao_5m": "Neutra",
      "net_flow_15m": -1150050.4935,
      "absorcao_15m": "Neutra",
      "computation_window_min": 1,
      "available_windows_min": [
        1,
        5,
        15
      ],
      "buy_sell_ratio": {
        "buy_sell_ratio": 0.2,
        "ratios": {
          "current": 0.2027,
          "imbalance_1m": -0.738,
          "imbalance_5m": -11.279,
          "imbalance_15m": -15.477
        },
        "sector_ratios": {
          "retail": 0.2684,
          "mid": 0.2504,
          "whale": 0
        },
        "pressure": "STRONG_SELL",
        "flow_trend": "consistent_selling",
        "buy_volume": 0.193,
        "sell_volume": 0.952
      }
    },
    "tipo_absorcao": "Neutra",
    "participant_analysis": {
      "retail": {
        "volume_pct": 100,
        "direction": "SELL",
        "sentiment": "BEARISH",
        "composite_score": -0.842,
        "imbalance": -0.663
      }
    },
    "liquidity_heatmap": {
      "clusters": [
        {
          "center": 64882.0125,
          "low": 64872.67,
          "high": 64892.32,
          "width": 19.65,
          "total_volume": 9.324,
          "buy_volume": 1.806,
          "sell_volume": 7.518,
          "imbalance": -5.711,
          "imbalance_ratio": -0.613,
          "trades_count": 2000,
          "avg_trade_size": 0.005,
          "recent_timestamp": 1786142165327,
          "recent_ts_ms": 1786142165327,
          "last_seen_ms": 1786142165327,
          "first_seen_ms": 1786141968903,
          "age_ms": 281.0,
          "cluster_duration_ms": 196424,
          "price_std": 7.2396,
          "volume_std": 0.033,
          "bin_threshold_usd": 194.646
        }
      ],
      "resistances": [
        64882.0125
      ],
      "clusters_count": 1
    },
    "absorption_analysis": {
      "current_absorption": {
        "index": 0.4892,
        "classification": "MODERATE_ABSORPTION",
        "label": "Neutra",
        "buyer_strength": 1.7,
        "seller_exhaustion": 6.6,
        "continuation_probability": 0.44,
        "delta_usd": -54836.804,
        "total_volume_usd": 74304.54,
        "flow_imbalance": -0.6629,
        "window_min": 1
      }
    }
  },
  "market_context": {
    "trading_session": "NY",
    "session_phase": "ACTIVE",
    "time_to_session_close": 12269,
    "day_of_week": 4,
    "is_holiday": 0,
    "market_hours_type": "EXTENDED"
  },
  "market_environment": {
    "volatility_regime": "NORMAL",
    "trend_direction": "UP",
    "market_structure": "RANGE_BOUND",
    "liquidity_environment": "NORMAL",
    "risk_sentiment": "BULLISH",
    "correlation_spy": 0.406,
    "correlation_dxy": -0.0867,
    "correlation_gold": 0.2401
  },
  "external_markets": {
    "FEAR_GREED": {
      "preco_atual": 29,
      "prev": 25,
      "movimento": "Alta",
      "classification": "Fear",
      "source": "alternative.me",
      "timestamp": "2026-08-07T19:30:17.741873"
    }
  },
  "contextual_snapshot": {
    "symbol": "BTCUSDT",
    "ohlc": {
      "open": 64882,
      "high": 64882,
      "low": 64872.9,
      "close": 64872.9,
      "open_time": 1786142100238,
      "close_time": 1786142159039,
      "vwap": 64879.5
    },
    "volume_total": 1.145,
    "volume_total_usdt": 74305,
    "volume_compra": 0.193,
    "volume_venda": 0.952,
    "num_trades": 532,
    "delta_minimo": -0.792,
    "delta_maximo": 0.092,
    "delta_fechamento": -0.759,
    "reversao_desde_minimo": 0.03,
    "reversao_desde_maximo": 0.85,
    "poc_price": 64881.8,
    "poc_volume": 0.65,
    "poc_percentage": 57.2,
    "dwell_price": 64873.1,
    "dwell_seconds": 43,
    "dwell_location": "Low",
    "trades_per_second": 9.05,
    "avg_trade_size": 0.002
  },
  "order_book_depth": {
    "L1": {
      "bids": 240501.2,
      "asks": 2781557.3,
      "flow_imbalance": -0.8408
    },
    "L5": {
      "bids": 250033.04,
      "asks": 3796737.03,
      "flow_imbalance": -0.8764
    },
    "L10": {
      "bids": 253145.45,
      "asks": 3818913.5,
      "flow_imbalance": -0.8757
    },
    "L25": {
      "bids": 261445.01,
      "asks": 4021423.3,
      "flow_imbalance": -0.8779
    },
    "total_depth_ratio": 0.07
  },
  "orderbook_quality": "live",
  "market_impact": {
    "slippage_matrix": {
      "100k_usd": {
        "buy": 0.05,
        "sell": 0.05
      },
      "1m_usd": {
        "buy": 0.05,
        "sell": 6.05
      }
    },
    "liquidity_score": 9.9985,
    "execution_quality": "EXCELLENT"
  },
  "institutional_analytics": {
    "status": "ok",
    "computed_at_ms": 1786142183775,
    "technical_extras": {
      "stoch_rsi": {
        "k": 4.19,
        "d": 36.13,
        "overbought": 0,
        "oversold": 1,
        "crossover": "none"
      },
      "williams_r": {
        "value": -82.22,
        "overbought": 0,
        "oversold": 1,
        "zone": "oversold",
        "source": "real"
      },
      "hurst_exponent": 0.3404,
      "shannon_entropy": 2.9572,
      "kalman_filter": {
        "kalman_price": 64937.88,
        "raw_price": 64882,
        "deviation_pct": -0.09,
        "trend_direction": "DOWN"
      },
      "regression_channel": {
        "slope_per_bar": -1.28,
        "trend_price": 64899.43,
        "upper_1sd": 64918.61,
        "lower_1sd": 64880.25,
        "upper_2sd": 64937.79,
        "lower_2sd": 64861.07,
        "deviation_from_trend": -17.43,
        "position_in_channel": 0.2728
      },
      "dominant_cycles": {
        "dominant_cycles": [
          100,
          40,
          33.3
        ],
        "cycle_strengths": [
          3040.54,
          1840.75,
          915.41
        ]
      },
      "fractal_dimension": 0.6244,
      "monte_carlo": {
        "median_price": 64869.02,
        "p10": 64811.82,
        "p25": 64840.38,
        "p75": 64902.39,
        "p90": 64932.9,
        "prob_up": 0.47,
        "horizon_bars": 12
      },
      "fair_value_gaps": [
        {
          "type": "BEARISH",
          "top": 64925,
          "bottom": 64892.3,
          "gap_size": 32.7,
          "gap_pct": 0.05
        }
      ],
      "market_structure": {
        "structure": "BEARISH",
        "bos_detected": 0,
        "last_swing_high": 64947.7,
        "last_swing_low": 64909.55,
        "higher_highs": 0,
        "higher_lows": 0
      }
    },
    "profile_analysis": {
      "poor_extremes": {
        "poor_high": {
          "detected": 1,
          "price": 64882,
          "volume_ratio": 6.119,
          "implication": "High likely to be revisited - unfinished auction"
        },
        "poor_low": {
          "detected": 1,
          "price": 64872.89,
          "volume_ratio": 1.62,
          "implication": "Low likely to be revisited - unfinished auction"
        },
        "excess_high": 0,
        "excess_low": 0,
        "session_high": 64882,
        "session_low": 64872.89,
        "action_bias": "expect_retest_both",
        "status": "success"
      },
      "profile_shape": {
        "shape": "P",
        "implication": "Short covering rally - bearish bias expected",
        "trading_signal": "BEARISH_AFTER",
        "distribution": {
          "lower_third_pct": 19.5,
          "middle_third_pct": 13,
          "upper_third_pct": 67.5
        },
        "dominant_zone": "upper",
        "status": "success"
      },
      "no_mans_land": {
        "zones": [
          {
            "range_low": 64512,
            "range_high": 64810,
            "gap_size": 298,
            "gap_size_pct": 0.462,
            "risk": "LOW",
            "lvn_confirmed": 1,
            "lvns_in_zone": 9,
            "nearest_hvn_below": 64512,
            "nearest_hvn_above": 64810,
            "distance_from_price": 62.9,
            "distance_pct": 0.1,
            "direction": "below"
          }
        ],
        "price_in_no_mans_land": 0,
        "nearest_no_mans_land": {
          "range_low": 64512,
          "range_high": 64810,
          "gap_size": 298,
          "gap_size_pct": 0.462,
          "risk": "LOW",
          "lvn_confirmed": 1,
          "lvns_in_zone": 9,
          "nearest_hvn_below": 64512,
          "nearest_hvn_above": 64810,
          "distance_from_price": 62.9,
          "distance_pct": 0.1,
          "direction": "below"
        },
        "total_zones": 1,
        "max_gap_pct": 0.46,
        "status": "success"
      },
      "va_volume_pct": {
        "value_area_volume_pct": 100,
        "interpretation": "extremely_compressed",
        "breakout_risk": "VERY_HIGH",
        "volume_in_va": 1,
        "total_volume": 1,
        "compression_signal": 1
      },
      "volume_node_strength": {
        "scored_hvns": [
          {
            "price": 64304,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64306,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64314,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64320,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64322,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.8,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          }
        ],
        "scored_lvns": [
          {
            "price": 64423,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.5,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64436,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64440,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64447,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64448,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          }
        ],
        "total_hvns": 52,
        "total_lvns": 101,
        "avg_hvn_strength": 66.3,
        "avg_lvn_strength": 63.1,
        "status": "success"
      }
    },
    "flow_analysis": {
      "passive_aggressive": {
        "aggressive": {
          "buy_pct": 16.85,
          "sell_pct": 83.15,
          "net_pct": -66.3,
          "dominance": "sellers",
          "buy_volume": 0.193,
          "sell_volume": 0.952
        },
        "passive": {
          "dominance": "sellers",
          "inference": "from_orderbook_depth",
          "bid_depth": 483385.34,
          "ask_depth": 4276334.39,
          "bid_ratio": 0.1,
          "ob_imbalance": -0.797
        },
        "composite": {
          "agreement": 1,
          "signal": "strong_bearish",
          "interpretation": "Both aggressive and passive sellers active - strong downward trend",
          "conviction": "HIGH"
        },
        "status": "success"
      },
      "whale_accumulation": {
        "score": -49,
        "classification": "MILD_DISTRIBUTION",
        "bias": "DISTRIBUTING",
        "components": {
          "flow": {
            "score": -20,
            "max": 30,
            "detail": {
              "divergence": "aligned",
              "whale_delta": -2,
              "mid_delta": -2.97,
              "retail_delta": -12.748,
              "primary_delta": -2
            }
          },
          "depth": {
            "score": -15.94,
            "max": 20,
            "detail": {
              "bid_depth": 483385.34,
              "ask_depth": 4276334.39,
              "ratio": -0.8,
              "deep_confirmation": 0
            }
          },
          "absorption": {
            "score": -14.7,
            "max": 25,
            "detail": {
              "buyer_strength": 1.7,
              "seller_exhaustion": 6.6,
              "net_absorption": -4.9,
              "index": 0.4892,
              "label": "Neutra"
            }
          },
          "derivatives": {
            "score": 1.49,
            "max": 25,
            "detail": {
              "long_short_ratio": 1.1,
              "lsr_score": 1.49
            }
          }
        },
        "trend": {
          "direction": "stable",
          "avg_score": -21.8,
          "recent_avg": -22.4,
          "momentum": -27.2,
          "samples": 6,
          "score_range": {
            "min": -49,
            "max": 14
          }
        },
        "status": "success"
      },
      "absorption_zones": {
        "total_zones": 0,
        "total_events": 0,
        "buy_zone_count": 0,
        "sell_zone_count": 0,
        "status": "no_events"
      }
    },
    "sr_analysis": {
      "defense_zones": {
        "buy_defense": [
          {
            "center": 64479.86,
            "range_low": 64376.35,
            "range_high": 64570.65,
            "strength": 62,
            "side": "buy",
            "sources": [
              "sr_level_ema_21_4h",
              "vp_hvn",
              "ema_ema_21_4h",
              "vp_val"
            ],
            "source_count": 4,
            "signals_in_zone": 5,
            "type": "confluence",
            "distance_from_price": 393.04,
            "distance_pct": 0.61
          },
          {
            "center": 64837.48,
            "range_low": 64740.21,
            "range_high": 64928.65,
            "strength": 49,
            "side": "buy",
            "sources": [
              "vp_hvn",
              "ema_ema_21_1h",
              "sr_level_ema_21_1h"
            ],
            "source_count": 3,
            "signals_in_zone": 7,
            "type": "confluence",
            "distance_from_price": 35.42,
            "distance_pct": 0.05
          },
          {
            "center": 64165.68,
            "range_low": 64100.93,
            "range_high": 64230.44,
            "strength": 46,
            "side": "buy",
            "sources": [
              "ema_ema_21_1d",
              "sr_level_vah_monthly"
            ],
            "source_count": 2,
            "signals_in_zone": 2,
            "type": "cluster",
            "distance_from_price": 707.22,
            "distance_pct": 1.09
          },
          {
            "center": 64336.25,
            "range_low": 64255.35,
            "range_high": 64442.65,
            "strength": 40,
            "side": "buy",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 9,
            "type": "cluster",
            "distance_from_price": 536.65,
            "distance_pct": 0.83
          }
        ],
        "sell_defense": [
          {
            "center": 64943.83,
            "range_low": 64860.35,
            "range_high": 65052.65,
            "strength": 59,
            "side": "sell",
            "sources": [
              "orderbook_ask_wall",
              "vp_hvn",
              "ema_ema_21_15m",
              "sr_level_hvn_daily"
            ],
            "source_count": 4,
            "signals_in_zone": 10,
            "type": "confluence",
            "distance_from_price": 70.93,
            "distance_pct": 0.11
          },
          {
            "center": 65037.38,
            "range_low": 64960.35,
            "range_high": 65128.65,
            "strength": 55,
            "side": "sell",
            "sources": [
              "sr_level_poc_daily",
              "vp_poc",
              "vp_hvn"
            ],
            "source_count": 3,
            "signals_in_zone": 9,
            "type": "confluence",
            "distance_from_price": 164.48,
            "distance_pct": 0.25
          },
          {
            "center": 65346,
            "range_low": 65297.35,
            "range_high": 65394.65,
            "strength": 53,
            "side": "sell",
            "sources": [
              "sr_level_vah_daily",
              "vp_hvn",
              "vp_vah"
            ],
            "source_count": 3,
            "signals_in_zone": 3,
            "type": "confluence",
            "distance_from_price": 473.1,
            "distance_pct": 0.73
          },
          {
            "center": 65157.17,
            "range_low": 65064.35,
            "range_high": 65257.65,
            "strength": 41,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 13,
            "type": "cluster",
            "distance_from_price": 284.27,
            "distance_pct": 0.44
          },
          {
            "center": 65252.3,
            "range_low": 65164.35,
            "range_high": 65350.65,
            "strength": 40,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 11,
            "type": "cluster",
            "distance_from_price": 379.4,
            "distance_pct": 0.58
          }
        ],
        "total_zones": 9,
        "strongest_buy": {
          "center": 64479.86,
          "range_low": 64376.35,
          "range_high": 64570.65,
          "strength": 62,
          "side": "buy",
          "sources": [
            "sr_level_ema_21_4h",
            "vp_hvn",
            "ema_ema_21_4h",
            "vp_val"
          ],
          "source_count": 4,
          "signals_in_zone": 5,
          "type": "confluence",
          "distance_from_price": 393.04,
          "distance_pct": 0.61
        },
        "strongest_sell": {
          "center": 64943.83,
          "range_low": 64860.35,
          "range_high": 65052.65,
          "strength": 59,
          "side": "sell",
          "sources": [
            "orderbook_ask_wall",
            "vp_hvn",
            "ema_ema_21_15m",
            "sr_level_hvn_daily"
          ],
          "source_count": 4,
          "signals_in_zone": 10,
          "type": "confluence",
          "distance_from_price": 70.93,
          "distance_pct": 0.11
        },
        "defense_asymmetry": {
          "ratio": 0.79,
          "bias": "slight_sell_defense",
          "description": "Slightly more sell defense",
          "buy_total_strength": 197,
          "sell_total_strength": 248
        },
        "status": "success"
      }
    },
    "quality": {
      "calendar": {
        "day_of_week": "Friday",
        "day_of_week_num": 4,
        "is_us_holiday": 0,
        "is_weekend": 0,
        "is_pre_holiday": 0,
        "is_post_holiday": 0,
        "expected_liquidity": "NORMAL",
        "liquidity_warning": 0
      },
      "latency": {
        "latency_ms": 23980,
        "latency_category": "CRITICAL",
        "data_freshness": "STALE",
        "is_acceptable": 0,
        "is_stale": 1
      },
      "spread_percentile": {
        "status": "ok",
        "current_spread": 0.1,
        "spread_percentile": 0,
        "spread_mean": 0.1,
        "spread_median": 0.1,
        "spread_std": 0,
        "spread_min": 0.1,
        "spread_max": 0.1,
        "spread_z_score": -0.9129,
        "is_tight": 1,
        "is_wide": 0,
        "is_anomalous": 0,
        "liquidity_signal": "EXCELLENT",
        "samples": 6,
        "window_minutes": 1440
      },
      "anomalies": {
        "anomalies_detected": 1,
        "count": 2,
        "anomalies": [
          {
            "type": "FLOW_EXTREME_IMBALANCE",
            "severity": "MEDIUM",
            "value": -0.6629,
            "direction": "SELL",
            "description": "Extreme flow imbalance: -66.29% toward sellers"
          },
          {
            "type": "DEPTH_EXTREME_ASYMMETRY",
            "severity": "MEDIUM",
            "ratio": 0.11,
            "bid_depth": 483385.34,
            "ask_depth": 4276334.39,
            "direction": "ASK_HEAVY",
            "description": "Order book depth ratio 0.11:1 is extreme"
          }
        ],
        "max_severity": "MEDIUM",
        "risk_elevated": 0,
        "types_found": [
          "DEPTH_EXTREME_ASYMMETRY",
          "FLOW_EXTREME_IMBALANCE"
        ],
        "summary": "2 anomalies detected (max severity: MEDIUM)"
      }
    },
    "candlestick_patterns": {
      "patterns_detected": 1,
      "patterns": [
        {
          "name": "bearish_engulfing",
          "type": "bearish",
          "confidence": 0.78,
          "candles_used": 2,
          "implication": "Strong bearish reversal - sellers overwhelmed buyers"
        }
      ],
      "dominant_signal": "bearish",
      "max_confidence": 0.78,
      "bullish_count": 0,
      "bearish_count": 1,
      "neutral_count": 0
    }
  },
  "sequence_id": 6,
  "primary_exchange": "BINANCE",
  "data_feed_type": "WEBSOCKET_L2",
  "data_quality_score": 9.25,
  "completeness_pct": 100,
  "reliability_score": 8.5,
  "bid": 64842.6,
  "ask": 64842.7,
  "tick_direction": -1,
  "twap": 64897.78,
  "pivot_points": {
    "daily": {
      "pivot": 64971.33,
      "r1": 65420.66,
      "r2": 65795.33,
      "r3": 66619.33,
      "s1": 64596.66,
      "s2": 64147.33,
      "s3": 63323.33,
      "vah": 65346,
      "val": 64522,
      "poc": 65046
    },
    "weekly": {
      "pivot": 62830.33,
      "r1": 63346.66,
      "r2": 64083.33,
      "r3": 65336.33,
      "s1": 62093.66,
      "s2": 61577.33,
      "s3": 60324.33,
      "vah": 63567,
      "val": 62314,
      "poc": 62610
    },
    "monthly": {
      "pivot": 63185.67,
      "r1": 64582.34,
      "r2": 65610.67,
      "r3": 68035.67,
      "s1": 62157.34,
      "s2": 60760.67,
      "s3": 58335.67,
      "vah": 64214,
      "val": 61789,
      "poc": 63554
    }
  },
  "immediate_support": [
    64837.48,
    64522,
    64479.86,
    64214,
    64165.68
  ],
  "support_strength": [
    49,
    94.6,
    62,
    71.9,
    46
  ],
  "immediate_resistance": [
    64943.83,
    64971.33,
    65037.38,
    65346
  ],
  "resistance_strength": [
    59,
    98.5,
    55,
    92.7
  ],
  "volatility_metrics": {
    "realized_vol_24h": 0.0184,
    "realized_vol_7d": 0.0487,
    "volatility_regime": "NORMAL"
  },
  "order_flow_extended": {
    "passive_buy_pct": 10.2,
    "passive_sell_pct": 89.8
  },
  "whale_activity": {
    "large_orders_1h": [
      {
        "size": 1,
        "price": 64896.34,
        "side": "SELL",
        "timestamp_ms": 1786141945725
      },
      {
        "size": 1,
        "price": 64883.14,
        "side": "SELL",
        "timestamp_ms": 1786142044496
      }
    ],
    "iceberg_activity": 1,
    "hidden_orders_detected": 0
  },
  "technical_indicators_extended": {
    "cci_1h": 99.97,
    "cci_signal": "NEUTRAL",
    "stochastic": {
      "k": 4.19,
      "d": 36.13,
      "signal": "OVERSOLD",
      "source": "real"
    },
    "williams_r": {
      "value": -82.22,
      "overbought": 0,
      "oversold": 1,
      "zone": "oversold",
      "source": "real"
    },
    "hurst_exponent": 0.3404,
    "shannon_entropy": 2.9572,
    "fractal_dimension": 0.6244,
    "kalman_filter": {
      "kalman_price": 64937.88,
      "raw_price": 64882,
      "deviation_pct": -0.09,
      "trend_direction": "DOWN"
    },
    "regression_channel": {
      "slope_per_bar": -1.28,
      "trend_price": 64899.43,
      "upper_1sd": 64918.61,
      "lower_1sd": 64880.25,
      "upper_2sd": 64937.79,
      "lower_2sd": 64861.07,
      "deviation_from_trend": -17.43,
      "position_in_channel": 0.2728
    },
    "dominant_cycles": {
      "dominant_cycles": [
        100,
        40,
        33.3
      ],
      "cycle_strengths": [
        3040.54,
        1840.75,
        915.41
      ]
    },
    "monte_carlo": {
      "median_price": 64869.02,
      "p10": 64811.82,
      "p25": 64840.38,
      "p75": 64902.39,
      "p90": 64932.9,
      "prob_up": 0.47,
      "horizon_bars": 12
    },
    "garch_forecast_1h": 0.0013
  },
  "backup_exchanges": [
    "COINBASE",
    "KRAKEN",
    "OKX"
  ],
  "alerts": {
    "active_alerts": [
      {
        "type": "DEPTH_DIVERGENCE",
        "level": -0.797,
        "severity": "MEDIUM",
        "probability": 0.8,
        "action": "PREPARE_SHORT",
        "description": "Orderbook ASK_HEAVY: imbalance=-0.797"
      }
    ],
    "alert_count": 1,
    "max_severity": "MEDIUM"
  },
  "regime_analysis": {
    "current_regime": "MEAN_REVERTING",
    "regime_probabilities": {
      "trending": 0,
      "mean_reverting": 0.538,
      "breakout": 0.462
    },
    "regime_change_probability": 0.46,
    "expected_regime_duration": "15m-1h",
    "avg_adx": 26
  },
  "data_reliability": {
    "has_options_data": 0,
    "onchain_coverage": "full",
    "latency_acceptable": 1,
    "price_targets_available": 1
  },
  "event_id": "bc8b572b",
  "timestamp_ny": "2026-08-07T18:36:00.000-04:00",
  "timestamp_sp": "2026-08-07T19:36:00.000-03:00",
  "_log_id": "bc8b572b"
}

----------------------------------------------------------------------------------------------------
 # Janela 7
----------------------------------------------------------------------------------------------------
EVENTO: ANALYSIS_TRIGGER | SYMBOL: BTCUSDT | JANELA: 7
UTC: 2026-08-07 22:37:00 UTC
NY:  2026-08-07 18:37:00 EST/EDT
SP:  2026-08-07 19:37:00 BRT
----------------------------------------------------------------------------------------------------
{
  "is_signal": 1,
  "tipo_evento": "ANALYSIS_TRIGGER",
  "descricao": "Evento automático para análise da IA",
  "symbol": "BTCUSDT",
  "raw_event": {
    "delta": -0.073,
    "volume_total": 0.44,
    "volume_compra": 0.183,
    "volume_venda": 0.257,
    "preco_fechamento": 64872.9,
    "advanced_analysis": {
      "symbol": "BTCUSDT",
      "price": 64872.9,
      "volume": 0.44,
      "adaptive_thresholds": {
        "current_volatility": 0.03,
        "volatility_factor": 2.0,
        "absorption_threshold": 0.3,
        "flow_threshold": 0.2
      },
      "onchain_metrics": {
        "difficulty": 126.23,
        "active_addresses": 717974,
        "mempool_size": 87732,
        "mempool_vsize_mb": 43.83,
        "mempool_total_fee_btc": 0.1697,
        "fees_fastest_sat_vb": 5,
        "fees_half_hour_sat_vb": 4,
        "fees_hour_sat_vb": 3,
        "fees_economy_sat_vb": 2,
        "difficulty_adjustment": {
          "progress_pct": 93.4,
          "estimated_change_pct": 0.72,
          "remaining_blocks": 133,
          "remaining_time_ms": 79270394,
          "previous_retarget_pct": -0.74
        },
        "minutes_between_blocks": 8.4423,
        "total_btc_sent_24h": 807619.08,
        "trade_volume_btc_24h": 3265.64,
        "data_source": "blockchain.info+mempool.space",
        "is_real_data": 1,
        "requires_paid_api": [
          "exchange_netflow",
          "whale_transactions",
          "exchange_reserves",
          "sopr"
        ]
      }
    }
  },
  "resultado_da_batalha": "N/A",
  "delta": -0.073,
  "volume_total": 0.44,
  "volume_compra": 0.183,
  "volume_venda": 0.257,
  "preco_fechamento": 64872.9,
  "timestamp": "2026-08-07T22:37:07Z",
  "epoch_ms": 1786142220000,
  "ml_features": {
    "price_features": {
      "returns_1": 0.0,
      "volatility_1": 0.0,
      "returns_5": 0.0,
      "volatility_5": 1e-07,
      "returns_15": 0.0,
      "volatility_15": 1e-07,
      "momentum_score": 0.0,
      "volatility_1h": 0.0023
    },
    "volume_features": {
      "volume_sma_ratio": 0.896,
      "volume_momentum": 0.526,
      "buy_sell_pressure": -0.1661,
      "liquidity_gradient": 0.52595017
    },
    "microstructure": {
      "order_book_slope": 4.526129,
      "flow_imbalance": -0.1661,
      "tick_rule_sum": -1,
      "trade_intensity": 20,
      "trade_intensity_v2": 1.4
    },
    "cross_asset": {
      "btc_eth_corr_7d": 0.8212,
      "btc_eth_corr_30d": 0.8599,
      "btc_dxy_corr_30d": -0.0603,
      "btc_dxy_corr_90d": -0.0669,
      "btc_ndx_corr_30d": 0.4146,
      "dxy_return_5d": -0.3561,
      "dxy_return_20d": -1.6548,
      "btc_dxy_correlation_stability": 0.0066,
      "btc_dxy_inverse_strength": 0.0636,
      "dxy_momentum": -1.29867555,
      "vix_current": 14.9,
      "us10y_yield": 4.66,
      "btc_dominance": 17.0386,
      "btc_dominance_change_7d": 0,
      "eth_dominance": 9.0982,
      "gold_price": 4342.3475,
      "oil_price": 77.08,
      "macro_regime": "RISK_ON",
      "correlation_regime": "DECORRELATED"
    }
  },
  "orderbook_data": {
    "mid": 64842.65,
    "spread": 0.1,
    "spread_percent": 0,
    "bid_depth_usd": 620781.19,
    "ask_depth_usd": 4014241.69,
    "imbalance": -0.732,
    "flow_imbalance": -0.7321,
    "volume_ratio": 0.155,
    "pressure": -0.7321,
    "consolidated_bias_score": 0.1958,
    "spread_bps": 0.0154,
    "is_valid": 1,
    "data_source": "live",
    "spread_volatility": 0.0
  },
  "historical_vp": {
    "daily": {
      "poc": 65046,
      "vah": 65346,
      "val": 64522,
      "status": "success"
    },
    "weekly": {
      "poc": 62610,
      "vah": 63567,
      "val": 62314,
      "status": "success"
    },
    "monthly": {
      "poc": 63554,
      "vah": 64214,
      "val": 61789,
      "status": "success"
    }
  },
  "multi_tf": {
    "15m": {
      "tendencia": "Baixa",
      "preco_atual": 64872.89,
      "mme_21": 64925.49,
      "atr": 70.07,
      "regime": "Range",
      "rsi_short": 38,
      "rsi_long": 44.14,
      "macd": 12.7253,
      "macd_signal": 16.6706,
      "adx": 23.54,
      "realized_vol": 0.0014
    },
    "1h": {
      "tendencia": "Alta",
      "preco_atual": 64872.89,
      "mme_21": 64788.88,
      "atr": 223.59,
      "regime": "Range",
      "rsi_short": 52.68,
      "rsi_long": 54.4,
      "macd": 100.4266,
      "macd_signal": 98.3149,
      "adx": 24.35,
      "realized_vol": 0.0023
    },
    "4h": {
      "tendencia": "Alta",
      "preco_atual": 64872.89,
      "mme_21": 64460.43,
      "atr": 495.32,
      "regime": "Range",
      "rsi_short": 61.98,
      "rsi_long": 61.12,
      "macd": 275.0187,
      "macd_signal": 256.7251,
      "adx": 30.01,
      "realized_vol": 0.0058
    },
    "1d": {
      "tendencia": "Alta",
      "preco_atual": 64924.96,
      "mme_21": 64149.58,
      "atr": 1369.69,
      "regime": "Manipulação",
      "rsi_short": 59.35,
      "rsi_long": 55.96,
      "macd": 71.3764,
      "macd_signal": 36.7677,
      "adx": 15.05,
      "realized_vol": 0.0184
    }
  },
  "data_context": "real_time",
  "price_targets": [
    {
      "level": 66819.09,
      "confidence": 0.3,
      "source": "fallback_r1",
      "weight": 0.1
    },
    {
      "level": 68765.27,
      "confidence": 0.2,
      "source": "fallback_r2",
      "weight": 0.05
    },
    {
      "level": 62926.71,
      "confidence": 0.3,
      "source": "fallback_s1",
      "weight": 0.1
    },
    {
      "level": 60980.53,
      "confidence": 0.2,
      "source": "fallback_s2",
      "weight": 0.05
    },
    {
      "level": 64872.9,
      "confidence": 0.4,
      "source": "fallback_poc",
      "weight": 0.15
    }
  ],
  "pattern_recognition": {
    "smart_money": {
      "fair_value_gaps": [
        {
          "type": "BEARISH",
          "top": 64925,
          "bottom": 64892.3,
          "gap_size": 32.7,
          "gap_pct": 0.05
        }
      ],
      "market_structure": {
        "structure": "BEARISH",
        "bos_detected": 0,
        "last_swing_high": 64925,
        "last_swing_low": 64909.55,
        "higher_highs": 0,
        "higher_lows": 0
      }
    }
  },
  "janela_numero": 7,
  "timestamp_utc": "2026-08-07T22:37:00.000Z",
  "derivatives": {
    "BTCUSDT": {
      "funding_rate_percent": 0.01,
      "open_interest": 106934.462,
      "open_interest_usd": 6934209575.49,
      "long_short_ratio": 1.1,
      "longs_usd": 3631418636.94,
      "shorts_usd": 3302790938.55
    },
    "ETHUSDT": {
      "funding_rate_percent": 0,
      "open_interest": 2279900.074,
      "open_interest_usd": 4362909179.41,
      "long_short_ratio": 2.07,
      "longs_usd": 2940607898.43,
      "shorts_usd": 1422301280.98
    }
  },
  "fluxo_continuo": {
    "cvd": -17.8,
    "whale_buy_volume": 0,
    "whale_sell_volume": 2,
    "whale_delta": -2,
    "bursts": {
      "count": 4,
      "max_burst_volume": 7.558
    },
    "sector_flow": {
      "retail": {
        "buy": 4.8595,
        "sell": 17.6898,
        "delta": -12.83
      },
      "mid": {
        "buy": 0.9922,
        "sell": 3.9619,
        "delta": -2.97
      },
      "whale": {
        "buy": 0,
        "sell": 2,
        "delta": -2
      }
    },
    "timestamp": "2026-08-07T22:37:00.000+00:00",
    "time_index": {
      "epoch_ms": 1786142220000,
      "timestamp_utc": "2026-08-07T22:37:00.000+00:00",
      "timestamp_ny": "2026-08-07T18:37:00.000-04:00",
      "timestamp_sp": "2026-08-07T19:37:00.000-03:00"
    },
    "order_flow": {
      "net_flow_1m": -5364.3374,
      "absorcao_1m": "Neutra",
      "buy_volume": 11900.93,
      "sell_volume": 16641.19,
      "total_volume": 28542.13,
      "buy_volume_btc": 0.183,
      "sell_volume_btc": 0.257,
      "total_volume_btc": 0.44,
      "whale_buy_volume_window": 0,
      "whale_sell_volume_window": 0,
      "whale_delta_window": 0,
      "flow_imbalance": -0.1661,
      "aggressive_buy_pct": 41.7,
      "aggressive_sell_pct": 58.3,
      "net_flow_5m": -827426.3552,
      "absorcao_5m": "Neutra",
      "net_flow_15m": -1155414.831,
      "absorcao_15m": "Neutra",
      "computation_window_min": 1,
      "available_windows_min": [
        1,
        5,
        15
      ],
      "buy_sell_ratio": {
        "buy_sell_ratio": 0.72,
        "ratios": {
          "current": 0.7151,
          "imbalance_1m": -0.188,
          "imbalance_5m": -28.99,
          "imbalance_15m": -40.481
        },
        "sector_ratios": {
          "retail": 0.2747,
          "mid": 0.2504,
          "whale": 0
        },
        "pressure": "SLIGHT_SELL",
        "flow_trend": "consistent_selling",
        "buy_volume": 0.183,
        "sell_volume": 0.257
      }
    },
    "tipo_absorcao": "Neutra",
    "participant_analysis": {
      "retail": {
        "volume_pct": 100,
        "direction": "SELL",
        "sentiment": "BEARISH",
        "composite_score": -0.491,
        "imbalance": -0.166
      }
    },
    "liquidity_heatmap": {
      "clusters": [
        {
          "center": 64881.4425,
          "low": 64872.67,
          "high": 64892.32,
          "width": 19.65,
          "total_volume": 9.5,
          "buy_volume": 1.716,
          "sell_volume": 7.784,
          "imbalance": -6.068,
          "imbalance_ratio": -0.639,
          "trades_count": 2000,
          "avg_trade_size": 0.005,
          "recent_timestamp": 1786142225653,
          "recent_ts_ms": 1786142225653,
          "last_seen_ms": 1786142225653,
          "first_seen_ms": 1786141969517,
          "age_ms": 276.0,
          "cluster_duration_ms": 256136,
          "price_std": 7.336,
          "volume_std": 0.033,
          "bin_threshold_usd": 194.6443
        }
      ],
      "resistances": [
        64881.4425
      ],
      "clusters_count": 1
    },
    "absorption_analysis": {
      "current_absorption": {
        "index": 0.0312,
        "classification": "NONE",
        "label": "Neutra",
        "buyer_strength": 4.2,
        "seller_exhaustion": 1.7,
        "continuation_probability": 0.03,
        "delta_usd": -5364.337,
        "total_volume_usd": 28542.13,
        "flow_imbalance": -0.1661,
        "window_min": 1
      }
    }
  },
  "market_context": {
    "trading_session": "NY",
    "session_phase": "ACTIVE",
    "time_to_session_close": 12269,
    "day_of_week": 4,
    "is_holiday": 0,
    "market_hours_type": "EXTENDED"
  },
  "market_environment": {
    "volatility_regime": "NORMAL",
    "trend_direction": "UP",
    "market_structure": "RANGE_BOUND",
    "liquidity_environment": "NORMAL",
    "risk_sentiment": "BULLISH",
    "correlation_spy": 0.406,
    "correlation_dxy": -0.0867,
    "correlation_gold": 0.2401
  },
  "external_markets": {
    "FEAR_GREED": {
      "preco_atual": 29,
      "prev": 25,
      "movimento": "Alta",
      "classification": "Fear",
      "source": "alternative.me",
      "timestamp": "2026-08-07T19:30:17.741873"
    }
  },
  "contextual_snapshot": {
    "symbol": "BTCUSDT",
    "ohlc": {
      "open": 64872.9,
      "high": 64872.9,
      "low": 64872.9,
      "close": 64872.9,
      "open_time": 1786142160358,
      "close_time": 1786142217447,
      "vwap": 64872.9
    },
    "volume_total": 0.44,
    "volume_total_usdt": 28542,
    "volume_compra": 0.183,
    "volume_venda": 0.257,
    "num_trades": 73,
    "delta_minimo": -0.153,
    "delta_maximo": 0.037,
    "delta_fechamento": -0.073,
    "reversao_desde_minimo": 0.08,
    "reversao_desde_maximo": 0.11,
    "poc_price": 64872.9,
    "poc_volume": 0.26,
    "poc_percentage": 58.3,
    "dwell_price": 64872.9,
    "dwell_seconds": 56,
    "dwell_location": "Low",
    "trades_per_second": 1.28,
    "avg_trade_size": 0.006
  },
  "order_book_depth": {
    "L1": {
      "bids": 309623.42,
      "asks": 2726505.85,
      "flow_imbalance": -0.796
    },
    "L5": {
      "bids": 313903.01,
      "asks": 3526796.52,
      "flow_imbalance": -0.8365
    },
    "L10": {
      "bids": 317339.62,
      "asks": 3545406.59,
      "flow_imbalance": -0.8357
    },
    "L25": {
      "bids": 326676.64,
      "asks": 3717245.02,
      "flow_imbalance": -0.8384
    },
    "total_depth_ratio": 0.09
  },
  "orderbook_quality": "live",
  "market_impact": {
    "slippage_matrix": {
      "100k_usd": {
        "buy": 0.05,
        "sell": 0.05
      },
      "1m_usd": {
        "buy": 0.05,
        "sell": 6.15
      }
    },
    "liquidity_score": 9.9985,
    "execution_quality": "EXCELLENT"
  },
  "institutional_analytics": {
    "status": "ok",
    "computed_at_ms": 1786142227158,
    "technical_extras": {
      "stoch_rsi": {
        "k": 4.19,
        "d": 15.3,
        "overbought": 0,
        "oversold": 1,
        "crossover": "none"
      },
      "williams_r": {
        "value": -99.62,
        "overbought": 0,
        "oversold": 1,
        "zone": "oversold",
        "source": "real"
      },
      "hurst_exponent": 0.3356,
      "shannon_entropy": 2.9459,
      "kalman_filter": {
        "kalman_price": 64935.87,
        "raw_price": 64872.9,
        "deviation_pct": -0.1,
        "trend_direction": "DOWN"
      },
      "regression_channel": {
        "slope_per_bar": -1.1836,
        "trend_price": 64898.73,
        "upper_1sd": 64915.9,
        "lower_1sd": 64881.57,
        "upper_2sd": 64933.06,
        "lower_2sd": 64864.4,
        "deviation_from_trend": -25.83,
        "position_in_channel": 0.1238
      },
      "dominant_cycles": {
        "dominant_cycles": [
          100,
          40,
          33.3
        ],
        "cycle_strengths": [
          3051.73,
          1853.49,
          930.27
        ]
      },
      "fractal_dimension": 0.6253,
      "monte_carlo": {
        "median_price": 64870.11,
        "p10": 64813.44,
        "p25": 64841.73,
        "p75": 64903.18,
        "p90": 64933.41,
        "prob_up": 0.48,
        "horizon_bars": 12
      },
      "fair_value_gaps": [
        {
          "type": "BEARISH",
          "top": 64925,
          "bottom": 64892.3,
          "gap_size": 32.7,
          "gap_pct": 0.05
        }
      ],
      "market_structure": {
        "structure": "BEARISH",
        "bos_detected": 0,
        "last_swing_high": 64925,
        "last_swing_low": 64909.55,
        "higher_highs": 0,
        "higher_lows": 0
      }
    },
    "profile_analysis": {
      "poor_extremes": {
        "poor_high": {
          "detected": 1,
          "price": 64872.9,
          "volume_ratio": 4.17,
          "implication": "High likely to be revisited - unfinished auction"
        },
        "poor_low": {
          "detected": 1,
          "price": 64872.89,
          "volume_ratio": 5.83,
          "implication": "Low likely to be revisited - unfinished auction"
        },
        "excess_high": 0,
        "excess_low": 0,
        "session_high": 64872.9,
        "session_low": 64872.89,
        "action_bias": "expect_retest_both",
        "status": "success"
      },
      "profile_shape": {
        "shape": "b",
        "implication": "Long liquidation - bullish bias expected",
        "trading_signal": "BULLISH_AFTER",
        "distribution": {
          "lower_third_pct": 58.3,
          "middle_third_pct": 0,
          "upper_third_pct": 41.7
        },
        "dominant_zone": "lower",
        "status": "success"
      },
      "no_mans_land": {
        "zones": [
          {
            "range_low": 64512,
            "range_high": 64810,
            "gap_size": 298,
            "gap_size_pct": 0.462,
            "risk": "LOW",
            "lvn_confirmed": 1,
            "lvns_in_zone": 9,
            "nearest_hvn_below": 64512,
            "nearest_hvn_above": 64810,
            "distance_from_price": 62.9,
            "distance_pct": 0.1,
            "direction": "below"
          }
        ],
        "price_in_no_mans_land": 0,
        "nearest_no_mans_land": {
          "range_low": 64512,
          "range_high": 64810,
          "gap_size": 298,
          "gap_size_pct": 0.462,
          "risk": "LOW",
          "lvn_confirmed": 1,
          "lvns_in_zone": 9,
          "nearest_hvn_below": 64512,
          "nearest_hvn_above": 64810,
          "distance_from_price": 62.9,
          "distance_pct": 0.1,
          "direction": "below"
        },
        "total_zones": 1,
        "max_gap_pct": 0.46,
        "status": "success"
      },
      "va_volume_pct": {
        "value_area_volume_pct": 100,
        "interpretation": "extremely_compressed",
        "breakout_risk": "VERY_HIGH",
        "volume_in_va": 1,
        "total_volume": 1,
        "compression_signal": 1
      },
      "volume_node_strength": {
        "scored_hvns": [
          {
            "price": 64304,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64306,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64314,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64320,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64322,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.8,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          }
        ],
        "scored_lvns": [
          {
            "price": 64423,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.5,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64436,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64440,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64447,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64448,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          }
        ],
        "total_hvns": 52,
        "total_lvns": 101,
        "avg_hvn_strength": 66.3,
        "avg_lvn_strength": 63.1,
        "status": "success"
      }
    },
    "flow_analysis": {
      "passive_aggressive": {
        "aggressive": {
          "buy_pct": 41.7,
          "sell_pct": 58.3,
          "net_pct": -16.6,
          "dominance": "sellers",
          "buy_volume": 0.183,
          "sell_volume": 0.257
        },
        "passive": {
          "dominance": "sellers",
          "inference": "from_orderbook_depth",
          "bid_depth": 620781.19,
          "ask_depth": 4014241.69,
          "bid_ratio": 0.13,
          "ob_imbalance": -0.732
        },
        "composite": {
          "agreement": 1,
          "signal": "strong_bearish",
          "interpretation": "Both aggressive and passive sellers active - strong downward trend",
          "conviction": "HIGH"
        },
        "status": "success"
      },
      "whale_accumulation": {
        "score": -26,
        "classification": "MILD_DISTRIBUTION",
        "bias": "DISTRIBUTING",
        "components": {
          "flow": {
            "score": -20,
            "max": 30,
            "detail": {
              "divergence": "aligned",
              "whale_delta": -2,
              "mid_delta": -2.97,
              "retail_delta": -12.83,
              "primary_delta": -2
            }
          },
          "depth": {
            "score": -14.64,
            "max": 20,
            "detail": {
              "bid_depth": 620781.19,
              "ask_depth": 4014241.69,
              "ratio": -0.73,
              "deep_confirmation": 0
            }
          },
          "absorption": {
            "score": 7.5,
            "max": 25,
            "detail": {
              "buyer_strength": 4.2,
              "seller_exhaustion": 1.7,
              "net_absorption": 2.5,
              "index": 0.0312,
              "label": "Neutra"
            }
          },
          "derivatives": {
            "score": 1.49,
            "max": 25,
            "detail": {
              "long_short_ratio": 1.1,
              "lsr_score": 1.49
            }
          }
        },
        "trend": {
          "direction": "increasing_distribution",
          "avg_score": -22.4,
          "recent_avg": -30.4,
          "momentum": -3.6,
          "samples": 7,
          "score_range": {
            "min": -49,
            "max": 14
          }
        },
        "status": "success"
      },
      "absorption_zones": {
        "total_zones": 0,
        "total_events": 0,
        "buy_zone_count": 0,
        "sell_zone_count": 0,
        "status": "no_events"
      }
    },
    "sr_analysis": {
      "defense_zones": {
        "buy_defense": [
          {
            "center": 64479.86,
            "range_low": 64376.35,
            "range_high": 64570.65,
            "strength": 62,
            "side": "buy",
            "sources": [
              "sr_level_ema_21_4h",
              "vp_hvn",
              "ema_ema_21_4h",
              "vp_val"
            ],
            "source_count": 4,
            "signals_in_zone": 5,
            "type": "confluence",
            "distance_from_price": 393.04,
            "distance_pct": 0.61
          },
          {
            "center": 64837.48,
            "range_low": 64740.23,
            "range_high": 64928.65,
            "strength": 49,
            "side": "buy",
            "sources": [
              "vp_hvn",
              "ema_ema_21_1h",
              "sr_level_ema_21_1h"
            ],
            "source_count": 3,
            "signals_in_zone": 7,
            "type": "confluence",
            "distance_from_price": 35.42,
            "distance_pct": 0.05
          },
          {
            "center": 64165.68,
            "range_low": 64100.93,
            "range_high": 64230.44,
            "strength": 46,
            "side": "buy",
            "sources": [
              "ema_ema_21_1d",
              "sr_level_vah_monthly"
            ],
            "source_count": 2,
            "signals_in_zone": 2,
            "type": "cluster",
            "distance_from_price": 707.22,
            "distance_pct": 1.09
          },
          {
            "center": 64336.25,
            "range_low": 64255.35,
            "range_high": 64442.65,
            "strength": 40,
            "side": "buy",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 9,
            "type": "cluster",
            "distance_from_price": 536.65,
            "distance_pct": 0.83
          }
        ],
        "sell_defense": [
          {
            "center": 64943.83,
            "range_low": 64860.35,
            "range_high": 65052.65,
            "strength": 59,
            "side": "sell",
            "sources": [
              "orderbook_ask_wall",
              "vp_hvn",
              "ema_ema_21_15m",
              "sr_level_hvn_daily"
            ],
            "source_count": 4,
            "signals_in_zone": 10,
            "type": "confluence",
            "distance_from_price": 70.93,
            "distance_pct": 0.11
          },
          {
            "center": 65037.38,
            "range_low": 64960.35,
            "range_high": 65128.65,
            "strength": 55,
            "side": "sell",
            "sources": [
              "sr_level_poc_daily",
              "vp_poc",
              "vp_hvn"
            ],
            "source_count": 3,
            "signals_in_zone": 9,
            "type": "confluence",
            "distance_from_price": 164.48,
            "distance_pct": 0.25
          },
          {
            "center": 65346,
            "range_low": 65297.35,
            "range_high": 65394.65,
            "strength": 53,
            "side": "sell",
            "sources": [
              "sr_level_vah_daily",
              "vp_hvn",
              "vp_vah"
            ],
            "source_count": 3,
            "signals_in_zone": 3,
            "type": "confluence",
            "distance_from_price": 473.1,
            "distance_pct": 0.73
          },
          {
            "center": 65157.17,
            "range_low": 65064.35,
            "range_high": 65257.65,
            "strength": 41,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 13,
            "type": "cluster",
            "distance_from_price": 284.27,
            "distance_pct": 0.44
          },
          {
            "center": 65252.3,
            "range_low": 65164.35,
            "range_high": 65350.65,
            "strength": 40,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 11,
            "type": "cluster",
            "distance_from_price": 379.4,
            "distance_pct": 0.58
          }
        ],
        "total_zones": 9,
        "strongest_buy": {
          "center": 64479.86,
          "range_low": 64376.35,
          "range_high": 64570.65,
          "strength": 62,
          "side": "buy",
          "sources": [
            "sr_level_ema_21_4h",
            "vp_hvn",
            "ema_ema_21_4h",
            "vp_val"
          ],
          "source_count": 4,
          "signals_in_zone": 5,
          "type": "confluence",
          "distance_from_price": 393.04,
          "distance_pct": 0.61
        },
        "strongest_sell": {
          "center": 64943.83,
          "range_low": 64860.35,
          "range_high": 65052.65,
          "strength": 59,
          "side": "sell",
          "sources": [
            "orderbook_ask_wall",
            "vp_hvn",
            "ema_ema_21_15m",
            "sr_level_hvn_daily"
          ],
          "source_count": 4,
          "signals_in_zone": 10,
          "type": "confluence",
          "distance_from_price": 70.93,
          "distance_pct": 0.11
        },
        "defense_asymmetry": {
          "ratio": 0.79,
          "bias": "slight_sell_defense",
          "description": "Slightly more sell defense",
          "buy_total_strength": 197,
          "sell_total_strength": 248
        },
        "status": "success"
      }
    },
    "quality": {
      "calendar": {
        "day_of_week": "Friday",
        "day_of_week_num": 4,
        "is_us_holiday": 0,
        "is_weekend": 0,
        "is_pre_holiday": 0,
        "is_post_holiday": 0,
        "expected_liquidity": "NORMAL",
        "liquidity_warning": 0
      },
      "latency": {
        "latency_ms": 7384,
        "latency_category": "POOR",
        "data_freshness": "DELAYED",
        "is_acceptable": 0,
        "is_stale": 0
      },
      "spread_percentile": {
        "status": "ok",
        "current_spread": 0.1,
        "spread_percentile": 0,
        "spread_mean": 0.1,
        "spread_median": 0.1,
        "spread_std": 0,
        "spread_min": 0.1,
        "spread_max": 0.1,
        "spread_z_score": 0,
        "is_tight": 1,
        "is_wide": 0,
        "is_anomalous": 0,
        "liquidity_signal": "EXCELLENT",
        "samples": 7,
        "window_minutes": 1440
      },
      "anomalies": {
        "anomalies_detected": 1,
        "count": 1,
        "anomalies": [
          {
            "type": "DEPTH_EXTREME_ASYMMETRY",
            "severity": "MEDIUM",
            "ratio": 0.15,
            "bid_depth": 620781.19,
            "ask_depth": 4014241.69,
            "direction": "ASK_HEAVY",
            "description": "Order book depth ratio 0.15:1 is extreme"
          }
        ],
        "max_severity": "MEDIUM",
        "risk_elevated": 0,
        "types_found": [
          "DEPTH_EXTREME_ASYMMETRY"
        ],
        "summary": "1 anomalies detected (max severity: MEDIUM)"
      }
    }
  },
  "sequence_id": 7,
  "primary_exchange": "BINANCE",
  "data_feed_type": "WEBSOCKET_L2",
  "data_quality_score": 9.75,
  "completeness_pct": 100,
  "reliability_score": 9.5,
  "bid": 64842.6,
  "ask": 64842.7,
  "tick_direction": 0,
  "twap": 64894.23,
  "pivot_points": {
    "daily": {
      "pivot": 64971.33,
      "r1": 65420.66,
      "r2": 65795.33,
      "r3": 66619.33,
      "s1": 64596.66,
      "s2": 64147.33,
      "s3": 63323.33,
      "vah": 65346,
      "val": 64522,
      "poc": 65046
    },
    "weekly": {
      "pivot": 62830.33,
      "r1": 63346.66,
      "r2": 64083.33,
      "r3": 65336.33,
      "s1": 62093.66,
      "s2": 61577.33,
      "s3": 60324.33,
      "vah": 63567,
      "val": 62314,
      "poc": 62610
    },
    "monthly": {
      "pivot": 63185.67,
      "r1": 64582.34,
      "r2": 65610.67,
      "r3": 68035.67,
      "s1": 62157.34,
      "s2": 60760.67,
      "s3": 58335.67,
      "vah": 64214,
      "val": 61789,
      "poc": 63554
    }
  },
  "immediate_support": [
    64837.48,
    64522,
    64479.86,
    64214,
    64165.68
  ],
  "support_strength": [
    49,
    94.6,
    62,
    71.9,
    46
  ],
  "immediate_resistance": [
    64943.83,
    64971.33,
    65037.38,
    65346
  ],
  "resistance_strength": [
    59,
    98.5,
    55,
    92.7
  ],
  "volatility_metrics": {
    "realized_vol_24h": 0.0184,
    "realized_vol_7d": 0.0487,
    "volatility_regime": "NORMAL"
  },
  "order_flow_extended": {
    "passive_buy_pct": 13.4,
    "passive_sell_pct": 86.6
  },
  "whale_activity": {
    "large_orders_1h": [
      {
        "size": 1,
        "price": 64896.34,
        "side": "SELL",
        "timestamp_ms": 1786141945725
      },
      {
        "size": 1,
        "price": 64883.14,
        "side": "SELL",
        "timestamp_ms": 1786142044496
      }
    ],
    "iceberg_activity": 1,
    "hidden_orders_detected": 1
  },
  "technical_indicators_extended": {
    "cci_1h": 100.2,
    "cci_signal": "OVERBOUGHT",
    "stochastic": {
      "k": 4.19,
      "d": 15.3,
      "signal": "OVERSOLD",
      "source": "real"
    },
    "williams_r": {
      "value": -99.62,
      "overbought": 0,
      "oversold": 1,
      "zone": "oversold",
      "source": "real"
    },
    "hurst_exponent": 0.3356,
    "shannon_entropy": 2.9459,
    "fractal_dimension": 0.6253,
    "kalman_filter": {
      "kalman_price": 64935.87,
      "raw_price": 64872.9,
      "deviation_pct": -0.1,
      "trend_direction": "DOWN"
    },
    "regression_channel": {
      "slope_per_bar": -1.1836,
      "trend_price": 64898.73,
      "upper_1sd": 64915.9,
      "lower_1sd": 64881.57,
      "upper_2sd": 64933.06,
      "lower_2sd": 64864.4,
      "deviation_from_trend": -25.83,
      "position_in_channel": 0.1238
    },
    "dominant_cycles": {
      "dominant_cycles": [
        100,
        40,
        33.3
      ],
      "cycle_strengths": [
        3051.73,
        1853.49,
        930.27
      ]
    },
    "monte_carlo": {
      "median_price": 64870.11,
      "p10": 64813.44,
      "p25": 64841.73,
      "p75": 64903.18,
      "p90": 64933.41,
      "prob_up": 0.48,
      "horizon_bars": 12
    },
    "garch_forecast_1h": 0.0014
  },
  "backup_exchanges": [
    "COINBASE",
    "KRAKEN",
    "OKX"
  ],
  "alerts": {
    "active_alerts": [
      {
        "type": "SUPPORT_TEST",
        "level": 64837.48,
        "severity": "HIGH",
        "probability": 0.89,
        "action": "MONITOR_CLOSELY",
        "description": "Preço testando suporte em 64837.48 (dist: 0.05%)"
      },
      {
        "type": "RESISTANCE_TEST",
        "level": 64943.83,
        "severity": "MEDIUM",
        "probability": 0.78,
        "action": "MONITOR_CLOSELY",
        "description": "Preço testando resistência em 64943.83 (dist: 0.11%)"
      },
      {
        "type": "WHALE_DISTRIBUTION",
        "level": -26,
        "severity": "MEDIUM",
        "probability": 0.52,
        "action": "AVOID_LONG",
        "description": "Sinal de distribuição de whales (score=-26)"
      }
    ],
    "alert_count": 3,
    "max_severity": "HIGH"
  },
  "regime_analysis": {
    "current_regime": "MEAN_REVERTING",
    "regime_probabilities": {
      "trending": 0,
      "mean_reverting": 0.636,
      "breakout": 0.364
    },
    "regime_change_probability": 0.36,
    "expected_regime_duration": "15m-1h",
    "avg_adx": 26
  },
  "data_reliability": {
    "has_options_data": 0,
    "onchain_coverage": "full",
    "latency_acceptable": 1,
    "price_targets_available": 1
  },
  "event_id": "894cbe83",
  "timestamp_ny": "2026-08-07T18:37:00.000-04:00",
  "timestamp_sp": "2026-08-07T19:37:00.000-03:00",
  "_log_id": "894cbe83"
}

----------------------------------------------------------------------------------------------------
 # Janela 8
----------------------------------------------------------------------------------------------------
EVENTO: ANALYSIS_TRIGGER | SYMBOL: BTCUSDT | JANELA: 8
UTC: 2026-08-07 22:38:00 UTC
NY:  2026-08-07 18:38:00 EST/EDT
SP:  2026-08-07 19:38:00 BRT
----------------------------------------------------------------------------------------------------
{
  "is_signal": 1,
  "tipo_evento": "ANALYSIS_TRIGGER",
  "descricao": "Evento automático para análise da IA",
  "symbol": "BTCUSDT",
  "raw_event": {
    "delta": -8.128,
    "volume_total": 8.708,
    "volume_compra": 0.29,
    "volume_venda": 8.418,
    "preco_fechamento": 64846,
    "advanced_analysis": {
      "symbol": "BTCUSDT",
      "price": 64846,
      "volume": 8.708,
      "adaptive_thresholds": {
        "current_volatility": 0.03,
        "volatility_factor": 2.0,
        "absorption_threshold": 0.3,
        "flow_threshold": 0.2
      },
      "onchain_metrics": {
        "difficulty": 126.23,
        "active_addresses": 717974,
        "mempool_size": 87732,
        "mempool_vsize_mb": 43.83,
        "mempool_total_fee_btc": 0.1697,
        "fees_fastest_sat_vb": 5,
        "fees_half_hour_sat_vb": 4,
        "fees_hour_sat_vb": 3,
        "fees_economy_sat_vb": 2,
        "difficulty_adjustment": {
          "progress_pct": 93.4,
          "estimated_change_pct": 0.72,
          "remaining_blocks": 133,
          "remaining_time_ms": 79270394,
          "previous_retarget_pct": -0.74
        },
        "minutes_between_blocks": 8.4423,
        "total_btc_sent_24h": 807619.08,
        "trade_volume_btc_24h": 3265.64,
        "data_source": "blockchain.info+mempool.space",
        "is_real_data": 1,
        "requires_paid_api": [
          "exchange_netflow",
          "whale_transactions",
          "exchange_reserves",
          "sopr"
        ]
      }
    }
  },
  "resultado_da_batalha": "N/A",
  "delta": -8.128,
  "volume_total": 8.708,
  "volume_compra": 0.29,
  "volume_venda": 8.418,
  "preco_fechamento": 64846,
  "timestamp": "2026-08-07T22:38:11Z",
  "epoch_ms": 1786142280000,
  "ml_features": {
    "price_features": {
      "returns_1": 1.5e-07,
      "volatility_1": 0.0,
      "returns_5": 1.5e-07,
      "volatility_5": 1.2e-07,
      "returns_15": -2.16e-06,
      "volatility_15": 5.4e-07,
      "momentum_score": -1.41421356,
      "volatility_1h": 0.0023
    },
    "volume_features": {
      "volume_sma_ratio": 1.165,
      "volume_momentum": -0.195,
      "buy_sell_pressure": -0.9334,
      "liquidity_gradient": -0.19455105
    },
    "microstructure": {
      "order_book_slope": -2.837923,
      "flow_imbalance": -0.9334,
      "tick_rule_sum": -223,
      "trade_intensity": 30,
      "trade_intensity_v2": 20.6167
    },
    "cross_asset": {
      "btc_eth_corr_7d": 0.8212,
      "btc_eth_corr_30d": 0.8599,
      "btc_dxy_corr_30d": -0.0603,
      "btc_dxy_corr_90d": -0.0669,
      "btc_ndx_corr_30d": 0.4146,
      "dxy_return_5d": -0.3561,
      "dxy_return_20d": -1.6548,
      "btc_dxy_correlation_stability": 0.0066,
      "btc_dxy_inverse_strength": 0.0636,
      "dxy_momentum": -1.29867555,
      "vix_current": 14.9,
      "us10y_yield": 4.66,
      "btc_dominance": 17.0386,
      "btc_dominance_change_7d": 0,
      "eth_dominance": 9.0982,
      "gold_price": 4342.3475,
      "oil_price": 77.08,
      "macro_regime": "RISK_ON",
      "correlation_regime": "DECORRELATED"
    }
  },
  "orderbook_data": {
    "mid": 64819.35,
    "spread": 0.1,
    "spread_percent": 0,
    "bid_depth_usd": 3585137.77,
    "ask_depth_usd": 307774.65,
    "imbalance": 0.842,
    "flow_imbalance": 0.8419,
    "volume_ratio": 11.649,
    "pressure": 0.8419,
    "consolidated_bias_score": 0.9526,
    "spread_bps": 0.0154,
    "is_valid": 1,
    "data_source": "live",
    "spread_volatility": 0.0
  },
  "historical_vp": {
    "daily": {
      "poc": 65046,
      "vah": 65346,
      "val": 64522,
      "status": "success"
    },
    "weekly": {
      "poc": 62610,
      "vah": 63567,
      "val": 62314,
      "status": "success"
    },
    "monthly": {
      "poc": 63554,
      "vah": 64214,
      "val": 61789,
      "status": "success"
    }
  },
  "multi_tf": {
    "15m": {
      "tendencia": "Baixa",
      "preco_atual": 64872.89,
      "mme_21": 64925.49,
      "atr": 70.07,
      "regime": "Range",
      "rsi_short": 38,
      "rsi_long": 44.14,
      "macd": 12.7253,
      "macd_signal": 16.6706,
      "adx": 23.54,
      "realized_vol": 0.0014
    },
    "1h": {
      "tendencia": "Alta",
      "preco_atual": 64872.89,
      "mme_21": 64788.88,
      "atr": 223.59,
      "regime": "Range",
      "rsi_short": 52.68,
      "rsi_long": 54.4,
      "macd": 100.4266,
      "macd_signal": 98.3149,
      "adx": 24.35,
      "realized_vol": 0.0023
    },
    "4h": {
      "tendencia": "Alta",
      "preco_atual": 64872.89,
      "mme_21": 64460.43,
      "atr": 495.32,
      "regime": "Range",
      "rsi_short": 61.98,
      "rsi_long": 61.12,
      "macd": 275.0187,
      "macd_signal": 256.7251,
      "adx": 30.01,
      "realized_vol": 0.0058
    },
    "1d": {
      "tendencia": "Alta",
      "preco_atual": 64924.96,
      "mme_21": 64149.58,
      "atr": 1369.69,
      "regime": "Manipulação",
      "rsi_short": 59.35,
      "rsi_long": 55.96,
      "macd": 71.3764,
      "macd_signal": 36.7677,
      "adx": 15.05,
      "realized_vol": 0.0184
    }
  },
  "data_context": "real_time",
  "price_targets": [
    {
      "level": 66791.38,
      "confidence": 0.3,
      "source": "fallback_r1",
      "weight": 0.1
    },
    {
      "level": 68736.76,
      "confidence": 0.2,
      "source": "fallback_r2",
      "weight": 0.05
    },
    {
      "level": 62900.62,
      "confidence": 0.3,
      "source": "fallback_s1",
      "weight": 0.1
    },
    {
      "level": 60955.24,
      "confidence": 0.2,
      "source": "fallback_s2",
      "weight": 0.05
    },
    {
      "level": 64846,
      "confidence": 0.4,
      "source": "fallback_poc",
      "weight": 0.15
    }
  ],
  "pattern_recognition": {
    "smart_money": {
      "fair_value_gaps": [
        {
          "type": "BEARISH",
          "top": 64925,
          "bottom": 64892.3,
          "gap_size": 32.7,
          "gap_pct": 0.05
        }
      ],
      "market_structure": {
        "structure": "BEARISH",
        "bos_detected": 0,
        "last_swing_high": 64925,
        "last_swing_low": 64909.55,
        "higher_highs": 0,
        "higher_lows": 0
      }
    }
  },
  "janela_numero": 8,
  "timestamp_utc": "2026-08-07T22:38:00.000Z",
  "derivatives": {
    "BTCUSDT": {
      "funding_rate_percent": 0.01,
      "open_interest": 106934.462,
      "open_interest_usd": 6934209575.49,
      "long_short_ratio": 1.1,
      "longs_usd": 3631418636.94,
      "shorts_usd": 3302790938.55
    },
    "ETHUSDT": {
      "funding_rate_percent": 0,
      "open_interest": 2279900.074,
      "open_interest_usd": 4362909179.41,
      "long_short_ratio": 2.07,
      "longs_usd": 2940607898.43,
      "shorts_usd": 1422301280.98
    }
  },
  "fluxo_continuo": {
    "cvd": -25.8286,
    "whale_buy_volume": 0,
    "whale_sell_volume": 4,
    "whale_delta": -4,
    "bursts": {
      "count": 6,
      "max_burst_volume": 7.558
    },
    "sector_flow": {
      "retail": {
        "buy": 5.2115,
        "sell": 24.0704,
        "delta": -18.859
      },
      "mid": {
        "buy": 0.9922,
        "sell": 3.9619,
        "delta": -2.97
      },
      "whale": {
        "buy": 0,
        "sell": 4,
        "delta": -4
      }
    },
    "timestamp": "2026-08-07T22:38:00.000+00:00",
    "time_index": {
      "epoch_ms": 1786142280000,
      "timestamp_utc": "2026-08-07T22:38:00.000+00:00",
      "timestamp_ny": "2026-08-07T18:38:00.000-04:00",
      "timestamp_sp": "2026-08-07T19:38:00.000-03:00"
    },
    "order_flow": {
      "net_flow_1m": -221340.9762,
      "absorcao_1m": "Neutra",
      "buy_volume": 18798.35,
      "sell_volume": 546005.92,
      "total_volume": 564804.27,
      "buy_volume_btc": 0.29,
      "sell_volume_btc": 8.418,
      "total_volume_btc": 8.708,
      "whale_buy_volume_window": 0,
      "whale_sell_volume_window": 2,
      "whale_delta_window": -2,
      "flow_imbalance": -0.9334,
      "aggressive_buy_pct": 3.33,
      "aggressive_sell_pct": 96.67,
      "net_flow_5m": -943196.9173,
      "absorcao_5m": "Neutra",
      "net_flow_15m": -1676180.7386,
      "absorcao_15m": "Neutra",
      "computation_window_min": 1,
      "available_windows_min": [
        1,
        5,
        15
      ],
      "buy_sell_ratio": {
        "buy_sell_ratio": 0.03,
        "ratios": {
          "current": 0.0344,
          "imbalance_1m": -0.392,
          "imbalance_5m": -1.67,
          "imbalance_15m": -2.968
        },
        "sector_ratios": {
          "retail": 0.2165,
          "mid": 0.2504,
          "whale": 0
        },
        "pressure": "STRONG_SELL",
        "flow_trend": "accelerating_selling",
        "buy_volume": 0.29,
        "sell_volume": 8.418
      }
    },
    "tipo_absorcao": "Neutra",
    "participant_analysis": {
      "retail": {
        "volume_pct": 77.03,
        "direction": "SELL",
        "sentiment": "BEARISH",
        "composite_score": -0.874,
        "imbalance": -0.914
      },
      "whale": {
        "volume_pct": 22.97,
        "direction": "SELL",
        "sentiment": "BEARISH",
        "composite_score": -0.493,
        "imbalance": -1
      }
    },
    "liquidity_heatmap": {
      "clusters": [
        {
          "center": 64866.9237,
          "low": 64846,
          "high": 64882,
          "width": 36,
          "total_volume": 10.579,
          "buy_volume": 0.934,
          "sell_volume": 9.645,
          "imbalance": -8.711,
          "imbalance_ratio": -0.823,
          "trades_count": 2000,
          "avg_trade_size": 0.005,
          "recent_timestamp": 1786142290301,
          "recent_ts_ms": 1786142290301,
          "last_seen_ms": 1786142290301,
          "first_seen_ms": 1786142082302,
          "age_ms": 292.0,
          "cluster_duration_ms": 207999,
          "price_std": 11.3692,
          "volume_std": 0.039,
          "bin_threshold_usd": 194.6008
        }
      ],
      "resistances": [
        64866.9237
      ],
      "clusters_count": 1
    },
    "absorption_analysis": {
      "current_absorption": {
        "index": 0.3658,
        "classification": "WEAK_ABSORPTION",
        "label": "Neutra",
        "buyer_strength": 0.3,
        "seller_exhaustion": 9.3,
        "continuation_probability": 0.33,
        "delta_usd": -221340.976,
        "total_volume_usd": 564804.27,
        "flow_imbalance": -0.9334,
        "window_min": 1
      }
    }
  },
  "market_context": {
    "trading_session": "NY",
    "session_phase": "ACTIVE",
    "time_to_session_close": 12269,
    "day_of_week": 4,
    "is_holiday": 0,
    "market_hours_type": "EXTENDED"
  },
  "market_environment": {
    "volatility_regime": "NORMAL",
    "trend_direction": "UP",
    "market_structure": "RANGE_BOUND",
    "liquidity_environment": "NORMAL",
    "risk_sentiment": "BULLISH",
    "correlation_spy": 0.406,
    "correlation_dxy": -0.0867,
    "correlation_gold": 0.2401
  },
  "external_markets": {
    "FEAR_GREED": {
      "preco_atual": 29,
      "prev": 25,
      "movimento": "Alta",
      "classification": "Fear",
      "source": "alternative.me",
      "timestamp": "2026-08-07T19:30:17.741873"
    }
  },
  "contextual_snapshot": {
    "symbol": "BTCUSDT",
    "ohlc": {
      "open": 64872.9,
      "high": 64872.9,
      "low": 64846,
      "close": 64846,
      "open_time": 1786142221164,
      "close_time": 1786142278547,
      "vwap": 64863.4
    },
    "volume_total": 8.708,
    "volume_total_usdt": 564804,
    "volume_compra": 0.29,
    "volume_venda": 8.418,
    "num_trades": 1215,
    "delta_minimo": -8.287,
    "delta_maximo": 0.004,
    "delta_fechamento": -8.128,
    "reversao_desde_minimo": 0.16,
    "reversao_desde_maximo": 8.13,
    "poc_price": 64872.2,
    "poc_volume": 3.1,
    "poc_percentage": 35.6,
    "dwell_price": 64872.2,
    "dwell_seconds": 43,
    "dwell_location": "High",
    "trades_per_second": 21.17,
    "avg_trade_size": 0.007
  },
  "order_book_depth": {
    "L1": {
      "bids": 1981590.82,
      "asks": 136768.93,
      "flow_imbalance": 0.8709
    },
    "L5": {
      "bids": 1986905.97,
      "asks": 141241.48,
      "flow_imbalance": 0.8673
    },
    "L10": {
      "bids": 1996887.94,
      "asks": 147140.1,
      "flow_imbalance": 0.8627
    },
    "L25": {
      "bids": 2632095.03,
      "asks": 175207.77,
      "flow_imbalance": 0.8752
    },
    "total_depth_ratio": 15.02
  },
  "orderbook_quality": "live",
  "market_impact": {
    "slippage_matrix": {
      "100k_usd": {
        "buy": 0.05,
        "sell": 0.05
      },
      "1m_usd": {
        "buy": 5.85,
        "sell": 0.05
      }
    },
    "liquidity_score": 9.9985,
    "execution_quality": "EXCELLENT"
  },
  "institutional_analytics": {
    "status": "ok",
    "computed_at_ms": 1786142291224,
    "technical_extras": {
      "stoch_rsi": {
        "k": 0,
        "d": 2.79,
        "overbought": 0,
        "oversold": 1,
        "crossover": "none"
      },
      "williams_r": {
        "value": -99.62,
        "overbought": 0,
        "oversold": 1,
        "zone": "oversold",
        "source": "real"
      },
      "hurst_exponent": 0.3329,
      "shannon_entropy": 2.9283,
      "kalman_filter": {
        "kalman_price": 64933.92,
        "raw_price": 64872.9,
        "deviation_pct": -0.09,
        "trend_direction": "DOWN"
      },
      "regression_channel": {
        "slope_per_bar": -1.1398,
        "trend_price": 64897.3,
        "upper_1sd": 64913.73,
        "lower_1sd": 64880.86,
        "upper_2sd": 64930.17,
        "lower_2sd": 64864.43,
        "deviation_from_trend": -24.4,
        "position_in_channel": 0.1289
      },
      "dominant_cycles": {
        "dominant_cycles": [
          100,
          40,
          33.3
        ],
        "cycle_strengths": [
          3043.32,
          1842.49,
          918.84
        ]
      },
      "fractal_dimension": 0.6245,
      "monte_carlo": {
        "median_price": 64842.06,
        "p10": 64785.73,
        "p25": 64813.86,
        "p75": 64874.94,
        "p90": 64904.98,
        "prob_up": 0.47,
        "horizon_bars": 12
      },
      "fair_value_gaps": [
        {
          "type": "BEARISH",
          "top": 64925,
          "bottom": 64892.3,
          "gap_size": 32.7,
          "gap_pct": 0.05
        }
      ],
      "market_structure": {
        "structure": "BEARISH",
        "bos_detected": 0,
        "last_swing_high": 64925,
        "last_swing_low": 64909.55,
        "higher_highs": 0,
        "higher_lows": 0
      }
    },
    "profile_analysis": {
      "poor_extremes": {
        "poor_high": {
          "detected": 1,
          "price": 64872.9,
          "volume_ratio": 4.768,
          "implication": "High likely to be revisited - unfinished auction"
        },
        "poor_low": {
          "detected": 0,
          "price": 64846,
          "volume_ratio": 0.29,
          "implication": "Low moderately tested - neutral"
        },
        "excess_high": 0,
        "excess_low": 0,
        "session_high": 64872.9,
        "session_low": 64846,
        "action_bias": "expect_retest_high",
        "status": "success"
      },
      "profile_shape": {
        "shape": "P",
        "implication": "Short covering rally - bearish bias expected",
        "trading_signal": "BEARISH_AFTER",
        "distribution": {
          "lower_third_pct": 23.8,
          "middle_third_pct": 23.6,
          "upper_third_pct": 52.6
        },
        "dominant_zone": "upper",
        "status": "success"
      },
      "no_mans_land": {
        "zones": [
          {
            "range_low": 64512,
            "range_high": 64810,
            "gap_size": 298,
            "gap_size_pct": 0.462,
            "risk": "LOW",
            "lvn_confirmed": 1,
            "lvns_in_zone": 9,
            "nearest_hvn_below": 64512,
            "nearest_hvn_above": 64810,
            "distance_from_price": 36,
            "distance_pct": 0.06,
            "direction": "below"
          }
        ],
        "price_in_no_mans_land": 0,
        "nearest_no_mans_land": {
          "range_low": 64512,
          "range_high": 64810,
          "gap_size": 298,
          "gap_size_pct": 0.462,
          "risk": "LOW",
          "lvn_confirmed": 1,
          "lvns_in_zone": 9,
          "nearest_hvn_below": 64512,
          "nearest_hvn_above": 64810,
          "distance_from_price": 36,
          "distance_pct": 0.06,
          "direction": "below"
        },
        "total_zones": 1,
        "max_gap_pct": 0.46,
        "status": "success"
      },
      "va_volume_pct": {
        "value_area_volume_pct": 100,
        "interpretation": "extremely_compressed",
        "breakout_risk": "VERY_HIGH",
        "volume_in_va": 1,
        "total_volume": 1,
        "compression_signal": 1
      },
      "volume_node_strength": {
        "scored_hvns": [
          {
            "price": 64304,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.8,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64306,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.8,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64314,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.9,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64320,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.9,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64322,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 26,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          }
        ],
        "scored_lvns": [
          {
            "price": 64396,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.5,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64399,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64404,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64423,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64436,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.8,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          }
        ],
        "total_hvns": 52,
        "total_lvns": 101,
        "avg_hvn_strength": 66.2,
        "avg_lvn_strength": 63.2,
        "status": "success"
      }
    },
    "flow_analysis": {
      "passive_aggressive": {
        "aggressive": {
          "buy_pct": 3.33,
          "sell_pct": 96.67,
          "net_pct": -93.34,
          "dominance": "sellers",
          "buy_volume": 0.29,
          "sell_volume": 8.418
        },
        "passive": {
          "dominance": "buyers",
          "inference": "from_orderbook_depth",
          "bid_depth": 3585137.77,
          "ask_depth": 307774.65,
          "bid_ratio": 0.92,
          "ob_imbalance": 0.842
        },
        "composite": {
          "agreement": 0,
          "signal": "sell_absorption",
          "interpretation": "Aggressive sellers hitting passive buy walls - potential reversal or breakdown",
          "conviction": "MEDIUM"
        },
        "status": "success"
      },
      "whale_accumulation": {
        "score": -37,
        "classification": "MILD_DISTRIBUTION",
        "bias": "DISTRIBUTING",
        "components": {
          "flow": {
            "score": -30,
            "max": 30,
            "detail": {
              "divergence": "aligned",
              "whale_delta": -4,
              "mid_delta": -2.97,
              "retail_delta": -18.859,
              "primary_delta": -4
            }
          },
          "depth": {
            "score": 16.84,
            "max": 20,
            "detail": {
              "bid_depth": 3585137.77,
              "ask_depth": 307774.65,
              "ratio": 0.84,
              "deep_confirmation": 0
            }
          },
          "absorption": {
            "score": -25,
            "max": 25,
            "detail": {
              "buyer_strength": 0.3,
              "seller_exhaustion": 9.3,
              "net_absorption": -9,
              "index": 0.3658,
              "label": "Neutra"
            }
          },
          "derivatives": {
            "score": 1.49,
            "max": 25,
            "detail": {
              "long_short_ratio": 1.1,
              "lsr_score": 1.49
            }
          }
        },
        "trend": {
          "direction": "increasing_distribution",
          "avg_score": -24.2,
          "recent_avg": -35.2,
          "momentum": -12.8,
          "samples": 8,
          "score_range": {
            "min": -49,
            "max": 14
          }
        },
        "status": "success"
      },
      "absorption_zones": {
        "total_zones": 0,
        "total_events": 0,
        "buy_zone_count": 0,
        "sell_zone_count": 0,
        "status": "no_events"
      }
    },
    "sr_analysis": {
      "defense_zones": {
        "buy_defense": [
          {
            "center": 64479.86,
            "range_low": 64376.37,
            "range_high": 64570.63,
            "strength": 62,
            "side": "buy",
            "sources": [
              "sr_level_ema_21_4h",
              "vp_hvn",
              "ema_ema_21_4h",
              "vp_val"
            ],
            "source_count": 4,
            "signals_in_zone": 5,
            "type": "confluence",
            "distance_from_price": 366.14,
            "distance_pct": 0.56
          },
          {
            "center": 64823.36,
            "range_low": 64732.52,
            "range_high": 64921.63,
            "strength": 62,
            "side": "buy",
            "sources": [
              "orderbook_bid_wall",
              "vp_hvn",
              "ema_ema_21_1h",
              "sr_level_ema_21_1h"
            ],
            "source_count": 4,
            "signals_in_zone": 7,
            "type": "confluence",
            "distance_from_price": 22.64,
            "distance_pct": 0.03
          },
          {
            "center": 64165.68,
            "range_low": 64100.95,
            "range_high": 64230.42,
            "strength": 46,
            "side": "buy",
            "sources": [
              "ema_ema_21_1d",
              "sr_level_vah_monthly"
            ],
            "source_count": 2,
            "signals_in_zone": 2,
            "type": "cluster",
            "distance_from_price": 680.32,
            "distance_pct": 1.05
          },
          {
            "center": 64336.25,
            "range_low": 64255.37,
            "range_high": 64442.63,
            "strength": 40,
            "side": "buy",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 9,
            "type": "cluster",
            "distance_from_price": 509.75,
            "distance_pct": 0.79
          }
        ],
        "sell_defense": [
          {
            "center": 65030.31,
            "range_low": 64944.37,
            "range_high": 65128.63,
            "strength": 54,
            "side": "sell",
            "sources": [
              "sr_level_poc_daily",
              "vp_poc",
              "vp_hvn"
            ],
            "source_count": 3,
            "signals_in_zone": 11,
            "type": "confluence",
            "distance_from_price": 184.31,
            "distance_pct": 0.28
          },
          {
            "center": 65346,
            "range_low": 65297.37,
            "range_high": 65394.63,
            "strength": 53,
            "side": "sell",
            "sources": [
              "sr_level_vah_daily",
              "vp_hvn",
              "vp_vah"
            ],
            "source_count": 3,
            "signals_in_zone": 3,
            "type": "confluence",
            "distance_from_price": 500,
            "distance_pct": 0.77
          },
          {
            "center": 64922.94,
            "range_low": 64831.37,
            "range_high": 64998.68,
            "strength": 49,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "ema_ema_21_15m",
              "sr_level_hvn_daily"
            ],
            "source_count": 3,
            "signals_in_zone": 8,
            "type": "confluence",
            "distance_from_price": 76.94,
            "distance_pct": 0.12
          },
          {
            "center": 65157.17,
            "range_low": 65064.37,
            "range_high": 65257.63,
            "strength": 41,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 13,
            "type": "cluster",
            "distance_from_price": 311.17,
            "distance_pct": 0.48
          },
          {
            "center": 65252.3,
            "range_low": 65164.37,
            "range_high": 65350.63,
            "strength": 40,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 11,
            "type": "cluster",
            "distance_from_price": 406.3,
            "distance_pct": 0.63
          }
        ],
        "total_zones": 9,
        "strongest_buy": {
          "center": 64479.86,
          "range_low": 64376.37,
          "range_high": 64570.63,
          "strength": 62,
          "side": "buy",
          "sources": [
            "sr_level_ema_21_4h",
            "vp_hvn",
            "ema_ema_21_4h",
            "vp_val"
          ],
          "source_count": 4,
          "signals_in_zone": 5,
          "type": "confluence",
          "distance_from_price": 366.14,
          "distance_pct": 0.56
        },
        "strongest_sell": {
          "center": 65030.31,
          "range_low": 64944.37,
          "range_high": 65128.63,
          "strength": 54,
          "side": "sell",
          "sources": [
            "sr_level_poc_daily",
            "vp_poc",
            "vp_hvn"
          ],
          "source_count": 3,
          "signals_in_zone": 11,
          "type": "confluence",
          "distance_from_price": 184.31,
          "distance_pct": 0.28
        },
        "defense_asymmetry": {
          "ratio": 0.89,
          "bias": "slight_sell_defense",
          "description": "Slightly more sell defense",
          "buy_total_strength": 210,
          "sell_total_strength": 237
        },
        "status": "success"
      }
    },
    "quality": {
      "calendar": {
        "day_of_week": "Friday",
        "day_of_week_num": 4,
        "is_us_holiday": 0,
        "is_weekend": 0,
        "is_pre_holiday": 0,
        "is_post_holiday": 0,
        "expected_liquidity": "NORMAL",
        "liquidity_warning": 0
      },
      "latency": {
        "latency_ms": 11450,
        "latency_category": "POOR",
        "data_freshness": "DELAYED",
        "is_acceptable": 0,
        "is_stale": 0
      },
      "spread_percentile": {
        "status": "ok",
        "current_spread": 0.1,
        "spread_percentile": 0,
        "spread_mean": 0.1,
        "spread_median": 0.1,
        "spread_std": 0,
        "spread_min": 0.1,
        "spread_max": 0.1,
        "spread_z_score": 0,
        "is_tight": 1,
        "is_wide": 0,
        "is_anomalous": 0,
        "liquidity_signal": "EXCELLENT",
        "samples": 8,
        "window_minutes": 1440
      },
      "anomalies": {
        "anomalies_detected": 1,
        "count": 2,
        "anomalies": [
          {
            "type": "FLOW_EXTREME_IMBALANCE",
            "severity": "HIGH",
            "value": -0.9334,
            "direction": "SELL",
            "description": "Extreme flow imbalance: -93.34% toward sellers"
          },
          {
            "type": "DEPTH_EXTREME_ASYMMETRY",
            "severity": "MEDIUM",
            "ratio": 11.65,
            "bid_depth": 3585137.77,
            "ask_depth": 307774.65,
            "direction": "BID_HEAVY",
            "description": "Order book depth ratio 11.65:1 is extreme"
          }
        ],
        "max_severity": "HIGH",
        "risk_elevated": 1,
        "types_found": [
          "DEPTH_EXTREME_ASYMMETRY",
          "FLOW_EXTREME_IMBALANCE"
        ],
        "summary": "2 anomalies detected (max severity: HIGH)"
      }
    },
    "candlestick_patterns": {
      "patterns_detected": 1,
      "patterns": [
        {
          "name": "doji",
          "type": "neutral",
          "confidence": 0.65,
          "candles_used": 1,
          "implication": "Indecision - watch next candle for direction"
        }
      ],
      "dominant_signal": "neutral",
      "max_confidence": 0.65,
      "bullish_count": 0,
      "bearish_count": 0,
      "neutral_count": 1
    }
  },
  "sequence_id": 8,
  "primary_exchange": "BINANCE",
  "data_feed_type": "WEBSOCKET_L2",
  "data_quality_score": 8.75,
  "completeness_pct": 100,
  "reliability_score": 7.5,
  "bid": 64819.3,
  "ask": 64819.4,
  "tick_direction": -1,
  "twap": 64888.2,
  "pivot_points": {
    "daily": {
      "pivot": 64971.33,
      "r1": 65420.66,
      "r2": 65795.33,
      "r3": 66619.33,
      "s1": 64596.66,
      "s2": 64147.33,
      "s3": 63323.33,
      "vah": 65346,
      "val": 64522,
      "poc": 65046
    },
    "weekly": {
      "pivot": 62830.33,
      "r1": 63346.66,
      "r2": 64083.33,
      "r3": 65336.33,
      "s1": 62093.66,
      "s2": 61577.33,
      "s3": 60324.33,
      "vah": 63567,
      "val": 62314,
      "poc": 62610
    },
    "monthly": {
      "pivot": 63185.67,
      "r1": 64582.34,
      "r2": 65610.67,
      "r3": 68035.67,
      "s1": 62157.34,
      "s2": 60760.67,
      "s3": 58335.67,
      "vah": 64214,
      "val": 61789,
      "poc": 63554
    }
  },
  "immediate_support": [
    64823.36,
    64522,
    64479.86,
    64214,
    64165.68
  ],
  "support_strength": [
    62,
    95,
    62,
    72.2,
    46
  ],
  "immediate_resistance": [
    64922.94,
    64971.33,
    65030.31,
    65346
  ],
  "resistance_strength": [
    49,
    98.1,
    54,
    92.3
  ],
  "volatility_metrics": {
    "realized_vol_24h": 0.0184,
    "realized_vol_7d": 0.0487,
    "volatility_regime": "NORMAL"
  },
  "order_flow_extended": {
    "passive_buy_pct": 92.1,
    "passive_sell_pct": 7.9
  },
  "whale_activity": {
    "large_orders_1h": [
      {
        "size": 1,
        "price": 64896.34,
        "side": "SELL",
        "timestamp_ms": 1786141945725
      },
      {
        "size": 1,
        "price": 64883.14,
        "side": "SELL",
        "timestamp_ms": 1786142044496
      },
      {
        "size": 1,
        "price": 64870.63,
        "side": "SELL",
        "timestamp_ms": 1786142264045
      },
      {
        "size": 1,
        "price": 64853.03,
        "side": "SELL",
        "timestamp_ms": 1786142272654
      }
    ],
    "iceberg_activity": 1,
    "hidden_orders_detected": 0
  },
  "technical_indicators_extended": {
    "cci_1h": 100.2,
    "cci_signal": "OVERBOUGHT",
    "stochastic": {
      "k": 0,
      "d": 2.79,
      "signal": "OVERSOLD",
      "source": "real"
    },
    "williams_r": {
      "value": -99.62,
      "overbought": 0,
      "oversold": 1,
      "zone": "oversold",
      "source": "real"
    },
    "hurst_exponent": 0.3329,
    "shannon_entropy": 2.9283,
    "fractal_dimension": 0.6245,
    "kalman_filter": {
      "kalman_price": 64933.92,
      "raw_price": 64872.9,
      "deviation_pct": -0.09,
      "trend_direction": "DOWN"
    },
    "regression_channel": {
      "slope_per_bar": -1.1398,
      "trend_price": 64897.3,
      "upper_1sd": 64913.73,
      "lower_1sd": 64880.86,
      "upper_2sd": 64930.17,
      "lower_2sd": 64864.43,
      "deviation_from_trend": -24.4,
      "position_in_channel": 0.1289
    },
    "dominant_cycles": {
      "dominant_cycles": [
        100,
        40,
        33.3
      ],
      "cycle_strengths": [
        3043.32,
        1842.49,
        918.84
      ]
    },
    "monte_carlo": {
      "median_price": 64842.06,
      "p10": 64785.73,
      "p25": 64813.86,
      "p75": 64874.94,
      "p90": 64904.98,
      "prob_up": 0.47,
      "horizon_bars": 12
    },
    "garch_forecast_1h": 0.0014
  },
  "backup_exchanges": [
    "COINBASE",
    "KRAKEN",
    "OKX"
  ],
  "alerts": {
    "active_alerts": [
      {
        "type": "DEPTH_DIVERGENCE",
        "level": 0.842,
        "severity": "MEDIUM",
        "probability": 0.84,
        "action": "PREPARE_LONG",
        "description": "Orderbook BID_HEAVY: imbalance=0.842"
      }
    ],
    "alert_count": 1,
    "max_severity": "MEDIUM"
  },
  "regime_analysis": {
    "current_regime": "BREAKOUT",
    "regime_probabilities": {
      "trending": 0.238,
      "mean_reverting": 0.333,
      "breakout": 0.429
    },
    "regime_change_probability": 0.5,
    "expected_regime_duration": "5m-30m",
    "avg_adx": 26
  },
  "data_reliability": {
    "has_options_data": 0,
    "onchain_coverage": "full",
    "latency_acceptable": 1,
    "price_targets_available": 1
  },
  "event_id": "46907d43",
  "timestamp_ny": "2026-08-07T18:38:00.000-04:00",
  "timestamp_sp": "2026-08-07T19:38:00.000-03:00",
  "_log_id": "46907d43"
}
----------------------------------------------------------------------------------------------------
EVENTO: AI_ANALYSIS | SYMBOL: BTCUSDT
UTC: 2026-08-07 22:38:13 UTC
NY:  2026-08-07 18:38:13 EST/EDT
SP:  2026-08-07 19:38:13 BRT
----------------------------------------------------------------------------------------------------
{
  "tipo_evento": "AI_ANALYSIS",
  "symbol": "BTCUSDT",
  "timestamp_ms": 1786142280000,
  "anchor_price": 64846,
  "anchor_window_id": 8,
  "ai_result": {
    "sentiment": "bearish",
    "confidence": 0.66,
    "action": "wait",
    "rationale": "Os 15m apontam queda com fluxo de venda intenso e desequilíbrio de ordem negativo, enquanto o 4h ainda indica tendência de alta, sugerindo uma retração dentro d",
    "region_type": "retração",
    "_is_fallback": 0,
    "_is_valid": 1
  },
  "ai_payload": {
    "symbol": "BTCUSDT",
    "epoch_ms": 1786142280000,
    "trigger": "AT",
    "price": {
      "c": 64846,
      "o": 64873,
      "h": 64873,
      "vw": 64863,
      "sh": "P",
      "auc": "expect_retest_high",
      "ph": 1,
      "brk_risk": "V_HI"
    },
    "regime": {
      "cs": "BULL",
      "cf": 0.8,
      "v": "NOR",
      "mode": "BRK",
      "dom": "4h",
      "bull%": 90,
      "bear%": 10
    },
    "qual": {
      "lat": "POOR",
      "ms": 11450
    },
    "flow": {
      "d1": "-221K",
      "delta": -8.128,
      "vol": 8.708,
      "buy_pct": 3,
      "ti": 20.6,
      "trs": -223,
      "obs": -2.838,
      "sf_w": -4,
      "sf_r": -18.859,
      "d5": "-943K",
      "d15": "-1.7M",
      "cvd": -25.8,
      "imb": -0.93,
      "ab": 3,
      "bsr": 0.03,
      "pa": "sell_absor",
      "conv": "M",
      "abs_buy_str": 0.3,
      "abs_sell_exh": 9.3,
      "abs_cont": 0.33
    },
    "ob": {
      "b": "3.6M",
      "a": "308K",
      "imb": 0.84,
      "bias": "BUY",
      "t5": 0.87,
      "spread_pct": 0,
      "slip_b": 5,
      "slip_s": 5
    },
    "tf": {
      "15m": {
        "t": "DN",
        "rsi": 38,
        "macd": [
          13,
          17
        ],
        "adx": 24,
        "atr": 70,
        "r": "RNG"
      },
      "1h": {
        "t": "UP",
        "rsi": 53,
        "macd": [
          100,
          98
        ],
        "adx": 24,
        "atr": 224,
        "r": "RNG"
      },
      "4h": {
        "t": "UP",
        "rsi": 62,
        "macd": [
          275,
          257
        ],
        "adx": 30,
        "atr": 495,
        "r": "RNG"
      },
      "1d": {
        "t": "UP",
        "rsi": 59,
        "macd": [
          71,
          37
        ],
        "adx": 15,
        "atr": 1370,
        "r": "MNP"
      }
    },
    "sr": {
      "r1": [
        65030,
        54
      ],
      "r1_dist": 184,
      "r1_conf": 3,
      "r2": [
        65346,
        53
      ],
      "r2_dist": 500,
      "r2_conf": 3,
      "s1": [
        64480,
        62
      ],
      "s1_dist": 366,
      "s1_conf": 4,
      "s2": [
        64823,
        62
      ],
      "s2_dist": 23,
      "s2_conf": 4,
      "def_bias": "slight_sel"
    },
    "w": {
      "s": -37,
      "c": "MD"
    },
    "ext": {
      "cci": "OB",
      "stoch": 0,
      "stoch_sig": "OS",
      "wr": -100,
      "garch": 0,
      "hurst": 0.33,
      "entropy": 2.93,
      "fd": 0.62,
      "kalman": {
        "kp": 64933.92,
        "dev": -0.094,
        "dir": "DOWN"
      },
      "reg": {
        "sl": -1.1398,
        "pos": 0.1289,
        "dev": -24.4
      },
      "mc": {
        "pu": 0.466,
        "p10": 64785.73,
        "p90": 64904.98
      },
      "cycles": [
        100,
        40
      ],
      "smc": {
        "fvg": 1,
        "fvg_last": "BE",
        "struct": "BEAR",
        "bos": 0
      }
    },
    "ctx": {
      "ses": "NY",
      "fg": 29,
      "poc": 65046,
      "val": 64522,
      "vah": 65346,
      "lsr": 1.1,
      "eth_lsr": 2.07,
      "oi": 107,
      "fr": 0.0063,
      "longs": "+3631.4M",
      "shorts": "+3302.8M",
      "eth7": 0.8,
      "dxy30": -0.06
    },
    "ofi": {
      "score": -0.933,
      "dir": "SELL",
      "src": "order_flow"
    },
    "vwap": {
      "dev": -0.027,
      "side": "below",
      "sig": "fair",
      "src": "ohlc"
    },
    "iceberg": {
      "det": 1,
      "src": "whale_activity"
    },
    "liq": [
      {
        "p": 64867,
        "side": "sell",
        "vol": 10.58
      }
    ],
    "cvd_div": {
      "det": 1,
      "type": "bearish_div",
      "src": "inferred"
    },
    "mr": {
      "score": 0.248,
      "sig": "stretched_bear",
      "src": "inferred"
    },
    "summary": {
      "flow": {
        "bias": "SELL",
        "type": "mixed",
        "actor": "retail",
        "conf": "M",
        "note": "Fluxo misto sem dominância clara. (varejo vendedor). prob. continuação 33%. [imbalance extremo de venda]"
      },
      "sr": {
        "nearest": "resistance",
        "compressed": 0,
        "conf_bias": "NEUTRAL",
        "note": "Resistência mais próxima em 65030 (força 54, confluência 3 fontes, dist 184 pts (0.8 ATR))",
        "r1_dist_atr": 0.82,
        "s1_dist_atr": 1.63
      },
      "regime": {
        "label": "Breakout",
        "strategies": [
          "aguardar confirmação de rompimento",
          "entrar no reteste do nível rompido",
          "usar stop apertado acima/abaixo do nível",
          "monitorar volume de confirmação"
        ],
        "avoid": [
          "entrar antes da confirmação",
          "ignorar falsos rompimentos",
          "operar range enquanto houver BRK ativo"
        ],
        "duration": "minutos a horas — confirmar rápido",
        "note": "Regime Breakout com consenso de alta (confiança alta: 80%). dominado pelo 4h."
      },
      "institutional": {
        "auction_state": "Leilão incompleto — máxima deve ser revisitada",
        "whale_bias": "DISTRIBUTING",
        "profile_bias": "NEUTRAL",
        "unfinished": [
          "high"
        ],
        "alignment": "BEAR_ALIGNED",
        "note": "Leilão incompleto — máxima deve ser revisitada. Extremo(s) incompleto(s): high — reteste esperado. Risco de breakout da Value Area muito alto. Sinais institucionais alinhados para baixa."
      },
      "quality": {
        "reliable": 1,
        "confidence_cap": 1,
        "note": "Dados em tempo real sem anomalias. Análise com confiança plena."
      }
    },
    "tipo_evento": "ANALYSIS_TRIGGER",
    "descricao": "Evento automático para análise da IA"
  },
  "epoch_ms": 1786142293862,
  "event_id": "85c52eed",
  "data_context": "real_time",
  "timestamp_utc": "2026-08-07T22:38:13.862+00:00",
  "timestamp_ny": "2026-08-07T18:38:13.862-04:00",
  "timestamp_sp": "2026-08-07T19:38:13.862-03:00",
  "timestamp": "2026-08-07T22:38:13.862+00:00",
  "_log_id": "85c52eed"
}

----------------------------------------------------------------------------------------------------
 # Janela 9
----------------------------------------------------------------------------------------------------
EVENTO: ANALYSIS_TRIGGER | SYMBOL: BTCUSDT | JANELA: 9
UTC: 2026-08-07 22:39:00 UTC
NY:  2026-08-07 18:39:00 EST/EDT
SP:  2026-08-07 19:39:00 BRT
----------------------------------------------------------------------------------------------------
{
  "is_signal": 1,
  "tipo_evento": "ANALYSIS_TRIGGER",
  "descricao": "Evento automático para análise da IA",
  "symbol": "BTCUSDT",
  "raw_event": {
    "delta": 3.664,
    "volume_total": 5.03,
    "volume_compra": 4.347,
    "volume_venda": 0.683,
    "preco_fechamento": 64872,
    "advanced_analysis": {
      "symbol": "BTCUSDT",
      "price": 64872,
      "volume": 5.03,
      "adaptive_thresholds": {
        "current_volatility": 0.03,
        "volatility_factor": 2.0,
        "absorption_threshold": 0.3,
        "flow_threshold": 0.2
      },
      "onchain_metrics": {
        "difficulty": 126.23,
        "active_addresses": 717974,
        "mempool_size": 87732,
        "mempool_vsize_mb": 43.83,
        "mempool_total_fee_btc": 0.1697,
        "fees_fastest_sat_vb": 5,
        "fees_half_hour_sat_vb": 4,
        "fees_hour_sat_vb": 3,
        "fees_economy_sat_vb": 2,
        "difficulty_adjustment": {
          "progress_pct": 93.4,
          "estimated_change_pct": 0.72,
          "remaining_blocks": 133,
          "remaining_time_ms": 79270394,
          "previous_retarget_pct": -0.74
        },
        "minutes_between_blocks": 8.4423,
        "total_btc_sent_24h": 807619.08,
        "trade_volume_btc_24h": 3265.64,
        "data_source": "blockchain.info+mempool.space",
        "is_real_data": 1,
        "requires_paid_api": [
          "exchange_netflow",
          "whale_transactions",
          "exchange_reserves",
          "sopr"
        ]
      }
    }
  },
  "resultado_da_batalha": "N/A",
  "delta": 3.664,
  "volume_total": 5.03,
  "volume_compra": 4.347,
  "volume_venda": 0.683,
  "preco_fechamento": 64872,
  "timestamp": "2026-08-07T22:39:07Z",
  "epoch_ms": 1786142340000,
  "ml_features": {
    "price_features": {
      "returns_1": 1.5e-07,
      "volatility_1": 0.0,
      "returns_5": 0.0,
      "volatility_5": 1e-07,
      "returns_15": 0.0,
      "volatility_15": 8e-08,
      "momentum_score": -0.70710678,
      "volatility_1h": 0.0023
    },
    "volume_features": {
      "volume_sma_ratio": 4.662,
      "volume_momentum": -0.374,
      "buy_sell_pressure": 0.7285,
      "liquidity_gradient": -0.3742797
    },
    "microstructure": {
      "order_book_slope": 0.457162,
      "flow_imbalance": 0.7284,
      "tick_rule_sum": 171,
      "trade_intensity": 35,
      "trade_intensity_v2": 17.2667
    },
    "cross_asset": {
      "btc_eth_corr_7d": 0.8212,
      "btc_eth_corr_30d": 0.8599,
      "btc_dxy_corr_30d": -0.0603,
      "btc_dxy_corr_90d": -0.0669,
      "btc_ndx_corr_30d": 0.4146,
      "dxy_return_5d": -0.3561,
      "dxy_return_20d": -1.6548,
      "btc_dxy_correlation_stability": 0.0066,
      "btc_dxy_inverse_strength": 0.0636,
      "dxy_momentum": -1.29867555,
      "vix_current": 14.9,
      "us10y_yield": 4.66,
      "btc_dominance": 17.0386,
      "btc_dominance_change_7d": 0,
      "eth_dominance": 9.0982,
      "gold_price": 4342.3475,
      "oil_price": 77.08,
      "macro_regime": "RISK_ON",
      "correlation_regime": "DECORRELATED"
    }
  },
  "orderbook_data": {
    "mid": 64843.05,
    "spread": 0.1,
    "spread_percent": 0,
    "bid_depth_usd": 810457.31,
    "ask_depth_usd": 2086052.74,
    "imbalance": -0.44,
    "flow_imbalance": -0.4404,
    "volume_ratio": 0.389,
    "pressure": -0.4404,
    "consolidated_bias_score": 0.3067,
    "spread_bps": 0.0154,
    "is_valid": 1,
    "data_source": "live",
    "spread_volatility": 0.0
  },
  "historical_vp": {
    "daily": {
      "poc": 65046,
      "vah": 65346,
      "val": 64522,
      "status": "success"
    },
    "weekly": {
      "poc": 62610,
      "vah": 63567,
      "val": 62314,
      "status": "success"
    },
    "monthly": {
      "poc": 63554,
      "vah": 64214,
      "val": 61789,
      "status": "success"
    }
  },
  "multi_tf": {
    "15m": {
      "tendencia": "Baixa",
      "preco_atual": 64856.87,
      "mme_21": 64924.03,
      "atr": 73.63,
      "regime": "Range",
      "rsi_short": 35.96,
      "rsi_long": 42.76,
      "macd": 11.4474,
      "macd_signal": 16.415,
      "adx": 23.83,
      "realized_vol": 0.0014
    },
    "1h": {
      "tendencia": "Alta",
      "preco_atual": 64866.66,
      "mme_21": 64788.32,
      "atr": 227.14,
      "regime": "Range",
      "rsi_short": 52.19,
      "rsi_long": 54.07,
      "macd": 99.9296,
      "macd_signal": 98.2155,
      "adx": 23.82,
      "realized_vol": 0.0023
    },
    "4h": {
      "tendencia": "Alta",
      "preco_atual": 64872.89,
      "mme_21": 64460.43,
      "atr": 495.32,
      "regime": "Range",
      "rsi_short": 61.98,
      "rsi_long": 61.12,
      "macd": 275.0187,
      "macd_signal": 256.7251,
      "adx": 30.01,
      "realized_vol": 0.0058
    },
    "1d": {
      "tendencia": "Alta",
      "preco_atual": 64924.96,
      "mme_21": 64149.58,
      "atr": 1369.69,
      "regime": "Manipulação",
      "rsi_short": 59.35,
      "rsi_long": 55.96,
      "macd": 71.3764,
      "macd_signal": 36.7677,
      "adx": 15.05,
      "realized_vol": 0.0184
    }
  },
  "data_context": "real_time",
  "price_targets": [
    {
      "level": 66818.16,
      "confidence": 0.3,
      "source": "fallback_r1",
      "weight": 0.1
    },
    {
      "level": 68764.32,
      "confidence": 0.2,
      "source": "fallback_r2",
      "weight": 0.05
    },
    {
      "level": 62925.84,
      "confidence": 0.3,
      "source": "fallback_s1",
      "weight": 0.1
    },
    {
      "level": 60979.68,
      "confidence": 0.2,
      "source": "fallback_s2",
      "weight": 0.05
    },
    {
      "level": 64872,
      "confidence": 0.4,
      "source": "fallback_poc",
      "weight": 0.15
    }
  ],
  "pattern_recognition": {
    "smart_money": {
      "fair_value_gaps": [
        {
          "type": "BEARISH",
          "top": 64925,
          "bottom": 64892.3,
          "gap_size": 32.7,
          "gap_pct": 0.05
        }
      ],
      "market_structure": {
        "structure": "BEARISH",
        "bos_detected": 0,
        "last_swing_high": 64925,
        "last_swing_low": 64909.55,
        "higher_highs": 0,
        "higher_lows": 0
      }
    }
  },
  "janela_numero": 9,
  "timestamp_utc": "2026-08-07T22:39:00.000Z",
  "derivatives": {
    "BTCUSDT": {
      "funding_rate_percent": 0.01,
      "open_interest": 106934.462,
      "open_interest_usd": 6934209575.49,
      "long_short_ratio": 1.1,
      "longs_usd": 3631418636.94,
      "shorts_usd": 3302790938.55
    },
    "ETHUSDT": {
      "funding_rate_percent": 0,
      "open_interest": 2279900.074,
      "open_interest_usd": 4362909179.41,
      "long_short_ratio": 2.07,
      "longs_usd": 2940607898.43,
      "shorts_usd": 1422301280.98
    }
  },
  "fluxo_continuo": {
    "cvd": -22.3837,
    "whale_buy_volume": 0,
    "whale_sell_volume": 4,
    "whale_delta": -4,
    "bursts": {
      "count": 7,
      "max_burst_volume": 7.558
    },
    "sector_flow": {
      "retail": {
        "buy": 8.9613,
        "sell": 24.9051,
        "delta": -15.944
      },
      "mid": {
        "buy": 1.522,
        "sell": 3.9619,
        "delta": -2.44
      },
      "whale": {
        "buy": 0,
        "sell": 4,
        "delta": -4
      }
    },
    "timestamp": "2026-08-07T22:39:00.000+00:00",
    "time_index": {
      "epoch_ms": 1786142340000,
      "timestamp_utc": "2026-08-07T22:39:00.000+00:00",
      "timestamp_ny": "2026-08-07T18:39:00.000-04:00",
      "timestamp_sp": "2026-08-07T19:39:00.000-03:00"
    },
    "order_flow": {
      "net_flow_1m": 49772.295,
      "absorcao_1m": "Neutra",
      "buy_volume": 281959.2,
      "sell_volume": 44302.3,
      "total_volume": 326261.5,
      "buy_volume_btc": 4.347,
      "sell_volume_btc": 0.683,
      "total_volume_btc": 5.03,
      "whale_buy_volume_window": 0,
      "whale_sell_volume_window": 0,
      "whale_delta_window": 0,
      "flow_imbalance": 0.7284,
      "aggressive_buy_pct": 86.42,
      "aggressive_sell_pct": 13.58,
      "net_flow_5m": -341706.2992,
      "absorcao_5m": "Absorção de Compra",
      "net_flow_15m": -1452768.9282,
      "absorcao_15m": "Neutra",
      "computation_window_min": 1,
      "available_windows_min": [
        1,
        5,
        15
      ],
      "buy_sell_ratio": {
        "buy_sell_ratio": 6.37,
        "ratios": {
          "current": 6.3658,
          "imbalance_1m": 0.153,
          "imbalance_5m": -1.047,
          "imbalance_15m": -4.453
        },
        "sector_ratios": {
          "retail": 0.3598,
          "mid": 0.3842,
          "whale": 0
        },
        "pressure": "STRONG_BUY",
        "flow_trend": "short_term_reversal_to_buy",
        "buy_volume": 4.347,
        "sell_volume": 0.683
      }
    },
    "tipo_absorcao": "Neutra",
    "participant_analysis": {
      "retail": {
        "volume_pct": 89.47,
        "direction": "BUY",
        "sentiment": "BULLISH",
        "composite_score": 0.836,
        "imbalance": 0.697
      },
      "mid": {
        "volume_pct": 10.53,
        "direction": "BUY",
        "sentiment": "BULLISH",
        "composite_score": 0.442,
        "imbalance": 1
      }
    },
    "liquidity_heatmap": {
      "clusters": [
        {
          "center": 64858.2875,
          "low": 64846,
          "high": 64872.89,
          "width": 26.89,
          "total_volume": 11.009,
          "buy_volume": 4.513,
          "sell_volume": 6.496,
          "imbalance": -1.984,
          "imbalance_ratio": -0.18,
          "trades_count": 2000,
          "avg_trade_size": 0.006,
          "recent_timestamp": 1786142345290,
          "recent_ts_ms": 1786142345290,
          "last_seen_ms": 1786142345290,
          "first_seen_ms": 1786142264045,
          "age_ms": 303.0,
          "cluster_duration_ms": 81245,
          "price_std": 8.5101,
          "volume_std": 0.04,
          "bin_threshold_usd": 194.5749
        }
      ],
      "resistances": [
        64858.2875
      ],
      "clusters_count": 1
    },
    "absorption_analysis": {
      "current_absorption": {
        "index": 0.1111,
        "classification": "WEAK_ABSORPTION",
        "label": "Neutra",
        "buyer_strength": 8.6,
        "seller_exhaustion": 7.3,
        "continuation_probability": 0.1,
        "delta_usd": 49772.295,
        "total_volume_usd": 326261.5,
        "flow_imbalance": 0.7284,
        "window_min": 1
      }
    }
  },
  "market_context": {
    "trading_session": "NY",
    "session_phase": "ACTIVE",
    "time_to_session_close": 12269,
    "day_of_week": 4,
    "is_holiday": 0,
    "market_hours_type": "EXTENDED"
  },
  "market_environment": {
    "volatility_regime": "NORMAL",
    "trend_direction": "UP",
    "market_structure": "RANGE_BOUND",
    "liquidity_environment": "NORMAL",
    "risk_sentiment": "BULLISH",
    "correlation_spy": 0.406,
    "correlation_dxy": -0.0867,
    "correlation_gold": 0.2401
  },
  "external_markets": {
    "FEAR_GREED": {
      "preco_atual": 29,
      "prev": 25,
      "movimento": "Alta",
      "classification": "Fear",
      "source": "alternative.me",
      "timestamp": "2026-08-07T19:30:17.741873"
    }
  },
  "contextual_snapshot": {
    "symbol": "BTCUSDT",
    "ohlc": {
      "open": 64846,
      "high": 64872,
      "low": 64846,
      "close": 64872,
      "open_time": 1786142280489,
      "close_time": 1786142339913,
      "vwap": 64859
    },
    "volume_total": 5.03,
    "volume_total_usdt": 326262,
    "volume_compra": 4.347,
    "volume_venda": 0.683,
    "num_trades": 1025,
    "delta_minimo": -0.001,
    "delta_maximo": 4.223,
    "delta_fechamento": 3.664,
    "reversao_desde_minimo": 3.66,
    "reversao_desde_maximo": 0.56,
    "poc_price": 64854.5,
    "poc_volume": 1.15,
    "poc_percentage": 23,
    "dwell_price": 64871.4,
    "dwell_seconds": 20,
    "dwell_location": "High",
    "trades_per_second": 17.25,
    "avg_trade_size": 0.005
  },
  "order_book_depth": {
    "L1": {
      "bids": 448713.56,
      "asks": 1244987.52,
      "flow_imbalance": -0.4701
    },
    "L5": {
      "bids": 544745.88,
      "asks": 1307950.33,
      "flow_imbalance": -0.4119
    },
    "L10": {
      "bids": 574054.52,
      "asks": 1333434.06,
      "flow_imbalance": -0.3981
    },
    "L25": {
      "bids": 667878.64,
      "asks": 1564284.34,
      "flow_imbalance": -0.4016
    },
    "total_depth_ratio": 0.43
  },
  "orderbook_quality": "live",
  "market_impact": {
    "slippage_matrix": {
      "100k_usd": {
        "buy": 0.05,
        "sell": 0.05
      },
      "1m_usd": {
        "buy": 0.05,
        "sell": 6.25
      }
    },
    "liquidity_score": 9.9985,
    "execution_quality": "EXCELLENT"
  },
  "institutional_analytics": {
    "status": "ok",
    "computed_at_ms": 1786142347270,
    "technical_extras": {
      "stoch_rsi": {
        "k": 0,
        "d": 1.4,
        "overbought": 0,
        "oversold": 1,
        "crossover": "none"
      },
      "williams_r": {
        "value": -100,
        "overbought": 0,
        "oversold": 1,
        "zone": "oversold",
        "source": "real"
      },
      "hurst_exponent": 0.3291,
      "shannon_entropy": 2.9377,
      "kalman_filter": {
        "kalman_price": 64931.19,
        "raw_price": 64846,
        "deviation_pct": -0.13,
        "trend_direction": "DOWN"
      },
      "regression_channel": {
        "slope_per_bar": -1.1725,
        "trend_price": 64893.66,
        "upper_1sd": 64910.77,
        "lower_1sd": 64876.54,
        "upper_2sd": 64927.89,
        "lower_2sd": 64859.42,
        "deviation_from_trend": -47.66,
        "position_in_channel": 0
      },
      "dominant_cycles": {
        "dominant_cycles": [
          100,
          40,
          33.3
        ],
        "cycle_strengths": [
          3066.18,
          1876.06,
          949.71
        ]
      },
      "fractal_dimension": 0.6255,
      "monte_carlo": {
        "median_price": 64866,
        "p10": 64809.1,
        "p25": 64837.51,
        "p75": 64899.21,
        "p90": 64929.56,
        "prob_up": 0.44,
        "horizon_bars": 12
      },
      "fair_value_gaps": [
        {
          "type": "BEARISH",
          "top": 64925,
          "bottom": 64892.3,
          "gap_size": 32.7,
          "gap_pct": 0.05
        }
      ],
      "market_structure": {
        "structure": "BEARISH",
        "bos_detected": 0,
        "last_swing_high": 64925,
        "last_swing_low": 64909.55,
        "higher_highs": 0,
        "higher_lows": 0
      }
    },
    "profile_analysis": {
      "poor_extremes": {
        "poor_high": {
          "detected": 1,
          "price": 64872,
          "volume_ratio": 1.86,
          "implication": "High likely to be revisited - unfinished auction"
        },
        "poor_low": {
          "detected": 1,
          "price": 64846,
          "volume_ratio": 1.203,
          "implication": "Low likely to be revisited - unfinished auction"
        },
        "excess_high": 0,
        "excess_low": 0,
        "session_high": 64872,
        "session_low": 64846,
        "action_bias": "expect_retest_both",
        "status": "success"
      },
      "profile_shape": {
        "shape": "B",
        "implication": "Double distribution - breakout imminent",
        "trading_signal": "BREAKOUT_EXPECTED",
        "distribution": {
          "lower_third_pct": 37.9,
          "middle_third_pct": 32.2,
          "upper_third_pct": 29.9
        },
        "dominant_zone": "lower",
        "status": "success"
      },
      "no_mans_land": {
        "zones": [
          {
            "range_low": 64512,
            "range_high": 64810,
            "gap_size": 298,
            "gap_size_pct": 0.462,
            "risk": "LOW",
            "lvn_confirmed": 1,
            "lvns_in_zone": 9,
            "nearest_hvn_below": 64512,
            "nearest_hvn_above": 64810,
            "distance_from_price": 62,
            "distance_pct": 0.1,
            "direction": "below"
          }
        ],
        "price_in_no_mans_land": 0,
        "nearest_no_mans_land": {
          "range_low": 64512,
          "range_high": 64810,
          "gap_size": 298,
          "gap_size_pct": 0.462,
          "risk": "LOW",
          "lvn_confirmed": 1,
          "lvns_in_zone": 9,
          "nearest_hvn_below": 64512,
          "nearest_hvn_above": 64810,
          "distance_from_price": 62,
          "distance_pct": 0.1,
          "direction": "below"
        },
        "total_zones": 1,
        "max_gap_pct": 0.46,
        "status": "success"
      },
      "va_volume_pct": {
        "value_area_volume_pct": 100,
        "interpretation": "extremely_compressed",
        "breakout_risk": "VERY_HIGH",
        "volume_in_va": 1,
        "total_volume": 1,
        "compression_signal": 1
      },
      "volume_node_strength": {
        "scored_hvns": [
          {
            "price": 64304,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64306,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64314,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64320,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64322,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.8,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          }
        ],
        "scored_lvns": [
          {
            "price": 64423,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.5,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64436,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64440,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64447,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64448,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          }
        ],
        "total_hvns": 52,
        "total_lvns": 101,
        "avg_hvn_strength": 66.3,
        "avg_lvn_strength": 63.1,
        "status": "success"
      }
    },
    "flow_analysis": {
      "passive_aggressive": {
        "aggressive": {
          "buy_pct": 86.42,
          "sell_pct": 13.58,
          "net_pct": 72.84,
          "dominance": "buyers",
          "buy_volume": 4.347,
          "sell_volume": 0.683
        },
        "passive": {
          "dominance": "sellers",
          "inference": "from_orderbook_depth",
          "bid_depth": 810457.31,
          "ask_depth": 2086052.74,
          "bid_ratio": 0.28,
          "ob_imbalance": -0.44
        },
        "composite": {
          "agreement": 0,
          "signal": "buy_absorption",
          "interpretation": "Aggressive buyers hitting passive sell walls - potential reversal or breakout",
          "conviction": "MEDIUM"
        },
        "status": "success"
      },
      "whale_accumulation": {
        "score": -33,
        "classification": "MILD_DISTRIBUTION",
        "bias": "DISTRIBUTING",
        "components": {
          "flow": {
            "score": -30,
            "max": 30,
            "detail": {
              "divergence": "aligned",
              "whale_delta": -4,
              "mid_delta": -2.44,
              "retail_delta": -15.944,
              "primary_delta": -4
            }
          },
          "depth": {
            "score": -8.81,
            "max": 20,
            "detail": {
              "bid_depth": 810457.31,
              "ask_depth": 2086052.74,
              "ratio": -0.44,
              "deep_confirmation": 0
            }
          },
          "absorption": {
            "score": 3.9,
            "max": 25,
            "detail": {
              "buyer_strength": 8.6,
              "seller_exhaustion": 7.3,
              "net_absorption": 1.3,
              "index": 0.1111,
              "label": "Neutra"
            }
          },
          "derivatives": {
            "score": 1.49,
            "max": 25,
            "detail": {
              "long_short_ratio": 1.1,
              "lsr_score": 1.49
            }
          }
        },
        "trend": {
          "direction": "increasing_distribution",
          "avg_score": -25.2,
          "recent_avg": -37.6,
          "momentum": -7.8,
          "samples": 9,
          "score_range": {
            "min": -49,
            "max": 14
          }
        },
        "status": "success"
      },
      "absorption_zones": {
        "total_zones": 0,
        "total_events": 0,
        "buy_zone_count": 0,
        "sell_zone_count": 0,
        "status": "no_events"
      }
    },
    "sr_analysis": {
      "defense_zones": {
        "buy_defense": [
          {
            "center": 64479.86,
            "range_low": 64376.35,
            "range_high": 64570.65,
            "strength": 62,
            "side": "buy",
            "sources": [
              "sr_level_ema_21_4h",
              "vp_hvn",
              "ema_ema_21_4h",
              "vp_val"
            ],
            "source_count": 4,
            "signals_in_zone": 5,
            "type": "confluence",
            "distance_from_price": 392.14,
            "distance_pct": 0.6
          },
          {
            "center": 64837.39,
            "range_low": 64739.67,
            "range_high": 64928.65,
            "strength": 49,
            "side": "buy",
            "sources": [
              "vp_hvn",
              "ema_ema_21_1h",
              "sr_level_ema_21_1h"
            ],
            "source_count": 3,
            "signals_in_zone": 7,
            "type": "confluence",
            "distance_from_price": 34.61,
            "distance_pct": 0.05
          },
          {
            "center": 64165.68,
            "range_low": 64100.93,
            "range_high": 64230.44,
            "strength": 46,
            "side": "buy",
            "sources": [
              "ema_ema_21_1d",
              "sr_level_vah_monthly"
            ],
            "source_count": 2,
            "signals_in_zone": 2,
            "type": "cluster",
            "distance_from_price": 706.32,
            "distance_pct": 1.09
          },
          {
            "center": 64336.25,
            "range_low": 64255.35,
            "range_high": 64442.65,
            "strength": 40,
            "side": "buy",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 9,
            "type": "cluster",
            "distance_from_price": 535.75,
            "distance_pct": 0.83
          }
        ],
        "sell_defense": [
          {
            "center": 64943.58,
            "range_low": 64860.35,
            "range_high": 65052.65,
            "strength": 59,
            "side": "sell",
            "sources": [
              "orderbook_ask_wall",
              "vp_hvn",
              "ema_ema_21_15m",
              "sr_level_hvn_daily"
            ],
            "source_count": 4,
            "signals_in_zone": 10,
            "type": "confluence",
            "distance_from_price": 71.58,
            "distance_pct": 0.11
          },
          {
            "center": 65037.38,
            "range_low": 64960.35,
            "range_high": 65128.65,
            "strength": 55,
            "side": "sell",
            "sources": [
              "sr_level_poc_daily",
              "vp_poc",
              "vp_hvn"
            ],
            "source_count": 3,
            "signals_in_zone": 9,
            "type": "confluence",
            "distance_from_price": 165.38,
            "distance_pct": 0.25
          },
          {
            "center": 65346,
            "range_low": 65297.35,
            "range_high": 65394.65,
            "strength": 53,
            "side": "sell",
            "sources": [
              "sr_level_vah_daily",
              "vp_hvn",
              "vp_vah"
            ],
            "source_count": 3,
            "signals_in_zone": 3,
            "type": "confluence",
            "distance_from_price": 474,
            "distance_pct": 0.73
          },
          {
            "center": 65157.17,
            "range_low": 65064.35,
            "range_high": 65257.65,
            "strength": 41,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 13,
            "type": "cluster",
            "distance_from_price": 285.17,
            "distance_pct": 0.44
          },
          {
            "center": 65252.3,
            "range_low": 65164.35,
            "range_high": 65350.65,
            "strength": 40,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 11,
            "type": "cluster",
            "distance_from_price": 380.3,
            "distance_pct": 0.59
          }
        ],
        "total_zones": 9,
        "strongest_buy": {
          "center": 64479.86,
          "range_low": 64376.35,
          "range_high": 64570.65,
          "strength": 62,
          "side": "buy",
          "sources": [
            "sr_level_ema_21_4h",
            "vp_hvn",
            "ema_ema_21_4h",
            "vp_val"
          ],
          "source_count": 4,
          "signals_in_zone": 5,
          "type": "confluence",
          "distance_from_price": 392.14,
          "distance_pct": 0.6
        },
        "strongest_sell": {
          "center": 64943.58,
          "range_low": 64860.35,
          "range_high": 65052.65,
          "strength": 59,
          "side": "sell",
          "sources": [
            "orderbook_ask_wall",
            "vp_hvn",
            "ema_ema_21_15m",
            "sr_level_hvn_daily"
          ],
          "source_count": 4,
          "signals_in_zone": 10,
          "type": "confluence",
          "distance_from_price": 71.58,
          "distance_pct": 0.11
        },
        "defense_asymmetry": {
          "ratio": 0.79,
          "bias": "slight_sell_defense",
          "description": "Slightly more sell defense",
          "buy_total_strength": 197,
          "sell_total_strength": 248
        },
        "status": "success"
      }
    },
    "quality": {
      "calendar": {
        "day_of_week": "Friday",
        "day_of_week_num": 4,
        "is_us_holiday": 0,
        "is_weekend": 0,
        "is_pre_holiday": 0,
        "is_post_holiday": 0,
        "expected_liquidity": "NORMAL",
        "liquidity_warning": 0
      },
      "latency": {
        "latency_ms": 7496,
        "latency_category": "POOR",
        "data_freshness": "DELAYED",
        "is_acceptable": 0,
        "is_stale": 0
      },
      "spread_percentile": {
        "status": "ok",
        "current_spread": 0.1,
        "spread_percentile": 0,
        "spread_mean": 0.1,
        "spread_median": 0.1,
        "spread_std": 0,
        "spread_min": 0.1,
        "spread_max": 0.1,
        "spread_z_score": 0,
        "is_tight": 1,
        "is_wide": 0,
        "is_anomalous": 0,
        "liquidity_signal": "EXCELLENT",
        "samples": 9,
        "window_minutes": 1440
      },
      "anomalies": {
        "anomalies_detected": 1,
        "count": 1,
        "anomalies": [
          {
            "type": "FLOW_EXTREME_IMBALANCE",
            "severity": "HIGH",
            "value": 0.7284,
            "direction": "BUY",
            "description": "Extreme flow imbalance: 72.84% toward buyers"
          }
        ],
        "max_severity": "HIGH",
        "risk_elevated": 1,
        "types_found": [
          "FLOW_EXTREME_IMBALANCE"
        ],
        "summary": "1 anomalies detected (max severity: HIGH)"
      }
    }
  },
  "sequence_id": 9,
  "primary_exchange": "BINANCE",
  "data_feed_type": "WEBSOCKET_L2",
  "data_quality_score": 9,
  "completeness_pct": 100,
  "reliability_score": 8,
  "bid": 64843,
  "ask": 64843.1,
  "tick_direction": 1,
  "twap": 64886.4,
  "pivot_points": {
    "daily": {
      "pivot": 64971.33,
      "r1": 65420.66,
      "r2": 65795.33,
      "r3": 66619.33,
      "s1": 64596.66,
      "s2": 64147.33,
      "s3": 63323.33,
      "vah": 65346,
      "val": 64522,
      "poc": 65046
    },
    "weekly": {
      "pivot": 62830.33,
      "r1": 63346.66,
      "r2": 64083.33,
      "r3": 65336.33,
      "s1": 62093.66,
      "s2": 61577.33,
      "s3": 60324.33,
      "vah": 63567,
      "val": 62314,
      "poc": 62610
    },
    "monthly": {
      "pivot": 63185.67,
      "r1": 64582.34,
      "r2": 65610.67,
      "r3": 68035.67,
      "s1": 62157.34,
      "s2": 60760.67,
      "s3": 58335.67,
      "vah": 64214,
      "val": 61789,
      "poc": 63554
    }
  },
  "immediate_support": [
    64837.39,
    64522,
    64479.86,
    64214,
    64165.68
  ],
  "support_strength": [
    49,
    94.6,
    62,
    71.9,
    46
  ],
  "immediate_resistance": [
    64943.58,
    64971.33,
    65037.38,
    65346
  ],
  "resistance_strength": [
    59,
    98.5,
    55,
    92.7
  ],
  "volatility_metrics": {
    "realized_vol_24h": 0.0184,
    "realized_vol_7d": 0.0487,
    "volatility_regime": "NORMAL"
  },
  "order_flow_extended": {
    "passive_buy_pct": 28,
    "passive_sell_pct": 72
  },
  "whale_activity": {
    "large_orders_1h": [
      {
        "size": 1,
        "price": 64896.34,
        "side": "SELL",
        "timestamp_ms": 1786141945725
      },
      {
        "size": 1,
        "price": 64883.14,
        "side": "SELL",
        "timestamp_ms": 1786142044496
      },
      {
        "size": 1,
        "price": 64870.63,
        "side": "SELL",
        "timestamp_ms": 1786142264045
      },
      {
        "size": 1,
        "price": 64853.03,
        "side": "SELL",
        "timestamp_ms": 1786142272654
      }
    ],
    "iceberg_activity": 1,
    "hidden_orders_detected": 0
  },
  "technical_indicators_extended": {
    "cci_1h": 91.97,
    "cci_signal": "NEUTRAL",
    "stochastic": {
      "k": 0,
      "d": 1.4,
      "signal": "OVERSOLD",
      "source": "real"
    },
    "williams_r": {
      "value": -100,
      "overbought": 0,
      "oversold": 1,
      "zone": "oversold",
      "source": "real"
    },
    "hurst_exponent": 0.3291,
    "shannon_entropy": 2.9377,
    "fractal_dimension": 0.6255,
    "kalman_filter": {
      "kalman_price": 64931.19,
      "raw_price": 64846,
      "deviation_pct": -0.13,
      "trend_direction": "DOWN"
    },
    "regression_channel": {
      "slope_per_bar": -1.1725,
      "trend_price": 64893.66,
      "upper_1sd": 64910.77,
      "lower_1sd": 64876.54,
      "upper_2sd": 64927.89,
      "lower_2sd": 64859.42,
      "deviation_from_trend": -47.66,
      "position_in_channel": 0
    },
    "dominant_cycles": {
      "dominant_cycles": [
        100,
        40,
        33.3
      ],
      "cycle_strengths": [
        3066.18,
        1876.06,
        949.71
      ]
    },
    "monte_carlo": {
      "median_price": 64866,
      "p10": 64809.1,
      "p25": 64837.51,
      "p75": 64899.21,
      "p90": 64929.56,
      "prob_up": 0.44,
      "horizon_bars": 12
    },
    "garch_forecast_1h": 0.0015
  },
  "backup_exchanges": [
    "COINBASE",
    "KRAKEN",
    "OKX"
  ],
  "alerts": {
    "active_alerts": [
      {
        "type": "SUPPORT_TEST",
        "level": 64837.39,
        "severity": "HIGH",
        "probability": 0.89,
        "action": "MONITOR_CLOSELY",
        "description": "Preço testando suporte em 64837.39 (dist: 0.05%)"
      },
      {
        "type": "RESISTANCE_TEST",
        "level": 64943.58,
        "severity": "MEDIUM",
        "probability": 0.78,
        "action": "MONITOR_CLOSELY",
        "description": "Preço testando resistência em 64943.58 (dist: 0.11%)"
      },
      {
        "type": "VOLUME_SPIKE",
        "threshold_exceeded": 4.66,
        "severity": "MEDIUM",
        "probability": 0.47,
        "action": "MONITOR",
        "description": "Volume 4.7x acima da média"
      },
      {
        "type": "WHALE_DISTRIBUTION",
        "level": -33,
        "severity": "MEDIUM",
        "probability": 0.66,
        "action": "AVOID_LONG",
        "description": "Sinal de distribuição de whales (score=-33)"
      },
      {
        "type": "BREAKOUT_SIGNAL",
        "severity": "MEDIUM",
        "probability": 0.65,
        "action": "PREPARE_BREAKOUT",
        "description": "Perfil tipo B detectado — breakout iminente"
      }
    ],
    "alert_count": 5,
    "max_severity": "HIGH"
  },
  "regime_analysis": {
    "current_regime": "BREAKOUT",
    "regime_probabilities": {
      "trending": 0,
      "mean_reverting": 0.476,
      "breakout": 0.524
    },
    "regime_change_probability": 0.52,
    "expected_regime_duration": "5m-30m",
    "avg_adx": 25.9
  },
  "data_reliability": {
    "has_options_data": 0,
    "onchain_coverage": "full",
    "latency_acceptable": 1,
    "price_targets_available": 1
  },
  "event_id": "8b08362f",
  "timestamp_ny": "2026-08-07T18:39:00.000-04:00",
  "timestamp_sp": "2026-08-07T19:39:00.000-03:00",
  "_log_id": "8b08362f"
}

----------------------------------------------------------------------------------------------------
 # Janela 10
----------------------------------------------------------------------------------------------------
EVENTO: ANALYSIS_TRIGGER | SYMBOL: BTCUSDT | JANELA: 10
UTC: 2026-08-07 22:40:00 UTC
NY:  2026-08-07 18:40:00 EST/EDT
SP:  2026-08-07 19:40:00 BRT
----------------------------------------------------------------------------------------------------
{
  "is_signal": 1,
  "tipo_evento": "ANALYSIS_TRIGGER",
  "descricao": "Evento automático para análise da IA",
  "symbol": "BTCUSDT",
  "raw_event": {
    "delta": -0.744,
    "volume_total": 0.927,
    "volume_compra": 0.092,
    "volume_venda": 0.835,
    "preco_fechamento": 64872,
    "advanced_analysis": {
      "symbol": "BTCUSDT",
      "price": 64872,
      "volume": 0.927,
      "adaptive_thresholds": {
        "current_volatility": 0.03,
        "volatility_factor": 2.0,
        "absorption_threshold": 0.3,
        "flow_threshold": 0.2
      },
      "onchain_metrics": {
        "difficulty": 126.23,
        "active_addresses": 717974,
        "mempool_size": 87732,
        "mempool_vsize_mb": 43.83,
        "mempool_total_fee_btc": 0.1697,
        "fees_fastest_sat_vb": 5,
        "fees_half_hour_sat_vb": 4,
        "fees_hour_sat_vb": 3,
        "fees_economy_sat_vb": 2,
        "difficulty_adjustment": {
          "progress_pct": 93.4,
          "estimated_change_pct": 0.72,
          "remaining_blocks": 133,
          "remaining_time_ms": 79270394,
          "previous_retarget_pct": -0.74
        },
        "minutes_between_blocks": 8.4423,
        "total_btc_sent_24h": 807619.08,
        "trade_volume_btc_24h": 3265.64,
        "data_source": "blockchain.info+mempool.space",
        "is_real_data": 1,
        "requires_paid_api": [
          "exchange_netflow",
          "whale_transactions",
          "exchange_reserves",
          "sopr"
        ]
      }
    }
  },
  "resultado_da_batalha": "N/A",
  "delta": -0.744,
  "volume_total": 0.927,
  "volume_compra": 0.092,
  "volume_venda": 0.835,
  "preco_fechamento": 64872,
  "timestamp": "2026-08-07T22:40:06Z",
  "epoch_ms": 1786142400000,
  "ml_features": {
    "price_features": {
      "returns_1": 0.0,
      "volatility_1": 0.0,
      "returns_5": 0.0,
      "volatility_5": 0.0,
      "returns_15": 1.5e-07,
      "volatility_15": 7e-08,
      "momentum_score": 1.41421356,
      "volatility_1h": 0.0023
    },
    "volume_features": {
      "volume_sma_ratio": 0.696,
      "volume_momentum": -0.587,
      "buy_sell_pressure": -0.8017,
      "liquidity_gradient": -0.58696613
    },
    "microstructure": {
      "order_book_slope": -0.96272,
      "flow_imbalance": -0.8017,
      "tick_rule_sum": 1,
      "trade_intensity": 35,
      "trade_intensity_v2": 1.6
    },
    "cross_asset": {
      "btc_eth_corr_7d": 0.8212,
      "btc_eth_corr_30d": 0.8599,
      "btc_dxy_corr_30d": -0.0603,
      "btc_dxy_corr_90d": -0.0669,
      "btc_ndx_corr_30d": 0.4146,
      "dxy_return_5d": -0.3561,
      "dxy_return_20d": -1.6548,
      "btc_dxy_correlation_stability": 0.0066,
      "btc_dxy_inverse_strength": 0.0636,
      "dxy_momentum": -1.29867555,
      "vix_current": 14.9,
      "us10y_yield": 4.66,
      "btc_dominance": 17.0386,
      "btc_dominance_change_7d": 0,
      "eth_dominance": 9.0982,
      "gold_price": 4342.3475,
      "oil_price": 77.08,
      "macro_regime": "RISK_ON",
      "correlation_regime": "DECORRELATED"
    }
  },
  "orderbook_data": {
    "mid": 64843.05,
    "spread": 0.1,
    "spread_percent": 0,
    "bid_depth_usd": 1075202.47,
    "ask_depth_usd": 1042189.74,
    "imbalance": 0.016,
    "flow_imbalance": 0.0156,
    "volume_ratio": 1.032,
    "pressure": 0.0156,
    "consolidated_bias_score": 0.5078,
    "spread_bps": 0.0154,
    "is_valid": 1,
    "data_source": "live",
    "spread_volatility": 0.0
  },
  "historical_vp": {
    "daily": {
      "poc": 65046,
      "vah": 65346,
      "val": 64522,
      "status": "success"
    },
    "weekly": {
      "poc": 62610,
      "vah": 63567,
      "val": 62314,
      "status": "success"
    },
    "monthly": {
      "poc": 63554,
      "vah": 64214,
      "val": 61789,
      "status": "success"
    }
  },
  "multi_tf": {
    "15m": {
      "tendencia": "Baixa",
      "preco_atual": 64856.87,
      "mme_21": 64924.03,
      "atr": 73.63,
      "regime": "Range",
      "rsi_short": 35.96,
      "rsi_long": 42.76,
      "macd": 11.4474,
      "macd_signal": 16.415,
      "adx": 23.83,
      "realized_vol": 0.0014
    },
    "1h": {
      "tendencia": "Alta",
      "preco_atual": 64866.66,
      "mme_21": 64788.32,
      "atr": 227.14,
      "regime": "Range",
      "rsi_short": 52.19,
      "rsi_long": 54.07,
      "macd": 99.9296,
      "macd_signal": 98.2155,
      "adx": 23.82,
      "realized_vol": 0.0023
    },
    "4h": {
      "tendencia": "Alta",
      "preco_atual": 64872.89,
      "mme_21": 64460.43,
      "atr": 495.32,
      "regime": "Range",
      "rsi_short": 61.98,
      "rsi_long": 61.12,
      "macd": 275.0187,
      "macd_signal": 256.7251,
      "adx": 30.01,
      "realized_vol": 0.0058
    },
    "1d": {
      "tendencia": "Alta",
      "preco_atual": 64924.96,
      "mme_21": 64149.58,
      "atr": 1369.69,
      "regime": "Manipulação",
      "rsi_short": 59.35,
      "rsi_long": 55.96,
      "macd": 71.3764,
      "macd_signal": 36.7677,
      "adx": 15.05,
      "realized_vol": 0.0184
    }
  },
  "data_context": "real_time",
  "price_targets": [
    {
      "level": 66818.16,
      "confidence": 0.3,
      "source": "fallback_r1",
      "weight": 0.1
    },
    {
      "level": 68764.32,
      "confidence": 0.2,
      "source": "fallback_r2",
      "weight": 0.05
    },
    {
      "level": 62925.84,
      "confidence": 0.3,
      "source": "fallback_s1",
      "weight": 0.1
    },
    {
      "level": 60979.68,
      "confidence": 0.2,
      "source": "fallback_s2",
      "weight": 0.05
    },
    {
      "level": 64872,
      "confidence": 0.4,
      "source": "fallback_poc",
      "weight": 0.15
    }
  ],
  "pattern_recognition": {
    "smart_money": {
      "fair_value_gaps": [
        {
          "type": "BEARISH",
          "top": 64925,
          "bottom": 64892.3,
          "gap_size": 32.7,
          "gap_pct": 0.05
        }
      ],
      "market_structure": {
        "structure": "BEARISH",
        "bos_detected": 0,
        "last_swing_high": 64925,
        "last_swing_low": 64909.55,
        "higher_highs": 0,
        "higher_lows": 0
      }
    }
  },
  "janela_numero": 10,
  "timestamp_utc": "2026-08-07T22:40:00.000Z",
  "derivatives": {
    "BTCUSDT": {
      "funding_rate_percent": 0.01,
      "open_interest": 106934.462,
      "open_interest_usd": 6934209575.49,
      "long_short_ratio": 1.1,
      "longs_usd": 3631418636.94,
      "shorts_usd": 3302790938.55
    },
    "ETHUSDT": {
      "funding_rate_percent": 0,
      "open_interest": 2279900.074,
      "open_interest_usd": 4362909179.41,
      "long_short_ratio": 2.07,
      "longs_usd": 2940607898.43,
      "shorts_usd": 1422301280.98
    }
  },
  "fluxo_continuo": {
    "cvd": -22.9734,
    "whale_buy_volume": 0,
    "whale_sell_volume": 4,
    "whale_delta": -4,
    "bursts": {
      "count": 7,
      "max_burst_volume": 7.558
    },
    "sector_flow": {
      "retail": {
        "buy": 9.053,
        "sell": 25.5866,
        "delta": -16.534
      },
      "mid": {
        "buy": 1.522,
        "sell": 3.9619,
        "delta": -2.44
      },
      "whale": {
        "buy": 0,
        "sell": 4,
        "delta": -4
      }
    },
    "timestamp": "2026-08-07T22:40:00.000+00:00",
    "time_index": {
      "epoch_ms": 1786142400000,
      "timestamp_utc": "2026-08-07T22:40:00.000+00:00",
      "timestamp_ny": "2026-08-07T18:40:00.000-04:00",
      "timestamp_sp": "2026-08-07T19:40:00.000-03:00"
    },
    "order_flow": {
      "net_flow_1m": -38256.309,
      "absorcao_1m": "Neutra",
      "buy_volume": 5964.98,
      "sell_volume": 54194.71,
      "total_volume": 60159.69,
      "buy_volume_btc": 0.092,
      "sell_volume_btc": 0.835,
      "total_volume_btc": 0.927,
      "whale_buy_volume_window": 0,
      "whale_sell_volume_window": 0,
      "whale_delta_window": 0,
      "flow_imbalance": -0.8017,
      "aggressive_buy_pct": 9.92,
      "aggressive_sell_pct": 90.08,
      "net_flow_5m": -395611.7113,
      "absorcao_5m": "Absorção de Compra",
      "net_flow_15m": -1491025.2373,
      "absorcao_15m": "Neutra",
      "computation_window_min": 1,
      "available_windows_min": [
        1,
        5,
        15
      ],
      "buy_sell_ratio": {
        "buy_sell_ratio": 0.11,
        "ratios": {
          "current": 0.1101,
          "imbalance_1m": -0.636,
          "imbalance_5m": -6.576,
          "imbalance_15m": -24.785
        },
        "sector_ratios": {
          "retail": 0.3538,
          "mid": 0.3842,
          "whale": 0
        },
        "pressure": "STRONG_SELL",
        "flow_trend": "consistent_selling",
        "buy_volume": 0.092,
        "sell_volume": 0.835
      }
    },
    "tipo_absorcao": "Neutra",
    "participant_analysis": {
      "retail": {
        "volume_pct": 100,
        "direction": "SELL",
        "sentiment": "BEARISH",
        "composite_score": -0.751,
        "imbalance": -0.802
      }
    },
    "liquidity_heatmap": {
      "clusters": [
        {
          "center": 64858.2846,
          "low": 64846,
          "high": 64872,
          "width": 26,
          "total_volume": 10.524,
          "buy_volume": 4.604,
          "sell_volume": 5.92,
          "imbalance": -1.316,
          "imbalance_ratio": -0.125,
          "trades_count": 2000,
          "avg_trade_size": 0.005,
          "recent_timestamp": 1786142405183,
          "recent_ts_ms": 1786142405183,
          "last_seen_ms": 1786142405183,
          "first_seen_ms": 1786142264045,
          "age_ms": 275.0,
          "cluster_duration_ms": 141138,
          "price_std": 8.504,
          "volume_std": 0.033,
          "bin_threshold_usd": 194.5749
        }
      ],
      "resistances": [
        64858.2846
      ],
      "clusters_count": 1
    },
    "absorption_analysis": {
      "current_absorption": {
        "index": 0.5098,
        "classification": "MODERATE_ABSORPTION",
        "label": "Neutra",
        "buyer_strength": 1,
        "seller_exhaustion": 8,
        "continuation_probability": 0.46,
        "delta_usd": -38256.309,
        "total_volume_usd": 60159.69,
        "flow_imbalance": -0.8017,
        "window_min": 1
      }
    }
  },
  "market_context": {
    "trading_session": "NY",
    "session_phase": "ACTIVE",
    "time_to_session_close": 12269,
    "day_of_week": 4,
    "is_holiday": 0,
    "market_hours_type": "EXTENDED"
  },
  "market_environment": {
    "volatility_regime": "NORMAL",
    "trend_direction": "UP",
    "market_structure": "RANGE_BOUND",
    "liquidity_environment": "NORMAL",
    "risk_sentiment": "BULLISH",
    "correlation_spy": 0.406,
    "correlation_dxy": -0.0867,
    "correlation_gold": 0.2401
  },
  "external_markets": {
    "FEAR_GREED": {
      "preco_atual": 29,
      "prev": 25,
      "movimento": "Alta",
      "classification": "Fear",
      "source": "alternative.me",
      "timestamp": "2026-08-07T19:30:17.741873"
    }
  },
  "contextual_snapshot": {
    "symbol": "BTCUSDT",
    "ohlc": {
      "open": 64872,
      "high": 64872,
      "low": 64872,
      "close": 64872,
      "open_time": 1786142340333,
      "close_time": 1786142399056,
      "vwap": 64872
    },
    "volume_total": 0.927,
    "volume_total_usdt": 60160,
    "volume_compra": 0.092,
    "volume_venda": 0.835,
    "num_trades": 90,
    "delta_minimo": -0.775,
    "delta_maximo": -0.009,
    "delta_fechamento": -0.744,
    "reversao_desde_minimo": 0.03,
    "reversao_desde_maximo": 0.73,
    "poc_price": 64872,
    "poc_volume": 0.84,
    "poc_percentage": 90.1,
    "dwell_price": 64872,
    "dwell_seconds": 58,
    "dwell_location": "High",
    "trades_per_second": 1.53,
    "avg_trade_size": 0.01
  },
  "order_book_depth": {
    "L1": {
      "bids": 488462.32,
      "asks": 349698.84,
      "flow_imbalance": 0.1656
    },
    "L5": {
      "bids": 604271.73,
      "asks": 478088.38,
      "flow_imbalance": 0.1166
    },
    "L10": {
      "bids": 702637.31,
      "asks": 486647.75,
      "flow_imbalance": 0.1816
    },
    "L25": {
      "bids": 856632.93,
      "asks": 539626.19,
      "flow_imbalance": 0.227
    },
    "total_depth_ratio": 1.59
  },
  "orderbook_quality": "live",
  "market_impact": {
    "slippage_matrix": {
      "100k_usd": {
        "buy": 0.05,
        "sell": 0.05
      },
      "1m_usd": {
        "buy": 4.55,
        "sell": 5.35
      }
    },
    "liquidity_score": 9.9985,
    "execution_quality": "EXCELLENT"
  },
  "institutional_analytics": {
    "status": "ok",
    "computed_at_ms": 1786142406755,
    "technical_extras": {
      "stoch_rsi": {
        "k": 19.52,
        "d": 6.51,
        "overbought": 0,
        "oversold": 1,
        "crossover": "bullish"
      },
      "williams_r": {
        "value": -67.09,
        "overbought": 0,
        "oversold": 0,
        "zone": "neutral",
        "source": "real"
      },
      "hurst_exponent": 0.3276,
      "shannon_entropy": 2.9466,
      "kalman_filter": {
        "kalman_price": 64929.36,
        "raw_price": 64872,
        "deviation_pct": -0.09,
        "trend_direction": "DOWN"
      },
      "regression_channel": {
        "slope_per_bar": -1.1637,
        "trend_price": 64891.82,
        "upper_1sd": 64908.85,
        "lower_1sd": 64874.8,
        "upper_2sd": 64925.87,
        "lower_2sd": 64857.78,
        "deviation_from_trend": -19.82,
        "position_in_channel": 0.2089
      },
      "dominant_cycles": {
        "dominant_cycles": [
          100,
          40,
          33.3
        ],
        "cycle_strengths": [
          3075.69,
          1891.38,
          962.05
        ]
      },
      "fractal_dimension": 0.624,
      "monte_carlo": {
        "median_price": 64866.05,
        "p10": 64809.11,
        "p25": 64837.54,
        "p75": 64899.29,
        "p90": 64929.66,
        "prob_up": 0.45,
        "horizon_bars": 12
      },
      "fair_value_gaps": [
        {
          "type": "BEARISH",
          "top": 64925,
          "bottom": 64892.3,
          "gap_size": 32.7,
          "gap_pct": 0.05
        }
      ],
      "market_structure": {
        "structure": "BEARISH",
        "bos_detected": 0,
        "last_swing_high": 64925,
        "last_swing_low": 64909.55,
        "higher_highs": 0,
        "higher_lows": 0
      }
    },
    "profile_analysis": {
      "poor_extremes": {
        "poor_high": {
          "detected": 1,
          "price": 64872,
          "volume_ratio": 0.992,
          "implication": "High likely to be revisited - unfinished auction"
        },
        "poor_low": {
          "detected": 1,
          "price": 64871.99,
          "volume_ratio": 9.008,
          "implication": "Low likely to be revisited - unfinished auction"
        },
        "excess_high": 0,
        "excess_low": 0,
        "session_high": 64872,
        "session_low": 64871.99,
        "action_bias": "expect_retest_both",
        "status": "success"
      },
      "profile_shape": {
        "shape": "b",
        "implication": "Long liquidation - bullish bias expected",
        "trading_signal": "BULLISH_AFTER",
        "distribution": {
          "lower_third_pct": 90.1,
          "middle_third_pct": 0,
          "upper_third_pct": 9.9
        },
        "dominant_zone": "lower",
        "status": "success"
      },
      "no_mans_land": {
        "zones": [
          {
            "range_low": 64512,
            "range_high": 64810,
            "gap_size": 298,
            "gap_size_pct": 0.462,
            "risk": "LOW",
            "lvn_confirmed": 1,
            "lvns_in_zone": 9,
            "nearest_hvn_below": 64512,
            "nearest_hvn_above": 64810,
            "distance_from_price": 62,
            "distance_pct": 0.1,
            "direction": "below"
          }
        ],
        "price_in_no_mans_land": 0,
        "nearest_no_mans_land": {
          "range_low": 64512,
          "range_high": 64810,
          "gap_size": 298,
          "gap_size_pct": 0.462,
          "risk": "LOW",
          "lvn_confirmed": 1,
          "lvns_in_zone": 9,
          "nearest_hvn_below": 64512,
          "nearest_hvn_above": 64810,
          "distance_from_price": 62,
          "distance_pct": 0.1,
          "direction": "below"
        },
        "total_zones": 1,
        "max_gap_pct": 0.46,
        "status": "success"
      },
      "va_volume_pct": {
        "value_area_volume_pct": 100,
        "interpretation": "extremely_compressed",
        "breakout_risk": "VERY_HIGH",
        "volume_in_va": 1,
        "total_volume": 1,
        "compression_signal": 1
      },
      "volume_node_strength": {
        "scored_hvns": [
          {
            "price": 64304,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64306,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64314,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64320,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64322,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.8,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          }
        ],
        "scored_lvns": [
          {
            "price": 64423,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.5,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64436,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64440,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64447,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64448,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          }
        ],
        "total_hvns": 52,
        "total_lvns": 101,
        "avg_hvn_strength": 66.3,
        "avg_lvn_strength": 63.1,
        "status": "success"
      }
    },
    "flow_analysis": {
      "passive_aggressive": {
        "aggressive": {
          "buy_pct": 9.92,
          "sell_pct": 90.08,
          "net_pct": -80.16,
          "dominance": "sellers",
          "buy_volume": 0.092,
          "sell_volume": 0.835
        },
        "passive": {
          "dominance": "balanced",
          "inference": "from_orderbook_depth",
          "bid_depth": 1075202.47,
          "ask_depth": 1042189.74,
          "bid_ratio": 0.51,
          "ob_imbalance": 0.016
        },
        "composite": {
          "signal": "mixed",
          "interpretation": "Mixed signals between aggressive and passive flow",
          "conviction": "LOW"
        },
        "status": "success"
      },
      "whale_accumulation": {
        "score": -49,
        "classification": "MILD_DISTRIBUTION",
        "bias": "DISTRIBUTING",
        "components": {
          "flow": {
            "score": -30,
            "max": 30,
            "detail": {
              "divergence": "aligned",
              "whale_delta": -4,
              "mid_delta": -2.44,
              "retail_delta": -16.534,
              "primary_delta": -4
            }
          },
          "depth": {
            "score": 0.31,
            "max": 20,
            "detail": {
              "bid_depth": 1075202.47,
              "ask_depth": 1042189.74,
              "ratio": 0.02,
              "deep_confirmation": 0
            }
          },
          "absorption": {
            "score": -21,
            "max": 25,
            "detail": {
              "buyer_strength": 1,
              "seller_exhaustion": 8,
              "net_absorption": -7,
              "index": 0.5098,
              "label": "Neutra"
            }
          },
          "derivatives": {
            "score": 1.49,
            "max": 25,
            "detail": {
              "long_short_ratio": 1.1,
              "lsr_score": 1.49
            }
          }
        },
        "trend": {
          "direction": "increasing_distribution",
          "avg_score": -27.6,
          "recent_avg": -38.8,
          "momentum": -21.4,
          "samples": 10,
          "score_range": {
            "min": -49,
            "max": 14
          }
        },
        "status": "success"
      },
      "absorption_zones": {
        "total_zones": 0,
        "total_events": 0,
        "buy_zone_count": 0,
        "sell_zone_count": 0,
        "status": "no_events"
      }
    },
    "sr_analysis": {
      "defense_zones": {
        "buy_defense": [
          {
            "center": 64479.86,
            "range_low": 64376.35,
            "range_high": 64570.65,
            "strength": 62,
            "side": "buy",
            "sources": [
              "sr_level_ema_21_4h",
              "vp_hvn",
              "ema_ema_21_4h",
              "vp_val"
            ],
            "source_count": 4,
            "signals_in_zone": 5,
            "type": "confluence",
            "distance_from_price": 392.14,
            "distance_pct": 0.6
          },
          {
            "center": 64837.39,
            "range_low": 64739.67,
            "range_high": 64928.65,
            "strength": 49,
            "side": "buy",
            "sources": [
              "vp_hvn",
              "ema_ema_21_1h",
              "sr_level_ema_21_1h"
            ],
            "source_count": 3,
            "signals_in_zone": 7,
            "type": "confluence",
            "distance_from_price": 34.61,
            "distance_pct": 0.05
          },
          {
            "center": 64165.68,
            "range_low": 64100.93,
            "range_high": 64230.44,
            "strength": 46,
            "side": "buy",
            "sources": [
              "ema_ema_21_1d",
              "sr_level_vah_monthly"
            ],
            "source_count": 2,
            "signals_in_zone": 2,
            "type": "cluster",
            "distance_from_price": 706.32,
            "distance_pct": 1.09
          },
          {
            "center": 64336.25,
            "range_low": 64255.35,
            "range_high": 64442.65,
            "strength": 40,
            "side": "buy",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 9,
            "type": "cluster",
            "distance_from_price": 535.75,
            "distance_pct": 0.83
          }
        ],
        "sell_defense": [
          {
            "center": 65037.38,
            "range_low": 64960.35,
            "range_high": 65128.65,
            "strength": 55,
            "side": "sell",
            "sources": [
              "sr_level_poc_daily",
              "vp_poc",
              "vp_hvn"
            ],
            "source_count": 3,
            "signals_in_zone": 9,
            "type": "confluence",
            "distance_from_price": 165.38,
            "distance_pct": 0.25
          },
          {
            "center": 65346,
            "range_low": 65297.35,
            "range_high": 65394.65,
            "strength": 53,
            "side": "sell",
            "sources": [
              "sr_level_vah_daily",
              "vp_hvn",
              "vp_vah"
            ],
            "source_count": 3,
            "signals_in_zone": 3,
            "type": "confluence",
            "distance_from_price": 474,
            "distance_pct": 0.73
          },
          {
            "center": 64944.32,
            "range_low": 64860.35,
            "range_high": 65052.65,
            "strength": 49,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "ema_ema_21_15m",
              "sr_level_hvn_daily"
            ],
            "source_count": 3,
            "signals_in_zone": 9,
            "type": "confluence",
            "distance_from_price": 72.32,
            "distance_pct": 0.11
          },
          {
            "center": 65157.17,
            "range_low": 65064.35,
            "range_high": 65257.65,
            "strength": 41,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 13,
            "type": "cluster",
            "distance_from_price": 285.17,
            "distance_pct": 0.44
          },
          {
            "center": 65252.3,
            "range_low": 65164.35,
            "range_high": 65350.65,
            "strength": 40,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 11,
            "type": "cluster",
            "distance_from_price": 380.3,
            "distance_pct": 0.59
          }
        ],
        "total_zones": 9,
        "strongest_buy": {
          "center": 64479.86,
          "range_low": 64376.35,
          "range_high": 64570.65,
          "strength": 62,
          "side": "buy",
          "sources": [
            "sr_level_ema_21_4h",
            "vp_hvn",
            "ema_ema_21_4h",
            "vp_val"
          ],
          "source_count": 4,
          "signals_in_zone": 5,
          "type": "confluence",
          "distance_from_price": 392.14,
          "distance_pct": 0.6
        },
        "strongest_sell": {
          "center": 65037.38,
          "range_low": 64960.35,
          "range_high": 65128.65,
          "strength": 55,
          "side": "sell",
          "sources": [
            "sr_level_poc_daily",
            "vp_poc",
            "vp_hvn"
          ],
          "source_count": 3,
          "signals_in_zone": 9,
          "type": "confluence",
          "distance_from_price": 165.38,
          "distance_pct": 0.25
        },
        "defense_asymmetry": {
          "ratio": 0.83,
          "bias": "slight_sell_defense",
          "description": "Slightly more sell defense",
          "buy_total_strength": 197,
          "sell_total_strength": 238
        },
        "status": "success"
      }
    },
    "quality": {
      "calendar": {
        "day_of_week": "Friday",
        "day_of_week_num": 4,
        "is_us_holiday": 0,
        "is_weekend": 0,
        "is_pre_holiday": 0,
        "is_post_holiday": 0,
        "expected_liquidity": "NORMAL",
        "liquidity_warning": 0
      },
      "latency": {
        "latency_ms": 6979,
        "latency_category": "POOR",
        "data_freshness": "DELAYED",
        "is_acceptable": 0,
        "is_stale": 0
      },
      "spread_percentile": {
        "status": "ok",
        "current_spread": 0.1,
        "spread_percentile": 0,
        "spread_mean": 0.1,
        "spread_median": 0.1,
        "spread_std": 0,
        "spread_min": 0.1,
        "spread_max": 0.1,
        "spread_z_score": 0,
        "is_tight": 1,
        "is_wide": 0,
        "is_anomalous": 0,
        "liquidity_signal": "EXCELLENT",
        "samples": 10,
        "window_minutes": 1440
      },
      "anomalies": {
        "anomalies_detected": 1,
        "count": 1,
        "anomalies": [
          {
            "type": "FLOW_EXTREME_IMBALANCE",
            "severity": "HIGH",
            "value": -0.8017,
            "direction": "SELL",
            "description": "Extreme flow imbalance: -80.17% toward sellers"
          }
        ],
        "max_severity": "HIGH",
        "risk_elevated": 1,
        "types_found": [
          "FLOW_EXTREME_IMBALANCE"
        ],
        "summary": "1 anomalies detected (max severity: HIGH)"
      }
    }
  },
  "sequence_id": 10,
  "primary_exchange": "BINANCE",
  "data_feed_type": "WEBSOCKET_L2",
  "data_quality_score": 9,
  "completeness_pct": 100,
  "reliability_score": 8,
  "bid": 64843,
  "ask": 64843.1,
  "tick_direction": 0,
  "twap": 64884.96,
  "pivot_points": {
    "daily": {
      "pivot": 64971.33,
      "r1": 65420.66,
      "r2": 65795.33,
      "r3": 66619.33,
      "s1": 64596.66,
      "s2": 64147.33,
      "s3": 63323.33,
      "vah": 65346,
      "val": 64522,
      "poc": 65046
    },
    "weekly": {
      "pivot": 62830.33,
      "r1": 63346.66,
      "r2": 64083.33,
      "r3": 65336.33,
      "s1": 62093.66,
      "s2": 61577.33,
      "s3": 60324.33,
      "vah": 63567,
      "val": 62314,
      "poc": 62610
    },
    "monthly": {
      "pivot": 63185.67,
      "r1": 64582.34,
      "r2": 65610.67,
      "r3": 68035.67,
      "s1": 62157.34,
      "s2": 60760.67,
      "s3": 58335.67,
      "vah": 64214,
      "val": 61789,
      "poc": 63554
    }
  },
  "immediate_support": [
    64837.39,
    64522,
    64479.86,
    64214,
    64165.68
  ],
  "support_strength": [
    49,
    94.6,
    62,
    71.9,
    46
  ],
  "immediate_resistance": [
    64944.32,
    64971.33,
    65037.38,
    65346
  ],
  "resistance_strength": [
    49,
    98.5,
    55,
    92.7
  ],
  "volatility_metrics": {
    "realized_vol_24h": 0.0184,
    "realized_vol_7d": 0.0487,
    "volatility_regime": "NORMAL"
  },
  "order_flow_extended": {
    "passive_buy_pct": 50.8,
    "passive_sell_pct": 49.2
  },
  "whale_activity": {
    "large_orders_1h": [
      {
        "size": 1,
        "price": 64896.34,
        "side": "SELL",
        "timestamp_ms": 1786141945725
      },
      {
        "size": 1,
        "price": 64883.14,
        "side": "SELL",
        "timestamp_ms": 1786142044496
      },
      {
        "size": 1,
        "price": 64870.63,
        "side": "SELL",
        "timestamp_ms": 1786142264045
      },
      {
        "size": 1,
        "price": 64853.03,
        "side": "SELL",
        "timestamp_ms": 1786142272654
      }
    ],
    "iceberg_activity": 1,
    "hidden_orders_detected": 0
  },
  "technical_indicators_extended": {
    "cci_1h": 91.97,
    "cci_signal": "NEUTRAL",
    "stochastic": {
      "k": 19.52,
      "d": 6.51,
      "signal": "OVERSOLD",
      "source": "real"
    },
    "williams_r": {
      "value": -67.09,
      "overbought": 0,
      "oversold": 0,
      "zone": "neutral",
      "source": "real"
    },
    "hurst_exponent": 0.3276,
    "shannon_entropy": 2.9466,
    "fractal_dimension": 0.624,
    "kalman_filter": {
      "kalman_price": 64929.36,
      "raw_price": 64872,
      "deviation_pct": -0.09,
      "trend_direction": "DOWN"
    },
    "regression_channel": {
      "slope_per_bar": -1.1637,
      "trend_price": 64891.82,
      "upper_1sd": 64908.85,
      "lower_1sd": 64874.8,
      "upper_2sd": 64925.87,
      "lower_2sd": 64857.78,
      "deviation_from_trend": -19.82,
      "position_in_channel": 0.2089
    },
    "dominant_cycles": {
      "dominant_cycles": [
        100,
        40,
        33.3
      ],
      "cycle_strengths": [
        3075.69,
        1891.38,
        962.05
      ]
    },
    "monte_carlo": {
      "median_price": 64866.05,
      "p10": 64809.11,
      "p25": 64837.54,
      "p75": 64899.29,
      "p90": 64929.66,
      "prob_up": 0.45,
      "horizon_bars": 12
    },
    "garch_forecast_1h": 0.0016
  },
  "backup_exchanges": [
    "COINBASE",
    "KRAKEN",
    "OKX"
  ],
  "alerts": {
    "alert_count": 0,
    "max_severity": "NONE"
  },
  "regime_analysis": {
    "current_regime": "MEAN_REVERTING",
    "regime_probabilities": {
      "trending": 0,
      "mean_reverting": 0.636,
      "breakout": 0.364
    },
    "regime_change_probability": 0.36,
    "expected_regime_duration": "15m-1h",
    "avg_adx": 25.9
  },
  "data_reliability": {
    "has_options_data": 0,
    "onchain_coverage": "full",
    "latency_acceptable": 1,
    "price_targets_available": 1
  },
  "event_id": "ab1845a1",
  "timestamp_ny": "2026-08-07T18:40:00.000-04:00",
  "timestamp_sp": "2026-08-07T19:40:00.000-03:00",
  "_log_id": "ab1845a1"
}

----------------------------------------------------------------------------------------------------
 # Janela 11
----------------------------------------------------------------------------------------------------
EVENTO: ANALYSIS_TRIGGER | SYMBOL: BTCUSDT | JANELA: 11
UTC: 2026-08-07 22:41:00 UTC
NY:  2026-08-07 18:41:00 EST/EDT
SP:  2026-08-07 19:41:00 BRT
----------------------------------------------------------------------------------------------------
{
  "is_signal": 1,
  "tipo_evento": "ANALYSIS_TRIGGER",
  "descricao": "Evento automático para análise da IA",
  "symbol": "BTCUSDT",
  "raw_event": {
    "delta": 0.007,
    "volume_total": 0.396,
    "volume_compra": 0.202,
    "volume_venda": 0.195,
    "preco_fechamento": 64872,
    "advanced_analysis": {
      "symbol": "BTCUSDT",
      "price": 64872,
      "volume": 0.396,
      "adaptive_thresholds": {
        "current_volatility": 0.03,
        "volatility_factor": 2.0,
        "absorption_threshold": 0.3,
        "flow_threshold": 0.2
      },
      "onchain_metrics": {
        "difficulty": 126.23,
        "active_addresses": 717974,
        "mempool_size": 84905,
        "mempool_vsize_mb": 43.24,
        "mempool_total_fee_btc": 0.1258,
        "fees_fastest_sat_vb": 4,
        "fees_half_hour_sat_vb": 4,
        "fees_hour_sat_vb": 3,
        "fees_economy_sat_vb": 2,
        "difficulty_adjustment": {
          "progress_pct": 93.45,
          "estimated_change_pct": 0.75,
          "remaining_blocks": 132,
          "remaining_time_ms": 78653520,
          "previous_retarget_pct": -0.74
        },
        "minutes_between_blocks": 8.4423,
        "total_btc_sent_24h": 807619.08,
        "trade_volume_btc_24h": 3265.64,
        "data_source": "blockchain.info+mempool.space",
        "is_real_data": 1,
        "requires_paid_api": [
          "exchange_netflow",
          "whale_transactions",
          "exchange_reserves",
          "sopr"
        ]
      }
    }
  },
  "resultado_da_batalha": "N/A",
  "delta": 0.007,
  "volume_total": 0.396,
  "volume_compra": 0.202,
  "volume_venda": 0.195,
  "preco_fechamento": 64872,
  "timestamp": "2026-08-07T22:41:21Z",
  "epoch_ms": 1786142460000,
  "ml_features": {
    "price_features": {
      "returns_1": 0.0,
      "volatility_1": 0.0,
      "returns_5": 1.5e-07,
      "volatility_5": 6e-08,
      "returns_15": 0.0,
      "volatility_15": 8e-08,
      "momentum_score": -0.70710678,
      "volatility_1h": 0.0023
    },
    "volume_features": {
      "volume_sma_ratio": 0.713,
      "volume_momentum": 0.071,
      "buy_sell_pressure": 0.0187,
      "liquidity_gradient": 0.07061527
    },
    "microstructure": {
      "order_book_slope": -1.49868,
      "flow_imbalance": 0.0187,
      "tick_rule_sum": 1,
      "trade_intensity": 35,
      "trade_intensity_v2": 1.85
    },
    "cross_asset": {
      "btc_eth_corr_7d": 0.8212,
      "btc_eth_corr_30d": 0.8599,
      "btc_dxy_corr_30d": -0.0605,
      "btc_dxy_corr_90d": -0.067,
      "btc_ndx_corr_30d": 0.4147,
      "dxy_return_5d": -0.3561,
      "dxy_return_20d": -1.6548,
      "btc_dxy_correlation_stability": 0.0065,
      "btc_dxy_inverse_strength": 0.0638,
      "dxy_momentum": -1.29867555,
      "vix_current": 14.9,
      "us10y_yield": 4.66,
      "btc_dominance": 17.0377,
      "btc_dominance_change_7d": 0,
      "eth_dominance": 9.0966,
      "gold_price": 4342.3515,
      "oil_price": 77.08,
      "macro_regime": "RISK_ON",
      "correlation_regime": "DECORRELATED"
    }
  },
  "orderbook_data": {
    "mid": 64843.05,
    "spread": 0.1,
    "spread_percent": 0,
    "bid_depth_usd": 1307853.23,
    "ask_depth_usd": 837991.67,
    "imbalance": 0.219,
    "flow_imbalance": 0.219,
    "volume_ratio": 1.561,
    "pressure": 0.219,
    "consolidated_bias_score": 0.6218,
    "spread_bps": 0.0154,
    "is_valid": 1,
    "data_source": "live",
    "spread_volatility": 0.0
  },
  "historical_vp": {
    "daily": {
      "poc": 65046,
      "vah": 65346,
      "val": 64522,
      "status": "success"
    },
    "weekly": {
      "poc": 62610,
      "vah": 63567,
      "val": 62314,
      "status": "success"
    },
    "monthly": {
      "poc": 63554,
      "vah": 64214,
      "val": 61789,
      "status": "success"
    }
  },
  "multi_tf": {
    "15m": {
      "tendencia": "Baixa",
      "preco_atual": 64871.99,
      "mme_21": 64925.41,
      "atr": 73.63,
      "regime": "Range",
      "rsi_short": 37.88,
      "rsi_long": 44.06,
      "macd": 12.6535,
      "macd_signal": 16.6562,
      "adx": 23.83,
      "realized_vol": 0.0014
    },
    "1h": {
      "tendencia": "Alta",
      "preco_atual": 64871.99,
      "mme_21": 64788.8,
      "atr": 227.14,
      "regime": "Range",
      "rsi_short": 52.6,
      "rsi_long": 54.35,
      "macd": 100.3548,
      "macd_signal": 98.3005,
      "adx": 23.82,
      "realized_vol": 0.0023
    },
    "4h": {
      "tendencia": "Alta",
      "preco_atual": 64872,
      "mme_21": 64460.35,
      "atr": 498.88,
      "regime": "Range",
      "rsi_short": 61.94,
      "rsi_long": 61.09,
      "macd": 274.9477,
      "macd_signal": 256.7109,
      "adx": 30.01,
      "realized_vol": 0.0058
    },
    "1d": {
      "tendencia": "Alta",
      "preco_atual": 64872,
      "mme_21": 64144.77,
      "atr": 1369.69,
      "regime": "Manipulação",
      "rsi_short": 58.86,
      "rsi_long": 55.63,
      "macd": 67.1517,
      "macd_signal": 35.9228,
      "adx": 15.05,
      "realized_vol": 0.0184
    }
  },
  "data_context": "real_time",
  "price_targets": [
    {
      "level": 66818.16,
      "confidence": 0.3,
      "source": "fallback_r1",
      "weight": 0.1
    },
    {
      "level": 68764.32,
      "confidence": 0.2,
      "source": "fallback_r2",
      "weight": 0.05
    },
    {
      "level": 62925.84,
      "confidence": 0.3,
      "source": "fallback_s1",
      "weight": 0.1
    },
    {
      "level": 60979.68,
      "confidence": 0.2,
      "source": "fallback_s2",
      "weight": 0.05
    },
    {
      "level": 64872,
      "confidence": 0.4,
      "source": "fallback_poc",
      "weight": 0.15
    }
  ],
  "pattern_recognition": {
    "smart_money": {
      "fair_value_gaps": [
        {
          "type": "BEARISH",
          "top": 64925,
          "bottom": 64892.3,
          "gap_size": 32.7,
          "gap_pct": 0.05
        }
      ],
      "market_structure": {
        "structure": "BEARISH",
        "bos_detected": 0,
        "last_swing_high": 64925,
        "last_swing_low": 64909.55,
        "higher_highs": 0,
        "higher_lows": 0
      }
    }
  },
  "janela_numero": 11,
  "timestamp_utc": "2026-08-07T22:41:00.000Z",
  "derivatives": {
    "BTCUSDT": {
      "funding_rate_percent": 0.01,
      "open_interest": 106945.765,
      "open_interest_usd": 6934687958.97,
      "long_short_ratio": 1.1,
      "longs_usd": 3631669164.51,
      "shorts_usd": 3303018794.46
    },
    "ETHUSDT": {
      "funding_rate_percent": 0,
      "open_interest": 2279164.179,
      "open_interest_usd": 4360633657.11,
      "long_short_ratio": 2.07,
      "longs_usd": 2939074192.69,
      "shorts_usd": 1421559464.42
    }
  },
  "fluxo_continuo": {
    "cvd": -22.9622,
    "whale_buy_volume": 0,
    "whale_sell_volume": 4,
    "whale_delta": -4,
    "bursts": {
      "count": 7,
      "max_burst_volume": 7.558
    },
    "sector_flow": {
      "retail": {
        "buy": 9.2809,
        "sell": 25.8033,
        "delta": -16.522
      },
      "mid": {
        "buy": 1.522,
        "sell": 3.9619,
        "delta": -2.44
      },
      "whale": {
        "buy": 0,
        "sell": 4,
        "delta": -4
      }
    },
    "timestamp": "2026-08-07T22:41:00.000+00:00",
    "time_index": {
      "epoch_ms": 1786142460000,
      "timestamp_utc": "2026-08-07T22:41:00.000+00:00",
      "timestamp_ny": "2026-08-07T18:41:00.000-04:00",
      "timestamp_sp": "2026-08-07T19:41:00.000-03:00"
    },
    "order_flow": {
      "net_flow_1m": 229.0004,
      "absorcao_1m": "Neutra",
      "buy_volume": 13098.95,
      "sell_volume": 12618.9,
      "total_volume": 25717.85,
      "buy_volume_btc": 0.202,
      "sell_volume_btc": 0.195,
      "total_volume_btc": 0.396,
      "whale_buy_volume_window": 0,
      "whale_sell_volume_window": 0,
      "whale_delta_window": 0,
      "flow_imbalance": 0.0187,
      "aggressive_buy_pct": 50.93,
      "aggressive_sell_pct": 49.07,
      "net_flow_5m": -342248.8551,
      "absorcao_5m": "Absorção de Compra",
      "net_flow_15m": -1490297.3712,
      "absorcao_15m": "Neutra",
      "computation_window_min": 1,
      "available_windows_min": [
        1,
        5,
        15
      ],
      "buy_sell_ratio": {
        "buy_sell_ratio": 1.04,
        "ratios": {
          "current": 1.038,
          "imbalance_1m": 0.009,
          "imbalance_5m": -13.308,
          "imbalance_15m": -57.948
        },
        "sector_ratios": {
          "retail": 0.3597,
          "mid": 0.3842,
          "whale": 0
        },
        "pressure": "NEUTRAL",
        "flow_trend": "short_term_reversal_to_buy",
        "buy_volume": 0.202,
        "sell_volume": 0.195
      }
    },
    "tipo_absorcao": "Neutra",
    "participant_analysis": {
      "retail": {
        "volume_pct": 100,
        "direction": "NEUTRAL",
        "sentiment": "NEUTRAL",
        "composite_score": 0.44,
        "imbalance": 0.019
      }
    },
    "liquidity_heatmap": {
      "clusters": [
        {
          "center": 64858.5789,
          "low": 64846,
          "high": 64872,
          "width": 26,
          "total_volume": 10.511,
          "buy_volume": 4.832,
          "sell_volume": 5.679,
          "imbalance": -0.847,
          "imbalance_ratio": -0.081,
          "trades_count": 2000,
          "avg_trade_size": 0.005,
          "recent_timestamp": 1786142464586,
          "recent_ts_ms": 1786142464586,
          "last_seen_ms": 1786142464586,
          "first_seen_ms": 1786142264050,
          "age_ms": 337.0,
          "cluster_duration_ms": 200536,
          "price_std": 8.8559,
          "volume_std": 0.033,
          "bin_threshold_usd": 194.5757
        }
      ],
      "clusters_count": 1
    },
    "absorption_analysis": {
      "current_absorption": {
        "index": 0.0002,
        "classification": "NONE",
        "label": "Neutra",
        "buyer_strength": 5.1,
        "seller_exhaustion": 0.2,
        "continuation_probability": 0,
        "delta_usd": 229,
        "total_volume_usd": 25717.85,
        "flow_imbalance": 0.0187,
        "window_min": 1
      }
    }
  },
  "market_context": {
    "trading_session": "NY",
    "session_phase": "ACTIVE",
    "time_to_session_close": 11964,
    "day_of_week": 4,
    "is_holiday": 0,
    "market_hours_type": "EXTENDED"
  },
  "market_environment": {
    "volatility_regime": "NORMAL",
    "trend_direction": "UP",
    "market_structure": "RANGE_BOUND",
    "liquidity_environment": "NORMAL",
    "risk_sentiment": "BULLISH",
    "correlation_spy": 0.4056,
    "correlation_dxy": -0.0854,
    "correlation_gold": 0.2379
  },
  "contextual_snapshot": {
    "symbol": "BTCUSDT",
    "ohlc": {
      "open": 64872,
      "high": 64872,
      "low": 64872,
      "close": 64872,
      "open_time": 1786142400265,
      "close_time": 1786142456614,
      "vwap": 64872
    },
    "volume_total": 0.396,
    "volume_total_usdt": 25718,
    "volume_compra": 0.202,
    "volume_venda": 0.195,
    "num_trades": 97,
    "delta_minimo": -0.064,
    "delta_maximo": 0.06,
    "delta_fechamento": 0.007,
    "reversao_desde_minimo": 0.07,
    "reversao_desde_maximo": 0.05,
    "poc_price": 64872,
    "poc_volume": 0.2,
    "poc_percentage": 50.9,
    "dwell_price": 64872,
    "dwell_seconds": 56,
    "dwell_location": "Low",
    "trades_per_second": 1.72,
    "avg_trade_size": 0.004
  },
  "order_book_depth": {
    "L1": {
      "bids": 670995.36,
      "asks": 366947.1,
      "flow_imbalance": 0.2929
    },
    "L5": {
      "bids": 763331.53,
      "asks": 389123.51,
      "flow_imbalance": 0.3247
    },
    "L10": {
      "bids": 882187.35,
      "asks": 400989.95,
      "flow_imbalance": 0.375
    },
    "L25": {
      "bids": 1040592.07,
      "asks": 448132.42,
      "flow_imbalance": 0.398
    },
    "total_depth_ratio": 2.32
  },
  "orderbook_quality": "live",
  "market_impact": {
    "slippage_matrix": {
      "100k_usd": {
        "buy": 0.05,
        "sell": 0.05
      },
      "1m_usd": {
        "buy": 5.45,
        "sell": 3.05
      }
    },
    "liquidity_score": 9.9985,
    "execution_quality": "EXCELLENT"
  },
  "institutional_analytics": {
    "status": "ok",
    "computed_at_ms": 1786142481966,
    "technical_extras": {
      "stoch_rsi": {
        "k": 39.05,
        "d": 19.52,
        "overbought": 0,
        "oversold": 0,
        "crossover": "none"
      },
      "williams_r": {
        "value": -67.09,
        "overbought": 0,
        "oversold": 0,
        "zone": "neutral",
        "source": "real"
      },
      "hurst_exponent": 0.3256,
      "shannon_entropy": 2.935,
      "kalman_filter": {
        "kalman_price": 64927.58,
        "raw_price": 64872,
        "deviation_pct": -0.09,
        "trend_direction": "DOWN"
      },
      "regression_channel": {
        "slope_per_bar": -1.1443,
        "trend_price": 64890.24,
        "upper_1sd": 64907.05,
        "lower_1sd": 64873.44,
        "upper_2sd": 64923.86,
        "lower_2sd": 64856.63,
        "deviation_from_trend": -18.24,
        "position_in_channel": 0.2286
      },
      "dominant_cycles": {
        "dominant_cycles": [
          100,
          40,
          33.3
        ],
        "cycle_strengths": [
          3098.95,
          1931.7,
          990.03
        ]
      },
      "fractal_dimension": 0.6229,
      "monte_carlo": {
        "median_price": 64866.32,
        "p10": 64809.39,
        "p25": 64837.82,
        "p75": 64899.54,
        "p90": 64929.91,
        "prob_up": 0.45,
        "horizon_bars": 12
      },
      "fair_value_gaps": [
        {
          "type": "BEARISH",
          "top": 64925,
          "bottom": 64892.3,
          "gap_size": 32.7,
          "gap_pct": 0.05
        }
      ],
      "market_structure": {
        "structure": "BEARISH",
        "bos_detected": 0,
        "last_swing_high": 64925,
        "last_swing_low": 64909.55,
        "higher_highs": 0,
        "higher_lows": 0
      }
    },
    "profile_analysis": {
      "poor_extremes": {
        "poor_high": {
          "detected": 1,
          "price": 64872,
          "volume_ratio": 5.093,
          "implication": "High likely to be revisited - unfinished auction"
        },
        "poor_low": {
          "detected": 1,
          "price": 64871.99,
          "volume_ratio": 4.907,
          "implication": "Low likely to be revisited - unfinished auction"
        },
        "excess_high": 0,
        "excess_low": 0,
        "session_high": 64872,
        "session_low": 64871.99,
        "action_bias": "expect_retest_both",
        "status": "success"
      },
      "profile_shape": {
        "shape": "P",
        "implication": "Short covering rally - bearish bias expected",
        "trading_signal": "BEARISH_AFTER",
        "distribution": {
          "lower_third_pct": 49.1,
          "middle_third_pct": 0,
          "upper_third_pct": 50.9
        },
        "dominant_zone": "upper",
        "status": "success"
      },
      "no_mans_land": {
        "zones": [
          {
            "range_low": 64512,
            "range_high": 64810,
            "gap_size": 298,
            "gap_size_pct": 0.462,
            "risk": "LOW",
            "lvn_confirmed": 1,
            "lvns_in_zone": 9,
            "nearest_hvn_below": 64512,
            "nearest_hvn_above": 64810,
            "distance_from_price": 62,
            "distance_pct": 0.1,
            "direction": "below"
          }
        ],
        "price_in_no_mans_land": 0,
        "nearest_no_mans_land": {
          "range_low": 64512,
          "range_high": 64810,
          "gap_size": 298,
          "gap_size_pct": 0.462,
          "risk": "LOW",
          "lvn_confirmed": 1,
          "lvns_in_zone": 9,
          "nearest_hvn_below": 64512,
          "nearest_hvn_above": 64810,
          "distance_from_price": 62,
          "distance_pct": 0.1,
          "direction": "below"
        },
        "total_zones": 1,
        "max_gap_pct": 0.46,
        "status": "success"
      },
      "va_volume_pct": {
        "value_area_volume_pct": 100,
        "interpretation": "extremely_compressed",
        "breakout_risk": "VERY_HIGH",
        "volume_in_va": 1,
        "total_volume": 1,
        "compression_signal": 1
      },
      "volume_node_strength": {
        "scored_hvns": [
          {
            "price": 64304,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64306,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64314,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64320,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64322,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.8,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          }
        ],
        "scored_lvns": [
          {
            "price": 64423,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.5,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64436,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.6,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64440,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64447,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64448,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          }
        ],
        "total_hvns": 52,
        "total_lvns": 101,
        "avg_hvn_strength": 66.3,
        "avg_lvn_strength": 63.1,
        "status": "success"
      }
    },
    "flow_analysis": {
      "passive_aggressive": {
        "aggressive": {
          "buy_pct": 50.93,
          "sell_pct": 49.07,
          "net_pct": 1.86,
          "dominance": "balanced",
          "buy_volume": 0.202,
          "sell_volume": 0.195
        },
        "passive": {
          "dominance": "buyers",
          "inference": "from_orderbook_depth",
          "bid_depth": 1307853.23,
          "ask_depth": 837991.67,
          "bid_ratio": 0.61,
          "ob_imbalance": 0.219
        },
        "composite": {
          "signal": "mixed",
          "interpretation": "Mixed signals between aggressive and passive flow",
          "conviction": "LOW"
        },
        "status": "success"
      },
      "whale_accumulation": {
        "score": -9,
        "classification": "NEUTRAL",
        "bias": "NEUTRAL",
        "components": {
          "flow": {
            "score": -30,
            "max": 30,
            "detail": {
              "divergence": "aligned",
              "whale_delta": -4,
              "mid_delta": -2.44,
              "retail_delta": -16.522,
              "primary_delta": -4
            }
          },
          "depth": {
            "score": 4.38,
            "max": 20,
            "detail": {
              "bid_depth": 1307853.23,
              "ask_depth": 837991.67,
              "ratio": 0.22,
              "deep_confirmation": 0
            }
          },
          "absorption": {
            "score": 14.7,
            "max": 25,
            "detail": {
              "buyer_strength": 5.1,
              "seller_exhaustion": 0.2,
              "net_absorption": 4.9,
              "index": 0.0002,
              "label": "Neutra"
            }
          },
          "derivatives": {
            "score": 1.49,
            "max": 25,
            "detail": {
              "long_short_ratio": 1.1,
              "lsr_score": 1.49
            }
          }
        },
        "trend": {
          "direction": "stable",
          "avg_score": -25.9,
          "recent_avg": -30.8,
          "momentum": 16.9,
          "samples": 11,
          "score_range": {
            "min": -49,
            "max": 14
          }
        },
        "status": "success"
      },
      "absorption_zones": {
        "total_zones": 0,
        "total_events": 0,
        "buy_zone_count": 0,
        "sell_zone_count": 0,
        "status": "no_events"
      }
    },
    "sr_analysis": {
      "defense_zones": {
        "buy_defense": [
          {
            "center": 64479.84,
            "range_low": 64376.35,
            "range_high": 64570.65,
            "strength": 62,
            "side": "buy",
            "sources": [
              "sr_level_ema_21_4h",
              "vp_hvn",
              "ema_ema_21_4h",
              "vp_val"
            ],
            "source_count": 4,
            "signals_in_zone": 5,
            "type": "confluence",
            "distance_from_price": 392.16,
            "distance_pct": 0.6
          },
          {
            "center": 64833.68,
            "range_low": 64740.15,
            "range_high": 64928.65,
            "strength": 61,
            "side": "buy",
            "sources": [
              "orderbook_bid_wall",
              "vp_hvn",
              "ema_ema_21_1h",
              "sr_level_ema_21_1h"
            ],
            "source_count": 4,
            "signals_in_zone": 8,
            "type": "confluence",
            "distance_from_price": 38.32,
            "distance_pct": 0.06
          },
          {
            "center": 64162.07,
            "range_low": 64096.12,
            "range_high": 64228.03,
            "strength": 46,
            "side": "buy",
            "sources": [
              "ema_ema_21_1d",
              "sr_level_vah_monthly"
            ],
            "source_count": 2,
            "signals_in_zone": 2,
            "type": "cluster",
            "distance_from_price": 709.93,
            "distance_pct": 1.09
          },
          {
            "center": 64336.25,
            "range_low": 64255.35,
            "range_high": 64442.65,
            "strength": 40,
            "side": "buy",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 9,
            "type": "cluster",
            "distance_from_price": 535.75,
            "distance_pct": 0.83
          }
        ],
        "sell_defense": [
          {
            "center": 65037.38,
            "range_low": 64960.35,
            "range_high": 65128.65,
            "strength": 55,
            "side": "sell",
            "sources": [
              "sr_level_poc_daily",
              "vp_poc",
              "vp_hvn"
            ],
            "source_count": 3,
            "signals_in_zone": 9,
            "type": "confluence",
            "distance_from_price": 165.38,
            "distance_pct": 0.25
          },
          {
            "center": 65346,
            "range_low": 65297.35,
            "range_high": 65394.65,
            "strength": 53,
            "side": "sell",
            "sources": [
              "sr_level_vah_daily",
              "vp_hvn",
              "vp_vah"
            ],
            "source_count": 3,
            "signals_in_zone": 3,
            "type": "confluence",
            "distance_from_price": 474,
            "distance_pct": 0.73
          },
          {
            "center": 64944.5,
            "range_low": 64860.35,
            "range_high": 65052.65,
            "strength": 49,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "ema_ema_21_15m",
              "sr_level_hvn_daily"
            ],
            "source_count": 3,
            "signals_in_zone": 9,
            "type": "confluence",
            "distance_from_price": 72.5,
            "distance_pct": 0.11
          },
          {
            "center": 65157.17,
            "range_low": 65064.35,
            "range_high": 65257.65,
            "strength": 41,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 13,
            "type": "cluster",
            "distance_from_price": 285.17,
            "distance_pct": 0.44
          },
          {
            "center": 65252.3,
            "range_low": 65164.35,
            "range_high": 65350.65,
            "strength": 40,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 11,
            "type": "cluster",
            "distance_from_price": 380.3,
            "distance_pct": 0.59
          }
        ],
        "total_zones": 9,
        "strongest_buy": {
          "center": 64479.84,
          "range_low": 64376.35,
          "range_high": 64570.65,
          "strength": 62,
          "side": "buy",
          "sources": [
            "sr_level_ema_21_4h",
            "vp_hvn",
            "ema_ema_21_4h",
            "vp_val"
          ],
          "source_count": 4,
          "signals_in_zone": 5,
          "type": "confluence",
          "distance_from_price": 392.16,
          "distance_pct": 0.6
        },
        "strongest_sell": {
          "center": 65037.38,
          "range_low": 64960.35,
          "range_high": 65128.65,
          "strength": 55,
          "side": "sell",
          "sources": [
            "sr_level_poc_daily",
            "vp_poc",
            "vp_hvn"
          ],
          "source_count": 3,
          "signals_in_zone": 9,
          "type": "confluence",
          "distance_from_price": 165.38,
          "distance_pct": 0.25
        },
        "defense_asymmetry": {
          "ratio": 0.88,
          "bias": "slight_sell_defense",
          "description": "Slightly more sell defense",
          "buy_total_strength": 209,
          "sell_total_strength": 238
        },
        "status": "success"
      }
    },
    "quality": {
      "calendar": {
        "day_of_week": "Friday",
        "day_of_week_num": 4,
        "is_us_holiday": 0,
        "is_weekend": 0,
        "is_pre_holiday": 0,
        "is_post_holiday": 0,
        "expected_liquidity": "NORMAL",
        "liquidity_warning": 0
      },
      "latency": {
        "latency_ms": 22176,
        "latency_category": "CRITICAL",
        "data_freshness": "STALE",
        "is_acceptable": 0,
        "is_stale": 1
      },
      "spread_percentile": {
        "status": "ok",
        "current_spread": 0.1,
        "spread_percentile": 0,
        "spread_mean": 0.1,
        "spread_median": 0.1,
        "spread_std": 0,
        "spread_min": 0.1,
        "spread_max": 0.1,
        "spread_z_score": 0,
        "is_tight": 1,
        "is_wide": 0,
        "is_anomalous": 0,
        "liquidity_signal": "EXCELLENT",
        "samples": 11,
        "window_minutes": 1440
      },
      "anomalies": {
        "anomalies_detected": 0,
        "count": 0,
        "max_severity": "NONE",
        "risk_elevated": 0,
        "summary": "No anomalies detected"
      }
    },
    "candlestick_patterns": {
      "patterns_detected": 1,
      "patterns": [
        {
          "name": "doji",
          "type": "neutral",
          "confidence": 0.65,
          "candles_used": 1,
          "implication": "Indecision - watch next candle for direction"
        }
      ],
      "dominant_signal": "neutral",
      "max_confidence": 0.65,
      "bullish_count": 0,
      "bearish_count": 0,
      "neutral_count": 1
    }
  },
  "sequence_id": 11,
  "primary_exchange": "BINANCE",
  "data_feed_type": "WEBSOCKET_L2",
  "data_quality_score": 9.5,
  "completeness_pct": 100,
  "reliability_score": 9,
  "bid": 64843,
  "ask": 64843.1,
  "tick_direction": 0,
  "twap": 64883.78,
  "pivot_points": {
    "daily": {
      "pivot": 64971.33,
      "r1": 65420.66,
      "r2": 65795.33,
      "r3": 66619.33,
      "s1": 64596.66,
      "s2": 64147.33,
      "s3": 63323.33,
      "vah": 65346,
      "val": 64522,
      "poc": 65046
    },
    "weekly": {
      "pivot": 62830.33,
      "r1": 63346.66,
      "r2": 64083.33,
      "r3": 65336.33,
      "s1": 62093.66,
      "s2": 61577.33,
      "s3": 60324.33,
      "vah": 63567,
      "val": 62314,
      "poc": 62610
    },
    "monthly": {
      "pivot": 63185.67,
      "r1": 64582.34,
      "r2": 65610.67,
      "r3": 68035.67,
      "s1": 62157.34,
      "s2": 60760.67,
      "s3": 58335.67,
      "vah": 64214,
      "val": 61789,
      "poc": 63554
    }
  },
  "immediate_support": [
    64833.68,
    64522,
    64479.84,
    64214,
    64162.07
  ],
  "support_strength": [
    61,
    94.6,
    62,
    71.9,
    46
  ],
  "immediate_resistance": [
    64944.5,
    64971.33,
    65037.38,
    65346
  ],
  "resistance_strength": [
    49,
    98.5,
    55,
    92.7
  ],
  "volatility_metrics": {
    "realized_vol_24h": 0.0184,
    "realized_vol_7d": 0.0487,
    "volatility_regime": "NORMAL"
  },
  "order_flow_extended": {
    "passive_buy_pct": 60.9,
    "passive_sell_pct": 39.1
  },
  "whale_activity": {
    "large_orders_1h": [
      {
        "size": 1,
        "price": 64896.34,
        "side": "SELL",
        "timestamp_ms": 1786141945725
      },
      {
        "size": 1,
        "price": 64883.14,
        "side": "SELL",
        "timestamp_ms": 1786142044496
      },
      {
        "size": 1,
        "price": 64870.63,
        "side": "SELL",
        "timestamp_ms": 1786142264045
      },
      {
        "size": 1,
        "price": 64853.03,
        "side": "SELL",
        "timestamp_ms": 1786142272654
      }
    ],
    "iceberg_activity": 1,
    "hidden_orders_detected": 0
  },
  "technical_indicators_extended": {
    "cci_1h": 97.67,
    "cci_signal": "NEUTRAL",
    "stochastic": {
      "k": 39.05,
      "d": 19.52,
      "signal": "NEUTRAL",
      "source": "real"
    },
    "williams_r": {
      "value": -67.09,
      "overbought": 0,
      "oversold": 0,
      "zone": "neutral",
      "source": "real"
    },
    "hurst_exponent": 0.3256,
    "shannon_entropy": 2.935,
    "fractal_dimension": 0.6229,
    "kalman_filter": {
      "kalman_price": 64927.58,
      "raw_price": 64872,
      "deviation_pct": -0.09,
      "trend_direction": "DOWN"
    },
    "regression_channel": {
      "slope_per_bar": -1.1443,
      "trend_price": 64890.24,
      "upper_1sd": 64907.05,
      "lower_1sd": 64873.44,
      "upper_2sd": 64923.86,
      "lower_2sd": 64856.63,
      "deviation_from_trend": -18.24,
      "position_in_channel": 0.2286
    },
    "dominant_cycles": {
      "dominant_cycles": [
        100,
        40,
        33.3
      ],
      "cycle_strengths": [
        3098.95,
        1931.7,
        990.03
      ]
    },
    "monte_carlo": {
      "median_price": 64866.32,
      "p10": 64809.39,
      "p25": 64837.82,
      "p75": 64899.54,
      "p90": 64929.91,
      "prob_up": 0.45,
      "horizon_bars": 12
    },
    "garch_forecast_1h": 0.0016
  },
  "backup_exchanges": [
    "COINBASE",
    "KRAKEN",
    "OKX"
  ],
  "alerts": {
    "active_alerts": [
      {
        "type": "SUPPORT_TEST",
        "level": 64833.68,
        "severity": "HIGH",
        "probability": 0.88,
        "action": "MONITOR_CLOSELY",
        "description": "Preço testando suporte em 64833.68 (dist: 0.06%)"
      },
      {
        "type": "RESISTANCE_TEST",
        "level": 64944.5,
        "severity": "MEDIUM",
        "probability": 0.78,
        "action": "MONITOR_CLOSELY",
        "description": "Preço testando resistência em 64944.50 (dist: 0.11%)"
      }
    ],
    "alert_count": 2,
    "max_severity": "HIGH"
  },
  "regime_analysis": {
    "current_regime": "MEAN_REVERTING",
    "regime_probabilities": {
      "trending": 0,
      "mean_reverting": 1,
      "breakout": 0
    },
    "regime_change_probability": 0.05,
    "expected_regime_duration": "15m-1h",
    "avg_adx": 25.9
  },
  "data_reliability": {
    "has_options_data": 0,
    "onchain_coverage": "full",
    "latency_acceptable": 1,
    "price_targets_available": 1
  },
  "event_id": "8a04173a",
  "timestamp_ny": "2026-08-07T18:41:00.000-04:00",
  "timestamp_sp": "2026-08-07T19:41:00.000-03:00",
  "_log_id": "8a04173a"
}
----------------------------------------------------------------------------------------------------
EVENTO: Alerta | JANELA: 11
UTC: 2026-08-07 22:41:22 UTC
NY:  2026-08-07 18:41:22 EST/EDT
SP:  2026-08-07 19:41:22 BRT
----------------------------------------------------------------------------------------------------
{
  "tipo_evento": "Alerta",
  "resultado_da_batalha": "VOLATILITY_SQUEEZE",
  "descricao": "Tipo: VOLATILITY_SQUEEZE",
  "timestamp": "2026-08-07T22:41:22+00:00",
  "severity": "LOW",
  "probability": 0.4,
  "action": "WATCH_FOR_EXPANSION",
  "context": {
    "price": 64872,
    "volume": 0.396,
    "average_volume": 4.046,
    "volatility": 6e-08
  },
  "data_context": "real_time",
  "janela_numero": 11,
  "epoch_ms": 1786142482499,
  "event_id": "a2e84d90",
  "timestamp_utc": "2026-08-07T22:41:22.499+00:00",
  "timestamp_ny": "2026-08-07T18:41:22.499-04:00",
  "timestamp_sp": "2026-08-07T19:41:22.499-03:00",
  "price_data": {
    "current": {
      "last": 64872,
      "volume": 0.396
    }
  },
  "volatility_metrics": {
    "realized_vol_24h": 0
  },
  "market_context": {
    "trading_session": "NY_OVERLAP",
    "session_phase": "ACTIVE"
  },
  "_log_id": "a2e84d90"
}

----------------------------------------------------------------------------------------------------
 # Janela 12
----------------------------------------------------------------------------------------------------
EVENTO: Exaustão | SYMBOL: BTCUSDT | JANELA: 12
UTC: 2026-08-07 22:42:00 UTC
NY:  2026-08-07 18:42:00 EST/EDT
SP:  2026-08-07 19:42:00 BRT
----------------------------------------------------------------------------------------------------
{
  "is_signal": 1,
  "tipo_evento": "Exaustão",
  "resultado_da_batalha": "Exaustão de Venda",
  "descricao": "Pico de venda 11.11 vs média 4.05",
  "ativo": "BTCUSDT",
  "window_open_ms": 1786142460293,
  "window_close_ms": 1786142519917,
  "window_duration_ms": 59624,
  "window_id": 1786142519917,
  "volume_total_btc": 11.105,
  "volume_compra_btc": 2.746,
  "volume_venda_btc": 8.359,
  "buy_notional_usdt": 178096.2612,
  "sell_notional_usdt": 542212.7234,
  "total_notional_usdt": 720308.9847,
  "volume_total": 11.105,
  "volume_compra": 2.746,
  "volume_venda": 8.359,
  "preco_abertura": 64872,
  "preco_maxima": 64872,
  "preco_minima": 64846,
  "preco_fechamento": 64856.34,
  "ohlc": {
    "open": 64872,
    "high": 64872,
    "low": 64846,
    "close": 64856.34
  },
  "delta_minimo": -8.094,
  "delta_maximo": 0.026,
  "delta_fechamento": -5.613,
  "reversao_desde_minimo": 2.4813,
  "reversao_desde_maximo": 5.6381,
  "dwell_price": 64871.35,
  "dwell_seconds": 39.379,
  "dwell_location": "High",
  "trades_per_second": 26.181,
  "avg_trade_size": 0.007,
  "layer": "signal",
  "data_context": "real_time",
  "source": {
    "exchange": "binance_futures",
    "stream": "trades"
  },
  "poc_price": 64871.278,
  "vah": 64871.278,
  "val": 64861.1665,
  "hvns": [
    64871.278
  ],
  "lvns": [64846.7215, 64848.1665, 64851.0555, ..., 64865.5, 64868.389, 64869.8335],
  "vpd_params": {
    "dynamic_bins": 18,
    "value_area_pct": 0.65,
    "hvn_sensitivity": 1.5,
    "lvn_sensitivity": 1.5,
    "volatility_factor": 0.5,
    "whale_factor": 1.4502,
    "trend_factor": 1.3
  },
  "fluxo_continuo": {
    "cvd": -28.5699,
    "whale_buy_volume": 0,
    "whale_sell_volume": 5,
    "whale_delta": -5,
    "bursts": {
      "count": 8,
      "max_burst_volume": 7.558
    },
    "sector_flow": {
      "retail": {
        "buy": 12.0048,
        "sell": 32.5694,
        "delta": -20.565
      },
      "mid": {
        "buy": 1.522,
        "sell": 4.5274,
        "delta": -3.005
      },
      "whale": {
        "buy": 0,
        "sell": 5,
        "delta": -5
      }
    },
    "timestamp": "2026-08-07T22:42:00.000+00:00",
    "time_index": {
      "epoch_ms": 1786142520000,
      "timestamp_utc": "2026-08-07T22:42:00.000+00:00",
      "timestamp_ny": "2026-08-07T18:42:00.000-04:00",
      "timestamp_sp": "2026-08-07T19:42:00.000-03:00"
    },
    "metadata": {
      "burst_window_ms": 200,
      "in_burst": 0,
      "last_reset_ms": 1786141813933,
      "config_version": 1,
      "num_trades": 1574,
      "window_sec": 60
    },
    "order_flow": {
      "net_flow_1m": 160810.2968,
      "absorcao_1m": "Neutra",
      "buy_volume": 178096.26,
      "sell_volume": 542212.72,
      "total_volume": 720308.98,
      "ui_sum_ok": 1,
      "buy_volume_btc": 2.746,
      "sell_volume_btc": 8.359,
      "total_volume_btc": 11.105,
      "whale_buy_volume_window": 0,
      "whale_sell_volume_window": 1,
      "whale_delta_window": -1,
      "flow_imbalance": -0.5055,
      "aggressive_buy_pct": 24.72,
      "aggressive_sell_pct": 75.28,
      "net_flow_5m": -196915.5833,
      "absorcao_5m": "Neutra",
      "net_flow_15m": -1854091.4037,
      "absorcao_15m": "Neutra",
      "computation_window_min": 1,
      "available_windows_min": [
        1,
        5,
        15
      ],
      "buy_sell_ratio": {
        "buy_sell_ratio": 0.33,
        "ratios": {
          "current": 0.3285,
          "imbalance_1m": 0.223,
          "imbalance_5m": -0.273,
          "imbalance_15m": -2.574
        },
        "sector_ratios": {
          "retail": 0.3686,
          "mid": 0.3362,
          "whale": 0
        },
        "pressure": "STRONG_SELL",
        "flow_trend": "short_term_reversal_to_buy",
        "buy_volume": 2.746,
        "sell_volume": 8.359
      }
    },
    "tipo_absorcao": "Neutra",
    "participant_analysis": {
      "retail": {
        "volume_pct": 85.9,
        "direction": "SELL",
        "sentiment": "BEARISH",
        "composite_score": -0.713,
        "imbalance": -0.424
      },
      "mid": {
        "volume_pct": 5.09,
        "direction": "SELL",
        "sentiment": "BEARISH",
        "composite_score": -0.421,
        "imbalance": -1
      },
      "whale": {
        "volume_pct": 9,
        "direction": "SELL",
        "sentiment": "BEARISH",
        "composite_score": -0.436,
        "imbalance": -1
      }
    },
    "liquidity_heatmap": {
      "clusters": [
        {
          "center": 64862.4746,
          "low": 64846,
          "high": 64872,
          "width": 26,
          "total_volume": 13.606,
          "buy_volume": 3.607,
          "sell_volume": 9.999,
          "imbalance": -6.391,
          "imbalance_ratio": -0.47,
          "trades_count": 2000,
          "avg_trade_size": 0.007,
          "recent_timestamp": 1786142526893,
          "recent_ts_ms": 1786142526893,
          "last_seen_ms": 1786142526893,
          "first_seen_ms": 1786142315761,
          "age_ms": 265.0,
          "cluster_duration_ms": 211132,
          "price_std": 10.0286,
          "volume_std": 0.037,
          "bin_threshold_usd": 194.5874
        }
      ],
      "resistances": [
        64862.4746
      ],
      "clusters_count": 1
    },
    "absorption_analysis": {
      "current_absorption": {
        "index": 0.1129,
        "classification": "WEAK_ABSORPTION",
        "label": "Neutra",
        "buyer_strength": 2.5,
        "seller_exhaustion": 5.1,
        "continuation_probability": 0.1,
        "delta_usd": 160810.297,
        "total_volume_usd": 720308.98,
        "flow_imbalance": -0.5055,
        "window_min": 1
      }
    },
    "data_quality": {
      "total_trades_processed": 7554,
      "invalid_trades": 0,
      "valid_rate_pct": 100,
      "flow_trades_count": 1574,
      "processing_time_ms": 8.323100000325212
    },
    "observability": {
      "processing_times_ms": {
        "p50": 0.05870000040886225,
        "p95": 0.12250000008862116,
        "p99": 0.20599999970727367,
        "max": 0.3475999997135659,
        "min": 0.04579999995257822,
        "avg": 0.07280430000537308,
        "count": 1000,
        "total_recorded": 7554
      },
      "memory": {
        "flow_trades_size": 7554,
        "flow_trades_capacity": 100000
      },
      "circuit_breaker": {
        "state": "CLOSED",
        "failures": 0,
        "successes": 7554,
        "time_in_state_ms": 713639,
        "recovery_remaining_ms": 0,
        "threshold": 5
      }
    },
    "invariants_ok": 1
  },
  "historical_vp": {
    "daily": {
      "poc": 65046,
      "vah": 65346,
      "val": 64522,
      "hvns": [64304, 64306, 64314, ..., 65301, 65302, 65346],
      "lvns": [64179, 64180, 64182, ..., 65103, 65121, 65215],
      "single_prints": [64223, 64255, 64266, ..., 65045, 65121, 65200],
      "volume_nodes": {
        "hvn_nodes": [
          {
            "price": 64304.0,
            "vol": 67.74,
            "str": 3.06
          },
          {
            "price": 64306.0,
            "vol": 53.94,
            "str": 2.44
          },
          {
            "price": 64314.0,
            "vol": 53.8,
            "str": 2.43
          },
          {
            "price": 64320.0,
            "vol": 60.97,
            "str": 2.76
          },
          {
            "price": 64322.0,
            "vol": 72.71,
            "str": 3.29
          },
          {
            "price": 65255.0,
            "vol": 51.04,
            "str": 2.31
          },
          {
            "price": 65300.0,
            "vol": 66.03,
            "str": 2.99
          },
          {
            "price": 65301.0,
            "vol": 39.67,
            "str": 1.79
          },
          {
            "price": 65302.0,
            "vol": 46.38,
            "str": 2.1
          },
          {
            "price": 65346.0,
            "vol": 106.18,
            "str": 4.8
          }
        ],
        "lvn_nodes": [
          {
            "price": 64179.0,
            "vol": 4.24,
            "str": 9.81
          },
          {
            "price": 64180.0,
            "vol": 3.58,
            "str": 9.84
          },
          {
            "price": 64182.0,
            "vol": 3.93,
            "str": 9.82
          },
          {
            "price": 64183.0,
            "vol": 3.28,
            "str": 9.85
          },
          {
            "price": 64187.0,
            "vol": 2.65,
            "str": 9.88
          },
          {
            "price": 65077.0,
            "vol": 5.06,
            "str": 9.77
          },
          {
            "price": 65090.0,
            "vol": 5.45,
            "str": 9.75
          },
          {
            "price": 65103.0,
            "vol": 3.82,
            "str": 9.83
          },
          {
            "price": 65121.0,
            "vol": 3.83,
            "str": 9.83
          },
          {
            "price": 65215.0,
            "vol": 4.63,
            "str": 9.79
          }
        ]
      },
      "status": "success"
    },
    "weekly": {
      "poc": 62610,
      "vah": 63567,
      "val": 62314,
      "hvns": [62399, 62418, 62598, ..., 63967, 63994, 64162],
      "lvns": [62506, 62515, 62524, ..., 63805, 63825, 63862],
      "single_prints": [62506, 62550, 62558, ..., 63825, 63882, 63885],
      "volume_nodes": {
        "hvn_nodes": [
          {
            "price": 62399.0,
            "vol": 294.08,
            "str": 7.51
          },
          {
            "price": 62418.0,
            "vol": 128.04,
            "str": 3.27
          },
          {
            "price": 62598.0,
            "vol": 138.69,
            "str": 3.54
          },
          {
            "price": 62600.0,
            "vol": 249.32,
            "str": 6.37
          },
          {
            "price": 62610.0,
            "vol": 391.62,
            "str": 10.0
          },
          {
            "price": 63884.0,
            "vol": 212.6,
            "str": 5.43
          },
          {
            "price": 63886.0,
            "vol": 203.09,
            "str": 5.19
          },
          {
            "price": 63967.0,
            "vol": 240.55,
            "str": 6.14
          },
          {
            "price": 63994.0,
            "vol": 161.99,
            "str": 4.14
          },
          {
            "price": 64162.0,
            "vol": 249.07,
            "str": 6.36
          }
        ],
        "lvn_nodes": [
          {
            "price": 62506.0,
            "vol": 15.07,
            "str": 9.62
          },
          {
            "price": 62515.0,
            "vol": 8.52,
            "str": 9.78
          },
          {
            "price": 62524.0,
            "vol": 10.81,
            "str": 9.72
          },
          {
            "price": 62530.0,
            "vol": 15.54,
            "str": 9.6
          },
          {
            "price": 62550.0,
            "vol": 16.4,
            "str": 9.58
          },
          {
            "price": 63716.0,
            "vol": 9.76,
            "str": 9.75
          },
          {
            "price": 63778.0,
            "vol": 12.59,
            "str": 9.68
          },
          {
            "price": 63805.0,
            "vol": 15.71,
            "str": 9.6
          },
          {
            "price": 63825.0,
            "vol": 13.27,
            "str": 9.66
          },
          {
            "price": 63862.0,
            "vol": 10.6,
            "str": 9.73
          }
        ]
      },
      "status": "success"
    },
    "monthly": {
      "poc": 63554,
      "vah": 64214,
      "val": 61789,
      "hvns": [62102, 62318, 62332, ..., 65164, 65313, 65380],
      "lvns": [62091, 62211, 62222, ..., 64914, 64968, 65010],
      "single_prints": [62714, 62839, 62841, ..., 64776, 64779, 64783],
      "volume_nodes": {
        "hvn_nodes": [
          {
            "price": 62102.0,
            "vol": 575.96,
            "str": 3.29
          },
          {
            "price": 62318.0,
            "vol": 860.81,
            "str": 4.91
          },
          {
            "price": 62332.0,
            "vol": 803.52,
            "str": 4.59
          },
          {
            "price": 62500.0,
            "vol": 574.93,
            "str": 3.28
          },
          {
            "price": 62556.0,
            "vol": 676.49,
            "str": 3.86
          },
          {
            "price": 65038.0,
            "vol": 473.55,
            "str": 2.7
          },
          {
            "price": 65118.0,
            "vol": 490.02,
            "str": 2.8
          },
          {
            "price": 65164.0,
            "vol": 722.65,
            "str": 4.13
          },
          {
            "price": 65313.0,
            "vol": 570.81,
            "str": 3.26
          },
          {
            "price": 65380.0,
            "vol": 540.93,
            "str": 3.09
          }
        ],
        "lvn_nodes": [
          {
            "price": 62091.0,
            "vol": 47.57,
            "str": 9.73
          },
          {
            "price": 62211.0,
            "vol": 60.49,
            "str": 9.65
          },
          {
            "price": 62222.0,
            "vol": 58.81,
            "str": 9.66
          },
          {
            "price": 62290.0,
            "vol": 40.82,
            "str": 9.77
          },
          {
            "price": 62325.0,
            "vol": 42.62,
            "str": 9.76
          },
          {
            "price": 64880.0,
            "vol": 52.7,
            "str": 9.7
          },
          {
            "price": 64897.0,
            "vol": 44.58,
            "str": 9.75
          },
          {
            "price": 64914.0,
            "vol": 45.53,
            "str": 9.74
          },
          {
            "price": 64968.0,
            "vol": 50.61,
            "str": 9.71
          },
          {
            "price": 65010.0,
            "vol": 36.97,
            "str": 9.79
          }
        ]
      },
      "status": "success"
    }
  },
  "timestamp": "2026-08-07T22:42:00.000Z",
  "epoch_ms": 1786142520000,
  "timestamp_utc": "2026-08-07T22:42:00.000Z",
  "timestamp_ny": "2026-08-07T18:42:00.000-04:00",
  "timestamp_sp": "2026-08-07T19:42:00.000-03:00",
  "event_id": "1b84c3d82fa62cbac78d6789338a640d8c527cf2f0432ca5f569cad2af3a88b8",
  "trades_count": 1561,
  "duration_s": 59.62,
  "tick_context_out": {
    "last_price": 64856.34,
    "last_m": 0
  },
  "delta": -5.613,
  "multi_tf": {
    "15m": {
      "tendencia": "Baixa",
      "preco_atual": 64871.99,
      "mme_21": 64925.41,
      "atr": 73.63,
      "regime": "Range",
      "rsi_short": 37.88,
      "rsi_long": 44.06,
      "macd": 12.6535,
      "macd_signal": 16.6562,
      "adx": 23.83,
      "realized_vol": 0.0014
    },
    "1h": {
      "tendencia": "Alta",
      "preco_atual": 64871.99,
      "mme_21": 64788.8,
      "atr": 227.14,
      "regime": "Range",
      "rsi_short": 52.6,
      "rsi_long": 54.35,
      "macd": 100.3548,
      "macd_signal": 98.3005,
      "adx": 23.82,
      "realized_vol": 0.0023
    },
    "4h": {
      "tendencia": "Alta",
      "preco_atual": 64872,
      "mme_21": 64460.35,
      "atr": 498.88,
      "regime": "Range",
      "rsi_short": 61.94,
      "rsi_long": 61.09,
      "macd": 274.9477,
      "macd_signal": 256.7109,
      "adx": 30.01,
      "realized_vol": 0.0058
    },
    "1d": {
      "tendencia": "Alta",
      "preco_atual": 64872,
      "mme_21": 64144.77,
      "atr": 1369.69,
      "regime": "Manipulação",
      "rsi_short": 58.86,
      "rsi_long": 55.63,
      "macd": 67.1517,
      "macd_signal": 35.9228,
      "adx": 15.05,
      "realized_vol": 0.0184
    }
  },
  "pattern_recognition": {
    "smart_money": {
      "fair_value_gaps": [
        {
          "type": "BEARISH",
          "top": 64925,
          "bottom": 64892.3,
          "gap_size": 32.7,
          "gap_pct": 0.05
        }
      ],
      "market_structure": {
        "structure": "BEARISH",
        "bos_detected": 0,
        "last_swing_high": 64925,
        "last_swing_low": 64909.55,
        "higher_highs": 0,
        "higher_lows": 0
      }
    }
  },
  "janela_numero": 12,
  "whale_sell_volume": 5,
  "derivatives": {
    "BTCUSDT": {
      "funding_rate_percent": 0.01,
      "open_interest": 106945.765,
      "open_interest_usd": 6934687958.97,
      "long_short_ratio": 1.1,
      "longs_usd": 3631669164.51,
      "shorts_usd": 3303018794.46
    },
    "ETHUSDT": {
      "funding_rate_percent": 0,
      "open_interest": 2279164.179,
      "open_interest_usd": 4360633657.11,
      "long_short_ratio": 2.07,
      "longs_usd": 2939074192.69,
      "shorts_usd": 1421559464.42
    }
  },
  "market_context": {
    "trading_session": "NY",
    "session_phase": "ACTIVE",
    "time_to_session_close": 11964,
    "day_of_week": 4,
    "is_holiday": 0,
    "market_hours_type": "EXTENDED"
  },
  "market_environment": {
    "volatility_regime": "NORMAL",
    "trend_direction": "UP",
    "market_structure": "RANGE_BOUND",
    "liquidity_environment": "NORMAL",
    "risk_sentiment": "BULLISH",
    "correlation_spy": 0.4056,
    "correlation_dxy": -0.0854,
    "correlation_gold": 0.2379
  },
  "features_window_id": 1786142520000,
  "ml_features": {
    "price_features": {
      "returns_1": 0.0,
      "volatility_1": 0.0,
      "returns_5": 1.5e-07,
      "volatility_5": 1.2e-07,
      "returns_15": 0.0,
      "volatility_15": 1e-07,
      "momentum_score": -0.70710678,
      "volatility_1h": 0.0023
    },
    "volume_features": {
      "volume_sma_ratio": 0.739,
      "volume_momentum": -0.511,
      "buy_sell_pressure": -0.5054,
      "liquidity_gradient": -0.51112327
    },
    "microstructure": {
      "order_book_slope": -4.385968,
      "flow_imbalance": -0.5055,
      "tick_rule_sum": -94,
      "trade_intensity": 40,
      "trade_intensity_v2": 26.2333
    },
    "cross_asset": {
      "btc_eth_corr_7d": 0.8212,
      "btc_eth_corr_30d": 0.8599,
      "btc_dxy_corr_30d": -0.0605,
      "btc_dxy_corr_90d": -0.067,
      "btc_ndx_corr_30d": 0.4147,
      "dxy_return_5d": -0.3561,
      "dxy_return_20d": -1.6548,
      "btc_dxy_correlation_stability": 0.0065,
      "btc_dxy_inverse_strength": 0.0638,
      "dxy_momentum": -1.29867555,
      "vix_current": 14.9,
      "us10y_yield": 4.66,
      "btc_dominance": 17.0377,
      "btc_dominance_change_7d": 0,
      "eth_dominance": 9.0966,
      "gold_price": 4342.3515,
      "oil_price": 77.08,
      "macro_regime": "RISK_ON",
      "correlation_regime": "DECORRELATED"
    },
    "data_quality": {
      "has_price_features": 1,
      "has_volume_features": 1,
      "has_microstructure": 1,
      "has_cross_asset": 1,
      "is_valid": 1
    }
  },
  "contextual_snapshot": {
    "symbol": "BTCUSDT",
    "ohlc": {
      "open": 64872,
      "high": 64872,
      "low": 64846,
      "close": 64856.3,
      "open_time": 1786142460293,
      "close_time": 1786142519917,
      "vwap": 64863.1
    },
    "volume_total": 11.105,
    "volume_total_usdt": 720309,
    "volume_compra": 2.746,
    "volume_venda": 8.359,
    "num_trades": 1561,
    "delta_minimo": -8.094,
    "delta_maximo": 0.025,
    "delta_fechamento": -5.613,
    "reversao_desde_minimo": 2.48,
    "reversao_desde_maximo": 5.64,
    "poc_price": 64871.4,
    "poc_volume": 3.54,
    "poc_percentage": 31.9,
    "dwell_price": 64871.4,
    "dwell_seconds": 39,
    "dwell_location": "High",
    "trades_per_second": 26.18,
    "avg_trade_size": 0.007
  },
  "enriched_snapshot": {
    "symbol": "BTCUSDT",
    "ohlc": {
      "open": 64872,
      "high": 64872,
      "low": 64846,
      "close": 64856.3,
      "open_time": 1786142460293,
      "close_time": 1786142519917,
      "vwap": 64863.1
    },
    "volume_total": 11.105,
    "volume_total_usdt": 720309,
    "volume_compra": 2.746,
    "volume_venda": 8.359,
    "num_trades": 1561,
    "delta_minimo": -8.094,
    "delta_maximo": 0.025,
    "delta_fechamento": -5.613,
    "reversao_desde_minimo": 2.48,
    "reversao_desde_maximo": 5.64,
    "poc_price": 64871.4,
    "poc_volume": 3.54,
    "poc_percentage": 31.9,
    "dwell_price": 64871.4,
    "dwell_seconds": 39,
    "dwell_location": "High",
    "trades_per_second": 26.18,
    "avg_trade_size": 0.007
  },
  "orderbook_data": {
    "mid": 64827.85,
    "spread": 0.1,
    "spread_percent": 0,
    "bid_depth_usd": 4099382.22,
    "ask_depth_usd": 571474.73,
    "imbalance": 0.755,
    "flow_imbalance": 0.7553,
    "volume_ratio": 7.173,
    "pressure": 0.7553,
    "consolidated_bias_score": 0.9266,
    "spread_bps": 0.0154,
    "is_valid": 1,
    "data_source": "live",
    "spread_volatility": 0.0
  },
  "order_book_depth": {
    "L1": {
      "bids": 2417104.52,
      "asks": 285696.56,
      "flow_imbalance": 0.7886
    },
    "L5": {
      "bids": 2501445.35,
      "asks": 289132.44,
      "flow_imbalance": 0.7928
    },
    "L10": {
      "bids": 2779876.48,
      "asks": 294967.03,
      "flow_imbalance": 0.8081
    },
    "L25": {
      "bids": 3287850.83,
      "asks": 299116.14,
      "flow_imbalance": 0.8332
    },
    "total_depth_ratio": 10.99
  },
  "orderbook_quality": "live",
  "market_impact": {
    "slippage_matrix": {
      "100k_usd": {
        "buy": 0.05,
        "sell": 0.05
      },
      "1m_usd": {
        "buy": 5.65,
        "sell": 0.05
      }
    },
    "liquidity_score": 9.9985,
    "execution_quality": "EXCELLENT"
  },
  "institutional_analytics": {
    "status": "ok",
    "computed_at_ms": 1786142528064,
    "technical_extras": {
      "stoch_rsi": {
        "k": 58.57,
        "d": 39.05,
        "overbought": 0,
        "oversold": 0,
        "crossover": "none"
      },
      "williams_r": {
        "value": -67.09,
        "overbought": 0,
        "oversold": 0,
        "zone": "neutral",
        "source": "real"
      },
      "hurst_exponent": 0.3246,
      "shannon_entropy": 2.9163,
      "kalman_filter": {
        "kalman_price": 64925.85,
        "raw_price": 64872,
        "deviation_pct": -0.08,
        "trend_direction": "DOWN"
      },
      "regression_channel": {
        "slope_per_bar": -1.1318,
        "trend_price": 64888.64,
        "upper_1sd": 64905.32,
        "lower_1sd": 64871.95,
        "upper_2sd": 64922.01,
        "lower_2sd": 64855.26,
        "deviation_from_trend": -16.64,
        "position_in_channel": 0.2508
      },
      "dominant_cycles": {
        "dominant_cycles": [
          100,
          40,
          33.3
        ],
        "cycle_strengths": [
          3118.06,
          1966.69,
          1010.51
        ]
      },
      "fractal_dimension": 0.6199,
      "monte_carlo": {
        "median_price": 64849.52,
        "p10": 64792.93,
        "p25": 64821.18,
        "p75": 64882.56,
        "p90": 64912.74,
        "prob_up": 0.43,
        "horizon_bars": 12
      },
      "fair_value_gaps": [
        {
          "type": "BEARISH",
          "top": 64925,
          "bottom": 64892.3,
          "gap_size": 32.7,
          "gap_pct": 0.05
        }
      ],
      "market_structure": {
        "structure": "BEARISH",
        "bos_detected": 0,
        "last_swing_high": 64925,
        "last_swing_low": 64909.55,
        "higher_highs": 0,
        "higher_lows": 0
      }
    },
    "profile_analysis": {
      "poor_extremes": {
        "poor_high": {
          "detected": 1,
          "price": 64872,
          "volume_ratio": 3.961,
          "implication": "High likely to be revisited - unfinished auction"
        },
        "poor_low": {
          "detected": 1,
          "price": 64846,
          "volume_ratio": 0.799,
          "implication": "Low likely to be revisited - unfinished auction"
        },
        "excess_high": 0,
        "excess_low": 0,
        "session_high": 64872,
        "session_low": 64846,
        "action_bias": "expect_retest_both",
        "status": "success"
      },
      "profile_shape": {
        "shape": "P",
        "implication": "Short covering rally - bearish bias expected",
        "trading_signal": "BEARISH_AFTER",
        "distribution": {
          "lower_third_pct": 30.3,
          "middle_third_pct": 8.3,
          "upper_third_pct": 61.4
        },
        "dominant_zone": "upper",
        "status": "success"
      },
      "no_mans_land": {
        "zones": [
          {
            "range_low": 64512,
            "range_high": 64810,
            "gap_size": 298,
            "gap_size_pct": 0.462,
            "risk": "LOW",
            "lvn_confirmed": 1,
            "lvns_in_zone": 9,
            "nearest_hvn_below": 64512,
            "nearest_hvn_above": 64810,
            "distance_from_price": 46.34,
            "distance_pct": 0.07,
            "direction": "below"
          }
        ],
        "price_in_no_mans_land": 0,
        "nearest_no_mans_land": {
          "range_low": 64512,
          "range_high": 64810,
          "gap_size": 298,
          "gap_size_pct": 0.462,
          "risk": "LOW",
          "lvn_confirmed": 1,
          "lvns_in_zone": 9,
          "nearest_hvn_below": 64512,
          "nearest_hvn_above": 64810,
          "distance_from_price": 46.34,
          "distance_pct": 0.07,
          "direction": "below"
        },
        "total_zones": 1,
        "max_gap_pct": 0.46,
        "status": "success"
      },
      "va_volume_pct": {
        "value_area_volume_pct": 100,
        "interpretation": "extremely_compressed",
        "breakout_risk": "VERY_HIGH",
        "volume_in_va": 1,
        "total_volume": 1,
        "compression_signal": 1
      },
      "volume_node_strength": {
        "scored_hvns": [
          {
            "price": 64304,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64306,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.8,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64314,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.8,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64320,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.9,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64322,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.9,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          }
        ],
        "scored_lvns": [
          {
            "price": 64404,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.5,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64423,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64436,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.8,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64440,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.8,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64447,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.8,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          }
        ],
        "total_hvns": 52,
        "total_lvns": 101,
        "avg_hvn_strength": 66.2,
        "avg_lvn_strength": 63.1,
        "status": "success"
      }
    },
    "flow_analysis": {
      "passive_aggressive": {
        "aggressive": {
          "buy_pct": 24.72,
          "sell_pct": 75.28,
          "net_pct": -50.56,
          "dominance": "sellers",
          "buy_volume": 2.746,
          "sell_volume": 8.359
        },
        "passive": {
          "dominance": "buyers",
          "inference": "from_orderbook_depth",
          "bid_depth": 4099382.22,
          "ask_depth": 571474.73,
          "bid_ratio": 0.88,
          "ob_imbalance": 0.755
        },
        "composite": {
          "agreement": 0,
          "signal": "sell_absorption",
          "interpretation": "Aggressive sellers hitting passive buy walls - potential reversal or breakdown",
          "conviction": "MEDIUM"
        },
        "status": "success"
      },
      "whale_accumulation": {
        "score": -21,
        "classification": "MILD_DISTRIBUTION",
        "bias": "DISTRIBUTING",
        "components": {
          "flow": {
            "score": -30,
            "max": 30,
            "detail": {
              "divergence": "aligned",
              "whale_delta": -5,
              "mid_delta": -3.005,
              "retail_delta": -20.565,
              "primary_delta": -5
            }
          },
          "depth": {
            "score": 15.11,
            "max": 20,
            "detail": {
              "bid_depth": 4099382.22,
              "ask_depth": 571474.73,
              "ratio": 0.76,
              "deep_confirmation": 0
            }
          },
          "absorption": {
            "score": -7.8,
            "max": 25,
            "detail": {
              "buyer_strength": 2.5,
              "seller_exhaustion": 5.1,
              "net_absorption": -2.6,
              "index": 0.1129,
              "label": "Neutra"
            }
          },
          "derivatives": {
            "score": 1.49,
            "max": 25,
            "detail": {
              "long_short_ratio": 1.1,
              "lsr_score": 1.49
            }
          }
        },
        "trend": {
          "direction": "stable",
          "avg_score": -25.5,
          "recent_avg": -29.8,
          "momentum": 4.5,
          "samples": 12,
          "score_range": {
            "min": -49,
            "max": 14
          }
        },
        "status": "success"
      },
      "absorption_zones": {
        "total_zones": 0,
        "total_events": 0,
        "buy_zone_count": 0,
        "sell_zone_count": 0,
        "status": "no_events"
      }
    },
    "sr_analysis": {
      "defense_zones": {
        "buy_defense": [
          {
            "center": 64479.84,
            "range_low": 64376.36,
            "range_high": 64570.64,
            "strength": 62,
            "side": "buy",
            "sources": [
              "sr_level_ema_21_4h",
              "vp_hvn",
              "ema_ema_21_4h",
              "vp_val"
            ],
            "source_count": 4,
            "signals_in_zone": 5,
            "type": "confluence",
            "distance_from_price": 376.5,
            "distance_pct": 0.58
          },
          {
            "center": 64831.72,
            "range_low": 64740.16,
            "range_high": 64928.64,
            "strength": 61,
            "side": "buy",
            "sources": [
              "orderbook_bid_wall",
              "vp_hvn",
              "ema_ema_21_1h",
              "sr_level_ema_21_1h"
            ],
            "source_count": 4,
            "signals_in_zone": 8,
            "type": "confluence",
            "distance_from_price": 24.62,
            "distance_pct": 0.04
          },
          {
            "center": 64162.07,
            "range_low": 64096.13,
            "range_high": 64228.02,
            "strength": 46,
            "side": "buy",
            "sources": [
              "ema_ema_21_1d",
              "sr_level_vah_monthly"
            ],
            "source_count": 2,
            "signals_in_zone": 2,
            "type": "cluster",
            "distance_from_price": 694.27,
            "distance_pct": 1.07
          },
          {
            "center": 64336.25,
            "range_low": 64255.36,
            "range_high": 64442.64,
            "strength": 40,
            "side": "buy",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 9,
            "type": "cluster",
            "distance_from_price": 520.09,
            "distance_pct": 0.8
          }
        ],
        "sell_defense": [
          {
            "center": 65037.38,
            "range_low": 64960.36,
            "range_high": 65128.64,
            "strength": 55,
            "side": "sell",
            "sources": [
              "sr_level_poc_daily",
              "vp_poc",
              "vp_hvn"
            ],
            "source_count": 3,
            "signals_in_zone": 9,
            "type": "confluence",
            "distance_from_price": 181.04,
            "distance_pct": 0.28
          },
          {
            "center": 65346,
            "range_low": 65297.36,
            "range_high": 65394.64,
            "strength": 53,
            "side": "sell",
            "sources": [
              "sr_level_vah_daily",
              "vp_hvn",
              "vp_vah"
            ],
            "source_count": 3,
            "signals_in_zone": 3,
            "type": "confluence",
            "distance_from_price": 489.66,
            "distance_pct": 0.76
          },
          {
            "center": 64944.5,
            "range_low": 64860.36,
            "range_high": 65052.64,
            "strength": 49,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "ema_ema_21_15m",
              "sr_level_hvn_daily"
            ],
            "source_count": 3,
            "signals_in_zone": 9,
            "type": "confluence",
            "distance_from_price": 88.16,
            "distance_pct": 0.14
          },
          {
            "center": 65157.17,
            "range_low": 65064.36,
            "range_high": 65257.64,
            "strength": 41,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 13,
            "type": "cluster",
            "distance_from_price": 300.83,
            "distance_pct": 0.46
          },
          {
            "center": 65252.3,
            "range_low": 65164.36,
            "range_high": 65350.64,
            "strength": 40,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 11,
            "type": "cluster",
            "distance_from_price": 395.96,
            "distance_pct": 0.61
          }
        ],
        "total_zones": 9,
        "strongest_buy": {
          "center": 64479.84,
          "range_low": 64376.36,
          "range_high": 64570.64,
          "strength": 62,
          "side": "buy",
          "sources": [
            "sr_level_ema_21_4h",
            "vp_hvn",
            "ema_ema_21_4h",
            "vp_val"
          ],
          "source_count": 4,
          "signals_in_zone": 5,
          "type": "confluence",
          "distance_from_price": 376.5,
          "distance_pct": 0.58
        },
        "strongest_sell": {
          "center": 65037.38,
          "range_low": 64960.36,
          "range_high": 65128.64,
          "strength": 55,
          "side": "sell",
          "sources": [
            "sr_level_poc_daily",
            "vp_poc",
            "vp_hvn"
          ],
          "source_count": 3,
          "signals_in_zone": 9,
          "type": "confluence",
          "distance_from_price": 181.04,
          "distance_pct": 0.28
        },
        "defense_asymmetry": {
          "ratio": 0.88,
          "bias": "slight_sell_defense",
          "description": "Slightly more sell defense",
          "buy_total_strength": 209,
          "sell_total_strength": 238
        },
        "status": "success"
      }
    },
    "quality": {
      "calendar": {
        "day_of_week": "Friday",
        "day_of_week_num": 4,
        "is_us_holiday": 0,
        "is_weekend": 0,
        "is_pre_holiday": 0,
        "is_post_holiday": 0,
        "expected_liquidity": "NORMAL",
        "liquidity_warning": 0
      },
      "latency": {
        "latency_ms": 8225,
        "latency_category": "POOR",
        "data_freshness": "DELAYED",
        "is_acceptable": 0,
        "is_stale": 0
      },
      "spread_percentile": {
        "status": "ok",
        "current_spread": 0.1,
        "spread_percentile": 0,
        "spread_mean": 0.1,
        "spread_median": 0.1,
        "spread_std": 0,
        "spread_min": 0.1,
        "spread_max": 0.1,
        "spread_z_score": -0.9574,
        "is_tight": 1,
        "is_wide": 0,
        "is_anomalous": 0,
        "liquidity_signal": "EXCELLENT",
        "samples": 12,
        "window_minutes": 1440
      },
      "anomalies": {
        "anomalies_detected": 1,
        "count": 2,
        "anomalies": [
          {
            "type": "FLOW_EXTREME_IMBALANCE",
            "severity": "MEDIUM",
            "value": -0.5055,
            "direction": "SELL",
            "description": "Extreme flow imbalance: -50.55% toward sellers"
          },
          {
            "type": "DEPTH_EXTREME_ASYMMETRY",
            "severity": "MEDIUM",
            "ratio": 7.17,
            "bid_depth": 4099382.22,
            "ask_depth": 571474.73,
            "direction": "BID_HEAVY",
            "description": "Order book depth ratio 7.17:1 is extreme"
          }
        ],
        "max_severity": "MEDIUM",
        "risk_elevated": 0,
        "types_found": [
          "DEPTH_EXTREME_ASYMMETRY",
          "FLOW_EXTREME_IMBALANCE"
        ],
        "summary": "2 anomalies detected (max severity: MEDIUM)"
      }
    },
    "candlestick_patterns": {
      "patterns_detected": 1,
      "patterns": [
        {
          "name": "doji",
          "type": "neutral",
          "confidence": 0.65,
          "candles_used": 1,
          "implication": "Indecision - watch next candle for direction"
        }
      ],
      "dominant_signal": "neutral",
      "max_confidence": 0.65,
      "bullish_count": 0,
      "bearish_count": 0,
      "neutral_count": 1
    }
  },
  "sequence_id": 12,
  "primary_exchange": "BINANCE",
  "data_feed_type": "WEBSOCKET_L2",
  "data_quality_score": 9.5,
  "completeness_pct": 100,
  "reliability_score": 9,
  "bid": 64827.8,
  "ask": 64827.9,
  "tick_direction": -1,
  "twap": 64881.49,
  "pivot_points": {
    "daily": {
      "pivot": 64971.33,
      "r1": 65420.66,
      "r2": 65795.33,
      "r3": 66619.33,
      "s1": 64596.66,
      "s2": 64147.33,
      "s3": 63323.33,
      "vah": 65346,
      "val": 64522,
      "poc": 65046
    },
    "weekly": {
      "pivot": 62830.33,
      "r1": 63346.66,
      "r2": 64083.33,
      "r3": 65336.33,
      "s1": 62093.66,
      "s2": 61577.33,
      "s3": 60324.33,
      "vah": 63567,
      "val": 62314,
      "poc": 62610
    },
    "monthly": {
      "pivot": 63185.67,
      "r1": 64582.34,
      "r2": 65610.67,
      "r3": 68035.67,
      "s1": 62157.34,
      "s2": 60760.67,
      "s3": 58335.67,
      "vah": 64214,
      "val": 61789,
      "poc": 63554
    }
  },
  "immediate_support": [
    64831.72,
    64522,
    64479.84,
    64214,
    64162.07
  ],
  "support_strength": [
    61,
    94.8,
    62,
    72.1,
    46
  ],
  "immediate_resistance": [
    64944.5,
    64971.33,
    65037.38,
    65346
  ],
  "resistance_strength": [
    49,
    98.2,
    55,
    92.5
  ],
  "volatility_metrics": {
    "realized_vol_24h": 0.0184,
    "realized_vol_7d": 0.0487,
    "volatility_regime": "NORMAL"
  },
  "order_flow_extended": {
    "passive_buy_pct": 87.8,
    "passive_sell_pct": 12.2
  },
  "whale_activity": {
    "large_orders_1h": [
      {
        "size": 1,
        "price": 64896.34,
        "side": "SELL",
        "timestamp_ms": 1786141945725
      },
      {
        "size": 1,
        "price": 64883.14,
        "side": "SELL",
        "timestamp_ms": 1786142044496
      },
      {
        "size": 1,
        "price": 64870.63,
        "side": "SELL",
        "timestamp_ms": 1786142264045
      },
      {
        "size": 1,
        "price": 64853.03,
        "side": "SELL",
        "timestamp_ms": 1786142272654
      },
      {
        "size": 1,
        "price": 64867.43,
        "side": "SELL",
        "timestamp_ms": 1786142499673
      }
    ],
    "iceberg_activity": 1,
    "hidden_orders_detected": 0
  },
  "technical_indicators_extended": {
    "cci_1h": 97.67,
    "cci_signal": "NEUTRAL",
    "stochastic": {
      "k": 58.57,
      "d": 39.05,
      "signal": "NEUTRAL",
      "source": "real"
    },
    "williams_r": {
      "value": -67.09,
      "overbought": 0,
      "oversold": 0,
      "zone": "neutral",
      "source": "real"
    },
    "hurst_exponent": 0.3246,
    "shannon_entropy": 2.9163,
    "fractal_dimension": 0.6199,
    "kalman_filter": {
      "kalman_price": 64925.85,
      "raw_price": 64872,
      "deviation_pct": -0.08,
      "trend_direction": "DOWN"
    },
    "regression_channel": {
      "slope_per_bar": -1.1318,
      "trend_price": 64888.64,
      "upper_1sd": 64905.32,
      "lower_1sd": 64871.95,
      "upper_2sd": 64922.01,
      "lower_2sd": 64855.26,
      "deviation_from_trend": -16.64,
      "position_in_channel": 0.2508
    },
    "dominant_cycles": {
      "dominant_cycles": [
        100,
        40,
        33.3
      ],
      "cycle_strengths": [
        3118.06,
        1966.69,
        1010.51
      ]
    },
    "monte_carlo": {
      "median_price": 64849.52,
      "p10": 64792.93,
      "p25": 64821.18,
      "p75": 64882.56,
      "p90": 64912.74,
      "prob_up": 0.43,
      "horizon_bars": 12
    },
    "garch_forecast_1h": 0.0017
  },
  "backup_exchanges": [
    "COINBASE",
    "KRAKEN",
    "OKX"
  ],
  "alerts": {
    "active_alerts": [
      {
        "type": "DEPTH_DIVERGENCE",
        "level": 0.755,
        "severity": "MEDIUM",
        "probability": 0.76,
        "action": "PREPARE_LONG",
        "description": "Orderbook BID_HEAVY: imbalance=0.755"
      }
    ],
    "alert_count": 1,
    "max_severity": "MEDIUM"
  },
  "regime_analysis": {
    "current_regime": "MEAN_REVERTING",
    "regime_probabilities": {
      "trending": 0,
      "mean_reverting": 0.714,
      "breakout": 0.286
    },
    "regime_change_probability": 0.29,
    "expected_regime_duration": "15m-1h",
    "avg_adx": 25.9
  },
  "data_reliability": {
    "has_options_data": 0,
    "onchain_coverage": "partial",
    "latency_acceptable": 1,
    "price_targets_available": 0
  },
  "_log_id": "1b84c3d82fa62cbac78d6789338a640d8c527cf2f0432ca5f569cad2af3a88b8"
}
----------------------------------------------------------------------------------------------------
EVENTO: ANALYSIS_TRIGGER | SYMBOL: BTCUSDT | JANELA: 12
UTC: 2026-08-07 22:41:59 UTC
NY:  2026-08-07 18:41:59 EST/EDT
SP:  2026-08-07 19:41:59 BRT
----------------------------------------------------------------------------------------------------
{
  "is_signal": 1,
  "tipo_evento": "ANALYSIS_TRIGGER",
  "descricao": "Evento automático para análise da IA",
  "symbol": "BTCUSDT",
  "raw_event": {
    "volume_total": 11.105,
    "volume_compra": 2.746,
    "volume_venda": 8.359,
    "delta": -5.613,
    "preco_fechamento": 64856.3,
    "advanced_analysis": {
      "symbol": "BTCUSDT",
      "price": 64856.3,
      "volume": 11.105,
      "timestamp": 1786142519917,
      "adaptive_thresholds": {
        "current_volatility": 0.0184,
        "volatility_factor": 2.0,
        "absorption_threshold": 0.3,
        "flow_threshold": 0.2
      },
      "onchain_metrics": {
        "difficulty": 126.23,
        "active_addresses": 717974,
        "mempool_size": 84905,
        "mempool_vsize_mb": 43.24,
        "mempool_total_fee_btc": 0.1258,
        "fees_fastest_sat_vb": 4,
        "fees_half_hour_sat_vb": 4,
        "fees_hour_sat_vb": 3,
        "fees_economy_sat_vb": 2,
        "difficulty_adjustment": {
          "progress_pct": 93.45,
          "estimated_change_pct": 0.75,
          "remaining_blocks": 132,
          "remaining_time_ms": 78653520,
          "previous_retarget_pct": -0.74
        },
        "minutes_between_blocks": 8.4423,
        "total_btc_sent_24h": 807619.08,
        "trade_volume_btc_24h": 3265.64,
        "data_source": "blockchain.info+mempool.space",
        "is_real_data": 1,
        "requires_paid_api": [
          "exchange_netflow",
          "whale_transactions",
          "exchange_reserves",
          "sopr"
        ]
      }
    },
    "orderbook_data": {
      "mid": 64827.85,
      "spread": 0.1,
      "spread_percent": 0,
      "bid_depth_usd": 4099382.22,
      "ask_depth_usd": 571474.73,
      "imbalance": 0.755,
      "flow_imbalance": 0.7553,
      "volume_ratio": 7.173,
      "pressure": 0.7553,
      "consolidated_bias_score": 0.9266
    },
    "timestamp_utc": 1786142519917
  },
  "resultado_da_batalha": "N/A",
  "delta": -5.613,
  "volume_total": 11.105,
  "volume_compra": 2.746,
  "volume_venda": 8.359,
  "preco_fechamento": 64856.3,
  "epoch_ms": 1786142519917,
  "price_targets": [
    {
      "level": 62471.85,
      "confidence": 0.978,
      "source": "confluence_poc_weekly_val_weekly",
      "weight": 0.356,
      "timestamp": "2026-08-07T22:42:07.833070"
    },
    {
      "level": 63560.88,
      "confidence": 0.99,
      "source": "confluence_poc_monthly_vah_weekly",
      "weight": 0.319,
      "timestamp": "2026-08-07T22:42:07.833070"
    },
    {
      "level": 65176.92,
      "confidence": 0.735,
      "source": "confluence_atr_1h_r1_atr_1h_r2",
      "weight": 0.343,
      "timestamp": "2026-08-07T22:42:07.833070"
    },
    {
      "level": 64437.05,
      "confidence": 0.72,
      "source": "confluence_atr_1h_s1_atr_1h_s2",
      "weight": 0.279,
      "timestamp": "2026-08-07T22:42:07.833070"
    },
    {
      "level": 61789,
      "confidence": 0.764,
      "source": "val_monthly",
      "weight": 0.18,
      "timestamp": "2026-08-07T22:42:07.833070"
    }
  ],
  "multi_tf": {
    "15m": {
      "tendencia": "Baixa",
      "preco_atual": 64871.99,
      "mme_21": 64925.41,
      "atr": 73.63,
      "regime": "Range",
      "rsi_short": 37.88,
      "rsi_long": 44.06,
      "macd": 12.6535,
      "macd_signal": 16.6562,
      "adx": 23.83,
      "realized_vol": 0.0014
    },
    "1h": {
      "tendencia": "Alta",
      "preco_atual": 64871.99,
      "mme_21": 64788.8,
      "atr": 227.14,
      "regime": "Range",
      "rsi_short": 52.6,
      "rsi_long": 54.35,
      "macd": 100.3548,
      "macd_signal": 98.3005,
      "adx": 23.82,
      "realized_vol": 0.0023
    },
    "4h": {
      "tendencia": "Alta",
      "preco_atual": 64872,
      "mme_21": 64460.35,
      "atr": 498.88,
      "regime": "Range",
      "rsi_short": 61.94,
      "rsi_long": 61.09,
      "macd": 274.9477,
      "macd_signal": 256.7109,
      "adx": 30.01,
      "realized_vol": 0.0058
    },
    "1d": {
      "tendencia": "Alta",
      "preco_atual": 64872,
      "mme_21": 64144.77,
      "atr": 1369.69,
      "regime": "Manipulação",
      "rsi_short": 58.86,
      "rsi_long": 55.63,
      "macd": 67.1517,
      "macd_signal": 35.9228,
      "adx": 15.05,
      "realized_vol": 0.0184
    }
  },
  "historical_vp": {
    "daily": {
      "poc": 65046,
      "vah": 65346,
      "val": 64522,
      "status": "success"
    },
    "weekly": {
      "poc": 62610,
      "vah": 63567,
      "val": 62314,
      "status": "success"
    },
    "monthly": {
      "poc": 63554,
      "vah": 64214,
      "val": 61789,
      "status": "success"
    }
  },
  "pattern_recognition": {
    "smart_money": {
      "fair_value_gaps": [
        {
          "type": "BEARISH",
          "top": 64925,
          "bottom": 64892.3,
          "gap_size": 32.7,
          "gap_pct": 0.05
        }
      ],
      "market_structure": {
        "structure": "BEARISH",
        "bos_detected": 0,
        "last_swing_high": 64925,
        "last_swing_low": 64909.55,
        "higher_highs": 0,
        "higher_lows": 0
      }
    }
  },
  "janela_numero": 12,
  "timestamp_utc": "2026-08-07T22:42:00.000Z",
  "timestamp": "2026-08-07 18:42:00-04:00",
  "derivatives": {
    "BTCUSDT": {
      "funding_rate_percent": 0.01,
      "open_interest": 106945.765,
      "open_interest_usd": 6934687958.97,
      "long_short_ratio": 1.1,
      "longs_usd": 3631669164.51,
      "shorts_usd": 3303018794.46
    },
    "ETHUSDT": {
      "funding_rate_percent": 0,
      "open_interest": 2279164.179,
      "open_interest_usd": 4360633657.11,
      "long_short_ratio": 2.07,
      "longs_usd": 2939074192.69,
      "shorts_usd": 1421559464.42
    }
  },
  "fluxo_continuo": {
    "cvd": -28.5699,
    "whale_buy_volume": 0,
    "whale_sell_volume": 5,
    "whale_delta": -5,
    "bursts": {
      "count": 8,
      "max_burst_volume": 7.558
    },
    "sector_flow": {
      "retail": {
        "buy": 12.0048,
        "sell": 32.5694,
        "delta": -20.565
      },
      "mid": {
        "buy": 1.522,
        "sell": 4.5274,
        "delta": -3.005
      },
      "whale": {
        "buy": 0,
        "sell": 5,
        "delta": -5
      }
    },
    "timestamp": "2026-08-07T22:42:00.000+00:00",
    "time_index": {
      "epoch_ms": 1786142520000,
      "timestamp_utc": "2026-08-07T22:42:00.000+00:00",
      "timestamp_ny": "2026-08-07T18:42:00.000-04:00",
      "timestamp_sp": "2026-08-07T19:42:00.000-03:00"
    },
    "order_flow": {
      "net_flow_1m": 160810.2968,
      "absorcao_1m": "Neutra",
      "buy_volume": 178096.26,
      "sell_volume": 542212.72,
      "total_volume": 720308.98,
      "buy_volume_btc": 2.746,
      "sell_volume_btc": 8.359,
      "total_volume_btc": 11.105,
      "whale_buy_volume_window": 0,
      "whale_sell_volume_window": 1,
      "whale_delta_window": -1,
      "flow_imbalance": -0.5055,
      "aggressive_buy_pct": 24.72,
      "aggressive_sell_pct": 75.28,
      "net_flow_5m": -196915.5833,
      "absorcao_5m": "Neutra",
      "net_flow_15m": -1854091.4037,
      "absorcao_15m": "Neutra",
      "computation_window_min": 1,
      "available_windows_min": [
        1,
        5,
        15
      ],
      "buy_sell_ratio": {
        "buy_sell_ratio": 0.33,
        "ratios": {
          "current": 0.3285,
          "imbalance_1m": 0.223,
          "imbalance_5m": -0.273,
          "imbalance_15m": -2.574
        },
        "sector_ratios": {
          "retail": 0.3686,
          "mid": 0.3362,
          "whale": 0
        },
        "pressure": "STRONG_SELL",
        "flow_trend": "short_term_reversal_to_buy",
        "buy_volume": 2.746,
        "sell_volume": 8.359
      }
    },
    "tipo_absorcao": "Neutra",
    "participant_analysis": {
      "retail": {
        "volume_pct": 85.9,
        "direction": "SELL",
        "sentiment": "BEARISH",
        "composite_score": -0.713,
        "imbalance": -0.424
      },
      "mid": {
        "volume_pct": 5.09,
        "direction": "SELL",
        "sentiment": "BEARISH",
        "composite_score": -0.421,
        "imbalance": -1
      },
      "whale": {
        "volume_pct": 9,
        "direction": "SELL",
        "sentiment": "BEARISH",
        "composite_score": -0.436,
        "imbalance": -1
      }
    },
    "liquidity_heatmap": {
      "clusters": [
        {
          "center": 64862.4746,
          "low": 64846,
          "high": 64872,
          "width": 26,
          "total_volume": 13.606,
          "buy_volume": 3.607,
          "sell_volume": 9.999,
          "imbalance": -6.391,
          "imbalance_ratio": -0.47,
          "trades_count": 2000,
          "avg_trade_size": 0.007,
          "recent_timestamp": 1786142526893,
          "recent_ts_ms": 1786142526893,
          "last_seen_ms": 1786142526893,
          "first_seen_ms": 1786142315761,
          "age_ms": 265.0,
          "cluster_duration_ms": 211132,
          "price_std": 10.0286,
          "volume_std": 0.037,
          "bin_threshold_usd": 194.5874
        }
      ],
      "resistances": [
        64862.4746
      ],
      "clusters_count": 1
    },
    "absorption_analysis": {
      "current_absorption": {
        "index": 0.1129,
        "classification": "WEAK_ABSORPTION",
        "label": "Neutra",
        "buyer_strength": 2.5,
        "seller_exhaustion": 5.1,
        "continuation_probability": 0.1,
        "delta_usd": 160810.297,
        "total_volume_usd": 720308.98,
        "flow_imbalance": -0.5055,
        "window_min": 1
      }
    }
  },
  "market_context": {
    "trading_session": "NY",
    "session_phase": "ACTIVE",
    "time_to_session_close": 11964,
    "day_of_week": 4,
    "is_holiday": 0,
    "market_hours_type": "EXTENDED"
  },
  "market_environment": {
    "volatility_regime": "NORMAL",
    "trend_direction": "UP",
    "market_structure": "RANGE_BOUND",
    "liquidity_environment": "NORMAL",
    "risk_sentiment": "BULLISH",
    "correlation_spy": 0.4056,
    "correlation_dxy": -0.0854,
    "correlation_gold": 0.2379
  },
  "ml_features": {
    "price_features": {
      "returns_1": 0.0,
      "volatility_1": 0.0,
      "returns_5": 1.5e-07,
      "volatility_5": 1.2e-07,
      "returns_15": 0.0,
      "volatility_15": 1e-07,
      "momentum_score": -0.70710678,
      "volatility_1h": 0.0023
    },
    "volume_features": {
      "volume_sma_ratio": 0.739,
      "volume_momentum": -0.511,
      "buy_sell_pressure": -0.5054,
      "liquidity_gradient": -0.51112327
    },
    "microstructure": {
      "order_book_slope": -4.385968,
      "flow_imbalance": -0.5055,
      "tick_rule_sum": -94,
      "trade_intensity": 40,
      "trade_intensity_v2": 26.2333
    },
    "cross_asset": {
      "btc_eth_corr_7d": 0.8212,
      "btc_eth_corr_30d": 0.8599,
      "btc_dxy_corr_30d": -0.0605,
      "btc_dxy_corr_90d": -0.067,
      "btc_ndx_corr_30d": 0.4147,
      "dxy_return_5d": -0.3561,
      "dxy_return_20d": -1.6548,
      "btc_dxy_correlation_stability": 0.0065,
      "btc_dxy_inverse_strength": 0.0638,
      "dxy_momentum": -1.29867555,
      "vix_current": 14.9,
      "us10y_yield": 4.66,
      "btc_dominance": 17.0377,
      "btc_dominance_change_7d": 0,
      "eth_dominance": 9.0966,
      "gold_price": 4342.3515,
      "oil_price": 77.08,
      "macro_regime": "RISK_ON",
      "correlation_regime": "DECORRELATED"
    }
  },
  "contextual_snapshot": {
    "symbol": "BTCUSDT",
    "ohlc": {
      "open": 64872,
      "high": 64872,
      "low": 64846,
      "close": 64856.3,
      "open_time": 1786142460293,
      "close_time": 1786142519917,
      "vwap": 64863.1
    },
    "volume_total": 11.105,
    "volume_total_usdt": 720309,
    "volume_compra": 2.746,
    "volume_venda": 8.359,
    "num_trades": 1561,
    "delta_minimo": -8.094,
    "delta_maximo": 0.025,
    "delta_fechamento": -5.613,
    "reversao_desde_minimo": 2.48,
    "reversao_desde_maximo": 5.64,
    "poc_price": 64871.4,
    "poc_volume": 3.54,
    "poc_percentage": 31.9,
    "dwell_price": 64871.4,
    "dwell_seconds": 39,
    "dwell_location": "High",
    "trades_per_second": 26.18,
    "avg_trade_size": 0.007
  },
  "orderbook_data": {
    "mid": 64827.85,
    "spread": 0.1,
    "spread_percent": 0,
    "bid_depth_usd": 4099382.22,
    "ask_depth_usd": 571474.73,
    "imbalance": 0.755,
    "flow_imbalance": 0.7553,
    "volume_ratio": 7.173,
    "pressure": 0.7553,
    "consolidated_bias_score": 0.9266,
    "spread_bps": 0.0154,
    "is_valid": 1,
    "data_source": "live",
    "spread_volatility": 0.0
  },
  "order_book_depth": {
    "L1": {
      "bids": 2417104.52,
      "asks": 285696.56,
      "flow_imbalance": 0.7886
    },
    "L5": {
      "bids": 2501445.35,
      "asks": 289132.44,
      "flow_imbalance": 0.7928
    },
    "L10": {
      "bids": 2779876.48,
      "asks": 294967.03,
      "flow_imbalance": 0.8081
    },
    "L25": {
      "bids": 3287850.83,
      "asks": 299116.14,
      "flow_imbalance": 0.8332
    },
    "total_depth_ratio": 10.99
  },
  "orderbook_quality": "live",
  "market_impact": {
    "slippage_matrix": {
      "100k_usd": {
        "buy": 0.05,
        "sell": 0.05
      },
      "1m_usd": {
        "buy": 5.65,
        "sell": 0.05
      }
    },
    "liquidity_score": 9.9985,
    "execution_quality": "EXCELLENT"
  },
  "institutional_analytics": {
    "status": "ok",
    "computed_at_ms": 1786142528811,
    "technical_extras": {
      "stoch_rsi": {
        "k": 58.57,
        "d": 39.05,
        "overbought": 0,
        "oversold": 0,
        "crossover": "none"
      },
      "williams_r": {
        "value": -67.09,
        "overbought": 0,
        "oversold": 0,
        "zone": "neutral",
        "source": "real"
      },
      "hurst_exponent": 0.3246,
      "shannon_entropy": 2.9163,
      "kalman_filter": {
        "kalman_price": 64925.85,
        "raw_price": 64872,
        "deviation_pct": -0.08,
        "trend_direction": "DOWN"
      },
      "regression_channel": {
        "slope_per_bar": -1.1318,
        "trend_price": 64888.64,
        "upper_1sd": 64905.32,
        "lower_1sd": 64871.95,
        "upper_2sd": 64922.01,
        "lower_2sd": 64855.26,
        "deviation_from_trend": -16.64,
        "position_in_channel": 0.2508
      },
      "dominant_cycles": {
        "dominant_cycles": [
          100,
          40,
          33.3
        ],
        "cycle_strengths": [
          3118.06,
          1966.69,
          1010.51
        ]
      },
      "fractal_dimension": 0.6199,
      "monte_carlo": {
        "median_price": 64849.48,
        "p10": 64792.89,
        "p25": 64821.14,
        "p75": 64882.52,
        "p90": 64912.7,
        "prob_up": 0.43,
        "horizon_bars": 12
      },
      "fair_value_gaps": [
        {
          "type": "BEARISH",
          "top": 64925,
          "bottom": 64892.3,
          "gap_size": 32.7,
          "gap_pct": 0.05
        }
      ],
      "market_structure": {
        "structure": "BEARISH",
        "bos_detected": 0,
        "last_swing_high": 64925,
        "last_swing_low": 64909.55,
        "higher_highs": 0,
        "higher_lows": 0
      }
    },
    "profile_analysis": {
      "poor_extremes": {
        "poor_high": {
          "detected": 1,
          "price": 64872,
          "volume_ratio": 3.961,
          "implication": "High likely to be revisited - unfinished auction"
        },
        "poor_low": {
          "detected": 1,
          "price": 64846,
          "volume_ratio": 0.799,
          "implication": "Low likely to be revisited - unfinished auction"
        },
        "excess_high": 0,
        "excess_low": 0,
        "session_high": 64872,
        "session_low": 64846,
        "action_bias": "expect_retest_both",
        "status": "success"
      },
      "profile_shape": {
        "shape": "P",
        "implication": "Short covering rally - bearish bias expected",
        "trading_signal": "BEARISH_AFTER",
        "distribution": {
          "lower_third_pct": 30.3,
          "middle_third_pct": 8.3,
          "upper_third_pct": 61.4
        },
        "dominant_zone": "upper",
        "status": "success"
      },
      "no_mans_land": {
        "zones": [
          {
            "range_low": 64512,
            "range_high": 64810,
            "gap_size": 298,
            "gap_size_pct": 0.462,
            "risk": "LOW",
            "lvn_confirmed": 1,
            "lvns_in_zone": 9,
            "nearest_hvn_below": 64512,
            "nearest_hvn_above": 64810,
            "distance_from_price": 46.3,
            "distance_pct": 0.07,
            "direction": "below"
          }
        ],
        "price_in_no_mans_land": 0,
        "nearest_no_mans_land": {
          "range_low": 64512,
          "range_high": 64810,
          "gap_size": 298,
          "gap_size_pct": 0.462,
          "risk": "LOW",
          "lvn_confirmed": 1,
          "lvns_in_zone": 9,
          "nearest_hvn_below": 64512,
          "nearest_hvn_above": 64810,
          "distance_from_price": 46.3,
          "distance_pct": 0.07,
          "direction": "below"
        },
        "total_zones": 1,
        "max_gap_pct": 0.46,
        "status": "success"
      },
      "va_volume_pct": {
        "value_area_volume_pct": 100,
        "interpretation": "extremely_compressed",
        "breakout_risk": "VERY_HIGH",
        "volume_in_va": 1,
        "total_volume": 1,
        "compression_signal": 1
      },
      "volume_node_strength": {
        "scored_hvns": [
          {
            "price": 64304,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64306,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.8,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64314,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.8,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64320,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.9,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64322,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.9,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          }
        ],
        "scored_lvns": [
          {
            "price": 64404,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.5,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64423,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64436,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.8,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64440,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.8,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64447,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.8,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          }
        ],
        "total_hvns": 52,
        "total_lvns": 101,
        "avg_hvn_strength": 66.2,
        "avg_lvn_strength": 63.1,
        "status": "success"
      }
    },
    "flow_analysis": {
      "passive_aggressive": {
        "aggressive": {
          "buy_pct": 24.72,
          "sell_pct": 75.28,
          "net_pct": -50.56,
          "dominance": "sellers",
          "buy_volume": 2.746,
          "sell_volume": 8.359
        },
        "passive": {
          "dominance": "buyers",
          "inference": "from_orderbook_depth",
          "bid_depth": 4099382.22,
          "ask_depth": 571474.73,
          "bid_ratio": 0.88,
          "ob_imbalance": 0.755
        },
        "composite": {
          "agreement": 0,
          "signal": "sell_absorption",
          "interpretation": "Aggressive sellers hitting passive buy walls - potential reversal or breakdown",
          "conviction": "MEDIUM"
        },
        "status": "success"
      },
      "whale_accumulation": {
        "score": -21,
        "classification": "MILD_DISTRIBUTION",
        "bias": "DISTRIBUTING",
        "components": {
          "flow": {
            "score": -30,
            "max": 30,
            "detail": {
              "divergence": "aligned",
              "whale_delta": -5,
              "mid_delta": -3.005,
              "retail_delta": -20.565,
              "primary_delta": -5
            }
          },
          "depth": {
            "score": 15.11,
            "max": 20,
            "detail": {
              "bid_depth": 4099382.22,
              "ask_depth": 571474.73,
              "ratio": 0.76,
              "deep_confirmation": 0
            }
          },
          "absorption": {
            "score": -7.8,
            "max": 25,
            "detail": {
              "buyer_strength": 2.5,
              "seller_exhaustion": 5.1,
              "net_absorption": -2.6,
              "index": 0.1129,
              "label": "Neutra"
            }
          },
          "derivatives": {
            "score": 1.49,
            "max": 25,
            "detail": {
              "long_short_ratio": 1.1,
              "lsr_score": 1.49
            }
          }
        },
        "trend": {
          "direction": "stable",
          "avg_score": -25.2,
          "recent_avg": -26.6,
          "momentum": 4.2,
          "samples": 13,
          "score_range": {
            "min": -49,
            "max": 14
          }
        },
        "status": "success"
      },
      "absorption_zones": {
        "total_zones": 0,
        "total_events": 0,
        "buy_zone_count": 0,
        "sell_zone_count": 0,
        "status": "no_events"
      }
    },
    "sr_analysis": {
      "defense_zones": {
        "buy_defense": [
          {
            "center": 64479.84,
            "range_low": 64376.36,
            "range_high": 64570.64,
            "strength": 62,
            "side": "buy",
            "sources": [
              "sr_level_ema_21_4h",
              "vp_hvn",
              "ema_ema_21_4h",
              "vp_val"
            ],
            "source_count": 4,
            "signals_in_zone": 5,
            "type": "confluence",
            "distance_from_price": 376.46,
            "distance_pct": 0.58
          },
          {
            "center": 64831.71,
            "range_low": 64740.16,
            "range_high": 64928.64,
            "strength": 61,
            "side": "buy",
            "sources": [
              "orderbook_bid_wall",
              "vp_hvn",
              "ema_ema_21_1h",
              "sr_level_ema_21_1h"
            ],
            "source_count": 4,
            "signals_in_zone": 8,
            "type": "confluence",
            "distance_from_price": 24.59,
            "distance_pct": 0.04
          },
          {
            "center": 64162.07,
            "range_low": 64096.13,
            "range_high": 64228.02,
            "strength": 46,
            "side": "buy",
            "sources": [
              "ema_ema_21_1d",
              "sr_level_vah_monthly"
            ],
            "source_count": 2,
            "signals_in_zone": 2,
            "type": "cluster",
            "distance_from_price": 694.23,
            "distance_pct": 1.07
          },
          {
            "center": 64336.25,
            "range_low": 64255.36,
            "range_high": 64442.64,
            "strength": 40,
            "side": "buy",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 9,
            "type": "cluster",
            "distance_from_price": 520.05,
            "distance_pct": 0.8
          }
        ],
        "sell_defense": [
          {
            "center": 65037.38,
            "range_low": 64960.36,
            "range_high": 65128.64,
            "strength": 55,
            "side": "sell",
            "sources": [
              "sr_level_poc_daily",
              "vp_poc",
              "vp_hvn"
            ],
            "source_count": 3,
            "signals_in_zone": 9,
            "type": "confluence",
            "distance_from_price": 181.08,
            "distance_pct": 0.28
          },
          {
            "center": 65346,
            "range_low": 65297.36,
            "range_high": 65394.64,
            "strength": 53,
            "side": "sell",
            "sources": [
              "sr_level_vah_daily",
              "vp_hvn",
              "vp_vah"
            ],
            "source_count": 3,
            "signals_in_zone": 3,
            "type": "confluence",
            "distance_from_price": 489.7,
            "distance_pct": 0.76
          },
          {
            "center": 64944.5,
            "range_low": 64860.36,
            "range_high": 65052.64,
            "strength": 49,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "ema_ema_21_15m",
              "sr_level_hvn_daily"
            ],
            "source_count": 3,
            "signals_in_zone": 9,
            "type": "confluence",
            "distance_from_price": 88.2,
            "distance_pct": 0.14
          },
          {
            "center": 65157.17,
            "range_low": 65064.36,
            "range_high": 65257.64,
            "strength": 41,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 13,
            "type": "cluster",
            "distance_from_price": 300.87,
            "distance_pct": 0.46
          },
          {
            "center": 65252.3,
            "range_low": 65164.36,
            "range_high": 65350.64,
            "strength": 40,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 11,
            "type": "cluster",
            "distance_from_price": 396,
            "distance_pct": 0.61
          }
        ],
        "total_zones": 9,
        "strongest_buy": {
          "center": 64479.84,
          "range_low": 64376.36,
          "range_high": 64570.64,
          "strength": 62,
          "side": "buy",
          "sources": [
            "sr_level_ema_21_4h",
            "vp_hvn",
            "ema_ema_21_4h",
            "vp_val"
          ],
          "source_count": 4,
          "signals_in_zone": 5,
          "type": "confluence",
          "distance_from_price": 376.46,
          "distance_pct": 0.58
        },
        "strongest_sell": {
          "center": 65037.38,
          "range_low": 64960.36,
          "range_high": 65128.64,
          "strength": 55,
          "side": "sell",
          "sources": [
            "sr_level_poc_daily",
            "vp_poc",
            "vp_hvn"
          ],
          "source_count": 3,
          "signals_in_zone": 9,
          "type": "confluence",
          "distance_from_price": 181.08,
          "distance_pct": 0.28
        },
        "defense_asymmetry": {
          "ratio": 0.88,
          "bias": "slight_sell_defense",
          "description": "Slightly more sell defense",
          "buy_total_strength": 209,
          "sell_total_strength": 238
        },
        "status": "success"
      }
    },
    "quality": {
      "calendar": {
        "day_of_week": "Friday",
        "day_of_week_num": 4,
        "is_us_holiday": 0,
        "is_weekend": 0,
        "is_pre_holiday": 0,
        "is_post_holiday": 0,
        "expected_liquidity": "NORMAL",
        "liquidity_warning": 0
      },
      "latency": {
        "latency_ms": 9023,
        "latency_category": "POOR",
        "data_freshness": "DELAYED",
        "is_acceptable": 0,
        "is_stale": 0
      },
      "spread_percentile": {
        "status": "ok",
        "current_spread": 0.1,
        "spread_percentile": 0,
        "spread_mean": 0.1,
        "spread_median": 0.1,
        "spread_std": 0,
        "spread_min": 0.1,
        "spread_max": 0.1,
        "spread_z_score": 0,
        "is_tight": 1,
        "is_wide": 0,
        "is_anomalous": 0,
        "liquidity_signal": "EXCELLENT",
        "samples": 13,
        "window_minutes": 1440
      },
      "anomalies": {
        "anomalies_detected": 1,
        "count": 2,
        "anomalies": [
          {
            "type": "FLOW_EXTREME_IMBALANCE",
            "severity": "MEDIUM",
            "value": -0.5055,
            "direction": "SELL",
            "description": "Extreme flow imbalance: -50.55% toward sellers"
          },
          {
            "type": "DEPTH_EXTREME_ASYMMETRY",
            "severity": "MEDIUM",
            "ratio": 7.17,
            "bid_depth": 4099382.22,
            "ask_depth": 571474.73,
            "direction": "BID_HEAVY",
            "description": "Order book depth ratio 7.17:1 is extreme"
          }
        ],
        "max_severity": "MEDIUM",
        "risk_elevated": 0,
        "types_found": [
          "DEPTH_EXTREME_ASYMMETRY",
          "FLOW_EXTREME_IMBALANCE"
        ],
        "summary": "2 anomalies detected (max severity: MEDIUM)"
      }
    },
    "candlestick_patterns": {
      "patterns_detected": 1,
      "patterns": [
        {
          "name": "doji",
          "type": "neutral",
          "confidence": 0.65,
          "candles_used": 1,
          "implication": "Indecision - watch next candle for direction"
        }
      ],
      "dominant_signal": "neutral",
      "max_confidence": 0.65,
      "bullish_count": 0,
      "bearish_count": 0,
      "neutral_count": 1
    }
  },
  "sequence_id": 13,
  "primary_exchange": "BINANCE",
  "data_feed_type": "WEBSOCKET_L2",
  "data_quality_score": 9.5,
  "completeness_pct": 100,
  "reliability_score": 9,
  "bid": 64827.8,
  "ask": 64827.9,
  "tick_direction": -1,
  "twap": 64879.56,
  "pivot_points": {
    "daily": {
      "pivot": 64971.33,
      "r1": 65420.66,
      "r2": 65795.33,
      "r3": 66619.33,
      "s1": 64596.66,
      "s2": 64147.33,
      "s3": 63323.33,
      "vah": 65346,
      "val": 64522,
      "poc": 65046
    },
    "weekly": {
      "pivot": 62830.33,
      "r1": 63346.66,
      "r2": 64083.33,
      "r3": 65336.33,
      "s1": 62093.66,
      "s2": 61577.33,
      "s3": 60324.33,
      "vah": 63567,
      "val": 62314,
      "poc": 62610
    },
    "monthly": {
      "pivot": 63185.67,
      "r1": 64582.34,
      "r2": 65610.67,
      "r3": 68035.67,
      "s1": 62157.34,
      "s2": 60760.67,
      "s3": 58335.67,
      "vah": 64214,
      "val": 61789,
      "poc": 63554
    }
  },
  "immediate_support": [
    64831.71,
    64522,
    64479.84,
    64214,
    64162.07
  ],
  "support_strength": [
    61,
    94.8,
    62,
    72.1,
    46
  ],
  "immediate_resistance": [
    64944.5,
    64971.33,
    65037.38,
    65346
  ],
  "resistance_strength": [
    49,
    98.2,
    55,
    92.4
  ],
  "volatility_metrics": {
    "realized_vol_24h": 0.0184,
    "realized_vol_7d": 0.0487,
    "volatility_regime": "NORMAL"
  },
  "order_flow_extended": {
    "passive_buy_pct": 87.8,
    "passive_sell_pct": 12.2
  },
  "whale_activity": {
    "large_orders_1h": [
      {
        "size": 1,
        "price": 64896.34,
        "side": "SELL",
        "timestamp_ms": 1786141945725
      },
      {
        "size": 1,
        "price": 64883.14,
        "side": "SELL",
        "timestamp_ms": 1786142044496
      },
      {
        "size": 1,
        "price": 64870.63,
        "side": "SELL",
        "timestamp_ms": 1786142264045
      },
      {
        "size": 1,
        "price": 64853.03,
        "side": "SELL",
        "timestamp_ms": 1786142272654
      },
      {
        "size": 1,
        "price": 64867.43,
        "side": "SELL",
        "timestamp_ms": 1786142499673
      }
    ],
    "iceberg_activity": 1,
    "hidden_orders_detected": 0
  },
  "technical_indicators_extended": {
    "cci_1h": 97.67,
    "cci_signal": "NEUTRAL",
    "stochastic": {
      "k": 58.57,
      "d": 39.05,
      "signal": "NEUTRAL",
      "source": "real"
    },
    "williams_r": {
      "value": -67.09,
      "overbought": 0,
      "oversold": 0,
      "zone": "neutral",
      "source": "real"
    },
    "hurst_exponent": 0.3246,
    "shannon_entropy": 2.9163,
    "fractal_dimension": 0.6199,
    "kalman_filter": {
      "kalman_price": 64925.85,
      "raw_price": 64872,
      "deviation_pct": -0.08,
      "trend_direction": "DOWN"
    },
    "regression_channel": {
      "slope_per_bar": -1.1318,
      "trend_price": 64888.64,
      "upper_1sd": 64905.32,
      "lower_1sd": 64871.95,
      "upper_2sd": 64922.01,
      "lower_2sd": 64855.26,
      "deviation_from_trend": -16.64,
      "position_in_channel": 0.2508
    },
    "dominant_cycles": {
      "dominant_cycles": [
        100,
        40,
        33.3
      ],
      "cycle_strengths": [
        3118.06,
        1966.69,
        1010.51
      ]
    },
    "monte_carlo": {
      "median_price": 64849.48,
      "p10": 64792.89,
      "p25": 64821.14,
      "p75": 64882.52,
      "p90": 64912.7,
      "prob_up": 0.43,
      "horizon_bars": 12
    },
    "garch_forecast_1h": 0.0017
  },
  "backup_exchanges": [
    "COINBASE",
    "KRAKEN",
    "OKX"
  ],
  "alerts": {
    "alert_count": 0,
    "max_severity": "NONE"
  },
  "regime_analysis": {
    "current_regime": "MEAN_REVERTING",
    "regime_probabilities": {
      "trending": 0,
      "mean_reverting": 0.714,
      "breakout": 0.286
    },
    "regime_change_probability": 0.29,
    "expected_regime_duration": "15m-1h",
    "avg_adx": 25.9
  },
  "data_reliability": {
    "has_options_data": 0,
    "onchain_coverage": "full",
    "latency_acceptable": 1,
    "price_targets_available": 1
  },
  "event_id": "a92dec97",
  "data_context": "real_time",
  "timestamp_ny": "2026-08-07T18:41:59.917-04:00",
  "timestamp_sp": "2026-08-07T19:41:59.917-03:00",
  "_log_id": "a92dec97"
}
----------------------------------------------------------------------------------------------------
EVENTO: Alerta | JANELA: 12
UTC: 2026-08-07 22:42:09 UTC
NY:  2026-08-07 18:42:09 EST/EDT
SP:  2026-08-07 19:42:09 BRT
----------------------------------------------------------------------------------------------------
{
  "tipo_evento": "Alerta",
  "resultado_da_batalha": "VOLATILITY_SQUEEZE",
  "descricao": "Tipo: VOLATILITY_SQUEEZE",
  "timestamp": "2026-08-07T22:42:09+00:00",
  "severity": "LOW",
  "probability": 0.4,
  "action": "WATCH_FOR_NORMALIZATION",
  "context": {
    "price": 64856.3,
    "volume": 11.105,
    "average_volume": 4.635,
    "volatility": 1.2e-07
  },
  "data_context": "real_time",
  "janela_numero": 12,
  "epoch_ms": 1786142529331,
  "event_id": "4cdd3c4d",
  "timestamp_utc": "2026-08-07T22:42:09.331+00:00",
  "timestamp_ny": "2026-08-07T18:42:09.331-04:00",
  "timestamp_sp": "2026-08-07T19:42:09.331-03:00",
  "price_data": {
    "current": {
      "last": 64856.3,
      "volume": 11.105
    }
  },
  "volatility_metrics": {
    "realized_vol_24h": 0
  },
  "market_context": {
    "trading_session": "NY_OVERLAP",
    "session_phase": "ACTIVE"
  },
  "_log_id": "4cdd3c4d"
}

----------------------------------------------------------------------------------------------------
 # Janela 13
----------------------------------------------------------------------------------------------------
EVENTO: ANALYSIS_TRIGGER | SYMBOL: BTCUSDT | JANELA: 13
UTC: 2026-08-07 22:43:00 UTC
NY:  2026-08-07 18:43:00 EST/EDT
SP:  2026-08-07 19:43:00 BRT
----------------------------------------------------------------------------------------------------
{
  "is_signal": 1,
  "tipo_evento": "ANALYSIS_TRIGGER",
  "descricao": "Evento automático para análise da IA",
  "symbol": "BTCUSDT",
  "raw_event": {
    "delta": -2.03,
    "volume_total": 2.311,
    "volume_compra": 0.14,
    "volume_venda": 2.17,
    "preco_fechamento": 64856.3,
    "advanced_analysis": {
      "symbol": "BTCUSDT",
      "price": 64856.3,
      "volume": 2.311,
      "adaptive_thresholds": {
        "current_volatility": 0.03,
        "volatility_factor": 2.0,
        "absorption_threshold": 0.3,
        "flow_threshold": 0.2
      },
      "onchain_metrics": {
        "difficulty": 126.23,
        "active_addresses": 717974,
        "mempool_size": 84905,
        "mempool_vsize_mb": 43.24,
        "mempool_total_fee_btc": 0.1258,
        "fees_fastest_sat_vb": 4,
        "fees_half_hour_sat_vb": 4,
        "fees_hour_sat_vb": 3,
        "fees_economy_sat_vb": 2,
        "difficulty_adjustment": {
          "progress_pct": 93.45,
          "estimated_change_pct": 0.75,
          "remaining_blocks": 132,
          "remaining_time_ms": 78653520,
          "previous_retarget_pct": -0.74
        },
        "minutes_between_blocks": 8.4423,
        "total_btc_sent_24h": 807619.08,
        "trade_volume_btc_24h": 3265.64,
        "data_source": "blockchain.info+mempool.space",
        "is_real_data": 1,
        "requires_paid_api": [
          "exchange_netflow",
          "whale_transactions",
          "exchange_reserves",
          "sopr"
        ]
      }
    }
  },
  "resultado_da_batalha": "N/A",
  "delta": -2.03,
  "volume_total": 2.311,
  "volume_compra": 0.14,
  "volume_venda": 2.17,
  "preco_fechamento": 64856.3,
  "timestamp": "2026-08-07T22:43:07Z",
  "epoch_ms": 1786142580000,
  "ml_features": {
    "price_features": {
      "returns_1": 0.0,
      "volatility_1": 0.0,
      "returns_5": 0.0,
      "volatility_5": 1e-07,
      "returns_15": 1.5e-07,
      "volatility_15": 1e-07,
      "momentum_score": 1.41421356,
      "volatility_1h": 0.0023
    },
    "volume_features": {
      "volume_sma_ratio": 5,
      "volume_momentum": 27.115,
      "buy_sell_pressure": -0.8785,
      "liquidity_gradient": 27.11466734
    },
    "microstructure": {
      "order_book_slope": -3.789736,
      "flow_imbalance": -0.8785,
      "tick_rule_sum": 1,
      "trade_intensity": 45,
      "trade_intensity_v2": 1.5167
    },
    "cross_asset": {
      "btc_eth_corr_7d": 0.8212,
      "btc_eth_corr_30d": 0.8599,
      "btc_dxy_corr_30d": -0.0605,
      "btc_dxy_corr_90d": -0.067,
      "btc_ndx_corr_30d": 0.4147,
      "dxy_return_5d": -0.3561,
      "dxy_return_20d": -1.6548,
      "btc_dxy_correlation_stability": 0.0065,
      "btc_dxy_inverse_strength": 0.0638,
      "dxy_momentum": -1.29867555,
      "vix_current": 14.9,
      "us10y_yield": 4.66,
      "btc_dominance": 17.0377,
      "btc_dominance_change_7d": 0,
      "eth_dominance": 9.0966,
      "gold_price": 4342.3515,
      "oil_price": 77.08,
      "macro_regime": "RISK_ON",
      "correlation_regime": "DECORRELATED"
    }
  },
  "orderbook_data": {
    "mid": 64827.85,
    "spread": 0.1,
    "spread_percent": 0,
    "bid_depth_usd": 2385544.18,
    "ask_depth_usd": 1070989.07,
    "imbalance": 0.38,
    "flow_imbalance": 0.3803,
    "volume_ratio": 2.227,
    "pressure": 0.3803,
    "consolidated_bias_score": 0.7368,
    "spread_bps": 0.0154,
    "is_valid": 1,
    "data_source": "live",
    "spread_volatility": 0.0
  },
  "historical_vp": {
    "daily": {
      "poc": 65046,
      "vah": 65346,
      "val": 64522,
      "status": "success"
    },
    "weekly": {
      "poc": 62610,
      "vah": 63567,
      "val": 62314,
      "status": "success"
    },
    "monthly": {
      "poc": 63554,
      "vah": 64214,
      "val": 61789,
      "status": "success"
    }
  },
  "multi_tf": {
    "15m": {
      "tendencia": "Baixa",
      "preco_atual": 64856.34,
      "mme_21": 64923.98,
      "atr": 73.63,
      "regime": "Range",
      "rsi_short": 35.9,
      "rsi_long": 42.72,
      "macd": 11.4051,
      "macd_signal": 16.4065,
      "adx": 23.83,
      "realized_vol": 0.0014
    },
    "1h": {
      "tendencia": "Alta",
      "preco_atual": 64856.34,
      "mme_21": 64787.38,
      "atr": 227.14,
      "regime": "Range",
      "rsi_short": 51.41,
      "rsi_long": 53.55,
      "macd": 99.1064,
      "macd_signal": 98.0508,
      "adx": 23.82,
      "realized_vol": 0.0023
    },
    "4h": {
      "tendencia": "Alta",
      "preco_atual": 64872,
      "mme_21": 64460.35,
      "atr": 498.88,
      "regime": "Range",
      "rsi_short": 61.94,
      "rsi_long": 61.09,
      "macd": 274.9477,
      "macd_signal": 256.7109,
      "adx": 30.01,
      "realized_vol": 0.0058
    },
    "1d": {
      "tendencia": "Alta",
      "preco_atual": 64872,
      "mme_21": 64144.77,
      "atr": 1369.69,
      "regime": "Manipulação",
      "rsi_short": 58.86,
      "rsi_long": 55.63,
      "macd": 67.1517,
      "macd_signal": 35.9228,
      "adx": 15.05,
      "realized_vol": 0.0184
    }
  },
  "data_context": "real_time",
  "price_targets": [
    {
      "level": 66801.99,
      "confidence": 0.3,
      "source": "fallback_r1",
      "weight": 0.1
    },
    {
      "level": 68747.68,
      "confidence": 0.2,
      "source": "fallback_r2",
      "weight": 0.05
    },
    {
      "level": 62910.61,
      "confidence": 0.3,
      "source": "fallback_s1",
      "weight": 0.1
    },
    {
      "level": 60964.92,
      "confidence": 0.2,
      "source": "fallback_s2",
      "weight": 0.05
    },
    {
      "level": 64856.3,
      "confidence": 0.4,
      "source": "fallback_poc",
      "weight": 0.15
    }
  ],
  "pattern_recognition": {
    "smart_money": {
      "fair_value_gaps": [
        {
          "type": "BEARISH",
          "top": 64925,
          "bottom": 64892.3,
          "gap_size": 32.7,
          "gap_pct": 0.05
        }
      ],
      "market_structure": {
        "structure": "BEARISH",
        "bos_detected": 0,
        "last_swing_high": 64925,
        "last_swing_low": 64909.55,
        "higher_highs": 0,
        "higher_lows": 0
      }
    }
  },
  "janela_numero": 13,
  "timestamp_utc": "2026-08-07T22:43:00.000Z",
  "derivatives": {
    "BTCUSDT": {
      "funding_rate_percent": 0.01,
      "open_interest": 106945.765,
      "open_interest_usd": 6934687958.97,
      "long_short_ratio": 1.1,
      "longs_usd": 3631669164.51,
      "shorts_usd": 3303018794.46
    },
    "ETHUSDT": {
      "funding_rate_percent": 0,
      "open_interest": 2279164.179,
      "open_interest_usd": 4360633657.11,
      "long_short_ratio": 2.07,
      "longs_usd": 2939074192.69,
      "shorts_usd": 1421559464.42
    }
  },
  "fluxo_continuo": {
    "cvd": -30.5804,
    "whale_buy_volume": 0,
    "whale_sell_volume": 5,
    "whale_delta": -5,
    "bursts": {
      "count": 9,
      "max_burst_volume": 7.558
    },
    "sector_flow": {
      "retail": {
        "buy": 12.1643,
        "sell": 33.9753,
        "delta": -21.811
      },
      "mid": {
        "buy": 1.522,
        "sell": 5.2914,
        "delta": -3.769
      },
      "whale": {
        "buy": 0,
        "sell": 5,
        "delta": -5
      }
    },
    "timestamp": "2026-08-07T22:43:00.000+00:00",
    "time_index": {
      "epoch_ms": 1786142580000,
      "timestamp_utc": "2026-08-07T22:43:00.000+00:00",
      "timestamp_ny": "2026-08-07T18:43:00.000-04:00",
      "timestamp_sp": "2026-08-07T19:43:00.000-03:00"
    },
    "order_flow": {
      "net_flow_1m": -130491.5829,
      "absorcao_1m": "Absorção de Compra",
      "buy_volume": 9099.99,
      "sell_volume": 140753.15,
      "total_volume": 149853.15,
      "buy_volume_btc": 0.14,
      "sell_volume_btc": 2.17,
      "total_volume_btc": 2.311,
      "whale_buy_volume_window": 0,
      "whale_sell_volume_window": 0,
      "whale_delta_window": 0,
      "flow_imbalance": -0.8785,
      "aggressive_buy_pct": 6.07,
      "aggressive_sell_pct": 93.93,
      "net_flow_5m": -305255.9042,
      "absorcao_5m": "Neutra",
      "net_flow_15m": -1984485.7022,
      "absorcao_15m": "Neutra",
      "computation_window_min": 1,
      "available_windows_min": [
        1,
        5,
        15
      ],
      "buy_sell_ratio": {
        "buy_sell_ratio": 0.06,
        "ratios": {
          "current": 0.0647,
          "imbalance_1m": -0.871,
          "imbalance_5m": -2.037,
          "imbalance_15m": -13.243
        },
        "sector_ratios": {
          "retail": 0.358,
          "mid": 0.2876,
          "whale": 0
        },
        "pressure": "STRONG_SELL",
        "flow_trend": "increasing_selling",
        "buy_volume": 0.14,
        "sell_volume": 2.17
      }
    },
    "tipo_absorcao": "Absorção de Compra",
    "participant_analysis": {
      "retail": {
        "volume_pct": 66.93,
        "direction": "SELL",
        "sentiment": "BEARISH",
        "composite_score": -0.622,
        "imbalance": -0.819
      },
      "mid": {
        "volume_pct": 33.07,
        "direction": "SELL",
        "sentiment": "BEARISH",
        "composite_score": -0.533,
        "imbalance": -1
      }
    },
    "liquidity_heatmap": {
      "clusters": [
        {
          "center": 64862.0139,
          "low": 64846,
          "high": 64872,
          "width": 26,
          "total_volume": 15.714,
          "buy_volume": 3.545,
          "sell_volume": 12.169,
          "imbalance": -8.623,
          "imbalance_ratio": -0.549,
          "trades_count": 2000,
          "avg_trade_size": 0.008,
          "recent_timestamp": 1786142586236,
          "recent_ts_ms": 1786142586236,
          "last_seen_ms": 1786142586236,
          "first_seen_ms": 1786142315774,
          "age_ms": 309.0,
          "cluster_duration_ms": 270462,
          "price_std": 10.0286,
          "volume_std": 0.043,
          "bin_threshold_usd": 194.586
        }
      ],
      "resistances": [
        64862.0139
      ],
      "clusters_count": 1
    },
    "absorption_analysis": {
      "current_absorption": {
        "index": 0.765,
        "classification": "STRONG_ABSORPTION",
        "label": "Absorção de Compra",
        "buyer_strength": 0.6,
        "seller_exhaustion": 0.6,
        "continuation_probability": 0.69,
        "delta_usd": -130491.583,
        "total_volume_usd": 149853.15,
        "flow_imbalance": -0.8785,
        "window_min": 1
      }
    }
  },
  "market_context": {
    "trading_session": "NY",
    "session_phase": "ACTIVE",
    "time_to_session_close": 11964,
    "day_of_week": 4,
    "is_holiday": 0,
    "market_hours_type": "EXTENDED"
  },
  "market_environment": {
    "volatility_regime": "NORMAL",
    "trend_direction": "UP",
    "market_structure": "RANGE_BOUND",
    "liquidity_environment": "NORMAL",
    "risk_sentiment": "BULLISH",
    "correlation_spy": 0.4056,
    "correlation_dxy": -0.0854,
    "correlation_gold": 0.2379
  },
  "contextual_snapshot": {
    "symbol": "BTCUSDT",
    "ohlc": {
      "open": 64856.3,
      "high": 64856.3,
      "low": 64856.3,
      "close": 64856.3,
      "open_time": 1786142520999,
      "close_time": 1786142579877,
      "vwap": 64856.3
    },
    "volume_total": 2.311,
    "volume_total_usdt": 149853,
    "volume_compra": 0.14,
    "volume_venda": 2.17,
    "num_trades": 82,
    "delta_minimo": -2.037,
    "delta_maximo": 0.057,
    "delta_fechamento": -2.03,
    "reversao_desde_minimo": 0.01,
    "reversao_desde_maximo": 2.09,
    "poc_price": 64856.3,
    "poc_volume": 2.17,
    "poc_percentage": 93.9,
    "dwell_price": 64856.3,
    "dwell_seconds": 58,
    "dwell_location": "High",
    "trades_per_second": 1.39,
    "avg_trade_size": 0.028
  },
  "order_book_depth": {
    "L1": {
      "bids": 1020324.74,
      "asks": 443876.63,
      "flow_imbalance": 0.3937
    },
    "L5": {
      "bids": 1164112.29,
      "asks": 455740.17,
      "flow_imbalance": 0.4373
    },
    "L10": {
      "bids": 1439950.05,
      "asks": 466825.88,
      "flow_imbalance": 0.5104
    },
    "L25": {
      "bids": 1809520.09,
      "asks": 502028.41,
      "flow_imbalance": 0.5656
    },
    "total_depth_ratio": 3.6
  },
  "orderbook_quality": "live",
  "market_impact": {
    "slippage_matrix": {
      "100k_usd": {
        "buy": 0.05,
        "sell": 0.05
      },
      "1m_usd": {
        "buy": 4.55,
        "sell": 0.05
      }
    },
    "liquidity_score": 9.9985,
    "execution_quality": "EXCELLENT"
  },
  "institutional_analytics": {
    "status": "ok",
    "computed_at_ms": 1786142587188,
    "technical_extras": {
      "stoch_rsi": {
        "k": 53.25,
        "d": 50.29,
        "overbought": 0,
        "oversold": 0,
        "crossover": "none"
      },
      "williams_r": {
        "value": -86.96,
        "overbought": 0,
        "oversold": 1,
        "zone": "oversold",
        "source": "real"
      },
      "hurst_exponent": 0.3232,
      "shannon_entropy": 2.9154,
      "kalman_filter": {
        "kalman_price": 64923.69,
        "raw_price": 64856.3,
        "deviation_pct": -0.1,
        "trend_direction": "DOWN"
      },
      "regression_channel": {
        "slope_per_bar": -1.1674,
        "trend_price": 64885.7,
        "upper_1sd": 64902.79,
        "lower_1sd": 64868.61,
        "upper_2sd": 64919.88,
        "lower_2sd": 64851.52,
        "deviation_from_trend": -29.4,
        "position_in_channel": 0.0699
      },
      "dominant_cycles": {
        "dominant_cycles": [
          100,
          40,
          33.3
        ],
        "cycle_strengths": [
          3152.84,
          2032.48,
          1042.2
        ]
      },
      "fractal_dimension": 0.6187,
      "monte_carlo": {
        "median_price": 64847.2,
        "p10": 64790.85,
        "p25": 64818.98,
        "p75": 64880.09,
        "p90": 64910.14,
        "prob_up": 0.42,
        "horizon_bars": 12
      },
      "fair_value_gaps": [
        {
          "type": "BEARISH",
          "top": 64925,
          "bottom": 64892.3,
          "gap_size": 32.7,
          "gap_pct": 0.05
        }
      ],
      "market_structure": {
        "structure": "BEARISH",
        "bos_detected": 0,
        "last_swing_high": 64925,
        "last_swing_low": 64909.55,
        "higher_highs": 0,
        "higher_lows": 0
      }
    },
    "profile_analysis": {
      "poor_extremes": {
        "poor_high": {
          "detected": 1,
          "price": 64856.34,
          "volume_ratio": 0.607,
          "implication": "High likely to be revisited - unfinished auction"
        },
        "poor_low": {
          "detected": 1,
          "price": 64856.33,
          "volume_ratio": 9.393,
          "implication": "Low likely to be revisited - unfinished auction"
        },
        "excess_high": 0,
        "excess_low": 0,
        "session_high": 64856.34,
        "session_low": 64856.33,
        "action_bias": "expect_retest_both",
        "status": "success"
      },
      "profile_shape": {
        "shape": "b",
        "implication": "Long liquidation - bullish bias expected",
        "trading_signal": "BULLISH_AFTER",
        "distribution": {
          "lower_third_pct": 93.9,
          "middle_third_pct": 0,
          "upper_third_pct": 6.1
        },
        "dominant_zone": "lower",
        "status": "success"
      },
      "no_mans_land": {
        "zones": [
          {
            "range_low": 64512,
            "range_high": 64810,
            "gap_size": 298,
            "gap_size_pct": 0.462,
            "risk": "LOW",
            "lvn_confirmed": 1,
            "lvns_in_zone": 9,
            "nearest_hvn_below": 64512,
            "nearest_hvn_above": 64810,
            "distance_from_price": 46.3,
            "distance_pct": 0.07,
            "direction": "below"
          }
        ],
        "price_in_no_mans_land": 0,
        "nearest_no_mans_land": {
          "range_low": 64512,
          "range_high": 64810,
          "gap_size": 298,
          "gap_size_pct": 0.462,
          "risk": "LOW",
          "lvn_confirmed": 1,
          "lvns_in_zone": 9,
          "nearest_hvn_below": 64512,
          "nearest_hvn_above": 64810,
          "distance_from_price": 46.3,
          "distance_pct": 0.07,
          "direction": "below"
        },
        "total_zones": 1,
        "max_gap_pct": 0.46,
        "status": "success"
      },
      "va_volume_pct": {
        "value_area_volume_pct": 100,
        "interpretation": "extremely_compressed",
        "breakout_risk": "VERY_HIGH",
        "volume_in_va": 1,
        "total_volume": 1,
        "compression_signal": 1
      },
      "volume_node_strength": {
        "scored_hvns": [
          {
            "price": 64304,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64306,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.8,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64314,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.8,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64320,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.9,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64322,
            "strength": 71,
            "volume_score": 15,
            "proximity_score": 25.9,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "weekly",
              "monthly"
            ],
            "in_value_area": 0
          }
        ],
        "scored_lvns": [
          {
            "price": 64404,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.5,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64423,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.7,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64436,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.8,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64440,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.8,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          },
          {
            "price": 64447,
            "strength": 65,
            "volume_score": 15,
            "proximity_score": 26.8,
            "multi_tf_confluence": 1,
            "confluence_sources": [
              "monthly"
            ],
            "in_value_area": 0
          }
        ],
        "total_hvns": 52,
        "total_lvns": 101,
        "avg_hvn_strength": 66.2,
        "avg_lvn_strength": 63.1,
        "status": "success"
      }
    },
    "flow_analysis": {
      "passive_aggressive": {
        "aggressive": {
          "buy_pct": 6.07,
          "sell_pct": 93.93,
          "net_pct": -87.86,
          "dominance": "sellers",
          "buy_volume": 0.14,
          "sell_volume": 2.17
        },
        "passive": {
          "dominance": "buyers",
          "inference": "from_orderbook_depth",
          "bid_depth": 2385544.18,
          "ask_depth": 1070989.07,
          "bid_ratio": 0.69,
          "ob_imbalance": 0.38
        },
        "composite": {
          "agreement": 0,
          "signal": "sell_absorption",
          "interpretation": "Aggressive sellers hitting passive buy walls - potential reversal or breakdown",
          "conviction": "MEDIUM"
        },
        "status": "success"
      },
      "whale_accumulation": {
        "score": -13,
        "classification": "NEUTRAL",
        "bias": "DISTRIBUTING",
        "components": {
          "flow": {
            "score": -30,
            "max": 30,
            "detail": {
              "divergence": "aligned",
              "whale_delta": -5,
              "mid_delta": -3.769,
              "retail_delta": -21.811,
              "primary_delta": -5
            }
          },
          "depth": {
            "score": 7.61,
            "max": 20,
            "detail": {
              "bid_depth": 2385544.18,
              "ask_depth": 1070989.07,
              "ratio": 0.38,
              "deep_confirmation": 0
            }
          },
          "absorption": {
            "score": 8,
            "max": 25,
            "detail": {
              "buyer_strength": 0.6,
              "seller_exhaustion": 0.6,
              "net_absorption": 0,
              "index": 0.765,
              "label": "Absorção de Compra"
            }
          },
          "derivatives": {
            "score": 1.49,
            "max": 25,
            "detail": {
              "long_short_ratio": 1.1,
              "lsr_score": 1.49
            }
          }
        },
        "trend": {
          "direction": "stable",
          "avg_score": -24.3,
          "recent_avg": -22.6,
          "momentum": 11.3,
          "samples": 14,
          "score_range": {
            "min": -49,
            "max": 14
          }
        },
        "status": "success"
      },
      "absorption_zones": {
        "total_zones": 0,
        "total_events": 0,
        "buy_zone_count": 0,
        "sell_zone_count": 0,
        "status": "no_events"
      }
    },
    "sr_analysis": {
      "defense_zones": {
        "buy_defense": [
          {
            "center": 64479.84,
            "range_low": 64376.36,
            "range_high": 64570.64,
            "strength": 62,
            "side": "buy",
            "sources": [
              "sr_level_ema_21_4h",
              "vp_hvn",
              "ema_ema_21_4h",
              "vp_val"
            ],
            "source_count": 4,
            "signals_in_zone": 5,
            "type": "confluence",
            "distance_from_price": 376.46,
            "distance_pct": 0.58
          },
          {
            "center": 64831.51,
            "range_low": 64738.74,
            "range_high": 64928.64,
            "strength": 61,
            "side": "buy",
            "sources": [
              "orderbook_bid_wall",
              "vp_hvn",
              "ema_ema_21_1h",
              "sr_level_ema_21_1h"
            ],
            "source_count": 4,
            "signals_in_zone": 8,
            "type": "confluence",
            "distance_from_price": 24.79,
            "distance_pct": 0.04
          },
          {
            "center": 64162.07,
            "range_low": 64096.13,
            "range_high": 64228.02,
            "strength": 46,
            "side": "buy",
            "sources": [
              "ema_ema_21_1d",
              "sr_level_vah_monthly"
            ],
            "source_count": 2,
            "signals_in_zone": 2,
            "type": "cluster",
            "distance_from_price": 694.23,
            "distance_pct": 1.07
          },
          {
            "center": 64336.25,
            "range_low": 64255.36,
            "range_high": 64442.64,
            "strength": 40,
            "side": "buy",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 9,
            "type": "cluster",
            "distance_from_price": 520.05,
            "distance_pct": 0.8
          }
        ],
        "sell_defense": [
          {
            "center": 65037.38,
            "range_low": 64960.36,
            "range_high": 65128.64,
            "strength": 55,
            "side": "sell",
            "sources": [
              "sr_level_poc_daily",
              "vp_poc",
              "vp_hvn"
            ],
            "source_count": 3,
            "signals_in_zone": 9,
            "type": "confluence",
            "distance_from_price": 181.08,
            "distance_pct": 0.28
          },
          {
            "center": 65346,
            "range_low": 65297.36,
            "range_high": 65394.64,
            "strength": 53,
            "side": "sell",
            "sources": [
              "sr_level_vah_daily",
              "vp_hvn",
              "vp_vah"
            ],
            "source_count": 3,
            "signals_in_zone": 3,
            "type": "confluence",
            "distance_from_price": 489.7,
            "distance_pct": 0.76
          },
          {
            "center": 64944.32,
            "range_low": 64860.36,
            "range_high": 65052.64,
            "strength": 49,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "ema_ema_21_15m",
              "sr_level_hvn_daily"
            ],
            "source_count": 3,
            "signals_in_zone": 9,
            "type": "confluence",
            "distance_from_price": 88.02,
            "distance_pct": 0.14
          },
          {
            "center": 65157.17,
            "range_low": 65064.36,
            "range_high": 65257.64,
            "strength": 41,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 13,
            "type": "cluster",
            "distance_from_price": 300.87,
            "distance_pct": 0.46
          },
          {
            "center": 65252.3,
            "range_low": 65164.36,
            "range_high": 65350.64,
            "strength": 40,
            "side": "sell",
            "sources": [
              "vp_hvn",
              "sr_level_hvn_daily"
            ],
            "source_count": 2,
            "signals_in_zone": 11,
            "type": "cluster",
            "distance_from_price": 396,
            "distance_pct": 0.61
          }
        ],
        "total_zones": 9,
        "strongest_buy": {
          "center": 64479.84,
          "range_low": 64376.36,
          "range_high": 64570.64,
          "strength": 62,
          "side": "buy",
          "sources": [
            "sr_level_ema_21_4h",
            "vp_hvn",
            "ema_ema_21_4h",
            "vp_val"
          ],
          "source_count": 4,
          "signals_in_zone": 5,
          "type": "confluence",
          "distance_from_price": 376.46,
          "distance_pct": 0.58
        },
        "strongest_sell": {
          "center": 65037.38,
          "range_low": 64960.36,
          "range_high": 65128.64,
          "strength": 55,
          "side": "sell",
          "sources": [
            "sr_level_poc_daily",
            "vp_poc",
            "vp_hvn"
          ],
          "source_count": 3,
          "signals_in_zone": 9,
          "type": "confluence",
          "distance_from_price": 181.08,
          "distance_pct": 0.28
        },
        "defense_asymmetry": {
          "ratio": 0.88,
          "bias": "slight_sell_defense",
          "description": "Slightly more sell defense",
          "buy_total_strength": 209,
          "sell_total_strength": 238
        },
        "status": "success"
      }
    },
    "quality": {
      "calendar": {
        "day_of_week": "Friday",
        "day_of_week_num": 4,
        "is_us_holiday": 0,
        "is_weekend": 0,
        "is_pre_holiday": 0,
        "is_post_holiday": 0,
        "expected_liquidity": "NORMAL",
        "liquidity_warning": 0
      },
      "latency": {
        "latency_ms": 7401,
        "latency_category": "POOR",
        "data_freshness": "DELAYED",
        "is_acceptable": 0,
        "is_stale": 0
      },
      "spread_percentile": {
        "status": "ok",
        "current_spread": 0.1,
        "spread_percentile": 0,
        "spread_mean": 0.1,
        "spread_median": 0.1,
        "spread_std": 0,
        "spread_min": 0.1,
        "spread_max": 0.1,
        "spread_z_score": 0,
        "is_tight": 1,
        "is_wide": 0,
        "is_anomalous": 0,
        "liquidity_signal": "EXCELLENT",
        "samples": 14,
        "window_minutes": 1440
      },
      "anomalies": {
        "anomalies_detected": 1,
        "count": 1,
        "anomalies": [
          {
            "type": "FLOW_EXTREME_IMBALANCE",
            "severity": "HIGH",
            "value": -0.8785,
            "direction": "SELL",
            "description": "Extreme flow imbalance: -87.85% toward sellers"
          }
        ],
        "max_severity": "HIGH",
        "risk_elevated": 1,
        "types_found": [
          "FLOW_EXTREME_IMBALANCE"
        ],
        "summary": "1 anomalies detected (max severity: HIGH)"
      }
    }
  },
  "sequence_id": 14,
  "primary_exchange": "BINANCE",
  "data_feed_type": "WEBSOCKET_L2",
  "data_quality_score": 9,
  "completeness_pct": 100,
  "reliability_score": 8,
  "bid": 64827.8,
  "ask": 64827.9,
  "tick_direction": -1,
  "twap": 64877.9,
  "pivot_points": {
    "daily": {
      "pivot": 64971.33,
      "r1": 65420.66,
      "r2": 65795.33,
      "r3": 66619.33,
      "s1": 64596.66,
      "s2": 64147.33,
      "s3": 63323.33,
      "vah": 65346,
      "val": 64522,
      "poc": 65046
    },
    "weekly": {
      "pivot": 62830.33,
      "r1": 63346.66,
      "r2": 64083.33,
      "r3": 65336.33,
      "s1": 62093.66,
      "s2": 61577.33,
      "s3": 60324.33,
      "vah": 63567,
      "val": 62314,
      "poc": 62610
    },
    "monthly": {
      "pivot": 63185.67,
      "r1": 64582.34,
      "r2": 65610.67,
      "r3": 68035.67,
      "s1": 62157.34,
      "s2": 60760.67,
      "s3": 58335.67,
      "vah": 64214,
      "val": 61789,
      "poc": 63554
    }
  },
  "immediate_support": [
    64831.51,
    64522,
    64479.84,
    64214,
    64162.07
  ],
  "support_strength": [
    61,
    94.8,
    62,
    72.1,
    46
  ],
  "immediate_resistance": [
    64944.32,
    64971.33,
    65037.38,
    65346
  ],
  "resistance_strength": [
    49,
    98.2,
    55,
    92.4
  ],
  "volatility_metrics": {
    "realized_vol_24h": 0.0184,
    "realized_vol_7d": 0.0487,
    "volatility_regime": "NORMAL"
  },
  "order_flow_extended": {
    "passive_buy_pct": 69,
    "passive_sell_pct": 31
  },
  "whale_activity": {
    "large_orders_1h": [
      {
        "size": 1,
        "price": 64896.34,
        "side": "SELL",
        "timestamp_ms": 1786141945725
      },
      {
        "size": 1,
        "price": 64883.14,
        "side": "SELL",
        "timestamp_ms": 1786142044496
      },
      {
        "size": 1,
        "price": 64870.63,
        "side": "SELL",
        "timestamp_ms": 1786142264045
      },
      {
        "size": 1,
        "price": 64853.03,
        "side": "SELL",
        "timestamp_ms": 1786142272654
      },
      {
        "size": 1,
        "price": 64867.43,
        "side": "SELL",
        "timestamp_ms": 1786142499673
      }
    ],
    "iceberg_activity": 1,
    "hidden_orders_detected": 0
  },
  "technical_indicators_extended": {
    "cci_1h": 80.96,
    "cci_signal": "NEUTRAL",
    "stochastic": {
      "k": 53.25,
      "d": 50.29,
      "signal": "NEUTRAL",
      "source": "real"
    },
    "williams_r": {
      "value": -86.96,
      "overbought": 0,
      "oversold": 1,
      "zone": "oversold",
      "source": "real"
    },
    "hurst_exponent": 0.3232,
    "shannon_entropy": 2.9154,
    "fractal_dimension": 0.6187,
    "kalman_filter": {
      "kalman_price": 64923.69,
      "raw_price": 64856.3,
      "deviation_pct": -0.1,
      "trend_direction": "DOWN"
    },
    "regression_channel": {
      "slope_per_bar": -1.1674,
      "trend_price": 64885.7,
      "upper_1sd": 64902.79,
      "lower_1sd": 64868.61,
      "upper_2sd": 64919.88,
      "lower_2sd": 64851.52,
      "deviation_from_trend": -29.4,
      "position_in_channel": 0.0699
    },
    "dominant_cycles": {
      "dominant_cycles": [
        100,
        40,
        33.3
      ],
      "cycle_strengths": [
        3152.84,
        2032.48,
        1042.2
      ]
    },
    "monte_carlo": {
      "median_price": 64847.2,
      "p10": 64790.85,
      "p25": 64818.98,
      "p75": 64880.09,
      "p90": 64910.14,
      "prob_up": 0.42,
      "horizon_bars": 12
    },
    "garch_forecast_1h": 0.0018
  },
  "backup_exchanges": [
    "COINBASE",
    "KRAKEN",
    "OKX"
  ],
  "alerts": {
    "active_alerts": [
      {
        "type": "SUPPORT_TEST",
        "level": 64831.51,
        "severity": "HIGH",
        "probability": 0.92,
        "action": "MONITOR_CLOSELY",
        "description": "Preço testando suporte em 64831.51 (dist: 0.04%)"
      },
      {
        "type": "RESISTANCE_TEST",
        "level": 64944.32,
        "severity": "MEDIUM",
        "probability": 0.73,
        "action": "MONITOR_CLOSELY",
        "description": "Preço testando resistência em 64944.32 (dist: 0.14%)"
      },
      {
        "type": "VOLUME_SPIKE",
        "threshold_exceeded": 5,
        "severity": "HIGH",
        "probability": 0.5,
        "action": "PREPARE_ENTRY",
        "description": "Volume 5.0x acima da média"
      }
    ],
    "alert_count": 3,
    "max_severity": "HIGH"
  },
  "regime_analysis": {
    "current_regime": "MEAN_REVERTING",
    "regime_probabilities": {
      "trending": 0,
      "mean_reverting": 1,
      "breakout": 0
    },
    "regime_change_probability": 0.05,
    "expected_regime_duration": "15m-1h",
    "avg_adx": 25.9
  },
  "data_reliability": {
    "has_options_data": 0,
    "onchain_coverage": "full",
    "latency_acceptable": 1,
    "price_targets_available": 1
  },
  "event_id": "16e36cc3",
  "timestamp_ny": "2026-08-07T18:43:00.000-04:00",
  "timestamp_sp": "2026-08-07T19:43:00.000-03:00",
  "_log_id": "16e36cc3"
}
```


## 7.2 Logs de erro/warning em logs/


`logs/run.log` — último bloco (2025-12-22, sincronização de clock):


```log
2025-12-22 21:20:45,116 - INFO - 📊 Nível de log configurado: INFO
2025-12-22 21:20:45,116 - INFO - 🚀 Iniciando bot para BTCUSDT...
2025-12-22 21:20:45,124 - INFO - 📊 Servidor Prometheus iniciado na porta 8000 (/metrics)
2025-12-22 21:20:45,124 - INFO - ✅ AsyncTradeBuffer inicializado | Max size: 2000 | Backpressure: 80% | Batch size: 50
2025-12-22 21:20:45,124 - INFO - ================================================================================
2025-12-22 21:20:45,124 - INFO - 🕐 TIMEMANAGER v2.1.2 - INICIALIZANDO
2025-12-22 21:20:45,124 - INFO -    Tempo local:     1766449245124 ms
2025-12-22 21:20:45,124 - INFO -    Timezone UTC:    UTC
2025-12-22 21:20:45,124 - INFO -    Timezone NY:     America/New_York
2025-12-22 21:20:45,124 - INFO -    Timezone SP:     America/Sao_Paulo
2025-12-22 21:20:45,124 - INFO -    ZoneInfo:        ✅ Disponível
2025-12-22 21:20:45,124 - INFO -    Max offset:      600 ms
2025-12-22 21:20:45,124 - INFO -    Sync samples:    5
2025-12-22 21:20:45,124 - INFO -    Sync interval:   1800 s (30 min)
2025-12-22 21:20:45,125 - INFO - 🔄 Tentativa 1/3
2025-12-22 21:20:45,125 - INFO - 🔄 Iniciando sincronização com Binance (5 amostras)...
2025-12-22 21:20:49,145 - INFO - ✅ Melhor amostra selecionada: RTT=663ms, Offset=180ms
2025-12-22 21:20:49,145 - INFO - ✅ Offset dentro do limite aceitável: 180ms
2025-12-22 21:20:49,145 - INFO - ✅ Sincronização bem-sucedida na tentativa 1
2025-12-22 21:20:49,145 - INFO - 
2025-12-22 21:20:49,145 - INFO - 🔍 DIAGNÓSTICO DO SISTEMA DE TEMPO
2025-12-22 21:20:49,145 - INFO - --------------------------------------------------------------------------------
2025-12-22 21:20:49,145 - INFO - ⏰ Timestamps:
2025-12-22 21:20:49,145 - INFO -    Local time (ms):  1766449249145
2025-12-22 21:20:49,145 - INFO -    Synced time (ms): 1766449249325
2025-12-22 21:20:49,145 - INFO -    Current offset:   180 ms (0.18s)
2025-12-22 21:20:49,145 - INFO - 🌍 Timezones:
2025-12-22 21:20:49,146 - INFO -    UTC: 2025-12-23 00:20:49 UTC
2025-12-22 21:20:49,146 - INFO -    NY:  2025-12-22 19:20:49 EST
2025-12-22 21:20:49,146 - INFO -    SP:  2025-12-22 21:20:49 -03
2025-12-22 21:20:49,146 - INFO -    Offset NY vs UTC: -5.0 horas
2025-12-22 21:20:49,146 - INFO -    Offset SP vs UTC: -3.0 horas
2025-12-22 21:20:49,146 - INFO -    ✅ Offset NY correto: -5.0 horas
2025-12-22 21:20:49,146 - INFO -    ✅ Offset SP correto: -3.0 horas
2025-12-22 21:20:49,146 - INFO - 📊 Estatísticas de Sincronização:
2025-12-22 21:20:49,146 - INFO -    Status:                ok
2025-12-22 21:20:49,146 - INFO -    Server offset:         180 ms
2025-12-22 21:20:49,146 - INFO -    Last RTT:              663 ms
2025-12-22 21:20:49,146 - INFO -    Best RTT:              663 ms
2025-12-22 21:20:49,146 - INFO -    Sync attempts:         1
2025-12-22 21:20:49,146 - INFO -    Sync failures:         0
2025-12-22 21:20:49,146 - INFO -    Auto corrections:      0
2025-12-22 21:20:49,146 - INFO -    Success rate:          100.0%
2025-12-22 21:20:49,146 - INFO -    ZoneInfo available:    ✅ Yes
2025-12-22 21:20:49,146 - INFO - ================================================================================
```


`logs/issues.log` — **arquivo corrompido/binário** (10.918 bytes, conteúdo ilegível em UTF-8; padrão de bytes repetidos). Não foi possível extrair registros legíveis. `logs/test_feature_store.log` — 4 erros históricos de arquivo Parquet não encontrado (2025-12-07), não relacionados ao runtime atual.


`logs/payload_metrics.jsonl` — registros de erro/warning relevantes (últimos):


```jsonl
{"fallback_v1": true, "payload_bytes": 1907, "error": "payload_v2 inválido: price_context.current_price é obrigatório - o LLM não pode analisar sem preço atual"}
{"payload_bytes": 23, "leak_blocked": true, "bytes_after": 0, "payload_root_name": "event", "error": "no_safe_candidate"}
{"payload_bytes": 51366, "leak_blocked": true, "bytes_after": 3737, "payload_root_name": "event"}
{"payload_bytes": 51967, "leak_blocked": true, "bytes_after": 3774, "payload_root_name": "event"}
```


# SEÇÃO 8 — EVIDÊNCIAS DE INCONSISTÊNCIAS


## 8.1 Instanciação real do AIThrottler: apenas em ai_runner.py


A configuração efetiva do throttler existe em **um único ponto de runtime**: `market_orchestrator/ai/ai_runner.py:41`. NÃO há chamada em `main.py` (linhas 215-239 só configuram logging) nem em `market_orchestrator/market_orchestrator.py` (cooldown duplo por `_ai_min_interval_sec` nas linhas 896, 946, 988).


```python
        BotAIRuntimeProtocol,
        BuildCompactPayloadProtocol,
    )

# [THROTTLE] Controle de frequencia de chamadas IA (v3 singleton)
try:
    from common.ai_throttler import get_throttler
    _ai_throttler = get_throttler(
        min_interval=60,
        hard_min_interval=30,
        daily_token_budget=85_000,
        max_calls_per_hour=10,
    )
except ImportError:
[... TRUNCADO: 816 linhas no total, mostradas 14 ...]
```


`analyzer_qwen.py:3455-3457` chama `get_throttler()` SEM kwargs — o singleton (`common/ai_throttler.py:341-352`) entrega a instância que for criada primeiro; se `ai_runner` não inicializou antes, os defaults do dataclass (`min_interval=180/hard=60/budget=50k/max=6`) valem silenciosamente, divergindo do 60/30/85k/10 pretendido. Comportamento dependente da ordem de import.


## 8.2 max_bytes=6144 (guardrail) × bytes_p95_max=90000 (tripwire): limites em camadas diferentes

- Limite POR CHAMADA (guardrail, roda sempre): `analyzer_qwen.py:3641` chama `_ensure_safe_llm_payload(event_data)` ANTES de montar o prompt (linha 3688). Internamente usa `_MAX_BYTES_LLM = 6144` (`llm_payload_guardrail.py:55`), aplicado em `compress_payload(payload, max_bytes=_MAX_BYTES_LLM)` nas linhas 318 (clean large) e 369 (safe candidate). Payload > 6144 B é compactado; sem candidato seguro → aborta (`GUARDRAIL_ABORT`, linha 346-356).
- Limite AGREGADO (tripwire, batch): `analyzer_qwen.py:3125` `_log_payload_tripwires(summary)` só roda a cada **200 métricas OU 600 s** (linha 3118). Usa `_evaluate_payload_tripwires` (definida em `:356`, chamada em `:397`), com config vinda de `get_llm_payload_config()` (`ai_payload_builder.py:404-416`), que carrega `config/model_config.yaml` → `llm_payload.tripwires.bytes_p95_max = 90000`.
- Ordem de execução no pipeline: **guardrail 6144 por evento → chamada ao modelo → summary agregado → tripwire 90000**. Como eventos brutos de 51 KB são bloqueados/compactados pelo guardrail ANTES de entrar no summary, o p95 real fica em ~3,8 KB e o limiar de 90 KB é inalcançável na prática (tripwire nunca dispara).
- `config/settings.py:221` define `PAYLOAD_TRIPWIRE_BYTES_P95_MAX = 90000`, mas o grep de uso em todo o repo retorna **apenas essa linha de definição** — constante morta; o valor efetivo vem do YAML (`model_config.yaml`).

## 8.3 Nenhum evento registra simultaneamente qual.lat e latency_acceptable


Varredura dos 18 eventos de `dados/eventos_fluxo.jsonl`: os campos são **mutuamente exclusivos** — nunca coexistem no mesmo evento.


```text
idx=3  AI_ANALYSIS   qual.lat=POOR ms=6984   latency_acceptable=ausente
idx=9  AI_ANALYSIS   qual.lat=POOR ms=11450  latency_acceptable=ausente
idx=14 Exaustão      qual.lat=ausente        latency_acceptable=1
(evento 16e36cc3 no eventos_visuais.log: latency_acceptable=1, sem bloco qual)
```


Evidência dos dois eventos em `dados/eventos_fluxo.jsonl`:


```json
// idx=3 (epoch_ms=1786141988998) — apenas qual.lat:
"qual": {"lat": "POOR", "ms": 6984}

// idx=14 (epoch_ms=1786142520000) — apenas latency_acceptable:
"data_reliability": {"has_options_data": 0, "onchain_coverage": "partial",
                     "latency_acceptable": 1, "price_targets_available": 0}
```

- Produtor de `qual.lat`: `market_orchestrator/ai/payload_builder_compact.py:1482-1501` — usa `latency_data.latency_category` (default "OK"), grava `qual["lat"] = cat[:4]` e `qual["ms"] = round(lat)`.
- Produtor de `latency_acceptable`: `institutional/enricher.py:2191` — usa `_latency["is_acceptable"]`.
- Inconsistência: os dois campos medem a mesma dimensão (qualidade de latência) com fontes independentes; nos eventos AI_ANALYSIS (onde a latência é POOR/11450 ms) o campo `latency_acceptable` nem é preenchido, e no evento que o preenche (Exaustão) o `qual` está ausente. Impossível correlacionar 11450 ms com "acceptable".
- Em `dados/eventos_visuais.log` só existem 2 blocos `"qual"` (linhas 3979 e 11047); o evento mais recente (16e36cc3, linhas 19990-19995) tem `latency_acceptable: 1` e não carrega bloco qual.

## 8.4 payload_metrics.jsonl: distribuição de erros e fallbacks (1.045 linhas)


Não existe campo `retry`/`attempt`/`strict_json` no arquivo — retries não são rastreáveis; a distribuição de falhas é:


```text
total de linhas ............ 1045
error = (nenhum) ............ 940
error = no_safe_candidate ...  76   (guardrail abortou: sem candidato seguro)
error = 'payload_v2 inválido: price_context.current_price é obrigatório' ... 29
fallback_v1: true ...........  29 / false 1016   → correlaciona 100% com o erro de validação v2
cache_hit: true ............. 244 / false 801
```


Conclusão: 76 chamadas abortadas por `no_safe_candidate` e 29 fallbacks para v1 (eventos sem `price_context.current_price` no payload v2) — 105 de 1.045 registros (10%) não chegaram ao LLM com o payload v2 intacto. Com `max_retries=3 × strict_json_modes=[True, False]` (ver 8.5) cada uma dessas falhas pode multiplicar chamadas reais sem nenhum rastro no metrics.


## 8.5 `_call_openai_compatible` completa (analyzer_qwen.py:3407-3533)


Função completa, que documenta o multiplicador de chamadas por análise:


```python
    def _call_openai_compatible(
        self, prompt: str, max_retries: int = 3
    ) -> Tuple[str, Optional[str]]:
        """Chama cliente OpenAI-compatível de forma síncrona."""
        if self.client is None:
            raise RuntimeError("Cliente não inicializado")

        params = self._get_model_params()
        base_delay = 1.0
        last_error_reason: Optional[str] = None

        # Modelos sem suporte a json_object mode nunca tentam strict_json
        _supports_json_mode = (
            self.mode == "groq"
            and self.model_name not in _MODELS_WITHOUT_JSON_MODE
        )
        for attempt in range(max_retries):
            strict_json_modes = [True, False] if _supports_json_mode else [False]
            try:
                for strict_json in strict_json_modes:
                    messages: List[ChatMessage] = [
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt},
                    ]
                    create_kwargs: Dict[str, Any] = dict(
                        model=self.model_name,
                        messages=messages,  # type: ignore[arg-type]
                        max_tokens=params["max_tokens"],
                        temperature=params["temperature"],
                        timeout=params["timeout"],
                    )
                    if strict_json:
                        create_kwargs["response_format"] = {"type": "json_object"}

                    try:
                        if params.get("reasoning_effort"):
                            create_kwargs["reasoning_effort"] = params["reasoning_effort"]
                        if params.get("top_p") is not None:
                            create_kwargs["top_p"] = params["top_p"]

                        response = self.client.chat.completions.create(**create_kwargs)

                        # Sucesso: registrar no throttler
                        tokens_used = 0
                        if hasattr(response, "usage") and response.usage:
                            tokens_used = getattr(response.usage, "total_tokens", 0)

                        try:
                            from common.ai_throttler import init_throttler
                            _throttler = init_throttler(
                                min_interval=60, hard_min_interval=30,
                                daily_token_budget=85_000, max_calls_per_hour=10,
                            )
                            _throttler.record_call(tokens_used)
                            _throttler.record_success()
                        except Exception:
                            pass

                    except Exception as e:
                        error_str = str(e)
                        reason = self._classify_provider_error(e)

                        if strict_json and reason == "json_validate_failed":
                            logging.warning(
                                "Groq JSON mode rejected response; retrying without provider JSON mode | attempt=%d",
                                attempt + 1,
                            )
                            last_error_reason = reason
                            continue

                        # Tratamento específico de 429: abortar retries
                        if reason == "rate_limited" or "429" in error_str:
                            try:
                                from common.ai_throttler import init_throttler
                                _throttler = init_throttler(
                                    min_interval=60, hard_min_interval=30,
                                    daily_token_budget=85_000, max_calls_per_hour=10,
                                )
                                retry_after = _throttler.parse_retry_after(error_str)
                                _throttler.record_rate_limit(retry_after)
                            except Exception:
                                pass

                            logging.warning(
                                "GROQ 429 (tentativa %d). Cooldown ativado. Abortando retries.",
                                attempt + 1,
                            )
                            return "", "rate_limited"

                        last_error_reason = reason
                        logging.error(
                            f"Erro {(self.mode or 'unknown').upper()} "
                            f"(tentativa {attempt + 1}/{max_retries}): {e}"
                        )
                        raise

                    if response.choices and len(response.choices) > 0:
                        choice = response.choices[0]
                        finish_reason = getattr(choice, "finish_reason", None)
                        content = self._sanitize_llm_text(
                            choice.message.content or ""
                        )
                        if finish_reason == "length":
                            logging.warning(
                                "LLM response truncated by max_tokens | mode=%s | attempt=%d | strict_json=%s | size=%d",
                                self.mode or "unknown",
                                attempt + 1,
                                strict_json,
                                len(content),
                            )
                        if content:
                            if self.mode == "groq":
                                logging.debug(
                                    "Groq replied (%d chars) | strict_json=%s | finish_reason=%s",
                                    len(content),
                                    strict_json,
                                    finish_reason,
                                )
                            return content, None
                    last_error_reason = "empty_response"
                    if strict_json and self.mode == "groq":
                        continue
            except Exception:
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)
                    logging.info(
                        "Retry em %.0fs (tentativa %d/%d)",
[... TRUNCADO: 4185 linhas no total, mostradas 127 ...]
```

- Multiplicador: `max_retries=3` × `strict_json_modes=[True, False]` (apenas quando `mode == 'groq'` e modelo suporta json mode) = **até 6 chamadas de API por prompt**.
- 429/rate_limited: aborta retries e retorna `("", "rate_limited")` (linhas 3474-3488) — nenhuma chamada extra, mas conta no throttler (`record_rate_limit`).
- Backoff exponencial `1s * 2**attempt` entre retries (linhas 3524-3530).
- `finish_reason == 'length'` (truncamento) não conta como falha: retorna o conteúdo truncado (linhas 3503-3510).
- Os `record_call`/`record_success` (linhas 3454-3460) usam `get_throttler()` sem kwargs — mesmo ponto sensível do 8.1.

## 8.6 Referências a "eventos-fluxo.json" (grep)


```text
events/event_saver.py:351  json_file_name = "eventos-fluxo.json"   ← runtime (snapshot_file)
scripts/audit_json_payload_costs.py:113  default="dados/eventos-fluxo.json"  (CLI de auditoria)
diagnostic_files/window_diagnostics/advanced_diagnostics.py:14
diagnostic_files/window_diagnostics/fix_duplicates_complete.py:166
diagnostic_files/window_diagnostics/window_diagnostics.py:16,116,117
scripts/structure/generate_updated_structure.py:352
```

- Apenas `events/event_saver.py:351` é runtime (sobrescrevível por `config.EVENT_SAVER_JSON_FILE`); as demais são ferramentas de diagnóstico/auditoria.
- Inconsistência docstring × código: `event_saver.py:340-341` afirma "v5.0.0: Migração completa para SQLite (JSON/JSONL removidos)", mas `event_saver.py:362-363` seta `write_json=True` e `write_jsonl=True` — o dual write continua ativo e os arquivos existem com 18 eventos idênticos ao DB.

## 8.7 FileHandler / logs modificados nos últimos 30 dias

- `common/logging_config.py:20` — `from logging.handlers import RotatingFileHandler`; `:109-116` cria `file_handler` com `log_file`, `max_bytes`, `backup_count`, encoding utf-8 e JSONFormatter.
- `main.py:215` — import; `:218-223` `issues_handler = RotatingFileHandler('logs/issues.log', maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')` nível WARNING.
- Últimos 30 dias (2026-07-08 → 2026-08-07): `dados/eventos_visuais.log` (07/08/2026 19:43:10, 530.239 bytes, 20.000 linhas) e `logs/issues.log` (07/08/2026 19:42:07, 10.918 bytes).
- `logs/run.log` **NÃO modificado nos últimos 30 dias** (conteúdo parado em 2025-12-22) — mesmo com bot ativo em 07/08.
- Inconsistência: `logs/issues.log` está sendo gravado hoje (19:42) mas é ilegível em UTF-8 (padrão binário/duplicado) — provável conflito de escrita entre os dois sistemas de logging (`configure_logging` do common/logging_config com JSONFormatter + `setup_logging` do main.py) ou encoding inconsistente; e o log principal de execução (`run.log`) não recebe mais registros, indicando que o runtime atual não usa o `log_file` padrão.

# SEÇÃO 9 — FECHAMENTO DE CAUSA RAIZ


## 9.1 Corpo completo de get_throttler() (common/ai_throttler.py:338-358)


Singleton com cache em módulo: a PRIMEIRA chamada cria a instância (absorve kwargs); chamadas subsequentes **ignoram kwargs** e devolvem a mesma instância:


```python
# ──────────────────────────────────────────────
# Singleton
# ──────────────────────────────────────────────

_throttler_instance: Optional[SmartAIThrottler] = None


def get_throttler(**kwargs) -> SmartAIThrottler:
    """Retorna instância singleton do throttler."""
    global _throttler_instance
    if _throttler_instance is None:
        _throttler_instance = SmartAIThrottler(**kwargs)
        logger.info(
            "AI Throttler inicializado: interval=%.0fs, budget=%s, max/hour=%d",
            _throttler_instance.min_interval,
            f"{_throttler_instance.daily_token_budget:,}",
            _throttler_instance.max_calls_per_hour,
        )
    return _throttler_instance


def init_throttler(**kwargs) -> SmartAIThrottler:
    """
    Garante o singleton inicializado com a configuração desejada.

[... TRUNCADO: 375 linhas no total, mostradas 25 ...]
```


Valores default (linhas 51-63) quando chamado SEM argumentos:


```python
    # --- Intervalos ---
    min_interval: float = 180.0          # soft min (s) — pode ser bypassed
    hard_min_interval: float = 60.0      # hard min (s) — nunca bypassed
    significant_imb_change: float = 0.5  # threshold de mudança de imbalance

    # --- Budget (ajustado para Groq free tier) ---
    daily_token_budget: int = 50_000     # Conservador para free tier
    tokens_per_call_estimate: int = 2_500
    max_calls_per_hour: int = 6          # Era 10 — reduzido para evitar 429

    # --- Cooldown 429 ---
    base_cooldown_429: float = 120.0     # 2 min base
    max_cooldown_429: float = 1800.0     # 30 min max
```

- `get_throttler()` sem kwargs → defaults 180s/60s/50k tokens/6 chamadas por hora.
- `get_throttler(min_interval=60, hard_min_interval=30, daily_token_budget=85_000, max_calls_per_hour=10)` (ai_runner.py:41) só tem efeito se for a **primeira** chamada no processo; quem chamar primeiro vence (dependência de ordem de import).
- `reset_throttler()` (linhas 355-358) zera o singleton (uso em testes).

## 9.2 grep write_json (events/event_saver.py)


Saída do grep (todas as ocorrências):


```text
362: self.write_json = True
363: self.write_jsonl = True
368: self.write_json = bool(
369:   getattr(config, "EVENT_SAVER_WRITE_JSON", True)
371: self.write_jsonl = bool(
372:   getattr(config, "EVENT_SAVER_WRITE_JSONL", True)
783: if not self.write_json:      ← guard em _save_to_json()
906: if not self.write_jsonl:     ← guard em _save_to_jsonl()
1068: if self.write_json:         ← flush visual: dispara _save_to_json()
1069:     self._save_to_json(cleaned_event)
1070: if self.write_jsonl:
1071:     self._save_to_jsonl(cleaned_event)
```


Flags (linhas 360-373):


```python
        self.snapshot_file = DATA_DIR / json_file_name
        self.history_file = DATA_DIR / jsonl_file_name
        self.write_json = False  # default: JSON snapshot desativado (SQLite é a fonte de verdade)
        self.write_jsonl = True
        self.max_json_events = 1000
        self.max_json_file_size = MAX_JSON_FILE_SIZE
        self.max_jsonl_bytes = MAX_JSONL_BYTES
        if config is not None:
            self.write_json = bool(
                getattr(config, "EVENT_SAVER_WRITE_JSON", True)
            )
            self.write_jsonl = bool(
                getattr(config, "EVENT_SAVER_WRITE_JSONL", True)
            )
            try:
[... TRUNCADO: 1777 linhas no total, mostradas 15 ...]
```


Uso na escrita do .json (linhas 781-784 e 1066-1071):


```python
    def _save_to_json(self, event: Dict) -> None:
        """Salva evento em arquivo JSON (snapshot) com lock e retry."""
        if not self.write_json:
            return
        ...
                cleaned_event = self._clean_event_data(event)
                if cleaned_event:
                    if self.write_json:
                        self._save_to_json(cleaned_event)
                    if self.write_jsonl:
                        self._save_to_jsonl(cleaned_event)
```


Causa raiz: `write_json=True` por default (linha 362) e NÃO existe `EVENT_SAVER_WRITE_JSON=false` em config.json — o dual write (SQLite + eventos-fluxo.json) está ativo apesar da docstring v5.0.0 (linhas 340-341) afirmar 'JSON/JSONL removidos'.


## 9.3 grep issues.log — modo de escrita


grep em todo o repo (findstr recursivo + grep tool, 100% dos .py):


```text
main.py:219:        "logs/issues.log",
main.py:242: logging.info("🚨 Logs de problemas em: logs/issues.log")
```


Única escrita em arquivo (main.py:215-229) — RotatingFileHandler em modo **TEXTO 'a'** (append), encoding utf-8. NÃO existe nenhum `open('ab')`/`open('wb')`/handler binário para issues.log no código:


```python

    from logging.handlers import RotatingFileHandler

    # Sanear issues.log legado com BOM UTF-16 (ff fe): o RotatingFileHandler
    # anexa em utf-8, mas um arquivo que começa com BOM UTF-16 quebra a leitura.
    # Renomeia para backup antes de abrir, para o handler recriar em utf-8 puro.
    issues_log_path = os.path.join("logs", "issues.log")
    if os.path.exists(issues_log_path):
        try:
            with open(issues_log_path, "rb") as _f:
                _head = _f.read(2)
            if _head == b"\xff\xfe":
                _legacy_path = f"{issues_log_path}.legacy-{int(time.time())}"
                os.replace(issues_log_path, _legacy_path)
                logging.warning(
                    "issues.log com BOM UTF-16 legado renomeado para %s",
[... TRUNCADO: 435 linhas no total, mostradas 16 ...]
```


Evidência de bytes do arquivo (10.918 bytes):


```text
primeiros 64 bytes (hex): ff fe 32 30 32 36 2d 30 34 2d 32 38 20 31 32 3a 35 31 ...
                        ^^^^ BOM UTF-16 LE
bytes 0x00 no arquivo inteiro: 0  → corpo NÃO é UTF-16 real
newlines (0x0A): 74
início (ASCII):  ..2026-04-28 12:51:29,733 - WARNING - EventSimilarity - Nenhum evento encontrado para similarity search...
fim   (ASCII):  ..2026-08-07 19:42:07,704 - WARNING - root - ... INVARIANTE VIOLADA (Delta): Buy (2.7462) - Sell (8.3588) = -5.6126 != Delta Armazenado (0.0000) [diff=-5.6126] em BTCUSDT..2026-08-07 19:42:07,799 - WARNING - root - ... Delta CORRIGIDO: 0.0000 -> -5.6126 (fonte: vol_buy - vol_sell)..
```

- Causa raiz: o arquivo nasceu em 28/04/2026 com **BOM UTF-16 LE (ff fe) + corpo ASCII de 1 byte** (provavelmente de ferramenta antiga que prefixou BOM, ex.: Out-File/encoding misto) — formato híbrido que quebra leitura UTF-8 (BOM inválido) e UTF-16 (pares de ASCII viram lixo).
- O RotatingFileHandler atual (utf-8, append) escreve registros NOVOS no fim (03/08/2026 19:42:07 — 2 WARNINGs reais: 'INVARIANTE VIOLADA (Delta)' e 'Delta CORRIGIDO'), legíveis em ASCII no tail.
- Conclusão: a 'corrupção' não vem de escrita binária — vem do cabeçalho antigo BOM+ASCII; recomenda-se apagar/rotacionar o arquivo para que o handler recrie em utf-8 puro (a rotação 5 MB/backup 3 só dispara ao exceder 5 MB).

## 9.4 grep filename= / RotatingFileHandler( — outros arquivos de log


Resultados em main.py, common/logging_config.py e monitoring/*.py:


```text
main.py:218:  issues_handler = RotatingFileHandler(
common\logging_config.py:109:  file_handler = RotatingFileHandler(
monitoring\*.py:  (0 resultados — nenhum handler de arquivo)
```


common/logging_config.py:109 (handler genérico recebe `log_file` — default **None**):


```python
    if mode == "production":
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_DEV_FORMAT, _DEV_DATE_FORMAT))

    root.addHandler(handler)

    if log_file:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
[... TRUNCADO: 125 linhas no total, mostradas 16 ...]
```


Quem chama `setup_logging(...)` com `log_file`: **ninguém** no runtime (grep de `setup_logging|log_file=` só acha o docstring `common/logging_config.py:9` sugerindo `logs/bot.log` — exemplo, não executado; e o `setup_logging` próprio do flow_analyzer/logging_config.py:192-213 que não usa RotatingFileHandler).

- Causa raiz do run.log parado: **nenhum call-site passa log_file** — o único arquivo de log do logging module é `logs/issues.log` (WARNING+) criado por main.py; `run.log` é residual de 2025-12-22 e não recebe saída há meses.
- A saída que run.log 'deveria receber' (INFO/DEBUG) vai hoje para console + issues.log (só WARNING+) + eventos_visuais.log (via EventSaver, fora do logging) + payload_metrics.jsonl.

## 9.5 Validação v2 e fallback: qual campo falha e o que acontece


NOTA: o arquivo `ai_analyzer_qwen.py` não existe no repo (só `legacy/ai_analyzer_disabled.py` e `legacy/ai_analyzer_qwen_patch2.py`); o runtime real é `market_orchestrator/ai/analyzer_qwen.py`. Os greps pedidos rodaram nos arquivos que existem — legacy: 0 matches.


grep `no_safe_candidate|fallback_v1|ValidationError` (market_orchestrator/):


```text
ai_payload_builder.py:1223:  _append_payload_metric({"fallback_v1": True, "payload_bytes": v1_bytes, "error": str(e)})
llm_payload_guardrail.py:349:  "reason=no_safe_candidate_found",
llm_payload_guardrail.py:354:  True, bytes_before, 0, "event", "no_safe_candidate"
payload_metrics_aggregator.py:134:  if rec.get("error") == "no_safe_candidate":
payload_metrics_aggregator.py:136:  if rec.get("fallback_v1") is True:
```


A validação v2 é `_validate_payload_v2()` (market_orchestrator/ai/ai_payload_builder.py:997-1048) — 4 verificações em ORDEM, a 4ª é a que falhou 29x no metrics:


```python
        # ── Verificação 1: É um dicionário? ─────────────────────────
        if not isinstance(payload, dict):
            raise ValueError(
                f"payload_v2 inválido: esperado dict, "
                f"recebi {type(payload).__name__}"
            )
        
        # ── Verificação 2: Symbol presente? ─────────────────────────
        if not payload.get("symbol"):
            raise ValueError(
                "payload_v2 inválido: campo 'symbol' é obrigatório "
                "e não pode ser vazio"
            )
        
        # ── Verificação 3: epoch_ms é int válido? ───────────────────
        # CRÍTICO: antes aceitava STRING como truthy - bug silencioso!
        epoch = payload.get("epoch_ms")
        
        if epoch is None:
            raise ValueError(
                "payload_v2 inválido: 'epoch_ms' é None - "
                "timestamp obrigatório para rastreabilidade"
            )
        
        if not isinstance(epoch, (int, float)):
            raise ValueError(
                f"payload_v2 inválido: 'epoch_ms' deve ser int ou float, "
                f"recebi type={type(epoch).__name__} value='{epoch}' - "
                f"STRING de timestamp não é aceita aqui"
            )
        
        if int(epoch) < 1_000_000_000_000:
            raise ValueError(
                f"payload_v2 inválido: 'epoch_ms' valor suspeito={epoch} - "
                f"esperado timestamp em milissegundos (> 1_000_000_000_000)"
            )
        
        # ── Verificação 4: Preço atual presente? ────────────────────
        price_ctx = payload.get("price_context") or {}
        if price_ctx.get("current_price") is None:
            raise ValueError(
                "payload_v2 inválido: price_context.current_price é obrigatório - "
                "o LLM não pode analisar sem preço atual"
            )
        
[... TRUNCADO: 1398 linhas no total, mostradas 45 ...]
```


O que acontece no fallback (ai_payload_builder.py:1176-1223): a validação roda DENTRO do build do payload, ANTES da chamada à IA. No `except`, o payload permanece o **v1** (construído antes) e o fluxo segue — **NÃO há nova chamada à IA**: a única chamada ao LLM (feita depois, no analyzer) usa o payload v1 maior. O `fallback_v1:true` é só um marcador de métrica:


```python
    if v2_enabled:
        try:
            # ── CORREÇÃO: Capturar current_price ANTES da compressão ──
            # O compressor pode descartar price_context durante otimização;
            # aqui garantimos que o preço atual sempre seja preservado.
            _v1_price_ctx = ai_payload.get("price_context") or {}
            _current_price_safe = (
                _v1_price_ctx.get("current_price")
                if isinstance(_v1_price_ctx, dict)
                else None
            )

            payload_v2 = compress_payload(ai_payload, max_bytes=max_bytes)
            
            # ── CORREÇÃO: Restaurar epoch_ms se compressor descartou ──
            # O compressor pode eliminar epoch_ms durante otimização de bytes
            # Aqui garantimos que o valor original sempre seja preservado
            if payload_v2.get("epoch_ms") is None:
                payload_v2["epoch_ms"] = _epoch_ms_safe
                logging.warning(
                    "EPOCH_MS_RESTORED_AFTER_COMPRESSION "
                    "restored_value=%s symbol=%s",
                    _epoch_ms_safe,
                    symbol
                )
            # Verifica se epoch_ms é int válido após compressão
            elif not isinstance(payload_v2.get("epoch_ms"), (int, float)):
                logging.warning(
                    "EPOCH_MS_WRONG_TYPE_AFTER_COMPRESSION "
                    "type=%s value=%s restoring=%s",
                    type(payload_v2.get("epoch_ms")).__name__,
                    payload_v2.get("epoch_ms"),
                    _epoch_ms_safe
                )
                payload_v2["epoch_ms"] = _epoch_ms_safe
            
            # Restaurar current_price se o compressor descartou o price_context
            # (mesmo padrão do EPOCH_MS_RESTORED acima)
            if _current_price_safe is not None:
                _v2_price_ctx = payload_v2.get("price_context")
                if (
                    not isinstance(_v2_price_ctx, dict)
                    or _v2_price_ctx.get("current_price") is None
                ):
                    if not isinstance(_v2_price_ctx, dict):
                        _v2_price_ctx = {}
                    _v2_price_ctx["current_price"] = _current_price_safe
                    payload_v2["price_context"] = _v2_price_ctx
[... TRUNCADO: 1398 linhas no total, mostradas 48 ...]
```


Validação da RESPOSTA (camada separada, market_orchestrator/ai/llm_response_validator.py:541-587): JSON inválido/estrutura errada → devolve `FALLBACK_RESPONSE.copy()` com `is_fallback=True` — também **sem re-chamada** dentro do validador; re-tentativas de chamada só ocorrem dentro de `_call_openai_compatible` (max_retries=3 × strict_json_modes, Seção 8.5), ANTES de chegar aqui:


```python
    valid, error_reason = validate_json_structure(data)
    
    if not valid:
        preview = sanitize_for_log(response, 150)
        if log_errors:
            logger.warning(f"ai_response_invalid: {error_reason} | preview={preview}")
        return ValidationResult(
            valid=False,
            parsed=FALLBACK_RESPONSE.copy(),
            error_reason=error_reason,
            is_fallback=True,
            raw_preview=preview,
        )
    
    # Sucesso - normaliza campos opcionais
    result = dict(data)
    
    for field in OPTIONAL_STRING_FIELDS:
        if field not in result:
            result[field] = None
    
    # Log de sucesso (sem dados sensíveis)
    logger.info(f"ai_response_valid: action={result.get('action')} confidence={result.get('confidence')}")
    
    return ValidationResult(
        valid=True,
        parsed=result,
        error_reason=None,
        is_fallback=False,
        raw_preview=None,
[... TRUNCADO: 654 linhas no total, mostradas 30 ...]
```


Em analyzer_qwen.py:3904-3909, resposta inválida vira `_build_structured_fallback(validation_error)` — resposta estruturada padrão (action=wait), sem gasto adicional:


```python
                is_fallback = not is_valid
                if is_fallback:
                    validation_error = str(
                        structured_out.get("_fallback_reason") or "validation_failed"
                    )

            if is_fallback and (
                not isinstance(structured_out, dict) or not structured_out.get("_is_fallback")
            ):
                structured_out = self._build_structured_fallback(
                    validation_error or "validation_failed"
                )
[... TRUNCADO: 4185 linhas no total, mostradas 12 ...]
```

- Campo que falha (29 casos no metrics): `price_context.current_price` ausente após `compress_payload` — o compressor pode descartar o preço durante otimização; o código só restaura `epoch_ms` (linhas 1180-1200), **não restaura current_price**.
- Fallback v1: **reusa o payload v1 já construído** (mesmo fluxo, sem nova chamada); custo extra = payload v1 maior (mais tokens na chamada única) + marcador fallback_v1 no metrics.
- Guardrail `no_safe_candidate` (76 casos): aborta a análise ANTES do prompt (analyzer_qwen.py:3644 → fallback `unsafe_payload`), sem chamada à IA.
- Nenhuma re-chamada à IA acontece por falha de validação; o único multiplicador de chamadas é o retry de rede dentro de `_call_openai_compatible`.

# SEÇÃO 10 — INVESTIGAÇÃO DELTA E VALIDAÇÃO FINAL


## 10.1 — INVESTIGAÇÃO COMPLETA DO "INVARIANTE VIOLADA (Delta)" [PRIORIDADE MÁXIMA]


### 10.1a) Função COMPLETA _validate_invariants (data_pipeline/pipeline.py:629-704) + contexto


Contexto (600-628) — fim do bloco _attach_advanced_analysis; a função seguinte é chamada nos call sites abaixo. Função completa:


```python
                raw_event = updated_event.get("raw_event", {})
                inner_raw = raw_event.get("raw_event", {})
                has_advanced = (
                    "advanced_analysis" in inner_raw
                    or "advanced_analysis" in raw_event
                )
                if has_advanced:
                    advanced = inner_raw.get("advanced_analysis", raw_event.get("advanced_analysis", {}))
                    self.logger.runtime_info(
                        f"✅ advanced_analysis adicionado com sucesso - "
                        f"keys={list(advanced.keys()) if isinstance(advanced, dict) else 'N/A'}"
                    )
                else:
                    self.logger.runtime_warning(
                        "⚠️ advanced_analysis NÃO foi adicionado ao raw_event"
                    )
            else:
                self.logger.runtime_warning(
                    "⚠️ enrich_event_with_advanced_analysis não retornou dados válidos"
                )
        except Exception as e:
            self.logger.runtime_error(
                f"❌ Erro ao chamar enrich_event_with_advanced_analysis: {e}"
            )

    # ============================
    # VALIDATE INVARIANTS
    # ============================

    def _validate_invariants(self, data: Dict[str, Any], context: str = "enrich") -> None:
        """
        Valida invariantes.

        Contexto 'enrich': Valida OHLC básico.
        Contexto 'signal': Valida consistência de volumes de compra/venda e delta.
        """
        try:
            TOLERANCE_VOL = 1e-4    # 0.0001 BTC — arredondamento float
            TOLERANCE_DELTA = 1e-2  # 0.01 BTC — delta acumula erros

            if context == "enrich":
                # Validação OHLC
                ohlc = data.get("ohlc", {})
                if ohlc:
                    h = float(ohlc.get("high", 0))
                    l = float(ohlc.get("low", 0))
                    c = float(ohlc.get("close", 0))
                    o = float(ohlc.get("open", 0))

                    if l > h:
                        logging.warning(f"⚠️ INVARIANTE VIOLADA (OHLC): Low ({l}) > High ({h}) em {self.symbol}")

                    if not (l <= c <= h) or not (l <= o <= h):
                        logging.warning(f"⚠️ INVARIANTE VIOLADA (OHLC): Open/Close fora dos limites High/Low")

            elif context == "signal":
                # Validação de Volumes de Sinal (onde temos a quebra Buy/Sell)
                vol_buy = float(data.get("volume_compra", 0.0))
                vol_sell = float(data.get("volume_venda", 0.0))
                vol_total = float(data.get("volume_total", 0.0))
                delta = float(data.get("delta", 0.0))

                # 1. Soma dos volumes
                vol_sum = vol_buy + vol_sell
                vol_diff = abs(vol_sum - vol_total)

                if vol_diff > TOLERANCE_VOL:
                    if vol_diff > 0.01:
                        logging.warning(
                            f"⚠️ INVARIANTE VIOLADA (Vol Sum): "
                            f"Buy + Sell ({vol_sum:.8f}) != Total ({vol_total:.8f}) "
                            f"diff={vol_diff:.8f} em {self.symbol}"
                        )
                    else:
                        logging.debug(
                            f"Volume ajustado por arredondamento: "
                            f"diff={vol_diff:.8f} BTC em {self.symbol}"
                        )

                # 2. Consistência do Delta — recalcular e corrigir automaticamente
                delta_calc = vol_buy - vol_sell
                if abs(delta_calc - delta) > TOLERANCE_DELTA:
                    delta_diff = delta_calc - delta
                    logging.warning(
                        f"⚠️ INVARIANTE VIOLADA (Delta): "
                        f"Buy ({vol_buy:.4f}) - Sell ({vol_sell:.4f}) = {delta_calc:.4f} != Delta Armazenado ({delta:.4f}) "
                        f"[diff={delta_diff:+.4f}] em {self.symbol}"
                    )
                    # Correção automática com validação
                    if abs(delta_calc) > 0.0001:
                        logging.warning(
                            f"✅ Delta CORRIGIDO: {delta:.4f} -> {delta_calc:.4f} "
                            f"(fonte: vol_buy - vol_sell)"
                        )
                        data["delta"] = delta_calc
                        if hasattr(self, 'enriched_data') and self.enriched_data:
                            self.enriched_data["delta_fechamento"] = delta_calc
                    else:
                        logging.warning(
                            f"⚠️ Delta calculado muito próximo de zero ({delta_calc:.4f}), "
                            f"mantendo valor armazenado: {delta:.4f}"
                        )
        except Exception as e:
            logging.debug(f"Erro na validação de invariantes Pipeline ({context}): {e}", exc_info=True)

[... TRUNCADO: 847 linhas no total, mostradas 105 ...]
```


Call sites de _validate_invariants (onde os dados chegam e a correção se aplica):


```text
343: self._validate_invariants(self.enriched_data, context="enrich")
492: self._validate_invariants(absorption_event, context="signal")
511: self._validate_invariants(exhaustion_event, context="signal")
523: self._validate_invariants(ob_event, context="signal")
562: self._validate_invariants(analysis_trigger, context="signal")
```


### 10.1b) grep delta no pipeline.py


```text
638: TOLERANCE_DELTA = 1e-2  # 0.01 BTC - delta acumula erros
660: delta = float(data.get("delta", 0.0))
680: delta_calc = vol_buy - vol_sell
681: if abs(delta_calc - delta) > TOLERANCE_DELTA:
682: delta_diff = delta_calc - delta
685: f"Buy ({vol_buy:.4f}) - Sell ({vol_sell:.4f}) = {delta_calc:.4f} != Delta Armazenado ({delta:.4f}) "
689: if abs(delta_calc) > 0.0001:
691: f"? Delta CORRIGIDO: {delta:.4f} -> {delta_calc:.4f} "
694: data["delta"] = delta_calc
696: self.enriched_data["delta_fechamento"] = delta_calc
699: f"?? Delta calculado muito proximo de zero ({delta_calc:.4f}), "
700: f"mantendo valor armazenado: {delta:.4f}
```


### 10.1c) Origem do "delta" ANTES da validação


1) Atribuição autoritativa do raw_event (data_pipeline/pipeline.py:530-545 — FIX 15):


```python
        # Evento de análise (sempre gerado)
        try:
            # Dados para o raw_event - INCLUIR DADOS CONTEXTUAIS COMPLETOS
            raw_event_data = {
                # Dados básicos do enriched
                "volume_total": self.enriched_data.get("volume_total", 0),
                "volume_compra": self.enriched_data.get("volume_compra", 0),
                "volume_venda": self.enriched_data.get("volume_venda", 0),
                # FIX 15: delta SEMPRE calculado a partir dos volumes autoritativos.
                # Antes: usava delta_fechamento (intra-candle), que pode ser 0.0
                # como default quando a computação falha — causando INVARIANTE
                # VIOLADA quando volumes são non-zero.
                "delta": (
                    float(self.enriched_data.get("volume_compra", 0))
                    - float(self.enriched_data.get("volume_venda", 0))
                ),
                "preco_fechamento": self.enriched_data.get("ohlc", {}).get("close", 0),
                "advanced_analysis": (
                    self.contextual_data.get("advanced_analysis", {})
                    if self.contextual_data
                    else self.enriched_data.get("advanced_analysis", {})
                ),
                # Dados contextuais necessários para enrich_event_with_advanced_analysis
                "multi_tf": self.contextual_data.get("multi_tf", {}) if self.contextual_data else {},
                "historical_vp": self.contextual_data.get("historical_vp", {}) if self.contextual_data else {},
                "liquidity_heatmap": self.contextual_data.get("liquidity_heatmap", {}) if self.contextual_data else {},
                "flow_metrics": self.contextual_data.get("flow_metrics", {}) if self.contextual_data else {},
                "orderbook_data": self.contextual_data.get("orderbook_data", {}) if self.contextual_data else {},
                "timestamp_utc": self.enriched_data.get("ohlc", {}).get("close_time"),
            }
            analysis_trigger = build_analysis_trigger_event(self.symbol, raw_event_data)
            analysis_trigger["epoch_ms"] = default_ts_ms
            self._validate_invariants(analysis_trigger, context="signal")

[... TRUNCADO: 847 linhas no total, mostradas 34 ...]
```


2) Propagação para o top-level do signal (data_processing/enrichment_integrator.py:37-49 — build_analysis_trigger_event):


```python
    event = {
        "is_signal": True,
        "tipo_evento": "ANALYSIS_TRIGGER",
        "descricao": "Evento automático para análise da IA",
        "symbol": symbol,
        "raw_event": raw_event,
        "resultado_da_batalha": "N/A",
        # Propagar delta e volumes para o nível top-level do sinal
        "delta": raw_event.get("delta", 0.0),
        "volume_total": raw_event.get("volume_total", 0.0),
        "volume_compra": raw_event.get("volume_compra", 0.0),
        "volume_venda": raw_event.get("volume_venda", 0.0),
    }
    
    # Adicionar preco_fechamento se disponível
[... TRUNCADO: 130 linhas no total, mostradas 15 ...]
```


3) Cálculo paralelo por barra de trades (institutional/cvd.py:141-144, 157-167, 191-193 — acumula vol. por lado e deriva delta):


```python
        # Acumular volume por lado
        if trade.side == Side.BUY:
            self._current_buy_vol += trade.quantity
        elif trade.side == Side.SELL:
            self._current_sell_vol += trade.quantity

        return completed_bar

    def process_trades(self, trades: list[Trade]) -> list[CVDBar]:
        """Processa lista de trades, retorna barras completadas."""
        completed: list[CVDBar] = []
        for trade in trades:
            bar = self.process_trade(trade)
            if bar is not None:
                completed.append(bar)
        return completed

    def _close_bar(self, timestamp: float) -> CVDBar:
        """Fecha a barra atual e inicia nova."""
        delta = self._current_buy_vol - self._current_sell_vol
        self.cumulative_delta += delta

        bar = CVDBar(
            timestamp=self._current_bar_start,
            buy_volume=self._current_buy_vol,
            sell_volume=self._current_sell_vol,
            delta=delta,
            cumulative_delta=self.cumulative_delta,
            price_open=self._current_price_open,
            price_close=self._current_price_close,
            price_high=self._current_price_high,
            price_low=self._current_price_low,
            trade_count=self._current_trade_count,
        )

[... TRUNCADO: 356 linhas no total, mostradas 35 ...]
```


### 10.1d) Frequência do evento (logs/issues.log — 75 linhas, híbrido BOM+ASCII)


```text
INVARIANTE VIOLADA: 6 ocorrências
Delta CORRIGIDO:    6 ocorrências (sempre em par com INVARIANTE)

Timestamps (todas as ocorrências, ordem cronológica):
2026-04-28 21:55:08,586   INVARIANTE + CORRIGIDO
2026-05-06 22:22:10,785   INVARIANTE + CORRIGIDO
2026-08-04 17:00:11,487   INVARIANTE + CORRIGIDO
2026-08-04 17:29:05,397   INVARIANTE + CORRIGIDO
2026-08-04 17:32:05,373   INVARIANTE + CORRIGIDO
2026-08-07 19:42:07,704   INVARIANTE + CORRIGIDO (19:42:07,799)
```


Padrão: ESPORÁDICO em rajadas — 3 ocorrências em 04/08 (17:00, 17:29, 17:32) e 1 em 07/08 (19:42). Não é constante nem por minuto; ocorre quando um evento carrega delta armazenado divergente dos volumes (ex.: delta_fechamento intra-candle que deu 0.0 — ver FIX 15).


### 10.1e) delta_calc é RECALCULADO DO ZERO (não derivado do delta errado)


Resposta objetiva: RECALCULADO DO ZERO. `delta_calc = vol_buy - vol_sell` (linha 680) usa `volume_compra`/`volume_venda` do MESMO dict que chegou à validação (lidos nas linhas 657-658) — não aplica patch incremental sobre o `delta` armazenado. O trecho comprovante (655-701):


```python
            elif context == "signal":
                # Validação de Volumes de Sinal (onde temos a quebra Buy/Sell)
                vol_buy = float(data.get("volume_compra", 0.0))
                vol_sell = float(data.get("volume_venda", 0.0))
                vol_total = float(data.get("volume_total", 0.0))
                delta = float(data.get("delta", 0.0))

                # 1. Soma dos volumes
                vol_sum = vol_buy + vol_sell
                vol_diff = abs(vol_sum - vol_total)

                if vol_diff > TOLERANCE_VOL:
                    if vol_diff > 0.01:
                        logging.warning(
                            f"⚠️ INVARIANTE VIOLADA (Vol Sum): "
                            f"Buy + Sell ({vol_sum:.8f}) != Total ({vol_total:.8f}) "
                            f"diff={vol_diff:.8f} em {self.symbol}"
                        )
                    else:
                        logging.debug(
                            f"Volume ajustado por arredondamento: "
                            f"diff={vol_diff:.8f} BTC em {self.symbol}"
                        )

                # 2. Consistência do Delta — recalcular e corrigir automaticamente
                delta_calc = vol_buy - vol_sell
                if abs(delta_calc - delta) > TOLERANCE_DELTA:
                    delta_diff = delta_calc - delta
                    logging.warning(
                        f"⚠️ INVARIANTE VIOLADA (Delta): "
                        f"Buy ({vol_buy:.4f}) - Sell ({vol_sell:.4f}) = {delta_calc:.4f} != Delta Armazenado ({delta:.4f}) "
                        f"[diff={delta_diff:+.4f}] em {self.symbol}"
                    )
                    # Correção automática com validação
                    if abs(delta_calc) > 0.0001:
                        logging.warning(
                            f"✅ Delta CORRIGIDO: {delta:.4f} -> {delta_calc:.4f} "
                            f"(fonte: vol_buy - vol_sell)"
                        )
                        data["delta"] = delta_calc
                        if hasattr(self, 'enriched_data') and self.enriched_data:
                            self.enriched_data["delta_fechamento"] = delta_calc
                    else:
                        logging.warning(
                            f"⚠️ Delta calculado muito próximo de zero ({delta_calc:.4f}), "
                            f"mantendo valor armazenado: {delta:.4f}"
                        )
[... TRUNCADO: 847 linhas no total, mostradas 47 ...]
```


### 10.1f) O delta corrigido alimenta o payload da IA? — SIM


grep `"delta"` nos produtores de payload:


```text
market_orchestrator/ai/ai_payload_builder.py:871:  ai_payload["delta"] = signal.get("delta")
flow_analyzer/core.py:1037:                 "delta": float(data['delta'])
institutional/cvd.py:159-167:  delta = self._current_buy_vol - self._current_sell_vol (CVDBar)
```

- Caminho completo: pipeline.py:542 (FIX 15) monta `raw_event_data['delta']` → enrichment_integrator.py:45 propaga para `analysis_trigger['delta']` → `_validate_invariants(analysis_trigger, context='signal')` (pipeline.py:562) corrige `data['delta'] = delta_calc` (linha 694) → `ai_payload_builder.py:871` copia `signal.get('delta')` para `ai_payload['delta']` → LLM.
- A correção em runtime (linha 694) acontece ANTES do evento seguir para o build do payload, então o payload da IA recebe o valor já corrigido.

## 10.2 — CONFIRMAÇÃO DA ARQUITETURA FINAL DO THROTTLER


### 10.2a) Todas as ocorrências de init_throttler/get_throttler


```text
common/ai_throttler.py:8       - Singleton via get_throttler()          (docstring)
common/ai_throttler.py:341    def get_throttler(**kwargs) -> SmartAIThrottler:
common/ai_throttler.py:355    def init_throttler(**kwargs) -> SmartAIThrottler:
common/ai_throttler.py:364        return get_throttler(**kwargs)
market_orchestrator/ai/analyzer_qwen.py:3455  from common.ai_throttler import init_throttler
market_orchestrator/ai/analyzer_qwen.py:3456  _throttler = init_throttler(          ← record_call/record_success
market_orchestrator/ai/analyzer_qwen.py:3480  from common.ai_throttler import init_throttler
market_orchestrator/ai/analyzer_qwen.py:3481  _throttler = init_throttler(          ← record_rate_limit
market_orchestrator/ai/ai_runner.py:40-41     from common.ai_throttler import get_throttler
                                              _ai_throttler = get_throttler(       ← import-time, 60/30/85k/10
tests/unit/test_ai_throttler_v3.py:11-12      imports; 318-360 testes singleton/init/reset
```


### 10.2b) Código atual (pós-patch) de init_throttler


```python
_throttler_instance: Optional[SmartAIThrottler] = None


def get_throttler(**kwargs) -> SmartAIThrottler:
    """Retorna instância singleton do throttler."""
    global _throttler_instance
    if _throttler_instance is None:
        _throttler_instance = SmartAIThrottler(**kwargs)
        logger.info(
            "AI Throttler inicializado: interval=%.0fs, budget=%s, max/hour=%d",
            _throttler_instance.min_interval,
            f"{_throttler_instance.daily_token_budget:,}",
            _throttler_instance.max_calls_per_hour,
        )
    return _throttler_instance


def init_throttler(**kwargs) -> SmartAIThrottler:
    """
    Garante o singleton inicializado com a configuração desejada.

    Idempotente: se o singleton já existe (quem chamou primeiro vence),
    os kwargs são ignorados e a instância existente é devolvida.
    Use nos callers que não passam por ai_runner para garantir
    os parâmetros pretendidos quando o throttler roda sozinho.
    """
    return get_throttler(**kwargs)


def reset_throttler_for_tests() -> None:
    """Alias explícito de reset_throttler para uso em fixtures de teste."""
    reset_throttler()


def reset_throttler():
[... TRUNCADO: 375 linhas no total, mostradas 35 ...]
```


Comportamento: idempotente silencioso — se o singleton já existe, os kwargs são ignorados e a instância existente é devolvida (linha 352). **Não lança RuntimeError** em 2ª chamada com valores diferentes, **nem loga WARNING/INFO** sobre qual chamada venceu. O único log existe na CRIAÇÃO (linhas 346-351, INFO).


### 10.2c) main.py chama init_throttler? — NÃO


O grep completo acima não contém nenhuma ocorrência em main.py. O main.py delega o throttler para o ai_runner (import-time, get_throttler com 60/30/85k/10) e para o analyzer_qwen (runtime, init_throttler com os mesmos valores).


### 10.2d) ai_runner.py:41 (pós-patch) — inalterado, ainda com argumentos


```python
try:
    from common.ai_throttler import get_throttler
    _ai_throttler = get_throttler(
        min_interval=60,
        hard_min_interval=30,
        daily_token_budget=85_000,
        max_calls_per_hour=10,
    )
except ImportError:
    _ai_throttler = None
[... TRUNCADO: 816 linhas no total, mostradas 10 ...]
```


### 10.2e) Múltiplas chamadas com MESMOS valores → log de quem venceu?


Totalmente SILENCIOSO. O `init_throttler` → `get_throttler` retorna direto na linha 352 quando `_throttler_instance` já existe — sem nenhum logging. O INFO 'AI Throttler inicializado' (346-351) só aparece na primeira criação. Ou seja: ai_runner (import) e analyzer_qwen (runtime) chamam com os mesmos 60/30/85k/10, mas o código não emite nenhum log informando qual chamada realmente criou o singleton.


## 10.3 — VALIDAÇÃO DE REGRESSÃO NO TESTE DE FALLBACK DE PROVIDER


Sequência executada (patches NÃO-commitados → git stash guardou os 5 patches):


```text
$ git stash
  Saved working directory and index state WIP on main: e8858c5 refactor(arch): eliminate src/ proxy layer...

$ pytest tests/integration/test_patch_2_fallback_controlado.py -v   (com os patches REMOVIDOS)
  FAILED ...::test_patch_2_groq_fail_com_fallback_openai     - AssertionError: None != 'openai' : Modo deve ser 'openai' (fallback)
  FAILED ...::test_patch_2_multiple_fallbacks                 - AssertionError: 'dashscope' != 'openai' (primeiro fallback)
  FAILED ...::test_patch_2_provider_nao_groq_vai_para_openai  - AssertionError: None != 'openai' : Modo deve ser 'openai'
  ======================== 3 failed, 2 passed in 50.22s ========================

$ git stash pop   → "Dropped refs/stash@{0}" — 8 arquivos modificados restaurados (git status confirmado)
```


CONCLUSÃO: o teste falhava da MESMA forma ANTES dos patches (3 failed, 2 passed, mesmos AssertionError de 'Modo deve ser openai'). **Pré-existente — NÃO é regressão introduzida pelos 5 patches.** A causa apontada no log: `AsyncClient.__init__() got an unexpected keyword argument 'proxies'` impede o fallback para OpenAI.


## 10.4 — CONFIRMAÇÃO DO run.log FUNCIONANDO


Nenhum processo python em execução no momento da auditoria (bot NÃO reiniciado após o Patch 3) — `logs/run.log` ainda contém o conteúdo antigo (2025-12-22, sincronização de clock):


```log
2025-12-22 21:20:45,116 - INFO - 📊 Nível de log configurado: INFO
2025-12-22 21:20:45,116 - INFO - 🚀 Iniciando bot para BTCUSDT...
2025-12-22 21:20:45,124 - INFO - 📊 Servidor Prometheus iniciado na porta 8000 (/metrics)
2025-12-22 21:20:45,124 - INFO - ✅ AsyncTradeBuffer inicializado | Max size: 2000 | Backpressure: 80% | Batch size: 50
2025-12-22 21:20:45,124 - INFO - ================================================================================
2025-12-22 21:20:45,124 - INFO - 🕐 TIMEMANAGER v2.1.2 - INICIALIZANDO
2025-12-22 21:20:45,124 - INFO -    Tempo local:     1766449245124 ms
2025-12-22 21:20:45,124 - INFO -    Timezone UTC:    UTC
2025-12-22 21:20:45,124 - INFO -    Timezone NY:     America/New_York
2025-12-22 21:20:45,124 - INFO -    Timezone SP:     America/Sao_Paulo
2025-12-22 21:20:45,124 - INFO -    ZoneInfo:        ✅ Disponível
2025-12-22 21:20:45,124 - INFO -    Max offset:      600 ms
2025-12-22 21:20:45,124 - INFO -    Sync samples:    5
2025-12-22 21:20:45,124 - INFO -    Sync interval:   1800 s (30 min)
2025-12-22 21:20:45,125 - INFO - 🔄 Tentativa 1/3
2025-12-22 21:20:45,125 - INFO - 🔄 Iniciando sincronização com Binance (5 amostras)...
2025-12-22 21:20:49,145 - INFO - ✅ Melhor amostra selecionada: RTT=663ms, Offset=180ms
2025-12-22 21:20:49,145 - INFO - ✅ Offset dentro do limite aceitável: 180ms
2025-12-22 21:20:49,145 - INFO - ✅ Sincronização bem-sucedida na tentativa 1
2025-12-22 21:20:49,145 - INFO - 
2025-12-22 21:20:49,145 - INFO - 🔍 DIAGNÓSTICO DO SISTEMA DE TEMPO
2025-12-22 21:20:49,145 - INFO - --------------------------------------------------------------------------------
2025-12-22 21:20:49,145 - INFO - ⏰ Timestamps:
2025-12-22 21:20:49,145 - INFO -    Local time (ms):  1766449249145
2025-12-22 21:20:49,145 - INFO -    Synced time (ms): 1766449249325
2025-12-22 21:20:49,145 - INFO -    Current offset:   180 ms (0.18s)
2025-12-22 21:20:49,145 - INFO - 🌍 Timezones:
2025-12-22 21:20:49,146 - INFO -    UTC: 2025-12-23 00:20:49 UTC
2025-12-22 21:20:49,146 - INFO -    NY:  2025-12-22 19:20:49 EST
2025-12-22 21:20:49,146 - INFO -    SP:  2025-12-22 21:20:49 -03
2025-12-22 21:20:49,146 - INFO -    Offset NY vs UTC: -5.0 horas
2025-12-22 21:20:49,146 - INFO -    Offset SP vs UTC: -3.0 horas
2025-12-22 21:20:49,146 - INFO -    ✅ Offset NY correto: -5.0 horas
2025-12-22 21:20:49,146 - INFO -    ✅ Offset SP correto: -3.0 horas
2025-12-22 21:20:49,146 - INFO - 📊 Estatísticas de Sincronização:
2025-12-22 21:20:49,146 - INFO -    Status:                ok
2025-12-22 21:20:49,146 - INFO -    Server offset:         180 ms
2025-12-22 21:20:49,146 - INFO -    Last RTT:              663 ms
2025-12-22 21:20:49,146 - INFO -    Best RTT:              663 ms
2025-12-22 21:20:49,146 - INFO -    Sync attempts:         1
2025-12-22 21:20:49,146 - INFO -    Sync failures:         0
2025-12-22 21:20:49,146 - INFO -    Auto corrections:      0
2025-12-22 21:20:49,146 - INFO -    Success rate:          100.0%
2025-12-22 21:20:49,146 - INFO -    ZoneInfo available:    ✅ Yes
2025-12-22 21:20:49,146 - INFO - ================================================================================
```


O Patch 3 adiciona `RotatingFileHandler('logs/run.log', maxBytes=10MB, backupCount=5, encoding='utf-8')` em main.py (nível `log_level` = INFO por default, formato `%(asctime)s - %(levelname)s - %(name)s - %(message)s`). O handler só passa a escrever na PRÓXIMA inicialização do main.py; o formato esperado (timestamps + nível INFO) será o mesmo do issues.log (ex.: `2026-08-07 19:42:07,704 - WARNING - root - ...`), com INFO+ em vez de WARNING+.


# SEÇÃO 11 — VALIDAÇÃO AO VIVO 2026-08-07/08: FIXES APLICADOS E CONFIRMADOS

Seção adicionada manualmente após a auditoria original (sessão noturna de 2026-08-07 → 08). Documenta a causa raiz das falhas de rede, os achados de latência/direção de texto e os fixes aplicados + validados em produção (20+ janelas).

## 11.1 — CAUSA RAIZ: aiodns 4.0.0 QUEBRADO (falha de resolução DNS 100% no aiohttp)

### Diagnóstico
- Bot sem processar janelas desde 21:55 (restarts PIDs 2792, 13252, 12636, 1900; todos morriam com `ws_max_reconnect_reached` 15/15).
- Testes isolados: `nslookup` (8.8.8.8/dns.google) OK; `Test-NetConnection api.binance.com:443` e `stream.binance.com:9443` → True; `socket.getaddrinfo`, `loop.getaddrinfo` e `websockets` (wss://stream.binance.com:9443/ws/btcusdt@trade) → trades reais recebidos.
- **AIOHTTP 3.13.2 + aiodns 4.0.0 → `AsyncResolver` (c-ares) falha 8/8 com `aiodns.error.DNSError: (11, 'Could not contact DNS servers')`** → `ClientConnectorDNSError: Cannot connect to host ... [Could not contact DNS servers]` em TODO `ClientSession` do bot (robust_connection.py:189-199, context_collector.py:224-231, macro_data_provider.py:338, orderbook_analyzer/core.py:579-586, onchain_fetcher.py:55-56, fred_fetcher.py:326, websocket_handler.py:87).

### Fix aplicado
- `pip uninstall aiodns` (aprovado pelo usuário) → aiohttp volta ao resolver nativo (threading).
- Confirmação pós-fix: `aiohttp.get('https://api.binance.com/api/v3/ping')` → **200**; `ws_connect` wss stream.binance.com → **trade real recebido**.

### Observação secundária (não relacionada, pré-existente)
- `api.coingecko.com` apresenta **certificado SSL EXPIRADO** (`SSLCertVerificationError: certificate has expired`) — causa dos avisos recorrentes de BTC Dominance nos logs desde 21:44; externo ao bot, tratado como warning (bot coleta 8/9 e usa fallback de dominância via Binance).

## 11.2 — FIX DE LATÊNCIA (janelas de 13-20s → 6-9s)

### Causa raiz (achado do log)
- Janelas lentas (#1=13.21s, #6=20.14s, #11=20.16s) coincidiam com cache miss de `get_all_macro_data()` (TTL `all_macro` = 60s): 8 fetches SEQUENCIAIS com `_safe_fetch` timeout fixo de 25s (macro_data_provider.py:937-946), re-fetch a cada ~5 min via TTL do cache de correlações (cross_asset_correlations.py:698, `_CORR_CACHE_TTL` = 5 min), entrando por `common/ml_features.py:511` → `cross_asset_correlations.py:702` → `_run_async_safely(timeout=30)` (:569).

### Fixes aplicados (2 partes)
- **Parte 1 — paralelização**: `_safe_fetch` ganhou parâmetro `timeout: float = 25.0` (default preserva callers); os 8 awaits sequenciais viraram `asyncio.gather(..., timeout=8.0, return_exceptions=True)` com exceções normalizadas para `None` (macro_data_provider.py:971-1004). Boot real com cache miss: **7,0-8,0s** (antes ~8,9s+, sem teto claro).
- **Parte 2 — TTL + aquecimento**: `_ttl_config["all_macro"]` 60 → **900s** (macro_data_provider.py:113, alinhado ao `CROSS_ASSET_INTERVAL`); `MacroUpdateService._update_loop` passou a chamar `await provider.get_all_macro_data()` logo após `fetch_cross_asset_data()` (macro_update_service.py:234-240). 
- **Parte 3 — force no aquecimento (bug de timing encontrado na validação)**: sem `force`, o aquecimento do bloco de 900s pegava cache HIT (o cache do boot ainda válido por ~16s), NÃO renovava o timestamp e o cache expirava no meio do bloco → janela pegava miss ~2 min depois do aquecimento (observado 23:51:09 aquecido → miss 23:53:10). Fix: `get_all_macro_data(force: bool = False)` — o UpdateService passa `force=True` (macro_data_provider.py:957-972; macro_update_service.py:234-241).

### Validação em produção (run 9376, 20 janelas)
- Cache miss macro: **1×/15 min** (antes 1×/5 min), sempre ~7s, sempre no aquecimento.
- **Zero** cache miss no caminho crítico das janelas após o warm-up.
- Sequência confirmada: boot 23:59:14 aquecido → bloco 00:13:59 → força refetch 00:14:15 → janelas #16-#17 (00:15-00:16) SEM miss.
- Janelas: 5,9-9,0s; picos isolados 12,1s (janela #1, boot) e 13,8s (janela #6, 1.475 trades + correção de invariante Delta — sem logs de macro envolvidos). **Nenhuma janela de 13-20s por cache macro.**

## 11.3 — FIX DE DIREÇÃO DO WHALE NO TEMPLATE (texto errado no payload)

### Causa raiz (achado do log)
- `flow_summary.py:227` usava `direction_w = "comprando" if sf_w > 0 else "vendendo"`, mas `sf_w` = `sector_flow.whale.delta` é **acumulado de sessão** (flow_analyzer/core.py:532-537, reset a cada `CVD_RESET_INTERVAL_HOURS`, :702-710), não o delta da janela corrente → direção do whale escrita errada (caso real janela #6: `sf_w=+16.461` > volume total 6.351 com `buy_pct=2`, `bsr=0.03`, `delta=-6.036` — dizia "whales comprando").

### Fix aplicado
- `_build_note` ganhou parâmetros `buy_pct: float = 50.0` e `bsr: float = 1.0` (defaults neutros); `direction_w = "comprando" if buy_pct > 50 else "vendendo"` (flow_summary.py:182-197, ~230); call site passa `flow.get("buy_pct", 50)` e `flow.get("bsr", 1)`. Caso real janela #6 (buy_pct=2) agora gera "(whales vendendo)".

## 11.4 — PATCHES ANTERIORES CONFIRMADOS EM PRODUÇÃO

- **Patch 2 (BOM no issues.log)**: `logs/issues.log.legacy-1786149845` criado no 1º run real (22:25:45) — arquivo legível, sem bytes binários.
- **Patch 3 (run.log ativo)**: `logs/run.log` recebendo INFO+ desde 22:25 (antes parado em 2025-12-22) — observabilidade do dia restaurada.
- **Patch 4 (JSON snapshot OFF)**: `dados/eventos-fluxo.json` NÃO é recriado (teste dirigido + 20+ janelas em execução prolongada); backup `dados/eventos-fluxo.json.legacy` (415.149 bytes); `eventos_fluxo.jsonl` ativo. Código: `event_saver.py:362-370` default `write_json=False` + `config/settings.py:150-153` `EVENT_SAVER_WRITE_JSON = False` (o `True` no config anulava o fallback — o getattr vencia).

## 11.5 — TESTES

- `pytest tests/payload/test_payload_sections.py` → **55 passed** (template fix).
- `pytest tests/integration/test_integrated_macro_provider.py tests/integration/test_macro_data_provider.py` → **11 passed** (gather/timeout; coverage gate global <10% falha NÃO relacionada).
- `pytest tests/integration/test_macro_singleton_fix.py tests/integration/test_integrated_macro_provider.py tests/payload/test_payload_sections.py` → **58 passed** (Parte 3).
- Teste dirigido `force=True` (cache setado manualmente): sem force → HIT (sem refetch); com force → refaz fetch e renova timestamp. ✅


# RESUMO FINAL


## Tamanho total do payload IA

- Payload compactado real (schema v1, keys longas): **~3.4-3.9 KB ≈ 850-970 tokens** por chamada (p50 do arquivo de eventos = 3.424 bytes).
- Payload v3.1 (`build_compact_payload`): alvo **~800 bytes ≈ 200 tokens** (medianas observadas: evento otimizado 814 bytes; payload_bytes p50=173 no metrics — muitos registros são linhas de cache/leak, não payloads finais).
- Evento bruto ANTES da compactação: **p50=21,3 KB (≈5.300 tokens)**; maior = 41,4 KB (Exaustão).
- Guardrail `leak_blocked` reduz eventos de 51-52 KB para ~3,7 KB (~93% de redução).
- Pior caso por análise (retries): prompt de ~1-4 KB × até 6 tentativas (`_call_openai_compatible`) ou 3 chamadas de fallback de modelo.

## Eventos processados nas últimas 24h

- `payload_metrics.jsonl`: **42 registros com epoch_ms nas últimas 24h** (de 1.045 linhas totais).
- `events` (SQLite): 18 eventos entre 2026-08-07T22:31 e 22:43 (≈13 min de execução observada); todos marcados com `is_signal` exceto Alerta e AI_ANALYSIS.
- `eventos-fluxo.json`: 18 eventos, mesmo intervalo (22:31→22:43 UTC).
- Dados de histórico: 2 eventos AI_ANALYSIS registrados no DB no período (ids 4 e 10).

## Possíveis duplicações identificadas

- `dados/eventos-fluxo.json` ≡ `dados/eventos_fluxo.jsonl` (mesmo conteúdo, 18 eventos, formatos diferentes).
- 3 construtores de payload (compact ativo; ai_payload_builder só config; compressor_v3 só testes) + 4 arquivos `.bak`.
- 3 implementações de detecção de whale (`institutional/whale_detector.py`, `flow_analyzer/whale_score.py`, thresholds em `flow_analyzer/core.py`).
- 2 módulos de absorção (`institutional/absorption_detector.py`, `flow_analyzer/absorption.py`).
- Dupla camada de throttling: `_ai_min_interval_sec` (orchestrator) + `SmartAIThrottler` (ai_runner) — com configurações divergentes (60/30/85k/10 vs 180/60/50k/6).
- `legacy/` com 7+ módulos antigos ainda no tree; `orderbook_analyzer/` com 4 variantes de analisador.
- `skill_bridge.py` não integrado (código futuro); `payload_compressor_v3` não usado no runtime.

## Inconsistências config × comportamento observado

- `model_config.yaml` llm_payload.max_bytes=**6144** e tripwire bytes_p95_max=**90000** (comentário: 'payloads reais chegam a 80KB') vs `config.json` sem esse parâmetro e payloads reais compactados de ~3,8 KB. Eventos brutos de 51 KB estão sendo bloqueados pelo guardrail — limites parecem incompatíveis com os fluxos v1/v2.
- Modelo principal `openai/gpt-oss-120b` com fallbacks `llama-3.1-8b-instant`/`mixtral-8x7b-32768` (model_config.yaml:6-8) vs apenas modelo no config.json ai.groq.model — dois locais de configuração de modelo (config.json vence onde lido primeiro).
- Throttler: ai_runner configura min_interval=60/hard=30/max_calls_per_hour=10, mas o dataclass default é 180/60/6; quem chamar `get_throttler()` sem kwargs primeiro define o singleton (comportamento dependente de ordem de import).
- `eventos_visuais.log` registra latency `POOR`/`ms=11450` no qual do último evento (qual.lat POOR no payload) enquanto `data_reliability.latency_acceptable: 1` no mesmo evento — sinais conflitantes de qualidade.
- payload_metrics mostra `fallback_v1: true` com erro 'price_context.current_price é obrigatório' — indica que alguns eventos falham na validação do payload v2 e caem para v1 (custos/tamanhos diferentes).
- `issues.log` corrompido (binário) — não é possível auditar erros acumulados; e `logs/` não contém log de execução do dia (run.log parado em 2025-12-22) — a observabilidade de erro parece estar sendo feita só via payload_metrics/eventos.


---
Fim do AUDIT_REPORT.md
