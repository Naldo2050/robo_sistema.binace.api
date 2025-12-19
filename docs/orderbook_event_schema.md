# Schema do Evento de OrderBook

Este documento descreve a estrutura do evento retornado por:

- `OrderBookAnalyzer.analyze(...)`
- `orderbook_wrapper.fetch_orderbook_with_retry(...)`
- `orderbook_wrapper.orderbook_fallback(...)` (modo emergência/erro)

O schema é estável em `schema_version = "2.1.0"` e a implementação atual do engine
está em `engine_version = "2.2.x"`.

---

## 1. Campos de identificação

```jsonc
{
  "schema_version": "2.1.0",
  "engine_version": "2.2.0",
  "tipo_evento": "OrderBook",
  "ativo": "BTCUSDT"
}