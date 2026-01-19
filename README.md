# Robo Sistema Binance API

Um bot para análise de dados de mercado da Binance com suporte a análise de IA.

## Arquitetura Geral

A arquitetura completa do Enhanced Market Bot – incluindo:

- fluxo do `main.py` até o `EnhancedMarketBot`,
- gerenciador de conexão (`RobustConnectionManager`),
- processamento de janelas (`window_processor` + `DataPipeline` + `FlowAnalyzer`),
- módulo de OrderBook (`OrderBookAnalyzer` + wrapper),
- camada de IA (quantitativa + generativa),
- EventBus, EventSaver e HealthMonitor,

está documentada em:

- [`docs/architecture.md`](docs/architecture.md)

Para operação do sistema e troubleshooting, veja também:

- [`docs/RUNBOOK.md`](docs/RUNBOOK.md)
- [`docs/troubleshooting.md`](docs/troubleshooting.md)

## Instalação

Recomenda-se o uso de um ambiente virtual (como `venv` ou `conda`).

```bash
# (Opcional) Criar e ativar um ambiente virtual
# python -m venv venv
# source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate     # Windows

# Instalar dependências
pip install python-dotenv # ou pip install -r requirements.txt (se tiver)
pip install dashscope     # Para DashScope
pip install openai        # Para OpenAI (opcional)
# ... outras dependências ...

# Dependências de desenvolvimento (tests/linters)
pip install -r requirements-dev.txt
```

## MarketOrchestrator

O `MarketOrchestrator` é o componente principal que coordena todos os analisadores e processadores do sistema. Ele gerencia as conexões, processa os dados de mercado e coordena a análise entre diferentes componentes.

```python
from market_orchestrator import MarketOrchestrator

# Inicialização do orchestrador
orchestrator = MarketOrchestrator(config=config)
await orchestrator.start()
```

## OrderBookAnalyzer – Uso recomendado

O `OrderBookAnalyzer` é 100% assíncrono e deve ser usado assim:

```python
import asyncio
from orderbook_analyzer import OrderBookAnalyzer

async def main():
    async with OrderBookAnalyzer(symbol="BTCUSDT") as oba:
        event = await oba.analyze()
        if event["is_valid"]:
            print("Bid depth:", event["orderbook_data"]["bid_depth_usd"])
            print("Ask depth:", event["orderbook_data"]["ask_depth_usd"])
        else:
            print("Erro de orderbook:", event.get("erro"))

if __name__ == "__main__":
    asyncio.run(main())
```

## Testes focados em payload/LLM

Para validar mudanças no compressor/guardrail de payload sem depender do gate global de cobertura, execute:

```bash
./scripts/test_payload.sh
```

Esse comando roda apenas os testes de payload (compressor + guardrail) com cobertura restrita aos módulos relevantes e sem fail-under global.

No Windows (PowerShell), use:

```powershell
.\test_payload.ps1
```

### Métodos deprecated

Os métodos abaixo ainda existem apenas por compatibilidade, mas serão removidos futuramente:

- `analyze_order_book(...)`
- `analyzeOrderBook(...)`
- `analyze_orderbook(...)`

Use sempre:

```python
await oba.analyze(...)
```

em vez desses shims.

Isso garante que qualquer desenvolvedor que abra o repo entenda o caminho certo, e veja que os outros nomes são apenas legados.

## Evento de OrderBook – Contrato (Schema)

O evento retornado por `OrderBookAnalyzer.analyze(...)` e pelo
`orderbook_wrapper.fetch_orderbook_with_retry(...)` segue um contrato estável
em `schema_version = "2.1.0"`.

A descrição detalhada de todos os campos (incluindo `orderbook_data`,
`spread_metrics`, `advanced_metrics`, `health_stats`, etc.) está em:

- `orderbook_analyzer.py`
- `orderbook_core/event_factory.py`

### Guia de Operação do OrderBook

Para entender como chamar o `OrderBookAnalyzer`, usar o wrapper no bot e
monitorar o módulo via métricas, logs e Circuit Breaker, veja:

- [`docs/RUNBOOK.md`](docs/RUNBOOK.md)
