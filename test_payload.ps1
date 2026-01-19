# PowerShell runner para testes focados em payload/LLM (sem WSL)

python -m pytest `
  tests/payload `
  -q `
  -m payload `
  --confcutdir=tests/payload `
  -o addopts= `
  --cov=market_orchestrator.ai.payload_compressor `
  --cov=market_orchestrator.ai.llm_payload_guardrail `
  --cov-report=term-missing `
  --no-cov-on-fail `
  --cov-fail-under=0
