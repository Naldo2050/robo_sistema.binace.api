# PowerShell runner para testes focados em payload/LLM (sem WSL)
Remove-Item Env:PYTEST_DISABLE_PLUGIN_AUTOLOAD -ErrorAction SilentlyContinue

python -m pytest `
  -q `
  tests/payload `
  -m payload `
  --rootdir=. `
  --confcutdir=tests/payload `
  -c tests/payload/pytest.ini `
  -o addopts= `
  --cov=market_orchestrator.ai.payload_compressor `
  --cov=market_orchestrator.ai.llm_payload_guardrail `
  --cov-report=term-missing `
  --no-cov-on-fail `
  --cov-fail-under=0
