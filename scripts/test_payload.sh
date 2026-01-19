#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

pytest -q \
  tests/test_payload_compressor.py \
  tests/test_payload_guardrail.py \
  --cov=market_orchestrator.ai.payload_compressor \
  --cov=ai_analyzer_qwen \
  --cov-report=term-missing \
  --no-cov-on-fail
