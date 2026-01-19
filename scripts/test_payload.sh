#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

pytest -q \
  tests/payload \
  -m payload \
  --rootdir=. \
  --confcutdir=tests/payload \
  -c tests/payload/pytest.ini \
  --cov=market_orchestrator.ai.payload_compressor \
  --cov=market_orchestrator.ai.llm_payload_guardrail \
  --cov-report=term-missing \
  --no-cov-on-fail \
  --cov-fail-under=0 \
  -o addopts=''
