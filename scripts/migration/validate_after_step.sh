#!/bin/bash
# scripts/migration/validate_after_step.sh
# Uso: ./validate_after_step.sh <numero_etapa>

set -e

ETAPA=$1
if [ -z "$ETAPA" ]; then
    echo "Uso: ./validate_after_step.sh <numero_etapa>"
    exit 1
fi

echo "=========================================="
echo "VALIDAÇÃO PÓS-ETAPA $ETAPA"
echo "=========================================="

# Rodar testes
python -m pytest tests/unit/ -v --tb=short 2>&1 | tee scripts/migration/reports/etapa${ETAPA}_unit.txt
python -m pytest tests/integration/ -v --tb=short 2>&1 | tee scripts/migration/reports/etapa${ETAPA}_integration.txt

# Comparar com baseline
BASELINE_PASS=$(python -c "import json; d=json.load(open('scripts/migration/reports/baseline_summary.json')); print(d['total_passed'])")
CURRENT_PASS=$(grep -c "PASSED" scripts/migration/reports/etapa${ETAPA}_unit.txt scripts/migration/reports/etapa${ETAPA}_integration.txt || echo "0")
CURRENT_FAIL=$(grep -c "FAILED" scripts/migration/reports/etapa${ETAPA}_unit.txt scripts/migration/reports/etapa${ETAPA}_integration.txt || echo "0")

echo ""
echo "=========================================="
echo "RESULTADO ETAPA $ETAPA"
echo "=========================================="
echo "Baseline:  $BASELINE_PASS testes passando"
echo "Agora:     $CURRENT_PASS passando, $CURRENT_FAIL falhando"

if [ "$CURRENT_FAIL" -gt "0" ]; then
    echo ""
    echo "⚠️  ATENÇÃO: Há testes falhando!"
    echo "Verifique antes de commitar."
    echo "Para reverter: git checkout -- ."
else
    echo ""
    echo "✅ Todos os testes passaram!"
    echo "Seguro para commitar."
fi
