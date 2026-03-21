#!/bin/bash
# ============================================
# ETAPA 0: Backup e Baseline de Testes
# Executar ANTES de qualquer mudança
# ============================================

set -e

echo "=========================================="
echo "ETAPA 0 - BACKUP E BASELINE"
echo "=========================================="

# 1. Criar branch de segurança
echo "[1/5] Criando branch de backup..."
git add -A
git commit -m "backup: estado antes da reorganização estrutural" --allow-empty
git tag "pre-refactor-$(date +%Y%m%d-%H%M%S)"
git push origin --tags

# 2. Criar branch de trabalho
echo "[2/5] Criando branch de trabalho..."
git checkout -b refactor/structural-cleanup
git push -u origin refactor/structural-cleanup

# 3. Rodar todos os testes e salvar baseline
echo "[3/5] Rodando testes - salvando baseline..."
mkdir -p scripts/migration/reports

python -m pytest tests/unit/ -v --tb=short 2>&1 | tee scripts/migration/reports/baseline_unit.txt
UNIT_EXIT=${PIPESTATUS[0]}

python -m pytest tests/integration/ -v --tb=short 2>&1 | tee scripts/migration/reports/baseline_integration.txt
INTEGRATION_EXIT=${PIPESTATUS[0]}

python -m pytest tests/e2e/ -v --tb=short 2>&1 | tee scripts/migration/reports/baseline_e2e.txt
E2E_EXIT=${PIPESTATUS[0]}

# 4. Contar testes que passam
echo "[4/5] Contando testes..."
UNIT_PASS=$(grep -c "PASSED" scripts/migration/reports/baseline_unit.txt || echo "0")
UNIT_FAIL=$(grep -c "FAILED" scripts/migration/reports/baseline_unit.txt || echo "0")
INT_PASS=$(grep -c "PASSED" scripts/migration/reports/baseline_integration.txt || echo "0")
INT_FAIL=$(grep -c "FAILED" scripts/migration/reports/baseline_integration.txt || echo "0")
E2E_PASS=$(grep -c "PASSED" scripts/migration/reports/baseline_e2e.txt || echo "0")
E2E_FAIL=$(grep -c "FAILED" scripts/migration/reports/baseline_e2e.txt || echo "0")

# 5. Gerar relatório
echo "[5/5] Gerando relatório baseline..."
cat > scripts/migration/reports/baseline_summary.json << EOF
{
  "timestamp": "$(date -Iseconds)",
  "unit": {"passed": $UNIT_PASS, "failed": $UNIT_FAIL, "exit_code": $UNIT_EXIT},
  "integration": {"passed": $INT_PASS, "failed": $INT_FAIL, "exit_code": $INTEGRATION_EXIT},
  "e2e": {"passed": $E2E_PASS, "failed": $E2E_FAIL, "exit_code": $E2E_EXIT},
  "total_passed": $(($UNIT_PASS + $INT_PASS + $E2E_PASS)),
  "total_failed": $(($UNIT_FAIL + $INT_FAIL + $E2E_FAIL))
}
EOF

echo ""
echo "=========================================="
echo "BASELINE COMPLETO"
echo "=========================================="
echo "Unit:        $UNIT_PASS passed, $UNIT_FAIL failed"
echo "Integration: $INT_PASS passed, $INT_FAIL failed"
echo "E2E:         $E2E_PASS passed, $E2E_FAIL failed"
echo "=========================================="
echo ""
echo "PRÓXIMO PASSO: Execute etapa1_dependency_map.py"
