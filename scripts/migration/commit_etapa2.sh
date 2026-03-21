#!/bin/bash
# scripts/migration/commit_etapa2.sh

echo "Validando etapa 2..."
python -m pytest tests/unit/ -x -q 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Testes falharam! Revertendo..."
    git checkout -- .
    exit 1
fi

python -m pytest tests/integration/ -x -q 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Testes de integração falharam! Revertendo..."
    git checkout -- .
    exit 1
fi

echo "✅ Testes passaram. Commitando..."
git add -A
git commit -m "refactor(etapa2): unificar exceptions em common/exceptions.py

- Mover todas as exceções para common/exceptions.py
- Manter re-exports nos arquivos originais para compatibilidade
- Zero mudança de comportamento
- Todos os testes passando"

git push origin refactor/structural-cleanup
echo "✅ Etapa 2 commitada e enviada!"
