#!/bin/bash
# scripts/migration/commit_etapa3.sh

echo "Validando etapa 3..."
python -m pytest tests/ -x -q --timeout=60 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Testes falharam! Verifique antes de commitar."
    echo "Para reverter: git checkout -- ."
    exit 1
fi

git add -A
git commit -m "refactor(etapa3): eliminar duplicações de módulos

- Merge de módulos duplicados identificados pelo diagnóstico
- Arquivos removidos convertidos em re-exports
- Zero mudança de comportamento
- Todos os testes passando"

git push origin refactor/structural-cleanup
echo "✅ Etapa 3 commitada!"
