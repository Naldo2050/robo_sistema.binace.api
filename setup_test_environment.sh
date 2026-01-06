#!/bin/bash
# setup_test_environment.sh

echo "🛠️  Configurando ambiente de testes..."

# Cria estrutura de diretórios
mkdir -p tests

# Copia arquivos de configuração
echo "📁 Copiando arquivos de configuração..."

# Configuração do coverage
cat > .coveragerc << 'EOF'
[run]
branch = True
source = 
    orderbook_analyzer/
    orderbook_core/
    market_orchestrator/
    ai_runner/
    risk_management/
    utils/
    
omit = 
    */tests/*
    */__pycache__/*
    */legacy/*
    */migrations/*
    venv/*
    .venv/*
    *setup.py
    *conftest.py

[report]
exclude_lines = 
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    @property
    @abstractmethod
    pass
    logger\.
    ^\s*$
    ^\s*#.*$

precision = 2
show_missing = True
skip_covered = False
fail_under = 65
EOF

# Configuração do pytest
cat > pytest.ini << 'EOF'
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --strict-markers
    --tb=short
    -p no:warnings
    
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: integration tests
    unit: unit tests
    async: async tests
    performance: performance benchmarks
    
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
    ignore::ResourceWarning
    
log_cli = true
log_cli_level = INFO
log_cli_format = %(asctime)s [%(levelname)s] %(message)s
log_cli_date_format = %Y-%m-%d %H:%M:%S
EOF

# Faz o script executável
chmod +x run_tests_with_coverage.sh
chmod +x setup_test_environment.sh

echo "✅ Configuração completa!"
echo ""
echo "📋 Próximos passos:"
echo "1. Copie os arquivos de teste para a pasta tests/"
echo "2. Execute: ./run_tests_with_coverage.sh"
echo "3. Para abrir relatório: ./run_tests_with_coverage.sh --open"