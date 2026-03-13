#!/bin/bash
# run_tests_with_coverage.sh

set -e

echo "🔬 Sistema de Trading - Execução de Testes"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configurações
PYTEST_OPTS="-v --tb=short --strict-markers"
COVERAGE_TARGET=65

echo -e "${BLUE}📊 Meta de cobertura: ${COVERAGE_TARGET}%${NC}"

# Função para verificar se o comando existe
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Função para limpar caches
cleanup() {
    echo -e "${YELLOW}🧹 Limpando caches...${NC}"
    rm -rf .coverage coverage_html .pytest_cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete
}

# Função para executar testes de um módulo
run_module_tests() {
    local module_name=$1
    local test_file=$2
    local target_coverage=$3
    
    echo -e "\n${BLUE}🧪 Testando $module_name...${NC}"
    
    if [ ! -f "$test_file" ]; then
        echo -e "${YELLOW}⚠️  Arquivo de teste não encontrado: $test_file${NC}"
        return 1
    fi
    
    if pytest "$test_file" $PYTEST_OPTS \
        --cov="$module_name" \
        --cov-report=term-missing \
        --cov-fail-under=$target_coverage; then
        echo -e "${GREEN}✅ $module_name - TESTES PASSARAM${NC}"
        return 0
    else
        echo -e "${RED}❌ $module_name - TESTES FALHARAM${NC}"
        return 1
    fi
}

# Função principal
main() {
    # Verifica se está no diretório correto
    if [ ! -d "orderbook_analyzer" ] && [ ! -d "tests" ]; then
        echo -e "${RED}❌ Execute este script da raiz do projeto!${NC}"
        exit 1
    fi
    
    cleanup
    
    # Array de módulos para testar (módulo, arquivo_teste, cobertura_alvo)
    modules=(
        "orderbook_analyzer tests/test_orderbook_analyzer_comprehensive.py 80"
        "orderbook_core tests/test_orderbook_core_comprehensive.py 85"
        "market_orchestrator tests/test_market_orchestrator_comprehensive.py 75"
        "ai_runner tests/test_ai_runner_comprehensive.py 70"
        "risk_management tests/test_risk_manager_comprehensive.py 80"
    )
    
    failed_modules=()
    
    # Testa módulos individualmente
    for module_info in "${modules[@]}"; do
        IFS=' ' read -r module test_file target <<< "$module_info"
        
        if ! run_module_tests "$module" "$test_file" "$target"; then
            failed_modules+=("$module")
        fi
    done
    
    # Teste completo com cobertura total
    echo -e "\n${BLUE}🚀 Executando todos os testes...${NC}"
    
    if pytest tests/ $PYTEST_OPTS \
        --cov=. \
        --cov-report=term-missing \
        --cov-report=html \
        --cov-fail-under=$COVERAGE_TARGET; then
        echo -e "${GREEN}✅ TODOS OS TESTES PASSARAM${NC}"
    else
        echo -e "${RED}❌ ALGUNS TESTES FALHARAM${NC}"
    fi
    
    # Gera relatório
    echo -e "\n${BLUE}📈 Gerando relatório de cobertura...${NC}"
    coverage html
    coverage report
    
    # Mostra resumo
    echo -e "\n${BLUE}📊 RESUMO FINAL${NC}"
    echo "============================"
    
    if [ ${#failed_modules[@]} -eq 0 ]; then
        echo -e "${GREEN}✅ Todos os módulos passaram nos testes${NC}"
    else
        echo -e "${RED}❌ Módulos com falha: ${failed_modules[*]}${NC}"
    fi
    
    # Mostra cobertura atual
    coverage_percent=$(coverage report | grep TOTAL | awk '{print $4}' | sed 's/%//')
    if [ -n "$coverage_percent" ]; then
        if (( $(echo "$coverage_percent >= $COVERAGE_TARGET" | bc -l) )); then
            echo -e "${GREEN}📈 Cobertura atual: ${coverage_percent}% (ATINGIU A META)${NC}"
        else
            echo -e "${YELLOW}⚠️  Cobertura atual: ${coverage_percent}% (ABAIXO DA META)${NC}"
        fi
    fi
    
    # Oferece para abrir relatório
    if command_exists "open" && [ "$1" == "--open" ]; then
        echo -e "${BLUE}🌐 Abrindo relatório no navegador...${NC}"
        open coverage_html/index.html
    elif command_exists "xdg-open" && [ "$1" == "--open" ]; then
        echo -e "${BLUE}🌐 Abrindo relatório no navegador...${NC}"
        xdg-open coverage_html/index.html
    else
        echo -e "\n${YELLOW}📁 Relatório HTML disponível em: coverage_html/index.html${NC}"
        echo -e "${YELLOW}👉 Execute com '--open' para abrir automaticamente${NC}"
    fi
    
    # Retorna código de saída apropriado
    if [ ${#failed_modules[@]} -gt 0 ]; then
        exit 1
    fi
}

# Instruções de uso
show_usage() {
    echo "Uso: $0 [OPÇÕES]"
    echo ""
    echo "Opções:"
    echo "  --open          Abre o relatório de cobertura no navegador"
    echo "  --help          Mostra esta mensagem"
    echo ""
    echo "Exemplos:"
    echo "  $0               # Executa todos os testes"
    echo "  $0 --open        # Executa testes e abre relatório"
}

# Parse arguments
case "$1" in
    --help|-h)
        show_usage
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac