@echo off
echo 🔬 Sistema de Trading - Execução de Testes (Windows)
echo ==========================================

echo 📊 Meta de cobertura: 65%%

echo.
echo 🧹 Limpando caches...
if exist .coverage del .coverage
if exist coverage_html rmdir /s /q coverage_html
if exist .pytest_cache rmdir /s /q .pytest_cache

echo.
echo 🧪 Testando orderbook_analyzer...
pytest tests/test_orderbook_analyzer_comprehensive.py -v --tb=short --strict-markers --cov=orderbook_analyzer --cov-report=term-missing --cov-fail-under=80

echo.
echo 🚀 Executando todos os testes...
pytest tests/ -v --tb=short --strict-markers --cov=. --cov-report=term-missing --cov-report=html --cov-fail-under=65

echo.
echo 📈 Gerando relatório de cobertura...
coverage html
coverage report

echo.
echo 📊 RESUMO FINAL
echo ============================
echo 📁 Relatório HTML disponível em: coverage_html/index.html

pause