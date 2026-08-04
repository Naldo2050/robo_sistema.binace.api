#!/bin/bash
# Script para instalar Python e dependências no servidor remoto

echo "=========================================="
echo "INSTALAÇÃO DE PYTHON E DEPENDÊNCIAS"
echo "=========================================="

cd ~/robo_deploy || { echo "Erro: diretório ~/robo_deploy não existe"; exit 1; }

# 1. Atualizar packages
echo -e "\n[1/5] Atualizando packages..."
sudo apt update && sudo apt install python3-pip python3-venv -y

# 2. Verificar Python
echo -e "\n[2/5] Verificando Python..."
python3 --version

# 3. Criar ambiente virtual
echo -e "\n[3/5] Criando ambiente virtual..."
if [ -d .venv ]; then
    echo "✓ Ambiente virtual já existe"
else
    python3 -m venv .venv
    echo "✓ Ambiente virtual criado"
fi

# 4. Ativar ambiente virtual e atualizar pip
echo -e "\n[4/5] Ativando ambiente virtual..."
source .venv/bin/activate

# Atualizar pip
pip install --upgrade pip

# 5. Instalar dependências
echo -e "\n[5/5] Instalando dependências..."
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
    echo "✓ Dependências instaladas com sucesso"
else
    echo "✗ Arquivo requirements.txt não encontrado"
    exit 1
fi

echo -e "\n=========================================="
echo "INSTALAÇÃO CONCLUÍDA COM SUCESSO!"
echo "=========================================="
echo -e "\nPara ativar o ambiente virtual em futuras sessões, execute:"
echo "source ~/robo_deploy/.venv/bin/activate"
