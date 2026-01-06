#!/bin/bash
# Script de verificação de saúde remota
# Uso: Copie para o servidor ou execute via ssh: ssh user@host 'bash -s' < scripts/remote_health_check.sh

echo "🔍 Iniciando verificação de saúde do Bot..."

TARGET_DIR=~/robo_sistema.binace.api

if [ -d "$TARGET_DIR" ]; then
    cd "$TARGET_DIR"
    echo "📂 Diretório encontrado: $TARGET_DIR"
    
    echo -e "\n1️⃣  Docker PS:"
    sudo docker ps
    
    echo -e "\n2️⃣  Health Check (market_bot_prod):"
    if sudo docker inspect market_bot_prod >/dev/null 2>&1; then
        sudo docker inspect market_bot_prod | grep -A5 Health
    else
        echo "❌ Container 'market_bot_prod' não encontrado ou parado."
    fi
else
    echo "❌ Erro: Diretório $TARGET_DIR não encontrado."
    exit 1
fi
