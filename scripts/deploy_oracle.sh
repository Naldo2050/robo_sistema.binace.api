#!/bin/bash
# Script de Deploy para Oracle Cloud (Oracle Linux 8/9 ou Ubuntu)

set -e

APP_DIR="/opt/market-bot"
REPO_URL="https://github.com/SEU_USUARIO/robo_sistema.binance.api.git" # Substitua pelo seu repo

echo "🚀 Iniciando Deploy na Oracle Cloud..."

# 1. Update System
echo "📦 Atualizando pacotes do sistema..."
if [ -f /etc/redhat-release ]; then
    sudo dnf update -y
    sudo dnf config-manager --add-repo=https://download.docker.com/linux/centos/docker-ce.repo
    sudo dnf install -y docker-ce docker-ce-cli containerd.io git
elif [ -f /etc/debian_version ]; then
    sudo apt-get update && sudo apt-get upgrade -y
    sudo apt-get install -y docker.io git
fi

# 2. Setup Docker
echo "🐳 Configurando Docker..."
sudo systemctl enable docker
sudo systemctl start docker
# Adiciona user atual ao docker group
sudo usermod -aG docker $USER || true

# Install Docker Compose (if not present plugin)
if ! docker compose version >/dev/null 2>&1; then
    echo "⬇️ Instalando plugin Docker Compose Global..."
    sudo mkdir -p /usr/local/lib/docker/cli-plugins
    sudo curl -SL https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-linux-x86_64 -o /usr/local/lib/docker/cli-plugins/docker-compose
    sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
fi

# 3. Setup Application Directory
echo "📂 Configurando diretório da aplicação em $APP_DIR..."
if [ ! -d "$APP_DIR" ]; then
    sudo mkdir -p $APP_DIR
    sudo chown $USER:$USER $APP_DIR
    git clone $REPO_URL $APP_DIR
else
    cd $APP_DIR
    git pull
fi

cd $APP_DIR

# 4. Check for .env
if [ ! -f .env ]; then
    echo "⚠️ ALERTA: Arquivo .env não encontrado!"
    echo "⚠️ Crie o arquivo .env com suas credenciais antes de continuar."
    echo "Use .env.template como base."
    exit 1
fi

# 5. Setup Systemd & Deploy
echo "Building..."
docker compose build --pull

echo "⚙️ Configurando serviço Systemd..."
sudo cp infrastructure/market-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable market-bot.service

echo "🚀 Iniciando serviço..."
sudo systemctl start market-bot.service

echo "✅ Deploy concluído com sucesso!"
echo "📜 Logs: docker compose logs -f"
