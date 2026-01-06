# Usar imagem oficial Python leve
FROM python:3.11-slim

# Definir variáveis de ambiente para Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=America/New_York

# Instalar dependências do sistema necessárias
# gcc e python3-dev para compilar certas libs pip
# curl para healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    max_user_watches=524288 \
    python3-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Configurar diretório de trabalho
WORKDIR /app

# Criar usuário não-root para segurança
RUN groupadd -r trader && useradd -r -g trader trader

# Copiar apenas requirements primeiro para cache do Docker
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Instalar playwright browsers se necessário (comentado se não for crítico para produção)
# RUN playwright install chromium --with-deps

# Copiar o restante do código
COPY . .

# Criar diretórios necessários e ajustar permissões
RUN mkdir -p data logs features && \
    chown -R trader:trader /app

# Mudar para o usuário não-root
USER trader

# Expor portas se necessário (ex: dashboard web)
# EXPOSE 8000

# Healthcheck
# Verifica se o arquivo de health escrito pelo health_monitor.py foi atualizado recentemente
# (O script python deve escrever timestamp em /tmp/health_status)
# Ajuste conforme implementação do health_monitor.py
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD test $(find /app/logs/health_status -mmin -1 2>/dev/null | wc -l) -gt 0 || exit 1

# Comando de entrada
CMD ["python", "main.py"]
