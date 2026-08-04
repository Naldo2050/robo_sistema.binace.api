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
RUN mkdir -p dados logs features && \
    chown -R trader:trader /app

# Mudar para o usuário não-root
USER trader

# Expor porta do servidor Prometheus (/metrics)
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/metrics || exit 1

# Comando de entrada
CMD ["python", "main.py"]
