#!/bin/bash
# Disaster Recovery Script
# Restaura o backup mais recente do OCI Object Storage

set -e

# Configurações (Preencha ou use Env Vars)
BUCKET_NAME="${OCI_BACKUP_BUCKET}"
NAMESPACE="${OCI_NAMESPACE}"

if [ -z "$BUCKET_NAME" ] || [ -z "$NAMESPACE" ]; then
    echo "❌ Erro: Env vars OCI_BACKUP_BUCKET e OCI_NAMESPACE são obrigatórias."
    echo "Exemplo: export OCI_BACKUP_BUCKET=meu-bucket-backup"
    exit 1
fi

echo "🚑 INICIANDO PROTOCOLO DE DISASTER RECOVERY..."

# Verificar OCI CLI
if ! command -v oci &> /dev/null; then
    echo "❌ OCI CLI não encontrado. Instalando..."
    bash -c "$(curl -L https://raw.githubusercontent.com/oracle/oci-cli/master/scripts/install/install.sh)"
fi

echo "🔍 Buscando backup mais recente..."

# Lista objetos, ordena por timestamp (nome do arquivo) e pega o último
LATEST_BACKUP=$(oci os object list --bucket-name $BUCKET_NAME --namespace $NAMESPACE \
    --query "data | sort_by([*], &name) | [-1].name" --raw-output)

if [ "$LATEST_BACKUP" == "None" ] || [ -z "$LATEST_BACKUP" ]; then
    echo "❌ Nenhum backup encontrado no bucket $BUCKET_NAME."
    exit 1
fi

echo "found: $LATEST_BACKUP"

echo "☁️ Baixando $LATEST_BACKUP..."
oci os object get --bucket-name $BUCKET_NAME --namespace $NAMESPACE \
    --name "$LATEST_BACKUP" --file "restore.tar.gz"

echo "📦 Restaurando arquivos..."
# Para o serviço antes de restaurar
docker compose down || true

tar -xzf restore.tar.gz

echo "✅ Arquivos restaurados: data/ logs/ features/"
rm restore.tar.gz

echo "🚀 Reiniciando serviço..."
docker compose up -d

echo "✅ DISASTER RECOVERY CONCLUÍDO COM SUCESSO."
