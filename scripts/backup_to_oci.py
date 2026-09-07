import oci
import os
import tarfile
import logging
from datetime import datetime
import shutil

# Configurações via Env Var
BUCKET_NAME = os.getenv("OCI_BACKUP_BUCKET")
NAMESPACE = os.getenv("OCI_NAMESPACE")
DATA_DIRS = ["data", "logs", "features"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OCIBackup")

def authenticate():
    try:
        signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
        return oci.object_storage.ObjectStorageClient(config={}, signer=signer)
    except:
        config = oci.config.from_file()
        return oci.object_storage.ObjectStorageClient(config)

# Arquivos nunca incluídos no backup (podem conter segredos/histórico sensível)
_EXCLUDED_SUFFIXES = (".db", ".sqlite3", ".sqlite", ".jsonl", ".env")

def _tar_filter(tarinfo):
    """Exclui segredos do pacote (fail-closed para backup)."""
    base = tarinfo.name.replace("\\", "/").rsplit("/", 1)[-1]
    if base.endswith(_EXCLUDED_SUFFIXES):
        logger.debug(f"   - Excluído do backup: {tarinfo.name}")
        return None
    return tarinfo

def create_archive(output_filename):
    logger.info(f"📦 Criando arquivo {output_filename}...")
    with tarfile.open(output_filename, "w:gz") as tar:
        for d in DATA_DIRS:
            if os.path.exists(d):
                tar.add(d, filter=_tar_filter)
                logger.info(f"   + Adicionado: {d}")
    return output_filename

def upload_to_oci(client, filename):
    if not BUCKET_NAME or not NAMESPACE:
        logger.error("❌ OCI_BACKUP_BUCKET ou OCI_NAMESPACE não definidos.")
        return

    object_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tar.gz"
    
    try:
        logger.info(f"☁️ Enviando para OCI Object Storage: {object_name}")
        with open(filename, "rb") as f:
            client.put_object(
                NAMESPACE,
                BUCKET_NAME,
                object_name,
                f
            )
        logger.info("✅ Upload concluído com sucesso!")
    except Exception as e:
        logger.error(f"❌ Falha no upload: {e}")

def main():
    if not BUCKET_NAME:
        logger.warning("⚠️ Backup pulado: OCI_BACKUP_BUCKET não configurado.")
        return

    client = authenticate()
    tmp_file = "backup_temp.tar.gz"
    
    try:
        create_archive(tmp_file)
        upload_to_oci(client, tmp_file)
    finally:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

if __name__ == "__main__":
    main()
