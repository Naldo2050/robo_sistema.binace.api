# infrastructure/oci/vault_helper.py
import oci
import base64
import logging
import os

logger = logging.getLogger("OCIVault")

class OCIVaultHelper:
    """
    Helper para recuperar segredos do OCI Vault.
    Cacheia segredos em memória para evitar requisições excessivas.
    """
    def __init__(self):
        self.secrets_client = None
        self.enabled = False
        self._authenticate()
        self.cache = {}

    def _authenticate(self):
        try:
            signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
            self.secrets_client = oci.secrets.SecretsClient(config={}, signer=signer)
            self.enabled = True
        except Exception:
            try:
                config = oci.config.from_file()
                self.secrets_client = oci.secrets.SecretsClient(config)
                self.enabled = True
            except Exception as e:
                if os.getenv("ENVIRONMENT", "dev").lower() == "dev":
                    logger.info(f"ℹ️ OCI Vault não configurado (Modo Dev - Ignorando): {e}")
                else:
                    logger.warning(f"⚠️ OCI Vault indisponível: {e}")
                self.enabled = False

    def get_secret(self, secret_ocid):
        """Recupera e decodifica um segredo dado seu OCID."""
        if not self.enabled or not secret_ocid:
            return None

        if secret_ocid in self.cache:
            return self.cache[secret_ocid]

        try:
            response = self.secrets_client.get_secret_bundle(secret_id=secret_ocid)
            base64_secret_content = response.data.secret_bundle_content.content
            secret_content = base64.b64decode(base64_secret_content).decode("utf-8")
            
            # Cache simples
            self.cache[secret_ocid] = secret_content
            logger.info(f"🔒 Segredo recuperado do Vault: {secret_ocid[:15]}...")
            return secret_content
        except Exception as e:
            logger.error(f"❌ Erro ao buscar secret {secret_ocid}: {e}")
            return None

# Singleton global para uso fácil
_vault_helper = None

def get_vault_secret(secret_ocid):
    global _vault_helper
    if _vault_helper is None:
        _vault_helper = OCIVaultHelper()
    
    # Se não parecer um OCID (ocid1.vaultsecret...), retorna o próprio valor (assumindo que já é a chave ou env var)
    if not secret_ocid or not str(secret_ocid).startswith("ocid1.vaultsecret"):
        return secret_ocid

    return _vault_helper.get_secret(secret_ocid)
