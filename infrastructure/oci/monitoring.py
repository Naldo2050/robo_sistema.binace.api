# infrastructure/oci/monitoring.py
import logging
import os
import oci
import platform
import psutil
import time
from datetime import datetime, timezone

try:
    from config import OCI_COMPARTMENT_ID
except ImportError:
    OCI_COMPARTMENT_ID = None

logger = logging.getLogger("OCIMonitoring")

class OCIMonitor:
    """
    Monitoramento nativo para Oracle Cloud Infrastructure.
    Usa Instance Principal Authentication para enviar métricas customizadas.
    """
    # Caminho do arquivo de configuração OCI
    oci_config_path = os.path.expanduser("~/.oci/config")
    
    def __init__(self, compartment_id=None):
        self.auth_method = "instance_principal"
        self.compartment_id = compartment_id or OCI_COMPARTMENT_ID
        self.signer = None
        self.monitoring_client = None
        self.namespace = "MarketBot_Prod"
        self.resource_group = "BotMetrics"
        
        self.enabled = False
        
        self._authenticate()

    def _authenticate(self):
        """Tenta autenticar usando Instance Principal (prod) ou Config File (dev)."""
        try:
            # Tenta Instance Principal primeiro (para rodar na VM)
            self.signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
            self.monitoring_client = oci.monitoring.MonitoringClient(
                config={}, 
                signer=self.signer
            )
            self.enabled = True
            logger.info("✅ OCI Monitoring: Autenticado via Instance Principal")
        except Exception as e:
            logger.debug(f"Falha ao autenticar com Instance Principal: {e}")
            try:
                # Fallback para config file (~/.oci/config) para testes locais
                config = oci.config.from_file(OCIMonitor.oci_config_path)
                self.monitoring_client = oci.monitoring.MonitoringClient(config)
                # Se não fornecido, tenta descobrir ou usa do config
                if not self.compartment_id:
                     self.compartment_id = config.get("tenancy") # fallback ruim, mas algo
                self.enabled = True
                self.auth_method = "config_file"
                logger.info("✅ OCI Monitoring: Autenticado via Config File")
            except Exception as e2:
                logger.warning(f"⚠️ OCI Monitoring DESATIVADO: Falha na autenticação. {e2}")
                self.enabled = False

    def post_metrics(self, metrics_dict):
        """
        Envia um dicionário de métricas para o OCI.
        metrics_dict: dict { 'NomeMetrica': valor, ... }
        """
        if not self.enabled or not self.compartment_id:
            return

        timestamp = datetime.now(timezone.utc)
        metric_data_details = []

        for name, value in metrics_dict.items():
            # Proteção contra valores nulos/inválidos
            if value is None: 
                continue
                
            metric_data_details.append(
                oci.monitoring.models.MetricDataDetails(
                    namespace=self.namespace,
                    resource_group=self.resource_group,
                    compartment_id=self.compartment_id,
                    name=name,
                    dimensions={
                        "host": platform.node(),
                        "app": "market_bot"
                    },
                    datapoints=[
                        oci.monitoring.models.Datapoint(
                            timestamp=timestamp,
                            value=float(value)
                        )
                    ]
                )
            )

        if not metric_data_details:
            return

        try:
            post_metric_data_response = self.monitoring_client.post_metric_data(
                post_metric_data_details=oci.monitoring.models.PostMetricDataDetails(
                    metric_data=metric_data_details
                )
            )
            logger.debug(f"📊 OCI Metrics sent: {list(metrics_dict.keys())}")
        except Exception as e:
            logger.error(f"❌ Erro ao enviar métricas OCI: {e}")

    def collect_system_metrics(self):
        """Coleta métricas de sistema básicas."""
        mem = psutil.virtual_memory()
        return {
            "MemoryUsagePercent": mem.percent,
            "MemoryAvailableMB": mem.available / 1024 / 1024,
            "CpuPercent": psutil.cpu_percent(interval=None)
        }
