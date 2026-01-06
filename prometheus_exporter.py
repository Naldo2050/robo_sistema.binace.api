#!/usr/bin/env python3
"""
Prometheus Exporter para Métricas do Sistema de Trading

Este módulo expõe as métricas coletadas via HTTP para que o Prometheus possa coletá-las.
"""

import logging
from prometheus_client import start_http_server, REGISTRY, CollectorRegistry
from metrics_collector import PROMETHEUS_AVAILABLE

logger = logging.getLogger(__name__)

# Criar registry customizado para nuestras métricas
_system_metrics_registry = CollectorRegistry()


def start_prometheus_exporter(port: int = 8000, addr: str = '0.0.0.0'):
    """
    Inicia o servidor HTTP do Prometheus exporter.

    Args:
        port: Porta para expor as métricas (padrão: 8000)
        addr: Endereço para bind (padrão: 0.0.0.0)
    """
    if not PROMETHEUS_AVAILABLE:
        logger.warning("Prometheus client não está disponível. Usando modo simulado.")
        return
    
    try:
        logger.info(f"Iniciando Prometheus exporter em {addr}:{port}")
        start_http_server(port=port, addr=addr, registry=_system_metrics_registry)
        logger.info(f"Prometheus exporter iniciado com sucesso em {addr}:{port}")
        logger.info(f"Métricas disponíveis em http://{addr}:{port}/metrics")
    except Exception as e:
        logger.error(f"Erro ao iniciar Prometheus exporter: {e}")
        raise


def register_metric(metric):
    """
    Registra uma métrica no registry customizado.
    
    Args:
        metric: Métrica do prometheus_client a ser registrada
    """
    if PROMETHEUS_AVAILABLE:
        _system_metrics_registry.register(metric)


if __name__ == "__main__":
    # Configuração básica de logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Iniciar exporter
    start_prometheus_exporter()