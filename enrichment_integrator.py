# enrichment_integrator.py
"""
Integra o DataEnricher no pipeline de eventos.
Uso: enriquecer eventos ANALYSIS_TRIGGER.
"""

from __future__ import annotations

from typing import Dict, Any
import logging

from data_enricher import DataEnricher

logger = logging.getLogger(__name__)


def build_analysis_trigger_event(symbol: str, raw_event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Constrói um evento ANALYSIS_TRIGGER com campos padrão.
    
    Args:
        symbol: Símbolo do ativo (ex: "BTCUSDT")
        raw_event: Dicionário com dados do evento bruto
        
    Returns:
        Dicionário representando o evento ANALYSIS_TRIGGER
    """
    # Extrair preço de fechamento de várias fontes possíveis
    preco_fechamento = (
        raw_event.get("preco_fechamento") or
        raw_event.get("price") or 
        raw_event.get("close") or
        (raw_event.get("raw_event", {}).get("preco_fechamento") if isinstance(raw_event.get("raw_event"), dict) else None) or
        (raw_event.get("raw_event", {}).get("price") if isinstance(raw_event.get("raw_event"), dict) else None)
    )
    
    event = {
        "is_signal": True,
        "tipo_evento": "ANALYSIS_TRIGGER",
        "descricao": "Evento automático para análise da IA",
        "symbol": symbol,
        "raw_event": raw_event,
        "resultado_da_batalha": "N/A",
    }
    
    # Adicionar preco_fechamento se disponível
    if preco_fechamento is not None:
        event["preco_fechamento"] = preco_fechamento
        # Garantir que também está no raw_event
        if "raw_event" not in event:
            event["raw_event"] = {}
        if isinstance(event["raw_event"], dict):
            event["raw_event"]["preco_fechamento"] = preco_fechamento
    
    return event


def enrich_analysis_trigger_event(
    event: Dict[str, Any], config_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Enriquece um evento ANALYSIS_TRIGGER adicionando raw_event.advanced_analysis.

    Chamar logo após montar o raw_event.
    """
    try:
        tipo = event.get("tipo_evento") or event.get("type")
        if tipo != "ANALYSIS_TRIGGER":
            return event

        raw_event = event.get("raw_event") or {}
        # Verificar preco_fechamento tanto na raiz quanto no nível interno
        inner_raw = raw_event.get("raw_event") if isinstance(raw_event.get("raw_event"), dict) else {}
        preco_fechamento = raw_event.get("preco_fechamento") or (inner_raw.get("preco_fechamento") if inner_raw else None)
        if preco_fechamento is None:
            logger.warning("EVENTO sem preco_fechamento (root e inner), não será enriquecido")
            return event

        # Se já tem advanced_analysis completo (do contextual), não sobrescrever
        if "advanced_analysis" in raw_event and isinstance(raw_event["advanced_analysis"], dict):
            advanced = raw_event["advanced_analysis"]
        else:
            enricher = DataEnricher(config_dict)
            # Usar a nova função que calcula usando raw_event EXTERNO
            # CORREÇÃO: Capturar retorno e garantir atualização
            updated_event = enricher.enrich_event_with_advanced_analysis(event)
            if updated_event:
                # Atualizar o evento original com os dados modificados
                event.update(updated_event)
                raw_event = updated_event.get("raw_event", event.get("raw_event", {}))
            else:
                raw_event = event.get("raw_event", {})
            # Pegar do interno se existir, senão do externo
            inner = raw_event.get("raw_event")
            if inner and "advanced_analysis" in inner:
                advanced = inner["advanced_analysis"]
            else:
                advanced = raw_event.get("advanced_analysis", {})

        # >>> ADICIONE ESTE LOG DE DIAGNÓSTICO <<<
        try:
            logger.info(
                "[EnrichmentIntegrator] advanced_analysis runtime keys=%s | "
                "price_targets_len=%s | symbol=%s | price=%s",
                list(advanced.keys()),
                len(advanced.get("price_targets", []))
                if isinstance(advanced.get("price_targets"), list)
                else "N/A",
                advanced.get("symbol"),
                advanced.get("price"),
            )
        except Exception:
            logger.exception("[EnrichmentIntegrator] Falha ao logar advanced_analysis")

        event.setdefault("raw_event", {})["advanced_analysis"] = advanced

        return event

    except Exception as e:
        logger.error(f"Erro ao enriquecer ANALYSIS_TRIGGER: {e}", exc_info=True)
        return event