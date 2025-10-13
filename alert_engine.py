"""
alert_engine.py

Motor simplificado para detecção de condições de mercado relevantes.
Apenas detecta spikes de volume e compressão de volatilidade.
REMOVEU todos os alertas de suporte/resistência conforme solicitado.
"""

import numpy as np
from typing import List, Dict, Any, Optional

# 🔹 IMPORTA UTILITÁRIOS DE FORMATAÇÃO
from format_utils import (
    format_price,
    format_quantity,
    format_percent,
    format_large_number,
    format_delta,
    format_scientific
)


def _clean_alert_data(alert: Dict[str, Any]) -> Dict[str, Any]:
    """
    Limpa e formata valores numéricos no alerta antes de retornar.
    Mantém os valores como números para processamento, mas adiciona versões formatadas.
    """
    cleaned = alert.copy()
    
    # Adiciona versões formatadas dos valores numéricos
    if 'threshold_exceeded' in cleaned:
        # Para ratios de volume, mostra como multiplicador (ex: 3.5x)
        cleaned['threshold_exceeded_display'] = f"{cleaned['threshold_exceeded']:.1f}x"
    
    if 'current_volume' in cleaned:
        cleaned['current_volume_display'] = format_large_number(cleaned['current_volume'])
    
    if 'average_volume' in cleaned:
        cleaned['average_volume_display'] = format_large_number(cleaned['average_volume'])
    
    if 'volatility_current' in cleaned:
        cleaned['volatility_current_display'] = format_scientific(cleaned['volatility_current'], decimals=5)
    
    if 'volatility_threshold' in cleaned:
        cleaned['volatility_threshold_display'] = format_scientific(cleaned['volatility_threshold'], decimals=5)
    
    if 'percentile' in cleaned:
        cleaned['percentile_display'] = format_percent(cleaned['percentile'])
    
    return cleaned


def detect_volume_spike(
    current_volume: float,
    average_volume: float,
    threshold_factor: float = 3.0,
    duration: str = "5_MINUTES",
) -> Optional[Dict[str, Any]]:
    """Volume spike detection com formatação adequada."""
    if average_volume <= 0:
        return None
    
    ratio = current_volume / average_volume
    
    if ratio >= threshold_factor:
        alert = {
            "type": "VOLUME_SPIKE",
            "threshold_exceeded": round(ratio, 2),
            "severity": "MEDIUM" if ratio < 5 else "HIGH",
            "duration": duration,
            # 🔹 Adiciona valores formatados para display
            "current_volume": current_volume,
            "average_volume": average_volume,
            "volume_ratio": ratio,
            "description": f"Volume {format_large_number(current_volume)} é {ratio:.1f}x a média de {format_large_number(average_volume)}"
        }
        return _clean_alert_data(alert)
    
    return None


def detect_volatility_squeeze(
    current_vol: float,
    recent_vols: List[float],
    low_percentile: float = 10.0,
    high_percentile: float = 90.0,
) -> Optional[Dict[str, Any]]:
    """Volatility squeeze detection com formatação adequada."""
    if not recent_vols:
        return None
    
    vols = np.array([v for v in recent_vols if v is not None])
    if vols.size == 0:
        return None
    
    low_thresh = np.percentile(vols, low_percentile)
    high_thresh = np.percentile(vols, high_percentile)
    
    # 🔹 Calcula percentil atual
    current_percentile = np.sum(vols <= current_vol) / len(vols) * 100
    
    if current_vol <= low_thresh:
        # Probabilidade aumenta quanto mais perto de zero
        prob = 0.6 + 0.3 * (1 - (current_vol / max(low_thresh, 1e-9)))
        severity = "HIGH" if current_percentile < 5 else "MEDIUM"
        alert = {
            "type": "VOLATILITY_SQUEEZE",
            "level": "LOW",
            "severity": severity,
            "probability": min(prob, 0.95), # Cap em 95%
            "action": "WATCH_FOR_BREAKOUT",
            # 🔹 Adiciona valores formatados
            "volatility_current": current_vol,
            "volatility_threshold": low_thresh,
            "percentile": current_percentile,
            "description": f"Volatilidade {format_scientific(current_vol, decimals=5)} está no {current_percentile:.0f}º percentil (baixa)"
        }
        return _clean_alert_data(alert)
    
    if current_vol >= high_thresh:
        # Probabilidade aumenta quanto mais acima do threshold
        prob = 0.6 + 0.3 * ((current_vol - high_thresh) / max(high_thresh, 1e-9))
        severity = "HIGH" if current_percentile > 95 else "MEDIUM"
        alert = {
            "type": "VOLATILITY_SQUEEZE",
            "level": "HIGH",
            "severity": severity,
            "probability": min(prob, 0.95), # Cap em 95%
            "action": "WATCH_FOR_REVERSION",
            # 🔹 Adiciona valores formatados
            "volatility_current": current_vol,
            "volatility_threshold": high_thresh,
            "percentile": current_percentile,
            "description": f"Volatilidade {format_scientific(current_vol, decimals=5)} está no {current_percentile:.0f}º percentil (alta)"
        }
        return _clean_alert_data(alert)
    
    return None


def format_alert_message(alert: Dict[str, Any]) -> str:
    """
    Formata um alerta em mensagem legível com números formatados.
    """
    alert_type = alert.get('type', 'UNKNOWN')
    
    if alert_type == 'VOLUME_SPIKE':
        ratio = alert.get('threshold_exceeded', 0)
        current = format_large_number(alert.get('current_volume', 0))
        average = format_large_number(alert.get('average_volume', 0))
        severity = alert.get('severity', 'MEDIUM')
        
        return (
            f"🔊 VOLUME SPIKE ({severity})\n"
            f"Volume atual: {current}\n"
            f"Média: {average}\n"
            f"Ratio: {ratio:.1f}x acima da média"
        )
    
    elif alert_type == 'VOLATILITY_SQUEEZE':
        level = alert.get('level', 'UNKNOWN')
        current = format_scientific(alert.get('volatility_current', 0), decimals=5)
        threshold = format_scientific(alert.get('volatility_threshold', 0), decimals=5)
        percentile = alert.get('percentile', 0)
        action = alert.get('action', '')
        
        emoji = "📉" if level == "LOW" else "📈"
        
        return (
            f"{emoji} VOLATILITY SQUEEZE ({level})\n"
            f"Volatilidade: {current}\n"
            f"Limite {level.lower()}: {threshold}\n"
            f"Percentil: {percentile:.0f}%\n"
            f"Ação: {action.replace('_', ' ').title()}"
        )
    
    else:
        return f"⚠️ Alerta: {alert_type}"


def generate_alerts(
    price: float,
    support_resistance: Dict[str, Any],
    current_volume: float,
    average_volume: float,
    current_volatility: float,
    recent_volatilities: List[float],
    delta: float = 0.0,
    monitor: Optional[Any] = None,  # Removido monitor específico
    volume_threshold: float = 3.0,
    tolerance_pct: float = 0.001,
) -> List[Dict[str, Any]]:
    """
    Gera apenas alertas de volume e volatilidade com formatação adequada.
    REMOVIDO: Todos os alertas de suporte/resistência.
    
    Args:
        price: Preço atual
        support_resistance: Dict com níveis (ignorado)
        current_volume: Volume atual
        average_volume: Volume médio
        current_volatility: Volatilidade atual
        recent_volatilities: Lista de volatilidades recentes
        delta: Delta atual (opcional)
        monitor: Monitor de alertas (opcional)
        volume_threshold: Fator multiplicador para spike de volume
        tolerance_pct: Tolerância percentual (não usado)
        
    Returns:
        Lista de alertas formatados
    """
    alerts = []

    # Spikes de volume
    vol_alert = detect_volume_spike(current_volume, average_volume, volume_threshold)
    if vol_alert:
        alerts.append(vol_alert)

    # Volatility squeeze
    vol_squeeze_alert = detect_volatility_squeeze(current_volatility, recent_volatilities)
    if vol_squeeze_alert:
        alerts.append(vol_squeeze_alert)

    # 🔹 Log de debug com valores formatados
    if alerts:
        import logging
        for alert in alerts:
            msg = format_alert_message(alert)
            logging.info(f"Alerta gerado:\n{msg}")

    return alerts


def create_alert_summary(alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Cria um sumário dos alertas com estatísticas formatadas.
    """
    if not alerts:
        return {
            "total_alerts": 0,
            "by_type": {},
            "summary": "Nenhum alerta ativo"
        }
    
    summary = {
        "total_alerts": len(alerts),
        "by_type": {},
        "high_severity": 0,
        "medium_severity": 0,
        "low_severity": 0
    }
    
    for alert in alerts:
        alert_type = alert.get('type', 'UNKNOWN')
        severity = alert.get('severity', 'LOW')
        
        # Conta por tipo
        if alert_type not in summary['by_type']:
            summary['by_type'][alert_type] = 0
        summary['by_type'][alert_type] += 1
        
        # Conta por severidade
        if severity == 'HIGH':
            summary['high_severity'] += 1
        elif severity == 'MEDIUM':
            summary['medium_severity'] += 1
        else:
            summary['low_severity'] += 1
    
    # 🔹 Cria texto resumo formatado
    summary_parts = []
    if summary['high_severity'] > 0:
        summary_parts.append(f"{summary['high_severity']} alta prioridade")
    if summary['medium_severity'] > 0:
        summary_parts.append(f"{summary['medium_severity']} média prioridade")
    if summary['low_severity'] > 0:
        summary_parts.append(f"{summary['low_severity']} baixa prioridade")
    
    summary['summary'] = f"Total: {len(alerts)} alertas ({', '.join(summary_parts)})"
    
    return summary


# 🔹 Função auxiliar para serialização JSON
def serialize_alerts_for_json(alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prepara alertas para serialização JSON, garantindo que números estejam formatados.
    """
    serialized = []
    
    for alert in alerts:
        clean_alert = {}
        
        for key, value in alert.items():
            # Mantém valores numéricos originais mas adiciona versões display
            if isinstance(value, (int, float)):
                clean_alert[key] = value
                
                # Adiciona versão formatada se ainda não existir
                display_key = f"{key}_display"
                if display_key not in alert:
                    if 'volume' in key:
                        clean_alert[display_key] = format_large_number(value)
                    elif 'volatility' in key:
                        clean_alert[display_key] = format_scientific(value, decimals=5)
                    elif 'ratio' in key or 'threshold' in key:
                        clean_alert[display_key] = f"{value:.2f}"
                    elif 'percentile' in key or 'probability' in key:
                        clean_alert[display_key] = format_percent(value)
                    else:
                        clean_alert[display_key] = str(value)
            else:
                clean_alert[key] = value
        
        serialized.append(clean_alert)
    
    return serialized