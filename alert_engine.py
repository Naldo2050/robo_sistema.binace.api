"""
alert_engine.py v2.1.0 - CORREÇÃO DE SEVERIDADE

Motor simplificado para detecção de condições de mercado relevantes.
Apenas detecta spikes de volume e compressão de volatilidade.
REMOVEU todos os alertas de suporte/resistência conforme solicitado.

🔹 CORREÇÕES v2.1.0:
  ✅ Severidade calculada corretamente baseada na intensidade do squeeze
  ✅ Campos renomeados para clareza (level → volatility_state)
  ✅ Descrições mais claras e informativas
  ✅ Validação de consistência entre dados e severidade
  ✅ Logs detalhados de alertas gerados
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging

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
    
    if 'probability' in cleaned:
        cleaned['probability_display'] = format_percent(cleaned['probability'])
    
    # 🆕 Adiciona campo numérico de intensidade (0.0 a 1.0)
    if 'intensity' in cleaned:
        cleaned['intensity_display'] = f"{cleaned['intensity']*100:.1f}%"
    
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
        # 🆕 Calcula intensidade normalizada
        intensity = min(1.0, (ratio - threshold_factor) / (10 - threshold_factor))
        
        alert = {
            "type": "VOLUME_SPIKE",
            "threshold_exceeded": round(ratio, 2),
            "severity": "HIGH" if ratio >= 5.0 else "MEDIUM",
            "intensity": round(intensity, 3),  # 🆕 0.0 a 1.0
            "duration": duration,
            # Valores brutos
            "current_volume": current_volume,
            "average_volume": average_volume,
            "volume_ratio": ratio,
            "description": (
                f"Volume atual {format_large_number(current_volume)} é "
                f"{ratio:.1f}x a média de {format_large_number(average_volume)} "
                f"(intensidade: {intensity*100:.0f}%)"
            )
        }
        return _clean_alert_data(alert)
    
    return None


def detect_volatility_squeeze(
    current_vol: float,
    recent_vols: List[float],
    low_percentile: float = 10.0,
    high_percentile: float = 90.0,
) -> Optional[Dict[str, Any]]:
    """
    Volatility squeeze detection com formatação adequada.
    
    🔹 CORREÇÕES v2.1.0:
      - Campo "level" renomeado para "volatility_state" (mais claro)
      - Severidade calculada baseada na intensidade do squeeze/expansion
      - Intensidade numérica adicionada (0.0 a 1.0)
      - Descrição mais clara sobre o significado
    """
    if not recent_vols:
        return None
    
    vols = np.array([v for v in recent_vols if v is not None])
    if vols.size == 0:
        return None
    
    low_thresh = np.percentile(vols, low_percentile)
    high_thresh = np.percentile(vols, high_percentile)
    
    # 🔹 Calcula percentil atual
    current_percentile = np.sum(vols <= current_vol) / len(vols) * 100
    
    # ============================================================
    # VOLATILIDADE BAIXA (SQUEEZE - compressão)
    # ============================================================
    if current_vol <= low_thresh:
        # 🆕 Calcula intensidade normalizada (0.0 = no limite, 1.0 = zero absoluto)
        # Quanto mais perto de zero, maior a intensidade
        intensity = 1.0 - (current_vol / max(low_thresh, 1e-9))
        intensity = max(0.0, min(1.0, intensity))  # Clamp [0, 1]
        
        # 🆕 CORREÇÃO: Severidade baseada na INTENSIDADE, não percentil
        # Intensidade > 0.7 (muito perto de zero) = CRITICAL
        # Intensidade > 0.5 = HIGH
        # Intensidade > 0.3 = MEDIUM
        # Intensidade <= 0.3 = LOW
        if intensity > 0.7:
            severity = "CRITICAL"
            action = "PREPARE_FOR_BREAKOUT_IMMINENT"
        elif intensity > 0.5:
            severity = "HIGH"
            action = "WATCH_FOR_BREAKOUT_LIKELY"
        elif intensity > 0.3:
            severity = "MEDIUM"
            action = "MONITOR_FOR_BREAKOUT"
        else:
            severity = "LOW"
            action = "WATCH_FOR_EXPANSION"
        
        # 🆕 Probabilidade baseada na intensidade
        # Intensidade alta = maior probabilidade de breakout
        probability = 0.4 + (0.55 * intensity)  # 40% a 95%
        probability = min(0.95, max(0.4, probability))
        
        alert = {
            "type": "VOLATILITY_SQUEEZE",
            "volatility_state": "COMPRESSED",  # 🆕 Renomeado de "level"
            "severity": severity,
            "intensity": round(intensity, 3),  # 🆕 0.0 a 1.0
            "probability": round(probability, 3),
            "action": action,
            # Valores brutos
            "volatility_current": current_vol,
            "volatility_threshold": low_thresh,
            "percentile": round(current_percentile, 1),
            "description": (
                f"SQUEEZE: Volatilidade {format_scientific(current_vol, decimals=5)} "
                f"está no {current_percentile:.0f}º percentil (muito baixa). "
                f"Intensidade do squeeze: {intensity*100:.0f}%. "
                f"Probabilidade de breakout: {probability*100:.0f}%. "
                f"Ação: {action.replace('_', ' ').title()}"
            )
        }
        
        # 🆕 Log detalhado
        logging.info(
            f"🔍 VOLATILITY SQUEEZE detectado: "
            f"vol={current_vol:.6f}, thresh={low_thresh:.6f}, "
            f"percentile={current_percentile:.1f}%, "
            f"intensity={intensity*100:.0f}%, severity={severity}"
        )
        
        return _clean_alert_data(alert)
    
    # ============================================================
    # VOLATILIDADE ALTA (EXPANSION - expansão)
    # ============================================================
    if current_vol >= high_thresh:
        # 🆕 Calcula intensidade normalizada (0.0 = no limite, 1.0 = muito acima)
        intensity = (current_vol - high_thresh) / max(high_thresh, 1e-9)
        intensity = max(0.0, min(1.0, intensity))  # Clamp [0, 1]
        
        # 🆕 CORREÇÃO: Severidade baseada na INTENSIDADE
        if intensity > 0.7:
            severity = "CRITICAL"
            action = "EXPECT_REVERSION_IMMINENT"
        elif intensity > 0.5:
            severity = "HIGH"
            action = "WATCH_FOR_REVERSION_LIKELY"
        elif intensity > 0.3:
            severity = "MEDIUM"
            action = "MONITOR_FOR_REVERSION"
        else:
            severity = "LOW"
            action = "WATCH_FOR_NORMALIZATION"
        
        # 🆕 Probabilidade baseada na intensidade
        probability = 0.4 + (0.55 * intensity)  # 40% a 95%
        probability = min(0.95, max(0.4, probability))
        
        alert = {
            "type": "VOLATILITY_SQUEEZE",
            "volatility_state": "EXPANDED",  # 🆕 Renomeado de "level"
            "severity": severity,
            "intensity": round(intensity, 3),  # 🆕 0.0 a 1.0
            "probability": round(probability, 3),
            "action": action,
            # Valores brutos
            "volatility_current": current_vol,
            "volatility_threshold": high_thresh,
            "percentile": round(current_percentile, 1),
            "description": (
                f"EXPANSION: Volatilidade {format_scientific(current_vol, decimals=5)} "
                f"está no {current_percentile:.0f}º percentil (muito alta). "
                f"Intensidade da expansão: {intensity*100:.0f}%. "
                f"Probabilidade de reversão: {probability*100:.0f}%. "
                f"Ação: {action.replace('_', ' ').title()}"
            )
        }
        
        # 🆕 Log detalhado
        logging.info(
            f"🔍 VOLATILITY EXPANSION detectado: "
            f"vol={current_vol:.6f}, thresh={high_thresh:.6f}, "
            f"percentile={current_percentile:.1f}%, "
            f"intensity={intensity*100:.0f}%, severity={severity}"
        )
        
        return _clean_alert_data(alert)
    
    return None


def format_alert_message(alert: Dict[str, Any]) -> str:
    """
    Formata um alerta em mensagem legível com números formatados.
    
    🆕 CORREÇÃO: Usa novos campos (volatility_state, intensity)
    """
    alert_type = alert.get('type', 'UNKNOWN')
    
    if alert_type == 'VOLUME_SPIKE':
        ratio = alert.get('threshold_exceeded', 0)
        current = format_large_number(alert.get('current_volume', 0))
        average = format_large_number(alert.get('average_volume', 0))
        severity = alert.get('severity', 'MEDIUM')
        intensity = alert.get('intensity', 0) * 100
        
        return (
            f"🔊 VOLUME SPIKE ({severity})\n"
            f"Volume atual: {current}\n"
            f"Média: {average}\n"
            f"Ratio: {ratio:.1f}x acima da média\n"
            f"Intensidade: {intensity:.0f}%"
        )
    
    elif alert_type == 'VOLATILITY_SQUEEZE':
        state = alert.get('volatility_state', 'UNKNOWN')  # 🆕 Renomeado
        current = format_scientific(alert.get('volatility_current', 0), decimals=5)
        threshold = format_scientific(alert.get('volatility_threshold', 0), decimals=5)
        percentile = alert.get('percentile', 0)
        intensity = alert.get('intensity', 0) * 100
        probability = alert.get('probability', 0) * 100
        action = alert.get('action', '')
        severity = alert.get('severity', 'MEDIUM')
        
        emoji = "📉" if state == "COMPRESSED" else "📈"
        state_label = "SQUEEZE" if state == "COMPRESSED" else "EXPANSION"
        
        return (
            f"{emoji} VOLATILITY {state_label} ({severity})\n"
            f"Estado: {state}\n"
            f"Volatilidade: {current}\n"
            f"Limite: {threshold}\n"
            f"Percentil: {percentile:.0f}%\n"
            f"Intensidade: {intensity:.0f}%\n"
            f"Probabilidade: {probability:.0f}%\n"
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
    monitor: Optional[Any] = None,
    volume_threshold: float = 3.0,
    tolerance_pct: float = 0.001,
) -> List[Dict[str, Any]]:
    """
    Gera apenas alertas de volume e volatilidade com formatação adequada.
    REMOVIDO: Todos os alertas de suporte/resistência.
    
    🔹 CORREÇÕES v2.1.0:
      - Alertas agora incluem campo "intensity" numérico (0.0 a 1.0)
      - Severidade calculada corretamente baseada na intensidade
      - Logs detalhados de cada alerta gerado
    
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
        for alert in alerts:
            msg = format_alert_message(alert)
            logging.info(f"📢 Alerta gerado:\n{msg}\n" + "="*60)
            
            # 🆕 Validação de consistência
            _validate_alert_consistency(alert)

    return alerts


def _validate_alert_consistency(alert: Dict[str, Any]) -> None:
    """
    🆕 Validação de consistência entre campos do alerta.
    
    Detecta e loga inconsistências como:
    - Intensidade muito baixa mas severidade alta
    - Probabilidade inconsistente com intensidade
    """
    try:
        alert_type = alert.get('type', '')
        severity = alert.get('severity', '')
        intensity = alert.get('intensity', 0)
        probability = alert.get('probability', 0)
        
        if alert_type == 'VOLATILITY_SQUEEZE':
            state = alert.get('volatility_state', '')
            
            # Validar: Intensidade vs Severidade
            if intensity <= 0.3 and severity in ['HIGH', 'CRITICAL']:
                logging.warning(
                    f"⚠️ INCONSISTÊNCIA: {alert_type} com intensidade baixa "
                    f"({intensity*100:.0f}%) mas severity={severity}. "
                    f"Esperava-se MEDIUM ou LOW."
                )
            
            if intensity > 0.7 and severity == 'LOW':
                logging.warning(
                    f"⚠️ INCONSISTÊNCIA: {alert_type} com intensidade alta "
                    f"({intensity*100:.0f}%) mas severity={severity}. "
                    f"Esperava-se HIGH ou CRITICAL."
                )
            
            # Validar: Intensidade vs Probabilidade
            expected_prob = 0.4 + (0.55 * intensity)
            if abs(probability - expected_prob) > 0.1:
                logging.warning(
                    f"⚠️ INCONSISTÊNCIA: Probabilidade ({probability:.2f}) "
                    f"não corresponde à intensidade ({intensity:.2f}). "
                    f"Esperava-se ~{expected_prob:.2f}"
                )
            
            # Log de validação bem-sucedida
            logging.debug(
                f"✅ Validação OK: {alert_type} | "
                f"State: {state} | Severity: {severity} | "
                f"Intensity: {intensity*100:.0f}% | Prob: {probability*100:.0f}%"
            )
    
    except Exception as e:
        logging.error(f"Erro ao validar alerta: {e}")


def create_alert_summary(alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Cria um sumário dos alertas com estatísticas formatadas.
    
    🆕 CORREÇÃO: Inclui estatísticas de intensidade média
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
        "critical_severity": 0,  # 🆕
        "high_severity": 0,
        "medium_severity": 0,
        "low_severity": 0,
        "avg_intensity": 0.0,  # 🆕
    }
    
    intensities = []
    
    for alert in alerts:
        alert_type = alert.get('type', 'UNKNOWN')
        severity = alert.get('severity', 'LOW')
        intensity = alert.get('intensity', 0)
        
        # Conta por tipo
        if alert_type not in summary['by_type']:
            summary['by_type'][alert_type] = 0
        summary['by_type'][alert_type] += 1
        
        # Conta por severidade
        if severity == 'CRITICAL':
            summary['critical_severity'] += 1
        elif severity == 'HIGH':
            summary['high_severity'] += 1
        elif severity == 'MEDIUM':
            summary['medium_severity'] += 1
        else:
            summary['low_severity'] += 1
        
        # Coleta intensidades
        if intensity > 0:
            intensities.append(intensity)
    
    # Calcula intensidade média
    if intensities:
        summary['avg_intensity'] = round(np.mean(intensities), 3)
    
    # 🔹 Cria texto resumo formatado
    summary_parts = []
    if summary['critical_severity'] > 0:
        summary_parts.append(f"{summary['critical_severity']} crítica")
    if summary['high_severity'] > 0:
        summary_parts.append(f"{summary['high_severity']} alta")
    if summary['medium_severity'] > 0:
        summary_parts.append(f"{summary['medium_severity']} média")
    if summary['low_severity'] > 0:
        summary_parts.append(f"{summary['low_severity']} baixa")
    
    summary['summary'] = (
        f"Total: {len(alerts)} alertas ({', '.join(summary_parts)}). "
        f"Intensidade média: {summary['avg_intensity']*100:.0f}%"
    )
    
    return summary


# 🔹 Função auxiliar para serialização JSON
def serialize_alerts_for_json(alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prepara alertas para serialização JSON, garantindo que números estejam formatados.
    
    🆕 CORREÇÃO: Trata novos campos (volatility_state, intensity)
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
                    elif 'percentile' in key or 'probability' in key or 'intensity' in key:
                        clean_alert[display_key] = format_percent(value)
                    else:
                        clean_alert[display_key] = str(value)
            else:
                clean_alert[key] = value
        
        serialized.append(clean_alert)
    
    return serialized