"""
alert_engine.py v2.1.1 - CORREÇÃO DE VOLATILITY SQUEEZE COM ZEROS

Correções v2.1.1:
  ✅ Evita alertas de VOLATILITY_SQUEEZE quando:
      - histórico é insuficiente (poucos pontos)
      - toda a série de volatilidade está praticamente em zero (spread ~ 0)
      - thresholds (low/high) são ~0 e current_vol ~0
  ✅ Garante que não haja casos como: vol=0.0, thresh=0.0, percentile=100%, severity=CRITICAL
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

# 🔹 PARÂMETROS DE SEGURANÇA PARA SQUEEZE
MIN_VOL_POINTS = 10        # mínimo de pontos de histórico para considerar squeeze
MIN_VOL_SPREAD = 1e-8      # spread mínimo entre min/max(vols) para considerar variação real
MIN_VALID_VOL = 1e-10      # volatilidade mínima “não-zero”


def _clean_alert_data(alert: Dict[str, Any]) -> Dict[str, Any]:
    """
    Limpa e formata valores numéricos no alerta antes de retornar.
    Mantém os valores como números para processamento, mas adiciona versões formatadas.
    """
    cleaned = alert.copy()
    
    # Adiciona versões formatadas dos valores numéricos
    if 'threshold_exceeded' in cleaned:
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
    
    # Intensidade (0.0 a 1.0)
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
        intensity = min(1.0, (ratio - threshold_factor) / (10 - threshold_factor))
        
        alert = {
            "type": "VOLUME_SPIKE",
            "threshold_exceeded": round(ratio, 2),
            "severity": "HIGH" if ratio >= 5.0 else "MEDIUM",
            "intensity": round(intensity, 3),
            "duration": duration,
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
    
    CORREÇÕES v2.1.1:
      - Ignora casos de histórico insuficiente ou série quase toda zerada
      - Evita alertas com vol=0.0 e thresholds=0.0
    """
    # Nenhum histórico
    if not recent_vols:
        return None
    
    # Filtra valores válidos (finitos e >= 0)
    vols = np.array(
        [v for v in recent_vols if v is not None and np.isfinite(v) and v >= 0.0],
        dtype=float
    )
    if vols.size == 0:
        return None

    # Histórico insuficiente → sem squeeze confiável
    if vols.size < MIN_VOL_POINTS:
        logging.debug(
            "Volatility squeeze: histórico insuficiente (%d pontos, min=%d). Nenhum alerta.",
            vols.size, MIN_VOL_POINTS
        )
        return None

    # Spread muito pequeno → volatilidade praticamente constante (ex: tudo zero)
    vol_min = float(np.min(vols))
    vol_max = float(np.max(vols))
    vol_spread = vol_max - vol_min

    if vol_spread < MIN_VOL_SPREAD:
        logging.debug(
            "Volatility squeeze: spread de volatilidade muito baixo (min=%.3e, max=%.3e, spread=%.3e). "
            "Provavelmente primeira(s) janela(s) com vol≈0. Nenhum alerta.",
            vol_min, vol_max, vol_spread
        )
        return None

    # Volatilidade atual não positiva → não gera squeeze
    if not np.isfinite(current_vol) or current_vol <= 0.0:
        logging.debug(
            "Volatility squeeze: current_vol inválido ou não positivo (%.3e). Nenhum alerta.",
            current_vol
        )
        return None
    
    low_thresh = float(np.percentile(vols, low_percentile))
    high_thresh = float(np.percentile(vols, high_percentile))

    # Se thresholds também são praticamente zero, aborta
    if low_thresh < MIN_VALID_VOL and high_thresh < MIN_VALID_VOL:
        logging.debug(
            "Volatility squeeze: thresholds muito baixos (low=%.3e, high=%.3e). "
            "Interpretado como falta de histórico real. Nenhum alerta.",
            low_thresh, high_thresh
        )
        return None
    
    # Calcula percentil atual
    current_percentile = float(np.sum(vols <= current_vol) / len(vols) * 100.0)
    
    # ============================================================
    # VOLATILIDADE BAIXA (SQUEEZE - compressão)
    # ============================================================
    if current_vol <= low_thresh:
        # Intensidade normalizada (0.0 = no limite, 1.0 = muito comprimido)
        intensity = 1.0 - (current_vol / max(low_thresh, MIN_VALID_VOL))
        intensity = max(0.0, min(1.0, intensity))
        
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
        
        probability = 0.4 + (0.55 * intensity)  # 40% a 95%
        probability = min(0.95, max(0.4, probability))
        
        alert = {
            "type": "VOLATILITY_SQUEEZE",
            "volatility_state": "COMPRESSED",
            "severity": severity,
            "intensity": round(intensity, 3),
            "probability": round(probability, 3),
            "action": action,
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
        
        logging.info(
            "🔍 VOLATILITY SQUEEZE detectado: "
            "vol=%.6f, low_thresh=%.6f, percentile=%.1f%%, intensity=%.0f%%, severity=%s",
            current_vol, low_thresh, current_percentile, intensity*100, severity
        )
        
        return _clean_alert_data(alert)
    
    # ============================================================
    # VOLATILIDADE ALTA (EXPANSION - expansão)
    # ============================================================
    if current_vol >= high_thresh:
        intensity = (current_vol - high_thresh) / max(high_thresh, MIN_VALID_VOL)
        intensity = max(0.0, min(1.0, intensity))
        
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
        
        probability = 0.4 + (0.55 * intensity)  # 40% a 95%
        probability = min(0.95, max(0.4, probability))
        
        alert = {
            "type": "VOLATILITY_SQUEEZE",
            "volatility_state": "EXPANDED",
            "severity": severity,
            "intensity": round(intensity, 3),
            "probability": round(probability, 3),
            "action": action,
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
        
        logging.info(
            "🔍 VOLATILITY EXPANSION detectado: "
            "vol=%.6f, high_thresh=%.6f, percentile=%.1f%%, intensity=%.0f%%, severity=%s",
            current_vol, high_thresh, current_percentile, intensity*100, severity
        )
        
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
        intensity = alert.get('intensity', 0) * 100
        
        return (
            f"🔊 VOLUME SPIKE ({severity})\n"
            f"Volume atual: {current}\n"
            f"Média: {average}\n"
            f"Ratio: {ratio:.1f}x acima da média\n"
            f"Intensidade: {intensity:.0f}%"
        )
    
    elif alert_type == 'VOLATILITY_SQUEEZE':
        state = alert.get('volatility_state', 'UNKNOWN')
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
    """
    alerts = []

    vol_alert = detect_volume_spike(current_volume, average_volume, volume_threshold)
    if vol_alert:
        alerts.append(vol_alert)

    vol_squeeze_alert = detect_volatility_squeeze(current_volatility, recent_volatilities)
    if vol_squeeze_alert:
        alerts.append(vol_squeeze_alert)

    if alerts:
        for alert in alerts:
            msg = format_alert_message(alert)
            logging.info(f"📢 Alerta gerado:\n{msg}\n" + "="*60)
            _validate_alert_consistency(alert)

    return alerts


def _validate_alert_consistency(alert: Dict[str, Any]) -> None:
    """
    Validação de consistência entre campos do alerta.
    """
    try:
        alert_type = alert.get('type', '')
        severity = alert.get('severity', '')
        intensity = alert.get('intensity', 0)
        probability = alert.get('probability', 0)
        
        if alert_type == 'VOLATILITY_SQUEEZE':
            state = alert.get('volatility_state', '')
            
            if intensity <= 0.3 and severity in ['HIGH', 'CRITICAL']:
                logging.warning(
                    "⚠️ INCONSISTÊNCIA: %s com intensidade baixa (%.0f%%) mas severity=%s. "
                    "Esperava-se MEDIUM ou LOW.",
                    alert_type, intensity*100, severity
                )
            
            if intensity > 0.7 and severity == 'LOW':
                logging.warning(
                    "⚠️ INCONSISTÊNCIA: %s com intensidade alta (%.0f%%) mas severity=%s. "
                    "Esperava-se HIGH ou CRITICAL.",
                    alert_type, intensity*100, severity
                )
            
            expected_prob = 0.4 + (0.55 * intensity)
            if abs(probability - expected_prob) > 0.1:
                logging.warning(
                    "⚠️ INCONSISTÊNCIA: Probabilidade (%.2f) não corresponde à intensidade (%.2f). "
                    "Esperava-se ~%.2f",
                    probability, intensity, expected_prob
                )
            
            logging.debug(
                "✅ Validação OK: %s | State: %s | Severity: %s | Intensity: %.0f%% | Prob: %.0f%%",
                alert_type, state, severity, intensity*100, probability*100
            )
    
    except Exception as e:
        logging.error(f"Erro ao validar alerta: {e}")


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
        "critical_severity": 0,
        "high_severity": 0,
        "medium_severity": 0,
        "low_severity": 0,
        "avg_intensity": 0.0,
    }
    
    intensities = []
    
    for alert in alerts:
        alert_type = alert.get('type', 'UNKNOWN')
        severity = alert.get('severity', 'LOW')
        intensity = alert.get('intensity', 0)
        
        if alert_type not in summary['by_type']:
            summary['by_type'][alert_type] = 0
        summary['by_type'][alert_type] += 1
        
        if severity == 'CRITICAL':
            summary['critical_severity'] += 1
        elif severity == 'HIGH':
            summary['high_severity'] += 1
        elif severity == 'MEDIUM':
            summary['medium_severity'] += 1
        else:
            summary['low_severity'] += 1
        
        if intensity > 0:
            intensities.append(intensity)
    
    if intensities:
        summary['avg_intensity'] = round(np.mean(intensities), 3)
    
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


def serialize_alerts_for_json(alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prepara alertas para serialização JSON, garantindo que números estejam formatados.
    """
    serialized = []
    
    for alert in alerts:
        clean_alert = {}
        
        for key, value in alert.items():
            if isinstance(value, (int, float)):
                clean_alert[key] = value
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