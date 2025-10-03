"""
alert_engine.py

Motor simplificado para detecção de condições de mercado relevantes.
Apenas detecta spikes de volume e compressão de volatilidade.
REMOVEU todos os alertas de suporte/resistência conforme solicitado.
"""

import numpy as np
from typing import List, Dict, Any, Optional

def detect_volume_spike(
    current_volume: float,
    average_volume: float,
    threshold_factor: float = 3.0,
    duration: str = "5_MINUTES",
) -> Optional[Dict[str, Any]]:
    """Volume spike detection."""
    if average_volume <= 0:
        return None
    ratio = current_volume / average_volume
    if ratio >= threshold_factor:
        return {
            "type": "VOLUME_SPIKE",
            "threshold_exceeded": round(ratio, 2),
            "severity": "MEDIUM" if ratio < 5 else "HIGH",
            "duration": duration,
        }
    return None

def detect_volatility_squeeze(
    current_vol: float,
    recent_vols: List[float],
    low_percentile: float = 10.0,
    high_percentile: float = 90.0,
) -> Optional[Dict[str, Any]]:
    """Volatility squeeze detection."""
    if not recent_vols:
        return None
    vols = np.array([v for v in recent_vols if v is not None])
    if vols.size == 0:
        return None
    low_thresh = np.percentile(vols, low_percentile)
    high_thresh = np.percentile(vols, high_percentile)
    if current_vol <= low_thresh:
        return {
            "type": "VOLATILITY_SQUEEZE",
            "level": "LOW",
            "severity": "MEDIUM",
            "probability": 0.6,
            "action": "WATCH_FOR_BREAKOUT",
        }
    if current_vol >= high_thresh:
        return {
            "type": "VOLATILITY_SQUEEZE",
            "level": "HIGH",
            "severity": "MEDIUM",
            "probability": 0.6,
            "action": "WATCH_FOR_REVERSION",
        }
    return None

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
    Gera apenas alertas de volume e volatilidade.
    REMOVIDO: Todos os alertas de suporte/resistência.
    """
    alerts = []

    # Spikes de volume
    vol_alert = detect_volume_spike(current_volume, average_volume, volume_threshold)
    if vol_alert:
        alerts.append(vol_alert)

    # Volatility squeeze
    vol_alert2 = detect_volatility_squeeze(current_volatility, recent_volatilities)
    if vol_alert2:
        alerts.append(vol_alert2)

    return alerts