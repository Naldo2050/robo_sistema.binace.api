"""
Monitor de bias direcional do modelo ML.

Detecta quando o modelo prediz a mesma direção em muitas janelas consecutivas,
o que pode indicar overfit ou features insuficientes.

Uso:
    from ml.bias_monitor import get_bias_monitor
    monitor = get_bias_monitor()
    monitor.record(prob_up=0.93, features={"rsi": 28, ...})
    adjustment = monitor.get_confidence_adjustment()
"""

import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BiasAlert:
    """Um alerta de bias emitido."""
    timestamp: float
    direction: str           # "bullish" ou "bearish"
    consecutive_pct: float   # % de predições na mesma direção
    avg_prob: float
    window_size: int
    features_snapshot: dict


@dataclass
class ModelBiasMonitor:
    """
    Monitora bias direcional do modelo XGBoost.

    Funcionalidades:
    - Detecta quando modelo prediz mesma direção N vezes seguidas
    - Calcula fator de ajuste de confiança quando bias detectado
    - Compara predição ML vs sinais macro para detectar incoerência
    - Emite alerta com features snapshot para debugging
    """

    # Configuração
    window_size: int = 20              # Janelas para avaliar
    bias_threshold: float = 0.85       # % mínimo para considerar bias
    extreme_prob_threshold: float = 0.90  # prob muito alta/baixa
    alert_cooldown_seconds: float = 600.0  # Min 10min entre alertas

    # Estado interno
    _predictions: deque = field(default_factory=lambda: deque(maxlen=50), repr=False)
    _last_alert_time: float = field(default=0.0, repr=False)
    _total_predictions: int = field(default=0, repr=False)
    _alerts_emitted: int = field(default=0, repr=False)
    _bias_active: bool = field(default=False, repr=False)

    def __post_init__(self):
        self._predictions = deque(maxlen=self.window_size)

    def record(
        self,
        prob_up: float,
        features: Optional[dict] = None,
        macro_context: Optional[dict] = None,
    ) -> Optional[BiasAlert]:
        """
        Registra uma predição e verifica bias.

        Args:
            prob_up: Probabilidade de alta (0.0 a 1.0)
            features: Features usadas na predição
            macro_context: Contexto macro (tendências por TF)

        Returns:
            BiasAlert se bias detectado, None caso contrário
        """
        self._total_predictions += 1
        self._predictions.append({
            "prob_up": prob_up,
            "timestamp": time.time(),
            "features": features or {},
            "macro": macro_context,
        })

        if len(self._predictions) < self.window_size:
            return None

        # Análise de bias
        alert = self._check_bias(features or {}, macro_context)

        # Verificar incoerência ML vs Macro
        if macro_context and features:
            self._check_macro_incoherence(prob_up, features, macro_context)

        return alert

    def _check_bias(
        self, features: dict, macro_context: Optional[dict]
    ) -> Optional[BiasAlert]:
        """Verifica se há bias direcional."""
        probs = [p["prob_up"] for p in self._predictions]

        bullish_count = sum(1 for p in probs if p > 0.6)
        bearish_count = sum(1 for p in probs if p < 0.4)
        total = len(probs)

        bullish_pct = bullish_count / total
        bearish_pct = bearish_count / total
        avg_prob = sum(probs) / total

        # Detectar bias bullish
        if bullish_pct >= self.bias_threshold:
            return self._emit_alert(
                direction="bullish",
                consecutive_pct=bullish_pct,
                avg_prob=avg_prob,
                features=features,
            )

        # Detectar bias bearish
        if bearish_pct >= self.bias_threshold:
            return self._emit_alert(
                direction="bearish",
                consecutive_pct=bearish_pct,
                avg_prob=avg_prob,
                features=features,
            )

        # Bias resolvido
        if self._bias_active:
            self._bias_active = False
            logger.info(
                "ML bias resolvido | "
                "bullish=%.0f%%, bearish=%.0f%%",
                bullish_pct * 100, bearish_pct * 100,
            )

        return None

    def _emit_alert(
        self,
        direction: str,
        consecutive_pct: float,
        avg_prob: float,
        features: dict,
    ) -> Optional[BiasAlert]:
        """Emite alerta de bias (respeitando cooldown)."""
        now = time.time()

        # Cooldown
        if now - self._last_alert_time < self.alert_cooldown_seconds:
            return None

        self._last_alert_time = now
        self._alerts_emitted += 1
        self._bias_active = True

        alert = BiasAlert(
            timestamp=now,
            direction=direction,
            consecutive_pct=consecutive_pct,
            avg_prob=avg_prob,
            window_size=self.window_size,
            features_snapshot=features,
        )

        logger.warning(
            "ML BIAS DETECTED: %.0f%% %s in last %d windows | "
            "avg_prob=%.3f | RSI=%s | BB_width=%s | "
            "Consider retraining with recent data",
            consecutive_pct * 100,
            direction,
            self.window_size,
            avg_prob,
            features.get("rsi", "?"),
            features.get("bb_w", "?"),
        )

        return alert

    def _check_macro_incoherence(
        self,
        prob_up: float,
        features: dict,
        macro_context: dict,
    ):
        """
        Verifica incoerência entre ML e sinais macro.
        Ex: ML diz 93% bullish mas RSI=28 e todas TFs em baixa.
        """
        rsi = features.get("rsi", 50)
        trend_signals = macro_context if isinstance(macro_context, dict) else {}

        # Contar TFs em baixa
        bearish_tfs = sum(
            1 for v in trend_signals.values()
            if isinstance(v, str) and v.lower() in ("baixa", "bearish", "down")
        )
        total_tfs = max(1, len(trend_signals))
        bearish_ratio = bearish_tfs / total_tfs

        # ML bullish + macro bearish = incoerência
        if prob_up > 0.85 and rsi < 35 and bearish_ratio > 0.7:
            logger.warning(
                "ML/MACRO INCOHERENCE: ML=%.0f%% bullish, "
                "RSI=%.0f (oversold), bearish TFs=%d/%d | "
                "Model may need macro features",
                prob_up * 100, rsi, bearish_tfs, total_tfs,
            )

        # ML bearish + macro bullish = incoerência
        if prob_up < 0.15 and rsi > 65 and bearish_ratio < 0.3:
            logger.warning(
                "ML/MACRO INCOHERENCE: ML=%.0f%% bearish, "
                "RSI=%.0f (overbought), bullish TFs=%d/%d",
                prob_up * 100, rsi, total_tfs - bearish_tfs, total_tfs,
            )

    def get_confidence_adjustment(self) -> dict:
        """
        Retorna dicionário com fator de ajuste e flag de bloqueio.

        Returns:
            {
                "factor": 0.5-1.0,  # fator multiplicador de confiança
                "block": bool,       # True se bias extremo (≥95%) → ML deve ser ignorado
                "detail": str,       # descrição do estado
            }
        """
        result = {"factor": 1.0, "block": False, "detail": "normal"}

        if len(self._predictions) < self.window_size:
            return result

        probs = [p["prob_up"] for p in self._predictions]
        bullish_pct = sum(1 for p in probs if p > 0.6) / len(probs)
        bearish_pct = sum(1 for p in probs if p < 0.4) / len(probs)

        max_directional = max(bullish_pct, bearish_pct)
        direction = "bullish" if bullish_pct >= bearish_pct else "bearish"

        # Bias extremo (≥95%): BLOQUEAR predições ML
        if max_directional >= 0.95:
            logger.warning(
                "ML BIAS EXTREME BLOCK: %.0f%% %s — ML predictions disabled",
                max_directional * 100, direction,
            )
            return {
                "factor": 0.0,
                "block": True,
                "detail": f"extreme_bias_{direction}_{max_directional:.0%}",
            }

        # Bias alto (≥threshold): reduzir confiança proporcionalmente
        if max_directional >= self.bias_threshold:
            excess = max_directional - self.bias_threshold
            max_excess = 1.0 - self.bias_threshold
            reduction = (excess / max_excess) * 0.5
            factor = max(0.5, 1.0 - reduction)
            return {
                "factor": factor,
                "block": False,
                "detail": f"bias_{direction}_{max_directional:.0%}_factor_{factor:.2f}",
            }

        return result

    def get_stats(self) -> dict:
        """Estatísticas do monitor."""
        if len(self._predictions) == 0:
            return {
                "total_predictions": 0,
                "bias_active": False,
                "confidence_adjustment": 1.0,
            }

        probs = [p["prob_up"] for p in self._predictions]
        return {
            "total_predictions": self._total_predictions,
            "window_size": len(self._predictions),
            "avg_prob_up": round(sum(probs) / len(probs), 4),
            "bullish_pct": round(
                sum(1 for p in probs if p > 0.6) / len(probs), 3
            ),
            "bearish_pct": round(
                sum(1 for p in probs if p < 0.4) / len(probs), 3
            ),
            "bias_active": self._bias_active,
            "alerts_emitted": self._alerts_emitted,
            "confidence_adjustment": self.get_confidence_adjustment()["factor"],
            "bias_blocked": self.get_confidence_adjustment()["block"],
        }


# ──────────────────────────────────────────────
# Singleton
# ──────────────────────────────────────────────

_monitor_instance: Optional[ModelBiasMonitor] = None


def get_bias_monitor(**kwargs) -> ModelBiasMonitor:
    """Retorna instância singleton."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = ModelBiasMonitor(**kwargs)
        logger.info(
            "ML Bias Monitor inicializado: window=%d, threshold=%.0f%%",
            _monitor_instance.window_size,
            _monitor_instance.bias_threshold * 100,
        )
    return _monitor_instance


def reset_bias_monitor():
    """Reset singleton (para testes)."""
    global _monitor_instance
    _monitor_instance = None
