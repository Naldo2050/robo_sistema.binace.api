# ml/hybrid_decision.py
# -*- coding: utf-8 -*-

"""
Módulo de Decisão Híbrida: XGBoost + LLM.

Combina previsões do modelo quantitativo local (XGBoost) com análise da IA
generativa (Qwen/Groq) para melhorar precisão dos sinais de trading.

Modos de operação:
- llm_primary: IA decide, modelo como contexto (default)
- model_primary: Modelo decide, IA comenta
- ensemble: Ponderação de ambos

Uso:
    from ml.hybrid_decision import HybridDecisionMaker
    
    decision_maker = HybridDecisionMaker()
    final_result = decision_maker.fuse_decisions(ml_prediction, ai_result)
"""

import logging
from typing import Any, Dict, Optional, Literal
from dataclasses import dataclass

logger = logging.getLogger("HybridDecision")


# =============================================================================
# CONFIGURAÇÕES (importadas de config.py se disponível)
# =============================================================================

try:
    import config
    HYBRID_ENABLED = getattr(config, "HYBRID_ENABLED", True)
    HYBRID_MODE = getattr(config, "HYBRID_MODE", "llm_primary")
    HYBRID_MODEL_WEIGHT = getattr(config, "HYBRID_MODEL_WEIGHT", 0.6)
    HYBRID_LLM_WEIGHT = getattr(config, "HYBRID_LLM_WEIGHT", 0.4)
    HYBRID_MODEL_MIN_CONFIDENCE = getattr(config, "HYBRID_MODEL_MIN_CONFIDENCE", 0.6)
except ImportError:
    HYBRID_ENABLED = True
    HYBRID_MODE = "llm_primary"
    HYBRID_MODEL_WEIGHT = 0.6
    HYBRID_LLM_WEIGHT = 0.4
    HYBRID_MODEL_MIN_CONFIDENCE = 0.6


# =============================================================================
# TIPOS E CONSTANTES
# =============================================================================

VALID_ACTIONS = {"buy", "sell", "hold", "flat", "wait", "avoid"}
DIRECTIONAL_ACTIONS = {"buy", "sell"}

# Mapeamento de ação para score numérico
ACTION_TO_SCORE = {
    "buy": 1.0,
    "sell": -1.0,
    "hold": 0.0,
    "flat": 0.0,
    "wait": None,  # Não pontua
    "avoid": None,  # Não pontua
}

# Mapeamento de score para ação
def score_to_action(score: float, threshold: float = 0.3) -> str:
    """Converte score numérico para ação."""
    if score > threshold:
        return "buy"
    elif score < -threshold:
        return "sell"
    else:
        return "flat"


@dataclass
class DecisionResult:
    """Resultado da decisão híbrida."""
    action: str
    confidence: float
    sentiment: str
    rationale: str
    source: str  # "model", "llm", "ensemble"
    model_prob_up: Optional[float] = None
    model_confidence: Optional[float] = None
    llm_action: Optional[str] = None
    llm_confidence: Optional[float] = None
    ensemble_score: Optional[float] = None
    entry_zone: Optional[str] = None
    invalidation_zone: Optional[str] = None
    region_type: Optional[str] = None


# =============================================================================
# CLASSE PRINCIPAL
# =============================================================================

class HybridDecisionMaker:
    """
    Combina decisões do modelo XGBoost e IA LLM.
    
    Modos:
    - llm_primary: IA decide, modelo como contexto
    - model_primary: Modelo decide, IA comenta
    - ensemble: Ponderação 60/40 (configurável)
    """
    
    def __init__(
        self,
        mode: Optional[str] = None,
        model_weight: Optional[float] = None,
        llm_weight: Optional[float] = None,
        model_min_confidence: Optional[float] = None,
    ):
        self.mode = mode or HYBRID_MODE
        self.model_weight = model_weight or HYBRID_MODEL_WEIGHT
        self.llm_weight = llm_weight or HYBRID_LLM_WEIGHT
        self.model_min_confidence = model_min_confidence or HYBRID_MODEL_MIN_CONFIDENCE
        
        # Valida pesos
        if abs((self.model_weight + self.llm_weight) - 1.0) > 0.01:
            logger.warning(
                f"Pesos do ensemble não somam 1.0: {self.model_weight} + {self.llm_weight} = "
                f"{self.model_weight + self.llm_weight}. Normalizando."
            )
            total = self.model_weight + self.llm_weight
            self.model_weight /= total
            self.llm_weight /= total
        
        logger.info(
            f"🧠 HybridDecisionMaker inicializado: "
            f"mode={self.mode}, model_weight={self.model_weight:.0%}, "
            f"llm_weight={self.llm_weight:.0%}"
        )
    
    def fuse_decisions(
        self,
        ml_prediction: Optional[Dict[str, Any]],
        ai_result: Optional[Dict[str, Any]],
    ) -> DecisionResult:
        """
        Combina decisões do modelo e IA.
        
        Args:
            ml_prediction: Output do MLInferenceEngine.predict()
                - prob_up: float (0-1)
                - prob_down: float (0-1)
                - confidence: float (0-1)
                - status: "ok" | "error"
            
            ai_result: Output do AIAnalyzer.analyze() → structured
                - action: str
                - sentiment: str
                - confidence: float
                - rationale: str
                - entry_zone: Optional[str]
                - invalidation_zone: Optional[str]
        
        Returns:
            DecisionResult com ação final e metadados
        """
        # Extrai dados do modelo
        model_ok = (
            ml_prediction is not None 
            and ml_prediction.get("status") == "ok"
        )
        model_prob_up = ml_prediction.get("prob_up", 0.5) if model_ok else None
        model_confidence = ml_prediction.get("confidence", 0.0) if model_ok else None
        
        # Extrai dados da IA
        llm_ok = ai_result is not None and isinstance(ai_result, dict)
        llm_action = ai_result.get("action", "wait") if llm_ok else "wait"
        llm_sentiment = ai_result.get("sentiment", "neutral") if llm_ok else "neutral"
        llm_confidence = ai_result.get("confidence", 0.0) if llm_ok else 0.0
        llm_rationale = ai_result.get("rationale", "") if llm_ok else ""
        llm_entry_zone = ai_result.get("entry_zone") if llm_ok else None
        llm_invalidation = ai_result.get("invalidation_zone") if llm_ok else None
        llm_region_type = ai_result.get("region_type") if llm_ok else None
        
        # Decide baseado no modo
        if self.mode == "model_primary":
            result = self._model_primary(
                model_prob_up, model_confidence,
                llm_action, llm_sentiment, llm_confidence, llm_rationale,
                llm_entry_zone, llm_invalidation, llm_region_type
            )
        elif self.mode == "ensemble":
            result = self._ensemble(
                model_prob_up, model_confidence,
                llm_action, llm_sentiment, llm_confidence, llm_rationale,
                llm_entry_zone, llm_invalidation, llm_region_type
            )
        else:  # llm_primary (default)
            result = self._llm_primary(
                model_prob_up, model_confidence,
                llm_action, llm_sentiment, llm_confidence, llm_rationale,
                llm_entry_zone, llm_invalidation, llm_region_type
            )
        
        # Adiciona metadados
        result.model_prob_up = model_prob_up
        result.model_confidence = model_confidence
        result.llm_action = llm_action
        result.llm_confidence = llm_confidence
        
        # Log da decisão
        self._log_decision(result)
        
        return result
    
    def _llm_primary(
        self,
        model_prob_up: Optional[float],
        model_confidence: Optional[float],
        llm_action: str,
        llm_sentiment: str,
        llm_confidence: float,
        llm_rationale: str,
        entry_zone: Optional[str],
        invalidation_zone: Optional[str],
        region_type: Optional[str],
    ) -> DecisionResult:
        """
        Modo LLM Primary: IA decide, modelo como contexto.
        
        A IA tem a palavra final, mas leva em conta a previsão do modelo
        (que já foi injetada no prompt via ai_payload_builder).
        """
        return DecisionResult(
            action=llm_action,
            confidence=llm_confidence,
            sentiment=llm_sentiment,
            rationale=llm_rationale,
            source="llm",
            entry_zone=entry_zone,
            invalidation_zone=invalidation_zone,
            region_type=region_type,
        )
    
    def _model_primary(
        self,
        model_prob_up: Optional[float],
        model_confidence: Optional[float],
        llm_action: str,
        llm_sentiment: str,
        llm_confidence: float,
        llm_rationale: str,
        entry_zone: Optional[str],
        invalidation_zone: Optional[str],
        region_type: Optional[str],
    ) -> DecisionResult:
        """
        Modo Model Primary: Modelo decide, IA comenta.
        
        Se o modelo tem confiança suficiente, ele decide a ação.
        IA é usada apenas para rationale e zonas.
        Se modelo não tem confiança, delega para IA.
        """
        # Fallback para IA se modelo não disponível ou confiança baixa
        if model_prob_up is None or model_confidence is None:
            logger.warning("⚠️ Modelo não disponível, usando IA como fallback")
            return self._llm_primary(
                model_prob_up, model_confidence,
                llm_action, llm_sentiment, llm_confidence, llm_rationale,
                entry_zone, invalidation_zone, region_type
            )
        
        if model_confidence < self.model_min_confidence:
            logger.info(
                f"📊 Confiança do modelo baixa ({model_confidence:.0%} < "
                f"{self.model_min_confidence:.0%}), delegando para IA"
            )
            return self._llm_primary(
                model_prob_up, model_confidence,
                llm_action, llm_sentiment, llm_confidence, llm_rationale,
                entry_zone, invalidation_zone, region_type
            )
        
        # Modelo decide
        if model_prob_up > 0.6:
            action = "buy"
            sentiment = "bullish"
        elif model_prob_up < 0.4:
            action = "sell"
            sentiment = "bearish"
        else:
            action = "flat"
            sentiment = "neutral"
        
        # Combina confiança (modelo tem mais peso)
        combined_confidence = (
            0.7 * model_confidence + 
            0.3 * llm_confidence
        )
        
        return DecisionResult(
            action=action,
            confidence=combined_confidence,
            sentiment=sentiment,
            rationale=f"[Modelo XGBoost: prob_up={model_prob_up:.1%}] {llm_rationale}",
            source="model",
            entry_zone=entry_zone,
            invalidation_zone=invalidation_zone,
            region_type=region_type,
        )
    
    def _ensemble(
        self,
        model_prob_up: Optional[float],
        model_confidence: Optional[float],
        llm_action: str,
        llm_sentiment: str,
        llm_confidence: float,
        llm_rationale: str,
        entry_zone: Optional[str],
        invalidation_zone: Optional[str],
        region_type: Optional[str],
    ) -> DecisionResult:
        """
        Modo Ensemble: Ponderação de ambos.
        
        Converte outputs para scores numéricos e combina.
        """
        # Se IA diz wait/avoid, respeita (não é direcional)
        if llm_action in ("wait", "avoid"):
            return DecisionResult(
                action=llm_action,
                confidence=llm_confidence,
                sentiment=llm_sentiment,
                rationale=llm_rationale,
                source="llm",  # IA vetou
                entry_zone=entry_zone,
                invalidation_zone=invalidation_zone,
                region_type=region_type,
            )
        
        # Fallback para IA se modelo não disponível
        if model_prob_up is None:
            return self._llm_primary(
                model_prob_up, model_confidence,
                llm_action, llm_sentiment, llm_confidence, llm_rationale,
                entry_zone, invalidation_zone, region_type
            )
        
        # Converte para scores
        # Modelo: prob_up → score (-1 a +1)
        model_score = (model_prob_up - 0.5) * 2  # 0.5 → 0, 1.0 → 1, 0.0 → -1
        
        # IA: action → score
        llm_score = ACTION_TO_SCORE.get(llm_action, 0.0)
        if llm_score is None:
            llm_score = 0.0
        
        # Ponderação
        ensemble_score = (
            self.model_weight * model_score + 
            self.llm_weight * llm_score
        )
        
        # Converte score final para ação
        final_action = score_to_action(ensemble_score, threshold=0.2)
        
        # Sentiment baseado no score
        if ensemble_score > 0.1:
            sentiment = "bullish"
        elif ensemble_score < -0.1:
            sentiment = "bearish"
        else:
            sentiment = "neutral"
        
        # Confiança combinada
        llm_conf = llm_confidence or 0.0
        model_conf = model_confidence or 0.5
        combined_confidence = (
            self.model_weight * model_conf + 
            self.llm_weight * llm_conf
        )
        
        return DecisionResult(
            action=final_action,
            confidence=combined_confidence,
            sentiment=sentiment,
            rationale=f"[Ensemble: modelo={model_score:+.2f}, IA={llm_score:+.2f}] {llm_rationale}",
            source="ensemble",
            ensemble_score=ensemble_score,
            entry_zone=entry_zone,
            invalidation_zone=invalidation_zone,
            region_type=region_type,
        )
    
    def _log_decision(self, result: DecisionResult) -> None:
        """Loga a cadeia de decisão de forma clara."""
        # Formata modelo
        if result.model_prob_up is not None:
            model_str = f"prob_up={result.model_prob_up:.0%}, conf={result.model_confidence:.0%}"
            if result.model_prob_up > 0.6:
                model_action = "BUY"
            elif result.model_prob_up < 0.4:
                model_action = "SELL"
            else:
                model_action = "FLAT"
            model_str = f"{model_action} ({model_str})"
        else:
            model_str = "N/A"
        
        # Formata IA
        llm_str = f"{result.llm_action.upper()} (conf={result.llm_confidence:.0%})"
        
        # Formata final
        final_str = f"{result.action.upper()} (conf={result.confidence:.0%})"
        
        logger.info(
            f"🧠 DECISÃO HÍBRIDA [{self.mode}]:\n"
            f"   Modelo XGBoost: {model_str}\n"
            f"   IA Generativa:  {llm_str}\n"
            f"   → FINAL:        {final_str} (source: {result.source})"
        )


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def get_hybrid_decision_maker() -> HybridDecisionMaker:
    """Retorna instância singleton do HybridDecisionMaker."""
    global _DECISION_MAKER
    if "_DECISION_MAKER" not in globals() or _DECISION_MAKER is None:
        _DECISION_MAKER = HybridDecisionMaker()
    return _DECISION_MAKER


def fuse_decisions(
    ml_prediction: Optional[Dict[str, Any]],
    ai_result: Optional[Dict[str, Any]],
) -> DecisionResult:
    """Função de conveniência para fusão de decisões."""
    return get_hybrid_decision_maker().fuse_decisions(ml_prediction, ai_result)


def decision_to_ai_result(decision: DecisionResult) -> Dict[str, Any]:
    """
    Converte DecisionResult para formato compatível com AITradeAnalysis.
    
    Mantém compatibilidade com o contrato Pydantic existente.
    """
    return {
        "sentiment": decision.sentiment,
        "confidence": decision.confidence,
        "action": decision.action,
        "rationale": decision.rationale,
        "entry_zone": decision.entry_zone,
        "invalidation_zone": decision.invalidation_zone,
        "region_type": decision.region_type,
        # Metadados extras
        "_hybrid_source": decision.source,
        "_model_prob_up": decision.model_prob_up,
        "_model_confidence": decision.model_confidence,
        "_llm_action": decision.llm_action,
        "_llm_confidence": decision.llm_confidence,
        "_ensemble_score": decision.ensemble_score,
    }


# Inicializa singleton
_DECISION_MAKER: Optional[HybridDecisionMaker] = None
