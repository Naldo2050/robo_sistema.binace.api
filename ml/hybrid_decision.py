# ml/hybrid_decision.py
# -*- coding: utf-8 -*-

"""
Módulo de Decisão Híbrida: XGBoost + LLM.

v6 - Correções:
- Singleton real (não recria a cada janela)
- Detecção de modelo congelado funcional
- Penalidade de conflito preservada entre janelas
- Fallback para ML quando LLM está indisponível (modo mock)
"""

import logging
from typing import Any, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger("HybridDecision")


# =============================================================================
# CONFIGURAÇÕES
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

ACTION_TO_SCORE = {
    "buy": 1.0,
    "sell": -1.0,
    "hold": 0.0,
    "flat": 0.0,
    "wait": None,
    "avoid": None,
}

OPPOSING_ACTIONS = {
    ("buy", "sell"), ("sell", "buy"),
    ("long", "short"), ("short", "long"),
}


def score_to_action(score: float, threshold: float = 0.3) -> str:
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
    source: str
    model_prob_up: Optional[float] = None
    model_confidence: Optional[float] = None
    llm_action: Optional[str] = None
    llm_confidence: Optional[float] = None
    ensemble_score: Optional[float] = None
    entry_zone: Optional[str] = None
    invalidation_zone: Optional[str] = None
    region_type: Optional[str] = None
    conflict_detected: bool = False
    llm_is_fallback: bool = False


# =============================================================================
# CLASSE PRINCIPAL
# =============================================================================

class HybridDecisionMaker:
    """
    Combina decisões do modelo XGBoost e IA LLM.
    
    v6 Correções:
    - Estado persistente entre janelas (frozen detection)
    - Quando LLM está em fallback/mock, usa ML como decisor
    - Penalidade de conflito com threshold mínimo de WAIT
    """
    
    _instance: Optional['HybridDecisionMaker'] = None
    _initialized: bool = False
    
    def __new__(cls, *args, **kwargs):
        """Singleton real - garante mesma instância."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        mode: Optional[str] = None,
        model_weight: Optional[float] = None,
        llm_weight: Optional[float] = None,
        model_min_confidence: Optional[float] = None,
    ):
        # Evitar re-inicialização do singleton
        if HybridDecisionMaker._initialized:
            return
        
        self.mode = mode or HYBRID_MODE
        self.model_weight = model_weight or HYBRID_MODEL_WEIGHT
        self.llm_weight = llm_weight or HYBRID_LLM_WEIGHT
        self.model_min_confidence = model_min_confidence or HYBRID_MODEL_MIN_CONFIDENCE
        
        # Valida pesos
        total = self.model_weight + self.llm_weight
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Pesos não somam 1.0: {total}. Normalizando.")
            self.model_weight /= total
            self.llm_weight /= total
        
        # ── Estado persistente (sobrevive entre janelas) ──
        self._last_model_prob: Optional[float] = None
        self._frozen_count: int = 0
        self._decision_count: int = 0
        self._conflict_count: int = 0
        self._model_disabled_count: int = 0
        
        HybridDecisionMaker._initialized = True
        
        logger.info(
            f"🧠 HybridDecisionMaker inicializado: "
            f"mode={self.mode}, model_weight={self.model_weight:.0%}, "
            f"llm_weight={self.llm_weight:.0%}"
        )
    
    def _check_frozen_model(self, prob_up: float) -> bool:
        """
        Detecta se o modelo está congelado (mesma saída repetida).
        Retorna True se modelo deve ser ignorado.
        """
        if self._last_model_prob is not None:
            if abs(prob_up - self._last_model_prob) < 1e-7:
                self._frozen_count += 1
            else:
                self._frozen_count = 0
        
        self._last_model_prob = prob_up
        
        if self._frozen_count >= 3:
            logger.error(
                f"🚨 ML MODEL FROZEN: prob_up={prob_up:.6f} "
                f"por {self._frozen_count + 1} janelas consecutivas. "
                f"Ignorando modelo (provavelmente features defaulting)."
            )
            self._model_disabled_count += 1
            return True
        
        return False
    
    def _check_model_validity(self, prob_up: float, confidence: float) -> bool:
        """
        Verifica se a predição do modelo é válida/confiável.
        Retorna False se modelo deve ser ignorado.
        """
        # Extremamente confident (provavelmente bug)
        if prob_up > 0.95 or prob_up < 0.05:
            logger.warning(
                f"[ML_EXTREME] prob={prob_up:.4f} é extrema, "
                f"provavelmente features ruins. Ignorando."
            )
            return False
        
        # Confiança muito baixa
        if confidence < 0.10:
            logger.warning(f"[ML_LOW_CONF] confidence={confidence:.4f} < 0.10. Ignorando.")
            return False
        
        return True
    
    def _detect_conflict(self, ml_action: str, llm_action: str) -> bool:
        """Detecta conflito direcional entre ML e LLM."""
        return (ml_action.lower(), llm_action.lower()) in OPPOSING_ACTIONS
    
    def _is_llm_fallback(self, ai_result: Optional[Dict[str, Any]]) -> bool:
        """Verifica se o resultado da IA é um fallback (mock/erro)."""
        if ai_result is None:
            return True
        if ai_result.get("_is_fallback", False):
            return True
        if ai_result.get("_fallback_reason"):
            return True
        if ai_result.get("confidence", 0) == 0.0 and ai_result.get("action") == "wait":
            rationale = ai_result.get("rationale", "")
            if "error" in rationale.lower() or "unavailable" in rationale.lower():
                return True
        return False
    
    def fuse_decisions(
        self,
        ml_prediction: Optional[Dict[str, Any]],
        ai_result: Optional[Dict[str, Any]],
    ) -> DecisionResult:
        """
        Combina decisões do modelo e IA.
        
        v6 MELHORIAS:
        1. Detecta LLM em modo fallback → usa ML como decisor
        2. Detecta modelo congelado → ignora ML
        3. Conflito direcional → penalidade de confiança
        4. Conflito severo (confiança similar) → força WAIT
        """
        self._decision_count += 1
        
        # ── 1. Extrai dados do modelo ──
        model_ok = False
        model_prob_up: Optional[float] = None
        model_confidence: Optional[float] = None

        if isinstance(ml_prediction, dict) and ml_prediction.get("status") == "ok":
            model_ok = True

            # prob_up
            prob_raw = ml_prediction.get("prob_up", 0.5)
            try:
                model_prob_up = float(prob_raw)
            except (TypeError, ValueError):
                model_prob_up = 0.5

            # confidence
            conf_raw = ml_prediction.get("confidence", 0.0)
            try:
                model_confidence = float(conf_raw)
            except (TypeError, ValueError):
                model_confidence = 0.0
        
        # Verificar se modelo está congelado
        if model_ok and model_prob_up is not None:
            if self._check_frozen_model(model_prob_up):
                model_ok = False
            elif model_confidence is not None and not self._check_model_validity(model_prob_up, model_confidence):
                model_ok = False
        
        # ── 2. Extrai dados da IA ──
        llm_ok = ai_result is not None and isinstance(ai_result, dict)
        llm_action: str = "wait"
        llm_sentiment: str = "neutral"
        llm_confidence: float = 0.0
        llm_rationale: str = ""
        if llm_ok and ai_result is not None:
            llm_action = ai_result.get("action", "wait") or "wait"
            llm_sentiment = ai_result.get("sentiment", "neutral") or "neutral"
            llm_confidence = float(ai_result.get("confidence", 0.0) or 0.0)
            llm_rationale = ai_result.get("rationale", "") or ""
        llm_is_fallback = self._is_llm_fallback(ai_result)
        
        # ── 3. Determinar ações direcionais ──
        ml_action = "wait"
        if model_ok and model_prob_up is not None:
            if model_prob_up > 0.6:
                ml_action = "buy"
            elif model_prob_up < 0.4:
                ml_action = "sell"
            else:
                ml_action = "flat"
        
        # ── 4. NOVO: Se LLM está em fallback, usar ML como decisor ──
        if llm_is_fallback and model_ok and model_prob_up is not None:
            logger.info(
                f"🔄 LLM indisponível (fallback), usando ML como decisor: "
                f"prob_up={model_prob_up:.4f}, action={ml_action}"
            )
            result = self._model_only_decision(
                model_prob_up, model_confidence or 0.0, ml_action,
                ai_result
            )
            result.llm_is_fallback = True
            result.llm_action = llm_action
            result.llm_confidence = llm_confidence
            result.model_prob_up = model_prob_up
            result.model_confidence = model_confidence
            self._log_decision(result)
            return result
        
        # ── 5. Se ambos indisponíveis, WAIT ──
        if not model_ok and llm_is_fallback:
            logger.warning("⚠️ Nem ML nem LLM disponíveis. Forçando WAIT.")
            result = DecisionResult(
                action="wait",
                confidence=0.0,
                sentiment="neutral",
                rationale="Sem dados: ML congelado/indisponível e LLM em fallback",
                source="none",
                llm_is_fallback=True,
            )
            self._log_decision(result)
            return result
        
        # ── 6. Detectar conflito direcional ──
        is_conflict = False
        if model_ok and ml_action in DIRECTIONAL_ACTIONS and llm_action in DIRECTIONAL_ACTIONS:
            is_conflict = self._detect_conflict(ml_action, llm_action)
        
        # ── 7. Decidir baseado no modo ──
        if self.mode == "model_primary":
            result = self._model_primary(
                model_prob_up if model_ok else None,
                model_confidence if model_ok else None,
                llm_action, llm_sentiment, llm_confidence, llm_rationale,
                ai_result
            )
        elif self.mode == "ensemble":
            result = self._ensemble(
                model_prob_up if model_ok else None,
                model_confidence if model_ok else None,
                llm_action, llm_sentiment, llm_confidence, llm_rationale,
                ai_result
            )
        else:  # llm_primary
            result = self._llm_primary(
                model_prob_up if model_ok else None,
                model_confidence if model_ok else None,
                llm_action, llm_sentiment, llm_confidence, llm_rationale,
                ai_result
            )
        
        # ── 8. Aplicar penalidade de conflito ──
        if is_conflict:
            self._conflict_count += 1
            result.conflict_detected = True
            old_conf = result.confidence
            
            # Calcular severidade do conflito
            conf_diff = abs((model_confidence or 0) - llm_confidence)
            
            if conf_diff < 0.15:
                # Conflito severo: ambos confiantes em direções opostas
                result.action = "wait"
                result.confidence = max(old_conf * 0.4, 0.0)
                result.source += "_severe_conflict"
                result.rationale = (
                    f"⚠️ CONFLITO SEVERO: ML({ml_action}/{(model_confidence or 0.0):.0%}) "
                    f"vs LLM({llm_action}/{llm_confidence:.0%}). "
                    f"Diferença <15% → WAIT forçado. " + result.rationale
                )
                logger.warning(
                    f"🚨 Conflito SEVERO: ML({ml_action}) vs LLM({llm_action}), "
                    f"diff={conf_diff:.0%} → WAIT (conf {old_conf:.0%}→{result.confidence:.0%})"
                )
            else:
                # Conflito moderado: penalidade de 40%
                result.confidence *= 0.6
                result.source += "_conflict_penalty"
                result.rationale = (
                    f"⚠️ CONFLITO: ML({ml_action}) vs LLM({llm_action}). "
                    f"Confiança reduzida. " + result.rationale
                )
                logger.warning(
                    f"⚠️ Conflito moderado: ML({ml_action}) vs LLM({llm_action}), "
                    f"conf {old_conf:.0%}→{result.confidence:.0%}"
                )
                
                # Se confiança cair abaixo de 55%, forçar WAIT
                if result.confidence < 0.55:
                    result.action = "wait"
                    logger.info("   → Confiança pós-penalidade < 55%, forçando WAIT")
        
        # ── 9. Metadados finais ──
        result.model_prob_up = model_prob_up if model_ok else None
        result.model_confidence = model_confidence if model_ok else None
        result.llm_action = llm_action
        result.llm_confidence = llm_confidence
        result.llm_is_fallback = llm_is_fallback
        
        self._log_decision(result)
        return result
    
    def _model_only_decision(
        self,
        prob_up: float,
        confidence: float,
        action: str,
        ai_result: Optional[Dict[str, Any]],
    ) -> DecisionResult:
        """
        Decisão baseada apenas no modelo ML.
        Usada quando LLM está indisponível/fallback.
        Aplica desconto de confiança por não ter confirmação da IA.
        """
        # Desconto de 25% por não ter IA confirmando
        adjusted_conf = confidence * 0.75
        
        # Se ação não é direcional com alta confiança, WAIT
        if action in ("flat", "wait") or adjusted_conf < 0.50:
            return DecisionResult(
                action="wait",
                confidence=adjusted_conf,
                sentiment="neutral",
                rationale=f"ML-only (LLM indisponível): prob_up={prob_up:.1%}, conf ajustada={adjusted_conf:.0%}",
                source="model_only",
                entry_zone=ai_result.get("entry_zone") if ai_result else None,
                invalidation_zone=ai_result.get("invalidation_zone") if ai_result else None,
                region_type=ai_result.get("region_type") if ai_result else None,
            )
        
        sentiment = "bullish" if action == "buy" else "bearish"
        
        return DecisionResult(
            action=action,
            confidence=adjusted_conf,
            sentiment=sentiment,
            rationale=f"ML-only (LLM indisponível): prob_up={prob_up:.1%}, conf ajustada={adjusted_conf:.0%}",
            source="model_only",
            entry_zone=ai_result.get("entry_zone") if ai_result else None,
            invalidation_zone=ai_result.get("invalidation_zone") if ai_result else None,
            region_type=ai_result.get("region_type") if ai_result else None,
        )
    
    def _llm_primary(
        self,
        model_prob_up: Optional[float],
        model_confidence: Optional[float],
        llm_action: str,
        llm_sentiment: str,
        llm_confidence: float,
        llm_rationale: str,
        ai_result: Optional[Dict[str, Any]],
    ) -> DecisionResult:
        """Modo LLM Primary: IA decide, modelo como contexto."""
        # Se ambos concordam, bonus de confiança
        bonus = 0.0
        if model_prob_up is not None:
            ml_direction = "buy" if model_prob_up > 0.6 else ("sell" if model_prob_up < 0.4 else "flat")
            if ml_direction == llm_action and llm_action in DIRECTIONAL_ACTIONS:
                bonus = 0.05  # +5% por consenso
                logger.info(f"✅ Consenso ML+LLM: {llm_action} (+5% conf bonus)")
        
        return DecisionResult(
            action=llm_action,
            confidence=min(1.0, llm_confidence + bonus),
            sentiment=llm_sentiment,
            rationale=llm_rationale,
            source="llm" + ("_consensus" if bonus > 0 else ""),
            entry_zone=ai_result.get("entry_zone") if ai_result else None,
            invalidation_zone=ai_result.get("invalidation_zone") if ai_result else None,
            region_type=ai_result.get("region_type") if ai_result else None,
        )
    
    def _model_primary(
        self,
        model_prob_up: Optional[float],
        model_confidence: Optional[float],
        llm_action: str,
        llm_sentiment: str,
        llm_confidence: float,
        llm_rationale: str,
        ai_result: Optional[Dict[str, Any]],
    ) -> DecisionResult:
        """Modo Model Primary: Modelo decide, IA comenta."""
        if model_prob_up is None or model_confidence is None:
            logger.warning("⚠️ Modelo não disponível, usando IA como fallback")
            return self._llm_primary(
                model_prob_up, model_confidence,
                llm_action, llm_sentiment, llm_confidence, llm_rationale,
                ai_result
            )
        
        if model_confidence < self.model_min_confidence:
            logger.info(
                f"📊 Confiança do modelo baixa ({model_confidence:.0%} < "
                f"{self.model_min_confidence:.0%}), delegando para IA"
            )
            return self._llm_primary(
                model_prob_up, model_confidence,
                llm_action, llm_sentiment, llm_confidence, llm_rationale,
                ai_result
            )
        
        if model_prob_up > 0.6:
            action = "buy"
            sentiment = "bullish"
        elif model_prob_up < 0.4:
            action = "sell"
            sentiment = "bearish"
        else:
            action = "flat"
            sentiment = "neutral"
        
        combined_confidence = 0.7 * model_confidence + 0.3 * llm_confidence
        
        return DecisionResult(
            action=action,
            confidence=combined_confidence,
            sentiment=sentiment,
            rationale=f"[Modelo XGBoost: prob_up={model_prob_up:.1%}] {llm_rationale}",
            source="model",
            entry_zone=ai_result.get("entry_zone") if ai_result else None,
            invalidation_zone=ai_result.get("invalidation_zone") if ai_result else None,
            region_type=ai_result.get("region_type") if ai_result else None,
        )
    
    def _ensemble(
        self,
        model_prob_up: Optional[float],
        model_confidence: Optional[float],
        llm_action: str,
        llm_sentiment: str,
        llm_confidence: float,
        llm_rationale: str,
        ai_result: Optional[Dict[str, Any]],
    ) -> DecisionResult:
        """Modo Ensemble: Ponderação de ambos."""
        if llm_action in ("wait", "avoid"):
            return DecisionResult(
                action=llm_action,
                confidence=llm_confidence,
                sentiment=llm_sentiment,
                rationale=llm_rationale,
                source="llm",
                entry_zone=ai_result.get("entry_zone") if ai_result else None,
                invalidation_zone=ai_result.get("invalidation_zone") if ai_result else None,
                region_type=ai_result.get("region_type") if ai_result else None,
            )
        
        if model_prob_up is None:
            return self._llm_primary(
                model_prob_up, model_confidence,
                llm_action, llm_sentiment, llm_confidence, llm_rationale,
                ai_result
            )
        
        model_score = (model_prob_up - 0.5) * 2
        llm_score = ACTION_TO_SCORE.get(llm_action, 0.0)
        if llm_score is None:
            llm_score = 0.0
        
        ensemble_score = (
            self.model_weight * model_score +
            self.llm_weight * llm_score
        )
        
        final_action = score_to_action(ensemble_score, threshold=0.2)
        
        if ensemble_score > 0.1:
            sentiment = "bullish"
        elif ensemble_score < -0.1:
            sentiment = "bearish"
        else:
            sentiment = "neutral"
        
        combined_confidence = (
            self.model_weight * (model_confidence or 0.5) +
            self.llm_weight * llm_confidence
        )
        
        return DecisionResult(
            action=final_action,
            confidence=combined_confidence,
            sentiment=sentiment,
            rationale=f"[Ensemble: modelo={model_score:+.2f}, IA={llm_score:+.2f}] {llm_rationale}",
            source="ensemble",
            ensemble_score=ensemble_score,
            entry_zone=ai_result.get("entry_zone") if ai_result else None,
            invalidation_zone=ai_result.get("invalidation_zone") if ai_result else None,
            region_type=ai_result.get("region_type") if ai_result else None,
        )
    
    def _log_decision(self, result: DecisionResult) -> None:
        """Loga a cadeia de decisão (defensivo contra None)."""
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
            model_str = "N/A (frozen/disabled)" if self._frozen_count >= 3 else "N/A"
        
        _llm_act = (result.llm_action or "N/A").upper()
        _llm_conf = result.llm_confidence if result.llm_confidence is not None else 0.0
        llm_str = f"{_llm_act} (conf={_llm_conf:.0%})"
        if result.llm_is_fallback:
            llm_str += " [FALLBACK]"
        
        _act = (result.action or "wait").upper()
        _conf = result.confidence if result.confidence is not None else 0.0
        final_str = f"{_act} (conf={_conf:.0%})"
        
        conflict_str = " ⚠️CONFLICT" if result.conflict_detected else ""
        
        logger.info(
            f"🧠 DECISÃO HÍBRIDA [{self.mode}]{conflict_str}:\n"
            f"   Modelo XGBoost: {model_str}\n"
            f"   IA Generativa:  {llm_str}\n"
            f"   → FINAL:        {final_str} (source: {result.source})"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do decision maker."""
        return {
            'total_decisions': self._decision_count,
            'conflicts': self._conflict_count,
            'model_disabled_count': self._model_disabled_count,
            'frozen_count': self._frozen_count,
            'last_model_prob': self._last_model_prob,
            'conflict_rate': (
                self._conflict_count / self._decision_count
                if self._decision_count > 0 else 0
            ),
        }


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def get_hybrid_decision_maker() -> HybridDecisionMaker:
    """Retorna instância singleton do HybridDecisionMaker."""
    return HybridDecisionMaker()


def fuse_decisions(
    ml_prediction: Optional[Dict[str, Any]],
    ai_result: Optional[Dict[str, Any]],
) -> DecisionResult:
    """Função de conveniência para fusão de decisões."""
    return get_hybrid_decision_maker().fuse_decisions(ml_prediction, ai_result)


def decision_to_ai_result(decision: DecisionResult) -> Dict[str, Any]:
    """Converte DecisionResult para formato compatível com AITradeAnalysis."""
    return {
        "sentiment": decision.sentiment,
        "confidence": decision.confidence,
        "action": decision.action,
        "rationale": decision.rationale,
        "entry_zone": decision.entry_zone,
        "invalidation_zone": decision.invalidation_zone,
        "region_type": decision.region_type,
        "_hybrid_source": decision.source,
        "_model_prob_up": decision.model_prob_up,
        "_model_confidence": decision.model_confidence,
        "_llm_action": decision.llm_action,
        "_llm_confidence": decision.llm_confidence,
        "_ensemble_score": decision.ensemble_score,
        "_conflict_detected": decision.conflict_detected,
        "_llm_is_fallback": decision.llm_is_fallback,
    }
