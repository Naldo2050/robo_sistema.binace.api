# ml/inference_engine.py
# -*- coding: utf-8 -*-
"""
Módulo de Inferência em Tempo Real para Modelo Quantitativo.

v3 — Correções (2026-03-17):
  - FEATURE_MAP: aliases diretos para bb_upper/bb_lower/bb_width
  - FEATURE_MAP: REMOVIDO fallback VAH/VAL (semanticamente errado)
  - RSI anomalo: fallback multi_tf antes de clampar
  - Metadados de warmup no retorno de predict()
"""

import logging
import xgboost as xgb
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Deque
from collections import deque

logger = logging.getLogger("MLInference")


class MLInferenceEngine:
    """
    Carrega o modelo XGBoost treinado e realiza previsões em tempo real.

    CORREÇÃO v3: FEATURE_MAP com aliases diretos para que valores do
    LiveFeatureCalculator (raiz do dict) sejam encontrados imediatamente.
    Removidos fallbacks VAH/VAL que corrompiam bb_upper/bb_lower.
    """
    
    # Features que o modelo foi TREINADO para esperar
    EXPECTED_FEATURES = [
        'price_close', 'return_1', 'return_5', 'return_10',
        'bb_upper', 'bb_lower', 'bb_width', 'rsi', 'volume_ratio'
    ]
    
    # Mapeamento: feature_do_modelo → lista de possíveis chaves no payload
    # REGRA: o nome direto da feature SEMPRE é o primeiro alias,
    # para que valores do LiveFeatureCalculator (na raiz do dict) sejam
    # encontrados antes de qualquer caminho aninhado ou fallback.
    FEATURE_MAP = {
        'price_close': [
            'price_close',                          # direto do LiveFeatureCalculator
            'enriched.ohlc.close',
            'preco_fechamento',
            'price',
            'current_price',
        ],
        'return_1': [
            'return_1',                             # direto do LiveFeatureCalculator
            'ml_features.price_features.returns_1',
            'ml_features.price_features.returns_1m',
        ],
        'return_5': [
            'return_5',                             # direto do LiveFeatureCalculator
            'ml_features.price_features.returns_5',
            'ml_features.price_features.returns_5m',
        ],
        'return_10': [
            'return_10',                            # direto do LiveFeatureCalculator
            'ml_features.price_features.returns_10',
            'ml_features.price_features.returns_10m',
        ],
        'bb_upper': [
            'bb_upper',                             # direto do LiveFeatureCalculator
            'ml_features.price_features.bb_upper',
            'technical.bb_upper',
            # REMOVIDO: 'contextual.historical_vp.daily.vah'
            # VAH é Volume Profile, semanticamente errado como BB Upper
        ],
        'bb_lower': [
            'bb_lower',                             # direto do LiveFeatureCalculator
            'ml_features.price_features.bb_lower',
            'technical.bb_lower',
            # REMOVIDO: 'contextual.historical_vp.daily.val'
            # VAL é Volume Profile, semanticamente errado como BB Lower
        ],
        'bb_width': [
            'bb_width',                             # direto do LiveFeatureCalculator
            'bb__width',                            # alias de compatibilidade
            'ml_features.price_features.bb_width',
            'technical.bb_width',
        ],
        'rsi': [
            'rsi',                                  # direto do LiveFeatureCalculator
            'ml_features.price_features.rsi',
            'technical.rsi',
            'contextual.multi_tf.15m.rsi',
            'multi_tf.15m.rsi_short',
            'multi_tf.1h.rsi_short',
            'multi_tf.4h.rsi_short',
            'multi_tf.1d.rsi_short',
        ],
        'volume_ratio': [
            'volume_ratio',                         # direto do LiveFeatureCalculator
            'ml_features.microstructure.volume_ratio',
            'volume_sma_ratio',
        ],
    }
    
    # Defaults seguros caso a feature não exista no payload
    FEATURE_DEFAULTS = {
        'price_close': 0.0,
        'return_1': 0.0,
        'return_5': 0.0,
        'return_10': 0.0,
        'bb_upper': 0.0,
        'bb_lower': 0.0,
        'bb_width': 0.0,
        'rsi': 50.0,
        'volume_ratio': 1.0
    }

    # ── Configurações do histórico interno ──
    HISTORY_MAXLEN = 100      # Janelas de histórico para rolling calculations
    BB_PERIOD = 20            # Período para Bollinger Bands
    BB_STD_MULT = 2.0         # Multiplicador de desvio padrão para BB
    RSI_PERIOD = 14           # Período para RSI
    VOL_SMA_PERIOD = 20       # Período para Volume SMA

    def __init__(self, model_dir: str = "ml/models"):
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "xgb_model_latest.json"
        self.model = None
        
        # ── Histórico interno para computar features rolling ──
        self._price_history: Deque[float] = deque(maxlen=self.HISTORY_MAXLEN)
        self._volume_history: Deque[float] = deque(maxlen=self.HISTORY_MAXLEN)
        self._window_count: int = 0
        
        self._load_model()

    def _load_model(self):
        """Carrega o modelo XGBoost."""
        try:
            if not self.model_path.exists():
                logger.warning(f"⚠️ Modelo ML não encontrado em: {self.model_path}")
                return

            self.model = xgb.Booster()
            self.model.load_model(str(self.model_path))
            
            logger.info(f"✅ Modelo Quantitativo (XGBoost) carregado: {self.model_path.name}")
        except Exception as e:
            logger.error(f"❌ Falha ao carregar modelo: {e}")
            self.model = None

    @staticmethod
    def _deep_get(d: dict, dotted_key: str) -> Any:
        """Busca chave aninhada 'a.b.c' em {'a': {'b': {'c': 1}}}."""
        keys = dotted_key.split('.')
        current = d
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None
        return current

    def _extract_price_from_payload(self, raw_features: dict) -> Optional[float]:
        """Extrai o preço de fechamento do payload usando todos os aliases conhecidos."""
        for alias in self.FEATURE_MAP['price_close']:
            # Busca direta
            if alias in raw_features and raw_features[alias] is not None:
                try:
                    return float(raw_features[alias])
                except (ValueError, TypeError):
                    continue
            # Busca aninhada
            if '.' in alias:
                val = self._deep_get(raw_features, alias)
                if val is not None:
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        continue
        return None

    def _extract_volume_from_payload(self, raw_features: dict) -> float:
        """Extrai volume total do payload."""
        for key in ['volume_total', 'volume', 'total_volume']:
            val = raw_features.get(key)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    continue
            # Busca aninhada
            for prefix in ['enriched.', 'raw_event.']:
                val = self._deep_get(raw_features, f"{prefix}{key}")
                if val is not None:
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        continue
        return 0.0

    # ── Cálculo interno de features rolling ──

    def _compute_returns(self) -> Dict[str, float]:
        """Computa retornos de 1, 5 e 10 períodos a partir do histórico."""
        prices = list(self._price_history)
        n = len(prices)
        result = {}
        
        if n >= 2:
            result['return_1'] = (prices[-1] / prices[-2]) - 1.0
        if n >= 6:
            result['return_5'] = (prices[-1] / prices[-5]) - 1.0
        if n >= 11:
            result['return_10'] = (prices[-1] / prices[-10]) - 1.0
        
        return result

    def _compute_bollinger_bands(self) -> Dict[str, float]:
        """Computa Bollinger Bands a partir do histórico de preços."""
        prices = list(self._price_history)
        n = len(prices)
        result = {}
        
        # Precisa de pelo menos alguns preços para calcular
        period = min(self.BB_PERIOD, n)
        if period < 5:
            return result
        
        window = prices[-period:]
        sma = np.mean(window)
        std = np.std(window, ddof=1) if period > 1 else 0.0
        
        bb_upper = sma + self.BB_STD_MULT * std
        bb_lower = sma - self.BB_STD_MULT * std
        
        result['bb_upper'] = bb_upper
        result['bb_lower'] = bb_lower
        result['bb_width'] = (bb_upper - bb_lower) / sma if sma > 0 else 0.0
        
        return result

    def _compute_rsi(self) -> Dict[str, float]:
        """Computa RSI a partir do histórico de preços."""
        prices = list(self._price_history)
        n = len(prices)
        result = {}
        
        period = min(self.RSI_PERIOD, n - 1)
        if period < 2:
            return result
        
        # Calcular deltas
        deltas = np.diff(prices[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss < 1e-10:
            # Sem quedas: se também sem ganhos significativos → neutro;
            # caso contrário, cap em 80 (não 95, que distorce XGBoost)
            result['rsi'] = 80.0 if avg_gain > 1e-10 else 50.0
        else:
            rs = avg_gain / avg_loss
            result['rsi'] = 100.0 - (100.0 / (1.0 + rs))
        
        return result

    def _compute_volume_ratio(self) -> Dict[str, float]:
        """Computa ratio do volume atual vs SMA."""
        volumes = list(self._volume_history)
        n = len(volumes)
        result = {}
        
        if n < 2:
            return result
        
        period = min(self.VOL_SMA_PERIOD, n)
        vol_sma = np.mean(volumes[-period:])
        
        if vol_sma > 1e-10:
            result['volume_ratio'] = volumes[-1] / vol_sma
        else:
            result['volume_ratio'] = 1.0
        
        return result

    def _compute_all_internal_features(self, price: float) -> Dict[str, float]:
        """
        Computa TODAS as features rolling internamente.
        Retorna apenas as que puderam ser calculadas (dependendo do histórico).
        """
        features = {'price_close': price}
        
        features.update(self._compute_returns())
        features.update(self._compute_bollinger_bands())
        features.update(self._compute_rsi())
        features.update(self._compute_volume_ratio())
        
        return features

    def _map_features(self, raw_features: dict) -> dict:
        """
        Mapeia features com PRIORIDADE:
        1. Valor computado externamente (se presente no payload via FEATURE_MAP)
        2. Valor computado INTERNAMENTE pelo histórico acumulado
        3. FEATURE_DEFAULTS (último recurso)
        """
        # Passo 0: Extrair preço e volume, atualizar histórico
        price = self._extract_price_from_payload(raw_features)
        volume = self._extract_volume_from_payload(raw_features)
        
        if price is not None and price > 0:
            self._price_history.append(price)
            self._volume_history.append(volume)
            self._window_count += 1
        
        # Passo 1: Computar features internas a partir do histórico
        internal_features = {}
        if price is not None and price > 0 and len(self._price_history) >= 2:
            internal_features = self._compute_all_internal_features(price)
        
        # Passo 2: Para cada feature esperada, tentar:
        #   a) Payload externo (FEATURE_MAP)
        #   b) Cálculo interno
        #   c) Default
        mapped = {}
        sources = {}
        from_internal = []
        from_external = []
        from_default = []
        
        for model_feat in self.EXPECTED_FEATURES:
            value = None
            source_found = None
            
            # (a) Tentar payload externo via FEATURE_MAP
            aliases = self.FEATURE_MAP.get(model_feat, [model_feat])
            for alias in aliases:
                if alias in raw_features and raw_features[alias] is not None:
                    try:
                        candidate = float(raw_features[alias])
                        # Aceitar valor externo apenas se não for um default óbvio
                        # (ex: return=0.0 pode ser legítimo, mas bb_upper=0.0 não)
                        if model_feat in ('return_1', 'return_5', 'return_10') or candidate != 0.0:
                            value = candidate
                            source_found = f"ext:{alias}"
                            break
                    except (ValueError, TypeError):
                        continue
                
                if '.' in alias:
                    deep_val = self._deep_get(raw_features, alias)
                    if deep_val is not None:
                        try:
                            candidate = float(deep_val)
                            if model_feat in ('return_1', 'return_5', 'return_10') or candidate != 0.0:
                                value = candidate
                                source_found = f"ext:{alias}"
                                break
                        except (ValueError, TypeError):
                            continue
            
            # (b) Se não encontrou externamente, usar cálculo interno
            if value is None and model_feat in internal_features:
                value = internal_features[model_feat]
                source_found = f"int:{model_feat}"
                from_internal.append(model_feat)
            elif value is not None:
                from_external.append(model_feat)
            
            # (c) Último recurso: default
            if value is None:
                value = self.FEATURE_DEFAULTS[model_feat]
                source_found = f"default:{model_feat}"
                from_default.append(model_feat)
            
            mapped[model_feat] = value
            sources[model_feat] = source_found
        
        # Logging detalhado
        if from_default:
            logger.warning(
                f"⚠️ ML features usando DEFAULT: {from_default} "
                f"({len(from_default)}/{len(self.EXPECTED_FEATURES)}) "
                f"[history={len(self._price_history)} windows]"
            )
        
        if from_internal:
            logger.info(
                f"📊 ML features computadas internamente: {from_internal} "
                f"({len(from_internal)}/{len(self.EXPECTED_FEATURES)})"
            )
        
        if from_external:
            logger.debug(
                f"✅ ML features do payload: {from_external} "
                f"({len(from_external)}/{len(self.EXPECTED_FEATURES)})"
            )
        
        if not from_default:
            logger.info(
                f"✅ ML features COMPLETAS: "
                f"{len(from_external)} ext + {len(from_internal)} int "
                f"= {len(self.EXPECTED_FEATURES)}/{len(self.EXPECTED_FEATURES)} "
                f"[history={len(self._price_history)}]"
            )
        
        return mapped

    def _compute_derived_features(self, features: dict) -> dict:
        """
        Computa features derivadas se estiverem zeradas mas o preço for conhecido.
        Safety net — raramente necessário com o FEATURE_MAP v3.
        """
        price = features.get('price_close', 0.0)
        
        if price > 0:
            # BB Check - só usar fallback se realmente não foi calculado
            if features.get('bb_upper', 0.0) == 0.0 and features.get('bb_lower', 0.0) == 0.0:
                features['bb_upper'] = price * 1.02
                features['bb_lower'] = price * 0.98
                features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / price
                logger.debug("[DERIVED] BB fallback aplicado (sem dados históricos)")
        
        return features

    def predict(self, raw_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Faz predição com o modelo XGBoost.
        
        FLUXO:
        1. Extrai preço/volume do payload → atualiza histórico interno
        2. Mapeia features: payload externo > cálculo interno > defaults
        3. Aplica derived features (safety net)
        4. Predição via XGBoost
        """
        if self.model is None:
            return {
                'prob_up': 0.5,
                'signal': 'neutral',
                'status': 'model_not_loaded'
            }
        
        try:
            # 1. Mapeamento (com histórico interno)
            mapped_features = self._map_features(raw_features)
            
            # 2. Features derivadas (safety net)
            mapped_features = self._compute_derived_features(mapped_features)

            # 2b. Validar RSI — nunca pode ser exatamente 0 ou 100
            rsi_val = mapped_features.get('rsi', 50.0)
            if rsi_val >= 100.0 or rsi_val <= 0.0:
                # Tentar multi_tf antes de usar 50.0
                rsi_fallback = 50.0
                multi_tf = raw_features.get("multi_tf", {})
                if isinstance(multi_tf, dict):
                    for tf in ("15m", "1h", "4h", "1d"):
                        tf_data = multi_tf.get(tf, {})
                        if isinstance(tf_data, dict):
                            for rsi_key in ("rsi_short", "rsi", "rsi_14"):
                                rsi_mtf = tf_data.get(rsi_key)
                                if rsi_mtf is not None:
                                    try:
                                        v = float(rsi_mtf)
                                        if 5.0 < v < 95.0:
                                            rsi_fallback = v
                                            break
                                    except (ValueError, TypeError):
                                        continue
                            if rsi_fallback != 50.0:
                                break
                logger.warning(f"RSI anomalo: {rsi_val:.1f}, corrigindo para {rsi_fallback:.1f}")
                mapped_features['rsi'] = rsi_fallback

            # 3. Ordenação conforme treino
            feature_values = [mapped_features[f] for f in self.EXPECTED_FEATURES]
            
            # 4. Criação da DMatrix
            dmatrix = xgb.DMatrix(
                data=np.array([feature_values]),
                feature_names=self.EXPECTED_FEATURES
            )
            
            # 5. Predição
            prob = float(self.model.predict(dmatrix)[0])
            
            # Signal classification
            if prob > 0.6:
                signal = 'bullish'
            elif prob < 0.4:
                signal = 'bearish'
            else:
                signal = 'neutral'
            
            # Log da predição com contexto
            logger.info(
                f"ML prediction: prob_up={prob:.4f} | signal={signal} | "
                f"features=[ret1={mapped_features.get('return_1', 0):.6f}, "
                f"rsi={mapped_features.get('rsi', 0):.1f}, "
                f"bb_w={mapped_features.get('bb_width', 0):.6f}, "
                f"vol_r={mapped_features.get('volume_ratio', 0):.2f}] | "
                f"history={len(self._price_history)}"
            )
            
            history_n = len(self._price_history)
            _min_history = max(self.BB_PERIOD, self.RSI_PERIOD + 1, 10)
            _warmup_ready = history_n >= _min_history
            _ml_usable = history_n >= (self.RSI_PERIOD + 1)  # >=15: returns + RSI disponíveis

            return {
                'prob_up': prob,
                'prob_down': 1.0 - prob,
                'signal': signal,
                'status': 'ok',
                'confidence': abs(prob - 0.5) * 2,
                'features_used': len(mapped_features),
                'features_from_history': history_n,
                'features_detail': {
                    k: round(v, 6) for k, v in mapped_features.items()
                },
                # Metadados de warmup — lidos por hybrid_decision.fuse_decisions()
                '_warmup_ready': _warmup_ready,
                '_ml_usable': _ml_usable,
                '_features_real_count': history_n,  # proxy: mais história = mais features reais
                '_history_count': history_n,
            }
            
        except Exception as e:
            logger.error(f"❌ Erro na extração/predição ML: {e}", exc_info=True)
            return {
                'prob_up': 0.5,
                'status': 'error',
                'msg': str(e)
            }

    def get_feature_importance(self) -> Dict[str, Any]:
        """Retorna importância das features se disponível."""
        if self.model is None:
            return {}
        try:
            importance = self.model.get_score(importance_type='weight')
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        except Exception:
            return {}
    
    def get_history_status(self) -> Dict[str, Any]:
        """Retorna status do histórico interno (para diagnóstico)."""
        return {
            'window_count': self._window_count,
            'price_history_len': len(self._price_history),
            'volume_history_len': len(self._volume_history),
            'last_price': self._price_history[-1] if self._price_history else None,
            'last_volume': self._volume_history[-1] if self._volume_history else None,
            'can_compute_return_1': len(self._price_history) >= 2,
            'can_compute_return_5': len(self._price_history) >= 6,
            'can_compute_return_10': len(self._price_history) >= 11,
            'can_compute_bb': len(self._price_history) >= 5,
            'can_compute_rsi': len(self._price_history) >= 3,
            'can_compute_vol_ratio': len(self._volume_history) >= 2,
        }
