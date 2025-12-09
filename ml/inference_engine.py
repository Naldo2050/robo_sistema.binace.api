# ml/inference_engine.py
import logging
import json
import pandas as pd
import xgboost as xgb
import numpy as np
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger("MLInference")

class MLInferenceEngine:
    """
    Carrega o modelo XGBoost treinado e realiza previsões em tempo real.
    """
    def __init__(self, model_dir: str = "ml/models"):
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "xgb_model_latest.json"
        self.meta_path = self.model_dir / "model_metadata.json"
        self.scaler_path = self.model_dir / "scaler.pkl"
        
        self.model = None
        self.features = []
        self.scaler = None
        self.threshold = 0.5
        
        self._load_model()

    def _load_model(self):
        """Carrega modelo e metadados do disco."""
        if not self.model_path.exists():
            logger.warning(f"⚠️ Modelo ML não encontrado em: {self.model_path}")
            return

        try:
            # Carrega metadados (para saber a ordem das colunas)
            if self.meta_path.exists():
                with open(self.meta_path, "r") as f:
                    meta = json.load(f)
                    self.features = meta.get("feature_names", [])
                    self.threshold = meta.get("threshold_used", 0.002)
                    logger.info(f"📊 Modelo treinado com {len(self.features)} features")
            
            # Carrega scaler se existir
            if self.scaler_path.exists():
                try:
                    import joblib
                    self.scaler = joblib.load(self.scaler_path)
                    logger.info("✅ Scaler carregado")
                except Exception as e:
                    logger.warning(f"⚠️ Falha ao carregar scaler de ML: {e}")
                    self.scaler = None
            
            # Carrega modelo XGBoost
            self.model = xgb.Booster()
            self.model.load_model(str(self.model_path))
            logger.info("✅ Modelo Quantitativo (XGBoost) carregado com sucesso.")
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo ML: {e}", exc_info=True)
            self.model = None

    def extract_ml_features(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extrai features de ML da estrutura de dados do evento.
        Compatível com o FeatureStore.
        """
        features = {}
        
        try:
            # 1. Features de microestrutura do fluxo
            flow_data = event_data.get("fluxo_continuo", {}) or event_data.get("flow_metrics", {})
            microstructure = flow_data.get("microstructure", {})
            
            if microstructure:
                features.update({
                    "tick_rule_sum": float(microstructure.get("tick_rule_sum", 0)),
                    "flow_imbalance": float(microstructure.get("flow_imbalance", 0)),
                    "aggressive_buy_ratio": float(microstructure.get("aggressive_buy_ratio", 0)),
                    "aggressive_sell_ratio": float(microstructure.get("aggressive_sell_ratio", 0)),
                })
            
            # 2. Features de orderbook
            ob_data = event_data.get("orderbook_data", {})
            if ob_data:
                features.update({
                    "bid_ask_ratio": float(ob_data.get("bid_ask_ratio", 1.0)),
                    "orderbook_imbalance": float(ob_data.get("imbalance", 0)),
                    "spread_percent": float(ob_data.get("spread_percent", 0)),
                })
            
            # 3. Features de volume e delta
            features.update({
                "delta": float(event_data.get("delta", 0)),
                "volume_total": float(event_data.get("volume_total", 0)),
                "volume_ratio": float(event_data.get("volume_ratio", 1.0)),
            })
            
            # 4. Features de price action
            ohlc = event_data.get("ohlc", {}) or event_data.get("enriched_snapshot", {}).get("ohlc", {})
            if ohlc:
                close = float(ohlc.get("close", 0))
                high = float(ohlc.get("high", 0))
                low = float(ohlc.get("low", 0))
                if high > low > 0:
                    features.update({
                        "price_range_percent": (high - low) / low * 100,
                        "close_position": (close - low) / (high - low) if (high - low) > 0 else 0.5,
                    })
            
            # 5. Features de whale activity
            whale_data = flow_data.get("whale_activity", {})
            if whale_data:
                features.update({
                    "whale_delta": float(whale_data.get("whale_delta", 0)),
                    "whale_buy_ratio": float(whale_data.get("whale_buy_ratio", 0)),
                })
            
            logger.debug(f"📊 Extraídas {len(features)} features para ML")
            
        except Exception as e:
            logger.error(f"❌ Erro ao extrair features ML: {e}", exc_info=True)
        
        return features

    def predict(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recebe os dados do evento atual e retorna a probabilidade.
        """
        # Evita usar bool(self.model) em objetos Booster; checa explicitamente None
        if self.model is None or not self.features:
            return {"prob_up": None, "status": "model_not_loaded"}

        try:
            # 1. Extrair features do evento
            extracted_features = self.extract_ml_features(event_data)
            
            if not extracted_features:
                return {"prob_up": None, "status": "no_features_extracted"}
            
            # 2. Criar DataFrame com features na ordem esperada
            # Primeiro cria dict com todas as features esperadas
            feature_dict = {feature: 0.0 for feature in self.features}
            
            # Preenche com valores extraídos
            for feature, value in extracted_features.items():
                if feature in feature_dict:
                    feature_dict[feature] = value
            
            # 3. Normalizar se scaler disponível
            if self.scaler:
                try:
                    # Converter para array 2D
                    values = np.array([feature_dict[f] for f in self.features]).reshape(1, -1)
                    scaled_values = self.scaler.transform(values)
                    # Atualizar feature_dict
                    for i, feature in enumerate(self.features):
                        feature_dict[feature] = float(scaled_values[0][i])
                except Exception as e:
                    logger.warning(f"⚠️ Falha ao normalizar: {e}")
            
            # 4. Criar DataFrame final
            input_df = pd.DataFrame([feature_dict])
            
            # Garante ordem correta e tipos numéricos
            input_df = input_df[self.features]
            input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # 5. Previsão
            dmatrix = xgb.DMatrix(input_df)
            prob = self.model.predict(dmatrix)[0]
            
            # 6. Interpretação da confiança
            confidence = abs(prob - 0.5) * 2  # 0-1, quanto mais longe de 0.5
            
            return {
                "prob_up": float(prob),
                "prob_down": 1.0 - float(prob),
                "confidence": float(confidence),
                "status": "ok",
                "features_used": len(extracted_features),
                "total_features": len(self.features)
            }

        except Exception as e:
            logger.error(f"❌ Erro na inferência ML: {e}", exc_info=True)
            return {"prob_up": None, "status": "error", "msg": str(e)}

    def get_feature_importance(self) -> Dict[str, float]:
        """Retorna importância das features se disponível."""
        if self.model is None:
            return {}
        
        try:
            # XGBoost feature importance
            importance = self.model.get_score(importance_type='weight')
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])
        except:
            return {}