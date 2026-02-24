# flow_analyzer/absorption.py
"""
Lógica de absorção do FlowAnalyzer.

Absorção ocorre quando:
- Grande volume de um lado (ex: vendas)
- Preço não se move significativamente
- Indica que o outro lado (compradores) está absorvendo a pressão

Tipos:
- Absorção de Compra: Vendedores agressivos, compradores absorvem
- Absorção de Venda: Compradores agressivos, vendedores absorvem
- Neutra: Sem absorção significativa
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

from .constants import (
    DEFAULT_ABSORCAO_DELTA_EPS,
    DEFAULT_ABSORCAO_ATR_MULTIPLIER,
    DEFAULT_ABSORCAO_VOL_MULTIPLIER,
    DEFAULT_ABSORCAO_MIN_PCT_TOLERANCE,
    DEFAULT_ABSORCAO_MAX_PCT_TOLERANCE,
    DEFAULT_ABSORCAO_FALLBACK_PCT_TOLERANCE,
    ABSORPTION_INTENSITY_THRESHOLD,
    ABSORPTION_IMBALANCE_THRESHOLD,
)
from .validation import validate_ohlc, guard_absorcao
from .utils import lazy_log, decimal_round, clamp


# ==============================================================================
# ABSORPTION CLASSIFIER
# ==============================================================================

@dataclass
class AbsorptionConfig:
    """Configuração para classificação de absorção."""
    
    eps: float = DEFAULT_ABSORCAO_DELTA_EPS
    atr_multiplier: float = DEFAULT_ABSORCAO_ATR_MULTIPLIER
    vol_multiplier: float = DEFAULT_ABSORCAO_VOL_MULTIPLIER
    min_pct_tolerance: float = DEFAULT_ABSORCAO_MIN_PCT_TOLERANCE
    max_pct_tolerance: float = DEFAULT_ABSORCAO_MAX_PCT_TOLERANCE
    fallback_pct_tolerance: float = DEFAULT_ABSORCAO_FALLBACK_PCT_TOLERANCE
    intensity_threshold: float = ABSORPTION_INTENSITY_THRESHOLD
    imbalance_threshold: float = ABSORPTION_IMBALANCE_THRESHOLD


class AbsorptionClassifier:
    """
    Classificador de absorção com contexto de volatilidade.
    
    Usa OHLC e delta para determinar se há absorção e de que tipo.
    Suporta tolerância dinâmica baseada em ATR ou volatilidade.
    
    Example:
        >>> classifier = AbsorptionClassifier()
        >>> classifier.update_volatility(atr=500.0)
        >>> label = classifier.classify(
        ...     delta_btc=-10.0,
        ...     open_p=50000, high_p=50100, low_p=49900, close_p=50050
        ... )
        >>> print(label)  # "Absorção de Compra"
    """
    
    def __init__(self, config: Optional[AbsorptionConfig] = None):
        self.config = config or AbsorptionConfig()
        self._atr_price: Optional[float] = None
        self._price_volatility: Optional[float] = None
    
    def update_volatility(
        self,
        atr: Optional[float] = None,
        price_volatility: Optional[float] = None
    ) -> None:
        """
        Atualiza contexto de volatilidade.
        
        Args:
            atr: Average True Range
            price_volatility: Volatilidade de preço (desvio padrão)
        """
        if isinstance(atr, (int, float)) and atr > 0:
            self._atr_price = float(atr)
        if isinstance(price_volatility, (int, float)) and price_volatility > 0:
            self._price_volatility = float(price_volatility)
    
    def _calculate_tolerance(self, base_price: float) -> float:
        """
        Calcula tolerância de preço dinâmica.
        
        Prioridade:
        1. ATR-based
        2. Volatility-based
        3. Fallback fixo
        
        Args:
            base_price: Preço base para cálculo percentual
            
        Returns:
            Tolerância como percentual do preço
        """
        if base_price <= 0:
            return self.config.fallback_pct_tolerance
        
        pct_tolerance: Optional[float] = None
        
        # ATR-based
        if self._atr_price is not None and self._atr_price > 0:
            pct_tolerance = (self.config.atr_multiplier * self._atr_price) / base_price
        
        # Volatility-based fallback
        if pct_tolerance is None and self._price_volatility is not None:
            if self._price_volatility > 0:
                pct_tolerance = (
                    self.config.vol_multiplier * self._price_volatility
                ) / base_price
        
        # Fallback fixo
        if pct_tolerance is None:
            pct_tolerance = self.config.fallback_pct_tolerance
        
        # Clamp
        return clamp(
            pct_tolerance,
            self.config.min_pct_tolerance,
            self.config.max_pct_tolerance
        )
    
    def classify(
        self,
        delta_btc: float,
        open_p: float,
        high_p: float,
        low_p: float,
        close_p: float,
        eps: Optional[float] = None,
    ) -> str:
        """
        Classifica absorção baseado em delta e OHLC.
        
        Lógica:
        - Delta negativo + preço não caiu muito = Absorção de Compra
          (compradores absorveram pressão vendedora)
        - Delta positivo + preço não subiu muito = Absorção de Venda
          (vendedores absorveram pressão compradora)
        
        Args:
            delta_btc: Delta de volume (compras - vendas)
            open_p: Preço de abertura
            high_p: Preço máximo
            low_p: Preço mínimo
            close_p: Preço de fechamento
            eps: Epsilon para delta neutro (usa config se None)
            
        Returns:
            "Absorção de Compra", "Absorção de Venda", ou "Neutra"
        """
        if eps is None:
            eps = self.config.eps
        
        try:
            # Validação básica
            if not all(isinstance(x, (int, float)) for x in 
                      [delta_btc, open_p, high_p, low_p, close_p, eps]):
                return "Neutra"
            
            if not validate_ohlc(open_p, high_p, low_p, close_p):
                return "Neutra"
            
            # Range do candle
            candle_range = high_p - low_p
            if candle_range <= 0:
                candle_range = 0.0001  # Evita divisão por zero
            
            # Posição do fechamento no range
            # close_pos_compra: quanto mais alto, mais força compradora
            close_pos_compra = (close_p - low_p) / candle_range
            # close_pos_venda: quanto mais baixo, mais força vendedora
            close_pos_venda = (high_p - close_p) / candle_range
            
            # Tolerância dinâmica
            base_price = close_p if close_p > 0 else open_p
            pct_tolerance = self._calculate_tolerance(base_price)
            
            # Bounds
            lower_bound = open_p * (1.0 - pct_tolerance)
            upper_bound = open_p * (1.0 + pct_tolerance)
            
            # Classificação
            # Absorção de Compra:
            # - Delta negativo (mais vendas)
            # - Mas preço não caiu (fechou acima do lower bound)
            # - Fechamento na metade superior do candle
            if (delta_btc < -abs(eps) and 
                close_p >= lower_bound and 
                close_pos_compra > 0.5):
                return "Absorção de Compra"
            
            # Absorção de Venda:
            # - Delta positivo (mais compras)
            # - Mas preço não subiu (fechou abaixo do upper bound)
            # - Fechamento na metade inferior do candle
            if (delta_btc > abs(eps) and 
                close_p <= upper_bound and 
                close_pos_venda > 0.5):
                return "Absorção de Venda"
            
            return "Neutra"
            
        except Exception as e:
            if lazy_log.should_log("absorption_classify_error"):
                logging.warning(f"Erro em classify: {e}")
            return "Neutra"
    
    def classify_simple(self, delta: float, eps: Optional[float] = None) -> str:
        """
        Classificador simples de absorção apenas por delta.
        
        Não considera OHLC, útil para análise rápida.
        
        Args:
            delta: Delta de volume
            eps: Epsilon para delta neutro
            
        Returns:
            "Absorção de Compra", "Absorção de Venda", ou "Neutra"
        """
        if eps is None:
            eps = self.config.eps
        
        try:
            d = float(delta)
        except (TypeError, ValueError):
            return "Neutra"
        
        if d < -eps:
            return "Absorção de Compra"
        if d > eps:
            return "Absorção de Venda"
        return "Neutra"
    
    @staticmethod
    def map_aggression_to_label(aggression_side: str) -> str:
        """
        Mapeia lado de agressão para rótulo de absorção.
        
        Args:
            aggression_side: "buy" ou "sell"
            
        Returns:
            Rótulo de absorção correspondente
        """
        side = (aggression_side or "").strip().lower()
        if side == "buy":
            return "Absorção de Compra"
        if side == "sell":
            return "Absorção de Venda"
        return "Absorção"


# ==============================================================================
# ABSORPTION ANALYSIS
# ==============================================================================

@dataclass
class AbsorptionAnalysis:
    """Resultado de análise de absorção."""
    
    index: float  # 0.0 a 1.0
    classification: str  # NONE, WEAK, MODERATE, STRONG
    label: str  # Absorção de Compra/Venda/Neutra
    buyer_strength: float  # 0-10
    seller_strength: float  # 0-10
    seller_exhaustion: float  # 0-10
    continuation_probability: float  # 0.0 a 1.0
    
    # Dados de suporte
    delta_usd: float
    total_volume_usd: float
    flow_imbalance: float
    window_min: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'index': self.index,
            'classification': self.classification,
            'label': self.label,
            'buyer_strength': self.buyer_strength,
            'seller_exhaustion': self.seller_exhaustion,
            'continuation_probability': self.continuation_probability,
            'delta_usd': self.delta_usd,
            'total_volume_usd': self.total_volume_usd,
            'flow_imbalance': self.flow_imbalance,
            'window_min': self.window_min,
        }


class AbsorptionAnalyzer:
    """
    Analisador avançado de absorção.
    
    Calcula índices de absorção, força de compradores/vendedores,
    e probabilidade de continuação.
    """
    
    def __init__(self, config: Optional[AbsorptionConfig] = None):
        self.config = config or AbsorptionConfig()
    
    def analyze(
        self,
        delta_usd: float,
        total_volume_usd: float,
        flow_imbalance: float,
        buy_pct: float,
        sell_pct: float,
        absorption_label: str,
        window_min: int,
    ) -> Optional[AbsorptionAnalysis]:
        """
        Analisa absorção a partir de métricas de flow.
        
        Args:
            delta_usd: Net flow em USD
            total_volume_usd: Volume total em USD
            flow_imbalance: Imbalance (-1 a 1)
            buy_pct: Percentual de compras
            sell_pct: Percentual de vendas
            absorption_label: Rótulo de absorção
            window_min: Janela de tempo em minutos
            
        Returns:
            AbsorptionAnalysis ou None se dados insuficientes
        """
        if total_volume_usd <= 0:
            return None
        
        try:
            # Índice de absorção: combinação de delta relativo e imbalance
            rel_delta = min(1.0, abs(delta_usd) / total_volume_usd)
            abs_flow = min(1.0, abs(flow_imbalance))
            absorption_index = decimal_round(rel_delta * abs_flow, decimals=4)
            
            # Classificação
            if absorption_index >= 0.7:
                classification = "STRONG_ABSORPTION"
            elif absorption_index >= 0.4:
                classification = "MODERATE_ABSORPTION"
            elif absorption_index > 0.1:
                classification = "WEAK_ABSORPTION"
            else:
                classification = "NONE"
            
            # Força de compradores/vendedores
            total_pct = buy_pct + sell_pct
            if total_pct > 0:
                buy_intensity = buy_pct / total_pct
            else:
                buy_intensity = 0.5
            
            buyer_strength = decimal_round(buy_intensity * 10, decimals=1)
            seller_strength = decimal_round((1 - buy_intensity) * 10, decimals=1)
            
            # Seller exhaustion baseado no tipo de absorção
            if "Compra" in absorption_label:
                # Absorção de compra = compradores absorvendo vendas
                seller_exhaustion = buyer_strength
            elif "Venda" in absorption_label:
                # Absorção de venda = vendedores absorvendo compras
                seller_exhaustion = seller_strength
            else:
                seller_exhaustion = decimal_round(abs_flow * 10, decimals=1)
            
            # Probabilidade de continuação
            continuation_probability = decimal_round(absorption_index * 0.9, decimals=2)
            
            return AbsorptionAnalysis(
                index=absorption_index,
                classification=classification,
                label=absorption_label,
                buyer_strength=buyer_strength,
                seller_strength=seller_strength,
                seller_exhaustion=seller_exhaustion,
                continuation_probability=continuation_probability,
                delta_usd=delta_usd,
                total_volume_usd=total_volume_usd,
                flow_imbalance=flow_imbalance,
                window_min=window_min,
            )
            
        except Exception as e:
            if lazy_log.should_log("absorption_analyze_error"):
                logging.debug(f"Erro em analyze: {e}")
            return None
    
    def refine_label_with_intensity(
        self,
        base_label: str,
        delta_btc: float,
        total_btc: float,
        flow_imbalance: float,
        eps: float,
    ) -> str:
        """
        Refina rótulo de absorção com base em intensidade.
        
        Só mantém rótulo de absorção se intensidade e imbalance
        forem significativos.
        
        Args:
            base_label: Rótulo original
            delta_btc: Delta em BTC
            total_btc: Volume total em BTC
            flow_imbalance: Imbalance (-1 a 1)
            eps: Epsilon
            
        Returns:
            Rótulo refinado
        """
        try:
            if total_btc <= 0:
                return "Neutra"
            
            intensidade = abs(delta_btc) / total_btc
            
            # Precisa de intensidade E imbalance significativos
            if (intensidade >= self.config.intensity_threshold and 
                abs(flow_imbalance) >= self.config.imbalance_threshold):
                
                if delta_btc < -eps:
                    return "Absorção de Compra"
                elif delta_btc > eps:
                    return "Absorção de Venda"
            
            return "Neutra"
            
        except Exception:
             return base_label


class AbsorptionZoneMapper:
    """
    Mapeia zonas onde absorções significativas ocorreram.
    
    Mantém histórico de eventos de absorção com timestamp e preço,
    agrupa por zona de preço, e identifica zonas recorrentes.
    
    Zonas com múltiplas absorções são defesas fortes e confiáveis.
    
    Uso:
        mapper = AbsorptionZoneMapper()
        mapper.record_event(price=64800, classification="Absorção de Compra",
                           index=0.65, timestamp_ms=1771888200000)
        zones = mapper.get_zones(current_price=64892)
    """

    def __init__(
        self,
        zone_tolerance_pct: float = 0.15,
        max_history_hours: int = 24,
        min_index_threshold: float = 0.1,
    ):
        """
        Args:
            zone_tolerance_pct: % de tolerância para agrupar eventos na mesma zona.
            max_history_hours: Máximo de horas de histórico a manter.
            min_index_threshold: Índice mínimo de absorção para registrar.
        """
        self._tolerance_pct = zone_tolerance_pct
        self._max_history_ms = max_history_hours * 3600 * 1000
        self._min_index = min_index_threshold
        self._events = []

    def record_event(
        self,
        price: float,
        classification: str,
        index: float = 0,
        timestamp_ms: Optional[int] = None,
        buyer_strength: float = 0,
        seller_exhaustion: float = 0,
        volume_usd: float = 0,
    ) -> None:
        """
        Registra um evento de absorção.
        
        Args:
            price: Preço onde a absorção ocorreu.
            classification: Tipo ("Absorção de Compra", "Absorção de Venda",
                            "STRONG_ABSORPTION", etc.)
            index: Índice de absorção (0-1).
            timestamp_ms: Timestamp em ms.
            buyer_strength: Força do comprador (0-10).
            seller_exhaustion: Exaustão do vendedor (0-10).
            volume_usd: Volume em USD durante a absorção.
        """
        import time

        if price <= 0:
            return

        if index < self._min_index and "NONE" not in classification.upper():
            # Absorção muito fraca, mas se for classificada, registrar
            if "Neutra" in classification or "NONE" in classification.upper():
                return

        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        # Normalizar classificação
        classification_upper = classification.upper()
        if "COMPRA" in classification_upper or "BUY" in classification_upper:
            side = "buy"
        elif "VENDA" in classification_upper or "SELL" in classification_upper:
            side = "sell"
        else:
            side = "neutral"

        self._events.append({
            "price": price,
            "side": side,
            "classification": classification,
            "index": index,
            "timestamp_ms": timestamp_ms,
            "buyer_strength": buyer_strength,
            "seller_exhaustion": seller_exhaustion,
            "volume_usd": volume_usd,
        })

        # Cleanup de eventos antigos (lazy)
        if len(self._events) > 500:
            self._cleanup()

    def _cleanup(self) -> None:
        """Remove eventos fora da janela temporal."""
        import time
        now_ms = int(time.time() * 1000)
        cutoff = now_ms - self._max_history_ms
        self._events = [e for e in self._events if e["timestamp_ms"] >= cutoff]

    def get_zones(self, current_price: float = 0, top_n: int = 10) -> dict:
        """
        Retorna zonas de absorção mapeadas.
        
        Args:
            current_price: Preço atual para calcular distâncias.
            top_n: Número máximo de zonas a retornar.
            
        Returns:
            Dict com zonas agrupadas, lado dominante e métricas.
        """
        import time

        self._cleanup()

        if not self._events:
            return {
                "zones": [],
                "total_zones": 0,
                "total_events": 0,
                "buy_zone_count": 0,
                "sell_zone_count": 0,
                "status": "no_events",
            }

        # Agrupar eventos por zona de preço
        zones_map = {}

        for event in self._events:
            # Encontrar zona existente ou criar nova
            assigned = False
            for zone_key, zone_data in zones_map.items():
                if abs(event["price"] - zone_data["center"]) / zone_data["center"] * 100 < self._tolerance_pct:
                    zone_data["events"].append(event)
                    # Atualizar center como média ponderada
                    total_events = len(zone_data["events"])
                    zone_data["center"] = sum(e["price"] for e in zone_data["events"]) / total_events
                    assigned = True
                    break

            if not assigned:
                zone_key = round(event["price"], 0)
                zones_map[zone_key] = {
                    "center": event["price"],
                    "events": [event],
                }

        # Construir resultado
        zones_result = []
        for zone_key, zone_data in zones_map.items():
            events = zone_data["events"]
            center = zone_data["center"]

            # Contagem por lado
            buy_events = [e for e in events if e["side"] == "buy"]
            sell_events = [e for e in events if e["side"] == "sell"]

            # Métricas
            total_index = sum(e["index"] for e in events)
            avg_index = total_index / len(events) if events else 0
            max_index = max((e["index"] for e in events), default=0)
            total_volume = sum(e["volume_usd"] for e in events)

            # Último evento
            latest = max(events, key=lambda e: e["timestamp_ms"])
            oldest = min(events, key=lambda e: e["timestamp_ms"])

            # Lado dominante
            if len(buy_events) > len(sell_events):
                dominant_side = "buy_defense"
            elif len(sell_events) > len(buy_events):
                dominant_side = "sell_defense"
            else:
                dominant_side = "contested"

            # Distância ao preço atual
            distance = 0
            distance_pct = 0
            direction = "unknown"
            if current_price > 0:
                distance = abs(center - current_price)
                distance_pct = (distance / current_price) * 100
                direction = "below" if center < current_price else "above"

            zones_result.append({
                "center": round(center, 2),
                "range_low": round(min(e["price"] for e in events), 2),
                "range_high": round(max(e["price"] for e in events), 2),
                "event_count": len(events),
                "buy_events": len(buy_events),
                "sell_events": len(sell_events),
                "dominant_side": dominant_side,
                "total_strength": round(total_index, 4),
                "avg_strength": round(avg_index, 4),
                "max_strength": round(max_index, 4),
                "total_volume_usd": round(total_volume, 2),
                "last_event_ms": latest["timestamp_ms"],
                "last_classification": latest["classification"],
                "zone_age_ms": latest["timestamp_ms"] - oldest["timestamp_ms"],
                "distance_from_price": round(distance, 2),
                "distance_pct": round(distance_pct, 4),
                "direction": direction,
                "reliability": (
                    "HIGH" if len(events) >= 5 and avg_index > 0.4
                    else "MEDIUM" if len(events) >= 3 or avg_index > 0.3
                    else "LOW"
                ),
            })

        # Ordenar por força total (mais forte primeiro)
        zones_result.sort(key=lambda z: z["total_strength"], reverse=True)
        zones_result = zones_result[:top_n]

        buy_zones = [z for z in zones_result if z["dominant_side"] == "buy_defense"]
        sell_zones = [z for z in zones_result if z["dominant_side"] == "sell_defense"]

        return {
            "zones": zones_result,
            "total_zones": len(zones_result),
            "total_events": len(self._events),
            "buy_zone_count": len(buy_zones),
            "sell_zone_count": len(sell_zones),
            "strongest_zone": zones_result[0] if zones_result else None,
            "status": "success",
        }

    def get_summary(self) -> dict:
        """Resumo rápido sem cálculos pesados."""
        if not self._events:
            return {"status": "empty", "total_events": 0}

        buy_events = sum(1 for e in self._events if e["side"] == "buy")
        sell_events = sum(1 for e in self._events if e["side"] == "sell")

        return {
            "status": "ok",
            "total_events": len(self._events),
            "buy_absorptions": buy_events,
            "sell_absorptions": sell_events,
            "avg_index": round(
                sum(e["index"] for e in self._events) / len(self._events), 4
            ),
            "dominant_side": "buy" if buy_events > sell_events else "sell" if sell_events > buy_events else "balanced",
        }

    def reset(self) -> None:
        """Limpa todo o histórico."""
        self._events.clear()


# ==============================================================================
# FUNÇÕES DE CONVENIÊNCIA
# ==============================================================================

# Instância global para uso simples
_default_classifier = AbsorptionClassifier()
_default_analyzer = AbsorptionAnalyzer()


def classify_absorption(
    delta_btc: float,
    open_p: float,
    high_p: float,
    low_p: float,
    close_p: float,
    eps: float = DEFAULT_ABSORCAO_DELTA_EPS,
    atr: Optional[float] = None,
    price_volatility: Optional[float] = None,
) -> str:
    """
    Função de conveniência para classificar absorção.
    
    Args:
        delta_btc: Delta de volume
        open_p, high_p, low_p, close_p: OHLC
        eps: Epsilon para delta neutro
        atr: ATR para tolerância dinâmica
        price_volatility: Volatilidade para tolerância dinâmica
        
    Returns:
        Rótulo de absorção
    """
    classifier = AbsorptionClassifier()
    classifier.update_volatility(atr=atr, price_volatility=price_volatility)
    return classifier.classify(delta_btc, open_p, high_p, low_p, close_p, eps)


def classify_absorption_simple(delta: float, eps: float = DEFAULT_ABSORCAO_DELTA_EPS) -> str:
    """
    Classificador simples apenas por delta.
    
    Args:
        delta: Delta de volume
        eps: Epsilon
        
    Returns:
        Rótulo de absorção
    """
    return _default_classifier.classify_simple(delta, eps)