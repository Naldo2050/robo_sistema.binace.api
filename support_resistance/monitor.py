"""
Monitoramento Institucional em Tempo Real
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Deque
from collections import deque
from datetime import datetime

from .constants import (
    LevelType, ReactionType, ConfidenceLevel, 
    MarketBias, CONSTANTS
)
from .config import MonitorConfig
from .utils import StatisticalUtils


@dataclass
class HealthCheckResult:
    """Resultado de health check do sistema"""
    status: str  # "healthy", "degraded", "unhealthy"
    checks: Dict[str, bool]
    details: Dict[str, str]
    timestamp: str


class InstitutionalMarketMonitor:
    """Monitoramento institucional em tempo real com métricas avançadas"""
    
    def __init__(self, support_levels: List[Dict], resistance_levels: List[Dict],
                 config: Optional[MonitorConfig] = None):
        """
        Inicializa o monitor
        
        Args:
            support_levels: Lista de níveis de suporte
            resistance_levels: Lista de níveis de resistência
            config: Configuração do monitor
        """
        if config is None:
            config = MonitorConfig()
        
        self.config = config
        self.support_levels = support_levels
        self.resistance_levels = resistance_levels
        
        # Usar deque com limite máximo
        self.level_tests: deque = deque(maxlen=config.max_test_history)
        self.price_history: deque = deque(maxlen=config.lookback_ticks)
        self.volume_history: deque = deque(maxlen=config.lookback_ticks)
        self.delta_history: deque = deque(maxlen=config.lookback_ticks)
        
        self.stats = {
            "total_tests": 0,
            "successful_defenses": 0,
            "failed_defenses": 0,
            "breakouts": 0,
            "false_breakouts": 0
        }
    
    def process_tick(self, price: float, volume: float, delta: float,
                    order_flow: Optional[Dict] = None) -> Optional[Dict]:
        """
        Processa um tick contra os níveis monitorados
        
        Args:
            price: Preço atual
            volume: Volume do tick
            delta: Delta (compra - venda)
            order_flow: Dados de order flow (opcional)
            
        Returns:
            Sinal se nível foi testado, None caso contrário
        """
        # deque automaticamente remove itens antigos, não precisa de pop manual
        self.price_history.append(price)
        self.volume_history.append(volume)
        self.delta_history.append(delta)
        
        tolerance_pct = self.config.tolerance_percent / 100
        
        for level in self.support_levels:
            tolerance = level["price"] * tolerance_pct
            if abs(price - level["price"]) <= tolerance:
                signal = self._analyze_level_test(
                    price, volume, delta, level, LevelType.SUPPORT, order_flow
                )
                self._update_stats(signal)
                return signal
        
        for level in self.resistance_levels:
            tolerance = level["price"] * tolerance_pct
            if abs(price - level["price"]) <= tolerance:
                signal = self._analyze_level_test(
                    price, volume, delta, level, LevelType.RESISTANCE, order_flow
                )
                self._update_stats(signal)
                return signal
        
        return None
    
    def _analyze_level_test(
        self, 
        price: float, 
        volume: float, 
        delta: float,
        level: Dict, 
        level_type: LevelType,
        order_flow: Optional[Dict] = None
    ) -> Dict:
        """
        Analisa teste de nível de forma unificada.
        
        Args:
            price: Preço atual
            volume: Volume do tick
            delta: Delta (compra - venda)
            level: Dados do nível
            level_type: SUPPORT ou RESISTANCE
            order_flow: Dados de order flow (opcional)
            
        Returns:
            Dict com análise completa do teste
        """
        is_support = (level_type == LevelType.SUPPORT)
        
        delta_ratio = delta / volume if volume > 0 else 0
        
        # Lógica de reação invertida para resistência
        effective_delta = delta_ratio if is_support else -delta_ratio
        
        if effective_delta > self.config.strong_delta_threshold:
            reaction = ReactionType.STRONG_DEFENSE
            confidence = ConfidenceLevel.HIGH
        elif effective_delta > self.config.moderate_delta_threshold:
            reaction = ReactionType.DEFENSE
            confidence = ConfidenceLevel.MEDIUM
        elif effective_delta < -self.config.strong_delta_threshold:
            reaction = ReactionType.WEAK_DEFENSE
            confidence = ConfidenceLevel.HIGH
        else:
            reaction = ReactionType.NEUTRAL
            confidence = ConfidenceLevel.LOW
        
        context = self._analyze_test_context(price, level_type.value.upper())
        volume_analysis = self._analyze_volume_profile(volume, level)
        
        signal = {
            "timestamp": datetime.now().isoformat(),
            "level_type": level_type.value.upper(),
            "level_price": level["price"],
            "level_strength": level.get("composite_score", 5.0),
            "test_price": price,
            "deviation_percent": StatisticalUtils.safe_divide(
                price - level["price"], level["price"], 0.0
            ) * 100,
            "volume": volume,
            "delta": delta,
            "delta_percent": delta_ratio * 100,
            "reaction": reaction.value,
            "confidence": confidence.value,
            "context": context,
            "volume_analysis": volume_analysis,
            "order_flow_analysis": order_flow,
            "recommendation": self._generate_recommendation(
                reaction, level, context, is_resistance=not is_support
            )
        }
        
        self.level_tests.append(signal)
        return signal
    
    # Métodos wrapper para retrocompatibilidade
    def _analyze_support_test(self, price, volume, delta, level, order_flow) -> Dict:
        """Deprecated: Use _analyze_level_test com LevelType.SUPPORT"""
        return self._analyze_level_test(price, volume, delta, level, LevelType.SUPPORT, order_flow)
    
    def _analyze_resistance_test(self, price, volume, delta, level, order_flow) -> Dict:
        """Deprecated: Use _analyze_level_test com LevelType.RESISTANCE"""
        return self._analyze_level_test(price, volume, delta, level, LevelType.RESISTANCE, order_flow)
    
    def _analyze_test_context(self, price: float, level_type: str) -> Dict:
        """Analisa contexto do teste"""
        if len(self.price_history) < 10:
            return {"trend": "UNKNOWN", "volatility": "UNKNOWN", "momentum": "NEUTRAL"}
        
        recent_prices = list(self.price_history)[-10:]
        first_price = recent_prices[0]
        
        if first_price > 0:
            price_change = (recent_prices[-1] - first_price) / first_price * 100
        else:
            price_change = 0
        
        strong_threshold = self.config.trend_strong_threshold
        weak_threshold = self.config.trend_weak_threshold
        
        if price_change > strong_threshold:
            trend = "STRONG_UP"
        elif price_change > weak_threshold:
            trend = "UP"
        elif price_change < -strong_threshold:
            trend = "STRONG_DOWN"
        elif price_change < -weak_threshold:
            trend = "DOWN"
        else:
            trend = "SIDEWAYS"
        
        if len(self.price_history) >= 20:
            recent_20 = list(self.price_history)[-20:]
            mean_20 = np.mean(recent_20)
            volatility = StatisticalUtils.safe_divide(np.std(recent_20), mean_20, 0.0) * 100
        else:
            volatility = 0
        
        if len(self.delta_history) >= 5 and len(self.volume_history) >= 5:
            recent_delta = np.mean(list(self.delta_history)[-5:])
            avg_volume = np.mean(list(self.volume_history)[-5:])
            momentum = StatisticalUtils.safe_divide(recent_delta, avg_volume, 0.0)
        else:
            momentum = 0
        
        high_vol_threshold = self.config.high_volatility_threshold
        
        return {
            "trend": trend,
            "trend_strength": abs(price_change),
            "volatility": volatility,
            "volatility_category": "HIGH" if volatility > high_vol_threshold else "LOW",
            "momentum": float(momentum),
            "momentum_direction": "BULLISH" if momentum > 0.1 else "BEARISH" if momentum < -0.1 else "NEUTRAL",
            "test_count": len([t for t in self.level_tests if t["level_type"] == level_type])
        }
    
    def _analyze_volume_profile(self, current_volume: float, level: Dict) -> Dict:
        """Analisa perfil de volume do teste"""
        if not self.volume_history:
            return {"volume_ratio": 1.0, "category": "NORMAL", "significance": "LOW"}
        
        if len(self.volume_history) >= 20:
            avg_volume = np.mean(list(self.volume_history)[-20:])
        else:
            avg_volume = np.mean(self.volume_history)
        
        volume_ratio = StatisticalUtils.safe_divide(current_volume, avg_volume, 1.0)
        
        if volume_ratio > CONSTANTS.EXTREME_VOLUME_RATIO:
            category = "EXTREME"
        elif volume_ratio > CONSTANTS.HIGH_VOLUME_RATIO:
            category = "HIGH"
        elif volume_ratio > CONSTANTS.ELEVATED_VOLUME_RATIO:
            category = "ELEVATED"
        elif volume_ratio < CONSTANTS.LOW_VOLUME_RATIO:
            category = "LOW"
        else:
            category = "NORMAL"
        
        significance = "HIGH" if category in ["EXTREME", "HIGH"] else "MEDIUM" if category == "ELEVATED" else "LOW"
        
        return {
            "volume_ratio": float(volume_ratio),
            "category": category,
            "compared_to_average": f"{volume_ratio:.1f}x",
            "significance": significance
        }
    
    def _generate_recommendation(self, reaction: ReactionType, level: Dict, 
                                context: Dict, is_resistance: bool = False) -> str:
        """Gera recomendação baseada na análise"""
        level_strength = level.get("composite_score", 5)
        
        if reaction == ReactionType.STRONG_DEFENSE:
            if is_resistance:
                return f"SELL - Resistência forte em {level['price']:.2f} (Score: {level_strength:.1f})"
            else:
                return f"BUY - Suporte forte em {level['price']:.2f} (Score: {level_strength:.1f})"
        
        elif reaction == ReactionType.WEAK_DEFENSE:
            if is_resistance:
                return f"WATCH FOR BREAKOUT - Resistência fraca em {level['price']:.2f}"
            else:
                return f"WATCH FOR BREAKDOWN - Suporte fraco em {level['price']:.2f}"
        
        elif reaction == ReactionType.DEFENSE:
            if is_resistance:
                return f"CAUTIOUS SELL - Resistência média em {level['price']:.2f}"
            else:
                return f"CAUTIOUS BUY - Suporte médio em {level['price']:.2f}"
        
        else:
            return f"MONITOR - Teste neutro em {level['price']:.2f}. Aguardar confirmação."
    
    def _update_stats(self, signal: Dict) -> None:
        """Atualiza estatísticas do monitor"""
        self.stats["total_tests"] += 1
        
        reaction = signal.get("reaction", "")
        
        if ReactionType.STRONG_DEFENSE.value in reaction or reaction == ReactionType.DEFENSE.value:
            self.stats["successful_defenses"] += 1
        elif ReactionType.WEAK_DEFENSE.value in reaction:
            self.stats["failed_defenses"] += 1
    
    def get_summary_report(self) -> Dict:
        """Retorna relatório resumido do monitor"""
        if not self.level_tests:
            return {"status": "NO_TESTS", "stats": self.stats}
        
        recent_tests = list(self.level_tests)[-10:] if len(self.level_tests) >= 10 else list(self.level_tests)
        
        total = self.stats["total_tests"]
        success_rate = StatisticalUtils.safe_divide(self.stats["successful_defenses"], total, 0.0) * 100
        
        level_tests_count: Dict[str, int] = {}
        for test in self.level_tests:
            key = f"{test['level_type']}_{test['level_price']:.2f}"
            level_tests_count[key] = level_tests_count.get(key, 0) + 1
        
        most_tested = sorted(level_tests_count.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "status": "ACTIVE",
            "stats": self.stats,
            "success_rate": float(success_rate),
            "most_tested_levels": most_tested,
            "recent_activity": len(recent_tests),
            "last_test": self.level_tests[-1] if self.level_tests else None,
            "monitoring_since": self.level_tests[0]["timestamp"] if self.level_tests else None
        }
    
    def get_test_history(self, as_dataframe: bool = True) -> Union[pd.DataFrame, List[Dict]]:
        """Retorna histórico de testes"""
        if not self.level_tests:
            return pd.DataFrame() if as_dataframe else []
        
        if as_dataframe:
            return pd.DataFrame(self.level_tests)
        else:
            return list(self.level_tests)
    
    def reset(self) -> None:
        """Reseta o monitor mantendo os níveis configurados"""
        self.level_tests.clear()
        self.price_history.clear()
        self.volume_history.clear()
        self.delta_history.clear()
        self.stats = {
            "total_tests": 0,
            "successful_defenses": 0,
            "failed_defenses": 0,
            "breakouts": 0,
            "false_breakouts": 0
        }
    
    def update_levels(self, support_levels: List[Dict], 
                     resistance_levels: List[Dict]) -> None:
        """Atualiza níveis monitorados sem perder histórico"""
        self.support_levels = support_levels
        self.resistance_levels = resistance_levels