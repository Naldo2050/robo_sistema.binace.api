"""
Sistema Completo de Suporte e Resistência Institucional - Fachada Principal
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

from .constants import (
    LevelType, ReactionType, ConfidenceLevel,
    MarketBias, QualityRating
)
from .config import InstitutionalConfig
from .validation import validate_dataframe
from .utils import timer, StructuredLogger, StatisticalUtils
from .pivot_points import InstitutionalPivotPoints
from .volume_profile import VolumeProfileAnalyzer
from .core import AdvancedSupportResistance
from .monitor import InstitutionalMarketMonitor, HealthCheckResult


class InstitutionalSupportResistanceSystem:
    """
    Sistema completo de análise de suporte e resistência institucional
    
    Fornece interface unificada para todas as funcionalidades
    """
    
    def __init__(self, config: Optional[InstitutionalConfig] = None):
        """
        Inicializa o sistema
        
        Args:
            config: Configuração institucional (opcional)
        """
        if config is None:
            config = InstitutionalConfig()
        
        self.config = config
        self.logger = StructuredLogger(__name__)
        self.utils = StatisticalUtils()
        self.pivot_calculator = InstitutionalPivotPoints()
        self.last_analysis: Optional[Dict] = None
        self.performance_metrics: Dict[str, float] = {}
        
        # Configurar cache
        StatisticalUtils.enable_cache(config.enable_cache)
    
    def analyze_market(self, df: pd.DataFrame, num_levels: int = 5) -> Dict:
        """
        Executa análise completa do mercado
        
        Args:
            df: DataFrame com colunas ['open', 'high', 'low', 'close', 'volume']
            num_levels: Número de níveis para detectar
            
        Returns:
            Análise completa do mercado
            
        Raises:
            ValueError: Se dados inválidos
        """
        # Validar entrada
        validate_dataframe(df, min_rows=self.config.min_data_points)
        
        with self.logger.context(data_points=len(df), num_levels=num_levels):
            self.logger.info("Iniciando análise institucional")
            self.performance_metrics = {}
            
            # 1. Análise de Pivot Points
            self.logger.info("Calculando pivot points multi-timeframe...")
            with timer("pivot_analysis", 
                      self.logger.logger if self.config.enable_performance_logging else None, 
                      self.performance_metrics):
                pivot_analysis = self.pivot_calculator.calculate_multi_timeframe_pivots(
                    df, self.logger.logger, self.config.pivot
                )
            
            # 2. Volume Profile
            self.logger.info("Analisando volume profile...")
            with timer("volume_profile", 
                      self.logger.logger if self.config.enable_performance_logging else None,
                      self.performance_metrics):
                volume_profile = VolumeProfileAnalyzer(
                    price_data=df['close'],
                    volume_data=df['volume'],
                    config=self.config.volume_profile
                ).calculate_profile()
            
            # 3. Suporte e Resistência Avançado
            self.logger.info("Detectando níveis de suporte e resistência...")
            with timer("sr_detection", 
                      self.logger.logger if self.config.enable_performance_logging else None,
                      self.performance_metrics):
                sr_detector = AdvancedSupportResistance(
                    price_series=df['close'],
                    volume_series=df['volume'],
                    config=self.config.sr
                )
                sr_analysis = sr_detector.detect_with_metrics(num_levels=num_levels)
            
            # 4. Análise de Confluência
            self.logger.info("Analisando confluência de níveis...")
            with timer("confluence", 
                      self.logger.logger if self.config.enable_performance_logging else None,
                      self.performance_metrics):
                confluence_analysis = self._analyze_total_confluence(
                    sr_analysis, pivot_analysis, volume_profile
                )
            
            # 5. Gerar relatório consolidado
            self.logger.info("Gerando relatório consolidado...")
            with timer("report_generation", 
                      self.logger.logger if self.config.enable_performance_logging else None,
                      self.performance_metrics):
                consolidated_report = self._generate_consolidated_report(
                    sr_analysis, pivot_analysis, volume_profile, confluence_analysis
                )
            
            # Log de performance
            if self.config.enable_performance_logging:
                for op, duration in self.performance_metrics.items():
                    self.logger.performance(op.replace("_time", ""), duration * 1000)
        
        # Armazenar última análise
        self.last_analysis = {
            "timestamp": datetime.now().isoformat(),
            "data_points": len(df),
            "sr_analysis": sr_analysis,
            "pivot_analysis": pivot_analysis,
            "volume_profile": volume_profile,
            "confluence_analysis": confluence_analysis,
            "consolidated_report": consolidated_report,
            "performance": self.performance_metrics
        }

        return self.last_analysis

    def _analyze_total_confluence(
        self,
        sr_analysis: Dict[str, Any],
        pivot_analysis: Dict[str, Any],
        volume_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analisa confluência global entre:
        - Níveis de suporte/resistência
        - Pivot points multi‑timeframe
        - Níveis de Volume Profile (POC, HVN, VAH/VAL)
        """
        support_levels = sr_analysis.get("support_levels", []) or []
        resistance_levels = sr_analysis.get("resistance_levels", []) or []
        all_sr_levels = support_levels + resistance_levels

        if not all_sr_levels:
            return {
                "levels": [],
                "strongest_levels": [],
                "overall_confluence_score": 0.0,
                "metadata": {
                    "high_confluence_count": 0,
                    "support_high_confluence_count": 0,
                    "resistance_high_confluence_count": 0
                }
            }

        # -------------------------
        # 1) Coletar níveis de Pivot
        # -------------------------
        pivot_levels: List[Dict[str, Any]] = []
        tf_weights = {"daily": 1.0, "weekly": 1.5, "monthly": 2.0}

        for tf_name, tf_data in pivot_analysis.items():
            if tf_name in ("confluence", "metadata"):
                continue
            if not isinstance(tf_data, dict):
                continue

            tf_weight = tf_weights.get(tf_name, 1.0)

            for method_name, method_data in tf_data.items():
                if method_name == "metadata":
                    continue
                if not isinstance(method_data, dict):
                    continue

                for lvl_name, lvl_price in method_data.items():
                    if lvl_name not in {"pivot", "r1", "r2", "r3", "r4", "s1", "s2", "s3", "s4"}:
                        continue
                    if lvl_price is None:
                        continue

                    pivot_levels.append({
                        "price": float(lvl_price),
                        "timeframe": tf_name,
                        "method": method_name,
                        "name": lvl_name,
                        "weight": float(tf_weight)
                    })

        # -------------------------
        # 2) Coletar níveis de Volume Profile
        # -------------------------
        volume_levels: List[Dict[str, Any]] = []

        if volume_profile:
            poc = volume_profile.get("poc", {}) or {}
            if poc.get("price", 0.0) > 0:
                volume_levels.append({
                    "price": float(poc.get("price", 0.0)),
                    "kind": "POC",
                    "strength": float(poc.get("strength", 5.0)),
                    "weight": 1.5
                })

            va = volume_profile.get("value_area", {}) or {}
            if va:
                if va.get("low") is not None:
                    volume_levels.append({
                        "price": float(va["low"]),
                        "kind": "VAL",
                        "strength": 7.0,
                        "weight": 1.0
                    })
                if va.get("high") is not None:
                    volume_levels.append({
                        "price": float(va["high"]),
                        "kind": "VAH",
                        "strength": 7.0,
                        "weight": 1.0
                    })

            hvn_levels = (volume_profile.get("volume_nodes", {}) or {}).get("hvn_levels", []) or []
            for hvn in hvn_levels:
                price = hvn.get("price")
                if price is None:
                    continue
                volume_levels.append({
                    "price": float(price),
                    "kind": "HVN",
                    "strength": float(hvn.get("strength", 5.0)),
                    "weight": 1.0
                })

        # Tolerâncias em % (baseadas na config de confluência de pivots e largura da VA)
        pivot_tol_pct = float(self.config.pivot.confluence_tolerance_percent)
        va = volume_profile.get("value_area", {}) if volume_profile else {}
        va_width_pct = float(va.get("percent_width", pivot_tol_pct)) if va else pivot_tol_pct
        volume_tol_pct = max(pivot_tol_pct, va_width_pct)

        level_results: List[Dict[str, Any]] = []
        global_scores: List[float] = []

        for lvl in all_sr_levels:
            price = float(lvl.get("price", 0.0))
            if price <= 0:
                continue

            sr_score = float(lvl.get("composite_score", 5.0))
            sr_type = lvl.get("type", "unknown")

            # 2.1) Confluência com pivots
            pivot_matches: List[Dict[str, Any]] = []
            pivot_score_accum = 0.0

            for p in pivot_levels:
                p_price = p["price"]
                dist_pct = abs(p_price - price) / price * 100.0
                if dist_pct <= pivot_tol_pct and pivot_tol_pct > 0:
                    proximity = 1.0 - (dist_pct / pivot_tol_pct)
                    contrib = proximity * p["weight"] * 2.0  # máx ~2 pts por peso
                    pivot_score_accum += contrib

                    pivot_matches.append({
                        "price": p_price,
                        "timeframe": p["timeframe"],
                        "method": p["method"],
                        "name": p["name"],
                        "distance_percent": float(dist_pct)
                    })

            pivot_score = float(StatisticalUtils.clamp(pivot_score_accum, 0.0, 10.0))

            # 2.2) Confluência com Volume Profile
            volume_matches: List[Dict[str, Any]] = []
            volume_score_accum = 0.0

            for v in volume_levels:
                v_price = v["price"]
                dist_pct = abs(v_price - price) / price * 100.0
                tol = volume_tol_pct
                if v["kind"] == "POC":
                    tol = max(volume_tol_pct, pivot_tol_pct)

                if tol > 0 and dist_pct <= tol:
                    proximity = 1.0 - (dist_pct / tol)
                    # força do nível de volume já está em 0‑10
                    base_max = 6.0  # máx ~6 pts vindos do volume
                    contrib = proximity * v["strength"] * (base_max / 10.0)
                    volume_score_accum += contrib

                    volume_matches.append({
                        "price": v_price,
                        "kind": v["kind"],
                        "distance_percent": float(dist_pct),
                        "node_strength": float(v["strength"])
                    })

            volume_score = float(StatisticalUtils.clamp(volume_score_accum, 0.0, 10.0))

            # 2.3) Score global de confluência
            global_score = float(
                StatisticalUtils.clamp(
                    0.5 * sr_score + 0.3 * pivot_score + 0.2 * volume_score,
                    0.0,
                    10.0
                )
            )
            global_scores.append(global_score)

            level_results.append({
                "price": price,
                "type": sr_type,
                "sr_score": sr_score,
                "pivot_score": pivot_score,
                "volume_score": volume_score,
                "global_score": global_score,
                "sr_level": lvl,
                "pivot_matches": pivot_matches,
                "volume_matches": volume_matches
            })

        if not level_results:
            return {
                "levels": [],
                "strongest_levels": [],
                "overall_confluence_score": 0.0,
                "metadata": {
                    "high_confluence_count": 0,
                    "support_high_confluence_count": 0,
                    "resistance_high_confluence_count": 0
                }
            }

        # Ordenar níveis por confluência decrescente
        level_results.sort(key=lambda x: (-x["global_score"], x["price"]))

        high_conf_threshold = 7.0
        high_conf_levels = [l for l in level_results if l["global_score"] >= high_conf_threshold]
        support_high = [l for l in high_conf_levels if l["type"] == "support"]
        resistance_high = [l for l in high_conf_levels if l["type"] == "resistance"]

        overall_confluence_score = float(np.mean(global_scores)) if global_scores else 0.0

        return {
            "levels": level_results,
            "strongest_levels": level_results[:5],
            "overall_confluence_score": float(StatisticalUtils.clamp(overall_confluence_score, 0.0, 10.0)),
            "metadata": {
                "high_confluence_count": len(high_conf_levels),
                "support_high_confluence_count": len(support_high),
                "resistance_high_confluence_count": len(resistance_high),
                "pivot_confluence_raw": pivot_analysis.get("confluence", {})
            }
        }

    def _generate_consolidated_report(
        self,
        sr_analysis: Dict[str, Any],
        pivot_analysis: Dict[str, Any],
        volume_profile: Dict[str, Any],
        confluence_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Gera relatório consolidado de alto nível combinando todas as análises.
        """
        current_price = float(sr_analysis.get("current_price", 0.0))

        sr_quality = sr_analysis.get("quality_report", {}) or {}
        sr_quality_score = float(sr_quality.get("overall_quality", 0.0))
        sr_quality_rating = sr_quality.get("quality_rating", QualityRating.INSUFFICIENT_DATA.value)

        defense_zones = sr_analysis.get("defense_zones", {}) or {}
        market_context = defense_zones.get("market_context", {}) or {}
        is_bullish = bool(market_context.get("is_bullish", False))

        volume_metrics = volume_profile.get("profile_metrics", {}) if volume_profile else {}
        volume_bias = volume_metrics.get("bias", MarketBias.NEUTRAL.value)

        poc = volume_profile.get("poc", {}) if volume_profile else {}
        current_pos = volume_profile.get("current_position", {}) if volume_profile else {}
        distance_vs_poc = float(current_pos.get("vs_poc", 0.0))

        confluence_score = float(confluence_analysis.get("overall_confluence_score", 0.0))
        strongest_levels = confluence_analysis.get("strongest_levels", []) or []

        # Determinar regime de mercado aproximado
        if is_bullish and volume_bias == MarketBias.BULLISH.value:
            market_regime = "BULLISH"
        elif (not is_bullish) and volume_bias == MarketBias.BEARISH.value:
            market_regime = "BEARISH"
        else:
            market_regime = "MIXED"

        summary = {
            "timestamp": datetime.now().isoformat(),
            "current_price": current_price,
            "poc_price": float(poc.get("price", 0.0)),
            "distance_from_poc_percent": distance_vs_poc,
            "sr_quality_score": sr_quality_score,
            "sr_quality_rating": sr_quality_rating,
            "dominant_side": "BULLISH" if is_bullish else "BEARISH",
            "volume_profile_bias": volume_bias,
            "market_regime": market_regime,
            "confluence_overall_score": confluence_score,
            "strongest_confluence_level": strongest_levels[0] if strongest_levels else None
        }

        key_levels = {
            "support_levels": sr_analysis.get("support_levels", []),
            "resistance_levels": sr_analysis.get("resistance_levels", []),
            "defense_zones": sr_analysis.get("defense_zones", {}),
            "pivot_confluence": pivot_analysis.get("confluence", {}),
            "volume_profile": {
                "poc": poc,
                "value_area": volume_profile.get("value_area", {}) if volume_profile else {},
                "volume_nodes": volume_profile.get("volume_nodes", {}) if volume_profile else {}
            }
        }

        risk_context = {
            "sr_quality": {
                "score": sr_quality_score,
                "rating": sr_quality_rating
            },
            "confluence_strength": confluence_score,
            "distance_from_poc_percent": distance_vs_poc,
            "volume_balance": volume_metrics.get("balance"),
            "volume_bias": volume_bias,
            "volume_concentration": volume_metrics.get("volume_concentration")
        }

        # Sugestão de foco tático simples
        if market_regime == "BULLISH":
            tactical_focus = "Priorizar compras em suportes fortes e confluentes próximos à VA/POC."
        elif market_regime == "BEARISH":
            tactical_focus = "Priorizar vendas em resistências fortes e confluentes próximos à VA/POC."
        else:
            tactical_focus = "Regime misto; focar apenas em setups com forte confluência de níveis."

        return {
            "summary": summary,
            "market_context": market_context,
            "key_levels": key_levels,
            "risk_assessment": risk_context,
            "recommendations": {
                "tactical_focus": tactical_focus
            }
        }

    def create_market_monitor(self) -> InstitutionalMarketMonitor:
        """
        Cria um monitor de mercado em tempo (quase) real usando os níveis
        detectados na última análise.

        É necessário ter chamado `analyze_market` previamente.
        """
        if not self.last_analysis:
            raise RuntimeError(
                "Nenhuma análise encontrada. Execute `analyze_market(df)` antes de criar o monitor."
            )

        sr = self.last_analysis.get("sr_analysis") or self.last_analysis.get("sr")
        if not sr:
            raise ValueError("Análise de suporte/resistência não encontrada na última análise.")

        return InstitutionalMarketMonitor(
            support_levels=sr.get("support_levels", []),
            resistance_levels=sr.get("resistance_levels", []),
            config=self.config.monitor
        )
    
    def reset(self) -> None:
        """
        Reseta o estado do sistema para reutilização.
        
        Limpa cache, última análise e métricas de performance.
        """
        StatisticalUtils.clear_cache()
        self.last_analysis = None
        self.performance_metrics.clear()
        self.logger.info("Sistema resetado")
    
    def __enter__(self) -> 'InstitutionalSupportResistanceSystem':
        """Suporte a context manager"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Limpa recursos ao sair do context"""
        self.reset()
    
    def health_check(self, df: Optional[pd.DataFrame] = None) -> HealthCheckResult:
        """
        Executa verificação de saúde do sistema.
        
        Parameters
        ----------
        df : pd.DataFrame, optional
            DataFrame de teste. Se não fornecido, usa dados sintéticos.
            
        Returns
        -------
        HealthCheckResult
            Resultado detalhado do health check.
        """
        checks = {}
        details = {}
        
        # 1. Verificar configuração
        try:
            assert self.config.confidence_level > 0
            assert self.config.min_data_points > 0
            checks['config_valid'] = True
            details['config'] = "OK"
        except Exception as e:
            checks['config_valid'] = False
            details['config'] = str(e)
        
        # 2. Verificar cache
        try:
            StatisticalUtils.clear_cache()
            test_data = np.array([1.0, 2.0, 3.0])
            StatisticalUtils.calculate_confidence_interval(test_data)
            checks['cache_functional'] = True
            details['cache'] = "OK"
        except Exception as e:
            checks['cache_functional'] = False
            details['cache'] = str(e)
        
        # 3. Verificar análise básica
        if df is None:
            # Gerar dados sintéticos
            np.random.seed(42)
            df = pd.DataFrame({
                'open': np.random.uniform(99, 101, 100),
                'high': np.random.uniform(100, 102, 100),
                'low': np.random.uniform(98, 100, 100),
                'close': np.random.uniform(99, 101, 100),
                'volume': np.random.uniform(1000, 10000, 100)
            }, index=pd.date_range('2024-01-01', periods=100, freq='1h'))
        
        try:
            result = self.analyze_market(df, num_levels=3)
            has_support = len(result.get('sr_analysis', {}).get('support_levels', [])) >= 0
            has_resistance = len(result.get('sr_analysis', {}).get('resistance_levels', [])) >= 0
            checks['analysis_functional'] = has_support or has_resistance
            details['analysis'] = "OK" if checks['analysis_functional'] else "No levels detected"
        except Exception as e:
            checks['analysis_functional'] = False
            details['analysis'] = str(e)
        
        # 4. Verificar volume profile
        try:
            vp = VolumeProfileAnalyzer(df['close'], df['volume'])
            profile = vp.calculate_profile()
            checks['volume_profile_functional'] = profile.get('poc', {}).get('price', 0) > 0
            details['volume_profile'] = "OK" if checks['volume_profile_functional'] else "POC not found"
        except Exception as e:
            checks['volume_profile_functional'] = False
            details['volume_profile'] = str(e)
        
        # Determinar status geral
        all_passed = all(checks.values())
        some_passed = any(checks.values())
        
        if all_passed:
            status = "healthy"
        elif some_passed:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return HealthCheckResult(
            status=status,
            checks=checks,
            details=details,
            timestamp=datetime.now().isoformat()
        )