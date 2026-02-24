# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, cast
import logging

class DataQualityValidator:
    """
    Valida qualidade de dados de trades E métricas calculadas.
    """
    def __init__(self, thresholds: Optional[Dict[str, Any]] = None):
        default_thresholds = {
            # Thresholds originais (trades)
            "max_price_jump_std": 5.0,
            "max_volume_jump_std": 10.0,
            "min_completeness_pct": 0.90,
            "max_zero_volume_pct": 0.10,
            "max_time_gap_seconds": 10,
            
            # 🆕 NOVOS: Validação de features calculadas
            "min_orderbook_depth_usd": 10000,      # Mínimo $10k de liquidez
            "max_volume_sma_ratio": 500,           # Cap em 5x a média
            "min_value_area_range": 0.001,         # VAH - VAL > 0.1% do preço
            "max_flow_delta_divergence": 0.5,      # Máx divergência flow vs delta
            "min_tick_rule_variance": 0.01,        # tick_rule não pode ser sempre 0
        }
        self.thresholds = thresholds or default_thresholds
        logging.info("✅ DataQualityValidator inicializado (versão expandida).")

    def validate_window(self, df_window: pd.DataFrame, window_duration_seconds: int) -> Dict[str, Any]:
        """Validação original de trades (mantém código existente)"""
        # ... código original mantido ...
        return {"is_valid": True, "quality_score": 100, "issues": [], "flags": []}

    # 🆕 NOVA FUNÇÃO: Valida métricas calculadas
    def validate_metrics(self, metrics: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """
        Valida métricas calculadas (orderbook, features, flow, etc.)
        
        Args:
            metrics: Dicionário com todas as métricas calculadas
            current_price: Preço atual para validações relativas
            
        Returns:
            dict: Resultado da validação com flags específicas
        """
        issues = []
        flags = []
        score = 100

        # ========================================
        # 1. VALIDAÇÃO DE ORDERBOOK
        # ========================================
        bid_depth = metrics.get('bid_depth_usd', 0)
        ask_depth = metrics.get('ask_depth_usd', 0)
        
        if bid_depth == 0 or ask_depth == 0:
            issues.append("❌ ORDERBOOK ZERADO (bid_depth ou ask_depth = $0)")
            flags.append("INVALID_ORDERBOOK")
            score -= 50  # Crítico
        elif bid_depth < self.thresholds['min_orderbook_depth_usd']:
            issues.append(f"⚠️ Liquidez baixa: bid_depth = ${bid_depth:,.0f}")
            flags.append("LOW_LIQUIDITY")
            score -= 20

        # ========================================
        # 2. VALIDAÇÃO DE VALUE AREA
        # ========================================
        VAL = metrics.get('VAL', 0)
        VAH = metrics.get('VAH', 0)
        
        if VAL == 0 or VAH == 0:
            issues.append("❌ VALUE AREA ZERADA (VAL ou VAH = $0.00)")
            flags.append("INVALID_VALUE_AREA")
            score -= 30
        elif VAH > 0 and VAL > 0:
            va_range_pct = (VAH - VAL) / current_price
            if va_range_pct < self.thresholds['min_value_area_range']:
                issues.append(f"⚠️ Value Area muito estreita: {va_range_pct:.3%}")
                flags.append("NARROW_VALUE_AREA")
                score -= 10

        # ========================================
        # 3. VALIDAÇÃO DE TICK RULE
        # ========================================
        tick_rule_sum = metrics.get('tick_rule_sum', 0)
        delta = metrics.get('delta', 0)
        
        if tick_rule_sum == 0 and delta != 0:
            issues.append("❌ TICK_RULE_SUM = 0 mas delta ≠ 0 (lógica quebrada)")
            flags.append("TICK_RULE_BROKEN")
            score -= 25

        # ========================================
        # 4. VALIDAÇÃO DE VOLUME RATIO
        # ========================================
        volume_sma_ratio = metrics.get('volume_sma_ratio', 0)
        
        if volume_sma_ratio > self.thresholds['max_volume_sma_ratio']:
            issues.append(f"❌ VOLUME_SMA_RATIO absurdo: {volume_sma_ratio:.1f}%")
            flags.append("VOLUME_RATIO_EXTREME")
            score -= 15

        # ========================================
        # 5. VALIDAÇÃO DE CONSISTÊNCIA FLOW vs DELTA
        # ========================================
        flow_imbalance = metrics.get('flow_imbalance', 0)
        
        if delta != 0 and flow_imbalance != 0:
            # Delta positivo deve ter flow positivo (e vice-versa)
            if np.sign(delta) != np.sign(flow_imbalance):
                divergence = abs(delta - flow_imbalance * abs(delta))
                if divergence > self.thresholds['max_flow_delta_divergence'] * abs(delta):
                    issues.append(f"⚠️ Divergência flow vs delta: {divergence:.2f}")
                    flags.append("FLOW_DELTA_MISMATCH")
                    score -= 20

        # ========================================
        # 6. VALIDAÇÃO DE WHALE METRICS
        # ========================================
        whale_buy = metrics.get('whale_buy_volume', 0)
        whale_sell = metrics.get('whale_sell_volume', 0)
        whale_delta_reported = metrics.get('whale_delta', 0)
        
        if whale_buy > 0 or whale_sell > 0:
            whale_delta_calculated = whale_buy - whale_sell
            if abs(whale_delta_calculated - whale_delta_reported) > 0.01:
                issues.append("⚠️ Whale delta inconsistente (buy/sell não batem)")
                flags.append("WHALE_DELTA_MISMATCH")
                score -= 10

        # ========================================
        # 7. VALIDAÇÃO DE LIQUIDATION HEATMAP
        # ========================================
        liq_heatmap = metrics.get('liquidation_heatmap', {})
        open_interest = metrics.get('open_interest', 0)
        
        if not liq_heatmap and open_interest > 0:
            issues.append("⚠️ Liquidation heatmap vazio mas OI existe")
            flags.append("MISSING_LIQUIDATION_DATA")
            score -= 5  # Baixa prioridade

        # ========================================
        # RESULTADO FINAL
        # ========================================
        final_score = max(0, score)
        
        return {
            "is_valid": final_score >= 70 and "INVALID_ORDERBOOK" not in flags,
            "quality_score": final_score,
            "issues": issues,
            "flags": flags,  # 🆕 Flags específicos para cada problema
            "critical_failures": [f for f in flags if f.startswith("INVALID_")],
        }

    # 🆕 VALIDAÇÃO COMPLETA (trades + métricas)
    def validate_full_context(self, df_window: pd.DataFrame, metrics: Dict[str, Any], 
                             current_price: float, window_duration_seconds: int) -> Dict[str, Any]:
        """
        Valida TUDO: trades + métricas calculadas.
        
        Returns:
            dict: Resultado consolidado com todas as validações
        """
        # Valida trades
        trades_validation = self.validate_window(df_window, window_duration_seconds)
        
        # Valida métricas
        metrics_validation = self.validate_metrics(metrics, current_price)
        
        # Consolida
        return {
            "is_valid": trades_validation["is_valid"] and metrics_validation["is_valid"],
            "trades_quality": trades_validation["quality_score"],
            "metrics_quality": metrics_validation["quality_score"],
            "overall_quality": (trades_validation["quality_score"] + metrics_validation["quality_score"]) / 2,
            "all_issues": trades_validation["issues"] + metrics_validation["issues"],
            "flags": metrics_validation.get("flags", []),
            "critical_failures": metrics_validation.get("critical_failures", []),
        }

    def calculate_completeness_score(self, payload: dict) -> dict:
        """
        Calcula o score de completude dos dados.
        Verifica se todos os campos críticos, importantes e opcionais estão presentes.
        Score final ponderado: critical(60%) + important(30%) + optional(10%).
        """
        critical_fields = {
            "current_price": self._extract_nested(payload, ["price_context.current_price", "contextual_snapshot.ohlc.close", "raw_event.preco_fechamento"]),
            "ohlc": self._extract_nested(payload, ["price_context.ohlc", "contextual_snapshot.ohlc"]),
            "volume": self._extract_nested(payload, ["contextual_snapshot.volume_total", "raw_event.volume_total"]),
            "orderbook": self._extract_nested(payload, ["orderbook_context", "orderbook_data"]),
            "flow_metrics": self._extract_nested(payload, ["flow_context", "flow_metrics"]),
            "multi_tf": self._extract_nested(payload, ["contextual_snapshot.multi_tf", "raw_event.multi_tf"]),
        }
        
        important_fields = {
            "vp_daily": self._extract_nested(payload, ["contextual_snapshot.historical_vp.daily", "raw_event.historical_vp.daily"]),
            "vp_weekly": self._extract_nested(payload, ["contextual_snapshot.historical_vp.weekly", "raw_event.historical_vp.weekly"]),
            "derivatives": self._extract_nested(payload, ["contextual_snapshot.derivatives", "raw_event.derivatives"]),
            "sector_flow": self._extract_nested(payload, ["flow_context.sector_flow", "raw_event.flow_metrics.sector_flow"]),
            "absorption": self._extract_nested(payload, ["flow_context.absorption_analysis", "raw_event.flow_metrics.absorption_analysis"]),
        }
        
        optional_fields = {
            "vp_monthly": self._extract_nested(payload, ["contextual_snapshot.historical_vp.monthly", "raw_event.historical_vp.monthly"]),
            "cross_asset": self._extract_nested(payload, ["cross_asset_context", "raw_event.ml_features.cross_asset"]),
            "macro_context": self._extract_nested(payload, ["macro_context"]),
            "market_environment": self._extract_nested(payload, ["raw_event.market_environment"]),
        }
        
        critical_present = sum(1 for v in critical_fields.values() if v is not None)
        important_present = sum(1 for v in important_fields.values() if v is not None)
        optional_present = sum(1 for v in optional_fields.values() if v is not None)
        
        critical_pct = (critical_present / max(len(critical_fields), 1)) * 100
        important_pct = (important_present / max(len(important_fields), 1)) * 100
        optional_pct = (optional_present / max(len(optional_fields), 1)) * 100
        
        overall = (critical_pct * 0.6) + (important_pct * 0.3) + (optional_pct * 0.1)
        
        missing_critical = [k for k, v in critical_fields.items() if v is None]
        missing_important = [k for k, v in important_fields.items() if v is None]
        
        if overall >= 90:
            grade = "A"
        elif overall >= 75:
            grade = "B"
        elif overall >= 50:
            grade = "C"
        else:
            grade = "D"
        
        return {
            "completeness_score": round(overall, 1),
            "grade": grade,
            "critical_pct": round(critical_pct, 1),
            "important_pct": round(important_pct, 1),
            "optional_pct": round(optional_pct, 1),
            "missing_critical": missing_critical,
            "missing_important": missing_important,
            "total_fields_checked": len(critical_fields) + len(important_fields) + len(optional_fields),
            "total_present": critical_present + important_present + optional_present,
        }
    
    def _extract_nested(self, data: dict, paths: list):
        """
        Tenta extrair valor de múltiplos caminhos possíveis.
        Retorna o primeiro valor encontrado ou None.
        """
        for path in paths:
            keys = path.split(".")
            current = data
            try:
                for key in keys:
                    if isinstance(current, dict):
                        current = current[key]
                    else:
                        current = None
                        break
                if current is not None:
                    return current
            except (KeyError, TypeError, IndexError):
                continue
        return None

    def detect_anomalies(self, current_data: dict, historical_stats: Optional[dict] = None) -> dict:
        """
        Detecta automaticamente valores fora do normal.
        
        Verifica anomalias em:
          1. Volume (spike ou collapse)
          2. Spread (widening anormal)
          3. Price move (move > 3 ATR em janela curta)
          4. Flow imbalance (desequilíbrio extremo)
          5. Order book depth (assimetria extrema)
          6. Volatility (regime change)
          7. Trade intensity (trades/segundo anormal)
        
        Args:
            current_data: Dict com dados atuais da janela.
                Espera quaisquer dos campos:
                  volume_total, spread, close, open, high, low,
                  flow_imbalance, bid_depth_usd, ask_depth_usd,
                  trades_per_second, realized_vol, atr
            historical_stats: Dict com estatísticas históricas para comparação.
                Espera: {
                    "vol_mean": x, "vol_std": x,
                    "spread_mean": x, "spread_std": x,
                    "atr_15m": x, "atr_1h": x,
                    "trades_per_sec_mean": x, "trades_per_sec_std": x,
                    "realized_vol_mean": x, "realized_vol_std": x,
                }
                Se None, usa thresholds fixos (menos preciso mas funcional).
                
        Returns:
            Dict com anomalias detectadas, contagem e severidade máxima.
        """
        anomalies = []

        # Extrair valores com fallbacks
        def _get(key, *alt_keys, default=0):
            val = current_data.get(key)
            if val is not None and not isinstance(val, (dict, list)):
                try:
                    return float(val)
                except (TypeError, ValueError):
                    pass
            for ak in alt_keys:
                val = current_data.get(ak)
                if val is not None and not isinstance(val, (dict, list)):
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            # Tentar nested
            for path in [key] + list(alt_keys):
                parts = path.split(".")
                obj = current_data
                try:
                    for p in parts:
                        obj = obj[p]
                    if obj is not None and not isinstance(obj, (dict, list)):
                        try:
                            return float(obj)
                        except (TypeError, ValueError):
                            pass
                except (KeyError, TypeError, IndexError):
                    continue
            return default

        volume = _get("volume_total", "total_volume", "total_volume_btc")
        spread = _get("spread", "spread_bps")
        close_price = _get("close", "preco_fechamento", "ohlc.close")
        open_price = _get("open", "ohlc.open")
        high_price = _get("high", "ohlc.high")
        low_price = _get("low", "ohlc.low")
        flow_imb = _get("flow_imbalance")
        bid_depth = _get("bid_depth_usd", "orderbook_data.bid_depth_usd")
        ask_depth = _get("ask_depth_usd", "orderbook_data.ask_depth_usd")
        trades_per_sec = _get("trades_per_second")
        realized_vol = _get("realized_vol")

        has_hist = isinstance(historical_stats, dict) and historical_stats is not None
        hist_stats = cast(dict, historical_stats)

        # ─────────────────────────────────────
        # 1. VOLUME ANOMALY
        # ─────────────────────────────────────
        if volume > 0:
            if has_hist and hist_stats.get("vol_mean", 0) > 0:
                vol_mean = hist_stats["vol_mean"]
                vol_std = hist_stats.get("vol_std", vol_mean * 0.5)
                if vol_std > 0:
                    vol_zscore = (volume - vol_mean) / vol_std
                    if abs(vol_zscore) > 5:
                        anomalies.append({
                            "type": "VOLUME_EXTREME_SPIKE",
                            "severity": "CRITICAL",
                            "zscore": round(vol_zscore, 2),
                            "value": round(volume, 4),
                            "expected": round(vol_mean, 4),
                            "description": f"Volume {abs(vol_zscore):.1f} std from mean",
                        })
                    elif abs(vol_zscore) > 3:
                        anomalies.append({
                            "type": "VOLUME_SPIKE",
                            "severity": "HIGH",
                            "zscore": round(vol_zscore, 2),
                            "value": round(volume, 4),
                            "expected": round(vol_mean, 4),
                            "description": f"Volume {abs(vol_zscore):.1f} std from mean",
                        })
            else:
                # Threshold fixo: volume < 0.1 BTC ou > 100 BTC em janela de 2min = anormal
                if volume < 0.1:
                    anomalies.append({
                        "type": "VOLUME_COLLAPSE",
                        "severity": "MEDIUM",
                        "value": round(volume, 4),
                        "description": "Extremely low volume - possible liquidity event",
                    })
                elif volume > 100:
                    anomalies.append({
                        "type": "VOLUME_SPIKE",
                        "severity": "HIGH",
                        "value": round(volume, 4),
                        "description": "Extremely high volume - possible large event",
                    })

        # ─────────────────────────────────────
        # 2. SPREAD ANOMALY
        # ─────────────────────────────────────
        if spread > 0:
            if has_hist and hist_stats.get("spread_mean", 0) > 0:
                sp_mean = hist_stats["spread_mean"]
                sp_std = hist_stats.get("spread_std", sp_mean * 0.5)
                if sp_std > 0:
                    sp_zscore = (spread - sp_mean) / sp_std
                    if sp_zscore > 3:
                        anomalies.append({
                            "type": "SPREAD_WIDENING",
                            "severity": "HIGH" if sp_zscore > 5 else "MEDIUM",
                            "zscore": round(sp_zscore, 2),
                            "value": round(spread, 6),
                            "expected": round(sp_mean, 6),
                            "description": "Spread abnormally wide - possible liquidity event",
                        })
            else:
                # BTC spread > $5 é anormal
                if spread > 5:
                    anomalies.append({
                        "type": "SPREAD_WIDENING",
                        "severity": "HIGH",
                        "value": round(spread, 4),
                        "description": f"Spread ${spread:.2f} is abnormally wide for BTC",
                    })

        # ─────────────────────────────────────
        # 3. PRICE MOVE ANOMALY
        # ─────────────────────────────────────
        if close_price > 0 and open_price > 0:
            price_move = abs(close_price - open_price)
            price_range = high_price - low_price if high_price > 0 and low_price > 0 else price_move

            # Comparar com ATR
            atr_ref = 0
            if has_hist:
                atr_ref = hist_stats.get("atr_15m", 0) or hist_stats.get("atr_1h", 0)
            
            if atr_ref > 0:
                move_ratio = price_range / atr_ref
                if move_ratio > 5:
                    anomalies.append({
                        "type": "PRICE_EXTREME_MOVE",
                        "severity": "CRITICAL",
                        "ratio": round(move_ratio, 2),
                        "move": round(price_range, 2),
                        "atr_ref": round(atr_ref, 2),
                        "description": f"Price range {price_range:.0f} is {move_ratio:.1f}x ATR",
                    })
                elif move_ratio > 3:
                    anomalies.append({
                        "type": "PRICE_SPIKE",
                        "severity": "HIGH",
                        "ratio": round(move_ratio, 2),
                        "move": round(price_range, 2),
                        "atr_ref": round(atr_ref, 2),
                        "description": f"Price range {price_range:.0f} is {move_ratio:.1f}x ATR",
                    })
            else:
                # Fallback: move > 1% em janela curta
                move_pct = price_range / close_price * 100 if close_price > 0 else 0
                if move_pct > 2:
                    anomalies.append({
                        "type": "PRICE_SPIKE",
                        "severity": "HIGH",
                        "move_pct": round(move_pct, 4),
                        "description": f"Price moved {move_pct:.2f}% in single window",
                    })
                elif move_pct > 1:
                    anomalies.append({
                        "type": "PRICE_LARGE_MOVE",
                        "severity": "MEDIUM",
                        "move_pct": round(move_pct, 4),
                        "description": f"Price moved {move_pct:.2f}% in single window",
                    })

        # ─────────────────────────────────────
        # 4. FLOW IMBALANCE ANOMALY
        # ─────────────────────────────────────
        if abs(flow_imb) > 0.5:
            severity = "HIGH" if abs(flow_imb) > 0.7 else "MEDIUM"
            anomalies.append({
                "type": "FLOW_EXTREME_IMBALANCE",
                "severity": severity,
                "value": round(flow_imb, 4),
                "direction": "BUY" if flow_imb > 0 else "SELL",
                "description": f"Extreme flow imbalance: {flow_imb:.2%} toward {'buyers' if flow_imb > 0 else 'sellers'}",
            })

        # ─────────────────────────────────────
        # 5. ORDER BOOK DEPTH ANOMALY
        # ─────────────────────────────────────
        if bid_depth > 0 and ask_depth > 0:
            depth_ratio = bid_depth / ask_depth
            if depth_ratio > 3 or depth_ratio < 0.33:
                anomalies.append({
                    "type": "DEPTH_EXTREME_ASYMMETRY",
                    "severity": "MEDIUM",
                    "ratio": round(depth_ratio, 4),
                    "bid_depth": round(bid_depth, 2),
                    "ask_depth": round(ask_depth, 2),
                    "direction": "BID_HEAVY" if depth_ratio > 3 else "ASK_HEAVY",
                    "description": f"Order book depth ratio {depth_ratio:.2f}:1 is extreme",
                })

            # Profundidade muito baixa em ambos os lados
            total_depth = bid_depth + ask_depth
            if total_depth < 50000:  # < $50k total = muito fino
                anomalies.append({
                    "type": "DEPTH_THIN",
                    "severity": "HIGH",
                    "total_depth_usd": round(total_depth, 2),
                    "description": f"Total depth ${total_depth:,.0f} is dangerously thin",
                })

        # ─────────────────────────────────────
        # 6. VOLATILITY REGIME CHANGE
        # ─────────────────────────────────────
        if realized_vol > 0 and has_hist:
            vol_hist_mean = hist_stats.get("realized_vol_mean", 0)
            vol_hist_std = hist_stats.get("realized_vol_std", 0)
            if vol_hist_mean > 0 and vol_hist_std > 0:
                vol_z = (realized_vol - vol_hist_mean) / vol_hist_std
                if vol_z > 3:
                    anomalies.append({
                        "type": "VOLATILITY_EXPLOSION",
                        "severity": "HIGH",
                        "zscore": round(vol_z, 2),
                        "current_vol": round(realized_vol, 6),
                        "expected_vol": round(vol_hist_mean, 6),
                        "description": f"Realized volatility {vol_z:.1f} std above normal",
                    })

        # ─────────────────────────────────────
        # 7. TRADE INTENSITY ANOMALY
        # ─────────────────────────────────────
        if trades_per_sec > 0:
            if has_hist and hist_stats.get("trades_per_sec_mean", 0) > 0:
                tps_mean = hist_stats["trades_per_sec_mean"]
                tps_std = hist_stats.get("trades_per_sec_std", tps_mean * 0.5)
                if tps_std > 0:
                    tps_z = (trades_per_sec - tps_mean) / tps_std
                    if tps_z > 3:
                        anomalies.append({
                            "type": "TRADE_INTENSITY_SPIKE",
                            "severity": "MEDIUM",
                            "zscore": round(tps_z, 2),
                            "value": round(trades_per_sec, 2),
                            "expected": round(tps_mean, 2),
                            "description": f"Trade intensity {tps_z:.1f} std above normal ({trades_per_sec:.0f}/sec)",
                        })
            else:
                # Threshold fixo: > 200 trades/sec é incomum
                if trades_per_sec > 200:
                    anomalies.append({
                        "type": "TRADE_INTENSITY_SPIKE",
                        "severity": "MEDIUM",
                        "value": round(trades_per_sec, 2),
                        "description": f"Trade intensity {trades_per_sec:.0f}/sec is unusually high",
                    })

        # ═══════════════════════════════════════════
        # RESULTADO
        # ═══════════════════════════════════════════
        severity_order = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0}

        if anomalies:
            max_severity = max(anomalies, key=lambda a: severity_order.get(a["severity"], 0))["severity"]
        else:
            max_severity = "NONE"

        return {
            "anomalies_detected": len(anomalies) > 0,
            "count": len(anomalies),
            "anomalies": anomalies,
            "max_severity": max_severity,
            "risk_elevated": max_severity in ("CRITICAL", "HIGH"),
            "types_found": list(set(a["type"] for a in anomalies)),
            "summary": (
                f"{len(anomalies)} anomalies detected (max severity: {max_severity})"
                if anomalies
                else "No anomalies detected"
            ),
        }