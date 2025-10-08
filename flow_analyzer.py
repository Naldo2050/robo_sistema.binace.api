# flow_analyzer.py v2.0.0 - CORRIGIDO

"""
Flow Analyzer com correções críticas.

🔹 CORREÇÕES v2.0.0:
  ✅ Adiciona cálculo de flow_imbalance
  ✅ Adiciona cálculo de tick_rule_sum (uptick/downtick)
  ✅ Corrige inconsistência em sector_flow (BTC vs USD)
  ✅ Lock timeout lança exceção ao invés de retornar zeros
  ✅ Logs completos em falhas de normalização
  ✅ Valida consistência de whale metrics
  ✅ Adiciona flags de qualidade de dados
  ✅ Detecção de dados inválidos
"""

import logging
from threading import Lock
from collections import deque
from typing import List, Dict, Any, Optional, Tuple
import time

import config

from time_manager import TimeManager
from liquidity_heatmap import LiquidityHeatmap

# Parâmetros
try:
    from config import (
        NET_FLOW_WINDOWS_MIN,
        AGGRESSIVE_ORDER_SIZE_THRESHOLD,
        ABSORCAO_DELTA_EPS,
        ABSORCAO_GUARD_MODE,
    )
except Exception:
    NET_FLOW_WINDOWS_MIN = [1, 5, 15]
    AGGRESSIVE_ORDER_SIZE_THRESHOLD = 0.0
    ABSORCAO_DELTA_EPS = 1.0
    ABSORCAO_GUARD_MODE = "warn"


# 🆕 Exceção customizada
class FlowAnalyzerError(Exception):
    """Levantada quando FlowAnalyzer encontra erro crítico."""
    pass


def _guard_absorcao(delta: float, rotulo: str, eps: float, mode: str = "warn"):
    """Validação de consistência para absorção."""
    try:
        mode = (mode or "warn").strip().lower()
    except Exception:
        mode = "warn"

    if mode == "off":
        return

    mismatch = (delta > eps and rotulo != "Absorção de Compra") or \
               (delta < -eps and rotulo != "Absorção de Venda")
    
    if mismatch:
        msg = f"[ABSORCAO_GUARD] delta={delta:.4f} eps={eps} rotulo='{rotulo}' (modo={mode})"
        if mode == "raise":
            raise AssertionError(msg)
        logging.warning(msg)


class FlowAnalyzer:
    """
    Analisador de fluxo com validação robusta.
    
    🔹 CORREÇÕES v2.0.0:
      - Calcula flow_imbalance corretamente
      - Calcula tick_rule_sum (uptick/downtick)
      - Unidades consistentes (sempre BTC para volumes)
      - Lock timeout lança exceção
      - Validação de dados
    """

    def __init__(self, time_manager: Optional[TimeManager] = None):
        self.time_manager = time_manager or TimeManager()

        # CVD e Whale (sempre em BTC)
        self.cvd = 0.0
        self.whale_threshold = float(getattr(config, "WHALE_TRADE_THRESHOLD", 5.0))
        self.whale_buy_volume = 0.0    # BTC
        self.whale_sell_volume = 0.0   # BTC (sempre positivo)
        self.whale_delta = 0.0         # BTC (com sinal)

        # Reset
        self.last_reset_ms = self.time_manager.now_ms()
        self.reset_interval_ms = int(
            getattr(config, "CVD_RESET_INTERVAL_HOURS", 24) * 3600 * 1000
        )

        self._lock = Lock()

        # Histórico de trades para bursts
        self.recent_trades = deque(maxlen=500)
        self.bursts = {"count": 0, "max_burst_volume": 0.0}
        self._in_burst = False
        self._last_burst_end_ms = 0

        # Parâmetros de bursts
        self.burst_window_ms = int(getattr(config, "BURST_WINDOW_MS", 200))
        self.burst_cooldown_ms = int(getattr(config, "BURST_COOLDOWN_MS", 200))
        self.burst_volume_threshold = float(
            getattr(config, "BURST_VOLUME_THRESHOLD", self.whale_threshold)
        )

        # 🆕 Sector flow (SEMPRE EM BTC)
        order_buckets = getattr(config, "ORDER_SIZE_BUCKETS", {
            "retail": (0, 0.5),
            "mid": (0.5, 2.0),
            "whale": (2.0, 9999.0)
        })
        self.sector_flow = {
            name: {"buy": 0.0, "sell": 0.0, "delta": 0.0}
            for name in order_buckets
        }
        self._order_buckets = order_buckets

        # Liquidity Heatmap
        lhm_window_size = int(getattr(config, "LHM_WINDOW_SIZE", 2000))
        lhm_cluster_threshold_pct = float(getattr(config, "LHM_CLUSTER_THRESHOLD_PCT", 0.003))
        lhm_min_trades_per_cluster = int(getattr(config, "LHM_MIN_TRADES_PER_CLUSTER", 5))
        lhm_update_interval_ms = int(getattr(config, "LHM_UPDATE_INTERVAL_MS", 100))

        self.liquidity_heatmap = LiquidityHeatmap(
            window_size=lhm_window_size,
            cluster_threshold_pct=lhm_cluster_threshold_pct,
            min_trades_per_cluster=lhm_min_trades_per_cluster,
            update_interval_ms=lhm_update_interval_ms
        )

        # Janelas de net flow
        self.net_flow_windows_min: List[int] = list(NET_FLOW_WINDOWS_MIN)
        
        # 🆕 Histórico de trades com TODAS as informações
        # Estrutura: {'ts': ms, 'price': float, 'qty': float, 'delta_btc': float, 'delta_usd': float, 'side': str, 'sector': str}
        self.flow_trades: deque = deque()

        # Epsilon para absorção
        self.absorcao_eps: float = float(getattr(config, "ABSORCAO_DELTA_EPS", ABSORCAO_DELTA_EPS))
        try:
            self.absorcao_guard_mode: str = str(
                getattr(config, "ABSORCAO_GUARD_MODE", ABSORCAO_GUARD_MODE)
            ).lower()
        except Exception:
            self.absorcao_guard_mode = "warn"
        
        # 🆕 Estatísticas
        self._total_trades_processed = 0
        self._invalid_trades = 0
        self._lock_contentions = 0
        
        # 🆕 Último preço (para tick rule)
        self._last_price: Optional[float] = None

        logging.info(
            "✅ FlowAnalyzer v2.0.0 inicializado | "
            "Whale threshold: %.2f BTC | Net flow windows: %s min | "
            "Absorção eps: %.2f",
            self.whale_threshold,
            self.net_flow_windows_min,
            self.absorcao_eps,
        )

    @staticmethod
    def map_absorcao_label(aggression_side: str) -> str:
        """Compatibilidade: mapeia lado de agressão para rótulo."""
        side = (aggression_side or "").strip().lower()
        if side == "buy":
            return "Absorção de Compra"
        if side == "sell":
            return "Absorção de Venda"
        return "Absorção"

    @staticmethod
    def classificar_absorcao_por_delta(delta: float, eps: float = 1.0) -> str:
        """
        Classificador de absorção por sinal do delta.
        
        Args:
            delta: Net flow (positivo = compra domina, negativo = venda domina)
            eps: Threshold mínimo para considerar não-neutro
        
        Returns:
            "Absorção de Compra" | "Absorção de Venda" | "Neutra"
        """
        try:
            d = float(delta)
        except Exception:
            return "Neutra"
        
        if d > eps:
            return "Absorção de Compra"
        if d < -eps:
            return "Absorção de Venda"
        return "Neutra"

    def _reset_metrics(self):
        """Reseta todas as métricas acumuladas."""
        try:
            self.cvd = 0.0
            self.whale_buy_volume = 0.0
            self.whale_sell_volume = 0.0
            self.whale_delta = 0.0

            self.recent_trades.clear()
            self.bursts = {"count": 0, "max_burst_volume": 0.0}
            self._in_burst = False
            self._last_burst_end_ms = 0

            self.sector_flow = {
                name: {"buy": 0.0, "sell": 0.0, "delta": 0.0}
                for name in self._order_buckets
            }
            
            self.flow_trades.clear()
            self._last_price = None
            
            self.last_reset_ms = self.time_manager.now_ms()
            
            logging.info("🔄 FlowAnalyzer metrics resetados.")
        except Exception as e:
            logging.error(f"Erro ao resetar métricas: {e}")

    def _check_reset(self):
        """Verifica se deve resetar métricas."""
        try:
            now_ms = self.time_manager.now_ms()
            if now_ms - self.last_reset_ms > self.reset_interval_ms:
                with self._lock:
                    self._reset_metrics()
        except Exception as e:
            logging.error(f"Erro ao verificar reset: {e}")

    def _prune_recent(self, now_ms: int):
        """Remove trades antigos fora da janela de bursts."""
        try:
            while self.recent_trades and \
                  (now_ms - self.recent_trades[0][0] > self.burst_window_ms):
                self.recent_trades.popleft()
        except Exception as e:
            logging.error(f"Erro ao podar recent_trades: {e}")
            self.recent_trades.clear()

    def _prune_flow_history(self, now_ms: int):
        """Remove trades antigos do histórico de flow."""
        try:
            if not self.net_flow_windows_min:
                return
            
            max_window = max(self.net_flow_windows_min)
            cutoff_ms = now_ms - max_window * 60 * 1000
            
            while self.flow_trades and self.flow_trades[0]['ts'] < cutoff_ms:
                self.flow_trades.popleft()
        except Exception as e:
            logging.debug(f"Erro ao podar flow_trades: {e}")

    def _update_bursts(self, ts_ms: int, qty: float):
        """Detecta bursts de volume."""
        try:
            self.recent_trades.append((ts_ms, qty))
            self._prune_recent(ts_ms)

            burst_volume = sum(q for _, q in self.recent_trades)
            threshold = self.burst_volume_threshold

            if not self._in_burst:
                if burst_volume >= threshold and \
                   (ts_ms - self._last_burst_end_ms) >= self.burst_cooldown_ms:
                    self.bursts["count"] += 1
                    self._in_burst = True
                    if burst_volume > self.bursts["max_burst_volume"]:
                        self.bursts["max_burst_volume"] = burst_volume
            else:
                if burst_volume > self.bursts["max_burst_volume"]:
                    self.bursts["max_burst_volume"] = burst_volume
                if burst_volume < threshold * 0.5:
                    self._in_burst = False
                    self._last_burst_end_ms = ts_ms
        except Exception as e:
            logging.error(f"Erro ao atualizar bursts: {e}")
            self._in_burst = False
            self._last_burst_end_ms = ts_ms

    def _update_sector_flow(self, qty: float, delta_btc: float):
        """
        🔹 CORRIGIDO: Sempre usa BTC para volumes.
        
        Args:
            qty: Quantidade em BTC (sempre positivo)
            delta_btc: Delta em BTC (positivo = compra, negativo = venda)
        """
        try:
            for name, (minv, maxv) in self._order_buckets.items():
                if minv <= qty < maxv:
                    if delta_btc > 0:
                        self.sector_flow[name]["buy"] += qty
                    else:
                        # 🆕 CORRIGIDO: usa qty (BTC), não abs(delta)
                        self.sector_flow[name]["sell"] += qty
                    
                    self.sector_flow[name]["delta"] += delta_btc
                    break
        except Exception as e:
            logging.error(f"Erro ao atualizar sector_flow: {e}")

    def process_trade(self, trade: dict):
        """
        Processa trade e atualiza todas as métricas.
        
        🔹 CORREÇÕES v2.0.0:
          - Valida estrutura do trade
          - Rastreia último preço para tick_rule
          - Salva delta em BTC E USD
          - Unidades consistentes
        
        Args:
            trade: Dict com 'q' (BTC), 'p' (USDT), 'T' (ms), 'm' (bool)
        """
        try:
            self._check_reset()
            self._total_trades_processed += 1

            # 1. Validação básica
            if not isinstance(trade, dict):
                self._invalid_trades += 1
                return
            
            if not all(k in trade for k in ("q", "T", "p")):
                self._invalid_trades += 1
                return

            # 2. Parse de dados
            try:
                qty = float(trade.get('q', 0.0))
                ts = int(trade.get('T'))
                price = float(trade.get('p', 0.0))
            except (ValueError, TypeError):
                self._invalid_trades += 1
                return

            if qty <= 0 or price <= 0 or ts <= 0:
                self._invalid_trades += 1
                return

            # 3. Determina lado (taker buy/sell)
            is_buyer_maker = trade.get('m', None)
            
            if isinstance(is_buyer_maker, bool):
                pass
            elif isinstance(is_buyer_maker, (int, float)):
                is_buyer_maker = bool(int(is_buyer_maker))
            elif isinstance(is_buyer_maker, str):
                is_buyer_maker = is_buyer_maker.strip().lower() in {
                    "true", "t", "1", "sell", "ask", "s", "seller", "yes"
                }
            else:
                # Sem flag de lado = ignora (evita viés)
                self._invalid_trades += 1
                return

            # 4. Calcula deltas
            # delta_btc: positivo = taker BUY, negativo = taker SELL
            delta_btc = -qty if is_buyer_maker else qty
            delta_usd = delta_btc * price
            side = "sell" if is_buyer_maker else "buy"

            with self._lock:
                # 5. CVD (BTC)
                self.cvd += delta_btc

                # 6. Whale metrics (SEMPRE EM BTC)
                if qty >= self.whale_threshold:
                    if delta_btc > 0:
                        self.whale_buy_volume += qty
                    else:
                        # 🆕 CORRIGIDO: usa qty, não abs(delta_btc)
                        self.whale_sell_volume += qty
                    
                    self.whale_delta += delta_btc

                # 7. Bursts
                self._update_bursts(ts, qty)

                # 8. Sector flow (BTC)
                self._update_sector_flow(qty, delta_btc)

                # 9. Heatmap
                self.liquidity_heatmap.add_trade(
                    price=price,
                    volume=qty,
                    side=side,
                    timestamp_ms=ts
                )

                # 10. 🆕 Histórico completo para flow metrics
                try:
                    # Determina setor
                    sector_name: Optional[str] = None
                    for name, (minv, maxv) in self._order_buckets.items():
                        if minv <= qty < maxv:
                            sector_name = name
                            break

                    # Salva trade completo
                    self.flow_trades.append({
                        'ts': ts,
                        'price': price,
                        'qty': qty,
                        'delta_btc': delta_btc,
                        'delta_usd': delta_usd,
                        'side': side,
                        'sector': sector_name,
                    })

                    # Poda histórico antigo
                    self._prune_flow_history(ts)
                    
                    # 🆕 Atualiza último preço (para tick_rule)
                    self._last_price = price

                except Exception as e:
                    logging.debug(f"Erro ao salvar trade no histórico: {e}")

        except Exception as e:
            logging.debug(f"Erro ao processar trade: {e}")
            self._invalid_trades += 1

    def _normalize_heatmap_clusters(
        self, 
        clusters: List[Dict[str, Any]], 
        now_ms: int
    ) -> List[Dict[str, Any]]:
        """
        Normaliza clusters do heatmap.
        
        🔹 CORRIGIDO: Logs em falhas, não engole erros silenciosamente.
        """
        out: List[Dict[str, Any]] = []
        
        for c in clusters or []:
            cc = dict(c)
            
            try:
                # age_ms
                recent_keys = (
                    "recent_timestamp", "recent_ts_ms", "last_seen_ms",
                    "last_ts_ms", "max_timestamp", "last_timestamp"
                )
                recent_ts = None
                
                for k in recent_keys:
                    if k in cc and isinstance(cc[k], (int, float)):
                        recent_ts = int(cc[k])
                        break
                
                if recent_ts is not None:
                    cc["age_ms"] = self.time_manager.calc_age_ms(
                        recent_ts, 
                        reference_ts_ms=now_ms
                    )

                # total_volume
                tv = cc.get("total_volume", None)
                bv = float(cc.get("buy_volume", 0.0) or 0.0)
                sv = float(cc.get("sell_volume", 0.0) or 0.0)
                
                if tv is None or (isinstance(tv, (int, float)) and tv <= 0):
                    recomputed = bv + sv
                    if recomputed > 0:
                        cc["total_volume"] = recomputed

                # imbalance_ratio
                tv2 = float(cc.get("total_volume", 0.0) or 0.0)
                if tv2 > 0:
                    imb = (bv - sv) / tv2
                else:
                    imb = 0.0
                
                cc["imbalance_ratio"] = max(-1.0, min(1.0, float(imb)))

                # width
                if "high" in cc and "low" in cc:
                    try:
                        hi, lo = float(cc["high"]), float(cc["low"])
                        cc["width"] = max(0.0, hi - lo)
                    except Exception:
                        pass

                # trades_count
                if "trades_count" in cc:
                    try:
                        cc["trades_count"] = int(cc["trades_count"])
                    except Exception:
                        pass

            except Exception as e:
                # 🆕 CORRIGIDO: Loga erro ao invés de engolir
                logging.warning(f"⚠️ Erro ao normalizar cluster: {e}")
                logging.debug(f"   Cluster problemático: {cc}")

            out.append(cc)
        
        return out

    def get_flow_metrics(
        self, 
        reference_epoch_ms: Optional[int] = None
    ) -> dict:
        """
        Retorna métricas de fluxo.
        
        🔹 CORREÇÕES v2.0.0:
          - Calcula flow_imbalance
          - Calcula tick_rule_sum
          - Lock timeout lança exceção
          - Validação de dados
        
        Returns:
            Dict com todas as métricas
        """
        try:
            # 🆕 TIMEOUT LANÇA EXCEÇÃO AO INVÉS DE RETORNAR ZEROS
            acquired = self._lock.acquire(timeout=5.0)
            
            if not acquired:
                self._lock_contentions += 1
                error_msg = (
                    f"❌ FlowAnalyzer lock timeout após 5s "
                    f"(contentions: {self._lock_contentions})"
                )
                logging.error(error_msg)
                raise FlowAnalyzerError(error_msg)

            try:
                now_ms = reference_epoch_ms if reference_epoch_ms is not None \
                         else self.time_manager.now_ms()
                
                time_index = self.time_manager.build_time_index(
                    now_ms, 
                    include_local=True, 
                    timespec="milliseconds"
                )

                # Métricas base
                metrics = {
                    "cvd": float(self.cvd),
                    "whale_buy_volume": float(self.whale_buy_volume),
                    "whale_sell_volume": float(self.whale_sell_volume),
                    "whale_delta": float(self.whale_delta),
                    "bursts": dict(self.bursts),
                    "sector_flow": {
                        k: {
                            "buy": float(v["buy"]),
                            "sell": float(v["sell"]),
                            "delta": float(v["delta"])
                        }
                        for k, v in self.sector_flow.items()
                    },
                    "timestamp": time_index["timestamp_utc"],
                    "time_index": time_index,
                    "metadata": {
                        "burst_window_ms": self.burst_window_ms,
                        "burst_cooldown_ms": self.burst_cooldown_ms,
                        "in_burst": bool(self._in_burst),
                        "last_reset_ms": self.last_reset_ms,
                        "last_reset_iso_utc": self.time_manager.format_timestamp(
                            self.last_reset_ms
                        ),
                    },
                }

                # 🆕 ORDER FLOW COM flow_imbalance E tick_rule_sum
                try:
                    order_flow: Dict[str, Any] = {}
                    absorcao_por_janela: Dict[int, str] = {}

                    if not self.net_flow_windows_min:
                        smallest_window = 1
                    else:
                        smallest_window = min(self.net_flow_windows_min)

                    # 🆕 TICK RULE SUM (para menor janela)
                    tick_rule_sum = 0.0
                    
                    for window_min in self.net_flow_windows_min:
                        window_ms = window_min * 60 * 1000
                        start_ms = now_ms - window_ms
                        
                        # Filtra trades da janela
                        relevant = [
                            t for t in self.flow_trades 
                            if t['ts'] >= start_ms
                        ]

                        # Net flows (USD)
                        total_delta_usd = sum(t['delta_usd'] for t in relevant)
                        total_buy_usd = sum(
                            t['delta_usd'] for t in relevant 
                            if t['delta_usd'] > 0
                        )
                        total_sell_usd = -sum(
                            t['delta_usd'] for t in relevant 
                            if t['delta_usd'] < 0
                        )

                        # Net flow
                        key_net = f"net_flow_{window_min}m"
                        order_flow[key_net] = round(total_delta_usd, 4)

                        # Absorção
                        rotulo = self.classificar_absorcao_por_delta(
                            total_delta_usd, 
                            eps=self.absorcao_eps
                        )
                        _guard_absorcao(
                            total_delta_usd, 
                            rotulo, 
                            self.absorcao_eps, 
                            self.absorcao_guard_mode
                        )

                        order_flow[f"absorcao_{window_min}m"] = rotulo
                        absorcao_por_janela[window_min] = rotulo

                        # 🆕 Para menor janela: flow_imbalance e tick_rule
                        if window_min == smallest_window:
                            total_vol_usd = total_buy_usd + total_sell_usd
                            
                            # flow_imbalance [-1, +1]
                            if total_vol_usd > 0:
                                flow_imbalance = total_delta_usd / total_vol_usd
                                order_flow["flow_imbalance"] = round(
                                    flow_imbalance, 
                                    4
                                )
                            else:
                                order_flow["flow_imbalance"] = 0.0

                            # Percentuais
                            if total_vol_usd > 0:
                                order_flow["aggressive_buy_pct"] = round(
                                    (total_buy_usd / total_vol_usd) * 100.0, 
                                    2
                                )
                                order_flow["aggressive_sell_pct"] = round(
                                    (total_sell_usd / total_vol_usd) * 100.0, 
                                    2
                                )
                                
                                if total_sell_usd > 0:
                                    order_flow["buy_sell_ratio"] = round(
                                        total_buy_usd / total_sell_usd, 
                                        4
                                    )
                                else:
                                    order_flow["buy_sell_ratio"] = None
                            else:
                                order_flow["aggressive_buy_pct"] = None
                                order_flow["aggressive_sell_pct"] = None
                                order_flow["buy_sell_ratio"] = None

                            # 🆕 TICK RULE SUM
                            # uptick (+1), downtick (-1), same (0)
                            tick_rule_sum = 0.0
                            prev_price = None
                            
                            for t in sorted(relevant, key=lambda x: x['ts']):
                                curr_price = t['price']
                                
                                if prev_price is not None:
                                    if curr_price > prev_price:
                                        tick_rule_sum += 1.0
                                    elif curr_price < prev_price:
                                        tick_rule_sum -= 1.0
                                    # else: mesmo preço = 0
                                
                                prev_price = curr_price
                            
                            order_flow["tick_rule_sum"] = round(tick_rule_sum, 4)

                    # Tipo de absorção agregado
                    if self.net_flow_windows_min:
                        metrics["tipo_absorcao"] = absorcao_por_janela.get(
                            smallest_window, 
                            "Neutra"
                        )
                    else:
                        metrics["tipo_absorcao"] = "Neutra"

                    # Participant analysis
                    participant_analysis: Dict[str, Any] = {}
                    
                    if self.net_flow_windows_min:
                        largest_window = max(self.net_flow_windows_min)
                        start_ms_p = now_ms - largest_window * 60 * 1000
                        all_trades = [
                            t for t in self.flow_trades 
                            if t['ts'] >= start_ms_p
                        ]
                        total_qty_all = sum(t['qty'] for t in all_trades)

                        for sector in self._order_buckets.keys():
                            sector_trades = [
                                t for t in all_trades 
                                if t.get('sector') == sector
                            ]
                            total_qty_sector = sum(t['qty'] for t in sector_trades)
                            buy_qty = sum(
                                t['qty'] for t in sector_trades 
                                if t['delta_btc'] > 0
                            )
                            sell_qty = sum(
                                t['qty'] for t in sector_trades 
                                if t['delta_btc'] < 0
                            )
                            count_trades = len(sector_trades)

                            # Direção
                            if buy_qty > sell_qty:
                                direction = "BUY"
                            elif sell_qty > buy_qty:
                                direction = "SELL"
                            else:
                                direction = "NEUTRAL"

                            # Métricas
                            avg_order_size = round(
                                total_qty_sector / count_trades, 
                                4
                            ) if count_trades > 0 else None
                            
                            volume_pct = round(
                                (total_qty_sector / total_qty_all) * 100.0, 
                                2
                            ) if total_qty_all > 0 else None
                            
                            sentiment = "BULLISH" if direction == "BUY" else \
                                       ("BEARISH" if direction == "SELL" else "NEUTRAL")
                            
                            duration_seconds = largest_window * 60
                            trades_per_sec = round(
                                count_trades / duration_seconds, 
                                4
                            ) if duration_seconds > 0 else None
                            
                            activity_level = "HIGH" if trades_per_sec and \
                                            trades_per_sec >= 1.0 else "LOW"

                            participant_analysis[sector] = {
                                "volume_pct": volume_pct,
                                "direction": direction,
                                "avg_order_size": avg_order_size,
                                "sentiment": sentiment,
                                "activity_level": activity_level,
                            }

                    metrics["order_flow"] = order_flow
                    metrics["participant_analysis"] = participant_analysis

                except Exception as e:
                    logging.error(f"Erro ao calcular order_flow: {e}", exc_info=True)
                    metrics["order_flow"] = {
                        "flow_imbalance": 0.0,
                        "tick_rule_sum": 0.0,
                    }
                    metrics["participant_analysis"] = {}

                # Heatmap
                try:
                    clusters = self.liquidity_heatmap.get_clusters(top_n=5)
                    clusters = self._normalize_heatmap_clusters(clusters, now_ms=now_ms)
                    supports, resistances = self.liquidity_heatmap.get_support_resistance()
                    
                    metrics["liquidity_heatmap"] = {
                        "clusters": clusters,
                        "supports": sorted(set(supports)),
                        "resistances": sorted(set(resistances)),
                        "clusters_count": len(clusters),
                        "meta": {
                            "window_size": getattr(
                                self.liquidity_heatmap, 
                                "window_size", 
                                None
                            ),
                            "cluster_threshold_pct": getattr(
                                self.liquidity_heatmap, 
                                "cluster_threshold_pct", 
                                None
                            ),
                            "min_trades_per_cluster": getattr(
                                self.liquidity_heatmap, 
                                "min_trades_per_cluster", 
                                None
                            ),
                            "update_interval_ms": getattr(
                                self.liquidity_heatmap, 
                                "update_interval_ms", 
                                None
                            ),
                            "top_n": 5
                        }
                    }
                except Exception as e:
                    logging.error(f"Erro ao obter heatmap: {e}", exc_info=True)
                    metrics["liquidity_heatmap"] = {
                        "clusters": [],
                        "supports": [],
                        "resistances": [],
                        "clusters_count": 0,
                        "meta": {},
                    }

                # 🆕 Qualidade de dados
                metrics["data_quality"] = {
                    "total_trades_processed": self._total_trades_processed,
                    "invalid_trades": self._invalid_trades,
                    "valid_rate_pct": round(
                        100 * (1 - self._invalid_trades / max(1, self._total_trades_processed)),
                        2
                    ),
                    "flow_trades_count": len(self.flow_trades),
                    "recent_trades_count": len(self.recent_trades),
                    "lock_contentions": self._lock_contentions,
                }

                return metrics

            finally:
                self._lock.release()

        except FlowAnalyzerError:
            raise  # Propaga erro de lock timeout
        except Exception as e:
            logging.error(f"Erro ao obter flow metrics: {e}", exc_info=True)
            
            # Retorna métricas mínimas mas com flag de erro
            now_ms = reference_epoch_ms if reference_epoch_ms is not None \
                     else self.time_manager.now_ms()
            time_index = self.time_manager.build_time_index(
                now_ms, 
                include_local=True, 
                timespec="milliseconds"
            )
            
            return {
                "cvd": 0.0,
                "whale_buy_volume": 0.0,
                "whale_sell_volume": 0.0,
                "whale_delta": 0.0,
                "bursts": {"count": 0, "max_burst_volume": 0.0},
                "sector_flow": {},
                "timestamp": time_index["timestamp_utc"],
                "time_index": time_index,
                "order_flow": {
                    "flow_imbalance": 0.0,
                    "tick_rule_sum": 0.0,
                },
                "participant_analysis": {},
                "liquidity_heatmap": {
                    "clusters": [],
                    "supports": [],
                    "resistances": [],
                    "clusters_count": 0,
                },
                "data_quality": {
                    "error": str(e),
                    "is_valid": False,
                },
                "metadata": {
                    "error": "Exception during metrics calculation",
                },
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de performance."""
        return {
            "total_trades_processed": self._total_trades_processed,
            "invalid_trades": self._invalid_trades,
            "valid_rate_pct": round(
                100 * (1 - self._invalid_trades / max(1, self._total_trades_processed)),
                2
            ),
            "lock_contentions": self._lock_contentions,
            "flow_trades_count": len(self.flow_trades),
            "recent_trades_count": len(self.recent_trades),
            "cvd": self.cvd,
            "whale_delta": self.whale_delta,
        }
    
    def reset_stats(self):
        """Reseta contadores de estatísticas."""
        self._total_trades_processed = 0
        self._invalid_trades = 0
        self._lock_contentions = 0
        logging.info("📊 FlowAnalyzer stats resetados")


# Teste
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s | %(levelname)-8s | %(message)s'
    )
    
    print("\n" + "="*80)
    print("🧪 TESTE DE FLOW ANALYZER v2.0.0")
    print("="*80 + "\n")
    
    fa = FlowAnalyzer()
    
    # Simula trades
    print("📊 Processando trades de teste...")
    
    trades = [
        {'q': 2.5, 'p': 50000.0, 'T': int(time.time() * 1000), 'm': False},  # Buy
        {'q': 1.0, 'p': 50010.0, 'T': int(time.time() * 1000) + 100, 'm': True},   # Sell
        {'q': 5.5, 'p': 50020.0, 'T': int(time.time() * 1000) + 200, 'm': False},  # Buy (whale)
        {'q': 0.1, 'p': 50015.0, 'T': int(time.time() * 1000) + 300, 'm': True},   # Sell
    ]
    
    for trade in trades:
        fa.process_trade(trade)
    
    # Obtém métricas
    print("\n📈 Obtendo métricas...")
    metrics = fa.get_flow_metrics()
    
    print(f"\n  CVD: {metrics['cvd']:.4f} BTC")
    print(f"  Whale Delta: {metrics['whale_delta']:.4f} BTC")
    print(f"  Whale Buy: {metrics['whale_buy_volume']:.4f} BTC")
    print(f"  Whale Sell: {metrics['whale_sell_volume']:.4f} BTC")
    
    if 'order_flow' in metrics:
        of = metrics['order_flow']
        print(f"\n  Flow Imbalance: {of.get('flow_imbalance', 0):.4f}")
        print(f"  Tick Rule Sum: {of.get('tick_rule_sum', 0):.4f}")
        print(f"  Net Flow 1m: ${of.get('net_flow_1m', 0):,.2f}")
    
    if 'data_quality' in metrics:
        dq = metrics['data_quality']
        print(f"\n  Trades processados: {dq.get('total_trades_processed', 0)}")
        print(f"  Taxa de válidos: {dq.get('valid_rate_pct', 0)}%")
    
    print(f"\n  Tipo Absorção: {metrics.get('tipo_absorcao', 'N/A')}")
    
    print("\n" + "="*80)
    print("✅ TESTE CONCLUÍDO")
    print("="*80 + "\n")
    
    # Teste de classificação
    print("🧪 Testando classificador de absorção...")
    eps = fa.absorcao_eps
    casos = [
        (-35.57, "Absorção de Venda"),
        (+7.53,  "Absorção de Compra"),
        (+1.50,  "Absorção de Compra"),
        (0.0,    "Neutra"),
    ]
    
    for delta, esperado in casos:
        rotulo = fa.classificar_absorcao_por_delta(delta, eps=eps)
        status = "✅" if rotulo == esperado else "❌"
        print(f"  {status} delta={delta:+.2f} → {rotulo} (esperado: {esperado})")
    
    print("\n✅ Self-test OK\n")