# flow_analyzer.py

import logging
from threading import Lock
from collections import deque
from typing import List, Dict, Any, Optional

import config

# Time Manager (injeção recomendada para usar uma única instância no app)
from time_manager import TimeManager

# Liquidity Heatmap
from liquidity_heatmap import LiquidityHeatmap

# Novos parâmetros para net flow e análise de participantes
try:
    from config import NET_FLOW_WINDOWS_MIN, AGGRESSIVE_ORDER_SIZE_THRESHOLD
except Exception:
    NET_FLOW_WINDOWS_MIN = [1, 5, 15]
    AGGRESSIVE_ORDER_SIZE_THRESHOLD = 0.0


class FlowAnalyzer:
    """
    Analisador contínuo de fluxo (CVD, Whale Flow, bursts, buckets e liquidity heatmap).

    Atualizações principais:
    - Aceita time_manager injetado (para unificar relógio em todo o app).
    - Todas as referências de tempo em milissegundos (ms).
    - get_flow_metrics aceita reference_epoch_ms para normalizar age_ms dos clusters.
    - Parâmetros (bursts/heatmap) podem vir via config.
    """

    def __init__(self, time_manager: Optional[TimeManager] = None):
        # TimeManager compartilhado
        self.time_manager = time_manager or TimeManager()

        # Métricas de CVD (BTC)
        self.cvd = 0.0

        # Whale Flow
        self.whale_threshold = float(getattr(config, "WHALE_TRADE_THRESHOLD", 5.0))
        self.whale_buy_volume = 0.0    # BTC
        self.whale_sell_volume = 0.0   # BTC
        self.whale_delta = 0.0         # BTC

        # Reset (milissegundos)
        self.last_reset_ms = self.time_manager.now_ms()
        self.reset_interval_ms = int(getattr(config, "CVD_RESET_INTERVAL_HOURS", 24) * 3600 * 1000)

        self._lock = Lock()
        logging.info("✅ Analisador de Fluxo inicializado (CVD, Whale Flow, Bursts, Buckets, Heatmap).")

        # Histórico de trades (para bursts)
        self.recent_trades = deque(maxlen=500)
        self.bursts = {"count": 0, "max_burst_volume": 0.0}
        self._in_burst = False
        self._last_burst_end_ms = 0

        # Parâmetros de bursts via config
        self.burst_window_ms = int(getattr(config, "BURST_WINDOW_MS", 200))
        self.burst_cooldown_ms = int(getattr(config, "BURST_COOLDOWN_MS", 200))
        self.burst_volume_threshold = float(getattr(config, "BURST_VOLUME_THRESHOLD", self.whale_threshold))

        # Segmentação por tamanho de ordem
        order_buckets = getattr(config, "ORDER_SIZE_BUCKETS", {
            "retail": (0, 0.5),
            "mid": (0.5, 2.0),
            "whale": (2.0, 9999.0)
        })
        self.sector_flow = {name: {"buy": 0.0, "sell": 0.0, "delta": 0.0} for name in order_buckets}
        self._order_buckets = order_buckets

        # Liquidity Heatmap (parametrizável)
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

        # Janelas para cálculo de net flows (em minutos) e histórico de trades para métricas temporais
        self.net_flow_windows_min: List[int] = list(NET_FLOW_WINDOWS_MIN)
        # Deque armazenando trades recentes para cálculo de flows e participação por segmento
        # Cada item: dict {'ts': timestamp_ms, 'delta': notional_delta (USD), 'qty': qty (BTC), 'side': 'buy'/'sell', 'sector': nome do bucket}
        self.flow_trades: deque = deque()

    @staticmethod
    def map_absorcao_label(aggression_side: str) -> str:
        """
        - 'buy'  → Agressão compradora absorvida → "Absorção de Compra"
        - 'sell' → Agressão vendedora absorvida → "Absorção de Venda"
        """
        side = (aggression_side or "").strip().lower()
        if side == "buy":
            return "Absorção de Compra"
        if side == "sell":
            return "Absorção de Venda"
        return "Absorção"

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

            self.sector_flow = {name: {"buy": 0.0, "sell": 0.0, "delta": 0.0} for name in self._order_buckets}
            self.last_reset_ms = self.time_manager.now_ms()
            logging.info("🔄 Métricas de Fluxo resetadas (CVD, Whale Flow, Bursts, Buckets).")
        except Exception as e:
            logging.error(f"Erro ao resetar métricas: {e}")

    def _check_reset(self):
        """Verifica se as métricas devem ser resetadas com base no tempo."""
        try:
            now_ms = self.time_manager.now_ms()
            if now_ms - self.last_reset_ms > self.reset_interval_ms:
                with self._lock:
                    self._reset_metrics()
        except Exception as e:
            logging.error(f"Erro ao verificar reset de métricas: {e}")

    def _prune_recent(self, now_ms: int):
        """Remove trades antigos fora da janela de bursts."""
        try:
            while self.recent_trades and (now_ms - self.recent_trades[0][0] > self.burst_window_ms):
                self.recent_trades.popleft()
        except Exception as e:
            logging.error(f"Erro ao remover trades antigos: {e}")
            self.recent_trades.clear()

    def _prune_flow_history(self, now_ms: int):
        """
        Remove entradas antigas do histórico de fluxo (flow_trades) com base na maior
        janela configurada em net_flow_windows_min.
        """
        try:
            if not self.net_flow_windows_min:
                return
            max_window = max(self.net_flow_windows_min)
            cutoff_ms = now_ms - max_window * 60 * 1000
            while self.flow_trades and self.flow_trades[0]['ts'] < cutoff_ms:
                self.flow_trades.popleft()
        except Exception as e:
            logging.debug(f"Erro ao podar histórico de fluxo: {e}")

    def _update_bursts(self, ts_ms: int, qty: float):
        """Detecta bursts em microtempo com cooldown."""
        try:
            self.recent_trades.append((ts_ms, qty))
            self._prune_recent(ts_ms)

            burst_volume = sum(q for _, q in self.recent_trades)
            threshold = self.burst_volume_threshold

            if not self._in_burst:
                if burst_volume >= threshold and (ts_ms - self._last_burst_end_ms) >= self.burst_cooldown_ms:
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

    def _update_sector_flow(self, qty: float, trade_delta: float):
        """Classifica trade em buckets (retail/mid/whale) e acumula fluxo."""
        try:
            for name, (minv, maxv) in self._order_buckets.items():
                if minv <= qty < maxv:
                    if trade_delta > 0:
                        self.sector_flow[name]["buy"] += qty
                    else:
                        self.sector_flow[name]["sell"] += abs(trade_delta)
                    self.sector_flow[name]["delta"] += trade_delta
                    break
        except Exception as e:
            logging.error(f"Erro ao atualizar sector flow: {e}")

    def process_trade(self, trade: dict):
        """
        Processa um trade para atualizar CVD/Whale/Bursts/Heatmap.
        - Requer q (BTC), p (USDT), T (ms).
        - Campo m (bool): True = taker SELL, False = taker BUY.
          Se 'm' estiver ausente/ruim, o trade é ignorado (evita viés no CVD).
        """
        try:
            self._check_reset()

            if not isinstance(trade, dict):
                return
            if not all(k in trade for k in ("q", "T", "p")):
                return

            try:
                qty = float(trade.get('q', 0.0))
                ts = int(trade.get('T'))
                price = float(trade.get('p', 0.0))
            except Exception:
                return

            if qty <= 0 or price <= 0 or ts <= 0:
                return

            is_buyer_maker = trade.get('m', None)
            if isinstance(is_buyer_maker, bool):
                pass
            elif isinstance(is_buyer_maker, (int, float)):
                is_buyer_maker = bool(int(is_buyer_maker))
            elif isinstance(is_buyer_maker, str):
                is_buyer_maker = is_buyer_maker.strip().lower() in {"true", "t", "1", "sell", "ask", "s", "seller", "yes"}
            else:
                return  # ignora para não enviesar

            trade_delta = -qty if is_buyer_maker else qty
            side = "sell" if is_buyer_maker else "buy"

            with self._lock:
                # CVD e Whale
                self.cvd += trade_delta
                if qty >= self.whale_threshold:
                    if trade_delta > 0:
                        self.whale_buy_volume += qty
                    else:
                        self.whale_sell_volume += abs(trade_delta)
                    self.whale_delta += trade_delta

                # Bursts e buckets
                self._update_bursts(ts, qty)
                self._update_sector_flow(qty, trade_delta)

                # Heatmap (usa a assinatura do teu LiquidityHeatmap)
                self.liquidity_heatmap.add_trade(
                    price=price,
                    volume=qty,
                    side=side,
                    timestamp_ms=ts
                )

                # ---- Histórico de fluxo para net flows e participante ----
                try:
                    # Determina o bucket/sector do trade
                    sector_name: Optional[str] = None
                    for name, (minv, maxv) in self._order_buckets.items():
                        if minv <= qty < maxv:
                            sector_name = name
                            break
                    # Delta notional em USD (trade_delta * price). trade_delta já contém o sinal (buy positive, sell negative)
                    delta_notional = trade_delta * price
                    # Armazena no deque
                    self.flow_trades.append({
                        'ts': ts,
                        'delta': float(delta_notional),
                        'qty': float(qty),
                        'side': side,
                        'sector': sector_name,
                    })
                    # Remove entradas antigas (maior janela)
                    self._prune_flow_history(ts)
                except Exception as e:
                    logging.debug(f"Erro ao registrar fluxo: {e}")

        except Exception as e:
            logging.debug(f"Erro ao processar trade (ignorado): {e}")

    # ----------------------------- CORREÇÃO AQUI -----------------------------
    def _normalize_heatmap_clusters(self, clusters: List[Dict[str, Any]], now_ms: int) -> List[Dict[str, Any]]:
        """
        Normaliza clusters para o payload final:
        - Recalcula age_ms com base em now_ms usando a MELHOR chave disponível:
          recent_timestamp | recent_ts_ms | last_seen_ms | last_ts_ms | max_timestamp | last_timestamp
        - Recalcula total_volume se vier 0/ausente usando buy_volume + sell_volume.
        - Recalcula/clampa imbalance_ratio em [-1, 1] com proteção a divisão por zero.
        - Garante width = max(0, high - low) e trades_count inteiro.
        """
        out: List[Dict[str, Any]] = []
        for c in clusters or []:
            cc = dict(c)
            try:
                # --- age_ms (recência) ---
                recent_keys = (
                    "recent_timestamp", "recent_ts_ms", "last_seen_ms",
                    "last_ts_ms", "max_timestamp", "last_timestamp"
                )
                recent_ts = None
                for k in recent_keys:
                    if k in cc and isinstance(cc[k], (int, float)):
                        recent_ts = int(cc[k])
                        break
                if recent_ts is None and isinstance(cc.get("age_ms"), (int, float)):
                    # já veio pronto (mantém)
                    pass
                else:
                    # calcula a idade em ms a partir do timestamp encontrado
                    if recent_ts is not None:
                        cc["age_ms"] = self.time_manager.calc_age_ms(recent_ts, reference_ts_ms=now_ms)

                # --- total_volume ---
                tv = cc.get("total_volume", None)
                bv = float(cc.get("buy_volume", 0.0) or 0.0)
                sv = float(cc.get("sell_volume", 0.0) or 0.0)
                if tv is None or (isinstance(tv, (int, float)) and tv <= 0):
                    recomputed = bv + sv
                    # só atualiza se houver algo para somar
                    if recomputed > 0:
                        cc["total_volume"] = recomputed

                # --- imbalance_ratio ---
                tv2 = float(cc.get("total_volume", 0.0) or 0.0)
                if tv2 > 0:
                    imb = (bv - sv) / tv2
                else:
                    imb = 0.0
                try:
                    # se veio do gerador, priorizamos o recalculado (consistente com total_volume corrigido)
                    cc["imbalance_ratio"] = max(-1.0, min(1.0, float(imb)))
                except Exception:
                    cc["imbalance_ratio"] = 0.0

                # --- width ---
                if "high" in cc and "low" in cc:
                    try:
                        hi, lo = float(cc["high"]), float(cc["low"])
                        cc["width"] = max(0.0, hi - lo)
                    except Exception:
                        pass

                # --- trades_count coerente ---
                if "trades_count" in cc:
                    try:
                        cc["trades_count"] = int(cc["trades_count"])
                    except Exception:
                        pass

            except Exception:
                # em caso de falha, devolve como veio
                pass

            out.append(cc)
        return out
    # ------------------------------------------------------------------------

    def get_flow_metrics(self, reference_epoch_ms: Optional[int] = None) -> dict:
        """
        Retorna as métricas de fluxo atuais.
        - reference_epoch_ms: se fornecido, normaliza age_ms dos clusters com base nesse epoch
          e constrói time_index derivado desse epoch (consistência com eventos).
        """
        try:
            acquired = self._lock.acquire(timeout=5.0)
            if not acquired:
                logging.warning("⚠️ Lock do FlowAnalyzer ocupado. Retornando métricas mínimas.")
                now_ms = reference_epoch_ms if reference_epoch_ms is not None else self.time_manager.now_ms()
                time_index = self.time_manager.build_time_index(now_ms, include_local=True, timespec="milliseconds")
                return {
                    "cvd": 0.0,
                    "whale_buy_volume": 0.0,
                    "whale_sell_volume": 0.0,
                    "whale_delta": 0.0,
                    "bursts": {"count": 0, "max_burst_volume": 0.0},
                    "sector_flow": {name: {"buy": 0.0, "sell": 0.0, "delta": 0.0} for name in self._order_buckets},
                    "timestamp": time_index["timestamp_utc"],
                    "time_index": time_index,
                    "liquidity_heatmap": {
                        "clusters": [],
                        "supports": [],
                        "resistances": [],
                        "clusters_count": 0,
                        "meta": {
                            "window_size": getattr(self.liquidity_heatmap, "window_size", None),
                            "cluster_threshold_pct": getattr(self.liquidity_heatmap, "cluster_threshold_pct", None),
                            "min_trades_per_cluster": getattr(self.liquidity_heatmap, "min_trades_per_cluster", None),
                            "update_interval_ms": getattr(self.liquidity_heatmap, "update_interval_ms", None),
                            "top_n": 5
                        }
                    },
                    "metadata": {
                        "burst_window_ms": self.burst_window_ms,
                        "burst_cooldown_ms": self.burst_cooldown_ms,
                        "in_burst": False,
                        "last_reset_ms": self.last_reset_ms,
                        "last_reset_iso_utc": self.time_manager.format_timestamp(self.last_reset_ms)
                    }
                }

            try:
                now_ms = reference_epoch_ms if reference_epoch_ms is not None else self.time_manager.now_ms()
                time_index = self.time_manager.build_time_index(now_ms, include_local=True, timespec="milliseconds")

                metrics = {
                    "cvd": float(self.cvd),
                    "whale_buy_volume": float(self.whale_buy_volume),
                    "whale_sell_volume": float(self.whale_sell_volume),
                    "whale_delta": float(self.whale_delta),
                    "bursts": dict(self.bursts),
                    "sector_flow": {k: {"buy": float(v["buy"]), "sell": float(v["sell"]), "delta": float(v["delta"])}
                                    for k, v in self.sector_flow.items()},
                    "timestamp": time_index["timestamp_utc"],
                    "time_index": time_index,
                    "metadata": {
                        "burst_window_ms": self.burst_window_ms,
                        "burst_cooldown_ms": self.burst_cooldown_ms,
                        "in_burst": bool(self._in_burst),
                        "last_reset_ms": self.last_reset_ms,
                        "last_reset_iso_utc": self.time_manager.format_timestamp(self.last_reset_ms)
                    }
                }

                # ----------------- Order Flow e Participantes -----------------
                try:
                    order_flow: Dict[str, Any] = {}
                    # Net flows para cada janela configurada
                    for window_min in self.net_flow_windows_min:
                        window_ms = window_min * 60 * 1000
                        start_ms = now_ms - window_ms
                        # Filtra trades da janela
                        relevant = [t for t in self.flow_trades if t['ts'] >= start_ms]
                        total_delta = sum(t['delta'] for t in relevant)
                        total_buy_notional = sum(t['delta'] for t in relevant if t['delta'] > 0)
                        total_sell_notional = -sum(t['delta'] for t in relevant if t['delta'] < 0)
                        order_flow[f"net_flow_{window_min}m"] = round(total_delta, 4)
                        # Para a menor janela, calcula percentuais de compra/venda e razão
                        if window_min == min(self.net_flow_windows_min):
                            total_vol = total_buy_notional + total_sell_notional
                            if total_vol > 0:
                                order_flow["aggressive_buy_pct"] = round((total_buy_notional / total_vol) * 100.0, 2)
                                order_flow["aggressive_sell_pct"] = round((total_sell_notional / total_vol) * 100.0, 2)
                                order_flow["buy_sell_ratio"] = round((total_buy_notional / total_sell_notional), 4) if total_sell_notional > 0 else None
                            else:
                                order_flow["aggressive_buy_pct"] = None
                                order_flow["aggressive_sell_pct"] = None
                                order_flow["buy_sell_ratio"] = None
                    # ----------------- Participantes -----------------
                    participant_analysis: Dict[str, Any] = {}
                    if self.net_flow_windows_min:
                        largest_window = max(self.net_flow_windows_min)
                        start_ms_p = now_ms - largest_window * 60 * 1000
                        all_trades = [t for t in self.flow_trades if t['ts'] >= start_ms_p]
                        total_qty_all = sum(t['qty'] for t in all_trades)
                        # Prepara contagem por setor
                        for sector in self._order_buckets.keys():
                            sector_trades = [t for t in all_trades if t.get('sector') == sector]
                            total_qty_sector = sum(t['qty'] for t in sector_trades)
                            buy_qty = sum(t['qty'] for t in sector_trades if t['delta'] > 0)
                            sell_qty = sum(t['qty'] for t in sector_trades if t['delta'] < 0)
                            count_trades = len(sector_trades)
                            direction = None
                            if buy_qty > sell_qty:
                                direction = "BUY"
                            elif sell_qty > buy_qty:
                                direction = "SELL"
                            else:
                                direction = "NEUTRAL"
                            avg_order_size = round(total_qty_sector / count_trades, 4) if count_trades > 0 else None
                            volume_pct = round((total_qty_sector / total_qty_all) * 100.0, 2) if total_qty_all > 0 else None
                            sentiment = "BULLISH" if direction == "BUY" else ("BEARISH" if direction == "SELL" else "NEUTRAL")
                            # Taxa de trades por segundo para atividade (apenas referência)
                            duration_seconds = largest_window * 60
                            trades_per_sec = round(count_trades / duration_seconds, 4) if duration_seconds > 0 else None
                            activity_level = None
                            # Consideramos o segmento com maior frequência de trades como HFT quando taxa >= 1 tps
                            activity_level = "HIGH" if trades_per_sec and trades_per_sec >= 1.0 else "LOW"
                            participant_analysis[sector] = {
                                "volume_pct": volume_pct,
                                "direction": direction,
                                "avg_order_size": avg_order_size,
                                "sentiment": sentiment,
                            }
                            # Inclui activity_level para todos os setores
                            participant_analysis[sector]["activity_level"] = activity_level
                    metrics["order_flow"] = order_flow
                    metrics["participant_analysis"] = participant_analysis
                except Exception as e:
                    logging.debug(f"Erro ao calcular order_flow ou participant_analysis: {e}")

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
                            "window_size": getattr(self.liquidity_heatmap, "window_size", None),
                            "cluster_threshold_pct": getattr(self.liquidity_heatmap, "cluster_threshold_pct", None),
                            "min_trades_per_cluster": getattr(self.liquidity_heatmap, "min_trades_per_cluster", None),
                            "update_interval_ms": getattr(self.liquidity_heatmap, "update_interval_ms", None),
                            "top_n": 5
                        }
                    }
                except Exception as e:
                    logging.error(f"Erro ao obter liquidity heatmap: {e}")
                    metrics["liquidity_heatmap"] = {
                        "clusters": [],
                        "supports": [],
                        "resistances": [],
                        "clusters_count": 0,
                        "meta": {
                            "window_size": getattr(self.liquidity_heatmap, "window_size", None),
                            "cluster_threshold_pct": getattr(self.liquidity_heatmap, "cluster_threshold_pct", None),
                            "min_trades_per_cluster": getattr(self.liquidity_heatmap, "min_trades_per_cluster", None),
                            "update_interval_ms": getattr(self.liquidity_heatmap, "update_interval_ms", None),
                            "top_n": 5
                        }
                    }

                return metrics
            finally:
                self._lock.release()

        except Exception as e:
            logging.error(f"Erro ao obter métricas de fluxo: {e}")
            now_ms = reference_epoch_ms if reference_epoch_ms is not None else self.time_manager.now_ms()
            time_index = self.time_manager.build_time_index(now_ms, include_local=True, timespec="milliseconds")
            return {
                "cvd": 0.0,
                "whale_buy_volume": 0.0,
                "whale_sell_volume": 0.0,
                "whale_delta": 0.0,
                "bursts": {"count": 0, "max_burst_volume": 0.0},
                "sector_flow": {name: {"buy": 0.0, "sell": 0.0, "delta": 0.0} for name in self._order_buckets},
                "timestamp": time_index["timestamp_utc"],
                "time_index": time_index,
                "liquidity_heatmap": {
                    "clusters": [],
                    "supports": [],
                    "resistances": [],
                    "clusters_count": 0,
                    "meta": {
                        "window_size": getattr(self.liquidity_heatmap, "window_size", None),
                        "cluster_threshold_pct": getattr(self.liquidity_heatmap, "cluster_threshold_pct", None),
                        "min_trades_per_cluster": getattr(self.liquidity_heatmap, "min_trades_per_cluster", None),
                        "update_interval_ms": getattr(self.liquidity_heatmap, "update_interval_ms", None),
                        "top_n": 5
                    }
                },
                "metadata": {
                    "burst_window_ms": self.burst_window_ms,
                    "burst_cooldown_ms": self.burst_cooldown_ms,
                    "in_burst": False,
                    "last_reset_ms": self.last_reset_ms,
                    "last_reset_iso_utc": self.time_manager.format_timestamp(self.last_reset_ms)
                }
            }
