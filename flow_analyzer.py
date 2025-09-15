import time
import logging
from threading import Lock
from collections import deque
import config
import random

# 🔹 IMPORTA TIME MANAGER
from time_manager import TimeManager

# 🔹 NOVO: IMPORTA LIQUIDITY HEATMAP
from liquidity_heatmap import LiquidityHeatmap

class FlowAnalyzer:
    def __init__(self):
        # Métricas de CVD
        self.cvd = 0.0

        # Whale Flow
        self.whale_threshold = getattr(config, "WHALE_TRADE_THRESHOLD", 5.0)
        self.whale_buy_volume = 0.0
        self.whale_sell_volume = 0.0
        self.whale_delta = 0.0

        # Controle de Reset
        self.last_reset_time = time.time()
        self.reset_interval_seconds = getattr(config, "CVD_RESET_INTERVAL_HOURS", 24) * 3600
        self._lock = Lock()
        
        # 🔹 Inicializa TimeManager
        self.time_manager = TimeManager()
        
        logging.info("✅ Analisador de Fluxo Contínuo inicializado (CVD, Whale Flow, Bursts, Buckets).")

        # Histórico de trades (para bursts)
        self.recent_trades = deque(maxlen=500)  # janela curta para detector
        self.bursts = {"count": 0, "max_burst_volume": 0.0}
        self.burst_window_ms = 200

        # Threshold dedicado para bursts (fallback para whale_threshold se não configurado)
        self.burst_volume_threshold = getattr(config, "BURST_VOLUME_THRESHOLD", self.whale_threshold)
        if not hasattr(config, "BURST_VOLUME_THRESHOLD"):
            logging.info(f"ℹ️ BURST_VOLUME_THRESHOLD não encontrado em config. Usando fallback = {self.burst_volume_threshold} BTC.")

        # Estado de burst (para evitar supercontagem)
        self._in_burst = False
        self._last_burst_end_ms = 0
        self.burst_cooldown_ms = 200  # período refratário mínimo entre bursts

        # Segmentação de players
        order_buckets = getattr(config, "ORDER_SIZE_BUCKETS", {
            "retail": (0, 0.5),
            "mid": (0.5, 2.0),
            "whale": (2.0, 9999.0)
        })
        self.sector_flow = {name: {"buy": 0.0, "sell": 0.0, "delta": 0.0} for name in order_buckets}
        self._order_buckets = order_buckets

        # 🔹 NOVO: Inicializa Liquidity Heatmap
        self.liquidity_heatmap = LiquidityHeatmap(
            window_size=2000,
            cluster_threshold_pct=0.003,  # 0.3% do preço
            min_trades_per_cluster=5,
            update_interval_ms=100
        )

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
            self.last_reset_time = time.time()
            logging.info("🔄 Métricas de Fluxo resetadas (CVD, Whale Flow, Bursts, Buckets).")
        except Exception as e:
            logging.error(f"Erro ao resetar métricas: {e}")
            # 🔹 Fallback: mantém valores atuais em caso de erro
            pass

    def _check_reset(self):
        """Verifica se as métricas devem ser resetadas com base no tempo."""
        try:
            # 🔹 USA TIME MANAGER PARA TIMESTAMP
            current_time = self.time_manager.now() / 1000.0  # converte para segundos
            if current_time - self.last_reset_time > self.reset_interval_seconds:
                with self._lock:
                    self._reset_metrics()
        except Exception as e:
            logging.error(f"Erro ao verificar reset de métricas: {e}")
            # 🔹 Fallback: não reseta em caso de erro
            pass

    def _prune_recent(self, now_ms: int):
        """Remove trades antigos fora da janela de bursts."""
        try:
            while self.recent_trades and (now_ms - self.recent_trades[0][0] > self.burst_window_ms):
                self.recent_trades.popleft()
        except Exception as e:
            logging.error(f"Erro ao remover trades antigos: {e}")
            # 🔹 Fallback: limpa a fila em caso de erro
            self.recent_trades.clear()

    def _update_bursts(self, ts_ms: int, qty: float):
        """Detecta bursts em microtempo (200ms) com controle de estado para evitar supercontagem."""
        try:
            # adiciona trade atual
            self.recent_trades.append((ts_ms, qty))
            # remove antigos fora da janela
            self._prune_recent(ts_ms)

            # volume acumulado dentro da janela
            burst_volume = sum(q for _, q in self.recent_trades)
            threshold = self.burst_volume_threshold

            # início de burst (transição)
            if not self._in_burst:
                if burst_volume >= threshold and (ts_ms - self._last_burst_end_ms) >= self.burst_cooldown_ms:
                    self.bursts["count"] += 1
                    self._in_burst = True
                    # atualiza pico de volume do burst
                    if burst_volume > self.bursts["max_burst_volume"]:
                        self.bursts["max_burst_volume"] = burst_volume
            else:
                # já dentro de burst: atualiza pico
                if burst_volume > self.bursts["max_burst_volume"]:
                    self.bursts["max_burst_volume"] = burst_volume
                # termina burst quando esvazia significativamente
                if burst_volume < threshold * 0.5:
                    self._in_burst = False
                    self._last_burst_end_ms = ts_ms
        except Exception as e:
            logging.error(f"Erro ao atualizar bursts: {e}")
            # 🔹 Fallback: reseta estado de burst em caso de erro
            self._in_burst = False
            self._last_burst_end_ms = ts_ms

    def _update_sector_flow(self, qty: float, trade_delta: float):
        """Classifica trade em buckets (retail/mid/whale)"""
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
            # 🔹 Fallback: não atualiza em caso de erro
            pass

    def process_trade(self, trade: dict):
        """Processa um único trade para atualizar o CVD e o Flow detalhado."""
        try:
            self._check_reset()
            
            # Validação robusta dos dados
            if not isinstance(trade, dict):
                logging.warning("Trade inválido: não é um dicionário")
                return
                
            qty = float(trade.get('q', 0)) if trade.get('q') is not None else 0.0
            is_buyer_maker = trade.get('m', False)
            ts_str = trade.get('T')
            
            if not ts_str:
                logging.debug("Trade recebido sem timestamp 'T'. Descartado.")
                return
                
            try:
                ts = int(ts_str)
            except (ValueError, TypeError):
                logging.warning(f"Timestamp inválido: {ts_str}")
                return

            trade_delta = -qty if is_buyer_maker else qty

            with self._lock:
                # Atualiza CVD
                self.cvd += trade_delta

                # Whale Flow (threshold dedicado)
                if qty >= self.whale_threshold:
                    if trade_delta > 0:
                        self.whale_buy_volume += qty
                    else:
                        self.whale_sell_volume += abs(trade_delta)
                    self.whale_delta += trade_delta

                # Bursts (microtempo)
                self._update_bursts(ts, qty)

                # Segmentação por buckets
                self._update_sector_flow(qty, trade_delta)

                # 🔹 NOVO: Adiciona trade ao Liquidity Heatmap
                side = "buy" if not is_buyer_maker else "sell"  # buyer_maker = vendedor
                self.liquidity_heatmap.add_trade(
                    price=float(trade.get('p', 0)),
                    volume=qty,
                    side=side,
                    timestamp_ms=ts
                )

        except Exception as e:
            logging.warning(f"Erro ao processar trade: {trade} - Erro: {e}")
            # 🔹 Fallback: continua processando outros trades

    def get_flow_metrics(self) -> dict:
        """Retorna as métricas de fluxo atuais de forma segura."""
        try:
            # 🔹 OTIMIZADO: usa timeout no lock (Fase 3)
            acquired = self._lock.acquire(timeout=5.0)  # 5 segundos de timeout
            if not acquired:
                logging.critical("💀 DEADLOCK DETECTADO NO FLOW ANALYZER!")
                # 🔹 Fallback: retorna métricas zeradas
                return {
                    "cvd": 0.0,
                    "whale_buy_volume": 0.0,
                    "whale_sell_volume": 0.0,
                    "whale_delta": 0.0,
                    "bursts": {"count": 0, "max_burst_volume": 0.0},
                    "sector_flow": {name: {"buy": 0.0, "sell": 0.0, "delta": 0.0} for name in self._order_buckets},
                    "timestamp": self.time_manager.now_iso(),
                    "liquidity_heatmap": {"clusters": [], "supports": [], "resistances": [], "clusters_count": 0}  # 🔹 NOVO
                }
            
            try:
                metrics = {
                    "cvd": self.cvd,
                    "whale_buy_volume": self.whale_buy_volume,
                    "whale_sell_volume": self.whale_sell_volume,
                    "whale_delta": self.whale_delta,
                    "bursts": self.bursts.copy(),
                    "sector_flow": {k: v.copy() for k, v in self.sector_flow.items()},
                    "timestamp": self.time_manager.now_iso()
                }
                
                # 🔹 NOVO: Adiciona dados do Liquidity Heatmap
                try:
                    clusters = self.liquidity_heatmap.get_clusters(top_n=5)
                    supports, resistances = self.liquidity_heatmap.get_support_resistance()
                    metrics["liquidity_heatmap"] = {
                        "clusters": clusters,
                        "supports": supports,
                        "resistances": resistances,
                        "clusters_count": len(clusters)
                    }
                except Exception as e:
                    logging.error(f"Erro ao obter liquidity heatmap: {e}")
                    metrics["liquidity_heatmap"] = {"clusters": [], "supports": [], "resistances": [], "clusters_count": 0}
                
                return metrics
            finally:
                self._lock.release()
                
        except Exception as e:
            logging.error(f"Erro ao obter métricas de fluxo: {e}")
            # 🔹 Fallback: retorna métricas zeradas em caso de erro
            return {
                "cvd": 0.0,
                "whale_buy_volume": 0.0,
                "whale_sell_volume": 0.0,
                "whale_delta": 0.0,
                "bursts": {"count": 0, "max_burst_volume": 0.0},
                "sector_flow": {name: {"buy": 0.0, "sell": 0.0, "delta": 0.0} for name in self._order_buckets},
                "timestamp": self.time_manager.now_iso(),
                "liquidity_heatmap": {"clusters": [], "supports": [], "resistances": [], "clusters_count": 0}  # 🔹 NOVO
            }