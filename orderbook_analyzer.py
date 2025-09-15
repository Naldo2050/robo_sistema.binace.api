import logging
import requests
import time
import numpy as np
from collections import deque
from datetime import datetime
from zoneinfo import ZoneInfo
from config import LIQUIDITY_FLOW_ALERT_PERCENTAGE, WALL_STD_DEV_FACTOR

# 🔹 IMPORTA TIME MANAGER
from time_manager import TimeManager

class OrderBookAnalyzer:
    def __init__(self, symbol: str, liquidity_flow_alert_percentage: float = 0.4, wall_std_dev_factor: float = 3.0):
        self.symbol = symbol
        self.liquidity_flow_alert_percentage = liquidity_flow_alert_percentage
        self.wall_std_dev_factor = wall_std_dev_factor
        self.ny_tz = ZoneInfo("America/New_York")
        
        # 🔹 Inicializa TimeManager
        self.time_manager = TimeManager()
        
        # URLs da Binance
        self.depth_url = "https://fapi.binance.com/fapi/v1/depth"
        self.last_snapshot = None
        self.last_liquidity_check_time = 0
        self.liquidity_check_interval = 5  # segundos

        # Histórico para detecção de spoofing/layering
        self.order_history = deque(maxlen=100)
        self.last_ob = None  # 🔹 NOVO: para detecção de recarga de icebergs

        logging.info(f"✅ OrderBook Analyzer inicializado para {symbol} | Alerta de fluxo: {liquidity_flow_alert_percentage*100}% | Wall STD: {wall_std_dev_factor}x")

    def _fetch_orderbook(self, limit: int = 500):
        """Busca o livro de ofertas da Binance."""
        try:
            params = {"symbol": self.symbol, "limit": limit}
            response = requests.get(self.depth_url, params=params, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Erro ao buscar orderbook: {e}")
            return None

    def _calculate_liquidity_metrics(self, bids: list, asks: list):
        """Calcula métricas de liquidez e pressão."""
        try:
            # Converte para arrays numpy
            bid_prices = np.array([float(bid[0]) for bid in bids])
            bid_volumes = np.array([float(bid[1]) for bid in bids])
            ask_prices = np.array([float(ask[0]) for ask in asks])
            ask_volumes = np.array([float(ask[1]) for ask in asks])

            # Profundidade total em USD
            bid_depth_usd = np.sum(bid_prices * bid_volumes)
            ask_depth_usd = np.sum(ask_prices * ask_volumes)

            # Imbalance e ratio
            total_liquidity = bid_depth_usd + ask_depth_usd
            imbalance = (bid_depth_usd - ask_depth_usd) / total_liquidity if total_liquidity > 0 else 0
            volume_ratio = bid_depth_usd / ask_depth_usd if ask_depth_usd > 0 else 0

            # Pressão (ponderada pela distância do mid)
            mid_price = (bid_prices[0] + ask_prices[0]) / 2 if len(bid_prices) > 0 and len(ask_prices) > 0 else 0
            if mid_price > 0:
                bid_pressure = np.sum(bid_volumes * (1 - np.abs(bid_prices - mid_price) / mid_price))
                ask_pressure = np.sum(ask_volumes * (1 - np.abs(ask_prices - mid_price) / mid_price))
                total_pressure = bid_pressure + ask_pressure
                pressure = (bid_pressure - ask_pressure) / total_pressure if total_pressure > 0 else 0
            else:
                pressure = 0

            return {
                "imbalance": float(imbalance),
                "volume_ratio": float(volume_ratio),
                "pressure": float(pressure),
                "spread": float(ask_prices[0] - bid_prices[0]) if len(ask_prices) > 0 and len(bid_prices) > 0 else 0,
                "spread_percent": float((ask_prices[0] - bid_prices[0]) / mid_price * 100) if mid_price > 0 and len(ask_prices) > 0 and len(bid_prices) > 0 else 0,
                "bid_depth_usd": float(bid_depth_usd),
                "ask_depth_usd": float(ask_depth_usd),
                "mid_price": float(mid_price)
            }
        except Exception as e:
            logging.error(f"Erro ao calcular métricas de liquidez: {e}")
            return {
                "imbalance": 0.0,
                "volume_ratio": 1.0,
                "pressure": 0.0,
                "spread": 0.0,
                "spread_percent": 0.0,
                "bid_depth_usd": 0.0,
                "ask_depth_usd": 0.0,
                "mid_price": 0.0
            }

    def _detect_walls(self, prices: np.array, volumes: np.array, side: str) -> list:
        """Detecta paredes de liquidez (ordens > X desvios padrão da média)."""
        if len(volumes) == 0:
            return []
        
        mean_vol = np.mean(volumes)
        std_vol = np.std(volumes)
        threshold = mean_vol + (self.wall_std_dev_factor * std_vol)
        
        walls = []
        for i, vol in enumerate(volumes):
            if vol >= threshold:
                walls.append({
                    "price": float(prices[i]),
                    "volume": float(vol),
                    "side": side,
                    "threshold": float(threshold)
                })
        return walls

    def _detect_spoofing_layering(self, current_snapshot):
        """Detecta spoofing/layering baseado em mudanças rápidas de ordens."""
        try:
            current_time = time.time()
            self.order_history.append({
                "timestamp": current_time,
                "snapshot": current_snapshot
            })

            if len(self.order_history) < 5:
                return {"avg_order_lifetime_ms": 0, "short_ttl_orders": 0, "spoofing_detected": False, "layering_detected": False}

            # Calcula vida média das ordens
            lifetimes = []
            for i in range(1, len(self.order_history)):
                prev = self.order_history[i-1]["snapshot"]
                curr = self.order_history[i]["snapshot"]
                # Simplificação: compara profundidade total
                prev_bid = sum(float(b[1]) for b in prev.get("bids", []))
                curr_bid = sum(float(b[1]) for b in curr.get("bids", []))
                if abs(prev_bid - curr_bid) / prev_bid > 0.5:  # 50% de mudança
                    lifetimes.append((self.order_history[i]["timestamp"] - self.order_history[i-1]["timestamp"]) * 1000)

            avg_lifetime = np.mean(lifetimes) if lifetimes else 0
            short_ttl = sum(1 for lt in lifetimes if lt < 1000)  # menos de 1s

            spoofing_detected = short_ttl > len(lifetimes) * 0.3  # 30% das ordens com TTL curto
            layering_detected = len(lifetimes) > 5 and avg_lifetime < 5000  # média < 5s

            return {
                "avg_order_lifetime_ms": float(avg_lifetime),
                "short_ttl_orders": int(short_ttl),
                "spoofing_detected": bool(spoofing_detected),
                "layering_detected": bool(layering_detected)
            }
        except Exception as e:
            logging.error(f"Erro ao detectar spoofing/layering: {e}")
            return {
                "avg_order_lifetime_ms": 0.0,
                "short_ttl_orders": 0,
                "spoofing_detected": False,
                "layering_detected": False
            }

    def detect_iceberg_reloads(self, current_snapshot, previous_snapshot, threshold_pct=0.8):
        """
        Detecta icebergs que "recarregam" — ordens que são repostas rapidamente após serem consumidas.
        """
        if not previous_snapshot:
            return False, 0, 0

        # Compara profundidade de asks e bids
        prev_bid_usd = previous_snapshot.get("spread_metrics", {}).get("bid_depth_usd", 0)
        curr_bid_usd = current_snapshot.get("spread_metrics", {}).get("bid_depth_usd", 0)
        
        prev_ask_usd = previous_snapshot.get("spread_metrics", {}).get("ask_depth_usd", 0)
        curr_ask_usd = current_snapshot.get("spread_metrics", {}).get("ask_depth_usd", 0)

        # Verifica se houve recarga significativa (>80% do volume anterior)
        bid_reloaded = curr_bid_usd > prev_bid_usd * threshold_pct
        ask_reloaded = curr_ask_usd > prev_ask_usd * threshold_pct

        return (bid_reloaded or ask_reloaded), curr_bid_usd, curr_ask_usd

    def analyze_order_book(self) -> dict:
        """Analisa o livro de ofertas e retorna evento estruturado."""
        try:
            ob = self._fetch_orderbook()
            if not ob:
                return {"is_signal": False, "error": "Falha ao buscar orderbook"}

            bids = ob.get("bids", [])[:20]  # Top 20 bids
            asks = ob.get("asks", [])[:20]  # Top 20 asks

            if len(bids) == 0 or len(asks) == 0:
                return {"is_signal": False, "error": "Orderbook vazio"}

            # Métricas de liquidez
            metrics = self._calculate_liquidity_metrics(bids, asks)

            # Detecção de paredes
            bid_prices = np.array([float(bid[0]) for bid in bids])
            bid_volumes = np.array([float(bid[1]) for bid in bids])
            ask_prices = np.array([float(ask[0]) for ask in asks])
            ask_volumes = np.array([float(ask[1]) for ask in asks])

            bid_walls = self._detect_walls(bid_prices, bid_volumes, "buy")
            ask_walls = self._detect_walls(ask_prices, ask_volumes, "sell")

            # Detecção de spoofing/layering
            lifecycle = self._detect_spoofing_layering(ob)

            # 🔹 NOVO: Detecção de recarga de icebergs
            iceberg_reloaded, bid_reload, ask_reload = self.detect_iceberg_reloads(ob, self.last_ob)
            self.last_ob = ob.copy()

            # Gera alertas
            alerts = []
            current_time = time.time()

            if current_time - self.last_liquidity_check_time > self.liquidity_check_interval:
                if self.last_snapshot:
                    # Compara profundidade
                    prev_bid_usd = self.last_snapshot.get("spread_metrics", {}).get("bid_depth_usd", 0)
                    prev_ask_usd = self.last_snapshot.get("spread_metrics", {}).get("ask_depth_usd", 0)
                    curr_bid_usd = metrics["bid_depth_usd"]
                    curr_ask_usd = metrics["ask_depth_usd"]

                    if prev_bid_usd > 0 and abs(curr_bid_usd - prev_bid_usd) / prev_bid_usd > self.liquidity_flow_alert_percentage:
                        change_pct = (curr_bid_usd - prev_bid_usd) / prev_bid_usd * 100
                        alerts.append(f"FLUXO: {'Aumento' if change_pct > 0 else 'Retirada'} de {abs(change_pct):.1f}% na liquidez de compra (Bids).")
                    
                    if prev_ask_usd > 0 and abs(curr_ask_usd - prev_ask_usd) / prev_ask_usd > self.liquidity_flow_alert_percentage:
                        change_pct = (curr_ask_usd - prev_ask_usd) / prev_ask_usd * 100
                        alerts.append(f"FLUXO: {'Aumento' if change_pct > 0 else 'Retirada'} de {abs(change_pct):.1f}% na liquidez de venda (Asks).")

                self.last_snapshot = {
                    "spread_metrics": {
                        "bid_depth_usd": metrics["bid_depth_usd"],
                        "ask_depth_usd": metrics["ask_depth_usd"]
                    }
                }
                self.last_liquidity_check_time = current_time

            # Adiciona paredes aos alertas
            for wall in bid_walls[:3]:
                alerts.append(f"PAREDE DE COMPRA: {wall['volume']:.2f} @ ${wall['price']:,.2f} (limite: {wall['threshold']:.2f})")
            for wall in ask_walls[:3]:
                alerts.append(f"PAREDE DE VENDA: {wall['volume']:.2f} @ ${wall['price']:,.2f} (limite: {wall['threshold']:.2f})")

            # Decisão de sinal
            is_signal = len(alerts) > 0
            resultado = "Neutro"
            if metrics["pressure"] < -0.7:
                resultado = "Oferta Forte"
            elif metrics["pressure"] > 0.7:
                resultado = "Demanda Forte"
            elif len(alerts) > 0:
                resultado = "Alerta de Liquidez"

            # 🔹 USA TIME MANAGER
            timestamp = self.time_manager.now_iso()

            event = {
                "is_signal": is_signal,
                "timestamp": timestamp,
                "tipo_evento": "OrderBook",
                "resultado_da_batalha": resultado,
                "descricao": " | ".join(alerts) if alerts else "Nenhum alerta gerado.",
                "ativo": self.symbol,
                "imbalance": metrics["imbalance"],
                "volume_ratio": metrics["volume_ratio"],
                "pressure": metrics["pressure"],
                "order_lifecycle": lifecycle,
                "spread_metrics": {
                    "spread": metrics["spread"],
                    "spread_percent": metrics["spread_percent"],
                    "bid_depth_usd": metrics["bid_depth_usd"],
                    "ask_depth_usd": metrics["ask_depth_usd"]
                },
                "market_impact_buy": self._simulate_market_impact(bid_prices, bid_volumes, 10),  # Simula impacto de comprar 10 BTC
                "market_impact_sell": self._simulate_market_impact(ask_prices, ask_volumes, 10),  # Simula impacto de vender 10 BTC
                "alertas_liquidez": alerts,
                "layer": "signal",
                "iceberg_reloaded": iceberg_reloaded,  # 🔹 NOVO: flag de recarga
                "bid_reload_usd": bid_reload,
                "ask_reload_usd": ask_reload
            }

            return event

        except Exception as e:
            logging.error(f"Erro ao analisar orderbook: {e}")
            return {"is_signal": False, "error": str(e)}

    def _simulate_market_impact(self, prices: np.array, volumes: np.array, target_volume: float):
        """Simula o impacto de mercado ao executar uma ordem de mercado."""
        try:
            if len(prices) == 0 or len(volumes) == 0:
                return {"impact_usd": 0.0, "slippage_percent": 0.0, "avg_filled_price": 0.0, "final_price": 0.0}

            total_volume = 0.0
            total_cost = 0.0
            final_price = prices[0]

            for i in range(len(prices)):
                if total_volume >= target_volume:
                    break
                vol = min(volumes[i], target_volume - total_volume)
                total_volume += vol
                total_cost += vol * prices[i]
                final_price = prices[i]

            avg_price = total_cost / total_volume if total_volume > 0 else prices[0]
            slippage = (avg_price - prices[0]) / prices[0] if len(prices) > 0 and prices[0] > 0 else 0

            return {
                "impact_usd": float(avg_price - prices[0]) if len(prices) > 0 else 0.0,
                "slippage_percent": float(slippage * 100),
                "avg_filled_price": float(avg_price),
                "final_price": float(final_price)
            }
        except Exception as e:
            logging.error(f"Erro ao simular impacto de mercado: {e}")
            return {
                "impact_usd": 0.0,
                "slippage_percent": 0.0,
                "avg_filled_price": 0.0,
                "final_price": 0.0
            }