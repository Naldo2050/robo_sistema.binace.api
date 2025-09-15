import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Any
import logging
import time

class LiquidityHeatmap:
    def __init__(self, window_size: int = 1000, cluster_threshold_pct: float = 0.005, 
                 min_trades_per_cluster: int = 3, update_interval_ms: int = 200):
        """
        Heatmap de Liquidez em Tempo Real — identifica clusters dinâmicos de liquidez.
        
        Parâmetros:
        - window_size: número máximo de trades para manter na janela de análise
        - cluster_threshold_pct: threshold de agrupamento (0.005 = 0.5% do preço)
        - min_trades_per_cluster: número mínimo de trades para formar um cluster
        - update_interval_ms: intervalo mínimo entre atualizações (ms)
        """
        self.window_size = window_size
        self.cluster_threshold_pct = cluster_threshold_pct
        self.min_trades_per_cluster = min_trades_per_cluster
        self.update_interval_ms = update_interval_ms
        
        # Armazenamento de trades
        self.price_levels = deque(maxlen=window_size)
        self.volume_levels = deque(maxlen=window_size)
        self.side_levels = deque(maxlen=window_size)
        self.timestamp_levels = deque(maxlen=window_size)
        
        # Clusters ativos
        self.clusters = []
        self.last_update_time = 0
        
        logging.info(f"✅ Liquidity Heatmap inicializado | Janela: {window_size} trades | Threshold: {cluster_threshold_pct*100}%")

    def add_trade(self, price: float, volume: float, side: str, timestamp_ms: int):
        """Adiciona um trade ao heatmap."""
        if not isinstance(price, (int, float)) or not isinstance(volume, (int, float)) or volume <= 0:
            return
            
        self.price_levels.append(price)
        self.volume_levels.append(volume)
        self.side_levels.append(side)
        self.timestamp_levels.append(timestamp_ms)
        
        # Atualiza clusters apenas se passou o intervalo mínimo
        if timestamp_ms - self.last_update_time >= self.update_interval_ms:
            self._update_clusters()
            self.last_update_time = timestamp_ms

    def _update_clusters(self):
        """Atualiza clusters de liquidez baseado nos trades recentes."""
        if len(self.price_levels) < self.min_trades_per_cluster:
            self.clusters = []
            return
            
        try:
            prices = list(self.price_levels)
            volumes = list(self.volume_levels)
            sides = list(self.side_levels)
            timestamps = list(self.timestamp_levels)
            
            if len(prices) == 0:
                self.clusters = []
                return
                
            # Ordena trades por preço
            sorted_indices = sorted(range(len(prices)), key=lambda i: prices[i])
            sorted_prices = [prices[i] for i in sorted_indices]
            sorted_volumes = [volumes[i] for i in sorted_indices]
            sorted_sides = [sides[i] for i in sorted_indices]
            sorted_timestamps = [timestamps[i] for i in sorted_indices]
            
            self.clusters = []
            current_cluster = None
            
            for i in range(len(sorted_prices)):
                price = sorted_prices[i]
                volume = sorted_volumes[i]
                side = sorted_sides[i]
                timestamp = sorted_timestamps[i]
                
                if current_cluster is None:
                    # Inicia novo cluster
                    current_cluster = {
                        "prices": [price],
                        "volumes": [volume],
                        "sides": [side],
                        "timestamps": [timestamp],
                        "buy_volume": volume if side == "buy" else 0,
                        "sell_volume": volume if side == "sell" else 0
                    }
                else:
                    # Verifica se o preço atual está dentro do threshold do cluster
                    cluster_prices = current_cluster["prices"]
                    cluster_center = np.mean(cluster_prices)
                    price_threshold = cluster_center * self.cluster_threshold_pct
                    
                    if abs(price - cluster_center) <= price_threshold:
                        # Adiciona ao cluster atual
                        current_cluster["prices"].append(price)
                        current_cluster["volumes"].append(volume)
                        current_cluster["sides"].append(side)
                        current_cluster["timestamps"].append(timestamp)
                        
                        if side == "buy":
                            current_cluster["buy_volume"] += volume
                        else:
                            current_cluster["sell_volume"] += volume
                    else:
                        # Finaliza cluster atual se tiver trades suficientes
                        if len(current_cluster["prices"]) >= self.min_trades_per_cluster:
                            self._finalize_cluster(current_cluster)
                        
                        # Inicia novo cluster
                        current_cluster = {
                            "prices": [price],
                            "volumes": [volume],
                            "sides": [side],
                            "timestamps": [timestamp],
                            "buy_volume": volume if side == "buy" else 0,
                            "sell_volume": volume if side == "sell" else 0
                        }
            
            # Finaliza último cluster
            if current_cluster and len(current_cluster["prices"]) >= self.min_trades_per_cluster:
                self._finalize_cluster(current_cluster)
                
            # Ordena clusters por volume total (decrescente)
            self.clusters.sort(key=lambda x: x["total_volume"], reverse=True)
            
        except Exception as e:
            logging.error(f"Erro ao atualizar clusters de liquidez: {e}")
            self.clusters = []

    def _finalize_cluster(self, cluster: Dict):
        """Finaliza e formata um cluster — versão corrigida com fallbacks."""
        try:
            if not cluster or len(cluster.get("prices", [])) == 0:
                return
                
            prices = np.array(cluster["prices"])
            volumes = np.array(cluster["volumes"])
            
            if len(prices) == 0:
                return
                
            # Valores com fallback
            center = float(np.mean(prices)) if len(prices) > 0 else 0.0
            low = float(np.min(prices)) if len(prices) > 0 else 0.0
            high = float(np.max(prices)) if len(prices) > 0 else 0.0
            width = float(high - low) if len(prices) > 1 else 0.0
            total_volume = float(np.sum(volumes)) if len(volumes) > 0 else 0.0
            buy_volume = float(cluster.get("buy_volume", 0.0))
            sell_volume = float(cluster.get("sell_volume", 0.0))
            imbalance = float(buy_volume - sell_volume)
            imbalance_ratio = float(imbalance / total_volume) if total_volume > 0 else 0.0
            trades_count = len(prices)
            avg_trade_size = float(np.mean(volumes)) if len(volumes) > 0 else 0.0
            recent_timestamp = int(np.max(cluster["timestamps"])) if len(cluster["timestamps"]) > 0 else int(time.time() * 1000)
            age_ms = int(time.time() * 1000 - recent_timestamp)
            price_std = float(np.std(prices)) if len(prices) > 1 else 0.0
            volume_std = float(np.std(volumes)) if len(volumes) > 1 else 0.0

            cluster_data = {
                "center": center,
                "low": low,
                "high": high,
                "width": width,
                "total_volume": total_volume,
                "buy_volume": buy_volume,
                "sell_volume": sell_volume,
                "imbalance": imbalance,
                "imbalance_ratio": imbalance_ratio,
                "trades_count": trades_count,
                "avg_trade_size": avg_trade_size,
                "recent_timestamp": recent_timestamp,
                "age_ms": age_ms,
                "price_std": price_std,
                "volume_std": volume_std
            }
            
            # Validação final: garante que todos os campos obrigatórios existem
            required_fields = ['center', 'total_volume', 'imbalance_ratio', 'trades_count', 'age_ms', 'high', 'low']
            for field in required_fields:
                if field not in cluster_data or not isinstance(cluster_data[field], (int, float)):
                    cluster_data[field] = 0.0

            self.clusters.append(cluster_data)
            
        except Exception as e:
            logging.error(f"Erro ao finalizar cluster: {e}")
            # Não adiciona cluster inválido

    def get_clusters(self, top_n: int = 10) -> List[Dict]:
        """Retorna top N clusters de liquidez."""
        return self.clusters[:top_n]

    def get_support_resistance(self) -> Tuple[List[float], List[float]]:
        """Identifica níveis de suporte (clusters com mais compras) e resistência (clusters com mais vendas)."""
        supports = []
        resistances = []
        
        for cluster in self.clusters:
            if cluster["imbalance"] > 0 and cluster["buy_volume"] > cluster["sell_volume"] * 1.2:
                supports.append(cluster["center"])
            elif cluster["imbalance"] < 0 and cluster["sell_volume"] > cluster["buy_volume"] * 1.2:
                resistances.append(cluster["center"])
        
        return supports, resistances

    def get_liquidity_zones(self) -> List[Dict]:
        """Retorna zonas de liquidez no formato compatível com LevelRegistry."""
        zones = []
        
        for cluster in self.clusters:
            # Zona principal (centro do cluster)
            zones.append({
                "kind": "LIQUIDITY_CLUSTER",
                "price": cluster["center"],
                "width_pct": self.cluster_threshold_pct * 2,
                "score": cluster["total_volume"] * (1 + abs(cluster["imbalance_ratio"])),
                "confluence": [f"LIQ_{cluster['trades_count']}trades", f"IMB_{cluster['imbalance_ratio']:.2f}"]
            })
            
            # Zonas de borda (high/low do cluster)
            if cluster["width"] > 0:
                zones.append({
                    "kind": "LIQUIDITY_HIGH",
                    "price": cluster["high"],
                    "width_pct": self.cluster_threshold_pct,
                    "score": cluster["total_volume"] * 0.8,
                    "confluence": ["LIQ_HIGH", f"VOL_{cluster['total_volume']:.0f}"]
                })
                
                zones.append({
                    "kind": "LIQUIDITY_LOW",
                    "price": cluster["low"],
                    "width_pct": self.cluster_threshold_pct,
                    "score": cluster["total_volume"] * 0.8,
                    "confluence": ["LIQ_LOW", f"VOL_{cluster['total_volume']:.0f}"]
                })
        
        return zones

    def get_current_liquidity_profile(self, current_price: float) -> Dict[str, Any]:
        """Retorna perfil de liquidez ao redor do preço atual."""
        nearby_clusters = []
        total_nearby_volume = 0.0
        
        for cluster in self.clusters:
            if abs(cluster["center"] - current_price) / current_price <= self.cluster_threshold_pct * 3:
                nearby_clusters.append(cluster)
                total_nearby_volume += cluster["total_volume"]
        
        if len(nearby_clusters) == 0:
            return {
                "status": "no_liquidity_nearby",
                "current_price": current_price,
                "liquidity_score": 0.0,
                "imbalance_ratio": 0.0,
                "clusters_count": 0
            }
        
        # Calcula métricas agregadas
        total_buy = sum(c["buy_volume"] for c in nearby_clusters)
        total_sell = sum(c["sell_volume"] for c in nearby_clusters)
        avg_imbalance = np.mean([c["imbalance_ratio"] for c in nearby_clusters])
        
        return {
            "status": "liquidity_detected",
            "current_price": current_price,
            "liquidity_score": total_nearby_volume,
            "imbalance_ratio": (total_buy - total_sell) / (total_buy + total_sell) if (total_buy + total_sell) > 0 else 0.0,
            "clusters_count": len(nearby_clusters),
            "avg_cluster_age_ms": int(np.mean([c["age_ms"] for c in nearby_clusters])),
            "total_volume": total_nearby_volume,
            "buy_volume": total_buy,
            "sell_volume": total_sell
        }

    def clear_old_data(self, max_age_ms: int = 300000):  # 5 minutos
        """Remove trades muito antigos para manter o heatmap atualizado."""
        if len(self.timestamp_levels) == 0:
            return
            
        current_time = time.time() * 1000
        cutoff_time = current_time - max_age_ms
        
        # Remove trades antigos do final das deques (os mais antigos)
        while len(self.timestamp_levels) > 0 and self.timestamp_levels[0] < cutoff_time:
            self.price_levels.popleft()
            self.volume_levels.popleft()
            self.side_levels.popleft()
            self.timestamp_levels.popleft()
        
        # Recalcula clusters
        self._update_clusters()