# liquidity_heatmap.py v2.1.0 - CORREÇÃO DE TIMESTAMPS

from __future__ import annotations
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Any, Optional
import logging
import time


def _now_ms() -> int:
    return int(time.time() * 1000)


def _norm_side(side: Any) -> str:
    """
    Normaliza o lado do trade para 'buy' ou 'sell'.
    Aceita bool, números e strings diversas.
    """
    if isinstance(side, bool):
        # sua convenção upstream: m=True -> taker SELL
        return "sell" if side else "buy"
    if isinstance(side, (int, float)):
        return "sell" if int(side) != 0 else "buy"
    if isinstance(side, str):
        s = side.strip().lower()
        if s in {"sell", "ask", "s", "venda", "vender"}:
            return "sell"
        if s in {"buy", "bid", "b", "compra", "comprar"}:
            return "buy"
    # fallback neutro (trataremos como buy para não "sumir" volume)
    return "buy"


class LiquidityHeatmap:
    def __init__(
        self,
        window_size: int = 1000,
        cluster_threshold_pct: float = 0.005,
        min_trades_per_cluster: int = 3,
        update_interval_ms: int = 200,
    ):
        """
        Heatmap de Liquidez em Tempo Real — identifica clusters dinâmicos de liquidez.

        🔹 CORREÇÕES v2.1.0:
          ✅ Validação de timestamps (first_seen <= last_seen)
          ✅ Cálculo correto de age_ms
          ✅ Proteção contra timestamps invertidos
          ✅ Contadores de validação

        Parâmetros:
        - window_size: número máximo de trades para manter na janela de análise
        - cluster_threshold_pct: threshold de agrupamento (0.005 = 0.5% do preço)
        - min_trades_per_cluster: número mínimo de trades para formar um cluster
        - update_interval_ms: intervalo mínimo entre atualizações (ms)
        """
        self.window_size = int(window_size)
        self.cluster_threshold_pct = float(cluster_threshold_pct)
        self.min_trades_per_cluster = int(min_trades_per_cluster)
        self.update_interval_ms = int(update_interval_ms)

        # Armazenamento de trades
        self.price_levels: deque[float] = deque(maxlen=self.window_size)
        self.volume_levels: deque[float] = deque(maxlen=self.window_size)
        self.side_levels: deque[str] = deque(maxlen=self.window_size)
        self.timestamp_levels: deque[int] = deque(maxlen=self.window_size)

        # Clusters ativos (lista de dicts "finalizados")
        self.clusters: List[Dict[str, Any]] = []
        self.last_update_time: int = 0

        # 🆕 Contadores de validação
        self._timestamp_corrections = 0
        self._invalid_timestamps = 0

        logging.info(
            "✅ Liquidity Heatmap v2.1.0 inicializado | Janela: %d trades | Threshold: %.3f%%",
            self.window_size,
            self.cluster_threshold_pct * 100.0,
        )

    # ------------------------------------------------------------------ #
    # Ingestão
    # ------------------------------------------------------------------ #
    def add_trade(self, price: float, volume: float, side: Any, timestamp_ms: int):
        """Adiciona um trade ao heatmap."""
        try:
            p = float(price)
            v = float(volume)
            t = int(timestamp_ms)
        except Exception:
            return

        if not (p > 0 and v > 0 and t > 0):
            return

        s = _norm_side(side)

        self.price_levels.append(p)
        self.volume_levels.append(v)
        self.side_levels.append(s)
        self.timestamp_levels.append(t)

        # Atualiza clusters apenas se passou o intervalo mínimo
        if t - self.last_update_time >= self.update_interval_ms or len(self.clusters) == 0:
            self._update_clusters()
            self.last_update_time = t

    # ------------------------------------------------------------------ #
    # Clusterização
    # ------------------------------------------------------------------ #
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

            if not prices:
                self.clusters = []
                return

            # Ordena trades por preço
            idx = sorted(range(len(prices)), key=lambda i: prices[i])
            sorted_prices = [prices[i] for i in idx]
            sorted_volumes = [volumes[i] for i in idx]
            sorted_sides = [sides[i] for i in idx]
            sorted_timestamps = [timestamps[i] for i in idx]

            clusters: List[Dict[str, Any]] = []
            current: Optional[Dict[str, Any]] = None

            for i in range(len(sorted_prices)):
                price = float(sorted_prices[i])
                volume = float(sorted_volumes[i])
                side = _norm_side(sorted_sides[i])
                ts = int(sorted_timestamps[i])

                if current is None:
                    # Inicia novo cluster
                    current = {
                        "prices": [price],
                        "volumes": [volume],
                        "sides": [side],
                        "timestamps": [ts],
                        "buy_volume": volume if side == "buy" else 0.0,
                        "sell_volume": volume if side == "sell" else 0.0,
                        "first_seen_ms": ts,
                        "last_seen_ms": ts,
                    }
                else:
                    # Verifica se o preço atual está dentro do threshold do cluster (baseado no centro corrente)
                    center = float(np.mean(current["prices"]))
                    price_threshold = max(0.01, center * self.cluster_threshold_pct)
                    if abs(price - center) <= price_threshold:
                        # Agrega no cluster atual
                        current["prices"].append(price)
                        current["volumes"].append(volume)
                        current["sides"].append(side)
                        current["timestamps"].append(ts)
                        if side == "buy":
                            current["buy_volume"] += volume
                        else:
                            current["sell_volume"] += volume
                        
                        # 🆕 CORREÇÃO CRÍTICA: Atualizar first_seen e last_seen corretamente
                        # Como os trades estão ordenados por PREÇO (não timestamp),
                        # precisamos verificar min/max
                        current["first_seen_ms"] = min(current["first_seen_ms"], ts)
                        current["last_seen_ms"] = max(current["last_seen_ms"], ts)
                        
                    else:
                        # Finaliza cluster atual se tiver trades suficientes
                        if len(current["prices"]) >= self.min_trades_per_cluster:
                            clusters.append(self._finalize_cluster(current))
                        # Inicia novo cluster
                        current = {
                            "prices": [price],
                            "volumes": [volume],
                            "sides": [side],
                            "timestamps": [ts],
                            "buy_volume": volume if side == "buy" else 0.0,
                            "sell_volume": volume if side == "sell" else 0.0,
                            "first_seen_ms": ts,
                            "last_seen_ms": ts,
                        }

            # Finaliza último cluster
            if current and len(current["prices"]) >= self.min_trades_per_cluster:
                clusters.append(self._finalize_cluster(current))

            # Ordena clusters por volume total (desc) e recência (desc)
            clusters.sort(key=lambda c: (c["total_volume"], c["last_seen_ms"]), reverse=True)
            self.clusters = clusters

        except Exception as e:
            logging.error(f"Erro ao atualizar clusters de liquidez: {e}")
            self.clusters = []

    def _finalize_cluster(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Finaliza e formata um cluster — versão robusta com fallbacks e aliases.
        
        🆕 CORREÇÕES v2.1.0:
          - Sempre recalcula first_seen e last_seen dos timestamps reais
          - Valida que first_seen <= last_seen
          - Calcula age_ms corretamente
        """
        prices = np.asarray(raw.get("prices", []), dtype=float)
        volumes = np.asarray(raw.get("volumes", []), dtype=float)
        timestamps = np.asarray(raw.get("timestamps", []), dtype=np.int64)

        now_ms = _now_ms()

        center = float(np.mean(prices)) if prices.size else 0.0
        low = float(np.min(prices)) if prices.size else 0.0
        high = float(np.max(prices)) if prices.size else 0.0
        width = max(0.0, high - low) if prices.size > 1 else 0.0

        total_volume = float(np.sum(volumes)) if volumes.size else 0.0
        buy_volume = float(raw.get("buy_volume", 0.0) or 0.0)
        sell_volume = float(raw.get("sell_volume", 0.0) or 0.0)
        # Fallback de total_volume se vier inconsistente
        if total_volume <= 0 and (buy_volume > 0 or sell_volume > 0):
            total_volume = buy_volume + sell_volume

        imbalance = float(buy_volume - sell_volume)
        imb_ratio = (imbalance / total_volume) if total_volume > 0 else 0.0
        # Clamp em [-1, 1]
        imb_ratio = max(-1.0, min(1.0, imb_ratio))

        trades_count = int(prices.size)
        avg_trade_size = float(np.mean(volumes)) if volumes.size else 0.0

        # 🆕 CORREÇÃO CRÍTICA: Sempre recalcular first_seen e last_seen dos timestamps REAIS
        # Não confiar nos valores de raw, pois podem estar invertidos
        if timestamps.size > 0:
            first_seen_ms = int(timestamps.min())
            last_seen_ms = int(timestamps.max())
        else:
            # Fallback se não houver timestamps
            first_seen_ms = int(raw.get("first_seen_ms", now_ms))
            last_seen_ms = int(raw.get("last_seen_ms", now_ms))
        
        # 🆕 VALIDAÇÃO: Garantir que first_seen <= last_seen
        if first_seen_ms > last_seen_ms:
            logging.warning(
                f"⚠️ TIMESTAMP INVERTIDO no cluster (center={center:.2f}): "
                f"first_seen ({first_seen_ms}) > last_seen ({last_seen_ms}). "
                f"Corrigindo..."
            )
            self._timestamp_corrections += 1
            # Trocar os valores
            first_seen_ms, last_seen_ms = last_seen_ms, first_seen_ms
        
        # Validar que timestamps são positivos
        if first_seen_ms <= 0 or last_seen_ms <= 0:
            logging.warning(
                f"⚠️ Timestamps inválidos no cluster (center={center:.2f}): "
                f"first={first_seen_ms}, last={last_seen_ms}. Usando now_ms."
            )
            self._invalid_timestamps += 1
            first_seen_ms = now_ms
            last_seen_ms = now_ms
        
        recent_timestamp = last_seen_ms  # alias primário
        
        # 🆕 CORREÇÃO: age_ms calculado corretamente
        # age_ms = tempo desde a última visualização do cluster
        age_ms = max(0, now_ms - last_seen_ms)
        
        # 🆕 VALIDAÇÃO ADICIONAL: Verificar consistência
        cluster_duration_ms = last_seen_ms - first_seen_ms
        if cluster_duration_ms < 0:
            logging.error(
                f"🔴 ERRO: cluster_duration_ms negativo! "
                f"first={first_seen_ms}, last={last_seen_ms}, duration={cluster_duration_ms}"
            )
            # Forçar correção
            first_seen_ms = last_seen_ms
            cluster_duration_ms = 0

        price_std = float(np.std(prices)) if prices.size > 1 else 0.0
        volume_std = float(np.std(volumes)) if volumes.size > 1 else 0.0

        # Tamanho do "bin" equivalente ao threshold atual (útil para debug)
        bin_threshold_usd = max(0.01, center * self.cluster_threshold_pct)

        cluster_data: Dict[str, Any] = {
            "center": center,
            "low": low,
            "high": high,
            "width": width,
            "total_volume": total_volume,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "imbalance": imbalance,
            "imbalance_ratio": imb_ratio,
            "trades_count": trades_count,
            "avg_trade_size": avg_trade_size,
            # recência / idade + aliases para o normalizador do FlowAnalyzer
            "recent_timestamp": recent_timestamp,
            "recent_ts_ms": recent_timestamp,
            "last_seen_ms": last_seen_ms,      # 🆕 Validado
            "first_seen_ms": first_seen_ms,    # 🆕 Validado
            "age_ms": age_ms,                  # 🆕 Correto
            "cluster_duration_ms": cluster_duration_ms,  # 🆕 Novo campo útil
            # dispersões
            "price_std": price_std,
            "volume_std": volume_std,
            # meta
            "bin_threshold_usd": bin_threshold_usd,
        }

        # Validação final: garante tipos numéricos
        for k in ("center", "low", "high", "width", "total_volume", "imbalance_ratio", "trades_count", "age_ms"):
            v = cluster_data.get(k)
            if k == "trades_count":
                try:
                    cluster_data[k] = int(v)
                except Exception:
                    cluster_data[k] = 0
            else:
                try:
                    cluster_data[k] = float(v)
                except Exception:
                    cluster_data[k] = 0.0

        return cluster_data

    # ------------------------------------------------------------------ #
    # API pública para consumo externo
    # ------------------------------------------------------------------ #
    def get_clusters(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Retorna top N clusters de liquidez (já finalizados)."""
        if not self.clusters:
            return []
        n = max(1, int(top_n))
        # retorna uma cópia rasa para evitar mutações externas
        return [dict(c) for c in self.clusters[:n]]

    def get_support_resistance(self) -> Tuple[List[float], List[float]]:
        """
        Identifica níveis de suporte (clusters com mais compras) e resistência (clusters com mais vendas).
        Critério simples: predominância > 20% e desequilíbrio na direção.
        """
        supports: List[float] = []
        resistances: List[float] = []

        for c in self.clusters:
            bv = float(c.get("buy_volume", 0.0))
            sv = float(c.get("sell_volume", 0.0))
            if bv <= 0 and sv <= 0:
                continue
            if bv > sv * 1.2:
                supports.append(float(c.get("center", 0.0)))
            elif sv > bv * 1.2:
                resistances.append(float(c.get("center", 0.0)))

        return supports, resistances

    def get_liquidity_zones(self) -> List[Dict[str, Any]]:
        """Retorna zonas de liquidez no formato compatível com LevelRegistry."""
        zones: List[Dict[str, Any]] = []

        for c in self.clusters:
            center = float(c.get("center", 0.0))
            total_vol = float(c.get("total_volume", 0.0))
            imb = float(c.get("imbalance_ratio", 0.0))
            width = float(c.get("width", 0.0))

            # Zona principal (centro do cluster)
            zones.append({
                "kind": "LIQUIDITY_CLUSTER",
                "price": center,
                "width_pct": self.cluster_threshold_pct * 2,
                "score": total_vol * (1.0 + abs(imb)),
                "confluence": [
                    f"LIQ_{int(c.get('trades_count', 0))}trades",
                    f"IMB_{imb:.2f}"
                ],
            })

            # Zonas de borda (high/low do cluster)
            if width > 0:
                high = float(c.get("high", center))
                low = float(c.get("low", center))
                zones.append({
                    "kind": "LIQUIDITY_HIGH",
                    "price": high,
                    "width_pct": self.cluster_threshold_pct,
                    "score": total_vol * 0.8,
                    "confluence": ["LIQ_HIGH", f"VOL_{total_vol:.3f}"],
                })
                zones.append({
                    "kind": "LIQUIDITY_LOW",
                    "price": low,
                    "width_pct": self.cluster_threshold_pct,
                    "score": total_vol * 0.8,
                    "confluence": ["LIQ_LOW", f"VOL_{total_vol:.3f}"],
                })

        return zones

    def get_current_liquidity_profile(self, current_price: float) -> Dict[str, Any]:
        """Retorna perfil de liquidez ao redor do preço atual."""
        try:
            cp = float(current_price)
        except Exception:
            return {
                "status": "invalid_price",
                "current_price": current_price,
            }

        nearby: List[Dict[str, Any]] = []
        total_nearby_volume = 0.0

        for c in self.clusters:
            center = float(c.get("center", 0.0))
            if center <= 0:
                continue
            # janela de 3x o threshold
            if abs(center - cp) / center <= self.cluster_threshold_pct * 3.0:
                nearby.append(c)
                total_nearby_volume += float(c.get("total_volume", 0.0))

        if not nearby:
            return {
                "status": "no_liquidity_nearby",
                "current_price": cp,
                "liquidity_score": 0.0,
                "imbalance_ratio": 0.0,
                "clusters_count": 0,
            }

        total_buy = sum(float(c.get("buy_volume", 0.0)) for c in nearby)
        total_sell = sum(float(c.get("sell_volume", 0.0)) for c in nearby)
        denom = total_buy + total_sell

        return {
            "status": "liquidity_detected",
            "current_price": cp,
            "liquidity_score": total_nearby_volume,
            "imbalance_ratio": (total_buy - total_sell) / denom if denom > 0 else 0.0,
            "clusters_count": len(nearby),
            "avg_cluster_age_ms": int(np.mean([int(c.get("age_ms", 0)) for c in nearby] or [0])),
            "total_volume": total_nearby_volume,
            "buy_volume": total_buy,
            "sell_volume": total_sell,
        }

    def clear_old_data(self, max_age_ms: int = 300000):  # 5 minutos
        """Remove trades muito antigos para manter o heatmap atualizado."""
        if not self.timestamp_levels:
            return

        current_time = _now_ms()
        cutoff_time = current_time - int(max_age_ms)

        # Remove trades antigos (os mais antigos ficam no início das deques)
        while self.timestamp_levels and self.timestamp_levels[0] < cutoff_time:
            self.price_levels.popleft()
            self.volume_levels.popleft()
            self.side_levels.popleft()
            self.timestamp_levels.popleft()

        # Recalcula clusters
        self._update_clusters()

    def get_stats(self) -> Dict[str, Any]:
        """🆕 Retorna estatísticas de validação."""
        return {
            "clusters_count": len(self.clusters),
            "trades_in_buffer": len(self.timestamp_levels),
            "timestamp_corrections": self._timestamp_corrections,
            "invalid_timestamps": self._invalid_timestamps,
            "last_update_time": self.last_update_time,
        }