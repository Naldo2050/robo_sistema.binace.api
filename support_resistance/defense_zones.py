"""
Defense Zones — Zonas de defesa institucional.

Identifica zonas de preço onde há concentração de defesa
(walls persistentes no order book + absorção histórica + HVN do Volume Profile).

Zonas de defesa são os níveis mais confiáveis para stops e entries.
Institucionais "protegem" suas posições nessas zonas.

Uso:
    detector = DefenseZoneDetector()
    zones = detector.detect(
        current_price=64892,
        orderbook_data={...},
        vp_data={...},
        sr_levels=[...],
        absorption_events=[...],
    )
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DefenseZoneDetector:
    """
    Detecta zonas de defesa institucional combinando múltiplas fontes.
    
    Uma defense zone é formada quando MÚLTIPLAS fontes concordam:
      - Order book mostra walls (grandes ordens limit)
      - Volume Profile mostra HVN (muitas transações históricas)
      - Absorção detectada (preço testou e não rompeu)
      - S/R com alto score de confluência
    """

    def __init__(
        self,
        zone_width_pct: float = 0.15,
        min_sources_for_zone: int = 2,
        max_zones_per_side: int = 5,
    ):
        """
        Args:
            zone_width_pct: Largura da zona em % do preço (0.15% default).
            min_sources_for_zone: Mínimo de fontes confirmando para criar zona.
            max_zones_per_side: Máximo de zonas por lado (buy/sell).
        """
        self._zone_width_pct = zone_width_pct
        self._min_sources = min_sources_for_zone
        self._max_zones = max_zones_per_side

    def detect(
        self,
        current_price: float,
        orderbook_data: Optional[dict] = None,
        vp_data: Optional[dict] = None,
        sr_levels: Optional[list] = None,
        absorption_events: Optional[list] = None,
        pivot_data: Optional[dict] = None,
        ema_values: Optional[dict] = None,
    ) -> dict:
        """
        Detecta zonas de defesa combinando todas as fontes disponíveis.
        
        Args:
            current_price: Preço atual.
            orderbook_data: Dados do order book.
                Espera: {"bid_depth_usd": x, "ask_depth_usd": x, "imbalance": x,
                         "depth_metrics": {"bid_liquidity_top5": x, ...}}
                Ou clusters: [{"center": x, "total_volume": x, "imbalance": x}, ...]
            vp_data: Volume Profile.
                Espera: {"poc": x, "vah": x, "val": x, "hvns": [...]}
            sr_levels: Lista de S/R já pontuados.
                Espera: [{"price": x, "strength": x, "type": "support", ...}]
            absorption_events: Histórico de eventos de absorção.
                Espera: [{"price": x, "type": "buy"/"sell", "strength": x}, ...]
            pivot_data: Pivot Points.
            ema_values: EMAs por timeframe.
            
        Returns:
            Dict com buy_defense e sell_defense zones.
        """
        if current_price <= 0:
            return self._empty_result()

        # 1. Coletar sinais de defesa de cada fonte
        signals = []

        # --- Order Book signals ---
        if orderbook_data:
            signals.extend(self._extract_orderbook_defense(orderbook_data, current_price))

        # --- Volume Profile signals ---
        if vp_data:
            signals.extend(self._extract_vp_defense(vp_data, current_price))

        # --- S/R Level signals ---
        if sr_levels:
            signals.extend(self._extract_sr_defense(sr_levels, current_price))

        # --- Absorption signals ---
        if absorption_events:
            signals.extend(self._extract_absorption_defense(absorption_events, current_price))

        # --- Pivot signals ---
        if pivot_data:
            signals.extend(self._extract_pivot_defense(pivot_data, current_price))

        # --- EMA signals ---
        if ema_values:
            signals.extend(self._extract_ema_defense(ema_values, current_price))

        if not signals:
            return self._empty_result()

        # 2. Agrupar sinais próximos em zonas
        zones = self._cluster_signals(signals, current_price)

        # 3. Filtrar zonas com confluência insuficiente
        strong_zones = [z for z in zones if z["source_count"] >= self._min_sources]

        # Se filtrar demais, relaxar critério
        if not strong_zones and zones:
            strong_zones = zones[:5]

        # 4. Separar em buy e sell defense
        buy_defense = sorted(
            [z for z in strong_zones if z["center"] < current_price],
            key=lambda z: z["strength"],
            reverse=True,
        )[:self._max_zones]

        sell_defense = sorted(
            [z for z in strong_zones if z["center"] >= current_price],
            key=lambda z: z["strength"],
            reverse=True,
        )[:self._max_zones]

        # 5. Adicionar distância ao preço
        for zone in buy_defense + sell_defense:
            zone["distance_from_price"] = round(
                abs(zone["center"] - current_price), 2
            )
            zone["distance_pct"] = round(
                abs(zone["center"] - current_price) / current_price * 100, 4
            )

        return {
            "buy_defense": buy_defense,
            "sell_defense": sell_defense,
            "total_zones": len(buy_defense) + len(sell_defense),
            "strongest_buy": buy_defense[0] if buy_defense else None,
            "strongest_sell": sell_defense[0] if sell_defense else None,
            "defense_asymmetry": self._calc_asymmetry(buy_defense, sell_defense),
            "status": "success",
        }

    def _extract_orderbook_defense(self, ob_data: dict, current_price: float) -> list:
        """Extrai sinais de defesa do order book."""
        signals = []

        # Imbalance geral: se bid > ask, há defesa compradora
        bid_depth = ob_data.get("bid_depth_usd", 0)
        ask_depth = ob_data.get("ask_depth_usd", 0)
        imbalance = ob_data.get("imbalance", 0)

        # Bid wall = defesa compradora
        if bid_depth > 0 and (imbalance > 0.05 or bid_depth > ask_depth * 1.1):
            # Estimar zona de defesa compradora (próximo ao preço)
            signals.append({
                "price": round(current_price * 0.999, 2),  # ~0.1% abaixo
                "source": "orderbook_bid_wall",
                "strength": min(40, abs(imbalance) * 200),
                "side": "buy",
            })

        # Ask wall = defesa vendedora
        if ask_depth > 0 and (imbalance < -0.05 or ask_depth > bid_depth * 1.1):
            signals.append({
                "price": round(current_price * 1.001, 2),  # ~0.1% acima
                "source": "orderbook_ask_wall",
                "strength": min(40, abs(imbalance) * 200),
                "side": "sell",
            })

        # Clusters de liquidez se disponíveis
        clusters = ob_data.get("clusters", [])
        if isinstance(clusters, list):
            for cluster in clusters:
                if isinstance(cluster, dict):
                    center = cluster.get("center", 0)
                    vol = cluster.get("total_volume", 0)
                    c_imbalance = cluster.get("imbalance_ratio", 0)
                    if center > 0:
                        side = "buy" if center < current_price else "sell"
                        signals.append({
                            "price": center,
                            "source": "orderbook_cluster",
                            "strength": min(35, vol * 3),
                            "side": side,
                        })

        # Depth metrics
        depth = ob_data.get("depth_metrics", {})
        if isinstance(depth, dict):
            depth_imb = depth.get("depth_imbalance", 0)
            if abs(depth_imb) > 0.1:
                side = "buy" if depth_imb > 0 else "sell"
                signals.append({
                    "price": current_price * (0.998 if side == "buy" else 1.002),
                    "source": "depth_asymmetry",
                    "strength": min(30, abs(depth_imb) * 100),
                    "side": side,
                })

        return signals

    def _extract_vp_defense(self, vp_data: dict, current_price: float) -> list:
        """Extrai sinais de defesa do Volume Profile."""
        signals = []

        poc = vp_data.get("poc", 0) or vp_data.get("poc_price", 0)
        vah = vp_data.get("vah", 0)
        val = vp_data.get("val", 0)
        hvns = vp_data.get("hvns", []) or []

        if poc > 0:
            side = "buy" if poc < current_price else "sell"
            signals.append({
                "price": poc,
                "source": "vp_poc",
                "strength": 45,  # POC é sempre forte
                "side": side,
            })

        if vah > 0:
            signals.append({
                "price": vah,
                "source": "vp_vah",
                "strength": 35,
                "side": "sell",  # VAH tende a ser resistência
            })

        if val > 0:
            signals.append({
                "price": val,
                "source": "vp_val",
                "strength": 35,
                "side": "buy",  # VAL tende a ser suporte
            })

        for hvn in hvns:
            if hvn and hvn > 0:
                side = "buy" if hvn < current_price else "sell"
                signals.append({
                    "price": hvn,
                    "source": "vp_hvn",
                    "strength": 25,
                    "side": side,
                })

        return signals

    def _extract_sr_defense(self, sr_levels: list, current_price: float) -> list:
        """Extrai sinais de defesa de S/R scoring."""
        signals = []
        for level in sr_levels:
            if not isinstance(level, dict):
                continue
            price = level.get("price", 0)
            strength = level.get("strength", 0)
            if price > 0 and strength > 30:  # Só S/R fortes
                side = "buy" if price < current_price else "sell"
                signals.append({
                    "price": price,
                    "source": f"sr_level_{level.get('primary_source', 'unknown')}",
                    "strength": min(50, strength * 0.5),
                    "side": side,
                })
        return signals

    def _extract_absorption_defense(self, events: list, current_price: float) -> list:
        """Extrai sinais de defesa de eventos de absorção históricos."""
        signals = []
        for event in events:
            if not isinstance(event, dict):
                continue
            price = event.get("price", 0)
            abs_type = event.get("type", "").lower()
            strength = event.get("strength", 0) or event.get("index", 0)

            if price <= 0:
                continue

            if "compra" in abs_type or "buy" in abs_type:
                signals.append({
                    "price": price,
                    "source": "absorption_buy",
                    "strength": min(40, float(strength) * 40),
                    "side": "buy",
                })
            elif "venda" in abs_type or "sell" in abs_type:
                signals.append({
                    "price": price,
                    "source": "absorption_sell",
                    "strength": min(40, float(strength) * 40),
                    "side": "sell",
                })

        return signals

    def _extract_pivot_defense(self, pivot_data: dict, current_price: float) -> list:
        """Extrai sinais de defesa dos Pivot Points."""
        signals = []
        if not isinstance(pivot_data, dict):
            return signals

        for method_name, levels in pivot_data.items():
            if not isinstance(levels, dict):
                continue
            for level_name, price in levels.items():
                if not isinstance(price, (int, float)) or price <= 0:
                    continue
                # S levels = suporte, R levels = resistência
                if level_name.startswith("S") or level_name == "PP":
                    side = "buy"
                else:
                    side = "sell"
                weight = 30 if level_name == "PP" else 20
                signals.append({
                    "price": price,
                    "source": f"pivot_{method_name}_{level_name}",
                    "strength": weight,
                    "side": side,
                })
        return signals

    def _extract_ema_defense(self, ema_values: dict, current_price: float) -> list:
        """Extrai sinais de defesa das EMAs."""
        signals = []
        if not isinstance(ema_values, dict):
            return signals

        weights = {"1d": 30, "4h": 25, "1h": 15, "15m": 10}

        for name, price in ema_values.items():
            if not isinstance(price, (int, float)) or price <= 0:
                continue
            # Determinar peso pelo timeframe
            w = 15
            for tf, tw in weights.items():
                if tf in name:
                    w = tw
                    break
            side = "buy" if price < current_price else "sell"
            signals.append({
                "price": price,
                "source": f"ema_{name}",
                "strength": w,
                "side": side,
            })
        return signals

    def _cluster_signals(self, signals: list, current_price: float) -> list:
        """Agrupa sinais próximos em zonas de defesa."""
        if not signals:
            return []

        tolerance = current_price * (self._zone_width_pct / 100)
        signals_sorted = sorted(signals, key=lambda s: s["price"])

        zones = []
        used = set()

        for i, sig in enumerate(signals_sorted):
            if i in used:
                continue

            group = [sig]
            used.add(i)

            for j in range(i + 1, len(signals_sorted)):
                if j in used:
                    continue
                if abs(signals_sorted[j]["price"] - sig["price"]) <= tolerance:
                    group.append(signals_sorted[j])
                    used.add(j)
                else:
                    break

            # Construir zona
            prices = [g["price"] for g in group]
            center = sum(prices) / len(prices)
            sources = list(set(g["source"] for g in group))
            total_strength = sum(g["strength"] for g in group)
            avg_strength = total_strength / len(group)

            # Side dominante
            buy_count = sum(1 for g in group if g["side"] == "buy")
            sell_count = sum(1 for g in group if g["side"] == "sell")
            dominant_side = "buy" if buy_count >= sell_count else "sell"

            # Score composto: força média × confluência
            composite_score = min(100, avg_strength * (1 + len(sources) * 0.3))

            zones.append({
                "center": round(center, 2),
                "range_low": round(min(prices) - tolerance * 0.5, 2),
                "range_high": round(max(prices) + tolerance * 0.5, 2),
                "strength": round(composite_score),
                "side": dominant_side,
                "sources": sources,
                "source_count": len(sources),
                "signals_in_zone": len(group),
                "type": "confluence" if len(sources) >= 3 else "cluster" if len(sources) >= 2 else "single",
            })

        zones.sort(key=lambda z: z["strength"], reverse=True)
        return zones

    def _calc_asymmetry(self, buy_zones: list, sell_zones: list) -> dict:
        """Calcula assimetria entre defesa compradora e vendedora."""
        buy_strength = sum(z["strength"] for z in buy_zones) if buy_zones else 0
        sell_strength = sum(z["strength"] for z in sell_zones) if sell_zones else 0
        total = buy_strength + sell_strength

        if total == 0:
            return {"ratio": 1.0, "bias": "neutral", "description": "No defense zones detected"}

        ratio = buy_strength / sell_strength if sell_strength > 0 else 99.0

        if ratio > 1.5:
            bias = "strong_buy_defense"
            desc = "Significantly more buy defense - supports likely to hold"
        elif ratio > 1.1:
            bias = "slight_buy_defense"
            desc = "Slightly more buy defense"
        elif ratio > 0.9:
            bias = "neutral"
            desc = "Balanced defense on both sides"
        elif ratio > 0.67:
            bias = "slight_sell_defense"
            desc = "Slightly more sell defense"
        else:
            bias = "strong_sell_defense"
            desc = "Significantly more sell defense - resistances likely to hold"

        return {
            "ratio": round(ratio, 4),
            "bias": bias,
            "description": desc,
            "buy_total_strength": round(buy_strength),
            "sell_total_strength": round(sell_strength),
        }

    def _empty_result(self) -> dict:
        return {
            "buy_defense": [],
            "sell_defense": [],
            "total_zones": 0,
            "strongest_buy": None,
            "strongest_sell": None,
            "defense_asymmetry": {"ratio": 1.0, "bias": "neutral", "description": "No data"},
            "status": "no_data",
        }