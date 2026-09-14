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


# Fontes derivadas do MESMO snapshot REST L2 (não são estrutura independente).
# wall+wall (ou wall+cluster/depth do mesmo book) NÃO conta como confluência
# estrutural: a origem física continua sendo uma única fotografia do book.
NON_STRUCTURAL_SOURCES = frozenset({
    "orderbook_bid_wall",
    "orderbook_ask_wall",
    "orderbook_cluster",
    "depth_asymmetry",
})

# Sources de wall observada (para o teste positivo de wall-only).
WALL_SOURCES = frozenset({"orderbook_bid_wall", "orderbook_ask_wall"})


def is_wall_only_zone(zone: object) -> bool:
    """Verdadeiro se a zona contém EXCLUSIVAMENTE walls do snapshot L2.

    wall-only = concentração de liquidez passiva observada (SNAPSHOT_LIQUIDITY),
    não S/R autônomo: não promover a immediate_support/resistance nem sr.s1/r1.
    Aceita tanto o flag novo (liquidity_only) quanto o teste por sources
    (compatibilidade com eventos armazenados antes do flag).
    """
    if not isinstance(zone, dict):
        return False
    if zone.get("liquidity_only") is True:
        return True
    sources = zone.get("sources")
    if isinstance(sources, list) and sources:
        if all(s in WALL_SOURCES for s in sources):
            return True
    return False


def has_structural_confluence(zone: object) -> bool:
    """Verdadeiro se a zona tem ≥1 fonte independente do snapshot L2 atual."""
    if not isinstance(zone, dict):
        return False
    if zone.get("has_structural_confluence") is True:
        return True
    sources = zone.get("sources")
    if isinstance(sources, list) and sources:
        if any(s not in NON_STRUCTURAL_SOURCES for s in sources):
            return True
    return False


def is_non_structural_zone(zone: object) -> bool:
    """Verdadeiro se a zona NÃO tem fonte estrutural independente.

    Generalização de is_wall_only_zone (fases anteriores): cobre wall-only
    (liquidity_only) E heurísticas projetadas puras (projected_only, ex.:
    depth_asymmetry isolado). Zonas sem flags nem sources (fixtures legados)
    retornam False — nunca bloquear por falta de metadados.
    """
    if not isinstance(zone, dict):
        return False
    if zone.get("liquidity_only") is True or zone.get("projected_only") is True:
        return True
    if zone.get("has_structural_confluence") is True:
        return False
    sources = zone.get("sources")
    if isinstance(sources, list) and sources:
        return all(s in NON_STRUCTURAL_SOURCES for s in sources)
    return False


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

        # 1b. Deduplicar sinais pela fonte canônica (rotas duplas = 1 observação)
        signals = self._dedupe_signals(signals, current_price)

        if not signals:
            return self._empty_result()

        # 2. Agrupar sinais próximos em zonas
        zones = self._cluster_signals(signals, current_price)

        # 3. Filtrar zonas com confluência insuficiente
        strong_zones = [z for z in zones if z["source_count"] >= self._min_sources]

        # Se filtrar demais, relaxar critério
        if not strong_zones and zones:
            strong_zones = zones[:5]

        # 4. Separar em buy e sell defense usando side da zona (não apenas posição do preço)
        buy_defense = sorted(
            [z for z in strong_zones if z.get("side") == "buy" or
             ("side" not in z and z["center"] < current_price)],
            key=lambda z: z["strength"],
            reverse=True,
        )[:self._max_zones]

        sell_defense = sorted(
            [z for z in strong_zones if z.get("side") == "sell" or
             ("side" not in z and z["center"] >= current_price)],
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

    @staticmethod
    def _wall_signal_strength(qty: float, threshold: float) -> float:
        """HEURISTIC — proeminência relativa da wall, NÃO força calibrada.

        ratio = qty / threshold mede o excesso do nível sobre a distribuição
        do próprio book (threshold = quantil-90% × multiplicador do detector).
        Escala conservadora 10–30 (cap abaixo do antigo 40/sinal e abaixo de
        vp_vah/vp_val=35): single-wall nunca atinge patamar de confluência
        (composite máximo 30×1.3=39 < antigo 52).
        NÃO é probabilidade, confiança nem institutional strength: nenhuma
        calibração contra outcomes. A evidência auditável viaja em
        wall_ratio/qty_btc/notional_usd — o score é apenas ordinal.
        Alternativas rejeitadas: constante fixa (perde ordenação entre walls),
        notional (escala com preço, incomparável entre regimes), distância ao
        preço (distância ≠ força), fórmula de imbalance (mede outra coisa).
        """
        try:
            ratio = float(qty) / float(threshold) if threshold and threshold > 0 else 1.0
        except (TypeError, ValueError):
            ratio = 1.0
        if ratio < 1.0:
            ratio = 1.0
        return min(30.0, 10.0 + 10.0 * min(ratio - 1.0, 2.0))

    @staticmethod
    def _wall_signal(wall: object, side: str, source: str) -> Optional[dict]:
        """Constrói um sinal de defesa a partir de UMA wall observada.

        Retorna None se a wall for inválida (nunca projeta preço).
        """
        if not isinstance(wall, dict):
            return None
        try:
            price = float(wall.get("price", 0) or 0)
            qty = float(wall.get("qty", 0) or 0)
        except (TypeError, ValueError):
            return None
        if price <= 0 or qty <= 0:
            return None
        threshold = wall.get("limit_threshold", 0)
        try:
            notional = float(price * qty)
        except (TypeError, ValueError):
            notional = 0.0
        try:
            wall_ratio = float(qty) / float(threshold) if threshold and float(threshold) > 0 else None
        except (TypeError, ValueError):
            wall_ratio = None
        return {
            "price": round(price, 2),
            "source": source,
            "strength": DefenseZoneDetector._wall_signal_strength(qty, threshold),
            "side": side,
            # Provenance da observação (não vai para o payload IA em full,
            # mas ancora a zona — ver _cluster_signals).
            "observed": True,
            "projected": False,
            "basis": "rest_l2_snapshot",
            "snapshot_scope": "top50",
            "snapshot_only": True,
            "persistence_confirmed": False,
            "wall_price": float(price),
            "wall_qty_btc": float(qty),
            "wall_notional_usd": round(notional, 2),
            "wall_threshold_qty": float(threshold) if threshold else None,
            "wall_ratio": round(wall_ratio, 4) if wall_ratio is not None else None,
        }

    def _extract_orderbook_defense(self, ob_data: dict, current_price: float) -> list:
        """Extrai sinais de defesa EXCLUSIVAMENTE de walls observadas no snapshot.

        Contrato de wall observada (fix provenance 2026-09):
          - center = wall.price (tick exato do book, sem projeção ±0.1%).
          - source orderbook_bid_wall / orderbook_ask_wall SÓ é emitido aqui,
            com observed=True / projected=False.
          - Imbalance global NÃO gera nível: não existe mais projeção
            current_price*0.999 / *1.001 (removida — era mislabeling).
          - Toda wall é SNAPSHOT_ONLY (fotografia REST única, sem persistência).
        """
        signals = []

        walls = (ob_data.get("walls") or {}) if isinstance(ob_data, dict) else {}
        bid_walls = walls.get("bids", []) if isinstance(walls, dict) else []
        ask_walls = walls.get("asks", []) if isinstance(walls, dict) else []

        for wall in bid_walls if isinstance(bid_walls, list) else []:
            sig = self._wall_signal(wall, "buy", "orderbook_bid_wall")
            if sig is not None:
                signals.append(sig)

        for wall in ask_walls if isinstance(ask_walls, list) else []:
            sig = self._wall_signal(wall, "sell", "orderbook_ask_wall")
            if sig is not None:
                signals.append(sig)

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

        # Depth metrics — PROJECTED_HEURISTIC (não é wall observada):
        # assimetria global do book projetada em current_price×0.998/1.002.
        # Provenance explícita; gating de S/R decide via projected_only.
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
                    "observed": False,
                    "projected": True,
                    "basis": "depth_asymmetry",
                    "snapshot_only": True,
                    "persistence_confirmed": False,
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

        pivot_keys = ("pivot", "pp", "r1", "r2", "r3", "s1", "s2", "s3")
        for method_name, levels in pivot_data.items():
            if not isinstance(levels, dict):
                continue
            for level_name, price in levels.items():
                if not isinstance(price, (int, float)) or price <= 0:
                    continue
                # Ignorar chaves auxiliares (high/low/close/quality/etc.)
                lname = str(level_name).lower()
                if lname not in pivot_keys:
                    continue
                # S levels = suporte, PP/pivot = neutro (convenção buy),
                # R levels = resistência
                if lname.startswith("s") or lname in ("pivot", "pp"):
                    side = "buy"
                else:
                    side = "sell"
                weight = 30 if lname in ("pivot", "pp") else 20
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
            # Prefixo ema_ preservado para o formato legado ("1d", "4h");
            # a rota real já traz o prefixo ("ema_21_1h") — sem duplicação.
            source = name if str(name).startswith("ema") else f"ema_{name}"
            signals.append({
                "price": price,
                "source": source,
                "strength": w,
                "side": side,
            })
        return signals

    @staticmethod
    def _canonical_source(source: str) -> str:
        """Identidade canônica de uma fonte: remove o prefixo de rota
        (sr_level_*) e normaliza aliases que representam a MESMA observação
        econômica (ex: vp_val vs val_daily vs sr_level_val_daily)."""
        s = source
        if s.startswith("sr_level_"):
            s = s[len("sr_level_"):]
        return {
            "poc_daily": "vp_poc",
            "vah_daily": "vp_vah",
            "val_daily": "vp_val",
            "hvn_daily": "vp_hvn",
        }.get(s, s)

    def _dedupe_signals(self, signals: list, current_price: float) -> list:
        """Remove sinais que representam a MESMA observação econômica.

        Identidade = (fonte canônica, tick do preço):
          - Renomeia TODOS os sinais para a fonte canônica, para que rotas
            duplas de ingestão (direta + sr_level_*) contem como 1 fonte na
            confluência do clustering.
          - Colapsa apenas sinais da mesma fonte canônica com o preço no
            MESMO tick (round 2 casas) — ex: vp_poc + sr_level_poc_daily
            derivados do mesmo valor.
          - NÃO colapsa por proximidade: bins distintos do produtor (o
            historical_profiler emite um node por bin de $1 — ver
            historical_profiler._compute_volume_profile) permanecem
            observações separadas. Agrupamento por proximidade é
            responsabilidade exclusiva do _cluster_signals.
        """
        if not signals:
            return signals
        buckets: dict = {}
        for sig in signals:
            key = self._canonical_source(sig.get("source", "unknown"))
            buckets.setdefault(key, []).append(sig)
        deduped = []
        for key, bucket in buckets.items():
            ticks: dict = {}
            for sig in bucket:
                ticks.setdefault(round(sig["price"], 2), []).append(sig)
            for group in ticks.values():
                best = dict(max(group, key=lambda s: (s.get("strength", 0), s["price"])))
                best["source"] = key
                deduped.append(best)
        return deduped

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

            # Lados de walls já presentes no grupo: walls observadas de lados
            # opostos (bid×ask, em geral $0.1–$1 apart no toque) NUNCA se fundem
            # — a média destruiria a distinção buy/sell e criaria um center que
            # não é nenhuma wall observada (§5). Confluência wall + sinal
            # não-wall (VP/pivot/EMA/absorção) continua permitida.
            group_wall_sides = set()
            if sig.get("wall_price"):
                group_wall_sides.add(sig.get("side"))

            for j in range(i + 1, len(signals_sorted)):
                if j in used:
                    continue
                if abs(signals_sorted[j]["price"] - sig["price"]) <= tolerance:
                    cand = signals_sorted[j]
                    if (cand.get("wall_price") and group_wall_sides
                            and cand.get("side") not in group_wall_sides):
                        continue
                    group.append(cand)
                    used.add(j)
                    if cand.get("wall_price"):
                        group_wall_sides.add(cand.get("side"))
                else:
                    break

            # Construir zona
            prices = [g["price"] for g in group]
            sources = list(set(g["source"] for g in group))
            total_strength = sum(g["strength"] for g in group)
            avg_strength = total_strength / len(group)

            # Âncora observada (§5/§8): se o grupo contém wall(s) real(is),
            # o center é a média dos preços observados (== wall.price no caso
            # single, o caso forense dominante) — nunca projeção. O range
            # ao redor permanece faixa derivada.
            wall_members = [g for g in group if g.get("wall_price")]
            if wall_members:
                wall_prices = [float(g["wall_price"]) for g in wall_members]
                center = sum(wall_prices) / len(wall_prices)
            else:
                center = sum(prices) / len(prices)

            # Classificação estrutural (toda zona, não só walls): só fontes
            # fora do snapshot L2 atual contam como independência estrutural
            # (wall+wall, wall+depth, depth+cluster NÃO são confluência).
            structural = sorted({s for s in sources if s not in NON_STRUCTURAL_SOURCES})
            has_projected = any(g.get("projected") is True for g in group)

            # Side dominante (empate → posição em relação ao preço atual)
            buy_count = sum(1 for g in group if g["side"] == "buy")
            sell_count = sum(1 for g in group if g["side"] == "sell")
            if buy_count > sell_count:
                dominant_side = "buy"
            elif sell_count > buy_count:
                dominant_side = "sell"
            else:
                dominant_side = "buy" if center < current_price else "sell"

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
                "structural_sources": structural,
                "has_structural_confluence": bool(structural),
                "has_projected_component": bool(has_projected),
            })
            if wall_members:
                # Provenance da zona: center observado, faixa derivada.
                # Âncora = preço da wall porque é o único tick point-in-time
                # observado do grupo; VP/pivot/EMA são bandas históricas ou
                # derivadas e permanecem cobertas pelo range + structural_sources.
                nearest_wall = min(
                    wall_members, key=lambda g: abs(float(g["wall_price"]) - center)
                )
                zones[-1].update({
                    "observed": True,
                    "projected": False,
                    "basis": "rest_l2_snapshot",
                    "snapshot_scope": "top50",
                    "snapshot_only": True,
                    "persistence_confirmed": False,
                    "center_origin": "observed_wall_price",
                    "wall_prices": sorted(set(float(g["wall_price"]) for g in wall_members)),
                    "wall_qty_btc": nearest_wall.get("wall_qty_btc"),
                    "wall_notional_usd": nearest_wall.get("wall_notional_usd"),
                    "wall_threshold_qty": nearest_wall.get("wall_threshold_qty"),
                    "wall_ratio": nearest_wall.get("wall_ratio"),
                    # Confluência estrutural: só fontes fora do snapshot L2
                    # atual contam (wall+wall NÃO é independência estrutural).
                    "liquidity_only": not structural,
                })
            elif not structural:
                # Zona puramente heurística (ex.: depth_asymmetry isolado):
                # sem tick observado e sem fonte estrutural — evidência
                # projetada do snapshot, nunca S/R autônomo.
                zones[-1].update({
                    "observed": False,
                    "projected": True,
                    "basis": "projected_heuristic",
                    "snapshot_only": True,
                    "persistence_confirmed": False,
                    "projected_only": True,
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