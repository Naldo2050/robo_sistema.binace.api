"""
S/R Strength Scoring — Pontuação de força de Suportes e Resistências.

Pontua cada nível de S/R de 0-100 baseado em:
  1. Toques históricos (quantas vezes o preço testou o nível)      → 0-25 pts
  2. Volume acumulado no nível (do Volume Profile)                  → 0-25 pts
  3. Confluência com outros indicadores (pivots, EMA, round numbers)→ 0-30 pts
  4. Recência (mais recente = mais forte)                          → 0-20 pts

Uso:
    scorer = SRStrengthScorer()
    levels = scorer.score_levels(
        current_price=64892,
        vp_data={"poc": 64888, "vah": 66055, "val": 64683, "hvns": [...], "lvns": [...]},
        pivot_data={"classic": {"PP": 64850, "R1": 65100, ...}},
        ema_values={"ema_21_1h": 65418, "ema_21_4h": 66695},
        recent_candles=df_candles,
    )
    print(levels)  # Lista de níveis com strength score
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SRStrengthScorer:
    """
    Calcula força de níveis de suporte e resistência.
    
    Combina múltiplas fontes de dados para produzir um score
    único por nível. Usado para priorizar quais S/R são mais
    confiáveis para stops, entries e targets.
    """

    def __init__(self, touch_tolerance_pct: float = 0.15, round_number_interval: int = 1000):
        """
        Args:
            touch_tolerance_pct: % de tolerância para considerar um "toque" no nível.
                                 0.15 = preço a 0.15% do nível conta como toque.
            round_number_interval: Intervalo para números redondos (1000 = 64000, 65000...).
        """
        self._touch_tolerance_pct = touch_tolerance_pct
        self._round_number_interval = round_number_interval

    def score_levels(
        self,
        current_price: float,
        vp_data: Optional[dict] = None,
        pivot_data: Optional[dict] = None,
        ema_values: Optional[dict] = None,
        recent_candles=None,
        weekly_vp: Optional[dict] = None,
        monthly_vp: Optional[dict] = None,
    ) -> dict:
        """
        Pontua todos os níveis de S/R identificáveis.
        
        Args:
            current_price: Preço atual do ativo.
            vp_data: Volume Profile diário.
                     Espera: {"poc": float, "vah": float, "val": float,
                              "hvns": [float], "lvns": [float]}
            pivot_data: Pivot Points calculados.
                        Espera: {"classic": {"PP": x, "R1": x, ...}, "fibonacci": {...}, ...}
            ema_values: EMAs de diferentes timeframes.
                        Espera: {"ema_21_15m": x, "ema_21_1h": x, "ema_21_4h": x, "ema_21_1d": x}
            recent_candles: DataFrame ou lista de candles recentes para contagem de toques.
                           Espera colunas/chaves: high, low, close
            weekly_vp: Volume Profile semanal (para confluência multi-TF).
                       Espera: {"poc": float, "vah": float, "val": float}
            monthly_vp: Volume Profile mensal.
                        Espera: {"poc": float, "vah": float, "val": float}
                        
        Returns:
            Dict com:
              - levels: Lista de níveis pontuados [{price, type, source, strength, ...}]
              - supports: Top suportes ordenados por strength
              - resistances: Top resistências ordenados por strength
              - nearest_support: Suporte mais forte abaixo do preço
              - nearest_resistance: Resistência mais forte acima do preço
        """
        if current_price <= 0:
            return {"levels": [], "supports": [], "resistances": [], "status": "invalid_price"}

        # 1. Coletar todos os candidatos a S/R
        candidates = self._collect_candidates(
            current_price, vp_data, pivot_data, ema_values, weekly_vp, monthly_vp
        )

        if not candidates:
            return {"levels": [], "supports": [], "resistances": [], "status": "no_candidates"}

        # 2. Mesclar candidatos próximos (evitar duplicatas)
        merged = self._merge_nearby_levels(candidates, current_price)

        # 3. Pontuar cada nível
        scored = []
        for level in merged:
            score = self._calculate_score(
                level, current_price, recent_candles
            )
            level["strength"] = score
            scored.append(level)

        # 4. Ordenar por strength
        scored.sort(key=lambda x: x["strength"], reverse=True)

        # 5. Classificar como suporte ou resistência
        supports = [l for l in scored if l["price"] < current_price]
        resistances = [l for l in scored if l["price"] >= current_price]

        # Nearest strong support/resistance
        nearest_support = supports[0] if supports else None
        nearest_resistance = resistances[0] if resistances else None

        return {
            "levels": scored[:20],  # Top 20
            "supports": supports[:10],
            "resistances": resistances[:10],
            "nearest_support": nearest_support,
            "nearest_resistance": nearest_resistance,
            "total_levels_found": len(scored),
            "status": "success",
        }

    def _collect_candidates(
        self, current_price, vp_data, pivot_data, ema_values, weekly_vp, monthly_vp
    ) -> list:
        """Coleta todos os candidatos a S/R de múltiplas fontes."""
        candidates = []
        has_other_sources = False

        # --- Volume Profile Diário ---
        if vp_data and isinstance(vp_data, dict):
            has_other_sources = True
            poc = vp_data.get("poc", 0) or vp_data.get("poc_price", 0)
            vah = vp_data.get("vah", 0)
            val = vp_data.get("val", 0)

            if poc > 0:
                candidates.append({"price": poc, "source": "poc_daily", "source_weight": 1.5})
            if vah > 0:
                candidates.append({"price": vah, "source": "vah_daily", "source_weight": 1.2})
            if val > 0:
                candidates.append({"price": val, "source": "val_daily", "source_weight": 1.2})

            for hvn in (vp_data.get("hvns", []) or []):
                if hvn and hvn > 0:
                    candidates.append({"price": hvn, "source": "hvn_daily", "source_weight": 0.8})

        # --- Volume Profile Semanal ---
        if weekly_vp and isinstance(weekly_vp, dict):
            has_other_sources = True
            for key, src in [("poc", "poc_weekly"), ("vah", "vah_weekly"), ("val", "val_weekly")]:
                val_w = weekly_vp.get(key, 0)
                if val_w and val_w > 0:
                    candidates.append({"price": val_w, "source": src, "source_weight": 1.8})

        # --- Volume Profile Mensal ---
        if monthly_vp and isinstance(monthly_vp, dict):
            has_other_sources = True
            for key, src in [("poc", "poc_monthly"), ("vah", "vah_monthly"), ("val", "val_monthly")]:
                val_m = monthly_vp.get(key, 0)
                if val_m and val_m > 0:
                    candidates.append({"price": val_m, "source": src, "source_weight": 2.0})

        # --- Pivot Points ---
        if pivot_data and isinstance(pivot_data, dict):
            has_other_sources = True
            for method_name, method_levels in pivot_data.items():
                if not isinstance(method_levels, dict):
                    continue
                weight = 1.3 if method_name == "classic" else 1.0
                for level_name, level_price in method_levels.items():
                    if isinstance(level_price, (int, float)) and level_price > 0:
                        candidates.append({
                            "price": level_price,
                            "source": f"pivot_{method_name}_{level_name}",
                            "source_weight": weight,
                        })

        # --- EMAs ---
        if ema_values and isinstance(ema_values, dict):
            has_other_sources = True
            ema_weights = {
                "ema_21_15m": 0.5,
                "ema_21_1h": 0.8,
                "ema_21_4h": 1.2,
                "ema_21_1d": 1.5,
                "mme_21": 1.0,  # nome alternativo
            }
            for ema_name, ema_price in ema_values.items():
                if isinstance(ema_price, (int, float)) and ema_price > 0:
                    weight = ema_weights.get(ema_name, 0.8)
                    candidates.append({
                        "price": ema_price,
                        "source": ema_name,
                        "source_weight": weight,
                    })

        # --- Números Redondos ---
        interval = self._round_number_interval
        if current_price > 0 and interval > 0 and has_other_sources:
            # 3 acima e 3 abaixo
            base = int(current_price / interval) * interval
            for offset in range(-3, 4):
                round_price = base + (offset * interval)
                if round_price > 0:
                    candidates.append({
                        "price": round_price,
                        "source": "round_number",
                        "source_weight": 0.6,
                    })

        return candidates

    def _merge_nearby_levels(self, candidates: list, current_price: float) -> list:
        """
        Mescla candidatos que estão muito próximos (dentro de tolerance_pct).
        Mantém o com maior source_weight e acumula confluences.
        """
        if not candidates:
            return []

        tolerance = current_price * (self._touch_tolerance_pct / 100)
        candidates_sorted = sorted(candidates, key=lambda c: c["price"])

        merged = []
        used = set()

        for i, candidate in enumerate(candidates_sorted):
            if i in used:
                continue

            group = [candidate]
            used.add(i)

            for j in range(i + 1, len(candidates_sorted)):
                if j in used:
                    continue
                if abs(candidates_sorted[j]["price"] - candidate["price"]) <= tolerance:
                    group.append(candidates_sorted[j])
                    used.add(j)
                else:
                    break  # Sorted, so no more nearby

            # Mesclar grupo
            best = max(group, key=lambda g: g["source_weight"])
            confluences = list(set(g["source"] for g in group))
            avg_price = sum(g["price"] for g in group) / len(group)

            merged.append({
                "price": round(avg_price, 2),
                "primary_source": best["source"],
                "confluences": confluences,
                "confluence_count": len(confluences),
                "max_source_weight": best["source_weight"],
                "sum_source_weight": sum(g["source_weight"] for g in group),
            })

        return merged

    def _calculate_score(self, level: dict, current_price: float, recent_candles=None) -> int:
        """
        Calcula score final de 0-100 para um nível.
        
        Componentes:
          1. Toques históricos  → 0-25 pontos
          2. Peso da fonte      → 0-25 pontos
          3. Confluência        → 0-30 pontos
          4. Proximidade        → 0-20 pontos
        """
        score = 0
        level_price = level["price"]

        # 1. TOQUES HISTÓRICOS (0-25 pontos)
        if recent_candles is not None:
            touches = self._count_touches(level_price, recent_candles)
            score += min(25, touches * 6)  # 4+ toques = 24-25 pts
            level["touches"] = touches
        else:
            # Sem dados de candles, dar score parcial baseado em confluência
            score += 8
            level["touches"] = None

        # 2. PESO DA FONTE (0-25 pontos)
        # source_weight vai de 0.5 (fraco) a 2.0 (forte)
        source_weight = level.get("sum_source_weight", level.get("max_source_weight", 1.0))
        weight_score = min(25, source_weight * 8)
        score += weight_score

        # 3. CONFLUÊNCIA (0-30 pontos)
        confluence_count = level.get("confluence_count", 1)
        # 1 confluência = 5pts, 2 = 12pts, 3 = 20pts, 4+ = 25-30pts
        if confluence_count >= 5:
            conf_score = 30
        elif confluence_count >= 4:
            conf_score = 25
        elif confluence_count >= 3:
            conf_score = 20
        elif confluence_count >= 2:
            conf_score = 12
        else:
            conf_score = 5
        score += conf_score

        # 4. PROXIMIDADE ao preço atual (0-20 pontos)
        # Mais próximo = mais relevante imediatamente
        if current_price > 0 and level_price > 0:
            distance_pct = abs(level_price - current_price) / current_price * 100
            proximity_score = max(0, 20 - (distance_pct * 3))
            score += proximity_score
            level["distance_pct"] = round(distance_pct, 4)
        else:
            level["distance_pct"] = None

        # Tipo: suporte ou resistência
        if level_price < current_price:
            level["type"] = "support"
        elif level_price > current_price:
            level["type"] = "resistance"
        else:
            level["type"] = "at_price"

        return min(round(score), 100)

    def _count_touches(self, level_price: float, candles, lookback: int = 100) -> int:
        """
        Conta quantas vezes o preço tocou um nível nos últimos N candles.
        
        Um "toque" = o high ou low do candle está dentro de tolerance_pct do nível.
        """
        tolerance = level_price * (self._touch_tolerance_pct / 100)
        touches = 0

        try:
            # Se é DataFrame pandas
            if hasattr(candles, 'iterrows'):
                data = candles.tail(lookback)
                for _, row in data.iterrows():
                    high = float(row.get("high", row.get("h", 0)))
                    low = float(row.get("low", row.get("l", 0)))
                    if high > 0 and low > 0:
                        if abs(high - level_price) <= tolerance or abs(low - level_price) <= tolerance:
                            touches += 1
                        elif low <= level_price <= high:
                            touches += 1

            # Se é lista de dicts
            elif isinstance(candles, list):
                data = candles[-lookback:]
                for candle in data:
                    if isinstance(candle, dict):
                        high = float(candle.get("high", candle.get("h", 0)))
                        low = float(candle.get("low", candle.get("l", 0)))
                        if high > 0 and low > 0:
                            if abs(high - level_price) <= tolerance or abs(low - level_price) <= tolerance:
                                touches += 1
                            elif low <= level_price <= high:
                                touches += 1

        except Exception as e:
            logger.debug(f"Touch counting error: {e}")

        return touches

    def quick_score(
        self,
        level_price: float,
        current_price: float,
        source: str = "unknown",
        confluence_count: int = 1,
    ) -> int:
        """
        Pontuação rápida sem dados históricos completos.
        Útil para scoring em tempo real de níveis individuais.
        """
        score = 0

        # Peso base por tipo de fonte
        source_weights = {
            "poc_daily": 20, "poc_weekly": 22, "poc_monthly": 25,
            "vah_daily": 15, "val_daily": 15,
            "vah_weekly": 18, "val_weekly": 18,
            "pivot_classic_PP": 18, "pivot_classic_R1": 14, "pivot_classic_S1": 14,
            "ema_21_1d": 16, "ema_21_4h": 14,
            "round_number": 10,
            "hvn_daily": 12,
        }
        score += source_weights.get(source, 10)

        # Confluência
        score += min(30, confluence_count * 8)

        # Proximidade
        if current_price > 0:
            dist_pct = abs(level_price - current_price) / current_price * 100
            score += max(0, int(20 - dist_pct * 3))

        return min(score, 100)