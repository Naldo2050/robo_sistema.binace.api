# support_resistance.py
import numpy as np
import pandas as pd

# =============================
#  PIVOT POINTS
# =============================

def calculate_pivot_points(high: float, low: float, close: float) -> dict:
    """
    Calculate classic pivot points and support/resistance levels for a given period.
    Returns a dictionary with pivot (P), R1, R2, S1, S2.
    """
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return {"pivot": pivot, "r1": r1, "s1": s1, "r2": r2, "s2": s2}

def _period_high_low_close(df: pd.DataFrame, period: str) -> tuple[float, float, float]:
    """
    Helper to extract high, low, close for a specified period ('D', 'W', 'M').
    Expects df with datetime index and columns high, low, close.
    Returns (high, low, close) for the last complete period.
    """
    if df.empty:
        return (0.0, 0.0, 0.0)
    resampled = df.resample(period).agg({'high': 'max', 'low': 'min', 'close': 'last'})
    if len(resampled) < 2:
        last = resampled.iloc[-1]
        return (float(last['high']), float(last['low']), float(last['close']))
    last_complete = resampled.iloc[-2]
    return (float(last_complete['high']), float(last_complete['low']), float(last_complete['close']))

def daily_pivot(df: pd.DataFrame) -> dict:
    """Compute daily pivot points using the last complete day."""
    high, low, close = _period_high_low_close(df, 'D')
    return calculate_pivot_points(high, low, close)

def weekly_pivot(df: pd.DataFrame) -> dict:
    """Compute weekly pivot points using the last complete week."""
    high, low, close = _period_high_low_close(df, 'W')
    return calculate_pivot_points(high, low, close)

def monthly_pivot(df: pd.DataFrame) -> dict:
    """Compute monthly pivot points using the last complete month."""
    high, low, close = _period_high_low_close(df, 'M')
    return calculate_pivot_points(high, low, close)

# =============================
#  SUPORTE & RESISTÊNCIA
# =============================

def detect_support_resistance(price_series: pd.Series, num_levels: int = 3) -> dict:
    """
    Detect immediate support and resistance levels from a price series.
    Handles collisions (same price as both support and resistance).
    """
    if price_series.empty:
        return {
            "immediate_support": [], "immediate_resistance": [],
            "support_strength": [], "resistance_strength": []
        }

    prices = price_series.astype(float).dropna().values
    supports, resistances = [], []

    # Identificação de mínimos/máximos locais
    for i in range(1, len(prices) - 1):
        if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
            supports.append(prices[i])
        if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
            resistances.append(prices[i])

    supports = sorted(set(supports))
    resistances = sorted(set(resistances), reverse=True)

    # Remove níveis duplicados como suporte e resistência
    overlap = set(supports) & set(resistances)
    if overlap:
        supports = [lvl for lvl in supports if lvl not in overlap]
        resistances = [lvl for lvl in resistances if lvl not in overlap]

    # Ordenação por proximidade ao último fechamento
    last_close = prices[-1]
    supports = sorted(supports, key=lambda x: abs(last_close - x))[:num_levels]
    resistances = sorted(resistances, key=lambda x: abs(last_close - x))[:num_levels]

    def calc_strength(level: float) -> float:
        touches = np.sum(np.isclose(prices, level, atol=level * 0.001))
        proximity = 1 / (abs(last_close - level) + 1e-6)
        return touches * proximity

    return {
        "immediate_support": supports,
        "immediate_resistance": resistances,
        "support_strength": [calc_strength(l) for l in supports],
        "resistance_strength": [calc_strength(l) for l in resistances],
    }

# =============================
#  ZONAS DE DEFESA
# =============================

def defense_zones(support_resistance: dict) -> dict:
    """
    Create bull and bear defense zones from support/resistance levels.
    """
    supports = support_resistance.get("immediate_support", [])
    resistances = support_resistance.get("immediate_resistance", [])
    support_strength = support_resistance.get("support_strength", [])
    resistance_strength = support_resistance.get("resistance_strength", [])

    bull_defense, bear_defense, no_mans_land = {}, {}, None

    if supports:
        p_price, p_strength = supports[0], support_strength[0] if support_strength else 0.0
        if len(supports) > 1:
            s_price, s_strength = supports[1], support_strength[1] if len(support_strength) > 1 else 0.0
            bull_defense = {
                "primary": {"price": p_price, "width": abs(p_price - s_price), "strength": p_strength},
                "secondary": {"price": s_price, "width": abs(p_price - s_price), "strength": s_strength},
            }
        else:
            bull_defense = {"primary": {"price": p_price, "width": 0.0, "strength": p_strength}, "secondary": None}

    if resistances:
        p_price, p_strength = resistances[0], resistance_strength[0] if resistance_strength else 0.0
        if len(resistances) > 1:
            s_price, s_strength = resistances[1], resistance_strength[1] if len(resistance_strength) > 1 else 0.0
            bear_defense = {
                "primary": {"price": p_price, "width": abs(p_price - s_price), "strength": p_strength},
                "secondary": {"price": s_price, "width": abs(p_price - s_price), "strength": s_strength},
            }
        else:
            bear_defense = {"primary": {"price": p_price, "width": 0.0, "strength": p_strength}, "secondary": None}

    if supports and resistances:
        sup, res = max(supports), min(resistances)
        if sup < res:
            no_mans_land = {"start": sup, "end": res}

    return {"bull_defense": bull_defense or None,
            "bear_defense": bear_defense or None,
            "no_mans_land": no_mans_land}

# =============================
#  MONITORAMENTO EM TEMPO REAL
# =============================

class MarketMonitor:
    """
    Monitora níveis de suporte e resistência em tempo real,
    analisando como o mercado reage quando eles são testados.
    """
    def __init__(self, support_levels, resistance_levels, tolerance=0.001):
        self.levels = {
            "support": [{"price": lvl, "status": "active"} for lvl in support_levels],
            "resistance": [{"price": lvl, "status": "active"} for lvl in resistance_levels],
        }
        self.tolerance = tolerance
        self.logs = []

    def process_tick(self, price: float, volume: float, delta: float):
        """
        Processa um tick/candle/trade contra os níveis.
        """
        for side in ["support", "resistance"]:
            for level in self.levels[side]:
                if level["status"] != "active":
                    continue
                distance = abs(price - level["price"]) / level["price"]
                if distance <= self.tolerance:
                    signal = self._analyze_level(price, volume, delta, side, level["price"])
                    self.logs.append(signal)
                    return signal  # retorna alerta em tempo real

    def _analyze_level(self, price, volume, delta, side, level_price):
        """
        Interpreta reações institucionais no nível.
        """
        if side == "support":
            if delta > 0:
                analysis = "Support defended (buyers absorbing supply)"
            elif delta < 0:
                analysis = "Support weak (sellers dominating)"
            else:
                analysis = "Neutral reaction at support"
        else:  # resistance
            if delta < 0:
                analysis = "Resistance defended (sellers absorbing demand)"
            elif delta > 0:
                analysis = "Resistance weak (buyers pushing through)"
            else:
                analysis = "Neutral reaction at resistance"

        return {
            "level_type": side,
            "level_price": level_price,
            "price": price,
            "volume": volume,
            "delta": delta,
            "analysis": analysis,
        }

    def report(self) -> pd.DataFrame:
        """Consolida todos os testes de nível em DataFrame para relatório."""
        return pd.DataFrame(self.logs)