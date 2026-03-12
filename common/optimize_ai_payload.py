"""
Otimizador de Payload para API de IA

Reduz tamanho de eventos JSON mantendo a qualidade dos dados necessários
para decisões da IA.

Uso:
    from optimize_ai_payload import optimize_event_for_ai, build_optimized_ai_payload

    optimized_event = optimize_event_for_ai(original_event)
    ai_payload = build_optimized_ai_payload(original_event)
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any, Dict, Iterable, List, Optional

from payload_optimizer_config import (
    FIELDS_TO_REMOVE,
    FILTER_CONFIG,
    ROUNDING_CONFIG,
    SAFETY_CONFIG,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# FUNÇÃO PRINCIPAL
# ==============================================================================


def optimize_event_for_ai(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Otimiza um evento para envio à API de IA.

    Ordem:
    1) Remove campos desnecessários
    2) Simplifica historical_vp (HVNs/LVNs)
    3) Remove duplicações (snapshots/estruturas repetidas)
    4) Arredonda números
    5) Filtra arrays grandes (clusters)
    6) Valida o resultado
    """

    if not event or not isinstance(event, dict):
        raise ValueError("Evento deve ser um dicionário não vazio")

    for field in SAFETY_CONFIG.get("abort_if_missing", []):
        if not has_nested_field(event, str(field)):
            raise ValueError(f"Evento inválido: campo obrigatório ausente: {field}")

    original: Optional[Dict[str, Any]] = None
    if SAFETY_CONFIG.get("create_backup", True):
        original = copy.deepcopy(event)

    try:
        optimized = remove_unnecessary_fields(event)
        optimized = simplify_historical_vp(optimized)
        optimized = remove_duplications(optimized)
        optimized = round_numbers(optimized)
        optimized = filter_large_arrays(optimized)
        validate_optimized_event(optimized)

        if original is not None:
            original_size = len(json.dumps(original, ensure_ascii=False, separators=(",", ":")))
            optimized_size = len(json.dumps(optimized, ensure_ascii=False, separators=(",", ":")))
            reduction_pct = ((original_size - optimized_size) / original_size * 100) if original_size else 0.0
            logger.info(
                "📊 Otimização completa: %s → %s bytes (-%.1f%%)",
                f"{original_size:,}",
                f"{optimized_size:,}",
                reduction_pct,
            )

        return optimized
    except Exception as exc:
        logger.exception("❌ Erro na otimização: %s", exc)
        if original is not None:
            logger.warning("⚠️ Retornando evento original (sem otimização)")
            return original
        raise


# ==============================================================================
# PASSO 1: REMOVER CAMPOS DESNECESSÁRIOS
# ==============================================================================


def remove_unnecessary_fields(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove campos que não são usados pela IA (recursivo).
    """

    removed_keys: set[str] = set()

    def recursive_remove(obj: Any) -> Any:
        if isinstance(obj, dict):
            out: Dict[str, Any] = {}
            for key, value in obj.items():
                if key in FIELDS_TO_REMOVE:
                    removed_keys.add(key)
                    continue
                out[key] = recursive_remove(value)
            return out
        if isinstance(obj, list):
            return [recursive_remove(item) for item in obj]
        return obj

    cleaned = recursive_remove(event)
    if SAFETY_CONFIG.get("log_removed_fields", True) and removed_keys:
        logger.debug("Removidos: %s", ", ".join(sorted(removed_keys)))
    return cleaned


# ==============================================================================
# PASSO 2: SIMPLIFICAR HISTORICAL_VP
# ==============================================================================


def simplify_historical_vp(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simplifica a estrutura `historical_vp` (daily/weekly/monthly) removendo arrays
    grandes e mantendo apenas níveis próximos do preço atual.
    """

    vp_locations: list[list[str]] = [
        ["raw_event", "historical_vp"],
        ["contextual_snapshot", "historical_vp"],
        ["historical_vp"],
    ]

    current_price = get_current_price(event)

    for location in vp_locations:
        vp_data = get_nested_value(event, location)
        if not isinstance(vp_data, dict):
            continue

        for timeframe in ("daily", "weekly", "monthly"):
            tf_data = vp_data.get(timeframe)
            if isinstance(tf_data, dict):
                vp_data[timeframe] = simplify_vp_timeframe(tf_data, current_price)

        set_nested_value(event, location, vp_data)

    return event


def simplify_vp_timeframe(tf_data: Dict[str, Any], current_price: float) -> Dict[str, Any]:
    """
    Mantém:
      - poc, vah, val (sempre)
      - hvns_nearby / lvns_nearby (máx 5 níveis próximos)
    """

    simplified: Dict[str, Any] = {
        "poc": tf_data.get("poc"),
        "vah": tf_data.get("vah"),
        "val": tf_data.get("val"),
        "status": tf_data.get("status", "success"),
    }

    nearby_cfg = FILTER_CONFIG.get("volume_profile_nearby", {})
    if nearby_cfg.get("enabled", True) and current_price:
        max_distance_pct = float(nearby_cfg.get("max_distance_percent", 5.0))
        max_levels = int(nearby_cfg.get("max_levels", 5))

        hvns = tf_data.get("hvns")
        if isinstance(hvns, list):
            hvns_nearby = filter_nearby_levels(hvns, current_price, max_distance_pct=max_distance_pct, max_levels=max_levels)
            if hvns_nearby:
                simplified["hvns_nearby"] = hvns_nearby

        lvns = tf_data.get("lvns")
        if isinstance(lvns, list):
            lvns_nearby = filter_nearby_levels(lvns, current_price, max_distance_pct=max_distance_pct, max_levels=max_levels)
            if lvns_nearby:
                simplified["lvns_nearby"] = lvns_nearby

    return simplified


def filter_nearby_levels(
    levels: List[Any],
    reference_price: float,
    max_distance_pct: float = 5.0,
    max_levels: int = 5,
) -> List[float]:
    """
    Filtra níveis numéricos (HVNs/LVNs) próximos ao `reference_price`.
    """

    if not levels or not reference_price:
        return []

    numeric_levels: list[float] = []
    for level in levels:
        if isinstance(level, (int, float)):
            numeric_levels.append(float(level))

    if not numeric_levels:
        return []

    max_distance = reference_price * (max_distance_pct / 100.0)
    nearby = [level for level in numeric_levels if abs(level - reference_price) <= max_distance]
    nearby.sort(key=lambda price: abs(price - reference_price))
    return nearby[:max_levels]


# ==============================================================================
# PASSO 3: REMOVER DUPLICAÇÕES
# ==============================================================================


def remove_duplications(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove estruturas duplicadas comuns no evento.
    """

    if "enriched_snapshot" in event and "contextual_snapshot" in event:
        del event["enriched_snapshot"]

    if "fluxo_continuo" in event and "flow_metrics" in event:
        if isinstance(event.get("fluxo_continuo"), dict) and isinstance(event.get("flow_metrics"), dict):
            if are_dicts_similar(event["fluxo_continuo"], event["flow_metrics"]):
                del event["fluxo_continuo"]

    # Promover orderbook_data para o nível principal (se existir em locais aninhados)
    orderbook_locations: list[list[str]] = [
        ["raw_event", "orderbook_data"],
        ["contextual_snapshot", "orderbook_data"],
    ]

    if not isinstance(event.get("orderbook_data"), dict):
        for location in orderbook_locations:
            candidate = get_nested_value(event, location)
            if isinstance(candidate, dict) and candidate:
                event["orderbook_data"] = candidate
                break

    for location in orderbook_locations:
        del_nested_value(event, location)

    return event


# ==============================================================================
# PASSO 4: ARREDONDAR NÚMEROS
# ==============================================================================


def round_numbers(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Arredonda números recursivamente para reduzir precisão desnecessária.
    """

    def round_recursive(obj: Any, context_field: str = "") -> Any:
        if isinstance(obj, dict):
            return {key: round_recursive(value, key) for key, value in obj.items()}
        if isinstance(obj, list):
            return [round_recursive(item, context_field) for item in obj]
        if isinstance(obj, float):
            decimals = get_decimal_precision(context_field, obj)
            if decimals <= 0:
                return int(round(obj))
            return round(obj, decimals)
        return obj

    return round_recursive(event)


def get_decimal_precision(field_name: str, value: float) -> int:
    """
    Determina quantas casas decimais manter baseado no nome do campo.
    """

    for category, config in ROUNDING_CONFIG.items():
        if category == "small_numbers":
            continue
        fields = config.get("fields") or []
        if field_name in fields:
            return int(config.get("decimals", 2))

    small_cfg = ROUNDING_CONFIG.get("small_numbers", {})
    threshold = float(small_cfg.get("threshold", 0.01))
    if abs(value) < threshold:
        return int(small_cfg.get("decimals", 4))

    return 2


# ==============================================================================
# PASSO 5: FILTRAR ARRAYS GRANDES
# ==============================================================================


def filter_large_arrays(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Limita tamanho de arrays grandes (ex.: clusters de liquidez).
    """

    clusters_cfg = FILTER_CONFIG.get("liquidity_clusters", {})
    if not clusters_cfg.get("enabled", True):
        return event

    cluster_locations: list[list[str]] = [
        ["liquidity_heatmap", "clusters"],
        ["flow_metrics", "liquidity_heatmap", "clusters"],
    ]

    min_vol = float(clusters_cfg.get("min_volume_threshold", 0.5))
    max_clusters = int(clusters_cfg.get("max_clusters", 3))

    for location in cluster_locations:
        clusters = get_nested_value(event, location)
        if not isinstance(clusters, list):
            continue

        filtered: list[dict[str, Any]] = []
        for cluster in clusters:
            if not isinstance(cluster, dict):
                continue
            total_volume = cluster.get("total_volume", 0)
            try:
                total_volume = float(total_volume)
            except Exception:
                total_volume = 0.0
            if total_volume >= min_vol:
                filtered.append(cluster)

        filtered.sort(key=lambda c: float(c.get("total_volume", 0) or 0), reverse=True)
        set_nested_value(event, location, filtered[:max_clusters])

    return event


# ==============================================================================
# PASSO 6: VALIDAR RESULTADO
# ==============================================================================


def validate_optimized_event(event: Dict[str, Any]) -> None:
    """
    Valida que o evento otimizado ainda contém dados críticos.
    """

    required_fields = [str(x) for x in SAFETY_CONFIG.get("required_fields", [])]
    for field in required_fields:
        if not has_nested_field(event, field):
            raise ValueError(f"Campo obrigatório ausente após otimização: {field}")

    try:
        json.dumps(event)
    except Exception as exc:
        raise ValueError(f"Evento otimizado não é JSON válido: {exc}") from exc


# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================


def get_current_price(event: Dict[str, Any]) -> float:
    """
    Extrai preço atual do evento (múltiplos locais possíveis).
    """

    price_locations: list[Any] = [
        "current_price",
        "preco_fechamento",
        ["raw_event", "preco_fechamento"],
        ["ohlc", "close"],
        ["raw_event", "ohlc", "close"],
        ["contextual_snapshot", "ohlc", "close"],
        ["enriched_snapshot", "ohlc", "close"],
    ]

    for location in price_locations:
        price = get_nested_value(event, location)
        if isinstance(price, (int, float)) and price:
            return float(price)

    return 0.0


def get_nested_value(obj: Dict[str, Any], path: Any) -> Any:
    """
    Obtém valor de caminho aninhado.
    """

    if isinstance(path, str):
        return obj.get(path)
    if isinstance(path, list):
        current: Any = obj
        for key in path:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current
    return None


def first_nested_value(obj: Dict[str, Any], paths: Iterable[Any]) -> Any:
    """
    Retorna o primeiro valor não-nulo encontrado entre vários caminhos.
    """

    for path in paths:
        value = get_nested_value(obj, path)
        if value is None:
            continue
        if value == "":
            continue
        return value
    return None


def set_nested_value(obj: Dict[str, Any], path: List[str], value: Any) -> None:
    """
    Define valor em caminho aninhado (criando dicts no caminho se necessário).
    """

    current: Any = obj
    for key in path[:-1]:
        if not isinstance(current, dict):
            return
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    if isinstance(current, dict):
        current[path[-1]] = value


def del_nested_value(obj: Dict[str, Any], path: List[str]) -> None:
    """
    Remove valor de caminho aninhado (se existir).
    """

    current: Any = obj
    for key in path[:-1]:
        if not isinstance(current, dict) or key not in current:
            return
        current = current[key]
    if isinstance(current, dict):
        current.pop(path[-1], None)


def has_nested_field(obj: Any, field: str) -> bool:
    """
    Verifica se `field` existe em qualquer nível do objeto (dict/list).
    """

    if isinstance(obj, dict):
        if field in obj:
            return True
        return any(has_nested_field(value, field) for value in obj.values())
    if isinstance(obj, list):
        return any(has_nested_field(item, field) for item in obj)
    return False


def flatten_keys(obj: Any, prefix: str = "") -> set[str]:
    """
    Retorna conjunto de chaves "achatadas" (recursivo), útil para comparar estruturas.
    """

    keys: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            full_key = f"{prefix}.{key}" if prefix else str(key)
            keys.add(full_key)
            keys.update(flatten_keys(value, full_key))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            full_key = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            keys.update(flatten_keys(item, full_key))
    return keys


def are_dicts_similar(dict1: Dict[str, Any], dict2: Dict[str, Any], threshold: float = 0.8) -> bool:
    """
    Verifica se dois dicts são estruturalmente similares (>= threshold).
    """

    keys1 = flatten_keys(dict1)
    keys2 = flatten_keys(dict2)
    if not keys1 or not keys2:
        return False
    common = keys1 & keys2
    total = keys1 | keys2
    similarity = len(common) / len(total)
    return similarity >= threshold


# ==============================================================================
# FUNÇÃO DE CONVENIÊNCIA PARA AI_PAYLOAD
# ==============================================================================


def build_optimized_ai_payload(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Constrói payload minimalista para IA a partir de um evento completo.
    """

    current_price = get_current_price(event)
    ohlc = extract_ohlc(event)

    flow_paths = {
        "net_flow_1m": [
            ["flow_metrics", "order_flow", "net_flow_1m"],
            ["fluxo_continuo", "order_flow", "net_flow_1m"],
        ],
        "cvd": [
            ["flow_metrics", "cvd"],
            ["fluxo_continuo", "cvd"],
        ],
        "flow_imbalance": [
            ["flow_metrics", "order_flow", "flow_imbalance"],
            ["fluxo_continuo", "order_flow", "flow_imbalance"],
        ],
        "aggressive_buy_pct": [
            ["flow_metrics", "order_flow", "aggressive_buy_pct"],
            ["fluxo_continuo", "order_flow", "aggressive_buy_pct"],
        ],
        "aggressive_sell_pct": [
            ["flow_metrics", "order_flow", "aggressive_sell_pct"],
            ["fluxo_continuo", "order_flow", "aggressive_sell_pct"],
        ],
        "absorption_type": [
            ["flow_metrics", "tipo_absorcao"],
            ["fluxo_continuo", "tipo_absorcao"],
        ],
    }

    clusters = first_nested_value(
        event,
        [
            ["liquidity_heatmap", "clusters"],
            ["flow_metrics", "liquidity_heatmap", "clusters"],
        ],
    )
    clusters_count = len(clusters) if isinstance(clusters, list) else 0

    payload: Dict[str, Any] = {
        "price_context": {
            "current_price": current_price,
            "ohlc": ohlc,
            "price_action": {"close_position": calculate_close_position_from_ohlc(ohlc)},
            "volume_profile_daily": extract_vp_levels(event, "daily", current_price),
            "volatility": {
                "volatility_regime": first_nested_value(
                    event,
                    [
                        ["market_environment", "volatility_regime"],
                        ["market_context", "volatility_regime"],
                    ],
                )
            },
        },
        "flow_context": {
            "net_flow": first_nested_value(event, flow_paths["net_flow_1m"]),
            "cvd_accumulated": first_nested_value(event, flow_paths["cvd"]),
            "flow_imbalance": first_nested_value(event, flow_paths["flow_imbalance"]),
            "aggressive_buyers": first_nested_value(event, flow_paths["aggressive_buy_pct"]),
            "aggressive_sellers": first_nested_value(event, flow_paths["aggressive_sell_pct"]),
            "liquidity_clusters_count": clusters_count,
            "absorption_type": first_nested_value(event, flow_paths["absorption_type"]),
        },
        "orderbook_context": {
            "bid_depth_usd": first_nested_value(event, [["orderbook_data", "bid_depth_usd"]]),
            "ask_depth_usd": first_nested_value(event, [["orderbook_data", "ask_depth_usd"]]),
            "imbalance": first_nested_value(event, [["orderbook_data", "imbalance"]]),
            "market_impact_score": first_nested_value(event, [["orderbook_data", "pressure"]]),
            "walls_detected": 1 if first_nested_value(event, [["orderbook_data", "depth_metrics"]]) else 0,
            "depth_metrics": first_nested_value(event, [["orderbook_data", "depth_metrics"]]),
        },
        "macro_context": {
            "session": first_nested_value(event, [["market_context", "trading_session"]]),
            "phase": first_nested_value(event, [["market_context", "session_phase"]]),
            "regime": {
                "structure": first_nested_value(event, [["market_environment", "market_structure"]]),
                "trend": first_nested_value(event, [["market_environment", "trend_direction"]]),
                "sentiment": first_nested_value(event, [["market_environment", "risk_sentiment"]]),
            },
        },
    }

    return round_numbers(payload)


def extract_ohlc(event: Dict[str, Any]) -> Dict[str, float]:
    """
    Extrai OHLC de múltiplas fontes possíveis.
    """

    ohlc_source = first_nested_value(
        event,
        [
            ["enriched_snapshot", "ohlc"],
            ["contextual_snapshot", "ohlc"],
            ["raw_event", "ohlc"],
            ["ohlc"],
        ],
    )
    if not isinstance(ohlc_source, dict):
        ohlc_source = {}

    def _to_float(value: Any) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        return 0.0

    return {
        "open": _to_float(ohlc_source.get("open")),
        "high": _to_float(ohlc_source.get("high")),
        "low": _to_float(ohlc_source.get("low")),
        "close": _to_float(ohlc_source.get("close")),
        "vwap": _to_float(ohlc_source.get("vwap")),
    }


def calculate_close_position_from_ohlc(ohlc: Dict[str, float]) -> float:
    """
    Calcula posição do close no candle (0=low, 1=high).
    """

    high = float(ohlc.get("high", 0.0))
    low = float(ohlc.get("low", 0.0))
    close = float(ohlc.get("close", 0.0))
    if high == low:
        return 0.5
    return (close - low) / (high - low)


def extract_vp_levels(event: Dict[str, Any], timeframe: str, current_price: float) -> Dict[str, Any]:
    """
    Extrai níveis de volume profile de forma otimizada.
    """

    vp_data = first_nested_value(
        event,
        [
            ["historical_vp", timeframe],
            ["contextual_snapshot", "historical_vp", timeframe],
            ["raw_event", "historical_vp", timeframe],
        ],
    )

    if not isinstance(vp_data, dict):
        vp_data = {}

    result: Dict[str, Any] = {
        "poc": vp_data.get("poc"),
        "vah": vp_data.get("vah"),
        "val": vp_data.get("val"),
    }

    if "hvns_nearby" in vp_data:
        result["hvns_nearby"] = vp_data.get("hvns_nearby")
    elif "hvns" in vp_data and isinstance(vp_data.get("hvns"), list):
        result["hvns_nearby"] = filter_nearby_levels(vp_data["hvns"], current_price, max_levels=5)

    if "lvns_nearby" in vp_data:
        result["lvns_nearby"] = vp_data.get("lvns_nearby")
    elif "lvns" in vp_data and isinstance(vp_data.get("lvns"), list):
        result["lvns_nearby"] = filter_nearby_levels(vp_data["lvns"], current_price, max_levels=5)

    return result

