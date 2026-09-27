# flow_analyzer/effort_response_dataset.py
"""
P1-E — Effort/Response Shadow Dataset v1 (Pre-Commit Hardening).

Coleta SHADOW estritamente observacional e causal para posterior avaliação
empírica de:
- Esforço contemporâneo (notionals buy/sell/total/net, shares);
- Resposta de preço contemporânea (displacement, range, posições relativas);
- Contexto observável no instante t (ordem, spread, tempo, volatilidade, regime);
- Outcomes futuros armazenados separadamente (1m, 5m, 15m).

REGRAS ESTRITAS DE ARQUITETURA E HARDENING:
1. Separação causal física e lógica: features_at_t vs context_at_t vs outcomes_future.
2. Zero lookahead nas features e contexto: excursões, retornos e preços futuros
   residem EXCLUSIVAMENTE em outcomes_future.
3. Builder de features é puro O(1) e consome diretamente compute_effort_response (P1-D).
4. Builder de outcomes é função separada e anexa dados por record_id de forma idempotente.
5. Record ID determinístico baseado em symbol, close timestamp e versões contratuais.
6. Armazenamento JSONL auditável, append-friendly, compatível com RFC8259 (sem NaN/Inf).
7. Modelo de concorrência: SINGLE_WRITER_ONLY. Não é multi-process safe sem locks externos.
8. Storage fail-closed: corrupção no meio do arquivo ou update órfão lança exceção.
9. Contrato temporal de boundary alinhado ao OutcomeTracker (tolerância 1000ms).
10. Whitelist estrita de campos em contexto e proveniência (zero cópia de raw_event/env).
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from flow_analyzer.effort_response import (
    COMPLETE,
    INVALID,
    PARTIAL,
    VALID,
    compute_effort_response,
)

logger = logging.getLogger("EffortResponseDataset")

SHADOW_SCHEMA_VERSION = "1.0.0"
FEATURE_CONTRACT_VERSION = "1.0.0"
CONCURRENCY_MODEL = "SINGLE_WRITER_ONLY"

# Tolerância temporal auditada de boundary do projeto (definida em trading/outcome_tracker.py)
OUTCOME_BOUNDARY_TOLERANCE_MS: int = 1000

# 1 semana tem no máximo 10.080 minutos teóricos (7 dias * 24 horas * 60 minutos)
MAX_THEORETICAL_MINUTES_PER_WEEK: int = 10080

# Chaves proibidas para garantir proteção contra vazamento de segredos / credenciais
FORBIDDEN_SECRET_KEYS: frozenset[str] = frozenset({
    "api_key",
    "apikey",
    "secret",
    "secret_key",
    "secretkey",
    "private_key",
    "token",
    "access_token",
    "password",
    "authorization",
    "auth",
    "credential",
    "credentials",
    "cookie",
    "cookies",
    "env",
    "llm_prompt",
    "system_prompt",
    "prompt",
})

# Substrings proibidas em features_at_t e context_at_t para prevenir vazamento de lookahead
FORBIDDEN_LOOKAHEAD_WORDS: frozenset[str] = frozenset({
    "future",
    "mfe",
    "mae",
    "win",
    "loss",
    "label",
    "target",
    "lookahead",
    "pnl",
    "drawdown",
})

FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "future",
    "lookahead",
    "mfe",
    "mae",
)

# Whitelist estrita de campos aceitos no context_at_t
ALLOWED_CONTEXT_KEYS: frozenset[str] = frozenset({
    "timestamp_utc",
    "symbol",
    "day_of_week",
    "session_time_bucket",
    "spread",
    "bid_depth",
    "ask_depth",
    "orderbook_imbalance",
    "orderbook_imbalance_source_type",
    "flow_temporal_validity",
    "realized_volatility",
    "volume_base",
    "trade_count",
    "activity_rate",
    "session_vwap_distance",
    "market_structure",
    "regime_current_at_t",
    "regime_status_at_t",
    "regime_calibration_status_at_t",
    "is_weekend",
    "is_holiday",
    "data_latency_ms",
    "data_freshness_ms",
    "feature_contract_version",
})

# Whitelist estrita de campos aceitos na proveniência
ALLOWED_PROVENANCE_KEYS: frozenset[str] = frozenset({
    "exchange",
    "stream",
    "symbol",
    "window_open_ms",
    "window_close_ms",
    "source_event_id",
    "orderbook_source_type",
    "orderbook_snapshot_ms",
    "observation_open_ms",
    "observation_close_ms",
    "causal_anchor_ms",
})


class StorageCorruptionError(Exception):
    """Lançado quando uma corrupção irrecuperável é encontrada no armazenamento."""
    pass


class StorageOrphanUpdateError(Exception):
    """Lançado quando um OUTCOME_UPDATE faz referência a um record_id inexistente."""
    pass


def _validate_no_secrets(obj: Any, path: str = "root") -> None:
    """Valida recursivamente se nenhum segredo ou credencial existe no objeto."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            k_lower = str(k).lower()
            for forbidden in FORBIDDEN_SECRET_KEYS:
                if forbidden in k_lower:
                    raise ValueError(
                        f"Campo sensível proibido detectado no dataset em '{path}.{k}': '{forbidden}'"
                    )
            _validate_no_secrets(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple, set)):
        for i, item in enumerate(obj):
            _validate_no_secrets(item, f"{path}[{i}]")


def _validate_no_lookahead_keys(mapping: Dict[str, Any], container_name: str) -> None:
    """Garante que nenhuma chave em features ou contexto contenha termos de lookahead."""
    for key in mapping.keys():
        k_lower = str(key).lower()
        # 1. Checagem por substring explícita para termos sem ambiguidade
        for sub in FORBIDDEN_SUBSTRINGS:
            if sub in k_lower:
                raise ValueError(
                    f"Violação de causalidade: termo de lookahead '{sub}' "
                    f"encontrado na chave '{key}' de '{container_name}'."
                )
        # 2. Checagem por token exato (delimitado por underscore) para termos curtos como 'win' ou 'loss'
        parts = k_lower.split("_")
        for p in parts:
            if p in FORBIDDEN_LOOKAHEAD_WORDS:
                raise ValueError(
                    f"Violação de causalidade: token de lookahead '{p}' "
                    f"encontrado na chave '{key}' de '{container_name}'."
                )


def build_deterministic_record_id(
    *,
    symbol: str,
    causal_anchor_ms: Optional[int] = None,
    window_close_ms: Optional[int] = None,
    feature_contract_version: str = FEATURE_CONTRACT_VERSION,
    shadow_schema_version: str = SHADOW_SCHEMA_VERSION,
) -> str:
    """Gera record_id determinístico invariante.

    Evolução P1-F: aceita causal_anchor_ms (boundary lógico canônico da janela).
    Preserva window_close_ms por retrocompatibilidade se causal_anchor_ms não for informado.
    """
    anchor = causal_anchor_ms if causal_anchor_ms is not None else window_close_ms
    if anchor is None:
        raise ValueError("build_deterministic_record_id requer causal_anchor_ms ou window_close_ms")
    clean_sym = symbol.strip().upper()
    return f"rec_{clean_sym}_{int(anchor)}_v{feature_contract_version}_s{shadow_schema_version}"


@dataclass(frozen=True)
class ShadowVersions:
    shadow_schema_version: str = SHADOW_SCHEMA_VERSION
    feature_contract_version: str = FEATURE_CONTRACT_VERSION


@dataclass(frozen=True)
class ShadowProvenance:
    exchange: str
    stream: str
    symbol: str
    window_open_ms: int
    window_close_ms: int
    source_event_id: Optional[str] = None
    orderbook_source_type: Optional[str] = None
    orderbook_snapshot_ms: Optional[int] = None
    observation_open_ms: Optional[int] = None
    observation_close_ms: Optional[int] = None
    causal_anchor_ms: Optional[int] = None

    def __post_init__(self) -> None:
        # Preenchimento defensivo retrocompatível
        if self.observation_open_ms is None:
            object.__setattr__(self, "observation_open_ms", self.window_open_ms)
        if self.observation_close_ms is None:
            object.__setattr__(self, "observation_close_ms", self.window_close_ms)
        if self.causal_anchor_ms is None:
            object.__setattr__(self, "causal_anchor_ms", self.window_close_ms)


@dataclass(frozen=True)
class ShadowQuality:
    core_validity: str
    optional_completeness: str
    flow_window_validity: Optional[str] = None
    latency_ms: Optional[float] = None
    freshness_ms: Optional[float] = None
    missing_context_fields: List[str] = field(default_factory=list)


@dataclass
class HorizonOutcome:
    horizon: str  # "1m", "5m", "15m"
    status: str   # "PENDING" | "RESOLVED" | "INSUFFICIENT_DATA" | "INVALID_BASE_PRICE"
    target_timestamp_ms: Optional[int] = None
    observed_price_timestamp_ms: Optional[int] = None
    timing_error_ms: Optional[int] = None
    future_price: Optional[float] = None
    return_bps: Optional[float] = None
    max_excursion_up_bps: Optional[float] = None
    max_excursion_down_bps: Optional[float] = None
    mfe_bps: Optional[float] = None  # Alias para max_excursion_up_bps (sem direção de trade)
    mae_bps: Optional[float] = None  # Alias para max_excursion_down_bps (sem direção de trade)
    max_high: Optional[float] = None
    min_low: Optional[float] = None
    observation_count: Optional[int] = None
    horizon_duration_ms: Optional[int] = None
    resolved_at_ms: Optional[int] = None
    excursion_status: Optional[str] = None  # "FULL" | "PARTIAL" | "INSUFFICIENT_DATA"
    excursion_coverage_start_ms: Optional[int] = None
    excursion_coverage_end_ms: Optional[int] = None
    excursion_observation_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EffortResponseShadowRecord:
    record_id: str
    versions: ShadowVersions
    provenance: ShadowProvenance
    quality: ShadowQuality
    features_at_t: Dict[str, Any]
    context_at_t: Dict[str, Any]
    outcomes_future: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "versions": asdict(self.versions),
            "provenance": asdict(self.provenance),
            "quality": asdict(self.quality),
            "features_at_t": dict(self.features_at_t),
            "context_at_t": dict(self.context_at_t),
            "outcomes_future": dict(self.outcomes_future),
        }

    def to_json(self) -> str:
        """Serialização estrita RFC8259 (sem NaN/Inf)."""
        d = self.to_dict()
        _validate_no_secrets(d)
        return json.dumps(d, allow_nan=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EffortResponseShadowRecord:
        _validate_no_secrets(data)
        v = data.get("versions", {})
        versions = ShadowVersions(
            shadow_schema_version=v.get("shadow_schema_version", SHADOW_SCHEMA_VERSION),
            feature_contract_version=v.get("feature_contract_version", FEATURE_CONTRACT_VERSION),
        )
        p = data.get("provenance", {})
        # Whitelist de proveniência
        for k in p.keys():
            if k not in ALLOWED_PROVENANCE_KEYS:
                raise ValueError(f"Campo não permitido na proveniência: '{k}'")

        obs_open = p.get("observation_open_ms")
        obs_close = p.get("observation_close_ms")
        anchor = p.get("causal_anchor_ms")

        provenance = ShadowProvenance(
            exchange=p.get("exchange", "binance_futures"),
            stream=p.get("stream", "aggTrade"),
            symbol=p.get("symbol", ""),
            window_open_ms=int(p.get("window_open_ms", 0)),
            window_close_ms=int(p.get("window_close_ms", 0)),
            source_event_id=p.get("source_event_id"),
            orderbook_source_type=p.get("orderbook_source_type"),
            orderbook_snapshot_ms=p.get("orderbook_snapshot_ms"),
            observation_open_ms=int(obs_open) if obs_open is not None else None,
            observation_close_ms=int(obs_close) if obs_close is not None else None,
            causal_anchor_ms=int(anchor) if anchor is not None else None,
        )
        q = data.get("quality", {})
        quality = ShadowQuality(
            core_validity=q.get("core_validity", INVALID),
            optional_completeness=q.get("optional_completeness", PARTIAL),
            flow_window_validity=q.get("flow_window_validity"),
            latency_ms=q.get("latency_ms"),
            freshness_ms=q.get("freshness_ms"),
            missing_context_fields=list(q.get("missing_context_fields", [])),
        )
        features = dict(data.get("features_at_t", {}))
        context = dict(data.get("context_at_t", {}))

        # Whitelist de contexto
        for k in context.keys():
            if k not in ALLOWED_CONTEXT_KEYS:
                raise ValueError(f"Campo não permitido no contexto: '{k}'")

        _validate_no_lookahead_keys(features, "features_at_t")
        _validate_no_lookahead_keys(context, "context_at_t")

        return cls(
            record_id=data.get("record_id", ""),
            versions=versions,
            provenance=provenance,
            quality=quality,
            features_at_t=features,
            context_at_t=context,
            outcomes_future=dict(data.get("outcomes_future", {})),
        )

    @classmethod
    def from_json(cls, json_str: str) -> EffortResponseShadowRecord:
        data = json.loads(json_str)
        return cls.from_dict(data)


def build_shadow_features_at_t(
    *,
    buy_notional_usd: Any,
    sell_notional_usd: Any,
    open: Any,
    high: Any,
    low: Any,
    close: Any,
    window_duration_ms: Any,
    vwap: Any = None,
    poc: Any = None,
) -> Tuple[Dict[str, Any], str, str, Dict[str, Any]]:
    """Constrói features puras no fechamento da janela t.

    Invoca diretamente compute_effort_response (P1-D), sem recalcular com
    fórmulas paralelas.
    Retorna: (features_dict, core_validity, optional_completeness, reasons).
    """
    raw_bundle = compute_effort_response(
        buy_notional_usd=buy_notional_usd,
        sell_notional_usd=sell_notional_usd,
        open=open,
        high=high,
        low=low,
        close=close,
        window_duration_ms=window_duration_ms,
        vwap=vwap,
        poc=poc,
    )

    disp_bps = raw_bundle["price_displacement_bps"]
    range_bps = raw_bundle["range_bps"]
    cfh_bps = raw_bundle["close_from_high_bps"]
    cfl_bps = raw_bundle["close_from_low_bps"]
    cvw_bps = raw_bundle["close_vs_vwap_bps"]
    cpoc_bps = raw_bundle["close_vs_poc_bps"]

    features: Dict[str, Any] = {
        "buy_notional_usd": raw_bundle["buy_notional_usd"],
        "sell_notional_usd": raw_bundle["sell_notional_usd"],
        "total_aggressive_notional_usd": raw_bundle["total_aggressive_notional_usd"],
        "net_aggressive_notional_usd": raw_bundle["net_aggressive_notional_usd"],
        "buy_share": raw_bundle["buy_share"],
        "sell_share": raw_bundle["sell_share"],
        "price_displacement_usd": raw_bundle["price_displacement_usd"],
        "price_displacement_pct": (disp_bps / 100.0) if disp_bps is not None else None,
        "price_displacement_bps": disp_bps,
        "range_usd": raw_bundle["range_usd"],
        "range_pct": (range_bps / 100.0) if range_bps is not None else None,
        "range_bps": range_bps,
        "close_from_high_usd": raw_bundle["close_from_high_usd"],
        "close_from_high_pct": (cfh_bps / 100.0) if cfh_bps is not None else None,
        "close_from_high_bps": cfh_bps,
        "close_from_low_usd": raw_bundle["close_from_low_usd"],
        "close_from_low_pct": (cfl_bps / 100.0) if cfl_bps is not None else None,
        "close_from_low_bps": cfl_bps,
        "close_vs_vwap_usd": raw_bundle["close_vs_vwap_usd"],
        "close_vs_vwap_pct": (cvw_bps / 100.0) if cvw_bps is not None else None,
        "close_vs_vwap_bps": cvw_bps,
        "close_vs_poc_usd": raw_bundle["close_vs_poc_usd"],
        "close_vs_poc_pct": (cpoc_bps / 100.0) if cpoc_bps is not None else None,
        "close_vs_poc_bps": cpoc_bps,
        "window_duration_ms": raw_bundle["window_duration_ms"],
    }

    _validate_no_lookahead_keys(features, "features_at_t")
    return (
        features,
        raw_bundle["core_validity"],
        raw_bundle["optional_completeness"],
        raw_bundle["reasons"],
    )


def build_shadow_context_at_t(
    *,
    symbol: str,
    window_close_ms: int,
    timestamp_utc: Optional[str] = None,
    day_of_week: Optional[int] = None,
    session_time_bucket: Optional[str] = None,
    spread: Optional[float] = None,
    bid_depth: Optional[float] = None,
    ask_depth: Optional[float] = None,
    orderbook_imbalance: Optional[float] = None,
    orderbook_imbalance_source_type: Optional[str] = None,
    flow_temporal_validity: Optional[Dict[str, str]] = None,
    realized_volatility: Optional[float] = None,
    volume_base: Optional[float] = None,
    trade_count: Optional[int] = None,
    activity_rate: Optional[float] = None,
    session_vwap_distance: Optional[float] = None,
    market_structure: Optional[str] = None,
    regime_current_at_t: Optional[str] = None,
    regime_status_at_t: Optional[str] = None,
    regime_calibration_status_at_t: Optional[str] = None,
    is_weekend: Optional[bool] = None,
    is_holiday: Optional[bool] = None,
    data_latency_ms: Optional[float] = None,
    data_freshness_ms: Optional[float] = None,
    **extra_kwargs: Any,
) -> Tuple[Dict[str, Any], List[str]]:
    """Constrói contexto contemporâneo observável no instante t.

    A fonte de verdade cronológica é SEMPRE window_close_ms (epoch ms UTC).
    timestamp_utc, day_of_week, is_weekend e session_time_bucket são derivados
    diretamente do epoch caso não sejam fornecidos.
    Campos ausentes permanecem explicitamente None (sem defaults semânticos).
    Campos fora da whitelist lançam ValueError imediatamente.
    """
    for k in extra_kwargs.keys():
        if k not in ALLOWED_CONTEXT_KEYS:
            raise ValueError(f"Campo proibido fora da whitelist no contexto: '{k}'")
    dt = datetime.fromtimestamp(window_close_ms / 1000.0, tz=timezone.utc)
    if timestamp_utc is None:
        timestamp_utc = dt.isoformat()
    if day_of_week is None:
        day_of_week = dt.weekday()
    if is_weekend is None:
        day_idx = dt.weekday()  # Monday is 0, Sunday is 6
        is_weekend = (day_idx in (5, 6))
    if session_time_bucket is None:
        hour = dt.hour
        session_time_bucket = f"{(hour // 4) * 4:02d}:00-{((hour // 4) * 4 + 4):02d}:00"

    context: Dict[str, Any] = {
        "timestamp_utc": timestamp_utc,
        "symbol": symbol.strip().upper(),
        "day_of_week": day_of_week,
        "session_time_bucket": session_time_bucket,
        "spread": spread,
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "orderbook_imbalance": orderbook_imbalance,
        "orderbook_imbalance_source_type": orderbook_imbalance_source_type,
        "flow_temporal_validity": flow_temporal_validity,
        "realized_volatility": realized_volatility,
        "volume_base": volume_base,
        "trade_count": trade_count,
        "activity_rate": activity_rate,
        "session_vwap_distance": session_vwap_distance,
        "market_structure": market_structure,
        "regime_current_at_t": regime_current_at_t,
        "regime_status_at_t": regime_status_at_t,
        "regime_calibration_status_at_t": regime_calibration_status_at_t,
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "data_latency_ms": data_latency_ms,
        "data_freshness_ms": data_freshness_ms,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
    }

    # Validação de whitelist
    for k in context.keys():
        if k not in ALLOWED_CONTEXT_KEYS:
            raise ValueError(f"Campo proibido fora da whitelist no contexto: '{k}'")

    _validate_no_lookahead_keys(context, "context_at_t")

    missing = [k for k, v in context.items() if v is None]
    return context, missing


def build_shadow_record(
    *,
    symbol: str,
    window_open_ms: int,
    window_close_ms: int,
    window_data: Dict[str, Any],
    causal_anchor_ms: Optional[int] = None,
    observation_open_ms: Optional[int] = None,
    observation_close_ms: Optional[int] = None,
    context_data: Optional[Dict[str, Any]] = None,
    source_event_id: Optional[str] = None,
    orderbook_source_type: Optional[str] = None,
    orderbook_snapshot_ms: Optional[int] = None,
    flow_window_validity: Optional[str] = None,
    latency_ms: Optional[float] = None,
    freshness_ms: Optional[float] = None,
) -> EffortResponseShadowRecord:
    """Constrói registro contemporâneo no fechamento da janela t.

    Evolução P1-F:
    - Ancoragem causal estrita em causal_anchor_ms (window_end_ms lógico).
    - Preserva window_open_ms e window_close_ms para compatibilidade e auditoria física.
    - Outcomes futuros são inicializados como PENDING indexados a partir de causal_anchor_ms.
    """
    anchor_base = causal_anchor_ms if causal_anchor_ms is not None else window_close_ms
    obs_open = observation_open_ms if observation_open_ms is not None else window_open_ms
    obs_close = observation_close_ms if observation_close_ms is not None else window_close_ms

    record_id = build_deterministic_record_id(
        symbol=symbol,
        causal_anchor_ms=anchor_base,
    )

    features, core_val, opt_comp, _reasons = build_shadow_features_at_t(
        buy_notional_usd=window_data.get("buy_notional_usd"),
        sell_notional_usd=window_data.get("sell_notional_usd"),
        open=window_data.get("open"),
        high=window_data.get("high"),
        low=window_data.get("low"),
        close=window_data.get("close"),
        window_duration_ms=window_data.get("window_duration_ms", window_close_ms - window_open_ms),
        vwap=window_data.get("vwap"),
        poc=window_data.get("poc"),
    )

    ctx_in = dict(context_data or {})
    ctx_in["symbol"] = symbol
    ctx_in["window_close_ms"] = obs_close
    context, missing_context = build_shadow_context_at_t(**ctx_in)

    provenance = ShadowProvenance(
        exchange="binance_futures",
        stream="aggTrade",
        symbol=symbol.strip().upper(),
        window_open_ms=int(window_open_ms),
        window_close_ms=int(window_close_ms),
        source_event_id=source_event_id,
        orderbook_source_type=orderbook_source_type,
        orderbook_snapshot_ms=orderbook_snapshot_ms,
        observation_open_ms=int(obs_open),
        observation_close_ms=int(obs_close),
        causal_anchor_ms=int(anchor_base),
    )

    quality = ShadowQuality(
        core_validity=core_val,
        optional_completeness=opt_comp,
        flow_window_validity=flow_window_validity,
        latency_ms=latency_ms,
        freshness_ms=freshness_ms,
        missing_context_fields=missing_context,
    )

    outcomes_future: Dict[str, Any] = {
        "status": "PENDING",
        "horizons": {
            "1m": HorizonOutcome(
                horizon="1m",
                status="PENDING",
                target_timestamp_ms=anchor_base + 60_000,
                horizon_duration_ms=60_000,
            ).to_dict(),
            "5m": HorizonOutcome(
                horizon="5m",
                status="PENDING",
                target_timestamp_ms=anchor_base + 300_000,
                horizon_duration_ms=300_000,
            ).to_dict(),
            "15m": HorizonOutcome(
                horizon="15m",
                status="PENDING",
                target_timestamp_ms=anchor_base + 900_000,
                horizon_duration_ms=900_000,
            ).to_dict(),
        },
    }

    record = EffortResponseShadowRecord(
        record_id=record_id,
        versions=ShadowVersions(),
        provenance=provenance,
        quality=quality,
        features_at_t=features,
        context_at_t=context,
        outcomes_future=outcomes_future,
    )

    _validate_no_secrets(record.to_dict())
    return record


_HORIZON_MS: Dict[str, int] = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
}


def build_shadow_outcomes(
    *,
    close_price_at_t: float,
    window_close_ms: int,
    future_observations: Sequence[Dict[str, Any]],
    horizons: Sequence[str] = ("1m", "5m", "15m"),
    policy: str = "FIRST_ON_OR_AFTER",
) -> Dict[str, Any]:
    """Calcula outcomes futuros estritamente posteriores a window_close_ms.

    CONTRATO TEMPORAL ESTRITO:
    - Excursões (max_excursion_up_bps, max_excursion_down_bps):
      Utilizam EXCLUSIVAMENTE observações com window_close_ms < timestamp <= target_timestamp_ms.
      Nenhuma observação com timestamp <= window_close_ms entra no cálculo!
    - future_price:
      - Se policy="FIRST_ON_OR_AFTER" (padrão OutcomeTracker): primeira observação com
        target_ms <= timestamp <= target_ms + OUTCOME_BOUNDARY_TOLERANCE_MS.
      - Se policy="LAST_ON_OR_BEFORE": última observação com
        target_ms - OUTCOME_BOUNDARY_TOLERANCE_MS <= timestamp <= target_ms.
      - Registra explicitamente target_timestamp_ms, observed_price_timestamp_ms e timing_error_ms.
      - Se nenhuma observação aceitável existir dentro da tolerância de 1000ms:
        status permanece PENDING (se tempo futuro insuficiente) ou INSUFFICIENT_DATA (se boundary perdido).
    """
    if not math.isfinite(close_price_at_t) or close_price_at_t <= 0:
        return {
            "status": "INVALID_BASE_PRICE",
            "horizons": {
                h: HorizonOutcome(
                    horizon=h,
                    status="INVALID_BASE_PRICE",
                    target_timestamp_ms=window_close_ms + _HORIZON_MS.get(h, 0),
                ).to_dict()
                for h in horizons
            },
        }

    outcomes: Dict[str, Any] = {
        "status": "RESOLVED",
        "horizons": {},
    }

    sorted_obs = sorted(future_observations, key=lambda x: int(x.get("timestamp_ms", 0)))
    max_available_ts = max((int(obs.get("timestamp_ms", 0)) for obs in sorted_obs), default=0)

    for h in horizons:
        horizon_duration = _HORIZON_MS.get(h)
        if horizon_duration is None:
            continue

        target_ms = window_close_ms + horizon_duration

        # 1. Checagem de maturidade cronológica
        if max_available_ts < target_ms:
            outcomes["horizons"][h] = HorizonOutcome(
                horizon=h,
                status="PENDING",
                target_timestamp_ms=target_ms,
                horizon_duration_ms=horizon_duration,
            ).to_dict()
            outcomes["status"] = "PARTIALLY_RESOLVED" if outcomes["status"] == "RESOLVED" else outcomes["status"]
            continue

        # 2. Excursões: estritamente window_close_ms < timestamp <= target_ms
        excursion_obs = [
            obs for obs in sorted_obs
            if window_close_ms < int(obs.get("timestamp_ms", 0)) <= target_ms
        ]

        if not excursion_obs:
            outcomes["horizons"][h] = HorizonOutcome(
                horizon=h,
                status="INSUFFICIENT_DATA",
                target_timestamp_ms=target_ms,
                horizon_duration_ms=horizon_duration,
            ).to_dict()
            continue

        highs: List[float] = []
        lows: List[float] = []
        for obs in excursion_obs:
            if "high" in obs and "low" in obs:
                hv = float(obs["high"])
                lv = float(obs["low"])
                if math.isfinite(hv) and math.isfinite(lv):
                    highs.append(hv)
                    lows.append(lv)
            elif "price" in obs:
                pv = float(obs["price"])
                if math.isfinite(pv):
                    highs.append(pv)
                    lows.append(pv)

        if not highs or not lows:
            outcomes["horizons"][h] = HorizonOutcome(
                horizon=h,
                status="INSUFFICIENT_DATA",
                target_timestamp_ms=target_ms,
                horizon_duration_ms=horizon_duration,
            ).to_dict()
            continue

        max_high = max(highs)
        min_low = min(lows)
        max_up_bps = (max_high - close_price_at_t) / close_price_at_t * 10000.0
        max_down_bps = (min_low - close_price_at_t) / close_price_at_t * 10000.0

        # 3. Determinação explícita de future_price com tolerância auditada
        future_price: Optional[float] = None
        obs_price_ts: Optional[int] = None
        timing_error: Optional[int] = None

        if policy == "FIRST_ON_OR_AFTER":
            # Primeiro preço observado >= target_ms dentro de [target_ms, target_ms + 1000]
            candidates = [
                obs for obs in sorted_obs
                if target_ms <= int(obs.get("timestamp_ms", 0)) <= (target_ms + OUTCOME_BOUNDARY_TOLERANCE_MS)
            ]
            if candidates:
                cand = candidates[0]
                obs_price_ts = int(cand.get("timestamp_ms", 0))
                future_price = float(cand.get("close", cand.get("price", 0.0)))
                timing_error = obs_price_ts - target_ms
        elif policy == "LAST_ON_OR_BEFORE":
            # Último preço observado <= target_ms dentro de [target_ms - 1000, target_ms]
            candidates = [
                obs for obs in excursion_obs
                if (target_ms - OUTCOME_BOUNDARY_TOLERANCE_MS) <= int(obs.get("timestamp_ms", 0)) <= target_ms
            ]
            if candidates:
                cand = candidates[-1]
                obs_price_ts = int(cand.get("timestamp_ms", 0))
                future_price = float(cand.get("close", cand.get("price", 0.0)))
                timing_error = obs_price_ts - target_ms
        else:
            raise ValueError(f"Política de preço futuro desconhecida: {policy}")

        # Se não encontrou preço elegível dentro do boundary auditado
        if future_price is None or not math.isfinite(future_price):
            outcomes["horizons"][h] = HorizonOutcome(
                horizon=h,
                status="INSUFFICIENT_DATA",
                target_timestamp_ms=target_ms,
                max_excursion_up_bps=max_up_bps,
                max_excursion_down_bps=max_down_bps,
                mfe_bps=max_up_bps,
                mae_bps=max_down_bps,
                max_high=max_high,
                min_low=min_low,
                observation_count=len(excursion_obs),
                horizon_duration_ms=horizon_duration,
            ).to_dict()
            continue

        ret_bps = (future_price - close_price_at_t) / close_price_at_t * 10000.0

        outcomes["horizons"][h] = HorizonOutcome(
            horizon=h,
            status="RESOLVED",
            target_timestamp_ms=target_ms,
            observed_price_timestamp_ms=obs_price_ts,
            timing_error_ms=timing_error,
            future_price=future_price,
            return_bps=ret_bps,
            max_excursion_up_bps=max_up_bps,
            max_excursion_down_bps=max_down_bps,
            mfe_bps=max_up_bps,
            mae_bps=max_down_bps,
            max_high=max_high,
            min_low=min_low,
            observation_count=len(excursion_obs),
            horizon_duration_ms=horizon_duration,
            resolved_at_ms=obs_price_ts,
        ).to_dict()

    all_resolved = all(h_dict.get("status") == "RESOLVED" for h_dict in outcomes["horizons"].values())
    any_resolved = any(h_dict.get("status") == "RESOLVED" for h_dict in outcomes["horizons"].values())
    if all_resolved:
        outcomes["status"] = "RESOLVED"
    elif any_resolved:
        outcomes["status"] = "PARTIALLY_RESOLVED"
    else:
        outcomes["status"] = "PENDING"

    return outcomes


def attach_shadow_outcomes(
    record: EffortResponseShadowRecord,
    outcomes: Dict[str, Any],
) -> EffortResponseShadowRecord:
    """Anexa outcomes futuros ao registro shadow de forma idempotente."""
    _validate_no_secrets(outcomes)
    new_outcomes = dict(record.outcomes_future)
    if "status" in outcomes:
        new_outcomes["status"] = outcomes["status"]
    if "horizons" in outcomes:
        curr_horizons = dict(new_outcomes.get("horizons", {}))
        for h, data in outcomes["horizons"].items():
            curr_horizons[h] = dict(data)
        new_outcomes["horizons"] = curr_horizons

    return EffortResponseShadowRecord(
        record_id=record.record_id,
        versions=record.versions,
        provenance=record.provenance,
        quality=record.quality,
        features_at_t=dict(record.features_at_t),
        context_at_t=dict(record.context_at_t),
        outcomes_future=new_outcomes,
    )


class EffortResponseShadowStorage:
    """Armazenamento JSONL append-friendly auditável com política fail-closed.

    CONCURRENCY MODEL: SINGLE_WRITER_ONLY.
    Não é seguro para múltiplos processos gravando concorrentemente.
    """

    def __init__(self, filepath: Union[str, Path] = "dados/datasets/shadow_effort_response.jsonl"):
        self.filepath = Path(filepath)
        self._seen_record_ids: Set[str] = set()
        self._init_storage()

    def _init_storage(self) -> None:
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        if not self.filepath.exists():
            return

        lines = self._read_lines_safely()
        for idx, line in enumerate(lines, start=1):
            try:
                d = json.loads(line)
                entry_type = d.get("type")
                if entry_type == "OUTCOME_UPDATE":
                    continue
                rid = d.get("record_id")
                if rid:
                    self._seen_record_ids.add(rid)
            except Exception as e:
                # Corrupção durante inicialização
                raise StorageCorruptionError(
                    f"Erro fatal de inicialização na linha {idx} do arquivo '{self.filepath}': {e}"
                ) from e

    def _read_lines_safely(self) -> List[str]:
        """Lê linhas com política de crash-recovery na cauda."""
        if not self.filepath.exists():
            return []

        raw_bytes = self.filepath.read_bytes()
        if not raw_bytes:
            return []

        # Detecção de linha final truncada por crash
        ends_with_newline = raw_bytes.endswith(b"\n")
        lines = [line.strip() for line in raw_bytes.decode("utf-8", errors="replace").splitlines() if line.strip()]

        if not ends_with_newline and lines:
            # Testa se a última linha é JSON válido
            last_line = lines[-1]
            try:
                json.loads(last_line)
            except Exception:
                logger.warning(
                    f"Crash recovery: última linha incompleta truncada descartada em '{self.filepath}'"
                )
                lines.pop()

        return lines

    def append_record(self, record: EffortResponseShadowRecord) -> bool:
        """Adiciona novo registro ao JSONL.

        Idempotente: retorna False se o record_id já existir.
        """
        if record.record_id in self._seen_record_ids:
            return False

        line = record.to_json()
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        self._seen_record_ids.add(record.record_id)
        return True

    def update_record_outcomes(
        self,
        record_id: str,
        outcomes: Dict[str, Any],
        strict: bool = True,
    ) -> bool:
        """Grava atualização de outcomes em append-log auditável.

        Se strict=True e o record_id for desconhecido, lança StorageOrphanUpdateError.
        """
        _validate_no_secrets(outcomes)
        if strict and record_id not in self._seen_record_ids:
            raise StorageOrphanUpdateError(
                f"Tentativa de atualizar record_id desconhecido '{record_id}' em '{self.filepath}'"
            )

        update_entry = {
            "type": "OUTCOME_UPDATE",
            "record_id": record_id,
            "outcomes_future": outcomes,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        line = json.dumps(update_entry, allow_nan=False)
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        return True

    def read_records(self) -> List[EffortResponseShadowRecord]:
        """Materializa registros a partir do log JSONL.

        Fail-closed:
        - Corrupção no meio do arquivo lança StorageCorruptionError;
        - Update órfão lança StorageOrphanUpdateError;
        - Dois CREATEs com mesmo ID e dados divergentes lança StorageCorruptionError.
        """
        if not self.filepath.exists():
            return []

        records_map: Dict[str, EffortResponseShadowRecord] = {}
        lines = self._read_lines_safely()

        for idx, line in enumerate(lines, start=1):
            try:
                data = json.loads(line)
            except Exception as e:
                raise StorageCorruptionError(
                    f"Corrupção de JSON na linha {idx} de '{self.filepath}': {e}"
                ) from e

            entry_type = data.get("type")
            rid = data.get("record_id")

            if entry_type == "OUTCOME_UPDATE":
                if not rid or rid not in records_map:
                    raise StorageOrphanUpdateError(
                        f"OUTCOME_UPDATE órfão para record_id '{rid}' na linha {idx}"
                    )
                records_map[rid] = attach_shadow_outcomes(records_map[rid], data["outcomes_future"])
            else:
                if not rid:
                    raise StorageCorruptionError(f"Registro sem record_id na linha {idx}")
                rec = EffortResponseShadowRecord.from_dict(data)
                if rid in records_map:
                    # Se for CREATE repetido exatamente idêntico, aceita idempotentemente
                    if rec.to_dict() != records_map[rid].to_dict():
                        raise StorageCorruptionError(
                            f"CREATE conflitante duplicado para record_id '{rid}' na linha {idx}"
                        )
                else:
                    records_map[rid] = rec

        return list(records_map.values())
