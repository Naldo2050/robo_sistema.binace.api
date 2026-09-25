# paper_trading/config.py
"""
Hermetic configuration and validation for Shadow Paper Trading (Gate C3-C-B1).

Enforces explicit, fail-closed configuration without silent economic defaults.
Zero credential reads, zero network I/O, zero AI coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import re
from typing import Any, Mapping, Optional, Tuple, Literal

from paper_trading.contracts import PaperCostConfig
from paper_trading.cost_model import DEFAULT_PAPER_COST_CONFIG

ShadowProviderType = Literal["fixed_long", "fixed_short", "seeded_random"]
ALLOWED_SHADOW_PROVIDERS: Tuple[ShadowProviderType, ...] = (
    "fixed_long",
    "fixed_short",
    "seeded_random",
)

ALLOWED_TIMEFRAMES: Tuple[str, ...] = (
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
)

COHORT_ID_REGEX = re.compile(r"^[A-Za-z0-9_.-]{3,64}$")

TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})
FALSE_TOKENS = frozenset({"0", "false", "no", "off"})


@dataclass(frozen=True)
class ShadowPaperConfig:
    """
    Immutable configuration for Shadow Paper Trading runtime.

    When enabled is False, economic and operational parameters are optional.
    When enabled is True, all operational parameters must be explicitly specified and valid.
    """

    enabled: bool = False
    cohort_id: Optional[str] = None
    mode: str = "BASELINE"
    provider: Optional[ShadowProviderType] = None
    random_seed: Optional[int] = None
    symbol: str = "BTCUSDT"
    timeframe: str = "1m"
    notional_usdt: Optional[float] = None
    horizon_s: Optional[int] = None
    order_ttl_ms: Optional[int] = None
    maker_fee_bps: Optional[float] = None
    taker_fee_bps: Optional[float] = None
    entry_slippage_bps: Optional[float] = None
    exit_slippage_bps: Optional[float] = None
    cost_source: Optional[str] = None
    cost_effective_at: Optional[str] = None
    strategy_version: str = "c3_shadow_v1.0.0"

    def to_cost_config(self) -> PaperCostConfig:
        """
        Convert configured cost parameters to an immutable PaperCostConfig contract.

        Returns DEFAULT_PAPER_COST_CONFIG if runtime is disabled or if costs are unset.
        """
        if not self.enabled:
            return DEFAULT_PAPER_COST_CONFIG

        if (
            self.maker_fee_bps is None
            or self.taker_fee_bps is None
            or self.entry_slippage_bps is None
            or self.exit_slippage_bps is None
            or self.cost_source is None
            or self.cost_effective_at is None
        ):
            return DEFAULT_PAPER_COST_CONFIG

        return PaperCostConfig(
            maker_fee_bps=self.maker_fee_bps,
            taker_fee_bps=self.taker_fee_bps,
            entry_slippage_bps=self.entry_slippage_bps,
            exit_slippage_bps=self.exit_slippage_bps,
            source=self.cost_source,
            effective_at=self.cost_effective_at,
        )


@dataclass(frozen=True)
class ShadowConfigResult:
    """
    Explicit result container for parsing and validating ShadowPaperConfig.

    Allows runtime owners to fail-closed on paper trading without crashing the bot.
    """

    config: Optional[ShadowPaperConfig]
    error: Optional[str]
    is_valid: bool


def _parse_bool(raw: Optional[str], default: bool = False) -> Tuple[Optional[bool], Optional[str]]:
    if raw is None:
        return default, None
    token = raw.strip().lower()
    if token in TRUE_TOKENS:
        return True, None
    if token in FALSE_TOKENS:
        return False, None
    return None, f"Invalid boolean value: {raw!r}. Must be one of {sorted(TRUE_TOKENS)} or {sorted(FALSE_TOKENS)}"


def _parse_int(raw: Any, field_name: str) -> Tuple[Optional[int], Optional[str]]:
    if raw is None:
        return None, f"{field_name} is required"
    if isinstance(raw, bool):
        return None, f"{field_name} must be an integer, got bool: {raw}"
    if isinstance(raw, int):
        return raw, None
    if isinstance(raw, str):
        try:
            val = int(raw.strip())
            return val, None
        except (ValueError, TypeError):
            return None, f"{field_name} must be an integer, got {raw!r}"
    return None, f"{field_name} must be an integer, got {type(raw).__name__}"


def _parse_float(raw: Any, field_name: str) -> Tuple[Optional[float], Optional[str]]:
    if raw is None:
        return None, f"{field_name} is required"
    if isinstance(raw, bool):
        return None, f"{field_name} must be a float, got bool: {raw}"
    if isinstance(raw, (int, float)):
        val = float(raw)
        if not math.isfinite(val):
            return None, f"{field_name} must be finite, got {val}"
        return val, None
    if isinstance(raw, str):
        try:
            val = float(raw.strip())
            if not math.isfinite(val):
                return None, f"{field_name} must be finite, got {raw!r}"
            return val, None
        except (ValueError, TypeError):
            return None, f"{field_name} must be a valid float, got {raw!r}"
    return None, f"{field_name} must be a float, got {type(raw).__name__}"


def _parse_iso_timezone(raw: Optional[str], field_name: str) -> Tuple[Optional[str], Optional[str]]:
    if not raw or not isinstance(raw, str) or not raw.strip():
        return None, f"{field_name} is required and cannot be empty"
    val = raw.strip()
    try:
        normalized_iso = val.replace("Z", "+00:00") if val.endswith("Z") else val
        dt = datetime.fromisoformat(normalized_iso)
        if dt.tzinfo is None:
            return None, f"{field_name} must be timezone-aware ISO-8601 string, got naive: {val!r}"
        return val, None
    except Exception as exc:
        return None, f"{field_name} invalid ISO-8601: {val!r} ({exc})"


def parse_shadow_config(env: Mapping[str, str]) -> ShadowConfigResult:
    """
    Parse and validate ShadowPaperConfig from an explicit environment mapping.

    Guarantees:
    - Never reads credentials (BINANCE_API_KEY, GROQ_API_KEY, etc.).
    - When PAPER_SHADOW_ENABLED is false/missing: returns valid disabled config.
    - When PAPER_SHADOW_ENABLED is true: strictly requires all economic and operational
      parameters without silent defaults.
    """
    raw_enabled = env.get("PAPER_SHADOW_ENABLED")
    enabled, bool_err = _parse_bool(raw_enabled, default=False)
    if bool_err is not None:
        return ShadowConfigResult(config=None, error=bool_err, is_valid=False)

    if not enabled:
        # OFF means OFF: no requirements for cohort, seed, costs, etc.
        return ShadowConfigResult(
            config=ShadowPaperConfig(enabled=False),
            error=None,
            is_valid=True,
        )

    # 1. Symbol and Timeframe
    raw_symbol = env.get("PAPER_SYMBOL", "BTCUSDT").strip().upper()
    if not raw_symbol:
        return ShadowConfigResult(config=None, error="PAPER_SYMBOL cannot be empty", is_valid=False)

    raw_timeframe = env.get("PAPER_TIMEFRAME", "1m").strip().lower()
    if raw_timeframe not in ALLOWED_TIMEFRAMES:
        return ShadowConfigResult(
            config=None,
            error=f"PAPER_TIMEFRAME {raw_timeframe!r} not in allowed timeframes: {ALLOWED_TIMEFRAMES}",
            is_valid=False,
        )

    # 2. Cohort ID
    raw_cohort = env.get("PAPER_COHORT_ID")
    if not raw_cohort or not isinstance(raw_cohort, str) or not raw_cohort.strip():
        return ShadowConfigResult(config=None, error="PAPER_COHORT_ID is required when enabled", is_valid=False)
    cohort_id = raw_cohort.strip()
    if not COHORT_ID_REGEX.match(cohort_id):
        return ShadowConfigResult(
            config=None,
            error=f"PAPER_COHORT_ID {cohort_id!r} invalid: must be 3-64 chars [A-Za-z0-9_.-]",
            is_valid=False,
        )

    # 3. Mode, Provider and Seed
    raw_follow = env.get("FOLLOW_SIGNAL")
    follow_signal, _ = _parse_bool(raw_follow, default=False)
    raw_mode = env.get("PAPER_MODE")

    mode: str
    provider: Optional[ShadowProviderType] = None
    random_seed: Optional[int] = None

    if follow_signal is True or (raw_mode and raw_mode.strip() == "FOLLOW_SIGNAL"):
        mode = "FOLLOW_SIGNAL"
        provider = None
        random_seed = None
    else:
        mode = "BASELINE"
        raw_provider = env.get("PAPER_PROVIDER")
        if not raw_provider or raw_provider.strip() not in ALLOWED_SHADOW_PROVIDERS:
            return ShadowConfigResult(
                config=None,
                error=f"PAPER_PROVIDER must be one of {ALLOWED_SHADOW_PROVIDERS}, got {raw_provider!r}",
                is_valid=False,
            )

        from typing import cast

        provider = cast(ShadowProviderType, raw_provider.strip())

        if provider == "seeded_random":
            raw_seed = env.get("PAPER_RANDOM_SEED")
            random_seed, seed_err = _parse_int(raw_seed, "PAPER_RANDOM_SEED")
            if seed_err is not None:
                return ShadowConfigResult(config=None, error=seed_err, is_valid=False)
        else:
            # fixed_long/fixed_short: seed is optional/ignored, but if present must not be malformed
            raw_seed = env.get("PAPER_RANDOM_SEED")
            if raw_seed is not None and raw_seed.strip():
                parsed_seed, seed_err = _parse_int(raw_seed, "PAPER_RANDOM_SEED")
                if seed_err is not None:
                    return ShadowConfigResult(config=None, error=seed_err, is_valid=False)
                random_seed = parsed_seed

    # 4. Notional, Horizon, TTL
    notional_usdt, notional_err = _parse_float(env.get("PAPER_NOTIONAL_USDT"), "PAPER_NOTIONAL_USDT")
    if notional_err is not None:
        return ShadowConfigResult(config=None, error=notional_err, is_valid=False)
    if notional_usdt is None or notional_usdt <= 0:
        return ShadowConfigResult(config=None, error=f"PAPER_NOTIONAL_USDT must be > 0, got {notional_usdt}", is_valid=False)

    horizon_s, horizon_err = _parse_int(env.get("PAPER_HORIZON_S"), "PAPER_HORIZON_S")
    if horizon_err is not None:
        return ShadowConfigResult(config=None, error=horizon_err, is_valid=False)
    if horizon_s is None or horizon_s <= 0:
        return ShadowConfigResult(config=None, error=f"PAPER_HORIZON_S must be > 0, got {horizon_s}", is_valid=False)

    order_ttl_ms, ttl_err = _parse_int(env.get("PAPER_ORDER_TTL_MS"), "PAPER_ORDER_TTL_MS")
    if ttl_err is not None:
        return ShadowConfigResult(config=None, error=ttl_err, is_valid=False)
    if order_ttl_ms is None or order_ttl_ms <= 0:
        return ShadowConfigResult(config=None, error=f"PAPER_ORDER_TTL_MS must be > 0, got {order_ttl_ms}", is_valid=False)

    # 5. Cost Parameters
    maker_fee_bps, maker_err = _parse_float(env.get("PAPER_MAKER_FEE_BPS"), "PAPER_MAKER_FEE_BPS")
    if maker_err is not None:
        return ShadowConfigResult(config=None, error=maker_err, is_valid=False)
    if maker_fee_bps is None or maker_fee_bps < 0:
        return ShadowConfigResult(config=None, error=f"PAPER_MAKER_FEE_BPS must be >= 0, got {maker_fee_bps}", is_valid=False)

    taker_fee_bps, taker_err = _parse_float(env.get("PAPER_TAKER_FEE_BPS"), "PAPER_TAKER_FEE_BPS")
    if taker_err is not None:
        return ShadowConfigResult(config=None, error=taker_err, is_valid=False)
    if taker_fee_bps is None or taker_fee_bps < 0:
        return ShadowConfigResult(config=None, error=f"PAPER_TAKER_FEE_BPS must be >= 0, got {taker_fee_bps}", is_valid=False)

    entry_slippage_bps, entry_slip_err = _parse_float(env.get("PAPER_ENTRY_SLIPPAGE_BPS"), "PAPER_ENTRY_SLIPPAGE_BPS")
    if entry_slip_err is not None:
        return ShadowConfigResult(config=None, error=entry_slip_err, is_valid=False)
    if entry_slippage_bps is None or entry_slippage_bps < 0:
        return ShadowConfigResult(config=None, error=f"PAPER_ENTRY_SLIPPAGE_BPS must be >= 0, got {entry_slippage_bps}", is_valid=False)

    exit_slippage_bps, exit_slip_err = _parse_float(env.get("PAPER_EXIT_SLIPPAGE_BPS"), "PAPER_EXIT_SLIPPAGE_BPS")
    if exit_slip_err is not None:
        return ShadowConfigResult(config=None, error=exit_slip_err, is_valid=False)
    if exit_slippage_bps is None or exit_slippage_bps < 0:
        return ShadowConfigResult(config=None, error=f"PAPER_EXIT_SLIPPAGE_BPS must be >= 0, got {exit_slippage_bps}", is_valid=False)

    cost_source = env.get("PAPER_COST_SOURCE")
    if not cost_source or not isinstance(cost_source, str) or not cost_source.strip():
        return ShadowConfigResult(config=None, error="PAPER_COST_SOURCE is required and cannot be empty", is_valid=False)

    cost_effective_at, eff_err = _parse_iso_timezone(env.get("PAPER_COST_EFFECTIVE_AT"), "PAPER_COST_EFFECTIVE_AT")
    if eff_err is not None:
        return ShadowConfigResult(config=None, error=eff_err, is_valid=False)

    # 6. Strategy Version
    strategy_version = env.get("PAPER_STRATEGY_VERSION", "c3_shadow_v1.0.0").strip()
    if not strategy_version:
        return ShadowConfigResult(config=None, error="PAPER_STRATEGY_VERSION cannot be empty", is_valid=False)

    config = ShadowPaperConfig(
        enabled=True,
        cohort_id=cohort_id,
        mode=mode,
        provider=provider,
        random_seed=random_seed,
        symbol=raw_symbol,
        timeframe=raw_timeframe,
        notional_usdt=notional_usdt,
        horizon_s=horizon_s,
        order_ttl_ms=order_ttl_ms,
        maker_fee_bps=maker_fee_bps,
        taker_fee_bps=taker_fee_bps,
        entry_slippage_bps=entry_slippage_bps,
        exit_slippage_bps=exit_slippage_bps,
        cost_source=cost_source.strip(),
        cost_effective_at=cost_effective_at,
        strategy_version=strategy_version,
    )
    return ShadowConfigResult(config=config, error=None, is_valid=True)
