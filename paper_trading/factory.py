# paper_trading/factory.py
"""
Hermetic Factory for Shadow Paper Trading Runtime (Gate C3-C-B3-B).

Responsible for:
- Parsing environment-driven configuration safely.
- Returning explicit ShadowFactoryResult (DISABLED, RUNNING, FAILED).
- Enforcing mandatory PAPER_DB_PATH and PAPER_GIT_SHA when enabled.
- Safe lifecycle initialization with automatic cleanup on failure.
- Zero network I/O, zero credential access, zero AI coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any, Callable, Mapping, Optional

from paper_trading.config import parse_shadow_config
from paper_trading.ledger import PaperLedger
from paper_trading.shadow_runtime import ShadowPaperRuntime

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShadowFactoryResult:
    """
    Explicit result container returned by create_shadow_runtime.

    status:
        - "DISABLED": shadow runtime was explicitly or by default turned off.
        - "RUNNING": shadow runtime was successfully constructed and started.
        - "FAILED": shadow runtime configuration or initialization failed.
    """

    status: str
    runtime: Optional[ShadowPaperRuntime]
    reason: Optional[str] = None


def create_shadow_runtime(
    env: Optional[Mapping[str, str]] = None,
    *,
    clock_ms: Optional[Callable[[], int]] = None,
    risk_manager: Optional[Any] = None,
    risk_adapter: Optional[Any] = None,
    execution_sink: Optional[Any] = None,
    signal_adapter: Optional[Any] = None,
    ledger: Optional[Any] = None,
    git_sha: Optional[str] = None,
) -> ShadowFactoryResult:
    """
    Construct and start a ShadowPaperRuntime according to environment configuration.

    Guarantees:
    - Never throws unexpected exceptions to caller; returns status="FAILED" on errors.
    - Zero reads of live exchange/AI credentials (BINANCE_API_KEY, GROQ_API_KEY, etc.).
    - When PAPER_SHADOW_ENABLED is false/missing: returns DISABLED without side effects.
    - When PAPER_SHADOW_ENABLED is true: requires valid economics, DB path, and git SHA.
    - If construction or start fails, cleans up any internally created ledger resources.
    """
    env_map: Mapping[str, str] = os.environ if env is None else env

    try:
        # 1. Parse configuration safely through hermetic parser
        config_result = parse_shadow_config(env_map)
        if not config_result.is_valid or config_result.config is None:
            return ShadowFactoryResult(
                status="FAILED",
                runtime=None,
                reason=config_result.error or "Invalid shadow configuration",
            )

        config = config_result.config

        # 2. If disabled, return immediately without opening files or creating threads
        if not config.enabled:
            return ShadowFactoryResult(
                status="DISABLED",
                runtime=None,
                reason=None,
            )

        # 3. Real runtime validation: explicit DB path and git SHA are mandatory
        db_path = env_map.get("PAPER_DB_PATH")
        if ledger is None and (not db_path or not db_path.strip()):
            return ShadowFactoryResult(
                status="FAILED",
                runtime=None,
                reason="PAPER_DB_PATH is required when shadow paper trading is enabled",
            )

        resolved_git_sha = git_sha or env_map.get("PAPER_GIT_SHA")
        if not resolved_git_sha or not resolved_git_sha.strip():
            return ShadowFactoryResult(
                status="FAILED",
                runtime=None,
                reason="PAPER_GIT_SHA is required when shadow paper trading is enabled",
            )
        resolved_git_sha = resolved_git_sha.strip()

        # 4. Instantiate or adopt ledger
        created_ledger: Optional[PaperLedger] = None
        ledger_to_use = ledger
        owns_ledger = False

        if ledger_to_use is None and db_path is not None:
            try:
                created_ledger = PaperLedger(db_path=db_path.strip())
                ledger_to_use = created_ledger
                owns_ledger = True
            except Exception as ledger_exc:
                return ShadowFactoryResult(
                    status="FAILED",
                    runtime=None,
                    reason=f"Failed to create PaperLedger at {db_path!r}: {ledger_exc}",
                )

        # 5. Build runtime and execute strict start lifecycle
        try:
            runtime = ShadowPaperRuntime(
                config=config,
                clock_ms=clock_ms,
                risk_manager=risk_manager,
                risk_adapter=risk_adapter,
                execution_sink=execution_sink,
                signal_adapter=signal_adapter,
                ledger=ledger_to_use,
                git_sha=resolved_git_sha,
                ledger_is_owner=owns_ledger,
            )

            started = runtime.start()
            if not started:
                # Start failed (e.g. duplicate cohort or unhealthy persistence)
                if created_ledger is not None:
                    try:
                        created_ledger.close()
                    except Exception:
                        pass
                return ShadowFactoryResult(
                    status="FAILED",
                    runtime=None,
                    reason="ShadowPaperRuntime.start() returned False (cohort creation or persistence check failed)",
                )

            return ShadowFactoryResult(
                status="RUNNING",
                runtime=runtime,
                reason=None,
            )

        except Exception as runtime_exc:
            if created_ledger is not None:
                try:
                    created_ledger.close()
                except Exception:
                    pass
            return ShadowFactoryResult(
                status="FAILED",
                runtime=None,
                reason=f"ShadowPaperRuntime initialization error: {runtime_exc}",
            )

    except Exception as unexpected_exc:
        logger.error(f"Unexpected error in create_shadow_runtime: {unexpected_exc}", exc_info=True)
        return ShadowFactoryResult(
            status="FAILED",
            runtime=None,
            reason=f"Unexpected error: {unexpected_exc}",
        )
