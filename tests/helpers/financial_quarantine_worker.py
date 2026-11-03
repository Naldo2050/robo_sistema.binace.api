# tests/helpers/financial_quarantine_worker.py
"""
Processo worker autônomo para testes herméticos de concorrência multiprocesso
com FinancialQuarantine (Gate H1-B2-Q1-H).

Executado exclusivamente via subprocess em tmp_path.
"""

from __future__ import annotations

import argparse
import decimal
import json
import os
import sys
from pathlib import Path

# Adiciona raiz ao sys.path para imports herméticos
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.financial_quarantine import FinancialQuarantine
from common.financial_validator import FinancialValidationResult


def main() -> int:
    parser = argparse.ArgumentParser(description="Worker de concorrência para quarentena")
    parser.add_argument("--quarantine-dir", required=True, type=Path)
    parser.add_argument("--caller-id", required=True, type=str)
    parser.add_argument("--num-events", required=True, type=int)
    parser.add_argument("--category", default="INVALID", type=str)
    args = parser.parse_args()

    q = FinancialQuarantine(args.quarantine_dir)
    val_res = FinancialValidationResult(
        is_valid=False,
        status="INVALID",
        category="FINANCIAL_SIGNAL",
        rejection_reason="CONCURRENT_STRESS_TEST",
    )

    counts = {
        "caller_id": args.caller_id,
        "attempted": args.num_events,
        "confirmed": 0,
        "lock_failed": 0,
        "uncertain": 0,
        "other_error": 0,
    }

    for i in range(args.num_events):
        payload = {
            "worker_id": args.caller_id,
            "seq": i,
            "notional": decimal.Decimal(f"{1000 + i}.12345678"),
        }
        res = q.quarantine_event(
            payload=payload,
            validation_result=val_res,
            source_caller=args.caller_id,
            category=args.category,
        )

        if res.status == "CONFIRMED":
            counts["confirmed"] += 1
        elif res.status == "FAILED_BEFORE_WRITE" and res.error_type == "LOCK_ACQUISITION_FAILED":
            counts["lock_failed"] += 1
        elif res.status == "PERSISTENCE_UNCERTAIN":
            counts["uncertain"] += 1
        else:
            counts["other_error"] += 1

    # Emite resumo em stdout para o processo pai
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
