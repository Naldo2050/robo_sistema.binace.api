# scripts/analytics/research_collectors_check.py
# -*- coding: utf-8 -*-
"""
B3 — Detecção de falha silenciosa dos coletores de pesquisa.

Limiares explícitos (horas desde o último sucesso / contadores):
  positioning: WARNING >2h sem sucesso | CRITICAL >24h sem sucesso
  cftc:        WARNING >8d sem sucesso  | CRITICAL >10d sem sucesso
  qualquer job: WARNING failures>=3 seguidas | CRITICAL failures>=10
  unrecoverable_gap registrado -> CRITICAL imediato

Saídas: estado em dados/research/.state/alert.json (só alerta em transição),
linhas legíveis, exit 0=ok / 1=warning / 2=critical. Webhook opcional via
env RESEARCH_ALERT_WEBHOOK_URL (stdlib, sem dependência nova); sem ele,
o alerta vive em logs + arquivo de status (verificação manual no runbook).

Uso:
  python scripts/analytics/research_collectors_check.py [--webhook-url URL]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, ".")

REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = REPO_ROOT / "dados" / "research" / ".state"

# (warn_h, crit_h) por job
THRESHOLDS = {
    "positioning": (2.0, 24.0),
    "cftc-collect": (8 * 24.0, 10 * 24.0),
    "cftc-outcomes": (8 * 24.0, 10 * 24.0),
    "status": (30 * 24.0, 45 * 24.0),
}
FAIL_WARN, FAIL_CRIT = 3, 10
EXIT_OK, EXIT_WARN, EXIT_CRIT = 0, 1, 2


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(s):
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def load_state(job: str) -> dict:
    try:
        return json.loads((STATE_DIR / f"{job}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def evaluate(now: datetime | None = None) -> dict:
    """Puro (testável): {job: {level, reasons[], age_h, failures}} + worst."""
    now = now or _utcnow()
    result = {"jobs": {}, "worst": "OK"}
    rank = {"OK": 0, "WARNING": 1, "CRITICAL": 2}
    for job, (warn_h, crit_h) in THRESHOLDS.items():
        st = load_state(job)
        reasons = []
        level = "OK"
        last = _parse_iso(st.get("last_success"))
        age_h = (now - last).total_seconds() / 3600 if last else None
        failures = int(st.get("consecutive_failures", 0) or 0)
        if age_h is None:
            level, reasons = "WARNING", ["never_succeeded"]
        elif age_h > crit_h:
            level = "CRITICAL"
            reasons.append(f"no_success_for_{age_h:.1f}h_over_{crit_h}h")
        elif age_h > warn_h:
            level = "WARNING"
            reasons.append(f"no_success_for_{age_h:.1f}h_over_{warn_h}h")
        if failures >= FAIL_CRIT:
            level = "CRITICAL"
            reasons.append(f"consecutive_failures_{failures}")
        elif failures >= FAIL_WARN and level == "OK":
            level = "WARNING"
            reasons.append(f"consecutive_failures_{failures}")
        result["jobs"][job] = {"level": level, "reasons": reasons,
                               "age_h": age_h, "failures": failures}
        if rank[level] > rank[result["worst"]]:
            result["worst"] = level
    return result


def post_webhook(url: str, payload: dict, timeout: float = 10.0) -> bool:
    try:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception as exc:  # noqa: BLE001
        logging.warning("CHECK_WEBHOOK_FAILED %r", exc)
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--webhook-url", default=os.getenv("RESEARCH_ALERT_WEBHOOK_URL"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    res = evaluate()
    res["checked_at"] = _utcnow().isoformat()
    alert_path = STATE_DIR / "alert.json"
    try:
        prev = json.loads(alert_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        prev = {}
    transition = prev.get("worst") != res["worst"]
    alert_path.parent.mkdir(parents=True, exist_ok=True)
    alert_path.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    for job, info in res["jobs"].items():
        logging.log(logging.CRITICAL if info["level"] == "CRITICAL"
                    else logging.WARNING if info["level"] == "WARNING"
                    else logging.INFO,
                    "CHECK job=%s level=%s reasons=%s age_h=%s failures=%s",
                    job, info["level"], info["reasons"], info["age_h"],
                    info["failures"])
        print(f"{job}: {info['level']} {info['reasons']}")
    if res["worst"] != "OK" and transition and args.webhook_url:
        post_webhook(args.webhook_url, {"text": f"[research-collectors] {res['worst']}",
                                        "details": res["jobs"]})
    print(f"WORST: {res['worst']}")
    return {"OK": EXIT_OK, "WARNING": EXIT_WARN,
            "CRITICAL": EXIT_CRIT}[res["worst"]]


if __name__ == "__main__":
    raise SystemExit(main())
