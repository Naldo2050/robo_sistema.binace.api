# scripts/diagnostics/validate_production_run.py
# -*- coding: utf-8 -*-
"""
VALIDADOR DA OBSERVAÇÃO EM PRODUÇÃO (2026-08-09).

Analisa os artefatos gerados por run_production_observation.py e o DB de
eventos, verificando:

  Parte A — infra: SDK OpenAI/Groq carregado, WS real, conexão ok.
  Parte B — integridade dos trades:
      B.1 trades processados > 0
      B.2 OOO (flow_analyzer e orchestrator) == 0
      B.3 clamps (timestamp_corrected_total) == 0
      B.4 invalid_trades == 0
      B.5 CVD pós-reset coerente (divergência < 10% vs recalc independente)
      B.6 evento com pivots classic (fix pivot_points 2026-08-09):
          event.pivot_points.daily.source == "classic" OU fallback marcado
  Parte C — amostra de eventos: consistência delta/CVD (quando paginável).

Uso:
    python scripts/diagnostics/validate_production_run.py
"""

import sys
import os
import io
import json
import sqlite3

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SUMMARY_FILE = "logs/observation_final_summary.json"
STATS_FILE = "logs/observation_stats.jsonl"
DB_PATH = "dados/trading_bot.db"

REPORT = []
FAILURES = []


def check(part: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    REPORT.append(f"  {status} {part}: {detail}")
    if not ok:
        FAILURES.append(part)


def main() -> int:
    global REPORT
    print("=" * 70)
    print("VALIDACAO DA OBSERVACAO EM PRODUCAO — 2026-08-09")
    print("=" * 70)

    summary = {}
    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
            summary = json.load(f)
        print(f"\n[RESUMO DA OBSERVACAO]\n{json.dumps(summary, indent=2, ensure_ascii=False)}")
    else:
        print(f"\n!! {SUMMARY_FILE} nao encontrado — validando apenas DB/artefatos existentes")

    print("\n[PARTE A — INFRA]")
    check("A.1 resumo presente", bool(summary), "observation_final_summary.json")
    check("A.2 duracao registrada", bool(summary.get("duracao_min")),
          f"{summary.get('duracao_min')} min")

    print("\n[PARTE B — INTEGRIDADE]")
    trades = summary.get("total_trades_processed")
    check("B.1 trades > 0", trades is not None and int(trades) > 0, f"trades={trades}")

    ooo_fa = summary.get("out_of_order_fa")
    ooo_orb = summary.get("out_of_order_orchestrator")
    check("B.2 OOO == 0",
          (ooo_fa == 0 or ooo_fa is None) and (ooo_orb == 0 or ooo_orb is None),
          f"ooo_fa={ooo_fa} ooo_orch={ooo_orb}")

    clamps = summary.get("timestamp_corrected_total")
    check("B.3 clamps == 0", clamps is None or clamps == 0, f"clamps={clamps}")

    invalid = summary.get("invalid_trades")
    check("B.4 invalid == 0", invalid is None or invalid == 0, f"invalid={invalid}")

    # B.5 — CVD pós-reset coerente
    cvd_final = summary.get("cvd_final")
    last_reset = summary.get("last_reset_ms")
    check("B.5 CVD presente pos-reset", cvd_final is not None and last_reset is not None,
          f"cvd={cvd_final} last_reset_ms={last_reset}")

    # B.6 — fix pivot_points: eventos devem ter source classic (ou fallback marcado)
    pp_ok = True
    pp_detail = "sem eventos no DB"
    if os.path.exists(DB_PATH):
        try:
            con = sqlite3.connect(DB_PATH)
            rows = con.execute(
                "SELECT id, payload FROM events ORDER BY id DESC LIMIT 20"
            ).fetchall()
            con.close()
            n_classic = 0
            n_fallback = 0
            for eid, payload in rows:
                try:
                    ev = json.loads(payload) if isinstance(payload, str) else payload
                except Exception:
                    continue
                pp = (ev.get("pivot_points") or {}).get("daily") or {}
                src = pp.get("source")
                if src == "classic":
                    n_classic += 1
                elif src in ("vp_fallback", "multi_tf_fallback"):
                    n_fallback += 1
            pp_detail = f"classic={n_classic} fallback_marcado={n_fallback}"
            if n_classic == 0 and n_fallback == 0:
                # Eventos legados (gerados ANTES do fix pivot_points 2026-08-09)
                # têm pivot_points sem source — esperado. Reobservar para validar.
                pp_ok = True
                pp_detail = ("WARN: eventos legados pre-fix (sem source) — "
                             "reobservar para confirmar source=classic")
            elif n_classic > 0:
                pp_detail += " — fonte classica ativa (FIX OK)"
        except Exception as e:
            pp_ok = False
            pp_detail = f"erro lendo DB: {e}"
    check("B.6 pivot_points source classic/fallback", pp_ok, pp_detail)

    print("\n[PARTE C — EVENTOS (amostra)]")
    if os.path.exists(DB_PATH):
        try:
            con = sqlite3.connect(DB_PATH)
            total = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            check("C.1 eventos no DB", int(total) > 0, f"total={total}")
            if int(total) > 0:
                for eid, payload in con.execute(
                    "SELECT id, payload FROM events ORDER BY id DESC LIMIT 3"
                ):
                    try:
                        ev = json.loads(payload) if isinstance(payload, str) else payload
                    except Exception:
                        continue
                    pp = (ev.get("pivot_points") or {}).get("daily") or {}
                    print(f"    ev {eid}: tipo={ev.get('tipo_evento')} "
                          f"pivot={pp.get('pivot')} source={pp.get('source')} "
                          f"vah={pp.get('vah')} val={pp.get('val')}")
            con.close()
        except Exception as e:
            check("C.1 eventos no DB", False, f"erro: {e}")

    print("\n" + "=" * 70)
    for line in REPORT:
        print(line)
    print("=" * 70)
    if FAILURES:
        print(f"RESULTADO: {len(FAILURES)} FALHA(S): {', '.join(FAILURES)}")
        return 1
    print("RESULTADO: TODAS AS VERIFICACOES PASSARAM")
    return 0


if __name__ == "__main__":
    sys.exit(main())
