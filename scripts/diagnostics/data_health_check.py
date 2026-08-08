#!/usr/bin/env python3
"""
data_health_check.py — Diagnóstico de saúde dos dados persistidos do bot.

Checks:
  1. PRICE CONSISTENCY: preco_fechamento vs orderbook_data.mid vs
     multi_tf.<tf>.preco_atual (15m/1h/4h/1d). Alerta se divergência > 0.05%.
  2. VALUE AREA DEGENERADA: historical_vp.daily/weekly/monthly com
     val < poc < vah violado (val == poc ou vah == poc ou fora do intervalo).
  3. FALLBACK TRACKING: adaptive_thresholds.current_volatility == 0.03 exato
     (valor de fallback do data_enricher). Conta janelas seguidas desde o boot
     e identifica quando passa a ser valor calculado.
  4. Resumo agregado das últimas N janelas.

Fonte primária: dados/eventos_visuais.log (dump completo com raw_event).
Fallback: dados/eventos_fluxo.jsonl (pode ter linhas compactadas).

Uso:
    python scripts/diagnostics/data_health_check.py [--janelas N] [--alerta-pct P]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

if sys.stdout and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

FALLBACK_VOL = 0.03  # data_processing/data_enricher.py:707
ALERTA_PCT_DEFAULT = 0.05
PERIODOS_VP = ["daily", "weekly", "monthly"]
TFS = ["15m", "1h", "4h", "1d"]

CAMINHO_VISUAIS = os.path.join(REPO_DIR, "dados", "eventos_visuais.log")
CAMINHO_JSONL = os.path.join(REPO_DIR, "dados", "eventos_fluxo.jsonl")


# ─────────────────────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json_entre(texto: str) -> dict | None:
    ini = texto.find("{")
    fim = texto.rfind("}")
    if ini == -1 or fim == -1:
        return None
    try:
        return json.loads(texto[ini : fim + 1])
    except json.JSONDecodeError:
        return None


def ler_eventos_visuais(caminho: str) -> list[dict]:
    """Lê o dump visual: blocos separados por '----' com linha EVENTO: ..."""
    with open(caminho, encoding="utf-8", errors="replace") as f:
        linhas = f.readlines()
    evs = [i for i, l in enumerate(linhas) if l.startswith("EVENTO:")]
    objetos = []
    for k in range(len(evs)):
        j = evs[k + 1] if k + 1 < len(evs) else len(linhas)
        obj = _parse_json_entre("".join(linhas[evs[k] + 1 : j]))
        if obj is not None:
            objetos.append(obj)
    return objetos


def ler_eventos_jsonl(caminho: str) -> list[dict]:
    with open(caminho, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def carregar_eventos(fonte: str | None, max_janelas: int) -> tuple[list[dict], str]:
    """Retorna (eventos ANALYSIS_TRIGGER completos, fonte usada)."""
    caminho = None
    if fonte == "jsonl":
        if os.path.exists(CAMINHO_JSONL):
            caminho = CAMINHO_JSONL
    elif fonte == "visuais":
        if os.path.exists(CAMINHO_VISUAIS):
            caminho = CAMINHO_VISUAIS
    else:  # auto
        for p in (CAMINHO_VISUAIS, CAMINHO_JSONL):
            if os.path.exists(p):
                caminho = p
                break

    if caminho is None:
        print(f"❌ Nenhuma fonte encontrada (visuais em {CAMINHO_VISUAIS}, jsonl em {CAMINHO_JSONL})")
        sys.exit(1)

    if caminho == CAMINHO_JSONL:
        eventos = ler_eventos_jsonl(caminho)
    else:
        eventos = ler_eventos_visuais(caminho)

    completos = [
        e
        for e in eventos
        if e.get("tipo_evento") == "ANALYSIS_TRIGGER"
        and e.get("epoch_ms")
        and e.get("preco_fechamento") is not None
    ]
    completos.sort(key=lambda e: e.get("epoch_ms", 0))
    completos = completos[-max_janelas:]

    nome_fonte = "dados/eventos_visuais.log" if caminho == CAMINHO_VISUAIS else "dados/eventos_fluxo.jsonl"
    print(f"📄 Fonte: {nome_fonte} | eventos ANALYSIS_TRIGGER lidos: {len(eventos)} | últimos {len(completos)} janelas")
    return completos, nome_fonte


# ─────────────────────────────────────────────────────────────────────────────
# Check 1 — PRICE CONSISTENCY
# ─────────────────────────────────────────────────────────────────────────────

def _num(v) -> float | None:
    if isinstance(v, (int, float)) and v > 0:
        return float(v)
    return None


def check_price_consistency(eventos: list[dict], alerta_pct: float) -> tuple[list[dict], int]:
    """
    Para cada evento compara preco_fechamento vs orderbook_data.mid vs
    multi_tf.*.preco_atual. Divergência relativa ao preco_fechamento.
    """
    divergencias: list[dict] = []
    eventos_alertados = 0
    for e in eventos:
        ref = _num(e.get("preco_fechamento"))
        if ref is None:
            continue
        comparacoes: list[tuple[str, float | None]] = [("orderbook.mid", _num(e.get("orderbook_data", {}).get("mid")))]
        for tf in TFS:
            mtf = e.get("multi_tf", {}).get(tf, {})
            if isinstance(mtf, dict):
                comparacoes.append((f"multi_tf.{tf}", _num(mtf.get("preco_atual"))))

        linhas = {}
        for nome, valor in comparacoes:
            if valor is None:
                linhas[nome] = None
                continue
            pct = abs(valor - ref) / ref * 100.0
            linhas[nome] = round(pct, 4)
        divergencias.append(
            {
                "janela": e.get("janela_numero"),
                "epoch_ms": e.get("epoch_ms"),
                "preco_fechamento": ref,
                "pcts": linhas,
                "alerta": any(v is not None and v > alerta_pct for v in linhas.values()),
            }
        )
        if divergencias[-1]["alerta"]:
            eventos_alertados += 1

    print("\n" + "=" * 72)
    print("CHECK 1 — PRICE CONSISTENCY (preco_fechamento como referência)")
    print("=" * 72)
    if not divergencias:
        print("  sem eventos comparáveis")
        return divergencias, 0

    pct_por_campo: dict[str, list[float]] = {}
    for d in divergencias:
        for campo, v in d["pcts"].items():
            if v is not None:
                pct_por_campo.setdefault(campo, []).append(v)

    for campo, vals in pct_por_campo.items():
        max_v = max(vals)
        alertas = sum(1 for v in vals if v > alerta_pct)
        flag = " ⚠️" if alertas else " ✅"
        print(f"  {campo:<14} n={len(vals):>4} | max={max_v:.4f}% | >{alerta_pct}%: {alertas}{flag}")

    if eventos_alertados:
        print(f"\n  ⚠️ EVENTOS ALERTADOS ({eventos_alertados}):")
        for d in divergencias:
            if d["alerta"]:
                detalhes = ", ".join(f"{k}={v}%" for k, v in d["pcts"].items() if v is not None and v > alerta_pct)
                print(f"    janela #{d['janela']} (ref={d['preco_fechamento']}) — {detalhes}")
    else:
        print(f"\n  ✅ Nenhuma divergência > {alerta_pct}% em {len(divergencias)} janelas")
    return divergencias, eventos_alertados


# ─────────────────────────────────────────────────────────────────────────────
# Check 2 — VALUE AREA DEGENERADA
# ─────────────────────────────────────────────────────────────────────────────

def check_value_area(eventos: list[dict]) -> dict:
    print("\n" + "=" * 72)
    print("CHECK 2 — VALUE AREA (historical_vp: esperado val < poc < vah)")
    print("=" * 72)

    resumo: dict[str, dict] = {}
    for periodo in PERIODOS_VP:
        degenerados = []
        fora_intervalo = []
        sem_status = []
        for e in eventos:
            vp = e.get("historical_vp", {}).get(periodo, {})
            if not isinstance(vp, dict):
                continue
            val, poc, vah = (_num(vp.get("val")), _num(vp.get("poc")), _num(vp.get("vah")))
            if val is None or poc is None or vah is None:
                continue
            if vp.get("status") != "success":
                sem_status.append((e.get("janela_numero"), vp.get("status")))
            if val == poc or vah == poc or poc < val or poc > vah:
                degenerados.append((e.get("janela_numero"), val, poc, vah))
        resumo[periodo] = {
            "degenerados": degenerados,
            "fora_intervalo": fora_intervalo,
            "sem_status": sem_status,
        }
        total = len(eventos)
        print(f"  {periodo:<8} degenerado: {len(degenerados)}/{total} janelas" + (" ⚠️" if degenerados else " ✅"))
        for j, val, poc, vah in degenerados[:8]:
            causa = "val==poc" if val == poc else ("vah==poc" if vah == poc else "poc fora de [val,vah]")
            print(f"      janela #{j}: val={val} poc={poc} vah={vah} — {causa}")
        if len(degenerados) > 8:
            print(f"      ... e mais {len(degenerados) - 8}")
    return resumo


# ─────────────────────────────────────────────────────────────────────────────
# Check 3 — FALLBACK TRACKING (current_volatility == 0.03)
# ─────────────────────────────────────────────────────────────────────────────

def check_fallback_tracking(eventos: list[dict]) -> dict:
    print("\n" + "=" * 72)
    print(f"CHECK 3 — FALLBACK TRACKING (adaptive_thresholds.current_volatility == {FALLBACK_VOL})")
    print("=" * 72)

    rastreio = []
    for e in eventos:
        aa = e.get("raw_event", {}).get("advanced_analysis", {})
        at = aa.get("adaptive_thresholds", {}) if isinstance(aa, dict) else {}
        vol = at.get("current_volatility")
        rastreio.append(
            {
                "janela": e.get("janela_numero"),
                "epoch_ms": e.get("epoch_ms"),
                "vol": vol,
                "is_fallback": isinstance(vol, (int, float)) and abs(float(vol) - FALLBACK_VOL) < 1e-9,
            }
        )

    # corrente de fallback desde o boot
    seguidas = 0
    primeira_nao_fallback = None
    for r in rastreio:
        if r["is_fallback"]:
            seguidas += 1
        else:
            primeira_nao_fallback = r
            break

    calculadas = sum(1 for r in rastreio if not r["is_fallback"])
    total = len(rastreio)

    if primeira_nao_fallback is not None:
        boot_ms = rastreio[0]["epoch_ms"]
        fim_ms = primeira_nao_fallback["epoch_ms"]
        dur_min = (fim_ms - boot_ms) / 60000.0
        ts = datetime.fromtimestamp(fim_ms / 1000.0, tz=timezone.utc).strftime("%H:%M:%S")
        print(f"  Janelas em fallback desde o boot: {seguidas} consecutivas")
        print(f"  Primeira janela calculada: #{primeira_nao_fallback['janela']} às {ts} UTC "
              f"(vol={primeira_nao_fallback['vol']})")
        print(f"  Warmup de fallback durou {dur_min:.1f} min ⚠️" if dur_min > 0 else "  boot direto em fallback")
    else:
        print(f"  ⚠️ TODAS as {total} janelas ainda em fallback ({FALLBACK_VOL}) — "
              f"realized_vol nunca veio de multi_tf")

    # valores calculados distintos de 0.03 (últimos 8)
    distintos = [r for r in rastreio if not r["is_fallback"]][-8:]
    if distintos:
        print("  Valores calculados observados:")
        for r in distintos:
            print(f"      janela #{r['janela']}: vol={r['vol']}")

    contagem = Counter("fallback" if r["is_fallback"] else "calculado" for r in rastreio)
    print(f"  Distribuição: {dict(contagem)}")
    return {"seguidas": seguidas, "primeira_nao_fallback": primeira_nao_fallback, "rastreio": rastreio}


# ─────────────────────────────────────────────────────────────────────────────
# Resumo
# ─────────────────────────────────────────────────────────────────────────────

def resumo_final(
    eventos: list[dict],
    divergencias: list[dict],
    alertados: int,
    resumo_vp: dict,
    rastreio: dict,
    alerta_pct: float,
) -> None:
    n = len(eventos)
    pct_div = (alertados / n * 100.0) if n else 0.0
    vp_daily = len(resumo_vp.get("daily", {}).get("degenerados", []))
    total_fallback = rastreio["seguidas"]
    primeira = rastreio["primeira_nao_fallback"]
    if primeira is not None:
        dur_min = (primeira["epoch_ms"] - eventos[0]["epoch_ms"]) / 60000.0
        warmup = f"{dur_min:.1f} min (primeira calculada na janela #{primeira['janela']})"
    else:
        warmup = f"Ainda em fallback após {total_fallback} janelas"

    print("\n" + "=" * 72)
    print("RESUMO")
    print("=" * 72)
    print(f"  Nas últimas {n} janelas: {pct_div:.1f}% com price divergence > {alerta_pct}%,")
    print(f"  {vp_daily} janelas com VP diário degenerado, warmup de fallback durou {warmup}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnóstico de saúde dos dados persistidos")
    parser.add_argument("--janelas", type=int, default=50, help="N de janelas a analisar (default: 50)")
    parser.add_argument("--alerta-pct", type=float, default=ALERTA_PCT_DEFAULT, help="Limiar de divergência %% (default: 0.05)")
    parser.add_argument("--fonte", choices=["auto", "visuais", "jsonl"], default="auto", help="Fonte de dados (default: auto)")
    args = parser.parse_args()

    eventos, fonte = carregar_eventos(args.fonte, args.janelas)
    if not eventos:
        print("❌ Nenhum evento ANALYSIS_TRIGGER completo encontrado")
        sys.exit(1)

    divergencias, alertados = check_price_consistency(eventos, args.alerta_pct)
    resumo_vp = check_value_area(eventos)
    rastreio = check_fallback_tracking(eventos)
    resumo_final(eventos, divergencias, alertados, resumo_vp, rastreio, args.alerta_pct)


if __name__ == "__main__":
    main()
