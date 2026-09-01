"""
REPLAY J1-J4 — ETAPA 6, FASE 17.

Reconstrói os 4 eventos observados em 2026-08-10 (~14:16-14:19 UTC) com os
VALORES REAIS do histórico (cross_asset / external_markets) e verifica o
contrato após o patch 1+2+3:

  - Persistência SQLite (EventStore): RFC 8259 estrito, NaN/±Inf -> null,
    0.0 e None preservados.
  - Persistência JSONL (EventSaver._save_to_jsonl): idem.
  - Payload IA (build_compact_payload + ensure_safe_llm_payload): nenhum
    literal non-finite; campos não-finitos omitidos/None.

Saída: matriz FIELD | VALUE | SOURCE | OBSERVED | FETCHED | AGE | STATUS.

Uso: python scripts/diagnostics/replay_etapa6_j1j4.py [--tmp-dir DIR]
NÃO escreve em bancos/arquivos de produção; apenas em tmp.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def strict_json_loads(text: str):
    """Parser estrito RFC 8259: NaN/Infinity literals rejeitados."""

    def _reject(token: str):
        raise ValueError(f"literal non-JSON rejeitado: {token}")

    return json.loads(text, parse_constant=_reject)


# ---------------------------------------------------------------------------
# Campos observados (J1-J4 reais, 2026-08-10 ~14:16-14:19 UTC)
# ---------------------------------------------------------------------------
# SOURCE: proveniência mapeada na FASE 3/9; FETCHED: instante do fetch
# (desconhecido no replay -> None); AGE: TTL do provider (segundos).
FIELDS = [
    # (field, value, source, fetched, age_s, status)
    ("vix_current", 15.16, "Yahoo ^VIX (5d)", None, 60, "OK"),
    ("us10y_yield", 4.688, "TwelveData TNX -> Yahoo ^TNX", None, 300, "OK"),
    ("us10y_change_1d", float("nan"), "None hardcoded provider -> float('nan') ml_features", None, None, "NaN->null"),
    ("us2y_yield", float("nan"), "nunca buscado -> float('nan') ml_features", None, None, "NaN->null"),
    ("us2y_change_1d", float("nan"), "nunca buscado -> float('nan') ml_features", None, None, "NaN->null"),
    ("btc_dominance", 16.772727, "fallback volume share Binance (CoinGecko indisponível)", None, 120, "OK"),
    ("btc_dominance_change_7d", 0.0, "FABRICADO (cross_asset_correlations :657)", None, None, "0.0 (fabricado)"),
    ("eth_dominance", 7.493592, "volume share Binance", None, 120, "OK"),
    ("usdt_dominance", float("nan"), "nunca buscado -> float('nan') ml_features", None, None, "NaN->null"),
    ("gold_price", 4329.24, "TwelveData XAU/USD", None, 600, "OK"),
    ("oil_price", 80.64, "Yahoo CL=F (clamp 10-250)", None, 60, "OK"),
    ("vix_change_1d", None, "placeholder None hardcoded (:580)", None, None, "null"),
    ("gold_change_1d", None, "placeholder None hardcoded (:617)", None, None, "null"),
    ("oil_change_1d", None, "placeholder None hardcoded (:622)", None, None, "null"),
    ("btc_eth_corr_7d", 0.8261, "correlação real (rede)", None, 300, "OK"),
    ("btc_eth_corr_30d", 0.8602, "correlação real (rede)", None, 300, "OK"),
    ("btc_dxy_corr_30d", 0.0922, "correlação real (rede)", None, 300, "OK"),
    ("btc_dxy_corr_90d", 0.1075, "correlação real (rede)", None, 300, "OK"),
    ("btc_ndx_corr_30d", -0.334, "correlação real (rede)", None, 300, "OK"),
    ("dxy_return_5d", -0.121135, "Yahoo DX-Y.NYB (fonte de verdade)", None, 600, "OK"),
    ("dxy_return_20d", -1.160100, "Yahoo DX-Y.NYB (fonte de verdade)", None, 600, "OK"),
    ("btc_dxy_correlation_stability", 0.0153, "proxy: |corr30d - corr90d|", None, None, "proxy"),
    ("btc_dxy_inverse_strength", 0.09985, "proxy: |media das corrs|", None, None, "proxy"),
    ("dxy_momentum", -1.03896, "proxy: dxy 20d - 5d (em %)", None, None, "proxy"),
    ("btc_vix_corr_30d", None, "placeholder None hardcoded (:646)", None, None, "null"),
    ("btc_gold_corr_30d", None, "placeholder None hardcoded (:650)", None, None, "null"),
    ("btc_oil_corr_30d", None, "placeholder None hardcoded (:654)", None, None, "null"),
    ("btc_yields_corr_30d", None, "placeholder None hardcoded (:633)", None, None, "null"),
    ("macro_regime", "TRANSITION", "regime cross-asset", None, None, "OK"),
    ("correlation_regime", "DECORRELATED", "regime correlação", None, None, "OK"),
]

EXTERNAL_MARKETS = {
    "DXY": {"preco_atual": 103.456},
    "VIX": {"preco_atual": 15.16},
    "TNX": {"preco_atual": 4.688},
    "SP500": {"preco_atual": float("nan")},  # TwelveData sem chave -> None -> nan
    "GOLD": {"preco_atual": 4329.24},
    "WTI": {"preco_atual": 80.64},
    "FEAR_GREED": {"preco_atual": 62},
}


def make_event(j: int, base_epoch_ms: int) -> dict:
    cross = {}
    for field, value, *_ in FIELDS:
        cross[field] = value
    return {
        "epoch_ms": base_epoch_ms + j * 60_000,
        "tipo_evento": "ANALYSIS_TRIGGER",
        "symbol": "BTCUSDT",
        "window_id": f"j{j}",
        "is_signal": True,
        "ml_features": {"cross_asset": dict(cross)},
        "external_markets": {k: dict(v) for k, v in EXTERNAL_MARKETS.items()},
        "market_context": {"trading_session": "ny", "session_phase": "open"},
        "orderbook_data": {},
        "institutional_analytics": {"quality": {}},
    }


def run_replay(tmp_dir: Path) -> int:
    from database.event_store import EventStore
    from events.event_saver import EventSaver
    from market_orchestrator.ai.llm_payload_guardrail import ensure_safe_llm_payload
    from market_orchestrator.ai.payload_builder_compact import build_compact_payload

    base_epoch = int(datetime(2026, 8, 10, 14, 16, 0, tzinfo=timezone.utc).timestamp() * 1000)
    events = [make_event(j, base_epoch) for j in range(1, 5)]
    is_ts_utc = datetime.fromtimestamp(base_epoch / 1000, tz=timezone.utc)

    store = EventStore(db_path=str(tmp_dir / "replay_events.db"))
    saver = EventSaver.__new__(EventSaver)
    saver.write_jsonl = True
    saver.history_file = tmp_dir / "replay_fluxo.jsonl"
    saver.max_jsonl_bytes = 1_000_000
    saver.logger = __import__("logging").getLogger("replay_etapa6")
    saver.history_file.parent.mkdir(parents=True, exist_ok=True)

    fails = []
    for event in events:
        store.save_event(event)
        saver._save_to_jsonl(event)

    # --- verificação SQLite ---
    rows = store.get_recent_events(limit=10)
    if len(rows) != 4:
        fails.append(f"SQLite: esperado 4 eventos, obtido {len(rows)}")
    for row in rows:
        try:
            strict_json_loads(json.dumps(row, ensure_ascii=False))
        except Exception as exc:
            fails.append(f"SQLite: RFC 8259 violado: {exc}")
            break
        ca = row["ml_features"]["cross_asset"]
        for key in ("us2y_yield", "us10y_change_1d", "usdt_dominance"):
            if ca.get(key) is not None:
                fails.append(f"SQLite: {key} não virou null (valor={ca.get(key)!r})")
        if ca["btc_dominance_change_7d"] != 0.0:
            fails.append("SQLite: 0.0 fabricado não preservado")
        if ca["vix_current"] != 15.16:
            fails.append("SQLite: finito alterado")

    # --- verificação JSONL ---
    lines = saver.history_file.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) != 4:
        fails.append(f"JSONL: esperado 4 linhas, obtido {len(lines)}")
    for line in lines:
        try:
            strict_json_loads(line)
        except Exception as exc:
            fails.append(f"JSONL: RFC 8259 violado: {exc}")

    # --- payload IA ---
    for row in rows:
        payload = build_compact_payload(row)
        payload = ensure_safe_llm_payload(payload)
        if payload is None:
            fails.append("payload IA: ensure_safe_llm_payload retornou None")
            continue
        try:
            json.dumps(payload, ensure_ascii=False, allow_nan=False)
            strict_json_loads(json.dumps(payload, ensure_ascii=False))
        except Exception as exc:
            fails.append(f"payload IA: non-finite vazou: {exc}")

    # ------------------------------------------------------------------
    # Matriz FIELD | VALUE | SOURCE | OBSERVED | FETCHED | AGE | STATUS
    # ------------------------------------------------------------------
    print(f"REPLAY J1-J4 — base ts (UTC): {is_ts_utc.isoformat()}  eventos: {len(events)}")
    print(f"{"=" * 130}")
    print(f"{'FIELD':<34} | {'VALUE':<18} | {'SOURCE':<44} | {'OBSERVED':<26} | {'FETCHED':<10} | {'AGE':<6} | STATUS")
    print(f"{'-' * 130}")
    j1_ca = events[0]["ml_features"]["cross_asset"]
    for field, value, source, fetched, age_s, status in FIELDS:
        observed = j1_ca[field]
        observed_s = "NaN" if isinstance(observed, float) and math.isnan(observed) else repr(observed)
        fetched_s = fetched if fetched is not None else "n/a"
        age_s_s = f"{age_s}s" if age_s is not None else "n/a"
        print(f"{field:<34} | {observed_s:<18} | {source:<44} | {observed_s:<26} | {fetched_s:<10} | {age_s_s:<6} | {status}")

    # ------------------------------------------------------------------
    # Payload IA: quais campos cross_asset chegaram ao LLM?
    # ------------------------------------------------------------------
    print(f"{'-' * 130}")
    sample = ensure_safe_llm_payload(build_compact_payload(rows[0]))
    ca_in_payload = {}
    for k, v in (sample or {}).items():
        if isinstance(v, dict) and any(kk.startswith("btc_") or kk in FIELDS for kk in v):
            ca_in_payload[k] = v
    print("cross_asset no payload IA (compacto):")
    print(json.dumps(ca_in_payload or sample.get("static", {}), ensure_ascii=False, indent=2, allow_nan=False))

    print(f"{'-' * 130}")
    if fails:
        print(f"RESULTADO: FAIL ({len(fails)} problema(s))")
        for f in fails:
            print(f"  - {f}")
        return 1
    print(f"RESULTADO: PASS — SQLite/JSONL RFC 8259, payload IA sem non-finite")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay J1-J4 ETAPA 6 FASE 17")
    parser.add_argument("--tmp-dir", type=Path, default=None)
    args = parser.parse_args()
    tmp_dir = args.tmp_dir or Path(tempfile.mkdtemp(prefix="etapa6_replay_"))
    try:
        return run_replay(tmp_dir)
    finally:
        if args.tmp_dir is None:
            pass  # mantém tmp p/ inspeção; sem artefato em produção


if __name__ == "__main__":
    sys.exit(main())
