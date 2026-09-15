# fetchers/cftc_cot_fetcher.py
# -*- coding: utf-8 -*-
"""
Coletor oficial CFTC/CME COT — Traders in Financial Futures (Futures Only).
Fase P3 (Arquitetura Context-Only, fora do hot path).

Fonte canônica (P1):
  Socrata `https://publicreporting.cftc.gov/resource/gpe5-46if.json`
  (TFF Futures Only; sem token; SoQL $where/$order/$limit/$offset).

Garantias:
  - Mapeamento explícito symbol -> contract code (sem fuzzy matching).
  - Símbolos fora do mapa => snapshot UNSUPPORTED (nunca fallback).
  - Números Socrata chegam como string ("21083", "-96", "61.8"):
    parse defensivo; bool/NaN/Inf/vazio => None.
  - Nenhum NaN/Inf chega ao snapshot (sanitização RFC 8259).
  - `report_as_of_date` deve ser terça-feira (America/New_York); senão INVALID.
  - Sem `published_at` inventado: disponibilidade = calendário oficial
    (sexta 15:30 ET + feriados) + `first_seen_at` próprio.
  - Raw cache append-only (`dados/cftc_cot_cache.json`): revisão nunca
    sobrescreve a versão já persistida (content_hash + revision).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from common.json_safe import sanitize_json_safe

logger = logging.getLogger("CftcCotFetcher")

DATASET_ID = "gpe5-46if"
BASE_URL = "https://publicreporting.cftc.gov"
RESOURCE_PATH = f"/resource/{DATASET_ID}.json"

_REQUEST_TIMEOUT = 10.0
_MAX_RETRIES = 2
_RETRY_BACKOFF_S = 1.0
_PAGE_LIMIT = 1000
_MIN_INTERVAL_S = 2.0  # throttle conservador (API sem quota publicada)

# Mapa explícito symbol -> (contract_code, market_and_exchange_name oficial).
# Micro incluído (linhas separadas oficiais); nunca somar standard+micro.
SYMBOL_TO_CONTRACT: Dict[str, Tuple[str, str]] = {
    "BTCUSDT": ("133741", "BITCOIN - CHICAGO MERCANTILE EXCHANGE"),
    "MBTUSDT": ("133742", "MICRO BITCOIN - CHICAGO MERCANTILE EXCHANGE"),
    "ETHUSDT": ("146021", "ETHER CASH SETTLED - CHICAGO MERCANTILE EXCHANGE"),
    "METUSDT": ("146022", "MICRO ETHER - CHICAGO MERCANTILE EXCHANGE"),
}

# Campos numéricos Socrata consumidos (categorias oficiais TFF + OI).
_INT_FIELDS = (
    "open_interest_all",
    "dealer_positions_long_all", "dealer_positions_short_all",
    "dealer_positions_spread_all",
    "asset_mgr_positions_long", "asset_mgr_positions_short",
    "asset_mgr_positions_spread",
    "lev_money_positions_long", "lev_money_positions_short",
    "lev_money_positions_spread",
    "other_rept_positions_long", "other_rept_positions_short",
    "other_rept_positions_spread",
    "tot_rept_positions_long_all", "tot_rept_positions_short",
    "nonrept_positions_long_all", "nonrept_positions_short_all",
)

_DEFAULT_CACHE_PATH = Path("dados/cftc_cot_cache.json")

# Invariante single-writer (MEDIUM-1): produção opera UMA instância do bot
# por diretório de trabalho. O lock abaixo é defesa em profundidade com
# stdlib (fcntl/msvcrt, mesmo padrão de events/event_saver.py), sem
# dependência nova. Segundo writer falha fechado na escrita (retorna False,
# memória intacta, contador cache_write_errors). Leitores nunca observam
# JSON parcial (tmp + os.replace mantido).


def _try_interprocess_lock(lock_path: Path):
    """Tenta lock não-bloqueante. Retorna (fh, enforced).

    (None, True) = outro writer detém o lock -> fail closed.
    (fh, False) = locking indisponível nesta plataforma -> prossegue
    (documentado; invariante single-writer continua valendo).
    """
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(lock_path, "w")
    except OSError:
        return None, True
    try:
        import fcntl  # posix

        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fh, True
        except (OSError, IOError):
            fh.close()
            return None, True
    except ImportError:
        pass
    try:
        import msvcrt  # win32

        try:
            fh.write("x")
            fh.flush()
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            return fh, True
        except (OSError, IOError):
            fh.close()
            return None, True
    except ImportError:
        pass
    return fh, False


def _release_interprocess_lock(fh, enforced: bool) -> None:
    if fh is None:
        return
    try:
        if enforced:
            try:
                import fcntl  # posix

                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except ImportError:
                try:
                    import msvcrt  # win32

                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                except (ImportError, OSError):
                    pass
    finally:
        try:
            fh.close()
        except Exception:  # noqa: BLE001
            pass


def _merge_record_lists(base: List[CftcRawRecord],
                        incoming: List[CftcRawRecord]) -> List[CftcRawRecord]:
    """União por (revision, content_hash); nunca apaga revisão de outro writer."""
    seen = {(r.revision, r.content_hash) for r in base}
    merged = list(base)
    for r in incoming:
        if (r.revision, r.content_hash) not in seen:
            seen.add((r.revision, r.content_hash))
            merged.append(r)
    merged.sort(key=lambda r: (r.revision, r.content_hash))
    return merged


def _safe_int(val: Any) -> Optional[int]:
    """Parse defensivo de inteiro Socrata (string numérica)."""
    if val is None or isinstance(val, bool):
        return None
    try:
        if isinstance(val, float):
            if not math.isfinite(val) or val != int(val):
                return None
            return int(val)
        s = str(val).strip().replace(",", "")
        if not s:
            return None
        f = float(s)
        if not math.isfinite(f):
            return None
        return int(f)
    except (ValueError, TypeError):
        return None


def _norm_name(name: Any) -> str:
    return " ".join(str(name or "").strip().split())


def _content_hash(raw: Dict[str, Any]) -> str:
    canon = json.dumps(raw, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


def _parse_asof(value: Any) -> Optional[date]:
    """'clock 2026-09-08T00:00:00.000' -> date. Apenas data de referência."""
    if not value:
        return None
    try:
        s = str(value).strip()
        if "T" in s:
            s = s.split("T")[0]
        return date.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _is_tuesday(d: date) -> bool:
    # Terça = weekday 1 (data de referência CFTC; calendário civil).
    return d.weekday() == 1


@dataclass
class CftcRawRecord:
    """Registro raw versionado (append-only)."""
    contract_code: str
    report_as_of_date: str  # YYYY-MM-DD
    source_row_id: str
    raw: Dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    revision: int = 0
    first_seen_at: str = ""  # ISO UTC
    retrieved_at: str = ""  # ISO UTC
    dataset_id: str = DATASET_ID


class CftcCotFetcher:
    """Cliente assíncrono do COT oficial (TFF futures-only)."""

    def __init__(
        self,
        base_url: str = BASE_URL,
        cache_path: Optional[Path] = None,
        min_interval_s: float = _MIN_INTERVAL_S,
    ):
        self.base_url = base_url.rstrip("/")
        self.cache_path = Path(cache_path) if cache_path else _DEFAULT_CACHE_PATH
        self.min_interval_s = min_interval_s
        self._last_request_mono: float = 0.0
        # cache: (code, asof) -> CftcRawRecord (todas as revisões preservadas:
        # lista ordenada por revision).
        self._records: Dict[Tuple[str, str], List[CftcRawRecord]] = {}
        self.cache_write_errors = 0
        self.cache_lock_contended = 0
        self._load_cache()

    # -- throttle ------------------------------------------------------
    async def _throttle(self) -> None:
        now = time.monotonic()
        wait = self.min_interval_s - (now - self._last_request_mono)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_request_mono = time.monotonic()

    # -- HTTP ----------------------------------------------------------
    async def _get_json(
        self,
        session: aiohttp.ClientSession,
        params: Dict[str, Any],
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        url = f"{self.base_url}{RESOURCE_PATH}"
        timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT)
        for attempt in range(_MAX_RETRIES):
            await self._throttle()
            try:
                async with session.get(url, params=params, timeout=timeout) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if isinstance(data, list):
                            return data, None
                        return None, "schema_error"
                    if resp.status == 429:
                        logger.warning("CFTC COT HTTP 429 (rate limit)")
                        await asyncio.sleep(_RETRY_BACKOFF_S * (attempt + 1) * 2)
                        continue
                    if resp.status >= 500:
                        logger.debug("CFTC COT HTTP %s (tentativa %d)", resp.status, attempt + 1)
                        await asyncio.sleep(_RETRY_BACKOFF_S * (attempt + 1))
                        continue
                    logger.warning("CFTC COT HTTP %s", resp.status)
                    return None, f"http_{resp.status}"
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                logger.debug("CFTC COT falha de conexão (tentativa %d): %s", attempt + 1, e)
                await asyncio.sleep(_RETRY_BACKOFF_S * (attempt + 1))
                continue
            except Exception as e:  # noqa: BLE001 - fail-closed com erro honesto
                logger.warning("CFTC COT erro inesperado: %s", e)
                return None, "fetch_error"
        return None, "fetch_error"

    async def fetch_contract_history(
        self,
        contract_code: str,
        session: Optional[aiohttp.ClientSession] = None,
        since: Optional[str] = None,
        limit_total: int = 5000,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Busca todas as linhas de um contrato (paginado por $offset)."""
        own_session = session is None
        if own_session:
            connector = aiohttp.TCPConnector(force_close=True, enable_cleanup_closed=True)
            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                connector=connector,
                headers={"User-Agent": "MarketBot-CFTC-COT/1.0"},
            )
        assert session is not None
        try:
            out: List[Dict[str, Any]] = []
            offset = 0
            where = f"cftc_contract_market_code='{contract_code}'"
            if since:
                where += f" AND report_date_as_yyyy_mm_dd >= '{since}'"
            while len(out) < limit_total:
                params = {
                    "$where": where,
                    "$order": "report_date_as_yyyy_mm_dd ASC",
                    "$limit": min(_PAGE_LIMIT, limit_total - len(out)),
                    "$offset": offset,
                }
                rows, err = await self._get_json(session, params)
                if err:
                    return out, err
                if not rows:
                    break
                out.extend(rows)
                if len(rows) < min(_PAGE_LIMIT, limit_total - len(out) + len(rows)):
                    break
                offset += len(rows)
            return sanitize_json_safe(out), None
        finally:
            if own_session and session:
                await session.close()

    async def fetch_latest(
        self,
        contract_code: str,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Busca a linha mais recente de um contrato (DESC, limit 1)."""
        own_session = session is None
        if own_session:
            connector = aiohttp.TCPConnector(force_close=True, enable_cleanup_closed=True)
            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                connector=connector,
                headers={"User-Agent": "MarketBot-CFTC-COT/1.0"},
            )
        assert session is not None
        try:
            params = {
                "$where": f"cftc_contract_market_code='{contract_code}'",
                "$order": "report_date_as_yyyy_mm_dd DESC",
                "$limit": 1,
            }
            rows, err = await self._get_json(session, params)
            if rows:
                row = rows[0]
                if isinstance(row, dict) and row.get("report_date_as_yyyy_mm_dd"):
                    return sanitize_json_safe(row), None
            return None, err or "fetch_error"
        finally:
            if own_session and session:
                await session.close()

    # -- ingestão versionada -------------------------------------------
    def ingest_row(
        self,
        contract_code: str,
        row: Dict[str, Any],
        retrieved_at: Optional[str] = None,
    ) -> Tuple[Optional[CftcRawRecord], Optional[str]]:
        """Valida e versiona uma linha Socrata. Retorna (record, error_code)."""
        now_iso = retrieved_at or datetime.now(timezone.utc).isoformat()
        asof = _parse_asof(row.get("report_date_as_yyyy_mm_dd"))
        if asof is None:
            return None, "schema_error"
        if not _is_tuesday(asof):
            return None, "schema_error"
        if row.get("futonly_or_combined") not in (None, "FutOnly"):
            return None, "schema_error"
        if str(row.get("cftc_contract_market_code", "")).strip() != contract_code:
            return None, "schema_error"
        clean = sanitize_json_safe(dict(row))
        record = CftcRawRecord(
            contract_code=contract_code,
            report_as_of_date=asof.isoformat(),
            source_row_id=str(row.get("id", "")),
            raw=clean,
            content_hash=_content_hash(clean),
            retrieved_at=now_iso,
        )
        key = (contract_code, record.report_as_of_date)
        existing = self._records.get(key, [])
        for prev in existing:
            if prev.content_hash == record.content_hash:
                return prev, None  # duplicata idêntica: sem nova revisão
        record.revision = len(existing)
        # Nova versão (primeira ou revisão): first_seen é ESTE instante.
        # Revisão nunca herda o first_seen da v0 (cada versão tem o seu).
        record.first_seen_at = now_iso
        existing.append(record)
        self._records[key] = existing
        self._save_cache()
        return record, None

    def latest_record(self, contract_code: str) -> Optional[CftcRawRecord]:
        cands = [(k, recs[-1]) for k, recs in self._records.items() if k[0] == contract_code]
        if not cands:
            return None
        cands.sort(key=lambda kv: kv[0][1])
        return cands[-1][1]

    def history(self, contract_code: str) -> List[CftcRawRecord]:
        cands = [(k, recs[-1]) for k, recs in self._records.items() if k[0] == contract_code]
        cands.sort(key=lambda kv: kv[0][1])
        return [rec for _, rec in cands]

    # -- cache em disco (append-only) -----------------------------------
    def _read_cache_file(self) -> Dict[Tuple[str, str], List[CftcRawRecord]]:
        """Lê o arquivo sem tocar no estado em memória (para merge)."""
        result: Dict[Tuple[str, str], List[CftcRawRecord]] = {}
        try:
            if not self.cache_path.exists():
                return result
            # Arquivo .tmp órfão (processo morto durante append) é ignorado:
            # somente o caminho canônico é lido; tmp nunca é promovido.
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return result
            for key, recs in data.get("records", {}).items():
                code, asof = key.split("|", 1)
                loaded = []
                for r in recs:
                    try:
                        loaded.append(CftcRawRecord(
                            contract_code=r["contract_code"],
                            report_as_of_date=r["report_as_of_date"],
                            source_row_id=r.get("source_row_id", ""),
                            raw=r.get("raw", {}),
                            content_hash=r.get("content_hash", ""),
                            revision=int(r.get("revision", 0)),
                            first_seen_at=r.get("first_seen_at", ""),
                            retrieved_at=r.get("retrieved_at", ""),
                            dataset_id=r.get("dataset_id", DATASET_ID),
                        ))
                    except (KeyError, TypeError, ValueError):
                        continue
                if loaded:
                    result[(code, asof)] = sorted(loaded, key=lambda x: x.revision)
        except Exception as e:  # noqa: BLE001 - cache corrompido nunca derruba o fetch
            logger.warning("CFTC COT disk cache load failed (ignorado): %s", e)
        return result

    def _load_cache(self) -> None:
        self._records = self._read_cache_file()

    def _save_cache(self) -> bool:
        """Persiste com lock interprocess + merge. False = fail closed."""
        lock_path = self.cache_path.with_suffix(".lock")
        fh, enforced = _try_interprocess_lock(lock_path)
        if fh is None:
            self.cache_write_errors += 1
            self.cache_lock_contended += 1
            logger.warning("CFTC COT cache lock contended (escrita adiada, memória intacta)")
            return False
        try:
            # Merge com o disco: outro writer pode ter avançado desde _load_cache.
            on_disk = self._read_cache_file()
            merged: Dict[Tuple[str, str], List[CftcRawRecord]] = {}
            for key in set(on_disk) | set(self._records):
                merged[key] = _merge_record_lists(
                    on_disk.get(key, []), self._records.get(key, []))
            self._records = merged
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "dataset_id": DATASET_ID,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "records": {
                    f"{code}|{asof}": [asdict(r) for r in recs]
                    for (code, asof), recs in self._records.items()
                },
            }
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.cache_path)
            return True
        except Exception as e:  # noqa: BLE001 - falha de cache nunca é fatal
            self.cache_write_errors += 1
            logger.warning("CFTC COT disk cache save failed (ignorado): %s", e)
            return False
        finally:
            _release_interprocess_lock(fh, enforced)
