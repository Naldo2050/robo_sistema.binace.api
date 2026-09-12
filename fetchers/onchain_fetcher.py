# onchain_fetcher.py
"""
Fetcher de métricas on-chain REAIS usando APIs gratuitas (sem chave de API).

Fontes:
- blockchain.info: hash_rate, difficulty, mempool_size, avg_block_size
- mempool.space: recommended_fees, mempool_stats, difficulty_adjustment
- Binance fapi: funding_rates reais (já coletados pelo context_collector)

Cache interno para respeitar rate limits (1 req/10s blockchain.info, 1 req/5s mempool.space).
Intervalo recomendado: chamar a cada 5 minutos (alinhado com janelas do sistema).
"""

import os
import asyncio
import aiohttp
import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("OnchainFetcher")

# Cache TTL em segundos
_CACHE_TTL = 300  # 5 minutos (alinhado com janelas)
_REQUEST_TIMEOUT = 10  # segundos


# P04: proveniência por campo. Estados NUNCA colapsam em 0:
#   VALID      = API respondeu valor numérico (0 explícito => REAL_ZERO por _zero_ok)
#   MISSING    = HTTP 200 + JSON válido, campo ausente/não-numérico
#   API_ERROR  = exceção / HTTP != 200 / JSON inválido
# Ausência é None (nunca 0). Status viajam em data["_status"] (interno,
# consumido por _merge_metrics; nunca é campo de evidência).
def _classify_present(value: Any) -> str:
    """VALID ou REAL_ZERO para valor presente (0 explícito é dado real)."""
    try:
        return "REAL_ZERO" if float(value) == 0.0 else "VALID"
    except (TypeError, ValueError):
        return "VALID"


class OnchainFetcher:
    """
    Coleta métricas on-chain reais de APIs públicas gratuitas.
    Projetado para ser chamado a cada 5 minutos (1 janela).
    """

    def __init__(self, cache_ttl: int = _CACHE_TTL):
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Any] = {}
        self._cache_ts: float = 0.0
        self._last_valid: Dict[str, Any] = {}

    async def fetch_all(self, session: Optional[aiohttp.ClientSession] = None) -> Dict[str, Any]:
        """
        Retorna todas as métricas on-chain disponíveis.
        Usa cache se dentro do TTL.
        """
        if os.getenv("BOT_TEST_MODE") == "1":
            logger.debug("[TEST_MODE] Onchain: Ignorando busca real")
            return self._last_valid or self._merge_metrics({}, {})

        now = time.time()
        if self._cache and (now - self._cache_ts) < self.cache_ttl:
            return self._cache

        own_session = session is None
        if own_session:
            connector = aiohttp.TCPConnector(force_close=True, enable_cleanup_closed=True)
            session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30), connector=connector)

        try:
            results = await asyncio.gather(
                self._fetch_blockchain_info(session),
                self._fetch_mempool_space(session),
                return_exceptions=True,
            )

            blockchain_data = results[0] if not isinstance(results[0], Exception) else {}
            mempool_data = results[1] if not isinstance(results[1], Exception) else {}

            if isinstance(results[0], Exception):
                logger.warning(f"blockchain.info falhou: {results[0]}")
            if isinstance(results[1], Exception):
                logger.warning(f"mempool.space falhou: {results[1]}")

            merged = self._merge_metrics(blockchain_data, mempool_data)

            if merged:
                self._cache = merged
                self._cache_ts = now
                self._last_valid = merged
            else:
                merged = self._last_valid

            return merged

        finally:
            if own_session and session:
                await session.close()

    async def _fetch_blockchain_info(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """
        Busca métricas da API blockchain.info (gratuita, sem chave).
        Endpoints usados:
        - /q/hashrate (TH/s)
        - /q/getdifficulty
        - /q/unconfirmedcount (mempool size)
        - /q/avgtxsize (avg tx size em bytes)
        """
        base = "https://blockchain.info"
        timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT)
        data: Dict[str, Any] = {}
        status: Dict[str, str] = {}

        endpoints = {
            "hash_rate": "/q/hashrate",
            "difficulty": "/q/getdifficulty",
            "unconfirmed_txs": "/q/unconfirmedcount",
            "avg_tx_size": "/q/avgtxsize",
        }

        for key, path in endpoints.items():
            try:
                async with session.get(f"{base}{path}", timeout=timeout) as resp:
                    if resp.status == 200:
                        try:
                            text = await resp.text()
                            val = float(text.strip())
                        except (TypeError, ValueError):
                            # P04: 200 com corpo inválido = MISSING, não 0
                            status[key] = "MISSING"
                            continue
                        if key == "hash_rate":
                            val = val / 1e6  # Converter de GH/s para EH/s
                        data[key] = val
                        status[key] = _classify_present(val)
                    else:
                        # P04: HTTP != 200 = API_ERROR (ausência, nunca 0)
                        status[key] = "API_ERROR"
                        logger.debug(f"blockchain.info {key}: HTTP {resp.status}")
            except Exception as e:
                status[key] = "API_ERROR"
                logger.debug(f"blockchain.info {key} falhou: {e}")

        # Buscar stats gerais (1 request para múltiplos dados)
        try:
            async with session.get(f"{base}/stats?format=json", timeout=timeout) as resp:
                if resp.status == 200:
                    try:
                        stats = await resp.json()
                    except Exception:
                        stats = None
                    if not isinstance(stats, dict):
                        status["stats"] = "API_ERROR"
                    else:
                        status["stats"] = "VALID"

                        def _stat_num(raw: Any, scale: float = 1.0) -> Optional[float]:
                            # P04: campo ausente/não-numérico => None (MISSING)
                            if raw is None or isinstance(raw, bool):
                                return None
                            try:
                                return float(raw) / scale
                            except (TypeError, ValueError):
                                return None

                        _stat_fields = {
                            "hash_rate_eh": ("hash_rate", 1e18),  # H/s -> EH/s
                            "total_btc_sent_24h": ("total_btc_sent", 1e8),  # satoshi -> BTC
                            "n_tx_24h": ("n_tx", 1.0),
                            "minutes_between_blocks": ("minutes_between_blocks", 1.0),
                            "market_price_usd": ("market_price_usd", 1.0),
                            "trade_volume_btc_24h": ("trade_volume_btc", 1.0),
                            "miners_revenue_btc_24h": ("miners_revenue_btc", 1e8),
                            "total_fees_btc_24h": ("total_fees_btc", 1e8),
                        }
                        for dst, (src, scale) in _stat_fields.items():
                            v = _stat_num(stats.get(src), scale)
                            if v is None:
                                # total_fees negativo é inválido; demais: ausente
                                status[dst] = "MISSING"
                                continue
                            if dst == "total_fees_btc_24h" and v < 0:
                                status[dst] = "MISSING"
                                continue
                            data[dst] = v
                            status[dst] = _classify_present(v)
                else:
                    status["stats"] = "API_ERROR"
        except Exception as e:
            status["stats"] = "API_ERROR"
            logger.debug(f"blockchain.info stats falhou: {e}")

        data["_status"] = status
        return data

    async def _fetch_mempool_space(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """
        Busca métricas da API mempool.space (gratuita, sem chave).
        Endpoints:
        - /api/v1/fees/recommended (fees recomendadas)
        - /api/mempool (estatísticas do mempool)
        - /api/v1/difficulty-adjustment (ajuste de dificuldade)
        """
        base = "https://mempool.space"
        timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT)
        data: Dict[str, Any] = {}
        status: Dict[str, str] = {}

        def _opt_num(raw: Any, scale: float = 1.0) -> Optional[float]:
            # P04: ausente/não-numérico => None (MISSING); nunca 0 fabricado
            if raw is None or isinstance(raw, bool):
                return None
            try:
                return float(raw) / scale
            except (TypeError, ValueError):
                return None

        # Fees recomendadas
        try:
            _req_at = int(time.time() * 1000)
            async with session.get(f"{base}/api/v1/fees/recommended", timeout=timeout) as resp:
                _recv_at = int(time.time() * 1000)
                _status = resp.status
                if resp.status == 200:
                    try:
                        fees = await resp.json()
                    except Exception:
                        fees = None
                    if not isinstance(fees, dict):
                        status["fees"] = "API_ERROR"
                    else:
                        data["fees"] = {}
                        for dst, src in (("fastest_sat_vb", "fastestFee"),
                                         ("half_hour_sat_vb", "halfHourFee"),
                                         ("hour_sat_vb", "hourFee"),
                                         ("economy_sat_vb", "economyFee"),
                                         ("minimum_sat_vb", "minimumFee")):
                            v = _opt_num(fees.get(src))
                            if v is None:
                                status[f"fees.{dst}"] = "MISSING"
                            else:
                                data["fees"][dst] = v
                                status[f"fees.{dst}"] = _classify_present(v)
                        if not data["fees"]:
                            del data["fees"]
                    # FORENSIC-AUDIT: observa API externa sem alterar retorno
                    try:
                        import os as _os

                        if _os.getenv("FORENSIC_CAPTURE", "0") == "1":
                            from audit_live.hooks import on_external_api as _fext

                            _fext({"source": "mempool.space/fees/recommended",
                                   "requested_at": _req_at, "received_at": _recv_at,
                                   "http_status": _status, "success": True,
                                   "raw_value": fees, "normalized_value": data.get("fees"),
                                   "fallback_used": False, "cache_used": False,
                                   "classification": "VALID" if fees else "MISSING"})
                    except Exception as _fe:
                        logger.warning("FORENSIC ext fees hook falhou: %s", _fe, exc_info=True)
                else:
                    # P04: HTTP != 200 = API_ERROR (ausência, nunca 0)
                    status["fees"] = "API_ERROR"
                    # FORENSIC-AUDIT: HTTP não-200 observado
                    try:
                        import os as _os2

                        if _os2.getenv("FORENSIC_CAPTURE", "0") == "1":
                            from audit_live.hooks import on_external_api as _fext2

                            _fext2({"source": "mempool.space/fees/recommended",
                                    "requested_at": _req_at, "received_at": _recv_at,
                                    "http_status": _status, "success": False,
                                    "raw_value": None, "normalized_value": None,
                                    "fallback_used": True, "cache_used": False,
                                    "classification": "API_ERROR"})
                    except Exception as _fe2:
                        logger.warning("FORENSIC ext fees err hook falhou: %s", _fe2, exc_info=True)
        except Exception as e:
            # P04: exceção de rede = API_ERROR para todos os campos de fees
            for _dst in ("fastest_sat_vb", "half_hour_sat_vb", "hour_sat_vb",
                         "economy_sat_vb", "minimum_sat_vb"):
                status.setdefault(f"fees.{_dst}", "API_ERROR")
            status.setdefault("fees", "API_ERROR")
            # FORENSIC-AUDIT: exceção de rede observada
            try:
                import os as _os3

                if _os3.getenv("FORENSIC_CAPTURE", "0") == "1":
                    from audit_live.hooks import on_external_api as _fext3

                    _fext3({"source": "mempool.space/fees/recommended",
                            "requested_at": int(time.time() * 1000), "received_at": int(time.time() * 1000),
                            "http_status": None, "success": False,
                            "raw_value": None, "normalized_value": None,
                            "fallback_used": True, "cache_used": False,
                            "classification": "API_ERROR", "error": str(e)[:300]})
            except Exception as _fe3:
                logger.warning("FORENSIC ext fees exc hook falhou: %s", _fe3, exc_info=True)
            logger.debug(f"mempool.space fees falhou: {e}")

        # Mempool stats
        try:
            _req_at2 = int(time.time() * 1000)
            async with session.get(f"{base}/api/mempool", timeout=timeout) as resp:
                _recv_at2 = int(time.time() * 1000)
                _status2 = resp.status
                if resp.status == 200:
                    try:
                        mempool = await resp.json()
                    except Exception:
                        mempool = None
                    if not isinstance(mempool, dict):
                        status["mempool"] = "API_ERROR"
                        mempool = {}
                    else:
                        data["mempool"] = {}
                        _c = _opt_num(mempool.get("count"))
                        if _c is None:
                            status["mempool.count"] = "MISSING"
                        else:
                            data["mempool"]["count"] = _c
                            status["mempool.count"] = _classify_present(_c)
                        _v = _opt_num(mempool.get("vsize"))
                        if _v is None:
                            status["mempool.vsize_bytes"] = "MISSING"
                        else:
                            data["mempool"]["vsize_bytes"] = _v
                            status["mempool.vsize_bytes"] = _classify_present(_v)
                        _f = _opt_num(mempool.get("total_fee"), 1e8)
                        if _f is None:
                            status["mempool.total_fee_btc"] = "MISSING"
                        else:
                            data["mempool"]["total_fee_btc"] = _f
                            status["mempool.total_fee_btc"] = _classify_present(_f)
                        if not data["mempool"]:
                            del data["mempool"]
                    try:
                        import os as _os4

                        if _os4.getenv("FORENSIC_CAPTURE", "0") == "1":
                            from audit_live.hooks import on_external_api as _fext4

                            _cls = "VALID"
                            try:
                                _c = mempool.get("count", None)
                                _v = mempool.get("vsize", None)
                                if _c is None:
                                    _cls = "MISSING"
                                elif _c == 0 and _v == 0:
                                    _cls = "MISSING"
                            except Exception:
                                _cls = "VALID"
                            _fext4({"source": "mempool.space/mempool",
                                    "requested_at": _req_at2, "received_at": _recv_at2,
                                    "http_status": _status2, "success": True,
                                    "raw_value": mempool, "normalized_value": data.get("mempool"),
                                    "fallback_used": False, "cache_used": False,
                                    "classification": _cls})
                    except Exception as _fe4:
                        logger.warning("FORENSIC ext mempool hook falhou: %s", _fe4, exc_info=True)
                else:
                    # P04: HTTP != 200 = API_ERROR
                    status["mempool"] = "API_ERROR"
                    try:
                        import os as _os5

                        if _os5.getenv("FORENSIC_CAPTURE", "0") == "1":
                            from audit_live.hooks import on_external_api as _fext5

                            _fext5({"source": "mempool.space/mempool",
                                    "requested_at": _req_at2, "received_at": _recv_at2,
                                    "http_status": _status2, "success": False,
                                    "raw_value": None, "normalized_value": None,
                                    "fallback_used": True, "cache_used": False,
                                    "classification": "API_ERROR"})
                    except Exception as _fe5:
                        logger.warning("FORENSIC ext mempool err hook falhou: %s", _fe5, exc_info=True)
        except Exception as e:
            # P04: exceção = API_ERROR para campos do mempool
            for _dst in ("mempool.count", "mempool.vsize_bytes", "mempool.total_fee_btc"):
                status.setdefault(_dst, "API_ERROR")
            status.setdefault("mempool", "API_ERROR")
            try:
                import os as _os6

                if _os6.getenv("FORENSIC_CAPTURE", "0") == "1":
                    from audit_live.hooks import on_external_api as _fext6

                    _fext6({"source": "mempool.space/mempool",
                            "requested_at": int(time.time() * 1000), "received_at": int(time.time() * 1000),
                            "http_status": None, "success": False,
                            "raw_value": None, "normalized_value": None,
                            "fallback_used": True, "cache_used": False,
                            "classification": "API_ERROR", "error": str(e)[:300]})
            except Exception as _fe6:
                logger.warning("FORENSIC ext mempool exc hook falhou: %s", _fe6, exc_info=True)
            logger.debug(f"mempool.space mempool falhou: {e}")

        # Difficulty adjustment
        try:
            async with session.get(f"{base}/api/v1/difficulty-adjustment", timeout=timeout) as resp:
                if resp.status == 200:
                    try:
                        diff = await resp.json()
                    except Exception:
                        diff = None
                    if not isinstance(diff, dict):
                        status["difficulty_adjustment"] = "API_ERROR"
                    else:
                        data["difficulty_adjustment"] = {}
                        for dst, src, nd in (("progress_pct", "progressPercent", 2),
                                             ("estimated_change_pct", "difficultyChange", 2),
                                             ("remaining_blocks", "remainingBlocks", None),
                                             ("remaining_time_ms", "remainingTime", None),
                                             ("previous_retarget_pct", "previousRetarget", 2)):
                            raw_v = diff.get(src)
                            if raw_v is None or isinstance(raw_v, bool):
                                status[f"difficulty_adjustment.{dst}"] = "MISSING"
                                continue
                            try:
                                v = round(float(raw_v), nd) if nd is not None else int(raw_v)
                            except (TypeError, ValueError):
                                status[f"difficulty_adjustment.{dst}"] = "MISSING"
                                continue
                            data["difficulty_adjustment"][dst] = v
                            status[f"difficulty_adjustment.{dst}"] = _classify_present(v)
                        if not data["difficulty_adjustment"]:
                            del data["difficulty_adjustment"]
                else:
                    status["difficulty_adjustment"] = "API_ERROR"
        except Exception as e:
            status.setdefault("difficulty_adjustment", "API_ERROR")
            logger.debug(f"mempool.space difficulty falhou: {e}")

        data["_status"] = status
        return data

    def _merge_metrics(
        self, blockchain: Dict[str, Any], mempool: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Consolida dados das duas fontes no formato esperado pelo sistema.
        Mantém compatibilidade com o schema de onchain_metrics existente.

        P04: ausência/erro é None + status em "_field_status" (VALID /
        REAL_ZERO / MISSING / API_ERROR). NUNCA fabrica 0 para campo sem
        fonte — exceto os campos pagos documentados (exchange_netflow,
        whale_transactions, exchange_reserves, sopr), que seguem marcados
        em requires_paid_api e são removidos pelo updater (NEVER_EVIDENCE).
        """
        def _num(v: Any) -> Optional[float]:
            if v is None or isinstance(v, bool):
                return None
            try:
                f = float(v)
            except (TypeError, ValueError):
                return None
            return f

        def _st(v: Any, fallback: str) -> str:
            # P04: 0 explícito é REAL_ZERO (dado real); ausência herda o
            # status da fonte (API_ERROR se a fonte falhou, senão MISSING).
            if v is None:
                return fallback
            return _classify_present(v)

        b_status: Dict[str, str] = dict(blockchain.get("_status") or {})
        m_status: Dict[str, str] = dict(mempool.get("_status") or {})
        field_status: Dict[str, str] = {}

        def _b_src(*names: str) -> str:
            for n in names:
                s = b_status.get(n)
                if s in ("API_ERROR", "MISSING"):
                    return s
            return "MISSING"

        def _m_src(*names: str) -> str:
            for n in names:
                s = m_status.get(n)
                if s in ("API_ERROR", "MISSING"):
                    return s
            return "MISSING"

        hash_rate = _num(blockchain.get("hash_rate_eh", blockchain.get("hash_rate")))
        difficulty = _num(blockchain.get("difficulty"))
        unconfirmed = _num(blockchain.get("unconfirmed_txs"))

        mempool_data = mempool.get("mempool", {}) or {}
        fees_data = mempool.get("fees", {}) or {}
        diff_adj = mempool.get("difficulty_adjustment", {}) or {}

        def _r2(v: Optional[float]) -> Optional[float]:
            return round(v, 2) if v is not None else None

        def _r4(v: Optional[float]) -> Optional[float]:
            return round(v, 4) if v is not None else None

        def _r6(v: Optional[float]) -> Optional[float]:
            return round(v, 6) if v is not None else None

        out: Dict[str, Any] = {}
        out["hash_rate"] = _r2(hash_rate)
        # difficulty: blockchain.info retorna H; escala p/ T quando magnitude indicar
        if difficulty is None:
            out["difficulty"] = None
        else:
            out["difficulty"] = _r2(difficulty / 1e12) if difficulty > 1e10 else _r2(difficulty)
        out["active_addresses"] = _num(blockchain.get("n_tx_24h"))  # proxy: tx count 24h
        out["exchange_netflow"] = 0.0  # Requer API paga (Glassnode/CryptoQuant)
        out["whale_transactions"] = 0  # Requer API paga (Whale Alert)
        out["miner_flows"] = _r4(_num(blockchain.get("miners_revenue_btc_24h")))
        out["exchange_reserves"] = 0.0  # Requer API paga
        out["sopr"] = 0.0  # Requer API paga (Glassnode)

        # P04 (seção 6): objeto PARCIAL por fonte. mempool_size prefere
        # mempool.space(count); cai para blockchain.info(unconfirmed_txs) com
        # status da fonte que realmente forneceu o valor.
        _count = _num(mempool_data.get("count"))
        if _count is not None:
            out["mempool_size"] = _count
            field_status["mempool_size"] = _st(_count, _m_src("mempool.count", "mempool"))
        elif unconfirmed is not None:
            out["mempool_size"] = unconfirmed
            field_status["mempool_size"] = _st(unconfirmed, _b_src("unconfirmed_txs"))
        else:
            out["mempool_size"] = None
            field_status["mempool_size"] = _m_src("mempool.count", "mempool") \
                if "mempool.count" in m_status or "mempool" in m_status \
                else _b_src("unconfirmed_txs")

        _vsize = _num(mempool_data.get("vsize_bytes"))
        out["mempool_vsize_mb"] = _r2(_vsize / 1e6) if _vsize is not None else None
        _tfee = _num(mempool_data.get("total_fee_btc"))
        out["mempool_total_fee_btc"] = _r6(_tfee) if _tfee is not None else None

        for dst, src in (("fees_fastest_sat_vb", "fastest_sat_vb"),
                         ("fees_half_hour_sat_vb", "half_hour_sat_vb"),
                         ("fees_hour_sat_vb", "hour_sat_vb"),
                         ("fees_economy_sat_vb", "economy_sat_vb")):
            out[dst] = _num(fees_data.get(src))

        out["difficulty_adjustment"] = diff_adj if isinstance(diff_adj, dict) else {}

        out["minutes_between_blocks"] = _num(blockchain.get("minutes_between_blocks"))
        out["total_btc_sent_24h"] = _r2(_num(blockchain.get("total_btc_sent_24h")))
        out["total_fees_btc_24h"] = _r6(_num(blockchain.get("total_fees_btc_24h")))
        out["trade_volume_btc_24h"] = _r2(_num(blockchain.get("trade_volume_btc_24h")))

        for _f, _s in (("hash_rate", _b_src("hash_rate_eh", "hash_rate", "stats")),
                       ("difficulty", _b_src("difficulty")),
                       ("active_addresses", _b_src("n_tx_24h", "stats")),
                       ("miner_flows", _b_src("miners_revenue_btc_24h", "stats")),
                       ("mempool_vsize_mb", _m_src("mempool.vsize_bytes", "mempool")),
                       ("mempool_total_fee_btc", _m_src("mempool.total_fee_btc", "mempool")),
                       ("fees_fastest_sat_vb", _m_src("fees.fastest_sat_vb", "fees")),
                       ("fees_half_hour_sat_vb", _m_src("fees.half_hour_sat_vb", "fees")),
                       ("fees_hour_sat_vb", _m_src("fees.hour_sat_vb", "fees")),
                       ("fees_economy_sat_vb", _m_src("fees.economy_sat_vb", "fees")),
                       ("minutes_between_blocks", _b_src("minutes_between_blocks", "stats")),
                       ("total_btc_sent_24h", _b_src("total_btc_sent_24h", "stats")),
                       ("total_fees_btc_24h", _b_src("total_fees_btc_24h", "stats")),
                       ("trade_volume_btc_24h", _b_src("trade_volume_btc_24h", "stats"))):
            field_status[_f] = _st(out.get(_f), _s)

        # Metadata
        out["data_source"] = "blockchain.info+mempool.space"
        out["is_real_data"] = True
        out["requires_paid_api"] = [
            "exchange_netflow",
            "whale_transactions",
            "exchange_reserves",
            "sopr",
        ]
        out["_field_status"] = field_status
        return out