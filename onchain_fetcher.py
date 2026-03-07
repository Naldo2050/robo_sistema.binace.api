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

import asyncio
import aiohttp
import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("OnchainFetcher")

# Cache TTL em segundos
_CACHE_TTL = 300  # 5 minutos (alinhado com janelas)
_REQUEST_TIMEOUT = 10  # segundos


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
        now = time.time()
        if self._cache and (now - self._cache_ts) < self.cache_ttl:
            return self._cache

        own_session = session is None
        if own_session:
            session = aiohttp.ClientSession()

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
                        text = await resp.text()
                        val = float(text.strip())
                        if key == "hash_rate":
                            val = val / 1e6  # Converter de GH/s para EH/s
                        data[key] = val
                    else:
                        logger.debug(f"blockchain.info {key}: HTTP {resp.status}")
            except Exception as e:
                logger.debug(f"blockchain.info {key} falhou: {e}")

        # Buscar stats gerais (1 request para múltiplos dados)
        try:
            async with session.get(f"{base}/stats?format=json", timeout=timeout) as resp:
                if resp.status == 200:
                    stats = await resp.json()
                    data["hash_rate_eh"] = stats.get("hash_rate", 0) / 1e18  # H/s -> EH/s
                    data["total_btc_sent_24h"] = stats.get("total_btc_sent", 0) / 1e8  # satoshi -> BTC
                    data["n_tx_24h"] = stats.get("n_tx", 0)
                    data["minutes_between_blocks"] = stats.get("minutes_between_blocks", 0)
                    data["market_price_usd"] = stats.get("market_price_usd", 0)
                    data["trade_volume_btc_24h"] = stats.get("trade_volume_btc", 0)
                    data["miners_revenue_btc_24h"] = stats.get("miners_revenue_btc", 0) / 1e8
                    data["total_fees_btc_24h"] = stats.get("total_fees_btc", 0) / 1e8
        except Exception as e:
            logger.debug(f"blockchain.info stats falhou: {e}")

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

        # Fees recomendadas
        try:
            async with session.get(f"{base}/api/v1/fees/recommended", timeout=timeout) as resp:
                if resp.status == 200:
                    fees = await resp.json()
                    data["fees"] = {
                        "fastest_sat_vb": fees.get("fastestFee", 0),
                        "half_hour_sat_vb": fees.get("halfHourFee", 0),
                        "hour_sat_vb": fees.get("hourFee", 0),
                        "economy_sat_vb": fees.get("economyFee", 0),
                        "minimum_sat_vb": fees.get("minimumFee", 0),
                    }
        except Exception as e:
            logger.debug(f"mempool.space fees falhou: {e}")

        # Mempool stats
        try:
            async with session.get(f"{base}/api/mempool", timeout=timeout) as resp:
                if resp.status == 200:
                    mempool = await resp.json()
                    data["mempool"] = {
                        "count": mempool.get("count", 0),
                        "vsize_bytes": mempool.get("vsize", 0),
                        "total_fee_btc": mempool.get("total_fee", 0) / 1e8,
                    }
        except Exception as e:
            logger.debug(f"mempool.space mempool falhou: {e}")

        # Difficulty adjustment
        try:
            async with session.get(f"{base}/api/v1/difficulty-adjustment", timeout=timeout) as resp:
                if resp.status == 200:
                    diff = await resp.json()
                    data["difficulty_adjustment"] = {
                        "progress_pct": round(diff.get("progressPercent", 0), 2),
                        "estimated_change_pct": round(diff.get("difficultyChange", 0), 2),
                        "remaining_blocks": diff.get("remainingBlocks", 0),
                        "remaining_time_ms": diff.get("remainingTime", 0),
                        "previous_retarget_pct": round(diff.get("previousRetarget", 0), 2),
                    }
        except Exception as e:
            logger.debug(f"mempool.space difficulty falhou: {e}")

        return data

    def _merge_metrics(
        self, blockchain: Dict[str, Any], mempool: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Consolida dados das duas fontes no formato esperado pelo sistema.
        Mantém compatibilidade com o schema de onchain_metrics existente.
        """
        hash_rate = blockchain.get("hash_rate_eh", blockchain.get("hash_rate", 0))
        difficulty = blockchain.get("difficulty", 0)
        unconfirmed = blockchain.get("unconfirmed_txs", 0)

        mempool_data = mempool.get("mempool", {})
        fees_data = mempool.get("fees", {})
        diff_adj = mempool.get("difficulty_adjustment", {})

        return {
            # Campos compatíveis com o schema atual do sistema
            "hash_rate": round(hash_rate, 2),
            "difficulty": round(difficulty / 1e12, 2) if difficulty > 1e10 else round(difficulty, 2),
            "active_addresses": blockchain.get("n_tx_24h", 0),  # proxy: tx count 24h
            "exchange_netflow": 0.0,  # Requer API paga (Glassnode/CryptoQuant)
            "whale_transactions": 0,  # Requer API paga (Whale Alert)
            "miner_flows": round(blockchain.get("miners_revenue_btc_24h", 0), 4),
            "exchange_reserves": 0.0,  # Requer API paga
            "sopr": 0.0,  # Requer API paga (Glassnode)

            # Dados extras disponíveis gratuitamente
            "mempool_size": mempool_data.get("count", unconfirmed),
            "mempool_vsize_mb": round(mempool_data.get("vsize_bytes", 0) / 1e6, 2),
            "mempool_total_fee_btc": round(mempool_data.get("total_fee_btc", 0), 6),

            "fees_fastest_sat_vb": fees_data.get("fastest_sat_vb", 0),
            "fees_half_hour_sat_vb": fees_data.get("half_hour_sat_vb", 0),
            "fees_hour_sat_vb": fees_data.get("hour_sat_vb", 0),
            "fees_economy_sat_vb": fees_data.get("economy_sat_vb", 0),

            "difficulty_adjustment": diff_adj,

            "minutes_between_blocks": blockchain.get("minutes_between_blocks", 0),
            "total_btc_sent_24h": round(blockchain.get("total_btc_sent_24h", 0), 2),
            "total_fees_btc_24h": round(blockchain.get("total_fees_btc_24h", 0), 6),
            "trade_volume_btc_24h": round(blockchain.get("trade_volume_btc_24h", 0), 2),

            # Metadata
            "data_source": "blockchain.info+mempool.space",
            "is_real_data": True,
            "requires_paid_api": [
                "exchange_netflow",
                "whale_transactions",
                "exchange_reserves",
                "sopr",
            ],
        }
