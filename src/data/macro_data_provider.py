"""
MacroDataProvider - Singleton para coleta de dados macroeconômicos.
Usa padrão Singleton para evitar múltiplas instâncias e chamadas duplicadas.
"""
import os
import time
import logging
import asyncio
import threading
import requests
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import aiohttp

logger = logging.getLogger(__name__)


class MacroDataProvider:
    """
    Provedor unificado de dados macroeconômicos.
    Implementa padrão Singleton para garantir única instância.
      
    Hierarquia de fallback:
    1. Twelve Data (SPX via SPY, GOLD via XAU/USD, TNX via TNX)
    2. Yahoo Finance (DXY via DX-Y.NYB - fonte de verdade absoluta)
    3. FRED API (DESATIVADO - valores incorretos)
    4. Alpha Vantage (commodities, stocks)
    5. Binance (dominância crypto)
      
    Notas:
    - Treasury 10Y disponível via Twelve Data (TNX) com fallback para Yahoo Finance (^TNX)
    - DXY usa Yahoo Finance (DX-Y.NYB) como fonte de verdade devido a discrepâncias na Twelve Data
    """
    
    # ══════════════════════════════════════════════════════════════════════════
    # SINGLETON PATTERN
    # ══════════════════════════════════════════════════════════════════════════

    _instance: Optional['MacroDataProvider'] = None
    _initialized: bool = False
    _init_lock = threading.Lock()  # Lock de thread para singleton
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Evitar reinicialização
        if MacroDataProvider._initialized:
            return
        
        # Configuração de APIs
        self.fred_key = os.getenv("FRED_API_KEY")
        self.alpha_key = os.getenv("ALPHA_VANTAGE_API_KEY") or os.getenv("ALPHAVANTAGE_API_KEY")
        
        # ══════════════════════════════════════════════════════════════════════
        # SISTEMA DE CACHE
        # ══════════════════════════════════════════════════════════════════════
        
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_ttl: int = 300  # 5 minutos padrão
        
        # TTLs específicos por tipo de dado
        self._ttl_config = {
            "vix": 60,              # VIX atualiza a cada 1 minuto
            "treasury_10y": 300,    # Yields a cada 5 minutos
            "treasury_2y": 300,
            "dxy": 600,             # DXY a cada 10 minutos (Twelve Data)
            "gold": 600,            # Gold a cada 10 minutos (Twelve Data)
            "oil": 60,
            "btc_dominance": 120,   # Dominância a cada 2 minutos
            "eth_dominance": 120,
            "usdt_dominance": 120,
            "sp500": 600,           # SP500 a cada 10 minutos (Twelve Data)
            "nasdaq": 60,
            "all_macro": 60,        # Cache agregado
        }
        
        # Controle de rate limiting
        self._last_api_call: Dict[str, float] = {}
        self._min_call_interval = {
            "fred": 0.5,       # FRED: 2 calls/segundo max
            "alpha": 1.0,      # Alpha Vantage: 1 call/segundo (free tier)
            "yahoo": 0.2,      # Yahoo: 5 calls/segundo
            "binance": 0.1,    # Binance: 10 calls/segundo
            "twelve": 7.5,     # Twelve Data: 8 calls/minuto = 1 call/7.5s
        }
        
        # ═══════════════════════════════════════════════════════════════════════
        # HTTP SESSIONS POR LOOP (evita "attached to different loop")
        # ═══════════════════════════════════════════════════════════════════════
        self._sessions: dict = {}  # {loop_id: session}
        
        # ═══════════════════════════════════════════════════════════════════════
        # LOCKS PARA THREAD-SAFETY
        # ═══════════════════════════════════════════════════════════════════════
        self._cache_lock = threading.Lock()  # Para acesso síncrono ao cache
        # Usa threading.Lock para cache para thread-safety
        
        # Marcar como inicializado
        MacroDataProvider._initialized = True
        
        logger.info("✅ MacroDataProvider inicializado (SINGLETON com locks)")
        logger.info(f"   FRED API: {'✅' if self.fred_key else '❌'}")
        logger.info(f"   Alpha Vantage: {'✅' if self.alpha_key else '❌'}")
        logger.info(f"   yfinance: ✅")
    
    
    @classmethod
    def get_instance(cls) -> 'MacroDataProvider':
        """Retorna instância singleton (alternativa ao __new__)"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset para testes - NÃO usar em produção"""
        cls._instance = None
        cls._initialized = False
        logger.warning("⚠️ MacroDataProvider reset (apenas para testes)")
    
    # ══════════════════════════════════════════════════════════════════════════
    # SISTEMA DE CACHE THREAD-SAFE
    # ══════════════════════════════════════════════════════════════════════════
    
    def _get_ttl(self, key: str) -> int:
        """Retorna TTL específico para o tipo de dado"""
        return self._ttl_config.get(key, self._cache_ttl)
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """
        Retorna valor do cache se ainda válido.
        
        Args:
            key: Chave do cache
            
        Returns:
            Valor cacheado ou None se expirado/inexistente
        """
        if key not in self._cache:
            return None
        
        cached_time = self._cache_timestamps.get(key, 0)
        ttl = self._get_ttl(key)
        
        if time.time() - cached_time < ttl:
            logger.debug(f"📦 Cache HIT: {key} (age: {time.time() - cached_time:.1f}s)")
            return self._cache[key]
        
        logger.debug(f"📦 Cache EXPIRED: {key}")
        return None
    
    def _get_cached_thread_safe(self, key: str) -> Optional[Any]:
        """Acesso thread-safe ao cache"""
        with self._cache_lock:
            return self._get_cached(key)
    
    def _set_cache(self, key: str, value: Any) -> None:
        """
        Salva valor no cache com timestamp.
        
        Args:
            key: Chave do cache
            value: Valor a cachear
        """
        if value is not None:
            self._cache[key] = value
            self._cache_timestamps[key] = time.time()
            logger.debug(f"📦 Cache SET: {key}")
    
    def _set_cache_thread_safe(self, key: str, value: Any) -> None:
        """Set thread-safe no cache"""
        with self._cache_lock:
            self._set_cache(key, value)
    
    def _can_call_api(self, api: str) -> bool:
        """
        Verifica se pode fazer chamada respeitando rate limit.
        
        Args:
            api: Nome da API (fred, alpha, yahoo, binance)
            
        Returns:
            True se pode chamar, False se deve aguardar
        """
        last_call = self._last_api_call.get(api, 0)
        min_interval = self._min_call_interval.get(api, 0.5)
        
        if time.time() - last_call < min_interval:
            return False
        
        self._last_api_call[api] = time.time()
        return True
    
    async def _wait_for_rate_limit(self, api: str) -> None:
        """Aguarda até poder fazer chamada respeitando rate limit"""
        while not self._can_call_api(api):
            await asyncio.sleep(0.1)
    
    def clear_cache(self, key: Optional[str] = None) -> None:
        """
        Limpa cache (todo ou específico).
        
        Args:
            key: Chave específica ou None para limpar tudo
        """
        with self._cache_lock:
            if key:
                self._cache.pop(key, None)
                self._cache_timestamps.pop(key, None)
                logger.info(f"🗑️ Cache limpo: {key}")
            else:
                self._cache.clear()
                self._cache_timestamps.clear()
                logger.info("🗑️ Cache completamente limpo")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do cache"""
        now = time.time()
        stats = {
            "total_keys": len(self._cache),
            "keys": {},
        }
        
        for key in self._cache:
            cached_time = self._cache_timestamps.get(key, 0)
            ttl = self._get_ttl(key)
            age = now - cached_time
            stats["keys"][key] = {
                "age_seconds": round(age, 1),
                "ttl_seconds": ttl,
                "expires_in": round(max(0, ttl - age), 1),
                "is_valid": age < ttl,
            }
        
        return stats
    
    # ══════════════════════════════════════════════════════════════════════════
    # HTTP SESSIONS POR LOOP
    # ══════════════════════════════════════════════════════════════════════════
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Retorna ou cria HTTP session para o loop ATUAL.
        Cada loop tem sua própria session para evitar conflitos.
        """
        try:
            loop = asyncio.get_running_loop()
            loop_id = id(loop)
        except RuntimeError:
            # Não tem loop rodando
            loop_id = "default"
        
        # Verificar se já tem session para este loop
        if loop_id in self._sessions:
            session = self._sessions[loop_id]
            if not session.closed:
                return session
        
        # Criar nova session para este loop
        timeout = aiohttp.ClientTimeout(total=10)
        connector = aiohttp.TCPConnector(
            limit=10,
            limit_per_host=5,
            force_close=True,  # Importante para evitar reuso em loop errado
        )
        session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector
        )
        self._sessions[loop_id] = session
        
        logger.debug(f"📡 Nova HTTP session criada para loop {loop_id}")
        return session
    
    async def _close_session_for_loop(self, loop_id=None) -> None:
        """Fecha session de um loop específico"""
        if loop_id is None:
            try:
                loop = asyncio.get_running_loop()
                loop_id = id(loop)
            except RuntimeError:
                return
        
        if loop_id in self._sessions:
            session = self._sessions.pop(loop_id)
            if not session.closed:
                await session.close()
    
    async def close_all_sessions(self) -> None:
        """Fecha todas as sessions"""
        for loop_id, session in list(self._sessions.items()):
            if not session.closed:
                try:
                    await session.close()
                except Exception:
                    pass
        self._sessions.clear()
        logger.info("🔌 Todas as HTTP sessions fechadas")
    
    async def close(self) -> None:
        """Fecha HTTP session do loop atual"""
        await self._close_session_for_loop()
        logger.info("🔌 MacroDataProvider session fechada")
    
    # ══════════════════════════════════════════════════════════════════════════
    # FALLBACK SÍNCRONO PARA DOMINANCE (evita problemas de event loop)
    # ══════════════════════════════════════════════════════════════════════════
    
    def _calculate_dominance_sync(self, coin: str = "BTC") -> Optional[float]:
        """
        Calcula dominância de forma SÍNCRONA (fallback).
        Evita problemas de event loop.
        """
        try:
            response = requests.get(
                "https://api.binance.com/api/v3/ticker/24hr",
                timeout=10
            )
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            
            coin_volume = 0
            total_volume = 0
            
            for ticker in data:
                symbol = ticker.get("symbol", "")
                quote_volume = float(ticker.get("quoteVolume", 0))
                
                if symbol.endswith("USDT"):
                    total_volume += quote_volume
                    if symbol.startswith(coin):
                        coin_volume += quote_volume
            
            if total_volume > 0:
                dominance = (coin_volume / total_volume) * 100
                logger.info(f"✅ {coin} Dominance (sync): {dominance:.2f}%")
                return dominance
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Erro calculando {coin} dominance (sync): {e}")
            return None
    
    # ══════════════════════════════════════════════════════════════════════════
    # MÉTODOS DE COLETA (manter implementação existente, adicionar cache)
    # ══════════════════════════════════════════════════════════════════════════
    
    async def get_vix(self) -> Optional[float]:
        """VIX - Fear Index"""
        cached = self._get_cached_thread_safe("vix")
        if cached is not None:
            return cached
        
        value = await self._fetch_vix_impl()
        self._set_cache_thread_safe("vix", value)
        return value
    
    async def _fetch_vix_impl(self) -> Optional[float]:
        """Implementação real de busca do VIX"""
        try:
            await self._wait_for_rate_limit("yahoo")
            import yfinance as yf
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="1d")
            if not hist.empty:
                value = float(hist['Close'].iloc[-1])
                logger.info(f"✅ VIX (Yahoo): {value:.2f}")
                return value
        except Exception as e:
            logger.warning(f"⚠️ Erro ao buscar VIX: {e}")
        
        return None
    
    async def get_treasury_10y(self) -> Optional[float]:
        """Treasury 10Y Yield"""
        cached = self._get_cached_thread_safe("treasury_10y")
        if cached is not None:
            return cached
        
        value = await self._fetch_treasury_10y_impl()
        self._set_cache_thread_safe("treasury_10y", value)
        return value
    
    async def _fetch_treasury_10y_impl(self) -> Optional[float]:
        """Implementação real de busca do Treasury 10Y"""
        # Tentar Twelve Data primeiro
        twelve_key = os.getenv("TWELVEDATA_API_KEY")
        if twelve_key:
            try:
                await self._wait_for_rate_limit("twelve")
                session = await self._get_session()
                url = f"https://api.twelvedata.com/time_series?symbol=TNX&interval=1day&apikey={twelve_key}"
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("values") and len(data["values"]) > 0:
                            value = float(data["values"][0]["close"])
                            logger.info(f"✅ Treasury 10Y (Twelve Data): {value:.2f}%")
                            return value
                    else:
                        logger.warning(f"⚠️ Twelve Data retornou status {resp.status} para TNX")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao buscar TNX da Twelve Data: {e}")
        else:
            logger.warning("⚠️ Chave da API Twelve Data não configurada para TNX")
        
        # Fallback para Yahoo Finance
        try:
            await self._wait_for_rate_limit("yahoo")
            import yfinance as yf
            tnx = yf.Ticker("^TNX")
            hist = tnx.history(period="1d")
            if not hist.empty:
                value = float(hist['Close'].iloc[-1])
                logger.info(f"✅ Treasury 10Y (Yahoo Finance): {value:.2f}%")
                return value
        except Exception as e:
            logger.warning(f"⚠️ Erro ao buscar TNX do Yahoo Finance: {e}")
        
        # Retornar último valor cacheado se disponível
        cached = self._get_cached_thread_safe("treasury_10y")
        if cached is not None:
            logger.warning(f"⚠️ Usando último valor cacheado para Treasury 10Y: {cached}%")
            return cached
        
        logger.error("❌ Não foi possível obter Treasury 10Y de nenhuma fonte")
        return None
    
    async def get_dxy(self) -> Optional[float]:
        """Dollar Index"""
        cached = self._get_cached_thread_safe("dxy")
        if cached is not None:
            return cached
        
        value = await self._fetch_dxy_impl()
        self._set_cache_thread_safe("dxy", value)
        return value
    
    async def _fetch_dxy_impl(self) -> Optional[float]:
        """Implementação real de busca do DXY - Yahoo Finance como fonte de verdade"""
        # Yahoo Finance é a fonte de verdade absoluta para DXY (DX-Y.NYB)
        try:
            await self._wait_for_rate_limit("yahoo")
            import yfinance as yf
            dxy = yf.Ticker("DX-Y.NYB")
            hist = dxy.history(period="1d")
            if not hist.empty:
                value = float(hist['Close'].iloc[-1])
                logger.info(f"✅ DXY (Yahoo Finance - DX-Y.NYB): {value:.2f}")
                return value
        except Exception as e:
            logger.warning(f"⚠️ Erro ao buscar DXY do Yahoo Finance: {e}")
        
        # Retornar último valor cacheado se disponível
        cached = self._get_cached_thread_safe("dxy")
        if cached is not None:
            logger.warning(f"⚠️ Usando último valor cacheado para DXY: {cached}")
            return cached
        
        logger.error("❌ Não foi possível obter DXY de nenhuma fonte")
        return None
    
    async def get_sp500(self) -> Optional[float]:
        """S&P 500 Index"""
        cached = self._get_cached_thread_safe("sp500")
        if cached is not None:
            return cached

        value = await self._fetch_sp500_impl()
        self._set_cache_thread_safe("sp500", value)
        return value

    async def _fetch_sp500_impl(self) -> Optional[float]:
        """Implementação real de busca do SPX"""
        # Usar Twelve Data como única fonte
        twelve_key = os.getenv("TWELVEDATA_API_KEY")
        if twelve_key:
            try:
                await self._wait_for_rate_limit("twelve")
                session = await self._get_session()
                url = f"https://api.twelvedata.com/time_series?symbol=SPY&interval=1day&apikey={twelve_key}"
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("values") and len(data["values"]) > 0:
                            value = float(data["values"][0]["close"])
                            logger.info(f"✅ SPX (Twelve Data): {value:.2f}")
                            return value
            except Exception as e:
                logger.warning(f"⚠️ Twelve Data SPX erro: {e}")
        else:
            logger.warning("⚠️ Chave da API Twelve Data não configurada para SPX")
        
        return None

    async def get_gold_price(self) -> Optional[float]:
        """Gold XAU/USD"""
        cached = self._get_cached_thread_safe("gold")
        if cached is not None:
            return cached

        value = await self._fetch_gold_impl()
        self._set_cache_thread_safe("gold", value)
        return value
    
    async def _fetch_gold_impl(self) -> Optional[float]:
        """Implementação real de busca do Gold"""
        # Usar Twelve Data como única fonte
        twelve_key = os.getenv("TWELVEDATA_API_KEY")
        if twelve_key:
            try:
                await self._wait_for_rate_limit("twelve")
                session = await self._get_session()
                url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1day&apikey={twelve_key}"
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("values") and len(data["values"]) > 0:
                            value = float(data["values"][0]["close"])
                            logger.info(f"✅ Gold (Twelve Data): ${value:.2f}")
                            return value
            except Exception as e:
                logger.warning(f"⚠️ Twelve Data Gold erro: {e}")
        else:
            logger.warning("⚠️ Chave da API Twelve Data não configurada para Gold")
        
        return None
    
    async def get_oil_price(self) -> Optional[float]:
        """WTI Oil"""
        cached = self._get_cached_thread_safe("oil")
        if cached is not None:
            return cached
        
        value = await self._fetch_oil_impl()
        self._set_cache_thread_safe("oil", value)
        return value
    
    async def _fetch_oil_impl(self) -> Optional[float]:
        """Implementação real de busca do Oil"""
        try:
            await self._wait_for_rate_limit("yahoo")
            import yfinance as yf
            oil = yf.Ticker("CL=F")
            hist = oil.history(period="5d")
            if not hist.empty:
                value = float(hist['Close'].iloc[-1])
                logger.info(f"✅ Oil (Yahoo): ${value:.2f}")
                return value
        except Exception as e:
            logger.warning(f"⚠️ Erro ao buscar Oil: {e}")
        
        return None
    
    async def calculate_btc_dominance(self) -> Optional[float]:
        """Calcula BTC Dominance via Binance"""
        cached = self._get_cached_thread_safe("btc_dominance")
        if cached is not None:
            return cached
        
        # Tentar async primeiro
        try:
            value = await asyncio.wait_for(
                self._fetch_btc_dominance_impl(),
                timeout=5
            )
            if value is not None:
                self._set_cache_thread_safe("btc_dominance", value)
                return value
        except asyncio.TimeoutError:
            logger.warning("⚠️ Timeout calculando BTC Dominance async")
        except Exception as e:
            logger.debug(f"Async BTC dominance falhou: {e}")
        
        # Fallback síncrono
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self._calculate_dominance_sync, "BTC")
            try:
                value = future.result(timeout=10)
                if value is not None:
                    self._set_cache_thread_safe("btc_dominance", value)
                return value
            except Exception as e:
                logger.warning(f"⚠️ Fallback BTC dominance também falhou: {e}")
                return None
    
    async def _fetch_btc_dominance_impl(self) -> Optional[float]:
        """Implementação real de cálculo do BTC Dominance"""
        try:
            await self._wait_for_rate_limit("binance")
            session = await self._get_session()
            
            async with session.get("https://api.binance.com/api/v3/ticker/24hr") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    btc_volume = 0
                    total_volume = 0
                    
                    for ticker in data:
                        symbol = ticker.get("symbol", "")
                        quote_volume = float(ticker.get("quoteVolume", 0))
                        
                        if symbol.endswith("USDT"):
                            total_volume += quote_volume
                            if symbol.startswith("BTC"):
                                btc_volume += quote_volume
                    
                    if total_volume > 0:
                        dominance = (btc_volume / total_volume) * 100
                        logger.info(f"✅ BTC Dominance (Binance): {dominance:.2f}%")
                        return dominance
        except Exception as e:
            logger.warning(f"⚠️ Erro ao calcular BTC Dominance: {e}")
        
        return None
    
    async def calculate_eth_dominance(self) -> Optional[float]:
        """Calcula ETH Dominance via Binance"""
        cached = self._get_cached_thread_safe("eth_dominance")
        if cached is not None:
            return cached
        
        # Tentar async primeiro
        try:
            value = await asyncio.wait_for(
                self._fetch_eth_dominance_impl(),
                timeout=5
            )
            if value is not None:
                self._set_cache_thread_safe("eth_dominance", value)
                return value
        except asyncio.TimeoutError:
            logger.warning("⚠️ Timeout calculando ETH Dominance async")
        except Exception as e:
            logger.debug(f"Async ETH dominance falhou: {e}")
        
        # Fallback síncrono
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self._calculate_dominance_sync, "ETH")
            try:
                value = future.result(timeout=10)
                if value is not None:
                    self._set_cache_thread_safe("eth_dominance", value)
                return value
            except Exception as e:
                logger.warning(f"⚠️ Fallback ETH dominance também falhou: {e}")
                return None
    
    async def _fetch_eth_dominance_impl(self) -> Optional[float]:
        """Implementação real de cálculo do ETH Dominance"""
        try:
            await self._wait_for_rate_limit("binance")
            session = await self._get_session()
            
            async with session.get("https://api.binance.com/api/v3/ticker/24hr") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    eth_volume = 0
                    total_volume = 0
                    
                    for ticker in data:
                        symbol = ticker.get("symbol", "")
                        quote_volume = float(ticker.get("quoteVolume", 0))
                        
                        if symbol.endswith("USDT"):
                            total_volume += quote_volume
                            if symbol.startswith("ETH"):
                                eth_volume += quote_volume
                    
                    if total_volume > 0:
                        dominance = (eth_volume / total_volume) * 100
                        logger.info(f"✅ ETH Dominance (Binance): {dominance:.2f}%")
                        return dominance
        except Exception as e:
            logger.warning(f"⚠️ Erro ao calcular ETH Dominance: {e}")
        
        return None
    
    
    # ══════════════════════════════════════════════════════════════════════════
    # MÉTODO AGREGADOR PRINCIPAL COM LOCKS
    # ══════════════════════════════════════════════════════════════════════════
    
    async def _safe_fetch(self, name: str, coro) -> Any:
        """Wrapper para fetch com timeout individual"""
        try:
            return await asyncio.wait_for(coro, timeout=8)
        except asyncio.TimeoutError:
            logger.warning(f"⚠️ Timeout buscando {name}")
            return None
        except Exception as e:
            logger.warning(f"⚠️ Erro buscando {name}: {e}")
            return None
    
    def _get_partial_cache(self) -> Dict[str, Any]:
        """Retorna dados parciais do cache individual"""
        partial = {}
        for key in ["vix", "treasury_10y", "dxy", "sp500", "gold", "oil", "btc_dominance", "eth_dominance"]:
            cached = self._get_cached_thread_safe(key)
            if cached is not None:
                partial[key] = cached
        return partial
    
    async def get_all_macro_data(self) -> Dict[str, Any]:
        """
        Retorna todos os dados macro agregados com cache.
        """
        key = "all_macro"

        # 1. Verificar cache primeiro
        cached = self._get_cached_thread_safe(key)
        if cached is not None:
            logger.debug(f"📦 Cache HIT: {key}")
            return cached

        logger.info("📊 Coletando dados macro (cache miss)...")

        # 2. Buscar dados com timeout individual e tratamento robusto de erros
        results = []

        # VIX
        try:
            vix_result = await self._safe_fetch("vix", self.get_vix())
            results.append(vix_result)
        except Exception as e:
            logger.error(f"Erro ao calcular VIX: {e}")
            results.append(None)

        # Treasury 10Y
        try:
            treasury_result = await self._safe_fetch("treasury_10y", self.get_treasury_10y())
            results.append(treasury_result)
        except Exception as e:
            logger.error(f"Erro ao calcular Treasury 10Y: {e}")
            results.append(None)

        # DXY
        try:
            dxy_result = await self._safe_fetch("dxy", self.get_dxy())
            results.append(dxy_result)
        except Exception as e:
            logger.error(f"Erro ao calcular DXY: {e}")
            results.append(None)

        # SPX
        try:
            spx_result = await self._safe_fetch("sp500", self.get_sp500())
            results.append(spx_result)
        except Exception as e:
            logger.error(f"Erro ao calcular SPX: {e}")
            results.append(None)

        # Gold
        try:
            gold_result = await self._safe_fetch("gold", self.get_gold_price())
            results.append(gold_result)
        except Exception as e:
            logger.error(f"Erro ao calcular Gold: {e}")
            results.append(None)

        # Oil
        try:
            oil_result = await self._safe_fetch("oil", self.get_oil_price())
            results.append(oil_result)
        except Exception as e:
            logger.error(f"Erro ao calcular Oil: {e}")
            results.append(None)

        # BTC Dominance
        try:
            btc_dom_result = await self._safe_fetch("btc_dominance", self.calculate_btc_dominance())
            results.append(btc_dom_result)
        except Exception as e:
            logger.error(f"Erro ao calcular BTC Dominance: {e}")
            results.append(None)

        # ETH Dominance
        try:
            eth_dom_result = await self._safe_fetch("eth_dominance", self.calculate_eth_dominance())
            results.append(eth_dom_result)
        except Exception as e:
            logger.error(f"Erro ao calcular ETH Dominance: {e}")
            results.append(None)

        # Processar resultados
        vix, treasury_10y, dxy, spx, gold, oil, btc_dom, eth_dom = results

        # Tratar exceções
        def safe_value(val):
            if isinstance(val, Exception):
                return None
            return val

        macro_data = {
            "vix": safe_value(vix),
            "treasury_10y": safe_value(treasury_10y),
            "treasury_2y": None,
            "yield_spread": None,
            "dxy": safe_value(dxy),
            "sp500": safe_value(spx),
            "gold": safe_value(gold),
            "oil": safe_value(oil),
            "btc_dominance": safe_value(btc_dom),
            "eth_dominance": safe_value(eth_dom),
            "usdt_dominance": None,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Calcular yield spread se possível
        if macro_data["treasury_10y"] and macro_data.get("treasury_2y"):
            macro_data["yield_spread"] = macro_data["treasury_10y"] - macro_data["treasury_2y"]

        # Contar sucessos
        success_count = sum(1 for v in macro_data.values() if v is not None and v != "")
        logger.info(f"✅ Coletados {success_count}/9 indicadores macro")

        # 4. Cachear resultado (se tiver dados válidos)
        if success_count > 0:
            self._set_cache_thread_safe(key, macro_data)
        else:
            # Retornar cache parcial se disponível
            partial = self._get_partial_cache()
            if partial:
                partial["timestamp"] = macro_data["timestamp"]
                logger.warning("⚠️ Retornando cache parcial")
                return partial

        return macro_data


# ══════════════════════════════════════════════════════════════════════════════
# FUNÇÃO HELPER PARA ACESSO GLOBAL
# ══════════════════════════════════════════════════════════════════════════════

def get_macro_provider() -> MacroDataProvider:
    """
    Função helper para obter instância singleton do MacroDataProvider.
    Uso: provider = get_macro_provider()
    """
    return MacroDataProvider.get_instance()
