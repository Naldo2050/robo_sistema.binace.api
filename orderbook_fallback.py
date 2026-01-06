# orderbook_fallback.py
"""
Fallback robusto para OrderBook quando WebSocket falha.
Implementa retry inteligente com jitter e Circuit Breaker integration.
"""

import asyncio
import logging
import random
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import aiohttp

try:
    from config import (
        ORDERBOOK_REST_FALLBACK_ENABLED,
        ORDERBOOK_REST_REQUEST_TIMEOUT,
        ORDERBOOK_REST_MAX_RETRIES,
        ORDERBOOK_REST_RETRY_BACKOFF,
        ORDERBOOK_REST_JITTER_RANGE,
    )
except ImportError:
    # Fallbacks seguros
    ORDERBOOK_REST_FALLBACK_ENABLED = True
    ORDERBOOK_REST_REQUEST_TIMEOUT = 15.0
    ORDERBOOK_REST_MAX_RETRIES = 5
    ORDERBOOK_REST_RETRY_BACKOFF = 2.0
    ORDERBOOK_REST_JITTER_RANGE = 0.25


@dataclass
class FallbackConfig:
    """Configuração do fallback REST API."""
    enabled: bool = ORDERBOOK_REST_FALLBACK_ENABLED
    request_timeout: float = ORDERBOOK_REST_REQUEST_TIMEOUT
    max_retries: int = ORDERBOOK_REST_MAX_RETRIES
    backoff_factor: float = ORDERBOOK_REST_RETRY_BACKOFF
    jitter_range: float = ORDERBOOK_REST_JITTER_RANGE
    base_delay: float = 1.0
    max_delay: float = 30.0


class OrderBookFallback:
    """
    Fallback robusto para OrderBook usando REST API.
    
    Características:
    - Retry exponencial com jitter
    - Circuit Breaker integration
    - Multiple endpoints de fallback
    - Rate limiting inteligente
    - Health checks
    """
    
    def __init__(self, config: Optional[FallbackConfig] = None):
        self.config = config or FallbackConfig()
        self.logger = logging.getLogger(__name__)
        
        # Multiple endpoints para redundância
        self.endpoints = [
            "https://api.binance.com/api/v3/depth",  # Spot API
            "https://api.binance.us/api/v3/depth",   # US API
            "https://fapi.binance.com/fapi/v1/depth", # Futures API
        ]
        
        # Rate limiting
        self._last_request_time = 0.0
        self._min_request_interval = 0.1  # 100ms between requests
        
        # Health tracking
        self._consecutive_failures = 0
        self._last_success_time = 0.0
        self._health_score = 1.0  # 0.0 = unhealthy, 1.0 = healthy
        
    def _calculate_retry_delay(self, attempt: int) -> float:
        """
        Calcula delay com backoff exponencial e jitter para evitar thundering herd.
        
        Args:
            attempt: Número da tentativa (0-indexed)
            
        Returns:
            Delay em segundos com jitter
        """
        # Backoff exponencial
        delay = min(
            self.config.max_delay,
            self.config.base_delay * (self.config.backoff_factor ** attempt)
        )
        
        # Jitter para evitar thundering herd
        jitter_range = delay * self.config.jitter_range
        jitter = random.uniform(-jitter_range, jitter_range)
        
        final_delay = max(0.1, delay + jitter)
        self.logger.debug(f"Retry delay calculation: attempt={attempt}, base_delay={delay:.2f}, jitter={jitter:.2f}, final={final_delay:.2f}")
        
        return final_delay
    
    def _check_rate_limit(self) -> bool:
        """Verifica se podemos fazer request (rate limiting)."""
        now = time.time()
        time_since_last = now - self._last_request_time
        
        if time_since_last < self._min_request_interval:
            return False
            
        self._last_request_time = now
        return True
    
    def _update_health_score(self, success: bool) -> None:
        """Atualiza score de saúde baseado em sucessos/falhas."""
        if success:
            self._consecutive_failures = 0
            self._last_success_time = time.time()
            # Incrementa saúde gradualmente
            self._health_score = min(1.0, self._health_score + 0.1)
        else:
            self._consecutive_failures += 1
            # Decrementa saúde baseado no número de falhas
            decay = min(0.3, self._consecutive_failures * 0.1)
            self._health_score = max(0.0, self._health_score - decay)
    
    def is_healthy(self) -> bool:
        """Verifica se o fallback está saudável."""
        # Considera não saudável se muitas falhas consecutivas ou score muito baixo
        return (self._consecutive_failures < 5 and 
                self._health_score > 0.2 and
                self.config.enabled)
    
    async def fetch_orderbook_fallback(
        self, 
        symbol: str, 
        limit: int = 100,
        session: Optional[aiohttp.ClientSession] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Busca orderbook via fallback REST API com retry inteligente.
        
        Args:
            symbol: Símbolo do par (ex: BTCUSDT)
            limit: Profundidade do orderbook
            session: Sessão HTTP opcional
            
        Returns:
            Snapshot do orderbook ou None se falhar
        """
        if not self.config.enabled:
            self.logger.warning("Fallback REST está desabilitado")
            return None
            
        if not self.is_healthy():
            self.logger.warning(f"Fallback não está saudável (score={self._health_score:.2f}, falhas={self._consecutive_failures})")
            return None
        
        # Verifica rate limiting
        if not self._check_rate_limit():
            self.logger.warning("Rate limit atingido no fallback")
            return None
        
        # Usa sessão fornecida ou cria uma temporária
        use_own_session = session is None
        if use_own_session:
            session = await self._create_session()
        
        try:
            # Tenta múltiplos endpoints com retry
            for attempt in range(self.config.max_retries):
                for endpoint in self.endpoints:
                    try:
                        self.logger.debug(
                            f"Fallback attempt {attempt + 1}/{self.config.max_retries} "
                            f"endpoint {endpoint} for {symbol}"
                        )
                        
                        # Construi URL
                        url = f"{endpoint}?symbol={symbol}&limit={limit}"
                        
                        # Faz request com timeout
                        async with session.get(
                            url, 
                            timeout=aiohttp.ClientTimeout(total=self.config.request_timeout)
                        ) as response:
                            
                            if response.status == 200:
                                data = await response.json()
                                
                                # Valida dados básicos
                                if self._validate_fallback_data(data, symbol):
                                    self._update_health_score(True)
                                    self.logger.info(
                                        f"✅ Fallback bem-sucedido: {symbol} via {endpoint}"
                                    )
                                    return self._convert_binance_response(data, symbol)
                                else:
                                    self.logger.warning(
                                        f"⚠️ Dados inválidos do fallback: {symbol}"
                                    )
                            else:
                                self.logger.warning(
                                    f"⚠️ HTTP {response.status} do fallback {endpoint}"
                                )
                                
                    except asyncio.TimeoutError:
                        self.logger.warning(f"⏱️ Timeout no fallback {endpoint}")
                    except aiohttp.ClientError as e:
                        self.logger.warning(f"🌐 Client error no fallback {endpoint}: {e}")
                    except Exception as e:
                        self.logger.error(f"💥 Erro inesperado no fallback {endpoint}: {e}")
                
                # Se não conseguiu nenhum endpoint, espera antes do próximo retry
                if attempt < self.config.max_retries - 1:
                    delay = self._calculate_retry_delay(attempt)
                    self.logger.info(
                        f"🔄 Fallback retry {attempt + 1}/{self.config.max_retries} "
                        f"após {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
            
            # Todas as tentativas falharam
            self._update_health_score(False)
            self.logger.error(f"💀 Fallback falhou completamente para {symbol}")
            return None
            
        finally:
            # Fecha sessão se criou uma própria
            if use_own_session and session and not session.closed:
                await session.close()
    
    def _validate_fallback_data(self, data: Dict[str, Any], symbol: str) -> bool:
        """Valida dados retornados pelo fallback."""
        try:
            # Verifica estrutura básica
            if not isinstance(data, dict):
                return False
                
            if "bids" not in data or "asks" not in data:
                return False
                
            # Verifica se bids e asks são listas
            if not isinstance(data["bids"], list) or not isinstance(data["asks"], list):
                return False
                
            # Verifica se tem pelo menos um nível
            if len(data["bids"]) == 0 or len(data["asks"]) == 0:
                return False
                
            # Verifica formato dos níveis
            for bid in data["bids"][:3]:  # Verifica apenas os primeiros 3
                if not isinstance(bid, list) or len(bid) < 2:
                    return False
                try:
                    float(bid[0]), float(bid[1])  # price, quantity
                except (ValueError, TypeError):
                    return False
                    
            for ask in data["asks"][:3]:  # Verifica apenas os primeiros 3
                if not isinstance(ask, list) or len(ask) < 2:
                    return False
                try:
                    float(ask[0]), float(ask[1])  # price, quantity
                except (ValueError, TypeError):
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.debug(f"Erro na validação do fallback: {e}")
            return False
    
    def _convert_binance_response(self, data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Converte resposta da Binance para formato padrão."""
        try:
            # Padroniza estrutura para compatibilidade
            converted = {
                "symbol": symbol,
                "bids": [[float(bid[0]), float(bid[1])] for bid in data.get("bids", [])],
                "asks": [[float(ask[0]), float(ask[1])] for ask in data.get("asks", [])],
                "lastUpdateId": data.get("lastUpdateId", int(time.time() * 1000)),
                "E": data.get("E", int(time.time() * 1000)),  # Event time
                "T": data.get("T", int(time.time() * 1000)),  # Transaction time
                "source": "fallback_rest",
                "fallback_endpoint": "binance_rest",
            }
            
            return converted
            
        except Exception as e:
            self.logger.error(f"Erro ao converter resposta do fallback: {e}")
            return {}
    
    async def _create_session(self) -> aiohttp.ClientSession:
        """Cria sessão HTTP otimizada para fallback."""
        timeout = aiohttp.ClientTimeout(
            total=self.config.request_timeout,
            connect=5.0,
            sock_read=self.config.request_timeout,
        )
        
        connector = aiohttp.TCPConnector(
            limit=5,
            limit_per_host=2,
            ttl_dns_cache=300,
            force_close=False,
            enable_cleanup_closed=True,
        )
        
        return aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            raise_for_status=False,
        )
    
    def get_fallback_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do fallback."""
        return {
            "enabled": self.config.enabled,
            "healthy": self.is_healthy(),
            "health_score": round(self._health_score, 3),
            "consecutive_failures": self._consecutive_failures,
            "last_success_time": self._last_success_time,
            "available_endpoints": len(self.endpoints),
            "rate_limit_interval": self._min_request_interval,
        }
    
    def reset_fallback_health(self) -> None:
        """Reseta saúde do fallback (útil para testing)."""
        self._consecutive_failures = 0
        self._health_score = 1.0
        self._last_success_time = time.time()
        self.logger.info("🔄 Fallback health reset")


# Instância global do fallback
_fallback_instance: Optional[OrderBookFallback] = None


def get_fallback_instance() -> OrderBookFallback:
    """Retorna instância global do fallback."""
    global _fallback_instance
    if _fallback_instance is None:
        _fallback_instance = OrderBookFallback()
    return _fallback_instance


async def fetch_with_fallback(
    symbol: str, 
    limit: int = 100,
    session: Optional[aiohttp.ClientSession] = None
) -> Optional[Dict[str, Any]]:
    """
    Função utilitária para buscar orderbook com fallback.
    
    Args:
        symbol: Símbolo do par
        limit: Profundidade
        session: Sessão HTTP opcional
        
    Returns:
        Snapshot do orderbook ou None
    """
    fallback = get_fallback_instance()
    return await fallback.fetch_orderbook_fallback(symbol, limit, session)