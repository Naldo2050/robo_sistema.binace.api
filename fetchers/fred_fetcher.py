# fred_fetcher.py
"""
FRED API Fetcher - Sistema de fallback para dados econômicos.
Federal Reserve Economic Data - API gratuita e ilimitada.
"""

import asyncio
import aiohttp
import json
import logging
import os
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timedelta

# ============================================================================
# ⚠️ IMPORTANTE: CARREGAR .env ANTES DE TUDO
# ============================================================================
from dotenv import load_dotenv
load_dotenv()  # ← Esta linha DEVE estar aqui!

logger = logging.getLogger("FREDFetcher")


class FREDFetcher:
    """
    Cliente para FRED API (Federal Reserve Economic Data).
    
    Características:
    - API gratuita e ilimitada
    - Dados de indicadores econômicos oficiais
    - Fallback confiável para yFinance
    """
    
    def __init__(self):
        self.api_key = os.getenv("FRED_API_KEY", "")
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
        
        # Mapeamento de símbolos para IDs do FRED
        self.symbol_map = {
            "DXY": "DTWEXM",          # Trade Weighted Dollar Index: Major Currencies (~80-110, próximo do ICE DXY)
            "TNX": "DGS10",           # 10-Year Treasury Constant Maturity Rate
            "TNY": "DGS2",            # 2-Year Treasury Constant Maturity Rate
            "FED_RATE": "DFF",        # Federal Funds Effective Rate
            "INFLATION": "CPIAUCSL",  # CPI (Consumer Price Index)
            "GDP": "GDP",             # Gross Domestic Product
        }
        
        # Cache em memória (TTL: 5 minutos)
        self._cache = {}
        self._cache_ttl = 300  # segundos

        # Cache persistente em disco (TTL: 24 horas)
        self._disk_cache_path = Path("dados/fred_cache.json")
        self._disk_cache_ttl = 86400  # 24h
        self._disk_cache = self._load_disk_cache()
        
        # 🆕 Cache de falhas específicas por símbolo (para fallback inteligente)
        self._failure_cache = {}  # {symbol: timestamp_falha}
        self._failure_ttl = 3600  # 1 hora de TTL para falhas
        
        if self.api_key:
            logger.info("✅ FRED API inicializada")
        else:
            logger.warning("⚠️ FRED_API_KEY não encontrada no .env")
    
    # ── Cache persistente em disco ──

    def _load_disk_cache(self) -> dict:
        """Carrega cache de disco (JSON)."""
        try:
            if self._disk_cache_path.exists():
                data = json.loads(self._disk_cache_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except Exception as e:
            logger.debug("FRED disk cache load failed: %s", e)
        return {}

    def _save_disk_cache(self) -> None:
        """Salva cache em disco."""
        try:
            self._disk_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._disk_cache_path.write_text(
                json.dumps(self._disk_cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.debug("FRED disk cache save failed: %s", e)

    def _get_from_disk_cache(self, symbol: str) -> Optional[float]:
        """Retorna valor do cache em disco se dentro do TTL."""
        entry = self._disk_cache.get(symbol)
        if not entry:
            return None
        ts = entry.get("ts", 0)
        if datetime.now().timestamp() - ts > self._disk_cache_ttl:
            return None
        return entry.get("value")

    def _set_disk_cache(self, symbol: str, value: float) -> None:
        """Salva valor no cache em disco."""
        self._disk_cache[symbol] = {
            "value": value,
            "ts": datetime.now().timestamp(),
            "updated": datetime.now().isoformat(),
        }
        self._save_disk_cache()

    def is_available(self) -> bool:
        """Verifica se a API está configurada."""
        return bool(self.api_key)
    
    def is_failing(self, symbol: str) -> bool:
        """
        Verifica se o símbolo está em modo de falha (fallback ativo).
        
        Args:
            symbol: Símbolo a verificar
            
        Returns:
            True se o símbolo falhou recentemente e está em modo fallback
        """
        if symbol not in self._failure_cache:
            return False
        
        failure_time = self._failure_cache[symbol]
        now = datetime.now().timestamp()
        
        if now - failure_time > self._failure_ttl:
            # TTL expirou, remover do cache
            del self._failure_cache[symbol]
            return False
        
        return True
    
    def mark_as_failing(self, symbol: str) -> None:
        """
        Marca um símbolo como falho, ativando o modo fallback.
        
        Args:
            symbol: Símbolo a marcar como falho
        """
        self._failure_cache[symbol] = datetime.now().timestamp()
        logger.info(f"[FRED] Símbolo {symbol} marcado como falho, fallback ativo por {self._failure_ttl//60} minutos")
    
    def clear_failure(self, symbol: str) -> None:
        """
        Limpa o status de falha de um símbolo (útil após sucesso).
        
        Args:
            symbol: Símbolo a limpar
        """
        if symbol in self._failure_cache:
            del self._failure_cache[symbol]
    
    async def fetch_latest_value(
        self, 
        symbol: str, 
        session: aiohttp.ClientSession,
        days_lookback: int = 30
    ) -> Optional[float]:
        """
        Busca o valor mais recente de um símbolo.
        
        Args:
            symbol: Símbolo (ex: "DXY", "TNX")
            session: Session aiohttp
            days_lookback: Dias para trás (padrão: 7)
        
        Returns:
            Valor float ou None se falhar
        """
        fred_id = self.symbol_map.get(symbol)
        if not fred_id:
            logger.debug(f"Símbolo {symbol} não mapeado no FRED")
            return None
        
        if not self.is_available():
            logger.debug("FRED API não configurada")
            return None
        
        # Verificar se símbolo está em modo de falha (fallback ativo)
        if self.is_failing(symbol):
            logger.debug(f"FRED: {symbol} está em modo fallback (falha recente)")
            # Tentar cache persistente como fallback
            disk_val = self._get_from_disk_cache(symbol)
            if disk_val is not None:
                logger.debug(f"FRED: {symbol} = {disk_val:.4f} (disk cache fallback)")
            return disk_val

        # Verificar cache em memória
        cache_key = f"{symbol}_{days_lookback}"
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if (datetime.now().timestamp() - timestamp) < self._cache_ttl:
                logger.debug(f"FRED cache hit para {symbol}")
                return cached_data

        try:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days_lookback)).strftime("%Y-%m-%d")

            params = {
                "series_id": fred_id,
                "api_key": self.api_key,
                "file_type": "json",
                "observation_start": start_date,
                "observation_end": end_date,
                "sort_order": "desc",
                "limit": 1
            }

            async with session.get(self.base_url, params=params, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()

                    if data.get("observations") and len(data["observations"]) > 0:
                        value_str = data["observations"][0]["value"]

                        # FRED pode retornar "." para dados ausentes
                        if value_str == ".":
                            logger.warning(f"FRED: {symbol} ({fred_id}) sem dados recentes")
                            return self._get_from_disk_cache(symbol)

                        value = float(value_str)

                        # Armazenar em ambos os caches
                        self._cache[cache_key] = (value, datetime.now().timestamp())
                        self._set_disk_cache(symbol, value)
                        self.clear_failure(symbol)

                        logger.info(f"FRED: {symbol} = {value:.4f} (serie: {fred_id})")
                        return value
                    else:
                        # DTWEXM (DXY) é série semanal — sem observações na maioria dos dias é normal
                        logger.info(f"FRED: {symbol} sem observações recentes, usando disk cache")
                        return self._get_from_disk_cache(symbol)

                elif resp.status == 400:
                    error_data = await resp.json()
                    logger.error(f"FRED erro 400 para {symbol}: {error_data.get('error_message', 'Desconhecido')}")
                    return self._get_from_disk_cache(symbol)
                else:
                    logger.warning(f"FRED status {resp.status} para {symbol}")
                    return self._get_from_disk_cache(symbol)

        except asyncio.TimeoutError:
            logger.warning(f"FRED timeout para {symbol}")
            return self._get_from_disk_cache(symbol)
        except Exception as e:
            logger.warning(f"FRED erro para {symbol}: {e}")
            return self._get_from_disk_cache(symbol)
    
    async def fetch_historical(
        self,
        symbol: str,
        session: aiohttp.ClientSession,
        days: int = 90
    ) -> pd.DataFrame:
        """
        Busca dados históricos.
        
        Args:
            symbol: Símbolo (ex: "DXY")
            session: Session aiohttp
            days: Dias históricos
        
        Returns:
            DataFrame com colunas ['date', 'value'] ou vazio
        """
        fred_id = self.symbol_map.get(symbol)
        if not fred_id or not self.is_available():
            return pd.DataFrame()
        
        try:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            params = {
                "series_id": fred_id,
                "api_key": self.api_key,
                "file_type": "json",
                "observation_start": start_date,
                "observation_end": end_date,
            }
            
            async with session.get(self.base_url, params=params, timeout=15) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    if data.get("observations"):
                        df = pd.DataFrame(data["observations"])
                        df = df[df['value'] != '.']  # Remover dados ausentes
                        df['value'] = pd.to_numeric(df['value'], errors='coerce')
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.dropna(subset=['value'])
                        
                        logger.info(f"✅ FRED histórico: {symbol} = {len(df)} pontos")
                        return df[['date', 'value']]
                    
        except Exception as e:
            logger.warning(f"FRED histórico erro para {symbol}: {e}")
        
        return pd.DataFrame()


# Teste standalone
async def test_fred():
    """Teste do FRED Fetcher."""
    fetcher = FREDFetcher()
    
    if not fetcher.is_available():
        print("❌ FRED_API_KEY não configurada no .env")
        print("📝 Obtenha em: https://fred.stlouisfed.org/docs/api/api_key.html")
        return
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        symbols = ["DXY", "TNX", "TNY", "FED_RATE"]
        
        print("\n[TESTE] Testando FRED API...")
        print("=" * 60)

        for symbol in symbols:
            try:
                value = await fetcher.fetch_latest_value(symbol, session)
            except Exception as e:
                logger.error(f"Erro em operação async: {e}")
                raise
            if value is not None:
                print(f"[OK] {symbol:12} = {value:>10.4f}")
            else:
                print(f"[ERRO] {symbol:12} = Falhou")

            await asyncio.sleep(0.5)  # Rate limiting gentil

        print("=" * 60)

        # Teste histórico
        print("\n[HISTORICO] Teste de dados históricos (DXY, 30 dias):")
        try:
            df = await fetcher.fetch_historical("DXY", session, days=30)
        except Exception as e:
            logger.error(f"Erro em operação async: {e}")
            raise
        if not df.empty:
            print(f"[OK] {len(df)} pontos obtidos")
            print(df.tail(3))
        else:
            print("[ERRO] Sem dados históricos")


if __name__ == "__main__":
    asyncio.run(test_fred())