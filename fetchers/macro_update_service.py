"""
Serviço de atualização periódica de dados macro em background.
Evita chamadas repetidas mantendo cache sempre fresco.
Implementa cache inteligente e graceful shutdown.
"""
import asyncio
import logging
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import psutil

logger = logging.getLogger(__name__)

# Import das configurações de intervalo
try:
    import sys
    import os
    # Adicionar o diretório pai ao path para encontrar config.py
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from config import CROSS_ASSET_INTERVAL, ECONOMIC_DATA_INTERVAL
except ImportError:
    # Fallback caso config.py não esteja disponível
    CROSS_ASSET_INTERVAL = 900  # 15 minutos
    ECONOMIC_DATA_INTERVAL = 14400  # 4 horas
    logger.warning("config.py não encontrado, usando valores padrão para intervalos")


class MacroUpdateService:
    """
    Serviço singleton que atualiza dados macro periodicamente em background.
    Evita múltiplas chamadas para APIs mantendo cache sempre atualizado.
    """
    
    _instance: Optional['MacroUpdateService'] = None
    _running: bool = False
    _task: Optional[asyncio.Task] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        # Removido update_interval fixo - agora usa intervalos independentes
        self.last_update: Optional[datetime] = None
        self._running = False
        
        # Import do provider
        from fetchers.macro_data_provider import get_macro_provider
        self.provider = get_macro_provider()
         
        # Métricas de performance
        self._performance_metrics: Dict[str, Any] = {
            'total_updates': 0,
            'successful_updates': 0,
            'failed_updates': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'api_calls': 0,
            'last_api_call': None,
            'average_update_time': 0.0,
            'memory_usage_mb': 0,
        }
         
        # Health check
        self._health_status = {
            'status': 'healthy',
            'last_check': None,
            'issues': [],
            'uptime_seconds': 0,
            'start_time': datetime.utcnow(),
        }
         
        logger.info("✅ MacroUpdateService inicializado (SINGLETON)")

    def get_cache_metrics(self) -> Dict[str, Any]:
        """
        Retorna métricas de cache para monitoramento.
        
        Returns:
            Métricas de cache com taxa de acerto e informações de performance
        """
        total_requests = self._performance_metrics['cache_hits'] + self._performance_metrics['cache_misses']
        hit_rate = (self._performance_metrics['cache_hits'] / total_requests * 100) if total_requests > 0 else 0.0
        
        cache_stats = self.provider.get_cache_stats()
        
        return {
            'cache_hit_rate': round(hit_rate, 2),
            'cache_hits': self._performance_metrics['cache_hits'],
            'cache_misses': self._performance_metrics['cache_misses'],
            'total_requests': total_requests,
            'cache_keys': cache_stats.get('total_keys', 0),
            'api_calls': self._performance_metrics['api_calls'],
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'average_update_time_seconds': round(self._performance_metrics['average_update_time'], 3),
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Executa health check completo do serviço.
        
        Returns:
            Status de saúde com métricas e indicadores
        """
        now = datetime.utcnow()
        uptime = (now - self._health_status['start_time']).total_seconds()
        
        # Verificar problemas
        issues = []
        
        # Verificar última atualização
        if self.last_update:
            time_since_update = (now - self.last_update).total_seconds()
            if time_since_update > 300:  # 5 minutos sem atualização
                issues.append(f"Última atualização há {time_since_update:.0f}s")
        else:
            issues.append("Nenhuma atualização realizada")
        
        # Verificar memória
        try:
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            self._performance_metrics['memory_usage_mb'] = memory_mb
            if memory_mb > 500:  # Mais de 500MB
                issues.append(f"Alto uso de memória: {memory_mb:.1f}MB")
        except Exception:
            pass
        
        # Verificar taxa de sucesso
        total_updates = self._performance_metrics['total_updates']
        successful_updates = self._performance_metrics['successful_updates']
        if total_updates > 0:
            success_rate = (successful_updates / total_updates) * 100
            if success_rate < 80:
                issues.append(f"Taxa de sucesso baixa: {success_rate:.1f}%")
        
        # Atualizar status
        status = 'healthy' if len(issues) == 0 else 'degraded' if len(issues) <= 2 else 'unhealthy'
        
        health_status = {
            'status': status,
            'uptime_seconds': int(uptime),
            'last_check': now.isoformat(),
            'issues': issues,
            'metrics': self.get_cache_metrics(),
        }
        
        self._health_status.update(health_status)
        
        return health_status
    
    async def start(self):
        """Inicia o serviço de atualização em background"""
        if self._running:
            logger.warning("⚠️ MacroUpdateService já está rodando")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._update_loop())
        logger.info("✅ MacroUpdateService iniciado")
    
    async def stop(self, timeout: float = 5.0):
        """
        Para o serviço com graceful shutdown.
        
        Args:
            timeout: Tempo máximo para aguardar parada graceful (segundos)
        """
        if not self._running:
            logger.info("ℹ️ MacroUpdateService já estava parado")
            return
        
        logger.info("🛑 Iniciando parada graceful do MacroUpdateService...")
        self._running = False
        
        if self._task and not self._task.done():
            # Cancelar task com timeout
            self._task.cancel()
            
            try:
                await asyncio.wait_for(self._task, timeout=timeout)
                logger.info("✅ MacroUpdateService parado gracefully")
            except asyncio.CancelledError:
                logger.info("✅ MacroUpdateService parado (cancelado)")
            except asyncio.TimeoutError:
                logger.warning(f"⚠️ Timeout ao parar MacroUpdateService após {timeout}s")
                if not self._task.done():
                    logger.error("❌ Task não foi finalizado, possível leak")
            except Exception as e:
                logger.error(f"❌ Erro inesperado ao parar MacroUpdateService: {e}")
        else:
            logger.info("ℹ️ MacroUpdateService não tinha task ativo")
    
    async def _update_loop(self):
        """Loop principal de atualização com verificação independente de timestamps"""
        from fetchers.macro_data_provider import get_macro_provider

        provider = get_macro_provider()

        # Timestamps dos últimos updates
        last_cross_asset_update = 0
        last_economic_update = 0
        last_crypto_update = 0  # Para manter compatibilidade com lógica de cripto existente

        logger.info(f"🔄 Iniciando loop de atualização: Cross-Asset cada {CROSS_ASSET_INTERVAL}s, Economic cada {ECONOMIC_DATA_INTERVAL}s")

        while self._running:
            try:
                current_time = time.time()

                # Check Cross Asset (15 min)
                if current_time - last_cross_asset_update >= CROSS_ASSET_INTERVAL:
                    logger.info("🔄 Atualizando dados Cross-Asset...")
                    try:
                        cross_asset_data = await provider.fetch_cross_asset_data()
                        if cross_asset_data.get("status") == "ok":
                            logger.info(f"✅ Cross-Asset atualizado: {len(cross_asset_data.get('sources', []))} fontes")
                            last_cross_asset_update = current_time
                            self._performance_metrics['successful_updates'] += 1
                        else:
                            logger.warning(f"⚠️ Falha na atualização Cross-Asset: {cross_asset_data.get('error', 'desconhecido')}")
                            self._performance_metrics['failed_updates'] += 1
                    except Exception as e:
                        logger.error(f"❌ Erro atualizando Cross-Asset: {e}")
                        self._performance_metrics['failed_updates'] += 1

                # Check Economic (4 hours)
                if current_time - last_economic_update >= ECONOMIC_DATA_INTERVAL:
                    logger.info("🔄 Atualizando dados Econômicos...")
                    try:
                        economic_data = await provider.fetch_economic_data()
                        if economic_data.get("status") == "ok":
                            logger.info(f"✅ Dados Econômicos atualizados: {len(economic_data.get('sources', []))} fontes")
                            last_economic_update = current_time
                            self._performance_metrics['successful_updates'] += 1
                        else:
                            logger.warning(f"⚠️ Falha na atualização Econômica: {economic_data.get('error', 'desconhecido')}")
                            self._performance_metrics['failed_updates'] += 1
                    except Exception as e:
                        logger.error(f"❌ Erro atualizando dados Econômicos: {e}")
                        self._performance_metrics['failed_updates'] += 1

                # Check Binance/Crypto (Fast - manter lógica existente se houver)
                # Por enquanto, manter o comportamento antigo para compatibilidade
                # TODO: Implementar lógica específica para cripto se necessário

                # Atualizar métricas gerais
                self._performance_metrics['total_updates'] += 1
                self.last_update = datetime.utcnow()

            except Exception as e:
                logger.error(f"❌ Erro no loop de atualização: {e}")
                self._performance_metrics['failed_updates'] += 1

            # Loop rápido que não consome API, apenas checa o tempo
            await asyncio.sleep(1)
    
    def get_status(self) -> dict:
        """Retorna status do serviço"""
        return {
            "running": self._running,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "cross_asset_interval": CROSS_ASSET_INTERVAL,
            "economic_data_interval": ECONOMIC_DATA_INTERVAL,
            "loop_interval": 1,  # Loop roda a cada 1 segundo
        }


# Instância global
_service: Optional[MacroUpdateService] = None


def get_macro_update_service() -> MacroUpdateService:
    """Retorna instância do serviço"""
    global _service
    if _service is None:
        _service = MacroUpdateService()
    return _service


async def start_macro_service():
    """Helper para iniciar o serviço"""
    service = get_macro_update_service()
    await service.start()


async def stop_macro_service():
    """Helper para parar o serviço"""
    service = get_macro_update_service()
    await service.stop()