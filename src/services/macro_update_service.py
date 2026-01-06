"""
Serviço de atualização periódica de dados macro em background.
Evita chamadas repetidas mantendo cache sempre fresco.
Implementa cache inteligente e graceful shutdown.
"""
import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import psutil

logger = logging.getLogger(__name__)


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
        self.update_interval = 60  # Atualizar a cada 60 segundos
        self.last_update: Optional[datetime] = None
        self._running = False
        
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
        """Loop principal de atualização"""
        from src.data.macro_data_provider import get_macro_provider
        
        while self._running:
            try:
                provider = get_macro_provider()
                
                # Limpar cache antigo para forçar atualização
                provider.clear_cache("all_macro")
                
                # Buscar novos dados (isso atualiza o cache)
                data = await provider.get_all_macro_data()
                
                self.last_update = datetime.utcnow()
                logger.debug(f"📊 Macro data atualizado: {len(data)} campos")
                
            except Exception as e:
                logger.error(f"❌ Erro atualizando macro data: {e}")
            
            # Aguardar próximo ciclo
            await asyncio.sleep(self.update_interval)
    
    def get_status(self) -> dict:
        """Retorna status do serviço"""
        return {
            "running": self._running,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "update_interval": self.update_interval,
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