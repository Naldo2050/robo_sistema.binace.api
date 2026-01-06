# flow_analyzer/profiling.py
"""
Profiling e monitoramento de memória do FlowAnalyzer.

Inclui:
- MemoryProfiler: Tracking de uso de memória
- LockProfiler: Monitoramento de contenção de locks
- PerformanceProfiler: Benchmarking

DEPENDÊNCIAS OPCIONAIS:
- psutil: Para memory info no Windows (pip install psutil)
- tracemalloc: Incluído no Python 3.4+
"""

from __future__ import annotations

import gc
import sys
import time
import threading
import tracemalloc
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, TYPE_CHECKING
from contextlib import contextmanager

# Type checking imports (não executados em runtime)
if TYPE_CHECKING:
    import psutil as psutil_module
    import resource as resource_module


@dataclass
class MemorySnapshot:
    """Snapshot de uso de memória."""
    timestamp_ms: int
    current_bytes: int
    peak_bytes: int
    traced_objects: int
    gc_counts: tuple


class MemoryProfiler:
    """
    Profiler de memória com suporte a tracemalloc.
    
    Features:
    - Tracking de memória atual e pico
    - Snapshots periódicos
    - Detecção de leaks
    - Top alocações por arquivo
    
    Example:
        >>> profiler = MemoryProfiler()
        >>> profiler.start()
        >>> # ... código ...
        >>> stats = profiler.get_stats()
        >>> profiler.stop()
    """
    
    def __init__(self, snapshot_interval_sec: float = 60.0, max_snapshots: int = 100):
        self.snapshot_interval_sec = snapshot_interval_sec
        self.max_snapshots = max_snapshots
        self._snapshots: deque = deque(maxlen=max_snapshots)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._start_time: Optional[float] = None
        self._baseline_bytes: int = 0
    
    def start(self) -> None:
        """Inicia profiling de memória."""
        if self._running:
            return
        
        tracemalloc.start()
        self._start_time = time.time()
        self._baseline_bytes = self._get_current_memory()
        self._running = True
        
        # Thread de snapshots periódicos
        self._thread = threading.Thread(target=self._snapshot_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> Dict[str, Any]:
        """Para profiling e retorna estatísticas finais."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        
        stats = self.get_stats()
        tracemalloc.stop()
        return stats
    
    def take_snapshot(self) -> MemorySnapshot:
        """Tira snapshot manual."""
        current, peak = tracemalloc.get_traced_memory()
        
        snapshot = MemorySnapshot(
            timestamp_ms=int(time.time() * 1000),
            current_bytes=current,
            peak_bytes=peak,
            traced_objects=len(gc.get_objects()),
            gc_counts=gc.get_count(),
        )
        
        with self._lock:
            self._snapshots.append(snapshot)
        
        return snapshot
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de memória."""
        current, peak = tracemalloc.get_traced_memory()
        
        with self._lock:
            snapshots = list(self._snapshots)
        
        # Calcula growth
        growth_bytes = current - self._baseline_bytes
        runtime_sec = time.time() - (self._start_time or time.time())
        growth_rate = growth_bytes / max(runtime_sec, 1)  # bytes/sec
        
        return {
            'current_bytes': current,
            'current_mb': round(current / (1024 * 1024), 2),
            'peak_bytes': peak,
            'peak_mb': round(peak / (1024 * 1024), 2),
            'baseline_bytes': self._baseline_bytes,
            'growth_bytes': growth_bytes,
            'growth_rate_bytes_per_sec': round(growth_rate, 2),
            'runtime_sec': round(runtime_sec, 2),
            'snapshot_count': len(snapshots),
            'gc_counts': gc.get_count(),
            'gc_thresholds': gc.get_threshold(),
        }
    
    def get_top_allocations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retorna top alocações por arquivo/linha."""
        try:
            snapshot = tracemalloc.take_snapshot()
            stats = snapshot.statistics('lineno')
            
            return [
                {
                    'file': str(stat.traceback),
                    'size_bytes': stat.size,
                    'size_kb': round(stat.size / 1024, 2),
                    'count': stat.count,
                }
                for stat in stats[:limit]
            ]
        except Exception:
            return []
    
    def detect_leak(self, threshold_mb: float = 100.0) -> Optional[Dict[str, Any]]:
        """
        Detecta possível memory leak.
        
        Args:
            threshold_mb: Threshold de crescimento para considerar leak
            
        Returns:
            Dict com detalhes do leak, ou None se OK
        """
        stats = self.get_stats()
        growth_mb = stats['growth_bytes'] / (1024 * 1024)
        
        if growth_mb > threshold_mb:
            return {
                'leak_detected': True,
                'growth_mb': round(growth_mb, 2),
                'threshold_mb': threshold_mb,
                'runtime_sec': stats['runtime_sec'],
                'top_allocations': self.get_top_allocations(5),
            }
        return None
    
    def _get_current_memory(self) -> int:
        """
        Obtém memória atual do processo.
        
        Tenta usar:
        1. resource (Unix)
        2. psutil (Windows/cross-platform)
        3. Fallback para 0
        """
        # Tenta resource (Unix)
        try:
            import resource  # type: ignore[import-not-found]
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        except (ImportError, AttributeError):
            pass
        
        # Tenta psutil (Windows/cross-platform)
        try:
            import psutil  # type: ignore[import-not-found]
            return psutil.Process().memory_info().rss
        except ImportError:
            pass
        
        # Fallback
        return 0
    
    def _snapshot_loop(self) -> None:
        """Loop de snapshots periódicos."""
        while self._running:
            time.sleep(self.snapshot_interval_sec)
            if self._running:
                self.take_snapshot()


@dataclass
class LockStats:
    """Estatísticas de um lock."""
    acquisitions: int = 0
    contentions: int = 0
    total_wait_ms: float = 0.0
    max_wait_ms: float = 0.0
    total_hold_ms: float = 0.0
    max_hold_ms: float = 0.0


class LockProfiler:
    """
    Profiler de contenção de locks.
    
    Monitora:
    - Número de aquisições
    - Contenções (quando precisou esperar)
    - Tempo de espera
    - Tempo de retenção
    
    Example:
        >>> profiler = LockProfiler()
        >>> with profiler.track("main_lock"):
        ...     # código protegido
        >>> stats = profiler.get_stats()
    """
    
    def __init__(self):
        self._stats: Dict[str, LockStats] = {}
        self._meta_lock = threading.Lock()
    
    @contextmanager
    def track(self, lock_name: str, lock: Optional[threading.Lock] = None):
        """
        Context manager para rastrear aquisição de lock.
        
        Args:
            lock_name: Nome identificador do lock
            lock: Lock real a ser adquirido (opcional)
        """
        # Inicializa stats se necessário
        with self._meta_lock:
            if lock_name not in self._stats:
                self._stats[lock_name] = LockStats()
            stats = self._stats[lock_name]
        
        # Mede tempo de espera
        start_wait = time.perf_counter()
        
        if lock:
            # Tenta adquirir sem bloquear para detectar contenção
            acquired = lock.acquire(blocking=False)
            if not acquired:
                stats.contentions += 1
                lock.acquire()  # Bloqueia agora
        
        wait_time = (time.perf_counter() - start_wait) * 1000
        stats.acquisitions += 1
        stats.total_wait_ms += wait_time
        stats.max_wait_ms = max(stats.max_wait_ms, wait_time)
        
        # Mede tempo de retenção
        start_hold = time.perf_counter()
        
        try:
            yield
        finally:
            hold_time = (time.perf_counter() - start_hold) * 1000
            stats.total_hold_ms += hold_time
            stats.max_hold_ms = max(stats.max_hold_ms, hold_time)
            
            if lock:
                lock.release()
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Retorna estatísticas de todos os locks."""
        with self._meta_lock:
            result = {}
            for name, stats in self._stats.items():
                avg_wait = stats.total_wait_ms / max(stats.acquisitions, 1)
                avg_hold = stats.total_hold_ms / max(stats.acquisitions, 1)
                contention_rate = stats.contentions / max(stats.acquisitions, 1)
                
                result[name] = {
                    'acquisitions': stats.acquisitions,
                    'contentions': stats.contentions,
                    'contention_rate': round(contention_rate, 4),
                    'avg_wait_ms': round(avg_wait, 3),
                    'max_wait_ms': round(stats.max_wait_ms, 3),
                    'avg_hold_ms': round(avg_hold, 3),
                    'max_hold_ms': round(stats.max_hold_ms, 3),
                }
            return result
    
    def reset(self) -> None:
        """Reseta todas as estatísticas."""
        with self._meta_lock:
            self._stats.clear()


class PerformanceProfiler:
    """
    Profiler de performance para benchmarks.
    
    Features:
    - Timing de operações
    - Throughput (ops/sec)
    - Percentis
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        self._times: deque = deque(maxlen=10000)
        self._total_ops = 0
        self._start_time = time.time()
        self._lock = threading.Lock()
    
    @contextmanager
    def measure(self):
        """Context manager para medir operação."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            with self._lock:
                self._times.append(elapsed)
                self._total_ops += 1
    
    def record(self, duration_ms: float) -> None:
        """Registra duração manualmente."""
        with self._lock:
            self._times.append(duration_ms)
            self._total_ops += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de performance."""
        with self._lock:
            if not self._times:
                return {'name': self.name, 'ops': 0}
            
            sorted_times = sorted(self._times)
            n = len(sorted_times)
            runtime = time.time() - self._start_time
            
            return {
                'name': self.name,
                'ops': self._total_ops,
                'ops_per_sec': round(self._total_ops / max(runtime, 1), 2),
                'runtime_sec': round(runtime, 2),
                'p50_ms': round(sorted_times[int(n * 0.5)], 3),
                'p95_ms': round(sorted_times[min(int(n * 0.95), n - 1)], 3),
                'p99_ms': round(sorted_times[min(int(n * 0.99), n - 1)], 3),
                'max_ms': round(sorted_times[-1], 3),
                'min_ms': round(sorted_times[0], 3),
                'avg_ms': round(sum(sorted_times) / n, 3),
            }
    
    def reset(self) -> None:
        """Reseta estatísticas."""
        with self._lock:
            self._times.clear()
            self._total_ops = 0
            self._start_time = time.time()


def run_benchmark(
    func: Callable[[], Any],
    iterations: int = 10000,
    warmup: int = 1000,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executa benchmark de uma função.
    
    Args:
        func: Função a ser benchmarkada (sem argumentos)
        iterations: Número de iterações
        warmup: Iterações de warmup (não contadas)
        name: Nome do benchmark
        
    Returns:
        Estatísticas do benchmark
    """
    profiler = PerformanceProfiler(name or getattr(func, '__name__', 'benchmark'))
    
    # Warmup
    for _ in range(warmup):
        func()
    
    # Benchmark
    profiler.reset()
    for _ in range(iterations):
        with profiler.measure():
            func()
    
    stats = profiler.get_stats()
    stats['iterations'] = iterations
    stats['warmup'] = warmup
    
    return stats


# ==============================================================================
# HELPERS PARA VERIFICAR DEPENDÊNCIAS
# ==============================================================================

def has_psutil() -> bool:
    """Verifica se psutil está disponível."""
    try:
        import psutil  # type: ignore[import-not-found]
        return True
    except ImportError:
        return False


def has_resource() -> bool:
    """Verifica se resource está disponível (Unix)."""
    try:
        import resource  # type: ignore[import-not-found]
        return True
    except ImportError:
        return False


def get_memory_info() -> Dict[str, Any]:
    """
    Obtém informações de memória do processo (helper standalone).
    
    Returns:
        Dict com rss, vms, percent (se disponível)
    """
    result: Dict[str, Any] = {'available': False}
    
    try:
        import psutil  # type: ignore[import-not-found]
        proc = psutil.Process()
        mem = proc.memory_info()
        result = {
            'available': True,
            'rss_bytes': mem.rss,
            'rss_mb': round(mem.rss / (1024 * 1024), 2),
            'vms_bytes': mem.vms,
            'vms_mb': round(mem.vms / (1024 * 1024), 2),
            'percent': round(proc.memory_percent(), 2),
        }
    except ImportError:
        try:
            import resource  # type: ignore[import-not-found]
            usage = resource.getrusage(resource.RUSAGE_SELF)
            result = {
                'available': True,
                'rss_bytes': usage.ru_maxrss * 1024,
                'rss_mb': round(usage.ru_maxrss * 1024 / (1024 * 1024), 2),
            }
        except ImportError:
            pass
    
    return result