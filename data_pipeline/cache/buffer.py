# data_pipeline/cache/buffer.py
from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Set

try:
    import orjson
except ImportError:
    orjson = None  # type: ignore

try:
    import xxhash
except ImportError:
    xxhash = None  # type: ignore


@dataclass
class EventBatch:
    """
    Lote de eventos com metadados completos de rastreabilidade.

    Permite auditoria completa do ciclo de vida dos eventos:
    - Quando foram criados
    - Quando foram enviados
    - Quantos eventos foram dedupados
    - ID único rastreável

    Attributes:
        batch_id: ID único do lote (formato: sessionid-timestamp-counter)
        events: Lista de eventos do lote
        created_at: Timestamp de criação do primeiro evento
        flushed_at: Timestamp de envio do lote
        event_count: Quantidade de eventos no lote
        dedup_count: Quantidade de duplicatas removidas
    """

    batch_id: str
    events: List[Dict[str, Any]]
    created_at: float
    flushed_at: Optional[float] = None
    event_count: int = 0
    dedup_count: int = 0

    def __post_init__(self) -> None:
        """Calcula event_count automaticamente."""
        self.event_count = len(self.events)

    def to_dict(self) -> Dict[str, Any]:
        """
        Converte para dicionário com metadados completos.

        Returns:
            Dicionário com todas as informações do lote
        """
        duration_ms = None
        if self.flushed_at:
            duration_ms = (self.flushed_at - self.created_at) * 1000

        return {
            'batch_id': self.batch_id,
            'event_count': self.event_count,
            'dedup_count': self.dedup_count,
            'created_at': self.created_at,
            'created_at_iso': datetime.fromtimestamp(self.created_at).isoformat(),
            'flushed_at': self.flushed_at,
            'flushed_at_iso': datetime.fromtimestamp(self.flushed_at).isoformat() if self.flushed_at else None,
            'duration_ms': round(duration_ms, 2) if duration_ms else None,
            'events': self.events
        }

    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumo sem os eventos (para logging)."""
        result = self.to_dict()
        result.pop('events', None)
        return result


class EventBuffer:
    """
    Buffer circular de eventos com deduplicação e rastreabilidade.

    Características:
    - Deduplicação automática por checksum (ID ou hash)
    - Flush automático baseado em tamanho ou idade
    - Rastreamento completo com batch_id único
    - Histórico de lotes enviados
    - Estatísticas detalhadas

    Exemplo de uso:
        buffer = EventBuffer(max_size=100, max_age_seconds=60, min_events=20)

        # Adicionar eventos
        buffer.add({"type": "trade", "price": 67000})

        # Verificar se deve enviar
        if buffer.should_flush():
            batch = buffer.get_events()
            print(f"Enviando {batch.batch_id}: {batch.event_count} eventos")
    """

    def __init__(
        self,
        max_size: int = 100,
        max_age_seconds: int = 60,
        min_events: int = 20
    ) -> None:
        """
        Inicializa buffer de eventos.

        Args:
            max_size: Tamanho máximo do buffer
            max_age_seconds: Idade máxima em segundos antes do flush
            min_events: Quantidade mínima de eventos para considerar flush por idade
        """
        self.buffer: deque = deque(maxlen=max_size)
        self.event_checksums: Set[str] = set()
        self.max_age_seconds = max_age_seconds
        self.min_events = min_events
        self.first_event_time: Optional[float] = None

        # Lock para thread-safety
        self._lock = threading.Lock()

        # Rastreabilidade
        self._batch_counter = 0
        self._session_id = str(uuid.uuid4())[:8]
        self._batch_history: deque = deque(maxlen=100)

        # Estatísticas
        self.stats: Dict[str, Any] = {
            'total_received': 0,
            'duplicates_filtered': 0,
            'batches_sent': 0,
            'total_events_sent': 0,
            'session_id': self._session_id,
            'created_at': time.time()
        }

    def add(self, event: Dict[str, Any]) -> bool:
        """
        Adiciona evento ao buffer se não for duplicado.

        Otimizações:
        - Se o evento tiver um ID único (event_id, id, tradeId, trade_id),
          usa esse ID diretamente para deduplicação (sem serializar tudo).
        - Caso contrário, serializa com orjson (se disponível) e hasheia com xxhash (se disponível).
        """
        with self._lock:
            self.stats['total_received'] += 1

            # 1) Tentar usar ID único diretamente
            dedup_key: Optional[str] = None
            for k in ("event_id", "id", "tradeId", "trade_id"):
                v = event.get(k)
                if v is not None:
                    dedup_key = f"{k}:{v}"
                    break

            if dedup_key is not None:
                checksum = dedup_key
            else:
                # 2) Serializar evento e calcular hash rápido
                event_bytes = self._serialize_event(event)
                checksum = self._hash_bytes(event_bytes)

            # Verificar duplicata
            if checksum in self.event_checksums:
                self.stats['duplicates_filtered'] += 1
                return False

            # Adicionar ao buffer
            self.buffer.append({
                'data': event,
                'checksum': checksum,
                'timestamp': time.time()
            })
            self.event_checksums.add(checksum)

            # Marcar tempo do primeiro evento
            if self.first_event_time is None:
                self.first_event_time = time.time()

            # Limpar checksums antigos se buffer muito grande
            if len(self.event_checksums) > (self.buffer.maxlen or 100) * 2:
                self._cleanup_checksums()

            return True

    def should_flush(self, force: bool = False) -> bool:
        """
        Determina se o buffer deve ser enviado.

        Args:
            force: Se True, força flush se houver eventos

        Returns:
            True se deve fazer flush
        """
        with self._lock:
            if force and self.buffer:
                return True

            if not self.buffer:
                return False

            max_len = self.buffer.maxlen or 100

            # Flush se buffer 80% cheio
            if len(self.buffer) >= max_len * 0.8:
                return True

            # Flush se tiver eventos mínimos E idade suficiente
            if len(self.buffer) >= self.min_events:
                if self.first_event_time:
                    age = time.time() - self.first_event_time
                    if age > self.max_age_seconds:
                        return True

            return False

    def get_events(self, clear: bool = True) -> EventBatch:
        """
        Obtém eventos do buffer como um lote rastreável.

        Args:
            clear: Se True, limpa o buffer após obter eventos

        Returns:
            EventBatch com batch_id único e metadados completos
        """
        with self._lock:
            # Gerar batch_id rastreável
            timestamp = int(time.time())
            self._batch_counter += 1
            batch_id = f"{self._session_id}-{timestamp}-{self._batch_counter:04d}"

            # Criar lote
            events = [item['data'] for item in self.buffer]
            created_at = self.first_event_time or time.time()
            flushed_at = time.time()

            batch = EventBatch(
                batch_id=batch_id,
                events=events,
                created_at=created_at,
                flushed_at=flushed_at,
                dedup_count=self.stats['duplicates_filtered']
            )

            # Atualizar estatísticas
            self.stats['batches_sent'] += 1
            self.stats['total_events_sent'] += len(events)
            self.stats['last_batch_id'] = batch_id
            self.stats['last_flush_time'] = flushed_at

            # Armazenar histórico
            self._batch_history.append(batch)

            if clear:
                self.buffer.clear()
                self.first_event_time = None

            return batch

    def get_batch_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retorna histórico de lotes enviados.

        Args:
            limit: Quantidade máxima de lotes a retornar

        Returns:
            Lista com dicionários de metadados dos lotes
        """
        return [
            batch.get_summary()
            for batch in list(self._batch_history)[-limit:]
        ]

    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas completas do buffer.

        Returns:
            Dicionário com todas as estatísticas
        """
        current_age = (
            time.time() - self.first_event_time
            if self.first_event_time else 0
        )

        dedup_rate = (
            self.stats['duplicates_filtered'] /
            max(self.stats['total_received'], 1) * 100
        )

        avg_batch_size = (
            self.stats['total_events_sent'] /
            max(self.stats['batches_sent'], 1)
        )

        uptime = time.time() - self.stats['created_at']

        return {
            **self.stats,
            'current_size': len(self.buffer),
            'buffer_age_seconds': round(current_age, 2),
            'dedup_rate_pct': round(dedup_rate, 2),
            'avg_batch_size': round(avg_batch_size, 2),
            'uptime_seconds': round(uptime, 2),
            'events_per_second': round(
                self.stats['total_events_sent'] / max(uptime, 1), 2
            )
        }

    def _serialize_event(self, event: Dict[str, Any]) -> bytes:
        """
        Serializa evento para cálculo de checksum.

        Usa orjson se disponível (muito mais rápido que json padrão).
        """
        if orjson is not None:
            # OPT_SORT_KEYS garante determinismo
            return orjson.dumps(event, option=orjson.OPT_SORT_KEYS)
        # Fallback para json padrão
        return json.dumps(event, sort_keys=True, default=str).encode("utf-8")

    def _hash_bytes(self, data: bytes) -> str:
        """
        Calcula hash rápido de um blob de bytes.

        Usa xxhash se disponível; caso contrário, md5 como fallback.
        """
        if xxhash is not None:
            return xxhash.xxh64_hexdigest(data)[:16]
        return hashlib.md5(data).hexdigest()[:16]

    def _cleanup_checksums(self) -> None:
        """Remove checksums de eventos que já saíram do buffer."""
        current_checksums = {item['checksum'] for item in self.buffer}
        self.event_checksums = current_checksums