from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class OrchestratorSnapshotProtocol(Protocol):
    """
    Contrato mínimo para expor estado operacional normalizado.

    A consolidação dos orquestradores deve começar por uma interface de leitura
    e ciclo de vida estável antes de tentar unificar as implementações.
    """

    symbol: str

    def snapshot_state(self) -> Dict[str, Any]:
        """Retorna um snapshot normalizado do runtime."""
        ...


@runtime_checkable
class OrchestratorControlProtocol(OrchestratorSnapshotProtocol, Protocol):
    """Contrato mínimo de controle para adapters de orquestradores."""

    async def start_runtime(self) -> None:
        """Inicia o runtime subjacente."""
        ...

    async def stop_runtime(self) -> None:
        """Encerra o runtime subjacente."""
        ...
