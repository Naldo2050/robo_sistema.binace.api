# -*- coding: utf-8 -*-
"""
Harness de Testes de Integração Hermética da CLI Apply — Gate G4-D B3-B2-C2-B2.4-B-H2.

Módulo exclusivo de infraestrutura de testes (tests/helpers/).
Não faz parte do código de produção e não é exportado por scripts operacionais.

Garantias Arquiteturais e Contenção Positiva:
1. Contenção Positiva Obrigatória: Toda execução exige um sandbox_root de teste explícito.
   Qualquer tentativa de execução sem sandbox ou com arquivos fora de sandbox_root é rejeitada
   antes de qualquer conexão ou abertura financeira.
2. Imunidade a Instalações Externas: Rejeita qualquer caminho pertencente à produção local ou
   a qualquer caminho fora do sandbox_root fornecido pelo harness.
3. Validação Rigorosa de Symlinks Pré-Resolução: Inspeciona a representação original e todos os
   seus nós componentes antes de chamar Path.resolve(), evitando que symlinks intermediários
   sejam suprimidos antes da validação.
4. Mapeamento de Exit Codes e Classificação Fail-Closed: Preserva os exit codes formais do SO
   (0, 1, 3, 4, 5) sob governança estrita e fail-closed.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.runtime_lock import (
    MANIFEST_FILENAME,
    OperationalRuntimeLock,
    RuntimeState,
    read_runtime_manifest,
)
from events.fallback_reconciler import (
    RECONCILIATION_AUDIT_FILENAME,
    FallbackReconciler,
    ReconciliationTransactionStatus,
    execute_reconciliation_transaction_hermetic,
)


class HermeticSandboxContext:
    """
    Contexto de contenção positiva para execuções herméticas em testes.
    Exige que a raiz do sandbox seja fornecida explicitamente pelo ambiente de teste
    (ex.: tmp_path do pytest) e valida que todos os 9 caminhos estejam contidos nele.
    """

    def __init__(self, sandbox_root: Optional[Union[Path, str]]):
        if sandbox_root is None:
            raise PermissionError("CONTESTAÇÃO DE SANDBOX: É obrigatório fornecer um sandbox_root de teste explícito.")

        self.raw_root = Path(sandbox_root)
        if not self.raw_root.exists() or not self.raw_root.is_dir():
            raise PermissionError(f"CONTESTAÇÃO DE SANDBOX: sandbox_root não existe ou não é diretório: {self.raw_root}")

        self.canonical_root = self.raw_root.resolve()

        # Bloqueio incondicional contra raiz do projeto e diretório operacional real
        immutable_prod = (PROJECT_ROOT / "dados").resolve()
        if (
            self.canonical_root == immutable_prod
            or immutable_prod in self.canonical_root.parents
            or self.canonical_root == PROJECT_ROOT.resolve()
            or PROJECT_ROOT.resolve() in self.canonical_root.parents
        ):
            raise PermissionError(
                f"CONTESTAÇÃO DE SANDBOX: sandbox_root não pode englobar ou coincidir com a produção operacional ({self.canonical_root})."
            )

    def check_symlinks_pre_resolve(self, label: str, p: Union[Path, str]) -> Optional[str]:
        """Inspeciona symlinks no caminho original antes de qualquer resolve()."""
        raw_p = Path(p)

        # 1. Nó final
        if raw_p.is_symlink():
            return f"Origem operacional insegura: link simbólico detectado em '{label}': {raw_p}"

        # 2. Componentes intermediários existentes na hierarquia original
        curr = raw_p if raw_p.is_absolute() else raw_p.absolute()
        for parent in curr.parents:
            if parent.exists() and parent.is_symlink():
                return f"Origem operacional insegura: link simbólico intermediário detectado no diretório pai de '{label}': {parent}"

        return None

    def validate_and_canonicalize(self, label: str, p: Union[Path, str]) -> Path:
        """Aplica resolve() e valida estritamente a contenção positiva dentro de sandbox_root."""
        raw_p = Path(p)
        canonical_p = raw_p.expanduser().resolve()

        try:
            canonical_p.relative_to(self.canonical_root)
        except ValueError:
            raise PermissionError(
                f"CONTESTAÇÃO DE SANDBOX: O caminho '{label}' ({canonical_p}) está fora do sandbox autorizado ({self.canonical_root}). Operação proibida."
            )

        return canonical_p


def execute_apply_integration_hermetic(
    fallback_file: Union[Path, str],
    sqlite_path: Union[Path, str],
    jsonl_path: Union[Path, str],
    archive_dir: Union[Path, str],
    backup_dir: Optional[Union[Path, str]] = None,
    lock_path: Optional[Union[Path, str]] = None,
    manifest_path: Optional[Union[Path, str]] = None,
    ledger_path: Optional[Union[Path, str]] = None,
    operator_reason: str = "Reconciliacao hermetica autorizada C2-B2.4-B-H2",
    operational_lock: Optional[OperationalRuntimeLock] = None,
    audit_id: Optional[str] = None,
    repo_root: Optional[Union[Path, str]] = None,
    sandbox_root: Optional[Union[Path, str]] = None,
) -> Tuple[int, Dict[str, Any]]:
    """
    Adaptador de testes para integração hermética com o orquestrador transacional.
    Uso exclusivo da infraestrutura de testes.
    """
    # =========================================================================
    # 1. VALIDAÇÃO DE CONTEXTO DO SANDBOX (CONTENÇÃO POSITIVA)
    # =========================================================================
    sandbox_ctx = HermeticSandboxContext(sandbox_root)

    # Derivação inicial de caminhos não resolvidos para auditoria pré-resolve
    raw_sqlite = Path(sqlite_path)
    raw_data = raw_sqlite.parent
    raw_jsonl = Path(jsonl_path)
    raw_archive = Path(archive_dir)
    raw_fallback = Path(fallback_file)

    raw_manifest = Path(manifest_path) if manifest_path else (raw_data / MANIFEST_FILENAME)
    raw_ledger = Path(ledger_path) if ledger_path else (raw_data / RECONCILIATION_AUDIT_FILENAME)
    raw_lock = Path(lock_path) if lock_path else (raw_data / ".operational_runtime.lock")
    raw_backup = Path(backup_dir) if backup_dir else (raw_data.parent / "backups" / "reconciliation_snapshots")

    all_raw_paths = [
        ("data_dir", raw_data),
        ("sqlite_path", raw_sqlite),
        ("jsonl_path", raw_jsonl),
        ("archive_dir", raw_archive),
        ("fallback_file", raw_fallback),
        ("manifest_path", raw_manifest),
        ("ledger_path", raw_ledger),
        ("lock_path", raw_lock),
        ("backup_dir", raw_backup),
    ]

    # =========================================================================
    # 2. INSPEÇÃO RIGOROSA DE SYMLINKS PRÉ-RESOLVE (NÓS FINAIS E INTERMEDIÁRIOS)
    # =========================================================================
    for label, raw_item in all_raw_paths:
        symlink_err = sandbox_ctx.check_symlinks_pre_resolve(label, raw_item)
        if symlink_err:
            return 1, {
                "status": ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED.value,
                "exit_code": 1,
                "stage_reached": "SECURITY_VALIDATION",
                "errors": [symlink_err],
                "requires_manual_recovery": False,
            }

    # =========================================================================
    # 3. CANONICALIZAÇÃO E VALIDAÇÃO DE CONTENÇÃO POSITIVA
    # =========================================================================
    p_data = sandbox_ctx.validate_and_canonicalize("data_dir", raw_data)
    p_sqlite = sandbox_ctx.validate_and_canonicalize("sqlite_path", raw_sqlite)
    p_jsonl = sandbox_ctx.validate_and_canonicalize("jsonl_path", raw_jsonl)
    p_archive = sandbox_ctx.validate_and_canonicalize("archive_dir", raw_archive)
    p_fallback = sandbox_ctx.validate_and_canonicalize("fallback_file", raw_fallback)
    p_manifest = sandbox_ctx.validate_and_canonicalize("manifest_path", raw_manifest)
    p_ledger = sandbox_ctx.validate_and_canonicalize("ledger_path", raw_ledger)
    p_lock = sandbox_ctx.validate_and_canonicalize("lock_path", raw_lock)
    p_backup = sandbox_ctx.validate_and_canonicalize("backup_dir", raw_backup)

    # Coerência do cluster de dados: todos os artefatos de dados pertencem a data_dir
    for label, p_target in [
        ("sqlite_path", p_sqlite),
        ("jsonl_path", p_jsonl),
        ("manifest_path", p_manifest),
        ("ledger_path", p_ledger),
        ("lock_path", p_lock),
    ]:
        if p_target.parent != p_data:
            raise PermissionError(
                f"INCONSISTÊNCIA DE SANDBOX: O artefato '{label}' ({p_target}) deve residir estritamente dentro de data_dir ({p_data})."
            )

    if p_backup == p_data:
        raise PermissionError(
            f"INCONSISTÊNCIA DE SANDBOX: backup_dir ({p_backup}) não pode ser idêntico a data_dir."
        )

    # =========================================================================
    # 4. PRÉ-VALIDAÇÕES E REGRAS DE NEGÓCIO
    # =========================================================================
    if not p_fallback.exists() or not p_fallback.is_file():
        return 1, {
            "status": ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED.value,
            "exit_code": 1,
            "stage_reached": "PREFLIGHT",
            "errors": [f"Arquivo de fallback não encontrado ou inválido: {p_fallback}"],
            "requires_manual_recovery": False,
        }

    if p_fallback.suffix.lower() not in (".jsonl", ".json"):
        return 1, {
            "status": ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED.value,
            "exit_code": 1,
            "stage_reached": "PREFLIGHT",
            "errors": [f"Extensão de arquivo fallback não suportada ({p_fallback.suffix}). Exigido .jsonl ou .json."],
            "requires_manual_recovery": False,
        }

    # Validação de fixture versionada no Git
    canonical_repo = Path(repo_root).resolve() if repo_root else sandbox_ctx.canonical_root
    reconciler_helper = FallbackReconciler(
        sqlite_path=p_sqlite,
        jsonl_path=p_jsonl,
        archive_dir=p_archive,
        repo_root=canonical_repo,
    )
    if reconciler_helper.is_git_tracked_fixture(p_fallback):
        return 1, {
            "status": ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED.value,
            "exit_code": 1,
            "stage_reached": "SECURITY_VALIDATION",
            "errors": [f"Arquivo '{p_fallback.name}' é uma fixture versionada no Git. Modificação proibida."],
            "requires_manual_recovery": False,
        }

    # Leitura e ancoragem na identidade do ambiente via manifesto
    manifest_data = read_runtime_manifest(p_manifest)
    if not manifest_data or not manifest_data.get("environment_uuid"):
        return 1, {
            "status": ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED.value,
            "exit_code": 1,
            "stage_reached": "PARAM_VALIDATION",
            "errors": [f"Manifesto operacional inválido, ausente ou corrompido em {p_manifest}. Identidade do ambiente não pôde ser comprovada."],
            "requires_manual_recovery": False,
        }
    env_uuid = str(manifest_data["environment_uuid"]).strip()

    clean_reason = (operator_reason or "").strip()
    if len(clean_reason) < 10:
        return 1, {
            "status": ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED.value,
            "exit_code": 1,
            "stage_reached": "PARAM_VALIDATION",
            "errors": ["operator_reason deve conter justificativa auditável com pelo menos 10 caracteres."],
            "requires_manual_recovery": False,
        }

    # Modelo de Lock Opção A: objeto destravado delegado ao orquestrador
    target_lock = operational_lock or OperationalRuntimeLock(lock_path=p_lock, holder_name="fallback_reconciler_cli_hermetic")

    # Invocação do orquestrador transacional hermético
    res = execute_reconciliation_transaction_hermetic(
        data_dir=p_data,
        sqlite_path=p_sqlite,
        jsonl_path=p_jsonl,
        archive_dir=p_archive,
        source_fallback_file=p_fallback,
        manifest_path=p_manifest,
        ledger_path=p_ledger,
        backup_base_dir=p_backup,
        environment_uuid=env_uuid,
        operator_reason=clean_reason,
        audit_id=audit_id,
        operational_lock=target_lock,
        repo_root=canonical_repo,
    )

    # Mapeamento formal de Exit Codes
    if res.status == ReconciliationTransactionStatus.SUCCESS:
        exit_code = 0
    elif res.status == ReconciliationTransactionStatus.NO_CHANGES:
        exit_code = 0
    elif res.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED:
        is_lock_conflict = False
        if res.stage_reached == "LOCK_ACQUISITION":
            is_lock_conflict = any(
                "Conflito de exclusividade" in str(e) or "retido por processo ativo" in str(e)
                for e in res.errors
            )
        exit_code = 3 if is_lock_conflict else 1
    elif res.status == ReconciliationTransactionStatus.PREPARED_INCOMPLETE:
        exit_code = 4
    elif res.status == ReconciliationTransactionStatus.PERSISTENCE_UNCERTAIN:
        exit_code = 5
    else:
        exit_code = 1

    payload = {
        "status": res.status.value,
        "exit_code": exit_code,
        "audit_id": res.audit_id,
        "environment_uuid": res.environment_uuid,
        "stage_reached": res.stage_reached,
        "prepared_recorded": res.prepared_recorded,
        "committed_recorded": res.committed_recorded,
        "manifest_updated": res.manifest_updated,
        "sqlite_inserted": res.sqlite_inserted,
        "jsonl_appended": res.jsonl_appended,
        "errors": res.errors,
        "details": res.details,
        "requires_manual_recovery": res.status in (
            ReconciliationTransactionStatus.PREPARED_INCOMPLETE,
            ReconciliationTransactionStatus.PERSISTENCE_UNCERTAIN,
        ),
    }

    return exit_code, payload
