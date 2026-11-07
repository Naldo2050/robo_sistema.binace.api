#!/usr/bin/env python3
"""
scripts/ci/validate_deploy_gate.py - Gate Determinístico de Deploy OCI.

Valida as garantias de segurança antes do deploy em produção:
1. Validação de jobs obrigatórios do workflow CI com suporte a paginação (--paginate).
2. Validação exaustiva de matrizes de jobs (e.g. test-unit (3.11) e (3.12)).
3. Rejeição fail-closed se qualquer job obrigatório estiver ausente, skipped ou failed.
4. Validação de SHA estrito contra o HEAD de origin/main (imunidade a forks/divergência).
5. Tratamento explícito de head_repository nulo ou de repositório terceiro.
6. Tratamento de falhas de rede/git no checkout/fetch de forma fail-closed.
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_REQUIRED_JOBS = [
    "lint",
    "test-unit",
    "test-integration",
    "test-all-with-coverage",
]


def validate_head_repository(head_repo: Optional[str], expected_repo: str) -> Tuple[bool, List[str]]:
    """Valida que o evento originou-se do mesmo repositório e não de um fork ou contexto nulo."""
    errors = []
    if not head_repo:
        errors.append("head_repository é nulo ou vazio (possível fork não autenticado ou evento órfão).")
    elif head_repo.strip().lower() != expected_repo.strip().lower():
        errors.append(
            f"head_repository '{head_repo}' diverge do repositório esperado '{expected_repo}'."
        )
    return len(errors) == 0, errors


def validate_sha_alignment(head_sha: str, expected_sha: str) -> Tuple[bool, List[str]]:
    """Valida se o commit do workflow_run coincide com o commit esperado (origin/main)."""
    errors = []
    if not head_sha or not expected_sha:
        errors.append("SHA do commit ou SHA de origin/main ausente para validação.")
    elif head_sha.strip().lower() != expected_sha.strip().lower():
        errors.append(
            f"SHA do workflow_run ({head_sha[:10]}) diverge de origin/main ({expected_sha[:10]}). "
            "A branch main avançou ou o build rodou em commit obsoleto."
        )
    return len(errors) == 0, errors


def get_origin_main_sha(repo_dir: str = ".") -> Tuple[Optional[str], Optional[str]]:
    """Obtém o SHA de origin/main localmente via git."""
    try:
        # Primeiro tenta rev-parse direto se refs remotas já foram fetched
        cmd = ["git", "rev-parse", "origin/main"]
        res = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip(), None

        # Se falhar, tenta fetch explícito
        fetch_cmd = ["git", "fetch", "origin", "main"]
        fetch_res = subprocess.run(fetch_cmd, cwd=repo_dir, capture_output=True, text=True)
        if fetch_res.returncode != 0:
            return None, f"Falha no git fetch origin main: {fetch_res.stderr.strip()}"

        res = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip(), None
        return None, f"origin/main não encontrado após fetch: {res.stderr.strip()}"
    except Exception as e:
        return None, f"Exceção ao resolver origin/main via git: {e}"


def fetch_jobs_from_github_api(
    repo: str,
    run_id: int,
    token: str,
    per_page: int = 100,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    Busca todas as páginas de jobs de um workflow_run usando a API REST do GitHub.
    Lida com paginação transparente (per_page até 100).
    """
    if not token:
        return None, "GITHUB_TOKEN não fornecido para consulta à API."

    all_jobs: List[Dict[str, Any]] = []
    page = 1

    while True:
        url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs?per_page={per_page}&page={page}"
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "oci-deploy-gate",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return None, f"Erro HTTP ao consultar jobs do run {run_id} na página {page}: {e.code} {e.reason}"
        except urllib.error.URLError as e:
            return None, f"Erro de rede ao consultar jobs do run {run_id}: {e.reason}"
        except Exception as e:
            return None, f"Falha inesperada ao consultar jobs do run {run_id}: {e}"

        jobs = data.get("jobs", [])
        all_jobs.extend(jobs)

        total_count = data.get("total_count", 0)
        if len(all_jobs) >= total_count or not jobs:
            break
        page += 1

    return all_jobs, None


def validate_jobs_payload(
    jobs: List[Dict[str, Any]],
    required_job_names: List[str],
) -> Tuple[bool, List[str]]:
    """
    Valida a lista de jobs:
    1. Garante que cada nome obrigatório esteja presente.
    2. Suporta matrizes: se existir 'test-unit (3.11)' e 'test-unit (3.12)', ambos são verificados.
    3. Rejeita se status != completed ou conclusion != success.
    """
    errors: List[str] = []
    jobs_by_base_name: Dict[str, List[Dict[str, Any]]] = {}

    for job in jobs:
        name = job.get("name", "")
        # Agrupa pelo nome base (removendo sufixos de matriz como ' (3.11)')
        base_name = name.split(" (")[0].strip()
        jobs_by_base_name.setdefault(base_name, []).append(job)
        jobs_by_base_name.setdefault(name, []).append(job)

    for req in required_job_names:
        matching_jobs = jobs_by_base_name.get(req, [])
        if not matching_jobs:
            errors.append(f"Job obrigatório ausente: '{req}' não encontrado no payload de jobs.")
            continue

        for mj in matching_jobs:
            name = mj.get("name", req)
            status = mj.get("status", "")
            conclusion = mj.get("conclusion", "")

            if status != "completed":
                errors.append(f"Job '{name}' ainda não está concluído (status: '{status}').")
            elif conclusion != "success":
                errors.append(
                    f"Job '{name}' não foi aprovado com sucesso (conclusion: '{conclusion}')."
                )

    return len(errors) == 0, errors


def run_gate(
    event_name: str,
    expected_repo: str,
    head_repo: Optional[str],
    head_sha: str,
    run_id: int,
    required_jobs: List[str],
    token: str,
    jobs_json_file: Optional[str] = None,
    expected_sha: Optional[str] = None,
) -> int:
    """Executa todas as verificações do gate e retorna exit code (0 aprovado, 1 reprovado)."""
    errors: List[str] = []

    print("=== DEPLOY GATE VALIDATOR ===")
    print(f"Evento: {event_name}")
    print(f"Repositório: {expected_repo}")
    print(f"Required Jobs: {required_jobs}")

    # 1. Validação de dispatch manual desativada nesta rodada
    if event_name == "workflow_dispatch":
        print("Disparo manual workflow_dispatch bloqueado nesta rodada.", file=sys.stderr)
        return 1

    # 2. Validação de repositório oficial / fork
    ok_repo, repo_errs = validate_head_repository(head_repo, expected_repo)
    if not ok_repo:
        errors.extend(repo_errs)

    # 3. Validação de alinhamento de SHA com origin/main
    origin_sha = expected_sha
    if not origin_sha:
        origin_sha, sha_err = get_origin_main_sha()
        if sha_err:
            errors.append(f"Falha ao validar SHA de origin/main: {sha_err}")
    if origin_sha:
        ok_sha, sha_errs = validate_sha_alignment(head_sha, origin_sha)
        if not ok_sha:
            errors.extend(sha_errs)

    # 4. Obtenção e validação dos jobs do workflow
    if jobs_json_file:
        try:
            with open(jobs_json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            jobs = data if isinstance(data, list) else data.get("jobs", [])
            err_jobs = None
        except Exception as e:
            jobs, err_jobs = None, f"Erro ao ler arquivo JSON de jobs: {e}"
    else:
        jobs, err_jobs = fetch_jobs_from_github_api(
            repo=expected_repo,
            run_id=run_id,
            token=token,
        )

    if err_jobs:
        errors.append(f"Falha ao obter lista de jobs: {err_jobs}")
    elif jobs is not None:
        ok_jobs, job_errs = validate_jobs_payload(jobs, required_jobs)
        if not ok_jobs:
            errors.extend(job_errs)

    if errors:
        print("\n❌ DEPLOY GATE REPROVADO (FAIL-CLOSED):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("\n✅ DEPLOY GATE APROVADO: Todos os checks de segurança foram satisfeitos com sucesso.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Validador determinístico para gate de deploy OCI.")
    parser.add_argument("--event-name", required=True, help="Nome do evento do GitHub Actions (e.g. workflow_run)")
    parser.add_argument("--expected-repo", default="Naldo2050/robo_sistema.binace.api", help="Repositório esperado")
    parser.add_argument("--head-repo", default="", help="Repositório de origem do build")
    parser.add_argument("--head-sha", required=True, help="SHA do commit executado")
    parser.add_argument("--run-id", type=int, default=0, help="ID do workflow_run no GitHub Actions")
    parser.add_argument("--jobs-json-file", default="", help="Arquivo JSON de jobs para execução local/offline")
    parser.add_argument("--required-jobs", default=",".join(DEFAULT_REQUIRED_JOBS), help="Lista de jobs obrigatórios separados por vírgula")
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""), help="GitHub Token")
    parser.add_argument("--expected-sha", default="", help="SHA de origin/main (opcional se puder resolver via git)")
    args = parser.parse_args()

    required_jobs_list = [j.strip() for j in args.required_jobs.split(",") if j.strip()]

    code = run_gate(
        event_name=args.event_name,
        expected_repo=args.expected_repo,
        head_repo=args.head_repo,
        head_sha=args.head_sha,
        run_id=args.run_id,
        required_jobs=required_jobs_list,
        token=args.token,
        jobs_json_file=args.jobs_json_file or None,
        expected_sha=args.expected_sha or None,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
