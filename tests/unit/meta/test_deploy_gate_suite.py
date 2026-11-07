import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ci.validate_deploy_gate import (  # noqa: E402
    run_gate,
    validate_head_repository,
    validate_jobs_payload,
    validate_sha_alignment,
)


def test_job_ausente_rejeitado():
    """Valida que job obrigatório ausente causa reprovação fail-closed."""
    jobs = [
        {"name": "test-unit", "status": "completed", "conclusion": "success"},
        {"name": "lint", "status": "completed", "conclusion": "success"},
    ]
    required = ["test-unit", "test-all", "lint"]
    ok, errors = validate_jobs_payload(jobs, required)
    assert not ok
    assert any("test-all" in e for e in errors)


def test_job_skipped_rejeitado():
    """Valida que job obrigatório pulado (skipped) causa reprovação."""
    jobs = [
        {"name": "test-unit", "status": "completed", "conclusion": "success"},
        {"name": "test-all", "status": "completed", "conclusion": "skipped"},
    ]
    required = ["test-unit", "test-all"]
    ok, errors = validate_jobs_payload(jobs, required)
    assert not ok
    assert any("skipped" in e for e in errors)


def test_job_failure_rejeitado():
    """Valida que job obrigatório com falha causa reprovação imediata."""
    jobs = [
        {"name": "test-unit", "status": "completed", "conclusion": "failure"},
    ]
    required = ["test-unit"]
    ok, errors = validate_jobs_payload(jobs, required)
    assert not ok
    assert any("não foi aprovado com sucesso" in e for e in errors)


def test_matriz_2_entradas_todas_aprovadas():
    """Valida matriz com 2+ entradas onde todas passam com sucesso."""
    jobs = [
        {"name": "test-unit (3.11)", "status": "completed", "conclusion": "success"},
        {"name": "test-unit (3.12)", "status": "completed", "conclusion": "success"},
        {"name": "test-all", "status": "completed", "conclusion": "success"},
    ]
    required = ["test-unit", "test-all"]
    ok, errors = validate_jobs_payload(jobs, required)
    assert ok
    assert len(errors) == 0


def test_matriz_2_entradas_uma_com_falha():
    """Valida matriz com 2+ entradas onde uma falha -> reprova tudo."""
    jobs = [
        {"name": "test-unit (3.11)", "status": "completed", "conclusion": "success"},
        {"name": "test-unit (3.12)", "status": "completed", "conclusion": "failure"},
    ]
    required = ["test-unit"]
    ok, errors = validate_jobs_payload(jobs, required)
    assert not ok
    assert any("test-unit (3.12)" in e for e in errors)


def test_paginacao_jobs_encontra_job_em_pagina_posterior():
    """Valida que lista extensa de jobs (simulando paginação da API) é verificada com sucesso."""
    dummy_jobs = [
        {"name": f"helper-job-{i}", "status": "completed", "conclusion": "success"}
        for i in range(1, 45)
    ]
    target_job = {"name": "test-integration", "status": "completed", "conclusion": "success"}
    jobs = dummy_jobs + [target_job]

    ok, errors = validate_jobs_payload(jobs, ["test-integration"])
    assert ok
    assert len(errors) == 0


def test_head_repository_nulo_rejeitado():
    """Valida que head_repository nulo/ausente é rejeitado fail-closed."""
    ok, errors = validate_head_repository(None, "Naldo2050/robo_sistema.binace.api")
    assert not ok
    assert any("nulo ou vazio" in e for e in errors)


def test_head_repository_fork_divergente_rejeitado():
    """Valida que head_repository de fork terceiro é rejeitado."""
    ok, errors = validate_head_repository("hacker/robo_sistema.binace.api", "Naldo2050/robo_sistema.binace.api")
    assert not ok
    assert any("diverge do repositório esperado" in e for e in errors)


def test_head_repository_valido():
    """Valida que repositório oficial é aceito (case-insensitive)."""
    ok, errors = validate_head_repository("Naldo2050/robo_sistema.binace.api", "naldo2050/robo_sistema.binace.api")
    assert ok
    assert len(errors) == 0


def test_sha_divergente_rejeitado():
    """Valida que SHA diferente do origin/main é rejeitado."""
    head_sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    expected_sha = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    ok, errors = validate_sha_alignment(head_sha, expected_sha)
    assert not ok
    assert any("diverge de origin/main" in e for e in errors)


def test_sha_alinhado_aprovado():
    """Valida que SHA alinhado com origin/main é aprovado."""
    sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    ok, errors = validate_sha_alignment(sha, sha)
    assert ok
    assert len(errors) == 0


def test_falha_sha_ausente_fail_closed():
    """Valida que ausência de SHA causa reprovação fail-closed."""
    ok, errors = validate_sha_alignment("", "abc")
    assert not ok


def test_workflow_dispatch_bloqueado_nesta_rodada():
    """Valida que workflow_dispatch manual é bloqueado de forma fail-closed."""
    code = run_gate(
        event_name="workflow_dispatch",
        expected_repo="Naldo2050/robo_sistema.binace.api",
        head_repo="Naldo2050/robo_sistema.binace.api",
        head_sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        run_id=1,
        required_jobs=["lint"],
        token="",
    )
    assert code == 1


def test_deploy_workflow_yaml_requires_push_official_repo_and_success():
    """
    R8-7 c: Valida semanticamente e de forma estrutural o campo if do job deploy
    em .github/workflows/deploy_oci.yml, sem usar substring bruta.
    Garante estruturalmente:
    1. conclusion == 'success'
    2. workflow_run.event == 'push'
    3. head_repository.full_name == github.repository
    """
    import yaml

    deploy_yml = PROJECT_ROOT / ".github" / "workflows" / "deploy_oci.yml"
    assert deploy_yml.exists(), f"Workflow file not found: {deploy_yml}"

    with open(deploy_yml, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    jobs = doc.get("jobs", {})
    assert "deploy" in jobs, "Job 'deploy' ausente no workflow deploy_oci.yml"

    deploy_job = jobs["deploy"]
    raw_if = deploy_job.get("if")
    assert raw_if is not None, "Cláusula 'if' obrigatória ausente no job 'deploy'"

    # Decodificação estrutural da expressão booleana
    cleaned_if = str(raw_if).strip()
    if cleaned_if.startswith("${{") and cleaned_if.endswith("}}"):
        cleaned_if = cleaned_if[3:-2].strip()

    clauses = [part.strip() for part in cleaned_if.split("&&")]
    structured_conditions = {}
    for clause in clauses:
        if "==" in clause:
            lhs, rhs = clause.split("==", 1)
            lhs_clean = lhs.strip()
            rhs_clean = rhs.strip().strip("'\"")
            structured_conditions[lhs_clean] = rhs_clean

    assert structured_conditions.get("github.event.workflow_run.conclusion") == "success", (
        f"Condição estrutural 'conclusion == success' não satisfeita. Encontrado: {structured_conditions}"
    )

    assert structured_conditions.get("github.event.workflow_run.event") == "push", (
        f"Condição estrutural 'workflow_run.event == push' não satisfeita. Encontrado: {structured_conditions}"
    )

    assert structured_conditions.get("github.event.workflow_run.head_repository.full_name") == "github.repository", (
        f"Condição estrutural de repositório oficial não satisfeita. Encontrado: {structured_conditions}"
    )
