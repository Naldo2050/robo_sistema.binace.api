import re
from pathlib import Path


def find_unprotected_secrets_in_workflow(workflow_path: Path):
    assert workflow_path.exists(), f"Workflow {workflow_path} deve existir"
    text = workflow_path.read_text(encoding="utf-8")

    guarded_pattern = re.compile(
        r'^\$\{\{\s*github\.event_name\s*==\s*["\']push["\']\s*&&\s*secrets\.[A-Za-z0-9_]+\s*\|\|\s*["\']["\']\s*\}\}$'
    )

    unprotected = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if "secrets." in line:
            matches = re.findall(r'(\$\{\{[^}]+\}\})', line)
            if not matches:
                unprotected.append((line_no, line.strip(), "sem expressao ${{ }}"))
            for m in matches:
                if "secrets." in m:
                    if not guarded_pattern.match(m.strip()):
                        unprotected.append((line_no, m.strip(), "nao condicionado a push"))
    return unprotected


def test_tests_workflow_all_secrets_guarded():
    wf = Path(".github/workflows/tests.yml")
    unprotected = find_unprotected_secrets_in_workflow(wf)
    assert not unprotected, (
        f"Foram encontradas referências a secrets desprotegidas em {wf}:\n"
        + "\n".join(f"  Linha {line_num}: {expr} ({reason})" for line_num, expr, reason in unprotected)
    )


def test_ci_workflow_all_secrets_guarded():
    wf = Path(".github/workflows/ci.yml")
    unprotected = find_unprotected_secrets_in_workflow(wf)
    assert not unprotected, (
        f"Foram encontradas referências a secrets desprotegidas em {wf}:\n"
        + "\n".join(f"  Linha {line_num}: {expr} ({reason})" for line_num, expr, reason in unprotected)
    )


def test_guard_rejects_unprotected_even_if_push_keyword_is_present():
    dummy_content = """
    name: Dummy Tests
    on:
      push:
        branches: [main]
      pull_request:
    jobs:
      test:
        runs-on: ubuntu-latest
        steps:
          - name: Step with push in name
            run: echo "running on push"
            env:
              FUTURE_SECRET: ${{ secrets.FUTURE_NEW_SECRET }}
    """
    tmp_dummy = Path("temp_dummy_workflow.yml")
    tmp_dummy.write_text(dummy_content, encoding="utf-8")
    try:
        unprotected = find_unprotected_secrets_in_workflow(tmp_dummy)
        assert len(unprotected) == 1
        assert "FUTURE_NEW_SECRET" in unprotected[0][1]
        assert unprotected[0][2] == "nao condicionado a push"
    finally:
        tmp_dummy.unlink(missing_ok=True)
