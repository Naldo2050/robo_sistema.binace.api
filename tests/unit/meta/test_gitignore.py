import subprocess
from pathlib import Path


def test_docker_context_excludes_local_artifacts() -> None:
    lines = {
        line.strip()
        for line in Path(".dockerignore").read_text(encoding="utf-8").splitlines()
    }
    assert {"setup_arena.sh", "scratch/"} <= lines


def test_setup_arena_script_is_ignored() -> None:
    script = Path("setup_arena.sh")
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    try:
        result = subprocess.run(
            ["git", "check-ignore", "-q", "setup_arena.sh"],
            capture_output=True,
        )
        assert result.returncode == 0
    finally:
        script.unlink(missing_ok=True)


def test_gitignore_has_no_duplicate_scratch():
    lines = [
        line.strip()
        for line in Path(".gitignore").read_text(encoding="utf-8").splitlines()
    ]
    scratch_occurrences = [i for i, line in enumerate(lines, 1) if line == "scratch/"]
    assert len(scratch_occurrences) == 1, f"scratch/ duplicado no .gitignore nas linhas {scratch_occurrences}"
