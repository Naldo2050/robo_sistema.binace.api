import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

_CODE_DIRS = (
    "events",
    "common",
    "database",
    "monitoring",
    "trading",
    "orderbook_core",
)
_CONFIG_FILES = ("__init__.py", "settings.py", "env_policy.py")


def _copy_code(source_root: Path, target_root: Path) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    for dirname in _CODE_DIRS:
        shutil.copytree(
            source_root / dirname,
            target_root / dirname,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )

    (target_root / "config").mkdir()
    for filename in _CONFIG_FILES:
        shutil.copy2(
            source_root / "config" / filename,
            target_root / "config" / filename,
        )


def test_events_import_does_not_create_operational_artifacts(tmp_path):
    source_root = Path(__file__).resolve().parents[2]
    sandbox = tmp_path / "import_sandbox"
    _copy_code(source_root, sandbox)

    script = textwrap.dedent(
        """
        import sys
        import threading
        from pathlib import Path

        import events
        import events.event_memory
        import events.event_saver
        import events.fallback_reconciler

        root = Path.cwd().resolve()
        assert Path(events.__file__).resolve().is_relative_to(root), events.__file__
        assert Path(events.event_memory.__file__).resolve().is_relative_to(root)
        assert Path(events.event_saver.__file__).resolve().is_relative_to(root)
        assert Path(events.fallback_reconciler.__file__).resolve().is_relative_to(root)

        for name in ("dados", "fallback_events", "logs", "memory"):
            assert not (root / name).exists(), name

        for pattern in (
            "*.db",
            "*.db-wal",
            "*.db-shm",
            "*.db-journal",
            "*.sqlite",
            "*.sqlite3",
        ):
            found = list(root.rglob(pattern))
            assert not found, (pattern, found)

        assert not list(root.rglob("__pycache__"))

        extra_threads = [
            thread.name
            for thread in threading.enumerate()
            if thread is not threading.main_thread()
        ]
        assert not extra_threads, extra_threads

        print("IMPORT_ISOLATION_OK")
        """
    )

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(sandbox),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONIOENCODING": "utf-8",
            "LOAD_DOTENV": "0",
        }
    )
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONSTARTUP", None)

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(sandbox),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}\n"
        f"returncode: {result.returncode}"
    )
    assert "IMPORT_ISOLATION_OK" in result.stdout
