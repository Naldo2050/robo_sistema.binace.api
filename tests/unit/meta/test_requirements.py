from pathlib import Path


def test_dev_declares_asyncio() -> None:
    lines = {
        line.strip()
        for line in Path("requirements-dev.txt").read_text(encoding="utf-8").splitlines()
    }
    assert "pytest-asyncio==1.4.1.post1" in lines


def test_runtime_declares_psutil() -> None:
    lines = {
        line.strip()
        for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines()
    }
    assert "psutil==7.2.2" in lines
