from pathlib import Path
import configparser


def test_dev_requirements_declares_pytest_asyncio() -> None:
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


def test_mypy_files_are_comma_separated_and_exist() -> None:
    config = configparser.ConfigParser()
    config.read("mypy.ini")
    files = [
        item.strip() for item in config["mypy"]["files"].split(",") if item.strip()
    ]
    assert files == ["orderbook_core", "orderbook_analyzer.py"]
    assert all(Path(item).exists() for item in files)
