# tests/unit/test_backup_to_oci_excludes.py
"""
Regressão de segurança (SEC-3): backup OCI não deve vazar segredos/dados.

Bug: scripts/backup_to_oci.py empacotava DATA_DIRS inteiros ("logs",
"features") sem excluir *.db / *.jsonl / .env — logs contêm payloads da IA
e fragmentos de chaves; um bucket vazado exporia histórico + segredos.

Contrato exigido:
- create_archive() deve EXCLUIR arquivos *.db, *.sqlite3, *.jsonl e .env
  em qualquer nível do pacote.
- Arquivos normais (.log, .txt, .parquet) continuam incluídos.
"""

import importlib.util
import sys
import tarfile
from pathlib import Path


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "backup_to_oci", str(Path("scripts/backup_to_oci.py"))
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["backup_to_oci"] = module
    spec.loader.exec_module(module)
    return module


def test_archive_excludes_secrets(tmp_path, monkeypatch):
    mod = _load_module()

    payload = tmp_path / "logs"
    payload.mkdir()
    (payload / "run.log").write_text("log normal", encoding="utf-8")
    (payload / "trading_bot.db").write_text("FAKE-DB", encoding="utf-8")
    (payload / "eventos_fluxo.jsonl").write_text('{"a": 1}', encoding="utf-8")
    (payload / ".env").write_text("GROQ_API_KEY=SECRET", encoding="utf-8")
    nested = payload / "nested"
    nested.mkdir()
    (nested / "cache.db").write_text("FAKE-DB", encoding="utf-8")
    (nested / "keep.txt").write_text("ok", encoding="utf-8")

    monkeypatch.setattr(mod, "DATA_DIRS", [str(payload)])
    monkeypatch.chdir(tmp_path)
    out = str(tmp_path / "out.tar.gz")
    mod.create_archive(out)

    with tarfile.open(out, "r:gz") as tar:
        names = tar.getnames()

    assert not any(n.endswith(".db") for n in names), names
    assert not any(n.endswith(".jsonl") for n in names), names
    assert not any(n.rsplit("/", 1)[-1] == ".env" for n in names), names
    assert any(n.endswith("run.log") for n in names), names
    assert any(n.endswith("keep.txt") for n in names), names
