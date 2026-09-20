# scripts/analytics/research_backup.py
# -*- coding: utf-8 -*-
"""
Backup local imediato dos datasets de pesquisa (independente de produção).

Inclui SOMENTE:
  dados/research/binance_positioning/
  dados/research/cftc/
  dados/research/cftc/r4_prospective/  (coberto pelo prefixo cftc/)

NUNCA inclui: .env, secrets/credenciais, database produtivo (trading_bot.db),
logs produtivos, working dataset além do snapshot copiado.

Formato: zip timestampado UTC + manifest.json + SHA256SUMS.
Retenção mínima: 35 dias (prune automático do que exceder).
Restore testável em diretório temporário (ver --restore-test).

Uso:
  python scripts/analytics/research_backup.py [--dest DIR] [--retention-days 35]
  python scripts/analytics/research_backup.py --restore-test <arquivo.zip>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEST = REPO_ROOT.parent / "robo_research_backup"
SOURCES = [
    REPO_ROOT / "dados" / "research" / "binance_positioning",
    REPO_ROOT / "dados" / "research" / "cftc",
]
FORBIDDEN_SUFFIXES = (".env", ".key", ".pem", ".secret")
FORBIDDEN_NAMES = {"trading_bot.db"}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_files():
    for src in SOURCES:
        if not src.exists():
            continue
        for p in sorted(src.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix in FORBIDDEN_SUFFIXES or p.name in FORBIDDEN_NAMES:
                continue
            if ".locks" in p.parts:  # lock efêmero, não é dado
                continue
            yield src, p


def create_backup(dest: str | Path, retention_days: int = 35) -> dict:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    stamp = _utcnow().strftime("%Y%m%dT%H%M%SZ")
    tmp_zip = dest / f".tmp-research-backup-{stamp}.zip"
    final_zip = dest / f"research-backup-{stamp}.zip"
    files, total_bytes = [], 0
    with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, p in _iter_files():
            arc = Path(src.parent.name) / src.name / p.relative_to(src)
            zf.write(p, arc.as_posix())
            files.append({"path": arc.as_posix(), "bytes": p.stat().st_size,
                          "sha256": _sha256(p)})
            total_bytes += p.stat().st_size
    manifest = {"created_at": _utcnow().isoformat(), "files": files,
                "total_bytes": total_bytes, "zip_sha256": None,
                "zip_sha256_note": ("hash do zip final está no sidecar .sha256; "
                                    "o manifest interno não pode conter o próprio hash"),
                "retention_days": retention_days,
                "excluded": [".env/secrets", "trading_bot.db",
                             "logs produtivos", ".locks"]}
    # manifest dentro + fora do zip; checksum APÓS o zip final (inclui manifest)
    with zipfile.ZipFile(tmp_zip, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps({**manifest, "zip_sha256": "pending"},
                                                ensure_ascii=False, indent=2))
    checksum = _sha256(tmp_zip)
    manifest["zip_sha256"] = checksum
    os.replace(tmp_zip, final_zip)  # atômico
    (dest / f"research-backup-{stamp}.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (dest / f"research-backup-{stamp}.sha256").write_text(
        f"{checksum}  {final_zip.name}\n", encoding="utf-8")
    pruned = prune(dest, retention_days)
    return {"zip": str(final_zip), "sha256": checksum, "files": len(files),
            "bytes": total_bytes, "pruned": pruned}


def prune(dest: Path, retention_days: int) -> list:
    """Remove backups além da retenção. Retorna removidos."""
    from datetime import timedelta

    cutoff = _utcnow() - timedelta(days=retention_days)
    removed = []
    for zf in dest.glob("research-backup-*.zip"):
        try:
            ts = datetime.strptime(zf.stem.replace("research-backup-", ""),
                                   "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if ts < cutoff:
            base = str(zf)[:-4]  # remove ".zip"
            for ext in (".zip", ".manifest.json", ".sha256"):
                q = Path(base + ext)
                if q.exists():
                    q.unlink()
                    removed.append(q.name)
    return removed


def restore_test(zip_path: str | Path) -> dict:
    """Restaura em diretório temporário e verifica. Nunca toca o live."""
    zip_path = Path(zip_path)
    with open(zip_path.parent / (zip_path.stem + ".sha256"), encoding="utf-8") as fh:
        expected = fh.read().split()[0]
    actual = _sha256(zip_path)
    if actual != expected:
        return {"ok": False, "reason": "checksum_mismatch"}
    tmp = Path(tempfile.mkdtemp(prefix="research-restore-"))
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)
        names = zf.namelist()
    manifest = json.loads((tmp / "manifest.json").read_text(encoding="utf-8"))
    checked = 0
    for f in manifest["files"]:
        p = tmp / f["path"]
        if not p.exists() or _sha256(p) != f["sha256"]:
            shutil.rmtree(tmp, ignore_errors=True)
            return {"ok": False, "reason": f"file_mismatch:{f['path']}"}
        checked += 1
    # contagens de auditoria: first/last/N por dataset principal
    summary = {}
    for pat, label in (("research/binance_positioning/normalized/*.jsonl", "b2_norm"),
                       ("research/cftc/r4_prospective/observations.jsonl", "r4_obs")):
        import glob as _glob

        hits = _glob.glob(str(tmp / pat))
        n = 0
        first = last = None
        for h in hits:
            for line in open(h, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                n += 1
                rec = json.loads(line)
                ts = rec.get("source_timestamp") or rec.get("report_as_of_date")
                if first is None or ts < first:
                    first = ts
                if last is None or ts > last:
                    last = ts
        summary[label] = {"rows": n, "first": first, "last": last}
    shutil.rmtree(tmp, ignore_errors=True)
    return {"ok": True, "files_checked": checked, "summary": summary}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.getenv("RESEARCH_BACKUP_DEST",
                                                str(DEFAULT_DEST)))
    ap.add_argument("--retention-days", type=int, default=35)
    ap.add_argument("--restore-test", default=None)
    args = ap.parse_args(argv)
    if args.restore_test:
        res = restore_test(args.restore_test)
        print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
        return 0 if res["ok"] else 1
    res = create_backup(args.dest, args.retention_days)
    print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
