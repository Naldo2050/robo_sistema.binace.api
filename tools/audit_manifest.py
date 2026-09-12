# tools/audit_manifest.py
# -*- coding: utf-8 -*-
"""Gera manifest.json da captura forense + SHA-256 dos artefatos.

Uso offline pós-captura:
  .venv\\Scripts\\python.exe tools/audit_manifest.py --dir dados/audit/live_XXXX
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    base = args.dir
    files = sorted(
        f for f in os.listdir(base)
        if os.path.isfile(os.path.join(base, f)) and f != "manifest.json"
    )
    hashes = {}
    for fn in files:
        try:
            hashes[fn] = sha256_file(os.path.join(base, fn))
        except Exception as e:
            hashes[fn] = f"ERROR:{e}"
    # Tenta enriquecer com contexto em memória se disponível via sidecar
    manifest = {
        "capture_run_id": os.path.basename(os.path.normpath(base)),
        "files": hashes,
    }
    sidecar = os.path.join(base, "manifest_base.json")
    if os.path.exists(sidecar):
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                manifest.update(json.load(f))
            manifest["files"] = hashes
        except Exception as e:
            manifest["sidecar_error"] = str(e)[:300]
    with open(os.path.join(base, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"MANIFEST OK: {os.path.join(base, 'manifest.json')} files={len(hashes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
