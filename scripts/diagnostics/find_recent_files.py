import os
import datetime
from pathlib import Path

repo = Path(r"c:\Users\Micro\Documents\Visual Studio\robo_sistema.binace.api")
cutoff = datetime.datetime(2026, 9, 4, 0, 0, 0)
ignored = {".git", ".venv", ".mypy_cache", ".pytest_cache", "__pycache__", ".ruff_cache"}

recent_files = []
for root, dirs, files in os.walk(repo):
    dirs[:] = [d for d in dirs if d not in ignored]
    for f in files:
        p = Path(root) / f
        try:
            mtime = datetime.datetime.fromtimestamp(p.stat().st_mtime)
            if mtime >= cutoff:
                recent_files.append((mtime, p.stat().st_size, str(p.relative_to(repo))))
        except Exception:
            pass

recent_files.sort(key=lambda x: x[0], reverse=True)
print(f"Total files modified since 2026-09-04: {len(recent_files)}")
for mtime, size, path in recent_files[:60]:
    print(f"{mtime.strftime('%Y-%m-%d %H:%M:%S')} | {size:10d} B | {path}")
