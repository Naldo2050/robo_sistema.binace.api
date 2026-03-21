"""
ETAPA 1: Gerar mapa de dependências para diagnóstico.
Identifica imports circulares, duplicações e módulos órfãos.
Executar: python scripts/migration/etapa1_dependency_map.py
"""

import ast
import os
import json
from collections import defaultdict
from pathlib import Path


def find_python_files(root: str, exclude_dirs: set) -> list[str]:
    """Encontra todos os .py do projeto."""
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for f in filenames:
            if f.endswith(".py"):
                files.append(os.path.join(dirpath, f))
    return files


def extract_imports(filepath: str) -> list[dict]:
    """Extrai imports de um arquivo Python."""
    imports = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            tree = ast.parse(f.read(), filename=filepath)
    except SyntaxError:
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    "type": "import",
                    "module": alias.name,
                    "name": alias.asname or alias.name,
                })
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append({
                    "type": "from",
                    "module": module,
                    "name": alias.name,
                })
    return imports


def detect_duplicates(files: list[str]) -> dict:
    """Detecta módulos com mesmo nome em locais diferentes."""
    name_map = defaultdict(list)
    for f in files:
        name = Path(f).stem
        name_map[name].append(f)
    return {k: v for k, v in name_map.items() if len(v) > 1}


def detect_proxy_files(root: str) -> list[dict]:
    """Detecta proxies na raiz (arquivos < 10 linhas com re-export)."""
    proxies = []
    for f in Path(root).glob("*.py"):
        if f.name.startswith("__"):
            continue
        try:
            content = f.read_text(encoding="utf-8")
            lines = [l.strip() for l in content.splitlines() if l.strip() and not l.strip().startswith("#")]
            if len(lines) <= 5:
                has_reexport = any("from " in l and "import" in l for l in lines)
                if has_reexport:
                    proxies.append({
                        "file": str(f),
                        "lines": len(lines),
                        "content": lines,
                    })
        except Exception:
            pass
    return proxies


def find_proxy_importers(proxy_name: str, files: list[str]) -> list[str]:
    """Encontra quem importa de um proxy."""
    importers = []
    for f in files:
        try:
            content = Path(f).read_text(encoding="utf-8", errors="ignore")
            if f"from {proxy_name} import" in content or f"import {proxy_name}" in content:
                if Path(f).stem != proxy_name:
                    importers.append(f)
        except Exception:
            pass
    return importers


def detect_circular_candidates(files: list[str], root: str) -> list[tuple]:
    """Detecta possíveis imports circulares."""
    module_imports = {}
    for f in files:
        rel = os.path.relpath(f, root).replace(os.sep, ".").replace(".py", "")
        if rel.endswith(".__init__"):
            rel = rel[:-9]
        imports = extract_imports(f)
        module_imports[rel] = {i["module"] for i in imports if i["module"]}

    circulars = []
    for mod_a, imports_a in module_imports.items():
        for mod_b, imports_b in module_imports.items():
            if mod_a != mod_b:
                a_imports_b = any(mod_b in imp or imp.startswith(mod_b + ".") for imp in imports_a)
                b_imports_a = any(mod_a in imp or imp.startswith(mod_a + ".") for imp in imports_b)
                if a_imports_b and b_imports_a:
                    pair = tuple(sorted([mod_a, mod_b]))
                    if pair not in circulars:
                        circulars.append(pair)
    return circulars


def find_exception_files(files: list[str]) -> list[str]:
    """Encontra todos os arquivos de exceções."""
    return [f for f in files if "exception" in Path(f).stem.lower() or "error" in Path(f).stem.lower()]


def main():
    root = os.getcwd()
    exclude = {
        ".git", "__pycache__", ".venv", "venv", "node_modules",
        ".mypy_cache", ".pytest_cache", "backups", "infrastructure",
    }

    print("=" * 60)
    print("ETAPA 1 - MAPA DE DEPENDÊNCIAS")
    print("=" * 60)

    # 1. Encontrar arquivos
    files = find_python_files(root, exclude)
    print(f"\n📁 Total de arquivos Python: {len(files)}")

    # 2. Duplicações de nome
    print("\n" + "=" * 60)
    print("🔍 MÓDULOS DUPLICADOS (mesmo nome, locais diferentes)")
    print("=" * 60)
    duplicates = detect_duplicates(files)
    for name, locations in sorted(duplicates.items()):
        print(f"\n  📦 {name}:")
        for loc in locations:
            print(f"     └── {os.path.relpath(loc, root)}")

    # 3. Proxies
    print("\n" + "=" * 60)
    print("🔀 PROXIES NA RAIZ")
    print("=" * 60)
    proxies = detect_proxy_files(root)
    for p in proxies:
        name = Path(p["file"]).stem
        importers = find_proxy_importers(name, files)
        print(f"\n  📄 {Path(p['file']).name} ({p['lines']} linhas)")
        print(f"     Conteúdo: {p['content'][0]}")
        print(f"     Importadores: {len(importers)}")
        for imp in importers[:5]:
            print(f"       └── {os.path.relpath(imp, root)}")
        if len(importers) > 5:
            print(f"       └── ... e mais {len(importers)-5}")

    # 4. Circulares
    print("\n" + "=" * 60)
    print("🔄 POSSÍVEIS IMPORTS CIRCULARES")
    print("=" * 60)
    circulars = detect_circular_candidates(files, root)
    if circulars:
        for a, b in circulars[:20]:
            print(f"  ⚠️  {a} ←→ {b}")
    else:
        print("  ✅ Nenhum detectado (análise estática)")

    # 5. Exceptions
    print("\n" + "=" * 60)
    print("⚠️  ARQUIVOS DE EXCEÇÕES (candidatos a unificação)")
    print("=" * 60)
    exc_files = find_exception_files(files)
    for f in exc_files:
        print(f"  └── {os.path.relpath(f, root)}")

    # 6. Salvar relatório
    report = {
        "total_files": len(files),
        "duplicates": {k: [os.path.relpath(v2, root) for v2 in v] for k, v in duplicates.items()},
        "proxies": [{
            "file": Path(p["file"]).name,
            "importers_count": len(find_proxy_importers(Path(p["file"]).stem, files)),
        } for p in proxies],
        "circular_candidates": [list(c) for c in circulars[:20]],
        "exception_files": [os.path.relpath(f, root) for f in exc_files],
    }

    os.makedirs("scripts/migration/reports", exist_ok=True)
    with open("scripts/migration/reports/etapa1_dependency_map.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print(f"📊 Relatório salvo em: scripts/migration/reports/etapa1_dependency_map.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
