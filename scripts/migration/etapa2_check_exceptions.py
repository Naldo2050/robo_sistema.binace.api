"""
ETAPA 2: Diagnosticar exceções duplicadas.
Verifica se é seguro unificar em common/exceptions.py
Executar: python scripts/migration/etapa2_check_exceptions.py
"""

import ast
import os
from pathlib import Path
from collections import defaultdict


EXCEPTION_FILES = [
    "common/exceptions.py",
    "ai_runner/exceptions.py",
    "risk_management/exceptions.py",
    "orderbook_core/exceptions.py",
    "flow_analyzer/errors.py",
    "data_pipeline/config.py",  # pode ter exceções inline
]


def extract_exception_classes(filepath: str) -> list[dict]:
    """Extrai classes de exceção de um arquivo."""
    exceptions = []
    if not os.path.exists(filepath):
        return exceptions

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except SyntaxError:
        return exceptions

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(f"{ast.dump(base)}")
            if any("Error" in b or "Exception" in b for b in bases) or "Error" in node.name or "Exception" in node.name:
                exceptions.append({
                    "name": node.name,
                    "bases": bases,
                    "file": filepath,
                    "line": node.lineno,
                })
    return exceptions


def find_exception_usages(exc_name: str, root: str) -> list[str]:
    """Encontra onde uma exceção é usada (raise ou except)."""
    usages = []
    for dirpath, _, filenames in os.walk(root):
        if any(skip in dirpath for skip in [".git", "__pycache__", ".venv", "venv"]):
            continue
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                content = Path(fpath).read_text(encoding="utf-8", errors="ignore")
                if exc_name in content:
                    usages.append(os.path.relpath(fpath, root))
            except Exception:
                pass
    return usages


def main():
    root = os.getcwd()

    print("=" * 60)
    print("ETAPA 2 - DIAGNÓSTICO DE EXCEÇÕES")
    print("=" * 60)

    all_exceptions = []
    for filepath in EXCEPTION_FILES:
        full_path = os.path.join(root, filepath)
        exceptions = extract_exception_classes(full_path)
        all_exceptions.extend(exceptions)
        if exceptions:
            print(f"\n📄 {filepath}:")
            for exc in exceptions:
                print(f"   class {exc['name']}({', '.join(exc['bases'])})")

    # Detectar duplicações
    print("\n" + "=" * 60)
    print("🔍 EXCEÇÕES COM MESMO NOME EM LOCAIS DIFERENTES")
    print("=" * 60)

    name_map = defaultdict(list)
    for exc in all_exceptions:
        name_map[exc["name"]].append(exc)

    duplicates = {k: v for k, v in name_map.items() if len(v) > 1}

    if duplicates:
        for name, locations in duplicates.items():
            print(f"\n  ⚠️  {name}:")
            for loc in locations:
                print(f"     └── {loc['file']}:{loc['line']}")
    else:
        print("  ✅ Nenhuma duplicação de nome encontrada")

    # Verificar uso de cada exceção
    print("\n" + "=" * 60)
    print("📊 USO DE CADA EXCEÇÃO")
    print("=" * 60)

    for exc in all_exceptions:
        usages = find_exception_usages(exc["name"], root)
        # Filtrar o próprio arquivo de definição
        usages = [u for u in usages if u != exc["file"]]
        status = "✅" if len(usages) > 0 else "⚠️  SEM USO"
        print(f"\n  {status} {exc['name']} (definido em {exc['file']})")
        print(f"     Usado em {len(usages)} arquivo(s)")
        for u in usages[:3]:
            print(f"       └── {u}")
        if len(usages) > 3:
            print(f"       └── ... +{len(usages)-3} mais")

    # Recomendação
    print("\n" + "=" * 60)
    print("📋 RECOMENDAÇÃO")
    print("=" * 60)

    safe_to_merge = []
    needs_attention = []

    for filepath in EXCEPTION_FILES:
        if filepath == "common/exceptions.py":
            continue
        full_path = os.path.join(root, filepath)
        exceptions = extract_exception_classes(full_path)
        for exc in exceptions:
            usages = find_exception_usages(exc["name"], root)
            usages = [u for u in usages if u != filepath]
            if len(usages) <= 3:
                safe_to_merge.append(exc)
            else:
                needs_attention.append(exc)

    print(f"\n  🟢 Seguro para mover para common/exceptions.py: {len(safe_to_merge)}")
    for exc in safe_to_merge:
        print(f"     └── {exc['name']} (de {exc['file']})")

    print(f"\n  🟡 Requer atenção (muitos importadores): {len(needs_attention)}")
    for exc in needs_attention:
        print(f"     └── {exc['name']} (de {exc['file']})")


if __name__ == "__main__":
    main()
