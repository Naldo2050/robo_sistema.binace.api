"""
ETAPA 7: Verificar __init__.py de cada pacote.
Garante que cada pacote tem API pública bem definida.
Executar: python scripts/migration/etapa7_check_contracts.py
"""

import ast
import os
from pathlib import Path


PACKAGES = [
    "events",
    "trading",
    "fetchers",
    "market_analysis",
    "data_processing",
    "monitoring",
    "common",
    "flow_analyzer",
    "market_orchestrator",
    "support_resistance",
    "orderbook_core",
    "orderbook_analyzer",
    "ai_runner",
    "risk_management",
    "ml",
    "data_pipeline",
]


def analyze_init(package_path: str) -> dict:
    """Analisa o __init__.py de um pacote."""
    init_path = os.path.join(package_path, "__init__.py")
    result = {
        "exists": False,
        "has_all": False,
        "all_items": [],
        "has_imports": False,
        "is_empty": True,
        "lines": 0,
    }

    if not os.path.exists(init_path):
        return result

    result["exists"] = True
    content = Path(init_path).read_text(encoding="utf-8", errors="ignore")
    lines = [l for l in content.splitlines() if l.strip() and not l.strip().startswith("#")]
    result["lines"] = len(lines)
    result["is_empty"] = len(lines) == 0

    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        result["has_all"] = True
                        if isinstance(node.value, ast.List):
                            result["all_items"] = [
                                elt.s if isinstance(elt, ast.Constant) else str(elt)
                                for elt in node.value.elts
                            ]
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                result["has_imports"] = True
    except SyntaxError:
        pass

    return result


def count_public_symbols(package_path: str) -> list[str]:
    """Conta símbolos públicos (classes, funções) nos módulos do pacote."""
    symbols = []
    for f in Path(package_path).glob("*.py"):
        if f.name == "__init__.py":
            continue
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="ignore"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                    symbols.append(f"{f.stem}.{node.name}")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not node.name.startswith("_"):
                        symbols.append(f"{f.stem}.{node.name}")
        except SyntaxError:
            pass
    return symbols


def main():
    print("=" * 60)
    print("ETAPA 7 - CONTRATOS DE PACOTES (__init__.py)")
    print("=" * 60)

    for pkg in PACKAGES:
        if not os.path.isdir(pkg):
            continue

        info = analyze_init(pkg)
        symbols = count_public_symbols(pkg)

        status = "✅" if info["has_all"] and info["has_imports"] else "⚠️ "
        if not info["exists"]:
            status = "❌"
        elif info["is_empty"]:
            status = "🟡"

        print(f"\n{status} {pkg}/")
        print(f"   __init__.py: {'existe' if info['exists'] else 'NÃO EXISTE'}")
        print(f"   __all__: {'sim' if info['has_all'] else 'não'}")
        print(f"   imports: {'sim' if info['has_imports'] else 'não'}")
        print(f"   linhas: {info['lines']}")
        print(f"   símbolos públicos: {len(symbols)}")

        if info["has_all"]:
            print(f"   __all__ = {info['all_items'][:5]}{'...' if len(info['all_items']) > 5 else ''}")

        if not info["has_all"] and symbols:
            print(f"   Sugestão de __all__:")
            classes = [s.split(".")[-1] for s in symbols if s[0].isupper() or "class" in s.lower()][:10]
            if classes:
                print(f"   __all__ = {classes}")


if __name__ == "__main__":
    main()
