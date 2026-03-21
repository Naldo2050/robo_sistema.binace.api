"""
ETAPA 3: Diagnosticar duplicações reais de módulos.
Compara conteúdo de módulos com mesmo nome.
Executar: python scripts/migration/etapa3_check_duplicates.py
"""

import os
import difflib
from pathlib import Path


# Pares suspeitos de duplicação
SUSPECT_PAIRS = [
    {
        "name": "AI Runner",
        "file_a": "ai_runner/ai_runner.py",
        "file_b": "market_orchestrator/ai/ai_runner.py",
        "description": "Dois AI Runners - qual é o real?",
    },
    {
        "name": "Signal Processor",
        "file_a": "market_orchestrator/flow/signal_processor.py",
        "file_b": "market_orchestrator/signals/signal_processor.py",
        "description": "Dois processadores de sinais no orchestrator",
    },
    {
        "name": "Risk Manager",
        "file_a": "risk_management/risk_manager.py",
        "file_b": "market_orchestrator/flow/risk_manager.py",
        "description": "Dois gerenciadores de risco",
    },
    {
        "name": "Macro Fetcher",
        "file_a": "fetchers/macro_fetcher.py",
        "file_b": "fetchers/macro_data_fetcher.py",
        "description": "Dois fetchers de macro com nomes similares",
    },
    {
        "name": "Logging Config",
        "file_a": "common/logging_config.py",
        "file_b": "flow_analyzer/logging_config.py",
        "description": "Duas configs de logging",
    },
    {
        "name": "Health Monitor",
        "file_a": "monitoring/health_monitor.py",
        "file_b": "auto_fixer/monitor/health_monitor.py",
        "description": "Dois health monitors",
    },
    {
        "name": "Payload Compressor",
        "file_a": "market_orchestrator/ai/payload_compressor.py",
        "file_b": "market_orchestrator/ai/payload_compressor_v3.py",
        "description": "Duas versões de compressor",
    },
    {
        "name": "Async Helpers",
        "file_a": "common/async_helpers.py",
        "file_b": "src/utils/async_helpers.py",
        "description": "Dois helpers async",
    },
]


def read_file_safe(filepath: str) -> str:
    """Lê arquivo com tratamento de erro."""
    try:
        return Path(filepath).read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return ""


def get_classes_and_functions(filepath: str) -> list[str]:
    """Extrai nomes de classes e funções."""
    import ast
    content = read_file_safe(filepath)
    if not content:
        return []

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    names = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(f"def {node.name}")
        elif isinstance(node, ast.ClassDef):
            names.append(f"class {node.name}")
    return names


def find_importers(module_name: str, root: str) -> list[str]:
    """Encontra quem importa este módulo."""
    importers = []
    for dirpath, _, filenames in os.walk(root):
        if any(skip in dirpath for skip in [".git", "__pycache__", ".venv"]):
            continue
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                content = Path(fpath).read_text(encoding="utf-8", errors="ignore")
                # Buscar por import do módulo
                module_parts = module_name.replace("/", ".").replace(".py", "")
                if module_parts in content:
                    rel = os.path.relpath(fpath, root)
                    if rel != module_name:
                        importers.append(rel)
            except Exception:
                pass
    return importers


def main():
    root = os.getcwd()

    print("=" * 60)
    print("ETAPA 3 - DIAGNÓSTICO DE DUPLICAÇÕES")
    print("=" * 60)

    for pair in SUSPECT_PAIRS:
        path_a = os.path.join(root, pair["file_a"])
        path_b = os.path.join(root, pair["file_b"])

        exists_a = os.path.exists(path_a)
        exists_b = os.path.exists(path_b)

        print(f"\n{'='*60}")
        print(f"📦 {pair['name']}: {pair['description']}")
        print(f"{'='*60}")
        print(f"   A: {pair['file_a']} {'✅ existe' if exists_a else '❌ não existe'}")
        print(f"   B: {pair['file_b']} {'✅ existe' if exists_b else '❌ não existe'}")

        if not exists_a or not exists_b:
            print("   ➡️  Apenas um existe - sem duplicação real")
            continue

        # Comparar conteúdo
        content_a = read_file_safe(path_a)
        content_b = read_file_safe(path_b)

        size_a = len(content_a)
        size_b = len(content_b)
        print(f"   Tamanho A: {size_a:,} bytes")
        print(f"   Tamanho B: {size_b:,} bytes")

        # Similaridade
        ratio = difflib.SequenceMatcher(None, content_a, content_b).ratio()
        print(f"   Similaridade: {ratio:.1%}")

        # Classes e funções
        names_a = get_classes_and_functions(path_a)
        names_b = get_classes_and_functions(path_b)

        common = set(names_a) & set(names_b)
        only_a = set(names_a) - set(names_b)
        only_b = set(names_b) - set(names_a)

        if common:
            print(f"   🔴 Em comum ({len(common)}):")
            for n in sorted(common):
                print(f"      └── {n}")
        if only_a:
            print(f"   🔵 Só em A ({len(only_a)}):")
            for n in sorted(only_a)[:5]:
                print(f"      └── {n}")
        if only_b:
            print(f"   🟢 Só em B ({len(only_b)}):")
            for n in sorted(only_b)[:5]:
                print(f"      └── {n}")

        # Quem importa
        importers_a = find_importers(pair["file_a"], root)
        importers_b = find_importers(pair["file_b"], root)

        print(f"   Importadores de A: {len(importers_a)}")
        for imp in importers_a[:3]:
            print(f"      └── {imp}")
        print(f"   Importadores de B: {len(importers_b)}")
        for imp in importers_b[:3]:
            print(f"      └── {imp}")

        # Recomendação
        if ratio > 0.8:
            print(f"   📋 RECOMENDAÇÃO: MERGE - código muito similar ({ratio:.0%})")
            if len(importers_a) >= len(importers_b):
                print(f"      Manter A ({pair['file_a']}), converter B em re-export")
            else:
                print(f"      Manter B ({pair['file_b']}), converter A em re-export")
        elif ratio > 0.3:
            print(f"   📋 RECOMENDAÇÃO: REVISAR MANUALMENTE - código parcialmente similar")
        else:
            print(f"   📋 RECOMENDAÇÃO: MANTER SEPARADOS - código diferente ({ratio:.0%})")


if __name__ == "__main__":
    main()
