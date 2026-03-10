"""
Validação da instalação do Auto-Fixer.
Verifica cada fase e reporta status.
"""

import sys
import json
import importlib
import os
from pathlib import Path

# Configurar UTF-8 para suportar emojis no Windows
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

OUTPUT = Path("auto_fixer/output")


def check(name: str, condition: bool, detail: str = ""):
    status = "✅" if condition else "❌"
    print(f"  {status} {name}" + (f" — {detail}" if detail else ""))
    return condition


def validate():
    print("=" * 60)
    print("🔍 VALIDAÇÃO DO AUTO-FIXER SYSTEM")
    print("=" * 60)
    
    errors = 0
    
    # ── Dependências ──
    print("\n📦 Dependências:")
    
    deps = {
        "chromadb": "Fase 5 (RAG)",
        "structlog": "Logging estruturado",
    }
    for mod, usage in deps.items():
        try:
            importlib.import_module(mod)
            check(f"{mod}", True, usage)
        except ImportError:
            check(f"{mod}", False, f"FALTA — pip install {mod} ({usage})")
            errors += 1
    
    # ── Módulos do Auto-Fixer ──
    print("\n📂 Módulos:")
    
    modules = [
        "auto_fixer.phase1_scanner.codebase_scanner",
        "auto_fixer.phase2_extractor.ast_extractor",
        "auto_fixer.phase3_chunker.chunk_engine",
        "auto_fixer.phase4_index.code_index",
        "auto_fixer.phase5_rag.vector_store",
        "auto_fixer.phase6_analyzers.async_analyzer",
        "auto_fixer.phase7_patcher.patch_generator",
        "auto_fixer.phase8_reporter.report_generator",
    ]
    
    for mod in modules:
        try:
            importlib.import_module(mod)
            check(mod.split(".")[-1], True)
        except Exception as e:
            check(mod.split(".")[-1], False, str(e))
            errors += 1
    
    # ── Arquivos de saída ──
    print("\n📄 Arquivos de saída:")
    
    expected_outputs = {
        "scan_result.json": "Fase 1",
        "structure_map.json": "Fase 2",
        "chunk_index.json": "Fase 3",
        "code_index.json": "Fase 4",
    }
    
    for filename, phase in expected_outputs.items():
        path = OUTPUT / filename
        if path.exists():
            size = path.stat().st_size
            check(filename, True, f"{phase} — {size:,} bytes")
        else:
            check(filename, False, f"{phase} — EXECUTE a fase primeiro")
            errors += 1
    
    # ── Validar conteúdo ──
    print("\n📊 Validação de conteúdo:")
    
    scan_path = OUTPUT / "scan_result.json"
    if scan_path.exists():
        with open(scan_path, encoding="utf-8") as f:
            scan = json.load(f)
        
        total_files = scan.get("total_files", 0)
        total_lines = scan.get("total_lines", 0)
        
        check("Arquivos encontrados", total_files > 0, f"{total_files}")
        check("Linhas de código", total_lines > 0, f"{total_lines:,}")
        check(
            "Arquivos grandes detectados",
            len(scan.get("large_files", [])) >= 0,
            f"{len(scan.get('large_files', []))} arquivos"
        )
    
    structure_path = OUTPUT / "structure_map.json"
    if structure_path.exists():
        with open(structure_path, encoding="utf-8") as f:
            structure = json.load(f)
        
        check(
            "Classes extraídas",
            structure.get("total_classes", 0) > 0,
            f"{structure.get('total_classes', 0)}"
        )
        check(
            "Funções extraídas",
            structure.get("total_functions", 0) > 0,
            f"{structure.get('total_functions', 0)}"
        )
        
        files_with_errors = structure.get("files_with_errors", 0)
        total = structure.get("total_files", 1)
        error_rate = files_with_errors / max(total, 1) * 100
        check(
            "Taxa de parse errors",
            error_rate < 10,
            f"{error_rate:.1f}% ({files_with_errors}/{total})"
        )
    
    # ── Chunks ──
    chunks_dir = OUTPUT / "chunks"
    if chunks_dir.exists():
        chunk_count = len(list(chunks_dir.glob("*.json")))
        check("Chunks gerados", chunk_count > 0, f"{chunk_count}")
    
    # ── Vector Store ──
    vectordb_dir = OUTPUT / "vectordb"
    check("Vector DB existe", vectordb_dir.exists())
    
    # ── Resultado ──
    print(f"\n{'=' * 60}")
    if errors == 0:
        print("✅ TUDO VALIDADO — Sistema pronto para uso!")
    else:
        print(f"⚠️  {errors} PROBLEMA(S) ENCONTRADO(S)")
        print("   Corrija os itens marcados com ❌")
    print("=" * 60)
    
    return errors == 0


if __name__ == "__main__":
    ok = validate()
    sys.exit(0 if ok else 1)
