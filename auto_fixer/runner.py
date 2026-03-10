"""
Runner principal - Executa todas as fases do auto-fixer.
"""

import argparse
import logging
import sys
import io
from pathlib import Path
from dataclasses import asdict

# Configurar encoding UTF-8 para Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("auto_fixer")


def run_phase1(project_root: str):
    """Fase 1: Scanner."""
    from auto_fixer.phase1_scanner.codebase_scanner import CodebaseScanner
    scanner = CodebaseScanner(project_root)
    result = scanner.scan()
    print(f"✅ Fase 1 completa: {result.total_files} arquivos encontrados")
    return result


def run_phase2():
    """Fase 2: Extrator AST."""
    from auto_fixer.phase2_extractor.ast_extractor import ASTExtractor
    extractor = ASTExtractor()
    result = extractor.extract_all()
    print(
        f"✅ Fase 2 completa: {result['total_classes']} classes, "
        f"{result['total_functions']} funções"
    )
    return result


def run_phase3():
    """Fase 3: Chunker."""
    from auto_fixer.phase3_chunker.chunk_engine import ChunkEngine
    engine = ChunkEngine()
    result = engine.process_all()
    print(f"✅ Fase 3 completa: {result.total_chunks} chunks criados")
    return result


def run_phase4():
    """Fase 4: Code Index."""
    from auto_fixer.phase4_index.code_index import CodeIndex
    index = CodeIndex()
    index.build()
    print("✅ Fase 4 completa: Índice construído")
    return index


def run_phase5():
    """Fase 5: RAG."""
    import json
    from auto_fixer.phase5_rag.vector_store import CodeVectorStore
    
    store = CodeVectorStore()
    
    # Carregar chunks
    chunks_dir = Path("auto_fixer/output/chunks")
    chunks = []
    if chunks_dir.exists():
        for f in chunks_dir.glob("*.json"):
            with open(f, encoding="utf-8") as fh:
                chunks.append(json.load(fh))
    
    if chunks:
        store.add_chunks(chunks)
        stats = store.get_stats()
        print(f"✅ Fase 5 completa: {stats.get('total_chunks', 0)} chunks indexados")
    else:
        print("⚠️ Fase 5: Nenhum chunk encontrado, pulando...")
    
    return store


def run_phase6(ai_client=None):
    """Fase 6: Analyzers."""
    import json
    from auto_fixer.phase6_analyzers.async_analyzer import AsyncAnalyzer
    from auto_fixer.phase6_analyzers.api_analyzer import APIAnalyzer
    from auto_fixer.phase6_analyzers.websocket_analyzer import WebSocketAnalyzer
    from auto_fixer.phase6_analyzers.import_analyzer import ImportAnalyzer
    
    # Carregar chunks
    chunks_dir = Path("auto_fixer/output/chunks")
    chunks = []
    for f in chunks_dir.glob("*.json"):
        with open(f, encoding="utf-8") as fh:
            chunks.append(json.load(fh))
    
    # Todos os analyzers
    analyzers = [
        AsyncAnalyzer(ai_client=ai_client),
        APIAnalyzer(ai_client=ai_client),
        WebSocketAnalyzer(ai_client=ai_client),
        ImportAnalyzer(ai_client=ai_client),
    ]
    
    all_issues = []
    for analyzer in analyzers:
        issues = analyzer.analyze_all(chunks)
        all_issues.extend(issues)
        print(f"  {analyzer.name}: {len(issues)} issues")
    
    print(f"✅ Fase 6 completa: {len(all_issues)} issues totais")
    return all_issues


def run_phase7(issues: list, ai_client=None):
    """Fase 7: Patch Generator."""
    from auto_fixer.phase7_patcher import (
        PatchGenerator, PatchValidator, PatchApplier
    )
    from dataclasses import asdict
    
    generator = PatchGenerator(ai_client=ai_client)
    validator = PatchValidator()
    applier = PatchApplier()
    
    patches = []
    for issue in issues:
        if isinstance(issue, dict):
            issue_dict = issue
        else:
            issue_dict = asdict(issue)
        
        if not issue_dict.get("auto_fixable"):
            continue
        
        patch = generator.generate_patch(issue_dict)
        if patch:
            valid, reason = validator.validate(patch)
            patch.validated = valid
            if valid:
                patches.append(patch)
                print(f"  ✅ Patch gerado e validado: {patch.patch_id}")
            else:
                print(f"  ❌ Patch inválido: {reason}")
    
    print(f"✅ Fase 7 completa: {len(patches)} patches válidos")
    return patches


def run_phase8():
    """Fase 8: Report."""
    from auto_fixer.phase8_reporter import ReportGenerator
    reporter = ReportGenerator()
    report = reporter.generate()
    print("✅ Fase 8 completa: Relatório gerado")
    print(f"\n{'='*60}")
    print(report[:2000])  # Preview
    print(f"{'='*60}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Auto-Fixer System")
    parser.add_argument(
        "--phase", type=int, default=0,
        help="Executar fase específica (1-8). 0 = todas"
    )
    parser.add_argument(
        "--project-root", type=str, default=".",
        help="Raiz do projeto"
    )
    parser.add_argument(
        "--apply-patches", action="store_true",
        help="Aplicar patches automaticamente"
    )
    
    args = parser.parse_args()
    
    print("🔧 Auto-Fixer System")
    print(f"   Projeto: {args.project_root}")
    print(f"   Fase: {'Todas' if args.phase == 0 else args.phase}")
    print()
    
    phases = {
        1: lambda: run_phase1(args.project_root),
        2: lambda: run_phase2(),
        3: lambda: run_phase3(),
        4: lambda: run_phase4(),
        5: lambda: run_phase5(),
        6: lambda: run_phase6(),
        7: lambda: run_phase7([]),
        8: lambda: run_phase8(),
    }
    
    issues_result = None
    
    if args.phase == 0:
        # Executar todas
        for phase_num in sorted(phases.keys()):
            print(f"\n{'='*40}")
            print(f"📌 FASE {phase_num}")
            print(f"{'='*40}")
            try:
                # Passar dados entre fases
                if phase_num == 1:
                    result = phases[phase_num]()
                elif phase_num == 6:
                    issues_result = phases[phase_num]()
                elif phase_num == 7 and issues_result:
                    run_phase7(issues_result)
                else:
                    result = phases[phase_num]()
                    
            except Exception as e:
                logger.error(f"Erro na Fase {phase_num}: {e}")
                print(f"❌ Fase {phase_num} falhou: {e}")
                if phase_num <= 2:
                    print("Fases 1-2 são obrigatórias. Abortando.")
                    sys.exit(1)
    else:
        if args.phase in phases:
            phases[args.phase]()
        else:
            print(f"Fase {args.phase} não existe (use 1-8)")


def run_full_pipeline(
    project_root: str = ".",
    apply_patches: bool = False,
    run_tests: bool = True,
    ai_enabled: bool = False,
):
    """
    Pipeline completo:
    Scan → Extract → Chunk → Index → Analyze → Patch → Test → Report
    
    Args:
        project_root: Raiz do projeto
        apply_patches: Se True, aplica patches automaticamente
        run_tests: Se True, executa testes após aplicar patches
        ai_enabled: Se True, habilita análise com IA
    """
    from auto_fixer.ai_client import AutoFixerAIClient
    from auto_fixer.feedback.fix_tracker import FixTracker
    from auto_fixer.test_runner import PatchTestRunner
    from auto_fixer.phase7_patcher import (
        PatchApplier, PatchValidator, RollbackManager
    )
    
    ai_client = None
    if ai_enabled:
        ai_client = AutoFixerAIClient()
        if ai_client.is_available():
            print("🤖 IA disponível")
        else:
            print("⚠️ IA não disponível, usando apenas análise estática")
            ai_client = None
    
    # Fases 1-4: Indexação
    print("\n📊 INDEXAÇÃO")
    run_phase1(project_root)
    run_phase2()
    run_phase3()
    run_phase4()
    
    # Fase 5: RAG (se IA ativa)
    if ai_client:
        print("\n🧠 RAG")
        run_phase5()
    
    # Fase 6: Análise
    print("\n🔍 ANÁLISE")
    issues = run_phase6(ai_client)
    
    # Separar por severidade
    critical = [i for i in issues if getattr(i, 'severity', None) 
                and i.severity.value == "CRITICAL"]
    fixable = [i for i in issues if getattr(i, 'auto_fixable', False)]
    
    print(f"\n📋 Resumo:")
    print(f"   Total issues: {len(issues)}")
    print(f"   CRITICAL: {len(critical)}")
    print(f"   Auto-fixáveis: {len(fixable)}")
    
    # Fase 7: Patches
    if fixable and apply_patches:
        print("\n🔧 PATCHES")
        patches = run_phase7(
            [asdict(i) if hasattr(i, '__dataclass_fields__') else i 
             for i in fixable],
            ai_client
        )
        
        # Aplicar e testar
        applier = PatchApplier(project_root)
        validator = PatchValidator(project_root)
        rollback_mgr = RollbackManager(project_root)
        test_runner = PatchTestRunner(project_root)
        tracker = FixTracker()
        
        for patch in patches:
            if not patch.validated:
                continue
            
            print(f"\n  Aplicando: {patch.patch_id}")
            
            # Aplicar
            ok, msg = applier.apply(patch)
            if not ok:
                print(f"  ❌ Falha: {msg}")
                continue
            
            # Testar
            if run_tests:
                tests_ok, test_output = test_runner.run_tests_for_file(
                    patch.file_path
                )
                
                if not tests_ok:
                    print(f"  ❌ Testes falharam, revertendo...")
                    rollback_mgr.rollback(patch.file_path)
                    tracker.record_fix(
                        patch_id=patch.patch_id,
                        issue_id=patch.issue_id,
                        file_path=patch.file_path,
                        severity=getattr(patch, 'severity', ''),
                        category=getattr(patch, 'category', ''),
                        applied=True,
                        tests_passed=False,
                        reverted=True,
                    )
                else:
                    print(f"  ✅ Patch aplicado e testado")
                    tracker.record_fix(
                        patch_id=patch.patch_id,
                        issue_id=patch.issue_id,
                        file_path=patch.file_path,
                        severity=getattr(patch, 'severity', ''),
                        category=getattr(patch, 'category', ''),
                        applied=True,
                        tests_passed=True,
                        reverted=False,
                    )
    
    # Fase 8: Relatório
    print("\n📄 RELATÓRIO")
    run_phase8()
    
    print("\n✅ Pipeline completo!")


if __name__ == "__main__":
    main()