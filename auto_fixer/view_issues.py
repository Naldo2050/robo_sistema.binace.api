"""
Visualizador de Issues do Auto-Fixer.
Uso:
  python -m auto_fixer.view_issues                    # Todos
  python -m auto_fixer.view_issues --severity CRITICAL # Só críticos
  python -m auto_fixer.view_issues --category async    # Só async
  python -m auto_fixer.view_issues --file orderbook    # Por arquivo
  python -m auto_fixer.view_issues --fixable           # Só auto-fixáveis
  python -m auto_fixer.view_issues --summary           # Resumo
"""

import json
import argparse
import sys
from pathlib import Path
from collections import Counter

# Configurar encoding UTF-8 para Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def load_all_issues() -> list[dict]:
    """Carrega todos os issues de todos os analyzers."""
    results_dir = Path("auto_fixer/output/analysis_results")
    all_issues = []
    
    if not results_dir.exists():
        print("❌ Nenhum resultado encontrado. Execute a Fase 6 primeiro.")
        return []
    
    for f in results_dir.glob("*_results.json"):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
            for issue in data.get("issues", []):
                issue["_analyzer"] = data.get("analyzer", "unknown")
                all_issues.append(issue)
    
    return all_issues


def print_issue(issue: dict, index: int):
    """Imprime um issue formatado."""
    severity_icons = {
        "CRITICAL": "🔴",
        "HIGH": "🟠",
        "MEDIUM": "🟡",
        "LOW": "🟢",
        "INFO": "ℹ️",
    }
    icon = severity_icons.get(issue["severity"], "❓")
    
    print(f"\n{icon} #{index} [{issue['severity']}] {issue['title']}")
    print(f"   Arquivo:    {issue['file_path']}")
    print(f"   Linha:      {issue['line_start']}")
    print(f"   Categoria:  {issue['category']}")
    print(f"   Confiança:  {issue.get('confidence', 0):.0%}")
    print(f"   Auto-fix:   {'✅ Sim' if issue.get('auto_fixable') else '❌ Não'}")
    
    snippet = issue.get("code_snippet", "")
    if snippet and snippet != "N/A":
        # Limitar snippet
        if len(snippet) > 120:
            snippet = snippet[:120] + "..."
        print(f"   Código:     {snippet}")
    
    if issue.get("suggested_fix"):
        print(f"   Sugestão:   {issue['suggested_fix']}")


def print_summary(issues: list[dict]):
    """Imprime resumo dos issues."""
    print("\n" + "=" * 60)
    print("📊 RESUMO DE ISSUES")
    print("=" * 60)
    
    # Por severidade
    severity_count = Counter(i["severity"] for i in issues)
    print("\n  Por Severidade:")
    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
        count = severity_count.get(sev, 0)
        bar = "█" * min(count, 50)
        print(f"    {sev:10s} {count:4d}  {bar}")
    
    # Por categoria
    cat_count = Counter(i["category"] for i in issues)
    print("\n  Por Categoria:")
    for cat, count in cat_count.most_common():
        bar = "█" * min(count, 50)
        print(f"    {cat:15s} {count:4d}  {bar}")
    
    # Auto-fixáveis
    fixable = sum(1 for i in issues if i.get("auto_fixable"))
    print(f"\n  Auto-fixáveis: {fixable}/{len(issues)}")
    
    # Top arquivos com mais problemas
    file_count = Counter(i["file_path"] for i in issues)
    print("\n  Top 10 Arquivos com Mais Issues:")
    for filepath, count in file_count.most_common(10):
        # Contar por severidade deste arquivo
        file_issues = [i for i in issues if i["file_path"] == filepath]
        crits = sum(1 for i in file_issues if i["severity"] == "CRITICAL")
        highs = sum(1 for i in file_issues if i["severity"] == "HIGH")
        
        flags = ""
        if crits > 0:
            flags += f" 🔴{crits}"
        if highs > 0:
            flags += f" 🟠{highs}"
        
        print(f"    {count:4d}  {filepath}{flags}")
    
    print(f"\n  Total: {len(issues)} issues em "
          f"{len(file_count)} arquivos")


def main():
    parser = argparse.ArgumentParser(description="Visualizador de Issues")
    parser.add_argument(
        "--severity", "-s", 
        choices=["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"],
        help="Filtrar por severidade"
    )
    parser.add_argument(
        "--category", "-c",
        help="Filtrar por categoria (async, api, websocket, import)"
    )
    parser.add_argument(
        "--file", "-f",
        help="Filtrar por nome de arquivo (busca parcial)"
    )
    parser.add_argument(
        "--fixable", action="store_true",
        help="Mostrar apenas auto-fixáveis"
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Mostrar apenas resumo"
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=50,
        help="Limitar número de issues mostrados"
    )
    
    args = parser.parse_args()
    
    issues = load_all_issues()
    if not issues:
        return
    
    # Aplicar filtros
    filtered = issues
    
    if args.severity:
        filtered = [i for i in filtered if i["severity"] == args.severity]
    
    if args.category:
        filtered = [
            i for i in filtered 
            if args.category.lower() in i["category"].lower()
        ]
    
    if args.file:
        filtered = [
            i for i in filtered 
            if args.file.lower() in i["file_path"].lower()
        ]
    
    if args.fixable:
        filtered = [i for i in filtered if i.get("auto_fixable")]
    
    # Ordenar: CRITICAL primeiro, depois HIGH, etc.
    severity_order = {
        "CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4
    }
    filtered.sort(key=lambda i: severity_order.get(i["severity"], 5))
    
    # Mostrar
    if args.summary:
        print_summary(filtered)
    else:
        print(f"\n📋 {len(filtered)} issues encontrados", end="")
        if args.severity:
            print(f" (severidade: {args.severity})", end="")
        if args.category:
            print(f" (categoria: {args.category})", end="")
        if args.file:
            print(f" (arquivo: {args.file})", end="")
        print()
        
        for i, issue in enumerate(filtered[:args.limit], 1):
            print_issue(issue, i)
        
        if len(filtered) > args.limit:
            print(f"\n... e mais {len(filtered) - args.limit} issues")
            print(f"Use --limit {len(filtered)} para ver todos")
        
        print()
        print_summary(filtered)


if __name__ == "__main__":
    main()
