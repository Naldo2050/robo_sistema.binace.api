"""
Report Generator - Fase 8.
Gera relatórios de auditoria.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Gera relatórios de auditoria de código."""
    
    def __init__(self, output_dir: str = "auto_fixer/output"):
        self.output_dir = Path(output_dir)
        self.reports_dir = self.output_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(self) -> str:
        """Gera relatório completo."""
        # Carregar dados
        scan = self._load("scan_result.json")
        structure = self._load("structure_map.json")
        
        # Carregar resultados de análise
        analysis_dir = self.output_dir / "analysis_results"
        analyses = {}
        if analysis_dir.exists():
            for f in analysis_dir.glob("*_results.json"):
                with open(f, encoding="utf-8") as fh:
                    analyses[f.stem] = json.load(fh)
        
        # Carregar patches
        patches_dir = self.output_dir / "patches"
        patches = []
        if patches_dir.exists():
            for f in patches_dir.glob("*.json"):
                with open(f, encoding="utf-8") as fh:
                    patches.append(json.load(fh))
        
        # Gerar relatório
        report = self._build_report(scan, structure, analyses, patches)
        
        # Salvar
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"audit_report_{timestamp}.md"
        report_path.write_text(report, encoding="utf-8")
        
        # Salvar também como latest
        latest_path = self.reports_dir / "audit_report_latest.md"
        latest_path.write_text(report, encoding="utf-8")
        
        logger.info(f"Relatório gerado: {report_path}")
        return report
    
    def _build_report(
        self, scan: dict, structure: dict, 
        analyses: dict, patches: list
    ) -> str:
        """Constrói o relatório em Markdown."""
        
        # Contar issues por severidade
        all_issues = []
        for analysis in analyses.values():
            all_issues.extend(analysis.get("issues", []))
        
        severity_counts = {}
        for issue in all_issues:
            sev = issue.get("severity", "UNKNOWN")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        report = f"""# 🔍 Relatório de Auditoria de Código

**Data:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Projeto:** Robo Binance API  

---

## 📊 Resumo Executivo

| Métrica | Valor |
|---------|-------|
| Arquivos escaneados | {scan.get('total_files', 0)} |
| Linhas de código | {scan.get('total_lines', 0):,} |
| Classes encontradas | {structure.get('total_classes', 0)} |
| Funções encontradas | {structure.get('total_functions', 0)} |
| Problemas encontrados | {len(all_issues)} |
| Patches gerados | {len(patches)} |

### Problemas por Severidade

| Severidade | Quantidade |
|------------|-----------|
| 🔴 CRITICAL | {severity_counts.get('CRITICAL', 0)} |
| 🟠 HIGH | {severity_counts.get('HIGH', 0)} |
| 🟡 MEDIUM | {severity_counts.get('MEDIUM', 0)} |
| 🟢 LOW | {severity_counts.get('LOW', 0)} |
| ℹ️ INFO | {severity_counts.get('INFO', 0)} |

---

## 🔴 Issues CRITICAL

"""
        # Issues CRITICAL
        critical = [i for i in all_issues if i.get("severity") == "CRITICAL"]
        if critical:
            for i, issue in enumerate(critical, 1):
                report += f"""### {i}. {issue['title']}
- **Arquivo:** `{issue['file_path']}`
- **Linha:** {issue['line_start']}
- **Descrição:** {issue['description']}
- **Auto-fixável:** {'✅' if issue.get('auto_fixable') else '❌'}
- **Confiança:** {issue.get('confidence', 0):.0%}

```
{issue.get('code_snippet', '')}
```

"""
        else:
            report += "Nenhum issue CRITICAL encontrado. ✅\n\n"

        # Issues HIGH
        report += "## 🟠 Issues HIGH\n\n"
        high = [i for i in all_issues if i.get("severity") == "HIGH"]
        if high:
            for i, issue in enumerate(high, 1):
                report += (
                    f"### {i}. {issue['title']}\n"
                    f"- **Arquivo:** `{issue['file_path']}` "
                    f"(linha {issue['line_start']})\n"
                    f"- {issue['description']}\n\n"
                )
        else:
            report += "Nenhum issue HIGH encontrado. ✅\n\n"
        
        # Issues MEDIUM
        report += "## 🟡 Issues MEDIUM\n\n"
        medium = [i for i in all_issues if i.get("severity") == "MEDIUM"]
        if medium:
            for i, issue in enumerate(medium[:10], 1):  # Limitar a 10
                report += (
                    f"- **{issue['title']}** - "
                    f"`{issue['file_path']}` (linha {issue['line_start']})\n"
                )
            if len(medium) > 10:
                report += f"\n*... e mais {len(medium) - 10} issues*\n"
        else:
            report += "Nenhum issue MEDIUM encontrado. ✅\n\n"
        
        # Issues LOW
        report += "## 🟢 Issues LOW\n\n"
        low = [i for i in all_issues if i.get("severity") == "LOW"]
        report += f"Total: {len(low)} issues LOW encontrados\n\n"
        
        # Arquivos grandes
        report += f"""---

## ⚠️ Arquivos Grandes (>1000 linhas)

Estes arquivos devem ser considerados para refatoração:

"""
        for f in scan.get("large_files", [])[:20]:
            report += f"- {f}\n"
        
        if len(scan.get("large_files", [])) > 20:
            report += f"\n*... e mais {len(scan.get('large_files', [])) - 20} arquivos*\n"
        
        # Patches
        report += f"""---

## 🔧 Patches Gerados

| Status | Quantidade |
|--------|-----------|
| Total | {len(patches)} |
| Aplicados | {sum(1 for p in patches if p.get('applied'))} |
| Pendentes | {sum(1 for p in patches if not p.get('applied'))} |

"""
        
        # Recomendações
        report += f"""
---

## 💡 Recomendações

1. **Corrigir issues CRITICAL imediatamente** — {severity_counts.get('CRITICAL', 0)} encontrados
2. **Revisar issues HIGH** — {severity_counts.get('HIGH', 0)} encontrados
3. **Refatorar arquivos grandes** — {len(scan.get('large_files', []))} arquivos com >1000 linhas
4. **Adicionar type hints** — Melhora detecção estática de bugs
5. **Aumentar cobertura de testes** — Especialmente nos módulos core

---

*Relatório gerado automaticamente pelo Auto-Fixer System*
"""
        return report
    
    def _load(self, filename: str) -> dict:
        path = self.output_dir / filename
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return {}
