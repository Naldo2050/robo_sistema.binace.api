"""
API HTTP do Robo Binance.
Roda separado do bot de trading.
Porta 8001 (Prometheus usa 8000).
"""

from fastapi import FastAPI
import json
from pathlib import Path
from datetime import datetime

from typing import Optional

app = FastAPI(
    title="Robo Binance API - Monitor",
    version="1.0.0",
    description="API de monitoramento e auto-fixer"
)


# ═══════════════════════════════════════
# Health / Status
# ═══════════════════════════════════════

@app.get("/health")
def health():
    """Verifica se a API está rodando."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/status")
def status():
    """Status geral do sistema."""
    # Verificar se o bot está rodando
    import psutil
    bot_running = False
    for proc in psutil.process_iter(['name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline') or []
            if any('main.py' in str(c) for c in cmdline):
                bot_running = True
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    return {
        "bot_running": bot_running,
        "timestamp": datetime.now().isoformat(),
    }


# ═══════════════════════════════════════
# Auto-Fixer Endpoints
# ═══════════════════════════════════════

@app.get("/auto-fixer/summary")
def autofixer_summary():
    """Resumo dos issues encontrados."""
    results_dir = Path("auto_fixer/output/analysis_results")
    
    if not results_dir.exists():
        return {"error": "Nenhuma análise encontrada. Execute fase 6."}
    
    all_issues = []
    for f in results_dir.glob("*_results.json"):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
            all_issues.extend(data.get("issues", []))
    
    severity_count = {}
    category_count = {}
    for issue in all_issues:
        sev = issue.get("severity", "UNKNOWN")
        cat = issue.get("category", "unknown")
        severity_count[sev] = severity_count.get(sev, 0) + 1
        category_count[cat] = category_count.get(cat, 0) + 1
    
    return {
        "total_issues": len(all_issues),
        "by_severity": severity_count,
        "by_category": category_count,
        "auto_fixable": sum(
            1 for i in all_issues if i.get("auto_fixable")
        ),
    }


@app.get("/auto-fixer/issues")
def autofixer_issues(
    severity: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 50
):
    """Lista issues com filtros."""
    results_dir = Path("auto_fixer/output/analysis_results")
    
    if not results_dir.exists():
        return {"issues": [], "total": 0}
    
    all_issues = []
    for f in results_dir.glob("*_results.json"):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
            all_issues.extend(data.get("issues", []))
    
    # Filtrar
    if severity:
        all_issues = [
            i for i in all_issues 
            if i["severity"] == severity.upper()
        ]
    if category:
        all_issues = [
            i for i in all_issues 
            if i["category"] == category.lower()
        ]
    
    # Ordenar por severidade
    sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    all_issues.sort(key=lambda i: sev_order.get(i["severity"], 4))
    
    return {
        "issues": all_issues[:limit],
        "total": len(all_issues),
        "showing": min(limit, len(all_issues)),
    }


@app.get("/auto-fixer/report")
def autofixer_report():
    """Retorna último relatório gerado."""
    report_path = Path(
        "auto_fixer/output/reports/audit_report_latest.md"
    )
    
    if not report_path.exists():
        return {"error": "Nenhum relatório encontrado. Execute fase 8."}
    
    content = report_path.read_text(encoding="utf-8")
    return {"report": content}


@app.post("/auto-fixer/run")
def autofixer_run(phase: int = 0):
    """Dispara execução do auto-fixer."""
    import subprocess
    
    cmd = ["python", "-m", "auto_fixer.runner", "--phase", str(phase)]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout[-2000:],
            "stderr": result.stderr[-1000:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Timeout (5 min)"}


@app.get("/auto-fixer/scan-info")
def scan_info():
    """Informações do último scan."""
    scan_path = Path("auto_fixer/output/scan_result.json")
    if not scan_path.exists():
        return {"error": "Nenhum scan encontrado"}
    
    with open(scan_path, encoding="utf-8") as f:
        data = json.load(f)
    
    return {
        "total_files": data.get("total_files"),
        "total_lines": data.get("total_lines"),
        "large_files": data.get("large_files", []),
        "categories": data.get("categories", {}),
        "scan_timestamp": data.get("scan_timestamp"),
    }


# ═══════════════════════════════════════
# Monitor Endpoints
# ═══════════════════════════════════════

@app.get("/monitor/health")
def monitor_health():
    """Estado de saúde do sistema."""
    health_file = Path("auto_fixer/output/system_health.json")
    if health_file.exists():
        with open(health_file, encoding="utf-8") as f:
            return json.load(f)
    return {"status": "no_data", "message": "Health monitor não rodando"}


@app.get("/monitor/changes")
def monitor_changes(limit: int = 20):
    """Mudanças recentes nos arquivos."""
    state_file = Path("auto_fixer/output/watch_state.json")
    if state_file.exists():
        with open(state_file, encoding="utf-8") as f:
            state = json.load(f)
        changes = state.get("changes_history", [])
        return {
            "total": len(changes),
            "recent": changes[-limit:],
            "files_monitored": len(state.get("file_hashes", {})),
        }
    return {"total": 0, "recent": [], "files_monitored": 0}


@app.get("/monitor/alerts")
def monitor_alerts():
    """Alertas ativos."""
    alerts_file = Path("auto_fixer/output/alerts/active_alerts.jsonl")
    if not alerts_file.exists():
        return {"alerts": [], "total": 0}

    alerts = []
    with open(alerts_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                alerts.append(json.loads(line))

    return {
        "alerts": alerts[-50:],
        "total": len(alerts),
    }
