"""
Scheduler - Execução automática do auto-fixer.
"""

import time
import logging
import schedule  # type: ignore[import-untyped]  # pip install schedule
from datetime import datetime

logger = logging.getLogger(__name__)


def run_auto_audit():
    """Executa auditoria completa."""
    from auto_fixer.runner import (
        run_phase1, run_phase2, run_phase3,
        run_phase4, run_phase6, run_phase8
    )
    
    logger.info(f"🔄 Iniciando auditoria automática: {datetime.now()}")
    
    try:
        # Fases essenciais
        run_phase1(".")
        run_phase2()
        run_phase3()
        run_phase4()
        
        # Análise
        issues = run_phase6()
        
        # Relatório
        run_phase8()
        
        # Verificar CRITICAL
        critical = [
            i for i in issues 
            if hasattr(i, 'severity') and i.severity.value == "CRITICAL"
        ]
        
        if critical:
            logger.critical(
                f"⚠️ {len(critical)} ISSUES CRÍTICOS ENCONTRADOS!"
            )
            _send_alert(critical)
        else:
            logger.info("✅ Nenhum issue crítico encontrado")
    
    except Exception as e:
        logger.error(f"Erro na auditoria: {e}")


def _send_alert(issues):
    """Envia alerta (adaptar para seu sistema de notificação)."""
    # Opção 1: Salvar arquivo de alerta
    from pathlib import Path
    alert_path = Path("auto_fixer/output/CRITICAL_ALERT.txt")
    
    with open(alert_path, "w", encoding="utf-8") as f:
        f.write(f"⚠️ ALERTA CRÍTICO - {datetime.now()}\n\n")
        for issue in issues:
            f.write(f"- {issue.title}\n")
            f.write(f"  Arquivo: {issue.file_path}\n")
            f.write(f"  Linha: {issue.line_start}\n\n")
    
    logger.warning(f"Alerta salvo em: {alert_path}")


def start_scheduler():
    """Inicia o scheduler."""
    # A cada 6 horas
    schedule.every(6).hours.do(run_auto_audit)
    
    # Também à meia-noite
    schedule.every().day.at("00:00").do(run_auto_audit)
    
    logger.info("📅 Scheduler iniciado")
    logger.info("   - A cada 6 horas")
    logger.info("   - Diariamente à meia-noite")
    
    # Executar uma vez agora
    run_auto_audit()
    
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start_scheduler()
