# scripts/diagnostics/backup_shadow_dataset.py
# -*- coding: utf-8 -*-
"""
Rotina de Backup Não-Bloqueante e Teste de Restauração — Fase O1.
Utiliza a API canônica SQLite Online Backup (conn.backup) para:
1. Executar snapshot atômico sem pausar ou bloquear o processo escritor (WAL mode).
2. Abrir a cópia em banco separado e executar PRAGMA integrity_check.
3. Verificar a legibilidade de todas as tabelas críticas (events, positioning_shadow_dataset, signal_outcomes).
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import sys
import time
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SQLiteBackup")

DB_PATH = "dados/trading_bot.db"
BACKUP_DIR = "backups"


def perform_safe_backup_and_restore_test(
    src_db_path: str = DB_PATH,
    backup_dir: str = BACKUP_DIR,
    custom_tag: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executa backup a quente e valida a integridade da cópia restaurada.
    """
    t0 = time.time()
    os.makedirs(backup_dir, exist_ok=True)

    if not os.path.exists(src_db_path):
        return {
            "success": False,
            "error": f"Banco de origem não encontrado: {src_db_path}",
            "elapsed_seconds": 0.0,
        }

    tag = custom_tag or time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    backup_filename = f"shadow_dataset_{tag}.db"
    backup_path = os.path.join(backup_dir, backup_filename)

    logger.info(f"Iniciando backup online não-bloqueante: {src_db_path} -> {backup_path}")

    # 1. Executa SQLite Online Backup
    try:
        src_conn = sqlite3.connect(src_db_path, timeout=10.0)
        dest_conn = sqlite3.connect(backup_path)

        # Executa em fatias para garantir zero contenção com escritores concorrentes
        src_conn.backup(dest_conn, pages=250, sleep=0.01)

        dest_conn.close()
        src_conn.close()
        backup_size_bytes = os.path.getsize(backup_path)
        logger.info(f"Backup online concluído ({backup_size_bytes / 1024:.1f} KB)")
    except Exception as e:
        logger.error(f"Falha durante a execução do backup online: {e}")
        return {
            "success": False,
            "error": f"Erro durante cópia online: {e}",
            "elapsed_seconds": time.time() - t0,
        }

    # 2. Teste de Restauração e Integridade da Cópia
    logger.info(f"Iniciando teste de restauração e integridade em: {backup_path}")
    try:
        verify_conn = sqlite3.connect(backup_path)
        cur = verify_conn.cursor()

        # Checagem 1: PRAGMA integrity_check
        cur.execute("PRAGMA integrity_check;")
        integrity_res = cur.fetchall()
        integrity_ok = len(integrity_res) == 1 and integrity_res[0][0] == "ok"
        if not integrity_ok:
            raise ValueError(f"PRAGMA integrity_check falhou: {integrity_res}")

        # Checagem 2: Legibilidade de tabelas críticas
        table_counts: Dict[str, int] = {}
        for table in ["events", "positioning_shadow_dataset", "signal_outcomes"]:
            try:
                cur.execute(f"SELECT count(*) FROM {table}")
                table_counts[table] = cur.fetchone()[0] or 0
            except sqlite3.OperationalError:
                table_counts[table] = -1  # Tabela pode ainda não ter sido criada se banco novo

        verify_conn.close()

        elapsed = time.time() - t0
        logger.info(
            f"✅ Teste de restauração aprovado: integrity_check=OK, "
            f"tabelas={table_counts}, tempo={elapsed:.2f}s"
        )

        return {
            "success": True,
            "backup_path": backup_path,
            "backup_size_bytes": backup_size_bytes,
            "integrity_check": "ok",
            "table_counts": table_counts,
            "elapsed_seconds": round(elapsed, 3),
        }

    except Exception as e:
        logger.error(f"Falha na validação do backup restaurado: {e}")
        return {
            "success": False,
            "backup_path": backup_path,
            "error": f"Validação da cópia falhou: {e}",
            "elapsed_seconds": time.time() - t0,
        }


if __name__ == "__main__":
    res = perform_safe_backup_and_restore_test(custom_tag="preflight_test")
    print("\n" + "=" * 60)
    print("RESULTADO DO TESTE DE BACKUP & RESTAURAÇÃO (PRE-FLIGHT):")
    print(f"Sucesso:        {res.get('success')}")
    print(f"Caminho:        {res.get('backup_path')}")
    print(f"Integridade:    {res.get('integrity_check')}")
    print(f"Tabelas:        {res.get('table_counts')}")
    print(f"Tempo decorrido:{res.get('elapsed_seconds')}s")
    print("=" * 60)
    if not res.get("success"):
        sys.exit(1)
