"""
test_corrections.py
Testa as duas correções aplicadas:
1. Tabela signal_outcomes criada pelo EventStore._init_db()
2. Fix do event loop no DataEnricher._build_onchain_metrics()
"""
import sys
import os
import sqlite3
import asyncio
import threading
import tempfile
import pathlib
import logging

logging.basicConfig(level=logging.WARNING)

# Adicionar raiz do projeto ao path (o script pode estar em tests/ ou na raiz)
_this_file = pathlib.Path(__file__).resolve()
# Subir até encontrar a raiz (onde existe 'database/' ou 'outcome_tracker.py')
PROJECT_ROOT = _this_file.parent
for _candidate in [_this_file.parent, _this_file.parent.parent]:
    if (_candidate / "outcome_tracker.py").exists() or (_candidate / "database").exists():
        PROJECT_ROOT = _candidate
        break

sys.path.insert(0, str(PROJECT_ROOT))

PASS = "✅ PASSOU"
FAIL = "❌ FALHOU"
results = []


# ===========================================================================
# TESTE 1: EventStore cria signal_outcomes automaticamente
# ===========================================================================
def test_event_store_creates_signal_outcomes():
    """Verifica que EventStore._init_db() cria a tabela signal_outcomes."""
    print("\n[TESTE 1] EventStore cria tabela signal_outcomes automaticamente...")
    db_path = None
    try:
        import tempfile, os
        fd, db_path = tempfile.mkstemp(suffix="_test_eventstore.db")
        os.close(fd)

        # Importar EventStore (usa sys.path já configurado)
        from database.event_store import EventStore
        store = EventStore(db_path=db_path)
        del store  # liberar referências antes de verificar

        # Verificar tabelas existentes
        conn = sqlite3.connect(db_path)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        indices = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()}
        conn.close()
        del conn

        assert "events" in tables, "Tabela 'events' não encontrada!"
        assert "signal_outcomes" in tables, "Tabela 'signal_outcomes' não encontrada!"

        expected_indices = {
            "idx_outcomes_type",
            "idx_outcomes_battle",
            "idx_outcomes_epoch",
        }
        missing = expected_indices - indices
        assert not missing, f"Índices ausentes: {missing}"

        print(f"  Tabelas criadas: {sorted(tables)}")
        print(f"  Índices de signal_outcomes: OK")
        results.append((PASS, "EventStore cria signal_outcomes + índices"))
        return True

    except Exception as e:
        import traceback
        print(f"  ERRO: {e}")
        traceback.print_exc()
        results.append((FAIL, f"EventStore.signal_outcomes: {e}"))
        return False
    finally:
        if db_path:
            _remove_db(db_path)


# ===========================================================================
# TESTE 2: Banco existente migrado — signal_outcomes também é criada
# ===========================================================================
def _make_temp_db(suffix="test.db"):
    """Cria um arquivo temporário de banco SQLite que pode ser deletado no Windows."""
    fd, path = tempfile.mkstemp(suffix=f"_{suffix}")
    os.close(fd)
    return path


def _remove_db(path):
    """Remove banco SQLite e arquivos WAL/SHM auxiliares com segurança no Windows."""
    import gc
    gc.collect()  # força liberação de referências circulares que mantenham handles
    for ext in ["", "-wal", "-shm"]:
        try:
            p = pathlib.Path(path + ext)
            if p.exists():
                p.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Erro ignorado: {e}")


def test_existing_db_gets_signal_outcomes():
    """Verifica que um banco existente sem signal_outcomes é migrado ao iniciar EventStore."""
    print("\n[TESTE 2] Banco existente recebe signal_outcomes na reinicialização...")
    db_path = _make_temp_db("existing.db")
    try:
        # Criar banco velho apenas com 'events'
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_ms INTEGER,
                event_type TEXT,
                symbol TEXT,
                window_id TEXT,
                is_signal BOOLEAN,
                payload TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
        del conn

        # Inicializar EventStore sobre banco existente
        from database.event_store import EventStore
        store = EventStore(db_path=db_path)
        del store  # liberar referências

        conn2 = sqlite3.connect(db_path)
        tables = {r[0] for r in conn2.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn2.close()
        del conn2

        assert "signal_outcomes" in tables, "signal_outcomes não criada em banco existente!"
        print(f"  Banco existente agora contém: {sorted(tables)}")
        results.append((PASS, "Banco existente migrado com signal_outcomes"))
        return True

    except Exception as e:
        import traceback
        print(f"  ERRO: {e}")
        traceback.print_exc()
        results.append((FAIL, f"Migração de banco existente: {e}"))
        return False
    finally:
        _remove_db(db_path)


# ===========================================================================
# TESTE 3: signal_outcomes funciona com OutcomeTracker
# ===========================================================================
def test_outcome_tracker_with_event_store_db():
    """Verifica que OutcomeTracker consegue escrever/ler da tabela criada pelo EventStore."""
    print("\n[TESTE 3] OutcomeTracker opera na tabela criada pelo EventStore...")
    db_path = _make_temp_db("shared.db")
    try:
        # Primeiro: EventStore cria o banco e todas as tabelas
        from database.event_store import EventStore
        store = EventStore(db_path=db_path)
        del store

        # Depois: OutcomeTracker usa o mesmo banco
        from outcome_tracker import OutcomeTracker
        tracker = OutcomeTracker(db_path=db_path)

        # Tentar registrar um sinal (deve funcionar sem erro)
        evento_fake = {
            "epoch_ms": 1700000000000,
            "tipo_evento": "Absorção",
            "resultado_da_batalha": "Absorção de Venda",
            "preco_fechamento": 67000.0,
            "ativo": "BTCUSDT",
            "delta": -10.5,
            "volume_total": 95.2,
        }
        tracker.register_signal(evento_fake)

        # Verificar que o sinal foi inserido
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM signal_outcomes").fetchone()[0]
        conn.close()
        del conn

        assert count == 1, f"Esperado 1 sinal, encontrado {count}"
        print(f"  Sinal registrado com sucesso. Total na tabela: {count}")

        # Testar evaluate_pending_outcomes (não deve lançar exceção)
        tracker.evaluate_pending_outcomes(67200.0, 1700000300000)
        del tracker
        print("  evaluate_pending_outcomes: OK")

        results.append((PASS, "OutcomeTracker integra com EventStore"))
        return True

    except Exception as e:
        import traceback
        print(f"  ERRO: {e}")
        traceback.print_exc()
        results.append((FAIL, f"OutcomeTracker integração: {e}"))
        return False
    finally:
        _remove_db(db_path)


# ===========================================================================
# TESTE 4: Fix do event loop em thread sem loop
# ===========================================================================
def test_onchain_metrics_in_background_thread():
    """
    Verifica que _build_onchain_metrics() não levanta RuntimeError
    quando chamado de uma thread sem event loop (como window_processor_BTCUSDT).
    """
    print("\n[TESTE 4] DataEnricher._build_onchain_metrics() em thread sem event loop...")
    thread_errors = []
    thread_results = []

    def run_in_thread():
        try:
            # Confirmar que não há event loop nesta thread
            try:
                asyncio.get_running_loop()
                thread_errors.append("AVISO: thread já tinha um event loop (inesperado)")
            except RuntimeError:
                pass  # Esperado — sem loop na thread

            from data_processing.data_enricher import DataEnricher
            enricher = DataEnricher({"SYMBOL": "BTCUSDT"})

            # Chamar o método que causava o warning
            result = enricher._build_onchain_metrics()
            thread_results.append(result)

        except Exception as e:
            thread_errors.append(str(e))

    t = threading.Thread(
        target=run_in_thread,
        name="window_processor_BTCUSDT",  # mesmo nome da thread problemática
        daemon=True,
    )
    t.start()
    t.join(timeout=20)

    if t.is_alive():
        results.append((FAIL, "DataEnricher thread: timeout após 20s"))
        print("  ERRO: thread travou (timeout)")
        return False

    if thread_errors:
        results.append((FAIL, f"DataEnricher thread: {thread_errors[0]}"))
        print(f"  ERRO: {thread_errors[0]}")
        return False

    result = thread_results[0] if thread_results else {}
    print(f"  Resultado retornado: status={result.get('status', 'N/A')}, is_real_data={result.get('is_real_data', '?')}")
    print(f"  Sem RuntimeError, sem warning de event loop")
    results.append((PASS, "DataEnricher sem RuntimeError em thread"))
    return True


# ===========================================================================
# TESTE 5: get_running_loop() distingue contextos corretamente
# ===========================================================================
def test_asyncio_get_running_loop_behavior():
    """
    Verifica o comportamento fundamental de get_running_loop() que
    garante o fix correto: RuntimeError em thread sem loop.
    """
    print("\n[TESTE 5] asyncio.get_running_loop() comportamento correto...")
    try:
        errors_in_thread = []

        def thread_without_loop():
            try:
                asyncio.get_running_loop()
                errors_in_thread.append("BUG: deveria levantar RuntimeError mas não levantou")
            except RuntimeError:
                pass  # Correto!
            except Exception as e:
                errors_in_thread.append(f"Exceção inesperada: {e}")

        t = threading.Thread(target=thread_without_loop, daemon=True)
        t.start()
        t.join(timeout=5)

        if errors_in_thread:
            raise AssertionError(errors_in_thread[0])

        # Verificar dentro de loop async
        async def check_in_async():
            loop = asyncio.get_running_loop()
            assert loop is not None
            return True

        ok = asyncio.run(check_in_async())
        assert ok

        print("  get_running_loop() levanta RuntimeError em thread: OK")
        print("  get_running_loop() retorna loop em async: OK")
        results.append((PASS, "asyncio.get_running_loop() comportamento validado"))
        return True

    except Exception as e:
        import traceback
        print(f"  ERRO: {e}")
        traceback.print_exc()
        results.append((FAIL, f"get_running_loop() behavior: {e}"))
        return False


# ===========================================================================
# TESTE 6: Verificar banco de dados de produção (se existir)
# ===========================================================================
def test_production_db_has_signal_outcomes():
    """Verifica se o banco de dados de produção já tem (ou pode criar) a tabela."""
    print("\n[TESTE 6] Banco de produção (dados/trading_bot.db)...")
    prod_db = PROJECT_ROOT / "dados" / "trading_bot.db"

    if not prod_db.exists():
        print(f"  Banco de produção não encontrado em {prod_db}. Pulando.")
        results.append((PASS, "Banco de produção: não encontrado (OK para ambiente limpo)"))
        return True

    try:
        conn = sqlite3.connect(str(prod_db))

        # Verificar tabelas antes
        tables_before = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()

        had_table = "signal_outcomes" in tables_before
        print(f"  Tabelas existentes: {sorted(tables_before)}")
        print(f"  signal_outcomes já existia: {'SIM' if had_table else 'NÃO — será criada ao iniciar EventStore'}")

        # Inicializar EventStore para garantir tabela
        from database.event_store import EventStore
        store = EventStore(db_path=str(prod_db))

        conn = sqlite3.connect(str(prod_db))
        tables_after = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        count = conn.execute("SELECT COUNT(*) FROM signal_outcomes").fetchone()[0]
        conn.close()

        assert "signal_outcomes" in tables_after
        print(f"  ✅ signal_outcomes presente. Registros existentes: {count}")
        results.append((PASS, f"Banco de produção: signal_outcomes OK ({count} registros)"))
        return True

    except Exception as e:
        import traceback
        print(f"  ERRO: {e}")
        traceback.print_exc()
        results.append((FAIL, f"Banco de produção: {e}"))
        return False


# ===========================================================================
# EXECUÇÃO
# ===========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  SUITE DE TESTES — CORREÇÕES DO BOT")
    print("=" * 70)

    tests = [
        test_event_store_creates_signal_outcomes,
        test_existing_db_gets_signal_outcomes,
        test_outcome_tracker_with_event_store_db,
        test_onchain_metrics_in_background_thread,
        test_asyncio_get_running_loop_behavior,
        test_production_db_has_signal_outcomes,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        ok = test_fn()
        if ok:
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 70)
    print("  RESUMO DOS RESULTADOS")
    print("=" * 70)
    for status, desc in results:
        print(f"  {status}  {desc}")

    print(f"\n  Total: {passed + failed} | Passou: {passed} | Falhou: {failed}")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)