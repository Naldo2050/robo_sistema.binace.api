# diagnostics/final_validation.py
# -*- coding: utf-8 -*-
"""
Script de Validação Final de Integração (Smoke Test).

Testa:
1. ContextCollector (Async IO + APIs)
2. FeatureStore (Escrita Parquet)
3. EventStore -> Backtester -> Dashboard (Fluxo de Banco de Dados)
"""

import asyncio
import logging
import os
import sys
import shutil
import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path

# Adiciona raiz ao path para imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IntegrationTest")

# Imports dos Módulos
try:
    from context_collector import ContextCollector
    from feature_store import FeatureStore
    from database.event_store import EventStore
    from tests.backtester import Backtester
except ImportError as e:
    logger.error(f"Erro de importação: {e}")
    logger.error("Execute este script da raiz do projeto: python diagnostics/final_validation.py")
    sys.exit(1)

# ==========================================
# 1. TESTE DO CONTEXT COLLECTOR (ASYNC)
# ==========================================
async def test_context_collector():
    logger.info("🔵 [1/4] Testando ContextCollector (Async)...")
    
    symbol = "BTCUSDT"
    collector = ContextCollector(symbol)
    collector.update_interval = 1  # intervalo curto para o loop dele

    try:
        collector.start()
        logger.info("   ⏳ Aguardando 30s para primeira coleta de contexto...")

        ctx = {}
        timeout = 30
        step = 5
        for _ in range(0, timeout, step):
            await asyncio.sleep(step)
            ctx = collector.get_context()
            if ctx.get("timestamp"):
                break

        checks = {
            "MTF Trends": bool(ctx.get("mtf_trends")),
            "Market Env": bool(ctx.get("market_environment")),
            "Derivatives": bool(ctx.get("derivatives")),
            "Timestamp": bool(ctx.get("timestamp")),
        }

        all_ok = all(checks.values())

        if all_ok:
            logger.info("   ✅ Contexto coletado com sucesso!")
            env = ctx.get("market_environment", {})
            logger.info(
                f"   📊 Regime detectado: {env.get('market_structure', 'N/A')} | Vol: {env.get('volatility_regime', 'N/A')}"
            )
        else:
            logger.error(f"   ❌ Falha na coleta. Checks: {checks}")

    except Exception as e:
        logger.error(f"   ❌ Erro no ContextCollector: {e}")
    finally:
        collector.stop()

# ==========================================
# 2. TESTE DO FEATURE STORE (PARQUET)
# ==========================================
def test_feature_store():
    logger.info("🔵 [2/4] Testando FeatureStore (Parquet)...")
    
    # Usa diretório temporário para não sujar o real
    test_dir = "features_test"
    store = FeatureStore(base_dir=test_dir)
    store.buffer_size_limit = 1  # Força salvar imediatamente
    
    dummy_data = {
        "price_close": 50000.0,
        "rsi_14": 55.5,
        "volume_sma": 1.2,
        "complex_data": {"a": 1, "b": 2}  # Testa flattening
    }
    
    try:
        store.save_features(window_id="TEST_WINDOW_001", features=dummy_data)
        
        # Verifica se arquivo existe
        today = datetime.utcnow().strftime("%Y-%m-%d")
        partition_path = Path(test_dir) / f"date={today}"
        
        if not partition_path.exists():
            logger.error("   ❌ Diretório de partição não criado.")
            return

        files = list(partition_path.glob("*.parquet"))
        if not files:
            logger.error("   ❌ Arquivo Parquet não encontrado.")
            return
            
        # Tenta ler de volta
        df = pd.read_parquet(files[0])
        if not df.empty and "price_close" in df.columns:
            logger.info(f"   ✅ Leitura/Escrita Parquet OK. Arquivo: {files[0]}")
            logger.info(f"   📄 Colunas: {list(df.columns)}")
        else:
            logger.error("   ❌ Parquet corrompido ou vazio.")

    except Exception as e:
        logger.error(f"   ❌ Erro no FeatureStore: {e}")
    finally:
        # Limpeza
        if os.path.exists(test_dir):
            try:
                shutil.rmtree(test_dir)
                logger.info("   🧹 Limpeza de teste realizada.")
            except Exception:
                pass

# ==========================================
# 3. TESTE DE INTEGRAÇÃO (DB -> BACKTESTER)
# ==========================================
def test_db_integration():
    logger.info("🔵 [3/4] Testando Integração Banco de Dados -> Backtester...")
    
    db_path = "dados/test_trading_bot.db"
    
    # 1. Configura Banco de Teste (remove anterior se possível)
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except PermissionError:
            logger.warning(
                "   ⚠️ Não foi possível remover DB de teste anterior (arquivo em uso). "
                "Prosseguindo com o mesmo arquivo."
            )
        except Exception as e:
            logger.warning(f"   ⚠️ Erro inesperado ao remover DB de teste anterior: {e}")
        
    store = EventStore(db_path=db_path)
    
    # 2. Insere Evento Simulado (Sinal de Compra)
    import json
    dummy_signal = {
        "tipo_evento": "AI_ANALYSIS",
        "is_signal": True,
        "epoch_ms": int(datetime.now().timestamp() * 1000),
        "symbol": "BTCUSDT",
        "resultado_da_batalha": "Absorção de Venda (Bullish)",
        "preco_fechamento": 60000.0,
        "delta": 150.0,
        "volume_total": 500.0,
        "descricao": "Teste de integração"
    }
    
    store.save_event(dummy_signal)
    logger.info("   💾 Evento de teste salvo no SQLite.")
    
    # 3. Tenta ler com Backtester
    bt = Backtester(db_path=Path(db_path))
    df = bt.load_signals()
    
    if not df.empty and len(df) == 1:
        logger.info("   ✅ Backtester leu o sinal corretamente!")
        logger.info(f"   📈 Preço lido: {df.iloc[0]['price']} | Side: {df.iloc[0]['side']}")
    else:
        logger.error(f"   ❌ Backtester falhou em ler o sinal. DF: {df}")

    # 4. Limpeza (tolerante a erro de arquivo em uso no Windows)
    try:
        # Se EventStore tiver close(), usar
        if hasattr(store, "close"):
            store.close()
    except Exception as e:
        logger.warning(f"   ⚠️ Falha ao fechar EventStore de teste: {e}")

    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            logger.info("   🧹 Banco de teste removido.")
        except PermissionError:
            logger.warning(
                "   ⚠️ Não foi possível remover o banco de teste (arquivo em uso). "
                "Isso não afeta o teste; o arquivo será reutilizado ou limpo depois."
            )
        except Exception as e:
            logger.warning(f"   ⚠️ Erro inesperado ao remover banco de teste: {e}")

# ==========================================
# 4. SIMULAÇÃO DE DASHBOARD QUERY
# ==========================================
def test_dashboard_query():
    logger.info("🔵 [4/4] Testando Query do Dashboard...")
    
    # Usa o banco real se existir, senão pula
    real_db = Path("dados/trading_bot.db")
    if not real_db.exists():
        logger.warning("   ⚠️ Banco real não encontrado, pulando teste de dashboard.")
        return

    try:
        conn = sqlite3.connect(f"file:{real_db}?mode=ro", uri=True)
        # Query idêntica à do dashboard.py
        query = """
            SELECT timestamp_ms, event_type, symbol, is_signal 
            FROM events 
            ORDER BY timestamp_ms DESC 
            LIMIT 5
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if not df.empty:
            logger.info(f"   ✅ Query do Dashboard OK. Retornou {len(df)} eventos recentes.")
        else:
            logger.info("   ✅ Query do Dashboard OK (mas banco estava vazio).")
            
    except Exception as e:
        logger.error(f"   ❌ Erro na query do Dashboard: {e}")

# ==========================================
# EXECUTOR
# ==========================================
async def main():
    print("=" * 60)
    print("🧪 INICIANDO VALIDAÇÃO DE SISTEMA (INSTITUCIONAL)")
    print("=" * 60)
    
    await test_context_collector()
    print("-" * 60)
    
    # Executa testes síncronos
    test_feature_store()
    print("-" * 60)
    
    test_db_integration()
    print("-" * 60)
    
    test_dashboard_query()
    
    print("=" * 60)
    print("🏁 VALIDAÇÃO CONCLUÍDA")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())