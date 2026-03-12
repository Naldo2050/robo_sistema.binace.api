#!/usr/bin/env python3
"""Teste das importações do config.py"""

print("Testando importacoes do config.py...")

try:
    # Teste das importações principais que estavam falhando
    from config import ORDER_BOOK_DEPTH_LEVELS, CONTEXT_TIMEFRAMES
    print("OK - ORDER_BOOK_DEPTH_LEVELS importado:", ORDER_BOOK_DEPTH_LEVELS)
    print("OK - CONTEXT_TIMEFRAMES importado:", CONTEXT_TIMEFRAMES)
    
    # Teste de outras importações importantes
    from config import (
        SPREAD_TIGHT_THRESHOLD_BPS,
        ORDERBOOK_CRITICAL_IMBALANCE,
        CONTEXT_EMA_PERIOD,
        VP_ADVANCED,
        HEALTH_CHECK_TIMEOUT
    )
    print("OK - Todas as importacoes estao funcionando!")
    print("   SPREAD_TIGHT_THRESHOLD_BPS:", SPREAD_TIGHT_THRESHOLD_BPS)
    print("   ORDERBOOK_CRITICAL_IMBALANCE:", ORDERBOOK_CRITICAL_IMBALANCE)
    print("   CONTEXT_EMA_PERIOD:", CONTEXT_EMA_PERIOD)
    print("   VP_ADVANCED:", VP_ADVANCED)
    print("   HEALTH_CHECK_TIMEOUT:", HEALTH_CHECK_TIMEOUT)
    
except ImportError as e:
    print("ERRO de importacao:", e)
    exit(1)
except Exception as e:
    print("ERRO geral:", e)
    exit(1)

print("\nSUCESSO - Configuracoes carregadas corretamente!")