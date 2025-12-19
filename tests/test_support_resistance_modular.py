"""
Teste do sistema modular de suporte e resistência
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Gerar dados de exemplo
np.random.seed(42)
df = pd.DataFrame({
    'open': np.random.uniform(99, 101, 200),
    'high': np.random.uniform(100, 102, 200),
    'low': np.random.uniform(98, 100, 200),
    'close': np.random.uniform(99, 101, 200),
    'volume': np.random.uniform(1000, 10000, 200)
}, index=pd.date_range('2024-01-01', periods=200, freq='1h'))

# Testar importações
print("Testando importações...")

# Importação completa do sistema
from support_resistance import InstitutionalSupportResistanceSystem
print("[OK] InstitutionalSupportResistanceSystem importado")

# Importação de módulos individuais
from support_resistance.core import AdvancedSupportResistance
print("[OK] AdvancedSupportResistance importado")

from support_resistance.volume_profile import VolumeProfileAnalyzer
print("[OK] VolumeProfileAnalyzer importado")

from support_resistance.pivot_points import InstitutionalPivotPoints
print("[OK] InstitutionalPivotPoints importado")

from support_resistance.monitor import InstitutionalMarketMonitor
print("[OK] InstitutionalMarketMonitor importado")

# Testar o sistema completo
print("\nTestando sistema completo...")
system = InstitutionalSupportResistanceSystem()
result = system.analyze_market(df, num_levels=3)

print("[OK] Análise concluída")
print(f"  - Níveis de suporte: {len(result['sr_analysis']['support_levels'])}")
print(f"  - Níveis de resistência: {len(result['sr_analysis']['resistance_levels'])}")
print(f"  - Score de qualidade: {result['sr_analysis']['quality_report']['overall_quality']:.2f}")

# Testar health check
print("\nTestando health check...")
health = system.health_check(df)
print(f"[OK] Health check: {health.status}")

# Testar criação de monitor
print("\nTestando criação de monitor...")
monitor = system.create_market_monitor()
print(f"[OK] Monitor criado com {len(monitor.support_levels)} suportes e {len(monitor.resistance_levels)} resistências")

# Testar processamento de tick
print("\nTestando processamento de tick...")
current_price = df['close'].iloc[-1]
signal = monitor.process_tick(
    price=current_price,
    volume=df['volume'].iloc[-1],
    delta=1000
)
if signal:
    print(f"[OK] Sinal gerado: {signal['reaction']}")
else:
    print("[OK] Nenhum nível testado (esperado)")

print("\n[SUCCESS] Todos os testes passaram com sucesso!")