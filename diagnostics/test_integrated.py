
import asyncio
import os
import sys

# Adiciona diretório para imports
sys.path.append(os.getcwd())

async def run_diagnostics():
    print("🚀 INICIANDO DIAGNÓSTICO INTEGRADO DO SISTEMA")
    print("="*60)
    
    # 1. Latência
    from diagnostics.test_latency import LatencyTester
    lt = LatencyTester()
    print("\n[1/3] Testando Latência...")
    await asyncio.gather(lt.test_binance_latency(10), lt.test_groq_latency(2))
    lt_report = lt.analyze()
    
    # 2. Modelo ML
    from diagnostics.test_ml_model import MLModelTester
    mt = MLModelTester()
    print("\n[2/3] Validando Modelo ML...")
    ml_ok = mt.test_model_integrity()
    
    # 3. Decision System
    from diagnostics.test_decision_system import DecisionSystemTester
    dt = DecisionSystemTester()
    print("\n[3/3] Validando Sistema de Decisão...")
    decision_ok = dt.run_tests()
    
    # Resumo Final
    print("\n" + "="*60)
    print("📈 STATUS GERAL DO SISTEMA")
    print("="*60)
    
    binance_lat = lt_report.get('binance', {}).get('avg', 0)
    print(f"📡 Latência Binance: {binance_lat:.0f}ms {'✅' if binance_lat < 300 else '⚠️'}")
    print(f"🧠 Modelo ML:        {'✅ OK' if ml_ok else '❌ FALHA'}")
    print(f"🎯 Sistema Decisão:  {'✅ OK' if decision_ok else '❌ FALHA'}")
    
    if all([ml_ok, decision_ok, binance_lat < 500]):
        print("\n🎉 SISTEMA OPERACIONAL")
    else:
        print("\n⚠️ ATENÇÃO: Verifique os avisos acima antes de operar em real.")

if __name__ == "__main__":
    asyncio.run(run_diagnostics())
