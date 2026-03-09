
import asyncio
import time
import aiohttp
import os
import statistics
from datetime import datetime
import json
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

class LatencyTester:
    def __init__(self):
        self.results = {
            'binance': [],
            'groq': [],
        }
    
    async def test_binance_latency(self, count=10):
        """Testa latência com Binance"""
        url = "https://api.binance.com/api/v3/time"
        async with aiohttp.ClientSession() as session:
            for i in range(count):
                start = time.perf_counter()
                try:
                    async with session.get(url, timeout=5) as response:
                        await response.json()
                        latency = (time.perf_counter() - start) * 1000
                        self.results['binance'].append(latency)
                        # print(f"Binance {i+1}: {latency:.0f}ms")
                except Exception as e:
                    print(f"Binance error: {e}")
                await asyncio.sleep(0.1)
    
    async def test_groq_latency(self, count=3):
        """Testa latência com Groq"""
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            print("⚠️ GROQ_API_KEY não encontrada no .env. Pulando teste Groq.")
            return

        url = "https://api.groq.com/openai/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        
        async with aiohttp.ClientSession() as session:
            for i in range(count):
                start = time.perf_counter()
                try:
                    async with session.get(url, headers=headers, timeout=10) as response:
                        await response.json()
                        latency = (time.perf_counter() - start) * 1000
                        self.results['groq'].append(latency)
                        # print(f"Groq {i+1}: {latency:.0f}ms")
                except Exception as e:
                    print(f"Groq error: {e}")
                await asyncio.sleep(0.5)
    
    def analyze(self):
        """Analisa resultados"""
        print("\n" + "="*50)
        print("📊 ANÁLISE DE LATÊNCIA")
        print("="*50)
        
        analysis_report = {}
        for service, latencies in self.results.items():
            if latencies:
                avg = statistics.mean(latencies)
                min_l = min(latencies)
                max_l = max(latencies)
                p95 = sorted(latencies)[int(len(latencies)*0.95)] if len(latencies) > 0 else avg
                
                print(f"\n{service.upper()}:")
                print(f"  Média: {avg:.0f}ms")
                print(f"  Min: {min_l:.0f}ms")
                print(f"  Max: {max_l:.0f}ms")
                print(f"  P95: {p95:.0f}ms")
                
                status = "✅ OK"
                if service == 'binance' and avg > 300:
                    status = "❌ ALTA"
                    print(f"  ❌ LATÊNCIA ALTA (alvo: <300ms)")
                elif service == 'binance':
                    print("  ✅ LATÊNCIA ACEITÁVEL")
                    
                analysis_report[service] = {"avg": avg, "status": status}
                        
        return analysis_report

async def main():
    tester = LatencyTester()
    print("🔍 Iniciando testes de latência...")
    
    await asyncio.gather(
        tester.test_binance_latency(10),
        tester.test_groq_latency(3)
    )
    
    tester.analyze()

if __name__ == "__main__":
    asyncio.run(main())
