"""
Teste de modelos Groq - VERSÃO ATUALIZADA (Março 2025)
Uso: python tools/test_groq_models_v2.py
"""

import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("❌ GROQ_API_KEY não encontrada!")
    sys.exit(1)

print(f"✅ GROQ_API_KEY: {api_key[:8]}...{api_key[-4:]}\n")

try:
    import httpx
except ImportError:
    print("❌ httpx não instalado: pip install httpx")
    sys.exit(1)


def test_model(model: str) -> dict:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Responda OK"}],
        "temperature": 0,
        "max_tokens": 5
    }
    
    start = time.time()
    try:
        resp = httpx.post(url, headers=headers, json=payload, timeout=15)
        elapsed = time.time() - start
        
        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            return {"status": "OK", "time": elapsed, "response": content}
        else:
            error = resp.json().get("error", {})
            msg = error.get("message", "")[:80] if isinstance(error, dict) else str(error)[:80]
            return {"status": "ERRO", "time": elapsed, "code": resp.status_code, "msg": msg}
    except Exception as e:
        return {"status": "ERRO", "time": time.time() - start, "msg": str(e)[:80]}


# Modelos ATUAIS da Groq (Março 2025)
MODELS = [
    # Llama 3.3/3.2 (atuais)
    "llama-3.3-70b-versatile",
    "llama-3.3-70b-specdec",
    "llama-3.2-90b-vision-preview",
    "llama-3.2-11b-vision-preview",
    "llama-3.2-3b-preview",
    "llama-3.2-1b-preview",
    
    # Llama 3.1 (ainda suportados)
    "llama-3.1-8b-instant",
    
    # Novos modelos Groq
    "compound-beta",
    "compound-beta-mini",
    
    # Outros
    "llama-guard-3-8b",
    "whisper-large-v3",
    "whisper-large-v3-turbo",
]

print("=" * 60)
print("🧪 TESTE DE MODELOS GROQ (Março 2025)")
print("=" * 60)

ok = []
err = []

for model in MODELS:
    print(f"\n🔄 {model}...", end=" ", flush=True)
    r = test_model(model)
    
    if r["status"] == "OK":
        print(f"✅ OK ({r['time']:.2f}s)")
        ok.append({"model": model, **r})
    else:
        code = r.get("code", "?")
        msg = r.get("msg", "")
        if "blocked" in msg.lower():
            print(f"🔒 BLOQUEADO")
        elif "decommissioned" in msg.lower():
            print(f"💀 DESCONTINUADO")
        else:
            print(f"❌ {code}")
        err.append({"model": model, **r})

print("\n" + "=" * 60)
print("📊 RESUMO")
print("=" * 60)

if ok:
    print(f"\n✅ FUNCIONANDO: {len(ok)}")
    for r in ok:
        print(f"   ✅ {r['model']} ({r['time']:.2f}s)")
    
    # Recomendação por prioridade
    priority = ["llama-3.3-70b-versatile", "llama-3.3-70b-specdec", 
                "llama-3.1-8b-instant", "compound-beta"]
    best = ok[0]["model"]
    for p in priority:
        if any(r["model"] == p for r in ok):
            best = p
            break
    
    print(f"\n🏆 RECOMENDADO: {best}")
    print(f"   Adicione no .env: GROQ_MODEL={best}")
else:
    print("\n❌ NENHUM modelo funcionou!")
    print("\n🔧 AÇÕES NECESSÁRIAS:")
    print("   1. Acesse: https://console.groq.com/settings/project/limits")
    print("   2. Habilite os modelos bloqueados")
    print("   3. Rode este teste novamente")
