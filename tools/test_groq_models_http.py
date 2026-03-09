"""
Teste de modelos Groq via HTTP direto (sem depender do pacote openai).
Uso: python tools/test_groq_models_http.py
"""

import os
import sys
import time
import json
from pathlib import Path

# Carrega .env
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)
    print(f"✅ .env carregado de: {env_path}")
except ImportError:
    print("⚠️  python-dotenv não instalado, usando variáveis do ambiente")

# Verifica a chave
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("❌ GROQ_API_KEY não encontrada!")
    sys.exit(1)

print(f"✅ GROQ_API_KEY encontrada: {api_key[:8]}...{api_key[-4:]}")

# Tenta httpx primeiro, senão usa requests
try:
    import httpx
    HTTP_CLIENT = "httpx"
    print("✅ Usando httpx")
except ImportError:
    try:
        import requests
        HTTP_CLIENT = "requests"
        print("✅ Usando requests")
    except ImportError:
        print("❌ Instale httpx ou requests: pip install httpx")
        sys.exit(1)


def call_groq(model: str, api_key: str) -> dict:
    """Faz chamada direta à API Groq via HTTP."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Responda apenas OK."},
            {"role": "user", "content": "Teste de conectividade."}
        ],
        "temperature": 0,
        "max_tokens": 10
    }

    if HTTP_CLIENT == "httpx":
        resp = httpx.post(url, headers=headers, json=payload, timeout=15)
    else:
        resp = requests.post(url, headers=headers, json=payload, timeout=15)

    return {
        "status_code": resp.status_code,
        "body": resp.json() if resp.status_code != 500 else {"error": resp.text}
    }


# Modelos para testar
MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "deepseek-r1-distill-llama-70b",
    "qwen-qwq-32b",
]

print()
print("=" * 70)
print("🧪 TESTE DE MODELOS GROQ (HTTP direto)")
print("=" * 70)

results = []

for model in MODELS:
    print(f"\n🔄 Testando: {model}")
    start = time.time()

    try:
        resp = call_groq(model, api_key)
        elapsed = time.time() - start
        code = resp["status_code"]
        body = resp["body"]

        if code == 200:
            content = body["choices"][0]["message"]["content"].strip()
            tokens = body.get("usage", {}).get("total_tokens", "?")
            print(f"   ✅ SUCESSO | Resposta: '{content}' | "
                  f"Tempo: {elapsed:.2f}s | Tokens: {tokens}")
            results.append({
                "model": model, "status": "OK",
                "time": f"{elapsed:.2f}s", "response": content
            })
        else:
            error = body.get("error", {})
            if isinstance(error, dict):
                msg = error.get("message", str(error))[:100]
                err_code = error.get("code", code)
            else:
                msg = str(error)[:100]
                err_code = code

            status_map = {
                401: "API KEY inválida",
                403: "BLOQUEADO no projeto",
                404: "Modelo não existe",
                429: "Rate limit excedido",
            }
            short = status_map.get(code, f"HTTP {code}")

            print(f"   ❌ {short} | {msg} | Tempo: {elapsed:.2f}s")
            results.append({
                "model": model, "status": "ERRO",
                "time": f"{elapsed:.2f}s", "error": f"{short}: {msg}"
            })

    except Exception as e:
        elapsed = time.time() - start
        print(f"   ❌ EXCEÇÃO | {type(e).__name__}: {str(e)[:80]} | "
              f"Tempo: {elapsed:.2f}s")
        results.append({
            "model": model, "status": "ERRO",
            "time": f"{elapsed:.2f}s",
            "error": f"{type(e).__name__}: {str(e)[:80]}"
        })

# Resumo
print()
print("=" * 70)
print("📊 RESUMO")
print("=" * 70)

ok_models = [r for r in results if r["status"] == "OK"]
err_models = [r for r in results if r["status"] == "ERRO"]

print(f"\n✅ Funcionando: {len(ok_models)}/{len(results)}")
for r in ok_models:
    print(f"   ✅ {r['model']} ({r['time']})")

if err_models:
    print(f"\n❌ Com erro: {len(err_models)}/{len(results)}")
    for r in err_models:
        print(f"   ❌ {r['model']} → {r['error']}")

# Recomendação
print()
print("=" * 70)
print("💡 RECOMENDAÇÃO")
print("=" * 70)

if ok_models:
    # Prioriza modelos maiores
    priority = [
        "llama-3.1-8b-instant",
        "llama-3.1-70b-versatile",
        "deepseek-r1-distill-llama-70b",
        "qwen-qwq-32b",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
        "llama-3.1-8b-instant",
        "llama3-8b-8192",
    ]
    best = ok_models[0]
    for p in priority:
        for r in ok_models:
            if r["model"] == p:
                best = r
                break
        if best["model"] in priority:
            break

    print(f"\n   🏆 Modelo recomendado: {best['model']}")
    print(f"   ⏱️  Tempo de resposta: {best['time']}")
    print()
    print("   Atualize no .env:")
    print(f'   GROQ_MODEL={best["model"]}')
else:
    print("\n   ⚠️  NENHUM modelo funcionou!")
    print("   1. Verifique GROQ_API_KEY")
    print("   2. Acesse https://console.groq.com/settings/project/limits")
    print("   3. Habilite os modelos no projeto")
