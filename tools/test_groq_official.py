"""
Teste de conectividade com Groq API usando a biblioteca oficial groq.
Uso: python tools/test_groq_official.py
"""

import os
import sys
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
    print("   Defina a variável GROQ_API_KEY ou atualize o arquivo .env")
    sys.exit(1)

print(f"✅ GROQ_API_KEY encontrada: {api_key[:8]}...{api_key[-4:]}")

# Importa cliente oficial groq
try:
    from groq import Groq
except ImportError:
    print("❌ Biblioteca 'groq' não instalada.")
    print("   Instale com: pip install groq")
    sys.exit(1)

print("✅ Biblioteca groq importada com sucesso")

# Configura cliente
client = Groq(api_key=api_key)

# Modelo a testar
MODEL = "llama-3.1-8b-instant"

print()
print("=" * 70)
print(f"🧪 TESTE DE CONECTIVIDADE GROQ - Modelo: {MODEL}")
print("=" * 70)

try:
    print("\n🔄 Enviando requisição...")
    
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "Olá! Responda apenas OK se puder me ouvir."
            }
        ],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,  # Não usar stream para teste simples
        stop=None
    )
    
    response = completion.choices[0].message.content
    print(f"\n✅ SUCESSO!")
    print(f"   Modelo: {MODEL}")
    print(f"   Resposta: {response}")
    print(f"   Tokens utilizados: {completion.usage.total_tokens if completion.usage else 'N/A'}")
    
    # Salva recomendação no arquivo
    print()
    print("=" * 70)
    print("💡 CONFIGURAÇÃO RECOMENDADA")
    print("=" * 70)
    print(f"\n   Adicione ao seu arquivo .env:")
    print(f"   GROQ_MODEL={MODEL}")
    print()
    print(f"   Para testar streaming, modifique o código para usar stream=True")
    
except Exception as e:
    print(f"\n❌ ERRO: {type(e).__name__}")
    print(f"   Mensagem: {str(e)}")
    print()
    print("   Verifique:")
    print("   1. Sua GROQ_API_KEY está correta?")
    print("   2. Acesse https://console.groq.com/settings/project/limits")
    print("   3. Verifique se o modelo está habilitado")
