#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para diagnosticar problemas com variaveis de ambiente
"""
import os
from pathlib import Path

print("=" * 60)
print("DIAGNOSTICO DE VARIAVEIS DE AMBIENTE")
print("=" * 60)

# Verificar se dotenv esta instalado
try:
    from dotenv import load_dotenv
    print("[OK] python-dotenv instalado")
except ImportError:
    print("[ERRO] python-dotenv NAO instalado!")
    print("   Execute: pip install python-dotenv")
    exit(1)

# Verificar arquivo .env
env_files = [
    Path('.env'),
    Path('.env.local'),
    Path('config/.env'),
]

env_found = None
for env_file in env_files:
    if env_file.exists():
        print(f"[OK] Arquivo encontrado: {env_file.absolute()}")
        env_found = env_file
        break
    else:
        print(f"[ ] Nao encontrado: {env_file.absolute()}")

if not env_found:
    print("\n[ERRO] NENHUM ARQUIVO .env ENCONTRADO!")
    print("   Crie um arquivo .env no diretorio do projeto")
    exit(1)

# Carregar .env
print(f"\nCarregando: {env_found.absolute()}")
load_dotenv(env_found)

# Verificar variaveis
print("\n" + "=" * 60)
print("VARIAVEIS DE AMBIENTE")
print("=" * 60)

# Lista de variaveis para verificar
vars_to_check = [
    'FRED_API_KEY',
    'ALPHAVANTAGE_API_KEY',
    'ALPHA_VANTAGE_API_KEY',
    'ALPHA_VANTAGE_KEY',
    'GROQ_API_KEY',
    'BINANCE_API_KEY',
]

for var in vars_to_check:
    value = os.getenv(var)
    if value:
        masked = value[:8] + "..." if len(value) > 8 else value
        print(f"[OK] {var} = {masked}")
    else:
        print(f"[ERRO] {var} = NAO DEFINIDA")

# Mostrar conteudo do .env (apenas nomes das variaveis)
print("\n" + "=" * 60)
print("CONTEUDO DO .env (apenas nomes das variaveis)")
print("=" * 60)

with open(env_found, 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            if '=' in line:
                var_name = line.split('=')[0]
                print(f"   {var_name}")
            else:
                print(f"   {line}")

print("\n" + "=" * 60)
print("DIAGNOSTICO CONCLUIDO")
print("=" * 60)
