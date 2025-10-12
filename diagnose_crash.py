#!/usr/bin/env python3
"""
Diagnostica por que o sistema parou sozinho.
"""

import os
import sys
from pathlib import Path

def find_websocket_files():
    """Encontra arquivos que gerenciam WebSocket."""
    patterns = [
        "*websocket*.py",
        "main.py",
        "*binance*.py",
        "*stream*.py",
        "*connection*.py"
    ]
    
    found = []
    for pattern in patterns:
        found.extend(Path(".").glob(pattern))
    
    return sorted(set(found))


def analyze_file(filepath):
    """Analisa arquivo para problemas de reconnect."""
    issues = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')
    
    # Busca por padrões problemáticos
    has_reconnect = any('reconnect' in line.lower() for line in lines)
    has_retry = any('retry' in line.lower() for line in lines)
    has_ping_pong = any('ping' in line.lower() or 'pong' in line.lower() for line in lines)
    has_on_close = any('on_close' in line.lower() for line in lines)
    has_exception_handler = any('except' in line for line in lines)
    
    if not has_reconnect:
        issues.append("❌ Sem lógica de reconnect automático")
    
    if not has_retry:
        issues.append("⚠️  Sem retry logic")
    
    if has_ping_pong and not has_on_close:
        issues.append("❌ Ping/pong sem tratamento de on_close")
    
    # Busca por sys.exit() ou return sem retry
    for i, line in enumerate(lines, 1):
        if 'sys.exit' in line.lower() or 'raise SystemExit' in line:
            issues.append(f"🔴 Linha {i}: Encerramento forçado sem retry")
        
        if 'connection' in line.lower() and 'close' in line.lower():
            if i < len(lines) - 5:
                next_lines = '\n'.join(lines[i:i+5])
                if 'reconnect' not in next_lines.lower():
                    issues.append(f"⚠️  Linha {i}: Close sem reconnect visível")
    
    return issues


def main():
    print("\n" + "="*70)
    print("🔍 DIAGNÓSTICO DE CRASH DO SISTEMA")
    print("="*70 + "\n")
    
    ws_files = find_websocket_files()
    
    if not ws_files:
        print("❌ Nenhum arquivo de WebSocket encontrado!")
        return
    
    print(f"📁 Arquivos encontrados: {len(ws_files)}\n")
    
    all_issues = {}
    
    for filepath in ws_files:
        print(f"📝 Analisando: {filepath}")
        try:
            issues = analyze_file(filepath)
            if issues:
                all_issues[str(filepath)] = issues
                for issue in issues:
                    print(f"   {issue}")
            else:
                print("   ✅ Sem problemas óbvios")
        except Exception as e:
            print(f"   ⚠️  Erro ao analisar: {e}")
        print()
    
    # Resumo
    print("="*70)
    print("📊 RESUMO")
    print("="*70 + "\n")
    
    if all_issues:
        print(f"❌ {len(all_issues)} arquivo(s) com problemas:\n")
        for filepath, issues in all_issues.items():
            print(f"📄 {filepath}:")
            for issue in issues:
                print(f"   • {issue}")
            print()
    else:
        print("✅ Nenhum problema óbvio detectado")
    
    print("\n💡 RECOMENDAÇÕES:")
    print("   1. Implemente reconnect automático no WebSocket")
    print("   2. Adicione retry logic com backoff exponencial")
    print("   3. Evite sys.exit() - use loop de retry")
    print("   4. Log detalhado de erros de conexão")
    print("   5. Ping/pong timeout deve triggar reconnect, não shutdown")
    print()
    
    # Lista arquivos para mostrar
    print("📋 Mostre-me o conteúdo destes arquivos:")
    for f in ws_files[:3]:  # Primeiros 3
        print(f"   • {f}")
    print()


if __name__ == "__main__":
    main()