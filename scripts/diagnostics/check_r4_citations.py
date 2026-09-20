#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/diagnostics/check_r4_citations.py
Verifica a existência de todos os arquivos e linhas citados no relatório R4.
"""

import os
import re

r4_file = "docs/audit/AUDITORIA_JANELAS_EXTRAIDAS_R4_2026-09-03.md"
with open(r4_file, "r", encoding="utf-8") as f:
    text = f.read()

# Expressão regular para capturar caminhos como caminho/arquivo.ext:linha ou caminho/arquivo.ext
# Exemplos: config/settings.py:133, orderbook_analyzer/core.py:1055, etc.
pattern = re.compile(r'([a-zA-Z0-9_\-\.\/]+\.(?:py|md|json|log|db|parquet|csv))(?::(\d+)(?:-(\d+))?)?')

citations = []
for match in pattern.finditer(text):
    path = match.group(1)
    start_line = match.group(2)
    end_line = match.group(3)
    citations.append((path, start_line, end_line, match.group(0)))

# Normalizar caminhos e verificar os.path.exists
unique_paths = sorted(list(set(c[0] for c in citations)))

print(f"Total de arquivos referenciados no texto da R4: {len(unique_paths)}")

results = []
for p in unique_paths:
    # Se começar com / ou relativo
    norm_p = os.path.normpath(p)
    exists = os.path.exists(norm_p)
    total_lines = 0
    if exists and os.path.isfile(norm_p):
        try:
            with open(norm_p, "r", encoding="utf-8", errors="ignore") as f_in:
                total_lines = len(f_in.readlines())
        except Exception:
            pass
    results.append((p, exists, total_lines))

print("\n--- RESULTADO DA VERIFICAÇÃO DE ARQUIVOS CITADOS ---")
not_found = []
for p, exists, n_lines in results:
    status = f"OK ({n_lines} linhas)" if exists else "NÃO EXISTE"
    print(f"{p:<60} -> {status}")
    if not exists:
        not_found.append(p)

print(f"\nTotal não encontrados: {len(not_found)}")
for nf in not_found:
    print(f"  [X] {nf}")

# Verificar agora citações com linha específica
print("\n--- VERIFICAÇÃO DE CITAÇÃO ESPECÍFICA ARQUIVO:LINHA ---")
line_errors = []
for path, s_line, e_line, raw in citations:
    if not os.path.exists(path):
        continue
    if s_line:
        s_num = int(s_line)
        # Ler total de linhas do arquivo
        with open(path, "r", encoding="utf-8", errors="ignore") as f_in:
            lines = f_in.readlines()
        max_l = len(lines)
        if s_num > max_l:
            line_errors.append((raw, f"Linha {s_num} excede total {max_l} de {path}"))

print(f"Erros de linha excedente: {len(line_errors)}")
for err in line_errors:
    print(" ", err)
