#!/usr/bin/env python3
"""
check_diff_ruff.py - Gate incremental do Ruff para CI.

Verifica apenas as linhas adicionadas ou modificadas em relacao a uma base git.
Impede a introducao de novas violacoes sem bloquear o build pelas 12k violacoes
legadas pre-existentes no repositorio.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Dict, List, Set


def get_modified_lines(base: str, head: str = "HEAD") -> Dict[str, Set[int]]:
    """
    Executa git diff -U0 e retorna um dicionario mapeando o caminho do arquivo
    para o conjunto de numeros de linhas adicionadas/modificadas.
    """
    if head == "HEAD" or not head:
        cmd = ["git", "diff", "-U0", base]
    else:
        cmd = ["git", "diff", "-U0", f"{base}...{head}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Tenta diff direto caso three-dot falhe
        cmd = ["git", "diff", "-U0", base, head]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Falha ao executar git diff: {result.stderr}")

    modified_lines: Dict[str, Set[int]] = {}
    current_file = None

    # Regex para capturar cabecalho de arquivo no diff: +++ b/path/to/file.py
    file_re = re.compile(r"^\+\+\+ b/(.+)$")
    # Regex para capturar hunks: @@ -a,b +start,count @@ ou @@ -a +start @@
    hunk_re = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

    for line in result.stdout.splitlines():
        file_match = file_re.match(line)
        if file_match:
            current_file = file_match.group(1).replace("\\", "/")
            continue

        if current_file is None:
            continue

        hunk_match = hunk_re.match(line)
        if hunk_match:
            start = int(hunk_match.group(1))
            count_str = hunk_match.group(2)
            count = int(count_str) if count_str is not None else 1
            if count > 0:
                if current_file not in modified_lines:
                    modified_lines[current_file] = set()
                for line_num in range(start, start + count):
                    modified_lines[current_file].add(line_num)

    # Se estiver comparando contra a working tree (head == "HEAD" ou vazio),
    # inclui tambem arquivos untracked (novos ainda nao adicionados via git add).
    if head == "HEAD" or not head:
        cmd_untracked = ["git", "ls-files", "--others", "--exclude-standard"]
        res_untracked = subprocess.run(cmd_untracked, capture_output=True, text=True)
        if res_untracked.returncode == 0:
            for untracked_path in res_untracked.stdout.splitlines():
                untracked_path = untracked_path.strip().replace("\\", "/")
                if untracked_path.endswith(".py") and os.path.exists(untracked_path):
                    try:
                        with open(untracked_path, "r", encoding="utf-8", errors="replace") as uf:
                            line_count = len(uf.readlines())
                        modified_lines[untracked_path] = set(range(1, line_count + 1))
                    except Exception:
                        pass

    return modified_lines


def run_ruff_incremental(
    base: str,
    head: str = "HEAD",
    select: str = "E,F,W",
    ignore: str = "E501",
    ruff_bin: str = "ruff",
) -> int:
    """
    Executa o ruff e filtra diagnosticos apenas para as linhas alteradas.
    """
    try:
        modified_lines = get_modified_lines(base, head)
    except Exception as e:
        print(f"Erro ao calcular diff git: {e}", file=sys.stderr)
        return 2

    # Filtra apenas arquivos Python que ainda existem no filesystem
    py_files = [f for f in modified_lines.keys() if f.endswith(".py") and os.path.exists(f)]

    if not py_files:
        print("Nenhum arquivo Python alterado no diff.")
        return 0

    # Executa o ruff nos arquivos alterados com saida JSON
    if ruff_bin == "ruff":
        resolved = shutil.which("ruff")
        base_cmd = [resolved] if resolved else [sys.executable, "-m", "ruff"]
    else:
        base_cmd = [ruff_bin]

    cmd = [
        *base_cmd,
        "check",
        "--output-format=json",
        f"--select={select}",
        f"--ignore={ignore}",
        *py_files,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    try:
        diagnostics = json.loads(result.stdout) if result.stdout.strip() else []
    except json.JSONDecodeError:
        print(f"Erro ao interpretar saida do Ruff: {result.stderr or result.stdout}", file=sys.stderr)
        return 2

    new_violations: List[Dict] = []
    legacy_count = 0

    for d in diagnostics:
        filepath = d.get("filename", "").replace("\\", "/")
        # Normaliza caminho relativo ao repo
        for py_f in py_files:
            if filepath.endswith(py_f):
                filepath = py_f
                break

        row = d.get("location", {}).get("row", 0)
        file_lines = modified_lines.get(filepath, set())

        if row in file_lines:
            new_violations.append({
                "file": filepath,
                "line": row,
                "col": d.get("location", {}).get("column", 0),
                "code": d.get("code", "UNKNOWN"),
                "message": d.get("message", ""),
            })
        else:
            legacy_count += 1

    if new_violations:
        print(f"FALHA: {len(new_violations)} nova(s) violacao(oes) do Ruff encontrada(s) no diff:")
        for v in new_violations:
            print(f"  {v['file']}:{v['line']}:{v['col']} [{v['code']}] {v['message']}")
        print(f"\n({legacy_count} violacoes pre-existentes em linhas intocadas foram ignoradas)")
        return 1

    print(
        f"SUCESSO: Nenhuma nova violacao do Ruff nas {sum(len(lines) for lines in modified_lines.values())} "
        f"linha(s) alterada(s) em {len(py_files)} arquivo(s) Python."
    )
    if legacy_count > 0:
        print(f"({legacy_count} violacoes pre-existentes em linhas intocadas foram ignoradas pelo gate incremental)")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Verifica novas violacoes do Ruff nas linhas alteradas.")
    parser.add_argument("--base", default="origin/main", help="Commit ou branch base para o diff (padrao: origin/main)")
    parser.add_argument("--head", default="HEAD", help="Commit ou branch head (padrao: HEAD)")
    parser.add_argument("--select", default="E,F,W", help="Regras selecionadas do Ruff (padrao: E,F,W)")
    parser.add_argument("--ignore", default="E501", help="Regras ignoradas do Ruff (padrao: E501)")
    parser.add_argument("--ruff-bin", default="ruff", help="Caminho do binario do Ruff (padrao: ruff)")
    args = parser.parse_args()

    sys.exit(run_ruff_incremental(
        base=args.base,
        head=args.head,
        select=args.select,
        ignore=args.ignore,
        ruff_bin=args.ruff_bin,
    ))


if __name__ == "__main__":
    main()
