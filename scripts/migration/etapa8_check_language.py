"""
ETAPA 8: Encontrar arquivos com nomes em português.
Executar: python scripts/migration/etapa8_check_language.py
"""

import os
from pathlib import Path

# Palavras em português comuns em nomes de arquivo
PT_WORDS = [
    "teste", "verificar", "validar", "correcao", "relatorio",
    "diagnostico", "rapido", "separador", "final", "otimizacao",
    "duplicata", "evento", "auditoria",
]


def find_pt_files(root: str) -> list[dict]:
    """Encontra arquivos com nomes em português."""
    results = []
    for dirpath, _, filenames in os.walk(root):
        if any(skip in dirpath for skip in [".git", "__pycache__", ".venv", "Regras", "docs"]):
            continue
        for fname in filenames:
            if not fname.endswith((".py", ".sh", ".md")):
                continue
            fname_lower = fname.lower()
            for word in PT_WORDS:
                if word in fname_lower:
                    fpath = os.path.join(dirpath, fname)
                    rel = os.path.relpath(fpath, root)
                    results.append({
                        "file": rel,
                        "word": word,
                        "location": "legacy" if "legacy" in rel else
                                   "tests" if "tests" in rel else
                                   "scripts" if "scripts" in rel else
                                   "production",
                    })
                    break
    return results


def main():
    root = os.getcwd()

    print("=" * 60)
    print("ETAPA 8 - ARQUIVOS COM NOMES EM PORTUGUES")
    print("=" * 60)

    files = find_pt_files(root)

    # Agrupar por localização
    by_location = {}
    for f in files:
        loc = f["location"]
        if loc not in by_location:
            by_location[loc] = []
        by_location[loc].append(f)

    for loc, items in sorted(by_location.items()):
        priority = {"production": "ALTA", "tests": "MEDIA", "scripts": "MEDIA", "legacy": "BAIXA"}
        print(f"\n{priority.get(loc, '?')} - {loc.upper()} ({len(items)} arquivos)")
        for item in items:
            print(f"   {item['file']} (palavra: '{item['word']}')")

    print(f"\n{'='*60}")
    print("RECOMENDACAO")
    print(f"{'='*60}")
    print("""
  PRIORIDADE:
  1. Producao: renomear (com proxy se necessario)
  2. Tests/Scripts: renomear quando tocar no arquivo
  3. Legacy: ignorar (vai ser removido eventualmente)

  Esta etapa e COSMETICA - faca por ultimo.
  Foque nas etapas anteriores primeiro.
    """)


if __name__ == "__main__":
    main()
