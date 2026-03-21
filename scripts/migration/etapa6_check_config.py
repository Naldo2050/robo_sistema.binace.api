"""
ETAPA 6: Diagnosticar arquivos de configuração espalhados.
Executar: python scripts/migration/etapa6_check_config.py
"""

import os
from pathlib import Path


CONFIG_FILES = [
    "config.py",
    "config.json",
    "config/__init__.py",
    "config/model_config.yaml",
    ".env.example",
    "orderbook_core/orderbook_config.py",
    "data_pipeline/config.py",
    "support_resistance/config.py",
    "auto_fixer/config.json",
    "flow_analyzer/constants.py",
    "orderbook_core/constants.py",
    "support_resistance/constants.py",
]


def analyze_config_file(filepath: str) -> dict:
    """Analisa um arquivo de configuração."""
    result = {
        "exists": False,
        "size": 0,
        "type": "unknown",
        "importers": 0,
    }

    full_path = filepath
    if not os.path.exists(full_path):
        return result

    result["exists"] = True
    result["size"] = os.path.getsize(full_path)

    if filepath.endswith(".py"):
        result["type"] = "python"
    elif filepath.endswith(".json"):
        result["type"] = "json"
    elif filepath.endswith(".yaml") or filepath.endswith(".yml"):
        result["type"] = "yaml"
    elif filepath.endswith(".env") or "env" in filepath:
        result["type"] = "env"

    # Contar importadores
    stem = Path(filepath).stem
    parent = Path(filepath).parent.name
    search_term = f"from {parent}.{stem}" if parent != "." else f"from {stem}"

    count = 0
    for dirpath, _, filenames in os.walk("."):
        if any(skip in dirpath for skip in [".git", "__pycache__", ".venv"]):
            continue
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                content = Path(fpath).read_text(encoding="utf-8", errors="ignore")
                if search_term in content or f"import {stem}" in content:
                    count += 1
            except Exception:
                pass
    result["importers"] = count

    return result


def main():
    print("=" * 60)
    print("ETAPA 6 - DIAGNÓSTICO DE CONFIGURAÇÃO")
    print("=" * 60)

    for cf in CONFIG_FILES:
        info = analyze_config_file(cf)
        if not info["exists"]:
            print(f"  ❌ {cf} - não encontrado")
            continue

        print(f"\n  📄 {cf}")
        print(f"     Tipo: {info['type']}")
        print(f"     Tamanho: {info['size']:,} bytes")
        print(f"     Importadores: ~{info['importers']}")

    print(f"\n{'='*60}")
    print("📋 RECOMENDAÇÃO")
    print(f"{'='*60}")
    print("""
  PLANO:
  1. config.py (raiz) -> config/settings.py (manter proxy na raiz)
  2. config.json (raiz) -> config/defaults.json
  3. config/model_config.yaml -> manter (já está no lugar certo)
  4. Configs de pacotes (orderbook_config, etc) -> manter locais
  5. .env.example -> manter na raiz (padrão)

  ⚠️  config.py na raiz provavelmente tem MUITOS importadores.
  Recomenda-se manter como proxy após mover para config/settings.py
    """)


if __name__ == "__main__":
    main()
