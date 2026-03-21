"""
ETAPA 4: Diagnosticar o diretório src/ e planejar absorção.
Executar: python scripts/migration/etapa4_check_src.py
"""

import os
import ast
from pathlib import Path


SRC_MAPPING = {
    "src/analysis/regime_detector.py": {
        "destino": "market_analysis/regime_detector.py",
        "razao": "Análise de mercado pertence a market_analysis/",
    },
    "src/analysis/regime_integration.py": {
        "destino": "market_analysis/regime_integration.py",
        "razao": "Integração de regime pertence a market_analysis/",
    },
    "src/analysis/integrate_regime_detector.py": {
        "destino": "market_analysis/integrate_regime_detector.py",
        "razao": "Integração de regime pertence a market_analysis/",
    },
    "src/analysis/ai_payload_integrator.py": {
        "destino": "common/ai_payload_integrator.py",
        "razao": "Integração de payload pertence a common/",
    },
    "src/bridges/async_bridge.py": {
        "destino": "common/async_bridge.py",
        "razao": "Bridge async é utilitário comum",
    },
    "src/data/macro_data_provider.py": {
        "destino": "fetchers/macro_data_provider.py",
        "razao": "Provider de dados macro pertence a fetchers/",
    },
    "src/rules/regime_rules.py": {
        "destino": "market_analysis/regime_rules.py",
        "razao": "Regras de regime pertencem a market_analysis/",
    },
    "src/services/macro_service.py": {
        "destino": "fetchers/macro_service.py",
        "razao": "Serviço de macro pertence a fetchers/",
    },
    "src/services/macro_update_service.py": {
        "destino": "fetchers/macro_update_service.py",
        "razao": "Serviço de update macro pertence a fetchers/",
    },
    "src/utils/ai_payload_optimizer.py": {
        "destino": "common/ai_payload_optimizer.py",
        "razao": "Otimizador pertence a common/",
    },
    "src/utils/async_helpers.py": {
        "destino": "VERIFICAR",
        "razao": "Verificar duplicação com common/async_helpers.py",
    },
}


def find_src_importers(src_module: str, root: str) -> list[str]:
    """Encontra quem importa módulos de src/."""
    importers = []
    module_path = src_module.replace("/", ".").replace(".py", "")

    for dirpath, _, filenames in os.walk(root):
        if any(skip in dirpath for skip in [".git", "__pycache__", ".venv"]):
            continue
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                content = Path(fpath).read_text(encoding="utf-8", errors="ignore")
                if module_path in content or f"from {module_path}" in content:
                    rel = os.path.relpath(fpath, root)
                    if rel != src_module:
                        importers.append(rel)
            except Exception:
                pass
    return importers


def main():
    root = os.getcwd()

    print("=" * 60)
    print("ETAPA 4 - DIAGNÓSTICO src/")
    print("=" * 60)

    for src_file, info in SRC_MAPPING.items():
        full_path = os.path.join(root, src_file)
        exists = os.path.exists(full_path)

        if not exists:
            continue

        importers = find_src_importers(src_file, root)
        size = os.path.getsize(full_path)

        print(f"\n📄 {src_file}")
        print(f"   Tamanho: {size:,} bytes")
        print(f"   Destino: {info['destino']}")
        print(f"   Razão: {info['razao']}")
        print(f"   Importadores: {len(importers)}")
        for imp in importers:
            print(f"      └── {imp}")

        dest_path = os.path.join(root, info["destino"])
        if os.path.exists(dest_path):
            print(f"   ⚠️  DESTINO JÁ EXISTE: {info['destino']}")
            print(f"      Verificar se é duplicação!")

    data_files = [
        "src/data/indices_futures.csv",
        "src/data/macro_data.json",
    ]
    print(f"\n{'='*60}")
    print("📊 ARQUIVOS DE DADOS em src/data/")
    print(f"{'='*60}")
    for df in data_files:
        full_path = os.path.join(root, df)
        if os.path.exists(full_path):
            size = os.path.getsize(full_path)
            print(f"   📄 {df} ({size:,} bytes)")
            print(f"      Mover para: dados/{Path(df).name}")

    stub = "src/utils/types_fredapi.pyi"
    if os.path.exists(os.path.join(root, stub)):
        print(f"\n   📄 {stub}")
        print(f"      Mover para: stubs/fredapi.pyi ou manter como está")


if __name__ == "__main__":
    main()
