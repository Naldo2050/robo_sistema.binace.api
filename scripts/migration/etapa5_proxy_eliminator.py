"""
ETAPA 5: Eliminar proxies da raiz em lotes seguros.
Analisa cada proxy, conta importadores e gera plano de eliminação.
Executar: python scripts/migration/etapa5_proxy_eliminator.py [--lote N]
"""

import os
import sys
from pathlib import Path
from collections import defaultdict


# Todos os proxies conhecidos
ALL_PROXIES = {
    "event_bus.py": "events.event_bus",
    "event_saver.py": "events.event_saver",
    "event_memory.py": "events.event_memory",
    "trade_buffer.py": "trading.trade_buffer",
    "fred_fetcher.py": "fetchers.fred_fetcher",
    "cross_asset_correlations.py": "market_analysis.cross_asset_correlations",
    "dynamic_volume_profile.py": "market_analysis.dynamic_volume_profile",
    "levels_registry.py": "market_analysis.levels_registry",
    "data_handler.py": "data_processing.data_handler",
    "data_enricher.py": "data_processing.data_enricher",
    "data_validator.py": "data_processing.data_validator",
    "data_quality_validator.py": "data_processing.data_quality_validator",
    "time_manager.py": "monitoring.time_manager",
    "health_monitor.py": "monitoring.health_monitor",
    "metrics_collector.py": "monitoring.metrics_collector",
    "format_utils.py": "common.format_utils",
    "context_collector.py": "fetchers.context_collector",
    "enrichment_integrator.py": "data_processing.enrichment_integrator",
    "feature_store.py": "data_processing.feature_store",
    "export_signals.py": "trading.export_signals",
    "historical_profiler.py": "market_analysis.historical_profiler",
    "report_generator.py": "common.report_generator",
    "optimize_ai_payload.py": "common.optimize_ai_payload",
    "payload_optimizer_config.py": "common.payload_optimizer_config",
    "ai_payload_compressor.py": "common.ai_payload_compressor",
    "ai_response_validator.py": "common.ai_response_validator",
    "fix_optimization.py": "data_processing.fix_optimization",
    "diagnose_optimization.py": "scripts.diagnostics.diagnose_optimization",
    "orderbook_fallback.py": "orderbook_core.orderbook_fallback",
}


def find_importers_detailed(proxy_name: str, real_module: str, root: str) -> dict:
    """Encontra importadores detalhados de um proxy."""
    stem = proxy_name.replace(".py", "")
    importers = {"from_proxy": [], "from_real": [], "ambiguous": []}

    for dirpath, _, filenames in os.walk(root):
        if any(skip in dirpath for skip in [".git", "__pycache__", ".venv", "venv"]):
            continue
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(dirpath, fname)
            rel = os.path.relpath(fpath, root)

            # Pular o próprio proxy e o módulo real
            if rel == proxy_name:
                continue

            try:
                content = Path(fpath).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            lines = content.splitlines()
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue

                # Import do proxy (raiz)
                if f"from {stem} import" in stripped or f"import {stem}" in stripped:
                    if f"from {real_module}" not in stripped:
                        importers["from_proxy"].append({
                            "file": rel,
                            "line": i + 1,
                            "code": stripped,
                        })

                # Import do módulo real
                if f"from {real_module}" in stripped:
                    importers["from_real"].append({
                        "file": rel,
                        "line": i + 1,
                        "code": stripped,
                    })

    return importers


def main():
    root = os.getcwd()
    lote_num = 1

    if "--lote" in sys.argv:
        idx = sys.argv.index("--lote")
        if idx + 1 < len(sys.argv):
            lote_num = int(sys.argv[idx + 1])

    print("=" * 60)
    print(f"ETAPA 5 - ANÁLISE DE PROXIES")
    print("=" * 60)

    # Ordenar por número de importadores (menos primeiro = mais seguro)
    proxy_stats = []
    for proxy_file, real_module in ALL_PROXIES.items():
        full_path = os.path.join(root, proxy_file)
        if not os.path.exists(full_path):
            continue

        importers = find_importers_detailed(proxy_file, real_module, root)
        proxy_stats.append({
            "proxy": proxy_file,
            "real": real_module,
            "proxy_importers": len(importers["from_proxy"]),
            "real_importers": len(importers["from_real"]),
            "details": importers,
        })

    # Ordenar: menos importadores do proxy = mais fácil de eliminar
    proxy_stats.sort(key=lambda x: x["proxy_importers"])

    # Mostrar todos
    print(f"\n{'Proxy':<35} {'Importadores do Proxy':>22} {'Já usam real':>15}")
    print("-" * 75)
    for ps in proxy_stats:
        safety = "🟢" if ps["proxy_importers"] <= 2 else "🟡" if ps["proxy_importers"] <= 5 else "🔴"
        print(f"  {safety} {ps['proxy']:<32} {ps['proxy_importers']:>5} {'':>10} {ps['real_importers']:>5}")

    # Definir lotes de 5
    lote_size = 5
    total_lotes = (len(proxy_stats) + lote_size - 1) // lote_size

    print(f"\n{'='*60}")
    print(f"📋 LOTE {lote_num} de {total_lotes}")
    print(f"{'='*60}")

    start = (lote_num - 1) * lote_size
    end = min(start + lote_size, len(proxy_stats))
    current_batch = proxy_stats[start:end]

    for ps in current_batch:
        print(f"\n📄 {ps['proxy']} → {ps['real']}")
        print(f"   Importadores que usam o proxy: {ps['proxy_importers']}")

        if ps["details"]["from_proxy"]:
            print("   Arquivos para atualizar:")
            for imp in ps["details"]["from_proxy"]:
                print(f"      {imp['file']}:{imp['line']}")
                print(f"         ATUAL:  {imp['code']}")
                old_stem = ps["proxy"].replace(".py", "")
                new_mod = ps["real"]
                new_code = imp["code"].replace(f"from {old_stem}", f"from {new_mod}")
                new_code = new_code.replace(f"import {old_stem}", f"from {new_mod} import *")
                print(f"         NOVO:   {new_code}")
        else:
            print("   ✅ Nenhum importador - pode deletar diretamente!")


if __name__ == "__main__":
    main()
