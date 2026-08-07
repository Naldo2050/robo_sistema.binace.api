import compileall, sys, os

print("==> Compile all...")
compileall.compile_dir('.', quiet=1, force=True)
print("OK")

print("==> Test import of critical modules...")
modules = [
    "events.event_bus", "trading.trade_buffer", "fetchers.fred_fetcher",
    "market_analysis.cross_asset_correlations", "data_processing.data_handler",
    "monitoring.time_manager", "common.format_utils", "institutional.base",
    "flow_analyzer", "support_resistance.core", "market_orchestrator.market_orchestrator",
    "orderbook_analyzer", "orderbook_core.orderbook", "ai_runner.ai_runner",
    "config.settings", "risk_management.risk_manager", "ml.inference_engine"
]
for mod in modules:
    try:
        __import__(mod)
        print(f"  {mod}: OK")
    except Exception as e:
        print(f"  {mod}: FAIL - {e}")

print("==> Checking for problematic residual imports...")
bad_patterns = [
    "from utils import", "import utils.", "from utils.",
    "import institutional_enricher", "from institutional_enricher",
    "import build_compact_payload", "from build_compact_payload",
    "import ai_analyzer_qwen", "from ai_analyzer_qwen",
    "from src.utils", "import src.utils",
    "from src.data", "import src.data"
]
# (grep nao existe no Windows; scan equivalente em Python puro)
found = []
for d, dirs, files in os.walk('.'):
    dirs[:] = [x for x in dirs if x not in ('__pycache__', '.venv', '.git', '.pytest_cache', 'htmlcov', 'coverage_html', 'node_modules')]
    for f in files:
        if not f.endswith('.py'):
            continue
        p = os.path.join(d, f)
        try:
            lines = open(p, encoding='utf-8', errors='surrogateescape').read().splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines, 1):
            if any(b in line for b in bad_patterns):
                found.append(f"{p}:{i}: {line.strip()}")
if found:
    print("Found residual imports:")
    print("\n".join(found))
else:
    print("None found.")
