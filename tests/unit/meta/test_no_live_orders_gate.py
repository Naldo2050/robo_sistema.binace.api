"""
test_no_live_orders_gate.py - Gate Meta: Inexistência de Envio de Ordens Reais.

Garante de forma estática e em nível de meta-governança que:
1. Nenhum SDK externo de execução de ordens (python-binance, ccxt) está presente
   nos arquivos de dependências de produção (requirements.txt).
2. O código de produção e scripts operacionais não realizam chamadas a endpoints
   privados de envio de ordem da Binance (/api/v3/order, /fapi/v1/order, futures_create_order).
3. O repositório opera estritamente como PASSIVE OBSERVER / ANALYTICS / PAPER TRADING,
   garantindo risco financeiro zero de execução inadvertida.

Limitação Técnica Documentada:
O scanner realiza análise léxica estática via varredura de strings e regex em código Python.
Ele identifica endpoints e imports explícitos. Não cobre chamadas dinâmicas ofuscadas via
metaprogramação reflexiva (e.g. getattr(obj, "order" + "create")()) ou execução de código
dinâmico (eval/exec). A proteção em profundidade do bot conta com credenciais somente de leitura
e ausência de permissões de trade no nível da API da Binance.
"""

import os
import re
from pathlib import Path
from unittest.mock import patch
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

FORBIDDEN_ORDER_ENDPOINTS = [
    "/api/v3/order",
    "/fapi/v1/order",
    "/fapi/v2/order",
    "/fapi/v1/batchOrders",
    "futures_create_order",
    "order_market_buy",
    "order_market_sell",
]

FORBIDDEN_SDKS = [
    "python-binance",
    "ccxt",
]

FORBIDDEN_IMPORT_PATTERNS = [
    re.compile(r"^\s*(from|import)\s+ccxt\b"),
    re.compile(r"^\s*from\s+binance\.client\s+import\b"),
    re.compile(r"^\s*from\s+binance\.(spot|futures)\s+import\b"),
    re.compile(r"^\s*import\s+binance\.client\b"),
]

OPERATIONAL_DIRS = [
    "market_orchestrator",
    "fetchers",
    "paper_trading",
    "monitoring",
    "common",
    "infrastructure",
    "config",
    "scripts",
]

# Exclusão explícita justificada por caminho relativo exato:
# scripts/diagnostics/verify_safe_mode.py é um script forense de diagnóstico cujo propósito
# exclusivo é conter a lista de endpoints proibidos para varrer e atestar a segurança do código.
EXCLUDED_OPERATIONAL_RELATIVE_PATHS = {
    "scripts/diagnostics/verify_safe_mode.py",
}


def _walk_error_handler(err: OSError) -> None:
    """Dispara erro imediatamente em caso de falha de acesso a subdiretório no os.walk."""
    raise OSError(f"Falha de permissão ou I/O ao percorrer diretório: {err.filename} ({err})")


def _scan_for_forbidden_patterns(
    target_dir: Path, repo_root: Path = REPO_ROOT
) -> tuple[list[tuple[str, int, str]], list[tuple[str, int, str]]]:
    """
    Varre recursivamente o diretório em busca de endpoints proibidos e imports proibidos de SDK.
    Retorna (endpoint_violations, import_violations).
    Nenhum erro de leitura de arquivo ou de acesso a diretório é silenciado.
    Falha imediatamente se o diretório não existir (fail-closed).
    """
    if not target_dir.exists():
        raise FileNotFoundError(f"Diretório operacional obrigatório não encontrado para auditoria: {target_dir}")

    endpoint_violations = []
    import_violations = []

    resolved_root = repo_root.resolve()

    for root, _, files in os.walk(target_dir, onerror=_walk_error_handler):
        # Ignora caches, venvs e testes
        if any(ignored in root for ignored in [".git", ".venv", "__pycache__", "tests"]):
            continue
        for f in files:
            fpath = Path(root) / f
            try:
                rel_path = fpath.resolve().relative_to(resolved_root).as_posix()
            except ValueError:
                rel_path = fpath.as_posix()

            if rel_path in EXCLUDED_OPERATIONAL_RELATIVE_PATHS:
                continue

            if f.endswith(".py"):
                fpath = Path(root) / f
                # Leitura estrita sem engolir exceções: falha se o arquivo for ilegível
                text = fpath.read_text(encoding="utf-8", errors="strict")
                for line_no, line in enumerate(text.splitlines(), 1):
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        continue

                    # 1. Checagem de endpoints de ordens privadas
                    for endpoint in FORBIDDEN_ORDER_ENDPOINTS:
                        if endpoint in line:
                            endpoint_violations.append((str(fpath), line_no, endpoint))

                    # 2. Checagem de imports proibidos de SDK de exchange
                    for pat in FORBIDDEN_IMPORT_PATTERNS:
                        if pat.search(line):
                            import_violations.append((str(fpath), line_no, stripped))
    return endpoint_violations, import_violations


def test_no_trading_sdk_in_requirements():
    """Garante que nenhum SDK de execução de trade está listado em requirements*.txt."""
    repo_root = Path(__file__).resolve().parents[3]
    req_files = [repo_root / "requirements.txt", repo_root / "requirements-dev.txt"]

    found_sdks = []
    for req_file in req_files:
        if req_file.exists():
            content = req_file.read_text(encoding="utf-8").lower()
            for sdk in FORBIDDEN_SDKS:
                if sdk in content:
                    found_sdks.append((req_file.name, sdk))

    assert not found_sdks, f"SDKs de trading proibidos detectados em requirements: {found_sdks}"


def test_no_live_order_endpoints_or_sdk_imports_in_production_code():
    """Garante inexistência de chamadas a endpoints privados de ordem e imports de SDK em código operacional."""
    repo_root = Path(__file__).resolve().parents[3]

    # Checa main.py isoladamente
    main_file = repo_root / "main.py"
    if main_file.exists():
        content = main_file.read_text(encoding="utf-8", errors="strict")
        for line_no, line in enumerate(content.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for endpoint in FORBIDDEN_ORDER_ENDPOINTS:
                assert endpoint not in line, f"Endpoint proibido em main.py:{line_no} -> {endpoint}"
            for pat in FORBIDDEN_IMPORT_PATTERNS:
                assert not pat.search(line), f"Import proibido em main.py:{line_no} -> {stripped}"

    # Checa todos os diretórios operacionais
    all_ep_violations = []
    all_imp_violations = []
    for op_dir in OPERATIONAL_DIRS:
        target_path = repo_root / op_dir
        ep_v, imp_v = _scan_for_forbidden_patterns(target_path)
        all_ep_violations.extend(ep_v)
        all_imp_violations.extend(imp_v)

    assert not all_ep_violations, (
        "Endpoints privados de envio de ordem detectados no código operacional:\n"
        + "\n".join(f"  {path}:{line} -> {endpoint}" for path, line, endpoint in all_ep_violations)
    )
    assert not all_imp_violations, (
        "Imports de SDK de trading proibidos detectados no código operacional:\n"
        + "\n".join(f"  {path}:{line} -> {imp}" for path, line, imp in all_imp_violations)
    )


def test_verify_safe_mode_diagnostic_is_explicitly_excluded():
    """Confirma que o script forense scripts/diagnostics/verify_safe_mode.py está excluído por caminho relativo explícito."""
    assert "scripts/diagnostics/verify_safe_mode.py" in EXCLUDED_OPERATIONAL_RELATIVE_PATHS


def test_scanner_detects_synthetic_forbidden_endpoint(tmp_path):
    """
    Exemplo sintético 1: comprova formalmente que o scanner detecta e acusa
    violação caso um endpoint de ordem privada seja adicionado ao código.
    """
    bad_module = tmp_path / "bad_executor.py"
    bad_module.write_text(
        'def send_order():\n'
        '    endpoint = "/api/v3/order"\n'
        '    return endpoint\n',
        encoding="utf-8",
    )

    ep_v, imp_v = _scan_for_forbidden_patterns(tmp_path)
    assert len(ep_v) == 1, f"Deveria detectar 1 violação de endpoint, obteve: {ep_v}"
    assert ep_v[0][2] == "/api/v3/order"
    assert len(imp_v) == 0


def test_scanner_detects_synthetic_forbidden_sdk_import(tmp_path):
    """
    Exemplo sintético 2: comprova formalmente que o scanner detecta e acusa
    violação caso um import de SDK de trade seja introduzido.
    """
    bad_module = tmp_path / "bad_connector.py"
    bad_module.write_text(
        'import ccxt\n'
        'from binance.client import Client\n'
        'def get_client():\n'
        '    return ccxt.binance()\n',
        encoding="utf-8",
    )

    ep_v, imp_v = _scan_for_forbidden_patterns(tmp_path)
    assert len(imp_v) == 2, f"Deveria detectar 2 violações de import, obteve: {imp_v}"
    imported_texts = [v[2] for v in imp_v]
    assert any("import ccxt" in t for t in imported_texts)
    assert any("from binance.client import Client" in t for t in imported_texts)
    assert len(ep_v) == 0


def test_scanner_fails_on_unreadable_file(tmp_path):
    """
    Exemplo sintético 3: comprova formalmente que um arquivo que não pode ser lido
    (por erro de I/O ou permissão) NÃO é ignorado silenciosamente e aciona falha imediata.
    """
    fake_module = tmp_path / "locked_module.py"
    fake_module.write_text("print('test')", encoding="utf-8")

    with patch.object(Path, "read_text", side_effect=OSError("Simulated I/O Permission Error")):
        with pytest.raises(OSError, match="Simulated I/O Permission Error"):
            _scan_for_forbidden_patterns(tmp_path)


def test_scanner_fails_when_directory_is_missing(tmp_path):
    """
    Exemplo sintético 4: comprova que se um diretório obrigatório não existir,
    o scanner falha imediatamente com FileNotFoundError (fail-closed).
    """
    non_existent = tmp_path / "missing_dir"
    with pytest.raises(FileNotFoundError, match="não encontrado para auditoria"):
        _scan_for_forbidden_patterns(non_existent)


def test_scanner_fails_on_directory_walk_access_error(tmp_path):
    """
    Exemplo sintético 5: comprova que erro de permissão/acesso durante o os.walk
    não é ignorado e aciona falha imediata (fail-closed).
    """
    def mock_walk_error(target, onerror=None):
        if onerror:
            onerror(PermissionError(13, "Acesso negado simulado", str(target)))
        return iter([])

    with patch("os.walk", side_effect=mock_walk_error):
        with pytest.raises(OSError, match="Falha de permissão ou I/O"):
            _scan_for_forbidden_patterns(tmp_path)


def test_scanner_exclusion_requires_relative_path_not_basename(tmp_path):
    """
    Comprova que a exclusão exige caminho relativo exato.
    Se um arquivo com mesmo nome base 'verify_safe_mode.py' for criado fora de
    scripts/diagnostics/ (ex: em market_orchestrator), ele NÃO pode ser ignorado.
    """
    fake_orch = tmp_path / "market_orchestrator"
    fake_orch.mkdir(parents=True)
    fake_file = fake_orch / "verify_safe_mode.py"
    fake_file.write_text('ORDER_URL = "/api/v3/order"\\n', encoding="utf-8")

    ep_v, _ = _scan_for_forbidden_patterns(fake_orch, repo_root=tmp_path)
    assert len(ep_v) == 1, "Arquivo verify_safe_mode.py fora de scripts/diagnostics foi ignorado indevidamente por basename!"
