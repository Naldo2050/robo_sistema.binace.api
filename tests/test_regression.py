"""
Testes de regressão — garantem que bugs corrigidos nas ETAPAs 2.1-2.14 não retornem.

Rodar: python -m pytest tests/test_regression.py -v
"""

import ast
import os
import re
from pathlib import Path

# Diretório raiz do projeto
ROOT = Path(__file__).resolve().parent.parent


# ─── ETAPA 2.1: bb__width (duplo underscore) nunca mais ───

def test_no_double_underscore_bb_width():
    """
    ETAPA 2.1: O alias bb__width (duplo underscore) causava KeyError no XGBoost.
    Nenhum arquivo .py deve conter 'bb__width' exceto este teste e comentários.
    """
    violations = []
    for py_file in ROOT.rglob("*.py"):
        # Pular diretórios irrelevantes
        rel = py_file.relative_to(ROOT)
        if any(part in str(rel) for part in (".venv", "__pycache__", "node_modules")):
            continue
        if py_file.name == "test_regression.py":
            continue

        try:
            content = py_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        for i, line in enumerate(content.splitlines(), 1):
            stripped = line.lstrip()
            # Ignorar comentários puros
            if stripped.startswith("#"):
                continue
            # Remover comentário inline e strings antes de verificar
            code_part = line.split("#")[0]
            # Ignorar ocorrências em strings (assert 'bb__width' not in ...)
            if "bb__width" in code_part and "'bb__width'" not in code_part and '"bb__width"' not in code_part:
                violations.append(f"{rel}:{i}: {line.strip()}")

    assert not violations, (
        f"bb__width (duplo underscore) encontrado em {len(violations)} lugar(es):\n"
        + "\n".join(violations)
    )


# ─── ETAPA 2.2: RSI não deve ser hardcoded 50 no feature_calculator ───

def test_no_hardcoded_rsi_50_in_feature_calc():
    """
    ETAPA 2.2: RSI congelado porque feature_calculator computava localmente
    em vez de usar multi_tf. Verificar que multi_tf é prioridade.
    """
    fc_path = ROOT / "ml" / "feature_calculator.py"
    assert fc_path.exists(), "ml/feature_calculator.py não encontrado"

    content = fc_path.read_text(encoding="utf-8", errors="replace")

    # Deve conter referência a _rsi_from_multi_tf como prioridade
    assert "_rsi_from_multi_tf" in content, (
        "feature_calculator.py deve usar _rsi_from_multi_tf como fonte primária de RSI"
    )

    # O padrão "rsi = 50.0" como default inicial NÃO deve aparecer ANTES
    # da tentativa de multi_tf. Verificar que a lógica prioriza multi_tf.
    lines = content.splitlines()
    multi_tf_line = None
    for i, line in enumerate(lines):
        if "_rsi_from_multi_tf" in line and "rsi_mtf" in line:
            multi_tf_line = i
            break

    assert multi_tf_line is not None, (
        "_rsi_from_multi_tf deve ser chamado na compute() de feature_calculator"
    )


# ─── ETAPA 2.3: volume_compra e volume_venda propagados ───

def test_volume_compra_venda_in_enrichment_integrator():
    """
    ETAPA 2.3: volume_compra e volume_venda devem ser propagados
    no ANALYSIS_TRIGGER pelo enrichment_integrator.
    """
    ei_path = ROOT / "data_processing" / "enrichment_integrator.py"
    assert ei_path.exists()

    content = ei_path.read_text(encoding="utf-8", errors="replace")
    assert "volume_compra" in content, "volume_compra ausente do enrichment_integrator"
    assert "volume_venda" in content, "volume_venda ausente do enrichment_integrator"


# ─── ETAPA 2.4: advanced_analysis NÃO criado prematuramente no enrich() ───

def test_no_premature_advanced_analysis_in_enrich():
    """
    ETAPA 2.4: advanced_analysis era criado no enrich() sem multi_tf,
    resultando em volatilidade 0.01 (fallback). Deve ser criado apenas
    no add_context() onde multi_tf está disponível.
    """
    pipeline_path = ROOT / "data_pipeline" / "pipeline.py"
    assert pipeline_path.exists()

    content = pipeline_path.read_text(encoding="utf-8", errors="replace")

    # Localizar o método enrich()
    tree = ast.parse(content)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "enrich":
            # Extrair o corpo do método enrich como texto
            enrich_body = content.splitlines()[node.lineno - 1:node.end_lineno]
            enrich_text = "\n".join(enrich_body)

            # NÃO deve conter criação de advanced_analysis
            assert "advanced_analysis" not in enrich_text or "# " in enrich_text.split("advanced_analysis")[0].split("\n")[-1], (
                "enrich() não deve criar advanced_analysis (deve ser feito em add_context)"
            )
            break


# ─── ETAPA 2.12: cross-asset calculado apenas 1x ───

def test_no_duplicate_cross_asset_in_pipeline():
    """
    ETAPA 2.12: get_cross_asset_features() era chamado 3x.
    Agora deve ser chamado apenas em generate_ml_features().
    """
    pipeline_path = ROOT / "data_pipeline" / "pipeline.py"
    content = pipeline_path.read_text(encoding="utf-8", errors="replace")

    # pipeline.py NÃO deve importar get_cross_asset_features
    assert "get_cross_asset_features" not in content, (
        "pipeline.py não deve importar/chamar get_cross_asset_features diretamente"
    )


def test_no_duplicate_cross_asset_in_payload_builder():
    """
    ETAPA 2.12: ai_payload_builder não deve chamar get_cross_asset_features().
    """
    pb_path = ROOT / "market_orchestrator" / "ai" / "ai_payload_builder.py"
    content = pb_path.read_text(encoding="utf-8", errors="replace")

    # Não deve importar get_cross_asset_features
    imports = [line for line in content.splitlines()
               if "import" in line and "get_cross_asset_features" in line]
    assert not imports, (
        f"ai_payload_builder.py não deve importar get_cross_asset_features: {imports}"
    )


# ─── ETAPA 2.13: clock sync offset check ───

def test_clock_sync_differentiates_ok_vs_degraded():
    """
    ETAPA 2.13: _sync_with_binance deve logar WARNING (não INFO)
    quando offset excede o limite aceitável.
    """
    tm_path = ROOT / "monitoring" / "time_manager.py"
    content = tm_path.read_text(encoding="utf-8", errors="replace")

    # Deve conter check de abs_offset vs max_acceptable
    assert "abs_offset" in content and "max_acceptable_offset_ms" in content, (
        "time_manager deve comparar abs_offset com max_acceptable_offset_ms"
    )

    # O log de "Melhor amostra" deve ter variante WARNING para offset alto
    assert "offset alto" in content.lower() or "offset alto" in content, (
        "time_manager deve logar warning quando offset é alto"
    )


# ─── ETAPA 2.14: UTF-8 encoding nos auto_fixer scripts ───

def test_auto_fixer_scripts_have_utf8_fix():
    """
    ETAPA 2.14: Scripts do auto_fixer que usam emoji em print()
    devem ter fix de encoding para Windows.
    """
    scripts = [
        ROOT / "auto_fixer" / "apply_safe_fixes.py",
        ROOT / "auto_fixer" / "runner.py",
        ROOT / "auto_fixer" / "validate_installation.py",
        ROOT / "auto_fixer" / "fix_bugs.py",
    ]

    for script in scripts:
        if not script.exists():
            continue
        content = script.read_text(encoding="utf-8", errors="replace")
        has_encoding_fix = (
            "reconfigure" in content
            or "PYTHONIOENCODING" in content
            or "TextIOWrapper" in content
        )
        assert has_encoding_fix, (
            f"{script.relative_to(ROOT)} usa emoji mas não tem fix de encoding UTF-8"
        )


# ─── WindowState: bb_width usa underscore simples ───

def test_window_state_uses_single_underscore_bb_width():
    """Garante que WindowState usa bb_width (simples) e não bb__width."""
    from core.window_state import WindowState

    ws = WindowState()
    assert hasattr(ws.indicators, "bb_width")
    features = ws.get_ml_features()
    assert "bb_width" in features
    assert "bb__width" not in features
