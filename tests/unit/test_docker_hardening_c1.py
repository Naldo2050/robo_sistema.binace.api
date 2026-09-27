# tests/unit/test_docker_hardening_c1.py
# -*- coding: utf-8 -*-
"""
Testes de Cloud Hardening C1 para Docker, Timezone e Modelos ML:
1. Contrato ML_AVAILABLE / ML_MISSING no MLInferenceEngine.
2. Comportamento fail-closed se modelo ausente (sem fabricar inferência).
3. Auditoria estática de Dockerfile e docker-compose.yml (UID/GID, volumes persistentes, TZ=UTC).
"""

from pathlib import Path
import pytest

from ml.inference_engine import MLInferenceEngine


def test_ml_inference_engine_missing_model_contract(tmp_path, monkeypatch):
    """Garante que quando o modelo não existe, retorna ML_MISSING e prob neutra sem fabricar inferência."""
    import config

    empty_dir = tmp_path / "empty_models"
    empty_dir.mkdir()

    engine = MLInferenceEngine(model_dir=str(empty_dir))

    assert engine.model is None
    assert engine.ml_status == "ML_MISSING"

    # Caso 1: HYBRID_ENABLED=True -> status="ML_MISSING"
    monkeypatch.setattr("config.settings.HYBRID_ENABLED", True)
    result = engine.predict({"price": 65000.0, "volume": 10.0})
    assert result["ml_status"] == "ML_MISSING"
    assert result["status"] == "ML_MISSING"
    assert result["prob_up"] == 0.5
    assert result["signal"] == "neutral"

    # Caso 2: HYBRID_ENABLED=False -> status="hybrid_disabled" mas ml_status="ML_MISSING"
    monkeypatch.setattr("config.settings.HYBRID_ENABLED", False)
    result_disabled = engine.predict({"price": 65000.0, "volume": 10.0})
    assert result_disabled["ml_status"] == "ML_MISSING"
    assert result_disabled["status"] == "hybrid_disabled"


def test_dockerfile_hardening_contract():
    """Valida requisitos estritos de segurança e configuração no Dockerfile."""
    dockerfile_path = Path("Dockerfile")
    assert dockerfile_path.exists(), "Dockerfile não encontrado"

    content = dockerfile_path.read_text(encoding="utf-8")

    # Timezone UTC
    assert "TZ=UTC" in content, "Dockerfile deve fixar TZ=UTC"

    # UID/GID parametrizáveis
    assert "ARG APP_UID=1000" in content, "Dockerfile deve conter ARG APP_UID=1000"
    assert "ARG APP_GID=1000" in content, "Dockerfile deve conter ARG APP_GID=1000"
    assert "useradd -u ${APP_UID}" in content or "useradd" in content, "Dockerfile deve associar APP_UID ao trader"

    # Diretórios persistentes
    assert "mkdir -p dados logs features ml/models" in content, "Dockerfile deve preparar ml/models e diretórios de dados"

    # Usuário non-root
    assert "USER trader" in content, "Dockerfile deve rodar como USER trader"


def test_docker_compose_hardening_contract():
    """Valida mapeamento de volumes e variáveis no docker-compose.yml."""
    compose_path = Path("docker-compose.yml")
    assert compose_path.exists(), "docker-compose.yml não encontrado"

    content = compose_path.read_text(encoding="utf-8")

    # Timezone UTC
    assert "TZ=UTC" in content, "docker-compose deve definir TZ=UTC no ambiente"

    # Volumes obrigatórios
    assert "./dados:/app/dados" in content, "docker-compose deve montar ./dados"
    assert "./logs:/app/logs" in content, "docker-compose deve montar ./logs"
    assert "./features:/app/features" in content, "docker-compose deve montar ./features"
    assert "/app/ml/models:ro" in content, "docker-compose deve montar ml/models como :ro (read-only)"

    # Segurança: .env NUNCA deve ser montado como volume no container
    assert "./.env:" not in content and ".env:/app" not in content, "NÃO montar .env como volume!"

    # Args de build
    assert "APP_UID: ${APP_UID:-1000}" in content, "docker-compose deve passar APP_UID aos build args"
    assert "APP_GID: ${APP_GID:-1000}" in content, "docker-compose deve passar APP_GID aos build args"
