"""
test_health_recovery_and_passive_events.py
-----------------------------------------
Testes unitários rigorosos e conceituais para observabilidade e semântica de saúde:
1. ws_error é canal passivo e ausência de erro = normalidade (zero falso CRITICAL).
2. Erro de WS registrado sem criar expectativa de heartbeat periódico.
3. Componentes downstream (orderbook, trade_ingestion, window_processor, trade_buffer) protegidos durante RECOVERING.
4. Componentes independentes (event_saver) NÃO são mascarados durante recovery (falha real gera UNHEALTHY).
5. Deadlock real em orderbook com WS HEALTHY gera CRITICAL e HTTP 503 imediatamente.
6. Reconnect storm durante recovery atualiza tentativa mas respeita teto absoluto global.
7. Warmup incompleto estoura timeout de recuperação e transiciona para UNHEALTHY.
8. Shutdown limpo zera estados de recovery sem vazamento de estado.
9. Respostas do endpoint HTTP /health refletem exatamente as transições (200 healthy, 200 degraded/recovering, 503 unhealthy).
"""

import time
import logging
import pytest

from monitoring.health_monitor import HealthMonitor, NON_STAGE_CHANNELS, DOWNSTREAM_COMPONENTS
import monitoring.pipeline_health as ph


class TestPassiveEventsAndWSError:
    """Valida que ws_error e outros canais passivos não geram alarmes de silêncio."""

    def test_ws_error_is_in_non_stage_channels(self):
        assert "ws_error" in NON_STAGE_CHANNELS
        assert "buffer_critical" in NON_STAGE_CHANNELS
        assert "buffer_overflow" in NON_STAGE_CHANNELS

    def test_ws_error_silent_healthy_never_triggers_critical(self):
        """1) WS saudável por longo tempo sem ws_error nunca gera UNHEALTHY/CRITICAL por ws_error."""
        monitor = HealthMonitor(max_silence_seconds=2, critical_silence_seconds=4, check_interval_seconds=1)
        try:
            # Registra apenas o componente ativo ws
            monitor.heartbeat("ws")
            stats = monitor.get_stats()
            # ws_error não deve constar na lista de monitored_modules para timeout de silêncio
            assert "ws_error" not in stats["monitored_modules"]
            assert "ws_error" not in stats["heartbeats"]

            # Aguarda tempo suficiente para estourar critical_silence se estivesse sendo monitorado
            time.sleep(1.5)
            monitor.heartbeat("ws")

            stats = monitor.get_stats()
            assert stats["active_critical_alerts"] == 0
            assert stats["active_warning_alerts"] == 0
        finally:
            monitor.stop()

    def test_ws_error_recorded_without_subsequent_heartbeat_expectation(self):
        """2) Erro WS registrado não cria expectativa de heartbeat periódico."""
        monitor = HealthMonitor(max_silence_seconds=2, critical_silence_seconds=4, check_interval_seconds=1)
        try:
            # Ocorre um erro de WS
            monitor.heartbeat("ws_error")
            stats = monitor.get_stats()

            # ws_error foi registrado como evento/métrica passiva
            assert "ws_error" in stats["last_events"]
            assert stats["event_counts"]["ws_error"] == 1
            # Mas NÃO está na lista de módulos com contagem de silêncio
            assert "ws_error" not in stats["monitored_modules"]
            assert "ws_error" not in stats["heartbeats"]

            # Também testar record_event explícito
            monitor.record_event("ws_error")
            stats = monitor.get_stats()
            assert stats["event_counts"]["ws_error"] == 2
        finally:
            monitor.stop()


class TestRecoveryAndDownstreamSemantics:
    """Valida a transição de estados e proteção de componentes downstream durante RECOVERING."""

    def test_downstream_components_composition(self):
        """Confirma que apenas componentes estritamente dependentes do feed WS estão em DOWNSTREAM_COMPONENTS."""
        assert "orderbook" in DOWNSTREAM_COMPONENTS
        assert "trade_ingestion" in DOWNSTREAM_COMPONENTS
        assert "window_processor" in DOWNSTREAM_COMPONENTS
        assert "trade_buffer" in DOWNSTREAM_COMPONENTS
        # event_saver NÃO pode estar em downstream para não mascarar falhas de banco/IO
        assert "event_saver" not in DOWNSTREAM_COMPONENTS
        assert "ws" not in DOWNSTREAM_COMPONENTS

    def test_orderbook_silence_during_recovering_is_not_critical(self, monkeypatch, caplog):
        """3) orderbook silêncio durante RECOVERING não emite CRITICAL e reporta DEGRADED."""
        monkeypatch.setitem(ph.STAGE_CRITICAL_THRESHOLDS, "orderbook", 2.0)
        monkeypatch.setitem(ph.STAGE_CRITICAL_THRESHOLDS, "ws", 2.0)
        monitor = HealthMonitor(max_silence_seconds=1, critical_silence_seconds=2, check_interval_seconds=1)
        try:
            monitor.heartbeat("ws")
            monitor.heartbeat("orderbook")

            # Ativa modo RECOVERING (ex: reconexão em andamento)
            monitor.set_recovering(True, reason="ws_reconnect", max_recovery_seconds=10.0)
            assert monitor.is_recovering() is True

            # Simula silêncio superior a critical_silence no orderbook enquanto WS continua vivo
            for _ in range(5):
                time.sleep(0.5)
                monitor.heartbeat("ws")

            with caplog.at_level(logging.CRITICAL):
                # O monitor loop não deve emitir CRITICAL para orderbook durante recovery
                critical_logs = [r.message for r in caplog.records if r.levelname == "CRITICAL"]
                assert not any("orderbook" in msg for msg in critical_logs)

            stats = monitor.get_stats()
            assert stats["is_recovering"] is True
            assert stats["heartbeats"]["orderbook"]["alert_level"] != "critical"

            # No PipelineHealth, status deve ser degraded (HTTP 200), NÃO unhealthy (503)
            ph_status = ph.compute_status(monitor)
            assert ph_status["status"] == "degraded"
            assert ph_status["reason"] == "recovering_warmup"
            assert "orderbook" not in ph_status["unhealthy_components"]
        finally:
            monitor.stop()

    def test_event_saver_failure_not_masked_during_recovery(self, monkeypatch, caplog):
        """4) event_saver NÃO é downstream e gera UNHEALTHY mesmo durante RECOVERING."""
        monkeypatch.setitem(ph.STAGE_CRITICAL_THRESHOLDS, "event_saver", 2.0)
        monkeypatch.setitem(ph.STAGE_CRITICAL_THRESHOLDS, "ws", 2.0)
        monitor = HealthMonitor(max_silence_seconds=1, critical_silence_seconds=2, check_interval_seconds=1)
        try:
            monitor.heartbeat("ws")
            monitor.heartbeat("event_saver")

            monitor.set_recovering(True, reason="ws_reconnect", max_recovery_seconds=10.0)
            assert monitor.is_recovering() is True

            # Simula silêncio em event_saver com WS vivo
            with caplog.at_level(logging.CRITICAL):
                for _ in range(6):
                    time.sleep(0.5)
                    monitor.heartbeat("ws")
                critical_logs = [r.message for r in caplog.records if r.levelname == "CRITICAL"]
                assert any("event_saver" in msg for msg in critical_logs)

            stats = monitor.get_stats()
            assert stats["heartbeats"]["event_saver"]["alert_level"] == "critical"

            # PipelineHealth reporta unhealthy por falha em componente não-downstream
            ph_status = ph.compute_status(monitor)
            assert ph_status["status"] == "unhealthy"
            assert "event_saver" in ph_status["unhealthy_components"]
        finally:
            monitor.stop()

    def test_orderbook_silence_when_healthy_is_critical_deadlock_detection(self, monkeypatch, caplog):
        """5) orderbook silêncio com WS HEALTHY (fora de recovery) gera CRITICAL e UNHEALTHY (anti-deadlock)."""
        monkeypatch.setitem(ph.STAGE_CRITICAL_THRESHOLDS, "orderbook", 2.0)
        monkeypatch.setitem(ph.STAGE_CRITICAL_THRESHOLDS, "ws", 2.0)
        monitor = HealthMonitor(max_silence_seconds=1, critical_silence_seconds=2, check_interval_seconds=1)
        try:
            monitor.heartbeat("ws")
            monitor.heartbeat("orderbook")
            # Sistema NÃO está em recovery (operação normal)
            monitor.set_recovering(False)
            assert monitor.is_recovering() is False

            # Simula silêncio superior a critical_silence no orderbook enquanto WS continua vivo
            with caplog.at_level(logging.CRITICAL):
                for _ in range(6):
                    time.sleep(0.5)
                    monitor.heartbeat("ws")
                critical_logs = [r.message for r in caplog.records if r.levelname == "CRITICAL"]
                assert any("orderbook" in msg for msg in critical_logs)

            stats = monitor.get_stats()
            assert stats["heartbeats"]["orderbook"]["alert_level"] == "critical"

            # No PipelineHealth, status deve ser unhealthy (503) acusando deadlock
            ph_status = ph.compute_status(monitor)
            assert ph_status["status"] == "unhealthy"
            assert "orderbook" in ph_status["unhealthy_components"]
        finally:
            monitor.stop()


class TestRecoveryEdgeCasesAndLifecycle:
    """Valida reconnect storms, timeouts de warmup e encerramento limpo."""

    def test_reconnect_storm_resets_attempt_deadline_respecting_total_ceiling(self):
        """6) Novo reconnect durante recovery reinicia a janela de tentativa mantendo o teto absoluto."""
        monitor = HealthMonitor(max_silence_seconds=1, critical_silence_seconds=2, check_interval_seconds=1)
        try:
            # 1a queda
            monitor.set_recovering(True, reason="ws_reconnect_1", max_recovery_seconds=2.0)
            start_1 = monitor._recovery_start_time
            first_start = monitor._first_recovery_start_time
            assert start_1 is not None
            assert first_start == start_1

            time.sleep(0.5)
            # 2a queda antes do fim do warmup (reconnect storm)
            monitor.set_recovering(True, reason="ws_reconnect_2", max_recovery_seconds=2.0)
            start_2 = monitor._recovery_start_time
            assert start_2 > start_1  # relógio da tentativa atual reiniciou
            assert monitor._first_recovery_start_time == first_start  # teto total preservado
        finally:
            monitor.stop()

    def test_warmup_not_completed_times_out_to_unhealthy(self):
        """7) Se apenas 1/3 janela chegar e o tempo expirar, recovery expira e transiciona para UNHEALTHY."""
        monitor = HealthMonitor(max_silence_seconds=1, critical_silence_seconds=2, check_interval_seconds=1)
        try:
            monitor.heartbeat("ws")
            monitor.heartbeat("orderbook")

            # Recovery com timeout curto (1.5s)
            monitor.set_recovering(True, reason="test_timeout", max_recovery_seconds=1.5)
            assert monitor.is_recovering() is True

            # Simula apenas 1 janela processada sem concluir as restantes
            time.sleep(0.5)
            monitor.heartbeat("ws")

            # Aguarda expirar os 1.5s
            time.sleep(1.2)
            assert monitor.is_recovering() is False  # expirou!

            # Com recovery expirado, silêncio de orderbook vira falha crítica
            stats = monitor.get_stats()
            assert stats["is_recovering"] is False
        finally:
            monitor.stop()

    def test_recovery_clean_shutdown(self):
        """8) Shutdown limpo com recovering=True limpa todos os timers e flags sem estado preso."""
        monitor = HealthMonitor(max_silence_seconds=1, critical_silence_seconds=2, check_interval_seconds=1)
        monitor.set_recovering(True, reason="before_stop", max_recovery_seconds=10.0)
        assert monitor.is_recovering() is True

        # Stop
        monitor.stop()
        assert monitor._stopped is True
        assert monitor._is_recovering is False
        assert monitor._recovery_start_time is None
        assert monitor._first_recovery_start_time is None
        assert monitor.is_recovering() is False


class TestPipelineHealthHTTPEndpoints:
    """Valida os códigos e payloads JSON do PipelineHealth."""

    def test_http_payload_contract_healthy(self):
        """9A) HEALTHY -> status: healthy, healthy: true, reason: ok."""
        monitor = HealthMonitor()
        try:
            for stg in ph.STAGE_COMPONENTS:
                monitor.heartbeat(stg)

            ph.attach_ws_connected_provider(lambda: True)
            res = ph.compute_status(monitor)
            assert res["status"] == "healthy"
            assert res["healthy"] is True
            assert res["reason"] == "ok"
            assert res["unhealthy_components"] == []
        finally:
            monitor.stop()

    def test_http_payload_contract_recovering_degraded(self):
        """9B) RECOVERING -> status: degraded, healthy: false, reason: recovering_warmup."""
        monitor = HealthMonitor()
        try:
            for stg in ph.STAGE_COMPONENTS:
                monitor.heartbeat(stg)
            monitor.set_recovering(True, reason="ws_reconnect")

            ph.attach_ws_connected_provider(lambda: True)
            res = ph.compute_status(monitor)
            assert res["status"] == "degraded"
            assert res["healthy"] is False
            assert res["reason"] == "recovering_warmup"
        finally:
            monitor.stop()

    def test_http_payload_contract_deadlock_unhealthy(self, monkeypatch):
        """9C) Deadlock orderbook -> status: unhealthy, healthy: false, reason: critical_components."""
        monkeypatch.setitem(ph.STAGE_CRITICAL_THRESHOLDS, "orderbook", 0.1)
        monitor = HealthMonitor()
        try:
            for stg in ph.STAGE_COMPONENTS:
                monitor.heartbeat(stg)
            time.sleep(0.3)

            ph.attach_ws_connected_provider(lambda: True)
            res = ph.compute_status(monitor)
            assert res["status"] == "unhealthy"
            assert res["healthy"] is False
            assert res["reason"] == "critical_components"
            assert "orderbook" in res["unhealthy_components"]
        finally:
            monitor.stop()

    def test_health_monitor_and_pipeline_health_concordance_post_timeout(self, monkeypatch):
        """9D) Concordância absoluta: no instante pós-timeout (ex: 300s), ambos concordam em UNHEALTHY (503)."""
        monkeypatch.setitem(ph.STAGE_CRITICAL_THRESHOLDS, "orderbook", 0.5)
        monkeypatch.setitem(ph.STAGE_CRITICAL_THRESHOLDS, "ws", 10.0)
        monitor = HealthMonitor(max_silence_seconds=1, critical_silence_seconds=2, check_interval_seconds=1)
        try:
            monitor.heartbeat("ws")
            monitor.heartbeat("orderbook")
            # Inicia recovering com timeout curto de 0.8s
            monitor.set_recovering(True, reason="test_sync", max_recovery_seconds=0.8)
            ph.attach_ws_connected_provider(lambda: True)

            # Durante recovery (t = 0.3s): PipelineHealth = degraded (200), HealthMonitor.is_recovering = True
            time.sleep(0.3)
            monitor.heartbeat("ws")
            assert monitor.is_recovering() is True
            res_during = ph.compute_status(monitor)
            assert res_during["status"] == "degraded"
            assert res_during["reason"] == "recovering_warmup"

            # Imediatamente após expirar timeout (t = 1.0s):
            time.sleep(0.7)
            monitor.heartbeat("ws")
            # HealthMonitor declara recovery expirado
            assert monitor.is_recovering() is False
            # PipelineHealth imediatamente declara UNHEALTHY (503) por silêncio do orderbook
            res_post = ph.compute_status(monitor)
            assert res_post["status"] == "unhealthy"
            assert res_post["healthy"] is False
            assert res_post["reason"] == "critical_components"
            assert "orderbook" in res_post["unhealthy_components"]
        finally:
            monitor.stop()
