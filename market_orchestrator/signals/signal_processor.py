# signals/signal_processor.py
# -*- coding: utf-8 -*-

"""
Processamento de sinais do EnhancedMarketBot.

CÓPIA FIEL do método EnhancedMarketBot._process_signals do arquivo
market_orchestrator.py original, apenas movido para este módulo e
trocando `self` por `bot`.

Nenhuma lógica foi alterada.
"""

import logging
from typing import Any, Dict, List

import pandas as pd

import config
from data_pipeline import DataPipeline
from data_processing.enrichment_integrator import enrich_analysis_trigger_event, build_analysis_trigger_event

# imports opcionais, como no arquivo original
try:
    import support_resistance as _sr
    detect_support_resistance = getattr(_sr, "detect_support_resistance", None)
    defense_zones = getattr(_sr, "defense_zones", None)
except Exception:
    detect_support_resistance = None
    defense_zones = None

try:
    from market_analysis.pattern_recognition import recognize_patterns
except Exception:
    recognize_patterns = None


def process_signals(
    bot,
    signals: List[Dict[str, Any]],
    pipeline: DataPipeline,
    flow_metrics: Dict[str, Any],
    historical_profile: Dict[str, Any],
    macro_context: Dict[str, Any],
    ob_event: Dict[str, Any],
    enriched: Dict[str, Any],
    close_ms: int,
    total_buy_volume: float,
    total_sell_volume: float,
    valid_window_data: List[Dict[str, Any]],
) -> None:
    """
    Equivalente ao método EnhancedMarketBot._process_signals original.
    """

    # --------------------------------------------------------
    # Garante que exista pelo menos um ANALYSIS_TRIGGER
    # --------------------------------------------------------
    has_real_signal = any(
        s.get("tipo_evento") not in ("ANALYSIS_TRIGGER", "OrderBook")
        for s in signals
    )

    if not signals or not has_real_signal:
        # Evento de análise automática
        raw_event_data = {
            "delta": enriched.get("delta_fechamento", 0.0),
            "volume_total": enriched.get("volume_total", 0.0),
            "volume_compra": enriched.get("volume_compra", 0.0) or total_buy_volume,
            "volume_venda": enriched.get("volume_venda", 0.0) or total_sell_volume,
            "preco_fechamento": enriched.get("ohlc", {}).get("close", 0.0),
        }
        trigger_signal = build_analysis_trigger_event(bot.symbol, raw_event_data)
        
        # Adicionar campos específicos do contexto
        trigger_signal.update({
            "timestamp": bot.time_manager.now_utc_iso(timespec="seconds"),
            "epoch_ms": close_ms,
            "ml_features": pipeline.get_final_features().get("ml_features", {}),
            "orderbook_data": ob_event,
            "historical_vp": historical_profile,
            "multi_tf": macro_context.get("mtf_trends", {}),
            "data_context": "real_time",
        })

        if not signals:
            signals.append(trigger_signal)
        elif not has_real_signal:
            signals = [
                s
                for s in signals
                if s.get("tipo_evento") != "ANALYSIS_TRIGGER"
            ]
            signals.append(trigger_signal)

    # 🔧 ENRICHMENT: Adiciona análise avançada aos ANALYSIS_TRIGGER
    config_dict = {k: v for k, v in vars(config).items() if not k.startswith("__")}
    for signal in signals:
        if signal.get("tipo_evento") == "ANALYSIS_TRIGGER":
            signal = enrich_analysis_trigger_event(signal, config_dict)

    # --------------------------------------------------------
    # Log do heatmap de liquidez (igual ao original)
    # --------------------------------------------------------
    bot._log_liquidity_heatmap(flow_metrics)

    # --------------------------------------------------------
    # Features finais + persistência no FeatureStore
    # --------------------------------------------------------
    features = pipeline.get_final_features()
    bot.feature_store.save_features(
        window_id=str(close_ms),
        features=features,
    )

    ml_payload = features.get("ml_features", {}) or {}
    enriched_snapshot = features.get("enriched", {}) or {}
    contextual_snapshot = features.get("contextual", {}) or {}

    derivatives_context = macro_context.get("derivatives", {})

    # --------------------------------------------------------
    # PATTERN RECOGNITION & PRICE TARGETS
    # --------------------------------------------------------
    pattern_recognition_data: Dict[str, Any] = {}
    price_targets: Dict[str, Any] = {}

    if recognize_patterns is not None:
        try:
            current_ohlc = enriched_snapshot.get("ohlc") or {}
            bars: List[Dict[str, float]] = list(bot.pattern_ohlc_history)

            if current_ohlc:
                bars.append(
                    {
                        "open": float(current_ohlc.get("open", current_ohlc.get("close", 0.0))),
                        "high": float(current_ohlc.get("high", 0.0)),
                        "low": float(current_ohlc.get("low", 0.0)),
                        "close": float(current_ohlc.get("close", 0.0)),
                    }
                )

            if len(bars) >= 3:
                df_patterns = pd.DataFrame(bars)
                pattern_recognition_data = recognize_patterns(df_patterns) or {}

                last_price = float(current_ohlc.get("close", 0.0)) if current_ohlc else 0.0
                price_targets = bot._build_price_targets(
                    pattern_recognition_data,
                    last_price,
                )
        except Exception as e:
            logging.debug(f"Falha em pattern_recognition: {e}")

    # --------------------------------------------------------
    # SUPORTE / RESISTÊNCIA E DEFENSE ZONES
    # --------------------------------------------------------
    support_resistance: Dict[str, Any] = {}
    defense_zones_data: Dict[str, Any] = {}

    if detect_support_resistance is not None:
        try:
            price_series = (
                pipeline.df["p"]
                if hasattr(pipeline, "df") and pipeline.df is not None
                else None
            )
            if price_series is not None:
                support_resistance = detect_support_resistance(
                    price_series, num_levels=3
                )
                if defense_zones is not None:
                    defense_zones_data = defense_zones(
                        support_resistance
                    )
        except Exception as e:
            logging.debug(
                f"Falha ao calcular suporte/resistência: {e}"
            )

    # ─── Sincronizar dados ricos do sinal real → ANALYSIS_TRIGGER ─────
    # Fase 1 (pré-enrichment): multi_tf e historical_vp
    _real_sig = next(
        (s for s in signals
         if s.get("tipo_evento") not in ("ANALYSIS_TRIGGER", "OrderBook")),
        None,
    )
    for _s in signals:
        if _s.get("tipo_evento") != "ANALYSIS_TRIGGER":
            continue
        # multi_tf: preferir sinal real → macro_context, nunca deixar vazio se há dados
        _has_rich = lambda tf: isinstance(tf, dict) and any(
            isinstance(v, dict) and (v.get("rsi_short") or v.get("tendencia"))
            for v in tf.values()
        )
        if not _has_rich(_s.get("multi_tf", {})):
            if _real_sig and _has_rich(_real_sig.get("multi_tf", {})):
                _s["multi_tf"] = _real_sig["multi_tf"]
            else:
                _mtf = macro_context.get("mtf_trends", {})
                if _mtf:
                    _s["multi_tf"] = _mtf
        # historical_vp: garantir VP completo (com HVN/LVN nodes)
        if not _s.get("historical_vp") and _real_sig and _real_sig.get("historical_vp"):
            _s["historical_vp"] = _real_sig["historical_vp"]
        elif not _s.get("historical_vp") and historical_profile:
            _s["historical_vp"] = historical_profile
    # ──────────────────────────────────────────────────────────────────

    # --------------------------------------------------------
    # LOG DE SINAIS DA JANELA
    # --------------------------------------------------------
    logging.info(
        f"📊 JANELA #{bot.window_count} - "
        f"Processando {len(signals)} sinal(is):"
    )
    for i, sig in enumerate(signals, 1):
        logging.info(
            f"  {i}. {sig.get('tipo_evento')} / "
            f"{sig.get('resultado_da_batalha', 'N/A')} | "
            f"delta={sig.get('delta', 0):.2f} | "
            f"volume={sig.get('volume_total', 0):.2f}"
        )

    # --------------------------------------------------------
    # ENRIQUECIMENTO DE CADA SINAL
    # --------------------------------------------------------
    for signal in signals:
        if signal.get("is_signal", False):
            # Anexa patterns/targets no próprio sinal
            if pattern_recognition_data:
                signal["pattern_recognition"] = pattern_recognition_data
            if price_targets:
                signal["price_targets"] = price_targets

            bot._enrich_signal(
                signal,
                derivatives_context,
                flow_metrics,
                total_buy_volume,
                total_sell_volume,
                macro_context,
                close_ms,
                ml_payload,
                enriched_snapshot,
                contextual_snapshot,
                ob_event,
                valid_window_data,
                support_resistance,
                defense_zones_data,
            )

    # ─── Fase 2 (pós-enrichment): propagar dados que o institutional enricher
    # adiciona ao sinal real, mas pode falhar no ANALYSIS_TRIGGER por falta
    # de multi_tf rico no momento do cálculo.
    if _real_sig is not None:
        for _s in signals:
            if _s.get("tipo_evento") != "ANALYSIS_TRIGGER":
                continue
            # technical_indicators_extended (CCI, Stoch, Williams, GARCH)
            if not _s.get("technical_indicators_extended") and \
               _real_sig.get("technical_indicators_extended"):
                _s["technical_indicators_extended"] = _real_sig["technical_indicators_extended"]
            # alerts — substituir apenas se ANALYSIS_TRIGGER está sem alertas reais
            _at_sev = _s.get("alerts", {}).get("max_severity", "NONE")
            _re_sev = _real_sig.get("alerts", {}).get("max_severity", "NONE")
            if _at_sev in ("NONE", None) and _re_sev not in ("NONE", None):
                _s["alerts"] = _real_sig["alerts"]
            # institutional_analytics: sr_analysis e defense_zones
            _re_inst = _real_sig.get("institutional_analytics", {})
            if _re_inst:
                _at_inst = _s.setdefault("institutional_analytics", {})
                for _k in ("sr_analysis", "defense_zones", "sr_strength"):
                    if _k not in _at_inst and _k in _re_inst:
                        _at_inst[_k] = _re_inst[_k]
    # ──────────────────────────────────────────────────────────────────

    # --------------------------------------------------------
    # PÓS-PROCESSAMENTO (históricos, alerts, logs)
    # --------------------------------------------------------
    bot._check_zone_touches(enriched, signals)
    bot._update_histories(enriched, ml_payload)
    bot._log_ml_features(ml_payload)
    bot._process_institutional_alerts(enriched, pipeline)
    bot._log_health_check()
    bot._log_window_summary(enriched, historical_profile, macro_context)