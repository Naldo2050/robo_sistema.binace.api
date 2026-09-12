"""
tests/payload/test_p02_semantic_contract.py — Validação do Contrato Semântico P02

Testa rigorosamente:
1. Item 13: Renomeação semântica flow.trade_imb (+0.81) e ob.depth_imb (-0.69),
   sem chaves ambíguas antigas (flow.imb, ob.imb, ob.t5).
2. Item 14: Aceitação de sinais opostos (trade_imb > 0 / depth_imb < 0 e vice-versa)
   sem erro e sem forçar absorção espúria.
3. Item 15: Omissão de trade_bar_flow quando institutional score estiver ausente
   (ausência de sinal permanece ausência; sem fallback de trade flow_imbalance 1m).
4. Item 16: Preservação de trade_bar_flow quando institucional real estiver presente,
   confirmando independência de flow.trade_imb.
5. Item 12: Preservação das novas chaves através do compressor e guardrail.
"""

import pytest
from market_orchestrator.ai.payload_builder_compact import build_compact_payload
from market_orchestrator.ai.payload_compressor import compress_payload
from market_orchestrator.ai.llm_payload_guardrail import ensure_safe_llm_payload


def _make_base_event() -> dict:
    return {
        "timestamp": 1741400000.0,
        "symbol": "BTCUSDT",
        "tipo_evento": "ANALISE_TECNICA",
        "descricao": "Teste Contrato Semântico P02",
        "ativo": "BTCUSDT",
        "preco_atual": 65000.0,
        "fechamento": 65000.0,
        "abertura": 64900.0,
        "maxima": 65100.0,
        "minima": 64800.0,
        "vwap": 64950.0,
        "fluxo_continuo": {
            "order_flow": {
                "net_flow_1m": 50000.0,
                "net_flow_5m": 120000.0,
                "net_flow_15m": 300000.0,
                "flow_imbalance": 0.81,
                "buy_sell_ratio": {"buy_sell_ratio": 9.5},
            },
            "cvd": 12.5,
        },
        "orderbook_data": {
            "bid_depth_usd": 1550000.0,
            "ask_depth_usd": 8450000.0,
            "flow_imbalance": -0.69,
            "imbalance": -0.69,
        },
        "order_book_depth": {
            "L5": {
                "flow_imbalance": -0.75,
                "imbalance": -0.75,
            }
        },
        "timeframes": {
            "1m": {"rsi": 62.0, "macd": {"macd": 10.0, "signal": 5.0}, "adx": 25.0, "atr": 45.0, "trend": "UP"},
        },
        "suporte_resistencia": {
            "nearest_support": {"price": 64500.0, "strength": 75},
            "nearest_resistance": {"price": 65500.0, "strength": 80},
        },
        "regime": {"volatility": "M", "trend": "UP", "sentiment": "BULL"},
    }


def test_p02_item13_semantic_keys_and_values():
    """
    Item 13: Evento com trade imbalance = +0.81 e book depth imbalance = -0.69.
    Resultado FINAL deve conter:
      flow.trade_imb = +0.81
      ob.depth_imb = -0.69
      ob.depth_t5 = -0.75
    E NÃO:
      flow.imb
      ob.imb
      ob.t5
    """
    event = _make_base_event()
    compact = build_compact_payload(event)

    # Chaves de Flow
    assert "flow" in compact
    flow = compact["flow"]
    assert "trade_imb" in flow, "flow.trade_imb deve estar presente no payload final"
    assert flow["trade_imb"] == 0.81, "Valor de trade_imb deve ser exatamente +0.81"
    assert "imb" not in flow, "Chave ambígua flow.imb não deve existir no payload gerado"

    # Chaves de Orderbook
    assert "ob" in compact
    ob = compact["ob"]
    assert "depth_imb" in ob, "ob.depth_imb deve estar presente no payload final"
    assert ob["depth_imb"] == -0.69, "Valor de depth_imb deve ser exatamente -0.69"
    assert "depth_t5" in ob, "ob.depth_t5 deve estar presente no payload final"
    assert ob["depth_t5"] == -0.75, "Valor de depth_t5 deve ser exatamente -0.75"
    assert "imb" not in ob, "Chave ambígua ob.imb não deve existir no payload gerado"
    assert "t5" not in ob, "Chave ambígua ob.t5 não deve existir no payload gerado"


def test_p02_item14_opposing_signals_no_forced_error_or_absorption():
    """
    Item 14: Sinais opostos (trade_imb > 0 com depth_imb < 0, e trade_imb < 0 com depth_imb > 0)
    devem ser aceitos normalmente pelo pipeline sem erro e sem atribuir absorção automaticamente.
    """
    # Cenário A: Taker comprador (+0.70) com book ask-heavy (-0.60)
    event_a = _make_base_event()
    event_a["fluxo_continuo"]["order_flow"]["flow_imbalance"] = 0.70
    event_a["orderbook_data"]["flow_imbalance"] = -0.60
    compact_a = build_compact_payload(event_a)

    assert compact_a["flow"]["trade_imb"] == 0.70
    assert compact_a["ob"]["depth_imb"] == -0.60
    # Não força tipo absorção no summary se os indicadores de absorção não estiverem disparados
    flow_sum_a = compact_a.get("summary", {}).get("flow", {})
    assert flow_sum_a.get("type") != "absorption", "Divergência isolada não deve ser taxada de absorção"

    # Cenário B: Taker vendedor (-0.65) com book bid-heavy (+0.55)
    event_b = _make_base_event()
    event_b["fluxo_continuo"]["order_flow"]["flow_imbalance"] = -0.65
    event_b["orderbook_data"]["flow_imbalance"] = 0.55
    compact_b = build_compact_payload(event_b)

    assert compact_b["flow"]["trade_imb"] == -0.65
    assert compact_b["ob"]["depth_imb"] == 0.55
    flow_sum_b = compact_b.get("summary", {}).get("flow", {})
    assert flow_sum_b.get("type") != "absorption", "Divergência isolada não deve ser taxada de absorção"


def test_p02_item15_absent_institutional_omits_section():
    """
    Item 15: Quando institutional score não estiver disponível:
    - trade_bar_flow deve ser OMITIDO do payload final.
    - flow.trade_imb deve continuar presente.
    - Nunca deve copiar flow_imbalance 1m para trade_bar_flow.
    """
    event = _make_base_event()
    # Garantir ausência total de institutional_analytics.order_flow_imbalance
    event.pop("institutional_analytics", None)

    compact = build_compact_payload(event)

    assert "trade_bar_flow" not in compact, "trade_bar_flow deve ser omitido quando ausente"
    assert "ofi" not in compact, "ofi não deve ser emitido via fallback espúrio"
    assert "trade_imb" in compact["flow"], "flow.trade_imb deve permanecer presente"
    assert compact["flow"]["trade_imb"] == 0.81


def test_p02_item16_real_institutional_score_preserved():
    """
    Item 16: Quando score institucional existir:
    - trade_bar_flow deve conter score e dir reais.
    - Deve ser independente de flow.trade_imb.
    """
    event = _make_base_event()
    event["institutional_analytics"] = {
        "order_flow_imbalance": {
            "score": -0.42,
            "direction": "SELL",
            "bars_evaluated": 5,
        }
    }
    event["fluxo_continuo"]["order_flow"]["flow_imbalance"] = 0.81  # Valor diferente

    compact = build_compact_payload(event)

    assert "trade_bar_flow" in compact, "trade_bar_flow deve estar presente com dados reais"
    tbf = compact["trade_bar_flow"]
    assert tbf["score"] == -0.42, "Score institucional deve ser preservado exatamente"
    assert tbf["dir"] == "SELL"
    # Provar independência numérica
    assert tbf["score"] != compact["flow"]["trade_imb"], "trade_bar_flow.score não deve ser cópia de flow.trade_imb"


def test_p02_pipeline_guardrail_and_compressor_survival():
    """
    Item 12: Garantir que flow.trade_imb, ob.depth_imb, ob.depth_t5 e trade_bar_flow
    sobrevivem ao compressor e ao guardrail sem serem eliminados.
    """
    event = _make_base_event()
    event["institutional_analytics"] = {
        "order_flow_imbalance": {
            "score": 0.35,
            "direction": "BUY",
        }
    }

    compact = build_compact_payload(event)
    compressed = compress_payload(compact, max_bytes=6144)
    guarded = ensure_safe_llm_payload(compressed)

    assert "trade_imb" in guarded["flow"], "flow.trade_imb deve sobreviver ao guardrail"
    assert guarded["flow"]["trade_imb"] == 0.81

    assert "depth_imb" in guarded["ob"], "ob.depth_imb deve sobreviver ao guardrail"
    assert guarded["ob"]["depth_imb"] == -0.69

    assert "depth_t5" in guarded["ob"], "ob.depth_t5 deve sobreviver ao guardrail"
    assert guarded["ob"]["depth_t5"] == -0.75

    assert "trade_bar_flow" in guarded, "trade_bar_flow deve sobreviver ao guardrail"
    assert guarded["trade_bar_flow"]["score"] == 0.35
    assert guarded["trade_bar_flow"]["dir"] == "BUY"


def test_p02_sanity_check_no_new_ofi():
    """
    Sanity Check Exigido:
    Garantir que build_compact_payload -> compress_payload -> guardrail
    NUNCA gera chave raiz 'ofi' para nenhum payload NOVO, em nenhum dos 4 cenários:
      A) institutional score presente
      B) institutional score ausente
      C) flow.trade_imb presente (com valor alto)
      D) ML flow imbalance presente
    """
    # Cenário A: Institutional presente
    event_a = _make_base_event()
    event_a["institutional_analytics"] = {
        "order_flow_imbalance": {"score": 0.44, "direction": "BUY"}
    }
    res_a = ensure_safe_llm_payload(compress_payload(build_compact_payload(event_a), max_bytes=6144))
    assert "ofi" not in res_a, "Cenário A: ofi NUNCA deve ser gerado pelo builder novo"
    assert "trade_bar_flow" in res_a, "Cenário A: trade_bar_flow deve estar presente com dados reais"
    assert res_a["trade_bar_flow"]["score"] == 0.44

    # Cenário B: Institutional ausente
    event_b = _make_base_event()
    event_b.pop("institutional_analytics", None)
    res_b = ensure_safe_llm_payload(compress_payload(build_compact_payload(event_b), max_bytes=6144))
    assert "ofi" not in res_b, "Cenário B: ofi NUNCA deve ser gerado pelo builder novo"
    assert "trade_bar_flow" not in res_b, "Cenário B: trade_bar_flow deve ser OMITIDO"

    # Cenário C: Apenas flow.trade_imb presente (com valor forte)
    event_c = _make_base_event()
    event_c.pop("institutional_analytics", None)
    event_c["fluxo_continuo"]["order_flow"]["flow_imbalance"] = 0.95
    res_c = ensure_safe_llm_payload(compress_payload(build_compact_payload(event_c), max_bytes=6144))
    assert "ofi" not in res_c, "Cenário C: ofi NUNCA deve ser gerado pelo builder novo"
    assert "trade_bar_flow" not in res_c, "Cenário C: trade_bar_flow deve ser OMITIDO (sem cópia de trade_imb)"
    assert res_c["flow"]["trade_imb"] == 0.95

    # Cenário D: Apenas ML microstructure flow imbalance presente
    event_d = _make_base_event()
    event_d.pop("institutional_analytics", None)
    event_d["fluxo_continuo"]["order_flow"].pop("flow_imbalance", None)
    event_d["ml_features"] = {"microstructure": {"flow_imbalance": -0.77}}
    res_d = ensure_safe_llm_payload(compress_payload(build_compact_payload(event_d), max_bytes=6144))
    assert "ofi" not in res_d, "Cenário D: ofi NUNCA deve ser gerado pelo builder novo"
    assert "trade_bar_flow" not in res_d, "Cenário D: trade_bar_flow deve ser OMITIDO (sem cópia de ML)"


def test_p02_legacy_ofi_compatibility_preserved():
    """
    Garante que payloads LEGADOS contendo 'ofi' ainda são aceitos
    pelo compressor e guardrail através da whitelist retrocompatível.
    """
    legacy_payload = {
        "symbol": "BTCUSDT",
        "epoch_ms": 1741400000000,
        "trigger": "AT",
        "price": {"c": 65000.0},
        "flow": {"d1": "+10K", "trade_imb": 0.2},
        "ob": {"b": "1M", "a": "1M", "depth_imb": 0.0},
        "ofi": {"score": 0.15, "dir": "BUY"},
    }
    compressed = compress_payload(legacy_payload, max_bytes=6144)
    assert "ofi" in compressed, "Compressor deve preservar ofi em payload legado"
    guarded = ensure_safe_llm_payload(compressed)
    assert "ofi" in guarded, "Guardrail deve aceitar ofi legado via whitelist"
    assert guarded["ofi"]["score"] == 0.15
