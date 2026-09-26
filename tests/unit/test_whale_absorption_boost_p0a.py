# tests/unit/test_whale_absorption_boost_p0a.py — P0-A (histórico) -> P0-A2 (contrato vigente)
#
# HISTÓRICO PRESERVADO (não apagar — evidência do bug anterior):
# - Pré P0-A (bug comprovado pelos testes abaixo falhando antes do fix):
#     STRONG + "Absorção de Compra" => boost +8 (bullish em evento bearish) [ERRADO]
#     STRONG + "Absorção de Venda"  => boost -8 (bearish em evento bullish) [ERRADO]
#     J2 (8.6/1.4/Compra/STRONG): base (8.6-1.4)*3=+21.6 +8 = 29.6 => min(25)=25.0
# - P0-A (correção do boost, arquivo então vigente):
#     Compra/BUY => -8; Venda/SELL => +8.
#     J2: 21.6 - 8 = +13.6 (ainda bullish em evento bearish — motivou P0-A2).
#     Provas: test_canonical_*_boost (net=0 isolava o boost), test_j2 13.6.
# - P0-A2 (contrato vigente NESTE arquivo): fail-closed NON-VOTING.
#     components.absorption.score == 0.0 sempre; direção vira só evidência
#     explicativa (canonical_direction), nunca número no total.
#
# Contrato canônico (referência):
# - flow_analyzer/absorption.py:10-16
#     Venda = SELL absorvido por BUY = bullish; Compra = BUY absorvido por SELL = bearish
# - data_processing/data_handler.py:1160-1169 (side espelhado)
#
# P0-A2 NÃO altera: flow, depth, derivatives, cvd, thresholds, classification,
# bias, alert thresholds, pesos máximos. NÃO converte direção em número.

import pytest

from flow_analyzer.whale_score import (
    WhaleAccumulationCalculator,
    _canonical_absorption_direction,
)


def _absorption_component(*, buyer_strength, seller_exhaustion, label,
                          classification, index=0.5):
    calc = WhaleAccumulationCalculator()
    out = calc.calculate(
        sector_flow={},
        orderbook_data={},
        absorption_data={
            "current_absorption": {
                "buyer_strength": buyer_strength,
                "seller_exhaustion": seller_exhaustion,
                "index": index,
                "classification": classification,
                "label": label,
            }
        },
        derivatives_data={},
        onchain_data={},
        cvd=0.0,
    )
    return out["components"]["absorption"]


def _full_result(*, sector_flow=None, orderbook_data=None, absorption_kwargs=None):
    calc = WhaleAccumulationCalculator()
    return calc.calculate(
        sector_flow=sector_flow or {},
        orderbook_data=orderbook_data or {},
        absorption_data=(
            {"current_absorption": absorption_kwargs} if absorption_kwargs else None
        ),
        derivatives_data={},
        onchain_data={},
        cvd=0.0,
    )


# ── Mapeamento canônico unitário ─────────────────────────────────────────────

@pytest.mark.parametrize("label", [
    "Absorção de Compra",
    "absorção de compra",
    "COMPRA",
    "BUY ABSORPTION",
    "buy",
])
def test_canonical_direction_compra_is_bearish(label):
    assert _canonical_absorption_direction(label) == "BEARISH"


@pytest.mark.parametrize("label", [
    "Absorção de Venda",
    "absorção de venda",
    "VENDA",
    "SELL ABSORPTION",
    "sell",
])
def test_canonical_direction_venda_is_bullish(label):
    assert _canonical_absorption_direction(label) == "BULLISH"


@pytest.mark.parametrize("label", ["Neutra", "NEUTRAL", "", None, "UNKNOWN", "ruído"])
def test_canonical_direction_neutral_unknown(label):
    assert _canonical_absorption_direction(label) == "NEUTRAL"


# ── Propriedades P0-A2: score sempre 0, direção só explicativa ───────────────

@pytest.mark.parametrize("buyer,seller,classification", [
    (8.6, 1.4, "STRONG_ABSORPTION"),
    (10.0, 0.0, "STRONG_ABSORPTION"),
    (0.0, 10.0, "STRONG_ABSORPTION"),
    (5.0, 5.0, "STRONG_ABSORPTION"),
    (8.6, 1.4, "MODERATE_ABSORPTION"),
    (8.6, 1.4, "WEAK_ABSORPTION"),
    (8.6, 1.4, "NONE"),
    (5.9, 1.8, "NONE"),
])
def test_compra_always_non_voting_bearish(buyer, seller, classification):
    comp = _absorption_component(
        buyer_strength=buyer, seller_exhaustion=seller,
        label="Absorção de Compra", classification=classification,
    )
    assert comp["score"] == 0.0
    assert -25 <= comp["score"] <= 25
    assert comp["max"] == 25
    assert comp["status"] == "NON_VOTING_UNVALIDATED_MAGNITUDE"
    assert comp["canonical_direction"] == "BEARISH"
    assert comp["detail"]["canonical_direction"] == "BEARISH"
    # Evidência preservada + legado só-diagnóstico:
    assert comp["detail"]["buyer_strength"] == buyer
    assert comp["detail"]["seller_exhaustion"] == seller
    assert comp["detail"]["label"] == "Absorção de Compra"
    assert comp["detail"]["legacy_unvalidated_metric"] is True


@pytest.mark.parametrize("buyer,seller,classification", [
    (1.4, 1.4, "STRONG_ABSORPTION"),
    (0.0, 10.0, "STRONG_ABSORPTION"),
    (10.0, 0.0, "STRONG_ABSORPTION"),
    (5.0, 5.0, "STRONG_ABSORPTION"),
    (1.4, 8.6, "MODERATE_ABSORPTION"),
    (5.0, 5.0, "NONE"),
])
def test_venda_always_non_voting_bullish(buyer, seller, classification):
    comp = _absorption_component(
        buyer_strength=buyer, seller_exhaustion=seller,
        label="Absorção de Venda", classification=classification,
    )
    assert comp["score"] == 0.0
    assert -25 <= comp["score"] <= 25
    assert comp["status"] == "NON_VOTING_UNVALIDATED_MAGNITUDE"
    assert comp["canonical_direction"] == "BULLISH"
    assert comp["detail"]["canonical_direction"] == "BULLISH"
    assert comp["detail"]["legacy_unvalidated_metric"] is True


@pytest.mark.parametrize("label", ["Neutra", "NEUTRAL", "", "UNKNOWN", "ruído"])
@pytest.mark.parametrize("buyer,seller", [(5.0, 5.0), (5.9, 1.8), (8.6, 1.4)])
def test_neutral_unknown_always_non_voting_neutral(label, buyer, seller):
    comp = _absorption_component(
        buyer_strength=buyer, seller_exhaustion=seller,
        label=label, classification="NONE",
    )
    assert comp["score"] == 0.0
    assert comp["status"] == "NON_VOTING_UNVALIDATED_MAGNITUDE"
    assert comp["canonical_direction"] == "NEUTRAL"


def test_absent_absorption_is_non_voting_neutral():
    calc = WhaleAccumulationCalculator()
    out = calc.calculate(sector_flow={}, orderbook_data={},
                         absorption_data=None, derivatives_data={},
                         onchain_data={}, cvd=0.0)
    comp = out["components"]["absorption"]
    assert comp["score"] == 0.0
    assert comp["max"] == 25
    assert comp["status"] == "NON_VOTING_UNVALIDATED_MAGNITUDE"
    assert comp["canonical_direction"] == "NEUTRAL"


def test_legacy_net_kept_only_for_diagnostics():
    # J2: net legado (8.6-1.4)=7.2 ainda calculado p/ diagnóstico, mas score=0.
    comp = _absorption_component(
        buyer_strength=8.6, seller_exhaustion=1.4,
        label="Absorção de Compra", classification="STRONG_ABSORPTION",
    )
    assert comp["detail"]["net_absorption"] == 7.2
    assert comp["detail"]["legacy_unvalidated_metric"] is True
    assert comp["score"] == 0.0


# ── J2 obrigatório: decomposição do total ────────────────────────────────────

def test_j2_absorption_non_voting_and_total_decomposition():
    # J2 observada: flow=+30 (whale_delta grande, clamp), depth≈-17.09 (ASK-heavy),
    # absorption Compra 8.6/1.4 (bearish por contrato).
    # Depth -17.09 => ratio -0.8545 => ask/bid ≈ 12.7 (ex: bid 100k, ask 1.2745M).
    absorption = {
        "buyer_strength": 8.6,
        "seller_exhaustion": 1.4,
        "index": 0.5062,
        "classification": "STRONG_ABSORPTION",
        "label": "Absorção de Compra",
    }
    out = _full_result(
        sector_flow={"whale": {"delta": 75.171}, "mid": {"delta": 0.0},
                     "retail": {"delta": 0.0}},
        orderbook_data={"bid_depth_usd": 100000.0, "ask_depth_usd": 1274500.0},
        absorption_kwargs=absorption,
    )
    comp = out["components"]["absorption"]
    assert comp["score"] == 0.0
    assert comp["canonical_direction"] == "BEARISH"
    assert comp["status"] == "NON_VOTING_UNVALIDATED_MAGNITUDE"

    flow_s = out["components"]["flow"]["score"]
    depth_s = out["components"]["depth"]["score"]
    deriv_s = out["components"]["derivatives"]["score"]
    assert flow_s == 30.0  # clamp de 75.171*10
    assert depth_s == pytest.approx(-17.09, abs=0.05)
    # Total = flow+depth+derivatives+0 (absorption não vota). Prova matemática,
    # sem hardcodar 13/14: o valor exato segue dos inputs de depth/deriv.
    assert out["score"] == round(flow_s + depth_s + deriv_s + comp["score"])
    assert out["score"] == pytest.approx(30.0 + depth_s, abs=0.1)
    # Documentação da diferença vs evento observado (~35 com absorption votando):
    # antes (P0-A, STRONG): 30 - 17.09 + 13.6 ≈ 26~27 (ou ~35 com base 21.6 MOD);
    # após P0-A2: ≈ 12.9~13 (só flow+depth+deriv). Queda ≈ comp antigo.
