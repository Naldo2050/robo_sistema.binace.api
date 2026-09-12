# tests/unit/test_orderbook_market_impact_contract.py
"""
Auditoria e Validação do Contrato Matemático Canônico de Market Impact.
Casos cobertos:
  A) BUY 100% preenchível (distinção entre VWAP slippage e terminal move)
  B) SELL 100% preenchível (distinção entre VWAP slippage e terminal move)
  C) BUY 50% preenchível (insuficiente -> full requested é None, observados preservados)
  D) SELL 50% preenchível (insuficiente -> full requested é None, observados preservados)
  E) Profundidade zero (fail-closed, todos os valores observados são None, nunca 0.0)
  F) Ordem exatamente igual à profundidade disponível
  G) Teste multinível canônico: $200 USD contra 2 níveis ($100 @ 101 e $100 @ 103)
  H) Caso Real Auditado: $1M SELL com profundidade top50 de ~$483k (fill 48.31%)
  I) Integração com market_orchestrator (slippage_matrix com VWAP execution slippage)
  J) Invariantes matemáticos fundamentais
"""

import pytest
from orderbook_analyzer.core import _simulate_market_impact
from market_orchestrator.market_orchestrator import EnhancedMarketBot


def test_case_a_buy_100_percent():
    """Caso A: BUY 100% preenchível em 1 nível."""
    asks = [(100.0, 2.0), (101.0, 5.0)]
    mid = 99.5
    amt = 200.0

    res = _simulate_market_impact(asks, amt, side="buy", mid=mid)
    assert res["insufficient_liquidity"] is False
    assert res["full_fill"] is True
    assert res["fill_ratio"] == 1.0
    assert res["usd_filled"] == 200.0

    # No preenchimento em nível único, VWAP = terminal_price = 100.0
    assert res["execution_vwap"] == 100.0
    assert res["execution_slippage_usd"] == 0.5
    assert res["terminal_price"] == 100.0
    assert res["terminal_move_usd"] == 0.5

    # Aliases legados
    assert res["move_usd"] == 0.5
    assert res["observed_move_usd"] == 0.5
    assert res["final_price"] == 100.0
    assert res["vwap"] == 100.0
    assert res["bps"] == pytest.approx((0.5 / 99.5) * 10000.0, rel=1e-4)


def test_case_b_sell_100_percent():
    """Caso B: SELL 100% preenchível em 1 nível."""
    bids = [(100.0, 2.0), (99.0, 5.0)]
    mid = 100.5
    amt = 200.0

    res = _simulate_market_impact(bids, amt, side="sell", mid=mid)
    assert res["insufficient_liquidity"] is False
    assert res["full_fill"] is True
    assert res["fill_ratio"] == 1.0
    assert res["usd_filled"] == 200.0

    assert res["execution_vwap"] == 100.0
    assert res["execution_slippage_usd"] == 0.5
    assert res["terminal_price"] == 100.0
    assert res["terminal_move_usd"] == 0.5

    # Aliases legados
    assert res["move_usd"] == 0.5
    assert res["observed_move_usd"] == 0.5
    assert res["final_price"] == 100.0
    assert res["vwap"] == 100.0
    assert res["bps"] == pytest.approx((0.5 / 100.5) * 10000.0, rel=1e-4)


def test_case_c_buy_50_percent_insufficient():
    """Caso C: BUY 50% preenchível (insuficiência de liquidez)."""
    asks = [(100.0, 1.0)]
    mid = 99.0
    amt = 200.0

    res = _simulate_market_impact(asks, amt, side="buy", mid=mid)
    assert res["insufficient_liquidity"] is True
    assert res["full_fill"] is False
    assert res["fill_ratio"] == 0.5
    assert res["usd_filled"] == 100.0

    # Full requested é desconhecido -> None
    assert res["execution_vwap"] is None
    assert res["execution_slippage_usd"] is None
    assert res["execution_slippage_bps"] is None
    assert res["terminal_price"] is None
    assert res["terminal_move_usd"] is None
    assert res["terminal_move_bps"] is None
    assert res["move_usd"] is None
    assert res["bps"] is None

    # Parcela observada é preservada
    assert res["observed_execution_vwap"] == 100.0
    assert res["observed_execution_slippage_usd"] == 1.0
    assert res["observed_terminal_price"] == 100.0
    assert res["observed_terminal_move_usd"] == 1.0
    assert res["observed_move_usd"] == 1.0
    assert res["observed_bps"] == pytest.approx((1.0 / 99.0) * 10000.0, rel=1e-4)


def test_case_d_sell_50_percent_insufficient():
    """Caso D: SELL 50% preenchível (insuficiência de liquidez)."""
    bids = [(100.0, 1.0)]
    mid = 101.0
    amt = 200.0

    res = _simulate_market_impact(bids, amt, side="sell", mid=mid)
    assert res["insufficient_liquidity"] is True
    assert res["full_fill"] is False
    assert res["fill_ratio"] == 0.5
    assert res["usd_filled"] == 100.0

    # Full requested é desconhecido -> None
    assert res["execution_vwap"] is None
    assert res["execution_slippage_usd"] is None
    assert res["execution_slippage_bps"] is None
    assert res["terminal_price"] is None
    assert res["terminal_move_usd"] is None
    assert res["terminal_move_bps"] is None
    assert res["move_usd"] is None
    assert res["bps"] is None

    # Parcela observada é preservada
    assert res["observed_execution_vwap"] == 100.0
    assert res["observed_execution_slippage_usd"] == 1.0
    assert res["observed_terminal_price"] == 100.0
    assert res["observed_terminal_move_usd"] == 1.0
    assert res["observed_move_usd"] == 1.0
    assert res["observed_bps"] == pytest.approx((1.0 / 101.0) * 10000.0, rel=1e-4)


def test_case_e_zero_depth():
    """Caso E: profundidade zero (nenhum zero falso)."""
    res = _simulate_market_impact([], 1000.0, side="buy", mid=100.0)
    assert res["insufficient_liquidity"] is True
    assert res["full_fill"] is False
    assert res["fill_ratio"] == 0.0
    assert res["usd_filled"] == 0.0
    assert res["levels"] == 0

    # Todos os valores de slippage e move devem ser None, NUNCA 0.0
    assert res["execution_vwap"] is None
    assert res["execution_slippage_usd"] is None
    assert res["execution_slippage_bps"] is None
    assert res["terminal_price"] is None
    assert res["terminal_move_usd"] is None
    assert res["terminal_move_bps"] is None
    assert res["observed_execution_vwap"] is None
    assert res["observed_execution_slippage_usd"] is None
    assert res["observed_execution_slippage_bps"] is None
    assert res["observed_terminal_price"] is None
    assert res["observed_terminal_move_usd"] is None
    assert res["observed_terminal_move_bps"] is None
    assert res["move_usd"] is None
    assert res["observed_move_usd"] is None
    assert res["bps"] is None
    assert res["observed_bps"] is None


def test_case_f_exact_depth_match():
    """Caso F: ordem exatamente igual à profundidade disponível."""
    levels = [(100.0, 1.0), (100.0, 2.0)]
    amt = 300.0
    mid = 99.0

    res = _simulate_market_impact(levels, amt, side="buy", mid=mid)
    assert res["insufficient_liquidity"] is False
    assert res["full_fill"] is True
    assert res["fill_ratio"] == 1.0
    assert res["usd_filled"] == 300.0
    assert res["execution_slippage_usd"] == 1.0
    assert res["terminal_move_usd"] == 1.0
    assert res["move_usd"] == 1.0
    assert res["observed_move_usd"] == 1.0


def test_case_g_multilevel_canonical_buy_and_sell():
    """
    Caso G (Item 12): Teste multinível canônico.
    BUY:
      mid = 100.0
      asks: $100 @ 101.0 (qty=100/101), $100 @ 103.0 (qty=100/103)
      requested = $200.0
      Esperado:
        terminal_price = 103.0
        terminal_move = 3.0 USD
        VWAP ≈ 101.9902 USD
        execution_slippage ≈ 1.9902 USD (NÃO 3.0!)
    SELL:
      mid = 100.0
      bids: $100 @ 99.0 (qty=100/99), $100 @ 97.0 (qty=100/97)
      requested = $200.0
      Esperado:
        terminal_price = 97.0
        terminal_move = 3.0 USD
        VWAP ≈ 97.9898 USD
        execution_slippage ≈ 2.0102 USD (NÃO 3.0!)
    """
    mid = 100.0

    # BUY
    asks = [(101.0, 100.0 / 101.0), (103.0, 100.0 / 103.0)]
    res_b = _simulate_market_impact(asks, 200.0, side="buy", mid=mid)
    assert res_b["full_fill"] is True
    assert res_b["terminal_price"] == 103.0
    assert res_b["terminal_move_usd"] == 3.0
    assert res_b["execution_vwap"] == pytest.approx(101.9902, abs=1e-4)
    assert res_b["execution_slippage_usd"] == pytest.approx(1.9902, abs=1e-4)
    # Garante que slippage != terminal move
    assert res_b["execution_slippage_usd"] != res_b["terminal_move_usd"]

    # SELL
    bids = [(99.0, 100.0 / 99.0), (97.0, 100.0 / 97.0)]
    res_s = _simulate_market_impact(bids, 200.0, side="sell", mid=mid)
    assert res_s["full_fill"] is True
    assert res_s["terminal_price"] == 97.0
    assert res_s["terminal_move_usd"] == 3.0
    assert res_s["execution_vwap"] == pytest.approx(97.9898, abs=1e-4)
    assert res_s["execution_slippage_usd"] == pytest.approx(2.0102, abs=1e-4)
    # Garante que slippage != terminal move
    assert res_s["execution_slippage_usd"] != res_s["terminal_move_usd"]


def test_case_h_real_1m_audit_reproduction():
    """
    Caso H (Item 14): Reprodução fiel do caso real de $1M SELL auditado.
    Top 50 bids com volume total de $483,081.91.
    Mid = 77283.05, último nível top50 = 77276.10.
    Esperado:
      fill_ratio ≈ 0.4831
      execution_slippage_usd = None
      terminal_move_usd = None
      observed_execution_slippage_usd ≈ 2.31 USD
      observed_terminal_move_usd ≈ 6.95 USD
    """
    p1, q1 = 77280.0, 5.0
    p2, q2 = 77276.10, (483081.91 - 5.0 * 77280.0) / 77276.10
    levels = [(p1, q1), (p2, q2)]
    mid = 77283.05
    requested = 1_000_000.0

    res = _simulate_market_impact(levels, requested, side="sell", mid=mid)

    assert res["insufficient_liquidity"] is True
    assert res["full_fill"] is False
    assert res["fill_ratio"] == pytest.approx(0.4831, abs=1e-3)
    assert res["usd_filled"] == pytest.approx(483081.91, abs=1.0)
    assert res["execution_slippage_usd"] is None
    assert res["terminal_move_usd"] is None
    assert res["move_usd"] is None
    assert res["bps"] is None
    assert res["observed_terminal_move_usd"] == pytest.approx(6.95, abs=1e-2)
    assert res["observed_execution_slippage_usd"] == pytest.approx(3.83, abs=1e-2)


def test_case_i_market_orchestrator_matrix_contract():
    """
    Caso I: Validar que MarketOrchestrator preenche slippage_matrix
    com VWAP EXECUTION SLIPPAGE, cria terminal_move_matrix separada,
    e trata partial fill como None na matriz de slippage.
    """
    # Ordem de 100k full com slippage ínfimo (bps=1.0 -> liq_score=9.8 >= 8)
    # Ordem de 1M partial (apenas ~$100k disponível -> fill_ratio=0.10)
    mid = 100.0
    asks = [(100.01, 1000.0)] # total depth = $100,010
    mi_100k = _simulate_market_impact(asks, 100_000.0, side="buy", mid=mid) # full fill
    mi_1m = _simulate_market_impact(asks, 1_000_000.0, side="buy", mid=mid)   # partial fill (~0.10)

    ob_event = {
        "is_valid": True,
        "bids": [(99.0, 10.0)],
        "asks": asks,
        "market_impact_buy": {
            "100k": mi_100k,
            "1M": mi_1m,
        },
        "market_impact_sell": {
            "100k": mi_100k,
            "1M": mi_1m,
        },
        "orderbook_data_quality": {"data_source": "live"}
    }

    orch = EnhancedMarketBot.__new__(EnhancedMarketBot)
    orch.institutional_analytics = None
    signal = {}
    orch._enrich_orderbook_metrics(signal, ob_event)

    mi = signal["market_impact"]
    # 1. slippage_matrix deve conter VWAP execution slippage para full fill
    assert mi["slippage_matrix"]["100k_usd"]["buy"] == pytest.approx(mi_100k["execution_slippage_usd"])
    # 2. slippage_matrix deve ser None para 1M partial
    assert mi["slippage_matrix"]["1m_usd"]["buy"] is None
    # 3. terminal_move_matrix deve existir separada
    assert mi["terminal_move_matrix"]["100k_usd"]["buy"] == pytest.approx(mi_100k["terminal_move_usd"])
    assert mi["terminal_move_matrix"]["1m_usd"]["buy"] is None
    # 4. observed_partial_matrix deve conter o slippage observado
    assert mi["observed_partial_matrix"]["1m_usd"]["buy"] == pytest.approx(mi_1m["observed_execution_slippage_usd"])
    # 5. execution_quality deve sinalizar PARTIAL_1M
    assert mi["execution_quality"] == "PARTIAL_1M"


def test_invariants():
    """Validação dos invariantes matemáticos fundamentais."""
    levels = [(100.0, 1.0), (101.0, 2.0), (102.0, 3.0)]
    mid = 100.0

    for requested in [50.0, 100.0, 300.0, 608.0, 1000.0]:
        res = _simulate_market_impact(levels, requested, side="buy", mid=mid)
        assert 0.0 <= res["fill_ratio"] <= 1.0
        assert res["usd_filled"] <= requested + 1e-6
        if res["full_fill"]:
            assert res["fill_ratio"] == 1.0
            assert res["insufficient_liquidity"] is False
            assert res["execution_slippage_usd"] is not None
            assert res["terminal_move_usd"] is not None
            # Terminal move é sempre >= VWAP slippage em orderbooks monotonically increasing
            assert res["terminal_move_usd"] >= res["execution_slippage_usd"] - 1e-6
        else:
            assert res["fill_ratio"] < 1.0
            assert res["insufficient_liquidity"] is True
            assert res["execution_slippage_usd"] is None
            assert res["terminal_move_usd"] is None
            assert res["observed_execution_slippage_usd"] is not None
            assert res["observed_terminal_move_usd"] is not None


def test_fallback_zero_vs_none_compatibility():
    """
    Sanity pré-commit: validação rigorosa de fallback 0.0 vs None vs Chave Ausente.
    A) Schema novo: execution_slippage_usd=0.0, move_usd=5.0 -> resultado: 0.0
    B) Schema novo partial: execution_slippage_usd=None, move_usd=5.0 -> resultado: None (NUNCA terminal move)
    C) Schema legado: sem chave execution_slippage_usd, move_usd=5.0 -> resultado: 5.0
    D) Schema novo full: execution_slippage_usd=2.0, move_usd=5.0 -> resultado: 2.0
    E) Schema novo zero liquidity: execution_slippage_usd=None, observed_execution_slippage_usd=None -> resultado: None
    """
    from market_orchestrator.market_orchestrator import EnhancedMarketBot

    # Caso A: Schema novo zero real (slippage é legitimamente 0.0)
    sig_a = {}
    ob_a = {"is_valid": True, "market_impact_buy": {"100k": {"execution_slippage_usd": 0.0, "move_usd": 5.0}}}
    EnhancedMarketBot._enrich_orderbook_metrics(sig_a, ob_a)
    assert sig_a["market_impact"]["slippage_matrix"]["100k_usd"]["buy"] == 0.0

    # Caso B: Schema novo partial fill (slippage total = None, move_usd pode ser residual ou None)
    sig_b = {}
    ob_b = {"is_valid": True, "market_impact_buy": {"100k": {"execution_slippage_usd": None, "move_usd": 5.0, "observed_execution_slippage_usd": 2.5}}}
    EnhancedMarketBot._enrich_orderbook_metrics(sig_b, ob_b)
    # NÃO pode acionar fallback para move_usd=5.0! Deve ser None!
    assert sig_b["market_impact"]["slippage_matrix"]["100k_usd"]["buy"] is None
    assert sig_b["market_impact"]["observed_partial_matrix"]["100k_usd"]["buy"] == 2.5

    # Caso C: Schema legado (chave execution_slippage_usd ausente, apenas move_usd existe)
    sig_c = {}
    ob_c = {"is_valid": True, "market_impact_buy": {"100k": {"move_usd": 5.0}}}
    EnhancedMarketBot._enrich_orderbook_metrics(sig_c, ob_c)
    # Fallback legado acionado corretamente
    assert sig_c["market_impact"]["slippage_matrix"]["100k_usd"]["buy"] == 5.0

    # Caso D: Schema novo full fill (execution_slippage_usd=2.0, move_usd=5.0)
    sig_d = {}
    ob_d = {"is_valid": True, "market_impact_buy": {"100k": {"execution_slippage_usd": 2.0, "move_usd": 5.0}}}
    EnhancedMarketBot._enrich_orderbook_metrics(sig_d, ob_d)
    assert sig_d["market_impact"]["slippage_matrix"]["100k_usd"]["buy"] == 2.0

    # Caso E: Schema novo zero liquidity (livro vazio: tudo None)
    sig_e = {}
    ob_e = {"is_valid": True, "market_impact_buy": {"100k": {"execution_slippage_usd": None, "observed_execution_slippage_usd": None, "move_usd": None}}}
    EnhancedMarketBot._enrich_orderbook_metrics(sig_e, ob_e)
    assert sig_e["market_impact"]["slippage_matrix"]["100k_usd"]["buy"] is None
    assert sig_e["market_impact"]["observed_partial_matrix"]["100k_usd"]["buy"] is None
