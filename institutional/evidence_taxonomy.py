# institutional/evidence_taxonomy.py
"""
P1-B — Evidence Taxonomy + Lineage Map v1 (somente declaração, sem fiação).

- Registry declarativo de field_ids estáveis -> family/evidence_type/fonte/
  horizon/derived_from/calibration. Sem weight, TTL, half-life, family cap,
  confidence (proibidos nesta fase; teste trava).
- `derived_from` reflete o comportamento ATUAL (pós-P0): ex. whale.score NÃO
  lista absorption como dependência numérica (NON_VOTING P0-A2, só em notes);
  regime lista só o que realmente vota (defaults removidos P0-D2).
- `primary_ancestors()` = fecho transitivo de ancestrais primários para
  auditoria futura (NÃO deduplica, NÃO decide voto). Composites nunca entram
  no resultado (atravessa-se por eles); ids `raw.*` são terminais primários.
- Ciclo no registry = ValueError no import (fail-fast) + teste dedicado.
- Este módulo é folha (só stdlib + institutional.evidence). Sem ciclo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from institutional.evidence import (
    EvidenceCalibration,
    EvidenceFamily,
    EvidenceType,
)

M1 = 60_000
M5 = 300_000
M15 = 900_000
H1 = 3_600_000
H4 = 14_400_000
D1 = 86_400_000

CAL_UNCAL = EvidenceCalibration.UNCALIBRATED_HEURISTIC
CAL_NA = EvidenceCalibration.NOT_APPLICABLE
CAL_UNK = EvidenceCalibration.UNKNOWN

FAM_FLOW = EvidenceFamily.EXECUTED_FLOW
FAM_OB = EvidenceFamily.ORDERBOOK_SNAPSHOT
FAM_PRICE = EvidenceFamily.PRICE_RESPONSE
FAM_STRUCT = EvidenceFamily.MARKET_STRUCTURE
FAM_DERIV = EvidenceFamily.DERIVATIVES
FAM_CROSS = EvidenceFamily.CROSS_ASSET
FAM_MACRO = EvidenceFamily.MACRO
FAM_UNK = EvidenceFamily.UNKNOWN

T_TRADES = EvidenceType.CONTINUOUS_TRADES
T_L2 = EvidenceType.POINT_IN_TIME_L2
T_SLOW = EvidenceType.SLOW_CONTEXT
T_DERIVED = EvidenceType.DERIVED
T_UNK = EvidenceType.UNKNOWN


@dataclass(frozen=True)
class TaxonomyEntry:
    """Uma entrada declarativa. Sem weight/TTL/half-life/confidence por desenho."""
    field_id: str
    family: EvidenceFamily
    evidence_type: EvidenceType
    logical_source: str
    horizon_ms: Optional[int] = None
    derived_from: tuple = ()
    calibration: EvidenceCalibration = EvidenceCalibration.UNKNOWN
    is_composite: bool = False
    validity_source: Optional[str] = None
    notes: str = ""


def _e(field_id: str, family, evidence_type: str | EvidenceType,
        logical_source: str, horizon_ms=None, derived_from=(),
        calibration=CAL_UNK, is_composite: bool = False,
        validity_source=None, notes: str = "") -> TaxonomyEntry:
    return TaxonomyEntry(
        field_id=field_id, family=family,
        evidence_type=(EvidenceType(evidence_type)
                       if isinstance(evidence_type, str) else evidence_type),
        logical_source=logical_source, horizon_ms=horizon_ms,
        derived_from=tuple(derived_from), calibration=calibration,
        is_composite=is_composite, validity_source=validity_source,
        notes=notes)


_AGG = ("raw.aggtrade.buy_notional", "raw.aggtrade.sell_notional")
_AGG_Q = ("raw.aggtrade.buy_qty", "raw.aggtrade.sell_qty")
_L2 = ("raw.l2.bids", "raw.l2.asks")
_L2_DEPTH = ("raw.l2.bid_depth_usd", "raw.l2.ask_depth_usd")
_CDL = ("raw.candle.open", "raw.candle.high", "raw.candle.low",
        "raw.candle.close")

_ENTRIES = [
    # ── EXECUTED FLOW (flow_analyzer/order_flow; validity flow_window_integrity)
    _e("flow.net.1m", FAM_FLOW, T_TRADES, "flow_analyzer.order_flow", M1,
       _AGG, CAL_NA, validity_source="flow_window_integrity.1m"),
    _e("flow.net.5m", FAM_FLOW, T_TRADES, "flow_analyzer.order_flow", M5,
       _AGG, CAL_NA, validity_source="flow_window_integrity.5m"),
    _e("flow.net.15m", FAM_FLOW, T_TRADES, "flow_analyzer.order_flow", M15,
       _AGG, CAL_NA, validity_source="flow_window_integrity.15m"),
    _e("flow.imbalance.1m", FAM_FLOW, T_TRADES, "flow_analyzer.order_flow", M1,
       _AGG, CAL_NA, validity_source="flow_window_integrity.1m"),
    _e("flow.imbalance.5m", FAM_FLOW, T_TRADES, "flow_analyzer.order_flow", M5,
       _AGG, CAL_NA, validity_source="flow_window_integrity.5m"),
    _e("flow.imbalance.15m", FAM_FLOW, T_TRADES, "flow_analyzer.order_flow", M15,
       _AGG, CAL_NA, validity_source="flow_window_integrity.15m"),
    _e("flow.buy_sell_ratio", FAM_FLOW, T_TRADES, "flow_analyzer.order_flow",
       None, _AGG, CAL_NA,
       notes="janela de computação (usualmente 1m); bijeção com imbalance_1m"),
    _e("flow.aggressive_buy_pct.1m", FAM_FLOW, T_TRADES,
       "flow_analyzer.order_flow", M1, _AGG + _AGG_Q, CAL_NA,
       notes="somente quando observed (status/amostra)"),
    _e("flow.aggressive_sell_pct.1m", FAM_FLOW, T_TRADES,
       "flow_analyzer.order_flow", M1, _AGG + _AGG_Q, CAL_NA,
       notes="somente quando observed (status/amostra)"),
    _e("flow.delta", FAM_FLOW, T_TRADES, "flow_analyzer.order_flow",
       None, _AGG, CAL_NA, notes="janela do evento (variável)"),
    _e("flow.cvd", FAM_FLOW, T_TRADES, "flow_analyzer.order_flow",
       None, _AGG, CAL_NA,
       notes="acumulado com reset (até ~4h); NÃO é a janela 1m corrente"),
    _e("participants.whale.delta.window", FAM_FLOW, T_TRADES,
       "flow_analyzer.sector_flow", None,
       ("raw.aggtrade.whale_buy_qty", "raw.aggtrade.whale_sell_qty"), CAL_NA,
       notes="partição do delta total por tamanho (>=2 BTC); janela de computação"),
    _e("participants.mid.delta.window", FAM_FLOW, T_TRADES,
       "flow_analyzer.sector_flow", None,
       ("raw.aggtrade.mid_buy_qty", "raw.aggtrade.mid_sell_qty"), CAL_NA,
       notes="janela de computação"),
    _e("participants.retail.delta.window", FAM_FLOW, T_TRADES,
       "flow_analyzer.sector_flow", None,
       ("raw.aggtrade.retail_buy_qty", "raw.aggtrade.retail_sell_qty"), CAL_NA,
       notes="janela de computação"),
    _e("participants.whale.delta.4h", FAM_FLOW, T_TRADES,
       "flow_analyzer.sector_flow", H4,
       ("raw.aggtrade.whale_buy_qty", "raw.aggtrade.whale_sell_qty"), CAL_NA,
       notes="acumulado de sessão (compacto sf_w_4h)"),
    _e("participants.retail.delta.4h", FAM_FLOW, T_TRADES,
       "flow_analyzer.sector_flow", H4,
       ("raw.aggtrade.retail_buy_qty", "raw.aggtrade.retail_sell_qty"), CAL_NA,
       notes="acumulado de sessão (compacto sf_r_4h)"),
    _e("absorption.current", FAM_FLOW, T_TRADES,
       "flow_analyzer.absorption_analysis", None,
       ("flow.aggressive_buy_pct.1m", "flow.aggressive_sell_pct.1m",
        "flow.net.1m", "flow.imbalance.1m"), CAL_UNCAL, is_composite=True,
       validity_source="flow_window_integrity.1m",
       notes="índice/label buyer_strength; magnitude NÃO validada (P0-A2)"),
    _e("flow.trend", FAM_FLOW, T_TRADES, "flow_analyzer.order_flow",
       None, ("flow.imbalance.1m", "flow.imbalance.5m"), CAL_UNCAL,
       is_composite=True, validity_source="flow_window_integrity.1m+5m",
       notes="threshold 0.05 herdado; P0-B2 exige 1m&5m VALID p/ votar"),
    _e("flow.passive_aggressive.composite", FAM_FLOW, T_TRADES,
       "flow_analyzer.aggregates", None,
       ("flow.aggressive_buy_pct.1m", "flow.aggressive_sell_pct.1m",
        "orderbook.snapshot.bid_depth", "orderbook.snapshot.ask_depth"),
       CAL_UNCAL, is_composite=True,
       notes="cruza taker x book (fenômenos distintos por legenda)"),
    # ── ORDERBOOK SNAPSHOT (orderbook_analyzer; REST ~60s, sem decay P1-C)
    _e("orderbook.snapshot.bid_depth", FAM_OB, T_L2, "orderbook_analyzer.book",
       None, ("raw.l2.bid_depth_usd",), CAL_NA,
       notes="snapshot pontual; horizon=None NÃO significa infinito"),
    _e("orderbook.snapshot.ask_depth", FAM_OB, T_L2, "orderbook_analyzer.book",
       None, ("raw.l2.ask_depth_usd",), CAL_NA,
       notes="snapshot pontual; horizon=None NÃO significa infinito"),
    _e("orderbook.snapshot.mid", FAM_OB, T_L2, "orderbook_analyzer.book",
       None, ("raw.l2.mid",), CAL_NA),
    _e("orderbook.snapshot.spread", FAM_OB, T_L2, "orderbook_analyzer.book",
       None, ("raw.l2.spread",), CAL_NA),
    _e("orderbook.snapshot.imbalance", FAM_OB, T_L2, "orderbook_analyzer.book",
       None, ("orderbook.snapshot.bid_depth", "orderbook.snapshot.ask_depth"),
       CAL_NA, notes="(bid-ask)/(bid+ask)"),
    _e("orderbook.snapshot.pressure", FAM_OB, T_L2, "orderbook_analyzer.book",
       None, ("orderbook.snapshot.imbalance",), CAL_NA,
       notes="alias de imbalance (P0 audit D3)"),
    _e("orderbook.snapshot.volume_ratio", FAM_OB, T_L2,
       "orderbook_analyzer.book", None,
       ("orderbook.snapshot.bid_depth", "orderbook.snapshot.ask_depth"),
       CAL_NA, notes="bid/ask"),
    _e("orderbook.snapshot.depth_t5", FAM_OB, T_L2, "orderbook_analyzer.book",
       None, ("raw.l2.bids", "raw.l2.asks"), CAL_NA,
       notes="slice L5 (order_book_depth), não top50"),
    _e("orderbook.wall.bid", FAM_OB, T_L2, "orderbook_analyzer.walls",
       None, ("raw.l2.bids",), CAL_UNCAL,
       notes="detecção por quantil; snapshot_only"),
    _e("orderbook.wall.ask", FAM_OB, T_L2, "orderbook_analyzer.walls",
       None, ("raw.l2.asks",), CAL_UNCAL,
       notes="detecção por quantil; snapshot_only"),
    _e("market_impact.slippage.buy.100k", FAM_OB, T_L2,
       "orderbook_analyzer.market_impact", None,
       ("raw.l2.asks", "orderbook.snapshot.mid"), CAL_NA,
       notes="VWAP-slip USD vs mid (não terminal move)"),
    _e("market_impact.slippage.sell.100k", FAM_OB, T_L2,
       "orderbook_analyzer.market_impact", None,
       ("raw.l2.bids", "orderbook.snapshot.mid"), CAL_NA,
       notes="VWAP-slip USD vs mid (não terminal move)"),
    _e("market_impact.liquidity.score", FAM_OB, T_L2,
       "market_orchestrator.market_impact", None,
       ("raw.l2.bids", "raw.l2.asks"), CAL_UNCAL, is_composite=True,
       notes="10-avg_terminal_bps/5; média mascara assimetria (P0 audit §4)"),
    _e("market_impact.execution_quality", FAM_OB, T_L2,
       "market_orchestrator.market_impact", None,
       ("market_impact.liquidity.score",), CAL_UNCAL,
       notes="tier EXCEL/GOOD/FAIR/POOR/P1M/INSUF sobre o score"),
    _e("orderbook.iceberg.heuristic", FAM_OB, T_L2,
       "orderbook_analyzer.iceberg", None, ("raw.l2.bids", "raw.l2.asks"),
       CAL_UNCAL, validity_source="capabilities.ICEBERG_DETECTION_SUPPORTED",
       notes="2 snapshots; UNCONFIRMED/CONTINUOUS_L2_UNAVAILABLE (P0-C)"),
    _e("orderbook.bias_score", FAM_OB, T_L2, "orderbook_analyzer.bias",
       None, ("orderbook.snapshot.imbalance", "orderbook.snapshot.volume_ratio"),
       CAL_UNCAL, is_composite=True,
       notes="0.5+imb*0.3+ratio_adj*0.2; conta bid/ask 2x (P0 audit D3)"),
    _e("orderbook.bias", FAM_OB, T_L2, "payload_builder_compact.ob",
       None, ("orderbook.snapshot.imbalance",), CAL_UNCAL,
       notes="discretização BUY/SELL/NEUT em ±0.1"),
    # ── PRICE RESPONSE (fonte candle não cravada: trades vs klines)
    _e("price.close", FAM_PRICE, T_UNK, "contextual_snapshot.ohlc",
       None, ("raw.candle.close",), CAL_NA,
       notes="fonte candle não cravada no path (trades vs klines)"),
    _e("price.open", FAM_PRICE, T_UNK, "contextual_snapshot.ohlc",
       None, ("raw.candle.open",), CAL_NA,
       notes="fonte candle não cravada no path (trades vs klines)"),
    _e("price.high", FAM_PRICE, T_UNK, "contextual_snapshot.ohlc",
       None, ("raw.candle.high",), CAL_NA,
       notes="fonte candle não cravada no path (trades vs klines)"),
    _e("price.low", FAM_PRICE, T_UNK, "contextual_snapshot.ohlc",
       None, ("raw.candle.low",), CAL_NA,
       notes="fonte candle não cravada no path (trades vs klines)"),
    _e("price.vwap", FAM_PRICE, T_UNK, "contextual_snapshot.ohlc",
       None, ("raw.candle.close",), CAL_NA,
       notes="vwap do candle; TWAP existe no analyzer mas NÃO chega ao compacto"),
    # displacement/range/close-from-extreme: NÃO EXISTEM como campos (P0 §6).
    _e("tf.trend.15m", FAM_PRICE, T_SLOW, "technical_indicators.multi_tf",
       M15, _CDL, CAL_NA, notes="klines amostradas; sem TTL nesta fase"),
    _e("tf.trend.1h", FAM_PRICE, T_SLOW, "technical_indicators.multi_tf",
       H1, _CDL, CAL_NA, notes="klines amostradas; sem TTL nesta fase"),
    _e("tf.trend.4h", FAM_PRICE, T_SLOW, "technical_indicators.multi_tf",
       H1 * 4, _CDL, CAL_NA, notes="klines amostradas; sem TTL nesta fase"),
    _e("tf.trend.1d", FAM_PRICE, T_SLOW, "technical_indicators.multi_tf",
       D1, _CDL, CAL_NA, notes="klines amostradas; sem TTL nesta fase"),
    _e("tf.adx.15m", FAM_PRICE, T_SLOW, "technical_indicators.multi_tf",
       M15, _CDL, CAL_NA, notes="alimenta regime ADX; sem TTL nesta fase"),
    _e("tf.adx.1h", FAM_PRICE, T_SLOW, "technical_indicators.multi_tf",
       H1, _CDL, CAL_NA, notes="alimenta regime ADX; sem TTL nesta fase"),
    _e("tf.adx.4h", FAM_PRICE, T_SLOW, "technical_indicators.multi_tf",
       H1 * 4, _CDL, CAL_NA, notes="alimenta regime ADX; sem TTL nesta fase"),
    _e("tf.rsi.15m", FAM_PRICE, T_SLOW, "technical_indicators.multi_tf",
       M15, _CDL, CAL_NA, notes="só display LLM; sem TTL nesta fase"),
    _e("tf.rsi.1h", FAM_PRICE, T_SLOW, "technical_indicators.multi_tf",
       H1, _CDL, CAL_NA, notes="só display LLM; sem TTL nesta fase"),
    _e("tf.macd.15m", FAM_PRICE, T_SLOW, "technical_indicators.multi_tf",
       M15, _CDL, CAL_NA, notes="só display LLM; sem TTL nesta fase"),
    _e("tf.macd.1h", FAM_PRICE, T_SLOW, "technical_indicators.multi_tf",
       H1, _CDL, CAL_NA, notes="só display LLM; sem TTL nesta fase"),
    _e("tf.atr.15m", FAM_PRICE, T_SLOW, "technical_indicators.multi_tf",
       M15, _CDL, CAL_NA, notes="só display LLM; sem TTL nesta fase"),
    _e("tf.atr.1h", FAM_PRICE, T_SLOW, "technical_indicators.multi_tf",
       H1, _CDL, CAL_NA, notes="só display LLM; sem TTL nesta fase"),
    _e("tf.regime.15m", FAM_PRICE, T_SLOW, "technical_indicators.multi_tf",
       M15, ("tf.trend.15m", "tf.adx.15m"), CAL_UNCAL, is_composite=True,
       notes="RNG/ACC/TRD/MNP por timeframe; fórmula não rastreada em P1-B"),
    _e("tf.regime.1h", FAM_PRICE, T_SLOW, "technical_indicators.multi_tf",
       H1, ("tf.trend.1h", "tf.adx.1h"), CAL_UNCAL, is_composite=True,
       notes="RNG/ACC/TRD/MNP por timeframe; fórmula não rastreada em P1-B"),
    # ── MARKET STRUCTURE
    _e("profile.shape", FAM_STRUCT, T_TRADES,
       "institutional_analytics.profile_analysis", None,
       ("raw.trades.price_qty",), CAL_UNCAL, is_composite=True,
       notes="agregação p/q em forma (P/b/D/B)"),
    _e("profile.poc", FAM_STRUCT, T_TRADES,
       "institutional_analytics.profile_analysis", None,
       ("raw.trades.price_qty",), CAL_NA,
       notes="nível selecionado da distribuição"),
    _e("profile.vah", FAM_STRUCT, T_TRADES,
       "institutional_analytics.profile_analysis", None,
       ("raw.trades.price_qty",), CAL_NA,
       notes="nível selecionado da distribuição"),
    _e("profile.val", FAM_STRUCT, T_TRADES,
       "institutional_analytics.profile_analysis", None,
       ("raw.trades.price_qty",), CAL_NA,
       notes="nível selecionado da distribuição"),
    _e("profile.breakout_risk", FAM_STRUCT, T_TRADES,
       "institutional_analytics.profile_analysis", None,
       ("profile.shape",), CAL_UNCAL, is_composite=True,
       notes="HI/V_HI/MOD de compressão; fórmula não rastreada em P1-B"),
    _e("market_structure.bos", FAM_STRUCT, T_UNK,
       "institutional.smart_money", None,
       ("raw.candle.high", "raw.candle.low"), CAL_UNCAL,
       notes="detecção de padrão, não agregação"),
    _e("market_structure.sweep", FAM_STRUCT, T_UNK,
       "institutional.smart_money", None,
       ("raw.candle.high", "raw.candle.low"), CAL_UNCAL,
       notes="detecção de padrão, não agregação"),
    _e("market_structure.fvg", FAM_STRUCT, T_UNK,
       "pattern_recognition.fvg", None,
       ("raw.candle.high", "raw.candle.low"), CAL_UNCAL,
       notes="detecção de padrão, não agregação"),
    _e("market_structure.poor_high", FAM_STRUCT, T_TRADES,
       "institutional_analytics.profile_analysis", None,
       ("raw.trades.price_qty",), CAL_UNCAL,
       notes="detecção de leilão incompleto"),
    _e("market_structure.poor_low", FAM_STRUCT, T_TRADES,
       "institutional_analytics.profile_analysis", None,
       ("raw.trades.price_qty",), CAL_UNCAL,
       notes="detecção de leilão incompleto"),
    _e("sr.levels", FAM_STRUCT, T_UNK,
       "support_resistance.defense_zones", None, (), CAL_UNCAL,
       is_composite=True,
       notes="multi-fonte (VP/pivot/EMA/walls); linhagem parcial em P1-B"),
    # ── DERIVATIVES (derivatives_data ← exchange REST; SLOW_CONTEXT)
    _e("derivatives.funding", FAM_DERIV, T_SLOW, "derivatives_data.BTCUSDT",
       None, ("raw.derivatives.funding",), CAL_NA,
       notes="taxa publicada pela exchange"),
    _e("derivatives.lsr", FAM_DERIV, T_SLOW, "derivatives_data.BTCUSDT",
       None, ("raw.derivatives.lsr",), CAL_NA,
       notes="long/short ratio publicado"),
    _e("derivatives.oi", FAM_DERIV, T_SLOW, "derivatives_data.BTCUSDT",
       None, ("raw.derivatives.oi",), CAL_NA,
       notes="nível em contratos; compacto usa milhares"),
    _e("derivatives.oi_delta.1h", FAM_DERIV, T_SLOW,
       "institutional.positioning", H1, ("raw.derivatives.oi_series",),
       CAL_NA, notes="variação da série de OI"),
    _e("derivatives.oi_delta.4h", FAM_DERIV, T_SLOW,
       "institutional.positioning", H4, ("raw.derivatives.oi_series",),
       CAL_NA, notes="variação da série de OI"),
    _e("derivatives.longs_usd", FAM_DERIV, T_SLOW, "derivatives_data.BTCUSDT",
       None, ("raw.derivatives.longs_usd",), CAL_NA),
    _e("derivatives.shorts_usd", FAM_DERIV, T_SLOW, "derivatives_data.BTCUSDT",
       None, ("raw.derivatives.shorts_usd",), CAL_NA),
    _e("positioning.ratio.global", FAM_DERIV, T_SLOW,
       "institutional.positioning", None,
       ("raw.positioning.global_account_ratio",), CAL_NA,
       notes="coorte Binance, NÃO institucional"),
    _e("positioning.ratio.top_account", FAM_DERIV, T_SLOW,
       "institutional.positioning", None,
       ("raw.positioning.top_account_ratio",), CAL_NA,
       notes="coorte top Binance, NÃO institucional"),
    _e("positioning.ratio.top_position", FAM_DERIV, T_SLOW,
       "institutional.positioning", None,
       ("raw.positioning.top_position_ratio",), CAL_NA,
       notes="coorte top Binance, NÃO institucional"),
    # ── CROSS ASSET (ml_features.cross_asset ← polling externo)
    _e("cross.corr.eth_7d", FAM_CROSS, T_SLOW, "ml_features.cross_asset",
       None, ("raw.external.price_returns",), CAL_NA,
       notes="pearson; método/n fora do mapa (telemetria cross.*)"),
    _e("cross.corr.eth_30d", FAM_CROSS, T_SLOW, "ml_features.cross_asset",
       None, ("raw.external.price_returns",), CAL_NA,
       notes="pearson; método/n fora do mapa (telemetria cross.*)"),
    _e("cross.corr.dxy_30d", FAM_CROSS, T_SLOW, "ml_features.cross_asset",
       None, ("raw.external.price_returns",), CAL_NA,
       notes="pearson; método/n fora do mapa (telemetria cross.*)"),
    _e("cross.corr.dxy_90d", FAM_CROSS, T_SLOW, "ml_features.cross_asset",
       None, ("raw.external.price_returns",), CAL_NA,
       notes="pearson; método/n fora do mapa (telemetria cross.*)"),
    _e("cross.corr.ndx_30d", FAM_CROSS, T_SLOW, "ml_features.cross_asset",
       None, ("raw.external.price_returns",), CAL_NA,
       notes="proxy QQQ/IXIC; método/n fora do mapa"),
    # ── MACRO (external_markets ← yahoo/FRED/alternative.me)
    _e("macro.fear_greed", FAM_MACRO, T_SLOW, "external_markets", None,
       ("raw.external.fear_greed",), CAL_NA,
       notes="alternative.me; presença varia por modo ctx"),
    _e("macro.vix", FAM_MACRO, T_SLOW, "external_markets", None,
       ("raw.external.vix",), CAL_NA, notes="yahoo ^VIX delay ~15min"),
    _e("macro.dxy", FAM_MACRO, T_SLOW, "external_markets", None,
       ("raw.external.dxy",), CAL_NA, notes="presença varia por modo ctx"),
    _e("macro.us10y", FAM_MACRO, T_SLOW, "external_markets", None,
       ("raw.external.treasury_10y",), CAL_NA,
       notes="via TNX; presença varia por modo ctx"),
    _e("macro.gold", FAM_MACRO, T_SLOW, "external_markets", None,
       ("raw.external.gold",), CAL_NA, notes="presença varia por modo ctx"),
    _e("macro.wti", FAM_MACRO, T_SLOW, "external_markets", None,
       ("raw.external.wti",), CAL_NA, notes="presença varia por modo ctx"),
    # ── DERIVED COMPOSITES (type DERIVED; nunca ancestrais primários)
    _e("whale.score", FAM_UNK, T_DERIVED, "flow_analyzer.whale_score",
       None,
       ("participants.whale.delta.window", "participants.mid.delta.window",
        "participants.retail.delta.window", "orderbook.snapshot.bid_depth",
        "orderbook.snapshot.ask_depth", "flow.cvd", "derivatives.lsr",
        "derivatives.funding"),
       CAL_UNCAL, is_composite=True,
       notes="multi-família (flow+depth+deriv); absorption é NON_VOTING P0-A2 "
             "(só notes, fora de derived_from); netflow-bônus fora do mapa "
             "(sem família ONCHAIN no enum v1)"),
    _e("whale.classification", FAM_UNK, T_DERIVED, "flow_analyzer.whale_score",
       None, ("whale.score",), CAL_UNCAL, is_composite=True,
       notes="mapeamento por threshold 50/20 de whale.score; informação "
             "contida no score, nunca evidência primária independente"),
    _e("whale.bias", FAM_UNK, T_DERIVED, "flow_analyzer.whale_score",
       None, ("whale.score",), CAL_UNCAL, is_composite=True,
       notes="mapeamento por threshold ±10 de whale.score; nunca independente"),
    _e("regime.current", FAM_UNK, T_DERIVED, "institutional.enricher",
       None, ("flow.trend", "profile.shape", "orderbook.snapshot.imbalance",
              "whale.score"),
       CAL_UNCAL, is_composite=True, validity_source="regime.status",
       notes="só o que vota pós P0-D2 (defaults removidos); multi-família"),
    _e("regime.distribution", FAM_UNK, T_DERIVED, "institutional.enricher",
       None, ("regime.current",), CAL_UNCAL, is_composite=True,
       notes="normalização dos mesmos votos (não evidência nova)"),
    _e("regime.change_prob", FAM_UNK, T_DERIVED, "institutional.enricher",
       None, ("regime.distribution",), CAL_UNCAL, is_composite=True,
       notes="heurística rotulada probability; não calibrada"),
    _e("regime.duration", FAM_UNK, T_DERIVED, "institutional.enricher",
       None, ("regime.current",), CAL_UNCAL, is_composite=True,
       notes="lookup fixo por regime; não calibrado"),
    _e("regime.mode", FAM_UNK, T_DERIVED, "payload_builder_compact.regime",
       None, ("regime.current",), CAL_UNCAL, is_composite=True,
       notes="MR/TRD/BRK/RB/UNK a partir de current_regime (+fallback legado)"),
    _e("regime.consensus", FAM_UNK, T_DERIVED,
       "payload_builder_compact.regime", None,
       ("tf.trend.15m", "tf.trend.1h", "tf.trend.4h", "tf.trend.1d",
        "macro.fear_greed"),
       CAL_UNCAL, is_composite=True, notes="votos 1d=4..15m=1 + FG extremo"),
    _e("alerts.active", FAM_UNK, T_DERIVED, "trading.alert_engine",
       None, (), CAL_UNCAL, is_composite=True,
       notes="multi-detector; linhagem por alerta fora do mapa P1-B"),
    _e("ml.prob_up", FAM_UNK, T_DERIVED, "ml.inference", None, (),
       CAL_UNCAL, is_composite=True,
       notes="saída de modelo; feature lineage fora do escopo P1-B"),
    _e("ml.confidence", FAM_UNK, T_DERIVED, "ml.inference", None, (),
       CAL_UNCAL, is_composite=True,
       notes="saída de modelo; feature lineage fora do escopo P1-B"),
    _e("smart_money.score", FAM_UNK, T_DERIVED,
       "institutional_analytics.smart_money", None, (), CAL_UNCAL,
       is_composite=True, notes="inputs não rastreados em P1-B"),
    _e("mean_reversion.score", FAM_UNK, T_DERIVED,
       "institutional_analytics.mean_reversion", None, (), CAL_UNCAL,
       is_composite=True, notes="inputs não rastreados em P1-B"),
    _e("liquidity.clusters", FAM_UNK, T_DERIVED,
       "flow_analyzer.liquidity_heatmap", None, (), CAL_UNCAL,
       is_composite=True, notes="inputs não rastreados em P1-B"),
    # ── EFFORT VS RESULT RAW (P1-D; flow_analyzer/effort_response) ──────────
    # Notionals vêm da MESMA partição m_flags dos notionals do evento
    # (data_handler): compartilham ancestry com EXECUTED_FLOW. OHLC vem dos
    # preços dos MESMOS aggTrades: linhagem honestamente sobreposta, nunca
    # disjunta. Sem ratio esforço/preço (sem contrato de denominador).
    _e("effort.notional.buy", FAM_FLOW, T_TRADES, "data_handler.notionals",
       None, ("raw.aggtrade.buy_notional",), CAL_NA,
       notes="agressão BUY já classificada; NUNCA re-multiplicar por pct"),
    _e("effort.notional.sell", FAM_FLOW, T_TRADES, "data_handler.notionals",
       None, ("raw.aggtrade.sell_notional",), CAL_NA,
       notes="agressão SELL já classificada; NUNCA re-multiplicar por pct"),
    _e("effort.notional.total", FAM_FLOW, T_TRADES, "data_handler.notionals",
       None, ("raw.aggtrade.buy_notional", "raw.aggtrade.sell_notional"),
       CAL_NA, notes="buy+sell por construção do producer"),
    _e("effort.notional.net", FAM_FLOW, T_TRADES, "data_handler.notionals",
       None, ("raw.aggtrade.buy_notional", "raw.aggtrade.sell_notional"),
       CAL_NA, notes="buy-sell; zero permitido; sinal fica no value"),
    _e("effort.share.buy", FAM_FLOW, T_TRADES, "flow_analyzer.effort_response",
       None, ("effort.notional.buy", "effort.notional.sell"), CAL_NA,
       notes="buy/total; null se total zero"),
    _e("effort.share.sell", FAM_FLOW, T_TRADES, "flow_analyzer.effort_response",
       None, ("effort.notional.buy", "effort.notional.sell"), CAL_NA,
       notes="sell/total; null se total zero"),
    _e("effort.price.displacement_usd", FAM_PRICE, T_TRADES,
       "flow_analyzer.effort_response", None, ("raw.aggtrade.price",),
       CAL_NA, notes="close-open; dependência computacional estrita de preço"),
    _e("effort.price.displacement_bps", FAM_PRICE, T_TRADES,
       "flow_analyzer.effort_response", None, ("raw.aggtrade.price",),
       CAL_NA, notes="(...)/open*10000; dependência computacional estrita de preço"),
    _e("effort.price.range_usd", FAM_PRICE, T_TRADES,
       "flow_analyzer.effort_response", None, ("raw.aggtrade.price",),
       CAL_NA, notes="high-low; dependência computacional estrita de preço"),
    _e("effort.price.range_bps", FAM_PRICE, T_TRADES,
       "flow_analyzer.effort_response", None, ("raw.aggtrade.price",),
       CAL_NA, notes="(...)/open*10000; dependência computacional estrita de preço"),
    _e("effort.price.close_from_high_usd", FAM_PRICE, T_TRADES,
       "flow_analyzer.effort_response", None, ("raw.aggtrade.price",),
       CAL_NA, notes="close-high; dependência computacional estrita de preço"),
    _e("effort.price.close_from_high_bps", FAM_PRICE, T_TRADES,
       "flow_analyzer.effort_response", None, ("raw.aggtrade.price",),
       CAL_NA, notes="(...)/open*10000; dependência computacional estrita de preço"),
    _e("effort.price.close_from_low_usd", FAM_PRICE, T_TRADES,
       "flow_analyzer.effort_response", None, ("raw.aggtrade.price",),
       CAL_NA, notes="close-low; dependência computacional estrita de preço"),
    _e("effort.price.close_from_low_bps", FAM_PRICE, T_TRADES,
       "flow_analyzer.effort_response", None, ("raw.aggtrade.price",),
       CAL_NA, notes="(...)/open*10000; dependência computacional estrita de preço"),
    _e("effort.price.close_vs_vwap_usd", FAM_PRICE, T_TRADES,
       "flow_analyzer.effort_response", None,
       ("raw.aggtrade.price", "raw.trades.price_qty"),
       CAL_NA, notes="close-vwap; vwap usa preco e volume ponderado"),
    _e("effort.price.close_vs_vwap_bps", FAM_PRICE, T_TRADES,
       "flow_analyzer.effort_response", None,
       ("raw.aggtrade.price", "raw.trades.price_qty"),
       CAL_NA, notes="(...)/open*10000; vwap usa preco e volume ponderado"),
    _e("effort.price.close_vs_poc_usd", FAM_PRICE, T_TRADES,
       "flow_analyzer.effort_response", None,
       ("raw.aggtrade.price", "raw.trades.price_qty"),
       CAL_NA, notes="poc de volume profile usa bins de preco e volume"),
    _e("effort.price.close_vs_poc_bps", FAM_PRICE, T_TRADES,
       "flow_analyzer.effort_response", None,
       ("raw.aggtrade.price", "raw.trades.price_qty"),
       CAL_NA, notes="(...)/open*10000; poc de volume profile"),
    _e("effort.response", FAM_UNK, T_DERIVED, "flow_analyzer.effort_response",
       None,
       ("effort.notional.buy", "effort.notional.sell",
        "effort.notional.total", "effort.notional.net", "effort.share.buy",
        "effort.share.sell", "effort.price.displacement_usd",
        "effort.price.displacement_bps", "effort.price.range_usd",
        "effort.price.range_bps", "effort.price.close_from_high_usd",
        "effort.price.close_from_high_bps", "effort.price.close_from_low_usd",
        "effort.price.close_from_low_bps", "effort.price.close_vs_vwap_usd",
        "effort.price.close_vs_vwap_bps", "effort.price.close_vs_poc_usd",
        "effort.price.close_vs_poc_bps"),
       CAL_NA, is_composite=True,
       notes="container sem julgamento; sem direito a voto"),
]

FIELDS: dict[str, TaxonomyEntry] = {e.field_id: e for e in _ENTRIES}

# ── Aliases compactos -> field_id canônico (alias nunca cria evidência) ──────

ALIASES: dict[str, str] = {
    # flow (compacto)
    "d1": "flow.net.1m",
    "d5": "flow.net.5m",
    "d15": "flow.net.15m",
    "trade_imb": "flow.imbalance.1m",
    "imb": "flow.imbalance.1m",
    "bsr": "flow.buy_sell_ratio",
    "ab": "flow.aggressive_buy_pct.1m",
    "cvd_4h": "flow.cvd",
    "sf_w_4h": "participants.whale.delta.4h",
    "sf_r_4h": "participants.retail.delta.4h",
    # whale (compacto v3/abreviado)
    "w.s": "whale.score",
    "w.c": "whale.classification",
    # orderbook (compacto/v3)
    "ob.imb": "orderbook.snapshot.imbalance",
    "depth_imb": "orderbook.snapshot.imbalance",
    "b": "orderbook.snapshot.bid_depth",
    "a": "orderbook.snapshot.ask_depth",
    "slip_b": "market_impact.slippage.buy.100k",
    "slip_s": "market_impact.slippage.sell.100k",
    "liq_score": "market_impact.liquidity.score",
    "exec_qual": "market_impact.execution_quality",
    # price/profile (compacto)
    "c": "price.close",
    "o": "price.open",
    "h": "price.high",
    "l": "price.low",
    "vw": "price.vwap",
    "sh": "profile.shape",
    "brk_risk": "profile.breakout_risk",
    # derivatives/contexto (compacto)
    "fr": "derivatives.funding",
    "lsr": "derivatives.lsr",
    "oi": "derivatives.oi",
    "eth7": "cross.corr.eth_7d",
    "dxy30": "cross.corr.dxy_30d",
    # quant/modelo (compacto)
    "pu": "ml.prob_up",
    # regime (compacto)
    "cs": "regime.consensus",
    "cf": "regime.consensus",
    "mode": "regime.mode",
    "prob_trend": "regime.distribution",
    "prob_rev": "regime.distribution",
    "prob_break": "regime.distribution",
}


def resolve_field_id(alias_or_id: str) -> Optional[str]:
    """Alias compacto -> field_id canônico. Desconhecido => None (sem chute)."""
    if not isinstance(alias_or_id, str) or not alias_or_id:
        return None
    if alias_or_id in ALIASES:
        return ALIASES[alias_or_id]
    return alias_or_id if alias_or_id in FIELDS else None


def primary_ancestors(field_id: str) -> frozenset:
    """Fecho transitivo de ancestrais primários (só auditoria, sem dedup/voto).

    - Atravessa composites e campos comuns, mas NUNCA inclui um composite nem
      o próprio campo consultado.
    - Ids `raw.*` não registrados são terminais primários (incluídos).
    - Ciclo => ValueError. Desconhecido => KeyError.
    """
    if field_id not in FIELDS:
        raise KeyError(f"field_id desconhecido: {field_id!r}")
    result: set = set()
    visiting: list = []

    def _visit(node: str) -> None:
        if node in visiting:
            raise ValueError(
                "ciclo no registry: " + " -> ".join([*visiting, node]))
        if node not in FIELDS:
            result.add(node)  # terminal raw.* (primário por definição)
            return
        visiting.append(node)
        try:
            for dep in FIELDS[node].derived_from:
                if dep == field_id:
                    continue  # nunca inclui o próprio consultado
                if dep in FIELDS and FIELDS[dep].is_composite:
                    _visit(dep)  # atravessa, não inclui
                else:
                    _visit(dep)
        finally:
            visiting.pop()
        if node != field_id and not FIELDS[node].is_composite:
            result.add(node)

    _visit(field_id)
    return frozenset(result)


def shared_ancestor_groups(min_size: int = 2) -> dict:
    """Agrupa field_ids pelo conjunto de ancestrais primários.

    Retorna {frozenset(ancestrais): [field_ids]} só com grupos de >=min_size
    membros (candidatos a double counting — relatório, sem correção).
    """
    groups: dict = {}
    for fid in FIELDS:
        try:
            ancestors = primary_ancestors(fid)
        except (KeyError, ValueError):
            continue
        if not ancestors:
            continue
        groups.setdefault(ancestors, []).append(fid)
    return {k: sorted(v) for k, v in groups.items() if len(v) >= min_size}


def _validate_registry() -> None:
    """Fail-fast no import: ids/aliases únicos, aliases válidos, sem ciclos."""
    if len(FIELDS) != len(_ENTRIES):
        seen: set = set()
        dupes = sorted(e.field_id for e in _ENTRIES
                       if e.field_id in seen or seen.add(e.field_id))
        raise ValueError(f"field_ids duplicados: {dupes}")
    for alias, target in ALIASES.items():
        if target not in FIELDS:
            raise ValueError(f"alias {alias!r} aponta para {target!r} inexistente")
    if len(set(ALIASES)) != len(ALIASES):
        raise ValueError("aliases duplicados")
    for fid in FIELDS:
        for dep in FIELDS[fid].derived_from:
            if not dep.startswith("raw.") and dep not in FIELDS:
                raise ValueError(
                    f"{fid}: derived_from {dep!r} nem registrado nem raw.*")
    for fid in FIELDS:  # detecta ciclos (levanta ValueError)
        primary_ancestors(fid)


_validate_registry()
