"""
common/ai_field_legend.py — Legenda de Campos do Payload Compacto

Fornece a legenda que deve ser incluída no system prompt da IA
para que os campos abreviados apresentem significado inequívoco e semântica rigorosa.

Uso:
    from common.ai_field_legend import FIELD_LEGEND
    system_prompt = f"{BASE_PROMPT}\n\n{FIELD_LEGEND}"
"""

FIELD_LEGEND: str = """
=== FIELD REFERENCE ===
mkt=market(fut_perp=Binance USD-M perpetual; flow AND ob ARE THE SAME BOOK; never mix with spot)
t=trigger type (AT=analysis,ABS=absorption,EXH=exhaustion,BRK=breakout,WHL=whale,DIV=divergence)
p=price: c=close,o=open,h=high,l=low,vw=vwap,sh=profile_shape(B=bimodal,P=P-shape,b=b-shape,D=D-shape),auc=auction_type,ph=poor_high(1=yes),pl=poor_low(1=yes)
r=regime: v=volatility(L=low,M=med,H=high),tr=trend(DN=down,UP=up,SW=sideways),st=sentiment(BEAR/BULL/NEUT)
f=flow: d1/d5/d15=net_delta_USD_1m/5m/15m(+buy,-sell),cvd_4h=cumulative_volume_delta_BTC_since_last_reset(up_to_4h_resets_automatically)_NOT_current_1m_window,sf_r_4h=sector_flow_retail_delta_BTC_accumulated_4h(retail<0.2BTC),sf_w_4h=sector_flow_whale_delta_BTC_accumulated_4h(whale>=2.0BTC),trade_imb=taker_trade_imbalance_1m[-1_sell,+1_buy](source:Binance_Futures_aggTrades_1m,formula:(buy_usd-sell_usd)/(buy_usd+sell_usd),domain:[-1,+1],positive=aggressive_BUY,negative=aggressive_SELL),ab=aggressive_buy_pct(0-100,only_when_observed),ab_s=aggressive_status(observed|insufficient|no_volume|invalid),ab_n=aggressive_sample_trades,missing_ab=not_observed_never_50,bsr=DERIVED_REDUNDANCY(buy_sell_ratio,bijection:imb=(bsr-1)/(bsr+1),>1=buyers_dominate,<1=sellers_dominate),q=window_temporal_quality:s=status(full=>=99%_coverage_no_truncation|warm=warming_up|trunc=capacity_truncated_partial),c=effective_coverage_pct(0-100);when_s!=full_treat_d1/d5/d15_of_that_timeframe_as_PARTIAL_not_complete
ob=orderbook: b=bid_depth_USD_top50,a=ask_depth_USD_top50,depth_imb=L2_depth_top50_snapshot[-1_ask_heavy,+1_bid_heavy](USD_notional_depth:(bid-ask)/(bid+ask),snapshot_static_NOT_flow),depth_t5=L2_depth_top5_snapshot[-1_ask_heavy,+1_bid_heavy](USD_notional_depth_top5:(bid-ask)/(bid+ask),snapshot_static_NOT_flow),spread_pct=spread_percent
market_impact (mi/ob): slip_100k/slip_1m={buy,sell}=VWAP_execution_slippage_USD_relative_to_mid(point-in-time_REST_L2_snapshot_NOT_continuous_L2;null=insufficient_observed_depth_for_full_requested_notional_fail_closed),bf/sf=fraction_of_requested_notional_observable_in_snapshot_when_partial(0-1),exec_qual=execution_quality_tier(EXCEL/GOOD/FAIR/POOR/P1M=partial_1M/INSUF=insufficient_100k),liq_score=liquidity_score(0-10_based_on_100k_USD_sweep_distance_reference_notional_usd=100000;null_if_100k_insufficient). terminal_move(when_present_in_extended_event)=final_price_minus_mid(sweep_distance_NOT_execution_slippage).
divergence_note: trade_imb and depth_imb measure different phenomena (executed taker aggression vs passive book depth asymmetry); signs may legitimately diverge and divergence alone is NOT automatic absorption (absorption requires additional evidence)
trade_bar_flow: score=executed_trade_bar_direction_score_5x30s[-1_sell,+1_buy],dir=direction(BUY/SELL/NEU)(source:executed_trades,bars:5_discrete_bars_30s_approx_150s_nominal,directional_dominance_between_bars;omitted_when_unavailable;NOT_L2_orderbook_OFI)
w=whale_score(-100=strong_distribution,+100=strong_accumulation,0=neutral,whale>=2.0BTC)
q=quant/ML: pu=probability_up(0-1),c=confidence(0-1)
tf=timeframes: t=trend(DN/UP/SW),rsi(0-100),macd=[line,signal],adx(0-100),atr=avg_true_range,r=regime(RNG=range,ACC=accumulation,TRD=trending,MNP=manipulation)
ctx=context(sent every 5min): ses=session,dxy/tnx/spx/ndx/gold/wti/vix=market_prices,fg=fear_greed(0-100),poc/val/vah=volume_profile_daily,lsr=btc_long_short_ratio,eth_lsr=eth_long_short_ratio,oi=btc_open_interest_thousands,eth7=btc_eth_corr_7d,dxy30=btc_dxy_corr_30d
cross=cross_asset(st=fresh/stale/unavailable,method=shared_session_returns_v2|positional_v1[legacy],n=min_returns_used_for_pearson,inst_dxy/inst_ndx=actual_ticker_used;ndx_field_is_QQQ_or_IXIC_PROXY_never_the_index;positional_v1_values_are_NOT_temporally_aligned_do_not_compare_across_methods)
pivot_points (evento completo, quando presente): Pivot clássico (H+L+C)/3 do período anterior COMPLETO (iloc[-2]); FIXO durante o dia quando source=classic. Quando source=vp_fallback (dados insuficientes p/ clássico), reflete volume profile INTRADAY PARCIAL (dia atual 00:00Z→agora, muda a cada atualização) — não é pivot clássico fixo. pivot_points NÃO é enviado no payload compacto (ver sr.* e ctx.poc/val/vah).
sr=defense_zones: r1/r2/s1/s2=[preco,forca], forca(0-100)=HEURISTIC composite zone score (media_forca×(1+0.3×n_fontes)), NOT probability/confidence/persistencia; source_count/conf=n_fontes=confluencia; OBS_WALL(when r*_src/s*_src present)=point-in-time REST L2 liquidity concentration (snapshot_only, sem persistencia, NOT standalone support/resistance, nunca CONF/PERSISTENT); r*_dist/s*_dist=distancia ao preco; r*_conf/s*_conf=fontes do nivel; def_bias=viés da defesa. ctx.poc/val/vah = Volume Profile diario REAL.
Number suffixes: K=thousands,M=millions. Always in USD unless noted as BTC. Signs: +=buy/positive,-=sell/negative.
When ctx is absent, use the last received context values.
""".strip()

# Versão ultra-compacta para economizar tokens no prompt (~75 tokens)
FIELD_LEGEND_COMPACT: str = """
KEYS: t=trigger,p=price(c/o/h/l/vw/sh/auc/ph/pl),r=regime(v/tr/st),f=flow(d1/d5/d15=deltaUSD,cvd_4h=BTC[acc4h],sf_w_4h/sf_r_4h=BTC[acc4h],trade_imb[-1sell+1buy],ab=aggBuy%[observed_only]+ab_s/ab_n,bsr[DERIVED_REDUNDANCY]),ob(b/a=depthUSD,depth_imb[-1ask+1bid],depth_t5),mi(slip_100k/slip_1m=VWAP_slip_USD[null=insuf],bf/sf=fill_ratio,exec_qual[P1M=partial1M/EXCEL/GOOD/FAIR/POOR/INSUF],liq_score[0-10,ref100k,null=insuf]),trade_bar_flow(score[-1+1],dir),w=whaleScore[-100dist+100accum],q(pu=probUp,c=conf),tf(t=trend,rsi,macd,adx,atr,r=regime),ctx(ses,dxy,tnx,spx,ndx,gold,wti,vix,fg,poc,val,vah,lsr,oi,eth7,dxy30),cross(st,method,n,inst;ndx=proxy).K=1000,M=1M.+buy/-sell.USD unless BTC noted.REST_snapshot_NOT_continuous_L2.No ctx=use last.
""".strip()
