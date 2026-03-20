"""
common/ai_field_legend.py — Legenda de Campos do Payload Compacto

Fornece a legenda que deve ser incluída no system prompt da IA
para que ela interprete corretamente os campos abreviados.

Uso:
    from common.ai_field_legend import FIELD_LEGEND
    system_prompt = f"{BASE_PROMPT}\n\n{FIELD_LEGEND}"
"""

FIELD_LEGEND: str = """
=== FIELD REFERENCE ===
t=trigger type (AT=analysis,ABS=absorption,EXH=exhaustion,BRK=breakout,WHL=whale,DIV=divergence)
p=price: c=close,o=open,h=high,l=low,vw=vwap,sh=profile_shape(B=bimodal,P=P-shape,b=b-shape,D=D-shape),auc=auction_type,ph=poor_high(1=yes),pl=poor_low(1=yes)
r=regime: v=volatility(L=low,M=med,H=high),tr=trend(DN=down,UP=up,SW=sideways),st=sentiment(BEAR/BULL/NEUT)
f=flow: d1/d5/d15=net_delta_USD_1m/5m/15m(+buy,-sell),cvd=cumulative_volume_delta_BTC,imb=flow_imbalance(-1=strong_sell,+1=strong_buy),ab=aggressive_buy_pct(0-100),bsr=buy_sell_ratio(>1=buyers_dominate,<1=sellers_dominate)
ob=orderbook: b=bid_depth_USD,a=ask_depth_USD,imb=depth_imbalance(-1=ask_heavy,+1=bid_heavy),t5=top5_levels_imbalance
w=whale_score(-100=strong_distribution,+100=strong_accumulation,0=neutral)
q=quant/ML: pu=probability_up(0-1),c=confidence(0-1)
tf=timeframes: t=trend(DN/UP/SW),rsi(0-100),macd=[line,signal],adx(0-100),atr=avg_true_range,r=regime(RNG=range,ACC=accumulation,TRD=trending,MNP=manipulation)
ctx=context(sent every 5min): ses=session,dxy/tnx/spx/ndx/gold/wti/vix=market_prices,fg=fear_greed(0-100),poc/val/vah=volume_profile_daily,lsr=btc_long_short_ratio,eth_lsr=eth_long_short_ratio,oi=btc_open_interest_thousands,eth7=btc_eth_corr_7d,dxy30=btc_dxy_corr_30d
Number suffixes: K=thousands,M=millions. Always in USD unless noted as BTC. Signs: +=buy/positive,-=sell/negative.
When ctx is absent, use the last received context values.
""".strip()

# Versão ultra-compacta para economizar tokens no prompt (~60 tokens)
FIELD_LEGEND_COMPACT: str = """
KEYS: t=trigger,p=price(c/o/h/l/vw/sh/auc/ph/pl),r=regime(v/tr/st),f=flow(d1/d5/d15=deltaUSD,cvd=BTC,imb[-1sell+1buy],ab=aggBuy%,bsr[>1=buyDom]),ob(b/a=depthUSD,imb,t5),w=whaleScore[-100dist+100accum],q(pu=probUp,c=conf),tf(t=trend,rsi,macd,adx,atr,r=regime),ctx(ses,dxy,tnx,spx,ndx,gold,wti,vix,fg,poc,val,vah,lsr,oi,eth7,dxy30).K=1000,M=1M.+buy/-sell.USD unless BTC noted.No ctx=use last.
""".strip()
