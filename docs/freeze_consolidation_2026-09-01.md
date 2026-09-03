# Consolidação do Paper Trading Freeze — 01/09/2026

- **Tag de freeze:** paper_trading_freeze_2026-09-01
- **Commit HEAD:** 4c599349055e78d1409639010de210b5df14cb01
- **Estado do repositório:** working tree limpo, staging limpo, origin/main sincronizado
- **Processo shadow (PID 13728):** ativo, sem crash, 0 traceback fatal
- **Pipeline de captura (trades/book/fluxo/institucional/SR/regime):** validado, sem corrupção silenciosa
- **Persistência (SQLite + JSONL + visual):** funcionando, dados acumulados em disco
- **Integridade matemática:** invariantes de CVD, S/R, volume, orderbook, regime — todas PASS
- **Qualidade temporal:** market_data_age separada em start/end; flow_window_status real (WARMING_UP/FULL); sem false-full
- **Orderbook:** fonte diferenciada (live/cache) com fallback estrito para "unknown" (nunca assumido como "live")
- **Payload compacto:** 720 tokens médios, 10/10 grupos de informação preservados
- **Enrichment duplicado:** 2 janelas com 2 sinais (explicável por múltiplos triggers, não duplicação acidental)
- **API de IA externa (Groq):** DESATIVADA nesta fase; pipeline opera sem ela
- **Latência com IA removida:** pipeline p50 ~7.5s (vs ~13.0s com IA), reduction ~42.5%; decision delay p50 ~8.7s (vs ~14.0s), reduction ~37.7%
- **Status de paper trading:** BASELINE INTEGRO; pipeline pronto para operação de observação contínua; decisão de entrada suspensa até substituição da API de IA
- **Ações pendentes antes de retomar com nova IA:** validar se entry_zone/invalidation_zone são preenchidos corretamente quando a nova API retornar action=buy/sell (não apenas action=wait); continuar monitorando persistência no banco; verificar se gaps ofi/vwap/liq são situacionais ou estruturais após troca de API.
