# cross_asset_correlations.py
"""
Módulo para cálculo de correlações cross-asset para BTCUSDT.

Foco especial:
- BTC x DXY (inversa) - correlação esperada negativa
- BTC x NDX - correlação com mercado tech
- BTC x ETH - correlação entre principais cryptos

Fontes de dados:
- Binance (velas 1h): BTCUSDT, ETHUSDT
- yfinance (diário): BTC-USD, DXY, ^NDX

CONTRATO TEMPORAL v2 (F5-C; shared-session):
- BTC x TradFi: closes nas mesmas DATAS (join por data UTC, nunca timestamp
  exato — barras diárias têm tz/hora de close distintos) e retornos calculados
  DEPOIS do alinhamento, sobre os mesmos endpoints temporais. Ex.: segunda usa
  BTC_seg/BTC_sex e TradFi_seg/TradFi_sex (nunca BTC_seg/BTC_dom).
- Somente sessões com availability_time <= decision_time. Conservador: barras
  com data >= data da decisão são proibidas (sessão ainda aberta ou parcial);
  não hardcodamos delay de fonte (market closed != source available).
- Sem forward-fill e sem ASOF para Pearson (carregar sexta p/ fds fabrica
  retornos zero). Weekends/holidays excluídos naturalmente pela interseção.
- BTC x ETH (1h, mesma exchange): inner join por open_time, candle ainda
  aberto (close_time > decision) excluído, retornos após alinhamento.
- N contado APÓS os retornos; N < CORR_MIN_POINTS => NaN/insufficient, nunca 0.
- Método/instrumento/N persistidos junto da feature (method/instrument/n);
  linhas antigas sem a chave = positional_v1 (não misturar em treino).
- Features cross-asset CONTINUAM BLOQUEADAS para treinamento ML.
- `btc_dominance_change_7d` NÃO é observação real (placeholder omitido);
  nunca confundir com dado medido.
"""

import logging
import asyncio
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

# Import para novas fontes de dados macro
try:
    from fetchers.macro_data_provider import MacroDataProvider
    _MACRO_DATA_OK = True
except ImportError as e:
    _MACRO_DATA_OK = False
    logging.warning(f"macro_data_provider indisponível: {e}")

# Configuração de logging
logger = logging.getLogger("CrossAssetCorrelations")

# ===============================
# Funções utilitárias internas
# ===============================

CORR_MIN_POINTS = 10  # Mínimo de pontos para correlação confiável
# DÍVIDA ESTATÍSTICA (F5-C): 10 é convenção herdada, não calibração.
# Não alterar aqui; `n` real chega à IA/payload para ponderar confiança.

CORR_METHOD = "shared_session_returns_v2"
CORR_CONTRACT_VERSION = 2  # 1 = positional_v1 legado (linhas sem a chave)

# Tipo para valores do resultado (pode ser str, float, int, None, etc.)
ResultValue = Union[str, float, int, None]


def _log_returns(series: pd.Series) -> pd.Series:
    """
    Calcula retornos logarítmicos de uma série de preços.

    Args:
        series: Série de preços

    Returns:
        Série de retornos logarítmicos (diff de log)
    """
    # CORREÇÃO: Usar pd.Series explicitamente para manter tipo correto
    log_series = pd.Series(np.log(series.values), index=series.index)
    return log_series.diff().dropna()


def _corr_last_window(series_a: pd.Series,
                      series_b: pd.Series,
                      window: int) -> float:
    """
    Calcula correlação de Pearson entre duas séries de retornos,
    alinhando por posição (não por timestamp) para evitar NaN por falta de interseção de datas.

    Args:
        series_a: Primeira série de retornos
        series_b: Segunda série de retornos
        window: Número de pontos a considerar

    Returns:
        Correlação de Pearson ou NaN se dados insuficientes
    """
    max_len = min(len(series_a), len(series_b), window)
    if max_len < CORR_MIN_POINTS:
        return float("nan")

    a = series_a.tail(max_len).reset_index(drop=True)
    b = series_b.tail(max_len).reset_index(drop=True)
    corr = a.corr(b)

    return float(round(corr, 4)) if not pd.isna(corr) else float("nan")


# ===============================
# Alinhamento temporal v2 (F5-C)
# ===============================

def _decision_date(now_utc: Optional[datetime]) -> Any:
    """Data de decisão (UTC). Naive é interpretado como UTC (convenção local)."""
    if now_utc is None:
        return datetime.now(timezone.utc).date()
    if now_utc.tzinfo is None:
        return now_utc.replace(tzinfo=timezone.utc).date()
    return now_utc.astimezone(timezone.utc).date()


def _decision_ms(now_utc: Optional[datetime]) -> int:
    """Decisão em epoch ms (para klines intradiários)."""
    if now_utc is None:
        return int(datetime.now(timezone.utc).timestamp() * 1000)
    if now_utc.tzinfo is None:
        return int(now_utc.replace(tzinfo=timezone.utc).timestamp() * 1000)
    return int(now_utc.timestamp() * 1000)


def _daily_by_date(close: pd.Series) -> Dict[Any, float]:
    """Closes diários indexados por DATA (UTC). Dedup keep-last, ordenado fora.

    Barras diárias de fontes distintas têm tz/hora de close distintos
    (BTC 00:00 UTC vs TradFi 00:00 ET = 04:00 UTC); join por timestamp exato
    seria vazio. A data é a chave correta para sessão diária.
    """
    series = close.dropna()
    series = series[~series.index.duplicated(keep="last")]
    out: Dict[Any, float] = {}
    for ts, val in series.items():
        try:
            t = pd.Timestamp(ts)
            if t.tzinfo is None:
                t = t.tz_localize("UTC")  # naive = UTC (convenção local)
            else:
                t = t.tz_convert("UTC")
            day = t.date()
        except (TypeError, ValueError):
            continue
        try:
            out[day] = float(val)
        except (TypeError, ValueError):
            continue
    return out


def shared_session_corr(a_close: pd.Series, b_close: pd.Series,
                        window: int,
                        decision_date: Optional[Any] = None) -> Dict[str, Any]:
    """Pearson sobre retornos de SESSÕES COMPARTILHADAS (F5-C).

    1. interseção de datas com availability <= decision (data < decision_date);
    2. closes nessas datas; 3. retornos DEPOIS do alinhamento (mesmos endpoints).
    N contado após os retornos; insuficiente => NaN (nunca 0).
    """
    try:
        ma = _daily_by_date(a_close)
        mb = _daily_by_date(b_close)
        shared = sorted(set(ma) & set(mb))
        if decision_date is not None:
            shared = [d for d in shared if d < decision_date]
        tail = shared[-(window + 1):] if window else shared
        n_ret = len(tail) - 1
        if n_ret < CORR_MIN_POINTS:
            return {"corr": float("nan"), "n": max(n_ret, 0),
                    "first": tail[0].isoformat() if tail else None,
                    "last": tail[-1].isoformat() if tail else None,
                    "pairs": [(d.isoformat(), d.isoformat()) for d in tail[:5]]}
        av = pd.Series([ma[d] for d in tail])
        bv = pd.Series([mb[d] for d in tail])
        ra = _log_returns(av.reset_index(drop=True))
        rb = _log_returns(bv.reset_index(drop=True))
        corr = _corr_last_window(ra, rb, len(ra))
        return {"corr": corr, "n": len(ra),
                "first": tail[0].isoformat(), "last": tail[-1].isoformat(),
                "pairs": [(tail[i].isoformat(), tail[i].isoformat())
                          for i in range(1, min(6, len(tail)))]}
    except Exception as e:
        logger.debug(f"shared_session_corr falhou: {e}")
        return {"corr": float("nan"), "n": 0, "first": None,
                "last": None, "pairs": []}


def intraday_join_corr(btc_df: pd.DataFrame, eth_df: pd.DataFrame,
                       window: int,
                       decision_ms: Optional[int] = None) -> Dict[str, Any]:
    """BTC x ETH 1h: inner join por timestamp, sem candle aberto (F5-C).

    Remove velas com close_time > decision (ainda abertas), join interno por
    open_time compartilhado, retornos DEPOIS do alinhamento. Gaps excluídos
    naturalmente. N contado após os retornos; insuficiente => NaN.
    """
    try:
        b = btc_df.copy()
        e = eth_df.copy()
        for df in (b, e):
            df["close_time_ms"] = pd.to_numeric(df["close_time"], errors="coerce")
        if decision_ms is not None:
            b = b[b["close_time_ms"] <= decision_ms]
            e = e[e["close_time_ms"] <= decision_ms]
        b = b[~b.index.duplicated(keep="last")].sort_index()
        e = e[~e.index.duplicated(keep="last")].sort_index()
        idx = b.index.intersection(e.index).sort_values()
        tail = idx[-(window + 1):] if window else idx
        n_ret = len(tail) - 1
        if n_ret < CORR_MIN_POINTS:
            return {"corr": float("nan"), "n": max(n_ret, 0),
                    "first": tail[0].isoformat() if len(tail) else None,
                    "last": tail[-1].isoformat() if len(tail) else None,
                    "pairs": []}
        av = pd.to_numeric(b.loc[tail, "close"], errors="coerce").dropna()
        bv = pd.to_numeric(e.loc[tail, "close"], errors="coerce").dropna()
        common_n = min(len(av), len(bv))
        if common_n - 1 < CORR_MIN_POINTS:
            return {"corr": float("nan"), "n": max(common_n - 1, 0),
                    "first": None, "last": None, "pairs": []}
        av = av.tail(common_n).reset_index(drop=True)
        bv = bv.tail(common_n).reset_index(drop=True)
        ra = _log_returns(av)
        rb = _log_returns(bv)
        corr = _corr_last_window(ra, rb, len(ra))
        return {"corr": corr, "n": len(ra),
                "first": tail[0].isoformat(), "last": tail[-1].isoformat(),
                "pairs": [(t.isoformat(), t.isoformat()) for t in list(tail[1:6])]}
    except Exception as e:
        logger.debug(f"intraday_join_corr falhou: {e}")
        return {"corr": float("nan"), "n": 0, "first": None,
                "last": None, "pairs": []}


# PF-S2: shutdown cooperativo. stop_event (threading.Event do updater)
# é checado ENTRE estágios externos; nenhuma nova operação de rede começa
# após stop_requested. I/O sync já em andamento termina no próprio timeout;
# stop() faz join bounded e retorna clean=False se ainda viva. Sem matar thread.
def _stopped(stop_event) -> bool:
    try:
        return stop_event is not None and stop_event.is_set()
    except Exception:
        return False


def _sleep_coop(seconds: float, stop_event) -> bool:
    """Sleep interrompível por stop. Retorna True se stop foi pedido."""
    end = time.monotonic() + max(0.0, float(seconds))
    while True:
        if _stopped(stop_event):
            return True
        remaining = end - time.monotonic()
        if remaining <= 0:
            return _stopped(stop_event)
        time.sleep(min(0.2, remaining))


# F5-C7: dimensionamento da aquisição (sem magic number).
# target_returns -> closes necessários -> dias corridos com margem explícita:
#   sessões = target + 2 (+1 close p/ returns, +1 sessão comida pela exclusão
#   do dia da decisão); corridos = ceil(sessões * 7/5) [semana útil] +
#   HOLIDAY_MARGIN_DAYS [US ~9 feriados/ano; 12 dias ≈ 8-9 sessões cobre o
#   cluster Natal/Ano-Novo + dispersos em qualquer janela de ~4,5 meses].
# O fetch maior NÃO garante n=target; o contrato segue n=min(target, real).
TRADING_WEEK_RATIO = 7 / 5
HOLIDAY_MARGIN_DAYS = 12


def _fetch_calendar_lookback(target_returns: int) -> str:
    """Lookback 'Nd' derivado do alvo: 30->57d, 90->141d."""
    sessions_needed = int(target_returns) + 2
    calendar_days = math.ceil(sessions_needed * TRADING_WEEK_RATIO)
    return f"{calendar_days + HOLIDAY_MARGIN_DAYS}d"


def _period_to_start_end(period: str) -> Optional[Tuple[str, str]]:
    """'Nd' -> (start, end) ISO p/ yfinance (aceita range arbitrário,
    ao contrário de period que só admite valores fechados)."""
    m = re.fullmatch(r"\s*(\d+)\s*d\s*", period or "")
    if not m:
        return None
    days = int(m.group(1))
    end = datetime.now(timezone.utc).date() + timedelta(days=1)
    start = end - timedelta(days=days)
    return start.isoformat(), end.isoformat()


def _fetch_with_instrument(name: str, period: str = "90d",
                           interval: str = "1d", stop_event=None):
    """Busca com fallbacks + instrumento efetivo. (df, ticker_ou_None)."""
    # Mesma ordem de tentativa de _fetch_yfinance_data_with_fallbacks,
    # expondo o ticker vencedor (instrumento efetivo p/ metadata).
    candidates = _FALLBACK_TICKERS.get(name, [name])
    range_se = _period_to_start_end(period)
    for ticker in candidates:
        if _stopped(stop_event):
            break
        try:
            import yfinance as yf
            from concurrent.futures import ThreadPoolExecutor

            def _fetch(t=ticker):
                if range_se is not None:
                    return yf.Ticker(t).history(
                        start=range_se[0], end=range_se[1],
                        interval=interval, raise_errors=False,
                    )
                return yf.Ticker(t).history(
                    period=period, interval=interval, raise_errors=False
                )

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_fetch)
                try:
                    df = future.result(timeout=15)
                except FuturesTimeoutError:
                    logger.warning(f"⏰ Timeout (15s) ao buscar {ticker} para {name}")
                    continue
            if df is None or df.empty:
                continue
            if 'Close' not in df.columns or df['Close'].isna().all():
                continue
            out = df.rename(columns={'Close': 'close'})[['close']].dropna()
            if len(out) >= 5:
                logger.info(f"✅ Sucesso: {ticker} forneceu {len(out)} pontos para {name}")
                return out, ticker
        except Exception as e:
            logger.debug(f"Erro ao buscar {ticker}: {e}")
            continue
    logger.warning(f"❌ Falha: nenhum ticker funcionou para {name} (candidatos={candidates})")
    return pd.DataFrame(), None


# ===============================
# Funções de coleta de dados
# ===============================

def _fetch_binance_klines(symbol: str, interval: str = "1h", limit: int = 720,
                          stop_event=None) -> pd.DataFrame:
    """
    Busca velas da Binance usando a API REST.
    
    Args:
        symbol: Par de trading (ex: BTCUSDT)
        interval: Intervalo das velas (1h, 4h, 1d, etc.)
        limit: Número de velas a buscar (máx 1000)
        stop_event: threading.Event opcional; se setado, nenhum retry novo
            começa e o backoff é abortado (retorna vazio).
        
    Returns:
        DataFrame com colunas: open_time, open, high, low, close, volume
    """
    import requests
    import time
    
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": min(limit, 1000)
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        if _stopped(stop_event):
            break
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not isinstance(data, list):
                logger.warning(f"Resposta inesperada da Binance: {data}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'qav', 'num_trades', 'tbbav', 'tbqav', 'ignore'
            ])
            
            # Converte tipos
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df = df.set_index('open_time')
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Tentativa {attempt + 1}/{max_retries} falhou: {e}")
            if attempt < max_retries - 1:
                if _sleep_coop(1 * (attempt + 1), stop_event):
                    break
        except Exception as e:
            logger.error(f"Erro inesperado: {e}")
            return pd.DataFrame()
    
    return pd.DataFrame()


# Tickers de fallback por ativo (mesmo padrão do macro_fetcher.py)
_FALLBACK_TICKERS: Dict[str, list[str]] = {
    "BTC-USD": ["BTC-USD"],                        # BTC principal (BTCUSD=X quebrado no yfinance 1.0)
    "DXY": ["DX-Y.NYB", "UUP"],                   # Dollar Index (ICE) / USD Bull ETF como proxy
    "NDX": ["QQQ", "^IXIC"],                      # ETF Nasdaq / Nasdaq Comp
    "SPX": ["SPY", "^GSPC"]                       # ETF S&P 500 / S&P 500 índice
}


def _fetch_yfinance_data_with_fallbacks(name: str, period: str = "90d", interval: str = "1d",
                                          stop_event=None) -> pd.DataFrame:
    """
    Busca dados históricos do yfinance com fallbacks robustos.
    
    Args:
        name: Nome do ativo (BTC-USD, DXY, NDX, SPX)
        period: Período de dados (ex: 30d, 90d, 1y)
        interval: Intervalo (1d, 1wk, 1mo)
        stop_event: ver _fetch_binance_klines (nenhum ticker novo após stop).
        
    Returns:
        DataFrame com dados históricos
    """
    # Delega (fonte única da lógica de retry/timeout); instrumento descartado.
    df, _ticker = _fetch_with_instrument(name, period=period, interval=interval,
                                         stop_event=stop_event)
    return df


def _fetch_yfinance_data(ticker: str, period: str = "30d", interval: str = "1d",
                         stop_event=None) -> pd.DataFrame:
    """
    Busca dados históricos do yfinance (compatibilidade com versão anterior).
    
    Args:
        ticker: Ticker do ativo (ex: BTC-USD, DXY, ^NDX)
        period: Período de dados (ex: 30d, 90d, 1y)
        interval: Intervalo (1d, 1wk, 1mo)
        
    Returns:
        DataFrame com dados históricos
    """
    # Mapeia tickers antigos para nomes novos
    ticker_mapping = {
        "^NDX": "NDX",
        "^GSPC": "SPX", 
        "^IXIC": "NDX"
    }
    
    name = ticker_mapping.get(ticker, ticker)
    return _fetch_yfinance_data_with_fallbacks(name, period, interval,
                                               stop_event=stop_event)


# ===============================
# Funções principais de correlação
# ===============================

def get_btc_eth_correlations(now_utc: Optional[datetime] = None,
                             stop_event=None) -> Dict[str, Any]:
    """
    Calcula correlações entre BTCUSDT e ETHUSDT usando velas 1h da Binance.
    
    Args:
        now_utc: Timestamp atual em UTC (opcional)
        stop_event: ver _fetch_binance_klines (nenhum fetch novo após stop;
            retorna status "cancelled").
        
    Returns:
        Dict com:
        - btc_eth_corr_7d: correlação dos últimos 7 dias (7*24 pontos)
        - btc_eth_corr_30d: correlação dos últimos 30 dias (30*24 pontos)
        - status: ok, failed ou cancelled
        - error: mensagem de erro (se aplicável)
    """
    if _stopped(stop_event):
        return {"status": "cancelled"}
    result: Dict[str, Any] = {
        "status": "ok",
        "btc_eth_corr_7d": float("nan"),
        "btc_eth_corr_30d": float("nan"),
        "correlation_method": CORR_METHOD,
        "correlation_contract_version": CORR_CONTRACT_VERSION,
        "btc_eth_instrument": "BINANCE:BTCUSDT/ETHUSDT_1h",
    }

    try:
        # F5-C7: UM fetch alimenta 7d e 30d. Limite = alvo (720 retornos) +1
        # close +1 candle aberto (excluído) +2 folga de borda (<=1000 da API).
        klines_limit = 24 * 30 + 4
        btc_df = _fetch_binance_klines("BTCUSDT", "1h", klines_limit,
                                       stop_event=stop_event)
        if _stopped(stop_event):
            return {"status": "cancelled"}
        eth_df = _fetch_binance_klines("ETHUSDT", "1h", klines_limit,
                                       stop_event=stop_event)

        if btc_df.empty or eth_df.empty:
            raise ValueError("Dados insuficientes da Binance")

        # F5-C: join por timestamp compartilhado, sem candle aberto,
        # retornos calculados DEPOIS do alinhamento (nunca positional cego).
        dec_ms = _decision_ms(now_utc)
        r7 = intraday_join_corr(btc_df, eth_df, 24 * 7, decision_ms=dec_ms)
        r30 = intraday_join_corr(btc_df, eth_df, 24 * 30, decision_ms=dec_ms)

        result["btc_eth_corr_7d"] = r7["corr"]
        result["btc_eth_corr_30d"] = r30["corr"]
        result["btc_eth_corr_7d_n"] = r7["n"]
        result["btc_eth_corr_30d_n"] = r30["n"]
        # Observabilidade da heurística de lookback (sem alterar o cálculo).
        for _label, _r, _t in (("btc_eth_corr_7d", r7, 24 * 7),
                               ("btc_eth_corr_30d", r30, 24 * 30)):
            if _r["n"] < _t:
                logger.warning("corr window short: %s n=%d target=%d",
                               _label, _r["n"], _t)

        logger.info(f"Correlações BTC/ETH calculadas: 7d={result['btc_eth_corr_7d']:.4f} (n={r7['n']}), 30d={result['btc_eth_corr_30d']:.4f} (n={r30['n']})")
        
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        logger.error(f"Erro ao calcular correlações BTC/ETH: {e}")
    
    return result


def get_btc_macro_correlations(now_utc: Optional[datetime] = None,
                               stop_event=None) -> Dict[str, Any]:
    """
    Calcula correlações entre BTC e ativos macro (DXY, NDX) usando yfinance.
    
    Args:
        now_utc: Timestamp atual em UTC (opcional)
        stop_event: ver _fetch_binance_klines (nenhum fetch novo após stop;
            retorna status "cancelled").
        
    Returns:
        Dict com:
        - btc_dxy_corr_30d: correlação BTC x DXY (30 dias)
        - btc_dxy_corr_90d: correlação BTC x DXY (90 dias)
        - btc_ndx_corr_30d: correlação BTC x NDX (30 dias)
        - dxy_return_5d: retorno DXY nos últimos 5 dias
        - dxy_return_20d: retorno DXY nos últimos 20 dias
        - status: ok, failed ou cancelled
        - error: mensagem de erro (se aplicável)
    """
    if _stopped(stop_event):
        return {"status": "cancelled"}
    result: Dict[str, Any] = {
        "status": "ok",
        "btc_dxy_corr_30d": float("nan"),
        "btc_dxy_corr_90d": float("nan"),
        "btc_ndx_corr_30d": float("nan"),
        "dxy_return_5d": float("nan"),
        "dxy_return_20d": float("nan"),
        "correlation_method": CORR_METHOD,
        "correlation_contract_version": CORR_CONTRACT_VERSION,
    }

    try:
        # F5-C7: UM fetch conservador por ativo alimenta 30 e 90 (sem rede dupla).
        # Profundidade derivada do maior alvo (90 retornos -> "141d"); a janela
        # efetiva continua sendo o tail (30/31 ou 90/91 closes compartilhados).
        lookback = _fetch_calendar_lookback(90)
        btc_df, _btc_ticker = _fetch_with_instrument("BTC-USD", period=lookback,
                                                     stop_event=stop_event)
        if _stopped(stop_event):
            return {"status": "cancelled"}
        dxy_df, dxy_ticker = _fetch_with_instrument("DXY", period=lookback,
                                                    stop_event=stop_event)
        if _stopped(stop_event):
            return {"status": "cancelled"}
        ndx_df, ndx_ticker = _fetch_with_instrument("NDX", period=lookback,
                                                    stop_event=stop_event)

        result["btc_dxy_instrument"] = dxy_ticker
        result["nasdaq_instrument"] = ndx_ticker
        result["nasdaq_role"] = "nasdaq_proxy"

        if btc_df.empty or dxy_df.empty:
            raise ValueError("Dados insuficientes do yfinance")

        # F5-C: shared-session (closes nas mesmas datas, só sessões com
        # availability <= decision; retornos DEPOIS do alinhamento).
        dec_date = _decision_date(now_utc)
        r30 = shared_session_corr(btc_df['close'], dxy_df['close'], 30,
                                  decision_date=dec_date)
        r90 = shared_session_corr(btc_df['close'], dxy_df['close'], 90,
                                  decision_date=dec_date)

        result["btc_dxy_corr_30d"] = r30["corr"]
        result["btc_dxy_corr_90d"] = r90["corr"]
        result["btc_dxy_corr_30d_n"] = r30["n"]
        result["btc_dxy_corr_90d_n"] = r90["n"]
        # Observabilidade da heurística de lookback (sem alterar o cálculo).
        for _label, _r, _t in (("btc_dxy_corr_30d", r30, 30),
                               ("btc_dxy_corr_90d", r90, 90)):
            if _r["n"] < _t:
                logger.warning("corr window short: %s n=%d target=%d",
                               _label, _r["n"], _t)

        # Retornos DXY sobre closes FECHADOS (barra do dia da decisão excluída).
        dxy_closed = _daily_by_date(dxy_df['close'])
        dxy_days = sorted(d for d in dxy_closed if d < dec_date)
        if len(dxy_days) >= 5:
            result["dxy_return_5d"] = float(
                (dxy_closed[dxy_days[-1]] / dxy_closed[dxy_days[-5]] - 1) * 100)
        if len(dxy_days) >= 20:
            result["dxy_return_20d"] = float(
                (dxy_closed[dxy_days[-1]] / dxy_closed[dxy_days[-20]] - 1) * 100)

        # Calcula correlação NASDAQ-proxy se dados disponíveis
        # (campo mantido por compatibilidade; instrumento real em metadata).
        if not ndx_df.empty:
            rn = shared_session_corr(btc_df['close'], ndx_df['close'], 30,
                                     decision_date=dec_date)
            result["btc_ndx_corr_30d"] = rn["corr"]
            result["btc_ndx_corr_30d_n"] = rn["n"]
            if rn["n"] < 30:
                logger.warning("corr window short: btc_ndx_corr_30d n=%d target=30",
                               rn["n"])

        logger.info(f"Correlações macro calculadas: DXY 30d={result['btc_dxy_corr_30d']:.4f} (n={r30['n']}), 90d={result['btc_dxy_corr_90d']:.4f} (n={r90['n']})")
        
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        logger.error(f"Erro ao calcular correlações macro: {e}")
    
    return result


def _calculate_correlation_regime(btc_dxy_corr: Optional[float]) -> str:
    """
    Calcula regime de correlação baseado na correlação BTC x DXY.
    
    Args:
        btc_dxy_corr: Correlação BTC x DXY (pode ser None)
        
    Returns:
        String: "CORRELATED", "DECORRELATED", "INVERSE" ou "UNKNOWN"
    """
    try:
        if btc_dxy_corr is None or pd.isna(btc_dxy_corr):
            return "UNKNOWN"
        
        # DXY correlação inversa esperada
        if btc_dxy_corr < -0.4:
            return "INVERSE"  # Correlação inversa forte
        elif abs(btc_dxy_corr) < 0.2:
            return "DECORRELATED"  # Baixa correlação
        else:
            return "CORRELATED"  # Correlação positiva ou fraca inversa
            
    except Exception as e:
        logger.error(f"Erro ao calcular regime de correlação: {e}")
        return "UNKNOWN"


def _calculate_macro_regime_simple(macro_data: Dict[str, Any]) -> str:
    """Calcula regime macro simplificado baseado nos dados disponíveis."""
    try:
        risk_score = 0
        factors = 0
        
        # VIX: > 25 = risk off, < 15 = risk on
        vix = macro_data.get("vix")
        if vix is not None:
            factors += 1
            if vix > 25:
                risk_score += 2
            elif vix < 15:
                risk_score -= 1
        
        # BTC Dominance: > 50% = risk off, < 40% = risk on
        btc_dom = macro_data.get("btc_dominance")
        if btc_dom is not None:
            factors += 1
            if btc_dom > 50:
                risk_score += 1
            elif btc_dom < 40:
                risk_score -= 1
        
        # Treasury Yields: subida = risk off
        treasury_10y = macro_data.get("treasury_10y")
        treasury_2y = macro_data.get("treasury_2y")
        if treasury_10y is not None and treasury_2y is not None:
            factors += 1
            # Spread como proxy de mudança
            spread = treasury_10y - treasury_2y
            if spread > 0.5:  # Yield curve steepening = risk off
                risk_score += 1
            elif spread < 0:  # Yield curve inversion = risk off
                risk_score += 2
        
        if factors == 0:
            return "UNKNOWN"
        
        avg_score = risk_score / factors
        
        if avg_score >= 1.0:
            return "RISK_OFF"
        elif avg_score <= -1.0:
            return "RISK_ON"
        else:
            return "TRANSITION"
            
    except Exception as e:
        logger.error(f"Erro ao calcular regime macro: {e}")
        return "UNKNOWN"


# ===============================
# Funções assíncronas seguras
# ===============================

def _is_event_loop_running() -> bool:
    """Verifica se há um event loop rodando no thread atual."""
    try:
        loop = asyncio.get_running_loop()
        return loop is not None
    except RuntimeError:
        return False


def _get_macro_data_sync() -> Dict[str, Any]:
    """
    Versão SÍNCRONA para obter dados macro.
    Retorna dict vazio em contexto async - dados serão obtidos via MacroUpdateService.
    """
    if not _MACRO_DATA_OK:
        return {}
    
    try:
        # Em contexto async, não tentamos buscar dados síncronos
        # O MacroUpdateService já atualiza em background
        # Retornamos vazio e as correlações básicas (BTC/ETH, DXY, NDX) já funcionam
        logger.debug("Macro data: contexto async detectado, usando valores já calculados")
        return {}
        
    except Exception as e:
        logger.warning(f"Erro ao obter macro data sync: {e}")
        return {}


async def _get_macro_data_async() -> Dict[str, Any]:
    """Função auxiliar para executar MacroDataProvider de forma assíncrona.

    PF-M2: fecha as sessões do loop efêmero antes dele terminar (opção A).
    Fecha SOMENTE o loop atual — nunca sessões de outros loops (ex:
    MacroUpdateService) — e nunca cria sessão em um loop para fechar/usar
    em outro: tudo acontece dentro deste mesmo loop efêmero.
    """
    if not _MACRO_DATA_OK:
        return {}

    try:
        provider = MacroDataProvider()
        try:
            return await provider.get_all_macro_data()
        finally:
            try:
                await provider.close_sessions_for_current_loop()
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Erro ao obter macro data async: {e}")
        return {}


def _run_async_safely(coro: Any, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
    """
    Executa uma coroutine de forma segura, detectando o contexto.
    
    Args:
        coro: Coroutine a executar
        timeout: Timeout em segundos
        
    Returns:
        Resultado da coroutine ou None se falhar
    """
    try:
        # Se já tem loop rodando, não podemos usar asyncio.run()
        if _is_event_loop_running():
            # Estamos dentro de um contexto async - usar abordagem síncrona
            logger.debug("Event loop detectado - usando fallback síncrono")
            return _get_macro_data_sync()
        
        # Sem loop rodando - podemos criar um novo
        return asyncio.run(asyncio.wait_for(coro, timeout=timeout))
        
    except asyncio.TimeoutError:
        logger.warning(f"⚠️ Timeout ao executar coroutine ({timeout}s)")
        return None
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            logger.debug("Event loop conflict - usando fallback síncrono")
            return _get_macro_data_sync()
        logger.error(f"RuntimeError: {e}")
        return None
    except Exception as e:
        logger.error(f"Erro ao executar coroutine: {e}")
        return None


def get_enhanced_cross_asset_correlations(now_utc: Optional[datetime] = None,
                                            stop_event=None) -> Dict[str, Any]:
    """
    Calcula correlações cross-asset ENHANCED com todas as novas métricas.
    
    Inclui:
    - BTC x ETH (crypto)
    - BTC x DXY, NDX (macro tradicional)  
    - BTC x VIX, Gold, Oil, Treasury Yields (novo)
    - Crypto Dominance
    - Regime Detection
    
    Args:
        now_utc: Timestamp atual em UTC (opcional)
        stop_event: ver _fetch_binance_klines. Após stop, nenhuma nova
            operação externa começa; retorna status "cancelled".
        
    Returns:
        Dict com todas as métricas cross-asset enhanced
    """
    if _stopped(stop_event):
        return {"status": "cancelled"}
    result: Dict[str, Any] = {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat() if now_utc is None else now_utc.isoformat()
    }
    
    # 1. CORRELAÇÕES TRADICIONAIS
    # Crypto (Binance)
    crypto_corr = get_btc_eth_correlations(now_utc, stop_event=stop_event)
    if crypto_corr.get("status") == "cancelled" or _stopped(stop_event):
        return {"status": "cancelled"}
    if crypto_corr.get("status") == "ok":
        result.update(crypto_corr)
    else:
        result["status"] = "partial"
        result["crypto_error"] = crypto_corr.get("error")
    
    # Macro tradicional (yfinance)
    macro_corr = get_btc_macro_correlations(now_utc, stop_event=stop_event)
    if macro_corr.get("status") == "cancelled" or _stopped(stop_event):
        return {"status": "cancelled"}
    if macro_corr.get("status") == "ok":
        result.update(macro_corr)
    else:
        result["status"] = "partial"
        result["macro_error"] = macro_corr.get("error")
    
    # 2. NOVAS MÉTRICAS CROSS-ASSET via MacroDataProvider
    if _stopped(stop_event):
        return {"status": "cancelled"}
    if _MACRO_DATA_OK:
        try:
            # CORREÇÃO: Usar abordagem segura que detecta contexto
            macro_data: Optional[Dict[str, Any]] = None
            
            if _is_event_loop_running():
                # Contexto async - usar versão síncrona/cache
                macro_data = _get_macro_data_sync()
            else:
                # Contexto sync - pode usar asyncio.run
                # Timeout aumentado para 30s para permitir múltiplas chamadas de API
                macro_data = _run_async_safely(_get_macro_data_async(), timeout=30.0)
            
            # TRATAR None de forma segura
            if macro_data is None or not macro_data:
                logger.debug("Enhanced correlations: usando valores de cache ou padrão")
                macro_data = {}
            
            # VIX metrics
            vix_value = macro_data.get("vix")
            if vix_value is not None:
                result["vix_current"] = vix_value
                result["vix_change_1d"] = None  # Calcular se necessário
            
            # Treasury Yields
            treasury_10y = macro_data.get("treasury_10y")
            if treasury_10y is not None:
                result["us10y_yield"] = treasury_10y
                
            treasury_2y = macro_data.get("treasury_2y")
            if treasury_2y is not None:
                result["us2y_yield"] = treasury_2y
                
            yield_spread = macro_data.get("yield_spread")
            if yield_spread is not None:
                result["us10y_change_1d"] = yield_spread  # Proxy
            
            # Crypto Dominance
            btc_dominance = macro_data.get("btc_dominance")
            if btc_dominance is not None:
                result["btc_dominance"] = btc_dominance
                
            eth_dominance = macro_data.get("eth_dominance")
            if eth_dominance is not None:
                result["eth_dominance"] = eth_dominance
                
            usdt_dominance = macro_data.get("usdt_dominance")
            if usdt_dominance is not None:
                result["usdt_dominance"] = usdt_dominance
            
            # DXY (Dollar Index)
            dxy_value = macro_data.get("dxy")
            if dxy_value is not None:
                result["dxy_current"] = dxy_value
            
            # Commodities
            gold_value = macro_data.get("gold")
            if gold_value is not None:
                result["gold_price"] = gold_value
                result["gold_change_1d"] = None  # Calcular se necessário
                
            oil_value = macro_data.get("oil")
            if oil_value is not None:
                result["oil_price"] = oil_value
                result["oil_change_1d"] = None  # Calcular se necessário
            
            # NOVAS CORRELAÇÕES (placeholders)
            btc_returns: Optional[pd.Series] = None
            
            # BTC x VIX correlation (se dados disponíveis)
            if vix_value is not None and not _stopped(stop_event):
                try:
                    btc_df = _fetch_yfinance_data("BTC-USD", period="30d",
                                                  stop_event=stop_event)
                    if not btc_df.empty:
                        btc_returns = _log_returns(btc_df['close'])
                        result["btc_vix_corr_30d"] = None  # Placeholder
                except Exception:
                    pass
            
            # BTC x Gold correlation
            if gold_value is not None and not _stopped(stop_event):
                if btc_returns is None:
                    try:
                        btc_df = _fetch_yfinance_data("BTC-USD", period="30d",
                                                      stop_event=stop_event)
                        if not btc_df.empty:
                            btc_returns = _log_returns(btc_df['close'])
                    except Exception:
                        pass
                result["btc_gold_corr_30d"] = None  # Placeholder
            
            # BTC x Oil correlation
            if oil_value is not None:
                result["btc_oil_corr_30d"] = None  # Placeholder
            
            # BTC x Treasury Yields correlation
            if treasury_10y is not None:
                result["btc_yields_corr_30d"] = None  # Placeholder
            
            # Dominance change (7d): SEM medição real neste pipeline.
            # B-P0-4: chave omitida (missing permanece missing); o 0.0
            # anterior era placeholder e se passava por variação observada.
            # (ver dívida temporal no docstring do módulo).
            
            # Correlation Regime (baseado em BTC x DXY)
            btc_dxy_corr = result.get("btc_dxy_corr_30d")
            # Garantir que é float ou None antes de passar para a função
            if isinstance(btc_dxy_corr, (int, float)) and not pd.isna(btc_dxy_corr):
                result["correlation_regime"] = _calculate_correlation_regime(float(btc_dxy_corr))
            else:
                result["correlation_regime"] = "UNKNOWN"
            
            # Macro Regime (baseado em dados disponíveis)
            result["macro_regime"] = _calculate_macro_regime_simple(macro_data)
            
            logger.info(f"✅ Enhanced cross-asset correlations calculadas: {len([k for k in result.keys() if not k.startswith('_')])} features")
            
        except Exception as e:
            logger.error(f"Erro ao calcular enhanced correlations: {e}")
            result["enhanced_error"] = str(e)
            if result["status"] == "ok":
                result["status"] = "partial"
    else:
        logger.warning("macro_data_provider não disponível, pulando enhanced metrics")
        result["enhanced_status"] = "unavailable"
    
    return result


_CORR_CACHE: Dict[str, Any] = {}
_CORR_CACHE_TTL = 300  # 5 minutos — correlações diárias não mudam a cada janela


def get_all_correlations(now_utc: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Calcula todas as correlações cross-asset para BTCUSDT.
    Usa cache com TTL de 5 minutos para evitar requisições redundantes.
    """
    global _CORR_CACHE
    now_ts = time.monotonic()
    cached = _CORR_CACHE.get("result")
    cached_at = _CORR_CACHE.get("cached_at", 0.0)

    if cached is not None and (now_ts - cached_at) < _CORR_CACHE_TTL:
        logger.debug("cross_asset_correlations: cache hit (age=%.0fs)", now_ts - cached_at)
        return cached

    result = get_enhanced_cross_asset_correlations(now_utc)
    _CORR_CACHE["result"] = result
    _CORR_CACHE["cached_at"] = now_ts
    return result


# Função principal para integração com ml_features
get_cross_asset_features = get_all_correlations


def build_cross_asset_context(cross_asset: Dict[str, Any]) -> Dict[str, Any]:
    """
    Constrói contexto cross-asset para uso em decisões de trading.
    
    Args:
        cross_asset: Dict com dados de correlações cross-asset
        
    Returns:
        Dict estruturado com análise de contexto
    """
    corr_dxy = cross_asset.get("btc_dxy_corr_30d")
    dxy_ret_5d = cross_asset.get("dxy_return_5d")

    # Classificar regime de correlação BTC x DXY
    if corr_dxy is None:
        regime = "UNKNOWN"
    elif corr_dxy < -0.4:
        regime = "STRONG_INVERSE"
    elif corr_dxy > 0.2:
        regime = "POSITIVE_OR_WEAK_INVERSE"
    else:
        regime = "NEUTRAL"

    # Classificar tendência recente do DXY (5 dias)
    if dxy_ret_5d is None:
        dxy_trend_5d = "UNKNOWN"
    elif dxy_ret_5d > 0.005:
        dxy_trend_5d = "UP"
    elif dxy_ret_5d < -0.005:
        dxy_trend_5d = "DOWN"
    else:
        dxy_trend_5d = "FLAT"

    # Efeito esperado no BTC, dado regime inverso
    if regime == "STRONG_INVERSE":
        if dxy_trend_5d == "UP":
            expected_effect = "HEADWIND"   # vento contra BTC
        elif dxy_trend_5d == "DOWN":
            expected_effect = "TAILWIND"   # vento a favor BTC
        else:
            expected_effect = "NEUTRAL"
    else:
        expected_effect = "NEUTRAL"

    return {
        "dxy_link": {
            "btc_dxy_corr_30d": corr_dxy,
            "btc_dxy_corr_90d": cross_asset.get("btc_dxy_corr_90d"),
            "dxy_trend_5d": dxy_trend_5d,
            "relationship_regime": regime,
            "expected_effect_on_btc": expected_effect
        },
        "crypto_sector": {
            "btc_eth_corr_30d": cross_asset.get("btc_eth_corr_30d")
        },
        "macro_links": {
            "btc_ndx_corr_30d": cross_asset.get("btc_ndx_corr_30d")
        }
    }


if __name__ == "__main__":
    # Teste básico
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*80)
    print("TESTE DE CROSS_ASSET_CORRELATIONS")
    print("="*80 + "\n")
    
    # Testa correlações crypto
    print("Testando correlações BTC/ETH...")
    crypto_result = get_btc_eth_correlations()
    print(f"  Status: {crypto_result['status']}")
    if crypto_result['status'] == 'ok':
        btc_eth_7d = crypto_result['btc_eth_corr_7d']
        btc_eth_30d = crypto_result['btc_eth_corr_30d']
        print(f"  7d correlation: {btc_eth_7d:.4f}" if not pd.isna(btc_eth_7d) else "  7d correlation: N/A")
        print(f"  30d correlation: {btc_eth_30d:.4f}" if not pd.isna(btc_eth_30d) else "  30d correlation: N/A")
    else:
        print(f"  Error: {crypto_result.get('error')}")
    
    # Testa correlações macro
    print("\nTestando correlações macro...")
    macro_result = get_btc_macro_correlations()
    print(f"  Status: {macro_result['status']}")
    if macro_result['status'] == 'ok':
        btc_dxy_30d = macro_result['btc_dxy_corr_30d']
        btc_dxy_90d = macro_result['btc_dxy_corr_90d']
        dxy_ret_5d = macro_result['dxy_return_5d']
        dxy_ret_20d = macro_result['dxy_return_20d']
        
        print(f"  BTC x DXY (30d): {btc_dxy_30d:.4f}" if not pd.isna(btc_dxy_30d) else "  BTC x DXY (30d): N/A")
        print(f"  BTC x DXY (90d): {btc_dxy_90d:.4f}" if not pd.isna(btc_dxy_90d) else "  BTC x DXY (90d): N/A")
        print(f"  DXY return (5d): {dxy_ret_5d:.2f}%" if not pd.isna(dxy_ret_5d) else "  DXY return (5d): N/A")
        print(f"  DXY return (20d): {dxy_ret_20d:.2f}%" if not pd.isna(dxy_ret_20d) else "  DXY return (20d): N/A")
    else:
        print(f"  Error: {macro_result.get('error')}")
    
    # Testa função combinada
    print("\nTestando função combinada...")
    all_result = get_all_correlations()
    print(f"  Status: {all_result['status']}")
    print(f"  Total features: {len([k for k in all_result.keys() if not k.startswith('_')])}")
    
    print("\n" + "="*80)
    print("TESTE CONCLUIDO")
    print("="*80 + "\n")