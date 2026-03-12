# macro_fetcher.py
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import os
import json

try:
    import config as app_config  # opcional para ler chave do Alpha Vantage
except Exception:  # pragma: no cover
    app_config = None

try:
    import yfinance as yf  # fonte principal
    _YF_OK = True
except Exception as e:
    _YF_OK = False
    logging.warning(f"yfinance indisponível: {e}. Macro vai retornar status=failed.")

# ---------------------------------------------------------------------------
#  Suporte opcional ao Alpha Vantage para fallback quando yfinance falhar
#  Requer definir a variável de ambiente ALPHAVANTAGE_API_KEY ou em config.py
# ---------------------------------------------------------------------------
import requests  # usado somente quando _YF_OK falha

# Carrega chave API do AlphaVantage de env ou config
_AV_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
if not _AV_API_KEY and app_config is not None:
    _AV_API_KEY = getattr(app_config, "ALPHAVANTAGE_API_KEY", None)

# Possíveis mapeamentos de símbolos para AlphaVantage (índices, commodities)
_ALPHA_CANDIDATES = {
    "DXY": ["DX-Y.NYB", "DXY"],
    "GOLD": ["XAUUSD", "GOLD"],
    "SPX": ["SPX", "SPY"],
    "NDX": ["IXIC", "QQQ", "NDX"],
}

def _dl_alphavantage(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Baixa dados diários do AlphaVantage para um símbolo.
    Retorna dicionário com close, pct_change_1d e timestamp ou None em caso de erro.
    """
    if not _AV_API_KEY:
        return None
    # Usa endpoint TIME_SERIES_DAILY. Poderá ser mudado para outras funções conforme necessidade.
    url = (
        "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
        f"&symbol={symbol}&apikey={_AV_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        # A API devolve os dados sob a chave 'Time Series (Daily)'
        ts = data.get("Time Series (Daily)")
        if not ts:
            return None
        # ordena as datas para extrair o último dia disponível
        dates = sorted(ts.keys())
        if not dates:
            return None
        last_day = dates[-1]
        last_close = float(ts[last_day].get("4. close"))
        # calcula variação em relação ao dia anterior, se disponível
        pct = None
        if len(dates) > 1:
            prev_day = dates[-2]
            prev_close = float(ts[prev_day].get("4. close"))
            pct = ((last_close - prev_close) / prev_close) * 100.0 if prev_close else None
        # normaliza timestamp para ISO (data + tempo 00:00 UTC)
        ts_iso = f"{last_day}T00:00:00Z"
        return {
            "close": last_close,
            "pct_change_1d": pct,
            "timestamp": ts_iso,
        }
    except Exception:
        return None

# --- TICKERS DE BACKUP POR ATIVO ---
# Obs.: yfinance às vezes muda cookies/rotas; manter múltiplas opções ajuda.
_FALLBACKS: Dict[str, List[str]] = {
    "DXY":   ["DX-Y.NYB", "DX=F", "DXY"],        # Dollar Index (ICE) / futures / índice
    "GOLD":  ["GC=F", "XAUUSD=X", "GOLD"],       # Ouro futures / spot
    "SPX":   ["^GSPC", "SPY"],                   # S&P 500 índice / ETF
    "NDX":   ["^IXIC", "QQQ"],                   # Nasdaq Comp / ETF
}

_PERIODS = ["2d", "5d", "1mo"]
# intervals aceitos variam; alguns símbolos não suportam 1h
_INTERVALS = ["1h", "1d"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dl_yf(ticker: str, period: str, interval: str):
    """Wrapper de download: retorna df (ou None) sem explodir exceção."""
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
        if df is not None and not df.empty and "Close" in df.columns:
            return df
        return None
    except Exception:
        return None


def _extract_snapshot(df) -> Optional[Dict[str, Any]]:
    try:
        last = df.iloc[-1]
        close = float(last["Close"])
        # pct change diária se possível
        if len(df) >= 2:
            prev = float(df["Close"].iloc[-2])
            pct = ((close - prev) / prev) * 100.0 if prev else 0.0
        else:
            pct = None
        ts = df.index[-1]
        # normaliza timestamp ISO
        if hasattr(ts, "tz_convert"):
            ts = ts.tz_convert("UTC")
        ts_iso = ts.isoformat() if hasattr(ts, "isoformat") else _now_iso()
        return {"close": close, "pct_change_1d": pct, "timestamp": ts_iso}
    except Exception:
        return None


def _try_tickers(name: str, candidates: List[str]) -> Dict[str, Any]:
    """
    Itera tickers candidatos e períodos/intervalos via yfinance.
    Em caso de falha, tenta fallback via AlphaVantage se configurado.
    Retorna payload com status ok/failed sem levantar exceção.
    """
    # 1) Tenta yfinance para cada candidato e combinação de períodos/intervalos
    for symbol in candidates:
        for period in _PERIODS:
            for interval in _INTERVALS:
                df = _dl_yf(symbol, period=period, interval=interval) if _YF_OK else None
                if df is None:
                    continue
                snap = _extract_snapshot(df)
                if not snap:
                    continue
                snap["symbol_used"] = symbol
                snap["status"] = "ok"
                snap["source"] = "yfinance"
                snap["period"] = period
                snap["interval"] = interval
                return snap

    # 2) Fallback via AlphaVantage se disponível
    if _AV_API_KEY:
        alpha_candidates = _ALPHA_CANDIDATES.get(name, candidates)
        for symbol in alpha_candidates:
            snap = _dl_alphavantage(symbol)
            if not snap:
                continue
            # incorpora campos extras
            snap["symbol_used"] = symbol
            snap["status"] = "ok"
            snap["source"] = "alphavantage"
            return snap

    # 3) Sem dados disponíveis
    logging.warning(f"[Macro] {name}: dados indisponíveis em yfinance e AlphaVantage (candidatos={candidates})")
    return {
        "status": "failed",
        "symbol_used": None,
        "close": None,
        "pct_change_1d": None,
        "timestamp": _now_iso(),
        "source": "yfinance",
    }


def safe_get_macro_snapshots() -> Dict[str, Any]:
    """
    Snapshot macro robusto (nunca levanta exceção).
    Keys: DXY, GOLD, SPX, NDX.
    """
    out: Dict[str, Any] = {}

    for name in ["DXY", "GOLD", "SPX", "NDX"]:
        candidates = _FALLBACKS.get(name, [])
        if not _YF_OK or not candidates:
            logging.warning(f"[Macro] {name}: yfinance indisponível ou sem candidatos.")
            out[name] = {
                "status": "failed",
                "symbol_used": None,
                "close": None,
                "pct_change_1d": None,
                "timestamp": _now_iso(),
                "source": "yfinance",
            }
            continue

        out[name] = _try_tickers(name, candidates)

    return out
