# macro_fetcher.py
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

try:
    import yfinance as yf
    _YF_OK = True
except Exception as e:
    _YF_OK = False
    logging.warning(f"yfinance indisponível: {e}. Macro vai retornar status=failed.")

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
    Itera tickers candidatos e períodos/intervalos.
    Retorna payload com status ok/failed sem levantar exceção.
    """
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

    # falhou nos candidatos
    logging.warning(f"[Macro] {name}: dados indisponíveis em yfinance (candidatos={candidates})")
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
