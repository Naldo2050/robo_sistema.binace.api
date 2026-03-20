"""
WindowState v1.0 — Fonte Única de Verdade para dados de janela.

REGRA DE OURO:
- Dados são ESCRITOS aqui UMA VEZ pelo módulo responsável
- Todos os outros módulos LEEM daqui
- NINGUÉM recalcula o que já está no WindowState

Módulos escritores (ÚNICOS permitidos):
- Pipeline/TradeProcessor → price, volumes, delta, num_trades
- IndicatorEngine/MultiTF → rsi, bb_width, macd, atr
- ContextCollector → dxy, sp500, vix, fear_greed
- OrderBookAnalyzer → bid_depth, ask_depth, imbalance, spread
- FlowAnalyzer → cvd, sector_flow, absorption
- DerivativesCollector → funding_rate, open_interest, long_short_ratio

Todos os demais módulos são LEITORES.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class PriceData:
    """Dados de preço — escritos APENAS pelo Pipeline"""
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    vwap: float = 0.0
    twap: float = 0.0


@dataclass
class VolumeData:
    """Dados de volume — escritos APENAS pelo Pipeline"""
    total: float = 0.0
    buy: float = 0.0
    sell: float = 0.0
    delta: float = 0.0  # = buy - sell (calculado na validação)
    num_trades: int = 0
    avg_trade_size: float = 0.0

    def validate(self) -> List[str]:
        errors = []
        # Invariante: buy + sell = total (tolerância 0.01)
        vol_sum = self.buy + self.sell
        if abs(vol_sum - self.total) > 0.01 and self.total > 0:
            errors.append(
                f"VOLUME_SUM_MISMATCH: buy({self.buy:.4f}) + "
                f"sell({self.sell:.4f}) = {vol_sum:.4f} != "
                f"total({self.total:.4f})"
            )
        # Invariante: delta = buy - sell
        calc_delta = self.buy - self.sell
        if abs(calc_delta - self.delta) > 0.01 and self.total > 0:
            errors.append(
                f"DELTA_MISMATCH: buy-sell({calc_delta:.4f}) != "
                f"delta({self.delta:.4f})"
            )
        # Buy e sell não podem ser negativos
        if self.buy < 0:
            errors.append(f"NEGATIVE_BUY: {self.buy}")
        if self.sell < 0:
            errors.append(f"NEGATIVE_SELL: {self.sell}")
        return errors


@dataclass
class IndicatorData:
    """Indicadores técnicos — escritos APENAS pelo IndicatorEngine"""
    # RSI do timeframe de referência (15m por default)
    rsi: float = 50.0
    rsi_source_tf: str = "15m"

    # Bollinger Bands (NOTA: underscore SIMPLES)
    bb_upper: float = 0.0
    bb_lower: float = 0.0
    bb_width: float = 0.0  # NÃO bb__width

    # Volatilidade REAL (do timeframe diário)
    realized_vol: float = 0.0
    realized_vol_source: str = "1d"

    # Outros
    atr: float = 0.0
    macd: float = 0.0
    macd_signal: float = 0.0
    adx: float = 0.0
    cci: float = 0.0

    def validate(self) -> List[str]:
        errors = []
        if self.rsi < 0 or self.rsi > 100:
            errors.append(f"RSI_OUT_OF_RANGE: {self.rsi}")
        if self.bb_width < 0:
            errors.append(f"BB_WIDTH_NEGATIVE: {self.bb_width}")
        if self.realized_vol <= 0:
            errors.append(f"REALIZED_VOL_ZERO_OR_NEGATIVE: {self.realized_vol}")
        return errors


@dataclass
class OrderBookData:
    """Dados do order book — escritos APENAS pelo OrderBookAnalyzer"""
    mid_price: float = 0.0
    spread_bps: float = 0.0
    bid_depth_usd: float = 0.0
    ask_depth_usd: float = 0.0
    imbalance: float = 0.0  # -1 a +1
    pressure: float = 0.0
    data_source: str = "live"
    is_valid: bool = True

    def validate(self) -> List[str]:
        errors = []
        if self.bid_depth_usd < 0:
            errors.append(f"NEGATIVE_BID_DEPTH: {self.bid_depth_usd}")
        if self.ask_depth_usd < 0:
            errors.append(f"NEGATIVE_ASK_DEPTH: {self.ask_depth_usd}")
        if abs(self.imbalance) > 1.0:
            errors.append(f"IMBALANCE_OUT_OF_RANGE: {self.imbalance}")
        return errors


@dataclass
class FlowData:
    """Dados de fluxo — escritos APENAS pelo FlowAnalyzer"""
    cvd: float = 0.0
    flow_imbalance: float = 0.0
    buy_sell_ratio: float = 1.0
    pressure_label: str = "NEUTRAL"  # SLIGHT_BUY, SLIGHT_SELL, etc.

    # Setores
    retail_buy: float = 0.0
    retail_sell: float = 0.0
    retail_delta: float = 0.0
    mid_buy: float = 0.0
    mid_sell: float = 0.0
    mid_delta: float = 0.0
    whale_buy: float = 0.0
    whale_sell: float = 0.0
    whale_delta: float = 0.0

    # Absorção
    absorption_index: float = 0.0
    absorption_label: str = "Neutra"
    buyer_strength: float = 0.0
    seller_exhaustion: float = 0.0


@dataclass
class MacroData:
    """Dados macro — escritos APENAS pelo ContextCollector"""
    dxy: Optional[float] = None
    dxy_source: str = ""  # "FRED", "yfinance", etc.
    sp500: Optional[float] = None
    sp500_source: str = ""
    nasdaq: Optional[float] = None
    gold: Optional[float] = None
    wti: Optional[float] = None
    vix: Optional[float] = None
    tnx: Optional[float] = None
    fear_greed: Optional[int] = None
    fear_greed_label: str = ""
    btc_dominance: Optional[float] = None
    btc_dominance_source: str = ""

    def validate(self) -> List[str]:
        errors = []
        if self.dxy is not None and (self.dxy < 80 or self.dxy > 130):
            errors.append(f"DXY_SUSPICIOUS: {self.dxy} (expected 80-130)")
        if self.sp500 is not None and self.sp500 < 1000:
            errors.append(
                f"SP500_SUSPICIOUS: {self.sp500} "
                f"(expected >1000, got ETF price?)"
            )
        if self.nasdaq is not None and self.nasdaq < 1000:
            errors.append(
                f"NASDAQ_SUSPICIOUS: {self.nasdaq} "
                f"(expected >1000, got ETF price?)"
            )
        if self.btc_dominance is not None:
            if self.btc_dominance < 30 or self.btc_dominance > 80:
                errors.append(
                    f"BTC_DOM_SUSPICIOUS: {self.btc_dominance}% "
                    f"(expected 30-80%)"
                )
        if self.fear_greed is not None:
            if self.fear_greed < 0 or self.fear_greed > 100:
                errors.append(
                    f"FEAR_GREED_OUT_OF_RANGE: {self.fear_greed}"
                )
        return errors


@dataclass
class DerivativesData:
    """Dados de derivativos — escritos APENAS pelo DerivativesCollector"""
    btc_open_interest: float = 0.0
    btc_open_interest_usd: float = 0.0
    btc_funding_rate: Optional[float] = None
    btc_long_short_ratio: float = 1.0
    eth_open_interest: float = 0.0
    eth_funding_rate: Optional[float] = None
    eth_long_short_ratio: float = 1.0


@dataclass
class OnChainData:
    """Dados on-chain — escritos APENAS pelo OnChainCollector"""
    difficulty: float = 0.0
    active_addresses: int = 0
    mempool_size: int = 0
    fees_fastest_sat_vb: int = 0
    total_btc_sent_24h: float = 0.0
    total_fees_btc_24h: Optional[float] = None  # None se inválido

    def validate(self) -> List[str]:
        errors = []
        if (self.total_fees_btc_24h is not None
                and self.total_fees_btc_24h < 0):
            errors.append(
                f"NEGATIVE_FEES: {self.total_fees_btc_24h} "
                f"(fees cannot be negative)"
            )
            self.total_fees_btc_24h = None  # Auto-corrige
        return errors


@dataclass
class WindowState:
    """
    FONTE UNICA DE VERDADE para uma janela de análise.

    Uso:
        state = WindowState(symbol="BTCUSDT", window_number=3)

        # Pipeline preenche (WRITE):
        state.price.close = 70544.7
        state.volume.buy = 3.225
        state.volume.sell = 2.761
        state.volume.total = 5.986
        state.volume.delta = 0.464

        # Indicadores preenchem (WRITE):
        state.indicators.rsi = 64.65  # DO MULTI_TF REAL
        state.indicators.bb_width = 0.000185  # REAL, não 0.04
        state.indicators.realized_vol = 0.0315  # DO DAILY REAL

        # TODOS os outros módulos (READ):
        rsi = state.indicators.rsi  # Lê, não recalcula
        vol = state.volume.buy      # Lê, não recalcula

        # Validação antes de enviar para IA:
        errors = state.validate_all()
        if errors:
            logger.error(f"HALT: {errors}")
            return  # NÃO envia dados corrompidos
    """

    # Identificação
    symbol: str = "BTCUSDT"
    window_number: int = 0
    timestamp_utc: Optional[datetime] = None
    epoch_ms: int = 0

    # Sub-estados (cada um escrito por UM módulo)
    price: PriceData = field(default_factory=PriceData)
    volume: VolumeData = field(default_factory=VolumeData)
    indicators: IndicatorData = field(default_factory=IndicatorData)
    orderbook: OrderBookData = field(default_factory=OrderBookData)
    flow: FlowData = field(default_factory=FlowData)
    macro: MacroData = field(default_factory=MacroData)
    derivatives: DerivativesData = field(default_factory=DerivativesData)
    onchain: OnChainData = field(default_factory=OnChainData)

    # Controle de quem já escreveu
    _writers: Dict[str, bool] = field(default_factory=lambda: {
        'pipeline': False,
        'indicators': False,
        'orderbook': False,
        'flow': False,
        'macro': False,
        'derivatives': False,
        'onchain': False,
    })

    def mark_written(self, module: str):
        """Marca que um módulo já escreveu seus dados"""
        if module not in self._writers:
            logger.warning(
                f"Modulo desconhecido tentou escrever: {module}"
            )
            return
        if self._writers[module]:
            logger.warning(
                f"Modulo '{module}' tentou escrever DUAS VEZES "
                f"na janela {self.window_number}!"
            )
        self._writers[module] = True

    def validate_all(self) -> List[str]:
        """
        Validação COMPLETA antes de enviar para IA.
        Retorna lista de erros. Lista vazia = tudo OK.
        """
        all_errors = []

        # Validar cada sub-estado
        all_errors.extend(self.volume.validate())
        all_errors.extend(self.indicators.validate())
        all_errors.extend(self.orderbook.validate())
        all_errors.extend(self.macro.validate())
        all_errors.extend(self.onchain.validate())

        # Verificar que módulos críticos escreveram
        critical = ['pipeline', 'indicators', 'orderbook']
        for mod in critical:
            if not self._writers.get(mod, False):
                all_errors.append(
                    f"MISSING_WRITER: modulo '{mod}' nao escreveu "
                    f"na janela {self.window_number}"
                )

        if all_errors:
            logger.error(
                f"WindowState VALIDATION FAILED "
                f"(janela {self.window_number}): "
                f"{len(all_errors)} erros"
            )
            for err in all_errors:
                logger.error(f"   -> {err}")
        else:
            logger.info(
                f"WindowState VALID "
                f"(janela {self.window_number})"
            )

        return all_errors

    def get_ml_features(self) -> Dict[str, float]:
        """
        Retorna features para o XGBoost.
        FONTE UNICA — nenhum outro módulo deve montar este dict.
        """
        return {
            'price_close': self.price.close,
            'return_1': self._safe_return(),
            'return_5': 0.0,  # Será preenchido com histórico
            'return_10': 0.0,
            'bb_upper': self.indicators.bb_upper,
            'bb_lower': self.indicators.bb_lower,
            'bb_width': self.indicators.bb_width,
            'rsi': self.indicators.rsi,
            'volume_ratio': self._safe_volume_ratio(),
        }

    def _safe_return(self) -> float:
        """Calcula return_1 com segurança"""
        if self.price.open > 0 and self.price.close > 0:
            return (self.price.close - self.price.open) / self.price.open
        return 0.0

    def _safe_volume_ratio(self) -> float:
        """Volume ratio com segurança"""
        if self.volume.sell > 0:
            return self.volume.buy / self.volume.sell
        return 1.0

    def to_summary(self) -> Dict[str, Any]:
        """Resumo para logging"""
        return {
            'window': self.window_number,
            'price': self.price.close,
            'vol_total': self.volume.total,
            'vol_buy': self.volume.buy,
            'vol_sell': self.volume.sell,
            'delta': self.volume.delta,
            'rsi': self.indicators.rsi,
            'bb_w': self.indicators.bb_width,
            'real_vol': self.indicators.realized_vol,
            'ob_imb': self.orderbook.imbalance,
            'dxy': self.macro.dxy,
            'writers': {k: v for k, v in self._writers.items() if v},
        }
