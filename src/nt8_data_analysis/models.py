from typing import NamedTuple, List, Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class Order:
    buy: bool
    sell: bool
    contract_size: int
    tp: float
    sl: float


@dataclass(frozen=True)
class PriceData:
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class TechnicalIndicators:
    ema_8: Optional[float] = None
    ema_slope: float = 0.0
    slope_deviation: float = 0.0
    hurst_exponent: float = 0.0


@dataclass(frozen=True)
class EnrichedPriceData:
    price_data: PriceData
    indicators: TechnicalIndicators


class MarketState(NamedTuple):
    data: List[EnrichedPriceData]
    time_cache: frozenset[str]
    max_entries: int


class AnalysisResult(NamedTuple):
    price_direction: str
    ema_slope: float
    ema_direction: str
    slope_deviation: float
    hurst_exponent: float
    trading_signal: Optional[str]
