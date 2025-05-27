from pydantic import BaseModel


class Order(BaseModel):
    buy: bool
    sell: bool
    contract_size: int
    tp: float
    sl: float


class PriceData(BaseModel):
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: float 