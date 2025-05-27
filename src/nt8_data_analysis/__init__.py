"""NT8 Data Analysis Package - Technical analysis and trading signals for NT8 platform."""

from .main import app
from .data_processor import DataProcessor
from .models import Order, PriceData
from . import config

__version__ = "1.0.0"
__author__ = "NT8 Data Analysis Team"

__all__ = [
    "app",
    "DataProcessor", 
    "Order",
    "PriceData",
    "config"
]
