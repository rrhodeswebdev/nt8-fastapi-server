# NT8 Data Analysis

A FastAPI-based technical analysis server for processing price data and generating trading signals.

## Project Structure

```
src/nt8_data_analysis/
├── __init__.py          # Package initialization
├── main.py              # FastAPI application and endpoints
├── models.py            # Pydantic data models
├── data_processor.py    # Core data processing and technical analysis
└── config.py            # Configuration constants and settings
```

## Features

- **Real-time Price Data Processing**: Handles tab or comma-delimited price data
- **Technical Indicators**:
  - Exponential Moving Average (EMA)
  - EMA Slope calculation and trend direction
  - Slope deviation analysis
  - Hurst Exponent for trend persistence
- **Trading Signals**: Automatic buy/sell signals based on slope deviation changes
- **Performance Optimized**:
  - O(1) duplicate checking with caching
  - Vectorized calculations
  - Incremental indicator updates

## API Endpoints

### POST `/`

Process price data and return analysis results or trading signals.

**Input**: Raw price data string (comma or tab delimited)
**Format**: `time,open,high,low,close,volume`

**Returns**:

- `"buy"` - when slope deviation increases ≥50%
- `"sell"` - when slope deviation decreases ≥50%
- Analysis string with price direction, EMA slope, slope deviation, and Hurst exponent

### POST `/reset`

Reset all stored price data and clear cache.

### GET `/`

Health check endpoint.

## Configuration

All settings are centralized in `config.py`:

- `MAX_DATA_ENTRIES`: Maximum number of price records to keep (default: 1000)
- `EMA_WINDOW`: EMA calculation period (default: 8)
- `SLOPE_DEVIATION_PERIOD`: Period for slope deviation calculation (default: 8)
- `HURST_WINDOW`: Number of recent data points for Hurst calculation (default: 100)
- `BUY_SIGNAL_THRESHOLD`: Percentage increase for buy signal (default: 50%)
- `SELL_SIGNAL_THRESHOLD`: Percentage decrease for sell signal (default: -50%)

## Usage

### Running the Server

```bash
uvicorn src.nt8_data_analysis.main:app --reload
```

### Using the DataProcessor Directly

```python
from src.nt8_data_analysis import DataProcessor

processor = DataProcessor()
processor.process_data("2023-01-01 09:30:00,100.0,101.0,99.0,100.5,1000")

# Get analysis
direction = processor.determine_price_direction()
slope, slope_direction = processor.calculate_ema_slope()
deviation = processor.get_slope_deviation()
hurst = processor.calculate_hurst_exponent()
```

## Dependencies

- FastAPI
- pandas
- numpy
- ta (Technical Analysis library)
- hurst
- pydantic

## Performance Improvements

The refactored code includes several performance optimizations:

1. **Efficient Duplicate Checking**: O(1) time complexity using set-based caching
2. **Vectorized Operations**: NumPy operations replace slower pandas apply functions
3. **Incremental Calculations**: Indicators only recalculated when new data is added
4. **Reduced Memory Allocations**: Pre-allocated columns and cached values
5. **Eliminated Redundant Operations**: Removed unnecessary sorting and DataFrame operations

## Code Organization Benefits

- **Separation of Concerns**: Business logic separated from API endpoints
- **Maintainability**: Modular structure makes code easier to understand and modify
- **Testability**: Individual components can be tested in isolation
- **Configurability**: Centralized configuration management
- **Reusability**: DataProcessor can be used independently of the API
