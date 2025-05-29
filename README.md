# NT8 Data Analysis

A FastAPI-based technical analysis server for processing price data and generating trading signals. This project has been refactored to use **Functional Programming principles** for improved maintainability, testability, and safety.

## 🔄 Functional Programming Refactor

This project demonstrates a complete transformation from Object-Oriented Programming (OOP) to Functional Programming (FP):

- **Immutable Data Structures**: All data models use frozen dataclasses and NamedTuples
- **Pure Functions**: Side-effect-free functions for predictable behavior
- **Function Composition**: Modular pipeline architecture
- **Explicit State Management**: No hidden mutable state

📖 **See [FUNCTIONAL_REFACTOR.md](./FUNCTIONAL_REFACTOR.md) for detailed documentation**

## Project Structure

```
src/nt8_data_analysis/
├── __init__.py                # Package initialization
├── main.py                   # FastAPI application (functional approach)
├── models.py                 # Immutable data models
├── functional_processor.py   # Pure functions for data processing ✨ NEW
├── data_processor.py         # Original OOP implementation (preserved)
└── config.py                 # Configuration constants
```

## Features

- **Real-time Price Data Processing**: Handles tab or comma-delimited price data
- **Technical Indicators**:
  - Exponential Moving Average (EMA)
  - EMA Slope calculation and trend direction
  - Slope deviation analysis
  - Hurst Exponent for trend persistence
- **Trading Signals**: Automatic buy/sell signals based on multiple indicators
- **Functional Architecture**:
  - Immutable state management
  - Pure function composition
  - Thread-safe operations
  - Easy testing and debugging

## API Endpoints

### POST `/`

Process price data and return analysis results or trading signals.

**Input**: Raw price data string (comma or tab delimited)
**Format**: `time,open,high,low,close,volume`

**Returns**:
- `"buy"` - when all conditions met: EMA slope > 50°, slope deviation > 50, Hurst > 0.5
- `"sell"` - when all conditions met: EMA slope < -50°, slope deviation > 50, Hurst > 0.5
- Analysis string with price direction, EMA slope, slope deviation, and Hurst exponent

### POST `/reset`

Reset all stored price data and create fresh market state.

### GET `/`

Health check endpoint.

## Usage

### Running the Server

```bash
uvicorn src.nt8_data_analysis.main:app --reload
```

### Using the Functional Processor

```python
from src.nt8_data_analysis.functional_processor import (
    create_initial_market_state,
    create_data_processing_pipeline
)

# Initialize immutable state
state = create_initial_market_state(max_entries=1000)
pipeline = create_data_processing_pipeline(time_period=1)

# Process data (returns new state)
data = "2024-01-01 09:30:00,100.0,101.0,99.0,100.5,1000"
new_state, analysis = pipeline(data, state)

print(f"Direction: {analysis.price_direction}")
print(f"Signal: {analysis.trading_signal}")
```

### Custom Function Composition

```python
from src.nt8_data_analysis.functional_processor import (
    parse_price_data_string,
    add_price_data_to_state,
    update_market_state_with_indicators,
    analyze_market_state
)

def custom_pipeline(data_string: str, state: MarketState):
    price_data = parse_price_data_string(data_string)
    updated_state = add_price_data_to_state(price_data, state)
    state_with_indicators = update_market_state_with_indicators(updated_state)
    analysis = analyze_market_state(state_with_indicators)
    return state_with_indicators, analysis
```

## Configuration

All settings are centralized in `config.py`:

- `MAX_DATA_ENTRIES`: Maximum number of price records to keep (default: 1000)
- `EMA_WINDOW`: EMA calculation period (default: 8)
- `SLOPE_DEVIATION_PERIOD`: Period for slope deviation calculation (default: 8)
- `HURST_WINDOW`: Number of recent data points for Hurst calculation (default: 100)

## Dependencies

- FastAPI
- pandas
- numpy
- ta (Technical Analysis library)
- hurst

## Functional Programming Benefits

### 🔒 Immutability
- **Thread Safety**: No race conditions with concurrent access
- **Debugging**: State can be inspected at any point without side effects
- **Undo/Redo**: Previous states are preserved automatically

### 🧪 Pure Functions
- **Testability**: Functions are deterministic and isolated
- **Reusability**: Functions can be composed in different ways
- **Caching**: Results can be memoized safely

### 🔧 Composability
- **Modularity**: Each function has a single responsibility
- **Flexibility**: Easy to create custom processing pipelines
- **Extensibility**: New functionality can be added without breaking existing code

## Performance Comparison

| Approach | Time (50 entries) | Memory | Characteristics |
|----------|------------------|---------|----------------|
| OOP | 0.177s | Lower | Mutable state, in-place operations |
| Functional | 0.271s | Higher | Immutable state, safer operations |

*The functional approach trades ~1.5x performance for significantly improved safety and maintainability.*

## Testing

```bash
# Run comprehensive equivalence verification
python verify_equivalence.py

# Run demonstration comparing both approaches
python comparison_demo.py

# Run functional programming tests
python -m pytest tests/test_functional.py -v
```

## Migration Notes

The functional refactor maintains **100% API compatibility** with the original OOP implementation:

- ✅ All endpoints work identically
- ✅ Same input/output formats
- ✅ Identical calculation results
- ✅ Drop-in replacement for existing code

## Further Reading

- [FUNCTIONAL_REFACTOR.md](./FUNCTIONAL_REFACTOR.md) - Complete refactoring documentation
- [comparison_demo.py](./comparison_demo.py) - Live comparison of both approaches
- [verify_equivalence.py](./verify_equivalence.py) - Verification that both approaches are equivalent

## License

This project demonstrates functional programming principles in financial data analysis and serves as a reference for refactoring OOP codebases to functional paradigms.