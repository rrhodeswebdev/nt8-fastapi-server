import pandas as pd
import numpy as np
import logging
from ta.trend import EMAIndicator
from hurst import compute_Hc
from typing import List, Tuple, Optional, Callable
from functools import reduce
from .models import PriceData, TechnicalIndicators, EnrichedPriceData, MarketState, AnalysisResult
from .config import (
    MAX_DATA_ENTRIES, EMA_WINDOW, SLOPE_DEVIATION_PERIOD,
    HURST_WINDOW, ENABLE_DATAFRAME_DISPLAY, VERBOSE_LOGGING,
    COMPACT_DATAFRAME_DISPLAY, MAX_DATAFRAME_ROWS
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Utility function for DataFrame creation and printing
def create_dataframe_from_enriched_data(data: List[EnrichedPriceData]) -> pd.DataFrame:
    """Create a pandas DataFrame from enriched price data."""
    if not data:
        return pd.DataFrame()

    # Sort data chronologically (most recent first for display)
    sorted_data = sort_data_chronologically(data, ascending=False)

    rows = []
    for item in sorted_data:
        row = {
            'time': item.price_data.time,
            'open': item.price_data.open,
            'high': item.price_data.high,
            'low': item.price_data.low,
            'close': item.price_data.close,
            'volume': item.price_data.volume,
            'ema_8': item.indicators.ema_8,
            'ema_200': item.indicators.ema_200,
            'ema_slope': item.indicators.ema_slope,
            'slope_deviation': item.indicators.slope_deviation,
            'hurst_exponent': item.indicators.hurst_exponent
        }
        rows.append(row)

    return pd.DataFrame(rows)


def print_dataframe_summary(data: List[EnrichedPriceData], title: str = "Market Data"):
    """Print a formatted summary of the current market data."""
    if not ENABLE_DATAFRAME_DISPLAY:
        return

    df = create_dataframe_from_enriched_data(data)

    if df.empty:
        if not COMPACT_DATAFRAME_DISPLAY:
            print(f"\n{title}: No data available")
        return

    # Print summary statistics for the most recent entry
    if len(df) > 0:
        latest = df.iloc[0]
        print("\nMost Recent Entry Summary:")
        print(f"  Time: {latest['time']}")
        print(f"  Price: {latest['close']:.2f} (O:{latest['open']:.2f} H:{latest['high']:.2f} L:{latest['low']:.2f})")
        if pd.notna(latest['ema_8']):
            print(f"  EMA(8): {latest['ema_8']:.4f}")
        if pd.notna(latest['ema_200']):
            print(f"  EMA(200): {latest['ema_200']:.4f}")
        print(f"  EMA Slope: {latest['ema_slope']:.4f}°")
        print(f"  Slope Deviation: {latest['slope_deviation']:.2f}")
        print(f"  Hurst Exponent: {latest['hurst_exponent']:.4f}")

    print("=" * 80)


# Pure functions for data parsing
def parse_price_data_string(data_string: str) -> PriceData:
    """Parse a single comma-delimited price data string into PriceData."""
    time, open_val, high, low, close, volume = data_string.split(",")
    return PriceData(
        time=time,
        open=float(open_val),
        high=float(high),
        low=float(low),
        close=float(close),
        volume=float(volume)
    )


def parse_multiple_data_strings(data: str) -> List[PriceData]:
    """Parse tab or comma delimited data strings into a list of PriceData."""
    if '\t' in data:
        # Split on tabs and process each comma-delimited string
        tab_separated_strings = data.split('\t')
        return [
            parse_price_data_string(single_data_string.strip())
            for single_data_string in tab_separated_strings
            if single_data_string.strip()
        ]
    else:
        # Process as a single comma-delimited string
        return [parse_price_data_string(data)]


# Pure functions for state management
def create_initial_market_state(max_entries: int = MAX_DATA_ENTRIES) -> MarketState:
    """Create an initial empty market state."""
    return MarketState(
        data=[],
        time_cache=frozenset(),
        max_entries=max_entries
    )


def is_data_already_present(time: str, time_cache: frozenset[str]) -> bool:
    """Check if data with this timestamp already exists."""
    return time in time_cache


def add_price_data_to_state(price_data: PriceData, state: MarketState) -> MarketState:
    """Add new price data to market state if not already present."""
    if is_data_already_present(price_data.time, state.time_cache):
        if VERBOSE_LOGGING:
            print(f"⚠️  Duplicate data ignored for time: {price_data.time}")
        return state

    if VERBOSE_LOGGING:
        print(f"📊 Adding new price data: {price_data.time} - Close: {price_data.close}")

    # Create enriched data with empty indicators
    enriched_data = EnrichedPriceData(
        price_data=price_data,
        indicators=TechnicalIndicators()
    )

    # Add to data list and sort chronologically to maintain proper order
    new_data = state.data + [enriched_data]

    # Sort chronologically (oldest to newest)
    times = [item.price_data.time for item in new_data]
    time_series = pd.to_datetime(times)
    sorted_indices = time_series.argsort()
    sorted_data = [new_data[i] for i in sorted_indices]

    new_time_cache = state.time_cache | {price_data.time}

    # Trim if necessary - keep the most recent entries
    if len(sorted_data) > state.max_entries:
        trimmed_data = sorted_data[-state.max_entries:]
        # Update time cache to only include remaining entries
        remaining_times = {item.price_data.time for item in trimmed_data}
        new_time_cache = frozenset(remaining_times)
        sorted_data = trimmed_data
        if VERBOSE_LOGGING:
            print(f"📝 Data trimmed to {len(trimmed_data)} entries (max: {state.max_entries})")

    # Print raw data after addition (before indicators)
    print_dataframe_summary(sorted_data, f"Real-time Market Data Update ({len(sorted_data)} entries)")

    return MarketState(
        data=sorted_data,
        time_cache=new_time_cache,
        max_entries=state.max_entries
    )


def process_multiple_price_data(price_data_list: List[PriceData], state: MarketState) -> MarketState:
    """Process multiple price data entries using functional composition."""
    # For large datasets (likely historical data), use optimized batch processing
    if len(price_data_list) > 10:
        return process_batch_price_data(price_data_list, state)

    # For real-time data (small batches), process incrementally
    return reduce(
        lambda current_state, price_data: add_price_data_to_state(price_data, current_state),
        price_data_list,
        state
    )


def process_batch_price_data(price_data_list: List[PriceData], state: MarketState) -> MarketState:
    """
    Optimized batch processing for large historical datasets.
    Handles entire batch at once to avoid excessive logging and ensure proper sorting.
    """
    if not price_data_list:
        return state

    if VERBOSE_LOGGING:
        print(f"📊 Processing batch of {len(price_data_list)} historical data entries")

    # Filter out duplicates and convert to enriched data
    enriched_batch = []
    new_times = set()

    for price_data in price_data_list:
        if not is_data_already_present(price_data.time, state.time_cache):
            enriched_data = EnrichedPriceData(
                price_data=price_data,
                indicators=TechnicalIndicators()
            )
            enriched_batch.append(enriched_data)
            new_times.add(price_data.time)

    if not enriched_batch:
        if VERBOSE_LOGGING:
            print("⚠️  All batch data already present, no new data added")
        return state

    # Combine with existing data
    combined_data = state.data + enriched_batch

    # Sort chronologically (oldest to newest) and trim to max_entries from the most recent end
    times = [item.price_data.time for item in combined_data]
    time_series = pd.to_datetime(times)
    sorted_indices = time_series.argsort()
    sorted_data = [combined_data[i] for i in sorted_indices]

    # Keep only the most recent entries
    if len(sorted_data) > state.max_entries:
        trimmed_data = sorted_data[-state.max_entries:]
        if VERBOSE_LOGGING:
            print(f"📝 Batch data trimmed to {len(trimmed_data)} entries (max: {state.max_entries})")
            oldest_time = trimmed_data[0].price_data.time
            newest_time = trimmed_data[-1].price_data.time
            print(f"📅 Data range: {oldest_time} to {newest_time}")
    else:
        trimmed_data = sorted_data

    # Update time cache with remaining entries
    new_time_cache = frozenset(item.price_data.time for item in trimmed_data)

    if VERBOSE_LOGGING:
        print(f"✅ Batch processed: {len(enriched_batch)} new entries added, {len(trimmed_data)} total entries")

    # Print raw data after batch addition (single summary)
    print_dataframe_summary(trimmed_data, f"Historical Market Data Loaded ({len(trimmed_data)} entries)")

    return MarketState(
        data=trimmed_data,
        time_cache=new_time_cache,
        max_entries=state.max_entries
    )


# Pure functions for sorting and chronological ordering
def get_chronological_order(data: List[EnrichedPriceData]) -> List[int]:
    """Get indices sorted chronologically."""
    times = [item.price_data.time for item in data]
    time_series = pd.to_datetime(times)
    return time_series.argsort().tolist()


def sort_data_chronologically(data: List[EnrichedPriceData], ascending: bool = True) -> List[EnrichedPriceData]:
    """Sort data chronologically."""
    if not data:
        return data

    times = [item.price_data.time for item in data]
    time_series = pd.to_datetime(times)
    sorted_indices = time_series.argsort()

    if not ascending:
        sorted_indices = sorted_indices[::-1]

    return [data[i] for i in sorted_indices]


# Pure functions for technical indicator calculations
def calculate_ema_values(data: List[EnrichedPriceData], window: int) -> List[float]:
    """Calculate EMA values for chronologically sorted data."""
    if len(data) < window:
        return [np.nan] * len(data)

    sorted_data = sort_data_chronologically(data, ascending=True)
    close_prices = pd.Series([item.price_data.close for item in sorted_data])

    ema_indicator = EMAIndicator(close=close_prices, window=window)
    return ema_indicator.ema_indicator().tolist()


def calculate_slope_degrees(current_value: float, previous_value: float, time_period: int = 1) -> float:
    """Calculate slope in degrees between two values."""
    if pd.isna(current_value) or pd.isna(previous_value):
        return 0.0

    raw_slope = (current_value - previous_value) / time_period
    return np.round(np.degrees(np.arctan(raw_slope)), 4)


def calculate_slopes_from_ema(ema_values: List[float], close_prices: List[float], time_period: int = 1) -> List[float]:
    """Calculate slopes from EMA values, falling back to close prices for early values."""
    slopes = [0.0]  # First value has no previous value

    for i in range(1, len(ema_values)):
        current_ema = ema_values[i]
        prev_ema = ema_values[i-1]

        if pd.notna(current_ema) and pd.notna(prev_ema):
            slope = calculate_slope_degrees(current_ema, prev_ema, time_period)
        else:
            # Fallback to price-based slope for early bars
            current_price = close_prices[i]
            prev_price = close_prices[i-1]
            slope = calculate_slope_degrees(current_price, prev_price, time_period)

        slopes.append(slope)

    return slopes


def calculate_slope_deviations(slopes: List[float], period: int = SLOPE_DEVIATION_PERIOD) -> List[float]:
    """Calculate slope deviations."""
    deviations = [0.0] * len(slopes)

    for i in range(period, len(slopes)):
        prev_slopes = slopes[i-period:i]
        avg_slope = float(np.mean(prev_slopes))
        current_slope = slopes[i]
        deviation = round(abs(current_slope - avg_slope), 2)
        deviations[i] = deviation

    return deviations


def calculate_hurst_exponent_for_data(close_prices: List[float], window: int = HURST_WINDOW) -> float:
    """Calculate Hurst exponent for price data."""
    if len(close_prices) < 100:
        return 0.0

    # Use the most recent 100 data points
    recent_prices = close_prices[-100:]

    try:
        H, c, data = compute_Hc(recent_prices, kind="price", simplified=True)
        if 0 <= H <= 1:
            return round(H, 2)
        else:
            logging.warning(f"Hurst exponent {H} out of valid range [0,1]")
            return 0.0
    except Exception as e:
        logging.error(f"Error calculating Hurst exponent: {e}")
        return 0.0


def update_indicators_for_data(data: List[EnrichedPriceData], time_period: int = 1) -> List[EnrichedPriceData]:
    """Update technical indicators for all data points."""
    if len(data) <= EMA_WINDOW:
        return data

    # Sort data chronologically for calculations
    sorted_data = sort_data_chronologically(data, ascending=True)
    close_prices = [item.price_data.close for item in sorted_data]

    # Calculate all indicators
    ema_8_values = calculate_ema_values(sorted_data, EMA_WINDOW)
    ema_200_values = calculate_ema_values(sorted_data, 200)
    slopes = calculate_slopes_from_ema(ema_8_values, close_prices, time_period)
    slope_deviations = calculate_slope_deviations(slopes)

    # Calculate Hurst exponent only for the most recent point
    hurst_exponent = calculate_hurst_exponent_for_data(close_prices) if len(close_prices) >= 100 else 0.0

    # Create new enriched data with updated indicators
    updated_data = []
    for i, item in enumerate(sorted_data):
        # Only assign hurst exponent to the most recent point
        hurst_value = hurst_exponent if i == len(sorted_data) - 1 else item.indicators.hurst_exponent

        new_indicators = TechnicalIndicators(
            ema_8=ema_8_values[i] if not pd.isna(ema_8_values[i]) else None,
            ema_200=ema_200_values[i] if not pd.isna(ema_200_values[i]) else None,
            ema_slope=slopes[i],
            slope_deviation=slope_deviations[i],
            hurst_exponent=hurst_value
        )

        updated_data.append(EnrichedPriceData(
            price_data=item.price_data,
            indicators=new_indicators
        ))

    return updated_data


def update_market_state_with_indicators(state: MarketState, time_period: int = 1) -> MarketState:
    """Update market state with calculated indicators."""
    if len(state.data) <= EMA_WINDOW:
        return state

    updated_data = update_indicators_for_data(state.data, time_period)

    # Print DataFrame after updating indicators
    print_dataframe_summary(updated_data, "Updated Market Data with Indicators")

    return MarketState(
        data=updated_data,
        time_cache=state.time_cache,
        max_entries=state.max_entries
    )


# Pure functions for analysis
def determine_price_direction(data: List[EnrichedPriceData]) -> str:
    """Determine if price is moving up, down, or neutral."""
    if len(data) < 2:
        return "not enough data"

    sorted_data = sort_data_chronologically(data, ascending=False)  # Most recent first
    current_close = sorted_data[0].price_data.close
    previous_close = sorted_data[1].price_data.close

    if current_close > previous_close:
        return "up"
    elif current_close < previous_close:
        return "down"
    else:
        return "neutral"

def determine_market_structure(data: List[EnrichedPriceData]) -> str:
    """Determine structure based on hurst exponent."""
    if len(data) < 100:
        return "not enough data"

    # Get the most recent Hurst exponent from the latest data point
    sorted_data = sort_data_chronologically(data, ascending=False)  # Most recent first
    hurst_exponent = sorted_data[0].indicators.hurst_exponent

    if hurst_exponent > 0.6:
        return "trending"
    elif hurst_exponent < 0.4:
        return "ranging"
    elif hurst_exponent > 0.5 and hurst_exponent <= 0.6:
        return "weak trend"
    elif hurst_exponent >= 0.4 and hurst_exponent <= 0.5:
        return "weak ranging"
    else:
        return "random-walk"

def get_most_recent_indicators(data: List[EnrichedPriceData]) -> Optional[TechnicalIndicators]:
    """Get the most recent technical indicators."""
    if not data:
        return None

    sorted_data = sort_data_chronologically(data, ascending=False)  # Most recent first
    return sorted_data[0].indicators


def calculate_ema_slope_and_direction(data: List[EnrichedPriceData], period: int = EMA_WINDOW) -> Tuple[float, str]:
    """Calculate EMA slope and direction from existing indicators."""
    if len(data) < period + 1:
        return 0.0, "not enough data"

    indicators = get_most_recent_indicators(data)
    if not indicators:
        return 0.0, "not enough data"

    slope_degrees = indicators.ema_slope

    if slope_degrees > 0:
        direction = "up"
    elif slope_degrees < 0:
        direction = "down"
    else:
        direction = "neutral"

    return slope_degrees, direction


def check_trading_signals(indicators: TechnicalIndicators, current_price: float) -> Optional[str]:
    """Check for buy/sell signals based on threshold criteria and 200 EMA filter."""
    ema_slope = indicators.ema_slope
    slope_deviation = indicators.slope_deviation
    hurst_exponent = indicators.hurst_exponent
    ema_200 = indicators.ema_200

    # Ensure all values are valid numbers
    if pd.isna(ema_slope) or pd.isna(slope_deviation) or pd.isna(hurst_exponent) or pd.isna(ema_200):
        return None

    # Log current indicator values for signal checking
    logging.info(f"Signal Check - EMA Slope: {ema_slope}, Slope Deviation: {slope_deviation}, "
                f"Hurst Exponent: {hurst_exponent}, EMA 200: {ema_200}, Price: {current_price}")

    # Check if price is above or below 200 EMA
    price_above_ema200 = current_price > ema_200
    
    # Check for BUY signal: ema_slope > 35, slope_deviation > 25, hurst_exponent > 0.50
    # Only valid if price is above 200 EMA
    if ema_slope > 35 and slope_deviation > 25 and hurst_exponent > 0.50 and price_above_ema200:
        logging.info("BUY signal generated!")
        return "buy"

    # Check for SELL signal: ema_slope < -35, slope_deviation > 25, hurst_exponent > 0.50
    # Only valid if price is below 200 EMA
    elif ema_slope < -35 and slope_deviation > 25 and hurst_exponent > 0.50 and not price_above_ema200:
        logging.info("SELL signal generated!")
        return "sell"

    return None


def analyze_market_state(state: MarketState) -> AnalysisResult:
    """Perform complete market analysis and return results."""
    if not state.data:
        if VERBOSE_LOGGING:
            print("📈 Analysis: Not enough data for analysis")
        return AnalysisResult(
            price_direction="not enough data",
            ema_slope=0.0,
            ema_direction="not enough data",
            slope_deviation=0.0,
            hurst_exponent="not enough data",
            trading_signal=None
        )

    price_direction = determine_price_direction(state.data)
    ema_slope, ema_direction = calculate_ema_slope_and_direction(state.data)

    # Get most recent data point
    sorted_data = sort_data_chronologically(state.data, ascending=False)
    latest_data = sorted_data[0]
    current_price = latest_data.price_data.close
    
    indicators = get_most_recent_indicators(state.data)
    slope_deviation = indicators.slope_deviation if indicators else 0.0
    hurst_exponent = determine_market_structure(state.data)
    trading_signal = None
    
    if indicators and not pd.isna(indicators.ema_200):
        trading_signal = check_trading_signals(indicators, current_price)

    # Print analysis summary
    if VERBOSE_LOGGING:
        print("\n📈 Market Analysis Summary:")
        print(f"   Price Direction: {price_direction}")
        print(f"   EMA Slope: {ema_slope:.4f}° ({ema_direction})")
        print(f"   Slope Deviation: {slope_deviation:.2f}")
        print(f"   Hurst Exponent: {hurst_exponent}")
        if indicators and not pd.isna(indicators.ema_200):
            print(f"   Price vs EMA200: {'Above' if current_price > indicators.ema_200 else 'Below'} ({current_price:.2f} vs {indicators.ema_200:.2f})")
        else:
            print(f"   EMA200: Not available yet")
        if trading_signal:
            print(f"   🚨 TRADING SIGNAL: {trading_signal.upper()}")
        else:
            print("   📊 No trading signal generated")

    return AnalysisResult(
        price_direction=price_direction,
        ema_slope=ema_slope,
        ema_direction=ema_direction,
        slope_deviation=slope_deviation,
        hurst_exponent=hurst_exponent,
        trading_signal=trading_signal
    )


# Higher-order functions for processing pipeline
def create_data_processing_pipeline(time_period: int = 1) -> Callable[[str, MarketState], Tuple[MarketState, AnalysisResult]]:
    """Create a data processing pipeline function."""
    def process_pipeline(data_string: str, state: MarketState) -> Tuple[MarketState, AnalysisResult]:
        if VERBOSE_LOGGING:
            print(f"\n🔄 Processing new data: {data_string[:50]}{'...' if len(data_string) > 50 else ''}")

        # Parse data
        price_data_list = parse_multiple_data_strings(data_string)
        if VERBOSE_LOGGING:
            print(f"📝 Parsed {len(price_data_list)} price data entries")

        # Update state with new data
        updated_state = process_multiple_price_data(price_data_list, state)

        # Calculate indicators if we have enough data
        if len(updated_state.data) > EMA_WINDOW:
            if VERBOSE_LOGGING:
                print(f"🧮 Calculating technical indicators (have {len(updated_state.data)} entries, need >{EMA_WINDOW})")
            state_with_indicators = update_market_state_with_indicators(updated_state, time_period)
        else:
            if VERBOSE_LOGGING:
                print(f"⏳ Not enough data for indicators (have {len(updated_state.data)}, need >{EMA_WINDOW})")
            state_with_indicators = updated_state
            # Still print the raw data
            print_dataframe_summary(updated_state.data, "Current Market Data (No Indicators Yet)")

        # Analyze the market
        analysis = analyze_market_state(state_with_indicators)

        if VERBOSE_LOGGING:
            print(f"✅ Processing complete for {len(state_with_indicators.data)} total entries\n")

        return state_with_indicators, analysis

    return process_pipeline


# Utility functions for integration with existing API
def get_analysis_response_string(analysis: AnalysisResult) -> str:
    """Convert analysis result to response string format."""
    if analysis.trading_signal:
        return analysis.trading_signal

    return (f"price_direction: {analysis.price_direction}, "
            f"ema_slope: {analysis.ema_slope}, "
            f"slope_deviation: {analysis.slope_deviation}, "
            f"hurst_exponent: {analysis.hurst_exponent}")
