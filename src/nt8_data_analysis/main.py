from fastapi import FastAPI, Request
from .data_processor import DataProcessor
from .config import (
    API_TITLE, TEST_RESPONSE, RESET_RESPONSE, HEALTH_CHECK_RESPONSE,
    MAX_DATA_ENTRIES, EMA_WINDOW
)

app = FastAPI(title=API_TITLE)

# Initialize the data processor
data_processor = DataProcessor(max_entries=MAX_DATA_ENTRIES)


@app.post("/")
async def process_string(request: Request):
    """
    Process price data and return trading signals based on slope deviation changes.

    Returns:
        - "buy" if slope deviation increases by 50% or more
        - "sell" if slope deviation decreases by 50% or more
        - Price direction, EMA slope, and slope deviation info otherwise
    """
    # Get the raw bytes from the request body
    raw_data = await request.body()

    # Decode bytes to string
    processed_string = raw_data.decode('utf-8')

    if processed_string == "test":
        return TEST_RESPONSE

    # Process the string data
    data_processor.process_data(processed_string)

    # Get analysis results
    price_direction = data_processor.determine_price_direction()
    ema_slope, ema_direction = data_processor.calculate_ema_slope(period=EMA_WINDOW)
    slope_deviation = data_processor.get_slope_deviation()
    hurst_exponent = data_processor.calculate_hurst_exponent()

    # Check for trading signals
    signal = data_processor.check_trading_signals()
    if signal:
        return signal

    # Return the default response if no signal was triggered
    return f"price_direction: {price_direction}, ema_slope: {ema_slope}, slope_deviation: {slope_deviation}, hurst_exponent: {hurst_exponent}"


@app.post('/reset')
async def reset():
    """Reset all stored price data."""
    data_processor.reset()
    return RESET_RESPONSE


@app.get("/")
async def root():
    """Health check endpoint."""
    return HEALTH_CHECK_RESPONSE
