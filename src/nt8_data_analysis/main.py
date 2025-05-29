from fastapi import FastAPI, Request
from .functional_processor import (
    create_initial_market_state,
    create_data_processing_pipeline,
    get_analysis_response_string
)
from .config import (
    API_TITLE, TEST_RESPONSE, RESET_RESPONSE, HEALTH_CHECK_RESPONSE,
    MAX_DATA_ENTRIES
)

app = FastAPI(title=API_TITLE)

# Initialize market state and processing pipeline using functional approach
market_state = create_initial_market_state(max_entries=MAX_DATA_ENTRIES)
process_data = create_data_processing_pipeline(time_period=1)


@app.post("/")
async def process_string(request: Request):
    """
    Process price data and return trading signals based on slope deviation changes.

    Returns:
        - "buy" if slope deviation increases by 50% or more
        - "sell" if slope deviation decreases by 50% or more
        - Price direction, EMA slope, and slope deviation info otherwise
    """
    global market_state

    # Get the raw bytes from the request body
    raw_data = await request.body()

    # Decode bytes to string
    processed_string = raw_data.decode('utf-8')

    if processed_string == "test":
        return TEST_RESPONSE

    # Process the string data using functional pipeline
    market_state, analysis = process_data(processed_string, market_state)

    # Return response based on analysis
    return get_analysis_response_string(analysis)


@app.post('/reset')
async def reset():
    """Reset all stored price data."""
    global market_state
    market_state = create_initial_market_state(max_entries=MAX_DATA_ENTRIES)
    return RESET_RESPONSE


@app.get("/")
async def root():
    """Health check endpoint."""
    return HEALTH_CHECK_RESPONSE
