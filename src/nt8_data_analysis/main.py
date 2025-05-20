from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import pandas as pd
from ta.trend import EMAIndicator

app = FastAPI(title="NT8 AI Server")

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

# Initialize an empty DataFrame with columns matching PriceData
data_list = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

def process_data(data: str) -> None:
    [time, open_val, high, low, close, volume] = data.split(",")
    # Convert string values to appropriate types immediately
    price_data = {
        "time": time,
        "open": float(open_val),
        "high": float(high),
        "low": float(low),
        "close": float(close),
        "volume": float(volume),
    }
    
    # Check if data with this timestamp already exists in the dataframe
    global data_list
    if time not in data_list["time"].values:
        # Append new data to DataFrame only if time doesn't already exist
        data_list = pd.concat([data_list, pd.DataFrame([price_data])], ignore_index=True)
        
        # Keep only the latest 1000 entries if DataFrame grows too large
        if len(data_list) > 1000:
            data_list = data_list.iloc[-1000:]
        
        # Sort the data and update EMA values
        sorted_data = sort_data(data_list)
        if len(sorted_data) > 8:  # Need minimum data for 8 EMA
            # Calculate EMA for all data
            ema_indicator = EMAIndicator(close=sorted_data["close"], window=8)
            ema_values = ema_indicator.ema_indicator()
            
            # Add EMA values to the dataframe
            sorted_data["ema_8"] = ema_values
            
            # Calculate slope values (current EMA minus previous EMA)
            # With ascending=True sort, we use the previous value (earlier in time)
            sorted_data["ema_slope"] = sorted_data["ema_8"] - sorted_data["ema_8"].shift(1)
            
            # Fill NaN values in the first row where we can't calculate slope
            sorted_data["ema_slope"] = sorted_data["ema_slope"].fillna(0)
            
            # Update the global data_list with the sorted data including the new columns
            data_list = sorted_data

def sort_data(data_list: pd.DataFrame) -> pd.DataFrame:
    # Sort DataFrame by time in descending order
    return data_list.sort_values(by="time", ascending=True).reset_index(drop=True)

def determine_price_direction(data_list: pd.DataFrame) -> str:
    print(data_list)
    # Check if we have at least 2 items to compare
    if len(data_list) >= 2:
        # Compare close values of first and second items
        if data_list.iloc[0]["close"] > data_list.iloc[1]["close"]:
            return "up"
        elif data_list.iloc[0]["close"] < data_list.iloc[1]["close"]:
            return "down"
        else:
            return "neutral"
    # If we don't have enough data, return a default
    return "not enough data"

def calculate_ema_slope(data_list: pd.DataFrame, period: int = 8):
    """
    Calculate the slope of the EMA for the given period and return the slope value and direction.
    Returns:
        tuple: (slope_value: float, direction: str)
    """
    if len(data_list) < period + 1:
        return 0.0, "not enough data"

    # Calculate the EMA using TA library
    ema_indicator = EMAIndicator(close=data_list["close"], window=period, fillna=True)
    ema = ema_indicator.ema_indicator()
    
    # Slope: difference between the last and previous EMA values
    slope = ema.iloc[-1] - ema.iloc[-2]
    if slope > 0:
        direction = "up"
    elif slope < 0:
        direction = "down"
    else:
        direction = "neutral"
    return slope, direction

@app.post("/")
async def process_string(request: Request):
    """
    Process a string and return the result.
    
    This endpoint takes a string and returns it in uppercase.
    """
    # Get the raw bytes from the request body
    raw_data = await request.body()
    
    # Decode bytes to string
    processed_string = raw_data.decode('utf-8')

    if processed_string == "test":
        return "test complete"
    
    # Process the string data
    process_data(processed_string)
    sorted_data = sort_data(data_list)
    price_direction = determine_price_direction(sorted_data)
    ema_slope, ema_direction = calculate_ema_slope(sorted_data, period=8)
    
    # Return both the price direction and EMA slope with direction as a string
    return f"price_direction: {price_direction}, ema_slope: {ema_slope}, ema_direction: {ema_direction}"

@app.post('/reset')
async def reset():
    global data_list
    data_list = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    return "data reset"

@app.get("/")
async def root():
    return "Hello World"