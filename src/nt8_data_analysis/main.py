from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import pandas as pd
from ta.trend import EMAIndicator
import math
from hurst import compute_Hc

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
        
        # For EMA calculation, we first need chronological order (oldest to newest)
        chrono_data = data_list.sort_values(by="time", ascending=True).reset_index(drop=True)
        
        if len(chrono_data) > 8:  # Need minimum data for 8 EMA
            # Calculate EMA for all data in chronological order
            ema_indicator = EMAIndicator(close=chrono_data["close"], window=8)
            chrono_data["ema_8"] = ema_indicator.ema_indicator()
            
            # Calculate raw slope in chronological order (current minus previous)
            raw_slope = chrono_data["ema_8"] - chrono_data["ema_8"].shift(1)
            
            # Convert slope to degrees: degrees = atan(slope) * (180/π)
            chrono_data["ema_slope"] = raw_slope.apply(lambda x: round(math.degrees(math.atan(x)), 2) if pd.notnull(x) else 0)
            
            # Calculate deviation from the average of slope values in a period (last 5 values)
            # First, make sure we have enough data points
            period = 8
            if len(chrono_data) > period:
                # Calculate rolling average of slope values
                # We use min_periods=1 to handle the initial values where we don't have a full window yet
                chrono_data["slope_avg"] = chrono_data["ema_slope"].rolling(window=period, min_periods=1).mean().round(2)
                
                # Calculate deviation of latest slope from the average (excluding the latest point)
                for i in range(period, len(chrono_data)):
                    # Calculate average of previous 'period' slopes (excluding current)
                    avg_without_current = chrono_data.loc[i-period:i-1, "ema_slope"].mean()
                    # Calculate absolute deviation from that average
                    chrono_data.loc[i, "slope_deviation"] = round(abs(chrono_data.loc[i, "ema_slope"] - avg_without_current), 2)
                
                # Fill NaN values with 0 for rows where we couldn't calculate
                chrono_data["slope_deviation"] = chrono_data["slope_deviation"].fillna(0)
            else:
                # Not enough data yet, set deviation to 0
                chrono_data["slope_deviation"] = 0
            
            # Sort back to descending order for displaying most recent first
            data_list = chrono_data.sort_values(by="time", ascending=False).reset_index(drop=True)

def sort_data(data_list: pd.DataFrame) -> pd.DataFrame:
    # Sort DataFrame by time in descending order (most current time first)
    return data_list.sort_values(by="time", ascending=False).reset_index(drop=True)

def determine_price_direction(data_list: pd.DataFrame) -> str:
    print(data_list)
    # Check if we have at least 2 items to compare
    if len(data_list) >= 2:
        # With descending order, index 0 is the most recent, index 1 is the previous
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
    Calculate the slope of the EMA for the given period and return the slope value in degrees and direction.
    Returns:
        tuple: (slope_in_degrees: float, direction: str)
    """
    if len(data_list) < period + 1:
        return 0.0, "not enough data"
    
    if "ema_slope" in data_list.columns:
        # Slope is already calculated in the dataframe
        slope_degrees = data_list["ema_slope"].iloc[0]  # Get the most recent slope value
        if slope_degrees > 0:
            direction = "up"
        elif slope_degrees < 0:
            direction = "down"
        else:
            direction = "neutral"
        return slope_degrees, direction
    else:
        # For EMA calculation, we need chronological order (oldest to newest)
        temp_data = data_list.sort_values(by="time", ascending=True).reset_index(drop=True)
        
        # Calculate the EMA using TA library
        ema_indicator = EMAIndicator(close=temp_data["close"], window=period, fillna=True)
        ema = ema_indicator.ema_indicator()
        
        # Calculate raw slope
        raw_slope = ema.iloc[-1] - ema.iloc[-2]
        
        # Convert to degrees: degrees = atan(slope) * (180/π)
        slope_degrees = round(math.degrees(math.atan(raw_slope)), 2)
        
        if slope_degrees > 0:
            direction = "up"
        elif slope_degrees < 0:
            direction = "down"
        else:
            direction = "neutral"
        return slope_degrees, direction

def get_slope_deviation(data_list: pd.DataFrame) -> float:
    """
    Get the most recent absolute deviation of the EMA slope from the average of previous periods.
    Returns the magnitude of deviation between the current trend angle and the average trend.
    """
    if len(data_list) < 2 or 'slope_deviation' not in data_list.columns:
        return 0.0
    
    # Get the most recent slope deviation
    return data_list["slope_deviation"].iloc[0]

def calculate_hurst_exponent(data_list: pd.DataFrame) -> float:
    """
    Calculate the Hurst Exponent for the given price data.
    Uses only the most recent 100 rows of data for the calculation.
    
    Returns:
        float: The Hurst Exponent value.
    """
    if len(data_list) < 2:
        return 0.0
    
    # Get the most recent 100 rows (or all rows if less than 100)
    recent_data = data_list.head(100)
    
    # Get the price data from the recent data
    price_data = recent_data["close"].values
    
    # Calculate the Hurst Exponent
    H, c, data = compute_Hc(price_data, kind="price")
    
    return H


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
        return "test complete"
    
    # Process the string data
    process_data(processed_string)
    sorted_data = sort_data(data_list)
    price_direction = determine_price_direction(sorted_data)
    ema_slope, ema_direction = calculate_ema_slope(sorted_data, period=8)
    slope_deviation = get_slope_deviation(sorted_data)
    hurst_exponent = calculate_hurst_exponent(sorted_data)
    # Check for significant changes in slope deviation if we have enough data
    if 'slope_deviation' in sorted_data.columns and len(sorted_data) >= 2:
        current_deviation = sorted_data['slope_deviation'].iloc[0]
        previous_deviation = sorted_data['slope_deviation'].iloc[1]
        
        # Only proceed if both values are valid
        if pd.notnull(current_deviation) and pd.notnull(previous_deviation) and previous_deviation > 0:
            # Calculate percentage change
            percentage_change = ((current_deviation - previous_deviation) / previous_deviation) * 100
            
            # Debug info
            print(f"Slope deviation: current={current_deviation}, previous={previous_deviation}, change={percentage_change}%")
            
            # Return signals based on percentage change
            if percentage_change >= 50:
                return "buy"
            elif percentage_change <= -50:
                return "sell"
    
    # Return the default response if no signal was triggered
    return f"price_direction: {price_direction}, ema_slope: {ema_slope}, slope_deviation: {slope_deviation}, hurst_exponent: {hurst_exponent}"

@app.post('/reset')
async def reset():
    global data_list
    data_list = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    return "data reset"

@app.get("/")
async def root():
    return "Hello World"