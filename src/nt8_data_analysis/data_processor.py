import pandas as pd
import numpy as np
import math
import logging
from ta.trend import EMAIndicator
from hurst import compute_Hc
from typing import Tuple, Optional
from .config import (
    MAX_DATA_ENTRIES, EMA_WINDOW, SLOPE_DEVIATION_PERIOD,
    HURST_WINDOW
)

# Configure logging to display in console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataProcessor:
    """Handles price data processing and technical indicator calculations."""

    def __init__(self, max_entries: int = MAX_DATA_ENTRIES):
        self.data_list = pd.DataFrame({
            "time": pd.Series(dtype='object'),
            "open": pd.Series(dtype='float64'),
            "high": pd.Series(dtype='float64'),
            "low": pd.Series(dtype='float64'),
            "close": pd.Series(dtype='float64'),
            "volume": pd.Series(dtype='float64'),
            "hurst_exponent": pd.Series(dtype='float64')
        })
        self.time_cache = set()
        self.max_entries = max_entries

    def process_data(self, data: str) -> None:
        """Process incoming price data string (tab or comma delimited)."""
        if '\t' in data:
            # Split on tabs and process each comma-delimited string
            tab_separated_strings = data.split('\t')
            for single_data_string in tab_separated_strings:
                if single_data_string.strip():
                    self._process_single_data_string(single_data_string.strip())
        else:
            # Process as a single comma-delimited string
            self._process_single_data_string(data)

    def _process_single_data_string(self, data: str) -> None:
        """Process a single comma-delimited price data string."""
        time, open_val, high, low, close, volume = data.split(",")

        # Check if data with this timestamp already exists using cached set (O(1) operation)
        if time not in self.time_cache:
            # Convert string values to appropriate types
            price_data = {
                "time": time,
                "open": float(open_val),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": float(volume),
                "hurst_exponent": 0.0
            }

            # Add to cache
            self.time_cache.add(time)

            # Append new data to DataFrame
            self.data_list = pd.concat([self.data_list, pd.DataFrame([price_data])], ignore_index=True)

            # Keep only the latest entries if DataFrame grows too large
            if len(self.data_list) > self.max_entries:
                # Remove oldest entries from cache when trimming DataFrame
                removed_times = self.data_list.iloc[:-self.max_entries]["time"].values
                self.time_cache.difference_update(removed_times)
                self.data_list = self.data_list.iloc[-self.max_entries:].reset_index(drop=True)

            # Only recalculate if we have enough data and this is new data
            if len(self.data_list) > EMA_WINDOW:
                self._calculate_indicators()

    def _calculate_indicators(self) -> None:
        """Calculate all technical indicators in one pass."""
        # Sort once for chronological order (oldest to newest)
        chrono_data = self.data_list.sort_values(by="time", ascending=True).reset_index(drop=True)

        # Calculate EMA for all data in chronological order
        ema_indicator = EMAIndicator(close=pd.Series(chrono_data["close"]), window=EMA_WINDOW)
        chrono_data["ema_8"] = ema_indicator.ema_indicator()

        # Calculate raw slope in chronological order (current minus previous)
        raw_slope = chrono_data["ema_8"] - chrono_data["ema_8"].shift(1)

        # Vectorized slope to degrees conversion
        chrono_data["ema_slope"] = np.where(
            pd.notnull(raw_slope),
            np.round(np.degrees(np.arctan(raw_slope)), 2),
            0
        )

        # Calculate deviation from the average of slope values
        if len(chrono_data) > SLOPE_DEVIATION_PERIOD:
            # Pre-allocate slope_deviation column
            chrono_data["slope_deviation"] = 0.0

            # Calculate slope deviation for each point
            for i in range(SLOPE_DEVIATION_PERIOD, len(chrono_data)):
                # Calculate average of previous 'period' slopes (excluding current)
                avg_without_current = chrono_data.iloc[i-SLOPE_DEVIATION_PERIOD:i]["ema_slope"].mean()
                # Calculate absolute deviation from that average
                chrono_data.iloc[i, chrono_data.columns.get_loc("slope_deviation")] = round(
                    abs(chrono_data.iloc[i]["ema_slope"] - avg_without_current), 2
                )
        else:
            # Not enough data yet, set deviation to 0
            chrono_data["slope_deviation"] = 0.0

        # Calculate Hurst Exponent - ensure column exists and fill any NaN values
        if 'hurst_exponent' not in chrono_data.columns:
            chrono_data["hurst_exponent"] = 0.0
        else:
            # Fill any NaN values with 0.0
            chrono_data["hurst_exponent"] = chrono_data["hurst_exponent"].fillna(0.0)
        
        logging.info(f"Hurst calculation - Data length: {len(chrono_data)}, Column exists: {'hurst_exponent' in chrono_data.columns}")
        
        if len(chrono_data) >= 100:  # Need minimum 100 data points for Hurst calculation
            current_hurst = chrono_data.iloc[-1]["hurst_exponent"]
            logging.info(f"Current Hurst value for most recent point: {current_hurst}")
            
            # Only calculate if the most recent point doesn't have a Hurst value
            if pd.isna(current_hurst) or current_hurst == 0.0:
                try:
                    # Use the most recent HURST_WINDOW data points
                    hurst_data = chrono_data.tail(min(HURST_WINDOW, len(chrono_data)))["close"].values
                    logging.info(f"Hurst data shape: {hurst_data.shape}, first few values: {hurst_data[:5]}")
                    
                    H, c, data = compute_Hc(hurst_data, kind="price", simplified=True)
                    logging.info(f"Computed Hurst exponent: {H}")
                    
                    if 0 <= H <= 1:  # Validate Hurst exponent is in valid range
                        # Set the Hurst value for the most recent data point
                        hurst_idx = chrono_data.columns.get_loc("hurst_exponent")
                        chrono_data.iloc[-1, hurst_idx] = round(H, 4)
                        logging.info(f"Set Hurst value {round(H, 4)} at index {hurst_idx}")
                    else:
                        logging.warning(f"Hurst exponent {H} out of valid range [0,1]")
                except Exception as e:
                    logging.error(f"Error calculating Hurst exponent: {e}")
            else:
                logging.info(f"Skipping Hurst calculation - already has value: {current_hurst}")
        else:
            logging.info(f"Not enough data for Hurst calculation - need 100, have {len(chrono_data)}")

        # Sort back to descending order for displaying most recent first and update data
        self.data_list = chrono_data.sort_values(by="time", ascending=False).reset_index(drop=True)
        
        # Log the final dataframe for debugging
        logging.info(f"Final DataFrame generated with {len(self.data_list)} rows:")
        logging.info(f"\n{self.data_list.head(10).to_string()}")

    def get_sorted_data(self) -> pd.DataFrame:
        """Get data sorted by time (most recent first)."""
        return self.data_list

    def determine_price_direction(self) -> str:
        """Determine if price is moving up, down, or neutral."""
        if len(self.data_list) >= 2:
            current_close = self.data_list.iloc[0]["close"]
            previous_close = self.data_list.iloc[1]["close"]

            if current_close > previous_close:
                return "up"
            elif current_close < previous_close:
                return "down"
            else:
                return "neutral"

        return "not enough data"

    def calculate_ema_slope(self, period: int = EMA_WINDOW) -> Tuple[float, str]:
        """Calculate EMA slope and direction."""
        if len(self.data_list) < period + 1:
            return 0.0, "not enough data"

        if "ema_slope" in self.data_list.columns:
            # Slope is already calculated in the dataframe
            slope_degrees = self.data_list["ema_slope"].iloc[0]

            if slope_degrees > 0:
                direction = "up"
            elif slope_degrees < 0:
                direction = "down"
            else:
                direction = "neutral"

            return slope_degrees, direction
        else:
            # Fallback calculation if slope not pre-calculated
            temp_data = self.data_list.sort_values(by="time", ascending=True).reset_index(drop=True)

            ema_indicator = EMAIndicator(close=pd.Series(temp_data["close"]), window=period, fillna=True)
            ema = ema_indicator.ema_indicator()

            raw_slope = ema.iloc[-1] - ema.iloc[-2]
            slope_degrees = round(math.degrees(math.atan(raw_slope)), 2)

            if slope_degrees > 0:
                direction = "up"
            elif slope_degrees < 0:
                direction = "down"
            else:
                direction = "neutral"

            return slope_degrees, direction

    def get_slope_deviation(self) -> float:
        """Get the most recent slope deviation value."""
        if len(self.data_list) < 2 or 'slope_deviation' not in self.data_list.columns:
            return 0.0

        return self.data_list["slope_deviation"].iloc[0]

    def calculate_hurst_exponent(self) -> float:
        """Calculate Hurst Exponent for trend persistence analysis."""
        if len(self.data_list) < 100:
            return 0.0

        # Return pre-calculated value if available
        if 'hurst_exponent' in self.data_list.columns and len(self.data_list) > 0:
            hurst_value = self.data_list["hurst_exponent"].iloc[0]
            if hurst_value > 0:  # Only return if we have a valid calculated value
                return hurst_value

        # Fallback calculation if not pre-calculated or value is 0
        recent_data = self.data_list.head(HURST_WINDOW)
        price_data = recent_data["close"].values

        if len(price_data) < 100:
            return 0.0

        try:
            H, c, data = compute_Hc(price_data, kind="price", simplified=True)
            if 0 <= H <= 1:
                return round(H, 4)
            else:
                return 0.0
        except Exception:
            return 0.0

    def check_trading_signals(self) -> Optional[str]:
        """Check for buy/sell signals based on threshold criteria."""
        if len(self.data_list) == 0:
            return None
            
        # Get the most recent values for all indicators
        current_row = self.data_list.iloc[0]
        
        # Extract values with validation
        ema_slope = current_row.get('ema_slope', 0) if 'ema_slope' in self.data_list.columns else 0
        slope_deviation = current_row.get('slope_deviation', 0) if 'slope_deviation' in self.data_list.columns else 0
        hurst_exponent = current_row.get('hurst_exponent', 0) if 'hurst_exponent' in self.data_list.columns else 0
        
        # Ensure all values are valid numbers
        if pd.isna(ema_slope) or pd.isna(slope_deviation) or pd.isna(hurst_exponent):
            return None
        
        # Convert to float to ensure numeric comparison
        ema_slope = float(ema_slope)
        slope_deviation = float(slope_deviation)
        hurst_exponent = float(hurst_exponent)
    
        # Log current indicator values for signal checking
        logging.info(f"Signal Check - EMA Slope: {ema_slope}, Slope Deviation: {slope_deviation}, Hurst Exponent: {hurst_exponent}")
    
        # Check for BUY signal: ema_slope > 50, slope_deviation > 50, hurst_exponent > 50
        if ema_slope > 50 and slope_deviation > 50 and hurst_exponent > 50:
            logging.info("BUY signal generated!")
            return "buy"
    
        # Check for SELL signal: ema_slope < -50, slope_deviation > 50, hurst_exponent > 50
        elif ema_slope < -50 and slope_deviation > 50 and hurst_exponent > 50:
            logging.info("SELL signal generated!")
            return "sell"
        
        return None

    def reset(self) -> None:
        """Reset all data and cache."""
        self.data_list = pd.DataFrame({
            "time": pd.Series(dtype='object'),
            "open": pd.Series(dtype='float64'),
            "high": pd.Series(dtype='float64'),
            "low": pd.Series(dtype='float64'),
            "close": pd.Series(dtype='float64'),
            "volume": pd.Series(dtype='float64'),
            "hurst_exponent": pd.Series(dtype='float64')
        })
        self.time_cache.clear()
