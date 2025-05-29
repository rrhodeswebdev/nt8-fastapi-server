"""Configuration settings for NT8 Data Analysis."""

# Data processing settings
MAX_DATA_ENTRIES = 250
EMA_WINDOW = 8
SLOPE_DEVIATION_PERIOD = 8
HURST_WINDOW = 100

# Trading signal thresholds
BUY_SIGNAL_THRESHOLD = 70.0  # Percentage increase in slope deviation
SELL_SIGNAL_THRESHOLD = -75.0  # Percentage decrease in slope deviation

# API settings
API_TITLE = "NT8 AI Server"
TEST_RESPONSE = "test complete"
RESET_RESPONSE = "data reset"
HEALTH_CHECK_RESPONSE = "Hello World"

# DataFrame display settings
ENABLE_DATAFRAME_DISPLAY = True  # Set to False to disable DataFrame printing
VERBOSE_LOGGING = True  # Set to False for minimal output
COMPACT_DATAFRAME_DISPLAY = False  # Set to True for more compact DataFrame output
MAX_DATAFRAME_ROWS = 10  # Maximum rows to show in DataFrame (0 = show all)
