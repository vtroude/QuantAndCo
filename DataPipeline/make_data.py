import os
import time

from datetime   import datetime

from DataPipeline.binance_data              import get_time_series
from DataPipeline.technicals_indicators     import TechnicalIndicators


"""
File to gather data and technical indicators from different sources
"""

#######################################################################################################################

def make_filename(symbol: str, interval: str, date1: str, date2: str, data_type: str) -> str:
    """
    Make file name
    
    Input:
        - symbol:       asset symbol e.g. 'BTCUSD'
        - interval:     Candlestick time interval e.g. '1m'
        - date1:        Date from which the data has been gathered in %Y-%m-%d-%H-%M-%S
        - date2:        Date to which the data has been gathered in %Y-%m-%d-%H-%M-%S
        - data_type:    type of data e.g. ohlc, ti (technical indicators), etc...
    
    Output:
        - Path to data
    """

    return f"Data/{symbol}_{interval}_{date1}_{date2}_{data_type}.csv"

#######################################################################################################################

def get_binance_data(
                        start_time: int,
                        end_time: int,
                        symbol: str,
                        interval: str,
                        **kwargs
                    ) -> None:
    """
    Get Candlestick and gather technical indicators

    start_time: Time from which we gather the data in Unix timstamp * 1000
    end_time:   Time to which we gather the data in Unix timstamp * 1000
    symbol:     Symbol of the data e.g. 'BTCUSDT'
    interval:   Time interval over which the candle stick is build e.g. '1m', '1h', '1d', '1w'
    **kwargs:   Arguments to pass to TechnicalIndicators().get()
    """

    ###############################################################################################
    """Get Data"""
    ###############################################################################################

    # Get OHLC data from Binance
    data    = get_time_series(symbol, start_time, end_time, interval, chunk=1000)
    # Format the columns
    data    = data.rename(columns={o.lower(): o for o in ['Open', 'High', 'Low', 'Close', 'Volume']})
    # Get rolling technical indicators e.g. RSI, TSI, Money FLow, etc...
    # And Statistics e.g. volatility, mean return, kurtosis, etc...
    ta, ta_last = TechnicalIndicators().get(data, **kwargs)

    ###############################################################################################
    """Save Data"""
    ###############################################################################################

    # Format start and end time from Unix timestamp to Datetime string
    date1   = datetime.fromtimestamp(start_time//1000).strftime('%Y-%m-%d-%H:%M:%S')
    date2   = datetime.fromtimestamp(end_time//1000).strftime('%Y-%m-%d-%H:%M:%S')

    # Check if a directory Data exists, and if not, make one
    if not os.path.exists("Data"):
        os.mkdir("Data")

    data.to_csv(make_filename(symbol, interval, date1, date2, "ohlc"))      # Save OHLC data
    ta.to_csv(make_filename(symbol, interval, date1, date2, "ti"))          # Save technical indicators
    ta_last.to_csv(make_filename(symbol, interval, date1, date2, "last"))   # Save data to build future statistics

#######################################################################################################################

if __name__ == "__main__":
    import time

    # Calculate timestamps for the beginning and end of the 3-year period
    end_time    = int(time.time() * 1000)                       # Binance API requires milliseconds
    start_time  = end_time - (3 * 365 * 24 * 60 * 60 * 1000)    # Last 3 year

    symbol      = 'ETHUSDT'  #'BTCUSDT'     # Symbol over which we gather the data
    interval    = ['1m', '1h', '1d', '1w']  # Time interval to make the candlestick

    span        = [10, 20, 60]      # Different windows over which we compute the technical indicators
    stat_span   = [20, 100, 500]    # Different windows over which we compute statistics

    for i in interval:
        get_binance_data(start_time, end_time, symbol, i, span=span, stat_span=stat_span)