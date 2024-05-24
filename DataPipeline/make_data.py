import os

import pandas   as pd

from typing import Union

from DataPipeline.fetch_data                import get_time_series
from DataPipeline.technicals_indicators     import TechnicalIndicators
from DataPipeline.utils import convert_unix_to_datetime


"""
File to gather data and technical indicators from different sources
"""

#######################################################################################################################

def make_filename(
                    market: str,
                    symbol: str,
                    interval: str,
                    date1: Union[str, int],
                    date2: Union[str, int],
                    data_type: str
                ) -> str:
    """
    Make file name
    
    Input:
        - market:       market of the asset e.g. 'crypto', 'forex', etc...
        - symbol:       asset symbol e.g. 'BTCUSD'
        - interval:     Candlestick time interval e.g. '1m'
        - date1:        Date from which the data has been gathered in unix timestamp
        - date2:        Date to which the data has been gathered in unix timestamp
        - data_type:    type of data e.g. ohlc, ti (technical indicators), etc...
    
    Output:
        - Path to data
    """

    return f"Data/{market}/{symbol}/{interval}/{data_type}/{date1}_{date2}.csv"

#######################################################################################################################

def make_directory_structure(market: str, symbol: str, interval: str, data_type: str) -> None:
    """
    Create necessary directories to save files according to the path:
        market -> symbol -> interval -> data_type -> csv file
    
    Input:
        - market:       market of the asset e.g. 'crypto', 'forex', etc...
        - symbol:       asset symbol e.g. 'BTCUSD'
        - interval:     Candlestick time interval e.g. '1m'
        - data_type:    type of data e.g. ohlc, ti (technical indicators), etc...
    """

    repo    = ""

    architecture    = ["Data", market, symbol, interval, data_type]
    for arch in architecture:
        if not os.path.exists(f"{repo}{arch}"):
            os.mkdir(f"{repo}{arch}")
        
        repo    = f"{repo}{arch}/"

#######################################################################################################################

def get_and_save_timeseries(
                                symbol: str,
                                market: str,
                                start_time: int,
                                end_time: int,
                                interval: str = '1m',
                                chunk: int = 100,
                                n_jobs: int = 1
                            ) -> None:
    """
    Retrieve time series data as OHLC + Volume Candlesticks withe Datetime index and save
    
    Input:
        - symbol:       asset pair e.g. BTCUSDT
        - market:       market of the asset e.g. 'crypto', 'forex', etc...
        - start_time:   start time in unix timestamp x 1000 (unit in milisecond)
        - end_time:     end time in unix timestamp x 1000 (unit in milisecond)
        - interval:     time interval over which the candlestick is measured e.g. '1m', '5m', '1h', etc...
        - chunk:        Retrieve the data from start_tim to end_time in chunk number of times
        - n_jobs:       Number of jobs to use 
    """

    ###############################################################################################
    """Get Data"""
    ###############################################################################################

    # Get OHLC data 
    data    = get_time_series(symbol, market, start_time, end_time, interval, chunk, n_jobs)
    print(f"Successfully downloaded OHLC time series for {symbol}")

    ###############################################################################################
    """Save Data"""
    ###############################################################################################

    # make necessary directory to save the data
    make_directory_structure(market, symbol, interval, "OHLC")


    if market == 'forex':
        start_str, end_str = convert_unix_to_datetime(start_time), convert_unix_to_datetime(end_time)
    else:
        start_str, end_str = start_time, end_time

    # Save OHLC data
    data.to_csv(make_filename(market, symbol, interval, start_str, end_str, "OHLC"))
    print(f"Successfully saved OHLC time series for {symbol}")

#######################################################################################################################

def get_and_save_indicators(
                                symbol: str,
                                market: str,
                                start_time: int,
                                end_time: int,
                                interval: str,
                                **kwargs
                            ) -> None:
    """
    From saved time series get all technical indicators and save them
    
    Input:
        - symbol:       asset pair e.g. BTCUSDT
        - market:       market of the asset e.g. 'crypto', 'forex', etc...
        - start_time:   start time in unix timestamp x 1000 (unit in milisecond)
        - end_time:     end time in unix timestamp x 1000 (unit in milisecond)
        - interval:     time interval over which the candlestick is measured e.g. '1m', '5m', '1h', etc...
    """

    ###############################################################################################
    """ Get OHLC + Volume data from File """
    ###############################################################################################

    if market == 'forex':
        start_str, end_str = convert_unix_to_datetime(start_time), convert_unix_to_datetime(end_time)
    else:
        start_str, end_str = start_time, end_time


    # Format start and end time from Unix timestamp to Datetime string
    ohlc        = pd.read_csv(make_filename(market, symbol, interval, start_str, end_str, "OHLC"), index_col=0)
    ohlc.index  = pd.to_datetime(ohlc.index)

    ###############################################################################################
    """ Get Technical Indicators """
    ###############################################################################################

    ta, ta_last = TechnicalIndicators().get(ohlc, **kwargs)
    print(f"Successfully downloaded Technical Indicators time series for {symbol}")

    ###############################################################################################
    """ Save Indicators """
    ###############################################################################################

    # make necessary directory to save the data
    make_directory_structure(market, symbol, interval, "Indicators")
    # make necessary directory to save the data
    make_directory_structure(market, symbol, interval, "Last")

    ta.to_csv(make_filename(market, symbol, interval, start_time, end_time, "Indicators"))  # Save technical indicators
    ta_last.to_csv(make_filename(market, symbol, interval, start_time, end_time, "Last"))   # Save data to build future statistics
    print(f"Successfully saved Technical Indicators time series for {symbol}")

#######################################################################################################################

if __name__ == "__main__":
    import time

    chunk       = 1000
    n_per_chunk = 2500

    # Calculate timestamps for the beginning and end
    #end_time    = int(time.time())
    #start_time  = end_time - ( chunk * n_per_chunk * 60 )

    start_time, end_time    = 1563535876, 1713535876
    start_str, end_str = convert_unix_to_datetime(start_time), convert_unix_to_datetime(end_time)
    print(start_str)
    print(end_str)

    span        = [10, 30, 50, 100, 200, 500, 1000, 2000, 5000, 10_000, 15_000, 20_000, 50_000, 100_000]      # Different windows over which we compute the technical indicators
    stat_span   = [20, 100, 500]    # Different windows over which we compute statistics

    symbols      = ["CHF_USD"]                # Symbol over which we gather the data      'BTCUSDT'
    market      = "forex"                   # Market from which we gather data          'crypto'
    interval    = ['1m']        # Time interval to make the candlestick

    span        = [10, 30, 90, 200, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000]      # Different windows over which we compute the technical indicators
    stat_span   = [20, 50, 100]    # Different windows over which we compute statistics
    
    for symbol in symbols:
        for i in interval:
            get_and_save_timeseries(symbol, market, start_time, end_time, i, chunk, n_jobs=10)
            print(f"Successfully downloaded and save OHLC data for {symbol} at {i} interval")
            #get_and_save_indicators(symbol, market, start_time, end_time, i, span=span, stat_span=stat_span, n_jobs=10)
            #print(f"Successfully downloaded and save TI data for {symbol} at {i} interval")
