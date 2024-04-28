import os
import re
import oandapyV20

import numpy    as np
import pandas   as pd

from typing             import Optional, Any, Callable
from dotenv             import load_dotenv
from functools          import partial
from binance.client     import Client
from multiprocessing    import Pool

from DataPipeline   import binance_data, forex_oanda_data

#######################################################################################################################

def get_client(market: str) -> Any:
    """
    Gather a web client to fetch data from Binance, OANDA, etc...

    Inputs:
        - market:   market from which we are fetching data e.g. 'crypto', 'forex', etc...
    
    Return:
        - Web client
    """

    load_dotenv()   # load variable from env file
    
    if market == "crypto":
        # Get Binance API keys
        api_key     = os.getenv("BINANCE_API_KEY")
        api_secret  = os.getenv("BINANCE_PRIVATE_KEY")

        # Return Binance Client
        return Client(api_key, api_secret)
    
    elif market == "forex":
        # Return OANDA client
        return oandapyV20.API(access_token=os.getenv("OANDA_API_KEY"))
    
    return None

#######################################################################################################################

def get_interval_format(market: str, interval: str) -> str:
    """
    Inputs:
        - market:       market of the asset e.g. 'crypto', 'forex', etc...
        - interval:     time interval over which the candlestick is measured e.g. '1m', '5m', '1h', etc...
    
    Return:
        - interval in the right format
    """

    if market == "forex":
        for l in ["d", "w"]:
            if l in interval:
                return l.upper()
        
        match = re.match(r"(\d+)([a-zA-Z])", interval)
        
        number = match.group(1)
        letter = match.group(2).upper()  # Convert to upper case
        
        # Rearrange format
        return f"{letter}{number}"

    return interval


######################################################################################################################

def get_fetch_candlestick(client: Any, symbol: str, interval: str, market: str) -> Optional[Callable]:
    """
    Get function to fetch candlestick from a database such that the function will only take as an input
    the starting and ending time in unix timestamp

    Inputs:
        - Client:       web client to fetch data e.g. binance or oanda
        - symbol:       asset pair e.g. BTCUSDT
        - interval:     time interval over which the candlestick is measured e.g. '1m', '5m', '1h', etc...
        - market:       market of the asset e.g. 'crypto', 'forex', etc...

    Return:
        - Function which takes start and end time to fetch data
    """

    if market == "crypto":
        return partial(binance_data.fetch_candlesticks, client, symbol, interval)
    elif market == "forex":
        return partial(forex_oanda_data.fetch_candlesticks, client, symbol, interval)

    return None

#######################################################################################################################



#######################################################################################################################

def get_time_series(
                        symbol: str,
                        market: str,
                        start_time: int,
                        end_time: int,
                        interval: str,
                        chunk: int = 100,
                        n_jobs: int = 1
                    ) -> pd.DataFrame:
    """
    Retrieve time series data as OHLC + Volume Candlesticks withe Datetime index
    
    Input:
        - symbol:       asset pair e.g. BTCUSDT
        - market:       market of the asset e.g. 'crypto', 'forex', etc...
        - start_time:   start time in unix timestamp x 1000 (unit in milisecond)
        - end_time:     end time in unix timestamp x 1000 (unit in milisecond)
        - interval:     time interval over which the candlestick is measured e.g. '1m', '5m', '1h', etc...
        - chunk:        Retrieve the data from start_tim to end_time in chunk number of times
        - n_jobs:       Number of jobs to use 
    
    Return:
        - Candlestick (if succeed)
    """

    ###############################################################################################
    """ Set Binance Client """
    ###############################################################################################

    client  = get_client(market)

    ###############################################################################################
    """ Prepare Data """
    ###############################################################################################
    
    # If market is crypto, convert second to millisecond for binance api
    start_time  = start_time*1000 if market == "crypto" else start_time
    end_time    = end_time*1000 if market == "crypto" else end_time

    # Prepare time to chunk to fetch data from t_{i-1} to t_{i} in loop
    time_chunk  = np.linspace(start_time, end_time, chunk)
    time_chunk  = [(str(time_chunk[i-1]), str(time_chunk[i])) for i in range(1, chunk)]

    # Get interval over which to build the candle stick in the right format
    interval    = get_interval_format(market, interval)

    # Prepare function to fetch data with a single input
    fetch_data  = get_fetch_candlestick(client, symbol, interval, market)

    ###############################################################################################
    """ Fetch Data """
    ###############################################################################################

    data = pd.DataFrame()
    if n_jobs > 1:
        # Fetch data directly
        for time_c in time_chunk:
            data = pd.concat([data, fetch_data(time_c[0], time_c[1])])

    else:
        # Fetch data in parallel by using multiprocessing
        with Pool(processes=n_jobs) as p:
            for df in p.starmap(fetch_data, time_chunk):
                if not df is None:
                    data = pd.concat([data, df])
    
    # Drop duplicates if any and sort datetime index
    data    = data.drop_duplicates().sort_index()

    # return OHLC + Volume data
    return data

#######################################################################################################################



#######################################################################################################################

if __name__ == "__main__":
    import time

    chunk       = 100
    n_per_chunk = 2500

    # Calculate timestamps for the beginning and end of the 3-year period
    end_time    = int(time.time())                       # Unix timestamp in second
    start_time  = end_time - (chunk * n_per_chunk * 60)   # 1h in second

    symbol      = 'EUR_USD'  # "BTCUSD"
    interval    = '1m'
    market      = 'forex'  # "crypto"

    data    = get_time_series(symbol, market, start_time, end_time, interval, chunk, n_jobs=5)

    print(data)