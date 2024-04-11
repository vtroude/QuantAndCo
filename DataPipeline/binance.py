import os

import numpy    as np
import pandas   as pd

from dotenv             import load_dotenv
from functools          import partial
from binance.client     import Client
from multiprocessing    import Pool

def fetch_candlesticks(client, symbol, interval, start_str, end_str):
    """Fetch historical candlestick data from Binance"""
    try:
        candlesticks = client.get_historical_klines(symbol, interval, start_str=start_str, end_str=end_str)
        df = pd.DataFrame(candlesticks, columns=['open_time', 'open', 'high', 'low', 'close', 'volume',
                                                 'close_time', 'quote_asset_volume', 'number_of_trades',
                                                 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        df.set_index('close_time', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        
        return df
    except:
        return None

def get_time_series(symbol, start_time, end_time, interval='1m', chunk=100):
    """Retrieve time series data in chunks"""

    load_dotenv()

    api_key     = os.getenv("BINANCE_API_KEY")
    api_secret  = os.getenv("BINANCE_PRIVATE_KEY")

    client = Client(api_key, api_secret)

    time_chunk    = np.linspace(start_time, end_time, chunk)
    time_chunk  = [(str(time_chunk[i-1]), str(time_chunk[i])) for i in range(1, chunk)]

    fetch_data = partial(fetch_candlesticks, client, symbol, interval)

    data = pd.DataFrame()
    with Pool(processes=10) as p:
        for df in p.starmap(fetch_data, time_chunk):
            if not df is None:
                data = pd.concat([data, df])
    
    data    = data.drop_duplicates().sort_index()

    return data

if __name__=="__main__":
    import time

    # Calculate timestamps for the beginning and end of the 3-year period
    end_time    = int(time.time() * 1000)                       # Binance API requires milliseconds
    start_time  = end_time - (3 * 365 * 24 * 60 * 60 * 1000)    # 3 years in milliseconds

    symbol      = 'BTCUSDT'
    interval    = '1m'

    data    = get_time_series(symbol, start_time, end_time, interval, chunk=1000)
    

