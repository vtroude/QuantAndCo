import os

import numpy    as np
import pandas   as pd
from functools          import partial
from binance.client     import Client
from multiprocessing    import Pool
from datetime import datetime


from DataPipeline.make_data import make_filename

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

if __name__=="__main__":
    symbol = 'BTCUSDT'
    interval = '5m'
    start_date = '2020-01-01'
    start_str = datetime.strptime(start_date, '%Y-%m-%d')
    start_str = start_str.strftime('%b %-d, %Y')
    end_date = '2024-12-04'
    end_str = datetime.strptime(end_date, '%Y-%m-%d')
    end_str = end_str.strftime('%b %-d, %Y')
    api_key     = os.getenv("BINANCE_API_KEY")
    api_secret  = os.getenv("BINANCE_PRIVATE_KEY")
    client = Client(api_key, api_secret)
    df = client.get_historical_klines(symbol, interval=interval, start_str=start_str, end_str=end_str)
    df = pd.DataFrame(df)
    df = df.iloc[:, :6]
    df.columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    df.timestamp = pd.to_datetime(df.timestamp, unit='ms')
    df.set_index('timestamp', inplace=True)
    for col in df.columns:
        df[col] = df[col].astype('float')

    # Check if a directory Data exists, and if not, make one
    if not os.path.exists("Data"):
        os.mkdir("Data")

    df.to_csv(make_filename(symbol, interval, start_date, end_date, "ohlc"))
