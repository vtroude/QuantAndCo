import pandas   as pd

from binance.client     import Client

#######################################################################################################################

def fetch_candlesticks(client: Client, symbol: str, interval: str, start_str: str, end_str: str) -> pd.DataFrame:
    """
    Fetch historical candlestick data from Binance
    
    Input:
        - Client:   binance client to fetch data
        - symbol:   asset pair e.g. BTCUSDT
        - interval: time interval over which the candlestick is measured e.g. '1m', '5m', '1h', etc...
        - start_str:    start time in unix timestamp x 1000 (unit in milisecond)
        - end_str:      end time in unix timestamp x 1000 (unit in milisecond)
    
    Return:
        - Candlestick (if succeed)
    """
    
    try:
        # Get data from binance API as dictionary
        candlesticks = client.get_historical_klines(symbol, interval, start_str=start_str, end_str=end_str)
        # Make Dataframe from dictionary
        df = pd.DataFrame(candlesticks, columns=['open_time', 'open', 'high', 'low', 'close', 'volume',
                                                 'close_time', 'quote_asset_volume', 'number_of_trades',
                                                 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        # Set datetime index
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        df.set_index('close_time', inplace=True)
        # force data to be float
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        # Format column names
        data    = data.rename(columns={o.lower(): o for o in ['Open', 'High', 'Low', 'Close', 'Volume']})
        
        return df
    except:
        return None

#######################################################################################################################

if __name__=="__main__":
    import os
    import time

    from dotenv import load_dotenv

    # load variable from env file
    load_dotenv()

    # Get Binance API keys
    api_key     = os.getenv("BINANCE_API_KEY")
    api_secret  = os.getenv("BINANCE_PRIVATE_KEY")

    # Return Binance Client
    client  = Client(api_key, api_secret)

    # Calculate timestamps for the beginning and end of the 3-year period
    end_time    = int(time.time() * 1000)           # Binance API requires milliseconds
    start_time  = end_time - (24 * 60 * 60 * 1000)  # 24h in milliseconds

    symbol      = 'BTCUSDT'
    interval    = '1m'

    data    = fetch_candlesticks(client, symbol, interval, str(start_time), str(end_time))

    print(data)
    

