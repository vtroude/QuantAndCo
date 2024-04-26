import pandas   as pd

from typing import List, Union

from binance            import AsyncClient, BinanceSocketManager
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

async def websocket(symbols: Union[str, List[str]]):
    """
    Create a Binance WebSocket to stream data for the specified symbol.

    Args:
    - symbol (str): Trading pair symbol, e.g., 'BNBBTC'.

    Returns:
    - Async generator yielding streamed data.
    """
    
    client  = await AsyncClient.create()
    bm      = BinanceSocketManager(client)

    if isinstance(symbols, list):

        # Create a list of stream names
        streams = [symbol.lower() + '@trade' for symbol in symbols]

        async with bm.multiplex_socket(streams) as multiplex_socket:
            while True:
                res = await multiplex_socket.recv()
                yield res  # Yield the websocket response for external processing
    else:

        async with bm.trade_socket(symbols) as trade_socket:
            while True:
                res = await trade_socket.recv()
                yield res  # Yield the websocket response for external processing

    await client.close_connection()

#######################################################################################################################
""" Test Method """
#######################################################################################################################

def test_fetch_candlesticks() -> None:
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

#######################################################################################################################

if __name__=="__main__":
    import asyncio

    async def process_streams():
        """
        Process messages from Binance WebSocket.
        """
        async for message in websocket(["BTCUSDT", "ETHUSDT"]):
            print("Received message:", message)
    
    loop    = asyncio.get_event_loop()
    loop.run_until_complete(process_streams())
    

