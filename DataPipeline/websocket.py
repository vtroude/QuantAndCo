import numpy    as np

from typing import List, Callable, Optional

from DataPipeline.forex_oanda_data  import websocket    as fx_socket
from DataPipeline.binance_data      import websocket    as crypto_socket

#######################################################################################################################

def websocket(
                action: Callable[[str, float], None],
                symbols: List[str],
                market: str
            ) -> Optional[Callable]:
    
    """
    Open a Websocket to take action at each new price for each price

    Input:
        - action:   A callable function which takes a symbol and the price
        - symbols:  List of symbols
        - market:   Type of market e.g. 'forex', 'crypto', ...
    """

    if market == "forex":
        symbols = ",".join(symbols)
        # Create the streaming service to take action on forex
        async def process_streams():
            async for ticks in fx_socket(symbols):
                try:
                    bid = float(ticks["closeoutBid"])  # np.min([float(b["price"]) for b in ticks['bids']])
                    ask = float(ticks["closeoutAsk"])  # np.max([float(b["price"]) for b in ticks['asks']])

                    action(ticks["instrument"], 0.5*(bid + ask))
                except:
                    pass

        return process_streams
    
    elif market == "crypto":

        # Create the streaming service to take action on cryptos
        async def process_streams():
            async for ticks in crypto_socket(symbols):
                action(ticks["data"]["s"], float(ticks["data"]["p"]))

        return process_streams
    
    return None

#######################################################################################################################

if __name__ == "__main__":
    import asyncio

    #market  = "forex"
    #symbols = ["EUR_USD", "AUD_USD"]

    market  = "crypto"
    symbols = ["BTCUSDT", "ETHUSDT"]

    def action(symbol: str, price: float):
        print(symbol, price)
    
    process = websocket(action, symbols, market)
    loop    = asyncio.get_event_loop()
    loop.run_until_complete(process())





