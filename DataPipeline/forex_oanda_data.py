import pandas   as pd

import oandapyV20
import oandapyV20.endpoints.instruments as instruments

#######################################################################################################################

def fetch_candlesticks(client: oandapyV20.API, symbol: str, interval: str, start_str: str, end_str: str) -> pd.DataFrame:
    """
    Fetch historical candlestick data from Binance
    
    Input:
        - Client:   oanda client to fetch data
        - symbol:   asset pair e.g. EUR_USD
        - interval: time interval over which the candlestick is measured e.g. 'M1', 'M5', 'H1', etc...
        - start_str:    start time in unix timestamp (unit in second)
        - end_str:      end time in unix timestamp (unit in second)
    
    Return:
        - Candlestick (if succeed)
    """
    
    try:
        ###########################################################################################
        """ Fetch Data """
        ###########################################################################################

        # Request Data from OANDA data base as dictionary
        r   = instruments.InstrumentsCandles(instrument=symbol, params={"from": start_str, "to": end_str, "granularity": interval})
        client.request(r)
        response    = r.response
        
        ###########################################################################################
        """ Format Data """
        ###########################################################################################

        # Get data 
        df  = []
        for entry in response["candles"]:
            df.append(
                {
                    "close_time":   entry["time"],
                    "Volume":       entry["volume"],
                    "Open":         entry["mid"]["o"],
                    "High":         entry["mid"]["h"],
                    "Low":          entry["mid"]["l"],
                    "Close":        entry["mid"]["c"]
                }
            )

        # Make DataFrame
        df  = pd.DataFrame(df)

        # Preparde Datetime index
        df['close_time']    = pd.to_datetime(df['close_time'])
        df.set_index('close_time', inplace=True)

        # Set the contenet of the DataFrame to be float
        df    = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

        print('Data was successfully downloaded.')
        
        # Return candlestick
        return df
    except:
        return None

#######################################################################################################################

if __name__ == "__main__":
    import os
    import time

    from dotenv import load_dotenv

    # Calculate timestamps for the beginning and end of the 3-year period
    end_time    = int(time.time())              # Unix timestamp in second
    start_time  = end_time - (3* 24 * 60 * 60)  # 3 days in second

    symbol      = "EUR_USD"
    interval    = 'M1'

    load_dotenv()

    client = oandapyV20.API(access_token=os.getenv("OANDA_API_KEY"))

    data    = fetch_candlesticks(client, symbol, interval, str(start_time), str(end_time))

    print(data)

