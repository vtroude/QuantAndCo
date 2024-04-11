import time

from datetime   import datetime

from DataPipeline.binance               import get_time_series
from DataPipeline.technicals_indicators import TechnicalIndicators

def make_filename(symbol: str, interval: str, date1: str, date2: str, data_type: str):
    """
    Make file name
    
    Input:
        - symbol:       asset symbol e.g. 'BTCUSD'
        - interval:     Candlestick time interval e.g. '1m'
        - date1:        Date from which the data has been gathered in %Y-%m-%d-%H-%M-%S
        - date2:        Date to which the data has been gathered in %Y-%m-%d-%H-%M-%S
        - data_type:    type of data e.g. ohlc, ti (technical indicators), etc...
    """

    return f"Data/{symbol}_{interval}_{date1}_{date2}_{data_type}.csv"

def get_binance_data(start_time, end_time, symbol, interval, span=10, stat_span=20, n_process=5):
    """
    Get Candlestick and gather technical indicators
    """

    data    = get_time_series(symbol, start_time, end_time, interval, chunk=1000)
    data    = data.rename(columns={o.lower(): o for o in ['Open', 'High', 'Low', 'Close', 'Volume']})
    ta, ta_last = TechnicalIndicators().get(data, span=span, stat_span=stat_span, n_processes=n_process)

    date1   = datetime.fromtimestamp(start_time//1000).strftime('%Y-%m-%d-%H:%M:%S')
    date2   = datetime.fromtimestamp(end_time//1000).strftime('%Y-%m-%d-%H:%M:%S')

    data.to_csv(make_filename(symbol, interval, date1, date2, "ohlc"))
    ta.to_csv(make_filename(symbol, interval, date1, date2, "ti"))
    ta_last.to_csv(make_filename(symbol, interval, date1, date2, "last"))

if __name__ == "__main__":
    import time

    # Calculate timestamps for the beginning and end of the 3-year period
    end_time    = int(time.time() * 1000)                       # Binance API requires milliseconds
    start_time  = end_time - (3 * 365 * 24 * 60 * 60 * 1000)    # Last 3 year

    symbol      = 'BTCUSDT'
    interval    = ['1m', '1h', '1d', '1w']

    span        = [10, 20, 60]
    stat_span   = [20, 100, 500]

    for i in interval:
        get_binance_data(start_time, end_time, symbol, i, span, stat_span)