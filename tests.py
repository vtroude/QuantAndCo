import sys
import time

import numpy    as np

from QuantAnCo.DataPipeline.technicals_indicators  import TechnicalIndicators

sys.path.append("/home/virgile/Desktop/Trading/NonNormalStudy/")
from binance_get_price  import get_time_series

if __name__ == "__main__":
    import time

    # Calculate timestamps for the beginning and end of the 3-year period
    end_time    = int(time.time() * 1000)           # Binance API requires milliseconds
    start_time  = end_time - (24 * 60 * 60 * 1000)  # Last 1 year

    symbol      = 'BTCUSDT'
    interval    = '1m'

    data    = get_time_series(symbol, start_time, end_time, interval, chunk=1000)
    data    = data.rename(columns={o.lower(): o for o in ['Open', 'High', 'Low', 'Close', 'Volume']})
    print(data)
    ta      = TechnicalIndicators().get(data, span=[10, 20, 60], stat_span=[20, 100, 500])
    print(ta)