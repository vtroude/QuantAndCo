from DataPipeline.technicals_indicators     import TechnicalIndicators
from DataPipeline.make_data import make_filename
import pandas as pd

if __name__ == '__main__':

    symbol = 'BTCUSDT'
    interval = '5m'
    start_date = '2020-01-01'
    end_date = '2024-12-04'

    data = pd.read_csv(make_filename(symbol, interval, start_date, end_date, 'ohlc'))
    data = data.rename(columns={o.lower(): o for o in ['Open', 'High', 'Low', 'Close', 'Volume']})
    data.timestamp = pd.to_datetime(data.timestamp)
    data.set_index('timestamp', inplace=True)
    # Get rolling technical indicators e.g. RSI, TSI, Money FLow, etc...
    # And Statistics e.g. volatility, mean return, kurtosis, etc...
    span        = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10_000, 12_500, 15_000, 20_000, 30_000, 45_000, 100_000]      # Different windows over which we compute the technical indicators
    stat_span   = [20] 

    ta, ta_last = TechnicalIndicators().get(data, span=span, stat_span=stat_span)
    ti_filename = 'ti_' + str([s for s in span])
    ta.to_csv(make_filename(symbol, interval, start_date, end_date, ti_filename))          # Save technical indicators
    #ta_last.to_csv(make_filename(symbol, interval, start_date, end_date, "last")) 
    print('Successfully saved Technical Indicator data')