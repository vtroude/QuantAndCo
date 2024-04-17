from Strategy.backtest import backtest_RSI_strategy
from DataPipeline.technicals_indicators     import TechnicalIndicators
from DataPipeline.make_data import make_filename
import pandas as pd
import time

if __name__ == '__main__':

    start_time = time.time()
    symbol = 'BTCUSDT'
    interval = '5m'
    start_date = '2020-01-01'
    end_date = '2024-12-04'

    data = pd.read_csv(make_filename(symbol, interval, start_date, end_date, 'ohlc'))
    data = data.rename(columns={o.lower(): o for o in ['Open', 'High', 'Low', 'Close', 'Volume']})
    ti = pd.read_csv(make_filename(symbol, interval, start_date, end_date, 'ti'))
    span        = [20, 100, 200, 500] 
    ti_cols = ['StochRSI-' + str(s) for s in span]
    ti = ti[ti_cols + ['timestamp']]
    backtest_df = backtest_RSI_strategy(data, ti, entry_long_thres=0.1, entry_short_thres=0.9,
                                        exit_long_thres=0.2, exit_short_thres=0.8, leverage=5)
    file_name = 'backtest_RSI_' + str([s for s in span])
    backtest_df.to_csv(make_filename(symbol, interval, start_date, end_date, file_name))
    end_time = time.time()

    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Elapsed time: {elapsed_time} seconds")

    #print(backtest_df.head())