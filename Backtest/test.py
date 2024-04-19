import pandas as pd
from Strategy.rsi_strategy import RSIStrategy
from Backtest.metrics import backtest_metrics, save_plot_strategy
from DataPipeline.make_data import make_filename
from DataPipeline.technicals_indicators import TechnicalIndicators
from Backtest.backtest import Backtest_Strategy, backtest_metrics
from Backtest.metrics import save_plot_strategy
import time
import os

if __name__ == '__main__':

    start_time = time.time()
    symbol = 'BTCUSDT'
    interval = '5m'
    start_date = '2020-01-01'
    end_date = '2024-12-04'
    backtest_end_date = '2021-01-01'
    price_df_path = make_filename(symbol, interval, start_date, end_date, 'ohlc')
    price_df = pd.read_csv(price_df_path)
    price_df = price_df.rename(columns={o.lower(): o for o in ['Open', 'High', 'Low', 'Close', 'Volume']})
    price_df.index = pd.to_datetime(price_df.timestamp)
    span        = [20, 100,200, 500, 1000, 2000, 5000, 10_000] 
    stat_span   = [20]
    ti_filename = 'ti_' + str([s for s in span])
    ti_path = make_filename(symbol, interval, start_date, backtest_end_date, ti_filename)
    if not os.path.exists(ti_path):
        ti, _ = TechnicalIndicators().get(price_df, span=span, stat_span=stat_span)
        ti.to_csv(ti_path)
    else:
        ti = pd.read_csv(ti_path)
    ti_cols = ['StochRSI-' + str(s) for s in span]
    ti = ti[ti_cols + ['timestamp']]
    file_name = 'backtest_RSI_' + str([s for s in span])
    path = make_filename(symbol, interval, start_date, backtest_end_date, file_name)

    if not os.path.exists(path):
        strategy = RSIStrategy(price_df, ti, ti_cols, long_thres=0.1, short_thres=0.9)
        strategy_df = strategy.prepare_data()
        strategy_df['signal'] = strategy.entry_signals()
        backtest_df = Backtest_Strategy(strategy_df, end_date=backtest_end_date, leverage=10)
        backtest_df.to_csv(path)
    else:
        backtest_df = pd.read_csv(path)


    metrics = backtest_metrics(backtest_df, interval, n_trading_hours=24, n_trading_days=365,
                     wealth_column='Wealth')
    
    print(metrics)
    save_plot_strategy(backtest_df)


    end_time = time.time()

    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Elapsed time: {elapsed_time} seconds")