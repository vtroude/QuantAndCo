import pandas as pd
from Backtest.backtest import Backtest
#from Backtest.metrics import backtest_metrics, save_plot_strategy
from DataPipeline.make_data import make_filename
from DataPipeline.technicals_indicators import TechnicalIndicators
#from Backtest.backtest import Backtest_Strategy, backtest_metrics
#from Backtest.metrics import save_plot_strategy
from Strategy.RSI.rsi_strategy import RSIStrategy
import time
import os
from Strategy.RSI.utils import prepare_data

if __name__ == '__main__':

    backtest_end_date = '2024-04-01'

    start_time = time.time()
    price_df = pd.read_csv("Data/forex/EUR_USD/1m/OHLC/2019-07-19 13:31:16_2024-04-19 16:11:16.csv")
    price_df = price_df.rename(columns={o.lower(): o for o in ['Open', 'High', 'Low', 'Close', 'Volume']})
    price_df['timestamp'] = price_df['close_time']
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Time to read Price DF: {elapsed_time} seconds')
    #price_df.index = pd.to_datetime(price_df.timestamp)
    start_time = time.time()
    ti = pd.read_csv("Data/forex/EUR_USD/1m/Indicators/2019-07-19 13:31:16_2024-04-19 16:11:16.csv")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Time to read TI DF: {elapsed_time} seconds')
    ti['timestamp'] = ti['close_time']
    span = [120, 500, 1000, 5000, 10080] 
    stat_span   = [20, 100, 500]
    ti_cols = ['StochTSI-' + str(s) for s in span]
    start_time = time.time()
    strategy_df = prepare_data(price_df, ti)
    strategy = RSIStrategy(strategy_df=strategy_df, indicators_cols=ti_cols, 
                           long_thres=0.1, short_thres=0.9)
    strategy_df['signal'] = strategy.generate_signals()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Time to generate signals: {elapsed_time} seconds')
    #strategy_df.to_csv('Data/forex/EUR_USD/S.csv')
    start_time = time.time()
    backtesting = Backtest(strategy_df=strategy_df, symbol='EUR_USD', market='forex',
                           interval='1m', end_date=backtest_end_date, leverage=5,
                           fees=0.00007, take_profit=0.2, stop_loss=-0.05)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Time to backtest strategy: {elapsed_time} seconds')

    
    #file_name = 'backtest_RSI_' + str([s for s in span]) + '.csv'
    #backtest_df.to_csv('Data/forex/EUR_USD/backtest_test.csv')
    #backtest_df.to_csv(make_filename('forex', 'EUR_USD', '1m', '2019-07-19', backtest_end_date, file_name))
    #print(ti.head(10))

    #metrics = backtest_metrics(backtest_df, '1m', n_trading_hours=24, n_trading_days=252,
    #                 wealth_column='Wealth')
    
    #print(metrics)
    #save_plot_strategy(backtest_df)


    #end_time = time.time()

    #elapsed_time = end_time - start_time  # Calculate elapsed time
    #print(f"Elapsed time: {elapsed_time} seconds")
    