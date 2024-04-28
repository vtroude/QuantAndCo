import pandas as pd
import numpy as np
from matplotlib import pylab as pl
import os
import logging

import logging
import os

def set_logs(log_file_name):
    # Create a logger object
    logger = logging.getLogger(__name__)

    # Check if the logger has handlers already
    if not logger.handlers:
        # Set the level of the logger
        logger.setLevel(logging.DEBUG)

        # Ensure the directory for logs exists
        log_directory = "Backtest/Logs"
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)  # Use makedirs which can create intermediate directories if needed

        # Create a file handler which logs even debug messages
        fh = logging.FileHandler(f'{log_directory}/{log_file_name}.log')
        fh.setLevel(logging.DEBUG)

        # Create a console handler that also logs debug messages
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger

def market_trading_rules(market):
    if market == 'forex':
        n_trading_hours = 24
        n_trading_days = 252
    elif market == 'crypto':
        n_trading_hours = 24
        n_trading_days = 365
    else:
        raise ValueError(f'{market} not available. Please choose "crypto" or "forex"')
    
    return n_trading_hours, n_trading_days

def convert_interval(interval, n_trading_hours, n_trading_days, n_years):
    if interval in ['1m', '5m', '30m']:
        interval = interval[:-1]
        return 60/int(interval) * n_trading_hours * n_trading_days * n_years
    elif interval == '1h':
        return n_trading_hours * n_trading_days * n_years
    elif interval == '1d':
        return n_trading_days * n_years
    elif interval == '1w':
        n_trading_weeks = 52 * n_trading_days / 365
        return n_trading_weeks * n_years
    elif interval == '1m':
        n_trading_months = 30 * n_trading_days / 365
        return n_trading_months * n_years
    else:
        raise ValueError (f'Interval {interval} not covered')
    
def check_column_index(df, column):
    if df.index.name != column:
        if column not in df.columns:
            raise ValueError(f'{column} column missing from dataframe')
        else:
            df[column] = pd.to_datetime(df[column])
            df.set_index(column, inplace=True)

def check_expected_bars(df, interval, n_trading_hours, n_trading_days):
    start_date = df.index[0]
    end_date = df.index[-1]
    time_delta = end_date - start_date
    n_years  = time_delta.days / 365.25 + time_delta.seconds / (365.25 * 24 * 3600)
    expected_n_bars = convert_interval(interval, n_trading_hours, n_trading_days, n_years)
    n_bars = len(df)
    delta_n_bars = (n_bars - expected_n_bars) / expected_n_bars
    if abs(delta_n_bars) > 0.05:
        print('Dataset is missing more than 5% of expected bars in this time frequency')
    
    return n_bars, n_years
    
def backtest_metrics(backtest_df:pd.DataFrame, interval:str, n_trading_hours:float, n_trading_days:int,
                     wealth_column='Wealth', buy_hold_column='BuyHold_Return') -> list:

    wealth_df = backtest_df[wealth_column]

    if backtest_df.index.name != 'timestamp':
        if 'timestamp' not in backtest_df.columns:
            raise ValueError('"timestamp" column missing in price_backtest_df')
        else:
            backtest_df['timestamp'] = pd.to_datetime(backtest_df['timestamp'])
            backtest_df.set_index('timestamp', inplace=True)

    start_date = backtest_df.index[0]
    end_date = backtest_df.index[-1]
    time_delta = end_date - start_date
    n_years  = time_delta.days / 365.25 + time_delta.seconds / (365.25 * 24 * 3600)
    expected_n_bars = convert_interval(interval, n_trading_hours, n_trading_days, n_years)
    n_bars = len(backtest_df)
    delta_n_bars = (n_bars - expected_n_bars) / expected_n_bars
    if abs(delta_n_bars) > 0.05:
        print('Dataset is missing more than 5% of expected bars in this time frequency')

    CAGR = (wealth_df.iloc[-1] / wealth_df.iloc[0]) ** (1/n_years) - 1
    total_perf = (wealth_df.iloc[-1] / wealth_df.iloc[0]) - 1
    avg_return = wealth_df.pct_change().dropna().mean()
    avg_ann_return = avg_return * n_bars / n_years
    volatility = wealth_df.pct_change().dropna().std()
    ann_vol = volatility * np.sqrt( n_bars / n_years )
    sharpe = avg_ann_return / ann_vol


    metrics = [round(total_perf*100, 2), round(CAGR*100, 2), round(100*avg_ann_return, 2), round(100*ann_vol, 2), round(sharpe, 2)]
    df_metrics = pd.DataFrame(metrics).T
    df_metrics.columns = ['Total Performance [%]', 'CAGR [%]', 'Avg. Return (Ann.) [%]', 'Volatility (Ann.) [%]', 'Sharpe Ratio (Ann.)']
    df_metrics.index.name = 'Portfolio Metrics'
    return df_metrics

def save_plot_strategy(backtest_df: pd.DataFrame) -> None:
    """
    Plot B&H Vs Portfolio Value / Wealth

    Input:
        - price:        Price time-series
        - wealth:       Wealth / Portfolio Value
        - model_name:   model or strategy name
        - labels:       extra data e.g. symbol, date, etc...
    """

    fig, ax = pl.subplots(1, 1, figsize=(24,14))

    #ax.set_title(f"Backtest {labels['symbol']} from {labels['date start']} to {labels['date end']} ($k={labels['thresh']}$ $n={labels['n points']}$)", fontsize=20)

    backtest_df['Wealth'].plot(ax=ax, grid=True, logy=False) #Was set to true before
    (backtest_df["Close"]/backtest_df["Close"].iloc[0]).plot(ax=ax, color="k", grid=True, logy=False)

    ax.set_ylabel("Portfolio Value", fontsize=18)
    ax.set_xlabel("Date", fontsize=18)
    ax.legend(fontsize=26)

    fig.subplots_adjust(left=0.055, bottom=0.093, right=0.962, top=0.92)
    if not os.path.exists("Backtest/Figure"):
        os.mkdir("Backtest/Figure")

    fig.savefig(f"Backtest/Figure/plot_test.png")

if __name__ == '__main__':
    # Setup logging
    logger = set_logs("test_logs")

    # Examples of logging at different levels
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")

 
