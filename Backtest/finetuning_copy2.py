import pandas as pd
from Strategy.MeanReversion.pairs_trading import Pairs_Trading, get_ols_results, pairs_trading_signals
from Backtest.backtest import Backtest
import optuna
import os
import numpy as np
from functools import partial
import itertools


def get_random_periods(df, N, num_bars):
    max_start = len(df) - num_bars
    random_starts = np.random.choice(range(max_start), size=N, replace=False)
    return [(start, start + num_bars) for start in random_starts]

def Backtest_Pairs_Trading(sub_df, entry_zScore, exit_zScore, leverage, take_profit, stop_loss):
    pairs_trading = pairs_trading_signals(sub_df, "EUR_USD", "GBP_USD", alpha, beta, resid, entry_zScore, exit_zScore)
    backtesting = Backtest(signal_df=pairs_trading, symbol='EUR_USD-GBP_USD', market='forex',
                           interval='1m', leverage=leverage, fees=0.00007, slippage=0.01/100,
                           take_profit=take_profit, stop_loss=stop_loss)
    perf = backtesting.backtest_metrics(return_metric="total_perf")
    sharpe = backtesting.backtest_metrics(return_metric="sharpe")
    return perf, sharpe

if __name__ == "__main__":

    y = pd.read_csv("Data/forex/EUR_USD/1m/OHLC/1563535876_1713535876.csv")
    x = pd.read_csv("Data/forex/GBP_USD/1m/OHLC/1563535876_1713535876.csv")
    y["timestamp"] = pd.to_datetime(y["close_time"])
    y.set_index("timestamp", inplace=True)
    y = y[["Close"]].rename(columns={"Close": "EUR_USD"})
    x["timestamp"] = pd.to_datetime(x["close_time"])
    x.set_index("timestamp", inplace=True)
    x = x[["Close"]].rename(columns={"Close": "GBP_USD"})
    df = pd.merge(x, y, right_index=True, left_index=True, how="inner")

    end_train_period = '2022-06-01'

    train_df = df.loc[df.index<=end_train_period]
    x=train_df[["GBP_USD"]]
    y=train_df[["EUR_USD"]]
    alpha, beta, resid = get_ols_results(x, y)


    backtest_df = df.loc[df.index > end_train_period]

    param_grid = {
        'entry_zScore': np.linspace(0.1, 2, num=50),
        'exit_zScore': np.linspace(0, 1.9, num=50),
        'leverage': np.linspace(0.1, 10, num=50),
        'take_profit': np.linspace(0.01, 1, num=50),
        'stop_loss': np.linspace(-1, -0.01, num=50)
    }
    # Initialize DataFrame to store results
    columns = ['Start Date', 'End Date', 'Combination #', 'Entry ZScore', 'Exit ZScore', 'Leverage', 'Take Profit', 'Stop Loss', 'Performance', 'Sharpe Ratio']
    results_df = pd.DataFrame(columns=columns)

    i = 0

    n_dates = 1000

    filePath = "Data/Backtest/Optimization/Pairs_trading_EUR_USD_optimization_trials_results.csv"

    for i in range(n_dates):

        random_periods = get_random_periods(backtest_df, N=1, num_bars=50_000)
        start, end = random_periods[0]
        sub_df = backtest_df.iloc[start:end]
        start_date = sub_df.index[0].strftime('%Y-%m-%d')
        end_date = sub_df.index[-1].strftime('%Y-%m-%d')
        combination_count = 0

        for params in itertools.product(*param_grid.values()):
            entry_zScore, exit_zScore, leverage, take_profit, stop_loss = params

        # Skip the iteration if exit_zScore is not less than entry_zScore
            if exit_zScore >= entry_zScore:
                continue

            perf, sharpe = Backtest_Pairs_Trading(sub_df, *params)
            combination_count += 1
            current_row = pd.DataFrame([start_date, end_date, combination_count, *params, perf, sharpe], index=columns).transpose()
            results_df = pd.concat([results_df, current_row], ignore_index=True)

            file_exists = os.path.exists(filePath)

            # Append to the CSV file
            results_df.to_csv(filePath, mode='a', header=not file_exists, index=False)
                    


    print(results_df)