import pandas as pd
from Strategy.MeanReversion.pairs_trading import Pairs_Trading, get_ols_results, pairs_trading_signals
from Backtest.backtest import Backtest
import itertools
import os
import numpy as np

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import time




def get_random_periods(df, N, num_bars):
    max_start = len(df) - num_bars
    random_starts = np.random.choice(range(max_start), size=N, replace=False)
    return [(start, start + num_bars) for start in random_starts]

def Backtest_Pairs_Trading(sub_df, entry_zScore, exit_zScore, leverage, take_profit, stop_loss):
    print(entry_zScore)
    print(exit_zScore)
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

        # Define parameter space
    space = {
        'entry_zScore': hp.uniform('entry_zScore', 0.1, 2.5),
        'exit_zScore_factor': hp.uniform('exit_zScore_factor', 0, 0.95)
    }


    # Pre-select random periods before optimization
    num_periods = 100  # Define how many periods you want to test each parameter set against
    random_periods = get_random_periods(backtest_df, N=num_periods, num_bars=50_000)

    def objective(params):
        leverage = 1
        take_profit = 1
        stop_loss = -1
        print(params)
        entry_zscore = params['entry_zScore']
        exit_zscore = params['exit_zScore_factor'] * entry_zscore
        assert exit_zscore < entry_zscore, "exit_zScore must be less than entry_zScore"

        all_results = []
        
        for start, end in random_periods:
            sub_df = backtest_df.iloc[start:end]
            start_date = sub_df.index[0].strftime('%Y-%m-%d')
            end_date = sub_df.index[-1].strftime('%Y-%m-%d')

            print(start_date)
            print(end_date)

            perf, sharpe = Backtest_Pairs_Trading(sub_df,
                                                entry_zscore,
                                                exit_zscore,
                                                leverage,
                                                take_profit,
                                                stop_loss)
            print(perf)
            result_details = {
                'start_date': start_date,
                'end_date': end_date,
                'entry_zScore': entry_zscore,
                'exit_zScore': exit_zscore,
                'leverage': leverage,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'performance': perf,
                'sharpe_ratio': sharpe,
                'loss': -perf  # Store negative Sharpe for optimization purposes
            }
            all_results.append(result_details)

        # Instead of returning a single loss value, store all results temporarily
        return {
            'status': STATUS_OK,
            'loss': np.mean([r['loss'] for r in all_results]),  # Use mean loss for optimization
            'all_results': all_results  # Include all individual results for later processing
        }
    
    
    start_time = time.time()
    
    # Run the optimization
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials
    )

    # Gather and save all detailed results from all trials
    expanded_results = []
    for trial in trials.trials:
       trial_results = trial['result'].get('all_results', [])
       expanded_results.extend(trial_results)

    # Convert results to DataFrame and save as CSV
    results_df = pd.DataFrame(expanded_results)
    filePath = "Data/Backtest/Optimization/Pairs_trading_EUR_USD_optimization_trials_detailed_results_BayesianOpt_new.csv"
    results_df.to_csv(filePath, index=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)
