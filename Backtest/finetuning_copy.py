import pandas as pd
from Strategy.MeanReversion.pairs_trading import Pairs_Trading, get_ols_results, pairs_trading_signals
from Backtest.backtest import Backtest
import optuna
import os
import numpy as np
from functools import partial



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

def get_random_periods(df, N, num_bars):
    max_start = len(df) - num_bars
    random_starts = np.random.choice(range(max_start), size=N, replace=False)
    return [(start, start + num_bars) for start in random_starts]


train_df = df.loc[df.index<=end_train_period]
x=train_df[["GBP_USD"]]
y=train_df[["EUR_USD"]]
alpha, beta, resid = get_ols_results(x, y)


backtest_df = df.loc[df.index > end_train_period]

def Backtest_Pairs_Trading(backtest_df, entry_zScore, exit_zScore, leverage, take_profit, stop_loss):
    results = []
    pairs_trading = pairs_trading_signals(backtest_df, "EUR_USD", "GBP_USD", alpha, beta, resid, entry_zScore, exit_zScore)
    pairs_trading.rename(columns={"Portfolio": "Close"}, inplace=True)
    print(pairs_trading.head())
    backtesting = Backtest(signal_df=pairs_trading, symbol='EUR_USD-GBP_USD', market='forex',
                           interval='1m', leverage=leverage, fees=0.00007, slippage=0.01/100,
                           take_profit=take_profit, stop_loss=stop_loss)
    perf = backtesting.backtest_metrics(return_metric="total_perf")
    sharpe = backtesting.backtest_metrics(return_metric="sharpe")
    results.append((perf, sharpe))
    return results

def objective(trial, backtest_df):
    # Define hyperparameters within the trial
    exit_threshold = trial.suggest_float("exit_threshold", 0.25, 1.5)
    entry_threshold = trial.suggest_float("entry_threshold", exit_threshold + 0.01, 2)
    stop_loss = trial.suggest_float("stop_loss", -0.5, -0.01)
    take_profit = trial.suggest_float("take_profit", 0.01, 0.5)
    leverage = trial.suggest_float("leverage", 0.1, 10)

    # Execute the strategy using the passed dataframe
    results = Backtest_Pairs_Trading(backtest_df, entry_threshold, exit_threshold, leverage, take_profit, stop_loss)
    perf, sharpe = results[0]  # Assuming you run one period at a time
    trial.set_user_attr("total_performance", perf)
    trial.set_user_attr("sharpe_ratio", sharpe)
    return -perf

N = 10
all_results = []

for i in range(N):
    random_periods = get_random_periods(backtest_df, N=1, num_bars=1000)
    start, end = random_periods[0]
    sub_df = backtest_df.iloc[start:end]
    print(sub_df.head())
    # Use a partial function to pass the sub_df to the objective function
    objective_with_data = partial(objective, backtest_df=sub_df)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_with_data, n_trials=2)  # Optimize with the sub_df

    # Collect and store results for analysis
    for trial in study.trials:
        trial_results = {
            "trial_number": trial.number,
            # Collect and map other attributes as before...
        }
        all_results.append(trial_results)

# Convert results to DataFrame
results_df = pd.DataFrame(all_results)
print(results_df.head())  # Display first few rows of the results

file_path = 'Data/Backtest/Optimization/Pairs_trading_EUR_USD_optimization_trials_results.csv'

# Check if the file already exists
file_exists = os.path.exists(file_path)

# Append to the CSV file
results_df.to_csv(file_path, mode='a', header=not file_exists, index=False)


# Best parameters found
#print("Best parameters:", study.best_params)
#print("Best value (negative return):", study.best_value)

# Use these parameters to set up your trading strategy
#best_params = study.best_params