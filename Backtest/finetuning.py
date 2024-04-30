import pandas as pd
from Strategy.MeanReversion.pairs_trading import Pairs_Trading
from Backtest.backtest import Backtest
import optuna
import os
import numpy as np


y = pd.read_csv("Data/forex/EUR_USD/1m/OHLC/1563535876_1713535876.csv")
x = pd.read_csv("Data/forex/GBP_USD/1m/OHLC/1563535876_1713535876.csv")
y["timestamp"] = pd.to_datetime(y["close_time"])
y.set_index("timestamp", inplace=True)
y = y[["Close"]].rename(columns={"Close": "EUR_USD"})
x["timestamp"] = pd.to_datetime(x["close_time"])
x.set_index("timestamp", inplace=True)
x = x[["Close"]].rename(columns={"Close": "GBP_USD"})
df = pd.merge(x, y, right_index=True, left_index=True, how="inner")

def get_random_periods(df, N, num_bars):
    max_start = len(df) - num_bars
    random_starts = np.random.choice(range(max_start), size=N, replace=False)
    return [(start, start + num_bars) for start in random_starts]

print('random dates:')

random_periods = get_random_periods(df, N=10, num_bars=720)
for start, end in random_periods:
    sub_df = df.iloc[start:end]
    print(sub_df.index[0].strftime('%Y-%m-%d'))

end_train_period = '2023-01-01'
end_backtest = "2023-03-20"

df = df.loc[df.index<=end_backtest]

def Backtest_Pairs_Trading(entry_zScore, exit_zScore, leverage, take_profit, stop_loss):
    pairs_trading = Pairs_Trading(df, "EUR_USD", "GBP_USD", entry_zScore=entry_zScore, 
                                  exit_zScore=exit_zScore, end_train_period=end_train_period,
                                dynamic_regression=False)
    
    pairs_trading.rename(columns={"Portfolio": "Close"}, inplace=True)

    backtesting = Backtest(signal_df=pairs_trading, symbol='EUR_USD-GBP_USD', market='forex',
                           interval='1m', start_date=end_train_period, leverage=leverage,
                           fees=0.00007, slippage=0.01/100, 
                           take_profit=take_profit, stop_loss=stop_loss)

    perf = backtesting.backtest_metrics(return_metric="total_perf")
    sharpe = backtesting.backtest_metrics(return_metric="sharpe")

    return perf, sharpe

def objective(trial):
    # Defining the hyperparameters' range
    exit_threshold = trial.suggest_float("exit_threshold", 0.25, 1.5) 
    entry_threshold = trial.suggest_float("entry_threshold", exit_threshold+0.01, 2)  # Entry threshold is always larger than exit_threshold
    stop_loss = trial.suggest_float("stop_loss", -0.5, -0.01)  # Stop-loss between 0.01 and 0.1
    take_profit = trial.suggest_float("take_profit", 0.01, 0.5)
    leverage = trial.suggest_float("leverage", 0.1, 10)

    print(f"Entry_threshold: {entry_threshold}")
    print(f"Exit_threshold: {exit_threshold}")
    print(f"Stop loss: {stop_loss}")
    print(f'Take Profit: {take_profit}')
    print(f"Leverage: {leverage}")

    # Execute the strategy
    perf, sharpe = Backtest_Pairs_Trading(entry_threshold, exit_threshold, leverage, take_profit, stop_loss)

    # Log trial details and result
    trial.set_user_attr("entry_threshold", entry_threshold)
    trial.set_user_attr("exit_threshold", exit_threshold)
    trial.set_user_attr("stop_loss", stop_loss)
    trial.set_user_attr("take_profit", take_profit)
    trial.set_user_attr("leverage", leverage)
    trial.set_user_attr("total_performance", perf)
    trial.set_user_attr("sharpe_ratio", sharpe)

    return -perf

# Create a study object that will find the best hyperparameters
study = optuna.create_study(direction="minimize")  # We minimize -return to maximize return
study.optimize(objective, n_trials=2)  # You can increase n_trials for more thorough search

start_date, end_date = end_train_period, end_backtest

results = []
for trial in study.trials:
    trial_results = {
        "trial_number": trial.number,
        "total_performance": -trial.value if trial.value is not None else None,  # Converting back to positive
        "sharpe_ratio": trial.user_attrs.get("sharpe_ratio", None),
        "entry_threshold": trial.user_attrs["entry_threshold"],
        "exit_threshold": trial.user_attrs["exit_threshold"],
        "stop_loss": trial.user_attrs["stop_loss"],
        "take_profit": trial.user_attrs["take_profit"],
        "leverage": trial.user_attrs["leverage"],
        "start_date": start_date,
        "end_date": end_date
    }
    results.append(trial_results)

results_df = pd.DataFrame(results)
print(results_df.head())  # Display first few rows of the results

file_path = 'Data/Backtest/Optimization/Pairs_trading_EUR_USD_optimization_trials_results.csv'

# Check if the file already exists
file_exists = os.path.exists(file_path)

# Append to the CSV file
results_df.to_csv(file_path, mode='a', header=not file_exists, index=False)


# Best parameters found
print("Best parameters:", study.best_params)
print("Best value (negative return):", study.best_value)

# Use these parameters to set up your trading strategy
best_params = study.best_params