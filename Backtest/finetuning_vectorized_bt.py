import numpy as np
import pandas as pd
from Backtest.backtest import Backtest
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from Strategy.MeanReversion.pairs_trading import Pairs_Trading, get_ols_results, pairs_trading_signals
import time



#1. Estimate hedge ratio on train data
#2. Generate random dates on test data
#3. For each random date, backtest the strategy on the space of hyperparameters


df = pd.read_csv('Data/Backtest/Combined_EURUSD_GBPUSD.csv')
df.set_index("timestamp", inplace=True)
df.index = pd.to_datetime(df.index)
print(df.head())


end_train_period = '2022-06-01'

train_df = df.loc[df.index<=end_train_period]
x=train_df[["GBP_USD"]]
y=train_df[["EUR_USD"]]
alpha, beta, resid = get_ols_results(x, y)


leverage = 1

test_df = df.loc[df.index > end_train_period]


def get_random_periods(df, N, num_bars):
    max_start = len(df) - num_bars
    random_starts = np.random.choice(range(max_start), size=N, replace=False)
    return [(start, start + num_bars) for start in random_starts]


# Pre-select random periods before optimization
num_periods = 100  # Define how many periods you want to test each parameter set against
random_periods = get_random_periods(test_df, N=num_periods, num_bars=50_000)

import hashlib

# Dictionary to store parameter combinations and their respective indices
param_indices = {}
# Counter to keep track of the next available index
current_index = 0


def objective(params):
    global current_index
    long_thres, short_thres = params['long_thres'], params['short_thres']
    param_tuple = (params['long_thres'], params['short_thres'])

    # Check if this combination of parameters has already been seen
    if param_tuple not in param_indices:
        # Assign a new index to this combination
        param_indices[param_tuple] = current_index
        current_index += 1  # Increment the counter for the next new combination

    # Get the assigned index for the current parameter set
    param_index = param_indices[param_tuple]

    all_results = []

    for start, end in random_periods:
        sub_df = test_df.iloc[start:end]
        start_date = sub_df.index[0].strftime('%Y-%m-%d')
        end_date = sub_df.index[-1].strftime('%Y-%m-%d')
        pairs_trading = pairs_trading_signals(sub_df, "EUR_USD", "GBP_USD", 
                                              alpha, beta, resid, long_thres, 
                                              short_thres)
        


        bt = Backtest(signal_df=pairs_trading, symbol='EUR_USD-GBP_USD', market='forex',
                                interval='1m', leverage=leverage, initial_wealth=10_000,
                                fees=0.00007)

        bt_df = bt.Vectorized_BT_PairsTrading("EUR_USD", "GBP_USD")
        perf = bt.calculate_perf(bt_df)


        result_details = {
            'Combination #': param_index,
            'start_date': start_date,
            'end_date': end_date,
            'long_thres': long_thres,
            'short_thres': short_thres,
            'leverage': leverage,
            'performance': perf,
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


# Define parameter space
space = {
    'long_thres': hp.uniform('long_thres', -2.5, -0.01),
    'short_thres': hp.uniform('short_thres', 0.01, 2.5)
}

# Run the optimization
trials = Trials()
best = fmin(
fn=objective,
space=space,
algo=tpe.suggest,
max_evals=100,
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