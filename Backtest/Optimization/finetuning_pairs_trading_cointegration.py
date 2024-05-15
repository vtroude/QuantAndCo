import numpy as np
import pandas as pd
from Backtest.backtest import Backtest
from Strategy.MeanReversion.pairs_trading import dynamicOLS_PairsTrading, Spread_PairsTrading

#The below code does the following:
#1. It downloads the combined data of EUR_USD and GBP_USD. If not already generated, please generate it.
#2. It backtests the strategy for different parameters (OLS window, long_thres and short_thres) on different random periods of 33'000 bars ( = approx 1 month of data for minute bars)
#3. All the results are saved in a table. We can then plot the distributions of the different 1-month performances 
#4. See the Optimization/finetuning_analysis.ipynb file an example on how to plot the results obtained here.

df = pd.read_csv('Data/Backtest/Combined_EURUSD_GBPUSD.csv')
df.set_index("timestamp", inplace=True)
df.index = pd.to_datetime(df.index)


def get_random_periods(df, N, num_bars):
    max_start = len(df) - num_bars
    random_starts = np.random.choice(range(max_start), size=N, replace=False)
    return [(start, start + num_bars) for start in random_starts]

# Pre-select random periods before optimization
num_periods = 10 # Define how many periods you want to test each parameter set against
random_periods = get_random_periods(df, N=num_periods, num_bars=33_000)

long_thres = [-2, -1.5, -1, -0.5, 0]
short_thres = [0, 0.5, 1, 1.5, 2]
#windows = [10_000, 50_000, 100_000, 200_000, 500_000] 
windows = [100, 500, 1000]
nb_combinations = len(long_thres) * len(short_thres) * len(windows)
total_rows = nb_combinations * num_periods

combination = 0
all_results = []

pd.options.mode.chained_assignment = None  # Suppress SettingWithCopyWarning

for window in windows:
    print(f'Generating dynamic OLS regression with {window} window...')
    #For long windows, the below code will take a lot of time to run - think about saving the file generated if re-used.

    pairs_trading = dynamicOLS_PairsTrading(df, asset_x="GBP_USD", asset_y='EUR_USD', window=window)
    #pairs_trading = Spread_PairsTrading(df, asset_x='GBP_USD', asset_y="EUR_USD", window=window)
    #pairs_trading = pd.read_csv(f'Data/PairsTrading_EURUSD_GBPUSD_window-{window}.csv')
    #pairs_trading["timestamp"] = pd.to_datetime(pairs_trading["timestamp"])
    #pairs_trading.set_index('timestamp', inplace=True)
    for long_ in long_thres:
        for short_ in short_thres:
            combination+=1
            for start, end in random_periods:
                sub_df = pairs_trading.iloc[start:end]
                start_date = sub_df.index[0].strftime('%Y-%m-%d')
                end_date = sub_df.index[-1].strftime('%Y-%m-%d')
                sub_df["signal"] = (pairs_trading["z_score"] <= long_) * 1 + (pairs_trading["z_score"] >= short_) * -1
                #sub_df["signal"] = (pairs_trading["spread"] <= pairs_trading["spread_moving_avg"] + long_ * pairs_trading["spread_moving_std"]) * 1 + (pairs_trading["spread"] >= pairs_trading["spread_moving_avg"] + short_ * pairs_trading["spread_moving_std"]) * -1
                try:
                    #because of dropna in backtest function, some early dates can't be processed
                    bt = Backtest(signal_df=sub_df, symbol='EUR_USD-GBP_USD', market='forex',
                                            interval='1m', leverage=1, initial_wealth=10_000,
                                            fees=0.00007)

                    vec_bt = bt.vectorized_backtest_pairs_trading(asset_x="GBP_USD", 
                                                    asset_y="EUR_USD").dropna()
                    perf = bt.calculate_perf(vec_bt['Wealth'])
                except Exception as e:
                    print(f'Error: {e}')
                    print(start_date)
                    print(end_date)
                    print(len(vec_bt))
                    continue
                result_details = {
                    'Combination #': combination,
                    'start_date': start_date,
                    'end_date': end_date,
                    'window': window,
                    'long_thres': long_,
                    'short_thres': short_,
                    'leverage': 1,
                    'performance': perf
                }
                all_results.append(result_details)
            print(f'{combination}/{nb_combinations} combinations completed...')

results_df = pd.DataFrame(all_results)
filePath = "Data/Backtest/Optimization/Pairs_trading_EUR_USD_optimization_20222024.csv"
results_df.to_csv(filePath, index=False)