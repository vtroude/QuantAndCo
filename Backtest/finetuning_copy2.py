import numpy as np
import pandas as pd
from Backtest.backtest import Backtest
from Strategy.MeanReversion.pairs_trading import Pairs_Trading, get_ols_results, pairs_trading_signals, dynamicOLS, dynamicOLS_PairsTrading
import time

df = pd.read_csv('Data/Backtest/Combined_EURUSD_GBPUSD.csv')
df.set_index("timestamp", inplace=True)
df.index = pd.to_datetime(df.index)

df = df.loc[df.index<='2020-09-01']

pairs_trading = dynamicOLS_PairsTrading(df, asset_x="GBP_USD", asset_y='EUR_USD', window=100)
pairs_trading["signal"] = (pairs_trading["z_score"] <= -1.5) *1 + (pairs_trading["z_score"] >= 1.5) * -1
bt = Backtest(signal_df=pairs_trading, symbol='EUR_USD-GBP_USD', market='forex',
                        interval='1m', leverage=3, initial_wealth=10_000,
                        fees=0.00007)

#vec_bt = bt.Vectorized_BT_PairsTrading(asset_y="EUR_USD", asset_x="GBP_USD")
#bt_df = bt.Backtest_PairsTrading_loop(asset_x='GBP_USD', asset_y='EUR_USD')
#bt.backtest_metrics(backtest_df=bt_df)
market = 'forex'
symbol = 'EUR_USD-GBP_USD'
interval='1m'
first_date, last_date = pairs_trading.index[0], pairs_trading.index[-1]
bt_df = pd.read_csv(f'Data/Backtest/{market}-{symbol}-{interval}-{first_date}-{last_date}-Backtest_DF.csv')
bt_df['position_change'] = bt_df.position.diff()
bt_df.set_index('timestamp', inplace=True)
bt_df.index = pd.to_datetime(bt_df.index)
metrics= bt.backtest_metrics(backtest_df=bt_df)
print(metrics)
#print(bt_df.head())

#bt_df.to_csv('Data/Backtest/Optimization/Pairs_trading_EUR_USD_Optimal_Backtest.csv')


# def get_random_periods(df, N, num_bars):
#     max_start = len(df) - num_bars
#     random_starts = np.random.choice(range(max_start), size=N, replace=False)
#     return [(start, start + num_bars) for start in random_starts]

# # Pre-select random periods before optimization
# num_periods = 100 # Define how many periods you want to test each parameter set against
# random_periods = get_random_periods(df, N=num_periods, num_bars=50_000)

# long_thres = [-0.5, -1, 1.5]
# short_thres = [0.5, 1, 1.5]
# combination = 0
# all_results = []
# windows = [100, 200, 500, 1000, 1500]
# nb_combinations = len(long_thres) * len(short_thres) * len(windows)
# total_rows = nb_combinations * num_periods

# for window in windows:
#     print(f'Generating dynamic OLS regression with {window} window...')
#     pairs_trading = dynamicOLS_PairsTrading(df, asset_x="GBP_USD", asset_y='EUR_USD', window=window)
#     for long_ in long_thres:
#         for short_ in short_thres:
#             combination+=1
#             for start, end in random_periods:
#                 sub_df = pairs_trading.iloc[start:end]
#                 start_date = sub_df.index[0].strftime('%Y-%m-%d')
#                 end_date = sub_df.index[-1].strftime('%Y-%m-%d')
#                 sub_df["signal"] = (pairs_trading["z_score"] <= long_) * 1 + (pairs_trading["z_score"] >= short_) * -1
#                 bt = Backtest(signal_df=sub_df, symbol='EUR_USD-GBP_USD', market='forex',
#                               interval='1m', leverage=1, initial_wealth=10_000,
#                               fees=0.00007)
#                 bt_df = bt.Vectorized_BT_PairsTrading(asset_y="EUR_USD", asset_x="GBP_USD")
#                 perf = bt.calculate_perf(bt_df)
#                 result_details = {
#                     'Combination #': combination,
#                     'start_date': start_date,
#                     'end_date': end_date,
#                     'window': window,
#                     'long_thres': long_,
#                     'short_thres': short_,
#                     'leverage': 1,
#                     'performance': perf
#                 }
#                 all_results.append(result_details)
#             print(f'{combination}/{nb_combinations} combinations completed...')

# results_df = pd.DataFrame(all_results)
# filePath = "Data/Backtest/Optimization/Pairs_trading_EUR_USD_optimization_20222024_shortWindows.csv"
# results_df.to_csv(filePath, index=False)