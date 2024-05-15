import numpy as np
import pandas as pd
from Backtest.backtest import Backtest
from Strategy.MeanReversion.pairs_trading import Pairs_Trading, get_ols_results, pairs_trading_signals, dynamicOLS, dynamicOLS_PairsTrading, Spread_PairsTrading
import time

df = pd.read_csv('Data/Backtest/Combined_EURUSD_GBPUSD.csv')
df.set_index("timestamp", inplace=True)
df.index = pd.to_datetime(df.index)

#pairs_trading = dynamicOLS_PairsTrading(df, asset_x="GBP_USD", asset_y='EUR_USD', window=50_000)
#for window in [50_000, 100_000, 200_000, 500_000, 1_000_000]:
#    pairs_trading = dynamicOLS_PairsTrading(df, asset_x="GBP_USD", asset_y='EUR_USD', window=window)
#    pairs_trading.to_csv(f'Data/PairsTrading_EURUSD_GBPUSD_window-{window}.csv')
#    print(f'Successfully ran pairs trading dynamic OLS regression for window = {window}')


#pairs_trading = pd.read_csv('Data/PairsTrading_EURUSD_GBPUSD_window-500000.csv')
#pairs_trading = dynamicOLS_PairsTrading(df, asset_x="GBP_USD", asset_y='EUR_USD', window=5000)
#pairs_trading['timestamp'] = pd.to_datetime(pairs_trading['timestamp'])
#pairs_trading.set_index('timestamp', inplace=True)

pairs_trading = Spread_PairsTrading(df, asset_x='GBP_USD', asset_y="EUR_USD", window=33_000)

pairs_trading = pairs_trading.loc[(pairs_trading.index >= '2023-02-10') & (pairs_trading.index <= '2023-03-16')]

pairs_trading["signal"] = (pairs_trading["spread"] <= pairs_trading["spread_moving_avg"] - 0.5 * pairs_trading["spread_moving_std"]) * 1 + (pairs_trading["spread"] >= pairs_trading["spread_moving_avg"] + 0.5 * pairs_trading["spread_moving_std"]) * -1

#pairs_trading["signal"] = (pairs_trading["z_score"] <= -1.5) * 1 + (pairs_trading["z_score"] >= 1.5) * -1
bt = Backtest(signal_df=pairs_trading, symbol='EUR_USD-GBP_USD', market='forex',
                        interval='1m', leverage=1, initial_wealth=10_000,
                        fees=0.00007)

vec_bt = bt.backtest_pairs_trading(asset_x="GBP_USD", 
                                asset_y="EUR_USD")

vec_bt.to_csv('Data/Backtest-test.csv')

metrics = bt.backtest_metrics(vec_bt)
print(metrics)

#print(vec_bt.head())
#print(vec_bt.tail())
