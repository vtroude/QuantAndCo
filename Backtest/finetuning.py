import pandas as pd
from Strategy.MeanReversion.pairs_trading import Pairs_Trading
from Backtest.backtest import Backtest


y = pd.read_csv("Data/forex/EUR_USD/1m/OHLC/1563535876_1713535876.csv")
x = pd.read_csv("Data/forex/GBP_USD/1m/OHLC/1563535876_1713535876.csv")
y["timestamp"] = pd.to_datetime(y["close_time"])
y.set_index("timestamp", inplace=True)
y = y[["Close"]].rename(columns={"Close": "EUR_USD"})
x["timestamp"] = pd.to_datetime(x["close_time"])
x.set_index("timestamp", inplace=True)
x = x[["Close"]].rename(columns={"Close": "GBP_USD"})
df = pd.merge(x, y, right_index=True, left_index=True, how="inner")

backtest_end_date = '2023-01-01'

df = df.loc[df.index<=backtest_end_date]

end_train_period = '2022-01-01'

pairs_trading = Pairs_Trading(df, "EUR_USD", "GBP_USD", entry_zScore=0.5, exit_zScore=0.25, end_train_period="2022-01-01",
                                window=10_000)

backtesting = Backtest(signal_df=pairs_trading, symbol='EUR_USD-GBP_USD', market='forex',
                           interval='1m', start_date=end_train_period, end_date=backtest_end_date, leverage=3,
                           fees=0.00007, slippage=0.01/100, 
                           take_profit=100, stop_loss=-100)

perf = backtesting.backtest_metrics(return_metric="total_perf")

print(perf)