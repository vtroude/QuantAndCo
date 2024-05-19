import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from Backtest.backtest import Backtest

def calculate_periodic_returns(df, start_time, end_time):
    df = df.copy()

    start_prices = df.at_time(start_time)
    end_prices = df.at_time(end_time)

    start_prices['date'] = start_prices.index.normalize().date
    end_prices['date'] = end_prices.index.normalize().date

    start_prices.set_index('date', inplace=True)
    end_prices.set_index('date', inplace=True)


    comb_df = pd.merge(start_prices, end_prices, left_index=True, right_index=True, suffixes=('_start', '_end'))

    return comb_df


def plot_histogram(df, start_time, end_time, column="Close"):

    returns = df[f"{column}_end"] / df[f"{column}_start"] - 1

    start_date, end_date = df.index[0], df.index[-1]

    mean_returns = returns.mean()
    std_returns = returns.std()

    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=50, alpha=0.75, color='blue')
    plt.title(f'Histogram of Intraday Returns between {start_time} and {end_time} \n {start_date} to {end_date}')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    
    # Standard deviation lines
    for i in range(1, 4):
        plt.axvline(x=mean_returns + i*std_returns, color='r', linestyle='dashed', linewidth=1, label=f'Mean + {i} STD')
        plt.axvline(x=mean_returns - i*std_returns, color='g', linestyle='dashed', linewidth=1, label=f'Mean - {i} STD')

    plt.grid(True)
    plt.legend()
    plt.show()

    return mean_returns, std_returns


def MeanReversionIntraday(df, start_time, end_time, 
                          last_train_date, price_column, long_thres, short_thres):
    train_df = df.loc[df.index<=last_train_date]
    comb_train = calculate_periodic_returns(train_df, start_time, end_time)
    mu, sigma = plot_histogram(comb_train, start_time, end_time, price_column)
    test_df_init = df.loc[df.index > last_train_date]

    comb_test = calculate_periodic_returns(test_df_init, start_time, end_time)

    end_test_time = datetime.strptime(end_time, '%H:%M') + timedelta(minutes=1)
    str_test_time = end_test_time.strftime('%H:%M')

    test_df = test_df_init.between_time(str_test_time, '23:59')

    test_df['date'] = test_df.index.normalize().date
    test_df.date = pd.to_datetime(test_df.date)
    comb_test[f"{start_time[:2]}-{end_time[:2]}_return"] = (comb_test[f"{price_column}_end"] - comb_test[f"{price_column}_start"]) / comb_test[f"{price_column}_start"]
    test_returns = comb_test[[f"{start_time[:2]}-{end_time[:2]}_return"]]
    test_returns.index = pd.to_datetime(test_returns.index)

    test_df.reset_index(inplace=True)
    test_df.set_index('date', inplace=True)
    test_df.index = pd.to_datetime(test_df.index)


    test_df = test_df[["close_time", "Close"]]

    test_df = pd.merge(test_df, test_returns, right_index=True, left_index=True, how='left')

    test_df["signal"] = (test_df[f"{start_time[:2]}-{end_time[:2]}_return"] <= mu + long_thres * sigma) * 1 + (test_df[f"{start_time[:2]}-{end_time[:2]}_return"] >= mu + short_thres * sigma) * -1

    test_df = pd.merge(test_df_init, test_df.set_index('close_time').drop(columns="Close"), right_index=True, left_index=True, how='left')
    test_df["signal"].ffill(inplace=True)
    test_df[f"{start_time[:2]}-{end_time[:2]}_return"].ffill(inplace=True)

    test_df["timestamp"] = pd.to_datetime(test_df.index)
    previous_hour = str(int(end_time[:2]) - 1) + ':59'
    filtered_df = test_df[(test_df["timestamp"].dt.time >= pd.to_datetime('00:00').time()) & (test_df["timestamp"].dt.time <= pd.to_datetime(end_time).time())]

    test_df.loc[filtered_df.index, "signal"] = 0 #always exit the position at the end of the day and can only re-start after end_time
    
    return test_df


if __name__ == "__main__":
    df = pd.read_csv('/root/QuantAndCo/Data/forex/EUR_USD/1h/OHLC/2019-07-19 13:31:16_2024-04-19 16:11:16.csv')
    df.close_time = pd.to_datetime(df.close_time)
    df.set_index('close_time', inplace=True)
    signal_df = MeanReversionIntraday(df, start_time='08:00', end_time='14:00', 
                                  last_train_date='2020-06-01', price_column="Close", 
                                  long_thres=-1.0, short_thres=1.0)
    bt = Backtest(signal_df, 'EURUSD', 'forex', '1h', initial_wealth=10_000,
                  leverage=10, fees=0.00007)
    
    bt_df = bt.vectorized_backtesting()
    metrics = bt.backtest_metrics(bt_df)
    print(metrics)
    