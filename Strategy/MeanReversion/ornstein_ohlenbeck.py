import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Backtest.backtest import Backtest

def ornstein_uhlenbeck_process(mu, theta, sigma, X0, dt, T):
    """
    Simulate an Ornstein-Uhlenbeck process.

    Parameters:
    mu (float): Long-term mean
    theta (float): Rate of reversion to the mean
    sigma (float): Volatility parameter
    X0 (float): Initial value
    dt (float): Time step
    T (float): Total time

    Returns:
    np.ndarray: Simulated values of the process
    """
    np.random.seed(901)

    N = int(T / dt)
    t = np.linspace(0, T, N)
    X = np.zeros(N)
    X[0] = X0
    for i in range(1, N):
        dW = np.sqrt(dt) * np.random.normal()
        X[i] = X[i-1] + theta * (mu - X[i-1]) * dt + sigma * dW
    
    return t, X

def mean_reverting_strategy(df, long_thres, short_thres, price_column="Close"):
    df["signal"] = (df[price_column] <= long_thres) * 1 + (df[price_column] >= short_thres) * -1

    # Remove consecutive identical signals
    df["filtered_signal"] = df["signal"].ne(df["signal"].shift()).astype(int) * df["signal"]
    
    # Plotting the price data
    plt.figure(figsize=(12, 6))
    plt.plot(df[price_column], label='Price')
    
    # Plot buy signals
    buy_signals = df[df["filtered_signal"] == 1]
    plt.plot(buy_signals.index, buy_signals[price_column], '^', markersize=10, color='g', lw=0, label='Buy Signal')
    
    # Plot sell signals
    sell_signals = df[df["filtered_signal"] == -1]
    plt.plot(sell_signals.index, sell_signals[price_column], 'v', markersize=10, color='r', lw=0, label='Sell Signal')
    
    # Add labels and legend
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Mean Reverting Strategy Signals')
    plt.legend()
    plt.savefig('mean_reverting_signals.png')
    return df

def simulate_data(start_date, n_dates, data, frequency='D'):
    # Generate the date range
    date_range = pd.date_range(start=start_date, periods=n_dates, freq=frequency)
    # Create an empty DataFrame with the generated date range as the index
    df = pd.DataFrame(index=date_range, data=data)
    df.columns=["Close"]
    return df

if __name__ == "__main__":


    # Parameters
    mu = 1.1      # Long-term mean
    theta = 0.01  # Rate of reversion
    sigma = 0.02    # Volatility
    X0 = 1.08       # Initial value
    T = 252       # Total time
    dt = 1
    N = int(T / dt)

    long_thres = mu - 7 * sigma
    short_thres = mu + 7 * sigma

    # Simulate the process
    t, X = ornstein_uhlenbeck_process(mu, theta, sigma, X0, dt, T)

    df = simulate_data(start_date='2023-01-01', n_dates=N, data=X, frequency='D')
    df = mean_reverting_strategy(df, long_thres, short_thres)

    bt = Backtest(df, 'EURUSD', 'forex', '1d', initial_wealth=10_000,
                  leverage=1, fees=0.00007)
    
    bt_df = bt.vectorized_backtesting()
    metrics = bt.backtest_metrics(bt_df)
    print(metrics)
    bt.plot_strategy(bt_df, long_thres=long_thres, short_thres=short_thres)