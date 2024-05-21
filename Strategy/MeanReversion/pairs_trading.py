import numpy    as np
import pandas   as pd

from typing import List

from pyfinance.ols  import PandasRollingOLS

from statsmodels.tsa.stattools  import adfuller, kpss

from Research.statistical_tests import OLS_reg, get_half_life # , Check_Mean_Reversion, adf_test

def PairsTrading(x, y):

    print("Step 1: Performing OLS regression of asset_y on asset_x")
    ols = OLS_reg(x, y)
    results = ols.fit()
    intercept, hedge_ratio = results.params
    #1. Check if hedge_ratio is statistically significantly diff. than 0
    print("="*50)
    y_hat = ols.predict(results.params)
    #2. Perform ADF on residuals
    resid = y - y_hat
    print("Step 2: Checking if the two assets are co-integrated")
    #print("Performing stationarity tests...")
    #print("ADF test:")
    #adf = adf_test(resid)
    half_life = get_half_life(resid)
    print(f'Half life: {half_life}')
    if hedge_ratio < 0:
        sign = '-'
    else:
        sign = '+'
    print(f'Y = {intercept: .2f} {sign} {hedge_ratio: .2f} * X')
    return resid


def get_ols_results(x, y):
    ols = OLS_reg(x, y)
    results = ols.fit()
    alpha, beta = results.params
    y_hat = ols.predict(results.params)
    resid = y.values.reshape(-1,) - y_hat

    return alpha, beta, resid

def pairs_trading_signals(df, asset_y, asset_x, alpha, beta, resids, long_thres, short_thres):

    """
    Hedge ratio and residuals are estimated on a train data. Then, signals are generated on a subsequent test data.
    Later on, we will want to dynamically re-estimate the hedge ratio.
    """

    df['hedge_ratio'] = beta

    df["y_hat"] = alpha + beta * df[asset_x]
    resid_vali = df[asset_y] - df["y_hat"]
    resid_mean, resid_std = np.mean(resids), np.std(resids)
    resid_zscore = (resid_vali - resid_mean) / resid_std
    df["z_score"] = resid_zscore
    df["signal"] = (resid_zscore <= long_thres) *1 + (resid_zscore >= short_thres) * -1

    return df

def Spread_PairsTrading(df, asset_x, asset_y, window):
    df["spread"] = df[asset_y] - df[asset_x]
    df["spread_moving_avg"] = df["spread"].rolling(window=window, closed='left').mean() #closed='left' excludes the last point in the window - that way, we are avoiding look ahead bias.
    df["spread_moving_std"] = df["spread"].rolling(window=window, closed='left').std()
    return df
    

def dynamicOLS_PairsTrading(df, asset_x, asset_y, window):

    y = df[asset_y]
    x = df[asset_x]
    model = PandasRollingOLS(y=y, x=x, window=window)
    #In the below merge, we are shifting the regression results by 1 period, such that at each time, the OLS 
    #parameters are estimated based on the previous "window" bars
    df = pd.merge(df, model.beta.rename(columns={'feature1': 'hedge_ratio'}).shift(1), right_index=True, left_index=True, how='left')
    df = pd.merge(df, pd.DataFrame(model.alpha).shift(1), right_index=True, left_index=True, how='left')
    df['y_hat'] = df['intercept'] + df['hedge_ratio'] * df[asset_x]
    df['residuals'] = df[asset_y] - df["y_hat"]
    df["resid_mean"] = df.residuals.rolling(window).mean()
    df["resid_std"] = df.residuals.rolling(window).std()
    df['z_score'] = ( df["residuals"] - df["resid_mean"] ) / df["resid_std"]

    return df

#######################################################################################################################

def pair_portfolio(data: pd.DataFrame, confidence_level: float=0.05) -> pd.Series:
    """
    Extract a pair trading portfolio from a pair of time series.

    Parameters:
    - data:             A pandas DataFrame containing the 2 time series.
    - confidence_level: The confidence level for the stationarity tests (default 0.05).

    Returns:
    - portfolio: A pandas Series containing the trading signals.
    """

    # Perform the OLS on a pair of data
    alpha, beta, resid  = get_ols_results(data[data.columns[0]], data[data.columns[1]])

    # Check if the time-series is stationary
    if adfuller(resid)[1] > confidence_level or kpss(resid)[1] < confidence_level:
        return pd.Series({c: None for c in ["z-score", "mean", "std", "char_time", "alpha"] + data.columns.to_list()[:2]})
    
    # Normalize Portfolio
    beta_1  = 1. / (1 + np.abs(beta))
    beta_2  = beta_1 * beta
    alpha   = alpha * beta_1

    # Calculate the z-score and the trading signal
    mean, std   = beta_1*np.mean(resid), beta_1*np.std(resid)
    z_score     = (beta_1*resid[-1] - mean) / std
    char_time   = get_half_life(resid)

    return pd.Series({"z-score": z_score, "mean": mean, "std": std, "char_time": char_time, "alpha": alpha, data.columns[0]: beta_1, data.columns[1]: beta_2})


#######################################################################################################################

def pair_signal(
                    data: pd.DataFrame,
                    entry_threshold: float=1.,
                    time_scale: int=100,
                    **kwargs,
                ) -> pd.Series:
    """
    Extract a pair trading portfolio & return the trading signal associated with it.

    Parameters:
    - data:             A pandas DataFrame containing the time series data.
    - entry_threshold:  The z-score threshold for entering a trade (default 1.).
    - time_scale:       The time scale over which mean-reversion occurs (default 100) i.e. time_scale >> mean-reverting time
    - kwargs:           Additional keyword arguments to pass to the johansen_portfolios function.

    Returns:
    - signal: A pandas Series containing the trading signals.
    """

    # Get the pai trading portfolio
    portfolios  = pair_portfolio(data, **kwargs)

    if portfolios["z-score"] is None or time_scale <= portfolios["char_time"]:
        return pd.Series({c: None for c in ["signal", "exit"] + [f'{c} weight'  for c in data.columns.to_list()[:2]]})

    # Get sign of the trade i.e. +1 long and -1 short
    long    = -1.*np.sign(portfolios["z-score"])
    signal  = portfolios[["mean", "std", "alpha"] + data.columns.to_list()[:2]]
    
    signal["signal"]    = None if long*portfolios["z-score"] > -entry_threshold else long
    signal["exit"]      = int(portfolios["char_time"] * 3)

    signal  = signal.rename({data.columns[0]: f'{data.columns[0]} weight', data.columns[1]: f'{data.columns[1]} weight'})

    return signal

#######################################################################################################################

def pair_entry_exit(
                        data: pd.DataFrame,
                        window_length: int=100,
                        exit_threshold: float=0.,
                        cols: List[str]=["close"],
                        **kwargs,
                    ) -> pd.Series:
    """
    Extract a pair trading portfolio & return the trading signal and associated exit time.

    Parameters:
    - data:             A numpy array or pandas DataFrame containing the time series data.
    - window_length:    The length of the rolling window (default 100).
    - exit_threshold:   The z-score threshold for exiting a trade (default 0).
    - cols:             The column to use for the trading signal (default "close")
    - kwargs:           Additional keyword arguments to pass to the pair

    Returns:
    - signal: A pandas Series containing the trading signals.
    """

    cols_w  = [f'{c} weight'  for c in cols]

    # Get the pair trading signal
    signal  = pair_signal(data[cols].iloc[:window_length], **kwargs)

    if signal["signal"] is None:
        return pd.Series({c: None for c in ["signal", "exit"] + cols_w})
    
    # Get the z-score in real time
    index_0 = int(np.minimum(window_length + signal["exit"], len(data)))
    z_score = data[cols].iloc[window_length:index_0].to_numpy().dot(signal[cols_w].to_numpy())
    z_score = (z_score  - signal["mean"] - signal["alpha"]) / signal["std"]

    # Find the exit point
    j   = np.where(signal["signal"]*z_score > -1.*exit_threshold)[0]

    signal["exit"]  = data.index[index_0-1]
    if len(j) > 0:
        signal["exit"]  = data.index[window_length + j[0]]

    # Return the trading signals
    return signal[cols_w + ["signal", "exit"]]


#######################################################################################################################

def allan_test():
    import matplotlib.pyplot as plt

    y = pd.read_csv("Data/forex/EUR_USD/1m/OHLC/1563535876_1713535876.csv")
    x = pd.read_csv("Data/forex/GBP_USD/1m/OHLC/1563535876_1713535876.csv")
    y["timestamp"] = pd.to_datetime(y["close_time"])
    y.set_index("timestamp", inplace=True)
    y = y[["Close"]].rename(columns={"Close": "EUR_USD"})
    x["timestamp"] = pd.to_datetime(x["close_time"])
    x.set_index("timestamp", inplace=True)
    x = x[["Close"]].rename(columns={"Close": "GBP_USD"})
    df = pd.merge(x, y, right_index=True, left_index=True, how="inner")
    df.to_csv('Data/Backtest/Combined_EURUSD_GBPUSD.csv')

    #df = pd.read_csv('Data/Backtest/Combined_EURUSD_GBPUSD.csv')
    #df.set_index("timestamp", inplace=True)
    #print(df.head())


    #ols = dynamicOLS(df, "GBP_USD", "EUR_USD", 1000)

    #ols.to_csv("Data/Backtest/dynamicOLS.csv")

    #print(pairs_trading.loc[(pairs_trading.z_score <= -0.5)])
    #print(pairs_trading.loc[pairs_trading.signal != pairs_trading.exit_signal])
    
    #print(pairs_trading.hedge_ratio.std())
    
    #pairs_trading.to_csv('Data/Backtest/PairsTradingStrategy-EUR_USD-GBP_USD.csv')
    #print(pairs_trading.loc[pairs_trading.signal==1])
    #resid = PairsTrading(df.EUR_USD, df.GBP_USD)
    #print(resid)
    #Check_Mean_Reversion(resid)


    #ax = df[['EUR_USD', 'GBP_USD']].plot(figsize=(8, 6), title='EUR/USD and GBP/USD')
    #plt.show()

    # Save the figure
    #fig = ax.get_figure()
    #fig.savefig('Backtest/Figure/pairs_trading_test.png')

    #PairsTrading(df.AUD_USD, df.CAD_USD)

    #PairsTrading(df.GBP_USD, df.EUR_USD)

#######################################################################################################################

def virgile_test():
    from Strategy.utils import get_strategy

    def generate_ar1_process(n, phi, sigma):
        """
        Generate an AR(1) process.

        Parameters:
        - n:     The number of time steps.
        - phi:   The autoregressive coefficient.
        - sigma: The standard deviation of the noise.

        Returns:
        - process: A numpy array containing the AR(1) process.
        """
        process = np.zeros(n)
        process[0] = np.random.normal(0, sigma)
        for i in range(1, n):
            process[i] = (1-phi) * process[i-1] + np.random.normal(0, sigma)
        return process

    # Example usage
    n = 1000
    phi = 0.1
    sigma = 1.0
    residual    = generate_ar1_process(n, phi, sigma)

    x_t = np.cumsum(np.random.normal(0, 1, n))
    y_t = 1. + 2.*x_t + residual

    data = pd.DataFrame({"x": x_t, "y": y_t})

    signal = get_strategy(data, pair_entry_exit)
    print(signal[signal["signal"] == 1])

#######################################################################################################################

if __name__ == "__main__":

    virgile_test()



