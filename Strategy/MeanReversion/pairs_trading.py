from Research.statistical_tests import adf_test, OLS_reg, get_half_life, Check_Mean_Reversion
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyfinance.ols import PandasRollingOLS

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

if __name__ == "__main__":
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



