from Research.statistical_tests import adf_test, OLS_reg, get_half_life, Check_Mean_Reversion
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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


def Pairs_Trading(df, asset_x, asset_y, entry_long, entry_short, exit_long, exit_short, end_train_period):

    lookback_period = df.reset_index().loc[df.index>=end_train_period].index[0]
    y = df[[asset_y]]
    x = df[[asset_x]]
    y_train = y.iloc[:lookback_period]
    x_train = x.iloc[:lookback_period]
    ols = OLS_reg(x_train, y_train)
    results = ols.fit()
    print(results.summary())
    intercept, hedge_ratio = results.params
    y_hat = ols.predict(results.params)
    resid = y_train.values.reshape(-1,) - y_hat
    resid_mean = np.mean(resid)
    resid_std = np.std(resid)
    y_train["residuals"] = resid
    #Check_Mean_Reversion(y_train[["residuals"]])
    y_test = y.iloc[lookback_period:]
    x_test = x.iloc[lookback_period:]
    y_test["y_hat"] = intercept + hedge_ratio * x_test
    resid_test = y_test[asset_y] - y_test["y_hat"]
    resid_zscore = (resid_test - resid_mean) / resid_std
    df['signal'] = 0
    df['exit_signal'] = 0
    df['z_score'] = np.nan
    df.loc[df.index[lookback_period:], "z_score"] = resid_zscore
    df.loc[df.index[lookback_period:], "signal"] = (resid_zscore <= entry_long) *1 + (resid_zscore >= entry_short) * -1
    df.loc[df.index[lookback_period:], "exit_signal"] = (resid_zscore >= exit_long) * -1 + (resid_zscore <= exit_short) * 1
    df["Portfolio"] = df[asset_y] - hedge_ratio * df[asset_x]
    df["Hedge_Ratio"] = hedge_ratio

    df_ = df[["Portfolio", "Hedge_Ratio", "z_score", "signal", "exit_signal"]].iloc[lookback_period:]

    return df_

 


if __name__ == "__main__":

    y = pd.read_csv("Data/forex/EUR_USD/1m/OHLC/1563535876_1713535876.csv")
    x = pd.read_csv("Data/forex/GBP_USD/1m/OHLC/1563535876_1713535876.csv")
    y["timestamp"] = pd.to_datetime(y["close_time"])
    y.set_index("timestamp", inplace=True)
    y = y[["Close"]].rename(columns={"Close": "EUR_USD"})
    x["timestamp"] = pd.to_datetime(x["close_time"])
    x.set_index("timestamp", inplace=True)
    x = x[["Close"]].rename(columns={"Close": "GBP_USD"})
    df = pd.merge(x, y, right_index=True, left_index=True, how="inner")

    pairs_trading = Pairs_Trading(df, "EUR_USD", "GBP_USD", entry_long=-1, 
                                  exit_long=-0.5, entry_short=1, exit_short=0.5, end_train_period="2023-01-01")
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



