from Research.statistical_tests import adf_test, OLS_reg, get_half_life, Check_Mean_Reversion
import pandas as pd
import matplotlib.pyplot as plt

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
    print(f'Y = {intercept} {sign} {hedge_ratio} * X')
    return resid

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
    resid = PairsTrading(df.EUR_USD, df.GBP_USD)
    #print(resid)
    Check_Mean_Reversion(resid)


    #ax = df[['EUR_USD', 'GBP_USD']].plot(figsize=(8, 6), title='EUR/USD and GBP/USD')
    #plt.show()

    # Save the figure
    #fig = ax.get_figure()
    #fig.savefig('Backtest/Figure/pairs_trading_test.png')

    #PairsTrading(df.AUD_USD, df.CAD_USD)

    #PairsTrading(df.GBP_USD, df.EUR_USD)



