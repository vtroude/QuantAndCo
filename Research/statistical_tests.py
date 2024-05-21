from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
import numpy as np
import pandas as pd
import warnings
#from arch.unitroot import VarianceRatio
import matplotlib.pyplot as plt

def assert_timestamp(timeseries, time_col='timestamp'):
    if isinstance(timeseries, np.ndarray):
        return pd.Series(timeseries)

    if isinstance(timeseries, pd.DataFrame):
        if time_col != timeseries.index.name:
            if time_col not in timeseries.columns:
                raise ValueError(f"{time_col} is missing from timeseries")
            else:
                timeseries.index = pd.to_datetime(timeseries[time_col])
        else:
            if isinstance(timeseries.index, pd.DatetimeIndex) == False:
                timeseries.index = pd.to_datetime(timeseries.index)
    
    return timeseries.sort_index(ascending=True)


def adf_test(timeseries):
    timeseries = assert_timestamp(timeseries)
    dftest = adfuller(timeseries, autolag="AIC", maxlag=1)
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value

    test_stat = dfoutput["Test Statistic"]
    for pct in [1, 5, 10]:
        adf_value = dfoutput[f"Critical Value ({str(pct)}%)"]
        if test_stat < adf_value:
            print(f"ADF Test statistic = {test_stat: .2f} < {adf_value: .2f} ({pct}%-Critical Value)")
            print(f"Unit-root (= non-stationarity) rejected at {100-pct}% confidence interval.")
            print('\n')
        else:
            print(f"ADF Test statistic = {test_stat: .2f} > {adf_value: .2f} ({pct}%-Critical Value)")
            print(f"Unit-root (= non-stationarity) CANNOT be rejected at {100-pct}% confidence interval.")
            print('\n')
        
    return dfoutput

def OLS_reg(x, y, print_results=False):
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    results = model.fit()
    if print_results:
        print(results.summary())
    return model


def auto_reg(timeseries, lag=1):
    timeseries = assert_timestamp(timeseries)
    delta_y = timeseries.diff().iloc[1:]
    y_shift = timeseries.shift(1).iloc[1:]
    ols = OLS_reg(y_shift, delta_y, print_results=False)
    results = ols.fit()
    alpha, beta = results.params
    return alpha, beta

def get_half_life(timeseries, lag=1):
    alpha, beta = auto_reg(timeseries, lag)
    half_life = -np.log(2) / beta
    return half_life


def CADF(asset_y, asset_x):
    ols = OLS_reg(asset_x, asset_y)
    results = ols.fit()
    intercept, hedge_ratio = results.params
    #1. Check if hedge_ratio is statistically significantly diff. than 0
    y_hat = ols.predict(results.params)
    #2. Perform ADF on residuals
    resid = asset_y - y_hat
    adf = adf_test(resid)
    return hedge_ratio


####################################
####################################
##   written in Matlab : Tomaso Aste, 30/01/2013 ##
##   translated to Python (3.6) : Peter Rupprecht, p.t.r.rupprecht (AT) gmail.com, 25/05/2017 ##


def HurstExponent(S,q=2):
    """"""""""
    Returns Hurst-Exponent H.
    H = 0.5 -> Random Walk
    H < 0.5 -> Mean-Reverting
    H > 0.5 -> Trending

    H can be interpreted as the degree of mean-reversion/trendiness.
    The closer to zero, the more mean-reverting; the closer to 1, the more trending.
    """""""""""

    L=len(S)       
    if L < 100:
        warnings.warn('Data series very short!')
       
    H = np.zeros((len(range(5,20)),1))
    k = 0
    
    for Tmax in range(5,20):
        
        x = np.arange(1,Tmax+1,1)
        mcord = np.zeros((Tmax,1))
        
        for tt in range(1,Tmax+1):
            dV = S[np.arange(tt,L,tt)] - S[np.arange(tt,L,tt)-tt] 
            VV = S[np.arange(tt,L+tt,tt)-tt]
            N = len(dV) + 1
            X = np.arange(1,N+1,dtype=np.float64)
            Y = VV
            mx = np.sum(X)/N
            SSxx = np.sum(X**2) - N*mx**2
            my = np.sum(Y)/N
            SSxy = np.sum( np.multiply(X,Y))  - N*mx*my
            cc1 = SSxy/SSxx
            cc2 = my - cc1*mx
            ddVd = dV - cc1
            VVVd = VV - np.multiply(cc1,np.arange(1,N+1,dtype=np.float64)) - cc2
            mcord[tt-1] = np.mean( np.abs(ddVd)**q )/np.mean( np.abs(VVVd)**q )
            
        mx = np.mean(np.log10(x))
        SSxx = np.sum( np.log10(x)**2) - Tmax*mx**2
        my = np.mean(np.log10(mcord))
        SSxy = np.sum( np.multiply(np.log10(x),np.transpose(np.log10(mcord)))) - Tmax*mx*my
        H[k] = SSxy/SSxx
        k = k + 1
        
    mH = np.mean(H)/q
    
    return mH

'''
def Variance_Ratio_test(timeseries, lags=2):
    """
    Null hypothesis: the process is a random-walk (i.e. non-stationary / non mean-reverting)
    Pvalue is the prob. that the null hypothesis is true.
    """
    vr = VarianceRatio(timeseries, lags=lags)
    ratio = vr.vr
    pvalue = vr.pvalue
    return ratio, pvalue
'''

def Check_Mean_Reversion(timeseries):
    #1. We plot the timeseries - a mean reverting time series should be "pulled back" to its mean
    #when it deviates too much from it
    timeseries = assert_timestamp(timeseries)
    mean = np.mean(timeseries, axis=0).values[0]
    stdev = np.std(timeseries, axis=0).values[0]
    #mean = timeseries.mean(axis=0)
    #stdev = timeseries.std(axis=0)
    ax = timeseries.plot(figsize=(10, 6), title='Price Time Series with Mean and StdDev Lines')
    # Add horizontal lines for mean, mean + 3*stdev, and mean - 3*stdev
    ax.axhline(mean, color='red', linestyle='--', linewidth=2, label='Mean')
    ax.axhline(mean + 1 * stdev, color='green', linestyle='--', linewidth=2, label='Mean + StdDev')
    ax.axhline(mean - 1 * stdev, color='blue', linestyle='--', linewidth=2, label='Mean - StdDev')

    # Adding legend to the plot
    ax.legend()

    fig = ax.get_figure()
    fig.savefig('Backtest/Figure/Mean_reversion_check.png')

    # Show the plot
    plt.show()
    #2. TO DO: Plot the distribution

    #3. Augmented Dickey Fuller test
    adf = adf_test(timeseries)
    #print(adf)

    #4. Hurst Exponent
    #log_p = np.log(timeseries).values.reshape(-1, )
    p = timeseries.values.reshape(-1, )

    #### TO DO: ADD ABOVE A STEP THAT CHECKS THE SHAPE OF THE DATA
    H = HurstExponent(p, q=2)
    if H == 0.5:
        print('Hurst Exponent = 0.5: The timeseries is a random walk.')
    elif H < 0.5:
        if H>=0.45:
            print(f'Hurst Exponent = {H: .2f} < 0.5: the process is weakly mean-reverting.')
        else:
            print(f'Hurst Exponent = {H: .2f} << 0.5: the process is mean-reverting.')
    else:
        if H <= 0.55:
            print(f'Hurst Exponent = {H: .2f} > 0.5: the process is weakly trending.')
        else:
            print(f'Hurst Exponent = {H: .2f} >> 0.5: the process is trending.')

    
if __name__ == '__main__':
    df = pd.read_csv("Data/forex/EUR_USD/1m/OHLC/2019-07-19 13:31:16_2024-04-19 16:11:16.csv")
    df['timestamp'] = pd.to_datetime(df.close_time)
    #df.set_index('timestamp', inplace=True)
    df = assert_timestamp(df)
    df = df.loc[df.index<='2020-01-01']
    df = df[['Close']]
    Check_Mean_Reversion(df)

