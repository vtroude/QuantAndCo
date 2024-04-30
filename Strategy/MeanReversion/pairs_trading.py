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


def get_ols_results(x, y):
    ols = OLS_reg(x, y)
    results = ols.fit()
    alpha, beta = results.params
    y_hat = ols.predict(results.params)
    resid = y.values.reshape(-1,) - y_hat

    return alpha, beta, resid

def pairs_trading_signals(df, asset_y, asset_x, alpha, beta, resids, entry_zScore, exit_zScore):

    df['hedge_ratio'] = beta

    df["y_hat"] = alpha + beta * df[asset_x]
    resid_vali = df[asset_y] - df["y_hat"]
    resid_mean, resid_std = np.mean(resids), np.std(resids)
    resid_zscore = (resid_vali - resid_mean) / resid_std
    df["z_score"] = resid_zscore
    df["signal"] = (resid_zscore <= - entry_zScore) *1 + (resid_zscore >= entry_zScore) * -1
    df["exit_signal"] = (resid_zscore >= - exit_zScore) * -1 + (resid_zscore <= exit_zScore) * 1
    df["Portfolio"] = df[asset_y] - beta * df[asset_x]

    return df




def Pairs_Trading(df, asset_x, asset_y, entry_zScore, exit_zScore, end_train_period, dynamic_regression=False,
                  window=1000):
    """
    The strategy is trained with a train set and a validation set.
    The first hedge ratio is estimated on the train set.
    Then, it is dynamically adjusted every "window" bar by adding the new data, and used for the subsequent "window" bars.
    """
    train_index = df.reset_index().loc[df.index>=end_train_period].index[0]
    y = df[[asset_y]]
    x = df[[asset_x]]
    y_train = y.iloc[:train_index]
    x_train = x.iloc[:train_index]
    intercept, hedge_ratio, resid = get_ols_results(x_train, y_train)
    resid_means = [np.mean(resid)]
    resid_stds = [np.std(resid)]
    y_train["residuals"] = resid
    betas = [hedge_ratio]
    alphas = [intercept]
    #Check_Mean_Reversion(y_train[["residuals"]])

    df['signal'] = 0
    df['exit_signal'] = 0
    df['z_score'] = np.nan
    df['hedge_ratio'] = np.nan

    df_vali = df.iloc[train_index:]
    y_vali = y.iloc[train_index:]
    x_vali = x.iloc[train_index:]

    if dynamic_regression == False:
        beta = betas[-1]
        alpha = alphas[-1]
        y_vali["y_hat"] = alpha + beta * x_vali
        resid_vali = y_vali[asset_y] - y_vali["y_hat"]
        resid_mean, resid_std = resid_means[-1], resid_stds[-1]
        resid_zscore = (resid_vali - resid_mean) / resid_std
        df_vali["hedge_ratio"] = beta
        df_vali["z_score"] = resid_zscore
        df_vali["signal"] = (resid_zscore <= - entry_zScore) *1 + (resid_zscore >= entry_zScore) * -1
        df_vali["exit_signal"] = (resid_zscore >= - exit_zScore) * -1 + (resid_zscore <= exit_zScore) * 1
        df_vali["Portfolio"] = df_vali[asset_y] - beta * df_vali[asset_x]
    
    else:
        n_windows = int( len(y_vali) / window )
        if n_windows < 3:
            raise ValueError(f'{window} is too large for dataset')
        
        for t in range(n_windows):
            w_init = t * window
            w_last = (t+1) * window
            sub_y_vali = y_vali.iloc[w_init:w_last]
            sub_x_vali = x_vali.iloc[w_init:w_last]
            beta = betas[-1]
            alpha = alphas[-1]
            sub_y_vali["y_hat"] = alpha + beta * sub_x_vali
            resid_vali = sub_y_vali[asset_y] - sub_y_vali["y_hat"]
            resid_mean, resid_std = resid_means[-1], resid_stds[-1]
            resid_zscore = (resid_vali - resid_mean) / resid_std
            df_vali.loc[df_vali.index[w_init:w_last], "hedge_ratio"] = beta
            df_vali.loc[df_vali.index[w_init:w_last], "z_score"] = resid_zscore
            df_vali.loc[df_vali.index[w_init:w_last], "signal"] = (resid_zscore <= - entry_zScore) *1 + (resid_zscore >= entry_zScore) * -1
            df_vali.loc[df_vali.index[w_init:w_last], "exit_signal"] = (resid_zscore >= - exit_zScore) * -1 + (resid_zscore <= exit_zScore) * 1
            df_vali.loc[df_vali.index[w_init:w_last], "Portfolio"] = df_vali.loc[df_vali.index[w_init:w_last], asset_y] - beta * df_vali.loc[df_vali.index[w_init:w_last], asset_x]

            new_x_train = x.iloc[:w_last]
            new_y_train = y.iloc[:w_last]
            new_alpha, new_beta, new_resid = get_ols_results(new_x_train, new_y_train)

            resid_means.append(np.mean(new_resid))
            resid_stds.append(np.std(new_resid))
            betas.append(new_beta)
            alphas.append(new_alpha)

        if w_last < len(y_vali) - 1:
            sub_y_vali = y_vali.iloc[w_last:]
            sub_x_vali = x_vali.iloc[w_last:]
            beta = betas[-1]
            alpha = alphas[-1]
            sub_y_vali["y_hat"] = alpha + beta * sub_x_vali
            resid_vali = sub_y_vali[asset_y] - sub_y_vali["y_hat"]
            resid_mean, resid_std = resid_means[-1], resid_stds[-1]
            resid_zscore = (resid_vali - resid_mean) / resid_std
            df_vali.loc[df_vali.index[w_last:], "hedge_ratio"] = beta
            df_vali.loc[df_vali.index[w_last:], "z_score"] = resid_zscore
            df_vali.loc[df_vali.index[w_last:], "signal"] = (resid_zscore <= - entry_zScore) *1 + (resid_zscore >= entry_zScore) * -1
            df_vali.loc[df_vali.index[w_last:], "exit_signal"] = (resid_zscore >= - exit_zScore) * -1 + (resid_zscore <= exit_zScore) * 1
            df_vali.loc[df_vali.index[w_last:], "Portfolio"] = df_vali.loc[df_vali.index[w_last:], asset_y] - beta * df_vali.loc[df_vali.index[w_last:], asset_x]

    df_ = df_vali[[asset_y, asset_x, "hedge_ratio", "Portfolio", "z_score", "signal", "exit_signal"]]

    return df_

    #return pd.concat([df.iloc[:train_index][[asset_y, asset_x, "hedge_ratio", "z_score", "signal", "exit_signal"]], df_], axis=0)

    #y_vali["y_hat"] = intercept + hedge_ratio * x_vali
    #resid_test = y_vali[asset_y] - y_vali["y_hat"]
    #resid_zscore = (resid_test - resid_mean) / resid_std
    #df['signal'] = 0
    #df['exit_signal'] = 0
    #df['z_score'] = np.nan
    #df.loc[df.index[train_index:], "z_score"] = resid_zscore
    #df.loc[df.index[train_index:], "signal"] = (resid_zscore <= - entry_zScore) *1 + (resid_zscore >= entry_zScore) * -1
    #df.loc[df.index[train_index:], "exit_signal"] = (resid_zscore >= - exit_zScore) * -1 + (resid_zscore <= exit_zScore) * 1
    #df["Portfolio"] = df[asset_y] - hedge_ratio * df[asset_x]
    #df["Hedge_Ratio"] = hedge_ratio

    #df_ = df[["Portfolio", "Hedge_Ratio", "z_score", "signal", "exit_signal"]].iloc[train_index:]

    #return df_

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

    df = df.loc[df.index<='2023-01-01']

    pairs_trading = Pairs_Trading(df, "EUR_USD", "GBP_USD", entry_zScore=1, exit_zScore=0.5, end_train_period="2022-01-01")
    
    print(pairs_trading.hedge_ratio.std())
    
    pairs_trading.to_csv('Data/Backtest/PairsTradingStrategy-EUR_USD-GBP_USD.csv')
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



