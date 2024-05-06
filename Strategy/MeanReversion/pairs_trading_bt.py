from Research.statistical_tests import adf_test, OLS_reg, get_half_life
import numpy as np
import pandas as pd

class PairsTrading:
    def __init__(self, df, asset_x, asset_y, end_train_period, entry_threshold, exit_threshold):
        self.df = df
        self.asset_x = asset_x
        self.asset_y = asset_y
        self.end_train_period = end_train_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.train_df = self.df.loc[self.df.index <= self.end_train_period]
        self.test_df = self.df.loc[self.df.index > self.end_train_period]
        self.x_train = self.train_df[[self.asset_x]]
        self.y_train = self.train_df[[self.asset_y]]
        self.alpha, self.beta, self.resid = self.get_ols_results()
        self.x_test = self.test_df[[self.asset_x]]
        self.y_test = self.test_df[[self.asset_y]]
        self.resid_zscore = self.calculate_zscore()
        self.long_entries = pd.DataFrame()
        self.short_entries = pd.DataFrame()
        self.long_exits = pd.DataFrame()
        self.short_exits = pd.DataFrame()


    def get_ols_results(self):
        ols = OLS_reg(self.x_train, self.y_train)
        results = ols.fit()
        alpha, beta = results.params
        y_hat = ols.predict(results.params)
        resid = self.y_train.values.reshape(-1,) - y_hat

        return alpha, beta, resid
    
    def calculate_zscore(self):
        self.test_df['hedge_ratio'] = self.beta

        self.test_df["y_hat"] = self.alpha + self.beta * self.test_df[self.asset_x]
        resid_vali = self.test_df[self.asset_y] - self.test_df["y_hat"]
        resid_mean, resid_std = np.mean(self.resid), np.std(self.resid)
        resid_zscore = (resid_vali - resid_mean) / resid_std
        self.test_df["z_score"] = resid_zscore

        return resid_zscore
    
    def generate_entries(self):

        self.long_entries[self.asset_y] = self.resid_zscore <= - self.entry_threshold
        self.short_entries[self.asset_y] = self.resid_zscore >= self.entry_threshold

        self.short_entries[self.asset_x] = self.long_entries[self.asset_y]
        self.long_entries[self.asset_x] = self.short_entries[self.asset_y]

        return self.long_entries, self.short_entries
    
    def generate_exits(self):

        self.long_exits[self.asset_y] = self.resid_zscore >= - self.exit_threshold
        self.short_exits[self.asset_y] = self.resid_zscore <= self.exit_threshold

        self.long_exits[self.asset_x] = self.short_exits[self.asset_y]
        self.short_exits[self.asset_x] = self.long_exits[self.asset_y]

        return self.long_exits, self.short_exits
    

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
    pairs_trading = PairsTrading(df, "EUR_USD", "GBP_USD", "2022-06-01", 1, 0.5)
    entries = pairs_trading.generate_signals()
    print(entries.head())