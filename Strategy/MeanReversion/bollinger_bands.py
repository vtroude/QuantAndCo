import pandas as pd

class BollingerBands:
    def __init__(self, price_df, entry_threshold, exit_threshold, lookback_period, price_col="Close"):
        self.price_df = price_df
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.lookback_period = lookback_period
        self.price_col = price_col
        self.z_score = self.calculate_zscore()
    
    def calculate_zscore(self):
        rolling_mean = self.price_df[[self.price_col]].rolling(self.lookback_period).mean()
        rolling_std = self.price_df[[self.price_col]].rolling(self.lookback_period).std()
        z_score = (self.price_df[[self.price_col]] - rolling_mean) / rolling_std
        return z_score
    
    def generate_entry_signals(self):
        self.long_entry = (self.z_score < -self.entry_threshold) * 1
        self.short_entry = (self.z_score > self.entry_threshold) * -1
        return self.long_entry + self.short_entry
    
    def generate_exit_signals(self):
        self.long_exit = (self.z_score >= -self.exit_threshold) * -1
        self.short_exit = (self.z_score <= self.exit_threshold) * 1
        return self.long_exit + self.short_exit

if __name__ == "__main__":
    df = pd.read_csv("Data/forex/EUR_USD/1d/OHLC/2019-07-19 13:31:16_2024-04-19 16:11:16.csv")
    BBands = BollingerBands(df, entry_threshold=1, exit_threshold=0, lookback_period=30, price_col="Close")
    df["z_score"] = BBands.calculate_zscore()
    df["entry_signal"] = BBands.generate_entry_signals()
    df['exit_signal'] = BBands.generate_exit_signals()
    print(df.loc[df.entry_signal==-1])