import numpy as np

class Breakout:
    def __init__(self, price_df, volume_col, price_col, range_period, ma_period, ma_threshold,
                volume_period, volume_thres):
        
        ###Function should give you the choice to choose your breakout definition: is it range or MA, or both, and do you want to apply a volume filter, if yes by how much?
        #Ultimately, shoud be able to define any kind of range and breakout definition
        self.price_df = price_df
        self.volume_col = volume_col
        self.price_col = price_col
        self.range_period = range_period
        self.volume_period = volume_period
        self.volume_thres = volume_thres
        #self.breakout_method = breakout_method
        self.ma_threshold = ma_threshold
        self.ma_period = ma_period
        self.breakout_sum = self.combine_breakouts()
    
    def Range_Breakout(self):
        #1. Identify lowest low and highest high - the range - in the range_period
        self.highest_high = self.price_df["High"].shift(1).rolling(self.range_period).max()
        self.lowest_low = self.price_df["Low"].shift(1).rolling(self.range_period).min()
        return (self.price_df[self.price_col] > self.highest_high) * 1 + (self.price_df[self.price_col] < self.lowest_low) * -1
    
    def MA_Breakout(self):
        self.ma = self.price_df[self.price_col].rolling(self.ma_period).mean()
        return (self.price_df[self.price_col] > self.ma_threshold * self.ma) * 1 + (self.price_df[self.price_col] < self.ma_threshold * (1-self.ma)) * -1

    def Volume_Breakout(self):
        self.volume = self.price_df["Volume"].rolling(self.volume_period).mean()
        return (self.price_df["Volume"] > self.volume_thres * self.volume) * 1 + (self.price_df["Volume"] < (1-self.volume_thres) * self.volume) * -1
    
    def combine_breakouts(self):
        range_breakout = self.Range_Breakout()
        ma_breakout = self.MA_Breakout()
        volume_breakout = self.Volume_Breakout()
        breakout_sum = range_breakout + ma_breakout + volume_breakout
        return breakout_sum
    
    def generate_entry_signals(self):
        return np.where(self.breakout_sum == 3, 1, np.where(self.breakout_sum == -3, -1, 0))



