import pandas as pd
from typing import List

class TSIStrategy:
    def __init__(self, strategy_df: pd.DataFrame, indicators_cols: List[str], 
                 long_thres: float, short_thres: float):
        """
        Initialize the TSI strategy with thresholds.

        Args:
        - price_df:    Close price time-series. Index must be timestamp.
        - indicators_df: Time-series containing one or more Stochastic TSI indicators (with different spans). Must only contain Stochastic TSI columns + timestamp as index.
        - indicators_cols: List of indicators used to build the signals.
        - long_thres: Float between 0 and 1. Threshold for generating a long signal.
        - short_thres: Float between 0 and 1. Threshold for generating a short signal.

        

        By default, the strategy will exit a position when the current signal is no longer matching the signal that generated the position.

        """
        self.strategy_df = strategy_df
        self.indicators_cols = indicators_cols
        self.long_thres = long_thres
        self.short_thres = short_thres
        self.check_thresholds()
    
    def check_thresholds(self):
        if self.long_thres >= self.short_thres:
            raise ValueError(f'This is a mean-reverting strategy. Long threshold must be smaller than short thresold.')
    
    def long_signals(self):
        return (self.strategy_df[self.indicators_cols] < self.long_thres).all(axis=1)*1
    
    def short_signals(self):
        return (self.strategy_df[self.indicators_cols] > self.short_thres).all(axis=1)*-1
    
    def generate_signals(self):
        return self.long_signals() + self.short_signals()
        




