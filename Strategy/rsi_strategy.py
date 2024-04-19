import pandas as pd
from typing import List

class RSIStrategy:
    def __init__(self, price_df:pd.DataFrame, indicators_df:pd.DataFrame, indicators_cols: List[str], 
                 long_thres: float, short_thres: float):
        """
        Initialize the RSI strategy with thresholds.

        Args:
        - price_df:    Close price time-series. Index must be timestamp.
        - indicators_df: Time-series containing one or more Stochastic RSI indicators (with different spans). Must only contain Stochastic RSI columns + timestamp as index.
        - indicators_cols: List of indicators used to build the signals.
        - entry_long_thres: Float between 0 and 1. Threshold for generating a long signal.
        - entry_short_thres: Float between 0 and 1. Threshold for generating a short signal.

        By default, the strategy will exit a position when the current signal is no longer matching the signal that generated the position.

        """

        self.price_df = price_df
        self.indicators_df = indicators_df
        self.indicators_cols = indicators_cols
        self.entry_long_thres = long_thres
        self.entry_short_thres = short_thres
        self.check_thresholds()
        self.strategy_df = self.prepare_data()
    
    def check_thresholds(self):
        if self.entry_long_thres >= self.entry_short_thres:
            raise ValueError(f'This is a mean-reverting strategy. Long threshold must be smaller than short thresold.')
    
    def prepare_data(self):
        if self.price_df.index.name != 'timestamp':
            if 'timestamp' not in self.price_df.columns:
                raise ValueError('"timestamp" column missing in price_df')
            else:
                self.price_df['timestamp'] = pd.to_datetime(self.price_df['timestamp'])
                self.price_df.set_index('timestamp', inplace=True)

        if self.indicators_df.index.name != 'timestamp':
            if 'timestamp' not in self.indicators_df.columns:
                raise ValueError('"timestamp" column missing in indicators_df')
            else:
                self.indicators_df['timestamp'] = pd.to_datetime(self.indicators_df['timestamp'])
                self.indicators_df.set_index('timestamp', inplace=True)
        
        if len(self.price_df) != len(self.indicators_df):
            print(f'Datasets have different lengths. price_df has {len(self.price_df)} rows while indicators_df has {len(self.indicators_df)} rows.')

        return pd.merge(self.price_df, self.indicators_df, right_index=True, left_index=True)

    def entry_signals(self):
        """
        Returns 1 if long signal, -1 if short signal
        """
        long_entry = (self.strategy_df[self.indicators_cols] < self.entry_long_thres).all(axis=1) * 1
        short_entry = (self.strategy_df[self.indicators_cols] > self.entry_short_thres).all(axis=1) * -1
        return long_entry + short_entry
        
    
class RF_strategy():
    pass




