import pandas as pd

class RSIStrategy:
    def __init__(self, price_df, indicators_df, indicators_cols):
        """
        Initialize the RSI strategy with thresholds.

        Args:
        entry_long_thres (float): Threshold for generating a long entry signal.
        entry_short_thres (float): Threshold for generating a short entry signal.
        exit_long_thres (float): Threshold for exiting long positions.
        exit_short_thres (float): Threshold for exiting short positions.
        """

        self.price_df = price_df
        self.indicators_df = indicators_df
        self.indicators_cols = indicators_cols
    
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

        return pd.merge(self.price_df, self.indicators_df, right_index=True, left_index=True)

    def entry_signal(self, strategy_df, entry_long_thres, entry_short_thres):
        """
        Calculate entry signals for a DataFrame.

        Args:
        df (pd.DataFrame): DataFrame containing market data.
        indicators_columns (list of str): Column names in df that contain indicator values.

        Returns:
        pd.Series: Entry signals for each row in df.
        """
        long_signal = (strategy_df[self.indicators_cols] < entry_long_thres).all(axis=1) * 1
        short_signal = (strategy_df[self.indicators_cols] > entry_short_thres).all(axis=1) * -1
        return long_signal + short_signal #Only one signal can be true - so the sum will either return 1, -1 or 0

    def exit_long(self, strategy_df, exit_long_thres):
        """
        Calculate exit signals for long positions based on a DataFrame.

        Returns:
        pd.Series: Boolean series where True indicates a signal to exit a long position.
        """
        return (strategy_df[self.indicators_cols] >= exit_long_thres).any(axis=1)

    def exit_short(self, strategy_df, exit_short_thres):
        """
        Calculate exit signals for short positions based on a DataFrame.

        Returns:
        pd.Series: Boolean series where True indicates a signal to exit a short position.
        """
        return (strategy_df[self.indicators_cols] <= exit_short_thres).any(axis=1)
    

    class RF_strategy():
        pass




