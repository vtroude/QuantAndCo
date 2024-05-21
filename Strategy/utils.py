import pandas as pd

from typing import List, Optional, Callable
  
def prepare_data(price_df, indicators_df):
    if price_df.index.name != 'timestamp':
        if 'timestamp' not in price_df.columns:
            raise ValueError('"timestamp" column missing in price_df')
        else:
            price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
            price_df.set_index('timestamp', inplace=True)

    if indicators_df.index.name != 'timestamp':
        if 'timestamp' not in indicators_df.columns:
            raise ValueError('"timestamp" column missing in indicators_df')
        else:
            indicators_df['timestamp'] = pd.to_datetime(indicators_df['timestamp'])
            indicators_df.set_index('timestamp', inplace=True)
    
    if len(price_df) != len(indicators_df):
        print(f'Datasets have different lengths. price_df has {len(price_df)} rows while indicators_df has {len(indicators_df)} rows.')

    return pd.merge(price_df, indicators_df, right_index=True, left_index=True)

#######################################################################################################################

def get_strategy(data: pd.DataFrame, get_signal: Callable, window_length=100, cols: Optional[List[str]]=None,  **kwargs) -> pd.DataFrame:
    """
    Perform a strategy from a signal function

    Parameters:
    - data:             A numpy array or pandas DataFrame containing the time series data.
    - window_length:    The length of the rolling window (default 100).
    - kwargs:           Additional keyword arguments to pass to the get_signal function.

    Returns:
    - signal: A pandas DataFrame containing the trading signals.
    """

    if cols is None:
        cols = data.columns.to_list()

    # Create a DataFrame to store the signals
    signal  = pd.DataFrame(index=data.index, columns=["signal", "exit"] + [f"{c} weight" for c in cols])

    # Get the signal on a rolling window
    for i in range(len(data)-window_length):
        # Get the signal portfolio
        signal.loc[data.index[i]]   = get_signal(data.iloc[i:], cols=cols, **kwargs)

    # Return the trading signals
    return pd.concat([signal, data], axis=1)