import pandas as pd

from typing import List, Optional, Callable, Tuple, Union

from multiprocessing    import Pool

from joblib import Parallel, delayed
  
#######################################################################################################################

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

def get_strategy_for_parallelization(
                                        i: int,
                                        data: pd.DataFrame,
                                        get_signal: Callable,
                                        **kwargs
                                    ) -> Tuple[Union[str, pd.Timestamp, pd.DatetimeIndex, int, pd.Index], pd.Series]:
    """
    Get a signal at the i-th index of the data.

    Parameters:
    - i:                The index of the data.
    - data:             A pandas DataFrame containing the time series data.
    - get_signal:       The signal function to use.
    - kwargs:           Additional keyword arguments to pass to the get_signal function.

    Returns:
    - index:    The index of the data.
    - signal:   A pandas Series containing the trading signals.
    """

    return data.index[i], get_signal(data.iloc[i:], **kwargs)

#######################################################################################################################

def get_strategy(
                    data: pd.DataFrame,
                    get_signal: Callable,
                    window_length: int=100,
                    cols: Optional[List[str]]=None,
                    n_jobs: Optional[int]=1, 
                    **kwargs
                ) -> pd.DataFrame:
    """
    Perform a strategy from a signal function

    Parameters:
    - data:             A numpy array or pandas DataFrame containing the time series data.
    - get_signal:       The signal function to use.
    - window_length:    The length of the rolling window (default 100).
    - cols:             The columns to use in the signal function (default all columns).
    - n_jobs:           The number of jobs to use for parallelization (default 1).
    - kwargs:           Additional keyword arguments to pass to the get_signal function.

    Returns:
    - signal: A pandas DataFrame containing the trading signals + price time-series
    """

    # Get the columns to use
    if cols is None:
        cols = data.columns.to_list()

    # Create a DataFrame to store the signals
    signal = pd.DataFrame(index=data.index, columns=["signal", "exit"] + [f"{c} weight" for c in cols])

    # Get the signal on a rolling window
    if n_jobs is not None and n_jobs > 1:
        results = Parallel(n_jobs=n_jobs)(
                                            delayed(get_strategy_for_parallelization)(i, data, get_signal, cols=cols, **kwargs)
                                            for i in range(len(data) - window_length)
                                        )

        for res in results:
            signal.loc[res[0]] = res[1]
    else:
        for i in range(len(data) - window_length):
            signal.loc[data.index[i]]   = get_signal(data.iloc[i:], cols=cols, **kwargs)

    # Return the trading signals
    return pd.concat([signal, data], axis=1)
