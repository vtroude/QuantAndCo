import pandas   as pd

from abc    import ABC, abstractmethod
from joblib import Parallel, delayed
from typing import Optional, List, Tuple, Union

class Signal(ABC):
    
    def __init__(self, **kwargs) -> None:
        pass

    #########################################################################################################
    """ Signal & Backtest Functions """
    #########################################################################################################

    @abstractmethod
    def signal(
                self,
                time_series: pd.DataFrame,
                signal_threshold: float=1.,
                use_log: bool=False,
                look_a_head: int=1
            ) -> Optional[pd.Series]:
        """
        Get a trading signal

        Parameters:
        - time_series:          A pandas DataFrame containing the time series data.
        - signale_threshold:    The threshold for the trading signal (default 1)
        - use_log:              Whether to use the log of the time series (default False)
        - look_ahead:           The number of time steps to look ahead to search for a signal (default 1)

        Returns:
        - A pandas Series containing the signal, portfolio, and residual mean, standard deviation and half-time
        """
        pass

    @abstractmethod
    def backtest(
                    self,
                    data: pd.DataFrame,
                    **kwargs,
                ) -> pd.Series:
        """
        Perform a mean-reversion strategy based on the Hurst exponent.

        Parameters:
        - data:             A numpy array or pandas DataFrame containing the time series data.
        - kwargs:           Additional keyword arguments to pass to the signal function

        Returns:
        - signal: A pandas Series containing the trading signals.
        """
        pass

    #########################################################################################################
    """ Help Functions """
    #########################################################################################################

    def __str__(self) -> str:
        return "AbstractSignal"

    ###############################################################################################
    # Data Functions
    ###############################################################################################

    @abstractmethod
    def get_data(
                    self,
                    data: pd.DataFrame,
                    cols: Optional[List[str]]=None,
                    **kwargs
                ) -> Union[pd.DataFrame, List[str]]:
        """
        Get the data to use for the trading signal.

        Parameters:
        - data: A pandas DataFrame containing the time series data.
        - cols: The columns to use for the trading signal (default all columns).

        Returns:
        - A list containing the columns to use for the trading signal.
        """
        pass

    def get_cols_weight(self, cols: List[str]) -> List[str]:
        """
        Get the columns to use for the trading signal.

        Parameters:
        - cols: The columns to use for the trading signal.

        Returns:
        - A list containing the columns to use for the trading signal.
        """

        return [f"{c} weight" for c in cols]

    #########################################################################################################
    """ Backtest Parallelization Functions """
    #########################################################################################################

    def parallelization_backtest(
                                    self,
                                    i: int,
                                    **kwargs
                                ) -> pd.Series:
        """
        Get a signal at the i-th index of the data.

        Parameters:
        - i:                The index of the data.
        - data:             A pandas DataFrame containing the time series data.
        - get_signal:       The signal function to use.
        - kwargs:           Additional keyword arguments to pass to the get_signal function.

        Returns:
        - signal:   A pandas Series containing the trading signals and the exit time.
        """
        
        return self.backtest(self.data.iloc[i:], **kwargs)

    def full_backtest(
                        self,
                        data: pd.DataFrame,
                        window_length: int=100,
                        cols: Optional[List[str]]=None,
                        n_jobs: Optional[int]=1,
                        look_a_head: int=1,
                        step: int=1,
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

        self.data = data

        # Get the columns to use
        if cols is None:
            cols = data.columns.to_list()

        # Get the signal on a rolling window
        if n_jobs is not None and n_jobs > 1:
            signal = Parallel(n_jobs=n_jobs)(
                                                delayed(self.parallelization_backtest)(i, window_length=window_length, cols=cols, look_a_head=look_a_head, **kwargs)
                                                for i in range(0, len(data) - window_length- look_a_head, step)
                                            )
        else:
            signal  = []
            for i in range(0, len(data) - window_length - look_a_head, step):
                signal.append(self.parallelization_backtest(i, window_length=window_length, look_a_head=look_a_head, cols=cols, **kwargs))
        

        signal  = [s for s in signal if not s is None]
        if len(signal) == 0:
            return pd.DataFrame()
        
        signal  = pd.concat(signal, axis=0).dropna()
        signal  = signal[signal["signal"] != 0].sort_values("entry")
        signal  = signal[signal["exit"] > signal["entry"]].reset_index(drop=True)

        # Return the trading signals
        return signal