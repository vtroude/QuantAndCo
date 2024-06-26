import warnings

import numpy    as np
import pandas   as pd

from typing     import Union, List, Optional

from statsmodels.tsa.stattools      import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from Strategy.signal                        import Signal
from Strategy.MeanReversion.pairs_trading   import get_ols_results

from Research.statistical_tests import HurstExponent, get_half_life


import pylab as pl

#######################################################################################################################

class MeanReversion(Signal):

    def __init__(
                    self,
                    confidence_level: float=0.05,
                    hurst_threshold: float=0.5,
                    minimal_time_scale: int=1,
                    maximal_time_scale: int=100,
                    time_multiplier: int=3,
                    resol: int=1000,
                    use_adfuller: bool=True,
                    use_kpss: bool=False,
                    no_overlap: bool=True
                ):
        """
        Initialize the mean-reversion strategy.

        Parameters:
        - confidence_level:   The confidence level for the ADF & KPSS test (default 0.05)
        - hurst_threshold:    The threshold for the Hurst exponent (default 0.5)
        - minimal_time_scale: The minimal time scale over which mean-reversion occurs (default 1)
        - maximal_time_scale: The maximal time scale over which mean-reversion occurs (default 100)
        - time_multiplier:    The time multiplier for the exit signal (default 3)
        """

        self.confidence_level   = confidence_level      # Confidence level for the ADF & KPSS test
        self.hurst_threshold    = hurst_threshold       # Threshold for the Hurst exponent
        self.minimal_time_scale = minimal_time_scale    # Minimal time scale over which mean-reversion occurs
        self.maximal_time_scale = maximal_time_scale    # Maximal time scale over which mean-reversion occurs
        self.time_multiplier    = time_multiplier       # Time multiplier for the exit signal
        self.resol              = resol
        self.use_adfuller       = use_adfuller
        self.use_kpss           = use_kpss         
        self.no_overlap         = no_overlap        

    #########################################################################################################
    """ Help Functions """
    #########################################################################################################

    def __str__(self) -> str:
        return "MeanReversion"

    ###############################################################################################
    # Data Functions
    ###############################################################################################

    def get_data(
                    self,
                    data: pd.DataFrame, cols:
                    Optional[List[str]]=None,
                    use_log: bool=False
                ) -> Union[pd.DataFrame, List[str]]:
        """
        Get the data to use for the trading signal.

        Parameters:
        - data: A pandas DataFrame containing the time series data.
        - cols: The columns to use for the trading signal (default all columns).

        Returns:
        - A list containing the columns to use for the trading signal.
        """

        # Get the columns to use
        if cols is None:
            cols = data.columns.to_list()
        
        x   = data[cols[0]] if not use_log else np.log(data[cols[0]])

        return x, cols[:1]

    ###############################################################################################
    # Z-Score Functions
    ###############################################################################################

    def z_score(self, x: pd.Series, signal: pd.Series, use_log: bool=False) -> pd.Series:
        """
        Get the z-score of a given time series.

        Parameters:
        - x:        A pandas Series containing the time series data.
        - signal:   A pandas Series containing the mean and standard deviation of the time series.
        - use_log:  Whether to use the log of the time series (default False)

        Returns:
        - A pandas Series containing the z-score of the time series.
        """

        # Use the log or not
        z_score = x if not use_log else np.log(x)

        # Return z-score
        return (z_score - signal["mean"]) / signal["std"]

    def get_z_score(
                        self, x: pd.DataFrame,
                        signal: pd.Series,
                        use_log: bool=False,
                        cols_w: Optional[List[str]]=None
                    ) -> Union[pd.Series, int]:
        """
        Get the z-score of a given time series.

        Parameters:
        - x:        A pandas Series containing the time series data.
        - mean:     The mean of the time series.
        - std:      The standard deviation of the time series.
        - use_log:  Whether to use the log of the time series (default False)

        Returns:
        - A pandas Series containing the z-score of the time series.
        """

        # Get the index of the minimum z-score
        index_0 = int(np.minimum(signal["exit"].max(), len(x)))
        # Get the z-score of the spread in real-time
        spread  = x.iloc[:index_0] if cols_w is None or len(cols_w) < 2 else x.iloc[:index_0].to_numpy().dot(signal[cols_w].to_numpy())
        z_score = self.z_score(spread, signal, use_log=use_log)

        # Return the z-score and the index
        return z_score, index_0

    ###############################################################################################
    # Mean-Reversion Test Functions
    ###############################################################################################

    def test_stationarity(self, time_series: np.ndarray, use_log: bool=False) -> pd.Series:
        """
        Test the stationarity of a given time series.

        Parameters:
        - time_series:      A numpy array containing the time series data.
        - confidence_level: The confidence level for the ADF & KPSS test (default 0.05)

        Returns:
        - A pandas Series containing the Hurst exponent, mean, standard deviation and the characteristic time of the mean-reversion
        """

        # Check if the time-series is stationary
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            if self.use_adfuller:
                if adfuller(time_series[::len(time_series)//self.resol])[1] > self.confidence_level:
                    return None
            if self.use_kpss:
                if kpss(time_series[::len(time_series)//self.resol])[1] < 1. - self.confidence_level:
                    return None

        # Calculate the fractal dimension using Hurst exponent
        x   = np.log(time_series) if use_log else time_series
        hurst_exponent = HurstExponent(x)
        if hurst_exponent > self.hurst_threshold:
            return None

        # Estimate the time-scale over which mean-reversion occurs
        time_scale = self.time_multiplier*get_half_life(pd.Series(x))
        if time_scale < self.minimal_time_scale or time_scale > self.maximal_time_scale:
            return None

        # Return the hurst exponent, mean, standard deviation and the characteristic time
        return pd.Series({"mean": np.mean(x), "std": np.std(x), "half-life": time_scale})
    
    #########################################################################################################
    """ Signal & Backtest Functions """
    #########################################################################################################

    ###############################################################################################
    # Signal Functions
    ###############################################################################################

    def get_signal(
                    self,
                    para: pd.Series,
                    residual: float,
                    signal_threshold: float=1.,
                    use_log: bool=False
                ) -> pd.DataFrame:
        """
        Get signal based on residual z-score & parameters.

        Parameters:
        - para:             A pandas Series containing the mean, standard deviation and the characteristic time of the mean-reversion.
        - residual:         A pandas Series containing the residual time series.
        - signal_threshold: The threshold for the trading signal (default 1)
        - use_log:          Whether to use the log of the residual (default False)

        Returns:
        - A pandas Series containing the trading signal.
        """

        # Check if the time-series is stationary
        if para is None:
            return None
        
        # Calculate the z-score and the trading signal
        z_score = self.z_score(residual, para, use_log=use_log)
        #signal  = -1.*np.sign(z_score) if abs(z_score) > signal_threshold else 0
        #exit    = int(para["half-life"]*self.time_multiplier)

        signal  = pd.DataFrame(np.where(np.abs(z_score) > signal_threshold, -1.*np.sign(z_score), 0), columns=["signal"])
        signal["exit"]  = int(para["half-life"])
        signal["mean"]  = para["mean"]
        signal["std"]   = para["std"]

        # Return the trading signal
        #return pd.Series({"signal": signal, "exit": exit, "mean": para["mean"], "std": para["std"]})
        return signal

    def signal(
                self,
                time_series: pd.Series,
                signal_threshold: float=1.,
                use_log: bool=False,
                look_a_head: int=1,
            ) -> Optional[pd.DataFrame]:
        """
        Get a trading signal based on the Hurst exponent of a given time series.

        Parameters:
        - time_series:      A pandas DataFrame containing the time series data.
        - threshold:        The threshold for the trading signal (default 1)
        - hurst_threshold:  The threshold for the Hurst exponent (default 0.5)
        - time_scale:       The time scale over which mean-reversion occurs (default 100) i.e. time_scale >> mean-reverting time
        - cols:             The column to use for the trading signal (default "close")
        - kwargs:           Additional keyword arguments to pass to the test_stationarity function

        Returns:
        - A pandas Series containing the Hurst exponent, mean, standard deviation and the characteristic time of the mean-reversion
        """

        # Test the stationarity of the time series
        para    = self.test_stationarity(time_series.to_numpy()[:-look_a_head], use_log=use_log)

        # Get the trading signal
        signal  = self.get_signal(para, time_series.iloc[-look_a_head:], signal_threshold=signal_threshold, use_log=use_log)

        # If the time-series is not stationary or threshold not reach, return None
        if signal is None:
            return None
        
        # Set the strategy weight
        signal[f"{time_series.name} weight"] = 1

        # Return the trading signal
        return signal

    ###############################################################################################
    # Backtest
    ###############################################################################################

    def backtest(
                    self,
                    data: pd.DataFrame,
                    window_length: int=100,
                    exit_threshold: float=0.,
                    cols: Optional[List[str]]=["close"],
                    use_log: bool=False,
                    look_a_head: int=1,
                    frac: Optional[int]=None,
                    **kwargs,
                ) -> pd.Series:
        """
        Perform a mean-reversion strategy based on the Hurst exponent.

        Parameters:
        - data:             A numpy array or pandas DataFrame containing the time series data.
        - exit_threshold:   The z-score threshold for exiting a trade (default 0.).
        - window_length:    The length of the rolling window (default 100).
        - cols:             The column to use for the trading signal (default "close")
        - kwargs:           Additional keyword arguments to pass to the hurst_signal function.

        Returns:
        - signal: A pandas Series containing the trading signals.
        """

        # Set the time scales if needed based on the look back (window_length)
        if not frac is None and isinstance(frac, int):
            self.minimal_time_scale = np.maximum(window_length // frac, 10)
            self.maximal_time_scale = np.maximum(window_length // (frac // 50), 100)

        x, cols = self.get_data(data, cols=cols, use_log=use_log)
        cols_w  = self.get_cols_weight(cols)

        # Perform the Johansen test
        signal = self.signal(x.iloc[:window_length+look_a_head], look_a_head=look_a_head, **kwargs)

        if signal is None:
            return None

        # Get the index of the minimum z-score
        z_score, _    = self.get_z_score(x.iloc[window_length+1:], signal.iloc[0], cols_w=cols_w)
    
        # Calculate the exit signal
        signal["entry"]  = data.index[window_length + 1:window_length + 1 + len(signal)]
        if signal["signal"].iloc[0] == 0:
            signal.loc[signal.index[0], "exit"] = signal.loc[signal.index[0], "entry"]
        for i, index in enumerate(signal.index):
            if signal["signal"][index] != 0 and not (self.no_overlap and i > 0 and signal["exit"][signal.index[i-1]] > signal["entry"][index]):
                j   = np.where(signal["signal"][index]*z_score[i:] > signal["signal"][index]*exit_threshold)[0]
                
                index_0 = np.minimum(int(signal["exit"][index] + window_length + 1 + i), len(data)-1) if len(j)==0 else window_length + 1 + i + np.minimum(j[0], signal["exit"][index])
                signal.loc[i, "exit"]   = data.index[index_0]
            elif i > 0:
                signal.loc[index, "signal"]   = 0
                signal.loc[index, "exit"]     = signal["exit"][signal.index[i-1]]

        return signal
    
#######################################################################################################################

class PairsMeanReversion(MeanReversion):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    ###############################################################################################
    """ Data Functions """
    ###############################################################################################

    def __str__(self) -> str:
        return "PairsMeanReversion"

    def get_data(self, data: pd.DataFrame, cols: Optional[List[str]]=None, use_log: bool=False) -> List[str]:
        """
        Get the data to use for the trading signal.

        Parameters:
        - data: A pandas DataFrame containing the time series data.
        - cols: The columns to use for the trading signal (default all columns).

        Returns:
        - A list containing the columns to use for the trading signal.
        """

        # Get the columns to use
        if cols is None:
            cols = data.columns.to_list()
        
        x   = data[cols[:2]] if not use_log else np.log(data[cols[:2]])

        return x, cols[:2]

    #########################################################################################################
    """ Signal Functions """
    #########################################################################################################

    def signal(
                self,
                time_series: pd.DataFrame,
                signal_threshold: float=1.,
                use_log: bool=False,
                look_a_head: int=1,
            ) -> Optional[pd.Series]:
        """
        Get a trading signal based on the Hurst exponent of a given time series.

        Parameters:
        - time_series:          A pandas DataFrame containing the time series data.
        - signale_threshold:    The threshold for the trading signal (default 1)
        - use_log:              Whether to use the log of the time series (default False)

        Returns:
        - A pandas Series containing the signal, portfolio, and residual mean, standard deviation and half-time
        """

        # Use log data for the OLS
        if use_log:
            x, y    = np.log(time_series[time_series.columns[0]]), np.log(time_series[time_series.columns[1]])
        else:
            x, y    = time_series[time_series.columns[0]], time_series[time_series.columns[1]]

        # Perform the OLS on a pair of data
        alpha, beta = get_ols_results(x[:-look_a_head], y[:-look_a_head], get_resid=False)

        # Normalize Portfolio
        beta_1  = 1. / (1 + np.abs(beta))
        beta_2  = -1.*beta_1 * beta
        alpha   = alpha * beta_1

        # Get the residuals
        resid   = (beta_1*y + beta_2*x - alpha).to_numpy()
        
        # Test the stationarity of the time series
        para    = self.test_stationarity(resid[:-look_a_head])
        if para is None:
            return None

        # Normalize the mean
        para["mean"]    += alpha

        # Get the trading signal
        signal  = self.get_signal(para, resid[-look_a_head:], signal_threshold=signal_threshold)
        if signal is None:
            return None

        signal[f"{time_series.columns[0]} weight"] = beta_2
        signal[f"{time_series.columns[1]} weight"] = beta_1

        # Return the trading signal
        return signal
    
#######################################################################################################################

class MultiMeanReversion(MeanReversion):

    def __init__(self, johansen_level=0, **kwargs):
        """
        Perform the Johansen test of cointegration on a given dataset.

        Parameters:
        - johansen_level:   The confidence level for the critical values 0 for 90%, 1 for 95%, 2 for 99% (default 2)
        - kwargs:           Additional keyword arguments to pass to the test
        """

        self.johansen_level = johansen_level

        super().__init__(**kwargs)

    ###############################################################################################
    """ Data Functions """
    ###############################################################################################

    def __str__(self) -> str:
        return "MultiMeanReversion"

    def get_data(self, data: pd.DataFrame, cols: Optional[List[str]]=None, use_log: bool=False) -> List[str]:
        """
        Get the data to use for the trading signal.

        Parameters:
        - data: A pandas DataFrame containing the time series data.
        - cols: The columns to use for the trading signal (default all columns).

        Returns:
        - A list containing the columns to use for the trading signal.
        """

        # Get the columns to use
        if cols is None:
            cols = data.columns.to_list()
        
        x   = data[cols] if not use_log else np.log(data[cols])

        return x, cols
    
    #########################################################################################################
    """ Signal Functions """
    #########################################################################################################

    def signal(
                self,
                time_series: pd.DataFrame,
                signal_threshold: float=1.,
                use_log: bool=False,
                look_a_head: int=1,
            ) -> Optional[pd.Series]:
        """
        Get a trading signal based on the Hurst exponent of a given time series.

        Parameters:
        - time_series:          A pandas DataFrame containing the time series data.
        - signale_threshold:    The threshold for the trading signal (default 1)
        - use_log:              Whether to use the log of the time series (default False)

        Returns:
        - A pandas Series containing the signal, portfolio, and residual mean, standard deviation and half-time
        """

        x   = time_series if not use_log else np.log(time_series)

        # Perform the Johansen test
        try:
            result = coint_johansen(x.iloc[:-look_a_head], det_order=0, k_ar_diff=1)
        except:
            return None

        # Determine the rank of the cointegration matrix
        rank_trace  = np.sum(result.lr1 > result.cvt[:, self.johansen_level])
        rank_eig    = np.sum(result.lr2 > result.cvm[:, self.johansen_level])
        rank        = min(rank_trace, rank_eig)

        if rank == 0:
            return None

        # Extract the Johansen portfolios by normalizing the eigenvectors
        johansen_portfolios = result.evec[:, :rank].T #/np.sum(np.abs(result.evec[:, :rank]), axis=0)
        johansen_portfolios /= np.sum(np.abs(johansen_portfolios), axis=1)

        # Calculate the spread & keep the portfolio with maximal z-score
        para, portfolio, spread = None, None, None
        for j in johansen_portfolios:
            # Normalize Portfolio
            j   /= np.sum(np.abs(j))

            # Calculate the spread
            spread_ = np.dot(x, j)
            para_   = self.test_stationarity(spread_[:-look_a_head])
            if para_ is None:
                continue
            
            if para is None or para_["half-life"] < para["half-life"]:
                para, portfolio, spread = para_, j, spread_[-look_a_head:]

        # Get the trading signal
        signal  = self.get_signal(para, spread, signal_threshold=signal_threshold)
        if signal is None:
            return None
        
        # Set the strategy weight
        for i, c in enumerate(time_series.columns):
            signal[f"{c} weight"] = portfolio[i]

        return signal

#######################################################################################################################
""" Main Function """
#######################################################################################################################

if __name__ == "__main__":

    # Generate AR(1) process data
    np.random.seed(10)

    n = 1000

    alpha   = -1.*np.array([0.05, 0.1])
    theta   = 2.*np.pi*np.random.uniform()

    A   = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    A   = np.eye(2) + A.dot(np.diag(alpha)).dot(A.T)

    epsilon = np.random.normal(0, 1, n)
    x = np.zeros((n, 2))
    for i in range(1, n):
        x[i] = A.dot(x[i-1]) + epsilon[i]
    
    # Create a DataFrame with the data
    data = pd.DataFrame(x, columns=["x", "y"])
    
    window_length   = 200
    look_a_head     = 10
    step            = 5

    # Run the backtest
    signal  = MeanReversion().full_backtest(data[["x"]], window_length=window_length, look_a_head=look_a_head, step=step, exit_threshold=0.5, cols=["x"])
    print(signal.dropna())
    signal  = PairsMeanReversion().full_backtest(data, window_length=window_length, look_a_head=look_a_head, step=step, exit_threshold=0.5, cols=["x", "y"])
    print(signal.dropna())
    signal  = MultiMeanReversion().full_backtest(data, window_length=window_length, look_a_head=look_a_head, step=step, exit_threshold=0.5, cols=["x", "y"])
    print(signal.dropna())