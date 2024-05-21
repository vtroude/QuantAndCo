import numpy    as np
import pandas   as pd

from typing     import Union, List

from statsmodels.tsa.stattools  import adfuller, kpss

from Research.statistical_tests import HurstExponent, get_half_life

#######################################################################################################################

def test_stationarity(time_series: pd.Series, confidence_level: float=0.05) -> pd.Series:
    """
    Test the stationarity of a given time series.

    Parameters:
    - time_series:      A pandas Series containing the time series data.
    - confidence_level: The confidence level for the ADF & KPSS test (default 0.05)

    Returns:
    - A pandas Series containing the Hurst exponent, mean, standard deviation and the characteristic time of the mean-reversion
    """


    # Check if the time-series is stationary
    if adfuller(time_series)[1] > confidence_level or kpss(time_series)[1] < confidence_level:
        return pd.Series({"hurst": None, "mean": None, "std": None, "char_time": None})

    # Calculate the fractal dimension using Hurst exponent
    hurst_exponent = HurstExponent(time_series.to_numpy())

    # Estimate the time-scale over which mean-reversion occurs
    time_scale = get_half_life(time_series)

    # Return the hurst exponent, mean, standard deviation and the characteristic time
    return pd.Series({"hurst": hurst_exponent, "mean": np.mean(time_series), "std": np.std(time_series), "char_time": time_scale})

#######################################################################################################################

def hurst_signal(
                    time_series: pd.DataFrame,
                    signal_threshold: float=1.,
                    hurst_threshold: float=0.5,
                    time_scale: int=100,
                    **kwargs
                ) -> Union[pd.Series, None]:
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
    hurst   = test_stationarity(time_series, **kwargs)

    # Check if the time-series is stationary
    if hurst["hurst"] is None or (hurst["hurst"] > hurst_threshold and time_scale <= hurst["char_time"]):
        return None
    
    # Calculate the z-score and the trading signal
    z_score = (time_series.iloc[-1] - hurst["mean"]) / hurst["std"]
    signal  = -1.*np.sign(z_score) if abs(z_score) > signal_threshold else 0
    exit    = int(hurst["char_time"]*3)

    # Return the trading signal
    return pd.Series({"signal": signal, "exit": exit, "weight": 1, "mean": hurst["mean"], "std": hurst["std"]})

#######################################################################################################################

def hurst_entry_exit(
                        data: pd.DataFrame,
                        window_length: int=100,
                        exit_threshold: float=0.,
                        cols: List[str]=["close"],
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

    cols_w  = f'{cols[0]} weight'

    # Perform the Johansen test
    signal = hurst_signal(data[cols[0]].iloc[:window_length], time_scale=window_length, **kwargs)

    if signal is None:
        return pd.Series({c: None for c in ["signal", "exit", cols_w]})

    signal  = signal.rename({"weight": cols_w})
    print(signal)

    # Get the index of the minimum z-score
    index_0 = int(np.minimum(window_length + signal["exit"], len(data)))
    z_score = (data[cols[0]].iloc[window_length:window_length+index_0] - signal["mean"]) / signal["std"]
 
    # Calculate the exit signal
    j   = np.where(signal["signal"]*z_score > exit_threshold)[0]

    signal["exit"]  = data.index[index_0-1]
    if len(j) > 0:
        signal["exit"]  = data.index[window_length + j[0]]

    return signal[[cols_w, "signal", "exit"]]

#######################################################################################################################

if __name__ == "__main__":
    from Strategy.utils import get_strategy

    n   = 1000
    a   = 0.1
    # Generate a stationary mean-reverting time-series
    np.random.seed(102)
    data = pd.Series(np.random.normal(0, 1, n), name="close")
    for i in range(n-1):
        data[i+1]   += (1. - a)*data[i]

    data    = pd.DataFrame(data)

    # Get the trading signals
    signal = get_strategy(data, hurst_entry_exit)

    # Print the trading signals
    print(signal[signal["signal"] == 1])