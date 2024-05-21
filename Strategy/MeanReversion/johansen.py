import numpy    as np
import pandas   as pd
import pylab    as pl

from typing import Union, Tuple, List, Optional

from statsmodels.tsa.stattools      import adfuller, kpss
from statsmodels.tsa.arima.model    import ARIMA
from statsmodels.tsa.vector_ar.vecm import coint_johansen

#######################################################################################################################


def fit_and_detrend(data: pd.Series,  **kwargs) -> Tuple[np.ndarray, float, float, float]:
    """
    Measure the mean-reverting rate of an AR(1) process.

    Parameters:
    - data:     A numpy array or pandas Series containing the time series data.
    - kwargs:   Additional keyword arguments to pass to the ARIMA model.

    Returns:
    - z_score:      The z-score of the data.
    - mean:         The mean of the data.
    - std:          The standard deviation of the data.
    - char_time:    The characteristic mean-reverting time of the data.
    """

    # Fit an AR(1) model
    model       = ARIMA(data, **kwargs)
    model_fit   = model.fit()

    # Extract the AR(1) coefficient
    ar_coef = model_fit.arparams[0]

    # Calculate the characteristic mean-reversing time
    char_time   = -1 / np.log(abs(ar_coef))
    
    # Calculate the z-score
    mean, std   = np.mean(data), np.std(data)
    z_score     = (data - mean) / std

    # Return the results
    return z_score, mean, std, char_time

#######################################################################################################################

def johansen_portfolios(
                            data: pd.DataFrame,
                            johansen_level: int=2,
                            confidence_level: float=0.05,
                        ) -> Union[None, pd.DataFrame]:
    """
    Perform the Johansen test of cointegration on a given dataset.

    Parameters:
    - data:             A numpy array or pandas DataFrame containing the time series data.
    - johansen_level:   The confidence level for the critical values 0 for 90%, 1 for 95%, 2 for 99% (default 2)
    - confidence_level: The confidence level for the ADF & KPSS test (default 0.05)

    Returns:
    - result: A pandas DataFrame containing the z-score, mean, std, characteristic time and the johansen portfolios.
    """

    # Perform the Johansen test
    result = coint_johansen(data, det_order=0, k_ar_diff=1)

    # Determine the rank of the cointegration matrix
    rank_trace  = np.sum(result.lr1 > result.cvt[:, johansen_level])
    rank_eig    = np.sum(result.lr2 > result.cvm[:, johansen_level])
    rank        = min(rank_trace, rank_eig)

    # Extract the Johansen portfolios by normalizing the eigenvectors
    johansen_portfolios = result.evec[:, :rank]/np.sum(np.abs(result.evec[:, :rank]), axis=0)
    
    # Calculate the spread
    spread  = np.dot(data, johansen_portfolios)

    # Perform the ADF and KPSS tests & keep only the cointegrated portfolios
    keep = np.where([(adfuller(spread[:, i], regression="c")[1] < confidence_level) & (kpss(spread[:, i], regression="c")[1] > confidence_level) for i in range(rank)])[0]
    if len (keep) == 0:
        return None

    # Keep only the cointegrated portfolios
    spread              = spread[:, keep]
    johansen_portfolios = johansen_portfolios[:, keep]

    # Fit and detrend the spread
    mean, std, char_time    = np.zeros(spread.shape[1]), np.zeros(spread.shape[1]), np.zeros(spread.shape[1])
    for i in range(spread.shape[1]):
        spread[:, i], mean[i], std[i], char_time[i] = fit_and_detrend(spread[:, i], order=(1, 0, 0), trend="c")

    # Create a DataFrame with the results
    johansen_portfolios = pd.DataFrame(np.concatenate((spread[-1].reshape(-1,1), mean.reshape(-1,1), std.reshape(-1,1), char_time.reshape(-1,1),  johansen_portfolios.T), axis=1),
                                       columns= ["z-score", "mean", "std", "char time"] + data.columns.to_list(), index=range(spread.shape[1]))

    # Return the results
    return johansen_portfolios

#######################################################################################################################

def johansen_signal(
                        data: pd.DataFrame,
                        entry_threshold: float=1.,
                        time_scale: int=100,
                        **kwargs,
                    ) -> pd.Series:
    """
    Perform the Johansen test of cointegration on a given dataset & return the trading signals.

    Parameters:
    - data:             A numpy array or pandas DataFrame containing the time series data.
    - entry_threshold:  The z-score threshold for entering a trade (default 1.).
    - time_scale:       The time scale over which mean-reversion occurs (default 100) i.e. time_scale >> mean-reverting time
    - kwargs:           Additional keyword arguments to pass to the johansen_portfolios function.

    Returns:
    - signal: A pandas Series containing the trading signals.
    """

    # Perform the Johansen test
    portfolios  = johansen_portfolios(data, **kwargs)

    if portfolios is None or time_scale <= portfolios["char_time"]:
        return pd.Series({c: None for c in ["signal", "exit"] + [f'{c} weight'  for c in data.columns.to_list()]})

    # Get the index of the minimum z-score
    index   = np.where(np.abs(portfolios["z-score"]) == np.abs(portfolios["z-score"]).max())[0][0]
    long    = -1.*np.sign(portfolios["z-score"][index])
    
    signal  = portfolios[data.columns].loc[index]
    # Add ' weight' at the end of each signal index
    signal.index    = [f'{c} weight' for c in signal.index]
    
    signal["mean"]      = portfolios["mean"][index]
    signal["std"]       = portfolios["std"][index]
    signal["signal"]    = None if long*portfolios["z-score"][index] > -entry_threshold else long
    signal["exit"]      = int(portfolios["char time"][index] * 3)

    return signal

#######################################################################################################################

def johansen_entry_exit(
                            data: pd.DataFrame,
                            window_length: int=100,
                            exit_threshold: float=0.,
                            cols: List[str]=["close"],
                            **kwargs,
                        ) -> pd.Series:
    """
    Perform a mean-reversion strategy based on the Johansen test of cointegration.

    Parameters:
    - data:             A numpy array or pandas DataFrame containing the time series data.
    - exit_threshold:   The z-score threshold for exiting a trade (default 0.).
    - window_length:    The length of the rolling window (default 100).
    - kwargs:           Additional keyword arguments to pass to the get_johansen_signal function.

    Returns:
    - signal: A pandas Series containing the trading signals.
    """

    cols_w    = [f'{c} weight'  for c in cols]

    # Perform the Johansen test
    signal = johansen_signal(data[cols].iloc[:window_length], **kwargs)

    if signal["signal"] is None:
        return signal[cols_w + ["signal", "exit"]]
    
    # Compute the z-score of the spread in real-time
    index_0 = int(np.minimum(window_length + signal["exit"], len(data)))
    z_score = data[cols].iloc[window_length:index_0].to_numpy().dot(signal[cols].to_numpy())
    z_score = (z_score  - signal["mean"]) / signal["std"]

    # Find the exit point
    j   = np.where(signal["signal"]*z_score > -1.*exit_threshold)[0]

    signal["exit"]  = data.index[index_0-1]
    if len(j) > 0:
        signal["exit"]  = data.index[window_length + j[0]]

    # Return the trading signals
    return signal[cols_w + ["signal", "exit"]]

#######################################################################################################################

if __name__ == "__main__":
    from Strategy.utils import get_strategy

    np.random.seed(102)

    T, N    = 1000, 10
    tau     = 100.
    # Generate some sample data
    data = np.zeros((T, N))

      
    # Generate a random orthogonal matrix
    Q = np.random.randn(N, N)
    Q, _ = np.linalg.qr(Q)

    A   = -1.*np.abs(np.random.randn(N))/tau
    B   = -1.*np.abs(np.random.randn(N))/tau

    A   = np.dot(Q.T, np.dot(np.diag(A), Q))
    B   = np.dot(Q.T, np.dot(np.diag(B), Q))
    
    beta_0      = 1.

    for i in range(1, T-1):
        data[i+1]   = data[i] + np.dot(A, data[i]) + np.dot(B, data[i] - data[i-1]) + beta_0 + np.random.randn(N)

    data    = pd.DataFrame(data)

    # Test the johansen_signal function
    signal  = get_strategy(data, johansen_entry_exit)
    print(signal[signal["signal"] == 1])
    