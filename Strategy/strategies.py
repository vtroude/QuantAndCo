import numpy    as np
import pandas   as pd

from typing import List, Union, Tuple

from DataPipeline.get_data                  import get_data_and_bars, get_all_data, get_price_data
from Model.Classifier.randomforest_barrier  import get_model

#######################################################################################################################

def get_hitting_time(price: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Get the first hitting time among two bars i.e. take & stop s.t. take >= Close >= stop

    Input:
        - price:    Close price + take and stop data to measure the first hitting time if any
    
    Return:
        - t_end:    Hitting time or last time
    """

    #############################################################################
    """ Get hitting time """
    #############################################################################

    p_pos   = price[price["Close"] >= price["take"]]    # Measure price > positive threshold
    p_neg   = price[price["Close"] <= price["stop"]]    # Measure price < negative threshold
    t_end   = price.index[-1]
    
    # If hit a bar take the first one
    if len(p_pos) > 0 and len(p_neg) > 0:
        t_end   = np.minimum(p_pos.index[0], p_neg.index[0])
    elif len(p_pos) > 0:
        t_end   = p_pos.index[0]
    elif len(p_neg) > 0:
        t_end   = p_neg.index[0]

    # Return hitting time or last time
    return t_end


#######################################################################################################################

def rf_bare_signal(
                    symbol: str,
                    date_test: pd.DatetimeIndex,
                    thres: float,
                    n_points: int,
                    interval: Union[List[str], str],
                    date1: str,
                    date2: str
                ) -> pd.DataFrame:
    """
    Get a RF strategy for a given volatility multiplier (thres) and a given horizon (n_points) over different signal threshold

    Input:
        - symbol:       Asset symbol e.g. 'BTCUSD'
        - date_test:    date from which we compute the signal to backtest
        - thres:        Threshold such that we defined P_{+/-} = P_t*exp(mean*n_points +/- thres*volatility*\sqrt{n_points})
        - n_points:     We are searching that if the price will hit a bar in the interval [t, t+n_points]
        - interval:     Candlestick time interval e.g. '1m'
        - date1:        Date from which the data has been gathered in %Y-%m-%d-%H-%M-%S
        - date2:        Date to which the data has been gathered in %Y-%m-%d-%H-%M-%S
    
    Return:
        - price:    Price time-series with (portfolio) weights in [-1,1]
    """


    ###############################################################################################
    """ Get Model """
    ###############################################################################################

    # Load trained model
    hitting     = get_model("hitting", symbol, date_test, thres, n_points)
    direction   = get_model("direction", symbol, date_test, thres, n_points)

    ###############################################################################################
    """ Get Data """
    ###############################################################################################

    # Get data (features) to do prediction
    data, price = get_data_and_bars(symbol, interval, date1, date2, thres=thres, n_points=n_points)
    data, price = data[data.index > date_test], price[price.index > date_test]

    ###############################################################################################
    """ Get Signal """
    ###############################################################################################

    # Build signal
    price["weight"] = hitting.predict_proba(data.to_numpy())[:,-1]
    price["weight"] *= 2.*(direction.predict_proba(data.to_numpy())[:,-1] - 0.5)

    # Return price + weight
    return price

#######################################################################################################################

def rf_bar_strategy(price: pd.DataFrame, n_points: int, thresh: float) -> pd.Series:
    """
    Get RF Bar Strategy

    Input:
        - price:    Close price and signal time-series
        - n_points: time horizon in number of data points
        - thresh:   signal threshold i.e. send anorder when |signal| > thresh
    
    Return:
        - Signal:   Strategy portfolio value
    """

    ###############################################################################################
    """ Initialize """
    ###############################################################################################

    price["signal"] = 1.    # Initialize signal
    weight          = price["weight"][np.abs(price["weight"]) >= thresh]
    i0              = price[price.index <= weight.index[0]].iloc[0]
    position        = np.sign(weight.index[0])

    ###############################################################################################
    """ Apply Weight """
    ###############################################################################################

    # Loop over price time-series
    while i0 < len(price):
        #############################################################################
        """ Update Position """
        #############################################################################

        # Get the time at which the first bar (take profit / stop loss) is hit from now to now + n_points
        t_end   = get_hitting_time(price.iloc[i0:i0+n_points])
        # Set position from last trade to new trade
        price.loc[price.index[i0]:t_end, "signal"]  = position
        # Get next trade time
        if weight.index[-1] > t_end:
            i0  = price[price.index <= weight[t_end:].index[1]].iloc[0]
        else:
            i0  = len(price)
    
    # Return signal
    return price["signal"]

#######################################################################################################################

def tsi_signal(
                symbol: str,
                date_test: pd.DatetimeIndex,
                interval: Union[List[str], str],
                date1: str,
                date2: str
                ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply the True Strength Index Strategy

    Input:
        - symbol:       Asset symbol e.g. 'BTCUSD'
        - date_test:    date from which we compute the signal to backtest
        - n_points:     We are searching that if the price will hit a bar in the interval [t, t+n_points]
        - interval:     Candlestick time interval e.g. '1m'
        - date1:        Date from which the data has been gathered in %Y-%m-%d-%H-%M-%S
        - date2:        Date to which the data has been gathered in %Y-%m-%d-%H-%M-%S
    
    Return:
        - price:    Price time-series with signals in [-1,1]
    """

    ###############################################################################################
    """ Get Data """
    ###############################################################################################

    data    = get_all_data(symbol, interval, date1, date2)
    price   = get_price_data(symbol, interval[0], date1, date2)

    ###############################################################################################
    """ Format Data """
    ###############################################################################################

    # Get data after date_test
    data    = data[data.index > date_test]
    # From [0,1] to [-1, 1] interval where -1 => Short and 1 => Buy
    data    = 2.*(data[[c for c in data.columns if "StochTSI" in c]].dropna() - 0.5)
    # Measure min / max of Stoch TSI among all time scale
    data["min"] = data.min(axis=1)
    data["max"] = data.max(axis=1)

    # Get price for all data index
    price   = price.loc[data.index]

    # Return min / max TSI and price
    return data[["min", "max"]], price

#######################################################################################################################

def tsi_strategy(
                    tsi: pd.DataFrame,
                    thresh: float

                ) -> pd.Series:
    
    """
    Extraction of trading signal from the rolling TSI over different time-scales.
    When the minimum / maximum over all time-scales reach thresh / -thresh, send a negative / positive signal (Short / Long)

    Input:
        - tsi:      Contains the minimum & maximum of the Stoch TSI among different time-scales
        - thresh:   Threshold above / below which to send signal
    
    Return:
        - signal:   Return signal
    """

    ###############################################################################################
    """ Initialization """
    ###############################################################################################

    # Initit Signal
    tsi["signal"]       = 0.
    # Get the TSI max / min, below / above the threshold
    tsi_max, tsi_min    = tsi["max"][tsi["max"] < -thresh], tsi["min"][tsi["min"] > thresh]
    
    # Initialize the first trade when the first signal is launched
    position    = 0.
    if tsi_min.index < tsi_max.index:
        position = 1.
        i0  = tsi[tsi.index <= tsi_min.index].iloc[-1]
    elif tsi_min.index > tsi_max.index:
        position = -1.
        i0  = tsi[tsi.index <= tsi_max.index].iloc[-1]
    
    ###############################################################################################
    """ Get Signal """
    ###############################################################################################

    # Loop over index, such that when until we reach the opposite threshold
    # (positive if we have a short position i.e. -1; negative if we have a long position i.e. +1)
    # keep the same position and switch to the opposite position i.e. position -> -1*postion;
    # once the new threshold is reached
    while i0 < len(tsi):
        if tsi_max.index[-1] >= tsi.index[i0] or tsi_min.index[-1] >= tsi.index[i0]:
            # Get next trading time
            t_end   = tsi_max.loc[tsi.index[i0]:].index[0] if position > 0 else tsi_min.loc[tsi.index[i0]:].index[0]
            # Set position from last trade to new trade
            tsi.loc[tsi.index[i0]:t_end, "signal"]  = position
            # Switch position
            position    *= -1.
            # Update the index of the last trade
            i0  = tsi[tsi.index <= t_end].iloc[-1]
        else:
            tsi.loc[tsi.index[i0]:, "signal"]  = position
            i0  = len(tsi)
    
    # Return signal
    return tsi["signal"]