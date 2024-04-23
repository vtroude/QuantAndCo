

import numpy    as np
import pandas   as pd

from typing import List, Union, Tuple

from DataPipeline.get_data                  import get_data_and_bars
from Model.Classifier.randomforest_barrier  import get_model

class RF_strategy():
    
    def __init__(self, symbol: str, interval: str, date_test: pd.DatetimeIndex, thres: float, n_points: int) -> None:
        """
        Get a RF strategy for a given volatility multiplier (thres) and a given horizon (n_points) over different signal threshold

        Input:
            - symbol:       Asset symbol e.g. 'BTCUSD'
            - interval:     Candlestick time interval e.g. '1m'
            - date_test:    date from which we compute the signal to backtest
            - thres:        Threshold such that we defined P_{+/-} = P_t*exp(mean*n_points +/- thres*volatility*\sqrt{n_points})
            - n_points:     We are searching that if the price will hit a bar in the interval [t, t+n_points]
        
        Return:
            - price:    Price time-series with (portfolio) weights in [-1,1]
        """

        ###############################################################################################
        """ Assert & Initialize Hyper-Parameters """
        ###############################################################################################

        # Assert Type
        assert isinstance(symbol, str), "'symbol' should be str"
        assert isinstance(interval, str), "'interval' should str"
        assert isinstance(date_test, str), "'date_test' should be DatetimeIndex"
        assert isinstance(thres, float) and thres > 0, "'thres' should be postive float"
        assert isinstance(n_points, int) and n_points > 0, "'n_points' should be a positive integer"

        # save strategy symbol
        self.symbol     = symbol        # Symbol over which models are trained
        self.interval   = interval      # candlestick interval over which models are trained
        self.date_test  = date_test     # date up to which the model is trained
        self.thres      = thres         # threshold to define the take profit and stop loss bars
        self.n_points   = n_points      # number of points to define the stop time

        ###############################################################################################
        """ Get Model """
        ###############################################################################################

        # Load trained model
        self.hitting    = get_model("hitting", symbol, date_test, thres, n_points)      # model to predict if we are hitting a bar
        self.direction  = get_model("direction", symbol, date_test, thres, n_points)    # model to predict which bar we will hit

    #########################################################################################################

    @staticmethod
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

    #########################################################################################################

    def prepare_backtest(self, date1, date2) -> None:
        """
        Prepare backtest by getting the input data, the bars and the price time-series

        Input:
            - date1:    Date from which the data has been gathered in %Y-%m-%d-%H-%M-%S
            - date2:    Date to which the data has been gathered in %Y-%m-%d-%H-%M-%S
        """

        ###############################################################################################
        """ Get Data """
        ###############################################################################################

        # Get data (features) to do prediction
        self.data, self.price   = get_data_and_bars(self.symbol, self.interval, date1, date2, thres=self.thres, n_points=self.n_points)
        self.data, self.price   = self.data[self.data.index > self.date_test], self.price[self.price.index > self.date_test]

    #########################################################################################################

    def get_weight(self, data: pd.DataFrame) -> pd.Series:
        """
        Get a RF weight for a given volatility multiplier (thres) and a given horizon (n_points)

        Input:
            - data: Input to give to the model to make a prediction
        
        Return:
            - weight:       (Portfolio) weights in [-1,1]
        """

        ###############################################################################################
        """ Get Weight """
        ###############################################################################################

        # Build weight from -1 to 1
        weight  = self.hitting.predict_proba(data.to_numpy())[:,-1]
        weight  *= 2.*(self.direction.predict_proba(data.to_numpy())[:,-1] - 0.5)

        # Return weight
        return weight

    #########################################################################################################

    def get_signal(self, price: pd.DataFrame, weight: pd.Series, signal_thresh: float) -> pd.Series:
        """
        Get RF Bar Signal

        Input:
            - price:    Close price and signal time-series
            - weight:   (Portfolio / Signal) weight
        
        Return:
            - Signal:   Strategy portfolio value
        """

        ###############################################################################################
        """ Initialize """
        ###############################################################################################

        price["signal"] = 1.                                                # Initialize signal
        weight          = weight[np.abs(weight) >= signal_thresh]           # Get weight above signal threshold
        i0              = price[price.index <= weight.index[0]].iloc[0]     # Initialize first trade
        position        = np.sign(weight.index[0])                          # First trade position

        ###############################################################################################
        """ Apply Weight """
        ###############################################################################################

        # Loop over price time-series
        while i0 < len(price):
            #############################################################################
            """ Update Position """
            #############################################################################

            # Get the time at which the first bar (take profit / stop loss) is hit from now to now + n_points
            t_end   = self.get_hitting_time(price.iloc[i0:i0+self.n_points])
            # Set position from last trade to new trade
            price.loc[price.index[i0]:t_end, "signal"]  = position
            # Get next trade time
            if weight.index[-1] > t_end:
                i0  = price[price.index <= weight[t_end:].index[1]].iloc[0]
            else:
                i0  = len(price)
        
        # Return signal
        return price["signal"]

    #########################################################################################################

    def get_backtest_signal(self, date1: str, date2: str, signal_thresh: float) -> pd.Series:

        # Load data from database
        if not self.price:
            self.prepare_backtest(date1, date2)
        
        weight  = self.get_weight(self.data)
        signal  = self.get_signal(self.price, weight, signal_thresh)

        return signal