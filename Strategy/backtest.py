import os
#import sys

import numpy    as np
import pandas   as pd
import pylab    as pl

from typing     import Tuple

from DataPipeline.get_data                  import get_data_and_bars
from DataPipeline.technicals_indicators     import TechnicalIndicators
from Model.Classifier.randomforest_barrier  import get_model

#sys.path.append("/home/virgile/Desktop/General/Packages/")
#from Indicators.best_gains_worst_losses import best_gains_worst_losses


#######################################################################################################################

def back_test_thresh(price: pd.DataFrame, n_points: int, thresh: float) -> pd.Series:
    """
    Backtest a signal

    Input:
        - price:    Close price and signal time-series
        - n_points: time horizon in number of data points
        - thresh:   signal threshold i.e. send anorder when |signal| > thresh
    
    Return:
        - Wealth:   Strategy portfolio value
    """

    ###############################################################################################
    """ Initialize """
    ###############################################################################################

    i0              = 0     # Start at index 0
    price["Wealth"] = 1.    # Initialize Wealth / Portfolio value

    ###############################################################################################
    """ Apply strategy """
    ###############################################################################################

    # Loop over price time-series
    while i0 < len(price) - n_points:
        # If |signal| > thresh take Long or Short position i.e. sign of signal
        # Otherwise do not take position
        position    = np.sign(price["signal"].iloc[i0]) if np.abs(price["signal"].iloc[i0]) >= thresh else 0

        # If take position
        if position != 0:

            #############################################################################
            """ Get hitting time """
            #############################################################################

            p       = price.iloc[i0:i0+n_points]    # Get price from now to time horizon
            p_pos   = p[p["Close"] >= p["take"]]    # Measure price > positive threshold
            p_neg   = p[p["Close"] <= p["stop"]]    # Measure price < negative threshold
            t_end   = p.index[-1]
            
            # If hit a bar take the first one
            if len(p_pos) > 0 and len(p_neg) > 0:
                t_end   = np.minimum(p_pos.index[0], p_neg.index[0])
            elif len(p_pos) > 0:
                t_end   = p_pos.index[0]
            elif len(p_neg) > 0:
                t_end   = p_neg.index[0]
            
            #############################################################################
            """ Measure return """
            #############################################################################

            # Measure the portfolio value from now to the hitting time
            price.loc[p.index[0]:t_end, "Wealth"]   = (1. + position*(p["Close"][:t_end] - p["Close"].iloc[0])/p["Close"].iloc[0])*price.loc[p.index[0], "Wealth"]
            # Set the all the value after the hitting time with the last portfolio value
            price.loc[t_end:, "Wealth"] = price["Wealth"][t_end]

            # Set the index at the last action
            i0  += len(p.loc[:t_end])
        else:
            # Update index
            i0  += 1
    
    # Return Portfolio Value
    return price["Wealth"]

#######################################################################################################################

def backtest(symbol: str, date_test: pd.DatetimeIndex, thres: float, n_points: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtest a RF strategy for a given volatility multiplier (thres) and a given horizon (n_points) over different signal threshold

    Input:
        - symbol:   Asset symbol e.g. 'BTCUSD'
        - thres:    Threshold such that we defined P_{+/-} = P_t*exp(mean*n_points +/- thres*volatility*\sqrt{n_points})
        - n_points: We are searching that if the price will hit a bar in the interval [t, t+n_points]
    
    Return:
        - price:    Price time-series
        - wealth:   Wealth of the strategy for different signal threshold
    """


    ###############################################################################################
    """Initialize"""
    ###############################################################################################

    # Load trained model
    hitting     = get_model("hitting", symbol, date_test, thres, n_points)
    direction   = get_model("direction", symbol, date_test, thres, n_points)

    # Get data (features) to do prediction
    data, price = get_data_and_bars(symbol, interval, date1, date2, thres=thres, n_points=n_points)
    data, price = data[data.index > date_test], price[price.index > date_test]

    # Build signal
    price["signal"] = hitting.predict_proba(data.to_numpy())[:,-1]
    price["signal"] *= 2.*(direction.predict_proba(data.to_numpy())[:,-1] - 0.5)

    ###############################################################################################
    """Backtest Strategy"""
    ###############################################################################################

    # Get different signal threshold
    strategy_thresh = [0.1, 0.3, 0.5, 0.7, 0.9, 1.]
    # Loop over all signal threshold and get the wealth time series
    wealth  = pd.concat([back_test_thresh(price, n_points, thresh=th) for th in strategy_thresh], axis=1)
    # Rename columns
    wealth.columns  = [f"$\epsilon={np.round(th,2)}$" for th in strategy_thresh]

    # Return price and wealth data
    return price, wealth

#######################################################################################################################

def save_plot_strategy(price: pd.DataFrame, wealth: pd.DataFrame, model_name: str, labels: pd.Series) -> None:
    """
    Plot B&H Vs Portfolio Value / Wealth

    Input:
        - price:        Price time-series
        - wealth:       Wealth / Portfolio Value
        - model_name:   model or strategy name
        - labels:       extra data e.g. symbol, date, etc...
    """

    fig, ax = pl.subplots(1, 1, figsize=(24,14))

    ax.set_title(f"Backtest {labels['symbol']} from {labels['date start']} to {labels['date end']} ($k={labels['thresh']}$ $n={labels['n points']}$)", fontsize=20)

    wealth.plot(ax=ax, grid=True, logy=True)
    (price["Close"]/price["Close"].iloc[0]).plot(ax=ax, color="k", grid=True, logy=True)

    ax.set_ylabel("Price", fontsize=18)
    ax.set_xlabel("Date", fontsize=18)
    ax.legend(fontsize=26)

    fig.subplots_adjust(left=0.055, bottom=0.093, right=0.962, top=0.92)
    fig.savefig(f"Figure/Backtest/{model_name}_{labels['symbol']}_{labels['date start']}_{labels['date end']}_k={labels['thresh']}_n={labels['n points']}.png")

#######################################################################################################################

def get_backtest_stats(price: pd.Series, model_name: str, to_append: pd.Series) -> None:
    """
    Measure Backtest success

    Input:
        -price:         price / portfolio value of an asset / strategy
        -model_name:    model or trading strategy name
        -to_append:     additional data to append e.g. symbol, date etc...
    """

    success = pd.Series()

    ###############################################################################################
    """ Portfolio / Price return """
    ###############################################################################################

    price_last      = price.shift(1)
    index           = price.index[price != price_last]
    price_return    = price/price_last - 1.

    traded_return   = price_return.loc[index]

    ###############################################################################################
    """ Get Statistics """
    ###############################################################################################

    success["mean"]     = traded_return.mean()
    success["std"]      = traded_return.std()
    success["skew"]     = traded_return.skew()
    success["kurt"]     = traded_return.kurtosis()
    success["VaR-5%"]   = traded_return.quantile(0.05)

    bgwl    = TechnicalIndicators().get_draw(price)
    success["MaxDrawup"]    = bgwl["cum log"].max()
    success["MaxDrawdown"]  = bgwl["cum log"].min()

    success["sharp"]    = success["mean"]/success["std"]
    success["trades"]   = np.round(len(index)/len(price)*100, 2)

    ###############################################################################################
    """ Save Data """
    ###############################################################################################

    success["model"]    = model_name
    for a in to_append.index:
        success[a]  = to_append[a]

    success = pd.DataFrame(success).transpose()

    if not os.path.exists("Score"):
        os.mkdir("Score")

    file_success  = f"Score/backtest.csv"
    if not os.path.exists(file_success):
        success.to_csv(file_success, index=False)
    else:
        success.to_csv(file_success, index=False, header=False, mode="a")

#######################################################################################################################

if __name__ == "__main__":

    ###############################################################################################
    """ Set Configuration """
    ###############################################################################################

    date1       = "2021-04-05-23:44:12"
    date2       = "2024-04-04-23:44:12"
    date_test   = pd.to_datetime("2024-01-01 00:00:00")

    symbol      = 'BTCUSDT'
    interval    = ['1m', '1h', '1d', '1w']

    # Multiplier such that the hitting bars are at P_{+/-} = P_t*exp(mu +/-  thres*volatility)
    thres       = [0.5, 0.8, 1., 1.2, 1.5, 1.8, 2., 2.5, 3.]
    # Time horizon over which we are estimating the first s=t,,t+1,...,t+n_points such that P_s > P_+ or P_s < P_-
    n_points    = [20, 40, 60, 100, 200, 400, 600]

    ###############################################################################################
    """ Backtest & Evaluate """
    ###############################################################################################

    # Backtest RF Classifier Trading strategy
    for th in thres:
        for n in n_points:
            try:
                # Get backtest over different signal threshold
                price, wealth   = backtest(symbol, date_test, th, n)
            except:
                # If any error return None
                wealth = None
            
            # If the Wealth is not None, evaluate trading success
            if not wealth is None:
                # Prepare data to save strategy success for a given symbol, etc...
                labels  = {"symbol": symbol, "date start": price.index[0], "date end": price.index[-1], "thresh": th, "n points": n}
                # Loop over signal threshold
                for c in wealth.columns:
                    labels_ = pd.Series(dict(**labels, **{"eps": c.replace("$", "").split("=")[-1]}))   # Make extra labels
                    get_backtest_stats(wealth[c], "rf classifier", labels_)                             # Compute and save strategy success
                
                # Make Backtest Portfolio Value Vs B&H time-series
                save_plot_strategy(price, wealth, "rf classifier", labels)

    # Save B&H strategy success
    labels  = pd.Series({"symbol": symbol, "date start": price.index[0], "date end": price.index[-1]})
    get_backtest_stats(price["Close"], "B&H", labels)

            
