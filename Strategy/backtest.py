import os
#import sys

import numpy    as np
import pandas   as pd
import matplotlib.pylab    as pl

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

def backtest_RSI_strategy(price_df: pd.DataFrame, rsi_df: pd.DataFrame, entry_long_thres: float, entry_short_thres:float, 
                          exit_long_thres: float, exit_short_thres: float, leverage=1) -> pd.DataFrame:
    """
    Backtest trading strategy based on Relative Strength Indicators (RSI).
    It is a mean-reverting strategy. Long signal is sent when price is below a certain threshold, and 
    conversely short signal is sent when price is above a certain threshold.

    Input:
        - price_df:    Close price time-series. Index must be timestamp.
        - rsi_df:      Time-series containing one or more Stochastic RSI indicators (with different spans). Must only contain Stochastic RSI columns + timestamp as index.
        - entry_long_thresh:   Between O and 1. When all indicators are below this threshold, long signal is sent.
        - entry_short_thres:   Between 0 and 1. When all indicators are above this threshold, short signal is sent.
        - exit_long_thres:     Between 0 and 1. When any indicator is above this threshold, if a long position is currently open, it will be closed on the next available bar.
        - exit_short_thres:    Between 0 and 1. When any indicator is below this threshold, if a short position is currently open, it will be closed on the next available bar.
        - leverage:            Float. Leverage used to amplify the positions.
    
    Return:
        - Wealth:   Strategy portfolio value


    TO DO:
    - Add commission and slippage (also for leverage)
    """

    ## First, we make sure timestamp is index in both price_df and rsi_df

    if price_df.index.name != 'timestamp':
        if 'timestamp' not in price_df.columns:
            raise ValueError('"timestamp" column missing in price_df')
        else:
            price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
            price_df.set_index('timestamp', inplace=True)

    if rsi_df.index.name != 'timestamp':
        if 'timestamp' not in rsi_df.columns:
            raise ValueError('"timestamp" column missing in rsi_df')
        else:
            rsi_df['timestamp'] = pd.to_datetime(rsi_df['timestamp'])
            rsi_df.set_index('timestamp', inplace=True)
    

    # 2. Combining price_df and rsi_df 

    df = pd.merge(price_df, rsi_df, right_index=True, left_index=True)

    # 3. We get rid of NaNs -> this ensures we will only start backtesting once all the indicators are ready

    df.dropna(subset=rsi_df.columns, inplace=True)

    #4. Signal generation

    def entry_signal(indicators_lst, entry_long_thres, entry_short_thres):
        """
        If all indicators are above entry_short_thres -> sends short signal (returns -1)
        If all indicators are below long_entry_thres -> sends long signal (returns 1)
        Else returns 0
        """
        if all(i > entry_short_thres for i in indicators_lst):
            return -1
        elif all(i < entry_long_thres for i in indicators_lst):
            return 1
        else: 
            return 0
    
    def exit_signal(position, indicators_lst, exit_long_thres, exit_short_thres):

        """
        If previous position is long (1), returns exit signal if any indicator is above exit_long_thres
        If previous position is short (-1), returns exit signal if any indicator is below exit_short_thres
        Else returns False.
        """

        exit_conditions = (position == 1 and any(i > exit_long_thres for i in indicators_lst)) or (position == -1 and any(i < exit_short_thres for i in indicators_lst))

        return exit_conditions

        #if position == 1:
        #    #condition = any(i > long_thres for i in indicators_lst)
        #    if any(i > exit_long_thres for i in indicators_lst):
        #        print(indicators_lst)
        #        print('>')
        #        print(exit_long_thres)
        #        return True
        
        #elif position == -1:
        #    if any(i < exit_short_thres for i in indicators_lst):
        #        print(indicators_lst)
        #        print('<')
        #        print(exit_short_thres)
        #        return True
        
        #else:
        #    return False

    def TP_TL():
        pass

    def update_wealth(previous_wealth, position, buy_and_hold_return, leverage):
        return previous_wealth * (1 + position * buy_and_hold_return * leverage)

    def calculate_trade_return(position, entry_price, exit_price, leverage, commission_fees=0., slippage_fees=0.):
        return position * (exit_price/entry_price - 1) * leverage



    # 5. Backtesting
    i0              = 1     # Start at index 1
    df["Wealth"] = 1.0    # Initialize Wealth / Portfolio value
    position = 0 #Initial position is 0 (Cash)
    df["Realized_Return"] = 0.0
    df['position'] = 0
    df['BuyHold_Return'] = df['Close'].pct_change()

    ###############################################################################################
    """ Apply strategy """
    ###############################################################################################

    # Loop over price time-series
    while i0 < len(df) - 1:
        ### Entry conditions: If all indicators are above or below
        ### Exit conditions: If any indicator is above or below

        rsi_indicators = df.iloc[i0][rsi_df.columns].values.tolist()
        wealth = df.loc[df.index[i0], "Wealth"]
        buy_and_hold_return = df.loc[df.index[i0], "BuyHold_Return"]

        if position == 0:
            position  = entry_signal(rsi_indicators, entry_long_thres, entry_short_thres)
            if position != 0:
                entry_price = df.iloc[i0+1]['Close'] #When entry signal is observed at i0, trade is entered at i0+1
                if position == 1:
                    trade = 'Long'
                else:
                    trade = 'Short'
                print(f'{trade} Trade entered at entry price: {entry_price}')

        if position != 0:
            signal_to_exit = exit_signal(position, rsi_indicators, exit_long_thres, exit_short_thres)
            if signal_to_exit == True:
                exit_price = df.iloc[i0+1]['Close'] #When exit signal is observed at i0, trade is exited at i0+1
                trade_return = calculate_trade_return(position, entry_price, exit_price, leverage)
                print(f'Trade exited at exit price: {exit_price}')
                print(f'Return of trade with leverage {leverage}: {trade_return}')
                df.loc[df.index[i0+1], "Realized_Return"] = trade_return

                position = 0

        
        df.loc[df.index[i0+1], "position"] = position #Position is updated at i0+1 based on signal observed at i0
        df.loc[df.index[i0+1], "Wealth"] = update_wealth(wealth, position, buy_and_hold_return, leverage)
        
        i0+=1
    
    if position != 0: #If a position is still open at last bar, close it
        exit_price = df.iloc[i0+1]['Close']
        trade_return = calculate_trade_return(position, entry_price, exit_price, leverage)
        position = 0
        df.loc[df.index[i0+1], "Realized_Return"] = trade_return
        df.loc[df.index[i0+1], "position"] = position #Position is updated at i0+1 based on signal observed at i0
        final_wealth = update_wealth(wealth, position, buy_and_hold_return, leverage)
        df.loc[df.index[i0+1], "Wealth"] = final_wealth
        print(f'Final wealth: {final_wealth}')

    n_bars = i0


    #df['position_change'] = df['position'].diff() #Position at T+1 minus Position at T


    return df












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

            
