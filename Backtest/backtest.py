import pandas as pd
from Strategy.rsi_strategy import RSIStrategy
from Backtest.metrics import backtest_metrics, save_plot_strategy
from DataPipeline.make_data import make_filename
from DataPipeline.technicals_indicators import TechnicalIndicators
import time
import os

def update_wealth(previous_wealth, position, buy_and_hold_return, leverage):
    return previous_wealth * (1 + position * buy_and_hold_return * leverage)

def calculate_trade_return(position, entry_price, exit_price, leverage, commission_fees=0., slippage_fees=0.):
    return position * (exit_price/entry_price - 1) * leverage

#1.Une fonction de stratégie qui génére les signaux en temps réel
#2.Une fonction de stratégie qui backtest les signaux

def Backtest_Strategy(df: pd.DataFrame, start_date: str=None, end_date: str=None, 
                      price_column: str = 'Close', leverage: float = 1.0)->pd.DataFrame:
    """
    This function takes as input a strategy dataframe which contains the price 
    of the asset and signals of the strategy, and returns the wealth of the portfolio.

    Input:
        - df:           Price and signal time-series. Index must be timestamp.
        df can contain both an entry signal (column "signal") and an exit signal (column "exit_signal"). 
        If there is no "exit_signal" column, the code assumes that an exit signal is simply the reverse 
        of an entry signal.

        - price_column: Price column used for orders. By default set to "Close". Orders are passed at t+1 while signals are generated at t.
        - leverage:     Float. Leverage used to amplify the positions.
    
    Return:
        - Wealth:   Strategy portfolio value


    TO DO:
    - Add commission and slippage (also for leverage)
    """


    ########################################################
    #################### DATA CHECKS #######################
    ########################################################

    if 'signal' not in df.columns:
        raise ValueError('"signal" column missing from df')
    
    if "exit_signal" not in df.columns:
        df["exit_signal"] = df["signal"]

    if price_column not in df.columns:
        raise ValueError(f'{price_column} missing from df')
    
    if 'timestamp' != df.index.name:
        if 'timestamp' not in df.columns:
            raise ValueError('"timestamp" column missing from df')
        else:
            df.set_index(pd.to_datetime(df.timestamp), inplace=True)
    else:
        if df.index.dtype != 'datetime64[ns]':
            df.index = pd.to_datetime(df.index)

    if start_date:
        df = df.loc[df.index>=start_date]
        if len(df) == 0:
            raise ValueError(f'{start_date} not in dataset. Please choose a less recent start date')

    if end_date:
        df = df.loc[df.index<=end_date]
        if  len(df) == 0:
            raise ValueError(f'{end_date} not in dataframe, please choose a more recent end_date')

    ########################################################
    #################### ALGORITHM #########################
    ########################################################

    i0              = 1     # Start at index 1
    df["Wealth"] = 1.0    # Initialize Wealth / Portfolio value
    position = 0 #Initial position is 0 (Cash)
    df['position'] = position 
    df["Realized_Return"] = 0.0 #Updated whenever a trade is closed, otherwise set to 0.
    df['BuyHold_Return'] = df[price_column].pct_change()
    # Loop over price time-series
    while i0 < len(df) - 1:

        wealth = df.loc[df.index[i0], "Wealth"]
        buy_and_hold_return = df.loc[df.index[i0], "BuyHold_Return"]
        entry_signal = df.loc[df.index[i0], "signal"]
        exit_signal = df.loc[df.index[i0], "exit_signal"]

        if position == 0:
            position  = entry_signal
            if position != 0:
                entry_price = df.iloc[i0+1][price_column] #When entry signal is observed at i0, trade is entered at i0+1
                if position == 1:
                    trade = 'Long'
                else:
                    trade = 'Short'
                print(f'{trade} Trade entered at entry price: {entry_price}')

        else:
            #When current position == 1 -> exit trade if current signal == 0 or -1
            #When current position == -1 -> exit trade if current signal == 0 or 1
            if (exit_signal != position): #Whenever the current signal is no longer aligned with the signal 
                #that generated our current position -> we close the position
                print(f'Closing {trade} position')
                exit_price = df.iloc[i0+1][price_column] #When exit signal is observed at i0, trade is exited at i0+1
                trade_return = calculate_trade_return(position, entry_price, exit_price, leverage)
                print(f'Trade exited at exit price: {exit_price}')
                print(f'Return of trade with leverage {leverage}: {trade_return}')
                df.loc[df.index[i0+1], "Realized_Return"] = trade_return
                position = 0

        
        df.loc[df.index[i0+1], "position"] = position #Position is updated at i0+1 based on signal observed at i0
        df.loc[df.index[i0+1], "Wealth"] = update_wealth(wealth, position, buy_and_hold_return, leverage)
        
        i0+=1
    
    if position != 0: #If a position is still open at last bar, close it
        exit_price = df.iloc[i0+1][price_column]
        trade_return = calculate_trade_return(position, entry_price, exit_price, leverage)
        position = 0
        df.loc[df.index[i0+1], "Realized_Return"] = trade_return
        df.loc[df.index[i0+1], "position"] = position #Position is updated at i0+1 based on signal observed at i0
        final_wealth = update_wealth(wealth, position, buy_and_hold_return, leverage)
        df.loc[df.index[i0+1], "Wealth"] = final_wealth
        print(f'Final wealth: {final_wealth}')

    return df