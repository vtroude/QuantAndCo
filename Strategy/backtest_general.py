import pandas as pd
from Strategy.strategies import RSIStrategy

def update_wealth(previous_wealth, position, buy_and_hold_return, leverage):
    return previous_wealth * (1 + position * buy_and_hold_return * leverage)

def calculate_trade_return(position, entry_price, exit_price, leverage, commission_fees=0., slippage_fees=0.):
    return position * (exit_price/entry_price - 1) * leverage

def Backtest_Strategy(df:pd.DataFrame, price_column='Close', leverage=1)->pd.DataFrame:
    """
    This function takes as input a strategy dataframe which contains the price 
    of the asset and signals of the strategy, and returns the wealth of the portfolio.
    """

    #1. Assert that the dataframe contains timestamp as index, that it is a datetime column,
    #that it contains a price column and an entry and exit signal columns.


    i0              = 1     # Start at index 1
    df["Wealth"] = 1.0    # Initialize Wealth / Portfolio value
    position = 0 #Initial position is 0 (Cash)
    df["Realized_Return"] = 0.0
    df['position'] = 0
    df['BuyHold_Return'] = df[price_column].pct_change()
    # Loop over price time-series
    while i0 < len(df) - 1:
        ### Entry conditions: If all indicators are above or below
        ### Exit conditions: If any indicator is above or below

        wealth = df.loc[df.index[i0], "Wealth"]
        buy_and_hold_return = df.loc[df.index[i0], "BuyHold_Return"]
        entry_signal = df.loc[df.index[i0], "entry_signal"]
        exit_long = df.loc[df.index[i0], "exit_long"]
        exit_short = df.loc[df.index[i0], "exit_short"]

        if position == 0:
            position  = entry_signal
            if position != 0:
                entry_price = df.iloc[i0+1][price_column] #When entry signal is observed at i0, trade is entered at i0+1
                if position == 1:
                    trade = 'Long'
                else:
                    trade = 'Short'
                print(f'{trade} Trade entered at entry price: {entry_price}')

        if (position == 1 and exit_long) or (position == -1 and exit_short):
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

if __name__ == '__main__':
    price_df = pd.read_csv('Data/BTCUSDT_5m_2020-01-01_2024-12-04_ohlc.csv')
    ti = pd.read_csv('Data/BTCUSDT_5m_2020-01-01_2024-12-04_ti.csv')
    span        = [20, 100, 200, 500] 
    ti_cols = ['StochRSI-' + str(s) for s in span]
    ti = ti[ti_cols + ['timestamp']]
    print(price_df.head(10))
    strategy = RSIStrategy(price_df, ti, ti_cols)
    strategy_df = strategy.prepare_data()
    strategy_df['entry_signal'] = strategy.entry_signal(strategy_df, entry_long_thres=0.1, entry_short_thres=0.9)
    strategy_df['exit_long'] = strategy.exit_long(strategy_df, exit_long_thres=0.2)
    strategy_df['exit_short'] = strategy.exit_short(strategy_df, exit_short_thres=0.8)
    backtest_df = Backtest_Strategy(strategy_df, leverage = 10)

    print(backtest_df.head(10))