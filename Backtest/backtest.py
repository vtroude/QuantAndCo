import pandas as pd
import numpy as np
from Backtest.utils import market_trading_rules, check_column_index, check_expected_bars
import os
from matplotlib import pylab as pl
from Backtest.utils import set_logs
import matplotlib.pyplot as plt

from typing import Callable, List, Optional

from Strategy.utils import get_strategy

class Backtest:
    def __init__(self, signal_df: pd.DataFrame, symbol: str, market: str, 
                interval: str, start_date: str=None, end_date: str=None,
                price_column: str='Close', initial_wealth: float=1.0,
                leverage: float=1.0, fees: float=0.0, slippage: float=0.0, 
                take_profit: Optional[float]=None, stop_loss: Optional[float]=None, check_data=False):
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

        """
        
        self.signal_df = signal_df
        self.start_date = start_date
        self.end_date = end_date
        self.price_column = price_column
        self.leverage = leverage
        self.fees = fees
        self.slippage = slippage
        self.position = 0
        self.symbol = symbol
        self.market = market
        self.interval = interval
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.buy_hold_column = 'buy_and_hold'
        self.initial_wealth = initial_wealth

        if self.start_date:
            self.signal_df = self.signal_df.loc[self.signal_df.index>=self.start_date]
        
        if self.end_date:
            self.signal_df = self.signal_df.loc[self.signal_df.index<=self.end_date]
        

        self.first_date = self.signal_df.index[0]
        self.last_date = self.signal_df.index[-1]
    
        
        log_file = f"logs-{self.symbol}-{self.interval}-{self.first_date}-{self.last_date}.log"
        self.logger = set_logs(log_file)

        if check_data:
            self.data_checks()

        if not os.path.exists("Data/Backtest"):
            os.mkdir("Data/Backtest")


        #self.backtest_df = self.Backtest_strategy()
        #self.metrics = self.backtest_metrics()
        #self.plot_equity_curve()

    
    def update_wealth(self, previous_wealth, buy_and_hold_return, position_change):
        """
        Current Wealth = Previous Wealth * (1 + position(-1,0,1) * Buy/Hold_Return * Leverage)
        Examples:
        If previous wealth is 10'000, open position is 1 (long), Asset Return between the 2 bars = 5%, and Leverage=1:
        Current wealth = 10'000 * (1+1*0.05*1) = 10'000*1.05 = 10'500
        If position is short -> 10'000 * (1-1*0.05) = 10'000 - 5%*10'000 = 9500
        If position is 0, current wealth = previous wealth

        """
        capital_appreciation = buy_and_hold_return * self.leverage
        total_fees = self.fees * self.leverage
        total_slippage = self.slippage * self.leverage
        new_wealth = previous_wealth * (1 + self.position * capital_appreciation) - position_change * (total_fees + total_slippage)
        return new_wealth

    def calculate_trade_return(self, entry_price, exit_price):
        return self.position * (exit_price/entry_price - 1) * self.leverage * (1 - self.slippage - self.fees)

    def calculate_pairs_trade_return(self, hedge_ratio, entry_price_y, exit_price_y, entry_price_x, exit_price_x):
        y_return = exit_price_y / entry_price_y - 1
        x_return = exit_price_x / entry_price_x - 1
        return self.position * (y_return - hedge_ratio * x_return) * self.leverage * (1 - self.slippage - self.fees)

    
    def update_unrealized_pnl(self, previous_pnl, wealth_change):
        return self.position * ( (1+wealth_change) * (1+previous_pnl) - 1 )
    
    def data_checks(self):

        self.logger.debug('Initializing data checks...')
        if 'signal' not in self.signal_df.columns:
            raise ValueError('"signal" column missing from df')
        
        if "exit_signal" not in self.signal_df.columns:
            self.signal_df["exit_signal"] = self.signal_df["signal"]

        if self.price_column not in self.signal_df.columns:
            raise ValueError(f'{self.price_column} missing from df')
        
        if 'timestamp' != self.signal_df.index.name:
            if 'timestamp' not in self.signal_df.columns:
                raise ValueError('"timestamp" column missing from df')
            else:
                self.signal_df.set_index(pd.to_datetime(self.signal_df.timestamp), inplace=True)
        else:
            if self.signal_df.index.dtype != 'datetime64[ns]':
                self.signal_df.index = pd.to_datetime(self.signal_df.index)

        if self.start_date and len(self.signal_df) == 0:
                raise ValueError(f'{self.start_date} not in dataset. Please choose a less recent start date')

        if self.end_date and len(self.signal_df) == 0:
                raise ValueError(f'{self.end_date} not in dataframe, please choose a more recent end_date')
    
        self.logger.debug('Data checks completed.')

    
    def Backtest_strategy(self):

        i0              = 1     # Start at index 1
        self.signal_df["Wealth"] = self.initial_wealth    # Initialize Wealth / Portfolio value
        self.signal_df['position'] = self.position 
        self.signal_df["Realized_Return"] = 0.0 #Updated whenever a trade is closed, otherwise set to 0.
        self.signal_df['Unrealized_PnL'] = 0.0
        self.signal_df["Stop Loss"] = False
        self.signal_df["Take Profit"] = False
        self.signal_df[self.buy_hold_column] = self.signal_df[self.price_column].pct_change()
        # Loop over price time-series
        self.logger.debug(f'Initializing backtesting...')
        while i0 < len(self.signal_df) - 1:

            bar = self.signal_df.index[i0]
            next_bar = self.signal_df.index[i0+1]
            wealth = self.signal_df.loc[self.signal_df.index[i0], "Wealth"]
            buy_and_hold_return = self.signal_df.loc[self.signal_df.index[i0+1], self.buy_hold_column] #buy and hold return of next bar, because it is only used for wealth calculated which is updated a next bar 
            entry_signal = self.signal_df.loc[self.signal_df.index[i0], "signal"]
            exit_signal = self.signal_df.loc[self.signal_df.index[i0], "exit_signal"]
            position_change = 0
            unrealized_pnl = self.signal_df.loc[self.signal_df.index[i0], "Unrealized_PnL"]

            if self.position == 0:
                self.position  = entry_signal 
                if self.position != 0:
                    position_change = 1
                    self.logger.debug('='*50)
                    entry_price = self.signal_df.iloc[i0+1][self.price_column] #When entry signal is observed at i0, trade is entered at i0+1
                    if self.position == 1:
                        direction = 'Long'
                    else:
                        direction = 'Short'
                    self.logger.debug(f'NEW {direction} signal detected on {bar}')
                    self.logger.debug(f'{direction} Trade entered on {next_bar} at entry price: {entry_price}')

            else:
                current_price = self.signal_df.loc[self.signal_df.index[i0], self.price_column]
                unrealized_pnl = self.position * (current_price / entry_price - 1) * self.leverage 

                exit_conditions = {
                'TP_rule': (unrealized_pnl >= self.take_profit, "Take Profit Rule"),
                'SL_rule': (unrealized_pnl <= self.stop_loss, "Stop Loss Rule"),
                'signal_rule': ((exit_signal != self.position) and (exit_signal != 0), "Change of Signal")
                }
                    
                if any(condition for condition, _ in exit_conditions.values()): #We exit the position as soon as we have a signal in the opposite direction. (OR TAKE PROFIT, TAKE LOSS)
                    self.logger.debug(f'CLOSING {direction} position on {next_bar}')
                    position_change = 1
                    true_conditions = [description for condition, description in exit_conditions.values() if condition]
                    self.logger.debug(f'Reason: {true_conditions}')
                    exit_price = self.signal_df.iloc[i0+1][self.price_column] #When exit signal is observed at i0, trade is exited at i0+1
                    trade_return = self.calculate_trade_return(entry_price, exit_price)
                    self.logger.debug(f'Trade exited at exit price: {exit_price}')
                    self.logger.debug(f'Return of trade with leverage {self.leverage}: {trade_return}')
                    self.logger.debug('='*50)
                    self.logger.debug('\n')
                    self.signal_df.loc[self.signal_df.index[i0+1], "Realized_Return"] = trade_return
                    self.position = 0
            

            self.signal_df.loc[self.signal_df.index[i0], "Unrealized_PnL"] = unrealized_pnl

            self.signal_df.loc[self.signal_df.index[i0], "Take Profit"] = True if unrealized_pnl>=self.take_profit else False
            self.signal_df.loc[self.signal_df.index[i0], "Stop Loss"] = True if unrealized_pnl<=self.stop_loss else False
            self.signal_df.loc[self.signal_df.index[i0], "position"] = self.position #We now assume position is immediately updated once signal is observed, however we apply negative slippage to the execution price
            updated_wealth = self.update_wealth(wealth, buy_and_hold_return, position_change)
            #self.logger.debug(f'Wealth: {updated_wealth}')
            self.signal_df.loc[self.signal_df.index[i0+1], "Wealth"] = updated_wealth #wealth is updated one bar after the position is entered

            if updated_wealth <= 0.0001:
                self.logger.debug('Backtested ended before end date. Reason: Wealth dropped to zero.')
                return self.signal_df

            
            i0+=1
        
        if self.position != 0: #If a position is still open at last bar, close it
            exit_price = self.signal_df.iloc[i0][self.price_column]
            trade_return = self.calculate_trade_return(entry_price, exit_price)
            self.logger.debug(f'CLOSING {direction} position on {bar}')
            self.logger.debug('Reason: End of backtesting, closing all positions.')
            self.logger.debug(f'Trade exited at exit price: {exit_price}')
            self.logger.debug(f'Return of trade with leverage {self.leverage}: {trade_return}')
            self.position = 0
            self.signal_df.loc[self.signal_df.index[i0], "Realized_Return"] = trade_return
            self.signal_df.loc[self.signal_df.index[i0], "position"] = self.position
            final_wealth = self.update_wealth(wealth, buy_and_hold_return, position_change=1)
            self.signal_df.loc[self.signal_df.index[i0], "Wealth"] = final_wealth
            self.logger.debug(f'Final wealth: {final_wealth}')
        
        self.logger.debug(f'Backtesting completed.')
        
        self.signal_df.to_csv(f"Data/Backtest/{self.market}-{self.symbol}-{self.interval}-{self.first_date}-{self.last_date}-Backtest_DF.csv")

        return self.signal_df

    
    def calculate_perf(self, wealth_df):

        return (wealth_df.iloc[-1] / wealth_df.iloc[0]) - 1
    
    def max_drawdown(self, backtest_df):
        df_returns = backtest_df[["Wealth"]].pct_change()
        cum_returns = (1 + df_returns).cumprod()
        drawdown =  1 - cum_returns.div(cum_returns.cummax())
        returns_arr = cum_returns.dropna().values
        dd_end = np.argmax(np.maximum.accumulate(returns_arr) - returns_arr) # end of the period
        dd_start = np.argmax(returns_arr[:dd_end]) # start of period
        max_dd = np.max(drawdown)
        max_dd_duration = backtest_df.index[dd_end] - backtest_df.index[dd_start]
        plt.plot(returns_arr)
        plt.plot([dd_end, dd_start], [returns_arr[dd_end], returns_arr[dd_start]], 'o', color='Red', markersize=10)
        

        return max_dd, max_dd_duration

    def trades_statistics(self, backtest_df):

        long_entries = backtest_df[(backtest_df["position"].shift(1)!=1) & (backtest_df["position"] == 1)]
        long_exits = backtest_df[(backtest_df["position"].shift(1)==1) & (backtest_df["position"] != 1)]

        short_entries = backtest_df[(backtest_df["position"].shift(1)!=-1) & (backtest_df["position"] == -1)]
        short_exits = backtest_df[(backtest_df["position"].shift(1)== -1) & (backtest_df["position"] != -1)]

        trades = (backtest_df['net_position_change'] != 0) & (backtest_df['position'] != backtest_df["net_position_change"])
        trades_df = backtest_df[trades.fillna(False)][["unrealized_pnl"]]
        trades_df.dropna(inplace=True)
        nb_orders = len(long_entries) + len(long_exits) + len(short_entries) + len(short_exits)
        avg_trade_return = np.mean(trades_df)
        best_trade = np.max(trades_df)
        worst_trade = np.min(trades_df)
        wins = trades_df.loc[trades_df.unrealized_pnl > 0]
        losses = trades_df.loc[trades_df.unrealized_pnl < 0]
        win_rate = len(wins) / len(trades_df)
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)

        return nb_orders, avg_trade_return, best_trade, worst_trade, win_rate, avg_win, avg_loss
        
    
    def backtest_metrics(self, backtest_df, fee_column='fees', 
                         return_metric=None):

        wealth_df = backtest_df["Wealth"].dropna()


        #check_column_index(wealth_df, "timestamp")
        n_trading_hours, n_trading_days = market_trading_rules(self.market)
        n_bars, n_years = check_expected_bars(wealth_df, self.interval, n_trading_hours, n_trading_days)

        CAGR = (wealth_df.iloc[-1] / wealth_df.iloc[0]) ** (1/n_years) - 1
        total_perf = self.calculate_perf(wealth_df)
        avg_return = wealth_df.pct_change().dropna().mean()
        avg_ann_return = avg_return * n_bars / n_years
        volatility = wealth_df.pct_change().dropna().std()
        ann_vol = volatility * np.sqrt( n_bars / n_years )
        sharpe = avg_ann_return / ann_vol
        start = wealth_df.index[0]
        end = wealth_df.index[-1]
        initial_value = wealth_df.iloc[0]
        period = end - start
        min_value = np.min(wealth_df)
        max_value = np.max(wealth_df)
        end_value = wealth_df.iloc[-1]
        backtest_df['signal_change'] = backtest_df.signal.diff()
        total_fees = backtest_df[fee_column].sum()
        max_dd, max_dd_duration = self.max_drawdown(backtest_df)
        nb_orders, avg_trade, best_trade, worst_trade, win_ratio, avg_win, avg_loss = self.trades_statistics(backtest_df)
        avg_trades_per_day = nb_orders / period.days

        
        metrics = [start, end, period, self.interval, initial_value, min_value, max_value, end_value, 
                   round(total_perf*100, 2), round(CAGR*100, 2), 
                   round(100*avg_ann_return, 2), round(100*ann_vol, 2), round(sharpe, 2), 
                   round(max_dd, 4), max_dd_duration, nb_orders, round(avg_trade*100, 2), 
                   round(win_ratio*100, 2), round(avg_win*100, 2), round(avg_loss*100, 2),
                   round(best_trade*100, 2), round(worst_trade*100, 2),
                   round(avg_trades_per_day, 2), self.leverage, total_fees, 
                   round(total_fees/initial_value*100, 2)]
        df_metrics = pd.DataFrame(metrics).T
        df_metrics.columns = ['Start', 'End', 'Period', 'Strategy Frequency', 'Start Value', 'Min Value', 'Max Value', 'End Value', 'Total Performance [%]', 'CAGR [%]', 'Avg. Return (Ann.) [%]', 'Volatility (Ann.) [%]', 'Sharpe Ratio (Ann.)',
                              'Max Drawdown', 'Max Drawdown Duration', 'Orders', 'Avg. Trade Return [%]', 'Win Rate [%]', 'Avg. Win [%]', 'Avg. Loss [%]', 'Best Trade [%]', 'Worst Trade [%]', 'Avg. Trades Per Day', 'Leverage', 'Total Fees [$]', 'Total Fees [%]']
        df_metrics.to_csv(f"Data/Backtest/{self.market}-{self.symbol}-{self.interval}-{self.first_date}-{self.last_date}-backtest_metrics.csv")
        if return_metric:
            if return_metric == 'total_perf':
                return total_perf
            elif return_metric == 'CAGR':
                return CAGR
            elif return_metric == 'avg_ann_return':
                return avg_ann_return
            elif return_metric == 'ann_vol':
                return ann_vol
            elif return_metric == 'sharpe':
                return sharpe
            else:
                raise ValueError(f"Metric {return_metric} not found")
            
        return df_metrics.T
    
    def plot_strategy(self, backtest_df, long_thres=None, short_thres=None):

        # Plotting the price data
        plt.figure(figsize=(12, 6))
        plt.plot(backtest_df[self.price_column], label='Price')

        long_entries = backtest_df[(backtest_df["position"].shift(1)!=1) & (backtest_df["position"] == 1)]
        long_exits = backtest_df[(backtest_df["position"].shift(1)==1) & (backtest_df["position"] != 1)]

        short_entries = backtest_df[(backtest_df["position"].shift(1)!=-1) & (backtest_df["position"] == -1)]
        short_exits = backtest_df[(backtest_df["position"].shift(1)== -1) & (backtest_df["position"] != -1)]

        # Plot long entries (blue upward triangles)
        plt.plot(long_entries.index, long_entries[self.price_column], '^', markersize=10, color='blue', lw=0, label='Long Entry')

        # Plot long exits (cyan circles)
        plt.plot(long_exits.index, long_exits[self.price_column], 'o', markersize=10, color='cyan', lw=0, label='Long Exit')

        # Plot short entries (red downward triangles)
        plt.plot(short_entries.index, short_entries[self.price_column], 'v', markersize=10, color='red', lw=0, label='Short Entry')

        # Plot short exits (magenta crosses)
        plt.plot(short_exits.index, short_exits[self.price_column], 'x', markersize=10, color='magenta', lw=0, label='Short Exit')
        
        if long_thres:
            plt.axhline(y=long_thres, color='green', linestyle='--', linewidth=1, label='Long Threshold')
        
        if short_thres:
            plt.axhline(y=short_thres, color='red', linestyle='--', linewidth=1, label='Short Threshold')


        # Add labels and legend
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Price with Long/Short Entries and Exits')
        plt.legend()

        # Show the plot
        plt.savefig('Data/Backtest/strategy_plot.png')

    def plot_equity_curve(self):
        """
        Plot B&H Vs Portfolio Value / Wealth

        Input:
            - price:        Price time-series
            - wealth:       Wealth / Portfolio Value
            - model_name:   model or strategy name
            - labels:       extra data e.g. symbol, date, etc...
        """

        fig, ax = pl.subplots(1, 1, figsize=(24,14))

        #ax.set_title(f"Backtest {labels['symbol']} from {labels['date start']} to {labels['date end']} ($k={labels['thresh']}$ $n={labels['n points']}$)", fontsize=20)

        self.backtest_df['Wealth'].plot(ax=ax, grid=True, logy=False) #Was set to true before
        (self.backtest_df[self.price_column]/self.backtest_df[self.price_column].iloc[0]).plot(ax=ax, color="k", grid=True, logy=False)

        ax.set_ylabel("Portfolio Value", fontsize=18)
        ax.set_xlabel("Date", fontsize=18)
        ax.legend(fontsize=26)

        fig.subplots_adjust(left=0.055, bottom=0.093, right=0.962, top=0.92)
        if not os.path.exists("Data/Backtest/Figure"):
            os.mkdir("Data/Backtest/Figure")
        
        file_name = f'Data/Backtest/Figure/{self.symbol}-{self.interval}-{self.first_date}-{self.last_date}-Equity_Curve.png'

        fig.savefig(file_name)

    def vectorized_backtesting(self):

        df = self.signal_df.copy()

        # Avoid look-ahead bias by shifting the signal forward by one period
        # Signals effectively become actionable the next bar
        df['position'] = df['signal'].shift(1).fillna(0) * self.leverage


        # Calculate daily returns for each asset
        df["return"] = df[self.price_column].pct_change().fillna(0)

        df["net_position_change"] = df['position'].diff().abs()

        # Calculate portfolio changes from returns
        df['portfolio_change'] = df[f'position'].shift(1) * df['return']

        # Determine positions where we go from 0 to 1 or -1
        df['new_position'] = (df['position'] != 0) & (df['position'].shift(1) == 0)

        # Set the entry price only when 'new_position' is True
        df['entry_price'] = df.apply(lambda row: row[self.price_column] if row['new_position'] else np.nan, axis=1)

        # Forward fill the entry prices to apply them until the position changes
        df['entry_price'].ffill(inplace=True)

        # Calculate unrealized pnl
        df['unrealized_pnl'] = df["position"].shift(1) * ( (df[self.price_column] - df['entry_price']) / df['entry_price'] )

        if not self.stop_loss is None and self.take_profit is None:
            df['stop_triggered'] = ((df['unrealized_pnl'] <= self.stop_loss) | (df['unrealized_pnl'] >= self.take_profit)).shift(1).fillna(False)
            df['position'] = df.apply(lambda row: 0 if row['stop_triggered'] else row['position'], axis=1)

        df['net_position_change'] = df['position'].diff().abs()
        df['portfolio_change'] = df['position'].shift(1) * df['return']

        # Calculate cumulative wealth starting from initial_wealth
        df['Wealth'] = (1+df['portfolio_change'] - self.fees * df["net_position_change"].shift(1)).cumprod() * self.initial_wealth
        df["fees"] = self.fees * df["Wealth"].shift(1) * df["net_position_change"].shift(1)

        return df

    def vectorized_backtest_pairs_trading(self, asset_x, asset_y):

        ## TO DO: Need to implement Stop Loss / Take Profit as in vectorized_backtesting()
        
        """
        Vectorized backtesting of a pairs trading strategy.

        Parameters:
        - df (pd.DataFrame): Input DataFrame containing asset_x and asset_y as columns, 'hedge_ratio', 'signal'
        - leverage (float or pd.Series): The leverage multiplier to scale the positions and costs.

        Adds a 'Wealth' column to the DataFrame which accumulates the wealth of the strategy over time.

        The strategy:
        - 'signal' = -1: Short the pair, short asset_y and long asset_x
        - 'signal' = 1: Long the pair, long asset_y and short asset_x
        - 'signal' = 0: No position
        
        We use the 'hedge_ratio' to balance the position sizes between asset_x and asset_y.
        Transaction costs are proportional to the leverage used.
        """

        df = self.signal_df.copy()

        if 'hedge_ratio' not in df.columns:
            df['hedge_ratio'] = 1.0

        # Avoid look-ahead bias by shifting the signal forward by one period
        # Signals effectively become actionable the next bar
        df['signal'] = df['signal'].shift(1).fillna(0)
        
        # Calculate positions for asset_x and asset_y based on signal and hedge ratio
        df[f'position_{asset_y}'] = df['signal'] * self.leverage  # Position in asset_y directly follows the signal
        df[f'position_{asset_x}'] = -df['signal'] * df['hedge_ratio'] * self.leverage  # Position in asset_x is opposite and scaled by hedge ratio

        return_x, return_y = f'return_{asset_x}', f'return_{asset_y}'

        # Calculate daily returns for each asset
        df[return_x] = df[asset_x].pct_change().fillna(0)
        df[return_y] = df[asset_y].pct_change().fillna(0)

        df["net_position_change"] = df[[f'position_{asset_x}', f'position_{asset_y}']].diff().abs().sum(axis=1)

        # Calculate portfolio changes from returns
        df['portfolio_change'] = df[f'position_{asset_x}'] * df[return_x] + df[f'position_{asset_y}'] * df[return_y]
        
        # Calculate cumulative wealth starting from initial wealth
        df['Wealth'] = (1+df['portfolio_change'] - self.fees * df["net_position_change"]).cumprod().shift(1) * self.initial_wealth
        df["fees"] = self.fees * df["Wealth"].shift(1) * df["net_position_change"].shift(1)

        return df

#######################################################################################################################

def backtest_signal(
                        data: pd.DataFrame,
                        signal_func: Callable,
                        market: str,
                        interval: str,
                        cols: List[str]=["close"],
                        n_jobs: Optional[int]=1,
                        **kwargs
                    ) -> pd.DataFrame:
    """
    Perform a backtest on a given dataset using a given signal function.

    Parameters:
    - data:         A numpy array or pandas DataFrame containing the time series data.
    - signal_func:  The signal function to use for the backtest.
    - kwargs:       Additional keyword arguments to pass to the signal function.

    Returns:
    - backtest_df:  A pandas DataFrame containing the backtest results.
    """

    # Get signal to backtest
    signal_to_backtest  = get_strategy(data, signal_func, cols=cols, n_jobs=n_jobs, **kwargs)

    # Get portfolio weight columns name
    cols_w  = [ f'{c} weight' for c in cols ]

    # Get triggered signals
    signals = signal_to_backtest.dropna()
    signals = signals[signals["signal"] != 0]

    # Get portfolio / symbol
    symbol  = "/".join(cols)

    for i in signals.index:
        to_backtest = pd.DataFrame()
        to_backtest["portfolio"]    = (signal_to_backtest[cols].loc[i:data.loc[signals.loc[i, "exit"]:].index[1]]*signal_to_backtest[cols_w].loc[i].to_numpy()).sum(axis=1)
        to_backtest["signal"]       = signals.loc[i, "signal"]
        to_backtest.loc[to_backtest.index[-2:], "signal"]   = 0

        df  = Backtest(to_backtest, symbol=symbol, market=market, interval=interval, price_column="portfolio").vectorized_backtesting()
        print(df)

#######################################################################################################################



if __name__ == "__main__":
    from DataPipeline.get_data  import get_price_data

    from Strategy.MeanReversion.hurst import hurst_entry_exit
    #from Strategy.utils import prepare_data

    #/home/virgile/Desktop/Trading/QuantDotCom/QuantAndCo/Data/forex/EUR_USD/1m/OHLC/1563535876_1713535876.csv

    n_jobs  = 10

    window_length   = 60 * 24 * 30

    market      = "forex"
    interval    = "1m"
    start       = 1563535876
    end         = 1713535876

    symbols = ["EUR_USD"]

    # Load the data
    data            = pd.concat([get_price_data(market=market, symbol=s, interval=interval, date1=start, date2=end)["Close"] for s in symbols], axis=1)
    data.columns    = symbols

    print(data)
    print(window_length)

    # Prepare the data
    #data = prepare_data(price_df, indicators_df)

    # Perform a backtest
    backtest_signal(data, hurst_entry_exit, market=market, interval=interval, cols=symbols, window_length=window_length, n_jobs=n_jobs)





    