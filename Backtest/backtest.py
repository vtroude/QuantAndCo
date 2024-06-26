import pandas as pd
import numpy as np
from Backtest.utils import market_trading_rules, check_column_index, check_expected_bars
import os
from matplotlib import pylab as pl
from Backtest.utils import set_logs
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

from typing import Callable, List, Optional

from Strategy.signal    import Signal
#from Strategy.utils     import get_strategy

import pylab    as pl

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

        #long_entries = backtest_df[(backtest_df["position"].shift(1)!=1) & (backtest_df["position"] == 1)]
        #long_exits = backtest_df[(backtest_df["position"].shift(1)==1) & (backtest_df["position"] != 1)]

        #short_entries = backtest_df[(backtest_df["position"].shift(1)!=-1) & (backtest_df["position"] == -1)]
        #short_exits = backtest_df[(backtest_df["position"].shift(1)== -1) & (backtest_df["position"] != -1)]

        #trades = (backtest_df['net_position_change'] != 0) & (backtest_df['position'] != backtest_df["net_position_change"])
        trades = backtest_df['net_position_change'] != 0
        trades_df = backtest_df[trades.fillna(False)][["unrealized_pnl"]]
        trades_df.dropna(inplace=True)
        #nb_orders = len(long_entries) + len(long_exits) + len(short_entries) + len(short_exits)
        nb_orders   = len(trades)
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
        #backtest_df['signal_change'] = backtest_df.signal.diff()
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
    
    def vectorized_backtest_multi_asset_trading(self, symbols):
        df          = self.signal_df.copy()
        s_weight    = [f"{s} weight" for s in symbols]
        s_entry     = [f"{s} entry" for s in symbols]

        df[s_weight]    = df[s_weight].shift(1).fillna(0) * self.leverage
        asset_pct       = df[symbols].pct_change().fillna(0)

        df['net_position_change']   = df[s_weight].diff().abs().sum(axis=1)

        df['new_position']                  = df['net_position_change'].shift(-1) != 0
        df.loc[df['new_position'], s_entry] = df[symbols][df['new_position']]
        
        df[s_entry].ffill(inplace=True)

        df['unrealized_pnl']    = (df[s_weight].shift(1) * ( (df[symbols] - df[s_entry]) / df[s_entry] )).sum(axis=1)
        if not self.stop_loss is None and self.take_profit is None:
            df['stop_triggered']    = ((df['unrealized_pnl'] <= self.stop_loss) | (df['unrealized_pnl'] >= self.take_profit)).shift(1).fillna(False)
            #df[s_weight]            = df.apply(lambda row: [0]*len(symbols) if row['stop_triggered'] else row[s_weight], axis=1)
            df.loc[df['stop_triggered'], s_weight]  = 0
        
        df['net_position_change']   = df[s_weight].diff().abs().sum(axis=1)

        df["portfolio_change"]  = (df[s_weight].to_numpy() * asset_pct).sum(axis=1)
        df["Wealth"]            = (1. + df['portfolio_change']  - self.fees * df["net_position_change"]).cumprod().shift(1) * self.initial_wealth

        df['fees']  = self.fees * df["Wealth"].shift(1) * df["net_position_change"].shift(1)

        return df


#######################################################################################################################

class BacktestSignal():

    def __init__(self, signal: Signal):
        """
        Initialize backtest procedure to test each signals over a defined portfolio

        Parameters:
        - signal:   Signal object to backtest
        """

        self.signal = signal

    def backtest_signal(self, signal: pd.Series) -> pd.Series:
        """
        Perform a backtest on a given signal, composed of an entry and exit point, on a given portfolio.

        Parameters:
        - signal:   A pandas Series containing the entry and exit points of the signal + portfolio weights

        Returns:
        - backtest: A pandas Series containing the backtest results for the given signal
        """

        growth_rate = self.pct.loc[signal["entry"]:signal["exit"]]
        if len(growth_rate) > 10:
            growth_rate *= signal["signal"]*signal[self.cols_w].to_numpy()
            growth_rate = growth_rate.to_numpy().astype("float") if len(self.cols_w) == 1 else growth_rate.sum(axis=1).to_numpy().astype("float")
            growth_rate += 1.
            
            log_r   = np.log(growth_rate)
            wealth  = np.cumprod(growth_rate)

            '''
            pl.figure()
            price   = self.data.loc[signal["entry"]:signal["exit"]]
            z_score = (np.log(price)*signal[self.cols_w].to_numpy()).sum(axis=1).to_numpy().astype("float")
            z_score = (z_score - signal["mean"])/signal["std"]
            price   /= price.iloc[0]

            price_  = self.data.loc[:signal["entry"]].iloc[-window_length:]
            z_score_ = (np.log(price_)*signal[self.cols_w].to_numpy()).sum(axis=1).to_numpy().astype("float")
            z_score_ = (z_score_ - signal["mean"])/signal["std"]

            pl.plot(price.index, price - 1.)
            pl.plot(price.index, wealth - 1., "k")
            pl.plot(price.index, z_score/100, "r")
            pl.plot(price_.index, z_score_/100, "g")
            pl.grid()
            pl.show()
            '''

            signal["success"]       = np.mean(log_r) / np.mean(np.abs(log_r))
            signal["Sharpe Ratio"]  = np.sqrt(365*24*60/len(wealth)) * np.mean(log_r) / np.std(log_r)
            signal["CAGR"]          = (wealth[-1]) ** (365*24*60/len(wealth)) - 1.
            signal['Max Drawdown']  = np.min(wealth) -1.
            signal['Max Drawup']    = np.max(wealth) -1.
            for q in [0.05, 0.25, 0.5, 0.75, 0.95]:
                signal[f'Quantile {q}'] = np.quantile(wealth, q) - 1.
            signal['Total'] = wealth[-1] - 1.

        return signal

    def backtest(self, data: pd.DataFrame, n_jobs: Optional[int]=1, cols: Optional[List[str]]=None, **kwargs) -> pd.DataFrame:
        """
        Perform a backtest on a given dataset using a given signal function.

        Parameters:
        - data:         A numpy array or pandas DataFrame containing the time series data.
        - n_jobs:       Number of parallel jobs to run. If None, the backtest will be run in a single process.
        - cols:         List of columns to use for the backtest.
        - kwargs:       Additional keyword arguments to pass to the signal function.

        Returns:
        - backtest_df:  A pandas DataFrame containing the backtest results for each entry/exit signal pair on a given portfolio
        """

        # Get signals
        signals = self.signal.full_backtest(data, n_jobs=n_jobs, cols=cols, **kwargs)
        print(signals)

        # Get data and define asset columns & protfolio weight column names
        self.data, self.cols    = self.signal.get_data(data, cols=cols)
        self.cols_w             = self.signal.get_cols_weight(cols=self.cols)
        
        # Shift the data to avoid look-ahead bias and get the growth rate
        self.pct    = self.data.shift(-2).pct_change()

        # Perform the backtest on each signal
        if n_jobs is None or n_jobs <2:
            return signals.apply(self.backtest_signal, axis=1)
        
        return pd.DataFrame(Parallel(n_jobs=n_jobs)(delayed(self.backtest_signal)(signal) for _, signal in signals.iterrows()))
    


#######################################################################################################################



if __name__ == "__main__":
    from DataPipeline.get_data  import get_price_data
    from Strategy.MeanReversion.mean_reversing  import MeanReversion, PairsMeanReversion, MultiMeanReversion

    n_jobs  = 7

    step            = 60 * 24
    look_a_head     = 60 * 24 * 2
    window_length   = 60 * 24 * 10
    min_timescale   = 30
    max_timescale   = 60 * 24
    use_log         = True

    entry_thresh    = 3.
    exit_thresh     = 1.

    market      = "forex"
    interval    = "1m"
    start       = 1563535876
    end         = 1713535876

    symbols = ["EUR_USD", "GBP_USD"]

    # Load the data
    data            = pd.concat([get_price_data(market=market, symbol=s, interval=interval, date1=start, date2=end)["Close"] for s in symbols], axis=1).dropna()
    data.columns    = symbols

    print(data)

    # Prepare the data
    #data = prepare_data(price_df, indicators_df)

    # Perform a backtest
    #backtest_signal(data, hurst_entry_exit, market=market, interval=interval, cols=symbols, window_length=window_length, n_jobs=n_jobs, use_log=True)
    signal  = BacktestSignal(MultiMeanReversion(minimal_time_scale=min_timescale, maximal_time_scale=max_timescale)).backtest(
                                                                                                                            data,
                                                                                                                            n_jobs=n_jobs,
                                                                                                                            cols=symbols,
                                                                                                                            window_length=window_length,
                                                                                                                            use_log=use_log,
                                                                                                                            step=step,
                                                                                                                            look_a_head=look_a_head,
                                                                                                                            exit_threshold=exit_thresh,
                                                                                                                            signal_threshold=entry_thresh
                                                                                                                        )

    signal  = signal.dropna()

    print(signal)

    pl.figure()
    pl.hist(signal["success"], bins=np.linspace(signal["success"].min(), signal["success"].max(), np.minimum(100, len(signal))))
    pl.xlabel("Success")
    pl.ylabel("Frequency")
    pl.title("Histogram of Success")
    pl.grid()
    pl.show()




    