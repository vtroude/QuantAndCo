import pandas as pd
import numpy as np
from Backtest.utils import backtest_metrics, save_plot_strategy, market_trading_rules, convert_interval, check_column_index, check_expected_bars
from DataPipeline.make_data import make_filename
from DataPipeline.technicals_indicators import TechnicalIndicators
import time
import os
from matplotlib import pylab as pl

class Backtest:
    def __init__(self, strategy_df: pd.DataFrame, symbol: str, market: str, 
                interval: str, start_date: str=None, end_date: str=None,
                price_column: str='Close', initial_wealth: float=1.0,
                leverage: float=1.0, fees: float=0.0, slippage: float=0.0, 
                take_profit: float=1000, stop_loss: float=-1000):
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
        
        self.strategy_df = strategy_df
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
            self.strategy_df = self.strategy_df.loc[self.strategy_df.index>=self.start_date]
        
        if self.end_date:
            self.strategy_df = self.strategy_df.loc[self.strategy_df.index<=self.end_date]

        self.data_checks()

        self.first_date = self.strategy_df.index[0]
        self.last_date = self.strategy_df.index[-1]

        if not os.path.exists("Data/Backtest"):
            os.mkdir("Data/Backtest")

        self.backtest_df = self.Backtest_strategy()
        self.metrics = self.backtest_metrics()
        self.plot_equity_curve()

    
    def update_wealth(self, previous_wealth, buy_and_hold_return):
        """
        Current Wealth = Previous Wealth * (1 + position(-1,0,1) * Buy/Hold_Return * Leverage)
        Examples:
        If previous wealth is 10'000, open position is 1 (long), Asset Return between the 2 bars = 5%, and Leverage=1:
        Current wealth = 10'000 * (1+1*0.05*1) = 10'000*1.05 = 10'500
        If position is short -> 10'000 * (1-1*0.05) = 10'000 - 5%*10'000 = 9500
        If position is 0, current wealth = previous wealth

        """
        return previous_wealth * (1 + self.position * buy_and_hold_return * self.leverage * (1 - self.slippage - self.fees))

    def calculate_trade_return(self, entry_price, exit_price):
        return self.position * (exit_price/entry_price - 1) * self.leverage * (1 - self.slippage - self.fees)
    
    def update_unrealized_pnl(self, previous_pnl, wealth_change):
        return self.position * ( (1+wealth_change) * (1+previous_pnl) - 1 )
    
    def data_checks(self):

        print('Initializing data checks...')
        if 'signal' not in self.strategy_df.columns:
            raise ValueError('"signal" column missing from df')
        
        if "exit_signal" not in self.strategy_df.columns:
            self.strategy_df["exit_signal"] = self.strategy_df["signal"]

        if self.price_column not in self.strategy_df.columns:
            raise ValueError(f'{self.price_column} missing from df')
        
        if 'timestamp' != self.strategy_df.index.name:
            if 'timestamp' not in self.strategy_df.columns:
                raise ValueError('"timestamp" column missing from df')
            else:
                self.strategy_df.set_index(pd.to_datetime(self.strategy_df.timestamp), inplace=True)
        else:
            if self.strategy_df.index.dtype != 'datetime64[ns]':
                self.strategy_df.index = pd.to_datetime(self.strategy_df.index)

        if self.start_date and len(self.strategy_df) == 0:
                raise ValueError(f'{self.start_date} not in dataset. Please choose a less recent start date')

        if self.end_date and len(self.strategy_df) == 0:
                raise ValueError(f'{self.end_date} not in dataframe, please choose a more recent end_date')
    
        print('Data checks completed.')

    
    def Backtest_strategy(self):

        i0              = 1     # Start at index 1
        self.strategy_df["Wealth"] = self.initial_wealth    # Initialize Wealth / Portfolio value
        self.strategy_df['position'] = self.position 
        self.strategy_df["Realized_Return"] = 0.0 #Updated whenever a trade is closed, otherwise set to 0.
        self.strategy_df['Unrealized_PnL'] = 0.0
        self.strategy_df["Stop Loss"] = False
        self.strategy_df["Take Profit"] = False
        self.strategy_df[self.buy_hold_column] = self.strategy_df[self.price_column].pct_change()
        # Loop over price time-series
        print(f'Initializing backtesting...')
        while i0 < len(self.strategy_df) - 1:

            bar = self.strategy_df.index[i0]
            next_bar = self.strategy_df.index[i0+1]
            previous_wealth = self.strategy_df.loc[self.strategy_df.index[i0-1], "Wealth"]
            wealth = self.strategy_df.loc[self.strategy_df.index[i0], "Wealth"]
            buy_and_hold_return = self.strategy_df.loc[self.strategy_df.index[i0], self.buy_hold_column]
            entry_signal = self.strategy_df.loc[self.strategy_df.index[i0], "signal"]
            exit_signal = self.strategy_df.loc[self.strategy_df.index[i0], "exit_signal"]
            wealth_change = wealth / previous_wealth - 1
            unrealized_pnl = self.strategy_df.loc[self.strategy_df.index[i0], "Unrealized_PnL"]

            if self.position == 0:
                self.position  = entry_signal
                if self.position != 0:
                    print('='*50)
                    entry_price = self.strategy_df.iloc[i0+1][self.price_column] #When entry signal is observed at i0, trade is entered at i0+1
                    if self.position == 1:
                        direction = 'Long'
                    else:
                        direction = 'Short'
                    print(f'NEW {direction} signal detected on {bar}')
                    print(f'{direction} Trade entered on {next_bar} at entry price: {entry_price}')

            else:
                current_price = self.strategy_df.loc[self.strategy_df.index[i0], self.price_column]
                unrealized_pnl = self.position * (current_price / entry_price - 1) * self.leverage 

                exit_conditions = {
                'TP_rule': (unrealized_pnl >= self.take_profit, "Take Profit Rule"),
                'SL_rule': (unrealized_pnl <= self.stop_loss, "Stop Loss Rule"),
                'signal_rule': ((exit_signal != self.position) and (exit_signal != 0), "Change of Signal")
                }
                    
                if any(condition for condition, _ in exit_conditions.values()): #We exit the position as soon as we have a signal in the opposite direction. (OR TAKE PROFIT, TAKE LOSS)
                    print(f'CLOSING {direction} position on {next_bar}')
                    true_conditions = [description for condition, description in exit_conditions.values() if condition]
                    print(f'Reason: {true_conditions}')
                    exit_price = self.strategy_df.iloc[i0+1][self.price_column] #When exit signal is observed at i0, trade is exited at i0+1
                    trade_return = self.calculate_trade_return(entry_price, exit_price)
                    print(f'Trade exited at exit price: {exit_price}')
                    print(f'Return of trade with leverage {self.leverage}: {trade_return}')
                    print('='*50)
                    print('\n')
                    self.strategy_df.loc[self.strategy_df.index[i0+1], "Realized_Return"] = trade_return
                    self.position = 0
            

            self.strategy_df.loc[self.strategy_df.index[i0], "Unrealized_PnL"] = unrealized_pnl

            self.strategy_df.loc[self.strategy_df.index[i0], "Take Profit"] = True if unrealized_pnl>=self.take_profit else False
            self.strategy_df.loc[self.strategy_df.index[i0], "Stop Loss"] = True if unrealized_pnl<=self.stop_loss else False
            self.strategy_df.loc[self.strategy_df.index[i0+1], "position"] = self.position #Position is updated at i0+1 based on signal observed at i0
            self.strategy_df.loc[self.strategy_df.index[i0+1], "Wealth"] = self.update_wealth(wealth, buy_and_hold_return)

            
            i0+=1
        
        if self.position != 0: #If a position is still open at last bar, close it
            exit_price = self.strategy_df.iloc[i0][self.price_column]
            trade_return = self.calculate_trade_return(entry_price, exit_price)
            self.position = 0
            self.strategy_df.loc[self.strategy_df.index[i0], "Realized_Return"] = trade_return
            self.strategy_df.loc[self.strategy_df.index[i0], "position"] = self.position
            final_wealth = self.update_wealth(wealth, buy_and_hold_return)
            self.strategy_df.loc[self.strategy_df.index[i0], "Wealth"] = final_wealth
            print(f'Final wealth: {final_wealth}')
        
        print(f'Backtesting completed.')
        
        self.strategy_df.to_csv(f"Data/Backtest/{self.market}-{self.symbol}-{self.interval}-{self.first_date}-{self.last_date}-Backtest_DF.csv")

        return self.strategy_df
    
    def backtest_metrics(self):

        wealth_df = self.backtest_df['Wealth']
        check_column_index(wealth_df, "timestamp")
        n_trading_hours, n_trading_days = market_trading_rules(self.market)
        n_bars, n_years = check_expected_bars(wealth_df, self.interval, n_trading_hours, n_trading_days)

        CAGR = (wealth_df.iloc[-1] / wealth_df.iloc[0]) ** (1/n_years) - 1
        total_perf = (wealth_df.iloc[-1] / wealth_df.iloc[0]) - 1
        avg_return = wealth_df.pct_change().dropna().mean()
        avg_ann_return = avg_return * n_bars / n_years
        volatility = wealth_df.pct_change().dropna().std()
        ann_vol = volatility * np.sqrt( n_bars / n_years )
        sharpe = avg_ann_return / ann_vol


        metrics = [round(total_perf*100, 2), round(CAGR*100, 2), round(100*avg_ann_return, 2), round(100*ann_vol, 2), round(sharpe, 2)]
        df_metrics = pd.DataFrame(metrics).T
        df_metrics.columns = ['Total Performance [%]', 'CAGR [%]', 'Avg. Return (Ann.) [%]', 'Volatility (Ann.) [%]', 'Sharpe Ratio (Ann.)']
        df_metrics.index.name = 'Portfolio Metrics'
        df_metrics.to_csv(f"Data/Backtest/{self.market}-{self.symbol}-{self.interval}-{self.first_date}-{self.last_date}-backtest_metrics.csv")
        return df_metrics

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