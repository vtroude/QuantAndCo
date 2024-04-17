import pandas as pd
import numpy as np

def convert_interval(interval, n_trading_hours, n_trading_days, n_years):
    if interval in ['1m', '5m', '30m']:
        interval = interval[:-1]
        return 60/int(interval) * n_trading_hours * n_trading_days * n_years
    elif interval == '1h':
        return n_trading_hours * n_trading_days * n_years
    elif interval == '1d':
        return n_trading_days * n_years
    elif interval == '1w':
        n_trading_weeks = 52 * n_trading_days / 365
        return n_trading_weeks * n_years
    elif interval == '1m':
        n_trading_months = 30 * n_trading_days / 365
        return n_trading_months * n_years
    else:
        raise ValueError (f'{interval} not covered by function')

def annualized_performance(wealth_df:pd.DataFrame, interval:str, n_trading_hours:float, n_trading_days:int) -> list:

    if wealth_df.index.name != 'timestamp':
        if 'timestamp' not in wealth_df.columns:
            raise ValueError('"timestamp" column missing in price_wealth_df')
        else:
            wealth_df['timestamp'] = pd.to_datetime(wealth_df['timestamp'])
            wealth_df.set_index('timestamp', inplace=True)

    start_date = wealth_df.index[0]
    end_date = wealth_df.index[-1]
    time_delta = end_date - start_date
    n_years  = time_delta.days / 365.25 + time_delta.seconds / (365.25 * 24 * 3600)
    expected_n_bars = convert_interval(interval, n_trading_hours, n_trading_days, n_years)
    n_bars = len(wealth_df)
    delta_n_bars = (n_bars - expected_n_bars) / expected_n_bars
    if abs(delta_n_bars) > 0.05:
        print('Dataset is missing more than 5% of expected bars in this time frequency')

    CAGR = (wealth_df.iloc[-1] / wealth_df.iloc[0]) ** (1/n_years) - 1
    total_perf = (wealth_df.iloc[-1] / wealth_df.iloc[0]) - 1
    avg_return = wealth_df.pct_change().dropna().mean()
    avg_ann_return = avg_return * n_bars / n_years
    volatility = wealth_df.pct_change().dropna().std()
    ann_vol = volatility * np.sqrt( n_bars / n_years )
    sharpe = avg_ann_return / ann_vol


    metrics = [round(total_perf*100, 2), round(CAGR[0]*100, 2), round(100*avg_ann_return[0], 2), round(100*ann_vol[0], 2), round(sharpe[0], 2)]
    df_metrics = pd.DataFrame(metrics).T
    df_metrics.columns = ['Total Performance [%]', 'CAGR [%]', 'Avg. Return (Ann.) [%]', 'Volatility (Ann.) [%]', 'Sharpe Ratio (Ann.)']
    df_metrics.index.name = 'Portfolio Metrics'
    return df_metrics

if __name__ == '__main__':
    df = pd.read_csv('/root/QuantAndCo/Data/BTCUSDT_5m_2020-01-01_2024-12-04_backtest_RSI_[20, 100, 200, 500].csv')
    df_wealth = df[['timestamp', 'Wealth']]
    display(annualized_performance(wealth_df=df_wealth, interval='5m', n_trading_hours=24, n_trading_days=365))

 
