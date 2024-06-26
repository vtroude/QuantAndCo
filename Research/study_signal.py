import numpy    as np
import pandas   as pd

from typing import List

def make_portfolio(data: pd.DataFrame, signals: pd.DataFrame, symbols: List[str]):
    s_weight    = [f"{s} weight"for s in symbols]

    data[s_weight]      = 0.
    data["multiplier"]  = 0.
    for _, s in signals.iterrows():
        if s["signal"] != 0:
            data_slice  = data[s_weight + ["multiplier"]][s["entry"]:s["exit"]].fillna(0.)
            
            #data_slice[s_weight]        *= data_slice["multiplier"].to_numpy().reshape(-1,1)
            data_slice[s_weight]        += s["signal"]*s[s_weight].to_numpy().reshape(1,-1)
            data_slice['multiplier']    += 1.
            #data_slice[s_weight]        /= data_slice["multiplier"].to_numpy().reshape(-1,1)

            data.loc[s["entry"]:s["exit"], s_weight + ["multiplier"]]    = data_slice

    data[s_weight]  /= data["multiplier"].max()

    return data

def apply_signal(data, signals):
    print(signals)
    data_s  = make_portfolio(data.copy(), signals, symbols)

    backtest    = Backtest(data_s, symbol=symbols[0], market=market, interval=interval, fees=0.007/100, leverage=100, take_profit=1., stop_loss=0.5)
    data_s      = backtest.vectorized_backtest_multi_asset_trading(symbols)

    print(backtest.backtest_metrics(data_s))

    return data_s["Wealth"]

if __name__ == "__main__":
    import pylab    as pl

    import matplotlib.gridspec  as gridspec

    from Backtest.backtest      import Backtest
    from DataPipeline.get_data  import get_multi_asset_price

    market      = "forex"
    interval    = "1m"
    start       = 1563535876
    end         = 1713535876

    path    = "Data/Backtest/PairsMeanReversion/"
    symbols = ["AUD_USD", "EUR_USD", "GBP_USD", "USD_CAD", "USD_HKD", "USD_JPY"]

    metric  = "Total"

    signals = pd.read_csv(f"{path}signal.csv").fillna(0)
    config  = pd.read_csv(f"{path}config.csv")
    signals["entry"]    = pd.to_datetime(signals["entry"])
    signals["exit"]     = pd.to_datetime(signals["exit"])
    signals["Total"]    += 1.
    data    = get_multi_asset_price(symbols=symbols, numerair="USD", market=market, interval=interval, date1=start, date2=end)
    symbols = data.columns
    
    meas    = signals[[metric, "config"]].groupby("config").prod().sort_values(metric)
    print(meas)

    T   = (data.index[-1] - data.index[0]).total_seconds()/60

    meas    = meas.loc[config[["config", "cols"]].groupby("cols").apply(lambda x: meas.loc[x["config"]].sort_values(metric).index[-1])]
    print(meas)

    #config      = meas.index[-1:]  #[57]  # meas.index    #[::-1][:20:5] [12]
    
    #config  = meas.sort_values(metric).index    #[-4:]
    config  = [3916, 3679, 2636, 3861, 740, 3246, 629, 3976, 4073, 3091, 3567, 820, 3629, 12, 750]
    s   = signals[np.isin(signals["config"], config)]  # signals["config"] == i]
    Wealth  = apply_signal(data, s)
    
    win_rate    = len(s[s["Total"] > 1])/len(s)*100
    win_mean    = (s[s["Total"] > 1]["Total"] -1.).mean()*100
    loss_mean   = (s[s["Total"] < 1]["Total"] -1.).mean()*100
    max_loss    = s["Max Drawdown"].min()*100
    avg_cagr    = (float(Wealth.iloc[-1])**(365*24*60./T) - 1)*100

    S   = (s["exit"] - s["entry"]).sum().total_seconds()/60
    avg_time_in = S/T*100
    norm_cagr   = (float(Wealth.iloc[-1])**(365*24*60./S) - 1)*100

    print("Start Value:\t\t", 1)
    print("End Value:\t\t", Wealth.iloc[-1])
    print("Win Rate [%]:\t\t", win_rate)
    print("Win Mean [%]:\t\t", win_mean)
    print("Loss Mean [%]:\t\t", loss_mean)
    print("Max Loss [%]:\t\t", max_loss)
    print("Avg CAGR [%]:\t\t", avg_cagr)
    print("Norm CAGR [%]:\t\t", norm_cagr)
    print("Time in Market [%]:\t\t", avg_time_in)

    data[symbols]   /= data[symbols].iloc[0]

    fig = pl.figure()
    gs  = gridspec.GridSpec(1, 1)
    ax  = fig.add_subplot(gs[0])
    
    data[symbols].plot(ax=ax, linewidth=0.3)
    Wealth.plot(ax=ax, linewidth=3, color="r")

    ax.set_yscale("log")
    ax.grid()
    ax.legend()

    pl.show()




