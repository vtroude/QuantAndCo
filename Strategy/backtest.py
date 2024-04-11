import os
#import sys

import numpy    as np
import pandas   as pd
import pylab    as pl

from DataPipeline.get_data                  import get_data_and_bars
from Model.Classifier.randomforest_barrier  import get_model

#sys.path.append("/home/virgile/Desktop/General/Packages/")
#from Indicators.best_gains_worst_losses import best_gains_worst_losses

def back_test_thresh(price, n_points, thresh):
    i0              = 0
    price["Wealth"] = 1.
    n_trade         = 0
    while i0 < len(price) - n_points:
        position    = -1.*np.sign(price["signal"].iloc[i0]) if np.abs(price["signal"].iloc[i0]) > thresh else 0
        if position != 0:
            n_trade += 1
            p       = price.iloc[i0:i0+n_points]
            p_pos   = p[p["Close"] >= p["take"]]
            p_neg   = p[p["Close"] <= p["stop"]]
            t_end   = p.index[-1]
            
            if len(p_pos) > 0 and len(p_neg) > 0:
                t_end   = np.minimum(p_pos.index[0], p_neg.index[0])
            elif len(p_pos) > 0:
                t_end   = p_pos.index[0]
            elif len(p_neg) > 0:
                t_end   = p_neg.index[0]
            
            price.loc[p.index[0]:t_end, "Wealth"]   = (1. + position*(p["Close"][:t_end] - p["Close"].iloc[0])/p["Close"].iloc[0])*price.loc[p.index[0], "Wealth"]
            price.loc[t_end:, "Wealth"]             = price["Wealth"][t_end]

            i0  += len(p.loc[:t_end])
        else:
            i0  += 1
    
    return price["Wealth"]

def backtest(symbol, date_test, thres, n_points):
    hitting     = get_model("hitting", symbol, date_test, thres, n_points)
    direction   = get_model("direction", symbol, date_test, thres, n_points)

    data, price = get_data_and_bars(symbol, interval, date1, date2, thres=thres, n_points=n_points)
    data, price = data[data.index > date_test], price[price.index > date_test]

    price["signal"] = hitting.predict_proba(data.to_numpy())[:,-1]
    price["signal"] *= 2.*(direction.predict_proba(data.to_numpy())[:,-1] - 1.)

    strategy_thresh = np.linspace(0, 1, 12)[1:-1]
    wealth  = pd.concat([back_test_thresh(price, n_points, thresh=th) for th in strategy_thresh], axis=1)
    wealth.columns  = [f"$\epsilon={np.round(th,2)}$" for th in strategy_thresh]

    return price, wealth

def save_plot_strategy(price, wealth, model_name, labels):
    fig, ax = pl.subplots(1, 1, figsize=(24,14))
    ax.set_title(f"Backtest {labels['symbol']} from {labels['date start']} to {labels['date end']} ($k={labels['thresh']}$ $n={labels['n points']}$)", fontsize=20)
    wealth.plot(ax=ax, grid=True, logy=True)
    (price["Close"]/price["Close"].iloc[0]).plot(ax=ax, color="k", grid=True, logy=True)

    ax.set_ylabel("Price", fontsize=18)
    ax.set_xlabel("Date", fontsize=18)
    ax.legend(fontsize=26)

    fig.subplots_adjust(left=0.055, bottom=0.093, right=0.962, top=0.92)
    fig.savefig(f"Figure/Backtest/{model_name}.png")

def get_backtest_stats(price: pd.Series, model_name: str, to_append: pd.Series):
    price_last      = price.shift(1)
    index           = price.index[price != price_last]
    price_return    = price/price_last - 1.

    traded_return   = price_return.loc[index]

    success = pd.Series()
    success["mean"]     = traded_return.mean()
    success["std"]      = traded_return.std()
    success["skew"]     = traded_return.skew()
    success["kurt"]     = traded_return.kurtosis()
    success["VaR-5%"]   = traded_return.quantile(0.05)

    #bgwl    = best_gains_worst_losses(price)
    #success["Best Gain"]    = bgwl["cum log"].max()
    #success["Worst Loss"]   = bgwl["cum log"].min()

    success["sharp"]    = success["mean"]/success["std"]
    success["trades"]   = np.round(len(index)/len(price)*100, 2)

    success["model"]    = model_name
    
    success[to_append.index]    = to_append

    file_success  = f"Score/backtest.csv"
    if not os.path.exists(file_success):
        success.to_csv(file_success, index=False)
    else:
        success.to_csv(file_success, index=False, header=False, mode="a")

if __name__ == "__main__":
    date1       = "2021-04-05-23:44:12"
    date2       = "2024-04-04-23:44:12"
    date_test   = pd.to_datetime("2024-01-01 00:00:00")

    symbol      = 'BTCUSDT'
    interval    = ['1m', '1h', '1d', '1w']

    thres       = [0.5, 0.8, 1., 1.2, 1.5, 1.8, 2., 2.5, 3.]
    n_points    = [20, 40, 60, 100, 200, 400, 600]
    I0, I1      = 1, 6

    for th in thres[:1]:
        for n in n_points[:1]:
            try:
                price, wealth   = backtest(symbol, date_test, th, n)
            except:
                wealth = None
            if not wealth is None:
                labels  = {"symbol": symbol, "date start": price.index[0], "date end": price.index[-1], "thresh": th, "n points": n}
                for c in wealth.columns:
                    labels_ = pd.Series(dict(**labels, **{"eps": c.replace("$", "").split("=")[-1]}))
                    get_backtest_stats(wealth[c], "rf classifier", labels_)
                save_plot_strategy(price, wealth)

    labels  = {"symbol": symbol, "date start": price.index[0], "date end": price.index[-1]}
    get_backtest_stats(price["Close"], "B&H", labels)

            
