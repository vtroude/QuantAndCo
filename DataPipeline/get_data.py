import numpy    as np
import pandas   as pd

from DataPipeline.make_data import make_filename

def get_data(symbol, interval, date1, date2):
    data        = pd.read_csv(make_filename(symbol, interval, date1, date2, "ti"), index_col=0)
    data.index  = pd.to_datetime(data.index, format="ISO8601")

    data.columns    = [f"{d}-{interval}" for d in data.columns]

    return data.sort_index().dropna(axis=1, how="all")

def get_all_data(symbol, interval, date1, date2):
    data    = get_data(symbol, interval[0], date1, date2)
    for i in interval[1:]:
        data_   = get_data(symbol, i, date1, date2)
        data    = pd.merge_asof(data, data_, left_index=True, right_index=True, direction='backward')

    return data.dropna(axis=1, how="all")

def get_bars(price, data, interval, n_points=60, thres=2):
    mean    = n_points*(data[f"mean-20-{interval}"] - 0.5*data[f"std-20-{interval}"]*data[f"std-20-{interval}"] )
    std_p   = thres*np.sqrt(n_points)*thres*data[f"std(+)-20-{interval}"]
    std_n   = thres*np.sqrt(n_points)*thres*data[f"std(-)-20-{interval}"]

    r_pos   = np.maximum(mean + std_p, std_p)
    r_neg   = np.minimum(mean - std_n, -std_n)

    bars            = pd.DataFrame(columns=["take", "stop"])
    bars["take"]    = price["Close"]*np.exp(r_pos)
    bars["stop"]    = price["Close"]*np.exp(r_neg)

    return pd.concat([price, bars], axis=1)

def get_hitting(x, take, stop):
    up      = x[x >= take]
    down    = x[x <= stop]
    if len(up) > 0 and len(down) > 0:
        if up.index[0] < down.index[0]:
            return 1
        
        return -1
    elif len(up) > 0:
        return 1
    elif len(down) > 0:
        return -1
    
    return 0

def get_targets(price, n_points=60):
    targets = price["Close"].rolling(window=n_points).apply(lambda x: get_hitting(x, price["take"][x.index[0]], price["stop"][x.index[0]]))

    targets = targets.shift(periods=-n_points)
    hits    = np.abs(targets)
    bar     = (targets[targets != 0]+1)//2

    return hits.dropna(), bar.dropna()

def get_data_and_bars(symbol, interval, date1, date2, thres=2, n_points=60):
    data    = get_all_data(symbol, interval, date1, date2).dropna()
    price   = pd.read_csv(make_filename(symbol, interval[0], date1, date2, "ohlc"), index_col=0)
    
    price.index = pd.to_datetime(price.index, format="ISO8601")   # "%Y-%m-%d %H:%M:%S")
    price       = price.rename(columns={o.lower(): o for o in ['Open', 'High', 'Low', 'Close', 'Volume']})
    price       = price.loc[data.index]
    price       = get_bars(price, data, interval[0], thres=thres, n_points=n_points)

    return data, price

def get_ml_bars_data(symbol, interval, date1, date2, thres=2, n_points=60):
    data, price = get_data_and_bars(symbol, interval, date1, date2, thres=thres, n_points=n_points)
    hits, bar   = get_targets(price)

    return data, hits, bar

if __name__ == "__main__":
    date1       = "2021-04-05-23:44:12"
    date2       = "2024-04-04-23:44:12"

    symbol      = 'BTCUSDT'
    interval    = ['1m', '1h', '1d', '1w']

    data, hits, bar = get_ml_bars_data(symbol, interval, date1, date2)

    print(data)
    print(hits)
    print(bar)