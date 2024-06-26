import pandas   as pd

from typing     import List, Optional
from itertools  import combinations

from Strategy.signal    import Signal
from Backtest.backtest  import BacktestSignal
import os

path_to_backtest    = f"/home/virgile/Desktop/Trading/QuantDotCom/QuantAndCo/Data/Backtest/"

def finetune_signal(signal: Signal, data: pd.DataFrame, para: List[dict], n_jobs: Optional[int]=1):
    signal_name = signal.__str__()
    path_to_signal  = f"{path_to_backtest}{signal_name}/"
    config_file = f"{path_to_signal}config.csv"
    res_file    = f"{path_to_signal}signal.csv"

    i   = 0
    print(f"Start {len(para)} backtest")
    for p in para:
        print(p)
        signal_res          = BacktestSignal(signal).backtest(data, n_jobs=n_jobs, **p).dropna()
        if len(signal_res) > 0:
            config_n    = 1
            if not os.path.exists(path_to_signal):
                os.makedirs(os.path.dirname(path_to_signal))
            elif os.path.exists(config_file):
                config_n    = len(pd.read_csv(config_file)) + 1
            
            signal_res["config"]    = config_n
            if not os.path.exists(res_file):
                signal_res.to_csv(res_file, index=False)
            else:
                pd.concat([pd.read_csv(res_file), signal_res], ignore_index=True, axis=0).to_csv(res_file, index=False)

            p_dict  = {k: [str(v)] if isinstance(v, list) else [v] for k, v in p.items()}
            config  = pd.DataFrame(dict(**{"config": config_n, "signal": signal_name}, **p_dict))
            if not os.path.exists(config_file):
                config.to_csv(config_file)
            else:
                pd.concat([pd.read_csv(config_file), config], ignore_index=True, axis=0).to_csv(config_file, index=False)

        i   += 1
        print(f"Finished {i} out of {len(para)} with # of signal = {len(signal_res)}")

if __name__ == "__main__":
    from DataPipeline.get_data  import get_price_data
    from Strategy.MeanReversion.mean_reversing  import MeanReversion, PairsMeanReversion, MultiMeanReversion

    n_jobs  = 5

    market      = "forex"
    interval    = "1m"
    start       = 1563535876
    end         = 1713535876

    symbols         = ["AUD_USD", "EUR_USD", "GBP_USD", "USD_CAD", "USD_HKD", "USD_JPY"]
    data            = pd.concat([get_price_data(market=market, symbol=s, interval=interval, date1=start, date2=end)["Close"] for s in symbols], axis=1).dropna()
    data.columns    = symbols
    for s in ["USD_CAD", "USD_HKD", "USD_JPY"]:
        s0, s1  = s.split("_")
        data[f"{s1}_{s0}"]  = 1./data[s]
        symbols             = [s_ if s_ != s else f"{s1}_{s0}" for s_ in symbols]
    
    data    = data[symbols]
    pairs   = [list(pair) for pair in combinations(symbols, 2)]

    n_min_per_days  = 60 * 24
    window_length   = [10, 60, 300]
    frac            = [100, 500]
    step            = [[1, 2], [1, 5, 10, 30], [1, 5, 10, 30, 100]]
    look_a_head     = [[1, 2], [1, 5, 10, 30], [1, 5, 10, 30, 100]]
    use_log         = [True, False]

    entry_thres = [1., 2., 3., 4.]
    exit_thresh = [[0.]]
    for i in range(len(entry_thres)):
        exit_thresh += [exit_thresh[-1] + [entry_thres[i]]]

    para_list   = []
    for w, s, l in zip(window_length, step, look_a_head):
        for s_ in s:
            for l_ in l:
                for f in frac:
                    for u in use_log:
                        for entry, exit in zip(entry_thres, exit_thresh):
                            for ex_ in exit:
                                for c in pairs:
                                    para_list.append(
                                                        {
                                                            "cols": c,
                                                            "window_length": w*n_min_per_days,
                                                            "step": s_*n_min_per_days,
                                                            "look_a_head": l_*n_min_per_days,
                                                            "frac": f,
                                                            "use_log": u,
                                                            "exit_threshold": ex_,
                                                            "signal_threshold": entry
                                                        }
                                                    )
                                    
    finetune_signal(PairsMeanReversion(), data, para_list, n_jobs=n_jobs)