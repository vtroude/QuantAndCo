import pandas   as pd

from Strategy.strategy  import StrategyAbstract
from Backtest.backtest  import Backtest

def __init__(price: pd.DataFrame, strategy_type: StrategyAbstract, fix_para: dict, scan_para: dict, backtest_para: dict) -> None:
    for para in scan_para:
        strategy        = strategy_type(**dict(**fix_para, **para))
        price["signal"] = strategy.generate_signals()
        backtesting     = Backtest(strategy_df=price, **backtest_para)

